"""Modal deployment for the Water Pattern Tool.

ONE-TIME SETUP
==============

1. pip install modal
2. modal token new
3. Accept the Gemma 2 and Gemma Scope licenses on huggingface.co
4. modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxxxxx

RUNNING
=======

    modal deploy modal_app.py

LIFECYCLE
=========

This file uses a Modal Class with @modal.enter(): the model and the
canonical Gemma Scope SAEs for the View 3 default layers (6, 12, 19)
load ONCE per container at startup, not per request. Requests reuse
the warm objects. `scaledown_window=1800` keeps containers warm for
30 minutes after the last request, so a research session of clicking,
thinking, clicking does not trigger a cold reload between every click.

`max_containers=1` plus @modal.concurrent(max_inputs=100) keeps every
request on the same warm container; the Gradio queue, enabled in
water_tool/app.py, lets long predicts stream progress back to the
browser rather than holding open a single HTTP request to its
timeout.

Replaces the previous @app.function + @modal.asgi_app() pattern, which
allowed the model to be loaded inside the request path on first click
of a fresh container — that path now does not exist.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.46,<5",   # dtype= replaces deprecated torch_dtype=
        "accelerate>=0.30",
        "sae-lens>=3.20",
        "gradio==4.44.1",
        "gradio-client==1.3.0",
        "pydantic<2.10",
        "pandas>=2.0",
        "numpy>=1.26",
        "scipy>=1.11",
        "matplotlib>=3.7",   # View 5 trajectory plot
        "requests>=2.31",
        "fastapi==0.112.2",
        "starlette==0.38.6",
    )
    .add_local_python_source("water_tool")
)

app = modal.App("water-pattern-tool", image=image)
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)


@app.cls(
    gpu="T4",
    timeout=3600,                # per-request execution ceiling
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
    scaledown_window=1800,       # 30 min warm — survives between session clicks
    max_containers=1,            # all requests land on one warm container
)
@modal.concurrent(max_inputs=100)
class WaterPatternServer:
    """Warm server. Model + SAEs load once per container at startup."""

    @modal.enter()
    def warmup(self):
        """Runs once when the container boots. Populates the module-level
        caches in water_tool.core.model and water_tool.core.sae so the
        per-request view handlers never hit a from_pretrained code path.
        """
        import os
        os.environ["HF_HOME"] = "/cache/hf"
        os.environ["NEURONPEDIA_CACHE"] = "/cache/neuronpedia"

        print(">>> warmup: loading Gemma 2 2B base (one-time per container) ...")
        from water_tool.core.model import load
        load()
        print(">>> warmup: model loaded.")

        from water_tool.core.sae import get_sae
        for layer in (6, 12, 19):
            print(f">>> warmup: loading Gemma Scope SAE layer {layer} ...")
            get_sae(layer)
        print(">>> warmup: ready. Subsequent requests reuse these objects.")

    @modal.asgi_app()
    def web(self):
        """Build the FastAPI app with the Gradio UI mounted at / and
        the /probe REST endpoints registered for programmatic access.
        Called once per container after warmup completes.
        """
        import gradio as gr
        from fastapi import FastAPI

        from water_tool.app import build
        from water_tool.views import (
            embedding, probability, features as features_view,
        )

        fastapi_app = FastAPI()

        # ---- /probe REST endpoints (programmatic; bypasses Gradio) ----
        @fastapi_app.post("/probe/view1")
        def probe_view1(req: dict):
            text = req["text"]
            mode = req.get("mode", "raw_lookup")
            k = int(req.get("k", 20))
            if mode.startswith("raw"):
                df = embedding.raw_lookup(text, k=k)
            else:
                df = embedding.contextual(text, k=k)
            return {"results": df.to_dict("records")}

        @fastapi_app.post("/probe/view2")
        def probe_view2(req: dict):
            prompts = req["prompts"]
            k = int(req.get("k", 20))
            results = []
            for p in prompts:
                df = probability.top_next_tokens(p, k=k)
                results.append({"prompt": p, "rows": df.to_dict("records")})
            return {"results": results}

        @fastapi_app.post("/probe/view3")
        def probe_view3(req: dict):
            sentence = req["sentence"]
            target = req["target"]
            layer = int(req["layer"])
            k = int(req.get("k", 15))
            df = features_view.top_features(sentence, target, layer=layer, k=k)
            return {"results": df.to_dict("records")}

        @fastapi_app.get("/probe/health")
        def probe_health():
            return {"ok": True}

        # ---- Gradio UI mounted at root ----
        demo = build()
        return gr.mount_gradio_app(fastapi_app, demo, path="/")
