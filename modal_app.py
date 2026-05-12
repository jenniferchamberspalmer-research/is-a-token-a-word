"""Modal deployment for the Water Pattern Tool.

ONE-TIME SETUP
==============

1. Install Modal locally:
       pip install modal

2. Authenticate (opens a browser):
       modal token new

3. Get a Hugging Face access token:
       - Make a free account at huggingface.co
       - Visit huggingface.co/google/gemma-2-2b and accept the license
       - Visit huggingface.co/google/gemma-scope-2b-pt-res and accept it too
       - At huggingface.co/settings/tokens, create a "Read" token

4. Give that token to Modal as a secret named "huggingface":
       modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

RUNNING
=======

Development (auto-reloads when you edit code, prints logs):
    modal serve modal_app.py

Stable deployment (gets a permanent URL, runs in the background):
    modal deploy modal_app.py

Either command prints a URL. Open it in a browser. The first request
will take ~30–60 seconds while the container loads the model and SAEs;
subsequent requests are fast. The container automatically shuts down
after a few minutes of inactivity, so you only pay while actively
using it.

COST
====

GPU: T4 (cheapest reliable option, ~$0.59/hr on Modal as of writing).
Gemma 2 2B in bfloat16 fits in ~6 GB of VRAM, well within the T4's 16 GB.
A typical research session of an hour or two is well under $2.

To bump up for faster forward passes, change `gpu="T4"` to `gpu="L4"`
(~$0.80/hr) or `gpu="A10G"` (~$1.10/hr) below.
"""

import modal

# The container image: a small Debian + Python 3.11 base with our
# dependencies pip-installed, plus our `water_tool` package mounted in.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.42,<5",
        "accelerate>=0.30",
        "sae-lens>=3.20",
        "gradio==4.44.1",
        "gradio-client==1.3.0",
        "pydantic<2.10",
        "pandas>=2.0",
        "requests>=2.31",
        "fastapi==0.112.2",
        "starlette==0.38.6",
    )
    .add_local_python_source("water_tool")
)

app = modal.App("water-pattern-tool", image=image)

# A persistent volume caches Hugging Face downloads (model + SAE weights)
# and Neuronpedia feature labels between container starts. Without this
# the model would re-download every cold start.
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)


@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
    scaledown_window=300,  # idle 5 min before shutting down
    max_containers=1,      # all requests hit the same container so Gradio
                           # sessions and SSE streams stay coherent
)
@modal.concurrent(max_inputs=100)  # one container, many concurrent requests
@modal.asgi_app()
def serve():
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["NEURONPEDIA_CACHE"] = "/cache/neuronpedia"

    import gradio as gr
    from fastapi import FastAPI

    from water_tool.app import build

    demo = build()
    fastapi_app = FastAPI()
    return gr.mount_gradio_app(fastapi_app, demo, path="/")
