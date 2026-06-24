"""Programmatic driver for the Water Pattern Tool.

Calls each of the three Gradio views via gradio_client against the
deployed Modal endpoint, collects all outputs as JSON (full metadata)
*and* CSV (Excel-friendly review form) in `data/<BATCH>/<word>/`.

Run from the repo root on any machine that can reach Modal:

    pip install gradio_client pandas
    python harness.py

The whole batch takes ~5-10 min wall-clock and ~$0.10-$0.50 of Modal
credit on a T4. The first call per process is slow (Modal cold start
+ model load); subsequent calls are fast.
"""

import csv
import json
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Missing dependency. Run:  pip install gradio_client pandas", file=sys.stderr)
    sys.exit(1)

# Monkey-patch httpx default timeout BEFORE importing gradio_client.
# gradio_client 1.3.0 calls httpx.post(...) / httpx.get(...) directly
# without exposing a timeout parameter, and its module-level default is
# ~5s — far too short for our server's first predict call. Cold-start
# model loading on Modal takes 30-60s, so the first send_data POST
# always times out. Patching the module attributes here means every
# internal call inherits a 180s budget instead.
_orig_post = httpx.post
_orig_get = httpx.get


def _patched_post(*args, **kwargs):
    kwargs.setdefault("timeout", 180)
    return _orig_post(*args, **kwargs)


def _patched_get(*args, **kwargs):
    kwargs.setdefault("timeout", 180)
    return _orig_get(*args, **kwargs)


httpx.post = _patched_post
httpx.get = _patched_get

try:
    from gradio_client import Client
except ImportError:
    print("Missing dependency. Run:  pip install gradio_client pandas", file=sys.stderr)
    sys.exit(1)


def warm_up(url: str, max_wait_s: int = 180) -> None:
    """Trigger Modal cold-start and wait for the Gradio /config endpoint
    to respond with a 200. gradio_client's default httpx timeout is too
    short for a cold container that has to load Gemma 2 2B + SAEs
    before answering its first request.
    """
    config_url = url.rstrip("/") + "/config"
    print(f"Warming up {url} ... (up to {max_wait_s}s for cold start)")
    deadline = time.time() + max_wait_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = httpx.get(config_url, timeout=60.0)
            if r.status_code == 200:
                print(f"  endpoint warm after {attempt} attempt(s)")
                return
            print(f"  attempt {attempt}: status {r.status_code}, retrying ...")
        except (httpx.ReadTimeout, httpx.ConnectTimeout,
                httpx.RemoteProtocolError, httpx.ConnectError) as e:
            print(f"  attempt {attempt}: {type(e).__name__}, retrying ...")
        time.sleep(5)
    raise RuntimeError(f"Endpoint never became reachable within {max_wait_s}s")


URL = "https://jenniferchamberspalmer-research--water-pattern-tool-serve.modal.run"
BATCH = "2026-06-15-sacramental-substances"
TOP_K = 20
V3_K = 15
V3_LAYERS = ["6", "12", "19"]


PROBES = {
    "water": {
        "view1_text": "water",
        "view2_prompts": ("The water is", "The holy water is", "The polluted water is"),
        "view3_sentence": "The priest blessed the water before the baptism.",
        "view3_target": "water",
    },
    "salt": {
        "view1_text": "salt",
        "view2_prompts": ("The salt is", "The holy salt is", "The tainted salt is"),
        "view3_sentence": "The patron tossed the salt over his shoulder.",
        "view3_target": "salt",
    },
    "bread": {
        "view1_text": "bread",
        "view2_prompts": ("The bread is", "The holy bread is", "The molded bread is"),
        "view3_sentence": "The priest broke the bread for communion.",
        "view3_target": "bread",
    },
}


def _df_to_records(df_obj):
    """Normalize Gradio Dataframe outputs into a list of dicts."""
    if hasattr(df_obj, "to_dict"):
        return df_obj.to_dict(orient="records")
    if isinstance(df_obj, dict) and "headers" in df_obj and "data" in df_obj:
        headers = df_obj["headers"]
        return [dict(zip(headers, row)) for row in df_obj["data"]]
    if isinstance(df_obj, list):
        return df_obj
    return df_obj


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  saved -> {path}")


def _save_csv(path: Path, rows: list) -> None:
    """Write a list of dicts as CSV. If rows is empty, write an empty file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        print(f"  saved -> {path}  (empty)")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved -> {path}")


def run():
    warm_up(URL)
    print(f"Connecting to {URL} ...")
    client = Client(URL, verbose=False)
    print("Connected.")

    out_root = Path("data") / BATCH
    out_root.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    for word, cfg in PROBES.items():
        print(f"\n=== {word} ===")
        wdir = out_root / word
        wdir.mkdir(exist_ok=True)

        # ---- View 1, raw embedding lookup ----
        print("  view 1 (raw_lookup) ...")
        t0 = time.time()
        df, _csv, _json, note = client.predict(
            cfg["view1_text"],
            "raw_lookup (embedding table)",
            TOP_K,
            fn_index=0,
        )
        records = _df_to_records(df)
        _save_json(wdir / "view1_raw.json", {
            "input": {"text": cfg["view1_text"], "mode": "raw_lookup", "k": TOP_K},
            "elapsed_s": round(time.time() - t0, 2),
            "note": note,
            "results": records,
        })
        _save_csv(wdir / "view1_raw.csv", records)

        # ---- View 1, contextualized hidden state ----
        print("  view 1 (contextual) ...")
        t0 = time.time()
        df, _csv, _json, note = client.predict(
            cfg["view1_text"],
            "contextual (final hidden state)",
            TOP_K,
            fn_index=0,
        )
        records = _df_to_records(df)
        _save_json(wdir / "view1_contextual.json", {
            "input": {"text": cfg["view1_text"], "mode": "contextual", "k": TOP_K},
            "elapsed_s": round(time.time() - t0, 2),
            "note": note,
            "results": records,
        })
        _save_csv(wdir / "view1_contextual.csv", records)

        # ---- View 2, three prompts in a single call ----
        print("  view 2 (three prompts) ...")
        p1, p2, p3 = cfg["view2_prompts"]
        t0 = time.time()
        out = client.predict(p1, p2, p3, TOP_K, fn_index=1)
        df1, df2, df3 = out[0], out[1], out[2]
        per_prompt = [
            (p1, _df_to_records(df1)),
            (p2, _df_to_records(df2)),
            (p3, _df_to_records(df3)),
        ]
        _save_json(wdir / "view2.json", {
            "input": {"prompts": list(cfg["view2_prompts"]), "k": TOP_K},
            "elapsed_s": round(time.time() - t0, 2),
            "results": [{"prompt": p, "top_k": r} for p, r in per_prompt],
        })
        # Long-form CSV: one row per (prompt, rank) — easy to filter in Excel
        long_rows = []
        for p, rows in per_prompt:
            for r in rows:
                long_rows.append({"prompt": p, **r})
        _save_csv(wdir / "view2.csv", long_rows)

        # ---- View 3, three layers ----
        for layer in V3_LAYERS:
            print(f"  view 3 (layer {layer}) ...")
            t0 = time.time()
            df, _csv, _json = client.predict(
                cfg["view3_sentence"],
                cfg["view3_target"],
                layer,
                V3_K,
                fn_index=2,
            )
            records = _df_to_records(df)
            _save_json(wdir / f"view3_layer{layer}.json", {
                "input": {
                    "sentence": cfg["view3_sentence"],
                    "target": cfg["view3_target"],
                    "layer": int(layer),
                    "k": V3_K,
                },
                "elapsed_s": round(time.time() - t0, 2),
                "results": records,
            })
            _save_csv(wdir / f"view3_layer{layer}.csv", records)

    print(f"\nAll done in {round(time.time() - total_start, 1)} s.")
    print(f"Output tree: {out_root.resolve()}")


if __name__ == "__main__":
    run()
