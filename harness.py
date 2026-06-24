"""Programmatic driver for the Water Pattern Tool.

Calls the /probe/* REST endpoints exposed by modal_app.py and writes
each probe result as JSON (full metadata) plus CSV (Excel-friendly
review form) under `data/<BATCH>/<word>/`.

This bypasses gradio_client entirely. The Gradio app the user
interacts with is still mounted at the same Modal URL; only the
automation path uses /probe.

Run from the repo root on any machine that can reach Modal:

    pip install httpx
    python harness.py

Whole batch takes ~5-10 min wall-clock and ~$0.10-$0.50 of Modal
credit on a T4. The first probe per process is slow (Modal cold
start + model load); subsequent probes are fast.
"""

import csv
import json
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Missing dependency. Run:  pip install httpx", file=sys.stderr)
    sys.exit(1)


URL = "https://jenniferchamberspalmer-research--water-pattern-tool-serve.modal.run"
BATCH = "2026-06-15-sacramental-substances"
TOP_K = 20
V3_K = 15
V3_LAYERS = [6, 12, 19]
REQUEST_TIMEOUT_S = 300  # ample headroom for cold-start model load
HEALTH_TIMEOUT_S = 60


PROBES = {
    "water": {
        "view1_text": "water",
        "view2_prompts": ["The water is", "The holy water is", "The polluted water is"],
        "view3_sentence": "The priest blessed the water before the baptism.",
        "view3_target": "water",
    },
    "salt": {
        "view1_text": "salt",
        "view2_prompts": ["The salt is", "The holy salt is", "The tainted salt is"],
        "view3_sentence": "The patron tossed the salt over his shoulder.",
        "view3_target": "salt",
    },
    "bread": {
        "view1_text": "bread",
        "view2_prompts": ["The bread is", "The holy bread is", "The molded bread is"],
        "view3_sentence": "The priest broke the bread for communion.",
        "view3_target": "bread",
    },
}


def warm_up(client: httpx.Client, max_wait_s: int = 180) -> None:
    """Hit /probe/health until it responds 200. This wakes the Modal
    container before the first real probe."""
    print(f"Warming up {URL} ... (up to {max_wait_s}s for cold start)")
    deadline = time.time() + max_wait_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = client.get("/probe/health", timeout=HEALTH_TIMEOUT_S)
            if r.status_code == 200:
                print(f"  endpoint warm after {attempt} attempt(s)")
                return
            print(f"  attempt {attempt}: status {r.status_code}, retrying ...")
        except httpx.HTTPError as e:
            print(f"  attempt {attempt}: {type(e).__name__}, retrying ...")
        time.sleep(5)
    raise RuntimeError(f"Endpoint never became reachable within {max_wait_s}s")


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  saved -> {path}")


def _save_csv(path: Path, rows: list) -> None:
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


def call(client: httpx.Client, path: str, body: dict) -> dict:
    r = client.post(path, json=body, timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def run():
    with httpx.Client(base_url=URL) as client:
        warm_up(client)

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
            res = call(client, "/probe/view1", {
                "text": cfg["view1_text"],
                "mode": "raw_lookup",
                "k": TOP_K,
            })
            rows = res["results"]
            _save_json(wdir / "view1_raw.json", {
                "input": {"text": cfg["view1_text"], "mode": "raw_lookup", "k": TOP_K},
                "elapsed_s": round(time.time() - t0, 2),
                "results": rows,
            })
            _save_csv(wdir / "view1_raw.csv", rows)

            # ---- View 1, contextualized hidden state ----
            print("  view 1 (contextual) ...")
            t0 = time.time()
            res = call(client, "/probe/view1", {
                "text": cfg["view1_text"],
                "mode": "contextual",
                "k": TOP_K,
            })
            rows = res["results"]
            _save_json(wdir / "view1_contextual.json", {
                "input": {"text": cfg["view1_text"], "mode": "contextual", "k": TOP_K},
                "elapsed_s": round(time.time() - t0, 2),
                "results": rows,
            })
            _save_csv(wdir / "view1_contextual.csv", rows)

            # ---- View 2, three prompts ----
            print("  view 2 (three prompts) ...")
            t0 = time.time()
            res = call(client, "/probe/view2", {
                "prompts": cfg["view2_prompts"],
                "k": TOP_K,
            })
            _save_json(wdir / "view2.json", {
                "input": {"prompts": cfg["view2_prompts"], "k": TOP_K},
                "elapsed_s": round(time.time() - t0, 2),
                "results": res["results"],
            })
            # Long-form CSV: one row per (prompt, rank)
            long_rows = []
            for r in res["results"]:
                prompt = r["prompt"]
                for row in r["rows"]:
                    long_rows.append({"prompt": prompt, **row})
            _save_csv(wdir / "view2.csv", long_rows)

            # ---- View 3, three layers ----
            for layer in V3_LAYERS:
                print(f"  view 3 (layer {layer}) ...")
                t0 = time.time()
                res = call(client, "/probe/view3", {
                    "sentence": cfg["view3_sentence"],
                    "target": cfg["view3_target"],
                    "layer": layer,
                    "k": V3_K,
                })
                rows = res["results"]
                _save_json(wdir / f"view3_layer{layer}.json", {
                    "input": {
                        "sentence": cfg["view3_sentence"],
                        "target": cfg["view3_target"],
                        "layer": layer,
                        "k": V3_K,
                    },
                    "elapsed_s": round(time.time() - t0, 2),
                    "results": rows,
                })
                _save_csv(wdir / f"view3_layer{layer}.csv", rows)

        print(f"\nAll done in {round(time.time() - total_start, 1)} s.")
        print(f"Output tree: {out_root.resolve()}")


if __name__ == "__main__":
    run()
