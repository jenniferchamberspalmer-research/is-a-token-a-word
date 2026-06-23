"""Programmatic driver for the Water Pattern Tool.

Calls each of the three Gradio views via gradio_client against the
deployed Modal endpoint, collects all outputs as JSON in
`data/<BATCH>/<word>/`, ready to be committed to the repository.

Run this from the repo root on a machine that can reach Modal:

    pip install gradio_client
    python harness.py

The model and SAEs load lazily on the first call per process on
Modal, so the first probe of each run takes ~30-60 seconds. Subsequent
probes are fast. Whole batch takes ~5-10 minutes wall-clock and burns
roughly $0.10-$0.50 of Modal credit on a T4.
"""

import json
import sys
import time
from pathlib import Path

try:
    from gradio_client import Client
except ImportError:
    print("Missing dependency. Run:  pip install gradio_client", file=sys.stderr)
    sys.exit(1)


URL = "https://jenniferchamberspalmer-research--water-pattern-tool-serve.modal.run"
BATCH = "2026-06-15-sacramental-substances"
TOP_K = 20      # views 1 and 2
V3_K = 15       # view 3 (matches the spec we've been using)
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
    """Normalize Gradio Dataframe outputs into a list of dicts.

    gradio_client may return a dataframe as:
      - a dict with 'headers' and 'data' keys (most common),
      - a pandas.DataFrame (if pandas is installed),
      - already a list of records.
    """
    # pandas DataFrame
    if hasattr(df_obj, "to_dict"):
        return df_obj.to_dict(orient="records")
    # Gradio dict form
    if isinstance(df_obj, dict) and "headers" in df_obj and "data" in df_obj:
        headers = df_obj["headers"]
        return [dict(zip(headers, row)) for row in df_obj["data"]]
    # already records
    if isinstance(df_obj, list):
        return df_obj
    # fallback
    return df_obj


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  saved -> {path}")


def run():
    print(f"Connecting to {URL} ...")
    client = Client(URL, verbose=False)
    print("Connected. Available endpoints:")
    try:
        print(client.view_api(return_format="str"))
    except Exception as e:
        print(f"  (could not retrieve api summary: {e})")

    out_root = Path("data") / BATCH
    out_root.mkdir(parents=True, exist_ok=True)

    # Copy the research plan into the data folder so a downloader gets it bundled.
    plan_src = out_root / "research_plan.md"
    if plan_src.exists():
        print(f"\nResearch plan already in place at {plan_src}")
    else:
        print(f"\nNote: research_plan.md not found in {out_root}; create it before publishing.")

    total_start = time.time()

    for word, cfg in PROBES.items():
        print(f"\n=== {word} ===")
        wdir = out_root / word
        wdir.mkdir(exist_ok=True)

        # --- View 1, raw embedding lookup ---
        print("  view 1 (raw_lookup) ...")
        t0 = time.time()
        out = client.predict(
            cfg["view1_text"],
            "raw_lookup (embedding table)",
            TOP_K,
            fn_index=0,
        )
        df, csv_path, json_path, note = out
        _save(wdir / "view1_raw.json", {
            "input": {"text": cfg["view1_text"], "mode": "raw_lookup", "k": TOP_K},
            "elapsed_s": round(time.time() - t0, 2),
            "results": _df_to_records(df),
            "note": note,
        })

        # --- View 1, contextualized hidden state ---
        print("  view 1 (contextual) ...")
        t0 = time.time()
        out = client.predict(
            cfg["view1_text"],
            "contextual (final hidden state)",
            TOP_K,
            fn_index=0,
        )
        df, csv_path, json_path, note = out
        _save(wdir / "view1_contextual.json", {
            "input": {"text": cfg["view1_text"], "mode": "contextual", "k": TOP_K},
            "elapsed_s": round(time.time() - t0, 2),
            "results": _df_to_records(df),
            "note": note,
        })

        # --- View 2, three prompts in a single call ---
        print("  view 2 (three prompts) ...")
        p1, p2, p3 = cfg["view2_prompts"]
        t0 = time.time()
        out = client.predict(p1, p2, p3, TOP_K, fn_index=1)
        # view2_run returns: df1, df2, df3, csv1, csv2, csv3, json_path
        df1, df2, df3 = out[0], out[1], out[2]
        _save(wdir / "view2.json", {
            "input": {"prompts": list(cfg["view2_prompts"]), "k": TOP_K},
            "elapsed_s": round(time.time() - t0, 2),
            "results": [
                {"prompt": p1, "top_k": _df_to_records(df1)},
                {"prompt": p2, "top_k": _df_to_records(df2)},
                {"prompt": p3, "top_k": _df_to_records(df3)},
            ],
        })

        # --- View 3, three layers ---
        for layer in V3_LAYERS:
            print(f"  view 3 (layer {layer}) ...")
            t0 = time.time()
            out = client.predict(
                cfg["view3_sentence"],
                cfg["view3_target"],
                layer,
                V3_K,
                fn_index=2,
            )
            df, csv_path, json_path = out
            _save(wdir / f"view3_layer{layer}.json", {
                "input": {
                    "sentence": cfg["view3_sentence"],
                    "target": cfg["view3_target"],
                    "layer": int(layer),
                    "k": V3_K,
                },
                "elapsed_s": round(time.time() - t0, 2),
                "results": _df_to_records(df),
            })

    print(f"\nAll done in {round(time.time() - total_start, 1)} s.")
    print(f"Output tree: {out_root.resolve()}")
    print("\nNext step: from the repo root,")
    print(f"  git add data/{BATCH}")
    print(f'  git commit -m "Add probe data: {BATCH}"')
    print( "  git push")


if __name__ == "__main__":
    run()
