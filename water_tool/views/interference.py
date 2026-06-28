"""View 4: Cross-field interference (β critical exponent).

For a multilingual concept set (a small {language: [word(s)]} mapping),
measures whether the cross-linguistic feature cluster at a chosen SAE
layer occupies a better-renormalized sub-field than a magnitude-matched
random null.

============================================================================
REPRESENTATION USED FOR FEATURE SELECTION  (read this before interpreting)
============================================================================

`select_crosslingual_features` operates on **SAE feature activations at
the chosen residual-stream layer**, NOT on the input embedding table.

Pipeline per language word:
  - tokenize the word,
  - run the model forward with a hook on `model.model.layers[layer]` so
    we capture the residual stream at the OUTPUT of that block,
  - encode the captured residual through `sae.encode()`,
  - pool across sub-token positions per `pooling`:
      * "last": only the final sub-token (default — resolves the
                multi-token vs single-token asymmetry across languages;
                água is multi-token, water is single-token, and mean-of-
                non-BOS would treat them differently).
      * "mean": all non-BOS positions (legacy, available as sensitivity
                check).

A feature is "active" in a language if its pooled activation is in the
top (1 - quantile) for that language. Cross-linguistic features are
those active in `min_languages` languages (default: all).

Implication. The cross-linguistic structure the selector identifies is
whatever structure the SAE at the chosen layer encodes about the bare
word. View 1 raw_lookup mode found that the input embedding table
clusters translation equivalents tightly (water / agua / Wasser); the
final-layer contextualized state shows English collocates instead.
The intermediate layers fall somewhere between, and where exactly the
cross-linguistic geometry survives is itself a per-layer empirical
question. The depth sweep (modal_layer_sweep.py — CLI reference, not
the served tab) is how that question gets mapped layer by layer; THIS
tab probes one layer at a time.

If you select a late layer (e.g. layer 19) and `selected_n` is very
small, that is the model telling you the cross-linguistic structure
has thinned out at that depth — not a tool failure. Try a shallower
layer.

============================================================================
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import gradio as gr
import numpy as np
import pandas as pd
import torch

from ..core.model import load
from ..core.sae import get_sae
from ..interference.experiment import (
    select_crosslingual_features,
    view1_experiment,
)
from ..interference.superposition import SuperpositionField


DEFAULT_CONCEPT_TEXT = (
    "en: water\n"
    "es: agua\n"
    "fr: eau\n"
    "de: Wasser\n"
    "it: acqua\n"
    "pt: água\n"
)


def parse_concept_list(text: str) -> dict:
    """Parse a multi-line concept list of the form

        en: water
        es: agua
        de: Wasser, Wässer

    Returns {lang: [word, ...]}.
    """
    concepts = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        lang, words_str = line.split(":", 1)
        words = [w.strip() for w in words_str.split(",") if w.strip()]
        if words:
            concepts[lang.strip()] = words
    return concepts


def _concept_mean_activations(model, tokenizer, sae, layer: int,
                              concepts: dict, pooling: str = "last") -> dict:
    """Per-language mean SAE feature activation for the concept set.

    Hook the residual stream at the output of `model.model.layers[layer]`,
    encode via sae.encode(), pool across token positions per `pooling`.
    Returns {lang: ndarray(d_sae)}.
    """
    device = next(model.parameters()).device
    d_sae = int(sae.W_dec.shape[0])
    captured = {}

    def hook(_m, _in, out):
        x = out[0] if isinstance(out, tuple) else out
        captured["resid"] = x.detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        acts_by_lang = {}
        with torch.no_grad():
            for lang, words in concepts.items():
                accum = np.zeros(d_sae)
                count = 0
                for word in words:
                    ids = tokenizer(word, return_tensors="pt").to(device)
                    model(**ids)
                    resid = captured["resid"][0].float()  # (seq, d_model)
                    feats = sae.encode(
                        resid.to(sae.W_enc.dtype)
                    ).float().cpu().numpy()  # (seq, d_sae)
                    if pooling == "last":
                        rows = np.array([feats.shape[0] - 1])
                    else:  # "mean"
                        rows = np.arange(1, feats.shape[0])
                    if len(rows):
                        accum += feats[rows].mean(axis=0)
                        count += 1
                acts_by_lang[lang] = accum / max(count, 1)
    finally:
        handle.remove()
    return acts_by_lang


def _field_from_sae(sae, name: str) -> SuperpositionField:
    """Build a SuperpositionField from sae.W_dec.

    sae_lens stores W_dec as (d_sae, d_model), so rows are already one
    feature's writing direction (no transpose needed). Defensive
    orientation check included.
    """
    W = sae.W_dec.detach().float().cpu().numpy()
    if W.ndim != 2:
        raise ValueError(f"unexpected W_dec shape {W.shape}")
    if W.shape[0] < W.shape[1]:
        W = W.T
    return SuperpositionField(W, name=name)


def run_interference(concept_text: str, layer, pooling: str,
                     quantile: float, min_languages,
                     n_boot, n_null,
                     progress=gr.Progress(track_tqdm=False)):
    """Top-level Gradio handler. Returns (summary_df, status_str,
    json_path_for_download)."""
    try:
        progress(0.02, desc="parsing concept list")
        concepts = parse_concept_list(concept_text)
        if len(concepts) < 2:
            raise gr.Error("Need at least 2 languages in the concept list.")

        layer = int(layer)
        if min_languages in (None, "", 0, "all", "All"):
            min_languages_val = len(concepts)
        else:
            min_languages_val = int(min_languages)

        progress(0.05, desc=f"loading model (cached) and SAE layer {layer}")
        model, tokenizer = load()
        sae = get_sae(layer)
        field = _field_from_sae(sae,
                                name=f"gemma-scope-2b-canonical L{layer}")

        progress(0.15, desc="per-language activations")
        acts = _concept_mean_activations(
            model, tokenizer, sae, layer, concepts, pooling=pooling
        )

        progress(0.55, desc="selecting cross-linguistic features")
        xling = select_crosslingual_features(
            acts, min_languages=min_languages_val, quantile=float(quantile)
        )
        if len(xling) < 3:
            raise gr.Error(
                f"Selected only {len(xling)} cross-linguistic features at "
                f"layer {layer}. Lower the quantile (currently {quantile}) "
                f"or reduce min_languages (currently {min_languages_val})."
            )

        progress(0.65, desc="computing magnitude-matched null and beta")
        mag = np.mean(np.stack(list(acts.values())), axis=0)
        res = view1_experiment(
            field, xling, activation_magnitude=mag,
            n_boot=int(n_boot), n_null=int(n_null), report=False,
        )

        progress(0.95, desc="writing summary")
        cr = res["cross"]
        ir = res["internal"]

        rows = [
            ("selected_n", int(len(xling))),
            ("layer", int(layer)),
            ("pooling", pooling),
            ("internal_beta", round(ir["beta"], 4)),
            ("internal_beta_ci_lo", round(ir["ci"][0], 4)),
            ("internal_beta_ci_hi", round(ir["ci"][1], 4)),
            ("cross_beta", round(cr["beta"], 4)),
            ("cross_beta_ci_lo", round(cr["ci"][0], 4)),
            ("cross_beta_ci_hi", round(cr["ci"][1], 4)),
            ("null_uniform_mean", round(cr["null_unif_mean"], 4)),
            ("null_matched_mean", round(cr["null_matched_mean"], 4)),
            ("cross_beta - null_matched_mean (HEADLINE)",
             round(cr["beta"] - cr["null_matched_mean"], 4)),
            ("cross_beta - null_uniform_mean (contrast)",
             round(cr["beta"] - cr["null_unif_mean"], 4)),
            ("matched_minus_uniform_null_gap",
             round(cr["null_matched_mean"] - cr["null_unif_mean"], 4)),
            ("widening_rate", round(cr["widening_rate"], 4)),
            ("max_hist_dev", round(cr["max_hist_dev"], 4)),
        ]
        df = pd.DataFrame(rows, columns=["metric", "value"])

        status_lines = [
            f"**layer** {layer} &nbsp;&nbsp; **pooling** {pooling} "
            f"&nbsp;&nbsp; **selected_n** {len(xling)}",
            f"**HEADLINE** (cross_beta - null_matched_mean) = "
            f"`{cr['beta'] - cr['null_matched_mean']:+.4f}` "
            f"(uniform-null contrast: "
            f"`{cr['beta'] - cr['null_unif_mean']:+.4f}`)",
        ]
        if cr["widening_rate"] is not None and cr["widening_rate"] > 0.05:
            status_lines.append(
                f"⚠ widening_rate = `{cr['widening_rate']:.3f}` > 0.05 — "
                f"n_bins (20) may be too fine for this magnitude distribution."
            )
        status_md = "  \n".join(status_lines)

        # JSON download
        json_dict = {
            "layer": int(layer),
            "pooling": pooling,
            "concepts": concepts,
            "min_languages": int(min_languages_val),
            "quantile": float(quantile),
            "selected_n": int(len(xling)),
            "selected_indices": [int(x) for x in xling],
            "internal_beta": float(ir["beta"]),
            "internal_ci": [float(x) for x in ir["ci"]],
            "cross_beta": float(cr["beta"]),
            "cross_ci": [float(x) for x in cr["ci"]],
            "null_uniform_mean": float(cr["null_unif_mean"]),
            "null_uniform_sd": float(cr["null_unif_sd"]),
            "null_matched_mean": float(cr["null_matched_mean"]),
            "null_matched_sd": float(cr["null_matched_sd"]),
            "headline_matched":
                float(cr["beta"] - cr["null_matched_mean"]),
            "headline_uniform":
                float(cr["beta"] - cr["null_unif_mean"]),
            "widening_rate": float(cr["widening_rate"]),
            "max_hist_dev": float(cr["max_hist_dev"]),
            "n_bins": int(cr["n_bins"]),
        }
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(tempfile.gettempdir(), "water_tool_exports")
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(
            out_dir,
            f"view4_interference_layer{layer}_{pooling}_{ts}.json"
        )
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2, default=str)

        progress(1.0, desc="done")
        return df, status_md, json_path
    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")
