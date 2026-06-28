"""View 4: Cross-field interference (β = interference_scaling_exponent).

For a multilingual concept SET (each language contributes a list of
words pooled together), measures whether the cross-linguistic feature
cluster at a chosen SAE layer occupies a better-renormalized sub-field
than a magnitude-matched random null. Reports a critical-exponent-
style scaling exponent β, NOT a renormalization-group flow.

============================================================================
REPRESENTATION USED FOR FEATURE SELECTION  (read this before interpreting)
============================================================================

`select_crosslingual_features` operates on **SAE feature activations at
the chosen residual-stream layer**, NOT on the input embedding table.

Per language word:
  - tokenize the word,
  - run the model forward with a hook on `model.model.layers[layer]`
    to capture the residual stream at the OUTPUT of that block,
  - encode the captured residual through `sae.encode()`,
  - pool across sub-token positions per `pooling`:
      * "last": only the final sub-token (default — resolves the
                multi-token vs single-token cross-language asymmetry).
      * "mean": all non-BOS positions (legacy, available as sensitivity
                check).

Per language, activations are AVERAGED across the language's word
list (so the "concept" is the SET of provided words, not a single
word). Selection then runs across the per-language pooled vectors.

============================================================================
SELECTION MODE
============================================================================

Two modes, exposed in the UI:

  ``top_k`` (default).
      Fixed cluster size: pick the top `top_k` features ranked by the
      min_languages-th highest per-language activation. This removes
      the variable-k confound — depth-sweep counts are commensurable
      across layers. If a layer has fewer than `top_k` features above
      a minimum activation floor, the result reports `insufficient=True`
      and the actual_k — the selector does NOT loosen the floor to
      fill.

  ``quantile``.
      Legacy: per-language top (1-quantile) of strictly positive
      activations; intersect across languages. Variable k by layer.
      Kept for comparison.

============================================================================
SHUFFLE / PERMUTATION CONTROL
============================================================================

Per-layer permutation control: for each language, permute the
activation vector across feature indices (preserving each language's
activation distribution but destroying the cross-linguistic
correspondence). Re-select features from the shuffled activations and
re-compute the matched-null headline. `headline_shuffled` should be
near zero if the real headline reflects genuine cross-linguistic
geometry; if it's similar to `headline_matched`, the effect is
architectural rather than semantic.

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
    "en: water, river, sea, rain, ice\n"
    "es: agua, río, mar, lluvia, hielo\n"
    "fr: eau, rivière, mer, pluie, glace\n"
    "de: Wasser, Fluss, Meer, Regen, Eis\n"
    "it: acqua, fiume, mare, pioggia, ghiaccio\n"
    "pt: água, rio, mar, chuva, gelo\n"
)


def parse_concept_list(text: str) -> dict:
    """Parse a multi-line concept list of the form

        en: water, river, sea
        es: agua, río, mar
        de: Wasser, Fluss, Meer

    Each language contributes a SET of words pooled by mean activation.
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
    """Per-language mean SAE feature activation for the concept SET.

    For each language, every word is tokenized and forwarded; the
    residual stream at the output of `model.model.layers[layer]` is
    captured via a hook, encoded by `sae.encode()`, and pooled across
    sub-token positions per `pooling`. The per-word vectors are then
    averaged across the language's word list to form one activation
    vector per language.

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
                    resid = captured["resid"][0].float()   # (seq, d_model)
                    feats = sae.encode(
                        resid.to(sae.W_enc.dtype)
                    ).float().cpu().numpy()                # (seq, d_sae)
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

    sae_lens stores W_dec as (d_sae, d_model), rows are unit features.
    """
    W = sae.W_dec.detach().float().cpu().numpy()
    if W.ndim != 2:
        raise ValueError(f"unexpected W_dec shape {W.shape}")
    if W.shape[0] < W.shape[1]:
        W = W.T
    return SuperpositionField(W, name=name)


def _shuffle_acts(acts_by_lang: dict, seed: int) -> dict:
    """Independently permute each language's activation vector across
    feature indices. Preserves each language's activation distribution
    but destroys cross-linguistic correspondence."""
    rng = np.random.default_rng(seed)
    return {
        lang: v[rng.permutation(len(v))]
        for lang, v in acts_by_lang.items()
    }


def _run_one_pipeline(field: SuperpositionField, acts: dict,
                      *, mode: str, top_k: int, quantile: float,
                      min_languages, min_activation_floor: float,
                      n_boot: int, n_null: int,
                      seed: int = 0):
    """Single-pipeline run: select features from `acts`, compute the
    cross-field result against the magnitude-matched null. Returns
    (res_dict, selected_indices, selection_info)."""
    if min_languages in (None, "", 0, "all", "All"):
        min_lang_val = len(acts)
    else:
        min_lang_val = int(min_languages)

    xling, info = select_crosslingual_features(
        acts,
        min_languages=min_lang_val,
        mode=mode,
        top_k=int(top_k),
        quantile=float(quantile),
        min_activation_floor=float(min_activation_floor),
        return_info=True,
    )
    if len(xling) < 3:
        return None, xling, info

    mag = np.mean(np.stack(list(acts.values())), axis=0)
    res = view1_experiment(
        field, xling, activation_magnitude=mag,
        n_boot=int(n_boot), n_null=int(n_null), report=False, seed=seed,
    )
    return res, xling, info


def run_interference(concept_text: str, layer, pooling: str,
                     mode: str, top_k: int, quantile: float,
                     min_languages, min_activation_floor: float,
                     n_boot, n_null,
                     compute_shuffle: bool,
                     progress=gr.Progress(track_tqdm=False)):
    """Top-level Gradio handler. Returns (summary_df, status_md,
    json_path_for_download)."""
    try:
        progress(0.02, desc="parsing concept list")
        concepts = parse_concept_list(concept_text)
        if len(concepts) < 2:
            raise gr.Error("Need at least 2 languages in the concept list.")

        layer = int(layer)

        progress(0.05, desc=f"loading model (cached) and SAE layer {layer}")
        model, tokenizer = load()
        sae = get_sae(layer)
        field = _field_from_sae(sae,
                                name=f"gemma-scope-2b-canonical L{layer}")

        progress(0.15, desc="per-language activations (pooled across word set)")
        acts = _concept_mean_activations(
            model, tokenizer, sae, layer, concepts, pooling=pooling
        )

        progress(0.40, desc="selecting cross-linguistic features (real)")
        res_real, xling, sel_info = _run_one_pipeline(
            field, acts,
            mode=mode, top_k=top_k, quantile=quantile,
            min_languages=min_languages,
            min_activation_floor=min_activation_floor,
            n_boot=n_boot, n_null=n_null, seed=0,
        )
        if res_real is None:
            actual_k = sel_info.get("actual_k", 0)
            raise gr.Error(
                f"Selected only {actual_k} cross-linguistic features at "
                f"layer {layer} (mode={mode}). Lower top_k / min_languages "
                f"or check that the selected concepts actually fire at "
                f"this layer."
            )

        cr = res_real["cross"]
        ir = res_real["internal"]

        headline_shuf = None
        shuf_info = None
        shuf_xling_n = None
        if compute_shuffle:
            progress(0.75,
                     desc="shuffle control (permuted language assignment)")
            shuf_acts = _shuffle_acts(acts, seed=12345)
            res_shuf, xling_shuf, shuf_sel = _run_one_pipeline(
                field, shuf_acts,
                mode=mode, top_k=top_k, quantile=quantile,
                min_languages=min_languages,
                min_activation_floor=min_activation_floor,
                n_boot=max(int(n_boot) // 2, 100),
                n_null=max(int(n_null) // 2, 100),
                seed=42,
            )
            shuf_xling_n = int(len(xling_shuf))
            if res_shuf is not None:
                crs = res_shuf["cross"]
                headline_shuf = (crs["beta"]
                                 - (crs["null_matched_mean"]
                                    if crs["null_matched_mean"] is not None
                                    else crs["null_unif_mean"]))
                shuf_info = dict(
                    cross_beta=float(crs["beta"]),
                    null_matched_mean=(
                        float(crs["null_matched_mean"])
                        if crs["null_matched_mean"] is not None else None
                    ),
                    null_uniform_mean=float(crs["null_unif_mean"]),
                    headline_significant=bool(crs["headline_significant"]),
                    headline_ci_lo=float(crs["headline_ci_lo"]),
                    headline_ci_hi=float(crs["headline_ci_hi"]),
                    actual_k=shuf_xling_n,
                )

        progress(0.95, desc="writing summary")

        headline_matched = (cr["beta"] - cr["null_matched_mean"]
                            if cr["null_matched_mean"] is not None
                            else cr["beta"] - cr["null_unif_mean"])
        headline_uniform = cr["beta"] - cr["null_unif_mean"]

        rows = [
            ("selected_n",                    int(len(xling))),
            ("selection_mode",                sel_info["mode"]),
            ("selection_insufficient",        bool(sel_info["insufficient"])),
            ("layer",                         int(layer)),
            ("pooling",                       pooling),
            ("interference_scaling_exponent_internal",
                                              round(ir["beta"], 4)),
            ("internal_ci_lo",                round(ir["ci"][0], 4)),
            ("internal_ci_hi",                round(ir["ci"][1], 4)),
            ("interference_scaling_exponent_cross",
                                              round(cr["beta"], 4)),
            ("cross_ci_lo",                   round(cr["ci"][0], 4)),
            ("cross_ci_hi",                   round(cr["ci"][1], 4)),
            ("null_uniform_mean",             round(cr["null_unif_mean"], 4)),
            ("null_matched_mean",
                round(cr["null_matched_mean"], 4)
                if cr["null_matched_mean"] is not None else "n/a"),
            ("HEADLINE_matched (β_cross - matched_null)",
                                              round(headline_matched, 4)),
            ("headline_ci_lo",                round(cr["headline_ci_lo"], 4)),
            ("headline_ci_hi",                round(cr["headline_ci_hi"], 4)),
            ("headline_significant (CI excludes 0)",
                                              bool(cr["headline_significant"])),
            ("HEADLINE_uniform (β_cross - uniform_null)",
                                              round(headline_uniform, 4)),
            ("matched_minus_uniform_null_gap",
                round((cr["null_matched_mean"] - cr["null_unif_mean"])
                      if cr["null_matched_mean"] is not None else 0.0, 4)),
            ("widening_rate",
                round(cr["widening_rate"], 4)
                if cr["widening_rate"] is not None else "n/a"),
            ("max_hist_dev",
                round(cr["max_hist_dev"], 4)
                if cr["max_hist_dev"] is not None else "n/a"),
            ("cosine_mu_mean",
                round(cr["cosine_stats"]["summary"]["mu_mean"], 5)),
            ("cosine_sig2_mean",
                round(cr["cosine_stats"]["summary"]["sig2_mean"], 5)),
        ]
        if compute_shuffle:
            if shuf_info is not None and headline_shuf is not None:
                rows.append((
                    "HEADLINE_shuffled (control; should be ~0)",
                    round(headline_shuf, 4),
                ))
                rows.append((
                    "shuffled_significant", shuf_info["headline_significant"]))
                rows.append(("shuffled_selected_n", shuf_xling_n))
            else:
                rows.append((
                    "HEADLINE_shuffled (control)",
                    f"insufficient (k={shuf_xling_n})"))

        df = pd.DataFrame(rows, columns=["metric", "value"])

        status_lines = [
            f"**layer** {layer} &nbsp;&nbsp; **pooling** {pooling} "
            f"&nbsp;&nbsp; **mode** {sel_info['mode']} &nbsp;&nbsp; "
            f"**selected_n** {len(xling)}"
            + (" ⚠ insufficient" if sel_info["insufficient"] else ""),
            f"**HEADLINE_matched** = `{headline_matched:+.4f}` &nbsp;&nbsp; "
            f"CI95 = [`{cr['headline_ci_lo']:+.4f}`, "
            f"`{cr['headline_ci_hi']:+.4f}`] &nbsp;&nbsp; "
            f"**{'significant' if cr['headline_significant'] else 'NOT significant'}**",
            f"(uniform-null contrast: `{headline_uniform:+.4f}`)",
        ]
        if compute_shuffle:
            if headline_shuf is not None:
                status_lines.append(
                    f"**Shuffle control headline** = `{headline_shuf:+.4f}` "
                    f"(should be near 0 if the real headline reflects "
                    f"genuine cross-linguistic geometry)"
                )
            else:
                status_lines.append(
                    f"**Shuffle control: selection eliminated** "
                    f"(k={shuf_xling_n}). Under permuted language "
                    f"assignment the selector finds no cluster — the "
                    f"strongest possible control. The cross-linguistic "
                    f"structure is necessary for the cluster to exist; "
                    f"there is nothing left to spuriously match."
                )
        if cr["widening_rate"] is not None and cr["widening_rate"] > 0.05:
            status_lines.append(
                f"⚠ widening_rate = `{cr['widening_rate']:.3f}` > 0.05 — "
                f"n_bins (20) may be too fine for this magnitude "
                f"distribution."
            )
        status_lines.append(
            "*β = interference_scaling_exponent: analog of a critical "
            "exponent in scaling analysis; NOT a renormalization-group "
            "flow.*"
        )
        status_md = "  \n".join(status_lines)

        # JSON download with full result
        json_dict = {
            "layer": int(layer),
            "pooling": pooling,
            "selection": {
                "mode": sel_info["mode"],
                "requested_k":      sel_info.get("requested_k"),
                "actual_k":         sel_info["actual_k"],
                "insufficient":     bool(sel_info["insufficient"]),
                "min_languages":    sel_info["min_languages"],
                "min_activation_floor": sel_info.get("min_activation_floor"),
                "score_threshold":  sel_info.get("score_threshold"),
                "n_eligible":       sel_info.get("n_eligible"),
                "quantile":         sel_info.get("quantile"),
            },
            "concepts": concepts,
            "selected_indices": [int(x) for x in xling],
            "interference_scaling_exponent_note":
                "Analog of a critical exponent (scaling-analysis term); "
                "NOT a renormalization-group flow.",
            "interference_scaling_exponent_internal": float(ir["beta"]),
            "internal_ci": [float(x) for x in ir["ci"]],
            "interference_scaling_exponent_cross": float(cr["beta"]),
            "cross_ci": [float(x) for x in cr["ci"]],
            "null_uniform_mean": float(cr["null_unif_mean"]),
            "null_uniform_sd": float(cr["null_unif_sd"]),
            "null_matched_mean": (
                float(cr["null_matched_mean"])
                if cr["null_matched_mean"] is not None else None
            ),
            "null_matched_sd": (
                float(cr["null_matched_sd"])
                if cr["null_matched_sd"] is not None else None
            ),
            "headline_matched": float(headline_matched),
            "headline_uniform": float(headline_uniform),
            "headline_ci_lo": float(cr["headline_ci_lo"]),
            "headline_ci_hi": float(cr["headline_ci_hi"]),
            "headline_significant": bool(cr["headline_significant"]),
            "widening_rate": (
                float(cr["widening_rate"])
                if cr["widening_rate"] is not None else None
            ),
            "max_hist_dev": (
                float(cr["max_hist_dev"])
                if cr["max_hist_dev"] is not None else None
            ),
            "n_bins": cr.get("n_bins"),
            "cosine_stats": cr["cosine_stats"],
            "shuffle_control": shuf_info,
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
