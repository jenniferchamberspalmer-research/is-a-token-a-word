"""View 5: cross-linguistic trajectory of a single source word.

UNIT INSTRUMENT — no aggregation, no selection, no β, no shuffle.
One source word traced through every transformer layer 0..n_layers-1.

Per layer L:
  - source_resid_L  : residual at the OUTPUT of model.model.layers[L],
                      pooled at the LAST sub-token of the source word.
  - For each comparison word w (cross-linguistic equivalent, synonym,
    or control equivalent):
      * w_resid_L    : same pooling for w.
      * cosine_to_source(L)[w] = cos(source_resid_L, w_resid_L).
      * rank_in_vocab(L)[w]   = position of w's last-token id in the
                                descending sort of cosines from
                                source_resid_L against the INPUT
                                embedding table (every vocab token).

Three comparison classes (toggleable):
  - translations          : cross-linguistic equivalents of the source
  - synonyms              : within-language near-synonyms of the source
                            (e.g. water → liquid, H2O)
  - control_translations  : a control source word + its cross-linguistic
                            equivalents, traced identically (default
                            chair + silla / chaise / Stuhl / sedia /
                            cadeira). For the control class the cosine
                            is to the control SOURCE's residual, and
                            the rank is in the control source's vocab
                            cosines — i.e. the control is a self-
                            contained second trajectory.

Outputs: per-layer wide-format table, a two-panel trajectory plot
(cosine vs layer + rank vs layer, log y), and a JSON dump of the raw
per-layer arrays.

Memory: trivial. Per layer, the only large allocation is a (vocab,)
cosine vector for the rank lookup; embed_norms is precomputed once.
No selection, no batched matmul over n_draws. No OOM possible at
realistic vocab sizes.
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


DEFAULT_SOURCE = "water"
DEFAULT_EQUIVALENTS = (
    "es: agua\n"
    "fr: eau\n"
    "de: Wasser\n"
    "it: acqua\n"
    "pt: água\n"
)
DEFAULT_SYNONYMS = "liquid, H2O"
DEFAULT_CONTROL_SOURCE = "chair"
DEFAULT_CONTROL_EQUIVALENTS = (
    "es: silla\n"
    "fr: chaise\n"
    "de: Stuhl\n"
    "it: sedia\n"
    "pt: cadeira\n"
)


def _parse_equivalents(text: str) -> dict:
    """Parse 'lang: word' lines. Returns {lang: word}."""
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        lang, word = line.split(":", 1)
        word = word.strip()
        if word:
            out[lang.strip()] = word
    return out


def _last_token_id(tokenizer, word: str) -> int:
    """Tokenize ' word' (with leading space, matching last-pooling) and
    return the LAST sub-token id."""
    ids = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty tokenization for {word!r}")
    return int(ids[-1])


@torch.no_grad()
def _last_token_residuals_all_layers(model, tokenizer, word: str):
    """One forward pass; return [residual_L0, ..., residual_L(n-1)],
    each (d_model,) float32 on CPU.

    Layer mapping (matches View 3 / View 4 convention): hidden_states[0]
    is the embedding input; hidden_states[L+1] is the output of
    model.model.layers[L] for L=0..n_layers-1.
    """
    device = next(model.parameters()).device
    enc = tokenizer(" " + word.strip(), return_tensors="pt").to(device)
    out = model(**enc, output_hidden_states=True)
    n_layers = len(model.model.layers)
    return [
        out.hidden_states[L + 1][0, -1, :].float().detach().cpu()
        for L in range(n_layers)
    ]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = (a.norm() * b.norm()).clamp(min=1e-8)
    return float((a @ b) / denom)


@torch.no_grad()
def _ranks_in_vocab(source_resid: torch.Tensor,
                    embed_table: torch.Tensor,
                    embed_norms: torch.Tensor,
                    token_ids: dict) -> dict:
    """For source_resid (one layer), compute cosine to every vocab
    token and return the rank of each id in `token_ids`.

    Operates on the device of embed_table; minimal memory."""
    src = source_resid.to(embed_table.device).to(embed_table.dtype)
    src_n = src / src.float().norm().clamp(min=1e-8)
    dots = embed_table @ src_n                       # (vocab,)
    cosines = (dots.float() / embed_norms).cpu().numpy()
    sorted_idx = np.argsort(-cosines)                # descending
    rank_of = np.empty_like(sorted_idx)
    rank_of[sorted_idx] = np.arange(len(sorted_idx))
    return {key: int(rank_of[tid]) for key, tid in token_ids.items()}


def _make_plot(layers, traj_data: dict, source: str,
               control_source: str, out_dir: str) -> str:
    """Two-panel plot: cosine + log-rank per layer.

    traj_data is keyed by display label and each value is
    {"cosine": [...], "rank": [...], "class": "trans"|"syn"|"ctl"}.
    Class drives the line style.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_for = {
        "trans": dict(linestyle="-", alpha=1.0),
        "syn":   dict(linestyle="--", alpha=1.0),
        "ctl":   dict(linestyle=":", alpha=0.7),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for label, data in traj_data.items():
        st = style_for.get(data["class"], style_for["trans"])
        ax1.plot(layers, data["cosine"], label=label, **st)
        # rank=0 is impossible to show on log; offset by 1 (rank 1-indexed)
        ranks = np.asarray(data["rank"], dtype=float) + 1.0
        ax2.plot(layers, ranks, label=label, **st)

    title_src = f"'{source}'" + (
        f"  (control: '{control_source}')" if control_source else ""
    )
    ax1.set_ylabel("cosine to source residual")
    ax1.set_title(f"Cross-linguistic trajectory — {title_src}")
    ax1.legend(fontsize=7, ncol=2, loc="best")
    ax1.grid(alpha=0.3)

    ax2.set_yscale("log")
    ax2.set_xlabel("layer (0 = post-layer-0, n-1 = post-last-layer)")
    ax2.set_ylabel("rank of last-token id (1-indexed, log) in source's "
                   "vocab cosines")
    ax2.grid(alpha=0.3, which="both")

    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(out_dir, f"view5_trajectory_{ts}.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def run_trajectory(source_word: str,
                   equivalents_text: str,
                   synonyms_text: str,
                   control_source: str,
                   control_equiv_text: str,
                   progress=gr.Progress(track_tqdm=False)):
    """Top-level Gradio handler. Returns
    (per_layer_df, status_md, plot_image_path, json_download_path)."""
    try:
        source_word = (source_word or "").strip() or DEFAULT_SOURCE
        equivalents = _parse_equivalents(equivalents_text)
        synonyms = [s.strip()
                    for s in (synonyms_text or "").split(",")
                    if s.strip()]
        control_source = (control_source or "").strip()
        control_equivs = _parse_equivalents(control_equiv_text)

        progress(0.05, desc="loading model (cached)")
        model, tokenizer = load()
        n_layers = len(model.model.layers)
        layers = list(range(n_layers))

        # ----- forward passes (one per word) -----
        words = {"source": source_word}
        for lang, w in equivalents.items():
            words[f"trans:{lang}"] = w
        for syn in synonyms:
            words[f"syn:{syn}"] = syn
        if control_source:
            words["control_source"] = control_source
            for lang, w in control_equivs.items():
                words[f"ctl:{lang}"] = w

        progress(0.10, desc=f"forward passes for {len(words)} words")
        residuals = {}
        for key, w in words.items():
            residuals[key] = _last_token_residuals_all_layers(
                model, tokenizer, w)

        # ----- precompute embed table norms for rank lookup -----
        progress(0.45, desc="precomputing input-embedding-table norms")
        embed_table = model.get_input_embeddings().weight
        embed_norms = embed_table.float().norm(dim=-1).clamp(min=1e-8)
        # Per-rank target tokens: each comparison word's LAST sub-token
        token_ids = {}
        for lang, w in equivalents.items():
            token_ids[f"trans:{lang}"] = _last_token_id(tokenizer, w)
        for syn in synonyms:
            token_ids[f"syn:{syn}"] = _last_token_id(tokenizer, syn)
        token_ids_ctl = {}
        if control_source:
            for lang, w in control_equivs.items():
                token_ids_ctl[f"ctl:{lang}"] = _last_token_id(tokenizer, w)

        # ----- per-layer loop -----
        progress(0.55, desc=f"computing per-layer cosines + ranks "
                            f"across {n_layers} layers")
        # Per-key arrays
        cos_arrs = {k: np.zeros(n_layers) for k in
                    list(token_ids) + list(token_ids_ctl)}
        rank_arrs = {k: np.zeros(n_layers, dtype=int) for k in
                     list(token_ids) + list(token_ids_ctl)}

        for L in layers:
            src_resid = residuals["source"][L]
            # Cosines + ranks for translations & synonyms
            for key in token_ids:
                cos_arrs[key][L] = _cos(src_resid, residuals[key][L])
            r = _ranks_in_vocab(src_resid, embed_table, embed_norms,
                                token_ids)
            for key, rk in r.items():
                rank_arrs[key][L] = rk
            # Cosines + ranks for control class (vs CONTROL source)
            if control_source:
                ctl_resid = residuals["control_source"][L]
                for key in token_ids_ctl:
                    cos_arrs[key][L] = _cos(ctl_resid, residuals[key][L])
                rc = _ranks_in_vocab(ctl_resid, embed_table, embed_norms,
                                     token_ids_ctl)
                for key, rk in rc.items():
                    rank_arrs[key][L] = rk

        # ----- assemble per-layer wide table -----
        def _display_label(key: str) -> tuple[str, str]:
            """Return (class_tag, display_label) for a key."""
            if key.startswith("trans:"):
                _, lang = key.split(":", 1)
                return "trans", f"{lang}:{equivalents.get(lang, '?')}"
            if key.startswith("syn:"):
                _, w = key.split(":", 1)
                return "syn", f"syn:{w}"
            if key.startswith("ctl:"):
                _, lang = key.split(":", 1)
                return "ctl", f"ctl:{lang}:{control_equivs.get(lang, '?')}"
            return "trans", key

        rows = []
        for L in layers:
            row = {"layer": L}
            for key in cos_arrs:
                _, lbl = _display_label(key)
                row[f"cos_{lbl}"] = round(float(cos_arrs[key][L]), 4)
                row[f"rank_{lbl}"] = int(rank_arrs[key][L])
            rows.append(row)
        df = pd.DataFrame(rows)

        # ----- plot -----
        progress(0.92, desc="rendering plot")
        out_dir = os.path.join(tempfile.gettempdir(), "water_tool_exports")
        traj_for_plot = {}
        for key in cos_arrs:
            cls, lbl = _display_label(key)
            traj_for_plot[lbl] = dict(
                cosine=[float(x) for x in cos_arrs[key]],
                rank=[int(x) for x in rank_arrs[key]],
                **{"class": cls},
            )
        plot_path = _make_plot(layers, traj_for_plot,
                               source_word, control_source, out_dir)

        # ----- JSON dump -----
        progress(0.97, desc="writing JSON")
        json_dict = {
            "source": source_word,
            "equivalents": equivalents,
            "synonyms": synonyms,
            "control_source": control_source,
            "control_equivalents": control_equivs,
            "layers": layers,
            "trajectory": {},
        }
        for key in cos_arrs:
            cls, lbl = _display_label(key)
            json_dict["trajectory"][lbl] = {
                "class": cls,
                "token_id": int(
                    token_ids.get(key, token_ids_ctl.get(key, -1))),
                "cosine_per_layer":
                    [float(x) for x in cos_arrs[key]],
                "rank_per_layer":
                    [int(x) for x in rank_arrs[key]],
            }
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(
            out_dir,
            f"view5_trajectory_{source_word}_{ts}.json"
        )
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2, default=str)

        # ----- status message -----
        nt = sum(1 for k in cos_arrs if k.startswith("trans:"))
        ns = sum(1 for k in cos_arrs if k.startswith("syn:"))
        nc = sum(1 for k in cos_arrs if k.startswith("ctl:"))
        status = (
            f"**source** `{source_word}` &nbsp;&nbsp; "
            f"**control** "
            f"{'`' + control_source + '`' if control_source else '(off)'}"
            f" &nbsp;&nbsp; "
            f"**translations** {nt} &nbsp;&nbsp; "
            f"**synonyms** {ns} &nbsp;&nbsp; "
            f"**control_equivalents** {nc} &nbsp;&nbsp; "
            f"**layers** {n_layers} (0..{n_layers - 1})"
        )

        progress(1.0, desc="done")
        return df, status, plot_path, json_path

    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")
