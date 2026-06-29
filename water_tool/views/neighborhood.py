"""View 6: relational neighborhood dynamics.

Extension of View 5 — reuses the same vector extraction (last-token
residual per layer) and the same neighbor space (resid_L @ embed_table.T:
each word's layer-L residual ranked against the static input embedding
table). UNIT-LEVEL relational instrument.

Purpose. Test whether View 5's early-layer cross-linguistic identity is
a full neighborhood-level structure or merely pairwise closeness between
the source and its translations.

Per layer L, for every word w, take the top-N=200 vocabulary tokens by
cosine of resid_L(w) against the embedding table. Slicing gives the
top-N=100 (headline) and top-N=50 (robustness).

For each pair (source, comparison_word), and for each N in {50, 100, 200},
compute:

  - JACCARD overlap (symmetric)              |S∩C| / |S∪C|
  - RECIPROCAL-RANK-WEIGHTED overlap         sum(1/(r_src+1) over shared)
                                              normalized by max possible
                                              over the source's top-N
                                              (always anchored to the
                                              SOURCE's ranking)
  - MEAN RANK of shared neighbors            mean(rank-in-source) over
                                              shared (1-indexed; NaN if
                                              no shared)
  - COUNT of shared neighbors                |S∩C|

Symmetry note: Jaccard is symmetric. The reciprocal-rank-weighted
overlap and mean rank are NOT — they are always anchored to the source
word's ranking so every comparison is measured on the same footing.

The headline metric is N=100. N=50 and N=200 are computed as
robustness checks; the trajectory shape at the headline must survive
those cutoffs or it's a cutoff artifact.

Three comparison classes:
  - translations  : source's cross-linguistic equivalents
  - synonyms      : source's within-language near-synonyms
  - control       : separate control_source + its translations, traced
                    as a self-contained second trajectory (each control
                    pair is anchored to the CONTROL source's ranking,
                    not the original source's)

Memory: trivial. Per word per layer we keep only the top-200 token
indices (200 ints). No batched matmul; the topk runs on the GPU one
word at a time.
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
from .trajectory import (
    _parse_equivalents,
    _last_token_id,
    _last_token_residuals_all_layers,
)


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

# Fixed across every word/layer/control. 100 is the headline; 50/200
# are robustness, not alternative headlines.
HEADLINE_N = 100
ROBUSTNESS_NS = (50, 100, 200)
N_MAX = max(ROBUSTNESS_NS)   # store this many neighbors per word per layer


@torch.no_grad()
def _topn_vocab_neighbors(source_resid: torch.Tensor,
                          embed_table: torch.Tensor,
                          embed_norms: torch.Tensor,
                          n_max: int = N_MAX) -> np.ndarray:
    """Top-`n_max` vocabulary token ids by cosine of source_resid
    against the static embedding table. Returned as a 1-D ndarray
    of length n_max, sorted descending by cosine.

    Same neighbor space as View 5 (resid @ embed_table.T). Do NOT
    substitute residual-vs-residual neighbors here — that's a
    different measurement.
    """
    src = source_resid.to(embed_table.device).to(embed_table.dtype)
    src_n = src / src.float().norm().clamp(min=1e-8)
    dots = embed_table @ src_n                      # (vocab,)
    cosines = (dots.float() / embed_norms)
    top_idx = torch.topk(cosines, n_max).indices    # (n_max,)
    return top_idx.detach().cpu().numpy().astype(np.int64)


def _pair_metrics_at_layer(src_top: np.ndarray,
                           cmp_top: np.ndarray,
                           n: int) -> dict:
    """Compute the four metrics for a single (source, comparison) pair
    at one layer, at neighborhood size `n`. Both `src_top` and
    `cmp_top` must be at least length `n`, sorted descending by source's
    and comparison's own rankings respectively.

    All rank-based metrics are anchored to the SOURCE's ranking — the
    source's top is sliced to `n`, the comparison's top is sliced to
    `n`, the intersection is taken, and for each shared token its rank
    is its position in `src_top` (0-indexed).
    """
    src_arr = src_top[:n]
    cmp_arr = cmp_top[:n]
    src_set = set(int(x) for x in src_arr.tolist())
    cmp_set = set(int(x) for x in cmp_arr.tolist())
    shared = src_set & cmp_set
    union = src_set | cmp_set

    jaccard = len(shared) / max(len(union), 1)
    count_shared = len(shared)

    # Rank of each shared token in the SOURCE's ordering.
    src_rank_of = {int(t): r for r, t in enumerate(src_arr.tolist())}
    shared_ranks = sorted(src_rank_of[t] for t in shared)

    if shared_ranks:
        mean_rank_1idx = float(np.mean(shared_ranks)) + 1.0
        # Reciprocal-rank-weighted overlap, normalized so 1.0 means
        # "every source-top-N slot is shared".
        weighted_sum = float(sum(1.0 / (r + 1) for r in shared_ranks))
        max_weighted = float(sum(1.0 / (r + 1) for r in range(n)))
        rr_weighted = weighted_sum / max_weighted if max_weighted else 0.0
    else:
        mean_rank_1idx = float("nan")
        rr_weighted = 0.0

    return dict(
        jaccard=float(jaccard),
        rr_weighted=float(rr_weighted),
        mean_rank_shared=mean_rank_1idx,
        count_shared=int(count_shared),
        shared_tokens=sorted(int(t) for t in shared),
    )


def _make_plots(layers, pairs_data: list, out_dir: str) -> str:
    """Render the 2x2 plot grid: Jaccard, reciprocal-rank-weighted,
    disconfirm test (translation mean vs control mean), and cutoff
    sensitivity. Returns the PNG path.

    pairs_data: list of dicts {label, class, metrics_n50, metrics_n100,
    metrics_n200}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_for = {
        "trans": dict(linestyle="-", alpha=1.0),
        "syn":   dict(linestyle="--", alpha=1.0),
        "ctl":   dict(linestyle=":", alpha=0.7),
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Plot 1: Jaccard (headline N=100) per pair
    for p in pairs_data:
        st = style_for.get(p["class"], style_for["trans"])
        ax1.plot(layers, p["metrics_n100"]["jaccard"],
                 label=p["label"], **st)
    ax1.set_title(f"Plot 1 — Jaccard overlap (N={HEADLINE_N}, headline)")
    ax1.set_xlabel("layer")
    ax1.set_ylabel("Jaccard")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, ncol=2, loc="best")

    # Plot 2: Reciprocal-rank-weighted overlap (headline N=100) per pair
    for p in pairs_data:
        st = style_for.get(p["class"], style_for["trans"])
        ax2.plot(layers, p["metrics_n100"]["rr_weighted"],
                 label=p["label"], **st)
    ax2.set_title(f"Plot 2 — reciprocal-rank-weighted overlap "
                  f"(N={HEADLINE_N}, source-anchored)")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("RR-weighted overlap")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, ncol=2, loc="best")

    # Plot 3: Disconfirm test — translation mean vs control mean
    trans_jaccs = np.array([p["metrics_n100"]["jaccard"] for p in pairs_data
                            if p["class"] == "trans"])
    ctl_jaccs = np.array([p["metrics_n100"]["jaccard"] for p in pairs_data
                          if p["class"] == "ctl"])
    if trans_jaccs.size:
        ax3.plot(layers, trans_jaccs.mean(axis=0),
                 label="translations (mean)", color="tab:blue", linewidth=2)
        ax3.fill_between(layers,
                         trans_jaccs.min(axis=0), trans_jaccs.max(axis=0),
                         color="tab:blue", alpha=0.15, label="trans min–max")
    if ctl_jaccs.size:
        ax3.plot(layers, ctl_jaccs.mean(axis=0),
                 label="controls (mean)", color="tab:gray",
                 linewidth=2, linestyle="--")
        ax3.fill_between(layers,
                         ctl_jaccs.min(axis=0), ctl_jaccs.max(axis=0),
                         color="tab:gray", alpha=0.15, label="ctl min–max")
    ax3.set_title("Plot 3 — disconfirm: translation mean vs control mean")
    ax3.set_xlabel("layer")
    ax3.set_ylabel(f"Jaccard (N={HEADLINE_N})")
    ax3.set_ylim(0, 1)
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8, loc="best")

    # Plot 4: Cutoff robustness — translation mean Jaccard at N=50/100/200
    if trans_jaccs.size:
        for N, color in [(50, "tab:orange"), (100, "tab:blue"),
                         (200, "tab:green")]:
            arrs = np.array([p[f"metrics_n{N}"]["jaccard"]
                             for p in pairs_data if p["class"] == "trans"])
            ax4.plot(layers, arrs.mean(axis=0), label=f"N={N}",
                     color=color, linewidth=2)
    ax4.set_title("Plot 4 — cutoff sensitivity (translation Jaccard mean)")
    ax4.set_xlabel("layer")
    ax4.set_ylabel("Jaccard (translation mean)")
    ax4.set_ylim(0, 1)
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=8, loc="best")

    fig.suptitle("View 6 — Relational neighborhood dynamics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"view6_neighborhood_{ts}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_neighborhood(source_word: str,
                     equivalents_text: str,
                     synonyms_text: str,
                     control_source: str,
                     control_equiv_text: str,
                     progress=gr.Progress(track_tqdm=False)):
    """Top-level Gradio handler. Returns (per_layer_df, status_md,
    plot_image_path, json_download_path)."""
    try:
        source_word = (source_word or "").strip() or DEFAULT_SOURCE
        equivalents = _parse_equivalents(equivalents_text)
        synonyms = [s.strip() for s in (synonyms_text or "").split(",")
                    if s.strip()]
        control_source = (control_source or "").strip()
        control_equivs = _parse_equivalents(control_equiv_text)

        progress(0.05, desc="loading model (cached)")
        model, tokenizer = load()
        n_layers = len(model.model.layers)
        layers = list(range(n_layers))

        # ----- collect every word that needs a residual trajectory -----
        # role -> word; role is the per-pair anchor we use for ranking
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
        residuals = {key: _last_token_residuals_all_layers(model, tokenizer, w)
                     for key, w in words.items()}

        progress(0.40, desc="precomputing embedding-table norms")
        embed_table = model.get_input_embeddings().weight
        embed_norms = embed_table.float().norm(dim=-1).clamp(min=1e-8)

        # ----- top-N_MAX neighbors per word per layer -----
        progress(0.50, desc=f"top-{N_MAX} vocab neighbors per word per layer")
        # neighbors[role][L] : ndarray of shape (N_MAX,)
        neighbors = {role: [] for role in words}
        for L in layers:
            for role in words:
                neighbors[role].append(
                    _topn_vocab_neighbors(residuals[role][L],
                                          embed_table, embed_norms, N_MAX)
                )

        # ----- pair definitions (source anchor on left) -----
        pairs = []
        for lang, w in equivalents.items():
            pairs.append(dict(
                label=f"{lang}:{w}", cls="trans",
                src_role="source", cmp_role=f"trans:{lang}",
                src_word=source_word, cmp_word=w))
        for syn in synonyms:
            pairs.append(dict(
                label=f"syn:{syn}", cls="syn",
                src_role="source", cmp_role=f"syn:{syn}",
                src_word=source_word, cmp_word=syn))
        if control_source:
            for lang, w in control_equivs.items():
                pairs.append(dict(
                    label=f"ctl:{lang}:{w}", cls="ctl",
                    src_role="control_source", cmp_role=f"ctl:{lang}",
                    src_word=control_source, cmp_word=w))

        progress(0.65, desc="per-pair per-layer metrics at N=50/100/200")
        for p in pairs:
            for N in ROBUSTNESS_NS:
                m = dict(jaccard=[], rr_weighted=[],
                         mean_rank_shared=[], count_shared=[],
                         shared_tokens=[])
                for L in layers:
                    src_top = neighbors[p["src_role"]][L]
                    cmp_top = neighbors[p["cmp_role"]][L]
                    row = _pair_metrics_at_layer(src_top, cmp_top, N)
                    m["jaccard"].append(row["jaccard"])
                    m["rr_weighted"].append(row["rr_weighted"])
                    m["mean_rank_shared"].append(row["mean_rank_shared"])
                    m["count_shared"].append(row["count_shared"])
                    m["shared_tokens"].append(row["shared_tokens"])
                p[f"metrics_n{N}"] = m

        # ----- wide table for the UI (headline N=100 only) -----
        progress(0.90, desc="building per-layer summary table")
        rows = []
        for L in layers:
            row = {"layer": L}
            for p in pairs:
                row[f"jaccard_n100_{p['cls']}_{p['label']}"] = round(
                    p["metrics_n100"]["jaccard"][L], 4)
                row[f"rr_weighted_n100_{p['cls']}_{p['label']}"] = round(
                    p["metrics_n100"]["rr_weighted"][L], 4)
                row[f"count_shared_n100_{p['cls']}_{p['label']}"] = (
                    int(p["metrics_n100"]["count_shared"][L]))
            rows.append(row)
        df = pd.DataFrame(rows)

        # ----- plots (one PNG, 2x2 grid) -----
        out_dir = os.path.join(tempfile.gettempdir(), "water_tool_exports")
        plot_path = _make_plots(layers, pairs, out_dir)

        # ----- JSON dump (all N, all metrics, raw neighbor lists) -----
        progress(0.96, desc="writing JSON")
        json_dict = {
            "source": source_word,
            "equivalents": equivalents,
            "synonyms": synonyms,
            "control_source": control_source,
            "control_equivalents": control_equivs,
            "Ns_robustness": list(ROBUSTNESS_NS),
            "headline_N": HEADLINE_N,
            "layers": layers,
            "neighbors_top_n_max": {
                role: [arr.tolist() for arr in neighbors[role]]
                for role in neighbors
            },
            "pairs": [
                {
                    "label": p["label"],
                    "class": p["cls"],
                    "src_word": p["src_word"],
                    "cmp_word": p["cmp_word"],
                    "metrics_n50": p["metrics_n50"],
                    "metrics_n100": p["metrics_n100"],
                    "metrics_n200": p["metrics_n200"],
                }
                for p in pairs
            ],
        }
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(
            out_dir, f"view6_neighborhood_{source_word}_{ts}.json"
        )
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2, default=str)

        # ----- status -----
        nt = sum(1 for p in pairs if p["cls"] == "trans")
        ns = sum(1 for p in pairs if p["cls"] == "syn")
        nc = sum(1 for p in pairs if p["cls"] == "ctl")
        status = (
            f"**source** `{source_word}` &nbsp;&nbsp; "
            f"**control** "
            f"{'`' + control_source + '`' if control_source else '(off)'}"
            f" &nbsp;&nbsp; "
            f"**translations** {nt} &nbsp;&nbsp; "
            f"**synonyms** {ns} &nbsp;&nbsp; "
            f"**control_equivalents** {nc} &nbsp;&nbsp; "
            f"**layers** {n_layers} &nbsp;&nbsp; "
            f"**headline N** = {HEADLINE_N}  "
            f"(robustness N ∈ {list(ROBUSTNESS_NS)})"
        )

        progress(1.0, desc="done")
        return df, status, plot_path, json_path

    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")
