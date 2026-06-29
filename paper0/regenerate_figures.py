"""Paper 0 figure regeneration — pure plotting from committed JSON.

Reads every JSON under ``data/processed/`` and emits a PNG under
``figures/`` with the fixes from ``figures/FIGURES_MANIFEST.md``
applied:

  V5  per-figure title (source + control source). Top panel captioned
      as ANISOTROPY DIAGNOSTIC; bottom panel captioned as THE FINDING
      (rank-in-vocab trajectory). The math is unchanged — this is the
      caption fix called out in manifest item #2.

  V6  per-figure title (source + control source) — replaces the
      leftover-brother suptitle called out in manifest item #1.
      Math, layout, and the 4-panel grid are bit-identical to the
      tool's _make_plots; only the suptitle changes.

  view6_brother_bird.png
      Generated from data/processed/view6/view6_brother_bird.json
      (manifest item: figure was absent, data present).

  view6_bread_chair_clean.png
      Bonus headline figure for the bread case. The committed
      view6_bread_chair plots all five Spanish/French/Italian/German/
      Portuguese translations including the three homographs
      (pan, pain, pane). The paper's headline claim is the CLEAN
      subset (Brot, pão), which fails all three controls. This file
      shows only the clean subset for the paper.

No model run. No view-logic edits. Reproducible from the JSONs alone.

Run:
    python paper0/regenerate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HEADLINE_N = 100

STYLE_FOR = {
    "trans": dict(linestyle="-", alpha=1.0),
    "syn":   dict(linestyle="--", alpha=1.0),
    "ctl":   dict(linestyle=":", alpha=0.7),
}


# ---------------------------------------------------------------- V5

def render_v5(json_data: dict, out_path: Path) -> None:
    """Cross-linguistic trajectory plot — 2 panels (cosine + log rank).

    The PER-FIGURE title carries source + control. Panel titles caption
    cosine as the anisotropy diagnostic and rank as the finding.
    """
    source = json_data["source"]
    control_source = json_data.get("control_source", "") or ""
    layers = json_data["layers"]
    trajectory = json_data["trajectory"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for label, data in trajectory.items():
        st = STYLE_FOR.get(data.get("class", "trans"), STYLE_FOR["trans"])
        ax1.plot(layers, data["cosine_per_layer"], label=label, **st)
        # rank is 0-indexed in JSON; +1 for log-scale display
        ranks = np.asarray(data["rank_per_layer"], dtype=float) + 1.0
        ax2.plot(layers, ranks, label=label, **st)

    # Top panel: ANISOTROPY DIAGNOSTIC (per manifest item #2)
    ax1.set_ylabel("cosine to source residual")
    ax1.set_title(
        "Anisotropy diagnostic — all classes rise to ~1.0 mid-network; "
        "this panel is the diagnostic, not the finding",
        fontsize=10,
    )
    ax1.legend(fontsize=7, ncol=2, loc="best")
    ax1.grid(alpha=0.3)

    # Bottom panel: THE FINDING (per manifest item #2)
    ax2.set_yscale("log")
    ax2.set_xlabel("layer (0 = post-layer-0, n-1 = post-last-layer)")
    ax2.set_ylabel("rank of last-token id (1-indexed, log)")
    ax2.set_title(
        "Finding — rank-in-vocab trajectory: early-layer cross-linguistic "
        "rank identity, collapse around layer 9",
        fontsize=10,
    )
    ax2.grid(alpha=0.3, which="both")

    title_src = f"'{source}'" + (
        f"  (control: '{control_source}')" if control_source else ""
    )
    fig.suptitle(
        f"View 5 — Cross-linguistic trajectory — {title_src}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- V6

def render_v6(json_data: dict, out_path: Path,
              title_suffix: str = "",
              trans_filter: callable | None = None) -> None:
    """Relational neighborhood dynamics — 2x2 grid.

    Per-figure title carries source + control (manifest item #1).
    Math is identical to the tool's _make_plots.

    trans_filter, if given, returns True for each trans-class pair to
    keep. Used by the bread-clean variant to restrict to the non-
    homograph subset (Brot, pão) without altering the source JSON.
    """
    source = json_data["source"]
    control_source = json_data.get("control_source", "") or ""
    layers = json_data["layers"]
    pairs_all = json_data["pairs"]

    if trans_filter is not None:
        pairs = []
        for p in pairs_all:
            if p["class"] == "trans" and not trans_filter(p):
                continue
            pairs.append(p)
    else:
        pairs = pairs_all

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Plot 1: Jaccard (headline N=100) per pair
    for p in pairs:
        st = STYLE_FOR.get(p["class"], STYLE_FOR["trans"])
        ax1.plot(layers, p["metrics_n100"]["jaccard"],
                 label=p["label"], **st)
    ax1.set_title(f"Plot 1 — Jaccard overlap (N={HEADLINE_N}, headline)")
    ax1.set_xlabel("layer"); ax1.set_ylabel("Jaccard"); ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7, ncol=2, loc="best")

    # Plot 2: Reciprocal-rank-weighted overlap (headline N=100) per pair
    for p in pairs:
        st = STYLE_FOR.get(p["class"], STYLE_FOR["trans"])
        ax2.plot(layers, p["metrics_n100"]["rr_weighted"],
                 label=p["label"], **st)
    ax2.set_title(
        f"Plot 2 — reciprocal-rank-weighted overlap "
        f"(N={HEADLINE_N}, source-anchored)"
    )
    ax2.set_xlabel("layer"); ax2.set_ylabel("RR-weighted overlap")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=7, ncol=2, loc="best")

    # Plot 3: translation mean vs control mean (disconfirm)
    trans_jaccs = np.array([p["metrics_n100"]["jaccard"] for p in pairs
                            if p["class"] == "trans"])
    ctl_jaccs = np.array([p["metrics_n100"]["jaccard"] for p in pairs
                          if p["class"] == "ctl"])
    if trans_jaccs.size:
        ax3.plot(layers, trans_jaccs.mean(axis=0),
                 label="translations (mean)", color="tab:blue", linewidth=2)
        ax3.fill_between(layers,
                         trans_jaccs.min(axis=0), trans_jaccs.max(axis=0),
                         color="tab:blue", alpha=0.15, label="trans min-max")
    if ctl_jaccs.size:
        ax3.plot(layers, ctl_jaccs.mean(axis=0),
                 label="controls (mean)", color="tab:gray",
                 linewidth=2, linestyle="--")
        ax3.fill_between(layers,
                         ctl_jaccs.min(axis=0), ctl_jaccs.max(axis=0),
                         color="tab:gray", alpha=0.15, label="ctl min-max")
    ax3.set_title("Plot 3 — disconfirm: translation mean vs control mean")
    ax3.set_xlabel("layer")
    ax3.set_ylabel(f"Jaccard (N={HEADLINE_N})")
    ax3.set_ylim(0, 1)
    ax3.grid(alpha=0.3); ax3.legend(fontsize=8, loc="best")

    # Plot 4: cutoff sensitivity (translation Jaccard mean)
    if trans_jaccs.size:
        for N, color in [(50, "tab:orange"), (100, "tab:blue"),
                         (200, "tab:green")]:
            arrs = np.array([p[f"metrics_n{N}"]["jaccard"] for p in pairs
                             if p["class"] == "trans"])
            ax4.plot(layers, arrs.mean(axis=0), label=f"N={N}",
                     color=color, linewidth=2)
    ax4.set_title("Plot 4 — cutoff sensitivity (translation Jaccard mean)")
    ax4.set_xlabel("layer")
    ax4.set_ylabel("Jaccard (translation mean)")
    ax4.set_ylim(0, 1)
    ax4.grid(alpha=0.3); ax4.legend(fontsize=8, loc="best")

    # Per-item suptitle (manifest item #1)
    title_full = (
        f"View 6 — Relational neighborhood dynamics — '{source}'"
        + (f"  (control: '{control_source}')" if control_source else "")
        + (f"  {title_suffix}" if title_suffix else "")
    )
    fig.suptitle(title_full, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    data_v5 = repo / "data" / "processed" / "view5"
    data_v6 = repo / "data" / "processed" / "view6"
    out_v5 = repo / "figures" / "view5"
    out_v6 = repo / "figures" / "view6"
    out_v5.mkdir(parents=True, exist_ok=True)
    out_v6.mkdir(parents=True, exist_ok=True)

    # ---- View 5 ----
    # Canonical 6-item set: water / brother / stone / fear / dog / bread.
    # brother has a supplementary synrerun JSON; we use the no-synonym
    # brother_chair.json for the canonical figure, matching the prior
    # committed PNG.
    v5_canonical = [
        ("water",   "view5_water_chair.json"),
        ("brother", "view5_brother_chair.json"),
        ("stone",   "view5_stone_chair.json"),
        ("fear",    "view5_fear_chair.json"),
        ("dog",     "view5_dog_chair.json"),
        ("bread",   "view5_bread_chair.json"),
    ]
    print("=== View 5 ===")
    for src, fn in v5_canonical:
        src_path = data_v5 / fn
        out_path = out_v5 / f"view5_{src}_chair.png"
        if not src_path.exists():
            print(f"  SKIP {src}: missing {src_path}")
            continue
        render_v5(json.loads(src_path.read_text(encoding="utf-8")), out_path)
        print(f"  WROTE {out_path.relative_to(repo)}")

    # ---- View 6 ----
    # All 18 = 6 sources x 3 controls.
    sources = ["water", "brother", "stone", "fear", "dog", "bread"]
    controls = ["chair", "window", "bird"]
    print("=== View 6 ===")
    for src in sources:
        for ctl in controls:
            fn = f"view6_{src}_{ctl}.json"
            src_path = data_v6 / fn
            out_path = out_v6 / f"view6_{src}_{ctl}.png"
            if not src_path.exists():
                print(f"  SKIP {src}/{ctl}: missing {src_path}")
                continue
            render_v6(json.loads(src_path.read_text(encoding="utf-8")),
                      out_path)
            print(f"  WROTE {out_path.relative_to(repo)}")

    # ---- View 6 bread clean (Brot + pão only) ----
    print("=== View 6 bread clean subset ===")
    bread_chair = data_v6 / "view6_bread_chair.json"
    if bread_chair.exists():
        j = json.loads(bread_chair.read_text(encoding="utf-8"))
        # Keep translations whose comparison word is NOT a homograph.
        keep = {"Brot", "pão"}
        render_v6(j, out_v6 / "view6_bread_chair_clean.png",
                  title_suffix="[clean subset: Brot, pão]",
                  trans_filter=lambda p: p.get("cmp_word") in keep)
        print(f"  WROTE figures/view6/view6_bread_chair_clean.png")
    else:
        print("  SKIP bread_chair_clean: source JSON missing")


if __name__ == "__main__":
    main()
