"""Step 5: unlabeled readouts.

Participation ratio PR = (sum lambda)^2 / sum(lambda^2) over the
eigenvalues of an n x d cloud's d x d matrix M. Since tr(M) and
||M||_F are shared with the n x n Gram matrix, PR = tr(G)^2 / ||G||_F^2
with no eigendecomposition.

Two matrices are available (config readouts.pr_matrix):
  covariance     rows minus the cloud's own mean. Subtracting a baseline
                 first changes nothing, so raw and centered agree exactly;
                 both are still written, and the identity is checked.
  second_moment  rows minus a fixed point c, no further mean removal:
                 c = 0 for raw, c = baseline mean for centered.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _pr_from_gram(G: np.ndarray, center: bool) -> float:
    if center:
        r = G.mean(axis=1)
        G = G - r[:, None] - r[None, :] + G.mean()
    tr = np.trace(G)
    fro2 = np.square(G).sum()
    return float(tr * tr / fro2) if fro2 > 0 else float("nan")


def participation_ratio(X: np.ndarray, c: np.ndarray | None, matrix: str) -> float:
    Z = X.astype(np.float64) - (0.0 if c is None else c.astype(np.float64))
    return _pr_from_gram(Z @ Z.T, center=(matrix == "covariance"))


def saturation_curve(X: np.ndarray, c: np.ndarray | None, matrix: str,
                     order: np.ndarray, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """PR of the first k instances (in `order`) for k = 2, 2+step, ..., n."""
    Z = X[order].astype(np.float64) - (0.0 if c is None else c.astype(np.float64))
    G = Z @ Z.T
    ks = np.arange(2, len(order) + 1, step)
    if ks[-1] != len(order):
        ks = np.append(ks, len(order))
    return ks, np.array([_pr_from_gram(G[:k, :k], center=(matrix == "covariance")) for k in ks])


def matrices(cfg: dict) -> list[str]:
    m = cfg["readouts"]["pr_matrix"]
    return ["covariance", "second_moment"] if m == "both" else [m]


def compute_all(states: np.ndarray, baseline_means: np.ndarray, cfg: dict,
                order: np.ndarray) -> dict:
    """states [n, 27, d]; baseline_means [27, d]."""
    n_layers = states.shape[1]
    step = int(cfg["readouts"]["saturation_step"])
    pr_rows, sat = [], {}
    for m in matrices(cfg):
        for layer in range(n_layers):
            X = states[:, layer]
            raw = participation_ratio(X, None, m)
            cen = participation_ratio(X, baseline_means[layer], m)
            if m == "covariance":
                assert np.isclose(raw, cen, rtol=1e-6), "covariance PR must ignore a constant offset"
            pr_rows.append({"matrix": m, "hidden_state": layer, "pr_raw": raw, "pr_centered": cen})
            for kind, c in (("raw", None), ("centered", baseline_means[layer])):
                ks, prs = saturation_curve(X, c, m, order, step)
                sat[(m, kind, layer)] = (ks, prs)
    return {"pr": pr_rows, "saturation": sat}


def write(out_dir: Path, label: str, res: dict, entropy: np.ndarray, caption: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{label}_participation_ratio.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["matrix", "hidden_state", "pr_raw", "pr_centered"])
        w.writeheader(); w.writerows(res["pr"])
    with open(out_dir / f"{label}_saturation.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["matrix", "states", "hidden_state", "k", "pr"])
        for (m, kind, layer), (ks, prs) in sorted(res["saturation"].items()):
            for k, p in zip(ks, prs):
                w.writerow([m, kind, layer, int(k), float(p)])
    with open(out_dir / f"{label}_entropy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["instance_index", "next_token_entropy_nats"])
        for i, e in enumerate(entropy):
            w.writerow([i, float(e)])

    ms = sorted({r["matrix"] for r in res["pr"]})
    fig, axes = plt.subplots(1, len(ms), figsize=(6 * len(ms), 4), squeeze=False)
    for ax, m in zip(axes[0], ms):
        rows = [r for r in res["pr"] if r["matrix"] == m]
        ax.plot([r["hidden_state"] for r in rows], [r["pr_raw"] for r in rows], "o-", label="raw")
        ax.plot([r["hidden_state"] for r in rows], [r["pr_centered"] for r in rows], "s--", label="baseline-centered")
        ax.set_xlabel("hidden-state index (0 = embedding output)"); ax.set_ylabel("participation ratio")
        ax.set_title(f"{label}: PR per layer ({m})"); ax.legend()
    fig.text(0.01, 0.01, caption, fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out_dir / f"{label}_participation_ratio.png", dpi=150); plt.close(fig)

    for m in ms:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        for ax, kind in zip(axes, ("raw", "centered")):
            for layer in range(max(r["hidden_state"] for r in res["pr"]) + 1):
                ks, prs = res["saturation"][(m, kind, layer)]
                ax.plot(ks, prs, color=plt.cm.viridis(layer / 26), lw=0.8)
            ax.set_xlabel("instances included"); ax.set_ylabel("participation ratio")
            ax.set_title(f"{label}: saturation ({m}, {kind}); colour = hidden-state index 0..26")
        fig.text(0.01, 0.01, caption, fontsize=7, wrap=True)
        fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out_dir / f"{label}_saturation_{m}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(entropy, bins=40)
    ax.set_xlabel("next-token entropy at target position (nats)"); ax.set_ylabel("instances")
    ax.set_title(f"{label}: next-token entropy")
    fig.text(0.01, 0.01, caption, fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(out_dir / f"{label}_entropy.png", dpi=150); plt.close(fig)
