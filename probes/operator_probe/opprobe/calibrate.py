"""Step 3: window calibration on the pilot sample only.

For each rung w of the pre-registered ladder, the pilot instances and
the baseline positions are each run with a w-token left context. The
pilot state at each rung is centered on the baseline mean at the same
rung and layer, then compared with the centered state at the largest
rung. The plateau rule is applied exactly as pre-registered.
"""

from __future__ import annotations

import numpy as np


def stability(centered: dict[int, np.ndarray], measure: str) -> dict[int, np.ndarray]:
    """centered[w]: [n, 27, d]. Returns {w: [n, 27]} comparing each rung
    with the largest rung."""
    wmax = max(centered)
    ref = centered[wmax].astype(np.float64)
    out = {}
    for w, x in centered.items():
        x = x.astype(np.float64)
        if measure == "cosine_to_max":
            num = (x * ref).sum(-1)
            den = np.linalg.norm(x, axis=-1) * np.linalg.norm(ref, axis=-1)
            out[w] = num / np.maximum(den, 1e-12)
        elif measure == "relative_l2_to_max":
            # 1 - ||x - ref|| / ||ref||, so 1 means identical, like cosine.
            out[w] = 1.0 - np.linalg.norm(x - ref, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-12)
        else:
            raise ValueError(f"unknown stability measure {measure!r}")
    return out


def aggregate(stab: dict[int, np.ndarray], how: str) -> dict[int, np.ndarray]:
    f = {"median": np.median, "mean": np.mean, "min": np.min}[how]
    return {w: f(s, axis=0) for w, s in stab.items()}


def plateau(agg: dict[int, np.ndarray], layers: list[int], threshold: float) -> int | None:
    """Smallest rung w such that for every listed layer the aggregate
    stability is >= threshold at w and at every larger rung."""
    rungs = sorted(agg)
    ok = [bool(np.all(agg[w][layers] >= threshold)) for w in rungs]
    for i, w in enumerate(rungs):
        if all(ok[i:]):
            return w
    return None


def calibrate(pilot_states: dict[int, np.ndarray], baseline_means: dict[int, np.ndarray],
              cfg: dict) -> dict:
    c = cfg["calibration"]
    centered = {w: pilot_states[w] - baseline_means[w][None] for w in pilot_states}
    stab = stability(centered, c["stability_measure"])
    agg = aggregate(stab, c["aggregate_over_instances"])
    layers = list(c["plateau_layers"])
    w_star = plateau(agg, layers, float(c["plateau_threshold"]))
    return {
        "ladder": sorted(pilot_states),
        "stability_measure": c["stability_measure"],
        "aggregate_over_instances": c["aggregate_over_instances"],
        "plateau_threshold": c["plateau_threshold"],
        "plateau_layers": layers,
        "fixed_window": w_star,
        "aggregate_stability": {int(w): a.tolist() for w, a in agg.items()},
        "per_instance_stability": {int(w): s.tolist() for w, s in stab.items()},
    }
