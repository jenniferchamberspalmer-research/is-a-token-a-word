"""Adversarial validation: the magnitude-matched null must absorb a
high-magnitude planted structure that the uniform null does not.

Builds a synthetic field where ~10% of features share a common partial
direction AND have higher magnitudes than the rest (the high-magnitude
subset is geometrically distinct). The "cluster" is k random features
drawn from that high-magnitude subset — it has no extra cross-linguistic
structure beyond living in the high-mag subspace.

Expected result:
  - cross_beta ≠ uniform null mean (uniform draws from the whole
    population, mostly low-mag features → its beta is the average
    geometry, not the high-mag-subset geometry)
  - cross_beta ≈ matched null mean (matched null draws from the same
    high-mag region → measures the same high-mag-subset geometry)
  → |cross_beta - null_matched_mean|  <  0.5 * |cross_beta - null_unif_mean|

If this passes, the matched null is doing what we said it does: it
absorbs the magnitude-driven bias that the uniform null does not.

Run:
    python -m water_tool.interference.validation
"""

from __future__ import annotations

import numpy as np

from .experiment import view1_experiment
from .superposition import SuperpositionField


def planted_magnitude_field(n_features: int = 4096, d_model: int = 256,
                            k: int = 24, high_frac: float = 0.10,
                            shared: float = 0.40, mag_scale: float = 3.0,
                            seed: int = 0):
    """Build the adversarial field.

    Returns (field, mag, cluster) where:
      - field has n_features unit directions in d_model dims,
      - the top `high_frac` of features (by magnitude) share a common
        partial direction (`shared` mixing weight),
      - the cluster is k random features drawn from that high-mag subset.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_features, d_model))

    n_high = int(n_features * high_frac)
    high_idx_orig = np.arange(n_high)

    shared_dir = rng.standard_normal((1, d_model))
    base[high_idx_orig] = ((1.0 - shared) * base[high_idx_orig]
                           + shared * shared_dir)
    base[high_idx_orig] *= mag_scale

    perm = rng.permutation(n_features)
    D = base[perm]

    high_idx_after = np.where(np.isin(perm, high_idx_orig))[0]
    cluster = rng.choice(high_idx_after, size=k, replace=False)
    mag = np.linalg.norm(D, axis=1)

    return (SuperpositionField(D, name="planted high-mag (not cross-lingual)"),
            mag, cluster)


def main() -> int:
    print("Adversarial validation: magnitude-matched null should absorb "
          "a high-magnitude planted structure.\n")

    field, mag, cluster = planted_magnitude_field()

    others = np.setdiff1d(np.arange(field.n_features), cluster)
    print(f"  n_features = {field.n_features}")
    print(f"  d_model    = {field.d_model}")
    print(f"  cluster k  = {len(cluster)}")
    print(f"  cluster mean |mag| = {mag[cluster].mean():.3f}")
    print(f"  others  mean |mag| = {mag[others].mean():.3f}\n")

    res = view1_experiment(
        field, cluster, activation_magnitude=mag,
        n_boot=300, n_null=300, report=True,
    )

    cr = res["cross"]
    headline_matched = cr["beta"] - cr["null_matched_mean"]
    headline_uniform = cr["beta"] - cr["null_unif_mean"]

    print()
    print(f"cross_beta - null_matched_mean = {headline_matched:+.4f}")
    print(f"cross_beta - null_unif_mean    = {headline_uniform:+.4f}")
    print()

    if abs(headline_matched) < 0.5 * abs(headline_uniform):
        print("PASS: matched null absorbed the high-magnitude structure "
              "more than the uniform null did.")
        return 0
    print("FAIL: matched null did not absorb the high-magnitude structure "
          "as expected. Inspect widening_rate and the bin histogram.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
