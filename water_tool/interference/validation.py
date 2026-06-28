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


def sparse_selector_check(n_features: int = 16384,
                          d_model_for_check: int = 256,
                          languages: int = 6,
                          n_active_per_lang: int = 80,
                          n_overlap: int = 24,
                          quantile: float = 0.5,
                          seed: int = 0) -> tuple[int, int]:
    """Mimic a JumpReLU SAE activation: most features zero, ~80 fire
    per language, with `n_overlap` features firing strongly in every
    language. Returns (k_selected, n_overlap_recovered).

    Pre-fix bug: selector returned every feature because
    np.quantile(v, 0.95) returned 0 when 99% of v was zero, so v >= 0
    matched everything. Post-fix: threshold is taken over POSITIVE
    activations only.
    """
    from .experiment import select_crosslingual_features

    rng = np.random.default_rng(seed)
    overlap_indices = rng.choice(n_features, size=n_overlap, replace=False)
    overlap_set = set(int(i) for i in overlap_indices)

    acts_by_lang = {}
    for lang_i in range(languages):
        v = np.zeros(n_features)
        # Core cross-linguistic features fire strongly in every language.
        v[overlap_indices] = rng.uniform(5.0, 10.0, size=n_overlap)
        # Some language-specific extras fire at a lower intensity.
        non_overlap = np.setdiff1d(np.arange(n_features), overlap_indices)
        extras = rng.choice(non_overlap,
                            size=n_active_per_lang - n_overlap,
                            replace=False)
        v[extras] = rng.uniform(0.5, 3.0,
                                size=n_active_per_lang - n_overlap)
        acts_by_lang[f"lang{lang_i}"] = v

    selected = select_crosslingual_features(
        acts_by_lang, min_languages=languages, quantile=quantile)
    recovered = sum(1 for s in selected if int(s) in overlap_set)
    return int(len(selected)), int(recovered)


def main() -> int:
    print("Adversarial validation #1: magnitude-matched null should "
          "absorb a high-magnitude planted structure.\n")

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

    adversarial_pass = abs(headline_matched) < 0.5 * abs(headline_uniform)
    if adversarial_pass:
        print("PASS #1: matched null absorbed the high-magnitude structure "
              "more than the uniform null did.\n")
    else:
        print("FAIL #1: matched null did not absorb the high-magnitude "
              "structure as expected.\n")

    print("Sparse-selector validation #2: simulate JumpReLU sparsity "
          "(most features 0; ~80 fire per language, 24 overlap).")
    k_sel, n_recovered = sparse_selector_check()
    print(f"  selected k = {k_sel}  (planted overlap = 24, "
          f"of which recovered = {n_recovered})")
    # PASS criteria: selector returns a small k (tens, not 16k) and
    # recovers most of the planted overlap.
    sparse_pass = (k_sel < 200) and (n_recovered >= 20)
    if sparse_pass:
        print("PASS #2: selector returns a small k on sparse activations "
              "and recovers the planted cross-linguistic overlap.\n")
    else:
        print(f"FAIL #2: selector returned k={k_sel} (expected tens), "
              f"recovered={n_recovered}/24.\n")

    if adversarial_pass and sparse_pass:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
