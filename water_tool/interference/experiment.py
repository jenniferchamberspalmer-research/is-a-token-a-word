"""View 1 interference experiment with magnitude-matched null.

Two measurements:

  INTERNAL β       interference among the cross-linguistic features themselves
  CROSS-FIELD β    interference the cross-linguistic features receive from
                   the broader feature population

Cross-field is reported against TWO nulls (the gap between them is itself
informative — it isolates how much of any observed dip is magnitude
geometry vs. cross-linguistic structure):

  uniform null            random k-feature subsets from the population
  magnitude-matched null  random k-feature subsets whose per-feature
                          magnitude-bin histogram matches the selected
                          set's, with selected features excluded

The matched null is the load-bearing baseline: cross-linguistic
features are selected for top-quantile activation in EVERY language, so
they are magnitude-biased. Without matching, a low β−null could be a
fact about feature magnitude geometry rather than about cross-linguistic
structure.

GUARDS ON THE MATCHED NULL
==========================

widening_rate
    Fraction of per-feature draws that fell back to a neighbor bin
    after the same-bin pool was exhausted by exclusion. >0.05 warns
    that n_bins is too fine for this magnitude distribution and should
    be lowered.

histogram assertion
    Hard-asserts the drawn null's bin histogram is within `tol`
    (default 0.02 max abs deviation) of the selected set's histogram.
    Fails the run on violation — a few features legitimately pushed to
    a neighbor bin must pass; structural drift toward uniform must not.

Sequencing note (carried from SESSION_NOTES.md addendum, commit
1f90fcd): the activation-matched null is the FIRST of two load-bearing
baseline fixes; the control-concept test is the second. Until the null
is magnitude-matched, a function-word control could fail to dip for
magnitude reasons rather than semantic ones, so control concepts test
nothing interpretable until this baseline is in place. The control-
concept test is held for a later phase.
"""

from __future__ import annotations

import numpy as np

from .superposition import SuperpositionField


# ----------------------------------------------------------- math

def _pool_stats(U, targets, pool):
    """Per-target mean (mu) and variance (sig2) of cosine against the
    distractor pool, excluding self-matches. U rows are unit vectors."""
    targets = np.asarray(targets)
    pool = np.asarray(pool)
    C = U[targets] @ U[pool].T
    self_mask = pool[None, :] == targets[:, None]
    C = np.where(self_mask, np.nan, C)
    mu = np.nanmean(C, axis=1)
    sig2 = np.nanvar(C, axis=1)
    return mu, sig2


def _rms_curve(mu, sig2, loads):
    """Closed-form RMS signed interference per feature at each load L.

    RMS^2(L) = mean_i[ (L-1)^2 * mu_i^2 + (L-1) * sig2_i ]
    """
    Ld = np.asarray(loads, dtype=float) - 1.0
    term = (Ld[:, None] ** 2) * (mu[None, :] ** 2) + Ld[:, None] * sig2[None, :]
    return np.sqrt(np.mean(term, axis=1))


def _fit_beta(loads, rms):
    rms = np.asarray(rms)
    loads = np.asarray(loads, dtype=float)
    ok = rms > 0
    if ok.sum() < 2:
        return np.nan
    x, y = np.log(loads[ok]), np.log(rms[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    beta = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return float(beta)


def targeted_beta(U, targets, pool, loads):
    """Analytic β for a target set against a distractor pool."""
    mu, sig2 = _pool_stats(U, targets, pool)
    return _fit_beta(loads, _rms_curve(mu, sig2, loads))


def bootstrap_beta(U, targets, pool, loads, n_boot=400, seed=0):
    """Bootstrap β by resampling targets with replacement."""
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    mu, sig2 = _pool_stats(U, targets, pool)
    k = len(targets)
    betas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, size=k)
        betas[b] = _fit_beta(loads, _rms_curve(mu[idx], sig2[idx], loads))
    betas = betas[np.isfinite(betas)]
    return dict(
        beta=targeted_beta(U, targets, pool, loads),
        lo=float(np.percentile(betas, 2.5)),
        hi=float(np.percentile(betas, 97.5)),
        boot=betas,
    )


def null_beta(U, k, pool, loads, candidate_targets=None,
              n_draws=400, seed=1):
    """Uniform null: random k-feature targets, fixed distractor pool."""
    rng = np.random.default_rng(seed)
    cand = (np.arange(U.shape[0]) if candidate_targets is None
            else np.asarray(candidate_targets))
    betas = np.empty(n_draws)
    for d in range(n_draws):
        t = rng.choice(cand, size=k, replace=False)
        betas[d] = targeted_beta(U, t, pool, loads)
    return betas[np.isfinite(betas)]


# ----------------------------------------- magnitude-matched null

def magnitude_matched_null_beta(U, selected, mag, pool, loads,
                                n_draws=400, n_bins=20, tol=0.02,
                                seed=2):
    """Draw null target sets whose per-feature magnitude-bin histogram
    matches the selected set's, with selected features excluded.

    Procedure:

      1. Bin all features by percentile of `mag` into `n_bins` equal-
         quantile bins (over the FULL population).
      2. For each draw, and for each selected feature s: draw ONE
         feature from s's bin (excluding any selected feature and,
         where possible, any feature already drawn in this draw). If
         s's bin is empty after exclusion, widen to the nearest non-
         empty bin and log the event.
      3. Compute β with that drawn set as targets against `pool`.

    Returns:
        dict(
            betas          (n_draws,) array of β values
            widening_rate  fraction of per-feature draws that widened
            mean, sd       summary of betas
            n_bins         the bin count used
            max_hist_dev   max |sel_hist[b] - null_hist[b]| across bins
        )

    HARD ASSERTS  max_hist_dev <= tol (default 0.02). On violation the
    run fails with a message that includes the deviation and the
    widening_rate — to distinguish legitimate widening (a few features
    pushed to a neighbor bin) from structural drift toward uniform.
    """
    rng = np.random.default_rng(seed)
    selected = np.asarray(selected, dtype=int)
    n_features = U.shape[0]
    mag = np.asarray(mag, dtype=float)
    if mag.shape[0] != n_features:
        raise ValueError(
            f"mag length ({mag.shape[0]}) must equal n_features "
            f"({n_features})")
    if len(selected) == 0:
        raise ValueError("selected must be non-empty")

    # Equal-quantile binning over all features.
    edges = np.quantile(mag, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    bin_of = np.digitize(mag, edges[1:-1], right=True).astype(int)
    bin_of = np.clip(bin_of, 0, n_bins - 1)
    sel_hist = (np.bincount(bin_of[selected], minlength=n_bins)
                / len(selected))

    excluded = set(int(i) for i in selected)
    bin_pool = {b: [] for b in range(n_bins)}
    for i, b in enumerate(bin_of):
        if i not in excluded:
            bin_pool[int(b)].append(i)
    for b in range(n_bins):
        bin_pool[b] = np.array(bin_pool[b], dtype=int)

    def widen_search(target_bin: int):
        if len(bin_pool[target_bin]) > 0:
            return bin_pool[target_bin], False
        for offset in range(1, n_bins):
            for direction in (-1, 1):
                b = target_bin + direction * offset
                if 0 <= b < n_bins and len(bin_pool[b]) > 0:
                    return bin_pool[b], True
        raise RuntimeError(
            "no non-empty bins remain; cannot construct magnitude-matched null")

    betas = np.empty(n_draws)
    widen_count = 0
    total_per_feat_draws = 0
    all_drawn_bins = []

    for d in range(n_draws):
        drawn = []
        used = set()
        for s in selected:
            target_bin = int(bin_of[s])
            total_per_feat_draws += 1
            pool_arr, widened = widen_search(target_bin)
            if widened:
                widen_count += 1
            candidates = [int(x) for x in pool_arr if int(x) not in used]
            if not candidates:
                candidates = [int(x) for x in pool_arr]
            pick = int(rng.choice(candidates))
            drawn.append(pick)
            used.add(pick)
        drawn_arr = np.array(drawn, dtype=int)
        all_drawn_bins.append(bin_of[drawn_arr])
        betas[d] = targeted_beta(U, drawn_arr, pool, loads)

    widening_rate = float(widen_count / max(total_per_feat_draws, 1))

    all_drawn_bins_flat = np.concatenate(all_drawn_bins)
    null_hist = (np.bincount(all_drawn_bins_flat, minlength=n_bins)
                 / len(all_drawn_bins_flat))
    max_dev = float(np.max(np.abs(sel_hist - null_hist)))
    if max_dev > tol:
        raise AssertionError(
            f"magnitude-matched null bin histogram drifted from selected "
            f"set (max deviation {max_dev:.3f} > tol {tol:.3f}); "
            f"widening_rate={widening_rate:.3f} — lower n_bins or "
            f"check the input magnitude distribution")

    betas = betas[np.isfinite(betas)]
    return dict(
        betas=betas,
        widening_rate=widening_rate,
        mean=float(np.mean(betas)),
        sd=float(np.std(betas) + 1e-12),
        n_bins=int(n_bins),
        max_hist_dev=max_dev,
    )


# ----------------------------------------------------- inference helpers

def _z_and_p(observed, null):
    mu, sd = float(np.mean(null)), float(np.std(null) + 1e-12)
    z = (observed - mu) / sd
    p = float((np.sum(np.abs(null - mu) >= abs(observed - mu)) + 1)
              / (len(null) + 1))
    return z, p, mu, sd


def select_crosslingual_features(acts_by_lang: dict,
                                 min_languages: int = None,
                                 quantile: float = 0.9):
    """Identify features active in the top (1-quantile) across `min_languages`
    languages.

    acts_by_lang : {lang: vector(n_features)} of per-feature activation
                   for the concept in that language.
    Returns: integer index array of selected features.
    """
    langs = list(acts_by_lang)
    n_features = len(next(iter(acts_by_lang.values())))
    if min_languages is None:
        min_languages = len(langs)
    active = np.zeros(n_features, dtype=int)
    for lang in langs:
        v = np.asarray(acts_by_lang[lang], dtype=float)
        thr = np.quantile(v, quantile)
        active += (v >= thr).astype(int)
    return np.where(active >= min_languages)[0]


# ------------------------------------------------------- entry point

def view1_experiment(field: SuperpositionField, xling_indices,
                     activation_magnitude=None,
                     pool_size: int = 4000, n_boot: int = 400,
                     n_null: int = 400, loads=None, seed: int = 0,
                     report: bool = True,
                     n_bins: int = 20, hist_tol: float = 0.02):
    """Run internal and cross-field interference scaling.

    activation_magnitude : (n_features,) per-feature magnitude metric
        used for the magnitude-matched null. When None, only the
        uniform null is reported. When provided, BOTH nulls are
        reported and the headline becomes cross_beta - null_matched_mean.
    """
    rng = np.random.default_rng(seed)
    U = field.U
    n = field.n_features
    xling = np.asarray(xling_indices, dtype=int)
    k = len(xling)
    if k < 3:
        raise ValueError("need at least 3 cross-linguistic features")

    pool_size = min(pool_size, n)
    full_pool = rng.choice(n, size=pool_size, replace=False)

    if loads is None:
        loads_cross = np.unique(np.round(
            np.geomspace(2, min(pool_size, 1024), 14)).astype(int))
        loads_int = np.unique(np.round(
            np.geomspace(2, max(k, 3), min(10, k))).astype(int))
    else:
        loads_cross = loads_int = np.asarray(loads)

    # CROSS-FIELD
    cross = bootstrap_beta(U, xling, full_pool, loads_cross, n_boot, seed)
    cross_null_unif = null_beta(
        U, k, full_pool, loads_cross,
        candidate_targets=None, n_draws=n_null, seed=seed + 7,
    )
    cz_u, cp_u, cmu_u, csd_u = _z_and_p(cross["beta"], cross_null_unif)

    matched = None
    cz_m = cp_m = cmu_m = csd_m = None
    if activation_magnitude is not None:
        matched = magnitude_matched_null_beta(
            U, xling, activation_magnitude, full_pool, loads_cross,
            n_draws=n_null, n_bins=n_bins, tol=hist_tol, seed=seed + 17,
        )
        cz_m, cp_m, cmu_m, csd_m = _z_and_p(cross["beta"], matched["betas"])

    # INTERNAL
    intern = bootstrap_beta(U, xling, xling, loads_int, n_boot, seed)
    int_null = np.empty(n_null)
    for d in range(n_null):
        t = rng.choice(n, size=k, replace=False)
        int_null[d] = targeted_beta(U, t, t, loads_int)
    int_null = int_null[np.isfinite(int_null)]
    iz, ip, imu, isd = _z_and_p(intern["beta"], int_null)

    cross_out = dict(
        beta=cross["beta"], ci=(cross["lo"], cross["hi"]),
        null_unif=cross_null_unif,
        null_unif_mean=cmu_u, null_unif_sd=csd_u,
        z_unif=cz_u, p_unif=cp_u,
        null_matched=(matched["betas"] if matched else None),
        null_matched_mean=cmu_m,
        null_matched_sd=csd_m,
        z_matched=cz_m, p_matched=cp_m,
        widening_rate=(matched["widening_rate"] if matched else None),
        max_hist_dev=(matched["max_hist_dev"] if matched else None),
        n_bins=(matched["n_bins"] if matched else None),
    )

    res = dict(
        k=k,
        cross=cross_out,
        internal=dict(
            beta=intern["beta"], ci=(intern["lo"], intern["hi"]),
            null_mean=imu, null_sd=isd, z=iz, p=ip,
            null=int_null, boot=intern["boot"],
        ),
    )

    if report:
        print(f"View 1 interference   field='{field.name}'  "
              f"d_model={field.d_model}  n_features={n}  k_xling={k}")
        print("-" * 70)
        ir = res["internal"]
        print("INTERNAL  (cluster vs itself)")
        print(f"    beta = {ir['beta']:.3f}  "
              f"CI [{ir['ci'][0]:.3f}, {ir['ci'][1]:.3f}]")
        print(f"    null = {ir['null_mean']:.3f} +/- {ir['null_sd']:.3f}"
              f"    z = {ir['z']:+.2f}  p = {ir['p']:.4f}")
        cr = res["cross"]
        print("CROSS-FIELD (cluster vs population)")
        print(f"    beta = {cr['beta']:.3f}  "
              f"CI [{cr['ci'][0]:.3f}, {cr['ci'][1]:.3f}]")
        print(f"    uniform null = {cr['null_unif_mean']:.3f} "
              f"+/- {cr['null_unif_sd']:.3f}    z = {cr['z_unif']:+.2f}")
        if matched is not None:
            print(f"    matched null = {cr['null_matched_mean']:.3f} "
                  f"+/- {cr['null_matched_sd']:.3f}    "
                  f"z = {cr['z_matched']:+.2f}")
            print(f"    widening_rate = {cr['widening_rate']:.3f}    "
                  f"max_hist_dev = {cr['max_hist_dev']:.3f}")
            print(f"    HEADLINE  cross_beta - null_matched_mean = "
                  f"{cr['beta'] - cr['null_matched_mean']:+.3f}")
        else:
            print(f"    HEADLINE  cross_beta - null_unif_mean = "
                  f"{cr['beta'] - cr['null_unif_mean']:+.3f}")
    return res
