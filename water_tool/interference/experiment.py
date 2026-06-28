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
    """Analytic β for a single target set against a distractor pool."""
    mu, sig2 = _pool_stats(U, targets, pool)
    return _fit_beta(loads, _rms_curve(mu, sig2, loads))


# -------- batched-over-draws fast path (used by the null phases) --------
#
# Bottleneck at real 16k-feature, 2304-d_model scale: each targeted_beta
# call re-copies U[pool] (a 73 MB tensor) before the matmul. With
# n_null=400 calls per null phase, that's 30+ GB of memory operations
# per phase. The batch helpers below extract U[pool] ONCE, stack all
# n_draws target sets, and run a single (n_draws, k, d_model) @
# (d_model, pool_size) matmul — one BLAS call replacing n_draws.

def _pool_stats_batch(U, targets_batch, pool_indices):
    """Batch version of _pool_stats.

    targets_batch : (n_draws, k) integer indices into U.
    pool_indices  : (pool_size,) integer indices into U.

    Returns mu (n_draws, k), sig2 (n_draws, k).

    Peak memory: ~ n_draws * k * pool_size * 8 bytes for C, plus
    n_draws * k * d_model * 8 for the extracted target tensor. For
    n_draws=400, k=24, pool_size=4000, d_model=2304: ~480 MB peak.
    """
    U_pool = U[pool_indices]                       # (pool_size, d_model)
    U_targets = U[targets_batch]                   # (n_draws, k, d_model)
    C = U_targets @ U_pool.T                       # (n_draws, k, pool_size)
    # Mask out self-matches (where pool index equals target index).
    self_mask = (pool_indices[None, None, :] ==
                 targets_batch[:, :, None])        # (n_draws, k, pool_size)
    C = np.where(self_mask, np.nan, C)
    mu = np.nanmean(C, axis=2)                     # (n_draws, k)
    sig2 = np.nanvar(C, axis=2)                    # (n_draws, k)
    return mu, sig2


def _rms_curve_batch(mu, sig2, loads):
    """Vectorized RMS curve across draws.

    mu, sig2 : (n_draws, k)
    loads    : (n_loads,)

    Returns rms (n_draws, n_loads).
    """
    Ld = np.asarray(loads, dtype=float) - 1.0     # (n_loads,)
    # term shape (n_draws, n_loads, k): Ld^2 * mu^2 + Ld * sig2
    term = ((Ld[None, :, None] ** 2) * (mu[:, None, :] ** 2)
            + Ld[None, :, None] * sig2[:, None, :])
    return np.sqrt(np.mean(term, axis=2))         # (n_draws, n_loads)


def _fit_beta_batch(loads, rms_batch):
    """Vectorized log-log slope per row.

    rms_batch : (n_draws, n_loads)
    Returns   : (n_draws,) array of β values; rows with all-zero rms
                map to NaN.
    """
    x = np.log(np.asarray(loads, dtype=float))    # (n_loads,)
    rms_batch = np.asarray(rms_batch, dtype=float)
    # Replace zeros with NaN so they drop out of the regression.
    bad = rms_batch <= 0
    y = np.where(bad, np.nan, np.log(np.where(bad, 1.0, rms_batch)))
    x_centered = x - x.mean()
    x_var = float((x_centered ** 2).sum()) or 1.0
    y_mean = np.nanmean(y, axis=1, keepdims=True)
    y_centered = y - y_mean
    num = np.nansum(y_centered * x_centered[None, :], axis=1)
    return num / x_var


def targeted_beta_batch(U, targets_batch, pool_indices, loads,
                         chunk_size: int = 50):
    """Batched β over rows of `targets_batch`, chunked to bound memory.

    Peak memory per chunk ≈ chunk_size * k * (pool_size + d_model) * 8
    bytes. With defaults (chunk_size=50, k=24, pool_size=4000,
    d_model=2304) that's ~75 MB. A full 400-draw batch without
    chunking would have been ~600 MB at k=24, which fits, but at
    larger k it OOMs hard — see the safety check in view1_experiment.

    Reducing `chunk_size` is the lever if the container is memory-
    constrained; larger `chunk_size` is faster but uses more memory.
    """
    targets_batch = np.asarray(targets_batch, dtype=int)
    pool_indices = np.asarray(pool_indices, dtype=int)
    n_draws = targets_batch.shape[0]
    betas = np.empty(n_draws)
    if chunk_size <= 0:
        chunk_size = n_draws
    for start in range(0, n_draws, chunk_size):
        end = min(start + chunk_size, n_draws)
        mu, sig2 = _pool_stats_batch(U, targets_batch[start:end], pool_indices)
        rms = _rms_curve_batch(mu, sig2, loads)
        betas[start:end] = _fit_beta_batch(loads, rms)
    return betas


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
    """Uniform null: random k-feature targets, fixed distractor pool.
    Uses the batched matmul path."""
    rng = np.random.default_rng(seed)
    cand = (np.arange(U.shape[0]) if candidate_targets is None
            else np.asarray(candidate_targets))
    pool = np.asarray(pool, dtype=int)
    targets_batch = np.empty((n_draws, k), dtype=int)
    for d in range(n_draws):
        targets_batch[d] = rng.choice(cand, size=k, replace=False)
    betas = targeted_beta_batch(U, targets_batch, pool, loads)
    return betas[np.isfinite(betas)]


# ----------------------------------------- magnitude-matched null

def magnitude_matched_null_beta(U, selected, mag, pool, loads,
                                n_draws=400, n_bins=20, tol=0.02,
                                seed=2,
                                heartbeat_every=50,
                                timeout_s=300):
    """Draw null target sets whose per-feature magnitude-bin histogram
    matches the selected set's, with selected features excluded.

    Procedure (vectorized):
      1. Bin all features by percentile of `mag` into `n_bins` equal-
         quantile bins (over the FULL population).
      2. For each selected feature, resolve a `source_bin` — its own
         bin if non-empty after exclusion, else widen to the nearest
         non-empty bin (this is deterministic per slot, computed once
         outside the draw loop).
      3. Group target slots by source_bin.
      4. For each draw: for each source_bin, draw n_slots features
         without replacement from that bin (with replacement only if
         the bin is too small). `numpy.random.Generator.choice` is the
         workhorse — no Python-level per-feature filtering.
      5. Compute β with each drawn set as targets against `pool`.

    Returns dict with `betas`, `widening_rate`, `mean`, `sd`, `n_bins`,
    `max_hist_dev`.

    HARD-ASSERTS `max_hist_dev <= tol` (default 0.02). On violation the
    run fails with a message that includes the deviation and the
    widening_rate — to distinguish legitimate widening (a few features
    pushed to a neighbor bin) from structural drift toward uniform.

    Emits a heartbeat to stdout every `heartbeat_every` draws (default
    50) so a long run is visible in the container log. Raises
    `TimeoutError` if total elapsed exceeds `timeout_s` (default 300s).

    Performance notes:
      - Inner loop is `n_draws * n_bins_used` numpy ops, not
        `n_draws * k * pool_per_bin` Python ops. The previous design
        called `int(x)` on every numpy scalar in every per-bin pool on
        every draw, which extrapolated to ~30s on a fast host and 150+s
        on Modal's T4 host CPU. The vectorized form runs in single-
        digit seconds on a 16384-feature SAE.
      - Histogram drift max_dev is computed exactly (per-draw bin
        counts accumulated as integers), not via floating sums.
    """
    import time

    t0 = time.time()
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

    # Per-bin available index arrays (selected features excluded).
    excluded_mask = np.zeros(n_features, dtype=bool)
    excluded_mask[selected] = True
    bin_arrays = []
    for b in range(n_bins):
        in_bin = (bin_of == b) & ~excluded_mask
        bin_arrays.append(np.where(in_bin)[0])

    # Resolve source bin per selected target slot (widen if necessary).
    sel_bins = bin_of[selected]
    source_bins = np.empty(len(selected), dtype=int)
    widened_flag = np.zeros(len(selected), dtype=bool)
    for i, tb in enumerate(sel_bins):
        tb = int(tb)
        if len(bin_arrays[tb]) > 0:
            source_bins[i] = tb
            continue
        found = False
        for offset in range(1, n_bins):
            for direction in (-1, 1):
                b = tb + direction * offset
                if 0 <= b < n_bins and len(bin_arrays[b]) > 0:
                    source_bins[i] = b
                    widened_flag[i] = True
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError(
                "no non-empty bins remain; cannot construct matched null")

    # Group target slots by their source bin so we can sample one bin
    # at a time with numpy-native rng.choice.
    slots_by_bin = {}
    for slot_i, sb in enumerate(source_bins):
        slots_by_bin.setdefault(int(sb), []).append(slot_i)
    slots_by_bin = {sb: np.asarray(slots, dtype=int)
                    for sb, slots in slots_by_bin.items()}

    # Step A: build all n_draws drawn sets (vectorized per-bin sampling)
    all_drawn = np.empty((n_draws, len(selected)), dtype=int)
    for d in range(n_draws):
        if d > 0 and d % heartbeat_every == 0:
            elapsed = time.time() - t0
            print(f"  [matched-null] sampling draw {d}/{n_draws}  "
                  f"elapsed {elapsed:.1f}s", flush=True)
            if elapsed > timeout_s:
                raise TimeoutError(
                    f"magnitude_matched_null_beta exceeded timeout "
                    f"({timeout_s}s) after {d} draws (sampling phase)")
        for sb, slots in slots_by_bin.items():
            pool_arr = bin_arrays[sb]
            n_needed = len(slots)
            if n_needed <= len(pool_arr):
                picks = rng.choice(pool_arr, size=n_needed, replace=False)
            else:
                picks = rng.choice(pool_arr, size=n_needed, replace=True)
            all_drawn[d, slots] = picks

    t_sample = time.time() - t0
    print(f"  [matched-null] sampling done in {t_sample:.1f}s; "
          f"running batched β fit ...", flush=True)

    # Step B: one batched β fit across all n_draws drawn sets
    pool_arr_for_beta = np.asarray(pool, dtype=int)
    betas = targeted_beta_batch(U, all_drawn, pool_arr_for_beta, loads)

    # Tally drawn-feature bin counts for the histogram assertion.
    all_drawn_bin_counts = np.bincount(bin_of[all_drawn.ravel()],
                                       minlength=n_bins).astype(np.int64)

    elapsed = time.time() - t0
    if elapsed > timeout_s:
        raise TimeoutError(
            f"magnitude_matched_null_beta exceeded timeout "
            f"({timeout_s}s) total elapsed")
    print(f"  [matched-null] done {n_draws} draws in {elapsed:.1f}s",
          flush=True)

    # widening_rate: source_bins is deterministic per slot in this
    # vectorized form, so per-draw widening is constant. Reported as
    # the fraction of target slots that widened — equivalent to the
    # original "fraction of per-feature draws" semantics.
    widening_rate = float(widened_flag.sum() / len(selected))

    total_drawn = int(all_drawn_bin_counts.sum())
    null_hist = all_drawn_bin_counts / total_drawn
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
    """Identify features active in the top (1 - quantile) of positive
    activations across `min_languages` languages.

    acts_by_lang : {lang: vector(n_features)} of per-feature activation
                   for the concept in that language.
    Returns: integer index array of selected features.

    Sparsity handling. Gemma Scope SAEs use JumpReLU non-linearities,
    so the vast majority of feature activations on any single token
    are EXACTLY ZERO. A naive `np.quantile(v, 0.95)` over a vector
    where 99% of entries are zero returns 0, and `v >= 0` then selects
    every feature — making the whole population "cross-linguistic" and
    blowing up downstream tensor sizes.

    Fix: the threshold is computed over STRICTLY POSITIVE activations
    only (`v[v > 0]`), and a feature only counts as active in a
    language if it is BOTH (a) strictly positive AND (b) at or above
    that language-specific positive-quantile threshold. The strict-
    positive guard makes the selection robust to arbitrary zero-mass
    in the activation distribution.
    """
    langs = list(acts_by_lang)
    n_features = len(next(iter(acts_by_lang.values())))
    if min_languages is None:
        min_languages = len(langs)
    active = np.zeros(n_features, dtype=int)
    for lang in langs:
        v = np.asarray(acts_by_lang[lang], dtype=float)
        positive = v[v > 0]
        if len(positive) == 0:
            # No feature fired for this language; nothing to threshold.
            continue
        thr = float(np.quantile(positive, quantile))
        active += ((v > 0) & (v >= thr)).astype(int)
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
    import time

    rng = np.random.default_rng(seed)
    U = field.U
    n = field.n_features
    xling = np.asarray(xling_indices, dtype=int)
    k = len(xling)
    if k < 3:
        raise ValueError("need at least 3 cross-linguistic features")

    # Safety guard: selection should produce a small k (tens, maybe
    # low hundreds). Anything beyond ~5% of the feature population
    # is implausible — almost certainly a selector failure on tied
    # values (e.g., zero-quantile collapse on sparse SAE activations)
    # — and the downstream batched tensors will OOM. Surface a clear
    # error before the matmul rather than after.
    k_max = max(1024, int(0.05 * n))
    if k > k_max:
        raise ValueError(
            f"cross-linguistic selection returned {k} features out of "
            f"{n} ({100 * k / n:.1f}%), well above the {k_max} safety "
            f"cap. Likely cause: the activation vectors are sparse "
            f"(JumpReLU SAE) and a quantile threshold collapsed to "
            f"zero, selecting every feature. Tighten the quantile or "
            f"check that the activation vectors have meaningful "
            f"non-zero variation across features.")

    pool_size = min(pool_size, n)
    full_pool = rng.choice(n, size=pool_size, replace=False)

    if loads is None:
        loads_cross = np.unique(np.round(
            np.geomspace(2, min(pool_size, 1024), 14)).astype(int))
        loads_int = np.unique(np.round(
            np.geomspace(2, max(k, 3), min(10, k))).astype(int))
    else:
        loads_cross = loads_int = np.asarray(loads)

    print(f"[view1_experiment] field={field.name}  d_model={field.d_model}  "
          f"n_features={n}  k={k}  pool_size={pool_size}  "
          f"n_boot={n_boot}  n_null={n_null}", flush=True)

    # CROSS-FIELD
    t_phase = time.time()
    print(f"  [phase] cross-field bootstrap ...", flush=True)
    cross = bootstrap_beta(U, xling, full_pool, loads_cross, n_boot, seed)
    print(f"  [phase] cross-field bootstrap done in "
          f"{time.time()-t_phase:.1f}s  beta={cross['beta']:.3f}",
          flush=True)

    t_phase = time.time()
    print(f"  [phase] uniform null ({n_null} draws) ...", flush=True)
    cross_null_unif = null_beta(
        U, k, full_pool, loads_cross,
        candidate_targets=None, n_draws=n_null, seed=seed + 7,
    )
    print(f"  [phase] uniform null done in {time.time()-t_phase:.1f}s",
          flush=True)
    cz_u, cp_u, cmu_u, csd_u = _z_and_p(cross["beta"], cross_null_unif)

    matched = None
    cz_m = cp_m = cmu_m = csd_m = None
    if activation_magnitude is not None:
        t_phase = time.time()
        print(f"  [phase] magnitude-matched null ({n_null} draws) ...",
              flush=True)
        matched = magnitude_matched_null_beta(
            U, xling, activation_magnitude, full_pool, loads_cross,
            n_draws=n_null, n_bins=n_bins, tol=hist_tol, seed=seed + 17,
        )
        print(f"  [phase] matched null done in {time.time()-t_phase:.1f}s  "
              f"widening_rate={matched['widening_rate']:.3f}  "
              f"max_hist_dev={matched['max_hist_dev']:.3f}",
              flush=True)
        cz_m, cp_m, cmu_m, csd_m = _z_and_p(cross["beta"], matched["betas"])

    # INTERNAL
    t_phase = time.time()
    print(f"  [phase] internal bootstrap + null ...", flush=True)
    intern = bootstrap_beta(U, xling, xling, loads_int, n_boot, seed)
    int_null = np.empty(n_null)
    for d in range(n_null):
        t = rng.choice(n, size=k, replace=False)
        int_null[d] = targeted_beta(U, t, t, loads_int)
    int_null = int_null[np.isfinite(int_null)]
    print(f"  [phase] internal done in {time.time()-t_phase:.1f}s  "
          f"internal_beta={intern['beta']:.3f}", flush=True)
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
