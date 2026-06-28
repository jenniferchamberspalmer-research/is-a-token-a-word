"""Superposition field — feature directions and interference scaling.

Each feature is a unit direction in d_model space. For an active set
S of L features, the signed interference on a target i is

    I_i(S) = < u_i, sum_{j in S, j != i} u_j >

Scaling fit:

    RMS_interference(L)  ~  L ** beta

beta is the critical exponent:

  ~0.5  sub-critical (incoherent): signed contributions cancel like a
        random walk — the geometry self-corrects.
  ~1.0  super-critical (coherent): contributions add in phase, runs
        away with L.
  ~0    free / orthogonal: negligible cross-talk.

The signed sum (not absolute) is what reveals whether the directions
cancel (sub-critical) or conspire (super-critical) at scale.

This module exposes:
  - SuperpositionField: a field of unit feature directions
  - random_field / coherent_field / orthogonal_field: synthetic fields
    with known answers, used by the validation script.

Note: the `from_gemma_scope(npz)` loader from the upstream draft is
intentionally NOT included here. SAEs are loaded through
water_tool.core.sae.get_sae() (which uses sae_lens canonical), and the
field is constructed by extracting `sae.W_dec` in the view handler.
There is a single SAE loader in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SuperpositionField:
    """A set of feature directions living in d_model-dimensional space.

    directions : (n_features, d_model) array. Rows are normalized to
                 unit length on construction; the un-normalized
                 magnitudes are retained in `_mag`.
    """
    directions: np.ndarray
    name: str = "field"

    def __post_init__(self):
        D = np.asarray(self.directions, dtype=np.float64)
        if D.ndim != 2:
            raise ValueError("directions must be 2D (n_features, d_model)")
        norms = np.linalg.norm(D, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._mag = norms.ravel()
        self.U = D / norms
        self.n_features, self.d_model = self.U.shape

    def interference_at_load(self, L: int, n_samples: int = 400,
                             active_sets: Optional[np.ndarray] = None,
                             rng: Optional[np.random.Generator] = None):
        """RMS + tail statistics of signed interference for active sets
        of size L, sampled `n_samples` times (or supplied)."""
        rng = rng or np.random.default_rng(0)
        if L < 2:
            return dict(L=L, rms=0.0, p95=0.0, p99=0.0, sir_median=np.inf)
        all_I = []
        for s in range(n_samples):
            if active_sets is not None:
                S = active_sets[s]
            else:
                S = rng.choice(self.n_features, size=L, replace=False)
            Us = self.U[S]
            G = Us @ Us.T
            np.fill_diagonal(G, 0.0)
            I_i = G.sum(axis=1)
            all_I.append(I_i)
        I = np.concatenate(all_I)
        rms = float(np.sqrt(np.mean(I**2)))
        p95 = float(np.percentile(np.abs(I), 95))
        p99 = float(np.percentile(np.abs(I), 99))
        sir_median = float(1.0 / (np.median(np.abs(I)) + 1e-12))
        return dict(L=L, rms=rms, p95=p95, p99=p99, sir_median=sir_median)

    def scaling_exponent(self, loads=None, n_samples: int = 400,
                         rng: Optional[np.random.Generator] = None):
        """Sweep load L and fit log RMS = beta * log L + c.

        Returns (beta, beta_stderr, sweep) where sweep is per-load
        records. Used for synthetic validation; the production analytic
        path is in experiment.targeted_beta(), which closes the curve
        without sampling.
        """
        rng = rng or np.random.default_rng(0)
        if loads is None:
            hi = min(self.n_features, max(8 * self.d_model, 64))
            loads = np.unique(np.round(
                np.geomspace(2, hi, num=16)).astype(int))
        sweep = [self.interference_at_load(int(L), n_samples=n_samples, rng=rng)
                 for L in loads]
        Ls = np.array([r["L"] for r in sweep], dtype=float)
        rms = np.array([r["rms"] for r in sweep], dtype=float)
        ok = rms > 0
        x, y = np.log(Ls[ok]), np.log(rms[ok])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, resid, *_ = np.linalg.lstsq(A, y, rcond=None)
        beta = float(coef[0])
        dof = max(len(x) - 2, 1)
        sigma2 = float(resid[0] / dof) if len(resid) else 0.0
        xvar = float(np.sum((x - x.mean())**2)) or 1.0
        beta_se = float(np.sqrt(sigma2 / xvar))
        return beta, beta_se, sweep


def random_field(n_features, d_model, seed=0):
    """Near-orthogonal Gaussian directions. Expected beta ~ 0.5."""
    rng = np.random.default_rng(seed)
    return SuperpositionField(rng.standard_normal((n_features, d_model)),
                              name="random (incoherent)")


def coherent_field(n_features, d_model, shared=0.6, seed=0):
    """Directions sharing a common component. Expected beta -> 1.0."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_features, d_model))
    common = rng.standard_normal((1, d_model))
    D = (1 - shared) * base + shared * common
    return SuperpositionField(D, name=f"coherent (shared={shared})")


def orthogonal_field(n_features, d_model, seed=0):
    """Orthonormal rows where n <= d. Expected beta ~ 0."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    n = min(n_features, d_model)
    return SuperpositionField(Q[:n], name="orthogonal (free)")
