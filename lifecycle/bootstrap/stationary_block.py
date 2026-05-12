"""Stationary block bootstrap (Politis-Romano 1994).

Block lengths are iid Geometric with mean `ell_mean`; block start indices
are uniform on [0, T) with wrap-around. The resampled series is stationary.
"""
from __future__ import annotations

import numpy as np


def stationary_block_indices(
    T: int,
    ell_mean: float,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """Return resampled donor row indices of shape (n_draws, T).

    Each row is a pseudo-time-series of length T drawn from the stationary
    block bootstrap with mean block length `ell_mean` and circular wrap-around.
    Deterministic given `seed`.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if ell_mean <= 0:
        raise ValueError(f"ell_mean must be positive, got {ell_mean}")
    if n_draws < 0:
        raise ValueError(f"n_draws must be non-negative, got {n_draws}")

    p = 1.0 / ell_mean
    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, T), dtype=np.int64)

    for b in range(n_draws):
        idx = np.empty(T, dtype=np.int64)
        t = 0
        while t < T:
            start = int(rng.integers(0, T))
            run_len = 1
            while rng.random() >= p and (t + run_len) < T:
                run_len += 1
            for k in range(run_len):
                idx[t + k] = (start + k) % T
            t += run_len
        out[b] = idx

    return out


def identity_indices(T: int) -> np.ndarray:
    return np.arange(T, dtype=np.int64).reshape(1, T)
