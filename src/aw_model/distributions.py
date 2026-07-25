"""Distribution sampling and Iman-Conover rank-correlation induction.

Sampling semantics are identical to the v0.1 code for the five supported
distribution families; when no correlation is configured the RNG stream is
consumed in exactly the legacy order so v0.1 results reproduce bit-for-bit.

Correlation is imposed with the distribution-free Iman & Conover (1982)
method: independent marginals are drawn first, then column values are
reordered to match the rank structure of a score matrix carrying the target
Spearman correlation. Marginal distributions are exactly preserved.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import CANONICAL_DIST_ORDER


class DistributionError(ValueError):
    pass


def _require_keys(spec: Dict[str, Any], keys: Tuple[str, ...], name: str) -> None:
    missing = [k for k in keys if k not in spec]
    if missing:
        raise DistributionError(f"Distribution '{name}' missing keys: {missing}")


def sample_dist(rng: np.random.Generator, spec: Dict[str, Any], size: int, name: str) -> np.ndarray:
    """Sample an array from a distribution specification.

    Supported 'dist' values: fixed {value}; uniform {min,max};
    loguniform {min,max} (positive); normal {mean,sd}; triangular {min,mode,max}.
    """
    if "dist" not in spec:
        raise DistributionError(f"Distribution '{name}' missing 'dist' field")
    d = str(spec["dist"]).lower()

    if d == "fixed":
        _require_keys(spec, ("value",), name)
        return np.full(size, float(spec["value"]), dtype=float)
    if d == "uniform":
        _require_keys(spec, ("min", "max"), name)
        return rng.uniform(float(spec["min"]), float(spec["max"]), size=size)
    if d == "loguniform":
        _require_keys(spec, ("min", "max"), name)
        a, b = float(spec["min"]), float(spec["max"])
        if a <= 0 or b <= 0:
            raise DistributionError(f"loguniform requires positive bounds for '{name}'")
        return np.exp(rng.uniform(np.log(a), np.log(b), size=size))
    if d == "normal":
        _require_keys(spec, ("mean", "sd"), name)
        return rng.normal(float(spec["mean"]), float(spec["sd"]), size=size)
    if d == "triangular":
        _require_keys(spec, ("min", "mode", "max"), name)
        return rng.triangular(float(spec["min"]), float(spec["mode"]), float(spec["max"]), size=size)

    raise DistributionError(f"Unsupported dist '{spec['dist']}' for '{name}'")


def sample_all(rng: np.random.Generator, cfg: Dict[str, Any], n: int) -> Dict[str, np.ndarray]:
    """Draw every configured distribution in the canonical stream order.

    The order is taken from CANONICAL_DIST_ORDER, not from the mapping, so a
    caller that assembles a config without going through ``load_config`` still
    consumes the RNG stream in the order the results of record were produced
    under. ``validate_config`` enforces that the two agree; this makes the
    guarantee hold even when it is bypassed.
    """
    dists = cfg["distributions"]
    missing = [name for name in CANONICAL_DIST_ORDER if name not in dists]
    if missing:
        raise DistributionError(f"Config missing distributions: {missing}")
    extra = [name for name in dists if name not in CANONICAL_DIST_ORDER]
    if extra:
        raise DistributionError(
            f"Config carries distributions outside CANONICAL_DIST_ORDER: {extra}. "
            "Adding one shifts every later variable's slice of the RNG stream."
        )
    return {name: sample_dist(rng, dists[name], n, name) for name in CANONICAL_DIST_ORDER}


# ----------------------------------------------------------------------------
# Iman-Conover rank correlation
# ----------------------------------------------------------------------------

def build_spearman_matrix(names: Sequence[str], pairs: Sequence[Sequence[Any]]) -> np.ndarray:
    """Build the full target Spearman matrix from (var1, var2, rho) triples.

    Unspecified pairs default to zero (independence). Raises if the resulting
    matrix is not positive definite (an infeasible correlation structure).
    """
    k = len(names)
    idx = {name: i for i, name in enumerate(names)}
    C = np.eye(k)
    for v1, v2, rho in pairs:
        i, j = idx[str(v1)], idx[str(v2)]
        C[i, j] = C[j, i] = float(rho)
    # Feasibility check
    eigmin = float(np.linalg.eigvalsh(C).min())
    if eigmin <= 1e-10:
        raise DistributionError(f"Target Spearman matrix not positive definite (min eig {eigmin:.3g})")
    return C


def iman_conover(samples: Dict[str, np.ndarray], names: Sequence[str],
                 spearman_target: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Impose a target Spearman rank correlation on independently drawn samples.

    Implementation of Iman & Conover (1982): build a score matrix with the
    target correlation (via Cholesky), then reorder each sample column so its
    ranks match the ranks of the corresponding score column. Marginals are
    exactly preserved; only the joint ordering changes.

    A dedicated RNG (derived from ``seed``) generates the score matrix, so the
    main sampling stream is untouched.
    """
    n = len(next(iter(samples.values())))
    k = len(names)

    # Spearman -> Pearson conversion for normal scores (Kruskal 1958)
    pearson = 2.0 * np.sin(np.pi / 6.0 * spearman_target)
    # Guard: conversion can nudge eigenvalues; re-check
    eigmin = float(np.linalg.eigvalsh(pearson).min())
    if eigmin <= 1e-10:
        raise DistributionError("Converted Pearson score matrix not positive definite")

    score_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1C0C]))
    M = score_rng.standard_normal((n, k))
    # Remove sampling noise in M's own correlation, then impose the target
    E = np.corrcoef(M, rowvar=False)
    Q = np.linalg.cholesky(E)
    P = np.linalg.cholesky(pearson)
    T = M @ np.linalg.inv(Q).T @ P.T  # n x k scores with corr ~= pearson

    out = dict(samples)
    for col, name in enumerate(names):
        x = samples[name]
        order_scores = np.argsort(np.argsort(T[:, col]))   # ranks of scores (0..n-1)
        x_sorted = np.sort(x)
        out[name] = x_sorted[order_scores]                 # x reordered to score ranks
    return out


def apply_correlation(draws: Dict[str, np.ndarray], cfg: Dict[str, Any], seed: int) -> Dict[str, np.ndarray]:
    """Apply configured rank correlation to sampled draws (no-op when unset)."""
    corr = cfg.get("correlation", {}) or {}
    pairs = corr.get("pairs", []) or []
    if not pairs:
        return draws
    # Only variables involved in pairs (plus nothing else) need reordering;
    # restricting to them keeps every other column exactly at its legacy order.
    involved: List[str] = []
    for v1, v2, _ in pairs:
        for v in (str(v1), str(v2)):
            if v not in involved:
                involved.append(v)
    # Preserve canonical ordering of the involved subset for determinism
    names = [n for n in CANONICAL_DIST_ORDER if n in involved]
    C = build_spearman_matrix(names, pairs)
    sub = {n: draws[n] for n in names}
    reordered = iman_conover(sub, names, C, seed)
    out = dict(draws)
    out.update(reordered)
    return out
