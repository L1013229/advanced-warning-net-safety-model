"""Sensitivity analysis: PRCC with bootstrap confidence intervals.

PRCC (partial rank correlation coefficient) between each sampled input and
deltaH, controlling for all other inputs, computed by the standard
regression-residual method on rank-transformed data (Marino et al. 2008;
Blower & Dowlatabadi 1994). Bootstrap CIs quantify Monte Carlo uncertainty
in the PRCC estimates themselves (supervisor comments #93/#107).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Inputs eligible for PRCC screening (sampled, non-degenerate by default config)
DEFAULT_PRCC_INPUTS = [
    "d_aw_m", "v_walk_m_s", "t_handle_s", "mu_v_kmh", "sigma_v_kmh",
    "p_R", "deltaV_kmh", "a_comfort_m_s2", "r_E_per_veh_km",
    "alpha_lat_m_inv", "c_work_m", "c_deploy_m", "m_striking_kg", "m_lcv_kg",
]


def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def prcc(trace: Dict[str, np.ndarray] | pd.DataFrame, x_cols: Sequence[str],
         y_col: str = "deltaH") -> pd.DataFrame:
    """PRCC of y against each x, controlling for the other x's."""
    get = (lambda c: np.asarray(trace[c])) if isinstance(trace, dict) else (lambda c: trace[c].to_numpy())
    Xr = np.column_stack([_ranks(get(c)) for c in x_cols])
    yr = _ranks(get(y_col))

    n, k = Xr.shape
    ones = np.ones((n, 1))
    out = []
    for i, name in enumerate(x_cols):
        others = np.hstack([ones, np.delete(Xr, i, axis=1)])
        # Residualise x_i and y on the other rank-transformed inputs
        beta_x, *_ = np.linalg.lstsq(others, Xr[:, i], rcond=None)
        beta_y, *_ = np.linalg.lstsq(others, yr, rcond=None)
        rx = Xr[:, i] - others @ beta_x
        ry = yr - others @ beta_y
        denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
        r = float(np.sum(rx * ry) / denom) if denom > 0 else 0.0
        out.append({"parameter": name, "PRCC": r})
    df = pd.DataFrame(out)
    df["abs_PRCC"] = df["PRCC"].abs()
    return df.sort_values("abs_PRCC", ascending=False).reset_index(drop=True)


def prcc_bootstrap(trace: Dict[str, np.ndarray] | pd.DataFrame, x_cols: Sequence[str],
                   y_col: str = "deltaH", n_boot: int = 1000, seed: int = 20260704,
                   subsample: Optional[int] = None) -> pd.DataFrame:
    """PRCC point estimates with percentile bootstrap 95% CIs.

    ``subsample`` (if set) bootstraps on smaller resamples for speed; CI width
    is then conservative for the full-sample estimate.
    """
    get = (lambda c: np.asarray(trace[c])) if isinstance(trace, dict) else (lambda c: trace[c].to_numpy())
    cols = {c: get(c) for c in list(x_cols) + [y_col]}
    n = len(cols[y_col])
    m = subsample or n
    rng = np.random.default_rng(seed)

    point = prcc(cols, x_cols, y_col).set_index("parameter")["PRCC"]

    boots: Dict[str, List[float]] = {c: [] for c in x_cols}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=m)
        sub = {c: v[idx] for c, v in cols.items()}
        b = prcc(sub, x_cols, y_col).set_index("parameter")["PRCC"]
        for c in x_cols:
            boots[c].append(float(b[c]))

    rows = []
    for c in x_cols:
        arr = np.array(boots[c])
        rows.append({"parameter": c, "PRCC": float(point[c]),
                     "ci_lo": float(np.percentile(arr, 2.5)),
                     "ci_hi": float(np.percentile(arr, 97.5)),
                     "boot_sd": float(arr.std(ddof=1))})
    df = pd.DataFrame(rows)
    df["abs_PRCC"] = df["PRCC"].abs()
    return df.sort_values("abs_PRCC", ascending=False).reset_index(drop=True)
