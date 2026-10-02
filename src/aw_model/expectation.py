"""Job-expectation iteration mode.

Each Monte Carlo iteration is one job under one draw of the uncertain
parameters. Quantities that vary from vehicle to vehicle are averaged inside
the iteration rather than drawn once per iteration:

  * operating speed  V ~ Normal(mu_V, sigma_V), clipped to [0, v_cap]
                     (Gauss-Hermite quadrature on the probabilists' weight);
  * striking-vehicle mass m1 ~ Uniform(lo, hi) (Gauss-Legendre quadrature);
  * driver response: a share p of departing drivers were made aware by the
    sign; an aware driver reacts dt_react_saved sooner and brakes at a_brake
    over the distance that time buys, so strikes at a lower speed. Every
    severity under S1 is the p-weighted mixture of the aware and unaware
    values. With independent allocation deltaH = C - p B exactly.

The representative-vehicle mode (one speed, one mass, one Bernoulli response
per iteration) is unchanged in ``model.py`` and remains behind the v0.1
reproduction gate; it yields the per-encounter statistic.
"""
from __future__ import annotations

from math import erf, sqrt
from typing import Any, Dict

import numpy as np

from .risk_functions import p_worker, p_occupant

N_HERMITE = 21
N_LEGENDRE = 5


def _hermite_nodes(n: int):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


def _legendre_nodes(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w / w.sum()


def _std_normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def _feasible_delta_v(v_kmh, a_m_s2, d_aw_m):
    v_ms = v_kmh / 3.6
    v1_sq = np.maximum(v_ms**2 - 2.0 * a_m_s2 * d_aw_m, 0.0)
    return np.maximum(v_ms - np.sqrt(v1_sq), 0.0) * 3.6


def expected_severities(draws: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-iteration expected severities over the within-site vehicle stream.

    Returns p_w0, p_o0 (no sign), p_w1, p_o1 (with sign, response mixed in),
    the responder-only severities p_w1_resp / p_o1_resp, and the effective
    response share actually applied (differs from p_R only under a tail
    allocation with clipping).
    """
    model = cfg.get("model", {}) or {}
    sev = cfg.get("severity", {}) or {}
    worker_curve = sev.get("worker_curve", "rosen_mais3plus")
    occupant_curve = sev.get("occupant_curve", "wang_mais3plus_all")
    alloc = str(model.get("response_allocation", "independent")).lower()
    nh = int(model.get("quadrature_speed_nodes", N_HERMITE))
    nl = int(model.get("quadrature_mass_nodes", N_LEGENDRE))

    xh, wh = _hermite_nodes(nh)
    xl, wl = _legendre_nodes(nl)

    mu = draws["mu_v_kmh"][:, None]
    sd = np.maximum(draws["sigma_v_kmh"], 0.1)[:, None]
    cap = draws["v_cap_kmh"][:, None]
    V = np.minimum(np.maximum(mu + sd * xh[None, :], 0.0), cap)          # (n, nh)

    p_R = np.minimum(np.maximum(draws["p_R"], 0.0), 1.0)[:, None]
    if alloc == "independent":
        p_k = np.broadcast_to(p_R, V.shape)
    else:
        # within-site speed rank F(V); linear weights w in [0, 2] with mean 1
        rank = _std_normal_cdf(xh)[None, :]
        w = 2.0 * rank if alloc == "high_tail" else 2.0 * (1.0 - rank) if alloc == "low_tail" else None
        if w is None:
            raise ValueError(f"Unknown response_allocation '{alloc}'")
        p_k = np.minimum(p_R * w, 1.0)

    # Awareness mechanism: a driver the sign made aware reacts dt sooner to the
    # departure and brakes at a_b over the distance that time buys, V dt. Impact speed
    # Vr^2 = V^2 - 2 a_b V dt (in m/s), floored at zero. Non-responders strike at V.
    dt = np.maximum(draws["dt_react_saved_s"], 0.0)[:, None]
    a_b = np.maximum(draws["a_brake_m_s2"], 0.1)[:, None]
    V_ms = V / 3.6
    Vr = np.sqrt(np.maximum(V_ms**2 - 2.0 * a_b * V_ms * dt, 0.0)) * 3.6   # responder speed

    # striking-vehicle mass nodes from the configured uniform range
    spec = cfg["distributions"]["m_striking_kg"]
    if spec.get("dist") == "fixed":
        m1 = np.array([float(spec["value"])]); wl = np.array([1.0])
    else:
        lo, hi = float(spec["min"]), float(spec["max"])
        m1 = 0.5 * (lo + hi) + 0.5 * (hi - lo) * xl
    m2 = np.maximum(draws["m_lcv_kg"], 100.0)[:, None, None]
    ratio = m2 / (m1[None, None, :] + m2)                                   # (n, 1, nl)

    def over_speed(f):   # (n, nh) -> (n,)
        return (f * wh[None, :]).sum(axis=1)

    def over_speed_mass(f):  # (n, nh, nl) -> (n,)
        return (f * wh[None, :, None] * wl[None, None, :]).sum(axis=(1, 2))

    pw_V, pw_Vr = p_worker(V, worker_curve), p_worker(Vr, worker_curve)
    po_V = p_occupant(V[:, :, None] * ratio, occupant_curve)
    po_Vr = p_occupant(Vr[:, :, None] * ratio, occupant_curve)

    p_w0 = over_speed(pw_V)
    p_o0 = over_speed_mass(po_V)
    p_w1 = over_speed((1.0 - p_k) * pw_V + p_k * pw_Vr)
    p_o1 = over_speed_mass((1.0 - p_k)[:, :, None] * po_V + p_k[:, :, None] * po_Vr)
    p_w1_resp = over_speed(pw_Vr)
    p_o1_resp = over_speed_mass(po_Vr)
    p_eff = over_speed(p_k)
    return {"p_w0": p_w0, "p_o0": p_o0, "p_w1": p_w1, "p_o1": p_o1,
            "p_w1_resp": p_w1_resp, "p_o1_resp": p_o1_resp,
            "achieved_response_share": float(np.mean(p_eff)),
            "target_response_share": float(np.mean(p_R))}


def compute_harm_expectation(freq: Dict[str, np.ndarray], sev: Dict[str, np.ndarray],
                             p_R: np.ndarray) -> Dict[str, np.ndarray]:
    """H_S0, H_S1, deltaH per job plus the threshold components.

    C = deployment harm (always added under S1); B = benefit per unit response
    share, so deltaH = C - p B under independent allocation (secant definition
    B = (C - deltaH)/p otherwise, which coincides where no clipping occurs).
    """
    H0 = freq["lambda_w0"] * sev["p_w0"] + freq["lambda_v0"] * sev["p_o0"]
    C = freq["lambda_d"] * sev["p_w0"]
    H1 = freq["lambda_w1"] * sev["p_w1"] + freq["lambda_v1"] * sev["p_o1"] + C
    dH = H1 - H0
    p = np.minimum(np.maximum(p_R, 0.0), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(p > 0, (C - dH) / p,
                     freq["lambda_w1"] * (sev["p_w0"] - sev["p_w1_resp"])
                     + freq["lambda_v1"] * (sev["p_o0"] - sev["p_o1_resp"]))
        p_star = np.where(B > 0, C / B, np.inf)       # required departer response share
    return {"H0": H0, "H1": H1, "deltaH": dH, "C_deploy": C, "B_per_share": B, "p_star": p_star}
