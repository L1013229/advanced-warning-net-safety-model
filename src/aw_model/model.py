"""Core simulation, decomposed into individually testable stages.

Stage functions are pure (no RNG); all randomness is consumed inside
``sample_inputs`` in the legacy v0.1 order so that, with correlation disabled
and the same seed, the rebuilt model reproduces v0.1 outputs bit-for-bit.

Equation map (manuscript section -> function):
  T_deploy = 2(2 d_aw / v_walk + t_handle)          -> compute_deployment_exposure
  dV_max: v1^2 = v0^2 - 2 a d                        -> feasible_delta_v_kmh
  V1 = max(V0 - I_R min(dV, dV_max), 0)              -> apply_sign_response
  lambda_j = N (r_E L_j) exp(-alpha c_j)             -> compute_encroachment_frequencies
  p_w(V), p_o(dV_occ), dV_occ = V m2/(m1+m2)         -> compute_severity
  H_S0, H_S1, deltaH                                 -> compute_harm
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .distributions import sample_all, apply_correlation
from .risk_functions import p_worker, p_occupant


# ---------------------------------------------------------------------------
# Small kinematic/exposure helpers (unchanged physics from v0.1)
# ---------------------------------------------------------------------------

def _clip(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(v, lo), hi)


def feasible_delta_v_kmh(v0_kmh: np.ndarray, a_m_s2: np.ndarray, d_aw_m: np.ndarray) -> np.ndarray:
    """Maximum feasible speed reduction (km/h) over d_aw at deceleration a."""
    v0_ms = v0_kmh / 3.6
    v1_sq = np.maximum(v0_ms**2 - 2.0 * a_m_s2 * d_aw_m, 0.0)
    delta_ms = np.maximum(v0_ms - np.sqrt(v1_sq), 0.0)
    return delta_ms * 3.6


def deployment_time_seconds(d_aw_m: np.ndarray, v_walk_m_s: np.ndarray, t_handle_s: np.ndarray) -> np.ndarray:
    """Total on-foot deployment+removal time (s): 2 * (2 d_aw / v_walk + t_handle)."""
    return 2.0 * (2.0 * d_aw_m / v_walk_m_s + t_handle_s)


def p_reach_exceedance(c_m: np.ndarray, alpha_m_inv: np.ndarray) -> np.ndarray:
    """Exponential lateral-extent exceedance: P(reach >= c) = exp(-alpha c)."""
    return np.exp(-alpha_m_inv * c_m)


# ---------------------------------------------------------------------------
# Stage 1: sampling (all RNG lives here)
# ---------------------------------------------------------------------------

def sample_inputs(rng: np.random.Generator, cfg: Dict[str, Any], n: int, seed: int) -> Dict[str, np.ndarray]:
    """Draw all uncertain inputs plus V0 and the response uniforms.

    RNG consumption order (legacy-compatible):
      1) each distribution in config order, 2) V0 normal draw, 3) response uniforms.
    Correlation (if configured) reorders already-drawn columns and consumes no
    draws from this stream (its score matrix uses a derived, dedicated RNG).
    """
    draws = sample_all(rng, cfg, n)

    # Correlation must be imposed BEFORE V0 is generated so that pairs
    # involving mu_v/sigma_v propagate into the speed actually used downstream.
    # apply_correlation consumes nothing from `rng`, so with correlation
    # disabled the stream is bit-identical to the legacy (v0.1) order.
    draws = apply_correlation(draws, cfg, seed)

    # Baseline operating speed V0 ~ Normal(mu_v, sigma_v), clipped to [0, v_cap].
    # Clipping (censoring) sets out-of-range draws to the bounds; on average
    # fewer than 0.1% of draws touch the cap under the base priors.
    mu_v = draws["mu_v_kmh"]
    sd_v = np.maximum(draws["sigma_v_kmh"], 0.1)
    V0 = _clip(rng.normal(mu_v, sd_v, size=n), 0.0, draws["v_cap_kmh"])

    # Uniforms deciding per-vehicle-population response (Bernoulli mixture)
    u_resp = rng.random(n)

    draws["V0_kmh"] = V0
    draws["u_resp"] = u_resp
    return draws


# ---------------------------------------------------------------------------
# Stage 2: deployment exposure
# ---------------------------------------------------------------------------

def compute_deployment_exposure(draws: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    T_deploy_s = deployment_time_seconds(draws["d_aw_m"], draws["v_walk_m_s"], draws["t_handle_s"])
    return {"T_deploy_s": T_deploy_s, "T_deploy_h": T_deploy_s / 3600.0}


# ---------------------------------------------------------------------------
# Stage 3: sign response and speeds
# ---------------------------------------------------------------------------

def apply_sign_response(draws: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Bernoulli response mixture with kinematic feasibility cap.

    response_allocation (config: model.response_allocation) controls WHICH
    drivers respond while holding the marginal response probability at p_R:
      - independent (default/legacy): responders drawn uniformly.
      - low_tail: responders concentrated among slower drivers (pessimistic
        for the sign, since severity benefit lives in the fast tail).
      - high_tail: responders concentrated among faster drivers (optimistic).
    Linear rank weights w in [0, 2] with mean 1 preserve the marginal response
    share to first order; achieved shares are reported in outputs.
    """
    n = len(draws["V0_kmh"])
    V0 = draws["V0_kmh"]
    p_R = _clip(draws["p_R"], 0.0, 1.0)

    alloc = str(cfg.get("model", {}).get("response_allocation", "independent")).lower()
    if alloc == "independent":
        p_eff = p_R
    else:
        ranks = np.argsort(np.argsort(V0)).astype(float) / max(n - 1, 1)  # 0..1, 1 = fastest
        if alloc == "high_tail":
            w = 2.0 * ranks
        elif alloc == "low_tail":
            w = 2.0 * (1.0 - ranks)
        else:
            raise ValueError(f"Unknown response_allocation '{alloc}'")
        p_eff = _clip(p_R * w, 0.0, 1.0)

    resp = draws["u_resp"] < p_eff

    deltaV = np.maximum(draws["deltaV_kmh"], 0.0)
    deltaV_max = feasible_delta_v_kmh(V0, np.maximum(draws["a_comfort_m_s2"], 0.1), draws["d_aw_m"])
    deltaV_used = np.minimum(deltaV, deltaV_max)
    V1 = np.maximum(V0 - resp.astype(float) * deltaV_used, 0.0)

    return {"resp": resp, "deltaV_used": deltaV_used, "V1_kmh": V1,
            "achieved_response_share": float(np.mean(resp)),
            "target_response_share": float(np.mean(p_R))}


# ---------------------------------------------------------------------------
# Stage 4: encroachment frequencies
# ---------------------------------------------------------------------------

def encroachment_rate_per_vehicle_km(Q_veh_h: float, cfg_model: Dict[str, Any],
                                     draws: Dict[str, np.ndarray]) -> np.ndarray:
    """r_E per vehicle-km (constant-draw mode, or Glennon-Wilton derived)."""
    mode = str(cfg_model.get("encroachment_rate_model", "constant_per_vehicle_km")).lower()
    if mode == "constant_per_vehicle_km":
        return draws["r_E_per_veh_km"]
    if mode == "glennon_wilton_urban_arterial":
        mult = float(cfg_model.get("Q_to_ADT_multiplier", 24.0))
        ADT = max(Q_veh_h * mult, 1.0)
        accidents_per_mile_year = 0.474 + 0.000254 * ADT
        encroachments_per_mile_year = 5.23 * accidents_per_mile_year
        r_E = (encroachments_per_mile_year / 1.609344) / (ADT * 365.0)
        return r_E * draws["k_rE"]
    raise ValueError(f"Unknown encroachment_rate_model: {mode}")


def compute_encroachment_frequencies(Q_veh_h: float, T_work_h: float,
                                     draws: Dict[str, np.ndarray],
                                     exposure: Dict[str, np.ndarray],
                                     cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Expected strikes per job for each pathway: lambda = N (r_E L) exp(-alpha c)."""
    n = len(draws["V0_kmh"])
    cfg_model = cfg.get("model", {})

    r_E0 = encroachment_rate_per_vehicle_km(Q_veh_h, cfg_model, draws)
    if bool(cfg_model.get("allow_frequency_pathway", False)):
        k_E = _clip(draws["k_E"], 0.0, 1.0)
    else:
        k_E = np.ones(n)
    r_E1 = r_E0 * k_E

    alpha = np.maximum(draws["alpha_lat_m_inv"], 1e-6)
    p_reach_work = p_reach_exceedance(np.maximum(draws["c_work_m"], 0.0), alpha)
    p_reach_deploy = p_reach_exceedance(np.maximum(draws["c_deploy_m"], 0.0), alpha)

    L_worker_km = np.maximum(draws["L_worker_m"], 0.1) / 1000.0
    L_lcv_km = np.maximum(draws["L_lcv_m"], 0.1) / 1000.0
    L_deploy_km = np.maximum(draws["L_worker_deploy_m"], 0.1) / 1000.0

    N_work = Q_veh_h * T_work_h                    # vehicles passing during work
    N_deploy = Q_veh_h * exposure["T_deploy_h"]    # vehicles passing during deploy+removal

    return {
        "r_E0": r_E0, "r_E1": r_E1, "k_E": k_E,
        "p_reach_work": p_reach_work, "p_reach_deploy": p_reach_deploy,
        "lambda_w0": N_work * (r_E0 * L_worker_km) * p_reach_work,
        "lambda_v0": N_work * (r_E0 * L_lcv_km) * p_reach_work,
        "lambda_w1": N_work * (r_E1 * L_worker_km) * p_reach_work,
        "lambda_v1": N_work * (r_E1 * L_lcv_km) * p_reach_work,
        "lambda_d": N_deploy * (r_E0 * L_deploy_km) * p_reach_deploy,
        "N_work": np.full(n, N_work), "N_deploy": N_deploy,
    }


# ---------------------------------------------------------------------------
# Stage 5: severity
# ---------------------------------------------------------------------------

def compute_severity(draws: Dict[str, np.ndarray], speeds: Dict[str, np.ndarray],
                     cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Injury probabilities for worker (impact speed) and occupant (delta-V)."""
    sev = cfg.get("severity", {}) or {}
    worker_curve = sev.get("worker_curve", "rosen_mais3plus")
    occupant_curve = sev.get("occupant_curve", "wang_mais3plus_all")

    V0, V1 = draws["V0_kmh"], speeds["V1_kmh"]
    m1 = np.maximum(draws["m_striking_kg"], 100.0)
    m2 = np.maximum(draws["m_lcv_kg"], 100.0)
    ratio = m2 / (m1 + m2)          # perfectly inelastic momentum transfer
    dv0, dv1 = V0 * ratio, V1 * ratio

    return {
        "p_w0": p_worker(V0, worker_curve), "p_w1": p_worker(V1, worker_curve),
        "dv_occ0": dv0, "dv_occ1": dv1,
        "p_o0": p_occupant(dv0, occupant_curve), "p_o1": p_occupant(dv1, occupant_curve),
    }


# ---------------------------------------------------------------------------
# Stage 6: harm accounting
# ---------------------------------------------------------------------------

def compute_harm(freq: Dict[str, np.ndarray], sev: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """H_S0, H_S1 and deltaH per iteration. Deployment severity uses V0."""
    H0 = freq["lambda_w0"] * sev["p_w0"] + freq["lambda_v0"] * sev["p_o0"]
    H1 = (freq["lambda_w1"] * sev["p_w1"] + freq["lambda_v1"] * sev["p_o1"]
          + freq["lambda_d"] * sev["p_w0"])
    return {"H0": H0, "H1": H1, "deltaH": H1 - H0}


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def wilson_ci(p_hat: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1.0 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def simulate_point(Q_veh_h: float, T_work_h: float, n_iter: int, seed: int,
                   cfg: Dict[str, Any], full_trace: bool = False) -> Dict[str, Any]:
    """Simulate one (Q, T_work) grid point.

    Returns summary stats, the deltaH array, and (optionally) the full
    per-iteration trace of every sampled input and intermediate quantity.
    """
    rng = np.random.default_rng(seed)
    draws = sample_inputs(rng, cfg, n_iter, seed)

    exposure = compute_deployment_exposure(draws)
    speeds = apply_sign_response(draws, cfg)
    freq = compute_encroachment_frequencies(Q_veh_h, T_work_h, draws, exposure, cfg)
    sev = compute_severity(draws, speeds, cfg)
    harm = compute_harm(freq, sev)

    deltaH = harm["deltaH"]
    p_benefit = float(np.mean(deltaH < 0.0))
    lo, hi = wilson_ci(p_benefit, n_iter)

    # Relative framing (supervisor comment #102): deltaH as share of baseline harm
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(harm["H0"] > 0, deltaH / harm["H0"], np.nan)

    # Achieved rank correlations for configured pairs (audit trail), including
    # the propagated correlation with the effective speed V0 for mu_v pairs.
    achieved_corr = {}
    for pair in (cfg.get("correlation", {}) or {}).get("pairs", []) or []:
        v1, v2, rho = str(pair[0]), str(pair[1]), float(pair[2])
        def _sp(a, b):
            ra = np.argsort(np.argsort(a)).astype(float)
            rb = np.argsort(np.argsort(b)).astype(float)
            return float(np.corrcoef(ra, rb)[0, 1])
        achieved_corr[f"{v1}~{v2}"] = {"target": rho, "achieved": _sp(draws[v1], draws[v2])}
        for v in (v1, v2):
            if v == "mu_v_kmh":
                other = v2 if v == v1 else v1
                achieved_corr[f"V0_kmh~{other}"] = {"target": None,
                                                    "achieved": _sp(draws["V0_kmh"], draws[other])}

    result: Dict[str, Any] = {
        "Q_veh_h": Q_veh_h,
        "T_work_h": T_work_h,
        "achieved_correlations": achieved_corr,
        "mean_deltaH": float(np.mean(deltaH)),
        "median_deltaH": float(np.median(deltaH)),
        "p_benefit": p_benefit,
        "p_benefit_lo95": lo,
        "p_benefit_hi95": hi,
        "mean_H0": float(np.mean(harm["H0"])),
        "median_H0": float(np.median(harm["H0"])),
        "median_rel_deltaH": float(np.nanmedian(rel)),
        "achieved_response_share": speeds["achieved_response_share"],
        "target_response_share": speeds["target_response_share"],
        "deltaH": deltaH,
    }

    if full_trace:
        trace: Dict[str, np.ndarray] = {}
        for src in (draws, exposure,
                    {k: v for k, v in speeds.items() if isinstance(v, np.ndarray)},
                    freq, sev, harm):
            trace.update(src)
        result["trace"] = trace

    return result


def convergence_curves(deltaH: np.ndarray, interval: int = 1000) -> Dict[str, np.ndarray]:
    """Running mean of deltaH and running P(deltaH<0) at checkpoint intervals."""
    n = len(deltaH)
    ks = np.arange(interval, n + 1, interval)
    csum = np.cumsum(deltaH)
    cneg = np.cumsum((deltaH < 0).astype(float))
    return {"n": ks,
            "running_mean": csum[ks - 1] / ks,
            "running_p_benefit": cneg[ks - 1] / ks}
