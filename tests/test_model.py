"""Hand-calculation and structural tests for the model stages."""
import math

import numpy as np
import pytest

from aw_model.model import (
    apply_sign_response,
    compute_deployment_exposure,
    compute_encroachment_frequencies,
    compute_harm,
    compute_severity,
    deployment_time_seconds,
    feasible_delta_v_kmh,
    simulate_point,
    wilson_ci,
    convergence_curves,
)
from aw_model.config import load_config
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"


def test_deployment_time_hand_calc():
    # d=50 m, walk 1.25 m/s, handle 20 s: 2*(2*50/1.25 + 20) = 2*(80+20) = 200 s
    t = deployment_time_seconds(np.array([50.0]), np.array([1.25]), np.array([20.0]))
    assert t[0] == pytest.approx(200.0, abs=1e-12)


def test_feasibility_hand_calc():
    # v0=50 km/h=13.888.. m/s, a=1.0 m/s2, d=50 m:
    # v1^2 = 13.888..^2 - 100 = 92.9012.. -> v1=9.6385.. m/s -> dv = 4.2503 m/s = 15.3011 km/h
    v0 = np.array([50.0])
    dv = feasible_delta_v_kmh(v0, np.array([1.0]), np.array([50.0]))
    v0_ms = 50.0 / 3.6
    expect = (v0_ms - math.sqrt(v0_ms**2 - 2 * 1.0 * 50.0)) * 3.6
    assert dv[0] == pytest.approx(expect, rel=1e-12)


def test_feasibility_full_stop():
    # Short distance, huge decel demand: cannot exceed v0 itself
    dv = feasible_delta_v_kmh(np.array([50.0]), np.array([10.0]), np.array([200.0]))
    assert dv[0] == pytest.approx(50.0, rel=1e-12)


def _fixed_cfg():
    """All inputs fixed -> deterministic deltaH verifiable by hand."""
    cfg = load_config(CONFIG_DIR / "base.yaml")
    fx = {
        "d_aw_m": 50.0, "v_walk_m_s": 1.25, "t_handle_s": 20.0,
        "mu_v_kmh": 50.0, "sigma_v_kmh": 0.0,       # sd clamped to 0.1 internally; negligible
        "v_cap_kmh": 80.0, "p_R": 0.0,               # no response -> V1 = V0
        "deltaV_kmh": 10.0, "a_comfort_m_s2": 2.0,
        "r_E_per_veh_km": 1e-6, "k_rE": 1.0, "alpha_lat_m_inv": 0.10,
        "c_work_m": 2.0, "c_deploy_m": 1.0, "k_E": 1.0,
        "L_worker_m": 10.0, "L_lcv_m": 6.0, "L_worker_deploy_m": 1.0,
        "m_striking_kg": 1500.0, "m_lcv_kg": 2500.0,
    }
    for k, v in fx.items():
        cfg["distributions"][k] = {"dist": "fixed", "value": v}
    return cfg


def test_deterministic_hand_calculation():
    """Full model vs hand-computed deltaH with fixed inputs (p_R=0)."""
    cfg = _fixed_cfg()
    Q, T = 400.0, 0.25
    res = simulate_point(Q, T, n_iter=200, seed=1, cfg=cfg)

    # Hand calculation (V0 ~= 50 with sd 0.1 clip; use the actual sampled mean)
    # With p_R = 0 the sign gives NO benefit; deltaH = lambda_d * p_w(V0) exactly.
    T_dep_h = 200.0 / 3600.0
    lam_d = Q * T_dep_h * (1e-6 * 1.0 / 1000.0) * math.exp(-0.10 * 1.0)
    # p_w at 50 km/h: 1/(1+exp(4.6-0.078*50))
    p_w50 = 1.0 / (1.0 + math.exp(4.6 - 0.078 * 50.0))
    expect = lam_d * p_w50

    # sd=0.1 sampling noise in V0 propagates weakly; tolerance reflects that
    assert res["mean_deltaH"] == pytest.approx(expect, rel=2e-3)
    assert res["p_benefit"] == 0.0  # deployment-only pathway can never help at p_R=0


def test_harm_identity_no_sign_effect():
    """With p_R=0 and k_E=1: H1 - H0 == lambda_d * p_w0 exactly, per iteration."""
    cfg = _fixed_cfg()
    res = simulate_point(400.0, 0.25, n_iter=500, seed=7, cfg=cfg, full_trace=True)
    tr = res["trace"]
    np.testing.assert_allclose(tr["deltaH"], tr["lambda_d"] * tr["p_w0"], rtol=1e-12)


def test_tail_allocation_preserves_marginal_and_orders_effect():
    """low_tail / high_tail keep the response share ~= p_R but shift who responds."""
    cfg = load_config(CONFIG_DIR / "base.yaml", [CONFIG_DIR / "scenarios/high_pr.yaml"])
    out = {}
    for mode in ("independent", "low_tail", "high_tail"):
        cfg["model"]["response_allocation"] = mode
        res = simulate_point(400.0, 0.25, n_iter=20000, seed=99, cfg=cfg)
        out[mode] = res
        # marginal preserved within a few percent (clipping bites at p_R=0.6)
        assert abs(res["achieved_response_share"] - 0.6) < 0.05, mode
    # Mean-benefit ordering: responses in the fast tail cut more expected harm
    # (severity nonlinearity), so mean deltaH is most negative for high_tail.
    assert out["high_tail"]["mean_deltaH"] < out["low_tail"]["mean_deltaH"]
    assert out["high_tail"]["mean_deltaH"] < out["independent"]["mean_deltaH"]
    # The benefit PROBABILITY is comparatively insensitive to allocation
    # (deployment cost co-varies with V0) — a reported finding, not a bug.
    ps = [out[m]["p_benefit"] for m in out]
    assert max(ps) - min(ps) < 0.05


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(0.5, 100)
    assert lo == pytest.approx(0.404, abs=2e-3)
    assert hi == pytest.approx(0.596, abs=2e-3)


def test_convergence_curves_shapes():
    x = np.random.default_rng(0).normal(size=5000)
    c = convergence_curves(x, interval=1000)
    assert list(c["n"]) == [1000, 2000, 3000, 4000, 5000]
    assert c["running_mean"][-1] == pytest.approx(x.mean(), rel=1e-12)
    assert c["running_p_benefit"][-1] == pytest.approx((x < 0).mean(), rel=1e-12)
