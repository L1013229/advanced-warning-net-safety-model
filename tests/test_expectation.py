"""Job-expectation mode (v1.5): identities and agreement with brute-force inner sampling."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aw_model.config import load_config  # noqa: E402
from aw_model.model import simulate_point  # noqa: E402
from aw_model.risk_functions import p_worker, p_occupant  # noqa: E402

CONFIG = ROOT / "config"


def _cfg(**model_overrides):
    cfg = load_config(CONFIG / "base.yaml")
    cfg["model"]["iteration"] = "job_expectation"
    cfg["model"].update(model_overrides)
    return cfg


def test_zero_response_gives_deployment_harm_only():
    cfg = _cfg()
    cfg["distributions"]["p_R"] = {"dist": "fixed", "value": 0.0}
    res = simulate_point(400.0, 0.25, 2000, 7, cfg, full_trace=True)
    tr = res["trace"]
    np.testing.assert_allclose(tr["deltaH"], tr["lambda_d"] * tr["p_w0"], rtol=1e-12)
    assert res["p_benefit"] == 0.0


def test_delta_h_is_linear_in_response_share():
    """deltaH(p) = C - p B: the value at p=0.3 must equal the linear prediction from p=0 and p=0.1."""
    out = {}
    for p in (0.0, 0.1, 0.3):
        cfg = _cfg()
        cfg["distributions"]["p_R"] = {"dist": "fixed", "value": p}
        out[p] = simulate_point(400.0, 1.0, 1500, 11, cfg, full_trace=True)["trace"]["deltaH"]
    pred = out[0.0] + 3.0 * (out[0.1] - out[0.0])
    np.testing.assert_allclose(out[0.3], pred, rtol=1e-9, atol=1e-30)


@pytest.mark.parametrize("T", [0.05, 0.25, 2.0])
def test_quadrature_matches_brute_force_inner_sampling(T):
    cfg = _cfg()
    res = simulate_point(400.0, T, 300, 3, cfg, full_trace=True)
    tr = res["trace"]
    rng = np.random.default_rng(99)
    N = 100_000
    rel = []
    for i in range(0, 300, 30):
        V = np.clip(rng.normal(tr["mu_v_kmh"][i], max(tr["sigma_v_kmh"][i], 0.1), N), 0, tr["v_cap_kmh"][i])
        m1 = rng.uniform(1200, 2000, N)
        V_ms = V / 3.6
        Vr = np.sqrt(np.maximum(V_ms**2 - 2.0 * max(tr["a_brake_m_s2"][i], 0.1) * V_ms * tr["dt_react_saved_s"][i], 0.0)) * 3.6
        ratio = tr["m_lcv_kg"][i] / (m1 + tr["m_lcv_kg"][i])
        p = tr["p_R"][i]
        pw1 = (1 - p) * p_worker(V).mean() + p * p_worker(Vr).mean()
        po1 = (1 - p) * p_occupant(V * ratio).mean() + p * p_occupant(Vr * ratio).mean()
        H1 = tr["lambda_w1"][i] * pw1 + tr["lambda_v1"][i] * po1 + tr["lambda_d"][i] * p_worker(V).mean()
        H0 = tr["lambda_w0"][i] * p_worker(V).mean() + tr["lambda_v0"][i] * p_occupant(V * ratio).mean()
        rel.append(abs((H1 - H0) - tr["deltaH"][i]) / max(abs(tr["H0"][i]), 1e-30))
    assert max(rel) < 2e-2, rel


def test_job_expectation_is_the_default_and_the_overlay_restores_the_old_mode():
    cfg = load_config(CONFIG / "base.yaml")
    assert cfg["model"]["iteration"] == "job_expectation"
    old = load_config(CONFIG / "base.yaml", [CONFIG / "scenarios/representative_vehicle.yaml"])
    assert old["model"]["iteration"] == "representative_vehicle"
    # the per-encounter construction cannot exceed the mean response share (0.15 under Uniform(0, 0.30))
    res = simulate_point(400.0, 0.25, 2000, 5, old)
    assert 0.0 <= res["p_benefit"] <= 0.20
