"""PRCC sanity tests on synthetic data with known structure."""
import numpy as np
import pytest

from aw_model.sensitivity import prcc, prcc_bootstrap


def _synthetic(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)          # strong positive driver
    x2 = rng.normal(0, 1, n)           # moderate negative driver
    x3 = rng.uniform(0, 1, n)          # pure noise
    y = 5.0 * x1 - 1.0 * x2 + rng.normal(0, 0.5, n)
    return {"x1": x1, "x2": x2, "x3": x3, "deltaH": y}


def test_prcc_recovers_structure():
    d = _synthetic()
    df = prcc(d, ["x1", "x2", "x3"])
    r = df.set_index("parameter")["PRCC"]
    assert r["x1"] > 0.9
    assert r["x2"] < -0.5
    assert abs(r["x3"]) < 0.03
    # Ranking: x1 strongest
    assert df.iloc[0]["parameter"] == "x1"


def test_prcc_bootstrap_cis_bracket_point():
    d = _synthetic(n=5000)
    df = prcc_bootstrap(d, ["x1", "x2", "x3"], n_boot=200, subsample=2000)
    for _, row in df.iterrows():
        assert row["ci_lo"] <= row["PRCC"] <= row["ci_hi"]
    noise = df.set_index("parameter").loc["x3"]
    assert noise["ci_lo"] < 0 < noise["ci_hi"]  # noise CI straddles zero
