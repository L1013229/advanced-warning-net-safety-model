"""Sampler moment checks and correlation-induction tests."""
import numpy as np
import pytest

from aw_model.distributions import (
    DistributionError,
    build_spearman_matrix,
    iman_conover,
    sample_dist,
)


RNG = lambda: np.random.default_rng(42)
N = 200_000


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def test_fixed():
    x = sample_dist(RNG(), {"dist": "fixed", "value": 3.5}, 100, "x")
    assert np.all(x == 3.5)


def test_uniform_moments():
    x = sample_dist(RNG(), {"dist": "uniform", "min": 2.0, "max": 6.0}, N, "x")
    assert x.mean() == pytest.approx(4.0, abs=0.01)
    assert x.var() == pytest.approx(16.0 / 12.0, rel=0.02)
    assert x.min() >= 2.0 and x.max() <= 6.0


def test_loguniform_moments():
    a, b = 1e-7, 1e-5
    x = sample_dist(RNG(), {"dist": "loguniform", "min": a, "max": b}, N, "x")
    # log(x) ~ Uniform(log a, log b)
    lx = np.log(x)
    assert lx.mean() == pytest.approx((np.log(a) + np.log(b)) / 2, rel=1e-3)
    assert lx.var() == pytest.approx((np.log(b) - np.log(a)) ** 2 / 12.0, rel=0.02)


def test_normal_moments():
    x = sample_dist(RNG(), {"dist": "normal", "mean": 47.0, "sd": 3.0}, N, "x")
    assert x.mean() == pytest.approx(47.0, abs=0.03)
    assert x.std() == pytest.approx(3.0, rel=0.02)


def test_triangular_moments():
    lo, mode, hi = 0.9, 1.3, 1.7
    x = sample_dist(RNG(), {"dist": "triangular", "min": lo, "mode": mode, "max": hi}, N, "x")
    assert x.mean() == pytest.approx((lo + mode + hi) / 3.0, abs=0.005)
    var = (lo**2 + mode**2 + hi**2 - lo * mode - lo * hi - mode * hi) / 18.0
    assert x.var() == pytest.approx(var, rel=0.02)


def test_loguniform_rejects_nonpositive():
    with pytest.raises(DistributionError):
        sample_dist(RNG(), {"dist": "loguniform", "min": 0.0, "max": 1.0}, 10, "x")


# ---------------------------------------------------------------------------
# Iman-Conover
# ---------------------------------------------------------------------------

def test_iman_conover_hits_target_correlation():
    rng = RNG()
    n = 20_000
    samples = {
        "a": rng.normal(47, 3, n),
        "b": np.exp(rng.uniform(np.log(1e-7), np.log(1e-5), n)),
        "c": rng.uniform(0, 1, n),
    }
    names = ["a", "b", "c"]
    C = build_spearman_matrix(names, [["a", "b", 0.3], ["a", "c", -0.4]])
    out = iman_conover(samples, names, C, seed=123)

    assert _spearman(out["a"], out["b"]) == pytest.approx(0.3, abs=0.02)
    assert _spearman(out["a"], out["c"]) == pytest.approx(-0.4, abs=0.02)
    assert abs(_spearman(out["b"], out["c"])) < 0.03  # unspecified stays ~0


def test_iman_conover_preserves_marginals_exactly():
    rng = RNG()
    n = 5000
    samples = {"a": rng.normal(0, 1, n), "b": rng.uniform(0, 1, n)}
    C = build_spearman_matrix(["a", "b"], [["a", "b", 0.5]])
    out = iman_conover(samples, ["a", "b"], C, seed=5)
    # Same multiset of values, different order
    np.testing.assert_allclose(np.sort(out["a"]), np.sort(samples["a"]))
    np.testing.assert_allclose(np.sort(out["b"]), np.sort(samples["b"]))
    assert not np.array_equal(out["a"], samples["a"])


def test_infeasible_matrix_rejected():
    with pytest.raises(DistributionError):
        build_spearman_matrix(["a", "b", "c"],
                              [["a", "b", 0.9], ["b", "c", 0.9], ["a", "c", -0.9]])


def test_correlation_propagates_to_effective_speed():
    """mu_v pairs must reach V0 (the speed used downstream), diluted by sigma_v."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from aw_model.config import load_config
    from aw_model.model import simulate_point
    cfg = load_config(Path(__file__).parent.parent / "config" / "base.yaml")
    cfg["correlation"] = {"pairs": [["mu_v_kmh", "r_E_per_veh_km", 0.3]]}
    res = simulate_point(400.0, 0.25, 20000, 123, cfg)
    ac = res["achieved_correlations"]
    assert ac["mu_v_kmh~r_E_per_veh_km"]["achieved"] == pytest.approx(0.3, abs=0.03)
    # V0 = mu_v + sigma_v*z: mu_v carries ~sqrt(9/(9+E[sigma^2])) of the rank signal
    assert 0.05 < ac["V0_kmh~r_E_per_veh_km"]["achieved"] < 0.20
