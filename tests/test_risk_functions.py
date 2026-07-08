"""Severity-curve checks against hand-computed benchmark values."""
import math

import numpy as np
import pytest

from aw_model.risk_functions import p_occupant, p_worker


def _logit(a, b, v):
    return 1.0 / (1.0 + math.exp(-(a + b * v)))


@pytest.mark.parametrize("v", [30.0, 50.0, 70.0])
def test_worker_rosen_benchmarks(v):
    expect = _logit(-4.6, 0.078, v)
    assert p_worker(np.array([v]))[0] == pytest.approx(expect, rel=1e-12)


def test_worker_rosen_hand_values():
    # Hand-computed anchors: p(30)=0.0946.., p(50)=0.3318.., p(70)=0.7027..
    p = p_worker(np.array([30.0, 50.0, 70.0]))
    assert p[0] == pytest.approx(1 / (1 + math.exp(4.6 - 2.34)), rel=1e-9)
    assert p[1] == pytest.approx(0.33181, abs=1e-4)
    assert p[2] == pytest.approx(0.70274, abs=1e-4)


def test_occupant_kahane_mph_conversion():
    # Legacy v0.1 curve (reproduction gate only): dV = 30 mph -> z = -1.6577
    dv_kmh = 30.0 * 1.609344
    expect = 1.0 / (1.0 + math.exp(1.6577))
    assert p_occupant(np.array([dv_kmh]), "kahane_mais3plus")[0] == pytest.approx(expect, rel=1e-6)


def test_monotonicity_and_bounds():
    v = np.linspace(0, 120, 200)
    pw = p_worker(v)
    po = p_occupant(v)
    assert np.all(np.diff(pw) > 0) and np.all(np.diff(po) > 0)
    assert pw.min() >= 0 and pw.max() <= 1 and po.min() >= 0 and po.max() <= 1


def test_custom_curve_spec():
    # Custom logistic in km/h: a=-7, b=0.1 at 70 km/h -> p=0.5
    p = p_worker(np.array([70.0]), {"a": -7.0, "b": 0.1, "speed_unit": "kmh"})
    assert p[0] == pytest.approx(0.5, rel=1e-12)


def test_unknown_named_curve_rejected():
    with pytest.raises(ValueError):
        p_worker(np.array([50.0]), "no_such_curve")


def test_occupant_wang_benchmarks():
    # Wang (2022) Table 4-2 anchor: at dV=30 mph, MAIS3+ p = e^(-6.954+0.1637*30)/(1+...)
    dv_kmh = 30.0 * 1.609344
    z = -6.9540 + 0.1637 * 30.0
    expect = math.exp(z) / (1 + math.exp(z))
    assert p_occupant(np.array([dv_kmh]), "wang_mais3plus_all")[0] == pytest.approx(expect, rel=1e-9)


def test_worker_fatality_below_severe():
    # Fatality risk must sit below severe-injury risk at any speed
    v = np.linspace(5, 100, 50)
    from aw_model.risk_functions import p_worker as pw
    assert np.all(pw(v, "rosen_fatality") < pw(v, "rosen_mais3plus"))


def test_occupant_fatality_below_mais3():
    v = np.linspace(5, 100, 50)
    assert np.all(p_occupant(v, "wang_fatality_all") < p_occupant(v, "wang_mais3plus_all"))
