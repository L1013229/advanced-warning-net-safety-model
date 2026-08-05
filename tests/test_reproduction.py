"""Reproduction gate: the rebuilt model must reproduce v0.1 outputs bit-for-bit.

The fixtures are the actual grid_results.csv files produced by the v0.1 code
(December 2025). Same seeds, same sampling order, correlation disabled ->
identical mean/median/P(deltaH<0) at every one of the 49 grid points.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aw_model.config import load_config
from aw_model.model import simulate_point

HERE = Path(__file__).parent
CONFIG_DIR = HERE.parent / "config"


def _run_grid(cfg):
    seed0 = int(cfg["monte_carlo"]["seed"])
    n_iter = int(cfg["monte_carlo"]["n_iter"])
    rows, idx = [], 0
    for Q in cfg["grid"]["Q_veh_h"]:
        for T in cfg["grid"]["T_work_h"]:
            res = simulate_point(float(Q), float(T), n_iter, seed0 + idx, cfg)
            rows.append({k: res[k] for k in ("Q_veh_h", "T_work_h", "mean_deltaH", "median_deltaH", "p_benefit")})
            idx += 1
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fixture,overlays", [
    ("v01_baseline_grid_results.csv", []),
    ("v01_highPR_grid_results.csv", ["scenarios/high_pr.yaml"]),
])
def test_reproduces_v01_grid(fixture, overlays):
    cfg = load_config(CONFIG_DIR / "base.yaml", [CONFIG_DIR / o for o in overlays])
    # v0.1 used the legacy occupant curve (unverifiable provenance; superseded
    # by Wang 2022 in production). Reproduction must use what v0.1 used.
    cfg["severity"]["occupant_curve"] = "kahane_mais3plus"
    got = _run_grid(cfg)
    # float_precision="round_trip" is not optional here. pandas' default CSV float
    # parser is fast rather than correctly rounded, so reading a full-precision
    # fixture silently shifts the last bits -- measured on the results-of-record CSV,
    # 93 of its values changed on a plain read/write cycle and 0 changed with this
    # flag. Comparing at rtol 1e-12 against a value the reader itself perturbed is
    # comparing to the wrong number.
    want = pd.read_csv(HERE / "fixtures" / fixture, float_precision="round_trip")

    merged = got.merge(want, on=["Q_veh_h", "T_work_h"], suffixes=("_new", "_v01"))
    assert len(merged) == 49

    for col in ("mean_deltaH", "median_deltaH", "p_benefit"):
        np.testing.assert_allclose(
            merged[f"{col}_new"], merged[f"{col}_v01"], rtol=1e-12, atol=0,
            err_msg=f"{fixture}: {col} does not reproduce v0.1",
        )
