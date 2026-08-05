"""Reproducibility gate on the CURRENT model's results of record.

`tests/test_reproduction.py` pins the *previous* model version: it proves the
rebuild did not change the v0.1 answer. Nothing pinned the answer this
version reports -- so a change to the code, a config value, or the dependency
graph could move a published number with no test going red. That is the exact
failure this file closes.

TOLERANCE POLICY (declared, not implied)

  p_benefit          BIT-EXACT (==). It is k/n_iter for an integer k, so an
                     identical Monte Carlo stream gives an identical float.
                     Any perturbation that moves one draw across zero moves k.
  p_benefit_lo95     BIT-EXACT. Pure arithmetic on p_benefit and n_iter.
  p_benefit_hi95
  mean_deltaH        rtol 1e-12. Correlation-free cases run through
  median_deltaH      elementwise NumPy only (exp/log/clip); identical inputs
  median_rel_deltaH  give identical bits on one build, and 1e-12 absorbs
                     last-ulp differences in libm across NumPy builds without
                     coming close to any digit the paper reports.

  The `*_corr_*` case is the exception and is held to rtol 1e-9. Iman-Conover
  goes through np.linalg.cholesky and np.corrcoef, i.e. the system
  BLAS/LAPACK, whose last bits legitimately differ across backends and thread
  counts. Claiming 1e-12 there would be a tolerance we cannot honour, which
  is worse than declaring the looser one.

Both bounds are many orders of magnitude tighter than the last digit of any
reported value (Table 3 prints 3 decimal places on a value of order 1e-2), so
a real change in a reported number cannot hide inside either.
"""
from __future__ import annotations

from pathlib import Path

import io

import numpy as np
import pandas as pd
import pytest

from aw_model.config import load_config
from aw_model.model import simulate_point

HERE = Path(__file__).parent
ROOT = HERE.parent
CONFIG_DIR = ROOT / "config"
RECORD = ROOT / "results" / "results_of_record.csv"

CASES: dict[str, list[str]] = {
    "baseline": [],
    "highPR": ["scenarios/high_pr.yaml"],
    "baseline_fastDeploy": ["scenarios/fast_deploy.yaml"],
    "baseline_corr_plausible": ["correlation/plausible.yaml"],
}

BIT_EXACT_COLUMNS = ("p_benefit", "p_benefit_lo95", "p_benefit_hi95")
TOLERANCE_COLUMNS = ("mean_deltaH", "median_deltaH", "median_rel_deltaH")

# Declared tolerances, by path through the numerics.
RTOL_ELEMENTWISE = 1e-12
RTOL_LAPACK = 1e-9


def rtol_for(case: str) -> float:
    return RTOL_LAPACK if "_corr_" in case else RTOL_ELEMENTWISE


def run_case(case: str, overlays: list[str]) -> pd.DataFrame:
    cfg = load_config(CONFIG_DIR / "base.yaml", [CONFIG_DIR / o for o in overlays])
    return run_cfg(case, cfg)


def run_cfg(case: str, cfg: dict) -> pd.DataFrame:
    seed0 = int(cfg["monte_carlo"]["seed"])
    n_iter = int(cfg["monte_carlo"]["n_iter"])
    rows, idx = [], 0
    for q in cfg["grid"]["Q_veh_h"]:
        for t in cfg["grid"]["T_work_h"]:
            res = simulate_point(float(q), float(t), n_iter, seed0 + idx, cfg)
            rows.append(
                {
                    "case": case,
                    "Q_veh_h": res["Q_veh_h"],
                    "T_work_h": res["T_work_h"],
                    **{k: res[k] for k in TOLERANCE_COLUMNS + BIT_EXACT_COLUMNS},
                }
            )
            idx += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def record() -> pd.DataFrame:
    assert RECORD.is_file(), (
        f"{RECORD} is missing. Regenerate with scripts/make_results_of_record.py -- "
        "a reproducibility gate with no reference reproduces nothing."
    )
    # float_precision="round_trip" is load-bearing, not decoration: pandas'
    # default C parser is not correctly rounded, and reads 0.1483 back as
    # 0.1482999999999999. A gate that compared against a mis-parsed reference
    # would fail on unchanged code -- or, with a loose tolerance, would hide a
    # real change behind its own parse error.
    return pd.read_csv(RECORD, float_precision="round_trip")


def test_record_covers_every_declared_case(record: pd.DataFrame) -> None:
    assert set(record["case"]) == set(CASES)
    for case in CASES:
        assert len(record[record["case"] == case]) == 49


@pytest.mark.parametrize("case", sorted(CASES))
def test_current_code_reproduces_the_results_of_record(case: str, record: pd.DataFrame) -> None:
    got = run_case(case, CASES[case])
    want = record[record["case"] == case]
    merged = got.merge(want, on=["Q_veh_h", "T_work_h"], suffixes=("_now", "_rec"))
    assert len(merged) == 49, f"{case}: grid points did not line up"

    for col in BIT_EXACT_COLUMNS:
        mismatched = merged[merged[f"{col}_now"] != merged[f"{col}_rec"]]
        assert mismatched.empty, (
            f"{case}: {col} is not bit-exact against the results of record at "
            f"{len(mismatched)} of 49 grid points (first: "
            f"Q={mismatched.iloc[0]['Q_veh_h']}, T={mismatched.iloc[0]['T_work_h']}, "
            f"now={mismatched.iloc[0][f'{col}_now']!r}, "
            f"record={mismatched.iloc[0][f'{col}_rec']!r})"
        )

    rtol = rtol_for(case)
    for col in TOLERANCE_COLUMNS:
        np.testing.assert_allclose(
            merged[f"{col}_now"],
            merged[f"{col}_rec"],
            rtol=rtol,
            atol=0,
            err_msg=f"{case}: {col} exceeds the declared tolerance rtol={rtol:g}",
        )


def test_the_record_is_not_itself_a_rounding_step(record: pd.DataFrame) -> None:
    """The CSV must round-trip float64 exactly.

    Storing the reference at reduced precision would make the gate compare a
    rounded number to a full one -- the same double-rounding shape that put a
    wrong last digit into a reported table in the first place.

    THIS TESTS STORAGE, AND ONLY STORAGE.

    The previous implementation re-ran the model and demanded `==` against the
    record for TOLERANCE_COLUMNS as well as BIT_EXACT_COLUMNS. That conflated two
    different properties and contradicted the suite's own vocabulary: mean_deltaH
    and median_deltaH are declared TOLERANCE columns precisely because they come
    through a LAPACK path whose last bits are not fixed across microarchitectures,
    and the sibling test checks them at rtol 1e-9.

    So it passed on the machine the record was generated on and failed everywhere
    else. Green on branch 2026-07-27, red on main from 2026-07-29 with the same
    pinned interpreter and the same numpy 2.4.2 / pandas 3.0.1 -- the difference was
    the CPU, not the code. A reproducibility gate that can only pass on one host is
    not a reproducibility gate.

    Round-tripping the record through CSV and back tests the stated property with no
    arithmetic in the way, so it holds on any machine. Bit-exactness where it IS
    required is unchanged: test_current_code_reproduces_the_results_of_record still
    demands `!=`-free equality on BIT_EXACT_COLUMNS.
    """
    buffer = io.StringIO()
    record.to_csv(buffer, index=False)
    reread = pd.read_csv(io.StringIO(buffer.getvalue()), float_precision="round_trip")
    for col in TOLERANCE_COLUMNS + BIT_EXACT_COLUMNS:
        assert (record[col] == reread[col]).all(), (
            f"{col}: the checked-in record does not round-trip float64 exactly -- "
            f"the stored text loses bits, so the gate would compare a rounded number "
            f"to a full one"
        )


# ---------------------------------------------------------------------------
# The gate must be able to FAIL. A reproducibility test only ever run against
# unchanged inputs proves nothing about its own sensitivity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "knob,factor",
    [
        ("c_work_m", 1.000001),      # 1 part per million on a lateral offset
        ("r_E_per_veh_km", 1.0001),  # 1 part in ten thousand on the encroachment rate
    ],
)
def test_a_perturbed_input_is_detected(knob: str, factor: float, record: pd.DataFrame) -> None:
    """Perturb one input by a hair; the gate must report a difference."""
    cfg = load_config(CONFIG_DIR / "base.yaml")
    spec = cfg["distributions"][knob]
    for key in ("value", "min", "mode", "max", "mean"):
        if key in spec:
            spec[key] = float(spec[key]) * factor

    got = run_cfg("baseline", cfg)
    want = record[record["case"] == "baseline"]
    merged = got.merge(want, on=["Q_veh_h", "T_work_h"], suffixes=("_now", "_rec"))

    within_tolerance = all(
        np.allclose(merged[f"{c}_now"], merged[f"{c}_rec"], rtol=RTOL_ELEMENTWISE, atol=0)
        for c in TOLERANCE_COLUMNS
    )
    assert not within_tolerance, (
        f"perturbing {knob} by x{factor} left every result inside the declared "
        f"tolerance -- the gate cannot detect a changed input"
    )


def test_a_perturbed_seed_is_detected(record: pd.DataFrame) -> None:
    """Changing the seed must move the answer.

    If it did not, the recorded numbers would not be the Monte Carlo estimate
    they claim to be.
    """
    cfg = load_config(CONFIG_DIR / "base.yaml")
    cfg["monte_carlo"]["seed"] = int(cfg["monte_carlo"]["seed"]) + 1

    got = run_cfg("baseline", cfg)
    want = record[record["case"] == "baseline"]
    merged = got.merge(want, on=["Q_veh_h", "T_work_h"], suffixes=("_now", "_rec"))
    assert not (merged["p_benefit_now"] == merged["p_benefit_rec"]).all(), (
        "a different seed produced identical p_benefit at all 49 grid points"
    )


def test_repeated_runs_are_identical() -> None:
    """Determinism, stated as a test rather than assumed."""
    first = run_case("baseline", [])
    second = run_case("baseline", [])
    pd.testing.assert_frame_equal(first, second, check_exact=True)
