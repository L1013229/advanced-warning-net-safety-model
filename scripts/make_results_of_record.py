#!/usr/bin/env python3
"""Regenerate the results of record that the reproducibility gate tests against.

The results of record are the model's published-grid outputs at the current
released configuration. They are checked into `results/results_of_record.csv`
so that any later change to the code, the configuration, or the dependency
graph shows up as a diff against a number someone reported, rather than as a
silently different run.

This is deliberately NOT the same artefact as `tests/fixtures/v01_*.csv`.
Those pin the *previous* model version (v0.1) and exist to prove the rebuild
did not change the legacy answer. These pin the *current* one.

Usage:
    python scripts/make_results_of_record.py            # rewrite the CSV
    python scripts/make_results_of_record.py --check    # exit 1 if it would change

Regenerating is a deliberate act: it means the reported numbers changed, and
the accompanying commit has to say why.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aw_model.config import load_config  # noqa: E402
from aw_model.model import simulate_point  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
OUT = ROOT / "results" / "results_of_record.csv"

# One row per (case, grid point). Cases span the correlation-free path (pure
# elementwise NumPy, where the tightest tolerance applies) and the
# Iman-Conover path (which routes through LAPACK and therefore cannot be held
# to the same bound across BLAS backends -- see results/RESULTS_OF_RECORD.md).
CASES: dict[str, list[str]] = {
    "baseline": [],
    "highPR": ["scenarios/high_pr.yaml"],
    "baseline_fastDeploy": ["scenarios/fast_deploy.yaml"],
    "baseline_corr_plausible": ["correlation/plausible.yaml"],
}

COLUMNS = [
    "case",
    "Q_veh_h",
    "T_work_h",
    "mean_deltaH",
    "median_deltaH",
    "median_rel_deltaH",
    "p_benefit",
    "p_benefit_lo95",
    "p_benefit_hi95",
]


def run_case(case: str, overlays: list[str]) -> pd.DataFrame:
    cfg = load_config(CONFIG_DIR / "base.yaml", [CONFIG_DIR / o for o in overlays])
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
                    "mean_deltaH": res["mean_deltaH"],
                    "median_deltaH": res["median_deltaH"],
                    "median_rel_deltaH": res["median_rel_deltaH"],
                    "p_benefit": res["p_benefit"],
                    "p_benefit_lo95": res["p_benefit_lo95"],
                    "p_benefit_hi95": res["p_benefit_hi95"],
                }
            )
            idx += 1
    return pd.DataFrame(rows, columns=COLUMNS)


def build() -> pd.DataFrame:
    return pd.concat([run_case(c, o) for c, o in CASES.items()], ignore_index=True)


# Columns whose last bits are NOT fixed across microarchitectures, and the tolerance each is
# allowed. Same split the test suite declares: p_benefit and its bounds come from a counting
# path and are bit-exact; the deltaH family comes through LAPACK, where reduction order differs
# by CPU.
BIT_EXACT_COLUMNS = ("p_benefit", "p_benefit_lo95", "p_benefit_hi95")
TOLERANCE_COLUMNS = ("mean_deltaH", "median_deltaH", "median_rel_deltaH")
RTOL = 1e-9


def _check_against(fresh) -> int:
    """Compare a fresh build to the checked-in record, VALUE-wise rather than byte-wise.

    A raw text comparison demands that the record regenerate byte-identically on any machine.
    That is false by construction here: mean_deltaH and its siblings come through a LAPACK path
    whose reduction order depends on the CPU, so the last bits move between hosts.

    The consequence was not theoretical. This step was green on the branch that produced the
    record (2026-07-27) and red on main from the 2026-07-29 merge, with the SAME pinned
    interpreter and the same numpy 2.4.2 / pandas 3.0.1. The only difference was the runner's
    CPU. A reproducibility gate that can only pass on the machine that wrote the artifact is not
    checking reproducibility; it is checking that nothing moved.

    So: bit-exact where the suite says bit-exact, declared tolerance where it says tolerance, and
    a real failure -- a changed model, a changed config -- still fails, because it moves values
    far outside 1e-9.
    """
    import numpy as np
    import pandas as pd

    want = pd.read_csv(OUT, float_precision="round_trip")
    if len(want) != len(fresh):
        print(f"{OUT}: row count changed {len(want)} -> {len(fresh)}", file=sys.stderr)
        return 1
    if list(want.columns) != list(fresh.columns):
        print(f"{OUT}: columns changed", file=sys.stderr)
        return 1

    merged = fresh.merge(want, on=["case", "Q_veh_h", "T_work_h"], suffixes=("_now", "_rec"))
    if len(merged) != len(fresh):
        print(f"{OUT}: grid points did not line up", file=sys.stderr)
        return 1

    for col in BIT_EXACT_COLUMNS:
        bad = merged[merged[f"{col}_now"] != merged[f"{col}_rec"]]
        if not bad.empty:
            print(f"{OUT}: {col} is not bit-exact at {len(bad)} of {len(merged)} points",
                  file=sys.stderr)
            return 1

    for col in TOLERANCE_COLUMNS:
        try:
            np.testing.assert_allclose(merged[f"{col}_now"], merged[f"{col}_rec"],
                                       rtol=RTOL, atol=0)
        except AssertionError as exc:
            print(f"{OUT}: {col} exceeds rtol={RTOL:g}\n{exc}", file=sys.stderr)
            return 1

    print(f"{OUT}: up to date ({len(fresh)} rows; "
          f"{len(BIT_EXACT_COLUMNS)} bit-exact, {len(TOLERANCE_COLUMNS)} at rtol={RTOL:g})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the CSV would change")
    args = parser.parse_args(argv)

    # repr precision: full float64 round-trip, so the CSV is not itself a
    # rounding step. Double rounding through a low-precision intermediate is
    # exactly how a reported last digit drifts away from the pipeline.
    fresh = build()
    text = fresh.to_csv(index=False, float_format="%.17g")

    if args.check:
        if not OUT.exists():
            print(f"{OUT} does not exist", file=sys.stderr)
            return 1
        return _check_against(fresh)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(fresh)} rows, {fresh['case'].nunique()} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
