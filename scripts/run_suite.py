#!/usr/bin/env python3
"""Run the full manuscript analysis suite.

Phases (all on by default; select with --phases):
  primary      4 assumption sets x 49-point grid (baseline, fastDeploy, highPR, both)
  correlation  independence robustness: plausible + stress structures on baseline & highPR
  tail         response-allocation robustness (low_tail / high_tail) on baseline & highPR
  ke           k_E encroachment-reduction sweep (0.50..0.95) on baseline settings
  sensitivity  full-trace runs at Q=400 x {3 min, 15 min, 2 h} + PRCC with bootstrap CIs
  convergence  running-mean / running-P curves for representative points
  validation   deterministic hand-calculation comparison table
  severity     harm-definition robustness (requires config/severity/*.yaml overlays
               with verified curve coefficients; skipped with a notice if absent)

Every run writes grid_results.csv + config_used.yaml + run_metadata.json under
outputs/<tag>/ so any number in the manuscript traces to an exact config.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from aw_model.config import load_config, config_hash  # noqa: E402
from aw_model.io import write_run_outputs, write_trace  # noqa: E402
from aw_model.model import simulate_point, convergence_curves, wilson_ci  # noqa: E402
from aw_model.sensitivity import prcc_bootstrap, DEFAULT_PRCC_INPUTS  # noqa: E402

CONFIG = ROOT / "config"
OUT = ROOT / "outputs"

SUMMARY_COLS = ["Q_veh_h", "T_work_h", "mean_deltaH", "median_deltaH", "p_benefit",
                "p_benefit_lo95", "p_benefit_hi95", "mean_H0", "median_H0",
                "median_rel_deltaH", "achieved_response_share", "target_response_share"]


def run_grid(cfg: dict, tag: str | None = None, keep_deltaH: bool = False) -> pd.DataFrame:
    tag = tag or cfg["meta"]["tag"]
    seed0 = int(cfg["monte_carlo"]["seed"])
    n_iter = int(cfg["monte_carlo"]["n_iter"])
    rows, arrays, idx = [], {}, 0
    t0 = time.time()
    for Q in cfg["grid"]["Q_veh_h"]:
        for T in cfg["grid"]["T_work_h"]:
            res = simulate_point(float(Q), float(T), n_iter, seed0 + idx, cfg)
            rows.append({k: res[k] for k in SUMMARY_COLS})
            if keep_deltaH:
                arrays[f"dH_Q{Q}_T{T}"] = res["deltaH"]
            idx += 1
    df = pd.DataFrame(rows)
    out_dir = write_run_outputs(OUT / tag, cfg, df, {"runtime_s": round(time.time() - t0, 2)})
    if keep_deltaH:
        np.savez_compressed(out_dir / "deltaH_arrays.npz", **arrays)
    print(f"  [{tag}] 49 points in {time.time()-t0:.1f}s  "
          f"p_benefit {df.p_benefit.min():.4f}..{df.p_benefit.max():.4f}")
    return df


def phase_primary() -> None:
    print("== primary ==")
    run_grid(load_config(CONFIG / "base.yaml"), keep_deltaH=True)
    run_grid(load_config(CONFIG / "base.yaml", [CONFIG / "scenarios/fast_deploy.yaml"]), keep_deltaH=True)
    run_grid(load_config(CONFIG / "base.yaml", [CONFIG / "scenarios/high_pr.yaml"]), keep_deltaH=True)
    run_grid(load_config(CONFIG / "base.yaml", [CONFIG / "scenarios/high_pr_fast_deploy.yaml"]), keep_deltaH=True)


def phase_correlation() -> None:
    print("== correlation ==")
    for scen, sname in [(None, "baseline"), ("scenarios/high_pr.yaml", "highPR")]:
        for corr in ("plausible", "stress", "behavioural"):
            if sname == "highPR" and corr == "behavioural":
                continue  # p_R is fixed under highPR; a p_R correlation is undefined
            corr_file = corr
            if sname == "highPR" and corr == "stress":
                corr_file = "stress_fixed_pr"  # stress minus the p_R pair (undefined for fixed p_R)
            overlays = ([CONFIG / scen] if scen else []) + [CONFIG / f"correlation/{corr_file}.yaml"]
            cfg = load_config(CONFIG / "base.yaml", overlays)
            cfg["meta"]["tag"] = f"{sname}_corr_{corr}"
            run_grid(cfg)
            # Audit trail: achieved rank correlations at a representative point
            probe = simulate_point(400.0, 0.25, int(cfg["monte_carlo"]["n_iter"]),
                                   int(cfg["monte_carlo"]["seed"]) + 23, cfg)
            (OUT / cfg["meta"]["tag"] / "achieved_correlations.json").write_text(
                json.dumps(probe["achieved_correlations"], indent=1))


def phase_tail() -> None:
    print("== tail allocation ==")
    for scen, sname in [(None, "baseline"), ("scenarios/high_pr.yaml", "highPR")]:
        for mode in ("low_tail", "high_tail"):
            cfg = load_config(CONFIG / "base.yaml", [CONFIG / scen] if scen else [])
            cfg["model"]["response_allocation"] = mode
            cfg["meta"]["tag"] = f"{sname}_alloc_{mode}"
            run_grid(cfg)


def phase_ke() -> None:
    print("== k_E sweep ==")
    for ke in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        cfg = load_config(CONFIG / "base.yaml")
        cfg["model"]["allow_frequency_pathway"] = True
        cfg["distributions"]["k_E"] = {"dist": "fixed", "value": ke}
        cfg["meta"]["tag"] = f"baseline_KE{str(ke).replace('.', 'p')}"
        run_grid(cfg)


def phase_sensitivity() -> None:
    print("== sensitivity (PRCC + bootstrap) ==")
    cfg = load_config(CONFIG / "base.yaml")
    seed0 = int(cfg["monte_carlo"]["seed"])
    n_iter = int(cfg["monte_carlo"]["n_iter"])
    Q_list, T_list = cfg["grid"]["Q_veh_h"], cfg["grid"]["T_work_h"]
    frames = []
    frames_g = []
    for T in (0.05, 0.25, 2.0):
        idx = Q_list.index(400) * len(T_list) + T_list.index(T)
        res = simulate_point(400.0, float(T), n_iter, seed0 + idx, cfg, full_trace=True)
        tr = {**res["trace"]}
        write_trace(OUT / "sensitivity", f"trace_Q400_T{str(T).replace('.', 'p')}", tr)
        df = prcc_bootstrap(tr, DEFAULT_PRCC_INPUTS, n_boot=1000, subsample=5000)
        df["T_work_h"] = T
        frames.append(df)
        # Decision-scale sensitivity: every harm term is proportional to r_E, so
        # sign(deltaH) is invariant to it. PRCC against G = deltaH / r_E isolates
        # the inputs that drive the deploy/omit DECISION rather than the harm scale.
        tr["G_decision"] = tr["deltaH"] / tr["r_E0"]
        gx = [c for c in DEFAULT_PRCC_INPUTS if c != "r_E_per_veh_km"]
        dg = prcc_bootstrap(tr, gx, y_col="G_decision", n_boot=1000, subsample=5000)
        dg["T_work_h"] = T
        frames_g.append(dg)
        print(f"  T={T}: top3 {list(df.head(3).parameter)} | decision-scale top3 {list(dg.head(3).parameter)}")
    (OUT / "sensitivity").mkdir(parents=True, exist_ok=True)
    for fr, name in ((frames, "prcc"), (frames_g, "prcc_decision_scale")):
        allp = pd.concat(fr, ignore_index=True)
        pivot = allp.pivot_table(index="parameter", columns="T_work_h", values="PRCC")
        pivot["mean_abs_PRCC"] = pivot.abs().mean(axis=1)
        pivot = pivot.sort_values("mean_abs_PRCC", ascending=False)
        allp.to_csv(OUT / "sensitivity" / f"{name}_bootstrap.csv", index=False)
        pivot.to_csv(OUT / "sensitivity" / f"{name}_summary.csv")


def phase_convergence() -> None:
    print("== convergence ==")
    cfg = load_config(CONFIG / "base.yaml")
    seed0 = int(cfg["monte_carlo"]["seed"])
    Q_list, T_list = cfg["grid"]["Q_veh_h"], cfg["grid"]["T_work_h"]
    out = OUT / "convergence"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (Q, T) in [(400, 0.05), (400, 0.25), (400, 2.0), (50, 0.05), (1000, 4.0)]:
        idx = Q_list.index(Q) * len(T_list) + T_list.index(T)
        # 50k iterations to show stabilisation well beyond the production 20k
        res = simulate_point(float(Q), float(T), 50_000, seed0 + idx, cfg)
        c = convergence_curves(res["deltaH"], interval=1000)
        pd.DataFrame(c).to_csv(out / f"convergence_Q{Q}_T{str(T).replace('.', 'p')}.csv", index=False)
        p20, p50 = c["running_p_benefit"][19], c["running_p_benefit"][-1]
        rows.append({"Q": Q, "T": T, "p_at_20k": p20, "p_at_50k": p50, "abs_drift": abs(p50 - p20)})
    pd.DataFrame(rows).to_csv(out / "convergence_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def phase_validation() -> None:
    """Deterministic hand-calculation vs code, written as a table for the supplement."""
    print("== validation ==")
    cfg = load_config(CONFIG / "base.yaml")
    fixed = {
        "d_aw_m": 50.0, "v_walk_m_s": 1.25, "t_handle_s": 20.0,
        "mu_v_kmh": 50.0, "sigma_v_kmh": 0.0, "v_cap_kmh": 80.0,
        "p_R": 0.0, "deltaV_kmh": 10.0, "a_comfort_m_s2": 2.0,
        "r_E_per_veh_km": 1e-6, "k_rE": 1.0, "alpha_lat_m_inv": 0.10,
        "c_work_m": 2.0, "c_deploy_m": 1.0, "k_E": 1.0,
        "L_worker_m": 10.0, "L_lcv_m": 6.0, "L_worker_deploy_m": 1.0,
        "m_striking_kg": 1500.0, "m_lcv_kg": 2500.0,
    }
    for k, v in fixed.items():
        cfg["distributions"][k] = {"dist": "fixed", "value": v}

    Q, T = 400.0, 0.25
    # Hand calculation (V0 = 50 exactly in the limit sd->0; code clamps sd to 0.1)
    T_dep_h = (2 * (2 * 50.0 / 1.25 + 20.0)) / 3600.0
    p_w50 = 1 / (1 + math.exp(4.6 - 0.078 * 50.0))
    dv = 50.0 * 2500.0 / (1500.0 + 2500.0)
    # Occupant curve = Wang (2022) all-crashes MAIS3+: z = -6.9540 + 0.1637*dv_mph
    p_o = 1 / (1 + math.exp(-(0.1637 * (dv / 1.609344) - 6.9540)))
    lam_w = Q * T * 1e-6 * (10.0 / 1000.0) * math.exp(-0.10 * 2.0)
    lam_v = Q * T * 1e-6 * (6.0 / 1000.0) * math.exp(-0.10 * 2.0)
    lam_d = Q * T_dep_h * 1e-6 * (1.0 / 1000.0) * math.exp(-0.10 * 1.0)
    H0 = lam_w * p_w50 + lam_v * p_o
    H1 = lam_w * p_w50 + lam_v * p_o + lam_d * p_w50   # p_R=0 -> V1=V0
    hand = {"T_deploy_s": T_dep_h * 3600, "lambda_w0": lam_w, "lambda_v0": lam_v,
            "lambda_d": lam_d, "p_w(V0)": p_w50, "p_o(dv0)": p_o,
            "H0": H0, "H1": H1, "deltaH": H1 - H0}

    res = simulate_point(Q, T, 2000, 42, cfg, full_trace=True)
    tr = res["trace"]
    code = {"T_deploy_s": float(tr["T_deploy_s"].mean()), "lambda_w0": float(tr["lambda_w0"].mean()),
            "lambda_v0": float(tr["lambda_v0"].mean()), "lambda_d": float(tr["lambda_d"].mean()),
            "p_w(V0)": float(tr["p_w0"].mean()), "p_o(dv0)": float(tr["p_o0"].mean()),
            "H0": float(tr["H0"].mean()), "H1": float(tr["H1"].mean()), "deltaH": float(tr["deltaH"].mean())}

    rows = [{"quantity": k, "hand_calc": hand[k], "model": code[k],
             "rel_diff": abs(code[k] - hand[k]) / abs(hand[k])} for k in hand]
    df = pd.DataFrame(rows)
    out = OUT / "validation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "deterministic_check.csv", index=False)
    worst = df.rel_diff.max()
    print(df.to_string(index=False))
    print(f"  worst relative difference: {worst:.2e}")
    if worst > 5e-3:
        raise SystemExit("VALIDATION FAILURE: hand calculation and model disagree")


def phase_severity() -> None:
    print("== severity robustness ==")
    sev_dir = CONFIG / "severity"
    overlays = sorted(sev_dir.glob("*.yaml")) if sev_dir.exists() else []
    if not overlays:
        print("  NOTE: no config/severity/*.yaml overlays present (fatality-curve "
              "coefficients pending verification) — skipped.")
        return
    for ov in overlays:
        for scen, sname in [(None, "baseline"), ("scenarios/high_pr.yaml", "highPR")]:
            cfg = load_config(CONFIG / "base.yaml", ([CONFIG / scen] if scen else []) + [ov])
            cfg["meta"]["tag"] = f"{sname}_sev_{ov.stem}"
            run_grid(cfg)


PHASES = {"primary": phase_primary, "correlation": phase_correlation, "tail": phase_tail,
          "ke": phase_ke, "sensitivity": phase_sensitivity, "convergence": phase_convergence,
          "validation": phase_validation, "severity": phase_severity}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", choices=list(PHASES), default=list(PHASES))
    args = ap.parse_args()
    t0 = time.time()
    for name in args.phases:
        PHASES[name]()
    print(f"\nSuite complete in {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
