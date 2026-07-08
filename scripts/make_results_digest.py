#!/usr/bin/env python3
"""Assemble every number the manuscript quotes into one machine-generated digest.

Reads outputs/<tag>/grid_results.csv (+ sensitivity/convergence/validation
artifacts) and writes outputs/results_digest.json + a human-readable
outputs/results_digest.md. The manuscript text/tables are built from these
values only — no hand transcription.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

CASES = {
    "baseline": "(a) standard response, standard deployment",
    "baseline_fastDeploy": "(b) standard response, faster deployment",
    "highPR": "(c) optimistic response, standard deployment",
    "high_PR_fastDeploy": "(d) optimistic response, faster deployment",
}
PM = 1e6  # per-million-jobs scaling


def g(tag: str) -> pd.DataFrame:
    return pd.read_csv(OUT / tag / "grid_results.csv")


def rng_(s: pd.Series, scale: float = 1.0, nd: int = 4) -> list:
    return [round(float(s.min()) * scale, nd), round(float(s.max()) * scale, nd)]


def case_summary(tag: str) -> dict:
    df = g(tag)
    med_pos_all = bool((df.median_deltaH > 0).all())
    med_neg_all = bool((df.median_deltaH < 0).all())
    mean_by_T = {float(T): rng_(sub.mean_deltaH, PM, 4)
                 for T, sub in df.groupby("T_work_h")}
    return {
        "p_benefit_range": rng_(df.p_benefit),
        "p_upper95_max": round(float(df.p_benefit_hi95.max()), 4),
        "p_lower95_min": round(float(df.p_benefit_lo95.min()), 4),
        "p_lower95_max": round(float(df.p_benefit_lo95.max()), 4),
        "mean_dH_pm_range": rng_(df.mean_deltaH, PM, 4),
        "median_dH_pm_range": rng_(df.median_deltaH, PM, 5),
        "median_positive_everywhere": med_pos_all,
        "median_negative_everywhere": med_neg_all,
        "mean_dH_pm_by_T": mean_by_T,
        "median_rel_dH_range_pct": rng_(df.median_rel_deltaH * 100, 1.0, 2),
        "mean_H0_pm_range": rng_(df.mean_H0, PM, 4),
    }


def rep_table(tags=("baseline", "baseline_fastDeploy", "highPR", "high_PR_fastDeploy"),
              Q=400.0, Ts=(0.05, 0.25, 2.0)) -> list:
    rows = []
    for tag in tags:
        df = g(tag)
        for T in Ts:
            r = df[(df.Q_veh_h == Q) & (df.T_work_h == T)].iloc[0]
            rows.append({
                "case": CASES[tag], "tag": tag, "Q": Q, "T_h": T,
                "mean_dH_pm": round(r.mean_deltaH * PM, 4),
                "median_dH_pm": round(r.median_deltaH * PM, 5),
                "p_benefit": round(r.p_benefit, 3),
                "wilson_lo": round(r.p_benefit_lo95, 3),
                "wilson_hi": round(r.p_benefit_hi95, 3),
                "median_rel_dH_pct": round(r.median_rel_deltaH * 100, 2),
            })
    return rows


def break_even(tag: str, thresholds=(0.5, 0.8)) -> dict:
    df = g(tag)
    out = {}
    for p_star in thresholds:
        by_Q = {}
        for Q, sub in df.groupby("Q_veh_h"):
            sub = sub.sort_values("T_work_h")
            hit = sub[sub.p_benefit >= p_star]
            by_Q[float(Q)] = (float(hit.T_work_h.iloc[0]) if len(hit) else None)
        out[str(p_star)] = by_Q
    return out


def decision_classification(tag: str) -> dict:
    df = g(tag)
    supported = int((df.p_benefit_lo95 >= 0.8).sum())
    not_supported = int((df.p_benefit_hi95 < 0.5).sum())
    return {"supported": supported, "not_supported": not_supported,
            "uncertain": int(len(df) - supported - not_supported)}


def ke_requirements() -> dict:
    """Smallest reduction (1-k_E) needed per duration to hit thresholds at ALL flows."""
    kes = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    frames = {ke: g(f"baseline_KE{str(ke).replace('.', 'p')}") for ke in kes}
    durations = sorted(frames[0.95].T_work_h.unique())
    out = {}
    for p_star in (0.5, 0.8):
        by_T = {}
        for T in durations:
            req = None
            for ke in sorted(kes, reverse=True):  # smallest reduction first
                sub = frames[ke]
                ok = bool((sub[sub.T_work_h == T].p_benefit >= p_star).all())
                if ok:
                    req = round(1 - ke, 2)
                    break
            by_T[float(T)] = req
        out[str(p_star)] = by_T
    return out


def robustness_compare(base_tag: str, other_tags: list[str]) -> dict:
    base = g(base_tag)
    out = {}
    for t in other_tags:
        o = g(t)
        m = base.merge(o, on=["Q_veh_h", "T_work_h"], suffixes=("_b", "_o"))
        out[t] = {
            "p_range": rng_(o.p_benefit),
            "max_abs_dP": round(float((m.p_benefit_o - m.p_benefit_b).abs().max()), 4),
            "mean_dH_pm_range": rng_(o.mean_deltaH, PM, 4),
            "mean_dH_ratio_median": round(float((m.mean_deltaH_o / m.mean_deltaH_b).median()), 3),
        }
    return out


def main() -> None:
    digest: dict = {"cases": {t: case_summary(t) for t in CASES}}
    digest["representative_Q400"] = rep_table()
    digest["break_even"] = {t: break_even(t) for t in CASES}
    digest["decision_classification"] = {t: decision_classification(t) for t in CASES}
    digest["ke_requirements"] = ke_requirements()

    digest["correlation_robustness"] = {
        "baseline": robustness_compare("baseline",
                                       ["baseline_corr_plausible", "baseline_corr_stress",
                                        "baseline_corr_behavioural"]),
        "highPR": robustness_compare("highPR", ["highPR_corr_plausible", "highPR_corr_stress"]),
    }
    ach = {}
    for t in ("baseline_corr_plausible", "baseline_corr_stress", "baseline_corr_behavioural"):
        p = OUT / t / "achieved_correlations.json"
        if p.exists():
            ach[t] = json.loads(p.read_text())
    digest["achieved_correlations"] = ach

    digest["severity_robustness"] = {
        "baseline": robustness_compare("baseline",
                                       ["baseline_sev_fatality_only", "baseline_sev_occupant_frontal",
                                        "baseline_sev_legacy_occupant"]),
        "highPR": robustness_compare("highPR",
                                     ["highPR_sev_fatality_only", "highPR_sev_occupant_frontal",
                                      "highPR_sev_legacy_occupant"]),
    }
    digest["tail_allocation"] = {
        "baseline": robustness_compare("baseline",
                                       ["baseline_alloc_low_tail", "baseline_alloc_high_tail"]),
        "highPR": robustness_compare("highPR",
                                     ["highPR_alloc_low_tail", "highPR_alloc_high_tail"]),
    }

    digest["prcc"] = pd.read_csv(OUT / "sensitivity" / "prcc_summary.csv").to_dict("records")
    digest["prcc_bootstrap"] = pd.read_csv(OUT / "sensitivity" / "prcc_bootstrap.csv").to_dict("records")
    digest["prcc_decision_scale"] = pd.read_csv(OUT / "sensitivity" / "prcc_decision_scale_summary.csv").to_dict("records")
    digest["convergence"] = pd.read_csv(OUT / "convergence" / "convergence_summary.csv").to_dict("records")
    digest["validation"] = pd.read_csv(OUT / "validation" / "deterministic_check.csv").to_dict("records")

    (OUT / "results_digest.json").write_text(json.dumps(digest, indent=1))

    # Human-readable companion
    lines = ["# Results digest (machine-generated — source of every number in the manuscript)\n"]
    for tag, label in CASES.items():
        c = digest["cases"][tag]
        lines.append(f"## {label}")
        lines.append(f"- P(dH<0) range: {c['p_benefit_range'][0]:.3f}-{c['p_benefit_range'][1]:.3f} "
                     f"(upper95 max {c['p_upper95_max']:.3f}, lower95 max {c['p_lower95_max']:.3f})")
        lines.append(f"- mean dH per million jobs: {c['mean_dH_pm_range'][0]:+.3f} to {c['mean_dH_pm_range'][1]:+.3f}; "
                     f"median: {c['median_dH_pm_range'][0]:+.4f} to {c['median_dH_pm_range'][1]:+.4f} "
                     f"(median>0 everywhere: {c['median_positive_everywhere']}, "
                     f"median<0 everywhere: {c['median_negative_everywhere']})")
        lines.append(f"- median relative dH: {c['median_rel_dH_range_pct'][0]:+.1f}% to "
                     f"{c['median_rel_dH_range_pct'][1]:+.1f}% of baseline per-job harm\n")
    lines.append("## k_E requirements (reduction needed at ALL flows)")
    for thr, by_T in digest["ke_requirements"].items():
        lines.append(f"- threshold {thr}: " + ", ".join(
            f"T={T}h: {'-' if v is None else f'{v:.0%}'}" for T, v in by_T.items()))
    lines.append("\n## Sensitivity (mean |PRCC|, value scale then decision scale)")
    for row in digest["prcc"][:6]:
        lines.append(f"- {row['parameter']}: {row['mean_abs_PRCC']:.3f}")
    lines.append("--- decision scale (G = dH/r_E) ---")
    for row in digest["prcc_decision_scale"][:6]:
        lines.append(f"- {row['parameter']}: {row['mean_abs_PRCC']:.3f}")
    (OUT / "results_digest.md").write_text("\n".join(lines))
    print(f"digest written: {OUT/'results_digest.json'}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
