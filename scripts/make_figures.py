#!/usr/bin/env python3
"""Generate every manuscript figure from the outputs/ artifacts.

Publication conventions (AA&P submission, rebuilt 2026-07):
  * Every figure is generated at a physical width of 16.5 cm (6.496 in) so it
    embeds in the Word manuscript at the full text-body width (~16 cm) at 1:1
    with no shrinkage. At 1:1 the OBSERVED point size equals the DESIGNED point
    size, so setting all text >= 10 pt guarantees the reviewer's >= 9 pt floor.
  * Figures are saved WITHOUT bbox_inches="tight" and laid out with
    constrained_layout, so the saved canvas keeps its exact 16.5 cm width (a
    tight crop would change the physical width and break the 1:1 guarantee).
  * Colourblind-safe Okabe-Ito qualitative palette; RdBu / RdBu_r diverging maps
    with a consistent "blue = sign beneficial, red = sign harmful" semantic.
  * Vector PDF + >= 300 dpi PNG for every figure; TrueType embedded (fonttype 42).

A per-figure font/size audit is written to outputs/figures/_font_audit.json and
printed at the end (proves min observed size >= 9 pt at 16.5 cm).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon, Circle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

from aw_model.risk_functions import p_worker, p_occupant  # noqa: E402
from aw_model.model import feasible_delta_v_kmh  # noqa: E402

# --------------------------------------------------------------------------- #
# Physical geometry and style
# --------------------------------------------------------------------------- #
CM = 1.0 / 2.54
W_CM = 16.5                 # embed width: full text-body width of an A4 page
W_IN = W_CM * CM            # 6.496 in -- the width of EVERY figure
MIN_PT = 10.0              # design floor; observed >= 9 pt after 1:1 embed

# Okabe-Ito colourblind-safe qualitative palette
OI_BLACK = "#000000"; OI_ORANGE = "#E69F00"; OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"; OI_YELLOW = "#F0E442"; OI_BLUE = "#0072B2"
OI_VERM = "#D55E00"; OI_PURPLE = "#CC79A7"; GREY = "#6E6E6E"; LGREY = "#BDBDBD"

# Consistent semantics across figures
C_NOSIGN = OI_BLUE         # S0 / baseline / "no sign"
C_SIGN = OI_VERM           # S1 / "with sign"
C_MAIS = OI_BLUE           # severe injury MAIS 3+
C_FATAL = OI_VERM          # fatality
C_FRONTAL = OI_SKY         # frontal variant
C_NOTSUP = OI_VERM         # decision: not supported
C_UNCERT = GREY            # decision: uncertain
C_SUP = OI_GREEN           # decision: supported

plt.rcParams.update({
    "font.size": MIN_PT,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": MIN_PT,
    "ytick.labelsize": MIN_PT,
    "legend.fontsize": MIN_PT,
    "legend.title_fontsize": MIN_PT,
    "figure.titlesize": 12,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.bbox": None,           # NEVER tight -- keep exact 16.5 cm width
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,     # ASCII hyphen everywhere (matches manuscript)
})

CASES = [
    ("baseline", "(a) Standard response,\nstandard deployment"),
    ("baseline_fastDeploy", "(b) Standard response,\nfaster deployment"),
    ("highPR", "(c) Optimistic response,\nstandard deployment"),
    ("high_PR_fastDeploy", "(d) Optimistic response,\nfaster deployment"),
]
T_LABELS = {0.05: "3 min", 0.1: "6 min", 0.25: "15 min", 0.5: "30 min",
            1.0: "1 h", 2.0: "2 h", 4.0: "4 h"}

_AUDIT: list[dict] = []


def g(tag: str) -> pd.DataFrame:
    return pd.read_csv(OUT / tag / "grid_results.csv")


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="T_work_h", columns="Q_veh_h", values=col).sort_index()


def _frame(ax):
    """Restore all four spines (for heatmaps / schematics that want a frame)."""
    for s in ax.spines.values():
        s.set_visible(True)


def _heat_axes(ax, piv):
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{int(q)}" for q in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([T_LABELS[t] for t in piv.index])
    ax.tick_params(length=0)


def _save(fig, name: str) -> None:
    # Audit every visible, non-empty text artist BEFORE closing.
    sizes = []
    for t in fig.findobj(mtext.Text):
        try:
            if not t.get_visible():
                continue
            s = t.get_text()
            if s is None or str(s).strip() == "":
                continue
            sizes.append(float(t.get_fontsize()))
        except Exception:
            continue
    w_in, h_in = fig.get_size_inches()
    min_pt = min(sizes) if sizes else float("nan")
    # observed pt after embedding the saved canvas at exactly 16.5 cm
    obs_scale = W_CM / (w_in / CM)
    _AUDIT.append({
        "figure": name, "width_cm": round(w_in / CM, 3), "height_cm": round(h_in / CM, 3),
        "n_text": len(sizes), "min_design_pt": round(min_pt, 2),
        "min_observed_pt": round(min_pt * obs_scale, 2),
        "png_px_w": int(round(w_in * 300)), "png_px_h": int(round(h_in * 300)),
    })
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name:26s} {w_in/CM:5.1f} x {h_in/CM:4.1f} cm  min_text={min_pt:.1f} pt")


# --------------------------------------------------------------------------- #
# Schematic primitives (drawn in a centimetre coordinate system)
# --------------------------------------------------------------------------- #
def _schematic(w_cm: float, h_cm: float):
    """Full-bleed axis in centimetre coordinates (1 data unit = 1 cm)."""
    fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_cm); ax.set_ylim(0, h_cm)
    ax.axis("off")
    ax.set_aspect("equal")
    return fig, ax


def _box(ax, x, y, w, h, lines, fc, ec, *, title=None, title_pt=10.5, body_pt=10.0,
         title_color=None, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18",
                                fc=fc, ec=ec, lw=1.3, mutation_aspect=1.0))
    cx = x + w / 2
    ty = y + h - 0.42
    if title:
        ax.text(cx, ty, title, ha="center", va="top", fontsize=title_pt,
                fontweight="bold", color=title_color or ec)
        ty -= 0.62
    if lines:
        if align == "center":
            ax.text(cx, ty, "\n".join(lines), ha="center", va="top",
                    fontsize=body_pt, color="#1a1a1a", linespacing=1.35)
        else:
            ax.text(x + 0.34, ty, "\n".join(lines), ha="left", va="top",
                    fontsize=body_pt, color="#1a1a1a", linespacing=1.35)


def _arrow(ax, p1, p2, *, color=GREY, lw=1.5, style="-|>", ms=13, ls="-",
           connection=None):
    kw = dict(arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
              linestyle=ls, shrinkA=1, shrinkB=1, joinstyle="miter")
    if connection:
        kw["connectionstyle"] = connection
    ax.add_patch(FancyArrowPatch(p1, p2, **kw))


def _dim(ax, p1, p2, label, *, off=0.0, pt=10.0, color="#333333", lab_dy=0.28,
         lab_dx=0.0, va="bottom", ha="center"):
    """A double-headed dimension line with a centred label."""
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="<|-|>", mutation_scale=9,
                                 lw=1.1, color=color, shrinkA=0, shrinkB=0))
    mx, my = (p1[0] + p2[0]) / 2 + lab_dx, (p1[1] + p2[1]) / 2 + lab_dy
    ax.text(mx, my, label, ha=ha, va=va, fontsize=pt, color=color)


# =========================================================================== #
# fig1 -- model structure schematic (rebuilt)
# =========================================================================== #
def fig1_model_schematic():
    H = 9.7
    fig, ax = _schematic(W_CM, H)

    cream, cream_e = "#f4efe3", "#9c7f3d"
    blue_f, blue_e = "#e7f0f7", OI_BLUE
    red_f, red_e = "#fbebe3", OI_VERM
    grey_f, grey_e = "#eef0f2", "#5a6b78"
    green_f, green_e = "#e6f4ec", OI_GREEN

    # Column A -- inputs
    _box(ax, 0.35, 5.05, 3.35, 3.2, [
        "Uncertain inputs", "(Table 1)", "Monte Carlo,", "20 000 draws"],
        cream, cream_e, title=None, body_pt=10.0)
    _box(ax, 0.35, 1.15, 3.35, 2.9, [
        "Scenario grid", "Q: 50-1000 veh/h", "T: 3 min to 4 h"],
        cream, cream_e, body_pt=10.0)

    # Column B -- strategies
    _box(ax, 4.45, 5.05, 6.5, 3.2, [
        "Work-period pathways:", "•  worker strike", "•  work-vehicle strike"],
        blue_f, blue_e, title="S₀ : no sign", title_color=blue_e,
        align="left", body_pt=10.0)
    _box(ax, 4.45, 0.75, 6.5, 3.85, [
        "Work-period pathways", "(speed reduced for responders):",
        "•  worker strike", "•  work-vehicle strike"],
        red_f, red_e, title="S₁ : sign deployed", title_color=red_e,
        align="left", body_pt=10.0)
    # emphasise the added deployment pathway inside S1
    ax.text(4.79, 1.28, "•  deployment exposure (added)",
            ha="left", va="center", fontsize=10.0, color=OI_VERM, fontweight="bold")

    # Column C -- harm
    _box(ax, 11.55, 5.55, 2.2, 1.95, ["expected", "harm"], grey_f, grey_e,
         title="H(S₀)", title_color=grey_e, body_pt=10.0)
    _box(ax, 11.55, 1.6, 2.2, 1.95, ["expected", "harm"], grey_f, grey_e,
         title="H(S₁)", title_color=grey_e, body_pt=10.0)

    # Column D -- net change / decision
    _box(ax, 13.95, 2.95, 2.5, 3.6, [
        "H(S₁)", "- H(S₀)", "", "decision:", "P(ΔH < 0)"],
        green_f, green_e, title="ΔH =", title_color=green_e, body_pt=10.0)

    # arrows: inputs feed both strategies (crossed pair softened)
    _arrow(ax, (3.75, 6.65), (4.4, 6.7))
    _arrow(ax, (3.75, 2.55), (4.4, 2.6))
    _arrow(ax, (3.75, 5.9), (4.4, 3.3), color=LGREY, connection="arc3,rad=0.12")
    _arrow(ax, (3.75, 3.2), (4.4, 6.0), color=LGREY, connection="arc3,rad=-0.12")
    # strategies -> harm
    _arrow(ax, (11.0, 6.6), (11.5, 6.5))
    _arrow(ax, (11.0, 2.5), (11.5, 2.55))
    # harm -> net
    _arrow(ax, (13.8, 6.15), (14.6, 5.6))
    _arrow(ax, (13.8, 2.6), (14.6, 3.4))

    ax.text(W_CM / 2, 9.35,
            "Deploying the sign adds an upstream deployment exposure (S₁) "
            "absent under S₀",
            ha="center", va="center", fontsize=10.0, style="italic", color="#444444")
    _save(fig, "fig1_model_schematic")


# =========================================================================== #
# fig0a -- worksite layout (NEW)
# =========================================================================== #
def fig0a_worksite_layout():
    H = 9.3
    fig, ax = _schematic(W_CM, H)

    x0, x1 = 0.55, 15.95
    road_lo, road_hi = 1.0, 2.6            # travelled way band (cm)
    sign_x = 2.5
    worker_x = 11.95                       # worker on foot
    veh_x = 13.85                          # work-vehicle centre
    walk_y = 3.5                           # deployment walk path (c_deploy, exagg.)
    worker_y = 5.0                         # worker + vehicle (c_work, exagg.)
    daw_y = 7.0                            # d_AW dimension line

    # road surface + markings
    ax.add_patch(Rectangle((x0, road_lo), x1 - x0, road_hi - road_lo,
                           fc="#e9ecee", ec="none", zorder=0))
    ax.plot([x0, x1], [road_hi, road_hi], color="#2b2b2b", lw=1.8, zorder=2)   # edgeline
    ax.plot([x0, x1], [road_lo, road_lo], color="#7d868c", lw=1.0, zorder=2)
    ax.plot([x0, x1], [(road_lo + road_hi) / 2] * 2, color="#c8a02a", lw=1.4,
            ls=(0, (7, 6)), zorder=2)                                          # centreline
    ax.text(x0 + 0.15, road_lo + 0.32, "travelled way / live lane", ha="left",
            va="center", fontsize=10.0, color="#4a4a4a")
    ax.text(8.6, road_hi + 0.13, "edgeline", ha="center", va="bottom",
            fontsize=10.0, color="#2b2b2b")
    # direction of travel
    _arrow(ax, (10.6, road_lo + 0.32), (14.2, road_lo + 0.32), color="#5a6b78",
           lw=1.6, ms=15)
    ax.text(12.4, road_lo + 0.32, "direction of travel", ha="center", va="bottom",
            fontsize=10.0, color="#5a6b78")

    # advanced warning sign (upstream)
    post_top = road_hi + 0.8
    ax.plot([sign_x, sign_x], [road_hi, post_top], color="#333333", lw=2.0, zorder=4)
    ax.add_patch(Polygon([[sign_x, post_top], [sign_x - 0.36, post_top + 0.64],
                          [sign_x + 0.36, post_top + 0.64]], closed=True,
                         fc=OI_ORANGE, ec="#5a3d00", lw=1.3, zorder=5))
    ax.text(sign_x, post_top + 0.96, "portable\nadvanced warning sign", ha="center",
            va="bottom", fontsize=10.0, color="#5a3d00", linespacing=1.2,
            fontweight="bold")

    # work vehicle + worker (downstream)
    ax.add_patch(FancyBboxPatch((veh_x - 1.2, worker_y - 0.36), 2.4, 0.72,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=OI_SKY, ec="#123", lw=1.3, zorder=4))
    ax.text(veh_x, worker_y, "work vehicle", ha="center", va="center",
            fontsize=10.0, color="#0b2233", zorder=5)
    ax.add_patch(Circle((worker_x, worker_y), 0.18, fc=OI_VERM, ec="#5a2400",
                        lw=1.0, zorder=6))
    ax.text(worker_x, worker_y + 0.42, "worker on foot", ha="center", va="bottom",
            fontsize=10.0, color="#5a2400")

    # deployment walk path (there and back) at offset c_deploy
    ax.add_patch(FancyArrowPatch((worker_x, walk_y), (sign_x + 0.2, walk_y),
                                 arrowstyle="<|-|>", mutation_scale=12, lw=1.8,
                                 color=OI_VERM, ls=(0, (5, 3)), zorder=3))
    ax.text(7.4, walk_y + 0.12, "on-foot deployment walk\n(place + remove)",
            ha="center", va="bottom", fontsize=10.0, color=OI_VERM, linespacing=1.2)

    # dimension: d_AW (with dotted leaders from sign and worker)
    ax.plot([sign_x, sign_x], [post_top + 0.6, daw_y], color="#aaaaaa", lw=0.7,
            ls=":", zorder=1)
    ax.plot([worker_x, worker_x], [5.78, daw_y], color="#aaaaaa", lw=0.7, ls=":",
            zorder=1)
    _dim(ax, (sign_x, daw_y), (worker_x, daw_y),
         "d$_{AW}$ ≈ 50 m   (advanced warning distance)", pt=10.0, lab_dy=0.26)

    # dimension: c_deploy (lateral, left of centre)
    _dim(ax, (4.7, road_hi), (4.7, walk_y), "", pt=10.0)
    ax.text(4.45, (road_hi + walk_y) / 2, "c$_{deploy}$\n(0.5-2 m)", ha="right",
            va="center", fontsize=10.0, color="#333333", linespacing=1.2)
    # dimension: c_work (lateral, left of the worker; label placed above the walk band)
    _dim(ax, (11.3, road_hi), (11.3, worker_y), "", pt=10.0)
    ax.text(11.05, 4.55, "c$_{work}$ (1-3 m)", ha="right", va="center",
            fontsize=10.0, color="#333333")

    ax.text(x0, 0.34,
            "Plan view (schematic). Lateral offsets are exaggerated relative to the "
            "along-road distance for clarity.",
            ha="left", va="center", fontsize=10.0, style="italic", color="#555555")
    _save(fig, "fig0a_worksite_layout")


# =========================================================================== #
# fig0b -- parameter map (NEW)
# =========================================================================== #
def fig0b_parameter_map():
    H = 11.8
    fig, ax = _schematic(W_CM, H)

    cream, cream_e = "#f4efe3", "#9c7f3d"
    blue_f, blue_e = "#e7f0f7", OI_BLUE
    red_f, red_e = "#fbebe3", OI_VERM
    green_f, green_e = "#e6f4ec", OI_GREEN

    ax.text(W_CM / 2, H - 0.3, "Inputs  →  three collision pathways  →  decision",
            ha="center", va="top", fontsize=11.0, fontweight="bold", color="#333333")

    # ---- Column 1: input groups ----
    ix, iw = 0.35, 4.0
    groups = [
        ("Encroachment", ["rate r$_E$, reach α", "offsets c, lengths L"], 8.55),
        ("Speed environment", ["μ$_V$, σ$_V$  →  V₀"], 6.95),
        ("Sign response", ["p$_R$, ΔV, a, d$_{AW}$  →  V₁"], 5.45),
        ("Deployment", ["v$_{walk}$, t$_{handle}$  →  T$_{deploy}$"], 3.95),
        ("Severity curves", ["p$_w$(V),  p$_o$(ΔV)"], 2.45),
    ]
    icy = []
    for title, lines, y in groups:
        h = 1.2 if len(lines) == 1 else 1.55
        _box(ax, ix, y, iw, h, lines, cream, cream_e, title=title, title_pt=10.5,
             body_pt=10.0, align="left")
        icy.append(y + h / 2)

    # ---- Column 2: three pathways ----
    px, pw = 6.05, 5.2
    paths = [
        ("Pathway 1", ["Worker strike (work)", "λ$_w$ · p$_w$(V)"], 7.75, blue_f, blue_e),
        ("Pathway 2", ["Work-vehicle strike (work)", "λ$_v$ · p$_o$(ΔV$_{occ}$)"], 5.35, blue_f, blue_e),
        ("Pathway 3  (S₁ only)", ["Worker strike (deployment)", "λ$_d$ · p$_w$(V₀)"], 2.95, red_f, red_e),
    ]
    pcy = []
    for title, lines, y, fc, ec in paths:
        _box(ax, px, y, pw, 1.8, lines, fc, ec, title=title, title_pt=10.5,
             body_pt=10.0, align="center")
        pcy.append(y + 1.8 / 2)

    # ---- Column 3: outputs (Decision placed between the two harm terms so both
    #      H(S0) and H(S1) feed it with clean, non-crossing arrows) ----
    ox, ow = 12.15, 4.0
    cx = ox + ow / 2
    _box(ax, ox, 7.05, ow, 1.7, ["P1(V₀) + P2(V₀)"], blue_f, blue_e,
         title="H(S₀)  no sign", title_pt=10.5, body_pt=10.0)
    _box(ax, ox, 3.95, ow, 2.1, ["ΔH = H(S₁) - H(S₀)", "", "deploy if P(ΔH<0) ≥ p*"],
         green_f, green_e, title="Decision", title_pt=10.5, body_pt=10.0)
    _box(ax, ox, 1.0, ow, 1.95, ["P1(V₁) + P2(V₁)", "+ P3 deploy"], red_f, red_e,
         title="H(S₁)  sign", title_pt=10.5, body_pt=10.0)

    # ---- arrows: input groups -> pathways (kept sparse to avoid clutter) ----
    a_in = dict(color=LGREY, lw=1.2, ms=11)
    _arrow(ax, (ix + iw, icy[0]), (px, pcy[0] + 0.35), **a_in)   # encroachment -> P1
    _arrow(ax, (ix + iw, icy[0] - 0.2), (px, pcy[1] + 0.2), **a_in)  # encroachment -> P2
    _arrow(ax, (ix + iw, icy[1]), (px, pcy[0]), **a_in)         # speed -> P1
    _arrow(ax, (ix + iw, icy[2]), (px, pcy[1] - 0.1), **a_in)   # response -> P2
    _arrow(ax, (ix + iw, icy[3]), (px, pcy[2]), **a_in)         # deployment -> P3
    _arrow(ax, (ix + iw, icy[4]), (px, pcy[2] - 0.35), **a_in)  # severity -> P3

    # ---- arrows: pathways -> harm terms ----
    a_p = dict(color=GREY, lw=1.5, ms=13)
    _arrow(ax, (px + pw, pcy[0]), (ox, 8.1), **a_p)            # P1 -> H0
    _arrow(ax, (px + pw, pcy[1]), (ox, 7.6), **a_p)           # P2 -> H0
    _arrow(ax, (px + pw, pcy[0] - 0.2), (ox, 2.55), **a_p)    # P1 -> H1
    _arrow(ax, (px + pw, pcy[1] - 0.2), (ox, 2.1), **a_p)     # P2 -> H1
    _arrow(ax, (px + pw, pcy[2]), (ox, 1.65), color=OI_VERM, lw=1.6, ms=13)  # P3 -> H1
    # both harm terms -> decision (down from H0, up from H1; no box piercing)
    _arrow(ax, (cx, 7.05), (cx, 6.05), **a_p)                 # H0 -> decision
    _arrow(ax, (cx, 2.95), (cx, 3.95), **a_p)                 # H1 -> decision

    _save(fig, "fig0b_parameter_map")


# =========================================================================== #
# fig2 -- warning effect
# =========================================================================== #
def fig2_warning_effect():
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 8.0 * CM), constrained_layout=True)

    rng = np.random.default_rng(4)
    n = 300_000
    mu, sd = 47.0, 9.0
    V0 = np.clip(rng.normal(mu, sd, n), 0, 80)
    p_R, dV = 0.3, 10.0
    resp = rng.random(n) < p_R
    V1 = np.maximum(V0 - resp * dV, 0)
    bins = np.linspace(0, 80, 90)
    axes[0].hist(V0, bins=bins, density=True, histtype="step", lw=1.8,
                 color=C_NOSIGN, label="V$_0$  (no sign)")
    axes[0].hist(V1, bins=bins, density=True, histtype="step", lw=1.8,
                 color=C_SIGN, label=f"V$_1$  (sign; p$_R$={p_R:.1f}, ΔV={dV:.0f} km/h)")
    axes[0].set_xlabel("Operating speed at work location (km/h)")
    axes[0].set_ylabel("Probability density")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title("(a) Bernoulli mixture of responders", loc="left")

    v0 = np.linspace(20, 80, 200)
    styles = {30: (":", "shortest 30 m"), 50: ("-", "typical 50 m"), 70: ("--", "longest 70 m")}
    for d_aw, (ls, dlab) in styles.items():
        for a, c in ((0.8, OI_SKY), (2.5, OI_BLUE)):
            dmax = feasible_delta_v_kmh(v0, np.full_like(v0, a), np.full_like(v0, d_aw))
            axes[1].plot(v0, np.minimum(dmax, v0), ls=ls, color=c, lw=1.6)
    axes[1].axhline(15, color=C_SIGN, lw=1.4, ls="-.")
    axes[1].set_xlabel("Approach speed V$_0$ (km/h)")
    axes[1].set_ylabel("Maximum feasible ΔV (km/h)")
    axes[1].set_title("(b) Kinematic feasibility bound", loc="left")
    # two-part legend kept compact and inside
    h_dist = [Line2D([0], [0], color="k", ls=ls, lw=1.6) for ls, _ in styles.values()]
    h_acc = [Line2D([0], [0], color=OI_SKY, lw=1.6), Line2D([0], [0], color=OI_BLUE, lw=1.6),
             Line2D([0], [0], color=C_SIGN, lw=1.4, ls="-.")]
    leg1 = axes[1].legend(h_dist, [d for _, d in styles.values()], frameon=False,
                          loc="upper left", title="d$_{AW}$", handlelength=2.2)
    axes[1].add_artist(leg1)
    axes[1].legend(h_acc, ["a = 0.8 m/s$^2$", "a = 2.5 m/s$^2$", "sampled ΔV cap (15 km/h)"],
                   frameon=False, loc="lower right", handlelength=2.2)
    _save(fig, "fig2_warning_effect")


# =========================================================================== #
# fig3 -- severity curves
# =========================================================================== #
def fig3_severity_curves():
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 7.8 * CM), constrained_layout=True)
    v = np.linspace(0, 100, 300)

    axes[0].plot(v, p_worker(v, "rosen_mais3plus"), color=C_MAIS, lw=2.0,
                 label="Severe injury (MAIS 3+)")
    axes[0].plot(v, p_worker(v, "rosen_fatality"), color=C_FATAL, lw=2.0, ls="--",
                 label="Fatality")
    axes[0].axvspan(35, 60, color="#ececec", zorder=0)
    axes[0].text(47.5, 0.62, "modelled\nspeed range", ha="center", va="center",
                 fontsize=10.0, color="#555555")
    axes[0].set_xlabel("Impact speed (km/h)")
    axes[0].set_ylabel("Probability of outcome")
    axes[0].set_title("(a) Worker (person on foot)", loc="left")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].plot(v, p_occupant(v, "wang_mais3plus_all"), color=C_MAIS, lw=2.0,
                 label="MAIS 3+ (all crashes)")
    axes[1].plot(v, p_occupant(v, "wang_mais3plus_frontal"), color=C_FRONTAL, lw=1.8,
                 ls=":", label="MAIS 3+ (frontal)")
    axes[1].plot(v, p_occupant(v, "wang_fatality_all"), color=C_FATAL, lw=2.0, ls="--",
                 label="Fatality (all crashes)")
    axes[1].set_xlabel("Occupant ΔV (km/h)")
    axes[1].set_ylabel("Probability of outcome")
    axes[1].set_title("(b) Occupant of striking vehicle", loc="left")
    axes[1].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.set_ylim(0, 1)
    _save(fig, "fig3_severity_curves")


# =========================================================================== #
# fig4 -- P(dH<0) heatmaps (0.5-centred diverging, zoomed + annotated)
# =========================================================================== #
def fig4_probability_heatmaps():
    fig, axes = plt.subplots(2, 2, figsize=(W_IN, 15.6 * CM), sharex=True, sharey=True,
                             constrained_layout=True)
    norm = mcolors.TwoSlopeNorm(vmin=0.1, vcenter=0.5, vmax=0.9)
    cmap = plt.get_cmap("RdBu")
    im = None
    for ax, (tag, label) in zip(axes.flat, CASES):
        piv = _pivot(g(tag), "p_benefit")
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        _heat_axes(ax, piv)
        _frame(ax)
        ax.set_title(label, loc="left", fontsize=10.5)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                color = "white" if val < 0.30 else "#111111"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10.0,
                        color=color)
    for ax in axes[1]:
        ax.set_xlabel("Traffic flow Q (veh/h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Work duration")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02,
                      ticks=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9], extend="both")
    cb.set_label("P(ΔH < 0)   (blue = sign beneficial)")
    # mark the two decision thresholds on the colourbar itself
    for yv in (0.5, 0.8):
        cb.ax.axhline(yv, color="#111111", lw=1.2, ls=(0, (3, 2)))
    fig.suptitle("Colour scale centred at 0.5; no scenario reaches the p* = 0.8 "
                 "threshold (highest = 0.60)", fontsize=10.5, color="#333333")
    _save(fig, "fig4_probability_heatmaps")


# =========================================================================== #
# fig5 -- mean dH heatmaps
# =========================================================================== #
def _fmt_dh(v: float) -> str:
    # compact (<= 4 glyphs) so 7 columns of annotations never collide at >=10 pt
    a = abs(v)
    if a < 0.05:
        return "0"
    return f"{v:+.1f}"


def fig5_mean_dh_heatmaps():
    fig, axes = plt.subplots(2, 2, figsize=(W_IN, 15.6 * CM), sharex=True, sharey=True,
                             constrained_layout=True)
    vals = pd.concat([g(t)["mean_deltaH"] for t, _ in CASES]) * 1e6
    lim = float(np.abs(vals).max())
    norm = mcolors.SymLogNorm(linthresh=0.01, vmin=-lim, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")
    im = None
    for ax, (tag, label) in zip(axes.flat, CASES):
        piv = _pivot(g(tag), "mean_deltaH") * 1e6
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        _heat_axes(ax, piv)
        _frame(ax)
        ax.set_title(label, loc="left", fontsize=10.5)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                nv = norm(val)
                color = "white" if (nv < 0.22 or nv > 0.78) else "#111111"
                ax.text(j, i, _fmt_dh(val), ha="center", va="center", fontsize=10.0,
                        color=color)
    for ax in axes[1]:
        ax.set_xlabel("Traffic flow Q (veh/h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Work duration")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02, aspect=32)
    cb.set_label("Mean ΔH  (MAIS 3+ per 10⁶ jobs)")
    fig.suptitle("Blue = sign reduces mean harm (ΔH < 0);  red = sign adds harm",
                 fontsize=10.5, color="#333333")
    _save(fig, "fig5_mean_dh_heatmaps")


# =========================================================================== #
# fig6 -- P(dH<0) vs duration
# =========================================================================== #
def fig6_p_vs_duration():
    fig, axes = plt.subplots(2, 2, figsize=(W_IN, 14.5 * CM), sharex=True, sharey=True,
                             constrained_layout=True)
    colors = {50: OI_SKY, 400: OI_BLUE, 1000: "#083b5c"}
    for ax, (tag, label) in zip(axes.flat, CASES):
        df = g(tag)
        for Q, c in colors.items():
            sub = df[df.Q_veh_h == Q].sort_values("T_work_h")
            ax.plot(sub.T_work_h, sub.p_benefit, marker="o", ms=4, lw=1.6,
                    color=c, label=f"Q = {Q} veh/h")
            ax.fill_between(sub.T_work_h, sub.p_benefit_lo95, sub.p_benefit_hi95,
                            color=c, alpha=0.16, lw=0)
        ax.axhline(0.5, color="#777777", lw=1.0, ls="--")
        ax.axhline(0.8, color="#777777", lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(list(T_LABELS))
        ax.set_xticklabels(list(T_LABELS.values()), rotation=40, ha="right")
        ax.set_title(label, loc="left", fontsize=10.5)
        ax.set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.03, 0.74))
    axes[0, 1].text(0.05, 0.5, "0.5", transform=axes[0, 1].get_yaxis_transform(),
                    fontsize=10.0, color="#666666", va="bottom", ha="left")
    axes[0, 1].text(0.05, 0.8, "0.8", transform=axes[0, 1].get_yaxis_transform(),
                    fontsize=10.0, color="#666666", va="bottom", ha="left")
    for ax in axes[1]:
        ax.set_xlabel("Work duration")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(ΔH < 0)")
    _save(fig, "fig6_p_vs_duration")


# =========================================================================== #
# fig7 -- decision summary (REDESIGNED from block-colour heatmaps)
# =========================================================================== #
def fig7_decision_classification():
    fig, ax = plt.subplots(figsize=(W_IN, 9.6 * CM), constrained_layout=True)

    rows = []
    for tag, label in CASES:
        df = g(tag)
        rows.append({
            "label": label.replace("\n", " "),
            "pmin": df.p_benefit.min(), "pmax": df.p_benefit.max(),
            "lo": df.p_benefit_lo95.min(), "hi": df.p_benefit_hi95.max(),
        })

    # decision zones
    ax.axvspan(0.0, 0.5, color=C_NOTSUP, alpha=0.07, zorder=0)
    ax.axvspan(0.5, 0.8, color=C_UNCERT, alpha=0.10, zorder=0)
    ax.axvspan(0.8, 1.0, color=C_SUP, alpha=0.10, zorder=0)
    ax.axvline(0.5, color="#555555", lw=1.1, ls="--")
    ax.axvline(0.8, color="#555555", lw=1.1, ls=":")

    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        supported = r["lo"] >= 0.8
        notsup = r["hi"] < 0.5
        col = C_SUP if supported else (C_NOTSUP if notsup else C_UNCERT)
        verdict = ("sign supported" if supported else
                   ("sign not supported" if notsup else "uncertain"))
        # case description sits ABOVE its bar (frees the y-axis of long labels)
        ax.text(0.012, yi + 0.30, r["label"], ha="left", va="bottom", fontsize=10.0,
                color="#333333")
        # 95% CI envelope across the whole grid (thin) + point-estimate span (thick)
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, lw=2.0, alpha=0.55,
                solid_capstyle="round", zorder=3)
        ax.plot([r["pmin"], r["pmax"]], [yi, yi], color=col, lw=9.0,
                solid_capstyle="round", zorder=4)
        ax.plot([r["pmin"], r["pmax"]], [yi, yi], color="white", lw=1.0, zorder=5)
        ax.annotate(verdict, (max(r["hi"], r["pmax"]) + 0.02, yi), va="center",
                    ha="left", fontsize=10.0, color=col, fontweight="bold")

    ax.set_yticks([])
    ax.set_ylim(-0.7, len(rows) + 0.05)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("P(ΔH < 0):   probability the sign reduces net serious harm")
    ax.spines["left"].set_visible(False)
    # short zone captions along the top
    ax.text(0.25, len(rows) - 0.32, "not supported", ha="center", va="bottom",
            fontsize=10.0, color=C_NOTSUP)
    ax.text(0.65, len(rows) - 0.32, "uncertain", ha="center", va="bottom",
            fontsize=10.0, color="#4a4a4a")
    ax.text(0.90, len(rows) - 0.32, "supported", ha="center", va="bottom",
            fontsize=10.0, color=C_SUP)
    handles = [
        Line2D([0], [0], color=GREY, lw=9.0, solid_capstyle="round"),
        Line2D([0], [0], color=GREY, lw=2.0, alpha=0.55, solid_capstyle="round"),
    ]
    ax.legend(handles, ["P(ΔH<0) across the 49-scenario grid",
                        "95% Wilson-interval envelope"],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title("Decision is identical across all 49 scenarios; no case reaches "
                 "'supported'", loc="left", fontsize=10.5)
    _save(fig, "fig7_decision_classification")


# =========================================================================== #
# fig8 -- PRCC (stacked vertically so long parameter labels fit)
# =========================================================================== #
def fig8_prcc():
    boot = pd.read_csv(OUT / "sensitivity" / "prcc_bootstrap.csv")
    boot_g = pd.read_csv(OUT / "sensitivity" / "prcc_decision_scale_bootstrap.csv")
    nice = {"r_E_per_veh_km": "Encroachment rate r$_E$", "p_R": "Response probability p$_R$",
            "mu_v_kmh": "Mean baseline speed μ$_V$", "d_aw_m": "Warning distance d$_{AW}$",
            "v_walk_m_s": "Walking speed v$_{walk}$", "t_handle_s": "Handling time t$_{handle}$",
            "alpha_lat_m_inv": "Lateral reach α", "deltaV_kmh": "Speed reduction ΔV",
            "c_deploy_m": "Deploy offset c$_{deploy}$", "c_work_m": "Work offset c$_{work}$",
            "sigma_v_kmh": "Speed spread σ$_V$", "a_comfort_m_s2": "Deceleration a",
            "m_striking_kg": "Striking mass m$_1$", "m_lcv_kg": "Work-vehicle mass m$_2$"}
    dur_colors = {0.05: OI_SKY, 0.25: OI_BLUE, 2.0: "#083b5c"}

    fig, axes = plt.subplots(2, 1, figsize=(W_IN, 15.0 * CM), constrained_layout=True)
    panels = ((axes[0], boot, "(a) Sensitivity of ΔH  (harm scale)"),
              (axes[1], boot_g, "(b) Sensitivity of ΔH / r$_E$  (decision scale)"))
    for ax, data, title in panels:
        order = (data.groupby("parameter").abs_PRCC.mean()
                 .sort_values(ascending=True).tail(8).index.tolist())
        ys = np.arange(len(order))
        h = 0.26
        for k, (T, c) in enumerate(dur_colors.items()):
            sub = data[data.T_work_h == T].set_index("parameter").reindex(order)
            ax.barh(ys + (k - 1) * h, sub.PRCC, height=h, color=c, label=T_LABELS[T])
            ax.errorbar(sub.PRCC, ys + (k - 1) * h,
                        xerr=[sub.PRCC - sub.ci_lo, sub.ci_hi - sub.PRCC],
                        fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
        ax.set_yticks(ys)
        ax.set_yticklabels([nice.get(p, p) for p in order])
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel("PRCC  (95% bootstrap CI)")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlim(-1, 1)
    axes[0].legend(frameon=False, title="Work duration", loc="lower right", ncol=3,
                   columnspacing=1.1, handlelength=1.3)
    _save(fig, "fig8_prcc")


# =========================================================================== #
# fig9 -- KE requirement
# =========================================================================== #
def fig9_ke_requirement():
    digest = json.loads((OUT / "results_digest.json").read_text())
    ke = digest["ke_requirements"]
    fig, ax = plt.subplots(figsize=(W_IN, 8.4 * CM), constrained_layout=True)
    Ts = sorted(float(t) for t in ke["0.5"].keys())
    vals = [ke["0.5"][str(T)] for T in Ts]
    ax.plot(Ts, vals, marker="o", ms=6, lw=1.8, color=OI_BLUE,
            label="Required departure reduction")
    for T, v in zip(Ts, vals):
        ax.annotate(f"{int(round(v*100))}%", (T, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10.0, color="#1a1a1a")
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([T_LABELS[t] for t in Ts], rotation=40, ha="right")
    ax.set_yticks([0.0, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0%", "5%", "10%", "15%"])
    ax.set_ylim(0, 0.18)
    ax.set_xlabel("Work duration")
    ax.set_ylabel("Minimum reduction in\nwork-period road departures")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Requirement is identical for the p* = 0.5 and p* = 0.8 thresholds",
                 loc="left", fontsize=10.5)
    _save(fig, "fig9_ke_requirement")


# =========================================================================== #
# figS1 -- convergence
# =========================================================================== #
def figS1_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 8.0 * CM), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    files = sorted((OUT / "convergence").glob("convergence_Q*.csv"))
    for k, f in enumerate(files):
        df = pd.read_csv(f)
        label = f.stem.replace("convergence_", "").replace("_", ", ").replace("p", ".")
        c = cmap(k / max(len(files) - 1, 1))
        axes[0].plot(df.n, df.running_mean * 1e6, lw=1.3, color=c, label=label)
        axes[1].plot(df.n, df.running_p_benefit, lw=1.3, color=c, label=label)
    axes[0].set_ylabel("Running mean ΔH\n(MAIS 3+ per million jobs)")
    axes[0].set_title("(a) Mean net harm", loc="left")
    axes[1].set_ylabel("Running P(ΔH < 0)")
    axes[1].set_title("(b) Probability of net benefit", loc="left")
    for ax in axes:
        ax.axvline(20000, color="#888888", lw=1.0, ls="--")
        ax.set_xlabel("Monte Carlo iterations")
    axes[1].legend(frameon=False, ncol=1, loc="upper right")
    axes[1].set_ylim(0, 0.3)
    _save(fig, "figS1_convergence")


if __name__ == "__main__":
    print("figures ->", FIG)
    fig0a_worksite_layout()
    fig0b_parameter_map()
    fig1_model_schematic()
    fig2_warning_effect()
    fig3_severity_curves()
    fig4_probability_heatmaps()
    fig5_mean_dh_heatmaps()
    fig6_p_vs_duration()
    fig7_decision_classification()
    fig8_prcc()
    fig9_ke_requirement()
    figS1_convergence()

    (FIG / "_font_audit.json").write_text(json.dumps(_AUDIT, indent=2))
    min_obs = min(a["min_observed_pt"] for a in _AUDIT)
    print("\nFont/size audit (min observed pt at 16.5 cm embed):")
    for a in _AUDIT:
        flag = "" if a["min_observed_pt"] >= 9.0 else "  <<< BELOW 9pt"
        print(f"  {a['figure']:26s} {a['width_cm']:5.1f}x{a['height_cm']:4.1f}cm  "
              f"design={a['min_design_pt']:.1f}pt  observed={a['min_observed_pt']:.2f}pt{flag}")
    print(f"\nGLOBAL min observed size = {min_obs:.2f} pt  "
          f"({'PASS' if min_obs >= 9.0 else 'FAIL'} >= 9 pt)")
