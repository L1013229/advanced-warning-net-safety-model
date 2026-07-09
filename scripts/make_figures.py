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
    """Plan view. New Zealand (left-hand traffic): the worksite sits on the left
    shoulder, so the ADJACENT (near) lane runs left-to-right and passes the sign
    before it reaches the worker. The opposing lane runs right-to-left."""
    H = 9.15
    fig, ax = _schematic(W_CM, H)

    x0, x1 = 0.55, 15.95
    road_lo, road_hi = 0.70, 3.00          # travelled way band (cm)
    road_mid = (road_lo + road_hi) / 2     # centreline
    sign_x = 2.5
    worker_x = 11.95                       # worker on foot
    veh_x = 13.85                          # work-vehicle centre
    walk_y = 4.30                          # deployment walk path (c_deploy, exagg.)
    worker_y = 6.25                        # worker + vehicle (c_work, exagg.)
    daw_y = 8.15                           # d_AW dimension line

    # road surface + markings
    ax.add_patch(Rectangle((x0, road_lo), x1 - x0, road_hi - road_lo,
                           fc="#e9ecee", ec="none", zorder=0))
    ax.plot([x0, x1], [road_hi, road_hi], color="#2b2b2b", lw=1.8, zorder=2)   # edgeline
    ax.plot([x0, x1], [road_lo, road_lo], color="#7d868c", lw=1.0, zorder=2)
    ax.plot([x0, x1], [road_mid, road_mid], color="#c8a02a", lw=1.4,
            ls=(0, (7, 6)), zorder=2)                                          # centreline
    ax.text(8.6, road_hi + 0.13, "edgeline", ha="center", va="bottom",
            fontsize=10.0, color="#2b2b2b")

    # Near lane (adjacent to the worker) runs LEFT -> RIGHT, so traffic passes the
    # sign before the work area. Opposing lane runs RIGHT -> LEFT.
    near_y = road_lo + 0.75 * (road_hi - road_lo)
    far_y = road_lo + 0.25 * (road_hi - road_lo)
    _arrow(ax, (9.55, near_y), (14.45, near_y), color="#41586b", lw=1.7, ms=15)
    ax.text(12.0, near_y + 0.34, "direction of travel", ha="center", va="center",
            fontsize=10.0, color="#41586b")
    _arrow(ax, (14.45, far_y), (9.55, far_y), color="#93a3ad", lw=1.3, ms=13)
    ax.text(x0 + 0.15, far_y, "travelled way (two-way)", ha="left", va="center",
            fontsize=10.0, color="#4a4a4a")

    # advanced warning sign (upstream of the work area in the near lane)
    post_top = road_hi + 0.80
    ax.plot([sign_x, sign_x], [road_hi, post_top], color="#333333", lw=2.0, zorder=4)
    ax.add_patch(Polygon([[sign_x, post_top], [sign_x - 0.36, post_top + 0.64],
                          [sign_x + 0.36, post_top + 0.64]], closed=True,
                         fc=OI_ORANGE, ec="#5a3d00", lw=1.3, zorder=5))
    ax.text(sign_x, post_top + 0.95, "portable\nadvanced warning sign", ha="center",
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
    ax.text(worker_x, worker_y + 0.50, "worker on foot", ha="center", va="bottom",
            fontsize=10.0, color="#5a2400")

    # deployment walk path (there and back) at offset c_deploy
    ax.add_patch(FancyArrowPatch((worker_x, walk_y), (sign_x + 0.48, walk_y),
                                 arrowstyle="<|-|>", mutation_scale=12, lw=1.8,
                                 color=OI_VERM, ls=(0, (5, 3)), zorder=3))
    ax.text(7.9, walk_y + 0.16, "on-foot deployment walk\n(place + remove)",
            ha="center", va="bottom", fontsize=10.0, color=OI_VERM, linespacing=1.2)

    # dimension: d_AW (dotted leaders start clear of every label)
    ax.plot([sign_x, sign_x], [5.78, daw_y], color="#aaaaaa", lw=0.7, ls=":", zorder=1)
    ax.plot([worker_x, worker_x], [7.12, daw_y], color="#aaaaaa", lw=0.7, ls=":",
            zorder=1)
    _dim(ax, (sign_x, daw_y), (worker_x, daw_y),
         "d$_{AW}$ ≈ 50 m   (advanced warning distance)", pt=10.0, lab_dy=0.26)

    # dimension: c_deploy (lateral)
    _dim(ax, (4.95, road_hi), (4.95, walk_y), "", pt=10.0)
    ax.text(4.70, (road_hi + walk_y) / 2, "c$_{deploy}$\n(0.5-2 m)", ha="right",
            va="center", fontsize=10.0, color="#333333", linespacing=1.2)
    # dimension: c_work (lateral); label sits in the clear band above the walk label
    _dim(ax, (11.30, road_hi), (11.30, worker_y), "", pt=10.0)
    ax.text(11.05, 5.72, "c$_{work}$ (1-3 m)", ha="right", va="center",
            fontsize=10.0, color="#333333")

    _save(fig, "fig0a_worksite_layout")


# =========================================================================== #
# fig0b -- parameter map (NEW)
# =========================================================================== #
def _box_h(n_lines: int, *, title: bool = True) -> float:
    """Height that actually contains the text _box() draws.

    _box() starts the title 0.42 cm below the top, drops 0.62 cm to the body, and
    each body line occupies ~0.47 cm at 10 pt with linespacing 1.35. Undersized
    boxes were letting body text spill through the bottom border.
    """
    top = 0.42 + (0.62 if title else 0.0)
    return top + 0.47 * n_lines + 0.34


def fig0b_parameter_map():
    H = 13.4
    fig, ax = _schematic(W_CM, H)

    cream, cream_e = "#f4efe3", "#9c7f3d"
    blue_f, blue_e = "#e7f0f7", OI_BLUE
    red_f, red_e = "#fbebe3", OI_VERM
    green_f, green_e = "#e6f4ec", OI_GREEN

    ax.text(W_CM / 2, H - 0.30, "Inputs  →  three collision pathways  →  decision",
            ha="center", va="top", fontsize=11.0, fontweight="bold", color="#333333")

    # ---- Column 1: input groups (top-down, each box sized to its own text) ----
    ix, iw = 0.35, 4.15
    groups = [
        ("Encroachment", ["rate r$_E$, reach α", "offsets c, lengths L"]),
        ("Speed environment", ["μ$_V$, σ$_V$  →  V₀"]),
        ("Sign response", ["p$_R$, ΔV, a, d$_{AW}$  →  V₁"]),
        ("Deployment", ["v$_{walk}$, t$_{handle}$  →  T$_{deploy}$"]),
        ("Severity curves", ["p$_w$(V),  p$_o$(ΔV)"]),
    ]
    icy = []
    top = H - 1.35
    for title, lines in groups:
        h = _box_h(len(lines))
        y = top - h
        _box(ax, ix, y, iw, h, lines, cream, cream_e, title=title, title_pt=10.5,
             body_pt=10.0, align="left")
        icy.append(y + h / 2)
        top = y - 0.40

    # ---- Column 2: three pathways ----
    px, pw = 5.95, 5.35
    ph = _box_h(2)
    paths = [
        ("Pathway 1", ["Worker strike (work)", "λ$_w$ · p$_w$(V)"], 9.30, blue_f, blue_e),
        ("Pathway 2", ["Work-vehicle strike (work)", "λ$_v$ · p$_o$(ΔV$_{occ}$)"], 5.95, blue_f, blue_e),
        ("Pathway 3  (S₁ only)", ["Worker strike (deployment)", "λ$_d$ · p$_w$(V₀)"], 2.60, red_f, red_e),
    ]
    pcy = []
    for title, lines, y, fc, ec in paths:
        _box(ax, px, y, pw, ph, lines, fc, ec, title=title, title_pt=10.5,
             body_pt=10.0, align="center")
        pcy.append(y + ph / 2)

    # ---- Column 3: outputs (Decision sits between the two harm terms) ----
    ox, ow = 12.05, 4.10
    cx = ox + ow / 2
    h0_h, dec_h, h1_h = _box_h(1), _box_h(3), _box_h(2)
    h0_y, dec_y, h1_y = 9.75, 5.60, 1.60
    _box(ax, ox, h0_y, ow, h0_h, ["P1(V₀) + P2(V₀)"], blue_f, blue_e,
         title="H(S₀)  no sign", title_pt=10.5, body_pt=10.0)
    _box(ax, ox, dec_y, ow, dec_h, ["ΔH = H(S₁) - H(S₀)", "", "deploy if P(ΔH<0) ≥ p*"],
         green_f, green_e, title="Decision", title_pt=10.5, body_pt=10.0)
    _box(ax, ox, h1_y, ow, h1_h, ["P1(V₁) + P2(V₁)", "+ P3 deploy"], red_f, red_e,
         title="H(S₁)  sign", title_pt=10.5, body_pt=10.0)

    # ---- arrows: input groups -> pathways ----
    a_in = dict(color=LGREY, lw=1.2, ms=11)
    _arrow(ax, (ix + iw, icy[0]), (px, pcy[0] + 0.45), **a_in)       # encroachment -> P1
    _arrow(ax, (ix + iw, icy[0] - 0.25), (px, pcy[1] + 0.45), **a_in)  # encroachment -> P2
    _arrow(ax, (ix + iw, icy[1]), (px, pcy[0] - 0.10), **a_in)       # speed -> P1
    _arrow(ax, (ix + iw, icy[2]), (px, pcy[1] - 0.10), **a_in)       # response -> P2
    _arrow(ax, (ix + iw, icy[3]), (px, pcy[2] + 0.30), **a_in)       # deployment -> P3
    _arrow(ax, (ix + iw, icy[4]), (px, pcy[2] - 0.30), **a_in)       # severity -> P3

    # ---- pathways -> harm terms.
    # P1 and P2 feed BOTH strategies (evaluated at V0 and at V1), so they are
    # bracketed once and routed with two labelled arrows instead of four crossing
    # ones. P3 exists only under S1.
    # The V0 / V1 evaluation is already written inside the H(S0) and H(S1) boxes,
    # so the arrows carry no text and nothing lands on a box.
    a_p = dict(color=GREY, lw=1.5, ms=13)
    bx = px + pw + 0.42
    ax.plot([bx, bx], [pcy[1], pcy[0]], color=GREY, lw=1.4, solid_capstyle="round")
    ax.plot([px + pw, bx], [pcy[0], pcy[0]], color=GREY, lw=1.4)
    ax.plot([px + pw, bx], [pcy[1], pcy[1]], color=GREY, lw=1.4)
    bmid = (pcy[0] + pcy[1]) / 2
    _arrow(ax, (bx, bmid), (ox, h0_y + 0.55), **a_p)                 # P1+P2 -> H(S0)
    _arrow(ax, (bx, bmid), (ox, h1_y + h1_h - 0.55), **a_p)          # P1+P2 -> H(S1)
    _arrow(ax, (px + pw, pcy[2]), (ox, h1_y + 0.62), color=OI_VERM, lw=1.6, ms=13)

    # both harm terms -> decision (down from H0, up from H1; no box piercing)
    _arrow(ax, (cx, h0_y), (cx, dec_y + dec_h), **a_p)               # H0 -> decision
    _arrow(ax, (cx, h1_y + h1_h), (cx, dec_y), **a_p)                # H1 -> decision

    _save(fig, "fig0b_parameter_map")


# =========================================================================== #
# fig2 -- warning effect
# =========================================================================== #
def fig2_warning_effect():
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 9.9 * CM), constrained_layout=True)

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
    axes[0].set_ylim(0, 0.056)      # headroom so the legend clears the V0 peak
    axes[0].legend(frameon=False, loc="upper left", fontsize=10.0)
    axes[0].set_title("(a) Bernoulli mixture of responders", loc="left")

    # (b) The sampled reduction is ΔV ~ U(0,15) km/h, CAPPED by what a driver can
    # actually shed over d_AW at a ~ U(0.8, 2.5) m/s^2. The question the panel must
    # answer is simply: where does that cap bind? Six double-encoded curves buried
    # it, so show the bound as an envelope against the sampled support instead.
    def _bound(v, a, d):
        return np.minimum(feasible_delta_v_kmh(v, np.full_like(v, a),
                                               np.full_like(v, d)), v)

    v0 = np.linspace(20, 80, 400)
    lo = _bound(v0, 0.8, 30.0)      # shortest distance, gentlest braking
    hi = _bound(v0, 2.5, 70.0)      # longest distance, firmest braking
    typ = _bound(v0, 1.65, 50.0)    # mean deceleration at the typical distance

    ax_b = axes[1]
    ax_b.axvspan(35, 60, color="#ececec", zorder=0, label="modelled speed range")
    ax_b.axhspan(0, 15, color=C_SIGN, alpha=0.13, lw=0, zorder=1,
                 label="sampled ΔV (≤ 15 km/h)")
    ax_b.fill_between(v0, lo, hi, color=OI_BLUE, alpha=0.20, lw=0, zorder=2,
                      label="bound, d$_{AW}$ 30-70 m")
    ax_b.plot(v0, typ, color=OI_BLUE, lw=1.9, zorder=3, label="bound, typical")
    ax_b.plot(v0, lo, color=OI_BLUE, lw=0.9, ls=":", zorder=3)
    ax_b.plot(v0, hi, color=OI_BLUE, lw=0.9, ls=":", zorder=3)

    # Where the worst-case bound drops under the 15 km/h cap, the kinematics -- not
    # the sampled cap -- limit the achievable reduction.
    # Mark where the worst-case bound crosses the cap; the sentence explaining it
    # lives in the caption, so no text sits on a curve.
    below = np.where(lo < 15.0)[0]
    if below.size:
        v_cross = float(v0[below[0]])
        ax_b.plot([v_cross], [15.0], marker="o", ms=4.5, color="#333333", zorder=5)
        ax_b.annotate(f"≈{v_cross:.0f} km/h", xy=(v_cross, 15.0), xytext=(33.0, 5.5),
                      fontsize=10.0, color="#333333", ha="left", va="center",
                      arrowprops=dict(arrowstyle="-", color="#666666", lw=0.9))

    ax_b.set_xlim(20, 80)
    ax_b.set_ylim(0, 74)
    ax_b.set_xlabel("Approach speed V$_0$ (km/h)")
    ax_b.set_ylabel("Achievable ΔV (km/h)")
    ax_b.set_title("(b) Kinematic bound on ΔV", loc="left")
    ax_b.legend(frameon=False, loc="upper left", fontsize=10.0, handlelength=1.5,
                borderpad=0.2, labelspacing=0.3, handletextpad=0.5)
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
    # Sit the band label vertically along the band's leading edge, in the empty
    # wedge above both curves and below the legend, so it never crosses a line.
    axes[0].text(37.3, 0.55, "modelled speed range", rotation=90, ha="center",
                 va="center", fontsize=10.0, color="#555555")
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
# fig4 -- P(dH<0) vs duration, with the traffic-flow range as a band.
#
# This replaces the old 2x2 P heatmap AND the old 2x2 P-vs-duration line grid.
# Both plotted the same quantity over the same Q x T grid: the heatmap spent a
# full 0-1 diverging scale on data that never left 0.12-0.16 or 0.51-0.60, and
# the line grid drew three Q series that landed exactly on top of one another.
# The reason both looked empty is itself the finding -- P is invariant to flow --
# so plot the flow range as a band and let its width carry that result.
# =========================================================================== #
def fig4_p_vs_duration():
    fig, ax = plt.subplots(figsize=(W_IN, 11.2 * CM), constrained_layout=True)

    # colour = response assumption, line style = deployment speed. Encoding the two
    # factors separately keeps the legend to four short entries instead of four
    # long ones that squeezed the axes.
    series = [
        ("baseline", C_SIGN, "-", "o"),
        ("baseline_fastDeploy", C_SIGN, "--", "s"),
        ("highPR", OI_BLUE, "-", "o"),
        ("high_PR_fastDeploy", OI_BLUE, "--", "s"),
    ]

    # decision zones
    ax.axhspan(0.0, 0.5, color=C_NOTSUP, alpha=0.06, lw=0, zorder=0)
    ax.axhspan(0.5, 0.8, color=C_UNCERT, alpha=0.09, lw=0, zorder=0)
    ax.axhspan(0.8, 1.0, color=C_SUP, alpha=0.10, lw=0, zorder=0)
    ax.axhline(0.5, color="#777777", lw=1.0, ls="--", zorder=1)
    ax.axhline(0.8, color="#777777", lw=1.0, ls=":", zorder=1)

    widest = 0.0
    for tag, c, ls, mk in series:
        df = g(tag)
        grp = df.groupby("T_work_h").p_benefit
        Ts = np.array(sorted(df.T_work_h.unique()))
        lo, hi = grp.min().reindex(Ts).values, grp.max().reindex(Ts).values
        mid = grp.mean().reindex(Ts).values
        widest = max(widest, float(np.max(hi - lo)))
        ax.fill_between(Ts, lo, hi, color=c, alpha=0.30, lw=0, zorder=2)
        ax.plot(Ts, mid, color=c, ls=ls, marker=mk, ms=4.5, lw=1.7, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(list(T_LABELS))
    ax.set_xticklabels(list(T_LABELS.values()))
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlim(0.043, 4.8)
    ax.set_xlabel("Work duration")
    ax.set_ylabel("P(ΔH < 0)")

    # zone captions in the empty left margin of each band, clear of every series
    for yv, txt, col in ((0.30, "sign not supported", C_NOTSUP),
                         (0.70, "uncertain", "#4a4a4a"),
                         (0.91, "sign supported", C_SUP)):
        ax.text(0.046, yv, txt, ha="left", va="center", fontsize=10.0, color=col)

    ax.set_title(f"Band spans the seven traffic flows (50-1000 veh/h); width "
                 f"≤ {widest:.3f},\nso P(ΔH < 0) is effectively independent of flow",
                 loc="left", fontsize=10.5, color="#333333")

    handles = [
        Line2D([0], [0], color=C_SIGN, lw=2.6),
        Line2D([0], [0], color=OI_BLUE, lw=2.6),
        Line2D([0], [0], color="#444444", lw=1.7, ls="-", marker="o", ms=4.5),
        Line2D([0], [0], color="#444444", lw=1.7, ls="--", marker="s", ms=4.5),
    ]
    labels = ["Standard response", "Optimistic response",
              "Standard deployment", "Faster deployment"]
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=2,
               fontsize=10.0, handlelength=2.2, columnspacing=2.4)
    _save(fig, "fig4_p_vs_duration")


# =========================================================================== #
# fig5 -- mean dH heatmaps
# =========================================================================== #
def fig5_mean_dh_heatmaps():
    """Mean dH over the Q x T grid.

    Per-cell numbers were removed. Mean dH spans four orders of magnitude across
    the grid (1.5e-4 to 4.3 per 10^6 jobs), so no fixed-decimal label resolves it:
    at one decimal place a third of the cells printed a bare "0" while staying
    strongly coloured, and at two decimal places the labels collided across the
    seven columns. The symlog colourbar already carries magnitude and sign, so the
    only thing missing was the sign change itself -- now drawn as the dH = 0
    contour. Exact representative values are tabulated in Table 3.
    """
    fig, axes = plt.subplots(2, 2, figsize=(W_IN, 14.6 * CM), sharex=True, sharey=True,
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
        # dH = 0 boundary: where the sign stops adding harm and starts removing it
        if piv.values.min() < 0.0 < piv.values.max():
            nr, nc = piv.shape
            ax.contour(np.arange(nc), np.arange(nr), piv.values, levels=[0.0],
                       colors="#1a1a1a", linewidths=1.4, linestyles="--")
    for ax in axes[1]:
        ax.set_xlabel("Traffic flow Q (veh/h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Work duration")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02, aspect=32)
    cb.set_label("Mean ΔH  (MAIS 3+ per 10⁶ jobs)")
    fig.suptitle("Blue = sign reduces mean harm (ΔH < 0);  red = sign adds harm;  "
                 "dashed line = ΔH = 0", fontsize=10.5, color="#333333")
    _save(fig, "fig5_mean_dh_heatmaps")


# =========================================================================== #
# fig7 -- decision summary (REDESIGNED from block-colour heatmaps)
# =========================================================================== #
def fig7_decision_classification():
    fig, ax = plt.subplots(figsize=(W_IN, 9.6 * CM), constrained_layout=True)

    rows = []
    for tag, label in CASES:
        df = g(tag)
        rows.append({
            "label": label,                      # keep the two-line form for the y axis
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
        # Match the zone captions verbatim and keep them short enough that the
        # label never runs across the p* = 0.5 threshold line.
        verdict = ("supported" if supported else
                   ("not supported" if notsup else "uncertain"))
        # 95% CI envelope across the whole grid (thin) + point-estimate span (thick)
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, lw=2.0, alpha=0.55,
                solid_capstyle="round", zorder=3)
        ax.plot([r["pmin"], r["pmax"]], [yi, yi], color=col, lw=9.0,
                solid_capstyle="round", zorder=4)
        ax.plot([r["pmin"], r["pmax"]], [yi, yi], color="white", lw=1.0, zorder=5)
        ax.annotate(verdict, (max(r["hi"], r["pmax"]) + 0.02, yi), va="center",
                    ha="left", fontsize=10.0, color=col, fontweight="bold")

    # Case descriptions belong on the y axis. Drawn inside the axes they ran
    # straight through the p* = 0.5 threshold line.
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.15)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability the sign reduces net serious harm, P(ΔH < 0)")
    ax.spines["left"].set_visible(False)
    # short zone captions along the top
    ax.text(0.25, len(rows) - 0.55, "not supported", ha="center", va="bottom",
            fontsize=10.0, color=C_NOTSUP)
    ax.text(0.65, len(rows) - 0.55, "uncertain", ha="center", va="bottom",
            fontsize=10.0, color="#4a4a4a")
    ax.text(0.90, len(rows) - 0.55, "supported", ha="center", va="bottom",
            fontsize=10.0, color=C_SUP)
    handles = [
        Line2D([0], [0], color=GREY, lw=9.0, solid_capstyle="round"),
        Line2D([0], [0], color=GREY, lw=2.0, alpha=0.55, solid_capstyle="round"),
    ]
    fig.legend(handles, ["P(ΔH<0) across the 49-scenario grid",
                         "95% Wilson-interval envelope"],
               frameon=False, loc="outside lower center", ncol=2, columnspacing=2.4)
    # Long two-line y labels narrow the axes, so keep the title short enough to fit.
    ax.set_title("Decision is identical across all 49 scenarios", loc="left",
                 fontsize=10.5)
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
        # No PRCC (or CI bound) leaves [-0.35, 0.80]; a full [-1, 1] axis spent
        # half its width on emptiness and forced the legend on top of the bars.
        ax.set_xlim(-0.42, 0.86)
    # One legend for both panels, below the figure -- never over the data.
    # "outside lower center" makes constrained_layout reserve the strip for it.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, title="Work duration",
               loc="outside lower center", ncol=3, columnspacing=1.6,
               handlelength=1.4)
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
    ax.plot(Ts, vals, marker="o", ms=6, lw=1.8, color=OI_BLUE)
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
    # Single series, already point-labelled -- a legend adds nothing.
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
    # Manuscript figure order (9 figures):
    #   1 fig0a  2 fig0b  3 fig2  4 fig3  5 fig4  6 fig5  7 fig7  8 fig8  9 fig9
    # fig1_model_schematic is superseded by fig0b and is not used in the paper.
    fig0a_worksite_layout()
    fig0b_parameter_map()
    fig2_warning_effect()
    fig3_severity_curves()
    fig4_p_vs_duration()
    fig5_mean_dh_heatmaps()
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
