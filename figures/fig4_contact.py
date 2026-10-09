#!/usr/bin/env python
"""Render the two data panels of Figure 4 (contact), one file pair per panel.

The paper's Figure 4 places these panels next to a hand-drawn robot sketch, in a layout made by
hand; this script writes each data panel as its own figure:

  fig4a  natural VLM activations (text-token mean, layer 24) on their top three principal
         components, 6,000 of the 18,837 queries, coloured by contact; floor, shadow, points faded
         with camera depth; "PC1 (9.1%)" / "PC2 (7.6%)" along the floor edges        11 x 9 in
  fig4b  off-schedule detection: fraction of poses where the held-out-house probe scores contact
         placements above no-contact placements, for early touch (hand open / forced closed) and
         induced slip (hand closed / forced open), binomial SE bars; dashed line = the probe's
         out-of-house accuracy on the natural rows ("baseline"), dotted line = chance  9.2 x 5.6 in

Input: <panels>/contact.npz (pipelines/contact/70_panel.py; default $VLAGEOM_DATA/panels):
  P, contact, evr, show (panel a); bar_cells, frac_pos, frac_pos_se, n_pose, baseline_acc (panel b).

Output: <out-dir>/fig4a.pdf, fig4a.png (350 dpi, full canvas) and fig4b.pdf, fig4b.png (350 dpi,
tight bounding box), default `vlageom.paths.figures_dir()`.

Style: `vlageom.plot.style` (Product Sans from $VLAGEOM_FONT_DIR, Computer Modern math). The
colours are set as hex values: contact #2563eb, no contact #8b5cf6 (panel a); early touch
#2563eb / #93c5fd, slip #8b5cf6 / #c4b5fd (panel b, natural / forced hand).

Run:  python figures/fig4_contact.py [--panel a b] [--panels DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from vlageom import paths  # noqa: E402
from vlageom.plot import style as ps  # noqa: E402

PANELS = ("a", "b")
DPI = 350

# ---- panel a ----
A_FIGSIZE = (11, 9)
A_RECT3D = (-0.10, -0.06, 1.20, 1.12)       # 3-D axes rectangle (figure fractions)
A_ELEV, A_AZIM = 20.0, -55.0
COL_ON, COL_OFF = ps.PRIMARY, ps.VIOLET     # contact / no contact (#2563eb, #8b5cf6)
CLOUD_SIZE = 14.0
CLOUD_ALPHA = 1.15                          # scale of the base alpha 0.30
DEPTH_NEAR, DEPTH_FAR = 2.3, 0.22           # alpha factors of the nearest / farthest points
FLOOR_COLOR, FLOOR_ALPHA = "#F8FAFC", 0.92
FLOOR_PAD = 1.12                            # square floor size relative to the 1-99% range + 8%
SHADOW_ALPHA = 0.06
LEGEND_FONTSIZE = 30.0
PC_LABEL_FONTSIZE = 26.0
PC_LABEL_PAD = 0.18                         # label offset outside the floor edge / floor half-width

# ---- panel b ----
B_FIGSIZE = (9.2, 5.6)
B_FS = dict(tick=22, axis=26, group=24, legend=20, value=20, tag=18)
B_BARS = [("early_touch_open", 0.0, "#2563eb", "Open\n(natural)"),
          ("early_touch_closed", 1.0, "#93c5fd", "Closed\n(forced)"),
          ("slip_closed", 2.5, "#8b5cf6", "Closed\n(natural)"),
          ("slip_open_out", 3.5, "#c4b5fd", "Open\n(forced)")]
B_GROUPS = [(0.0, 1.0, "Early touch"), (2.5, 3.5, "Induced slip")]
B_YTICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_panel(panels_dir: Path) -> dict:
    f = panels_dir / "contact.npz"
    if not f.exists():
        raise SystemExit(f"{f} not found; run pipelines/contact/70_panel.py first")
    with np.load(f, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- panel a
def render_a(p, stem: Path):
    """3-D cloud of the natural rows on PC1-3, coloured by contact."""
    ps.apply_style()
    P, contact, evr = p["P"], p["contact"], p["evr"]
    show = p["show"]
    Pv, cv = P[show], contact[show]

    def _rng2(i):
        lo, hi = np.percentile(P[:, i], [1, 99])
        return lo, hi
    xlo, xhi = _rng2(0)
    ylo, yhi = _rng2(1)
    zlo, zhi = _rng2(2)
    xpad = (xhi - xlo) * 0.08
    ypad = (yhi - ylo) * 0.08
    zpad = (zhi - zlo) * 0.06
    xlo, xhi, ylo, yhi = xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    half = max(xhi - xlo, yhi - ylo) / 2 * FLOOR_PAD
    xlo, xhi, ylo, yhi = cx - half, cx + half, cy - half, cy + half
    zfloor = zlo - zpad
    shadow_z = zfloor + (zhi - zfloor) * 0.01

    fig = plt.figure(figsize=A_FIGSIZE, facecolor="white")
    ax = fig.add_axes([float(v) for v in A_RECT3D], projection="3d")
    ax.set_facecolor("white")
    ax.computed_zorder = False
    ax.add_collection3d(Poly3DCollection(
        [[[xlo, ylo, zfloor], [xhi, ylo, zfloor], [xhi, yhi, zfloor], [xlo, yhi, zfloor]]],
        facecolor=FLOOR_COLOR, alpha=FLOOR_ALPHA, edgecolor="none", zorder=0))
    ax.scatter(Pv[:, 0], Pv[:, 1], np.full(len(Pv), shadow_z), c=ps.INK, s=CLOUD_SIZE * 0.9,
               depthshade=False, edgecolors="none", alpha=SHADOW_ALPHA, rasterized=True, zorder=2)

    # two colours, per-point alpha faded with camera depth, drawn far to near
    e_, a_ = np.radians(A_ELEV), np.radians(A_AZIM)
    cam = np.array([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_), np.sin(e_)])
    depth = (Pv - Pv.mean(0)) @ cam
    lo, hi = np.percentile(depth, [2, 98])
    dn = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1)
    afac = DEPTH_FAR + (DEPTH_NEAR - DEPTH_FAR) * dn
    base_a = 0.30 * CLOUD_ALPHA
    rgb = np.where(cv[:, None], matplotlib.colors.to_rgb(COL_ON), matplotlib.colors.to_rgb(COL_OFF))
    rgba = np.clip(np.concatenate([rgb, (base_a * afac)[:, None]], 1), 0, 1)
    order = np.argsort(depth)
    ax.scatter(Pv[order, 0], Pv[order, 1], Pv[order, 2], c=rgba[order], s=CLOUD_SIZE,
               depthshade=False, edgecolors="none", rasterized=True, zorder=4)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zfloor, zhi + (zhi - zfloor) * 0.02)
    ax.set_box_aspect((xhi - xlo, yhi - ylo, zhi - zfloor))
    ax.view_init(elev=A_ELEV, azim=A_AZIM)
    ax.set_axis_off()

    # PC1 / PC2 labels in the floor plane, along the two floor edges nearest the camera; zdir is
    # the edge direction (Text3D sets the on-screen angle from it at draw time)
    y_e, ysgn = (ylo, -1) if cam[1] < 0 else (yhi, 1)
    x_e, xsgn = (xhi, 1) if cam[0] > 0 else (xlo, -1)
    pad = half * PC_LABEL_PAD
    for text, dirvec, pos in [
        (f"PC1 ({100 * evr[0]:.1f}%)", (1, 0, 0), (cx, y_e + ysgn * pad, zfloor)),
        (f"PC2 ({100 * evr[1]:.1f}%)", (0, 1, 0), (x_e + xsgn * pad, cy, zfloor)),
    ]:
        ax.text(*pos, text, zdir=dirvec, color=ps.INK, fontsize=PC_LABEL_FONTSIZE,
                ha="center", va="center", zorder=30)

    ms = LEGEND_FONTSIZE * 13 / 22
    handles = [Line2D([], [], ls="none", marker="o", ms=ms, mfc=COL_ON, mec="none", label="contact"),
               Line2D([], [], ls="none", marker="o", ms=ms, mfc=COL_OFF, mec="none", label="no contact")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.94, 0.95),
               fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.9, edgecolor="none")

    stem.parent.mkdir(parents=True, exist_ok=True)
    outs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outs[0], dpi=DPI, facecolor="white", transparent=False)
    fig.savefig(outs[1], dpi=DPI, facecolor="white", transparent=False)
    plt.close(fig)
    return outs


# ----------------------------------------------------------------------------- panel b
def render_b(p, stem: Path):
    """Bars of P(correct) per cell with SE, the natural-probe baseline and chance."""
    cells = [str(c) for c in p["bar_cells"]]
    val_of = {c: float(v) for c, v in zip(cells, p["frac_pos"])}
    se_of = {c: float(v) for c, v in zip(cells, p["frac_pos_se"])}
    n_of = {c: int(v) for c, v in zip(cells, p["n_pose"])}
    nat = float(p["baseline_acc"])

    ps.apply_style()
    plt.rcParams.update({"font.size": B_FS["tick"], "axes.labelsize": B_FS["axis"],
                         "xtick.labelsize": B_FS["tick"], "ytick.labelsize": B_FS["tick"]})
    fig, ax = plt.subplots(figsize=B_FIGSIZE)
    for k, x, c, _ in B_BARS:
        val = val_of[k]
        se = se_of.get(k)
        if se is None:
            se = math.sqrt(val * (1.0 - val) / n_of[k])
        ax.bar(x, val, width=0.72, color=c, edgecolor="none")
        if se is not None and math.isfinite(se):
            ax.errorbar(x, val, yerr=se, fmt="none", ecolor=ps.INK,
                        elinewidth=1.2, capsize=3.5, capthick=1.2, zorder=4)
        else:
            se = 0.0
        ax.text(x, val + se + 0.012, f"{val:.3f}", ha="center", va="bottom",
                fontsize=B_FS["value"], color=ps.SLATE[3], zorder=5,
                bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none"))

    ax.axhline(nat, color=ps.META_LINE, ls="--", lw=1.2, zorder=0, label="baseline")
    ax.axhline(0.5, color=ps.INK, ls=":", lw=1.0, zorder=0, label="chance")
    ax.text(4.15, nat + 0.008, f"{nat:.3f}", ha="right", va="bottom",
            fontsize=B_FS["tag"], color=ps.META_LINE)

    ax.set_xticks([x for _, x, _, _ in B_BARS])
    ax.set_xticklabels([lab for _, _, _, lab in B_BARS])
    ax.set_ylabel("P(correct)")
    ax.set_xlim(-0.7, 4.2)
    ax.set_ylim(0.45, 1.0)
    ax.set_yticks(B_YTICKS)
    ps.finish(ax)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=B_FS["legend"],
              loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    # group labels below the tick labels, placed from the rendered tick-label extent; two
    # place-then-tight_layout passes, since tight_layout moves the axes
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    grp = [ax.text((x0 + x1) / 2, 0.0, lab, transform=trans, ha="center", va="top",
                   fontsize=B_FS["group"], color=ps.INK, clip_on=False)
           for x0, x1, lab in B_GROUPS]
    fig.tight_layout()
    for _ in range(2):
        fig.canvas.draw()
        inv = ax.transAxes.inverted()
        y_tick_bottom = min(inv.transform(t.get_window_extent().get_points())[0, 1]
                            for t in ax.get_xticklabels())
        for t in grp:
            t.set_y(y_tick_bottom - 0.04)
        fig.tight_layout()

    stem.parent.mkdir(parents=True, exist_ok=True)
    outs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outs[0], dpi=DPI, bbox_inches="tight", transparent=False)
    fig.savefig(outs[1], dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close(fig)
    return outs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--panel", nargs="+", choices=PANELS, default=list(PANELS),
                    help="panels to render (default: a b)")
    ap.add_argument("--panels", type=Path, default=None,
                    help="directory with contact.npz (default $VLAGEOM_DATA/panels)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default $VLAGEOM_FIGURES)")
    a = ap.parse_args(argv)
    panels_dir = a.panels if a.panels is not None else paths.panels_dir()
    out_dir = a.out_dir if a.out_dir is not None else paths.figures_dir(create=True)
    p = load_panel(panels_dir)
    render = {"a": render_a, "b": render_b}
    for name in a.panel:
        for f in render[name](p, out_dir / f"fig4{name}"):
            print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
