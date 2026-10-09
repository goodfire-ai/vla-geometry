#!/usr/bin/env python
"""Render the data panels of Figure 6 (wrist orientation).

Figure 6 is a composite: a hand-drawn robot sketch with the panel letters and the title, panel a,
and panels b-d placed as one block. This script renders the two data files the composite uses:

  fig6_a.pdf      panel a: the wrist roll sweep of one house on its own PC1-2 (action-expert
                  layer 9, action token 0), points coloured by the joint angle, with the path
                  through the points in angle order and a colourbar (7.4 x 6.2 in before the tight
                  bounding box, 350 dpi)
  fig6_bcd.pdf    panels b-d in one row (14.3 in wide): b, the natural rollouts' activations on
                  the chart's circle for rotation about the vertical axis, coloured by the hand's
                  yaw, with the circular mean per 10-degree yaw bin; c, the rotation of the decoded
                  action chunk after one edit of -40, -20, +20 or +40 deg (dots: scenes, ticks:
                  means, dashed diagonal: equal to the written angle); d, the executed wrist
                  rotation over 32 steps with the edit held at every query, in one house, for the
                  four written angles and the unedited rollout
  fig6_bcd.png    the same at 350 dpi

Input: $VLAGEOM_DATA/panels/fig6_a.npz (Z2, roll, evr) and fig6_bcd.npz (ph, yaw, r, bin_ph,
bin_ctr, c_theta, c_resp, d_theta, d_rot), written by pipelines/wrist_steer/70_panels.py.

Fonts: Product Sans from $VLAGEOM_FONT_DIR (else DejaVu Sans; see vlageom.plot.style). Panel b-d's
margins and gaps are measured from the rendered text, so with another font the axes move as well.
The graphite colour of panel c's dots and of panel b's direction arrow is set in this script.

Run:  python figures/fig6.py [--panel a|bcd|all] [--panels DIR] [--out-dir DIR] [--png-a]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors  # noqa: E402

from vlageom import paths  # noqa: E402
from vlageom.plot import style as ps  # noqa: E402

log = logging.getLogger("fig6")

CMAP_SPEC = "vanimo~plum:0.30-0.70"     # vanimo with its dark middle bridged through plum
GRAPHITE = ps.GRAPHITE                  # panel c dots, panel b direction arrow and theta
_CMAP_NAME = None


def wrist_cmap_name() -> str:
    """Registered name of the CMAP_SPEC colormap (registered on the first call)."""
    global _CMAP_NAME
    if _CMAP_NAME is None:
        _CMAP_NAME = ps.cmap_spec(CMAP_SPEC)
    return _CMAP_NAME


# ---- panel a ------------------------------------------------------------------------------------ #
A_FONT_SCALE = 1.75
A_CBAR_LABEL = r"wrist angle $\theta$ (deg)"


def render_a(panel: dict, out_pdf: Path, png: Path | None = None) -> None:
    """Panel a from the arrays of fig6_a.npz; writes `out_pdf` (and `png` at 350 dpi if given)."""
    s = A_FONT_SCALE
    ps.apply_style()
    plt.rcParams.update({
        "font.size": 14 * s, "axes.labelsize": 16 * s, "axes.titlesize": 17 * s,
        "xtick.labelsize": 14 * s, "ytick.labelsize": 14 * s, "legend.fontsize": 14 * s,
    })
    cmap = wrist_cmap_name()
    Z2, roll, evr = panel["Z2"], panel["roll"], panel["evr"]
    norm = colors.Normalize(vmin=float(roll.min()), vmax=float(roll.max()))
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig, a = plt.subplots(figsize=(7.4, 6.2))
    a.plot(Z2[:, 0], Z2[:, 1], "-", color=ps.META_LINE, lw=1.3, alpha=0.9, zorder=1)
    a.scatter(Z2[:, 0], Z2[:, 1], c=roll, cmap=cmap, norm=norm, s=72,
              edgecolors="none", alpha=0.95, zorder=3)
    a.set_xlabel(f"PC1 ({float(evr[0]) * 100:.0f}% var)")
    a.set_ylabel(f"PC2 ({float(evr[1]) * 100:.0f}% var)")
    a.set_aspect("equal", "datalim")
    ps.finish(a)
    ticks = [float(roll.min()), -90, 0, 90, float(roll.max())]   # the data ends, and -90, 0, 90
    cb = fig.colorbar(smap, ax=a, ticks=ticks, pad=0.02)
    cb.set_ticklabels([f"{t:.0f}" for t in ticks])
    cb.set_label(A_CBAR_LABEL, color=ps.INK, fontsize=15 * s)
    cb.ax.yaxis.set_tick_params(color=ps.INK, labelcolor=ps.INK, labelsize=13 * s)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=350, bbox_inches="tight", transparent=False)
    if png is not None:
        fig.savefig(png, dpi=350, bbox_inches="tight", transparent=False)
    plt.close(fig)


# ---- panels b-d --------------------------------------------------------------------------------- #
BCD_FONT_SCALE = 1.35
LINE_T = [0.10, 0.25, 0.75, 0.90]        # panel d line colours (colormap positions) for -40..+40
WIDTH = 14.3                             # figure width (in); the height follows the text
AXES_HEIGHT = 3.2                        # axes height (in)
PANEL_IN = [2.9, 3.5, 2.3]               # requested axes widths (in), scaled to the free width
CIRCLE_LABEL = "encoded wrist angle"     # caption under the circle
CIRCLE_LABEL_DY = 0.0
NORM = plt.Normalize(-180.0, 180.0)
BASE_LAB, BASE_ANN = 18.0, 13.0          # font sizes before scaling: labels / ticks, annotations

SPINE_OUT = 10 / 72          # ps.finish moves the left/bottom spines 10 pt outward
TICK_OUT = 7 / 72            # tick length 3.5 pt + tick-label pad 3.5 pt
LABPAD = 4 / 72              # axes.labelpad
CLEAR = 0.15                 # clearance between neighbouring text blocks (in)


def _measure(fig, text, fontsize, linespacing=1.3, rotation=0):
    """(width, height) in inches of `text` rendered at `fontsize`."""
    t = fig.text(0, 0, text, fontsize=fontsize, linespacing=linespacing, rotation=rotation)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def layout(fig, width, s, wrap, letters, axes_h, panel_in):
    """Margins and gaps (in) from the measured text extents at a given width; the height follows.
    Returns (figsize, gridspec kwargs, font sizes, panel widths, layout record)."""
    lab, ann = BASE_LAB * s, BASE_ANN * s
    ylabs = [("decoded wrist\nrotation (°)" if wrap else "decoded wrist rotation (°)"),
             ("executed wrist\nrotation (°)" if wrap else "executed wrist rotation (°)")]
    xlab_b = "rotation written\ninto the belief (°)" if wrap else "rotation written into the belief (°)"
    w_ann = max(_measure(fig, f"written in: {t:+.0f}°", ann)[0] for t in (-40, -20, 20, 40))
    right = w_ann + 4 / 72 + CLEAR
    w_tick = max(_measure(fig, t, lab)[0] for t in ("-20", "-50", "-10"))
    w_ylab = max(_measure(fig, t, lab, rotation=90)[0] for t in ylabs)
    gap = SPINE_OUT + TICK_OUT + w_tick + LABPAD + w_ylab + CLEAR
    h_xtick = _measure(fig, "-40", lab)[1]
    h_xlab = _measure(fig, xlab_b, lab)[1]
    bottom = SPINE_OUT + TICK_OUT + h_xtick + LABPAD + h_xlab + 0.12
    xlab_top = bottom - SPINE_OUT - TICK_OUT - h_xtick - LABPAD
    top = 0.45 if letters else 0.15
    left = 0.50
    height = bottom + axes_h + top
    axes_total = width - left - right - 2 * gap
    pw = np.asarray(panel_in, float) * axes_total / float(np.sum(panel_in))
    gs = dict(width_ratios=[float(x) for x in pw], wspace=float(gap / pw.mean()), left=left / width,
              right=1 - right / width, bottom=bottom / height, top=1 - top / height)
    fonts = dict(lab=lab, tick=lab, ann=ann, deg=lab, theta=lab, letter=lab * 19 / 18)
    w40 = _measure(fig, "-40", lab)[0]
    spacing_b = 20 / 104 * pw[1]
    diag = dict(right_in=round(right, 3), gap_in=round(gap, 3), bottom_in=round(bottom, 3),
                left_in=left, panel_in=[round(x, 3) for x in pw],
                xtick_b_spacing_in=round(spacing_b, 3), xtick_b_label_in=round(w40, 3),
                xtick_b_clearance_in=round(spacing_b - w40, 3), xlab_top_in=round(xlab_top, 3))
    return (width, height), gs, fonts, pw, diag


def render_bcd(panel: dict, out_pdf: Path, out_png: Path | None) -> dict:
    """Panels b-d from the arrays of fig6_bcd.npz; writes `out_pdf` and `out_png`. Returns the
    layout record."""
    s = BCD_FONT_SCALE
    wrap = s > 1.15
    letters = False
    ps.apply_style()
    fig = plt.figure(figsize=(WIDTH, 4.0))
    figsize, gskw, F, pw, diag = layout(fig, WIDTH, s, wrap, letters, AXES_HEIGHT, PANEL_IN)
    fig.set_size_inches(*figsize)
    log.info("layout %s", diag)
    plt.rcParams["axes.labelsize"] = F["lab"]
    plt.rcParams["xtick.labelsize"] = F["tick"]
    plt.rcParams["ytick.labelsize"] = F["tick"]
    cmap = matplotlib.colormaps[wrist_cmap_name()]
    ph, yaw, r = panel["ph"], panel["yaw"], float(panel["r"])
    gs = fig.add_gridspec(1, 3, **gskw)

    # ---- b: the natural cloud on the circle --------------------------------------------------- #
    ax = fig.add_subplot(gs[0])
    t = np.linspace(0, 2 * np.pi, 361)
    ax.plot(r * np.cos(t), r * np.sin(t), color=ps.META_LINE, ls="--", lw=1.2, zorder=0)
    ax.scatter(r * np.cos(ph), r * np.sin(ph), c=np.degrees(yaw), cmap=cmap, norm=NORM,
               s=9, alpha=0.35, edgecolors="none", zorder=1)
    mph, ctr = panel["bin_ph"], panel["bin_ctr"]
    ax.scatter(r * np.cos(mph), r * np.sin(mph), c=np.degrees(ctr), cmap=cmap, norm=NORM,
               s=32, alpha=0.95, edgecolors="none", zorder=2)
    # Degree marks are anchored just outside their ticks, facing away from the centre; the axes
    # limits are solved in inches so the circle is as large as the cell allows ("±180°" may extend
    # up to 0.4 in into the empty left margin).
    pad = 0.08
    w180, h180 = _measure(fig, "±180°", F["deg"])
    w0, _ = _measure(fig, "0°", F["deg"])
    _, h90 = _measure(fig, "+90°", F["deg"])
    _, hth = _measure(fig, r"$\theta$", F["theta"])
    spill = min(0.40, gskw["left"] * figsize[0] - 0.12)
    r_in = (pw[0] - 2 * pad - w180 - w0 + spill) / (2 * 1.12)
    c_in = 1.12 * r_in + pad + max(h90, h180) + 0.05
    if 2 * c_in > AXES_HEIGHT:
        r_in = (AXES_HEIGHT / 2 - pad - max(h90, h180) - 0.05) / 1.12
        c_in = AXES_HEIGHT / 2
    A = (1.12 * r_in + pad + w180 - spill) / r_in
    B = (1.12 * r_in + pad + w0) / r_in
    C = c_in / r_in
    rp = 1.12 + pad / r_in
    for adeg, lab, ha, va in ((0, "0°", "left", "center"), (90, "+90°", "center", "bottom"),
                              (180, "±180°", "right", "center"), (-90, "-90°", "center", "top")):
        arad = np.deg2rad(adeg)
        ax.plot([1.03 * r * np.cos(arad), 1.09 * r * np.cos(arad)],
                [1.03 * r * np.sin(arad), 1.09 * r * np.sin(arad)],
                color=ps.INK, lw=1.0, zorder=3)
        ax.text(rp * r * np.cos(arad), rp * r * np.sin(arad), lab, ha=ha, va=va,
                color=ps.INK, fontsize=F["deg"])
    a0, a1 = np.deg2rad(14), np.deg2rad(50)          # positive direction of theta
    ra = 1.10 * r
    ax.annotate("", xy=(ra * np.cos(a1), ra * np.sin(a1)),
                xytext=(ra * np.cos(a0), ra * np.sin(a0)),
                arrowprops=dict(arrowstyle="-|>", color=GRAPHITE, lw=1.4,
                                connectionstyle="arc3,rad=0.16", shrinkA=0, shrinkB=0))
    rth = 1.12 + (pad + 0.5 * hth) / r_in
    ax.text(rth * r * np.cos(np.deg2rad(32)), rth * r * np.sin(np.deg2rad(32)),
            r"$\theta$", color=GRAPHITE, ha="center", va="center", fontsize=F["theta"])
    ax.set_xlim(-A * r, B * r)
    ax.set_ylim(-C * r, C * r)
    circle_cx_frac = A / (A + B)
    ax.set_aspect("equal")
    ax.set_axis_off()
    circle_ax = ax

    # ---- c: one-shot edits -------------------------------------------------------------------- #
    ax = fig.add_subplot(gs[1])
    resp: dict = {}
    for th, v in zip(panel["c_theta"], panel["c_resp"]):
        resp.setdefault(float(th), []).append(v)
    ax.axhline(0.0, color=ps.META_LINE, ls="--", lw=1.0, zorder=0)
    ax.plot([-21, 21], [-21, 21], color=ps.META_LINE, ls="--", lw=1.2, zorder=0)
    for th, vals in sorted(resp.items()):
        x = th + (np.arange(len(vals)) - (len(vals) - 1) / 2) * 1.1
        ax.scatter(x, vals, s=26, color=GRAPHITE, alpha=0.75, edgecolors="none", zorder=2)
        m = float(np.mean(vals))
        ax.plot([th - 5, th + 5], [m, m], color=ps.INK, lw=2.2, zorder=3)
    ax.set_xlim(-52, 52)
    ax.set_ylim(-22, 24)
    ax.set_xticks([-40, -20, 20, 40])
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_xlabel("rotation written\ninto the belief (°)" if wrap else "rotation written into the belief (°)",
                  linespacing=1.3)
    ax.set_ylabel("decoded wrist\nrotation (°)" if wrap else "decoded wrist rotation (°)",
                  linespacing=1.3)
    ps.finish(ax)

    # ---- d: held edits ------------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[2])
    steps = np.arange(32)
    for st in (8, 16, 24):                       # a new action chunk every 8 steps
        ax.axvline(st - 0.5, color=ps.SLATE[1], lw=0.8, zorder=0)
    ax.axhline(0.0, color=ps.META_LINE, ls="--", lw=1.2, zorder=0)
    by_theta = {float(th): row for th, row in zip(panel["d_theta"], panel["d_rot"])}
    for th, tt in zip([float(t) for t in panel["d_theta"] if t != 0.0], LINE_T):
        color = cmap(tt)
        y = np.asarray(by_theta[th])
        ax.plot(steps, y, color=color, lw=1.8)
        ax.annotate(f"written in: {th:+.0f}°", (steps[-1], y[-1]), xytext=(4, 0),
                    textcoords="offset points", color=color, va="center", fontsize=F["ann"])
    yref = np.asarray(by_theta[0.0])
    ax.plot(steps, yref, color=ps.META_LINE, lw=1.4)
    ax.annotate("written in: 0°", (steps[-1], yref[-1]), xytext=(4, 0),
                textcoords="offset points", color=ps.META_LINE, va="center", fontsize=F["ann"])
    ax.set_xlim(0, 31)
    ax.set_ylim(-70, 70)
    ax.set_xticks([0, 8, 16, 24, 31])
    ax.set_xlabel("rollout step")
    ax.set_ylabel("executed wrist\nrotation (°)" if wrap else "executed wrist rotation (°)",
                  linespacing=1.3)
    ps.finish(ax)

    if CIRCLE_LABEL:
        # centred under the circle, its top on the neighbours' x-label top line
        circle_ax.apply_aspect()
        bb = circle_ax.get_position()
        cx = bb.x0 + circle_cx_frac * bb.width
        cy = (diag["xlab_top_in"] - CIRCLE_LABEL_DY) / figsize[1]
        fig.text(cx, cy, CIRCLE_LABEL, ha="center", va="top", color=ps.INK,
                 fontsize=F["lab"], linespacing=1.3)

    fig.savefig(str(out_pdf), transparent=False)
    if out_png is not None:
        fig.savefig(str(out_png), dpi=350, bbox_inches=None, transparent=False)
    plt.close(fig)
    return dict(figsize=[round(x, 3) for x in figsize], layout=diag, n_natural=int(len(ph)),
                circle_radius=r, circle=dict(r_in=round(r_in, 3), xlim_r=[round(-A, 3), round(B, 3)],
                                             ylim_r=round(C, 3), spill_in=round(spill, 3)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--panel", choices=["a", "bcd", "all"], default="all")
    ap.add_argument("--panels", type=Path, default=None,
                    help="directory with fig6_a.npz / fig6_bcd.npz (default $VLAGEOM_DATA/panels)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default $VLAGEOM_FIGURES)")
    ap.add_argument("--png-a", action="store_true", help="also write fig6_a.png (350 dpi)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[fig6] %(message)s")
    pdir = args.panels or paths.panels_dir()
    out = args.out_dir or paths.figures_dir(create=True)
    out.mkdir(parents=True, exist_ok=True)
    if args.panel in ("a", "all"):
        with np.load(pdir / "fig6_a.npz", allow_pickle=False) as z:
            pa = {k: z[k] for k in z.files}
        render_a(pa, out / "fig6_a.pdf", out / "fig6_a.png" if args.png_a else None)
        log.info("wrote %s", out / "fig6_a.pdf")
    if args.panel in ("bcd", "all"):
        with np.load(pdir / "fig6_bcd.npz", allow_pickle=False) as z:
            pb = {k: z[k] for k in z.files}
        rec = render_bcd(pb, out / "fig6_bcd.pdf", out / "fig6_bcd.png")
        log.info("wrote %s and %s (%s in)", out / "fig6_bcd.pdf", out / "fig6_bcd.png", rec["figsize"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
