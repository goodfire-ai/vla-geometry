#!/usr/bin/env python
"""Render the four data panels of Figure 3 (distance to the target object), one file pair per panel.

The paper's Figure 3 places these panels next to a hand-drawn robot sketch, with the title and the
labels a-d, in a layout made by hand. This script writes each data panel as its own figure:

  fig3a  held-out R^2 of the near probe (< 15 cm) and the far probe (15-70 cm) against the number
         k of erased directions; solid lines with filled circles: far-probe directions erased,
         dashed lines with open squares: near-probe directions erased; symlog x axis
         (7.2 x 4.4 in)
  fig3b  R^2 on the rows below c (x) against R^2 on the rows at or above c (y), one point per
         boundary c, for the range-split pair, the round-to-c pair and the full-range probe
         recalibrated per range; the range-split points are labelled with c (6.8 x 6.2 in)
  fig3c  cosine between the near and far probe weights against c, for the range-split and the
         round-to-c pairs: measured (band = std over folds), split-half ceiling (dashed) and
         synthetic activations with one distance direction (dotted) (7.2 x 6.2 in)
  fig3d  centimetre-probe read-out against metre-probe read-out (out of house), coloured by the
         true distance in metres (left, 0.15-0.70 m) and in centimetres (right, 0-15 cm), with
         the per-bin mean path and dashed lines at 15 cm (11.6 x 5.0 in)

Inputs (under `vlageom.paths.panels_dir()`, default $VLAGEOM_DATA/panels), written by
pipelines/distance/90_fig3_panels.py:

  fig3a_erasure.npz  k, near_erase_far, far_erase_far, near_erase_near, far_erase_near
  fig3b_cutoff.npz   grid_cm, split_near, split_far, clip_near, clip_far, n1_near, n1_far
  fig3c_cosine.npz   grid_cm, split_cos, split_cos_std, split_ceil, split_plant,
                     round_cos, round_cos_std, round_ceil, round_plant
  fig3d_plane.npz    d_cm, x_m, y_cm, path_x_m, path_y_cm, cut_cm

Output: <out-dir>/fig3<panel>.pdf and .png (default `vlageom.paths.figures_dir()`), 350 dpi, tight
bounding box, white background; each file is drawn from a new figure (sizes above are before the
tight crop).

Style: `vlageom.plot.style` (Product Sans from $VLAGEOM_FONT_DIR, Computer Modern math). Text
sizes: axis titles 26 pt, tick labels 22 pt, legends and the c labels 20 pt. Colours are fixed hex
values: near probe / range-split #2563eb (blue), far probe / full-range #8b5cf6 (violet),
round-to-c #64748b (slate); panel d uses the managua colormap.

Run:  python figures/fig3_distance.py [--panel a b c d] [--panels DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

from vlageom import distance as D  # noqa: E402
from vlageom import paths  # noqa: E402
from vlageom.plot import style as ps  # noqa: E402

PANELS = ("a", "b", "c", "d")
PANEL_FILES = {"a": "fig3a_erasure.npz", "b": "fig3b_cutoff.npz", "c": "fig3c_cosine.npz",
               "d": "fig3d_plane.npz"}
DPI = 350
FS = dict(tick=22, label=26, legend=20, annot=20)

BLUE = ps.PRIMARY     # #2563eb: near probe (a), range-split pair (b, c)
VIOLET = ps.VIOLET    # #8b5cf6: far probe (a), full-range probe (b)
SLATE = ps.SLATE[3]   # #64748b: round-to-c pair (b, c)

# panel a
A_FIGSIZE = (7.2, 4.4)
A_XTICKS = [0, 1, 2, 4, 8, 16, 32]          # labelled; the other scored k get unlabelled ticks
A_XLIM = (-0.15, 40)
A_YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8]
A_LW, A_MS = 2.2, 7.0
# panel b
B_FIGSIZE = (6.8, 6.2)
B_LIM = (-0.05, 0.9)
B_LABEL_ABOVE_MAX_C = 15                    # c <= 15: label above the point, else to its right
B_LABEL_OFFSETS = {20: ((11, 9), "left", "center")}   # (offset in pt, ha, va) for single labels
# panel c
C_FIGSIZE = (7.2, 6.2)
C_XTICKS = [10, 20, 30, 40, 50]
C_YLIM = (-0.12, 1.02)
C_YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# panel d
D_FIGSIZE = (11.6, 5.0)
D_XTICKS = [0.0, 0.2, 0.4, 0.6]
D_YTICKS = [0, 5, 10, 15]
D_CMAP = "managua"
D_METRE_NORM = (D.CUT_M, D.DIST_MAX)        # colour range of the left panel (m): the far range
D_METRE_TICKS = [0.15, 0.30, 0.45, 0.60]
D_CM_TICKS = [0, 5, 10, 15, 20, 25, 30]     # ticks of the right colour bar up to its maximum (cm)


def _save(fig, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"wrote {out_dir / stem}.pdf, .png")


def render_a(p: dict, out_dir: Path) -> None:
    """Panel a: erasure curves (see the module docstring)."""
    ps.apply_style()
    ks = [int(k) for k in p["k"]]
    series = ((BLUE, "-", "o", p["near_erase_far"]), (VIOLET, "-", "o", p["far_erase_far"]),
              (BLUE, "--", "s", p["near_erase_near"]), (VIOLET, "--", "s", p["far_erase_near"]))
    fig, ax = plt.subplots(figsize=A_FIGSIZE)
    ax.axhline(0, color=ps.META_LINE, ls="--", lw=1.2, zorder=0)
    for col, ls, mk, vals in series:
        ax.plot(ks, np.asarray(vals, np.float64).tolist(), color=col, ls=ls, marker=mk, lw=A_LW,
                ms=A_MS, markerfacecolor=col if ls == "-" else "white", markeredgecolor=col,
                markeredgewidth=1.4)
    ax.set_xscale("symlog", base=2, linthresh=1, linscale=0.6)
    ax.set_xlabel("directions erased ($k$)", fontsize=FS["label"])
    ax.set_ylabel("held-out $R^2$", fontsize=FS["label"])
    ps.finish(ax)
    ax.xaxis.set_major_locator(FixedLocator(A_XTICKS))
    ax.set_xticklabels([str(t) for t in A_XTICKS])
    ax.xaxis.set_minor_locator(FixedLocator([k for k in ks if k not in A_XTICKS]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(*A_XLIM)
    ax.tick_params(axis="both", which="major", labelsize=FS["tick"])
    ax.set_yticks(A_YTICKS)
    cut_cm, max_cm = round(100 * D.CUT_M), round(100 * D.DIST_MAX)
    handles = [Line2D([], [], color=BLUE, lw=A_LW, label=f"near decode (<{cut_cm} cm)"),
               Line2D([], [], color=VIOLET, lw=A_LW, label=f"far decode ({cut_cm}-{max_cm} cm)"),
               Line2D([], [], color=ps.INK, lw=A_LW, ls="-", marker="o", ms=A_MS,
                      label="far axis erased"),
               Line2D([], [], color=ps.INK, lw=A_LW, ls="--", marker="s", ms=A_MS,
                      markerfacecolor="white", markeredgewidth=1.4, label="near axis erased")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
               fontsize=FS["legend"], frameon=False, columnspacing=1.2, handlelength=1.6,
               handletextpad=0.6)
    _save(fig, out_dir, "fig3a")


def render_b(p: dict, out_dir: Path) -> None:
    """Panel b: near R^2 against far R^2 across the boundaries c (see the module docstring)."""
    ps.apply_style()
    grid = [int(c) for c in p["grid_cm"]]
    fig, ax = plt.subplots(figsize=B_FIGSIZE)
    for fam, color, lw, ms, label in (("clip", SLATE, 1.2, 4, "round to c"),
                                      ("n1", VIOLET, 1.2, 4, "full-range"),
                                      ("split", BLUE, 1.9, 6, "range-split")):
        xs = np.asarray(p[f"{fam}_near"], float)
        ys = np.asarray(p[f"{fam}_far"], float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        ax.plot(xs[ok], ys[ok], color=color, lw=lw, marker="o", ms=ms, label=label,
                zorder=3 if fam == "split" else 2)
        if fam != "split":
            continue
        for c, x, y in zip(grid, xs, ys):          # label the points inside the axes window
            if not (np.isfinite(x) and np.isfinite(y) and B_LIM[0] <= y <= B_LIM[1]):
                continue
            if c <= B_LABEL_ABOVE_MAX_C:
                off, ha, va = (0, 9), "center", "bottom"
            else:
                off, ha, va = (11, 0), "left", "center"
            off, ha, va = B_LABEL_OFFSETS.get(c, (off, ha, va))
            ax.annotate(f"{c}", (x, y), textcoords="offset points", xytext=off, ha=ha, va=va,
                        fontsize=FS["annot"], color=ps.INK)
    ax.set_xlim(*B_LIM)
    ax.set_ylim(*B_LIM)
    ax.set_xlabel("$R^2$ near (< c)", fontsize=FS["label"])
    ax.set_ylabel("$R^2$ far ($\\geq$ c)", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=FS["legend"],
              loc="lower left", bbox_to_anchor=(0.10, 0.10))
    ps.finish(ax)
    _save(fig, out_dir, "fig3b")


def render_c(p: dict, out_dir: Path) -> None:
    """Panel c: near/far weight cosines across the boundaries c (see the module docstring)."""
    ps.apply_style()
    grid = [int(c) for c in p["grid_cm"]]
    fig, ax = plt.subplots(figsize=C_FIGSIZE)
    ax.axhline(0, color=ps.META_LINE, ls="--", lw=1.4, zorder=0)
    ax.text(grid[-1], -0.02, "perpendicular", fontsize=FS["annot"], color=SLATE, ha="right",
            va="top")
    for fam, color in (("split", BLUE), ("round", SLATE)):
        mean_ = np.asarray(p[f"{fam}_cos"], np.float64)
        std_ = np.asarray(p[f"{fam}_cos_std"], np.float64)
        ax.fill_between(grid, mean_ - std_, mean_ + std_, color=color, alpha=0.16,
                        edgecolor="none")
        ax.plot(grid, mean_, color=color, marker="o", ms=8, lw=2.6, zorder=3)
        ax.plot(grid, np.asarray(p[f"{fam}_ceil"], np.float64), color=color, ls="--", marker="o",
                ms=5.5, lw=1.8, alpha=0.75)
        ax.plot(grid, np.asarray(p[f"{fam}_plant"], np.float64), color=color, ls=":", marker="o",
                ms=6.5, lw=2.4, markerfacecolor="white", markeredgewidth=1.6, zorder=2)
    ax.set_xlabel("boundary c (cm)", fontsize=FS["label"])
    ax.set_ylabel("cosine (near, far)", fontsize=FS["label"])
    ax.tick_params(axis="both", which="major", labelsize=FS["tick"])
    ax.set_xticks(C_XTICKS)
    ax.set_ylim(*C_YLIM)
    ax.set_yticks(C_YTICKS)
    ps.finish(ax)
    handles = [Line2D([], [], color=BLUE, lw=2.6, label="range-split"),
               Line2D([], [], color=SLATE, lw=2.6, label="round to c"),
               Line2D([], [], color="none"),
               Line2D([], [], color=ps.INK, lw=2.6, marker="o", ms=8, label="measured"),
               Line2D([], [], color=ps.INK, lw=1.8, ls="--", marker="o", ms=5.5,
                      label="same-axis ceiling"),
               Line2D([], [], color=ps.INK, lw=2.4, ls=":", marker="o", ms=6.5,
                      markerfacecolor="white", markeredgewidth=1.6,
                      label="synthetic, one direction")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=FS["legend"], frameon=False, columnspacing=1.0, handlelength=1.8,
              handletextpad=0.6, borderaxespad=0.2)
    _save(fig, out_dir, "fig3c")


def render_d(p: dict, out_dir: Path) -> None:
    """Panel d: the two rulers' read-outs against each other (see the module docstring)."""
    ps.apply_style()
    cut = float(p["cut_cm"])
    d_cm = np.asarray(p["d_cm"], np.float64)
    x_m = np.asarray(p["x_m"], np.float64)
    y_cm = np.asarray(p["y_cm"], np.float64)
    # (norm, colour-bar ticks, extend, scale from cm to the colour unit, colour-bar title)
    metre = (mcolors.Normalize(*D_METRE_NORM), D_METRE_TICKS, "min", 0.01, "distance (m)")
    centi = (mcolors.Normalize(0, cut), D_CM_TICKS, "max", 1.0, "distance (cm)")
    fig, axes = plt.subplots(1, 2, figsize=D_FIGSIZE, sharey=True)
    for ax, (norm, ticks, ext, scale, clabel), title in zip(axes, (metre, centi),
                                                            ("Meter ruler", "CM ruler")):
        pts = ax.scatter(x_m, y_cm, c=d_cm * scale, cmap=D_CMAP, norm=norm, s=8, alpha=0.85,
                         edgecolors="none")
        ax.plot(np.asarray(p["path_x_m"], np.float64), np.asarray(p["path_y_cm"], np.float64),
                color=ps.INK, lw=1.4, marker="o", ms=3.5, zorder=3)
        ax.axvline(cut / 100.0, color=ps.META_LINE, ls="--", lw=1.0, zorder=0)
        ax.axhline(cut, color=ps.META_LINE, ls="--", lw=1.0, zorder=0)
        ax.set_xlabel("m-probe readout (m)")
        ax.set_title(title)
        ax.set_xticks(D_XTICKS)
        ax.set_yticks(D_YTICKS)
        cb = fig.colorbar(pts, ax=ax, pad=0.02, extend=ext,
                          ticks=[t for t in ticks if t <= (norm.vmax + 1e-9)])
        cb.outline.set_visible(False)
        ps.finish(ax)
        ax.tick_params(labelsize=FS["tick"])
        ax.xaxis.label.set_size(FS["label"])
        ax.yaxis.label.set_size(FS["label"])
        ax.title.set_size(FS["label"])
        cb.ax.tick_params(labelsize=FS["tick"])
        cb.set_label(clabel, fontsize=FS["label"])
    axes[0].set_ylabel("cm-probe readout (cm)", fontsize=FS["label"])
    fig.tight_layout()              # two layout passes: the second one settles the colour bars
    _save(fig, out_dir, "fig3d")


RENDER = {"a": render_a, "b": render_b, "c": render_c, "d": render_d}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--panel", nargs="+", choices=PANELS, default=list(PANELS),
                    help="panels to render (default: all)")
    ap.add_argument("--panels", type=Path, default=None,
                    help="directory with the fig3*.npz panel files (default $VLAGEOM_DATA/panels)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default $VLAGEOM_FIGURES)")
    a = ap.parse_args(argv)
    pdir = a.panels or paths.panels_dir()
    out = a.out_dir or paths.figures_dir(create=True)
    out.mkdir(parents=True, exist_ok=True)
    for panel in a.panel:
        with np.load(pdir / PANEL_FILES[panel], allow_pickle=False) as z:
            p = {k: z[k] for k in z.files}
        RENDER[panel](p, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
