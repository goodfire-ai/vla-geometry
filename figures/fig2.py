#!/usr/bin/env python
"""Render the data panels of Figure 2 (EE and target object position), one file pair per panel.

The paper's Figure 2 is assembled by hand from three rendered files; the panel letters a-e, the
row titles ("EE Position", "Target Object Position") and the "World" / "Model" labels are added
in that layout. This script writes the three files:

  fig2_top        the top row, panels a-e, with the shared legend above them (40 x 13 in before
                  the tight bounding box; PDF 350 dpi, PNG 200 dpi):
                  a  held-out error per axis and 3-D (cm), EE and target as paired bars with fold
                     std error bars and the value of each bar; dashed line: gripper width (8.7 cm)
                  b  held-out R^2 at read-out rank 1, 2, 3
                  c  singular-value fractions of the read-out
                  d  relative residual of the world -> activation displacement map vs layer
                     (solid: general linear, dashed: orthogonal Procrustes; reference line at 1.0:
                     no linear relation)
                  e  median turning angle of straight paths in the subspace vs distance to the
                     gripper (solid), and over random 3-D projections (dashed)
  fig2_ee_grid    the EE trace grid, 2 x 5, text-free: top = the hand's realized paths in the
                  base frame (figure-8, x line, y line, circle, square), bottom = the same
                  waypoints' activations (hdr_txt, layer 20) in the rank-3 read-out subspace fit
                  without the house of the paths; camera elevation 22 deg, azimuth -58 deg
  fig2_target_grid
                  the target trace grid, 2 x 5, text-free: the object's paths on its table
                  (figure-8, x line, y line, ellipse, square) and their activations (txt_instr,
                  layer 21); camera elevation 40 deg, azimuth -58 deg

Inputs (under `vlageom.paths.panels_dir()`, default $VLAGEOM_DATA/panels):
  fig2_top.npz          pipelines/position/70_top_panel.py (keys: vlageom.position.top_panel)
  fig2_ee_grid.npz, fig2_target_grid.npz
                        pipelines/position/90_grid_panels.py (keys: vlageom.position.grid_panel)

Grid rendering: each panel is drawn as its own matplotlib 3-D figure (8 x 8 in, 170 dpi) on one
shared box and floor (every panel at the same scale), rasterized, cropped to the union of the
drawn pixels of all ten panels, and the ten images are placed on one figure (12.75 in wide, the
two rows overlapping by 3.5% of the height). Points are coloured by position: x -> red,
y -> green, z -> blue. Paths are cubic smoothing splines through the waypoints (s = 0.0008 in the
world panels, 0.02 in the subspace panels); a grey shadow of every path is drawn on the floor.
PDF at 200 dpi, PNG at 130 dpi.

Fonts (top row): Product Sans from $VLAGEOM_FONT_DIR (else DejaVu Sans; see vlageom.plot.style).
Sizes: axis titles 48 pt, ticks 42 pt, legend 35 pt, bar values and in-plot labels 32-33 pt.
The bar-value labels are placed from the measured axes size: above their bar, stacked when the
two of a pair would overlap, kept clear of the gripper-width line.

Run:  python figures/fig2.py [--panel top ee_grid target_grid] [--panels DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from vlageom import paths  # noqa: E402
from vlageom.plot import style as ps  # noqa: E402

PANELS = ("top", "ee_grid", "target_grid")

# ---- top row ---------------------------------------------------------------------------------- #
FS_LABEL = 48          # axis titles
FS_TICK = 42           # tick labels
FS_LEGEND = 35         # legend entries (in-plot labels: FS_LEGEND - 2)
FS_BARVAL_PAIR = 32    # bar-value labels
FIG_W = 40.0           # inches, five panels
FIG_H = 13.0
LEGEND_GAP = 0.3       # inches between the legend and the panel tops
BLUE, VIOLET = ps.PRIMARY, ps.VIOLET
COL = {"ee": BLUE, "obj": VIOLET}
NAME = {"ee": "EE position", "obj": "target object position"}
BAR_W = 0.38


def pair_labels(ax, bars, tops, fmt, fontsize, ax_w_in, ax_h_in, avoid=None):
    """Value labels of paired bars, coloured by variable. Each label sits just above its bar's top
    (`tops`, the error-bar cap in panel a). When the two labels of a pair would be closer than one
    text height, the taller bar's label is moved up so they stack. A label wider than its bar whose
    neighbour bar rises into its line is aligned away from that bar (right-aligned on the left
    bar's right edge, left-aligned on the right bar's left edge) when it then stays inside the
    axes and clear of the other bars; otherwise both labels are stacked above the taller bar.
    Labels that would cross the line at `avoid` are moved just above it. Call after the layout is
    final, with the axes size in inches."""
    ylo, yhi = ax.get_ylim()
    xlo, xhi = ax.get_xlim()
    sep = (fontsize * 1.25 / 72.0) * (yhi - ylo) / ax_h_in
    unit_per_in = (xhi - xlo) / ax_w_in
    GLYPH = {".": 0.30, "-": 0.36}

    def text_w(t):
        return sum(GLYPH.get(ch, 0.56) for ch in t) * fontsize / 72.0 * unit_per_in
    dy = 0.10 * sep
    geo = {k: [(b.get_x(), b.get_x() + b.get_width()) for b in bars[k]] for k in ("ee", "obj")}
    n = len(bars["ee"])
    for i in range(n):
        y = {k: tops[k][i] + dy for k in ("ee", "obj")}
        txt = {k: fmt.format(bars[k][i].get_height()) for k in ("ee", "obj")}
        xa = {k: ((geo[k][i][0] + geo[k][i][1]) / 2, "center") for k in ("ee", "obj")}
        lo, hi = sorted(y, key=y.get)
        if y[hi] - y[lo] < sep:
            y[hi] = y[lo] + sep
        wide = text_w(txt[lo]) > (geo[lo][i][1] - geo[lo][i][0])
        if wide and tops[hi][i] > y[lo] - 0.2 * sep:
            gap = 0.05 * unit_per_in
            if lo == "ee":
                xanc, ha = geo["ee"][i][1] - gap, "right"
                span = (xanc - text_w(txt[lo]), xanc)
            else:
                xanc, ha = geo["obj"][i][0] + gap, "left"
                span = (xanc, xanc + text_w(txt[lo]))
            pad = 0.02 * (xhi - xlo)
            overflow = 0.35 * unit_per_in
            inside = xlo + pad <= span[0] and span[1] <= xhi + overflow
            clash = any(x0 < span[1] and x1 > span[0] and tops[k2][j] > y[lo] - 0.2 * sep
                        for k2 in ("ee", "obj") for j, (x0, x1) in enumerate(geo[k2]) if j != i)
            if inside and not clash:
                xa[lo] = (xanc, ha)
            else:
                y[lo] = tops[hi][i] + 0.5 * dy
                y[hi] = y[lo] + sep
        if avoid is not None:
            for k in (lo, hi):
                if y[k] < avoid and y[k] + 1.0 * sep > avoid:
                    y[k] = avoid + 0.3 * sep
            if y[hi] - y[lo] < sep:
                y[hi] = y[lo] + sep
        for k in ("ee", "obj"):
            ax.annotate(txt[k], (xa[k][0], y[k]), ha=xa[k][1], va="bottom", fontsize=fontsize,
                        color=COL[k])


def render_top(p: dict, out_pdf: Path, out_png: Path) -> list:
    """The top row from the arrays of fig2_top.npz; writes the PDF, then the PNG."""
    ps.apply_style()
    plt.rcParams.update({"font.size": FS_TICK, "axes.labelsize": FS_LABEL,
                         "axes.titlesize": FS_LABEL, "xtick.labelsize": FS_TICK,
                         "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND})
    gripper_cm = float(p["gripper_cm"])
    vi = {"ee": 0, "obj": 1}
    labels = {k: f"{NAME[k]} (VLM layer {int(p['layers'][vi[k]])})" for k in ("ee", "obj")}
    fig, ax = plt.subplots(1, 5, figsize=(FIG_W, FIG_H), facecolor="white")
    offs = {"ee": -BAR_W / 2, "obj": BAR_W / 2}

    # a: error per quantity
    a = ax[0]
    qs = ["x", "y", "z", "3-D"]
    xs = np.arange(len(qs))
    top = gripper_cm
    bars1, tops1 = {}, {}
    for k in ("ee", "obj"):
        m, e = p["mae_cm"][vi[k]], p["mae_cm_fold_std"][vi[k]]
        bars1[k] = a.bar(xs + offs[k], m, width=BAR_W, color=COL[k], edgecolor="none", yerr=e,
                         capsize=3, error_kw={"lw": 1.2, "ecolor": ps.INK}, zorder=3,
                         label=labels[k])
        tops1[k] = list(m + e)
        top = max(top, float((m + e).max()))
    a.axhline(gripper_cm, color=ps.META_LINE, ls="--", lw=1.3, zorder=2, label="gripper width")
    a.annotate("gripper width", (a.get_xlim()[0], gripper_cm), xytext=(6, 4),
               textcoords="offset points", ha="left", va="bottom", fontsize=FS_LEGEND - 2,
               color=ps.SLATE[3], zorder=4)
    a.set_xticks(list(xs))
    a.set_xticklabels(qs)
    a.set_ylabel("MAE (cm)")
    sep_est = (FS_BARVAL_PAIR * 1.25 / 72.0) / (FIG_H * 0.62)    # text height / axes height
    ylim_top = max(top * 1.15, gripper_cm * 1.25, top / (1 - 2.6 * sep_est))
    a.set_ylim(0, ylim_top)
    a.set_yticks([t for t in range(0, 41, 2) if t <= ylim_top])
    deferred = [(a, bars1, tops1, "{:.1f}", gripper_cm)]
    handles, hlabels = a.get_legend_handles_labels()

    # b: R^2 at read-out rank 1-3
    a = ax[1]
    bars2, tops2 = {}, {}
    for k in ("ee", "obj"):
        rrr = list(p["rrr"][vi[k]])
        bars2[k] = a.bar(np.array([1, 2, 3]) + offs[k], rrr, width=BAR_W, color=COL[k],
                         edgecolor="none", zorder=3)
        tops2[k] = rrr
    a.set_xlabel("readout\nrank  $r$")
    a.set_ylabel("held-out $R^2$")
    a.set_xticks([1, 2, 3])
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a.set_ylim(0, max(1.0, max(max(v) for v in tops2.values()) / (1 - 2.0 * sep_est)))
    deferred.append((a, bars2, tops2, "{:.2f}", None))

    # c: singular-value fractions
    a = ax[2]
    svmax = 0.0
    bars3, tops3 = {}, {}
    for k in ("ee", "obj"):
        svf = list(p["sv_frac"][vi[k]])
        svmax = max(svmax, max(svf))
        bars3[k] = a.bar(np.array([1, 2, 3]) + offs[k], svf, width=BAR_W, color=COL[k],
                         edgecolor="none", zorder=3)
        tops3[k] = svf
    a.set_xlabel("singular-value\nrank")
    a.set_ylabel("singular-value\nfraction")
    a.set_xticks([1, 2, 3])
    a.set_ylim(0, max(0.75, svmax + 0.12, svmax / (1 - 2.6 * sep_est)))
    a.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    deferred.append((a, bars3, tops3, "{:.2f}", None))

    # d: residual vs layer
    a = ax[3]
    a.axhline(1.0, color=ps.META_LINE, ls="--", lw=1.2, zorder=1)
    Ls = [int(L) for L in p["proc_layers"]]
    for k in ("ee", "obj"):
        a.plot(Ls, list(p["proc_resid"][vi[k]]), "o-", color=COL[k], lw=1.8, ms=4, zorder=3)
        a.plot(Ls, list(p["proc_ortho"][vi[k]]), "s--", color=COL[k], lw=1.5, ms=3.5, zorder=3)
    a.set_xlabel("layer")
    a.set_ylabel("relative residual")
    a.set_ylim(0, 1.12)
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a.set_xticks([0, 10, 20, 30])
    a.legend([mlines.Line2D([], [], color=ps.INK, ls="-", marker="o", ms=4),
              mlines.Line2D([], [], color=ps.INK, ls="--", marker="s", ms=3.5)],
             ["general linear", "orthogonal Procrustes"], loc="lower left",
             bbox_to_anchor=(0.0, 0.0),
             frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.4, handletextpad=0.4,
             borderpad=0.3, labelspacing=0.3, fontsize=FS_LEGEND - 2)
    a.text(35.5, 1.0, "no linear relation", ha="right", va="bottom", fontsize=FS_LEGEND - 2,
           color=ps.SLATE[3])

    # e: curvature vs distance to the gripper
    a = ax[4]
    first = True
    for k in ("ee", "obj"):
        cb = p[f"curv_{k}"]
        xc = [(row[0] + row[1]) / 2 * 100 for row in cb]
        a.plot(xc, list(cb[:, 2]), "o-", color=COL[k], lw=2, ms=6, zorder=3)
        a.plot(xc, list(cb[:, 3]), "--", color=COL[k], lw=1.3, alpha=0.45, zorder=2,
               label="random 3-D projection" if first else None)
        first = False
    a.set_xlabel("distance to\ngripper (cm)")
    a.set_ylabel("path curvature (deg)")
    a.set_xticks([20, 30, 40, 50, 60])
    a.set_ylim(0, 100)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.text(a.get_xlim()[1], 99, "random 3-D projection", ha="right", va="top",
           fontsize=FS_LEGEND - 2, color=ps.SLATE[3])

    for a in ax:
        ps.finish(a)
    order = [hlabels.index(labels["ee"]), hlabels.index(labels["obj"])]
    leg = fig.legend([handles[i] for i in order], [hlabels[i] for i in order], loc="upper center",
                     ncol=len(order), frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
                     handletextpad=0.5, columnspacing=1.8)
    fig.canvas.draw()
    leg_h = leg.get_window_extent().height / fig.dpi
    rect_top = 1.0 - (leg_h + LEGEND_GAP) / FIG_H
    fig.tight_layout(rect=(0, 0, 1, rect_top))
    for a_, bars_, tops_, fmt_, avoid_ in deferred:
        bb = a_.get_position()
        pair_labels(a_, bars_, tops_, fmt_, FS_BARVAL_PAIR, bb.width * FIG_W, bb.height * FIG_H,
                    avoid=avoid_)
    fig.savefig(out_pdf, dpi=350, bbox_inches="tight", transparent=False, facecolor="white")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return [out_pdf, out_png]


# ---- trace grids ------------------------------------------------------------------------------ #
GRID_VIEW = {"ee_grid": (22.0, -58.0), "target_grid": (40.0, -58.0)}   # (elevation, azimuth)
SMOOTH_WORLD = 0.0008
SMOOTH_SUB = 0.02
LINE_W = 3.0
POINT_SIZE = 44.0
EDGE = "0.2"
FLOOR_COLOR = "#F8FAFC"
SHADOW_COLOR = "0.55"
PAD = 0.06
GRID_COL_W = 2.55       # inches per column
ROW_GAP = -0.035        # rows overlap by 3.5% of the figure height
GRID_PDF_DPI = 200
GRID_PNG_DPI = 130


def draw_path(ax, P, rgb, closed, smooth, lw, size, edge, floor_color, shadow_color, flim, fbox,
              floor, floor_alpha=0.9, shadow_alpha=0.5):
    """A square floor, the path's shadow on it, then the coloured path (smoothing spline, segment
    colours interpolated from the waypoints) and its waypoints, on fixed axis limits."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    from vlageom.target import smooth_curve

    try:
        ax.computed_zorder = False
    except Exception:
        pass
    curve, u, uu = smooth_curve(P, closed, smooth)
    cc = np.clip(np.stack([np.interp(uu, u, rgb[:, c]) for c in range(3)], 1), 0, 1)
    rng = P.max(0) - P.min(0)
    hi = P.max(0) + rng * 0.06
    fcx, fcy, fhalf, zf = floor
    x0, x1, y0, y1 = fcx - fhalf, fcx + fhalf, fcy - fhalf, fcy + fhalf
    ax.add_collection3d(Poly3DCollection([[[x0, y0, zf], [x1, y0, zf], [x1, y1, zf], [x0, y1, zf]]],
                                         facecolor=floor_color, alpha=floor_alpha, edgecolor="none",
                                         zorder=0))
    sz = zf + (hi[2] - zf) * 0.004
    ax.plot(curve[:, 0], curve[:, 1], np.full(len(curve), sz), color=shadow_color, lw=lw * 0.8,
            alpha=shadow_alpha, zorder=1)
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), sz), c=shadow_color, s=size * 0.55,
               alpha=shadow_alpha, depthshade=False, edgecolors="none", zorder=1)
    segs = np.stack([curve[:-1], curve[1:]], 1)
    ax.add_collection3d(Line3DCollection(segs, colors=(cc[:-1] + cc[1:]) / 2, linewidths=lw,
                                         zorder=5))
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=rgb, s=size, depthshade=True, edgecolors=edge,
               linewidths=0.4, zorder=6)
    ax.set_xlim(flim[0], flim[1])
    ax.set_ylim(flim[2], flim[3])
    ax.set_zlim(flim[4], flim[5])
    ax.set_box_aspect(fbox, zoom=1.0)


def render_cell(P, rgb, closed, smooth, elev, azim, flim, fbox, floor):
    """One grid panel as an RGBA array: an 8 x 8 in, 170 dpi transparent 3-D figure."""
    f = plt.figure(figsize=(8, 8), dpi=170)
    f.patch.set_alpha(0.0)
    a = f.add_axes([0, 0, 1, 1], projection="3d")
    a.patch.set_alpha(0.0)
    draw_path(a, P, rgb, closed, smooth, LINE_W, POINT_SIZE, EDGE, FLOOR_COLOR, SHADOW_COLOR,
              flim, fbox, floor)
    a.set_axis_off()
    a.view_init(elev=elev, azim=azim)
    f.canvas.draw()
    arr = np.asarray(f.canvas.buffer_rgba()).copy()
    plt.close(f)
    return arr


def alpha_bbox(img):
    ys, xs = np.where(img[..., 3] > 0)
    return ys.min(), ys.max(), xs.min(), xs.max()


def render_grid(p: dict, which: str, out_pdf: Path, out_png: Path) -> list:
    """A 2 x 5 trace grid from the arrays of fig2_{ee,target}_grid.npz."""
    matplotlib.rcdefaults()
    elev, azim = GRID_VIEW[which]
    n = len(p["kinds"])
    shapes = [(p[f"world_{i}"], p[f"sub_{i}"], p[f"rgb_{i}"], bool(p["closed"][i]))
              for i in range(n)]
    halfxy = 0.5 + PAD
    zhi = max(max(o[:, 2].max(), s[:, 2].max()) for o, s, _, _ in shapes) + PAD
    flim = (-halfxy, halfxy, -halfxy, halfxy, -PAD, zhi)
    fbox = (2 * halfxy, 2 * halfxy, zhi + PAD)
    floor = (0.0, 0.0, halfxy, -PAD / 2)
    cols = []
    for P_n, sub_n, rgb, closed in shapes:
        cols.append((render_cell(P_n, rgb, closed, SMOOTH_WORLD, elev, azim, flim, fbox, floor),
                     render_cell(sub_n, rgb, closed, SMOOTH_SUB, elev, azim, flim, fbox, floor)))
    bbs = [alpha_bbox(im) for T, B in cols for im in (T, B)]
    pad = 8
    y0 = max(min(b[0] for b in bbs) - pad, 0)
    y1 = max(b[1] for b in bbs) + pad + 1
    x0 = max(min(b[2] for b in bbs) - pad, 0)
    x1 = max(b[3] for b in bbs) + pad + 1
    cols = [(T[y0:y1, x0:x1], B[y0:y1, x0:x1]) for T, B in cols]
    nrow, ncol = 2, n
    mx, my, cgap = 0.008, 0.012, 0.018
    col_w = (1 - 2 * mx - (ncol - 1) * cgap) / ncol
    row_h = (1 - 2 * my - (nrow - 1) * ROW_GAP) / nrow
    h_px, w_px = cols[0][0].shape[:2]
    fig_w = GRID_COL_W * ncol
    fig_h = fig_w * col_w * h_px / (row_h * w_px)
    a2f = fig_h / fig_w
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def fit_k(img):
        h, w = img.shape[:2]
        return min(row_h / h, col_w / (w * a2f))

    def place(img, r, c, k):
        h, w = img.shape[:2]
        wf, hf = w * k * a2f, h * k
        x = mx + c * (col_w + cgap)
        y = 1 - my - row_h - r * (row_h + ROW_GAP)
        ax = fig.add_axes([x + (col_w - wf) / 2, y + (row_h - hf) / 2, wf, hf])
        ax.imshow(img)
        ax.axis("off")

    for i, (imgT, imgB) in enumerate(cols):
        k = min(fit_k(imgT), fit_k(imgB))
        place(imgT, 0, i, k)
        place(imgB, 1, i, k)
    fig.savefig(out_pdf, facecolor="white", dpi=GRID_PDF_DPI)
    fig.savefig(out_png, facecolor="white", dpi=GRID_PNG_DPI)
    plt.close(fig)
    return [out_pdf, out_png]


def load(panels_dir: Path, name: str) -> dict:
    with np.load(panels_dir / f"{name}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Panel files: see the module docstring.")
    ap.add_argument("--panel", nargs="+", choices=PANELS, default=list(PANELS),
                    help="files to render (default: all three)")
    ap.add_argument("--panels", type=Path, default=None,
                    help="directory with fig2_top.npz, fig2_ee_grid.npz, fig2_target_grid.npz "
                         "(default: $VLAGEOM_DATA/panels)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: $VLAGEOM_FIGURES, else $VLAGEOM_DATA/figures)")
    args = ap.parse_args(argv)

    panels_dir = args.panels if args.panels is not None else paths.panels_dir()
    out_dir = args.out_dir if args.out_dir is not None else paths.figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in args.panel:
        pdf, png = out_dir / f"fig2_{name}.pdf", out_dir / f"fig2_{name}.png"
        if name == "top":
            written += render_top(load(panels_dir, "fig2_top"), pdf, png)
        else:
            written += render_grid(load(panels_dir, f"fig2_{name}"), name, pdf, png)
    for w in written:
        print(f"[fig2] wrote {w}", flush=True)


if __name__ == "__main__":
    main()
