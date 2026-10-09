#!/usr/bin/env python
"""Render the five data panels of Figure 5 (task phase), one file pair per panel.

The paper's Figure 5 places these panels, a robot sketch, the title and the labels a-d in a layout
made by hand. This script writes each data panel as its own figure:

  fig5_strip  example pick-and-place trajectory: the true phase of every query (ribbon, named),
              two exo and wrist frames per phase framed in the colour of the phase probes at that
              query, and the probe colour of every query ("recovered") along the time axis
                                                                    page 9.63 x 4.04 in
  fig5a       the successful natural queries in the first two LDA directions, by true phase,
              legend above the axes                                 page 4.85 x 5.13 in
  fig5b       "Phase subspace": PC1-3 of the natural queries (6,000 points, by true phase) on a
              grey floor, with the example trajectory as a spline coloured by the probe readouts
              (a 350 dpi raster; the PDF holds the same picture)    page 4.95 x 5.06 in
  fig5c       decoded correctly (%), natural vs frozen, one line per phase, the phase names as a
              one-row legend above the axes, chance at 25%; right of a rule: the most frequent
              decoded phase(s) when frozen with their shares: the top two for reach and place
              (C_RUNNER_UP, a fixed presentation choice of the figure, not computed from the
              data), the top one for grasp and transport            page 6.40 x 4.39 in
  fig5d       change in the fraction of the 8 executed chunk steps that command a closed gripper,
              frozen - natural, per phase (mean +- 95% CI), y axis -0.8 to 0.6
                                                                    page 4.37 x 3.51 in

Inputs (under `vlageom.paths.panels_dir()`, default $VLAGEOM_DATA/panels), written by
pipelines/phase/70_panels.py:
  fig5_strip.npz  lab_q (Q,), rgb (Q, 3), cols (8,), frames_exo / frames_wrist (8, H, W, 3)
  fig5_a.npz      emb (n, 2), y (n,)
  fig5_b.npz      traj (Q, 3), rgb (Q, 3), bg (6000, 3), bg_lab (6000,), evr (3,)
  fig5_c.npz      phases, own_natural, own_frozen, top, top_pct, second, second_pct (4,)
  fig5_d.npz      phases, mean, ci (4,)

Output: <out-dir>/fig5_strip, fig5a .. fig5d (.pdf and .png, 350 dpi; default
`vlageom.paths.figures_dir()`). Page sizes above are those of the PDFs (tight crop).

Style: `vlageom.plot.style` (Product Sans from $VLAGEOM_FONT_DIR, Computer Modern math, TrueType
fonts embedded in the PDFs). Phase colours: reach #2563eb, grasp #8b5cf6, transport #f43f5e,
place #f59e0b.

Run:  python figures/fig5.py [--panel strip a b c d] [--panels DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from vlageom import paths  # noqa: E402
from vlageom import phase as P  # noqa: E402
from vlageom.plot import style as ps  # noqa: E402

PANELS = ("strip", "a", "b", "c", "d")
DPI = 350
NAMES = P.NAMES
COLOR = {P.PHASE_NAMES[k]: P.PHASE_COLORS[k] for k in range(P.N_PHASE)}
PHASE_RGB = [to_rgb(P.PHASE_COLORS[k]) for k in range(P.N_PHASE)]
GREY = ps.SLATE[3]                       # #64748b
FAMILY = None                            # body font family, set by apply()


def apply():
    global FAMILY
    FAMILY = ps.apply_style()
    plt.rcParams["pdf.fonttype"] = 42


# ===================================================================================== strip
STRIP_FIGW = 10.0
STRIP_ML, STRIP_MR = 1.28, 0.30          # left gutter holds the "ground truth" / "recovered" tags
STRIP_GAP_C = 0.035
STRIP_NCOL = 8
STRIP_CW = STRIP_FIGW - STRIP_ML - STRIP_MR
STRIP_COLW = (STRIP_CW - (STRIP_NCOL - 1) * STRIP_GAP_C) / STRIP_NCOL


def _strip_ribbon(fig, lab_q, Q, y_in, h_in, figh, tag):
    """The true phase of each query as coloured, named segments on the time axis."""
    ax = fig.add_axes([STRIP_ML / STRIP_FIGW, y_in / figh, STRIP_CW / STRIP_FIGW, h_in / figh])
    ax.set_xlim(0, Q)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    gap = 0.055 * Q / max(P.N_PHASE, 1)
    for p in range(P.N_PHASE):
        for run in P.runs_of(lab_q, p):
            x0, x1 = run[0], run[-1] + 1
            a = x0 + (gap / 2 if x0 > 0 else 0.0)
            b = x1 - (gap / 2 if x1 < Q else 0.0)
            ax.add_patch(plt.Rectangle((a, 0.06), b - a, 0.88, color=PHASE_RGB[p], ec="none"))
            if (x1 - x0) >= 1.2:
                ax.text((x0 + x1) / 2, 0.5, P.PHASE_NAMES[p], ha="center", va="center",
                        color="white", fontsize=13, fontfamily=FAMILY)
    ax.text(-0.008, 0.5, tag, transform=ax.transAxes, ha="right", va="center",
            fontsize=11, color=ps.INK, fontfamily=FAMILY)
    return ax


def _strip_filmstrip(fig, frames_exo, frames_wrist, cols, y_top, h_img, tint):
    """Two rows of frames (exo over wrist) stacked down from y_top (in from the bottom)."""
    figh = fig.get_figheight()
    for j, q in enumerate(cols):
        x = (STRIP_ML + j * (STRIP_COLW + STRIP_GAP_C)) / STRIP_FIGW
        for r, stack in ((0, frames_exo), (1, frames_wrist)):
            ax = fig.add_axes([x, (y_top - (r + 1) * h_img) / figh,
                               STRIP_COLW / STRIP_FIGW, (h_img - 0.012) / figh])
            if q < 0:
                ax.set_facecolor("#f1f5f9")
                ax.text(.5, .5, "—", ha="center", va="center", color="#94a3b8", fontsize=11)
            else:
                ax.imshow(stack[j], aspect="auto", interpolation="antialiased")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(1.3)
                s.set_color(tint[j])
            if j == 0:
                ax.text(0.05, 0.94, "Exo" if r == 0 else "Wrist", transform=ax.transAxes,
                        ha="left", va="top", fontsize=17, color=ps.INK, fontfamily=FAMILY,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.92))


def render_strip(d, out_dir: Path):
    rgb, lab_q, cols = d["rgb"], d["lab_q"].astype(int), [int(c) for c in d["cols"]]
    Q = len(rgb)
    h_img, h_rib, h_spec = 1.06, 0.34, 0.40
    pad = 0.26
    figh = h_rib + 2 * h_img + h_spec + pad + 0.82
    fig = plt.figure(figsize=(STRIP_FIGW, figh))
    fig.patch.set_facecolor("white")
    y = figh - 0.12
    _strip_ribbon(fig, lab_q, Q, y - h_rib, h_rib, figh, tag="ground truth   ")
    y -= h_rib + 0.30
    tint = [rgb[q] if q >= 0 else np.array([.85, .87, .9]) for q in cols]
    _strip_filmstrip(fig, d["frames_exo"], d["frames_wrist"], cols, y, h_img, tint)
    y -= 2 * h_img + 0.12
    axs = fig.add_axes([STRIP_ML / STRIP_FIGW, (y - h_spec) / figh,
                        STRIP_CW / STRIP_FIGW, h_spec / figh])
    axs.imshow(rgb[None, :, :], aspect="auto", interpolation="bicubic", extent=[0, Q, 0, 1])
    axs.set_xticks([])
    axs.set_yticks([])
    for sp in axs.spines.values():
        sp.set_visible(False)
    axs.text(-0.008, 0.5, "recovered   ", transform=axs.transAxes, ha="right", va="center",
             fontsize=11, color=ps.INK, fontfamily=FAMILY)
    y -= h_spec + pad
    ft = fig.add_axes([STRIP_ML / STRIP_FIGW, (y - 0.30) / figh,
                       STRIP_CW / STRIP_FIGW, 0.28 / figh])
    ft.axis("off")
    ft.annotate("", xy=(0.062, .58), xytext=(0.0, .58), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_width=.19,head_length=.38",
                                color=ps.INK, lw=1.9, shrinkA=0, shrinkB=0))
    ft.text(0.074, .58, "t", ha="left", va="center", fontsize=18, color=ps.INK, fontfamily=FAMILY)
    fig.savefig(out_dir / "fig5_strip.png", dpi=DPI, bbox_inches="tight", transparent=False)
    fig.savefig(out_dir / "fig5_strip.pdf", bbox_inches="tight", transparent=False)
    plt.close(fig)


# ===================================================================================== a
A_FS = {"tick": 20, "label": 26, "legend": 20}
A_TICKS = {"x": [-10, -8, -6, -4, -2, 0, 2, 4, 6], "y": [-6, -4, -2, 0, 2, 4, 6]}
A_LABELPAD = 8.0


def render_a(d, out_dir: Path):
    e2, y = d["emb"], d["y"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    hands = []
    for k in range(P.N_PHASE):
        mk = y == k
        hands.append(ax.scatter(e2[mk, 0], e2[mk, 1], s=7, alpha=0.55, c=P.PHASE_COLORS[k],
                                label=NAMES[k], edgecolors="none"))
    ax.set_xlabel("LD1", fontsize=A_FS["label"], labelpad=A_LABELPAD)
    ax.set_ylabel("LD2", fontsize=A_FS["label"], labelpad=A_LABELPAD)
    xl, yl = ax.get_xlim(), ax.get_ylim()          # pin the ticks without widening the limits
    ax.set_xticks(A_TICKS["x"])
    ax.set_yticks(A_TICKS["y"])
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ps.finish(ax)
    ax.tick_params(labelsize=A_FS["tick"])
    fig.tight_layout()
    order = [0, 2, 1, 3]                           # 2 x 2 legend: reach / grasp, transport / place
    ax.legend([hands[i] for i in order], [NAMES[i] for i in order],
              frameon=True, framealpha=0.9, edgecolor="none", markerscale=4.0,
              fontsize=A_FS["legend"], ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handletextpad=0.4)
    fig.savefig(out_dir / "fig5a.pdf", bbox_inches="tight", transparent=False)
    fig.savefig(out_dir / "fig5a.png", dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close(fig)


# ===================================================================================== b
B_FS = {"pc_label": 28, "pc_off_pt": 24, "title": 40}
B_TITLE = "Phase subspace"


def _floor_edge_labels(f, ax, labels, fontsize, off_pt):
    """Axis names laid along the projected floor edges, pushed off_pt outward."""
    from mpl_toolkits.mplot3d import proj3d
    f.canvas.draw()
    cx, cy, zf = np.mean(ax.get_xlim()), np.mean(ax.get_ylim()), ax.get_zlim()[0]
    u, v, _ = proj3d.proj_transform(cx, cy, zf, ax.get_proj())
    ctr = np.asarray(ax.transData.transform((u, v)))
    for axis, lab in zip((ax.xaxis, ax.yaxis), labels):
        p0, p1 = np.asarray(ax.transData.transform(np.column_stack(axis.line.get_data())))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        if abs(ang) > 90:
            ang -= np.sign(ang) * 180
        perp = np.array([p0[1] - p1[1], p1[0] - p0[0]])
        perp /= np.linalg.norm(perp)
        mid = (p0 + p1) / 2
        if perp @ (mid - ctr) < 0:
            perp = -perp
        pos = f.transFigure.inverted().transform(mid + perp * off_pt * f.dpi / 72)
        f.text(*pos, lab, rotation=ang, rotation_mode="anchor", ha="center", va="center",
               fontsize=fontsize, color=ps.INK, fontfamily=FAMILY)


def _pca_figure(traj, rgb, bg, bg_lab, evr, figsize=(8.4, 6.4), elev=20, azim=-60):
    from scipy.interpolate import make_interp_spline
    Q = len(traj)
    f = plt.figure(figsize=figsize)
    f.patch.set_facecolor("white")
    ax = f.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d", computed_zorder=False)
    x3, y3, z3 = traj[:, 0], traj[:, 1], traj[:, 2]
    lo, hi = np.percentile(bg, [1, 99], axis=0)        # frame on the cloud's bulk
    keep = ((bg >= lo) & (bg <= hi)).all(1)
    zf = ps.setup_floor(ax, np.vstack([bg[keep], traj]), pad=1.06, color=ps.META)
    base = np.array(PHASE_RGB)
    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], c=base[np.clip(bg_lab, 0, P.N_PHASE - 1)],
               s=2.6, alpha=0.20, edgecolors="none", depthshade=False, zorder=0)
    tq = np.arange(Q)
    ts = np.linspace(0, Q - 1, 24 * max(Q - 1, 1))
    k = 3 if Q > 3 else max(Q - 1, 1)
    sp = np.stack([make_interp_spline(tq, traj[:, i], k=k)(ts) for i in range(3)], 1)
    sc = np.stack([np.interp(ts, tq, rgb[:, i]) for i in range(3)], 1)
    ax.scatter(sp[:, 0], sp[:, 1], np.full(len(sp), zf), c=ps.INK, s=8, alpha=0.05,
               edgecolors="none", depthshade=False, zorder=1)                 # shadow
    for i in range(len(sp) - 1):                                              # path
        ax.plot(sp[i:i + 2, 0], sp[i:i + 2, 1], sp[i:i + 2, 2], color=sc[i], lw=3.0,
                alpha=0.9, zorder=2, solid_capstyle="round")
    ax.scatter(x3, y3, z3, c=rgb, s=330, alpha=0.18, edgecolors="none", depthshade=False, zorder=3)
    ax.scatter(x3, y3, z3, c=rgb, s=110, alpha=0.98, edgecolors="none", depthshade=False, zorder=4)
    ax.scatter(x3, y3, z3, c="white", s=210, alpha=0.55, edgecolors="none", depthshade=False,
               zorder=3)
    ax.scatter(x3, y3, z3, c=rgb, s=110, alpha=1.0, edgecolors="none", depthshade=False, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_zlabel("")
    ax.view_init(elev=elev, azim=azim)
    names = [f"{n} ({100 * v:.0f}%)" for n, v in zip(("PC1", "PC2"), evr)]
    _floor_edge_labels(f, ax, names, B_FS["pc_label"], B_FS["pc_off_pt"])
    return f


def _crop_raster(f, dpi=DPI, pad=8):
    """Rasterise a figure and trim its white margin (pixels < 248 in any channel count as ink)."""
    from PIL import Image
    buf = io.BytesIO()
    f.savefig(buf, dpi=dpi, transparent=False, facecolor="white")
    buf.seek(0)
    im = np.asarray(Image.open(buf).convert("RGB"))
    ys, xs = np.where((im < 248).any(2))
    return im[max(ys.min() - pad, 0):ys.max() + pad + 1, max(xs.min() - pad, 0):xs.max() + pad + 1]


def _title_raster(im, title, dpi=DPI, fontsize=B_FS["title"], gap_in=0.07):
    """Set a title above a cropped raster and crop again."""
    h, w = im.shape[:2]
    w_in, h_in = w / dpi, h / dpi
    band = fontsize / 72 * 1.35 + gap_in
    f = plt.figure(figsize=(w_in, h_in + band), dpi=dpi)
    f.patch.set_facecolor("white")
    ax = f.add_axes([0, 0, 1, h_in / (h_in + band)])
    ax.imshow(im, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    f.text(0.5, 1.0 - 0.004, title, ha="center", va="top", fontsize=fontsize, color=ps.INK,
           fontfamily=FAMILY)
    out = _crop_raster(f, dpi=dpi, pad=6)
    plt.close(f)
    return out


def _raster_to_pdf(im, path, dpi=DPI):
    """A PDF page of the raster's physical size at `dpi` holding just the raster."""
    h, w = im.shape[:2]
    f = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    f.patch.set_facecolor("white")
    ax = f.add_axes([0, 0, 1, 1])
    ax.imshow(im, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    f.savefig(path, dpi=dpi, transparent=False, facecolor="white")
    plt.close(f)


def render_b(d, out_dir: Path):
    from PIL import Image
    f = _pca_figure(d["traj"], d["rgb"], d["bg"], d["bg_lab"].astype(int), d["evr"])
    raster = _crop_raster(f)
    plt.close(f)
    raster = _title_raster(raster, B_TITLE)
    Image.fromarray(raster).save(out_dir / "fig5b.png")
    _raster_to_pdf(raster, out_dir / "fig5b.pdf")


# ===================================================================================== c
C_FS = {"tick": 22, "label": 26, "legend": 20, "annot": 20, "chance": 20}
C_FIGSIZE = (6.4, 4.4)
C_PAD_IN = 0.1           # tight-crop pad and the in-figure content margin
C_GAP_PT = 9.0           # marker centre -> "chance" ink, and marker centre -> separator rule
C_RUN_GAP_PT = 5.0       # white between a leader head and its label
C_LABEL_SEP_PT = 3.0     # minimum white between two stacked labels
C_LINE_SEP_PT = 2.0      # line pitch spacer of two-line labels
C_BLOCK_PAD_PT = 3.0     # extra white above and below a two-line label
C_MARKER_S = 80          # marker area (pt^2)
C_MARK_SEP_PT = 1.5      # white between two markers moved apart at the same end
C_LW = 2.6
C_LEGEND = dict(handlelength=0.6, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.6,
                borderpad=0)     # one-row phase-name legend above the axes, centred on them
C_HALO_PT = 1.5          # white ring around a moved marker that sits on another phase's line
C_LINE_CLEAR_PT = 1.5
C_LEADER_TAIL_PT = 3.0   # separator rule -> leader tail
C_LEADER_DX_PT = 30.0    # leader length along x
C_LEADER_LW = 1.5
C_SEP_LW = 1.0
C_CHANCE = 100.0 / P.N_PHASE       # % decoded correctly by chance (four phases)
# Presentation choice of the figure (not computed from the data): the phases whose right-hand
# label also names the second most frequent decoded phase when frozen, with its share.
C_RUNNER_UP = ("reach", "place")
C_COND = ("natural", "frozen")


def _dodge_steps(vals, thresh):
    """Marker-pitch steps per phase at one end: walking the markers from the top down, a marker
    within `thresh` of the one above it takes one more step than that one."""
    order = sorted(NAMES, key=lambda k: -vals[k])
    steps, run = {order[0]: 0}, 0
    for above, k in zip(order, order[1:]):
        run = run + 1 if vals[above] - vals[k] < thresh else 0
        steps[k] = run
    return steps


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.hypot(*(p - (a + t * ab))))


def _spread(y, half, sep, lo, hi):
    """Minimal-displacement 1-D spreading of label boxes [p - half, p + half], at least `sep`
    apart and inside [lo, hi]."""
    y = np.asarray(y, float)
    half = np.asarray(half, float)
    order = np.argsort(y)
    p, h = y[order].copy(), half[order]
    for _ in range(200):
        moved = False
        for i in range(1, len(p)):
            over = (h[i - 1] + h[i] + sep) - (p[i] - p[i - 1])
            if over > 1e-9:
                p[i - 1] -= over / 2
                p[i] += over / 2
                moved = True
        p = np.clip(p, lo + h, hi - h)
        if not moved:
            break
    out = np.empty_like(p)
    out[order] = p
    return out


def render_c(d, out_dir: Path):
    phases = [str(s) for s in d["phases"]]
    own_u = dict(zip(phases, d["own_natural"]))
    own_f = dict(zip(phases, d["own_frozen"]))
    dest = {k: dict(top=str(t), top_pct=tp, second=str(s), second_pct=sp)
            for k, t, tp, s, sp in zip(phases, d["top"], d["top_pct"], d["second"],
                                       d["second_pct"])}

    W, H = C_FIGSIZE
    fig = plt.figure(figsize=C_FIGSIZE)
    ax = fig.add_axes([0.30, 0.20, 0.66, 0.72])
    ax.set_ylim(0, 100)
    lines, marks = {}, {}
    for k in NAMES:
        lines[k], = ax.plot([0, 1], [own_u[k], own_f[k]], color=COLOR[k], lw=C_LW,
                            solid_capstyle="round", zorder=3, clip_on=False)
        marks[k] = [ax.scatter([x], [y], s=C_MARKER_S, color=COLOR[k], edgecolors="none", zorder=4,
                               clip_on=False) for x, y in ((0, own_u[k]), (1, own_f[k]))]
    ax.plot([-0.03, 1.03], [C_CHANCE, C_CHANCE], ls=(0, (4, 3)), lw=1.2, color=ps.META_LINE,
            zorder=1, clip_on=False)
    chance_txt = ax.annotate("chance", xy=(0, C_CHANCE), xytext=(-C_GAP_PT, 0),
                             textcoords="offset points", ha="right", va="center",
                             fontsize=C_FS["chance"], color=GREY, annotation_clip=False)
    # right column: "<phase> NN%" (comma-wrapped onto a second line with the runner-up read)
    runs, run_line = {}, {}
    for k in NAMES:
        dk = dest[k]
        share = int(np.floor(dk["top_pct"] + 0.5))             # nearest integer, half up
        comma = "," if k in C_RUNNER_UP else ""
        parts = [(f"{dk['top']} {share}%{comma}", COLOR[dk["top"]], 0)]
        if k in C_RUNNER_UP:
            share2 = int(np.floor(dk["second_pct"] + 0.5))
            parts.append((f"{dk['second']} {share2}%", COLOR[dk["second"]], 1))
        runs[k] = [ax.annotate(txt, xy=(1, own_f[k]), xytext=(0, 0), textcoords="offset points",
                               ha="left", va="center", fontsize=C_FS["annot"], color=col,
                               annotation_clip=False) for txt, col, _ in parts]
        run_line[k] = [ln for *_, ln in parts]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(C_COND))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("decoded correctly (%)", fontsize=C_FS["label"], labelpad=8)
    ps.finish(ax, trim=True)
    ax.tick_params(labelsize=C_FS["tick"])
    # phase names: one-row legend above the axes, centred on them; made before the layout loop,
    # which lowers the axes top to make room for it (the page size is fixed)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOR[k], lw=C_LW, marker="o", ms=np.sqrt(C_MARKER_S),
                      solid_capstyle="round") for k in NAMES]
    leg = ax.legend(handles, NAMES, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                    bbox_transform=ax.transAxes, ncol=len(NAMES), frameon=False,
                    fontsize=C_FS["legend"], labelcolor=[COLOR[k] for k in NAMES], **C_LEGEND)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pt = lambda px: px * 72.0 / fig.dpi  # noqa: E731
    mark_r = np.sqrt(C_MARKER_S) / 2.0
    run_w = {k: [pt(t.get_window_extent(r).width) for t in runs[k]] for k in NAMES}
    Lw = pt(chance_txt.get_window_extent(r).width)
    th = pt(runs[NAMES[0]][0].get_window_extent(r).height)
    line_pitch = th + C_LINE_SEP_PT
    R0 = C_GAP_PT + C_LEADER_TAIL_PT + C_LEADER_DX_PT + C_RUN_GAP_PT   # marker centre -> label
    Rw = 0.0
    for k in NAMES:
        n_lines = max(run_line[k]) + 1
        for ln in range(n_lines):
            x = R0
            dy = (n_lines - 1) * line_pitch / 2 - ln * line_pitch
            for i, (t, w) in enumerate(zip(runs[k], run_w[k])):
                if run_line[k][i] != ln:
                    continue
                t.xyann = (x, dy)
                x += w + C_RUN_GAP_PT
            Rw = max(Rw, x - C_RUN_GAP_PT - R0)
    right_half = {k: ((max(run_line[k]) + 1) * th + max(run_line[k]) * C_LINE_SEP_PT) / 2
                     + (C_BLOCK_PAD_PT if max(run_line[k]) else 0.0) for k in NAMES}
    line0_dy = {k: max(run_line[k]) * line_pitch / 2 for k in NAMES}
    L = Lw + C_GAP_PT + mark_r
    R = R0 + Rw + mark_r

    state = {"sep": None, "extra": []}

    def relayout():
        ax_w = ax.get_position().width * W * 72
        ax_h = ax.get_position().height * H * 72
        S = ax_w - L - R                                   # slope span (pt)
        ax.set_xlim(-L / S, 1 + R / S)
        ps.finish(ax, trim=True)
        ax.tick_params(labelsize=C_FS["tick"])
        u = 100.0 / ax_h                                   # pt -> % units
        pitch = 2 * np.sqrt(C_MARKER_S) / 2.0 + C_MARK_SEP_PT
        st_u, st_f = _dodge_steps(own_u, pitch * u), _dodge_steps(own_f, pitch * u)
        for k in NAMES:                                    # move coincident markers inward
            xu, xf = st_u[k] * pitch / S, 1 - st_f[k] * pitch / S
            lines[k].set_data([xu, xf], [own_u[k], own_f[k]])
            marks[k][0].set_offsets([[xu, own_u[k]]])
            marks[k][1].set_offsets([[xf, own_f[k]]])
        for a in state["extra"]:
            a.remove()
        state["extra"] = []

        def Pt(j, e):
            return np.array([lines[j].get_xdata()[e] * S, lines[j].get_ydata()[e] / u])

        for k in NAMES:                                    # halo: moved marker on another line
            for e, st in ((0, st_u[k]), (1, st_f[k])):
                if not st:
                    continue
                dmin = min(_seg_dist(Pt(k, e), Pt(j, 0), Pt(j, 1)) for j in NAMES if j != k)
                if dmin - mark_r - C_LW / 2 < C_LINE_CLEAR_PT:
                    off = marks[k][e].get_offsets()[0]
                    state["extra"].append(ax.scatter([off[0]], [off[1]],
                                                     s=(2 * (mark_r + C_HALO_PT)) ** 2,
                                                     color="white", edgecolors="none", zorder=3.5,
                                                     clip_on=False))
                    lines[k].set_zorder(3.6)
        sep = C_LABEL_SEP_PT * u
        yr = _spread([own_f[k] for k in NAMES], [right_half[k] * u for k in NAMES], sep, 0, 100)
        for k, yf in zip(NAMES, yr):
            for t in runs[k]:
                t.xy = (1, yf)
        if state["sep"] is not None:
            state["sep"].remove()
        padu = 2.0 * u
        y0 = min(yf - right_half[k] * u for k, yf in zip(NAMES, yr)) - padu
        y1 = max(yf + right_half[k] * u for k, yf in zip(NAMES, yr)) + padu
        tr = mtransforms.offset_copy(ax.transData, fig=fig, x=C_GAP_PT, y=0, units="points")
        state["sep"], = ax.plot([1, 1], [y0, y1], color=ps.META_LINE, lw=C_SEP_LW,
                                solid_capstyle="butt", transform=tr, zorder=2, clip_on=False)
        x_t, x_h = C_GAP_PT + C_LEADER_TAIL_PT, C_GAP_PT + C_LEADER_TAIL_PT + C_LEADER_DX_PT
        for k, yf in zip(NAMES, yr):                       # leader: marker height -> label line 1
            state["extra"].append(ax.annotate(
                "", xy=(1, yf + line0_dy[k] * u), xytext=(1, own_f[k]), annotation_clip=False,
                xycoords=mtransforms.offset_copy(ax.transData, fig=fig, x=x_h, y=0, units="points"),
                textcoords=mtransforms.offset_copy(ax.transData, fig=fig, x=x_t, y=0,
                                                   units="points"),
                arrowprops=dict(arrowstyle="->,head_length=0.38,head_width=0.22", color=GREY,
                                lw=C_LEADER_LW, mutation_scale=20, shrinkA=0, shrinkB=0,
                                capstyle="round"), zorder=2))
        return S

    for _ in range(4):        # fit the axes so the content fills the canvas minus C_PAD_IN
        relayout()
        fig.canvas.draw()
        bb = fig.get_tightbbox(r)
        pos = ax.get_position()
        lft, b, rgt, top = pos.x0 * W, pos.y0 * H, pos.x1 * W, pos.y1 * H
        lft += C_PAD_IN - bb.x0
        rgt += (W - C_PAD_IN) - bb.x1
        b += C_PAD_IN - bb.y0
        top += (H - C_PAD_IN) - bb.y1
        ax.set_position([lft / W, b / H, (rgt - lft) / W, (top - b) / H])
    S = relayout()
    fig.canvas.draw()

    texts = [t for k in NAMES for t in runs[k]] + [chance_txt] + ax.get_xticklabels() + \
        ax.get_yticklabels() + [ax.yaxis.label] + leg.get_texts()
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in texts if t.get_text()]
    clash = [f"{boxes[i][0]!r} x {boxes[j][0]!r}" for i in range(len(boxes))
             for j in range(i + 1, len(boxes)) if boxes[i][1].overlaps(boxes[j][1])]
    lb = leg.get_window_extent(r)
    top_mark = max(ax.transData.transform(m.get_offsets()[0])[1] for k in NAMES for m in marks[k])
    leg_gap = pt(lb.y0 - top_mark) - mark_r                # legend bottom -> top marker edge
    print(f"c: slope span {S:.0f} pt; " + ("text overlaps: " + "; ".join(clash) if clash
                                           else "no text overlaps")
          + f"; legend {pt(lb.width):.0f} pt wide, {leg_gap:.1f} pt above the highest marker")
    fig.savefig(out_dir / "fig5c.pdf", bbox_inches="tight", pad_inches=C_PAD_IN, transparent=False)
    fig.savefig(out_dir / "fig5c.png", dpi=DPI, bbox_inches="tight", pad_inches=C_PAD_IN,
                transparent=False)
    plt.close(fig)


# ===================================================================================== d
D_FS = {"tick": 22, "label": 26, "annot": 22}
D_YLIM = (-0.8, 0.6)
D_YTICKS = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
D_YLABEL = "Δ gripper closed\nfraction"
D_XSTAGGER = (0, 1, 0, 1)      # x labels alternate between two lines
D_XLINESPACING = 1.0
D_TL_PAD = 0.2


def render_d(d, out_dir: Path):
    phases = [str(s) for s in d["phases"]]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for i, k in enumerate(phases):
        ax.bar(i, d["mean"][i], width=0.62, color=COLOR[k])
        ax.errorbar(i, d["mean"][i], yerr=d["ci"][i], color=ps.INK, lw=1.3, capsize=3)
    ax.axhline(0, color=ps.INK, lw=0.9)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([("\n" * s) + n for s, n in zip(D_XSTAGGER, phases)], fontsize=D_FS["tick"])
    ax.set_ylim(*D_YLIM)
    ax.set_yticks(D_YTICKS)
    ax.set_ylabel(D_YLABEL, fontsize=D_FS["label"])
    ax.tick_params(labelsize=D_FS["tick"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for x, y, ha, va, txt in [(0.985, 0.98, "right", "top", "closes\nmore $\\uparrow$"),
                              (0.015, 0.02, "left", "bottom", "opens\nmore $\\downarrow$")]:
        ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=D_FS["annot"], color=GREY)
    ps.finish(ax)
    for t in ax.get_xticklabels():          # after finish, which rebuilds the tick labels
        t.set_linespacing(D_XLINESPACING)
    for _ in range(2):
        fig.tight_layout(pad=D_TL_PAD)
    extra = dict(bbox_inches="tight", transparent=False,
                 bbox_extra_artists=fig.get_default_bbox_extra_artists() + [ax.yaxis.label])
    fig.savefig(out_dir / "fig5d.pdf", **extra)
    fig.savefig(out_dir / "fig5d.png", dpi=DPI, **extra)
    plt.close(fig)


RENDER = {"strip": render_strip, "a": render_a, "b": render_b, "c": render_c, "d": render_d}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--panel", nargs="+", choices=PANELS, default=list(PANELS))
    ap.add_argument("--panels", type=Path, default=None, help="panel files (default <data>/panels)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default $VLAGEOM_FIGURES")
    a = ap.parse_args(argv)
    panels_dir = a.panels or paths.panels_dir()
    out_dir = a.out_dir or paths.figures_dir(create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply()
    for p in a.panel:
        name = "strip" if p == "strip" else p
        with np.load(panels_dir / f"fig5_{name}.npz", allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        RENDER[p](d, out_dir)
        stem = "fig5_strip" if p == "strip" else f"fig5{p}"
        print(f"wrote {out_dir / (stem + '.pdf')} and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
