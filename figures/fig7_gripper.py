#!/usr/bin/env python
"""Render the four data panels of Figure 7 (gripper control), one file pair per panel.

The paper's Figure 7 places these four panels side by side, with the labels a-d, in a layout made
by hand. This script writes each panel as its own figure:

  fig7a  explained-variance ratio of edit PC1-16 (bars), with "stable rank = ..."    5.5 x 4.6 in
  fig7b  R^2 of the gripper-command probe on the top k edit PCs (k = 1 circled), and
         the probe on the full activation (dashed)                                   7.0 x 5.0 in
  fig7c  activations near the grasp on edit PC1-PC2, coloured by the decoded gripper
         command, with the binned mean edit per grid cell (close / open)              7.6 x 6.6 in
  fig7d  median arm leakage of the edits restricted to the top k PCs, relative to the
         unrestricted 768-D edit (%), closing and opening edits                      7.5 x 5.6 in

Inputs (under `vlageom.paths.panels_dir()`, default $VLAGEOM_DATA/panels):

  gripper_fig7.npz   (pipelines/gripper_fig7/50_panel.py)
                     scree_evr (16,), stable_rank          panel a
                     probe_k, probe_r2 (10,), probe_r2_full  panel b
                     leak_k (13,), leak_close, leak_open (13,), free_leak_close, free_leak_open
                                                           panel d
  gripper.npz        (pipelines/gripper/20_panel.py, the Figure 1 gripper tile)
                     px, py (cloud on edit PC1/PC2), aper (decoded gripper command, -1 .. +1),
                     fc, fo (binned field of the closing / opening edits: x, y, dx, dy per
                     cell), scale (arrow scale), evr (axis-label percentages)   panel c

Output: <out-dir>/fig7<panel>.pdf and .png (default `vlageom.paths.figures_dir()`), 350 dpi, tight
bounding box, white background; each file is drawn from a new figure.

Style: `vlageom.plot.style` (Product Sans from $VLAGEOM_FONT_DIR, Computer Modern math). Text
sizes: axis titles 26 pt, tick labels 22 pt, legends and the in-plot text 22 pt, colorbar title
24 pt and ticks 20 pt. The tick locations of the y axes of panels a and b and of both axes of
panel c are set by constants (SCREE_YTICKS, PROBE_YTICKS, FIELD_XTICKS, FIELD_YTICKS). Panel c's
colorbar is the scatter colormap mixed half and half with white, close to the apparent colour of
the semi-transparent points (alpha 0.65) on white.

Run:  python figures/fig7_gripper.py [--panel a b c d] [--panels DIR] [--out-dir DIR]

Without --panel all four panels are rendered, except that panel d is left out, with a message,
when gripper_fig7.npz has no panel-d arrays (stages 30 and 40 not run). With --panel d a missing
panel-d array is an error.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

from vlageom import gripper_fig7 as F  # noqa: E402
from vlageom import paths  # noqa: E402
from vlageom.plot import style as ps  # noqa: E402

PANELS = ("a", "b", "c", "d")
DPI = 350
FONTS = dict(label=26, tick=22, legend=22, annot=22, cbar_label=24, cbar_tick=20)

CLOSE_COLOR = ps.GRIPPER_CLOSE      # closing edits (before the grasp)
OPEN_COLOR = ps.GRIPPER_OPEN        # opening edits (after the grasp)

# panel a
SCREE_FIGSIZE = (5.5, 4.6)
SCREE_XTICKS = [1, 4, 8, 12, 16]
SCREE_YTICKS = [0.1 * i for i in range(8)]                 # 0 .. 0.7
SCREE_YLABEL = "explained\nvariance ratio"
# panel b
PROBE_FIGSIZE = (7.0, 5.0)
PROBE_YTICKS = [0.025 * i for i in range(32, 41)]          # 0.8 .. 1.0
# panel c
FIELD_FIGSIZE = (7.6, 6.6)
FIELD_CMAP = "berlin_r"             # decoded gripper command: red at -1 (open), blue at +1 (closed)
FIELD_XTICKS = [2.0 * i for i in range(-2, 4)]             # -4 .. 6
FIELD_YTICKS = [2.0 * i for i in range(-4, 4)]             # -8 .. 6
FIELD_CBAR_LABEL = "gripper aperture"
FIELD_CBAR_BLEND = 0.5              # colorbar = FIELD_CMAP * 0.5 + white * 0.5
# panel d
LEAK_FIGSIZE = (7.5, 5.6)
LEAK_YTICKS = [100, 200, 400]
LEAK_YMAX = 400
LEAK_XLABEL = "k  (number of $\\delta$-PCA components)"
LEAK_YLABEL = "leak / {dim}-D optimum  (%)"   # dim: the largest k, the full activation width


def _style():
    ps.apply_style()
    plt.rcParams.update({"axes.labelsize": FONTS["label"], "xtick.labelsize": FONTS["tick"],
                         "ytick.labelsize": FONTS["tick"], "legend.fontsize": FONTS["legend"]})


def _thin(labels):
    """Every other label plus the last one (the tick marks stay)."""
    return [str(k) if (i % 2 == 0 or i == len(labels) - 1) else "" for i, k in enumerate(labels)]


# ----------------------------------------------------------------------------- panel a
def render_a(p, path: Path):
    """Bars of the explained-variance ratio of edit PC1-16 and the stable-rank text."""
    evr = np.asarray(p["scree_evr"], np.float64)
    n_bars = len(evr)
    with plt.rc_context():
        _style()
        fig, ax = plt.subplots(figsize=SCREE_FIGSIZE)
        ax.bar(np.arange(1, n_bars + 1), evr[:n_bars], color=ps.PRIMARY, width=0.72,
               edgecolor="none")
        ax.set_xlabel("$\\delta$-PC index")
        ax.set_ylabel(SCREE_YLABEL, linespacing=1.3)
        ax.set_xticks(SCREE_XTICKS)
        ax.set_xlim(0.3, n_bars + 0.7)
        ax.set_ylim(0, max(0.62, evr[0] * 1.12))
        ax.set_yticks(SCREE_YTICKS)
        ax.text(0.97, 0.97, f"stable rank = {float(p['stable_rank']):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=FONTS["annot"],
                color=ps.META_LINE)
        ps.finish(ax)
        ps.save(fig, path, dpi=DPI)
    return [path]


# ----------------------------------------------------------------------------- panel b
def render_b(p, path: Path):
    """Probe R^2 against k (log axis, k = 1 circled) and the full-activation probe (dashed)."""
    ks = [int(k) for k in p["probe_k"]]
    r2 = [float(v) for v in p["probe_r2"]]
    with plt.rc_context():
        _style()
        fig, ax = plt.subplots(figsize=PROBE_FIGSIZE)
        ax.plot(ks, r2, "-o", ms=6, lw=2, color=ps.PRIMARY, label="top-$k$")
        ax.scatter([ks[0]], [r2[0]], s=170, facecolors="none", edgecolors=ps.PRIMARY, lw=2,
                   zorder=5)
        ax.axhline(float(p["probe_r2_full"]), ls="--", lw=2, color=ps.VIOLET,
                   label="full activation")
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(ks)
        ax.set_xticklabels(_thin(ks))
        ax.set_xlabel("k  (number of $\\delta$-PCs)")
        ax.set_ylabel("aperture decoder $R^2$")
        ax.set_ylim(min(0.8, min(r2) - 0.03), 1.01)
        ax.set_yticks(PROBE_YTICKS)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
        ps.finish(ax)
        ps.save(fig, path, dpi=DPI)
    return [path]


# ----------------------------------------------------------------------------- panel c
def render_c(g, path: Path):
    """Activations on edit PC1/PC2 coloured by the decoded gripper command, binned edit field."""
    px, py, aper = g["px"], g["py"], g["aper"]
    fc, fo = g["fc"], g["fo"]
    scale = np.float32(g["scale"])          # the arrow scale is computed in float32
    evr = g["evr"]
    with plt.rc_context():
        _style()
        fig, ax = plt.subplots(figsize=FIELD_FIGSIZE)
        ax.scatter(px, py, c=aper, cmap=FIELD_CMAP, s=9, alpha=0.65, vmin=-1, vmax=1,
                   edgecolors="none")
        qkw = dict(angles="xy", scale_units="xy", scale=1, width=0.005, headwidth=4.5,
                   headlength=5, alpha=0.9, zorder=4)
        ax.quiver(fc[:, 0], fc[:, 1], fc[:, 2] * scale, fc[:, 3] * scale, color=CLOSE_COLOR,
                  **qkw)
        ax.quiver(fo[:, 0], fo[:, 1], fo[:, 2] * scale, fo[:, 3] * scale, color=OPEN_COLOR,
                  **qkw)
        base = matplotlib.colormaps[FIELD_CMAP](np.linspace(0, 1, 256))[:, :3]
        blend = FIELD_CBAR_BLEND
        cmap_cb = ListedColormap(blend * base + (1.0 - blend), name=f"{FIELD_CMAP}_blend")
        cb = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=cmap_cb), ax=ax,
                          fraction=0.046, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1], format="%g")
        cb.set_label(FIELD_CBAR_LABEL, fontsize=FONTS["cbar_label"])
        cb.ax.tick_params(labelsize=FONTS["cbar_tick"])
        cb.outline.set_visible(False)
        handles = [Line2D([0], [0], color=CLOSE_COLOR, lw=3.2, label="close"),
                   Line2D([0], [0], color=OPEN_COLOR, lw=3.2, label="open")]
        ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9,
                  edgecolor="none")
        ax.set_xlabel(f"PC1  ({evr[0]:.0%} var)")
        ax.set_ylabel(f"PC2  ({evr[1]:.0%} var)")
        ax.set_xticks(FIELD_XTICKS)
        ax.set_yticks(FIELD_YTICKS)
        ps.finish(ax)
        ps.save(fig, path, dpi=DPI)
    return [path]


# ----------------------------------------------------------------------------- panel d
def render_d(p, path: Path):
    """100 x median leak(k) / median leak of the 768-D edit, per landmark; categorical k axis."""
    ks = [int(k) for k in p["leak_k"]]
    xpos = {k: i for i, k in enumerate(ks)}
    series = (("close", CLOSE_COLOR, p["leak_close"], p["free_leak_close"]),
              ("open", OPEN_COLOR, p["leak_open"], p["free_leak_open"]))
    with plt.rc_context():
        _style()
        fig, axes = plt.subplots(1, 1, figsize=LEAK_FIGSIZE, squeeze=False)
        ax = axes[0][0]
        for label, color, leak, free in series:
            ys = [100.0 * float(v) / float(free) for v in leak]
            ax.plot([xpos[k] for k in ks], ys, "-o", ms=6, lw=1.8, color=color, label=label)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticks(LEAK_YTICKS)
        ax.set_yticklabels([f"{v:g}%" for v in LEAK_YTICKS])
        ax.set_ylim(90, LEAK_YMAX * 1.05)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(_thin(ks))
        ax.set_xlabel(LEAK_XLABEL)
        ax.set_ylabel(LEAK_YLABEL.format(dim=ks[-1]))
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")
        ps.finish(ax)
        ps.save(fig, path, dpi=DPI)
    return [path]


# ----------------------------------------------------------------------------- main
def load(panels_dir: Path, name: str, keys) -> dict:
    f = panels_dir / f"{name}.npz"
    if not f.exists():
        raise SystemExit(f"missing panel file {f}")
    d = dict(np.load(f, allow_pickle=False))
    missing = [k for k in keys if k not in d]
    if missing:
        raise SystemExit(f"{f}: missing arrays {missing}")
    return d


def has_panel_d(panels_dir: Path) -> bool:
    f = panels_dir / "gripper_fig7.npz"
    if not f.exists():
        return True                     # load() reports the missing file
    with np.load(f, allow_pickle=False) as z:
        return all(k in z.files for k in F.PANEL_KEYS_D)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Panel files: see the module docstring.")
    ap.add_argument("--panel", nargs="+", choices=PANELS, default=None,
                    help="panels to render (default: all four; d only when its arrays exist)")
    ap.add_argument("--panels", type=Path, default=None,
                    help="directory with gripper_fig7.npz and gripper.npz "
                         "(default: $VLAGEOM_DATA/panels)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: $VLAGEOM_FIGURES, else $VLAGEOM_DATA/figures)")
    args = ap.parse_args(argv)

    panels_dir = args.panels if args.panels is not None else paths.panels_dir()
    out_dir = args.out_dir if args.out_dir is not None else paths.figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    renders = dict(a=render_a, b=render_b, c=render_c, d=render_d)
    names = list(args.panel) if args.panel is not None else list(PANELS)
    if args.panel is None and not has_panel_d(panels_dir):
        names.remove("d")
        print("[fig7] panel d left out: gripper_fig7.npz has no panel-d arrays (run stages 30 and "
              "40, then 50; --panel d requires them)", flush=True)
    written = []
    for name in names:
        if name == "c":
            data = load(panels_dir, "gripper", ("px", "py", "aper", "fc", "fo", "scale", "evr"))
        else:
            data = load(panels_dir, "gripper_fig7",
                        F.PANEL_KEYS_D if name == "d" else F.PANEL_KEYS_AB)
        for ext in ("pdf", "png"):
            written += renders[name](data, out_dir / f"fig7{name}.{ext}")
    for w in written:
        print(f"[fig7] wrote {w}", flush=True)


if __name__ == "__main__":
    main()
