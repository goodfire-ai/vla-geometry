"""Shared plot style for the paper figures, and the colormap helpers the figures use.

The style is Thomas Fel's house style: ink-on-white, Product Sans body text with Computer Modern
math, no grid, detached spines, slate greys. Call `apply_style()` once before drawing.

Fonts. Product Sans is not distributed with this repository. `apply_style()` registers every
.ttf/.otf file under $VLAGEOM_FONT_DIR (see `vlageom.paths.font_dir`) and uses Product Sans when it
is available from there or from the system fonts. Otherwise it uses DejaVu Sans, which ships with
matplotlib; text then has slightly different widths than in the paper, and the layout (set in
inches and points) is otherwise unchanged.

Colormaps. `cmap_spec(spec)` turns a spec string into a registered matplotlib colormap name:
  "name"                 a colormap that already exists (returned unchanged)
  "name:lo-hi[,lo-hi]"   the listed sub-ranges of `name`, concatenated
  "name~CTRL:lo-hi"      `name` with its middle t in [lo, hi] replaced by a CIELAB quadratic Bezier
                         from the color at lo to the color at hi through the control color
                         BRIDGE_CTRL[CTRL] (L*, a*, b*); the outer ranges keep their colors
`register_managua_red()` registers "managua_red": matplotlib's managua with the hue of its light
(gold) end turned 41 degrees toward red, tapering to no change at the dark center.

Helpers: `finish(ax)` cleans up 2-D axes, `setup_floor(ax, pts)` draws the grey floor of a 3-D
point cloud, `save(fig, path)` writes a panel with a tight bounding box.

"""
# Adapted from Thomas Fel's house plotting style (apply_style, finish, the slate palette).
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ----------------------------------------------------------------------------- palette
PRIMARY = "#2563eb"     # blue-600: main condition / variable of interest
META = "#e2e8f0"        # slate-200: neutral fills, bands, 3-D floors
META_LINE = "#94a3b8"   # slate-400: thin reference lines
INK = "#020617"         # slate-950: axes, ticks, all text

GRIPPER_CLOSE = PRIMARY     # gripper-close edits
GRIPPER_OPEN = "#dc2626"    # red-600: gripper-open edits

VIOLET = "#8b5cf6"          # violet-500: second condition / variable next to PRIMARY
GRAPHITE = "#354753"        # graphite: dark marks on light fills (Figure 6 b-d)

# Tailwind slate ramp, light to dark. META = SLATE[0], META_LINE = SLATE[2], INK = SLATE[-1].
SLATE = ["#e2e8f0", "#cbd5e1", "#94a3b8", "#64748b", "#475569",
         "#334155", "#1e293b", "#0f172a", "#020617"]

FIELD_CMAP = "managua"      # continuous coordinate on a point cloud

# ----------------------------------------------------------------------------- fonts
FONT_FAMILY = "Product Sans"
FALLBACK_FAMILY = "DejaVu Sans"     # bundled with matplotlib
_FONT_EXT = (".ttf", ".otf")
_warned = False


def _register_font_dir():
    """Register every font file under $VLAGEOM_FONT_DIR (sorted by path). Returns the count."""
    from vlageom.paths import font_dir

    d = font_dir()
    if d is None:
        return 0
    files = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in _FONT_EXT)
    for f in files:
        font_manager.fontManager.addfont(str(f))
    return len(files)


def body_family():
    """The body font family in use: FONT_FAMILY if matplotlib knows it, else FALLBACK_FAMILY."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    return FONT_FAMILY if FONT_FAMILY in names else FALLBACK_FAMILY


def apply_style():
    """Reset rcParams, register the figure fonts, set the body font and Computer Modern math.

    Returns the body font family that was set (Product Sans, or DejaVu Sans when Product Sans is
    not available; a warning is issued once in that case)."""
    global _warned
    matplotlib.rcdefaults()
    _register_font_dir()
    family = body_family()
    if family != FONT_FAMILY and not _warned:
        warnings.warn(f"{FONT_FAMILY} not found (set VLAGEOM_FONT_DIR to a directory with its font "
                      f"files); using {FALLBACK_FAMILY}. Text widths differ slightly from the paper.",
                      stacklevel=2)
        _warned = True
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False    # ascii hyphen-minus
    plt.rcParams["mathtext.fontset"] = "cm"        # Computer Modern for $...$
    return family


def finish(ax, offset=10, trim=False):
    """2-D axes cleanup: no grid, top/right spines hidden, left/bottom spines moved out by
    `offset` points, ink-colored spines and ticks. trim=True ends each spine at its outermost
    visible (major or minor) tick."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", offset))
    ax.spines["bottom"].set_position(("outward", offset))
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.grid(False)
    if trim:
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        xt = [t for t in (*ax.get_xticks(), *ax.get_xticks(minor=True)) if x0 <= t <= x1]
        yt = [t for t in (*ax.get_yticks(), *ax.get_yticks(minor=True)) if y0 <= t <= y1]
        if xt:
            ax.spines["bottom"].set_bounds(min(xt), max(xt))
        if yt:
            ax.spines["left"].set_bounds(min(yt), max(yt))


def save(fig, path, dpi=350):
    """tight_layout, then save `fig` to `path` at `dpi` with a tight bounding box and an opaque
    white background, and close it. Creates the parent directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=False)
    plt.close(fig)


def setup_floor(ax, pts, pad: float = 1.15, color: str = META):
    """Equal-cube 3-D axes around `pts` (half width = pad x half the largest extent) with a grey
    floor at the bottom of the cube; panes, grid and the z axis removed. Returns the floor
    height. The axes must be created with computed_zorder=False."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    pts = np.asarray(pts)
    c = pts.mean(0)
    r = (pts.max(0) - pts.min(0)).max() / 2
    half = r * pad
    xlim = (c[0] - half, c[0] + half)
    ylim = (c[1] - half, c[1] + half)
    zlim = (c[2] - half, c[2] + half)
    zf = zlim[0]
    floor = [[(xlim[0], ylim[0], zf), (xlim[1], ylim[0], zf),
              (xlim[1], ylim[1], zf), (xlim[0], ylim[1], zf)]]
    ax.add_collection3d(Poly3DCollection(floor, facecolor=color, edgecolor="none",
                                         alpha=0.7, zsort="min", zorder=0))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_aspect("equal")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
    ax.zaxis.line.set_visible(False)
    ax.set_zticks([])
    ax.set_zlabel("")
    ax.tick_params(colors=INK)
    return zf


# ----------------------------------------------------------------------------- colormaps
def register_managua_red():
    """Register "managua_red": managua with its light end's hue shifted -41 deg (gold -> red),
    the shift tapering linearly to zero at the center; the dark middle and the blue half are
    unchanged, as are saturation and value. Returns the name."""
    base = matplotlib.colormaps["managua"](np.linspace(0.0, 1.0, 256))
    hsv = mcolors.rgb_to_hsv(base[:, :3])
    t = np.linspace(0.0, 1.0, 256)
    w = np.clip(1.0 - t / 0.5, 0.0, 1.0)                # 1 at the light end, 0 at the center
    hsv[:, 0] = (hsv[:, 0] - w * (41.0 / 360.0)) % 1.0
    out = base.copy()
    out[:, :3] = mcolors.hsv_to_rgb(hsv)
    matplotlib.colormaps.register(mcolors.ListedColormap(out, name="managua_red"), force=True)
    return "managua_red"


def srgb_to_lab(rgb):
    """sRGB in [0, 1] (..., 3 or 4) -> CIELAB (..., 3), D65 white."""
    rgb = np.asarray(rgb, float)[..., :3]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    """CIELAB (..., 3) -> sRGB in [0, 1] (..., 3), clipped to the gamut in linear RGB."""
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    Minv = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                     [0.0556434, -0.2040259, 1.0572252]])
    lin = np.clip(xyz @ Minv.T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


# control colors (L*, a*, b*) for cmap_bridge
BRIDGE_CTRL = dict(brown=(30.0, 22.0, 18.0), plum=(32.0, 28.0, -28.0), teal=(34.0, -22.0, -14.0),
                   olive=(36.0, -8.0, 30.0))


def cmap_bridge(name, lo, hi, ctrl="brown"):
    """Register `name` with its middle t in (lo, hi) replaced by a CIELAB quadratic Bezier from the
    color at t=lo to the color at t=hi with control point BRIDGE_CTRL[ctrl]. 256 entries.
    Registered and returned as "<name>_br_<ctrl>_<lo>_<hi>" (dots written as "p")."""
    t = np.linspace(0, 1, 256)
    base = matplotlib.colormaps[name]
    lab = srgb_to_lab(base(t))
    mid = (t > lo) & (t < hi)
    P0 = srgb_to_lab(base(lo))[:3]
    P2 = srgb_to_lab(base(hi))[:3]
    P1 = np.asarray(BRIDGE_CTRL[ctrl], float)
    u = ((t[mid] - lo) / (hi - lo))[:, None]
    lab[mid] = (1 - u) ** 2 * P0 + 2 * (1 - u) * u * P1 + u ** 2 * P2
    cols = np.c_[lab_to_srgb(lab), np.ones(len(t))]
    newname = f"{name}_br_{ctrl}_{lo:g}_{hi:g}".replace(".", "p")
    matplotlib.colormaps.register(mcolors.ListedColormap(cols, name=newname), force=True)
    return newname


def cmap_spec(spec):
    """Colormap name for a spec string (see the module docstring); registers derived maps."""
    if "~" in spec:                                   # "name~CTRL:lo-hi"
        name, rest = spec.split("~", 1)
        ctrl, rng = rest.split(":", 1)
        lo, hi = (float(x) for x in rng.split("-"))
        return cmap_bridge(name, lo, hi, ctrl)
    if ":" not in spec:
        return spec
    name, ranges = spec.split(":", 1)                 # "name:lo-hi[,lo-hi...]"
    ts = [np.linspace(float(a), float(b), max(int(256 * abs(float(b) - float(a))), 32))
          for a, b in (r.split("-") for r in ranges.split(","))]
    cols = matplotlib.colormaps[name](np.concatenate(ts))
    newname = f"{name}_" + ranges.replace(",", "_").replace("-", "_").replace(".", "p")
    matplotlib.colormaps.register(mcolors.ListedColormap(cols, name=newname), force=True)
    return newname
