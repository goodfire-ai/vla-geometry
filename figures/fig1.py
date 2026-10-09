#!/usr/bin/env python
"""Render Figure 1: four 3-D tiles in one row (gripper, wrist, target position, distance).

Input: the panel files under `vlageom.paths.panels_dir()` (default $VLAGEOM_DATA/panels), each
written by the last stage of its tile pipeline:

  gripper.npz   px, py, pz        (n,)   activations near the grasp on PC1-3 of the gripper edits,
                                         centered on the mean activation
                ddx, ddy, ddz     (n,)   the edit vectors on PC1-3
                target            (n,)   +1 for an edit fitted to close the gripper, -1 to open it
                aper              (n,)   gripper action before the edit (-1 open .. +1 closed)
                evr               (3,)   explained-variance ratios of PC1-3 of the edits
  wrist.npz     Z3 (k, 3), roll (k,)     one house's wrist-joint sweep on PC1-3 and its joint
                                         angles (deg)
                nat3 (m, 3), nat_roll (m,)  the sweep points of all houses and their angles
                evr (3,)                 explained-variance ratios of PC1-3
  target.npz    bg (n, 3), rgb (n, 3)    activations in the 3-D object-position read-out and their
                                         colours (object x, y, z mapped to R, G, B)
                sub (k, 3), sub_rgb (k, 3)      one house's object path (placements)
                curve (c, 3), curve_rgb (c, 3)  the smoothed path
                origin (3,)              the zero activation in the same coordinates
                note_r2 ()               mean held-out R^2 of the object position (x, y, z)
  distance.npz  pts (n, 3), d (n,)       per row: metre-range probe read-out, centimetre-range
                                         probe read-out, true distance, each scaled to [0, 1] over
                                         a fixed range; d = true distance (m)
                path (k, 3), path_d (k,) mean of pts per distance bin, and the bins' mean distance
                cross (2,)               15 cm on the two read-out axes (same scaling)
                r2_m (), r2_cm ()        held-out R^2 of the metre-range probe on rows at or above
                                         15 cm, and of the centimetre-range probe on rows below 15 cm

Other keys in the panel files are ignored. The legend notes are formatted from note_r2, r2_m,
r2_cm, the range of the wrist `roll` array and the shape of the wrist trace `Z3`.

Output: <out-dir>/<name>.png and .pdf (default `vlageom.paths.figures_dir()`, i.e. $VLAGEOM_FIGURES
or $VLAGEOM_DATA/figures, and name "fig1"), 25.2 x 9.15 in at 350 dpi.

Layout. Each tile is a 3-D box with a square floor. All four boxes have the same height,
BOX_HEIGHT (in units of half the floor width), so the four floors render at the same size.
BOX_HEIGHT and the gripper tile's floor (GRIPPER_FLOOR) are fixed layout constants of the figure;
the wrist, target and distance floors are fitted to their panel data. Positions and sizes are set
in inches, so the figure does not depend on the screen or the backend.

Fonts. The style (`vlageom.plot.style.apply_style`) uses Product Sans when it is found in
$VLAGEOM_FONT_DIR or the system fonts, otherwise DejaVu Sans; the titles use the bold face.

Run:  python figures/fig1.py [--panels DIR] [--out-dir DIR] [--name fig1]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from vlageom import distance as tile_distance  # noqa: E402
from vlageom import paths  # noqa: E402
from vlageom import target as tile_target  # noqa: E402
from vlageom import wrist as tile_wrist  # noqa: E402
from vlageom.plot import style as ps  # noqa: E402

TILES = ("gripper", "wrist", "target", "distance")
log = logging.getLogger("fig1")

# ----------------------------------------------------------------------------- settings
DPI = 350
FONT_SCALE = 1.3          # multiplies every text size of the row (titles, axis labels, legends)
NOTE_Y_IN = 1.20          # legend band: height of the note line / steer legend (in from the bottom)
ORIGIN_Y_IN = 0.27        # legend band: height of the target tile's 'origin' legend

GRIPPER_CMAP = "coolwarm_r"                 # gripper action before the edit, open .. closed
CLOSE_COLOR = ps.GRIPPER_CLOSE              # close-edit arrows
OPEN_COLOR = ps.GRIPPER_OPEN                # open-edit arrows
WRIST_CMAP_SPEC = "vanimo~plum:0.30-0.70"   # vanimo with its dark middle bridged through plum
WRIST_CBAR_LABEL = r"wrist angle $\theta$ (deg)"
DISTANCE_CMAP = "managua"                   # metre-range copy of the distance cloud
DISTANCE_CMAP_CM = "managua_red"            # centimetre-range (mirrored) copy

TITLES = dict(gripper="Gripper", wrist="Wrist\nOrientation", target="Target Object\nPosition",
              distance="Distance to\nTarget Object")

# target tile: mark sizes and colours
TARGET_KW = dict(elev=45, azim=30, cloud_size=14, cloud_alpha=1.1, shadow_color="0.58",
                 shadow_alpha=0.8, line_width=3.2, point_size=64, origin_size=36, halo=True,
                 clip_y_in=7.6)
TARGET_CLOUD_VALUE_FLOOR = 0.55     # cloud colours: HSV value raised to at least this
TARGET_TRACE_SV = (0.9, 0.80)       # path colours: HSV saturation and value set to these

# distance tile
DISTANCE_AZIM = 120
DISTANCE_ZSCALE = 0.7               # true-distance axis drawn at this fraction of the unit range
DIST_NORMS = dict(m=(0.15, 0.70), cm=(0.0, 0.15))   # colour ranges (m) of the two copies

# ----------------------------------------------------------------------------- layout
FIG_W, BAND_IN = 25.2, 1.55         # figure width; legend band below the tiles
FIG_H = 7.6 + BAND_IN               # 7.0 in of tiles + 0.6 in of titles + the band
TITLE_Y_IN = 7.15 + BAND_IN         # first title line
SLOT_X = (-0.0639, 0.1754, 0.4147, 0.6540)   # left edge of each tile's axes, in units of 25.2 in
SLOT_CX = (0.1411, 0.3804, 0.6196, 0.8589)   # centre of each tile's floor (titles, legend band)

# Fixed layout constants: the shared box height (in units of half the floor width) and the gripper
# tile's floor (x / y limits, half width, floor and shadow heights, in edit-PC units).
BOX_HEIGHT = np.float32(2.1065874099731445)
GRIPPER_FLOOR = dict(
    xlo=np.float32(-5.843289375305176), xhi=np.float32(6.568027496337891),
    ylo=np.float32(-6.042619705200195), yhi=np.float32(6.368697166442871),
    half=np.float32(6.205658435821533), zfloor=-6.46465528352356,
    shadow_z=-6.3364909456159975, H=BOX_HEIGHT)


def _fs(pt):
    return pt * FONT_SCALE


# ----------------------------------------------------------------------------- inputs
REQUIRED_KEYS = dict(
    gripper=("px", "py", "pz", "ddx", "ddy", "ddz", "target", "aper", "evr"),
    wrist=("Z3", "roll", "nat3", "nat_roll", "evr"),
    target=("bg", "rgb", "sub", "sub_rgb", "curve", "curve_rgb", "origin"),
    distance=("pts", "d", "path", "path_d", "cross"))
# legend-note values: name -> (panel file, array key)
NOTE_SOURCES = dict(target_r2=("target", "note_r2"), distance_r2_m=("distance", "r2_m"),
                    distance_r2_cm=("distance", "r2_cm"))


def load_panels(panels_dir: Path) -> dict:
    """{tile: {key: array}} from <panels_dir>/<tile>.npz; raises if a file or key is missing."""
    panels = {}
    for tile in TILES:
        f = panels_dir / f"{tile}.npz"
        if not f.exists():
            raise SystemExit(f"missing panel file {f} (run the last stage of pipelines/{tile}/)")
        d = dict(np.load(f, allow_pickle=False))
        missing = [k for k in REQUIRED_KEYS[tile] if k not in d]
        if missing:
            raise SystemExit(f"{f}: missing arrays {missing}")
        panels[tile] = d
    return panels


def load_notes(panels: dict) -> dict:
    """The legend-note values, read from the panel files."""
    missing = [f"{tile}.npz:{key}" for tile, key in NOTE_SOURCES.values()
               if key not in panels[tile]]
    if missing:
        raise SystemExit(f"panel files without {missing}: run the last stage of each tile pipeline")
    notes = {k: float(panels[tile][key]) for k, (tile, key) in NOTE_SOURCES.items()}
    notes["wrist_roll"] = np.asarray(panels["wrist"]["roll"])
    notes["wrist_trace"] = np.asarray(panels["wrist"]["Z3"])
    return notes


def note_strings(notes: dict) -> dict:
    """Legend-band text of the wrist, target and distance tiles (formatted by the tile modules)."""
    return dict(
        wrist=tile_wrist.sweep_note(notes["wrist_roll"], notes["wrist_trace"]),
        target=tile_target.r2_note(notes["target_r2"]),
        distance=tile_distance.r2_note(float(notes["distance_r2_m"]),
                                       float(notes["distance_r2_cm"])))


def adjust_panels(panels: dict) -> None:
    """Display adjustments made after loading (in place): target cloud colours get an HSV value
    floor, the target path colours a fixed saturation and value (hue kept), and the distance
    tile's true-distance axis is scaled by DISTANCE_ZSCALE."""
    p = panels["target"]
    hsv = mcolors.rgb_to_hsv(np.clip(np.asarray(p["rgb"], float), 0, 1))
    hsv[:, 2] = np.maximum(hsv[:, 2], TARGET_CLOUD_VALUE_FLOOR)
    p["rgb"] = mcolors.hsv_to_rgb(hsv)
    s_, v_ = TARGET_TRACE_SV
    for k in ("sub_rgb", "curve_rgb"):
        hsv = mcolors.rgb_to_hsv(np.clip(np.asarray(p[k], float), 0, 1))
        hsv[:, 1], hsv[:, 2] = s_, v_
        p[k] = mcolors.hsv_to_rgb(hsv)
    d = panels["distance"]
    d["pts"][:, 2] *= DISTANCE_ZSCALE
    d["path"][:, 2] *= DISTANCE_ZSCALE


# ----------------------------------------------------------------------------- frames
def _frame(pts, pcts=(1, 99), zmin_frac=0.0, extra=None):
    """3-D box for a point set: square floor over the pcts-percentile range of axes 0/1 (plus 8%
    padding), height from the range of axis 2. `extra` points are always inside. H = box height
    in units of half the floor width."""
    lo = np.percentile(pts, pcts[0], axis=0)
    hi = np.percentile(pts, pcts[1], axis=0)
    if extra is not None:
        lo, hi = np.minimum(lo, extra.min(0)), np.maximum(hi, extra.max(0))
    xpad = (hi[0] - lo[0]) * 0.08 + 1e-6
    ypad = (hi[1] - lo[1]) * 0.08 + 1e-6
    xlo, xhi, ylo, yhi = lo[0] - xpad, hi[0] + xpad, lo[1] - ypad, hi[1] + ypad
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    half = max(xhi - xlo, yhi - ylo) / 2
    xlo, xhi, ylo, yhi = cx - half, cx + half, cy - half, cy + half
    zlo, zhi = float(lo[2]), float(hi[2])
    if zmin_frac and (zhi - zlo) < zmin_frac * 2 * half:   # flat data: keep a minimum height
        cz = (zlo + zhi) / 2
        zlo, zhi = cz - zmin_frac * half, cz + zmin_frac * half
    zfloor = zlo - (zhi - zlo) * 0.06 - 1e-6
    ztop = zhi + (zhi - zfloor) * 0.02
    return dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, half=half, zfloor=zfloor,
                shadow_z=zfloor + (zhi - zfloor) * 0.01, H=(ztop - zfloor) / half)


def frame_wrist(p):
    return _frame(p["nat3"], pcts=(1, 99), zmin_frac=0.35, extra=p["Z3"])


def frame_target(p):
    return _frame(p["bg"], pcts=(1, 99), extra=p["sub"])


def frame_distance(p):
    """Box whose floor is the plane of zero true distance (z = 0)."""
    allp = np.vstack([p["pts"], p["path"]])
    lo = np.minimum(np.percentile(allp, 1, axis=0), p["path"].min(0))
    hi = np.maximum(np.percentile(allp, 99, axis=0), p["path"].max(0))
    xpad = (hi[0] - lo[0]) * 0.08 + 1e-6
    ypad = (hi[1] - lo[1]) * 0.08 + 1e-6
    xlo, xhi, ylo, yhi = lo[0] - xpad, hi[0] + xpad, lo[1] - ypad, hi[1] + ypad
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    half = max(xhi - xlo, yhi - ylo) / 2
    xlo, xhi, ylo, yhi = cx - half, cx + half, cy - half, cy + half
    ztop = float(hi[2]) * 1.02
    return dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, half=half, zfloor=0.0,
                shadow_z=ztop * 0.01, H=ztop / half)


def _apply_frame(ax, fr, box_h):
    """Draw the floor, set the limits and a (2, 2, box_h) box aspect. Returns the shadow height."""
    ax.set_facecolor("none")
    ax.computed_zorder = False
    xlo, xhi, ylo, yhi, zfloor = fr["xlo"], fr["xhi"], fr["ylo"], fr["yhi"], fr["zfloor"]
    H = box_h
    ax.add_collection3d(Poly3DCollection(
        [[[xlo, ylo, zfloor], [xhi, ylo, zfloor], [xhi, yhi, zfloor], [xlo, yhi, zfloor]]],
        facecolor="#F8FAFC", alpha=0.92, edgecolor="none", zorder=0))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zfloor, zfloor + H * fr["half"])
    ax.set_box_aspect((2, 2, H))
    return fr["shadow_z"]


# ----------------------------------------------------------------------------- 3-D helpers
def _camera(elev, azim):
    e_, a_ = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_), np.sin(e_)])


def _depth_alpha(pts, elev, azim, base_a):
    """Per-point alpha that grows toward the camera (the view of view_init(elev, azim))."""
    depth = (pts - pts.mean(0)) @ _camera(elev, azim)
    lo, hi = np.percentile(depth, [2, 98])
    dn = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1)
    return base_a * (0.12 + (2.3 - 0.12) * dn)


def _binned_field3(pts, vecs, sel, nbins=8, min_count=4):
    """Mean position and mean vector of the selected rows per cell of an nbins^3 grid spanning all
    rows; cells with fewer than min_count rows are left out. Returns (rows (r, 6), cell width)."""
    edges = [np.linspace(pts[:, i].min(), pts[:, i].max(), nbins + 1) for i in range(3)]
    idx = [np.clip(np.digitize(pts[:, i], edges[i]) - 1, 0, nbins - 1) for i in range(3)]
    rows = []
    for i in range(nbins):
        for j in range(nbins):
            for k in range(nbins):
                m = sel & (idx[0] == i) & (idx[1] == j) & (idx[2] == k)
                if int(m.sum()) >= min_count:
                    rows.append(np.concatenate([pts[m].mean(0), vecs[m].mean(0)]))
    return np.asarray(rows), float(edges[0][1] - edges[0][0])


def _axis_labels3d(ax, labels):
    """Floor-edge axis lines in ink with the x/y labels; no panes, ticks or vertical axis."""
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.set_ticks([])
        axis.line.set_color(ps.INK)
        axis.line.set_linewidth(1.1)
    ax.zaxis.line.set_visible(False)
    ax.set_xlabel(labels[0], fontsize=_fs(21), labelpad=-4, color=ps.INK)
    ax.set_ylabel(labels[1], fontsize=_fs(21), labelpad=-4, color=ps.INK)


def _mirror_edge_labels(ax, fr, labels, off=0.048):
    """Label the back pair of floor edges (the axes of the mirrored copy in the distance tile),
    placed and rotated in 2-D from the projected edge. Call after view_init and the limits."""
    from mpl_toolkits.mplot3d import proj3d

    M = ax.get_proj()
    xlo, xhi, ylo, yhi, z = fr["xlo"], fr["xhi"], fr["ylo"], fr["yhi"], fr["zfloor"]
    cam = _camera(ax.elev, ax.azim)

    def to_ax(pt):
        x2, y2, _ = proj3d.proj_transform(pt[0], pt[1], pt[2], M)
        return ax.transAxes.inverted().transform(ax.transData.transform((x2, y2)))

    center = to_ax(((xlo + xhi) / 2, (ylo + yhi) / 2, z))
    edge_pairs = ((((xlo, ylo, z), (xhi, ylo, z)), ((xlo, yhi, z), (xhi, yhi, z))),
                  (((xlo, ylo, z), (xlo, yhi, z)), ((xhi, ylo, z), (xhi, yhi, z))))
    for (ea, eb), txt in zip(edge_pairs, labels):
        pick = min((ea, eb), key=lambda e: (np.mean(e, axis=0) @ cam))   # farther from the camera
        p0, p1 = to_ax(pick[0]), to_ax(pick[1])
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        if ang > 90 or ang <= -90:
            ang += 180 if ang <= -90 else -180
        mid = (p0 + p1) / 2
        out = mid - center
        out /= np.linalg.norm(out) + 1e-12
        ax.text2D(mid[0] + out[0] * off, mid[1] + out[1] * off, txt,
                  transform=ax.transAxes, rotation=ang, rotation_mode="anchor",
                  ha="center", va="center", fontsize=_fs(21), color=ps.INK)


# ----------------------------------------------------------------------------- tiles
def draw_gripper(ax, p, fr, box_h, labels, elev=45, azim=-60, cloud_size=10, cloud_alpha=1.5):
    """Activation cloud coloured by the gripper action before the edit, with the binned mean edit
    vectors of close edits (blue) and open edits (red) on a 3-D grid."""
    pts = np.stack([p["px"], p["py"], p["pz"]], 1)
    vecs = np.stack([p["ddx"], p["ddy"], p["ddz"]], 1)
    tgt = p["target"]
    shadow_z = _apply_frame(ax, fr, box_h)

    ax.scatter(pts[:, 0], pts[:, 1], np.full(len(pts), shadow_z), c=ps.INK, s=5,
               alpha=0.05, depthshade=False, edgecolors="none", rasterized=True, zorder=2)
    base = matplotlib.colormaps[GRIPPER_CMAP](np.clip((p["aper"] + 1) / 2, 0, 1))[:, :3]
    rgba = np.clip(np.concatenate(
        [base, _depth_alpha(pts, elev, azim, 0.30 * cloud_alpha)[:, None]], 1), 0, 1)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgba, s=cloud_size, depthshade=False,
               edgecolors="none", rasterized=True, zorder=4)

    fc3, cellw = _binned_field3(pts, vecs, tgt > 0)
    fo3, _ = _binned_field3(pts, vecs, tgt < 0)
    mags = np.concatenate([np.linalg.norm(fc3[:, 3:], axis=1), np.linalg.norm(fo3[:, 3:], axis=1)])
    scale = (0.75 * cellw) / (np.median(mags) + 1e-9)
    cap = 2.0 * cellw                                    # length cap for sparse cells
    for f, col in ((fc3, CLOSE_COLOR), (fo3, OPEN_COLOR)):
        v = f[:, 3:] * scale
        m = np.linalg.norm(v, axis=1)
        v *= np.minimum(1.0, cap / (m + 1e-12))[:, None]
        ax.quiver(f[:, 0], f[:, 1], f[:, 2], v[:, 0], v[:, 1], v[:, 2],
                  color=col, linewidths=1.7, arrow_length_ratio=0.3, alpha=0.9, zorder=6)
    ax.view_init(elev=elev, azim=azim)
    _axis_labels3d(ax, labels)


def draw_wrist(ax, p, fr, box_h, labels, cmap, elev=45, azim=-60, point_size=80,
               cloud_size=10, cloud_alpha=0.35, cloud_n=3600):
    """All houses' sweep points (thin, flat alpha) and one house's sweep as a colour-graded path
    with beads, coloured by joint angle on the highlighted sweep's range; floor shadow in grey."""
    Z3, roll = p["Z3"], p["roll"]
    nat, nroll = p["nat3"], p["nat_roll"]
    shadow_z = _apply_frame(ax, fr, box_h)
    ax.plot(Z3[:, 0], Z3[:, 1], np.full(len(Z3), shadow_z), color="0.55", lw=1.6,
            alpha=0.9, zorder=2)
    ax.scatter(Z3[:, 0], Z3[:, 1], np.full(len(Z3), shadow_z), c="0.55", s=point_size * 0.4,
               depthshade=False, edgecolors="none", alpha=0.9, zorder=2)
    if cloud_n and cloud_n < len(nat):
        sel = np.random.default_rng(0).choice(len(nat), cloud_n, replace=False)
        nat, nroll = nat[sel], nroll[sel]
    norm = mcolors.Normalize(vmin=float(roll.min()), vmax=float(roll.max()))
    base = matplotlib.colormaps[cmap](norm(nroll))[:, :3]
    a = np.full(len(nat), 0.60 * cloud_alpha)
    rgba = np.clip(np.concatenate([base, a[:, None]], 1), 0, 1)
    hal = np.clip(np.concatenate([base, (a * 0.30)[:, None]], 1), 0, 1)
    ax.scatter(nat[:, 0], nat[:, 1], nat[:, 2], c=hal, s=cloud_size * 2.6,
               depthshade=False, edgecolors="none", rasterized=True, zorder=4)
    ax.scatter(nat[:, 0], nat[:, 1], nat[:, 2], c=rgba, s=cloud_size, depthshade=False,
               edgecolors="none", rasterized=True, zorder=4)

    cols = matplotlib.colormaps[cmap](norm(roll))[:, :3]
    segs = np.stack([Z3[:-1], Z3[1:]], 1)
    ax.add_collection3d(Line3DCollection(segs, colors=(cols[:-1] + cols[1:]) / 2,
                                         linewidths=3.2, zorder=5, capstyle="round"))
    ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], c=roll, cmap=cmap, norm=norm,
               s=point_size * 3, depthshade=False, edgecolors="none", alpha=0.20, zorder=6)
    ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], c=roll, cmap=cmap, norm=norm,
               s=point_size * 1.35, depthshade=False, edgecolors="none", alpha=1.0, zorder=6)
    ax.view_init(elev=elev, azim=azim)
    _axis_labels3d(ax, labels)


def draw_target(ax, p, fr, box_h, labels, elev, azim, cloud_size, cloud_alpha, shadow_color,
                shadow_alpha, line_width, point_size, origin_size, halo, clip_y_in):
    """Cloud with depth-faded alpha, the object path (smoothed curve + beads, grey floor shadow)
    and the origin marker. Cloud points whose projected position lies above clip_y_in inches from
    the bottom of the figure are not drawn (they would overlap the titles)."""
    bg, rgb, origin = p["bg"], p["rgb"], p["origin"]
    sub, curve = p["sub"], p["curve"]
    shadow_z = _apply_frame(ax, fr, box_h)

    ax.scatter(sub[:, 0], sub[:, 1], np.full(len(sub), shadow_z), c=shadow_color,
               s=point_size * 0.45, depthshade=False, edgecolors="none",
               alpha=shadow_alpha, zorder=2)
    ax.plot(curve[:, 0], curve[:, 1], np.full(len(curve), shadow_z), color=shadow_color,
            lw=line_width, alpha=shadow_alpha, zorder=2)
    ax.scatter(origin[0], origin[1], shadow_z, c=shadow_color, s=origin_size * 1.3,
               depthshade=False, edgecolors="none", alpha=shadow_alpha, zorder=3)

    depth = (bg - bg.mean(0)) @ _camera(elev, azim)
    lo, hi = np.percentile(depth, [2, 98])
    dn = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1)
    afac = 0.12 + (2.3 - 0.12) * dn
    rgba = np.clip(np.concatenate([rgb, (0.30 * cloud_alpha * afac)[:, None]], 1), 0, 1)
    if clip_y_in is not None:
        from mpl_toolkits.mplot3d import proj3d
        ax.view_init(elev=elev, azim=azim)          # projection as drawn (set again below)
        x2, y2, _ = proj3d.proj_transform(bg[:, 0], bg[:, 1], bg[:, 2], ax.get_proj())
        y_page_in = ax.transData.transform(np.c_[x2, y2])[:, 1] / ax.figure.dpi
        keep = y_page_in <= clip_y_in
        print(f"[fig1] target: {int((~keep).sum())} of {len(bg)} cloud points lie above "
              f"{clip_y_in} in and are not drawn", flush=True)
        bg, rgba = bg[keep], rgba[keep]
    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], c=rgba, s=cloud_size, depthshade=False,
               edgecolors="none", rasterized=True, zorder=4)

    segs = np.stack([curve[:-1], curve[1:]], 1)
    seg_col = (p["curve_rgb"][:-1] + p["curve_rgb"][1:]) / 2
    ax.add_collection3d(Line3DCollection(segs, colors=seg_col, linewidths=line_width, zorder=6))
    if halo:
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=p["sub_rgb"], s=point_size * 3,
                   depthshade=False, edgecolors="none", alpha=0.20, zorder=7)
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=p["sub_rgb"], s=point_size * 1.35,
                   depthshade=False, edgecolors="none", alpha=1.0, zorder=7)
    else:
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=p["sub_rgb"], s=point_size,
                   depthshade=False, edgecolors="none", zorder=7)
    ax.scatter(origin[0], origin[1], origin[2], s=origin_size, c="white", depthshade=False,
               edgecolors="black", linewidths=0.7, zorder=10)
    ax.view_init(elev=elev, azim=azim)
    _axis_labels3d(ax, labels)


def draw_distance(ax, p, fr, box_h, labels, cmap, cmap_cm, elev=45, azim=DISTANCE_AZIM,
                  cloud_size=7.0, cloud_alpha=0.85, cloud_n=8000, point_size=64):
    """Two copies of the read-out cloud and of the per-bin mean path: the first coloured on the
    metre range (DIST_NORMS['m']) and read against the front floor edges, the second rotated
    180 degrees about the vertical axis through the floor centre, coloured on the centimetre range
    and read against the back edges. Height is the true distance; the floor is zero distance.
    Dashed lines on the floor mark 15 cm on both read-out axes."""
    from scipy.interpolate import splev, splprep

    pts, d, path = p["pts"], p["d"], p["path"]
    if cloud_n and cloud_n < len(pts):
        sel = np.random.default_rng(0).choice(len(pts), cloud_n, replace=False)
        pts, d = pts[sel], d[sel]
    shadow_z = _apply_frame(ax, fr, box_h)
    xlo, xhi, ylo, yhi = fr["xlo"], fr["xhi"], fr["ylo"], fr["yhi"]

    def mir(q):
        out = np.array(q, float)
        out[:, 0] = (xlo + xhi) - out[:, 0]
        out[:, 1] = (ylo + yhi) - out[:, 1]
        return out

    clouds = [(pts, "m", cmap), (mir(pts), "cm", cmap_cm)]
    paths = [(path, "m", cmap), (mir(path), "cm", cmap_cm)]
    crosses = [(float(p["cross"][0]), float(p["cross"][1]))]
    crosses.append(((xlo + xhi) - crosses[0][0], (ylo + yhi) - crosses[0][1]))

    for cxx, cyy in crosses:
        ax.plot([cxx, cxx], [ylo, yhi], [shadow_z, shadow_z], color=ps.META_LINE,
                ls="--", lw=1.2, zorder=1)
        ax.plot([xlo, xhi], [cyy, cyy], [shadow_z, shadow_z], color=ps.META_LINE,
                ls="--", lw=1.2, zorder=1)

    # both copies as one cloud, drawn far-to-near so they overlap correctly
    P = np.vstack([c for c, _, _ in clouds])
    base = np.vstack([matplotlib.colormaps[cm_](
        mcolors.Normalize(*DIST_NORMS[r])(d))[:, :3] for _, r, cm_ in clouds])
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), shadow_z), c=ps.INK, s=5,
               alpha=0.05, depthshade=False, edgecolors="none", rasterized=True, zorder=2)
    a = _depth_alpha(P, elev, azim, 0.30 * cloud_alpha)
    order = np.argsort((P - P.mean(0)) @ _camera(elev, azim))
    rgba = np.clip(np.concatenate([base, a[:, None]], 1), 0, 1)[order]
    ax.scatter(P[order, 0], P[order, 1], P[order, 2], c=rgba, s=cloud_size,
               depthshade=False, edgecolors="none", rasterized=True, zorder=4)

    for pp, r, cm_ in paths:            # interpolating spline + beads, grey floor shadow
        norm = mcolors.Normalize(*DIST_NORMS[r])
        tck, upar = splprep([pp[:, 0], pp[:, 1], pp[:, 2]], s=0.0, k=3)
        uu = np.linspace(0.0, 1.0, 400)
        cur = np.array(splev(uu, tck)).T
        pcol = matplotlib.colormaps[cm_](norm(p["path_d"]))[:, :3]
        ccol = np.clip(np.stack([np.interp(uu, np.asarray(upar), pcol[:, c])
                                 for c in range(3)], 1), 0, 1)
        ax.scatter(pp[:, 0], pp[:, 1], np.full(len(pp), shadow_z), c="0.55",
                   s=point_size * 0.45, depthshade=False, edgecolors="none", alpha=0.8,
                   zorder=2)
        ax.plot(cur[:, 0], cur[:, 1], np.full(len(cur), shadow_z), color="0.55", lw=3.2,
                alpha=0.8, zorder=2)
        segs = np.stack([cur[:-1], cur[1:]], 1)
        ax.add_collection3d(Line3DCollection(segs, colors=(ccol[:-1] + ccol[1:]) / 2,
                                             linewidths=3.2, zorder=6, capstyle="round"))
        ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2], c=pcol, s=point_size * 3,
                   depthshade=False, edgecolors="none", alpha=0.20, zorder=7)
        ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2], c=pcol, s=point_size * 1.35,
                   depthshade=False, edgecolors="none", alpha=1.0, zorder=7)
    ax.view_init(elev=elev, azim=azim)
    _axis_labels3d(ax, labels)
    _mirror_edge_labels(ax, fr, labels)


# ----------------------------------------------------------------------------- legend band
def _hbar(fig, cx, y_in, mappable, ticks, label, ticklabels=None, bar_w=3.2, bar_h=0.15,
          extend=None):
    cax = fig.add_axes([cx - bar_w / 2 / FIG_W, y_in / FIG_H, bar_w / FIG_W, bar_h / FIG_H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks,
                      **(dict(extend=extend) if extend else {}))
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cb.set_label(label, fontsize=_fs(17), color=ps.INK)
    cb.ax.tick_params(labelsize=_fs(15), colors=ps.INK)
    cb.outline.set_visible(False)
    return cb


def _note(fig, cx, txt):
    fig.text(cx, NOTE_Y_IN / FIG_H, txt, ha="center", va="center", fontsize=_fs(18), color=ps.INK)


def band_gripper(fig, cx):
    """Close/open edit legend and the open..closed colour bar."""
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color=CLOSE_COLOR, lw=3.6, label="steer close"),
                        Line2D([0], [0], color=OPEN_COLOR, lw=3.6, label="steer open")],
               loc="center", bbox_to_anchor=(cx, NOTE_Y_IN / FIG_H), ncol=2,
               frameon=True, framealpha=0.9, edgecolor="none", fontsize=_fs(18),
               handlelength=1.6, columnspacing=1.4)
    _hbar(fig, cx, 0.72, cm.ScalarMappable(norm=mcolors.Normalize(-1, 1), cmap=GRIPPER_CMAP),
          [-1, 1], "", ticklabels=["Open", "Closed"])


def band_wrist(fig, cx, wrist, cmap, note):
    """Sweep note and the joint-angle colour bar over the highlighted sweep's range."""
    _note(fig, cx, note)
    vmin, vmax = float(wrist["roll"].min()), float(wrist["roll"].max())
    ticks = [vmin, -90, 0, 90, vmax]
    _hbar(fig, cx, 0.72, cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap),
          ticks, WRIST_CBAR_LABEL, ticklabels=[f"{t:.0f}" for t in ticks])


def band_target(fig, cx, note):
    """R^2 note, the 'x = R  y = G  z = B' colour key and the origin legend."""
    from matplotlib.lines import Line2D
    _note(fig, cx, note)
    for k, (txt, col) in enumerate((("$x$ = R", "#d62728"), ("$y$ = G", "#2ca02c"),
                                    ("$z$ = B", "#1f77b4"))):
        fig.text(cx + (k - 1) * 0.95 * FONT_SCALE / FIG_W, 0.78 / FIG_H, txt, ha="center",
                 va="center", fontsize=_fs(18), color=col)
    fig.legend(handles=[Line2D([0], [0], ls="none", marker="o", mfc="white",
                               mec="black", mew=1.0, ms=9, label="origin")],
               loc="center", bbox_to_anchor=(cx, ORIGIN_Y_IN / FIG_H),
               frameon=True, framealpha=0.9, edgecolor="none", fontsize=_fs(18),
               handlelength=1.0, handletextpad=0.5)


def band_distance(fig, cx, note):
    """R^2 note and the two colour bars: metre range (left) and centimetre range (right)."""
    _note(fig, cx, note)
    for r, cm_, dx, ticks, tl, ext, lab in (
            ("m", DISTANCE_CMAP, -1.45, [0.15, 0.40, 0.70],
             ["0.15", "0.40", "0.70"], "min", "distance (m)"),
            ("cm", DISTANCE_CMAP_CM, 1.45,
             [0.0, 0.05, 0.10, 0.15], ["0", "5", "10", "15"], "max", "distance (cm)")):
        _hbar(fig, cx + dx / FIG_W, 0.72,
              cm.ScalarMappable(norm=mcolors.Normalize(*DIST_NORMS[r]), cmap=cm_),
              ticks, lab, ticklabels=tl, bar_w=2.2, extend=ext)


def _title(fig, cx, text):
    for i, line in enumerate(text.split("\n")):
        fig.text(cx, (TITLE_Y_IN - 0.52 * FONT_SCALE * i) / FIG_H, line, ha="center",
                 va="center", fontsize=_fs(34), color=ps.INK, fontweight="bold")


# ----------------------------------------------------------------------------- figure
def axis_labels(panels):
    gevr, wevr = panels["gripper"]["evr"], panels["wrist"]["evr"]
    return dict(gripper=(f"$\\delta$ PC1  ({gevr[0]:.0%} var)", f"$\\delta$ PC2  ({gevr[1]:.0%})"),
                wrist=(f"PC1  ({wevr[0]:.0%} var)", f"PC2  ({wevr[1]:.0%})"),
                target=("$x$", "$y$"),
                distance=("m ruler", "cm ruler"))


def render(panels: dict, notes: dict):
    """Draw Figure 1 from loaded panels and note values. Returns the matplotlib figure."""
    wrist_cmap = ps.cmap_spec(WRIST_CMAP_SPEC)
    adjust_panels(panels)
    ps.apply_style()
    plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 14})
    ps.register_managua_red()

    frames = dict(gripper=GRIPPER_FLOOR, wrist=frame_wrist(panels["wrist"]),
                  target=frame_target(panels["target"]),
                  distance=frame_distance(panels["distance"]))
    for k, fr in frames.items():
        if fr["H"] > BOX_HEIGHT:
            log.debug("%s: data box height %.4f > BOX_HEIGHT %.4f; points above the box top are "
                      "drawn outside it", k, float(fr["H"]), float(BOX_HEIGHT))
    texts = note_strings(notes)
    labs = axis_labels(panels)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    W3 = 0.4051 * 25.2 / FIG_W
    H3, Y0 = 1.0598 * 7.0 / FIG_H, (0.0066 * 7.0 + BAND_IN) / FIG_H
    xs = [x * 25.2 / FIG_W for x in SLOT_X]
    cxs = [c * 25.2 / FIG_W for c in SLOT_CX]
    for cx, key in zip(cxs, TILES):
        _title(fig, cx, TITLES[key])
    for x, key in zip(xs, TILES):
        ax = fig.add_axes([x, Y0, W3, H3], projection="3d")
        if key == "gripper":
            draw_gripper(ax, panels[key], frames[key], BOX_HEIGHT, labs[key])
        elif key == "wrist":
            draw_wrist(ax, panels[key], frames[key], BOX_HEIGHT, labs[key], wrist_cmap)
        elif key == "target":
            draw_target(ax, panels[key], frames[key], BOX_HEIGHT, labs[key], **TARGET_KW)
        else:
            draw_distance(ax, panels[key], frames[key], BOX_HEIGHT, labs[key],
                          DISTANCE_CMAP, DISTANCE_CMAP_CM)
    band_gripper(fig, cxs[0])
    band_wrist(fig, cxs[1], panels["wrist"], wrist_cmap, texts["wrist"])
    band_target(fig, cxs[2], texts["target"])
    band_distance(fig, cxs[3], texts["distance"])
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Panel files: see the module docstring.")
    ap.add_argument("--panels", type=Path, default=None,
                    help="directory with gripper/wrist/target/distance.npz "
                         "(default: $VLAGEOM_DATA/panels)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: $VLAGEOM_FIGURES, else $VLAGEOM_DATA/figures)")
    ap.add_argument("--name", default="fig1", help="output file stem (default: fig1)")
    args = ap.parse_args(argv)

    panels_dir = args.panels if args.panels is not None else paths.panels_dir()
    out_dir = args.out_dir if args.out_dir is not None else paths.figures_dir()
    panels = load_panels(panels_dir)
    notes = load_notes(panels)
    for k, v in note_strings(notes).items():
        print(f"[fig1] note {k}: {v}", flush=True)

    fig = render(panels, notes)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.name
    # no bbox_inches="tight": it would crop the 3-D axes and change the page size
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, transparent=False)
    fig.savefig(stem.with_suffix(".pdf"), dpi=DPI, transparent=False)
    plt.close(fig)
    print(f"[fig1] wrote {stem}.png and {stem}.pdf", flush=True)


if __name__ == "__main__":
    main()
