#!/usr/bin/env python
"""Object paths of the target trace grid (GPU + EGL): Figure 2, bottom row.

In one house (default 44, the house left out of the grid's subspace fit; arm at its home pose,
seed 104729 + house), the pickup object is moved along five paths on its support surface
(`vlageom.position.object_shape_design`, inside the support-surface box intersected with the reach
envelope of the object-teleport harvest): a figure-8 (x = a sin t, y = b sin 2t in the valid
region's principal axes, 64 waypoints), an x line (along the robot's forward axis, 24 waypoints),
a y line (along its lateral axis, 24), an ellipse (64) and a square (64, corners exact). Per
waypoint: restore the settled state, move the object, settle, keep it when it stays within 2 cm
(xy) / 1 cm (z) of the command on the same support geom, render (cameras at the robot's pose),
reset the policy, run the first model query and encode it: VLM 11 token groups x 36 layers and
the action expert per token x 36 blocks, as the crossed harvest.

After a path is written its saved exo frames are checked (`vlageom.position.exo_check`):
waypoints with a dark frame (mean pixel value < 15) are removed from acts.npz and the manifest;
a path whose frames do not change from waypoint to waypoint is removed (its manifest is renamed
manifest.unused.json) and the process ends with status 3, so the job can be run again on another
node. The result of the check is written to _exo_check.json.

Output: $VLAGEOM_DATA/position/shapes/obj_{kind}/h{house}/ (acts.npz, manifest.json,
exo_{i:02d}.png, _exo_check.json). About 7 min on one H200.
"""
from __future__ import annotations

import vlageom.sim  # noqa: F401  (first: sets up the file locks before molmo_spaces is imported)

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.position import (N_LAYERS, SHAPE_HOUSE, crossed_house_list_path, encode_query_all,
                              exo_check, harvest_object_shape, read_house_list, save_npz,
                              shapes_root)
from vlageom.sim import bench_engine, build_house_list, sample_bench_episode

log = logging.getLogger("object_shapes")
KINDS = ["infinity", "line_fwd", "line_lat", "ellipse", "square"]


def apply_exo_check(out, dark_px, static_frac) -> bool:
    """Check the written path in `out`; drop dark waypoints; False (path removed) when static."""
    qc = exo_check(out, dark_px, static_frac)
    if qc["exo_static"]:
        os.replace(out / "manifest.json", out / "manifest.unused.json")
        (out / "_exo_check.json").write_text(json.dumps(qc, indent=1))
        log.error("%s: the exo frame does not change across the path; manifest renamed", out)
        return False
    if qc["dark"]:
        man = json.loads((out / "manifest.json").read_text())
        keep = np.setdiff1d(np.arange(len(man["points"])), qc["dark"])
        az = dict(np.load(out / "acts.npz"))
        n = len(man["points"])
        for k, v in az.items():
            if getattr(v, "ndim", 0) >= 1 and v.shape[0] == n and \
                    k not in ("layers_llm", "layers_ae", "variants"):
                az[k] = v[keep]
        save_npz(out / "acts.npz", az, compressed=True)
        man["points"] = [man["points"][i] for i in keep]
        man.setdefault("audit", {})["dropped_dark"] = len(qc["dark"])
        (out / "manifest.json.tmp").write_text(json.dumps(man, indent=1))
        os.replace(out / "manifest.json.tmp", out / "manifest.json")
        log.warning("%s: removed %d waypoints with a dark exo frame", out, len(qc["dark"]))
    (out / "_exo_check.json").write_text(json.dumps(qc, indent=1))
    return True


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--out", type=Path, default=None,
                    help="shapes root (default $VLAGEOM_DATA/position/shapes); "
                         "obj_{kind}/h{house}/ below it")
    ap.add_argument("--house", type=int, default=SHAPE_HOUSE)
    ap.add_argument("--house-list", type=Path, default=crossed_house_list_path())
    ap.add_argument("--shapes", nargs="+", choices=KINDS, default=KINDS)
    ap.add_argument("--n-lines", type=int, default=24)
    ap.add_argument("--n-closed", type=int, default=64)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--bench-mini", default=None)
    ap.add_argument("--bench-hard", default=None)
    ap.add_argument("--fwd-lo", type=float, default=0.25)
    ap.add_argument("--fwd-hi", type=float, default=0.75)
    ap.add_argument("--r-hi", type=float, default=0.70)
    ap.add_argument("--bearing-deg", type=float, default=30.0)
    ap.add_argument("--fallback-patch", type=float, default=0.15)
    ap.add_argument("--xy-tol", type=float, default=0.02)
    ap.add_argument("--z-tol", type=float, default=0.01)
    ap.add_argument("--settle-max", type=int, default=500)
    ap.add_argument("--settle-window", type=int, default=50)
    ap.add_argument("--settle-tol", type=float, default=1e-3)
    ap.add_argument("--dark-px", type=float, default=15.0)
    ap.add_argument("--static-frac", type=float, default=1e-3)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()
    args.llm_layers = list(range(N_LAYERS))
    args.ae_layers = list(range(N_LAYERS))

    listed = {h: (b, i) for h, b, i in read_house_list(args.house_list)}
    if args.house not in listed:
        raise SystemExit(f"house {args.house} is not in {args.house_list}")
    roots = paths.bench_roots()
    roots = {"mini": args.bench_mini or roots["mini"], "hard": args.bench_hard or roots["hard"]}
    catalog = {h: (b, i, sp) for h, b, i, sp in build_house_list(roots, 950)}
    bench, bench_idx, spec = catalog[args.house]
    if (bench, bench_idx) != listed[args.house]:
        raise SystemExit(f"h{args.house}: benchmark episode {bench}#{bench_idx} != list "
                         f"{listed[args.house]}")

    def encode(eng, kw, seed):
        return encode_query_all(eng, kw, args.llm_layers, args.ae_layers, seed)

    out_base = args.out or shapes_root()
    eng, sim_envs = bench_engine(args.checkpoint, str(out_base / "_env"), need_mini=bench == "mini")
    n_ok, n_tried = 0, 0
    try:
        for kind in args.shapes:
            out_root = out_base / f"obj_{kind}"
            out_root.mkdir(parents=True, exist_ok=True)
            hdir = out_root / f"h{args.house}"
            if args.skip_existing and (hdir / "_exo_check.json").exists() and \
                    (hdir / "manifest.json").exists():
                log.info("h%d obj %s exists, skipped", args.house, kind)
                continue
            n_tried += 1
            n = args.n_lines if kind.startswith("line") else args.n_closed
            try:
                sample_bench_episode(eng, sim_envs, bench, spec, seed=104729 + args.house,
                                     task_horizon=16)
                ok = harvest_object_shape(eng, spec, bench, args, out_root, kind, n, encode)
                n_ok += int(bool(ok and apply_exo_check(hdir, args.dark_px, args.static_frac)))
            except Exception:
                log.exception("h%d obj %s failed; continuing", args.house, kind)
            finally:
                eng.task = eng.policy = None
                gc.collect()
    finally:
        eng.close()
    log.info("done: %d/%d paths written", n_ok, n_tried)
    if n_tried and n_ok < n_tried:
        sys.exit(3)


if __name__ == "__main__":
    main()
