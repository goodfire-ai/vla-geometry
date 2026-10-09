#!/usr/bin/env python
"""Straight object lines (GPU + EGL): the target paths of the curvature panel (Figure 2e).

Per house of lists/position/line_houses.csv (48 houses; arm at its home pose, seed 104729 +
house), the pickup object is moved along three straight lines of 24 waypoints on its support
surface: along the robot's forward axis (line_fwd), along its lateral axis (line_lat), and along
the ray from the hand's natural position through the centre of the valid region (line_ee, the
largest sweep of hand-object distance). The valid region is the support-surface box intersected
with the reach envelope of the object-teleport harvest (forward 0.25-0.75 m, radius <= 0.70 m,
|bearing| <= 30 deg); each line is centred on the region's extent along its axis and shrunk until
every waypoint is inside it (`vlageom.position.object_shape_design`). Per waypoint: restore the
settled state, move the object, settle, keep it when it stays within 2 cm (xy) / 1 cm (z) of the
command on the same support geom, render (cameras at the robot's pose), run the first model query
and encode it with the object-teleport harvest's readout: VLM 11 token groups x 13 layers
(0, 4, 8, 12, 16, 20, 21, 24, 27, 29, 31, 33, 35) and the action expert per token x 11 layers.
A line on which fewer than 60% of the waypoints are free of contacts with other objects, or fewer
than half are kept, is not written (in the paper's run 120 of the 144 lines were written, in 43
houses).

Output: $VLAGEOM_DATA/position/object_lines/shape_{kind}/h{house}/ (acts.npz, manifest.json,
exo_{i:02d}.png; see `vlageom.position.harvest_object_shape`). About 1.6 GPU-hours in all
(8 jobs of 6 houses).

Work split: --shard K --houses-per-shard 6 (K defaults to $SLURM_ARRAY_TASK_ID); --houses H ...
"""
from __future__ import annotations

import vlageom.sim  # noqa: F401  (first: sets up the file locks before molmo_spaces is imported)

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

from vlageom import paths
from vlageom.position import (OBJECT_LINE_KINDS, harvest_object_shape, line_house_list_path,
                              object_lines_root, read_house_list)
from vlageom.sim import (AE_LAYERS, LLM_LAYERS, bench_engine, build_house_list, encode_query,
                         sample_bench_episode)

log = logging.getLogger("object_lines")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default $VLAGEOM_DATA/position/object_lines)")
    ap.add_argument("--house-list", type=Path, default=line_house_list_path())
    ap.add_argument("--houses", type=int, nargs="+", default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--houses-per-shard", type=int, default=6)
    ap.add_argument("--shapes", nargs="+", choices=OBJECT_LINE_KINDS,
                    default=list(OBJECT_LINE_KINDS))
    ap.add_argument("--n-lines", type=int, default=24, help="waypoints per line")
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
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.llm_layers, args.ae_layers = list(LLM_LAYERS), list(AE_LAYERS)

    out_root = args.out or object_lines_root()
    listed = read_house_list(args.house_list)
    if args.houses:
        bad = [h for h in args.houses if h not in {r[0] for r in listed}]
        if bad:
            raise SystemExit(f"houses not in {args.house_list}: {bad}")
        listed = [r for r in listed if r[0] in set(args.houses)]
    else:
        shard = args.shard
        if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID", "") != "":
            shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if shard is not None:
            listed = listed[shard * args.houses_per_shard:(shard + 1) * args.houses_per_shard]
    if not listed:
        log.info("nothing to do")
        return
    roots = paths.bench_roots()
    roots = {"mini": args.bench_mini or roots["mini"], "hard": args.bench_hard or roots["hard"]}
    catalog = {h: (b, i, sp) for h, b, i, sp in build_house_list(roots, 950)}
    mine = []
    for house, bench, bench_idx in listed:
        if house not in catalog or catalog[house][:2] != (bench, bench_idx):
            raise SystemExit(f"h{house}: benchmark episode {bench}#{bench_idx} not found")
        mine.append((house, bench, catalog[house][2]))
    log.info("%d houses: %s", len(mine), [h for h, *_ in mine])
    if args.dry_run:
        return

    def encode(eng, kw, seed):
        enc = encode_query(eng, kw, args.llm_layers, args.ae_layers, seed)
        return None if enc is None else (enc[0], enc[1])

    eng, sim_envs = bench_engine(args.checkpoint, str(out_root / "_env"),
                                 need_mini=any(b == "mini" for _h, b, _s in mine))
    n_ok, n_tried = 0, 0
    try:
        for house, bench, spec in mine:
            for kind in args.shapes:
                kdir = out_root / f"shape_{kind}"
                kdir.mkdir(parents=True, exist_ok=True)
                if args.skip_existing and (kdir / f"h{house}" / "manifest.json").exists():
                    log.info("h%d %s exists, skipped", house, kind)
                    n_ok += 1
                    continue
                n_tried += 1
                try:
                    sample_bench_episode(eng, sim_envs, bench, spec, seed=104729 + house,
                                         task_horizon=16)
                    n_ok += int(harvest_object_shape(eng, spec, bench, args, kdir, kind,
                                                     args.n_lines, encode))
                except Exception:
                    log.exception("h%d %s failed; continuing", house, kind)
                finally:
                    eng.task = eng.policy = None
                    gc.collect()
    finally:
        eng.close()
    log.info("done: %d/%d (house, line) jobs written", n_ok, len(mine) * len(args.shapes))
    if n_tried and n_ok == 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
