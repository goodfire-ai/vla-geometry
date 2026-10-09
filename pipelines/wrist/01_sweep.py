"""Wrist tile, stage 1 (GPU + EGL): wrist roll sweep of one episode.

For one row of lists/wrist_episodes.csv: roll out the episode with MolmoBot-DROID (at most 600
steps), take the first query after query 0 at which the hand is open, and set the wrist roll joint
(arm joint 6) to 38 evenly spaced angles over its limits in the simulator plus the episode's own
angle (39 rows). For each angle the cameras follow the hand, both cameras are rendered and the policy
is queried once with the rollout's flow noise; the action expert's per-token output at the final flow
step is stored for layers 0, 4, 9, 16, 24 and 32.

One episode per process. Output: <data>/wrist/sweeps/ep<catalog_index>.npz (keys:
`vlageom.wrist.SWEEP_KEYS`). An existing output is kept unless --overwrite is given.

Exit status: 0 written or already present, 3 the exo-camera frames are dark (mean pixel value below
--min-px; nothing is written, so the episode can be run again on another node).

  python pipelines/wrist/01_sweep.py --row 0          # rows 0..63 (default $SLURM_ARRAY_TASK_ID)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import vlageom.sim  # noqa: E402,F401  (before anything imports molmo_spaces)
import numpy as np  # noqa: E402

from vlageom import paths, wrist  # noqa: E402

log = logging.getLogger("wrist.01_sweep")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--row", type=int,
                   help="row of lists/wrist_episodes.csv (0-based; default $SLURM_ARRAY_TASK_ID)")
    g.add_argument("--catalog-index", type=int, help="catalog_index of the episode in lists/wrist_episodes.csv")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default <data>/wrist/sweeps)")
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--min-px", type=float, default=wrist.MIN_PX,
                    help="minimum mean exo-camera pixel value over the sweep (default %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rows = wrist.read_episodes()
    if args.row is None and args.catalog_index is None:
        if os.environ.get("SLURM_ARRAY_TASK_ID", "") == "":
            ap.error("give --row or --catalog-index (or run as a SLURM array task)")
        args.row = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.row is not None:
        if not 0 <= args.row < len(rows):
            ap.error(f"--row must be in 0..{len(rows) - 1}")
        ep = rows[args.row]
    else:
        match = [r for r in rows if r["catalog_index"] == args.catalog_index]
        if not match:
            ap.error(f"catalog_index {args.catalog_index} is not in lists/wrist_episodes.csv")
        ep = match[0]
    out = wrist.sweep_path(ep["catalog_index"], args.out_dir)
    if out.exists() and not args.overwrite:
        log.info("%s exists; pass --overwrite to replace it", out)
        return 0
    log.info("catalog_index %d: episode %d, house %d, seed %d -> %s", ep["catalog_index"],
             ep["episode_index"], ep["house"], ep["seed"], out)

    from vlageom.sim import BranchEngine
    eng = BranchEngine({"config": wrist.CONFIG, "checkpoint": args.checkpoint,
                        "output_dir": str(paths.tile_dir(wrist.TILE, "env"))})
    eng.load()
    try:
        sweep = wrist.sweep_episode(eng, ep["episode_index"], ep["house"], ep["seed"])
    finally:
        eng.close()

    frames = [x for x in sweep["frames"] if x is not None]
    px = (float(np.mean([np.asarray(x, np.uint8).mean() for x in frames]))
          if len(frames) == len(sweep["frames"]) else 0.0)
    if px < args.min_px:
        log.error("mean exo-camera pixel value %.1f < %.1f (dark or missing frames): nothing written", px,
                  args.min_px)
        return 3
    wrist.save_sweep(out, sweep)
    log.info("saved %s (%d rows, mean exo px %.1f)", out, len(sweep["jval"]), px)
    return 0


if __name__ == "__main__":
    sys.exit(main())
