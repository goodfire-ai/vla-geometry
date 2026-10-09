"""Wrist steering, stage 20 (CPU): natural cloud of panel b.

Reads the natural episode files of stage 10 in the row order of lists/wrist_steer/natural_episodes.csv
and keeps every model query of the episodes with at least 5 queries (one episode, index 194, has
fewer): per query the token-0 and token-mean action-expert activations of blocks 0, 4, 9, 16, 24, 32,
the EE rotation and position before that step (robot base frame), the phase (query step / (steps - 1)),
house, episode, query index and episode success.

Outputs:
  <data>/wrist_steer/natural/natural_cache.npz   X, Xm (N, 6, 768) token 0 / token mean, R (N, 3, 3),
                                                 pos (N, 3), phase, house, ep, q, success, layers,
                                                 pass_idx
  <data>/wrist_steer/natural/natural.npz         X (N, 768) token mean of layer 9, R, phase, house,
                                                 ep, q, success

--from-cache FILE builds natural.npz from an existing natural cache file instead.

  python pipelines/wrist_steer/20_natural_cloud.py
"""
# Adapted in part from Mathilde Papillon's code.
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from vlageom import wrist_steer as ws

log = logging.getLogger("wrist_steer.20_natural_cloud")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-dir", type=Path, default=None,
                    help="natural episode files (default <data>/wrist_steer/natural/episodes)")
    ap.add_argument("--from-cache", type=Path, default=None,
                    help="build natural.npz from this natural cache file instead of the episodes")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default <data>/wrist_steer/natural)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = args.out_dir or ws.tile_path("natural")
    out.mkdir(parents=True, exist_ok=True)

    if args.from_cache is not None:
        cache = dict(np.load(args.from_cache, allow_pickle=False))
        print(f"natural cache {args.from_cache}")
    else:
        eps = ws.read_natural_episodes()
        files = [ws.natural_episode_path(e["episode_index"], args.episodes_dir) for e in eps]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} of {len(files)} natural episode files are missing, e.g. "
                             f"{missing[:5]}; run stage 10 for their partitions")
        cache = ws.natural_cache(files)
        np.savez(out / "natural_cache.npz", **cache)
        print(f"natural cache -> {out / 'natural_cache.npz'}")
    cloud = ws.natural_cloud(cache, ws.LAYER)
    np.savez(out / "natural.npz", **cloud)
    eps_in = np.unique(cloud["ep"])
    print(f"natural cloud -> {out / 'natural.npz'}")
    print(f"  {len(cloud['X'])} queries, {len(eps_in)} episodes, {len(np.unique(cloud['house']))} houses, "
          f"layer {ws.LAYER} token mean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
