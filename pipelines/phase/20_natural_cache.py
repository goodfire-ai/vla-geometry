"""Phase, stage 20 (CPU): one table of the natural rollouts' model queries.

Reads the natural episode files of stage 10 for the episodes of lists/phase/natural_episodes.csv
(in list order) and writes one row per model query whose action-expert activations are finite:
the activations of the 11 action-expert blocks (X), the phase label of the query step from the
episode's per-step records (vlageom.phase.label_steps, step resolution) and the per-query and
per-episode fields listed in vlageom.phase.build_cache.

Output: --out (default <data>/phase/natural/cache_ae.npz). Prints the episode, house, success and
row counts (the paper's rollouts: 530 episodes, 181 houses, 334 successful episodes (63%), 20,086
query rows, 7,738 of them from 334 successful episodes in 167 houses).

  python pipelines/phase/20_natural_cache.py [--episodes-dir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom import phase as P


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-dir", type=Path,
                    default=paths.tile_dir("phase", "natural", "episodes"),
                    help="natural episode files of stage 10 (default "
                         "<data>/phase/natural/episodes)")
    ap.add_argument("--out", type=Path, default=paths.tile_dir("phase", "natural", "cache_ae.npz"))
    a = ap.parse_args(argv)

    listed = P.read_natural_episodes()
    files, missing = [], []
    for r in listed:
        f = a.episodes_dir / P.natural_name(r["episode_index"], r["house"])
        (files if f.exists() else missing).append(f if f.exists() else r["episode_index"])
    names = {P.natural_name(r["episode_index"], r["house"]) for r in listed}
    extra = sorted(f.name for f in a.episodes_dir.glob("ep*_h*.npz") if f.name not in names)
    if missing:
        print(f"{len(missing)} listed episodes have no file (not used): {missing[:20]}")
    if extra:
        print(f"{len(extra)} files in {a.episodes_dir} are not listed (not used): {extra[:10]}")
    if not files:
        print(f"no episode files in {a.episodes_dir}")
        return 1

    d, counts = P.build_cache(files)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **d)
    n_ep = counts["episodes"]
    print(f"{n_ep} episodes, {counts['houses']} houses, {counts['success_episodes']} successful "
          f"({100 * counts['success_episodes'] / n_ep:.0f}%); {counts['rows']} query rows, "
          f"{counts['success_rows']} from successful episodes in "
          f"{counts['success_houses']} houses; "
          f"labels {json.dumps(counts['labels'])}; non-finite queries left out: "
          f"{counts['dropped_nonfinite_queries']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
