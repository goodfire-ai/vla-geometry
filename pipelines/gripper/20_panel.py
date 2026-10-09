"""Stage 20 (CPU): the gripper panel of Figure 1.

Reads the edit files of the listed episodes (lists/gripper_episodes.csv) from --edits, in file-name
order, and computes (vlageom.gripper.panel_arrays):
  * the principal directions of the edits: SVD of the edits centered on their mean;
  * px, py, pz: the layer-32 activations without the edit (base_act), centered on their mean,
    projected on edit PC1-3; ddx, ddy, ddz: the edits projected on PC1-3;
  * target (+1 close, -1 open), aper (decoded gripper action before the edit, -1 open .. +1 closed),
    evr (explained-variance ratio of PC1-3), and a 2-D binned edit field (fc, fo, scale).

Writes --out (default $VLAGEOM_DATA/panels/gripper.npz) and prints the counts and evr. With the
paper's edits: 1,666 edits (863 close, 803 open) from 304 episodes in 170 houses; evr
0.703 / 0.056 / 0.020. Takes seconds.

The projections are matrix-vector products whose last bit depends on the number of BLAS threads.
With the pinned library versions on an x86-64 CPU with AVX-512 and 8 BLAS threads (--threads 8,
the default) the panel equals the paper's to the last bit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlageom import gripper as G
from vlageom import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--edits", type=Path, default=paths.tile_dir("gripper", "deltas"),
                    help="directory of edit files (default: $VLAGEOM_DATA/gripper/deltas)")
    ap.add_argument("--episodes", type=Path, default=None,
                    help="episode list CSV (default lists/gripper_episodes.csv)")
    ap.add_argument("--out", type=Path, default=None,
                    help="panel file (default: $VLAGEOM_DATA/panels/gripper.npz)")
    ap.add_argument("--threads", type=int, default=8, help="BLAS threads (default 8)")
    a = ap.parse_args()

    episodes = G.read_episodes(a.episodes)
    files = G.edit_files(a.edits, episodes)
    if not files:
        raise SystemExit(f"no edit files of the listed episodes in {a.edits}")
    E = G.load_edits(files)
    have = set(zip(E["episode_index"].tolist(), E["house"].tolist()))
    missing = [f"ep{e}_h{h}" for e, h, *_ in episodes if (e, h) not in have]

    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=a.threads):
        panel = G.panel_arrays(E["H"], E["D"], E["aper"], E["target"])
    out = a.out or paths.panels_dir(create=True) / "gripper.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **panel)

    summary = dict(
        edits=len(files), close=int((E["target"] > 0).sum()), open=int((E["target"] < 0).sum()),
        episodes=len(have), houses=len(set(E["house"].tolist())),
        listed_episodes=len(episodes), listed_without_edits=len(missing),
        evr=[round(float(v), 6) for v in panel["evr"]], out=str(out))
    print(json.dumps(summary))
    if missing:
        print(f"listed episodes without edit files ({len(missing)}): {' '.join(missing[:20])}"
              + (" ..." if len(missing) > 20 else ""))


if __name__ == "__main__":
    main()
