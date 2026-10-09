"""Stage 10 (CPU, torch): principal components of the edits of 80% of the houses (Figure 7a, 7b).

Reads the edit files of the listed episodes (lists/gripper_episodes.csv) from --edits, in file-name
order (vlageom.gripper_fig7.load_listed_edits). The houses of these edits are split once: the
sorted unique houses are permuted by numpy RandomState(0) and the first 80% (rounded) are the train
houses. The edits of the train houses, minus their mean, go into one SVD (torch.linalg.svd on the
CPU, float32). Writes --out (default $VLAGEOM_DATA/gripper_fig7/basis.npz):

  mu (768,), V (768, 768)       mean train edit and the principal components (columns)
  evr, sing (768,)              explained-variance ratios and singular values
  mu_close, mu_open (768,)      mean train edit of the closing / opening edits
  centered_rank                 singular values above 1e-5 x the largest
  train_houses, heldout_houses  the house split; split_seed, train_frac
  n_train_edits, n_edits        edit counts

With the paper's edits: 170 houses -> 136 train / 34 held-out, 1,329 of 1,666 edits, evr of PC1
0.698, stable rank 1.43 (the panel-a text), centered rank 768. Takes about 1 min
(most of it reading the edit files).

The last bits of torch's CPU SVD depend on the number of threads. With the pinned library
versions on an x86-64 CPU with AVX-512 and 8 threads (--threads 8, the default) the basis equals
the paper's to the last bit, apart from the signs of the columns of V (an SVD determines each column
only up to its sign; nothing downstream depends on the signs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlageom import gripper_fig7 as F
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
                    help="basis file (default: $VLAGEOM_DATA/gripper_fig7/basis.npz)")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads (default 8)")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)

    from vlageom import gripper as G
    E = F.load_listed_edits(a.edits, G.read_episodes(a.episodes))
    basis = F.edit_basis(E["D"], E["target"], E["house"])
    out = a.out or paths.tile_dir(F.TILE, create=True) / "basis.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **basis)
    print(json.dumps(dict(
        edits=int(basis["n_edits"]), train_edits=int(basis["n_train_edits"]),
        houses=int(len(basis["train_houses"]) + len(basis["heldout_houses"])),
        train_houses=int(len(basis["train_houses"])), evr1=round(float(basis["evr"][0]), 6),
        stable_rank=round(F.stable_rank(basis["sing"]), 6),
        centered_rank=int(basis["centered_rank"]), out=str(out))))


if __name__ == "__main__":
    main()
