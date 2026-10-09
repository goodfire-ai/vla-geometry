"""Check lists/folds/gripper_probe.csv (the house folds of the Figure 7b probes) against an edit store.

The rows of the probes are the edit files of the episodes in lists/gripper_episodes.csv, in
file-name order (pipelines/gripper_fig7/20_probe.py). The fold map was assigned once by
scikit-learn GroupKFold(5) over the house of each row: a house's fold is the index of the split in
which its rows are the test rows. This script repeats that assignment from the file names in
--edits (no file is opened) and compares it with the shipped map.

GroupKFold orders houses with equal row counts by a sort whose result depends on the CPU's sort
kernel, so the assignment repeats exactly only on an x86-64 CPU with AVX-512 (the machine type the
map was made on); elsewhere the script reports how many houses fall into other folds.

  python tests/provenance/check_gripper_fig7_lists.py --edits <gripper edit files>

Exit status 0 when the assignment equals the shipped map.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from vlageom import gripper as G
from vlageom.probes import FoldMap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--edits", type=Path, required=True,
                    help="directory of edit files ep<E>_h<H>_<landmark>_q<Q>_aw300_sw3000.npz")
    ap.add_argument("--episodes", type=Path, default=None,
                    help="episode list CSV (default lists/gripper_episodes.csv)")
    ap.add_argument("--n-folds", type=int, default=5)
    a = ap.parse_args(argv)

    from sklearn.model_selection import GroupKFold

    files = G.edit_files(a.edits, G.read_episodes(a.episodes))
    house = np.asarray([int(G.EDIT_RE.fullmatch(f.name).group(2)) for f in files], np.int64)
    fold_of = {}
    X = np.zeros((len(house), 1))
    for i, (_, te) in enumerate(GroupKFold(a.n_folds).split(X, groups=house)):
        for h in np.unique(house[te]):
            fold_of[int(h)] = i

    shipped = FoldMap.load("gripper_probe")
    ref = dict(zip(np.asarray(shipped.house).tolist(), np.asarray(shipped.fold).tolist()))
    print(f"gripper_probe.csv: {len(files)} rows, {len(fold_of)} houses from {a.edits}; "
          f"shipped map {len(ref)} houses, sha1 {shipped.sha1()}")
    ok = set(fold_of) == set(ref)
    if not ok:
        print(f"  FAIL  house sets differ: {len(set(fold_of) - set(ref))} only derived, "
              f"{len(set(ref) - set(fold_of))} only shipped")
    moved = sorted(h for h in set(fold_of) & set(ref) if fold_of[h] != ref[h])
    sizes = np.bincount(list(fold_of.values()), minlength=a.n_folds).tolist()
    print(f"  houses per derived fold: {sizes}")
    if moved:
        ok = False
        print(f"  FAIL  {len(moved)} houses in another fold than shipped (first: {moved[:10]})")
    print("  ok    derived folds equal the shipped map" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
