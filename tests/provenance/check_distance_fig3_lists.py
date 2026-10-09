"""Check the fold maps read by the Figure 3 distance stages.

  plant     lists/folds/distance_plant.csv, the house -> fold map of the 120 synthetic houses of
            pipelines/distance/80_plant_cosines.py. It was assigned once by scikit-learn
            GroupKFold(5) over the synthetic house label of each of the 24,000 rows
            (vlageom.distance.linear_plant): a house's fold is the index of the split in which its
            rows are the test rows. The script repeats that assignment and compares it with the
            shipped map. No data are needed.
  teleport  lists/folds/distance_teleport.csv applied to the rows of the erasure stage
            (pipelines/distance/50_erasure.py: teleport-store rows with distance < 0.70 m): prints
            the row and house counts and the md5 of the concatenated test-row indices of folds
            0..4 (int64), and compares the md5 with --expect-md5 when given.

GroupKFold orders houses with equal row counts by a sort whose result depends on the CPU's sort
kernel (105 of the 120 synthetic houses tie with another house), so the plant assignment repeats
exactly only on an x86-64 CPU with AVX-512, the machine type the map was made on; elsewhere the
script reports how many houses fall into other folds.

  python tests/provenance/check_distance_fig3_lists.py plant
  python tests/provenance/check_distance_fig3_lists.py teleport [--teleport-store DIR] \
      [--expect-md5 45590095eeec32a93dac233b50356fde]

Exit status 0 when the derived values equal the shipped (or expected) ones.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from vlageom.probes import FoldMap


def check_plant(a) -> int:
    from sklearn.model_selection import GroupKFold

    from vlageom.distance import PLANT_FOLDS, linear_plant

    X, _, houses, _ = linear_plant()
    fold_of = {}
    for i, (_, te) in enumerate(GroupKFold(5).split(X, groups=houses)):
        for h in np.unique(houses[te]):
            fold_of[int(h)] = i
    shipped = FoldMap.load(PLANT_FOLDS)
    ref = dict(zip(shipped.house.tolist(), shipped.fold.tolist()))
    counts = np.bincount(houses, minlength=max(ref) + 1)
    n_tied = int(sum(1 for h in ref if (counts == counts[h]).sum() > 1))
    print(f"{PLANT_FOLDS}.csv: {len(ref)} houses, sha1 {shipped.sha1()}; derived {len(fold_of)} "
          f"houses ({n_tied} share their row count with another house)")
    moved = sorted(h for h in ref if fold_of.get(h) != ref[h])
    ok = set(fold_of) == set(ref) and not moved
    print(f"  {'PASS' if ok else 'FAIL'}  {len(moved)} houses in another fold"
          + (f": {moved[:20]}" if moved else ""))
    return 0 if ok else 1


def check_teleport(a) -> int:
    from vlageom.distance import ERASE_FOLDS, erasure_rows, teleport_prep

    folds = FoldMap.load(ERASE_FOLDS)
    X, y, houses = erasure_rows(a.teleport_store or teleport_prep(), folds)
    del X
    splits = folds.splits(houses)
    md5 = hashlib.md5(np.concatenate([te.astype(np.int64) for _, te in splits]).tobytes())
    md5 = md5.hexdigest()
    print(f"{ERASE_FOLDS}.csv: {len(folds)} houses, sha1 {folds.sha1()}; erasure rows {len(y)} "
          f"in {len(np.unique(houses))} houses ({int((y < 0.15).sum())} below 15 cm); test rows "
          f"per fold {[len(te) for _, te in splits]}; test-row md5 {md5}")
    if a.expect_md5 is None:
        return 0
    ok = md5 == a.expect_md5
    print(f"  {'PASS' if ok else 'FAIL'}  expected {a.expect_md5}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plant")
    s = sub.add_parser("teleport")
    s.add_argument("--teleport-store", type=Path, default=None,
                   help="packed object-teleport store (default $VLAGEOM_DATA/target/teleport/prep)")
    s.add_argument("--expect-md5", default=None, help="expected test-row md5")
    a = ap.parse_args(argv)
    return check_plant(a) if a.cmd == "plant" else check_teleport(a)


if __name__ == "__main__":
    sys.exit(main())
