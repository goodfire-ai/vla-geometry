"""House-level cross-validation folds read from the shipped fold maps (lists/folds/*.csv).

Every probe in the paper is scored out of house: all rows of a house fall in the same one of 5
folds. The fold of each house is read from a shipped map (columns `house,fold`) rather than computed
at run time, because scikit-learn's GroupKFold breaks ties between equally sized houses in an order
that depends on the CPU's sort kernel, so the same data can give different folds on different
machines.

Maps shipped in lists/folds/:
  target_position.csv     906 houses; target-position probes on the first-query rows of the
                          object-teleport store (Figure 1)
  distance.csv            905 houses; distance probes on the combined object-teleport and
                          near-range rows (distance < 0.70 m; Figures 1 and 3)
  distance_teleport.csv   905 houses; distance probes on the object-teleport rows alone (Figure 3a)
  distance_plant.csv      120 synthetic houses; probes on synthetic activations (Figure 3c)
  position.csv            945 houses; probes and Procrustes maps of the crossed arm x object
                          store (Figure 2)
  contact_natural.csv     192 houses; contact probes on the natural queries (Figure 4)
  phase.csv               167 houses; phase decoders on the successful natural queries (Figure 5)
  gripper_probe.csv       170 houses; gripper-command probes on the edit rows (Figure 7b)

`FoldMap.splits(houses)` yields (train, test) row-index arrays for fold 0..4 in that order, with
test = the rows whose house maps to the fold (ascending) and train = all other rows (ascending).
This is the same index layout GroupKFold.split returns, so code written around GroupKFold gives the
same per-fold arrays.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

N_FOLDS = 5


class FoldMap:
    """A house -> fold assignment (folds 0..N_FOLDS-1)."""

    def __init__(self, house, fold, name=""):
        house = np.asarray(house, np.int64)
        fold = np.asarray(fold, np.int64)
        if house.shape != fold.shape or house.ndim != 1:
            raise ValueError("house and fold must be 1-D arrays of equal length")
        if len(np.unique(house)) != len(house):
            raise ValueError(f"fold map {name!r}: a house appears more than once")
        if sorted(np.unique(fold).tolist()) != list(range(N_FOLDS)):
            raise ValueError(f"fold map {name!r}: folds are not 0..{N_FOLDS - 1}")
        order = np.argsort(house, kind="stable")
        self.house = house[order]
        self.fold = fold[order]
        self.name = name

    @classmethod
    def load(cls, name_or_path):
        """Read a map by name ("target_position" -> <repo>/lists/folds/target_position.csv) or
        by path to a `house,fold` CSV file."""
        p = Path(name_or_path)
        if p.suffix != ".csv":
            from vlageom.paths import lists_dir
            p = lists_dir() / "folds" / f"{name_or_path}.csv"
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        return cls([int(r["house"]) for r in rows], [int(r["fold"]) for r in rows], name=p.stem)

    def __len__(self):
        return len(self.house)

    def sha1(self):
        """Content hash of the (house, fold) pairs sorted by house (int64 bytes)."""
        pairs = np.stack([self.house, self.fold], 1)
        return hashlib.sha1(pairs.tobytes()).hexdigest()

    def contains(self, houses):
        """Boolean mask: which entries of `houses` have a fold in this map."""
        return np.isin(np.asarray(houses).astype(np.int64), self.house)

    def fold_ids(self, houses):
        """Per-row fold id for the house labels `houses`. Raises if a house is not in the map."""
        g = np.asarray(houses).astype(np.int64)
        pos = np.clip(np.searchsorted(self.house, g), 0, len(self.house) - 1)
        bad = self.house[pos] != g
        if bad.any():
            raise ValueError(f"fold map {self.name!r}: {int(bad.sum())} rows from houses outside "
                             f"the map, e.g. {np.unique(g[bad])[:10].tolist()}")
        return self.fold[pos]

    def splits(self, houses):
        """[(train_idx, test_idx)] for folds 0..N_FOLDS-1 over the rows labelled by `houses`."""
        f = self.fold_ids(houses)
        return [(np.nonzero(f != k)[0], np.nonzero(f == k)[0]) for k in range(N_FOLDS)]


def load_fold_map(name_or_path):
    """Shorthand for FoldMap.load."""
    return FoldMap.load(name_or_path)
