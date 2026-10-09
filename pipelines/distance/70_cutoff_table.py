"""Stage 70 (CPU, seconds): near and far scores of three probe families per boundary c (Figure 3b).

Reads the probe bank of stage 60 (the stored float32 out-of-fold read-outs, the true distances
and the fold of every row). For every boundary c and each range (near = rows with d < c, far =
rows with d >= c) it scores, against the true distance in cm:

  split  the range-split probe of that range (nr<c> near, fr<c> far)
  clip   the round-to-c probe of that range (cm<c> near, far<c> far)
  n0     the full-range probe m (metres, read in cm)
  n1     the full-range probe recalibrated per range: in each fold, a + b * read-out fit by least
         squares on the out-of-fold read-outs of the range's training rows, applied to its test
         rows (vlageom.distance.recal_oof)
  floor  the training mean of the range

Scores: n, MAE (cm) and R^2 over the range's rows (vlageom.distance.side_scores).
Writes --out (default $VLAGEOM_DATA/distance/cutoff_table_last_tok_L21.json):
  {"grid_cm", "n_rows", "n_below15", "families", "table": {c: {"near"|"far": {"metrics":
  {family: {"n", "mae_cm", "r2"}}}}}, "bank", "bank_sha1"}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.distance import (CUTOFF_FAMILIES, FIG3_CELL, bank_probe_names, cutoff_predictions,
                              cutoff_table)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", type=Path,
                    default=paths.tile_dir("distance", "probe_bank_last_tok_L21.npz"),
                    help="stage 60 output "
                         "(default: $VLAGEOM_DATA/distance/probe_bank_last_tok_L21.npz)")
    ap.add_argument("--out", type=Path,
                    default=paths.tile_dir("distance", "cutoff_table_last_tok_L21.json"))
    a = ap.parse_args()

    bank = np.load(a.bank, allow_pickle=False)
    y_cm = bank["dist"].astype(np.float64) * 100
    fold = bank["fold_id"].astype(int)
    grid = [int(c) for c in bank["grid_cm"]]
    oof = {k: bank[f"oof__{k}"].astype(np.float64) for k in bank_probe_names(grid)}
    table = cutoff_table(cutoff_predictions(oof, y_cm, fold, grid), y_cm)
    variant, layer = FIG3_CELL
    out = {"cell": f"{variant}@L{layer}", "grid_cm": grid, "n_rows": int(len(y_cm)),
           "n_below15": int((y_cm < 15).sum()), "families": list(CUTOFF_FAMILIES),
           "table": table, "bank": str(a.bank),
           "bank_sha1": hashlib.sha1(a.bank.read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for c in grid:
        line = []
        for reg in ("near", "far"):
            met = table[str(c)][reg]["metrics"]
            line.append(f"{reg} R^2 split {met['split']['r2']:.4f} clip {met['clip']['r2']:.4f} "
                        f"n1 {met['n1']['r2']:.4f}")
        print(f"  c={c:>2}: " + " | ".join(line))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
