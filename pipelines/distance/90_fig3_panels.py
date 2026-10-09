"""Stage 90 (CPU, seconds): the panel files of Figure 3 and the numbers the text quotes.

Each panel is written from the output of one earlier stage (all under $VLAGEOM_DATA/distance):

  a  erasure/erase_far.json + erase_near.json (stage 50)   -> panels/fig3a_erasure.npz
       k (m,) int64, and held-out R^2 of the near and far probes after k directions of the far
       probe (near_erase_far, far_erase_far) or of the near probe (near_erase_near,
       far_erase_near) were erased
  b  cutoff_table_last_tok_L21.json (stage 70)              -> panels/fig3b_cutoff.npz
       grid_cm (10,) int64; R^2 on the rows below c (<family>_near) and at or above c
       (<family>_far) for family split (range-split pair), clip (round-to-c pair) and n1
       (full-range probe recalibrated per range)
  c  probe_bank_last_tok_L21.json (stage 60) + plant_cosines.json (stage 80)
                                                            -> panels/fig3c_cosine.npz
       grid_cm; per pair split (nr<c>, fr<c>) and round (cm<c>, far<c>): mean and std over folds
       of the cosine between the near and far weights (<pair>_cos, <pair>_cos_std), the weaker
       probe's median split-half ceiling (<pair>_ceil) and the mean cosine on the synthetic
       activations (<pair>_plant)
  d  ruler_probes.npz (stage 30, the two Figure 1 rulers)   -> panels/fig3d_plane.npz
       the stratified plot sample of Figure 1 (vlageom.distance.plot_samples): d_cm (true
       distance, cm), x_m (metre-ruler read-out, m), y_cm (centimetre-ruler read-out, cm); the
       per-bin mean path of the sample (path_x_m, path_y_cm; bins with at least 30 rows) and
       cut_cm = 15

Every panel file also holds `src`, a JSON record of its input files and their sha1.
After writing, the stage prints the values the paper quotes from these analyses (erasure R^2 and
MAE at k = 0 and 32, the ruler scores, the c = 15 cosines and ceilings, the synthetic cosines at
c = 15, the near and far R^2 of the range-split pair at c = 15, row counts), and the row counts of
the combined rows per model query index (q0..q3 from both stores, q4..q7 from the near store
only).

  python pipelines/distance/90_fig3_panels.py [--panel a b c d] [--data DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.distance import (RULER_CUT_CM, cosine_panel, cutoff_panel, erasure_panel,
                              plane_panel, query_counts, ruler_scores)

PANELS = ("a", "b", "c", "d")
PANEL_FILES = {"a": "fig3a_erasure.npz", "b": "fig3b_cutoff.npz", "c": "fig3c_cosine.npz",
               "d": "fig3d_plane.npz"}
INPUTS = {"a": ["erasure/erase_far.json", "erasure/erase_near.json"],
          "b": ["cutoff_table_last_tok_L21.json"],
          "c": ["probe_bank_last_tok_L21.json", "plant_cosines.json"],
          "d": ["ruler_probes.npz"]}


def _src(files, root):
    """JSON record {input path relative to the data directory: sha1}."""
    return json.dumps({str(f.relative_to(root)): hashlib.sha1(f.read_bytes()).hexdigest()
                       for f in files}, sort_keys=True)


def build(panel, files):
    """Panel arrays and the printed lines of one panel from its input files."""
    lines = []
    if panel == "a":
        far, near = (json.loads(f.read_text()) for f in files)
        arr = erasure_panel(far["curve"], near["curve"])
        c0 = far["curve"][0]
        lines.append(f"a: {far['n_rows']} rows in {far['n_houses']} houses; near range "
                     f"{far['ranges']['near']['n']} rows ({far['ranges']['near']['n_houses']} "
                     f"houses); penalties {far['alpha']}")
        lines.append(f"a: k = 0: near R^2 {c0['near']['r2']:.4f} (MAE "
                     f"{c0['near']['mae_cm']:.4f} cm), far R^2 {c0['far']['r2']:.4f} (MAE "
                     f"{c0['far']['mae_cm']:.4f} cm)")
        for name, run in (("far", far), ("near", near)):
            e = run["curve"][-1]
            lines.append(f"a: k = {e['k']} {name}-probe directions erased: near R^2 "
                         f"{e['near']['r2']:.4f}, far R^2 {e['far']['r2']:.4f}")
    elif panel == "b":
        tab = json.loads(files[0].read_text())
        arr = cutoff_panel(tab["table"], tab["grid_cm"])
        m15 = tab["table"]["15"]
        lines.append(f"b: range-split pair at c = 15: near R^2 "
                     f"{m15['near']['metrics']['split']['r2']:.4f}, far R^2 "
                     f"{m15['far']['metrics']['split']['r2']:.4f}")
        for c, xn, xf in zip(tab["grid_cm"], arr["split_near"], arr["split_far"]):
            lines.append(f"b:   c = {c:>2}: split near {xn:.4f} far {xf:.4f}, distance to (1, 1) "
                         f"{np.hypot(1 - xn, 1 - xf):.3f}")
    elif panel == "c":
        bank, plant = (json.loads(f.read_text()) for f in files)
        grid = bank["grid_cm"]
        arr = cosine_panel(bank["catalog"], bank["cosines"]["per_fold"], plant, grid)
        i = grid.index(15)
        lines.append(f"c: {bank['n_rows']} rows in {bank['n_houses']} houses")
        for pair in ("round", "split"):
            lines.append(f"c: {pair} pair at c = 15: cosine {arr[f'{pair}_cos'][i]:.4f} (std "
                         f"{arr[f'{pair}_cos_std'][i]:.4f}), ceiling {arr[f'{pair}_ceil'][i]:.4f}, "
                         f"synthetic {arr[f'{pair}_plant'][i]:.4f}")
        pre = bank["catalog"]["cm15"].get("mae_below15_precontact_cm")
        if pre is not None:
            mae = bank["catalog"]["cm15"]["mae_below15_cm"]
            lines.append(f"c: cm15 MAE below 15 cm: {mae:.4f} cm, on the near-store rows where "
                         f"the object touches nothing but its support {pre:.4f} cm")
    else:
        z = np.load(files[0])
        arr = plane_panel(z["d_m"], z["far15"], z["cm15"])
        d_cm = z["d_m"] * 100
        sc = ruler_scores(d_cm, {"far15": z["far15"], "cm15": z["cm15"]})
        lines.append(f"d: {len(d_cm)} rows ({sc['n_below15']} below {RULER_CUT_CM:g} cm), "
                     f"{len(arr['d_cm'])} points, {len(arr['path_x_m'])} path points; metre "
                     f"ruler R^2 {sc['far15']['r2_above15']:.10f}, centimetre ruler R^2 "
                     f"{sc['cm15']['r2_below15']:.10f}")
        q = query_counts(z["chunk_q"], z["source"], z["d_m"])
        lines.append(f"d: rows per query index q0..q7: {q['per_query']} (teleport store "
                     f"{q['per_query_teleport']}, near store {q['per_query_near']})")
        lines.append(f"d: queries q4-q7 give {q['late']} of {q['n']} rows "
                     f"({100 * q['late'] / q['n']:.1f}%) and {q['late_below15']} of "
                     f"{q['n_below15']} rows below {RULER_CUT_CM:g} cm "
                     f"({100 * q['late_below15'] / q['n_below15']:.1f}%)")
    return arr, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", nargs="+", choices=PANELS, default=list(PANELS),
                    help="panels to write (default: all)")
    ap.add_argument("--data", type=Path, default=paths.tile_dir("distance"),
                    help="distance stage outputs (default: $VLAGEOM_DATA/distance)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="panel directory (default: $VLAGEOM_DATA/panels)")
    a = ap.parse_args()
    out_dir = a.out_dir or paths.panels_dir(create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for panel in a.panel:
        files = [a.data / f for f in INPUTS[panel]]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"panel {panel}: missing input {missing}")
        arr, lines = build(panel, files)
        out = out_dir / PANEL_FILES[panel]
        np.savez(out, **arr, src=np.array(_src(files, a.data)))
        for line in lines:
            print(line)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
