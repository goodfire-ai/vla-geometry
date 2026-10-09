#!/usr/bin/env python
"""Panel arrays of the top row of Figure 2 (CPU, seconds): $VLAGEOM_DATA/panels/fig2_top.npz.

Inputs: the probe JSONs of the two cells (30_probe.py; panels a-c use the q1 rows), the merged
Procrustes result of the q0 rows (40_procrustes.py --merge; panel d) and the curvature result
(60_curvature.py; panel e). Output keys: see `vlageom.position.top_panel` (rows: EE, target).

Printed: the drawn values (error per axis and 3-D, rank R^2, singular-value fractions, the
residuals at the read-out layers, the curvature bins) and the numbers of the text: held-out R^2,
within-house R^2 and the R^2 of predicting one variable's x and y from the other's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.position import (CELLS, FIG_GRAIN, PROC_GRAIN, curvature_json_default, probe_dir,
                              probe_json_path, procrustes_dir, top_panel)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dir", type=Path, default=None,
                    help="30_probe.py output directory (default $VLAGEOM_DATA/position/probe)")
    ap.add_argument("--procrustes", type=Path, default=None,
                    help=f"merged Procrustes JSON (default $VLAGEOM_DATA/position/procrustes/"
                         f"procrustes_{PROC_GRAIN}.json)")
    ap.add_argument("--curvature", type=Path, default=None,
                    help="60_curvature.py output "
                         "(default $VLAGEOM_DATA/position/curvature/curvature_q0.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="panel file (default $VLAGEOM_DATA/panels/fig2_top.npz)")
    args = ap.parse_args()

    pdir = args.probe_dir or probe_dir()
    probe = {v: json.loads(probe_json_path(*CELLS[v], pdir).read_text()) for v in ("ee", "obj")}
    proc_p = args.procrustes or procrustes_dir() / f"procrustes_{PROC_GRAIN}.json"
    curv_p = args.curvature or curvature_json_default()
    proc = json.loads(proc_p.read_text())
    curv = json.loads(curv_p.read_text())
    panel = top_panel(probe, proc, curv)
    panel["src"] = np.array(json.dumps({"probe": str(pdir), "procrustes": str(proc_p),
                                        "curvature": str(curv_p)}, sort_keys=True))
    out = args.out or paths.panels_dir(create=True) / "fig2_top.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **panel)

    g = FIG_GRAIN
    for i, v in enumerate(("ee", "obj")):
        c = probe[v]["cells"][g][v]
        b = probe[v]["grains"][g]["baselines"]["label_echo_r2"]
        echo = b["obj_from_ee"] if v == "obj" else b["ee_from_obj"]
        print(f"{v} ({panel['cells'][i]}, layer {panel['layers'][i]}, {g}): "
              f"R^2 {c['ridge_full']:.4f}, "
              f"within-house {c['r2_within']:.4f}; error x/y/z/3-D cm "
              f"{np.round(panel['mae_cm'][i], 2).tolist()} (fold std "
              f"{np.round(panel['mae_cm_fold_std'][i], 2).tolist()}); rank R^2 "
              f"{np.round(panel['rrr'][i], 3).tolist()}; sv fractions "
              f"{np.round(panel['sv_frac'][i], 3).tolist()}; R^2 of x, y from the other variable "
              f"{np.round(echo[:2], 4).tolist()}", flush=True)
    for i, v in enumerate(("ee", "obj")):
        L = int(CELLS[v][1])
        j = list(panel["proc_layers"]).index(L)
        pl = proc["per_layer"][str(L)][v]
        print(f"{v} Procrustes ({PROC_GRAIN}) at layer {L}: general linear "
              f"{panel['proc_resid'][i, j]:.3f}, "
              f"orthogonal {panel['proc_ortho'][i, j]:.3f}, shuffled {pl['shuffle_resid']:.3f}, "
              f"{pl['n_pairs']} pairs", flush=True)
        cb = panel[f"curv_{v}"]
        print(f"{v} curvature: subspace {np.round(cb[:, 2], 1).tolist()} deg, random projections "
              f"{np.round(cb[:, 3], 1).tolist()} deg over "
              f"{cb[0, 0] * 100:.1f}-{cb[-1, 1] * 100:.1f} cm",
              flush=True)
    print(f"rows {int(panel['n_rows'])}, houses {int(panel['n_houses'])}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
