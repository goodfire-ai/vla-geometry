"""Stage 40 (CPU, seconds): the Figure 1 distance panel.

Reads the out-of-fold ruler read-outs of stage 30, draws the stratified plot sample
(vlageom.distance.plot_samples: every row below 15 cm, at most 400 rows per distance bin above,
numpy default_rng(45)) and the per-bin mean path, and writes $VLAGEOM_DATA/panels/distance.npz:
  pts     (n, 3) float32   far15 read-out, cm15 read-out, true distance; each axis mapped to
                           [0, 1] over (-0.06, 0.76) m, (-1.5, 18) cm and (0, 0.71) m
  d       (n,) float32     true distance (m), colours the points
  path    (k, 3) float32   per-bin means of the sample in the same coordinates
  path_d  (k,) float32     true distance of each path point (m)
  cross   (2,) float64     the 15 cm point on the x and y axes
  r2_m    ()  float64      R^2 of the metre ruler (far15) on d >= 15 cm
  r2_cm   ()  float64      R^2 of the centimetre ruler (cm15) on d < 15 cm
  note    ()  str          the legend text, e.g. "$R^2$ = 0.81 (m) / 0.70 (cm)"
  src     ()  str          JSON record of the input file and its sha1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.distance import panel_arrays, plot_samples, r2_note, ruler_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probes", type=Path, default=paths.tile_dir("distance", "ruler_probes.npz"),
                    help="stage 30 output (default: $VLAGEOM_DATA/distance/ruler_probes.npz)")
    ap.add_argument("--out", type=Path, default=None,
                    help="panel file (default: $VLAGEOM_DATA/panels/distance.npz)")
    a = ap.parse_args()
    out = a.out or paths.panels_dir(create=True) / "distance.npz"

    z = np.load(a.probes)
    d_cm = z["d_m"] * 100
    oof = {"far15": z["far15"], "cm15": z["cm15"]}
    sc = ruler_scores(d_cm, oof)
    s = plot_samples(z["d_m"])["stratified"]
    pan = panel_arrays(d_cm[s], oof["far15"][s], oof["cm15"][s])
    r2_m, r2_cm = sc["far15"]["r2_above15"], sc["cm15"]["r2_below15"]
    src = json.dumps({"probes": a.probes.name,
                      "sha1": hashlib.sha1(a.probes.read_bytes()).hexdigest(),
                      "n_rows": int(len(d_cm)), "n_points": int(len(s))}, sort_keys=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **pan, src=np.array(src), r2_m=np.float64(r2_m), r2_cm=np.float64(r2_cm),
             note=np.array(r2_note(r2_m, r2_cm)))
    print(f"{len(s)} points ({int((d_cm[s] < 15).sum())} below 15 cm), {len(pan['path'])} path "
          f"points; R^2 m {r2_m:.4f} -> '{r2_m:.2f}', cm {r2_cm:.4f} -> '{r2_cm:.2f}'")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
