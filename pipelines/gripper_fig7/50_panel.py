"""Stage 50 (CPU): the arrays of Figure 7 panels a, b and d -> panels/gripper_fig7.npz.

Reads stage 10's basis, stage 20's probe scores and stage 40's summary
(vlageom.gripper_fig7.panel_arrays; without the summary only the arrays of panels a and b are
written):

  scree_evr (16,)                        panel a bars: explained-variance ratio of PC1-16
  stable_rank ()                         panel a text: sum(s^2) / s_1^2
  probe_k (10,), probe_r2 (10,)          panel b curve
  probe_r2_full ()                       panel b dashed line
  leak_k (13,)                           panel d k grid (k > 0)
  leak_close, leak_open (13,)            median arm leakage per k (closing / opening edits)
  free_leak_close, free_leak_open ()     median arm leakage of the unrestricted 768-D edits
  knee_close, knee_open ()               knee of each landmark (stage 40; not drawn)

Panel c is drawn from panels/gripper.npz (pipelines/gripper/20_panel.py). Prints the values
written into the figure. Takes seconds.
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
    d = paths.tile_dir(F.TILE)
    ap.add_argument("--basis", type=Path, default=d / "basis.npz", help="stage 10 output")
    ap.add_argument("--probe", type=Path, default=d / "probe.json", help="stage 20 output")
    ap.add_argument("--summary", type=Path, default=d / "ksweep_summary.json",
                    help="stage 40 output")
    ap.add_argument("--out", type=Path, default=None,
                    help="panel file (default: $VLAGEOM_DATA/panels/gripper_fig7.npz)")
    a = ap.parse_args()

    with np.load(a.basis) as z:
        basis = {k: z[k] for k in ("evr", "sing")}
    probe = json.loads(a.probe.read_text())
    summary = json.loads(a.summary.read_text()) if a.summary.exists() else None
    if summary is None:
        print(f"no stage 40 summary {a.summary}: panel d arrays not written")
    panel = F.panel_arrays(basis, probe, summary)
    out = a.out or paths.panels_dir(create=True) / "gripper_fig7.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **panel)

    print(f"a: PC1 {panel['scree_evr'][0]:.1%} of the edit variance, "
          f"stable rank = {panel['stable_rank']:.2f}")
    print("b: R^2 " + " ".join(f"{k}:{r:.4f}" for k, r in zip(panel["probe_k"], panel["probe_r2"]))
          + f"; full activation {panel['probe_r2_full']:.4f}")
    for lm, key in (("close", "leak_close"), ("open", "leak_open")):
        if key not in panel:
            break
        ref = panel[f"free_{key}"]
        print(f"d: {lm} leak / 768-D leak (%) " + " ".join(
            f"{k}:{100.0 * v / ref:.1f}" for k, v in zip(panel["leak_k"], panel[key])))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
