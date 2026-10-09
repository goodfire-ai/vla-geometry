#!/usr/bin/env python
"""Compare a target-tile panel file (pipelines/target/50_panel.py) and its probe score
(40_probe.py) with reference files.

  python tests/compare_target_panel.py --reference <reference panel .npz> \
      [--panel $VLAGEOM_DATA/panels/target.npz] \
      [--ref-probe <reference probe .json>] [--new-probe $VLAGEOM_DATA/target/probe/last_tok_L21.json]

Panel: every array of the reference file is compared with the same key of the new file (shape,
dtype, values with np.array_equal; the largest absolute difference is printed otherwise). The
string `src` records input locations and is printed, not required to match. Keys only in the new
file (note_r2, note_r2_xyz) are compared with the reference probe file when it is given.

Probe: per-axis R^2 (x, y, z) and their mean. A reference probe file may use this repository's
schema ("targets" / "note_r2") or a {"q0|raw": {"fwd": {"r2": ...}, ...}} block.

Exit status 0 when every numeric panel array is bit-identical and (if given) the printed probe
value (2 decimals) is the same; differences below 1e-12 in the probe R^2 are reported as rounding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import paths  # noqa: E402

AXES = ("fwd", "lat", "up")


def probe_r2(j: dict) -> dict:
    if "targets" in j:
        return {k: float(j["targets"][k]["r2"]) for k in AXES}
    if "q0|raw" in j:
        return {k: float(j["q0|raw"][k]["r2"]) for k in AXES}
    raise SystemExit("probe json: no 'targets' or 'q0|raw' block")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", "--new", dest="new", type=Path, default=None,
                    help="panel to check (default $VLAGEOM_DATA/panels/target.npz)")
    ap.add_argument("--reference", "--ref", dest="ref", type=Path, required=True,
                    help="reference panel .npz")
    ap.add_argument("--new-probe", type=Path, default=None,
                    help="default $VLAGEOM_DATA/target/probe/last_tok_L21.json")
    ap.add_argument("--ref-probe", type=Path, default=None, help="reference probe .json")
    args = ap.parse_args()

    new_p = args.new or paths.panels_dir() / "target.npz"
    a, b = np.load(new_p), np.load(args.ref)
    ok = True
    print(f"panel  new {new_p}\n       ref {args.ref}")
    for k in b.files:
        y = b[k]
        if k not in a.files:
            print(f"  MISSING {k}")
            ok = False
            continue
        x = a[k]
        if y.dtype.kind == "U":
            same = str(x) == str(y)
            print(f"  {'same ' if same else 'other'} {k} (provenance string; not required to match)")
            if not same:
                print(f"         new {str(x)}\n         ref {str(y)}")
            continue
        if x.shape != y.shape or x.dtype != y.dtype:
            print(f"  DIFF  {k}: new {x.dtype}{x.shape} vs ref {y.dtype}{y.shape}")
            ok = False
            continue
        eq = np.array_equal(x, y)
        md = 0.0 if eq else float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        print(f"  {'EQUAL' if eq else 'DIFF '} {k} {y.dtype}{y.shape}" + ("" if eq else f" max|diff| {md:.3g}"))
        ok &= eq
    extra = [k for k in a.files if k not in b.files]
    if extra:
        print(f"  keys only in the new file: {extra}")

    new_j = args.new_probe or paths.tile_dir("target", "probe", "last_tok_L21.json")
    jn = json.loads(Path(new_j).read_text()) if Path(new_j).exists() else None
    if "note_r2" in a.files:
        r = [float(v) for v in np.atleast_1d(a["note_r2_xyz"])] if "note_r2_xyz" in a.files else None
        print(f"panel note_r2 {float(a['note_r2'])!r} (prints {float(a['note_r2']):.2f}); per axis {r}")
        if jn is not None and float(a["note_r2"]) != float(jn["note_r2"]):
            print(f"  DIFF  panel note_r2 != {new_j} note_r2 {jn['note_r2']!r}")
            ok = False
    if args.ref_probe is not None:
        if jn is None:
            raise SystemExit(f"no probe json at {new_j}")
        rn, rr = probe_r2(jn), probe_r2(json.loads(args.ref_probe.read_text()))
        print(f"probe  new {new_j}\n       ref {args.ref_probe}")
        for k in AXES:
            d = rn[k] - rr[k]
            tag = "EQUAL" if d == 0 else ("ROUND" if abs(d) < 1e-12 else "DIFF ")
            print(f"  {tag} r2 {k}: new {rn[k]!r} ref {rr[k]!r} diff {d:.3g}")
            ok &= abs(d) < 1e-12
        mn, mr = float(np.mean([rn[k] for k in AXES])), float(np.mean([rr[k] for k in AXES]))
        same_print = f"{mn:.2f}" == f"{mr:.2f}"
        print(f"  {'EQUAL' if mn == mr else 'DIFF '} mean R^2: new {mn!r} ref {mr!r} "
              f"(prints {mn:.2f} vs {mr:.2f})")
        ok &= same_print
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
