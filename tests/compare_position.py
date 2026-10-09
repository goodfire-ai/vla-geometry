#!/usr/bin/env python
"""Compare the CPU outputs of pipelines/position/ with reference outputs (paper or earlier runs).

  probe       30_probe.py JSONs vs reference JSONs of the same cells (same layout)
      python tests/compare_position.py probe --new $VLAGEOM_DATA/position/probe \
          --ref <reference dir> [--cells hdr_txt:20 txt_instr:21]
      every leaf of the reference's "grains" and "cells" (NaN == NaN), exactly; the reference may
      hold extra statistics, which are listed and not compared.

  procrustes  40_procrustes.py part files or merged files
      python tests/compare_position.py procrustes --new <dir> --ref <dir> [--files q0_L20.json ...]
      every file of --files (default: every *.json of --new) compared leaf by leaf with the file of
      the same name in --ref; --skip-shuffle leaves out the shuffled reference (its permutations
      depend on how the layers were split into tasks).

  curvature   60_curvature.py output vs a reference JSON, leaf by leaf (plus the two subspaces with
              --ref-charts EE_NPZ OBJ_NPZ: scaler mean, scale and Q arrays)

  top-panel   panels/fig2_top.npz vs the arrays `vlageom.position.top_panel` builds from reference
              JSONs: --ref-probe EE_JSON OBJ_JSON --ref-procrustes FILE --ref-curvature FILE

  grid        panels/fig2_{ee,target}_grid.npz vs the arrays built with a reference subspace fit
              (--ref-chart NPZ with mean, scale_, B, A, center, scale, srt) and the paths under
              --shapes; prints the largest difference of every array and passes only when all
              are identical (--atol to accept a tolerance)

  render      rendered files vs reference files: --new STEM --ref STEM compares STEM.png byte for
              byte and STEM.pdf with its /CreationDate removed (e.g. --new
              $VLAGEOM_FIGURES/fig2_top --ref <paper top-row file without extension>)

Exit status 0 when everything compared is equal.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

FAILS: list[str] = []
NOTES: list[str] = []


def leaf_cmp(name, x, y, skip=()):
    """Compare JSON values; dict keys only in the reference are listed in NOTES."""
    if isinstance(y, dict):
        if not isinstance(x, dict):
            FAILS.append(f"{name}: not a dict in the new file")
            return 0
        n = 0
        for k, v in y.items():
            if k in skip:
                continue
            if k not in x:
                NOTES.append(f"{name}.{k}")
                continue
            n += leaf_cmp(f"{name}.{k}", x[k], v, skip)
        return n
    if isinstance(y, list):
        if not isinstance(x, list) or len(x) != len(y):
            FAILS.append(f"{name}: list length differs")
            return 0
        return sum(leaf_cmp(f"{name}[{i}]", u, v, skip) for i, (u, v) in enumerate(zip(x, y)))
    same = (x == y) or (isinstance(x, float) and isinstance(y, float)
                        and math.isnan(x) and math.isnan(y))
    if not same:
        FAILS.append(f"{name}: new {x!r} ref {y!r}")
    return 1


def run_probe(a):
    for cell in a.cells:
        s, L = cell.split(":")
        fn = f"{s}_L{int(L):02d}.json"
        x = json.loads((Path(a.new) / fn).read_text())
        y = json.loads((Path(a.ref) / fn).read_text())
        n0 = len(FAILS)
        n = leaf_cmp(fn, {"grains": x["grains"], "cells": x["cells"]},
                     {"grains": y["grains"], "cells": y["cells"]})
        print(f"{fn}: {n} values compared, {len(FAILS) - n0} differ")


def run_procrustes(a):
    files = a.files or sorted(p.name for p in Path(a.new).glob("*.json"))
    skip = ("shuffle_resid",) if a.skip_shuffle else ()
    for fn in files:
        x = json.loads((Path(a.new) / fn).read_text())
        y = json.loads((Path(a.ref) / fn).read_text())
        n0 = len(FAILS)
        n = leaf_cmp(fn, x, y, skip)
        print(f"{fn}: {n} values compared, {len(FAILS) - n0} differ")


def run_curvature(a):
    x, y = json.loads(Path(a.new).read_text()), json.loads(Path(a.ref).read_text())
    n0 = len(FAILS)
    n = leaf_cmp("curvature", x, y)
    print(f"curvature: {n} values compared, {len(FAILS) - n0} differ")
    if a.ref_charts:
        newdir = Path(a.new).parent
        for var, ref in zip(("ee", "obj"), a.ref_charts):
            zn, zr = np.load(newdir / f"chart_{var}.npz"), np.load(ref)
            for k in ("mean", "scale", "Q", "n"):
                same = np.array_equal(zn[k], zr[k])
                print(f"  chart_{var}.{k}: {'equal' if same else 'DIFF'}")
                if not same:
                    FAILS.append(f"chart_{var}.{k}")


def cmp_arrays(tag, new: dict, ref: dict, atol=0.0):
    for k in sorted(ref):
        if k == "src":
            continue
        if k not in new:
            FAILS.append(f"{tag}:{k} missing")
            continue
        x, y = np.asarray(new[k]), np.asarray(ref[k])
        if x.shape != y.shape:
            FAILS.append(f"{tag}:{k} shape {x.shape} vs {y.shape}")
            continue
        if x.dtype.kind in "fc":
            d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) \
                if x.size else 0.0
            ok = d <= atol
            print(f"  {tag}:{k} {x.dtype}{x.shape} max|diff| {d:.3g}{'' if ok else '  DIFF'}")
        else:
            ok = np.array_equal(x, y)
            print(f"  {tag}:{k} {x.dtype}{x.shape} {'equal' if ok else 'DIFF'}")
        if not ok:
            FAILS.append(f"{tag}:{k}")


def run_top(a):
    from vlageom.position import top_panel
    probe = {"ee": json.loads(Path(a.ref_probe[0]).read_text()),
             "obj": json.loads(Path(a.ref_probe[1]).read_text())}
    ref = top_panel(probe, json.loads(Path(a.ref_procrustes).read_text()),
                    json.loads(Path(a.ref_curvature).read_text()))
    with np.load(a.panel) as z:
        new = {k: z[k] for k in z.files}
    cmp_arrays("fig2_top", new, ref)


def run_grid(a):
    from vlageom.position import grid_panel
    z = np.load(a.ref_chart)
    chart = {k: z[k] for k in z.files if k != "fitstats_json"}
    chart["cell"] = a.cell
    ref = grid_panel(chart, a.shapes, a.variable, a.house)
    with np.load(a.panel) as zz:
        new = {k: zz[k] for k in zz.files}
    cmp_arrays(Path(a.panel).stem, new, ref, a.atol)


def run_render(a):
    import re
    date = re.compile(rb"/CreationDate \(D:[^)]*\)")
    for ext in ("png", "pdf"):
        x = Path(f"{a.new}.{ext}").read_bytes()
        y = Path(f"{a.ref}.{ext}").read_bytes()
        if ext == "pdf":
            x, y = date.sub(b"/CreationDate ()", x), date.sub(b"/CreationDate ()", y)
        same = x == y
        print(f"{Path(a.new).name}.{ext}: {'identical' if same else 'DIFFERENT'}"
              + (" (apart from /CreationDate)" if ext == "pdf" and same else ""))
        if not same:
            FAILS.append(f"{a.new}.{ext}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("probe")
    s.add_argument("--new", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--cells", nargs="+", default=["hdr_txt:20", "txt_instr:21"])
    s = sub.add_parser("procrustes")
    s.add_argument("--new", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--files", nargs="+", default=None)
    s.add_argument("--skip-shuffle", action="store_true")
    s = sub.add_parser("curvature")
    s.add_argument("--new", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--ref-charts", nargs=2, default=None)
    s = sub.add_parser("top-panel")
    s.add_argument("--panel", required=True)
    s.add_argument("--ref-probe", nargs=2, required=True)
    s.add_argument("--ref-procrustes", required=True)
    s.add_argument("--ref-curvature", required=True)
    s = sub.add_parser("grid")
    s.add_argument("--panel", required=True)
    s.add_argument("--ref-chart", required=True)
    s.add_argument("--shapes", required=True)
    s.add_argument("--variable", choices=["ee", "obj"], required=True)
    s.add_argument("--cell", required=True, help="e.g. hdr_txt@L20")
    s.add_argument("--house", type=int, default=44)
    s.add_argument("--atol", type=float, default=0.0)
    s = sub.add_parser("render")
    s.add_argument("--new", required=True, help="path of the rendered files without extension")
    s.add_argument("--ref", required=True, help="path of the reference files without extension")
    a = ap.parse_args(argv)
    {"probe": run_probe, "procrustes": run_procrustes, "curvature": run_curvature,
     "top-panel": run_top, "grid": run_grid, "render": run_render}[a.cmd](a)
    if NOTES:
        keys = sorted(set(n.split(".")[-1] for n in NOTES))
        print(f"reference keys not in the new files (not compared): {keys}")
    for f in FAILS[:30]:
        print("  DIFF", f)
    print("EQUAL" if not FAILS else f"DIFFERENT ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
