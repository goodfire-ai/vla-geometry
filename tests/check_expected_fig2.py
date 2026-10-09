#!/usr/bin/env python
"""Compare the Figure 2 outputs with the values recorded in tests/expected_fig2.json.

Checked, without any reference data:

  probe       every value under "grains" and "cells" of $VLAGEOM_DATA/position/probe/
              hdr_txt_L20.json and txt_instr_L21.json (stage 30)
  procrustes  MD5 of position/procrustes/procrustes_q0.json (stage 40 --merge; --skip-procrustes
              when only some layers were run)
  curvature   MD5 of position/curvature/curvature_q0.json (stage 60)
  panels      every array of $VLAGEOM_DATA/panels/fig2_top.npz, fig2_ee_grid.npz and
              fig2_target_grid.npz except the path record `src`: dtype, shape and the SHA-1 of its
              bytes (np.ascontiguousarray(a).tobytes())
  renders     MD5 of $VLAGEOM_FIGURES/fig2_{top,ee_grid,target_grid}.png, and of the three PDFs
              with their /CreationDate removed; they match only with the Product Sans font
              (VLAGEOM_FONT_DIR) and the pinned library versions (--skip-render to leave them out)

The recorded values are those of the CPU stages run on the paper's stores on an x86-64 CPU with
AVX-512 (8 BLAS threads). The renders are the paper's files: the PNGs have the paper's bytes and
the PDFs equal the paper's apart from the creation date. Exit status 0 when everything checked
equals the expected values.

  python tests/check_expected_fig2.py [--data DIR] [--panels DIR] [--figures DIR]
      [--skip-procrustes] [--skip-render]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from vlageom import paths

EXPECTED = Path(__file__).resolve().parent / "expected_fig2.json"
CREATION_DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")
PANEL_FILES = ("fig2_top", "fig2_ee_grid", "fig2_target_grid")
PROBE_FILES = ("hdr_txt_L20.json", "txt_instr_L21.json")


def array_record(a) -> dict:
    a = np.asarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()}


def current(data: Path, panels: Path, figures: Path, skip_proc: bool, skip_render: bool) -> dict:
    out = {"probe": {}}
    for fn in PROBE_FILES:
        d = json.loads((data / "probe" / fn).read_text())
        out["probe"][fn] = {"grains": d["grains"], "cells": d["cells"]}
    if not skip_proc:
        p = data / "procrustes" / "procrustes_q0.json"
        out["procrustes"] = hashlib.md5(p.read_bytes()).hexdigest()
    p = data / "curvature" / "curvature_q0.json"
    out["curvature"] = hashlib.md5(p.read_bytes()).hexdigest()
    out["panels"] = {}
    for name in PANEL_FILES:
        with np.load(panels / f"{name}.npz", allow_pickle=False) as z:
            out["panels"][name] = {k: array_record(z[k]) for k in z.files if k != "src"}
    if not skip_render:
        r = {}
        for name in PANEL_FILES:
            r[f"{name}.png"] = hashlib.md5((figures / f"{name}.png").read_bytes()).hexdigest()
            pdf = CREATION_DATE.sub(b"/CreationDate ()", (figures / f"{name}.pdf").read_bytes())
            r[f"{name}.pdf"] = hashlib.md5(pdf).hexdigest()
        out["renders"] = r
    return out


def _walk(exp, cur, path=""):
    """[(path, expected, current)] of the entries that differ."""
    if isinstance(exp, dict) and isinstance(cur, dict):
        out = []
        for k in sorted(set(exp) | set(cur)):
            out += _walk(exp.get(k, "<missing>"), cur.get(k, "<missing>"), f"{path}/{k}")
        return out
    if isinstance(exp, list) and isinstance(cur, list) and len(exp) == len(cur):
        out = []
        for i, (e, c) in enumerate(zip(exp, cur)):
            out += _walk(e, c, f"{path}[{i}]")
        return out
    return [] if exp == cur else [(path, exp, cur)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--data", type=Path, default=None,
                    help="stage outputs (default $VLAGEOM_DATA/position)")
    ap.add_argument("--panels", type=Path, default=None,
                    help="panels directory (default $VLAGEOM_DATA/panels)")
    ap.add_argument("--figures", type=Path, default=None,
                    help="rendered panels (default $VLAGEOM_FIGURES)")
    ap.add_argument("--skip-procrustes", action="store_true",
                    help="do not check the merged Procrustes file")
    ap.add_argument("--skip-render", action="store_true", help="do not check the renders")
    ap.add_argument("--expected", type=Path, default=EXPECTED, help="expected-values file")
    ap.add_argument("--write", action="store_true",
                    help="write the current outputs to --expected instead of checking them")
    a = ap.parse_args(argv)

    cur = current(a.data or paths.tile_dir("position"), a.panels or paths.panels_dir(),
                  a.figures or paths.figures_dir(), a.skip_procrustes, a.skip_render)
    if a.write:
        a.expected.write_text(json.dumps(cur, indent=1) + "\n")
        print(f"wrote {a.expected}")
        return 0

    exp = json.loads(a.expected.read_text())
    if a.skip_render:
        exp.pop("renders", None)
    if a.skip_procrustes:
        exp.pop("procrustes", None)
    diffs = _walk(exp, cur)
    for section in exp:
        n_bad = sum(1 for p, *_ in diffs if p == f"/{section}" or p.startswith(f"/{section}/"))
        print(f"{section:12s} {'as expected' if n_bad == 0 else f'{n_bad} DIFFERENT'}")
    for p, e, c in diffs[:40]:
        print(f"  DIFF {p}: expected {e!r}, got {c!r}")
    print("AS EXPECTED" if not diffs else f"DIFFERENT ({len(diffs)})")
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
