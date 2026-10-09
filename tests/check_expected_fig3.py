#!/usr/bin/env python
"""Compare the Figure 3 outputs with the values recorded in tests/expected_fig3.json.

Checked, without any reference data:

  panels   every array of $VLAGEOM_DATA/panels/fig3{a_erasure,b_cutoff,c_cosine,d_plane}.npz
           (stage 90) except the input record `src`: dtype, shape and the SHA-1 of its bytes
           (np.ascontiguousarray(a).tobytes())
  table    n, MAE and R^2 of every family, boundary and range of
           $VLAGEOM_DATA/distance/cutoff_table_last_tok_L21.json (stage 70)
  plant    every value of the per-boundary rows and the cosines to the planted direction in
           $VLAGEOM_DATA/distance/plant_cosines.json (stage 80)
  renders  MD5 of $VLAGEOM_FIGURES/fig3{a,b,c,d}.png, and of the four PDFs with their
           /CreationDate removed; they match only with the Product Sans font (VLAGEOM_FONT_DIR)
           and the pinned library versions (--skip-render to leave them out)

The recorded values are those the paper's Figure 3 panels were drawn from; the CPU stages give
them from the paper's packed stores with the pinned library versions on an x86-64 CPU with
AVX-512, at the BLAS thread counts the stages set. Sections whose files are missing are reported
and count as different (--only to check some sections). Exit status 0 when everything checked
equals the expected values.

  python tests/check_expected_fig3.py [--data DIR] [--panels DIR] [--figures DIR] [--skip-render]
                                      [--only panels table plant renders]
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

EXPECTED = Path(__file__).resolve().parent / "expected_fig3.json"
CREATION_DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")
PANEL_FILES = {"a": "fig3a_erasure.npz", "b": "fig3b_cutoff.npz", "c": "fig3c_cosine.npz",
               "d": "fig3d_plane.npz"}
SECTIONS = ("panels", "table", "plant", "renders")


def array_record(a) -> dict:
    a = np.asarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()}


def _nan_safe(x):
    """JSON-comparable copy (NaN as the string "nan")."""
    if isinstance(x, dict):
        return {k: _nan_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nan_safe(v) for v in x]
    if isinstance(x, float) and x != x:
        return "nan"
    return x


def current(data: Path, panels: Path, figures: Path, sections) -> dict:
    out = {}
    if "panels" in sections:
        out["panels"] = {}
        for name in PANEL_FILES.values():
            f = panels / name
            if not f.exists():
                out["panels"][name] = "<missing>"
                continue
            with np.load(f, allow_pickle=False) as z:
                out["panels"][name] = {k: array_record(z[k]) for k in z.files if k != "src"}
    if "table" in sections:
        f = data / "cutoff_table_last_tok_L21.json"
        if f.exists():
            tab = json.loads(f.read_text())["table"]
            out["table"] = _nan_safe({c: {reg: e["metrics"] for reg, e in regs.items()}
                                      for c, regs in tab.items()})
        else:
            out["table"] = "<missing>"
    if "plant" in sections:
        f = data / "plant_cosines.json"
        if f.exists():
            pl = json.loads(f.read_text())
            out["plant"] = _nan_safe({"split": pl["split"], "round": pl["round"],
                                      "cos_to_planted_axis": pl["cos_to_planted_axis"]})
        else:
            out["plant"] = "<missing>"
    if "renders" in sections:
        r = {}
        for p in PANEL_FILES:
            png, pdf = figures / f"fig3{p}.png", figures / f"fig3{p}.pdf"
            r[f"fig3{p}.png"] = (hashlib.md5(png.read_bytes()).hexdigest() if png.exists()
                                 else "<missing>")
            r[f"fig3{p}.pdf"] = (hashlib.md5(CREATION_DATE.sub(b"/CreationDate ()",
                                                               pdf.read_bytes())).hexdigest()
                                 if pdf.exists() else "<missing>")
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
                    help="distance stage outputs (default $VLAGEOM_DATA/distance)")
    ap.add_argument("--panels", type=Path, default=None,
                    help="panels directory (default $VLAGEOM_DATA/panels)")
    ap.add_argument("--figures", type=Path, default=None,
                    help="rendered panels (default $VLAGEOM_FIGURES)")
    ap.add_argument("--skip-render", action="store_true", help="do not check the renders")
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=None,
                    help="check only these sections")
    ap.add_argument("--expected", type=Path, default=EXPECTED, help="expected-values file")
    ap.add_argument("--write", action="store_true",
                    help="write the current outputs to --expected instead of checking them")
    a = ap.parse_args(argv)

    sections = list(a.only or SECTIONS)
    if a.skip_render and "renders" in sections:
        sections.remove("renders")
    cur = current(a.data or paths.tile_dir("distance"), a.panels or paths.panels_dir(),
                  a.figures or paths.figures_dir(), sections)
    if a.write:
        a.expected.write_text(json.dumps(cur, indent=1) + "\n")
        print(f"wrote {a.expected}")
        return 0

    exp = {k: v for k, v in json.loads(a.expected.read_text()).items() if k in sections}
    diffs = _walk(exp, cur)
    for section in exp:
        n_bad = sum(1 for p, *_ in diffs if p == f"/{section}" or p.startswith(f"/{section}/"))
        print(f"{section:12s} {'as expected' if n_bad == 0 else f'{n_bad} DIFFERENT'}")
    for p, e, c in diffs[:50]:
        print(f"  DIFF {p}: expected {e!r}, got {c!r}")
    print("AS EXPECTED" if not diffs else f"DIFFERENT ({len(diffs)})")
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
