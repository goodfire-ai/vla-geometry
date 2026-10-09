#!/usr/bin/env python
"""Compare the Figure 6 outputs with the values recorded in tests/expected_fig6.json.

Checked, without any reference data:

  panels   every array of $VLAGEOM_DATA/panels/fig6_a.npz (except the path record `src`) and
           fig6_bcd.npz: dtype, shape and the SHA-1 of its bytes
  metrics  the values of panels/fig6_a_metrics.json (except the input path `npz`; the largest H1
           persistence only when it was computed, which needs ripser)
  numbers  every value of $VLAGEOM_DATA/wrist_steer/fig6_numbers.json (stage 80)
  operator SHA-1 of every array of $VLAGEOM_DATA/wrist_steer/operator_L9.npz (stage 40;
           --skip-operator to leave it out)
  renders  MD5 of $VLAGEOM_FIGURES/fig6_bcd.png, and of fig6_a.pdf and fig6_bcd.pdf with their
           /CreationDate removed; they match only with the Product Sans font (VLAGEOM_FONT_DIR) and
           the pinned library versions (--skip-render to leave them out)

The recorded values are those of the CPU stages run on the paper's stores (natural cloud,
orientation sweeps, chart, steering runs and the house-0 roll sweep). Exit status 0 when everything
checked equals the expected values.

  python tests/check_expected_fig6.py [--panels DIR] [--numbers FILE] [--operator FILE]
      [--figures DIR] [--skip-operator] [--skip-render]
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

EXPECTED = Path(__file__).resolve().parent / "expected_fig6.json"
CREATION_DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")


def array_record(a) -> dict:
    a = np.asarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()}


def current(panels: Path, numbers: Path, figures: Path, skip_render: bool,
            operator: Path | None) -> dict:
    out = {"panels": {}}
    if operator is not None:
        with np.load(operator, allow_pickle=False) as z:
            out["operator"] = {k: array_record(z[k]) for k in z.files}
    for name in ("fig6_a", "fig6_bcd"):
        with np.load(panels / f"{name}.npz", allow_pickle=False) as z:
            out["panels"][name] = {k: array_record(z[k]) for k in z.files if k != "src"}
    m = json.loads((panels / "fig6_a_metrics.json").read_text())
    m.pop("npz", None)
    out["metrics"] = m
    out["numbers"] = json.loads(numbers.read_text())
    if not skip_render:
        out["renders"] = {
            "fig6_bcd.png": hashlib.md5((figures / "fig6_bcd.png").read_bytes()).hexdigest(),
            **{n: hashlib.md5(CREATION_DATE.sub(b"", (figures / n).read_bytes())).hexdigest()
               for n in ("fig6_a.pdf", "fig6_bcd.pdf")}}
    return out


def leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--panels", type=Path, default=None, help="default $VLAGEOM_DATA/panels")
    ap.add_argument("--numbers", type=Path, default=None,
                    help="default $VLAGEOM_DATA/wrist_steer/fig6_numbers.json")
    ap.add_argument("--figures", type=Path, default=None, help="default $VLAGEOM_FIGURES")
    ap.add_argument("--operator", type=Path, default=None,
                    help="default $VLAGEOM_DATA/wrist_steer/operator_L9.npz")
    ap.add_argument("--skip-operator", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--expected", type=Path, default=EXPECTED)
    ap.add_argument("--write", action="store_true",
                    help="write the current outputs to --expected instead of checking them")
    a = ap.parse_args(argv)
    operator = None if a.skip_operator else (a.operator or paths.tile_dir("wrist_steer", "operator_L9.npz"))
    cur = current(a.panels or paths.panels_dir(),
                  a.numbers or paths.tile_dir("wrist_steer", "fig6_numbers.json"),
                  a.figures or paths.figures_dir(), a.skip_render, operator)
    if a.write:
        a.expected.write_text(json.dumps(cur, indent=1) + "\n")
        print(f"wrote {a.expected}")
        return 0
    exp = json.loads(a.expected.read_text())
    if a.skip_render:
        exp.pop("renders", None)
    if a.skip_operator:
        exp.pop("operator", None)
    if cur["metrics"].get("H1_top_persistence", 0) is None:
        # written as None by stage 70 when ripser is not installed; no figure or number uses it
        print("  note: H1_top_persistence not computed (ripser not installed); not compared")
        exp.get("metrics", {}).pop("H1_top_persistence", None)
        cur["metrics"].pop("H1_top_persistence")
    le, lc = dict(leaves(exp)), dict(leaves(json.loads(json.dumps(cur))))
    bad = 0
    for k in sorted(set(le) | set(lc)):
        if k not in lc:
            print(f"  MISSING  {k}")
        elif k not in le:
            print(f"  EXTRA    {k} (not in the expected file)")
        elif le[k] != lc[k]:
            print(f"  DIFF     {k}: {lc[k]!r} (expected {le[k]!r})")
        else:
            continue
        bad += 1
    for part in ("panels", "metrics", "numbers", "operator", "renders"):
        keys = [k for k in le if k.startswith(f"/{part}/")]
        if keys:
            n = sum(1 for k in keys if lc.get(k) == le[k])
            print(f"{part:8s} {n}/{len(keys)} values as expected")
    print("AS EXPECTED" if bad == 0 else f"DIFFERENT ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
