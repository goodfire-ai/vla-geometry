#!/usr/bin/env python
"""Compare the Figure 7 outputs with the values recorded in tests/expected_fig7.json.

Checked, without any reference data:

  basis    every array of $VLAGEOM_DATA/gripper_fig7/basis.npz (stage 10): dtype, shape and the
           SHA-1 of its bytes (np.ascontiguousarray(a).tobytes())
  probe    every value of gripper_fig7/probe.json (stage 20)
  panel    the arrays of panels a and b in $VLAGEOM_DATA/panels/gripper_fig7.npz (stage 50);
           panel c's arrays (panels/gripper.npz) are checked with Figure 1
  renders  MD5 of $VLAGEOM_FIGURES/fig7{a,b,c}.png and of the PDFs
           with their /CreationDate removed; they match only with the Product Sans font
           (VLAGEOM_FONT_DIR) and the pinned library versions (--skip-render to leave them out)

The recorded values are those of the CPU stages run on the gripper edit files. Panel d (stage 30 on
the GPU, then stages 40 and 50) is checked by the canary tests/canary_gripper_ksweep.sh. Exit status 0 when everything checked equals the expected values.

  python tests/check_expected_fig7.py [--data DIR] [--panels DIR] [--figures DIR] [--skip-render]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from vlageom import gripper_fig7 as F
from vlageom import paths

EXPECTED = Path(__file__).resolve().parent / "expected_fig7.json"
CREATION_DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")
PANELS = ("a", "b", "c")          # rendered panels with recorded hashes


def array_record(a) -> dict:
    a = np.asarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()}


def current(data: Path, panels: Path, figures: Path, skip_render: bool) -> dict:
    out = {}
    with np.load(data / "basis.npz", allow_pickle=False) as z:
        out["basis"] = {k: array_record(z[k]) for k in z.files}
    out["probe"] = json.loads((data / "probe.json").read_text())
    with np.load(panels / "gripper_fig7.npz", allow_pickle=False) as z:
        out["panel"] = {k: array_record(z[k]) if k in z.files else "<missing>"
                        for k in F.PANEL_KEYS_AB}
    if not skip_render:
        r = {}
        for p in PANELS:
            r[f"fig7{p}.png"] = hashlib.md5((figures / f"fig7{p}.png").read_bytes()).hexdigest()
            pdf = CREATION_DATE.sub(b"/CreationDate ()", (figures / f"fig7{p}.pdf").read_bytes())
            r[f"fig7{p}.pdf"] = hashlib.md5(pdf).hexdigest()
        out["renders"] = r
    return out


def _walk(exp, cur, path=""):
    """[(path, expected, current)] of the entries that differ."""
    if isinstance(exp, dict) and isinstance(cur, dict):
        out = []
        for k in sorted(set(exp) | set(cur)):
            out += _walk(exp.get(k, "<missing>"), cur.get(k, "<missing>"), f"{path}/{k}")
        return out
    return [] if exp == cur else [(path, exp, cur)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--data", type=Path, default=None,
                    help="stage outputs (default $VLAGEOM_DATA/gripper_fig7)")
    ap.add_argument("--panels", type=Path, default=None,
                    help="panels directory (default $VLAGEOM_DATA/panels)")
    ap.add_argument("--figures", type=Path, default=None,
                    help="rendered panels (default $VLAGEOM_FIGURES)")
    ap.add_argument("--skip-render", action="store_true", help="do not check the renders")
    ap.add_argument("--expected", type=Path, default=EXPECTED, help="expected-values file")
    ap.add_argument("--write", action="store_true",
                    help="write the current outputs to --expected instead of checking them")
    a = ap.parse_args(argv)

    cur = current(a.data or paths.tile_dir(F.TILE), a.panels or paths.panels_dir(),
                  a.figures or paths.figures_dir(), a.skip_render)
    if a.write:
        a.expected.write_text(json.dumps(cur, indent=1) + "\n")
        print(f"wrote {a.expected}")
        return 0

    exp = json.loads(a.expected.read_text())
    if a.skip_render:
        exp.pop("renders", None)
    diffs = _walk(exp, cur)
    for section in exp:
        n_bad = sum(1 for p, *_ in diffs if p == f"/{section}" or p.startswith(f"/{section}/"))
        print(f"{section:12s} {'as expected' if n_bad == 0 else f'{n_bad} DIFFERENT'}")
    for p, e, c in diffs:
        print(f"  DIFF {p}: expected {e!r}, got {c!r}")
    print("AS EXPECTED" if not diffs else f"DIFFERENT ({len(diffs)})")
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
