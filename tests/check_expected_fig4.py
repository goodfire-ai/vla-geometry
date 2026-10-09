#!/usr/bin/env python
"""Compare the Figure 4 outputs with the values recorded in tests/expected_fig4.json.

Checked, without any reference data:

  cache    every array of $VLAGEOM_DATA/contact/natural_cache.npz (stage 20) except `harvest`
           (the input directory): dtype, shape and the SHA-1 of its bytes
           (np.ascontiguousarray(a).tobytes()); reads the whole 5.8 GB file (--skip-cache to
           leave it out)
  pool     every array of contact/offschedule_L24.npz (stage 40) except `subdir`
  probe    the cells and the majority accuracy of contact/natural_probe.json (stage 50)
  scores   n_pts, n_eps, counts and every value under `variants` of
           contact/detection_heldout.json (stage 60)
  panel    every array of $VLAGEOM_DATA/panels/contact.npz (stage 70)
  renders  MD5 of $VLAGEOM_FIGURES/fig4a.png and fig4b.png, and of the two PDFs with their
           /CreationDate removed; they match only with the Product Sans font (VLAGEOM_FONT_DIR)
           and the pinned library versions (--skip-render to leave them out)

The recorded values are those of the CPU stages run on the paper's stores. Exit status 0 when
everything checked equals the expected values.

  python tests/check_expected_fig4.py [--data DIR] [--panels DIR] [--figures DIR] [--skip-cache]
                                      [--skip-render]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from vlageom import contact as C
from vlageom import paths

EXPECTED = Path(__file__).resolve().parent / "expected_fig4.json"
CREATION_DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")
PANELS = ("a", "b")


def array_record(a) -> dict:
    a = np.asarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()}


def npz_records(path: Path, skip=()) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: array_record(z[k]) for k in z.files if k not in skip}


def current(data: Path, panels: Path, figures: Path, skip_cache: bool, skip_render: bool) -> dict:
    out = {}
    if not skip_cache:
        out["cache"] = npz_records(data / "natural_cache.npz", skip=("harvest",))
    out["pool"] = npz_records(data / f"offschedule_L{C.PROBE_LAYER}.npz", skip=("subdir",))
    probe = json.loads((data / "natural_probe.json").read_text())
    out["probe"] = {"cells": probe["cells"], "majority_acc": probe["majority_acc"],
                    "n": probe["n"], "n_houses": probe["n_houses"]}
    scores = json.loads((data / "detection_heldout.json").read_text())
    out["scores"] = {k: scores[k] for k in ("n_pts", "n_eps", "counts", "variants")}
    out["panel"] = npz_records(panels / "contact.npz")
    if not skip_render:
        r = {}
        for p in PANELS:
            r[f"fig4{p}.png"] = hashlib.md5((figures / f"fig4{p}.png").read_bytes()).hexdigest()
            pdf = CREATION_DATE.sub(b"/CreationDate ()", (figures / f"fig4{p}.pdf").read_bytes())
            r[f"fig4{p}.pdf"] = hashlib.md5(pdf).hexdigest()
        out["renders"] = r
    return out


def _walk(exp, cur, path=""):
    """[(path, expected, current)] of the entries that differ."""
    if isinstance(exp, dict) and isinstance(cur, dict):
        out = []
        for k in sorted(set(exp) | set(cur)):
            out += _walk(exp.get(k, "<missing>"), cur.get(k, "<missing>"), f"{path}/{k}")
        return out
    if isinstance(exp, float) and isinstance(cur, float) and exp != exp and cur != cur:
        return []
    return [] if exp == cur else [(path, exp, cur)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--data", type=Path, default=None,
                    help="stage outputs (default $VLAGEOM_DATA/contact)")
    ap.add_argument("--panels", type=Path, default=None,
                    help="panels directory (default $VLAGEOM_DATA/panels)")
    ap.add_argument("--figures", type=Path, default=None,
                    help="rendered panels (default $VLAGEOM_FIGURES)")
    ap.add_argument("--skip-cache", action="store_true", help="do not hash the natural cache")
    ap.add_argument("--skip-render", action="store_true", help="do not check the renders")
    ap.add_argument("--expected", type=Path, default=EXPECTED, help="expected-values file")
    ap.add_argument("--write", action="store_true",
                    help="write the current outputs to --expected instead of checking them")
    a = ap.parse_args(argv)

    cur = current(a.data or paths.tile_dir(C.TILE), a.panels or paths.panels_dir(),
                  a.figures or paths.figures_dir(), a.skip_cache, a.skip_render)
    if a.write:
        a.expected.write_text(json.dumps(cur, indent=1) + "\n")
        print(f"wrote {a.expected}")
        return 0

    exp = json.loads(a.expected.read_text())
    if a.skip_cache:
        exp.pop("cache", None)
    if a.skip_render:
        exp.pop("renders", None)
    diffs = _walk(exp, cur)
    for section in exp:
        n_bad = sum(1 for p, *_ in diffs if p == f"/{section}" or p.startswith(f"/{section}/"))
        print(f"{section:8s} {'as expected' if n_bad == 0 else f'{n_bad} DIFFERENT'}")
    for p, e, c in diffs:
        print(f"  DIFF {p}: expected {e!r}, got {c!r}")
    print("AS EXPECTED" if not diffs else f"DIFFERENT ({len(diffs)})")
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
