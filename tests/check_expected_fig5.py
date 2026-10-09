#!/usr/bin/env python
"""Compare the Figure 5 outputs with the values recorded in tests/expected_fig5.json.

Checked, without any reference data:

  cache    every array of $VLAGEOM_DATA/phase/natural/cache_ae.npz (stage 20): dtype, shape and the
           SHA-1 of its bytes (np.ascontiguousarray(a).tobytes())
  decoder  the best layer of phase/decoder.json (stage 30) and, at every layer, the balanced
           accuracy and the confusion matrix, exactly
  decode   the composition and the decoder record of phase/freeze/decode_ae.json (stage 50)
  panels   every array of $VLAGEOM_DATA/panels/fig5_{strip,a,b,c,d}.npz (stage 70)
  renders  MD5 of $VLAGEOM_FIGURES/fig5_strip.png, fig5{a,b,c,d}.png, and of the five PDFs with
           their /CreationDate removed; they match only with the Product Sans font
           (VLAGEOM_FONT_DIR) and the pinned library versions (--skip-render to leave them out)

The recorded values are those of the CPU stages run on the paper's stores. --only SECTION ...
checks a subset. Exit status 0 when everything checked equals the expected values.

  python tests/check_expected_fig5.py [--data DIR] [--panels DIR] [--figures DIR] [--skip-render]
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

EXPECTED = Path(__file__).resolve().parent / "expected_fig5.json"
CREATION_DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")
PANELS = ("strip", "a", "b", "c", "d")
STEMS = ("fig5_strip", "fig5a", "fig5b", "fig5c", "fig5d")
SECTIONS = ("cache", "decoder", "decode", "panels", "renders")


def array_record(a) -> dict:
    a = np.asarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()}


def npz_records(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: array_record(z[k]) for k in z.files}


def current(data: Path, panels: Path, figures: Path, sections) -> dict:
    out = {}
    if "cache" in sections:
        out["cache"] = npz_records(data / "natural" / "cache_ae.npz")
    if "decoder" in sections:
        j = json.loads((data / "decoder.json").read_text())
        out["decoder"] = {"best_layer": j["best_layer"],
                          "balanced_accuracy": {L: v["bal"] for L, v in j["per_layer"].items()},
                          "confusion": {L: v["confusion"] for L, v in j["per_layer"].items()}}
    if "decode" in sections:
        j = json.loads((data / "freeze" / "decode_ae.json").read_text())
        out["decode"] = {k: j[k] for k in ("n_files", "n_rows", "layer", "decoder", "composition")}
    if "panels" in sections:
        out["panels"] = {p: npz_records(panels / f"fig5_{p}.npz") for p in PANELS}
    if "renders" in sections:
        r = {}
        for s in STEMS:
            r[f"{s}.png"] = hashlib.md5((figures / f"{s}.png").read_bytes()).hexdigest()
            pdf = CREATION_DATE.sub(b"/CreationDate ()", (figures / f"{s}.pdf").read_bytes())
            r[f"{s}.pdf"] = hashlib.md5(pdf).hexdigest()
        out["renders"] = r
    return out


def _walk(exp, cur, path=""):
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
                    help="stage outputs (default $VLAGEOM_DATA/phase)")
    ap.add_argument("--panels", type=Path, default=None, help="default $VLAGEOM_DATA/panels")
    ap.add_argument("--figures", type=Path, default=None, help="default $VLAGEOM_FIGURES")
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=None)
    ap.add_argument("--skip-render", action="store_true", help="do not check the renders")
    ap.add_argument("--expected", type=Path, default=EXPECTED, help="expected-values file")
    ap.add_argument("--write", action="store_true",
                    help="write the current outputs to --expected instead of checking them")
    a = ap.parse_args(argv)
    sections = [s for s in (a.only or SECTIONS) if not (a.skip_render and s == "renders")]

    cur = current(a.data or paths.tile_dir("phase"), a.panels or paths.panels_dir(),
                  a.figures or paths.figures_dir(), sections)
    if a.write:
        stored = json.loads(a.expected.read_text()) if a.expected.exists() else {}
        stored.update(cur)
        keep = {s: stored[s] for s in SECTIONS if s in stored}
        a.expected.write_text(json.dumps(keep, indent=1) + "\n")
        print(f"wrote {a.expected}")
        return 0
    exp = {k: v for k, v in json.loads(a.expected.read_text()).items() if k in sections}
    diffs = _walk(exp, cur)
    for section in exp:
        n_bad = sum(1 for p, *_ in diffs if p.startswith(f"/{section}/"))
        print(f"{section:8s} {'as expected' if n_bad == 0 else f'{n_bad} DIFFERENT'}")
    for p, e, c in diffs[:40]:
        print(f"  DIFF {p}: expected {e!r}, got {c!r}")
    print("AS EXPECTED" if not diffs else f"DIFFERENT ({len(diffs)})")
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
