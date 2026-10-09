#!/usr/bin/env python
"""Compare the outputs of Figures 1-7 with the values recorded in tests/expected_fig<N>.json.

Figures 2-7 are checked by tests/check_expected_fig<N>.py (see each script for what it checks and
for its own options); this script runs them in turn after the Figure 1 check. For Figure 1 it
checks, without any reference data:

  panels   every array of $VLAGEOM_DATA/panels/{gripper,wrist,target,distance}.npz except the path
           records `src` and `files`: dtype, shape and the SHA-1 of its bytes
           (np.ascontiguousarray(a).tobytes()); arrays missing from the file or not listed in the
           expected file are reported
  scores   per-axis R^2 of the target probe (target/probe/last_tok_L21.json) and the two R^2 of
           the distance probes (distance/ruler_probes.json), exactly
  png      MD5 of the rendered figure ($VLAGEOM_FIGURES/fig1.png); it matches only with the
           Product Sans font (VLAGEOM_FONT_DIR) and the pinned library versions (--skip-png to
           leave it out)

Exit status 0 when everything checked equals the expected values.

  python tests/check_expected.py                    # Figures 1-7
  python tests/check_expected.py --fig 1 3          # selected figures
  python tests/check_expected.py --skip-render      # without the renders (no fonts needed)
  python tests/check_expected.py --fig 1 [--panels DIR] [--png FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import paths

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected_fig1.json"
FIGURES = (1, 2, 3, 4, 5, 6, 7)
TILES = ("gripper", "wrist", "target", "distance")
PATH_KEYS = ("src", "files")          # input locations; they differ between machines


def array_record(a: np.ndarray) -> dict:
    a = np.asarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()}


def panel_records(panels_dir: Path) -> dict:
    out = {}
    for tile in TILES:
        with np.load(panels_dir / f"{tile}.npz", allow_pickle=False) as z:
            out[tile] = {k: array_record(z[k]) for k in z.files if k not in PATH_KEYS}
    return out


def score_records(target_probe: Path, distance_scores: Path) -> dict:
    t = json.loads(target_probe.read_text())
    d = json.loads(distance_scores.read_text())
    return {"target_r2": {k: t["targets"][k]["r2"] for k in ("fwd", "lat", "up")},
            "target_note_r2": t["note_r2"],
            "distance_r2_m": d["scores"]["far15"]["r2_above15"],
            "distance_r2_cm": d["scores"]["cm15"]["r2_below15"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--fig", type=int, nargs="+", choices=FIGURES, default=list(FIGURES),
                    help="figures to check (default: all)")
    ap.add_argument("--skip-render", "--skip-png", dest="skip_png", action="store_true",
                    help="do not check the rendered figures")
    ap.add_argument("--skip-cache", action="store_true",
                    help="Figure 4: do not hash the 5.8 GB natural cache")
    ap.add_argument("--write", action="store_true",
                    help="write the current outputs to the expected-values files instead of "
                         "checking them")
    ap.add_argument("--panels", type=Path, default=None,
                    help="Figure 1 panels directory (default $VLAGEOM_DATA/panels)")
    ap.add_argument("--target-probe", type=Path, default=None,
                    help="default $VLAGEOM_DATA/target/probe/last_tok_L21.json")
    ap.add_argument("--distance-scores", type=Path, default=None,
                    help="default $VLAGEOM_DATA/distance/ruler_probes.json")
    ap.add_argument("--png", type=Path, default=None,
                    help="Figure 1 render (default $VLAGEOM_FIGURES/fig1.png)")
    ap.add_argument("--expected", type=Path, default=EXPECTED,
                    help="Figure 1 expected-values file")
    a = ap.parse_args(argv)

    results = {}
    for fig in sorted(set(a.fig)):
        print(f"==== Figure {fig}", flush=True)
        try:
            if fig == 1:
                results[fig] = check_fig1(a)
                continue
            spec = importlib.util.spec_from_file_location(f"check_expected_fig{fig}",
                                                          HERE / f"check_expected_fig{fig}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            args = (["--skip-render"] if a.skip_png else []) + (["--write"] if a.write else [])
            if fig == 4 and a.skip_cache:
                args.append("--skip-cache")
            results[fig] = mod.main(args)
        except (FileNotFoundError, KeyError) as e:
            print(f"  MISSING output: {e}")
            results[fig] = 1
    print("====")
    for fig, rc in results.items():
        print(f"Figure {fig}: {'as expected' if rc == 0 else 'DIFFERENT'}")
    bad = [f for f, rc in results.items() if rc != 0]
    print("ALL AS EXPECTED" if not bad else f"DIFFERENT: Figure(s) {bad}")
    return 0 if not bad else 1


def check_fig1(a) -> int:
    panels_dir = a.panels or paths.panels_dir()
    target_probe = a.target_probe or paths.tile_dir("target", "probe", "last_tok_L21.json")
    distance_scores = a.distance_scores or paths.tile_dir("distance", "ruler_probes.json")
    png = a.png or paths.figures_dir() / "fig1.png"

    cur = {"panels": panel_records(panels_dir),
           "scores": score_records(target_probe, distance_scores)}
    if not a.skip_png:
        cur["fig1_png_md5"] = hashlib.md5(png.read_bytes()).hexdigest()

    if a.write:
        a.expected.write_text(json.dumps(cur, indent=1) + "\n")
        print(f"wrote {a.expected}")
        return 0

    exp = json.loads(a.expected.read_text())
    bad = 0
    for tile in TILES:
        e, c = exp["panels"][tile], cur["panels"][tile]
        for k in sorted(set(e) | set(c)):
            if k not in c:
                print(f"  MISSING  {tile}.npz:{k}")
            elif k not in e:
                print(f"  EXTRA    {tile}.npz:{k} (not in the expected file)")
            elif c[k] != e[k]:
                print(f"  DIFF     {tile}.npz:{k}  {c[k]['dtype']}{tuple(c[k]['shape'])}  "
                      f"expected {e[k]['dtype']}{tuple(e[k]['shape'])} sha1 {e[k]['sha1'][:12]}, "
                      f"got {c[k]['sha1'][:12]}")
            else:
                continue
            bad += 1
        n = sum(1 for k in e if c.get(k) == e[k])
        print(f"{tile:9s} {n}/{len(e)} arrays as expected")
    for k, v in exp["scores"].items():
        same = cur["scores"].get(k) == v
        bad += not same
        print(f"{'equal' if same else 'DIFF '}  score {k}: {cur['scores'].get(k)!r}"
              + ("" if same else f" (expected {v!r})"))
    if not a.skip_png:
        same = cur["fig1_png_md5"] == exp["fig1_png_md5"]
        bad += not same
        print(f"{'equal' if same else 'DIFF '}  fig1.png md5 {cur['fig1_png_md5']}"
              + ("" if same else f" (expected {exp['fig1_png_md5']}; needs Product Sans and the "
                                 "pinned libraries)"))
    print("AS EXPECTED" if bad == 0 else f"DIFFERENT ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
