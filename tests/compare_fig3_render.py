#!/usr/bin/env python
"""Compare the Figure 3 panel renders (fig3a-d .png / .pdf) with reference files.

By default the script renders the requested panels with figures/fig3_distance.py (in a separate
process) from a panels directory into a temporary directory; --rendered DIR compares existing
renders instead. Per panel and format:

  PNG   identical bytes (md5); otherwise the number of differing pixels, their bounding box and
        the largest channel difference (tests/compare_fig1_render.compare_png)
  PDF   identical bytes apart from the /CreationDate entry; otherwise (with PyMuPDF installed)
        the text spans and images that differ (tests/compare_fig1_render.compare_pdf)

References are given per panel as PANEL=FILE, e.g. --ref-png a=<file> --ref-pdf a=<file>.
Text is rendered with the font `vlageom.plot.style.apply_style` finds: set VLAGEOM_FONT_DIR to the
Product Sans files to compare against the paper's panels.

  python tests/compare_fig3_render.py --ref-png a=<a.png> b=<b.png> c=<c.png> d=<d.png> \
      --ref-pdf a=<a.pdf> b=<b.pdf> c=<c.pdf> d=<d.pdf> [--panels DIR | --rendered DIR]

Exit status 0 when every comparison is identical.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_fig1_render import compare_pdf, compare_png  # noqa: E402

FIG3 = HERE.parent / "figures" / "fig3_distance.py"
PANELS = ("a", "b", "c", "d")


def _pairs(items):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        if k not in PANELS or not v:
            raise SystemExit(f"expected PANEL=FILE with PANEL in a-d, got {s!r}")
        out[k] = Path(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--ref-png", nargs="+", default=None, metavar="PANEL=FILE")
    ap.add_argument("--ref-pdf", nargs="+", default=None, metavar="PANEL=FILE")
    ap.add_argument("--panels", type=Path, default=None,
                    help="panels directory to render from (default: $VLAGEOM_DATA/panels)")
    ap.add_argument("--rendered", type=Path, default=None, metavar="DIR",
                    help="compare existing renders DIR/fig3<panel>.png/.pdf instead of rendering")
    a = ap.parse_args(argv)
    ref_png, ref_pdf = _pairs(a.ref_png), _pairs(a.ref_pdf)
    panels = sorted(set(ref_png) | set(ref_pdf))
    if not panels:
        ap.error("give at least one --ref-png or --ref-pdf")
    missing = [str(p) for p in (*ref_png.values(), *ref_pdf.values()) if not p.is_file()]
    if missing:
        ap.error(f"reference file(s) not found: {missing}")

    ok = True
    with tempfile.TemporaryDirectory(prefix="fig3_compare_") as tmp:
        out = a.rendered
        if out is None:
            out = Path(tmp)
            cmd = [sys.executable, str(FIG3), "--panel", *panels, "--out-dir", str(out)]
            if a.panels is not None:
                cmd += ["--panels", str(a.panels)]
            print("render:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        for p in panels:
            print(f"--- panel {p}")
            if p in ref_png:
                ok = compare_png(out / f"fig3{p}.png", ref_png[p]) and ok
            if p in ref_pdf:
                ok = compare_pdf(out / f"fig3{p}.pdf", ref_pdf[p]) and ok
    print("RESULT", "IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
