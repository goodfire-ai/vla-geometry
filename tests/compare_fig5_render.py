"""Compare the Figure 5 panel renders (fig5_strip, fig5a .. fig5d; .png and .pdf) with reference
files.

By default the requested panels are rendered with figures/fig5.py (in a separate process) from a
panels directory into a temporary directory; --rendered DIR compares existing renders instead.
Per panel and format:

  PNG   identical bytes (md5); otherwise the number of differing pixels and their bounding box
  PDF   identical bytes apart from the /CreationDate entry; otherwise (with PyMuPDF installed) the
        text spans and images that differ (tests/compare_fig1_render.compare_pdf)

References are given per panel as PANEL=FILE (panels strip, a, b, c, d), e.g.
--ref-pdf c=<file> --ref-png c=<file>. --images-in PANEL=FILE:PAGE checks instead that every image
embedded in the rendered PDF of PANEL is also embedded in page PAGE (1-based) of FILE, by image size
and the MD5 of its decoded pixels (needs PyMuPDF); use it to compare a panel with the page of a
compiled paper that holds it. Text is rendered with the font
`vlageom.plot.style.apply_style` finds: set VLAGEOM_FONT_DIR to the Product Sans files to compare
with the paper's panels.

  python tests/compare_fig5_render.py --ref-pdf a=<a.pdf> c=<c.pdf> --ref-png a=<a.png> \
      [--panels DIR | --rendered DIR]
  python tests/compare_fig5_render.py --images-in strip=<paper.pdf>:7

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

FIG5 = HERE.parent / "figures" / "fig5.py"
STEM = {"strip": "fig5_strip", "a": "fig5a", "b": "fig5b", "c": "fig5c", "d": "fig5d"}


def _refs(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in STEM:
            raise SystemExit(f"unknown panel {k!r} (one of {', '.join(STEM)})")
        out[k] = Path(v)
    return out


def _image_keys(doc, page_index):
    """{(width, height, md5 of the decoded RGB samples)} of the images on one page."""
    import hashlib

    import pymupdf
    out = set()
    for img in doc[page_index].get_images(full=True):
        pix = pymupdf.Pixmap(doc, img[0])
        if pix.n - pix.alpha != 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        out.add((pix.width, pix.height, hashlib.md5(pix.samples).hexdigest()))
    return out


def compare_images(new_pdf: Path, spec: str) -> bool:
    import pymupdf
    path, page = spec.rsplit(":", 1)
    mine = _image_keys(pymupdf.open(str(new_pdf)), 0)
    theirs = _image_keys(pymupdf.open(path), int(page) - 1)
    found = mine & theirs
    print(f"IMG  {len(found)} of {len(mine)} images of {new_pdf.name} found on page {page} "
          f"of {path}")
    for w, h, m in sorted(mine - theirs):
        print(f"     not found: {w}x{h} {m}")
    return found == mine and len(mine) > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--ref-pdf", nargs="*", default=[], metavar="PANEL=FILE")
    ap.add_argument("--ref-png", nargs="*", default=[], metavar="PANEL=FILE")
    ap.add_argument("--images-in", nargs="*", default=[], metavar="PANEL=FILE:PAGE")
    ap.add_argument("--panels", type=Path, default=None, help="panel files (default <data>/panels)")
    ap.add_argument("--rendered", type=Path, default=None, help="directory with existing renders")
    a = ap.parse_args(argv)
    pdfs, pngs = _refs(a.ref_pdf), _refs(a.ref_png)
    imgs = dict(it.split("=", 1) for it in a.images_in)
    want = sorted(set(pdfs) | set(pngs) | set(imgs), key=list(STEM).index)
    if not want:
        ap.error("give at least one --ref-pdf, --ref-png or --images-in")

    tmp = None
    out_dir = a.rendered
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory()
        out_dir = Path(tmp.name)
        cmd = [sys.executable, str(FIG5), "--out-dir", str(out_dir), "--panel", *want]
        if a.panels is not None:
            cmd += ["--panels", str(a.panels)]
        print("render:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    ok = True
    for p in want:
        print(f"--- panel {p}")
        if p in pngs:
            ok &= compare_png(out_dir / f"{STEM[p]}.png", pngs[p])
        if p in pdfs:
            ok &= compare_pdf(out_dir / f"{STEM[p]}.pdf", pdfs[p])
        if p in imgs:
            ok &= compare_images(out_dir / f"{STEM[p]}.pdf", imgs[p])
    print("IDENTICAL" if ok else "DIFFERENT")
    if tmp is not None:
        tmp.cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
