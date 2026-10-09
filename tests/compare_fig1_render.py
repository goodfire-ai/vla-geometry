#!/usr/bin/env python
"""Compare a Figure 1 render with reference files (PNG, and optionally PDF).

By default the script renders Figure 1 with figures/fig1.py from a panels directory (in a separate
process, so the plot style starts from a clean state) into a temporary directory, then compares:

  PNG   identical bytes (md5). Otherwise both images are decoded and the script prints the number
        of differing pixels, their bounding box, the largest channel difference, and the count per
        tile column.
  PDF   identical bytes apart from the /CreationDate entry. Otherwise, with PyMuPDF installed
        (`pip install pymupdf`), the text spans (text, font, size, position) and the embedded
        images of the two files are compared; without it, the offset of the first differing byte
        is printed.

Exit status 0 when every requested comparison is identical, 1 otherwise.

Text is rendered with the font found by `vlageom.plot.style.apply_style`: set VLAGEOM_FONT_DIR to
a directory with the Product Sans files to compare against the paper figure. The comparison is
meaningful only with the pinned library versions (matplotlib, numpy, Pillow, FreeType as bundled
with matplotlib).

Examples:
  python tests/compare_fig1_render.py --reference-png paper/fig1.png --reference-pdf paper/fig1.pdf
  python tests/compare_fig1_render.py --panels /path/to/panels --reference-png paper/fig1.png
  python tests/compare_fig1_render.py --rendered out/fig1 --reference-png paper/fig1.png
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG1 = REPO / "figures" / "fig1.py"
# tile column boundaries as fractions of the figure width (midpoints between the floor centres)
TILE_EDGES = (0.0, 0.26075, 0.5, 0.73925, 1.0)
TILE_NAMES = ("gripper", "wrist", "target", "distance")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def compare_png(new: Path, ref: Path) -> bool:
    from PIL import Image

    m_new, m_ref = md5(new), md5(ref)
    print(f"PNG  new {m_new}  {new}\n     ref {m_ref}  {ref}")
    if m_new == m_ref:
        print("PNG  identical (same bytes)")
        return True
    a = np.asarray(Image.open(new).convert("RGBA"), dtype=np.int16)
    b = np.asarray(Image.open(ref).convert("RGBA"), dtype=np.int16)
    if a.shape != b.shape:
        print(f"PNG  DIFFERENT size: new {a.shape[1]}x{a.shape[0]}, ref {b.shape[1]}x{b.shape[0]}")
        return False
    diff = np.abs(a - b).max(axis=2)
    bad = diff > 0
    n = int(bad.sum())
    if n == 0:
        print("PNG  same pixels, different encoding (metadata or compression)")
        return True
    ys, xs = np.nonzero(bad)
    print(f"PNG  DIFFERENT: {n} of {bad.size} pixels ({100 * n / bad.size:.4f}%), "
          f"bbox x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()} (of {a.shape[1]}x{a.shape[0]}), "
          f"max channel difference {int(diff.max())}")
    w = a.shape[1]
    for name, lo, hi in zip(TILE_NAMES, TILE_EDGES[:-1], TILE_EDGES[1:]):
        cols = slice(round(lo * w), round(hi * w))
        print(f"     {name:9s} {int(bad[:, cols].sum())} differing pixels")
    return False


def _strip_date(data: bytes) -> bytes:
    return re.sub(rb"/CreationDate \([^)]*\)", b"/CreationDate ()", data)


def _pdf_spans(path: Path):
    import pymupdf

    spans, images = [], []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for s in line["spans"]:
                        spans.append((s["text"], s["font"], round(s["size"], 2),
                                      tuple(round(v, 2) for v in s["bbox"])))
            for img in page.get_images(full=True):
                images.append((img[2], img[3]))          # width, height
    return spans, images


def compare_pdf(new: Path, ref: Path) -> bool:
    a, b = new.read_bytes(), ref.read_bytes()
    print(f"PDF  new {len(a)} bytes  {new}\n     ref {len(b)} bytes  {ref}")
    sa, sb = _strip_date(a), _strip_date(b)
    if sa == sb:
        print("PDF  identical (same bytes apart from /CreationDate)")
        return True
    try:
        spans_a, img_a = _pdf_spans(new)
        spans_b, img_b = _pdf_spans(ref)
    except ImportError:
        i = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), min(len(sa), len(sb)))
        print(f"PDF  DIFFERENT: first differing byte at offset {i} "
              "(install pymupdf for a text and image comparison)")
        return False
    only_new = [s for s in spans_a if s not in spans_b]
    only_ref = [s for s in spans_b if s not in spans_a]
    print(f"PDF  DIFFERENT bytes; text spans: {len(spans_a)} new, {len(spans_b)} ref, "
          f"{len(only_new)} only in new, {len(only_ref)} only in ref; "
          f"images: {len(img_a)} new, {len(img_b)} ref, same sizes: {sorted(img_a) == sorted(img_b)}")
    for tag, rows in (("new", only_new), ("ref", only_ref)):
        for s in rows[:20]:
            print(f"     only in {tag}: {s}")
    return False


def render(panels: Path | None, out_dir: Path) -> Path:
    cmd = [sys.executable, str(FIG1), "--out-dir", str(out_dir), "--name", "fig1"]
    if panels is not None:
        cmd += ["--panels", str(panels)]
    print("render:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir / "fig1"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--reference-png", type=Path, required=True, help="reference PNG")
    ap.add_argument("--reference-pdf", type=Path, default=None, help="reference PDF (optional)")
    ap.add_argument("--panels", type=Path, default=None,
                    help="panels directory to render from (default: $VLAGEOM_DATA/panels)")
    ap.add_argument("--rendered", type=Path, default=None, metavar="STEM",
                    help="compare an existing render <STEM>.png/.pdf instead of rendering")
    args = ap.parse_args(argv)
    need = [args.reference_png, args.reference_pdf]
    if args.rendered is not None:
        need += [args.rendered.with_suffix(".png")]
        if args.reference_pdf is not None:
            need += [args.rendered.with_suffix(".pdf")]
    missing = [str(p) for p in need if p is not None and not p.is_file()]
    if missing:
        ap.error(f"file(s) not found: {missing}")

    with tempfile.TemporaryDirectory(prefix="fig1_compare_") as tmp:
        stem = args.rendered if args.rendered is not None else render(args.panels, Path(tmp))
        ok = compare_png(stem.with_suffix(".png"), args.reference_png)
        if args.reference_pdf is not None:
            ok = compare_pdf(stem.with_suffix(".pdf"), args.reference_pdf) and ok
    print("RESULT", "IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
