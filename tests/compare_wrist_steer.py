"""CPU comparisons of the Figure 6 (wrist steering) outputs with reference files.

  natural   natural cache and natural cloud arrays (stage 20), key by key
              --new-cache F --ref-cache F [--new-natural F --ref-natural F]
  operator  operator file (stage 40): held-out houses must be equal; V, c, Lhat are reported with
            their max relative difference and must be equal unless --rtol is given (their last
            digits depend on the CPU)
              --new F --ref F [--rtol R]
  metrics   panel-a sweep measures (fig6_a_metrics.json), all keys but the input path `npz`
              --new F --ref F
  numbers   numbers json (stage 80) against a reference numbers json; every value present in both
            must be equal (keys present in only one are listed). The reference's key names in
            REF_NAMES are compared under this repository's names.
              --new F --ref F [--ref-key KEY]   (KEY: compare with the reference's sub-dict KEY)
  render    rendered panels (figures/fig6.py) against the paper's panel files: PDFs byte-identical
            apart from /CreationDate, PNG byte-identical (differing pixels counted otherwise)
              --new-dir DIR [--ref-a PDF] [--ref-bcd-pdf PDF] [--ref-bcd-png PNG]

Exit status 0 when everything compared is identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

CREATION_DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")
# Key names of the paper's numbers file (left) and the names stage 80 writes (right), as leaf-path
# prefixes.
REF_NAMES = {
    "/oneshot/comp_mean_incl6": "/oneshot/comp_mean_all",
    "/oneshot/comp_sd_incl6": "/oneshot/comp_sd_all",
    "/edit_size/frac_mean_32": "/edit_size/frac_mean_all",
    "/edit_size/frac_sd_32": "/edit_size/frac_sd_all",
    "/edit_size/n32": "/edit_size/n_all",
    "/edit_size/frac_mean_28": "/edit_size/frac_mean_kept",
    "/edit_size/frac_sd_28": "/edit_size/frac_sd_kept",
    "/persist/fig6d_house10/": "/persist/fig6d_example/final_rot/",
}


def cmp_arrays(a: dict, b: dict, label: str) -> bool:
    ok = True
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            print(f"  MISSING  {label}:{k} (in the {'new' if k not in a else 'reference'} file only)")
            ok = False
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        eq = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
        msg = f"  {'equal' if eq else 'DIFF':8s} {label}:{k} {x.dtype}{x.shape}"
        if not eq and x.shape == y.shape and x.dtype.kind in "fiu":
            msg += f" max|diff| {np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))):.3e}"
        print(msg)
        ok = ok and eq
    return ok


def load_npz(p) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def renamed(path: str, names: dict) -> str:
    for old, new in names.items():
        if path == old or (old.endswith("/") and path.startswith(old)):
            return new + path[len(old):]
    return path


def cmp_json(a, b, names: dict | None = None) -> bool:
    la = dict(leaves(a))
    lb = {renamed(k, names or {}): v for k, v in leaves(b)}
    ok = True
    for k in sorted(set(la) & set(lb)):
        x, y = la[k], lb[k]
        if not (x == y or (isinstance(x, float) and isinstance(y, float) and x != x and y != y)):
            print(f"  DIFF     {k}: {x!r} vs {y!r}")
            ok = False
    for k in sorted(set(la) - set(lb)):
        print(f"  only new {k}")
    for k in sorted(set(lb) - set(la)):
        print(f"  only ref {k}")
    common = set(la) & set(lb)
    print(f"  {sum(1 for k in common if la[k] == lb[k] or la[k] != la[k])}/"
          f"{len(common)} common values equal")
    if not common:
        print("  no values in common (for a reference that nests the numbers under a key, give "
              "--ref-key)")
        ok = False
    return ok


def pdf_norm(p: Path) -> bytes:
    return CREATION_DATE.sub(b"", p.read_bytes())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("natural")
    s.add_argument("--new-cache", type=Path)
    s.add_argument("--ref-cache", type=Path)
    s.add_argument("--new-natural", type=Path)
    s.add_argument("--ref-natural", type=Path)
    s = sub.add_parser("operator")
    s.add_argument("--new", type=Path, required=True)
    s.add_argument("--ref", type=Path, required=True)
    s.add_argument("--rtol", type=float, default=0.0)
    s = sub.add_parser("metrics")
    s.add_argument("--new", type=Path, required=True)
    s.add_argument("--ref", type=Path, required=True)
    s = sub.add_parser("numbers")
    s.add_argument("--new", type=Path, required=True)
    s.add_argument("--ref", type=Path, required=True)
    s.add_argument("--ref-key", default=None)
    s = sub.add_parser("render")
    s.add_argument("--new-dir", type=Path, required=True)
    s.add_argument("--ref-a", type=Path, default=None)
    s.add_argument("--ref-bcd-pdf", type=Path, default=None)
    s.add_argument("--ref-bcd-png", type=Path, default=None)
    a = ap.parse_args(argv)
    ok = True
    if a.cmd == "natural":
        for new, ref, label in ((a.new_cache, a.ref_cache, "cache"), (a.new_natural, a.ref_natural, "natural")):
            if new and ref:
                print(f"{label}: {new} vs {ref}")
                ok = cmp_arrays(load_npz(new), load_npz(ref), label) and ok
    elif a.cmd == "operator":
        x, y = load_npz(a.new), load_npz(a.ref)
        eq = np.array_equal(x["test_houses"], y["test_houses"])
        print(f"  {'equal' if eq else 'DIFF':8s} test_houses ({len(y['test_houses'])} houses)")
        ok = eq
        for k in ("V", "c", "Lhat"):
            d = float(np.max(np.abs(x[k] - y[k])) / (np.max(np.abs(y[k])) + 1e-300))
            same = np.array_equal(x[k], y[k])
            within = same or d <= a.rtol
            print(f"  {'equal' if same else ('within' if within else 'DIFF'):8s} {k} {x[k].shape} "
                  f"max rel diff {d:.2e}")
            ok = ok and within
    elif a.cmd == "metrics":
        x, y = json.loads(a.new.read_text()), json.loads(a.ref.read_text())
        x.pop("npz", None)
        y.pop("npz", None)
        ok = cmp_json(x, y)
    elif a.cmd == "numbers":
        x, y = json.loads(a.new.read_text()), json.loads(a.ref.read_text())
        if a.ref_key:
            y = y[a.ref_key]
        ok = cmp_json(x, y, REF_NAMES)
    elif a.cmd == "render":
        for name, ref in (("fig6_a.pdf", a.ref_a), ("fig6_bcd.pdf", a.ref_bcd_pdf)):
            if ref is None:
                continue
            new = a.new_dir / name
            eq = pdf_norm(new) == pdf_norm(ref)
            print(f"  {'equal' if eq else 'DIFF':8s} {name} vs {ref} (apart from /CreationDate; "
                  f"{new.stat().st_size} vs {ref.stat().st_size} bytes)")
            ok = ok and eq
        if a.ref_bcd_png is not None:
            new = a.new_dir / "fig6_bcd.png"
            ma = hashlib.md5(new.read_bytes()).hexdigest()
            mb = hashlib.md5(a.ref_bcd_png.read_bytes()).hexdigest()
            eq = ma == mb
            msg = f"  {'equal' if eq else 'DIFF':8s} fig6_bcd.png md5 {ma} vs {mb}"
            if not eq:
                from PIL import Image
                pa = np.asarray(Image.open(new).convert("RGBA"))
                pb = np.asarray(Image.open(a.ref_bcd_png).convert("RGBA"))
                if pa.shape == pb.shape:
                    msg += f"; {int(np.any(pa != pb, axis=-1).sum())} of {pa.shape[0] * pa.shape[1]} pixels differ"
                else:
                    msg += f"; size {pa.shape} vs {pb.shape}"
            print(msg)
            ok = ok and eq
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
