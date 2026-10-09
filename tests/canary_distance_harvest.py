"""Compare a near-range harvest (pipelines/distance/10_near_harvest.py) or its packed store
(20_pack_near.py) with reference data, bit for bit.

  house   per-house files: <root>/houses/h<house>.npz (every key: dtype, shape, values; NaN equal
          to NaN) and the q0 thumbnails <root>/frames/h<house>.npz. When activations differ, the
          (placement, query) rows that differ are listed.
  store   packed stores: X_last_tok.npy and X_ae.npy byte for byte, every labels.npz array and
          meta.json.

GPU canary (one H200, EGL; about 5 minutes including the model load):
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python pipelines/distance/10_near_harvest.py --houses 0 --out <canary dir>
  python tests/canary_distance_harvest.py house --new-root <canary dir> \
      --ref-root <reference near-range store root> --houses 0

The policy runs in bfloat16 on the GPU, so activations are bit-reproducible only on the same GPU
type, driver and library versions as the reference.
Exit code 0 when everything matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import sys
from pathlib import Path

import numpy as np


def _same(x, y):
    if x.dtype != y.dtype or x.shape != y.shape:
        return False
    if x.dtype.kind in "fc":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def _maxdiff(x, y):
    try:
        if x.shape != y.shape or x.dtype.kind not in "fciu":
            return float("nan")
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        return float(np.nanmax(d)) if d.size else 0.0
    except Exception:
        return float("nan")


def compare_npz(new_p, ref_p, row_keys=None):
    """Compare two .npz files key by key; returns (ok, lines)."""
    a, b = np.load(new_p, allow_pickle=False), np.load(ref_p, allow_pickle=False)
    lines, ok = [], True
    ka, kb = set(a.files), set(b.files)
    if ka != kb:
        ok = False
        lines.append(f"    keys only new {sorted(ka - kb)}, only reference {sorted(kb - ka)}")
    bad = []
    for k in sorted(ka & kb):
        x, y = a[k], b[k]
        if not _same(x, y):
            bad.append(k)
            lines.append(f"    DIFF {k}: new {x.dtype} {x.shape} / ref {y.dtype} {y.shape}, "
                         f"max |diff| {_maxdiff(x, y):.3g}")
    ok &= not bad
    if bad and row_keys is not None and all(k in ka & kb for k in row_keys):
        if _same(a[row_keys[0]], b[row_keys[0]]) and _same(a[row_keys[1]], b[row_keys[1]]):
            rows = set()
            for k in bad:
                x, y = a[k], b[k]
                if x.shape == y.shape and x.ndim >= 1 and len(x) == len(a[row_keys[0]]):
                    diff = ~np.all((x == y) | (np.isnan(x) & np.isnan(y))
                                   if x.dtype.kind == "f" else (x == y),
                                   axis=tuple(range(1, x.ndim)))
                    rows |= set(np.nonzero(diff)[0].tolist())
            if rows:
                pc = [(int(a[row_keys[0]][r]), int(a[row_keys[1]][r])) for r in sorted(rows)]
                lines.append(f"    differing rows (placement, query): {pc[:40]}"
                             f"{' ...' if len(pc) > 40 else ''} ({len(pc)} of "
                             f"{len(a[row_keys[0]])})")
    if "audit_json" in ka & kb and "audit_json" in bad:
        lines.append(f"    audit new {bytes(a['audit_json']).decode()}")
        lines.append(f"    audit ref {bytes(b['audit_json']).decode()}")
    name = f"{new_p.parent.name}/{new_p.name}"
    lines.insert(0, f"  {'ok  ' if ok else 'FAIL'} {name}: {len(ka & kb)} keys"
                    f"{'' if ok else f', {len(bad)} differ'}")
    return ok, lines


def cmd_house(a):
    ok = True
    for h in a.houses:
        for sub, row_keys in (("houses", ("config_id", "chunk_q")), ("frames", None)):
            n, r = a.new_root / sub / f"h{h}.npz", a.ref_root / sub / f"h{h}.npz"
            if not n.exists() or not r.exists():
                if sub == "frames" and not r.exists():
                    print(f"  skip {sub}/h{h}.npz (no reference)")
                    continue
                print(f"  FAIL {sub}/h{h}.npz missing ({'new' if not n.exists() else 'reference'})")
                ok = False
                continue
            good, lines = compare_npz(n, r, row_keys)
            print("\n".join(lines))
            ok &= good
    return ok


def cmd_store(a):
    ok = True
    for f in ("X_last_tok.npy", "X_ae.npy"):
        good = filecmp.cmp(a.new / f, a.ref / f, shallow=False)
        print(f"  {'ok  ' if good else 'FAIL'} {f} {'byte-identical' if good else 'differs'}")
        ok &= good
    good, lines = compare_npz(a.new / "labels.npz", a.ref / "labels.npz")
    print("\n".join(lines))
    ok &= good
    mn = json.loads((a.new / "meta.json").read_text())
    mr = json.loads((a.ref / "meta.json").read_text())
    print(f"  {'ok  ' if mn == mr else 'FAIL'} meta.json {mn if mn != mr else ''}")
    return ok and mn == mr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("house", help="compare per-house harvest files")
    p.add_argument("--new-root", type=Path, required=True, help="harvest output root (--out)")
    p.add_argument("--ref-root", type=Path, required=True, help="reference store root")
    p.add_argument("--houses", type=int, nargs="+", required=True)
    p = sp.add_parser("store", help="compare packed stores")
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    a = ap.parse_args()
    ok = cmd_house(a) if a.cmd == "house" else cmd_store(a)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
