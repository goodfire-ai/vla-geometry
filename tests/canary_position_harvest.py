#!/usr/bin/env python
"""Row-for-row comparison of position harvest outputs with reference outputs (canaries).

  crossed   one house of the crossed harvest (pipelines/position/10_crossed_harvest.py)
      python tests/canary_position_harvest.py crossed --house 44 --new <new store root> \
          --ref <reference store root>
      compares houses/h{house}.npz key by key (every array bit for bit, NaN == NaN; audit_json
      as a dict).

  lines     one house of the hand lines (50_hand_lines.py)
      python tests/canary_position_harvest.py lines --house 44 --new <new dir> --ref <reference dir>
      compares h{house}.npz key by key.

  shape     one path directory of 55_object_lines.py, 80_hand_shapes.py or 85_object_shapes.py
      python tests/canary_position_harvest.py shape --new <new dir> --ref <reference dir>
      compares acts.npz key by key, manifest.json value by value (every key both have, except the
      audit counters, which are printed; keys only one file has are listed) and the exo_*.png
      pixels.

  pack      the packed store (20_pack_store.py)
      python tests/canary_position_harvest.py pack --new <new prep dir> --ref <reference prep dir>
      compares labels.npz key by key, meta.json, the list of left-out houses (with
      --ref-excluded FILE) and every X_*.npy of the new store byte for byte with the reference's.

Activations come from bfloat16 inference on the GPU and EGL rendering, so a harvest canary is
expected to match bit for bit only on the same GPU type and driver as the reference run.
Exit status 0 when everything compared is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FAILS: list[str] = []


def report(name, same, detail=""):
    print(f"  {'EQUAL' if same else 'DIFF '} {name}" + (f"  {detail}" if detail else ""))
    if not same:
        FAILS.append(name)


def cmp_array(name, x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        report(name, False, f"new {x.dtype}{x.shape} vs ref {y.dtype}{y.shape}")
        return
    if name.endswith("audit_json") and x.dtype == np.uint8:
        jx, jy = json.loads(bytes(x).decode()), json.loads(bytes(y).decode())
        report(name, jx == jy, "" if jx == jy else f"new {jx} ref {jy}")
        return
    if x.dtype.kind in "fc":
        if np.array_equal(x, y, equal_nan=True):
            report(name, True, f"{y.dtype}{y.shape}")
            return
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        nrow = int(np.any(d.reshape(len(d), -1) > 0, axis=1).sum()) if d.ndim > 0 and len(d) else 0
        report(name, False,
               f"{y.dtype}{y.shape} max|diff| {np.nanmax(d):.4g}, rows differing {nrow}")
        return
    eq = np.array_equal(x, y)
    report(name, eq, f"{y.dtype}{y.shape}" + ("" if eq else f" new {x!r} ref {y!r}"[:300]))


def cmp_npz(tag, new_p, ref_p):
    print(f"{tag}\n  new {new_p}\n  ref {ref_p}")
    a, b = np.load(new_p), np.load(ref_p)
    for k in b.files:
        if k not in a.files:
            report(f"{tag}:{k}", False, "missing in the new file")
            continue
        cmp_array(f"{tag}:{k}", a[k], b[k])
    extra = sorted(set(a.files) - set(b.files))
    if extra:
        report(f"{tag}: keys only in the new file", False, str(extra))


def cmp_value(name, x, y):
    """Compare JSON values; keys that only one of two dicts has are printed, not compared."""
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                print(f"  note {name}.{k}: only in the "
                      + ("reference" if k not in x else "new file"))
            else:
                cmp_value(f"{name}.{k}", x[k], y[k])
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            report(name, False, f"length {len(x)} vs {len(y)}")
            return
        for i, (u, v) in enumerate(zip(x, y)):
            cmp_value(f"{name}[{i}]", u, v)
    elif x != y and not (isinstance(x, float) and isinstance(y, float)
                         and np.isnan(x) and np.isnan(y)):
        report(name, False, f"new {x!r} ref {y!r}")


def run_crossed(a):
    cmp_npz(f"h{a.house}", Path(a.new) / "houses" / f"h{a.house}.npz",
            Path(a.ref) / "houses" / f"h{a.house}.npz")


def run_lines(a):
    cmp_npz(f"h{a.house}", Path(a.new) / f"h{a.house}.npz", Path(a.ref) / f"h{a.house}.npz")


def run_shape(a):
    import imageio.v2 as imageio

    new, ref = Path(a.new), Path(a.ref)
    cmp_npz("acts", new / "acts.npz", ref / "acts.npz")
    mn = json.loads((new / "manifest.json").read_text())
    mr = json.loads((ref / "manifest.json").read_text())
    n0 = len(FAILS)
    for k in sorted(set(mn) & set(mr) - {"audit", "protocol"}):
        cmp_value(f"manifest.{k}", mn[k], mr[k])
    only = sorted(set(mn) ^ set(mr))
    print(f"  manifest: {len(set(mn) & set(mr))} common keys compared"
          + (f"; keys in one file only (not compared): {only}" if only else ""))
    if len(FAILS) == n0:
        report("manifest values", True, f"{len(mr['points'])} waypoints")
    print(f"  audit new {mn.get('audit')}\n  audit ref {mr.get('audit')}")
    pngs = sorted(p.name for p in ref.glob("exo_*.png"))
    bad = [p for p in pngs if not (new / p).exists() or
           not np.array_equal(imageio.imread(new / p), imageio.imread(ref / p))]
    report(f"exo frames ({len(pngs)})", not bad, f"differing {bad[:10]}" if bad else "")


def run_pack(a):
    new, ref = Path(a.new), Path(a.ref)
    cmp_npz("labels", new / "labels.npz", ref / "labels.npz")
    mn = json.loads((new / "meta.json").read_text())
    mr = json.loads((ref / "meta.json").read_text())
    report("meta.json", mn == mr)
    if a.ref_excluded:
        en = (new / "excluded_houses.txt").read_text().split()
        er = Path(a.ref_excluded).read_text().split()
        report("left-out houses", en == er, f"new {en} ref {er}")
    for p in sorted(new.glob("X_*.npy")):
        q = ref / p.name
        if not q.exists():
            report(p.name, False, "no reference array")
            continue
        x, y = np.load(p, mmap_mode="r"), np.load(q, mmap_mode="r")
        if x.shape != y.shape or x.dtype != y.dtype:
            report(p.name, False, f"new {x.dtype}{x.shape} vs ref {y.dtype}{y.shape}")
            continue
        step, same = 2048, True
        for i in range(0, len(x), step):
            if not np.array_equal(np.asarray(x[i:i + step]), np.asarray(y[i:i + step]),
                                  equal_nan=True):
                same = False
                print(f"    {p.name}: rows {i}..{i + step} differ")
                break
        report(p.name, same, f"{x.dtype}{x.shape}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("crossed", "lines"):
        s = sub.add_parser(name)
        s.add_argument("--house", type=int, required=True)
        s.add_argument("--new", required=True)
        s.add_argument("--ref", required=True)
    s = sub.add_parser("shape")
    s.add_argument("--new", required=True)
    s.add_argument("--ref", required=True)
    s = sub.add_parser("pack")
    s.add_argument("--new", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--ref-excluded", default=None, help="reference list of left-out houses")
    a = ap.parse_args(argv)
    {"crossed": run_crossed, "lines": run_lines, "shape": run_shape, "pack": run_pack}[a.cmd](a)
    print("IDENTICAL" if not FAILS else f"DIFFERENT ({len(FAILS)}): {FAILS[:20]}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
