#!/usr/bin/env python
"""Row-for-row comparison of target-tile harvest outputs with reference outputs (canaries).

Three checks, one per stage:

  teleport  one house of the object-teleport harvest (10_teleport_harvest.py)
      python tests/canary_target_harvest.py teleport --house 44 \
          --new <new store root> --ref <reference store root>
      compares houses/h{house}.npz key by key (every array bit for bit, NaN == NaN; audit_json as
      a dict) and frames/h{house}.npz key by key (thumbnails exo, wrist, config_id_q0, dark_json)
      when both exist.

  square    the object-path harvest (30_square_path_harvest.py)
      python tests/canary_target_harvest.py square --new <new h44 dir> --ref <reference h44 dir>
      compares acts.npz key by key, manifest.json value by value, and the exo_*.png pixels.

  pack      the packed store (20_pack_store.py), e.g. built from a subset of house files
      python tests/canary_target_harvest.py pack --new <new store root> --ref <reference store root>
      matches rows by (house, config_id, chunk_q) and compares every labels.npz key and every
      X_*.npy array on the matched rows; meta.json layer lists must match.

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
    if name.endswith(("audit_json", "dark_json")) and x.dtype == np.uint8:
        jx, jy = json.loads(bytes(x).decode()), json.loads(bytes(y).decode())
        report(name, jx == jy, "" if jx == jy else f"new {jx} ref {jy}")
        return
    if x.dtype.kind in "fc":
        eq = np.array_equal(x, y, equal_nan=True)
        if eq:
            report(name, True, f"{y.dtype}{y.shape}")
            return
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        nrow = int(np.any(d.reshape(len(d), -1) > 0, axis=1).sum()) if d.ndim > 0 and len(d) else 0
        report(name, False, f"{y.dtype}{y.shape} max|diff| {np.nanmax(d):.4g}, rows differing {nrow}")
        return
    eq = np.array_equal(x, y)
    report(name, eq, f"{y.dtype}{y.shape}" + ("" if eq else f" new {x!r} ref {y!r}"[:300]))


def cmp_npz(tag, new_p, ref_p, require_all=True):
    """Compare every key of the reference file; with require_all=False, reference keys that the
    new file does not have are listed instead of counted as differences."""
    print(f"{tag}\n  new {new_p}\n  ref {ref_p}")
    a, b = np.load(new_p), np.load(ref_p)
    for k in b.files:
        if k not in a.files:
            if require_all:
                report(f"{tag}:{k}", False, "missing in the new file")
            else:
                print(f"  (reference key {k} not written by this harvester; not compared)")
            continue
        cmp_array(f"{tag}:{k}", a[k], b[k])
    extra = sorted(set(a.files) - set(b.files))
    if extra:
        print(f"  keys only in the new file: {extra}")


def cmp_json(name, x, y, path=""):
    if isinstance(y, dict):
        if not isinstance(x, dict):
            report(f"{name}{path}", False, "type")
            return
        for k in y:
            if k not in x:
                report(f"{name}{path}/{k}", False, "missing in the new file")
            else:
                cmp_json(name, x[k], y[k], f"{path}/{k}")
        extra = sorted(set(x) - set(y))
        if extra:
            print(f"  {name}{path}: keys only in the new file: {extra}")
        return
    if isinstance(y, list) and y and isinstance(y[0], dict):
        if not isinstance(x, list) or len(x) != len(y):
            report(f"{name}{path}", False, f"list length new {len(x)} ref {len(y)}")
            return
        for i, (xi, yi) in enumerate(zip(x, y)):
            cmp_json(name, xi, yi, f"{path}[{i}]")
        return
    if x != y:
        report(f"{name}{path}", False, f"new {x!r} ref {y!r}"[:300])


def cmd_teleport(args):
    h = f"h{args.house}.npz"
    cmp_npz("houses/" + h, args.new / "houses" / h, args.ref / "houses" / h)
    fn, fr = args.new / "frames" / h, args.ref / "frames" / h
    if fn.exists() and fr.exists():
        cmp_npz("frames/" + h, fn, fr)
    else:
        print(f"frames/{h}: not compared (new exists {fn.exists()}, ref exists {fr.exists()})")


def cmd_square(args):
    cmp_npz("acts.npz", args.new / "acts.npz", args.ref / "acts.npz")
    mn = json.loads((args.new / "manifest.json").read_text())
    mr = json.loads((args.ref / "manifest.json").read_text())
    print("manifest.json")
    n0 = len(FAILS)
    cmp_json("manifest", mn, mr)
    if len(FAILS) == n0:
        print(f"  EQUAL manifest.json ({len(mr.get('points', []))} points, all values)")
    from PIL import Image
    pngs = sorted(p.name for p in args.ref.glob("exo_*.png"))
    bad = []
    for name in pngs:
        pn = args.new / name
        if not pn.exists() or not np.array_equal(np.asarray(Image.open(pn)),
                                                 np.asarray(Image.open(args.ref / name))):
            bad.append(name)
    report(f"exo_*.png ({len(pngs)} frames)", not bad, f"differing: {bad[:8]}" if bad else "")


def cmd_pack(args):
    pn, pr = args.new / "prep", args.ref / "prep"
    mn, mr = json.loads((pn / "meta.json").read_text()), json.loads((pr / "meta.json").read_text())
    for k in ("layers_llm", "layers_ae", "n_tok"):
        report(f"meta.{k}", mn[k] == mr[k])
    ln = dict(np.load(pn / "labels.npz", allow_pickle=False))
    lr = dict(np.load(pr / "labels.npz", allow_pickle=False))
    key_r = {(int(h), int(c), int(q)): i for i, (h, c, q) in
             enumerate(zip(lr["house"], lr["config_id"], lr["chunk_q"]))}
    idx_n = np.arange(len(ln["house"]))
    try:
        idx_r = np.array([key_r[(int(h), int(c), int(q))] for h, c, q in
                          zip(ln["house"], ln["config_id"], ln["chunk_q"])], np.int64)
    except KeyError as e:
        raise SystemExit(f"new row {e} has no reference row")
    print(f"pack: {len(idx_n)} new rows matched to reference rows "
          f"{'(the reference row prefix)' if np.array_equal(idx_r, idx_n) else ''}")
    for k in lr:
        if k not in ln:
            report(f"labels:{k}", False, "missing")
            continue
        cmp_array(f"labels:{k}", ln[k], lr[k][idx_r])
    for f in sorted(pr.glob("X_*.npy")):
        g = pn / f.name
        if not g.exists():
            report(f.name, False, "missing")
            continue
        xn = np.load(g, mmap_mode="r")
        xr = np.load(f, mmap_mode="r")
        if xn.shape[1:] != xr.shape[1:] or xn.dtype != xr.dtype or len(xn) != len(idx_n):
            report(f.name, False, f"new {xn.dtype}{xn.shape} vs ref {xr.dtype}{xr.shape}")
            continue
        n_bad, md = 0, 0.0
        for s in range(0, len(idx_n), 256):                # row blocks: bounded memory
            a = np.asarray(xn[s:s + 256])
            b = np.asarray(xr[idx_r[s:s + 256]])
            if not np.array_equal(a, b, equal_nan=True):
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                n_bad += int(np.any(d.reshape(len(d), -1) > 0, axis=1).sum())
                md = max(md, float(np.nanmax(d)))
        report(f.name, n_bad == 0, f"{xr.dtype}(n, {', '.join(map(str, xr.shape[1:]))})"
               + ("" if n_bad == 0 else f" rows differing {n_bad}, max|diff| {md:.4g}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teleport")
    t.add_argument("--house", type=int, required=True)
    t.add_argument("--new", type=Path, required=True, help="new store root (houses/, frames/)")
    t.add_argument("--ref", type=Path, required=True, help="reference store root")
    s = sub.add_parser("square")
    s.add_argument("--new", type=Path, required=True, help="new h{house} directory")
    s.add_argument("--ref", type=Path, required=True, help="reference h{house} directory")
    p = sub.add_parser("pack")
    p.add_argument("--new", type=Path, required=True, help="new store root (prep/)")
    p.add_argument("--ref", type=Path, required=True, help="reference store root (prep/)")
    args = ap.parse_args()
    {"teleport": cmd_teleport, "square": cmd_square, "pack": cmd_pack}[args.cmd](args)
    print(f"{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} differing item(s)"
          + (f": {FAILS[:12]}" if FAILS else ""))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
