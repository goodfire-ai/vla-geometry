#!/usr/bin/env python
"""Compare the contact CPU outputs (Figure 4) with reference outputs.

Subcommands (every location is an argument; the defaults are this repository's outputs under
$VLAGEOM_DATA):

  cache   natural cache (stage 20) vs a reference natural cache: every array bit for bit (dtype,
          shape, values; NaN == NaN), except `harvest` (the input directory, a path record)
      python tests/compare_contact.py cache --reference <reference natural cache .npz> [--new FILE]
      With --pool-only EP [EP ...]: pool those episodes' harvest files (--harvest DIR) with
      vlageom.contact.pool_natural_shard and compare them with their rows of the reference cache,
      without building the whole cache.

  pool    off-schedule pool (stage 40) vs a reference pool: every array bit for bit, except
          `subdir` (the harvest directory's name)
      python tests/compare_contact.py pool --reference <reference pool .npz> [--new FILE]

  probe   natural probe scores (stage 50) vs a reference: either a file written by stage 50, or a
          grid file with `variants`, `layers` and `acc` / `auc` / `balanced_acc` [variant][layer]
          and `baseline.majority_acc`; values compared exactly
      python tests/compare_contact.py probe --reference <json> [--new FILE]

  scores  held-out detection scores (stage 60) vs a reference file of the same layout: n_pts,
          n_eps, counts, phase_timer_within_pose_auc and every value under `variants`, exactly
          (the path and description keys cache, subdir, probe, scheme, source are not compared)
      python tests/compare_contact.py scores --reference <json> [--new FILE]

  panel   panel file (stage 70) vs reference values: --ref-pca an .npz with P, contact, evr,
          origin_p (panel a); --ref-scores a held-out scores json and --ref-probe a probe json
          (panel b: frac_pos, frac_pos_se, n_pose of the txt_all bars, and the txt_all layer-24
          accuracy); `show` is checked against numpy default_rng(0).permutation(n)[:6000]
      python tests/compare_contact.py panel --ref-pca <npz> --ref-scores <json> --ref-probe <json>

Exit status 0 when everything compared is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import contact as C
from vlageom import paths
from vlageom.util import npz_member

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
    if x.dtype.kind in "fc":
        if np.array_equal(x, y, equal_nan=True):
            report(name, True, f"{y.dtype}{y.shape}")
            return
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        rows = int(np.any(d.reshape(len(d), -1) > 0, axis=1).sum()) if d.ndim and len(d) else 0
        report(name, False, f"{y.dtype}{y.shape} max|diff| {np.nanmax(d):.4g}, rows differing {rows}")
        return
    eq = np.array_equal(x, y)
    report(name, eq, f"{y.dtype}{y.shape}")


def cmp_npz(new, ref, skip=()):
    print(f"new {new}\nref {ref}")
    with np.load(new, mmap_mode="r") as a, np.load(ref, mmap_mode="r") as b:
        for k in b.files:
            if k in skip:
                print(f"  (not compared: {k}, a path record)")
                continue
            if k not in a.files:
                report(k, False, "missing in the new file")
                continue
            cmp_array(k, a[k], b[k])
        extra = sorted(set(a.files) - set(b.files))
        if extra:
            report("extra keys", False, str(extra))
        order_same = [k for k in a.files] == [k for k in b.files]
        print(f"  key order {'equal' if order_same else 'differs'}")


def member_map(path, key):
    """Member `key` of an uncompressed .npz file as a read-only memory map (only the indexed rows
    are read)."""
    off, shape, dtype, fortran = npz_member(path, key)
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=tuple(shape),
                     order="F" if fortran else "C")


def cmd_cache(a):
    if a.pool_only:
        harvest = Path(a.harvest or C.natural_dir())
        meta = {r["episode_index"]: r for r in C.read_natural_episodes()}
        with np.load(a.reference, mmap_mode="r") as z:
            ep_all = np.asarray(z["ep"])
            for e in a.pool_only:
                files = sorted(harvest.glob(f"ep{e}_h*.npz"))
                if not files:
                    report(f"ep{e}", False, f"no harvest file in {harvest}")
                    continue
                r = C.pool_natural_shard(files[0], meta[e]["orig_n_steps"], meta[e]["orig_contact"])
                idx = np.flatnonzero(ep_all == e)
                print(f"episode {e}: {len(idx)} reference rows")
                for v in C.VARIANTS:
                    cmp_array(f"ep{e}:X_{v}", r["X"][v],
                              np.asarray(member_map(a.reference, f"X_{v}")[idx[0]:idx[-1] + 1]))
                for k in ("contact", "delta", "grip", "phase", "q", "step", "agree"):
                    cmp_array(f"ep{e}:{k}", r[k], np.asarray(z[k])[idx])
                for k, dt in (("house", np.int32), ("success", bool), ("onset", np.int32),
                              ("onset_phase", np.float32), ("orig_n_steps", np.int32),
                              ("n_steps_ep", np.int32), ("replay", bool)):
                    cmp_array(f"ep{e}:{k}", np.full(r["n"], r[k], dt), np.asarray(z[k])[idx])
        return
    cmp_npz(a.new or paths.tile_dir(C.TILE) / "natural_cache.npz", a.reference, skip=("harvest",))


def cmd_pool(a):
    cmp_npz(a.new or paths.tile_dir(C.TILE) / f"offschedule_L{C.PROBE_LAYER}.npz", a.reference,
            skip=("subdir",))


def _probe_cells(d):
    """{(pool, layer): {"acc", "auc", "bal_acc"}} and the majority accuracy of a probe json."""
    if "cells" in d:
        return ({(c["pool"], c["layer"]): {k: c[k] for k in ("acc", "auc", "bal_acc")}
                 for c in d["cells"].values()}, d["majority_acc"])
    out = {}
    for vi, v in enumerate(d["variants"]):
        for li, L in enumerate(d["layers"]):
            out[(v, int(L))] = {"acc": d["acc"][vi][li], "auc": d["auc"][vi][li],
                                "bal_acc": d["balanced_acc"][vi][li]}
    return out, d["baseline"]["majority_acc"]


def cmd_probe(a):
    new = json.loads(Path(a.new or paths.tile_dir(C.TILE) / "natural_probe.json").read_text())
    ref = json.loads(Path(a.reference).read_text())
    nc, nm = _probe_cells(new)
    rc, rm = _probe_cells(ref)
    for key, v in nc.items():
        if key not in rc:
            report(f"{key[0]}@{key[1]}", False, "not in the reference")
            continue
        for k in ("acc", "auc", "bal_acc"):
            same = v[k] == rc[key][k]
            report(f"{key[0]}@{key[1]} {k}", same, f"{v[k]!r}" + ("" if same else f" ref {rc[key][k]!r}"))
    report("majority_acc", nm == rm, f"{nm!r}" + ("" if nm == rm else f" ref {rm!r}"))


def _walk(x, y, path):
    if isinstance(y, dict):
        if not isinstance(x, dict):
            report(path, False, "not a dict in the new file")
            return
        for k in y:
            if k not in x:
                report(f"{path}.{k}", False, "missing in the new file")
            else:
                _walk(x[k], y[k], f"{path}.{k}")
        for k in set(x) - set(y):
            report(f"{path}.{k}", False, "not in the reference")
        return
    same = (x == y) or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y))
    if not same:
        report(path, False, f"new {x!r} ref {y!r}")
    else:
        _walk.n_equal += 1


def cmd_scores(a):
    new = json.loads(Path(a.new or paths.tile_dir(C.TILE) / "detection_heldout.json").read_text())
    ref = json.loads(Path(a.reference).read_text())
    _walk.n_equal = 0
    for k in ("n_pts", "n_eps", "counts", "phase_timer_within_pose_auc", "variants"):
        _walk(new.get(k), ref[k], k)
    print(f"  {_walk.n_equal} values equal")
    for p in C.POOLS:
        r = new["variants"][p]
        print(f"  {p}: " + ", ".join(f"{c} {r[c]['frac_pos']:.3f} (n {r[c]['n_pose']})" for c in C.CELLS))


def cmd_panel(a):
    new = dict(np.load(a.new or paths.panels_dir() / "contact.npz"))
    if a.ref_pca:
        ref = np.load(a.ref_pca)
        for k in ("P", "contact", "evr", "origin_p"):
            cmp_array(k, new[k], ref[k])
    cmp_array("show", new["show"], C.show_index(len(new["P"])))
    if a.ref_scores:
        r = json.loads(Path(a.ref_scores).read_text())["variants"]["txt_all"]
        cmp_array("bar_cells", new["bar_cells"], np.array(C.BAR_CELLS))
        cmp_array("frac_pos", new["frac_pos"], np.array([r[c]["frac_pos"] for c in C.BAR_CELLS]))
        cmp_array("frac_pos_se", new["frac_pos_se"],
                  np.array([r[c]["frac_pos_se"] for c in C.BAR_CELLS]))
        cmp_array("n_pose", new["n_pose"], np.array([r[c]["n_pose"] for c in C.BAR_CELLS], np.int64))
    if a.ref_probe:
        cells, _ = _probe_cells(json.loads(Path(a.ref_probe).read_text()))
        cmp_array("baseline_acc", new["baseline_acc"], np.float64(cells[("txt_all", C.PROBE_LAYER)]["acc"]))
    print(f"  evr {np.round(new['evr'], 4)}, bars {np.round(new['frac_pos'], 3)}, "
          f"baseline {float(new['baseline_acc']):.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cache")
    s.add_argument("--reference", type=Path, required=True)
    s.add_argument("--new", type=Path, default=None)
    s.add_argument("--pool-only", type=int, nargs="+", default=None, metavar="EP")
    s.add_argument("--harvest", type=Path, default=None)
    for name in ("pool", "probe", "scores"):
        s = sub.add_parser(name)
        s.add_argument("--reference", type=Path, required=True)
        s.add_argument("--new", type=Path, default=None)
    s = sub.add_parser("panel")
    s.add_argument("--new", type=Path, default=None)
    s.add_argument("--ref-pca", type=Path, default=None)
    s.add_argument("--ref-scores", type=Path, default=None)
    s.add_argument("--ref-probe", type=Path, default=None)
    a = ap.parse_args(argv)
    {"cache": cmd_cache, "pool": cmd_pool, "probe": cmd_probe, "scores": cmd_scores,
     "panel": cmd_panel}[a.cmd](a)
    print("IDENTICAL" if not FAILS else f"DIFFERENT ({len(FAILS)}): {FAILS[:10]}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
