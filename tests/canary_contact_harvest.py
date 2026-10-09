#!/usr/bin/env python
"""GPU canaries of the two contact harvests (Figure 4): one episode, compared with a reference file.

  natural      one natural harvest file (pipelines/contact/10_natural_harvest.py)
      python pipelines/contact/10_natural_harvest.py --task 0 --out $C/contact
      python tests/canary_contact_harvest.py natural \
          --new $C/contact/natural/ep0_h0.npz --reference <reference natural harvest dir>/ep0_h0.npz

      Checked: the same keys in the same order with the same dtypes and per-query shapes;
      identical scene identifiers (layers, house, ep, seed, descr) and identical text tail
      (txt_ids, txt_str), i.e. the same task instruction; the rollout fields (n_queries, n_steps,
      base_success, q_idx, query_steps_kept, contact_series, replay_grip, replay_grasp,
      layout_imgs) bit for bit; every query's activations within --rtol relative L2 for every
      layer and token group. This is the check on the GPU type, driver and library versions of
      the reference run, where the bfloat16 rollout repeats exactly.

      --lenient (other hardware, where two runs of an episode can separate after some queries):
      only the first query (the scene after reset) is required within --rtol; the later queries
      and the rollout fields are reported (relative L2 per query, the first query above --rtol,
      agreement of contact_series and of the executed gripper commands, n_queries and n_steps).

  offschedule  one off-schedule harvest file (pipelines/contact/30_offschedule_harvest.py)
      python pipelines/contact/30_offschedule_harvest.py --task 0 --out $C/contact
      python tests/canary_contact_harvest.py offschedule \
          --new $C/contact/offschedule/ep0_h0.npz --reference <reference off-schedule dir>/ep0_h0.npz

      Checked: the same keys in the same order and dtypes; every placement and pose field
      (contact, disp, dist, pose_id, step_k, event, phase, qc, grasp0, pose_chunk, qpos0, pose_event,
      pose_qc, pose_sep0, pose_grasp_forced, pose_dir, layers, house, ep, n_txt, txt_ids, txt_str)
      bit for bit; `act` within --rtol relative L2 per placement and layer, reported per token group
      (exo patches, wrist patches, text tail, sequence mean).

Activations come from bfloat16 inference on the GPU and EGL rendering, so a canary matches the
reference closely only on the same GPU type, driver and library versions. Exit status 0 when
every check passes.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

from vlageom.util import npz_member

FAILS: list[str] = []


def report(name, ok, detail=""):
    print(f"  {'OK   ' if ok else 'FAIL '} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def info(name, detail):
    print(f"  info  {name}  {detail}")


def member_order(path):
    with zipfile.ZipFile(path) as zf:
        return [n[:-4] for n in zf.namelist()]


def exact(name, x, y):
    x, y = np.asarray(x), np.asarray(y)
    ok = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y, equal_nan=(x.dtype.kind == "f"))
    det = f"{y.dtype}{y.shape}"
    if not ok:
        if x.dtype != y.dtype or x.shape != y.shape:
            det = f"new {x.dtype}{x.shape} vs ref {y.dtype}{y.shape}"
        elif x.dtype.kind == "f":
            det += f" max|diff| {float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))):.4g}"
    report(name, ok, det)


def rel_l2(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def header(path, key):
    """(shape, dtype) of member `key` of an .npz file, read from its .npy header only."""
    with zipfile.ZipFile(path) as zf, zf.open(key + ".npy") as f:
        ver = np.lib.format.read_magic(f)
        rd = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
              else np.lib.format.read_array_header_2_0)
        shape, _fortran, dtype = rd(f)
    return tuple(shape), np.dtype(dtype)


def big(path, key):
    """Member `key` of an .npz file as a read-only memory map (uncompressed members, as written by
    the harvest stages); arrays are read from disk only where they are indexed."""
    off, shape, dtype, fortran = npz_member(path, key)
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=tuple(shape),
                     order="F" if fortran else "C")


def schema(new, ref, per_row=()):
    """Same keys in the same order; same dtypes; same shapes (per-row keys: same shape after the
    first axis). Reads the .npy headers only. Returns the two opened files."""
    on, orf = member_order(new), member_order(ref)
    report("keys and their order", on == orf, "" if on == orf else f"new {on} ref {orf}")
    for k in orf:
        if k not in on:
            continue
        (xs, xd), (ys, yd) = header(new, k), header(ref, k)
        if k in per_row:
            ok = xd == yd and xs[1:] == ys[1:]
            report(f"{k} dtype and per-row shape", ok, f"new {xd}{xs} ref {yd}{ys}")
        elif xd != yd:
            report(f"{k} dtype", False, f"new {xd} ref {yd}")
    return np.load(new), np.load(ref)


def cmd_natural(a):
    print(f"new {a.new}\nref {a.reference}")
    per_row = ("act_img", "act_txt", "act_seqmean", "q_idx", "query_steps_kept", "layout_imgs",
               "replay_grasp", "replay_grip", "contact_series")
    z, r = schema(a.new, a.reference, per_row)
    for k in ("layers", "house", "ep", "seed", "descr", "txt_ids", "txt_str"):
        exact(k, z[k], r[k])
    groups = {"exo": (slice(None), slice(0, 196)), "wrist": (slice(None), slice(196, 392))}
    ai, bi = big(a.new, "act_img"), big(a.reference, "act_img")
    at, bt = big(a.new, "act_txt"), big(a.reference, "act_txt")
    asq, bsq = big(a.new, "act_seqmean"), big(a.reference, "act_seqmean")
    L = bi.shape[1]
    if ai.shape[1:] != bi.shape[1:] or at.shape[1:] != bt.shape[1:]:
        report("query 0 activations", False, "per-query shapes differ")
        return

    def query_worst(q):
        """Largest relative L2 of query q over the layers and token groups."""
        w = 0.0
        for li in range(L):
            for _g, (_, sl) in groups.items():
                w = max(w, rel_l2(ai[q, li, sl], bi[q, li, sl]))
            w = max(w, rel_l2(at[q, li], bt[q, li]), rel_l2(asq[q, li], bsq[q, li]))
        return w

    nq = min(len(ai), len(bi))
    per_q = [query_worst(q) for q in range(nq)]
    first = next((q for q, v in enumerate(per_q) if v > a.rtol), None)
    report(f"query 0 activations within rtol {a.rtol:g}", per_q[0] <= a.rtol,
           f"max relative L2 over layers and token groups {per_q[0]:.3g}")
    if not a.lenient:
        for k in ("n_queries", "n_steps", "base_success", "q_idx", "query_steps_kept",
                  "contact_series", "replay_grip", "replay_grasp", "layout_imgs"):
            exact(k, z[k], r[k])
        report(f"every query's activations within rtol {a.rtol:g}",
               len(ai) == len(bi) and first is None,
               f"{nq} common queries, max relative L2 {max(per_q):.3g}"
               + ("" if first is None else f"; first above rtol: q{first}"))
    info("queries", f"new {len(ai)} / ref {len(bi)}; n_steps new {int(z['n_steps'])} / ref "
                    f"{int(r['n_steps'])}; success new {bool(z['base_success'])} / ref "
                    f"{bool(r['base_success'])}")
    info("first query above rtol", "none" if first is None else
         f"q{first} (relative L2 {per_q[first]:.3g}; queries before it agree)")
    info("relative L2 per query", " ".join(f"{v:.2g}" for v in per_q[:16]) + (" ..." if nq > 16 else ""))
    cs, cr = z["contact_series"], r["contact_series"]
    n = min(len(cs), len(cr))
    info("contact_series", f"agree on {float((cs[:n] == cr[:n]).mean()):.3f} of {n} common steps; "
                           f"on-fraction new {cs.mean():.3f} / ref {cr.mean():.3f}")
    g1, g2 = z["replay_grip"], r["replay_grip"]
    m = min(len(g1), len(g2))
    dg = np.flatnonzero(g1[:m] != g2[:m])
    info("executed gripper", "identical over the common queries" if not len(dg)
         else f"first difference at query {int(dg[0])}")


def cmd_offschedule(a):
    print(f"new {a.new}\nref {a.reference}")
    z, r = schema(a.new, a.reference, per_row=("act",))
    for k in ("contact", "disp", "dist", "pose_id", "step_k", "event", "phase", "qc", "grasp0",
              "pose_chunk", "qpos0", "pose_event", "pose_qc", "pose_sep0", "pose_grasp_forced",
              "pose_dir", "layers", "house", "ep", "n_txt", "txt_ids", "txt_str"):
        exact(k, z[k], r[k])
    x, y = big(a.new, "act"), big(a.reference, "act")
    if x.shape != y.shape:
        report("act shape", False, f"new {x.shape} ref {y.shape}")
        return
    T = int(r["n_txt"])
    groups = {"exo": slice(0, 196), "wrist": slice(196, 392), "text": slice(392, 392 + T),
              "seqmean": slice(392 + T, 393 + T)}
    worst = {g: 0.0 for g in groups}
    for i in range(len(y)):
        xi, yi = np.asarray(x[i], np.float32), np.asarray(y[i], np.float32)
        for li in range(y.shape[1]):
            for g, sl in groups.items():
                worst[g] = max(worst[g], rel_l2(xi[li, sl], yi[li, sl]))
    for g, v in worst.items():
        report(f"act {g} within rtol {a.rtol:g}", v <= a.rtol,
               f"max relative L2 per placement and layer {v:.3g}")
    ev = np.asarray(r["event"])
    info("placements", f"{len(y)}; events 0/1/2/3: " + "/".join(str(int((ev == e).sum())) for e in range(4)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, rtol in (("natural", 1e-3), ("offschedule", 1e-3)):
        s = sub.add_parser(name)
        s.add_argument("--new", type=Path, required=True)
        s.add_argument("--reference", type=Path, required=True)
        s.add_argument("--rtol", type=float, default=rtol,
                       help="relative L2 tolerance of the activations (default %(default)g)")
        if name == "natural":
            s.add_argument("--lenient", action="store_true",
                           help="require only the first query (for hardware other than the "
                                "reference run's); report the rest")
    a = ap.parse_args(argv)
    {"natural": cmd_natural, "offschedule": cmd_offschedule}[a.cmd](a)
    print("PASS" if not FAILS else f"FAIL ({len(FAILS)}): {FAILS[:10]}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
