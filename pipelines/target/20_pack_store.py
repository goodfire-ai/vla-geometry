#!/usr/bin/env python
"""Pack the object-teleport house files into memory-mappable arrays (CPU).

Reads <store>/houses/h*.npz (10_teleport_harvest.py) and writes <store>/prep/:
  X_{variant}.npy   (N, 13, 2560) float16, one per VLM token group (11 files)
  X_ae.npy          (N, 11, 768) float16
  X_ae_tok.npy      (N, 11, 16, 768) float16
  X_chunk.npy       (N, 16, 8) float32
  labels.npz        per-row labels of the house files (obj_base, ee, rel_ee, u, cmd_xy, dist, fwd,
                    bearing_deg, settle_disp, config_id, chunk_q, step, lean_contacts, in_env) plus
                    house, obj_type, bench and g_conv (approach convergence over the 4 queries:
                    the object-to-EE distance shrinks by >= 5 cm, or is < 15 cm at the last query)
  meta.json         {n, layers_llm, layers_ae, n_tok, houses, n_houses, gate_pass_frac}

Rows are in lexical order of the house file names, then in file order. The store needs about
50 GB for 906 houses; packing takes about 2 minutes with 16 worker processes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.target import PACK_STREAMS, pack_one, teleport_root


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=None,
                    help="store root holding houses/ (default $VLAGEOM_DATA/target/teleport)")
    ap.add_argument("--houses-dir", type=Path, default=None, help="default <store>/houses")
    ap.add_argument("--out", type=Path, default=None, help="default <store>/prep")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--house-list", type=Path, default=paths.lists_dir() / "teleport_houses.csv",
                    help="the houses expected in houses/ (a mismatch is reported, not fatal)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing packed store")
    args = ap.parse_args()

    store = args.store or teleport_root()
    hdir = args.houses_dir or store / "houses"
    out = args.out or store / "prep"
    if (out / "meta.json").exists() and not args.overwrite:
        raise SystemExit(f"{out} already holds a packed store (use --overwrite to replace it)")

    files = sorted(hdir.glob("h*.npz"))
    if not files:
        raise SystemExit(f"no house files under {hdir}")
    found = {int(f.stem[1:]) for f in files}
    if args.house_list and args.house_list.exists():
        with open(args.house_list, newline="") as fh:
            want = {int(r["house"]) for r in csv.DictReader(fh)}
        if found != want:
            print(f"note: {len(found)} house files; {len(want - found)} listed houses missing "
                  f"(e.g. {sorted(want - found)[:10]}), {len(found - want)} not in the list",
                  file=sys.stderr)
    print(f"pack: {len(files)} house files -> {out}", flush=True)
    out.mkdir(parents=True, exist_ok=True)

    z0 = np.load(files[0], allow_pickle=False)
    layers_llm = z0["layers_llm"].tolist()
    layers_ae = z0["layers_ae"].tolist()
    n_tok = int(z0["ae_tok"].shape[2])
    counts = [len(np.load(f, allow_pickle=False)["dist"]) for f in files]
    n_total = int(np.sum(counts))
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)

    ck0 = z0["chunk"].shape
    mm_paths = {}
    for v in PACK_STREAMS:
        dtype = np.float16
        if v == "ae":
            shape = (n_total, len(layers_ae), 768)
        elif v == "ae_tok":
            shape = (n_total, len(layers_ae), n_tok, 768)
        elif v == "chunk":
            shape = (n_total, ck0[1], ck0[2])
            dtype = np.float32
        else:
            shape = (n_total, len(layers_llm), 2560)
        p = out / f"X_{v}.npy"
        np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=shape)
        mm_paths[v] = str(p)

    jobs = [(str(f), int(o), PACK_STREAMS, mm_paths) for f, o in zip(files, offs)]
    labs = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for row0, n, lab in ex.map(pack_one, jobs):
            labs[row0] = lab
            print(f"  rows {row0}..{row0 + n}", flush=True)
    order = sorted(labs)
    merged = {k: np.concatenate([labs[o][k] for o in order]) for k in labs[order[0]]}
    np.savez(out / "labels.npz", **merged)

    houses = sorted(set(merged["house"].tolist()))
    meta = {"n": n_total, "layers_llm": layers_llm, "layers_ae": layers_ae, "n_tok": n_tok,
            "houses": houses, "n_houses": len(houses),
            "gate_pass_frac": float(merged["g_conv"].mean())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"pack done: N={n_total} rows, {len(houses)} houses, "
          f"approach-convergence pass {meta['gate_pass_frac']:.2f}", flush=True)


if __name__ == "__main__":
    main()
