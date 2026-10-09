#!/usr/bin/env python
"""Pack the crossed-store house files into memory-mappable arrays (CPU).

Reads <store>/houses/h*.npz (10_crossed_harvest.py) and writes <store>/prep/:
  X_{variant}.npy     (N, 36, 2560) float16 for every VLM token group in --streams
  X_ae.npy, X_ae_tok.npy, X_chunk.npy
                      (N, 36, 768) / (N, 36, 16, 768) float16 and (N, 16, 8) float32, when listed
  labels.npz          per-row labels: obj_base, ee, rel_ee, u, cmd_xy, ee_target, ee_tele, qpos_set,
                      qpos_q, obj_world, dist, fwd, bearing_deg, settle_disp, ik_err, exo_px,
                      config_id, pose_id, chunk_q, step, lean_contacts, in_env, ee44, plus house,
                      obj_type and bench
  meta.json           {n, layers_llm, layers_ae, n_tok, variants_llm, houses, n_houses,
                      rows_per_grain, ee_std_cm, obj_std_cm}
  excluded_houses.txt the houses left out (one per line)

A house is left out when its exo-camera frame does not change across its rows (the spread of the
mean pixel value is below 1e-6, every 64 px thumbnail is identical, or the median pixel value is
below 20): in such a scene the exo camera sees neither the moved arm nor the moved object. Of the
947 harvested houses this leaves out houses 499 and 567, so the store holds 945 houses and 90,814
rows (45,407 per query).

Rows are in lexical order of the house file names, then in file order. --streams sets the arrays
that are packed: by default the two token groups Figure 2 reads (hdr_txt, txt_instr; 2 x 16.7 GB);
`--streams all` packs every token group, the action expert and the chunk (252 GB). The labels and
meta.json do not depend on --streams.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vlageom.position import (ALL_STREAMS, PACK_DEFAULT_STREAMS, crossed_house_list_path,
                              crossed_root, exo_unchanging, pack_one)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--store", type=Path, default=None,
                    help="store root holding houses/ (default $VLAGEOM_DATA/position/crossed)")
    ap.add_argument("--houses-dir", type=Path, default=None, help="default <store>/houses")
    ap.add_argument("--out", type=Path, default=None, help="default <store>/prep")
    ap.add_argument("--streams", nargs="+", default=PACK_DEFAULT_STREAMS,
                    help=f"arrays to pack: token groups, ae, ae_tok, chunk, or 'all' "
                         f"(default: {' '.join(PACK_DEFAULT_STREAMS)})")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--house-list", type=Path, default=crossed_house_list_path(),
                    help="the houses expected in houses/ (a mismatch is reported, not fatal)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing packed store")
    args = ap.parse_args()

    streams = ALL_STREAMS if args.streams == ["all"] else list(args.streams)
    bad = [s for s in streams if s not in ALL_STREAMS]
    if bad:
        raise SystemExit(f"unknown streams {bad}; choose from {ALL_STREAMS} or 'all'")
    store = args.store or crossed_root()
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
    out.mkdir(parents=True, exist_ok=True)

    keep, excluded = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        (excluded.append(int(z["house"])) if exo_unchanging(z) else keep.append(f))
    (out / "excluded_houses.txt").write_text("".join(f"{h}\n" for h in excluded))
    print(f"pack: {len(files)} house files, {len(excluded)} left out {excluded} -> {out}",
          flush=True)
    files = keep

    z0 = np.load(files[0], allow_pickle=False)
    layers_llm = z0["layers_llm"].tolist()
    layers_ae = z0["layers_ae"].tolist()
    n_tok = int(z0["ae_tok"].shape[2])
    variants_llm = [str(v) for v in z0["variants"]]
    counts = [len(np.load(f, allow_pickle=False)["dist"]) for f in files]
    n_total = int(np.sum(counts))
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)

    ck0 = z0["chunk"].shape
    mm_paths = {}
    for v in streams:
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

    jobs = [(str(f), int(o), streams, mm_paths) for f, o in zip(files, offs)]
    labs = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for row0, n, lab in ex.map(pack_one, jobs):
            labs[row0] = lab
            if len(labs) % 100 == 0:
                print(f"  {len(labs)}/{len(files)} houses", flush=True)
    order = sorted(labs)
    merged = {k: np.concatenate([labs[o][k] for o in order]) for k in labs[order[0]]}
    np.savez(out / "labels.npz", **merged)

    houses = sorted(set(merged["house"].tolist()))
    q = merged["chunk_q"]
    meta = {"n": n_total, "layers_llm": layers_llm, "layers_ae": layers_ae, "n_tok": n_tok,
            "variants_llm": variants_llm, "houses": houses, "n_houses": len(houses),
            "rows_per_grain": {"q0": int((q == 0).sum()), "q1": int((q == 1).sum())},
            "ee_std_cm": (merged["ee"].std(0) * 100).round(2).tolist(),
            "obj_std_cm": (merged["obj_base"].std(0) * 100).round(2).tolist()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"pack done: N={n_total} rows, {len(houses)} houses, per query {meta['rows_per_grain']}, "
          f"packed {streams}", flush=True)


if __name__ == "__main__":
    main()
