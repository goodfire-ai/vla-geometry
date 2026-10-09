#!/usr/bin/env python
"""Compare the Figure 3 stage outputs and panel files with reference files.

Every value is compared for exact equality (NaN equals NaN); the largest absolute difference is
printed otherwise. Subcommands:

  erasure  stage 50 output (erase_far.json / erase_near.json, or the directory holding them)
           against a reference erasure result: per erased range and scored k, n / R^2 / MAE /
           slope of both range probes, the penalties and the fold test-row md5. The reference is
           either a stage 50 output of the same form, or one json with both directions in the
           layout {"t4": {"deflate_FAR15_probe_NEAR15": [...], "deflate_NEAR15_probe_FAR15":
           [...]}, "modal_alpha", "alphas", "folds": {"hash"}} with entries {"k", "tgt",
           "src_kill"} (tgt = the range whose probe is not erased). Only the k present in both are
           compared, so a run with --k-max 2 checks the first three points.
  bank     stage 60 output (.npz and its .json) against a reference bank: row arrays (dist,
           house, fold_id), per probe the out-of-fold read-outs, the per-fold weights and the
           split-half ceilings, and from the .json files the per-fold weight cosines of the
           near/far pairs and every probe's ceilings. Only probes present in both are compared.
  bank-subset
           recomputes part of the bank from the packed stores (the folds given by --folds and the
           first --draws ceiling draws, at --threads BLAS threads) and compares it with the same
           part of a reference bank .npz; a short check of stage 60 without the full run.
  table    stage 70 output against a reference table (same layout): n, MAE and R^2 of every
           family, boundary and range present in the reference.
  plant    stage 80 output against a reference (same layout): per boundary and pair the
           cosine mean / std / min / max, folds, weaker ceiling; the cosines to the planted
           direction.
  panels   the panel files of stage 90 against the data records written next to reference
           renders (JSON): --ref-a {"k", "series": {"near|erase_far", "far|erase_far",
           "near|erase_near", "far|erase_near"}}, --ref-b {"grid_cm", "series": {family:
           {"near_r2", "far_r2"}}}, --ref-c {"grid_cm", "split"|"round": {"cos_mean",
           "cos_std_folds", "ceil_weaker_median"}, "plant": {"split_cos_mean",
           "round_cos_mean"}}, --ref-d {"mean_path": {"far_m", "cm"}} and, for the points of
           panel d, --ref-points {"samples": {"stratified": {"d_cm", "far15", "cm15"}}}.

  python tests/compare_fig3.py erasure --new $VLAGEOM_DATA/distance/erasure --reference <file>
  python tests/compare_fig3.py bank --new $VLAGEOM_DATA/distance/probe_bank_last_tok_L21.npz \
      --reference <bank .npz> [--ref-json <bank .json>]
  python tests/compare_fig3.py bank-subset --reference <bank .npz> [--folds 0] [--draws 1]
  python tests/compare_fig3.py table --new $VLAGEOM_DATA/distance/cutoff_table_last_tok_L21.json \
      --reference <table .json>
  python tests/compare_fig3.py plant --new $VLAGEOM_DATA/distance/plant_cosines.json \
      --reference <plant .json>
  python tests/compare_fig3.py panels [--panels DIR] --ref-a <a.json> --ref-b <b.json> \
      --ref-c <c.json> --ref-d <d.json> --ref-points <points .json>

Exit status 0 when everything compared is identical.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RANGES = ("near", "far")
SCORES = ("n", "r2", "mae_cm", "slope")


class Tally:
    """Counts compared and differing values; prints the differences."""

    def __init__(self):
        self.n = 0
        self.bad = []

    def value(self, name, new, ref):
        self.n += 1
        same = (new == ref) or (isinstance(new, float) and isinstance(ref, float)
                                and math.isnan(new) and math.isnan(ref))
        if not same:
            self.bad.append(name)
            d = abs(new - ref) if isinstance(new, (int, float)) and isinstance(ref, (int, float)) \
                else None
            print(f"  DIFF {name}: new {new!r}, ref {ref!r}" + (f" (|d| {d:.3g})" if d else ""))

    def array(self, name, new, ref):
        self.n += 1
        new, ref = np.asarray(new), np.asarray(ref)
        if new.shape != ref.shape:
            self.bad.append(name)
            print(f"  DIFF {name}: shape {new.shape} vs {ref.shape}")
            return
        if new.dtype.kind == "f" or ref.dtype.kind == "f":
            same = np.array_equal(new, ref, equal_nan=True)
        else:
            same = np.array_equal(new, ref)
        if not same:
            self.bad.append(name)
            if new.dtype.kind in "fiu" and ref.dtype.kind in "fiu":
                d = np.abs(new.astype(np.float64) - ref.astype(np.float64))
                d = d[np.isfinite(d)]
                print(f"  DIFF {name}: {int((new != ref).sum())} entries differ, max |d| "
                      f"{d.max() if d.size else float('nan'):.3g}")
            else:
                print(f"  DIFF {name}")

    def report(self, what):
        print(f"{what}: {self.n} values compared, {len(self.bad)} different")
        print("RESULT", "IDENTICAL" if not self.bad else "DIFFERENT")
        return 0 if not self.bad else 1


# ---- erasure ----------------------------------------------------------------------------------- #
def _reference_erasure(path):
    """{erase: {"curve": {k: {"near": scores, "far": scores}}, "alpha", "alpha_per_fold",
    "fold_md5"}} from either reference layout."""
    path = Path(path)
    files = sorted(path.glob("erase_*.json")) if path.is_dir() else [path]
    out = {}
    for f in files:
        r = json.loads(f.read_text())
        if "t4" in r:
            name = {"NEAR15": "near", "FAR15": "far"}
            for key, curve in r["t4"].items():
                src, tgt = key.replace("deflate_", "").split("_probe_")
                erase, other = name[src], name[tgt]
                out[erase] = {
                    "curve": {c["k"]: {other: c["tgt"], erase: c["src_kill"]}
                              for c in curve if "tgt" in c},
                    "alpha": {name[b]: v for b, v in r["modal_alpha"].items()},
                    "alpha_per_fold": {name[b]: v for b, v in r["alphas"].items()},
                    "fold_md5": r["folds"]["hash"]}
        else:
            out[r["erase"]] = {"curve": {c["k"]: c for c in r["curve"]}, "alpha": r["alpha"],
                               "alpha_per_fold": r["alpha_per_fold"],
                               "fold_md5": r["fold_test_rows_md5"]}
    return out


def cmd_erasure(a):
    new = _reference_erasure(a.new)
    ref = _reference_erasure(a.reference)
    t = Tally()
    for erase in sorted(new):
        if erase not in ref:
            print(f"erase {erase}: not in the reference")
            continue
        n, r = new[erase], ref[erase]
        t.value(f"{erase}/fold_md5", n["fold_md5"], r["fold_md5"])
        for rg in RANGES:
            t.value(f"{erase}/alpha/{rg}", float(n["alpha"][rg]), float(r["alpha"][rg]))
            t.value(f"{erase}/alpha_per_fold/{rg}", [float(v) for v in n["alpha_per_fold"][rg]],
                    [float(v) for v in r["alpha_per_fold"][rg]])
        ks = sorted(set(n["curve"]) & set(r["curve"]))
        print(f"erase {erase}: k compared {ks}")
        for k in ks:
            for rg in RANGES:
                for s in SCORES:
                    t.value(f"{erase}/k{k}/{rg}/{s}", n["curve"][k][rg][s], r["curve"][k][rg][s])
    return t.report("erasure")


# ---- bank -------------------------------------------------------------------------------------- #
def _bank_npz(path):
    z = np.load(path, allow_pickle=False)
    probes = sorted({k.split("__", 1)[1] for k in z.files if k.startswith("oof__")})
    return z, probes


def _rows(t, new_dist, new_house, new_fold, ref):
    t.array("rows/dist", np.asarray(new_dist, np.float32), ref["dist"])
    t.array("rows/house", np.asarray(new_house).astype(np.int64),
            np.asarray(ref["house"]).astype(np.int64))
    if new_fold is not None:
        t.array("rows/fold_id", np.asarray(new_fold, np.int64), ref["fold_id"].astype(np.int64))


def cmd_bank(a):
    zn, pn = _bank_npz(a.new)
    zr, pr = _bank_npz(a.reference)
    common = [p for p in pn if p in pr]
    print(f"probes: {len(pn)} new, {len(pr)} reference, {len(common)} compared")
    t = Tally()
    _rows(t, zn["dist"], zn["house"], zn["fold_id"], zr)
    for p in common:
        for key in ("oof", "w_raw", "ceil"):
            t.array(f"{key}/{p}", zn[f"{key}__{p}"], zr[f"{key}__{p}"])
    ref_json = a.ref_json or Path(a.reference).with_suffix(".json")
    new_json = Path(a.new).with_suffix(".json")
    if ref_json.exists() and new_json.exists():
        jn, jr = json.loads(new_json.read_text()), json.loads(ref_json.read_text())
        for p in common:
            t.value(f"catalog/{p}/ceil_raw_cos", jn["catalog"][p]["ceil_raw_cos"],
                    jr["catalog"][p]["ceil_raw_cos"])
        pf_n, pf_r = jn["cosines"]["per_fold"], jr["cosines"]["per_fold"]
        for key in sorted(pf_n):
            if key in pf_r:
                for s in ("mean", "std", "n_folds"):
                    t.value(f"cosines/{key}/{s}", pf_n[key][s], pf_r[key][s])
    else:
        print(f"(no .json pair next to the files, cosines not compared: {new_json}, {ref_json})")
    return t.report("bank")


def cmd_bank_subset(a):
    from threadpoolctl import threadpool_limits

    from vlageom.distance import (BANK_FOLDS, bank_rows, fit_bank, near_root,
                                  split_half_ceilings, teleport_prep)
    from vlageom.probes import FoldMap

    zr, pr = _bank_npz(a.reference)
    folds = FoldMap.load(a.fold_map or BANK_FOLDS)
    with threadpool_limits(limits=a.threads):
        X, lab = bank_rows(a.teleport_store or teleport_prep(),
                           a.near_store or near_root() / "prep", folds)
        d_cm = lab["dist"].astype(np.float64) * 100
        houses = lab["house"]
        splits = folds.splits(houses)
        t = Tally()
        _rows(t, lab["dist"], houses, None, zr)
        fold_id = np.asarray(zr["fold_id"], np.int64)
        for f in a.folds:
            t.array(f"rows/fold{f}_test", splits[f][1], np.nonzero(fold_id == f)[0])
        for f in a.folds:
            print(f"fold {f}: fitting", flush=True)
            bank = fit_bank(X, d_cm, houses, [splits[f]])     # fold f's fits land at index 0
            te = splits[f][1]
            for p in bank["oof"]:
                if p not in pr:
                    continue
                t.array(f"oof/{p}/fold{f}", bank["oof"][p][te].astype(np.float32),
                        zr[f"oof__{p}"][te])
                t.array(f"w_raw/{p}/fold{f}", bank["w_raw"][p][0].astype(np.float32),
                        zr[f"w_raw__{p}"][f])
        if a.draws > 0:
            print(f"ceilings: {a.draws} draw(s)", flush=True)
            ceil = split_half_ceilings(X, d_cm, houses, n_draws=a.draws)
            for p in ceil:
                if p in pr:
                    t.array(f"ceil/{p}", ceil[p], zr[f"ceil__{p}"][: a.draws])
    return t.report("bank subset")


# ---- table, plant ------------------------------------------------------------------------------ #
def cmd_table(a):
    jn, jr = json.loads(Path(a.new).read_text()), json.loads(Path(a.reference).read_text())
    t = Tally()
    for c, regs in jr["table"].items():
        for reg, e in regs.items():
            for fam, met in e["metrics"].items():
                got = jn["table"].get(c, {}).get(reg, {}).get("metrics", {}).get(fam)
                if got is None:
                    print(f"  ({c}/{reg}/{fam} not in the new table)")
                    continue
                for s in ("n", "mae_cm", "r2"):
                    t.value(f"{c}/{reg}/{fam}/{s}", got[s], met[s])
    return t.report("table")


def cmd_plant(a):
    jn, jr = json.loads(Path(a.new).read_text()), json.loads(Path(a.reference).read_text())
    t = Tally()
    t.value("grid_cm", jn["grid_cm"], jr["grid_cm"])
    for fam in ("split", "round"):
        for rn, rr in zip(jn[fam], jr[fam]):
            for s in rr:
                t.value(f"{fam}/c{rr['c']}/{s}", rn[s], rr[s])
    for k, v in jr["cos_to_planted_axis"].items():
        t.value(f"cos_to_planted_axis/{k}", jn["cos_to_planted_axis"][k], v)
    return t.report("plant")


# ---- panels ------------------------------------------------------------------------------------ #
def cmd_panels(a):
    from vlageom import paths
    pdir = a.panels or paths.panels_dir()
    t = Tally()
    if a.ref_a:
        p, r = np.load(pdir / "fig3a_erasure.npz"), json.loads(a.ref_a.read_text())
        t.array("a/k", p["k"], np.asarray(r["k"], np.int64))
        for key, name in (("near|erase_far", "near_erase_far"), ("far|erase_far", "far_erase_far"),
                          ("near|erase_near", "near_erase_near"),
                          ("far|erase_near", "far_erase_near")):
            t.array(f"a/{name}", p[name], np.asarray(r["series"][key], np.float64))
    if a.ref_b:
        p, r = np.load(pdir / "fig3b_cutoff.npz"), json.loads(a.ref_b.read_text())
        t.array("b/grid_cm", p["grid_cm"], np.asarray(r["grid_cm"], np.int64))
        for fam in ("split", "clip", "n1"):
            for reg in RANGES:
                t.array(f"b/{fam}_{reg}", p[f"{fam}_{reg}"],
                        np.asarray(r["series"][fam][f"{reg}_r2"], float))
    if a.ref_c:
        p, r = np.load(pdir / "fig3c_cosine.npz"), json.loads(a.ref_c.read_text())
        t.array("c/grid_cm", p["grid_cm"], np.asarray(r["grid_cm"], np.int64))
        for fam in ("split", "round"):
            for key, name in (("cos_mean", "cos"), ("cos_std_folds", "cos_std"),
                              ("ceil_weaker_median", "ceil")):
                t.array(f"c/{fam}_{name}", p[f"{fam}_{name}"], np.asarray(r[fam][key], float))
            t.array(f"c/{fam}_plant", p[f"{fam}_plant"],
                    np.asarray(r["plant"][f"{fam}_cos_mean"], float))
    if a.ref_d:
        p, r = np.load(pdir / "fig3d_plane.npz"), json.loads(a.ref_d.read_text())
        t.array("d/path_x_m", p["path_x_m"], np.asarray(r["mean_path"]["far_m"], float))
        t.array("d/path_y_cm", p["path_y_cm"], np.asarray(r["mean_path"]["cm"], float))
    if a.ref_points:
        p = np.load(pdir / "fig3d_plane.npz")
        s = json.loads(a.ref_points.read_text())["samples"]["stratified"]
        t.array("d/d_cm", p["d_cm"], np.asarray(s["d_cm"], float))
        t.array("d/x_m", p["x_m"], np.asarray(s["far15"], float) / 100.0)
        t.array("d/y_cm", p["y_cm"], np.asarray(s["cm15"], float))
    return t.report("panels")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("erasure")
    s.add_argument("--new", type=Path, required=True, help="stage 50 output file or directory")
    s.add_argument("--reference", type=Path, required=True)
    s = sub.add_parser("bank")
    s.add_argument("--new", type=Path, required=True, help="stage 60 output .npz")
    s.add_argument("--reference", type=Path, required=True, help="reference bank .npz")
    s.add_argument("--ref-json", type=Path, default=None,
                   help="reference bank .json (default: next to --reference)")
    s = sub.add_parser("bank-subset")
    s.add_argument("--reference", type=Path, required=True, help="reference bank .npz")
    s.add_argument("--folds", type=int, nargs="+", default=[0], help="folds to recompute")
    s.add_argument("--draws", type=int, default=1, help="ceiling draws to recompute (first n)")
    s.add_argument("--threads", type=int, default=None,
                   help="BLAS threads (default: stage 60's, vlageom.distance.BANK_THREADS)")
    s.add_argument("--fold-map", default=None, help="fold map (default: stage 60's)")
    s.add_argument("--teleport-store", type=Path, default=None)
    s.add_argument("--near-store", type=Path, default=None)
    for name in ("table", "plant"):
        s = sub.add_parser(name)
        s.add_argument("--new", type=Path, required=True)
        s.add_argument("--reference", type=Path, required=True)
    s = sub.add_parser("panels")
    s.add_argument("--panels", type=Path, default=None,
                   help="panel directory (default $VLAGEOM_DATA/panels)")
    for k in ("a", "b", "c", "d"):
        s.add_argument(f"--ref-{k}", type=Path, default=None)
    s.add_argument("--ref-points", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "bank-subset" and a.threads is None:
        from vlageom.distance import BANK_THREADS
        a.threads = BANK_THREADS
    return {"erasure": cmd_erasure, "bank": cmd_bank, "bank-subset": cmd_bank_subset,
            "table": cmd_table, "plant": cmd_plant, "panels": cmd_panels}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
