"""Contact (Figure 4), stage 60 (CPU): off-schedule detection with held-out-house probes (panel b).

Probe: for each pool (txt_all, last_tok) at layer 24, StandardScaler + LDA fit on natural rows
(natural cache, stage 20). Every off-schedule placement of house h (stage 40) is scored by the
probe fit on the natural rows of all houses except h (one fit per house and pool; a house without
natural rows would keep the probe fit on all natural rows; every off-schedule house has natural
rows).

Scores per cell (vlageom.contact.CELLS: early_touch_open = event 0, early_touch_closed = event 2,
slip_closed = event 1, slip_open_out / slip_open_in = event 3 with the object moved away from /
toward the opened hand), and over all rows; within_pose over the poses with both contact classes:
frac_pos = fraction of poses whose mean contact score exceeds their mean no-contact score (the
bars of panel b, "P(correct)"), with SE sqrt(p (1 - p) / n_pose); auc = within-pose AUC weighted by
contact x no-contact pairs, with a Kish SE; mean_delta; n_pose; n_rows. pose_matched: the
forced-hand minus natural-hand per-pose AUC at the same anchor, on rows within 3 steps of the
contact change. natural_auc = the out-of-house AUC of the same pool on the natural rows (stage 50).

Output: $VLAGEOM_DATA/contact/detection_heldout.json (--out), keys n_pts, n_eps, counts,
phase_timer_within_pose_auc (0.5: a phase timer is constant within a pose), cache, subdir,
probe, variants.{txt_all,last_tok}. Each refit runs on one BLAS thread in one of --workers forked
processes; about 8 minutes with 12 workers, 20-60 minutes with 4 on a shared machine.

  python pipelines/contact/60_heldout_scores.py [--cache FILE] [--pool FILE] [--probe FILE] [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from vlageom import contact as C  # noqa: E402
from vlageom import paths  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--cache", type=Path, default=None,
                    help="natural cache (default $VLAGEOM_DATA/contact/natural_cache.npz)")
    ap.add_argument("--pool", type=Path, default=None,
                    help="off-schedule pool (default $VLAGEOM_DATA/contact/offschedule_L24.npz)")
    ap.add_argument("--probe", type=Path, default=None,
                    help="natural probe scores (default $VLAGEOM_DATA/contact/natural_probe.json)")
    ap.add_argument("--workers", type=int, default=4, help="refit processes (default 4)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default $VLAGEOM_DATA/contact/detection_heldout.json)")
    a = ap.parse_args(argv)
    import scipy.linalg  # noqa: F401  (load scipy's BLAS so that the limit below covers it)
    from threadpoolctl import threadpool_limits

    t0 = time.time()
    cache = a.cache or paths.tile_dir(C.TILE) / "natural_cache.npz"
    pool_path = a.pool or paths.tile_dir(C.TILE) / f"offschedule_L{C.PROBE_LAYER}.npz"
    probe_path = a.probe or paths.tile_dir(C.TILE) / "natural_probe.json"
    out = a.out or paths.tile_dir(C.TILE) / "detection_heldout.json"

    probe = json.loads(Path(probe_path).read_text())
    natural_auc = {p: probe["cells"][f"{p}@{C.PROBE_LAYER}"]["auc"] for p in C.POOLS}
    Xnat = {p: C.load_layer(cache, f"X_{p}", C.PROBE_LAYER) for p in C.POOLS}
    with np.load(cache, mmap_mode="r") as z:
        ynat = np.asarray(z["contact"]).astype(int)
        nat_house = np.asarray(z["house"]).astype(np.int64)
    print(f"[60] natural: {len(ynat)} rows, {len(np.unique(nat_house))} houses "
          f"({time.time() - t0:.0f}s)", flush=True)
    z = np.load(pool_path)
    if int(z["layer"]) != C.PROBE_LAYER:
        raise SystemExit(f"{pool_path} is layer {int(z['layer'])}, expected {C.PROBE_LAYER}")
    D = {k: z[k] for k in z.files if k not in ("layer", "subdir")}
    subdir = str(z["subdir"])
    print(f"[60] off-schedule: {len(D['contact'])} rows, {len(D['shards'])} episodes, "
          f"{len(np.unique(D['house']))} houses", flush=True)

    def progress(i, n):
        if i % 20 == 0 or i == n:
            print(f"[60]  {i}/{n} houses refit ({time.time() - t0:.0f}s)", flush=True)

    with threadpool_limits(limits=1):
        S = C.heldout_scores(Xnat, ynat, nat_house, {p: D[f"X_{p}"] for p in C.POOLS},
                             D["house"], workers=a.workers, progress=progress)
        table = C.detection_table(D, S, natural_auc)
    for p in C.POOLS:
        r = table[p]
        print(f"[60] {p:8s} " + " ".join(
            f"{c}={r[c]['auc']:.3f}/{r[c]['frac_pos']:.3f}(n{r[c]['n_pose']})" for c in C.CELLS),
            flush=True)
    counts = {c: int(C.cell_mask(D, c).sum()) for c in C.CELLS}
    counts.update(early_touch_open=int((D["event"] == 0).sum()), slip_closed=int((D["event"] == 1).sum()))
    res = {"n_pts": int(len(D["contact"])), "n_eps": int(len(D["shards"])), "counts": counts,
           "phase_timer_within_pose_auc": 0.5, "cache": str(cache), "subdir": subdir,
           "probe": C.HELDOUT_NOTE, "variants": table}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[DONE] {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
