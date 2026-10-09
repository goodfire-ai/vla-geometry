"""Phase, stage 70 (CPU): the arrays the five Figure 5 data panels are drawn from.

Inputs: the natural table (stage 20), the decoder scan (stage 30, numbers only), the freeze-test
files (stage 40), the decoded landmarks (stage 50) and the trajectory capture (stage 60). Every
panel reads action-expert layer 24.

  strip  the trajectory capture: query labels, the two filmstrip columns per phase
         (vlageom.phase.pick_columns) with their exo and wrist frames, and the per-query colour of
         the four one-vs-rest phase probes (vlageom.phase.fit_probes, fit on the successful natural
         queries of all houses but the capture's; vlageom.phase.latent_colours)
  a      the successful natural queries in the first two LDA directions (StandardScaler + LDA fit
         on all of them) with their phase labels, and the out-of-house balanced accuracy over
         lists/folds/phase.csv (printed)
  b      the first three principal components of the successful natural queries of all houses but
         the capture's: the capture's queries in them (coloured as in the strip), 6,000 of those
         queries (numpy RandomState(0) choice) with their labels, and the explained-variance ratios
  c      per phase: % of the decoded landmarks of stage 50 (797 on the paper's stores) decoded as
         their own phase, natural and frozen, and the two most frequent decoded phases when frozen
         (grasp pools the open- and closed-hand grasp landmarks)
  d      per phase: per landmark, the fraction of the 8 executed steps of the commanded chunk with
         a closed gripper, frozen minus natural; mean and 95% CI. Landmarks of all 126 freeze-test
         files, selected as in stage 50 (vlageom.phase.used_landmarks: of an episode's closed-hand
         place landmarks only the last; 819 landmarks on the paper's stores)

Output: <data>/panels/fig5_{strip,a,b,c,d}.npz and <data>/phase/fig5_numbers.json (the numbers of
the figure, its caption and the text). Runs at 2 BLAS threads (--threads); 1-2 minutes.

  python pipelines/phase/70_panels.py [--panel strip a b c d]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom import phase as P
from vlageom.probes import FoldMap

PANELS = ("strip", "a", "b", "c", "d")
N_CLOUD = 6000


def panel_strip_b(cache, cap_path: Path, numbers: dict, want):
    cap = np.load(cap_path, allow_pickle=True)
    house = int(cap["house"])
    lab_q = cap["label_q"].astype(int)
    li = [int(v) for v in cap["ae_layers"]].index(P.LAYER)
    ae_q = cap["ae"][:, li, :]
    projn, meta = P.fit_probes(cache, P.LAYER, house, ae_q)
    rgb, W = P.latent_colours(projn)
    dom = np.argmax(W, 1)
    cols = P.pick_columns(lab_q)
    flat = [q for pair in cols for q in pair]
    numbers["strip"] = {
        "episode": int(cap["ep"]), "house": house, "seed": int(cap["seed"]),
        "task": str(cap["task_description"]), "n_queries": int(cap["n_queries"]),
        "success": bool(cap["success"]),
        "query_counts": {P.NAMES[k]: int((lab_q == k).sum()) for k in range(P.N_PHASE)},
        "columns": flat, "probe": meta,
        "dominant_probe_equals_label": int((dom == lab_q).sum())}
    print(f"strip: ep{int(cap['ep'])} h{house}, {int(cap['n_queries'])} queries "
          f"{numbers['strip']['query_counts']}; probes fit on {meta['n_train_rows']} rows of "
          f"{meta['n_train_houses']} houses (house {house} held out, "
          f"{meta['n_rows_held_out']} rows); columns {flat}; strongest probe = label on "
          f"{int((dom == lab_q).sum())}/{len(lab_q)} queries")
    out = {}
    if "strip" in want:
        out["strip"] = dict(lab_q=lab_q.astype(np.int8), rgb=rgb, projn=projn, cols=np.array(flat),
                            frames_exo=cap["frames_exo"][flat],
                            frames_wrist=cap["frames_wrist"][flat])
    if "b" in want:
        layers = [int(v) for v in cache["layers"]]
        X = np.ascontiguousarray(cache["X"][:, layers.index(P.LAYER), :])
        m = cache["success"].astype(bool) & (cache["house"].astype(int) != house)
        scores, evr, comps, mu = P.pca_embed(X[m], k=3)
        traj = (np.asarray(ae_q, float) - mu) @ comps.T
        lab = cache["label"].astype(int)[m]
        idx = np.random.RandomState(0).choice(len(scores), N_CLOUD, replace=False)
        evr = np.asarray(evr[:3], float)
        out["b"] = dict(traj=traj, rgb=rgb, bg=scores[idx], bg_lab=lab[idx], evr=evr)
        numbers["b"] = {"evr": evr.tolist(), "n_cloud": N_CLOUD, "n_basis_rows": int(m.sum())}
        pcts = ", ".join(f"{100 * v:.0f}%" for v in evr)
        print(f"b: PC1-3 explain {100 * evr.sum():.0f}% ({pcts}) "
              f"of {int(m.sum())} rows; {N_CLOUD} cloud points")
    return out


def panel_a(cache, folds, numbers):
    layers = [int(v) for v in cache["layers"]]
    m = cache["success"].astype(bool)
    Xl = np.ascontiguousarray(cache["X"][m][:, layers.index(P.LAYER), :])
    lab = cache["label"][m].astype(int)
    house = cache["house"][m].astype(int)
    r = P.lda_cv(Xl, lab, house, folds)
    emb = P.lda_embedding(Xl, lab)
    numbers["a"] = {"balanced_accuracy": r["bal"], "majority_rate": r["maj"], "n": r["n"],
                    "n_houses": int(len(np.unique(house)))}
    print(f"a: layer {P.LAYER} balanced accuracy {r['bal']:.5f} (majority rate {r['maj']:.2f}), "
          f"n {r['n']}, {len(np.unique(house))} houses")
    return dict(emb=emb[:, :2], y=lab)


def panel_c(decode_json: Path, numbers):
    dec = json.loads(decode_json.read_text())
    D = P.own_phase_data(dec["composition"])
    names = P.NAMES
    out = dict(phases=np.array(names),
               own_natural=np.array([D["own"][P.COND_NATURAL][k] for k in names]),
               own_frozen=np.array([D["own"][P.COND_FROZEN][k] for k in names]),
               top=np.array([D["dest"][k]["top"] for k in names]),
               top_pct=np.array([D["dest"][k]["top_pct"] for k in names]),
               second=np.array([D["dest"][k]["second"] for k in names]),
               second_pct=np.array([D["dest"][k]["second_pct"] for k in names]),
               n=np.array([D["n"][k] for k in names]),
               n_landmarks=np.int64(dec["n_rows"]), n_episodes=np.int64(dec["n_files"]))
    numbers["c"] = {"n_landmarks": dec["n_rows"], "n_episodes": dec["n_files"],
                    "decoder": dec["decoder"], "n_per_phase": D["n"],
                    "decoded_correctly_pct": {"natural": D["own"][P.COND_NATURAL],
                                              "frozen": D["own"][P.COND_FROZEN]},
                    "frozen_most_read": D["dest"]}
    print(f"c: {dec['n_rows']} landmarks of {dec['n_files']} episodes; decoded correctly "
          "natural -> frozen: " + ", ".join(
              f"{k} {D['own'][P.COND_NATURAL][k]:.2f} -> {D['own'][P.COND_FROZEN][k]:.2f}"
              for k in names))
    for k in names:
        d = D["dest"][k]
        print(f"   frozen {k}: {d['top']} {d['top_pct']:.2f}%, "
              f"{d['second']} {d['second_pct']:.2f}%")
    return out


def panel_d(freeze_dir: Path, numbers):
    files = sorted(freeze_dir.glob("ep*_h*.npz"), key=str)
    listed = {P.freeze_name(r["episode_index"], r["house"]) for r in P.read_freeze_episodes()}
    files = [f for f in files if f.name in listed]
    chunks = P.load_gripper_chunks(files)
    G = P.gripper_delta_data(chunks)
    names = [n for n, _c in P.PANEL_KINDS]
    out = {f"d_{k}": G[k]["d"] for k in names}
    out.update(phases=np.array(names), mean=np.array([G[k]["mean"] for k in names]),
               ci=np.array([G[k]["ci"] for k in names]),
               n=np.array([len(G[k]["d"]) for k in names]),
               closed_natural=np.stack([G[k]["closed_natural"] for k in names]),
               closed_frozen=np.stack([G[k]["closed_frozen"] for k in names]),
               n_episodes=np.int64(len(files)))
    numbers["d"] = {"n_episodes": len(files), "n_landmarks": int(sum(len(G[k]["d"]) for k in names)),
                    "executed_steps": P.EXECUTED,
                    "per_phase": {k: {"n": int(len(G[k]["d"])),
                                      "delta_closed_mean": float(G[k]["mean"]),
                                      "delta_closed_ci95": float(G[k]["ci"]),
                                      "closed_natural": G[k]["closed_natural"].tolist(),
                                      "closed_frozen": G[k]["closed_frozen"].tolist()}
                                  for k in names}}
    print(f"d: {len(files)} episodes, {sum(len(G[k]['d']) for k in names)} landmarks; " + ", ".join(
        f"{k} {G[k]['mean']:+.3f} +- {G[k]['ci']:.3f} (n {len(G[k]['d'])})" for k in names))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", nargs="+", choices=PANELS, default=list(PANELS))
    ap.add_argument("--cache", type=Path,
                    default=paths.tile_dir("phase", "natural", "cache_ae.npz"))
    ap.add_argument("--trajectory", type=Path, default=None,
                    help="trajectory capture of stage 60 (default <data>/phase/trajectory/"
                         "traj_ep<E>_h<H>.npz of lists/phase/trajectory_episode.csv)")
    ap.add_argument("--decode", type=Path,
                    default=paths.tile_dir("phase", "freeze", "decode_ae.json"))
    ap.add_argument("--freeze-dir", type=Path,
                    default=paths.tile_dir("phase", "freeze", "episodes"))
    ap.add_argument("--decoder-scan", type=Path, default=paths.tile_dir("phase", "decoder.json"))
    ap.add_argument("--folds", default="phase")
    ap.add_argument("--threads", type=int, default=2, help="BLAS threads (default 2)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <data>/panels")
    ap.add_argument("--numbers", type=Path, default=paths.tile_dir("phase", "fig5_numbers.json"))
    a = ap.parse_args(argv)
    want = set(a.panel)
    out_dir = a.out_dir or paths.panels_dir(create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.trajectory is None:
        t = P.read_trajectory_episode()
        a.trajectory = paths.tile_dir("phase", "trajectory",
                                      f"traj_ep{t['episode_index']}_h{t['house']}.npz")

    numbers = json.loads(a.numbers.read_text()) if a.numbers.exists() else {}
    arrays = {}
    import scipy.linalg  # noqa: F401  (load scipy's BLAS so that the limit below covers it)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=a.threads):
        if want & {"strip", "a", "b"}:
            cache = P.load_cache(a.cache, ["X", "label", "house", "success", "layers", "ep"])
            succ_eps = np.unique(cache["ep"][cache["success"]])
            numbers["natural"] = {"episodes": int(len(np.unique(cache["ep"]))),
                                  "houses": int(len(np.unique(cache["house"]))),
                                  "success_episodes": int(len(succ_eps)),
                                  "rows": int(len(cache["label"])),
                                  "success_rows": int(cache["success"].sum())}
            if want & {"strip", "b"}:
                arrays.update(panel_strip_b(cache, a.trajectory, numbers, want))
            if "a" in want:
                arrays["a"] = panel_a(cache, FoldMap.load(a.folds), numbers)
        if "c" in want:
            arrays["c"] = panel_c(a.decode, numbers)
        if "d" in want:
            arrays["d"] = panel_d(a.freeze_dir, numbers)
    if a.decoder_scan.exists():
        scan = json.loads(a.decoder_scan.read_text())
        numbers["decoder_scan"] = {"best_layer": scan["best_layer"],
                                   "balanced_accuracy": {L: v["bal"]
                                                         for L, v in scan["per_layer"].items()}}
    for name, arr in arrays.items():
        np.savez_compressed(out_dir / f"fig5_{name}.npz", **arr)
        print(f"wrote {out_dir / f'fig5_{name}.npz'}")
    a.numbers.parent.mkdir(parents=True, exist_ok=True)
    a.numbers.write_text(json.dumps(numbers, indent=1))
    print(f"wrote {a.numbers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
