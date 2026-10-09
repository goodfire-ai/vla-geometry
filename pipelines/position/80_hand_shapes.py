#!/usr/bin/env python
"""Hand paths of the EE trace grid (GPU + EGL): Figure 2, middle row.

In one house (default 44, the house left out of the grid's subspace fit; object at its natural
placement, seed 104729 + house), the hand is moved along six paths in the robot base frame
(`vlageom.position.hand_shape_design`): a figure-8 (64 waypoints, two tangent circles in a plane
tilted from far-low to near-high: forward 24-56 cm, 12-38 cm above the object's resting height),
an x line (forward 20-60 cm, 24 waypoints), a y line (sideways -30..30 cm), a z line (up 5-40 cm;
harvested, not drawn), a circle (radius 15 cm in the figure-8's plane, 64 waypoints) and a square
(forward 30-56 cm x sideways +-14 cm, low side 12 cm and high side 34 cm above the resting height,
48 waypoints). A closed path whose low region lies over the object is raised by 6 cm as a whole
(in house 44 the circle and the square).

Each path is walked by damped-least-squares inverse kinematics, every waypoint starting from the
previous one's joints; closed paths first walk two laps without recording, so that the joints
follow one periodic branch. IK iterates until the EE is within 2 mm of the target (at most 150
iterations); a waypoint is kept when the EE is within 1.5 cm of it and no robot body penetrates
anything. At each kept waypoint: restore the settled state, set and hold the joints, move the
robot-mounted cameras to the hand, render, reset the policy, run one model query and encode it
(VLM 11 token groups x 36 layers, action-expert token mean x 36 blocks). Waypoints with a dark exo
frame (mean pixel value < 15) are skipped.

A path is not written when fewer than half its waypoints are kept, or when the exo frames do not
change from waypoint to waypoint (8 consecutive frames differing in fewer than 0.1% of their
pixels, all frames identical, mean-pixel spread < 1e-6 or median pixel value < 20), or when one
waypoint takes longer than 60 s; the job can then be run again (on another node).

Output: $VLAGEOM_DATA/position/shapes/ee_{kind}/h{house}/: acts.npz (uncompressed; X_{variant}
(n, 36, 2560), ae (n, 36, 768), ee, ee_target, obj_base, dist, qpos, ik_err, exo_px), manifest.json
(design, closed, home pose, audit, and per waypoint target, realized EE and object positions,
hand-object distance, IK error, joints, exo PNG name) and exo_{i:02d}.png. About 15 min on one
H200 for the six paths.
"""
from __future__ import annotations

import vlageom.sim  # noqa: F401  (first: sets up the file locks before molmo_spaces is imported)

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.position import (HAND_SHAPE_KINDS, N_LAYERS, OPTIONAL_VARIANTS, SHAPE_HOUSE,
                              VARIANTS_LLM, arm_qpos, crossed_house_list_path, encode_query_all,
                              hand_shape_design, ik_base, place_arm, read_house_list,
                              robot_root_ids, run_queries, save_npz, shapes_root)
from vlageom.sim import (as_hw3, base_transform, bench_engine, build_house_list, exo_frame,
                         fresh_obs, get_pickup_body_name, mj_handles, obj_root_id,
                         sample_bench_episode, settle, to_base)

log = logging.getLogger("hand_shapes")


def harvest_hand_shape(eng, spec, bench, args, out_root, kind) -> bool:
    import imageio.v2 as imageio

    house = int(spec.house_index)
    out = out_root / f"h{house}"
    pickup = get_pickup_body_name(eng.task)
    if not pickup:
        log.error("h%d: no pickup_obj_name", house)
        return False
    model, data = mj_handles(eng)
    obs0, _ = eng.task.reset()
    st_nat = settle(eng, args.settle_max, args.settle_window, args.settle_tol)
    q_home = arm_qpos(eng)
    ee_home = np.asarray(eng.read_ee(), np.float64)
    snap0 = eng._snapshot(obs0)
    R, t = base_transform(eng)
    p_nat = np.asarray(eng.read_object(), np.float64)
    ob_nat = to_base(R, t, p_nat)
    z_s = float(ob_nat[2])
    root = obj_root_id(eng, pickup)
    roots = robot_root_ids(eng)
    T, closed, design = hand_shape_design(kind, z_s, ob_nat, args)
    log.info("h%d %s: %d targets, closed=%s, lift %.2f", house, kind, len(T), closed,
             float(design.get("lift", 0.0)))

    q_seed, _, err0 = ik_base(eng, q_home, T[0], n_iter=args.ik_iters, tol=args.ik_stop)
    for _lap in range(args.warmup_laps if closed else 0):
        for tgt in T:
            q_seed, _, _ = ik_base(eng, q_seed, tgt, n_iter=args.ik_iters, tol=args.ik_stop)

    out.mkdir(parents=True, exist_ok=True)
    audit = {"waypoints": 0, "kept": 0, "ik_reject": 0, "contact_reject_scene": 0,
             "contact_reject_obj": 0, "no_query": 0, "dark_frames": 0, "encode_fail": 0,
             "near_obj_lt_min": 0, "natural_settle_disp": st_nat["last_disp"],
             "warmup_laps": int(args.warmup_laps if closed else 0)}
    X = {v: [] for v in VARIANTS_LLM}
    AE, pts, THUMBS, fracs = [], [], [], []
    prev_img = None
    t0 = time.time()
    for i, tgt in enumerate(T):
        audit["waypoints"] += 1
        t_wp = time.time()
        eng._restore(snap0)
        data.qacc_warmstart[:] = 0.0
        eng.policy.reset()
        eng._chunk = 0
        q, _ee, err = ik_base(eng, q_seed, tgt, n_iter=args.ik_iters, tol=args.ik_stop)
        if not np.isfinite(err) or err > args.ik_tol:
            audit["ik_reject"] += 1
            continue
        q_seed = q
        ns, no = place_arm(eng, q, model, data, roots, root)
        if ns or no:
            audit["contact_reject_scene" if ns else "contact_reject_obj"] += 1
            continue
        obs = fresh_obs(eng)
        img = as_hw3(exo_frame(obs))
        caps, states = run_queries(eng, obs, 1, max_loop=2)
        if len(caps) < 1 or len(states) < 1:
            audit["no_query"] += 1
            continue
        stt = states[0]
        if np.isfinite(stt["exo_px"]) and stt["exo_px"] < args.dark_px:
            audit["dark_frames"] += 1
            continue
        enc = encode_query_all(eng, caps[0]["kw"], args.llm_layers, args.ae_layers, eng.scene_seed)
        if enc is None or any(enc[0][v] is None for v in VARIANTS_LLM
                              if v not in OPTIONAL_VARIANTS):
            audit["encode_fail"] += 1
            continue
        llm, ae_tok = enc
        for v in VARIANTS_LLM:
            X[v].append(llm[v].astype(np.float16) if llm[v] is not None
                        else np.full((len(args.llm_layers), 2560), np.nan, np.float16))
        AE.append(ae_tok.mean(1).astype(np.float16))
        THUMBS.append(np.asarray(stt["thumb"]))
        if img is not None and prev_img is not None and img.shape == prev_img.shape:
            fracs.append(float((img != prev_img).mean()))
        prev_img = img
        dt_wp = time.time() - t_wp
        if dt_wp > args.wp_time_cap:
            log.error("h%d %s: waypoint %d took %.0f s (> %.0f s); path not written", house, kind,
                      i, dt_wp, args.wp_time_cap)
            return False
        if len(fracs) >= 8 and max(fracs[-8:]) < args.static_frac:
            log.error("h%d %s: the exo frame did not change over 8 waypoints; path not written",
                      house, kind)
            return False
        png = None
        if img is not None:
            png = f"exo_{i:02d}.png"
            imageio.imwrite(out / png, img)
        ee_b = np.asarray(stt["ee_b"], np.float64)
        ob_b = to_base(R, t, stt["obj_w"])
        d = float(np.linalg.norm(ob_b - ee_b))
        if d < args.min_obj_dist:
            audit["near_obj_lt_min"] += 1
        pts.append({"i": int(i), "target_base": [float(v) for v in tgt],
                    "ee_base": [float(v) for v in ee_b], "obj_base": [float(v) for v in ob_b],
                    "dist": d, "ik_err": float(err), "qpos_arm": [float(v) for v in q],
                    "exo_px": float(stt["exo_px"]), "exo_png": png})
        audit["kept"] += 1
    if len(pts) < 0.5 * len(T):
        log.error("h%d %s: only %d/%d waypoints kept; path not written (audit=%s)", house, kind,
                  len(pts), len(T), audit)
        return False
    px = np.array([p["exo_px"] for p in pts], float)
    audit["exo_px_mean"] = float(np.nanmean(px))
    audit["exo_px_std"] = float(np.nanstd(px))
    audit["exo_px_median"] = float(np.nanmedian(px))
    thumbs_same = bool(len(THUMBS) > 3 and all(np.array_equal(THUMBS[0], th) for th in THUMBS[1:]))
    audit["exo_changed_frac_min"] = float(min(fracs)) if fracs else float("nan")
    audit["exo_changed_frac_median"] = float(np.median(fracs)) if fracs else float("nan")
    audit["exo_static"] = bool(len(px) > 3 and (
        np.nanstd(px) < 1e-6 or thumbs_same or np.nanmedian(px) < 20
        or (len(fracs) >= 3 and max(fracs) < args.static_frac)))
    if audit["exo_static"]:
        log.error("h%d %s: the exo frame does not change across the path; path not written",
                  house, kind)
        return False
    E = np.array([p["ee_base"] for p in pts])
    Tk = np.array([p["target_base"] for p in pts])
    audit["realized_vs_target_cm_max"] = float(np.linalg.norm(E - Tk, axis=1).max() * 100)
    payload = dict(ae=np.stack(AE), layers_llm=np.array(args.llm_layers, np.int32),
                   layers_ae=np.array(args.ae_layers, np.int32), variants=np.array(VARIANTS_LLM),
                   ee=E.astype(np.float32), ee_target=Tk.astype(np.float32),
                   obj_base=np.array([p["obj_base"] for p in pts], np.float32),
                   dist=np.array([p["dist"] for p in pts], np.float32),
                   qpos=np.array([p["qpos_arm"] for p in pts], np.float32),
                   ik_err=np.array([p["ik_err"] for p in pts], np.float32),
                   exo_px=px.astype(np.float32))
    for v in VARIANTS_LLM:
        payload[f"X_{v}"] = np.stack(X[v])
    save_npz(out / "acts.npz", payload)
    (out / "manifest.json.tmp").write_text(json.dumps(
        {"house": house, "bench": str(bench), "variable": "ee", "shape": kind,
         "closed": bool(closed), "n_requested": int(len(T)), "design": design,
         "corners": (design.get("corners") or design.get("corners_ABCD")
                     or [T[0].tolist(), T[-1].tolist()]),
         "z_surf_base": z_s, "natural_obj_base": [float(v) for v in ob_nat],
         "q_home": [float(v) for v in q_home], "ee_home": [float(v) for v in ee_home],
         "base_pose": np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]).tolist(),
         "audit": audit, "points": pts}, indent=1))
    os.replace(out / "manifest.json.tmp", out / "manifest.json")
    log.info("h%d %s -> %s (%d/%d waypoints, %.1f min; realized vs target <= %.2f cm)", house, kind,
             out, len(pts), len(T), (time.time() - t0) / 60.0, audit["realized_vs_target_cm_max"])
    return True


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--out", type=Path, default=None,
                    help="shapes root (default $VLAGEOM_DATA/position/shapes); "
                         "ee_{kind}/h{house}/ below it")
    ap.add_argument("--house", type=int, default=SHAPE_HOUSE)
    ap.add_argument("--house-list", type=Path, default=crossed_house_list_path())
    ap.add_argument("--shapes", nargs="+", choices=HAND_SHAPE_KINDS, default=list(HAND_SHAPE_KINDS))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--bench-mini", default=None)
    ap.add_argument("--bench-hard", default=None)
    ap.add_argument("--n-lines", type=int, default=24)
    ap.add_argument("--n-closed", type=int, default=64, help="figure-8 and circle waypoints")
    ap.add_argument("--n-square", type=int, default=48, help="square waypoints (12 per side)")
    ap.add_argument("--warmup-laps", type=int, default=2)
    ap.add_argument("--pose-fwd-lo", type=float, default=0.20)
    ap.add_argument("--pose-fwd-hi", type=float, default=0.60)
    ap.add_argument("--pose-lat", type=float, default=0.30)
    ap.add_argument("--pose-up-lo", type=float, default=0.05)
    ap.add_argument("--pose-up-hi", type=float, default=0.40)
    ap.add_argument("--fig8-fwd", type=float, nargs=2, default=[0.24, 0.56],
                    help="forward position of the figure-8's high (near) and low (far) ends")
    ap.add_argument("--fig8-up", type=float, nargs=2, default=[0.12, 0.38],
                    help="height above the resting height of the low and high ends")
    ap.add_argument("--circle-r", type=float, default=0.15)
    ap.add_argument("--sq-fwd", type=float, nargs=2, default=[0.30, 0.56])
    ap.add_argument("--sq-lat", type=float, default=0.14)
    ap.add_argument("--sq-up", type=float, nargs=2, default=[0.12, 0.34])
    ap.add_argument("--closed-lift", type=float, default=0.06)
    ap.add_argument("--ik-iters", type=int, default=150)
    ap.add_argument("--ik-tol", type=float, default=0.015)
    ap.add_argument("--ik-stop", type=float, default=0.002)
    ap.add_argument("--min-obj-dist", type=float, default=0.12,
                    help="waypoints closer to the object are counted in the audit (and kept)")
    ap.add_argument("--dark-px", type=float, default=15.0)
    ap.add_argument("--wp-time-cap", type=float, default=60.0)
    ap.add_argument("--static-frac", type=float, default=1e-3)
    ap.add_argument("--settle-max", type=int, default=500)
    ap.add_argument("--settle-window", type=int, default=50)
    ap.add_argument("--settle-tol", type=float, default=1e-3)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()
    args.llm_layers = list(range(N_LAYERS))
    args.ae_layers = list(range(N_LAYERS))

    listed = {h: (b, i) for h, b, i in read_house_list(args.house_list)}
    if args.house not in listed:
        raise SystemExit(f"house {args.house} is not in {args.house_list}")
    roots = paths.bench_roots()
    roots = {"mini": args.bench_mini or roots["mini"], "hard": args.bench_hard or roots["hard"]}
    catalog = {h: (b, i, sp) for h, b, i, sp in build_house_list(roots, 950)}
    bench, bench_idx, spec = catalog[args.house]
    if (bench, bench_idx) != listed[args.house]:
        raise SystemExit(f"h{args.house}: benchmark episode {bench}#{bench_idx} != list "
                         f"{listed[args.house]}")
    out_base = args.out or shapes_root()
    eng, sim_envs = bench_engine(args.checkpoint, str(out_base / "_env"), need_mini=bench == "mini")
    n_ok, n_tried = 0, 0
    try:
        for kind in args.shapes:
            out_root = out_base / f"ee_{kind}"
            out_root.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and (out_root / f"h{args.house}" / "manifest.json").exists():
                log.info("h%d ee %s exists, skipped", args.house, kind)
                continue
            n_tried += 1
            try:
                sample_bench_episode(eng, sim_envs, bench, spec, seed=104729 + args.house,
                                     task_horizon=16)
                n_ok += int(harvest_hand_shape(eng, spec, bench, args, out_root, kind))
            except Exception:
                log.exception("h%d ee %s failed; continuing", args.house, kind)
            finally:
                eng.task = eng.policy = None
                gc.collect()
    finally:
        eng.close()
    log.info("done: %d/%d paths written", n_ok, n_tried)
    if n_tried and n_ok < n_tried:
        sys.exit(3)


if __name__ == "__main__":
    main()
