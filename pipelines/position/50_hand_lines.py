#!/usr/bin/env python
"""Straight hand lines (GPU + EGL): the EE paths of the curvature panel (Figure 2e).

Per house of lists/position/line_houses.csv (48 houses; the object at its natural placement,
seed 104729 + house as in the crossed harvest), 4 straight lines of 24 waypoints in the robot base
frame, through the centre of the crossed harvest's pose box (forward 40 cm, lateral 0, 22.5 cm
above the object's resting height):
  line 0  forward, 20-60 cm
  line 1  sideways, -30..30 cm
  line 2  up, 5-40 cm above the resting height
  line 3  from the box centre toward the object, ending 12 cm short of it (only when the object is
          more than 17 cm from the centre)
Each line is walked by damped-least-squares inverse kinematics, every waypoint starting from the
joints of the previous one (so the arm stays on one joint-space branch), iterating until the EE is
within 2 mm of the target (at most 150 iterations); a waypoint is kept when the EE is within
1.5 cm of it and no robot body penetrates anything. At each kept waypoint: restore the settled
state, set the joints and hold them, move the robot-mounted cameras to the hand, render, reset the
policy, run one model query and encode it: VLM 11 token groups x 36 layers (float16) and the
action expert's token mean at 36 blocks (float16).

Output: $VLAGEOM_DATA/position/hand_lines/h{house}.npz (uncompressed): path_id, step_k, ee
(realized), ee_target, obj_base, dist (|object - EE|), qpos, ik_err, exo_px, exo_thumb, ae
(n, 36, 768), X_{variant} (n, 36, 2560), house-level paths, q_home, z_surf_base,
natural_obj_base, base_pose, audit_json. About 10 min per house on one H200, 1.4 GPU-hours in all
(8 jobs of 6 houses).

Work split: --shard K --houses-per-shard 6 (K defaults to $SLURM_ARRAY_TASK_ID); --houses H ...
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
from vlageom.position import (N_LAYERS, OPTIONAL_VARIANTS, VARIANTS_LLM, arm_qpos,
                              encode_query_all, hand_line_targets, hand_lines_root, ik_base,
                              line_house_list_path, place_arm, read_house_list, robot_root_ids,
                              run_queries, save_npz)
from vlageom.sim import (base_transform, bench_engine, build_house_list, fresh_obs,
                         get_pickup_body_name, mj_handles, obj_root_id, sample_bench_episode,
                         settle, to_base)

log = logging.getLogger("hand_lines")
PATHS = ["fwd", "lat", "up", "obj"]


def harvest_house(eng, spec, bench, args):
    house = int(spec.house_index)
    pickup = get_pickup_body_name(eng.task)
    if not pickup:
        return None
    model, data = mj_handles(eng)
    obs0, _ = eng.task.reset()
    settle(eng, args.settle_max, args.settle_window, args.settle_tol)
    q_home = arm_qpos(eng)
    snap0 = eng._snapshot(obs0)
    R, t = base_transform(eng)
    p_nat = np.asarray(eng.read_object(), np.float64)
    ob = to_base(R, t, p_nat)
    z_surf_b = float(ob[2])
    root = obj_root_id(eng, pickup)
    roots = robot_root_ids(eng)
    lines = hand_line_targets(args, z_surf_b, ob)

    rows = {k: [] for k in ["path_id", "step_k", "ee", "ee_target", "obj_base", "dist", "qpos",
                            "ik_err", "exo_px"]}
    X = {v: [] for v in VARIANTS_LLM}
    AE, THUMBS = [], []
    audit = {"waypoints": 0, "kept": 0, "ik_reject": 0, "contact_reject": 0, "encode_fail": 0,
             "no_query": 0}
    t0 = time.time()
    for pid, tgts in lines.items():
        q_seed = q_home.astype(np.float32)
        for k, tgt in enumerate(tgts):
            audit["waypoints"] += 1
            eng._restore(snap0)
            eng.policy.reset()
            eng._chunk = 0
            q, _ee, err = ik_base(eng, q_seed, tgt, n_iter=args.ik_iters, tol=args.ik_stop)
            if not np.isfinite(err) or err > args.ik_tol:
                audit["ik_reject"] += 1
                continue
            q_seed = q
            ns, no = place_arm(eng, q, model, data, roots, root)
            if ns or no:
                audit["contact_reject"] += 1
                continue
            obs = fresh_obs(eng)
            caps, states = run_queries(eng, obs, 1, max_loop=2)
            if len(caps) < 1 or len(states) < 1:
                audit["no_query"] += 1
                continue
            enc = encode_query_all(eng, caps[0]["kw"], args.llm_layers, args.ae_layers,
                                   eng.scene_seed)
            if enc is None or any(enc[0][v] is None for v in VARIANTS_LLM
                                  if v not in OPTIONAL_VARIANTS):
                audit["encode_fail"] += 1
                continue
            llm, ae_tok = enc
            for v in VARIANTS_LLM:
                X[v].append(llm[v].astype(np.float16) if llm[v] is not None
                            else np.full((len(args.llm_layers), 2560), np.nan, np.float16))
            AE.append(ae_tok.mean(1).astype(np.float16))
            THUMBS.append(states[0]["thumb"])
            ee_b = np.asarray(states[0]["ee_b"], np.float32)
            obw = to_base(R, t, states[0]["obj_w"]).astype(np.float32)
            rows["path_id"].append(pid)
            rows["step_k"].append(k)
            rows["ee"].append(ee_b)
            rows["ee_target"].append(np.asarray(tgt, np.float32))
            rows["obj_base"].append(obw)
            rows["dist"].append(float(np.linalg.norm(obw - ee_b)))
            rows["qpos"].append(np.asarray(q, np.float32))
            rows["ik_err"].append(float(err))
            rows["exo_px"].append(float(states[0]["exo_px"]))
            audit["kept"] += 1
    if not rows["dist"]:
        log.warning("h%d: nothing kept (%s)", house, audit)
        return None
    payload = dict(path_id=np.array(rows["path_id"], np.int32),
                   step_k=np.array(rows["step_k"], np.int32),
                   ee=np.stack(rows["ee"]), ee_target=np.stack(rows["ee_target"]),
                   obj_base=np.stack(rows["obj_base"]), dist=np.array(rows["dist"], np.float32),
                   qpos=np.stack(rows["qpos"]), ik_err=np.array(rows["ik_err"], np.float32),
                   exo_px=np.array(rows["exo_px"], np.float32), exo_thumb=np.stack(THUMBS),
                   ae=np.stack(AE), layers_llm=np.array(args.llm_layers, np.int32),
                   layers_ae=np.array(args.ae_layers, np.int32), variants=np.array(VARIANTS_LLM),
                   paths=np.array(PATHS), house=house, bench=str(bench),
                   q_home=q_home.astype(np.float32), z_surf_base=np.float32(z_surf_b),
                   natural_obj_base=ob.astype(np.float32),
                   base_pose=np.asarray(np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]),
                                        np.float32),
                   audit_json=np.frombuffer(json.dumps(audit).encode(), np.uint8))
    for v in VARIANTS_LLM:
        payload[f"X_{v}"] = np.stack(X[v])
    log.info("h%d: %d/%d waypoints kept over %d lines (%.1f min; %s)", house, audit["kept"],
             audit["waypoints"], len(lines), (time.time() - t0) / 60, audit)
    return payload


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default $VLAGEOM_DATA/position/hand_lines)")
    ap.add_argument("--house-list", type=Path, default=line_house_list_path())
    ap.add_argument("--houses", type=int, nargs="+", default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--houses-per-shard", type=int, default=6)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--bench-mini", default=None)
    ap.add_argument("--bench-hard", default=None)
    ap.add_argument("--n-way", type=int, default=24, help="waypoints per line")
    ap.add_argument("--pose-fwd-lo", type=float, default=0.20)
    ap.add_argument("--pose-fwd-hi", type=float, default=0.60)
    ap.add_argument("--pose-lat", type=float, default=0.30)
    ap.add_argument("--pose-up-lo", type=float, default=0.05)
    ap.add_argument("--pose-up-hi", type=float, default=0.40)
    ap.add_argument("--min-obj-dist", type=float, default=0.12)
    ap.add_argument("--ik-iters", type=int, default=150)
    ap.add_argument("--ik-stop", type=float, default=0.002, help="IK stops below this error (m)")
    ap.add_argument("--ik-tol", type=float, default=0.015,
                    help="waypoints with a larger error are skipped (m)")
    ap.add_argument("--settle-max", type=int, default=500)
    ap.add_argument("--settle-window", type=int, default=50)
    ap.add_argument("--settle-tol", type=float, default=1e-3)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.llm_layers = list(range(N_LAYERS))
    args.ae_layers = list(range(N_LAYERS))

    out = args.out or hand_lines_root()
    listed = read_house_list(args.house_list)
    if args.houses:
        bad = [h for h in args.houses if h not in {r[0] for r in listed}]
        if bad:
            raise SystemExit(f"houses not in {args.house_list}: {bad}")
        listed = [r for r in listed if r[0] in set(args.houses)]
    else:
        shard = args.shard
        if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID", "") != "":
            shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if shard is not None:
            listed = listed[shard * args.houses_per_shard:(shard + 1) * args.houses_per_shard]
    if not listed:
        log.info("nothing to do")
        return
    roots = paths.bench_roots()
    roots = {"mini": args.bench_mini or roots["mini"], "hard": args.bench_hard or roots["hard"]}
    catalog = {h: (b, i, sp) for h, b, i, sp in build_house_list(roots, 950)}
    mine = []
    for house, bench, bench_idx in listed:
        if house not in catalog or catalog[house][:2] != (bench, bench_idx):
            raise SystemExit(f"h{house}: benchmark episode {bench}#{bench_idx} not found")
        mine.append((house, bench, catalog[house][2]))
    log.info("%d houses: %s", len(mine), [h for h, *_ in mine])
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    eng, sim_envs = bench_engine(args.checkpoint, str(out / "_env"),
                                 need_mini=any(b == "mini" for _h, b, _s in mine))
    n_done, n_tried = 0, 0
    try:
        for house, bench, spec in mine:
            dst = out / f"h{house}.npz"
            if args.skip_existing and dst.exists():
                log.info("h%d exists, skipped", house)
                continue
            n_tried += 1
            try:
                sample_bench_episode(eng, sim_envs, bench, spec, seed=104729 + house,
                                     task_horizon=16)
                payload = harvest_house(eng, spec, bench, args)
            except Exception:
                log.exception("h%d failed; continuing with the next house", house)
                continue
            finally:
                eng.task = eng.policy = None
                gc.collect()
            if payload is None:
                continue
            save_npz(dst, payload)
            n_done += 1
            log.info("h%d -> %s (%d rows)", house, dst, len(payload["dist"]))
    finally:
        eng.close()
    log.info("done: %d/%d houses written", n_done, len(mine))
    if n_tried and n_done == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
