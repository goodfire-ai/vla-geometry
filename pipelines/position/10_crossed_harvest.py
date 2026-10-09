#!/usr/bin/env python
"""Crossed arm x object harvest (GPU + EGL): VLM and action-expert activations for 5 object
placements x 10 arm poses in each of the 947 benchmark houses (lists/position/crossed_houses.csv).

Per house (the house's MolmoSpaces benchmark episode, seed 104729 + house):
  1. reset the episode and settle the physics with the robot held in place; the arm's joint angles
     here are its home pose;
  2. object placements: the sampler of the object-teleport harvest (vlageom.sim.sample_targets,
     RNG seed 7919 * house + 37, 16 placements drawn) gives the same placements as that harvest;
     the first 5 are used (placement 0 = the natural position);
  3. arm poses: 10 EE targets drawn uniformly (RNG seed 104729 * house + 52) in the box 20-60 cm
     forward, up to 30 cm to either side, 5-40 cm above the object's resting height (robot base
     frame), at most 62 cm from the base axis horizontally and at least 5 cm apart; each reached by
     damped-least-squares inverse kinematics from the home pose (FR3 joint limits kept) and kept
     only when the EE gets within 1.5 cm of the target and no robot body penetrates the scene or the
     object at its natural placement;
  4. for every placement: restore the settled state, move the object there, settle, and keep the
     placement when the object stays within 2 cm (xy) and 1 cm (z) of the command on the same
     support geom (as the object-teleport harvest); then for every arm pose: restore that
     settled placement, skip the cell when the posed EE is closer than 12 cm to the object, set the
     arm's joints, re-anchor the joint controllers there (the robot holds the pose), skip the
     cell if a robot body penetrates anything, move the robot-mounted cameras to the new hand pose
     and render the observation, reset the policy, and run two model queries: q0 right after the
     move and q1 after the policy has executed one chunk (8 steps). Because the policy is reset, the
     model sees only the current frame pair at q0 (2 image blocks); at q1 it sees the frames of q1
     and of q0, both taken after the move;
  5. per query: VLM activations pooled over 11 token groups at all 36 layers (float16), the
     action expert's per-token output at all 36 blocks on the final flow step (float16; flow
     noise seeded with seed + 1000 q), the predicted chunk, the object and EE positions the model
     saw, the arm joints, and the exo frame's mean pixel value and a 64 px thumbnail. Cells whose
     exo frame is dark (mean pixel value < 15) are skipped.

A house in which at least half of 10 or more tried cells have a dark exo frame ends the process
with status 3 without writing that house (the renderer of this process is not producing images;
run the job again, houses already written are skipped).

Output: $VLAGEOM_DATA/position/crossed/houses/h{house}.npz, one uncompressed file per house
(<= 100 rows, about 300 MB): X_{variant} (n, 36, 2560), ae_tok (n, 36, 16, 768), ae (n, 36, 768),
chunk (n, 16, 8), exo_thumb (n, 64, 64, 3), per-row labels (config_id, pose_id, chunk_q, obj_base,
ee, rel_ee, dist, u, obj_world, cmd_xy, ee_target, ee_tele, qpos_set, qpos_q, ik_err, step,
settle_disp, lean_contacts, fwd, bearing_deg, ee44, in_env, exo_px), per-house pose arrays
(pose_targets, pose_q, pose_ee, pose_ik_err, q_home, ee_home, z_surf_base) and audit_json.

Work split: --shard K --houses-per-shard N takes houses [K*N, (K+1)*N) of the list (K defaults to
$SLURM_ARRAY_TASK_ID; the paper: 158 shards of 6); --houses H [H ...] runs only those houses.
About 2.5 min per house on one H200, about 40 GPU-hours in all.

Example (one GPU):
  source scripts/harvest_env.sh && python pipelines/position/10_crossed_harvest.py --houses 44
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
                              crossed_house_list_path, crossed_root, encode_query_all, place_arm,
                              read_house_list, robot_root_ids, run_queries, sample_poses, save_npz)
from vlageom.sim import (bad_contact_count, base_transform, bench_engine, build_house_list,
                         envelope_ok, fresh_obs, get_pickup_body_name, mj_handles, obj_root_id,
                         resolve_surface, sample_bench_episode, sample_targets, set_body_pos,
                         settle, to_base)

log = logging.getLogger("crossed_harvest")
N_QUERIES = 2


def harvest_house(eng, spec, bench, bench_idx, args):
    """All cells of one house -> (payload for houses/h{house}.npz or None, audit)."""
    from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom

    house = int(spec.house_index)
    pickup = get_pickup_body_name(eng.task)
    if not pickup:
        log.error("h%d: no pickup_obj_name on task config", house)
        return None, None
    model, data = mj_handles(eng)

    obs0, _ = eng.task.reset()
    q_pre = arm_qpos(eng)
    st_nat = settle(eng, args.settle_max, args.settle_window, args.settle_tol)
    q_home = arm_qpos(eng)
    arm_drift = float(np.linalg.norm(q_home - q_pre))
    if arm_drift > 0.02:
        log.warning("h%d: arm moved %.4f rad during the natural settle", house, arm_drift)
    snap0 = eng._snapshot(obs0)
    R, t = base_transform(eng)
    p_nat = np.asarray(eng.read_object(), np.float64)
    z_nat = float(p_nat[2])
    ee_home = np.asarray(eng.read_ee(), np.float64)
    z_surf_b = float(to_base(R, t, p_nat)[2])
    surf = resolve_surface(eng, pickup, args)
    root = obj_root_id(eng, pickup)
    roots = robot_root_ids(eng)

    audit = {"configs_tried": 0, "kept_configs": 0, "cells_tried": 0, "kept_cells": 0,
             "kept_rows": 0, "bad_settle": 0, "geom_changed": 0, "encode_fail": 0,
             "short_rollout": 0, "reject_contact": 0, "samples_short": 0, "sample_tries": 0,
             "lean_flags": 0,
             "pose_reject_ik": 0, "pose_reject_contact": 0, "pose_tries": 0, "poses_short": 0,
             "cell_skip_near_obj": 0, "cell_skip_contact": 0, "dark_frames": 0,
             "surface_fallback": bool(surf["fallback"]), "natural_settle_disp": st_nat["last_disp"],
             "arm_settle_drift_rad": arm_drift}

    rng_obj = np.random.default_rng(7919 * house + 37)
    targets = sample_targets(eng, rng_obj, surf, pickup, R, t, z_nat, args, audit)
    targets = targets[: args.n_obj]
    eng._restore(snap0)

    rng_pose = np.random.default_rng(104729 * house + 52)
    poses = sample_poses(eng, rng_pose, q_home, z_surf_b, root, args, audit)
    eng._restore(snap0)
    if len(poses) < 2:
        log.warning("h%d: only %d arm poses, house skipped (audit=%s)", house, len(poses), audit)
        return None, audit

    X = {v: [] for v in VARIANTS_LLM}
    AE_TOK, AE_POOL, CHUNKS, THUMBS = [], [], [], []
    lab = {k: [] for k in ["config_id", "pose_id", "chunk_q", "obj_base", "ee", "rel_ee", "dist",
                           "u", "obj_world", "cmd_xy", "ee_target", "ee_tele", "qpos_set", "qpos_q",
                           "ik_err", "step", "settle_disp", "lean_contacts", "fwd", "bearing_deg",
                           "ee44", "in_env", "exo_px"]}
    t_house0 = time.time()

    for ci, xy in enumerate(targets):
        audit["configs_tried"] += 1
        obs = eng._restore(snap0)
        data.qacc_warmstart[:] = 0.0
        eng.policy.reset()
        eng._chunk = 0
        if ci > 0:
            set_body_pos(eng.task, pickup, [float(xy[0]), float(xy[1]), z_nat])
        st = settle(eng, args.settle_max, args.settle_window, args.settle_tol)
        ow = np.asarray(eng.read_object(), np.float64)
        if ci > 0 and float(np.hypot(*(ow[:2] - xy))) > args.xy_tol:
            audit["bad_settle"] += 1
            continue
        if abs(float(ow[2]) - z_nat) > args.z_tol:
            audit["bad_settle"] += 1
            continue
        if surf["support_root"] >= 0:
            gid2 = get_supporting_geom(data, root)
            if gid2 is None or int(gid2) != surf["gid"]:
                audit["geom_changed"] += 1
                continue
        lean = bad_contact_count(model, data, root, surf["support_root"]) \
            if surf["support_root"] >= 0 else 0
        if lean:
            audit["lean_flags"] += 1
        ob_place = to_base(R, t, ow)
        in_env = envelope_ok(ob_place, args)
        snap_cfg = eng._snapshot(obs)
        cfg_kept = 0

        for pi, pose in enumerate(poses):
            audit["cells_tried"] += 1
            obs = eng._restore(snap_cfg)
            data.qacc_warmstart[:] = 0.0
            eng.policy.reset()
            eng._chunk = 0
            if float(np.linalg.norm(pose["ee"] - ob_place)) < args.min_obj_dist:
                audit["cell_skip_near_obj"] += 1
                continue
            n_scene, n_obj = place_arm(eng, pose["q"], model, data, roots, root)
            if n_scene > 0 or n_obj > 0:
                audit["cell_skip_contact"] += 1
                continue
            obs = fresh_obs(eng)
            ee_tele = np.asarray(eng.read_ee(), np.float64)
            caps, states = run_queries(eng, obs, N_QUERIES, max_loop=N_QUERIES * 8 + 4)
            if len(caps) < N_QUERIES or len(states) < N_QUERIES:
                audit["short_rollout"] += 1
                continue
            if any(np.isfinite(s["exo_px"]) and s["exo_px"] < args.dark_px for s in states):
                audit["dark_frames"] += 1
                continue

            rows, rows_ok = [], True
            for q in range(N_QUERIES):
                enc = encode_query_all(eng, caps[q]["kw"], args.llm_layers, args.ae_layers,
                                       eng.scene_seed + 1000 * q)
                if enc is None or any(enc[0][v] is None for v in VARIANTS_LLM
                                      if v not in OPTIONAL_VARIANTS):
                    rows_ok = False
                    break
                if any(enc[0][v] is None for v in ("phrase_pickup", "phrase_pickup_end")):
                    audit["phrase_nan"] = audit.get("phrase_nan", 0) + 1
                rows.append((enc, caps[q], states[q]))
            if not rows_ok:
                audit["encode_fail"] += 1
                continue

            for q, (enc, cap, stt) in enumerate(rows):
                llm, ae_tok = enc
                for v in VARIANTS_LLM:
                    X[v].append(llm[v].astype(np.float16) if llm[v] is not None
                                else np.full((len(args.llm_layers), 2560), np.nan, np.float16))
                AE_TOK.append(ae_tok.astype(np.float16))
                AE_POOL.append(ae_tok.mean(1).astype(np.float16))
                CHUNKS.append(cap["actions"].astype(np.float32))
                THUMBS.append(stt["thumb"])
                ob = to_base(R, t, stt["obj_w"]).astype(np.float32)
                ee = np.asarray(stt["ee_b"], np.float32)
                rel = ob - ee
                d = float(np.linalg.norm(rel))
                lab["config_id"].append(ci)
                lab["pose_id"].append(pi)
                lab["chunk_q"].append(q)
                lab["obj_base"].append(ob)
                lab["ee"].append(ee)
                lab["rel_ee"].append(rel)
                lab["dist"].append(d)
                lab["u"].append((rel / max(d, 1e-9)).astype(np.float32))
                lab["obj_world"].append(np.asarray(stt["obj_w"], np.float32))
                lab["cmd_xy"].append(np.asarray(xy, np.float32))
                lab["ee_target"].append(np.asarray(pose["target"], np.float32))
                lab["ee_tele"].append(np.asarray(ee_tele, np.float32))
                lab["qpos_set"].append(np.asarray(pose["q"], np.float32))
                lab["qpos_q"].append(np.asarray(stt["qpos"], np.float32))
                lab["ik_err"].append(float(pose["ik_err"]))
                lab["step"].append(int(stt["step"]))
                lab["settle_disp"].append(float(st["last_disp"]))
                lab["lean_contacts"].append(int(lean))
                lab["fwd"].append(float(ob[0]))
                lab["bearing_deg"].append(float(np.degrees(np.arctan2(ob[1], ob[0]))))
                m44 = stt["ee44"]
                lab["ee44"].append(np.asarray(m44, np.float32) if m44 is not None
                                   else np.full((4, 4), np.nan, np.float32))
                lab["in_env"].append(int(in_env))
                lab["exo_px"].append(float(stt["exo_px"]))
            cfg_kept += 1
            audit["kept_cells"] += 1
            audit["kept_rows"] += N_QUERIES
        if cfg_kept:
            audit["kept_configs"] += 1

    if not lab["dist"]:
        log.warning("h%d: nothing kept (audit=%s)", house, audit)
        return None, audit

    payload = dict(
        ae_tok=np.stack(AE_TOK), ae=np.stack(AE_POOL), chunk=np.stack(CHUNKS),
        exo_thumb=np.stack(THUMBS),
        obj_base=np.stack(lab["obj_base"]), ee=np.stack(lab["ee"]),
        rel_ee=np.stack(lab["rel_ee"]), u=np.stack(lab["u"]),
        dist=np.array(lab["dist"], np.float32),
        config_id=np.array(lab["config_id"], np.int32),
        pose_id=np.array(lab["pose_id"], np.int32),
        chunk_q=np.array(lab["chunk_q"], np.int32),
        obj_world=np.stack(lab["obj_world"]), cmd_xy=np.stack(lab["cmd_xy"]),
        ee_target=np.stack(lab["ee_target"]), ee_tele=np.stack(lab["ee_tele"]),
        qpos_set=np.stack(lab["qpos_set"]), qpos_q=np.stack(lab["qpos_q"]),
        ik_err=np.array(lab["ik_err"], np.float32),
        step=np.array(lab["step"], np.int32),
        settle_disp=np.array(lab["settle_disp"], np.float32),
        lean_contacts=np.array(lab["lean_contacts"], np.int32),
        fwd=np.array(lab["fwd"], np.float32),
        bearing_deg=np.array(lab["bearing_deg"], np.float32),
        ee44=np.stack(lab["ee44"]),
        in_env=np.array(lab["in_env"], np.int32),
        exo_px=np.array(lab["exo_px"], np.float32),
        pose_targets=np.stack([p["target"] for p in poses]).astype(np.float32),
        pose_q=np.stack([p["q"] for p in poses]).astype(np.float32),
        pose_ee=np.stack([p["ee"] for p in poses]).astype(np.float32),
        pose_ik_err=np.array([p["ik_err"] for p in poses], np.float32),
        q_home=q_home.astype(np.float32), ee_home=ee_home.astype(np.float32),
        z_surf_base=np.float32(z_surf_b),
        layers_llm=np.array(args.llm_layers, np.int32),
        layers_ae=np.array(args.ae_layers, np.int32),
        variants=np.array(VARIANTS_LLM), house=house, bench=str(bench), bench_idx=int(bench_idx),
        obj_name=str(spec.task.get("pickup_obj_name", "") if isinstance(spec.task, dict) else ""),
        instruction=json.dumps(getattr(spec, "language", None).model_dump()
                               if hasattr(getattr(spec, "language", None), "model_dump") else {}),
        base_pose=np.asarray(np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]), np.float32),
        natural_obj_world=np.asarray(p_nat, np.float32),
        surface=np.array([surf["xy_min"][0], surf["xy_min"][1], surf["xy_max"][0],
                          surf["xy_max"][1], surf["top_z"], float(surf["gid"])], np.float32),
        audit_json=np.frombuffer(json.dumps(audit).encode(), np.uint8))
    for v in VARIANTS_LLM:
        payload[f"X_{v}"] = np.stack(X[v])
    log.info("h%d [%s#%d]: %d rows / %d cells / %d placements kept, %d poses (%.1f min; audit=%s)",
             house, bench, bench_idx, audit["kept_rows"], audit["kept_cells"],
             audit["kept_configs"], len(poses), (time.time() - t_house0) / 60.0, audit)
    return payload, audit


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--out", type=Path, default=None,
                    help="store root (default $VLAGEOM_DATA/position/crossed)")
    ap.add_argument("--house-list", type=Path, default=crossed_house_list_path(),
                    help="CSV house,bench,bench_idx (default lists/position/crossed_houses.csv)")
    ap.add_argument("--houses", type=int, nargs="+", default=None,
                    help="run only these houses (must be in the house list)")
    ap.add_argument("--shard", type=int, default=None,
                    help="shard index (default $SLURM_ARRAY_TASK_ID; all houses if neither is set)")
    ap.add_argument("--houses-per-shard", type=int, default=6)
    ap.add_argument("--checkpoint", default=None, help="default $VLAGEOM_CHECKPOINT")
    ap.add_argument("--bench-mini", default=None, help="default $VLAGEOM_BENCH_MINI")
    ap.add_argument("--bench-hard", default=None, help="default $VLAGEOM_BENCH_HARD")
    # object placements (the object-teleport harvest's sampler settings)
    ap.add_argument("--n-obj", type=int, default=5, help="object placements per house")
    ap.add_argument("--n-targets", type=int, default=16,
                    help="placements drawn by the sampler (the first --n-obj are used)")
    ap.add_argument("--fwd-lo", type=float, default=0.25)
    ap.add_argument("--fwd-hi", type=float, default=0.75)
    ap.add_argument("--r-hi", type=float, default=0.70)
    ap.add_argument("--bearing-deg", type=float, default=30.0)
    ap.add_argument("--min-sep", type=float, default=0.03)
    ap.add_argument("--max-tries", type=int, default=600)
    ap.add_argument("--fallback-patch", type=float, default=0.15)
    ap.add_argument("--xy-tol", type=float, default=0.02)
    ap.add_argument("--z-tol", type=float, default=0.01)
    ap.add_argument("--settle-max", type=int, default=500)
    ap.add_argument("--settle-window", type=int, default=50)
    ap.add_argument("--settle-tol", type=float, default=1e-3)
    # arm poses
    ap.add_argument("--n-poses", type=int, default=10)
    ap.add_argument("--pose-fwd-lo", type=float, default=0.20)
    ap.add_argument("--pose-fwd-hi", type=float, default=0.60)
    ap.add_argument("--pose-lat", type=float, default=0.30)
    ap.add_argument("--pose-up-lo", type=float, default=0.05,
                    help="above the object's resting height (m)")
    ap.add_argument("--pose-up-hi", type=float, default=0.40)
    ap.add_argument("--pose-r-hi", type=float, default=0.62)
    ap.add_argument("--pose-min-sep", type=float, default=0.05)
    ap.add_argument("--pose-max-tries", type=int, default=400)
    ap.add_argument("--ik-iters", type=int, default=100)
    ap.add_argument("--ik-tol", type=float, default=0.015)
    ap.add_argument("--min-obj-dist", type=float, default=0.12)
    ap.add_argument("--dark-px", type=float, default=15.0)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="re-harvest houses whose output file exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the houses against the benchmark episodes, print them and exit")
    args = ap.parse_args()
    args.llm_layers = list(range(N_LAYERS))
    args.ae_layers = list(range(N_LAYERS))

    root = args.out or crossed_root()
    listed = read_house_list(args.house_list)
    if args.houses:
        known = {h for h, _, _ in listed}
        bad = [h for h in args.houses if h not in known]
        if bad:
            raise SystemExit(f"houses not in {args.house_list}: {bad}")
        listed = [r for r in listed if r[0] in set(args.houses)]
    else:
        shard = args.shard
        if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID", "") != "":
            shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if shard is not None:
            n = args.houses_per_shard
            listed = listed[shard * n:(shard + 1) * n]
    if not listed:
        log.info("nothing to do")
        return

    roots = paths.bench_roots()
    roots = {"mini": args.bench_mini or roots["mini"], "hard": args.bench_hard or roots["hard"]}
    catalog = {h: (b, i, sp) for h, b, i, sp in build_house_list(roots, 950)}
    mine = []
    for house, bench, bench_idx in listed:
        if house not in catalog or catalog[house][:2] != (bench, bench_idx):
            raise SystemExit(f"h{house}: benchmark episode {bench}#{bench_idx} not found "
                             f"(catalog has {catalog.get(house, (None, None))[:2]})")
        mine.append((house, bench, bench_idx, catalog[house][2]))
    log.info("%d houses: %s", len(mine), [h for h, *_ in mine])
    if args.dry_run:
        return

    out = root / "houses"
    out.mkdir(parents=True, exist_ok=True)
    eng, sim_envs = bench_engine(args.checkpoint, str(root / "_env"),
                                 need_mini=any(b == "mini" for _h, b, *_ in mine))
    task_horizon = N_QUERIES * 8 + 8
    n_done, n_tried, dark = 0, 0, False
    try:
        for house, bench, bench_idx, spec in mine:
            dst = out / f"h{house}.npz"
            if args.skip_existing and dst.exists():
                log.info("h%d exists, skipped", house)
                continue
            n_tried += 1
            payload, audit = None, None
            try:
                sample_bench_episode(eng, sim_envs, bench, spec, seed=104729 + house,
                                     task_horizon=task_horizon)
                payload, audit = harvest_house(eng, spec, bench, bench_idx, args)
            except Exception:
                log.exception("h%d failed; continuing with the next house", house)
                continue
            finally:
                eng.task = eng.policy = None
                gc.collect()
            if audit is not None and audit["cells_tried"] >= 10 and \
                    audit["dark_frames"] >= 0.5 * audit["cells_tried"]:
                log.error("h%d: %d/%d cells with a dark exo frame (mean pixel value < %.0f): the "
                          "renderer of this process is not producing images; exit 3", house,
                          audit["dark_frames"], audit["cells_tried"], args.dark_px)
                dark = True
                break
            if payload is None:
                continue
            save_npz(dst, payload)
            log.info("h%d -> %s (%d rows)", house, dst, len(payload["dist"]))
            n_done += 1
    finally:
        eng.close()
    log.info("done: %d/%d houses written (%d attempted)", n_done, len(mine), n_tried)
    if dark:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    if n_tried > 0 and n_done == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
