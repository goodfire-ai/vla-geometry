#!/usr/bin/env python
"""Object-teleport harvest (GPU + EGL): VLM and action-expert activations for 16 placements of
the pickup object in each of the 906 benchmark houses (lists/teleport_houses.csv).

Per house (the house's MolmoSpaces benchmark episode, seed 104729 + house):
  1. reset the episode and settle the physics with the robot held in place; this state is
     placement 0 (the natural object position);
  2. sample 15 more object positions uniformly on the support surface under the object (its
     world-xy box, inset by the object footprint) inside the reach envelope (forward 0.25-0.75 m,
     horizontal radius <= 0.70 m, |bearing| <= 30 deg), at least 3 cm apart, without penetrating
     other objects (RNG seed 7919 * house + 37);
  3. for every placement: restore the settled state, move the object there (natural height),
     settle, and keep the placement only if the object stays within 2 cm (xy) and 1 cm (z) of
     the command on the same support geom; render the observation; let the policy run its own
     approach for 4 model queries (8 executed steps between queries) and record, at every query,
     the object and end-effector state the model saw; then re-encode every query context:
       VLM: 11 token-group pools (seq_all ... last_tok) x 13 layers, float16
       action expert, final flow step: 16 action tokens x 11 layers (and their mean), float16
     plus the predicted 16-step action chunk.

The settle step ends by moving the robot-mounted cameras to the robot's current pose, so every
placement's first observation is rendered from the arm pose the policy starts from.

Frame check: the first-query exo and wrist frames of each placement must have a mean pixel value
of at least --min-px; otherwise the renderer of this process is not producing images, and the
process exits with status 3 (so a job scheduler can restart it; houses already written are skipped).

Output (store root, default $VLAGEOM_DATA/target/teleport):
  houses/h{house}.npz   rows = kept placements x 4 queries; arrays X_{variant} (n, 13, 2560),
                        ae_tok (n, 11, 16, 768), ae (n, 11, 768), chunk (n, 16, 8) and labels
                        (obj_base, ee, rel_ee, dist, u, config_id, chunk_q, ...), audit_json
  frames/h{house}.npz   exo / wrist 96 px thumbnails of each kept placement's first query, their
                        config_id, and dark_json (a JSON list of placements with a dark
                        first-query frame; always empty, since a dark frame ends the process)

Work split: --shard K --houses-per-shard N takes houses [K*N, (K+1)*N) of the list (K defaults to
$SLURM_ARRAY_TASK_ID); --houses H [H ...] runs only those houses. Houses are independent.

Example (one GPU):
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python pipelines/target/10_teleport_harvest.py --houses 44
"""
from __future__ import annotations

import vlageom.sim  # noqa: F401  (first: sets up the file locks before molmo_spaces is imported)

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.sim import (AE_LAYERS, LLM_LAYERS, OPTIONAL_VARIANTS, VARIANTS, bad_contact_count,
                         base_transform, bench_engine, build_house_list, encode_query, envelope_ok,
                         fresh_obs, get_pickup_body_name, mj_handles, obj_root_id, obs_frames,
                         resolve_surface, run_chunks, sample_bench_episode, sample_targets,
                         set_body_pos, settle, thumb, to_base)
from vlageom.target import VARIANTS_LLM, teleport_root
from vlageom.util import save_npz

log = logging.getLogger("teleport_harvest")
assert VARIANTS == VARIANTS_LLM


def read_house_list(path):
    with open(path, newline="") as f:
        return [(int(r["house"]), r["bench"], int(r["bench_idx"])) for r in csv.DictReader(f)]


def check_frames(obs, min_px):
    """(exo, wrist) first-query frames; exit with status 3 if either is (nearly) black."""
    exo, wrist = obs_frames(obs)
    if exo is None or wrist is None:
        raise RuntimeError("observation: exo or wrist frame not found")
    me, mw = float(exo.mean()), float(wrist.mean())
    if me < min_px or mw < min_px:
        log.error("first-query frame mean pixel exo %.1f / wrist %.1f < %.1f: the renderer of this "
                  "process is not producing images; exit 3", me, mw, min_px)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    return exo, wrist


def harvest_house(eng, spec, bench, bench_idx, args):
    """All placements of one house -> (payload dict for houses/h{house}.npz, thumbnails) or
    (None, None) if no placement was kept."""
    from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom

    house = int(spec.house_index)
    pickup = get_pickup_body_name(eng.task)
    if not pickup:
        log.error("h%d: no pickup_obj_name on task config", house)
        return None, None
    model, data = mj_handles(eng)

    def arm_qpos():
        try:
            rv = eng.task._env.robots[0].robot_view
            return np.asarray(rv.get_move_group("arm").joint_pos, np.float64)
        except Exception:
            return None

    obs0, _ = eng.task.reset()
    q_pre = arm_qpos()
    st_nat = settle(eng, args.settle_max, args.settle_window, args.settle_tol)
    q_post = arm_qpos()
    arm_drift = float(np.linalg.norm(q_post - q_pre)) if q_pre is not None and q_post is not None \
        else float("nan")
    if np.isfinite(arm_drift) and arm_drift > 0.02:
        log.warning("h%d: arm moved %.4f rad during the natural settle", house, arm_drift)
    snap0 = eng._snapshot(obs0)
    R, t = base_transform(eng)
    p_nat = np.asarray(eng.read_object(), np.float64)
    z_nat = float(p_nat[2])
    surf = resolve_surface(eng, pickup, args)
    root = obj_root_id(eng, pickup)

    audit = {"configs_tried": 0, "kept_rows": 0, "kept_configs": 0, "bad_settle": 0,
             "geom_changed": 0, "encode_fail": 0, "short_rollout": 0, "reject_contact": 0,
             "samples_short": 0, "sample_tries": 0, "lean_flags": 0,
             "surface_fallback": bool(surf["fallback"]), "natural_settle_disp": st_nat["last_disp"],
             "arm_settle_drift_rad": arm_drift}
    rng = np.random.default_rng(7919 * house + 37)
    targets = sample_targets(eng, rng, surf, pickup, R, t, z_nat, args, audit)

    X = {v: [] for v in VARIANTS}
    AE_TOK, AE_POOL, CHUNKS = [], [], []
    lab = {k: [] for k in ["config_id", "chunk_q", "obj_base", "ee", "rel_ee", "dist", "u",
                           "obj_world", "cmd_xy", "step", "settle_disp", "lean_contacts",
                           "fwd", "bearing_deg", "ee44", "in_env"]}
    thumbs = {"exo": [], "wrist": []}
    t_house0 = time.time()

    for ci, xy in enumerate(targets):
        audit["configs_tried"] += 1
        eng._restore(snap0)
        data.qacc_warmstart[:] = 0.0                       # placement-independent solver warmstart
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
            audit["lean_flags"] += 1                       # recorded, not rejected

        obs = fresh_obs(eng)
        exo, wrist = check_frames(obs, args.min_px)
        ob_place = to_base(R, t, ow)
        in_env = envelope_ok(ob_place, args)               # realized placement (0 may be outside)
        caps, states = run_chunks(eng, obs, args.n_chunks, max_loop=args.n_chunks * 8 + 4)
        if len(caps) < args.n_chunks or len(states) < args.n_chunks:
            audit["short_rollout"] += 1
            continue

        rows_ok = True
        rows = []
        for q in range(args.n_chunks):
            enc = encode_query(eng, caps[q]["kw"], args.llm_layers, args.ae_layers,
                               eng.scene_seed + 1000 * q)
            if enc is None or any(enc[0][v] is None for v in VARIANTS if v not in OPTIONAL_VARIANTS):
                rows_ok = False
                break
            if any(enc[0][v] is None for v in ("phrase_pickup", "phrase_pickup_end")):
                audit["phrase_nan"] = audit.get("phrase_nan", 0) + 1
            rows.append((enc, caps[q], states[q]))
        if not rows_ok:
            audit["encode_fail"] += 1
            continue

        for q, (enc, cap, stt) in enumerate(rows):
            llm, ae_tok, _ids = enc
            for v in VARIANTS:
                X[v].append(llm[v].astype(np.float16) if llm[v] is not None
                            else np.full((len(args.llm_layers), 2560), np.nan, np.float16))
            AE_TOK.append(ae_tok.astype(np.float16))
            AE_POOL.append(ae_tok.mean(1).astype(np.float16))
            CHUNKS.append(cap["actions"].astype(np.float32))
            ob = to_base(R, t, stt["obj_w"]).astype(np.float32)
            ee = np.asarray(stt["ee_b"], np.float32)
            rel = ob - ee
            d = float(np.linalg.norm(rel))
            lab["config_id"].append(ci)
            lab["chunk_q"].append(q)
            lab["obj_base"].append(ob)
            lab["ee"].append(ee)
            lab["rel_ee"].append(rel)
            lab["dist"].append(d)
            lab["u"].append((rel / max(d, 1e-9)).astype(np.float32))
            lab["obj_world"].append(np.asarray(stt["obj_w"], np.float32))
            lab["cmd_xy"].append(np.asarray(xy, np.float32))
            lab["step"].append(int(stt["step"]))
            lab["settle_disp"].append(float(st["last_disp"]))
            lab["lean_contacts"].append(int(lean))
            lab["fwd"].append(float(ob[0]))
            lab["bearing_deg"].append(float(np.degrees(np.arctan2(ob[1], ob[0]))))
            m44 = stt["ee44"]
            lab["ee44"].append(np.asarray(m44, np.float32) if m44 is not None
                               else np.full((4, 4), np.nan, np.float32))
            lab["in_env"].append(int(in_env))
        thumbs["exo"].append(thumb(exo))
        thumbs["wrist"].append(thumb(wrist))
        audit["kept_configs"] += 1
        audit["kept_rows"] += args.n_chunks

    if not lab["dist"]:
        log.warning("h%d: nothing kept (audit=%s)", house, audit)
        return None, None

    payload = dict(
        ae_tok=np.stack(AE_TOK), ae=np.stack(AE_POOL), chunk=np.stack(CHUNKS),
        obj_base=np.stack(lab["obj_base"]), ee=np.stack(lab["ee"]),
        rel_ee=np.stack(lab["rel_ee"]), u=np.stack(lab["u"]),
        dist=np.array(lab["dist"], np.float32),
        config_id=np.array(lab["config_id"], np.int32),
        chunk_q=np.array(lab["chunk_q"], np.int32),
        obj_world=np.stack(lab["obj_world"]), cmd_xy=np.stack(lab["cmd_xy"]),
        step=np.array(lab["step"], np.int32),
        settle_disp=np.array(lab["settle_disp"], np.float32),
        lean_contacts=np.array(lab["lean_contacts"], np.int32),
        fwd=np.array(lab["fwd"], np.float32),
        bearing_deg=np.array(lab["bearing_deg"], np.float32),
        ee44=np.stack(lab["ee44"]),
        in_env=np.array(lab["in_env"], np.int32),
        layers_llm=np.array(args.llm_layers, np.int32),
        layers_ae=np.array(args.ae_layers, np.int32),
        variants=np.array(VARIANTS), house=house, bench=str(bench), bench_idx=int(bench_idx),
        obj_name=str(getattr(spec.task, "get", lambda *_: "")("pickup_obj_name", "")
                     if isinstance(spec.task, dict) else ""),
        instruction=json.dumps(getattr(spec, "language", None).model_dump()
                               if hasattr(getattr(spec, "language", None), "model_dump") else {}),
        base_pose=np.asarray(np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]), np.float32),
        spec_base_pose=np.asarray(spec.task.get("robot_base_pose", [np.nan] * 7), np.float32)
        if isinstance(spec.task, dict) else np.full(7, np.nan, np.float32),
        spec_obj_pose=np.asarray(spec.task.get("pickup_obj_start_pose", [np.nan] * 7), np.float32)
        if isinstance(spec.task, dict) else np.full(7, np.nan, np.float32),
        natural_obj_world=np.asarray(p_nat, np.float32),
        surface=np.array([surf["xy_min"][0], surf["xy_min"][1], surf["xy_max"][0],
                          surf["xy_max"][1], surf["top_z"], float(surf["gid"])], np.float32),
        audit_json=np.frombuffer(json.dumps(audit).encode(), np.uint8))
    for v in VARIANTS:
        payload[f"X_{v}"] = np.stack(X[v])
    frames = dict(exo=np.stack(thumbs["exo"]), wrist=np.stack(thumbs["wrist"]),
                  config_id_q0=payload["config_id"][payload["chunk_q"] == 0],
                  dark_json=np.frombuffer(json.dumps([]).encode(), np.uint8))
    log.info("h%d [%s#%d]: %d rows / %d placements kept (%.1f min; audit=%s)",
             house, bench, bench_idx, audit["kept_rows"], audit["kept_configs"],
             (time.time() - t_house0) / 60.0, audit)
    return payload, frames


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="store root (default $VLAGEOM_DATA/target/teleport)")
    ap.add_argument("--house-list", type=Path, default=paths.lists_dir() / "teleport_houses.csv",
                    help="CSV house,bench,bench_idx (default lists/teleport_houses.csv)")
    ap.add_argument("--houses", type=int, nargs="+", default=None,
                    help="run only these houses (must be in the house list)")
    ap.add_argument("--shard", type=int, default=None,
                    help="shard index (default $SLURM_ARRAY_TASK_ID; all houses if neither is set)")
    ap.add_argument("--houses-per-shard", type=int, default=8)
    ap.add_argument("--checkpoint", default=None, help="default $VLAGEOM_CHECKPOINT")
    ap.add_argument("--bench-mini", default=None, help="default $VLAGEOM_BENCH_MINI")
    ap.add_argument("--bench-hard", default=None, help="default $VLAGEOM_BENCH_HARD")
    ap.add_argument("--n-targets", type=int, default=16, help="placements per house, incl. natural")
    ap.add_argument("--n-chunks", type=int, default=4, help="model queries per placement")
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
    ap.add_argument("--min-px", type=float, default=3.0,
                    help="minimum mean pixel value of the first-query exo and wrist frames")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="re-harvest houses whose output file exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the houses against the benchmark episodes, print them and exit")
    args = ap.parse_args()
    args.llm_layers, args.ae_layers = list(LLM_LAYERS), list(AE_LAYERS)

    root = args.out or teleport_root()
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
    fdir = root / "frames"
    out.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    eng, sim_envs = bench_engine(args.checkpoint, str(root / "_env"),
                                 need_mini=any(b == "mini" for _h, b, *_ in mine))
    task_horizon = args.n_chunks * 8 + 8
    n_done = 0
    try:
        for house, bench, bench_idx, spec in mine:
            dst = out / f"h{house}.npz"
            if args.skip_existing and dst.exists():
                log.info("h%d exists, skipped", house)
                continue
            try:
                sample_bench_episode(eng, sim_envs, bench, spec, seed=104729 + house,
                                     task_horizon=task_horizon)
                payload, frames = harvest_house(eng, spec, bench, bench_idx, args)
            except Exception:
                log.exception("h%d failed; continuing with the next house", house)
                continue
            finally:
                eng.task = eng.policy = None
                gc.collect()
            if payload is None:
                continue
            save_npz(fdir / f"h{house}.npz", frames, compressed=True)
            save_npz(dst, payload, compressed=True)
            log.info("h%d -> %s (%d rows)", house, dst, len(payload["dist"]))
            n_done += 1
    finally:
        eng.close()
    log.info("done: %d/%d houses written", n_done, len(mine))


if __name__ == "__main__":
    main()
