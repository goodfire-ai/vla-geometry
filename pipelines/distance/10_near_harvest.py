"""Stage 10 (GPU + EGL): near-range harvest of the distance tile.

For each house of lists/near_houses.csv the benchmark episode is loaded, and the target object is
placed at 8 positions: placement 0 is its natural position, placements 1-7 are random positions on
its support surface whose 3-D distance to the natural end-effector position is at most 0.35 m (and
more than 5 cm in front of the robot, within 75 degrees of straight ahead). For every placement the
scene is restored to the episode start, the object is moved, physics settles with the robot held
still, and the policy runs closed loop for up to 8 model queries (q0..q7). The rollout stops early
when the task reports success; the queries before that are kept. At every query the VLM token pools
(11 pools x 13 layers), the action-expert tokens (11 layers) and labels are stored: object and
end-effector positions in the robot base frame, their distance, gripper closure, object contacts
with anything other than its support, and task success.

Output (under --out, default $VLAGEOM_DATA/distance/near):
  houses/h<house>.npz   one file per house, written when the house is done
  frames/h<house>.npz   96 x 96 thumbnails of the exo and wrist frames at q0 of every placement

Before the first query of each placement the exo and wrist frames are checked: if either has a
mean pixel value below --min-frame-px, the EGL renderer of this process has failed, and the
process exits with code 3 without writing that house. Houses already written are skipped on the
next run, so a job array can simply restart the task. Any other per-house error is logged, the
house is skipped and the process exits with code 1 at the end.

Sharding: the 300 houses (in list order) are split into shards of --houses-per-shard houses; run
shard --shard (default $SLURM_ARRAY_TASK_ID). --houses runs an explicit set instead.
Example (one house): MUJOCO_GL=egl python pipelines/distance/10_near_harvest.py --houses 0
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
from types import SimpleNamespace

import numpy as np

from vlageom import paths
from vlageom.distance import near_root
from vlageom.sim import (AE_LAYERS, LLM_LAYERS, OPTIONAL_VARIANTS, VARIANTS, bad_contact_count,
                         base_transform, bench_engine, build_house_list, encode_query,
                         envelope_ok, fresh_obs, get_pickup_body_name, mj_handles, obs_frames,
                         obj_root_id, resolve_surface, sample_bench_episode, set_body_pos, settle,
                         thumb, to_base)

log = logging.getLogger("distance.near_harvest")

# Harvest settings of the paper's near-range store.
CFG = SimpleNamespace(
    n_targets=8,          # placements per house, including placement 0 (natural position)
    n_chunks=8,           # model queries per placement (q0..q7)
    near_hi=0.35,         # max 3-D distance (m) from a placement to the natural EE position
    min_fwd=0.05,         # placements must lie at least this far (m) in front of the robot base ...
    max_bearing=75.0,     # ... and within this angle (deg) of straight ahead
    min_sep=0.02,         # min xy distance (m) between placements
    max_tries=600,        # random surface positions tried per house
    fallback_patch=0.15,  # half-size (m) of the square used when no support surface is found
    xy_tol=0.02,          # a settled object must stay within this xy distance (m) of its placement
    z_tol=0.01,           # ... and within this height (m) of its natural height
    settle_max=500, settle_window=50, settle_tol=1e-3,
    # reach envelope of the teleport harvest; here it only sets the `in_env` label
    fwd_lo=0.25, fwd_hi=0.75, r_hi=0.70, bearing_deg=30.0,
    episode_seed0=104729,  # episode seed = episode_seed0 + house (same as the teleport harvest)
    placement_seed=(7919, 45),  # placement rng = default_rng(7919 * house + 45)
)


def read_grip(eng):
    """Gripper closure: the sum of the gripper drive-joint angles (about 0 = open, about 1.6 =
    fully closed), or the gripper actuator's control value if the move group cannot be read; NaN
    on failure. Stored as `grip_width` (larger = more closed)."""
    try:
        rv = eng.task._env.robots[0].robot_view
        return float(np.sum(np.asarray(rv.get_move_group("gripper").joint_pos, np.float64)))
    except Exception:
        pass
    try:
        model, data = mj_handles(eng)
        for a in range(model.nu):
            if "gripper" in (model.actuator(a).name or "").lower():
                return float(data.ctrl[a])
    except Exception:
        pass
    return float("nan")


def sample_near_targets(eng, rng, surf, pickup, R, t, z_nat, ee_w, audit):
    """Placement xy positions: [natural xy] + up to n_targets-1 random surface positions within
    near_hi (3-D) of the natural end-effector position, rejected if the object would penetrate
    anything other than its support. Moves the object while testing and puts it back."""
    model, data = mj_handles(eng)
    p_nat = np.asarray(eng.read_object(), np.float64)
    accepted = [p_nat[:2].copy()]
    dz = float(ee_w[2] - z_nat)
    audit["near_dz"] = dz
    if dz * dz >= CFG.near_hi ** 2:          # the sphere around the end effector misses the surface
        audit["near_infeasible"] = 1
        return accepted
    tries = 0
    while len(accepted) < CFG.n_targets and tries < CFG.max_tries:
        tries += 1
        xy = rng.uniform(surf["xy_min"], surf["xy_max"])
        if any(np.hypot(*(xy - a)) < CFG.min_sep for a in accepted):
            continue
        if float(np.linalg.norm(np.array([xy[0], xy[1], z_nat]) - ee_w)) > CFG.near_hi:
            continue
        pb = to_base(R, t, [xy[0], xy[1], z_nat])
        if pb[0] < CFG.min_fwd or abs(np.degrees(np.arctan2(pb[1], pb[0]))) > CFG.max_bearing:
            continue
        set_body_pos(eng.task, pickup, [float(xy[0]), float(xy[1]), float(z_nat)])
        if surf["support_root"] >= 0 and \
                bad_contact_count(model, data, obj_root_id(eng, pickup), surf["support_root"]) > 0:
            audit["reject_contact"] += 1
            continue
        accepted.append(xy.copy())
    audit["sample_tries"] = tries
    if len(accepted) < CFG.n_targets:
        audit["samples_short"] = CFG.n_targets - len(accepted)
        log.warning("near sampler: %d/%d targets (tries=%d)", len(accepted), CFG.n_targets, tries)
    set_body_pos(eng.task, pickup, [float(p_nat[0]), float(p_nat[1]), float(p_nat[2])])
    return accepted


def run_chunks_near(eng, obs, n_chunks, max_loop, model, data, root, support_root):
    """Closed-loop rollout from `obs` until `n_chunks` model queries have fired, the task succeeds,
    or `max_loop` env steps. Query q uses flow noise seeded with eng.scene_seed + 1000 * q.

    After every query the state the model saw is read: object (world) and end-effector (base)
    positions, EE pose, env step, gripper closure, contact count and task success. Success comes
    from task.judge_success() (the benchmark tasks do not end by themselves), and the rollout stops
    at the first successful query. Returns (caps, states, done)."""
    import torch
    caps, states = [], []

    def gen(**kw):
        q = eng._chunk
        eng._chunk += 1
        seed = eng.scene_seed + 1000 * q
        kw2, steps = eng._seeded_kw(kw, seed)
        kwc = {k: (v.detach().clone() if torch.is_tensor(v) else v)
               for k, v in kw2.items() if k != "generator"}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = eng._gen_orig(num_steps=steps, **kw2)
        caps.append({"kw": kwc, "actions": out[0].detach().float().cpu().numpy(), "q": q})
        return out

    orig = eng.model.generate_actions
    eng.model.generate_actions = gen
    try:
        step = 0
        while step < max_loop and not eng.task.is_done():
            pre = len(caps)
            action = eng.policy.get_action(obs)
            if action is None:
                break
            if len(caps) > pre:
                ow = np.asarray(eng.read_object(), np.float64)
                ee = np.asarray(eng.read_ee(), np.float64)
                if ow.size != 3 or ee.size != 3:
                    break
                succ = False
                try:
                    succ = bool(eng.task.judge_success())
                except Exception:
                    pass
                states.append({"obj_w": ow, "ee_b": ee, "ee44": eng.read_ee_full(),
                               "step": step, "grip": read_grip(eng), "succ": succ,
                               "contact": bad_contact_count(model, data, root, support_root)})
                if succ:
                    break
            if len(caps) >= n_chunks:
                break
            obs, *_ = eng.task.step(action)
            step += 1
    finally:
        eng.model.generate_actions = orig
    done = bool(states and states[-1]["succ"])
    return caps, states, done


def check_frames(obs, min_px, thumbs):
    """Exit the process (code 3) if the exo or wrist frame of `obs` is blank, else keep
    thumbnails of both."""
    exo, wrist = obs_frames(obs)
    if exo is None or wrist is None:
        raise RuntimeError("q0 observation: exo or wrist frame not found")
    me, mw = float(exo.mean()), float(wrist.mean())
    if me < min_px or mw < min_px:
        log.error("q0 blank frame (exo %.1f, wrist %.1f mean px): renderer failed -> exit 3",
                  me, mw)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    thumbs["exo"].append(thumb(exo))
    thumbs["wrist"].append(thumb(wrist))


def harvest_house_near(eng, spec, bench, bench_idx, min_px, thumbs):
    """All placements of one house -> the house payload dict (None if no row was kept)."""
    from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom

    house = int(spec.house_index)
    pickup = get_pickup_body_name(eng.task)
    if not pickup:
        log.error("h%d: no pickup_obj_name", house)
        return None
    model, data = mj_handles(eng)

    obs0, _ = eng.task.reset()
    st_nat = settle(eng, CFG.settle_max, CFG.settle_window, CFG.settle_tol)
    snap0 = eng._snapshot(obs0)
    R, t = base_transform(eng)
    p_nat = np.asarray(eng.read_object(), np.float64)
    z_nat = float(p_nat[2])
    ee_b0 = np.asarray(eng.read_ee(), np.float64)
    ee_w = R @ ee_b0 + t                                   # natural EE position, world frame
    surf = resolve_surface(eng, pickup, CFG)
    root = obj_root_id(eng, pickup)

    audit = {"configs_tried": 0, "kept_rows": 0, "kept_configs": 0, "partial_configs": 0,
             "bad_settle": 0, "geom_changed": 0, "encode_fail": 0, "short_rollout": 0,
             "reject_contact": 0, "samples_short": 0, "sample_tries": 0, "lean_flags": 0,
             "done_configs": 0, "surface_fallback": bool(surf["fallback"]),
             "natural_settle_disp": st_nat["last_disp"]}
    a, b = CFG.placement_seed
    rng = np.random.default_rng(a * house + b)
    targets = sample_near_targets(eng, rng, surf, pickup, R, t, z_nat, ee_w, audit)

    X = {v: [] for v in VARIANTS}
    AE_TOK, AE_POOL, CHUNKS = [], [], []
    lab = {k: [] for k in ["config_id", "chunk_q", "obj_base", "ee", "rel_ee", "dist", "u",
                           "obj_world", "cmd_xy", "step", "settle_disp", "lean_contacts",
                           "fwd", "bearing_deg", "ee44", "in_env",
                           "grip_width", "contact_other", "ep_done", "success_q"]}
    t0 = time.time()

    for ci, xy in enumerate(targets):
        audit["configs_tried"] += 1
        eng._restore(snap0)
        data.qacc_warmstart[:] = 0.0
        eng.policy.reset()
        eng._chunk = 0
        if ci > 0:
            set_body_pos(eng.task, pickup, [float(xy[0]), float(xy[1]), z_nat])
        st = settle(eng, CFG.settle_max, CFG.settle_window, CFG.settle_tol)
        ow = np.asarray(eng.read_object(), np.float64)
        if ci > 0 and float(np.hypot(*(ow[:2] - xy))) > CFG.xy_tol:
            audit["bad_settle"] += 1
            continue
        if abs(float(ow[2]) - z_nat) > CFG.z_tol:
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

        obs = fresh_obs(eng)              # wrist camera at the restored arm pose, real renders
        in_env = envelope_ok(to_base(R, t, ow), CFG)
        check_frames(obs, min_px, thumbs)
        caps, states, done = run_chunks_near(eng, obs, CFG.n_chunks, CFG.n_chunks * 8 + 4,
                                             model, data, root, surf["support_root"])
        n_rows = min(len(caps), len(states))
        if n_rows == 0:
            audit["short_rollout"] += 1
            continue
        if done:
            audit["done_configs"] += 1
        if n_rows < CFG.n_chunks:
            audit["partial_configs"] += 1

        rows = []
        for q in range(n_rows):
            enc = encode_query(eng, caps[q]["kw"], LLM_LAYERS, AE_LAYERS, eng.scene_seed + 1000 * q)
            if enc is None or any(enc[0][v] is None
                                  for v in VARIANTS if v not in OPTIONAL_VARIANTS):
                audit["encode_fail"] += 1
                break                                    # keep the rows encoded so far
            rows.append((enc, caps[q], states[q]))
        if not rows:
            continue

        for q, (enc, cap, stt) in enumerate(rows):
            llm, ae_tok, _ids = enc
            for v in VARIANTS:
                X[v].append(llm[v].astype(np.float16) if llm[v] is not None
                            else np.full((len(LLM_LAYERS), 2560), np.nan, np.float16))
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
            lab["grip_width"].append(float(stt["grip"]))
            lab["contact_other"].append(int(stt["contact"]))
            lab["ep_done"].append(int(done))
            lab["success_q"].append(int(stt["succ"]))
        audit["kept_configs"] += 1
        audit["kept_rows"] += len(rows)

    if not lab["dist"]:
        log.warning("h%d: nothing kept (audit=%s)", house, audit)
        return None

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
        grip_width=np.array(lab["grip_width"], np.float32),
        contact_other=np.array(lab["contact_other"], np.int32),
        ep_done=np.array(lab["ep_done"], np.int32),
        success_q=np.array(lab["success_q"], np.int32),
        layers_llm=np.array(LLM_LAYERS, np.int32),
        layers_ae=np.array(AE_LAYERS, np.int32),
        variants=np.array(VARIANTS), house=house, bench=str(bench), bench_idx=int(bench_idx),
        obj_name=str(getattr(spec.task, "get", lambda *_: "")("pickup_obj_name", "")
                     if isinstance(spec.task, dict) else ""),
        instruction=json.dumps(getattr(spec, "language", None).model_dump()
                               if hasattr(getattr(spec, "language", None), "model_dump")
                               else {}),
        base_pose=np.asarray(np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]), np.float32),
        natural_obj_world=np.asarray(p_nat, np.float32),
        natural_ee_world=np.asarray(ee_w, np.float32),
        surface=np.array([surf["xy_min"][0], surf["xy_min"][1], surf["xy_max"][0],
                          surf["xy_max"][1], surf["top_z"], float(surf["gid"])], np.float32),
        audit_json=np.frombuffer(json.dumps(audit).encode(), np.uint8))
    for v in VARIANTS:
        payload[f"X_{v}"] = np.stack(X[v])
    log.info("h%d [%s#%d]: %d rows / %d configs (%d partial, %d done) in %.1f min (audit=%s)",
             house, bench, bench_idx, audit["kept_rows"], audit["kept_configs"],
             audit["partial_configs"], audit["done_configs"], (time.time() - t0) / 60.0, audit)
    return payload


def read_house_list(path):
    with open(path, newline="") as f:
        return [(int(r["house"]), r["bench"], int(r["bench_idx"])) for r in csv.DictReader(f)]


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Near-range harvest of the distance tile (GPU + EGL).",
        epilog="Writes <out>/houses/h<house>.npz and <out>/frames/h<house>.npz.")
    ap.add_argument("--out", type=Path, default=near_root(),
                    help="output root (default: $VLAGEOM_DATA/distance/near)")
    ap.add_argument("--house-list", type=Path, default=paths.lists_dir() / "near_houses.csv",
                    help="CSV with columns house, bench, bench_idx "
                         "(default: lists/near_houses.csv)")
    ap.add_argument("--shard", type=int, default=None,
                    help="shard index (default: $SLURM_ARRAY_TASK_ID, else 0)")
    ap.add_argument("--houses-per-shard", type=int, default=4,
                    help="houses per shard (default 4: 75 shards for 300 houses)")
    ap.add_argument("--houses", type=int, nargs="+", default=None,
                    help="harvest these houses (from the house list) instead of a shard")
    ap.add_argument("--checkpoint", default=None,
                    help="policy checkpoint (default: $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--min-frame-px", type=float, default=3.0,
                    help="blank-frame threshold on the mean pixel value of the q0 frames")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="re-harvest houses whose output file exists")
    args = ap.parse_args()

    wanted = read_house_list(args.house_list)
    catalog = {h: (bench, idx, spec) for h, bench, idx, spec in build_house_list(None, 950)}
    picked = []
    for house, bench, idx in wanted:
        if house not in catalog or catalog[house][:2] != (bench, idx):
            raise SystemExit(f"h{house}: benchmark episode {bench}#{idx} not found in the "
                             f"benchmark catalog (got {catalog.get(house, (None, None))[:2]})")
        picked.append((house, bench, idx, catalog[house][2]))
    if args.houses is not None:
        missing = set(args.houses) - {p[0] for p in picked}
        if missing:
            raise SystemExit(f"houses not in {args.house_list}: {sorted(missing)}")
        mine = [p for p in picked if p[0] in set(args.houses)]
    else:
        k = args.shard
        if k is None:
            env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            k = int(env_id) if env_id not in (None, "") else 0
        mine = picked[k * args.houses_per_shard:(k + 1) * args.houses_per_shard]
    if not mine:
        log.info("nothing to do")
        return
    log.info("%d houses: %s", len(mine), [p[0] for p in mine])

    out_h = args.out / "houses"
    out_f = args.out / "frames"
    out_h.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    eng, sim_envs = bench_engine(args.checkpoint or paths.checkpoint(),
                                 output_dir=str(args.out / "_env"),
                                 need_mini=any(b == "mini" for _h, b, *_ in mine))
    task_horizon = CFG.n_chunks * 8 + 8
    n_done, n_fail = 0, 0
    try:
        for house, bench, bench_idx, spec in mine:
            dst = out_h / f"h{house}.npz"
            if args.skip_existing and dst.exists():
                log.info("h%d exists -> skip", house)
                n_done += 1
                continue
            thumbs = {"exo": [], "wrist": []}
            try:
                sample_bench_episode(eng, sim_envs, bench, spec, seed=CFG.episode_seed0 + house,
                                     task_horizon=task_horizon)
                payload = harvest_house_near(eng, spec, bench, bench_idx, args.min_frame_px,
                                             thumbs)
            except Exception:
                log.exception("h%d failed; continuing", house)
                n_fail += 1
                continue
            finally:
                eng.task = eng.policy = None
                gc.collect()
            if payload is None:
                continue
            if thumbs["exo"]:
                np.savez_compressed(out_f / f"h{house}.npz", exo=np.stack(thumbs["exo"]),
                                    wrist=np.stack(thumbs["wrist"]))
            tmp = dst.with_name(dst.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **payload)
            os.replace(tmp, dst)
            log.info("h%d -> %s (%d rows)", house, dst, len(payload["dist"]))
            n_done += 1
    finally:
        eng.close()
    log.info("done: %d ok / %d failed / %d assigned", n_done, n_fail, len(mine))
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
