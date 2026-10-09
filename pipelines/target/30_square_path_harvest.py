#!/usr/bin/env python
"""Object-path harvest (GPU + EGL): move the pickup object around a square on its support
surface in one house and record the first-query activations at every waypoint.

This is the path drawn through the cloud in the target tile of Figure 1 (house 44, a mug, 64
waypoints). House 44 is left out of the read-out fit (50_panel.py), so the path is a held-out
prediction.

Per house (its MolmoSpaces benchmark episode, seed 104729 + house, as in 10_teleport_harvest.py):
  1. reset and settle with the robot held in place; take the support surface box as in
     10_teleport_harvest.py;
  2. design the square: the valid region is the surface box intersected with the reach envelope
     (forward 0.25-0.75 m, radius <= 0.70 m, |bearing| <= 30 deg) on a 120 x 120 grid; the square
     is centred on the region's mean, aligned with its principal axes, half-sides = the 95th
     percentile of |offset| along each axis, then scaled by 0.97 x 0.94^k with the smallest k
     that puts all waypoints inside the region; the 4 corners are waypoints and the rest are
     evenly spaced along the sides (counter-clockwise);
  3. skip waypoints where the object would penetrate another object; for each other waypoint:
     restore the settled state, move the object there, settle, keep it if it stays within 2 cm
     (xy) / 1 cm (z) of the command on the same support geom, render the observation, run the
     first model query (the robot does not move) and encode it: VLM 11 token-group pools x 13
     layers and action expert per token x 11 layers (as 10_teleport_harvest.py).

The settle step ends by moving the robot-mounted cameras to the robot's current pose.

Output (default $VLAGEOM_DATA/target/shape_square/h{house}/):
  acts.npz        X_{variant} (n, 13, 2560) float16 x 11, ae_tok (n, 11, 16, 768), ae, chunk,
                  layers_llm, layers_ae, variants
  manifest.json   house, shape, closed, waypoint list (realized obj_base / obj_world / ee_base,
                  commanded xy, settle displacement), corners, square centre / axes / half-sides /
                  scale, support surface, robot base pose, audit counters
  exo_{i:02d}.png the exo frame of waypoint i

Example (one GPU, about 2 min):
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python pipelines/target/30_square_path_harvest.py
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
from vlageom.sim import (AE_LAYERS, LLM_LAYERS, OPTIONAL_VARIANTS, VARIANTS, as_hw3,
                         bad_contact_count, base_transform, bench_engine, build_house_list,
                         encode_query, envelope_ok, exo_frame, fresh_obs, get_pickup_body_name,
                         mj_handles, obj_root_id, resolve_surface, run_chunks,
                         sample_bench_episode, set_body_pos, settle, to_base)
from vlageom.target import HELDOUT_HOUSE, square_path_root

log = logging.getLogger("square_path_harvest")


def square_uv(sa, sb, n):
    """n waypoints on the perimeter of the (+-sa, +-sb) rectangle, counter-clockwise from
    (sa, sb); each side starts at its corner, so the 4 corners are waypoints."""
    cs = np.array([[sa, sb], [-sa, sb], [-sa, -sb], [sa, -sb]], float)
    n4, rem = divmod(n, 4)
    out = []
    for k in range(4):
        m = n4 + (1 if k < rem else 0)
        frac = np.linspace(0.0, 1.0, m, endpoint=False)[:, None]
        out.append(cs[k] + frac * (cs[(k + 1) % 4] - cs[k]))
    return np.concatenate(out, 0)


def design_square(surf, R, t, z_nat, n, args):
    """The square path in world xy (see the module docstring), or None if the valid region has
    fewer than 40 grid points or no scale >= 0.25 fits. Returns dict(xy (n, 2), t (n,), closed,
    center, e1, e2, a, b, scale)."""
    lo, hi = np.asarray(surf["xy_min"], float), np.asarray(surf["xy_max"], float)
    gx = np.linspace(lo[0], hi[0], 120)
    gy = np.linspace(lo[1], hi[1], 120)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    P = np.stack([GX.ravel(), GY.ravel(), np.full(GX.size, z_nat)], 1)
    ok = np.array([envelope_ok(to_base(R, t, p), args) for p in P])
    if ok.sum() < 40:
        return None
    V = P[ok][:, :2]
    ctr = V.mean(0)
    Cv = V - ctr
    _ev, evec = np.linalg.eigh(np.cov(Cv.T))
    e1, e2 = evec[:, 1], evec[:, 0]                      # long, short principal axes
    a = float(np.percentile(np.abs(Cv @ e1), 95))
    b = float(np.percentile(np.abs(Cv @ e2), 95))
    tpar = np.arange(n, dtype=float) / n

    def gen(s):
        return ctr + square_uv(s * a, s * b, n) @ np.stack([e1, e2])

    s = 0.97
    while s > 0.25:
        W = gen(s)
        in_box = np.all((W >= lo) & (W <= hi), axis=1)
        in_env = np.array([envelope_ok(to_base(R, t, [w[0], w[1], z_nat]), args) for w in W])
        if bool(np.all(in_box & in_env)):
            break
        s *= 0.94
    else:
        return None
    return {"kind": "square", "closed": True, "xy": W, "t": tpar, "center": ctr,
            "e1": np.asarray(e1, float), "e2": np.asarray(e2, float), "a": a, "b": b, "scale": s}


def harvest_square(eng, spec, bench, args, out_root, n):
    """One house: design the square, harvest every waypoint, write acts.npz + manifest.json."""
    import imageio.v2 as imageio
    from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom

    house = int(spec.house_index)
    out = out_root / f"h{house}"
    pickup = get_pickup_body_name(eng.task)
    if not pickup:
        log.error("h%d: no pickup_obj_name", house)
        return False
    model, data = mj_handles(eng)

    obs0, _ = eng.task.reset()
    st_nat = settle(eng, args.settle_max, args.settle_window, args.settle_tol)
    snap0 = eng._snapshot(obs0)
    R, t = base_transform(eng)
    p_nat = np.asarray(eng.read_object(), np.float64)
    z_nat = float(p_nat[2])
    surf = resolve_surface(eng, pickup, args)
    root = obj_root_id(eng, pickup)
    if surf["fallback"]:
        log.error("h%d: no support surface found", house)
        return False

    des = design_square(surf, R, t, z_nat, n, args)
    if des is None:
        log.error("h%d: no square fits the valid region", house)
        return False
    log.info("h%d: square center=%s a=%.2f b=%.2f scale=%.2f (natural settle disp %.4f)",
             house, np.round(des["center"], 2).tolist(), des["a"], des["b"], des["scale"],
             st_nat["last_disp"])

    pre_ok = []
    for xy in des["xy"]:
        set_body_pos(eng.task, pickup, [float(xy[0]), float(xy[1]), z_nat])
        pre_ok.append(bad_contact_count(model, data, root, surf["support_root"]) == 0)
    set_body_pos(eng.task, pickup, [float(p_nat[0]), float(p_nat[1]), float(p_nat[2])])
    pre_ok = np.asarray(pre_ok)
    log.info("h%d: contact pre-check %d/%d pass", house, int(pre_ok.sum()), n)
    if pre_ok.sum() < n * 0.6:
        log.error("h%d: too many waypoints penetrate other objects", house)
        return False

    out.mkdir(parents=True, exist_ok=True)
    audit = {"skipped_contact": int((~pre_ok).sum()), "bad_settle": 0, "geom_changed": 0,
             "encode_fail": 0, "short_rollout": 0, "lean_flags": 0,
             "natural_settle_disp": st_nat["last_disp"]}
    X = {v: [] for v in VARIANTS}
    AE_TOK, AE_POOL, CHUNKS, pts = [], [], [], []
    t0 = time.time()

    for i, (xy, tpar) in enumerate(zip(des["xy"], des["t"])):
        if not pre_ok[i]:
            continue
        eng._restore(snap0)
        data.qacc_warmstart[:] = 0.0
        eng.policy.reset()
        eng._chunk = 0
        set_body_pos(eng.task, pickup, [float(xy[0]), float(xy[1]), z_nat])
        st = settle(eng, args.settle_max, args.settle_window, args.settle_tol)
        ow = np.asarray(eng.read_object(), np.float64)
        if float(np.hypot(*(ow[:2] - xy))) > args.xy_tol or abs(float(ow[2]) - z_nat) > args.z_tol:
            audit["bad_settle"] += 1
            continue
        gid2 = get_supporting_geom(data, root)
        if gid2 is None or int(gid2) != surf["gid"]:
            audit["geom_changed"] += 1
            continue
        lean = bad_contact_count(model, data, root, surf["support_root"])
        if lean:
            audit["lean_flags"] += 1

        obs = fresh_obs(eng)
        img = as_hw3(exo_frame(obs))
        caps, states = run_chunks(eng, obs, 1, max_loop=12)
        if len(caps) < 1 or len(states) < 1:
            audit["short_rollout"] += 1
            continue
        enc = encode_query(eng, caps[0]["kw"], args.llm_layers, args.ae_layers, eng.scene_seed)
        if enc is None or any(enc[0][v] is None for v in VARIANTS if v not in OPTIONAL_VARIANTS):
            audit["encode_fail"] += 1
            continue
        llm, ae_tok, _ids = enc

        png = f"exo_{i:02d}.png"
        imageio.imwrite(out / png, img)
        for v in VARIANTS:
            X[v].append(llm[v].astype(np.float16) if llm[v] is not None
                        else np.full((len(args.llm_layers), 2560), np.nan, np.float16))
        AE_TOK.append(ae_tok.astype(np.float16))
        AE_POOL.append(ae_tok.mean(1).astype(np.float16))
        CHUNKS.append(caps[0]["actions"].astype(np.float32))
        ob = to_base(R, t, states[0]["obj_w"])
        ee = np.asarray(states[0]["ee_b"], np.float64)
        pts.append({"i": int(i), "t_par": float(tpar),
                    "cmd_xy": [float(xy[0]), float(xy[1])],
                    "obj_base": [float(v) for v in ob],
                    "obj_world": [float(v) for v in states[0]["obj_w"]],
                    "ee_base": [float(v) for v in ee],
                    "dist": float(np.linalg.norm(ob - ee)),
                    "settle_disp": float(st["last_disp"]), "lean": int(lean),
                    "exo_png": png})
        if (i + 1) % 8 == 0:
            log.info("h%d: %d/%d waypoints (%d kept, %.1f min)", house, i + 1, n,
                     len(pts), (time.time() - t0) / 60.0)

    if len(pts) < n * 0.5:
        log.error("h%d: only %d/%d waypoints kept (audit=%s)", house, len(pts), n, audit)
        return False

    # corners = realized extremes along the square's axes, in base coordinates
    OB = np.array([p["obj_base"] for p in pts])
    e1b = to_base(R, t, t + np.array([des["e1"][0], des["e1"][1], 0.0]))
    e2b = to_base(R, t, t + np.array([des["e2"][0], des["e2"][1], 0.0]))
    p1, p2 = OB @ e1b, OB @ e2b
    corners = [OB[int(np.argmax(p1))], OB[int(np.argmin(p1))],
               OB[int(np.argmax(p2))], OB[int(np.argmin(p2))]]

    payload = dict(ae_tok=np.stack(AE_TOK), ae=np.stack(AE_POOL), chunk=np.stack(CHUNKS),
                   layers_llm=np.array(args.llm_layers, np.int32),
                   layers_ae=np.array(args.ae_layers, np.int32),
                   variants=np.array(VARIANTS))
    for v in VARIANTS:
        payload[f"X_{v}"] = np.stack(X[v])
    with open(out / "acts.npz.tmp", "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(out / "acts.npz.tmp", out / "acts.npz")

    (out / "manifest.json.tmp").write_text(json.dumps(
        {"house": house, "bench": str(bench), "shape": "square", "closed": bool(des["closed"]),
         "n_requested": int(n), "corners": [[float(v) for v in c] for c in corners],
         "center_world": des["center"].tolist(),
         "axes_world": [des["e1"].tolist(), des["e2"].tolist()],
         "half_extents": [des["a"], des["b"]], "scale": des["scale"], "z_nat": z_nat,
         "surface": {"xy_min": np.asarray(surf["xy_min"]).tolist(),
                     "xy_max": np.asarray(surf["xy_max"]).tolist(), "top_z": surf["top_z"]},
         "base_pose": np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]).tolist(),
         "audit": audit, "points": pts}, indent=1))
    os.replace(out / "manifest.json.tmp", out / "manifest.json")
    log.info("h%d -> %s (%d/%d waypoints, %.1f min; audit=%s)",
             house, out, len(pts), n, (time.time() - t0) / 60.0, audit)
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="output root; the house directory h{house}/ is created under it "
                         "(default $VLAGEOM_DATA/target/shape_square)")
    ap.add_argument("--house", type=int, default=HELDOUT_HOUSE)
    ap.add_argument("--n", type=int, default=64, help="waypoints")
    ap.add_argument("--house-list", type=Path, default=paths.lists_dir() / "teleport_houses.csv",
                    help="CSV house,bench,bench_idx giving the benchmark episode of the house")
    ap.add_argument("--checkpoint", default=None, help="default $VLAGEOM_CHECKPOINT")
    ap.add_argument("--bench-mini", default=None, help="default $VLAGEOM_BENCH_MINI")
    ap.add_argument("--bench-hard", default=None, help="default $VLAGEOM_BENCH_HARD")
    ap.add_argument("--fwd-lo", type=float, default=0.25)
    ap.add_argument("--fwd-hi", type=float, default=0.75)
    ap.add_argument("--r-hi", type=float, default=0.70)
    ap.add_argument("--bearing-deg", type=float, default=30.0)
    ap.add_argument("--fallback-patch", type=float, default=0.15)
    ap.add_argument("--xy-tol", type=float, default=0.02)
    ap.add_argument("--z-tol", type=float, default=0.01)
    ap.add_argument("--settle-max", type=int, default=500)
    ap.add_argument("--settle-window", type=int, default=50)
    ap.add_argument("--settle-tol", type=float, default=1e-3)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="re-harvest when manifest.json exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the house's benchmark episode, print it and exit")
    args = ap.parse_args()
    args.llm_layers, args.ae_layers = list(LLM_LAYERS), list(AE_LAYERS)

    out_root = args.out or square_path_root()
    with open(args.house_list, newline="") as f:
        listed = {int(r["house"]): (r["bench"], int(r["bench_idx"])) for r in csv.DictReader(f)}
    if args.house not in listed:
        raise SystemExit(f"house {args.house} is not in {args.house_list}")
    roots = paths.bench_roots()
    roots = {"mini": args.bench_mini or roots["mini"], "hard": args.bench_hard or roots["hard"]}
    catalog = {h: (b, i, sp) for h, b, i, sp in build_house_list(roots, 950)}
    bench, bench_idx, spec = catalog[args.house]
    if (bench, bench_idx) != listed[args.house]:
        raise SystemExit(f"h{args.house}: benchmark episode {bench}#{bench_idx} != list "
                         f"{listed[args.house]}")
    log.info("h%d: benchmark episode %s#%d", args.house, bench, bench_idx)
    if args.dry_run:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (out_root / f"h{args.house}" / "manifest.json").exists():
        log.info("h%d exists, skipped", args.house)
        return

    eng, sim_envs = bench_engine(args.checkpoint, str(out_root / "_env"), need_mini=bench == "mini")
    ok = False
    try:
        sample_bench_episode(eng, sim_envs, bench, spec, seed=104729 + args.house, task_horizon=16)
        ok = harvest_square(eng, spec, bench, args, out_root, args.n)
    except Exception:
        log.exception("h%d failed", args.house)
    finally:
        eng.task = eng.policy = None
        gc.collect()
        eng.close()
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
