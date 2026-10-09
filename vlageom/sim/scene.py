"""Scene edits and geometry: object placement, support surfaces, contacts, physics settling.

Frames: "world" is the MuJoCo world frame; "base" is the robot base frame (x forward, y left,
z up), `p_base = R.T @ (p_world - t)` with `(R, t) = base_transform(eng)`.
"""
from __future__ import annotations

import logging

import numpy as np

from vlageom.sim.render import refresh_cameras

log = logging.getLogger(__name__)


# ---- MuJoCo handles --------------------------------------------------------------------------- #
def mj_handles(eng):
    """(mj_model, mj_data) of the active episode."""
    env = eng.task._env
    return env.mj_model, env.mj_datas[0]


def get_pickup_body_name(task) -> str:
    """MuJoCo body name of the task's pickup object ("" if the task has none)."""
    tc = getattr(getattr(task, "config", None), "task_config", None)
    return (getattr(tc, "pickup_obj_name", None) or "") if tc else ""


def _free_qadr(task, name: str) -> int:
    """qpos address of body `name`'s free joint (raises if the body is welded or not free)."""
    import mujoco

    env = task._env
    model, data = env.mj_model, env.mj_datas[0]
    bid = int(data.body(name).id)
    jntadr = int(model.body_jntadr[bid])
    if jntadr < 0:
        raise RuntimeError(f"Body '{name}' has no joint (welded?); cannot move it.")
    if int(model.jnt_type[jntadr]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError(
            f"Body '{name}' first joint (type={int(model.jnt_type[jntadr])}) is not a free joint; "
            "cannot move via qpos."
        )
    return int(model.jnt_qposadr[jntadr])


def set_body_pos(task, name: str, xyz):
    """Place free body `name` at world position `xyz` (orientation unchanged), then mj_forward.
    Returns (old_xyz, new_xyz) read back from mjData."""
    import mujoco

    env = task._env
    model, data = env.mj_model, env.mj_datas[0]
    qadr = _free_qadr(task, name)
    old = np.array(data.qpos[qadr: qadr + 3], dtype=float)
    data.qpos[qadr: qadr + 3] = np.asarray(xyz, dtype=float)
    mujoco.mj_forward(model, data)
    return old, np.array(data.qpos[qadr: qadr + 3], dtype=float)


def move_pickup_object(task, dx, dy, dz=0.0):
    """Translate the pickup object's free body by (dx, dy, dz) metres (qpos edit + mj_forward) and
    shift the task's recorded pickup start pose by the same amount. Returns (old_xyz, new_xyz)."""
    import mujoco

    name = get_pickup_body_name(task)
    if not name:
        raise RuntimeError("the task has no pickup object")
    env = task._env
    model, data = env.mj_model, env.mj_datas[0]
    qadr = _free_qadr(task, name)
    old = np.array(data.qpos[qadr: qadr + 3], dtype=float)
    data.qpos[qadr + 0] += dx
    data.qpos[qadr + 1] += dy
    data.qpos[qadr + 2] += dz
    mujoco.mj_forward(model, data)
    new = np.array(data.qpos[qadr: qadr + 3], dtype=float)
    try:
        tc = task.config.task_config
        sp = list(getattr(tc, "pickup_obj_start_pose", []) or [])
        if len(sp) >= 2:
            sp[0] += dx
            sp[1] += dy
            if len(sp) >= 3:
                sp[2] += dz
            tc.pickup_obj_start_pose = sp
    except Exception as e:
        log.warning("pickup start pose not updated: %s", e)
    log.debug("moved pickup %r by (%+.3f, %+.3f, %+.3f) m", name, float(dx), float(dy), float(dz))
    return old, new


# ---- frames ----------------------------------------------------------------------------------- #
def base_transform(eng, dtype=np.float64):
    """(R (3,3), t (3,)) of the robot base in the world frame, as `dtype` (float64 by default)."""
    rv = eng.task._env.robots[0].robot_view
    T = np.asarray(rv.base.pose, dtype)
    return T[:3, :3], T[:3, 3]


def to_base(R, t, p_world):
    """World point -> robot base frame."""
    return R.T @ (np.asarray(p_world, np.float64) - t)


def envelope_ok(pb, args) -> bool:
    """Is base-frame point `pb` inside the reach envelope? `args` provides fwd_lo, fwd_hi (m, along
    x), r_hi (m, horizontal radius) and bearing_deg (|atan2(y, x)| limit)."""
    fwd, lat = float(pb[0]), float(pb[1])
    r = float(np.hypot(fwd, lat))
    if not (args.fwd_lo <= fwd <= args.fwd_hi) or r > args.r_hi:
        return False
    return abs(np.degrees(np.arctan2(lat, fwd))) <= args.bearing_deg


# ---- contacts and support surface ------------------------------------------------------------- #
def obj_root_id(eng, name) -> int:
    model, data = mj_handles(eng)
    bid = int(data.body(name).id)
    return int(model.body_rootid[bid])


def bad_contact_count(model, data, obj_root, support_root) -> int:
    """Number of penetrating contacts (dist < 0) between the object's root body and any root body
    other than its support. Margin contacts (dist >= 0) are not counted."""
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        r1 = int(model.body_rootid[model.geom_bodyid[c.geom[0]]])
        r2 = int(model.body_rootid[model.geom_bodyid[c.geom[1]]])
        if float(c.dist) >= 0.0:
            continue
        if (r1 == obj_root) ^ (r2 == obj_root):
            other = r2 if r1 == obj_root else r1
            if other != support_root:
                n += 1
    return n


def sample_targets(eng, rng, surf, pickup, R, t, z_nat, args, audit):
    """Placement 0 = the natural object xy; then uniform xy on the surface box until n_targets
    are accepted (inside the envelope, >= min_sep from every accepted point, no penetrating
    contact with anything but the support). Moves the object while testing and puts it back."""
    model, data = mj_handles(eng)
    p_nat = np.asarray(eng.read_object(), np.float64)
    accepted = [p_nat[:2].copy()]
    tries = 0
    while len(accepted) < args.n_targets and tries < args.max_tries:
        tries += 1
        xy = rng.uniform(surf["xy_min"], surf["xy_max"])
        if any(np.hypot(*(xy - a)) < args.min_sep for a in accepted):
            continue
        if not envelope_ok(to_base(R, t, [xy[0], xy[1], z_nat]), args):
            continue
        set_body_pos(eng.task, pickup, [float(xy[0]), float(xy[1]), float(z_nat)])
        if surf["support_root"] >= 0 and \
                bad_contact_count(model, data, obj_root_id(eng, pickup), surf["support_root"]) > 0:
            audit["reject_contact"] += 1
            continue
        accepted.append(xy.copy())
    audit["sample_tries"] = tries
    if len(accepted) < args.n_targets:
        audit["samples_short"] = args.n_targets - len(accepted)
        log.warning("surface/envelope yielded %d/%d targets (tries=%d)",
                    len(accepted), args.n_targets, tries)
    set_body_pos(eng.task, pickup, [float(p_nat[0]), float(p_nat[1]), float(p_nat[2])])
    return accepted


def resolve_surface(eng, pickup, args) -> dict:
    """The support surface under the pickup object at the current (settled) state.

    Returns dict(gid, support_root, xy_min, xy_max, top_z, fallback, obj_dims): the world-xy box of
    the supporting geom, inset by the object's xy half-extent plus the offset between the object's
    AABB centre and its body origin, so a body origin placed anywhere in the box keeps the object on
    the slab. If no supporting geom is found, or the slab is smaller than the object, the box is a
    +-args.fallback_patch square around the object and `fallback` is True (gid -1 when no geom)."""
    from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb
    from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom

    model, data = mj_handles(eng)
    root = obj_root_id(eng, pickup)
    gid = get_supporting_geom(data, root)
    obj_geoms = [g for g in range(model.ngeom)
                 if int(model.body_rootid[model.geom_bodyid[g]]) == root]
    c_o, obj_dims = geom_aabb(model, data, obj_geoms)
    p_nat = np.asarray(eng.read_object(), np.float64)
    inset = np.asarray(obj_dims[:2], np.float64) / 2.0 + np.abs(np.asarray(c_o[:2]) - p_nat[:2])
    if gid is None:
        lo = p_nat[:2] - args.fallback_patch
        hi = p_nat[:2] + args.fallback_patch
        return {"gid": -1, "support_root": -1, "xy_min": lo, "xy_max": hi,
                "top_z": float(p_nat[2]), "fallback": True, "obj_dims": obj_dims}
    c_s, d_s = geom_aabb(model, data, [int(gid)])
    lo = c_s[:2] - d_s[:2] / 2.0 + inset
    hi = c_s[:2] + d_s[:2] / 2.0 - inset
    if np.any(lo >= hi):
        lo = p_nat[:2] - args.fallback_patch
        hi = p_nat[:2] + args.fallback_patch
    return {"gid": int(gid), "support_root": int(model.body_rootid[model.geom_bodyid[int(gid)]]),
            "xy_min": lo, "xy_max": hi, "top_z": float(c_s[2] + d_s[2] / 2.0),
            "fallback": False, "obj_dims": obj_dims}


# ---- physics settling ------------------------------------------------------------------------- #
def hold_robot(eng) -> None:
    """Write hold-position controls for every robot controller before stepping physics directly.

    `ctrl` is otherwise written only inside `task.step()` (scene reset leaves it at 0, which would
    drive the position actuators away from the current pose). `set_to_stationary()` sets each
    controller's target to the current joint positions and `compute_control()` writes it to ctrl."""
    for robot in eng.task._env.robots:
        for c in robot.controllers.values():
            c.set_to_stationary()
        robot.compute_control()


def settle(eng, max_steps=500, window=50, tol=1e-3) -> dict:
    """Step physics (robot held in place) until the pickup object moves less than `tol` metres in
    `window` steps, at most `max_steps` steps; then mj_forward and refresh robot-mounted cameras.
    Returns {"steps": n, "last_disp": metres moved in the last window}."""
    import mujoco

    model, data = mj_handles(eng)
    hold_robot(eng)
    name = get_pickup_body_name(eng.task)
    p_prev = np.asarray(data.body(name).xpos, np.float64).copy()
    n, disp = 0, 0.0
    while n < max_steps:
        for _ in range(window):
            mujoco.mj_step(model, data)
        n += window
        p = np.asarray(data.body(name).xpos, np.float64).copy()
        disp = float(np.linalg.norm(p - p_prev))
        if disp < tol:
            break
        p_prev = p
    mujoco.mj_forward(model, data)
    refresh_cameras(eng.task._env)
    return {"steps": n, "last_disp": disp}
