"""Arm configuration helpers: set joint angles directly, read joint limits, pick query poses.

`set_arm_qpos` changes the arm state without `env.step()`; render the result with
`vlageom.sim.render.fresh_obs`, which moves the wrist camera to the new hand pose first.
"""
from __future__ import annotations

import numpy as np

from vlageom.steer.delta import GRIP


def get_arm_movegroup(rv):
    """The arm MoveGroup of a robot view (the named "arm" group, else the first non-gripper group)."""
    try:
        mg = rv.get_move_group("arm")
        if mg is not None:
            return mg
    except Exception:
        pass
    grip_ids = set(rv.get_gripper_movegroup_ids())
    for mid in rv.get_move_group_ids():
        if mid not in grip_ids:
            return rv.get_move_group(mid)
    raise RuntimeError("no arm move group found")


def set_arm_qpos(eng, qpos7) -> None:
    """Set the 7 arm joints to `qpos7` (radians) and run forward kinematics (no physics step)."""
    import mujoco

    rv = eng.task._env.robots[0].robot_view
    mg = get_arm_movegroup(rv)
    mg.joint_pos = np.asarray(qpos7, np.float32)[:GRIP]
    mujoco.mj_forward(eng.task._env.mj_model, eng.task._env.mj_datas[0])


def arm_joint_limits(eng) -> np.ndarray:
    """(7, 2) [lo, hi] limits (radians) of the arm joints, read from the sim model
    (MoveGroup.joint_pos_limits, i.e. mj_model.jnt_range of the group's joints)."""
    rv = eng.task._env.robots[0].robot_view
    lim = np.asarray(get_arm_movegroup(rv).joint_pos_limits, np.float64).reshape(-1, 2)
    if lim.shape[0] < GRIP or not np.all(np.isfinite(lim[:GRIP])) or np.any(lim[:GRIP, 1] <= lim[:GRIP, 0]):
        raise RuntimeError(f"arm joint limits unavailable or invalid: {lim.tolist()}")
    return lim[:GRIP]


def pick_poses(baseline: dict, n_pose: int) -> list[int]:
    """Up to `n_pose` query indices of a `BranchEngine.run_baseline` result where the hand is open
    (executed gripper < 127) and the arm config and Jacobian were recorded, excluding the first and
    last query, spread evenly over the approach."""
    grip = list(baseline["grip_series"])
    qarm, ejac = baseline["qpos_arm"], baseline["ee_jac"]
    qsteps = baseline["query_steps"]
    cand = [c for c in range(len(grip))
            if grip[c] < 127.0 and c < len(qarm) and qarm[c] is not None
            and c < len(ejac) and ejac[c] is not None and c < len(qsteps)]
    cand = [c for c in cand if 1 <= c <= len(grip) - 2]
    if len(cand) <= n_pose:
        return cand
    idx = np.linspace(0, len(cand) - 1, n_pose).round().astype(int)
    return [cand[i] for i in sorted(set(idx.tolist()))]


class GripperControl:
    """Set the finger joints of the gripper to their open or closed limit (qpos edit + mj_forward).

    The finger joints are the slide or hinge joints whose names contain "finger" (or "gripper" with
    a "/"). Which limit is open is found once per episode, on first use, by setting both limits and
    comparing the distance between the left and right finger pads (`calibration`); `reset()` forgets
    it (call it after sampling a new episode)."""

    def __init__(self, eng):
        self.eng = eng
        self._cal = None

    def reset(self):
        self._cal = None

    def _mj(self):
        env = self.eng.task._env
        return env.mj_model, env.mj_datas[0]

    def finger_dofs(self):
        """[(qpos address, lo, hi)] of the finger joints."""
        import mujoco
        model, data = self._mj()
        dofs = []
        for j in range(model.njnt):
            nm = (model.joint(j).name or "").lower()
            if ("finger" in nm) or ("gripper" in nm and "/" in nm):
                if int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_SLIDE),
                                              int(mujoco.mjtJoint.mjJNT_HINGE)):
                    qadr = int(model.jnt_qposadr[j])
                    lo, hi = [float(x) for x in model.jnt_range[j]]
                    if hi > lo:
                        dofs.append((qadr, lo, hi))
        if dofs:
            return dofs
        # no named finger joints: move the gripper move group and see which qpos entries change
        rv = self.eng.task._env.robots[0].robot_view
        mg = rv.get_move_group(rv.get_gripper_movegroup_ids()[0])
        cur = np.asarray(mg.joint_pos, float).copy()
        q0 = np.array(data.qpos, float)
        mg.joint_pos = cur + 0.005
        mujoco.mj_forward(model, data)
        q1 = np.array(data.qpos, float)
        mg.joint_pos = cur
        mujoco.mj_forward(model, data)
        qadr_to_jnt = {int(model.jnt_qposadr[j]): j for j in range(model.njnt)}
        for a in np.where(np.abs(q1 - q0) > 1e-9)[0].tolist():
            if a in qadr_to_jnt:
                j = qadr_to_jnt[a]
                lo, hi = [float(x) for x in model.jnt_range[j]]
                if hi > lo:
                    dofs.append((int(a), lo, hi))
        return dofs

    def _pad_geoms(self):
        model, _ = self._mj()
        left, right = [], []
        for g in range(model.ngeom):
            nm = (model.geom(g).name or "").lower()
            if nm.startswith("robot_0/gripper/") and any(t in nm for t in ("pad", "follower", "finger")):
                (left if "left" in nm else right if "right" in nm else left).append(g)
        return left, right

    def finger_separation(self) -> float:
        """Distance (m) between the centroids of the left and right finger geoms; nan if not found."""
        model, data = self._mj()
        left, right = self._pad_geoms()
        if not left or not right:
            gids = left + right
            if len(gids) < 2:
                return float("nan")
            pts = np.array([data.geom_xpos[g] for g in gids])
            return float(np.linalg.norm(pts.max(0) - pts.min(0)))
        cl = np.mean([data.geom_xpos[g] for g in left], 0)
        cr = np.mean([data.geom_xpos[g] for g in right], 0)
        return float(np.linalg.norm(cl - cr))

    def calibration(self) -> dict:
        """{"open": (G,), "closed": (G,), "sep_open", "sep_closed", "addrs"}: the open limit is the
        one with the larger finger separation. qpos is put back afterwards."""
        import mujoco
        if self._cal is not None:
            return self._cal
        model, data = self._mj()
        dofs = self.finger_dofs()
        if not dofs:
            raise RuntimeError("no gripper finger joints found")
        los = np.array([lo for (_, lo, _) in dofs])
        his = np.array([hi for (_, _, hi) in dofs])
        addrs = [a for (a, _, _) in dofs]
        saved = np.array(data.qpos, float)

        def _set(vals):
            for a, v in zip(addrs, vals):
                data.qpos[a] = v
            mujoco.mj_forward(model, data)

        _set(his)
        sep_hi = self.finger_separation()
        _set(los)
        sep_lo = self.finger_separation()
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        if np.isfinite(sep_hi) and np.isfinite(sep_lo) and sep_lo > sep_hi:
            op, cl, so, sc = los, his, sep_lo, sep_hi
        else:
            op, cl, so, sc = his, los, sep_hi, sep_lo
        self._cal = {"open": op, "closed": cl, "sep_open": float(so), "sep_closed": float(sc),
                     "addrs": addrs}
        return self._cal

    def set(self, which: str):
        """Set the finger joints to the "open" or "closed" limit, then mj_forward."""
        import mujoco
        model, data = self._mj()
        cal = self.calibration()
        for a, v in zip(cal["addrs"], cal[which]):
            data.qpos[a] = v
        mujoco.mj_forward(model, data)
