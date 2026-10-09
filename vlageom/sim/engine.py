"""BranchEngine: one MolmoBot-DROID episode at a time, with per-query model context capture,
per-step snapshots and direct access to the action expert's flow loop.

What it provides:

  * `sample(house, seed)` builds the episode (scene, task, fresh policy), seeds python / numpy /
    torch with `seed`, and installs query-frame rendering
    (`vlageom.sim.render.install_query_frame_rendering`). `new_env()` replaces the MolmoSpaces
    environment and its task sampler by new ones, so the next `sample` creates a new task sampler.
  * `run_baseline(max_steps)` rolls out the policy. At every model query it records the
    `generate_actions` inputs (`self._captured[q]`), the executed gripper command, grasp contact,
    object height, arm joint angles, EE pose and Jacobians; it snapshots sim + policy state at
    every step (`self._baseline_snaps`), so later work restores a state instead of re-simulating
    the prefix (bf16 inference is not bit-reproducible across runs).
  * Flow loop access. MolmoBot's action expert integrates a 10-step Euler flow. For a captured
    context, `_build_backbone_cache` runs the VLM backbone once, `_compute_prefix` runs flow steps
    0..8 (flow noise seeded with `scene_seed + 1000 * query`, the same seed the rollout used), and
    `_final_step` runs step 9 with an optional vector added to the output of
    `action_expert.blocks.{layer}` and returns the action chunk plus that layer's activation (mean
    over the 16 action tokens, or per token).
  * `fit_deltas` fits the gripper edit at a query (see `vlageom.steer.delta.optimize_delta`),
    `fit_deltas_basis` the edit restricted to the span of a basis
    (`vlageom.steer.delta.optimize_delta_basis`); `save_capture` / `load_capture` store the
    per-query contexts so fits can be repeated without rendering.
  * Readers: `read_ee` / `read_ee_full` (EE pose in the robot base frame), `read_object` (pickup
    object world position), `_read_grasp`, `_read_ee_jacobian`.
"""
from __future__ import annotations

import copy
import logging
import random

import numpy as np

from vlageom import paths
from vlageom.steer.delta import GRIP

log = logging.getLogger(__name__)


def seed_all(seed: int) -> None:
    """Seed python `random`, numpy's global RNG and torch (CPU + CUDA) with `seed`."""
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gripper_of(action: dict) -> float:
    """The gripper command of a policy action dict (0 = open ... 255 = closed)."""
    g = action.get("gripper", 0.0)
    if isinstance(g, (list, tuple, np.ndarray)):
        return float(np.asarray(g).reshape(-1)[0])
    return float(g)


class BranchEngine:
    """See the module docstring. `config` keys: "config" (MolmoSpaces config name), "checkpoint"
    (default `vlageom.paths.checkpoint()`), "output_dir" (MolmoSpaces output dir, optional)."""

    _CAPTURE_BASELINE_KEYS = ("n_steps", "n_queries", "query_steps", "grip_series", "grasp_q",
                              "objz_q", "qpos_arm", "ee_jac", "ee_pos", "base_R", "success")

    def __init__(self, config: dict):
        self.config = config
        self.sim_env = self.task = self.policy = self.model = None
        self.device = None
        self.num_steps = 10
        self.scene_seed = None
        self.house = None
        self._gen_orig = None
        self._gen_unwrapped = None
        self.site_width = 768

        self._chunk = 0
        self._captured: list = []          # per-query generate_actions kwargs
        self._cap_grip: list = []
        self._cap_arm: list = []
        self._baseline_snaps: list = []    # per-step snapshots
        self._baseline_query_steps: list = []
        self._baseline_grip_series: list = []

    # ------------------------------------------------------------------------------------------ #
    def load(self):
        """Build the MolmoSpaces env with the MolmoBot-DROID policy."""
        from vlageom.sim.env import build_molmobot_env
        self.sim_env = build_molmobot_env(
            self.config["config"], self.config.get("checkpoint") or paths.checkpoint(),
            output_dir=self.config.get("output_dir"))

    def new_env(self):
        """Close the MolmoSpaces environment and build a new one from the config (`load`). The next
        `sample` then creates a new task sampler (see `SimEnv.ensure_task_sampler`)."""
        if self.sim_env is not None:
            self.sim_env.close()
        self.load()

    def sample(self, house: int, seed: int, reload_house: bool = False):
        """Sample the episode for `house` with all RNGs seeded by `seed`; install query-frame
        rendering. `reload_house=True` loads the house from file even when it is the house already
        loaded (`SimEnv.sample_episode`)."""
        self.house = int(house)
        self.scene_seed = int(seed)
        seed_all(self.scene_seed)
        self.task, self.policy = self.sim_env.sample_episode(house_index=int(house),
                                                             reload_house=reload_house)
        self.model = self.policy.agent.model
        self.device = next(self.model.parameters()).device
        self.num_steps = int(getattr(self.policy.agent, "num_flow_steps", 10) or 10)
        self._gen_orig = self.model.generate_actions
        self._gen_unwrapped = getattr(type(self.model).generate_actions, "__wrapped__", None)
        self._chunk = 0
        try:
            from vlageom.sim.render import install_query_frame_rendering
            install_query_frame_rendering(self.task, self.policy)
        except Exception as e:
            log.warning("query-frame rendering not installed (every step renders): %s", e)

    # ---- backbone cache + flow loop --------------------------------------------------------- #
    def _seeded_kw(self, kw, seed):
        """generate_actions kwargs with a torch.Generator seeded by `seed`; returns (kw, num_steps)."""
        import torch
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        kw2 = {k: v for k, v in kw.items() if k != "num_steps"}
        kw2["generator"] = g
        return kw2, (kw.get("num_steps") or self.num_steps)

    def _build_backbone_cache(self, kw):
        """Run the VLM backbone once on captured kwargs; returns the action expert's inputs."""
        import torch
        m = self.model
        states = kw.get("states")
        if states is None:
            raise RuntimeError("context has no `states`; cannot run flow loop")
        fk = {k: v for k, v in kw.items() if k not in ("states", "num_steps", "generator")}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eam = m._get_encoder_attention_mask(kw["input_ids"], kw.get("attention_mask"))
            _, layer_states = m._run_backbone(
                collect_layer_hidden_states=True, output_hidden_states=False, **fk)
            layer_states = m._select_layer_states(layer_states)
            states_ad = m.adapt_state_based_on_mode(states)
        return {"layer_states": layer_states, "eam": eam, "states": states_ad,
                "steps": self.num_steps, "ah": int(m.config.action_horizon),
                "ad": int(m.config.action_dim), "states_mode": m.config.states_mode}

    def _compute_prefix(self, cache, seed):
        """Flow steps 0..N-2 (no edit); returns the trajectory entering the final step."""
        import torch
        c = cache
        B = c["layer_states"][0].shape[0]
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        traj = torch.randn((B, c["ah"], c["ad"]), device=self.device, generator=g)
        dt = 1.0 / c["steps"]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for i in range(c["steps"] - 1):
                tt = torch.full((B,), i / c["steps"], device=self.device)
                vel = self.model.action_expert(traj, tt, c["layer_states"],
                                               encoder_attention_mask=c["eam"],
                                               state_embeddings=c["states"], states_mode=c["states_mode"])
                traj = traj + dt * vel
        return traj

    def _final_step(self, cache, prefix, delta, layer, grad, per_token=False):
        """The final Euler step, with `delta` (or None) added to the output of
        action_expert.blocks.{layer}. Returns (action chunk (B, H, 8), activation at that layer
        after the edit): mean over the action tokens (768,), or (n_token, 768) with per_token=True."""
        import torch
        c = cache
        B = c["layer_states"][0].shape[0]
        dt = 1.0 / c["steps"]
        i = c["steps"] - 1
        block = self.model.get_submodule(f"action_expert.blocks.{layer}")
        cap = {}

        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if delta is not None:
                t = t + delta.to(t.dtype)
            _flat = t.detach().float().reshape(-1, t.shape[-1])
            cap["act"] = _flat if per_token else _flat.mean(0)
            if delta is not None:
                return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
            return None

        h = block.register_forward_hook(hook)
        cm = torch.enable_grad() if grad else torch.no_grad()
        try:
            with cm, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tt = torch.full((B,), i / c["steps"], device=self.device)
                vel = self.model.action_expert(prefix, tt, c["layer_states"],
                                               encoder_attention_mask=c["eam"],
                                               state_embeddings=c["states"], states_mode=c["states_mode"])
                out = prefix + dt * vel
        finally:
            h.remove()
        return out, cap.get("act")

    # ---- action normalizer ------------------------------------------------------------------ #
    def arm_action_normalizer(self):
        """The policy's action normalizer (quantile q01/q99 stats), or None. Actions and the
        proprioceptive state use different normalizers, so an arm-action target built from joint
        angles must use this one."""
        try:
            agent = self.policy.agent
            post = agent.action_postprocessor
            repo = getattr(agent, "norm_repo_id", None)
            return post._get_normalizer(post.action_normalizers, repo)
        except Exception as e:
            log.warning("no action normalizer available: %s", e)
            return None

    def freeze_arm_target(self, qpos_arm):
        """Arm joint angles (radians) -> the normalized arm action (GRIP,) that commands exactly that
        configuration. Repeated over the action horizon it means "do not move the arm". None if the
        normalizer or the angles are unavailable."""
        norm = self.arm_action_normalizer()
        if norm is None or qpos_arm is None:
            return None
        lo = getattr(norm, "q_low", None)
        if lo is None:
            lo = getattr(norm, "min_val", None)
        if lo is None:
            return None
        width = int(np.asarray(lo).reshape(-1).shape[0])
        full = np.zeros(max(width, GRIP), np.float32)
        full[:GRIP] = np.asarray(qpos_arm, np.float32).reshape(-1)[:GRIP]
        out = np.asarray(norm.normalize(full[:width]), np.float32).reshape(-1)
        return out[:GRIP].astype(np.float32)

    def arm_action_bounds(self):
        """(lo, hi) normalizer bounds of the GRIP arm channels, each (GRIP,): q01/q99 for quantile
        normalizers, min/max otherwise. A normalized action a in [-1, 1] maps to radians as
        lo + (a + 1) / 2 * (hi - lo). None if unavailable."""
        norm = self.arm_action_normalizer()
        if norm is None:
            return None
        lo = getattr(norm, "q_low", None)
        hi = getattr(norm, "q_high", None)
        if lo is None or hi is None:
            lo = getattr(norm, "min_val", None)
            hi = getattr(norm, "max_val", None)
        if lo is None or hi is None:
            return None
        lo = np.asarray(lo, np.float32).reshape(-1)[:GRIP]
        hi = np.asarray(hi, np.float32).reshape(-1)[:GRIP]
        return lo, hi

    # ---- Jacobians -------------------------------------------------------------------------- #
    def _read_ee_jacobian(self):
        """(J3 (3, GRIP) translational EE Jacobian in the base frame, R (3,3) world<-base), or
        (None, None)."""
        try:
            rv = self.task._env.robots[0].robot_view
            mg = rv.get_gripper_movegroup_ids()[0]
            J = np.asarray(rv.get_jacobian(mg, ["arm"]), np.float32)
            J3w = J[:3, :GRIP]
            R = np.asarray(rv.base.pose, np.float32)[:3, :3]
            J3b = (R.T @ J3w).astype(np.float32)
            return J3b, R.astype(np.float32)
        except Exception as e:
            log.warning("no EE jacobian available: %s", e)
            return None, None

    # ---- gripper edit fit ------------------------------------------------------------------- #
    def fit_deltas(self, chunk_idx: int, target_sign: float, layers: list[int],
                   *, iters=120, lr=0.1, lam=1e-4,
                   arm_target=None, arm_weight=1.0, grip_weight=1.0, smooth_weight=0.0,
                   n_target_steps=None) -> dict:
        """Fit the gripper edit at each layer in `layers` for the captured context of query
        `chunk_idx` (shared backbone cache and flow prefix). See `optimize_delta` for the loss.

        Returns {layer: {"fit": DeltaFit, "base_act": (768,), "steered_act": (768,)}}: the layer's
        token-mean activation at the final flow step without and with the fitted edit."""
        import torch
        from vlageom.steer.delta import optimize_delta
        kw = self._captured[chunk_idx]
        cache = self._build_backbone_cache(kw)
        seed = self.scene_seed + 1000 * chunk_idx
        prefix = self._compute_prefix(cache, seed)
        d = self.site_width
        out: dict = {}
        for L in layers:
            def final_step_fn(delta, L=L):
                return self._final_step(cache, prefix, delta, L,
                                        grad=(delta is not None and delta.requires_grad))[0]

            def new_delta_fn():
                return torch.zeros(d, device=self.device, requires_grad=True)

            fit = optimize_delta(final_step_fn, new_delta_fn, target_sign=target_sign,
                                 iters=iters, lr=lr, lam=lam, n_target_steps=n_target_steps,
                                 arm_target=arm_target, arm_weight=arm_weight,
                                 grip_weight=grip_weight, smooth_weight=smooth_weight)
            _, base_act = self._final_step(cache, prefix, None, L, grad=False)
            _, steered_act = self._final_step(
                cache, prefix, torch.tensor(fit.delta, device=self.device), L, grad=False)
            out[L] = {"fit": fit, "base_act": base_act.cpu().numpy(),
                      "steered_act": steered_act.cpu().numpy()}
        return out

    def fit_deltas_basis(self, chunk_idx, target_sign, layer, basis, mean, k_list, *,
                         arm_target=None, arm_weight=300.0, grip_weight=1.0, smooth_weight=3000.0,
                         iters=1000, lr=0.1, lam=1e-4, init_coeffs_fn=None) -> dict:
        """Low-rank edit fits at query `chunk_idx` of the captured contexts, one per k in `k_list`
        (`vlageom.steer.delta.optimize_delta_basis`: delta = mean + basis[:, :k] @ a). The
        backbone and flow steps 0..8 run once (flow noise seeded with scene_seed + 1000 * query,
        as in the rollout); each fit re-runs the final flow step only. `init_coeffs_fn(k,
        target_sign)` gives the start coefficients.
        Returns {k: {"fit": DeltaFit, "coeffs": numpy (k,), "k_eff": int, "loss": float}}."""
        from vlageom.steer.delta import optimize_delta_basis
        kw = self._captured[chunk_idx]
        cache = self._build_backbone_cache(kw)
        prefix = self._compute_prefix(cache, self.scene_seed + 1000 * chunk_idx)

        def final_step_fn(delta):
            return self._final_step(cache, prefix, delta, layer,
                                    grad=(delta is not None and delta.requires_grad))[0]

        out: dict = {}
        for k in k_list:
            init = init_coeffs_fn(int(k), target_sign) if init_coeffs_fn is not None else None
            fit, coeffs, meta = optimize_delta_basis(
                final_step_fn, basis=basis, mean=mean, n_coeffs=int(k), target_sign=target_sign,
                init_coeffs=init, iters=iters, lr=lr, lam=lam, arm_target=arm_target,
                arm_weight=arm_weight, grip_weight=grip_weight, smooth_weight=smooth_weight)
            out[int(k)] = {"fit": fit, "coeffs": coeffs, "k_eff": meta["k_eff"], "loss": meta["loss"]}
        return out

    # ---- baseline rollout ------------------------------------------------------------------- #
    def _capture_gen(self, **kw):
        """generate_actions replacement for run_baseline: seeds the flow noise with
        scene_seed + 1000 * query and stores the kwargs."""
        import torch
        ci = self._chunk
        self._chunk += 1
        seed = self.scene_seed + 1000 * ci
        kw2, steps = self._seeded_kw(kw, seed)
        self._captured.append({k: (v.detach().clone() if torch.is_tensor(v) else v)
                               for k, v in kw2.items() if k != "generator"})
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self._gen_orig(num_steps=steps, **kw2)
        self._cap_grip.append(float(out[0, :, GRIP].float().mean()))
        self._cap_arm.append(out[0, :, :GRIP].float().mean(0).detach().cpu().numpy())
        return out

    def _snapshot(self, obs):
        """Sim state (qpos, qvel, act, ctrl, mocap, time), policy buffers and task counters, plus a
        copy of `obs`."""
        env = self.task._env
        datas = []
        for d in env.mj_datas:
            snap = {"qpos": np.array(d.qpos), "qvel": np.array(d.qvel),
                    "act": np.array(d.act) if d.act.size else None,
                    "time": float(d.time), "ctrl": np.array(d.ctrl) if d.ctrl.size else None}
            for fld in ("mocap_pos", "mocap_quat"):
                arr = getattr(d, fld, None)
                snap[fld] = np.array(arr) if (arr is not None and np.asarray(arr).size) else None
            datas.append(snap)
        p = self.policy
        pol = {"action_buffer": list(getattr(p, "action_buffer", []) or []),
               "buffer_index": int(getattr(p, "buffer_index", 0)),
               "step_count": int(getattr(p, "step_count", 0)),
               "obs_history": list(getattr(p, "obs_history", []) or [])}
        tk = self.task
        taskstate = {k: getattr(tk, k) for k in
                     ("episode_step_count", "_num_steps_taken", "_cumulative_reward")
                     if hasattr(tk, k)}
        return {"datas": datas, "policy": pol, "task": taskstate,
                "chunk": self._chunk, "obs": copy.deepcopy(obs)}

    def _restore(self, snap):
        """Restore a `_snapshot` (followed by mj_forward); returns a copy of its obs. Robot-mounted
        cameras are not moved here: render with `vlageom.sim.render.fresh_obs`."""
        import mujoco
        env = self.task._env
        for d, s in zip(env.mj_datas, snap["datas"]):
            d.qpos[:] = s["qpos"]
            d.qvel[:] = s["qvel"]
            if s["act"] is not None and d.act.size:
                d.act[:] = s["act"]
            if s["ctrl"] is not None and d.ctrl.size:
                d.ctrl[:] = s["ctrl"]
            for fld in ("mocap_pos", "mocap_quat"):
                if s.get(fld) is not None:
                    getattr(d, fld)[:] = s[fld]
            d.time = s["time"]
            mujoco.mj_forward(env.mj_model, d)
        p = self.policy
        p.action_buffer = list(snap["policy"]["action_buffer"])
        p.buffer_index = snap["policy"]["buffer_index"]
        p.step_count = snap["policy"]["step_count"]
        p.obs_history = list(snap["policy"]["obs_history"])
        for k, v in snap["task"].items():
            setattr(self.task, k, v)
        self._chunk = snap["chunk"]
        return copy.deepcopy(snap["obs"])

    def run_baseline(self, max_steps: int) -> dict:
        """Roll out the policy from `task.reset()` for at most `max_steps` steps, snapshotting every
        step and recording per query: step index, executed gripper command, grasp contact, object
        height, arm joint angles, EE Jacobian, EE position and base rotation.

        Returns dict(n_steps, n_queries, query_steps, grip_series, grasp_q, objz_q, qpos_arm,
        ee_jac, ee_pos, base_R, frames={}, success)."""
        seed_all(self.scene_seed)
        self.policy.reset()
        self._chunk = 0
        self._captured, self._cap_grip, self._cap_arm = [], [], []
        self._baseline_snaps, self._baseline_query_steps, self._baseline_grip_series = [], [], []
        self._baseline_qpos_arm = []
        self._baseline_ee_jac, self._baseline_ee_pos, self._baseline_base_R = [], [], []
        grasp_q, objz_q = [], []
        orig = self.model.generate_actions
        self.model.generate_actions = self._capture_gen
        try:
            obs, _ = self.task.reset()
            step = 0
            while not self.task.is_done() and step < max_steps:
                self._baseline_snaps.append(self._snapshot(obs))
                pre = self._chunk
                action = self.policy.get_action(obs)
                if action is None:
                    break
                if self._chunk > pre:                       # a model query fired this step
                    self._baseline_query_steps.append(step)
                    self._baseline_grip_series.append(gripper_of(action))
                    grasp_q.append(self._read_grasp())
                    obj = self.read_object()
                    objz_q.append(float(obj[2]) if len(obj) == 3 else float("nan"))
                    self._baseline_qpos_arm.append(self._obs_arm_qpos(obs))
                    J3b, baseR = self._read_ee_jacobian()
                    self._baseline_ee_jac.append(J3b)
                    self._baseline_base_R.append(baseR)
                    ee = self.read_ee()
                    self._baseline_ee_pos.append(np.asarray(ee, np.float32) if ee else None)
                obs, *_ = self.task.step(action)
                step += 1
        finally:
            self.model.generate_actions = orig
        return {"n_steps": step, "n_queries": self._chunk,
                "query_steps": list(self._baseline_query_steps),
                "grip_series": list(self._baseline_grip_series),
                "grasp_q": grasp_q, "objz_q": objz_q,
                "qpos_arm": list(self._baseline_qpos_arm),
                "ee_jac": list(self._baseline_ee_jac), "ee_pos": list(self._baseline_ee_pos),
                "base_R": list(self._baseline_base_R), "frames": {},
                "success": bool(self.task.judge_success()) if hasattr(self.task, "judge_success") else False}

    # ---- stored contexts (fits without rendering) ------------------------------------------- #
    def save_capture(self, path, baseline: dict):
        """Save the per-query generate_actions contexts and the baseline's per-query signals (the
        inputs of the branch-query selection) to one torch file. Snapshots are not saved."""
        import torch
        from pathlib import Path as _P
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        captured_cpu = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in kw.items()}
                        for kw in self._captured]
        base_meta = {k: baseline[k] for k in self._CAPTURE_BASELINE_KEYS if k in baseline}
        payload = {
            "format": 1,
            "captured": captured_cpu,
            "scene_seed": int(self.scene_seed),
            "house": int(self.house) if self.house is not None else -1,
            "num_steps": int(self.num_steps),
            "site_width": int(self.site_width),
            "baseline": base_meta,
        }
        torch.save(payload, str(path))
        return path

    def load_capture(self, path) -> dict:
        """Load a `save_capture` file into this engine (after `load()` + `sample()`, which set up the
        model). Returns the baseline-signal dict, shaped like a `run_baseline` result."""
        import torch
        if self.device is None:
            raise RuntimeError("load_capture needs a loaded model — call load()+sample() first")
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        self._captured = [{k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in kw.items()}
                          for kw in payload["captured"]]
        self.scene_seed = int(payload["scene_seed"])
        self.house = int(payload.get("house", -1))
        self.num_steps = int(payload.get("num_steps", self.num_steps))
        self.site_width = int(payload.get("site_width", self.site_width))
        self._chunk = len(self._captured)
        base = dict(payload.get("baseline", {}))
        self._baseline_query_steps = list(base.get("query_steps", []))
        self._baseline_grip_series = list(base.get("grip_series", []))
        self._baseline_qpos_arm = list(base.get("qpos_arm", []))
        base.setdefault("frames", {})
        return base

    # ---- readers ---------------------------------------------------------------------------- #
    @staticmethod
    def _obs_arm_qpos(obs):
        """Arm joint angles (radians, (GRIP,)) from a policy observation (`obs["qpos"]["arm"]`), or
        None."""
        try:
            o = obs[0] if isinstance(obs, (list, tuple)) else obs
            return np.asarray(o["qpos"]["arm"], np.float32).reshape(-1)[:GRIP]
        except Exception:
            return None

    def _read_grasp(self) -> bool:
        """True iff a gripper geom is in positive-force contact with the pickup object."""
        from vlageom.sim.scene import get_pickup_body_name
        try:
            env = self.task._env
            data = env.mj_datas[0]
            model = env.mj_model
            name = get_pickup_body_name(self.task)
            if not name:
                return False
            pickup_body_id = int(data.body(name).id)
            rootid = int(model.body(pickup_body_id).rootid.item())
            geom_bodyid = np.asarray(model.geom_bodyid)
            if rootid == 0:
                pickup_geoms = set(np.where(geom_bodyid == pickup_body_id)[0].tolist())
            else:
                geom_roots = np.asarray(model.body_rootid)[geom_bodyid]
                pickup_geoms = set(np.where(geom_roots == rootid)[0].tolist())

            def gname(gid):
                n = model.geom(gid).name
                return n or (model.body(int(model.geom_bodyid[gid])).name or "")

            for i in range(int(data.ncon)):
                c = data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                in1, in2 = g1 in pickup_geoms, g2 in pickup_geoms
                if not (in1 or in2):
                    continue
                other = gname(g2) if in1 else gname(g1)
                f = 0.0
                try:
                    if int(c.efc_address) >= 0:
                        f = float(data.efc_force[int(c.efc_address)])
                except Exception:
                    pass
                if "gripper" in other.lower() and f > 0:
                    return True
            return False
        except Exception:
            return False

    def _render_real(self, camera_name: str):
        """One direct render of `camera_name`; see `vlageom.sim.render.render_real`."""
        from vlageom.sim.render import render_real
        return render_real(self, camera_name)

    def read_ee(self):
        """EE position (3,) in the robot base frame as a list, or [] if unavailable."""
        try:
            rv = self.task._env.robots[0].robot_view
            mg = rv.get_gripper_movegroup_ids()[0]
            m = np.asarray(rv.get_move_group(mg).leaf_frame_to_robot)
            return m[:3, 3].tolist()
        except Exception:
            return []

    def read_ee_full(self):
        """EE pose (4, 4) float32 in the robot base frame (`m[:3, 3]` == read_ee()), or None."""
        try:
            rv = self.task._env.robots[0].robot_view
            mg = rv.get_gripper_movegroup_ids()[0]
            return np.asarray(rv.get_move_group(mg).leaf_frame_to_robot, np.float32)
        except Exception:
            return None

    def read_object(self):
        """Pickup object world position (3,) as a list, or [] if unavailable."""
        from vlageom.sim.scene import get_pickup_body_name
        try:
            data = self.task._env.mj_datas[0]
            name = get_pickup_body_name(self.task)
            return np.asarray(data.body(name).xpos, dtype=float).tolist()
        except Exception:
            return []

    def close(self):
        try:
            if self.sim_env is not None:
                self.sim_env.close()
        except Exception:
            pass
