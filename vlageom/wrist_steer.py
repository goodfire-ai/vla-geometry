"""Wrist orientation figure (Figure 6): the wrist roll sweep on its own principal components
(panel a), a circular autoencoder chart of the wrist orientation code (panel b), and the response of
the policy to edits written through that chart (panels c and d).

Data, in pipeline order (under `data_root()/wrist_steer/` unless noted):

  natural rollouts    GPU + EGL. The 200 natural MolmoBot-DROID rollouts of
                      lists/wrist_steer/natural_episodes.csv, one process per partition of 16
                      episodes (`NaturalRecorder`, `run_natural_partition`). Per model query: the
                      output of action-expert blocks 0, 4, 9, 16, 24, 32 at the final flow step,
                      rounded to bfloat16 per token and reduced to action token 0 and to the mean
                      over the 16 action tokens; the predicted action chunk, EE pose, EE Jacobian,
                      object position and robot joint velocities; per env step the EE pose.
  natural cloud       CPU. Every query of the rollouts with at least 5 queries (`natural_cache`),
                      then the token-mean layer-9 activations with their EE rotation
                      (`natural_cloud`): the points of panel b.
  orientation sweeps  GPU + EGL. For 200 episodes (lists/wrist_steer/orient_episodes.csv): roll out
                      the policy, take 4 open-hand queries, and turn the hand about the base x, y
                      and z axes in 10-degree steps (up to 18 each way) while holding the EE
                      position (`step_orient`), rendering both cameras and reading the action
                      expert at every step (`harvest_orient_episode`).
  operator            CPU. The held-out house split (a third of the sweep houses) and a linear
                      rotation operator in a 64-d PCA of the sweeps (`fit_operator`); the steering
                      stage uses it for its random-generator control rows.
  chart               GPU. The circular autoencoder (`train_chart`): 768-512-256-32 encoder whose
                      code dimensions (0,1), (2,3), (4,5) are projected onto circles of learned
                      radius (rotation about base x, y, z), and a 32-256-512-768 decoder, trained on
                      layer-9 token-mean activations of the training houses' sweeps.
  steering            GPU + EGL. Chart edits in held-out scenes (lists/wrist_steer/steer_episodes.csv):
                      one-shot edits at one query and held edits that are recomputed at every
                      query of a 32-step rollout (`steer_episode`, `persistent_chart_continuation`).
  panels, numbers     CPU. Panel arrays (`panel_a`, `panel_bcd`) and the numbers quoted in the text
                      (`fig6_numbers`).

numpy only at import; torch, the simulator and plotting are imported inside the functions that
need them.
"""
# The chart and steering code is adapted from Mathilde Papillon's code.
from __future__ import annotations

import csv
import glob
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from vlageom import paths, util

log = logging.getLogger(__name__)

TILE = "wrist_steer"
CONFIG = "FrankaPickDroidDataGenConfig"     # MolmoSpaces config of every rollout
GRIP = 7                                    # arm joints; index 7 of an action is the gripper
LAYER = 9                                   # action-expert layer of the chart and the edits

# Natural rollouts.
NATURAL_LAYERS = [0, 4, 9, 16, 24, 32]      # action-expert blocks kept per query
NATURAL_PASS = 9                            # flow step read (0-based; the 10th and last)
NATURAL_PER_PARTITION = 16                  # episodes per partition (one process each)
NATURAL_N_HOUSES = 200                      # episode i is meant for house i % 200
NATURAL_MAX_ATTEMPTS = 5                    # sampling attempts per episode
NATURAL_RETRY_STRIDE = 3                    # attempt a (1..3) samples house i + 3a
NATURAL_MIN_QUERIES = 5                     # rollouts with fewer queries are not in the cloud
H_CHUNK, ADIM = 16, 8                       # action chunk: 16 steps x (7 arm + gripper)

# Orientation sweeps.
ORIENT_LAYERS = [0, 4, 8, 9, 12, 16, 20, 24, 28, 32, 35]
ORIENT_MAX_STEPS = 600                      # rollout length limit
ORIENT_N_POSE = 4                           # open-hand queries per episode
ORIENT_STEP_DEG = 10.0                      # rotation per step (deg)
ORIENT_K_HALF = 18                          # steps each way per axis
ORIENT_MAX_POS_DRIFT = 0.03                 # m, EE position tolerance of a sweep
MIN_PX = 3.0                                # minimum mean pixel value of a rendered camera frame
AXIS_DIRS = [("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0)), ("z", (0.0, 0.0, 1.0))]
AXIS_LABELS = [a for a, _ in AXIS_DIRS]
AXIS_U = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
AXIS_ID = {"x": 0, "y": 1, "z": 2}

# Held-out split, operator and chart.
SPLIT_SEED = 0                              # numpy default_rng seed of the house shuffle
PCA_K = 64                                  # operator PCA dimension
RIDGE = 1.0                                 # operator ridge
# LAPACK's blocked SVD splits its work by the number of BLAS threads, so the last bits of the
# operator depend on that number. The paper's operator was fitted with 24 OpenBLAS threads;
# `fit_operator` runs with the same number.
OPERATOR_BLAS_THREADS = 24
PLANE = {0: (0, 1), 1: (2, 3), 2: (4, 5)}   # chart code pair of each rotation axis
CHART_ARGS = dict(z_dim=32, steps=8000, batch=512, lr=1e-3, w_recon=1.0, w_latent=1.0,
                  w_decode=2.0, max_pair_deg=340.0, seed=0)

# Steering.
STEER_MAX_STEPS = 140                       # rollout length limit of the steering episodes
STEER_THETAS = [20.0, 40.0]                 # written angles (deg), both signs
STEER_N_EXEC = 8                            # executed steps per action chunk
PERSIST_STEPS = 32                          # steps of a held-edit rollout

# Panels.
# Panel c leaves out the scenes whose unedited chunk turns the wrist by more than DRIFT_MAX_DEG.
DRIFT_MAX_DEG = 10.0
HOUSE_D = 10          # house of panel d
CHUNK_STEPS = 8       # panel c: per-step rotation x 8 executed steps
D_THETAS = [-40.0, -20.0, 20.0, 40.0]   # panel d: written angles (with 0, the unedited rollout)


# ---- shipped lists ----------------------------------------------------------------------------- #
def _read_csv(name, path=None) -> list[dict]:
    path = Path(path) if path is not None else paths.lists_dir() / TILE / name
    with open(path, newline="") as f:
        return [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]


def read_natural_episodes(path=None) -> list[dict]:
    """lists/wrist_steer/natural_episodes.csv: episode_index, partition, house, seed. The row order
    is the row order of the natural cloud."""
    return _read_csv("natural_episodes.csv", path)


def read_orient_episodes(path=None) -> list[dict]:
    """lists/wrist_steer/orient_episodes.csv: task, episode_index, house, seed."""
    return _read_csv("orient_episodes.csv", path)


def read_steer_episodes(path=None) -> list[dict]:
    """lists/wrist_steer/steer_episodes.csv: episode_index, house, seed, in run order."""
    return _read_csv("steer_episodes.csv", path)


def read_heldout_houses(path=None) -> list[int]:
    """lists/wrist_steer/heldout_houses.csv: the 34 houses held out of the chart and operator fits."""
    return [r["house"] for r in _read_csv("heldout_houses.csv", path)]


def rollout_houses() -> list[int]:
    """Every house a Figure 6 GPU stage samples: the natural rollouts (listed houses and the houses
    of their failed sampling attempts, `sampling_attempts`), the orientation sweeps and the steering
    scenes. Their scenes must be installed before the offline harvest stages run."""
    nat = read_natural_episodes()
    houses = {h for p in sorted({e["partition"] for e in nat})
              for att in sampling_attempts(nat, p) for h, _s in att}
    houses |= {e["house"] for e in read_orient_episodes()}
    houses |= {e["house"] for e in read_steer_episodes()}
    return sorted(houses)


def tile_path(*parts: str, create: bool = False) -> Path:
    """`data_root()/wrist_steer/<parts...>`; with create=True the parent directory is made."""
    p = paths.tile_dir(TILE, *parts)
    if create:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


class DarkFrames(RuntimeError):
    """A rendered camera frame is missing or its mean pixel value is below MIN_PX (an EGL failure of
    the process)."""


# ---- natural rollouts (simulator) -------------------------------------------------------------- #
class NaturalRecorder:
    """Per-query capture of one natural rollout.

    `hook_model` registers forward hooks on action_expert.blocks.{L} for L in `layers`; each call of a
    block appends its output tensor (one call per flow step). `take_query`, called after
    `policy.get_action`, returns the outputs of flow step NATURAL_PASS of the query that just ran
    (or None when the action came from the stored chunk) and clears the buffers. The tensors are
    converted after `get_action` returns, then rounded to bfloat16.

    `attach` wraps the episode's policy so that after every model query it records the predicted
    action chunk, the robot joint velocities, the simulator time, the EE pose and Jacobian (robot
    base frame) and the pickup object position (`records`)."""

    def __init__(self, layers=NATURAL_LAYERS):
        self.layers = list(layers)
        self._handles = []
        self._model_id = None
        self._acts = {L: [] for L in self.layers}
        self._wrapped_agents: set[int] = set()
        self.records: list[dict] = []
        self._last_chunk = None
        self.qvel_groups: list[str] = []
        self.qvel_sizes: list[int] = []

    def hook_model(self, model) -> None:
        """Hook `model` (again when the policy built a new model object)."""
        if id(model) == self._model_id:
            return
        for h in self._handles:
            h.remove()
        self._handles = []
        for L in self.layers:
            mod = model.get_submodule(f"action_expert.blocks.{L}")
            self._handles.append(mod.register_forward_hook(self._make_hook(L)))
        self._model_id = id(model)

    def _make_hook(self, L):
        def hook(_m, _i, out):
            self._acts[L].append(out[0] if isinstance(out, tuple) else out)
        return hook

    def take_query(self):
        """{layer: (16, 768) float32} of flow step NATURAL_PASS, or None if the model did not run."""
        import torch
        if not any(self._acts[L] for L in self.layers):
            return None
        out = {}
        for L in self.layers:
            passes = self._acts[L]
            if len(passes) != NATURAL_PASS + 1:
                log.warning("block %d ran %d times in one query (expected %d)", L, len(passes),
                            NATURAL_PASS + 1)
            t = passes[NATURAL_PASS]
            arr = t.detach().cpu().to(torch.bfloat16).float().cpu().numpy()
            out[L] = arr.reshape(-1, arr.shape[-1])
        for L in self.layers:
            self._acts[L] = []
        return out

    def clear(self) -> None:
        for L in self.layers:
            self._acts[L] = []

    def attach(self, task, policy) -> None:
        from vlageom.sim.scene import get_pickup_body_name
        self.records = []
        self._last_chunk = None
        self.clear()
        agent = policy.agent
        if id(agent) not in self._wrapped_agents:
            orig_chunk = agent.get_action_chunk

            def wrapped_chunk(*a, **k):
                ch = orig_chunk(*a, **k)
                try:
                    self._last_chunk = np.asarray(ch, dtype=np.float32)
                except Exception:
                    self._last_chunk = None
                return ch

            agent.get_action_chunk = wrapped_chunk
            self._wrapped_agents.add(id(agent))
        env = task._env
        robot_view = env.robots[0].robot_view
        pickup_name = get_pickup_body_name(task)
        orig_pop = policy._populate_action_buffer

        def wrapped_pop(observation):
            orig_pop(observation)
            try:
                self._snapshot(env, robot_view, pickup_name)
            except Exception as e:
                log.warning("query record failed: %s", e)

        policy._populate_action_buffer = wrapped_pop

    def _snapshot(self, env, robot_view, pickup_name) -> None:
        data = env.mj_datas[0]
        gid = robot_view.get_gripper_movegroup_ids()[0]
        ee_pose = np.asarray(robot_view.get_move_group(gid).leaf_frame_to_robot, dtype=float)
        mgids = robot_view.move_group_ids()
        grip_ids = robot_view.get_gripper_movegroup_ids()
        inputs = ["arm"] if "arm" in mgids else [m for m in mgids if m not in grip_ids]
        try:
            ee_jac = np.asarray(robot_view.get_jacobian(gid, inputs), dtype=np.float32)
        except Exception:
            ee_jac = np.full((6, 7), np.nan, np.float32)
        obj_base = np.full(3, np.nan)
        ee_to_obj = np.full(3, np.nan)
        obj_dist = float("nan")
        try:
            base_pose = np.asarray(robot_view.base.pose, dtype=float)
            obj_world = np.asarray(data.body(pickup_name).xpos, dtype=float)
            obj_base = (np.linalg.inv(base_pose) @ np.append(obj_world, 1.0))[:3]
            ee_to_obj = obj_base - ee_pose[:3, 3]
            obj_dist = float(np.linalg.norm(ee_to_obj))
        except Exception:
            pass
        chunk = self._last_chunk
        if chunk is None:
            chunk = np.full((H_CHUNK, ADIM), np.nan, np.float32)
        try:
            qvd = robot_view.get_qvel_dict()
            groups = sorted(qvd.keys())
            qvel = np.concatenate(
                [np.asarray(qvd[g], dtype=np.float32).reshape(-1) for g in groups]
            ) if qvd else np.zeros(0, np.float32)
            self.qvel_groups = groups
            self.qvel_sizes = [int(np.asarray(qvd[g]).reshape(-1).size) for g in groups]
        except Exception:
            qvel = np.zeros(0, np.float32)
        self.records.append({
            "pred_chunk": chunk, "sim_time": float(getattr(data, "time", np.nan)), "qvel": qvel,
            "ee_pose": ee_pose.astype(np.float32), "ee_jac": ee_jac,
            "obj_pos_base": obj_base.astype(np.float32), "ee_to_obj": ee_to_obj.astype(np.float32),
            "obj_dist": obj_dist})


def _step_ee_pose(task):
    """Pre-step EE pose (4, 4) in the robot base frame as a float64 array (NaN if unavailable),
    read together with the arm and gripper joint positions."""
    try:
        robot_view = task._env.robots[0].robot_view
        robot_view.get_qpos_dict()
        gid = robot_view.get_gripper_movegroup_ids()[0]
        return np.asarray(robot_view.get_move_group(gid).leaf_frame_to_robot, np.float64)
    except Exception:
        return np.full((4, 4), np.nan)


def check_frames(obs, min_px: float = MIN_PX) -> None:
    """Raise DarkFrames when the exo or the wrist frame of an observation is missing or its mean
    pixel value is below `min_px`."""
    from vlageom.sim.render import obs_frames
    exo, wrist = obs_frames(obs)
    if exo is None or wrist is None:
        raise DarkFrames("exo or wrist frame not found in the observation")
    me, mw = float(exo.mean()), float(wrist.mean())
    if me < min_px or mw < min_px:
        raise DarkFrames(f"dark frame (exo {me:.1f}, wrist {mw:.1f} mean px)")


def run_natural_episode(task, policy, recorder: NaturalRecorder, min_px: float | None = MIN_PX) -> dict:
    """Roll out one sampled episode until the task is done or succeeds (MolmoSpaces task horizon:
    500 steps). The policy's own flow sampling is used (torch's global generator, seeded before the
    episode was sampled). With `min_px`, the observation of every model query is checked with
    `check_frames`. Returns the per-episode arrays (see NATURAL_KEYS) without the episode index,
    house and seed."""
    obs, _info = task.reset()
    query_steps, tok0, tmean, ee_steps = [], [], [], []
    layers = recorder.layers
    step = 0
    while not task.is_done():
        action = policy.get_action(obs)
        if action is None:
            break
        acts = recorder.take_query()
        if acts is not None:
            if min_px is not None:
                check_frames(obs, min_px)
            query_steps.append(step)
            tok0.append(np.stack([acts[L][0] for L in layers]))
            tmean.append(np.stack([acts[L].mean(0) for L in layers]))
        ee_steps.append(_step_ee_pose(task))
        obs, _reward, _term, _trunc, _info = task.step(action)
        step += 1
        if hasattr(task, "judge_success") and task.judge_success():
            break
    success = bool(task.judge_success()) if hasattr(task, "judge_success") else False
    desc = task.get_task_description() if hasattr(task, "get_task_description") else ""
    recs = recorder.records

    def stack(key):
        return np.stack([r[key] for r in recs]) if recs else np.zeros((0,), np.float32)

    if len(recs) != len(query_steps):
        log.warning("%d query records vs %d query steps", len(recs), len(query_steps))
    d = len(layers)
    return dict(
        success=success, n_queries=np.int64(len(recs)), n_query_steps=np.int64(len(query_steps)),
        query_steps=np.asarray(query_steps, np.int64), pred_chunk=stack("pred_chunk"),
        sim_time=np.asarray([r["sim_time"] for r in recs], np.float64), qvel=stack("qvel"),
        qvel_groups=np.asarray(recorder.qvel_groups), qvel_sizes=np.asarray(recorder.qvel_sizes,
                                                                           np.int64),
        ee_pose=stack("ee_pose"), ee_jac=stack("ee_jac"), obj_pos_base=stack("obj_pos_base"),
        ee_to_obj=stack("ee_to_obj"), obj_dist=np.asarray([r["obj_dist"] for r in recs], np.float64),
        n_steps=np.int64(step), task_description=np.str_(desc),
        step_ee_pose=(np.stack(ee_steps) if ee_steps else np.zeros((0, 4, 4))).astype(np.float64),
        layers=np.asarray(layers, np.int64),
        act_tok0=(np.stack(tok0) if tok0 else np.zeros((0, d, 768))).astype(np.float32),
        act_mean=(np.stack(tmean) if tmean else np.zeros((0, d, 768))).astype(np.float32))


# Keys of a natural episode file (episode_<index>.npz); q = model query of the rollout:
NATURAL_KEYS = {
    "seed": "() seed set before the episode was sampled (python random, numpy, torch)",
    "episode_index": "() episode index", "house_index": "() house",
    "success": "() task success at the end of the rollout",
    "n_queries": "() model queries recorded", "n_query_steps": "() model queries with activations",
    "query_steps": "(q,) env step of each model query",
    "pred_chunk": "(q, 16, 8) float32 predicted action chunk (normalized)",
    "sim_time": "(q,) float64 simulator time", "qvel": "(q, 9) float32 robot joint velocities",
    "qvel_groups": "joint groups of qvel", "qvel_sizes": "joint count per group",
    "ee_pose": "(q, 4, 4) float32 EE pose, robot base frame",
    "ee_jac": "(q, 6, 7) float32 EE Jacobian (world frame rows: linear, angular)",
    "obj_pos_base": "(q, 3) float32 pickup object position, base frame",
    "ee_to_obj": "(q, 3) float32 object minus EE position, base frame",
    "obj_dist": "(q,) float64 EE-object distance (m)",
    "n_steps": "() env steps of the rollout", "task_description": "() task text",
    "step_ee_pose": "(n_steps, 4, 4) float64 EE pose before every env step, base frame",
    "layers": "(6,) action-expert blocks of act_tok0 / act_mean",
    "act_tok0": "(q, 6, 768) float32 block output at the final flow step, action token 0",
    "act_mean": "(q, 6, 768) float32 the same, mean over the 16 action tokens",
}


def natural_episode_path(episode_index: int, root=None) -> Path:
    """`<root>/episode_<index:04d>.npz` (root default <data>/wrist_steer/natural/episodes)."""
    root = Path(root) if root is not None else tile_path("natural", "episodes")
    return root / f"episode_{int(episode_index):04d}.npz"


def attempt_house(i: int, attempt: int, last_good: int) -> int:
    """House of sampling attempt `attempt` (0-based) of natural episode i: house i, then i + 3,
    i + 6, i + 9 (modulo 200), and for the last attempt the house of the previous episode."""
    if attempt == 0:
        return i % NATURAL_N_HOUSES
    if attempt < NATURAL_MAX_ATTEMPTS - 1:
        return (i + NATURAL_RETRY_STRIDE * attempt) % NATURAL_N_HOUSES
    return last_good


def sampling_attempts(episodes: list[dict], partition: int) -> list[list[tuple[int, int]]]:
    """The sampling attempts of the listed episodes of one partition, in episode order: per episode
    a list of (house, seed), the attempts that could not be sampled followed by the listed (house,
    seed).

    One seed is used per attempt, counting from partition * 16, so the seeds skipped between two
    listed episodes are the failed attempts of the second one; attempt a samples
    `attempt_house(i, a, ...)`. Raises ValueError when the listed house is not the house of the
    implied attempt."""
    eps = sorted((e for e in episodes if e["partition"] == partition),
                 key=lambda e: e["episode_index"])
    prev_seed = partition * NATURAL_PER_PARTITION - 1
    last_good = (partition * NATURAL_PER_PARTITION) % NATURAL_N_HOUSES
    out = []
    for e in eps:
        n_fail = e["seed"] - prev_seed - 1
        if not 0 <= n_fail < NATURAL_MAX_ATTEMPTS:
            raise ValueError(f"episode {e['episode_index']}: seed {e['seed']} implies {n_fail} failed "
                             f"sampling attempts")
        att = [(attempt_house(e["episode_index"], a, last_good), prev_seed + 1 + a)
               for a in range(n_fail + 1)]
        if att[-1] != (e["house"], e["seed"]):
            raise ValueError(f"episode {e['episode_index']}: listed house {e['house']} is not the house "
                             f"of sampling attempt {n_fail} ({att[-1][0]})")
        out.append(att)
        prev_seed, last_good = e["seed"], e["house"]
    return out


def run_natural_partition(sim_env, partition: int, episodes: list[dict], out_root=None, *,
                          recorder: NaturalRecorder | None = None, skip_existing=True,
                          min_px: float | None = MIN_PX) -> list[int]:
    """Run the listed episodes of one partition in one process, in episode order, and write the
    episode files that are missing (all of them with skip_existing=False).

    Every sampling attempt of the listed run is repeated (`sampling_attempts`): python `random`,
    numpy and torch are seeded with the attempt's seed and the attempt's house is sampled. The
    attempts before the listed one are houses that MolmoSpaces could not sample in the listed run;
    they are made for their effect on the task sampler of the process, which the first attempt
    creates (it draws a number from the seeded numpy RNG and reseeds the three RNGs before it
    samples) and which samples every later episode.
    If such an attempt samples a task now, a warning is logged and the listed house is sampled
    next as listed. After the listed attempt, query-step frame rendering is installed and the
    episode is rolled out. Episodes up to the last missing one are rolled out, since each episode
    starts from the task sampler its predecessors used. Returns the episode indices written."""
    from vlageom.sim.engine import seed_all
    from vlageom.sim.render import install_query_frame_rendering

    recorder = recorder or NaturalRecorder()
    eps = sorted((e for e in episodes if e["partition"] == partition),
                 key=lambda e: e["episode_index"])
    if not eps:
        raise ValueError(f"no listed episodes in partition {partition}")
    attempts = sampling_attempts(eps, partition)
    missing = [i for i, e in enumerate(eps)
               if not (skip_existing and natural_episode_path(e["episode_index"], out_root).exists())]
    if not missing:
        return []
    written = []
    for e, att in zip(eps[:missing[-1] + 1], attempts):
        out = natural_episode_path(e["episode_index"], out_root)
        for house, seed in att[:-1]:
            seed_all(seed)
            try:
                sim_env.sample_episode(house_index=house)
            except Exception as exc:                              # noqa: BLE001
                log.info("episode %d: house %d (seed %d) cannot be sampled, as listed (%s: %s)",
                         e["episode_index"], house, seed, type(exc).__name__, exc)
            else:
                log.warning("episode %d: house %d (seed %d) was sampled, although the listed run "
                            "could not sample it; continuing with the listed house %d",
                            e["episode_index"], house, seed, e["house"])
        seed_all(e["seed"])
        task, policy = sim_env.sample_episode(house_index=e["house"])
        recorder.hook_model(policy.agent.model)
        install_query_frame_rendering(task, policy)
        recorder.attach(task, policy)
        res = run_natural_episode(task, policy, recorder, min_px)
        log.info("episode %d house %d seed %d: %s, %d steps, %d queries", e["episode_index"],
                 e["house"], e["seed"], "success" if res["success"] else "failure",
                 int(res["n_steps"]), len(res["query_steps"]))
        if skip_existing and out.exists():
            continue
        util.save_npz(out, dict(seed=np.int64(e["seed"]),
                                episode_index=np.int64(e["episode_index"]),
                                house_index=np.int64(e["house"]), **res))
        written.append(e["episode_index"])
    return written


# ---- natural cloud (CPU) ----------------------------------------------------------------------- #
def natural_cache_rows(path, min_queries: int = NATURAL_MIN_QUERIES) -> list[tuple]:
    """Rows of one natural episode file: (episode, query, house, success, phase, EE position (3,),
    EE rotation (3, 3), token-0 activations (L, 768), token-mean activations (L, 768)). Episodes with
    fewer than `min_queries` queries give no rows; phase = query step / (n_steps - 1)."""
    z = np.load(path, allow_pickle=False)
    ep = int(z["episode_index"])
    qs = [int(s) for s in z["query_steps"]]
    n_steps = int(z["n_steps"]) or 1
    house = int(z["house_index"])
    succ = bool(z["success"])
    steps_ee = z["step_ee_pose"]
    if len(qs) < min_queries:
        return []
    rows = []
    for q in range(min(len(qs), len(z["act_tok0"]))):
        X = np.asarray(z["act_tok0"][q], np.float32)
        Xm = np.asarray(z["act_mean"][q], np.float32)
        if not (np.isfinite(X).all() and np.isfinite(Xm).all()):
            continue
        step = qs[q]
        if step >= len(steps_ee):
            continue
        ee = np.asarray(steps_ee[step], float)
        if ee.shape != (4, 4) or not np.isfinite(ee).all():
            continue
        rows.append((ep, q, house, succ, step / max(1, n_steps - 1),
                     ee[:3, 3].astype(np.float32), ee[:3, :3].astype(np.float32), X, Xm))
    return rows


def natural_cache(files: list, layers=NATURAL_LAYERS) -> dict:
    """Stack the rows of the natural episode files `files` (in the given order) into the natural
    cache arrays: X (N, L, 768) token 0, Xm (N, L, 768) token mean, R (N, 3, 3), pos (N, 3),
    phase (N,), house, ep, q, success, layers, pass_idx."""
    rows = []
    for f in files:
        rows.extend(natural_cache_rows(f))
    if not rows:
        raise ValueError("no natural rows")
    return dict(X=np.stack([r[7] for r in rows]).astype(np.float32),
                Xm=np.stack([r[8] for r in rows]).astype(np.float32),
                R=np.stack([r[6] for r in rows]).astype(np.float32),
                pos=np.stack([r[5] for r in rows]).astype(np.float32),
                phase=np.array([r[4] for r in rows], np.float32),
                house=np.array([r[2] for r in rows]), ep=np.array([r[0] for r in rows]),
                q=np.array([r[1] for r in rows]), success=np.array([r[3] for r in rows], bool),
                layers=np.array(list(layers)), pass_idx=NATURAL_PASS)


def natural_cloud(cache: dict, layer: int = LAYER) -> dict:
    """The panel-b cloud from a natural cache: X = token-mean activations of `layer` (N, 768),
    R, phase, house, ep, q, success."""
    layers = [int(x) for x in cache["layers"]]
    li = layers.index(layer)
    return dict(X=np.asarray(cache["Xm"])[:, li, :].astype(np.float32),
                R=np.asarray(cache["R"]).astype(np.float32),
                phase=np.asarray(cache["phase"]).astype(np.float32),
                house=np.asarray(cache["house"]).astype(np.int64),
                ep=np.asarray(cache["ep"]).astype(np.int64),
                q=np.asarray(cache["q"]).astype(np.int64),
                success=np.asarray(cache["success"]).astype(bool))


# ---- orientation sweeps (simulator) ------------------------------------------------------------ #
def read_ee_jacobian_full(eng):
    """(J6 (6, 7) float32 EE Jacobian in the robot base frame (rows 0-2 linear, 3-5 angular),
    R (3, 3) world<-base rotation). Raises if unavailable."""
    rv = eng.task._env.robots[0].robot_view
    mg = rv.get_gripper_movegroup_ids()[0]
    J = np.asarray(rv.get_jacobian(mg, ["arm"]), np.float32)
    R = np.asarray(rv.base.pose, np.float32)[:3, :3]
    J6b = np.empty((6, GRIP), np.float32)
    J6b[:3, :] = R.T @ J[:3, :GRIP]
    J6b[3:, :] = R.T @ J[3:6, :GRIP]
    return J6b, R


def read_ee_jacobian_full_or_none(eng):
    """`read_ee_jacobian_full`, or (None, None) when the Jacobian cannot be read."""
    try:
        J6b, R = read_ee_jacobian_full(eng)
        return J6b, R.astype(np.float32)
    except Exception as e:
        log.warning("no full EE jacobian available: %s", e)
        return None, None


def axis_twist(uhat, dalpha):
    """Base-frame pure-rotation twist [0, 0, 0, u * dalpha] (6,) float32."""
    t = np.zeros(6, np.float32)
    t[3:] = np.asarray(uhat, np.float32) * float(dalpha)
    return t


def checked_obs(eng, min_px: float = MIN_PX):
    """`vlageom.sim.render.fresh_obs` (cameras moved to the current robot state, both rendered),
    raising DarkFrames when the exo or the wrist frame is missing or darker than `min_px`."""
    from vlageom.sim.render import fresh_obs
    obs = fresh_obs(eng)
    check_frames(obs, min_px)
    return obs


def encode_orient(eng, layers, seed, min_px: float = MIN_PX):
    """Render at the current state and read the action expert: (len(layers), 768) float32 token
    means at the final flow step with flow noise seeded by `seed`, or None if no query fired."""
    from vlageom.sim.encode import encode_obs
    obs = checked_obs(eng, min_px)
    act = encode_obs(eng, obs, layers, seed)
    if act is None:
        return None
    return np.stack([act[L] for L in layers]).astype(np.float32)


def step_orient(eng, snap, q0, R_start, ee_start, uhat, n_steps, step_rad, arm_lo, arm_hi,
                encode_fn, layers, seed, max_gain=2.5, min_cos=0.5, max_pos_drift=ORIENT_MAX_POS_DRIFT):
    """Closed-loop rotation of the hand about base-frame unit axis `uhat` with the EE position held.
    The caller restores `snap` and sets the arm to `q0` first. Each step: dq = pinv(J6(q)) @
    [0, 0, 0, uhat * step_rad] with the full Jacobian at the current joint angles, q = clip(q + dq,
    arm_lo, arm_hi), restore `snap`, set the arm to q, read the EE pose. The walk stops when the
    realized step rotation is below 1e-4 rad or above `max_gain` x step_rad, its axis has
    |cos| < `min_cos` to `uhat`, or the EE moved more than `max_pos_drift` from `ee_start`.
    Returns [(k, R (3, 3), ee (3,), encode_fn(eng, layers, seed))]."""
    from vlageom.sim.kinematics import set_arm_qpos
    from vlageom.sim.so3 import so3_axis, so3_geodesic
    out = []
    q = np.asarray(q0, np.float32).copy()
    uhat = np.asarray(uhat, np.float64)
    prev_R = np.asarray(R_start, np.float64)
    ee0 = np.asarray(ee_start, np.float64)
    for k in range(1, n_steps + 1):
        J6, _ = read_ee_jacobian_full_or_none(eng)
        if J6 is None:
            break
        dq = (np.linalg.pinv(np.asarray(J6, np.float32)) @ axis_twist(uhat, step_rad)).astype(np.float32)
        q = np.clip(q + dq, arm_lo, arm_hi).astype(np.float32)
        eng._restore(snap)
        set_arm_qpos(eng, q)
        m = eng.read_ee_full()
        if m is None:
            break
        R = np.asarray(m[:3, :3], np.float64)
        ee = np.asarray(m[:3, 3], np.float64)
        ang = so3_geodesic(prev_R, R)
        gain = ang / (step_rad + 1e-9)
        cosax = abs(float(np.dot(so3_axis(prev_R.T @ R), uhat)))
        posdrift = float(np.linalg.norm(ee - ee0))
        if ang < 1e-4 or gain > max_gain or cosax < min_cos or posdrift > max_pos_drift:
            break
        vec = encode_fn(eng, layers, seed)
        if vec is None:
            break
        out.append((k, R.astype(np.float32), ee.astype(np.float32), vec))
        prev_R = R
    return out


def harvest_orient_episode(eng, base, ep_idx, house, *, layers=ORIENT_LAYERS, n_pose=ORIENT_N_POSE,
                           step_rad=float(np.radians(ORIENT_STEP_DEG)), k_half=ORIENT_K_HALF,
                           max_pos_drift=ORIENT_MAX_POS_DRIFT, min_px: float = MIN_PX):
    """Orientation sweeps of one rolled-out episode (`base` = `eng.run_baseline(...)`), or None.

    Poses: `vlageom.sim.kinematics.pick_poses(base, n_pose)` (open-hand queries spread over the
    approach). Per pose: restore the pre-query snapshot, set the arm to the query's joint angles and
    record the unrotated row (axis -1, step 0); then for each base axis x, y, z and each direction
    walk `step_orient` up to `k_half` steps of `step_rad`, recording every step (signed step index).
    Every row is rendered with the cameras at the current robot state and encoded with the flow seed
    of that query (`eng.scene_seed + 1000 * query`). Returns the arrays of ORIENT_KEYS."""
    from vlageom.sim.kinematics import pick_poses, set_arm_qpos

    def enc(e, ls, s):
        return encode_orient(e, ls, s, min_px)

    poses = pick_poses(base, n_pose)
    if not poses:
        log.warning("ep%d h%d: no open-hand poses", ep_idx, house)
        return None
    bounds = eng.arm_action_bounds()
    if bounds is None:
        log.warning("ep%d h%d: no action-normalizer bounds; skip", ep_idx, house)
        return None
    arm_lo = np.asarray(bounds[0], np.float32).reshape(-1)[:GRIP]
    arm_hi = np.asarray(bounds[1], np.float32).reshape(-1)[:GRIP]
    snaps = eng._baseline_snaps
    qsteps = base["query_steps"]
    act_rows, R_rows, ee_rows, pose_ids, axis_ids, step_ks = [], [], [], [], [], []
    R0s, ee0s, qpos0s, pose_chunks = [], [], [], []
    pid = 0
    for c in poses:
        step = int(qsteps[c])
        if not (0 <= step < len(snaps)):
            continue
        snap = snaps[step]
        q = int(snap["chunk"])
        seed = eng.scene_seed + 1000 * q
        q0 = np.asarray(base["qpos_arm"][c], np.float32)[:GRIP]
        eng._restore(snap)
        set_arm_qpos(eng, q0)
        m0 = eng.read_ee_full()
        if m0 is None:
            continue
        R0 = np.asarray(m0[:3, :3], np.float32)
        ee0 = np.asarray(m0[:3, 3], np.float32)
        base_vec = enc(eng, layers, seed)
        if base_vec is None:
            log.warning("ep%d h%d pose c%d: no query fired; skip pose", ep_idx, house, c)
            continue
        act_rows.append(base_vec)
        R_rows.append(R0)
        ee_rows.append(ee0)
        pose_ids.append(pid)
        axis_ids.append(-1)
        step_ks.append(0)
        R0s.append(R0)
        ee0s.append(ee0)
        qpos0s.append(q0)
        pose_chunks.append(q)
        for axis_idx, (_aname, avec) in enumerate(AXIS_DIRS):
            uhat = np.asarray(avec, np.float64)
            for sgn in (-1.0, 1.0):
                eng._restore(snap)
                set_arm_qpos(eng, q0)
                walk = step_orient(eng, snap, q0, R0, ee0, sgn * uhat, k_half, step_rad, arm_lo,
                                   arm_hi, enc, layers, seed, max_pos_drift=max_pos_drift)
                for (k, R, ee, vec) in walk:
                    act_rows.append(vec)
                    R_rows.append(R)
                    ee_rows.append(ee)
                    pose_ids.append(pid)
                    axis_ids.append(axis_idx)
                    step_ks.append(int(sgn * k))
        pid += 1
    if not act_rows:
        return None
    return dict(
        act=np.stack(act_rows).astype(np.float32), R=np.stack(R_rows).astype(np.float32),
        ee=np.stack(ee_rows).astype(np.float32),
        pose_id=np.array(pose_ids, np.int32), axis_id=np.array(axis_ids, np.int32),
        step_k=np.array(step_ks, np.int32),
        R0=np.stack(R0s).astype(np.float32), ee0=np.stack(ee0s).astype(np.float32),
        qpos0=np.stack(qpos0s).astype(np.float32), pose_chunk=np.array(pose_chunks, np.int32),
        layers=np.array(layers), axes=np.array(AXIS_LABELS), step_rad=float(step_rad),
        house=int(house), ep=int(ep_idx))


# Keys of an orientation sweep file (ep<episode_index>_h<house>.npz); row = one rendered hand pose:
ORIENT_KEYS = {
    "act": "(rows, 11, 768) float32 action-expert token mean at the final flow step, ORIENT_LAYERS",
    "R": "(rows, 3, 3) float32 realized EE rotation, base frame",
    "ee": "(rows, 3) float32 realized EE position, base frame",
    "pose_id": "(rows,) int32 pose (0..3)",
    "axis_id": "(rows,) int32 rotation axis 0=x 1=y 2=z, -1 for the unrotated row of a pose",
    "step_k": "(rows,) int32 signed step index (0 = unrotated)",
    "R0": "(poses, 3, 3) float32 unrotated EE rotation per pose",
    "ee0": "(poses, 3) float32 unrotated EE position per pose",
    "qpos0": "(poses, 7) float32 arm joint angles per pose",
    "pose_chunk": "(poses,) int32 model query of each pose",
    "layers": "(11,) action-expert layers of act", "axes": "('x', 'y', 'z')",
    "step_rad": "() rotation per step (rad)", "house": "() house", "ep": "() episode index",
}


def orient_path(episode_index: int, house: int, root=None) -> Path:
    """`<root>/ep<episode_index>_h<house>.npz` (root default <data>/wrist_steer/orient)."""
    root = Path(root) if root is not None else tile_path("orient")
    return root / f"ep{int(episode_index)}_h{int(house)}.npz"


def orient_files(root=None) -> list[str]:
    """Orientation sweep files under `root` in sorted path order (the order every fit reads them)."""
    root = Path(root) if root is not None else tile_path("orient")
    return sorted(glob.glob(str(Path(root) / "*.npz")))


# ---- held-out split and rotation operator (CPU) ------------------------------------------------ #
def _axis_angle(Rm):
    """(angle (rad), unit axis) of rotation matrix Rm; axis (0, 0, 1) for angle < 1e-8."""
    tr = np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(tr))
    if th < 1e-8:
        return 0.0, np.array([0.0, 0.0, 1.0])
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    n = np.linalg.norm(v)
    return th, (v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0]))


def heldout_split(houses, seed: int = SPLIT_SEED) -> list[int]:
    """The held-out houses: numpy default_rng(seed) shuffle of the unique houses, first third."""
    rng = np.random.default_rng(seed)
    uh = np.unique(np.asarray(houses))
    rng.shuffle(uh)
    return sorted(uh[: max(1, len(uh) // 3)].tolist())


def fit_operator(files: list, layer: int = LAYER, seed: int = SPLIT_SEED, pca_k: int = PCA_K,
                 ridge: float = RIDGE, blas_threads: int = OPERATOR_BLAS_THREADS) -> dict:
    """Linear rotation operator of the orientation sweeps.

    Held-out houses: `heldout_split`. On the training rows: mean c, PCA basis V (pca_k columns,
    numpy SVD of the centered rows). Per axis, pairs of rows of one sweep whose cumulative rotation
    (signed against the base axis) differs by 10 +- 3 degrees give a ridge map M10 (Ai -> Aj in PCA
    coordinates), and Lhat = real(logm(M10)) / radians(10). The linear algebra runs with
    `blas_threads` BLAS threads (see OPERATOR_BLAS_THREADS). Returns V, c, Lhat (3, k, k),
    test_houses, layer."""
    import scipy.linalg  # noqa: F401  (scipy's BLAS must be loaded before the thread count is set)
    with util.blas_threads(blas_threads):
        return _fit_operator(files, layer, seed, pca_k, ridge)


def _fit_operator(files, layer, seed, pca_k, ridge) -> dict:
    from scipy.linalg import logm
    samples = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        li = d["layers"].tolist().index(layer)
        A = d["act"][:, li, :].astype(np.float64)
        for i in range(len(A)):
            samples.append((A[i], int(d["house"]), int(d["ep"]), int(d["pose_id"][i]),
                            int(d["axis_id"][i]), int(d["step_k"][i]), d["R"][i].astype(np.float64)))
    houses = np.array([s[1] for s in samples])
    rng = np.random.default_rng(seed)
    uh = np.unique(houses)
    rng.shuffle(uh)
    test_h = sorted(uh[: max(1, len(uh) // 3)].tolist())
    tr_mask = np.array([h not in set(test_h) for h in houses])
    Xall = np.stack([s[0] for s in samples])
    c = Xall[tr_mask].mean(0)
    _, _, Vt = np.linalg.svd(Xall[tr_mask] - c, full_matrices=False)
    V = Vt[:pca_k].T
    Z = (Xall - c) @ V
    fibers = defaultdict(list)
    for i, s in enumerate(samples):
        fibers[(s[2], s[3], s[4])].append(i)

    def cum_angles(idxs, axis):
        u = AXIS_U["xyz"[axis]]
        idxs = sorted(idxs, key=lambda i: samples[i][5])
        cum = [0.0]
        for a_, b_ in zip(idxs[:-1], idxs[1:]):
            th, v = _axis_angle(samples[a_][6].T @ samples[b_][6])
            v_base = samples[a_][6] @ v
            cum.append(cum[-1] + (1.0 if float(v_base @ u) >= 0 else -1.0) * th)
        return idxs, np.array(cum)

    Lhat = np.zeros((3, pca_k, pca_k))
    for ax in range(3):
        Ai, Aj = [], []
        for (_ep, _pose, axis), idxs in fibers.items():
            if axis != ax or not tr_mask[idxs[0]]:
                continue
            oi, cum = cum_angles(idxs, axis)
            for a_ in range(len(oi)):
                for b_ in range(a_ + 1, len(oi)):
                    dth = np.degrees(cum[b_] - cum[a_])
                    if abs(abs(dth) - 10.0) > 3.0:
                        continue
                    i, j = (oi[a_], oi[b_]) if dth > 0 else (oi[b_], oi[a_])
                    Ai.append(Z[i])
                    Aj.append(Z[j])
        Ai = np.stack(Ai)
        Aj = np.stack(Aj)
        M10 = np.linalg.solve(Ai.T @ Ai + ridge * np.eye(pca_k), Ai.T @ Aj)
        Lhat[ax] = np.real(logm(M10)) / np.deg2rad(10.0)
        log.info("operator axis %d: %d training pairs, |Lhat| = %.2f", ax, len(Ai),
                 np.linalg.norm(Lhat[ax]))
    return {"V": V, "c": c, "Lhat": Lhat, "test_houses": np.array(test_h), "layer": layer}


def load_operator(path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in ("V", "c", "Lhat", "test_houses")}


def steer_delta(op, base_act_768, axis: str, theta_deg: float, Lmat=None):
    """Operator edit V (expm(theta L) - I) V^T (a - c) for activation a; Lmat replaces Lhat[axis]."""
    from scipy.linalg import expm
    L = Lmat if Lmat is not None else op["Lhat"][AXIS_ID[axis]]
    z = op["V"].T @ (np.asarray(base_act_768, np.float64) - op["c"])
    dz = (expm(np.deg2rad(theta_deg) * L) - np.eye(L.shape[0])) @ z
    return (op["V"] @ dz).astype(np.float32)


def random_generator(op, seed: int = 1):
    """Random antisymmetric generator (numpy default_rng(seed)) with the mean Frobenius norm of the
    three operator generators: the control edit of the one-shot run."""
    rng = np.random.default_rng(seed)
    Wr = rng.standard_normal(op["Lhat"][0].shape)
    Lrand = (Wr - Wr.T)
    Lrand *= np.mean([np.linalg.norm(op["Lhat"][a]) for a in range(3)]) / np.linalg.norm(Lrand)
    return Lrand


# ---- circular autoencoder chart (training on GPU) ---------------------------------------------- #
def load_chart_pairs(files: list, layer: int = LAYER, seed: int = SPLIT_SEED,
                     max_pair_deg: float = 340.0):
    """Training data of the chart from the orientation sweep files (sorted path order).

    X (N, 768) float32 token-mean activations of `layer`; is_te (N,) rows of held-out houses
    (`heldout_split`); pairs: every ordered pair (i, j) of rows of one sweep (episode, pose, axis)
    whose cumulative rotation (signed against the base axis) differs by 4..max_pair_deg degrees,
    with (axis, dth = rotation from i to j in rad); pair_te: pairs whose first row is held out."""
    U = {0: np.array([1.0, 0, 0]), 1: np.array([0, 1.0, 0]), 2: np.array([0, 0, 1.0])}
    X, house, meta = [], [], []
    fibers = defaultdict(list)
    for f in files:
        d = np.load(f, allow_pickle=True)
        li = d["layers"].tolist().index(layer)
        A = d["act"][:, li, :].astype(np.float32)
        for i in range(len(A)):
            k = len(X)
            X.append(A[i])
            house.append(int(d["house"]))
            meta.append((int(d["ep"]), int(d["pose_id"][i]), int(d["axis_id"][i]),
                         int(d["step_k"][i]), d["R"][i].astype(np.float64)))
            fibers[(int(d["ep"]), int(d["pose_id"][i]), int(d["axis_id"][i]))].append(k)
    X = np.stack(X)
    house = np.array(house)
    rng = np.random.default_rng(seed)
    uh = np.unique(house)
    rng.shuffle(uh)
    test_h = set(uh[: len(uh) // 3].tolist())
    is_te = np.array([h in test_h for h in house])
    pairs = []
    for (_ep, _pose, axis), idxs in fibers.items():
        if axis not in (0, 1, 2):
            continue
        idxs = sorted(idxs, key=lambda k: meta[k][3])
        cum, u = [0.0], U[axis]
        for a_, b_ in zip(idxs[:-1], idxs[1:]):
            th, v = _axis_angle(meta[a_][4].T @ meta[b_][4])
            v_base = meta[a_][4] @ v
            cum.append(cum[-1] + (1.0 if float(v_base @ u) >= 0 else -1.0) * th)
        for a_ in range(len(idxs)):
            for b_ in range(len(idxs)):
                if a_ == b_:
                    continue
                dth = cum[b_] - cum[a_]
                if abs(np.degrees(dth)) < 4.0 or abs(np.degrees(dth)) > max_pair_deg:
                    continue
                pairs.append((idxs[a_], idxs[b_], axis, dth))
    pairs = np.array(pairs, dtype=[("i", "i4"), ("j", "i4"), ("axis", "i4"), ("dth", "f4")])
    pair_te = is_te[pairs["i"]]
    log.info("rows %d (held out %d) | pairs %d (held out %d) | at most %.0f deg",
             len(X), is_te.sum(), len(pairs), pair_te.sum(), max_pair_deg)
    return X, is_te, pairs, pair_te


def _mlp(d_in, d_out):
    from torch import nn
    return nn.Sequential(nn.Linear(d_in, 512), nn.LayerNorm(512), nn.GELU(),
                         nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                         nn.Linear(256, d_out))


def _decoder(z_dim):
    from torch import nn
    return nn.Sequential(nn.Linear(z_dim, 256), nn.LayerNorm(256), nn.GELU(),
                         nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(),
                         nn.Linear(512, 768))


def circular_encoder(z_dim: int = 32):
    """The chart encoder: an MLP trunk 768-512-256-z_dim, then code pairs (0,1), (2,3), (4,5) are
    each projected onto a circle of learned radius exp(log_r[axis]); the other dimensions pass
    through."""
    import torch
    from torch import nn

    class CircularEncoder(nn.Module):
        def __init__(self, z_dim=32):
            super().__init__()
            self.trunk = _mlp(768, z_dim)
            self.log_r = nn.Parameter(torch.log(torch.full((3,), 0.1)))

        def forward(self, x):
            h = self.trunk(x)
            z = h.clone()
            for a in range(3):
                p0, p1 = PLANE[a]
                v = h[:, [p0, p1]]
                u = v / (v.norm(dim=1, keepdim=True) + 1e-6)
                r = torch.exp(self.log_r[a])
                z[:, p0] = r * u[:, 0]
                z[:, p1] = r * u[:, 1]
            return z

    return CircularEncoder(z_dim)


def block_rotate(z, axis, dth):
    """Rotate each code's pair PLANE[axis] by dth (rad); axis is a long tensor (batched)."""
    import torch
    z2 = z.clone()
    for a in range(3):
        m = axis == a
        if m.any():
            p0, p1 = PLANE[a]
            c, s = torch.cos(dth[m]), torch.sin(dth[m])
            z2[m, p0] = c * z[m, p0] - s * z[m, p1]
            z2[m, p1] = s * z[m, p0] + c * z[m, p1]
    return z2


def _r2(y, yhat):
    return float(1.0 - ((y - yhat) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12))


def train_chart(X, is_te, pairs, pair_te, *, z_dim=32, steps=8000, batch=512, lr=1e-3,
                w_recon=1.0, w_latent=1.0, w_decode=2.0, max_pair_deg=340.0, seed=0,
                device=None) -> tuple[dict, dict]:
    """Train the chart (CUDA when available).

    Inputs are standardized with the training rows' mean and RMS norm. Each step draws `batch`
    training rows (torch generator of `device`) and `batch` training pairs (numpy global RNG) and
    minimizes w_recon * |dec(enc(x)) - x|^2 + w_latent * |R_dth enc(x_i) - enc(x_j)|^2
    + w_decode * |dec(R_dth enc(x_i)) - x_j|^2 (means) with Adam. Every 1000 steps and at the end
    the held-out reconstruction R^2 and the held-out decode-transport R^2 per axis (2,000 held-out
    pairs drawn with default_rng(0)) are logged.

    Returns (checkpoint dict, report dict). torch.manual_seed(seed) and np.random.seed(seed) are set
    here, before the networks are built."""
    import torch
    args = dict(z_dim=z_dim, steps=steps, batch=batch, lr=lr, w_recon=w_recon, w_latent=w_latent,
                w_decode=w_decode, max_pair_deg=max_pair_deg, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean = X[~is_te].mean(0)
    scale = float(np.sqrt(((X[~is_te] - mean) ** 2).sum(1).mean()))
    Xn = torch.tensor((X - mean) / scale, device=dev)
    Ptr = pairs[~pair_te]
    Pte = pairs[pair_te]
    enc = circular_encoder(z_dim).to(dev)
    dec = _decoder(z_dim).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    ntr_i = torch.tensor(np.where(~is_te)[0], device=dev)

    def draw(P, n):
        k = np.random.randint(0, len(P), n)
        return (torch.tensor(P["i"][k].astype(np.int64), device=dev),
                torch.tensor(P["j"][k].astype(np.int64), device=dev),
                torch.tensor(P["axis"][k].astype(np.int64), device=dev),
                torch.tensor(P["dth"][k], device=dev))

    rec_te, per_ax = float("nan"), {}
    for s in range(steps):
        ni = ntr_i[torch.randint(0, len(ntr_i), (batch,), device=dev)]
        z = enc(Xn[ni])
        L_rec = ((dec(z) - Xn[ni]) ** 2).mean()
        i, j, ax, dth = draw(Ptr, batch)
        zi = enc(Xn[i])
        zj = enc(Xn[j])
        zrot = block_rotate(zi, ax, dth)
        L_lat = ((zrot - zj) ** 2).mean()
        L_dec = ((dec(zrot) - Xn[j]) ** 2).mean()
        loss = w_recon * L_rec + w_latent * L_lat + w_decode * L_dec
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 1000 == 0 or s == steps - 1:
            with torch.no_grad():
                te_i = torch.tensor(np.where(is_te)[0], device=dev)
                rec_te = _r2(Xn[te_i], dec(enc(Xn[te_i])))
                per_ax = {}
                for a, nm in ((0, "x"), (1, "y"), (2, "z")):
                    Pa = Pte[Pte["axis"] == a]
                    if not len(Pa):
                        continue
                    k = np.random.default_rng(0).integers(0, len(Pa), min(2000, len(Pa)))
                    ii = torch.tensor(Pa["i"][k].astype(np.int64), device=dev)
                    jj = torch.tensor(Pa["j"][k].astype(np.int64), device=dev)
                    tt = torch.tensor(Pa["dth"][k], device=dev)
                    aa = torch.full((len(k),), a, dtype=torch.long, device=dev)
                    per_ax[nm] = round(_r2(Xn[jj], dec(block_rotate(enc(Xn[ii]), aa, tt))), 3)
            log.info("step %4d held-out reconstruction R2 %.3f, decode-transport R2 %s", s, rec_te,
                     per_ax)
    ck = {"enc_trunk": enc.trunk.state_dict(), "log_r": enc.log_r.detach().cpu(),
          "dec": dec.state_dict(), "mean": mean, "scale": scale, "z_dim": z_dim,
          "version": "circular2", "args": args}
    report = {"recon_r2_test": rec_te, "decode_transport_r2_test": per_ax,
              "radii": torch.exp(enc.log_r).detach().cpu().tolist(), "args": args}
    return ck, report


class Chart:
    """A trained chart loaded for use: `enc(x)` (standardized inputs, (n, 768) -> (n, z_dim)),
    `dec`, `mean`, `scale`, `radii`. `delta` / `delta_torch` give the edit that rotates the
    activation's point on the circle of `axis` by theta_deg: scale * (dec(R_theta enc(x)) - dec(enc(x)))
    for the standardized activation x."""

    def __init__(self, path, device="cpu"):
        import torch
        ck = torch.load(path, map_location=device, weights_only=False)
        zd = ck["z_dim"]
        trunk = _mlp(768, zd).to(device)
        self.dec = _decoder(zd).to(device)
        trunk.load_state_dict(ck["enc_trunk"])
        trunk.eval()
        self.radii = torch.exp(ck["log_r"]).to(device)
        self._trunk = trunk
        self.dec.load_state_dict(ck["dec"])
        self.dec.eval()
        self.mean = ck["mean"]
        self.scale = float(ck["scale"])
        self.device = device

    def enc(self, x):
        h = self._trunk(x)
        z = h.clone()
        for a in range(3):
            p0, p1 = 2 * a, 2 * a + 1
            v = h[:, [p0, p1]]
            u = v / (v.norm(dim=1, keepdim=True) + 1e-6)
            z[:, p0] = self.radii[a] * u[:, 0]
            z[:, p1] = self.radii[a] * u[:, 1]
        return z

    def encode(self, X):
        """Codes (n, z_dim) float32 of activations X (n, 768), on the CPU."""
        import torch
        with torch.no_grad():
            return self.enc(torch.tensor((X - self.mean) / self.scale, dtype=torch.float32)).numpy()

    def delta(self, base_act_768, axis: str, theta_deg: float):
        import torch
        pooled = torch.tensor(np.asarray(base_act_768, np.float32), device=self.device)
        return self.delta_torch(pooled, axis, theta_deg).cpu().numpy().astype(np.float32)

    def delta_torch(self, pooled_768_t, axis: str, theta_deg: float):
        import torch
        if not hasattr(self, "_mean_t"):
            self._mean_t = torch.tensor(self.mean, device=self.device, dtype=torch.float32)
        a = AXIS_ID[axis]
        x = ((pooled_768_t.float() - self._mean_t) / self.scale).unsqueeze(0)
        with torch.no_grad():
            z = self.enc(x)
            z2 = z.clone()
            th = torch.tensor(np.deg2rad(theta_deg), device=self.device, dtype=z.dtype)
            c, s = torch.cos(th), torch.sin(th)
            z2[:, 2 * a] = c * z[:, 2 * a] - s * z[:, 2 * a + 1]
            z2[:, 2 * a + 1] = s * z[:, 2 * a] + c * z[:, 2 * a + 1]
            dv = (self.dec(z2) - self.dec(z)).squeeze(0) * self.scale
        return dv


# ---- steering (simulator) ---------------------------------------------------------------------- #
def read_arm_qpos(eng):
    """Arm joint angles (7,) float32 from the robot's joint-position dict."""
    rv = eng.task._env.robots[0].robot_view
    qd = rv.get_qpos_dict()
    for key in ("arm", "panda_arm", "franka_arm"):
        if key in qd:
            return np.asarray(qd[key], np.float32).reshape(-1)[:GRIP]
    return np.concatenate([np.asarray(v, np.float32).reshape(-1) for v in qd.values()])[:GRIP]


def read_ee_rot_base(eng):
    """EE rotation (3, 3) float32 in the robot base frame."""
    rv = eng.task._env.robots[0].robot_view
    mg = rv.get_gripper_movegroup_ids()[0]
    m = np.asarray(rv.get_move_group(mg).leaf_frame_to_robot, np.float32)
    return m[:3, :3]


def measure(a0, sf, bounds, J6, q0, omega, grip_hold, n_exec=STEER_N_EXEC):
    """Rotation commanded by a decoded action chunk `sf` (1, 16, 8): the arm channels are mapped to
    joint angles with the normalizer bounds, the first `n_exec` per-step joint changes (from q0) are
    mapped to EE twists with the base-frame Jacobian J6, and their mean angular velocity w is
    compared with the target `omega` (rad per step, base frame).

    Returns ang_cos (cos(w, omega)), gain (|w| / |omega|), ang_realized_deg_perstep (|w| in deg),
    pos_leak (RMS linear velocity), grip_after (mean gripper channel), grip_leak (RMS gripper change
    from grip_hold), arm_jerk (RMS step-to-step arm channel change)."""
    lo, hi = bounds
    scale, bias = (hi - lo) / 2.0, (hi + lo) / 2.0

    def twist(chunk):
        q = chunk[0, :, :GRIP] * scale + bias
        n = min(n_exec, q.shape[0])
        q_prev = np.concatenate([q0[None], q[: n - 1]], 0)
        return (q[:n] - q_prev) @ J6.T

    tw = twist(sf)
    lin, ang = tw[:, :3], tw[:, 3:]
    w_mean = ang.mean(0)
    wn, tn = np.linalg.norm(w_mean), np.linalg.norm(omega)
    return {
        "ang_cos": float(w_mean @ omega / (wn * tn)) if wn * tn > 1e-12 else float("nan"),
        "gain": float(wn / (tn + 1e-12)),
        "ang_realized_deg_perstep": float(np.degrees(wn)),
        "pos_leak": float(np.sqrt((lin ** 2).sum(1).mean())),
        "grip_after": float(sf[0, :, GRIP].mean()),
        "grip_leak": float(np.sqrt(((sf[0, :, GRIP] - grip_hold) ** 2).mean())),
        "arm_jerk": float(np.sqrt(((sf[0, 1:, :GRIP] - sf[0, :-1, :GRIP]) ** 2).mean())),
    }


def find_landmark(base):
    """The query of a rollout where edits are written: the first query of a stable grasp (grasp
    contact at this and the next query; returns the next query, 'after_grasp'), else the middle
    query ('mid_traj') when the rollout has at least 4 queries, else (None, None)."""
    g = base.get("grasp_q", [])
    for q in range(len(g) - 1):
        if g[q] and g[q + 1]:
            return min(q + 1, len(base["query_steps"]) - 1), "after_grasp"
    nq = len(base["query_steps"])
    if nq >= 4:
        return nq // 2, "mid_traj"
    return None, None


def persistent_chart_continuation(eng, from_step: int, layer: int, chart, axis: str,
                                  theta_deg: float, *, n_steps=PERSIST_STEPS, u_axis=None) -> dict:
    """Held edit: restore the pre-query snapshot at `from_step` and run the policy for `n_steps`
    env steps; at the final flow step of every model query the chart edit for theta_deg is computed
    from the live token-mean activation of action_expert.blocks.{layer} and added to all action
    tokens. chart=None or theta_deg=0 is the unedited reference. Flow noise of query q is seeded with
    eng.scene_seed + 1000 * q, as in the rollout.

    Returns rot_series_deg (rotation of the hand relative to the restore point after every step,
    signed about u_axis, default the base axis of `axis`), final_rot_deg, pos_drift (EE
    displacement, m), n_steps, grip_series (executed gripper command per step), bias_series_deg
    (the written angle at every edited query)."""
    import torch
    from vlageom.sim.engine import gripper_of
    u = AXIS_U[axis] if u_axis is None else u_axis
    block = eng.model.get_submodule(f"action_expert.blocks.{layer}")
    steer_on = chart is not None and abs(theta_deg) > 1e-9
    state = {"flow_i": 0, "bias": []}

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        state["flow_i"] += 1
        if steer_on and state["flow_i"] == eng.num_steps:
            pooled = t.detach().float().reshape(-1, t.shape[-1]).mean(0)
            r = theta_deg
            state["bias"].append(round(r, 2))
            dv = chart.delta_torch(pooled, axis, r)
            t = t + dv.to(t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return None

    def persist_gen(**kw):
        eng._chunk += 1
        seed = eng.scene_seed + 1000 * (eng._chunk - 1)
        kw2, steps = eng._seeded_kw(kw, seed)
        state["flow_i"] = 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return eng._gen_orig(num_steps=steps, **kw2)

    h = block.register_forward_hook(hook, prepend=True)
    orig = eng.model.generate_actions
    eng.model.generate_actions = persist_gen
    angles, grips = [], []
    try:
        obs = eng._restore(eng._baseline_snaps[from_step])
        R0 = read_ee_rot_base(eng)
        p0 = np.asarray(eng.read_ee(), np.float32)
        step = from_step
        while not eng.task.is_done() and step < from_step + n_steps:
            action = eng.policy.get_action(obs)
            if action is None:
                break
            obs, *_ = eng.task.step(action)
            step += 1
            Rt = read_ee_rot_base(eng)
            ang, v = _axis_angle(R0.T @ Rt)
            v_base = R0 @ v
            angles.append(float(np.degrees(ang) * np.sign(v_base @ u + 1e-12)))
            grips.append(float(gripper_of(action)))
        p1 = np.asarray(eng.read_ee(), np.float32)
    finally:
        eng.model.generate_actions = orig
        h.remove()
    return {"rot_series_deg": angles, "final_rot_deg": angles[-1] if angles else 0.0,
            "pos_drift": float(np.linalg.norm(p1 - p0)), "n_steps": len(angles),
            "grip_series": grips, "bias_series_deg": state["bias"]}


def _signed(thetas):
    return [t for s in (1, -1) for t in [s * x for x in thetas]]


def steer_episode(eng, ep_idx, house, seed, *, mode, chart, op, Lrand, layer=LAYER,
                  thetas=STEER_THETAS, axes=("z",), n_exec=STEER_N_EXEC,
                  max_steps=STEER_MAX_STEPS, persist=PERSIST_STEPS, control=True):
    """Roll out one episode and write chart edits at its landmark query (`find_landmark`).

    mode "oneshot": rows noop (z, 0 deg), chart (each axis in `axes`, each signed theta) and, with
    `control`, random_gen (z, +40 and -40 deg, operator edit with the random generator `Lrand`);
    each row is `measure` of the decoded chunk with the edit added at the final flow step of the
    landmark query, plus delta_norm. mode "persist": a persist_noop reference row and one persist
    row per axis and signed theta (`persistent_chart_continuation` over `persist` steps), with
    net_rot_deg = final rotation minus the reference's.

    Returns (rows, landmark record) or (None, None) when the episode is too short."""
    import torch
    eng.sample(house, seed)
    base = eng.run_baseline(max_steps)
    q, lm = find_landmark(base)
    if q is None:
        log.warning("ep%d h%d: episode too short (%d queries); skip", ep_idx, house,
                    base["n_queries"])
        return None, None
    from_step = base["query_steps"][q]
    eng._restore(eng._baseline_snaps[from_step])
    q0 = read_arm_qpos(eng)
    J6b, _ = read_ee_jacobian_full(eng)
    bounds = eng.arm_action_bounds()
    if bounds is None or J6b is None:
        log.warning("ep%d: missing bounds/jacobian; skip", ep_idx)
        return None, None
    kw = eng._captured[q]
    cache = eng._build_backbone_cache(kw)
    prefix = eng._compute_prefix(cache, eng.scene_seed + 1000 * q)
    a0_t, base_act = eng._final_step(cache, prefix, None, layer, grad=False)
    a0 = a0_t.float().cpu().numpy()
    base_act = base_act.cpu().numpy()
    grip_hold = float(a0[0, :, GRIP].mean())
    norm = float(np.linalg.norm(base_act))
    log.info("ep%d h%d seed%d: landmark %s q%d step%d (%d queries), \u2016base_act\u2016=%.1f", ep_idx,
             house, seed, lm, q, from_step, base["n_queries"], norm)
    landmark = dict(ep=ep_idx, house=house, seed=seed, q=q, landmark=lm, step=from_step,
                    n_queries=base["n_queries"], base_act_norm=norm)
    head = {"ep": ep_idx, "house": house, "seed": seed, "q": q, "landmark": lm}
    rows = []
    if mode == "persist":
        ref = persistent_chart_continuation(eng, from_step, layer, None, "z", 0.0, n_steps=persist)
        rows.append({**head, "tag": "persist_noop", "axis": "z", "theta_deg": 0.0, **ref})
        log.info("  persist_noop: final_rot=%+.1f deg drift=%.3f (n=%d)", ref["final_rot_deg"],
                 ref["pos_drift"], ref["n_steps"])
        for ax in axes:
            for th in _signed(thetas):
                r = persistent_chart_continuation(eng, from_step, layer, chart, ax, th,
                                                  n_steps=persist)
                net = r["final_rot_deg"] - ref["final_rot_deg"]
                rows.append({**head, "tag": "persist", "axis": ax, "theta_deg": th,
                             "net_rot_deg": net, **r})
                log.info("  persist %s %+6.1f deg: final_rot=%+.1f deg (net %+.1f) drift=%.3f", ax,
                         th, r["final_rot_deg"], net, r["pos_drift"])
        return rows, landmark
    conds = [("z", 0.0, None, "noop")]
    conds += [(ax, th, None, "chart") for ax in axes for th in _signed(thetas)]
    if control:
        conds += [("z", th, Lrand, "random_gen") for th in (40.0, -40.0)]
    for ax, th, Lmat, tag in conds:
        if tag == "chart":
            dv = chart.delta(base_act, ax, th)
        else:
            dv = steer_delta(op, base_act, ax, th, Lmat=Lmat)
        sf_t, _ = eng._final_step(cache, prefix, torch.tensor(dv, device=eng.device), layer,
                                  grad=False)
        sf = sf_t.float().cpu().numpy()
        omega = np.deg2rad(th) / max(1, n_exec) * AXIS_U[ax]
        m = measure(a0, sf, bounds, J6b, q0, omega, grip_hold, n_exec=n_exec)
        rows.append({**head, "tag": tag, "axis": ax, "theta_deg": th,
                     "delta_norm": float(np.linalg.norm(dv)), **m})
        log.info("  %-10s %s %+6.1f deg: ang_cos=%s gain=%.2f pos_leak=%.4f grip_leak=%.3f |d|=%.1f",
                 tag, ax, th, ("%.2f" % m["ang_cos"]) if m["ang_cos"] == m["ang_cos"] else "nan",
                 m["gain"], m["pos_leak"], m["grip_leak"], float(np.linalg.norm(dv)))
    return rows, landmark


# ---- panel a (CPU) ----------------------------------------------------------------------------- #
def fit_circle(theta, Z2):
    """Least-squares Z2 ~ [cos theta, sin theta, 1] @ C. Returns (C (3, 2), relative residual,
    radius = mean of the cos/sin coefficient norms, anisotropy = their ratio)."""
    theta = np.asarray(theta, np.float64)
    Z2 = np.asarray(Z2, np.float64)
    D = np.column_stack([np.cos(theta), np.sin(theta), np.ones_like(theta)])
    C, *_ = np.linalg.lstsq(D, Z2, rcond=None)
    pred = D @ C
    denom = np.linalg.norm(Z2 - Z2.mean(0)) + 1e-12
    resid = float(np.linalg.norm(Z2 - pred) / denom)
    semis = np.array([np.linalg.norm(C[0]), np.linalg.norm(C[1])])
    return C, resid, float(semis.mean()), float(max(semis) / (min(semis) + 1e-12))


def closure_error(theta, Z2):
    """(distance between the means of the top-10% and bottom-10% angle points / (2 x RMS radius),
    angular coverage max(theta) - min(theta) (rad))."""
    theta = np.asarray(theta, np.float64)
    Z2 = np.asarray(Z2, np.float64)
    cov = float(theta.max() - theta.min())
    lo = theta <= np.percentile(theta, 10)
    hi = theta >= np.percentile(theta, 90)
    if lo.sum() < 1 or hi.sum() < 1:
        return float("nan"), cov
    gap = float(np.linalg.norm(Z2[hi].mean(0) - Z2[lo].mean(0)))
    r = float(np.sqrt(((Z2 - Z2.mean(0)) ** 2).sum(1).mean()))
    return float(gap / (2.0 * r + 1e-12)), cov


def phase_operator(Z, center=None):
    """Per-step phase advance of ordered 2-D points about `center`: lambda = median |z[k+1]| / |z[k]|
    x exp(i median arg(z[k+1] conj(z[k]))). Returns (lambda, T, eigenvalues of the least-squares map
    T: Z[:-1] -> Z[1:])."""
    Z = np.asarray(Z, np.float64)
    ctr = np.asarray(center, np.float64) if center is not None else Z[:, :2].mean(0)
    z = (Z[:, 0] - ctr[0]) + 1j * ((Z[:, 1] - ctr[1]) if Z.shape[1] > 1 else 0.0)
    d = z[1:] * np.conj(z[:-1])
    arg = float(np.median(np.angle(d[np.abs(d) > 0]))) if np.any(np.abs(d) > 0) else 0.0
    rad = np.abs(z)
    mag = float(np.median(rad[1:] / (rad[:-1] + 1e-12)))
    lam = mag * np.exp(1j * arg)
    X, Y = Z[:-1], Z[1:]
    T = (Y.T @ np.linalg.pinv(X.T))
    w = np.linalg.eigvals(T)
    return lam, T, w[np.argsort(np.abs(np.angle(w)))]


def panel_a(sweep_path, layer: int = LAYER, token: int = 0, n_pc: int = 12) -> tuple[dict, dict]:
    """Panel a from one wrist roll sweep file (`vlageom.wrist.SWEEP_KEYS`): rows sorted by joint
    angle, token `token` of `layer` minus the sweep mean, numpy SVD; PC signs as set below.

    Returns (panel, metrics): panel = Z2 (rows, 2) PC1-2 coordinates, roll (rows,) joint angle
    (deg, ascending), evr (2,) explained-variance ratios of PC1-2, src; metrics = closure_error,
    coverage_deg, H1_top_persistence (largest H1 persistence of the top-`n_pc` PCs, ripser with
    coefficients mod 2; None without ripser), eig_abs / eig_arg_deg (`phase_operator` about the
    fitted circle centre), circle_resid, anisotropy, radius (`fit_circle`), evr_pc1, evr_pc2."""
    from numpy.linalg import svd
    z = np.load(sweep_path, allow_pickle=True)
    layers = [int(v) for v in z["layers"]]
    li = layers.index(layer)
    roll = np.degrees(np.asarray(z["jval"], float))
    order = np.argsort(roll)
    th = np.radians(roll[order])
    Z = z["act"][order][:, li, token, :].astype(np.float64)
    Z = Z - Z.mean(0)
    _U, S, Vt = svd(Z, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    Z2 = Z @ Vt[:2].T
    # PC signs: PC1 increases with the joint angle, and the sweep's middle row lies on the positive
    # side of PC2 relative to its two ends.
    s1 = 1.0 if np.corrcoef(Z2[:, 0], th)[0, 1] >= 0 else -1.0
    s2 = 1.0 if Z2[len(Z2) // 2, 1] >= 0.5 * (Z2[0, 1] + Z2[-1, 1]) else -1.0
    if s1 < 0 or s2 < 0:
        Vt = Vt.copy()
        Vt[0] *= s1
        Vt[1] *= s2
        Z2 = Z @ Vt[:2].T
    C, resid, radius, aniso = fit_circle(th, Z2)
    closure, cov = closure_error(th, Z2)
    lam, _, _ = phase_operator(Z2, center=C[2])
    try:
        from ripser import ripser
        dgm = ripser(Z @ Vt[:n_pc].T, maxdim=1, coeff=2)["dgms"][1]
        pers = np.sort(dgm[:, 1] - dgm[:, 0])[::-1] if len(dgm) else np.array([0.0])
        h1 = float(pers[0])
    except ImportError:
        log.warning("ripser is not installed: H1_top_persistence is left out")
        h1 = None
    res = dict(closure_error=float(closure), coverage_deg=float(np.degrees(cov)),
               H1_top_persistence=h1, eig_abs=float(abs(lam)),
               eig_arg_deg=float(np.degrees(np.angle(lam))), circle_resid=float(resid),
               anisotropy=float(aniso), radius=float(radius),
               evr_pc1=float(evr[0]), evr_pc2=float(evr[1]))
    metrics = {k: (round(v, 4) if v is not None else None) for k, v in res.items()}
    rs = roll[order]
    metrics.update(angle="joint", npz=str(sweep_path), layer=layer, n=int(len(rs)),
                   roll_min_deg=round(float(rs.min()), 3), roll_max_deg=round(float(rs.max()), 3))
    panel = dict(Z2=Z2, roll=rs, evr=evr[:2], src=np.str_(str(sweep_path)))
    return panel, metrics


# ---- panels b-d (CPU) -------------------------------------------------------------------------- #
def natural_circle(chart: Chart, natural: dict):
    """Panel b: the natural cloud on the chart's circle for rotation about base z (code pair 4, 5).

    Returns (ph (N,) float64 display phase, yaw (N,) float64 EE yaw from R, r mean code radius).
    The circle has no preferred origin, so the phase is shifted by the circular mean of
    (phase - yaw): ph = angle(exp(i (phase - offset)))."""
    X, R = natural["X"], np.asarray(natural["R"]).astype(np.float64)
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    Z = chart.encode(X)
    zp = Z[:, [4, 5]]
    ph = np.arctan2(zp[:, 1], zp[:, 0])
    if abs(np.exp(1j * (ph + yaw)).mean()) > abs(np.exp(1j * (ph - yaw)).mean()):
        log.warning("the chart's z phase runs opposite to the hand's yaw: panel b's colours run "
                    "clockwise")
    off = np.angle(np.exp(1j * (ph - yaw)).mean())
    return np.angle(np.exp(1j * (ph - off))), yaw, float(np.hypot(*zp.T).mean())


def yaw_bin_means(ph, yaw, n_bins: int = 36):
    """Circular mean of ph in each of `n_bins` equal yaw bins over [-pi, pi): (means, bin centres)."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    ctr = (edges[:-1] + edges[1:]) / 2
    mph = np.array([np.angle(np.exp(1j * ph[(yaw >= a) & (yaw < b)]).mean())
                    for a, b in zip(edges[:-1], edges[1:])])
    return mph, ctr


def oneshot_response(r) -> float:
    """Rotation (deg) of the decoded chunk along the written direction, for a one-shot row:
    sign(theta) x ang_cos x per-step rotation x 8 executed steps."""
    return np.sign(r["theta_deg"]) * r["ang_cos"] * r["ang_realized_deg_perstep"] * CHUNK_STEPS


def noop_chunk_rotation(rows) -> dict:
    """{episode: |per-step rotation| x 8 executed steps (deg)} of the unedited ("noop") one-shot
    rows: how far each scene's unedited chunk turns the wrist."""
    return {int(r["ep"]): abs(r["ang_realized_deg_perstep"]) * float(CHUNK_STEPS) for r in rows
            if r["tag"] == "noop"}


def drift_episodes(rows, max_deg: float = DRIFT_MAX_DEG) -> list[int]:
    """Episodes of the one-shot rows whose unedited chunk turns the wrist by more than `max_deg`
    (`noop_chunk_rotation`); panel c and the compensation numbers leave them out."""
    return sorted(e for e, v in noop_chunk_rotation(rows).items() if v > max_deg)


def panel_bcd(chart: Chart, natural: dict, persist_rows: list, oneshot_rows: list,
              house_d: int = HOUSE_D) -> dict:
    """Panel b-d arrays:
      ph, yaw (N,), r              natural cloud on the z circle (`natural_circle`)
      bin_ph, bin_ctr (36,)        circular means of ph per yaw bin and the bin centres
      c_theta, c_resp (n,)         panel c: written angle and `oneshot_response` of the one-shot
                                   chart rows about z with theta != 0, the scenes of
                                   `drift_episodes` left out, in row order
      d_theta (5,), d_rot (5, 32)  panel d: rotation series of house `house_d`'s held-edit rows for
                                   -40, -20, +20, +40 and 0 deg (the unedited reference)"""
    ph, yaw, r = natural_circle(chart, natural)
    mph, ctr = yaw_bin_means(ph, yaw)
    drift = drift_episodes(oneshot_rows)
    c_theta, c_resp = [], []
    for row in oneshot_rows:
        if (row["tag"] == "chart" and row["axis"] == "z" and row["theta_deg"]
                and row["ep"] not in drift):
            c_theta.append(row["theta_deg"])
            c_resp.append(oneshot_response(row))
    by_theta = {row["theta_deg"]: row["rot_series_deg"] for row in persist_rows
                if row["house"] == house_d}
    d_theta = D_THETAS + [0.0]
    return dict(ph=ph, yaw=yaw, r=np.float64(r), bin_ph=mph, bin_ctr=ctr,
                c_theta=np.asarray(c_theta, np.float64), c_resp=np.asarray(c_resp, np.float64),
                d_theta=np.asarray(d_theta, np.float64),
                d_rot=np.stack([np.asarray(by_theta[t], np.float64) for t in d_theta]))


# ---- numbers (CPU) ----------------------------------------------------------------------------- #
def base_norms_from_log(text: str) -> dict:
    """{episode: |base activation| at the landmark, as logged to 1 decimal} from the log of a
    one-shot steering run (lines 'ep<E> h<H> seed<S>: landmark ... \u2016base_act\u2016=<norm>')."""
    out = {}
    for m in re.finditer(r"ep(\d+) h(\d+) seed\d+: landmark .*?\u2016base_act\u2016=([\d.]+)", text):
        out[int(m.group(1))] = float(m.group(3))
    return out


def oneshot_numbers(rows) -> dict:
    """Compensation ratio -response / theta over the panel-c rows: mean, s.d. (ddof 0 and 1), the
    same over all scenes (`_all`, the left-out scenes included), per written angle (mean response,
    ratio), count of rows whose response opposes the written angle, the left-out scenes
    (`drift_episodes`) and the unedited chunk rotation |w| x 8 of every scene."""
    drift = drift_episodes(rows)
    dots = [r for r in rows if r["tag"] == "chart" and r["axis"] == "z" and r["theta_deg"] != 0]
    keep = [r for r in dots if r["ep"] not in drift]
    ratio = np.array([-oneshot_response(r) / r["theta_deg"] for r in keep])
    ratio_all = np.array([-oneshot_response(r) / r["theta_deg"] for r in dots])
    per = {}
    for th in (-40.0, -20.0, 20.0, 40.0):
        v = [oneshot_response(r) for r in keep if r["theta_deg"] == th]
        per[th] = dict(mean_resp=float(np.mean(v)), n=len(v), ratio=float(np.mean([-x / th for x in v])))
    opposing = int(sum(np.sign(oneshot_response(r)) == -np.sign(r["theta_deg"]) for r in keep))
    noop = noop_chunk_rotation(rows)
    return dict(n_dots=len(keep), n_scenes=len({r["ep"] for r in keep}), opposing=opposing,
                comp_mean=float(ratio.mean()), comp_sd0=float(ratio.std(ddof=0)),
                comp_sd1=float(ratio.std(ddof=1)), comp_mean_all=float(ratio_all.mean()),
                comp_sd_all=float(ratio_all.std(ddof=0)), per_angle=per,
                left_out=drift, drift_max_deg=DRIFT_MAX_DEG,
                noop_chunk_rot_deg={int(k): float(v) for k, v in sorted(noop.items())})


def edit_size(rows, base_norms: dict) -> dict:
    """Edit norm / |base activation| over the one-shot chart rows about z (theta != 0) of the scenes
    in `base_norms`: mean and s.d. (ddof 0) over all of them (`_all`), over the scenes panel c keeps
    (`_kept`, without `drift_episodes`), and per |theta|."""
    drift = drift_episodes(rows)
    dots = [r for r in rows if r["tag"] == "chart" and r["axis"] == "z" and r["theta_deg"] != 0
            and r["ep"] in base_norms]
    if not dots:
        return dict(note="no base activation norms")
    f = np.array([r["delta_norm"] / base_norms[r["ep"]] for r in dots])
    fk = np.array([r["delta_norm"] / base_norms[r["ep"]] for r in dots if r["ep"] not in drift])
    by = {th: float(np.mean([r["delta_norm"] / base_norms[r["ep"]] for r in dots
                             if abs(r["theta_deg"]) == th])) for th in (20.0, 40.0)}
    return dict(frac_mean_all=float(f.mean()), frac_sd_all=float(f.std(ddof=0)), n_all=len(f),
                frac_mean_kept=float(fk.mean()), frac_sd_kept=float(fk.std(ddof=0)),
                n_kept=len(fk), frac_by_abs_theta=by, base_norms=base_norms)


def persist_numbers(rows, house_d: int = HOUSE_D) -> dict:
    """Held edits: per written angle the mean and s.d. of net = final rotation - reference final
    rotation of the same scene, and the ratio -mean(net) / theta; sign-correct count; scenes; house
    `house_d`'s raw and net final rotations (`fig6d_example`, the house of panel d); EE drift of
    edited and reference rollouts."""
    ref = {r["ep"]: r["final_rot_deg"] for r in rows if r["tag"] == "persist_noop"}
    ed = [r for r in rows if r["tag"] == "persist"]
    out = {}
    for th in (-40.0, -20.0, 20.0, 40.0):
        nets = np.array([r["final_rot_deg"] - ref[r["ep"]] for r in ed if r["theta_deg"] == th])
        out[th] = dict(net_mean=float(nets.mean()), net_sd=float(nets.std(ddof=0)), n=len(nets),
                       ratio=float(-nets.mean() / th))
    sign_ok = int(sum(np.sign(r["final_rot_deg"] - ref[r["ep"]]) == -np.sign(r["theta_deg"]) for r in ed))
    per_cond = [-(r["final_rot_deg"] - ref[r["ep"]]) / r["theta_deg"] for r in ed]
    h = {r["theta_deg"]: r["final_rot_deg"] for r in rows if r["house"] == house_d}
    h_ref = h.get(0.0, 0.0)
    d6 = {th: dict(raw=float(h[th]), net=float(h[th] - h_ref), ratio_raw=float(-h[th] / th),
                   ratio_net=float(-(h[th] - h_ref) / th))
          for th in (-40.0, -20.0, 20.0, 40.0) if th in h}
    drift_ed = np.array([r["pos_drift"] for r in ed])
    drift_ref = {r["ep"]: r["pos_drift"] for r in rows if r["tag"] == "persist_noop"}
    return dict(per_angle=out,
                ratio_range=[float(min(v["ratio"] for v in out.values())),
                             float(max(v["ratio"] for v in out.values()))],
                sign_correct=sign_ok, n_cond=len(ed),
                per_cond_ratio_range=[float(min(per_cond)), float(max(per_cond))],
                scenes=sorted({int(r["house"]) for r in ed}),
                fig6d_example=dict(house=int(house_d), final_rot=d6),
                fig6d_ref_final=float(h_ref), pos_drift_edited_mean=float(drift_ed.mean()),
                pos_drift_ref_mean=float(np.mean(list(drift_ref.values()))),
                n_edited_drift_gt_ref=int(sum(r["pos_drift"] > drift_ref[r["ep"]] for r in ed)))


def calibration(chart: Chart, files: list, layer: int = LAYER, seed: int = SPLIT_SEED) -> dict:
    """Degrees on the z circle per physical degree: for every z sweep of a held-out house spanning
    more than 120 deg, |slope| of the unwrapped code phase (pair 4, 5) against the cumulative
    rotation about base z; median and interquartile range over the sweeps."""
    import torch
    U = {0: np.array([1.0, 0, 0]), 1: np.array([0, 1.0, 0]), 2: np.array([0, 0, 1.0])}
    X, house, meta, fibers = [], [], [], defaultdict(list)
    for f in files:
        d = np.load(f, allow_pickle=True)
        li = d["layers"].tolist().index(layer)
        A = d["act"][:, li, :].astype(np.float32)
        for i in range(len(A)):
            k = len(X)
            X.append(A[i])
            house.append(int(d["house"]))
            meta.append((int(d["step_k"][i]), d["R"][i].astype(np.float64)))
            fibers[(int(d["ep"]), int(d["pose_id"][i]), int(d["axis_id"][i]))].append(k)
    X = np.stack(X)
    house = np.array(house)
    rng = np.random.default_rng(seed)
    uh = np.unique(house)
    rng.shuffle(uh)
    test_h = set(uh[: len(uh) // 3].tolist())
    with torch.no_grad():
        Z = chart.enc(torch.tensor((X - chart.mean) / chart.scale)).numpy()
    slopes = []
    for (_ep, _pose, axis), idxs in fibers.items():
        if axis != 2 or house[idxs[0]] not in test_h:
            continue
        idxs = sorted(idxs, key=lambda k: meta[k][0])
        cum = [0.0]
        for a_, b_ in zip(idxs[:-1], idxs[1:]):
            th, v = _axis_angle(meta[a_][1].T @ meta[b_][1])
            cum.append(cum[-1] + (1.0 if float((meta[a_][1] @ v) @ U[2]) >= 0 else -1.0) * th)
        cum = np.degrees(np.array(cum))
        if cum.max() - cum.min() <= 120.0:
            continue
        ph = np.degrees(np.unwrap(np.arctan2(Z[idxs, 5], Z[idxs, 4])))
        slopes.append(abs(np.polyfit(cum, ph, 1)[0]))
    s = np.array(slopes)
    return dict(n_eps=len(files), n_houses=int(len(uh)), n_test_houses=len(test_h),
                n_fibers=len(s), slope_median=float(np.median(s)) if len(s) else None,
                slope_iqr=[float(np.percentile(s, 25)), float(np.percentile(s, 75))] if len(s) else None)


def fig6_numbers(oneshot_rows, persist_rows, base_norms: dict, chart: Chart, orient: list,
                 report: dict) -> dict:
    """Every number of the figure and its text: `oneshot_numbers`, `edit_size`, `persist_numbers`,
    `calibration` and the chart report (held-out reconstruction R^2, decode-transport R^2, radii)."""
    return dict(oneshot=oneshot_numbers(oneshot_rows), edit_size=edit_size(oneshot_rows, base_norms),
                persist=persist_numbers(persist_rows), calibration=calibration(chart, orient),
                chart_report={k: report[k] for k in ("recon_r2_test", "decode_transport_r2_test",
                                                     "radii") if k in report})


def read_json(path):
    with open(path) as f:
        return json.load(f)
