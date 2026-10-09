"""Task phase (Figure 5): pick-and-place rollouts, phase labels, the freeze test, the trajectory
capture and the arrays the Figure 5 panels are drawn from.

The task is MolmoSpaces pick-and-place ("Pick up the X and place it in or on the Y",
`FrankaPickAndPlaceDroidDataGenConfig`). Every query step of a rollout gets one of four phase
labels:

  reach      from the start until 4 steps before the first commanded gripper close
  grasp      from there until the object is lifted in the gripper run that ends at the placing
             release
  transport  the carry: from that lift until the final descent of the held object begins
  place      the final descent, the release and the rest of the episode

(`label_steps`; episodes that never place have no place segment). A rollout stops at the first step
after which MolmoSpaces judges the task successful.

Parts of this module:

  lists                read_natural_episodes, read_freeze_episodes, read_trajectory_episode,
                       read_trajectory_preceding
  labels               label_steps, labels_at_queries, time_bins
  rollout recording    install_recorder, finalize_record, rollout, label_queries
  activations          encode_context, encode_all_queries (action expert and four VLM token pools)
  natural rollouts     collect_episode, build_cache
  decoders             lda_cv, natural_decoder, decode_composition
  freeze test          pick_landmarks, encode_fresh_query, advance_policy, perturb_landmark,
                       run_freeze_episode, freeze_payload; used_landmarks, load_freeze_rows,
                       load_gripper_chunks (the landmarks the analyses read)
  trajectory capture   install_frame_recorder, capture_episode
  panel arrays         fit_probes, latent_colours, pick_columns, pca_embed, lda_embedding,
                       own_phase_data, gripper_delta_data

Numpy only at import; the simulator, torch, scikit-learn and matplotlib are imported by the
functions that need them.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

import numpy as np

from vlageom import paths

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------- constants
CONFIG = "FrankaPickAndPlaceDroidDataGenConfig"
MAX_STEPS = 520                      # task horizon 500 + slack
SEED_BASE = 20_000_000               # natural rollouts: seed = SEED_BASE + 1000 * round + house

REACH, GRASP, TRANSPORT, PLACE = 0, 1, 2, 3
N_PHASE = 4
PHASE_NAMES = {REACH: "reach", GRASP: "grasp", TRANSPORT: "transport", PLACE: "place"}
NAMES = [PHASE_NAMES[k] for k in range(N_PHASE)]
PHASE_COLORS = {REACH: "#2563eb", GRASP: "#8b5cf6", TRANSPORT: "#f43f5e", PLACE: "#f59e0b"}

AE_LAYERS = [0, 4, 8, 9, 12, 16, 20, 24, 28, 32, 35]      # action-expert blocks read per query
LLM_LAYERS = [21, 24, 27, 29, 31, 33, 34, 35]              # VLM blocks read per query
POOLS = ("vis", "lang", "tail", "visprev")                 # VLM token pools (see encode_context)
LAYER = 24          # action-expert block of every Figure 5 panel (the best layer of stage 30)

# 4-phase labels at query resolution (one query = 8 env steps): no reach backoff, the settle
# window in queries, and the descent tolerance per query (8 x the per-step tolerance).
QUERY_LABEL_KW = dict(reach_backoff=0, settle_win=5, descent_tol=0.016)

# freeze test
PAUSE_KS = (1, 2, 4)                 # number of extra model queries on the held observation
CONDS = ("entry",) + tuple(f"pause{k}" for k in PAUSE_KS) + ("detour", "off")
KIND_CODE = {"reach": 0, "grasp_open": 1, "grasp_closed": 2, "transport": 3,
             "place_closed": 4, "place_released": 5}
KINDS = {v: k for k, v in KIND_CODE.items()}
N_REACH = 3                          # reach landmarks per episode
DETOUR_STEPS = 8                     # detour: excursion in this many arm-pose steps ...
DETOUR_MAG = 0.05                    # ... to this EE offset (m) along base x
DETOUR_AXIS = np.array([1.0, 0.0, 0.0])
MIN_PX = 3.0                         # a rendered camera frame darker than this (mean px) is blank
GRIP = 7                             # gripper channel of an action chunk
EXECUTED = 8                         # steps of a chunk that are executed before the next query

# panel conditions of the freeze test: natural = the landmark's own two frames, frozen = the
# current frame shown twice (two extra queries on the held observation)
COND_NATURAL, COND_FROZEN = "entry", "pause2"
PANEL_KINDS = [("reach", (0,)), ("grasp", (1, 2)), ("transport", (3,)), ("place", (4,))]

CAPTURE_CAMS = ("exo_camera_1", "wrist_camera")


# ----------------------------------------------------------------------------------- lists
def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_natural_episodes(path=None) -> list[dict]:
    """Rows of lists/phase/natural_episodes.csv: episode_index, partition, house, seed, new_sampler
    (ints), in file order (episode order)."""
    path = Path(path) if path else paths.lists_dir() / "phase" / "natural_episodes.csv"
    return [{k: int(v) for k, v in r.items()} for r in _rows(path)]


def read_freeze_episodes(path=None) -> list[dict]:
    """Rows of lists/phase/freeze_episodes.csv: task, episode_index, house, seed, decode (ints)."""
    path = Path(path) if path else paths.lists_dir() / "phase" / "freeze_episodes.csv"
    return [{k: int(v) for k, v in r.items()} for r in _rows(path)]


def read_trajectory_episode(path=None) -> dict:
    """The single row of lists/phase/trajectory_episode.csv: episode_index, house, seed,
    new_sampler."""
    path = Path(path) if path else paths.lists_dir() / "phase" / "trajectory_episode.csv"
    rows = _rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one row, found {len(rows)}")
    return {k: int(v) for k, v in rows[0].items()}


def read_trajectory_preceding(path=None) -> list:
    """The rows of lists/phase/trajectory_preceding.csv, in order: the episodes rolled out in the
    same process before the example rollout (episode_index, house, seed, new_sampler)."""
    path = Path(path) if path else paths.lists_dir() / "phase" / "trajectory_preceding.csv"
    return [{k: int(v) for k, v in r.items()} for r in _rows(path)]


def natural_name(episode_index: int, house: int) -> str:
    return f"ep{episode_index:05d}_h{house}.npz"


def freeze_name(episode_index: int, house: int) -> str:
    return f"ep{episode_index}_h{house}.npz"


# ----------------------------------------------------------------------------------- labels
def label_steps(grip, objz, supported, *, success: bool = False, close_thr: float = 127.5,
                reach_backoff: int = 4, lift_thr: float = 0.003, settle_win: int = 40,
                descent_tol: float = 0.002, descent_min: float = 0.010, end_slack: int = 1) -> dict:
    """Per-step phase labels of one episode.

    grip       commanded gripper per step (0 = open .. 255 = closed)
    objz       height of the pickup object per step (m, world frame)
    supported  per step: the object is supported by the target receptacle (task info
               `supported_by_receptacle`, read before the step)
    success    the episode's final success judgement

    A step is closed when grip > close_thr. reach = [0, grasp onset) with grasp onset = first
    close - reach_backoff. The placing release r* is, in this order: the last close -> open edge
    followed by support within settle_win steps; for a successful episode, the last such edge
    within settle_win steps of the end; for a successful episode without one, the first supported
    step after the first close, else the last step. The committed close is the last open -> close
    edge whose closed run reaches r*; transport starts at the first step of that run where the
    object is above its pre-grasp height + lift_thr. place starts where the object's final
    descent into r* begins (walking back from r* while the height does not rise by more than
    descent_tol per step), or at r* when that descent is shorter than descent_min; it is kept
    after the transport onset. Without r* (no placing release), transport is the last close
    whose run holds to the end (within end_slack) once the object is lifted, and there is no
    place segment. An episode that never closes is all reach.

    Returns dict(label (int8 per step), grasp_onset, transport_onset, place_onset,
    release_step (None when absent), attempts (number of open -> close edges), has_transport,
    has_place, descent_m)."""
    grip = np.asarray(grip, float)
    objz = np.asarray(objz, float)
    sup = np.asarray(supported, bool)
    n = len(grip)
    lab = np.full(n, REACH, np.int8)
    out = {"label": lab, "grasp_onset": None, "transport_onset": None, "place_onset": None,
           "release_step": None, "attempts": 0, "has_transport": False, "has_place": False,
           "descent_m": float("nan")}
    closed = grip > close_thr
    if n == 0 or not closed.any():
        return out
    first_close = int(np.argmax(closed))
    grasp_onset = max(0, first_close - reach_backoff)
    zbase = float(objz[:first_close + 1].min())
    edges = list(np.where((~closed[:-1]) & closed[1:])[0] + 1)
    if closed[0]:
        edges = [0] + edges
    attempts = len(edges)
    rels = list(np.where(closed[:-1] & (~closed[1:]))[0] + 1)

    r_star = None
    for r in reversed(rels):
        if sup[r:min(n, r + settle_win)].any():
            r_star = r
            break
    if r_star is None and success and rels and rels[-1] >= n - settle_win:
        r_star = rels[-1]
    if r_star is None and success:
        s_after = np.where(sup[first_close:])[0]
        r_star = first_close + int(s_after[0]) if len(s_after) else n - 1

    trans = None
    place = None
    if r_star is not None:
        e_star = None
        for e in reversed(edges):
            if e >= r_star:
                continue
            j = e
            while j < n and closed[j]:
                j += 1
            if j >= r_star:
                e_star = e
                break
        if e_star is not None:
            lifted = np.where(closed[e_star:r_star] & (objz[e_star:r_star] > zbase + lift_thr))[0]
            if len(lifted):
                trans = e_star + int(lifted[0])
        p = r_star
        t = r_star - 1
        lo = trans + 1 if trans is not None else (e_star if e_star is not None else grasp_onset)
        while t >= lo and objz[t + 1] <= objz[t] + descent_tol:
            p = t
            t -= 1
        seg = objz[p:r_star + 1]
        descent = float(seg.max() - objz[r_star]) if len(seg) else 0.0
        place = p if descent >= descent_min else r_star
        if trans is not None and place <= trans:
            place = trans + 1
        out["descent_m"] = descent
    else:
        for e in reversed(edges):
            j = e
            while j < n and closed[j]:
                j += 1
            if (j - 1) < n - 1 - end_slack:
                continue
            lifted = np.where(closed[e:] & (objz[e:] > zbase + lift_thr))[0]
            if len(lifted):
                trans = e + int(lifted[0])
            break

    lab[grasp_onset:] = GRASP
    if trans is not None:
        lab[trans:] = TRANSPORT
    if place is not None and place < n:
        lab[place:] = PLACE
    out.update({"grasp_onset": grasp_onset, "transport_onset": trans, "place_onset": place,
                "release_step": r_star, "attempts": attempts,
                "has_transport": trans is not None, "has_place": place is not None})
    return out


def labels_at_queries(query_steps, step_label) -> np.ndarray:
    """The per-step labels at each query's step (clamped to the episode), int8."""
    qs = np.clip(np.asarray(query_steps, int), 0, len(step_label) - 1)
    return step_label[qs].astype(np.int8)


def time_bins(phase, n_bins: int = N_PHASE) -> np.ndarray:
    """Equal-time bins (int8 0..n_bins-1) of the normalised episode time `phase` in [0, 1]."""
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    bins = np.digitize(np.asarray(phase, float), edges[1:-1])
    return np.clip(bins, 0, n_bins - 1).astype(np.int8)


# ----------------------------------------------------------------------------------- recording
_REC_KEYS = ("grip", "objz", "objx", "objy", "supported", "robot_contact",
             "pos_err", "success", "contact", "eex", "eey", "eez")
_REC_FLOAT = ("grip", "objz", "objx", "objy", "pos_err", "eex", "eey", "eez")


def _gripper_cmd(action) -> float:
    from vlageom.sim.engine import gripper_of
    try:
        return float(gripper_of(action))
    except Exception:
        try:
            a = action[0] if isinstance(action, (list, tuple)) else action
            return float(np.asarray(a["gripper"]).ravel()[0])
        except Exception:
            return float("nan")


def install_recorder(eng, *, stop_on_success: bool = True) -> dict:
    """Wrap `eng.task.step` to record, before every step, the commanded gripper, the object
    position, the task info (supported_by_receptacle, robot_contact, position_error, success),
    gripper-object contact and the EE position; with stop_on_success, `task.is_done()` turns true
    after the first step at which `task.judge_success()` is true. Install after `eng.sample` and
    before `eng.run_baseline`. Returns the record (lists, filled as the rollout runs)."""
    task = eng.task
    rec = {k: [] for k in _REC_KEYS}
    rec["stopped_on_success"] = False
    orig_step = task.step
    orig_is_done = task.is_done

    def step(action, *a, **kw):
        rec["grip"].append(_gripper_cmd(action))
        obj = eng.read_object()
        if len(obj) == 3:
            rec["objx"].append(float(obj[0]))
            rec["objy"].append(float(obj[1]))
            rec["objz"].append(float(obj[2]))
        else:
            for k in ("objx", "objy", "objz"):
                rec[k].append(float("nan"))
        try:
            ti = task.get_info()[0]
        except Exception:
            ti = {}
        rec["supported"].append(bool(ti.get("supported_by_receptacle", False)))
        rec["robot_contact"].append(bool(ti.get("robot_contact", False)))
        rec["pos_err"].append(float(ti.get("position_error", np.nan)))
        rec["success"].append(bool(ti.get("success", False)))
        rec["contact"].append(bool(eng._read_grasp()))
        ee = eng.read_ee()
        if len(ee) == 3:
            rec["eex"].append(float(ee[0]))
            rec["eey"].append(float(ee[1]))
            rec["eez"].append(float(ee[2]))
        else:
            for k in ("eex", "eey", "eez"):
                rec[k].append(float("nan"))
        out = orig_step(action, *a, **kw)
        if stop_on_success and not rec["stopped_on_success"]:
            try:
                if task.judge_success():
                    rec["stopped_on_success"] = True
            except Exception:
                pass
        return out

    def is_done():
        return rec["stopped_on_success"] or orig_is_done()

    task.step = step
    task.is_done = is_done
    return rec


def finalize_record(rec: dict, base: dict) -> dict:
    """Per-step arrays `step_<key>` and their values at the query steps (`<key>_q` for grip,
    objz, supported, robot_contact, pos_err, contact; record index i = state before step i)."""
    out = {}
    for k in _REC_KEYS:
        out[f"step_{k}"] = np.asarray(rec[k], float if k in _REC_FLOAT else bool)
    qs = np.asarray(base["query_steps"], int)
    n = len(out["step_grip"])
    qs_c = np.clip(qs, 0, max(n - 1, 0))
    for k in ("grip", "objz", "supported", "robot_contact", "pos_err", "contact"):
        out[f"{k}_q"] = out[f"step_{k}"][qs_c] if n else np.array([])
    out["stopped_on_success"] = bool(rec["stopped_on_success"])
    return out


def rollout(eng, max_steps: int = MAX_STEPS, *, stop_on_success: bool = True):
    """`install_recorder` + `eng.run_baseline(max_steps)`; returns (base, record arrays)."""
    rec = install_recorder(eng, stop_on_success=stop_on_success)
    base = eng.run_baseline(max_steps)
    return base, finalize_record(rec, base)


def label_queries(base: dict, rec: dict):
    """Phase labels at query resolution of the rollout itself (`label_steps` on the per-query
    gripper, object height and support with QUERY_LABEL_KW), or None without queries. Missing
    object heights are filled with the median of the others."""
    grip = np.asarray([float(g) for g in base["grip_series"]], float)
    if grip.size == 0:
        return None
    objz = np.asarray(rec["objz_q"], float)
    if not np.isfinite(objz).all():
        fin = np.isfinite(objz)
        if not fin.any():
            return None
        objz = np.where(fin, objz, np.nanmedian(objz[fin]))
    sup = np.asarray(rec["supported_q"], bool)
    return label_steps(grip, objz, sup, success=bool(base.get("success", False)), **QUERY_LABEL_KW)


# ----------------------------------------------------------------------------------- activations
def encode_context(eng, kw: dict, seed: int, ae_layers=None, llm_layers=None) -> dict:
    """Activations of one captured `generate_actions` context (no new rollout step, no render).

    ae          (len(ae_layers), 768): output of action_expert.blocks.{L} at the final flow step
                (flow noise seeded by `seed`), mean over the 16 action tokens
    llm_vis     (len(llm_layers), 2560): transformer.blocks.{L} output, mean over the patches of
                the two current camera frames
    llm_lang    mean over every token that is not an image patch
    llm_tail    mean over the text after the last image block
    llm_visprev mean over the patches of the earlier frames (NaN on the first query)
    A pool whose token parse fails is NaN."""
    from vlageom.sim.encode import token_map
    ae_layers = ae_layers or AE_LAYERS
    llm_layers = llm_layers or LLM_LAYERS
    llm_raw, hooks = {}, []

    def mk_hook(L):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            llm_raw[L] = t.detach().float()[0].cpu().numpy()
        return hook

    for L in llm_layers:
        block = eng.model.get_submodule(f"transformer.blocks.{L}")
        hooks.append(block.register_forward_hook(mk_hook(L)))
    try:
        cache = eng._build_backbone_cache(kw)
    finally:
        for h in hooks:
            h.remove()
    prefix = eng._compute_prefix(cache, seed)
    ae = np.stack([eng._final_step(cache, prefix, None, L, grad=False)[1]
                   .detach().float().cpu().numpy() for L in ae_layers]).astype(np.float32)

    d = llm_raw[llm_layers[0]].shape[-1]
    out = {"ae": ae}
    try:
        ids = np.asarray(kw["input_ids"].detach().cpu().numpy())[0]
        tm = token_map(ids)
        seq = tm["seq_len"]
        vis_idx = np.concatenate([np.arange(s, s + ln)
                                  for s, ln in (tm["cur_exo"], tm["cur_wrist"])])
        patch_idx = np.concatenate([np.arange(s, s + ln) for s, ln in tm["runs"]])
        lang_idx = np.setdiff1d(np.arange(seq), patch_idx)
        tail_idx = np.arange(tm["tail_start"], seq)
        prev = tm["runs"][:-2]
        prev_idx = np.concatenate([np.arange(s, s + ln) for s, ln in prev]) if prev else None
        pools = (("vis", vis_idx), ("lang", lang_idx), ("tail", tail_idx), ("visprev", prev_idx))
        for name, idx in pools:
            if idx is None or len(idx) == 0:
                out[f"llm_{name}"] = np.full((len(llm_layers), d), np.nan, np.float32)
            else:
                out[f"llm_{name}"] = np.stack([llm_raw[L][idx].mean(0)
                                               for L in llm_layers]).astype(np.float32)
    except Exception as e:
        log.warning("token parse failed (%s); VLM pools NaN for this query", e)
        for name in POOLS:
            out[f"llm_{name}"] = np.full((len(llm_layers), d), np.nan, np.float32)
    return out


def encode_all_queries(eng, ae_layers=None, llm_layers=None) -> dict:
    """`encode_context` for every query of the last `eng.run_baseline`, query q with flow seed
    scene_seed + 1000 q. Returns ae (Q, len(ae_layers), 768) and llm_<pool> (Q, len(llm_layers),
    2560), NaN rows for queries that failed to encode."""
    ae_layers = ae_layers or AE_LAYERS
    llm_layers = llm_layers or LLM_LAYERS
    Q = len(eng._captured)
    ae = np.full((Q, len(ae_layers), 768), np.nan, np.float32)
    llm = {p: None for p in POOLS}
    for q in range(Q):
        seed = eng.scene_seed + 1000 * q
        try:
            enc = encode_context(eng, eng._captured[q], seed, ae_layers, llm_layers)
        except Exception as e:
            log.warning("encode failed at query %d: %s", q, e)
            continue
        ae[q] = enc["ae"]
        for p in POOLS:
            v = enc[f"llm_{p}"]
            if llm[p] is None:
                llm[p] = np.full((Q,) + v.shape, np.nan, np.float32)
            llm[p][q] = v
    out = {"ae": ae}
    for p in POOLS:
        out[f"llm_{p}"] = (llm[p] if llm[p] is not None
                           else np.full((Q, len(llm_layers), 2560), np.nan, np.float32))
    return out


def free_episode(eng) -> None:
    """Drop the episode's policy, model and per-step snapshots before the next episode (the
    policy builds a new model for every episode)."""
    import gc

    import torch
    eng._captured = []
    eng._baseline_snaps = []
    eng.task = eng.policy = eng.model = None
    eng._gen_orig = eng._gen_unwrapped = None
    if getattr(eng, "sim_env", None) is not None:
        for attr in ("_task", "_policy"):
            if hasattr(eng.sim_env, attr):
                setattr(eng.sim_env, attr, None)
    gc.collect()
    torch.cuda.empty_cache()


def sample_listed(eng, house: int, seed: int, new_sampler: int) -> None:
    """Sample one listed episode (see `vlageom.gripper.sample_episode`: new_sampler=1 samples it
    with a task sampler created for it, new_sampler=0 with an existing one)."""
    from vlageom.gripper import sample_episode
    sample_episode(eng, house, seed, new_sampler)


# ----------------------------------------------------------------------------------- natural
def collect_episode(eng, episode_index: int, house: int, seed: int, new_sampler: int,
                    max_steps: int = MAX_STEPS):
    """Roll out one natural episode and read its activations. Returns (payload for np.savez,
    summary dict), or (None, reason) when no query fired."""
    sample_listed(eng, house, seed, new_sampler)
    task_desc = eng.task.get_task_description()
    tcfg = eng.task.config.task_config
    base, rec = rollout(eng, max_steps)
    Q = len(base["grip_series"])
    if Q == 0:
        return None, "no queries fired"
    acts = encode_all_queries(eng)
    qpos_arm = np.stack([(q if q is not None else np.full(7, np.nan, np.float32))
                         for q in base["qpos_arm"]]).astype(np.float32)
    ee_pos = np.stack([(e if e is not None else np.full(3, np.nan, np.float32))
                       for e in base["ee_pos"]]).astype(np.float32)
    payload = {
        "ae": acts["ae"],
        **{f"llm_{p}": acts[f"llm_{p}"] for p in POOLS},
        "ae_layers": np.array(AE_LAYERS, np.int32),
        "llm_layers": np.array(LLM_LAYERS, np.int32),
        "query_steps": np.asarray(base["query_steps"], np.int32),
        "grip_q": np.asarray([float(g) for g in base["grip_series"]], np.float32),
        "objz_q": np.asarray(rec["objz_q"], np.float32),
        "supported_q": np.asarray(rec["supported_q"], bool),
        "robot_contact_q": np.asarray(rec["robot_contact_q"], bool),
        "pos_err_q": np.asarray(rec["pos_err_q"], np.float32),
        "contact_q": np.asarray(rec["contact_q"], bool),
        "qpos_arm": qpos_arm, "ee_q": ee_pos,
        **{k: rec[k] for k in rec if k.startswith("step_")},
        "n_steps": np.int32(base["n_steps"]), "n_queries": np.int32(Q),
        "success": bool(base["success"]),
        "success_any": bool(rec["step_success"].any()),
        "first_success_step": np.int32(int(np.argmax(rec["step_success"]))
                                       if rec["step_success"].any() else -1),
        "stopped_on_success": bool(rec["stopped_on_success"]),
        "ep": np.int32(episode_index), "house": np.int32(house), "seed": np.int64(seed),
        "task_description": str(task_desc),
        "pickup_obj_name": str(getattr(tcfg, "pickup_obj_name", "")),
        "place_receptacle_name": str(getattr(tcfg, "place_receptacle_name", "")),
        "place_receptacle_start_pose": np.asarray(
            getattr(tcfg, "place_receptacle_start_pose", np.full(7, np.nan)), np.float32),
    }
    info = label_steps(rec["step_grip"], rec["step_objz"], rec["step_supported"],
                       success=bool(base["success"]))
    summary = dict(ep=episode_index, house=house, seed=seed, n_steps=int(base["n_steps"]), n_q=Q,
                   success=bool(base["success"]),
                   labels={NAMES[k]: int((info["label"] == k).sum()) for k in range(N_PHASE)},
                   desc=task_desc)
    return payload, summary


_CACHE_QCOLS = {"gripper": "grip_q", "objz": "objz_q", "contact": "contact_q",
                "supported": "supported_q", "robot_contact": "robot_contact_q",
                "pos_err": "pos_err_q"}
_CACHE_EPMETA = ("success", "success_any", "stopped_on_success", "house", "seed", "n_queries")


def build_cache(files) -> tuple[dict, dict]:
    """One row per query of the natural episode files `files` (in the given order), with the
    action-expert activations and the step-resolution phase labels at the query steps.

    Keys: X (N, 11, 768) float32, label, label_time (4 equal-time bins), phase (query step /
    (n_steps - 1)), q, ep, gripper, objz, contact, supported, robot_contact, pos_err, success,
    success_any, stopped_on_success, house, seed, grasp_onset, transport_onset, place_onset,
    release_step (steps, -1 if absent), attempt_count, has_transport, has_place, descent_m,
    regime (episode has >= 30 queries), pickup_obj_name, place_receptacle_name, layers, n_q.
    Queries whose activations are not finite are left out. Returns (arrays, counts)."""
    import collections
    rows = collections.defaultdict(list)
    layers = None
    n_drop = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        X = np.asarray(z["ae"], np.float32)
        if layers is None:
            layers = np.asarray(z["ae_layers"], int)
        Q = X.shape[0]
        info = label_steps(np.asarray(z["step_grip"], float), np.asarray(z["step_objz"], float),
                           np.asarray(z["step_supported"], bool),
                           success=bool(np.asarray(z["success"]).item()))
        qsteps = np.asarray(z["query_steps"], int)
        lab_q = labels_at_queries(qsteps, info["label"])
        n_steps = int(z["n_steps"])
        phase = qsteps / max(n_steps - 1, 1)
        fin = np.isfinite(X.reshape(Q, -1)).all(1)
        n_drop += int((~fin).sum())
        idx = np.where(fin)[0]
        if len(idx) == 0:
            continue
        rows["X"].append(X[idx])
        rows["label"].append(lab_q[idx])
        rows["label_time"].append(time_bins(phase[idx]))
        rows["phase"].append(phase[idx].astype(np.float32))
        rows["q"].append(idx.astype(np.int32))
        rows["ep"].append(np.full(len(idx), int(z["ep"]), np.int32))
        for col, src in _CACHE_QCOLS.items():
            rows[col].append(np.asarray(z[src])[idx])
        for k in _CACHE_EPMETA:
            rows[k].append(np.full(len(idx), np.asarray(z[k]).item()))
        for k in ("grasp_onset", "transport_onset", "place_onset", "release_step"):
            v = info[k]
            rows[k].append(np.full(len(idx), -1 if v is None else int(v), np.int32))
        rows["attempt_count"].append(np.full(len(idx), info["attempts"], np.int16))
        rows["has_transport"].append(np.full(len(idx), info["has_transport"], bool))
        rows["has_place"].append(np.full(len(idx), info["has_place"], bool))
        rows["descent_m"].append(np.full(len(idx), info["descent_m"], np.float32))
        rows["regime"].append(np.full(len(idx), Q >= 30, bool))
        rows["pickup_obj_name"].append(np.array([str(z["pickup_obj_name"])] * len(idx)))
        rows["place_receptacle_name"].append(np.array([str(z["place_receptacle_name"])] * len(idx)))
    d = {k: np.concatenate(v) for k, v in rows.items()}
    d["layers"] = layers
    d["n_q"] = d.pop("n_queries").astype(np.int32)
    d["house"] = d["house"].astype(np.int32)
    d["seed"] = d["seed"].astype(np.int64)
    for k in ("success", "success_any", "stopped_on_success"):
        d[k] = d[k].astype(bool)
    succ_eps = np.unique(d["ep"][d["success"]])
    counts = {"rows": int(len(d["label"])), "episodes": int(len(np.unique(d["ep"]))),
              "houses": int(len(np.unique(d["house"]))), "success_episodes": int(len(succ_eps)),
              "success_rows": int(d["success"].sum()),
              "success_houses": int(len(np.unique(d["house"][d["success"]]))),
              "dropped_nonfinite_queries": n_drop,
              "labels": {NAMES[k]: int((d["label"] == k).sum()) for k in range(N_PHASE)}}
    return d, counts


def load_cache(path, keys=None) -> dict:
    """The arrays of a natural cache file (all keys, or `keys`)."""
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in (keys or z.files)}


# ----------------------------------------------------------------------------------- decoders
def fold_hash(fold_of_row) -> str:
    """First 12 hex digits of the SHA-1 of the per-row fold ids as int8 bytes."""
    return hashlib.sha1(np.asarray(fold_of_row).astype(np.int8).tobytes()).hexdigest()[:12]


def lda_cv(X, y, houses, folds) -> dict:
    """Four-class LDA (StandardScaler + LinearDiscriminantAnalysis, fit inside each fold) scored
    out of house over the fold map `folds` (`vlageom.probes.FoldMap`). Returns raw and balanced
    accuracy, majority-class chance, per-class recall and the confusion matrix."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    classes, counts = np.unique(y, return_counts=True)
    yt, yp = [], []
    for tr, te in folds.splits(houses):
        sc = StandardScaler().fit(X[tr])
        clf = LDA().fit(sc.transform(X[tr]), y[tr])
        yp.append(clf.predict(sc.transform(X[te])))
        yt.append(y[te])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    cm = confusion_matrix(yt, yp, labels=classes)
    rec = cm.diagonal() / cm.sum(1).clip(1)
    return {"raw": float((yt == yp).mean()), "bal": float(balanced_accuracy_score(yt, yp)),
            "maj": float(counts.max() / counts.sum()),
            "recall": {NAMES[int(c)]: float(r) for c, r in zip(classes, rec)},
            "confusion": cm.tolist(), "n": int(len(y))}


def natural_decoder(cache: dict, folds, layer: int = LAYER):
    """The phase decoder of the freeze test: on the successful natural queries at `layer`
    (float64), one StandardScaler + four-class LDA per fold of `folds`, each fit without that
    fold's houses. Returns (decoder_for_house, info): decoder_for_house(h) -> (scaler, lda) of the
    fold that holds out house h (a fit on all rows for a house outside the map); info has the
    out-of-fold balanced accuracy on the natural rows and the fold hash."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    li = list(cache["layers"]).index(layer)
    m = cache["success"] & (cache["label"] >= 0)
    X = np.asarray(cache["X"][m][:, li], np.float64)
    y, hs = cache["label"][m], cache["house"][m]
    splits = folds.splits(hs)
    fold_of = {int(h): k for k, (_tr, te) in enumerate(splits) for h in np.unique(hs[te])}
    fitted = {}

    def fit_wo(h):
        k = fold_of.get(int(h), -1)
        if k not in fitted:
            mask = np.ones(len(y), bool)
            if k != -1:
                mask[splits[k][1]] = False
            sc = StandardScaler().fit(X[mask])
            fitted[k] = (sc, LDA().fit(sc.transform(X[mask]), y[mask]))
        return fitted[k]

    yt, yp = [], []
    fa = np.zeros(len(y), np.int8)
    for k, (_tr, te) in enumerate(splits):
        sc, clf = fit_wo(int(hs[te][0]))
        yt.append(y[te])
        yp.append(clf.predict(sc.transform(X[te])))
        fa[te] = k
    info = {"layer": int(layer), "n_natural_rows": int(len(y)), "n_houses": int(len(np.unique(hs))),
            "natural_balanced_acc": float(balanced_accuracy_score(np.concatenate(yt),
                                                                  np.concatenate(yp))),
            "fold_hash": fold_hash(fa)}
    return fit_wo, info


def used_landmarks(kind, q) -> np.ndarray:
    """Boolean mask over the landmark rows of one freeze-test file (its `kind` and `q` arrays):
    the landmarks the analyses use. Of the closed-hand place landmarks of the episode only the one
    with the largest query index is used (the last query before the release); every other landmark
    (reach, open- and closed-hand grasp, transport, open-hand place) is used. The harvest records
    all landmarks; the decode set (panel c) and the gripper set (panel d) both read this mask."""
    kind = np.asarray(kind)
    keep = np.ones(len(kind), bool)
    idx = np.flatnonzero(kind == KIND_CODE["place_closed"])
    if len(idx) > 1:
        keep[idx] = False
        keep[idx[int(np.argmax(np.asarray(q)[idx]))]] = True
    return keep


def load_freeze_rows(files, layer: int = LAYER) -> dict:
    """Action-expert activations at `layer` (float64) of the landmarks in the freeze-test files
    `files` selected by `used_landmarks`, per condition, with kind, label, house and detour_ok per
    row. Rows whose entry or pause activations are not finite are dropped."""
    rows = {c: [] for c in CONDS}
    kind, label, det_ok, house, used = [], [], [], [], []
    n_files = 0
    for f in files:
        z = np.load(f)
        li = list(z["ae_layers"]).index(layer)
        P = z["kind"].shape[0]
        for c in CONDS:
            rows[c].append(np.asarray(z[f"ae_{c}"][:, li], np.float64))
        kind.append(z["kind"])
        label.append(z["label"])
        det_ok.append(z["detour_ok"])
        house.append(np.full(P, int(z["house"]), np.int32))
        used.append(used_landmarks(z["kind"], z["q"]))
        n_files += 1
    out = {c: np.concatenate(rows[c]) for c in CONDS}
    meta = dict(kind=np.concatenate(kind), label=np.concatenate(label),
                detour_ok=np.concatenate(det_ok), house=np.concatenate(house))
    fin = np.concatenate(used)
    for c in ("entry",) + tuple(f"pause{k}" for k in PAUSE_KS):
        fin &= np.isfinite(out[c]).all(1)
    out = {c: v[fin] for c, v in out.items()}
    out.update({k: v[fin] for k, v in meta.items()})
    out["n_files"], out["n_rows"] = n_files, int(fin.sum())
    return out


def decode_composition(rows: dict, decoder_for_house, conds=("entry", "pause2", "detour")) -> dict:
    """Decode every row with the fold model that holds out its house; per landmark kind and
    condition, the fraction of rows decoded as each phase and the row count n."""
    pred = {c: np.full(rows["n_rows"], -1, np.int8) for c in CONDS}
    for h in np.unique(rows["house"]):
        sel = rows["house"] == h
        sc, clf = decoder_for_house(int(h))
        for c in CONDS:
            V = rows[c][sel]
            fin = np.isfinite(V).all(1)
            if fin.any():
                o = np.full(int(sel.sum()), -1, np.int8)
                o[fin] = clf.predict(sc.transform(V[fin]))
                pred[c][sel] = o
    comp = {}
    for kd, name in KINDS.items():
        sel = rows["kind"] == kd
        if not sel.any():
            continue
        comp[name] = {}
        for c in conds:
            ok = sel & (pred[c] >= 0)
            if ok.any():
                comp[name][c] = {PHASE_NAMES[code]: float((pred[c][ok] == code).mean())
                                 for code in range(N_PHASE)}
                comp[name][c]["n"] = int(ok.sum())
    return comp


# ----------------------------------------------------------------------------------- freeze test
def pick_landmarks(base: dict, lab: dict, n_reach: int = N_REACH) -> list[tuple[int, str]]:
    """[(query, kind)] of one rollout from its query-resolution labels `lab` (`label_queries`):
    up to n_reach reach queries (the last reach query before the grasp left out, the rest spread
    evenly), the middle open-hand and the middle closed-hand grasp query (the first open-hand one
    when there is no closed-hand one and more than one open-hand one), the first and last transport
    queries, the first and last closed-hand place queries and the first open-hand place query.
    Only queries 1..Q-2 with a recorded arm pose and Jacobian are used."""
    grip = np.asarray([float(g) for g in base["grip_series"]], float)
    L, n = lab["label"], len(grip)
    qarm, ejac, qsteps = base["qpos_arm"], base["ee_jac"], base["query_steps"]

    def valid(c):
        return (1 <= c <= n - 2 and c < len(qarm) and qarm[c] is not None
                and c < len(ejac) and ejac[c] is not None and c < len(qsteps))

    reach = [c for c in range(n) if valid(c) and L[c] == REACH]
    if len(reach) > n_reach:
        reach = reach[:-1]
    if len(reach) > n_reach:
        idx = np.linspace(0, len(reach) - 1, n_reach).round().astype(int)
        reach = [reach[i] for i in sorted(set(idx.tolist()))]
    g_open = [c for c in range(n) if valid(c) and L[c] == GRASP and grip[c] < 127.0]
    g_closed = [c for c in range(n) if valid(c) and L[c] == GRASP and grip[c] >= 127.0]
    trans = [c for c in range(n) if valid(c) and L[c] == TRANSPORT]
    p_closed = [c for c in range(n) if valid(c) and L[c] == PLACE and grip[c] >= 127.0]
    p_open = [c for c in range(n) if valid(c) and L[c] == PLACE and grip[c] < 127.0]

    lms = [(c, "reach") for c in reach]
    if g_open:
        lms.append((g_open[len(g_open) // 2], "grasp_open"))
    if g_closed:
        lms.append((g_closed[len(g_closed) // 2], "grasp_closed"))
    elif len(g_open) > 1:
        lms.append((g_open[0], "grasp_open"))
    if trans:
        lms.append((trans[0], "transport"))
        if len(trans) > 1 and trans[-1] != trans[0]:
            lms.append((trans[-1], "transport"))
    if p_closed:
        lms.append((p_closed[0], "place_closed"))
        if len(p_closed) > 1 and p_closed[-1] != p_closed[0]:
            lms.append((p_closed[-1], "place_closed"))
    if p_open:
        lms.append((p_open[0], "place_released"))
    return lms


class BlankFrame(RuntimeError):
    """A rendered camera frame is missing or darker than MIN_PX (the renderer of this process
    failed)."""


def checked_obs(eng, min_px: float = MIN_PX):
    """`vlageom.sim.render.fresh_obs` (cameras moved to the current robot pose, every camera
    rendered), raising BlankFrame when the exo or wrist frame is missing or blank."""
    from vlageom.sim.render import fresh_obs, obs_frames
    obs = fresh_obs(eng)
    exo, wrist = obs_frames(obs)
    if exo is None or wrist is None:
        raise BlankFrame("observation has no exo or wrist frame")
    me, mw = float(exo.mean()), float(wrist.mean())
    if me < min_px or mw < min_px:
        raise BlankFrame(f"blank frame (exo {me:.1f}, wrist {mw:.1f} mean px)")
    return obs


def encode_fresh_query(eng, obs, seed: int):
    """Force one model query on `obs` (flow noise seeded by `seed`) and read its activations
    (`encode_context`) plus the commanded chunk (`chunk`, (16, 8): arm targets 0..6, gripper 7).
    None if no query fired."""
    import torch
    cap = {}

    def gen(**kw):
        kw2, steps = eng._seeded_kw(kw, seed)
        cap["kw"] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                     for k, v in kw2.items() if k != "generator"}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = eng._gen_orig(num_steps=steps, **kw2)
        cap["chunk"] = out[0].detach().float().cpu().numpy()
        return out

    p = eng.policy
    if hasattr(p, "buffer_index") and hasattr(p, "execute_horizon"):
        p.buffer_index = p.execute_horizon
    orig = eng.model.generate_actions
    eng.model.generate_actions = gen
    try:
        p.get_action(obs)
    finally:
        eng.model.generate_actions = orig
    if "kw" not in cap:
        return None
    enc = encode_context(eng, cap["kw"], seed)
    enc["chunk"] = cap.get("chunk")
    return enc


def advance_policy(eng, obs, n_queries: int) -> int:
    """Call the policy on the held observation `obs` until exactly n_queries model queries have
    fired (the simulator does not step). Returns the number that fired."""
    fired = 0
    orig = eng.model.generate_actions

    def gen(*a, **kw):
        nonlocal fired
        fired += 1
        return orig(*a, **kw)

    eng.model.generate_actions = gen
    try:
        guard = 0
        while fired < n_queries and guard < n_queries * 64 + 64:
            if eng.policy.get_action(obs) is None:
                break
            guard += 1
    finally:
        eng.model.generate_actions = orig
    return fired


def perturb_landmark(eng, snap, q0, J, seed: int, detour_ok: bool, min_px: float = MIN_PX):
    """All conditions at one landmark. Each condition restores the landmark's snapshot and arm
    pose q0 and renders the observation at that state (`checked_obs`):
      entry      one fresh query on the observation (the natural read)
      pause<k>   k model queries on the held observation, then a fresh query on it
      off        (detour_ok) the arm moved in DETOUR_STEPS steps to an EE offset of DETOUR_MAG
                 along base x (pseudo-inverse of the EE Jacobian J), one query per step; the fresh
                 query at the farthest pose
      detour     (detour_ok) then the arm put back at q0 in one step and a fresh query
    Returns dict(acts={cond: encode_fresh_query result}, fired={k: queries}, ee0, return_err,
    offroute_disp), or None if the entry query did not fire."""
    from vlageom.sim.kinematics import set_arm_qpos
    acts, fired = {}, {}

    def obs():
        return checked_obs(eng, min_px)

    eng._restore(snap)
    set_arm_qpos(eng, q0)
    ee0 = np.asarray(eng.read_ee(), np.float32)
    acts["entry"] = encode_fresh_query(eng, obs(), seed)
    if acts["entry"] is None:
        return None
    for k in PAUSE_KS:
        eng._restore(snap)
        set_arm_qpos(eng, q0)
        held = obs()
        fired[k] = advance_policy(eng, held, k)
        acts[f"pause{k}"] = encode_fresh_query(eng, held, seed)

    ee_ret = ee_off = None
    if detour_ok:
        Jpinv = np.linalg.pinv(np.asarray(J, np.float32))
        steps = max(DETOUR_STEPS, 1)
        eng._restore(snap)
        set_arm_qpos(eng, q0)
        for k in range(1, steps + 1):
            dq = (Jpinv @ (DETOUR_MAG * DETOUR_AXIS)).astype(np.float32) * (k / steps)
            set_arm_qpos(eng, q0 + dq)
            obs_d = obs()
            advance_policy(eng, obs_d, 1)
            if k == steps:
                ee_off = np.asarray(eng.read_ee(), np.float32)
                acts["off"] = encode_fresh_query(eng, obs_d, seed)
        set_arm_qpos(eng, q0)
        ee_ret = np.asarray(eng.read_ee(), np.float32)
        acts["detour"] = encode_fresh_query(eng, obs(), seed)

    return dict(acts=acts, fired=fired, ee0=ee0, ee_ret=ee_ret, ee_off=ee_off,
                return_err=(float(np.linalg.norm(ee_ret - ee0))
                            if ee_ret is not None and ee_ret.size and ee0.size else float("nan")),
                offroute_disp=(float(np.linalg.norm(ee_off - ee0))
                               if ee_off is not None and ee0.size else float("nan")))


def run_freeze_episode(eng, base: dict, rec: dict, min_px: float = MIN_PX):
    """Landmarks of one rollout (`pick_landmarks` on its own query labels) and all conditions at
    each (`perturb_landmark`; detour only at reach and open-hand grasp landmarks). The flow seed of
    a landmark is scene_seed + 1000 x its query index. Returns (rows, labels)."""
    lab = label_queries(base, rec)
    if lab is None:
        return [], None
    lms = pick_landmarks(base, lab)
    log.info("%d queries, onsets grasp %s transport %s place %s release %s, success %s: "
             "%d landmarks %s", len(base["grip_series"]), lab["grasp_onset"],
             lab["transport_onset"], lab["place_onset"], lab["release_step"], base.get("success"),
             len(lms), [k for _c, k in lms])
    snaps, qsteps = eng._baseline_snaps, base["query_steps"]
    out = []
    for c, kind in lms:
        step = int(qsteps[c])
        if not (0 <= step < len(snaps)):
            continue
        snap = snaps[step]
        seed = eng.scene_seed + 1000 * int(snap["chunk"])
        q0 = np.asarray(base["qpos_arm"][c], np.float32)
        J = np.asarray(base["ee_jac"][c], np.float32)
        detour_ok = kind in ("reach", "grasp_open")
        r = perturb_landmark(eng, snap, q0, J, seed, detour_ok, min_px)
        bad = (r is None or any(r["acts"].get(f"pause{k}") is None for k in PAUSE_KS)
               or (detour_ok and (r["acts"].get("detour") is None or r["acts"].get("off") is None)))
        if bad:
            log.warning("landmark q%d (%s) dropped: a query did not fire", c, kind)
            continue
        r.update(q=int(c), step=step, kind=kind, grip=float(base["grip_series"][c]),
                 label=int(lab["label"][c]), detour_ok=detour_ok)
        out.append(r)
    return out, lab


def freeze_payload(rows: list, lab: dict, base: dict, rec: dict, episode_index: int, house: int,
                   seed: int) -> dict:
    """The arrays of one freeze-test file (np.savez): per condition ae_<cond> (P, 11, 768),
    chunk_<cond> (P, 16, 8), llm_<pool>_<cond> (P, 8, 2560); per landmark fired, kind, q, step,
    grip, label, detour_ok, ee0, return_err, offroute_disp; the rollout's query series and onsets;
    success, layers, ep, house, seed."""
    P, Lae, Lllm = len(rows), len(AE_LAYERS), len(LLM_LAYERS)
    d_llm = rows[0]["acts"]["entry"]["llm_vis"].shape[-1]

    def blk(cond, key, shape):
        A = np.full((P,) + shape, np.nan, np.float32)
        for i, r in enumerate(rows):
            a = r["acts"].get(cond)
            v = a.get(key) if a is not None else None
            if v is not None:
                A[i] = v
        return A

    chunk_shape = rows[0]["acts"]["entry"]["chunk"].shape
    payload = {}
    for cond in CONDS:
        payload[f"ae_{cond}"] = blk(cond, "ae", (Lae, 768))
        payload[f"chunk_{cond}"] = blk(cond, "chunk", chunk_shape)
        for pool in POOLS:
            payload[f"llm_{pool}_{cond}"] = blk(cond, f"llm_{pool}", (Lllm, d_llm))
    fired = np.full((P, len(PAUSE_KS)), -1, np.int32)
    for i, r in enumerate(rows):
        for j, k in enumerate(PAUSE_KS):
            fired[i, j] = int(r["fired"].get(k, -1))

    def onset(v):
        return np.int32(-1 if v is None else v)

    payload.update(
        fired=fired, pause_ks=np.array(PAUSE_KS, np.int32),
        kind=np.array([KIND_CODE[r["kind"]] for r in rows], np.int8),
        q=np.array([r["q"] for r in rows], np.int32),
        step=np.array([r["step"] for r in rows], np.int32),
        grip=np.array([r["grip"] for r in rows], np.float32),
        label=np.array([r["label"] for r in rows], np.int8),
        detour_ok=np.array([r["detour_ok"] for r in rows], bool),
        ee0=np.stack([r["ee0"] for r in rows]).astype(np.float32),
        return_err=np.array([r["return_err"] for r in rows], np.float32),
        offroute_disp=np.array([r["offroute_disp"] for r in rows], np.float32),
        q_grip=np.asarray([float(g) for g in base["grip_series"]], np.float32),
        q_objz=np.asarray(rec["objz_q"], np.float32),
        q_supported=np.asarray(rec["supported_q"], bool),
        q_pos_err=np.asarray(rec["pos_err_q"], np.float32),
        q_label=lab["label"].astype(np.int8),
        grasp_onset=onset(lab["grasp_onset"]), transport_onset=onset(lab["transport_onset"]),
        place_onset=onset(lab["place_onset"]), release_step=onset(lab["release_step"]),
        attempts=np.int32(lab["attempts"]), success=bool(base.get("success", False)),
        ae_layers=np.array(AE_LAYERS), llm_layers=np.array(LLM_LAYERS),
        ep=int(episode_index), house=int(house), seed=int(seed))
    return payload


# ----------------------------------------------------------------------------------- capture
def _obs_images(obs) -> dict:
    """{camera: H x W x 3 uint8} of the policy observation `obs` for CAPTURE_CAMS."""
    o = obs[0] if isinstance(obs, (list, tuple)) else obs
    out = {}
    for cam in CAPTURE_CAMS:
        v = o.get(cam) if hasattr(o, "get") else None
        if v is None:
            continue
        a = np.asarray(v)
        if a.ndim == 4:
            a = a[0]
        if a.dtype != np.uint8:
            a = (a * 255 if a.max() <= 1.0 + 1e-6 else a).clip(0, 255).astype(np.uint8)
        out[cam] = a
    return out


def install_frame_recorder(eng) -> dict:
    """Record the camera images of every observation the rollout returns: index 0 from
    `task.reset()`, index i + 1 from the i-th `task.step()`. Install after `install_recorder`."""
    task = eng.task
    frames = {}
    reset0, step0 = task.reset, task.step
    ctr = {"i": 0}

    def reset(*a, **kw):
        out = reset0(*a, **kw)
        ctr["i"] = 0
        frames[0] = _obs_images(out[0] if isinstance(out, tuple) else out)
        return out

    def step(action, *a, **kw):
        out = step0(action, *a, **kw)
        ctr["i"] += 1
        frames[ctr["i"]] = _obs_images(out[0] if isinstance(out, tuple) else out)
        return out

    task.reset = reset
    task.step = step
    return frames


def capture_episode(eng, episode_index: int, house: int, seed: int, new_sampler: int,
                    max_steps: int = MAX_STEPS):
    """Roll out one episode keeping, per model query, the exo and wrist images the policy read and
    the activations of that query; phase labels from the rollout's own per-step records.
    Returns (payload, query counts per phase, success), or (None, {}, False)."""
    sample_listed(eng, house, seed, new_sampler)
    task_desc = eng.task.get_task_description()
    tcfg = eng.task.config.task_config
    log.info("ep%d h%d seed%d: %s", episode_index, house, seed, task_desc)
    rec = install_recorder(eng)
    frames = install_frame_recorder(eng)
    base = eng.run_baseline(max_steps)
    rec = finalize_record(rec, base)
    Q = len(base["grip_series"])
    if Q == 0:
        return None, {}, False
    acts = encode_all_queries(eng)
    info = label_steps(rec["step_grip"], rec["step_objz"], rec["step_supported"],
                       success=bool(base["success"]))
    qsteps = np.asarray(base["query_steps"], int)
    lab_q = labels_at_queries(qsteps, info["label"])
    have = sorted(frames)

    def frame_at(s, cam):
        if s in frames and cam in frames[s]:
            return frames[s][cam]
        prev = [t for t in have if t <= s and cam in frames[t]]
        return frames[prev[-1]][cam] if prev else None

    stacks = {}
    for cam in CAPTURE_CAMS:
        fr = [frame_at(int(s), cam) for s in qsteps]
        if any(f is None for f in fr):
            log.error("camera %s: frames missing", cam)
            return None, {}, False
        stacks[cam] = np.stack(fr).astype(np.uint8)

    def onset(v):
        return np.int32(-1 if v is None else v)

    payload = {
        "frames_exo": stacks["exo_camera_1"], "frames_wrist": stacks["wrist_camera"],
        "ae": acts["ae"], **{f"llm_{p}": acts[f"llm_{p}"] for p in POOLS},
        "ae_layers": np.array(AE_LAYERS, np.int32), "llm_layers": np.array(LLM_LAYERS, np.int32),
        "query_steps": qsteps.astype(np.int32), "label_q": lab_q.astype(np.int8),
        "grip_q": np.asarray([float(g) for g in base["grip_series"]], np.float32),
        "objz_q": np.asarray(rec["objz_q"], np.float32),
        "supported_q": np.asarray(rec["supported_q"], bool),
        "step_label": info["label"].astype(np.int8),
        **{k: rec[k] for k in rec if k.startswith("step_")},
        "n_steps": np.int32(base["n_steps"]), "n_queries": np.int32(Q),
        "success": bool(base["success"]),
        "grasp_onset": onset(info["grasp_onset"]),
        "transport_onset": onset(info["transport_onset"]),
        "place_onset": onset(info["place_onset"]), "release_step": onset(info["release_step"]),
        "ep": np.int32(episode_index), "house": np.int32(house), "seed": np.int64(seed),
        "task_description": str(task_desc),
        "pickup_obj_name": str(getattr(tcfg, "pickup_obj_name", "")),
        "place_receptacle_name": str(getattr(tcfg, "place_receptacle_name", "")),
    }
    spans = {NAMES[k]: int((lab_q == k).sum()) for k in range(N_PHASE)}
    return payload, spans, bool(base["success"])


# ----------------------------------------------------------------------------------- panel arrays
def pca_embed(X, k: int = 3):
    """Top-k PCA by SVD of the centred rows. Returns (scores (N, k), evr (k,), comps (k, D),
    mean (D,)); dtypes follow X."""
    mu = X.mean(0)
    Xc = X - mu
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    comps = Vt[:k]
    scores = Xc @ comps.T
    evr = (S ** 2) / max((S ** 2).sum(), 1e-12)
    return scores, evr[:k], comps, mu


def lda_embedding(X, y):
    """Full-data StandardScaler + four-class LDA; returns the LDA transform of X (N, 3)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LDA().fit(sc.transform(X), y)
    return clf.transform(sc.transform(X))


def fit_probes(cache: dict, layer: int, held_house: int, ae_q) -> tuple[np.ndarray, dict]:
    """One-vs-rest phase probes for the queries `ae_q` (Q, 768) of an episode from house
    `held_house`: on the successful natural queries of every other house at `layer`, a shared
    StandardScaler and one binary LDA per phase; the probe readout is the decision function
    (the projection on the probe's discriminant direction), normalised once over all phases and
    queries by its 2nd / 98th percentiles and clipped to [0, 1]. Returns (projn (Q, 4), meta)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    layers = [int(v) for v in cache["layers"]]
    Xl = np.ascontiguousarray(cache["X"][:, layers.index(layer), :])
    lab = cache["label"].astype(int)
    house = cache["house"].astype(int)
    succ = cache["success"].astype(bool)
    m = succ & (house != held_house)
    n_drop = int((succ & (house == held_house)).sum())
    if n_drop == 0:
        raise ValueError(f"house {held_house} has no successful natural queries to hold out")
    Xt, yt = Xl[m], lab[m]
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    Xq = sc.transform(np.asarray(ae_q, np.float64))
    proj = np.zeros((len(Xq), N_PHASE))
    for p in range(N_PHASE):
        y = (yt == p).astype(int)
        clf = LDA().fit(Xs, y)
        proj[:, p] = clf.decision_function(Xq)
    fin = np.isfinite(proj)
    lo, hi = float(np.percentile(proj[fin], 2)), float(np.percentile(proj[fin], 98))
    projn = np.clip((proj - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    projn[~fin] = np.nan
    meta = {"layer": int(layer), "held_out_house": int(held_house),
            "n_rows_held_out": n_drop, "n_train_rows": int(m.sum()),
            "n_train_houses": int(len(np.unique(house[m]))), "proj_norm_lo": lo,
            "proj_norm_hi": hi}
    return projn, meta


def latent_colours(projn, gamma: float = 2.2):
    """Per-query colour of the four probe readouts: weights w = r^gamma / sum(r^gamma), colour =
    sum_p w_p * phase colour p, rescaled so its largest channel is min(1.25 x max, 1).
    Returns (rgb (Q, 3), weights (Q, 4))."""
    from matplotlib.colors import to_rgb
    phase_rgb = np.array([to_rgb(PHASE_COLORS[k]) for k in range(N_PHASE)])
    V = np.nan_to_num(np.asarray(projn, float), nan=0.0).clip(0, None) ** gamma
    W = V / np.maximum(V.sum(1, keepdims=True), 1e-9)
    rgb = W @ phase_rgb
    mx = rgb.max(1, keepdims=True)
    rgb = np.clip(rgb / np.maximum(mx, 1e-9) * np.clip(mx * 1.25, 0, 1), 0, 1)
    return rgb, W


def runs_of(lab, p):
    """Contiguous index runs where lab == p."""
    idx = np.where(np.asarray(lab) == p)[0]
    if len(idx) == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)


def pick_columns(lab_q):
    """Two query indices per phase, at about 1/3 and 2/3 through its longest run (None, None for
    an absent phase). Returns [(a, b)] in phase order."""
    cols = []
    for p in range(N_PHASE):
        rs = runs_of(lab_q, p)
        if not rs:
            cols.append((None, None))
            continue
        run = max(rs, key=len)
        n = len(run)
        cols.append((int(run[min(int(n / 3.0), n - 1)]), int(run[min(int(n * 2.0 / 3.0), n - 1)])))
    return cols


def own_phase_data(comp: dict) -> dict:
    """Panel c numbers from the decoded-phase composition: grasp pools the open- and closed-hand
    grasp landmarks and place the closed- and open-hand place landmarks (n-weighted). Returns
    own[cond][phase] (% decoded as the landmark's own phase), and per phase the most and second
    most frequent decoded phase under the frozen condition with their %, and n."""
    def pool(kinds):
        out = {}
        for c in (COND_NATURAL, COND_FROZEN):
            rows = [comp[k][c] for k in kinds if k in comp and c in comp[k]]
            if not rows:
                continue
            ntot = sum(r["n"] for r in rows)
            out[c] = {p: sum(r.get(p, 0.0) * r["n"] for r in rows) / ntot for p in NAMES}
            out[c]["n"] = ntot
        return out

    agg = {"reach": comp.get("reach", {}), "grasp": pool(["grasp_open", "grasp_closed"]),
           "transport": comp.get("transport", {}),
           "place": pool(["place_closed", "place_released"])}
    own = {c: {k: agg[k][c][k] * 100 for k in NAMES} for c in (COND_NATURAL, COND_FROZEN)}
    dest = {}
    for k in NAMES:
        row = agg[k][COND_FROZEN]
        order = sorted(NAMES, key=lambda p: -row[p])
        dest[k] = dict(top=order[0], top_pct=row[order[0]] * 100,
                       second=order[1], second_pct=row[order[1]] * 100)
    n = {k: int(agg[k][COND_FROZEN]["n"]) for k in NAMES}
    return dict(own=own, dest=dest, n=n)


def load_gripper_chunks(files) -> dict:
    """{kind code: (natural gripper commands (n, 16), frozen gripper commands (n, 16))} over the
    landmarks of the freeze-test files selected by `used_landmarks`, landmarks with finite chunks
    only."""
    out = {code: [] for _n, codes in PANEL_KINDS for code in codes}
    for f in files:
        z = np.load(f, allow_pickle=True)
        used = used_landmarks(z["kind"], z["q"])
        for i in range(len(z["kind"])):
            if not used[i]:
                continue
            e, p = z[f"chunk_{COND_NATURAL}"][i], z[f"chunk_{COND_FROZEN}"][i]
            if int(z["kind"][i]) in out and np.isfinite(e).all() and np.isfinite(p).all():
                out[int(z["kind"][i])].append((e[:, GRIP], p[:, GRIP]))
    return {k: (np.array([e for e, _ in v]), np.array([p for _, p in v]))
            for k, v in out.items() if v}


def gripper_delta_data(chunks: dict) -> dict:
    """Panel d numbers per phase: d = per-landmark (fraction of the EXECUTED steps with a closed
    gripper command, > 0) frozen minus natural; bar = mean(d), ci = 1.96 sd(d, ddof 1) / sqrt(n);
    also the closed fraction per executed step under each condition."""
    out = {}
    for label, codes in PANEL_KINDS:
        E = np.concatenate([chunks[c][0] for c in codes])[:, :EXECUTED]
        P = np.concatenate([chunks[c][1] for c in codes])[:, :EXECUTED]
        d = (P > 0).mean(1) - (E > 0).mean(1)
        out[label] = dict(d=d, mean=d.mean(), ci=1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                          closed_natural=(E > 0).mean(0), closed_frozen=(P > 0).mean(0))
    return out
