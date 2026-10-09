"""Contact (Figure 4): gripper-object contact in the VLM of MolmoBot-DROID.

Contact = a gripper geom in positive-force contact with the pickup object
(`BranchEngine._read_grasp`). Figure 4 has two harvests and four CPU analyses:

  natural harvest      the 299 natural pick episodes of lists/contact/natural_episodes.csv are
                       rolled out (at most 600 steps). For every model query the VLM backbone is run
                       again from the query's inputs with forward hooks on transformer.blocks.{L} for
                       the 10 layers in `LAYERS`; the outputs of the current exo and wrist image
                       patches (196 + 196 tokens), of the text tail and the mean over the whole
                       sequence are stored in float16 (`NATURAL_KEYS`). The contact label of every
                       step comes from this rollout: each per-step snapshot is restored and contact is
                       read (`replay_contact_series`).
  natural cache        one row per query: six token pools per layer (`VARIANTS`), pooled from the
                       stored float16 tokens with a float32 mean and stored as float16, plus labels
                       (`pool_natural_shard`).
  off-schedule harvest for each episode of lists/contact/offschedule_episodes.csv: up to 4 queries
                       with the hand open (early-touch anchors) and up to 4 queries where the hand
                       holds the object (slip anchors). At each anchor the arm is held at the query's
                       joint angles and the object is moved along the hand-object line in 1 cm steps
                       until contact changes, then 3 steps further (`march_boundary`). Four events per
                       anchor (`EVENTS`): early touch with the hand as it is (open) or forced closed,
                       slip with the hand as it is (closed) or forced open. Every placement is
                       rendered with the cameras moved to the current hand pose and encoded like the
                       natural queries (`PlacementEncoder`).
  off-schedule pool    the text-token mean and the final token at layer 24 of every placement
                       (`pool_offschedule_shard`).
  natural probe        out-of-house LDA probe of contact on the natural rows (`oof_lda`, folds from
                       lists/folds/contact_natural.csv): the single-frame accuracy drawn as the
                       baseline of panel b.
  held-out scores      for each house of the off-schedule store, an LDA probe fit on the natural
                       rows of all other houses scores that house's placements; per anchor and event,
                       a vote is correct when the mean score of the contact placements exceeds the
                       mean score of the no-contact placements (`within_pose`).

Numpy only at import; simulator, torch and scikit-learn are imported inside the functions that
need them.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.util import npz_member

log = logging.getLogger(__name__)

TILE = "contact"
CONFIG = "FrankaPickDroidDataGenConfig"   # MolmoSpaces task config of the rollouts
MAX_STEPS = 600                           # rollout length cap (env steps)
GRIP = 7                                  # arm joints (the gripper is action channel 7)
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]   # VLM blocks read by both harvests
N_QUERIES = 128                           # natural harvest: queries kept per episode (all, in practice)
N_PATCH = 196                             # image patches per camera frame
VARIANTS = ["seq_all", "img_all", "img_exo", "img_wrist", "txt_all", "last_tok"]
PROBE_LAYER = 24                          # layer of the probes and of the off-schedule pool
POOLS = ("txt_all", "last_tok")           # token pools of the off-schedule pool and the probes
ORIG_QUERY_INTERVAL = 8                   # the listed rollouts queried the model at steps 0, 8, 16, ...
MIN_PX = 20.0                             # minimum mean exo-camera pixel value of a first query

# off-schedule placements
N_POSE = 4                                # anchors per kind (open hand / holding)
STEP_M = 0.01                             # object step along the hand-object line (m)
K_MAX = 24                                # step cap, hand as it is
K_MAX_FORCED = 40                         # step cap, forced hand state
STOP_AFTER = 3                            # steps taken after contact changes
Z_FLOOR = 0.02                            # placements with the object below this height are skipped
MAX_GAIN = 2.5                            # skip when the object moved more than 2.5x the command
MIN_COS = 0.5                             # skip when the object moved off the commanded direction
# event -> (name, forced finger state or None)
EVENTS = {0: ("early touch, hand open", None), 1: ("slip, hand closed", None),
          2: ("early touch, hand forced closed", "closed"), 3: ("slip, hand forced open", "open")}
# scored cells: name -> (event, branch); branch +1 = object moved out of the opened hand, -1 = moved
# back in until it touches again, 0 = either
CELLS = {"early_touch_open": (0, 0), "early_touch_closed": (2, 0), "slip_closed": (1, 0),
         "slip_open_out": (3, 1), "slip_open_in": (3, -1)}
BAR_CELLS = ("early_touch_open", "early_touch_closed", "slip_closed", "slip_open_out")

HELDOUT_NOTE = ("each pose scored by the probe fit on natural rows excluding that pose's house "
                "(per-house refit; houses absent from the natural set use the all-rows fit)")

# keys of a natural harvest file, in the order they are written
NATURAL_KEYS = ("act_img", "act_txt", "act_seqmean", "txt_ids", "txt_str", "contact_series", "q_idx",
                "query_steps_kept", "layout_imgs", "replay_grasp", "replay_grip", "layers", "house",
                "ep", "seed", "n_queries", "n_steps", "base_success", "descr")
# keys of an off-schedule harvest file, in the order they are written
OFFSCHEDULE_KEYS = ("act", "contact", "disp", "dist", "pose_id", "step_k", "event", "phase", "qc",
                    "grasp0", "pose_chunk", "qpos0", "pose_event", "pose_qc", "pose_sep0",
                    "pose_grasp_forced", "pose_dir", "layers", "house", "ep", "n_txt", "txt_ids",
                    "txt_str")
# per-row fields of the off-schedule pool
POOL_FIELDS = ("ep", "house", "pose_id", "gpose", "event", "pdir", "qc", "step_k", "contact", "disp",
               "dist", "phase", "pose_chunk", "grasp0", "pose_grasp_forced", "pose_sep0", "n_txt",
               "shard_idx")
# keys of panels/contact.npz
PANEL_KEYS = ("P", "contact", "evr", "origin_p", "show", "bar_cells", "frac_pos", "frac_pos_se",
              "n_pose", "baseline_acc")


# ---- locations ---------------------------------------------------------------------------------- #
def natural_dir(root=None) -> Path:
    """Natural harvest files: <root>/natural (root default $VLAGEOM_DATA/contact)."""
    return Path(root or paths.tile_dir(TILE)) / "natural"


def offschedule_dir(root=None) -> Path:
    """Off-schedule harvest files: <root>/offschedule (root default $VLAGEOM_DATA/contact)."""
    return Path(root or paths.tile_dir(TILE)) / "offschedule"


def shard_name(episode_index: int, house: int) -> str:
    return f"ep{int(episode_index)}_h{int(house)}.npz"


# ---- shipped lists ------------------------------------------------------------------------------ #
def read_natural_episodes(path=None) -> list[dict]:
    """Rows of lists/contact/natural_episodes.csv (file order = episode_index order):
    episode_index, house, seed, task_description, orig_n_steps, orig_contact.

    orig_n_steps and orig_contact describe the listed natural rollout of the episode: its length
    in steps and the gripper-object contact at each of its model queries (steps 0, 8, 16, ...;
    one character per query, "1" = contact). The cache uses them only for the `agree` and
    `orig_n_steps` fields."""
    path = Path(path) if path is not None else paths.lists_dir() / "contact" / "natural_episodes.csv"
    with open(path, newline="") as f:
        return [dict(episode_index=int(r["episode_index"]), house=int(r["house"]),
                     seed=int(r["seed"]), task_description=r["task_description"],
                     orig_n_steps=int(r["orig_n_steps"]), orig_contact=r["orig_contact"])
                for r in csv.DictReader(f)]


def read_offschedule_episodes(path=None) -> list[dict]:
    """Rows of lists/contact/offschedule_episodes.csv: task, episode_index, house, seed, outcome.

    outcome is what the harvest of the episode gives: "shard" (a file is written), "no_scene"
    (MolmoSpaces finds no valid robot placement for the house and raises HouseInvalidForTask) or
    "no_crossing" (no anchor crosses the contact boundary, so nothing is written)."""
    path = (Path(path) if path is not None
            else paths.lists_dir() / "contact" / "offschedule_episodes.csv")
    with open(path, newline="") as f:
        return [dict(task=int(r["task"]), episode_index=int(r["episode_index"]),
                     house=int(r["house"]), seed=int(r["seed"]), outcome=r["outcome"])
                for r in csv.DictReader(f)]


# ---- policy inputs and VLM readout ------------------------------------------------------------- #
def decode_token(tok, tid: int) -> str:
    try:
        return tok.decode([int(tid)])
    except Exception:
        return ""


def exo_mean_px(obs) -> float:
    """Mean pixel value of the exo-camera image of an observation (nan if none)."""
    from vlageom.sim.render import exo_frame
    img = exo_frame(obs)
    return float("nan") if img is None else float(np.asarray(img, np.float32).mean())


# ---- natural harvest ---------------------------------------------------------------------------- #
def even_subsample(n, k):
    """At most k evenly spaced indices of range(n) (all when n <= k)."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def replay_contact_series(eng, n_steps):
    """Per-step contact of the rollout just run: every per-step snapshot is restored (mj_forward)
    and `_read_grasp` is read. (n_steps,) bool."""
    ser = np.zeros(n_steps, bool)
    for i, snap in enumerate(eng._baseline_snaps[:n_steps]):
        eng._restore(snap)
        ser[i] = eng._read_grasp()
    return ser


def natural_payload(eng, tok, base, episode_index, house, seed, descr, layers=LAYERS,
                    n_q=N_QUERIES):
    """The natural harvest file of one rolled-out episode (keys `NATURAL_KEYS`), or None when no
    query fired. `base` is the `run_baseline` result; `descr` is the listed task description."""
    from vlageom.sim.encode import capture_blocks, token_map
    nq = int(base["n_queries"])
    if nq < 1 or not eng._captured:
        log.warning("ep%d h%d: no queries fired", episode_index, house)
        return None
    keep = even_subsample(min(nq, len(eng._captured)), n_q)
    grasp_q = list(base.get("grasp_q", []))
    grip_series = list(base.get("grip_series", []))
    cseries = replay_contact_series(eng, int(base["n_steps"]))

    txt_ids, txt_str, T = None, None, None
    img_rows, txt_rows, seq_rows, q_idx, layouts, rgrasp, rgrip = [], [], [], [], [], [], []
    for q in keep:
        acts, ids = capture_blocks(eng, eng._captured[int(q)], layers)
        if acts is None:
            log.warning("ep%d q%d: a block hook did not fire; query skipped", episode_index, q)
            continue
        tm = token_map(ids)
        tail = ids[tm["tail_start"]:]
        if txt_ids is None:
            txt_ids, T = tail.astype(np.int32), len(tail)
            txt_str = np.array([decode_token(tok, t) for t in txt_ids], dtype="U32")
        elif len(tail) != T or not np.array_equal(tail.astype(np.int32), txt_ids):
            log.warning("ep%d q%d: text tail differs from the first query; query skipped",
                        episode_index, q)
            continue
        (se, le), (sw, lw) = tm["cur_exo"], tm["cur_wrist"]
        if le != N_PATCH or lw != N_PATCH:
            log.warning("ep%d q%d: patch runs of length (%d, %d); query skipped", episode_index, q,
                        le, lw)
            continue
        img = np.stack([np.concatenate([acts[L][se:se + le], acts[L][sw:sw + lw]]) for L in layers])
        txt = np.stack([acts[L][tm["tail_start"]:] for L in layers])
        seqm = np.stack([acts[L].mean(0) for L in layers])
        img_rows.append(img.astype(np.float16))
        txt_rows.append(txt.astype(np.float16))
        seq_rows.append(seqm.astype(np.float16))
        q_idx.append(int(q))
        layouts.append(tm["n_img"])
        rgrasp.append(bool(grasp_q[q]) if q < len(grasp_q) else False)
        rgrip.append(float(grip_series[q]) if q < len(grip_series) else np.nan)
    if not img_rows:
        return None
    qsteps = base["query_steps"]
    return dict(
        act_img=np.stack(img_rows), act_txt=np.stack(txt_rows), act_seqmean=np.stack(seq_rows),
        txt_ids=txt_ids, txt_str=txt_str,
        contact_series=cseries,
        q_idx=np.array(q_idx, np.int32),
        query_steps_kept=np.array([qsteps[q] if q < len(qsteps) else -1 for q in q_idx], np.int32),
        layout_imgs=np.array(layouts, np.int8),
        replay_grasp=np.array(rgrasp, bool), replay_grip=np.array(rgrip, np.float32),
        layers=np.array(layers, np.int32),
        house=int(house), ep=int(episode_index), seed=int(seed),
        n_queries=int(base["n_queries"]), n_steps=int(base["n_steps"]),
        base_success=bool(base.get("success", False)),
        descr=str(descr))


# ---- natural cache ------------------------------------------------------------------------------ #
def rollout_instruction(txt_str) -> str:
    """The task instruction of a natural harvest file, from its decoded text tail (the text after
    "demo: " up to the end of that line)."""
    s = "".join(str(t) for t in np.asarray(txt_str).tolist())
    i = s.find("demo: ")
    if i < 0:
        return s.strip()
    return s[i + len("demo: "):].split("\n", 1)[0].strip()


def pool_natural_shard(path, orig_n_steps, orig_contact):
    """One natural harvest file -> dict of per-query token pools (`VARIANTS`, each (P, L, 2560)
    float16) and labels, or None when the episode has no listed rollout record.

    Pools are float32 means over the stored float16 tokens, stored as float16: img_all (the 392
    current-frame patches), img_exo / img_wrist (196 each), txt_all (the text-tail tokens other than
    the chat and multimodal control tokens), last_tok (the final token) and seq_all (the stored mean
    over all tokens). Labels come from the harvest's own rollout: contact = contact at the query's
    step, onset = first contact step, delta = step - onset, phase = step / (n_steps - 1),
    grip = executed gripper command / 255. `agree` is 1 / 0 when the listed rollout queried the
    model at the same step and its contact label agrees / disagrees, else -1.
    `instruction_differs` (not stored in the cache) is True when the instruction of the harvest's
    rollout differs from the listed task description (`descr`)."""
    from vlageom.sim.encode import tail_special_mask
    z = np.load(path, allow_pickle=False)
    ep, house = int(z["ep"]), int(z["house"])
    n_steps = int(orig_n_steps)
    orig_contact = str(orig_contact)
    if n_steps < 1 or not orig_contact:
        return None
    orig_qsteps = [ORIG_QUERY_INTERVAL * q for q in range(len(orig_contact))]

    ai = z["act_img"]
    at = z["act_txt"]
    aseq = z["act_seqmean"]
    txt_ids = z["txt_ids"]
    nonspec = ~tail_special_mask(txt_ids)
    if not nonspec.any():
        nonspec = np.ones(len(txt_ids), bool)
    q_idx = z["q_idx"]
    steps_kept = z["query_steps_kept"]
    cseries = z["contact_series"]
    rgrip = z["replay_grip"] if "replay_grip" in z.files else None
    n_steps_ep = int(z["n_steps"])
    onset_ep = int(np.argmax(cseries)) if cseries.any() else -1
    onset_phase_ep = onset_ep / max(1, n_steps_ep - 1) if onset_ep >= 0 else np.nan
    succ_ep = bool(z["base_success"])

    pools = {
        "seq_all": aseq.astype(np.float16),
        "img_all": ai.mean(axis=2, dtype=np.float32).astype(np.float16),
        "img_exo": ai[:, :, :N_PATCH].mean(axis=2, dtype=np.float32).astype(np.float16),
        "img_wrist": ai[:, :, N_PATCH:].mean(axis=2, dtype=np.float32).astype(np.float16),
        "txt_all": at[:, :, nonspec].mean(axis=2, dtype=np.float32).astype(np.float16),
        "last_tok": at[:, :, -1, :].astype(np.float16),
    }

    rows = dict(X={v: [] for v in VARIANTS}, contact=[], delta=[], grip=[], phase=[],
                q=[], step=[], agree=[], step_mismatch=0)
    for i, q in enumerate(q_idx):
        q = int(q)
        step = int(steps_kept[i])
        if not (0 <= step < len(cseries)):
            rows["step_mismatch"] += 1
            continue
        on = bool(cseries[step])
        if step < n_steps and q < len(orig_qsteps) and int(orig_qsteps[q]) == step:
            rows["agree"].append(int((orig_contact[q] == "1") == on))
        else:
            rows["agree"].append(-1)
        rows["delta"].append(float(step - onset_ep) if onset_ep >= 0 else np.nan)
        rows["grip"].append(float(rgrip[i]) / 255.0 if rgrip is not None else np.nan)
        for v in VARIANTS:
            rows["X"][v].append(pools[v][i])
        rows["contact"].append(on)
        rows["phase"].append(step / max(1, n_steps_ep - 1))
        rows["q"].append(q)
        rows["step"].append(step)
    n = len(rows["contact"])
    if n == 0:
        return None
    return dict(
        ep=ep, house=house, success=succ_ep, onset=onset_ep, onset_phase=onset_phase_ep,
        orig_n_steps=n_steps, n_steps_ep=n_steps_ep, replay=True,
        instruction_differs=rollout_instruction(z["txt_str"]) != str(z["descr"]).strip(),
        layers=z["layers"], n=n, step_mismatch=rows["step_mismatch"],
        X={v: np.stack(rows["X"][v]) for v in VARIANTS},
        contact=np.array(rows["contact"], bool),
        delta=np.array(rows["delta"], np.float32), grip=np.array(rows["grip"], np.float32),
        phase=np.array(rows["phase"], np.float32), q=np.array(rows["q"], np.int32),
        step=np.array(rows["step"], np.int32),
        agree=np.array(rows["agree"], np.int8))


def merge_natural_parts(parts) -> dict:
    """Concatenate `pool_natural_shard` results (any order) into the cache arrays, episodes in
    increasing order. Returns {"layers", "variants", "X_<variant>", labels...}; episodes whose layer
    grid differs from the lowest episode's are dropped (and listed in "dropped").
    "n_instruction_differs" counts the episodes whose rollout instruction differs from the listed
    task description."""
    parts = sorted(parts, key=lambda r: r["ep"])
    layers = parts[0]["layers"]
    dropped = [p["ep"] for p in parts if not np.array_equal(p["layers"], layers)]
    parts = [p for p in parts if np.array_equal(p["layers"], layers)]

    def cat(key):
        return np.concatenate([p[key] for p in parts])

    def bcast(key, dt):
        return np.concatenate([np.full(p["n"], p[key], dt) for p in parts])

    merged = {f"X_{v}": np.concatenate([p["X"][v] for p in parts]) for v in VARIANTS}
    merged.update(
        contact=cat("contact"), delta=cat("delta"), grip=cat("grip"),
        phase=cat("phase"), q=cat("q"), step=cat("step"), agree=cat("agree"),
        ep=bcast("ep", np.int32), house=bcast("house", np.int32),
        success=bcast("success", bool), onset=bcast("onset", np.int32),
        onset_phase=bcast("onset_phase", np.float32),
        orig_n_steps=bcast("orig_n_steps", np.int32), n_steps_ep=bcast("n_steps_ep", np.int32),
        replay=bcast("replay", bool))
    return {"layers": np.asarray(layers), "variants": np.array(VARIANTS), "merged": merged,
            "dropped": dropped, "step_mismatch": sum(p["step_mismatch"] for p in parts),
            "n_eps": len(parts),
            "n_instruction_differs": sum(bool(p.get("instruction_differs")) for p in parts)}


def load_layer(cache_path, key, layer, dtype=np.float32):
    """One layer of a (N, L, d) array of the natural cache, as `dtype` (the cache's `layers` gives
    the layer order)."""
    with np.load(cache_path, mmap_mode="r") as z:
        layers = [int(x) for x in z["layers"]]
        li = layers.index(int(layer))
        return np.asarray(z[key][:, li, :], dtype)


# ---- off-schedule harvest: scene edits ---------------------------------------------------------- #
def obj_world(eng):
    """Pickup object world position (3,) float32, or None."""
    o = np.asarray(eng.read_object(), np.float32)
    return o if o.size == 3 else None


def pose_hand(eng, snap, q0, grip=None, gripper=None):
    """Restore `snap`, set the arm joints to q0 and, when `grip` is "open" or "closed", the finger
    joints to that limit (`GripperControl.set`)."""
    from vlageom.sim.kinematics import set_arm_qpos
    eng._restore(snap)
    set_arm_qpos(eng, q0)
    if grip is not None:
        gripper.set(grip)


def pick_holding_poses(base, n_pose):
    """Up to n_pose query indices of a `run_baseline` result where the gripper holds the object
    (contact at the query) and the arm joints were recorded, spread evenly."""
    grasp = list(base.get("grasp_q", []))
    qarm = base["qpos_arm"]
    qsteps = base["query_steps"]
    cand = [c for c in range(len(grasp))
            if grasp[c] and c < len(qarm) and qarm[c] is not None and c < len(qsteps)]
    if len(cand) <= n_pose:
        return cand
    idx = np.linspace(0, len(cand) - 1, n_pose).round().astype(int)
    return [cand[i] for i in sorted(set(idx.tolist()))]


class PlacementEncoder:
    """Encode the current scene for the off-schedule store. For each placement: move the
    robot-mounted cameras to the current hand pose and render every camera
    (`vlageom.sim.render.fresh_obs`), run one policy query with the given flow-noise seed to get its
    inputs (`capture_kw`), and run the VLM backbone on them with hooks (`capture_blocks`). The
    policy keeps the frame history of the restored snapshot (it is not reset), so the query reads
    the current exo and wrist frames plus the rollout's frames from 8 steps earlier.

    Returns (L, 392 + T + 1, 2560) float32 per placement: rows 0-391 = current exo then wrist
    patches, rows 392 .. 392+T-1 = the text tail, last row = the mean over the whole sequence; None
    when no query fired, a hook did not fire, the patch runs are not 196 long or the text tail
    length differs from the episode's first placement. `reset()` starts a new episode."""

    def __init__(self, layers=LAYERS):
        self.layers = list(layers)
        self.tail_ids = None

    def reset(self):
        self.tail_ids = None

    def __call__(self, eng, seed):
        from vlageom.sim.encode import capture_blocks, capture_kw, token_map
        from vlageom.sim.render import fresh_obs
        obs = fresh_obs(eng)
        kw = capture_kw(eng, obs, seed)
        if kw is None:
            return None
        acts, ids = capture_blocks(eng, kw, self.layers)
        if acts is None:
            return None
        tm = token_map(ids)
        (se, le), (sw, lw) = tm["cur_exo"], tm["cur_wrist"]
        if le != N_PATCH or lw != N_PATCH:
            return None
        tail_ids = ids[tm["tail_start"]:].astype(np.int32)
        if self.tail_ids is None:
            self.tail_ids = tail_ids
        elif len(tail_ids) != len(self.tail_ids):
            return None
        out = []
        for L in self.layers:
            a = acts[L]
            out.append(np.concatenate([a[se:se + le], a[sw:sw + lw],
                                       a[tm["tail_start"]:], a.mean(0, keepdims=True)]))
        return np.stack(out).astype(np.float32)


def march_boundary(eng, snap, q0, encode, seed, toward_hand, k_max, want_contact, *, grip=None,
                   gripper=None, step_m=STEP_M, stop_after=STOP_AFTER, z_floor=Z_FLOOR,
                   max_gain=MAX_GAIN, min_cos=MIN_COS, verbose=False):
    """Hold the hand at q0 (and the forced finger state `grip`, if any) and move the object along
    the world hand-object line in steps of step_m, toward the hand when toward_hand, else away,
    until contact equals want_contact, then stop_after steps more (at most k_max steps). Every
    placement starts from the restored snapshot. A placement is skipped when the object ends below
    z_floor, moved more than max_gain times the command, did not move, or moved at cosine < min_cos
    to the command. Row 0 is the unmoved scene.

    Returns (rows, grasp0, obj_base0, dist0) with rows = [(k, disp, contact, dist, block)]
    (disp > 0 = away from the hand, dist = hand-object distance in the base frame, block = the
    `encode` output), or None when the unmoved scene cannot be encoded or contact never changes."""
    from vlageom.sim.scene import base_transform, move_pickup_object
    pose_hand(eng, snap, q0, grip, gripper)
    R, t = base_transform(eng, np.float32)
    ee_b = np.asarray(eng.read_ee(), np.float32)
    ow0 = obj_world(eng)
    if ee_b.size != 3 or ow0 is None:
        return None
    ee_w = R @ ee_b + t
    ray = ow0 - ee_w
    n = float(np.linalg.norm(ray))
    if n < 1e-4:
        return None
    u = (ray / n).astype(np.float32)
    move_u = (-u if toward_hand else u)
    ob0 = (R.T @ (ow0 - t)).astype(np.float32)
    dist0 = float(np.linalg.norm(ee_b - ob0))
    base_vec = encode(eng, seed)
    if base_vec is None:
        return None
    grasp0 = bool(eng._read_grasp())
    rows = [(0, 0.0, grasp0, dist0, base_vec)]
    crossed = None
    for k in range(1, k_max + 1):
        pose_hand(eng, snap, q0, grip, gripper)
        move_vec = float(k * step_m) * move_u
        move_pickup_object(eng.task, *move_vec)
        ow = obj_world(eng)
        if ow is None or ow[2] < z_floor:
            continue
        realized = ow - ow0
        rn = float(np.linalg.norm(realized))
        cmd_cos = float(np.dot(realized, move_u)) / (rn + 1e-9)
        if rn > max_gain * (k * step_m) or rn < 1e-6 or cmd_cos < min_cos:
            continue
        ob = (R.T @ (ow - t)).astype(np.float32)
        dist = float(np.linalg.norm(ee_b - ob))
        c = bool(eng._read_grasp())
        vec = encode(eng, seed)
        if vec is None:
            continue
        disp = float(k * step_m) * (1.0 if not toward_hand else -1.0)
        rows.append((k, disp, c, dist, vec))
        if verbose:
            log.info("    k=%+d disp=%+.3f dist=%.3f contact=%s gain=%.2f cos=%.2f", k, disp, dist,
                     c, rn / (k * step_m + 1e-9), cmd_cos)
        if c == want_contact:
            crossed = k if crossed is None else crossed
        if crossed is not None and k >= crossed + stop_after:
            break
    if len({r[2] for r in rows}) < 2:
        return None
    return rows, grasp0, ob0, dist0


def offschedule_payload(eng, base, episode_index, house, encode, gripper, *, n_pose=N_POSE,
                        step_m=STEP_M, k_max=K_MAX, k_max_forced=K_MAX_FORCED, verbose=False):
    """The off-schedule harvest file of one rolled-out episode (keys `OFFSCHEDULE_KEYS`), or None
    when no anchor crosses the contact boundary.

    Anchors: `vlageom.sim.kinematics.pick_poses` (hand open; early touch) and `pick_holding_poses`
    (slip), up to n_pose each. Plan, in this order: event 0 at every open-hand anchor, event 2 at
    every open-hand anchor, event 1 at every holding anchor, event 3 at every holding anchor. Before
    each march the scene is restored with the event's hand state and read once (contact, finger
    separation). Events 0 and 2 move the object toward the hand until contact; event 1 moves it away
    until contact ends; event 3 moves it away when the opened hand still touches the object (branch
    +1) and toward the hand otherwise (branch -1). Every (anchor, event) that crosses the boundary
    is one pose (pose_id); qc is the anchor's query index."""
    from vlageom.sim.kinematics import pick_poses
    encode.reset()
    gripper.reset()
    pre = pick_poses(base, n_pose)
    hold = pick_holding_poses(base, n_pose)
    snaps = eng._baseline_snaps
    qsteps = base["query_steps"]
    n_steps = int(base.get("n_steps", 1))
    plan = [(c, 0) for c in pre] + [(c, 2) for c in pre] + [(c, 1) for c in hold] + [(c, 3) for c in hold]

    acts, contacts, disps, dists, pose_ids, step_ks, events, phases, qcs = ([] for _ in range(9))
    grasp0s, chunks, qpos0s, p_events, p_qcs, p_seps, p_gforced, p_dirs = ([] for _ in range(8))
    pid = 0
    for (c, event) in plan:
        step = int(qsteps[c])
        if not (0 <= step < len(snaps)):
            continue
        snap = snaps[step]
        q = int(snap["chunk"])
        seed = eng.scene_seed + 1000 * q
        q0 = np.asarray(base["qpos_arm"][c], np.float32)[:GRIP]
        grip = EVENTS[event][1]
        pose_hand(eng, snap, q0, grip, gripper)
        grasp_forced = bool(eng._read_grasp())
        sep0 = float(gripper.finger_separation())
        if event in (0, 2):
            toward, want = True, True
        elif event == 1:
            toward, want = False, False
        else:
            toward, want = (False, False) if grasp_forced else (True, True)
        if verbose:
            log.info("  %s, anchor q%d (chunk %d): sep0=%.4f contact0=%s, object moves %s",
                     EVENTS[event][0], c, q, sep0, grasp_forced, "out" if not toward else "in")
        res = march_boundary(eng, snap, q0, encode, seed, toward,
                             k_max if grip is None else k_max_forced, want, grip=grip,
                             gripper=gripper, step_m=step_m, verbose=verbose)
        if res is None:
            continue
        rows, grasp0, _ob0, _dist0 = res
        ph = min(step, n_steps - 1) / max(1, n_steps - 1)
        for (k, disp, ctc, dist, vec) in rows:
            acts.append(vec)
            contacts.append(ctc)
            disps.append(disp)
            dists.append(dist)
            pose_ids.append(pid)
            step_ks.append(k)
            events.append(event)
            phases.append(ph)
            qcs.append(c)
        grasp0s.append(grasp0)
        chunks.append(q)
        qpos0s.append(q0)
        p_events.append(event)
        p_qcs.append(c)
        p_seps.append(sep0)
        p_gforced.append(grasp_forced)
        p_dirs.append(1 if not toward else -1)
        pid += 1
    if not acts:
        return None
    payload = dict(
        act=np.stack(acts).astype(np.float16), contact=np.array(contacts, bool),
        disp=np.array(disps, np.float32), dist=np.array(dists, np.float32),
        pose_id=np.array(pose_ids, np.int32), step_k=np.array(step_ks, np.int32),
        event=np.array(events, np.int8), phase=np.array(phases, np.float32),
        qc=np.array(qcs, np.int32),
        grasp0=np.array(grasp0s, bool), pose_chunk=np.array(chunks, np.int32),
        qpos0=np.stack(qpos0s).astype(np.float32), pose_event=np.array(p_events, np.int8),
        pose_qc=np.array(p_qcs, np.int32), pose_sep0=np.array(p_seps, np.float32),
        pose_grasp_forced=np.array(p_gforced, bool), pose_dir=np.array(p_dirs, np.int8),
        layers=np.array(encode.layers), house=int(house), ep=int(episode_index))
    payload["n_txt"] = int(payload["act"].shape[2]) - (2 * N_PATCH + 1)
    payload["txt_ids"] = encode.tail_ids if encode.tail_ids is not None else np.zeros(0, np.int32)
    payload["txt_str"] = np.array([], dtype="U32")
    return payload


# ---- off-schedule pool -------------------------------------------------------------------------- #
def read_text_block(path, layer_index, T):
    """`act[:, li:li+1, 392:392+T+1, :]` of an off-schedule harvest file, read row by row from
    the file (the text tail and the sequence-mean row of one layer). (Npts, 1, T+1, 2560)."""
    off, shape, dtype, fortran = npz_member(path, "act")
    if fortran:
        raise ValueError(f"{path}: act is stored in Fortran order")
    npts, nl, S, D = shape
    lo = 2 * N_PATCH
    if lo + T + 1 > S:
        raise ValueError(f"{path}: act has {S} token rows, expected at least {lo + T + 1}")
    item = np.dtype(dtype).itemsize
    nbytes = (T + 1) * D * item
    blk = np.empty((npts, 1, T + 1, D), dtype)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        for i in range(npts):
            start = off + (((i * nl) + layer_index) * S + lo) * D * item
            buf = os.pread(fd, nbytes, start)
            if len(buf) != nbytes:
                raise IOError(f"{path}: short read at row {i}")
            blk[i, 0] = np.frombuffer(buf, dtype).reshape(T + 1, D)
    finally:
        os.close(fd)
    return blk


def pool_text_block(blk, T, txt_ids):
    """Pools of a (Npts, 1, T+1, 2560) float16 text block: {"txt_all": float32 mean of the text-tail
    tokens other than the chat and multimodal control tokens, "last_tok": the final text token as
    float32}, each (Npts, 2560)."""
    from vlageom.sim.encode import tail_special_mask
    nonspec = ~tail_special_mask(txt_ids) if len(txt_ids) == T else np.ones(T, bool)
    if not nonspec.any():
        nonspec = np.ones(T, bool)
    return {"txt_all": blk[:, :, 0:T][:, :, nonspec].mean(axis=2, dtype=np.float32)[:, 0, :],
            "last_tok": blk[:, :, T - 1, :].astype(np.float32)[:, 0, :]}


def pool_offschedule_shard(path, layer=PROBE_LAYER):
    """One off-schedule harvest file -> ({pool: (Npts, 2560) float32}, per-row fields except
    shard_idx). Per-pose fields (pose_chunk, grasp0, pose_grasp_forced, pose_sep0, pose_dir as
    pdir) are broadcast to the rows of the pose; gpose = ep * 1000 + pose_id."""
    z = np.load(path)
    T = int(z["n_txt"])
    ids = z["txt_ids"]
    hl = [int(x) for x in z["layers"]]
    pools = pool_text_block(read_text_block(path, hl.index(int(layer)), T), T, ids)
    n = len(z["contact"])
    pi = z["pose_id"].astype(np.int64)
    ep = int(z["ep"])
    cols = dict(
        ep=np.full(n, ep, np.int32), house=np.full(n, int(z["house"]), np.int32),
        pose_id=pi.astype(np.int32), gpose=ep * 1000 + pi,
        event=z["event"].astype(np.int8), pdir=z["pose_dir"][pi].astype(np.int8),
        qc=z["qc"].astype(np.int32), step_k=z["step_k"].astype(np.int32),
        contact=z["contact"].astype(bool),
        disp=z["disp"].astype(np.float32), dist=z["dist"].astype(np.float32),
        phase=z["phase"].astype(np.float32),
        pose_chunk=z["pose_chunk"][pi].astype(np.int32),
        grasp0=z["grasp0"][pi].astype(bool),
        pose_grasp_forced=z["pose_grasp_forced"][pi].astype(bool),
        pose_sep0=z["pose_sep0"][pi].astype(np.float32),
        n_txt=np.full(n, T, np.int16))
    return pools, cols


# ---- probes ------------------------------------------------------------------------------------- #
def fit_lda(X, y):
    """StandardScaler + LinearDiscriminantAnalysis (svd solver) on all rows. Returns
    (scaler, lda, readout) with readout(Xnew) = lda.decision_function (positive = contact)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y).astype(int)
    sc = StandardScaler().fit(X)
    lda = LinearDiscriminantAnalysis().fit(sc.transform(X), y)

    def readout(Xnew):
        return lda.decision_function(sc.transform(np.asarray(Xnew, float)))
    return sc, lda, readout


def oof_lda(X, y, houses, folds):
    """Out-of-house LDA: per fold (a `vlageom.probes.FoldMap` over `houses`, or a list of
    (train, test)), StandardScaler + LDA fit on the training rows. Returns dict(auc, bal_acc, acc,
    score): ROC-AUC of the out-of-fold decision scores, balanced accuracy and accuracy of LDA's own
    predictions (decision threshold from the training class priors), and the scores."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from vlageom.probes import FoldMap
    y = np.asarray(y).astype(int)
    splits = folds.splits(houses) if isinstance(folds, FoldMap) else list(folds)
    score = np.full(len(y), np.nan, float)
    pred = np.full(len(y), -1, int)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        lda = LinearDiscriminantAnalysis().fit(sc.transform(X[tr]), y[tr])
        score[te] = lda.decision_function(sc.transform(X[te]))
        pred[te] = lda.predict(sc.transform(X[te]))
    ok = np.isfinite(score)
    yy = y[ok]
    auc = float(roc_auc_score(yy, score[ok])) if len(np.unique(yy)) == 2 else float("nan")
    good = ok.any() and (pred[ok] >= 0).all()
    bal = float(balanced_accuracy_score(yy, pred[ok])) if good else float("nan")
    acc = float((pred[ok] == yy).mean()) if good else float("nan")
    return {"auc": auc, "bal_acc": bal, "acc": acc, "score": score}


def within_pose(score, contact, gpose, mask=None):
    """Within-pose detection over the rows in `mask`: for every pose with both contact classes,
    delta = mean score of the contact rows - mean score of the no-contact rows, and the pose's AUC
    (contact rows ranked above no-contact rows, ties count one half). Returns n_pose, mean_delta,
    frac_pos (fraction of poses with delta > 0) with its binomial SE frac_pos_se, auc (per-pose AUCs
    weighted by the number of contact x no-contact pairs) with auc_se (weighted SD over the Kish
    effective number of poses)."""
    score = np.asarray(score, float)
    contact = np.asarray(contact).astype(bool)
    m = np.ones(len(score), bool) if mask is None else np.asarray(mask, bool)
    deltas, aucs, wts, npos = [], [], [], 0
    for p in np.unique(gpose[m]):
        sel = m & (gpose == p)
        cc = contact[sel]
        ss = score[sel]
        if cc.all() or (~cc).all():
            continue
        deltas.append(float(ss[cc].mean() - ss[~cc].mean()))
        on, off = ss[cc], ss[~cc]
        wins = (on[:, None] > off[None, :]).sum() + 0.5 * (on[:, None] == off[None, :]).sum()
        aucs.append(float(wins) / (len(on) * len(off)))
        wts.append(len(on) * len(off))
        npos += (deltas[-1] > 0)
    if not deltas:
        return {"n_pose": 0, "mean_delta": float("nan"), "frac_pos": float("nan"),
                "auc": float("nan"), "auc_se": float("nan"), "frac_pos_se": float("nan")}
    wts = np.array(wts, float)
    p = npos / len(deltas)
    auc_w = float(np.average(aucs, weights=wts))
    n_eff = float(wts.sum() ** 2 / (wts ** 2).sum())
    auc_se = float(np.sqrt(np.average((np.asarray(aucs) - auc_w) ** 2, weights=wts)
                           / max(n_eff - 1.0, 1.0)))
    return {"n_pose": len(deltas), "mean_delta": float(np.mean(deltas)),
            "frac_pos": float(p), "frac_pos_se": float(np.sqrt(p * (1.0 - p) / len(deltas))),
            "auc": auc_w, "auc_se": auc_se}


def per_pose_auc(score, contact, gpose):
    """{gpose: (auc, n_pairs)} over the poses with both contact classes."""
    out = {}
    for p in np.unique(gpose):
        sel = gpose == p
        cc, ss = contact[sel], score[sel]
        if cc.all() or (~cc).all():
            continue
        on, off = ss[cc], ss[~cc]
        wins = (on[:, None] > off[None, :]).sum() + 0.5 * (on[:, None] == off[None, :]).sum()
        out[int(p)] = (float(wins) / (len(on) * len(off)), len(on) * len(off))
    return out


def crossing_window_mask(gpose, stepk, contact, halfwidth=3):
    """True for rows within +-halfwidth steps of their pose's contact change (the first step whose
    contact differs from the unmoved row's)."""
    m = np.zeros(len(gpose), bool)
    for p in np.unique(gpose):
        sel = np.where(gpose == p)[0]
        order = sel[np.argsort(stepk[sel])]
        c0 = contact[order[0]]
        diff = order[contact[order] != c0]
        if len(diff) == 0:
            continue
        kstar = stepk[diff[0]]
        m[sel[np.abs(stepk[sel] - kstar) <= halfwidth]] = True
    return m


def cell_mask(D, cell):
    """Rows of the off-schedule pool `D` in a scored cell (`CELLS`)."""
    ev, pd = CELLS[cell]
    m = D["event"] == ev
    if pd:
        m &= D["pdir"] == pd
    return m


def pose_matched(D, s):
    """Forced-hand minus natural-hand per-pose AUC at the same anchor, on the rows within 3 steps
    of each pose's contact change: early touch (event 2 - event 0) and slip (event 3 moved out of
    the opened hand - event 1). Returns {family: {n_pairs, mean_delta_forced_minus_natural, sem}}."""
    wm = crossing_window_mask(D["gpose"], D["step_k"], D["contact"])
    pa = per_pose_auc(s[wm], D["contact"][wm], D["gpose"][wm])
    _up, _ui = np.unique(D["gpose"], return_index=True)
    ganchor = (D["gpose"] // 1000) * 100000 + D["qc"]
    anc = dict(zip(_up.tolist(), ganchor[_ui].tolist()))
    pev = dict(zip(_up.tolist(), D["event"][_ui].astype(int).tolist()))
    pdr = dict(zip(_up.tolist(), D["pdir"][_ui].astype(int).tolist()))
    res = {}
    for fam, (en, ef) in {"early_touch": (0, 2), "slip": (1, 3)}.items():
        by = {}
        for p, (auc, _w) in pa.items():
            if pev[p] in (en, ef):
                d = by.setdefault(anc[p], {})
                d[pev[p]] = auc
                if pev[p] == ef:
                    d["dir"] = pdr[p]
        pairs = [(d[ef] - d[en], d.get("dir", 0)) for d in by.values() if en in d and ef in d]
        if fam == "slip":
            pairs = [x for x in pairs if x[1] == 1]
        dl = np.array([x for x, _ in pairs])
        res[fam] = {"n_pairs": int(len(dl)),
                    "mean_delta_forced_minus_natural": float(dl.mean()) if len(dl) else float("nan"),
                    "sem": float(dl.std() / np.sqrt(len(dl))) if len(dl) > 1 else float("nan")}
    res["slip"]["note"] = "OUT branch only (true open-hand slip vs event 1)"
    return res


_G: dict = {}           # fork-shared arrays of the per-house refits


def _fit_house(h):
    """Refit the probe of every pool on the natural rows of all houses but h; score the store rows
    of house h."""
    m = _G["nat_house"] != h
    out = {"house": int(h), "store": {}}
    for p in _G["pools"]:
        _, _, ro = fit_lda(_G["Xnat"][p][m], _G["ynat"][m])
        out["store"][p] = ro(_G["Xst"][p][_G["st_house"] == h])
    return out


def heldout_scores(Xnat, ynat, nat_house, Xst, st_house, workers=4, progress=None):
    """Held-out-house probe scores of the off-schedule rows, per pool.

    Xnat, Xst: {pool: (n, d) float32}. Every store row starts with the score of the probe fit on all
    natural rows; the rows of each store house that also has natural rows are then rescored by the
    probe fit on the natural rows of all other houses (one fit per house and pool, in `workers`
    forked processes). Returns {pool: (n_store,) float64}."""
    from concurrent.futures import ProcessPoolExecutor
    pools = list(Xst)
    S = {p: fit_lda(Xnat[p], ynat)[2](Xst[p]) for p in pools}
    houses = sorted(set(np.unique(st_house).tolist()) & set(np.unique(nat_house).tolist()))
    _G.update(Xnat=Xnat, ynat=ynat, nat_house=nat_house, Xst=Xst, st_house=st_house, pools=pools)
    try:
        if workers > 1:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                results = ex.map(_fit_house, houses)
                for i, r in enumerate(results):
                    for p in pools:
                        S[p][st_house == r["house"]] = r["store"][p]
                    if progress is not None:
                        progress(i + 1, len(houses))
        else:
            for i, h in enumerate(houses):
                r = _fit_house(h)
                for p in pools:
                    S[p][st_house == h] = r["store"][p]
                if progress is not None:
                    progress(i + 1, len(houses))
    finally:
        _G.clear()
    return S


def detection_table(D, S, natural_auc, layer=PROBE_LAYER):
    """{pool: {layer, natural_auc, all, <cell>..., pose_matched}} from the off-schedule pool `D`
    and the per-pool scores `S`; each cell is a `within_pose` result plus n_rows."""
    out = {}
    for p, sc in S.items():
        r = {"layer": int(layer), "natural_auc": float(natural_auc[p])}
        r["all"] = within_pose(sc, D["contact"], D["gpose"], mask=np.ones(len(sc), bool))
        for c in CELLS:
            m = cell_mask(D, c)
            r[c] = within_pose(sc, D["contact"], D["gpose"], mask=m)
            r[c]["n_rows"] = int(m.sum())
        r["pose_matched"] = pose_matched(D, sc)
        out[p] = r
    return out


# ---- panel a: principal components of the natural cloud ---------------------------------------- #
def pca3(X, contact):
    """Top-3 principal components of X (float32, rows = natural queries) from the eigendecomposition
    of the covariance; each component's sign is set so that the mean projection of the contact rows
    exceeds that of the no-contact rows. Returns (P (n, 3), evr (3,), origin_p (3,)): projections of
    the centred rows, explained-variance ratios and the projection of the zero vector."""
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    w, V = np.linalg.eigh(C)
    W3 = V[:, ::-1][:, :3]
    for k in range(3):
        if (Xc[contact] @ W3[:, k]).mean() < (Xc[~contact] @ W3[:, k]).mean():
            W3[:, k] = -W3[:, k]
    P = Xc @ W3
    evr = w[::-1][:3] / w.sum()
    origin_p = -mu @ W3
    return P, evr, origin_p


N_SHOW = 6000          # points drawn in panel a (the PCA uses all rows)
SHOW_SEED = 0


def show_index(n, n_show=N_SHOW, seed=SHOW_SEED):
    """The rows drawn in panel a: numpy default_rng(seed).permutation(n)[:n_show]."""
    return np.random.default_rng(seed).permutation(n)[:n_show]


def bar_arrays(table, pool="txt_all"):
    """Panel b arrays from a `detection_table` entry: (cells, frac_pos, frac_pos_se, n_pose) over
    `BAR_CELLS`."""
    r = table[pool]
    return (np.array(BAR_CELLS), np.array([r[c]["frac_pos"] for c in BAR_CELLS], np.float64),
            np.array([r[c]["frac_pos_se"] for c in BAR_CELLS], np.float64),
            np.array([r[c]["n_pose"] for c in BAR_CELLS], np.int64))

