"""Camera rendering and observation helpers.

Two rules hold for every path in this package that renders an observation:

  * Robot-mounted cameras (the wrist camera) follow the robot body they are attached to.
    MolmoSpaces moves them inside `env.step()`; any code that changes the robot state without
    `env.step()` (snapshot restore, setting joint angles, physics settling) must call
    `refresh_cameras(env)` before rendering. `fresh_obs(eng)` does both and is the helper every
    teleport, sweep and settle path uses to build the observation the policy reads.

  * `install_query_frame_rendering(task, policy)` (called by `BranchEngine.sample`) renders camera
    frames only on the steps whose observation the policy reads. MolmoBot queries the model once
    every `execute_horizon` (8) steps and executes the rest of the action chunk open-loop, so the
    other frames are never looked at; on those steps the last rendered frame is returned. The
    policy and the frame cache are set for every episode.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def install_query_frame_rendering(task, policy) -> None:
    """Render camera frames only when the next policy query will read them; between queries
    `env.render_rgb_frame` returns the last rendered frame of that camera.

    The observation rendered inside `task.step()` at step t is read by `get_action` at t+1, which
    queries the model iff `policy.buffer_index >= execute_horizon` or the action buffer is empty. At
    render time that state is already set, so frames are rendered exactly on pre-query steps. The
    model's two-frame input (n_obs_steps=2, obs_step_delta=execute_horizon) reads only
    query-aligned frames, which are always rendered.

    The policy and the frame cache are kept on the env and set here for every episode; the wrapper
    is installed once per env. The renderer that renders every call is kept as
    `env._render_all_frames`."""
    env = task._env
    env._query_frames_policy = policy
    env._query_frames_cache = {}
    if getattr(env, "_query_frames_installed", False):
        return
    render_all = env.render_rgb_frame
    env._render_all_frames = render_all

    def _render_query_frames(camera_name):
        policy = env._query_frames_policy
        cache = env._query_frames_cache
        eh = int(getattr(policy, "execute_horizon", 8) or 8)
        will_query = (int(getattr(policy, "buffer_index", 0)) >= eh
                      or not getattr(policy, "action_buffer", None))
        if will_query:
            f = render_all(camera_name)
            cache[camera_name] = f
            return f
        c = cache.get(camera_name)
        return c if c is not None else render_all(camera_name)

    env.render_rgb_frame = _render_query_frames
    env._query_frames_installed = True


def refresh_cameras(env) -> list:
    """Move every robot-mounted camera to its body's current pose (no-op for static cameras and
    for bodies that have not moved). Returns the names of the cameras that moved."""
    return env.camera_manager.registry.update_all_cameras(env)


def observe_all_cameras(eng):
    """`task.get_observations()` at the current sim state, rendering every camera now (not from the
    query-frame cache). Does not move cameras; use `fresh_obs` unless the cameras were refreshed
    already."""
    env = eng.task._env
    cur = getattr(env, "render_rgb_frame", None)
    render_all = getattr(env, "_render_all_frames", None)
    if render_all is not None:
        env.render_rgb_frame = render_all
    try:
        return eng.task.get_observations()
    finally:
        if render_all is not None:
            env.render_rgb_frame = cur


def fresh_obs(eng):
    """The policy observation at the current sim state: refresh robot-mounted cameras, then render
    every camera and read proprioception. Call after any state change made without `env.step()`
    (snapshot restore, `set_arm_qpos`, `set_body_pos`, `settle`)."""
    refresh_cameras(eng.task._env)
    return observe_all_cameras(eng)


def render_real(eng, camera_name: str):
    """One render of `camera_name` at the current state (not from the query-frame cache), for
    saving frames. It does not change what the policy reads. Does not move cameras."""
    env = eng.task._env
    render_all = getattr(env, "_render_all_frames", None) or env.render_rgb_frame
    return render_all(camera_name)


def camera_names(eng) -> list[str]:
    try:
        return list(eng.task._env.camera_manager.registry.cameras.keys())
    except Exception:
        return []


def wrist_cam_name(eng):
    """Registry name of the wrist camera, or None."""
    return next((n for n in camera_names(eng) if "wrist" in n.lower()), None)


# ---- images inside an observation ------------------------------------------------------------ #
def iter_arrays(o, prefix=""):
    """Yield (path, ndarray) for every array in a nested observation (dict / list / tensor)."""
    if hasattr(o, "detach"):
        try:
            o = o.detach().cpu().numpy()
        except Exception:
            return
    if isinstance(o, np.ndarray):
        yield prefix, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from iter_arrays(v, f"{prefix}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from iter_arrays(v, f"{prefix}[{i}]")


def as_hw3(a):
    """An (H, W, 3) uint8 image from an image-shaped array (HWC or CHW, uint8 or [0, 1] float), or
    None if the array is not image-shaped."""
    if a.ndim != 3:
        return None
    if a.shape[-1] == 3:
        img = a
    elif a.shape[0] == 3:
        img = np.transpose(a, (1, 2, 0))
    else:
        return None
    if min(img.shape[:2]) < 16:
        return None
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 255) if float(img.max()) > 1.5 else np.clip(img * 255, 0, 255)).astype(np.uint8)
    return np.ascontiguousarray(img)


_OBS_LOGGED = False


def exo_frame(obs):
    """The exo (non-wrist) camera image (uint8 HxWx3) of an observation, or None."""
    global _OBS_LOGGED
    cands = []
    for path, a in iter_arrays(obs):
        img = as_hw3(a)
        if img is not None:
            cands.append((path, img))
    if not _OBS_LOGGED:
        _OBS_LOGGED = True
        log.info("obs image candidates: %s", [(p, c.shape) for p, c in cands] or "NONE")
    if not cands:
        return None
    return ([c for c in cands if "wrist" not in c[0].lower()] or cands)[0][1]


def obs_frames(obs):
    """(exo, wrist) uint8 HxWx3 images of an observation; the array path names the wrist camera."""
    exo = wrist = None
    for path, a in iter_arrays(obs):
        img = as_hw3(a)
        if img is None:
            continue
        if "wrist" in path.lower():
            wrist = img if wrist is None else wrist
        else:
            exo = img if exo is None else exo
    return exo, wrist


def wrist_frame_real(eng, name):
    """A real wrist-camera render (uint8 HxWx3, copied out of the renderer's buffer) at the current
    state, or None. The renderer reuses one output buffer, so the copy is required to keep frames."""
    if name is None:
        return None
    try:
        raw = render_real(eng, name)
    except Exception:
        return None
    if raw is None:
        return None
    img = as_hw3(np.asarray(raw))
    return None if img is None else np.array(img, copy=True)


def thumb(img, size: int = 96):
    """Nearest-neighbour size x size thumbnail of an HxWx3 image (uint8)."""
    h, w = img.shape[:2]
    ys = np.linspace(0, h - 1, size).astype(int)
    xs = np.linspace(0, w - 1, size).astype(int)
    return np.asarray(img)[ys][:, xs].astype(np.uint8)
