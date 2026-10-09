"""Simulation side: MolmoSpaces env + MolmoBot-DROID policy, the BranchEngine, scene edits,
rendering and activation readout.

Importing this package sets up the file locks (`vlageom.sim.locks`: no-op locks when
VLAGEOM_NOOP_FILELOCK=1, as set by scripts/harvest_env.sh), so import it before anything imports
`molmo_spaces`. No simulator, torch or policy code is imported until a function needs it.

Modules:
  env         build_molmobot_env, SimEnv, resolve_checkpoint, get_tokenizer
  engine      BranchEngine (rollout with per-query context capture and per-step snapshots, flow-loop
              access, gripper edit fits, readers), seed_all, gripper_of
  render      install_query_frame_rendering, refresh_cameras, fresh_obs, observe_all_cameras,
              render_real, image helpers
  scene       get_pickup_body_name, set_body_pos, move_pickup_object, base_transform, to_base,
              envelope_ok, obj_root_id, bad_contact_count, sample_targets, resolve_surface,
              hold_robot, settle
  kinematics  get_arm_movegroup, set_arm_qpos, arm_joint_limits, pick_poses, GripperControl
  encode      encode_obs, run_chunks, encode_query, capture_kw, capture_blocks,
              llm_variant_pools, token layout helpers, LLM_LAYERS / AE_LAYERS / VARIANTS
  bench       BENCH_CONFIGS, build_house_list, install_model_cache, bench_engine,
              sample_bench_episode
  so3         so3_angle, so3_axis, so3_geodesic, signed_angle_about
"""
from vlageom.sim import locks as _locks

_locks.install()

from vlageom.sim.bench import (BENCH_CONFIGS, bench_engine, build_house_list,  # noqa: E402
                               install_model_cache, sample_bench_episode)
from vlageom.sim.encode import (AE_LAYERS, LLM_LAYERS, OPTIONAL_VARIANTS, VARIANTS,  # noqa: E402
                                capture_blocks, capture_kw, encode_obs, encode_query,
                                llm_variant_pools, run_chunks)
from vlageom.sim.engine import BranchEngine, gripper_of, seed_all  # noqa: E402
from vlageom.sim.env import (SimEnv, build_molmobot_env, get_tokenizer,  # noqa: E402
                             resolve_checkpoint)
from vlageom.sim.kinematics import (GripperControl, arm_joint_limits,  # noqa: E402
                                    get_arm_movegroup, pick_poses, set_arm_qpos)
from vlageom.sim.render import (as_hw3, exo_frame, fresh_obs,  # noqa: E402
                                install_query_frame_rendering, observe_all_cameras, obs_frames,
                                refresh_cameras, render_real, thumb, wrist_cam_name,
                                wrist_frame_real)
from vlageom.sim.scene import (bad_contact_count, base_transform, envelope_ok,  # noqa: E402
                               get_pickup_body_name, hold_robot, mj_handles, move_pickup_object,
                               obj_root_id, resolve_surface, sample_targets, set_body_pos, settle,
                               to_base)

__all__ = [
    "AE_LAYERS", "BENCH_CONFIGS", "BranchEngine", "GripperControl", "LLM_LAYERS",
    "OPTIONAL_VARIANTS", "SimEnv", "VARIANTS", "arm_joint_limits", "as_hw3", "bad_contact_count",
    "base_transform", "bench_engine", "build_house_list", "build_molmobot_env", "capture_blocks",
    "capture_kw", "encode_obs", "encode_query", "envelope_ok", "exo_frame", "fresh_obs",
    "get_arm_movegroup", "get_pickup_body_name", "get_tokenizer", "gripper_of", "hold_robot",
    "install_model_cache", "install_query_frame_rendering", "llm_variant_pools", "mj_handles",
    "move_pickup_object", "obj_root_id", "observe_all_cameras", "obs_frames", "pick_poses",
    "refresh_cameras", "render_real", "resolve_checkpoint", "resolve_surface", "run_chunks",
    "sample_bench_episode", "sample_targets", "seed_all", "set_arm_qpos", "set_body_pos", "settle",
    "thumb", "to_base", "wrist_cam_name", "wrist_frame_real",
]
