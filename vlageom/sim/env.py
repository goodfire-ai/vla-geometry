"""MolmoSpaces environment + MolmoBot-DROID policy.

`build_molmobot_env(config_name, checkpoint)` returns a `SimEnv` whose policy is MolmoBot-DROID
(`olmo.eval.configure_real_robot.RealRobotVLAPolicy`) reading two cameras, the exo camera and the
wrist camera, with joint-position actions for arm and gripper. `SimEnv.sample_episode(house)` builds
the scene, the task and a fresh policy for one episode.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from vlageom import paths

log = logging.getLogger(__name__)


def _import_all_configs() -> None:
    """Import every MolmoSpaces config module so the config registry is populated."""
    from molmo_spaces.data_generation.config_registry import _MJT_CONFIG_REGISTRY

    if len(_MJT_CONFIG_REGISTRY) > 0:
        return
    import molmo_spaces.data_generation.config.door_opening_configs  # noqa: F401
    import molmo_spaces.data_generation.config.nav_to_obj_configs  # noqa: F401
    import molmo_spaces.data_generation.config.object_manipulation_datagen_configs  # noqa: F401

    try:
        import molmo_spaces.evaluation.configs.evaluation_configs  # noqa: F401
    except Exception:
        pass


def make_config(config_name: str, **overrides: Any):
    """Instantiate a MolmoSpaces experiment config by registry name, with optional overrides."""
    from molmo_spaces.data_generation.config_registry import get_config_class

    _import_all_configs()
    cls = get_config_class(config_name)
    return cls(**overrides) if overrides else cls()


class SimEnv:
    """A MolmoSpaces config plus its task sampler; `sample_episode` returns (task, policy).

    Usage::

        env = build_molmobot_env("FrankaPickDroidDataGenConfig", paths.checkpoint())
        task, policy = env.sample_episode(house_index=0)
        obs, info = task.reset()
        while not task.is_done():
            action = policy.get_action(obs)
            obs, reward, term, trunc, info = task.step(action)
    """

    def __init__(self, config_name: str, output_dir: str | None = None, **config_overrides: Any):
        self.config_name = config_name
        overrides = dict(config_overrides)
        if output_dir is not None:
            overrides["output_dir"] = Path(output_dir)
        self.exp_config = make_config(config_name, **overrides)
        self._task_sampler = None
        self._task = None
        self._policy = None

    @property
    def task(self):
        return self._task

    @property
    def policy(self):
        return self._policy

    @property
    def task_sampler(self):
        """The MolmoSpaces task sampler, or None before the first episode."""
        return self._task_sampler

    def ensure_task_sampler(self) -> None:
        """Create the MolmoSpaces task sampler if it does not exist yet, without sampling.

        When a task sampler is created it draws a number from numpy's global RNG and reseeds
        python `random`, numpy and torch with it (the config sets no sampler seed). An episode
        sampled by a newly created sampler therefore starts from that reseeded RNG state, and an
        episode sampled by an existing sampler starts from the RNG state set before the call."""
        if self._task_sampler is None:
            self._task_sampler = self.exp_config.task_sampler_config.task_sampler_class(
                self.exp_config
            )

    def sample_episode(self, house_index: int = 0, reload_house: bool = False):
        """Sample a new episode: scene, task and policy. Call `task.reset()` to start it.

        The task sampler is created on the first call (`ensure_task_sampler`). When the house is
        the one already loaded, MolmoSpaces keeps the loaded scene and its per-house sampling
        state; `reload_house=True` loads the house from file and resets that state, as for a
        house that is not loaded."""
        if self._task_sampler is None:
            self.ensure_task_sampler()
        elif reload_house:
            self._task_sampler._last_loaded_house_index = None
        task = self._task_sampler.sample_task(house_index=house_index)
        policy = self.exp_config.policy_config.policy_cls(self.exp_config, task)
        task.register_policy(policy)
        self._task = task
        self._policy = policy
        return task, policy

    def close(self) -> None:
        if self._task_sampler is not None:
            self._task_sampler.close()
            self._task_sampler = None


def resolve_checkpoint(checkpoint: str | None = None) -> str:
    """Local checkpoint directory for `checkpoint` (default `paths.checkpoint()`).

    A local directory must hold `model.pt` and `config.yaml`. A Hugging Face repo id ("org/name")
    is downloaded with `huggingface_hub.snapshot_download` into the Hugging Face cache, which needs
    network access (HF_HUB_OFFLINE unset or 0); the paper's repo id is pinned to the revision the
    paper used. `scripts/prefetch_assets.py` downloads the checkpoint once, before any harvest."""
    ck = checkpoint if checkpoint is not None else paths.checkpoint()
    if "/" in ck and not ck.startswith(("/", "~", ".")) and not Path(ck).exists():
        from huggingface_hub import snapshot_download

        rev = paths.CHECKPOINT_REVISION if ck == paths.CHECKPOINT_REPO else None
        log.info("Downloading %s (revision %s)", ck, rev or "latest")
        try:
            return snapshot_download(ck, revision=rev)
        except Exception as e:
            raise RuntimeError(
                f"could not get checkpoint {ck!r} from Hugging Face ({type(e).__name__}: {e}); "
                "with HF_HUB_OFFLINE=1 (set by scripts/harvest_env.sh) it must already be in the "
                "Hugging Face cache: run scripts/prefetch_assets.py once with network access, or "
                "set VLAGEOM_CHECKPOINT to a local directory") from e
    local = Path(os.path.expanduser(ck))
    missing = [f for f in ("model.pt", "config.yaml") if not (local / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint directory {local} has no {' or '.join(missing)}; download MolmoBot-DROID "
            f"there (scripts/prefetch_assets.py, or `huggingface-cli download "
            f"{paths.CHECKPOINT_REPO} --revision {paths.CHECKPOINT_REVISION} --local-dir {local}`) "
            "or set VLAGEOM_CHECKPOINT to the directory that holds it")
    return os.path.expanduser(ck)


def get_tokenizer(checkpoint=None):
    """The policy's tokenizer (Qwen3 with the multimodal special tokens), built from the
    checkpoint's config.yaml."""
    from olmo.models.model_config import BaseModelConfig

    ck = resolve_checkpoint(checkpoint)
    cfg = BaseModelConfig.load(str(Path(os.path.expanduser(ck)) / "config.yaml"), key="model")
    return cfg.build_tokenizer()


def build_molmobot_env(config_name: str, checkpoint: str | None = None, *,
                       output_dir: str | None = None) -> SimEnv:
    """A `SimEnv` for `config_name` whose policy is MolmoBot-DROID loaded from `checkpoint`."""
    checkpoint = resolve_checkpoint(checkpoint)
    sim_env = SimEnv(config_name, output_dir=output_dir)

    from olmo.eval.configure_real_robot import RealRobotVLAPolicy, RealRobotVLAPolicyConfig

    cam_names = [c.name for c in sim_env.exp_config.camera_config.cameras]
    exo_cam = next((n for n in cam_names if "wrist" not in n), cam_names[0])
    wrist_cam = next((n for n in cam_names if "wrist" in n), cam_names[-1])
    log.info("Policy cameras: exo=%s wrist=%s (env cams: %s)", exo_cam, wrist_cam, cam_names)

    sim_env.exp_config.policy_config = RealRobotVLAPolicyConfig(
        checkpoint_path=checkpoint,
        camera_names=[exo_cam, wrist_cam],
        action_type="joint_pos",
        action_keys={"arm": "joint_pos", "gripper": "joint_pos"},
    )
    sim_env.exp_config.policy_config.policy_cls = RealRobotVLAPolicy
    return sim_env
