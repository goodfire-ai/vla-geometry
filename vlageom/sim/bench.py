"""MolmoSpaces benchmark episodes as the scene source for the object-teleport harvests.

`build_house_list` turns the two benchmark JSON sets (FrankaPickDroidMiniBench and
FrankaPickHardBench) into one deterministic episode per house; `sample_bench_episode` loads that
exact episode into a BranchEngine. `bench_engine` builds the engine and one env per benchmark.
"""
from __future__ import annotations

from pathlib import Path

from vlageom import paths

BENCH_CONFIGS = {"mini": "FrankaPickDroidMiniBench", "hard": "FrankaPickHardBench"}


def find_bench_dir(root) -> Path:
    """The directory holding `benchmark.json` under `root` (directory names vary by release)."""
    hits = sorted(Path(root).expanduser().glob("**/benchmark.json"))
    if not hits:
        raise SystemExit(f"no benchmark.json under {root}")
    return hits[0].parent


def build_house_list(bench_roots: dict | None = None, n_houses: int = 950) -> list:
    """[(house, bench, bench_idx, spec), ...]: the first episode of each house in each benchmark,
    the hard benchmark's episode preferred when a house is in both, sorted by house index and capped
    at `n_houses`. `bench_roots` = {"mini": dir, "hard": dir} (default `paths.bench_roots()`)."""
    from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
    bench_roots = paths.bench_roots() if bench_roots is None else bench_roots
    chosen: dict = {}
    for bench in ("mini", "hard"):                       # hard second: overrides mini
        root = bench_roots.get(bench)
        if not root:
            continue
        eps = load_all_episodes(find_bench_dir(root))
        if not eps:
            raise SystemExit(f"load_all_episodes returned [] for {root}")
        seen = set()
        for i, sp in enumerate(eps):
            h = int(sp.house_index)
            if h in seen:
                continue
            seen.add(h)
            if bench == "hard" or h not in chosen:
                chosen[h] = (bench, i, sp)
    houses = sorted(chosen)[:n_houses]
    return [(h, *chosen[h]) for h in houses]


def install_model_cache() -> None:
    """Load the checkpoint once per process: every policy built afterwards (one per episode) reuses
    the loaded model instead of reading model.pt again."""
    import olmo.models.molmobot.inference_wrapper as _IW
    if getattr(_IW.SynthManipMolmoInferenceWrapper, "_cache_installed", False):
        return
    _orig = _IW.SynthManipMolmoInferenceWrapper._load_checkpoint
    _CACHE: dict = {}
    _ATTRS = ["model", "model_config", "max_seq_len", "action_horizon",
              "action_dim", "n_obs_steps", "num_flow_steps"]

    def _cached(self):
        key = self.checkpoint_path
        if key in _CACHE:
            for a, v in _CACHE[key].items():
                setattr(self, a, v)
            return
        _orig(self)
        _CACHE[key] = {a: getattr(self, a) for a in _ATTRS}

    _IW.SynthManipMolmoInferenceWrapper._load_checkpoint = _cached
    _IW.SynthManipMolmoInferenceWrapper._cache_installed = True


def bench_engine(checkpoint: str | None = None, output_dir: str | None = None, *,
                 need_mini: bool = True):
    """(eng, sim_envs): a loaded BranchEngine on the hard-benchmark config and {"hard": env,
    "mini": env} (mini only when `need_mini`). Installs the model cache first."""
    from vlageom.sim.engine import BranchEngine
    from vlageom.sim.env import build_molmobot_env

    install_model_cache()
    eng = BranchEngine({"config": BENCH_CONFIGS["hard"], "checkpoint": checkpoint,
                        "output_dir": output_dir})
    eng.load()
    sim_envs = {"hard": eng.sim_env}
    if need_mini:
        sim_envs["mini"] = build_molmobot_env(BENCH_CONFIGS["mini"], checkpoint,
                                              output_dir=output_dir)
    return eng, sim_envs


def sample_bench_episode(eng, sim_envs: dict, bench: str, spec, seed: int, task_horizon: int) -> None:
    """Load benchmark episode `spec` into `eng`: a JsonEvalTaskSampler over that spec replaces the
    config's default sampler, the policy reads the spec's exo and wrist cameras, and `eng.sample`
    builds the episode with all RNGs seeded by `seed`."""
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
    se = sim_envs[bench]
    se.exp_config.task_horizon = task_horizon
    se._task_sampler = JsonEvalTaskSampler(se.exp_config, episode_spec=spec)
    cams = [c.name for c in se.exp_config.camera_config.cameras]
    exo = next((c for c in cams if "wrist" not in c.lower()), cams[0])
    wrist = next((c for c in cams if "wrist" in c.lower()), cams[-1])
    se.exp_config.policy_config.camera_names = [exo, wrist]
    eng.sim_env = se
    eng.sample(int(spec.house_index), seed)
