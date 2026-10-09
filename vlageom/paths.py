"""Filesystem locations used by the pipelines and figures.

Every location comes from an environment variable with a default. Nothing else in the package
names an absolute path. The shipped lists (`lists/`) and the default data root (`data/`) are read
from the repository checkout, so the package is used from the checkout (`pip install -e .`, or
running the stages from the repository); see `repo_root`.

  VLAGEOM_DATA         root for harvested data, intermediate arrays and panel files
                       (default: <repo>/data)
  VLAGEOM_CHECKPOINT   MolmoBot-DROID checkpoint: a local directory with model.pt + config.yaml, or
                       a Hugging Face repo id such as "allenai/MolmoBot-DROID"
                       (default: ~/checkpoints/MolmoBot-DROID)
  VLAGEOM_FONT_DIR     optional directory that holds the figure font files (.ttf / .otf)
  VLAGEOM_FIGURES      where rendered figures are written (default: <data root>/figures)
  VLAGEOM_BENCH_MINI   MolmoSpaces FrankaPickDroidMiniBench episode directory
  VLAGEOM_BENCH_HARD   MolmoSpaces FrankaPickHardBench episode directory
                       (both default to the MolmoSpaces resource cache, MLSPACES_CACHE_DIR, whose own
                       default is ~/.cache/molmo-spaces-resources)

Layout under the data root: one directory per pipeline (`data_root()/<name>/`, see `tile_dir`)
and `data_root()/panels/*.npz` for the arrays the figure panels are drawn from.
"""
from __future__ import annotations

import os
from pathlib import Path

# The checkpoint the paper used (Hugging Face repo id and the revision that was downloaded).
CHECKPOINT_REPO = "allenai/MolmoBot-DROID"
CHECKPOINT_REVISION = "cbe6ec358958d07ddfb20d3aa54e560e9e1b18c9"
DEFAULT_CHECKPOINT = "~/checkpoints/MolmoBot-DROID"

# MolmoSpaces benchmark snapshot used for the object-teleport harvests.
BENCH_NAME = "molmospaces-bench-v1"
BENCH_VERSION = "20260408"
BENCH_SCENES = "procthor-10k"
BENCH_DIRS = {"mini": "FrankaPickDroidMiniBench", "hard": "FrankaPickHardBench"}


def repo_root() -> Path:
    """The repository checkout (the directory that contains `vlageom/` and `lists/`).

    Raises if `vlageom` was not imported from a checkout (for example after a non-editable
    `pip install .`, which copies only the package): install it with `pip install -e .`."""
    root = Path(__file__).resolve().parent.parent
    if not (root / "lists").is_dir():
        raise RuntimeError(
            f"vlageom is imported from {root}, which is not a checkout of the repository (no "
            "lists/ directory). Install the repository in editable mode (`pip install -e .` in "
            "the checkout) so the shipped lists and the default data root are found.")
    return root


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var, "").strip()
    return Path(v).expanduser() if v else default


def data_root() -> Path:
    """Root of all data written or read by the pipelines ($VLAGEOM_DATA, default <repo>/data)."""
    v = os.environ.get("VLAGEOM_DATA", "").strip()
    return Path(v).expanduser() if v else repo_root() / "data"


def tile_dir(tile: str, *parts: str, create: bool = False) -> Path:
    """`data_root()/<tile>/<parts...>`; with create=True the directory is made if missing."""
    p = data_root() / tile
    for s in parts:
        p = p / s
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def panels_dir(create: bool = False) -> Path:
    """`data_root()/panels`: one `<tile>.npz` per figure tile."""
    return tile_dir("panels", create=create)


def figures_dir(create: bool = False) -> Path:
    """Output directory for rendered figures ($VLAGEOM_FIGURES, default <data root>/figures)."""
    p = _env_path("VLAGEOM_FIGURES", data_root() / "figures")
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def lists_dir() -> Path:
    """`<repo>/lists`: the shipped episode, house and fold lists."""
    return repo_root() / "lists"


def checkpoint() -> str:
    """The policy checkpoint ($VLAGEOM_CHECKPOINT, default ~/checkpoints/MolmoBot-DROID).

    A value of the form "org/name" that is not an existing local path is returned unchanged and is
    treated as a Hugging Face repo id by `vlageom.sim.env.resolve_checkpoint`."""
    v = os.environ.get("VLAGEOM_CHECKPOINT", "").strip() or DEFAULT_CHECKPOINT
    p = Path(v).expanduser()
    if p.exists() or v.startswith(("/", "~", ".")) or "/" not in v:
        return str(p)
    return v


def font_dir() -> Path | None:
    """Directory with figure font files ($VLAGEOM_FONT_DIR), or None when unset or missing."""
    v = os.environ.get("VLAGEOM_FONT_DIR", "").strip()
    if not v:
        return None
    p = Path(v).expanduser()
    return p if p.is_dir() else None


def mlspaces_cache_dir() -> Path:
    """The MolmoSpaces resource cache (MLSPACES_CACHE_DIR, default ~/.cache/molmo-spaces-resources)."""
    return _env_path("MLSPACES_CACHE_DIR", Path("~/.cache/molmo-spaces-resources").expanduser())


def bench_roots() -> dict[str, Path]:
    """{"mini": dir, "hard": dir} of the benchmark episode JSONs used by the teleport harvests."""
    base = mlspaces_cache_dir() / "benchmarks" / BENCH_NAME / BENCH_VERSION / BENCH_SCENES
    return {"mini": _env_path("VLAGEOM_BENCH_MINI", base / BENCH_DIRS["mini"]),
            "hard": _env_path("VLAGEOM_BENCH_HARD", base / BENCH_DIRS["hard"])}
