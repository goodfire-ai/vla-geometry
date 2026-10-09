# Setup scripts

Part of [vla-geometry](../README.md).

| file | what it does |
|---|---|
| `prefetch_assets.py` | one-time download of the MolmoSpaces assets, the policy checkpoint, the tokenizer and the WordNet corpora (see [One-time downloads](#one-time-downloads-harvest-stages-only)) |
| `harvest_env.sh` | shell environment of the harvest stages: EGL, offline model cache, no-op file locks (see [Rendering (EGL)](#rendering-egl)) |
| `harvest_array.sbatch` | example SLURM job array: one GPU job per unit of a harvest stage, the unit taken from `SLURM_ARRAY_TASK_ID`; its header lists the array sizes of every GPU stage |

## Installation details

`pyproject.toml` pins the direct dependencies. `requirements-sim.lock` pins every package of the
harvest environment, including all indirect dependencies (226 packages, for Linux x86-64 and Python
3.11), to the versions of the paper's runs. MolmoSpaces and MolmoBot are installed from GitHub at
pinned commits. The `[analysis]` extra holds torch 2.7.1 (also part of `[sim]`), which the Figure 7
edit basis (`pipelines/gripper_fig7/10_basis.py`) and the Figure 6 b-d chart encodings
(`pipelines/wrist_steer/70_panels.py`, `80_numbers.py`) use on the CPU, and ripser, which adds one
optional measure to Figure 6a's metrics file. For these CPU stages, install the CPU build of torch
first, as in the [installation commands](../README.md#installation); otherwise pip installs the PyPI
build, which includes the CUDA libraries (several GB). Both builds give the same outputs for these
stages, bit for bit (`tests/check_expected.py --fig 6 7` passes with either). `pip install -e
".[test]"` adds PyMuPDF for the PDF text and image comparisons in `tests/`.

## One-time downloads (harvest stages only)

The harvest stages run offline (`scripts/harvest_env.sh` sets `HF_HUB_OFFLINE=1`) and many of them
run at once, so everything they read is downloaded once beforehand, in one process with network
access:

```bash
python scripts/prefetch_assets.py
python pipelines/phase/05_prepare_assets.py      # Figure 5 only: pick-and-place assets
```

`scripts/prefetch_assets.py` downloads:

- **MolmoSpaces assets** into `~/.cache/molmo-spaces-resources` (or `$MLSPACES_CACHE_DIR`): robot
  models, the benchmark episodes, THOR objects and DROID grasps, then the scene, its objects and
  its grasps for every house in `lists/` (the rollout houses 0-199 of the gripper, wrist, contact
  and wrist-steering lists, and the 947 benchmark houses of the object-teleport, near-range and
  crossed harvests), and the object-metadata database. The asset versions are set by the pinned
  MolmoSpaces commit. The object-teleport, near-range and crossed harvests start from the
  MolmoSpaces benchmark `molmospaces-bench-v1`, version `20260408`, `procthor-10k`
  (`FrankaPickHardBench` and `FrankaPickDroidMiniBench`), read from the asset cache (override with
  `VLAGEOM_BENCH_HARD` / `VLAGEOM_BENCH_MINI`).
- **NLTK WordNet corpora** (`wordnet`, `wordnet2022`), which MolmoSpaces reads when it is imported.
- **Policy checkpoint.** MolmoBot-DROID (`allenai/MolmoBot-DROID`, revision
  `cbe6ec358958d07ddfb20d3aa54e560e9e1b18c9`, about 20 GB) into `$VLAGEOM_CHECKPOINT` (default
  `~/checkpoints/MolmoBot-DROID`).
- **Tokenizer.** `Qwen/Qwen3-4B-Instruct-2507` at revision
  `cdbee75f17c01a7cc42f958dc650907174af0554` (the revision of the paper's runs) into the Hugging
  Face cache. The policy loads the tokenizer by name, which offline resolves to the revision in the
  cache's `refs/main`, so the script sets `refs/main` to this revision.

MolmoSpaces keeps a second directory next to the cache, its assets directory
(`$MLSPACES_ASSETS_DIR`, default `<site-packages>/assets` of the environment where MolmoSpaces is
installed). The resource manager writes a tree of links into the cache and an object database
(`.lmdb`) there, about 1.4 GB. It must be writable during the downloads, and the downloads, the
Figure 5 asset step and every harvest job must use the same directory: with the default, run them
all from the same `[sim]` environment; otherwise set `MLSPACES_ASSETS_DIR` (and
`MLSPACES_CACHE_DIR`) to the same values for all of them. `scripts/prefetch_assets.py` prints both
directories.

`--skip-assets`, `--skip-checkpoint` and `--skip-tokenizer` leave out a part; `--dry-run` lists
the scenes without installing them (it reads the benchmark episodes, so it needs the start-up
installs). The checkpoint, tokenizer and corpora can also be fetched by hand:

```bash
huggingface-cli download allenai/MolmoBot-DROID --revision cbe6ec358958d07ddfb20d3aa54e560e9e1b18c9 \
    --local-dir ~/checkpoints/MolmoBot-DROID
python scripts/prefetch_assets.py --skip-assets --skip-checkpoint     # tokenizer at its revision
python -c "import nltk; [nltk.download(c) for c in ('wordnet', 'wordnet2022')]"
```

`pipelines/phase/05_prepare_assets.py` installs, with the real file locks, the receptacles the
pick-and-place task sampler of Figure 5 can pick and the scenes of the 181 phase houses (see
[Figure 5](../pipelines/phase/README.md)).

`VLAGEOM_CHECKPOINT` may also be a Hugging Face repo id; that checkpoint is then downloaded into
the Hugging Face cache, which needs network access (`HF_HUB_OFFLINE=0`, so not with
`scripts/harvest_env.sh` sourced). A harvest stage stops with an error naming the missing file when
the checkpoint directory has no `model.pt` or `config.yaml`.

## Rendering (EGL)

MolmoSpaces renders its cameras with MuJoCo's EGL backend, so every harvest stage needs a GPU node
with EGL:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

The node needs `libEGL` with the NVIDIA EGL vendor library (on Debian/Ubuntu the `libegl1` and
`libglvnd0` packages, plus the NVIDIA driver). `source scripts/harvest_env.sh` sets the variables
above and, when the node has no system `libEGL.so.1`, downloads the Debian/Ubuntu EGL packages with
`apt-get` and unpacks them into a user directory (`$VLAGEOM_EGL_DIR`); this needs a Debian or Ubuntu
node with `apt-get`, `dpkg-deb` and `flock`. When the compute nodes cannot reach the package mirror,
unpack the packages once into a shared directory (with `<dir>/root/`) and set `VLAGEOM_EGL_DIR` to it.
On other systems install libEGL (GLVND) system-wide.
Analysis, panel and figure stages need neither a GPU nor EGL.

`scripts/harvest_env.sh` also sets `VLAGEOM_NOOP_FILELOCK=1`. Importing `vlageom.sim` then replaces
the `filelock` locks with no-op locks in the whole process (MolmoSpaces, its resource manager and
`huggingface_hub` included), because with many processes on a network filesystem the real locks
can fail or block each other. This is safe only when nothing is written to the caches any more,
that is after `scripts/prefetch_assets.py` (and, for Figure 5, `pipelines/phase/05_prepare_assets.py`)
has completed. Without `VLAGEOM_NOOP_FILELOCK=1` the real locks are kept.
