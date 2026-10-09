# Pipelines

Part of [vla-geometry](../README.md).

Each folder is one pipeline. Its README describes its figure: what each panel draws, the data, every
stage with its compute and outputs, the commands, notes on the stages and the numbers they print.

| pipeline | figure | README |
|---|---|---|
| `gripper` | 1, gripper tile (its edits are also the input of Figure 7) | [gripper/README.md](gripper/README.md) |
| `wrist` | 1, wrist tile (the sweep of row 0 is also Figure 6a) | [wrist/README.md](wrist/README.md) |
| `target` | 1, target tile (its store is also read by the distance tile and Figure 3) | [target/README.md](target/README.md) |
| `distance` | 1, distance tile; Figure 3 | [distance/README.md](distance/README.md) |
| `position` | 2 | [position/README.md](position/README.md) |
| `contact` | 4 | [contact/README.md](contact/README.md) |
| `phase` | 5 | [phase/README.md](phase/README.md) |
| `wrist_steer` | 6 | [wrist_steer/README.md](wrist_steer/README.md) |
| `gripper_fig7` | 7 | [gripper_fig7/README.md](gripper_fig7/README.md) |

The render scripts are in [`figures/`](../figures/README.md); the one-time downloads and the
shell environment of the harvest stages are in [`scripts/`](../scripts/README.md).

## Running the stages

Each pipeline is a sequence of numbered stages under `pipelines/<name>/`. Every stage reads from
and writes to `$VLAGEOM_DATA/<name>/`; the panel stages write `$VLAGEOM_DATA/panels/*.npz`, the
arrays each panel is drawn from, and the scripts under `figures/` render them. Every stage has
`--help`, and every input and output location can also be given on the command line.

Order: run `scripts/prefetch_assets.py` once (and `pipelines/phase/05_prepare_assets.py` before
Figure 5), then Figure 1. Figure 3 reads the two packed stores of Figure 1, Figure 6a one wrist
sweep of Figure 1, and Figure 7 the gripper edits of Figure 1 with the model inputs saved by its
harvest (`--save-captures`). Figures 2, 4 and 5 are independent of the others.

GPU stages are marked in each pipeline's README; run them as one job per unit (shard, task,
partition or row), e.g. with `scripts/harvest_array.sbatch`, which takes the unit from
`SLURM_ARRAY_TASK_ID` (its header lists the array sizes of every GPU stage). Source
`scripts/harvest_env.sh` first. Every GPU stage skips units whose output exists, so a failed job can
simply be run again; exit status 3 means the EGL renderer of that process returned blank frames, and
the unit should be run again (on another node if it repeats).

**Model inputs.** MolmoBot-DROID reads two frames per camera (exo and wrist; `n_obs_steps` 2,
`obs_step_delta` 8): the current frame and the frame 8 simulator steps earlier, the step of the
previous model query. A query right after a policy reset has no earlier frame, so its input holds 2
image blocks; later queries hold 4. Which frames a harvested state shows is stated per harvest in
the pipelines' READMEs: the object-teleport, near-range and crossed harvests reset the policy before
the first query of every placement, and the wrist sweeps and the off-schedule contact placements
restore a mid-episode snapshot without a reset.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `VLAGEOM_DATA` | `<repo>/data` | root of all harvested data, intermediate arrays (`<root>/<pipeline>/`) and panel files (`<root>/panels/`) |
| `VLAGEOM_CHECKPOINT` | `~/checkpoints/MolmoBot-DROID` | policy checkpoint directory (or Hugging Face repo id) |
| `VLAGEOM_FONT_DIR` | unset | directory with the figure font files |
| `VLAGEOM_FIGURES` | `<data root>/figures` | rendered figures |
| `VLAGEOM_BENCH_HARD`, `VLAGEOM_BENCH_MINI` | MolmoSpaces asset cache | benchmark episode directories |
| `MLSPACES_CACHE_DIR` | `~/.cache/molmo-spaces-resources` | MolmoSpaces asset cache |
| `MLSPACES_ASSETS_DIR` | `<site-packages>/assets` of the `[sim]` environment | MolmoSpaces assets directory (links into the cache, object database); the same, writable directory for the downloads and every harvest job (see [One-time downloads](../scripts/README.md#one-time-downloads-harvest-stages-only)) |
| `VLAGEOM_NOOP_FILELOCK` | unset (real locks); `1` in `scripts/harvest_env.sh` | `1`: no-op file locks in harvest processes (see [Rendering (EGL)](../scripts/README.md#rendering-egl)) |
| `VLAGEOM_EGL_DIR` | `/tmp/libegl-$USER` | where `scripts/harvest_env.sh` unpacks the EGL libraries when the node has none |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `1` in `scripts/harvest_env.sh` | Hugging Face offline mode of the harvest jobs (set `0` before sourcing to allow downloads) |
| `MUJOCO_GL`, `PYOPENGL_PLATFORM`, `JAX_PLATFORMS` | `egl`, `egl`, `cpu` in `scripts/harvest_env.sh` | renderer backend; JAX kept on the CPU |
| `PYTHON` | `python` | interpreter used by `scripts/harvest_array.sbatch` and the canary scripts (the environment with the `[sim]` extra) |

## Figure 1

Figure 1 is a row of four tiles:

| tile | variable | what is drawn |
|---|---|---|
| gripper | gripper open / close | action-expert activations near the grasp in the top-3 principal directions of the gripper edits, with the edit directions |
| wrist | wrist angle | action-expert activations while one wrist joint is swept through its range, pooled over 64 houses |
| target | target object position | VLM activations projected onto a 3-D ridge read-out of the object position, with one object path |
| distance | distance to the target object | held-out read-outs of two distance probes (metre range and centimetre range) against the true distance |

Each tile is a pipeline; its last stage writes `$VLAGEOM_DATA/panels/<tile>.npz`, and
`figures/fig1.py` draws the figure from the four panel files. Tiles are independent except that
the distance tile also reads the target tile's object-teleport store.

1. gripper: rollouts of the 304 listed episodes with edit fits at action-expert layer 32 -> panel
2. wrist: wrist-joint sweeps in 64 houses -> check -> pooled principal components -> panel
3. target: object-teleport harvest over 906 houses -> packed store; object path in one house ->
   probe score -> read-out basis and panel
4. distance: near-range harvest over 300 houses -> packed store -> distance probes on the combined
   rows of this store and step 3's packed store -> panel
5. render: `figures/fig1.py`

```bash
python scripts/prefetch_assets.py                           # once, with network access (see scripts/README.md)
source scripts/harvest_env.sh                               # before the GPU stages: EGL, offline model cache

python pipelines/gripper/10_harvest_edits.py --task K --save-captures   # GPU, K = 0..151
python pipelines/gripper/20_panel.py

python pipelines/wrist/01_sweep.py --row N                  # GPU, N = 0..63
python pipelines/wrist/02_check_sweeps.py
python pipelines/wrist/03_panel.py

python pipelines/target/10_teleport_harvest.py --shard K    # GPU, K = 0..113
python pipelines/target/30_square_path_harvest.py           # GPU, one job
python pipelines/target/20_pack_store.py
python pipelines/target/40_probe.py
python pipelines/target/50_panel.py

python pipelines/distance/10_near_harvest.py --shard K      # GPU, K = 0..74
python pipelines/distance/20_pack_near.py
python pipelines/distance/30_ruler_probes.py
python pipelines/distance/40_panel.py

python figures/fig1.py
```

(`--save-captures` is needed only for Figure 7.) Where the outputs go (under `$VLAGEOM_DATA`):

```
gripper/deltas/                     edit files                        gripper stage 10
gripper/captures/                   model inputs (--save-captures)    gripper stage 10
wrist/sweeps/ep<catalog_index>.npz  sweep files                       wrist stage 01
target/teleport/houses/, frames/    object-teleport house files       target stage 10
target/teleport/prep/               packed store (memory-mappable)    target stage 20
target/shape_square/h44/            object path of house 44           target stage 30
target/probe/last_tok_L21.json      probe score                       target stage 40
distance/near/houses/, frames/      near-range house files            distance stage 10
distance/near/prep/                 packed store                      distance stage 20
distance/ruler_probes.npz, .json    out-of-fold read-outs and scores  distance stage 30
panels/<tile>.npz                   arrays each tile is drawn from    last stage of each tile
figures/fig1.png, fig1.pdf          the figure ($VLAGEOM_FIGURES)     figures/fig1.py
```

Each tile's stages are described in the README of its pipeline: [gripper](gripper/README.md),
[wrist](wrist/README.md), [target](target/README.md), [distance](distance/README.md). The render
is described in [`figures/README.md`](../figures/README.md#figure-1).

## Compute and disk

GPU figures are H200 GPU-hours with one GPU per process; CPU stages run on one node.

| figure | GPU stages | GPU-hours | disk | CPU stages |
|---|---|---|---|---|
| 1 gripper | rollouts of the 304 listed episodes with edit fits at layer 32 | about 57 | 30 MB of edit files; 33 GB of model inputs with `--save-captures` (Figure 7) | seconds |
| 1 wrist | 64 joint sweeps (39 poses each) | about 3.5 | 7.3 GB | check and panel about 3 min each |
| 1 target | object-teleport harvest, 906 houses x 16 placements; object path, 64 placements | about 34 + 0.1 | 43 GB of house files, 50 GB packed store | packing about 2 min (16 processes); probe 1-10 min and panel 0.5-5 min (24 BLAS threads) |
| 1 distance | near-range harvest, 300 houses x 8 placements | about 12 | 14 GB of house files, 1.4 GB packed store | packing about 3 min; probes about 3 min and 4 GB of memory |
| 2 | crossed harvest, 947 houses; hand lines and object lines, 48 houses; trace-grid paths, house 44 | about 45 + 1.4 + 1.6 + 0.4 | 253 GB of house files, 33 GB packed (252 GB with every array); 8.7 + 2.5 + 1.1 GB | probes about 15-35 min; Procrustes about 4 hours on 8 cores; the rest about 20 min |
| 3 | none (the stores of Figure 1) | 0 | about 15 MB | erasure about 15 min on 24 cores; probe bank about 8 min on 16 cores (9 GB of memory); the rest about 5 min |
| 4 | natural harvest, 299 episodes; off-schedule harvest, 128 episodes | about 13 + 10.5 | 369 + 352 GB of harvest files; 5.8 GB natural cache, 369 MB off-schedule pool | cache 20-35 min (reads 369 GB); pool about 1 min; probes 4-7 min; held-out scores 8-60 min; panel about 1 min |
| 5 | house scan; natural rollouts, 530 episodes; freeze test, 128 episodes; example rollout | about 2 + 27 + 9 + 0.1 | 5.8 + 1.9 GB of harvest files; 0.5 GB table | 10-30 min in total |
| 6 | natural rollouts, 200 episodes; orientation sweeps, 200 episodes; chart; steering, both modes | about 5 + 17 + 0.1 + 0.3 | about 1.1 GB | about 5 min in total |
| 7 | low-rank refits, 304 episodes (from the stored model inputs of Figure 1) | about 210 | 33 MB of records | about 2 min in total |
| renders | none | 0 | small | about 10 s per figure |

In total about 450 H200 GPU-hours and 1.2 TB of disk, plus the 20 GB checkpoint and the
MolmoSpaces asset cache.

## Reproducibility

- Analysis, panel and figure stages are deterministic. Run on the paper's stores with the pinned
  library versions on an x86-64 CPU with AVX-512, they give the arrays the paper's panels were drawn
  from bit for bit (the exception is stated in the [Reproducibility notes of Figure
  2](position/README.md#reproducibility-figure-2)), and with Product Sans the renders have the bytes
  of the paper's PNG files and of its PDF files apart from their creation date. Where the last bits
  of a result depend on the number of BLAS threads, the stage sets that number in code
  (threadpoolctl) to the number the paper's arrays were computed with:

  | stage | BLAS threads |
  |---|---|
  | gripper `20_panel.py`, wrist `03_panel.py`, position `30`/`40`/`60`/`90`, gripper_fig7 `10_basis.py` (torch) and `20_probe.py` | 8 |
  | target `40_probe.py`, `50_panel.py`; distance `50_erasure.py`; wrist_steer `40_operator.py` | 24 |
  | distance `60_probe_bank.py`; phase `30_decoder.py` | 16 |
  | distance `80_plant_cosines.py`; contact `70_panel.py` | 4 |
  | phase `70_panels.py` | 2 |
  | distance `30_ruler_probes.py`; contact `50_natural_probe.py`, `60_heldout_scores.py`; phase `50_freeze_decode.py` | 1 |

  scipy loads a BLAS library of its own when it is first imported; the stages whose results depend
  on it (the LDA fits of Figures 4 and 5) import scipy before they set the limit, so that the limit
  covers both libraries. Other CPUs or thread counts change results at the level of floating-point
  rounding (the printed numbers do not change). Fold maps are shipped (see
  [`lists/`](../lists/README.md)), because `GroupKFold` breaks ties between equally sized houses
  differently on different CPUs.
- Harvests run the policy in bfloat16 on the GPU and render with EGL. Re-running a harvest on a
  different GPU type or driver can change activations slightly. On the same GPU type, driver and
  library versions, the GPU canaries ([`tests/README.md`](../tests/README.md)) compare one unit of
  every harvest with the paper's stores, within the tolerances they state. The shipped lists keep
  the set of episodes and houses the same.
- Rendering after a state change. Whenever a stage sets the simulator state without stepping it
  (restoring a snapshot, setting joints, moving the object), the robot-mounted cameras are moved
  to the robot's current pose before the cameras are rendered (`vlageom.sim.fresh_obs`).
