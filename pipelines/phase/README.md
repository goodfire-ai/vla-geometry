# Figure 5: task phase

Part of [vla-geometry](../../README.md) · [all pipelines](../README.md).

Figure 5 is a composite: the robot sketch, the title "Phase" and the panel letters are drawn by
hand, and five rendered data panels are placed in it (`$VLAGEOM_FIGURES/<file>.{pdf,png}`):

| file | panel | what is drawn |
|---|---|---|
| `fig5_strip` | top strip | an example pick-and-place rollout (house 76, 15 model queries): the true phase of every query as a named ribbon ("ground truth"), two exo and two wrist frames per phase framed in the colour of the phase probes at that query, and the probe colour of every query along the time axis ("recovered") |
| `fig5a` | a | the successful natural queries (7,738) in the first two directions of a four-class LDA (LD1, LD2), coloured by true phase |
| `fig5b` | b | "Phase subspace": the first three principal components of the natural queries (6,000 points, coloured by true phase) with the example rollout as a path coloured by the phase probes; axis labels PC1 (17%) and PC2 (8%) |
| `fig5c` | c | % of landmarks decoded as their own phase, natural vs frozen, one line per phase (phase names in a one-row legend above the plot), chance at 25%; right: the most frequent decoded phase when frozen (the top two for reach and place, a fixed choice of the figure: `C_RUNNER_UP` in `figures/fig5.py`) |
| `fig5d` | d | change in the fraction of the 8 executed chunk steps that command a closed gripper, frozen minus natural, per phase (mean and 95% CI over landmarks; y axis -0.8 to 0.6) |

Every panel reads the action expert at block 24 (output of `action_expert.blocks.24` at the final
flow step, mean over the 16 action tokens). The task is MolmoSpaces pick-and-place ("Pick up the X
and place it in or on the Y", `FrankaPickAndPlaceDroidDataGenConfig`) in ProcTHOR houses 0-199.
Phase labels (`vlageom.phase.label_steps`): reach until 4 steps before the first commanded gripper
close; grasp until the object is lifted in the closed-gripper run that ends at the placing release;
transport until the held object's final descent begins; place from there to the end. A rollout
stops after the first step at which MolmoSpaces judges the task successful (at most 520 steps).

**The freeze test** (panels c and d). At landmark queries of a rollout, the policy is queried at
the same moment twice: with its normal input, the current frame and the frame 8 steps earlier of
both cameras ("natural"), and after two extra model queries on the held observation, when its
two-frame input holds the current frame twice, so the world appears to have stopped ("frozen").
Panel c decodes both activations with the four-class LDA of the natural rollouts (one fit per
fold, each landmark decoded by the fold that holds out its house); panel d compares the gripper
channel of the two commanded chunks. The landmarks of an episode are taken at query resolution
from its own labels: up to 3 reach queries, the middle open- and closed-hand grasp queries, the
first and last transport queries, the first and last closed-hand place queries and the first
open-hand place query (`vlageom.phase.pick_landmarks`, stage 40). For place, the analyses use the
last closed-hand place landmark, the last query before the release
(`vlageom.phase.used_landmarks`); every other landmark is used as harvested. Panels c and d select
landmarks by this rule: 797 landmarks of 122 episodes for panel c (place n = 80) and 819 of 126
episodes for panel d (place n = 82).

The stages are under `pipelines/phase/`, the library code in `vlageom/phase.py`, the render in
`figures/fig5.py`.

| stage | what it does | compute | output (under `$VLAGEOM_DATA`) |
|---|---|---|---|
| `05_prepare_assets.py [--scan]` | installs the pick-and-place receptacle assets and the scenes of the 181 listed houses into the MolmoSpaces cache with the real file locks (the harvests run with no-op locks, `VLAGEOM_NOOP_FILELOCK=1` from `scripts/harvest_env.sh`; see [Rendering (EGL)](../../scripts/README.md#rendering-egl)); `--scan` samples houses 0-199 and records which can host the task (181) | one process; network; `--scan` GPU + EGL, about 2 H200 GPU-hours with the installs | asset cache; `phase/house_validity.json` |
| `10_natural_rollouts.py --partition K` | natural rollouts of `lists/phase/natural_episodes.csv` (530 episodes, K = 0..135, up to 4 episodes per process, in list order): per step the gripper command, object position, receptacle support, success, contact, EE position; per query the 11 action-expert blocks (token mean) and VLM blocks 21-35 in four token pools, read from the query's own inputs | GPU + EGL, about 3 min per episode, about 27 H200 GPU-hours | `phase/natural/episodes/ep<EEEEE>_h<H>.npz` (530 files, 5.8 GB) |
| `20_natural_cache.py` | one row per model query (20,086 rows) with the action-expert activations and the step-resolution phase label of its step | CPU, 5-15 min (reads 5.8 GB) | `phase/natural/cache_ae.npz` (0.5 GB) |
| `30_decoder.py` | out-of-house balanced accuracy of a four-class LDA (StandardScaler + LDA) at each of the 11 layers, on the 7,738 successful queries over `lists/folds/phase.csv`; the best layer is 24 (0.920) | CPU, 3-10 min (16 BLAS threads) | `phase/decoder.json` |
| `40_freeze_harvest.py --task K` | the freeze test of one episode of `lists/phase/freeze_episodes.csv` (K = 0..127): rollout, landmarks from its own query labels, and per landmark the conditions entry (natural), pause1/2/4 (frozen = pause2) and, at reach and open-hand grasp landmarks, a 5 cm detour; every condition restores the landmark's state and moves the robot-mounted cameras to the restored pose before rendering; activations and the commanded chunk per condition | GPU + EGL, about 4 min per episode, about 9 H200 GPU-hours | `phase/freeze/episodes/ep<E>_h<H>.npz` (126 files, 1.9 GB) |
| `50_freeze_decode.py` | decoded phase of the landmarks of the 122 episodes with decode=1 (797 landmarks; of an episode's closed-hand place landmarks, the last), natural, frozen and detour | CPU, 1-3 min (1 BLAS thread) | `phase/freeze/decode_ae.json` |
| `60_trajectory_capture.py` | the example rollout of `lists/phase/trajectory_episode.csv` (after the two episodes of `lists/phase/trajectory_preceding.csv`, in the same process) with the exo and wrist images of every query, its activations and its phase labels | GPU + EGL, about 11 min | `phase/trajectory/traj_ep249_h76.npz` (12 MB) |
| `70_panels.py` | the arrays of the five panels and the numbers of the figure | CPU, 1-2 min (2 BLAS threads) | `panels/fig5_{strip,a,b,c,d}.npz`, `phase/fig5_numbers.json` |
| `figures/fig5.py` | renders the five panels (`--panel strip a b c d` selects) | CPU, about 10 s | `fig5_strip`, `fig5a` .. `fig5d` (`.pdf`, `.png`) |

```bash
python pipelines/phase/05_prepare_assets.py                      # once, after scripts/prefetch_assets.py
source scripts/harvest_env.sh                                     # GPU stages: EGL, offline model cache
python pipelines/phase/10_natural_rollouts.py --partition K       # GPU, K = 0..135
python pipelines/phase/20_natural_cache.py
python pipelines/phase/30_decoder.py
python pipelines/phase/40_freeze_harvest.py --task K              # GPU, K = 0..127
python pipelines/phase/50_freeze_decode.py
python pipelines/phase/60_trajectory_capture.py                   # GPU, one job
python pipelines/phase/70_panels.py
python figures/fig5.py
```

Stages 40 and 60 do not need the natural rollouts (their episodes are listed); stage 70 needs the
outputs of 20, 40, 50 and 60.

Notes on the stages:

- **Sampler state.** The scene of a rollout is set by its house, its seed and whether the
  MolmoSpaces task sampler that samples it is created for it (a new task sampler draws a number from
  the seeded numpy RNG and reseeds python `random`, numpy and torch with it before it samples; see
  [Figure 1, gripper tile](../gripper/README.md)). `new_sampler` in the lists records this for every
  paper rollout, and the stages sample each episode the same way (`vlageom.phase.sample_listed`), so
  a partition can be split or run again in any order. 213 of the 530 natural episodes have
  new_sampler=1. Every freeze-test episode is sampled by a task sampler created for it (one episode
  per process), and the freeze test uses only its own rollout (labels, landmarks, states); its scene
  equals that of the natural rollout of the same index when that rollout also had new_sampler=1. The
  example rollout uses house 76 and seed 20001076 with an existing task sampler; its labels are its
  own.
- **Natural rollouts.** Stage 10 keeps existing files (`--overwrite` replaces them) and returns
  status 1 when an episode failed; the other episodes of the partition are written. `--episodes`
  runs listed episodes in one process. Of the 543 catalog entries (181 valid houses x 3 seeds),
  the 13 that MolmoSpaces cannot sample with their seed (HouseInvalidForTask) are not in the list.
- **Layer.** Stage 30 scores all 11 layers; layer 24 (balanced accuracy 0.91985) is ahead of
  layer 28 (0.91932) by 0.0005. The panels read `vlageom.phase.LAYER = 24`.
- **Folds.** `lists/folds/phase.csv` maps the 167 houses with successful natural rollouts to 5
  folds (scikit-learn `GroupKFold(5)` over the 7,738 successful queries on an AVX-512 machine; fold
  hash `c8b301a92149` over the rows, `vlageom.phase.fold_hash`). Stages 30, 50 and 70 read it.
- **Freeze test.** The catalog is the first 128 houses of the successful natural episodes, one
  episode per house, houses with a place label first. Tasks 24 and 72 cannot be sampled
  (HouseInvalidForTask) and write no file, so the store has 126 episodes. Panel d uses all 126
  (819 landmarks); panel c uses the 122 episodes with decode=1 in
  `lists/phase/freeze_episodes.csv` (797 landmarks). Rollouts that end without success are kept
  (87 of the 126 succeed, 85 of the 122). Stage 40 records every landmark of an episode, including
  both closed-hand place landmarks; stages 50 and 70 select the rows with
  `vlageom.phase.used_landmarks`. Stage 40 exits with status 3 when a rendered frame is blank
  (mean pixel value below 3); run the task again.
- **Panel c.** Grasp pools the open- and closed-hand grasp landmarks (n-weighted); shares are
  rounded half up. The frozen reach landmarks are decoded as reach and as grasp equally often
  (180 of 362 each, 49.72%); ties keep the phase order, so the label reads "reach 50%, grasp 50%".
  Coincident markers at one end are moved one marker pitch toward the middle (the natural reach
  and natural transport markers), with a white ring where a moved marker sits on another phase's
  line. The phase names form a one-row legend above the plot, centred on the plot axes; the axes
  top is lowered to make room for it at the fixed page size (6.4 x 4.4 in before the tight crop).
- **Panels strip and b.** The four phase probes are one-vs-rest LDAs on the successful natural
  queries of every house but 76 (7,646 rows); a query's colour mixes the four phase colours with
  weights (normalised probe readout)^2.2. The strongest probe equals the true phase at 15 of 15
  queries. Panel b's principal components are fit on the same rows; its point cloud is a 350 dpi
  raster inside the PDF.

## Numbers (Figure 5)

`70_panels.py` prints and stores (`phase/fig5_numbers.json`):

| number | value |
|---|---|
| natural rollouts | 530 episodes, 181 houses, 334 successful (63%); 20,086 query rows, 7,738 of them from successful episodes in 167 houses |
| phase decoder (panel a) | balanced accuracy 0.91985 at layer 24 (majority rate 0.31) |
| panel b | PC1-3 explain 17%, 8%, 6% (32%) of the 7,646 rows' variance |
| freeze-test sets | decode: 797 landmarks of 122 episodes; gripper: 819 landmarks of 126 episodes; 8 executed steps per chunk |
| panel c, decoded correctly natural -> frozen (%) | reach 96.96 -> 49.72, grasp 88.40 -> 65.19, transport 85.63 -> 89.08, place 98.75 -> 72.50 (n 362 / 181 / 174 / 80) |
| panel c, most frequent when frozen | reach: reach 49.72, grasp 49.72; grasp: grasp 65.19; transport: transport 89.08; place: place 72.50, transport 23.75 |
| panel d, frozen - natural closed fraction | reach +0.428 +- 0.049 (n 374), grasp +0.234 +- 0.058 (n 185), transport -0.002 +- 0.003 (n 178), place -0.675 +- 0.102 (n 82) |

## Reproducibility (Figure 5)

- Stage 30 runs its LDA fits with 16 BLAS threads (`--threads`), the count of the paper's scan, and
  gives the paper's scan exactly at every layer (balanced accuracy, recalls and confusion
  matrices). At layers 0 and 4, one or two of the 7,738 out-of-fold predictions depend on the
  thread count: with 1, 2, 4 or 8 threads the balanced accuracy at layer 0 differs in the sixth
  decimal (0.807693 vs 0.807688); layers 8 to 35, which include layer 24 (every panel), do not
  change.
- Stages 50 and 70 run with 1 and 2 BLAS threads (`--threads`).
- For the top strip the reference is the paper's Figure 5 page: the 17 images in `fig5_strip.pdf`
  (and panel b's raster) are pixel-identical to images embedded on that page
  (`tests/compare_fig5_render.py --images-in`).
