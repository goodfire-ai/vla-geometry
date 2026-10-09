# Figure 6: wrist orientation

Part of [vla-geometry](../../README.md) · [all pipelines](../README.md).

Figure 6 is a composite: a robot sketch with its coloured ring, the title and the panel letters
are drawn by hand, and two rendered data files are placed in it.

| file | panels | what is drawn |
|---|---|---|
| `fig6_a.pdf` | a | the wrist roll sweep of house 0 on its own PC1-2 (action-expert layer 9, action token 0), coloured by the joint angle: an open arc whose ends lie far apart although they are 14.4 deg apart physically |
| `fig6_bcd.pdf` (and `.png`) | b, c, d (one block) | b: the natural rollouts' activations on the autoencoder's circle for rotation about the vertical axis, coloured by the hand's yaw; c: the rotation of the decoded action chunk after one edit of +-20 or +-40 deg (dots: 7 held-out scenes, ticks: means); d: the executed wrist rotation over 32 steps with the edit held at every query, in house 10 |

The stages are under `pipelines/wrist_steer/`, the library code in `vlageom/wrist_steer.py`, the
render in `figures/fig6.py`. Panel a reuses the wrist tile's sweep of house 0 (Figure 1,
`pipelines/wrist/01_sweep.py --row 0`; see [Figure 1, wrist tile](../wrist/README.md), for its frame
history). The circular autoencoder ("chart") and the steering runs of panels b-d are Mathilde
Papillon's method and code (see [Credits](../../README.md#credits)).

| stage | what it does | compute | output (under `$VLAGEOM_DATA`) |
|---|---|---|---|
| `10_natural_rollouts.py --partition K` | natural MolmoBot-DROID rollouts of `lists/wrist_steer/natural_episodes.csv`, one process per partition of 16 episodes (K = 0..12, 200 episodes in all): seeded with the listed seed, sampled for the listed house (after the attempts of the listed run that could not be sampled), the policy's own flow sampling, until success or 500 steps. Per model query: the output of action-expert blocks 0, 4, 9, 16, 24, 32 at the final flow step (bfloat16 per token; token 0 and the mean of the 16 action tokens), predicted chunk, EE pose and Jacobian, object position, joint velocities; per step the EE pose | GPU + EGL, 15-30 min per partition, about 5 H200 GPU-hours | `wrist_steer/natural/episodes/episode_<index>.npz` |
| `20_natural_cloud.py` | every query of the episodes with at least 5 queries (6,555 queries, 199 episodes, 191 houses), in the list's row order; the token-mean layer-9 activations with the EE rotation are the natural cloud | CPU, seconds | `wrist_steer/natural/natural_cache.npz`, `natural.npz` |
| `30_orient_sweep.py --task K` | `lists/wrist_steer/orient_episodes.csv` (K = 0..199): roll out (at most 600 steps), at 4 open-hand queries turn the hand about the base x, y and z axes, both ways, in closed-loop 10-deg steps (at most 18 each way) with the EE position held within 3 cm; every pose rendered with the cameras moved to the hand and read from the action expert (token mean of 11 layers, final flow step, the query's flow noise) | GPU + EGL, about 6 min per episode, about 17 H200 GPU-hours | `wrist_steer/orient/ep<E>_h<H>.npz` (197 files, 25,748 rows, 102 houses; 833 MB) |
| `40_operator.py` | held-out houses (numpy `default_rng(0)` shuffle of the 102 sweep houses, first third: 34) and a linear rotation operator in a 64-d PCA of the training rows, used by the control edits of stage 60 | CPU, about 3 min with 24 BLAS threads | `wrist_steer/operator_L9.npz` |
| `50_train_chart.py` | the chart: encoder 768-512-256-32 with code pairs (0,1), (2,3), (4,5) projected onto circles of learned radius (rotation about base x, y, z), decoder 32-256-512-768; trained on the layer-9 token-mean activations of the training houses' sweeps with reconstruction, code-rotation and decoded-rotation losses over pairs of one sweep (8,000 Adam steps, seed 0) | GPU, about 2 min | `wrist_steer/chart.pt`, `chart_report.json` |
| `60_steer.py --mode oneshot` | 8 held-out scenes (`lists/wrist_steer/steer_episodes.csv`, one process): roll out (at most 140 steps), at the first stable-grasp query (else the middle query) decode the chunk without an edit, with chart edits of +-20 and +-40 deg about z, x and y, and with two control edits; measure the commanded EE rotation | GPU + EGL, about 9 min | `wrist_steer/steer/oneshot.json` |
| `60_steer.py --mode persist` | the first 6 of these scenes: 32 steps from the same query without an edit and with the z edit (+-20, +-40 deg) recomputed from the live activation at every query; the hand's rotation about base z after every step | GPU + EGL, about 9 min | `wrist_steer/steer/persist.json` |
| `70_panels.py` | panel a: house 0's roll sweep sorted by joint angle, token 0 of layer 9 minus the sweep mean, SVD; panels b-d: the natural cloud encoded with the chart, the one-shot z edits (scene 6 left out), house 10's held edits | CPU, about 10 s (panels b-d need torch; the CPU build is enough) | `panels/fig6_a.npz`, `fig6_a_metrics.json`, `fig6_bcd.npz` |
| `80_numbers.py` | the numbers of the figure and its text (below) | CPU, about 1 min (needs torch) | `wrist_steer/fig6_numbers.json` |
| `figures/fig6.py` | renders the two data files (`--panel a\|bcd\|all`) | CPU, about 5 s | `$VLAGEOM_FIGURES/fig6_a.pdf`, `fig6_bcd.pdf`, `fig6_bcd.png` |

```bash
source scripts/harvest_env.sh                                # GPU stages: EGL, offline model cache
python pipelines/wrist/01_sweep.py --row 0                   # GPU, panel a's sweep (house 0; also a Figure 1 stage)
python pipelines/wrist_steer/10_natural_rollouts.py --partition K   # GPU, K = 0..12
python pipelines/wrist_steer/20_natural_cloud.py
python pipelines/wrist_steer/30_orient_sweep.py --task K     # GPU, K = 0..199
python pipelines/wrist_steer/40_operator.py
python pipelines/wrist_steer/50_train_chart.py               # GPU
python pipelines/wrist_steer/60_steer.py --mode oneshot      # GPU
python pipelines/wrist_steer/60_steer.py --mode persist      # GPU
python pipelines/wrist_steer/70_panels.py
python pipelines/wrist_steer/80_numbers.py
python figures/fig6.py
```

Notes on the stages:

- **Natural rollouts.** Each partition is one process that runs its episodes in index order,
  because the MolmoSpaces task sampler of a process is created by its first sampling and later
  episodes are sampled by that sampler. Episode i was meant for house i; the seed counts the
  sampling attempts of the partition from partition x 16. For eight episodes (houses 2, 30, 80,
  104, 106, 138, 150, 192) MolmoSpaces found no valid task in the first attempt, and the episode
  went to house i + 3 with the next seed. The list records the house and seed of every rollout;
  stage 10 repeats the failed attempts (seed and house) before the listed one, since they act on
  the task sampler (in partitions 5 and 12 the first attempt of the process creates it). Stage 10
  keeps existing episode files and rolls a partition out from its first episode up to the last
  missing one. It exits with status 3 when a camera frame the policy reads at a query is blank;
  run the partition again.
- **Orientation sweeps (frame history).** Every pose of a sweep starts from the rollout's snapshot
  at the swept query, which includes the policy's frame history, and the policy is never reset;
  the arm is then set, the robot-mounted cameras are moved to the hand, and the cameras are
  rendered. As in the roll sweeps of Figure 1, the model input is the current frame pair at the
  rotated pose plus the rollout's own frame pair from 8 steps earlier (4 image blocks); the swept
  queries are query 1 or later, and within one pose's sweep the earlier pair is the same at every
  step. For three listed episodes MolmoSpaces finds no valid task (tasks 57, 191, 194; houses 30,
  104, 106); they write no file. Stage 30 exits with status 3 and writes nothing when a rendered
  frame is blank.
- **Held-out split.** Stages 40 and 50 and the calibration of stage 80 hold out the same 34 houses
  (`lists/wrist_steer/heldout_houses.csv`); stage 40 checks the split against the list and the
  steering stage checks that its scenes are held out.
- **Steering.** All scenes of a mode run in one process, in list order (the first is sampled by a
  new task sampler, the others by that sampler). The flow noise of query q is seeded with
  `seed + 1000 q` in the rollout and in every edited branch. The one-shot json also holds per scene
  the landmark query and the norm of its unedited activation. The edit size of stage 80 is the edit
  norm over the activation norm rounded to one decimal (`landmarks` of the one-shot json, or the
  landmark lines of a one-shot run's log with `--oneshot-log`).
- **Panel c** leaves out the scenes whose chunk at the landmark already turns the wrist by more
  than 10 deg without an edit (`vlageom.wrist_steer.drift_episodes`). On the paper's data this is
  scene 6 (house 6), at 22.9 deg, against at most 2.6 deg in the other scenes
  (`noop_chunk_rot_deg` and `left_out` in the numbers file; stages 70 and 80 print it).
- **Panel b's phase.** The circle has no preferred origin, so the encoded phase is shifted by the
  circular mean of (phase - yaw) before it is drawn. Stage 70 logs a warning when the encoded phase
  runs opposite to the yaw (the colours would then run clockwise round the circle).
- **Panel a's signs.** PC1 is signed to increase with the joint angle, and PC2 so that the middle of
  the sweep lies on its positive side relative to the two ends (`vlageom.wrist_steer.panel_a`).

## Numbers (Figure 6)

`80_numbers.py` prints and stores:

| number | value | definition |
|---|---|---|
| compensation (text, panel c) | 0.3291 +- 0.0831 ("0.33 +- 0.08"), 28 of 28 opposing | -response / written angle over the 28 panel-c edits, mean +- s.d. (ddof 0) |
| panel c means | +12.79, +6.78, -7.08, -12.14 deg for -40, -20, +20, +40 | mean response per written angle |
| held edit (text, panel d) | 1.214, 1.349, 1.173, 1.033 for -40, -20, +20, +40 ("1.0-1.3x"), 24 of 24 opposing | per angle, mean over 6 scenes of -(rotation after 32 steps minus the unedited rollout's) / written angle |
| panel d, house 10 | +60.80, +30.51, -30.48, -56.32 deg; unedited -0.29 | rotation after 32 steps |
| edit size | 18.90% +- 7.27% ("19% +- 7%") | edit norm / unedited activation norm over the 32 one-shot z edits |
| calibration | median slope 0.9992 (IQR 0.985-1.009) over 171 held-out z sweeps | degrees on the z circle per degree of rotation about base z, sweeps spanning more than 120 deg |
| split | 34 of 102 houses held out; 197 sweeps | |
| chart | held-out reconstruction R^2 0.946; decode-transport R^2 z/x/y 0.878/0.861/0.795; radii 0.111/0.092/0.098 | `chart_report.json` |

Panel a's measures (`fig6_a_metrics.json`): PC1 79% / PC2 18% of the variance, joint angles
-172.8 to +172.8 deg (coverage 345.6 deg, so the ends are 14.4 deg apart). The largest H1
persistence is computed only when `ripser` is installed; no figure or number uses it.

## Reproducibility (Figure 6)

- Stage 40 fits the operator with 24 BLAS threads, the count of the paper's run; with another
  thread count V and Lhat differ in their last digits (about 3e-14 relative) and the held-out
  houses do not. Only the control rows (`random_gen`) of the one-shot run use the operator; no
  panel or number uses them.
- The chart is trained on the GPU (CUDA, torch 2.7.1 on an H200 for the paper); training on another
  device or library version gives a different chart, which moves panels b-d slightly.
