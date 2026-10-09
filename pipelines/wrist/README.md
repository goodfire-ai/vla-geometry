# Figure 1: wrist tile

Part of [vla-geometry](../../README.md) · [all pipelines](../README.md). The sweep of row 0 (house
0) is also panel a of Figure 6 ([`wrist_steer`](../wrist_steer/README.md)).

The wrist tile shows action-expert activations (layer 9, action token 0, final flow step) while the
wrist roll joint is turned through its full range, in 64 houses, in one principal-component basis.

| stage | what it does | compute | output (under `$VLAGEOM_DATA`) |
|---|---|---|---|
| `01_sweep.py --row N` | episode N of `lists/wrist_episodes.csv` (N = 0..63): roll out the policy (at most 600 steps), take the first query after query 0 at which the hand is open, set the wrist roll joint (arm joint 6, FR3 joint 7) to 38 evenly spaced angles over its simulator limits (±3.0159 rad) plus the episode's own angle, and for each of the 39 angles render both cameras at the swept pose and read the action expert's per-token output at layers 0, 4, 9, 16, 24, 32 | GPU + EGL, 2-6 min per episode, about 3.5 H200 GPU-hours | `wrist/sweeps/ep<catalog_index>.npz` (114 MB each, 7.3 GB) |
| `02_check_sweeps.py` | checks every sweep file: 39 rows, joint 6, range ±3.0159 rad, EE position unchanged, wrist-camera image turns with the hand, exo images not dark, episode and house as listed; reports missing episodes | CPU, about 3 min | report, exit status |
| `03_panel.py` | token 0 of layer 9 from every sweep, each house minus its own sweep mean, one SVD over the 2,496 points; house 0's sweep is the drawn trace | CPU, about 3 min | `panels/wrist.npz` |

```bash
source scripts/harvest_env.sh                               # GPU stages: EGL, offline model cache
python pipelines/wrist/01_sweep.py --row N                  # GPU, N = 0..63
python pipelines/wrist/02_check_sweeps.py
python pipelines/wrist/03_panel.py
```

`figures/fig1.py` draws Figure 1 from the four tile panels (see
[`figures/README.md`](../../figures/README.md#figure-1)).

`01_sweep.py` keeps an existing output (pass `--overwrite` to replace it), so running all rows again
fills in only missing episodes. It exits with status 3 and writes nothing when the exo-camera
frames are dark (mean pixel value below 20), which happens on nodes whose EGL rendering fails;
run that episode again on another node.

**Frame history.** Every swept pose starts from the rollout's snapshot at the swept query, which
includes the policy's frame history, and the policy is never reset. The arm is then set to the
swept joint angles, the robot-mounted cameras are moved to the hand, and the cameras are rendered.
The model input is therefore the current frame pair, which shows the swept pose, plus the
rollout's own frame pair from 8 steps earlier (4 image blocks). The swept query is query 1 or
later, so the earlier pair always exists, and within one sweep it is the same at every angle; the
angles of one sweep differ only in the current frames and the joint state. The camera images are
stored with every row, which makes the files large; `02_check_sweeps.py` and `03_panel.py` read
all 64 files.

The figure note "346° sweep · open arc" is computed from the panel by
`vlageom.wrist.sweep_note(panel["roll"], panel["Z3"])`: the range of the trace's joint angles, and
"open arc" when the distance between the two ends of the trace in PC1-3 is more than 10% of its
length ("closed loop" otherwise; 66% for the paper's trace). The axis labels use the panel's
explained-variance ratios (PC1 77%, PC2 16%).
