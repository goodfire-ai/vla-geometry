# Figure 1: gripper tile

Part of [vla-geometry](../../README.md) · [all pipelines](../README.md). The edits are also the
input of Figure 7 ([`gripper_fig7`](../gripper_fig7/README.md)).

The gripper tile shows action-expert activations near the grasp, and the gripper edits fitted at
those states, on the top three principal directions of the edits. An edit is a vector added to the
output of action-expert block 32 on the last flow step, fitted so that the decoded gripper command
flips (open to close before the grasp, close to open after it) while the decoded arm actions stay at
the current joint angles. The tile uses the 1,666 edits of the 304 episodes in
`lists/gripper_episodes.csv` (170 houses).

| stage | what it does | compute | output (under `$VLAGEOM_DATA`) |
|---|---|---|---|
| `10_harvest_edits.py` | for each listed episode: MolmoBot-DROID rollout (up to 600 steps), stable-grasp onset, up to 3 open queries before and 3 closed-and-holding queries after it (typically 3-9 queries from the onset), one edit fit per query (Adam, 1000 iterations) | GPU + EGL, about 57 H200 GPU-hours (152 tasks of 2 episodes: rollouts about 10, edit fits about 42, model loading about 5) | `gripper/deltas/ep<E>_h<H>_<landmark>_q<Q>_aw300_sw3000.npz` (1,666 files, 30 MB) |
| `20_panel.py` | centered SVD of the edits of the listed episodes; activations and edits projected on edit PC1-3 | CPU, seconds | `panels/gripper.npz` |

```bash
source scripts/harvest_env.sh                                           # GPU stages: EGL, offline model cache
python pipelines/gripper/10_harvest_edits.py --task K --save-captures   # GPU, K = 0..151
python pipelines/gripper/20_panel.py
```

`figures/fig1.py` draws Figure 1 from the four tile panels (see
[`figures/README.md`](../../figures/README.md#figure-1)).

The scene of an episode (pickup object, robot placement) is set by its row of
`lists/gripper_episodes.csv`: `eng.sample` seeds python `random`, numpy and torch with `seed`, and
`new_sampler` says which MolmoSpaces task sampler samples it. With `new_sampler=1` the environment
and its task sampler are created for the episode; a new task sampler draws a number from the seeded
numpy generator and reseeds all three generators with it before it samples. With `new_sampler=0` an
existing task sampler samples the episode from the state set by `seed`, with the house loaded from
file (`vlageom.gripper.sample_episode`). Episodes can therefore be split into tasks in any way.
Stage 10 skips episodes that already have edit files, so a failed task can simply be run again. An
episode without a stable grasp writes no files; stage 20 uses the edit files that exist and prints
the listed episodes that have none.

`--save-captures` also stores the model inputs of every model query of each grasping episode
(`gripper/captures/ep<E>_h<H>.pt`, about 110 MB per episode, 33 GB in total); Figure 7d's stage
reads them. With `--save-captures` an episode is skipped only when this file exists as well. With
`--from-captures <dir>`, stage 10 fits the edits from these files instead of running the rollout,
which repeats the fits without per-step rendering.

Stage 20 prints the edit counts and the explained-variance ratio of PC1-3: 1,666 edits (863 close,
803 open), 304 episodes, 170 houses, 0.703 / 0.056 / 0.020 ("δ PC1 (70% var)", "δ PC2 (6%)").
The signs of PC1-3 are set by rules in `vlageom.gripper.pc_signs`.
