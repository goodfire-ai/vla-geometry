# Shipped lists

Part of [vla-geometry](../README.md).

`lists/` holds the episodes, houses and folds the paper used, so every stage runs on exactly the
paper's units:

| file | rows | content |
|---|---|---|
| `gripper_episodes.csv` | 304 | episode_index, house, seed, new_sampler of the gripper edits (Figures 1 and 7) |
| `wrist_episodes.csv` | 64 | catalog_index, episode_index, house, seed of the wrist sweeps, one per house (Figure 1; row 0 is Figure 6a) |
| `teleport_houses.csv` | 906 | house, bench, bench_idx of the object-teleport harvest (Figures 1 and 3) |
| `near_houses.csv` | 300 | house, bench, bench_idx of the near-range harvest (Figures 1 and 3) |
| `position/crossed_houses.csv` | 947 | house, bench, bench_idx of the crossed arm x object harvest (Figure 2): the benchmark catalog (`vlageom.sim.build_house_list(950)`) |
| `position/line_houses.csv` | 48 | house, bench, bench_idx of the straight hand and object lines (Figure 2e) |
| `position/ee_color_reference.csv` | 4,000 | x, y, z (m, robot base frame): hand positions at the model queries of natural rollouts, the colour reference of the EE trace grid (Figure 2) |
| `contact/natural_episodes.csv` | 299 | episode_index, house, seed, orig_n_steps, orig_contact, task_description of the natural contact rollouts (Figure 4) |
| `contact/offschedule_episodes.csv` | 128 | task, episode_index, house, seed, outcome of the off-schedule contact harvest (Figure 4b) |
| `phase/natural_episodes.csv` | 530 | episode_index, partition, house, seed, new_sampler of the natural pick-and-place rollouts (Figure 5) |
| `phase/freeze_episodes.csv` | 128 | task, episode_index, house, seed, decode of the freeze test (Figure 5c-d) |
| `phase/trajectory_episode.csv` | 1 | the example rollout of Figure 5 (top strip, panel b) |
| `phase/trajectory_preceding.csv` | 2 | the episodes rolled out before the example rollout in the same process |
| `wrist_steer/natural_episodes.csv` | 200 | episode_index, partition, house, seed of the natural rollouts of Figure 6b |
| `wrist_steer/orient_episodes.csv` | 200 | task, episode_index, house, seed of the orientation sweeps (Figure 6 b-d) |
| `wrist_steer/heldout_houses.csv` | 34 | the held-out houses of the chart, the operator and the calibration (Figure 6) |
| `wrist_steer/steer_episodes.csv` | 8 | episode_index, house, seed of the steering scenes (Figure 6 c-d) |
| `folds/*.csv` | | house -> fold maps (5 folds) of the out-of-house probes: `target_position` (Figure 1 target), `distance` (Figures 1 and 3), `distance_teleport` (Figure 3a), `distance_plant` (synthetic activations, Figure 3c), `position` (Figure 2), `contact_natural` (Figure 4), `phase` (Figure 5), `gripper_probe` (Figure 7b) |

Every probe is scored out of house with these fold maps (`vlageom.probes.FoldMap`), not with folds
computed at run time: scikit-learn's `GroupKFold` orders equally sized houses differently on
different CPUs. [`tests/provenance/README.md`](../tests/provenance/README.md) describes how each
list was made, with the md5 of every file and the scripts that re-derive them from the paper's data.
