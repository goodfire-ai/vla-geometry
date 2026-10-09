# Provenance of the shipped lists

The files in `lists/` define which episodes, houses and cross-validation folds the paper used.
They were written once from the paper's data stores. The scripts in this folder re-check them
against those stores; every store location is a command-line argument, and each check runs only
when its argument is given.

| script | lists | command |
|---|---|---|
| `check_lists.py` | Figure 1: `gripper_episodes.csv`, `wrist_episodes.csv`, `teleport_houses.csv`, `near_houses.csv`, `folds/{target_position,distance,distance_teleport}.csv` | `python tests/provenance/check_lists.py --seed-catalog <seed catalog> --gripper-edits <gripper edits> --wrist-sweeps <wrist sweeps> --teleport-store <teleport store root> --near-store <near-range store root> --fold-npz target_position=<npz> --fold-npz distance=<npz> --fold-npz distance_teleport=<npz>` |
| `check_position_lists.py` | Figure 2: `position/*.csv`, `folds/position.csv` | `python tests/provenance/check_position_lists.py --bench --crossed-houses <crossed house files> --hand-lines <hand-line dir> --object-lines <object-line root> --labels <packed labels.npz> --natural-cache <natural rollouts' query cache>` |
| `check_distance_fig3_lists.py` | Figure 3: `folds/distance_plant.csv`, `folds/distance_teleport.csv` on the erasure rows | `python tests/provenance/check_distance_fig3_lists.py plant`; `... teleport [--teleport-store DIR] --expect-md5 45590095eeec32a93dac233b50356fde` |
| `check_contact_lists.py` | Figure 4: `contact/*.csv`, `folds/contact_natural.csv` | `python tests/provenance/check_contact_lists.py --seed-catalog <seed catalog> --episodes-dir <rollout metadata> --episode-set <natural cache> --natural-cache <natural cache> --offschedule-dir <off-schedule harvest>` |
| `check_phase_lists.py` | Figure 5: `phase/*.csv`, `folds/phase.csv` | `python tests/provenance/check_phase_lists.py --validity <house scan> --rollouts <natural episodes> --accounting <job accounting> --cache <natural table> --freeze-store <freeze-test files> --decode-store <decode-set dir> --fold-bank <fold bank> --trajectory <capture>` (see its `--help`) |
| `check_wrist_steer_lists.py` | Figure 6: `wrist_steer/*.csv` | `python tests/provenance/check_wrist_steer_lists.py --seed-catalog <seed catalog> --natural-cache <natural cache> --orient-dir <sweep store> --rollout-logs <rollout logs>` |
| `check_gripper_fig7_lists.py` | Figure 7: `folds/gripper_probe.csv` | `python tests/provenance/check_gripper_fig7_lists.py --edits <gripper edit files>` |

Every fold map is a house -> fold assignment (5 folds) made once by scikit-learn `GroupKFold(5)`
over the house of each row on an x86-64 machine with AVX-512: a house's fold is the index of the
split in which its rows are the test rows. `GroupKFold` orders houses with equal row counts by a
sort whose result depends on the CPU's sort kernel, so the scripts repeat an assignment exactly
only on such a machine; elsewhere they report how many houses fall into other folds. The sha1
values below are of the sorted (house, fold) int64 pairs.

## Figure 1

| file | rows | columns | content |
|---|---|---|---|
| `lists/gripper_episodes.csv` | 304 | episode_index, house, seed, new_sampler | The episodes whose gripper edits make up the paper's gripper catalog: 1,666 edits at action-expert layer 32 over 170 houses. `episode_index` is the episode's index in the rollout seed catalog (houses 0-199) and names the output files (`ep{episode_index}_h{house}_...`); `seed` is the simulator seed passed to the task sampler (it differs from `episode_index` for 70 episodes). `new_sampler` is 1 when the paper's rollout of the episode was sampled by a newly created MolmoSpaces task sampler (219 episodes) and 0 when it was sampled by an existing one (85), as in the paper's rollouts; `vlageom.gripper.sample_episode` samples each episode the same way. Sorted by (house, episode_index). |
| `lists/wrist_episodes.csv` | 64 | catalog_index, episode_index, house, seed | The wrist-sweep episodes, one per house: entry `catalog_index` of the catalog that keeps the first successful episode of each house (191 houses), for catalog_index 0, 3, ..., 90, 93, 97, ..., 190. |
| `lists/teleport_houses.csv` | 906 | house, bench, bench_idx | Houses of the object-teleport store. `bench` / `bench_idx` give the MolmoSpaces benchmark episode used for the house (the first episode of the house; FrankaPickHardBench when the house is in it, else FrankaPickDroidMiniBench). The benchmark catalog has 947 houses; the store holds these 906. |
| `lists/near_houses.csv` | 300 | house, bench, bench_idx | Houses of the near-range store: a random subset (numpy `default_rng(45)` permutation, first 300, kept in house order) of the 448 teleport houses where an object placement within 0.35 m of the natural end-effector position is feasible. |
| `lists/folds/target_position.csv` | 906 | house, fold | Target-position probes (first-query rows of the teleport store, 13,145 rows). sha1 `4ac4e9da4a0755531c9bdcc8a9e438b419d354e5`. |
| `lists/folds/distance.csv` | 905 | house, fold | Distance probes on the combined teleport + near-range rows with distance < 0.70 m (the `fold_id` of the distance probe bank, one fold per house); read by distance stages 30 and 60. sha1 `0f805672609d27d3a8df635423826fed3d50efd9`. |
| `lists/folds/distance_teleport.csv` | 905 | house, fold | Distance probes on the teleport rows alone: 52,210 rows with distance < 0.70 m in 905 houses (house 803 has none); read by `pipelines/distance/50_erasure.py` (Figure 3a). md5 of the concatenated test-row indices of folds 0-4 on the paper's store `45590095eeec32a93dac233b50356fde`. sha1 `44bd2c7b4dc3c26f6e183684b34fc60ecdb246a4`. |

## Figure 2

| file | rows | content |
|---|---|---|
| `lists/position/crossed_houses.csv` | 947 | house, bench, bench_idx: the benchmark catalog of the object-teleport harvests (`vlageom.sim.build_house_list(950)`), all of whose houses the crossed harvest runs (`--bench` re-derives it from the benchmark episodes) |
| `lists/position/line_houses.csv` | 48 | house, bench, bench_idx: the houses of the straight hand and object lines, a random pick of 48 of the 698 object-teleport houses in which all 16 placements were kept |
| `lists/position/ee_color_reference.csv` | 4,000 | x, y, z (m, robot base frame): the colour reference of the EE trace grid, a numpy `default_rng(0)` subsample of the hand positions at the model queries of 299 natural MolmoBot-DROID rollouts (9,800 queries, 192 houses), float64 representations of the stored float32 values |
| `lists/folds/position.csv` | 945 | house -> fold of position stages 30 and 40: `GroupKFold(5)` over house on the q1 rows (the same on the q0 rows); `FoldMap.splits` equals GroupKFold's index arrays. sha1 `1a434eb17771ba144626e1ea3ad7253582955191` |

## Figure 3

| file | rows | content |
|---|---|---|
| `lists/folds/distance_plant.csv` | 120 | house -> fold of the synthetic activations of `pipelines/distance/80_plant_cosines.py` (24 houses per fold): `GroupKFold(5)` over the synthetic house label of the 24,000 rows; 105 of the 120 houses share their row count with another house. sha1 `31ce9064d9f16c709058d653b5a0c319a64739ce`. `check_distance_fig3_lists.py plant` repeats the assignment (no data needed). |

Stage 50 reads `lists/folds/distance_teleport.csv` and stages 30 and 60 read
`lists/folds/distance.csv` (Figure 1 above).

## Figure 4

| file | rows | content |
|---|---|---|
| `lists/contact/natural_episodes.csv` | 299 | episode_index, house, seed, orig_n_steps, orig_contact, task_description: episodes 0-299 of the natural rollouts (the first 300 metadata files by name) that have at least 5 model queries (episode 194 has 4); house and seed as recorded in the seed catalog, the last three columns from the listed rollout's metadata (length, gripper-object contact at each query, task description) |
| `lists/contact/offschedule_episodes.csv` | 128 | task, episode_index, house, seed, outcome: the first 128 rows of the catalog that keeps, for each house in increasing order, its successful natural rollout with the lowest episode index; `outcome` (`shard`, `no_scene`, `no_crossing`) the result of the paper's harvest (tasks 29, 99 and 101 raise HouseInvalidForTask; tasks 30 and 120 give no placement) |
| `lists/folds/contact_natural.csv` | 192 | house -> fold of the natural contact probes: `GroupKFold(5)` over the house of each of the 18,837 natural rows; 99 houses have 63 rows each. sha1 `98b0e978cdcbb1bbae914a6c3c8e55cbba2527c0` |

## Figure 5

| file | rows | content |
|---|---|---|
| `lists/phase/natural_episodes.csv` | 530 | episode_index, partition, house, seed, new_sampler: catalog = the 181 valid houses (increasing) x 3 seed rounds, episode_index = round * 181 + position, seed = 20,000,000 + 1000 round + house, partition = episode_index // 4; the rows are the episodes with a file (13 catalog entries cannot be sampled with their seed). new_sampler = 1 when the episode was the first one the process that wrote it sampled, from the job records and file times of the paper's rollouts (confirmed by the rollout logs for 430 episodes) |
| `lists/phase/freeze_episodes.csv` | 128 | task, episode_index, house, seed, decode: the first 128 houses of the successful natural episodes, one episode per house, houses with a place label first; decode = 1 for the 122 episodes of the decode set (panel c) |
| `lists/phase/trajectory_episode.csv` | 1 | episode_index, house, seed, new_sampler of the example rollout (249, 76, 20001076, 0) |
| `lists/phase/trajectory_preceding.csv` | 2 | the two episodes the capture process rolled out before the example rollout, in order (431, 77, 20002077, 1; 528, 183, 20002183, 0), as in the paper's capture |
| `lists/folds/phase.csv` | 167 | house -> fold of the phase decoders: `GroupKFold(5)` over the 7,738 successful natural queries (fold hash over the rows `c8b301a92149`, `vlageom.phase.fold_hash`) |

`check_phase_lists.py --write DIR` writes the lists from the given inputs instead of comparing
them.

## Figure 6

| file | rows | content |
|---|---|---|
| `lists/wrist_steer/natural_episodes.csv` | 200 | episode_index, partition, house, seed of the natural rollouts (episodes 0-199 of the natural rollout catalog of houses 0-199); the row order is the row order of the natural cloud |
| `lists/wrist_steer/orient_episodes.csv` | 200 | task, episode_index, house, seed: the first 200 entries of the catalog that keeps, per house in increasing order, the first two successful natural rollouts (houses 0-109) |
| `lists/wrist_steer/heldout_houses.csv` | 34 | the held-out houses of the chart, the operator and the calibration: the first third of a numpy `default_rng(0)` shuffle of the 102 sweep houses |
| `lists/wrist_steer/steer_episodes.csv` | 8 | episode_index, house, seed of the steering scenes: the first successful natural rollout of each of the first 8 held-out houses (in house order) |

## Figure 7

| file | rows | content |
|---|---|---|
| `lists/folds/gripper_probe.csv` | 170 | house -> fold of the Figure 7b probes: `GroupKFold(5)` over the house of each of the 1,666 edit rows of the listed episodes (file-name order), 34 houses per fold. sha1 `9d668f53af7dbb5e720e3a7887717f1f1a43f48c` |

## md5 of the list files as shipped

```
0aa1eba03363c58d54311b33573ade30  lists/gripper_episodes.csv
c4f5c9c115e8e1c9ca06fb373f4bf469  lists/wrist_episodes.csv
81913f8c78689fadd45b368690c1b064  lists/teleport_houses.csv
493867246dd95e2136f5756c5dd75b16  lists/near_houses.csv
6d91ea1ada116ddc3c1073a1be02f3c7  lists/folds/target_position.csv
126ab6ccc7a883d222a0dd3e39c086e2  lists/folds/distance.csv
cc2169b5b11f57ce87a7d6e5774ff824  lists/folds/distance_teleport.csv
65a0599f4a3878b9b2e71b6774b7790c  lists/folds/distance_plant.csv
47079b5554edc6e7245dfc00162cf6fd  lists/folds/position.csv
3e67fc1a465b2fc9e65e8b0bcad499a9  lists/folds/contact_natural.csv
97fda904a0dbbce24810736b026cce67  lists/folds/phase.csv
3a48f051e37a8d173a303dedca6190f3  lists/folds/gripper_probe.csv
09894795c2fd3949a74b839f38d0aacc  lists/position/crossed_houses.csv
5586f9fb4a52b63b54f099b1ba16d5b0  lists/position/line_houses.csv
2cad2af0ef270daf0235da0a6a673a64  lists/position/ee_color_reference.csv
da63ac05e83e82d764d564534f320eda  lists/contact/natural_episodes.csv
572075b3cf168b4c39ddc023aacf9913  lists/contact/offschedule_episodes.csv
1a773a4fda691e3778ef5e7fd149baca  lists/phase/natural_episodes.csv
d4fbcab4daed0286897dc5daaf832914  lists/phase/freeze_episodes.csv
0de4d85be11b43398509c56a4157bc67  lists/phase/trajectory_episode.csv
f169736895626bf4929e91ebe4cde2cd  lists/phase/trajectory_preceding.csv
1f9429d89da6bed0f049a6264b0ee79b  lists/wrist_steer/natural_episodes.csv
988c3a6843285aa241d1a927c9619ede  lists/wrist_steer/orient_episodes.csv
5722ce9e492fd4fe4259563d7838307b  lists/wrist_steer/heldout_houses.csv
4b1d49f2d421ac565f732372d2c14fb6  lists/wrist_steer/steer_episodes.csv
```
