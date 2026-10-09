# Figure 7: gripper control

Part of [vla-geometry](../../README.md) · [all pipelines](../README.md).

Figure 7 has four data panels, placed side by side with the labels a-d in a layout made by hand;
the repository writes each panel as its own file pair (`$VLAGEOM_FIGURES/fig7{a,b,c,d}.{pdf,png}`):

| panel | file | what is drawn | drawn from |
|---|---|---|---|
| a | `fig7a` | explained-variance ratio of the first 16 principal components of the gripper edits, and the stable rank (sum of squared singular values / the largest one squared) | `panels/gripper_fig7.npz` (`scree_evr`, `stable_rank`) |
| b | `fig7b` | R² of a ridge probe from the layer-32 activation, projected onto the top k edit components, to the decoded gripper command, against k; dashed: the same probe on the full 768-D activation | `panels/gripper_fig7.npz` (`probe_k`, `probe_r2`, `probe_r2_full`) |
| c | `fig7c` | activations near the grasp on edit PC1-PC2, coloured by the decoded gripper command, with the mean closing and opening edit per grid cell | `panels/gripper.npz` (the Figure 1 gripper panel) |
| d | `fig7d` | arm leakage of edits restricted to the mean and top k components of the edits of all other houses, as the median over queries relative to the median leakage of the unrestricted 768-D edit (%), for closing and opening edits | `panels/gripper_fig7.npz` (`leak_k`, `leak_close`, `leak_open`, `free_leak_close`, `free_leak_open`) |

All four panels start from the gripper edits of the Figure 1 gripper tile (`pipelines/gripper/`,
1,666 edits of the 304 episodes in `lists/gripper_episodes.csv`, 170 houses). Panels a and b use
the principal components of the edits of 80% of the houses (136 houses, 1,329 edits); panel c uses
those of all 1,666 edits (the Figure 1 panel); panel d builds, for each episode, the components of
the listed edits of all houses except the episode's own. The stages are under
`pipelines/gripper_fig7/`, the library code in `vlageom/gripper_fig7.py` (the low-rank edit fit is
`vlageom.steer.delta.optimize_delta_basis`), the render in `figures/fig7_gripper.py`.

| stage | what it does | compute | output (under `$VLAGEOM_DATA`) |
|---|---|---|---|
| `pipelines/gripper/10_harvest_edits.py --save-captures` | the gripper edits (see [Figure 1, gripper tile](../gripper/README.md)), and the per-query model inputs of each grasping episode, which stage 30 reads | GPU + EGL, about 57 H200 GPU-hours | `gripper/deltas/` (1,666 files), `gripper/captures/` (33 GB) |
| `pipelines/gripper/20_panel.py` | the Figure 1 gripper panel, also panel c | CPU, seconds | `panels/gripper.npz` |
| `10_basis.py` | houses split 80/20 (numpy `RandomState(0)` permutation of the sorted houses); SVD (torch, CPU, float32) of the train houses' edits minus their mean | CPU, about 1 min (needs torch) | `gripper_fig7/basis.npz` |
| `20_probe.py` | scikit-learn `Ridge(alpha=10)` from the activation (projected onto the top k = 1, 2, 4, ..., 256, 768 components of stage 10, and the full activation) to the decoded gripper command before the edit; pooled out-of-fold R² over the house folds of `lists/folds/gripper_probe.csv` | CPU, about 20 s | `gripper_fig7/probe.json` |
| `30_ksweep.py --task K` | per episode: bases from the listed edits of all other houses (torch SVD on the GPU); at one query per grasp landmark (before_grasp, after_grasp), the gripper-edit loss refitted with the edit restricted to mu + V[:, :k] a for k = 0, 1, 2, 4, ..., 512, 640, 768 (Adam on a, 1000 iterations, started at the coordinates of the mean closing or opening edit), and the unrestricted 768-D edit at the same query | GPU, about 40 min per episode on one H200; 304 episodes, about 210 GPU-hours. Reads the stored model inputs, so nothing is rendered | `gripper_fig7/ksweep/ep<E>_h<H>_<landmark>.json` (one record per landmark, about 50 KB each) |
| `40_ksweep_summary.py` | per landmark and k: flip rate, median arm leakage and jerk; the same for the 768-D edits; the knee; the same summary over the queries whose unedited command is opposite to the target | CPU, seconds | `gripper_fig7/ksweep_summary.json` |
| `50_panel.py` | the arrays of panels a, b and d (a and b only, when stage 40's summary does not exist) | CPU, seconds | `panels/gripper_fig7.npz` |
| `figures/fig7_gripper.py` | renders the four panels (`--panel a b c d` selects; by default panel d is left out, with a message, when the panel file has no panel-d arrays) | CPU, about 10 s | `figures/fig7{a,b,c,d}.{pdf,png}` |

```bash
source scripts/harvest_env.sh                                           # GPU stages: EGL, offline model cache
python pipelines/gripper/10_harvest_edits.py --task K --save-captures   # GPU, K = 0..151 (the Figure 1 gripper harvest)
python pipelines/gripper/20_panel.py

python pipelines/gripper_fig7/10_basis.py
python pipelines/gripper_fig7/20_probe.py
python pipelines/gripper_fig7/30_ksweep.py --task K                     # GPU, K = 0..303 (one episode per task)
python pipelines/gripper_fig7/40_ksweep_summary.py
python pipelines/gripper_fig7/50_panel.py

python figures/fig7_gripper.py                                          # a-d; a-c without stages 30-40
```

Stage 30 takes about 210 GPU-hours. Without it (and stage 40), stage 50 writes the arrays of
panels a and b only, `figures/fig7_gripper.py` renders a-c and says that d is left out, and
`tests/check_expected.py --fig 7` checks panels a-c (panel d's stage 30 is checked by the canary
`tests/canary_gripper_ksweep.sh`).

Notes:

- **Stage 10 needs torch** (the CPU build is enough; `[analysis]` or `[sim]`). The SVD runs with
  8 torch threads (`--threads`); the last bits of the singular values depend on the number of
  threads. The signs of the columns of V are those the SVD returns; nothing downstream depends on
  them (the probe scores are the same for either sign of a column).
- **Stage 20 folds.** The probes are scored over the shipped fold map `lists/folds/gripper_probe.csv`
  (170 houses, 34 per fold), assigned once by scikit-learn `GroupKFold(5)` over the house of each of
  the 1,666 rows on an AVX-512 machine. `FoldMap.splits` gives the same (train, test) indices as
  `GroupKFold.split` did there.
- **Stage 30 inputs.** The bases are built from the edit files of the listed episodes in
  `$VLAGEOM_DATA/gripper/deltas` (`--edits`), and the model inputs are read from
  `$VLAGEOM_DATA/gripper/captures`, written by `pipelines/gripper/10_harvest_edits.py
  --save-captures` (`--from-captures` to read them elsewhere). The fits depend only on these stored
  inputs, not on the simulator scene. Without stored inputs, `--rollout` rolls each episode out
  again (GPU + EGL). Records that exist are skipped (`--overwrite` fits them again); a task exits
  with status 1 when an episode failed or has no stored inputs. Stage 40 summarizes the records
  that exist and prints the listed episodes without records.
- **Landmark queries.** Stage 30 fits one query per landmark (`vlageom.steer.landmarks.find_landmarks`:
  the query 3 before the stable-grasp onset, or the nearest earlier query with the gripper open,
  and the query 3 after it within the closed-and-holding run), while the gripper tile's edits use
  up to 3 queries per landmark.
- **Panel c** is drawn from the Figure 1 gripper panel file; its arrow scale is stored as float64
  and used in float32, the precision it was computed in.

Printed values with the paper's edits:

- stage 10: 1,329 train edits, evr of PC1 0.698, stable rank 1.43 (panel a text), centered rank 768;
- stage 20: R² 0.860 (k = 1), 0.869, 0.897, 0.929, 0.954, 0.977, 0.992, 0.997, 0.998, 0.9989
  (k = 768); full activation 0.9989 (Pearson r 0.9994); 1,666 rows, 170 houses;
- panel c: "PC1 (70% var)", "PC2 (6% var)".

## Reproducibility (Figure 7)

- Stages 10, 20 and 50 and the renders reproduce panels a-c bit for bit from the gripper edits
  (`tests/check_expected.py --fig 7`).
- Panel d comes from stage 30 (GPU) and stages 40 and 50. The canary
  `tests/canary_gripper_ksweep.sh` checks stage 30 on one episode against the records of an earlier
  run fitted from the same edit files and model inputs.
