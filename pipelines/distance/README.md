# Figure 1 distance tile and Figure 3

Part of [vla-geometry](../../README.md) · [all pipelines](../README.md).

Stages 10-40 make the distance tile of Figure 1. Figure 3 uses stage 30 (panel d) and stages
50-90, and reads the packed stores of the [target](../target/README.md) and distance tiles.

## Figure 1: distance tile

The distance tile plots, for rows with true distance d < 0.70 m, the held-out read-outs of two ridge
probes on VLM last-token activations at layer 21: the metre ruler (target max(d, 15 cm), x axis) and
the centimetre ruler (target min(d, 15 cm), y axis), against the true distance (height). Rows come
from two harvests: the [target tile](../target/README.md)'s object-teleport store and this tile's
near-range harvest, which adds placements within 0.35 m of the hand so that the last 15 cm of the
approach is well covered. Placement 0 of the near-range harvest repeats the teleport harvest's
natural placement with the same seeds, so its queries q0-q3 are already in the teleport store and
are taken from there only.

| stage | what it does | compute | output (under `$VLAGEOM_DATA`) |
|---|---|---|---|
| `10_near_harvest.py` | 300 houses (`lists/near_houses.csv`) x 8 object placements, up to 8 model queries each; VLM token pools, action-expert tokens and labels per query | GPU + EGL, about 12 H200 GPU-hours (75 shards of 4 houses) | `distance/near/houses/h<house>.npz` (14 GB), `distance/near/frames/` |
| `20_pack_near.py` | packs the house files into memory-mappable arrays | CPU, about 3 min | `distance/near/prep/` (1.4 GB) |
| `30_ruler_probes.py` | combined rows (67,778 rows, 905 houses), out-of-house ridge probes over `lists/folds/distance.csv`, one BLAS thread | CPU, about 3 min, 4 GB of memory | `distance/ruler_probes.npz` + `.json` |
| `40_panel.py` | stratified plot sample (15,934 points), per-bin mean path, scores | CPU, seconds | `panels/distance.npz` |

```bash
source scripts/harvest_env.sh                             # GPU stages: EGL, offline model cache
python pipelines/distance/10_near_harvest.py --shard K    # GPU, K = 0..74 (default $SLURM_ARRAY_TASK_ID)
python pipelines/distance/20_pack_near.py
python pipelines/distance/30_ruler_probes.py              # needs $VLAGEOM_DATA/target/teleport/prep
python pipelines/distance/40_panel.py
```

`figures/fig1.py` draws Figure 1 from the four tile panels (see
[`figures/README.md`](../../figures/README.md#figure-1)).

The object-teleport store records the first four model queries of every placement (`chunk_q`
0-3); the near-range harvest records up to eight (`chunk_q` 0-7, steps 0-56) and stops a rollout
at the first query at which the task reports success, so later queries have fewer rows (see
Figure 3, [Data](#data)). Both harvests reset the policy before the first query of every placement.
Stage 10 exits with code 3 when a rendered camera frame is blank (an EGL failure of that process)
and skips houses already written, so a failed shard can simply be run again. Stage 30 reads the
target tile's packed store from `$VLAGEOM_DATA/target/teleport/prep` (`--teleport-store` to read it
elsewhere) and prints the two scores shown in the figure legend: R^2 = 0.8117 for the metre ruler on
rows with d >= 15 cm and 0.6965 for the centimetre ruler on rows with d < 15 cm
("0.81 (m) / 0.70 (cm)").

## Figure 3 (distance to the target object)

Figure 3 is a composite: the robot sketch, the title "Distance to Target Object", the panel
letters and the layout are drawn by hand, and four rendered data panels are placed in it, one
file pair each (`$VLAGEOM_FIGURES/fig3{a,b,c,d}.{pdf,png}`):

| file | panel | what is drawn | drawn from |
|---|---|---|---|
| `fig3a` | a | held-out R² of the near probe (< 15 cm) and the far probe (15-70 cm) after k = 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 directions of one of them are erased; solid lines, filled circles: far-probe directions erased; dashed lines, open squares: near-probe directions erased (7.2 x 4.4 in) | `panels/fig3a_erasure.npz` (stage 50) |
| `fig3b` | b | one point per boundary c = 3 ... 56 cm: R² on the rows below c (x) against R² on the rows at or above c (y), for the range-split pair, the round-to-c pair and the full-range probe recalibrated per range; range-split points labelled with c (6.8 x 6.2 in) | `panels/fig3b_cutoff.npz` (stages 60, 70) |
| `fig3c` | c | cosine between the near and far probe weights against c, for the range-split pair and the round-to-c pair: measured (mean over folds, band = std over folds), split-half ceiling (dashed), and the same fit on synthetic activations that carry the distance along one direction (dotted) (7.2 x 6.2 in) | `panels/fig3c_cosine.npz` (stages 60, 80) |
| `fig3d` | d | centimetre-ruler read-out against metre-ruler read-out (the two Figure 1 rulers, out of house), 15,934 rows coloured by the true distance in metres (left, 0.15-0.70 m) and in centimetres (right, 0-15 cm), with the per-bin mean path and dashed lines at 15 cm (11.6 x 5.0 in) | `panels/fig3d_plane.npz` (stage 30) |

All panels are rendered at 350 dpi with a tight bounding box; the sizes above are before the crop.
The stages are under `pipelines/distance/` (stages 50-90), the library code in
`vlageom/distance.py`, the render in `figures/fig3_distance.py`.

### Data

Figure 3 needs no harvest of its own. All panels read the VLM's last-token activation at layer 21
from the two packed stores of Figure 1: the object-teleport store of the target tile
(`target/teleport/prep`) and the near-range store of the distance tile (`distance/near/prep`).

- **Model queries.** The policy queries the model every 8 simulator steps. The object-teleport
  harvest records the first four queries of every object placement (`chunk_q` 0-3, steps 0, 8,
  16, 24). The near-range extension records up to eight (`chunk_q` 0-7, steps 0-56) and stops a
  rollout at the first query at which the task reports success, so later queries have fewer rows
  (in the near store, 2,142 rows at q0 and q1, then 2,141, 2,141, 2,133, 2,123, 2,050 and 1,920
  at q7). In the combined rows of panels b-d (67,778 rows), queries q4-q7 give 8,202 rows (12%)
  and 5,344 of the 11,534 rows below 15 cm (46%). Stage 90 prints these counts.
- **Frame history.** Both harvests reset the policy before the first query of every placement, so
  at q0 the model sees only the current frame pair. From q1 on, the policy's input also holds the
  frame pair of 8 simulator steps earlier, from the same rollout.
- **Row sets.**
  - Erasure (panel a): the teleport-store rows alone with distance d < 0.70 m: 52,210 rows in 905
    houses (house 803 has no placement within 0.70 m), 4,422 of them below 15 cm (404 houses).
    Folds: `lists/folds/distance_teleport.csv`.
  - Probe bank, cutoff table, cosines, plane (panels b-d): the combined rows of the Figure 1
    rulers (stage 30): 67,778 rows in 905 houses, 15,568 of them from the near store, 11,534
    below 15 cm. Folds: `lists/folds/distance.csv`.

Every probe is a closed-form ridge probe on activations standardized on its fit rows, scored out of
house: the probe of fold f is fit on the houses of the other four folds and read out on fold f's
houses.

### Stages

| stage | what it does | compute | output (under `$VLAGEOM_DATA/distance`) |
|---|---|---|---|
| `30_ruler_probes.py` | the two Figure 1 rulers (also panel d's read-outs) | CPU, about 2 min, 1 BLAS thread | `ruler_probes.npz`, `.json` |
| `50_erasure.py` | panel a. Near probe (rows < 15 cm) and far probe (rows >= 15 cm) of d in metres. Ridge penalty per probe chosen on held-out training houses: in each fold every 5th sorted training house is a validation house, penalties 1, 10, ..., 1e6, best validation R²; the most frequent pick over the folds is used (1e3 near, 1e4 far). Per fold and erased probe, 32 times: refit the probe on the current activations, make its weight vector orthogonal to the directions already erased (Gram-Schmidt), normalize it and project it out of the training and test activations. At k = 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 both probes are refit and read out. R², MAE and calibration slope pooled over folds | CPU, 24 BLAS threads: about 9 min (far directions) + 6 min (near directions) on 24 cores, 3 GB | `erasure/erase_far.json`, `erasure/erase_near.json` |
| `60_probe_bank.py` | panels b and c. For every c in 3, 5, 7, 10, 15, 20, 25, 30, 40, 56 cm: round-to-c probes `cm<c>` (target min(d, c)) and `far<c>` (max(d, c)) fit on all rows, range-split probes `nr<c>` / `fr<c>` (target d) fit only on the rows below / at or above c (at least 150 rows from 20 houses), and the full-range probe `m` (d in metres); penalty 100. Out-of-fold read-outs on every held-out row, weights per fold in activation units, per-fold cosine between the near and far weights of each pair, and split-half ceilings: each probe fit on two random halves of the houses (3 draws; the first half of a numpy `default_rng(45)` permutation of the sorted integer house ids), cosine of the two weight vectors | CPU, 16 BLAS threads: about 8 min on 16 cores, 9 GB | `probe_bank_last_tok_L21.npz`, `.json` |
| `70_cutoff_table.py` | panel b. Per c and range (near: rows < c, far: rows >= c), from the stored out-of-fold read-outs: R² and MAE of the range-split probe, the round-to-c probe, the full-range probe raw and recalibrated per range (in each fold, a + b * read-out fit by least squares on the range's training rows, using their out-of-fold read-outs), and the training mean | CPU, seconds | `cutoff_table_last_tok_L21.json` |
| `80_plant_cosines.py` | panel c, dotted curves. Synthetic activations (numpy `default_rng(11)`): 24,000 rows in 120 houses, 256 features, X = 0.06 * d_cm * w + 0.5 * N(0, 1) + a per-house offset N(0, 0.3²), one random unit direction w, d_cm uniform on 1-15 cm (30% of the rows) or 1-70 cm; the probes of stage 60 fit on it over `lists/folds/distance_plant.csv`; per c the near/far weight cosines of both pairs | CPU, 4 BLAS threads, 0.5-2 min | `plant_cosines.json` |
| `90_fig3_panels.py` | the four panel files; prints the numbers quoted in the text and the row counts per model query | CPU, seconds | `$VLAGEOM_DATA/panels/fig3{a_erasure,b_cutoff,c_cosine,d_plane}.npz` |
| `figures/fig3_distance.py` | renders the four panels (`--panel a b c d` selects) | CPU, about 10 s | `$VLAGEOM_FIGURES/fig3{a,b,c,d}.{pdf,png}` |

```bash
# inputs, as for Figure 1: target stages 10 and 20 (pipelines/target/), distance stages 10 and 20
python pipelines/distance/30_ruler_probes.py                # panel d (and Figure 1)
python pipelines/distance/50_erasure.py                     # panel a
python pipelines/distance/60_probe_bank.py                  # panels b, c
python pipelines/distance/70_cutoff_table.py                # panel b
python pipelines/distance/80_plant_cosines.py               # panel c
python pipelines/distance/90_fig3_panels.py
python figures/fig3_distance.py
```

Notes:

- **BLAS threads.** The last bits of a ridge fit depend on the number of BLAS threads (1, 8, 16 and
  24 threads give four different results on the same rows). Each stage sets the count of the
  paper's run in code (threadpoolctl): 24 for stage 50, 16 for stage 60, 4 for stage 80 and 1 for
  stage 30 (`--threads`); stages 70 and 90 use no BLAS. On a machine with fewer cores the same
  counts are used and the stages take longer (on 8 cores, stage 60 about 110 min);
  `OPENBLAS_THREAD_TIMEOUT=4` in the environment shortens the idle spinning of the BLAS threads and
  does not change the results.
- **Erasure.** Each erased direction is the refit probe's weight vector after it is made orthogonal
  to the directions erased before it (twice, for numerical accuracy), so k passes erase k
  orthonormal directions. Each output file records, per fold, the largest off-diagonal entry of
  the Gram matrix of the erased directions (about 1e-15) and the largest |activation · direction|
  left on the erased probe's training rows (about 5e-6, float32). `--k-max` stops earlier (a quick
  check of the first points); `--erase far` or `--erase near` runs one direction.
- **Probe bank.** The range-split probes are standardized on their own fit rows, and their weights
  are mapped back to activation units (weights / per-feature std) before cosines are taken, so the
  near and far weights are compared in one frame. They read out every held-out row, inside and
  outside their range; panel b scores each probe on its own range only. The split-half halves are
  drawn from the integer house ids.
- **Panel b labels.** The c labels of the range-split points sit 9 pt above the point for c <= 15
  and 11 pt to its right for c >= 20; the label of c = 20 sits 9 pt higher as well
  (`B_LABEL_OFFSETS` in `figures/fig3_distance.py`) so that it clears the label of c = 25. Points
  outside the axes window (c = 56 for the range-split pair, c = 3 and 56 for round-to-c) are in the
  panel file but not drawn.
- **Panel d** draws the read-outs of the Figure 1 rulers, `cm15` = min(d, 15 cm) and `far15` =
  max(d, 15 cm), on the stratified plot sample of Figure 1 (`vlageom.distance.plot_samples`: every
  row below 15 cm, at most 400 rows per distance bin above, numpy `default_rng(45)`); the path is
  the per-bin mean of that sample (bins with at least 30 rows). Point colours are the true distance
  in metres (left) and in centimetres (right).
- **Synthetic folds.** `lists/folds/distance_plant.csv` holds the house -> fold map of the 120
  synthetic houses; 105 of them share their row count with another house, so a fold assignment
  computed at run time by `GroupKFold` would depend on the CPU's sort kernel.

### Numbers (Figure 3)

Printed by stage 90 (these are the values in the text):

- panel a (52,210 rows, 905 houses; 4,422 near rows in 404 houses; penalties 1e3 near, 1e4 far):
  with nothing erased, near R² 0.7084 (MAE 1.30 cm) and far R² 0.8435 (MAE 3.63 cm); after 32 far
  directions are erased, near 0.7068 and far 0.0282; after 32 near directions, near 0.0573 and far
  0.8436;
- panel b: the range-split pair at c = 15 scores near R² 0.7012 and far R² 0.7964;
- panel c (67,778 rows, 905 houses): at c = 15 the round-to-c pair has cosine 0.0447 (std over folds
  0.0113) against a ceiling of 0.2308, the range-split pair 0.0399 (0.0167) against 0.1772; on the
  synthetic activations 0.9509 (round-to-c) and 0.9239 (range-split); the cm15 probe's MAE below
  15 cm is 1.4544 cm, and 1.4555 cm on the near-store rows where the object touches nothing but its
  support;
- panel d (67,778 rows, 11,534 below 15 cm; 15,934 points, 17 path points): metre ruler R² 0.8117
  (rows >= 15 cm), centimetre ruler R² 0.6965 (rows < 15 cm); queries q4-q7: 8,202 rows (12.1%),
  5,344 of the rows below 15 cm (46.3%).
