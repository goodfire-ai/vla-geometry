# Checks

Three kinds of checks:

- **Expected values** (`check_expected.py`): the CPU outputs of Figures 1-7 (panel files,
  intermediate results, probe scores and the rendered files) are compared with values and hashes
  recorded in `expected_fig<N>.json`. This needs no reference data, only the outputs of the
  pipeline stages.
- **GPU canaries** run one unit of a GPU stage (one house, episode, partition or task) and compare
  its output with the same unit of a reference store, key by key and row by row.
- **CPU comparisons** compare packed stores, intermediate results, panel files and the rendered
  figures with reference files.

The GPU canaries, the CPU comparisons and the scripts in `provenance/` need reference stores from
the paper's runs (or from an earlier run of this repository); those data are not distributed with
the repository. Every script takes its input and reference locations as arguments, prints what
differs (largest absolute difference, differing rows or pixels) and exits with status 0 only when
everything it compared is identical (or within the tolerance it states).

## Expected values

```bash
python tests/check_expected.py                   # Figures 1-7
python tests/check_expected.py --fig 1           # one figure (also: --fig 2 5)
python tests/check_expected.py --skip-render     # without the renders (no fonts needed)
python tests/check_expected.py --fig 4 --skip-cache   # Figure 4 without hashing the 5.8 GB cache
```

| figure | script | what is recorded |
|---|---|---|
| 1 | `check_expected.py` (`expected_fig1.json`) | dtype, shape and SHA-1 of every array of the four panel files (except the path records `src`, `files`); the probe scores of the target and distance tiles; MD5 of `fig1.png` |
| 2 | `check_expected_fig2.py` | the probe values of both read-out cells, the MD5 of the merged Procrustes file of the q0 rows (`--skip-procrustes` leaves it out) and of the curvature result, the SHA-1 of every array of the three panel files, and the MD5 of the three PNGs and PDFs (PDF without /CreationDate) |
| 3 | `check_expected_fig3.py` | SHA-1 of every array of the four panel files, the values of the cutoff table (stage 70) and of the synthetic cosines (stage 80), and the MD5 of the four PNGs and PDFs (`--only` selects sections) |
| 4 | `check_expected_fig4.py` | SHA-1 of every array of the natural cache (`--skip-cache` leaves it out) and the off-schedule pool, the probe and held-out score values, the panel arrays, and the MD5 of the two PNGs and PDFs |
| 5 | `check_expected_fig5.py` | every array of the natural table, the decoder scan, the freeze-test decode, the five panel files, and the MD5 of the five PNGs and PDFs (`--only` selects sections) |
| 6 | `check_expected_fig6.py` | the panel arrays, panel a's metrics, the numbers file, the operator (`--skip-operator`), and the MD5 of the renders |
| 7 | `check_expected_fig7.py` | SHA-1 of every array of the edit basis, the probe values, the arrays of panels a and b, and the MD5 of the renders of panels a-c; panel d (stage 30) is checked by its canary |

The recorded values are those of the CPU stages run on the paper's stores with the pinned library
versions on an x86-64 CPU with AVX-512; each stage sets its BLAS thread count in code (see the
[Reproducibility](../pipelines/README.md#reproducibility) section of `pipelines/README.md`). The
renders match only with the Product Sans font (`VLAGEOM_FONT_DIR`) and the pinned matplotlib, numpy
and Pillow; with the DejaVu Sans fallback the text pixels differ. `--write` records the current
outputs instead of checking them.

## GPU canaries

Run each canary on one GPU with EGL (`source scripts/harvest_env.sh`), with the same GPU type as
the reference run (H200 for the paper). The policy runs in bfloat16, so a harvest reproduces
reference activations bit for bit only on the same GPU type, driver and library versions. Each GPU
stage skips units whose output exists, so write canary output to a fresh directory: `$C` below is
any empty scratch directory. Times include loading the model (about 2 min) on one H200.

| figure | stage | unit | run | compare | time |
|---|---|---|---|---|---|
| 1, 7 | `pipelines/gripper/10_harvest_edits.py` | episodes 0 (house 0) and 14 (house 14), 12 edit files | `python pipelines/gripper/10_harvest_edits.py --episode-index 0 14 --out $C/gripper` | `python tests/canary_gripper_harvest.py --out $C/gripper --reference <reference edits dir> --episode-index 0 14` | about 25 min |
| 1, 6a | `pipelines/wrist/01_sweep.py` | row 0 of `lists/wrist_episodes.csv` (house 0), 39 swept angles | `python pipelines/wrist/01_sweep.py --row 0 --out-dir $C/wrist` | `python tests/canary_wrist_harvest.py --canary $C/wrist/ep0.npz --reference <reference sweep file of catalog_index 0>` | about 5 min |
| 1, 3 | `pipelines/target/10_teleport_harvest.py` | house 44, 16 placements x 4 queries | `python pipelines/target/10_teleport_harvest.py --houses 44 --out $C/teleport` | `python tests/canary_target_harvest.py teleport --house 44 --new $C/teleport --ref <reference teleport store root>` | about 5 min |
| 1 | `pipelines/target/30_square_path_harvest.py` | house 44, 64 waypoints | `python pipelines/target/30_square_path_harvest.py --out $C/shape_square` | `python tests/canary_target_harvest.py square --new $C/shape_square/h44 --ref <reference h44 object-path dir>` | about 4 min |
| 1, 3 | `pipelines/distance/10_near_harvest.py` | house 0, 8 placements x up to 8 queries | `python pipelines/distance/10_near_harvest.py --houses 0 --out $C/near` | `python tests/canary_distance_harvest.py house --new-root $C/near --ref-root <reference near-range store root> --houses 0` | about 5 min |
| 2 | `pipelines/position/10_crossed_harvest.py` | house 44, 50 cells x 2 queries | `python pipelines/position/10_crossed_harvest.py --houses 44 --out $C/crossed` | `python tests/canary_position_harvest.py crossed --house 44 --new $C/crossed --ref <crossed store root>` | about 5 min |
| 2 | `pipelines/position/50_hand_lines.py` | house 44, 4 lines x 24 waypoints | `python pipelines/position/50_hand_lines.py --houses 44 --out $C/hand_lines` | `python tests/canary_position_harvest.py lines --house 44 --new $C/hand_lines --ref <hand-line dir>` | about 12 min |
| 2 | `pipelines/position/55_object_lines.py` | house 44, the line toward the hand | `python pipelines/position/55_object_lines.py --houses 44 --shapes line_ee --out $C/object_lines` | `python tests/canary_position_harvest.py shape --new $C/object_lines/shape_line_ee/h44 --ref <object-line root>/shape_line_ee/h44` | about 4 min |
| 2 | `pipelines/position/80_hand_shapes.py` | house 44, the x line | `python pipelines/position/80_hand_shapes.py --shapes line_x --out $C/shapes` | `python tests/canary_position_harvest.py shape --new $C/shapes/ee_line_x/h44 --ref <shapes root>/ee_line_x/h44` | about 4 min |
| 2 | `pipelines/position/85_object_shapes.py` | house 44, the x line | `python pipelines/position/85_object_shapes.py --shapes line_fwd --out $C/shapes` | `python tests/canary_position_harvest.py shape --new $C/shapes/obj_line_fwd/h44 --ref <shapes root>/obj_line_fwd/h44` | about 4 min |
| 4 | `pipelines/contact/10_natural_harvest.py` | task 0 (episode 0, house 0, 63 queries) | `python pipelines/contact/10_natural_harvest.py --task 0 --out $C/contact` | `python tests/canary_contact_harvest.py natural --new $C/contact/natural/ep0_h0.npz --reference <natural harvest dir>/ep0_h0.npz` | about 3 min |
| 4 | `pipelines/contact/30_offschedule_harvest.py` | task 0 (episode 0, house 0, 131 placements) | `python pipelines/contact/30_offschedule_harvest.py --task 0 --out $C/contact` | `python tests/canary_contact_harvest.py offschedule --new $C/contact/offschedule/ep0_h0.npz --reference <off-schedule dir>/ep0_h0.npz` | about 6 min |
| 5 | `pipelines/phase/05_prepare_assets.py --scan` | houses 0-3 (house 2 cannot host the task) | `python pipelines/phase/05_prepare_assets.py --skip-receptacles --skip-scenes --scan --houses 0 1 2 3 --out $C/house_validity.json` | `python tests/canary_phase.py scan --new $C/house_validity.json --ref <house validity scan>` | about 4 min |
| 5 | `pipelines/phase/10_natural_rollouts.py` | partition 0 (episodes 0-3, houses 0, 1, 3, 4; a new task sampler for episode 0, the existing one for 1-3) | `python pipelines/phase/10_natural_rollouts.py --partition 0 --out-dir $C/natural` | `python tests/canary_phase.py natural --new $C/natural --ref <natural episode store> --partition 0` | about 15 min |
| 5 | `pipelines/phase/40_freeze_harvest.py` | task 0 (episode 0, house 0, 8 landmarks, two of them closed-hand place) | `python pipelines/phase/40_freeze_harvest.py --task 0 --out-dir $C/freeze` | `python tests/canary_phase.py freeze --new $C/freeze --ref <freeze-test store> --task 0` | about 8 min |
| 5 | `pipelines/phase/60_trajectory_capture.py` | the example rollout | `python pipelines/phase/60_trajectory_capture.py --out $C/trajectory/traj_ep249_h76.npz` | `python tests/canary_phase.py trajectory --new $C/trajectory/traj_ep249_h76.npz --ref <reference capture>` | about 6 min |
| 6 | `pipelines/wrist_steer/10_natural_rollouts.py` | partition 0 (episodes 0-15, with the revisit of house 5) | `python pipelines/wrist_steer/10_natural_rollouts.py --partition 0 --out-dir $C/natural` | `python tests/canary_wrist_steer_natural.py --new $C/natural --ref-root <natural rollout store> --ref-cache <natural cache> --episodes 0-15` | about 30 min |
| 6 | `pipelines/wrist_steer/30_orient_sweep.py` | task 0 (house 0) | `python pipelines/wrist_steer/30_orient_sweep.py --task 0 --out-dir $C/orient` | `python tests/canary_wrist_steer_orient.py --canary $C/orient/ep0_h0.npz --reference <sweep store>/ep0_h0.npz` | about 8 min |
| 6 | `pipelines/wrist_steer/50_train_chart.py` | the full training on the reference sweep store | `python pipelines/wrist_steer/50_train_chart.py --orient-dir <sweep store> --out-dir $C/chart` | `python tests/canary_wrist_steer_chart.py --new $C/chart/chart.pt --reference <reference chart> --new-report $C/chart/chart_report.json --ref-report <reference report>` | about 4 min |
| 6 | `pipelines/wrist_steer/60_steer.py` | scene 1 (house 5), both modes, with the reference chart and operator | `python pipelines/wrist_steer/60_steer.py --mode persist --episodes 1 --chart <reference chart> --operator <reference operator> --out $C/steer/persist.json` (and `--mode oneshot ... --out $C/steer/oneshot.json`) | `python tests/canary_wrist_steer_steer.py --new $C/steer/persist.json --reference <reference persist json>` (and the one-shot pair) | about 5 min each |
| 7 | `pipelines/gripper_fig7/30_ksweep.py` | episode 0 (house 0), both grasp landmarks, from stored model inputs (no rendering) | `python pipelines/gripper_fig7/30_ksweep.py --episode-index 0 --edits <edits dir> --from-captures <model-input dir> --out $C/ksweep` | `python tests/canary_gripper_ksweep.py --run $C/ksweep --reference <reference records of stage 30> --episode-index 0` | about 45 min |

Figure 3 has no GPU stage: it reads the two packed stores of Figure 1, whose harvests are covered
by the target and distance canaries above.

Scripts that run the units of one figure in one job (reference locations from environment
variables, listed in each script's header; e.g. `C=<empty dir> REF_...=... sbatch --gpus=1
--time=01:30:00 tests/canary_position.sh all`):

| script | units | time |
|---|---|---|
| `tests/canary_position.sh [crossed\|hand_lines\|object_lines\|hand_shapes\|object_shapes\|all]` | Figure 2 | about 30 min |
| `tests/canary_contact.sh [natural\|offschedule\|all]` | Figure 4 | about 10 min |
| `tests/canary_phase.sh [scan\|natural\|freeze\|trajectory\|all]` | Figure 5 | about 35 min |
| `tests/canary_wrist_steer.sh [natural\|orient\|chart\|steer\|all]` | Figure 6 | about 60 min |
| `tests/canary_gripper_ksweep.sh` | Figure 7d | about 45 min |

Notes on single canaries:

- **Gripper.** Episode 0 has `new_sampler=1` and episode 14 has `new_sampler=0`, so the unit runs
  both ways of sampling a scene in one process (`vlageom.gripper.sample_episode`). The edit fits
  alone can be checked without a rollout from stored model inputs (written by a run with
  `--save-captures`): add `--from-captures <captures dir>` to the run command.
- **Wrist.** A second unit: `--row 1` (catalog_index 3, house 4) against the reference sweep of
  catalog_index 3. Row 0 is also the sweep of Figure 6a.
- **Target.** House 44 is the held-out house of the target tile's object path, so the two target
  canaries cover the same scene.
- **Distance.** `--shard 0` (houses 0, 5, 7, 11) also covers placements whose rollout ends early
  on success; compare with `--houses 0 5 7 11`.
- **Position.** Every unit is in house 44, the house of the trace grids. Larger units:
  `80_hand_shapes.py` with all its paths (about 15 min; compare each `ee_<kind>/h44`), a shard of
  stage 10 (`--shard 7`, houses 42-47). `canary_position_harvest.py` compares every key of a house
  file (the audit record as a dict) or, for a path, the activations, the manifest values and the
  exo frames; keys that only one side has are listed as notes.
- **Contact.** The natural canary requires the same keys, dtypes and per-query shapes, the same
  scene and instruction tokens, the rollout fields (query steps, query and step counts, contact
  series, executed gripper) bit for bit and every query within 1e-3 relative L2 per layer and token
  group. On other hardware the bfloat16 rollout can separate from the reference after some
  queries; `--lenient` (`LENIENT=1` for `canary_contact.sh`) then requires only the first query and
  reports the rest. The off-schedule canary requires every placement and pose field bit for bit and
  the activations within 1e-3 relative L2 per placement, layer and token group.
- **Phase.** Every array of a unit (all 41 of a natural episode file, 63 of a freeze-test file, 40
  of the capture) is compared with the reference file of the same name. `60_trajectory_capture.py`
  exits with status 2 when the rollout fails or misses a phase.
- **Wrist steering.** Second units: `--partition 12` (episodes 192-199; the first sampling attempt
  of the process finds no valid task) and `--task 1`. The natural canary compares the per-query
  fields and the cache rows of its episodes; the chart canary compares the weights and the
  training report; the steering canary compares every row of the json (the `frames` and
  `achieved_plane_deg` fields must be empty, as stage 60 writes them).
- **Gripper k-sweep.** The canary compares every scalar field, every metric of every k and of the
  768-D edit, and the rounded coefficients of both records with the reference records, which must
  be fitted from the same edit files and model inputs; for metrics that differ it prints the
  largest absolute and relative difference per k.
- A harvest stage that exits with status 3 found blank camera frames (an EGL failure of that
  process); run it again, on another node if it repeats.

## CPU comparisons

| figure | check | command |
|---|---|---|
| 1 | gripper panel | `python tests/compare_gripper_panel.py --reference <reference gripper panel .npz> [--panel $VLAGEOM_DATA/panels/gripper.npz]` |
| 1 | wrist panel | `python tests/compare_wrist_panel.py --reference <reference wrist panel .npz> [--panel ...] [--ignore files,src]` |
| 1 | target panel and probe score | `python tests/compare_target_panel.py --reference <reference target panel .npz> [--panel ...] [--ref-probe <reference probe .json>]` |
| 1 | distance panel and scores | `python tests/compare_distance_panel.py --reference <reference distance panel .npz> [--panel ...] [--ref-scores R2_M R2_CM]` |
| 1, 3 | packed teleport store | `python tests/canary_target_harvest.py pack --new $VLAGEOM_DATA/target/teleport --ref <reference teleport store root>` |
| 1, 3 | packed near-range store | `python tests/canary_distance_harvest.py store --new $VLAGEOM_DATA/distance/near/prep --ref <reference near-range store>/prep` |
| 1 | render | `python tests/compare_fig1_render.py --reference-png <reference .png> [--reference-pdf <reference .pdf>] [--panels DIR \| --rendered STEM]` (the PDF text comparison needs PyMuPDF: `pip install -e ".[test]"`) |
| 2 | packed crossed store | `python tests/canary_position_harvest.py pack --new $VLAGEOM_DATA/position/crossed --ref <reference crossed store root>` |
| 2 | probe results | `python tests/compare_position.py probe --new $VLAGEOM_DATA/position/probe --ref <reference probe dir>` |
| 2 | Procrustes parts or merged files | `python tests/compare_position.py procrustes --new $VLAGEOM_DATA/position/procrustes/parts --ref <reference parts dir>` |
| 2 | curvature result and its subspaces | `python tests/compare_position.py curvature --new $VLAGEOM_DATA/position/curvature/curvature_q0.json --ref <reference result> [--ref-charts <ee .npz> <obj .npz>]` |
| 2 | top-row panel | `python tests/compare_position.py top-panel --panel $VLAGEOM_DATA/panels/fig2_top.npz --ref-probe <ee json> <target json> --ref-procrustes <q0 json> --ref-curvature <json>` (the arrays built from the reference results) |
| 2 | trace-grid panels | `python tests/compare_position.py grid --panel <panel .npz> --ref-chart <reference subspace .npz> --shapes $VLAGEOM_DATA/position/shapes --variable ee\|obj --cell hdr_txt@L20\|txt_instr@L21 --atol 1e-9` (`--atol 1e-9` accepts the last-digit differences of the subspace fit) |
| 2 | renders | `python tests/compare_position.py render --new $VLAGEOM_FIGURES/fig2_top --ref <reference file without extension>` (and `fig2_ee_grid`, `fig2_target_grid`) |
| 3 | erasure (stage 50) | `python tests/compare_fig3.py erasure --new $VLAGEOM_DATA/distance/erasure --reference <reference erasure result>`; compares every scored k present in both, so a quick `50_erasure.py --k-max 2` run checks the first three points |
| 3 | probe bank (stage 60) | `python tests/compare_fig3.py bank --new $VLAGEOM_DATA/distance/probe_bank_last_tok_L21.npz --reference <reference bank .npz>`; or, without the full run, `python tests/compare_fig3.py bank-subset --reference <reference bank .npz> --folds 0 --draws 1` (recomputes fold 0 and the first ceiling draw at 16 threads) |
| 3 | cutoff table, synthetic cosines (stages 70, 80) | `python tests/compare_fig3.py table --new ... --reference ...`, `python tests/compare_fig3.py plant --new ... --reference ...` |
| 3 | panel files (stage 90) | `python tests/compare_fig3.py panels --ref-a <a.json> --ref-b <b.json> --ref-c <c.json> --ref-d <d.json> --ref-points <points .json>` (see the script's help for the layout of the reference records) |
| 3 | renders | `python tests/compare_fig3_render.py --ref-png a=<png> b=<png> c=<png> d=<png> --ref-pdf a=<pdf> b=<pdf> c=<pdf> d=<pdf> [--panels DIR \| --rendered DIR]` |
| 4 | natural cache, off-schedule pool, probe and held-out scores | `python tests/compare_contact.py cache\|pool\|probe\|scores --reference <reference file> [--new <file>]` |
| 4 | panel file | `python tests/compare_contact.py panel --ref-pca <reference PCA .npz> --ref-scores <reference scores .json> --ref-probe <reference probe .json>` |
| 4 | renders | `python tests/compare_fig4_render.py --ref-png a=<a.png> b=<b.png> --ref-pdf a=<a.pdf> b=<b.pdf>` |
| 5 | natural table | `python tests/compare_phase_arrays.py $VLAGEOM_DATA/phase/natural/cache_ae.npz <reference table>` |
| 5 | decoder scan, freeze-test decode | `python tests/compare_phase_results.py scan --new $VLAGEOM_DATA/phase/decoder.json --ref <reference scan>`; `... decode --new $VLAGEOM_DATA/phase/freeze/decode_ae.json --ref <reference fit>` |
| 5 | panel c and d numbers, landmark counts | `python tests/compare_phase_results.py slope\|gripper\|counts --new $VLAGEOM_DATA/phase/fig5_numbers.json --ref <reference summary>` |
| 5 | renders | `python tests/compare_fig5_render.py --ref-pdf a=<pdf> ... --ref-png a=<png> ... [--images-in strip=<paper pdf>:<page> b=<paper pdf>:<page>]` (`--images-in` compares the embedded images with those of a compiled page; needs PyMuPDF) |
| 6 | natural cache and cloud | `python tests/compare_wrist_steer.py natural --new-cache ... --ref-cache ... --new-natural ... --ref-natural ...` |
| 6 | operator | `python tests/compare_wrist_steer.py operator --new $VLAGEOM_DATA/wrist_steer/operator_L9.npz --ref <reference operator> [--rtol R]` (held-out houses exactly; V, c, Lhat with their relative difference) |
| 6 | panel a metrics, numbers | `python tests/compare_wrist_steer.py metrics\|numbers --new ... --ref ... [--ref-key KEY]` |
| 6 | renders | `python tests/compare_wrist_steer.py render --new-dir $VLAGEOM_FIGURES --ref-a <a.pdf> --ref-bcd-pdf <bcd.pdf> --ref-bcd-png <bcd.png>` |
| 7 | basis, probes, summary | `python tests/compare_gripper_fig7.py --ref-basis <reference basis .npz> --ref-probe <JSON with r2_by_k> --ref-probe-full <JSON with r2_full> [--ref-summary <reference stage 40 summary>]` (the basis columns are compared up to their signs) |
| 7 | renders | `python tests/compare_fig7_render.py --ref-png a=<png> b=<png> c=<png> d=<png> --ref-pdf a=<pdf> b=<pdf> c=<pdf> d=<pdf> [--panels DIR \| --rendered DIR]` |

`--panel` defaults to `$VLAGEOM_DATA/panels/<file>.npz`. Keys that only record where the inputs
were read from (`files` and `src` in the wrist panel, `src` in the target, distance and Figure 3
panels) differ when the inputs were read from another location; the comparison scripts report
them as ignored or do not require them to match. PNG renders are compared byte for byte, PDFs
byte for byte apart from their /CreationDate entry.

## End-to-end check of the CPU stages

To check every CPU stage of a figure at once against reference stores, place the outputs of its
GPU stages under a fresh data root (copies or symbolic links), run the CPU stages with their
defaults, then run `tests/check_expected.py --fig <N>` and the comparisons above.

| figure | GPU outputs under `$VLAGEOM_DATA` | CPU stages |
|---|---|---|
| 1 | `gripper/deltas/`, `wrist/sweeps/ep<catalog_index>.npz`, `target/teleport/houses/`, `target/shape_square/h44/`, `distance/near/houses/` | gripper 20; wrist 02, 03; target 20, 40, 50; distance 20, 30, 40; `figures/fig1.py` |
| 2 | `position/crossed/houses/`, `position/hand_lines/`, `position/object_lines/shape_line_*/`, `position/shapes/{ee,obj}_<kind>/h44/` | position 20, 30, 40 (`--task 0..11`, then `--merge`), 60, 70, 90; `figures/fig2.py` |
| 3 | the packed stores of Figure 1 (`target/teleport/prep/`, `distance/near/prep/`) | distance 30, 50, 60, 70, 80, 90; `figures/fig3_distance.py` |
| 4 | `contact/natural/`, `contact/offschedule/` | contact 20, 40, 50, 60, 70; `figures/fig4_contact.py` |
| 5 | `phase/natural/episodes/`, `phase/freeze/episodes/`, `phase/trajectory/traj_ep249_h76.npz` | phase 20, 30, 50, 70; `figures/fig5.py` |
| 6 | `wrist/sweeps/ep0.npz`, `wrist_steer/natural/episodes/` (or a natural cache, `20_natural_cloud.py --from-cache`), `wrist_steer/orient/`, `wrist_steer/chart.pt` + `chart_report.json`, `wrist_steer/steer/{oneshot,persist}.json` | wrist_steer 20, 40, 70, 80; `figures/fig6.py` |
| 7 | `gripper/deltas/`, `gripper_fig7/ksweep/` | gripper 20; gripper_fig7 10, 20, 40, 50; `figures/fig7_gripper.py` |

For example, for Figure 1:

```bash
python pipelines/gripper/20_panel.py
python pipelines/wrist/02_check_sweeps.py && python pipelines/wrist/03_panel.py
python pipelines/target/20_pack_store.py && python pipelines/target/40_probe.py && python pipelines/target/50_panel.py
python pipelines/distance/20_pack_near.py && python pipelines/distance/30_ruler_probes.py && python pipelines/distance/40_panel.py
VLAGEOM_FONT_DIR=<font dir> python figures/fig1.py
VLAGEOM_FONT_DIR=<font dir> python tests/check_expected.py --fig 1
```
