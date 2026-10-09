# Figure 2: hand and target-object position

Part of [vla-geometry](../../README.md) · [all pipelines](../README.md).

Figure 2 is a composite: the panel letters a-e, the row titles ("EE Position", "Target Object
Position") and the "World" / "Model" labels are added in the layout, and three rendered data files
are placed in it.

| file | rows / panels | what is drawn |
|---|---|---|
| `fig2_top.pdf` (and `.png`) | top row, a-e, with the legend above them | a: held-out error per axis and 3-D (cm) of the EE (VLM layer 20) and target object (VLM layer 21) read-outs, with fold std and the gripper width (8.7 cm); b: held-out R^2 at read-out rank 1, 2, 3; c: singular-value fractions of the read-out; d: relative residual of the map from world to activation displacements vs layer (solid: general linear, dashed: orthogonal Procrustes; line at 1.0: no linear relation); e: curvature of straight world paths inside the subspace vs distance to the gripper, and over random 3-D projections |
| `fig2_ee_grid.pdf` (and `.png`) | middle row (2 x 5, no text) | five hand paths in house 44 (figure-8, x line, y line, circle, square): the realized path in the robot base frame (top) and the same waypoints' activations in the EE subspace fit without house 44 (bottom) |
| `fig2_target_grid.pdf` (and `.png`) | bottom row (2 x 5, no text) | five object paths on the table in house 44 (figure-8, x line, y line, ellipse, square) and their activations in the target subspace fit without house 44 |

The stages are under `pipelines/position/`, the library code in `vlageom/position.py`, the render
in `figures/fig2.py`.

## Data: the crossed arm x object store

Both variables are read from the same rows. In each of the 947 benchmark houses
(`lists/position/crossed_houses.csv`: the first MolmoSpaces benchmark episode of every house,
FrankaPickHardBench preferred, the same catalog as the object-teleport harvest of the Figure 1
[target tile](../target/README.md)) the arm is moved by inverse kinematics to up to 10 poses and the
pickup object to up to 5 placements, and every (placement, pose) cell is captured twice: at q0,
right after the move, and at q1, after the policy has executed one action chunk (8 steps) from
there. The policy is reset after each move, so at q0 the model sees only the current frame pair (2
image blocks); at q1 it sees the frames of q1 and q0, both taken after the move. Panels a-c use the
q1 rows; panel d, the curvature subspaces and the trace grids use the q0 rows.

- 945 houses and 90,814 rows (45,407 per query). Two harvested houses (499 and 567) are left out
  when the store is packed: their exo-camera frame does not change across rows (the exo camera
  sees neither the arm nor the object there).
- Object placements per house: 5 in 888 houses, 4 in 25, 3 in 13, 2 in 3, 1 in 16, so 57 houses
  have fewer than five (a placement is dropped when the object does not stay where it was put, or
  no arm pose leaves a valid cell).
- Arm poses per house: 10 in 943 houses, 9 in 2. Cells whose posed hand is within 12 cm of the
  object, whose arm penetrates anything, or whose exo frame is dark are skipped; 582 houses have
  all 50 cells.
- Poses: EE targets drawn uniformly 20-60 cm forward, up to 30 cm to either side, 5-40 cm above the
  object's resting height (at most 62 cm from the base axis, at least 5 cm apart), reached within
  1.5 cm by damped-least-squares inverse kinematics from the home pose within the FR3 joint limits.
  Placements: the first 5 placements of the object-teleport harvest's sampler in the same house
  (`vlageom.sim.sample_targets`, same seed), so they coincide with that harvest's placements 0-4.
- Read-out cells: the EE from `hdr_txt` at layer 20 (the mean over the non-image tokens before the
  instruction: the "Image N View M" labels, the image start and end markers and the BOS token), the
  target from `txt_instr` at layer 21 (the mean over the instruction tokens). Among the VLM token
  groups at layers 20-21 these give the highest held-out R^2 for their variable;
  `30_probe.py --cells STREAM:LAYER ...` scores any other cell (pack its token group first).

## Stages

| stage | what it does | compute | output (under `$VLAGEOM_DATA`) |
|---|---|---|---|
| `10_crossed_harvest.py --shard K` | per house of `lists/position/crossed_houses.csv` (K = 0..157, 6 houses each): 5 placements x 10 arm poses x 2 queries; VLM 11 token groups x 36 layers, action expert per token x 36 blocks (float16), chunk, labels, exo thumbnails | GPU + EGL, about 2.5 min per house, about 45 H200 GPU-hours | `position/crossed/houses/h<house>.npz` (947 files, about 300 MB each, 253 GB) |
| `20_pack_store.py` | packs the house files into memory-mappable arrays and labels; leaves out the two houses with an unchanging exo frame; by default only the two token groups Figure 2 reads (`--streams all` for every array) | CPU, 1-7 min (I/O bound; 8 processes) | `position/crossed/prep/` (33 GB; 252 GB with `--streams all`) |
| `30_probe.py` | per cell and query: out-of-house RidgeCV read-out of the EE and object positions (5 folds over houses from `lists/folds/position.csv`), rank 1/2/3 R^2, singular values, per-axis and 3-D error, within-house R^2, label baselines | CPU, about 15-35 min for both cells (8 BLAS threads; 15 min of fits plus the first read of the 33 GB store) | `position/probe/{hdr_txt_L20,txt_instr_L21}.json` |
| `40_procrustes.py --task K` | per layer (K = 0..11, layers K, K+12, K+24) and query: subspace fit on the training folds, world -> activation displacement maps on pairs within a house (EE: two poses, same placement; target: two placements, same pose), median relative residual on the test houses; then `--merge` | CPU, about 3 min per layer and query (8 BLAS threads), about 20 min per task, 4 hours for all 12 tasks | `position/procrustes/parts/`, `procrustes_{q0,q1}.json` |
| `50_hand_lines.py --shard K` | per house of `lists/position/line_houses.csv` (K = 0..7): 4 straight hand lines x 24 waypoints, one query each, 36-layer readout | GPU + EGL, about 1.4 H200 GPU-hours | `position/hand_lines/h<house>.npz` (48 files, 8.7 GB) |
| `55_object_lines.py --shard K` | the same houses (K = 0..7): the object along 3 straight lines on its table (forward, sideways, toward the hand), 24 waypoints, one query each, the object-teleport harvest's 13-layer readout | GPU + EGL, about 1.6 H200 GPU-hours | `position/object_lines/shape_{line_fwd,line_lat,line_ee}/h<house>/` (120 lines written, 43 houses) |
| `60_curvature.py` | subspaces fit on the q0 rows without the path houses; per path the local turning angle (about 7 cm segments) in the subspace, in the world and over 8 random 3-D projections, binned by hand-object distance | CPU, about 1 min | `position/curvature/curvature_q0.json`, `chart_{ee,obj}.npz` |
| `70_top_panel.py` | collects the drawn values of panels a-e | CPU, seconds | `panels/fig2_top.npz` |
| `80_hand_shapes.py` | house 44: the hand along a figure-8, x, y and z lines, a circle and a square (6 paths, 24-64 waypoints) | GPU + EGL, about 15 min | `position/shapes/ee_<kind>/h44/` |
| `85_object_shapes.py` | house 44: the object along a figure-8, x and y lines, an ellipse and a square on its table | GPU + EGL, about 7 min | `position/shapes/obj_<kind>/h44/` |
| `90_grid_panels.py` | subspaces fit on the q0 rows without house 44; the paths projected, rotated to the decoded forward/left/up axes, normalized; colours | CPU, about 1 min | `panels/fig2_ee_grid.npz`, `fig2_target_grid.npz`, `position/grids/chart_{ee,obj}.{npz,json}` |
| `figures/fig2.py` | renders the three files (`--panel top ee_grid target_grid` selects) | CPU, about 10 s | `$VLAGEOM_FIGURES/fig2_{top,ee_grid,target_grid}.{pdf,png}` |

```bash
source scripts/harvest_env.sh                                 # GPU stages: EGL, offline model cache
python pipelines/position/10_crossed_harvest.py --shard K       # GPU, K = 0..157
python pipelines/position/20_pack_store.py
python pipelines/position/30_probe.py
python pipelines/position/40_procrustes.py --task K             # K = 0..11 (CPU)
python pipelines/position/40_procrustes.py --merge
python pipelines/position/50_hand_lines.py --shard K            # GPU, K = 0..7
python pipelines/position/55_object_lines.py --shard K          # GPU, K = 0..7
python pipelines/position/60_curvature.py
python pipelines/position/70_top_panel.py
python pipelines/position/80_hand_shapes.py                     # GPU, one job
python pipelines/position/85_object_shapes.py                   # GPU, one job
python pipelines/position/90_grid_panels.py
python figures/fig2.py
```

Notes on the stages:

- **Harvest restarts.** Every harvest skips houses (or paths) whose output exists. Stage 10 exits
  with status 3 and writes nothing for the house when at least half of 10 or more cells of a house
  have a dark exo frame (an EGL failure of that process). Stages 80 and 85 do not write a path
  whose exo frames do not change from waypoint to waypoint; run them again on another node.
- **Rendering after a move.** Every observation after an arm or object move is rendered with the
  robot-mounted cameras moved to the robot's current pose (`vlageom.sim.fresh_obs`).
- **Times.** The CPU times are for a packed store on a local disk with free page cache; reading
  one layer of a token group (N x 2,560 values) from a network filesystem can take several minutes.
- **Hand paths.** Stages 50 and 80 walk each path by inverse kinematics, every waypoint starting
  from the previous one's joints (closed paths first walk two laps without recording) and
  iterating to 2 mm; a waypoint is kept within 1.5 cm. A closed path of stage 80 whose low region
  lies over the object is raised by 6 cm (in house 44: the circle and the square). The realized
  hand positions are within 0.16 cm of the targets in house 44.
- **Object lines.** Stage 55 writes a line only when at least 60% of its waypoints are free of
  contacts with other objects and at least half are kept after settling: 120 of the 144 lines
  (38 forward, 42 sideways, 40 toward the hand) in 43 houses.
- **Folds.** `lists/folds/position.csv` assigns each of the 945 houses to one of 5 folds; it is the
  assignment of scikit-learn `GroupKFold(5)` over house on the q1 rows (the same on the q0 rows)
  on an x86-64 CPU with AVX-512. Stages 30 and 40 read it instead of computing folds at run time.
- **Panel d's reference.** The shuffled-pair residual (about 1.00) comes from one numpy generator
  (seed 52) per task of stage 40, drawn layer by layer with q1 before q0; its last digits depend on
  how the layers are split into tasks (the paper: 12 tasks of 3 interleaved layers). It is not
  drawn; the drawn residuals do not depend on the split.
- **Colours.** The EE grid colours a hand position by its per-axis rank among 4,000 reference hand
  positions (`lists/position/ee_color_reference.csv`: hand positions at the model queries of 299
  natural MolmoBot-DROID rollouts in houses 0-199, 9,800 queries in 192 houses, a numpy
  `default_rng(0)` subsample), rounded to 3 decimals; the target grid by its rank among the object
  positions of the subspace's fit rows.
- **Gripper width.** The dashed line of panel a is at 8.7 cm (`vlageom.position.GRIPPER_CM`), the
  maximum finger opening of the Robotiq 2F-85 on the simulated `franka_droid` robot.

## Numbers

Printed by `70_top_panel.py` (q1 rows, 45,407 rows, 945 houses):

| | EE (hdr_txt, layer 20) | target (txt_instr, layer 21) |
|---|---|---|
| held-out R^2 (text: 0.92 / 0.75) | 0.9176 | 0.7483 |
| R^2 at rank 1 / 2 / 3 (panel b) | 0.40 / 0.76 / 0.92 | 0.35 / 0.64 / 0.75 |
| singular-value fractions (panel c) | 0.39 / 0.37 / 0.24 | 0.41 / 0.36 / 0.23 |
| error x / y / z / 3-D, cm (panel a) | 2.7 / 4.0 / 2.7 / 6.3 | 4.4 / 6.8 / 4.2 / 10.5 |
| within-house R^2 | 0.924 | 0.575 |
| R^2 of x, y from the other variable's true position | 0.015 / 0.010 | 0.011 / 0.010 |
| panel d at the read-out layer (q0): general linear / orthogonal / shuffled | 0.223 / 0.322 / 1.00 | 0.455 / 0.501 / 1.00 |
| panel e: curvature in the subspace / random projections (deg) | 31-35 / 82-85 | 34-42 / 61-68 |

The largest per-axis error (6.8 cm) is below the gripper width (8.7 cm). Trace-grid subspaces (q0
rows, 944 houses, 45,357 rows, penalty 100): in-sample R^2 x/y/z 0.957/0.954/0.986 (EE) and
0.817/0.864/0.942 (target); mean absolute error on house 44: 2.1/3.0/1.7 cm (EE) and
3.5/4.7/1.7 cm (target).

## Reproducibility (Figure 2)

- The fits use 8 BLAS threads (`--blas-threads`), the count of the paper's runs. Stage 90's two
  subspace fits (float64) on an x86-64 CPU with AVX-512 match the paper's fits up to their last
  digits (basis vectors within 1.3e-12, the 95th-percentile radius within 7e-10, also with 1, 2 or
  4 threads), which moves the projected paths by at most 4e-10 of the panel scale; the scaler,
  labels and colours are identical, and the rendered grids have the paper's bytes. Every other
  array of the CPU stages is identical to the paper's.
- Stage 40 re-run in full takes about 4 hours on 8 cores; its merge step (`--merge`) combines any
  set of per-layer parts.
- The probe JSONs of stage 30 hold the values listed in its docstring; `tests/compare_position.py
  probe` lists the keys of a reference file that stage 30 does not write and does not compare
  them.
- Harvests: the GPU canaries of Figure 2 are in house 44.
