# Figure 4: contact

Part of [vla-geometry](../../README.md) · [all pipelines](../README.md).

Figure 4 is a composite: the robot sketch with its labels ("Contact", "no contact" / "contact")
and the panel letters are drawn by hand, and two rendered data files are placed in it
(`$VLAGEOM_FIGURES/fig4{a,b}.{pdf,png}`):

| file | panel | what is drawn | drawn from |
|---|---|---|---|
| `fig4a` | a | natural VLM activations (mean over the text tokens, layer 24) of 6,000 of the 18,837 natural queries on their top three principal components, coloured by contact; floor labels PC1 (9.1%) and PC2 (7.6%) | `panels/contact.npz` (`P`, `contact`, `evr`, `show`) |
| `fig4b` | b | off-schedule contact detection: for early touches (hand open as it is / forced closed) and induced slips (hand closed as it is / forced open), the fraction of poses at which a probe fit on the natural queries of the other houses scores the contact placements above the no-contact placements ("P(correct)", binomial SE); dashed line: the probe's out-of-house accuracy on the natural queries (0.913, "baseline"); dotted line: chance | `panels/contact.npz` (`bar_cells`, `frac_pos`, `frac_pos_se`, `n_pose`, `baseline_acc`) |

**Contact** is read from the simulator state (`BranchEngine._read_grasp`): the label is 1 when a
contact between the pickup object and a geom whose name (or, for an unnamed geom, its body's
name) contains "gripper" has a positive constraint force. Every label comes from the rollout
whose activations are stored with it, read by restoring that rollout's per-step snapshot.

The stages are under `pipelines/contact/`, the library code in `vlageom/contact.py`, the render in
`figures/fig4_contact.py`.

## Data

**Natural queries** (panel a and the probes). The 299 episodes of
`lists/contact/natural_episodes.csv` (192 houses) are rolled out with MolmoBot-DROID
(`FrankaPickDroidDataGenConfig`, at most 600 steps, one model query every 8 steps). At every query
the VLM backbone is run again on the query's inputs with hooks on blocks 0, 4, 8, ..., 32, 35.
The policy reads two frames per camera: the current frame and the frame of the previous query,
8 steps earlier. Query 0 has no earlier frame, so its input has 2 image blocks (exo and wrist) and
every later query has 4. The harvest stores the outputs at the 392 patches of the current exo and
wrist frames, at the text tokens, and the mean over all tokens of the sequence; the patches of
the earlier frames are not stored. The pool used in the figure, `txt_all`, is the mean over the
text tokens that follow the images, without the chat and multimodal control tokens; `last_tok`
is the final token.

**Off-schedule placements** (panel b). For each of the 128 episodes of
`lists/contact/offschedule_episodes.csv` (one per house), the rollout is repeated and up to 4
queries with the hand open (early-touch anchors; the first and last query are excluded) and up to
4 queries where the hand touches the object (slip anchors) are taken. At each anchor the arm is
held at the query's joint angles and the object is moved along the hand-object line in 1 cm steps
until contact changes, then 3 steps further. Four events per anchor: early touch with the hand as
it is (open) or with the fingers set to their closed limit; slip with the hand as it is (closed)
or with the fingers set to their open limit. In the last event the object is moved out of the
opened hand when it still touches it (the "Open (forced)" bar, 24 poses); otherwise it is moved
back in until it touches again (360 poses; not drawn, reported in the scores). 123 of the 128
episodes give placements (17,975 in all): houses 30, 104 and 106 have no valid robot placement
(MolmoSpaces raises `HouseInvalidForTask`), and in episodes 231 and 525 no anchor crosses the
contact boundary.

**Model input at a placement (frame history).** Every placement starts from the rollout's
snapshot at the anchor query, which includes the policy's frame history, and the policy is never
reset. The arm, the fingers (forced events) and the object are then set, the robot-mounted
cameras are moved to the current hand pose, every camera is rendered, and the policy is queried
once with the anchor query's flow-noise seed. Its input is therefore the current frame pair, which
shows the moved object and the hand at the anchor's joint angles, plus the rollout's own natural
frame pair from the previous query, 8 steps earlier, from before the object was moved (4 image
blocks). Every anchor is at query 1 or later (in the paper's store, early-touch anchors at query 1
or later and slip anchors at query 3 or later), so the earlier frame pair always exists. Within
one pose the earlier frame pair and the joint state are the same at every placement, so the
placements compared within a pose differ only in the object's position, which the current frames
show. The stored activations are the same token groups as in the natural harvest.

**Scoring** (panel b). For each off-schedule house h, StandardScaler + linear discriminant analysis
(scikit-learn, svd solver) is fit on the natural queries of all houses except h, and its decision
score is read at every placement of house h. Per pose, the vote is correct when the mean score of
the contact placements exceeds the mean score of the no-contact placements (ties count as
incorrect); P(correct) is the fraction of correct votes over the poses with both classes, with SE
sqrt(p (1 - p) / n).

## Stages

| stage | what it does | compute | output (under `$VLAGEOM_DATA`) |
|---|---|---|---|
| `10_natural_harvest.py --task K` | episode K of `lists/contact/natural_episodes.csv` (K = 0..298): rollout with per-step snapshots; contact at every step; for every query, the VLM blocks 0, 4, ..., 32, 35 at the 392 current image patches, the text tokens and the sequence mean, float16 | GPU + EGL, 2-4 min per episode including model loading, about 13 H200 GPU-hours | `contact/natural/ep<E>_h<H>.npz` (299 files, about 1.2 GB each, 369 GB) |
| `20_natural_cache.py` | one row per query (18,837 rows): six token pools per layer (float32 mean of the stored float16 tokens, stored as float16) and the labels of the harvest's own rollout | CPU, 20-35 min with 4 workers; reads all 369 GB | `contact/natural_cache.npz` (5.8 GB) |
| `30_offschedule_harvest.py --task K` | episode K of `lists/contact/offschedule_episodes.csv` (K = 0..127): rollout, anchors, the four events per anchor, every placement rendered and encoded at VLM blocks 0, 4, ..., 32, 35 | GPU + EGL, about 5 min per episode, about 10.5 H200 GPU-hours | `contact/offschedule/ep<E>_h<H>.npz` (123 files, about 2.9 GB each, 352 GB) |
| `40_offschedule_pool.py` | layer-24 text-token mean and final token of every placement, with the placement and pose fields; reads only those rows of each file | CPU, about 1 min (about 1.4 GB read) | `contact/offschedule_L24.npz` (17,975 rows, 369 MB) |
| `50_natural_probe.py` | out-of-house LDA probes of contact on the natural queries over `lists/folds/contact_natural.csv` (txt_all and last_tok at layer 24): accuracy, AUC, balanced accuracy, majority-class accuracy | CPU, 4-7 min (1 BLAS thread) | `contact/natural_probe.json` |
| `60_heldout_scores.py` | per off-schedule house, the probe refit without that house; per cell, P(correct), within-pose AUC, pose-matched forced-minus-natural AUC | CPU, about 8 min with 12 workers, 20-60 min with 4 on a shared machine (1 BLAS thread each) | `contact/detection_heldout.json` |
| `70_panel.py` | panel a: covariance eigendecomposition of the natural txt_all rows at layer 24 (all 18,837 rows), each component signed so that contact projects higher; the 6,000 drawn rows (numpy `default_rng(0).permutation(n)[:6000]`); panel b: the four bars and the baseline | CPU, about 1 min (4 BLAS threads) | `panels/contact.npz` |
| `figures/fig4_contact.py` | renders the two panels (`--panel a b` selects) | CPU, 10-20 s | `fig4a.{pdf,png}` (11 x 9 in), `fig4b.{pdf,png}` (9.2 x 5.6 in, tight bounding box), 350 dpi |

```bash
source scripts/harvest_env.sh                                 # GPU stages: EGL, offline model cache
python pipelines/contact/10_natural_harvest.py --task K        # GPU, K = 0..298
python pipelines/contact/30_offschedule_harvest.py --task K    # GPU, K = 0..127
python pipelines/contact/20_natural_cache.py --workers 4
python pipelines/contact/40_offschedule_pool.py
python pipelines/contact/50_natural_probe.py
python pipelines/contact/60_heldout_scores.py --workers 12
python pipelines/contact/70_panel.py
python figures/fig4_contact.py
```

Both harvest stages skip episodes whose file exists and exit with status 3 when the first query's
exo-camera frame is dark (an EGL failure of that process); run such a task again, on another node
if it repeats. Stage 30 treats the listed `no_scene` and `no_crossing` episodes as expected and
exits with status 0 for them.

Notes:

- **Lists.** `lists/contact/natural_episodes.csv` holds episodes 0-299 of the natural rollouts
  that have at least 5 model queries (episode 194 has 4). Its last three columns describe the
  listed natural rollout of each episode (length, contact at each model query as one character
  per query, task description); the cache uses them only for its `agree` and `orig_n_steps`
  fields, and the harvest stores the description as `descr`. The harvest's own rollout samples a
  different task instruction than the listed one in 272 of the 299 episodes (stage 20 prints the
  count); all labels and activations come from the harvest's own rollout.
  `lists/contact/offschedule_episodes.csv` holds, for each house in increasing order, its
  successful natural rollout with the lowest episode index (houses 0-132 except 2, 44, 60, 61 and
  75, which have none); `outcome` is `shard`, `no_scene` or `no_crossing`. All 123 off-schedule
  houses have natural queries, so the placements of every house are scored by a probe fit
  without that house's natural queries.
- **Pooling order.** Both caches pool the stored float16 tokens with a float32 mean; the natural
  cache stores the pools as float16, the off-schedule pool as float32. Stage 40 reads the text rows
  of layer 24 of each file directly at their offsets in the uncompressed `act` member.
- **Harvest reruns.** The policy runs in bfloat16. Two runs of the natural harvest of episode 0
  gave different trajectories after some queries (contact in 55% and 67% of the steps), so a rerun
  of stage 10 can give a different set of rows; the labels always come from the rows' own
  rollout. In the off-schedule harvest every placement starts from a snapshot of the rollout, and
  two runs gave the same anchors, placements and contact labels in all 123 episodes.

## Numbers (Figure 4)

- Stage 50: txt_all at layer 24 accuracy 0.9127 (the dashed line, and the 91.3% quoted in the
  paper), AUC 0.954, balanced accuracy 0.898; last_tok accuracy 0.9135, AUC 0.957; majority class
  0.654 (contact in 34.6% of the rows).
- Stage 60 (txt_all): P(correct) 0.860 (279 poses), 0.853 (361), 0.919 (419) and 0.750 (24) for
  the four bars, 0.767 (360) for the forced-open slips moved back in; forced-closed minus natural
  early-touch AUC at the same anchor +0.056 ± 0.028 over 233 pairs.
- Stage 70: explained-variance ratios 0.0909, 0.0765, 0.0468 (the axis labels use the first two).
