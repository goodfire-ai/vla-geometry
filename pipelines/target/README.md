# Figure 1: target tile

Part of [vla-geometry](../../README.md) · [all pipelines](../README.md).

The target tile shows the VLM's last-token activation at layer 21 for the first policy query of
13,145 object placements in 906 houses, projected onto a 3-D ridge read-out of the object position
and coloured by that position, with the path traced when the object is moved around a square on
its table in one held-out house (house 44). The printed R^2 is the mean over x, y and z of
out-of-house ridge probes of the object position.

| stage | what it does | compute | output (under `$VLAGEOM_DATA`) |
|---|---|---|---|
| `10_teleport_harvest.py` | per house of `lists/teleport_houses.csv`: 16 object placements on the support surface (the natural one + 15 sampled inside the reach envelope), 4 policy queries each, VLM (11 token groups x 13 layers) and action-expert (16 tokens x 11 layers) activations per query | GPU + EGL, about 2.6 min per house, about 34 H200 GPU-hours (114 jobs of 8 houses) | `target/teleport/houses/h<house>.npz` (43 GB), `target/teleport/frames/h<house>.npz` |
| `20_pack_store.py` | packs the house files into float16 memory-mappable arrays + labels | CPU, about 2 min with 16 processes | `target/teleport/prep/` (50 GB) |
| `30_square_path_harvest.py` | house 44: the object is moved to 64 waypoints on a square on its support surface; first-query activations at each | GPU + EGL, about 2 min plus model loading | `target/shape_square/h44/` |
| `40_probe.py` | out-of-house ridge probes of object x, y, z from the last-token activation at layer 21 on the first-query rows, folds from `lists/folds/target_position.csv` | CPU, 1-10 min | `target/probe/last_tok_L21.json` |
| `50_panel.py` | read-out basis (house 44 left out), cloud, smoothed path, printed R^2 | CPU, 0.5-5 min | `panels/target.npz` |

```bash
source scripts/harvest_env.sh                             # GPU stages: EGL, offline model cache
python pipelines/target/10_teleport_harvest.py --shard K  # GPU, K = 0..113, 8 houses each (default $SLURM_ARRAY_TASK_ID)
python pipelines/target/30_square_path_harvest.py         # GPU, house 44, one job
python pipelines/target/20_pack_store.py --workers 16
python pipelines/target/40_probe.py
python pipelines/target/50_panel.py
```

`figures/fig1.py` draws Figure 1 from the four tile panels (see
[`figures/README.md`](../../figures/README.md#figure-1)).

The object placements are drawn by `vlageom.sim.sample_targets` (numpy seed 7919 * house + 37);
the crossed harvest of Figure 2 uses the same sampler. The policy is reset before the first query
of every placement, so at that query (the rows of the tile) the model sees only the current frame
pair; from the second query on, its input also holds the frame pair of 8 steps earlier, from the
same placement. A harvest job that finds a nearly black first-query frame exits with status 3;
run it again, and houses that are already written are skipped. `--dry-run` prints the houses a job
would run. `40_probe.py` prints the per-axis R^2 and their mean (0.77 in the paper). `40_probe.py`
and `50_panel.py` use 24 BLAS threads (`--blas-threads`); on a machine with fewer cores they take
longer (the upper ends of the times above).

The [distance tile and Figure 3](../distance/README.md) read the same object-teleport store
(`$VLAGEOM_DATA/target/teleport/prep`), so stages 10 and 20 must run before the distance probes.
