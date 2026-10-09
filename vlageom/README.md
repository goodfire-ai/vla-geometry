# vlageom: library code

Part of [vla-geometry](../README.md).

| module | contents |
|---|---|
| `paths.py` | data, list, checkpoint and font locations (environment variables) |
| `util.py` | atomic .npz writes, .npy member offsets, file hashes, BLAS thread limits |
| `sim/` | MolmoSpaces environment, MolmoBot-DROID policy, rollout engine, scene edits, rendering and activation readout |
| `steer/` | gripper edit optimizer (full and low-rank) and grasp-window query selection |
| `probes/` | standardized ridge probes, out-of-house cross-validation over shipped fold maps |
| `plot/` | shared plot style and colormaps |
| `gripper.py`, `wrist.py`, `target.py`, `distance.py` | library code of the Figure 1 tiles (`distance.py` also holds Figure 3) |
| `position.py` | Figure 2 |
| `contact.py` | Figure 4 |
| `phase.py` | Figure 5 |
| `wrist_steer.py` | Figure 6 |
| `gripper_fig7.py` | Figure 7 |

The stages that use each module are described in the README of their pipeline
([`pipelines/`](../pipelines/README.md)).
