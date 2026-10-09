"""vlageom: code for "A Phenomenology of Neural Geometry in Vision-Language-Action Models".

The package has these parts:

  vlageom.paths   filesystem locations (data root, lists, checkpoint, fonts), all set by environment
                  variables
  vlageom.sim     MolmoSpaces environment + MolmoBot-DROID policy wiring, the BranchEngine (rollout,
                  snapshot/restore, per-layer activation readout), scene edits (object and arm
                  placement, settling) and observation/encode helpers
  vlageom.steer   the gripper edit optimizer (a vector added to one action-expert layer on the final
                  flow step) and the branch-query selection around the grasp
  vlageom.probes, vlageom.plot
                  linear probes and the shared plot style
  vlageom.util    atomic .npz writes, .npy member offsets, file hashes, BLAS thread limits
  vlageom.gripper, vlageom.wrist, vlageom.target, vlageom.distance
                  the library code of each Figure 1 tile (harvest helpers, panel arrays, legend
                  text); vlageom.distance also holds the Figure 3 analyses
  vlageom.position (Figure 2), vlageom.contact (Figure 4), vlageom.phase (Figure 5),
  vlageom.wrist_steer (Figure 6), vlageom.gripper_fig7 (Figure 7)
                  the library code of the other figures

The per-figure modules import numpy only at import time.

Importing `vlageom` itself is cheap: no simulator, torch or plotting import happens until a
submodule is used.
"""

__version__ = "0.1.0"
