"""Contact (Figure 4), stage 70 (CPU): the arrays of Figure 4 (panels a and b).

Panel a: the text-token mean at layer 24 of every natural row (natural cache, stage 20) as float32,
centred; the top three eigenvectors of its covariance (numpy eigh), each signed so that the contact
rows project higher on average than the no-contact rows (vlageom.contact.pca3). Stored: P (the
18,837 rows on PC1-3), contact, evr (explained-variance ratios; the axis labels "PC1 (9.1%)",
"PC2 (7.6%)"), origin_p (the zero vector on PC1-3) and show (the 6,000 drawn rows,
numpy default_rng(0).permutation(n)[:6000]).

Panel b: from the held-out scores (stage 60), the txt_all cells early_touch_open,
early_touch_closed, slip_closed, slip_open_out: bar_cells, frac_pos, frac_pos_se, n_pose; and
baseline_acc, the out-of-house accuracy of the txt_all probe at layer 24 on the natural rows
(stage 50).

BLAS runs on 4 threads (--threads) for the covariance and the eigendecomposition.

Output: $VLAGEOM_DATA/panels/contact.npz (--out), keys vlageom.contact.PANEL_KEYS.

  python pipelines/contact/70_panel.py [--cache FILE] [--scores FILE] [--probe FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import contact as C
from vlageom import paths
from vlageom.util import save_npz


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--cache", type=Path, default=None,
                    help="natural cache (default $VLAGEOM_DATA/contact/natural_cache.npz)")
    ap.add_argument("--scores", type=Path, default=None,
                    help="held-out scores (default $VLAGEOM_DATA/contact/detection_heldout.json)")
    ap.add_argument("--probe", type=Path, default=None,
                    help="natural probe scores (default $VLAGEOM_DATA/contact/natural_probe.json)")
    ap.add_argument("--threads", type=int, default=4, help="BLAS threads (default 4)")
    ap.add_argument("--out", type=Path, default=None,
                    help="panel file (default $VLAGEOM_DATA/panels/contact.npz)")
    a = ap.parse_args(argv)
    from threadpoolctl import threadpool_limits

    cache = a.cache or paths.tile_dir(C.TILE) / "natural_cache.npz"
    scores = json.loads(Path(a.scores or paths.tile_dir(C.TILE) / "detection_heldout.json").read_text())
    probe = json.loads(Path(a.probe or paths.tile_dir(C.TILE) / "natural_probe.json").read_text())
    out = a.out or paths.panels_dir(create=True) / "contact.npz"

    X = C.load_layer(cache, "X_txt_all", C.PROBE_LAYER)
    with np.load(cache, mmap_mode="r") as z:
        contact = np.asarray(z["contact"])
    with threadpool_limits(limits=a.threads):
        P, evr, origin_p = C.pca3(X, contact)
    show = C.show_index(len(P))
    cells, frac, se, n_pose = C.bar_arrays(scores["variants"], "txt_all")
    baseline = float(probe["cells"][f"txt_all@{C.PROBE_LAYER}"]["acc"])
    save_npz(out, dict(P=P, contact=contact, evr=evr, origin_p=origin_p, show=show, bar_cells=cells,
                       frac_pos=frac, frac_pos_se=se, n_pose=n_pose,
                       baseline_acc=np.float64(baseline)))
    print(f"panel a: {len(P)} rows, contact on {contact.mean():.3f}, evr {np.round(evr, 4)}; "
          f"{len(show)} drawn")
    print("panel b: " + ", ".join(f"{c} {f:.3f} +- {s:.3f} (n {n})"
                                  for c, f, s, n in zip(cells, frac, se, n_pose))
          + f"; baseline {baseline:.4f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
