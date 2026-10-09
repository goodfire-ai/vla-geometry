"""Phase, stage 50 (CPU): the decoded phase of the freeze-test landmarks, natural and frozen.

The decoder is the four-class LDA of stage 30 at action-expert layer 24, one fit per fold of
lists/folds/phase.csv on the successful natural queries of the other folds' houses
(StandardScaler + LDA on float64 activations). The landmarks are those of the freeze-test files of
the episodes with decode=1 in lists/phase/freeze_episodes.csv, selected by
vlageom.phase.used_landmarks: every landmark except that, of an episode's closed-hand place
landmarks, only the last (the last query before the release) is used (122 episodes, 797 landmarks
in the paper). Each is decoded by the fold model that holds out its house, under three conditions:
entry (the policy's normal two frames; "natural"), pause2 (the current frame shown twice;
"frozen") and detour.

Output: --out (default <data>/phase/freeze/decode_ae.json):
  composition   per landmark kind (reach, grasp_open, grasp_closed, transport, place_closed) and
                condition: fraction of landmarks decoded as each phase, and n
  decoder       layer, natural rows and houses, out-of-fold balanced accuracy of the decoder on the
                natural rows (0.920), and the fold hash (c8b301a92149)
  n_files, n_rows, layer

Runs at 1 BLAS thread (--threads), 1-3 minutes.

  python pipelines/phase/50_freeze_decode.py [--freeze-dir DIR] [--cache FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vlageom import paths
from vlageom import phase as P
from vlageom.probes import FoldMap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze-dir", type=Path,
                    default=paths.tile_dir("phase", "freeze", "episodes"),
                    help="freeze-test files of stage 40 (default <data>/phase/freeze/episodes)")
    ap.add_argument("--cache", type=Path,
                    default=paths.tile_dir("phase", "natural", "cache_ae.npz"),
                    help="natural table of stage 20")
    ap.add_argument("--folds", default="phase",
                    help="fold map name or house,fold CSV (default phase)")
    ap.add_argument("--threads", type=int, default=1, help="BLAS threads (default 1)")
    ap.add_argument("--out", type=Path, default=paths.tile_dir("phase", "freeze", "decode_ae.json"))
    a = ap.parse_args(argv)

    listed = [r for r in P.read_freeze_episodes() if r["decode"]]
    files = sorted((a.freeze_dir / P.freeze_name(r["episode_index"], r["house"]) for r in listed),
                   key=str)
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f"{len(missing)} decode-set episodes have no freeze-test file: {missing}")
        return 1

    import scipy.linalg  # noqa: F401  (load scipy's BLAS so that the limit below covers it)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=a.threads):
        rows = P.load_freeze_rows(files, P.LAYER)
        cache = P.load_cache(a.cache, ["X", "label", "house", "success", "layers"])
        decoder, info = P.natural_decoder(cache, FoldMap.load(a.folds), P.LAYER)
        comp = P.decode_composition(rows, decoder)
    out = {"layer": P.LAYER, "n_files": rows["n_files"], "n_rows": rows["n_rows"],
           "decoder": info, "composition": comp}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{rows['n_files']} episodes, {rows['n_rows']} landmarks; decoder: balanced accuracy "
          f"{info['natural_balanced_acc']:.5f} on {info['n_natural_rows']} natural rows, "
          f"{info['n_houses']} houses, fold hash {info['fold_hash']}")
    for kind, c in comp.items():
        print(f"  {kind:13s} n {c['entry']['n']:3d}  natural " +
              " ".join(f"{p} {100 * c['entry'][p]:5.1f}" for p in P.NAMES) + "  | frozen " +
              " ".join(f"{p} {100 * c['pause2'][p]:5.1f}" for p in P.NAMES))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
