"""Compare the Figure 7 analysis outputs with reference files.

Every comparison is optional; give the reference files to compare:

  --ref-basis NPZ     stage 10 basis (gripper_fig7/basis.npz): mu, evr, sing, mu_close, mu_open,
                      centered_rank, train_houses, heldout_houses must be identical
                      (np.array_equal, same dtype); V must be identical up to the sign of each
                      column (an SVD determines a column only up to its sign)
  --ref-probe JSON    a JSON with "r2_by_k" (stage 20 probe.json or an equivalent): every k
                      identical
  --ref-probe-full JSON
                      a JSON with "r2_full" (and optionally "pearson_full"): identical
  --ref-summary JSON  a stage 40 summary of another run: identical bytes, else the first
                      differing entries

The values printed are the ones the figure shows (PC1 share and stable rank, R^2 at k = 1 and of
the full activation, leakage relative to the 768-D edit at k = 1, 16, 768).

  python tests/compare_gripper_fig7.py --ref-basis <npz> --ref-probe <json> \
      --ref-probe-full <json> [--ref-summary <json>]

Exit status 0 when everything compared is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import gripper_fig7 as F
from vlageom import paths

BASIS_KEYS = ("mu", "evr", "sing", "mu_close", "mu_open", "centered_rank", "train_houses",
              "heldout_houses")


def _same(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)


def _diff(x, y) -> str:
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return f"shape {x.shape} vs {y.shape}"
    if x.dtype != y.dtype:
        extra = (f", values {'equal' if np.array_equal(x, y) else 'differ'}"
                 if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number) else "")
        return f"dtype {x.dtype} vs {y.dtype}{extra}"
    return f"max|diff| {np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))):.3e}"


def compare_basis(path: Path, ref: Path) -> bool:
    a, b = np.load(path), np.load(ref, allow_pickle=True)
    ok = True
    for k in BASIS_KEYS:
        same = _same(a[k], b[k])
        ok &= same
        print(f"  basis {k:15s} {'equal' if same else 'DIFF  ' + _diff(a[k], b[k])}")
    V, Vr = a["V"], b["V"]
    if V.shape == Vr.shape and V.dtype == Vr.dtype:
        sign = np.where(np.sum(V.astype(np.float64) * Vr.astype(np.float64), 0) < 0, -1, 1)
        same = np.array_equal(V * sign.astype(V.dtype), Vr)
        print(f"  basis V               {'equal' if same else 'DIFF'} up to column signs "
              f"({int((sign < 0).sum())} of {V.shape[1]} columns of opposite sign)")
    else:
        same = False
        print(f"  basis V               DIFF {_diff(V, Vr)}")
    ok &= same
    print(f"  shown: PC1 {float(a['evr'][0]):.1%}, stable rank {F.stable_rank(a['sing']):.2f} "
          f"(reference {float(b['evr'][0]):.1%}, {F.stable_rank(b['sing']):.2f})")
    return ok


def compare_probe(probe: dict, ref: dict) -> bool:
    ok = True
    for k, v in ref["r2_by_k"].items():
        mine = probe["r2_by_k"].get(str(k))
        same = mine == v
        ok &= same
        print(f"  probe R^2 k={k:>4}  {mine!r}  {'equal' if same else f'DIFF (reference {v!r})'}")
    return ok


def compare_probe_full(probe: dict, ref: dict) -> bool:
    ok = True
    for k in ("r2_full", "pearson_full"):
        if k in ref:
            same = probe.get(k) == ref[k]
            ok &= same
            tail = "equal" if same else f"DIFF (reference {ref[k]!r})"
            print(f"  probe {k}  {probe.get(k)!r}  {tail}")
    return ok


def _first_diffs(a, b, path="", out=None, limit=20):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'reference' if k not in a else 'run'}")
            else:
                _first_diffs(a[k], b[k], f"{path}/{k}", out, limit)
    elif a != b and not (isinstance(a, float) and isinstance(b, float)
                         and np.isnan(a) and np.isnan(b)):
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def compare_summary(path: Path, ref: Path) -> bool:
    if path.read_bytes() == ref.read_bytes():
        print("  summary identical bytes")
        return True
    diffs = _first_diffs(json.loads(path.read_text()), json.loads(ref.read_text()))
    print(f"  summary DIFFERENT ({len(diffs)}{'+' if len(diffs) >= 20 else ''} entries):")
    for d in diffs:
        print(f"    {d}")
    return not diffs


def shown(panel: dict) -> str:
    out = [f"a: PC1 {panel['scree_evr'][0]:.1%}, stable rank {float(panel['stable_rank']):.2f}",
           f"b: R^2(k=1) {panel['probe_r2'][0]:.4f}, full {float(panel['probe_r2_full']):.4f}"]
    ks = list(panel.get("leak_k", []))
    for tag in ("close", "open") if ks else ():
        rel = [100.0 * panel[f"leak_{tag}"][ks.index(k)] / panel[f"free_leak_{tag}"]
               for k in (1, 16, 768) if k in ks]
        out.append(f"d: {tag} leak / 768-D leak at k = 1, 16, 768: "
                   + ", ".join(f"{v:.1f}%" for v in rel))
    return "\n  ".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    d = paths.tile_dir(F.TILE)
    ap.add_argument("--basis", type=Path, default=d / "basis.npz")
    ap.add_argument("--probe", type=Path, default=d / "probe.json")
    ap.add_argument("--summary", type=Path, default=d / "ksweep_summary.json")
    ap.add_argument("--panel", type=Path, default=paths.panels_dir() / "gripper_fig7.npz")
    ap.add_argument("--ref-basis", type=Path, default=None)
    ap.add_argument("--ref-probe", type=Path, default=None)
    ap.add_argument("--ref-probe-full", type=Path, default=None)
    ap.add_argument("--ref-summary", type=Path, default=None)
    a = ap.parse_args(argv)

    ok, n = True, 0
    if a.ref_basis:
        print(f"basis   {a.basis}\n        reference {a.ref_basis}")
        ok &= compare_basis(a.basis, a.ref_basis)
        n += 1
    if a.ref_probe or a.ref_probe_full:
        probe = json.loads(a.probe.read_text())
        print(f"probe   {a.probe}")
        if a.ref_probe:
            ok &= compare_probe(probe, json.loads(a.ref_probe.read_text()))
        if a.ref_probe_full:
            ok &= compare_probe_full(probe, json.loads(a.ref_probe_full.read_text()))
        n += 1
    if a.ref_summary:
        print(f"summary {a.summary}\n        reference {a.ref_summary}")
        ok &= compare_summary(a.summary, a.ref_summary)
        n += 1
    panel = dict(np.load(a.panel)) if a.panel.exists() else None
    if panel is not None:
        print("  shown (panel file):\n  " + shown(panel))
    if n == 0:
        print("nothing compared (give at least one --ref-* file)")
        return 1
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
