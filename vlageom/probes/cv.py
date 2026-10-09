"""Out-of-house ridge probes: out-of-fold (OOF) predictions over a house-level fold map.

Two drivers, both standardizing on the training rows of each fold (`standardize`) and solving with
`RidgeEngine`:

  cv_ridge_select  penalty chosen per target and per fold on held-out training houses
                   (every 5th training house, in sorted house order, is the validation set);
                   the probe is then refit on all training rows at that penalty.
  oof_ridge        one given penalty for every target and fold.

Both return OOF predictions: row i is predicted by the probe fit on the folds that do not contain
row i's house. Score them with `vlageom.probes.ridge.r2` / `r2_finite` on all rows (pooled OOF R^2).
"""
from __future__ import annotations

import numpy as np

from .folds import FoldMap
from .ridge import ALPHAS_SELECT, RidgeEngine, r2, standardize


def _as_splits(folds, houses):
    if isinstance(folds, FoldMap):
        return folds.splits(houses)
    return list(folds)


def cv_ridge_select(X, targets, houses, folds, alphas=ALPHAS_SELECT):
    """OOF ridge predictions with the penalty selected on held-out training houses.

    X: (n, d) float32 activations. targets: {name: (n,) array}. houses: (n,) house label per row.
    folds: a FoldMap, or an explicit list of (train_idx, test_idx).
    Per fold: the training houses are split into fit houses and validation houses (every 5th of
    the sorted unique training houses); one engine fit on the fit rows scores every alpha on the
    validation rows and the alpha with the highest validation R^2 is kept (first in `alphas` on a
    tie); a second engine fit on all training rows predicts the test rows at that alpha.
    Returns (oof {name: (n,) float64}, chosen {name: [alpha per fold]})."""
    n = len(X)
    oof = {k: np.full(n, np.nan) for k in targets}
    chosen = {k: [] for k in targets}
    for tr, te in _as_splits(folds, houses):
        gtr = np.unique(houses[tr])
        val_g = set(gtr[::5].tolist())
        vmask = np.isin(houses[tr], list(val_g))
        ifit, ival = tr[~vmask], tr[vmask]
        Zfit, Zval, Zte = standardize(X[ifit], X[ival], X[te])
        eng_fit = RidgeEngine(Zfit)
        Ktest_val = Zval @ Zfit.T if eng_fit.dual else None
        best = {}
        for name, y in targets.items():
            yv = y[ival]
            sc = {a: eng_fit.scores(y[ifit], a, Ztest=Zval, Ktest=Ktest_val) for a in alphas}
            perf = {a: r2(yv.astype(np.float64), s) for a, s in sc.items()}
            best[name] = max(alphas, key=lambda a: (np.nan_to_num(perf[a], nan=-9)))
        Ztr, Zte2 = standardize(X[tr], X[te])
        eng_tr = RidgeEngine(Ztr)
        Ktest = Zte2 @ Ztr.T if eng_tr.dual else None
        for name, y in targets.items():
            oof[name][te] = eng_tr.scores(y[tr], best[name], Ztest=Zte2, Ktest=Ktest)
            chosen[name].append(float(best[name]))
    return oof, chosen


def oof_ridge(X, targets, houses, folds, alpha=100.0):
    """OOF ridge predictions at one penalty `alpha` for every target and fold.

    X: (n, d) float32 activations. targets: {name: (n,) array}. houses: (n,) house label per row.
    folds: a FoldMap, or an explicit list of (train_idx, test_idx).
    Returns {name: (n,) float64}."""
    oof = {k: np.full(len(X), np.nan) for k in targets}
    for tr, te in _as_splits(folds, houses):
        Zfit, Zte = standardize(X[tr], X[te])
        eng = RidgeEngine(Zfit)
        Kt = Zte @ Zfit.T if eng.dual else None
        for k, t in targets.items():
            oof[k][te] = eng.scores(t[tr], alpha, Ztest=Zte, Ktest=Kt)
    return oof
