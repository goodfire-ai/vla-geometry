"""Closed-form ridge regression on standardized activations, and the scores used to report probes.

`RidgeEngine` does one eigendecomposition of the fit-row Gram (or covariance) matrix and reuses it
for every target and every ridge penalty, so a whole battery of targets costs one decomposition per
fold. It solves the primal problem when there are more rows than features and the dual (kernel)
problem otherwise.

`ridgecv_fit` is the scikit-learn path (StandardScaler + RidgeCV with efficient leave-one-out
selection of the penalty) used to fit a single read-out basis on all training rows.
"""
from __future__ import annotations

import numpy as np

# Penalty grid searched on held-out houses by `vlageom.probes.cv.cv_ridge_select`.
ALPHAS_SELECT = [10.0 ** k for k in range(0, 7)]
# Penalty grid of the RidgeCV fit in `ridgecv_fit`.
ALPHAS_GCV = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def standardize(Xtr, *rest):
    """Z-score `Xtr` with its own per-feature mean and std (+1e-8); apply the same transform to
    every array in `rest`. Returns a list [Ztr, *Zrest] in the input dtype."""
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-8
    return [(Xtr - mu) / sd] + [(X - mu) / sd for X in rest]


class RidgeEngine:
    """Closed-form ridge with one eigendecomposition reused across every (y, alpha).

    Zfit: (n, d) standardized fit rows. The dual (Gram) form is used when n < d. The target is
    mean-centered internally and its mean is added back to the scores; there is no feature
    intercept because Zfit is already centered."""

    def __init__(self, Zfit):
        self.n, self.d = Zfit.shape
        self.dual = self.n < self.d
        M = (Zfit @ Zfit.T if self.dual else Zfit.T @ Zfit).astype(np.float64)
        self.evals, self.Q = np.linalg.eigh(M)
        self.Zfit = Zfit

    def scores(self, y, alpha, Ztest=None, Ktest=None):
        """Predictions for target `y` (fit rows) at penalty `alpha` on Ztest.

        In the dual form `Ktest = Ztest @ Zfit.T` can be passed to reuse it across targets."""
        ym = y.mean()
        yc = (y - ym).astype(np.float64)
        if self.dual:
            c = self.Q @ ((self.Q.T @ yc) / (self.evals + alpha))
            K = Ktest if Ktest is not None else Ztest @ self.Zfit.T
            return K @ c + ym
        b = self.Zfit.T @ yc
        w = self.Q @ ((self.Q.T @ b) / (self.evals + alpha))
        return Ztest @ w + ym

    def weights(self, y, alpha):
        """Weight vector (d,) in standardized feature coordinates for target `y` at `alpha`."""
        ym = y.mean()
        yc = (y - ym).astype(np.float64)
        if self.dual:
            c = self.Q @ ((self.Q.T @ yc) / (self.evals + alpha))
            return self.Zfit.T @ c
        b = self.Zfit.T @ yc
        return self.Q @ ((self.Q.T @ b) / (self.evals + alpha))


def r2(y, p):
    """Coefficient of determination 1 - SS_res / SS_tot (NaN when y is constant)."""
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else np.nan


def r2_finite(y, p):
    """`r2` over the rows where the prediction is finite (y as float64)."""
    y = np.asarray(y, np.float64)
    ok = np.isfinite(p)
    return r2(y[ok], p[ok])


def mae(y, pred):
    """Mean absolute error over the rows where the prediction is finite."""
    ok = np.isfinite(pred)
    return float(np.mean(np.abs(np.asarray(y, np.float64)[ok] - pred[ok]))) if ok.any() else float("nan")


def ridgecv_fit(X, Y, alphas=ALPHAS_GCV):
    """Fit StandardScaler on X, then RidgeCV (efficient leave-one-out over `alphas`) from the scaled
    X to Y (n,) or (n, k). Returns (scaler, model); predict with model.predict(scaler.transform(X)).
    The selected penalty is model.alpha_, the weights model.coef_ (k, d) in scaled coordinates."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    rg = RidgeCV(alphas=alphas).fit(sc.transform(X), Y)
    return sc, rg
