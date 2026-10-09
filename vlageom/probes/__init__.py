"""Linear probes: standardized closed-form ridge, out-of-house cross-validation over the shipped
house -> fold maps, and the RidgeCV read-out basis fit.

    from vlageom.probes import FoldMap, cv_ridge_select, oof_ridge, r2_finite

    folds = FoldMap.load("target_position")            # lists/folds/target_position.csv
    oof, alphas = cv_ridge_select(X, {"fwd": y_fwd}, houses, folds)
    print(r2_finite(y_fwd, oof["fwd"]))                # pooled out-of-house R^2
"""
from .cv import cv_ridge_select, oof_ridge
from .folds import N_FOLDS, FoldMap, load_fold_map
from .ridge import (ALPHAS_GCV, ALPHAS_SELECT, RidgeEngine, mae, r2, r2_finite, ridgecv_fit,
                    standardize)

__all__ = [
    "ALPHAS_GCV", "ALPHAS_SELECT", "N_FOLDS", "FoldMap", "RidgeEngine", "cv_ridge_select",
    "load_fold_map", "mae", "oof_ridge", "r2", "r2_finite", "ridgecv_fit", "standardize",
]
