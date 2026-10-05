import logging

import numpy as np
from sklearn.model_selection import KFold

from fixed_broadband_tract_reliability_model.modeling.utils.data_io import (
    load_modeling_dataset,
)
from fixed_broadband_tract_reliability_model.modeling.utils.preprocessing import (
    split_train_test,
)
from src.fixed_broadband_tract_reliability_model.modeling.gam_interpretability import (
    build_gam_interpretability_artifacts,
)
from src.fixed_broadband_tract_reliability_model.modeling.models_tier2 import build_gam
from src.fixed_broadband_tract_reliability_model.modeling.utils.evaluation import (
    evaluate_model,
)

logger = logging.getLogger(__name__)


def train_tier2_gam():
    """
    Registry‑compatible Tier 2 (GAM) training module.

    Instead of training the GAM directly, this function returns:
      - train_fn: trains and returns a GAM model
      - metrics_fn: evaluates the GAM model
      - interpretability_fn: builds GAM interpretability artifacts
      - data: X_train, X_test, y_train, y_test, feature_names

    The model registry will decide whether to:
      - load a saved GAM model, OR
      - call train_fn() to train a new one
    """

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    logger.info("Loading modeling dataset for Tier 2...")
    X, y, feature_cols = load_modeling_dataset(
        "data/processed/modeling/final_feature_set.parquet"
    )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------
    logger.info("Splitting train/test for Tier 2...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # ---------------------------------------------------------
    # Define train function
    # ---------------------------------------------------------
    def train_fn():
        logger.info("Training Generalized Additive Model...")
        gam = build_gam(n_features=X.shape[1])
        gam.fit(X_train, y_train)
        logger.info("GAM training complete.")
        return gam

    # ---------------------------------------------------------
    # Define metrics function
    # ---------------------------------------------------------
    def metrics_fn(model):
        logger.info("Evaluating GAM model...")

        # Standard evaluation
        eval_metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        # Manual CV (pyGAM does not integrate with sklearn CV)
        logger.info("Running manual CV for GAM...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        cv_mse, cv_mae, cv_r2 = [], [], []

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            gam_cv = build_gam(n_features=X.shape[1])
            gam_cv.fit(X_tr, y_tr)
            preds = gam_cv.predict(X_te)

            cv_mse.append(mean_squared_error(y_te, preds))
            cv_mae.append(mean_absolute_error(y_te, preds))
            cv_r2.append(r2_score(y_te, preds))

        cv_results = {
            "cv_mse_mean": float(np.mean(cv_mse)),
            "cv_mae_mean": float(np.mean(cv_mae)),
            "cv_r2_mean": float(np.mean(cv_r2)),
        }

        return {**eval_metrics, **cv_results}

    # ---------------------------------------------------------
    # Define interpretability function
    # ---------------------------------------------------------
    def interpretability_fn(model):
        logger.info("Building GAM interpretability artifacts...")
        return build_gam_interpretability_artifacts(model, feature_cols)

    # ---------------------------------------------------------
    # Return registry‑compatible structure
    # ---------------------------------------------------------
    return {
        "data": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_cols,
        },
        "gam": {
            "train_fn": train_fn,
            "metrics_fn": metrics_fn,
            "interpretability_fn": interpretability_fn,
        },
    }
