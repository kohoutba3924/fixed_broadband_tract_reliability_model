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


def train_gam_model():
    # Load data
    logger.info("Loading modeling dataset for Tier 2...")
    X, y, feature_cols = load_modeling_dataset(
        "data/processed/modeling/final_feature_set.parquet"
    )

    # Train/test split
    logger.info("Splitting train/test for Tier 2...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Build GAM
    logger.info("Training Generalized Additive Model...")
    gam = build_gam(n_features=X.shape[1])

    # Fit GAM (pyGAM uses its own fitting loop)
    gam.fit(X_train, y_train)

    # Evaluate on test set
    eval_metrics = evaluate_model(gam, X_train, X_test, y_train, y_test)

    # Manual CV because pyGAM does not integrate with sklearn CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse = []
    cv_mae = []
    cv_r2 = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        gam_cv = build_gam(n_features=X.shape[1])
        gam_cv.fit(X_tr, y_tr)
        preds = gam_cv.predict(X_te)

        # Compute metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        cv_mse.append(mean_squared_error(y_te, preds))
        cv_mae.append(mean_absolute_error(y_te, preds))
        cv_r2.append(r2_score(y_te, preds))

    cv_results = {
        "cv_mse_mean": np.mean(cv_mse),
        "cv_mae_mean": np.mean(cv_mae),
        "cv_r2_mean": np.mean(cv_r2),
    }

    interpretability = build_gam_interpretability_artifacts(gam, feature_cols)

    logger.info("Generalized Additive Model complete.")

    return {
        **eval_metrics,
        **cv_results,
        "model": gam,
        "feature_names": feature_cols,
        "interpretability": interpretability,
    }
