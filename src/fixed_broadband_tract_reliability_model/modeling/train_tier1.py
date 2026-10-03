import logging

from fixed_broadband_tract_reliability_model.modeling.utils.data_io import (
    load_modeling_dataset,
)
from fixed_broadband_tract_reliability_model.modeling.utils.evaluation import (
    cross_validate_model,
    evaluate_model,
)
from fixed_broadband_tract_reliability_model.modeling.utils.preprocessing import (
    split_train_test,
)
from src.fixed_broadband_tract_reliability_model.modeling.models_tier1 import (
    build_hgb,
    build_random_forest,
)

logger = logging.getLogger(__name__)


def train_tier1_models():
    # Load data
    logger.info("Loading modeling dataset for Tier 1...")
    X, y, feature_cols = load_modeling_dataset(
        "data/processed/modeling/final_feature_set.parquet"
    )

    # Train/test split
    logger.info("Splitting train/test for Tier 1...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    results = {}

    # Random Forest
    logger.info("Training Random Forest Regressor...")
    rf = build_random_forest()
    rf_eval = evaluate_model(rf, X_train, X_test, y_train, y_test)
    rf_cv = cross_validate_model(rf, X, y)
    results["random_forest"] = {**rf_eval, **rf_cv}
    logger.info("Random Forest Regressor complete.")

    # HistGradientBoostingRegressor
    logger.info("Training Hist Gradient Boosting Regressor...")
    hgb = build_hgb()
    hgb_eval = evaluate_model(hgb, X_train, X_test, y_train, y_test)
    hgb_cv = cross_validate_model(hgb, X, y)
    results["hgb"] = {**hgb_eval, **hgb_cv}
    logger.info("Hist Gradient Boosting Regressor complete.")

    return results
