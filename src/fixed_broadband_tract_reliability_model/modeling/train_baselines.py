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
from src.fixed_broadband_tract_reliability_model.modeling.models_baseline import (
    build_elasticnet,
    build_ridge,
)

logger = logging.getLogger(__name__)


def train_baseline_models():
    # Load data
    logger.info("Loading modeling dataset for baseline...")
    X, y, feature_cols = load_modeling_dataset(
        "data/processed/modeling/final_feature_set.parquet"
    )

    # Train/test split
    logger.info("Splitting train/test for baseline...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    results = {}

    # Ridge Regression
    logger.info("Training Ridge Regressor...")
    ridge = build_ridge(alpha=1.0)
    ridge_eval = evaluate_model(ridge, X_train, X_test, y_train, y_test)
    ridge_cv = cross_validate_model(ridge, X, y)
    results["ridge"] = {**ridge_eval, **ridge_cv}
    logger.info("Ridge Regressor complete.")

    # ElasticNet Regression
    logger.info("Training ElasticNet Regressor...")
    enet = build_elasticnet(alpha=1.0, l1_ratio=0.5)
    enet_eval = evaluate_model(enet, X_train, X_test, y_train, y_test)
    enet_cv = cross_validate_model(enet, X, y)
    results["elasticnet"] = {**enet_eval, **enet_cv}
    logger.info("ElasticNet Regressor complete.")

    return results
