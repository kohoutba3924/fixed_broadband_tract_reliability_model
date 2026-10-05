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
    """
    Registry‑compatible Tier 1 training module.

    Instead of training models directly, this function now returns:
      - train_fn: a function that trains the model when called
      - metrics_fn: a function that evaluates the model when called
      - data: X_train, X_test, y_train, y_test, feature_names

    The model registry will decide whether to:
      - load a saved model, OR
      - call train_fn() to train a new one
    """

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    logger.info("Loading modeling dataset for Tier 1...")
    X, y, feature_cols = load_modeling_dataset(
        "data/processed/modeling/final_feature_set.parquet"
    )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------
    logger.info("Splitting train/test for Tier 1...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

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
        "random_forest": {
            "train_fn": lambda: build_random_forest().fit(X_train, y_train),
            "metrics_fn": lambda model: {
                **evaluate_model(model, X_train, X_test, y_train, y_test),
                **cross_validate_model(model, X, y),
            },
        },
        "hist_gradient_boosting": {
            "train_fn": lambda: build_hgb().fit(X_train, y_train),
            "metrics_fn": lambda model: {
                **evaluate_model(model, X_train, X_test, y_train, y_test),
                **cross_validate_model(model, X, y),
            },
        },
    }
