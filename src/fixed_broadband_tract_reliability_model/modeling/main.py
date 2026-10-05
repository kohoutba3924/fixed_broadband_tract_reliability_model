import logging
import os

from src.fixed_broadband_tract_reliability_model.modeling.gam_plots import (
    plot_gam_feature_effect,
    plot_gam_full_ranking,
)
from src.fixed_broadband_tract_reliability_model.modeling.model_registry import (
    load_or_train_model,
)
from src.fixed_broadband_tract_reliability_model.modeling.tier1_interpretability import (
    compute_tier1_interpretability,
)
from src.fixed_broadband_tract_reliability_model.modeling.tier1_plots import (
    plot_tier1_interpretability,
)
from src.fixed_broadband_tract_reliability_model.modeling.train_baselines import (
    train_baseline_models,
)
from src.fixed_broadband_tract_reliability_model.modeling.train_tier1 import (
    train_tier1_models,
)
from src.fixed_broadband_tract_reliability_model.modeling.train_tier2 import (
    train_tier2_gam,
)
from src.fixed_broadband_tract_reliability_model.modeling.utils.data_io import (
    write_model_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting modeling pipeline...")

    # =========================================================
    # BASELINE MODELS (ridge + elasticnet)
    # =========================================================
    logger.info("Preparing baseline models...")
    baseline = train_baseline_models()
    baseline_data = baseline["data"]

    _X_train_b = baseline_data["X_train"]
    _X_test_b = baseline_data["X_test"]
    _y_train_b = baseline_data["y_train"]
    _y_test_b = baseline_data["y_test"]

    print("\nBaseline Model Results:")

    baseline_store = os.path.join("model_store", "baseline")
    os.makedirs(baseline_store, exist_ok=True)

    for model_name, model_info in baseline.items():
        if model_name == "data":
            continue

        save_path = os.path.join(baseline_store, f"{model_name}.pkl")

        # Load or train
        model = load_or_train_model(
            model_name=model_name,
            save_path=save_path,
            train_fn=model_info["train_fn"],
        )

        # Evaluate
        metrics = model_info["metrics_fn"](model)

        # Output directory
        if model_name == "ridge":
            output_dir = os.path.join("modeling_outputs", "baseline", "ridge")
        elif model_name == "elasticnet":
            output_dir = os.path.join("modeling_outputs", "baseline", "elastic_net")
        else:
            output_dir = os.path.join("modeling_outputs", "baseline", model_name)

        write_model_results(output_dir, model_name, metrics)

        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        logger.info(f"Baseline results written for model: {model_name}")

    # =========================================================
    # TIER 1 MODELS (RF + HGB)
    # =========================================================
    logger.info("Preparing Tier 1 models...")
    tier1 = train_tier1_models()
    tier1_data = tier1["data"]

    X_train = tier1_data["X_train"]
    X_test = tier1_data["X_test"]
    _y_train = tier1_data["y_train"]
    y_test = tier1_data["y_test"]
    feature_names = tier1_data["feature_names"]

    print("\nTier 1 Model Results:")

    tier1_store = os.path.join("model_store", "tier1")
    os.makedirs(tier1_store, exist_ok=True)

    for model_name, model_info in tier1.items():
        if model_name == "data":
            continue

        save_path = os.path.join(tier1_store, f"{model_name}.pkl")

        # Load or train
        model = load_or_train_model(
            model_name=model_name,
            save_path=save_path,
            train_fn=model_info["train_fn"],
        )

        # Evaluate
        metrics = model_info["metrics_fn"](model)

        # Output directory
        output_dir = os.path.join("modeling_outputs", "tier1", model_name)
        write_model_results(output_dir, model_name, metrics)

        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        logger.info(f"Tier 1 results written for model: {model_name}")

        # Interpretability
        logger.info(f"Computing interpretability artifacts for {model_name}...")
        artifacts = compute_tier1_interpretability(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            top_n=10,
            top_k_interactions=1,
        )

        logger.info(f"Saving interpretability plots for {model_name}...")
        plot_tier1_interpretability(
            model_name=model_name,
            feature_names=feature_names,
            artifacts=artifacts,
            output_dir=output_dir,
        )

    # =========================================================
    # TIER 2 MODEL (GAM)
    # =========================================================
    logger.info("Preparing Tier 2 GAM model...")
    tier2 = train_tier2_gam()
    tier2_data = tier2["data"]

    _X_train_g = tier2_data["X_train"]
    _X_test_g = tier2_data["X_test"]
    _y_train_g = tier2_data["y_train"]
    _y_test_g = tier2_data["y_test"]
    _feature_names_g = tier2_data["feature_names"]

    gam_store = os.path.join("model_store", "tier2")
    os.makedirs(gam_store, exist_ok=True)

    save_path = os.path.join(gam_store, "gam.pkl")

    # Load or train GAM
    gam_model = load_or_train_model(
        model_name="gam",
        save_path=save_path,
        train_fn=tier2["gam"]["train_fn"],
    )

    # Evaluate GAM
    gam_metrics = tier2["gam"]["metrics_fn"](gam_model)

    print("\nTier 2 GAM Model Results:")
    for metric, value in gam_metrics.items():
        print(f"  {metric}: {value:.4f}")

    gam_output_dir = os.path.join(
        "modeling_outputs", "tier2", "generalized_additive_model"
    )
    write_model_results(gam_output_dir, "gam", gam_metrics)
    logger.info("GAM results written.")

    # Interpretability
    logger.info("Computing GAM interpretability artifacts...")
    gam_artifacts = tier2["gam"]["interpretability_fn"](gam_model)

    # Ranking plot
    ranking_dir = os.path.join("modeling_outputs", "tier2", "gam_rankings")
    logger.info("Saving GAM full ranking plot...")
    plot_gam_full_ranking(gam_artifacts["full_ranking"], ranking_dir)

    # Effect plots
    effects_dir = os.path.join("modeling_outputs", "tier2", "gam_effects")
    logger.info("Saving GAM feature effect plots...")
    for feature_name, effect_data in gam_artifacts["effects"].items():
        plot_gam_feature_effect(feature_name, effect_data, effects_dir)

    logger.info("Modeling pipeline complete.")


if __name__ == "__main__":
    main()
