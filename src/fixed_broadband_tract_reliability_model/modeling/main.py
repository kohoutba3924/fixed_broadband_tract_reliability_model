import logging
import os

from src.fixed_broadband_tract_reliability_model.modeling.gam_plots import (
    plot_gam_feature_effect,
    plot_gam_top_features,
)
from src.fixed_broadband_tract_reliability_model.modeling.train_baselines import (
    train_baseline_models,
)
from src.fixed_broadband_tract_reliability_model.modeling.train_tier1 import (
    train_tier1_models,
)
from src.fixed_broadband_tract_reliability_model.modeling.train_tier2 import (
    train_gam_model,
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

    # -------------------------
    # Baselines
    # -------------------------
    logger.info("Running baseline models...")
    baseline_results = train_baseline_models()
    print("\nBaseline Model Results:")
    for model_name, metrics in baseline_results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Map model_name -> correct output directory
        if model_name == "ridge":
            output_dir = os.path.join("modeling_outputs", "baseline", "ridge")
        elif model_name == "elasticnet":
            output_dir = os.path.join("modeling_outputs", "baseline", "elastic_net")
        else:
            # fallback if new baselines are added later
            output_dir = os.path.join("modeling_outputs", "baseline", model_name)

        write_model_results(output_dir, model_name, metrics)
        logger.info(f"Baseline results written for model: {model_name}")

    # -------------------------
    # Tier 1 Models
    # -------------------------
    logger.info("Running Tier 1 models...")
    tier1_results = train_tier1_models()
    print("\nTier 1 Model Results:")
    for model_name, metrics in tier1_results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Map model_name -> correct output directory
        if model_name == "random_forest":
            output_dir = os.path.join("modeling_outputs", "tier1", "random_forest")
        elif model_name == "hgb":
            output_dir = os.path.join(
                "modeling_outputs", "tier1", "hist_gradient_boosting"
            )
        else:
            output_dir = os.path.join("modeling_outputs", "tier1", model_name)

        write_model_results(output_dir, model_name, metrics)
        logger.info(f"Tier 1 results written for model: {model_name}")

    # -------------------------
    # Tier 2 GAM
    # -------------------------
    logger.info("Running GAM model...")
    gam_results = train_gam_model()
    print("\nTier 2 GAM Model Results:")
    gam_metrics = {
        k: v
        for k, v in gam_results.items()
        if k not in ["model", "feature_names", "interpretability"]
    }
    for metric, value in gam_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Write GAM metrics to generalized_additive_model directory
    gam_output_dir = os.path.join(
        "modeling_outputs", "tier2", "generalized_additive_model"
    )
    write_model_results(gam_output_dir, "gam", gam_metrics)
    logger.info("GAM results written.")

    # -------------------------
    # GAM Interpretability Plots
    # -------------------------
    interp = gam_results["interpretability"]

    output_root = "modeling_outputs/tier2"
    effects_dir = os.path.join(output_root, "gam_effects")
    ranking_dir = os.path.join(output_root, "gam_rankings")

    logger.info("Saving GAM ranking plot...")
    plot_gam_top_features(interp["ranked_features"], ranking_dir)

    logger.info("Saving GAM feature effect plots...")
    for feature_name, effect_data in interp["effects"].items():
        plot_gam_feature_effect(feature_name, effect_data, effects_dir)

    logger.info("Modeling pipeline complete.")


if __name__ == "__main__":
    main()
