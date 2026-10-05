import logging

import numpy as np
import shap
from sklearn.inspection import partial_dependence, permutation_importance

logger = logging.getLogger(__name__)


def compute_tier1_interpretability(
    model,
    model_name,
    X_train,
    X_test,
    y_test,
    feature_names,
    top_n=10,
    top_k_interactions=1,
    shap_sample_size=2000,
):

    logger.info(f"Starting Tier 1 interpretability for model: {model_name}")

    # ---------------------------------------------------------
    # SHAP SKIP LOGIC
    # ---------------------------------------------------------
    skip_shap = model_name == "random_forest"
    if skip_shap:
        logger.info("SHAP computation skipped for Random Forest.")
    else:
        logger.info("SHAP computation enabled for this model.")

    # ---------------------------------------------------------
    # SHAP VALUES (only for HGB)
    # ---------------------------------------------------------
    if not skip_shap:
        logger.info("Sampling X_train for SHAP...")
        if shap_sample_size is not None and shap_sample_size < len(X_train):
            sample_idx = np.random.choice(
                len(X_train), size=shap_sample_size, replace=False
            )
            X_shap = X_train[sample_idx]
            logger.info(f"SHAP sample size: {len(X_shap)}")
        else:
            X_shap = X_train
            logger.info("SHAP using full X_train (no sampling).")

        logger.info("Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        logger.info("SHAP values computed.")

        logger.info("Computing mean absolute SHAP values...")
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        logger.info("Mean absolute SHAP values computed.")

        logger.info("Ranking features by SHAP importance...")
        idx_all = np.argsort(mean_abs_shap)[::-1]
        full_ranking = {feature_names[i]: float(mean_abs_shap[i]) for i in idx_all}
        logger.info("Full SHAP ranking computed.")

        logger.info(f"Selecting top {top_n} SHAP features...")
        top_idx = idx_all[:top_n]
        top_features = [feature_names[i] for i in top_idx]
        logger.info(f"Top SHAP features: {top_features}")

    else:
        # ---------------------------------------------------------
        # SHAP SKIPPED → Use permutation importance for ranking
        # ---------------------------------------------------------
        logger.info("Computing permutation importance for RF ranking...")
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        perm_mean = perm.importances_mean
        perm_std = perm.importances_std

        idx_all = np.argsort(perm_mean)[::-1]
        full_ranking = {feature_names[i]: float(perm_mean[i]) for i in idx_all}

        top_idx = idx_all[:top_n]
        top_features = [feature_names[i] for i in top_idx]

        shap_values = None
        mean_abs_shap = None

        logger.info(f"Top RF features (perm importance): {top_features}")

        # Store perm importance results now (no second call later)
        perm_top = {
            feature_names[i]: {
                "mean": float(perm_mean[i]),
                "std": float(perm_std[i]),
            }
            for i in top_idx
        }

    # ---------------------------------------------------------
    # Permutation importance (only if SHAP was used)
    # ---------------------------------------------------------
    if not skip_shap:
        logger.info("Computing permutation importance (top-N)...")
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        perm_mean = perm.importances_mean
        perm_std = perm.importances_std

        perm_top = {
            feature_names[i]: {
                "mean": float(perm_mean[i]),
                "std": float(perm_std[i]),
            }
            for i in top_idx
        }
        logger.info("Permutation importance computed.")

    # ---------------------------------------------------------
    # Tree-based feature importance
    # ---------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        logger.info("Extracting tree-based feature importance...")
        fi = model.feature_importances_
        fi_top = {feature_names[i]: float(fi[i]) for i in top_idx}
        logger.info("Tree-based feature importance extracted.")
    else:
        fi_top = None
        logger.info("Model has no feature_importances_ attribute.")

    # ---------------------------------------------------------
    # PDP 1D (top-5)
    # ---------------------------------------------------------
    logger.info("Computing PDP 1D for top 5 features...")
    pdp_1d_results = {}
    for i in top_idx[:5]:
        fname = feature_names[i]
        logger.info(f"PDP 1D: {fname}")
        pd_result = partial_dependence(model, X_train, [i])
        xs = pd_result["grid_values"][0]
        ys = pd_result["average"][0]
        pdp_1d_results[fname] = {
            "xs": xs,
            "ys": ys,
        }
    logger.info("PDP 1D computation complete.")

    # ---------------------------------------------------------
    # PDP 2D interactions (top-1)
    # ---------------------------------------------------------
    logger.info("Computing PDP 2D interaction for top feature pair...")
    top_for_interactions = top_idx[:top_k_interactions]
    interaction_pairs = [
        (top_for_interactions[i], top_for_interactions[j])
        for i in range(len(top_for_interactions))
        for j in range(i + 1, len(top_for_interactions))
    ]

    pdp_2d_results = {}
    for i_idx, j_idx in interaction_pairs[:1]:
        f1 = feature_names[i_idx]
        f2 = feature_names[j_idx]
        logger.info(f"PDP 2D: {f1} × {f2}")

        pd_result = partial_dependence(model, X_train, [(i_idx, j_idx)])
        xs = pd_result["grid_values"][0]
        ys = pd_result["grid_values"][1]
        Z = pd_result["average"][0].reshape(len(xs), len(ys))

        pdp_2d_results[f"{f1}__{f2}"] = {
            "xs": xs,
            "ys": ys,
            "Z": Z,
        }

    logger.info("PDP 2D computation complete.")
    logger.info("Tier 1 interpretability computation complete.")

    return {
        "shap_values": shap_values,
        "mean_abs_shap": mean_abs_shap,
        "full_ranking": full_ranking,
        "top_idx": top_idx,
        "top_features": top_features,
        "perm_top": perm_top,
        "fi_top": fi_top,
        "pdp_1d": pdp_1d_results,
        "pdp_2d": pdp_2d_results,
        "interaction_pairs": interaction_pairs[:1],
    }
