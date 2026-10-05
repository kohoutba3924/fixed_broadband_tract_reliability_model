import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_tier1_interpretability(
    model_name,
    feature_names,
    artifacts,
    output_dir,
):
    """
    Tier 1 plotting:
    - SHAP ranking only for HGB
    - Permutation importance bar chart
    - Feature importance bar chart
    - PDP 1D (top-5)
    - PDP 2D (top-1)
    """

    _ensure_dir(output_dir)

    shap_values = artifacts["shap_values"]
    mean_abs_shap = artifacts["mean_abs_shap"]
    full_ranking = artifacts["full_ranking"]
    perm_top = artifacts["perm_top"]
    fi_top = artifacts["fi_top"]
    pdp_1d = artifacts["pdp_1d"]
    pdp_2d = artifacts["pdp_2d"]

    # ---------------------------------------------------------
    # SHAP ranking plot (HGB only)
    # ---------------------------------------------------------
    if shap_values is not None and mean_abs_shap is not None:
        _save_json(full_ranking, os.path.join(output_dir, "shap_feature_ranking.json"))

        names_all = list(full_ranking.keys())
        vals_all = list(full_ranking.values())

        plt.figure(figsize=(12, max(6, len(names_all) * 0.25)))
        plt.barh(names_all[::-1], vals_all[::-1], color="slateblue")
        plt.title(f"{model_name} — Full Feature Importance (Mean |SHAP|)")
        plt.xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_feature_ranking.png"), dpi=300)
        plt.close()

    else:
        # SHAP skipped → write placeholder JSON
        _save_json(
            {"shap_skipped": True, "reason": "RandomForest SHAP disabled"},
            os.path.join(output_dir, "shap_feature_ranking.json"),
        )

    # ---------------------------------------------------------
    # Permutation importance bar chart
    # ---------------------------------------------------------
    if perm_top is not None:
        names = list(perm_top.keys())
        means = [perm_top[n]["mean"] for n in names]
        stds = [perm_top[n]["std"] for n in names]

        plt.figure(figsize=(10, max(5, len(names) * 0.3)))
        plt.barh(names[::-1], means[::-1], xerr=stds[::-1], color="darkorange")
        plt.title(f"{model_name} — Permutation Importance (Top-N)")
        plt.xlabel("Importance (mean decrease in score)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "perm_importance.png"), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Feature importance bar chart (tree-based)
    # ---------------------------------------------------------
    if fi_top is not None:
        names = list(fi_top.keys())
        vals = list(fi_top.values())

        plt.figure(figsize=(10, max(5, len(names) * 0.3)))
        plt.barh(names[::-1], vals[::-1], color="seagreen")
        plt.title(f"{model_name} — Tree-Based Feature Importance (Top-N)")
        plt.xlabel("Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tree_feature_importance.png"), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # PDP 1D (top-5)
    # ---------------------------------------------------------
    for fname, pd_data in pdp_1d.items():
        xs = pd_data["xs"]
        ys = pd_data["ys"]

        plt.figure(figsize=(8, 6))
        plt.plot(xs, ys, color="steelblue")
        plt.title(f"PDP: {fname}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pdp_{fname}.png"))
        plt.close()

    # ---------------------------------------------------------
    # PDP 2D interactions (top-1)
    # ---------------------------------------------------------
    for pair_key, pd_data in pdp_2d.items():
        xs = pd_data["xs"]
        ys = pd_data["ys"]
        Z = pd_data["Z"]

        Xs, Ys = np.meshgrid(xs, ys)

        plt.figure(figsize=(8, 6))
        cp = plt.contourf(Xs, Ys, Z.T, cmap="viridis")
        plt.colorbar(cp)
        plt.title(f"PDP Interaction: {pair_key.replace('__', ' × ')}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pdp_interaction_{pair_key}.png"))
        plt.close()
