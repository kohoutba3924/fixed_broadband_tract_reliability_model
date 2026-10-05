import json
import os

import matplotlib.pyplot as plt


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_gam_full_ranking(full_ranking, output_dir):
    """
    Plot full p-value ranking (all features).
    """
    _ensure_dir(output_dir)
    save_path = os.path.join(output_dir, "gam_pvalue_ranking.png")

    names = list(full_ranking.keys())
    pvals = list(full_ranking.values())

    plt.figure(figsize=(12, max(6, len(names) * 0.25)))
    plt.barh(names[::-1], pvals[::-1], color="steelblue")
    plt.title("GAM Feature Ranking (p-values)")
    plt.xlabel("p-value (lower = stronger effect)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gam_feature_effect(feature_name, effect_data, output_dir):
    """
    Save a GAM feature effect plot with confidence intervals.
    """
    _ensure_dir(output_dir)
    save_path = os.path.join(output_dir, f"gam_effect_{feature_name}.png")

    x = effect_data["x"]
    y = effect_data["y"]
    y_lower = effect_data["y_lower"]
    y_upper = effect_data["y_upper"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Effect", color="blue")
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, color="blue")

    plt.title(f"GAM Partial Effect — {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Effect on Reliability Index")
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
