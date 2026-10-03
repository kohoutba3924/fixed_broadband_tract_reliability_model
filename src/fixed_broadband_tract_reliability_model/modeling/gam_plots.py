import os

import matplotlib.pyplot as plt


def plot_gam_feature_effect(feature_name, effect_data, output_dir):
    """
    Save a GAM feature effect plot with confidence intervals.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{feature_name}_effect.png")

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


def plot_gam_top_features(ranked_features, output_dir, top_n=10):
    """
    Plot top additive features ranked by p-value (lower = stronger effect).
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "top_features.png")

    # Extract names and p-values
    names = [f[0] for f in ranked_features[:top_n]]
    pvals = [f[1]["p_value"] for f in ranked_features[:top_n]]

    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], pvals[::-1], color="steelblue")
    plt.title(f"Top {top_n} GAM Additive Feature Effects (Ranked by p-value)")
    plt.xlabel("p-value (lower = stronger effect)")
    plt.grid(axis="x")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
