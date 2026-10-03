def extract_gam_feature_effects(gam, feature_names):
    """
    Extract smooth feature effects and confidence intervals for each feature.
    Uses pyGAM's internal grid to avoid shape mismatches.
    """
    effects = {}

    for i, name in enumerate(feature_names):
        # Use pyGAM's internal grid for this term
        grid = gam.generate_X_grid(term=i)
        x = grid[:, i]

        # Partial dependence using internal grid (no X argument!)
        y = gam.partial_dependence(term=i)
        y_conf = gam.partial_dependence(term=i, width=0.95)

        # Confidence intervals
        y_lower = y_conf[0]
        y_upper = y_conf[1]

        # Ensure all arrays are 1-D
        x = x.flatten()
        y = y.flatten()
        y_lower = y_lower.flatten()
        y_upper = y_upper.flatten()

        # Final sanity check: lengths must match
        n = min(len(x), len(y), len(y_lower), len(y_upper))
        x = x[:n]
        y = y[:n]
        y_lower = y_lower[:n]
        y_upper = y_upper[:n]

        effects[name] = {
            "x": x,
            "y": y,
            "y_lower": y_lower,
            "y_upper": y_upper,
        }

    return effects


def extract_gam_statistics(gam, feature_names):
    """
    Extract per-feature p-values only.
    """
    stats = gam.statistics_
    p_values = stats["p_values"]

    feature_stats = {}

    # Skip intercept at index 0
    for i, name in enumerate(feature_names, start=1):
        feature_stats[name] = {"p_value": p_values[i]}

    return feature_stats


def rank_gam_features(feature_stats):
    """
    Rank features by p-value (lower = stronger effect).
    """
    return sorted(feature_stats.items(), key=lambda kv: kv[1]["p_value"])


def build_gam_interpretability_artifacts(gam, feature_names):
    """
    Build complete GAM interpretability artifact object.
    """
    effects = extract_gam_feature_effects(gam, feature_names)
    stats = extract_gam_statistics(gam, feature_names)
    ranked = rank_gam_features(stats)

    return {
        "effects": effects,
        "stats": stats,
        "ranked_features": ranked,
    }
