def extract_gam_statistics(gam, feature_names):
    """
    Extract per-feature p-values only.
    """
    p_values = gam.statistics_["p_values"]

    # Skip intercept at index 0
    stats = {
        name: {"p_value": float(p_values[i])}
        for i, name in enumerate(feature_names, start=1)
    }
    return stats


def rank_gam_features(feature_stats):
    """
    Rank features by p-value (lower = stronger effect).
    """
    return sorted(feature_stats.items(), key=lambda kv: kv[1]["p_value"])


def extract_gam_feature_effects(gam, feature_names, top_idx):
    """
    Extract smooth feature effects and confidence intervals for top-N features only.
    """
    effects = {}

    for i in top_idx:
        name = feature_names[i]

        grid = gam.generate_X_grid(term=i)
        x = grid[:, i]

        y = gam.partial_dependence(term=i)
        y_conf = gam.partial_dependence(term=i, width=0.95)

        y_lower = y_conf[0]
        y_upper = y_conf[1]

        x = x.flatten()
        y = y.flatten()
        y_lower = y_lower.flatten()
        y_upper = y_upper.flatten()

        n = min(len(x), len(y), len(y_lower), len(y_upper))
        effects[name] = {
            "x": x[:n],
            "y": y[:n],
            "y_lower": y_lower[:n],
            "y_upper": y_upper[:n],
        }

    return effects


def build_gam_interpretability_artifacts(gam, feature_names, top_n=10):
    """
    Build complete GAM interpretability artifact object.
    - full p-value ranking
    - top-N feature selection
    - effects only for top-N features
    """
    stats = extract_gam_statistics(gam, feature_names)
    ranked = rank_gam_features(stats)

    # Full ranking (all features)
    full_ranking = {name: d["p_value"] for name, d in ranked}

    # Top-N features
    top_features = ranked[:top_n]
    top_idx = [feature_names.index(name) for name, _ in top_features]

    # Extract effects only for top-N
    effects = extract_gam_feature_effects(gam, feature_names, top_idx)

    return {
        "stats": stats,
        "ranked_features": ranked,
        "full_ranking": full_ranking,
        "top_features": top_features,
        "top_idx": top_idx,
        "effects": effects,
    }
