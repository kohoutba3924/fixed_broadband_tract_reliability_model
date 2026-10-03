from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


def build_random_forest(
    n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1
):
    """
    Build a RandomForestRegressor model.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=42,
    )


def build_hgb(learning_rate=0.05, max_iter=500, max_depth=None):
    """
    Build a HistGradientBoostingRegressor model.
    """
    return HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        random_state=42,
    )
