from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Fit the model and compute test-set metrics.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "mse": mean_squared_error(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }


def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation and return mean CV metrics.
    """
    mse_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    return {
        "cv_mse_mean": mse_scores.mean(),
        "cv_mae_mean": mae_scores.mean(),
        "cv_r2_mean": r2_scores.mean(),
    }
