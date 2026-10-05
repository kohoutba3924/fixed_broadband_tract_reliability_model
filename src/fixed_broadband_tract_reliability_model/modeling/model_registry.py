import logging
import os

import joblib

logger = logging.getLogger(__name__)


def ensure_dir(path: str):
    """
    Ensures the directory exists.
    """
    os.makedirs(path, exist_ok=True)


def load_or_train_model(model_name: str, save_path: str, train_fn):
    """
    Loads a saved model if present; otherwise trains and saves it.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "random_forest", "ridge").
    save_path : str
        Full path to the .pkl file where the model is stored.
    train_fn : callable
        A function that trains and returns the model when called.

    Returns
    -------
    model : object
        The loaded or newly trained model.
    """

    # Ensure directory exists
    ensure_dir(os.path.dirname(save_path))

    # Load if present
    if os.path.exists(save_path):
        logger.info(f"Loading saved model: {model_name} from {save_path}")
        return joblib.load(save_path)

    # Train if not present
    logger.info(f"No saved model found for {model_name}. Training...")
    model = train_fn()

    # Save trained model
    joblib.dump(model, save_path)
    logger.info(f"Saved trained model: {model_name} → {save_path}")

    return model
