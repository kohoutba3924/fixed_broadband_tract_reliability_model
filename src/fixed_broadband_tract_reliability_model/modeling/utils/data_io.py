import os
from datetime import datetime

import polars as pl


def load_modeling_dataset(path: str):
    """
    Load the pruned modeling dataset from parquet and return X, y, feature names.
    """
    df = pl.read_parquet(path)

    feature_cols = [c for c in df.columns if c != "reliability_index"]
    target_col = "reliability_index"

    X = df.select(feature_cols).to_numpy()
    y = df.select(target_col).to_numpy().ravel()

    return X, y, feature_cols


def write_model_results(output_dir: str, model_name: str, metrics: dict):
    """
    Write model metrics to a text file in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{model_name}_results.txt")

    with open(file_path, "w") as f:
        f.write(f"Model Results: {model_name}\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        for metric, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
