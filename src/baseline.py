"""Statistical anomaly detection baseline."""

import numpy as np
import pandas as pd

from src.features import SENSOR_COLUMNS


def add_statistical_baseline(
    data: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Add a statistical anomaly score and flag."""

    result = data.copy()

    z_scores = []

    for sensor in SENSOR_COLUMNS:
        mean = result[f"{sensor}_rolling_mean"]
        std = result[f"{sensor}_rolling_std"]

        z_score = (
            (result[sensor] - mean)
            / std.replace(0, np.nan)
        ).abs().fillna(0)

        result[f"{sensor}_zscore"] = z_score
        z_scores.append(z_score)

    result["baseline_score"] = pd.concat(
        z_scores,
        axis=1,
    ).max(axis=1)

    result["baseline_anomaly"] = (
        result["baseline_score"] >= threshold
    )

    return result