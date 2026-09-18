"""Sensor-level anomaly attribution."""

import pandas as pd

from src.features import SENSOR_COLUMNS


def add_signal_attribution(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate relative sensor deviations."""

    result = data.copy()

    deviations = []

    for sensor in SENSOR_COLUMNS:
        baseline = result[f"{sensor}_rolling_mean"]

        deviation = (
            (result[sensor] - baseline)
            .abs()
            / baseline.abs().replace(0, pd.NA)
        ).fillna(0)

        result[f"{sensor}_deviation"] = deviation
        deviations.append(deviation)

    result["primary_signal"] = (
        pd.concat(deviations, axis=1)
        .idxmax(axis=1)
        .str.replace("_deviation", "", regex=False)
    )

    return result