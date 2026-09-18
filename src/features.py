"""Feature engineering for machine sensor data."""

import pandas as pd


SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "current",
    "rpm",
]


def add_sensor_features(
    data: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """Add rolling statistics and rate-of-change features."""

    features = data.copy()

    for sensor in SENSOR_COLUMNS:
        features[f"{sensor}_rolling_mean"] = (
            features[sensor]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        features[f"{sensor}_rolling_std"] = (
            features[sensor]
            .rolling(window=window, min_periods=2)
            .std()
            .fillna(0)
        )

        features[f"{sensor}_rate"] = features[sensor].diff().fillna(0)

    return features