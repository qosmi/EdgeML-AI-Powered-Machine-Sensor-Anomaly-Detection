"""Tests for feature engineering."""

import pandas as pd

from src.features import add_sensor_features


def test_sensor_features_are_created():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="min",
            ),
            "temperature": [70, 71, 72, 73, 74],
            "vibration": [0.3, 0.31, 0.32, 0.33, 0.34],
            "pressure": [5.0, 5.1, 5.0, 5.1, 5.0],
            "current": [12, 12.1, 12.0, 12.2, 12.1],
            "rpm": [1800, 1810, 1805, 1815, 1808],
            "condition": ["normal"] * 5,
        }
    )

    result = add_sensor_features(data, window=3)

    assert "temperature_rolling_mean" in result.columns
    assert "temperature_rolling_std" in result.columns
    assert "temperature_rate" in result.columns

    assert "vibration_rolling_mean" in result.columns
    assert "pressure_rolling_std" in result.columns
    assert "rpm_rate" in result.columns


def test_rate_of_change_is_calculated():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="min",
            ),
            "temperature": [70, 72, 75],
            "vibration": [0.3, 0.3, 0.3],
            "pressure": [5.0, 5.0, 5.0],
            "current": [12, 12, 12],
            "rpm": [1800, 1800, 1800],
            "condition": ["normal"] * 3,
        }
    )

    result = add_sensor_features(data, window=2)

    assert result["temperature_rate"].tolist() == [0, 2, 3]