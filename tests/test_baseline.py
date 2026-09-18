"""Tests for the statistical anomaly baseline."""

import pandas as pd

from src.baseline import add_statistical_baseline
from src.features import add_sensor_features


def test_baseline_creates_score_and_flag():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=10,
                freq="min",
            ),
            "temperature": [70] * 10,
            "vibration": [0.3] * 10,
            "pressure": [5] * 10,
            "current": [12] * 10,
            "rpm": [1800] * 10,
            "condition": ["normal"] * 10,
        }
    )

    features = add_sensor_features(data, window=3)
    result = add_statistical_baseline(features)

    assert "baseline_score" in result
    assert "baseline_anomaly" in result