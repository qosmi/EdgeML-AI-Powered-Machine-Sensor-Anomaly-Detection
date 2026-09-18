"""Tests for the anomaly detector."""

import pandas as pd

from src.anomaly_detector import AnomalyDetector


def test_detector_produces_predictions():
    data = pd.DataFrame(
        {
            "temperature": [70, 71, 69, 72, 100],
            "vibration": [0.3, 0.31, 0.29, 0.32, 1.2],
            "pressure": [5, 5.1, 4.9, 5, 8],
            "current": [12, 12.1, 11.9, 12, 20],
            "rpm": [1800, 1810, 1790, 1805, 2500],
        }
    )

    detector = AnomalyDetector(
        contamination=0.2,
    )

    detector.fit(data)

    result = detector.predict(data)

    assert "anomaly" in result
    assert "anomaly_score" in result
    assert len(result) == len(data)