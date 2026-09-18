"""End-to-end pipeline tests."""

from src.pipeline import analyze


def test_pipeline_analyzes_sensor_data():
    result = analyze("data/sensor_data.csv")

    assert result["records"] == 10_000
    assert result["anomalies"] >= 0
    assert 0 <= result["anomaly_rate"] <= 1