"""Tests for sensor data loading and validation."""

import pandas as pd
import pytest

from src.data_loader import validate_sensor_data


def valid_data() -> pd.DataFrame:
    """Return a minimal valid sensor dataset."""

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 00:01:00"]
            ),
            "temperature": [70.0, 71.0],
            "vibration": [0.35, 0.36],
            "pressure": [5.0, 5.1],
            "current": [12.0, 12.1],
            "rpm": [1800.0, 1810.0],
            "condition": ["normal", "normal"],
        }
    )


def test_valid_sensor_data_passes():
    data = valid_data()

    validate_sensor_data(data)


def test_missing_column_is_rejected():
    data = valid_data().drop(columns=["temperature"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_sensor_data(data)


def test_negative_rpm_is_rejected():
    data = valid_data()
    data.loc[0, "rpm"] = -100

    with pytest.raises(ValueError, match="RPM cannot be negative"):
        validate_sensor_data(data)


def test_duplicate_timestamp_is_rejected():
    data = valid_data()
    data.loc[1, "timestamp"] = data.loc[0, "timestamp"]

    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_sensor_data(data)