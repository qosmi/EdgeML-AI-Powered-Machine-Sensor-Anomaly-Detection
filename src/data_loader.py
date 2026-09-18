"""Load and validate machine sensor data."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "temperature",
    "vibration",
    "pressure",
    "current",
    "rpm",
    "condition",
}

NUMERIC_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "current",
    "rpm",
]


def load_sensor_data(path: str | Path) -> pd.DataFrame:
    """Load sensor data from a CSV file and validate it."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Sensor data not found: {path}")

    data = pd.read_csv(path)

    validate_sensor_data(data)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data


def validate_sensor_data(data: pd.DataFrame) -> None:
    """Validate the structure and values of sensor data."""

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Sensor dataset is empty.")

    if data["timestamp"].isna().any():
        raise ValueError("Dataset contains invalid timestamps.")

    if data["timestamp"].duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps.")

    if data[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing sensor values.")

    if not data[NUMERIC_COLUMNS].apply(
        lambda column: column.map(pd.api.types.is_number).all()
    ).all():
        raise ValueError("Sensor columns must contain numeric values.")

    if (data["vibration"] < 0).any():
        raise ValueError("Vibration cannot be negative.")

    if (data["pressure"] < 0).any():
        raise ValueError("Pressure cannot be negative.")

    if (data["current"] < 0).any():
        raise ValueError("Current cannot be negative.")

    if (data["rpm"] < 0).any():
        raise ValueError("RPM cannot be negative.")


if __name__ == "__main__":
    sensor_data = load_sensor_data("data/sensor_data.csv")

    print(f"Loaded {len(sensor_data):,} records.")
    print(f"Columns: {list(sensor_data.columns)}")
    print("\nValidation successful.")