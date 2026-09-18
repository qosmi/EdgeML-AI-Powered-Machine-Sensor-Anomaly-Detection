"""Generate synthetic machine sensor data for EdgeML."""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
SAMPLE_COUNT = 10_000


def generate_sensor_data(
    sample_count: int = SAMPLE_COUNT,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic machine sensor measurements."""

    rng = np.random.default_rng(random_seed)

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=sample_count,
        freq="min",
    )

    # Base operating conditions
    rpm = rng.normal(1800, 35, sample_count)
    temperature = rng.normal(70, 2.5, sample_count)
    vibration = rng.normal(0.35, 0.05, sample_count)
    pressure = rng.normal(5.0, 0.12, sample_count)
    current = rng.normal(12.0, 0.4, sample_count)

    condition = np.full(sample_count, "normal", dtype=object)

    # Inject controlled failure periods.
    failure_size = 300

    failure_starts = {
        "bearing_degradation": 2_000,
        "overheating": 4_000,
        "pressure_instability": 6_000,
        "electrical_fault": 8_000,
    }

    # Bearing degradation:
    # vibration increases and temperature rises gradually.
    start = failure_starts["bearing_degradation"]
    end = start + failure_size

    vibration[start:end] += np.linspace(0.15, 0.45, failure_size)
    temperature[start:end] += np.linspace(3, 10, failure_size)
    condition[start:end] = "bearing_degradation"

    # Overheating:
    # temperature becomes significantly elevated.
    start = failure_starts["overheating"]
    end = start + failure_size

    temperature[start:end] += np.linspace(10, 25, failure_size)
    current[start:end] += np.linspace(0.5, 2.0, failure_size)
    condition[start:end] = "overheating"

    # Pressure instability:
    # pressure becomes much more volatile.
    start = failure_starts["pressure_instability"]
    end = start + failure_size

    pressure[start:end] += rng.normal(0, 0.7, failure_size)
    condition[start:end] = "pressure_instability"

    # Electrical fault:
    # current increases while RPM becomes unstable.
    start = failure_starts["electrical_fault"]
    end = start + failure_size

    current[start:end] += np.linspace(2, 5, failure_size)
    rpm[start:end] += rng.normal(0, 180, failure_size)
    condition[start:end] = "electrical_fault"

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "current": current,
            "rpm": rpm,
            "condition": condition,
        }
    )


def main() -> None:
    """Generate and save the synthetic dataset."""

    output_path = Path("data/sensor_data.csv")

    data = generate_sensor_data()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Generated {len(data):,} sensor records.")
    print(f"Saved dataset to: {output_path}")
    print("\nCondition counts:")
    print(data["condition"].value_counts())


if __name__ == "__main__":
    main()