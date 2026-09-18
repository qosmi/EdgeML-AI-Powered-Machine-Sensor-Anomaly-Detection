"""Run an anomaly detector evaluation."""

from src.anomaly_detector import AnomalyDetector
from src.data_loader import load_sensor_data
from src.evaluation import evaluate_anomaly_detector


def main() -> None:
    data = load_sensor_data("data/sensor_data.csv")

    detector = AnomalyDetector()
    detector.fit(data)

    result = detector.predict(data)

    metrics = evaluate_anomaly_detector(result)

    print("Anomaly detector evaluation:")

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()