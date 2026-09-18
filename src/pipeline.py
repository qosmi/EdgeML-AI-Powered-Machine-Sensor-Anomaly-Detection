"""End-to-end EdgeML analysis pipeline."""

from src.anomaly_detector import AnomalyDetector
from src.attribution import add_signal_attribution
from src.baseline import add_statistical_baseline
from src.classifier import ConditionClassifier
from src.data_loader import load_sensor_data
from src.features import add_sensor_features


def analyze(path: str) -> dict:
    """Analyze a sensor dataset."""

    data = load_sensor_data(path)

    data = add_sensor_features(data)
    data = add_statistical_baseline(data)

    detector = AnomalyDetector()
    detector.fit(data)

    data = detector.predict(data)
    data = add_signal_attribution(data)

    classifier = ConditionClassifier()
    classifier.fit(data)

    data = classifier.predict(data)

    anomalies = data[data["anomaly"]]

    return {
        "records": len(data),
        "anomalies": len(anomalies),
        "anomaly_rate": len(anomalies) / len(data),
        "results": anomalies,
    }


def main() -> None:
    """Run the pipeline from the command line."""

    result = analyze("data/sensor_data.csv")

    print(f"Records analyzed: {result['records']:,}")
    print(f"Anomalies detected: {result['anomalies']:,}")
    print(f"Anomaly rate: {result['anomaly_rate']:.2%}")


if __name__ == "__main__":
    main()