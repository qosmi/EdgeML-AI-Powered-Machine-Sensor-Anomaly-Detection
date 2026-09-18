"""Machine-learning anomaly detector."""

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features import SENSOR_COLUMNS


FEATURE_COLUMNS = [
    sensor
    for sensor in SENSOR_COLUMNS
]


class AnomalyDetector:
    """Isolation Forest anomaly detector."""

    def __init__(
        self,
        contamination: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the detector."""

        self.model.fit(data[FEATURE_COLUMNS])

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return anomaly predictions and scores."""

        result = data.copy()

        predictions = self.model.predict(
            result[FEATURE_COLUMNS]
        )

        result["anomaly"] = predictions == -1

        result["anomaly_score"] = -self.model.score_samples(
            result[FEATURE_COLUMNS]
        )

        return result