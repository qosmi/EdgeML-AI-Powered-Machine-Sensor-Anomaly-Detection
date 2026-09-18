"""Machine condition classifier."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "current",
    "rpm",
]


class ConditionClassifier:
    """Classify machine operating conditions."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        )

    def fit(
        self,
        data: pd.DataFrame,
    ) -> None:
        """Train the classifier."""

        self.model.fit(
            data[FEATURE_COLUMNS],
            data["condition"],
        )

    def predict(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict machine conditions."""

        result = data.copy()

        result["predicted_condition"] = (
            self.model.predict(
                result[FEATURE_COLUMNS]
            )
        )

        return result