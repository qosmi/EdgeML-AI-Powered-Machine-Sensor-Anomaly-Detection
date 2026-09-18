"""Evaluation utilities for anomaly detection."""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_anomaly_detector(
    data: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate predicted anomalies against known conditions."""

    actual = data["condition"] != "normal"
    predicted = data["anomaly"]

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
    ).ravel()

    false_positive_rate = fp / (fp + tn)

    return {
        "precision": precision_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "f1": f1_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "false_positive_rate": false_positive_rate,
    }