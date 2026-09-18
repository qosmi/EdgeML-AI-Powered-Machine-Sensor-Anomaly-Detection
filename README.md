# Edge Sensor Anomaly Detector

A predictive-maintenance prototype for detecting anomalous machine
behavior from sensor streams.

## Goal

The system will analyze machine sensor data such as:

- temperature
- vibration
- pressure
- electrical current
- rotational speed

It will detect abnormal behavior, calculate anomaly scores, identify
which sensor signals contributed to an anomaly, and expose the analysis
through a FastAPI service.

## Planned pipeline

CSV sensor data
    ↓
Data validation
    ↓
Feature engineering
    ↓
Anomaly detection
    ↓
Anomaly scoring
    ↓
Evaluation
    ↓
Signal attribution
    ↓
Machine-condition classification
    ↓
FastAPI API
    ↓
Dashboard

## Project status

Initial project structure.