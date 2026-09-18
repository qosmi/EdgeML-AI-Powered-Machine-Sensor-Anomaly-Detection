# EdgeML: AI-Powered Machine Sensor Anomaly Detection

EdgeML is a machine-learning project for detecting anomalous behavior in industrial machine sensor data and identifying sensor signals associated with abnormal operating conditions.

The project demonstrates an end-to-end machine-learning workflow:

**data generation → validation → feature engineering → statistical baseline → anomaly detection → evaluation → signal attribution → condition classification → API → dashboard → testing → reproducibility**

> **Project status:** v1.0.0 — educational/portfolio prototype using synthetic sensor data.

---

## 1. Overview

Industrial machines continuously produce sensor measurements such as temperature, vibration, pressure, electrical current, and rotational speed.

Detecting abnormal sensor behavior can help identify potential equipment problems before they develop into more serious failures.

EdgeML explores this problem using a controlled synthetic dataset and two complementary machine-learning approaches:

1. **Unsupervised anomaly detection** using Isolation Forest
2. **Supervised machine-condition classification** using Random Forest

A statistical anomaly detector is also implemented as a baseline so that the ML approach can be compared against a simpler method.

The system can process sensor data through a reusable Python pipeline and expose the analysis through a FastAPI service and a small Streamlit dashboard.

---

## 2. Problem

Given a time series of machine sensor measurements, the system should answer two related questions:

### Question 1 — Is the machine behaving unusually?

The anomaly detector analyzes sensor measurements and produces:

* an anomaly flag
* an anomaly score

### Question 2 — What operating condition does the pattern resemble?

The classification component predicts one of the known synthetic operating conditions:

* `normal`
* `bearing_degradation`
* `overheating`
* `pressure_instability`
* `electrical_fault`

These are separate tasks.

An anomaly score indicates that an observation differs from expected behavior. It does **not** establish the physical cause of the anomaly.

The condition classifier is trained on synthetic labels and therefore represents a machine-learning experiment rather than a validated industrial diagnostic system.

---

## 3. Project Goals

The project focuses on several practical ML-engineering concepts:

* generating controlled sensor data
* validating incoming data
* engineering temporal features
* establishing a statistical baseline
* training an unsupervised anomaly detector
* training a supervised classifier
* evaluating model performance
* identifying influential sensor deviations
* building a reusable analysis pipeline
* exposing ML functionality through an API
* creating a lightweight visualization interface
* writing automated tests
* making experiments reproducible
* maintaining a meaningful Git history

---

## 4. Architecture

The current architecture is:

```text
                     Synthetic Sensor Data
                              │
                              ▼
                    ┌──────────────────┐
                    │ Data Validation  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature          │
                    │ Engineering      │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     Statistical Baseline          Isolation Forest
              │                             │
              │                             ▼
              │                     Anomaly Detection
              │                             │
              │                             ▼
              │                    Signal Attribution
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                   Condition Classifier
                             │
                             ▼
                    Analysis Pipeline
                       │           │
                       ▼           ▼
                   FastAPI      Dashboard
```

---

## 5. Repository Structure

```text
edge-ml-predictive-maintenance/
│
├── README.md
├── requirements.txt
├── requirements-lock.txt
│
├── data/
│   ├── .gitkeep
│   └── sensor_data.csv
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── data_loader.py
│   ├── features.py
│   ├── baseline.py
│   ├── anomaly_detector.py
│   ├── evaluation.py
│   ├── attribution.py
│   ├── classifier.py
│   ├── pipeline.py
│   ├── evaluate_detector.py
│   ├── api.py
│   ├── dashboard.py
│   └── config.py
│
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_features.py
    ├── test_baseline.py
    ├── test_anomaly_detector.py
    └── test_pipeline.py
```

---

## 6. Dataset

The project uses a synthetic machine sensor dataset generated by:

```bash
python3 -m src.generate_data
```

The generated dataset contains **10,000 sensor observations** sampled at one-minute intervals.

### Sensor measurements

| Sensor        | Description             |
| ------------- | ----------------------- |
| `temperature` | Machine temperature     |
| `vibration`   | Machine vibration level |
| `pressure`    | Operating pressure      |
| `current`     | Electrical current      |
| `rpm`         | Rotational speed        |

Each observation also contains:

```text
timestamp
condition
```

### Synthetic operating conditions

The dataset contains normal operation and four controlled abnormal conditions.

| Condition              | Simulated behavior                                  |
| ---------------------- | --------------------------------------------------- |
| `normal`               | Stable operating conditions                         |
| `bearing_degradation`  | Increasing vibration and temperature                |
| `overheating`          | Significant temperature increase and higher current |
| `pressure_instability` | Increased pressure variability                      |
| `electrical_fault`     | Increased current and unstable RPM                  |

The abnormal periods are deliberately injected into the dataset so that model behavior can be evaluated against known ground-truth labels.

---

## 7. Data Validation

Before sensor data enters the ML pipeline, it is validated by `data_loader.py`.

The validation layer checks:

* required columns
* empty datasets
* valid timestamps
* duplicate timestamps
* missing sensor measurements
* numeric sensor values
* physically invalid negative measurements

For example, negative RPM is rejected because it is outside the operating assumptions of this simulated machine.

This creates a distinction between **invalid data** and **anomalous data**.

### Invalid data

Examples:

```text
missing timestamp
duplicate timestamp
missing sensor value
negative RPM
```

These represent data-quality problems.

### Anomalous data

An observation can be valid but unusual:

```text
temperature = 94.2
vibration   = 0.82
```

The anomaly detector is designed to identify this second category.

---

## 8. Feature Engineering

The raw sensor values are transformed into additional temporal features.

For each sensor, EdgeML calculates:

### Rolling mean

The recent average sensor value.

```text
sensor_rolling_mean
```

The current implementation uses a 30-observation window.

Because observations are sampled once per minute, this corresponds approximately to a 30-minute window.

### Rolling standard deviation

Measures recent sensor variability.

```text
sensor_rolling_std
```

This is particularly useful for conditions such as pressure instability.

### Rate of change

Measures the difference between the current and previous sensor value.

```text
sensor_rate
```

This helps capture rapidly changing machine behavior.

The resulting feature set provides the anomaly detector with both current measurements and information about recent operating behavior.

---

## 9. Statistical Baseline

Before using machine learning, EdgeML implements a simple statistical baseline.

The baseline calculates absolute rolling z-scores:

```text
z = |current value - rolling mean| / rolling standard deviation
```

The largest sensor z-score is used as the baseline anomaly score.

An observation is flagged when its score exceeds the configured threshold.

The baseline is intentionally simple.

Its purpose is to establish a reference point:

> Can the machine-learning model provide useful anomaly detection beyond a straightforward statistical method?

This is an important part of the experimental methodology because a complex model should not automatically be assumed to be better than a simpler approach.

---

## 10. Anomaly Detection

The main unsupervised ML model is an **Isolation Forest**.

The model receives machine sensor measurements such as:

```text
temperature
vibration
pressure
current
rpm
```

It produces:

```text
anomaly
anomaly_score
```

### Why Isolation Forest?

Isolation Forest is appropriate for this experiment because it can identify unusual observations without requiring every observation to have a failure label during training.

The model is therefore useful for exploring the distinction between:

```text
normal machine behavior
```

and:

```text
unusual machine behavior
```

The model's anomaly score is a model-specific measure of abnormality. It should not be interpreted as a probability of physical machine failure.

---

## 11. Supervised Condition Classification

EdgeML also contains a supervised classification component based on a Random Forest classifier.

The classifier predicts:

```text
normal
bearing_degradation
overheating
pressure_instability
electrical_fault
```

Unlike the anomaly detector, the classifier uses the synthetic `condition` labels during training.

This creates two separate ML tasks:

```text
                 Sensor measurements
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      Isolation Forest         Random Forest
             │                       │
             ▼                       ▼
       Is it unusual?          What condition?
```

The classifier should not be interpreted as a validated physical diagnosis system because its training labels are synthetic.

---

## 12. Signal Attribution

When an observation is considered anomalous, EdgeML calculates the relative deviation of each sensor from its recent rolling baseline.

For example:

```text
temperature deviation   0.18
vibration deviation     0.42
pressure deviation      0.03
current deviation       0.08
rpm deviation           0.02
```

The largest deviation can be reported as the primary contributing signal:

```text
primary_signal = vibration
```

This provides a simple form of interpretability.

### Important limitation

Signal attribution does **not** prove causality.

If vibration is the largest deviation, EdgeML can report that vibration contributed strongly to the detected abnormal pattern.

It cannot conclude that vibration physically caused the machine failure.

---

## 13. Evaluation

The synthetic ground-truth labels allow anomaly detection to be evaluated quantitatively.

The project calculates:

* Precision
* Recall
* F1 score
* False-positive rate

The evaluation treats:

```text
normal
```

as non-anomalous and all simulated failure conditions as anomalous.

Run the evaluation with:

```bash
python3 -m src.evaluate_detector
```

The output has the following form:

```text
Anomaly detector evaluation:
precision: ...
recall: ...
f1: ...
false_positive_rate: ...
```

The exact results depend on the model configuration and implementation.

The metrics should be interpreted within the context of the synthetic dataset rather than as evidence of real-world predictive-maintenance performance.

---

## 14. End-to-End Pipeline

The individual components are combined in `pipeline.py`.

The pipeline performs:

```text
1. Load sensor data
2. Validate the dataset
3. Generate temporal features
4. Calculate statistical baseline
5. Train Isolation Forest
6. Generate anomaly predictions
7. Calculate sensor-level deviations
8. Train condition classifier
9. Generate condition predictions
10. Return analysis results
```

The complete pipeline can be executed with:

```bash
python3 -m src.pipeline
```

Example output:

```text
Records analyzed: 10,000
Anomalies detected: ...
Anomaly rate: ...%
```

---

## 15. FastAPI

The analysis pipeline is exposed through a FastAPI service.

Start the API with:

```bash
python3 -m uvicorn src.api:app --reload
```

The service provides:

```text
GET  /health
POST /analyze
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### Interactive API documentation

When the server is running, FastAPI provides interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

The `/analyze` endpoint accepts a dataset path and runs the EdgeML analysis pipeline.

---

## 16. Dashboard

EdgeML includes a lightweight Streamlit dashboard.

Start it with:

```bash
python3 -m streamlit run src/dashboard.py
```

The dashboard provides a simple interface for:

* selecting the sensor dataset
* running the analysis
* displaying the number of records
* displaying the number of detected anomalies
* displaying the anomaly rate

The dashboard is intentionally small. Its purpose is to demonstrate how the ML pipeline can be connected to a user-facing interface without making the frontend the primary focus of the project.

---

## 17. Testing

Automated tests are provided using `pytest`.

Run the complete test suite:

```bash
python3 -m pytest
```

The tests cover:

* sensor-data validation
* missing-column detection
* invalid sensor values
* duplicate timestamps
* feature generation
* rate-of-change calculation
* statistical baseline creation
* anomaly-detector predictions
* end-to-end pipeline execution

The project follows a simple principle:

> Every important component should be independently testable.

---

## 18. Reproducibility

The project uses a fixed random seed for synthetic data generation and machine-learning experiments.

The main configuration values are stored in:

```text
src/config.py
```

The project also maintains:

```text
requirements.txt
requirements-lock.txt
```

The first file records the intentionally selected dependencies.

The lock file records the installed dependency environment used during development.

To inspect the Python version used for the project:

```bash
python3 --version
```

---

## 19. Installation

Clone the repository and enter the project directory.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

Generate the dataset:

```bash
python3 -m src.generate_data
```

Run the tests:

```bash
python3 -m pytest
```

Run the pipeline:

```bash
python3 -m src.pipeline
```

---

## 20. Quick Start

After installation, the shortest path to running EdgeML is:

```bash
python3 -m src.generate_data
python3 -m pytest
python3 -m src.pipeline
```

To start the API:

```bash
python3 -m uvicorn src.api:app --reload
```

To start the dashboard:

```bash
python3 -m streamlit run src/dashboard.py
```

---

## 21. Git Development History

The project was intentionally developed incrementally rather than being created as a finished application.

The major development checkpoints are:

```text
chore: initialize EdgeML project
feat: add synthetic machine sensor data
feat: add sensor data validation
feat: add rolling sensor features
feat: add statistical anomaly baseline
feat: add isolation forest anomaly detector
feat: add anomaly detection evaluation
feat: add anomaly signal attribution
feat: add machine condition classifier
feat: add end to end analysis pipeline
feat: expose anomaly analysis through api
feat: add sensor anomaly dashboard
test: add end to end pipeline tests
chore: make experiments reproducible
release: EdgeML v1.0.0
```

This progression reflects the development of the system from data foundation through ML experimentation and finally into a usable application.

---

## 22. Limitations

EdgeML is a prototype and has several important limitations.

### Synthetic data

The dataset is artificially generated.

It does not represent measurements collected from a real industrial machine.

### Simulated failure modes

The abnormal conditions are manually injected into the data.

Real equipment failures are more complex and may involve interactions between multiple physical processes.

### Synthetic labels

The condition classifier is trained using synthetic labels.

Those labels should not be treated as equivalent to validated maintenance or failure records.

### No causal inference

The system identifies unusual patterns and sensor deviations.

It does not establish physical causality.

### No production validation

The models have not been validated against real industrial equipment, real maintenance records, or real failure events.

### No safety-critical use

The output should not be used as a standalone basis for maintenance, safety, or operational decisions.

---

## 23. Future Work

Possible future improvements include:

* validation against a real industrial sensor dataset
* time-series cross-validation
* more rigorous temporal train/test splitting
* hyperparameter experiments
* comparison with additional anomaly-detection algorithms
* model calibration
* improved feature selection
* automated model evaluation reports
* real-time sensor ingestion
* model persistence
* monitoring for model drift
* experiment tracking
* deployment to an edge device
* integration with real maintenance records

These extensions are deliberately outside the scope of the initial v1.0 project.

---

## 24. Technologies

The project uses:

* **Python** — primary programming language
* **NumPy** — numerical operations
* **Pandas** — data processing
* **scikit-learn** — machine learning
* **FastAPI** — API service
* **Pydantic** — API data validation
* **Uvicorn** — ASGI server
* **Streamlit** — dashboard
* **pytest** — automated testing
* **Git** — version control

---

## 25. Project Learning Outcomes

By completing EdgeML, the project demonstrates experience with:

### Python

* modules
* packages
* classes
* type hints
* command-line execution
* project structure

### Data engineering

* CSV processing
* schema validation
* timestamp handling
* missing-data validation
* data-quality constraints

### Statistics

* rolling statistics
* standard deviation
* z-scores
* baseline construction

### Machine Learning

* unsupervised learning
* anomaly detection
* supervised classification
* Random Forest
* Isolation Forest
* model evaluation

### ML Engineering

* feature engineering
* reusable pipelines
* model evaluation
* reproducibility
* testing
* interpretability

### Software Engineering

* modular architecture
* automated tests
* REST API
* dashboard development
* dependency management
* Git-based incremental development

---

## 26. Version

Current release:

```text
EdgeML v1.0.0
```

This release represents a complete educational ML pipeline built around synthetic machine sensor data.

---

## 27. License

This project is intended as an educational and portfolio piece and is licensed under the MIT License, meaning you're free to use, modify, and distribute the code — see the [LICENSE](LICENSE) file for the full text.