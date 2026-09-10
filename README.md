# Intelligent Predictive Maintenance System for Industrial Equipment

An end-to-end machine-learning system that predicts industrial machine failure risk from sensor data and converts the prediction into a practical maintenance recommendation.

## What this project demonstrates
- Exploratory data analysis and failure-pattern analysis
- Data preprocessing with median imputation, one-hot encoding and robust IQR outlier clipping
- Physics-informed feature engineering
- Comparison of Logistic Regression, Random Forest and XGBoost
- Class-imbalance handling and threshold selection
- Production-ready FastAPI inference service
- Transparent maintenance recommendation rules
- Automated API tests
- Docker/Docker Compose deployment
- Technical report and internship demo script

## Dataset
The project follows the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository (Dataset 601, DOI 10.24432/C5HS5C, CC BY 4.0).

Because the build environment could not retrieve the original CSV, `data/ai4i2020_reproducible.csv` is a deterministic synthetic fallback aligned to the UCI schema and published failure-mode rules. It is clearly labeled as such. For strict evaluation against the original dataset, download `ai4i2020.csv` from UCI and retrain the pipeline.

Official source: https://archive.ics.uci.edu/dataset/601/ai4i

## Model results
Held-out stratified test set (80/20, random_state=42), using the reproducible fallback dataset:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7700 | 0.1084 | 0.7714 | 0.1901 | 0.8340 | 0.3447 |
| **Random Forest** | **0.9925** | **0.9661** | **0.8143** | **0.8837** | **0.9904** | **0.9072** |
| XGBoost | 0.9855 | 0.7808 | 0.8143 | 0.7972 | 0.9848 | 0.8720 |

Random Forest is the selected production model. The saved operating decision threshold is 0.10 to favor failure detection.

## Run locally
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```
Open `http://127.0.0.1:8000/docs` for the interactive API.

### Example request
```json
{"machine_id":"DEMO-001","product_type":"M","air_temperature":300.0,"process_temperature":309.0,"rotational_speed":1300,"torque":55.0,"tool_wear":210}
```

## Tests
```bash
pytest -q tests
```
Expected result: **2 passed**.

## Docker
```bash
docker compose up --build
```
Then open `http://127.0.0.1:8000/docs`.

## Maintenance recommendation logic
- Probability >= 0.80: CRITICAL — immediate inspection / isolate equipment
- Probability >= 0.50: HIGH — schedule preventive maintenance at the earliest window
- Probability >= 0.25: MEDIUM — increase monitoring and plan inspection
- Otherwise: LOW — routine monitoring

Additional rules flag high tool wear, low temperature gap at low RPM, abnormal calculated power and high wear × torque load.

## Project structure
```text
predictive_maintenance/
├── data/                 # dataset and documentation
├── notebooks/            # EDA, preprocessing, feature engineering, training walkthroughs
├── src/                  # reusable preprocessing, maintenance logic and training
├── models/               # trained model and metadata
├── api/                  # FastAPI service
├── tests/                # API tests
├── reports/              # charts, metrics and technical report
├── demo/                 # internship presentation script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Limitations and next steps
The synthetic fallback is for reproducible offline demonstration, not a substitute for plant data. A real deployment should calibrate thresholds against downtime/maintenance costs, validate on time-ordered production data, monitor drift, integrate with a CMMS, and add authenticated alerting and observability.

## Dataset citation
Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
