# 5–7 Minute Demo Script

## 0:00–0:45 — Problem
"This project addresses unplanned industrial equipment downtime. The objective is to predict machine failure risk from sensor readings early enough to support planned maintenance."

## 0:45–1:30 — Dataset and processing
Show `data/` and the EDA notebook. Explain the AI4I schema, failure target, class imbalance, missing-value handling, IQR outlier clipping, and engineered physical features.

## 1:30–2:30 — Model training
Show the model-comparison table. Explain that accuracy alone is insufficient for rare failures, so PR-AUC, recall and F1 were emphasized. The Random Forest was selected as the production model.

## 2:30–3:30 — API
Run the server and open `/docs`. Demonstrate `POST /predict` with a machine showing high tool wear and a high load condition.

## 3:30–4:30 — Prediction and recommendation
Show the returned failure probability, risk level, recommended action, and inspection items. Explain that the ML prediction is combined with transparent maintenance rules.

## 4:30–5:30 — Docker
Build and run the Docker image. Refresh `/docs` and run the same request to prove the service works inside the container.

## 5:30–6:15 — Results
Show the evaluation plots and metrics. Highlight the held-out test results and explain the trade-off between false alarms and missed failures.

## 6:15–6:45 — Conclusion
"The final system turns industrial sensor data into a failure-risk score and a practical maintenance action. In a real factory, the next step would be calibration against actual maintenance costs and integration with the plant's CMMS or alerting platform."
