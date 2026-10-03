# Model Monitoring Report

**Overall status: 🔴 ALERT**

- Model version: `model-v1.0.0`
- Generated: 2026-10-03T11:43:53+00:00
- Predictions analysed: 379 (reference: 379 test images)
- Period: 2026-10-03T11:14:51.904887+00:00 → 2026-10-03T11:41:40.575413+00:00
- Model latency: p50 129.61 ms, p95 180.32 ms

PSI thresholds: < 0.10 OK, 0.10–0.25 WARNING, > 0.25 ALERT. KS = largest gap between the two cumulative distributions (0–1).

## 1. Data drift — 🔴 ALERT

| Image statistic | Reference mean | Production mean | PSI | KS | Status |
|---|---|---|---|---|---|
| brightness | 0.6421 | 0.2877 | 15.3262 | 0.9974 | 🔴 ALERT |
| contrast | 0.1798 | 0.0796 | 5.7197 | 0.7757 | 🔴 ALERT |
| saturation | 0.1794 | 0.1797 | 0.0724 | 0.0501 | 🟢 OK |
| aspect_ratio | 1.3333 | 1.3333 | 0.0 | 0.0 | 🟢 OK |

## 2. Prediction drift — 🔴 ALERT

Predicted-class distribution: PSI 0.0246 (🟢 OK)

| Class | Reference share | Production share |
|---|---|---|
| cardboard | 16.1% | 17.2% |
| glass | 20.8% | 17.4% |
| metal | 16.1% | 16.9% |
| paper | 21.9% | 21.1% |
| plastic | 19.5% | 18.5% |
| trash | 5.5% | 9.0% |

Confidence: PSI 0.561, KS 0.2612 (🔴 ALERT)

| | Reference | Production |
|---|---|---|
| Mean confidence | 0.9627 | 0.9238 |
| Share below 0.7 | 3.7% | 9.2% |

## 3. Model performance (ground-truth feedback) — 🟢 OK

Labelled predictions: 379 (100.0% of predictions)

Accuracy drop vs reference: +0.0237 (WARNING > 0.05, ALERT > 0.1)

| Metric | Reference | Production |
|---|---|---|
| Accuracy | 0.9182 | 0.8945 |
| Macro-F1 | 0.8839 | 0.8642 |
| F1 cardboard | 0.9587 | 0.928 |
| F1 glass | 0.9481 | 0.922 |
| F1 metal | 0.9431 | 0.9524 |
| F1 paper | 0.9302 | 0.8994 |
| F1 plastic | 0.9041 | 0.9014 |
| F1 trash | 0.619 | 0.5818 |
