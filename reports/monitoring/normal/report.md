# Model Monitoring Report

**Overall status: 🟢 OK**

- Model version: `model-v1.0.0`
- Generated: 2026-10-03T11:12:16+00:00
- Predictions analysed: 379 (reference: 379 test images)
- Period: 2026-10-03T10:39:44.119757+00:00 → 2026-10-03T11:07:17.743382+00:00
- Model latency: p50 253.25 ms, p95 375.8 ms

PSI thresholds: < 0.10 OK, 0.10–0.25 WARNING, > 0.25 ALERT. KS = largest gap between the two cumulative distributions (0–1).

## 1. Data drift — 🟢 OK

| Image statistic | Reference mean | Production mean | PSI | KS | Status |
|---|---|---|---|---|---|
| brightness | 0.6421 | 0.6436 | 0.0446 | 0.0396 | 🟢 OK |
| contrast | 0.1798 | 0.177 | 0.0805 | 0.0501 | 🟢 OK |
| saturation | 0.1794 | 0.1783 | 0.0767 | 0.0475 | 🟢 OK |
| aspect_ratio | 1.3333 | 1.3333 | 0.0 | 0.0 | 🟢 OK |

## 2. Prediction drift — 🟢 OK

Predicted-class distribution: PSI 0.0037 (🟢 OK)

| Class | Reference share | Production share |
|---|---|---|
| cardboard | 16.1% | 15.8% |
| glass | 20.8% | 18.7% |
| metal | 16.1% | 16.6% |
| paper | 21.9% | 23.5% |
| plastic | 19.5% | 19.5% |
| trash | 5.5% | 5.8% |

Confidence: PSI 0.0249, KS 0.0449 (🟢 OK)

| | Reference | Production |
|---|---|---|
| Mean confidence | 0.9627 | 0.959 |
| Share below 0.7 | 3.7% | 4.8% |

## 3. Model performance (ground-truth feedback) — 🟢 OK

Labelled predictions: 379 (100.0% of predictions)

Accuracy drop vs reference: -0.0422 (WARNING > 0.05, ALERT > 0.1)

| Metric | Reference | Production |
|---|---|---|
| Accuracy | 0.9182 | 0.9604 |
| Macro-F1 | 0.8839 | 0.9513 |
| F1 cardboard | 0.9587 | 0.9667 |
| F1 glass | 0.9481 | 0.9589 |
| F1 metal | 0.9431 | 0.976 |
| F1 paper | 0.9302 | 0.9775 |
| F1 plastic | 0.9041 | 0.9452 |
| F1 trash | 0.619 | 0.8837 |
