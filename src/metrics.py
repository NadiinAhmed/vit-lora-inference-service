"""Prometheus metrics, exposed in real time at GET /metrics.

Three kinds of metric, each answering one question:
  Counter   - "how many so far?"       (only goes up; Prometheus derives rates from it)
  Histogram - "how are values spread?" (counts per bucket -> p50 / p95 / p99)
  Gauge     - "what is it right now?"  (can go up and down)

Process CPU and memory (process_cpu_seconds_total, process_resident_memory_bytes)
are added automatically by prometheus_client on Linux, i.e. inside Docker.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram

# --- Infrastructure: is the service healthy? -------------------------------
HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests by endpoint and status code (error rate = 4xx/5xx share).",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "End-to-end request latency, including upload and decoding.",
    ["method", "path"],
)
MODEL_LOADED = Gauge("model_loaded", "1 when the model is loaded and serving.")
MODEL_INFO = Gauge("model_info", "Always 1; the label shows which model version is serving.", ["model_version"])

# --- Model: is the model behaving as usual? ---------------------------------
INFERENCE_LATENCY = Histogram("model_inference_seconds", "Model forward-pass latency only.")
PREDICTIONS = Counter("model_predictions_total", "Predictions per predicted class.", ["label"])
CONFIDENCE = Histogram(
    "model_prediction_confidence", "Confidence of the top prediction.",
    buckets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)


def observe_prediction(label: str, confidence: float, inference_seconds: float) -> None:
    PREDICTIONS.labels(label=label).inc()
    CONFIDENCE.observe(confidence)
    INFERENCE_LATENCY.observe(inference_seconds)


async def track_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """HTTP middleware: count and time every request except scrapes of /metrics itself."""
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    status_code = 500  # stays 500 if the handler raises an unexpected exception
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        # Use the route template ("/predict"), never the raw URL: random URLs from scanners
        # would otherwise create unlimited label values ("high cardinality").
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(method=request.method, path=path, status=str(status_code)).inc()
        HTTP_LATENCY.labels(method=request.method, path=path).observe(time.perf_counter() - start)
