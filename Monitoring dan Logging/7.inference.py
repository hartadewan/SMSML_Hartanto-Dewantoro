import os
import time
from pathlib import Path

import mlflow
import pandas as pd
from flask import Flask, jsonify, request
from prometheus_client import Counter, Gauge, Histogram, start_http_server

MODEL_URI = os.environ["MLFLOW_MODEL_URI"]
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns")
MODEL_VERSION = os.environ.get("MODEL_VERSION", MODEL_URI)

mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)
app = Flask(__name__)

REQUESTS = Counter(
    "credit_inference_requests_total",
    "Total inference requests",
    ["method", "endpoint", "status"],
)
ERRORS = Counter(
    "credit_inference_errors_total",
    "Total inference errors",
    ["endpoint", "error_type"],
)
LATENCY = Histogram(
    "credit_inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
ROWS = Counter(
    "credit_inference_input_rows_total",
    "Total input rows received",
)
PREDICTIONS = Counter(
    "credit_inference_predictions_total",
    "Total predictions returned",
    ["prediction"],
)
REQUEST_SIZE = Histogram(
    "credit_inference_input_rows_per_request",
    "Input rows per request",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)
MODEL_LOADED = Gauge(
    "credit_model_loaded",
    "Whether the configured model loaded successfully",
)
MODEL_INFO = Gauge(
    "credit_model_info",
    "Model metadata; value is always one",
    ["model_uri", "model_version"],
)
HEALTH_CHECKS = Counter(
    "credit_inference_health_checks_total",
    "Health endpoint calls",
)

MODEL_LOADED.set(1)
MODEL_INFO.labels(MODEL_URI, MODEL_VERSION).set(1)


@app.get("/ping")
def ping():
    return "OK", 200


@app.get("/health")
def health():
    HEALTH_CHECKS.inc()
    return jsonify({"status": "ok", "model": MODEL_URI}), 200


@app.post("/invocations")
def invocations():
    started = time.perf_counter()
    endpoint = "/invocations"
    status = "500"
    try:
        body = request.get_json(force=True)
        split = body["dataframe_split"]
        frame = pd.DataFrame(data=split["data"], columns=split["columns"])
        if frame.empty:
            raise ValueError("dataframe_split tidak boleh kosong")
        REQUEST_SIZE.observe(len(frame))
        ROWS.inc(len(frame))
        values = model.predict(frame)
        predictions = [value.item() if hasattr(value, "item") else value for value in values]
        for value in predictions:
            PREDICTIONS.labels(str(value)).inc()
        status = "200"
        return jsonify({"predictions": predictions})
    except (KeyError, TypeError, ValueError) as exc:
        ERRORS.labels(endpoint, type(exc).__name__).inc()
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        ERRORS.labels(endpoint, type(exc).__name__).inc()
        return jsonify({"error": "inference failed"}), 500
    finally:
        REQUESTS.labels(request.method, endpoint, status).inc()
        LATENCY.labels(endpoint).observe(time.perf_counter() - started)


if __name__ == "__main__":
    start_http_server(8001)
    app.run(host="0.0.0.0", port=8080)

