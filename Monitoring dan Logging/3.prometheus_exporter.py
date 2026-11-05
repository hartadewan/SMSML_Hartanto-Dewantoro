import os
import time

import requests
from prometheus_client import Counter, Gauge, Histogram, start_http_server

SERVING_URL = os.environ.get("SERVING_URL", "http://inference:8080")
CHECK_INTERVAL = float(os.environ.get("CHECK_INTERVAL", "5"))

UP = Gauge("credit_serving_up", "Serving endpoint reachable")
CHECKS = Counter("credit_serving_checks_total", "Serving health checks")
FAILURES = Counter("credit_serving_check_failures_total", "Failed health checks")
CHECK_LATENCY = Histogram(
    "credit_serving_check_latency_seconds",
    "Health-check latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


def check():
    started = time.perf_counter()
    CHECKS.inc()
    try:
        response = requests.get(f"{SERVING_URL}/health", timeout=2)
        response.raise_for_status()
        UP.set(1)
    except requests.RequestException:
        UP.set(0)
        FAILURES.inc()
    finally:
        CHECK_LATENCY.observe(time.perf_counter() - started)


if __name__ == "__main__":
    start_http_server(8000)
    while True:
        check()
        time.sleep(CHECK_INTERVAL)

