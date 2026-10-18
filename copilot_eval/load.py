"""Load test against the real in-process HTTP server (stdlib threads + http.client)."""
from __future__ import annotations

import http.client
import itertools
import json
import threading
import time

from copilot import Settings, make_server

from .golden import load_golden, select
from .metrics import _pct


def default_questions() -> list[dict]:
    """Dev questions except allowed refunds (the server shares one refund ledger, so repeats would just be denials)."""
    return [{"question": q["question"], "customer_id": q.get("customer_id")}
            for q in select(load_golden(), "dev") if q["type"] != "refund_allowed"]


def run_load(clients: int = 8, requests: int = 400, rate_capacity: float = 1_000_000.0, rate_refill_per_s: float = 1_000_000.0,
             questions: list[dict] | None = None, timeout: float = 15.0) -> dict:
    qs = questions or default_questions()
    srv = make_server(settings=Settings(auth_token=None, rate_capacity=rate_capacity, rate_refill_per_s=rate_refill_per_s,
                                        trace_dir=None), port=0).start()
    counter, lock = itertools.count(), threading.Lock()
    lat: list[float] = []
    codes: dict[str, int] = {}

    def worker():
        while True:
            with lock:
                i = next(counter)
            if i >= requests:
                return
            q = qs[i % len(qs)]
            body = json.dumps({k: v for k, v in q.items() if v is not None})
            t0 = time.perf_counter()
            try:
                c = http.client.HTTPConnection(srv.url.split("//")[1], timeout=timeout)
                c.request("POST", "/v1/ask", body, {"Content-Type": "application/json"})
                r = c.getresponse()
                r.read()
                key = str(r.status)
                c.close()
            except Exception as e:  # connection reset, timeout...
                key = type(e).__name__
            ms = (time.perf_counter() - t0) * 1000
            with lock:
                lat.append(ms)
                codes[key] = codes.get(key, 0) + 1

    try:
        t0 = time.perf_counter()
        ts = [threading.Thread(target=worker) for _ in range(clients)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        wall = time.perf_counter() - t0
    finally:
        srv.shutdown()
    n = len(lat)
    ok, limited = codes.get("200", 0), codes.get("429", 0)
    return {"clients": clients, "requests": requests, "completed": n, "wall_s": round(wall, 3),
            "throughput_rps": round(n / wall, 1), "status_counts": dict(sorted(codes.items())),
            "ok_rate": round(ok / n, 4), "rate_limited_429": round(limited / n, 4),
            "error_rate": round((n - ok - limited) / n, 4),
            "latency_ms": {"p50": _pct(lat, 0.50), "p95": _pct(lat, 0.95), "p99": _pct(lat, 0.99), "max": round(max(lat), 3)},
            "rate_capacity": rate_capacity, "rate_refill_per_s": rate_refill_per_s}
