"""Stdlib JSON API: POST /v1/ask, GET /healthz, GET /metrics."""
from __future__ import annotations

import hmac
import json
import math
import os
import re
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import Agent
from .config import Config
from .models import ModelError, RuleModel
from .retrieval import Retriever
from .tools import ToolRegistry
from .tracing import Tracer

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
KNOWN_PATHS = ("/v1/ask", "/healthz", "/metrics")
RID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


@dataclass
class Settings:
    auth_token: str | None = field(default_factory=lambda: os.environ.get("COPILOT_API_TOKEN") or None)
    max_body_bytes: int = 1_048_576
    rate_capacity: float = 30.0
    rate_refill_per_s: float = 10.0
    request_timeout_s: float = 10.0
    max_question_chars: int = 2000
    trace_dir: str | None = None


class RateLimiter:
    """Per-client token bucket."""

    def __init__(self, capacity: float, refill_per_s: float, clock=time.monotonic):
        self.capacity, self.refill, self.clock = capacity, refill_per_s, clock
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        with self._lock:
            now = self.clock()
            tokens, last = self._b.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= 1:
                self._b[key] = (tokens - 1, now)
                return True, 0.0
            self._b[key] = (tokens, now)
            return False, (1 - tokens) / self.refill if self.refill > 0 else 3600.0


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests: dict[tuple, int] = {}
        self.errors: dict[tuple, int] = {}
        self.bucket_counts = [0] * (len(BUCKETS) + 1)
        self.dur_sum = 0.0
        self.dur_count = 0

    def observe(self, method: str, path: str, status: int, seconds: float) -> None:
        path = path if path in KNOWN_PATHS else "other"
        with self._lock:
            k = (method, path, str(status))
            self.requests[k] = self.requests.get(k, 0) + 1
            if status >= 400:
                self.errors[(path, str(status))] = self.errors.get((path, str(status)), 0) + 1
            self.dur_sum += seconds
            self.dur_count += 1
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    self.bucket_counts[i] += 1
            self.bucket_counts[-1] += 1

    def render(self) -> str:
        with self._lock:
            out = ["# TYPE copilot_requests_total counter"]
            out += [f'copilot_requests_total{{method="{m}",path="{p}",status="{s}"}} {n}'
                    for (m, p, s), n in sorted(self.requests.items())]
            out.append("# TYPE copilot_errors_total counter")
            out += [f'copilot_errors_total{{path="{p}",status="{s}"}} {n}' for (p, s), n in sorted(self.errors.items())]
            out.append("# TYPE copilot_request_duration_seconds histogram")
            out += [f'copilot_request_duration_seconds_bucket{{le="{b}"}} {c}' for b, c in zip(BUCKETS, self.bucket_counts)]
            out.append(f'copilot_request_duration_seconds_bucket{{le="+Inf"}} {self.bucket_counts[-1]}')
            out.append(f"copilot_request_duration_seconds_sum {self.dur_sum:.6f}")
            out.append(f"copilot_request_duration_seconds_count {self.dur_count}")
            return "\n".join(out) + "\n"


class CopilotApp:
    """Shared service state: one retriever and one tool registry (so refund idempotency spans requests)."""

    def __init__(self, config: Config | None = None, settings: Settings | None = None, model_factory=None,
                 guardrails: bool = True):
        self.config = config or Config()
        self.settings = settings or Settings()
        self.retriever = Retriever.from_config(self.config)
        self.registry = ToolRegistry(self.config, self.retriever)
        self.model_factory = model_factory or (lambda: RuleModel(min_score=self.config.min_score))
        self.guardrails = guardrails
        self.tracer = Tracer(self.settings.trace_dir)
        self.metrics = Metrics()
        self.limiter = RateLimiter(self.settings.rate_capacity, self.settings.rate_refill_per_s)

    def ask(self, question: str, customer_id: str | None) -> dict:
        agent = Agent(self.config, self.model_factory(), self.registry, self.retriever, self.tracer, self.guardrails)
        r = agent.run(question, customer_id)
        return {"answer": r.answer, "citations": r.citations, "tool_calls": r.tool_calls, "trace_id": r.trace_id,
                "stop_reason": r.stop_reason, "abstained": r.abstained, "flags": r.flags}


class Handler(BaseHTTPRequestHandler):
    app: CopilotApp  # injected by CopilotServer
    server_version = "NimbusCopilot/1.0"

    def log_message(self, *args):  # silence default stderr access log
        pass

    # -- plumbing -----------------------------------------------------------------------------
    def _begin(self):
        self._t0 = time.perf_counter()
        rid = self.headers.get("X-Request-Id", "")
        self.rid = rid if RID_RE.match(rid) else uuid.uuid4().hex
        self.path_only = self.path.split("?", 1)[0]

    def _send(self, status: int, body, ctype: str = "application/json", headers: dict | None = None):
        data = (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Request-Id", self.rid)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)
        self.app.metrics.observe(self.command, self.path_only, status, time.perf_counter() - self._t0)

    def _error(self, status: int, code: str, message: str, headers: dict | None = None):
        self._send(status, {"error": {"code": code, "message": message, "request_id": self.rid}}, headers=headers)

    # -- routes -------------------------------------------------------------------------------
    def do_GET(self):
        self._begin()
        if self.path_only == "/healthz":
            self._send(200, {"status": "ok", "version": self.app.config.version, "config_hash": self.app.config.config_hash()})
        elif self.path_only == "/metrics":
            self._send(200, self.app.metrics.render(), "text/plain; version=0.0.4")
        else:
            self._error(404, "not_found", f"no route for GET {self.path_only}")

    def do_POST(self):
        self._begin()
        s = self.app.settings
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._error(400, "bad_content_length", "Content-Length must be an integer")
        if n > s.max_body_bytes:
            remaining = min(n, 16 * s.max_body_bytes)  # drain so the client sees the 413, not a reset
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True
            return self._error(413, "payload_too_large", f"body exceeds {s.max_body_bytes} bytes")
        raw = self.rfile.read(n) if n > 0 else b""
        if self.path_only != "/v1/ask":
            return self._error(404, "not_found", f"no route for POST {self.path_only}")
        if s.auth_token:
            given = self.headers.get("Authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {s.auth_token}".encode()):
                return self._error(401, "unauthorized", "missing or invalid bearer token",
                                   {"WWW-Authenticate": "Bearer"})
        allowed, retry = self.app.limiter.allow(self.client_address[0])
        if not allowed:
            return self._error(429, "rate_limited", "too many requests", {"Retry-After": str(max(1, math.ceil(retry)))})
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._error(400, "invalid_json", "request body must be valid JSON")
        q = body.get("question") if isinstance(body, dict) else None
        cust = body.get("customer_id") if isinstance(body, dict) else None
        if not isinstance(q, str) or not q.strip() or len(q) > s.max_question_chars:
            return self._error(400, "invalid_request", f"'question' must be a non-empty string up to {s.max_question_chars} chars")
        if cust is not None and not isinstance(cust, str):
            return self._error(400, "invalid_request", "'customer_id' must be a string")
        box: dict = {}

        def work():
            try:
                box["r"] = self.app.ask(q, cust)
            except BaseException as e:  # noqa: BLE001 - mapped to a structured error below
                box["e"] = e

        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(s.request_timeout_s)
        if t.is_alive():
            return self._error(504, "timeout", f"request exceeded {s.request_timeout_s}s")
        if "e" in box:
            if isinstance(box["e"], ModelError):
                return self._error(502, "model_error", "the model backend failed")
            return self._error(500, "internal_error", "internal error")  # details stay out of the response
        self._send(200, {**box["r"], "request_id": self.rid})


class _HTTPServer(ThreadingHTTPServer):
    # stdlib default is 5: with 32 concurrent clients on Windows, dropped SYNs were retried after ~1 s (p95 jumped to 1 s)
    request_queue_size = 128


class CopilotServer:
    def __init__(self, app: CopilotApp, host: str = "127.0.0.1", port: int = 0):
        handler = type("BoundHandler", (Handler,), {"app": app})
        self.app = app
        self.httpd = _HTTPServer((host, port), handler)
        self.httpd.daemon_threads = False  # server_close() joins in-flight requests (graceful drain)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "CopilotServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread:
            self._thread.join(5)

    def serve_forever(self) -> None:
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        self.start()
        stop.wait()
        self.shutdown()


def make_server(config: Config | None = None, settings: Settings | None = None, model_factory=None,
                host: str = "127.0.0.1", port: int = 0, guardrails: bool = True) -> CopilotServer:
    """Build (not start) a server; port=0 picks a free port. Call .start() / .shutdown()."""
    return CopilotServer(CopilotApp(config, settings, model_factory, guardrails), host, port)
