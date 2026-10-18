import json
import re
import threading
import time
import urllib.error
import urllib.request

import pytest

from copilot.models import ModelError
from copilot.server import RateLimiter, Settings
from helpers import http, say


def S(**kw):
    return Settings(**{"auth_token": None, "rate_capacity": 1000.0, **kw})


class SlowModel:
    name = "slow"

    def __init__(self, delay):
        self.delay = delay

    def generate(self, messages, tools):
        time.sleep(self.delay)
        return say("done")


class RaisingModel:
    name = "boom"

    def __init__(self, exc):
        self.exc = exc

    def generate(self, messages, tools):
        raise self.exc


def ask(srv, q="how do I reset my password", **kw):
    return http(srv.url + "/v1/ask", "POST", {"question": q, **kw.pop("body", {})}, **kw)


def test_healthz(server_factory):
    srv = server_factory(settings=S())
    status, _, body = http(srv.url + "/healthz")
    assert status == 200 and json.loads(body)["status"] == "ok" and len(json.loads(body)["config_hash"]) == 12


def test_ask_returns_answer_citations_tool_calls_and_trace_id(server_factory):
    srv = server_factory(settings=S())
    status, headers, body = ask(srv)
    d = json.loads(body)
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert d["citations"][0]["doc_id"] == "password-reset" and d["tool_calls"][0]["name"] == "search_kb"
    assert re.fullmatch(r"[0-9a-f]{16}", d["trace_id"]) and d["request_id"] == headers["X-Request-Id"]


def test_request_id_is_echoed_or_generated(server_factory):
    srv = server_factory(settings=S())
    _, h, body = ask(srv, headers={"X-Request-Id": "req-abc.123"})
    assert h["X-Request-Id"] == "req-abc.123" and json.loads(body)["request_id"] == "req-abc.123"
    _, h2, _ = ask(srv, headers={"X-Request-Id": "bad id with spaces!"})
    assert h2["X-Request-Id"] != "bad id with spaces!" and len(h2["X-Request-Id"]) == 32
    assert http(srv.url + "/healthz", headers={"X-Request-Id": "hz-1"})[1]["X-Request-Id"] == "hz-1"


@pytest.mark.parametrize("raw,code", [
    (b"{not json", "invalid_json"), (b"", "invalid_json"), (b"[]", "invalid_request"),
    (b'{"question": ""}', "invalid_request"), (b'{"question": 5}', "invalid_request"),
    (b'{"question": "hi", "customer_id": 7}', "invalid_request"),
])
def test_400_structured_errors(server_factory, raw, code):
    srv = server_factory(settings=S())
    status, h, body = http(srv.url + "/v1/ask", "POST", raw=raw)
    err = json.loads(body)["error"]
    assert status == 400 and err["code"] == code and err["request_id"] == h["X-Request-Id"]


def test_401_without_or_with_wrong_token_and_200_with_right_one(server_factory):
    srv = server_factory(settings=S(auth_token="s3cret"))
    s1, h1, b1 = ask(srv)
    assert s1 == 401 and h1["WWW-Authenticate"] == "Bearer" and json.loads(b1)["error"]["code"] == "unauthorized"
    assert ask(srv, headers={"Authorization": "Bearer nope"})[0] == 401
    assert ask(srv, headers={"Authorization": "Bearer s3cret"})[0] == 200
    assert http(srv.url + "/healthz")[0] == 200  # health stays open


def test_404_for_unknown_routes(server_factory):
    srv = server_factory(settings=S())
    s1, _, b1 = http(srv.url + "/nope")
    assert s1 == 404 and json.loads(b1)["error"]["code"] == "not_found"
    assert http(srv.url + "/v1/other", "POST", {"question": "x"})[0] == 404


def test_413_when_body_exceeds_one_megabyte_and_boundary_is_allowed(server_factory):
    srv = server_factory(settings=S())
    s1, _, b1 = http(srv.url + "/v1/ask", "POST", raw=b"x" * (1_048_576 + 1))
    assert s1 == 413 and json.loads(b1)["error"]["code"] == "payload_too_large"
    assert http(srv.url + "/v1/ask", "POST", raw=b"x" * 1_048_576)[0] == 400  # exactly at the limit: parsed, not 413


def test_429_after_bucket_is_drained_with_retry_after(server_factory):
    srv = server_factory(settings=S(rate_capacity=3.0, rate_refill_per_s=0.0001))
    assert [ask(srv)[0] for _ in range(3)] == [200, 200, 200]
    status, h, body = ask(srv)
    assert status == 429 and int(h["Retry-After"]) >= 1 and json.loads(body)["error"]["code"] == "rate_limited"
    assert http(srv.url + "/healthz")[0] == 200  # only /v1/ask is limited


def test_token_bucket_refills_with_injected_clock():
    now = [0.0]
    rl = RateLimiter(2, 1.0, clock=lambda: now[0])
    assert [rl.allow("a")[0] for _ in range(3)] == [True, True, False]
    assert rl.allow("b")[0] is True  # separate client, separate bucket
    now[0] = 1.0
    assert rl.allow("a")[0] is True and rl.allow("a")[0] is False


def test_500_does_not_leak_internals(server_factory):
    srv = server_factory(settings=S(), model_factory=lambda: RaisingModel(RuntimeError("boom secret detail")))
    status, h, body = ask(srv)
    assert status == 500 and json.loads(body)["error"]["code"] == "internal_error"
    assert "boom secret detail" not in body and json.loads(body)["error"]["request_id"] == h["X-Request-Id"]
    assert http(srv.url + "/healthz")[0] == 200  # server survives


def test_502_when_model_backend_fails(server_factory):
    srv = server_factory(settings=S(), model_factory=lambda: RaisingModel(ModelError("HTTP 500 upstream")))
    status, _, body = ask(srv)
    assert status == 502 and json.loads(body)["error"]["code"] == "model_error" and "upstream" not in body


def test_504_when_request_exceeds_timeout(server_factory):
    srv = server_factory(settings=S(request_timeout_s=0.1), model_factory=lambda: SlowModel(0.6))
    assert ask(srv)[0] == 504


def test_metrics_text_format_and_counters(server_factory):
    srv = server_factory(settings=S())
    ask(srv)
    ask(srv)
    http(srv.url + "/v1/ask", "POST", raw=b"{bad")
    http(srv.url + "/missing")
    status, h, text = http(srv.url + "/metrics")
    assert status == 200 and h["Content-Type"].startswith("text/plain")
    assert 'copilot_requests_total{method="POST",path="/v1/ask",status="200"} 2' in text
    assert 'copilot_requests_total{method="POST",path="/v1/ask",status="400"} 1' in text
    assert 'copilot_errors_total{path="/v1/ask",status="400"} 1' in text
    assert 'copilot_errors_total{path="other",status="404"} 1' in text
    assert "copilot_request_duration_seconds_count 4" in text
    assert 'copilot_request_duration_seconds_bucket{le="+Inf"} 4' in text
    assert "# TYPE copilot_request_duration_seconds histogram" in text


def test_twenty_concurrent_requests_do_not_cross_talk(server_factory, tmp_path):
    srv = server_factory(settings=S(trace_dir=str(tmp_path)))
    orders = srv.app.registry.orders
    out, barrier = {}, threading.Barrier(20)

    def go(i):
        oid = f"ORD-{1001 + i}"
        barrier.wait()
        out[i] = http(srv.url + "/v1/ask", "POST", {"question": f"Where is order {oid}?", "customer_id": orders[oid]["customer_id"]},
                      headers={"X-Request-Id": f"rid-{i}"})

    ts = [threading.Thread(target=go, args=(i,)) for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    trace_ids = set()
    for i in range(20):
        status, h, body = out[i]
        d, oid = json.loads(body), f"ORD-{1001 + i}"
        assert status == 200 and h["X-Request-Id"] == f"rid-{i}" == d["request_id"]
        assert set(re.findall(r"ORD-\d+", d["answer"])) == {oid}
        assert d["tool_calls"][0]["arguments"] == {"order_id": oid} and d["tool_calls"][0]["ok"]
        trace_ids.add(d["trace_id"])
    assert len(trace_ids) == 20 and len(list(tmp_path.glob("*.jsonl"))) == 20


def test_graceful_shutdown_drains_in_flight_request_then_closes_port(server_factory):
    srv = server_factory(settings=S(), model_factory=lambda: SlowModel(0.4))
    res = {}
    t = threading.Thread(target=lambda: res.update(r=ask(srv)))
    t.start()
    time.sleep(0.15)
    srv.shutdown()
    t.join()
    assert res["r"][0] == 200
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(srv.url + "/healthz", timeout=2)


def test_listen_backlog_is_large_enough_for_concurrent_clients(server_factory):
    # regression: the stdlib default backlog of 5 made 32 concurrent clients wait ~1 s for a SYN retry on Windows
    from copilot.server import Settings
    srv = server_factory(settings=Settings(auth_token=None, trace_dir=None))
    assert srv.httpd.request_queue_size >= 64
