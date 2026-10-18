import json
import threading

import pytest

from copilot import Config
from copilot.tools import Tool, ToolRegistry, validate_args

KEY = "idem-key-0001"


def refund(reg, oid="ORD-1001", amount=12900, key=KEY, reason="device arrived broken", ctx=None):
    return reg.call("request_refund", {"order_id": oid, "amount_cents": amount, "reason": reason,
                                       "idempotency_key": key}, ctx)


def code(res):
    return None if res["ok"] else res["error"]["code"]


# -- schema validation ---------------------------------------------------------------------------
@pytest.mark.parametrize("args,needle", [
    ({}, "missing required argument 'order_id'"),
    ({"order_id": 5}, "must be of type string"),
    ({"order_id": "ORD-1001", "x": 1}, "unexpected argument 'x'"),
    ({"order_id": "A"}, "shorter than 5"),
])
def test_lookup_order_rejects_bad_arguments(registry, args, needle):
    res = registry.call("lookup_order", args)
    assert code(res) == "invalid_arguments" and needle in res["error"]["message"]


def test_validate_args_types_enum_minimum():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}, "e": {"type": "string", "enum": ["a"]}},
              "required": ["n"]}
    assert validate_args({"n": 1, "e": "a"}, schema) == []
    assert validate_args({"n": True}, schema)  # bool is not an integer
    assert validate_args({"n": 1.5}, schema)
    assert validate_args({"n": 0}, schema)
    assert validate_args({"n": 1, "e": "b"}, schema)
    assert validate_args("nope", schema)


def test_unknown_tool_is_a_structured_error(registry):
    assert code(registry.call("drop_tables", {})) == "unknown_tool"


def test_tool_exception_becomes_internal_error_not_raise(registry):
    def boom(a, c):
        raise RuntimeError("secret internals")

    registry.register(Tool("boom", "x", {"type": "object", "properties": {}}, boom))
    res = registry.call("boom", {})
    assert code(res) == "internal_error" and "secret internals" not in json.dumps(res)


# -- lookup / search ---------------------------------------------------------------------------------
def test_lookup_order_ok(registry):
    res = registry.call("lookup_order", {"order_id": "ORD-1001"})
    assert res["ok"] and res["result"]["status"] == "delivered" and res["result"]["total_cents"] == 12900


def test_lookup_order_not_found(registry):
    assert code(registry.call("lookup_order", {"order_id": "ORD-9999"})) == "order_not_found"


def test_lookup_order_enforces_customer_scope(registry):
    assert code(registry.call("lookup_order", {"order_id": "ORD-1001"}, {"customer_id": "CUST-002"})) == "forbidden"
    assert registry.call("lookup_order", {"order_id": "ORD-1001"}, {"customer_id": "CUST-001"})["ok"]


def test_orders_dataset_is_synthetic_and_integer_cents(registry):
    assert len(registry.orders) == 60
    blob = json.dumps(list(registry.orders.values()))
    assert all(isinstance(o["total_cents"], int) for o in registry.orders.values())
    assert "@" not in blob.replace(".example.test", "").replace("ops@", "")  # only example.test mailboxes


def test_search_kb_returns_passages_with_ids(registry):
    res = registry.call("search_kb", {"query": "reset password"})
    assert res["ok"] and res["result"]["passages"][0]["doc_id"] == "password-reset"


def test_search_kb_area_filter_and_enum(registry):
    res = registry.call("search_kb", {"query": "limit", "area": "api"})
    assert all(p["area"] == "api" for p in res["result"]["passages"])
    assert code(registry.call("search_kb", {"query": "limit", "area": "cooking"})) == "invalid_arguments"


# -- refund policy -----------------------------------------------------------------------------------
def test_refund_happy_path_and_reflected_in_lookup(registry):
    res = refund(registry)
    assert res["ok"] and res["result"]["status"] == "approved" and res["result"]["replayed"] is False
    assert registry.call("lookup_order", {"order_id": "ORD-1001"})["result"]["refunded_cents"] == 12900


def test_refund_requires_delivered_order(registry):
    assert code(refund(registry, "ORD-1003", 100)) == "order_not_delivered"
    assert code(refund(registry, "ORD-1006", 100)) == "order_not_delivered"


def test_refund_only_once_per_order(registry):
    assert code(refund(registry, "ORD-1004", 100)) == "already_refunded"  # seeded prior refund
    assert refund(registry, key="idem-key-0001")["ok"]
    assert code(refund(registry, amount=100, key="idem-key-0002")) == "already_refunded"


def test_refund_amount_cannot_exceed_total(registry):
    assert code(refund(registry, "ORD-1007", 4501)) == "amount_exceeds_total"
    assert refund(registry, "ORD-1007", 4500)["ok"]


def test_refund_policy_max_amount_boundary(retriever):
    reg = ToolRegistry(Config(refund_max_cents=12899), retriever)
    assert code(refund(reg, "ORD-1001", 12900)) == "amount_exceeds_policy_limit"
    reg2 = ToolRegistry(Config(refund_max_cents=12900), retriever)
    assert refund(reg2, "ORD-1001", 12900)["ok"]
    assert code(refund(ToolRegistry(retriever=retriever), "ORD-1005", 60000)) == "amount_exceeds_policy_limit"


def test_refund_window_boundary(retriever):
    # ORD-1001 was delivered 2026-10-10; "today" is 2026-10-18 => 8 days old
    assert code(refund(ToolRegistry(Config(refund_window_days=7), retriever))) == "outside_refund_window"
    assert refund(ToolRegistry(Config(refund_window_days=8), retriever))["ok"]


def test_refund_outside_default_window(registry):
    res = refund(registry, "ORD-1002", 100)
    assert code(res) == "outside_refund_window" and "78 days" in res["error"]["message"]


def test_refund_unknown_order_and_wrong_customer(registry):
    assert code(refund(registry, "ORD-9999", 100)) == "order_not_found"
    assert code(refund(registry, ctx={"customer_id": "CUST-009"})) == "forbidden"


def test_refund_idempotent_retry_returns_original_result(registry):
    first = refund(registry)
    second = refund(registry)
    assert second["ok"] and second["result"]["replayed"] is True
    assert second["result"]["refund_id"] == first["result"]["refund_id"]
    assert sum(1 for a in registry.audit if a["tool"] == "request_refund" and a["ok"]) == 2  # both calls audited ok
    assert len(registry._refunds) == len([o for o in registry.orders.values() if o["refunded_cents"]]) + 1


def test_idempotency_key_reuse_with_different_args_is_rejected(registry):
    refund(registry)
    assert code(refund(registry, amount=100)) == "idempotency_key_reuse"


def test_refund_args_validated(registry):
    assert code(registry.call("request_refund", {"order_id": "ORD-1001"})) == "invalid_arguments"
    assert code(refund(registry, amount=0)) == "invalid_arguments"
    assert code(refund(registry, amount=10.5)) == "invalid_arguments"
    assert code(refund(registry, key="short")) == "invalid_arguments"


def test_every_call_is_audited_including_failures(registry, tmp_path):
    reg = ToolRegistry(registry.config, registry.retriever, audit_path=tmp_path / "audit.jsonl")
    reg.call("lookup_order", {"order_id": "ORD-9999"})
    refund(reg, "ORD-1003", 100)
    reg.call("nope", {})
    assert [(a["tool"], a["ok"]) for a in reg.audit] == [("lookup_order", False), ("request_refund", False), ("nope", False)]
    assert reg.audit[1]["error_code"] == "order_not_delivered" and reg.audit[1]["args"]["order_id"] == "ORD-1003"
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[1])["tool"] == "request_refund"


def test_concurrent_refunds_on_one_order_commit_exactly_once(registry):
    results = []
    barrier = threading.Barrier(10)

    def go(i):
        barrier.wait()
        results.append(refund(registry, amount=1000, key=f"idem-key-{i:04d}"))

    ts = [threading.Thread(target=go, args=(i,)) for i in range(10)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(r["ok"] for r in results) == 1
    assert sorted(code(r) for r in results if not r["ok"]) == ["already_refunded"] * 9
