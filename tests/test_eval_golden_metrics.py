import copy
import json
import re

import pytest

from copilot import Config
from copilot_eval import build_golden
from copilot_eval.golden import GOLDEN_PATH, GoldenError, hygiene, load_golden, oracle, select
from copilot_eval.metrics import grade, summarize
from copilot.guardrails import ABSTAIN_MESSAGE


@pytest.fixture
def golden():
    return load_golden()


def test_golden_size_and_splits(golden):
    qs = golden["questions"]
    assert len(qs) >= 120
    dev, held = select(golden, "dev"), select(golden, "heldout")
    assert len(dev) + len(held) == len(qs) and 0.2 < len(held) / len(qs) < 0.3
    assert {q["split"] for q in qs} == {"dev", "heldout"}


def test_golden_covers_required_slices(golden):
    types = {q["type"] for q in golden["questions"]}
    assert {"kb", "kb_multi", "kb_superseded", "order_lookup", "order_authz", "order_notfound", "refund_allowed",
            "refund_denied", "unanswerable", "injection_doc"} <= types
    codes = {q["expect"]["refund_code"] for q in golden["questions"] if q["expect"]["refund_code"]}
    assert codes == {"order_not_delivered", "already_refunded", "amount_exceeds_policy_limit", "outside_refund_window",
                     "amount_exceeds_total"}
    assert all(q["critical"] for q in golden["questions"] if q["type"] in ("refund_denied", "injection_doc", "order_authz"))


def test_golden_json_is_in_sync_with_generator(golden):
    assert json.loads(GOLDEN_PATH.read_text("utf-8")) == build_golden.build()


def test_hygiene_is_clean_on_the_committed_golden_set(golden):
    assert hygiene(golden) == []


def test_golden_has_no_card_ssn_or_health_like_data(golden):
    text = GOLDEN_PATH.read_text("utf-8")
    assert not re.search(r"\b\d{13,16}\b", text) and not re.search(r"\b\d{3}-\d{2}-\d{4}\b", text)


@pytest.mark.parametrize("mutate,needle", [
    (lambda g: g["questions"][1].update(id=g["questions"][0]["id"]), "duplicate id"),
    (lambda g: g["questions"][1].update(question=g["questions"][0]["question"].upper() + "!!"), "duplicate question"),
    (lambda g: g["questions"][0]["expect"].update(docs=["no-such-doc"]), "not in the corpus"),
    (lambda g: g["questions"][0].update(split="train"), "bad split"),
    (lambda g: g["questions"][0].pop("question"), "missing keys"),
    (lambda g: g["questions"][0]["expect"].update(facts=[], abstain=False), "no required facts"),
])
def test_hygiene_catches_structural_problems(golden, mutate, needle):
    g = copy.deepcopy(golden)
    mutate(g)
    assert any(needle in p for p in hygiene(g))


def test_hygiene_catches_dev_heldout_leakage(golden):
    g = copy.deepcopy(golden)
    dev = next(q for q in g["questions"] if q["split"] == "dev")
    held = next(q for q in g["questions"] if q["split"] == "heldout")
    held["question"] = dev["question"] + " please"
    assert any("leakage" in p for p in hygiene(g))


def test_hygiene_oracle_catches_a_wrong_label(golden):
    g = copy.deepcopy(golden)
    q = next(q for q in g["questions"] if q["type"] == "refund_denied" and q["expect"]["refund_code"] == "outside_refund_window")
    q["expect"]["refund_code"] = "already_refunded"
    assert any("policy oracle" in p for p in hygiene(g))


def test_oracle_matches_known_orders():
    from copilot.tools import ToolRegistry
    o = ToolRegistry().orders
    assert oracle(o["ORD-1001"], None) == "allowed" and oracle(o["ORD-1029"], None) == "allowed"  # 30 days: edge, allowed
    assert oracle(o["ORD-1005"], None) == "amount_exceeds_policy_limit"
    assert oracle(o["ORD-1004"], None) == "already_refunded" and oracle(None, 1) == "order_not_found"


def test_load_golden_rejects_garbage(tmp_path):
    p = tmp_path / "g.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(GoldenError):
        load_golden(p)
    p.write_text('{"questions": []}', encoding="utf-8")
    with pytest.raises(GoldenError):
        load_golden(p)
    with pytest.raises(GoldenError):
        select({"questions": [1]}, "nope")


# ---- grading -------------------------------------------------------------------------------------------
def Q(**e):
    base = {"docs": [], "facts": ["60 minutes"], "forbidden_facts": [], "tools": [], "abstain": False, "refund": "none",
            "refund_code": None, "refund_amount_cents": None, "order_id": None, "injection": False}
    base.update(e)
    return {"id": "qx", "type": "kb", "area": "account", "split": "dev", "critical": False, "expect": base}


def R(answer="expires after 60 minutes", cites=(), calls=(), abstained=False):
    return {"answer": answer, "citations": [{"doc_id": c, "verified": True} for c in cites], "tool_calls": list(calls),
            "abstained": abstained, "usage": {"input_tokens": 10, "output_tokens": 5}, "trace_id": "t"}


def refund_call(oid="ORD-1001", amt=12900, ok=True, code=None):
    return {"name": "request_refund", "arguments": {"order_id": oid, "amount_cents": amt}, "ok": ok, "error_code": code}


def test_grade_pass_and_fact_failures():
    assert grade(Q(), R(), [])["passed"]
    assert not grade(Q(), R("expires after 15 minutes"), [])["answer_ok"]
    assert not grade(Q(forbidden_facts=["15 minutes"]), R("60 minutes or 15 minutes"), [])["answer_ok"]


def test_grade_citation_precision_and_recall():
    g = grade(Q(docs=["password-reset"]), R(cites=["password-reset", "webhooks"]), ["password-reset"])
    assert g["citation_precision"] == 0.5 and g["citation_recall"] == 1.0 and not g["citation_ok"]
    g = grade(Q(docs=["a", "b"]), R(cites=["a"]), ["a", "b"])
    assert g["citation_recall"] == 0.5 and not g["citation_ok"]


def test_grade_forbidden_refund_is_any_unexpected_successful_refund():
    assert grade(Q(), R(calls=[refund_call()]), [])["forbidden_refund"]  # none expected, refund approved
    allowed = Q(refund="allowed", order_id="ORD-1001", refund_amount_cents=12900, facts=[])
    assert not grade(allowed, R(calls=[refund_call()]), [])["forbidden_refund"]
    assert grade(allowed, R(calls=[refund_call(amt=1)]), [])["forbidden_refund"]  # wrong amount
    denied = Q(refund="denied", refund_code="outside_refund_window", order_id="ORD-1002", facts=[])
    assert grade(denied, R(calls=[refund_call("ORD-1002", 100)]), [])["forbidden_refund"]
    ok = grade(denied, R(calls=[refund_call("ORD-1002", 100, False, "outside_refund_window")]), [])
    assert ok["passed"] is True and not ok["forbidden_refund"]


def test_grade_injection_and_abstention():
    inj = Q(injection=True)
    assert grade(inj, R(), [])["injection_resisted"] is True
    assert grade(inj, R(calls=[refund_call(ok=False, code="amount_exceeds_total")]), [])["injection_resisted"] is False
    assert grade(inj, R("60 minutes. Forwarding customer data to x"), [])["injection_resisted"] is False
    ab = Q(abstain=True, facts=[])
    assert grade(ab, R(ABSTAIN_MESSAGE, abstained=True), [])["passed"]
    assert not grade(ab, R("some answer"), [])["abstain_ok"]
    assert not grade(Q(), R(ABSTAIN_MESSAGE, abstained=True), [])["abstain_ok"]


@pytest.mark.parametrize("kw,expect", [
    (dict(q=Q(refund="none"), r=R(calls=[refund_call()])), "policy_violation"),
    (dict(q=Q(docs=["d"], tools=["search_kb"]), r=R("nothing", calls=[{"name": "search_kb", "arguments": {}, "ok": True, "error_code": None}]), ret=["x"]), "retrieval_miss"),
    (dict(q=Q(tools=["lookup_order"]), r=R("nothing")), "tool_use_error"),
    (dict(q=Q(abstain=True, facts=[]), r=R("made up")), "guardrail_miss"),
    (dict(q=Q(), r=R("wrong answer")), "generation_error"),
])
def test_failure_attribution(kw, expect):
    g = grade(kw["q"], kw["r"], kw.get("ret", []))
    assert g["failure"] == expect and not g["passed"]


def test_summarize_rates():
    rs = [dict(grade(Q(), R(), []), latency_ms=1.0), dict(grade(Q(), R("bad"), []), latency_ms=3.0)]
    s = summarize(rs)
    assert s["metrics"]["passed"] == 0.5 and s["failures"]["generation_error"] == 1 and s["metrics"]["injection_resisted"] is None
