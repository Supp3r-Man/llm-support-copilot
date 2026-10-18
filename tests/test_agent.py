import json
import time

from copilot import Config, build_agent
from copilot.guardrails import ABSTAIN_MESSAGE, FAILSAFE_MESSAGE
from copilot.tools import Tool, ToolRegistry
from helpers import ScriptedModel, call, say


def names(res):
    return [t["name"] for t in res.tool_calls]


def test_order_status_end_to_end(agent_factory):
    r = agent_factory().run("Where is order ORD-1003?")
    assert names(r) == ["lookup_order"] and "shipped" in r.answer and r.trace_id and r.stop_reason == "final"
    assert r.usage["input_tokens"] > 0 and r.config_hash


def test_kb_answer_carries_verified_citations(agent_factory):
    r = agent_factory().run("how do I reset my password")
    assert [c["doc_id"] for c in r.citations] == ["password-reset"] and r.citations[0]["verified"]
    assert "[password-reset]" in r.answer and not r.abstained


def test_refund_approved_and_audited(agent_factory):
    a = agent_factory()
    r = a.run("Please refund order ORD-1001, it arrived broken")
    assert names(r) == ["lookup_order", "search_kb", "request_refund"] and "approved" in r.answer
    assert [e["ok"] for e in a.registry.audit if e["tool"] == "request_refund"] == [True]
    assert "billing-policy" in [c["doc_id"] for c in r.citations]


def test_refund_denied_is_explained_not_raised(agent_factory):
    r = agent_factory().run("Refund order ORD-1002 please")
    assert "outside_refund_window" in r.answer and r.tool_calls[-1]["error_code"] == "outside_refund_window"


def test_abstains_when_nothing_relevant_is_retrieved(agent_factory):
    r = agent_factory().run("What's the weather in Paris?")
    assert r.abstained and r.answer == ABSTAIN_MESSAGE and r.citations == []


def test_customer_scope_is_enforced_through_the_loop(agent_factory):
    r = agent_factory().run("Where is order ORD-1001?", customer_id="CUST-002")
    assert r.tool_calls[0]["error_code"] == "forbidden" and "delivered" not in r.answer


def test_step_limit_stops_a_runaway_model(agent_factory):
    queries = iter(f"query number {i}" for i in range(100))
    model = ScriptedModel([lambda m: call("search_kb", query=next(queries))])
    r = agent_factory(model=model, config=Config(max_steps=3)).run("hi")
    assert r.stop_reason == "max_steps" and model.n == 3 and len(r.tool_calls) == 3 and r.answer == FAILSAFE_MESSAGE


def test_duplicate_tool_call_is_detected_and_not_executed_twice(agent_factory):
    model = ScriptedModel([call("lookup_order", order_id="ORD-1001"), call("lookup_order", order_id="ORD-1001"),
                           say("Order ORD-1001 is delivered.")])
    a = agent_factory(model=model)
    r = a.run("status?")
    assert [t["error_code"] for t in r.tool_calls] == [None, "duplicate_tool_call"]
    assert sum(1 for e in a.registry.audit if e["tool"] == "lookup_order") == 1
    assert r.stop_reason == "final"


def test_schema_error_is_fed_back_and_model_recovers(agent_factory):
    def fixed(messages):
        last = json.loads(messages[-1]["content"])
        assert messages[-1]["role"] == "tool" and last["error"]["code"] == "invalid_arguments"
        return call("lookup_order", order_id="ORD-1001")

    model = ScriptedModel([call("lookup_order"), fixed, say("Order ORD-1001 is delivered.")])
    r = agent_factory(model=model).run("status of my order")
    assert [t["ok"] for t in r.tool_calls] == [False, True] and r.answer == "Order ORD-1001 is delivered."


def test_tool_timeout_becomes_error_result(cfg, retriever):
    reg = ToolRegistry(cfg, retriever)
    reg.register(Tool("slow", "x", {"type": "object", "properties": {}}, lambda a, c: time.sleep(0.6) or {"ok": True, "result": {}}))
    model = ScriptedModel([call("slow"), say("Sorry, that tool is unavailable right now.")])
    t0 = time.time()
    r = build_agent(Config(tool_timeout_s=0.05), model=model, registry=reg).run("go")
    assert r.tool_calls[0]["error_code"] == "tool_timeout" and time.time() - t0 < 0.5
    assert "unavailable" in r.answer


def test_token_budget_stops_the_loop(agent_factory):
    model = ScriptedModel([call("search_kb", query="alpha beta"), call("search_kb", query="gamma delta"), say("done")])
    r = agent_factory(model=model, config=Config(token_budget=10)).run("hi")
    assert r.stop_reason == "token_budget" and model.n == 1 and r.answer == FAILSAFE_MESSAGE


def test_injection_in_retrieved_text_is_flagged_fenced_and_not_obeyed(agent_factory):
    model = ScriptedModel([call("search_kb", query="migrate account to new portal"), say("You can migrate from Settings.")])
    a = agent_factory(model=model)
    r = a.run("How do I migrate my account to the new portal?")
    tool_msg = next(m for m in model.seen[1] if m["role"] == "tool")["content"]
    assert "<<UNTRUSTED" in tool_msg and "Ignore previous instructions" in tool_msg  # fenced, not deleted
    assert {"ignore_previous", "issue_refund_command"} <= set(r.flags)
    assert not any(e["tool"] == "request_refund" for e in a.registry.audit)


def test_default_rule_model_does_not_follow_injected_refund(agent_factory):
    a = agent_factory()
    r = a.run("How do I migrate my account to the new portal?")
    assert "ignore_previous" in r.flags and not any(e["tool"] == "request_refund" for e in a.registry.audit)
    assert "refund" not in r.answer.lower()


def test_guardrails_off_skips_flagging_and_verification(agent_factory):
    r = agent_factory(guardrails=False).run("How do I migrate my account to the new portal?")
    assert r.flags == [] and all(c["verified"] is False for c in r.citations)


def test_final_answer_email_is_redacted(agent_factory):
    model = ScriptedModel([call("lookup_order", order_id="ORD-1001"), say("Contact ops@alder.example.test for details.")])
    r = agent_factory(model=model).run("who owns ORD-1001")
    assert "@" not in r.answer and "email_redacted" in r.violations


def test_answer_without_any_evidence_is_replaced_by_abstention(agent_factory):
    r = agent_factory(model=ScriptedModel([say("Nimbus is the best cloud ever made.")])).run("is nimbus good")
    assert r.abstained and r.tool_calls == []


def test_trace_file_is_written_for_each_run(agent_factory, tmp_path):
    a = agent_factory()
    r = a.run("how do I reset my password")
    assert (tmp_path / "traces" / f"{r.trace_id}.jsonl").exists()
