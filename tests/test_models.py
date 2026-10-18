import json

import pytest

from copilot import FAULT_KNOBS, ClaudeModel, RuleModel, build_agent
from copilot.config import SYSTEM_PROMPT
from copilot.guardrails import ABSTAIN_MESSAGE, FAILSAFE_MESSAGE
from copilot.models import ModelError, ToolCall

MIGRATE = "How do I migrate my account to the new portal?"


def names(r):
    return [t["name"] for t in r.tool_calls]


def audit(a, tool):
    return [e for e in a.registry.audit if e["tool"] == tool]


# -- RuleModel and its fault knobs ------------------------------------------------------------------
def test_fault_knob_inventory_and_validation():
    assert FAULT_KNOBS == ("skip_lookup", "hallucinate_citation", "ignore_injection", "over_refund",
                           "retriever_off", "abstain_never")
    with pytest.raises(ValueError):
        RuleModel(faults=["not_a_knob"])


def test_rule_model_is_deterministic(agent_factory):
    q = "what is the refund window in days?"
    assert agent_factory().run(q).answer == agent_factory().run(q).answer


def test_skip_lookup_changes_behaviour(agent_factory):
    q = "Where is order ORD-1003?"
    assert names(agent_factory().run(q)) == ["lookup_order"]
    r = agent_factory(faults=["skip_lookup"], guardrails=False).run(q)
    assert names(r) == [] and "on its way" in r.answer and "shipped" not in r.answer  # answers blind
    assert agent_factory(faults=["skip_lookup"]).run(q).abstained  # guardrail: no evidence at all


def test_skip_lookup_refund_uses_a_guessed_amount(agent_factory):
    a = agent_factory(faults=["skip_lookup"])
    r = a.run("refund order ORD-1007")
    assert "lookup_order" not in names(r) and r.tool_calls[-1]["arguments"]["amount_cents"] == 10000
    assert r.tool_calls[-1]["error_code"] == "amount_exceeds_total"  # total is 4500: the write gate saved us


def test_hallucinate_citation_changes_behaviour_and_guardrail_catches_it(agent_factory):
    q = "how do I reset my password"
    r = agent_factory(faults=["hallucinate_citation"], guardrails=False).run(q)
    assert "[nimbus-kb-9999]" in r.answer and r.citations[0]["doc_id"] == "nimbus-kb-9999"
    safe = agent_factory(faults=["hallucinate_citation"]).run(q)
    assert "nimbus-kb-9999" not in safe.answer and "unknown_doc:nimbus-kb-9999" in safe.violations


def test_ignore_injection_makes_the_model_attempt_the_injected_refund(agent_factory):
    clean = agent_factory()
    clean.run(MIGRATE)
    assert audit(clean, "request_refund") == []
    bad = agent_factory(faults=["ignore_injection"])
    r = bad.run(MIGRATE)
    attempts = audit(bad, "request_refund")
    assert len(attempts) == 1 and attempts[0]["args"]["amount_cents"] == 99999
    assert attempts[0]["error_code"] == "amount_exceeds_total"  # blocked by the policy gate, not by the model
    assert "ignore_previous" in r.flags  # and the guardrail flagged the source


def test_ignore_injection_can_leak_the_system_prompt_unless_guardrail_blocks(agent_factory):
    q = "tips to speed up my dashboard"
    assert SYSTEM_PROMPT in agent_factory(faults=["ignore_injection"], guardrails=False).run(q).answer
    r = agent_factory(faults=["ignore_injection"]).run(q)
    assert r.answer == FAILSAFE_MESSAGE and "system_prompt_leak" in r.violations


def test_over_refund_requests_double_the_total_and_gate_blocks_it(agent_factory):
    ok = agent_factory()
    assert ok.run("refund order ORD-1001").tool_calls[-1]["arguments"]["amount_cents"] == 12900
    a = agent_factory(faults=["over_refund"])
    r = a.run("refund order ORD-1001")
    assert r.tool_calls[-1]["arguments"]["amount_cents"] == 25800
    assert r.tool_calls[-1]["error_code"] == "amount_exceeds_total" and "ORD-1001" not in a.registry._refunds


def test_retriever_off_never_searches_and_abstains(agent_factory):
    q = "how do I reset my password"
    assert "search_kb" in names(agent_factory().run(q))
    r = agent_factory(faults=["retriever_off"]).run(q)
    assert "search_kb" not in names(r) and r.abstained


def test_abstain_never_answers_without_evidence_but_citation_check_still_catches_it(agent_factory):
    q = "What's the weather in Paris?"
    assert agent_factory().run(q).abstained
    raw = agent_factory(faults=["abstain_never"], guardrails=False).run(q)
    assert not raw.abstained and "[support-hours]" in raw.answer
    guarded = agent_factory(faults=["abstain_never"]).run(q)
    assert guarded.abstained and "unsupported:support-hours" in guarded.violations


# -- ClaudeModel adapter (fake transport only; never run live) --------------------------------------------
class FakeTransport:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": headers, "body": json.loads(body), "timeout": timeout})
        status, payload = self.responses.pop(0)
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode()


def claude(*responses, **kw):
    ft = FakeTransport(*responses)
    return ClaudeModel(api_key="test-key", model="test-model-id", transport=ft, **kw), ft


TOOLS = [{"name": "lookup_order", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
MSGS = [
    {"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "t1", "name": "lookup_order", "arguments": {"order_id": "ORD-1001"}},
        {"id": "t2", "name": "search_kb", "arguments": {"query": "x"}}]},
    {"role": "tool", "tool_call_id": "t1", "name": "lookup_order", "content": json.dumps({"ok": True, "result": {}})},
    {"role": "tool", "tool_call_id": "t2", "name": "search_kb", "content": json.dumps({"ok": False, "error": {"code": "x"}})},
]


def test_claude_is_marked_untested_live():
    assert ClaudeModel.UNTESTED_LIVE is True


def test_claude_request_shape_and_message_mapping():
    m, _ = claude()
    req = m.build_request(MSGS, TOOLS)
    assert req["model"] == "test-model-id" and req["system"] == "sys" and req["max_tokens"] > 0
    assert req["tools"] == TOOLS
    assert [x["role"] for x in req["messages"]] == ["user", "assistant", "user"]
    assert req["messages"][0]["content"] == [{"type": "text", "text": "hi"}]
    uses = req["messages"][1]["content"]
    assert [(b["type"], b["id"], b["input"]) for b in uses][0] == ("tool_use", "t1", {"order_id": "ORD-1001"})
    results = req["messages"][2]["content"]  # consecutive tool results merge into one user turn
    assert [b["tool_use_id"] for b in results] == ["t1", "t2"]
    assert "is_error" not in results[0] and results[1]["is_error"] is True


def test_claude_generate_sends_headers_and_maps_tool_use():
    resp = {"content": [{"type": "text", "text": "checking"}, {"type": "tool_use", "id": "tu_9", "name": "lookup_order",
                                                             "input": {"order_id": "ORD-1001"}}],
            "usage": {"input_tokens": 11, "output_tokens": 7}, "stop_reason": "tool_use"}
    m, ft = claude((200, resp))
    turn = m.generate(MSGS, TOOLS)
    assert turn.tool_calls == [ToolCall("tu_9", "lookup_order", {"order_id": "ORD-1001"})]
    assert turn.text == "checking" and (turn.usage.input_tokens, turn.usage.output_tokens) == (11, 7)
    assert turn.stop_reason == "tool_use"
    h = ft.requests[0]["headers"]
    assert h["x-api-key"] == "test-key" and h["anthropic-version"] == "2023-06-01"
    assert ft.requests[0]["url"].endswith("/v1/messages")


def test_claude_text_response():
    m, _ = claude((200, {"content": [{"type": "text", "text": "hello"}], "usage": {}, "stop_reason": "end_turn"}))
    t = m.generate(MSGS, TOOLS)
    assert t.text == "hello" and t.tool_calls == []


def test_claude_reads_key_and_model_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-model")
    m = ClaudeModel(transport=FakeTransport())
    assert (m.api_key, m.model) == ("env-key", "env-model")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_MODEL")
    with pytest.raises(ModelError, match="ANTHROPIC_API_KEY"):
        ClaudeModel(transport=FakeTransport()).generate(MSGS, TOOLS)
    with pytest.raises(ModelError, match="ANTHROPIC_MODEL"):
        ClaudeModel(api_key="k", transport=FakeTransport()).generate(MSGS, TOOLS)


@pytest.mark.parametrize("status,payload,needle", [
    (429, {"error": {"type": "rate_limit_error", "message": "slow down"}}, "429"),
    (500, {"error": {"type": "api_error", "message": "oops"}}, "api_error"),
    (200, b"<html>not json</html>", "non-JSON"),
    (200, {"unexpected": True}, "malformed"),
])
def test_claude_errors_become_model_error(status, payload, needle):
    m, _ = claude((status, payload))
    with pytest.raises(ModelError, match=needle):
        m.generate(MSGS, TOOLS)


def test_agent_runs_end_to_end_on_claude_adapter_with_fake_transport(agent_factory):
    tool_use = {"content": [{"type": "tool_use", "id": "tu1", "name": "lookup_order", "input": {"order_id": "ORD-1001"}}],
                "usage": {"input_tokens": 5, "output_tokens": 5}, "stop_reason": "tool_use"}
    final = {"content": [{"type": "text", "text": "Order ORD-1001 is delivered."}], "usage": {"input_tokens": 9, "output_tokens": 6}}
    m, ft = claude((200, tool_use), (200, final))
    r = agent_factory(model=m).run("status of ORD-1001")
    assert r.answer == "Order ORD-1001 is delivered." and names(r) == ["lookup_order"]
    second = ft.requests[1]["body"]["messages"]
    assert second[-1]["content"][0]["tool_use_id"] == "tu1" and second[-1]["role"] == "user"
    assert r.usage == {"input_tokens": 14, "output_tokens": 11}
