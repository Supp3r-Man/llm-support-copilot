import pytest
from helpers import ScriptedModel, say

from copilot.models import FAULT_KNOBS
from copilot_eval.gate import compare
from copilot_eval.judge import LLMJudge, RuleJudge, calibrate, load_labels
from copilot_eval.load import run_load
from copilot_eval.runner import EvalCrash, run_question, run_variant
from copilot_eval.golden import load_golden, select
from copilot_eval.variants import VARIANTS, UnknownVariant, get_variant


def verdict(eval_runs, name):
    return compare(eval_runs["baseline"], eval_runs[name], n_boot=500)


def test_baseline_is_decent_but_not_perfect(eval_runs):
    from copilot_eval.metrics import summarize
    s = summarize(eval_runs["baseline"])
    assert 0.8 < s["metrics"]["passed"] < 0.95 and s["forbidden_refunds"] == 0 and s["metrics"]["injection_resisted"] == 1.0
    assert s["failures"]["policy_violation"] == 0 and s["failures"]["guardrail_miss"] >= 1  # known abstention gap


@pytest.mark.parametrize("fault", FAULT_KNOBS)
def test_planted_fault_without_guardrails_is_detected(eval_runs, fault):
    r = verdict(eval_runs, f"raw_{fault}")
    assert r["verdict"] == "FAIL" and r["exit_code"] == 1 and r["newly_failed_total"] >= 10


def test_guarded_matrix_every_fault_has_an_explained_verdict(eval_runs):
    rows = {f: verdict(eval_runs, f"fault_{f}") for f in FAULT_KNOBS}
    missed = [f for f, r in rows.items() if r["verdict"] == "PASS"]
    assert missed == []  # measured: on this golden set the gate sees all six even with guardrails on
    # ... but detection paths differ: the refund gate blocks over_refund / ignore_injection harm, the gate still sees the attempt
    assert rows["ignore_injection"]["critical"]["injection_acted"]
    assert rows["over_refund"]["newly_failed_total"] >= 10 and not rows["over_refund"]["critical"]["forbidden_refunds"]


def test_guardrails_alone_are_invisible_without_a_misbehaving_model(eval_runs):
    r = verdict(eval_runs, "guardrails_off")
    assert r["verdict"] == "PASS" and r["newly_failed_total"] == 0  # RuleModel never misbehaves without a fault knob


def test_guardrails_matter_for_leaks_but_not_for_injected_tool_calls(eval_runs):
    raw = {x["id"]: x for x in eval_runs["raw_ignore_injection"]}
    on = {x["id"]: x for x in eval_runs["fault_ignore_injection"]}
    leaks_raw = [i for i, x in raw.items() if x["injection_resisted"] is False]
    leaks_on = [i for i, x in on.items() if x["injection_resisted"] is False]
    assert len(leaks_raw) > len(leaks_on) > 0  # output guard blocks prompt/email leaks, injected refund attempts remain


def test_bad_refund_policy_is_caught_only_by_the_critical_rule(eval_runs):
    r = verdict(eval_runs, "bad_refund_limit")
    assert r["verdict"] == "FAIL" and not r["regressions"] and r["critical"]["forbidden_refunds"]


def test_bad_retriever_top_k_is_caught_by_a_slice_not_the_overall_rate(eval_runs):
    r = verdict(eval_runs, "bad_top_k1")
    flagged = set(r["regressions"])
    assert "type=kb_multi" in flagged and "overall:passed" not in flagged


def test_stale_policy_retrieval_is_caught_via_critical_superseded_questions(eval_runs):
    r = verdict(eval_runs, "bad_latest_only_off")
    assert r["verdict"] == "FAIL" and r["critical"]["newly_failed"]


def test_subtle_regression_below_power_is_missed(eval_runs):
    """Honesty test: flip 2 non-critical passing questions; at n~140 the statistics cannot see it."""
    c = [dict(x) for x in eval_runs["baseline"]]
    flipped = 0
    for x in c:
        if x["passed"] and not x["critical"] and flipped < 2:
            x.update(passed=False, answer_ok=False, failure="generation_error")
            flipped += 1
    assert compare(eval_runs["baseline"], c, n_boot=500)["verdict"] == "PASS"


def test_runs_are_deterministic(eval_runs):
    qs = select(load_golden(), "dev")[:30]
    strip = lambda rs: [{k: v for k, v in r.items() if k not in ("latency_ms", "trace_id")} for r in rs]
    assert strip(run_variant("baseline", qs)) == strip([r for r in eval_runs["baseline"] if r["id"] in {q["id"] for q in qs}])


def test_variant_registry_and_crash_handling():
    assert {"baseline", "guardrails_off", "bad_top_k1", "bad_latest_only_off"} <= set(VARIANTS)
    assert all(f"fault_{f}" in VARIANTS and f"raw_{f}" in VARIANTS for f in FAULT_KNOBS)
    with pytest.raises(UnknownVariant):
        get_variant("nope")

    class Boom:
        name = "boom"

        def generate(self, messages, tools):
            raise RuntimeError("model fell over")

    with pytest.raises(EvalCrash):
        run_question("baseline", select(load_golden(), "dev")[0], model=Boom())


def test_every_result_has_trace_id_and_usage(eval_runs):
    assert all(r["trace_id"] and r["input_tokens"] > 0 and r["latency_ms"] >= 0 for r in eval_runs["baseline"])


# ---- judge calibration -----------------------------------------------------------------------------------
def test_labels_file_is_balanced():
    items = load_labels()
    assert len(items) >= 30 and {i["label"] for i in items} == {"pass", "fail"}
    assert abs(sum(i["label"] == "pass" for i in items) - len(items) / 2) <= 2


def test_rule_judge_is_measured_as_not_usable():
    rep = calibrate(RuleJudge(), load_labels())
    assert rep["kappa"] < 0.6 and not rep["usable"]  # misses paraphrases, accepts contradicting extras


def test_perfect_and_inverted_judges():
    items = load_labels()
    truth = {i["answer"]: i["label"] for i in items}
    perfect = lambda q, a, r: truth[a]
    inverse = lambda q, a, r: "fail" if truth[a] == "pass" else "pass"
    assert calibrate(perfect, items)["kappa"] == 1.0 and calibrate(perfect, items)["usable"]
    assert calibrate(inverse, items)["kappa"] < 0


def test_llm_judge_parses_verdicts_with_a_fake_model():
    assert LLMJudge(ScriptedModel([say("VERDICT: PASS")]))("q", "a", "r") == "pass"
    assert LLMJudge(ScriptedModel([say("thinking...\nverdict: fail")]))("q", "a", "r") == "fail"
    j = LLMJudge(ScriptedModel([say("I cannot decide")]))
    assert j("q", "a", "r") == "unparseable"
    rep = calibrate(j, load_labels())
    assert not rep["usable"] and rep["confusion_human_vs_judge"]["pass"]["unparseable"] > 0
    assert LLMJudge.UNTESTED_LIVE is True


# ---- load test -------------------------------------------------------------------------------------------
def test_load_small_run_has_no_errors():
    r = run_load(clients=3, requests=30)
    assert r["completed"] == 30 and r["ok_rate"] == 1.0 and r["error_rate"] == 0 and r["rate_limited_429"] == 0
    lat = r["latency_ms"]
    assert 0 < lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"] and r["throughput_rps"] > 0


def test_load_with_tight_rate_limit_reports_429_not_errors():
    r = run_load(clients=2, requests=30, rate_capacity=3, rate_refill_per_s=0.001)
    assert r["rate_limited_429"] > 0.7 and r["error_rate"] == 0 and r["status_counts"].get("200", 0) >= 3
