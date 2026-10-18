def rec(i, **kw):
    """A synthetic per-question result record shaped like copilot_eval.metrics.grade output."""
    d = {"id": f"q{i:03d}", "type": "kb", "area": "general", "split": "dev", "critical": False, "passed": True,
         "answer_ok": True, "citation_ok": True, "tool_ok": True, "abstain_ok": True, "injection_resisted": None,
         "forbidden_refund": False, "failure": None, "answer": "x", "input_tokens": 10, "output_tokens": 5,
         "latency_ms": 1.0}
    d.update(kw)
    return d


def failed(i, failure="generation_error", **kw):
    return rec(i, passed=False, answer_ok=False, failure=failure, **kw)
