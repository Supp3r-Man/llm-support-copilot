"""Runs a named variant over golden questions: one fresh agent (fresh refund state) per question."""
from __future__ import annotations

import time

from copilot import build_agent
from copilot.retrieval import Retriever
from copilot.tools import ToolRegistry

from .golden import load_golden, select
from .metrics import grade, summarize
from .variants import get_variant, variant_config


class EvalCrash(Exception):
    """The system under test (or the harness) crashed: the gate cannot judge."""


def run_question(variant: str, q: dict, trace_dir=None, model=None) -> dict:
    v, cfg = get_variant(variant), variant_config(variant)
    retr = Retriever.from_config(cfg)
    reg = ToolRegistry(cfg, retr)
    seen: list[dict] = []
    tool = reg.tools["search_kb"]
    orig = tool.fn

    def capture(a, ctx):
        res = orig(a, ctx)
        if res["ok"]:
            seen.extend(res["result"]["passages"])
        return res

    tool.fn = capture
    agent = build_agent(cfg, model=model, faults=v["faults"], trace_dir=trace_dir, guardrails=v["guardrails"], registry=reg)
    t0 = time.perf_counter()
    try:
        res = agent.run(q["question"], q.get("customer_id"))
    except Exception as e:  # a crash is not a quality score
        raise EvalCrash(f"variant {variant!r} crashed on {q['id']}: {type(e).__name__}: {e}") from e
    ms = (time.perf_counter() - t0) * 1000
    stale = any(p["version"] != retr.latest.get(p["doc_id"]) for p in seen)
    g = grade(q, res.to_dict(), [p["doc_id"] for p in seen], stale)
    g["latency_ms"] = round(ms, 3)
    return g


def run_variant(variant: str, questions: list[dict], trace_dir=None, model=None) -> list[dict]:
    get_variant(variant)  # fail fast on unknown names
    return [run_question(variant, q, trace_dir, model) for q in questions]


def run_named(variant: str, split: str = "all", golden_path=None, trace_dir=None, model=None) -> dict:
    qs = select(load_golden(golden_path), split)
    results = run_variant(variant, qs, trace_dir, model)
    return {"variant": variant, "split": split, "description": get_variant(variant)["desc"],
            "summary": summarize(results), "results": results}
