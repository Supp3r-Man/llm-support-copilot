"""The agent loop: step limit, per-tool timeout, duplicate-call detection, token budget,
schema-error feedback to the model, guardrailed final answer with citations."""
from __future__ import annotations

import contextvars
import json
import threading
from dataclasses import dataclass, field

from .config import Config
from .guardrails import (ABSTAIN_MESSAGE, CITE_RE, FAILSAFE_MESSAGE, check_output, scan_tool_result, verify_answer)
from .models import Model, RuleModel
from .retrieval import Retriever
from .tools import ToolRegistry, err
from .tracing import Tracer, span


@dataclass
class AgentResult:
    answer: str
    citations: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    trace_id: str | None = None
    steps: int = 0
    stop_reason: str = "final"  # final | max_steps | token_budget
    abstained: bool = False
    flags: list = field(default_factory=list)  # injection flags raised on tool/retrieved text
    violations: list = field(default_factory=list)  # dropped sentences, leaks, redactions
    usage: dict = field(default_factory=dict)
    config_hash: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Agent:
    def __init__(self, config: Config, model: Model, registry: ToolRegistry, retriever: Retriever,
                 tracer: Tracer | None = None, guardrails: bool = True):
        self.config, self.model, self.registry, self.retriever = config, model, registry, retriever
        self.tracer = tracer or Tracer(None)
        self.guardrails = guardrails

    def run(self, question: str, customer_id: str | None = None) -> AgentResult:
        cfg = self.config
        with self.tracer.trace("ask", prompt_version=cfg.prompt_version, config_hash=cfg.config_hash(),
                               config_version=cfg.version, model=self.model.name,
                               faults=sorted(getattr(self.model, "faults", [])), guardrails=self.guardrails,
                               retriever=cfg.retriever_settings(), customer_id=customer_id,
                               question=question) as (tr, root):
            res = self._loop(question, customer_id)
            root.set(stop_reason=res.stop_reason, steps=res.steps, abstained=res.abstained,
                     tool_calls=len(res.tool_calls), flags=res.flags)
        res.trace_id = tr.trace_id
        res.config_hash = cfg.config_hash()
        return res

    def _loop(self, question: str, customer_id) -> AgentResult:
        cfg = self.config
        messages = [{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": question}]
        specs, ctx = self.registry.specs(), {"customer_id": customer_id}
        seen, log, flags = set(), [], []
        tokens_in = tokens_out = 0
        final, stop, steps = None, "max_steps", 0
        for step in range(1, cfg.max_steps + 1):
            if tokens_in + tokens_out >= cfg.token_budget:
                stop = "token_budget"
                break
            steps = step
            with span("model_turn", step=step, model=self.model.name) as sp:
                turn = self.model.generate(messages, specs)
                sp.set(input_tokens=turn.usage.input_tokens, output_tokens=turn.usage.output_tokens,
                       tool_calls=len(turn.tool_calls))
            tokens_in += turn.usage.input_tokens
            tokens_out += turn.usage.output_tokens
            if not turn.tool_calls:
                final, stop = turn.text or "", "final"
                break
            messages.append({"role": "assistant", "content": turn.text, "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls]})
            for tc in turn.tool_calls:
                key = (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))
                if key in seen:
                    with span(f"tool:{tc.name}", duplicate=True):
                        result = err("duplicate_tool_call", "identical call already made; use the earlier result")
                else:
                    seen.add(key)
                    result = self._call_tool(tc, ctx)
                shown = result
                if self.guardrails:
                    with span("guardrail:injection") as sp:
                        shown, f = scan_tool_result(result)
                        sp.set(flags=f)
                    flags.extend(x for x in f if x not in flags)
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(shown)})
                log.append({"name": tc.name, "arguments": tc.arguments, "ok": result["ok"],
                            "error_code": None if result["ok"] else result["error"]["code"]})
        res = AgentResult("", tool_calls=log, steps=steps, stop_reason=stop, flags=flags,
                          usage={"input_tokens": tokens_in, "output_tokens": tokens_out})
        if final is None:
            res.answer = FAILSAFE_MESSAGE
            return res
        if not self.guardrails:
            res.answer = final
            res.citations = [{"doc_id": d, "verified": False} for d in dict.fromkeys(CITE_RE.findall(final))]
            res.abstained = final == ABSTAIN_MESSAGE
            return res
        with span("guardrail:finalize") as sp:
            v = verify_answer(final, self.retriever)
            answer = v.text
            if not answer.strip() or (not log and not v.citations):
                answer = ABSTAIN_MESSAGE
            answer, out_issues = check_output(answer)
            res.violations = [f"{i['reason']}:{i['doc_id']}" for i in v.issues] + out_issues
            sp.set(violations=res.violations)
        res.answer = answer
        res.citations = [{"doc_id": d, "verified": True} for d in v.citations] if answer != ABSTAIN_MESSAGE else []
        res.abstained = answer == ABSTAIN_MESSAGE
        return res

    def _call_tool(self, tc, ctx) -> dict:
        box: dict = {}
        c = contextvars.copy_context()  # keep span nesting inside the worker thread

        def work():
            box["r"] = c.run(self.registry.call, tc.name, tc.arguments, ctx)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(self.config.tool_timeout_s)
        if t.is_alive():
            with span(f"tool:{tc.name}", timeout=True) as sp:
                sp.status = "error"
            return err("tool_timeout", f"tool '{tc.name}' exceeded {self.config.tool_timeout_s}s")
        return box["r"]


def build_agent(config: Config | None = None, model: Model | None = None, faults=(), trace_dir=None,
                guardrails: bool = True, registry: ToolRegistry | None = None) -> Agent:
    """Wire config -> retriever -> tools -> model -> agent. `faults` only applies to the default RuleModel."""
    cfg = config or Config()
    retriever = Retriever.from_config(cfg)
    registry = registry or ToolRegistry(cfg, retriever)
    model = model or RuleModel(faults, cfg.min_score)
    return Agent(cfg, model, registry, retriever, Tracer(trace_dir), guardrails)
