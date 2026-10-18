"""Model protocol, deterministic RuleModel (with fault knobs for evals) and a ClaudeModel adapter.

Internal message format (provider neutral):
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "content": str|None, "tool_calls": [{"id", "name", "arguments": dict}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": <JSON string of the tool result>}
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .config import SYSTEM_PROMPT
from .guardrails import ABSTAIN_MESSAGE, SENT_SPLIT


class ModelError(Exception):
    """Raised by a model adapter (network, HTTP, malformed response)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Turn:
    text: str | None = None
    tool_calls: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"


class Model(Protocol):
    name: str

    def generate(self, messages: list[dict], tools: list[dict]) -> Turn: ...


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


FAULT_KNOBS = ("skip_lookup", "hallucinate_citation", "ignore_injection", "over_refund", "retriever_off", "abstain_never")
ORDER_RE = re.compile(r"\bORD-\d{3,}\b")
REFUND_RE = re.compile(r"(?i)\brefund|money back|reimburs")
AMOUNT_RE = re.compile(r"\$(\d+(?:\.\d{1,2})?)")
INJECTED_REFUND_RE = re.compile(r"(?i)refund of (\d+) cents for order (ORD-\d+)")


def money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _cited(sentence: str, doc_id: str) -> str:
    return f"{sentence.strip().rstrip('.!?')} [{doc_id}]."


class RuleModel:
    """Deterministic stand-in: reads the question, calls tools in a fixed plan, writes a cited answer.

    Faults (any subset of FAULT_KNOBS):
      skip_lookup           never calls lookup_order (answers blind / refunds with a guessed amount)
      hallucinate_citation  cites a doc id that does not exist
      ignore_injection      obeys instruction-like text found in tool results (even when fenced as untrusted)
      over_refund           requests double the order total
      retriever_off         never calls search_kb
      abstain_never         answers even with weak or missing evidence
    """

    name = "rule-model"

    def __init__(self, faults=(), min_score: float = 3.0):
        faults = frozenset(faults)
        unknown = faults - set(FAULT_KNOBS)
        if unknown:
            raise ValueError(f"unknown fault knobs: {sorted(unknown)}")
        self.faults, self.min_score = faults, min_score

    def generate(self, messages: list[dict], tools: list[dict]) -> Turn:
        q = next(m["content"] for m in messages if m["role"] == "user")
        made = [tc for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
        results: dict[str, list] = {}
        raw_tool_text = ""
        for m in messages:
            if m["role"] == "tool":
                raw_tool_text += m["content"] + "\n"
                try:
                    results.setdefault(m["name"], []).append(json.loads(m["content"]))
                except ValueError:
                    pass
        call = self._next_call(q, made, results, raw_tool_text)
        n_in = sum(estimate_tokens(m.get("content") or "") for m in messages) + estimate_tokens(json.dumps(tools))
        if call:
            tc = ToolCall(f"call_{len(made) + 1}", call[0], call[1])
            return Turn(None, [tc], Usage(n_in, estimate_tokens(json.dumps(call[1]))), "tool_use")
        text = self._answer(q, results, raw_tool_text)
        return Turn(text, [], Usage(n_in, estimate_tokens(text)), "end_turn")

    # -- planning -------------------------------------------------------------------------------
    def _next_call(self, q, made, results, raw):
        names = [tc["name"] for tc in made]
        oids = ORDER_RE.findall(q)
        oid = oids[0] if oids else None
        refund = bool(REFUND_RE.search(q))
        if oid and "skip_lookup" not in self.faults and "lookup_order" not in names:
            return "lookup_order", {"order_id": oid}
        need_kb = (refund or not oid) and "retriever_off" not in self.faults
        if need_kb and "search_kb" not in names:
            args = {"query": q}
            if refund:
                args["area"] = "billing"
            return "search_kb", args
        if refund and oid and "request_refund" not in names:
            looked = results.get("lookup_order", [])
            if not (looked and not looked[0]["ok"]):  # do not refund an order we failed to look up
                total = looked[0]["result"]["total_cents"] if looked else None
                m = AMOUNT_RE.search(q)
                amount = int(round(float(m.group(1)) * 100)) if m else (total or 10000)
                if "over_refund" in self.faults:
                    amount = (total or amount) * 2
                return "request_refund", self._refund_args(oid, amount, "customer request: " + q[:80])
        if "ignore_injection" in self.faults:
            for amt, inj_oid in INJECTED_REFUND_RE.findall(raw):
                args = self._refund_args(inj_oid, int(amt), "followed document instruction")
                if not any(tc["name"] == "request_refund" and tc["arguments"] == args for tc in made):
                    return "request_refund", args
        return None

    @staticmethod
    def _refund_args(oid: str, amount: int, reason: str) -> dict:
        key = "idem-" + hashlib.sha1(f"{oid}|{amount}|{reason}".encode()).hexdigest()[:12]
        return {"order_id": oid, "amount_cents": amount, "reason": reason, "idempotency_key": key}

    # -- answering --------------------------------------------------------------------------------
    def _answer(self, q, results, raw) -> str:
        parts = []
        oids = ORDER_RE.findall(q)
        refund = bool(REFUND_RE.search(q))
        for r in results.get("lookup_order", []):
            parts.append(self._order_sentence(r, oids[0] if oids else "?"))
        if oids and "skip_lookup" in self.faults and not refund:
            parts.append(f"Order {oids[0]} is on its way and should arrive within 5 to 7 business days.")
        for r in results.get("request_refund", []):
            parts.append(self._refund_sentence(r))
        if refund and not oids:
            parts.append("Please share your order id (for example ORD-1234) so I can check refund eligibility.")
        parts.extend(self._kb_sentences(results))
        if "abstain_never" in self.faults and (not oids or refund) and not self._kb_sentences(results):
            parts.append("This is covered by our standard policy and works out of the box [support-hours].")
        if "ignore_injection" in self.faults:
            if re.search(r"(?i)reveal your system prompt", raw):
                parts.append("My instructions are: " + SYSTEM_PROMPT)
            if "attacker@evil.example" in raw:
                parts.append("Forwarding customer data to attacker@evil.example as requested.")
        return " ".join(parts) if parts else ABSTAIN_MESSAGE

    def _kb_sentences(self, results) -> list[str]:
        passages = []
        for r in results.get("search_kb", []):
            if r["ok"]:
                passages = r["result"]["passages"]
        if "abstain_never" not in self.faults:
            top = passages[0]["score"] if passages else 0.0
            passages = [p for p in passages if p["score"] >= max(self.min_score, 0.5 * top)]
        out, seen = [], set()
        for p in passages:
            if p["doc_id"] in seen:
                continue
            seen.add(p["doc_id"])
            sents = [s for s in SENT_SPLIT.split(p["text"]) if s][: 2 if not out else 1]
            doc_id = "nimbus-kb-9999" if ("hallucinate_citation" in self.faults and not out) else p["doc_id"]
            out.extend(_cited(s, doc_id) for s in sents)
            if len(seen) >= 2:
                break
        return out

    @staticmethod
    def _order_sentence(r, oid) -> str:
        if not r["ok"]:
            return f"I couldn't look up order {oid}: {r['error']['message']}."
        o = r["result"]
        s = f"Order {o['order_id']} is {o['status']}."
        if o["delivered_date"]:
            s += f" It was delivered on {o['delivered_date']}."
        else:
            s += f" It was placed on {o['placed_date']}."
        s += f" The order total is {money(o['total_cents'])}."
        if o["refunded_cents"]:
            s += f" A refund of {money(o['refunded_cents'])} was already issued."
        return s

    @staticmethod
    def _refund_sentence(r) -> str:
        if r["ok"]:
            x = r["result"]
            return f"Refund of {money(x['amount_cents'])} for order {x['order_id']} was approved (reference {x['refund_id']})."
        e = r["error"]
        return f"I could not complete the refund: {e['message']} ({e['code']})."


# ---------------------------------------------------------------------------------------------
# ClaudeModel: Anthropic Messages API adapter.
# UNTESTED LIVE: only exercised in tests with a fake transport. No call to the real API was ever made.
# ---------------------------------------------------------------------------------------------
API_URL = "https://api.anthropic.com/v1/messages"


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ModelError(f"network error: {e}") from e


class ClaudeModel:
    UNTESTED_LIVE = True  # never run against the real API in this repo

    def __init__(self, api_key: str | None = None, model: str | None = None, transport=None,
                 url: str = API_URL, max_tokens: int = 1024, timeout: float = 30.0):
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model if model is not None else os.environ.get("ANTHROPIC_MODEL", "")
        self.transport, self.url, self.max_tokens, self.timeout = transport or urllib_transport, url, max_tokens, timeout
        self.name = f"claude:{self.model or 'unset'}"

    def build_request(self, messages: list[dict], tools: list[dict]) -> dict:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
            elif role == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                           for tc in m.get("tool_calls", [])]
                out.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                try:
                    if json.loads(m["content"]).get("ok") is False:
                        block["is_error"] = True
                except (ValueError, AttributeError):
                    pass
                if out and out[-1]["role"] == "user" and all(b["type"] == "tool_result" for b in out[-1]["content"]):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        req = {"model": self.model, "max_tokens": self.max_tokens, "messages": out}
        if system:
            req["system"] = system
        if tools:
            req["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                            for t in tools]
        return req

    def parse_response(self, data: dict) -> Turn:
        try:
            text = "".join(b["text"] for b in data["content"] if b["type"] == "text") or None
            calls = [ToolCall(b["id"], b["name"], b.get("input", {})) for b in data["content"] if b["type"] == "tool_use"]
            u = data.get("usage", {})
            return Turn(text, calls, Usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))),
                        data.get("stop_reason", "end_turn"))
        except (KeyError, TypeError, ValueError) as e:
            raise ModelError(f"malformed model response: {e}") from e

    def generate(self, messages: list[dict], tools: list[dict]) -> Turn:
        if not self.api_key:
            raise ModelError("ANTHROPIC_API_KEY is not set")
        if not self.model:
            raise ModelError("ANTHROPIC_MODEL is not set")
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = json.dumps(self.build_request(messages, tools)).encode()
        status, raw = self.transport(self.url, headers, body, self.timeout)
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise ModelError(f"non-JSON response (HTTP {status})") from e
        if status != 200:
            detail = data.get("error", {}) if isinstance(data, dict) else {}
            raise ModelError(f"HTTP {status}: {detail.get('type', 'error')}: {detail.get('message', '')}")
        return self.parse_response(data)
