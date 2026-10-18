"""Tool registry: JSON-Schema-style validation, gated idempotent refund write, full audit trail.

Every call returns a structured dict ({"ok": True, "result": ...} or {"ok": False, "error": {...}});
nothing raises through the agent loop.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config
from .retrieval import Retriever
from .tracing import span

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}


def validate_args(args, schema: dict) -> list[str]:
    """Validate `args` against a small JSON-Schema subset; returns a list of human-readable errors."""
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    errs = []
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            errs.append(f"missing required argument '{name}'")
    for name, val in args.items():
        spec = props.get(name)
        if spec is None:
            if schema.get("additionalProperties", True) is False:
                errs.append(f"unexpected argument '{name}'")
            continue
        typ = spec.get("type")
        py = _TYPES.get(typ)
        if py and (not isinstance(val, py) or (typ in ("integer", "number") and isinstance(val, bool))):
            errs.append(f"argument '{name}' must be of type {typ}")
            continue
        if typ == "string":
            if len(val) < spec.get("minLength", 0):
                errs.append(f"argument '{name}' is shorter than {spec['minLength']}")
            if len(val) > spec.get("maxLength", 10**9):
                errs.append(f"argument '{name}' is longer than {spec['maxLength']}")
        if typ in ("integer", "number"):
            if val < spec.get("minimum", float("-inf")):
                errs.append(f"argument '{name}' must be >= {spec['minimum']}")
        if "enum" in spec and val not in spec["enum"]:
            errs.append(f"argument '{name}' must be one of {spec['enum']}")
    return errs


def ok(result: dict) -> dict:
    return {"ok": True, "result": result}


def err(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable
    write: bool = False


class ToolRegistry:
    def __init__(self, config: Config | None = None, retriever: Retriever | None = None,
                 audit_path: str | Path | None = None):
        self.config = config or Config()
        self.retriever = retriever or Retriever.from_config(self.config)
        self.orders = {o["order_id"]: o for o in json.loads(Path(self.config.orders_path).read_text(encoding="utf-8"))}
        self.audit: list[dict] = []
        self.audit_path = Path(audit_path) if audit_path else None
        self._lock = threading.RLock()
        self._refunds: dict[str, dict] = {}
        self._idem: dict[str, tuple] = {}
        for oid, o in self.orders.items():
            if o.get("refunded_cents"):
                self._refunds[oid] = {"refund_id": f"RF-SEED-{oid[4:]}", "order_id": oid, "amount_cents": o["refunded_cents"]}
        self.tools: dict[str, Tool] = {}
        self.register(Tool("lookup_order", "Look up an order by id: status, items, totals, dates.", {
            "type": "object", "properties": {"order_id": {"type": "string", "minLength": 5, "maxLength": 20}},
            "required": ["order_id"], "additionalProperties": False}, self._lookup_order))
        self.register(Tool("search_kb", "Search the Nimbus help center. Optional product area filter.", {
            "type": "object", "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 500},
                "area": {"type": "string", "enum": ["billing", "account", "api", "storage", "shipping", "security", "general"]}},
            "required": ["query"], "additionalProperties": False}, self._search_kb))
        self.register(Tool("request_refund", "Request a refund for a delivered order (gated write, idempotent).", {
            "type": "object", "properties": {
                "order_id": {"type": "string", "minLength": 5, "maxLength": 20},
                "amount_cents": {"type": "integer", "minimum": 1},
                "reason": {"type": "string", "minLength": 3, "maxLength": 200},
                "idempotency_key": {"type": "string", "minLength": 8, "maxLength": 80}},
            "required": ["order_id", "amount_cents", "reason", "idempotency_key"], "additionalProperties": False},
            self._request_refund, write=True))

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def specs(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in self.tools.values()]

    # -- dispatch -----------------------------------------------------------------------------
    def call(self, name: str, args, ctx: dict | None = None) -> dict:
        ctx = ctx or {}
        t0 = time.perf_counter()
        with span(f"tool:{name}", tool=name, write=bool(self.tools.get(name) and self.tools[name].write)) as sp:
            try:
                tool = self.tools.get(name)
                if tool is None:
                    res = err("unknown_tool", f"no tool named '{name}'")
                else:
                    problems = validate_args(args, tool.schema)
                    res = err("invalid_arguments", "; ".join(problems)) if problems else tool.fn(args, ctx)
            except Exception as e:  # tool bugs become structured errors, never loop exceptions
                res = err("internal_error", f"{type(e).__name__} in tool '{name}'")
            code = None if res["ok"] else res["error"]["code"]
            sp.set(ok=res["ok"], error_code=code)
        self._audit({"ts": time.time(), "tool": name, "args": args if isinstance(args, dict) else str(args)[:200],
                     "customer_id": ctx.get("customer_id"), "ok": res["ok"], "error_code": code,
                     "duration_ms": round((time.perf_counter() - t0) * 1000, 3)})
        return res

    def _audit(self, entry: dict) -> None:
        with self._lock:
            self.audit.append(entry)
            if self.audit_path:
                with self.audit_path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(entry, sort_keys=True) + "\n")

    # -- tools --------------------------------------------------------------------------------
    def _lookup_order(self, a: dict, ctx: dict) -> dict:
        o = self.orders.get(a["order_id"])
        if o is None:
            return err("order_not_found", f"no order with id {a['order_id']}")
        if ctx.get("customer_id") and o["customer_id"] != ctx["customer_id"]:
            return err("forbidden", "this order does not belong to the requesting customer")
        view = {k: o[k] for k in ("order_id", "customer_id", "customer_name", "contact_email", "items", "total_cents",
                                  "status", "placed_date", "delivered_date")}
        view["refunded_cents"] = self._refunds.get(o["order_id"], {}).get("amount_cents", 0)
        return ok(view)

    def _search_kb(self, a: dict, ctx: dict) -> dict:
        with span("retrieval", query_len=len(a["query"]), area=a.get("area"), config_hash=self.config.config_hash()) as sp:
            ps = self.retriever.search(a["query"], k=self.config.top_k, area=a.get("area"))
            sp.set(hits=len(ps), top_doc=ps[0].doc_id if ps else None)
        return ok({"passages": [p.to_dict() for p in ps]})

    def _request_refund(self, a: dict, ctx: dict) -> dict:
        cfg = self.config
        with self._lock:
            key, oid = a["idempotency_key"], a["order_id"]
            fingerprint = (oid, a["amount_cents"], a["reason"])
            if key in self._idem:
                prev_fp, prev = self._idem[key]
                if prev_fp != fingerprint:
                    return err("idempotency_key_reuse", "this idempotency key was used with different arguments")
                return ok({**prev, "replayed": True})
            o = self.orders.get(oid)
            if o is None:
                return err("order_not_found", f"no order with id {oid}")
            if ctx.get("customer_id") and o["customer_id"] != ctx["customer_id"]:
                return err("forbidden", "this order does not belong to the requesting customer")
            if o["status"] != "delivered":
                return err("order_not_delivered", f"order is {o['status']}; only delivered orders can be refunded")
            if oid in self._refunds:
                return err("already_refunded", "this order has already been refunded once")
            if a["amount_cents"] > o["total_cents"]:
                return err("amount_exceeds_total", "refund amount cannot exceed the order total")
            if a["amount_cents"] > cfg.refund_max_cents:
                return err("amount_exceeds_policy_limit",
                           f"refunds above {cfg.refund_max_cents} cents need a human agent")
            age = (dt.date.fromisoformat(cfg.today) - dt.date.fromisoformat(o["delivered_date"])).days
            if age > cfg.refund_window_days:
                return err("outside_refund_window", f"delivered {age} days ago; window is {cfg.refund_window_days} days")
            result = {"refund_id": f"RF-{oid[4:]}-{len(self._refunds) + 1:04d}", "order_id": oid,
                      "amount_cents": a["amount_cents"], "reason": a["reason"], "status": "approved", "replayed": False}
            self._refunds[oid] = result
            self._idem[key] = (fingerprint, result)
            return ok(result)
