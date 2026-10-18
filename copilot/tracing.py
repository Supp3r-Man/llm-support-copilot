"""Nested spans written as JSONL (one file per trace) plus a text viewer."""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .redact import redact_value

_trace_var: ContextVar = ContextVar("copilot_trace", default=None)
_span_var: ContextVar = ContextVar("copilot_span", default=None)


class Span:
    def __init__(self, name: str, parent_id, attrs: dict):
        self.span_id = uuid.uuid4().hex[:8]
        self.parent_id = parent_id
        self.name = name
        self.attrs = dict(attrs)
        self.status = "ok"
        self.start = time.time()
        self._t0 = time.perf_counter()
        self.duration_ms = 0.0

    def set(self, **kv) -> None:
        self.attrs.update(kv)

    def to_dict(self, trace_id: str) -> dict:
        return {"trace_id": trace_id, "span_id": self.span_id, "parent_id": self.parent_id, "name": self.name,
                "start": round(self.start, 6), "duration_ms": round(self.duration_ms, 3), "status": self.status,
                "attrs": redact_value(self.attrs)}


class Trace:
    def __init__(self):
        self.trace_id = uuid.uuid4().hex[:16]
        self.spans: list[Span] = []
        self.lock = threading.Lock()


class Tracer:
    def __init__(self, trace_dir: str | Path | None = None):
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.last_trace_id: str | None = None

    def path_for(self, trace_id: str) -> Path | None:
        return self.trace_dir / f"{trace_id}.jsonl" if self.trace_dir else None

    @contextmanager
    def trace(self, name: str, **attrs):
        """Open a root span; on exit the whole trace is written to <trace_dir>/<trace_id>.jsonl."""
        tr = Trace()
        t_tok, s_tok = _trace_var.set(tr), _span_var.set(None)
        try:
            with span(name, **attrs) as root:
                yield tr, root
        finally:
            _trace_var.reset(t_tok)
            _span_var.reset(s_tok)
            self.last_trace_id = tr.trace_id
            p = self.path_for(tr.trace_id)
            if p:
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("w", encoding="utf-8", newline="\n") as f:
                    for sp in sorted(tr.spans, key=lambda s: s.start):
                        f.write(json.dumps(sp.to_dict(tr.trace_id), sort_keys=True) + "\n")


@contextmanager
def span(name: str, **attrs):
    """Child span of whatever span is current; a harmless no-op outside a trace."""
    tr = _trace_var.get()
    if tr is None:
        yield Span(name, None, attrs)
        return
    sp = Span(name, _span_var.get(), attrs)
    tok = _span_var.set(sp.span_id)
    try:
        yield sp
    except BaseException as e:
        sp.status = "error"
        sp.attrs["error"] = type(e).__name__
        raise
    finally:
        sp.duration_ms = (time.perf_counter() - sp._t0) * 1000
        _span_var.reset(tok)
        with tr.lock:
            tr.spans.append(sp)


def load_trace(trace_id: str, trace_dir: str | Path) -> list[dict]:
    p = Path(trace_dir) / f"{trace_id}.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_traces(trace_dir: str | Path) -> list[str]:
    d = Path(trace_dir)
    return [p.stem for p in sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)] if d.exists() else []


def format_trace(spans: list[dict]) -> str:
    kids: dict = {}
    for s in spans:
        kids.setdefault(s["parent_id"], []).append(s)
    lines = [f"trace {spans[0]['trace_id']}  ({len(spans)} spans)"] if spans else ["empty trace"]

    def walk(pid, depth):
        for s in sorted(kids.get(pid, []), key=lambda x: x["start"]):
            attrs = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in sorted(s["attrs"].items()))
            lines.append(f"{'  ' * depth}{s['name']}  {s['duration_ms']:.2f}ms  [{s['status']}]  {attrs}".rstrip())
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    return "\n".join(lines)
