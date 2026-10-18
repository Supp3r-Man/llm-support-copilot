"""CLI: python -m copilot serve|ask|trace."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import build_agent
from .config import Config
from .models import ClaudeModel
from .server import Settings, make_server
from .tracing import format_trace, list_traces, load_trace


def _model():
    return ClaudeModel() if os.environ.get("COPILOT_MODEL", "rule") == "claude" else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="copilot")
    p.add_argument("--trace-dir", default=os.environ.get("COPILOT_TRACE_DIR", "traces"))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--host", default="127.0.0.1")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--customer")
    a.add_argument("--json", action="store_true")
    t = sub.add_parser("trace")
    tsub = t.add_subparsers(dest="tcmd", required=True)
    show = tsub.add_parser("show")
    show.add_argument("trace_id")
    tsub.add_parser("list")
    args = p.parse_args(argv)

    if args.cmd == "serve":
        model = _model()
        srv = make_server(settings=Settings(trace_dir=args.trace_dir), host=args.host, port=args.port,
                          model_factory=(lambda: model) if model else None)
        print(f"copilot listening on {srv.url} (traces in {args.trace_dir})", flush=True)
        srv.serve_forever()
        return 0
    if args.cmd == "ask":
        r = build_agent(Config(), model=_model(), trace_dir=args.trace_dir).run(args.question, args.customer)
        if args.json:
            print(json.dumps(r.to_dict(), indent=2))
        else:
            print(r.answer)
            print(f"\ncitations: {[c['doc_id'] for c in r.citations]}  trace: {r.trace_id}")
        return 0
    if args.tcmd == "list":
        print("\n".join(list_traces(args.trace_dir)))
        return 0
    try:
        print(format_trace(load_trace(args.trace_id, args.trace_dir)))
    except FileNotFoundError:
        print(f"trace {args.trace_id} not found in {args.trace_dir}", file=sys.stderr)
        return 1
    return 0
