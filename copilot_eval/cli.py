"""python -m copilot_eval hygiene|run|gate|report|load|calibrate"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gate import N_BOOT, gate
from .golden import GoldenError, hygiene, load_golden
from .variants import VARIANTS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="copilot_eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hygiene", help="check the golden set (duplicates, missing docs, leakage, label oracle)")
    h.add_argument("--golden")
    r = sub.add_parser("run", help="run one variant, write per-question results JSON")
    r.add_argument("--variant", default="baseline", choices=sorted(VARIANTS))
    r.add_argument("--split", default="all", choices=["all", "dev", "heldout"])
    r.add_argument("--out")
    r.add_argument("--trace-dir")
    r.add_argument("--model", default="rule", choices=["rule", "claude"], help="claude = ClaudeModel adapter (UNTESTED LIVE; needs ANTHROPIC_API_KEY and ANTHROPIC_MODEL)")
    g = sub.add_parser("gate", help="regression gate: exit 0 pass, 1 regression, 2 could not judge")
    g.add_argument("--baseline", required=True, help="variant name or results .json from `run`")
    g.add_argument("--candidate", required=True)
    g.add_argument("--split", default="all", choices=["all", "dev", "heldout"])
    g.add_argument("--golden")
    g.add_argument("--out-dir")
    g.add_argument("--n-boot", type=int, default=N_BOOT)
    rp = sub.add_parser("report", help="regenerate eval_results/ (summary, gate matrix, power, calibration)")
    rp.add_argument("--out-dir", default="eval_results")
    rp.add_argument("--split", default="all", choices=["all", "dev", "heldout"])
    ld = sub.add_parser("load", help="load test the real in-process HTTP server")
    ld.add_argument("--clients", type=int, default=8)
    ld.add_argument("--requests", type=int, default=400)
    ld.add_argument("--rate-capacity", type=float, default=1_000_000.0)
    ld.add_argument("--rate-refill", type=float, default=1_000_000.0)
    ld.add_argument("--out")
    sub.add_parser("calibrate", help="Cohen's kappa of the deterministic RuleJudge vs the human labels")
    a = p.parse_args(argv)

    if a.cmd == "gate":
        return gate(a.baseline, a.candidate, a.split, a.golden, a.out_dir, a.n_boot)
    if a.cmd == "hygiene":
        try:
            problems = hygiene(load_golden(a.golden))
        except GoldenError as e:
            print(f"hygiene: {e}", file=sys.stderr)
            return 2
        print("hygiene: ok" if not problems else "hygiene: " + "\n".join(problems))
        return 0 if not problems else 1
    if a.cmd == "run":
        from .runner import run_named
        model = None
        if a.model == "claude":
            from copilot.models import ClaudeModel
            model = ClaudeModel()
        res = run_named(a.variant, a.split, trace_dir=a.trace_dir, model=model)
        s = res["summary"]
        print(json.dumps({k: s[k] for k in ("n", "metrics", "forbidden_refunds", "failures", "latency_ms")}, indent=1))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
        return 0
    if a.cmd == "report":
        from .report import build_all
        build_all(a.out_dir, a.split)
        return 0
    if a.cmd == "load":
        from .load import run_load
        res = run_load(a.clients, a.requests, a.rate_capacity, a.rate_refill)
        print(json.dumps(res, indent=1))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
        return 0
    from .report import calibration
    print(json.dumps(calibration(Path("eval_results")), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
