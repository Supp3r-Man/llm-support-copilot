"""Builds the committed eval_results/ files from real runs: per-variant metrics, gate matrix, power, calibration."""
from __future__ import annotations

import json
from pathlib import Path

from copilot.models import FAULT_KNOBS

from .gate import compare
from .golden import hygiene, load_golden, select
from .judge import RuleJudge, calibrate, load_labels
from .metrics import summarize
from .runner import run_variant
from .stats import smallest_detectable_drop
from .variants import VARIANTS


def _write(out: Path, name: str, obj=None, text=None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if obj is not None:
        (out / f"{name}.json").write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    if text is not None:
        (out / f"{name}.md").write_text(text, encoding="utf-8", newline="\n")


def run_all(split: str = "all") -> dict:
    qs = select(load_golden(), split)
    return {name: run_variant(name, qs) for name in VARIANTS}


def eval_summary(out: Path, all_results: dict, split: str) -> dict:
    rows = {n: summarize(r) for n, r in all_results.items()}
    L = [f"# Eval results per variant ({split} split, n={rows['baseline']['n']}, deterministic RuleModel, no real LLM)", "",
         "| variant | pass | answer | citation | tool | abstain | injection resisted | forbidden refunds | tokens/q (est.) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, s in rows.items():
        m = s["metrics"]
        L.append(f"| {n} | {m['passed']:.3f} | {m['answer_ok']:.3f} | {m['citation_ok']:.3f} | {m['tool_ok']:.3f} | "
                 f"{m['abstain_ok']:.3f} | {m['injection_resisted']:.3f} | {s['forbidden_refunds']} | {s['mean_tokens']} |")
    b = rows["baseline"]
    L += ["", "Baseline pass rate by question type: " + ", ".join(f"{t} {v['pass']:.2f} (n={v['n']})" for t, v in b["by_type"].items()),
          "", "Baseline failure attribution: " + ", ".join(f"{k}={v}" for k, v in b["failures"].items()),
          f"Baseline latency (agent only): p50 {b['latency_ms']['p50']} ms, p95 {b['latency_ms']['p95']} ms."]
    _write(out, "eval_summary", {"split": split, "variants": rows}, "\n".join(L) + "\n")
    return rows


def gate_matrix(out: Path, all_results: dict, split: str) -> dict:
    base = all_results["baseline"]
    rows = {}
    for name, res in all_results.items():
        if name == "baseline":
            continue
        rep = compare(base, res)
        rows[name] = {"verdict": rep["verdict"], "exit_code": rep["exit_code"], "cand_pass": rep["cand_pass"],
                      "regressions": rep["regressions"], "critical_fail": rep["critical"]["fail"],
                      "forbidden_refunds": len(rep["critical"]["forbidden_refunds"]),
                      "injection_acted": len(rep["critical"]["injection_acted"]),
                      "newly_failed": rep["newly_failed_total"], "failure_modes": rep["failure_modes"]}
    self_rep = compare(base, base)
    L = [f"# Gate matrix: every planted regression vs baseline ({split} split, n={len(base)})", "",
         f"Baseline vs itself: **{self_rep['verdict']}** (exit {self_rep['exit_code']}). Baseline pass rate {self_rep['base_pass']}.", "",
         "| candidate | verdict | cand pass | newly failing | critical fail | tests flagged | main attribution |", "|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        top = max(r["failure_modes"], key=r["failure_modes"].get) if r["failure_modes"] else "-"
        L.append(f"| {n} | {r['verdict']} | {r['cand_pass']} | {r['newly_failed']} | {r['critical_fail']} | {len(r['regressions'])} | {top} |")
    missed_guard = [k for k in FAULT_KNOBS if rows[f"fault_{k}"]["verdict"] == "PASS"]
    seen_raw = [k for k in FAULT_KNOBS if rows[f"raw_{k}"]["verdict"] == "FAIL"]
    L += ["", f"Faults detected with guardrails OFF: {len(seen_raw)}/{len(FAULT_KNOBS)} ({', '.join(seen_raw)}).",
          "Faults the gate CANNOT see with guardrails ON (verdict PASS): " + (", ".join(missed_guard) or "none") + "."]
    obj = {"split": split, "baseline_vs_itself": self_rep["verdict"], "candidates": rows,
           "detected_raw": seen_raw, "missed_with_guardrails": missed_guard}
    _write(out, "gate_matrix", obj, "\n".join(L) + "\n")
    return obj


def power(out: Path, all_results: dict, split: str) -> dict:
    base = [r["passed"] for r in all_results["baseline"]]
    drop, curve = smallest_detectable_drop(base)
    L = [f"# Power simulation (n={len(base)}, baseline pass rate {sum(base) / len(base):.3f}, {split} split)", "",
         "Single overall pass-rate test, alpha 0.05, observed drop >= 0.02, 100 simulated candidates per row "
         "(each baseline pass flips to fail at random so the expected drop matches).", "", "| true drop | detection power |", "|---|---|"]
    L += [f"| {d:.2f} | {p:.2f} |" for d, p in curve]
    L += ["", f"Smallest drop detected with >= 80% power: **{drop}**. Holm correction over ~30 tests makes the per-test bar stricter; "
          "drops confined to a small slice are usually only caught by the zero-tolerance critical rule."]
    obj = {"n": len(base), "smallest_detectable_drop_80pct": drop, "curve": [{"drop": d, "power": p} for d, p in curve]}
    _write(out, "power", obj, "\n".join(L) + "\n")
    return obj


def calibration(out: Path) -> dict:
    rep = calibrate(RuleJudge(), load_labels())
    _write(out, "judge_calibration", rep)
    return rep


def build_all(out_dir="eval_results", split="all", log=print) -> None:
    out = Path(out_dir)
    problems = hygiene(load_golden())
    if problems:
        raise SystemExit("golden set hygiene failed: " + "; ".join(problems))
    res = run_all(split)
    eval_summary(out, res, split)
    log("eval_summary written")
    gm = gate_matrix(out, res, split)
    log(f"gate_matrix written; missed with guardrails on: {gm['missed_with_guardrails']}")
    power(out, res, split)
    log("power written")
    log(f"judge calibration (RuleJudge) kappa={calibration(out)['kappa']}")
