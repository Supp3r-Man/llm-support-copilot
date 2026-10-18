"""Judge calibration: measure a judge against human labels (Cohen's kappa) before trusting it to grade answers.

RuleJudge is deterministic (substring of the reference fact). LLMJudge wraps any copilot Model (for example
ClaudeModel) and is UNTESTED LIVE: it is only exercised with scripted fake models in the tests.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .stats import cohen_kappa

LABELS_PATH = Path(__file__).resolve().parent / "judge_labels.json"
MIN_KAPPA = 0.6  # common "substantial agreement" bar; below it the judge is not used for gating


class RuleJudge:
    name = "rule-judge"

    def __call__(self, question: str, answer: str, reference: str) -> str:
        return "pass" if reference.lower() in answer.lower() else "fail"


class LLMJudge:
    UNTESTED_LIVE = True
    PROMPT = ("You grade a customer-support answer against a reference fact. Reply with exactly one line: "
              "'VERDICT: PASS' if the answer states the reference fact correctly and adds no contradicting or "
              "unsupported claim, otherwise 'VERDICT: FAIL'.\nQuestion: {q}\nReference fact: {ref}\nAnswer: {a}")

    def __init__(self, model):
        self.model, self.name = model, f"llm-judge:{getattr(model, 'name', 'model')}"

    def __call__(self, question: str, answer: str, reference: str) -> str:
        text = self.model.generate([{"role": "user", "content": self.PROMPT.format(q=question, ref=reference, a=answer)}], []).text or ""
        m = re.search(r"(?i)VERDICT:\s*(PASS|FAIL)", text)
        return m.group(1).lower() if m else "unparseable"


def load_labels(path=None) -> list[dict]:
    return json.loads(Path(path or LABELS_PATH).read_text(encoding="utf-8"))["items"]


def calibrate(judge, items: list[dict], min_kappa: float = MIN_KAPPA) -> dict:
    """items: {question, reference, answer, label in pass|fail}. Unparseable judge output counts as a miss."""
    human = [i["label"] for i in items]
    got = [judge(i["question"], i["answer"], i["reference"]) for i in items]
    labels = ["pass", "fail"]
    conf = {h: {j: sum(1 for a, b in zip(human, got) if a == h and b == j) for j in labels + ["unparseable"]} for h in labels}
    k = cohen_kappa(human, [g if g in labels else ("fail" if h == "pass" else "pass") for h, g in zip(human, got)])
    return {"judge": getattr(judge, "name", "judge"), "n": len(items), "kappa": round(k, 3),
            "agreement": round(sum(a == b for a, b in zip(human, got)) / len(items), 3), "confusion_human_vs_judge": conf,
            "usable": k >= min_kappa, "min_kappa": min_kappa}
