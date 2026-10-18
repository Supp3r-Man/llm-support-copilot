"""Golden set loading, splits and the hygiene check (duplicates, missing docs, leakage, label oracle)."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from copilot.config import Config
from copilot.retrieval import load_corpus, tokenize

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"
REQUIRED = ("id", "split", "type", "area", "critical", "question", "expect")
EXPECT_KEYS = ("docs", "facts", "forbidden_facts", "tools", "abstain", "refund", "refund_code", "refund_amount_cents",
               "order_id", "injection")
LEAK_JACCARD = 0.85


class GoldenError(Exception):
    """The golden set could not be loaded or is structurally broken."""


def load_golden(path=None) -> dict:
    p = Path(path) if path else GOLDEN_PATH
    try:
        g = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise GoldenError(f"cannot read golden set {p}: {e}") from e
    if not isinstance(g, dict) or not isinstance(g.get("questions"), list) or not g["questions"]:
        raise GoldenError("golden set has no 'questions' list")
    return g


def select(golden: dict, split: str = "all") -> list[dict]:
    if split not in ("all", "dev", "heldout"):
        raise GoldenError(f"unknown split {split!r}")
    return [q for q in golden["questions"] if split == "all" or q.get("split") == split]


def oracle(order: dict | None, amount_cents: int | None, cfg: Config | None = None) -> str:
    """Independent policy re-implementation used only to cross-check the hand labels: returns 'allowed' or a code."""
    cfg = cfg or Config()
    if order is None:
        return "order_not_found"
    if order["status"] != "delivered":
        return "order_not_delivered"
    if order["refunded_cents"]:
        return "already_refunded"
    amt = amount_cents if amount_cents is not None else order["total_cents"]
    if amt > order["total_cents"]:
        return "amount_exceeds_total"
    if amt > cfg.refund_max_cents:
        return "amount_exceeds_policy_limit"
    age = (dt.date.fromisoformat(cfg.today) - dt.date.fromisoformat(order["delivered_date"])).days
    return "outside_refund_window" if age > cfg.refund_window_days else "allowed"


def _toks(q: str) -> frozenset:
    return frozenset(tokenize(q))


def hygiene(golden: dict, cfg: Config | None = None) -> list[str]:
    """Returns a list of problems; empty means the golden set is fit to gate on."""
    cfg = cfg or Config()
    problems: list[str] = []
    qs = golden.get("questions", [])
    docs = {d.doc_id for d in load_corpus(cfg.corpus_dir)}
    orders = {o["order_id"]: o for o in json.loads(Path(cfg.orders_path).read_text(encoding="utf-8"))}
    ids, seen_q = set(), {}
    for q in qs:
        qid = q.get("id", "?")
        miss = [k for k in REQUIRED if k not in q]
        if miss:
            problems.append(f"{qid}: missing keys {miss}")
            continue
        e = q["expect"]
        if set(EXPECT_KEYS) - set(e):
            problems.append(f"{qid}: expect is missing {sorted(set(EXPECT_KEYS) - set(e))}")
            continue
        if qid in ids:
            problems.append(f"{qid}: duplicate id")
        ids.add(qid)
        if q["split"] not in ("dev", "heldout"):
            problems.append(f"{qid}: bad split {q['split']!r}")
        key = (re.sub(r"\W+", " ", q["question"].lower()).strip(), q.get("customer_id"))
        if key in seen_q:
            problems.append(f"{qid}: duplicate question text of {seen_q[key]}")
        seen_q[key] = qid
        for d in e["docs"]:
            if d not in docs:
                problems.append(f"{qid}: expected doc {d!r} is not in the corpus")
        if e["order_id"] and e["order_id"] not in orders and not e["facts"] == ["no order with id"]:
            problems.append(f"{qid}: order {e['order_id']} does not exist")
        if not e["facts"] and not e["abstain"]:
            problems.append(f"{qid}: no required facts and not an abstention case")
        if e["abstain"] and (e["docs"] or e["refund"] != "none"):
            problems.append(f"{qid}: abstention case must not expect docs or refunds")
        if e["refund"] in ("allowed", "denied") and e["order_id"] in orders:
            m = re.search(r"\$(\d+)", q["question"])
            got = oracle(orders[e["order_id"]], int(m.group(1)) * 100 if m else None, cfg)
            want = "allowed" if e["refund"] == "allowed" else e["refund_code"]
            if got != want:
                problems.append(f"{qid}: label says {want} but the policy oracle says {got}")
    splits = {q.get("split") for q in qs}
    if not {"dev", "heldout"} <= splits:
        problems.append("both a dev and a heldout split are required")
    dev = [(q["id"], _toks(q["question"])) for q in qs if q.get("split") == "dev" and "question" in q]
    held = [(q["id"], _toks(q["question"])) for q in qs if q.get("split") == "heldout" and "question" in q]
    for hid, ht in held:
        for did, dtk in dev:
            union = ht | dtk
            if union and len(ht & dtk) / len(union) >= LEAK_JACCARD:
                problems.append(f"leakage: heldout {hid} is a near-duplicate of dev {did}")
    return problems
