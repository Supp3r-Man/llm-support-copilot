"""Guardrails: injection flagging/wrapping, citation verification, abstention, output redaction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import SYSTEM_PROMPT
from .redact import redact_emails  # re-exported for callers
from .retrieval import Retriever, tokenize
from .tracing import span

ABSTAIN_MESSAGE = ("I don't have enough verified information to answer that. "
                   "I can route this to a human support agent.")
FAILSAFE_MESSAGE = "I wasn't able to finish this request safely. I can route it to a human support agent."

INJECTION_PATTERNS = [
    ("ignore_previous", re.compile(r"(?i)\bignore (all |any )?(previous|prior|above) (instructions|rules|messages)")),
    ("role_override", re.compile(r"(?i)(^|\s)system:|\byou are now\b|\bdeveloper mode\b")),
    ("reveal_prompt", re.compile(r"(?i)\b(reveal|print|show|leak)\b.{0,30}\b(system prompt|instructions)\b")),
    ("exfiltrate", re.compile(r"(?i)\b(send|email|forward)\b.{0,40}\b(customer data|credentials|passwords?)\b")),
    ("issue_refund_command", re.compile(r"(?i)\b(issue|process|approve)\s+a\s+refund\b")),
    ("conceal", re.compile(r"(?i)\bdo not (tell|inform|notify) the user\b")),
]
CITE_RE = re.compile(r"\[([a-z0-9][a-z0-9\-]*)\]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def scan_text(text: str) -> list[str]:
    return [name for name, rx in INJECTION_PATTERNS if rx.search(text)]


def wrap_untrusted(text: str, flags: list[str]) -> str:
    """Keep the text (so humans can audit it) but fence it as data. Never deletes content."""
    return f"<<UNTRUSTED flagged={','.join(flags)}>> {text} <</UNTRUSTED>>"


def scan_tool_result(obj, flags: list[str] | None = None):
    """Walk a tool result; wrap flagged strings. Returns (new_obj, flags)."""
    flags = [] if flags is None else flags
    if isinstance(obj, str):
        f = scan_text(obj)
        if f:
            flags.extend(x for x in f if x not in flags)
            return wrap_untrusted(obj, f), flags
        return obj, flags
    if isinstance(obj, dict):
        return {k: scan_tool_result(v, flags)[0] for k, v in obj.items()}, flags
    if isinstance(obj, list):
        return [scan_tool_result(v, flags)[0] for v in obj], flags
    return obj, flags


@dataclass
class Verification:
    text: str
    citations: list = field(default_factory=list)  # verified, unique, in order
    issues: list = field(default_factory=list)  # dropped sentences with reasons


def _support(sentence: str, doc_text: str) -> float:
    toks = {t for t in tokenize(CITE_RE.sub(" ", sentence)) if len(t) >= 3}
    if not toks:
        return 1.0
    return len(toks & set(tokenize(doc_text))) / len(toks)


def verify_answer(answer: str, retriever: Retriever, min_support: float = 0.6) -> Verification:
    """Drop sentences whose citations name unknown docs or are not lexically supported by the cited doc."""
    kept, cites, issues = [], [], []
    with span("guardrail:citations") as sp:
        for sent in [s for s in SENT_SPLIT.split(answer.strip()) if s]:
            ids = CITE_RE.findall(sent)
            bad = None
            for did in ids:
                if not retriever.has_doc(did):
                    bad = {"doc_id": did, "reason": "unknown_doc"}
                    break
                if _support(sent, retriever.doc_text(did)) < min_support:
                    bad = {"doc_id": did, "reason": "unsupported"}
                    break
            if bad:
                issues.append({**bad, "sentence": sent})
            else:
                kept.append(sent)
                cites.extend(i for i in ids if i not in cites)
        sp.set(kept=len(kept), dropped=len(issues))
    return Verification(" ".join(kept), cites, issues)


def check_output(text: str) -> tuple[str, list[str]]:
    """Final output pass: block system-prompt leakage, redact email-like strings."""
    issues = []
    if SYSTEM_PROMPT[:40].lower() in text.lower():
        issues.append("system_prompt_leak")
        text = FAILSAFE_MESSAGE
    red = redact_emails(text)
    if red != text:
        issues.append("email_redacted")
    return red, issues
