"""Redaction helpers shared by guardrails (outputs) and tracing (stored attributes)."""
from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+")
SENSITIVE_KEYS = ("api_key", "apikey", "authorization", "token", "password", "secret", "email")
REDACTED = "[REDACTED]"
EMAIL_MASK = "[redacted-email]"


def redact_emails(text: str) -> str:
    return EMAIL_RE.sub(EMAIL_MASK, text)


def redact_text(text: str) -> str:
    return BEARER_RE.sub("Bearer " + REDACTED, redact_emails(text))


def redact_value(value, key: str = ""):
    """Recursively redact: sensitive key names are masked, strings are scrubbed."""
    k = key.lower()
    if k and (k in SENSITIVE_KEYS or k.endswith(tuple("_" + s for s in SENSITIVE_KEYS))):
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(k): redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(v) for v in value]
    return value
