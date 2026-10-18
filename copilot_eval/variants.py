"""Named configuration variants the regression gate compares (baseline, each fault knob, guardrails off, bad configs)."""
from __future__ import annotations

from copilot.config import Config
from copilot.models import FAULT_KNOBS


class UnknownVariant(KeyError):
    pass


def _v(desc, faults=(), guardrails=True, **config):
    return {"desc": desc, "faults": tuple(faults), "guardrails": guardrails, "config": config}


VARIANTS: dict[str, dict] = {"baseline": _v("phase-1 defaults, guardrails on")}
for _f in FAULT_KNOBS:
    VARIANTS[f"fault_{_f}"] = _v(f"fault knob {_f}, guardrails ON", [_f])
for _f in FAULT_KNOBS:
    VARIANTS[f"raw_{_f}"] = _v(f"fault knob {_f}, guardrails OFF", [_f], guardrails=False)
VARIANTS.update({
    "guardrails_off": _v("no fault, guardrails off", guardrails=False),
    "bad_top_k1": _v("retriever top_k=1", top_k=1),
    "bad_latest_only_off": _v("retriever latest_only=False (superseded docs searchable)", latest_only=False),
    "bad_refund_window": _v("refund gate misconfigured: 365-day window", refund_window_days=365),
    "bad_refund_limit": _v("refund gate misconfigured: no amount limit", refund_max_cents=10**9),
})


def get_variant(name: str) -> dict:
    try:
        return VARIANTS[name]
    except KeyError:
        raise UnknownVariant(f"unknown variant {name!r}; known: {', '.join(VARIANTS)}") from None


def variant_config(name: str) -> Config:
    return Config().with_(**get_variant(name)["config"])
