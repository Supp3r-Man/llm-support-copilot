"""Paired bootstrap, Holm correction, Cohen's kappa and a power simulation. Stdlib only, seeded."""
from __future__ import annotations

import random


def paired_bootstrap(diffs: list[float], n_boot: int = 2000, seed: int = 1) -> dict:
    """diffs = candidate - baseline per question. One-sided p for 'true change is a drop' + 95% percentile CI."""
    n = len(diffs)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_drop": 1.0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot))
    return {"mean": sum(diffs) / n, "ci_lo": boots[int(0.025 * n_boot)], "ci_hi": boots[int(0.975 * n_boot) - 1],
            "p_drop": (sum(1 for b in boots if b >= 0) + 1) / (n_boot + 1)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def cohen_kappa(a: list, b: list) -> float:
    """Chance-corrected agreement between two label lists (1 = perfect, 0 = chance)."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be non-empty and equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def detection_power(base: list[bool], drop: float, alpha: float = 0.05, min_drop: float = 0.02, n_sims: int = 100,
                    n_boot: int = 300, seed: int = 5) -> float:
    """Share of simulated candidates (each baseline pass flips to fail so the expected pass-rate drop is `drop`)
    that the single overall pass-rate test flags (one-sided bootstrap p < alpha and observed drop >= min_drop)."""
    rng = random.Random(seed)
    p0 = sum(base) / len(base)
    q = min(1.0, drop / p0) if p0 else 0.0
    hits = 0
    for s in range(n_sims):
        diffs = [(-1.0 if (b and rng.random() < q) else 0.0) for b in base]
        r = paired_bootstrap(diffs, n_boot, seed=seed * 1000 + s)
        hits += r["mean"] <= -min_drop and r["p_drop"] < alpha
    return hits / n_sims


def smallest_detectable_drop(base: list[bool], target_power: float = 0.8, alpha: float = 0.05, grid=None, **kw):
    """Smallest drop on the grid detected with >= target_power; also returns the whole curve."""
    grid = grid or [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    curve = [(d, detection_power(base, d, alpha=alpha, **kw)) for d in grid]
    return next((d for d, p in curve if p >= target_power), None), curve
