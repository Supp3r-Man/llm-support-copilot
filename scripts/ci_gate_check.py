"""CI check for the regression gate itself (stdlib only).

Asserts: baseline vs itself exits 0 (no false alarm), a planted regression exits 1 (no missed alarm), a
critical-only regression exits 1, and an unknown candidate exits 2 (could not judge, not a regression).
Usage: python scripts/ci_gate_check.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [("baseline", "baseline", 0), ("baseline", "raw_over_refund", 1), ("baseline", "bad_refund_limit", 1),
         ("baseline", "no_such_variant", 2)]


def main() -> int:
    bad = 0
    for base, cand, want in CASES:
        p = subprocess.run([sys.executable, "-m", "copilot_eval", "gate", "--baseline", base, "--candidate", cand,
                            "--out-dir", str(ROOT / "eval_results" / "ci")], cwd=ROOT, capture_output=True, text=True)
        ok = p.returncode == want
        bad += not ok
        print(f"[{'ok' if ok else 'FAIL'}] gate {base} vs {cand}: exit {p.returncode}, expected {want}")
        if not ok:
            print(p.stdout + p.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
