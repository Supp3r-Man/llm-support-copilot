import json
import subprocess
import sys
from pathlib import Path

import pytest
from eval_helpers import failed, rec

from copilot_eval.gate import GateError, compare, render_md
from copilot_eval.stats import cohen_kappa, detection_power, holm, paired_bootstrap, smallest_detectable_drop

ROOT = Path(__file__).resolve().parents[1]


# ---- statistics ----------------------------------------------------------------------------------------
def test_holm_known_values():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}
    assert holm({"x": 0.9})["x"] == 0.9 and holm({"x": 0.9, "y": 0.8})["y"] == pytest.approx(1.0)


def test_kappa_hand_computed():
    assert cohen_kappa(["p", "p", "f", "f"], ["p", "f", "f", "f"]) == pytest.approx(0.5)
    assert cohen_kappa(["p", "f"] * 5, ["p", "f"] * 5) == 1.0
    assert cohen_kappa(["p", "f"] * 5, ["f", "p"] * 5) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_bootstrap_is_deterministic_and_sensible():
    d = [0.0] * 90 + [-1.0] * 10
    a, b = paired_bootstrap(d, 500, seed=3), paired_bootstrap(d, 500, seed=3)
    assert a == b and a["mean"] == pytest.approx(-0.1) and a["ci_lo"] < -0.1 < a["ci_hi"] < 0
    assert a["p_drop"] < 0.01
    assert paired_bootstrap([0.0] * 50, 300)["p_drop"] > 0.99
    assert paired_bootstrap([1.0] * 50, 300)["p_drop"] > 0.99  # improvements are never a "drop"
    assert paired_bootstrap([], 10)["p_drop"] == 1.0


def test_power_grows_with_drop_and_is_seeded():
    base = [True] * 100 + [False] * 14
    lo, hi = detection_power(base, 0.01, n_sims=40), detection_power(base, 0.10, n_sims=40)
    assert lo < 0.3 and hi > 0.95
    assert detection_power(base, 0.05, n_sims=30) == detection_power(base, 0.05, n_sims=30)


def test_power_at_golden_set_size(eval_runs):
    base = [r["passed"] for r in eval_runs["baseline"]]
    drop, curve = smallest_detectable_drop(base, n_sims=60, n_boot=250)
    assert drop is not None and drop <= 0.08  # a ~5-8 point overall drop is detectable at n~140; 1-2 points is not
    assert dict(curve)[0.02] < 0.6


# ---- gate logic on synthetic results --------------------------------------------------------------------
def base_set(n=100):
    return [rec(i, critical=(i < 20), type=("kb" if i % 2 else "order_lookup"), area=("general" if i % 2 else "orders")) for i in range(n)]


def test_self_comparison_passes():
    b = base_set()
    r = compare(b, b, n_boot=300)
    assert r["verdict"] == "PASS" and r["exit_code"] == 0 and not r["regressions"] and not r["critical"]["fail"]


def test_forbidden_refund_fails_regardless_of_statistics():
    b = base_set()
    c = base_set()
    c[3] = rec(3, critical=True, forbidden_refund=True)  # still 'passed' and one question of 100: statistically nothing
    r = compare(b, c, n_boot=300)
    assert not r["regressions"] and r["critical"]["forbidden_refunds"] == ["q003"] and r["verdict"] == "FAIL"


def test_forbidden_refund_fails_even_if_baseline_also_failed_that_question():
    b = base_set()
    b[5] = failed(5, critical=True)
    c = base_set()
    c[5] = failed(5, critical=True, forbidden_refund=True)
    assert compare(b, c, n_boot=200)["verdict"] == "FAIL"


def test_newly_failing_critical_question_fails_but_noncritical_noise_passes():
    b, c = base_set(), base_set()
    c[50] = failed(50)  # non-critical single flip: noise
    assert compare(b, c, n_boot=300)["verdict"] == "PASS"
    c[7] = failed(7, critical=True)
    r = compare(b, c, n_boot=300)
    assert r["verdict"] == "FAIL" and r["critical"]["newly_failed"][0]["id"] == "q007"


def test_injection_acted_on_fails():
    b, c = base_set(), base_set()
    b[1] = rec(1, critical=True, injection_resisted=True)
    c[1] = rec(1, critical=True, injection_resisted=False, passed=False, failure="policy_violation")
    assert compare(b, c, n_boot=200)["critical"]["injection_acted"] == ["q001"]
    # acting on an injection fails the gate on its own, even if the question is otherwise graded as passing
    for i in (2, 3, 4):  # several resisted injection questions so the statistics alone cannot flag one failure
        b[i] = c[i] = rec(i, critical=True, injection_resisted=True)
    c[1] = rec(1, critical=True, injection_resisted=False)
    r = compare(b, c, n_boot=200)
    assert not r["regressions"]
    assert r["critical"]["injection_acted"] == ["q001"] and not r["critical"]["newly_failed"] and r["verdict"] == "FAIL"


def test_slice_regression_is_caught_when_overall_is_flat():
    b = [failed(i, type="A", area="a") if i < 25 else rec(i, type="A", area="a") for i in range(50)]
    b += [rec(100 + i, type="B", area="b") for i in range(50)]
    c = [rec(i, type="A", area="a") for i in range(50)] + [failed(100 + i, type="B", area="b") if i < 25 else rec(100 + i, type="B", area="b") for i in range(50)]
    r = compare(b, c, n_boot=500)
    t = {x["name"]: x for x in r["tests"]}
    assert t["overall:passed"]["delta"] == 0 and not t["overall:passed"]["regression"]
    assert t["type=B"]["regression"] and not t["type=A"]["regression"] and r["verdict"] == "FAIL"


def test_improvement_never_fails():
    b = [failed(i) if i < 30 else rec(i) for i in range(100)]
    c = [rec(i) for i in range(100)]
    assert compare(b, c, n_boot=300)["verdict"] == "PASS"


def test_small_effect_below_min_drop_is_not_flagged_even_when_significant():
    b = [rec(i) for i in range(2000)]
    c = [failed(i) if 20 <= i < 40 else rec(i) for i in range(2000)]  # 1 point drop: significant, but below min_drop 0.02
    r = compare(b, c, n_boot=500)
    t = {x["name"]: x for x in r["tests"]}["overall:passed"]
    assert t["p_holm"] < 0.05 and t["delta"] == pytest.approx(-0.01) and not r["regressions"]


def test_holm_correction_stops_scattered_noise_from_failing_the_gate():
    b = base_set()
    c = base_set()
    for i in (30, 40, 50, 60):  # 4 scattered non-critical failures out of 100
        c[i] = failed(i)
    r = compare(b, c, n_boot=2000)
    t = {x["name"]: x for x in r["tests"]}["overall:passed"]
    assert t["p"] < 0.05 < t["p_holm"] and t["delta"] <= -0.02  # raw test would fire, Holm does not
    assert r["verdict"] == "PASS"


def test_holm_adjusted_p_is_at_least_raw_p_and_tiny_slices_are_reported_untestable():
    b = base_set()
    b.append(rec(900, type="rare", area="rare"))
    r = compare(b, b, n_boot=200)
    assert all(t["p_holm"] >= t["p"] - 1e-9 for t in r["tests"])
    assert {"slice": "type=rare", "n": 1} in r["untestable_slices"]


def test_mismatched_question_sets_cannot_be_judged():
    with pytest.raises(GateError):
        compare(base_set(10), base_set(11), n_boot=50)
    with pytest.raises(GateError):
        compare([], [], n_boot=50)


def test_markdown_report_mentions_verdict_and_critical():
    b = base_set()
    r = compare(b, b, n_boot=100)
    r.update(baseline="a", candidate="b", split="all")
    md = render_md(r)
    assert "Verdict: PASS" in md and "zero tolerance" in md and "Holm" in md


# ---- the gate as a real subprocess ------------------------------------------------------------------------
def gate_cli(*args):
    p = subprocess.run([sys.executable, "-m", "copilot_eval", "gate", "--n-boot", "300", *args], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def test_cli_baseline_vs_itself_exits_0_and_writes_reports(tmp_path):
    code, out = gate_cli("--baseline", "baseline", "--candidate", "baseline", "--out-dir", str(tmp_path))
    assert code == 0 and "PASS" in out
    rep = json.loads((tmp_path / "baseline__vs__baseline.json").read_text("utf-8"))
    assert rep["verdict"] == "PASS" and (tmp_path / "baseline__vs__baseline.md").read_text("utf-8").startswith("# Gate report")


def test_cli_planted_regression_exits_1(tmp_path):
    code, out = gate_cli("--baseline", "baseline", "--candidate", "raw_over_refund", "--out-dir", str(tmp_path))
    assert code == 1 and "FAIL" in out
    assert json.loads((tmp_path / "baseline__vs__raw_over_refund.json").read_text("utf-8"))["exit_code"] == 1


def test_cli_critical_only_regression_exits_1():
    code, out = gate_cli("--baseline", "baseline", "--candidate", "bad_refund_limit")
    assert code == 1 and "critical=FAIL" in out and "regressions=none" in out


def test_cli_unknown_variant_exits_2():
    code, out = gate_cli("--baseline", "baseline", "--candidate", "does_not_exist")
    assert code == 2 and "COULD NOT JUDGE" in out


def test_cli_broken_golden_exits_2(tmp_path):
    bad = tmp_path / "golden.json"
    bad.write_text("{oops", encoding="utf-8")
    assert gate_cli("--baseline", "baseline", "--candidate", "baseline", "--golden", str(bad))[0] == 2


def test_cli_golden_failing_hygiene_exits_2(tmp_path):
    g = json.loads((ROOT / "copilot_eval" / "golden.json").read_text("utf-8"))
    g["questions"][1]["id"] = g["questions"][0]["id"]
    p = tmp_path / "golden.json"
    p.write_text(json.dumps(g), encoding="utf-8")
    code, out = gate_cli("--baseline", "baseline", "--candidate", "baseline", "--golden", str(p))
    assert code == 2 and "hygiene" in out


def test_cli_crash_is_exit_2_not_a_regression(tmp_path):
    broken = tmp_path / "cand.json"
    ids = [q["id"] for q in json.loads((ROOT / "copilot_eval" / "golden.json").read_text("utf-8"))["questions"]]
    broken.write_text(json.dumps({"variant": "x", "results": [{"id": i} for i in ids]}), encoding="utf-8")  # ids match, fields missing
    code, out = gate_cli("--baseline", "baseline", "--candidate", str(broken))
    assert code == 2 and "COULD NOT JUDGE" in out and "KeyError" in out
    mism = tmp_path / "mism.json"
    mism.write_text(json.dumps({"variant": "x", "results": [{"id": "q001"}]}), encoding="utf-8")  # different question set
    assert gate_cli("--baseline", "baseline", "--candidate", str(mism))[0] == 2


def test_cli_accepts_saved_results_files(tmp_path):
    f = tmp_path / "base.json"
    r = subprocess.run([sys.executable, "-m", "copilot_eval", "run", "--variant", "baseline", "--out", str(f)], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    assert gate_cli("--baseline", str(f), "--candidate", "baseline")[0] == 0
