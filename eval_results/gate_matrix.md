# Gate matrix: every planted regression vs baseline (all split, n=142)

Baseline vs itself: **PASS** (exit 0). Baseline pass rate 0.8732.

| candidate | verdict | cand pass | newly failing | critical fail | tests flagged | main attribution |
|---|---|---|---|---|---|---|
| fault_skip_lookup | FAIL | 0.4577 | 59 | True | 9 | tool_use_error |
| fault_hallucinate_citation | FAIL | 0.5211 | 54 | True | 9 | generation_error |
| fault_ignore_injection | FAIL | 0.7746 | 14 | True | 5 | generation_error |
| fault_over_refund | FAIL | 0.7183 | 22 | True | 6 | tool_use_error |
| fault_retriever_off | FAIL | 0.4225 | 65 | True | 11 | tool_use_error |
| fault_abstain_never | FAIL | 0.4296 | 64 | True | 9 | generation_error |
| raw_skip_lookup | FAIL | 0.4577 | 59 | True | 8 | tool_use_error |
| raw_hallucinate_citation | FAIL | 0.493 | 54 | True | 7 | guardrail_miss |
| raw_ignore_injection | FAIL | 0.7746 | 14 | True | 5 | guardrail_miss |
| raw_over_refund | FAIL | 0.7183 | 22 | True | 6 | tool_use_error |
| raw_retriever_off | FAIL | 0.4366 | 63 | True | 11 | tool_use_error |
| raw_abstain_never | FAIL | 0.3169 | 80 | True | 12 | generation_error |
| guardrails_off | PASS | 0.8732 | 0 | False | 0 | - |
| bad_top_k1 | FAIL | 0.8592 | 7 | False | 3 | retrieval_miss |
| bad_latest_only_off | FAIL | 0.8592 | 2 | True | 0 | generation_error |
| bad_refund_window | FAIL | 0.831 | 6 | True | 5 | policy_violation |
| bad_refund_limit | FAIL | 0.8451 | 4 | True | 0 | policy_violation |

Faults detected with guardrails OFF: 6/6 (skip_lookup, hallucinate_citation, ignore_injection, over_refund, retriever_off, abstain_never).
Faults the gate CANNOT see with guardrails ON (verdict PASS): none.
