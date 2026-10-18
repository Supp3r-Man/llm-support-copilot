# Eval results per variant (all split, n=142, deterministic RuleModel, no real LLM)

| variant | pass | answer | citation | tool | abstain | injection resisted | forbidden refunds | tokens/q (est.) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.873 | 0.915 | 0.901 | 1.000 | 0.965 | 1.000 | 0 | 1323.5 |
| fault_skip_lookup | 0.458 | 0.634 | 0.901 | 0.578 | 0.796 | 1.000 | 12 | 1095.2 |
| fault_hallucinate_citation | 0.521 | 0.528 | 0.521 | 1.000 | 0.669 | 1.000 | 0 | 1323.1 |
| fault_ignore_injection | 0.775 | 0.852 | 0.901 | 0.965 | 0.965 | 0.643 | 0 | 1355.4 |
| fault_over_refund | 0.718 | 0.761 | 0.901 | 0.845 | 0.965 | 1.000 | 0 | 1322.2 |
| fault_retriever_off | 0.422 | 0.521 | 0.535 | 0.437 | 0.521 | 1.000 | 0 | 718.3 |
| fault_abstain_never | 0.430 | 0.922 | 0.451 | 1.000 | 0.965 | 1.000 | 0 | 1342.8 |
| raw_skip_lookup | 0.458 | 0.634 | 0.901 | 0.578 | 0.965 | 1.000 | 12 | 1093.1 |
| raw_hallucinate_citation | 0.493 | 0.915 | 0.493 | 1.000 | 0.965 | 1.000 | 0 | 1321.0 |
| raw_ignore_injection | 0.775 | 0.915 | 0.838 | 0.965 | 0.965 | 0.143 | 0 | 1352.6 |
| raw_over_refund | 0.718 | 0.761 | 0.901 | 0.845 | 0.965 | 1.000 | 0 | 1320.1 |
| raw_retriever_off | 0.437 | 0.542 | 0.535 | 0.437 | 0.585 | 1.000 | 0 | 718.3 |
| raw_abstain_never | 0.317 | 0.859 | 0.338 | 1.000 | 0.901 | 1.000 | 0 | 1340.5 |
| guardrails_off | 0.873 | 0.915 | 0.901 | 1.000 | 0.965 | 1.000 | 0 | 1321.4 |
| bad_top_k1 | 0.859 | 0.866 | 0.887 | 1.000 | 0.965 | 1.000 | 0 | 1142.5 |
| bad_latest_only_off | 0.859 | 0.908 | 0.894 | 1.000 | 0.965 | 1.000 | 0 | 1318.5 |
| bad_refund_window | 0.831 | 0.873 | 0.901 | 0.958 | 0.965 | 1.000 | 6 | 1324.1 |
| bad_refund_limit | 0.845 | 0.887 | 0.901 | 0.972 | 0.965 | 1.000 | 4 | 1323.7 |

Baseline pass rate by question type: injection_doc 1.00 (n=14), kb 0.83 (n=36), kb_multi 0.70 (n=10), kb_superseded 0.50 (n=6), order_authz 1.00 (n=4), order_lookup 1.00 (n=18), order_notfound 1.00 (n=2), refund_allowed 1.00 (n=12), refund_denied 0.96 (n=24), refund_noorder 1.00 (n=2), unanswerable 0.64 (n=14)

Baseline failure attribution: policy_violation=0, retrieval_miss=0, tool_use_error=0, guardrail_miss=5, generation_error=13
Baseline latency (agent only): p50 0.748 ms, p95 1.634 ms.
