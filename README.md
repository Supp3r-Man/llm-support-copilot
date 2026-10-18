# llm-support-copilot

![ci](https://github.com/Supp3r-Man/llm-support-copilot/actions/workflows/ci.yml/badge.svg)

An offline, stdlib-only **customer-support copilot service** for a fictional SaaS ("Nimbus Cloud"), plus the part that usually gets skipped: a **golden-set eval, a regression gate that knows noise from a real drop and has zero tolerance for forbidden refunds, a judge-calibration helper and a load test**. Retrieval, gated tools, an agent loop, guardrails, tracing and an HTTP API sit underneath.

> **What is verified, and what is not.** 189 tests pass locally on Windows with Python 3.13 (stdlib plus pytest). **CI has not run yet**: the workflow is committed but nothing has been pushed, so there is no green badge to claim. **No real LLM was ever used.** The model is a deterministic rule-based stand-in (`RuleModel`) with fault knobs, so every number below measures the *plumbing, the guardrails and the gate*, not how a real model answers. The `ClaudeModel` adapter and the `LLMJudge` are **untested against the live API** (fake transports and scripted models only). All data is synthetic and fictional. I also broke 12 key gate behaviours on purpose and the tests failed for each (see STEPS.md).

## Architecture (text overview)

```
question --> Agent loop (max steps, per-tool timeout, token budget, duplicate-call stop)
              |-- Model (RuleModel | ClaudeModel adapter | any object with .generate)
              |-- ToolRegistry: lookup_order, search_kb, request_refund (gated, idempotent, audited)
              |-- Retriever: stdlib BM25 over sections, area filter, latest-version-only
              '-- Guardrails: injection fencing, citation check, abstention, output redaction
 --> answer + verified citations + tool-call log + trace (one JSONL per request)
HTTP: POST /v1/ask, GET /healthz, GET /metrics (auth token, token-bucket rate limit, body cap)

copilot_eval:  golden.json (142 q) -> runner -> per-question grades + failure attribution
               variants (baseline, 6 fault knobs x guardrails on/off, bad configs)
               gate (paired bootstrap, Holm, slices, zero-tolerance critical) -> exit 0/1/2
               judge (Cohen's kappa)   load (real in-process server)   report (eval_results/)
```

## Eval: what is measured

142 hand-labelled questions (110 dev, 32 held out; stratified by type): KB lookups, two-document questions, superseded
policy versions (must use the latest), order lookups and authorization, refunds (12 allowed, 5 denial reasons), unanswerable
questions and two documents carrying prompt-injection text. 48 are flagged **critical**. `python -m copilot_eval hygiene`
checks duplicates, missing docs, dev/held-out leakage and re-derives every refund label with an independent policy oracle.

Per question: answer ok (required facts present, forbidden stale facts absent), citation precision and recall, tool-call
correctness, abstention correctness, injection resisted, **forbidden refund** (any approved refund that is not exactly the
labelled one), latency, token estimate (chars/4), trace id. Each failure is attributed to one of: policy violation,
retrieval miss, tool-use error, guardrail miss, generation error.

## Measured results (real runs, all 142 questions, deterministic; full tables in `eval_results/`)

| variant | pass | citation ok | tool ok | injection resisted | forbidden refunds |
|---|---|---|---|---|---|
| baseline | 0.873 | 0.901 | 1.000 | 1.000 | 0 |
| fault_skip_lookup (guardrails on) | 0.458 | 0.901 | 0.578 | 1.000 | 12 |
| fault_hallucinate_citation | 0.521 | 0.521 | 1.000 | 1.000 | 0 |
| fault_ignore_injection | 0.775 | 0.901 | 0.965 | 0.643 | 0 |
| fault_over_refund | 0.718 | 0.901 | 0.845 | 1.000 | 0 |
| fault_retriever_off | 0.422 | 0.535 | 0.437 | 1.000 | 0 |
| fault_abstain_never | 0.430 | 0.451 | 1.000 | 1.000 | 0 |
| raw_* (same faults, guardrails OFF) | 0.32 to 0.78 | | | ignore_injection 0.143 | skip_lookup 12 |
| bad_top_k1 / bad_latest_only_off | 0.859 / 0.859 | 0.887 / 0.894 | 1.000 | 1.000 | 0 |
| bad_refund_window / bad_refund_limit | 0.831 / 0.845 | 0.901 | 0.958 / 0.972 | 1.000 | 6 / 4 |

Baseline is **not perfect**: 0.873 overall (dev 0.882, held-out 0.844). Weak spots it really has:
5 of 14 unanswerable questions get a confident answer built from weakly matching passages (the min-score check is lexical),
the second retrieved document often adds an off-topic citation, and 3 of 10 two-document questions fail.

### Gate results (baseline vs candidate; `eval_results/gate_matrix.md`)

| candidate | gate | how it was caught |
|---|---|---|
| baseline vs itself | PASS (exit 0) | no false alarm |
| every fault knob, guardrails OFF | FAIL (exit 1), 6 of 6 | 5 to 12 statistical tests plus critical slice |
| every fault knob, guardrails ON | FAIL, 6 of 6 (none missed) | quality drop still visible; refund gate blocks the harm of over_refund and ignore_injection, the gate flags the attempt |
| guardrails_off, no fault | PASS | **invisible**: the rule model never misbehaves without a fault |
| bad_top_k1 | FAIL | slice `kb_multi` 0.70 to 0.00; overall pass rate barely moved (0.873 to 0.859) |
| bad_latest_only_off | FAIL | only the critical rule (2 stale-policy answers); no statistical test fired |
| bad_refund_limit | FAIL | only the critical rule (4 forbidden refunds); no statistical test fired |
| bad_refund_window | FAIL | 6 forbidden refunds plus 5 statistical tests |

**Power at this size** (n=142, baseline 0.873): a 4-point overall drop is detected with 87% power, a 2-point drop only 39%.
A 1 to 2 question regression is invisible to the statistics (a test asserts this); only critical-slice rules catch small safety regressions.

### Load test (`eval_results/load_*.json`, 400 requests each, one run each)

Windows 11 laptop, 8 logical cores, Python 3.13, real in-process HTTP server, **deterministic model, so this measures
service plumbing (HTTP, threads, retrieval, tools), not an LLM**; no network hop beyond loopback.

| scenario | throughput | p50 / p95 / p99 (ms) | non-2xx |
|---|---|---|---|
| 1 client, rate limit off | 78 req/s | 15.7 / 17.9 / 26.8 | 0 |
| 8 clients, rate limit off | 685 req/s | 11.4 / 15.7 / 17.2 | 0 |
| 32 clients, rate limit off | 678 req/s | 46.3 / 50.3 / 51.5 | 0 |
| 8 clients, default limiter (30 burst, 10/s) | 1520 req/s | 4.6 / 10.9 / 13.5 | 92% are 429, 0 errors |

Caveats: single runs; the 1-client p50 swung between 2 ms and 16 ms across runs (Windows timer/delayed-ACK behaviour, not
investigated). The first 32-client run showed p95 = 1035 ms: the listen backlog was the stdlib default of 5, fixed to 128.

## Run it

```
python -m pytest                                     # 189 tests (~25 s)
python -m copilot ask "How long do password reset links last?"
python -m copilot serve --port 8080                  # then POST /v1/ask {"question": "..."}
python -m copilot_eval hygiene                       # golden set checks
python -m copilot_eval run --variant baseline --out /tmp/base.json
python -m copilot_eval gate --baseline baseline --candidate raw_over_refund --out-dir out   # exit 1
python -m copilot_eval report                        # regenerate eval_results/
python -m copilot_eval load --clients 8 --requests 400
python -m copilot_eval calibrate                     # judge kappa vs human labels
```

Gate exit codes: **0** pass, **1** regression, **2** could not judge (crash, broken or unclean golden set, unknown config,
mismatched questions). A crash never looks like a regression. CI runs `scripts/ci_gate_check.py` after pytest.
Plugging in Claude: see STEPS.md Part B (`COPILOT_MODEL=claude`, `ANTHROPIC_API_KEY`); it has never been run live.

## Known gaps (honest)

- Citation support is a lexical token-overlap check (0.6): a copied but irrelevant passage passes.
- `abstain_never` with weakly matching passages is not stopped by guardrails; the baseline already answers 5 of 14 unanswerable questions this way.
- Token usage is a chars/4 estimate; latency is for a model that takes microseconds.
- The golden set is small and written by one person against a 21-file corpus; dev and held-out share authorship and style. Questions I wrote for the RuleModel's known weaknesses are in the set; a real model would need a new set.
- Slices under 8 questions are not statistically tested (listed in each report). Holm over ~17 tests costs power.
- The rule judge has kappa 0.33 against my 30 labels (not usable); the LLM judge was never run live.
- Load numbers are one run on one laptop; no soak test, no TLS, no real network.
- Paired bootstrap treats questions as independent; many are near-paraphrases.

License: MIT.
