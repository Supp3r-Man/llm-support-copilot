# STEPS

## Part A: what was done, in order (both phases)

Phase 1 (service, committed first, 115 tests, 25 of 25 mutations killed)
1. Config: one frozen `Config` (limits, retriever settings, refund policy, pinned "today" 2026-10-18) with a config hash recorded in every trace.
2. Corpus and retrieval: 21 synthetic Markdown files (two versions of the billing policy, two documents carrying injection text); stdlib BM25 over sections with area filter and latest-version-only.
3. Tools: `lookup_order`, `search_kb`, `request_refund` with schema validation, an idempotent gated refund (7 ordered denial checks) and an audit trail; seeded orders.
4. Models: deterministic `RuleModel` with six fault knobs, plus a `ClaudeModel` adapter exercised only with a fake transport.
5. Agent loop (step limit, tool timeout, token budget, duplicate-call stop), guardrails (injection fencing, citation check, abstention, output redaction), tracing (one JSONL per request), HTTP server (auth, token-bucket rate limit, body cap, metrics).
6. Mutation check of phase 1: 25 planted bugs, 25 killed. Known gaps left on purpose and measured later: lexical citation support, `abstain_never` with weak hits, chars/4 token estimate.

Phase 2 (evals, gate, load test, docs)
7. Golden set: wrote a compact table (`copilot_eval/build_golden.py`) that emits `golden.json`: 142 questions, 110 dev and 32 held out (every 4th per type), 48 critical. Order facts come from `orders.json`; refund labels are hand-written and cross-checked by an independent oracle in the hygiene check (duplicates, missing docs, leakage, schema).
8. Runner and grading: one fresh agent per question, capture of what `search_kb` returned, grades for answer, citations, tools, abstention, injection, forbidden refund, and a first-match failure attribution.
9. Variants: baseline, each fault knob with guardrails on and off, guardrails off, `top_k=1`, `latest_only=False`, two misconfigured refund gates.
10. Gate: paired bootstrap (seeded), one-sided, Holm across about 17 tests, minimum effect 0.02, per-slice tests (slices under 8 not tested), zero-tolerance critical rules, exit codes 0/1/2, JSON and Markdown reports.
11. Judge calibration (Cohen's kappa; rule judge measured at 0.33, not usable; LLM judge written, untested live), load test, power simulation, 74 new tests (one in the server tests), CI step, docs.

Problems hit in phase 2 (and what changed)
- First golden set had only single-document questions: `top_k=1` made the overall pass rate go UP (0.886 to 0.924). Added 10 two-document questions (kept all of them; the baseline fails 3). Now `top_k=1` is caught by the `kb_multi` slice while the overall rate barely moves.
- `latest_only=False` was invisible to the statistics (6 superseded questions, too few to test). I marked superseded-policy questions critical, so it is now caught, by the zero-tolerance rule only.
- A crash inside the gate would exit 1 (Python's default), which looks like a regression. Now every unexpected exception exits 2, with a test using a malformed results file.
- Load test at 32 clients showed p95 = 1035 ms: the stdlib listen backlog of 5 made Windows retry dropped connections after about 1 s. Raised to 128 (test added); p95 became 50 ms.
- Mutation check of the gate: 12 literal mutations. I predicted that removing Holm or the minimum-effect rule would survive the first tests and strengthened both beforehand. The real run still found two survivors: (a) the "acted on injection" critical rule, whose first fix was itself flawed (a single injection question was accidentally significant after Holm, so the statistics flagged it instead); (b) "crash mapped to exit 2", because my crash test used a results file with different question ids, which is a different error path. Both tests were rewritten and re-run.
- `guardrails_off` with no fault cannot be detected: the rule model only misbehaves through fault knobs. Reported, not hidden.
- 1-client latency is bimodal on Windows (2 ms or 16 ms between runs); not investigated, reported as a caveat.

Mutation result (final full re-run, script kept outside the repo): 12 mutations, 12 killed, 0 survivors, 0 PATTERN MISSED.

## Part B: what you run

Setup: Python 3.11+ and `python -m pip install pytest`. No other dependency.

1. Tests: `python -m pytest` (189 tests, about 25 s).
2. Ask the copilot once: `python -m copilot ask "I want a refund for ORD-1001" --json`, then `python -m copilot trace list` and `python -m copilot trace show <id>` (traces go to `./traces`, git-ignored).
3. Run the service: `python -m copilot serve --port 8080`; in another shell `curl -s localhost:8080/healthz` and `curl -s -XPOST localhost:8080/v1/ask -d "{\"question\":\"How long do password reset links last?\"}"`. Set `COPILOT_API_TOKEN` to require a bearer token.
4. Golden set: `python -m copilot_eval hygiene` (exit 0 clean).
5. Evals: `python -m copilot_eval run --variant baseline --out base.json`; all variants and the committed tables: `python -m copilot_eval report` (about 15 s, rewrites `eval_results/`).
6. Gate: `python -m copilot_eval gate --baseline baseline --candidate raw_over_refund --out-dir out; echo $?` (expect 1); `--candidate baseline` (expect 0); `--candidate nope` (expect 2). `--split heldout` gates on the held-out questions only. `python scripts/ci_gate_check.py` asserts all of these.
7. Load test: `python -m copilot_eval load --clients 8 --requests 400`; add `--rate-capacity 30 --rate-refill 10` to see 429s. Expect different numbers on your machine.
8. Plug in a real model (never run live by me; expect to debug): set `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` (a model id from the Anthropic docs), then `COPILOT_MODEL=claude python -m copilot ask "..."` or `COPILOT_MODEL=claude python -m copilot serve`. Start with one question and read the trace.
9. Evaluate that model: `python -m copilot_eval run --variant baseline --model claude --out claude.json` (cost: about 142 questions times up to 6 model calls), then `python -m copilot_eval gate --baseline base.json --candidate claude.json`. Real models are not deterministic: run twice and compare before trusting a difference. The fault-knob variants do not apply to a real model.
10. Judge: wrap the model in `copilot_eval.judge.LLMJudge`, run `calibrate(judge, load_labels())` and only use it for gating if kappa is at least 0.6. Extend `judge_labels.json` with real answers first (30 synthetic labels are a smoke test, not a calibration).
