# LedgerLens Final Engineering Audit

## Verification status

The current implementation is verified in GitHub Actions on Python 3.11 with:

- source compilation via `python -m compileall -q src app api scripts tests`
- **73 passing pytest tests**
- offline benchmark execution
- answer-key isolation audit

The CI benchmark intentionally runs without a Groq key so external model output cannot make the build nondeterministic. Ambiguous AI-gated records therefore degrade safely to `REVIEW`.

---

## Corrected benchmark accounting

The evaluator treats only `MATCHED` as a positive pair prediction. `REVIEW` and `UNMATCHED` are non-match predictions for pair metrics.

This means a true pair routed to human review is a **false negative for automated pair recall**. Earlier audit numbers that omitted REVIEW rows from TP/FP/FN/TN were optimistic and are superseded by the corrected results below.

### Checked-in canonical dataset

- Ledger rows: **225**
- Bank rows: **250**
- Result rows: **259**

### Reproducible no-key CI baseline

| Metric | Result |
|---|---:|
| Pair Precision | 88.24% |
| Pair Recall | 75.00% |
| F1 | 81.08% |
| Auto-Resolution Precision | 88.24% |
| Auto-Resolution Recall | 75.00% |
| Automated MATCHED Coverage | 60.44% |
| Review Rate | 36.00% |
| AI-Gate / Escalation Rate | 36.00% |
| False Positive Rate | 21.33% |
| False Negative Rate | 25.00% |

Confusion matrix:

- TP: 120
- FP: 16
- FN: 40
- TN: 59

Headline: **88.2% precision at 60.4% automated coverage**.

Live Groq runs are expected to differ and must be measured in the environment where they are executed rather than copied from a previous model/API run.

---

## Dataset difficulty checks

Current canonical dataset audit:

- exact reference rate: **68%**
- partial/noisy reference rate: **8%**
- no-reference rate: **24%**
- unique amount+date matchable: **49.33%**
- multiple amount+date candidates: **22.22%**
- no exact amount+date candidate: **28.44%**

The benchmark contains duplicate, ambiguous, fee, date-shift, false-positive-trap, unmatched, reversal, and fee-only scenarios.

---

## Safety and integrity gates

### Ground-truth isolation

Matching code does not load `answer_key.csv`. The answer key is restricted to evaluation/audit paths.

### AI boundaries

- Only deterministically generated candidate IDs can be selected.
- A hallucinated candidate ID is vetoed to `REVIEW`.
- Missing/malformed responses safely degrade to `REVIEW`.
- Untrusted ledger and bank text is sanitized before it enters the LLM prompt.
- AI cache identity fingerprints transaction/candidate content and relevant configuration, preventing stale decisions when IDs are reused with changed data.
- Rate limiting uses a 60-second sliding window.
- HTTP 429 retries use 2s, 4s, and 8s exponential backoff.

### Financial invariants

- Cross-currency contradictions are routed to `REVIEW`.
- One bank match cannot safely resolve multiple ledger records.
- Selected AI candidate evidence is preserved for the actual selected candidate rather than the top-ranked candidate.
- Fee auto-adjustment policy is bounded at ₹100 by default.

### Agent/audit invariants

Cases traverse audited lifecycle states beginning at `NEW`, then `INGESTING`, `NORMALIZING`, and `RECONCILING` before the matching outcome state.

Action execution and final verification transitions are logged through the same append-only audit mechanism. Repeated actions are protected by `case_id:action_type` idempotency keys and produce an idempotency audit event.

---

## API/UI consistency gates

- Streamlit reconciliation, agent processing, and benchmark evaluation reuse the same configuration and result DataFrame.
- A session-entered Groq key is stored in Streamlit session state rather than copied into global process environment state.
- Custom-data API requests never silently fall back to benchmark data when the custom pair is incomplete.
- Debug traces expose evidence breakdown and candidate/safety metadata.
- Upload validation checks all fields directly required by the runtime engine.
- `python-multipart` is installed for FastAPI file uploads.

---

## Remaining intentional limitations

- Live Razorpay settlement fetching is not implemented; the checked-in demo/canonical adapter is supported and tested.
- Batch aggregate detection is heuristic, not exhaustive arbitrary subset-sum optimization.
- Live FX conversion is not implemented; cross-currency records are routed to review.
- Live LLM performance is external and nondeterministic; CI uses the deterministic no-key baseline.
