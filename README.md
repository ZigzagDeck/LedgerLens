# LedgerLens: AI-Powered Financial Reconciliation Engine

LedgerLens reconciles internal sales/order ledgers against bank statements using a **deterministic-first, bounded-AI** architecture. Exact and high-confidence cases are handled locally; genuinely ambiguous cases may be escalated to Groq with a restricted candidate set, prompt-injection sanitization, deterministic vetoes, rate limiting, safe fallback, policy checks, action verification, and an append-only audit trail.

The repository includes a Streamlit dashboard, FastAPI API, synthetic benchmark generator, finance-controller layer, Razorpay demo adapter, agent workflow, observability traces, and automated tests.

---

## The problem

Financial reconciliation is difficult because the same transaction often looks different across internal systems and bank statements:

- **Gateway/MDR deductions:** an internal ₹5,000 order may arrive as a ₹4,950 bank credit.
- **Settlement delays:** a Monday order may settle several days later.
- **Narration noise:** bank text may add prefixes, codes, separators, or truncated references.
- **Duplicate amounts:** several legitimate orders can share the same amount and date.
- **Missing records:** a ledger transaction or bank credit may have no counterpart.
- **Safety requirements:** a reconciliation engine must never silently double-assign one bank credit or trust an LLM-generated identifier outside the candidate set.

---

## Architecture

```text
Ledger + Bank Statement
        |
        v
Schema Validation
        |
        v
Normalization
(amount/date/text/reference)
        |
        v
Tier 1: Exact Reference + Safety Checks
        |
        v
Tier 2: Unique Exact Amount + Date
        |
        v
Tier 3: Multi-Evidence Scoring
(ref 40%, amount 30%, date 20%, text 10%)
        |
        +--> score >= 0.82 ------------------> MATCHED
        |
        +--> score < 0.45 -------------------> UNMATCHED
        |
        +--> ambiguous / 0.45..0.82
                    |
                    v
           Bounded Groq Assistant
           - top 3 candidates only
           - untrusted text sanitized
           - candidate-ID veto
           - response schema validation
           - rate limiting + retry
           - safe REVIEW fallback
                    |
                    v
           Policy + Action + Verification
                    |
                    v
              Append-only Audit
```

The percentages of records handled by each tier depend on the dataset and thresholds. The checked-in canonical dataset currently produces **60.4% automated MATCHED coverage** in the no-key CI baseline, while **36.0% of ledger records reach the AI gate and safely become REVIEW when no Groq key is configured**.

---

## Matching rules

### Tier 1 — exact reference

A bank row is matched immediately only when the reference is present and deterministic safety conditions are satisfied. Currency contradictions are never matched; they are routed to `REVIEW` with `CURRENCY_MISMATCH`.

### Tier 2 — unique amount + date

A unique amount/date/currency candidate can be matched when there is no contradictory extracted order reference.

### Tier 3 — evidence scoring

Candidates inside the broad amount/date window are scored using:

| Evidence | Weight |
|---|---:|
| Reference similarity | 40% |
| Amount consistency | 30% |
| Date proximity | 20% |
| Customer/narration text | 10% |

- `score >= 0.82` → deterministic `MATCHED`
- `score < 0.45` → `UNMATCHED`
- intermediate or close competing candidates → bounded AI or `REVIEW`

A bank transaction can be assigned to at most one matched ledger record. Defensive one-to-one conflict resolution downgrades conflicting claims to `REVIEW`.

---

## Bounded AI and safety controls

LedgerLens does not send every transaction to an LLM.

For the checked-in 225-ledger-row benchmark with current thresholds, **81 rows (36.0%) reach the AI gate**. With no API key those rows safely become `REVIEW`. With a Groq key, the built-in rate limiter processes them across as many rate-limit windows as required.

Safety controls include:

1. **Top-3 candidate boundary** — the model can choose only from candidates supplied by deterministic generation.
2. **Candidate-ID hallucination veto** — an unknown `selected_bank_id` is rejected and routed to `REVIEW`.
3. **Structured response validation** — Groq JSON is validated with Pydantic before use.
4. **Prompt-injection sanitization** — ledger/customer/narration text is treated as untrusted data; URLs, control characters, jailbreak/override phrases, flooding, and oversized input are sanitized before entering the prompt.
5. **Content-aware AI cache** — cache identity fingerprints the full sanitized ledger/candidate payload and relevant configuration, preventing stale decisions when transaction content changes under the same IDs.
6. **Sliding-window rate limiter** — default `25` calls/minute via `GROQ_MAX_CALLS_PER_MINUTE`.
7. **429 exponential backoff** — retries use the documented `2s`, `4s`, `8s` sequence.
8. **Safe degradation** — missing key, network failure, invalid JSON, or validation failure defaults to `REVIEW` rather than an unsafe match.

---

## Stateful agent workflow

The agent follows explicit audited states rather than jumping directly to a final result:

```text
NEW
 -> INGESTING
 -> NORMALIZING
 -> RECONCILING
 -> MATCHED / AMBIGUOUS / UNMATCHED
 -> INVESTIGATING (when needed)
 -> RECOMMENDATION_READY
 -> POLICY_APPROVED or ACTION_PENDING_APPROVAL
 -> ACTION_EXECUTING
 -> ACTION_VERIFYING
 -> RESOLVED / UNMATCHED / ACTION_PENDING_APPROVAL
```

Every state change, action execution, verification result, and idempotency hit is recorded as an `AuditEvent`.

### Policy rules

- High-confidence deterministic matches can be marked reconciled automatically.
- AI-confirmed matches are accepted only after the bounded candidate checks and deterministic vetoes have passed.
- Fee adjustments up to **₹100** can be auto-approved by policy unless approval is configured as mandatory.
- Currency contradictions and one-to-one conflicts require human review.
- Unsupported or unsafe actions fall back to review behavior.

### Idempotency

Actions use a `case_id:action_type` idempotency key, preventing duplicate financial side effects during repeated execution.

---

## Streamlit dashboard

Run:

```bash
streamlit run app/app.py
```

### Standard mode

- Upload ledger and bank files in CSV or XLSX format.
- Load checked-in sample datasets.
- Preview the active data.
- Adjust amount/date/decision thresholds.
- Enable or disable Groq assistance.
- Enter a Groq key in a password field; the key is kept in **Streamlit session state**, not copied into process-global environment variables.
- Run reconciliation.
- Inspect matched, review, unmatched, exception, and evidence views.
- Inspect agent cases and the full append-only audit registry.
- Export reconciliation results and audit history as CSV.

### Benchmark mode

- Loads `data/ledger.csv` and `data/bank_statement.csv`.
- Evaluates the exact reconciliation result against `data/answer_key.csv`.
- Displays precision, recall, F1, coverage, review rate, AI escalation rate, and a 2×2 confusion matrix.

The agent and evaluator reuse the **same result DataFrame and the same user configuration** as the UI run; they do not secretly rerun reconciliation with default settings.

---

## Benchmark methodology and verified baseline

The checked-in benchmark contains:

- **225 ledger records**
- **250 bank records**
- 12 scenario classes including exact, noisy reference, fee difference, date shift, duplicates, ambiguity, false-positive traps, amount-near-match, unmatched ledger/bank, reversals, and fee-only settlements.

### Correct pair accounting

For pair-level metrics:

- `MATCHED` is the positive prediction.
- `REVIEW` and `UNMATCHED` are non-match predictions.
- A true match sent to `REVIEW` is therefore a **false negative** for auto-resolution metrics; it is not silently excluded from the confusion matrix.

### Reproducible CI baseline — no Groq key

The GitHub Actions benchmark intentionally runs without a Groq key so external model behavior cannot make CI nondeterministic. Ambiguous cases therefore degrade safely to `REVIEW`.

| Metric | Verified result |
|---|---:|
| Pair Precision | **88.24%** |
| Pair Recall | **75.00%** |
| F1 | **81.08%** |
| Auto-Resolution Precision | **88.24%** |
| Auto-Resolution Recall | **75.00%** |
| Automated MATCHED Coverage | **60.44%** |
| Review Rate | **36.00%** |
| AI-Gate / Escalation Rate | **36.00%** |
| False Positive Rate | **21.33%** |
| False Negative Rate | **25.00%** |

Confusion matrix:

```text
TP = 120
FP = 16
FN = 40
TN = 59
```

Headline: **88.2% precision at 60.4% automated coverage**.

These are the reproducible no-key baseline numbers from CI, not marketing estimates. A live Groq run may produce different match/coverage metrics depending on model behavior, API availability, and configured thresholds; rerun `python -m scripts.run_benchmark` in the target environment rather than hard-coding an unverified live-AI score.

---

## REST API

Run:

```bash
uvicorn app.api:app --reload --port 8000
```

### Endpoints

- `GET /api/v1/health` — default/custom dataset readiness and configuration status.
- `POST /api/v1/reconcile` — run reconciliation.
  - `?debug=true` includes per-record evidence breakdown, candidate metadata, AI outcome summary, and deterministic safety checks.
  - `?use_custom_data=true` requires both uploaded custom datasets; an incomplete pair returns `400` and **never silently falls back to benchmark data**.
- `POST /api/v1/custom-data/upload` — upload a validated ledger or bank CSV/XLSX file.
- `GET /api/v1/reconcile/{run_id}` — retrieve persisted recent reconciliation summaries (`latest` is supported by `/api/v1/reconcile/`).
- `POST /api/v1/agent/run` — run the bounded agent lifecycle.
- `GET /api/v1/cases` — list active agent cases.
- `GET /api/v1/cases/{case_id}` — inspect one case.
- `POST /api/v1/cases/{case_id}/approve` — execute an actual pending policy action when one exists; a mere `FLAG_FOR_REVIEW` cannot be “approved” into a financial action.
- `GET /api/v1/audit` — chronological append-only audit events from the active agent run.

Recent REST run summaries are stored in `data/.ledgerlens_cache.json`; this is an API run cache, while Groq decision caching is in-memory.

---

## Input schema

The engine validates every field it accesses directly before reconciliation.

### Ledger runtime requirements

- `order_id`
- `amount`
- `order_date`
- `currency`

Optional descriptive fields such as `customer_name` and `payment_method` improve evidence but are not required for runtime safety.

### Bank runtime requirements

- `utr_reference`
- `credited_amount`
- `value_date`
- `currency`
- `narration_text`

`narration_text` may be empty for rails that provide no useful description, but the column must exist.

---

## Razorpay connector

`src/connectors/razorpay.py` provides a checked-in demo adapter and canonical transaction conversion utilities for demonstrating ledger → Razorpay settlement → bank-credit flows.

The demo integration is implemented and tested. **Live Razorpay settlement fetching is not implemented**; `load_live_settlements()` intentionally raises `NotImplementedError` until real credentials/API wiring are added.

---

## Batch aggregate detection

`detect_batch_aggregates()` flags unmatched bank credits that may represent several nearby ledger orders combined into one gateway settlement. The current implementation is a bounded heuristic intended to surface candidates for review, not a proof of arbitrary subset-sum equivalence.

Full optimization for large arbitrary settlement groups remains a known limitation.

---

## Installation

Prerequisites: Python 3.10–3.12 (3.11 recommended) and Git.

```bash
git clone https://github.com/ZigzagDeck/LedgerLens.git
cd LedgerLens
python -m venv venv
```

Activate the environment:

```bash
# macOS/Linux
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

Optional live AI configuration:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MAX_CALLS_PER_MINUTE=25
LEDGERLENS_DEBUG_MODE=false
LEDGERLENS_CUSTOM_DATA_DIR=data/custom
```

Without a Groq key the system still runs; ambiguous cases safely become `REVIEW`.

---

## Developer commands

```bash
# Full suite — currently 73 tests
python -m pytest -q

# Syntax/compile verification
python -m compileall -q src app api scripts tests

# Reproducible offline benchmark
python -m scripts.run_benchmark

# Generate benchmark data
python -m scripts.generate_dataset --seed 123 --ledger-count 200 --bank-count 200 --output-dir data

# Repository/data audit
python -m scripts.audit_repo

# Throughput benchmark
python -m scripts.run_throughput
```

GitHub Actions runs dependency installation, compile checks, all tests, and the offline benchmark on fix branches and pull requests.

---

## Streamlit Cloud

1. Push or fork the repository to GitHub.
2. Create a Streamlit Cloud app using `app/app.py` as the main file.
3. Use Python 3.11 (the repo includes `.python-version`).
4. Add the optional secret in Streamlit Cloud settings:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

---

## Known limitations

1. **Live Razorpay API fetching:** demo/canonical adapter exists; live settlement API wiring is not implemented.
2. **Arbitrary large batch settlements:** current aggregate detection is heuristic rather than exhaustive subset-sum optimization.
3. **FX conversion:** live foreign-exchange conversion is not implemented; cross-currency contradictions are routed to `REVIEW`.
4. **Live-AI benchmark reproducibility:** external model output can change. CI therefore reports a deterministic no-key baseline and live-AI metrics must be rerun in the target environment.

---

## Verification

Current CI verification on Python 3.11:

- `compileall`: passed
- `pytest`: **73 passed**
- offline benchmark: passed
- answer-key isolation audit: passed

---

## License

MIT License.
