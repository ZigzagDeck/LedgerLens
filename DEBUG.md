# LedgerLens Debugging & Transaction Tracing Guide

LedgerLens can expose structured per-record reconciliation traces through the REST API. Debug output is intended to show what evidence was used, whether AI was involved, and which safety checks affected the final decision.

---

## Enable debug mode

Environment variable:

```bash
LEDGERLENS_DEBUG_MODE=true
```

Or request-level query parameter:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/reconcile?debug=true"
```

The query parameter takes precedence over the environment/default configuration for that request.

---

## Trace shape

A debug response contains a `traces` array. Each trace can include:

```json
{
  "ledger_id": "ORD-1043",
  "bank_id": "UTR500143",
  "status": "REVIEW",
  "matching_rule": "AI_REVIEW_REQUIRED",
  "score": 0.72,
  "reason": "AI Review: ...",
  "decision_source": "groq",
  "evidence_breakdown": {
    "ref": 0.55,
    "amount": 0.4,
    "date": 1.0,
    "text": 0.9
  },
  "ai_invoked": true,
  "ai_result_summary": "...",
  "ai_validation_result": "review_suggested",
  "hard_safety_checks": "passed",
  "candidate_id_check": "passed",
  "amount_safety_check": "passed",
  "currency_safety_check": "passed",
  "one_to_one_check": "passed",
  "final_decision": "REVIEW",
  "candidate_count": 3,
  "candidate_rank": 1,
  "amount_difference": 50.0,
  "date_difference": 0
}
```

The exact evidence values depend on the transaction and configuration.

---

## Common matching rules

- `EXACT_REFERENCE` — exact order reference plus deterministic safety checks.
- `EXACT_AMOUNT_DATE` — unique exact amount/date/currency candidate with no contradictory reference.
- `SCORE_MATCHED` — evidence score meets the high-confidence threshold.
- `AI_CONFIRMED_MATCH` — bounded AI selected a supplied candidate and passed deterministic vetoes.
- `AI_REVIEW_REQUIRED` — AI was inconclusive, unavailable, malformed, or safely rejected.
- `AMBIGUOUS_CANDIDATES` — competing candidates are too close for deterministic auto-resolution.
- `SCORE_REVIEW` — evidence is within the review band.
- `ONE_TO_ONE_CONFLICT` — a bank transaction was already claimed by a stronger match.
- `CURRENCY_MISMATCH` — currency contradiction; routed to review.
- `NO_CANDIDATE` — no acceptable bank candidate exists inside the broad generation window.
- `LOW_SCORE` — candidates existed but top evidence remained below the review threshold.
- `NO_MATCH` — bank record remained unlinked after ledger reconciliation.

---

## Retrieve a recent run

`POST /api/v1/reconcile` returns a `run_id`. Retrieve it with:

```bash
curl "http://127.0.0.1:8000/api/v1/reconcile/RUN-XXXXXXXX"
```

Retrieve the most recent cached run with:

```bash
curl "http://127.0.0.1:8000/api/v1/reconcile/"
```

Recent REST run summaries are persisted in `data/.ledgerlens_cache.json` (bounded to the most recent runs).

---

## Agent audit trail

Run the agent:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/agent/run"
```

Then retrieve chronological audit events:

```bash
curl "http://127.0.0.1:8000/api/v1/audit"
```

Agent events include early lifecycle transitions (`NEW → INGESTING → NORMALIZING → RECONCILING`), matching outcomes, investigation/policy events, action execution, verification, final state transitions, and idempotency hits.

---

## Custom-data debugging

Upload both datasets first, then request custom reconciliation:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/reconcile?use_custom_data=true&debug=true"
```

If either half of the custom ledger/bank pair is missing, the API returns HTTP 400. It does not silently fall back to the benchmark dataset.
