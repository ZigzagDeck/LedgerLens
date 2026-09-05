"""Deterministic & Evidence-based Financial Reconciliation Engine."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

try:
    from src.config import ReconciliationConfig, CONFIG
    from src.normalization import normalize_amount, normalize_date, normalize_text, extract_reference, fuzz
    from src.data_validation import validate_ledger_schema, validate_bank_schema
    from src.schemas import ReconciliationRecord, EvidenceBreakdown
except ModuleNotFoundError:
    from config import ReconciliationConfig, CONFIG
    from normalization import normalize_amount, normalize_date, normalize_text, extract_reference, fuzz
    from data_validation import validate_ledger_schema, validate_bank_schema
    from schemas import ReconciliationRecord, EvidenceBreakdown

try:
    from src.ai_matcher import evaluate_ambiguous_record
except ModuleNotFoundError:
    try:
        from ai_matcher import evaluate_ambiguous_record
    except ModuleNotFoundError:
        evaluate_ambiguous_record = None


def compute_evidence_score(
    l_row: pd.Series,
    b_row: pd.Series,
    config: ReconciliationConfig = CONFIG,
) -> Tuple[float, Dict[str, float]]:
    """Compute weighted multi-factor evidence score between ledger and bank row."""
    l_id, l_amt, l_date = l_row["order_id"], l_row["norm_amount"], l_row["norm_date"]
    l_cust = l_row.get("customer_name", "")
    b_narration, b_ref = b_row["norm_narration"], b_row["extracted_ref"]
    b_amt, b_date = b_row["norm_amount"], b_row["norm_date"]

    if b_ref == l_id or (l_id and l_id in b_narration):
        ref_score = 1.0
    elif l_id:
        fuzzy_ref_ratio = fuzz.token_set_ratio(l_id, b_narration) / 100.0
        ref_score = max(0.0, (fuzzy_ref_ratio - 0.5) * 2.0)
    else:
        ref_score = 0.0

    amt_diff, fee_amt = abs(l_amt - b_amt), l_amt - b_amt
    if amt_diff <= config.AMOUNT_TOLERANCE:
        amount_score = 1.0
    elif 0 < fee_amt <= config.MAX_FEE_AMOUNT:
        amount_score = 0.40
    elif l_amt and amt_diff <= (l_amt * config.BROAD_AMOUNT_TOLERANCE_PCT):
        amount_score = max(0.20, 0.80 - (amt_diff / (l_amt * config.BROAD_AMOUNT_TOLERANCE_PCT)))
    else:
        amount_score = 0.0

    date_diff = abs((l_date - b_date).days)
    if date_diff == 0:
        date_score = 1.0
    elif date_diff <= config.DATE_WINDOW_DAYS:
        date_score = 1.0 - (date_diff * 0.05)
    elif date_diff <= config.BROAD_DATE_WINDOW_DAYS:
        date_score = max(0.30, 0.85 - ((date_diff - config.DATE_WINDOW_DAYS) * 0.08))
    else:
        date_score = 0.0

    text_score = (
        fuzz.partial_ratio(normalize_text(l_cust), b_narration) / 100.0
        if l_cust and b_narration else 0.5
    )
    total_score = round(
        config.W_REF * ref_score +
        config.W_AMOUNT * amount_score +
        config.W_DATE * date_score +
        config.W_TEXT * text_score,
        4,
    )
    breakdown = EvidenceBreakdown(
        ref=round(ref_score, 2), amount=round(amount_score, 2),
        date=round(date_score, 2), text=round(text_score, 2),
    )
    return total_score, breakdown.to_dict()


def reconcile(
    df_ledger: pd.DataFrame,
    df_bank: pd.DataFrame,
    config: ReconciliationConfig = CONFIG,
) -> pd.DataFrame:
    """Perform deterministic-first, bounded-AI reconciliation with one-to-one assignment."""
    valid_l, errs_l = validate_ledger_schema(df_ledger, config)
    valid_b, errs_b = validate_bank_schema(df_bank, config)
    if not (valid_l and valid_b):
        raise ValueError(f"Schema validation failed: {errs_l + errs_b}")

    ledger, bank = df_ledger.copy(), df_bank.copy()
    ledger["norm_amount"] = ledger["amount"].apply(normalize_amount)
    ledger["norm_date"] = ledger["order_date"].apply(normalize_date)
    ledger["norm_curr"] = ledger["currency"].apply(normalize_text)
    bank["norm_amount"] = bank["credited_amount"].apply(normalize_amount)
    bank["norm_date"] = bank["value_date"].apply(normalize_date)
    bank["norm_curr"] = bank["currency"].apply(normalize_text)
    bank["norm_narration"] = bank["narration_text"].apply(normalize_text)
    bank["extracted_ref"] = bank["norm_narration"].apply(extract_reference)

    matched_results: List[Dict[str, Any]] = []
    finalized_ledger, assigned_bank = set(), set()

    def add_result(
        l_id: str, b_id: str, status: str, rule: str, score: float, reason: str,
        source: str = "deterministic", model: str = "none", ai_reason: str = "",
        orig_score: Optional[float] = None, amt_diff: float = 0.0, date_diff: int = 0,
        rank: int = 1, count: int = 1, evidence: Optional[Dict[str, float]] = None,
    ) -> None:
        rec = ReconciliationRecord(
            ledger_id=l_id, bank_id=b_id, status=status, matching_rule=rule,
            score=score, reason=reason, decision_source=source, model_used=model,
            ai_reason=ai_reason, original_score=orig_score if orig_score is not None else score,
            amount_difference=amt_diff, date_difference=date_diff,
            candidate_rank=rank, candidate_count=count,
            evidence_breakdown=evidence or {},
        )
        matched_results.append(rec.to_dict())
        if l_id and status in ("MATCHED", "REVIEW", "UNMATCHED"):
            finalized_ledger.add(l_id)
        if b_id and status == "MATCHED":
            assigned_bank.add(b_id)

    # Tier 1: exact reference + amount/date safety checks.
    for _, l_row in ledger.iterrows():
        l_id, l_amt, l_date, l_curr = (
            l_row["order_id"], l_row["norm_amount"], l_row["norm_date"], l_row["norm_curr"]
        )
        candidates = bank[
            (~bank["utr_reference"].isin(assigned_bank)) &
            ((bank["extracted_ref"] == l_id) | bank["norm_narration"].str.contains(l_id, regex=False))
        ]
        if candidates.empty:
            continue
        b_row = candidates.iloc[0]
        b_id = b_row["utr_reference"]
        amt_diff = abs(l_amt - b_row["norm_amount"])
        date_diff = abs((l_date - b_row["norm_date"]).days)
        _, breakdown = compute_evidence_score(l_row, b_row, config)
        if l_curr != b_row["norm_curr"]:
            add_result(
                l_id, b_id, "REVIEW", "CURRENCY_MISMATCH", 0.0,
                f"Currency mismatch ({l_curr} vs {b_row['norm_curr']})",
                amt_diff=amt_diff, date_diff=date_diff, evidence=breakdown,
            )
        elif amt_diff <= config.AMOUNT_TOLERANCE and date_diff <= config.DATE_WINDOW_DAYS:
            add_result(
                l_id, b_id, "MATCHED", "EXACT_REFERENCE", 1.0,
                "Exact reference, amount, and date match",
                amt_diff=amt_diff, date_diff=date_diff, evidence=breakdown,
            )

    # Tier 2: exact unique amount/date match without contradictory reference.
    for _, l_row in ledger[~ledger["order_id"].isin(finalized_ledger)].iterrows():
        l_id, l_amt, l_date, l_curr = (
            l_row["order_id"], l_row["norm_amount"], l_row["norm_date"], l_row["norm_curr"]
        )
        candidates = bank[
            (~bank["utr_reference"].isin(assigned_bank)) &
            (bank["norm_amount"] == l_amt) &
            (bank["norm_date"] == l_date) &
            (bank["norm_curr"] == l_curr)
        ]
        if len(candidates) == 1:
            b_row = candidates.iloc[0]
            b_ref = b_row["extracted_ref"]
            if b_ref and b_ref != l_id:
                continue
            _, breakdown = compute_evidence_score(l_row, b_row, config)
            add_result(
                l_id, b_row["utr_reference"], "MATCHED", "EXACT_AMOUNT_DATE", 0.9,
                "Exact amount and date match", amt_diff=0.0, date_diff=0, evidence=breakdown,
            )

    # Tier 3: bounded candidate generation, evidence scoring, and optional AI.
    for _, l_row in ledger[~ledger["order_id"].isin(finalized_ledger)].iterrows():
        l_id, l_curr, l_date, l_amt = (
            l_row["order_id"], l_row["norm_curr"], l_row["norm_date"], l_row["norm_amount"]
        )
        candidate_pool = []
        for _, b_row in bank[~bank["utr_reference"].isin(assigned_bank)].iterrows():
            if l_curr != b_row["norm_curr"]:
                continue
            date_diff = abs((l_date - b_row["norm_date"]).days)
            if date_diff > config.BROAD_DATE_WINDOW_DAYS:
                continue
            amt_diff = abs(l_amt - b_row["norm_amount"])
            fee_amt = l_amt - b_row["norm_amount"]
            max_amt_tol = max(config.AMOUNT_TOLERANCE, l_amt * config.BROAD_AMOUNT_TOLERANCE_PCT)
            if amt_diff > max_amt_tol and not (0 < fee_amt <= config.MAX_FEE_AMOUNT):
                continue
            score, breakdown = compute_evidence_score(l_row, b_row, config)
            candidate_pool.append(
                (score, b_row["utr_reference"], b_row, breakdown, amt_diff, date_diff)
            )

        candidate_pool.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidate_pool[:config.TOP_N_CANDIDATES]
        if not top_candidates:
            add_result(
                l_id, "", "UNMATCHED", "NO_CANDIDATE", 0.0,
                "No candidate within broad window and tolerance", count=0,
            )
            continue

        top_score, top_b_id, _, top_breakdown, top_amt_diff, top_date_diff = top_candidates[0]
        cand_count = len(top_candidates)
        is_ambiguous = cand_count > 1 and (
            top_score - top_candidates[1][0]
        ) < config.AMBIGUITY_MARGIN

        if is_ambiguous or (config.REVIEW_THRESHOLD <= top_score < config.HIGH_CONFIDENCE_THRESHOLD):
            if config.ENABLE_AI_ASSIST and evaluate_ambiguous_record is not None:
                ai_candidates = top_candidates[:config.AI_CANDIDATE_LIMIT]
                ai_input = [(c[0], c[1], c[2], c[3]) for c in ai_candidates]
                ai_eval = evaluate_ambiguous_record(l_row, ai_input, config)
                ai_status = ai_eval.get("status", "REVIEW")
                ai_reason = ai_eval.get("reason", "")
                ai_model = ai_eval.get("model_used", "none")
                selected_id = ai_eval.get("selected_bank_id") or top_b_id
                ai_candidate_ids = [c[1] for c in ai_candidates]
                if selected_id not in ai_candidate_ids:
                    ai_status = "REVIEW"
                    ai_reason = f"VETO: AI bank ID '{selected_id}' not in candidate pool."
                    selected_id = top_b_id

                selected = next((c for c in ai_candidates if c[1] == selected_id), ai_candidates[0])
                sel_score, _, _, sel_breakdown, sel_amt_diff, sel_date_diff = selected
                sel_rank = ai_candidate_ids.index(selected_id) + 1

                if ai_status == "MATCHED":
                    add_result(
                        l_id, selected_id, "MATCHED", "AI_CONFIRMED_MATCH", sel_score,
                        f"AI Confirmed: {ai_reason}", source="groq", model=ai_model,
                        ai_reason=ai_reason, orig_score=sel_score, amt_diff=sel_amt_diff,
                        date_diff=sel_date_diff, rank=sel_rank, count=cand_count,
                        evidence=sel_breakdown,
                    )
                else:
                    add_result(
                        l_id, top_b_id, "REVIEW", "AI_REVIEW_REQUIRED", top_score,
                        f"AI Review: {ai_reason}", source="groq", model=ai_model,
                        ai_reason=ai_reason, orig_score=top_score, amt_diff=top_amt_diff,
                        date_diff=top_date_diff, count=cand_count, evidence=top_breakdown,
                    )
            else:
                rule = "AMBIGUOUS_CANDIDATES" if is_ambiguous else "SCORE_REVIEW"
                add_result(
                    l_id, top_b_id, "REVIEW", rule, top_score,
                    f"Deterministic review required ({top_score:.2f})",
                    amt_diff=top_amt_diff, date_diff=top_date_diff, count=cand_count,
                    evidence=top_breakdown,
                )
        elif top_score >= config.HIGH_CONFIDENCE_THRESHOLD:
            add_result(
                l_id, top_b_id, "MATCHED", "SCORE_MATCHED", top_score,
                f"High confidence match ({top_score:.2f})",
                amt_diff=top_amt_diff, date_diff=top_date_diff, count=cand_count,
                evidence=top_breakdown,
            )
        else:
            add_result(
                l_id, "", "UNMATCHED", "LOW_SCORE", top_score,
                f"Top score ({top_score:.2f}) below review threshold",
                amt_diff=top_amt_diff, date_diff=top_date_diff, count=cand_count,
                evidence=top_breakdown,
            )

    # Defensive global one-to-one conflict resolution.
    confirmed_bank_map: Dict[str, Tuple[int, float]] = {}
    conflicted_indices = set()
    for idx, res in enumerate(matched_results):
        b_id, status, score = res["bank_id"], res["status"], res["score"]
        if status != "MATCHED" or not b_id:
            continue
        if b_id in confirmed_bank_map:
            prev_idx, prev_score = confirmed_bank_map[b_id]
            if score > prev_score:
                conflicted_indices.add(prev_idx)
                confirmed_bank_map[b_id] = (idx, score)
            else:
                conflicted_indices.add(idx)
        else:
            confirmed_bank_map[b_id] = (idx, score)

    for idx in conflicted_indices:
        res = matched_results[idx]
        old_bank_id = res["bank_id"]
        res["status"] = "REVIEW"
        res["matching_rule"] = "ONE_TO_ONE_CONFLICT"
        res["reason"] = f"One-to-one conflict: Bank ID '{old_bank_id}' claimed by higher confidence match."
        res["bank_id"] = ""

    claimed_bank_ids = {
        r["bank_id"] for r in matched_results if r["bank_id"] and r["status"] == "MATCHED"
    }
    reviewed_bank_ids = {
        r["bank_id"] for r in matched_results if r["bank_id"] and r["status"] == "REVIEW"
    }
    for b_id in bank[~bank["utr_reference"].isin(claimed_bank_ids | reviewed_bank_ids)]["utr_reference"]:
        add_result("", b_id, "UNMATCHED", "NO_MATCH", 0.0, "No matching ledger record found", count=0)

    return pd.DataFrame(matched_results)


if __name__ == "__main__":
    import os
    ledger_path = os.path.join("data", "ledger.csv")
    bank_path = os.path.join("data", "bank_statement.csv")
    if os.path.exists(ledger_path) and os.path.exists(bank_path):
        results = reconcile(pd.read_csv(ledger_path), pd.read_csv(bank_path))
        print(results["status"].value_counts().to_string())
