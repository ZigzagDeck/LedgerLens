"""Evaluation module for financial reconciliation engine against canonical ground truth."""

import os
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    from src.reconciliation import reconcile
    from src.config import ReconciliationConfig, CONFIG
except ModuleNotFoundError:
    from reconciliation import reconcile
    from config import ReconciliationConfig, CONFIG


def evaluate_reconciliation(
    data_dir: str = "data",
    config: Optional[ReconciliationConfig] = None,
    precomputed_results: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Evaluate final engine decisions against answer_key.csv.

    MATCHED is the positive prediction. REVIEW and UNMATCHED are both non-match
    predictions for pair-level precision/recall, so a true match sent to REVIEW is
    correctly counted as a false negative instead of disappearing from the matrix.
    """
    ledger_path = os.path.join(data_dir, "ledger.csv")
    bank_path = os.path.join(data_dir, "bank_statement.csv")
    answer_path = os.path.join(data_dir, "answer_key.csv")

    if not os.path.exists(answer_path):
        raise FileNotFoundError(f"Answer key missing in '{data_dir}/'. Cannot evaluate without ground truth.")

    df_answer = pd.read_csv(answer_path)
    if precomputed_results is not None:
        df_results = precomputed_results.copy()
    else:
        if not (os.path.exists(ledger_path) and os.path.exists(bank_path)):
            raise FileNotFoundError(f"Dataset CSVs missing in '{data_dir}/'. Run scripts/generate_dataset.py first.")
        df_ledger = pd.read_csv(ledger_path)
        df_bank = pd.read_csv(bank_path)
        df_results = reconcile(df_ledger, df_bank, config=config or CONFIG)

    ledger_results = df_results[df_results["ledger_id"].astype(str) != ""].copy()
    merged = pd.merge(
        df_answer,
        ledger_results,
        left_on="order_id",
        right_on="ledger_id",
        how="outer",
    )

    tp = fp = fn = tn = 0
    auto_tp = auto_fp = 0
    ai_tp = ai_fp = 0
    review_correct = review_total = 0
    invalid_ai_selection_count = 0
    ai_assisted_examples: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        exp_status = str(row.get("expected_status", "UNMATCHED")).strip().upper()
        act_status = str(row.get("status", "UNMATCHED")).strip().upper()
        decision_src = str(row.get("decision_source", "deterministic")).strip().lower()

        exp_utr = str(row.get("utr_reference", "")).strip() if pd.notna(row.get("utr_reference")) else ""
        act_utr = str(row.get("bank_id", "")).strip() if pd.notna(row.get("bank_id")) else ""

        expected_match = exp_status == "MATCHED"
        predicted_match = act_status == "MATCHED"
        correct_pair = bool(exp_utr and act_utr and exp_utr == act_utr)

        if predicted_match:
            if expected_match and correct_pair:
                tp += 1
                if decision_src == "groq":
                    ai_tp += 1
                else:
                    auto_tp += 1
            else:
                fp += 1
                if decision_src == "groq":
                    ai_fp += 1
                else:
                    auto_fp += 1
        else:
            if expected_match:
                fn += 1
            else:
                tn += 1

        if act_status == "REVIEW":
            review_total += 1
            if not expected_match:
                review_correct += 1

        if decision_src == "groq":
            if predicted_match and not (expected_match and correct_pair):
                invalid_ai_selection_count += 1
            ai_assisted_examples.append({
                "order_id": row.get("order_id", ""),
                "bank_id": row.get("bank_id", ""),
                "scenario": row.get("scenario", ""),
                "ai_reason": row.get("ai_reason", row.get("reason", "")),
                "model": row.get("model_used", ""),
                "final_decision": act_status,
                "orig_score": row.get("original_score", 0.0),
            })

    if os.path.exists(ledger_path):
        total_ledger = len(pd.read_csv(ledger_path))
    else:
        total_ledger = ledger_results["ledger_id"].nunique()
    total_bank = len(pd.read_csv(bank_path)) if os.path.exists(bank_path) else 0
    total_results = len(df_results)

    ai_calls_count = len(df_results[df_results["decision_source"] == "groq"])
    ai_matched_count = len(df_results[(df_results["decision_source"] == "groq") & (df_results["status"] == "MATCHED")])
    deterministic_matched = len(df_results[(df_results["decision_source"] == "deterministic") & (df_results["status"] == "MATCHED")])

    matches_count = len(df_results[df_results["status"] == "MATCHED"])
    reviews_count = len(df_results[df_results["status"] == "REVIEW"])
    unmatched_ledger_count = len(df_results[(df_results["ledger_id"] != "") & (df_results["status"] == "UNMATCHED")])
    unmatched_bank_count = len(df_results[(df_results["ledger_id"] == "") & (df_results["status"] == "UNMATCHED")])
    duplicate_assignments = len(df_results[df_results["matching_rule"] == "ONE_TO_ONE_CONFLICT"])

    pair_precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    pair_recall = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f1_score = round(2 * pair_precision * pair_recall / (pair_precision + pair_recall), 4) if (pair_precision + pair_recall) else 0.0

    auto_precision = round(auto_tp / (auto_tp + auto_fp), 4) if (auto_tp + auto_fp) else 0.0
    auto_recall = round(auto_tp / (tp + fn), 4) if (tp + fn) else 0.0
    ai_precision = round(ai_tp / (ai_tp + ai_fp), 4) if (ai_tp + ai_fp) else 0.0
    review_precision = round(review_correct / review_total, 4) if review_total else 0.0
    exception_recall = round(tn / (tn + fp), 4) if (tn + fp) else 0.0
    fpr = round(fp / (fp + tn), 4) if (fp + tn) else 0.0
    fnr = round(fn / (fn + tp), 4) if (fn + tp) else 0.0

    automated_coverage = round(matches_count / total_ledger, 4) if total_ledger else 0.0
    review_rate = round(reviews_count / total_ledger, 4) if total_ledger else 0.0
    unmatched_rate = round(unmatched_ledger_count / total_ledger, 4) if total_ledger else 0.0
    deterministic_match_rate = round(deterministic_matched / total_ledger, 4) if total_ledger else 0.0
    ai_escalation_rate = round(ai_calls_count / total_ledger, 4) if total_ledger else 0.0
    ai_match_rate = round(ai_matched_count / ai_calls_count, 4) if ai_calls_count else 0.0

    precision_at_coverage = f"{pair_precision*100:.1f}% precision at {automated_coverage*100:.1f}% automated coverage"

    metrics = {
        "denominators": {
            "total_ledger_records": total_ledger,
            "total_bank_records": total_bank,
            "total_results_rows": total_results,
        },
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "f1_score": f1_score,
        "auto_resolution_precision": auto_precision,
        "auto_resolution_recall": auto_recall,
        "ai_match_precision": ai_precision,
        "review_precision": review_precision,
        "exception_recall": exception_recall,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "headline": precision_at_coverage,
        "status_counts": {
            "MATCHED": matches_count,
            "REVIEW": reviews_count,
            "UNMATCHED_LEDGER": unmatched_ledger_count,
            "UNMATCHED_BANK": unmatched_bank_count,
        },
        "rates": {
            "automated_coverage": automated_coverage,
            "review_rate": review_rate,
            "unmatched_rate": unmatched_rate,
            "deterministic_match_rate": deterministic_match_rate,
            "ai_escalation_rate": ai_escalation_rate,
            "ai_match_rate": ai_match_rate,
        },
        "safety_checks": {
            "duplicate_assignment_conflicts": duplicate_assignments,
            "invalid_ai_selections": invalid_ai_selection_count,
        },
        "ai_examples": ai_assisted_examples[:3],
    }

    print("=" * 65)
    print("FINANCIAL RECONCILIATION BENCHMARK EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Ledger Records : {total_ledger}")
    print(f"Total Bank Records   : {total_bank}")
    print("-" * 65)
    print(f">>> {precision_at_coverage} <<<")
    print("-" * 65)
    print(f"Pair Precision           : {pair_precision:.4f}")
    print(f"Pair Recall              : {pair_recall:.4f}")
    print(f"F1 Score                 : {f1_score:.4f}")
    print(f"Auto-Resolution Precision: {auto_precision:.4f}")
    print(f"Auto-Resolution Recall   : {auto_recall:.4f}")
    print(f"Review Precision         : {review_precision:.4f}")
    print(f"Exception Recall         : {exception_recall:.4f}")
    print(f"False Positive Rate      : {fpr:.4f}")
    print(f"False Negative Rate      : {fnr:.4f}")
    print("=" * 65)

    results_excel_path = os.path.join(data_dir, "reconciliation_results.xlsx")
    try:
        with pd.ExcelWriter(results_excel_path, engine="openpyxl") as writer:
            df_results.to_excel(writer, sheet_name="Reconciliation Results", index=False)
            pd.DataFrame([metrics["rates"]]).to_excel(writer, sheet_name="Evaluation Metrics", index=False)
    except Exception as e:
        print(f"Note: Could not update Excel results ({e}). Results calculated successfully.")

    return metrics


if __name__ == "__main__":
    evaluate_reconciliation()
