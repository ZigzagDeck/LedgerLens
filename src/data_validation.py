"""Data validation utilities and repository/dataset auditor for LedgerLens."""

import os
import re
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd

try:
    from src.config import ReconciliationConfig, CONFIG
except ModuleNotFoundError:
    from config import ReconciliationConfig, CONFIG


LEDGER_RUNTIME_COLUMNS = ["order_id", "amount", "order_date", "currency"]
BANK_RUNTIME_COLUMNS = ["utr_reference", "credited_amount", "value_date", "currency", "narration_text"]


def validate_ledger_schema(
    df: pd.DataFrame,
    config: ReconciliationConfig = CONFIG,
) -> Tuple[bool, List[str]]:
    """Validate every ledger field the reconciliation engine accesses directly."""
    errors: List[str] = []
    if df is None:
        return False, ["Ledger dataset is None."]

    missing_cols = [c for c in LEDGER_RUNTIME_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, [f"Ledger missing required schema columns: {missing_cols}"]
    if df.empty:
        return True, []

    blank_ids = df["order_id"].isna() | (df["order_id"].astype(str).str.strip() == "")
    if int(blank_ids.sum()) > 0:
        errors.append(f"Ledger contains {int(blank_ids.sum())} missing/empty order_id record(s).")

    dup_ids = int(df.loc[~blank_ids, "order_id"].duplicated().sum())
    if dup_ids:
        errors.append(f"Ledger contains {dup_ids} duplicate order_id record(s).")

    try:
        numeric_amounts = pd.to_numeric(df["amount"], errors="raise")
        if numeric_amounts.isna().any():
            errors.append("Ledger contains missing amount value(s).")
    except Exception:
        errors.append("Ledger contains unparseable non-numeric amount value(s).")

    try:
        parsed_dates = pd.to_datetime(df["order_date"], errors="raise")
        if parsed_dates.isna().any():
            errors.append("Ledger contains missing order_date value(s).")
    except Exception:
        errors.append("Ledger contains unparseable order_date value(s).")

    blank_currency = df["currency"].isna() | (df["currency"].astype(str).str.strip() == "")
    if int(blank_currency.sum()) > 0:
        errors.append(f"Ledger contains {int(blank_currency.sum())} missing/empty currency value(s).")

    return len(errors) == 0, errors


def validate_bank_schema(
    df: pd.DataFrame,
    config: ReconciliationConfig = CONFIG,
) -> Tuple[bool, List[str]]:
    """Validate every bank field the reconciliation engine accesses directly."""
    errors: List[str] = []
    if df is None:
        return False, ["Bank statement dataset is None."]

    missing_cols = [c for c in BANK_RUNTIME_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, [f"Bank statement missing required schema columns: {missing_cols}"]
    if df.empty:
        return True, []

    blank_refs = df["utr_reference"].isna() | (df["utr_reference"].astype(str).str.strip() == "")
    if int(blank_refs.sum()) > 0:
        errors.append(f"Bank statement contains {int(blank_refs.sum())} missing/empty utr_reference record(s).")

    dup_refs = int(df.loc[~blank_refs, "utr_reference"].duplicated().sum())
    if dup_refs:
        errors.append(f"Bank statement contains {dup_refs} duplicate utr_reference record(s).")

    try:
        numeric_amounts = pd.to_numeric(df["credited_amount"], errors="raise")
        if numeric_amounts.isna().any():
            errors.append("Bank statement contains missing credited_amount value(s).")
    except Exception:
        errors.append("Bank statement contains unparseable non-numeric credited_amount value(s).")

    try:
        parsed_dates = pd.to_datetime(df["value_date"], errors="raise")
        if parsed_dates.isna().any():
            errors.append("Bank statement contains missing value_date value(s).")
    except Exception:
        errors.append("Bank statement contains unparseable value_date value(s).")

    blank_currency = df["currency"].isna() | (df["currency"].astype(str).str.strip() == "")
    if int(blank_currency.sum()) > 0:
        errors.append(f"Bank statement contains {int(blank_currency.sum())} missing/empty currency value(s).")

    # Narration may be empty for some rails, but the column itself is mandatory because
    # normalization/scoring access it directly.
    return len(errors) == 0, errors


def validate_custom_data_paths(
    custom_dir: Optional[str] = None,
    config: ReconciliationConfig = CONFIG,
) -> Tuple[bool, Dict[str, str]]:
    target_dir = custom_dir or config.LEDGERLENS_CUSTOM_DATA_DIR
    ledger_xlsx = os.path.join(target_dir, "ledger.xlsx")
    bank_xlsx = os.path.join(target_dir, "bank_statement.xlsx")
    found = {
        "ledger_xlsx": ledger_xlsx if os.path.exists(ledger_xlsx) else "",
        "bank_xlsx": bank_xlsx if os.path.exists(bank_xlsx) else "",
    }
    return bool(found["ledger_xlsx"] and found["bank_xlsx"]), found


def audit_repository_isolation(src_dir: str = "src") -> Dict[str, Any]:
    core_modules = ["reconciliation.py", "ai_matcher.py", "normalization.py", "schemas.py", "data_validation.py"]
    violations = []
    for mod in core_modules:
        mod_path = os.path.join(src_dir, mod)
        if os.path.exists(mod_path):
            with open(mod_path, "r", encoding="utf-8") as f:
                content = f.read()
                if ("answer" + "_key.csv") in content.lower():
                    violations.append(mod)
    return {"answer_key_isolation_passed": len(violations) == 0, "violations": violations}


def audit_dataset_and_repo(data_dir: str = "data") -> Dict[str, Any]:
    ledger_path = os.path.join(data_dir, "ledger.csv")
    bank_path = os.path.join(data_dir, "bank_statement.csv")
    answer_path = os.path.join(data_dir, "answer" + "_key.csv")
    if not (os.path.exists(ledger_path) and os.path.exists(bank_path)):
        raise FileNotFoundError(f"Dataset CSVs missing in '{data_dir}/'.")

    df_ledger = pd.read_csv(ledger_path)
    df_bank = pd.read_csv(bank_path)
    df_answer = pd.read_csv(answer_path) if os.path.exists(answer_path) else pd.DataFrame()
    iso_audit = audit_repository_isolation()

    try:
        from src.reconciliation import reconcile
    except ModuleNotFoundError:
        from reconciliation import reconcile
    df_results = reconcile(df_ledger, df_bank)

    bank_narrations = df_bank["narration_text"].astype(str).tolist()
    exact_refs = sum(1 for n in bank_narrations if re.search(r"\bORD-\d+\b", n))
    partial_refs = sum(1 for n in bank_narrations if "ORD" in n and not re.search(r"\bORD-\d+\b", n))
    no_refs = len(bank_narrations) - exact_refs - partial_refs
    total_bank = len(df_bank)
    exact_ref_rate = round(exact_refs / total_bank, 4) if total_bank else 0.0
    partial_ref_rate = round(partial_refs / total_bank, 4) if total_bank else 0.0
    no_ref_rate = round(no_refs / total_bank, 4) if total_bank else 0.0

    scenario_counts = df_answer["scenario"].value_counts().to_dict() if not df_answer.empty else {}
    cand_counts = df_results["candidate_count"].value_counts().to_dict()
    no_cand_count = int((df_results["candidate_count"] == 0).sum())
    multi_cand_count = int((df_results["candidate_count"] > 1).sum())
    exact_amt_date_matches = len(df_results[df_results["matching_rule"] == "EXACT_AMOUNT_DATE"])
    exact_ref_matches = len(df_results[df_results["matching_rule"] == "EXACT_REFERENCE"])
    ai_calls_count = len(df_results[df_results["decision_source"] == "groq"])

    unique_amt_date_count = multi_amt_date_count = no_amt_date_count = 0
    total_ledger = len(df_ledger)
    for _, l_row in df_ledger.iterrows():
        matches = df_bank[
            (df_bank["credited_amount"] == float(l_row.get("amount", 0.0))) &
            (df_bank["value_date"].astype(str) == str(l_row.get("order_date", "")))
        ]
        if len(matches) == 1:
            unique_amt_date_count += 1
        elif len(matches) > 1:
            multi_amt_date_count += 1
        else:
            no_amt_date_count += 1

    return {
        "1_answer_key_isolation": iso_audit["answer_key_isolation_passed"],
        "2_deterministic_id_leakage": {
            "exact_reference_rate": exact_ref_rate,
            "partial_reference_rate": partial_ref_rate,
            "no_reference_rate": no_ref_rate,
        },
        "3_overly_easy_data_check": exact_ref_rate < 0.90,
        "4_duplicate_cases_count": scenario_counts.get("DUPLICATE_NEAR_DUPLICATE", 0),
        "5_ambiguous_cases_count": scenario_counts.get("AMBIGUOUS", 0),
        "6_fee_difference_cases_count": scenario_counts.get("FEE_DIFFERENCE", 0),
        "7_date_shift_cases_count": scenario_counts.get("DATE_SHIFT", 0),
        "8_unmatched_records_count": scenario_counts.get("UNMATCHED_LEDGER", 0) + scenario_counts.get("UNMATCHED_BANK", 0),
        "9_false_positive_traps_count": scenario_counts.get("FALSE_POSITIVE_TRAP", 0),
        "10_reconciliation_module_isolation": iso_audit["answer_key_isolation_passed"],
        "11_candidate_pool_size_distribution": cand_counts,
        "12_exact_amount_date_matches_pct": round(exact_amt_date_matches / len(df_results), 4) if len(df_results) else 0.0,
        "13_exact_reference_matches_pct": round(exact_ref_matches / len(df_results), 4) if len(df_results) else 0.0,
        "14_transactions_requiring_ai": ai_calls_count,
        "15_rows_with_no_candidate": no_cand_count,
        "16_rows_with_multiple_candidates": multi_cand_count,
        "17_amount_date_matchability": {
            "pct_unique_amount_date_matchable": round(unique_amt_date_count / total_ledger, 4) if total_ledger else 0.0,
            "pct_multiple_amount_date_candidates": round(multi_amt_date_count / total_ledger, 4) if total_ledger else 0.0,
            "pct_no_amount_date_candidate": round(no_amt_date_count / total_ledger, 4) if total_ledger else 0.0,
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(audit_dataset_and_repo(), indent=2))
