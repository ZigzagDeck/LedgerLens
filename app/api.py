"""FastAPI REST API for LedgerLens reconciliation, agent workflow, and observability."""

import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CONFIG
from src.data_validation import validate_ledger_schema, validate_bank_schema, validate_custom_data_paths
from src.reconciliation import reconcile
from src.services.finance_controller import process_batch, detect_batch_aggregates
from src.agent import ReconciliationAgent, ActionType


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


_CACHE_FILE = Path("data") / ".ledgerlens_cache.json"
_MAX_CACHE_RUNS = 50
_RUN_CACHE: Dict[str, Dict[str, Any]] = {}
_ACTIVE_AGENT: Optional[ReconciliationAgent] = None


def _load_cache_from_disk() -> None:
    if not _CACHE_FILE.exists():
        return
    try:
        loaded = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            _RUN_CACHE.update(loaded)
    except Exception:
        return


def _save_cache_to_disk() -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        run_keys = [k for k in _RUN_CACHE if k != "latest"]
        if len(run_keys) > _MAX_CACHE_RUNS:
            for old_key in run_keys[:-_MAX_CACHE_RUNS]:
                _RUN_CACHE.pop(old_key, None)
        _CACHE_FILE.write_text(
            json.dumps(_RUN_CACHE, cls=_NumpyEncoder, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        return


@asynccontextmanager
async def _lifespan(_: "FastAPI"):
    _load_cache_from_disk()
    yield


app = FastAPI(
    title="LedgerLens API",
    description="Financial Reconciliation Engine with Bounded Groq AI & Observability Traces",
    version="1.1.0",
    lifespan=_lifespan,
)


def _load_input_data(use_custom_data: bool) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    custom_dir = CONFIG.LEDGERLENS_CUSTOM_DATA_DIR
    ledger_xlsx = os.path.join(custom_dir, "ledger.xlsx")
    bank_xlsx = os.path.join(custom_dir, "bank_statement.xlsx")

    if use_custom_data:
        missing = [
            path for path in (ledger_xlsx, bank_xlsx) if not os.path.exists(path)
        ]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Custom data requested but the custom dataset pair is incomplete. "
                    "Upload both ledger and bank files before using use_custom_data=true."
                ),
            )
        return (
            pd.read_excel(ledger_xlsx, engine="openpyxl"),
            pd.read_excel(bank_xlsx, engine="openpyxl"),
            "custom_xlsx",
        )

    ledger_csv = os.path.join("data", "ledger.csv")
    bank_csv = os.path.join("data", "bank_statement.csv")
    if not (os.path.exists(ledger_csv) and os.path.exists(bank_csv)):
        raise HTTPException(
            status_code=404,
            detail="Default dataset missing. Generate dataset first or upload custom files.",
        )
    return pd.read_csv(ledger_csv), pd.read_csv(bank_csv), "default_benchmark"


@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    custom_valid, custom_paths = validate_custom_data_paths()
    return {
        "status": "ok",
        "service": "LedgerLens Financial Reconciliation API",
        "version": "1.1.0",
        "default_data_available": os.path.exists("data/ledger.csv") and os.path.exists("data/bank_statement.csv"),
        "custom_data_dir": CONFIG.LEDGERLENS_CUSTOM_DATA_DIR,
        "custom_data_available": custom_valid,
        "custom_paths": custom_paths,
        "ai_enabled": CONFIG.ENABLE_AI_ASSIST,
        "groq_key_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
    }


@app.post("/api/v1/custom-data/upload")
async def upload_custom_data(
    file: UploadFile = File(...),
    file_type: str = Form(..., description="Type of dataset: 'ledger' or 'bank'"),
):
    ft = file_type.strip().lower()
    if ft not in {"ledger", "bank"}:
        raise HTTPException(status_code=400, detail="file_type must be either 'ledger' or 'bank'.")

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".xlsx", ".csv"}:
        raise HTTPException(status_code=400, detail="Only .xlsx or .csv files are supported for upload.")

    try:
        df = pd.read_excel(file.file, engine="openpyxl") if suffix == ".xlsx" else pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {exc}")

    if ft == "ledger":
        is_valid, errs = validate_ledger_schema(df)
        target_filename = "ledger.xlsx"
    else:
        is_valid, errs = validate_bank_schema(df)
        target_filename = "bank_statement.xlsx"
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Schema validation failed: {errs}")

    custom_dir = CONFIG.LEDGERLENS_CUSTOM_DATA_DIR
    os.makedirs(custom_dir, exist_ok=True)
    target_path = os.path.join(custom_dir, target_filename)
    try:
        with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save custom dataset: {exc}")

    return {
        "status": "success",
        "file_type": ft,
        "filename": filename,
        "saved_path": target_path,
        "row_count": len(df),
    }


def _trace_from_row(row: pd.Series) -> Dict[str, Any]:
    ai_invoked = str(row.get("decision_source", "")) == "groq" or bool(row.get("ai_reason", ""))
    ai_reason = str(row.get("ai_reason", ""))
    status = str(row.get("status", ""))
    rule = str(row.get("matching_rule", ""))
    reason = str(row.get("reason", ""))
    amount_difference = float(row.get("amount_difference", 0.0))

    candidate_id_check = (
        "failed_hallucinated_id"
        if "hallucinated" in ai_reason.lower() or "VETO: AI bank ID" in ai_reason
        else "passed"
    )
    currency_check = "failed" if rule == "CURRENCY_MISMATCH" or "currency mismatch" in reason.lower() else "passed"
    one_to_one_check = "conflict_downgraded" if rule == "ONE_TO_ONE_CONFLICT" else "passed"

    if rule in {"EXACT_REFERENCE", "EXACT_AMOUNT_DATE"}:
        amount_check = "passed"
    elif amount_difference <= CONFIG.MAX_FEE_AMOUNT:
        amount_check = "passed"
    elif status == "MATCHED":
        amount_check = "scored_broad_tolerance"
    else:
        amount_check = "not_applicable"

    hard_checks_passed = (
        candidate_id_check == "passed" and
        currency_check == "passed" and
        one_to_one_check == "passed"
    )
    if not ai_invoked:
        ai_validation = "not_invoked"
    elif candidate_id_check != "passed":
        ai_validation = "vetoed"
    elif status == "MATCHED":
        ai_validation = "passed"
    else:
        ai_validation = "review_suggested"

    return {
        "ledger_id": row.get("ledger_id", ""),
        "bank_id": row.get("bank_id", ""),
        "status": status,
        "matching_rule": rule,
        "score": row.get("score", 0.0),
        "reason": reason,
        "decision_source": row.get("decision_source", "deterministic"),
        "evidence_breakdown": row.get("evidence_breakdown", {}) or {},
        "ai_invoked": ai_invoked,
        "ai_result_summary": ai_reason if ai_invoked else "N/A",
        "ai_validation_result": ai_validation,
        "hard_safety_checks": "passed" if hard_checks_passed else "flagged",
        "candidate_id_check": candidate_id_check,
        "amount_safety_check": amount_check,
        "currency_safety_check": currency_check,
        "one_to_one_check": one_to_one_check,
        "final_decision": status,
        "candidate_count": row.get("candidate_count", 0),
        "candidate_rank": row.get("candidate_rank", 1),
        "amount_difference": amount_difference,
        "date_difference": row.get("date_difference", 0),
    }


@app.post("/api/v1/reconcile")
def run_reconciliation_endpoint(
    debug: Optional[bool] = Query(None, description="Enable structured reconciliation trace output"),
    use_custom_data: bool = Query(False, description="Use uploaded custom datasets"),
):
    debug_mode = debug if debug is not None else (
        os.getenv("LEDGERLENS_DEBUG_MODE", "").lower() == "true" or CONFIG.LEDGERLENS_DEBUG_MODE
    )
    df_ledger, df_bank, data_source = _load_input_data(use_custom_data)
    try:
        df_results = reconcile(df_ledger, df_bank)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reconciliation error: {exc}")

    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    status_counts = df_results["status"].value_counts().to_dict()
    sources = df_results["decision_source"].value_counts().to_dict()
    response_data: Dict[str, Any] = {
        "status": "success",
        "run_id": run_id,
        "data_source": data_source,
        "total_records": len(df_results),
        "matched_count": status_counts.get("MATCHED", 0),
        "review_count": status_counts.get("REVIEW", 0),
        "unmatched_count": status_counts.get("UNMATCHED", 0),
        "summary": {
            "matched": status_counts.get("MATCHED", 0),
            "review": status_counts.get("REVIEW", 0),
            "unmatched": status_counts.get("UNMATCHED", 0),
            "decision_sources": sources,
        },
    }
    if debug_mode:
        response_data["traces"] = [_trace_from_row(row) for _, row in df_results.iterrows()]

    response_data["batch"] = process_batch(df_results, run_id=run_id).to_dict()
    try:
        aggregates = detect_batch_aggregates(df_ledger, df_bank, df_results)
    except Exception:
        aggregates = []
    response_data["batch_aggregates"] = aggregates
    response_data["batch_aggregate_count"] = len(aggregates)

    _RUN_CACHE[run_id] = response_data
    _RUN_CACHE["latest"] = response_data
    _save_cache_to_disk()
    return response_data


@app.get("/api/v1/reconcile/")
@app.get("/api/v1/reconcile/{run_id}")
def get_reconciliation_run(run_id: str = "latest"):
    key = run_id or "latest"
    if key not in _RUN_CACHE:
        raise HTTPException(status_code=404, detail=f"Reconciliation run '{key}' not found.")
    return _RUN_CACHE[key]


@app.post("/api/v1/agent/run")
def run_agent_endpoint(
    use_custom_data: bool = Query(False, description="Use uploaded custom datasets"),
):
    global _ACTIVE_AGENT
    df_ledger, df_bank, _ = _load_input_data(use_custom_data)
    agent = ReconciliationAgent()
    try:
        summary = agent.observe_and_reconcile(df_ledger, df_bank)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {exc}")
    _ACTIVE_AGENT = agent
    return summary.to_dict()


@app.get("/api/v1/cases")
def list_cases_endpoint():
    if _ACTIVE_AGENT is None:
        raise HTTPException(status_code=404, detail="No agent run found. Call POST /api/v1/agent/run first.")
    return {
        "case_count": len(_ACTIVE_AGENT.cases),
        "cases": [c.to_dict() for c in _ACTIVE_AGENT.cases.values()],
    }


@app.get("/api/v1/cases/{case_id}")
def get_case_endpoint(case_id: str):
    if _ACTIVE_AGENT is None or case_id not in _ACTIVE_AGENT.cases:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return _ACTIVE_AGENT.cases[case_id].to_dict()


@app.post("/api/v1/cases/{case_id}/approve")
def approve_case_action_endpoint(case_id: str):
    if _ACTIVE_AGENT is None or case_id not in _ACTIVE_AGENT.cases:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    case = _ACTIVE_AGENT.cases[case_id]
    if not case.policy_decision:
        raise HTTPException(status_code=400, detail="Case has no policy decision.")
    if not case.policy_decision.requires_approval:
        raise HTTPException(status_code=400, detail="Case does not require human approval.")

    action_type = case.policy_decision.action_type
    if action_type == ActionType.FLAG_FOR_REVIEW.value:
        raise HTTPException(
            status_code=400,
            detail="This case has no safe executable financial action. Resolve the evidence first rather than approving a review flag.",
        )
    execution, verification = _ACTIVE_AGENT.action_service.execute_and_verify(case, action_type)
    return {
        "status": "success",
        "case": case.to_dict(),
        "execution": execution.to_dict(),
        "verification": verification.to_dict(),
    }


@app.get("/api/v1/audit")
def get_audit_trail_endpoint():
    if _ACTIVE_AGENT is None:
        raise HTTPException(status_code=404, detail="No agent run found. Call POST /api/v1/agent/run first.")
    events = [evt.to_dict() for case in _ACTIVE_AGENT.cases.values() for evt in case.audit_history]
    events.sort(key=lambda item: item.get("timestamp", ""))
    return {"event_count": len(events), "audit_events": events}
