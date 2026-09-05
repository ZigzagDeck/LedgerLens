"""Interactive Streamlit dashboard for LedgerLens."""

import io
import os
import sys
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import ReconciliationConfig, CONFIG
from src.reconciliation import reconcile
from src.evaluation import evaluate_reconciliation
from src.data_validation import validate_ledger_schema, validate_bank_schema
from src.agent import ReconciliationAgent


st.set_page_config(
    page_title="LedgerLens — AI Finance Controller",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {font-size:2.2rem;font-weight:700;margin-bottom:.2rem}
    .sub-header {font-size:1rem;opacity:.75;margin-bottom:1.3rem}
    .headline {padding:10px 14px;border-radius:8px;background:rgba(20,184,166,.12);font-weight:600}
    div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.18);border-radius:10px;padding:12px}
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_upload(uploaded) -> pd.DataFrame:
    return pd.read_excel(uploaded, engine="openpyxl") if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(uploaded)


def _clear_run_state() -> None:
    for key in ("reconciled_results", "eval_metrics", "agent_summary", "agent_audit"):
        st.session_state.pop(key, None)


def _load_samples(source_label: str) -> None:
    ledger_path = os.path.join(ROOT_DIR, "data", "ledger.csv")
    bank_path = os.path.join(ROOT_DIR, "data", "bank_statement.csv")
    if not (os.path.exists(ledger_path) and os.path.exists(bank_path)):
        st.error("Sample benchmark files are missing from data/.")
        return
    st.session_state["df_ledger"] = pd.read_csv(ledger_path)
    st.session_state["df_bank"] = pd.read_csv(bank_path)
    st.session_state["data_source_label"] = source_label
    _clear_run_state()


def _flatten_audit(summary) -> pd.DataFrame:
    events = []
    if not summary:
        return pd.DataFrame()
    for case in summary.cases:
        for event in case.get("audit_history", []):
            row = dict(event)
            row["ledger_id"] = case.get("ledger_id", "")
            row["bank_id"] = case.get("bank_id", "")
            events.append(row)
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values("timestamp")


st.sidebar.title("⚙️ Reconciliation Settings")
amt_tol = st.sidebar.number_input("Amount Tolerance (INR)", min_value=0.0, max_value=10.0, value=float(CONFIG.AMOUNT_TOLERANCE), step=0.01)
date_win = st.sidebar.number_input("Date Window (Days)", min_value=0, max_value=30, value=int(CONFIG.DATE_WINDOW_DAYS), step=1)
high_thresh = st.sidebar.slider("Auto-Match Threshold", 0.50, 0.95, float(CONFIG.HIGH_CONFIDENCE_THRESHOLD), 0.01)
review_thresh = st.sidebar.slider("Review Threshold", 0.10, 0.70, float(CONFIG.REVIEW_THRESHOLD), 0.01)
enable_ai = st.sidebar.checkbox("Enable Groq AI Assistance", value=bool(CONFIG.ENABLE_AI_ASSIST))
calls_per_min = st.sidebar.number_input("Max Groq Calls / Min", min_value=5, max_value=60, value=int(CONFIG.GROQ_MAX_CALLS_PER_MINUTE), step=5)

if enable_ai:
    configured_key = st.session_state.get("ledgerlens_groq_api_key", "")
    if not configured_key:
        try:
            configured_key = str(st.secrets.get("GROQ_API_KEY", ""))
        except Exception:
            configured_key = ""
    entered_key = st.sidebar.text_input(
        "Groq API Key (Live Inference)",
        value=configured_key,
        type="password",
        help="Stored only in this Streamlit session. Missing keys safely route ambiguous cases to REVIEW.",
    )
    st.session_state["ledgerlens_groq_api_key"] = entered_key.strip()
    if entered_key.strip():
        st.sidebar.caption("🟢 Live Groq key configured for this session")
    else:
        st.sidebar.caption("⚪ No key: ambiguous cases safely default to REVIEW")
else:
    st.session_state.pop("ledgerlens_groq_api_key", None)

run_mode = st.sidebar.radio("Evaluation Mode", ["Standard (Live Data)", "Benchmark (Ground Truth)"])
is_benchmark = run_mode.startswith("Benchmark")

user_config = ReconciliationConfig(
    AMOUNT_TOLERANCE=float(amt_tol),
    DATE_WINDOW_DAYS=int(date_win),
    HIGH_CONFIDENCE_THRESHOLD=float(high_thresh),
    REVIEW_THRESHOLD=float(review_thresh),
    ENABLE_AI_ASSIST=bool(enable_ai),
    GROQ_MAX_CALLS_PER_MINUTE=int(calls_per_min),
)

if st.session_state.get("active_mode") != run_mode:
    st.session_state["active_mode"] = run_mode
    _clear_run_state()
    if is_benchmark:
        _load_samples("Canonical Benchmark Dataset (data/)")
    else:
        for key in ("df_ledger", "df_bank", "data_source_label", "uploaded_filenames"):
            st.session_state.pop(key, None)
    st.rerun()

st.markdown('<div class="main-header">LedgerLens — AI Finance Controller</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deterministic Multi-Tier Matching & Guardrailed AI Assistance</div>', unsafe_allow_html=True)

if is_benchmark:
    st.info("🎯 Benchmark mode uses the canonical data/ datasets and evaluates final decisions against data/answer_key.csv.")
    if st.button("🔄 Reload Benchmark Dataset"):
        _load_samples("Canonical Benchmark Dataset (data/)")
        st.rerun()
    if "df_ledger" not in st.session_state or "df_bank" not in st.session_state:
        _load_samples("Canonical Benchmark Dataset (data/)")
else:
    c1, c2 = st.columns(2)
    with c1:
        ledger_file = st.file_uploader("Upload Internal Ledger (CSV or XLSX)", type=["csv", "xlsx"])
    with c2:
        bank_file = st.file_uploader("Upload Bank Statement (CSV or XLSX)", type=["csv", "xlsx"])

    if ledger_file and bank_file:
        pair = (ledger_file.name, bank_file.name)
        if st.session_state.get("uploaded_filenames") != pair:
            try:
                st.session_state["df_ledger"] = _read_upload(ledger_file)
                st.session_state["df_bank"] = _read_upload(bank_file)
                st.session_state["data_source_label"] = f"Uploaded ({ledger_file.name}, {bank_file.name})"
                st.session_state["uploaded_filenames"] = pair
                _clear_run_state()
                st.rerun()
            except Exception as exc:
                st.error(f"Could not parse uploaded files: {exc}")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("📁 Load Sample Datasets", use_container_width=True):
            _load_samples("Sample Datasets (data/)")
            st.rerun()
    with b2:
        if st.button("🔄 Reset / Clear Active Data", use_container_width=True):
            for key in ("df_ledger", "df_bank", "data_source_label", "uploaded_filenames"):
                st.session_state.pop(key, None)
            _clear_run_state()
            st.rerun()


df_ledger = st.session_state.get("df_ledger")
df_bank = st.session_state.get("df_bank")

if df_ledger is not None and df_bank is not None:
    st.success(
        f"✅ Active Data: {st.session_state.get('data_source_label', '')} · "
        f"Ledger {len(df_ledger):,} rows · Bank {len(df_bank):,} rows"
    )
    with st.expander("👁️ View Active Dataset Previews"):
        p1, p2 = st.columns(2)
        p1.dataframe(df_ledger.head(5), use_container_width=True, hide_index=True)
        p2.dataframe(df_bank.head(5), use_container_width=True, hide_index=True)

    valid_l, errs_l = validate_ledger_schema(df_ledger, user_config)
    valid_b, errs_b = validate_bank_schema(df_bank, user_config)
    if not valid_l or not valid_b:
        st.error("Dataset validation failed: " + "; ".join(errs_l + errs_b))
    else:
        label = "🎯 Run Reconciliation & Benchmark Evaluation" if is_benchmark else "🚀 Run Reconciliation Engine"
        if st.button(label, type="primary", use_container_width=True):
            with st.spinner("Reconciling transactions..."):
                try:
                    results = reconcile(df_ledger, df_bank, config=user_config)
                    agent = ReconciliationAgent()
                    agent_summary = agent.observe_and_reconcile(
                        df_ledger,
                        df_bank,
                        config=user_config,
                        precomputed_results=results,
                    )
                    st.session_state["reconciled_results"] = results
                    st.session_state["agent_summary"] = agent_summary
                    st.session_state["agent_audit"] = _flatten_audit(agent_summary)
                    if is_benchmark:
                        st.session_state["eval_metrics"] = evaluate_reconciliation(
                            data_dir=os.path.join(ROOT_DIR, "data"),
                            config=user_config,
                            precomputed_results=results,
                        )
                    else:
                        st.session_state["eval_metrics"] = {}
                except Exception as exc:
                    st.error(f"Reconciliation failed: {exc}")
                else:
                    st.rerun()
else:
    st.info("Load or upload a ledger and bank statement to begin.")


if "reconciled_results" in st.session_state:
    results = st.session_state["reconciled_results"]
    metrics = st.session_state.get("eval_metrics", {})
    agent_summary = st.session_state.get("agent_summary")
    audit_df = st.session_state.get("agent_audit", pd.DataFrame())

    total = len(results)
    matched = int((results["status"] == "MATCHED").sum())
    review = int((results["status"] == "REVIEW").sum())
    unmatched = int((results["status"] == "UNMATCHED").sum())
    ai_calls = int((results["decision_source"] == "groq").sum())

    st.markdown("---")
    st.subheader("📊 Reconciliation Performance Summary")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Records", total)
    k2.metric("Matched", matched)
    k3.metric("Review Required", review)
    k4.metric("Unmatched", unmatched)
    k5.metric("AI Escalations", ai_calls)

    if metrics:
        st.markdown(f'<div class="headline">🎯 {metrics.get("headline", "")}</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pair Precision", f"{metrics.get('pair_precision', 0):.4f}")
        m2.metric("Pair Recall", f"{metrics.get('pair_recall', 0):.4f}")
        m3.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
        m4.metric("Auto-Resolution Precision", f"{metrics.get('auto_resolution_precision', 0):.4f}")

    tabs = st.tabs(["🎯 Benchmark", "🔍 Exceptions", "🤖 Agent & Audit", "✅ Matched", "⚠️ Review", "❌ Unmatched"])
    tab_eval, tab_exc, tab_agent, tab_matched, tab_review, tab_unmatched = tabs

    with tab_eval:
        if metrics:
            cm = metrics.get("confusion_matrix", {})
            cm_df = pd.DataFrame(
                [[cm.get("TP", 0), cm.get("FN", 0)], [cm.get("FP", 0), cm.get("TN", 0)]],
                index=["Actual MATCHED", "Actual NON-MATCHED"],
                columns=["Predicted MATCHED", "Predicted NON-MATCHED"],
            )
            st.markdown("#### 2×2 Confusion Matrix")
            st.dataframe(cm_df, use_container_width=True)
            rates = metrics.get("rates", {})
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Automated Coverage", f"{rates.get('automated_coverage', 0)*100:.1f}%")
            r2.metric("Review Rate", f"{rates.get('review_rate', 0)*100:.1f}%")
            r3.metric("Deterministic Match Rate", f"{rates.get('deterministic_match_rate', 0)*100:.1f}%")
            r4.metric("AI Escalation Rate", f"{rates.get('ai_escalation_rate', 0)*100:.1f}%")
            with st.expander("Raw benchmark metrics"):
                st.json(metrics)
        else:
            st.info("Switch to Benchmark mode and run reconciliation to calculate ground-truth metrics.")

    with tab_exc:
        exceptions = results[results["status"].isin(["REVIEW", "UNMATCHED"])].copy()
        if exceptions.empty:
            st.success("No unresolved exceptions.")
        else:
            guidance = {
                "AMBIGUOUS_CANDIDATES": "Compare top candidates manually",
                "AI_REVIEW_REQUIRED": "AI inconclusive — manual evidence check",
                "SCORE_REVIEW": "Verify low-confidence evidence",
                "ONE_TO_ONE_CONFLICT": "Resolve duplicate bank claim",
                "NO_CANDIDATE": "Investigate missing settlement",
                "LOW_SCORE": "No plausible match found",
                "NO_MATCH": "Investigate unlinked bank deposit",
                "CURRENCY_MISMATCH": "Verify currency/FX externally",
            }
            exceptions["resolution_guidance"] = exceptions["matching_rule"].map(guidance).fillna("Manual review required")
            show = [c for c in ["ledger_id", "bank_id", "status", "matching_rule", "score", "reason", "resolution_guidance"] if c in exceptions.columns]
            st.dataframe(exceptions[show], use_container_width=True, hide_index=True)

    with tab_agent:
        if agent_summary:
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Cases", agent_summary.total_cases)
            a2.metric("Resolved", agent_summary.resolved_count)
            a3.metric("Fee Adjustments", agent_summary.fee_adjusted_count)
            a4.metric("Pending Approval", agent_summary.pending_approval_count)
            a5.metric("Verification", f"{agent_summary.verification_pass_rate*100:.1f}%")
            st.markdown(agent_summary.summary_markdown)
            case_df = pd.DataFrame(agent_summary.cases)
            case_cols = [c for c in ["case_id", "ledger_id", "bank_id", "state", "status", "exception_type", "score"] if c in case_df.columns]
            st.dataframe(case_df[case_cols], use_container_width=True, hide_index=True)

            st.markdown("#### Append-only Audit Registry")
            if audit_df.empty:
                st.info("No audit events generated.")
            else:
                audit_cols = [c for c in ["timestamp", "case_id", "ledger_id", "bank_id", "actor", "event_type", "from_state", "to_state"] if c in audit_df.columns]
                st.dataframe(audit_df[audit_cols], use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download Audit Trail (CSV)",
                    audit_df.to_csv(index=False).encode("utf-8"),
                    file_name="ledgerlens_audit.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("Run reconciliation to generate agent cases and audit events.")

    common_cols = [c for c in ["ledger_id", "bank_id", "status", "matching_rule", "score", "reason", "decision_source", "model_used", "evidence_breakdown"] if c in results.columns]
    with tab_matched:
        st.dataframe(results[results["status"] == "MATCHED"][common_cols], use_container_width=True, hide_index=True)
    with tab_review:
        st.dataframe(results[results["status"] == "REVIEW"][common_cols], use_container_width=True, hide_index=True)
    with tab_unmatched:
        st.dataframe(results[results["status"] == "UNMATCHED"][common_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Download Reconciliation Results (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
