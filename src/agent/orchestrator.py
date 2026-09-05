"""Reconciliation Agent Orchestrator managing the stateful agent execution loop."""

import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import pandas as pd

from src.agent.models import CaseState, ReconciliationCase, ActionType
from src.agent.policy import PolicyEngine
from src.agent.actions import ActionService
from src.agent.investigator import ExceptionInvestigator
from src.reconciliation import reconcile
from src.config import ReconciliationConfig, CONFIG
from src.services.finance_controller import _classify_exception


@dataclass
class AgentRunSummary:
    run_id: str
    batch_status: str
    total_cases: int = 0
    resolved_count: int = 0
    auto_resolved_count: int = 0
    fee_adjusted_count: int = 0
    pending_approval_count: int = 0
    unmatched_count: int = 0
    verification_pass_rate: float = 1.0
    audit_events_count: int = 0
    summary_markdown: str = ""
    cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReconciliationAgent:
    """Bounded reconciliation workflow: observe -> normalize -> reconcile -> investigate -> policy -> act -> verify -> audit."""

    def __init__(
        self,
        policy_engine: Optional[PolicyEngine] = None,
        action_service: Optional[ActionService] = None,
        investigator: Optional[ExceptionInvestigator] = None,
    ):
        self.policy_engine = policy_engine or PolicyEngine()
        self.action_service = action_service or ActionService()
        self.investigator = investigator or ExceptionInvestigator(policy_engine=self.policy_engine)
        self.cases: Dict[str, ReconciliationCase] = {}

    def observe_and_reconcile(
        self,
        df_ledger: pd.DataFrame,
        df_bank: pd.DataFrame,
        run_id: Optional[str] = None,
        config: ReconciliationConfig = CONFIG,
        precomputed_results: Optional[pd.DataFrame] = None,
    ) -> AgentRunSummary:
        """Execute the full agent pipeline using the caller's exact reconciliation config/results."""
        run_id = run_id or f"RUN-{uuid.uuid4().hex[:8].upper()}"
        df_results = precomputed_results.copy() if precomputed_results is not None else reconcile(df_ledger, df_bank, config=config)
        self.cases.clear()

        ledger_lookup = (
            df_ledger.set_index("order_id").to_dict("index") if "order_id" in df_ledger.columns else {}
        )
        bank_lookup = (
            df_bank.set_index("utr_reference").to_dict("index") if "utr_reference" in df_bank.columns else {}
        )

        for _, row in df_results.iterrows():
            l_id = str(row.get("ledger_id", ""))
            b_id = str(row.get("bank_id", ""))
            status = str(row.get("status", "UNMATCHED"))
            score = float(row.get("score", 0.0))
            rule = str(row.get("matching_rule", ""))
            init_state = (
                CaseState.MATCHED if status == "MATCHED"
                else CaseState.AMBIGUOUS if status == "REVIEW"
                else CaseState.UNMATCHED
            )

            exc_obj = _classify_exception(row)
            exc_type = exc_obj.exception_type if exc_obj else "NONE"
            evidence = {
                "score": score,
                "matching_rule": rule,
                "amount_difference": float(row.get("amount_difference", 0.0)),
                "date_difference": int(row.get("date_difference", 0)),
                "decision_source": str(row.get("decision_source", "deterministic")),
            }

            ledger_row_data = dict(ledger_lookup.get(l_id, {}))
            if l_id:
                ledger_row_data["order_id"] = l_id
            ledger_row_data.update({
                "ai_reason": str(row.get("ai_reason", "")),
                "matching_rule": rule,
                "decision_source": str(row.get("decision_source", "deterministic")),
            })

            bank_row_data = dict(bank_lookup.get(b_id, {}))
            if b_id:
                bank_row_data["utr_reference"] = b_id

            decision_source = str(row.get("decision_source", "deterministic"))
            if status == "REVIEW" and decision_source != "groq" and b_id and b_id in bank_lookup:
                candidates_for_case = [{
                    "score": score,
                    "utr_reference": b_id,
                    "bank_row": bank_row_data,
                    "breakdown": {
                        "ref": float(row.get("original_score", score)),
                        "amount": 0.0,
                        "date": 0.0,
                        "text": 0.0,
                    },
                }]
            else:
                candidates_for_case = []

            case = ReconciliationCase(
                case_id=f"CASE-{uuid.uuid4().hex[:8].upper()}",
                ledger_id=l_id,
                bank_id=b_id,
                state=CaseState.NEW,
                ledger_record=ledger_row_data,
                bank_record=bank_row_data,
                candidates=candidates_for_case,
                score=score,
                status=status,
                exception_type=exc_type,
                evidence=evidence,
            )

            case.transition_to(CaseState.INGESTING, "AGENT", "CASE_INGESTED")
            case.transition_to(CaseState.NORMALIZING, "AGENT", "CASE_NORMALIZED")
            case.transition_to(CaseState.RECONCILING, "RECONCILIATION_ENGINE", "CASE_RECONCILING")
            case.transition_to(
                init_state,
                actor="RECONCILIATION_ENGINE",
                event_type="RECONCILIATION_DECIDED",
                details={"matching_rule": rule, "score": score, "status": status},
            )
            self.cases[case.case_id] = case

        resolved_cnt = auto_resolved_cnt = fee_adjusted_cnt = 0
        pending_approval_cnt = unmatched_cnt = verified_cnt = total_actions = 0

        for case in self.cases.values():
            if case.state == CaseState.MATCHED:
                policy_dec = self.policy_engine.evaluate(case)
                case.policy_decision = policy_dec
                if policy_dec.allowed and not policy_dec.requires_approval:
                    _, verif_res = self.action_service.execute_and_verify(case, policy_dec.action_type)
                    resolved_cnt += 1
                    auto_resolved_cnt += 1
                    verified_cnt += int(verif_res.verified)
                    total_actions += 1
                else:
                    case.transition_to(
                        CaseState.ACTION_PENDING_APPROVAL,
                        "POLICY_ENGINE",
                        "APPROVAL_REQUIRED",
                        {"reason": policy_dec.reason},
                    )
                    pending_approval_cnt += 1

            elif case.state == CaseState.UNMATCHED:
                policy_dec = self.policy_engine.evaluate(case)
                case.policy_decision = policy_dec
                _, verif_res = self.action_service.execute_and_verify(
                    case, ActionType.MARK_UNMATCHED.value
                )
                unmatched_cnt += 1
                verified_cnt += int(verif_res.verified)
                total_actions += 1

            elif case.state == CaseState.AMBIGUOUS:
                case.transition_to(CaseState.INVESTIGATING, "AGENT", "INVESTIGATION_STARTED")
                investigation = self.investigator.investigate(case)
                case.investigation = investigation
                case.transition_to(
                    CaseState.RECOMMENDATION_READY,
                    "EXCEPTION_INVESTIGATOR",
                    "RECOMMENDATION_FORMULATED",
                    {"recommended_action": investigation.recommended_action},
                )
                policy_dec = self.policy_engine.evaluate(case, investigation)
                case.policy_decision = policy_dec

                if policy_dec.allowed and not policy_dec.requires_approval:
                    case.transition_to(
                        CaseState.POLICY_APPROVED,
                        "POLICY_ENGINE",
                        "POLICY_APPROVED",
                        {"policy_code": policy_dec.policy_code},
                    )
                    _, verif_res = self.action_service.execute_and_verify(case, policy_dec.action_type)
                    fee_adjusted_cnt += int(policy_dec.action_type == ActionType.CREATE_FEE_ADJUSTMENT.value)
                    resolved_cnt += 1
                    auto_resolved_cnt += 1
                    verified_cnt += int(verif_res.verified)
                    total_actions += 1
                else:
                    case.transition_to(
                        CaseState.ACTION_PENDING_APPROVAL,
                        "POLICY_ENGINE",
                        "APPROVAL_REQUIRED",
                        {"reason": policy_dec.reason},
                    )
                    pending_approval_cnt += 1

        total = len(self.cases)
        verif_rate = verified_cnt / total_actions if total_actions else 1.0
        batch_status = "READY_TO_CLOSE" if pending_approval_cnt == 0 else "REVIEW_REQUIRED"
        audit_cnt = sum(len(c.audit_history) for c in self.cases.values())
        summary_md = (
            f"### Agent Execution Summary for {run_id}\n\n"
            f"- **Total Cases**: {total}\n"
            f"- **Resolved**: {resolved_cnt} (Auto: {auto_resolved_cnt}, Fee Adjustments: {fee_adjusted_cnt})\n"
            f"- **Pending Human Approval**: {pending_approval_cnt}\n"
            f"- **Unmatched**: {unmatched_cnt}\n"
            f"- **Verification Pass Rate**: {verif_rate * 100:.1f}%\n"
            f"- **Audit Events Logged**: {audit_cnt}\n"
        )

        return AgentRunSummary(
            run_id=run_id,
            batch_status=batch_status,
            total_cases=total,
            resolved_count=resolved_cnt,
            auto_resolved_count=auto_resolved_cnt,
            fee_adjusted_count=fee_adjusted_cnt,
            pending_approval_count=pending_approval_cnt,
            unmatched_count=unmatched_cnt,
            verification_pass_rate=round(verif_rate, 4),
            audit_events_count=audit_cnt,
            summary_markdown=summary_md,
            cases=[c.to_dict() for c in self.cases.values()],
        )
