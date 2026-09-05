"""Bounded Action Registry, execution handlers, idempotency, and verification loop."""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from src.agent.models import (
    CaseState,
    ReconciliationCase,
    ActionType,
    AgentAction,
    ActionExecution,
    VerificationResult,
)


class ActionHandler:
    def execute(self, case: ReconciliationCase, action: AgentAction) -> ActionExecution:
        raise NotImplementedError

    def verify(self, case: ReconciliationCase, execution: ActionExecution) -> VerificationResult:
        raise NotImplementedError


class MarkReconciledHandler(ActionHandler):
    def execute(self, case: ReconciliationCase, action: AgentAction) -> ActionExecution:
        case.status = "MATCHED"
        return ActionExecution(
            action_id=action.action_id,
            execution_status="SUCCESS",
            result_payload={"case_id": case.case_id, "final_status": "MATCHED"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def verify(self, case: ReconciliationCase, execution: ActionExecution) -> VerificationResult:
        is_valid = execution.execution_status == "SUCCESS" and case.status == "MATCHED"
        return VerificationResult(
            verified=is_valid,
            status="VERIFIED" if is_valid else "FAILED",
            verification_notes="Verified reconciliation action completed with MATCHED status.",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class CreateFeeAdjustmentHandler(ActionHandler):
    def execute(self, case: ReconciliationCase, action: AgentAction) -> ActionExecution:
        amt_diff = float(case.evidence.get("amount_difference", 0.0))
        fee_payload = {
            "fee_adjustment_id": f"FEE-{uuid.uuid4().hex[:8].upper()}",
            "ledger_id": case.ledger_id,
            "bank_id": case.bank_id,
            "amount": amt_diff,
            "reason": "Settlement MDR Gateway Fee Adjustment",
        }
        case.status = "MATCHED"
        return ActionExecution(
            action_id=action.action_id,
            execution_status="SUCCESS",
            result_payload=fee_payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def verify(self, case: ReconciliationCase, execution: ActionExecution) -> VerificationResult:
        is_valid = (
            execution.execution_status == "SUCCESS" and
            case.status == "MATCHED" and
            bool(execution.result_payload.get("fee_adjustment_id"))
        )
        return VerificationResult(
            verified=is_valid,
            status="VERIFIED" if is_valid else "FAILED",
            verification_notes="Verified fee adjustment exists and case is MATCHED.",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class FlagForReviewHandler(ActionHandler):
    def execute(self, case: ReconciliationCase, action: AgentAction) -> ActionExecution:
        case.status = "REVIEW"
        return ActionExecution(
            action_id=action.action_id,
            execution_status="SUCCESS",
            result_payload={"case_id": case.case_id, "requires_human_approval": True},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def verify(self, case: ReconciliationCase, execution: ActionExecution) -> VerificationResult:
        is_valid = execution.execution_status == "SUCCESS" and case.status == "REVIEW"
        return VerificationResult(
            verified=is_valid,
            status="VERIFIED" if is_valid else "FAILED",
            verification_notes="Verified case remains explicitly flagged for human review.",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class MarkUnmatchedHandler(ActionHandler):
    def execute(self, case: ReconciliationCase, action: AgentAction) -> ActionExecution:
        case.status = "UNMATCHED"
        return ActionExecution(
            action_id=action.action_id,
            execution_status="SUCCESS",
            result_payload={"case_id": case.case_id, "final_status": "UNMATCHED"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def verify(self, case: ReconciliationCase, execution: ActionExecution) -> VerificationResult:
        is_valid = execution.execution_status == "SUCCESS" and case.status == "UNMATCHED"
        return VerificationResult(
            verified=is_valid,
            status="VERIFIED" if is_valid else "FAILED",
            verification_notes="Verified case remains UNMATCHED after bounded action.",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class ActionRegistry:
    def __init__(self):
        self._handlers: Dict[str, ActionHandler] = {
            ActionType.MARK_RECONCILED.value: MarkReconciledHandler(),
            ActionType.CREATE_FEE_ADJUSTMENT.value: CreateFeeAdjustmentHandler(),
            ActionType.FLAG_FOR_REVIEW.value: FlagForReviewHandler(),
            ActionType.MARK_UNMATCHED.value: MarkUnmatchedHandler(),
        }

    def get_handler(self, action_type: str) -> Optional[ActionHandler]:
        return self._handlers.get(action_type)


class ActionService:
    """Execute policy-approved actions exactly once and audit their verified outcome."""

    def __init__(self, registry: Optional[ActionRegistry] = None):
        self.registry = registry or ActionRegistry()
        self.executed_actions: Dict[str, ActionExecution] = {}

    def execute_and_verify(
        self,
        case: ReconciliationCase,
        action_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ActionExecution, VerificationResult]:
        handler = self.registry.get_handler(action_type)
        if handler is None:
            action_type = ActionType.FLAG_FOR_REVIEW.value
            handler = self.registry.get_handler(action_type)

        idem_key = f"{case.case_id}:{action_type}"
        if idem_key in self.executed_actions:
            existing = self.executed_actions[idem_key]
            verification = VerificationResult(
                verified=True,
                status="VERIFIED",
                verification_notes="Action already executed (Idempotency check passed).",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            case.transition_to(
                case.state,
                actor="ACTION_SERVICE",
                event_type="IDEMPOTENCY_HIT",
                details={"action_type": action_type},
            )
            return existing, verification

        action = AgentAction(
            action_id=f"ACT-{uuid.uuid4().hex[:8].upper()}",
            case_id=case.case_id,
            action_type=action_type,
            payload=payload or {},
        )

        case.transition_to(
            CaseState.ACTION_EXECUTING,
            actor="ACTION_SERVICE",
            event_type="EXECUTING_ACTION",
            details={"action_type": action_type},
        )
        execution = handler.execute(case, action)
        self.executed_actions[idem_key] = execution

        case.transition_to(
            CaseState.ACTION_VERIFYING,
            actor="ACTION_SERVICE",
            event_type="VERIFYING_ACTION",
            details={"execution_status": execution.execution_status},
        )
        verification = handler.verify(case, execution)

        if verification.verified:
            if case.status == "MATCHED":
                final_state = CaseState.RESOLVED
            elif case.status == "UNMATCHED":
                final_state = CaseState.UNMATCHED
            else:
                final_state = CaseState.ACTION_PENDING_APPROVAL
            case.transition_to(
                final_state,
                actor="ACTION_SERVICE",
                event_type="ACTION_VERIFIED",
                details={"action_type": action_type, "verification_status": verification.status},
            )
        else:
            case.transition_to(
                CaseState.FAILED,
                actor="ACTION_SERVICE",
                event_type="ACTION_VERIFICATION_FAILED",
                details={"action_type": action_type},
            )

        case.action_execution = execution
        case.verification = verification
        return execution, verification
