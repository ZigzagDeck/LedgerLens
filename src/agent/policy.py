"""Deterministic Policy Engine enforcing financial safety boundaries and autonomy rules."""

import re
from typing import Optional
from src.agent.models import (
    ReconciliationCase,
    ActionType,
    PolicyDecision,
    CaseInvestigation,
)


class PolicyEngine:
    """Evaluate recommendations against deterministic financial guardrails."""

    def __init__(
        self,
        max_auto_fee_amount: float = 100.0,
        min_auto_match_score: float = 0.82,
        require_approval_for_fees: bool = False,
    ):
        self.max_auto_fee_amount = max_auto_fee_amount
        self.min_auto_match_score = min_auto_match_score
        self.require_approval_for_fees = require_approval_for_fees

    def sanitize_untrusted_text(self, text: str, max_length: int = 250) -> str:
        """Defense-in-depth sanitizer for transaction text used by agent components."""
        if not text:
            return ""
        cleaned = str(text)
        cleaned = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", cleaned)
        cleaned = re.sub(r"https?://\S+", "[REDACTED_URL]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"www\.\S+", "[REDACTED_URL]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(.)\1{10,}", r"\1\1\1[REDACTED_FLOOD]", cleaned)
        injection_patterns = [
            r"ignore\s+all\s+previous", r"ignore\s+prior\s+instructions",
            r"disregard\s+(all\s+)?(previous|prior|earlier|above)\s+(instructions?|prompts?|context)",
            r"forget\s+(everything|all|prior|previous|your\s+instructions)",
            r"you\s+are\s+now\s+(a|an|the)",
            r"act\s+as\s+(a|an|the)?\s*\w+\s*(without\s+restrictions|freely|unrestricted)",
            r"pretend\s+(you\s+are|to\s+be)\s", r"roleplay\s+as",
            r"switch\s+to\s+developer\s+mode", r"jailbreak", r"system\s*prompt",
            r"override\s+policy", r"override\s+all\s+(rules|restrictions|policies)",
            r"bypass\s+(safety|filter|policy|rule)", r"disable\s+(safety|filter|restriction)",
            r"mark\s+as\s+reconciled", r"mark\s+as\s+matched",
            r"approve\s+(this\s+)?(transaction|payment|record)", r"force\s+(match|reconcile|approve)",
            r"auto[_\s]?approve", r"\[\s*INST\s*\]", r"<\s*system\s*>", r"<\s*user\s*>",
            r"\{\{\s*system",
        ]
        cleaned = re.sub(
            "|".join(f"(?:{p})" for p in injection_patterns),
            "[REDACTED_TEXT]", cleaned, flags=re.IGNORECASE,
        )
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "[TRUNCATED]"
        return cleaned.strip()

    def evaluate(
        self,
        case: ReconciliationCase,
        investigation: Optional[CaseInvestigation] = None,
    ) -> PolicyDecision:
        """Return the bounded action allowed for this case."""
        decision_source = str(case.evidence.get("decision_source", "deterministic")).lower()
        matching_rule = str(case.evidence.get("matching_rule", ""))

        # Deterministic high-confidence matches and already-veto-checked AI confirmations are final matches.
        if case.status == "MATCHED" and (
            case.score >= self.min_auto_match_score or
            (decision_source == "groq" and matching_rule == "AI_CONFIRMED_MATCH")
        ):
            return PolicyDecision(
                allowed=True,
                action_type=ActionType.MARK_RECONCILED.value,
                requires_approval=False,
                policy_code=(
                    "POL_AI_CONFIRMED_MATCH" if decision_source == "groq"
                    else "POL_HIGH_CONFIDENCE_MATCH"
                ),
                risk_level="LOW",
                reason=(
                    "Bounded AI match passed candidate-ID and deterministic safety vetoes"
                    if decision_source == "groq"
                    else f"Score {case.score:.4f} exceeds auto-match threshold {self.min_auto_match_score}"
                ),
            )

        if case.status == "UNMATCHED":
            return PolicyDecision(
                allowed=True,
                action_type=ActionType.MARK_UNMATCHED.value,
                requires_approval=False,
                policy_code="POL_CONFIRMED_UNMATCHED",
                risk_level="LOW",
                reason="No matching candidate found in bank statement",
            )

        rec_action = investigation.recommended_action if investigation else ActionType.FLAG_FOR_REVIEW.value
        reason_code = investigation.reason_code if investigation else "AMBIGUOUS"

        if reason_code == "FEE_ADJUSTMENT" or rec_action == ActionType.CREATE_FEE_ADJUSTMENT.value:
            amt_diff = float(case.evidence.get("amount_difference", 0.0))
            if 0 <= amt_diff <= self.max_auto_fee_amount and not self.require_approval_for_fees:
                return PolicyDecision(
                    allowed=True,
                    action_type=ActionType.CREATE_FEE_ADJUSTMENT.value,
                    requires_approval=False,
                    policy_code="POL_FEE_WITHIN_TOLERANCE",
                    risk_level="LOW",
                    reason=f"Fee variance ₹{amt_diff:.2f} is within auto-adjustment limit ₹{self.max_auto_fee_amount:.2f}",
                )
            return PolicyDecision(
                allowed=True,
                action_type=ActionType.CREATE_FEE_ADJUSTMENT.value,
                requires_approval=True,
                policy_code="POL_FEE_EXCEEDS_AUTO_LIMIT",
                risk_level="MEDIUM",
                reason=f"Fee variance ₹{amt_diff:.2f} exceeds limit ₹{self.max_auto_fee_amount:.2f} — human approval required",
            )

        if case.exception_type in ("ONE_TO_ONE_CONFLICT", "CURRENCY_MISMATCH"):
            return PolicyDecision(
                allowed=True,
                action_type=ActionType.FLAG_FOR_REVIEW.value,
                requires_approval=True,
                policy_code="POL_HIGH_RISK_EXCEPTION",
                risk_level="HIGH",
                reason=f"High risk exception type '{case.exception_type}' requires human review",
            )

        return PolicyDecision(
            allowed=True,
            action_type=ActionType.FLAG_FOR_REVIEW.value,
            requires_approval=True,
            policy_code="POL_REQUIRE_HUMAN_REVIEW",
            risk_level="MEDIUM",
            reason=f"Case status '{case.status}' with score {case.score:.4f} requires human review",
        )
