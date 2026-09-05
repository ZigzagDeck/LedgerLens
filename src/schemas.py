"""Typed schemas and data models for LedgerLens reconciliation system."""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ConfigDict


@dataclass
class EvidenceBreakdown:
    """Breakdown of individual evidence scores for candidate evaluation."""
    ref: float = 0.0
    amount: float = 0.0
    date: float = 0.0
    text: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class AIEvaluationSchema(BaseModel):
    """Pydantic schema for validating structured Groq LLM responses."""
    model_config = ConfigDict(extra="ignore")

    same_transaction: bool = False
    selected_bank_id: Optional[str] = None
    reference_evidence: str = ""
    amount_consistent: bool = False
    date_consistent: bool = False
    fee_explanation: str = ""
    reason: str = ""


@dataclass
class ReconciliationRecord:
    """Structured reconciliation result with complete observability fields."""
    ledger_id: str
    bank_id: str
    status: str
    matching_rule: str
    score: float
    reason: str
    decision_source: str = "deterministic"
    model_used: str = "none"
    ai_reason: str = ""
    original_score: float = 0.0
    amount_difference: float = 0.0
    date_difference: int = 0
    candidate_rank: int = 0
    candidate_count: int = 0
    evidence_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "bank_id": self.bank_id,
            "status": self.status,
            "matching_rule": self.matching_rule,
            "score": round(self.score, 4),
            "reason": self.reason,
            "decision_source": self.decision_source,
            "model_used": self.model_used,
            "ai_reason": self.ai_reason,
            "original_score": round(self.original_score, 4),
            "amount_difference": round(self.amount_difference, 2),
            "date_difference": self.date_difference,
            "candidate_rank": self.candidate_rank,
            "candidate_count": self.candidate_count,
            "evidence_breakdown": dict(self.evidence_breakdown or {}),
        }


@dataclass
class CandidateBank:
    """Structure representing a candidate bank statement match."""
    score: float
    utr_reference: str
    bank_row: Dict[str, Any]
    breakdown: EvidenceBreakdown


@dataclass
class EvaluationMetrics:
    """Metrics summary produced by evaluation benchmark against ground truth."""
    total_rows: int
    handled_without_ai: int
    sent_to_ai: int
    matches: int
    reviews: int
    unmatched: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    auto_match_rate: float
    review_rate: float
    unmatched_rate: float
    ai_examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
