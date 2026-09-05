"""Configuration constants and schema definitions for financial reconciliation."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ReconciliationConfig:
    """Config settings for reconciliation matching thresholds, weights, and limits."""
    # Phase 1 Deterministic Thresholds
    AMOUNT_TOLERANCE: float = 0.01
    DATE_WINDOW_DAYS: int = 3
    DEFAULT_CURRENCY: str = "INR"

    # Phase 2 Broad Candidate Generation Limits
    BROAD_DATE_WINDOW_DAYS: int = 10
    BROAD_AMOUNT_TOLERANCE_PCT: float = 0.05
    MAX_FEE_AMOUNT: float = 100.0
    TOP_N_CANDIDATES: int = 5
    AI_CANDIDATE_LIMIT: int = 3  # Must be <= TOP_N_CANDIDATES; controls how many candidates are shown to AI

    # Evidence Weights for Phase 2 Scoring (Sum = 1.0)
    W_REF: float = 0.40
    W_AMOUNT: float = 0.30
    W_DATE: float = 0.20
    W_TEXT: float = 0.10

    # Decision Engine Thresholds & Safety Guards
    HIGH_CONFIDENCE_THRESHOLD: float = 0.82
    REVIEW_THRESHOLD: float = 0.45
    AMBIGUITY_MARGIN: float = 0.08

    # Phase 3 Bounded Groq AI Assistance Settings
    GROQ_MODEL: str = "groq/compound"
    ENABLE_AI_ASSIST: bool = True
    GROQ_MAX_CALLS_PER_MINUTE: int = 25
    # Three retries produce the documented 2s, 4s, 8s exponential backoff sequence.
    GROQ_RETRY_ATTEMPTS: int = 3
    GROQ_RETRY_BACKOFF_BASE: float = 2.0

    # Custom Data & Debug Mode Configuration
    LEDGERLENS_CUSTOM_DATA_DIR: str = "data/custom"
    LEDGERLENS_DEBUG_MODE: bool = False

    # Schema column definitions. Some human-readable fields are optional at runtime,
    # but the canonical generated/exported schema includes all of these columns.
    LEDGER_COLUMNS: List[str] = field(
        default_factory=lambda: [
            "order_id",
            "customer_name",
            "amount",
            "currency",
            "order_date",
            "payment_method",
        ]
    )

    BANK_COLUMNS: List[str] = field(
        default_factory=lambda: [
            "utr_reference",
            "narration_text",
            "credited_amount",
            "currency",
            "value_date",
            "deduction_fee",
        ]
    )

    ANSWER_KEY_COLUMNS: List[str] = field(
        default_factory=lambda: [
            "order_id",
            "utr_reference",
            "scenario",
            "expected_status",
            "notes",
        ]
    )


# Default config instance
CONFIG = ReconciliationConfig()
