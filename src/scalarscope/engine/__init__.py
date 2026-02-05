"""ScalarScope training engine."""

from .loop import ScalarScopeEngine, CycleResult, TrainingMetrics
from .revision import (
    RevisionEngine,
    RevisionConfig,
    RevisionDecision,
    RevisionResult,
    RevisionTrigger,
)
from .revision_engine import RevisionScalarScopeEngine, RevisionCycleResult, RevisionMetrics

__all__ = [
    "ScalarScopeEngine",
    "CycleResult",
    "TrainingMetrics",
    "RevisionEngine",
    "RevisionConfig",
    "RevisionDecision",
    "RevisionResult",
    "RevisionTrigger",
    "RevisionScalarScopeEngine",
    "RevisionCycleResult",
    "RevisionMetrics",
]

# Re-export governor for convenience
from ..governor import TokenPool, GovernorConfig
