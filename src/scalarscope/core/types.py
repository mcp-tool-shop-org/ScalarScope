"""Core data types for ASPIRE training."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING
from .tokens import TokenVector

if TYPE_CHECKING:
    from ..student.onnx_student_v2 import GenerationStats


@dataclass
class TrainingItem:
    """A single test item for the student.

    Contains the prompt, ground truth, and optional teaching materials.
    """
    id: str
    prompt: str                           # The scenario/question
    gold_answer: str                      # Correct answer
    gold_rationale: str                   # Why it's correct (the teaching moment)
    difficulty: float = 0.5              # 0-1 scale
    domain: str = "general"
    near_misses: List[str] = field(default_factory=list)  # Common wrong answers


@dataclass
class StudentResponse:
    """What the student produces for a test item."""
    item_id: str
    answer: str
    reasoning_trace: str
    confidence: float                     # 0-1, student's self-assessed certainty
    latency_ms: float = 0.0              # How long inference took
    generation_stats: Optional["GenerationStats"] = None  # Logit-derived stats for V1 critic


@dataclass
class ProfessorCritique:
    """Feedback from a single professor."""
    professor_id: str
    tokens: TokenVector
    is_correct: bool
    critique_text: str                    # Natural language feedback
    specific_weaknesses: List[str] = field(default_factory=list)


@dataclass
class EnsembleEvaluation:
    """Aggregated evaluation from all professors."""
    critiques: List[ProfessorCritique]
    aggregated_tokens: TokenVector
    consensus_correct: bool              # Did majority agree it's correct?
    disagreement_score: float            # Variance across professors (0-1)

    @classmethod
    def from_critiques(
        cls,
        critiques: List[ProfessorCritique],
        strict_dims: Optional[List[str]] = None,
    ) -> "EnsembleEvaluation":
        """Aggregate critiques using min for strict dims, mean for others.

        Args:
            critiques: Individual professor evaluations
            strict_dims: Dimensions where we take the minimum (strictest critic wins)
        """
        from .tokens import TokenDimension

        if not critiques:
            return cls([], TokenVector(), False, 0.0)

        strict_dims = strict_dims or ["correctness", "coherence"]
        strict_set = {TokenDimension(d) for d in strict_dims}

        # Aggregate tokens
        aggregated = {}
        for dim in TokenDimension:
            values = [c.tokens.values[dim] for c in critiques]
            if dim in strict_set:
                aggregated[dim] = min(values)
            else:
                aggregated[dim] = sum(values) / len(values)

        # Consensus and disagreement
        correct_votes = sum(1 for c in critiques if c.is_correct)
        consensus = correct_votes > len(critiques) / 2

        # Disagreement = variance in correctness judgment
        if len(critiques) > 1:
            mean_correct = correct_votes / len(critiques)
            variance = sum(
                (1.0 if c.is_correct else 0.0 - mean_correct) ** 2
                for c in critiques
            ) / len(critiques)
            disagreement = min(variance * 4, 1.0)  # Scale to 0-1
        else:
            disagreement = 0.0

        return cls(
            critiques=critiques,
            aggregated_tokens=TokenVector(aggregated),
            consensus_correct=consensus,
            disagreement_score=disagreement,
        )


@dataclass
class TeachingMoment:
    """The reveal phase: what the student sees after evaluation."""
    item: TrainingItem
    student_response: StudentResponse
    evaluation: EnsembleEvaluation
    tokens_earned: TokenVector            # Final payout
    should_revise: bool                   # Critic thinks student should try again
