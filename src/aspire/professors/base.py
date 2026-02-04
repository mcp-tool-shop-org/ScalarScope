"""Professor base class and implementations."""

from abc import ABC, abstractmethod
from typing import List, Optional
import random

from ..core import (
    TokenVector,
    TokenDimension,
    TrainingItem,
    StudentResponse,
    ProfessorCritique,
    EnsembleEvaluation,
)


class Professor(ABC):
    """Base class for professor evaluators.

    Each professor represents a distinct evaluative perspective.
    They critique student reasoning and award tokens based on
    their particular values/priorities.
    """

    def __init__(self, professor_id: str, name: str, description: str):
        self.professor_id = professor_id
        self.name = name
        self.description = description

    @abstractmethod
    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> ProfessorCritique:
        """Evaluate a student response and produce critique + tokens."""
        pass

    def _base_correctness_check(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> bool:
        """Simple correctness check (override for domain-specific)."""
        # Normalize and compare
        student = response.answer.strip().lower()
        gold = item.gold_answer.strip().lower()
        return student == gold or gold in student or student in gold


class StrictLogician(Professor):
    """Prioritizes logical consistency and valid reasoning chains."""

    def __init__(self):
        super().__init__(
            "strict_logician",
            "Strict Logician",
            "Values airtight reasoning. Penalizes contradictions, unsupported claims."
        )

    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> ProfessorCritique:
        is_correct = self._base_correctness_check(item, response)
        weaknesses = []
        tokens = {}

        # Correctness (binary for this professor)
        tokens[TokenDimension.CORRECTNESS] = 1.0 if is_correct else 0.0

        # Coherence check (simplified: look for contradiction markers)
        reasoning = response.reasoning_trace.lower()
        contradiction_markers = ["however", "but then again", "on the other hand"]
        has_unresolved = any(m in reasoning for m in contradiction_markers)

        if has_unresolved and "therefore" not in reasoning:
            tokens[TokenDimension.COHERENCE] = 0.3
            weaknesses.append("Introduced counterpoints without resolving them")
        else:
            tokens[TokenDimension.COHERENCE] = 0.8 if is_correct else 0.4

        # Tradeoffs: logician doesn't care much
        tokens[TokenDimension.TRADEOFFS] = 0.5

        # Calibration: penalize overconfidence when wrong
        if not is_correct and response.confidence > 0.8:
            tokens[TokenDimension.CALIBRATION] = 0.1
            weaknesses.append("High confidence despite incorrect answer")
        elif is_correct and response.confidence < 0.3:
            tokens[TokenDimension.CALIBRATION] = 0.5
            weaknesses.append("Underconfident despite correct reasoning")
        else:
            tokens[TokenDimension.CALIBRATION] = 0.7

        # Clarity: logician values precision
        tokens[TokenDimension.CLARITY] = 0.6

        critique = "Logically sound." if is_correct and not weaknesses else \
            f"Issues: {'; '.join(weaknesses)}" if weaknesses else "Incorrect conclusion."

        return ProfessorCritique(
            professor_id=self.professor_id,
            tokens=TokenVector(tokens),
            is_correct=is_correct,
            critique_text=critique,
            specific_weaknesses=weaknesses,
        )


class PragmaticEngineer(Professor):
    """Prioritizes practical applicability and real-world tradeoffs."""

    def __init__(self):
        super().__init__(
            "pragmatic_engineer",
            "Pragmatic Engineer",
            "Values practical solutions. Rewards acknowledging constraints and tradeoffs."
        )

    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> ProfessorCritique:
        is_correct = self._base_correctness_check(item, response)
        weaknesses = []
        tokens = {}

        tokens[TokenDimension.CORRECTNESS] = 1.0 if is_correct else 0.0

        # Coherence: more lenient
        tokens[TokenDimension.COHERENCE] = 0.7 if is_correct else 0.5

        # Tradeoffs: this professor's focus
        reasoning = response.reasoning_trace.lower()
        tradeoff_markers = ["tradeoff", "trade-off", "downside", "risk", "cost", "alternatively"]
        acknowledged_tradeoffs = sum(1 for m in tradeoff_markers if m in reasoning)

        if acknowledged_tradeoffs >= 2:
            tokens[TokenDimension.TRADEOFFS] = 1.0
        elif acknowledged_tradeoffs == 1:
            tokens[TokenDimension.TRADEOFFS] = 0.7
        else:
            tokens[TokenDimension.TRADEOFFS] = 0.3
            weaknesses.append("Didn't acknowledge practical tradeoffs")

        # Calibration
        tokens[TokenDimension.CALIBRATION] = 0.6

        # Clarity: values directness
        if len(response.reasoning_trace) > 500:
            tokens[TokenDimension.CLARITY] = 0.5
            weaknesses.append("Overly verbose")
        else:
            tokens[TokenDimension.CLARITY] = 0.8

        critique = "Practical and well-considered." if is_correct and not weaknesses else \
            f"Concerns: {'; '.join(weaknesses)}"

        return ProfessorCritique(
            professor_id=self.professor_id,
            tokens=TokenVector(tokens),
            is_correct=is_correct,
            critique_text=critique,
            specific_weaknesses=weaknesses,
        )


class EmpathyAdvocate(Professor):
    """Prioritizes stakeholder impact and ethical considerations."""

    def __init__(self):
        super().__init__(
            "empathy_advocate",
            "Empathy Advocate",
            "Values human impact. Rewards considering stakeholders and ethical dimensions."
        )

    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> ProfessorCritique:
        is_correct = self._base_correctness_check(item, response)
        weaknesses = []
        tokens = {}

        tokens[TokenDimension.CORRECTNESS] = 1.0 if is_correct else 0.0
        tokens[TokenDimension.COHERENCE] = 0.6  # Less focused on logic

        # Tradeoffs: specifically human/stakeholder tradeoffs
        reasoning = response.reasoning_trace.lower()
        empathy_markers = ["user", "people", "stakeholder", "impact", "affect", "harm", "benefit"]
        human_consideration = sum(1 for m in empathy_markers if m in reasoning)

        if human_consideration >= 2:
            tokens[TokenDimension.TRADEOFFS] = 0.9
        elif human_consideration == 1:
            tokens[TokenDimension.TRADEOFFS] = 0.6
        else:
            tokens[TokenDimension.TRADEOFFS] = 0.2
            weaknesses.append("No consideration of human impact")

        # Calibration: values epistemic humility
        if "might" in reasoning or "could" in reasoning or "uncertain" in reasoning:
            tokens[TokenDimension.CALIBRATION] = 0.9
        else:
            tokens[TokenDimension.CALIBRATION] = 0.5

        # Clarity: values accessibility
        tokens[TokenDimension.CLARITY] = 0.7

        critique = "Thoughtfully considers impact." if not weaknesses else \
            f"Missing: {'; '.join(weaknesses)}"

        return ProfessorCritique(
            professor_id=self.professor_id,
            tokens=TokenVector(tokens),
            is_correct=is_correct,
            critique_text=critique,
            specific_weaknesses=weaknesses,
        )


class ProfessorEnsemble:
    """Manages multiple professors and aggregates their evaluations."""

    def __init__(self, professors: Optional[List[Professor]] = None):
        self.professors = professors or [
            StrictLogician(),
            PragmaticEngineer(),
            EmpathyAdvocate(),
        ]

    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
        strict_dims: Optional[List[str]] = None,
    ) -> EnsembleEvaluation:
        """Get critiques from all professors and aggregate."""
        critiques = [
            prof.evaluate(item, response)
            for prof in self.professors
        ]
        return EnsembleEvaluation.from_critiques(critiques, strict_dims)

    def add_professor(self, professor: Professor):
        self.professors.append(professor)
