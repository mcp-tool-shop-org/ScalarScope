"""Adversarial and rotating professor mechanisms.

This module addresses Schola AI's critique about "professor bias" by implementing:

1. ROTATING SELECTION: Cycle through professors rather than using all every time
   - Prevents student from learning to please all professors simultaneously
   - Forces genuine learning rather than consensus-seeking

2. ADVERSARIAL PROFESSOR: Trained to exploit student weaknesses
   - Finds patterns the student uses to "game" other professors
   - Penalizes hedging, surface patterns, and feature gaming
   - Provides a robustness check against shortcut learning

3. HOLDOUT VALIDATION: Reserve professors for validation
   - Never used during training
   - Measures true generalization vs professor-pleasing

4. COUNTER-PROFESSOR: Deliberately contradicts training professors
   - Provides adversarial pressure
   - Ensures student learns substance, not style

Usage:
    from scalarscope.professors import ProfessorEnsemble
    from scalarscope.professors.adversarial import (
        RotatingSelector,
        AdversarialProfessor,
        HoldoutManager,
    )

    # Rotating selection
    selector = RotatingSelector(professors, rotation_size=2)
    for cycle in training:
        active_profs = selector.get_active(cycle)
        evaluation = EnsembleEvaluation([p.evaluate(...) for p in active_profs])

    # Adversarial professor
    adversary = AdversarialProfessor()
    adversary.observe(response, critiques)  # Learn student patterns
    adversary_critique = adversary.evaluate(item, response)  # Penalize patterns

Literature:
- Adversarial Training: arXiv 2024 "Adversarial Training Can Provably Improve Robustness"
- Multi-Task Robustness: Springer "Multitask Learning Strengthens Adversarial Robustness"
- Ensemble Disagreement: NeurIPS 2017 "Deep Ensembles for Uncertainty Estimation"
See docs/LITERATURE_REVIEW.md Section 8 for full references.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from abc import ABC, abstractmethod
import random
import numpy as np

from .base import Professor
from ..core import (
    TokenVector,
    TokenDimension,
    TrainingItem,
    StudentResponse,
    ProfessorCritique,
    EnsembleEvaluation,
)


@dataclass
class RotationConfig:
    """Configuration for professor rotation."""
    # How many professors are active each cycle
    rotation_size: int = 2

    # Minimum cycles before same professor can be reselected
    cooldown_cycles: int = 3

    # Whether to ensure all professors get equal time
    balanced: bool = True

    # Random seed for reproducibility
    seed: int = 42


class RotatingSelector:
    """Selects a rotating subset of professors each training cycle.

    Instead of using ALL professors every cycle (which allows students
    to learn consensus patterns), this rotates through subsets.

    Benefits:
    1. Prevents "pleasing everyone" strategy
    2. Forces learning robust features that generalize
    3. More efficient (fewer evaluations per cycle)
    """

    def __init__(
        self,
        professors: List[Professor],
        config: Optional[RotationConfig] = None,
    ):
        self.professors = professors
        self.config = config or RotationConfig()
        self._rng = random.Random(self.config.seed)

        # Track usage for balanced selection
        self._usage_counts: Dict[str, int] = {
            p.professor_id: 0 for p in professors
        }
        self._last_active: Dict[str, int] = {
            p.professor_id: -self.config.cooldown_cycles for p in professors
        }
        self._current_cycle = 0

    def get_active(self, cycle: int) -> List[Professor]:
        """Get the active professors for this cycle.

        Args:
            cycle: Current training cycle

        Returns:
            List of professors to use this cycle
        """
        self._current_cycle = cycle

        # Filter by cooldown
        available = [
            p for p in self.professors
            if cycle - self._last_active[p.professor_id] >= self.config.cooldown_cycles
        ]

        # If not enough available (rare), allow some through
        if len(available) < self.config.rotation_size:
            available = self.professors.copy()

        # Select based on strategy
        if self.config.balanced:
            # Prefer least-used professors
            available.sort(key=lambda p: self._usage_counts[p.professor_id])
            selected = available[:self.config.rotation_size]
        else:
            # Random selection
            selected = self._rng.sample(
                available,
                min(self.config.rotation_size, len(available))
            )

        # Update tracking
        for p in selected:
            self._usage_counts[p.professor_id] += 1
            self._last_active[p.professor_id] = cycle

        return selected

    def get_usage_stats(self) -> Dict[str, int]:
        """Get professor usage counts."""
        return self._usage_counts.copy()

    def reset(self):
        """Reset usage tracking."""
        for pid in self._usage_counts:
            self._usage_counts[pid] = 0
            self._last_active[pid] = -self.config.cooldown_cycles


@dataclass
class PatternObservation:
    """Observation of a student pattern that may indicate gaming."""
    pattern_type: str  # "hedging", "verbosity", "marker_stuffing", etc.
    frequency: float  # How often this pattern appears
    correlated_with_success: float  # Correlation with high tokens
    confidence: float  # Confidence in detection


class AdversarialProfessor(Professor):
    """Professor that learns and penalizes student gaming patterns.

    This professor observes training and learns to detect:
    1. Hedging patterns (words like "might", "perhaps" used to inflate calibration)
    2. Marker stuffing (adding keywords other professors look for)
    3. Verbosity gaming (length optimization)
    4. Surface patterns that correlate with success but lack substance

    The adversarial professor provides:
    - LOW tokens when it detects gaming patterns
    - HIGH tokens when response seems genuine despite patterns

    This creates pressure against shortcut learning.
    """

    def __init__(self, sensitivity: float = 0.5):
        """Initialize adversarial professor.

        Args:
            sensitivity: How aggressively to penalize patterns (0-1)
        """
        super().__init__(
            "adversarial",
            "Adversarial Professor",
            "Detects and penalizes gaming patterns. Rewards genuine reasoning."
        )
        self.sensitivity = sensitivity

        # Learned patterns
        self._hedge_words: Set[str] = {
            "might", "perhaps", "could", "possibly", "maybe",
            "uncertain", "likely", "probably", "somewhat"
        }
        self._marker_words_by_prof: Dict[str, Set[str]] = {
            "strict_logician": {"therefore", "thus", "hence", "consequently"},
            "pragmatic_engineer": {"tradeoff", "risk", "cost", "practical"},
            "empathy_advocate": {"impact", "stakeholder", "user", "harm"},
        }

        # Pattern statistics
        self._observations: List[Tuple[Dict, float]] = []  # (features, success)
        self._pattern_scores: Dict[str, float] = {}  # Higher = more gaming

    def observe(
        self,
        response: StudentResponse,
        critiques: List[ProfessorCritique],
    ):
        """Observe a response and its critiques to learn patterns.

        Call this during training to let the adversary learn.
        """
        features = self._extract_features(response)

        # Success = average token total from professors
        if critiques:
            success = np.mean([c.tokens.total for c in critiques])
        else:
            success = 0.5

        self._observations.append((features, success))

        # Update pattern scores periodically
        if len(self._observations) % 50 == 0:
            self._update_pattern_scores()

    def _extract_features(self, response: StudentResponse) -> Dict[str, float]:
        """Extract features that might indicate gaming."""
        text = response.reasoning_trace.lower()
        words = text.split()
        n_words = len(words) + 1e-6

        features = {}

        # Hedge word density
        hedge_count = sum(1 for w in words if w in self._hedge_words)
        features["hedge_density"] = hedge_count / n_words

        # Marker word presence per professor
        for prof_id, markers in self._marker_words_by_prof.items():
            marker_count = sum(1 for w in words if w in markers)
            features[f"markers_{prof_id}"] = marker_count / n_words

        # Length (normalized)
        features["length"] = min(1.0, len(text) / 1000)

        # Confidence level
        features["confidence"] = response.confidence

        # Question marks (often indicates hedging)
        features["question_density"] = text.count("?") / n_words

        return features

    def _update_pattern_scores(self):
        """Update pattern gaming scores from observations."""
        if len(self._observations) < 20:
            return

        # Build correlation matrix
        feature_names = list(self._observations[0][0].keys())
        n_features = len(feature_names)

        X = np.array([[obs[0][f] for f in feature_names] for obs in self._observations])
        y = np.array([obs[1] for obs in self._observations])

        # Compute correlation of each feature with success
        for i, fname in enumerate(feature_names):
            if np.std(X[:, i]) > 1e-6 and np.std(y) > 1e-6:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                # High correlation = potentially gaming
                self._pattern_scores[fname] = corr
            else:
                self._pattern_scores[fname] = 0.0

    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> ProfessorCritique:
        """Evaluate response, penalizing detected gaming patterns."""
        features = self._extract_features(response)
        weaknesses = []
        tokens = {}

        # Base correctness (we still care about being right)
        is_correct = self._base_correctness_check(item, response)
        tokens[TokenDimension.CORRECTNESS] = 1.0 if is_correct else 0.0

        # Calculate gaming score
        gaming_score = 0.0
        gaming_reasons = []

        for fname, value in features.items():
            if fname in self._pattern_scores:
                # If feature correlates with success, it might be gaming
                pattern_score = self._pattern_scores[fname]
                if pattern_score > 0.3 and value > 0.2:  # Threshold
                    contribution = pattern_score * value * self.sensitivity
                    gaming_score += contribution
                    gaming_reasons.append(f"{fname}: {value:.2f}")

        gaming_score = min(1.0, gaming_score)

        # Penalize gaming patterns
        if gaming_score > 0.3:
            weaknesses.append(f"Gaming detected: {', '.join(gaming_reasons[:3])}")

        # Tokens inversely related to gaming
        anti_gaming = 1.0 - gaming_score

        tokens[TokenDimension.COHERENCE] = anti_gaming * 0.8 if is_correct else 0.3
        tokens[TokenDimension.TRADEOFFS] = anti_gaming * 0.7
        tokens[TokenDimension.CALIBRATION] = anti_gaming * 0.8
        tokens[TokenDimension.CLARITY] = anti_gaming * 0.7

        if gaming_score > 0.5:
            critique = f"Gaming patterns detected (score: {gaming_score:.2f}). Focus on substance."
        elif gaming_score > 0.3:
            critique = f"Minor gaming indicators. Some patterns correlate with reward without adding value."
        else:
            critique = "Genuine reasoning without obvious gaming patterns."

        return ProfessorCritique(
            professor_id=self.professor_id,
            tokens=TokenVector(tokens),
            is_correct=is_correct,
            critique_text=critique,
            specific_weaknesses=weaknesses,
        )

    def get_gaming_report(self) -> str:
        """Get human-readable report on detected gaming patterns."""
        lines = ["ADVERSARIAL PROFESSOR - GAMING PATTERN REPORT"]
        lines.append("=" * 50)
        lines.append(f"Observations: {len(self._observations)}")
        lines.append("")

        if self._pattern_scores:
            lines.append("PATTERN SCORES (correlation with success):")
            sorted_scores = sorted(
                self._pattern_scores.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for name, score in sorted_scores:
                indicator = "⚠️" if score > 0.3 else "  "
                lines.append(f"  {indicator} {name}: {score:+.3f}")
        else:
            lines.append("Not enough observations to detect patterns yet.")

        return "\n".join(lines)


class CounterProfessor(Professor):
    """Professor that deliberately contradicts the consensus.

    When most professors give high tokens, this gives low (and vice versa).
    Forces the student to learn robust features that work even under
    adversarial evaluation.

    Use sparingly - too much weight makes training unstable.
    """

    def __init__(self, inversion_strength: float = 0.5):
        """Initialize counter professor.

        Args:
            inversion_strength: How much to invert (0=no inversion, 1=full)
        """
        super().__init__(
            "counter_professor",
            "Counter Professor",
            "Deliberately provides contrarian evaluation. Tests robustness."
        )
        self.inversion_strength = inversion_strength
        self._last_consensus: Optional[TokenVector] = None

    def set_consensus(self, consensus: TokenVector):
        """Set the consensus tokens from other professors."""
        self._last_consensus = consensus

    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> ProfessorCritique:
        """Produce inverted evaluation relative to consensus."""
        is_correct = self._base_correctness_check(item, response)
        tokens = {}

        if self._last_consensus is None:
            # No consensus to invert, return neutral
            for dim in TokenDimension:
                tokens[dim] = 0.5
        else:
            # Invert consensus tokens
            for dim in TokenDimension:
                consensus_val = self._last_consensus.dimensions.get(dim, 0.5)
                inverted = 1.0 - consensus_val

                # Blend with neutral based on strength
                tokens[dim] = (
                    self.inversion_strength * inverted +
                    (1 - self.inversion_strength) * 0.5
                )

        # Always respect correctness somewhat
        tokens[TokenDimension.CORRECTNESS] = min(
            tokens.get(TokenDimension.CORRECTNESS, 0.5),
            1.0 if is_correct else 0.3
        )

        return ProfessorCritique(
            professor_id=self.professor_id,
            tokens=TokenVector(tokens),
            is_correct=is_correct,
            critique_text="Contrarian perspective for robustness testing.",
            specific_weaknesses=[],
        )


@dataclass
class HoldoutConfig:
    """Configuration for holdout validation."""
    # Fraction of professors to hold out
    holdout_fraction: float = 0.25

    # Or specific professor IDs to hold out
    holdout_ids: Optional[List[str]] = None

    # Seed for random selection
    seed: int = 42


class HoldoutManager:
    """Manages professor holdout for validation.

    Reserves some professors that are NEVER used during training.
    These can be used to test true generalization vs professor-pleasing.
    """

    def __init__(
        self,
        professors: List[Professor],
        config: Optional[HoldoutConfig] = None,
    ):
        self.config = config or HoldoutConfig()
        self._rng = random.Random(self.config.seed)

        # Separate training and holdout professors
        if self.config.holdout_ids:
            holdout_ids = set(self.config.holdout_ids)
            self.training_professors = [
                p for p in professors if p.professor_id not in holdout_ids
            ]
            self.holdout_professors = [
                p for p in professors if p.professor_id in holdout_ids
            ]
        else:
            # Random selection
            n_holdout = max(1, int(len(professors) * self.config.holdout_fraction))
            shuffled = professors.copy()
            self._rng.shuffle(shuffled)
            self.holdout_professors = shuffled[:n_holdout]
            self.training_professors = shuffled[n_holdout:]

    def get_training_professors(self) -> List[Professor]:
        """Get professors for training (excludes holdout)."""
        return self.training_professors

    def get_holdout_professors(self) -> List[Professor]:
        """Get holdout professors for validation only."""
        return self.holdout_professors

    def validate_generalization(
        self,
        items: List[TrainingItem],
        responses: List[StudentResponse],
    ) -> Dict[str, float]:
        """Evaluate on holdout professors to measure generalization.

        Args:
            items: Test items
            responses: Student responses

        Returns:
            Dict mapping professor_id to average token total
        """
        results = {}

        for prof in self.holdout_professors:
            totals = []
            for item, response in zip(items, responses):
                critique = prof.evaluate(item, response)
                totals.append(critique.tokens.total)
            results[prof.professor_id] = float(np.mean(totals))

        return results


class AdversarialEnsemble:
    """Enhanced ensemble with adversarial components.

    Combines:
    1. Rotating professor selection
    2. Adversarial professor (learns gaming patterns)
    3. Optional counter-professor
    4. Holdout management

    Usage:
        ensemble = AdversarialEnsemble(professors)

        for cycle, (item, response) in enumerate(training):
            # Get evaluation
            evaluation = ensemble.evaluate(item, response, cycle)

            # Train on evaluation
            ...

        # Validate on holdout
        generalization = ensemble.validate_generalization(test_items, test_responses)
    """

    def __init__(
        self,
        professors: List[Professor],
        rotation_config: Optional[RotationConfig] = None,
        holdout_config: Optional[HoldoutConfig] = None,
        use_adversarial: bool = True,
        use_counter: bool = False,
        adversarial_weight: float = 0.2,
        counter_weight: float = 0.1,
    ):
        # Setup holdout first
        self.holdout_manager = HoldoutManager(professors, holdout_config)
        training_profs = self.holdout_manager.get_training_professors()

        # Setup rotation
        self.selector = RotatingSelector(training_profs, rotation_config)

        # Setup adversarial components
        self.adversarial_prof = AdversarialProfessor() if use_adversarial else None
        self.counter_prof = CounterProfessor() if use_counter else None

        self.adversarial_weight = adversarial_weight
        self.counter_weight = counter_weight

    def evaluate(
        self,
        item: TrainingItem,
        response: StudentResponse,
        cycle: int,
    ) -> EnsembleEvaluation:
        """Evaluate with rotating selection and adversarial professors.

        Args:
            item: Training item
            response: Student response
            cycle: Current training cycle

        Returns:
            EnsembleEvaluation with all critiques
        """
        critiques = []

        # Get active professors for this cycle
        active_profs = self.selector.get_active(cycle)
        for prof in active_profs:
            critique = prof.evaluate(item, response)
            critiques.append(critique)

        # Add adversarial professor
        if self.adversarial_prof:
            # Let adversary observe
            self.adversarial_prof.observe(response, critiques)

            # Get adversarial critique
            adv_critique = self.adversarial_prof.evaluate(item, response)
            critiques.append(adv_critique)

        # Add counter professor
        if self.counter_prof and critiques:
            # Compute consensus
            token_avgs = {}
            for dim in TokenDimension:
                vals = [c.tokens.dimensions.get(dim, 0.5) for c in critiques]
                token_avgs[dim] = np.mean(vals)
            consensus = TokenVector(token_avgs)

            self.counter_prof.set_consensus(consensus)
            counter_critique = self.counter_prof.evaluate(item, response)
            critiques.append(counter_critique)

        return EnsembleEvaluation.from_critiques(critiques)

    def validate_generalization(
        self,
        items: List[TrainingItem],
        responses: List[StudentResponse],
    ) -> Dict[str, float]:
        """Validate on holdout professors."""
        return self.holdout_manager.validate_generalization(items, responses)

    def get_adversarial_report(self) -> str:
        """Get gaming pattern report from adversarial professor."""
        if self.adversarial_prof:
            return self.adversarial_prof.get_gaming_report()
        return "Adversarial professor not enabled."

    def get_usage_stats(self) -> Dict[str, int]:
        """Get professor rotation usage statistics."""
        return self.selector.get_usage_stats()
