"""Operational metrics for conscience formation.

These metrics transform "conscience" from a metaphor into a measurable property.

CORE METRICS:

1. Surprise Stability (σ_surprise)
   - Low variance in prediction error over time
   - High = consistent predictive alignment
   - Formula: 1 - (std(surprise) / mean(surprise))

2. Anisotropy Stability (σ_anisotropy)
   - Eigenvalue ratio remains stable across evaluation windows
   - High = consistent evaluation structure
   - Formula: 1 - cv(anisotropy_sequence)  # coefficient of variation

3. Cross-Professor Generalization (G_prof)
   - Prediction accuracy generalizes across held-out professors
   - High = learned judgment, not professor-pleasing
   - Formula: correlation(predicted_tokens, actual_tokens) on held-out professor

4. Geometric Persistence (P_geo)
   - Dimensional structure persists under input perturbation
   - High = robust internalization, not surface heuristics
   - Formula: similarity(structure_original, structure_perturbed)

CONSCIENCE SCORE:
Weighted combination: C = w1*σ_surprise + w2*σ_anisotropy + w3*G_prof + w4*P_geo

Threshold for "conscience present": C > 0.6

Literature:
- Prediction Error Learning: PubMed "Surprise beyond prediction error" (2014)
- Ensemble Disagreement: NeurIPS 2017 "Deep Ensembles for Uncertainty"
- Grokking/Phase Transitions: OpenReview "Information-Theoretic Progress Measures" (2024)
See docs/LITERATURE_REVIEW.md Sections 2, 5, 7 for full references.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..core import TokenDimension, TokenVector
from ..geometry import TrainingTrajectory, compute_effective_dimensionality, compute_anisotropy


@dataclass
class SurpriseStability:
    """Metrics for surprise (prediction error) stability."""
    # Raw statistics
    surprise_mean: float = 0.0
    surprise_std: float = 0.0
    surprise_min: float = 0.0
    surprise_max: float = 0.0

    # Trend (should decrease)
    surprise_trend: float = 0.0  # Slope of surprise over time

    # Stability score
    stability: float = 0.0  # 1 - (std / mean), clamped to [0, 1]

    @classmethod
    def compute(cls, surprises: List[float], window: int = 50) -> "SurpriseStability":
        """Compute from surprise history."""
        if len(surprises) < 2:
            return cls()

        arr = np.array(surprises)

        # Use late window for stability assessment
        if len(arr) > window:
            late = arr[-window:]
        else:
            late = arr

        mean = float(np.mean(late))
        std = float(np.std(late))

        # Stability: low variance relative to mean
        if mean > 1e-6:
            stability = max(0.0, min(1.0, 1.0 - (std / mean)))
        else:
            stability = 1.0  # Perfect if no surprise

        # Trend: fit line to full sequence
        if len(arr) > 10:
            x = np.arange(len(arr))
            trend = float(np.polyfit(x, arr, 1)[0])
        else:
            trend = 0.0

        return cls(
            surprise_mean=mean,
            surprise_std=std,
            surprise_min=float(np.min(arr)),
            surprise_max=float(np.max(arr)),
            surprise_trend=trend,
            stability=stability,
        )


@dataclass
class AnisotropyStability:
    """Metrics for anisotropy (eigenvalue ratio) stability."""
    # Anisotropy trajectory
    anisotropy_values: List[float] = field(default_factory=list)

    # Statistics
    anisotropy_mean: float = 1.0
    anisotropy_std: float = 0.0
    anisotropy_final: float = 1.0

    # Stability (coefficient of variation)
    stability: float = 0.0  # 1 - cv, clamped

    # Growth (should increase during training)
    growth_rate: float = 0.0

    @classmethod
    def compute(
        cls,
        trajectory: TrainingTrajectory,
        window: int = 20,
    ) -> "AnisotropyStability":
        """Compute from training trajectory."""
        if len(trajectory.snapshots) < window * 2:
            return cls()

        # Compute anisotropy over sliding windows
        states = [s.state for s in trajectory.snapshots]
        anisotropies = []

        for i in range(window, len(states) + 1):
            window_states = states[i - window:i]
            an = compute_anisotropy(window_states)
            anisotropies.append(an)

        if not anisotropies:
            return cls()

        arr = np.array(anisotropies)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        # Coefficient of variation
        cv = std / mean if mean > 1e-6 else 0.0
        stability = max(0.0, min(1.0, 1.0 - cv))

        # Growth rate
        if len(arr) > 10:
            x = np.arange(len(arr))
            growth = float(np.polyfit(x, arr, 1)[0])
        else:
            growth = 0.0

        return cls(
            anisotropy_values=anisotropies,
            anisotropy_mean=mean,
            anisotropy_std=std,
            anisotropy_final=float(arr[-1]) if len(arr) > 0 else 1.0,
            stability=stability,
            growth_rate=growth,
        )


@dataclass
class GeneralizationScore:
    """Metrics for cross-professor generalization.

    Tests whether the critic learned "how to judge" vs "who to please".
    """
    # Per-professor correlation when held out
    holdout_correlations: Dict[str, float] = field(default_factory=dict)

    # Mean generalization
    mean_generalization: float = 0.0

    # Worst-case generalization (important!)
    min_generalization: float = 0.0

    # Professor-specific overfitting detection
    overfitting_scores: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        predictions: List[TokenVector],
        actuals_by_professor: Dict[str, List[TokenVector]],
    ) -> "GeneralizationScore":
        """Compute generalization from held-out professor evaluations.

        For each professor:
        1. Train critic on other professors
        2. Test prediction on held-out professor
        3. Measure correlation
        """
        if not predictions or not actuals_by_professor:
            return cls()

        correlations = {}
        overfitting = {}

        for prof_id, prof_actuals in actuals_by_professor.items():
            if len(prof_actuals) != len(predictions):
                continue

            # Compute correlation of predicted total vs actual total
            pred_totals = np.array([p.total for p in predictions])
            actual_totals = np.array([a.total for a in prof_actuals])

            if np.std(pred_totals) > 1e-6 and np.std(actual_totals) > 1e-6:
                corr = float(np.corrcoef(pred_totals, actual_totals)[0, 1])
            else:
                corr = 0.0

            correlations[prof_id] = corr

            # Overfitting = variance in correlation across professors
            # (will be computed after loop)

        if not correlations:
            return cls()

        corr_values = list(correlations.values())
        mean_gen = float(np.mean(corr_values))
        min_gen = float(np.min(corr_values))

        # Overfitting: how much better on best professor vs worst
        for prof_id, corr in correlations.items():
            overfitting[prof_id] = corr - mean_gen

        return cls(
            holdout_correlations=correlations,
            mean_generalization=mean_gen,
            min_generalization=min_gen,
            overfitting_scores=overfitting,
        )


@dataclass
class ConscienceMetrics:
    """Complete conscience metrics for a training run."""
    # Component metrics
    surprise_stability: SurpriseStability = field(default_factory=SurpriseStability)
    anisotropy_stability: AnisotropyStability = field(default_factory=AnisotropyStability)
    generalization: GeneralizationScore = field(default_factory=GeneralizationScore)

    # Geometric metrics
    dimensional_collapse: float = 1.0  # final_dim / initial_dim
    geometric_persistence: float = 0.0  # structure similarity under perturbation

    # Training metadata
    total_cycles: int = 0
    final_accuracy: float = 0.0


@dataclass
class ConscienceScore:
    """Final conscience score with interpretation."""
    # Component scores (0-1)
    surprise_score: float = 0.0
    anisotropy_score: float = 0.0
    generalization_score: float = 0.0
    persistence_score: float = 0.0

    # Weighted total
    total_score: float = 0.0

    # Interpretation
    has_conscience: bool = False
    confidence: str = "none"  # none, weak, moderate, strong
    warnings: List[str] = field(default_factory=list)

    # Weights used
    weights: Dict[str, float] = field(default_factory=dict)


def compute_conscience_score(
    metrics: ConscienceMetrics,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.6,
) -> ConscienceScore:
    """Compute final conscience score from metrics.

    Args:
        metrics: ConscienceMetrics from training
        weights: Optional custom weights for components
        threshold: Score above which conscience is considered present

    Returns:
        ConscienceScore with interpretation
    """
    # Default weights
    if weights is None:
        weights = {
            "surprise": 0.3,
            "anisotropy": 0.2,
            "generalization": 0.35,
            "persistence": 0.15,
        }

    warnings = []

    # Component scores
    surprise_score = metrics.surprise_stability.stability

    # Anisotropy: want high stability AND growth
    aniso = metrics.anisotropy_stability
    aniso_base = aniso.stability
    # Bonus for growth (anisotropy should increase during training)
    growth_bonus = min(0.2, max(0.0, aniso.growth_rate * 10))
    anisotropy_score = min(1.0, aniso_base + growth_bonus)

    # Generalization: worst-case matters most
    gen = metrics.generalization
    generalization_score = gen.min_generalization if gen.min_generalization > 0 else gen.mean_generalization

    # Persistence (if available)
    persistence_score = metrics.geometric_persistence

    # Warnings for potential issues
    if metrics.dimensional_collapse < 0.3:
        warnings.append("HEURISTIC_COLLAPSE: Dimensional collapse too fast - may be surface learning")

    if gen.min_generalization < 0.3 and gen.mean_generalization > 0.6:
        warnings.append("PROFESSOR_PLEASING: High variance across professors - may not generalize")

    if metrics.surprise_stability.surprise_trend > 0:
        warnings.append("SURPRISE_INCREASING: Prediction error growing - critic not learning")

    if aniso.growth_rate < 0:
        warnings.append("ANISOTROPY_DECREASING: Structure collapsing - may be forgetting")

    # Compute total
    total = (
        weights["surprise"] * surprise_score +
        weights["anisotropy"] * anisotropy_score +
        weights["generalization"] * generalization_score +
        weights["persistence"] * persistence_score
    )

    # Interpretation
    has_conscience = total >= threshold and len(warnings) == 0

    if total < 0.3:
        confidence = "none"
    elif total < 0.5:
        confidence = "weak"
    elif total < 0.7:
        confidence = "moderate"
    else:
        confidence = "strong"

    # Downgrade if warnings
    if warnings:
        if confidence == "strong":
            confidence = "moderate"
        elif confidence == "moderate":
            confidence = "weak"

    return ConscienceScore(
        surprise_score=surprise_score,
        anisotropy_score=anisotropy_score,
        generalization_score=generalization_score,
        persistence_score=persistence_score,
        total_score=total,
        has_conscience=has_conscience,
        confidence=confidence,
        warnings=warnings,
        weights=weights,
    )
