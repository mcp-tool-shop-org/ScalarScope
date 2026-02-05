"""Conscience validation and failure mode detection.

This module provides tools to validate whether conscience formation
is genuine, and to detect specific failure modes.

FAILURE MODES (from Schola AI review):

1. HEURISTIC_COLLAPSE
   - Dimensional collapse happens too fast
   - System learned surface patterns, not deep judgment
   - Detection: early anisotropy spike, low effective dim

2. PROFESSOR_PLEASING
   - High accuracy on training professors
   - Poor generalization to held-out evaluators
   - Detection: high variance in cross-professor correlation

3. FEATURE_GAMING
   - Text features optimized (hedging words)
   - But logit uncertainty doesn't match
   - Detection: V1 critic detects mismatch

4. GEOMETRIC_INSTABILITY
   - Structure doesn't persist under perturbation
   - Conscience is fragile, not robust
   - Detection: low persistence score

5. SURPRISE_STAGNATION
   - Prediction error doesn't decrease
   - Critic not learning from feedback
   - Detection: flat or increasing surprise trend

Each failure mode has:
- Detection criteria
- Severity (warning vs critical)
- Suggested remediation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

from .metrics import ConscienceMetrics, ConscienceScore


class FailureMode(Enum):
    """Enumeration of conscience formation failure modes."""
    HEURISTIC_COLLAPSE = "heuristic_collapse"
    PROFESSOR_PLEASING = "professor_pleasing"
    FEATURE_GAMING = "feature_gaming"
    GEOMETRIC_INSTABILITY = "geometric_instability"
    SURPRISE_STAGNATION = "surprise_stagnation"
    OVERCONFIDENCE_UNCORRECTED = "overconfidence_uncorrected"
    REVISION_INEFFECTIVE = "revision_ineffective"


@dataclass
class FailureModeDetection:
    """Detection result for a specific failure mode."""
    mode: FailureMode
    detected: bool
    severity: str  # "warning", "critical"
    confidence: float  # 0-1
    evidence: Dict[str, float] = field(default_factory=dict)
    remediation: str = ""


@dataclass
class ValidationResult:
    """Complete validation result for a training run."""
    # Overall verdict
    is_valid: bool
    conscience_present: bool

    # Scores
    conscience_score: ConscienceScore

    # Failure modes
    failures: List[FailureModeDetection] = field(default_factory=list)
    warnings: List[FailureModeDetection] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append(f"Conscience Validation Result")
        lines.append("=" * 40)
        lines.append(f"Valid: {self.is_valid}")
        lines.append(f"Conscience Present: {self.conscience_present}")
        lines.append(f"Confidence: {self.conscience_score.confidence}")
        lines.append(f"Total Score: {self.conscience_score.total_score:.2f}")
        lines.append("")

        if self.failures:
            lines.append("CRITICAL FAILURES:")
            for f in self.failures:
                lines.append(f"  - {f.mode.value}: {f.remediation}")

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w.mode.value}: {w.remediation}")

        if self.recommendations:
            lines.append("RECOMMENDATIONS:")
            for r in self.recommendations:
                lines.append(f"  - {r}")

        return "\n".join(lines)


def detect_heuristic_collapse(
    metrics: ConscienceMetrics,
    threshold_collapse: float = 0.3,
    threshold_early_aniso: float = 5.0,
) -> FailureModeDetection:
    """Detect heuristic collapse (too-fast dimensional reduction)."""
    evidence = {
        "dimensional_collapse": metrics.dimensional_collapse,
        "anisotropy_final": metrics.anisotropy_stability.anisotropy_final,
    }

    # Check for early anisotropy spike
    aniso_values = metrics.anisotropy_stability.anisotropy_values
    early_spike = False
    if len(aniso_values) > 10:
        early_aniso = np.mean(aniso_values[:10])
        late_aniso = np.mean(aniso_values[-10:])
        early_spike = early_aniso > threshold_early_aniso and early_aniso > late_aniso

    evidence["early_aniso_spike"] = float(early_spike)

    # Detection
    detected = (
        metrics.dimensional_collapse < threshold_collapse or
        early_spike
    )

    severity = "critical" if metrics.dimensional_collapse < 0.2 else "warning"

    return FailureModeDetection(
        mode=FailureMode.HEURISTIC_COLLAPSE,
        detected=detected,
        severity=severity,
        confidence=1.0 - metrics.dimensional_collapse if detected else 0.0,
        evidence=evidence,
        remediation="Slow down learning rate; add regularization; increase task diversity",
    )


def detect_professor_pleasing(
    metrics: ConscienceMetrics,
    variance_threshold: float = 0.3,
) -> FailureModeDetection:
    """Detect professor pleasing (no cross-professor generalization)."""
    gen = metrics.generalization

    evidence = {
        "mean_generalization": gen.mean_generalization,
        "min_generalization": gen.min_generalization,
    }

    # Variance across professors
    if gen.holdout_correlations:
        variance = np.std(list(gen.holdout_correlations.values()))
        evidence["correlation_variance"] = float(variance)
    else:
        variance = 0.0

    # Detection: high variance OR low min with acceptable mean
    detected = (
        variance > variance_threshold or
        (gen.min_generalization < 0.3 and gen.mean_generalization > 0.5)
    )

    severity = "critical" if gen.min_generalization < 0.1 else "warning"

    return FailureModeDetection(
        mode=FailureMode.PROFESSOR_PLEASING,
        detected=detected,
        severity=severity,
        confidence=variance if detected else 0.0,
        evidence=evidence,
        remediation="Rotate professors during training; add adversarial evaluator; diversify evaluation criteria",
    )


def detect_feature_gaming(
    hedge_count_vs_entropy: Optional[Tuple[List[float], List[float]]] = None,
    confidence_vs_margin: Optional[Tuple[List[float], List[float]]] = None,
) -> FailureModeDetection:
    """Detect feature gaming (text doesn't match model state)."""
    evidence = {}
    detected = False

    # Check hedge words vs entropy correlation
    if hedge_count_vs_entropy:
        hedges, entropies = hedge_count_vs_entropy
        if len(hedges) > 10 and np.std(hedges) > 0 and np.std(entropies) > 0:
            corr = np.corrcoef(hedges, entropies)[0, 1]
            evidence["hedge_entropy_correlation"] = float(corr)
            # Should be positive: more hedging = higher entropy
            if corr < 0.2:
                detected = True

    # Check confidence vs margin correlation
    if confidence_vs_margin:
        confidences, margins = confidence_vs_margin
        if len(confidences) > 10 and np.std(confidences) > 0 and np.std(margins) > 0:
            corr = np.corrcoef(confidences, margins)[0, 1]
            evidence["confidence_margin_correlation"] = float(corr)
            # Should be positive: higher confidence = higher margin
            if corr < 0.2:
                detected = True

    return FailureModeDetection(
        mode=FailureMode.FEATURE_GAMING,
        detected=detected,
        severity="warning",
        confidence=0.7 if detected else 0.0,
        evidence=evidence,
        remediation="Use V1 critic with logit features; add consistency loss; penalize misaligned confidence",
    )


def detect_geometric_instability(
    metrics: ConscienceMetrics,
    persistence_threshold: float = 0.5,
) -> FailureModeDetection:
    """Detect geometric instability (structure doesn't persist)."""
    evidence = {
        "geometric_persistence": metrics.geometric_persistence,
        "anisotropy_stability": metrics.anisotropy_stability.stability,
    }

    detected = metrics.geometric_persistence < persistence_threshold

    return FailureModeDetection(
        mode=FailureMode.GEOMETRIC_INSTABILITY,
        detected=detected,
        severity="warning",
        confidence=1.0 - metrics.geometric_persistence if detected else 0.0,
        evidence=evidence,
        remediation="Add perturbation training; use dropout; test on out-of-distribution inputs",
    )


def detect_surprise_stagnation(
    metrics: ConscienceMetrics,
) -> FailureModeDetection:
    """Detect surprise stagnation (critic not learning)."""
    ss = metrics.surprise_stability

    evidence = {
        "surprise_trend": ss.surprise_trend,
        "surprise_stability": ss.stability,
        "surprise_mean": ss.surprise_mean,
    }

    # Trend should be negative or near-zero
    detected = ss.surprise_trend > 0.001  # Increasing surprise

    severity = "critical" if ss.surprise_trend > 0.01 else "warning"

    return FailureModeDetection(
        mode=FailureMode.SURPRISE_STAGNATION,
        detected=detected,
        severity=severity,
        confidence=min(1.0, ss.surprise_trend * 100) if detected else 0.0,
        evidence=evidence,
        remediation="Increase critic learning rate; add more features; check for label noise",
    )


def detect_failure_modes(
    metrics: ConscienceMetrics,
    hedge_entropy_data: Optional[Tuple[List[float], List[float]]] = None,
    confidence_margin_data: Optional[Tuple[List[float], List[float]]] = None,
) -> List[FailureModeDetection]:
    """Detect all failure modes from metrics.

    Returns list of detected failure modes (both warnings and critical).
    """
    detections = []

    # Run all detectors
    detectors = [
        detect_heuristic_collapse(metrics),
        detect_professor_pleasing(metrics),
        detect_feature_gaming(hedge_entropy_data, confidence_margin_data),
        detect_geometric_instability(metrics),
        detect_surprise_stagnation(metrics),
    ]

    for detection in detectors:
        if detection.detected:
            detections.append(detection)

    return detections


class ConscienceValidator:
    """Validates conscience formation in trained ASPIRE systems.

    Usage:
        validator = ConscienceValidator()
        result = validator.validate(metrics)
        print(result.summary())
    """

    def __init__(
        self,
        conscience_threshold: float = 0.6,
        allow_warnings: bool = True,
    ):
        self.conscience_threshold = conscience_threshold
        self.allow_warnings = allow_warnings

    def validate(
        self,
        metrics: ConscienceMetrics,
        hedge_entropy_data: Optional[Tuple[List[float], List[float]]] = None,
        confidence_margin_data: Optional[Tuple[List[float], List[float]]] = None,
    ) -> ValidationResult:
        """Validate conscience formation.

        Args:
            metrics: ConscienceMetrics from training
            hedge_entropy_data: Optional (hedges, entropies) for gaming detection
            confidence_margin_data: Optional (confidences, margins) for gaming detection

        Returns:
            ValidationResult with complete assessment
        """
        from .metrics import compute_conscience_score

        # Compute conscience score
        score = compute_conscience_score(metrics, threshold=self.conscience_threshold)

        # Detect failure modes
        all_detections = detect_failure_modes(
            metrics,
            hedge_entropy_data,
            confidence_margin_data,
        )

        # Separate failures and warnings
        failures = [d for d in all_detections if d.severity == "critical"]
        warnings = [d for d in all_detections if d.severity == "warning"]

        # Determine validity
        is_valid = len(failures) == 0
        if not self.allow_warnings:
            is_valid = is_valid and len(warnings) == 0

        conscience_present = score.has_conscience and is_valid

        # Generate recommendations
        recommendations = []

        if not conscience_present:
            if score.surprise_score < 0.5:
                recommendations.append("Improve critic learning: more cycles or higher learning rate")
            if score.generalization_score < 0.5:
                recommendations.append("Improve generalization: diversify professors or add adversarial evaluation")
            if score.anisotropy_score < 0.5:
                recommendations.append("Check dimensional structure: may need longer training or more data")

        for failure in failures:
            recommendations.append(f"Address {failure.mode.value}: {failure.remediation}")

        return ValidationResult(
            is_valid=is_valid,
            conscience_present=conscience_present,
            conscience_score=score,
            failures=failures,
            warnings=warnings,
            recommendations=recommendations,
        )
