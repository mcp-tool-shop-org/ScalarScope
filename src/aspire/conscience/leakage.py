"""Feature leakage detection via geometry analysis.

WHAT IS FEATURE LEAKAGE IN ASPIRE?

Feature leakage occurs when the model's success comes from exploiting
unintended correlations rather than genuine learning. In ASPIRE, this
manifests as:

1. TEXT-TO-TOKEN SHORTCUTS
   - Student learns specific text patterns that trigger high tokens
   - Example: Adding "tradeoff" gets high TRADEOFFS score without analysis
   - Detection: Text features correlate too strongly with specific dimensions

2. HEDGING WITHOUT UNCERTAINTY
   - Model uses hedge words ("might", "perhaps") to boost CALIBRATION
   - But internal model uncertainty (entropy) doesn't match
   - Detection: V1 critic shows hedge_count/entropy decorrelation

3. PROFESSOR FINGERPRINTING
   - Model identifies which professor will evaluate and optimizes for them
   - Tokens vary wildly based on detected evaluator
   - Detection: Token variance spikes with evaluator-predictive features

4. GEOMETRIC SHORTCUTS
   - Learning collapses to low-dimensional space too quickly
   - Suggests surface heuristics rather than deep understanding
   - Detection: Early anisotropy spike, high initial velocity in state space

HOW GEOMETRY REVEALS LEAKAGE

The key insight: genuine learning shows GRADUAL dimensional specialization.
Shortcut learning shows IMMEDIATE collapse to task-aligned dimensions.

METRICS:
- Velocity profile: genuine learning has high initial velocity, tapering off
- Curvature profile: genuine learning has curvature peaks at phase transitions
- Dimensional evolution: genuine learning shows gradual collapse
- Feature-dimension correlation: should be distributed, not concentrated

This module provides tools to detect and diagnose feature leakage.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..geometry import TrainingTrajectory, StateVector
from ..core import TokenDimension


@dataclass
class LeakageIndicator:
    """Single indicator of potential feature leakage."""
    indicator_type: str
    severity: str  # "low", "medium", "high"
    confidence: float
    evidence: Dict[str, float]
    description: str
    recommendation: str


@dataclass
class LeakageReport:
    """Complete feature leakage analysis report."""
    # Overall assessment
    leakage_detected: bool
    leakage_severity: str  # "none", "low", "medium", "high"
    leakage_confidence: float

    # Individual indicators
    indicators: List[LeakageIndicator] = field(default_factory=list)

    # Geometric evidence
    velocity_profile: Optional[np.ndarray] = None
    curvature_profile: Optional[np.ndarray] = None
    dimensional_evolution: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append("FEATURE LEAKAGE ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Leakage Detected: {self.leakage_detected}")
        lines.append(f"Severity: {self.leakage_severity}")
        lines.append(f"Confidence: {self.leakage_confidence:.1%}")
        lines.append("")

        if self.indicators:
            lines.append("INDICATORS:")
            for ind in self.indicators:
                severity_icon = {"low": "⚠️", "medium": "🟠", "high": "🔴"}.get(ind.severity, "")
                lines.append(f"\n  {severity_icon} {ind.indicator_type} ({ind.severity})")
                lines.append(f"    {ind.description}")
                lines.append(f"    Evidence: {ind.evidence}")
                lines.append(f"    → {ind.recommendation}")
        else:
            lines.append("No leakage indicators detected.")

        return "\n".join(lines)


def detect_early_collapse(
    trajectory: TrainingTrajectory,
    collapse_threshold: float = 0.5,
    early_fraction: float = 0.2,
) -> Optional[LeakageIndicator]:
    """Detect early dimensional collapse (shortcut learning).

    Genuine learning: dimensionality reduces gradually over training
    Shortcut learning: dimensionality collapses quickly then plateaus

    Args:
        trajectory: Training trajectory to analyze
        collapse_threshold: Threshold for "significant" collapse
        early_fraction: What fraction of training to consider "early"

    Returns:
        LeakageIndicator if early collapse detected, else None
    """
    if len(trajectory.snapshots) < 50:
        return None

    from ..geometry.state import compute_effective_dimensionality

    # Compute dimensional evolution
    states = [s.state for s in trajectory.snapshots]
    window = 20

    dims = []
    for i in range(window, len(states) + 1):
        window_states = states[i - window:i]
        ed = compute_effective_dimensionality(window_states)
        dims.append(ed)

    if len(dims) < 10:
        return None

    dims = np.array(dims)

    # Normalize
    initial_dim = dims[0]
    if initial_dim < 1e-6:
        return None

    normalized = dims / initial_dim

    # Check for early collapse
    early_end = int(len(normalized) * early_fraction)
    early_collapse = 1.0 - normalized[early_end]

    # Check for late stability (plateau)
    late_start = int(len(normalized) * 0.7)
    late_variance = np.std(normalized[late_start:])

    # Early collapse + late plateau = shortcut learning
    if early_collapse > collapse_threshold and late_variance < 0.1:
        return LeakageIndicator(
            indicator_type="early_collapse",
            severity="high" if early_collapse > 0.7 else "medium",
            confidence=min(1.0, early_collapse * 1.5),
            evidence={
                "early_collapse": float(early_collapse),
                "late_variance": float(late_variance),
                "initial_dim": float(initial_dim),
                "final_dim": float(dims[-1]),
            },
            description=(
                f"Dimensionality collapsed by {early_collapse:.0%} in first "
                f"{early_fraction:.0%} of training, then plateaued. "
                "This pattern suggests shortcut learning."
            ),
            recommendation=(
                "Slow learning rate early in training. "
                "Add regularization. "
                "Increase task diversity."
            ),
        )

    return None


def detect_velocity_anomaly(
    trajectory: TrainingTrajectory,
) -> Optional[LeakageIndicator]:
    """Detect anomalous velocity profile in state space.

    Genuine learning: velocity starts high (exploration), then decreases
    Shortcut learning: velocity is consistently high OR consistently low

    Args:
        trajectory: Training trajectory to analyze

    Returns:
        LeakageIndicator if velocity anomaly detected, else None
    """
    step_sizes = trajectory.compute_step_sizes()
    if len(step_sizes) < 30:
        return None

    # Compute early vs late velocity
    n = len(step_sizes)
    early = step_sizes[:n // 3]
    mid = step_sizes[n // 3:2 * n // 3]
    late = step_sizes[2 * n // 3:]

    early_mean = np.mean(early)
    mid_mean = np.mean(mid)
    late_mean = np.mean(late)

    # Expected pattern: early > mid > late (with some tolerance)
    # Anomaly 1: flat velocity (no exploration/exploitation tradeoff)
    total_mean = np.mean(step_sizes)
    cv = np.std([early_mean, mid_mean, late_mean]) / total_mean if total_mean > 0 else 0

    if cv < 0.1:
        return LeakageIndicator(
            indicator_type="flat_velocity",
            severity="medium",
            confidence=1.0 - cv * 10,
            evidence={
                "early_velocity": float(early_mean),
                "mid_velocity": float(mid_mean),
                "late_velocity": float(late_mean),
                "coefficient_of_variation": float(cv),
            },
            description=(
                "Velocity through state space is nearly constant. "
                "Expected pattern: high early (exploration), low late (convergence)."
            ),
            recommendation=(
                "Check learning rate schedule. "
                "Model may be stuck in local minimum or memorizing."
            ),
        )

    # Anomaly 2: velocity increases late (unstable learning)
    if late_mean > mid_mean * 1.5:
        return LeakageIndicator(
            indicator_type="late_velocity_spike",
            severity="high",
            confidence=min(1.0, late_mean / mid_mean - 1),
            evidence={
                "mid_velocity": float(mid_mean),
                "late_velocity": float(late_mean),
                "ratio": float(late_mean / mid_mean),
            },
            description=(
                "Velocity increases late in training. "
                "This suggests unstable learning or catastrophic forgetting."
            ),
            recommendation=(
                "Reduce learning rate. "
                "Add gradient clipping. "
                "Check for distribution shift in training data."
            ),
        )

    return None


def detect_curvature_anomaly(
    trajectory: TrainingTrajectory,
    expected_transitions: int = 2,
) -> Optional[LeakageIndicator]:
    """Detect anomalous curvature profile.

    Genuine learning: curvature peaks at phase transitions (grokking moments)
    Shortcut learning: no clear phase transitions OR too many erratic peaks

    Args:
        trajectory: Training trajectory to analyze
        expected_transitions: Minimum expected phase transitions

    Returns:
        LeakageIndicator if curvature anomaly detected, else None
    """
    curvatures = trajectory.compute_curvature()
    if len(curvatures) < 20:
        return None

    # Find peaks (phase transitions)
    threshold = np.pi / 4  # 45 degrees
    peaks = trajectory.find_phase_transitions(threshold=threshold)

    # Anomaly 1: no phase transitions (too smooth)
    if len(peaks) < expected_transitions:
        return LeakageIndicator(
            indicator_type="no_phase_transitions",
            severity="low",
            confidence=0.5,
            evidence={
                "peak_count": len(peaks),
                "expected_min": expected_transitions,
                "max_curvature": float(np.max(curvatures)) if len(curvatures) > 0 else 0,
            },
            description=(
                f"Only {len(peaks)} phase transitions detected. "
                "Genuine learning typically shows clear phase transitions."
            ),
            recommendation=(
                "May not be an issue, but check if learning is making progress. "
                "Consider longer training or curriculum learning."
            ),
        )

    # Anomaly 2: too many erratic peaks (unstable)
    if len(peaks) > len(curvatures) * 0.2:  # More than 20% are peaks
        return LeakageIndicator(
            indicator_type="erratic_curvature",
            severity="medium",
            confidence=min(1.0, len(peaks) / len(curvatures)),
            evidence={
                "peak_count": len(peaks),
                "total_points": len(curvatures),
                "peak_fraction": len(peaks) / len(curvatures),
            },
            description=(
                f"{len(peaks)} curvature peaks detected ({len(peaks) / len(curvatures):.0%}). "
                "This suggests erratic, unstable learning."
            ),
            recommendation=(
                "Reduce learning rate. "
                "Increase batch size. "
                "Add momentum or use adaptive optimizer."
            ),
        )

    return None


def detect_dimension_correlation(
    trajectory: TrainingTrajectory,
    feature_correlations: Optional[Dict[str, Dict[str, float]]] = None,
    concentration_threshold: float = 0.7,
) -> Optional[LeakageIndicator]:
    """Detect concentrated feature-to-dimension correlations.

    Genuine learning: features distributed across dimensions
    Shortcut learning: single features dominate single dimensions

    Args:
        trajectory: Training trajectory
        feature_correlations: Optional pre-computed correlations
        concentration_threshold: Above this = leakage

    Returns:
        LeakageIndicator if correlation concentration detected
    """
    # This requires external feature tracking - return if not provided
    if feature_correlations is None:
        return None

    # Find maximum correlation for each dimension
    max_correlations = {}
    dominant_features = {}

    for dim_name, feature_corrs in feature_correlations.items():
        if not feature_corrs:
            continue
        max_feature = max(feature_corrs.keys(), key=lambda f: abs(feature_corrs[f]))
        max_corr = abs(feature_corrs[max_feature])
        max_correlations[dim_name] = max_corr
        dominant_features[dim_name] = max_feature

    # Check for concentrated correlations
    high_concentration = {
        dim: corr for dim, corr in max_correlations.items()
        if corr > concentration_threshold
    }

    if high_concentration:
        return LeakageIndicator(
            indicator_type="feature_concentration",
            severity="high" if len(high_concentration) >= 2 else "medium",
            confidence=np.mean(list(high_concentration.values())),
            evidence={
                **{f"{dim}_max_corr": corr for dim, corr in high_concentration.items()},
                **{f"{dim}_dominant": dominant_features[dim] for dim in high_concentration},
            },
            description=(
                f"High feature concentration in {len(high_concentration)} dimensions. "
                "Single features dominate token dimensions, suggesting shortcuts."
            ),
            recommendation=(
                "Add feature decorrelation loss. "
                "Rotate features during training. "
                "Add adversarial professor to penalize patterns."
            ),
        )

    return None


def detect_text_token_shortcut(
    text_features: List[Dict[str, float]],
    token_scores: List[Dict[str, float]],
    shortcut_threshold: float = 0.6,
) -> Optional[LeakageIndicator]:
    """Detect text-to-token shortcuts (marker stuffing).

    Checks if simple text features predict token scores too well.

    Args:
        text_features: List of text feature dicts per sample
        token_scores: List of token score dicts per sample
        shortcut_threshold: Correlation above this indicates shortcut

    Returns:
        LeakageIndicator if text shortcut detected
    """
    if len(text_features) < 30 or len(token_scores) < 30:
        return None

    # Get feature and token names
    feature_names = list(text_features[0].keys())
    token_names = list(token_scores[0].keys())

    # Build matrices
    X = np.array([[f[fn] for fn in feature_names] for f in text_features])
    Y = np.array([[t[tn] for tn in token_names] for t in token_scores])

    # Compute correlation matrix
    shortcuts = {}
    for i, fname in enumerate(feature_names):
        for j, tname in enumerate(token_names):
            if np.std(X[:, i]) > 1e-6 and np.std(Y[:, j]) > 1e-6:
                corr = np.corrcoef(X[:, i], Y[:, j])[0, 1]
                if abs(corr) > shortcut_threshold:
                    shortcuts[f"{fname}->{tname}"] = float(corr)

    if shortcuts:
        return LeakageIndicator(
            indicator_type="text_token_shortcut",
            severity="high" if len(shortcuts) >= 3 else "medium",
            confidence=np.mean([abs(c) for c in shortcuts.values()]),
            evidence=shortcuts,
            description=(
                f"Found {len(shortcuts)} text-to-token shortcuts. "
                "Simple text features predict token scores too well."
            ),
            recommendation=(
                "Randomize text feature extraction. "
                "Use adversarial professor. "
                "Add noise to text features during training."
            ),
        )

    return None


class FeatureLeakageDetector:
    """Comprehensive feature leakage detection.

    Combines all leakage detection methods into a single analysis.

    Usage:
        detector = FeatureLeakageDetector()

        # After training
        report = detector.analyze(
            trajectory=training_trajectory,
            text_features=collected_text_features,
            token_scores=collected_token_scores,
        )

        print(report.summary())
    """

    def __init__(
        self,
        early_collapse_threshold: float = 0.5,
        shortcut_threshold: float = 0.6,
    ):
        self.early_collapse_threshold = early_collapse_threshold
        self.shortcut_threshold = shortcut_threshold

    def analyze(
        self,
        trajectory: TrainingTrajectory,
        text_features: Optional[List[Dict[str, float]]] = None,
        token_scores: Optional[List[Dict[str, float]]] = None,
        feature_correlations: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> LeakageReport:
        """Perform complete leakage analysis.

        Args:
            trajectory: Training trajectory
            text_features: Optional text features per sample
            token_scores: Optional token scores per sample
            feature_correlations: Optional pre-computed feature-dimension correlations

        Returns:
            LeakageReport with all findings
        """
        indicators = []

        # Geometric indicators
        early_collapse = detect_early_collapse(
            trajectory, self.early_collapse_threshold
        )
        if early_collapse:
            indicators.append(early_collapse)

        velocity = detect_velocity_anomaly(trajectory)
        if velocity:
            indicators.append(velocity)

        curvature = detect_curvature_anomaly(trajectory)
        if curvature:
            indicators.append(curvature)

        # Feature-based indicators
        if feature_correlations:
            dim_corr = detect_dimension_correlation(
                trajectory, feature_correlations
            )
            if dim_corr:
                indicators.append(dim_corr)

        if text_features and token_scores:
            shortcut = detect_text_token_shortcut(
                text_features, token_scores, self.shortcut_threshold
            )
            if shortcut:
                indicators.append(shortcut)

        # Compute overall severity
        if not indicators:
            severity = "none"
            confidence = 0.0
            detected = False
        else:
            severity_scores = {"low": 1, "medium": 2, "high": 3}
            max_severity = max(indicators, key=lambda i: severity_scores[i.severity])
            severity = max_severity.severity
            confidence = np.mean([i.confidence for i in indicators])
            detected = True

        # Collect geometric evidence
        velocity_profile = trajectory.compute_step_sizes() if trajectory.snapshots else None
        curvature_profile = trajectory.compute_curvature() if trajectory.snapshots else None

        return LeakageReport(
            leakage_detected=detected,
            leakage_severity=severity,
            leakage_confidence=float(confidence),
            indicators=indicators,
            velocity_profile=velocity_profile,
            curvature_profile=curvature_profile,
        )
