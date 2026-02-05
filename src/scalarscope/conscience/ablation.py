"""Ablation framework for ASPIRE conscience validation.

Ablation studies are critical for validating that each ASPIRE component
contributes to conscience formation. This module provides:

1. Ablation configurations (what to disable)
2. Ablation runner (runs training with ablations)
3. Comparison tools (measures impact)

STANDARD ABLATIONS:

1. NO_CRITIC: Remove critic prediction entirely
   - Expected: No surprise signal, no predictive alignment
   - If conscience still forms: critic is not essential

2. SCALAR_REWARD: Replace 5D tokens with scalar loss
   - Expected: Dimensional collapse, loss of nuance
   - If conscience still forms: token dimensions are redundant

3. NO_REVISION: Disable revision pass
   - Expected: No self-correction learning
   - If conscience still forms: revision is not essential

4. SINGLE_PROFESSOR: Use only one professor
   - Expected: Professor-pleasing, no generalization
   - If conscience still forms: ensemble is redundant

5. NO_LOGIT_FEATURES: Use V0 critic (text only)
   - Expected: Feature gaming possible
   - If conscience still forms: logit features are not essential

6. RANDOM_PROFESSORS: Professors give random evaluations
   - Expected: No learning signal
   - This is the NULL condition - conscience should NOT form
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import numpy as np

from .metrics import ConscienceMetrics, ConscienceScore, compute_conscience_score


class AblationType(Enum):
    """Types of ablations to test."""
    FULL = "full"  # No ablation (baseline)
    NO_CRITIC = "no_critic"
    SCALAR_REWARD = "scalar_reward"
    NO_REVISION = "no_revision"
    SINGLE_PROFESSOR = "single_professor"
    NO_LOGIT_FEATURES = "no_logit_features"
    RANDOM_PROFESSORS = "random_professors"  # Null condition


@dataclass
class AblationConfig:
    """Configuration for an ablation run."""
    ablation_type: AblationType
    name: str = ""
    description: str = ""

    # Specific settings per ablation
    settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.ablation_type.value

        # Set descriptions
        descriptions = {
            AblationType.FULL: "Full ASPIRE system (baseline)",
            AblationType.NO_CRITIC: "Remove critic prediction - no surprise signal",
            AblationType.SCALAR_REWARD: "Replace token vector with scalar loss",
            AblationType.NO_REVISION: "Disable revision pass - no self-correction",
            AblationType.SINGLE_PROFESSOR: "Use only one professor - no ensemble",
            AblationType.NO_LOGIT_FEATURES: "Use V0 critic - text features only",
            AblationType.RANDOM_PROFESSORS: "Random evaluation - NULL condition",
        }
        if not self.description:
            self.description = descriptions.get(self.ablation_type, "")


@dataclass
class AblationResult:
    """Result from a single ablation run."""
    config: AblationConfig
    metrics: ConscienceMetrics
    score: ConscienceScore

    # Training outcomes
    final_accuracy: float = 0.0
    total_cycles: int = 0
    training_time_ms: float = 0.0

    # Comparison to baseline (if available)
    delta_score: Optional[float] = None
    delta_accuracy: Optional[float] = None


@dataclass
class AblationComparison:
    """Comparison of ablation results against baseline."""
    baseline: AblationResult
    ablations: List[AblationResult]

    # Component contributions
    contributions: Dict[str, float] = field(default_factory=dict)

    # Statistical significance (if multiple runs)
    significance: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = []
        lines.append("ABLATION STUDY RESULTS")
        lines.append("=" * 60)
        lines.append("")

        # Baseline
        lines.append(f"BASELINE ({self.baseline.config.name}):")
        lines.append(f"  Conscience Score: {self.baseline.score.total_score:.3f}")
        lines.append(f"  Accuracy: {self.baseline.final_accuracy:.1%}")
        lines.append(f"  Conscience Present: {self.baseline.score.has_conscience}")
        lines.append("")

        # Ablations
        lines.append("ABLATIONS:")
        for result in self.ablations:
            delta_score = result.score.total_score - self.baseline.score.total_score
            delta_acc = result.final_accuracy - self.baseline.final_accuracy

            lines.append(f"\n  {result.config.name}:")
            lines.append(f"    {result.config.description}")
            lines.append(f"    Score: {result.score.total_score:.3f} ({delta_score:+.3f})")
            lines.append(f"    Accuracy: {result.final_accuracy:.1%} ({delta_acc:+.1%})")
            lines.append(f"    Conscience: {result.score.has_conscience}")

        # Contributions
        if self.contributions:
            lines.append("\nCOMPONENT CONTRIBUTIONS (score impact when removed):")
            sorted_contribs = sorted(
                self.contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for name, impact in sorted_contribs:
                lines.append(f"  {name:20} {impact:+.3f}")

        # Key findings
        lines.append("\nKEY FINDINGS:")

        # Which ablation hurt most?
        if self.ablations:
            worst = min(self.ablations, key=lambda x: x.score.total_score)
            lines.append(f"  Most impactful: {worst.config.name}")
            lines.append(f"    (removing this drops score by {self.baseline.score.total_score - worst.score.total_score:.3f})")

        # Did null condition work?
        null_results = [r for r in self.ablations if r.config.ablation_type == AblationType.RANDOM_PROFESSORS]
        if null_results:
            null = null_results[0]
            if null.score.has_conscience:
                lines.append("  WARNING: NULL condition showed conscience - validation failure!")
            else:
                lines.append("  NULL condition (random professors) correctly shows no conscience")

        return "\n".join(lines)


def compare_ablations(
    baseline: AblationResult,
    ablations: List[AblationResult],
) -> AblationComparison:
    """Compare ablation results against baseline.

    Args:
        baseline: Result with full ASPIRE system
        ablations: Results with various components removed

    Returns:
        AblationComparison with analysis
    """
    contributions = {}

    for result in ablations:
        # Contribution = how much score drops when this is removed
        contribution = baseline.score.total_score - result.score.total_score
        contributions[result.config.name] = contribution

        # Update result with delta
        result.delta_score = -contribution
        result.delta_accuracy = result.final_accuracy - baseline.final_accuracy

    return AblationComparison(
        baseline=baseline,
        ablations=ablations,
        contributions=contributions,
    )


class AblationRunner:
    """Runs ablation studies on ASPIRE training.

    Usage:
        runner = AblationRunner()

        # Define ablations to test
        configs = [
            AblationConfig(AblationType.FULL),
            AblationConfig(AblationType.NO_CRITIC),
            AblationConfig(AblationType.NO_REVISION),
        ]

        # Run (requires training function)
        results = runner.run(configs, train_fn, items)

        # Compare
        comparison = runner.compare(results)
        print(comparison.summary())
    """

    def __init__(self, n_runs: int = 1, random_seed: int = 42):
        """Initialize runner.

        Args:
            n_runs: Number of runs per ablation (for statistical significance)
            random_seed: Base random seed
        """
        self.n_runs = n_runs
        self.random_seed = random_seed

    def create_standard_ablations(self) -> List[AblationConfig]:
        """Create the standard set of ablations."""
        return [
            AblationConfig(AblationType.FULL),
            AblationConfig(AblationType.NO_CRITIC),
            AblationConfig(AblationType.SCALAR_REWARD),
            AblationConfig(AblationType.NO_REVISION),
            AblationConfig(AblationType.SINGLE_PROFESSOR),
            AblationConfig(AblationType.NO_LOGIT_FEATURES),
            AblationConfig(AblationType.RANDOM_PROFESSORS),
        ]

    def run_single(
        self,
        config: AblationConfig,
        train_fn: Callable[[AblationConfig, int], ConscienceMetrics],
        seed: int,
    ) -> AblationResult:
        """Run a single ablation.

        Args:
            config: Ablation configuration
            train_fn: Function that trains and returns metrics
                      Signature: (config, seed) -> ConscienceMetrics
            seed: Random seed for this run

        Returns:
            AblationResult
        """
        metrics = train_fn(config, seed)
        score = compute_conscience_score(metrics)

        return AblationResult(
            config=config,
            metrics=metrics,
            score=score,
            final_accuracy=metrics.final_accuracy,
            total_cycles=metrics.total_cycles,
        )

    def run(
        self,
        configs: List[AblationConfig],
        train_fn: Callable[[AblationConfig, int], ConscienceMetrics],
    ) -> List[AblationResult]:
        """Run all ablations.

        Args:
            configs: List of ablation configurations
            train_fn: Training function

        Returns:
            List of AblationResult
        """
        results = []

        for config in configs:
            # Run multiple times if n_runs > 1
            run_metrics = []
            for i in range(self.n_runs):
                seed = self.random_seed + i
                result = self.run_single(config, train_fn, seed)
                run_metrics.append(result)

            # If multiple runs, average the metrics
            if self.n_runs > 1:
                # For now, just take the first (TODO: proper averaging)
                results.append(run_metrics[0])
            else:
                results.append(run_metrics[0])

        return results

    def compare(self, results: List[AblationResult]) -> AblationComparison:
        """Compare ablation results.

        Args:
            results: List of results (first should be FULL baseline)

        Returns:
            AblationComparison
        """
        # Find baseline
        baseline = None
        ablations = []

        for r in results:
            if r.config.ablation_type == AblationType.FULL:
                baseline = r
            else:
                ablations.append(r)

        if baseline is None:
            # Use first as baseline
            baseline = results[0]
            ablations = results[1:]

        return compare_ablations(baseline, ablations)
