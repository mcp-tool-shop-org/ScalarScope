"""Core experiment runner infrastructure.

Provides the base classes and utilities for running ASPIRE falsification
experiments in a reproducible, comparable manner.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
from pathlib import Path
import json
import time
import numpy as np
from datetime import datetime


class Condition(Enum):
    """Experimental conditions for comparison."""
    # Experiment 1 conditions
    FULL_ASPIRE = "full_aspire"
    SCALAR_REWARD = "scalar_reward"
    RANDOM_PROFESSORS = "random_professors"

    # Experiment 2 conditions
    ALL_PROFESSORS = "all_professors"
    HOLDOUT_ONE = "holdout_one"
    SINGLE_PROFESSOR = "single_professor"

    # Experiment 3 conditions
    HONEST_STUDENT = "honest_student"
    ADVERSARIAL_NO_DEFENSE = "adversarial_no_defense"
    ADVERSARIAL_WITH_DEFENSE = "adversarial_with_defense"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    # Experiment identification
    name: str
    description: str
    version: str = "1.0"

    # Training configuration
    n_training_items: int = 100
    n_training_cycles: int = 50
    n_runs_per_condition: int = 5

    # Random seed management
    base_seed: int = 42

    # Output configuration
    output_dir: Path = Path("experiments/results")
    save_trajectories: bool = True
    save_checkpoints: bool = False

    # Metric collection
    collect_every_n_steps: int = 1

    def get_seed(self, run_idx: int, condition: Condition) -> int:
        """Get deterministic seed for a specific run."""
        # Hash condition name to get offset
        condition_offset = hash(condition.value) % 1000
        return self.base_seed + run_idx * 1000 + condition_offset


@dataclass
class TrajectoryPoint:
    """Single point in a training trajectory."""
    step: int
    timestamp: float

    # Core metrics
    surprise: float
    surprise_std: float
    conscience_score: float

    # Geometry metrics
    effective_dimensionality: float
    anisotropy: float
    participation_ratio: float

    # Generalization
    mean_generalization: float
    min_generalization: float

    # Per-professor scores (for detailed analysis)
    professor_scores: Dict[str, float] = field(default_factory=dict)

    # Failure mode flags
    active_warnings: List[str] = field(default_factory=list)

    # Leakage metrics
    style_correlations: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    # Run identification
    condition: Condition
    run_idx: int
    seed: int

    # Timing
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Final metrics
    final_conscience_score: float
    final_surprise_stability: float
    final_generalization_min: float
    final_generalization_mean: float

    # Trajectory (full history)
    trajectory: List[TrajectoryPoint] = field(default_factory=list)

    # Failure mode summary
    failure_modes_detected: List[str] = field(default_factory=list)
    warning_count: int = 0

    # Additional condition-specific data
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "condition": self.condition.value,
            "run_idx": self.run_idx,
            "seed": self.seed,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "final_conscience_score": self.final_conscience_score,
            "final_surprise_stability": self.final_surprise_stability,
            "final_generalization_min": self.final_generalization_min,
            "final_generalization_mean": self.final_generalization_mean,
            "failure_modes_detected": self.failure_modes_detected,
            "warning_count": self.warning_count,
            "trajectory_length": len(self.trajectory),
            "extra_data": self.extra_data,
        }


@dataclass
class ExperimentSummary:
    """Summary statistics across all runs of an experiment."""

    experiment_name: str
    config: ExperimentConfig

    # Results by condition
    results_by_condition: Dict[Condition, List[ExperimentResult]]

    # Aggregate statistics (computed)
    conscience_scores_by_condition: Dict[str, List[float]] = field(default_factory=dict)
    surprise_stability_by_condition: Dict[str, List[float]] = field(default_factory=dict)
    generalization_by_condition: Dict[str, List[float]] = field(default_factory=dict)

    # Falsification checks
    falsification_results: Dict[str, bool] = field(default_factory=dict)
    falsification_details: Dict[str, str] = field(default_factory=dict)

    def compute_statistics(self):
        """Compute aggregate statistics from results."""
        for condition, results in self.results_by_condition.items():
            key = condition.value
            self.conscience_scores_by_condition[key] = [
                r.final_conscience_score for r in results
            ]
            self.surprise_stability_by_condition[key] = [
                r.final_surprise_stability for r in results
            ]
            self.generalization_by_condition[key] = [
                r.final_generalization_min for r in results
            ]

    def get_condition_stats(self, condition: Condition) -> Dict[str, Any]:
        """Get statistics for a specific condition."""
        key = condition.value
        conscience = self.conscience_scores_by_condition.get(key, [])
        stability = self.surprise_stability_by_condition.get(key, [])
        gen = self.generalization_by_condition.get(key, [])

        return {
            "conscience": {
                "mean": np.mean(conscience) if conscience else 0,
                "std": np.std(conscience) if len(conscience) > 1 else 0,
                "min": np.min(conscience) if conscience else 0,
                "max": np.max(conscience) if conscience else 0,
            },
            "surprise_stability": {
                "mean": np.mean(stability) if stability else 0,
                "std": np.std(stability) if len(stability) > 1 else 0,
            },
            "generalization_min": {
                "mean": np.mean(gen) if gen else 0,
                "std": np.std(gen) if len(gen) > 1 else 0,
            },
        }


class ExperimentRunner:
    """Base class for running ASPIRE experiments.

    Subclasses implement specific experiment logic while this class
    handles common infrastructure: seeding, timing, result collection,
    and output management.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: Dict[Condition, List[ExperimentResult]] = {}

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all_conditions(self, conditions: List[Condition]) -> ExperimentSummary:
        """Run all conditions with configured number of runs each."""
        for condition in conditions:
            self.results[condition] = []

            for run_idx in range(self.config.n_runs_per_condition):
                seed = self.config.get_seed(run_idx, condition)
                print(f"Running {condition.value} run {run_idx + 1}/{self.config.n_runs_per_condition} (seed={seed})")

                result = self.run_single(condition, run_idx, seed)
                self.results[condition].append(result)

                # Save intermediate results
                if self.config.save_checkpoints:
                    self._save_checkpoint(condition, run_idx, result)

        # Create and return summary
        summary = ExperimentSummary(
            experiment_name=self.config.name,
            config=self.config,
            results_by_condition=self.results,
        )
        summary.compute_statistics()

        # Run falsification checks
        self.check_falsification(summary)

        # Save final results
        self._save_results(summary)

        return summary

    def run_single(
        self,
        condition: Condition,
        run_idx: int,
        seed: int,
    ) -> ExperimentResult:
        """Run a single experiment. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement run_single")

    def check_falsification(self, summary: ExperimentSummary):
        """Check falsification criteria. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement check_falsification")

    def _save_checkpoint(
        self,
        condition: Condition,
        run_idx: int,
        result: ExperimentResult,
    ):
        """Save intermediate checkpoint."""
        checkpoint_path = (
            self.config.output_dir /
            f"{self.config.name}_{condition.value}_run{run_idx}.json"
        )
        with open(checkpoint_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _save_results(self, summary: ExperimentSummary):
        """Save final experiment results."""
        results_path = self.config.output_dir / f"{self.config.name}_results.json"

        # Convert numpy bools to Python bools for JSON serialization
        falsification_results = {
            k: bool(v) if hasattr(v, 'item') else v
            for k, v in summary.falsification_results.items()
        }

        output = {
            "experiment": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "config": {
                "n_training_items": self.config.n_training_items,
                "n_training_cycles": self.config.n_training_cycles,
                "n_runs_per_condition": self.config.n_runs_per_condition,
                "base_seed": self.config.base_seed,
            },
            "conditions": {},
            "falsification": {
                "results": falsification_results,
                "details": summary.falsification_details,
            },
        }

        for condition, results in summary.results_by_condition.items():
            stats = summary.get_condition_stats(condition)
            output["conditions"][condition.value] = {
                "n_runs": len(results),
                "statistics": stats,
                "runs": [r.to_dict() for r in results],
            }

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {results_path}")


def compute_trajectory_metrics(
    surprises: List[float],
    token_scores: List[Dict[str, float]],
    professor_scores: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Compute metrics from a training trajectory.

    This is a helper function used by experiment implementations.
    """
    from ..conscience.metrics import SurpriseStability, GeneralizationScore
    from ..conscience.calibration import NullDistribution

    # Surprise stability
    surprise_result = SurpriseStability.compute(surprises)

    # Generalization (requires predictions vs actuals)
    # This is a simplified version - full version needs TokenVectors

    # Effective dimensionality from token score distribution
    if token_scores:
        # Compute variance explained by each dimension
        all_dims = list(token_scores[0].keys())
        dim_variances = []
        for dim in all_dims:
            values = [ts.get(dim, 0.5) for ts in token_scores]
            dim_variances.append(np.var(values))

        total_var = sum(dim_variances)
        if total_var > 0:
            # Participation ratio as effective dimensionality
            normalized = [v / total_var for v in dim_variances]
            participation_ratio = 1.0 / sum(p**2 for p in normalized if p > 0)
        else:
            participation_ratio = len(all_dims)
    else:
        participation_ratio = 5.0  # Default for 5 dimensions

    return {
        "surprise_stability": surprise_result.stability,
        "surprise_trend": surprise_result.surprise_trend,
        "surprise_mean": surprise_result.surprise_mean,
        "surprise_std": surprise_result.surprise_std,
        "participation_ratio": participation_ratio,
    }
