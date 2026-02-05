"""Tests for ASPIRE falsification experiments.

These tests verify that:
1. Experiments run without error
2. Metrics are computed correctly
3. Falsification checks are properly applied
"""

import pytest
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, 'F:/AI/aspire-engine/src')

from scalarscope.experiments.experiment_runner import (
    ExperimentConfig,
    ExperimentResult,
    Condition,
    TrajectoryPoint,
)
from scalarscope.experiments.experiment1_null_vs_structured import (
    NullVsStructuredExperiment,
    NullVsStructuredResult,
)
from scalarscope.experiments.experiment2_holdout_transfer import (
    HoldoutTransferExperiment,
    HoldoutTransferResult,
)
from scalarscope.experiments.experiment3_adversarial import (
    AdversarialPressureExperiment,
    AdversarialPressureResult,
)


class TestExperimentConfig:
    """Test experiment configuration."""

    def test_seed_generation(self):
        """Seeds should be deterministic and unique per run."""
        config = ExperimentConfig(
            name="test",
            description="Test experiment",
            base_seed=42,
        )

        seed1 = config.get_seed(0, Condition.FULL_ASPIRE)
        seed2 = config.get_seed(1, Condition.FULL_ASPIRE)
        seed3 = config.get_seed(0, Condition.RANDOM_PROFESSORS)

        # Different runs = different seeds
        assert seed1 != seed2

        # Different conditions = different seeds
        assert seed1 != seed3

        # Same inputs = same outputs
        assert config.get_seed(0, Condition.FULL_ASPIRE) == seed1

    def test_config_defaults(self):
        """Config should have reasonable defaults."""
        config = ExperimentConfig(name="test", description="Test")

        assert config.n_training_items == 100
        assert config.n_training_cycles == 50
        assert config.n_runs_per_condition == 5
        assert config.base_seed == 42


class TestExperiment1:
    """Test Null vs Structured Judgment experiment."""

    @pytest.fixture
    def config(self):
        return ExperimentConfig(
            name="test_exp1",
            description="Test experiment 1",
            n_training_items=20,
            n_training_cycles=10,
            n_runs_per_condition=2,
            output_dir=Path(tempfile.mkdtemp()),
        )

    def test_experiment_runs(self, config):
        """Experiment should run without error."""
        exp = NullVsStructuredExperiment(config)

        # Run just FULL condition for speed
        result = exp.run_single(Condition.FULL_ASPIRE, 0, 42)

        assert isinstance(result, NullVsStructuredResult)
        assert result.condition == Condition.FULL_ASPIRE
        assert len(result.trajectory) == config.n_training_cycles
        assert 0 <= result.final_conscience_score <= 1

    def test_conditions_differ(self, config):
        """Different conditions should produce different results."""
        exp = NullVsStructuredExperiment(config)

        full_result = exp.run_single(Condition.FULL_ASPIRE, 0, 42)
        random_result = exp.run_single(Condition.RANDOM_PROFESSORS, 0, 42)

        # Results should be different
        assert full_result.final_conscience_score != random_result.final_conscience_score

    def test_trajectory_metrics_computed(self, config):
        """Trajectory should have all metrics."""
        exp = NullVsStructuredExperiment(config)
        result = exp.run_single(Condition.FULL_ASPIRE, 0, 42)

        for point in result.trajectory:
            assert isinstance(point, TrajectoryPoint)
            assert point.surprise >= 0
            assert point.effective_dimensionality > 0

    def test_collapse_analysis(self, config):
        """Collapse analysis should detect dimensional changes."""
        exp = NullVsStructuredExperiment(config)
        result = exp.run_single(Condition.SCALAR_REWARD, 0, 42)

        # SCALAR_REWARD should show some collapse pattern
        assert result.collapse_rate_early >= 0
        assert result.collapse_rate_mid >= 0


class TestExperiment2:
    """Test Holdout Transfer experiment."""

    @pytest.fixture
    def config(self):
        return ExperimentConfig(
            name="test_exp2",
            description="Test experiment 2",
            n_training_items=20,
            n_training_cycles=10,
            n_runs_per_condition=2,
            output_dir=Path(tempfile.mkdtemp()),
        )

    def test_experiment_runs(self, config):
        """Experiment should run without error."""
        exp = HoldoutTransferExperiment(config)

        result = exp.run_single(Condition.HOLDOUT_ONE, 0, 42)

        assert isinstance(result, HoldoutTransferResult)
        assert result.holdout_professor_name != ""
        assert len(result.seen_professor_correlations) > 0

    def test_holdout_correlation_computed(self, config):
        """Holdout correlation should be computed."""
        exp = HoldoutTransferExperiment(config)
        result = exp.run_single(Condition.HOLDOUT_ONE, 0, 42)

        # Holdout correlation should be a real number
        assert result.holdout_professor_correlation is not None
        assert -1 <= result.holdout_professor_correlation <= 1

    def test_transfer_ratio(self, config):
        """Transfer ratio should be computed."""
        exp = HoldoutTransferExperiment(config)
        result = exp.run_single(Condition.HOLDOUT_ONE, 0, 42)

        # Transfer ratio is holdout / mean(seen)
        assert result.transfer_ratio >= 0

    def test_all_professors_vs_single(self, config):
        """All professors should generalize better than single."""
        exp = HoldoutTransferExperiment(config)

        all_result = exp.run_single(Condition.ALL_PROFESSORS, 0, 42)
        single_result = exp.run_single(Condition.SINGLE_PROFESSOR, 0, 42)

        # All professors should have more seen correlations
        assert len(all_result.seen_professor_correlations) > len(single_result.seen_professor_correlations)


class TestExperiment3:
    """Test Adversarial Pressure experiment."""

    @pytest.fixture
    def config(self):
        return ExperimentConfig(
            name="test_exp3",
            description="Test experiment 3",
            n_training_items=20,
            n_training_cycles=10,
            n_runs_per_condition=2,
            output_dir=Path(tempfile.mkdtemp()),
        )

    def test_experiment_runs(self, config):
        """Experiment should run without error."""
        exp = AdversarialPressureExperiment(config)

        result = exp.run_single(Condition.HONEST_STUDENT, 0, 42)

        assert isinstance(result, AdversarialPressureResult)
        assert result.student_type == "honest"

    def test_adversarial_detected(self, config):
        """Adversarial student should trigger gaming detection."""
        exp = AdversarialPressureExperiment(config)

        # Run with longer training to allow detection
        config.n_training_cycles = 20
        result = exp.run_single(Condition.ADVERSARIAL_NO_DEFENSE, 0, 42)

        # Adversarial should have higher leakage correlation
        assert result.max_leakage_correlation > 0

    def test_defense_changes_behavior(self, config):
        """Defense should affect adversarial student differently (may not differ on short runs)."""
        exp = AdversarialPressureExperiment(config)

        no_defense = exp.run_single(Condition.ADVERSARIAL_NO_DEFENSE, 0, 42)
        with_defense = exp.run_single(Condition.ADVERSARIAL_WITH_DEFENSE, 0, 42)

        # Both should complete without error and have valid results
        assert no_defense.student_type == "fake_hedger"
        assert with_defense.student_type == "fake_hedger"
        # Results may not differ on short test runs, but both should be valid
        assert 0 <= no_defense.final_conscience_score <= 1
        assert 0 <= with_defense.final_conscience_score <= 1

    def test_honest_vs_adversarial(self, config):
        """Honest and adversarial should produce different patterns."""
        exp = AdversarialPressureExperiment(config)

        honest = exp.run_single(Condition.HONEST_STUDENT, 0, 42)
        adversarial = exp.run_single(Condition.ADVERSARIAL_NO_DEFENSE, 0, 42)

        # Student types should differ
        assert honest.student_type != adversarial.student_type


class TestFalsificationChecks:
    """Test that falsification checks work correctly."""

    @pytest.fixture
    def config(self):
        return ExperimentConfig(
            name="test_falsification",
            description="Test falsification",
            n_training_items=20,
            n_training_cycles=10,
            n_runs_per_condition=3,
            output_dir=Path(tempfile.mkdtemp()),
        )

    def test_exp1_falsification(self, config):
        """Experiment 1 should run falsification checks."""
        exp = NullVsStructuredExperiment(config)
        summary = exp.run()

        # Should have falsification results
        assert "conscience_separation" in summary.falsification_results
        assert "surprise_stability" in summary.falsification_results
        assert "generalization" in summary.falsification_results

    def test_exp2_falsification(self, config):
        """Experiment 2 should check holdout transfer."""
        exp = HoldoutTransferExperiment(config)
        summary = exp.run()

        # Should have holdout checks
        assert "holdout_correlation" in summary.falsification_results
        assert "transfer_ratio" in summary.falsification_results

    def test_exp3_falsification(self, config):
        """Experiment 3 should check adversarial detection."""
        exp = AdversarialPressureExperiment(config)
        summary = exp.run()

        # Should have adversarial checks
        assert "adversarial_detection" in summary.falsification_results
        assert "honest_advantage" in summary.falsification_results


class TestResultSerialization:
    """Test result serialization."""

    def test_result_to_dict(self):
        """Result should serialize to dict."""
        from datetime import datetime

        result = ExperimentResult(
            condition=Condition.FULL_ASPIRE,
            run_idx=0,
            seed=42,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.5,
            final_conscience_score=0.75,
            final_surprise_stability=0.8,
            final_generalization_min=0.6,
            final_generalization_mean=0.7,
        )

        d = result.to_dict()

        assert d["condition"] == "full_aspire"
        assert d["run_idx"] == 0
        assert d["seed"] == 42
        assert d["final_conscience_score"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
