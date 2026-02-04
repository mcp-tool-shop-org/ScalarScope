"""Test suite for conscience formation validation.

Tests the conscience module's ability to:
1. Compute conscience metrics correctly
2. Detect failure modes
3. Identify feature leakage
4. Run ablation studies

These tests use synthetic data to verify the detection mechanisms work.
"""

import pytest
import numpy as np
from typing import List, Dict

# Add project root to path for imports
import sys
sys.path.insert(0, 'F:/AI/aspire-engine/src')

from aspire.conscience.metrics import (
    SurpriseStability,
    AnisotropyStability,
    GeneralizationScore,
    ConscienceMetrics,
    ConscienceScore,
    compute_conscience_score,
)
from aspire.conscience.validation import (
    FailureMode,
    detect_heuristic_collapse,
    detect_professor_pleasing,
    detect_surprise_stagnation,
    detect_geometric_instability,
    ConscienceValidator,
)
from aspire.conscience.leakage import (
    detect_early_collapse,
    detect_velocity_anomaly,
    detect_curvature_anomaly,
    detect_text_token_shortcut,
    FeatureLeakageDetector,
)
from aspire.conscience.ablation import (
    AblationType,
    AblationConfig,
    AblationRunner,
    compare_ablations,
)
from aspire.geometry.state import StateVector, StateSnapshot
from aspire.geometry.trajectory import TrainingTrajectory


class TestSurpriseStability:
    """Test surprise stability metric computation."""

    def test_compute_stable_surprises(self):
        """Stable surprises should give high stability score."""
        # Very stable surprises (low variance)
        surprises = [0.1 + np.random.normal(0, 0.01) for _ in range(100)]
        result = SurpriseStability.compute(surprises)

        assert result.stability > 0.8
        assert result.surprise_mean < 0.2
        assert result.surprise_std < 0.05

    def test_compute_unstable_surprises(self):
        """Unstable surprises should give low stability score."""
        # High variance surprises
        surprises = [0.5 + np.random.normal(0, 0.3) for _ in range(100)]
        result = SurpriseStability.compute(surprises)

        assert result.stability < 0.5
        assert result.surprise_std > 0.1

    def test_compute_decreasing_trend(self):
        """Decreasing surprise trend should be detected."""
        # Surprises that decrease over time (good learning)
        surprises = [0.5 - 0.004 * i + np.random.normal(0, 0.02) for i in range(100)]
        result = SurpriseStability.compute(surprises)

        assert result.surprise_trend < 0  # Negative trend = decreasing

    def test_compute_empty_input(self):
        """Empty input should return default values."""
        result = SurpriseStability.compute([])
        assert result.stability == 0.0
        assert result.surprise_mean == 0.0

    def test_compute_single_value(self):
        """Single value should return default."""
        result = SurpriseStability.compute([0.5])
        assert result.stability == 0.0


class TestAnisotropyStability:
    """Test anisotropy stability computation."""

    def test_requires_trajectory(self):
        """Anisotropy needs trajectory with sufficient data."""
        trajectory = TrainingTrajectory()
        result = AnisotropyStability.compute(trajectory)

        # Not enough data
        assert result.stability == 0.0
        assert len(result.anisotropy_values) == 0


class TestGeneralizationScore:
    """Test cross-professor generalization scoring."""

    def test_high_generalization(self):
        """High correlation across professors = high generalization."""
        from aspire.core import TokenVector, TokenDimension

        # Predictions that correlate well with all professors
        n_samples = 50
        predictions = [
            TokenVector({d: 0.3 + 0.4 * i / n_samples for d in TokenDimension})
            for i in range(n_samples)
        ]

        # All professors agree (high generalization)
        actuals = {
            "prof_a": [
                TokenVector({d: 0.3 + 0.4 * i / n_samples + np.random.normal(0, 0.05)
                            for d in TokenDimension})
                for i in range(n_samples)
            ],
            "prof_b": [
                TokenVector({d: 0.3 + 0.4 * i / n_samples + np.random.normal(0, 0.05)
                            for d in TokenDimension})
                for i in range(n_samples)
            ],
        }

        result = GeneralizationScore.compute(predictions, actuals)

        assert result.mean_generalization > 0.7
        assert result.min_generalization > 0.5

    def test_low_generalization(self):
        """Low correlation with one professor = low min generalization."""
        from aspire.core import TokenVector, TokenDimension

        n_samples = 50
        # Predictions that have variance
        predictions = [
            TokenVector({d: 0.3 + 0.4 * i / n_samples for d in TokenDimension})
            for i in range(n_samples)
        ]

        # One professor agrees well, one has low correlation
        actuals = {
            "prof_good": [
                TokenVector({d: 0.3 + 0.4 * i / n_samples + np.random.normal(0, 0.05)
                            for d in TokenDimension})
                for i in range(n_samples)
            ],
            "prof_bad": [
                TokenVector({d: 0.7 - 0.4 * i / n_samples + np.random.normal(0, 0.1)
                            for d in TokenDimension})  # Inverted correlation
                for i in range(n_samples)
            ],
        }

        result = GeneralizationScore.compute(predictions, actuals)

        # Prof_good should have high positive correlation
        # Prof_bad should have low or negative correlation
        # So min should be less than mean
        assert result.holdout_correlations["prof_good"] > result.holdout_correlations["prof_bad"]


class TestConscienceScore:
    """Test overall conscience score computation."""

    def test_high_conscience_score(self):
        """Good metrics should produce high conscience score."""
        metrics = ConscienceMetrics(
            surprise_stability=SurpriseStability(stability=0.9),
            anisotropy_stability=AnisotropyStability(stability=0.8, growth_rate=0.01),
            generalization=GeneralizationScore(
                mean_generalization=0.85,
                min_generalization=0.75,
            ),
            dimensional_collapse=0.6,
            geometric_persistence=0.7,
        )

        score = compute_conscience_score(metrics)

        assert score.total_score > 0.6
        assert score.has_conscience
        assert score.confidence in ["moderate", "strong"]
        assert len(score.warnings) == 0

    def test_low_conscience_score(self):
        """Poor metrics should produce low conscience score."""
        metrics = ConscienceMetrics(
            surprise_stability=SurpriseStability(stability=0.2),
            anisotropy_stability=AnisotropyStability(stability=0.3, growth_rate=-0.01),
            generalization=GeneralizationScore(
                mean_generalization=0.4,
                min_generalization=0.2,
            ),
            dimensional_collapse=0.1,
            geometric_persistence=0.2,
        )

        score = compute_conscience_score(metrics)

        assert score.total_score < 0.6
        assert not score.has_conscience
        assert score.confidence in ["none", "weak"]

    def test_warnings_generated(self):
        """Warnings should be generated for problematic patterns."""
        metrics = ConscienceMetrics(
            surprise_stability=SurpriseStability(
                stability=0.8,
                surprise_trend=0.05,  # Increasing - bad!
            ),
            anisotropy_stability=AnisotropyStability(
                stability=0.7,
                growth_rate=-0.02,  # Decreasing - bad!
            ),
            generalization=GeneralizationScore(
                mean_generalization=0.7,
                min_generalization=0.2,  # Low min with acceptable mean = pleasing
            ),
            dimensional_collapse=0.2,  # Too fast!
        )

        score = compute_conscience_score(metrics)

        # Should have warnings
        assert len(score.warnings) > 0
        # Warnings should downgrade confidence
        assert not score.has_conscience or score.confidence != "strong"


class TestFailureModeDetection:
    """Test failure mode detection functions."""

    def test_detect_heuristic_collapse(self):
        """Should detect when collapse happens too fast."""
        metrics = ConscienceMetrics(
            dimensional_collapse=0.2,  # Very fast collapse
            anisotropy_stability=AnisotropyStability(
                anisotropy_values=[5.0] * 10 + [4.5] * 90,  # Early spike
                anisotropy_final=4.5,
            ),
        )

        result = detect_heuristic_collapse(metrics)

        assert result.detected
        assert result.severity in ["warning", "critical"]

    def test_no_heuristic_collapse(self):
        """Should not flag gradual, healthy collapse."""
        metrics = ConscienceMetrics(
            dimensional_collapse=0.6,  # Moderate collapse
            anisotropy_stability=AnisotropyStability(
                anisotropy_values=[1.0 + 0.03 * i for i in range(100)],  # Gradual
                anisotropy_final=4.0,
            ),
        )

        result = detect_heuristic_collapse(metrics)

        assert not result.detected

    def test_detect_professor_pleasing(self):
        """Should detect when generalization varies too much by professor."""
        metrics = ConscienceMetrics(
            generalization=GeneralizationScore(
                holdout_correlations={
                    "prof_a": 0.9,
                    "prof_b": 0.85,
                    "prof_c": 0.2,  # Much worse on this one
                },
                mean_generalization=0.65,
                min_generalization=0.2,
            ),
        )

        result = detect_professor_pleasing(metrics)

        assert result.detected

    def test_detect_surprise_stagnation(self):
        """Should detect when surprise is increasing."""
        metrics = ConscienceMetrics(
            surprise_stability=SurpriseStability(
                surprise_trend=0.02,  # Increasing
                surprise_mean=0.5,
                stability=0.6,
            ),
        )

        result = detect_surprise_stagnation(metrics)

        assert result.detected


class TestConscienceValidator:
    """Test the ConscienceValidator class."""

    def test_validate_good_training(self):
        """Good training should pass validation."""
        metrics = ConscienceMetrics(
            surprise_stability=SurpriseStability(stability=0.85, surprise_trend=-0.01),
            anisotropy_stability=AnisotropyStability(stability=0.8, growth_rate=0.01),
            generalization=GeneralizationScore(
                mean_generalization=0.8,
                min_generalization=0.7,
                holdout_correlations={"a": 0.8, "b": 0.7},
            ),
            dimensional_collapse=0.5,
            geometric_persistence=0.7,
            total_cycles=500,
            final_accuracy=0.85,
        )

        validator = ConscienceValidator()
        result = validator.validate(metrics)

        assert result.is_valid
        assert result.conscience_present
        assert len(result.failures) == 0

    def test_validate_bad_training(self):
        """Bad training should fail validation."""
        metrics = ConscienceMetrics(
            surprise_stability=SurpriseStability(stability=0.3, surprise_trend=0.05),
            anisotropy_stability=AnisotropyStability(stability=0.3, growth_rate=-0.02),
            generalization=GeneralizationScore(
                mean_generalization=0.3,
                min_generalization=0.1,
            ),
            dimensional_collapse=0.1,
            geometric_persistence=0.2,
        )

        validator = ConscienceValidator()
        result = validator.validate(metrics)

        assert not result.conscience_present
        # Should have recommendations
        assert len(result.recommendations) > 0


class TestLeakageDetection:
    """Test feature leakage detection."""

    def test_detect_text_token_shortcut(self):
        """Should detect when text features predict tokens too well."""
        # Create correlated data
        n_samples = 100

        # Feature that predicts token perfectly (leakage!)
        text_features = [
            {"hedge_count": i / n_samples, "length": 0.5}
            for i in range(n_samples)
        ]
        token_scores = [
            {"calibration": i / n_samples + np.random.normal(0, 0.05), "coherence": 0.5}
            for i in range(n_samples)
        ]

        result = detect_text_token_shortcut(text_features, token_scores)

        assert result is not None
        assert result.indicator_type == "text_token_shortcut"
        assert "hedge_count->calibration" in result.evidence

    def test_no_shortcut_when_uncorrelated(self):
        """Should not flag uncorrelated features."""
        n_samples = 100

        text_features = [
            {"hedge_count": np.random.random(), "length": np.random.random()}
            for _ in range(n_samples)
        ]
        token_scores = [
            {"calibration": np.random.random(), "coherence": np.random.random()}
            for _ in range(n_samples)
        ]

        result = detect_text_token_shortcut(text_features, token_scores)

        # Should not detect shortcut (random correlation unlikely to be high)
        # Note: there's a small chance of false positive due to randomness
        assert result is None or result.confidence < 0.7


class TestAblationFramework:
    """Test the ablation study framework."""

    def test_ablation_config_creation(self):
        """Should create ablation configs correctly."""
        config = AblationConfig(AblationType.NO_CRITIC)

        assert config.ablation_type == AblationType.NO_CRITIC
        assert config.name == "no_critic"
        assert "critic" in config.description.lower()

    def test_standard_ablations(self):
        """Should create all standard ablations."""
        runner = AblationRunner()
        ablations = runner.create_standard_ablations()

        assert len(ablations) == 7  # Full + 6 ablations
        types = {a.ablation_type for a in ablations}
        assert AblationType.FULL in types
        assert AblationType.NO_CRITIC in types
        assert AblationType.RANDOM_PROFESSORS in types

    def test_compare_ablations(self):
        """Should compare ablation results against baseline."""
        from aspire.conscience.ablation import AblationResult, AblationComparison

        # Create mock results
        baseline_metrics = ConscienceMetrics(
            total_cycles=100,
            final_accuracy=0.85,
        )
        baseline_score = ConscienceScore(
            total_score=0.8,
            has_conscience=True,
        )
        baseline = AblationResult(
            config=AblationConfig(AblationType.FULL),
            metrics=baseline_metrics,
            score=baseline_score,
            final_accuracy=0.85,
        )

        # Ablated version (worse)
        ablated_metrics = ConscienceMetrics(
            total_cycles=100,
            final_accuracy=0.6,
        )
        ablated_score = ConscienceScore(
            total_score=0.4,
            has_conscience=False,
        )
        ablated = AblationResult(
            config=AblationConfig(AblationType.NO_CRITIC),
            metrics=ablated_metrics,
            score=ablated_score,
            final_accuracy=0.6,
        )

        comparison = compare_ablations(baseline, [ablated])

        assert comparison.baseline == baseline
        assert len(comparison.ablations) == 1
        # NO_CRITIC should have negative contribution (removing it hurt)
        assert comparison.contributions["no_critic"] > 0

    def test_comparison_summary(self):
        """Should generate readable comparison summary."""
        from aspire.conscience.ablation import AblationResult, AblationComparison

        baseline = AblationResult(
            config=AblationConfig(AblationType.FULL),
            metrics=ConscienceMetrics(),
            score=ConscienceScore(total_score=0.8, has_conscience=True),
            final_accuracy=0.85,
        )

        comparison = AblationComparison(
            baseline=baseline,
            ablations=[],
            contributions={"no_critic": 0.3},
        )

        summary = comparison.summary()

        assert "BASELINE" in summary
        assert "0.85" in summary or "85" in summary


class TestAdversarialProfessors:
    """Test adversarial professor mechanisms."""

    def test_rotating_selector(self):
        """Should rotate through professors."""
        from aspire.professors import (
            StrictLogician,
            PragmaticEngineer,
            EmpathyAdvocate,
        )
        from aspire.professors.adversarial import RotatingSelector, RotationConfig

        professors = [StrictLogician(), PragmaticEngineer(), EmpathyAdvocate()]
        selector = RotatingSelector(
            professors,
            RotationConfig(rotation_size=2, cooldown_cycles=2),
        )

        # Get active professors for several cycles
        selections = []
        for i in range(10):
            active = selector.get_active(i)
            selections.append([p.professor_id for p in active])

        # Should have rotated (not always the same)
        assert len(set(tuple(s) for s in selections)) > 1

        # Usage should be balanced
        usage = selector.get_usage_stats()
        counts = list(usage.values())
        assert max(counts) - min(counts) <= 2  # Reasonably balanced

    def test_adversarial_professor_observation(self):
        """Should learn patterns from observations."""
        from aspire.professors.adversarial import AdversarialProfessor
        from aspire.core import StudentResponse, TrainingItem, TokenVector, TokenDimension, ProfessorCritique

        adversary = AdversarialProfessor()

        # Create observations with a pattern
        for i in range(60):
            # Pattern: hedge words correlate with high tokens
            has_hedge = i % 2 == 0
            response = StudentResponse(
                item_id=f"item_{i}",
                answer="test",
                reasoning_trace="might perhaps maybe" if has_hedge else "definitely certainly",
                confidence=0.7,
            )

            # High tokens when hedging (the pattern)
            tokens = TokenVector({
                d: 0.8 if has_hedge else 0.3 for d in TokenDimension
            })
            critique = ProfessorCritique(
                professor_id="test",
                tokens=tokens,
                is_correct=True,
                critique_text="Test critique",
            )

            adversary.observe(response, [critique])

        # Should have learned the pattern
        report = adversary.get_gaming_report()
        assert "hedge_density" in report

    def test_holdout_manager(self):
        """Should correctly separate training and holdout professors."""
        from aspire.professors import (
            StrictLogician,
            PragmaticEngineer,
            EmpathyAdvocate,
        )
        from aspire.professors.adversarial import HoldoutManager, HoldoutConfig

        professors = [StrictLogician(), PragmaticEngineer(), EmpathyAdvocate()]
        manager = HoldoutManager(
            professors,
            HoldoutConfig(holdout_ids=["empathy_advocate"]),
        )

        training = manager.get_training_professors()
        holdout = manager.get_holdout_professors()

        assert len(training) == 2
        assert len(holdout) == 1
        assert holdout[0].professor_id == "empathy_advocate"


# Integration test
class TestConscienceIntegration:
    """Integration tests for the complete conscience validation pipeline."""

    def test_full_validation_pipeline(self):
        """Test the full pipeline from metrics to validation."""
        # Create realistic-looking metrics
        metrics = ConscienceMetrics(
            surprise_stability=SurpriseStability(
                surprise_mean=0.15,
                surprise_std=0.05,
                surprise_trend=-0.005,
                stability=0.7,
            ),
            anisotropy_stability=AnisotropyStability(
                anisotropy_values=list(np.linspace(1.0, 3.0, 50)),
                anisotropy_mean=2.0,
                anisotropy_std=0.5,
                anisotropy_final=3.0,
                stability=0.75,
                growth_rate=0.04,
            ),
            generalization=GeneralizationScore(
                holdout_correlations={
                    "strict_logician": 0.75,
                    "pragmatic_engineer": 0.72,
                    "empathy_advocate": 0.68,
                },
                mean_generalization=0.72,
                min_generalization=0.68,
            ),
            dimensional_collapse=0.55,
            geometric_persistence=0.65,
            total_cycles=500,
            final_accuracy=0.82,
        )

        # Run validation
        validator = ConscienceValidator(conscience_threshold=0.6)
        result = validator.validate(metrics)

        # Should pass
        assert result.is_valid
        assert result.conscience_present
        assert result.conscience_score.confidence in ["moderate", "strong"]

        # Summary should be readable
        summary = result.summary()
        assert "Conscience" in summary
        assert "Valid" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
