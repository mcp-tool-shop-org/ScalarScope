"""Experiment 2: Holdout Judgment Transfer.

The internalization test for ASPIRE. This experiment answers:

    "Is the student learning how to judge, or just who trained it?"

Setup:
    - ALL_PROFESSORS: Train with all professors (baseline)
    - HOLDOUT_ONE: Train without one professor, test transfer
    - SINGLE_PROFESSOR: Train with only one professor (control)

Falsification criteria (theory is WRONG if):
    1. Holdout correlation ≈ 0 (or negative)
    2. Critic confidence collapses on unseen professor
    3. Geometry looks "healthy" but generalization fails

Expected if theory is correct:
    - Holdout professor correlation significantly > 0
    - Transfer quality ~70-80% of seen-professor correlations
    - Brief surprise spike on holdout introduction, then stabilization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import random

from .experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    ExperimentSummary,
    TrajectoryPoint,
    Condition,
)
from ..core import TrainingItem, StudentResponse, TokenVector, TokenDimension


@dataclass
class HoldoutTransferResult(ExperimentResult):
    """Extended result with holdout-specific data."""

    # Professor correlations
    seen_professor_correlations: Dict[str, float] = field(default_factory=dict)
    holdout_professor_correlation: float = 0.0
    holdout_professor_name: str = ""

    # Transfer metrics
    transfer_ratio: float = 0.0  # holdout_corr / mean(seen_corr)

    # Adaptation dynamics
    surprise_before_holdout: float = 0.0
    surprise_at_holdout: float = 0.0
    surprise_after_adaptation: float = 0.0
    adaptation_steps: int = 0


class HoldoutTransferExperiment(ExperimentRunner):
    """Experiment 2: Holdout Judgment Transfer."""

    CONDITIONS = [
        Condition.ALL_PROFESSORS,
        Condition.HOLDOUT_ONE,
        Condition.SINGLE_PROFESSOR,
    ]

    # Define professors with distinct evaluation criteria
    PROFESSOR_CONFIGS = {
        "accuracy_prof": {
            "focus": "accuracy",
            "weight_correct": 0.8,
            "weight_format": 0.2,
        },
        "clarity_prof": {
            "focus": "clarity",
            "weight_structure": 0.5,
            "weight_concise": 0.3,
            "weight_jargon_free": 0.2,
        },
        "calibration_prof": {
            "focus": "calibration",
            "weight_confidence": 0.4,
            "weight_hedging": 0.4,
            "weight_consistency": 0.2,
        },
        "empathy_prof": {  # This will be the holdout
            "focus": "empathy",
            "weight_stakeholder": 0.4,
            "weight_harm": 0.3,
            "weight_benefit": 0.3,
        },
    }

    HOLDOUT_PROFESSOR = "empathy_prof"

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self._test_items = self._create_test_items()

    def _create_test_items(self) -> List[TrainingItem]:
        """Create test items that require different evaluation perspectives."""
        items = []

        # Items that test different professor dimensions
        prompts = [
            # Accuracy-focused
            ("What is 15% of 200?", "30", "15% of 200 = 0.15 × 200 = 30"),
            ("Name the largest planet", "Jupiter", "Jupiter is the largest planet"),

            # Clarity-focused
            ("Explain recursion simply", "A function that calls itself",
             "Recursion is when a function calls itself to solve smaller subproblems"),

            # Calibration-focused
            ("Will AI surpass human intelligence?", "Uncertain, depends on definitions",
             "This involves complex predictions and definitional questions"),

            # Empathy-focused (holdout professor territory)
            ("Should we automate customer service?", "Consider worker impact",
             "Must balance efficiency with effects on employees and customers"),
            ("How to handle employee layoffs?", "With dignity and support",
             "Prioritize respectful treatment and transition assistance"),
        ]

        item_id = 0
        # Repeat to get enough items
        for _ in range(self.config.n_training_items // len(prompts) + 1):
            for prompt, answer, rationale in prompts:
                if item_id >= self.config.n_training_items:
                    break
                items.append(TrainingItem(
                    id=f"exp2_{item_id}",
                    prompt=prompt,
                    gold_answer=answer,
                    gold_rationale=rationale,
                    domain="mixed",
                ))
                item_id += 1

        return items[:self.config.n_training_items]

    def run_single(
        self,
        condition: Condition,
        run_idx: int,
        seed: int,
    ) -> HoldoutTransferResult:
        """Run a single experiment."""
        np.random.seed(seed)
        random.seed(seed)

        start_time = datetime.now()

        # Determine which professors to use based on condition
        active_professors = self._get_active_professors(condition)
        holdout_name = "" if condition != Condition.HOLDOUT_ONE else self.HOLDOUT_PROFESSOR

        # Track predictions and actual evaluations for correlation
        critic_predictions: Dict[str, List[float]] = {p: [] for p in self.PROFESSOR_CONFIGS}
        actual_evaluations: Dict[str, List[float]] = {p: [] for p in self.PROFESSOR_CONFIGS}

        trajectory = []
        surprises = []

        # Phase 1: Training without holdout
        for cycle in range(self.config.n_training_cycles):
            cycle_surprises = []

            for item in self._test_items:
                # Generate response
                response = self._generate_response(item, seed + cycle)

                # Evaluate with active professors only
                for prof_name in active_professors:
                    actual_score = self._evaluate(prof_name, item, response)
                    actual_evaluations[prof_name].append(actual_score)

                    # Simulate critic prediction (learns over time)
                    predicted = self._critic_predict(
                        prof_name, item, response, cycle, seed
                    )
                    critic_predictions[prof_name].append(predicted)

                    # Compute surprise
                    surprise = abs(predicted - actual_score)
                    cycle_surprises.append(surprise)

            mean_surprise = np.mean(cycle_surprises)
            surprises.append(mean_surprise)

            # Record trajectory point
            trajectory.append(TrajectoryPoint(
                step=cycle,
                timestamp=cycle,
                surprise=mean_surprise,
                surprise_std=np.std(cycle_surprises),
                conscience_score=0.0,  # Computed at end
                effective_dimensionality=self._compute_dimensionality(
                    actual_evaluations, active_professors, cycle
                ),
                anisotropy=0.0,
                participation_ratio=len(active_professors),
                mean_generalization=0.0,
                min_generalization=0.0,
            ))

        # Phase 2: Test on holdout professor (if applicable)
        surprise_before = surprises[-1] if surprises else 0
        surprise_at_holdout = 0.0
        surprise_after = 0.0
        adaptation_steps = 0

        if condition == Condition.HOLDOUT_ONE:
            # Now evaluate with holdout professor
            holdout_surprises = []
            for item in self._test_items[:20]:  # Test subset
                response = self._generate_response(item, seed + 1000)

                actual = self._evaluate(holdout_name, item, response)
                actual_evaluations[holdout_name].append(actual)

                predicted = self._critic_predict(
                    holdout_name, item, response,
                    self.config.n_training_cycles, seed
                )
                critic_predictions[holdout_name].append(predicted)

                holdout_surprises.append(abs(predicted - actual))

            surprise_at_holdout = np.mean(holdout_surprises[:5])
            surprise_after = np.mean(holdout_surprises[-5:])
            adaptation_steps = len(holdout_surprises)

        end_time = datetime.now()

        # Compute correlations
        seen_correlations = {}
        for prof_name in active_professors:
            if len(critic_predictions[prof_name]) > 10:
                corr = np.corrcoef(
                    critic_predictions[prof_name],
                    actual_evaluations[prof_name]
                )[0, 1]
                seen_correlations[prof_name] = corr if not np.isnan(corr) else 0.0

        holdout_corr = 0.0
        if holdout_name and len(critic_predictions[holdout_name]) > 5:
            corr = np.corrcoef(
                critic_predictions[holdout_name],
                actual_evaluations[holdout_name]
            )[0, 1]
            holdout_corr = corr if not np.isnan(corr) else 0.0

        # Compute transfer ratio
        mean_seen = np.mean(list(seen_correlations.values())) if seen_correlations else 0
        transfer_ratio = holdout_corr / mean_seen if mean_seen > 0.1 else 0.0

        # Final metrics
        final_conscience = self._compute_conscience_score(
            seen_correlations, holdout_corr, condition
        )

        from ..conscience.metrics import SurpriseStability
        stability = SurpriseStability.compute(surprises)

        return HoldoutTransferResult(
            condition=condition,
            run_idx=run_idx,
            seed=seed,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            final_conscience_score=final_conscience,
            final_surprise_stability=stability.stability,
            final_generalization_min=holdout_corr if holdout_name else min(seen_correlations.values(), default=0),
            final_generalization_mean=mean_seen,
            trajectory=trajectory,
            failure_modes_detected=[],
            warning_count=0,
            seen_professor_correlations=seen_correlations,
            holdout_professor_correlation=holdout_corr,
            holdout_professor_name=holdout_name,
            transfer_ratio=transfer_ratio,
            surprise_before_holdout=surprise_before,
            surprise_at_holdout=surprise_at_holdout,
            surprise_after_adaptation=surprise_after,
            adaptation_steps=adaptation_steps,
        )

    def _get_active_professors(self, condition: Condition) -> List[str]:
        """Get list of active professors for condition."""
        all_profs = list(self.PROFESSOR_CONFIGS.keys())

        if condition == Condition.ALL_PROFESSORS:
            return all_profs
        elif condition == Condition.HOLDOUT_ONE:
            return [p for p in all_profs if p != self.HOLDOUT_PROFESSOR]
        elif condition == Condition.SINGLE_PROFESSOR:
            return [all_profs[0]]  # Just accuracy_prof
        return all_profs

    def _evaluate(
        self,
        professor_name: str,
        item: TrainingItem,
        response: StudentResponse,
    ) -> float:
        """Evaluate response according to professor's criteria."""
        config = self.PROFESSOR_CONFIGS[professor_name]
        focus = config["focus"]

        if focus == "accuracy":
            if item.gold_answer.lower() in response.answer.lower():
                return 0.8 + random.uniform(0, 0.2)
            return 0.2 + random.uniform(0, 0.2)

        elif focus == "clarity":
            score = 0.5
            text = response.reasoning_trace
            if len(text) > 20 and len(text) < 300:
                score += 0.2
            if any(w in text.lower() for w in ["because", "therefore"]):
                score += 0.2
            return min(1.0, score + random.uniform(-0.1, 0.1))

        elif focus == "calibration":
            hedging = any(w in response.reasoning_trace.lower()
                         for w in ["uncertain", "might", "possibly"])
            confident = response.confidence > 0.7

            if hedging and not confident:
                return 0.8 + random.uniform(0, 0.2)
            elif not hedging and confident:
                return 0.6 + random.uniform(-0.1, 0.1)
            return 0.4 + random.uniform(0, 0.2)

        elif focus == "empathy":
            text = response.reasoning_trace.lower()
            empathy_words = ["stakeholder", "impact", "consider", "affect", "people"]
            score = 0.3
            for word in empathy_words:
                if word in text:
                    score += 0.15
            return min(1.0, score + random.uniform(-0.1, 0.1))

        return 0.5

    def _critic_predict(
        self,
        professor_name: str,
        item: TrainingItem,
        response: StudentResponse,
        training_step: int,
        seed: int,
    ) -> float:
        """Simulate critic prediction that improves over training."""
        rng = np.random.default_rng(seed + training_step)

        # Ground truth (what critic is learning to predict)
        true_score = self._evaluate(professor_name, item, response)

        # Noise decreases with training (learning)
        learning_progress = min(1.0, training_step / 30)
        noise_scale = 0.4 * (1 - learning_progress) + 0.1

        # Prediction = truth + decaying noise
        prediction = true_score + rng.normal(0, noise_scale)

        return np.clip(prediction, 0, 1)

    def _generate_response(
        self,
        item: TrainingItem,
        seed: int,
    ) -> StudentResponse:
        """Generate simulated response."""
        rng = random.Random(seed)

        reasoning = f"Analyzing: {item.prompt[:30]}... "

        # Add some empathy-relevant content sometimes
        if rng.random() < 0.5:
            reasoning += rng.choice([
                "Considering the stakeholders involved, ",
                "Taking into account the impact on people, ",
                "Evaluating the tradeoffs, ",
            ])

        reasoning += rng.choice([
            "the answer requires careful analysis.",
            "we can determine a clear path forward.",
            "multiple factors must be considered.",
        ])

        if rng.random() < 0.6:
            answer = item.gold_answer
            confidence = rng.uniform(0.6, 0.9)
        else:
            answer = "Needs more context"
            confidence = rng.uniform(0.3, 0.5)

        return StudentResponse(
            item_id=item.id,
            answer=answer,
            reasoning_trace=reasoning,
            confidence=confidence,
        )

    def _compute_dimensionality(
        self,
        evaluations: Dict[str, List[float]],
        active_profs: List[str],
        current_step: int,
    ) -> float:
        """Compute effective dimensionality from evaluations."""
        if current_step < 5:
            return float(len(active_profs))

        variances = []
        for prof in active_profs:
            if len(evaluations[prof]) > 10:
                variances.append(np.var(evaluations[prof][-20:]))

        if not variances or sum(variances) == 0:
            return float(len(active_profs))

        # Participation ratio
        total = sum(variances)
        normalized = [v / total for v in variances]
        pr = 1.0 / sum(p**2 for p in normalized if p > 0)

        return pr

    def _compute_conscience_score(
        self,
        seen_correlations: Dict[str, float],
        holdout_corr: float,
        condition: Condition,
    ) -> float:
        """Compute conscience score emphasizing transfer."""
        if not seen_correlations:
            return 0.0

        mean_seen = np.mean(list(seen_correlations.values()))
        min_seen = min(seen_correlations.values())

        # Base score from seen professors
        score = 0.3 * mean_seen + 0.2 * min_seen

        # Bonus for holdout transfer (the key test)
        if condition == Condition.HOLDOUT_ONE and holdout_corr > 0.2:
            transfer_bonus = 0.3 * holdout_corr
            score += transfer_bonus

        # Penalty for single professor (no generalization)
        if condition == Condition.SINGLE_PROFESSOR:
            score *= 0.7

        return min(1.0, score)

    def check_falsification(self, summary: ExperimentSummary):
        """Check holdout transfer falsification criteria."""
        summary.compute_statistics()

        # Get holdout results
        holdout_results = summary.results_by_condition.get(Condition.HOLDOUT_ONE, [])

        if holdout_results:
            holdout_corrs = [r.holdout_professor_correlation for r in holdout_results
                           if isinstance(r, HoldoutTransferResult)]
            transfer_ratios = [r.transfer_ratio for r in holdout_results
                             if isinstance(r, HoldoutTransferResult)]

            # Falsification check 1: Holdout correlation must be significantly > 0
            mean_holdout = np.mean(holdout_corrs) if holdout_corrs else 0
            holdout_valid = mean_holdout > 0.2
            summary.falsification_results["holdout_correlation"] = holdout_valid
            summary.falsification_details["holdout_correlation"] = (
                f"Mean holdout correlation = {mean_holdout:.3f} "
                f"(threshold = 0.2)"
            )

            if not holdout_valid:
                summary.falsification_details["holdout_correlation"] += (
                    " *** THEORY FALSIFIED: No holdout transfer ***"
                )

            # Falsification check 2: Transfer ratio should be meaningful
            mean_transfer = np.mean(transfer_ratios) if transfer_ratios else 0
            transfer_valid = mean_transfer > 0.5
            summary.falsification_results["transfer_ratio"] = transfer_valid
            summary.falsification_details["transfer_ratio"] = (
                f"Mean transfer ratio = {mean_transfer:.3f} "
                f"(threshold = 0.5)"
            )

            if not transfer_valid:
                summary.falsification_details["transfer_ratio"] += (
                    " *** WARNING: Low transfer ratio suggests memorization ***"
                )

        # Compare all-professors vs single-professor
        all_prof_results = summary.results_by_condition.get(Condition.ALL_PROFESSORS, [])
        single_results = summary.results_by_condition.get(Condition.SINGLE_PROFESSOR, [])

        if all_prof_results and single_results:
            all_gen = np.mean([r.final_generalization_min for r in all_prof_results])
            single_gen = np.mean([r.final_generalization_min for r in single_results])

            gen_valid = all_gen > single_gen + 0.1
            summary.falsification_results["ensemble_benefit"] = gen_valid
            summary.falsification_details["ensemble_benefit"] = (
                f"All-prof generalization = {all_gen:.3f}, "
                f"Single-prof = {single_gen:.3f}"
            )

    def run(self) -> ExperimentSummary:
        """Run the full experiment."""
        return self.run_all_conditions(self.CONDITIONS)
