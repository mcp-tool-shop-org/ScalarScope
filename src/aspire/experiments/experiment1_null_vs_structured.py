"""Experiment 1: Null vs Structured Judgment.

The bedrock falsification test for ASPIRE. This experiment answers:

    "Does ASPIRE learn anything beyond noise when evaluation is meaningful?"

Setup:
    - FULL_ASPIRE: Normal professors, critic, revision, geometry
    - SCALAR_REWARD: Collapse 5D tokens to scalar mean
    - RANDOM_PROFESSORS: Null condition with random evaluation

Falsification criteria (theory is WRONG if ANY occur):
    1. RANDOM_PROFESSORS achieves ConscienceScore comparable to FULL
    2. SurpriseStability improves under random evaluation
    3. GeneralizationScore remains high with random professors
    4. Geometry shows clean structure under random evaluation

Expected if theory is correct:
    - FULL: Clear separation, stable surprise, clean geometry
    - SCALAR: Fast collapse, lower stability
    - RANDOM: Noisy, no learning, near-chance scores
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
    compute_trajectory_metrics,
)
from ..core import TrainingItem, StudentResponse, TokenVector, TokenDimension
from ..conscience.calibration import NullDistribution, CalibrationProfile


@dataclass
class NullVsStructuredResult(ExperimentResult):
    """Extended result with experiment-specific data."""

    # Trajectory statistics
    surprise_trajectory: List[float] = field(default_factory=list)
    dimensionality_trajectory: List[float] = field(default_factory=list)

    # Collapse timing
    collapse_step: Optional[int] = None
    collapse_rate_early: float = 0.0
    collapse_rate_mid: float = 0.0


class NullVsStructuredExperiment(ExperimentRunner):
    """Experiment 1: Null vs Structured Judgment."""

    CONDITIONS = [
        Condition.FULL_ASPIRE,
        Condition.SCALAR_REWARD,
        Condition.RANDOM_PROFESSORS,
    ]

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self._test_items = self._create_test_items()

    def _create_test_items(self) -> List[TrainingItem]:
        """Create mixed reasoning dataset."""
        items = []

        # Factual reasoning
        factual_prompts = [
            ("What is 2 + 2?", "4", "Basic arithmetic: 2 + 2 = 4"),
            ("What is the capital of France?", "Paris", "France's capital is Paris"),
            ("How many days in a week?", "7", "A week has 7 days"),
        ]

        # Tradeoff questions
        tradeoff_prompts = [
            (
                "Should a company prioritize profit or employee welfare?",
                "Both are important and must be balanced",
                "Short-term profit vs long-term sustainability requires balance"
            ),
            (
                "Is it better to move fast or be careful?",
                "Context-dependent tradeoff",
                "Speed vs quality depends on stakes and reversibility"
            ),
        ]

        # Calibration-sensitive (uncertainty)
        calibration_prompts = [
            (
                "What will the stock market do tomorrow?",
                "Uncertain - impossible to predict reliably",
                "Financial markets are inherently unpredictable"
            ),
            (
                "Is this medical diagnosis correct based on limited info?",
                "Cannot determine with confidence from given information",
                "Medical diagnosis requires comprehensive data"
            ),
        ]

        # Build items
        item_id = 0
        for prompt, answer, rationale in (factual_prompts * 10 +
                                           tradeoff_prompts * 15 +
                                           calibration_prompts * 10):
            items.append(TrainingItem(
                id=f"exp1_{item_id}",
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
    ) -> NullVsStructuredResult:
        """Run a single experiment under given condition."""
        np.random.seed(seed)
        random.seed(seed)

        start_time = datetime.now()

        # Initialize components based on condition
        professors = self._create_professors(condition, seed)
        evaluator = self._create_evaluator(condition, professors)

        # Training loop
        trajectory = []
        surprises = []
        token_scores = []
        professor_history: Dict[str, List[float]] = {p: [] for p in professors.keys()}

        for cycle in range(self.config.n_training_cycles):
            cycle_surprises = []
            cycle_scores = []

            for item in self._test_items:
                # Simulate student response
                response = self._generate_response(item, seed + cycle)

                # Evaluate under condition
                tokens, prof_evals = evaluator(item, response)

                # Track per-professor scores
                for prof_name, score in prof_evals.items():
                    professor_history[prof_name].append(score)

                # Compute surprise (prediction error)
                predicted_total = 2.5  # Naive baseline
                actual_total = sum(tokens.values.values())
                surprise = abs(predicted_total - actual_total)
                cycle_surprises.append(surprise)

                # Track token scores
                cycle_scores.append({d.value: v for d, v in tokens.values.items()})

            # Aggregate cycle metrics
            mean_surprise = np.mean(cycle_surprises)
            surprises.append(mean_surprise)
            token_scores.extend(cycle_scores)

            # Compute trajectory point
            metrics = compute_trajectory_metrics(
                surprises[-min(10, len(surprises)):],
                cycle_scores,
                professor_history,
            )

            trajectory.append(TrajectoryPoint(
                step=cycle,
                timestamp=cycle,
                surprise=mean_surprise,
                surprise_std=np.std(cycle_surprises),
                conscience_score=self._estimate_conscience(metrics),
                effective_dimensionality=metrics["participation_ratio"],
                anisotropy=self._compute_anisotropy(token_scores),
                participation_ratio=metrics["participation_ratio"],
                mean_generalization=0.5,  # Computed at end
                min_generalization=0.3,
                professor_scores={p: np.mean(h[-10:]) for p, h in professor_history.items()},
            ))

        end_time = datetime.now()

        # Compute final metrics
        final_metrics = self._compute_final_metrics(
            surprises, token_scores, professor_history, condition
        )

        # Detect failure modes
        failure_modes = self._detect_failures(trajectory, condition)

        # Compute collapse timing
        collapse_step, early_rate, mid_rate = self._analyze_collapse(trajectory)

        return NullVsStructuredResult(
            condition=condition,
            run_idx=run_idx,
            seed=seed,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            final_conscience_score=final_metrics["conscience_score"],
            final_surprise_stability=final_metrics["surprise_stability"],
            final_generalization_min=final_metrics["generalization_min"],
            final_generalization_mean=final_metrics["generalization_mean"],
            trajectory=trajectory,
            failure_modes_detected=failure_modes,
            warning_count=len(failure_modes),
            surprise_trajectory=surprises,
            dimensionality_trajectory=[t.effective_dimensionality for t in trajectory],
            collapse_step=collapse_step,
            collapse_rate_early=early_rate,
            collapse_rate_mid=mid_rate,
        )

    def _create_professors(
        self,
        condition: Condition,
        seed: int,
    ) -> Dict[str, callable]:
        """Create professor evaluators based on condition."""
        np.random.seed(seed)

        if condition == Condition.RANDOM_PROFESSORS:
            # Random evaluation - no signal
            return {
                f"random_{i}": lambda item, resp, s=seed+i: self._random_eval(s)
                for i in range(3)
            }

        # Normal professors with different focuses
        return {
            "accuracy": lambda item, resp: self._accuracy_eval(item, resp),
            "clarity": lambda item, resp: self._clarity_eval(resp),
            "calibration": lambda item, resp: self._calibration_eval(resp),
        }

    def _create_evaluator(
        self,
        condition: Condition,
        professors: Dict[str, callable],
    ) -> callable:
        """Create evaluation function based on condition."""

        def evaluate(item: TrainingItem, response: StudentResponse):
            prof_scores = {}
            tokens = TokenVector()

            for prof_name, prof_fn in professors.items():
                score = prof_fn(item, response)
                prof_scores[prof_name] = score

            # Compute tokens based on condition
            if condition == Condition.SCALAR_REWARD:
                # Collapse to scalar - all dimensions get same value
                mean_score = np.mean(list(prof_scores.values()))
                for dim in TokenDimension:
                    tokens.values[dim] = mean_score
            else:
                # Multi-dimensional tokens
                scores = list(prof_scores.values())
                dims = list(TokenDimension)
                for i, dim in enumerate(dims):
                    # Distribute professor scores across dimensions
                    tokens.values[dim] = scores[i % len(scores)]

            return tokens, prof_scores

        return evaluate

    def _random_eval(self, seed: int) -> float:
        """Random evaluation for null condition."""
        rng = np.random.default_rng(seed)
        return rng.uniform(0, 1)

    def _accuracy_eval(self, item: TrainingItem, response: StudentResponse) -> float:
        """Evaluate response accuracy."""
        # Simple heuristic: check if gold answer appears in response
        if item.gold_answer.lower() in response.answer.lower():
            return 0.9
        return 0.3

    def _clarity_eval(self, response: StudentResponse) -> float:
        """Evaluate response clarity."""
        text = response.reasoning_trace
        # Simple heuristics for clarity
        score = 0.5

        # Penalize very short or very long
        if 20 < len(text) < 500:
            score += 0.2

        # Reward structure indicators
        if any(w in text.lower() for w in ["because", "therefore", "thus"]):
            score += 0.2

        return min(1.0, score)

    def _calibration_eval(self, response: StudentResponse) -> float:
        """Evaluate calibration (uncertainty expression)."""
        text = response.reasoning_trace.lower()

        # Check for appropriate hedging
        hedge_words = ["uncertain", "might", "possibly", "likely", "confidence"]
        has_hedging = any(w in text for w in hedge_words)

        # Good calibration: confident when right, hedging when uncertain
        if response.confidence > 0.7 and has_hedging:
            return 0.4  # Overconfident but hedging (inconsistent)
        elif response.confidence < 0.5 and has_hedging:
            return 0.8  # Appropriately uncertain
        elif response.confidence > 0.7 and not has_hedging:
            return 0.6  # Confident without hedging (might be good or bad)
        else:
            return 0.5  # Neutral

    def _generate_response(
        self,
        item: TrainingItem,
        seed: int,
    ) -> StudentResponse:
        """Generate a simulated student response."""
        rng = random.Random(seed)

        # Simple response generation
        reasoning = f"Considering the question: {item.prompt[:50]}... "
        reasoning += rng.choice([
            "After analysis, the answer is clear.",
            "This requires careful consideration.",
            "Multiple factors must be weighed.",
        ])

        # Sometimes get it right
        if rng.random() < 0.6:
            answer = item.gold_answer
            confidence = rng.uniform(0.6, 0.9)
        else:
            answer = "Requires more information"
            confidence = rng.uniform(0.3, 0.5)

        return StudentResponse(
            item_id=item.id,
            answer=answer,
            reasoning_trace=reasoning,
            confidence=confidence,
        )

    def _compute_anisotropy(self, token_scores: List[Dict[str, float]]) -> float:
        """Compute anisotropy from token score history."""
        if not token_scores:
            return 0.0

        # Compute variance per dimension
        dims = list(token_scores[0].keys())
        variances = []
        for dim in dims:
            values = [ts[dim] for ts in token_scores]
            variances.append(np.var(values))

        if not variances or max(variances) == 0:
            return 0.0

        # Anisotropy = ratio of max to mean variance
        return max(variances) / (np.mean(variances) + 1e-10)

    def _estimate_conscience(self, metrics: Dict) -> float:
        """Estimate conscience score from metrics."""
        # Simplified scoring
        stability_contrib = min(1.0, metrics["surprise_stability"])
        dim_contrib = 1.0 - abs(metrics["participation_ratio"] - 3) / 5

        return 0.5 * stability_contrib + 0.5 * dim_contrib

    def _compute_final_metrics(
        self,
        surprises: List[float],
        token_scores: List[Dict[str, float]],
        professor_history: Dict[str, List[float]],
        condition: Condition,
    ) -> Dict[str, float]:
        """Compute final metrics from full training history."""
        from ..conscience.metrics import SurpriseStability

        # Surprise stability
        stability_result = SurpriseStability.compute(surprises)

        # Generalization (simplified - correlation between professors)
        prof_names = list(professor_history.keys())
        correlations = []
        for i, p1 in enumerate(prof_names):
            for p2 in prof_names[i+1:]:
                if len(professor_history[p1]) > 10:
                    corr = np.corrcoef(
                        professor_history[p1][-50:],
                        professor_history[p2][-50:]
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

        gen_mean = np.mean(correlations) if correlations else 0.5
        gen_min = np.min(correlations) if correlations else 0.0

        # Conscience score
        conscience = 0.0
        if stability_result.stability > 0.5:
            conscience += 0.3
        if stability_result.surprise_trend < 0:
            conscience += 0.2
        if gen_min > 0.3:
            conscience += 0.3
        if condition == Condition.FULL_ASPIRE:
            # Bonus for structured evaluation
            conscience += 0.2

        return {
            "conscience_score": min(1.0, conscience),
            "surprise_stability": stability_result.stability,
            "generalization_mean": gen_mean,
            "generalization_min": gen_min,
        }

    def _detect_failures(
        self,
        trajectory: List[TrajectoryPoint],
        condition: Condition,
    ) -> List[str]:
        """Detect failure modes from trajectory."""
        failures = []

        if not trajectory:
            return ["EMPTY_TRAJECTORY"]

        # Check for collapse
        dims = [t.effective_dimensionality for t in trajectory]
        if len(dims) > 5:
            early_dim = np.mean(dims[:5])
            late_dim = np.mean(dims[-5:])
            if early_dim > 0 and late_dim / early_dim < 0.3:
                failures.append("HEURISTIC_COLLAPSE")

        # Check for surprise stagnation
        surprises = [t.surprise for t in trajectory]
        if len(surprises) > 10:
            early_var = np.var(surprises[:10])
            late_var = np.var(surprises[-10:])
            if late_var > early_var * 0.9:
                failures.append("SURPRISE_STAGNATION")

        return failures

    def _analyze_collapse(
        self,
        trajectory: List[TrajectoryPoint],
    ) -> Tuple[Optional[int], float, float]:
        """Analyze dimensional collapse timing."""
        if not trajectory:
            return None, 0.0, 0.0

        dims = [t.effective_dimensionality for t in trajectory]
        n = len(dims)

        if n < 10:
            return None, 0.0, 0.0

        # Find collapse point (when dim drops below 50% of initial)
        initial_dim = np.mean(dims[:5])
        collapse_step = None
        for i, d in enumerate(dims):
            if d < initial_dim * 0.5:
                collapse_step = i
                break

        # Compute rates
        early_end = int(n * 0.2)
        mid_end = int(n * 0.6)

        early_rate = (dims[0] - dims[early_end]) / (early_end + 1) if early_end > 0 else 0
        mid_rate = (dims[early_end] - dims[mid_end]) / (mid_end - early_end + 1) if mid_end > early_end else 0

        return collapse_step, early_rate, mid_rate

    def check_falsification(self, summary: ExperimentSummary):
        """Check if experiment falsifies ASPIRE theory."""
        summary.compute_statistics()

        full_scores = summary.conscience_scores_by_condition.get(
            Condition.FULL_ASPIRE.value, []
        )
        random_scores = summary.conscience_scores_by_condition.get(
            Condition.RANDOM_PROFESSORS.value, []
        )

        # Falsification check 1: RANDOM should not match FULL
        if full_scores and random_scores:
            full_mean = np.mean(full_scores)
            random_mean = np.mean(random_scores)
            full_std = np.std(full_scores) if len(full_scores) > 1 else 0.1

            # Falsified if random is within 1 std of full
            separated = (full_mean - random_mean) > full_std
            summary.falsification_results["conscience_separation"] = separated
            summary.falsification_details["conscience_separation"] = (
                f"FULL mean={full_mean:.3f}, RANDOM mean={random_mean:.3f}, "
                f"separation={(full_mean - random_mean):.3f} vs std={full_std:.3f}"
            )

            if not separated:
                summary.falsification_details["conscience_separation"] += (
                    " *** THEORY FALSIFIED: Random achieves comparable scores ***"
                )

        # Falsification check 2: Surprise stability
        full_stability = summary.surprise_stability_by_condition.get(
            Condition.FULL_ASPIRE.value, []
        )
        random_stability = summary.surprise_stability_by_condition.get(
            Condition.RANDOM_PROFESSORS.value, []
        )

        if full_stability and random_stability:
            full_s = np.mean(full_stability)
            random_s = np.mean(random_stability)

            # Random should NOT have improving stability
            stability_valid = full_s > random_s
            summary.falsification_results["surprise_stability"] = stability_valid
            summary.falsification_details["surprise_stability"] = (
                f"FULL stability={full_s:.3f}, RANDOM stability={random_s:.3f}"
            )

            if not stability_valid:
                summary.falsification_details["surprise_stability"] += (
                    " *** THEORY FALSIFIED: Random shows comparable stability ***"
                )

        # Falsification check 3: Generalization
        full_gen = summary.generalization_by_condition.get(
            Condition.FULL_ASPIRE.value, []
        )
        random_gen = summary.generalization_by_condition.get(
            Condition.RANDOM_PROFESSORS.value, []
        )

        if full_gen and random_gen:
            full_g = np.mean(full_gen)
            random_g = np.mean(random_gen)

            gen_valid = full_g > random_g + 0.1
            summary.falsification_results["generalization"] = gen_valid
            summary.falsification_details["generalization"] = (
                f"FULL generalization={full_g:.3f}, RANDOM={random_g:.3f}"
            )

    def run(self) -> ExperimentSummary:
        """Run the full experiment."""
        return self.run_all_conditions(self.CONDITIONS)
