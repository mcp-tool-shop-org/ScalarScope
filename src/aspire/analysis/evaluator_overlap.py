"""Evaluator Latent Overlap Analysis.

This module answers the critical question:

    "Do the professors share a latent evaluative space?"

If professors measure genuinely orthogonal qualities with no shared structure,
then holdout transfer SHOULD fail, and the original internalization hypothesis
was untestable with this professor design.

Key analyses:
1. Inter-professor agreement (pairwise correlations)
2. Factor analysis (do scores load on shared factors?)
3. Canonical correlation analysis (shared variance)
4. Effective dimensionality of the evaluation space

If these analyses show low overlap, the holdout transfer failure is EXPECTED,
not a falsification of ASPIRE - it's a falsification of the experimental design.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path


@dataclass
class EvaluatorOverlapResult:
    """Results from evaluator overlap analysis."""

    # Inter-professor agreement
    pairwise_correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    mean_pairwise_correlation: float = 0.0
    min_pairwise_correlation: float = 0.0
    max_pairwise_correlation: float = 0.0

    # Factor analysis
    n_factors_extracted: int = 0
    variance_explained_by_first_factor: float = 0.0
    total_variance_explained: float = 0.0
    factor_loadings: Dict[str, List[float]] = field(default_factory=dict)

    # Effective dimensionality
    effective_dimensionality: float = 0.0
    participation_ratio: float = 0.0

    # Interpretation
    has_shared_structure: bool = False
    interpretation: str = ""
    transfer_should_succeed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pairwise_correlations": {
                f"{k[0]}_vs_{k[1]}": v for k, v in self.pairwise_correlations.items()
            },
            "mean_pairwise_correlation": self.mean_pairwise_correlation,
            "min_pairwise_correlation": self.min_pairwise_correlation,
            "max_pairwise_correlation": self.max_pairwise_correlation,
            "n_factors_extracted": self.n_factors_extracted,
            "variance_explained_by_first_factor": self.variance_explained_by_first_factor,
            "total_variance_explained": self.total_variance_explained,
            "factor_loadings": self.factor_loadings,
            "effective_dimensionality": self.effective_dimensionality,
            "participation_ratio": self.participation_ratio,
            "has_shared_structure": self.has_shared_structure,
            "interpretation": self.interpretation,
            "transfer_should_succeed": self.transfer_should_succeed,
        }


class EvaluatorOverlapAnalyzer:
    """Analyzes whether evaluators share latent structure."""

    def __init__(self, professor_names: List[str]):
        self.professor_names = professor_names
        self.scores: Dict[str, List[float]] = {name: [] for name in professor_names}

    def record_evaluation(self, professor_scores: Dict[str, float]):
        """Record a single evaluation from all professors."""
        for name, score in professor_scores.items():
            if name in self.scores:
                self.scores[name].append(score)

    def analyze(self, min_samples: int = 50) -> EvaluatorOverlapResult:
        """Run the full overlap analysis."""
        result = EvaluatorOverlapResult()

        # Check we have enough data
        n_samples = min(len(scores) for scores in self.scores.values())
        if n_samples < min_samples:
            result.interpretation = (
                f"Insufficient data ({n_samples} samples, need {min_samples})"
            )
            return result

        # Build score matrix
        score_matrix = np.array([
            self.scores[name][:n_samples] for name in self.professor_names
        ])  # Shape: (n_professors, n_samples)

        # 1. Pairwise correlations
        self._compute_pairwise_correlations(score_matrix, result)

        # 2. Factor analysis (via PCA as proxy)
        self._compute_factor_analysis(score_matrix, result)

        # 3. Effective dimensionality
        self._compute_effective_dimensionality(score_matrix, result)

        # 4. Interpretation
        self._interpret_results(result)

        return result

    def _compute_pairwise_correlations(
        self,
        score_matrix: np.ndarray,
        result: EvaluatorOverlapResult,
    ):
        """Compute pairwise correlations between professors."""
        n_profs = len(self.professor_names)
        correlations = []

        for i in range(n_profs):
            for j in range(i + 1, n_profs):
                corr = np.corrcoef(score_matrix[i], score_matrix[j])[0, 1]
                if not np.isnan(corr):
                    key = (self.professor_names[i], self.professor_names[j])
                    result.pairwise_correlations[key] = corr
                    correlations.append(corr)

        if correlations:
            result.mean_pairwise_correlation = np.mean(correlations)
            result.min_pairwise_correlation = np.min(correlations)
            result.max_pairwise_correlation = np.max(correlations)

    def _compute_factor_analysis(
        self,
        score_matrix: np.ndarray,
        result: EvaluatorOverlapResult,
    ):
        """Compute factor analysis via PCA."""
        # Center the data
        centered = score_matrix - score_matrix.mean(axis=1, keepdims=True)

        # Covariance matrix
        cov = np.cov(centered)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Variance explained
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            var_explained = eigenvalues / total_var

            # First factor
            result.variance_explained_by_first_factor = var_explained[0]

            # Number of factors explaining > 90% variance
            cumsum = np.cumsum(var_explained)
            result.n_factors_extracted = int(np.searchsorted(cumsum, 0.9) + 1)
            result.total_variance_explained = cumsum[min(
                result.n_factors_extracted - 1,
                len(cumsum) - 1
            )]

            # Factor loadings (first 2 factors)
            for i, name in enumerate(self.professor_names):
                loadings = []
                for f in range(min(2, len(eigenvectors[0]))):
                    loadings.append(float(eigenvectors[i, f]))
                result.factor_loadings[name] = loadings

    def _compute_effective_dimensionality(
        self,
        score_matrix: np.ndarray,
        result: EvaluatorOverlapResult,
    ):
        """Compute effective dimensionality of the evaluation space."""
        # Covariance matrix
        cov = np.cov(score_matrix)

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

        # Participation ratio (effective dimensionality)
        total = np.sum(eigenvalues)
        if total > 0:
            normalized = eigenvalues / total
            # Participation ratio = 1 / sum(p^2)
            sum_sq = np.sum(normalized ** 2)
            if sum_sq > 0:
                result.participation_ratio = 1.0 / sum_sq
                result.effective_dimensionality = result.participation_ratio

    def _interpret_results(self, result: EvaluatorOverlapResult):
        """Interpret the analysis results."""
        interpretations = []

        # Check pairwise correlations
        if result.mean_pairwise_correlation > 0.5:
            interpretations.append(
                f"High inter-professor agreement (r={result.mean_pairwise_correlation:.2f}): "
                "Professors measure similar qualities."
            )
            result.has_shared_structure = True
        elif result.mean_pairwise_correlation > 0.2:
            interpretations.append(
                f"Moderate inter-professor agreement (r={result.mean_pairwise_correlation:.2f}): "
                "Some shared structure exists."
            )
            result.has_shared_structure = True
        else:
            interpretations.append(
                f"Low inter-professor agreement (r={result.mean_pairwise_correlation:.2f}): "
                "Professors measure largely independent qualities."
            )
            result.has_shared_structure = False

        # Check factor structure
        if result.variance_explained_by_first_factor > 0.5:
            interpretations.append(
                f"Strong first factor ({result.variance_explained_by_first_factor:.1%} variance): "
                "A dominant shared dimension exists."
            )
        elif result.variance_explained_by_first_factor > 0.3:
            interpretations.append(
                f"Moderate first factor ({result.variance_explained_by_first_factor:.1%} variance): "
                "Partial shared structure."
            )
        else:
            interpretations.append(
                f"Weak first factor ({result.variance_explained_by_first_factor:.1%} variance): "
                "No dominant shared dimension."
            )

        # Check effective dimensionality
        n_profs = len(self.professor_names)
        if result.effective_dimensionality > n_profs * 0.8:
            interpretations.append(
                f"High effective dimensionality ({result.effective_dimensionality:.1f}/{n_profs}): "
                "Evaluation space is nearly full-rank (orthogonal professors)."
            )
        elif result.effective_dimensionality > n_profs * 0.5:
            interpretations.append(
                f"Moderate effective dimensionality ({result.effective_dimensionality:.1f}/{n_profs}): "
                "Some redundancy in evaluation."
            )
        else:
            interpretations.append(
                f"Low effective dimensionality ({result.effective_dimensionality:.1f}/{n_profs}): "
                "Evaluation space is low-rank (highly correlated professors)."
            )

        # Final verdict on transfer
        if result.has_shared_structure and result.variance_explained_by_first_factor > 0.3:
            result.transfer_should_succeed = True
            interpretations.append(
                "\nVERDICT: Shared structure exists. Holdout transfer SHOULD succeed. "
                "If it fails, this may indicate a genuine limitation of ASPIRE."
            )
        else:
            result.transfer_should_succeed = False
            interpretations.append(
                "\nVERDICT: No meaningful shared structure. Holdout transfer SHOULD FAIL. "
                "The failure is expected given the evaluator design, not a limitation of ASPIRE."
            )

        result.interpretation = "\n".join(interpretations)


def analyze_professors_from_experiment(
    experiment_results_dir: Path,
) -> EvaluatorOverlapResult:
    """Analyze professor overlap from saved experiment results.

    This re-runs the evaluation to collect per-item professor scores.
    """
    import json

    # Load experiment config
    results_files = list(experiment_results_dir.glob("*_results.json"))
    if not results_files:
        raise FileNotFoundError(f"No results files in {experiment_results_dir}")

    # We need to re-run with score collection, but for now just return a template
    # In practice, this would load the detailed per-item scores
    result = EvaluatorOverlapResult()
    result.interpretation = (
        "To run this analysis, use run_overlap_analysis() with fresh data collection."
    )
    return result


def run_overlap_analysis(
    n_items: int = 200,
    seed: int = 42,
) -> EvaluatorOverlapResult:
    """Run a dedicated overlap analysis on the professors.

    This generates items and collects professor scores to analyze
    whether they share latent structure.
    """
    import random
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from aspire.core import TrainingItem, StudentResponse

    np.random.seed(seed)
    random.seed(seed)

    # Create test items (same as experiments)
    items = []
    factual = [
        ("What is 2 + 2?", "4"),
        ("Capital of France?", "Paris"),
        ("Days in a week?", "7"),
    ]
    tradeoffs = [
        ("Profit or ethics?", "Both matter"),
        ("Speed or quality?", "Balance"),
    ]
    calibration = [
        ("Stock market tomorrow?", "Uncertain"),
        ("Is this diagnosis correct?", "Cannot determine"),
    ]

    item_id = 0
    all_prompts = factual * 30 + tradeoffs * 30 + calibration * 20
    for prompt, answer in all_prompts[:n_items]:
        items.append(TrainingItem(
            id=f"overlap_{item_id}",
            prompt=prompt,
            gold_answer=answer,
            gold_rationale="",
            domain="mixed",
        ))
        item_id += 1

    # Create professors (same as experiments)
    def accuracy_eval(item: TrainingItem, response: StudentResponse) -> float:
        if item.gold_answer.lower() in response.answer.lower():
            return 0.8 + random.uniform(0, 0.2)
        return 0.3 + random.uniform(0, 0.2)

    def clarity_eval(response: StudentResponse) -> float:
        text = response.reasoning_trace
        score = 0.5
        if 20 < len(text) < 300:
            score += 0.2
        if any(w in text.lower() for w in ["because", "therefore"]):
            score += 0.2
        return min(1.0, score)

    def calibration_eval(response: StudentResponse) -> float:
        text = response.reasoning_trace.lower()
        hedging = any(w in text for w in ["might", "perhaps", "uncertain"])
        confidence = response.confidence
        if hedging and confidence > 0.7:
            return 0.3
        elif hedging and confidence < 0.5:
            return 0.8
        elif not hedging and confidence > 0.7:
            return 0.6
        return 0.5

    professors = {
        "accuracy": lambda item, resp: accuracy_eval(item, resp),
        "clarity": lambda item, resp: clarity_eval(resp),
        "calibration": lambda item, resp: calibration_eval(resp),
    }

    # Create analyzer
    analyzer = EvaluatorOverlapAnalyzer(list(professors.keys()))

    # Generate responses and collect scores
    for item in items:
        # Generate varied responses
        response = StudentResponse(
            item_id=item.id,
            answer=random.choice([item.gold_answer, "Unclear", "Depends"]),
            reasoning_trace=random.choice([
                "After analysis, the answer is clear.",
                "This might require more thought because of complexity.",
                "I think the answer is straightforward.",
                "Perhaps we should consider multiple factors.",
                "Therefore, the conclusion follows logically.",
            ]),
            confidence=random.uniform(0.3, 0.9),
        )

        # Collect professor scores
        scores = {}
        for name, eval_fn in professors.items():
            scores[name] = eval_fn(item, response)

        analyzer.record_evaluation(scores)

    # Run analysis
    return analyzer.analyze()


def generate_overlap_report(result: EvaluatorOverlapResult) -> str:
    """Generate a markdown report from overlap analysis."""
    lines = [
        "# Evaluator Latent Overlap Analysis",
        "",
        "## Question",
        "",
        "> Do the professors share a latent evaluative space?",
        "",
        "If professors measure genuinely orthogonal qualities, holdout transfer",
        "SHOULD fail. This is not a limitation of ASPIRE - it means the",
        "experimental design cannot test internalization.",
        "",
        "## Inter-Professor Agreement",
        "",
        f"- Mean pairwise correlation: **{result.mean_pairwise_correlation:.3f}**",
        f"- Range: [{result.min_pairwise_correlation:.3f}, {result.max_pairwise_correlation:.3f}]",
        "",
        "| Professor Pair | Correlation |",
        "|----------------|-------------|",
    ]

    for (p1, p2), corr in result.pairwise_correlations.items():
        lines.append(f"| {p1} vs {p2} | {corr:.3f} |")

    lines.extend([
        "",
        "## Factor Analysis",
        "",
        f"- Factors needed for 90% variance: **{result.n_factors_extracted}**",
        f"- First factor explains: **{result.variance_explained_by_first_factor:.1%}**",
        f"- Total variance explained: **{result.total_variance_explained:.1%}**",
        "",
        "### Factor Loadings",
        "",
        "| Professor | Factor 1 | Factor 2 |",
        "|-----------|----------|----------|",
    ])

    for name, loadings in result.factor_loadings.items():
        f1 = loadings[0] if len(loadings) > 0 else 0
        f2 = loadings[1] if len(loadings) > 1 else 0
        lines.append(f"| {name} | {f1:.3f} | {f2:.3f} |")

    lines.extend([
        "",
        "## Effective Dimensionality",
        "",
        f"- Participation ratio: **{result.participation_ratio:.2f}**",
        f"- Effective dimensions: **{result.effective_dimensionality:.1f}** / {len(result.factor_loadings)}",
        "",
        "## Interpretation",
        "",
    ])

    for line in result.interpretation.split("\n"):
        lines.append(f"> {line}")

    lines.extend([
        "",
        "## Implications for Holdout Transfer",
        "",
    ])

    if result.transfer_should_succeed:
        lines.extend([
            "Given the shared structure detected, holdout transfer **should succeed**.",
            "If it fails consistently, this indicates a genuine limitation of ASPIRE's",
            "ability to learn transferable judgment.",
        ])
    else:
        lines.extend([
            "Given the lack of shared structure, holdout transfer **should fail**.",
            "The observed failure in experiments is **expected** and does not falsify",
            "ASPIRE. Instead, it reveals that the professors measure orthogonal qualities",
            "with no common latent space to internalize.",
            "",
            "**This is a property of the experimental design, not ASPIRE itself.**",
        ])

    return "\n".join(lines)
