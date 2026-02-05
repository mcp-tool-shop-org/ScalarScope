"""ASPIRE Analysis Tools.

This module provides diagnostic and analysis tools for understanding
ASPIRE behavior, including:

- Evaluator overlap analysis: Do professors share latent structure?
"""

from .evaluator_overlap import (
    EvaluatorOverlapAnalyzer,
    EvaluatorOverlapResult,
    run_overlap_analysis,
    generate_overlap_report,
)

__all__ = [
    "EvaluatorOverlapAnalyzer",
    "EvaluatorOverlapResult",
    "run_overlap_analysis",
    "generate_overlap_report",
]
