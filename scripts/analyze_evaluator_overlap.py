#!/usr/bin/env python
"""Analyze whether professors share latent evaluative structure.

This script answers the critical question:

    "Was holdout transfer even possible given the professor design?"

If professors measure genuinely orthogonal qualities, holdout transfer
SHOULD fail, and the experimental "falsification" is actually expected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scalarscope.analysis import (
    run_overlap_analysis,
    generate_overlap_report,
)


def main():
    print("=" * 60)
    print("EVALUATOR LATENT OVERLAP ANALYSIS")
    print("=" * 60)
    print()
    print("Question: Do the professors share a latent evaluative space?")
    print()

    # Run analysis with multiple seeds for stability
    print("Running analysis (200 items, 3 seeds)...")
    print()

    all_results = []
    for seed in [42, 123, 456]:
        result = run_overlap_analysis(n_items=200, seed=seed)
        all_results.append(result)
        print(f"  Seed {seed}: mean_corr={result.mean_pairwise_correlation:.3f}, "
              f"first_factor={result.variance_explained_by_first_factor:.1%}, "
              f"eff_dim={result.effective_dimensionality:.2f}")

    # Average results
    import numpy as np
    avg_corr = np.mean([r.mean_pairwise_correlation for r in all_results])
    avg_factor = np.mean([r.variance_explained_by_first_factor for r in all_results])
    avg_dim = np.mean([r.effective_dimensionality for r in all_results])

    print()
    print("-" * 60)
    print("AVERAGED RESULTS")
    print("-" * 60)
    print(f"Mean inter-professor correlation: {avg_corr:.3f}")
    print(f"Variance explained by first factor: {avg_factor:.1%}")
    print(f"Effective dimensionality: {avg_dim:.2f} / 3")
    print()

    # Interpretation
    print("-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    if avg_corr < 0.2:
        print("LOW INTER-PROFESSOR AGREEMENT (r < 0.2)")
        print()
        print("The professors measure largely INDEPENDENT qualities.")
        print("There is no shared latent space for the student to internalize.")
        print()
        print("IMPLICATION: Holdout transfer SHOULD FAIL.")
        print("The observed experimental failure is EXPECTED, not a falsification.")
    elif avg_corr < 0.5:
        print("MODERATE INTER-PROFESSOR AGREEMENT (0.2 < r < 0.5)")
        print()
        print("Some shared structure exists, but professors also measure")
        print("distinct qualities.")
        print()
        print("IMPLICATION: Holdout transfer is UNCERTAIN.")
        print("The failure may be partially explained by design, partially by ASPIRE.")
    else:
        print("HIGH INTER-PROFESSOR AGREEMENT (r > 0.5)")
        print()
        print("Professors measure similar qualities with substantial overlap.")
        print("A shared latent space exists.")
        print()
        print("IMPLICATION: Holdout transfer SHOULD SUCCEED.")
        print("If it fails, this is a genuine limitation of ASPIRE.")

    print()

    if avg_factor < 0.3:
        print("WEAK FIRST FACTOR (<30% variance)")
        print("No dominant shared dimension. Evaluation is multi-dimensional")
        print("with no clear common ground.")
    elif avg_factor < 0.5:
        print("MODERATE FIRST FACTOR (30-50% variance)")
        print("Partial shared structure exists.")
    else:
        print("STRONG FIRST FACTOR (>50% variance)")
        print("A dominant shared dimension exists.")

    print()

    if avg_dim > 2.5:
        print("HIGH EFFECTIVE DIMENSIONALITY (>2.5/3)")
        print("Professors are nearly orthogonal. Each measures something distinct.")
    elif avg_dim > 1.5:
        print("MODERATE EFFECTIVE DIMENSIONALITY (1.5-2.5/3)")
        print("Some redundancy, but still substantial independence.")
    else:
        print("LOW EFFECTIVE DIMENSIONALITY (<1.5/3)")
        print("High redundancy. Professors measure similar things.")

    print()
    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    if avg_corr < 0.2 and avg_factor < 0.3:
        print()
        print("The professors do NOT share meaningful latent structure.")
        print()
        print("This means:")
        print("  1. Holdout transfer failure is EXPECTED")
        print("  2. The 'falsification' is actually a design limitation")
        print("  3. ASPIRE correctly learned conditional evaluation")
        print("  4. Testing internalization requires different professors")
        print()
        print("The experimental design cannot distinguish 'learned judgment'")
        print("from 'learned evaluator-specific patterns' because there IS NO")
        print("shared judgment to learn.")
    else:
        print()
        print("The professors DO share some latent structure.")
        print()
        print("This means:")
        print("  1. Holdout transfer was possible in principle")
        print("  2. The failure indicates ASPIRE limitation")
        print("  3. Further investigation warranted")

    # Generate full report
    print()
    print("-" * 60)
    print("Saving full report...")

    output_dir = Path("experiments/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use last result for detailed report
    report = generate_overlap_report(all_results[-1])
    report_path = output_dir / "evaluator_overlap_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  {report_path}")

    # Save raw data
    import json
    data_path = output_dir / "evaluator_overlap_data.json"
    with open(data_path, "w") as f:
        json.dump({
            "seeds": [42, 123, 456],
            "results": [r.to_dict() for r in all_results],
            "averaged": {
                "mean_pairwise_correlation": avg_corr,
                "variance_explained_by_first_factor": avg_factor,
                "effective_dimensionality": avg_dim,
            }
        }, f, indent=2)
    print(f"  {data_path}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
