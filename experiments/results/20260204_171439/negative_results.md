## When Conscience Does Not Form

We tracked failure cases systematically to document the boundary conditions
of conscience formation. Not all failures indicate theoretical problems;
expected failures validate our understanding of the mechanism's limits.

Across 27 runs:
- 8 expected failures (theory predicts these)
- 1 unexpected failures
- 1 potential falsifications

### Expected Failure Patterns

**scalar_partial_conscience** (1 cases)
> SCALAR averaging doesn't fully destroy multi-dimensional signal. Document as boundary condition, not falsification.

**early_holdout_failure** (3 cases)
> Transfer learning requires sufficient training. Report minimum training threshold for reliable transfer.

**seed_variance** (4 cases)
> High run-to-run variance suggests sensitivity to initialization. Report confidence intervals and recommend larger n_runs.

### Unexpected Failures (Potential Falsifications)

**random_matches_full** (1 cases)
> RANDOM_PROFESSORS (mean=0.533) matches FULL_ASPIRE (mean=0.400±0.141). Random evaluation should not achieve comparable conscience.
> Interpretation: If random evaluation matches structured evaluation, conscience formation may be an artifact of training dynamics, not meaningful judgment learning.

### Boundary Conditions

- High variance (CV=1.18) in holdout_one makes conclusions uncertain
- Training duration (20 cycles) may be insufficient for stable conscience formation
- Sample size (30 items) may be too small for reliable evaluation
- Number of runs (3) may be too small for statistical significance
