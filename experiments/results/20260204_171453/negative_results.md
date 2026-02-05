## When Conscience Does Not Form

We tracked failure cases systematically to document the boundary conditions
of conscience formation. Not all failures indicate theoretical problems;
expected failures validate our understanding of the mechanism's limits.

Across 90 runs:
- 8 expected failures (theory predicts these)
- 11 unexpected failures
- 11 potential falsifications

### Expected Failure Patterns

**scalar_partial_conscience** (5 cases)
> SCALAR averaging doesn't fully destroy multi-dimensional signal. Document as boundary condition, not falsification.

**seed_variance** (3 cases)
> High run-to-run variance suggests sensitivity to initialization. Report confidence intervals and recommend larger n_runs.

### Unexpected Failures (Potential Falsifications)

**random_matches_full** (1 cases)
> RANDOM_PROFESSORS (mean=0.420) matches FULL_ASPIRE (mean=0.510±0.187). Random evaluation should not achieve comparable conscience.
> Interpretation: If random evaluation matches structured evaluation, conscience formation may be an artifact of training dynamics, not meaningful judgment learning.

**no_transfer_to_holdout** (10 cases)
> No transfer to holdout professor (corr=0.189) despite 50 training cycles. Theory predicts transfer should occur.
> Interpretation: Conscience may not generalize to unseen professors. This is a potential falsification if consistent.
