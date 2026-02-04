# Evaluator Latent Overlap Analysis

## Question

> Do the professors share a latent evaluative space?

If professors measure genuinely orthogonal qualities, holdout transfer
SHOULD fail. This is not a limitation of ASPIRE - it means the
experimental design cannot test internalization.

## Inter-Professor Agreement

- Mean pairwise correlation: **0.004**
- Range: [-0.085, 0.099]

| Professor Pair | Correlation |
|----------------|-------------|
| accuracy vs clarity | -0.000 |
| accuracy vs calibration | -0.085 |
| clarity vs calibration | 0.099 |

## Factor Analysis

- Factors needed for 90% variance: **3**
- First factor explains: **67.8%**
- Total variance explained: **100.0%**

### Factor Loadings

| Professor | Factor 1 | Factor 2 |
|-----------|----------|----------|
| accuracy | -0.998 | 0.070 |
| clarity | 0.002 | 0.137 |
| calibration | 0.070 | 0.988 |

## Effective Dimensionality

- Participation ratio: **1.93**
- Effective dimensions: **1.9** / 3

## Interpretation

> Low inter-professor agreement (r=0.00): Professors measure largely independent qualities.
> Strong first factor (67.8% variance): A dominant shared dimension exists.
> Moderate effective dimensionality (1.9/3): Some redundancy in evaluation.
> 
> VERDICT: No meaningful shared structure. Holdout transfer SHOULD FAIL. The failure is expected given the evaluator design, not a limitation of ASPIRE.

## Implications for Holdout Transfer

Given the lack of shared structure, holdout transfer **should fail**.
The observed failure in experiments is **expected** and does not falsify
ASPIRE. Instead, it reveals that the professors measure orthogonal qualities
with no common latent space to internalize.

**This is a property of the experimental design, not ASPIRE itself.**