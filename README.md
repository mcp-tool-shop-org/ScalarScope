<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/scalarscope/main/logo.png" alt="ScalarScope logo" width="200" />
</p>

<h1 align="center">ScalarScope</h1>

<p align="center">
  <strong>Evaluative Internalization Training Framework</strong><br>
  Train models to internalize scalar evaluations — developing genuine judgment, not just reward prediction.
</p>

<p align="center">
  <a href="https://pypi.org/project/scalarscope/"><img src="https://img.shields.io/pypi/v/scalarscope?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/scalarscope/"><img src="https://img.shields.io/pypi/pyversions/scalarscope" alt="Python versions"></a>
  <a href="https://github.com/mcp-tool-shop-org/scalarscope/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mcp-tool-shop-org/scalarscope" alt="License"></a>
  <a href="https://github.com/mcp-tool-shop-org/scalarscope/issues"><img src="https://img.shields.io/github/issues/mcp-tool-shop-org/scalarscope" alt="Issues"></a>
</p>

---

## Why ScalarScope?

Standard reward models collapse to a single signal. ScalarScope asks a harder question:

> **Can a model learn to predict how _multiple independent evaluators_ would rate its output — and internalize that judgment so evaluators aren't needed at inference time?**

Most RLHF setups treat evaluation as a black box. ScalarScope cracks it open:

- **Token-level scalar feedback** instead of sequence-level rewards — fine-grained learning signals that localize exactly _where_ quality changes.
- **Multi-evaluator geometry** — train against several evaluators simultaneously and analyze whether their criteria converge on a shared latent manifold.
- **Internalization detection** — measure whether the model develops genuine evaluative intuition (Path B) or just memorizes surface patterns (Path A).
- **Governor-controlled budgets** — adaptive token budgeting prevents runaway training costs.

If you're researching alignment, evaluation dynamics, or interpretable training signals, ScalarScope gives you the engine and the instrumentation.

## Installation

```bash
# Core (NumPy + Pydantic only)
pip install scalarscope

# With PyTorch backend
pip install "scalarscope[torch]"

# With ONNX Runtime (GPU inference)
pip install "scalarscope[onnx]"

# Development (adds pytest + ruff)
pip install "scalarscope[dev]"
```

**Requirements:** Python 3.11+

## Quick Start

```python
from scalarscope.engine import ScalarScopeEngine
from scalarscope.governor import TokenPool, GovernorConfig

# Set up token budget governance
config = GovernorConfig(
    max_tokens_per_cycle=1000,
    budget_strategy="adaptive",
)
pool = TokenPool(config)

# Create the training engine
engine = ScalarScopeEngine(
    model=your_model,
    evaluators=your_evaluators,
    token_pool=pool,
)

# Run a training cycle
result = engine.run_cycle(prompt="Your training prompt")
print(f"Loss:  {result.metrics.loss:.4f}")
print(f"Tokens used: {result.metrics.tokens_used}")
```

## Architecture

```
src/scalarscope/
├── engine/           # Core training loop + revision engine
├── governor/         # Token budget management
├── critic/           # Learned critic with logit-derived features
├── evaluators/       # Evaluator protocol + scalar head
├── export/           # Geometry export for visualization
├── geometry/         # Trajectory & eigenvalue analysis
├── conscience/       # Internalized evaluator probes
├── analysis/         # Post-hoc analysis utilities
├── adversarial/      # Adversarial robustness testing
├── professors/       # Multi-professor evaluation setups
├── student/          # Student model abstractions
└── core/             # Shared types and base classes
```

## Key Components

### ScalarScopeEngine

The core loop: generate, evaluate, update, export.

```python
engine = ScalarScopeEngine(model, evaluators, token_pool)
result = engine.run_cycle(prompt="...")
```

### RevisionScalarScopeEngine

Extended engine with self-correction. Detects when outputs need revision, applies targeted corrections, and learns from revision patterns.

### TokenPool and Governor

Adaptive token budgeting prevents runaway usage:

```python
config = GovernorConfig(max_tokens_per_cycle=2000, budget_strategy="adaptive")
pool = TokenPool(config)
remaining = pool.remaining  # check budget mid-cycle
```

### Geometry Export

Export training dynamics for visualization in [ScalarScope-Desktop](https://github.com/mcp-tool-shop-org/ScalarScope-Desktop) (WinUI 3 / .NET MAUI):

- State-vector trajectories
- Eigenvalue spectra
- Evaluator geometry overlays

### Learned Critic

Token-level scalar predictor that learns evaluative features from logits — the core of the internalization hypothesis.

## Examples

| Script | What it shows |
|--------|---------------|
| `demo_loop.py` | Basic training loop |
| `demo_revision.py` | Self-correction capabilities |
| `demo_geometry.py` | Geometry export for visualization |
| `demo_governor.py` | Token budget management |
| `demo_learned_critic.py` | Learned critic training |
| `demo_onnx_loop.py` | ONNX Runtime inference |
| `bench_kv_cache.py` | KV cache benchmarking |

Run any example:

```bash
cd examples
python demo_loop.py
```

## Scientific Background

ScalarScope explores a central question in AI alignment: whether models can internalize evaluative criteria rather than merely predicting rewards.

**Key findings from our experiments:**

- **Path B (success):** When evaluators share a latent evaluative manifold, internalization succeeds. The model develops genuine judgment.
- **Path A (failure):** When evaluators are orthogonal, the model resorts to surface-level pattern matching.

See `docs/RESULTS_AND_LIMITATIONS.md` for full experimental results and known limitations.

## Related Projects

- [ScalarScope-Desktop](https://github.com/mcp-tool-shop-org/ScalarScope-Desktop) — WinUI 3 visualization app for geometry export data

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
git clone https://github.com/mcp-tool-shop-org/scalarscope.git
cd scalarscope
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
