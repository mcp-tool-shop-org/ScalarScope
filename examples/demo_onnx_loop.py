"""Demo: Run the ASPIRE training loop with a real ONNX model.

Requirements:
1. Export a model to ONNX (see scripts/export_model.py)
2. Place the .onnx file in models/
3. Run this script

Example:
    python examples/demo_onnx_loop.py --model models/phi3-mini.onnx --tokenizer microsoft/Phi-3-mini-4k-instruct
"""

import argparse
from pathlib import Path

from aspire.core import TrainingItem
from aspire.student import ONNXStudentV1, GenerationConfig
from aspire.professors import ProfessorEnsemble
from aspire.critic import HeuristicCritic
from aspire.engine import AspireEngine


def create_test_items():
    """Create sample training items for testing."""
    return [
        TrainingItem(
            id="ethics_001",
            prompt="A self-driving car must choose between hitting one pedestrian or swerving into five. What should it do?",
            gold_answer="This is a trolley problem variant with no objectively correct answer",
            gold_rationale="The ethical dilemma involves competing frameworks: utilitarian vs deontological. The key insight is acknowledging the genuine moral tension.",
            difficulty=0.8,
            domain="ethics",
            near_misses=["Hit the one pedestrian", "Swerve into the five"],
        ),
        TrainingItem(
            id="code_review_001",
            prompt="Should we merge this PR that adds 2000 lines but has no tests?",
            gold_answer="No, require tests before merging",
            gold_rationale="Large changes without tests create maintenance burden. The tradeoff is velocity vs stability.",
            difficulty=0.4,
            domain="code_review",
            near_misses=["Yes, merge it", "Merge with a TODO"],
        ),
        TrainingItem(
            id="strategy_001",
            prompt="Our competitor just cut prices 30%. Should we match them?",
            gold_answer="It depends on cost structure, brand positioning, and customer loyalty",
            gold_rationale="Price matching may trigger a race to the bottom. Multiple valid responses exist.",
            difficulty=0.7,
            domain="business",
            near_misses=["Yes, match immediately", "No, hold prices"],
        ),
        TrainingItem(
            id="technical_001",
            prompt="Should we use a microservices architecture for our new project?",
            gold_answer="It depends on team size, expected scale, and operational maturity",
            gold_rationale="Microservices add complexity. For small teams or early-stage projects, a monolith is often better.",
            difficulty=0.6,
            domain="architecture",
            near_misses=["Yes, always use microservices", "No, monoliths are better"],
        ),
    ]


def on_cycle_complete(result):
    """Callback for each training cycle."""
    correct = "✓" if result.evaluation.consensus_correct else "✗"
    tokens = result.tokens_earned.total
    surprise = result.misalignment.total_surprise

    print(f"\n[{result.item.id}] {correct}")
    print(f"  Answer: {result.response.answer[:80]}...")
    print(f"  Confidence: {result.response.confidence:.2f}")
    print(f"  Tokens: {tokens:.2f} | Surprise: {surprise:.2f}")
    print(f"  Time: {result.cycle_time_ms:.0f}ms ({result.response.latency_ms:.0f}ms inference)")

    # Show professor verdicts
    for critique in result.evaluation.critiques:
        verdict = "✓" if critique.is_correct else "✗"
        print(f"  [{critique.professor_id}] {verdict} {critique.critique_text[:60]}")

    if result.teaching_moment.should_revise:
        print(f"  ⚠ High surprise - student should revise")


def main():
    parser = argparse.ArgumentParser(description="ASPIRE ONNX Demo")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name/path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "dml", "cpu"])
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("=" * 70)
    print("ASPIRE Engine - ONNX Demo")
    print("=" * 70)

    # Verify model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("\nTo export a model, run:")
        print("  python scripts/export_model.py --model microsoft/Phi-3-mini-4k-instruct")
        return

    # Create student
    print(f"\nLoading model: {model_path}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Device: {args.device}")

    config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )

    student = ONNXStudentV1(
        model_path=str(model_path),
        tokenizer_name_or_path=args.tokenizer,
        device=args.device,
        generation_config=config,
    )

    # Show model info
    info = student.get_model_info()
    print(f"Providers: {info['providers']}")
    print(f"Vocab size: {info['vocab_size']}")

    # Create other components
    professors = ProfessorEnsemble()
    critic = HeuristicCritic()

    # Create engine
    engine = AspireEngine(
        student=student,
        professors=professors,
        critic=critic,
        on_cycle_complete=on_cycle_complete,
    )

    # Run training
    print("\n" + "-" * 70)
    print("Starting training loop")
    print("-" * 70)

    items = create_test_items()
    metrics = engine.train(iter(items), max_cycles=args.cycles)

    # Report
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Total cycles:     {metrics.total_cycles}")
    print(f"Accuracy:         {metrics.accuracy:.1%}")
    print(f"Avg tokens:       {metrics.token_ledger.mean.total:.2f}")
    print(f"Avg surprise:     {metrics.avg_surprise:.3f}")
    print(f"Avg cycle time:   {metrics.avg_cycle_time_ms:.0f}ms")

    print("\nToken breakdown (mean):")
    for dim, val in metrics.token_ledger.mean.values.items():
        print(f"  {dim.value:15} {val:.2f}")


if __name__ == "__main__":
    main()
