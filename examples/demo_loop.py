"""Demo: Run the ScalarScope training loop with mock components."""

from scalarscope.core import TrainingItem
from scalarscope.student import MockStudent
from scalarscope.professors import ProfessorEnsemble
from scalarscope.critic import HeuristicCritic
from scalarscope.engine import ScalarScopeEngine


def create_test_items():
    """Create some sample training items."""
    items = [
        TrainingItem(
            id="ethics_001",
            prompt="A self-driving car must choose between hitting one pedestrian or swerving into five. What should it do?",
            gold_answer="This is a trolley problem variant with no objectively correct answer",
            gold_rationale="The ethical dilemma involves competing frameworks: utilitarian (minimize total harm) vs deontological (never actively cause harm). The key insight is acknowledging the genuine moral tension rather than claiming certainty.",
            difficulty=0.8,
            domain="ethics",
            near_misses=["Hit the one pedestrian", "Swerve into the five", "Apply brakes"],
        ),
        TrainingItem(
            id="code_review_001",
            prompt="Should we merge this PR that adds 2000 lines but has no tests?",
            gold_answer="No, require tests before merging",
            gold_rationale="Large changes without tests create maintenance burden and regression risk. The tradeoff is short-term velocity vs long-term stability. Requiring tests is the standard practice.",
            difficulty=0.4,
            domain="code_review",
            near_misses=["Yes, merge it", "Merge with a TODO for tests"],
        ),
        TrainingItem(
            id="strategy_001",
            prompt="Our competitor just cut prices 30%. Should we match them?",
            gold_answer="It depends on our cost structure, brand positioning, and customer loyalty",
            gold_rationale="Price matching may trigger a race to the bottom. Consider: can we sustain the margin hit? Do our customers value quality over price? Is this a sustainable strategy for competitor? Multiple valid responses exist.",
            difficulty=0.7,
            domain="business_strategy",
            near_misses=["Yes, match immediately", "No, hold prices", "Cut prices 15%"],
        ),
    ]

    # Repeat to simulate longer training
    return items * 10


def on_cycle(result):
    """Callback for each training cycle."""
    correct = "✓" if result.evaluation.consensus_correct else "✗"
    tokens = result.tokens_earned.total
    surprise = result.misalignment.total_surprise

    print(
        f"[{result.item.id}] {correct} "
        f"tokens={tokens:.2f} "
        f"surprise={surprise:.2f} "
        f"time={result.cycle_time_ms:.0f}ms"
    )

    # Show critiques occasionally
    if result.misalignment.should_have_hedged:
        print(f"  ⚠ Student should have hedged (high disagreement predicted)")


def main():
    print("=" * 60)
    print("ASPIRE Engine Demo")
    print("=" * 60)

    # Create components
    student = MockStudent(correct_rate=0.4)  # Start mediocre
    professors = ProfessorEnsemble()
    critic = HeuristicCritic()

    # Create engine
    engine = ScalarScopeEngine(
        student=student,
        professors=professors,
        critic=critic,
        on_cycle_complete=on_cycle,
    )

    # Run training
    print("\nStarting training loop...\n")
    items = create_test_items()
    metrics = engine.train(iter(items), max_cycles=30)

    # Report
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Total cycles:     {metrics.total_cycles}")
    print(f"Accuracy:         {metrics.accuracy:.1%}")
    print(f"Avg tokens:       {metrics.token_ledger.mean.total:.2f}")
    print(f"Avg surprise:     {metrics.avg_surprise:.3f}")
    print(f"Avg cycle time:   {metrics.avg_cycle_time_ms:.0f}ms")

    print("\nToken breakdown (mean per dimension):")
    for dim, val in metrics.token_ledger.mean.values.items():
        print(f"  {dim.value:15} {val:.2f}")

    print("\n✓ Loop validated - ready for real models")


if __name__ == "__main__":
    main()
