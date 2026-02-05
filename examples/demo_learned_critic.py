"""Demo: LearnedCritic v0 with text features.

Shows how the critic learns to predict professor evaluations over time,
demonstrating the "internalization" of judgment.

Key metrics to watch:
- Loss trend: Should decrease as critic learns
- Negative surprise rate: Should decrease
- MAE per dimension: Should decrease
- Feature importance: Shows what the critic learned matters

Usage:
    python examples/demo_learned_critic.py --cycles 200
"""

import argparse
from typing import List
import random

from scalarscope.core import TrainingItem, TokenVector, TokenDimension
from scalarscope.student import MockStudent
from scalarscope.professors import ProfessorEnsemble
from scalarscope.critic import LearnedCriticV0
from scalarscope.engine import ScalarScopeEngine


def create_diverse_test_items(count: int = 100) -> List[TrainingItem]:
    """Create diverse test items for training the critic."""
    templates = [
        # Ethics (high difficulty, high disagreement expected)
        {
            "prompt": "A company discovers their AI has bias against minorities. Should they pause deployment to fix it, knowing competitors will gain market share?",
            "gold_answer": "Pause and fix - ethical obligations outweigh short-term market concerns",
            "gold_rationale": "Deploying biased AI causes harm. Tradeoff between ethics and business.",
            "difficulty": 0.9,
            "domain": "ethics",
        },
        {
            "prompt": "Is it ethical to use AI-generated art in commercial products without disclosing it?",
            "gold_answer": "Disclosure is ethically preferred but legally complex",
            "gold_rationale": "Transparency vs creative ownership tension.",
            "difficulty": 0.7,
            "domain": "ethics",
        },
        # Code review (medium difficulty)
        {
            "prompt": "This PR adds a 500-line function. Should we merge it?",
            "gold_answer": "Request refactoring into smaller functions",
            "gold_rationale": "Large functions are hard to test and maintain.",
            "difficulty": 0.4,
            "domain": "code_review",
        },
        {
            "prompt": "The tests pass but code coverage dropped from 80% to 65%. Merge?",
            "gold_answer": "Request additional tests before merge",
            "gold_rationale": "Coverage drop indicates untested code paths.",
            "difficulty": 0.5,
            "domain": "code_review",
        },
        # Architecture (medium-high difficulty)
        {
            "prompt": "Should we migrate from REST to GraphQL for our mobile app?",
            "gold_answer": "Consider hybrid approach - GraphQL for complex queries, REST for simple ones",
            "gold_rationale": "Migration cost vs flexibility tradeoff.",
            "difficulty": 0.6,
            "domain": "architecture",
        },
        {
            "prompt": "Our monolith is 500k lines. Time to split into microservices?",
            "gold_answer": "Extract bounded contexts incrementally, not big bang",
            "gold_rationale": "Big rewrites fail. Incremental migration safer.",
            "difficulty": 0.7,
            "domain": "architecture",
        },
        # Business strategy (high disagreement)
        {
            "prompt": "Should we open-source our core product to compete with a well-funded rival?",
            "gold_answer": "Depends on business model and community potential",
            "gold_rationale": "Open source can build moat or destroy revenue.",
            "difficulty": 0.8,
            "domain": "business",
        },
        {
            "prompt": "Layoffs vs salary cuts across the board - which is more ethical?",
            "gold_answer": "Context-dependent - consider employee needs and company survival",
            "gold_rationale": "Both have tradeoffs. No clear right answer.",
            "difficulty": 0.9,
            "domain": "business",
        },
        # Simple questions (low difficulty, low disagreement)
        {
            "prompt": "Should we use TypeScript or JavaScript for a new project?",
            "gold_answer": "TypeScript for larger projects, JavaScript for quick prototypes",
            "gold_rationale": "Type safety vs development speed.",
            "difficulty": 0.3,
            "domain": "technical",
        },
        {
            "prompt": "Is it better to use Postgres or MySQL for a typical web app?",
            "gold_answer": "Both work well; Postgres has more features, MySQL is simpler",
            "gold_rationale": "Standard databases, minor differences.",
            "difficulty": 0.2,
            "domain": "technical",
        },
    ]

    items = []
    for i in range(count):
        template = random.choice(templates)
        items.append(TrainingItem(
            id=f"{template['domain']}_{i:04d}",
            prompt=template["prompt"],
            gold_answer=template["gold_answer"],
            gold_rationale=template["gold_rationale"],
            difficulty=template["difficulty"],
            domain=template["domain"],
        ))

    return items


def print_progress(cycle: int, result, critic: LearnedCriticV0, interval: int = 20):
    """Print progress at intervals."""
    if cycle % interval != 0:
        return

    metrics = critic.get_metrics_summary()

    print(f"\n{'='*60}")
    print(f"Cycle {cycle}")
    print(f"{'='*60}")
    print(f"Critic updates: {metrics['total_updates']}")
    print(f"Avg loss:       {metrics['avg_loss']:.4f}")
    print(f"Neg surprise:   {metrics['negative_surprise_rate']:.1%}")

    print(f"\nMAE per dimension:")
    for dim, mae in metrics['mae_per_dim'].items():
        bar = "█" * int(mae * 20)
        print(f"  {dim:15} {mae:.3f} {bar}")

    print(f"  {'disagreement':15} {metrics['disagreement_mae']:.3f}")


def print_final_report(critic: LearnedCriticV0, metrics):
    """Print final learning report."""
    print("\n" + "=" * 70)
    print("LEARNING REPORT")
    print("=" * 70)

    # Overall metrics
    critic_metrics = critic.get_metrics_summary()
    print(f"\nCritic Performance:")
    print(f"  Total updates:       {critic_metrics['total_updates']}")
    print(f"  Final avg loss:      {critic_metrics['avg_loss']:.4f}")
    print(f"  Neg surprise rate:   {critic_metrics['negative_surprise_rate']:.1%}")

    # Loss trend
    loss_history = critic.metrics.loss_history
    if len(loss_history) > 20:
        early_loss = sum(loss_history[:20]) / 20
        late_loss = sum(loss_history[-20:]) / 20
        improvement = (early_loss - late_loss) / (early_loss + 0.001)
        print(f"\nLoss Improvement:")
        print(f"  Early avg loss:  {early_loss:.4f}")
        print(f"  Late avg loss:   {late_loss:.4f}")
        print(f"  Improvement:     {improvement:.1%}")

    # Feature importance
    print(f"\nTop Features per Dimension:")
    top_features = critic.get_top_features(5)
    for dim_name, features in top_features.items():
        print(f"\n  {dim_name}:")
        for name, weight in features[:5]:
            print(f"    {name:30} {weight:.4f}")

    # Training metrics
    print(f"\nTraining Metrics:")
    print(f"  Total cycles:    {metrics.total_cycles}")
    print(f"  Accuracy:        {metrics.accuracy:.1%}")
    print(f"  Avg surprise:    {metrics.avg_surprise:.3f}")

    # Internalization assessment
    print(f"\n{'='*70}")
    print("INTERNALIZATION ASSESSMENT")
    print("{'='*70}")

    if critic_metrics['avg_loss'] < 0.05:
        print("✓ Critic has strongly internalized professor judgment patterns")
    elif critic_metrics['avg_loss'] < 0.1:
        print("◐ Critic has moderately internalized judgment patterns")
    else:
        print("○ Critic is still learning - more training recommended")

    if critic_metrics['negative_surprise_rate'] < 0.1:
        print("✓ Low negative surprise rate - predictions well calibrated")
    elif critic_metrics['negative_surprise_rate'] < 0.2:
        print("◐ Moderate surprise rate - room for improvement")
    else:
        print("○ High surprise rate - critic needs more training")


def main():
    parser = argparse.ArgumentParser(description="LearnedCritic Demo")
    parser.add_argument("--cycles", type=int, default=200, help="Training cycles")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="Critic LR")
    parser.add_argument("--print-interval", type=int, default=25, help="Print every N cycles")
    parser.add_argument("--save-critic", type=str, default=None, help="Save critic to file")
    args = parser.parse_args()

    print("=" * 70)
    print("ASPIRE LearnedCritic v0 Demo")
    print("=" * 70)
    print(f"\nThis demo shows how the critic learns to predict professor evaluations")
    print(f"using only text features (no embeddings). Watch for:")
    print(f"  - Decreasing loss over time")
    print(f"  - Decreasing negative surprise rate")
    print(f"  - Feature importance revealing what matters")

    # Create components
    student = MockStudent(correct_rate=0.5)
    professors = ProfessorEnsemble()
    critic = LearnedCriticV0(learning_rate=args.learning_rate)

    # Create engine
    cycle_count = [0]

    def on_cycle(result):
        cycle_count[0] += 1
        print_progress(cycle_count[0], result, critic, args.print_interval)

    engine = ScalarScopeEngine(
        student=student,
        professors=professors,
        critic=critic,
        on_cycle_complete=on_cycle,
    )

    # Create diverse training items
    print(f"\nGenerating {args.cycles} training items...")
    items = create_diverse_test_items(args.cycles)

    # Run training
    print(f"\nStarting training loop...")
    print(f"(Progress shown every {args.print_interval} cycles)")

    metrics = engine.train(iter(items), max_cycles=args.cycles)

    # Final report
    print_final_report(critic, metrics)

    # Save if requested
    if args.save_critic:
        critic.save(args.save_critic)
        print(f"\nCritic saved to: {args.save_critic}")


if __name__ == "__main__":
    main()
