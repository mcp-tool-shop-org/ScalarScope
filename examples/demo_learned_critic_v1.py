"""Demo: LearnedCritic V1 with logit-derived features.

Shows how the V1 critic uses model-internal uncertainty signals
(entropy, margin, EOS prob) to improve judgment prediction.

Key improvements over V0:
1. Detects fake hedging (text says uncertain, model isn't)
2. Better overconfidence detection via entropy
3. Learns when revision will actually help

Usage:
    python examples/demo_learned_critic_v1.py --cycles 50
"""

import argparse
import random
from typing import List, Optional
from dataclasses import dataclass

from scalarscope.core import TrainingItem, StudentResponse, TokenVector, TokenDimension
from scalarscope.student import MockStudent, GenerationStats, TokenStats
from scalarscope.professors import ProfessorEnsemble
from scalarscope.critic import LearnedCriticV1, TextFeatureExtractorV1


@dataclass
class MockGenerationStats(GenerationStats):
    """Mock generation stats for testing without real ONNX model."""
    pass


def create_mock_generation_stats(
    response: StudentResponse,
    is_uncertain: bool = False,
    is_overconfident: bool = False,
    is_repetitive: bool = False,
) -> GenerationStats:
    """Create mock generation stats based on response characteristics.

    This simulates what a real model would produce based on its
    internal uncertainty state.
    """
    stats = GenerationStats()

    # Simulate 50-100 tokens of stats
    n_tokens = random.randint(50, 100)

    for _ in range(n_tokens):
        if is_uncertain:
            # High entropy, low margin
            entropy = random.uniform(2.0, 4.0)
            margin = random.uniform(0.05, 0.2)
            top1 = random.uniform(0.2, 0.4)
        elif is_overconfident:
            # Low entropy but actually wrong (revealed later)
            entropy = random.uniform(0.3, 0.8)
            margin = random.uniform(0.6, 0.9)
            top1 = random.uniform(0.7, 0.95)
        else:
            # Normal calibrated model
            entropy = random.uniform(0.5, 2.0)
            margin = random.uniform(0.3, 0.7)
            top1 = random.uniform(0.4, 0.8)

        eos_prob = random.uniform(0.001, 0.05)

        stats.token_stats.append(TokenStats(
            entropy=entropy,
            margin=margin,
            top1_prob=top1,
            eos_prob=eos_prob,
        ))

    # Add repetition if specified
    if is_repetitive:
        # Simulate some repeated patterns
        for i in range(5):
            if len(stats.token_stats) > i + 10:
                stats.token_stats[i + 10] = stats.token_stats[i]

    # Compute aggregates
    token_ids = list(range(n_tokens))  # Simplified
    if is_repetitive:
        # Make some token IDs repeat
        for i in range(10, min(20, n_tokens)):
            token_ids[i] = token_ids[i - 10]

    stats.compute_aggregates(token_ids, eos_token_id=2)

    return stats


def create_test_scenarios() -> List[dict]:
    """Create diverse test scenarios with expected behaviors."""
    return [
        # Scenario 1: Genuinely uncertain response (should NOT be flagged as overconfident)
        {
            "name": "genuinely_uncertain",
            "prompt": "Is NoSQL or SQL better for a new e-commerce platform?",
            "gold": "Depends on specific requirements - SQL for transactions, NoSQL for catalog",
            "response_answer": "I think it depends on the use case...",
            "response_reasoning": "Perhaps SQL for transactions, but maybe NoSQL could work too...",
            "confidence": 0.4,
            "is_uncertain": True,
            "expected_revision": False,
            "description": "Genuinely uncertain - text matches model state",
        },
        # Scenario 2: Fake hedging (text hedges but model is confident)
        {
            "name": "fake_hedging",
            "prompt": "Should we use microservices or monolith?",
            "gold": "Context-dependent - consider team size and complexity",
            "response_answer": "Perhaps microservices might be better, but I'm not sure...",
            "response_reasoning": "Maybe we should consider microservices, although I'm uncertain...",
            "confidence": 0.5,
            "is_uncertain": False,  # Model is actually confident
            "is_overconfident": True,
            "expected_revision": True,
            "description": "Fake hedging - text says uncertain but model isn't",
        },
        # Scenario 3: Overconfident wrong answer
        {
            "name": "overconfident_wrong",
            "prompt": "Should passwords be stored in plain text for debugging?",
            "gold": "No, never store passwords in plain text",
            "response_answer": "Yes, for debugging purposes this is acceptable",
            "response_reasoning": "Storing in plain text makes debugging easier and is fine for internal use",
            "confidence": 0.9,
            "is_overconfident": True,
            "expected_revision": True,
            "description": "Overconfident wrong - high confidence on incorrect answer",
        },
        # Scenario 4: Appropriate confidence on clear answer
        {
            "name": "appropriate_confidence",
            "prompt": "Should unit tests cover financial calculations?",
            "gold": "Yes, financial code needs high test coverage",
            "response_answer": "Yes, definitely test financial calculations thoroughly",
            "response_reasoning": "Bugs in financial code can have serious consequences",
            "confidence": 0.85,
            "is_uncertain": False,
            "is_overconfident": False,
            "expected_revision": False,
            "description": "Appropriate confidence - model and text align on certainty",
        },
        # Scenario 5: Repetitive/low quality response
        {
            "name": "repetitive_response",
            "prompt": "How should we handle error logging?",
            "gold": "Use structured logging with appropriate severity levels",
            "response_answer": "Log errors to console and also log to file and log to database...",
            "response_reasoning": "We should log errors. Log them well. Make sure to log all errors.",
            "confidence": 0.6,
            "is_repetitive": True,
            "expected_revision": True,
            "description": "Repetitive response - bigram repetition indicates low quality",
        },
    ]


def run_demo(num_cycles: int, verbose: bool = False):
    """Run the V1 critic demo."""
    print("=" * 80)
    print("LearnedCritic V1 Demo: Logit-Aware Judgment")
    print("=" * 80)
    print("""
This demo shows how V1 uses model-internal signals (entropy, margin, etc.)
to detect issues that text-only features miss:

- Fake hedging: Text says "maybe" but model is internally confident
- Overconfidence: High confidence on wrong answer (low entropy but wrong)
- Repetition: Detected via token-level patterns

Watch for:
- V1 detecting mismatches between text and model state
- Learned revision head improving over time
- Logit feature importance growing
""")

    # Create components
    critic = LearnedCriticV1(
        learning_rate=0.02,
        min_samples_before_predict=5,
        enable_revision_learning=True,
    )
    professors = ProfessorEnsemble()
    scenarios = create_test_scenarios()

    print(f"\nRunning {num_cycles} cycles with {len(scenarios)} scenario types...")
    print("-" * 80)

    revision_actual_help = []
    revision_predictions = []

    for cycle in range(num_cycles):
        # Pick a scenario
        scenario = random.choice(scenarios)

        # Create training item
        item = TrainingItem(
            id=f"{scenario['name']}_{cycle:03d}",
            prompt=scenario["prompt"],
            gold_answer=scenario["gold"],
            gold_rationale="See gold answer",
            difficulty=random.uniform(0.3, 0.8),
            domain="technical",
        )

        # Create response with mock generation stats
        response = StudentResponse(
            item_id=item.id,
            answer=scenario["response_answer"],
            reasoning_trace=scenario["response_reasoning"],
            confidence=scenario["confidence"],
        )

        # Create mock generation stats
        gen_stats = create_mock_generation_stats(
            response,
            is_uncertain=scenario.get("is_uncertain", False),
            is_overconfident=scenario.get("is_overconfident", False),
            is_repetitive=scenario.get("is_repetitive", False),
        )

        # Critic predicts
        prediction = critic.predict(item, response, gen_stats)

        # Also predict revision
        should_revise, revise_prob = critic.predict_should_revise(item, response, gen_stats)

        # Professors evaluate
        evaluation = professors.evaluate(item, response)

        # Compute misalignment
        misalignment = critic.compute_misalignment(
            prediction,
            evaluation.aggregated_tokens,
            evaluation.disagreement_score,
            response.confidence,
        )

        # Update critic
        critic.update(
            prediction,
            evaluation.aggregated_tokens,
            evaluation.disagreement_score,
        )

        # Simulate revision outcome
        revision_would_help = scenario.get("expected_revision", False)
        revision_actual_help.append(revision_would_help)
        revision_predictions.append(should_revise)

        # Update revision head
        critic.update_revision_head(item, response, revision_would_help, gen_stats)

        # Print details for select cycles
        if verbose or cycle % 10 == 0 or cycle < 5:
            print(f"\n[Cycle {cycle + 1}] {scenario['name']}")
            print(f"  Scenario: {scenario['description']}")
            print(f"  Confidence: {response.confidence:.2f} (text)")
            print(f"  Entropy: {gen_stats.entropy_mean:.2f} (model internal)")
            print(f"  Margin: {gen_stats.margin_mean:.2f} (decision confidence)")
            print(f"  Predicted tokens: {prediction.expected_tokens.total:.2f}")
            print(f"  Actual tokens: {evaluation.aggregated_tokens.total:.2f}")
            print(f"  Surprise: {misalignment.total_surprise:.2f}")
            print(f"  Revision pred: {should_revise} (prob={revise_prob:.2f})")
            print(f"  Would help: {revision_would_help}")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    metrics = critic.get_metrics_summary()
    print(f"\nOverall Metrics:")
    print(f"  Updates: {metrics['total_updates']}")
    print(f"  Predictions: {metrics['total_predictions']}")
    print(f"  Avg loss: {metrics['avg_loss']:.4f}")
    print(f"  Negative surprise rate: {metrics['negative_surprise_rate']:.1%}")
    print(f"  Logit feature rate: {metrics['logit_feature_rate']:.1%}")

    print(f"\nRevision Learning:")
    print(f"  Revision accuracy: {metrics['revision_accuracy']:.1%}")
    correct_predictions = sum(
        1 for pred, actual in zip(revision_predictions, revision_actual_help)
        if pred == actual
    )
    print(f"  Overall revision prediction: {correct_predictions}/{len(revision_predictions)} correct")

    print(f"\nMAE per Token Dimension:")
    for dim, mae in metrics['mae_per_dim'].items():
        print(f"  {dim:15} {mae:.3f}")
    print(f"  {'disagreement':15} {metrics['disagreement_mae']:.3f}")

    # Feature importance
    print("\nTop Logit Feature Importance:")
    logit_importance = critic.get_logit_feature_importance()
    for dim in list(TokenDimension)[:2]:  # Just show first 2 dimensions
        print(f"\n  {dim.value}:")
        sorted_features = sorted(
            logit_importance[dim.value].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for name, importance in sorted_features:
            bar = "█" * int(importance * 20)
            print(f"    {name:30} {importance:.3f} {bar}")

    print(f"\n  revision head:")
    sorted_features = sorted(
        logit_importance["revision"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for name, importance in sorted_features:
        bar = "█" * int(importance * 20)
        print(f"    {name:30} {importance:.3f} {bar}")

    # V1 vs V0 comparison insight
    print("\n" + "=" * 80)
    print("V1 vs V0 INSIGHT")
    print("=" * 80)
    print("""
V1 critic uses logit-derived features that V0 cannot access:

1. FAKE HEDGING DETECTION
   V0: Sees hedge words, assumes uncertainty
   V1: Sees hedge words + low entropy = fake hedging

2. OVERCONFIDENCE DETECTION
   V0: Relies on stated confidence vs correctness
   V1: Also checks entropy - high confidence + high entropy = suspicious

3. REVISION PREDICTION
   V0: Uses fixed thresholds (surprise > 0.4, disagreement > 0.35)
   V1: Learns when revision actually helps from data

The key insight: models can write "I'm uncertain" while being
internally confident. Logit features reveal the truth.
""")


def main():
    parser = argparse.ArgumentParser(description="LearnedCritic V1 Demo")
    parser.add_argument("--cycles", type=int, default=50, help="Training cycles")
    parser.add_argument("--verbose", action="store_true", help="Show all cycles")
    args = parser.parse_args()

    run_demo(args.cycles, args.verbose)


if __name__ == "__main__":
    main()
