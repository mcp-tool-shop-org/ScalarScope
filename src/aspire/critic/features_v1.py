"""Extended feature extraction for LearnedCritic V1.

V1 adds logit-derived features from the student model's generation stats.
These capture model-internal uncertainty that text-only features cannot detect.

Key additions:
- Entropy (model uncertainty per token)
- Margin (top-1 vs top-2 confidence)
- EOS probability (model "wanting to stop")
- Vocabulary diversity

These signals are hard to game because they come from the model's own
probability distribution, not from text that the model generates.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ..core import TrainingItem, StudentResponse
from ..student.onnx_student_v2 import GenerationStats
from .features import TextFeatureExtractor, FeatureSet


@dataclass
class FeatureSetV1(FeatureSet):
    """Extended feature set with logit features."""
    has_logit_features: bool = False


class TextFeatureExtractorV1(TextFeatureExtractor):
    """Extended feature extractor with logit-derived features.

    V1 adds ~20 features from generation statistics:

    Entropy features (model uncertainty):
    - entropy_mean: Average uncertainty across tokens
    - entropy_std: Variability in uncertainty
    - entropy_max: Peak uncertainty moment
    - entropy_min: Minimum uncertainty (most confident moment)

    Margin features (decision confidence):
    - margin_mean: Average gap between top-1 and top-2
    - margin_min: Smallest gap (hardest decision)

    EOS features (completion behavior):
    - eos_prob_max: Peak "want to stop" signal
    - eos_prob_mean_last_5: Recent stopping tendency

    Top-1 features (selection confidence):
    - top1_prob_mean: Average confidence in chosen token
    - top1_prob_min: Lowest confidence choice

    Token patterns:
    - repeat_bigram_rate: Repetition indicator
    - unique_token_ratio: Vocabulary diversity

    Interaction features (combining signals):
    - difficulty_entropy_mean: Hard items should have high entropy
    - confidence_entropy_mean: High confidence but high entropy = overconfident
    - hedge_entropy_interaction: Hedging text but low entropy = fake hedging
    """

    # Logit feature names (added to base features)
    LOGIT_FEATURE_NAMES = [
        "entropy_mean",
        "entropy_std",
        "entropy_max",
        "entropy_min",
        "margin_mean",
        "margin_min",
        "eos_prob_max",
        "eos_prob_mean_last_5",
        "top1_prob_mean",
        "top1_prob_min",
        "repeat_bigram_rate",
        "unique_token_ratio",
        # Interaction features
        "difficulty_entropy_mean",
        "confidence_entropy_mean",
        "hedge_entropy_interaction",
        "tradeoff_margin_interaction",
        "confidence_margin_interaction",
        "entropy_variance_ratio",
    ]

    def __init__(self):
        super().__init__()
        self._feature_names_v1: Optional[List[str]] = None

    @property
    def feature_names_v1(self) -> List[str]:
        """Get ordered list of all feature names including logit features."""
        if self._feature_names_v1 is None:
            self._feature_names_v1 = self.feature_names + self.LOGIT_FEATURE_NAMES
        return self._feature_names_v1

    @property
    def num_features_v1(self) -> int:
        """Total number of features including logit features."""
        return len(self.feature_names_v1)

    def extract_v1(
        self,
        item: TrainingItem,
        response: StudentResponse,
        generation_stats: Optional[GenerationStats] = None,
    ) -> FeatureSetV1:
        """Extract all features including logit-derived features.

        Args:
            item: Training item
            response: Student response
            generation_stats: Optional logit statistics from generation.
                             If None, logit features are set to defaults.

        Returns:
            Extended feature set with has_logit_features flag
        """
        # Get base text features
        base_features = self.extract(item, response)
        features = list(base_features.values)
        names = list(base_features.names)

        # Use generation_stats from response if available
        stats = generation_stats or getattr(response, 'generation_stats', None)

        if stats is not None and hasattr(stats, 'entropy_mean'):
            # Logit features available
            has_logit = True

            # Entropy features
            features.append(stats.entropy_mean)
            names.append("entropy_mean")

            features.append(stats.entropy_std)
            names.append("entropy_std")

            features.append(stats.entropy_max)
            names.append("entropy_max")

            features.append(stats.entropy_min if stats.entropy_min < 1e8 else 0.0)
            names.append("entropy_min")

            # Margin features
            features.append(stats.margin_mean)
            names.append("margin_mean")

            features.append(stats.margin_min)
            names.append("margin_min")

            # EOS features
            features.append(stats.eos_prob_max)
            names.append("eos_prob_max")

            features.append(stats.eos_prob_mean_last_5)
            names.append("eos_prob_mean_last_5")

            # Top-1 features
            features.append(stats.top1_prob_mean)
            names.append("top1_prob_mean")

            features.append(stats.top1_prob_min)
            names.append("top1_prob_min")

            # Token patterns
            features.append(stats.repeat_bigram_rate)
            names.append("repeat_bigram_rate")

            features.append(stats.unique_token_ratio)
            names.append("unique_token_ratio")

            # Interaction features
            # Hard items should have high entropy - if not, student may be overconfident
            features.append(item.difficulty * stats.entropy_mean)
            names.append("difficulty_entropy_mean")

            # High confidence + high entropy = inconsistent = overconfident
            features.append(response.confidence * stats.entropy_mean)
            names.append("confidence_entropy_mean")

            # Get hedge count from base features
            hedge_idx = base_features.names.index("hedge_count") if "hedge_count" in base_features.names else -1
            hedge_count = float(base_features.values[hedge_idx]) if hedge_idx >= 0 else 0.0

            # Hedging text but low entropy = fake hedging (text says uncertain but model isn't)
            features.append(hedge_count * (1.0 - stats.entropy_mean))
            names.append("hedge_entropy_interaction")

            # Get tradeoff count from base features
            tradeoff_idx = base_features.names.index("tradeoff_count") if "tradeoff_count" in base_features.names else -1
            tradeoff_count = float(base_features.values[tradeoff_idx]) if tradeoff_idx >= 0 else 0.0

            # Tradeoffs + low margin = genuinely considering alternatives
            features.append(tradeoff_count * (1.0 - stats.margin_mean))
            names.append("tradeoff_margin_interaction")

            # High confidence + high margin = genuinely confident
            features.append(response.confidence * stats.margin_mean)
            names.append("confidence_margin_interaction")

            # Entropy variance ratio (how consistent is uncertainty)
            if stats.entropy_mean > 0:
                features.append(stats.entropy_std / (stats.entropy_mean + 1e-6))
            else:
                features.append(0.0)
            names.append("entropy_variance_ratio")

        else:
            # No logit features - use defaults (zeros with small noise to avoid dead regressors)
            has_logit = False
            for feat_name in self.LOGIT_FEATURE_NAMES:
                features.append(0.0)
                names.append(feat_name)

        return FeatureSetV1(
            values=np.array(features, dtype=np.float32),
            names=names,
            has_logit_features=has_logit,
        )

    def extract_batch_v1(
        self,
        items: List[TrainingItem],
        responses: List[StudentResponse],
        generation_stats_list: Optional[List[Optional[GenerationStats]]] = None,
    ) -> np.ndarray:
        """Extract V1 features for a batch of items/responses."""
        if generation_stats_list is None:
            generation_stats_list = [None] * len(items)

        return np.array([
            self.extract_v1(item, resp, stats).values
            for item, resp, stats in zip(items, responses, generation_stats_list)
        ])
