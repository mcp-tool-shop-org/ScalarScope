"""Critic components for ASPIRE."""

from .head import (
    Critic,
    HeuristicCritic,
    LearnedCritic,
    CriticPrediction,
    MisalignmentSignal,
)
from .features import TextFeatureExtractor, FeatureSet
from .features_v1 import TextFeatureExtractorV1, FeatureSetV1
from .learned_critic import LearnedCriticV0, CriticMetrics, OnlineSGDRegressor
from .learned_critic_v1 import LearnedCriticV1, CriticMetricsV1, OnlineLogisticRegressor

__all__ = [
    # Base critic
    "Critic",
    "HeuristicCritic",
    "LearnedCritic",
    "CriticPrediction",
    "MisalignmentSignal",
    # Feature extraction
    "TextFeatureExtractor",
    "FeatureSet",
    "TextFeatureExtractorV1",
    "FeatureSetV1",
    # V0 critic
    "LearnedCriticV0",
    "CriticMetrics",
    "OnlineSGDRegressor",
    # V1 critic (logit-aware)
    "LearnedCriticV1",
    "CriticMetricsV1",
    "OnlineLogisticRegressor",
]
