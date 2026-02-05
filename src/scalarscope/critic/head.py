"""Critic head for predicting token outcomes and misalignment detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from ..core import TokenVector, TokenDimension, StudentResponse, TrainingItem


@dataclass
class CriticPrediction:
    """What the critic predicts before seeing professor evaluations."""
    expected_tokens: TokenVector          # E[tokens_by_dim]
    expected_disagreement: float          # E[var(tokens)] across professors
    confidence: float                     # Critic's own confidence in prediction


@dataclass
class MisalignmentSignal:
    """The 'uh oh' feeling - gap between prediction and reality."""
    surprise: TokenVector                 # actual - predicted (per dimension)
    total_surprise: float                 # Magnitude of misalignment
    overconfidence_penalty: float         # Student confident + predicted disagreement high
    should_have_hedged: bool              # Critic thinks student should have been less certain


class Critic(ABC):
    """Base class for critic heads.

    The critic learns to predict what the professors will say/award.
    This is the "internalization" mechanism - when the critic can
    accurately predict outcomes, the student has internalized judgment.
    """

    @abstractmethod
    def predict(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> CriticPrediction:
        """Predict token outcomes before professor evaluation."""
        pass

    @abstractmethod
    def compute_misalignment(
        self,
        prediction: CriticPrediction,
        actual_tokens: TokenVector,
        actual_disagreement: float,
        student_confidence: float,
    ) -> MisalignmentSignal:
        """Compute the misalignment signal after seeing actuals."""
        pass

    @abstractmethod
    def update(
        self,
        prediction: CriticPrediction,
        actual_tokens: TokenVector,
        actual_disagreement: float,
    ):
        """Update critic based on prediction error."""
        pass


class HeuristicCritic(Critic):
    """Rule-based critic for bootstrapping.

    Uses simple heuristics to predict token outcomes.
    Good enough to test the loop before training a learned critic.
    """

    def __init__(self):
        # Track running statistics for calibration
        self._token_history: List[TokenVector] = []
        self._disagreement_history: List[float] = []

    def predict(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> CriticPrediction:
        # Base predictions on response characteristics
        reasoning = response.reasoning_trace.lower()

        tokens = {}

        # Correctness: hard to predict without knowing answer
        # Use confidence as proxy (calibrated students should be right when confident)
        tokens[TokenDimension.CORRECTNESS] = response.confidence * 0.8

        # Coherence: look for red flags
        contradiction_words = ["however", "but", "although", "yet"]
        resolution_words = ["therefore", "thus", "so", "conclude"]
        has_contradictions = any(w in reasoning for w in contradiction_words)
        has_resolution = any(w in reasoning for w in resolution_words)

        if has_contradictions and not has_resolution:
            tokens[TokenDimension.COHERENCE] = 0.4
        elif has_resolution:
            tokens[TokenDimension.COHERENCE] = 0.8
        else:
            tokens[TokenDimension.COHERENCE] = 0.6

        # Tradeoffs: did they acknowledge alternatives?
        tradeoff_words = ["tradeoff", "alternatively", "risk", "downside", "cost"]
        tradeoff_count = sum(1 for w in tradeoff_words if w in reasoning)
        tokens[TokenDimension.TRADEOFFS] = min(0.3 + tradeoff_count * 0.25, 1.0)

        # Calibration: does confidence match reasoning quality?
        reasoning_quality = (tokens[TokenDimension.COHERENCE] + tokens[TokenDimension.TRADEOFFS]) / 2
        confidence_gap = abs(response.confidence - reasoning_quality)
        tokens[TokenDimension.CALIBRATION] = max(0, 0.9 - confidence_gap)

        # Clarity: length-based heuristic
        word_count = len(reasoning.split())
        if 50 <= word_count <= 200:
            tokens[TokenDimension.CLARITY] = 0.8
        elif word_count < 50:
            tokens[TokenDimension.CLARITY] = 0.5  # Too brief
        else:
            tokens[TokenDimension.CLARITY] = 0.6  # Too verbose

        # Disagreement prediction: harder problems = more disagreement
        expected_disagreement = item.difficulty * 0.5 + 0.1

        # Confidence in our prediction
        if self._token_history:
            prediction_confidence = 0.7
        else:
            prediction_confidence = 0.3  # Low confidence early on

        return CriticPrediction(
            expected_tokens=TokenVector(tokens),
            expected_disagreement=expected_disagreement,
            confidence=prediction_confidence,
        )

    def compute_misalignment(
        self,
        prediction: CriticPrediction,
        actual_tokens: TokenVector,
        actual_disagreement: float,
        student_confidence: float,
    ) -> MisalignmentSignal:
        # Compute surprise per dimension
        surprise = {}
        for dim in TokenDimension:
            surprise[dim] = actual_tokens.values[dim] - prediction.expected_tokens.values[dim]

        surprise_vector = TokenVector(surprise)

        # Total surprise magnitude
        total_surprise = math.sqrt(sum(v ** 2 for v in surprise.values()))

        # Overconfidence penalty
        # If student was confident but there's high disagreement, that's bad
        overconfidence = 0.0
        if student_confidence > 0.7 and actual_disagreement > 0.3:
            overconfidence = (student_confidence - 0.5) * actual_disagreement

        # Should have hedged?
        should_hedge = prediction.expected_disagreement > 0.4 and student_confidence > 0.6

        return MisalignmentSignal(
            surprise=surprise_vector,
            total_surprise=total_surprise,
            overconfidence_penalty=overconfidence,
            should_have_hedged=should_hedge,
        )

    def update(
        self,
        prediction: CriticPrediction,
        actual_tokens: TokenVector,
        actual_disagreement: float,
    ):
        # Track history for better predictions
        self._token_history.append(actual_tokens)
        self._disagreement_history.append(actual_disagreement)

        # Keep bounded
        if len(self._token_history) > 1000:
            self._token_history = self._token_history[-500:]
            self._disagreement_history = self._disagreement_history[-500:]


class LearnedCritic(Critic):
    """Neural critic that learns to predict token outcomes.

    This is where the real internalization happens - the critic
    becomes a learned model of what the professors value.
    """

    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim
        self._model = None  # PyTorch model, loaded lazily

    def _build_model(self):
        """Build the critic network."""
        import torch
        import torch.nn as nn

        class CriticNetwork(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.token_head = nn.Linear(hidden_dim, output_dim)
                self.disagreement_head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                h = self.encoder(x)
                tokens = torch.sigmoid(self.token_head(h))
                disagreement = torch.sigmoid(self.disagreement_head(h))
                return tokens, disagreement

        # Input: response features (to be defined based on actual model outputs)
        # Output: token predictions + disagreement
        input_dim = 64  # Placeholder
        output_dim = len(TokenDimension)

        self._model = CriticNetwork(input_dim, self.hidden_dim, output_dim)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

    def predict(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> CriticPrediction:
        if self._model is None:
            self._build_model()

        # TODO: Extract features from response
        # TODO: Run through model
        raise NotImplementedError("Learned critic inference not yet implemented")

    def compute_misalignment(
        self,
        prediction: CriticPrediction,
        actual_tokens: TokenVector,
        actual_disagreement: float,
        student_confidence: float,
    ) -> MisalignmentSignal:
        # Same logic as heuristic critic
        surprise = {}
        for dim in TokenDimension:
            surprise[dim] = actual_tokens.values[dim] - prediction.expected_tokens.values[dim]

        surprise_vector = TokenVector(surprise)
        total_surprise = math.sqrt(sum(v ** 2 for v in surprise.values()))

        overconfidence = 0.0
        if student_confidence > 0.7 and actual_disagreement > 0.3:
            overconfidence = (student_confidence - 0.5) * actual_disagreement

        should_hedge = prediction.expected_disagreement > 0.4 and student_confidence > 0.6

        return MisalignmentSignal(
            surprise=surprise_vector,
            total_surprise=total_surprise,
            overconfidence_penalty=overconfidence,
            should_have_hedged=should_hedge,
        )

    def update(
        self,
        prediction: CriticPrediction,
        actual_tokens: TokenVector,
        actual_disagreement: float,
    ):
        if self._model is None:
            return

        # TODO: Compute loss and backprop
        # Loss = MSE(predicted_tokens, actual_tokens) + MSE(pred_disagreement, actual)
        raise NotImplementedError("Learned critic training not yet implemented")
