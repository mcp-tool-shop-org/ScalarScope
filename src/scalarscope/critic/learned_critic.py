"""LearnedCritic v0: Text-feature based online learning critic."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import numpy as np
from collections import deque

from ..core import (
    TokenVector,
    TokenDimension,
    TrainingItem,
    StudentResponse,
)
from .head import Critic, CriticPrediction, MisalignmentSignal
from .features import TextFeatureExtractor, FeatureSet


@dataclass
class CriticMetrics:
    """Metrics tracking critic learning progress."""
    total_updates: int = 0
    mae_per_dim: Dict[str, float] = field(default_factory=dict)
    disagreement_mae: float = 0.0
    negative_surprise_count: int = 0
    total_predictions: int = 0

    # Rolling history for trend tracking
    loss_history: List[float] = field(default_factory=list)
    surprise_history: List[float] = field(default_factory=list)

    @property
    def avg_loss(self) -> float:
        if not self.loss_history:
            return 0.0
        return sum(self.loss_history[-100:]) / min(len(self.loss_history), 100)

    @property
    def negative_surprise_rate(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.negative_surprise_count / self.total_predictions


class OnlineSGDRegressor:
    """Simple online SGD regressor (no sklearn dependency).

    Supports partial_fit for online learning from streaming data.
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        l2_reg: float = 0.001,
    ):
        self.n_features = n_features
        self.lr = learning_rate
        self.l2_reg = l2_reg

        # Initialize weights
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0

        # Running statistics for feature normalization
        self._mean = np.zeros(n_features, dtype=np.float32)
        self._var = np.ones(n_features, dtype=np.float32)
        self._count = 0

    def _update_stats(self, x: np.ndarray):
        """Update running mean/variance for normalization."""
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._var += (delta * delta2 - self._var) / self._count

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics."""
        std = np.sqrt(self._var + 1e-8)
        return (x - self._mean) / std

    def predict(self, x: np.ndarray) -> float:
        """Predict target value."""
        if self._count > 10:
            x = self._normalize(x)
        return float(np.dot(self.weights, x) + self.bias)

    def partial_fit(self, x: np.ndarray, y: float):
        """Update weights with single example."""
        self._update_stats(x)

        if self._count > 10:
            x = self._normalize(x)

        # Predict
        pred = np.dot(self.weights, x) + self.bias
        error = pred - y

        # Gradient descent with L2 regularization
        self.weights -= self.lr * (error * x + self.l2_reg * self.weights)
        self.bias -= self.lr * error

    def get_loss(self, x: np.ndarray, y: float) -> float:
        """Compute squared error loss."""
        pred = self.predict(x)
        return (pred - y) ** 2


class LearnedCriticV0(Critic):
    """Text-feature based learned critic with online SGD.

    V0 uses cheap text features (no embeddings) to predict:
    - Token awards per dimension
    - Professor disagreement
    - Whether the answer is correct

    Updates online after each cycle, so it improves during training.

    The "gut feeling" is the prediction error:
    - High negative surprise = predicted high tokens but got low
    - High ambiguity = predicted high disagreement
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        l2_reg: float = 0.001,
        min_samples_before_predict: int = 10,
    ):
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.min_samples = min_samples_before_predict

        # Feature extractor
        self.feature_extractor = TextFeatureExtractor()
        n_features = self.feature_extractor.num_features

        # One regressor per token dimension
        self.token_regressors: Dict[TokenDimension, OnlineSGDRegressor] = {
            dim: OnlineSGDRegressor(n_features, learning_rate, l2_reg)
            for dim in TokenDimension
        }

        # Regressor for disagreement
        self.disagreement_regressor = OnlineSGDRegressor(n_features, learning_rate, l2_reg)

        # Binary classifier for correctness (logistic regression style)
        self.correctness_regressor = OnlineSGDRegressor(n_features, learning_rate, l2_reg)

        # Metrics
        self.metrics = CriticMetrics()

        # Cache last prediction for update
        self._last_features: Optional[np.ndarray] = None
        self._last_prediction: Optional[CriticPrediction] = None

    def predict(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> CriticPrediction:
        """Predict token outcomes before seeing professor evaluation."""
        # Extract features
        features = self.feature_extractor.extract(item, response)
        self._last_features = features.values

        # If not enough training data, return prior
        if self.metrics.total_updates < self.min_samples:
            # Prior: assume moderate values
            expected_tokens = TokenVector({dim: 0.5 for dim in TokenDimension})
            expected_disagreement = 0.3
            confidence = 0.2  # Low confidence early

            pred = CriticPrediction(
                expected_tokens=expected_tokens,
                expected_disagreement=expected_disagreement,
                confidence=confidence,
            )
            self._last_prediction = pred
            return pred

        # Predict each dimension
        token_preds = {}
        for dim in TokenDimension:
            raw_pred = self.token_regressors[dim].predict(features.values)
            # Clamp to valid range
            token_preds[dim] = max(0.0, min(1.0, raw_pred))

        expected_tokens = TokenVector(token_preds)

        # Predict disagreement
        raw_disagreement = self.disagreement_regressor.predict(features.values)
        expected_disagreement = max(0.0, min(1.0, raw_disagreement))

        # Confidence based on training progress
        confidence = min(0.9, 0.3 + (self.metrics.total_updates / 1000))

        pred = CriticPrediction(
            expected_tokens=expected_tokens,
            expected_disagreement=expected_disagreement,
            confidence=confidence,
        )
        self._last_prediction = pred
        self.metrics.total_predictions += 1

        return pred

    def compute_misalignment(
        self,
        prediction: CriticPrediction,
        actual_tokens: TokenVector,
        actual_disagreement: float,
        student_confidence: float,
    ) -> MisalignmentSignal:
        """Compute the misalignment signal (surprise) after seeing actuals."""
        import math

        # Compute surprise per dimension
        surprise = {}
        negative_surprise_total = 0.0

        for dim in TokenDimension:
            pred_val = prediction.expected_tokens.values[dim]
            actual_val = actual_tokens.values[dim]
            diff = actual_val - pred_val
            surprise[dim] = diff

            # Track negative surprise (predicted high, got low)
            if diff < -0.2:
                negative_surprise_total += abs(diff)

        surprise_vector = TokenVector(surprise)

        # Total surprise magnitude
        total_surprise = math.sqrt(sum(v ** 2 for v in surprise.values()))

        # Overconfidence penalty
        overconfidence = 0.0
        if student_confidence > 0.7 and actual_disagreement > 0.3:
            overconfidence = (student_confidence - 0.5) * actual_disagreement

        # Should have hedged?
        should_hedge = prediction.expected_disagreement > 0.4 and student_confidence > 0.6

        # Track metrics
        if negative_surprise_total > 0.3:
            self.metrics.negative_surprise_count += 1
        self.metrics.surprise_history.append(total_surprise)

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
        """Update critic regressors with actual outcomes."""
        if self._last_features is None:
            return

        features = self._last_features

        # Update token regressors
        total_loss = 0.0
        for dim in TokenDimension:
            target = actual_tokens.values[dim]
            loss = self.token_regressors[dim].get_loss(features, target)
            total_loss += loss
            self.token_regressors[dim].partial_fit(features, target)

            # Update MAE tracking
            if dim.value not in self.metrics.mae_per_dim:
                self.metrics.mae_per_dim[dim.value] = 0.0
            pred = prediction.expected_tokens.values[dim]
            mae = abs(pred - target)
            # Exponential moving average
            alpha = 0.1
            self.metrics.mae_per_dim[dim.value] = (
                alpha * mae + (1 - alpha) * self.metrics.mae_per_dim[dim.value]
            )

        # Update disagreement regressor
        self.disagreement_regressor.partial_fit(features, actual_disagreement)
        disagreement_mae = abs(prediction.expected_disagreement - actual_disagreement)
        self.metrics.disagreement_mae = (
            0.1 * disagreement_mae + 0.9 * self.metrics.disagreement_mae
        )

        # Track loss
        avg_loss = total_loss / len(TokenDimension)
        self.metrics.loss_history.append(avg_loss)

        self.metrics.total_updates += 1
        self._last_features = None

    def fit_batch(
        self,
        items: List[TrainingItem],
        responses: List[StudentResponse],
        token_targets: List[TokenVector],
        disagreement_targets: List[float],
    ):
        """Batch training (for offline or warmup)."""
        for item, response, tokens, disagreement in zip(
            items, responses, token_targets, disagreement_targets
        ):
            features = self.feature_extractor.extract(item, response)

            for dim in TokenDimension:
                self.token_regressors[dim].partial_fit(
                    features.values, tokens.values[dim]
                )

            self.disagreement_regressor.partial_fit(features.values, disagreement)
            self.metrics.total_updates += 1

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance per dimension (weight magnitudes)."""
        feature_names = self.feature_extractor.feature_names
        importance = {}

        for dim in TokenDimension:
            weights = self.token_regressors[dim].weights
            importance[dim.value] = {
                name: float(abs(w))
                for name, w in zip(feature_names, weights)
            }

        importance["disagreement"] = {
            name: float(abs(w))
            for name, w in zip(feature_names, self.disagreement_regressor.weights)
        }

        return importance

    def get_top_features(self, n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Get top N most important features per dimension."""
        importance = self.get_feature_importance()
        top = {}

        for dim_name, weights in importance.items():
            sorted_features = sorted(
                weights.items(), key=lambda x: x[1], reverse=True
            )
            top[dim_name] = sorted_features[:n]

        return top

    def save(self, path: str):
        """Save critic state to file."""
        import pickle

        state = {
            "token_regressors": {
                dim.value: {
                    "weights": reg.weights,
                    "bias": reg.bias,
                    "mean": reg._mean,
                    "var": reg._var,
                    "count": reg._count,
                }
                for dim, reg in self.token_regressors.items()
            },
            "disagreement_regressor": {
                "weights": self.disagreement_regressor.weights,
                "bias": self.disagreement_regressor.bias,
                "mean": self.disagreement_regressor._mean,
                "var": self.disagreement_regressor._var,
                "count": self.disagreement_regressor._count,
            },
            "metrics": {
                "total_updates": self.metrics.total_updates,
                "mae_per_dim": self.metrics.mae_per_dim,
                "disagreement_mae": self.metrics.disagreement_mae,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """Load critic state from file."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        for dim in TokenDimension:
            reg_state = state["token_regressors"][dim.value]
            reg = self.token_regressors[dim]
            reg.weights = reg_state["weights"]
            reg.bias = reg_state["bias"]
            reg._mean = reg_state["mean"]
            reg._var = reg_state["var"]
            reg._count = reg_state["count"]

        dis_state = state["disagreement_regressor"]
        self.disagreement_regressor.weights = dis_state["weights"]
        self.disagreement_regressor.bias = dis_state["bias"]
        self.disagreement_regressor._mean = dis_state["mean"]
        self.disagreement_regressor._var = dis_state["var"]
        self.disagreement_regressor._count = dis_state["count"]

        self.metrics.total_updates = state["metrics"]["total_updates"]
        self.metrics.mae_per_dim = state["metrics"]["mae_per_dim"]
        self.metrics.disagreement_mae = state["metrics"]["disagreement_mae"]

    def get_metrics_summary(self) -> dict:
        """Get summary of critic performance metrics."""
        return {
            "total_updates": self.metrics.total_updates,
            "total_predictions": self.metrics.total_predictions,
            "avg_loss": self.metrics.avg_loss,
            "negative_surprise_rate": self.metrics.negative_surprise_rate,
            "mae_per_dim": self.metrics.mae_per_dim,
            "disagreement_mae": self.metrics.disagreement_mae,
        }
