"""LearnedCritic V1: Logit-aware critic with learned revision decision.

V1 improvements over V0:
1. Uses logit-derived features (entropy, margin, EOS prob) from student generation
2. Learns when to trigger revision (instead of fixed thresholds)
3. Better calibration through model-internal uncertainty signals

The key insight is that text features can be gamed (the model can write
"I'm uncertain" while being internally certain), but logit features
reflect true model uncertainty that's hard to fake.

Literature:
- Logit Uncertainty: arXiv 2025 "Estimating LLM Uncertainty with Logits"
- Entropy/Margin: ACL 2025 "Harmonized Uncertainty Estimation for LLMs"
- Confidence Neurons: arXiv 2024 "Confidence Regulation Neurons in Language Models"
- Calibration: ICML 2017 "On Calibration of Modern Neural Networks"
See docs/LITERATURE_REVIEW.md Sections 6, 10 for full references.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import numpy as np

from ..core import (
    TokenVector,
    TokenDimension,
    TrainingItem,
    StudentResponse,
    EnsembleEvaluation,
)
from ..student.onnx_student_v2 import GenerationStats
from .head import Critic, CriticPrediction, MisalignmentSignal
from .features_v1 import TextFeatureExtractorV1, FeatureSetV1
from .learned_critic import OnlineSGDRegressor, CriticMetrics


@dataclass
class CriticMetricsV1(CriticMetrics):
    """Extended metrics for V1 critic."""
    # Revision prediction metrics
    revision_predictions: int = 0
    revision_correct: int = 0  # Predicted revision helped and it did
    revision_false_positive: int = 0  # Predicted revision but it didn't help
    revision_false_negative: int = 0  # Didn't predict but should have

    # Logit feature impact tracking
    logit_feature_count: int = 0
    logit_feature_missing: int = 0

    @property
    def revision_accuracy(self) -> float:
        if self.revision_predictions == 0:
            return 0.0
        return self.revision_correct / self.revision_predictions

    @property
    def logit_feature_rate(self) -> float:
        total = self.logit_feature_count + self.logit_feature_missing
        if total == 0:
            return 0.0
        return self.logit_feature_count / total


class OnlineLogisticRegressor:
    """Simple online logistic regression for binary classification.

    Used for the should_revise head.
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

        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0

        # Running statistics
        self._mean = np.zeros(n_features, dtype=np.float32)
        self._var = np.ones(n_features, dtype=np.float32)
        self._count = 0

    def _update_stats(self, x: np.ndarray):
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._var += (delta * delta2 - self._var) / self._count

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._var + 1e-8)
        return (x - self._mean) / std

    def _sigmoid(self, z: float) -> float:
        # Clip to avoid overflow
        z = np.clip(z, -20, 20)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, x: np.ndarray) -> float:
        """Predict probability of positive class."""
        if self._count > 10:
            x = self._normalize(x)
        z = float(np.dot(self.weights, x) + self.bias)
        return self._sigmoid(z)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> bool:
        """Predict binary class."""
        return self.predict_proba(x) >= threshold

    def partial_fit(self, x: np.ndarray, y: bool):
        """Update weights with single example."""
        self._update_stats(x)

        if self._count > 10:
            x = self._normalize(x)

        prob = self._sigmoid(float(np.dot(self.weights, x) + self.bias))
        target = 1.0 if y else 0.0
        error = prob - target

        # Gradient descent
        self.weights -= self.lr * (error * x + self.l2_reg * self.weights)
        self.bias -= self.lr * error

    def get_loss(self, x: np.ndarray, y: bool) -> float:
        """Compute binary cross-entropy loss."""
        prob = self.predict_proba(x)
        target = 1.0 if y else 0.0
        # Add epsilon to avoid log(0)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        return -target * np.log(prob) - (1 - target) * np.log(1 - prob)


class LearnedCriticV1(Critic):
    """Logit-aware learned critic with revision prediction.

    V1 improvements:
    - Uses GenerationStats (entropy, margin, EOS prob) from student
    - Learns when revision will help (instead of fixed thresholds)
    - Better detects fake hedging (text says uncertain, model isn't)
    - Better calibration signals for overconfidence detection

    The "gut feeling" now incorporates model-internal uncertainty signals
    that the student can't easily manipulate through text generation.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        l2_reg: float = 0.001,
        min_samples_before_predict: int = 10,
        revision_threshold: float = 0.5,
        enable_revision_learning: bool = True,
    ):
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.min_samples = min_samples_before_predict
        self.revision_threshold = revision_threshold
        self.enable_revision_learning = enable_revision_learning

        # V1 feature extractor (includes logit features)
        self.feature_extractor = TextFeatureExtractorV1()
        n_features = self.feature_extractor.num_features_v1

        # Token dimension regressors
        self.token_regressors: Dict[TokenDimension, OnlineSGDRegressor] = {
            dim: OnlineSGDRegressor(n_features, learning_rate, l2_reg)
            for dim in TokenDimension
        }

        # Disagreement regressor
        self.disagreement_regressor = OnlineSGDRegressor(n_features, learning_rate, l2_reg)

        # Revision prediction head (learned)
        self.revision_classifier = OnlineLogisticRegressor(n_features, learning_rate, l2_reg)

        # V1 metrics
        self.metrics = CriticMetricsV1()

        # Cache for update
        self._last_features: Optional[np.ndarray] = None
        self._last_has_logit: bool = False
        self._last_prediction: Optional[CriticPrediction] = None

    def predict(
        self,
        item: TrainingItem,
        response: StudentResponse,
        generation_stats: Optional[GenerationStats] = None,
    ) -> CriticPrediction:
        """Predict token outcomes using V1 features."""
        # Extract V1 features (includes logit features if available)
        stats = generation_stats or getattr(response, 'generation_stats', None)
        features = self.feature_extractor.extract_v1(item, response, stats)

        self._last_features = features.values
        self._last_has_logit = features.has_logit_features

        # Track logit feature availability
        if features.has_logit_features:
            self.metrics.logit_feature_count += 1
        else:
            self.metrics.logit_feature_missing += 1

        # Early return if not enough training data
        if self.metrics.total_updates < self.min_samples:
            expected_tokens = TokenVector({dim: 0.5 for dim in TokenDimension})
            expected_disagreement = 0.3
            confidence = 0.2

            pred = CriticPrediction(
                expected_tokens=expected_tokens,
                expected_disagreement=expected_disagreement,
                confidence=confidence,
            )
            self._last_prediction = pred
            return pred

        # Predict each token dimension
        token_preds = {}
        for dim in TokenDimension:
            raw_pred = self.token_regressors[dim].predict(features.values)
            token_preds[dim] = max(0.0, min(1.0, raw_pred))

        expected_tokens = TokenVector(token_preds)

        # Predict disagreement
        raw_disagreement = self.disagreement_regressor.predict(features.values)
        expected_disagreement = max(0.0, min(1.0, raw_disagreement))

        # Confidence scales with training progress and logit feature availability
        base_confidence = 0.3 + (self.metrics.total_updates / 1000)
        if features.has_logit_features:
            base_confidence += 0.1  # Bonus for having richer features
        confidence = min(0.9, base_confidence)

        pred = CriticPrediction(
            expected_tokens=expected_tokens,
            expected_disagreement=expected_disagreement,
            confidence=confidence,
        )
        self._last_prediction = pred
        self.metrics.total_predictions += 1

        return pred

    def predict_should_revise(
        self,
        item: TrainingItem,
        response: StudentResponse,
        generation_stats: Optional[GenerationStats] = None,
    ) -> Tuple[bool, float]:
        """Predict whether revision would help.

        Returns:
            (should_revise, probability)
        """
        # Extract features
        stats = generation_stats or getattr(response, 'generation_stats', None)
        features = self.feature_extractor.extract_v1(item, response, stats)

        if self.metrics.total_updates < self.min_samples:
            # Prior: occasionally suggest revision
            return False, 0.3

        prob = self.revision_classifier.predict_proba(features.values)
        should_revise = prob >= self.revision_threshold

        self.metrics.revision_predictions += 1

        return should_revise, prob

    def compute_misalignment(
        self,
        prediction: CriticPrediction,
        actual_tokens: TokenVector,
        actual_disagreement: float,
        student_confidence: float,
    ) -> MisalignmentSignal:
        """Compute misalignment with enhanced overconfidence detection."""
        import math

        surprise = {}
        negative_surprise_total = 0.0

        for dim in TokenDimension:
            pred_val = prediction.expected_tokens.values[dim]
            actual_val = actual_tokens.values[dim]
            diff = actual_val - pred_val
            surprise[dim] = diff

            if diff < -0.2:
                negative_surprise_total += abs(diff)

        surprise_vector = TokenVector(surprise)
        total_surprise = math.sqrt(sum(v ** 2 for v in surprise.values()))

        # Enhanced overconfidence detection using logit features
        overconfidence = 0.0
        if student_confidence > 0.7 and actual_disagreement > 0.3:
            overconfidence = (student_confidence - 0.5) * actual_disagreement

        # Additional overconfidence signal from logit features
        if self._last_has_logit and self._last_features is not None:
            # Find entropy_mean in features
            feature_names = self.feature_extractor.feature_names_v1
            if "entropy_mean" in feature_names:
                entropy_idx = feature_names.index("entropy_mean")
                entropy_mean = float(self._last_features[entropy_idx])

                # High confidence + high entropy = overconfident
                if student_confidence > 0.7 and entropy_mean > 0.5:
                    overconfidence += (student_confidence - 0.5) * entropy_mean * 0.3

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
        """Update regressors with actual outcomes."""
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

            # Track MAE
            if dim.value not in self.metrics.mae_per_dim:
                self.metrics.mae_per_dim[dim.value] = 0.0
            pred = prediction.expected_tokens.values[dim]
            mae = abs(pred - target)
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

    def update_revision_head(
        self,
        item: TrainingItem,
        response: StudentResponse,
        revision_helped: bool,
        generation_stats: Optional[GenerationStats] = None,
    ):
        """Update the revision prediction head.

        Args:
            item: Training item
            response: Original response (before revision)
            revision_helped: True if revision improved tokens
            generation_stats: Optional logit stats from original generation
        """
        if not self.enable_revision_learning:
            return

        stats = generation_stats or getattr(response, 'generation_stats', None)
        features = self.feature_extractor.extract_v1(item, response, stats)

        # Get current prediction for metrics
        if self.metrics.total_updates >= self.min_samples:
            pred_prob = self.revision_classifier.predict_proba(features.values)
            predicted_revise = pred_prob >= self.revision_threshold

            if predicted_revise and revision_helped:
                self.metrics.revision_correct += 1
            elif predicted_revise and not revision_helped:
                self.metrics.revision_false_positive += 1
            elif not predicted_revise and revision_helped:
                self.metrics.revision_false_negative += 1

        # Update classifier
        self.revision_classifier.partial_fit(features.values, revision_helped)

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance including logit features."""
        feature_names = self.feature_extractor.feature_names_v1
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

        importance["revision"] = {
            name: float(abs(w))
            for name, w in zip(feature_names, self.revision_classifier.weights)
        }

        return importance

    def get_logit_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get importance of logit-derived features specifically."""
        all_importance = self.get_feature_importance()
        logit_names = set(self.feature_extractor.LOGIT_FEATURE_NAMES)

        logit_importance = {}
        for dim_name, weights in all_importance.items():
            logit_importance[dim_name] = {
                name: val
                for name, val in weights.items()
                if name in logit_names
            }

        return logit_importance

    def save(self, path: str):
        """Save V1 critic state to file."""
        import pickle

        state = {
            "version": 1,
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
            "revision_classifier": {
                "weights": self.revision_classifier.weights,
                "bias": self.revision_classifier.bias,
                "mean": self.revision_classifier._mean,
                "var": self.revision_classifier._var,
                "count": self.revision_classifier._count,
            },
            "metrics": {
                "total_updates": self.metrics.total_updates,
                "total_predictions": self.metrics.total_predictions,
                "mae_per_dim": self.metrics.mae_per_dim,
                "disagreement_mae": self.metrics.disagreement_mae,
                "revision_predictions": self.metrics.revision_predictions,
                "revision_correct": self.metrics.revision_correct,
                "logit_feature_count": self.metrics.logit_feature_count,
                "logit_feature_missing": self.metrics.logit_feature_missing,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """Load V1 critic state from file."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        # Token regressors
        for dim in TokenDimension:
            reg_state = state["token_regressors"][dim.value]
            reg = self.token_regressors[dim]
            reg.weights = reg_state["weights"]
            reg.bias = reg_state["bias"]
            reg._mean = reg_state["mean"]
            reg._var = reg_state["var"]
            reg._count = reg_state["count"]

        # Disagreement regressor
        dis_state = state["disagreement_regressor"]
        self.disagreement_regressor.weights = dis_state["weights"]
        self.disagreement_regressor.bias = dis_state["bias"]
        self.disagreement_regressor._mean = dis_state["mean"]
        self.disagreement_regressor._var = dis_state["var"]
        self.disagreement_regressor._count = dis_state["count"]

        # Revision classifier
        if "revision_classifier" in state:
            rev_state = state["revision_classifier"]
            self.revision_classifier.weights = rev_state["weights"]
            self.revision_classifier.bias = rev_state["bias"]
            self.revision_classifier._mean = rev_state["mean"]
            self.revision_classifier._var = rev_state["var"]
            self.revision_classifier._count = rev_state["count"]

        # Metrics
        metrics_state = state["metrics"]
        self.metrics.total_updates = metrics_state["total_updates"]
        self.metrics.total_predictions = metrics_state.get("total_predictions", 0)
        self.metrics.mae_per_dim = metrics_state["mae_per_dim"]
        self.metrics.disagreement_mae = metrics_state["disagreement_mae"]
        self.metrics.revision_predictions = metrics_state.get("revision_predictions", 0)
        self.metrics.revision_correct = metrics_state.get("revision_correct", 0)
        self.metrics.logit_feature_count = metrics_state.get("logit_feature_count", 0)
        self.metrics.logit_feature_missing = metrics_state.get("logit_feature_missing", 0)

    def get_metrics_summary(self) -> dict:
        """Get summary of V1 critic performance metrics."""
        return {
            "version": 1,
            "total_updates": self.metrics.total_updates,
            "total_predictions": self.metrics.total_predictions,
            "avg_loss": self.metrics.avg_loss,
            "negative_surprise_rate": self.metrics.negative_surprise_rate,
            "mae_per_dim": self.metrics.mae_per_dim,
            "disagreement_mae": self.metrics.disagreement_mae,
            "revision_accuracy": self.metrics.revision_accuracy,
            "logit_feature_rate": self.metrics.logit_feature_rate,
        }
