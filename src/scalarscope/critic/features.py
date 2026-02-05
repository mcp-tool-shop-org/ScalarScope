"""Text feature extraction for LearnedCritic."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re
import numpy as np

from ..core import TrainingItem, StudentResponse


@dataclass
class FeatureSet:
    """Extracted features with names for interpretability."""
    values: np.ndarray
    names: List[str]

    def __len__(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict:
        return {name: float(val) for name, val in zip(self.names, self.values)}


class TextFeatureExtractor:
    """Extract predictive features from student responses.

    These features are designed to predict:
    - Token awards per dimension (correctness, coherence, tradeoffs, calibration, clarity)
    - Professor disagreement
    - Likelihood of being correct

    Features are cheap to compute (no model inference required) but
    highly predictive of judgment quality.
    """

    # Hedging language (indicates uncertainty)
    HEDGE_WORDS = {
        "maybe", "perhaps", "possibly", "probably", "might", "could",
        "uncertain", "unsure", "not sure", "i think", "i believe",
        "it seems", "appears to", "likely", "unlikely",
    }

    # Confidence language (indicates certainty)
    CONFIDENCE_WORDS = {
        "definitely", "certainly", "always", "never", "must", "clearly",
        "obviously", "undoubtedly", "without doubt", "absolutely",
        "sure", "confident", "certain",
    }

    # Tradeoff/deliberation markers
    TRADEOFF_WORDS = {
        "however", "but", "although", "though", "on the other hand",
        "tradeoff", "trade-off", "depends", "it depends", "context",
        "alternatively", "whereas", "while", "yet", "nevertheless",
        "downside", "upside", "pro", "con", "advantage", "disadvantage",
    }

    # Options/alternatives markers
    OPTIONS_WORDS = {
        "option", "alternative", "approach", "method", "way",
        "choice", "possibility", "scenario", "case",
    }

    # Refusal/insufficiency markers
    REFUSAL_WORDS = {
        "can't", "cannot", "unable", "insufficient", "not enough",
        "need more", "unclear", "ambiguous", "vague", "missing",
    }

    # Logical structure markers
    LOGIC_WORDS = {
        "therefore", "thus", "hence", "so", "because", "since",
        "consequently", "as a result", "implies", "means",
    }

    # Contradiction indicators
    CONTRADICTION_PAIRS = [
        ({"always", "must", "never"}, {"depends", "might", "sometimes"}),
        ({"definitely", "certainly"}, {"uncertain", "unsure", "maybe"}),
    ]

    def __init__(self):
        self._feature_names: Optional[List[str]] = None

    @property
    def feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        if self._feature_names is None:
            # Generate by running extraction once
            dummy_item = TrainingItem(
                id="dummy", prompt="test", gold_answer="test", gold_rationale="test"
            )
            dummy_response = StudentResponse(
                item_id="dummy", answer="test", reasoning_trace="test", confidence=0.5
            )
            features = self.extract(dummy_item, dummy_response)
            self._feature_names = features.names
        return self._feature_names

    @property
    def num_features(self) -> int:
        """Number of features extracted."""
        return len(self.feature_names)

    def extract(
        self,
        item: TrainingItem,
        response: StudentResponse,
    ) -> FeatureSet:
        """Extract all features from item and response."""
        features = []
        names = []

        prompt = item.prompt.lower()
        answer = response.answer.lower()
        reasoning = response.reasoning_trace.lower()
        full_text = f"{reasoning} {answer}"

        # === Length / Structure Features ===

        # Prompt characteristics
        features.append(len(item.prompt))
        names.append("prompt_len_chars")

        features.append(len(item.prompt.split()))
        names.append("prompt_len_words")

        # Output characteristics
        features.append(len(response.answer))
        names.append("answer_len_chars")

        features.append(len(response.reasoning_trace))
        names.append("reasoning_len_chars")

        features.append(len(full_text.split()))
        names.append("output_len_words")

        # Structure
        features.append(reasoning.count("\n") + 1)
        names.append("num_lines")

        features.append(1.0 if "answer:" in full_text else 0.0)
        names.append("has_answer_marker")

        features.append(1.0 if "reasoning:" in full_text else 0.0)
        names.append("has_reasoning_marker")

        features.append(1.0 if "confidence:" in full_text else 0.0)
        names.append("has_confidence_marker")

        # === Hedging / Calibration Features ===

        hedge_count = self._count_markers(full_text, self.HEDGE_WORDS)
        features.append(hedge_count)
        names.append("hedge_count")

        confidence_count = self._count_markers(full_text, self.CONFIDENCE_WORDS)
        features.append(confidence_count)
        names.append("confidence_word_count")

        word_count = len(full_text.split()) + 1
        features.append(hedge_count / word_count)
        names.append("hedge_ratio")

        features.append(confidence_count / word_count)
        names.append("confidence_ratio")

        # Numeric confidence (e.g., "80%", "0.7")
        has_numeric = bool(re.search(r"\d+%|\d\.\d", full_text))
        features.append(1.0 if has_numeric else 0.0)
        names.append("has_numeric_confidence")

        # Student's stated confidence
        features.append(response.confidence)
        names.append("stated_confidence")

        # Calibration proxy: high confidence + hedge words = suspicious
        features.append(response.confidence * hedge_count)
        names.append("confidence_hedge_interaction")

        # === Tradeoff / Deliberation Features ===

        tradeoff_count = self._count_markers(full_text, self.TRADEOFF_WORDS)
        features.append(tradeoff_count)
        names.append("tradeoff_count")

        # Diminishing returns: saturate at 3
        features.append(min(tradeoff_count, 3))
        names.append("tradeoff_count_capped")

        options_count = self._count_markers(full_text, self.OPTIONS_WORDS)
        features.append(options_count)
        names.append("options_count")

        # Deliberation depth proxy
        features.append(tradeoff_count + options_count)
        names.append("deliberation_score")

        # === Logic / Coherence Features ===

        logic_count = self._count_markers(full_text, self.LOGIC_WORDS)
        features.append(logic_count)
        names.append("logic_marker_count")

        negation_count = full_text.count(" not ") + full_text.count(" no ") + full_text.count("n't")
        features.append(negation_count)
        names.append("negation_count")

        # Contradiction detection
        has_contradiction = self._detect_contradictions(full_text)
        features.append(1.0 if has_contradiction else 0.0)
        names.append("has_contradiction")

        # === Refusal / Uncertainty Behavior ===

        refusal_count = self._count_markers(full_text, self.REFUSAL_WORDS)
        features.append(refusal_count)
        names.append("refusal_count")

        question_count = full_text.count("?")
        features.append(question_count)
        names.append("question_count")

        # === Readability / Noise Features ===

        # Punctuation density
        punct_count = sum(1 for c in full_text if c in ".,;:!?-")
        features.append(punct_count / (len(full_text) + 1))
        names.append("punctuation_density")

        # Uppercase ratio (excluding start of sentences)
        upper_count = sum(1 for c in full_text if c.isupper())
        features.append(upper_count / (len(full_text) + 1))
        names.append("uppercase_ratio")

        # Repeated bigrams (noise indicator)
        words = full_text.split()
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        unique_bigrams = len(set(bigrams))
        repetition_ratio = 1.0 - (unique_bigrams / (len(bigrams) + 1)) if bigrams else 0.0
        features.append(repetition_ratio)
        names.append("bigram_repetition_ratio")

        # === Item Difficulty Proxy ===

        features.append(item.difficulty)
        names.append("item_difficulty")

        # Long prompts tend to be harder
        features.append(len(item.prompt.split()) / 100)
        names.append("prompt_complexity_proxy")

        # === Interaction Features ===

        # Short answer + long reasoning = deliberate
        answer_reasoning_ratio = (len(answer) + 1) / (len(reasoning) + 1)
        features.append(answer_reasoning_ratio)
        names.append("answer_reasoning_ratio")

        # High difficulty + high confidence = overconfidence risk
        features.append(item.difficulty * response.confidence)
        names.append("difficulty_confidence_interaction")

        # Tradeoffs + difficulty = appropriate deliberation
        features.append(tradeoff_count * item.difficulty)
        names.append("tradeoff_difficulty_interaction")

        return FeatureSet(
            values=np.array(features, dtype=np.float32),
            names=names,
        )

    def _count_markers(self, text: str, markers: set) -> int:
        """Count occurrences of marker words/phrases in text."""
        count = 0
        for marker in markers:
            count += text.count(marker)
        return count

    def _detect_contradictions(self, text: str) -> bool:
        """Detect potential contradictions via marker pairs."""
        for set_a, set_b in self.CONTRADICTION_PAIRS:
            has_a = any(marker in text for marker in set_a)
            has_b = any(marker in text for marker in set_b)
            if has_a and has_b:
                return True
        return False

    def extract_batch(
        self,
        items: List[TrainingItem],
        responses: List[StudentResponse],
    ) -> np.ndarray:
        """Extract features for a batch of items/responses."""
        return np.array([
            self.extract(item, resp).values
            for item, resp in zip(items, responses)
        ])
