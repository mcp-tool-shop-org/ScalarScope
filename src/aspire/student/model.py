"""Student model wrapper for ASPIRE training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
import time

from ..core import TrainingItem, StudentResponse


class StudentModel(ABC):
    """Base class for student models.

    The student generates reasoning traces and answers for test items.
    Can be backed by a local model, API, or mock for testing.
    """

    @abstractmethod
    def generate(
        self,
        item: TrainingItem,
        max_tokens: int = 256,
    ) -> StudentResponse:
        """Generate a response for a training item."""
        pass

    @abstractmethod
    def update(self, signal: "TrainingSignal"):
        """Update the model based on training signal."""
        pass


@dataclass
class TrainingSignal:
    """Signal for updating the student model."""
    item: TrainingItem
    response: StudentResponse
    token_reward: float                    # Scalar reward from tokens
    gold_answer: str
    gold_rationale: str
    critiques: list                        # Professor critiques


class MockStudent(StudentModel):
    """Mock student for testing the training loop.

    Generates plausible-looking responses without actual inference.
    Useful for validating the pipeline before plugging in real models.
    """

    def __init__(self, correct_rate: float = 0.5):
        self.correct_rate = correct_rate
        self._rng_state = 42

    def generate(
        self,
        item: TrainingItem,
        max_tokens: int = 256,
    ) -> StudentResponse:
        import random
        random.seed(self._rng_state)
        self._rng_state += 1

        start = time.perf_counter()

        # Simulate whether we get it right
        is_correct = random.random() < self.correct_rate

        if is_correct:
            answer = item.gold_answer
            reasoning = f"Considering the problem: {item.prompt[:50]}... " \
                       f"The answer is {answer} because of the following reasoning. " \
                       f"This approach has tradeoffs but seems correct."
            confidence = random.uniform(0.6, 0.95)
        else:
            # Pick a near-miss or make something up
            if item.near_misses:
                answer = random.choice(item.near_misses)
            else:
                answer = f"wrong_answer_{random.randint(0, 100)}"
            reasoning = f"Looking at {item.prompt[:50]}... " \
                       f"I believe the answer is {answer}. " \
                       f"This seems reasonable."
            confidence = random.uniform(0.3, 0.8)

        latency = (time.perf_counter() - start) * 1000

        return StudentResponse(
            item_id=item.id,
            answer=answer,
            reasoning_trace=reasoning,
            confidence=confidence,
            latency_ms=latency,
        )

    def update(self, signal: TrainingSignal):
        # Mock: adjust correct_rate based on reward
        if signal.token_reward > 0.5:
            self.correct_rate = min(0.95, self.correct_rate + 0.01)
        else:
            self.correct_rate = max(0.1, self.correct_rate - 0.005)


class ONNXStudent(StudentModel):
    """Student backed by ONNX Runtime for fast local inference.

    Designed for your RTX 5080 - uses DirectML or CUDA execution provider.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda",  # or "dml" for DirectML
    ):
        self.model_path = model_path
        self.device = device
        self._session = None
        self._tokenizer = None

    def _load(self):
        """Lazy load the model."""
        if self._session is not None:
            return

        import onnxruntime as ort

        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        elif self.device == "dml":
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(
            self.model_path,
            providers=providers,
        )

        # Tokenizer would be loaded here
        # self._tokenizer = ...

    def generate(
        self,
        item: TrainingItem,
        max_tokens: int = 256,
    ) -> StudentResponse:
        self._load()

        start = time.perf_counter()

        # TODO: Actual inference
        # input_ids = self._tokenizer.encode(item.prompt)
        # outputs = self._session.run(None, {"input_ids": input_ids})
        # answer, reasoning = self._decode(outputs)

        # Placeholder until model is set up
        raise NotImplementedError(
            "ONNX inference not yet implemented. "
            "Use MockStudent for testing the loop."
        )

    def update(self, signal: TrainingSignal):
        # ONNX models are typically frozen - updates happen offline
        # Log signal for later batch training
        pass
