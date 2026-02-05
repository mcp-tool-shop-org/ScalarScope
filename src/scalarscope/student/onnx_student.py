"""ONNX Runtime student model for GPU inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import time
import re

import numpy as np

from ..core import TrainingItem, StudentResponse
from .model import StudentModel, TrainingSignal


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    stop_strings: List[str] = None

    def __post_init__(self):
        if self.stop_strings is None:
            self.stop_strings = ["</s>", "<|endoftext|>", "\n\nAnswer:", "---"]


@dataclass
class StudentOutput:
    """Structured output from student generation."""
    answer: str
    reasoning: str
    raw_text: str
    token_ids: List[int]
    generation_time_ms: float


class ONNXStudentV1(StudentModel):
    """ONNX Runtime student with greedy/sampling decode.

    V1: No KV-cache. Full sequence recompute each token.
    Works but slower - use for validation before adding cache.

    Designed for RTX 5080 (CUDA) or DirectML fallback.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_name_or_path: str,
        device: str = "cuda",
        generation_config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.tokenizer_path = tokenizer_name_or_path
        self.device = device
        self.config = generation_config or GenerationConfig()
        self.system_prompt = system_prompt or self._default_system_prompt()

        self._session = None
        self._tokenizer = None
        self._loaded = False

    def _default_system_prompt(self) -> str:
        return (
            "You are a careful reasoning assistant. For each question:\n"
            "1. Think through the problem step by step\n"
            "2. Consider tradeoffs and alternatives\n"
            "3. State your confidence level\n"
            "4. Give your final answer\n\n"
            "Format your response as:\n"
            "REASONING: <your step-by-step thinking>\n"
            "CONFIDENCE: <low/medium/high>\n"
            "ANSWER: <your final answer>"
        )

    def _load(self):
        """Lazy load model and tokenizer."""
        if self._loaded:
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Set up execution providers
        providers = []
        provider_options = []

        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
            provider_options.append({
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 14 * 1024 * 1024 * 1024,  # 14GB for 5080
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            })
        elif self.device == "dml":
            providers.append("DmlExecutionProvider")
            provider_options.append({})

        providers.append("CPUExecutionProvider")
        provider_options.append({})

        # Load ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=list(zip(providers, provider_options)),
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )

        # Ensure pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._loaded = True

        # Log what we loaded
        print(f"[ONNXStudent] Loaded model: {self.model_path.name}")
        print(f"[ONNXStudent] Providers: {self._session.get_providers()}")
        print(f"[ONNXStudent] Vocab size: {self._tokenizer.vocab_size}")

    def _build_prompt(self, item: TrainingItem) -> str:
        """Build the full prompt for the model."""
        return f"{self.system_prompt}\n\nQUESTION: {item.prompt}"

    def _sample_token(
        self,
        logits: np.ndarray,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        """Sample next token from logits."""
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Convert to probabilities
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        # Top-k filtering
        if top_k > 0:
            top_k_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_k_indices] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)

            # Find cutoff
            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            mask = np.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx]] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)

        # Sample
        if temperature > 0:
            token_id = np.random.choice(len(probs), p=probs)
        else:
            token_id = np.argmax(probs)

        return int(token_id)

    def _apply_repetition_penalty(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
        penalty: float,
    ) -> np.ndarray:
        """Apply repetition penalty to logits."""
        if penalty == 1.0 or not generated_ids:
            return logits

        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty

        return logits

    def _check_stop_conditions(
        self,
        text: str,
        token_id: int,
        num_generated: int,
    ) -> bool:
        """Check if generation should stop."""
        # EOS token
        if token_id == self._tokenizer.eos_token_id:
            return True

        # Max tokens
        if num_generated >= self.config.max_new_tokens:
            return True

        # Stop strings
        for stop in self.config.stop_strings:
            if stop in text:
                return True

        return False

    def _generate_text(self, prompt: str) -> StudentOutput:
        """Run autoregressive generation."""
        start_time = time.perf_counter()

        # Tokenize prompt
        inputs = self._tokenizer(
            prompt,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=2048,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", np.ones_like(input_ids))

        generated_ids = []
        current_ids = input_ids.copy()
        current_mask = attention_mask.copy()

        # Get model input/output names
        input_names = [inp.name for inp in self._session.get_inputs()]
        output_names = [out.name for out in self._session.get_outputs()]

        # Autoregressive loop
        for step in range(self.config.max_new_tokens):
            # Prepare inputs
            feed_dict = {}
            if "input_ids" in input_names:
                feed_dict["input_ids"] = current_ids.astype(np.int64)
            if "attention_mask" in input_names:
                feed_dict["attention_mask"] = current_mask.astype(np.int64)

            # Run model
            outputs = self._session.run(output_names, feed_dict)

            # Get logits for last position
            logits = outputs[0]  # Assume first output is logits
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # Apply repetition penalty
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits,
                generated_ids,
                self.config.repetition_penalty,
            )

            # Sample next token
            if self.config.do_sample:
                next_token = self._sample_token(
                    next_token_logits,
                    self.config.temperature,
                    self.config.top_p,
                    self.config.top_k,
                )
            else:
                next_token = int(np.argmax(next_token_logits))

            generated_ids.append(next_token)

            # Decode current text to check stop conditions
            current_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            if self._check_stop_conditions(current_text, next_token, len(generated_ids)):
                break

            # Append to sequence for next iteration
            current_ids = np.concatenate([
                current_ids,
                np.array([[next_token]], dtype=np.int64)
            ], axis=1)
            current_mask = np.concatenate([
                current_mask,
                np.array([[1]], dtype=np.int64)
            ], axis=1)

        # Decode final text
        raw_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        generation_time = (time.perf_counter() - start_time) * 1000

        # Parse structured output
        answer, reasoning = self._parse_response(raw_text)

        return StudentOutput(
            answer=answer,
            reasoning=reasoning,
            raw_text=raw_text,
            token_ids=generated_ids,
            generation_time_ms=generation_time,
        )

    def _parse_response(self, text: str) -> Tuple[str, str]:
        """Parse structured response into answer and reasoning."""
        # Try to extract structured format
        answer = ""
        reasoning = ""

        # Look for ANSWER: pattern
        answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        # Look for REASONING: pattern
        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?:CONFIDENCE:|ANSWER:|$)",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Fallback: if no structure, use whole text
        if not answer:
            # Take last sentence or line as answer
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            answer = lines[-1] if lines else text[:100]

        if not reasoning:
            reasoning = text

        return answer, reasoning

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from response."""
        text_lower = text.lower()

        # Look for explicit confidence
        conf_match = re.search(r"confidence:\s*(low|medium|high|\d+\.?\d*)", text_lower)
        if conf_match:
            val = conf_match.group(1)
            if val == "low":
                return 0.3
            elif val == "medium":
                return 0.6
            elif val == "high":
                return 0.85
            else:
                try:
                    return min(1.0, max(0.0, float(val)))
                except ValueError:
                    pass

        # Heuristic: look for hedging language
        hedge_words = ["maybe", "perhaps", "possibly", "uncertain", "not sure", "might"]
        confidence_words = ["clearly", "obviously", "definitely", "certainly", "must be"]

        hedge_count = sum(1 for w in hedge_words if w in text_lower)
        conf_count = sum(1 for w in confidence_words if w in text_lower)

        base = 0.5
        base -= hedge_count * 0.1
        base += conf_count * 0.15

        return max(0.1, min(0.95, base))

    def generate(
        self,
        item: TrainingItem,
        max_tokens: int = 256,
    ) -> StudentResponse:
        """Generate a response for a training item."""
        self._load()

        # Override max tokens if specified
        original_max = self.config.max_new_tokens
        self.config.max_new_tokens = max_tokens

        try:
            prompt = self._build_prompt(item)
            output = self._generate_text(prompt)

            confidence = self._extract_confidence(output.raw_text)

            return StudentResponse(
                item_id=item.id,
                answer=output.answer,
                reasoning_trace=output.reasoning,
                confidence=confidence,
                latency_ms=output.generation_time_ms,
            )
        finally:
            self.config.max_new_tokens = original_max

    def update(self, signal: TrainingSignal):
        """Log training signal for offline learning.

        ONNX models are frozen at inference time.
        Signals are collected for batch fine-tuning later.
        """
        # TODO: Log to training buffer or file
        # For now, just track that we received it
        pass

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        self._load()

        return {
            "model_path": str(self.model_path),
            "tokenizer": self.tokenizer_path,
            "device": self.device,
            "providers": self._session.get_providers(),
            "vocab_size": self._tokenizer.vocab_size,
            "inputs": [inp.name for inp in self._session.get_inputs()],
            "outputs": [out.name for out in self._session.get_outputs()],
        }
