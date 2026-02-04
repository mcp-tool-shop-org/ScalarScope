"""Token-based reward system for ASPIRE training."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class TokenDimension(Enum):
    """Dimensions along which tokens are awarded.

    Each dimension represents a distinct "judgment virtue" that
    the student learns to optimize.
    """
    CORRECTNESS = "correctness"      # Got the right answer
    COHERENCE = "coherence"          # No internal contradictions
    TRADEOFFS = "tradeoffs"          # Acknowledged alternatives/risks
    CALIBRATION = "calibration"      # Appropriate uncertainty expression
    CLARITY = "clarity"              # Clear, understandable reasoning


@dataclass
class TokenVector:
    """Multi-dimensional token payout from a single evaluation.

    Tokens are the reward signal that teaches the student what
    "proper behavior" looks like. By making it a vector instead
    of a scalar, we avoid collapsing distinct virtues into one number.
    """
    values: Dict[TokenDimension, float] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize all dimensions to 0 if not provided
        for dim in TokenDimension:
            if dim not in self.values:
                self.values[dim] = 0.0

    @property
    def total(self) -> float:
        """Sum of all token dimensions."""
        return sum(self.values.values())

    def __add__(self, other: "TokenVector") -> "TokenVector":
        return TokenVector({
            dim: self.values[dim] + other.values[dim]
            for dim in TokenDimension
        })

    def __truediv__(self, scalar: float) -> "TokenVector":
        return TokenVector({
            dim: val / scalar
            for dim, val in self.values.items()
        })

    def min_with(self, other: "TokenVector") -> "TokenVector":
        """Element-wise minimum (for strict critic aggregation)."""
        return TokenVector({
            dim: min(self.values[dim], other.values[dim])
            for dim in TokenDimension
        })

    def to_tensor(self) -> List[float]:
        """Convert to ordered list for model input."""
        return [self.values[dim] for dim in TokenDimension]

    @classmethod
    def from_tensor(cls, tensor: List[float]) -> "TokenVector":
        """Create from ordered list."""
        return cls({
            dim: tensor[i]
            for i, dim in enumerate(TokenDimension)
        })


@dataclass
class TokenLedger:
    """Accumulated tokens over a training session.

    Tracks both totals and history for diagnostics.
    """
    history: List[TokenVector] = field(default_factory=list)

    def record(self, tokens: TokenVector):
        self.history.append(tokens)

    @property
    def total(self) -> TokenVector:
        if not self.history:
            return TokenVector()
        result = self.history[0]
        for tv in self.history[1:]:
            result = result + tv
        return result

    @property
    def mean(self) -> TokenVector:
        if not self.history:
            return TokenVector()
        return self.total / len(self.history)

    def by_dimension(self, dim: TokenDimension) -> List[float]:
        """Get history for a single dimension (for plotting)."""
        return [tv.values[dim] for tv in self.history]
