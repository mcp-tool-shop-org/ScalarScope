"""Professor evaluators for ASPIRE."""

from .base import (
    Professor,
    StrictLogician,
    PragmaticEngineer,
    EmpathyAdvocate,
    ProfessorEnsemble,
)
from .adversarial import (
    RotatingSelector,
    RotationConfig,
    AdversarialProfessor,
    CounterProfessor,
    HoldoutManager,
    HoldoutConfig,
    AdversarialEnsemble,
)

__all__ = [
    # Base professors
    "Professor",
    "StrictLogician",
    "PragmaticEngineer",
    "EmpathyAdvocate",
    "ProfessorEnsemble",
    # Adversarial components
    "RotatingSelector",
    "RotationConfig",
    "AdversarialProfessor",
    "CounterProfessor",
    "HoldoutManager",
    "HoldoutConfig",
    "AdversarialEnsemble",
]
