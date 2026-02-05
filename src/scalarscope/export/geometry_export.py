"""Geometry Export for ASPIRE Desktop Visualization.

Exports training run data in the format expected by the MAUI visualization app.
Schema version 1.0.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np


@dataclass
class RunMetadata:
    """Metadata about the training run."""
    run_id: str = ""
    condition: str = ""
    seed: int = 0
    training_items: int = 0
    cycles: int = 0
    holdout_professor: Optional[str] = None
    conscience_tier: str = "UNKNOWN"


@dataclass
class DimensionalityReduction:
    """PCA/UMAP reduction info."""
    method: str = "PCA"
    input_dim: int = 0
    output_dim: int = 2
    explained_variance: List[float] = field(default_factory=list)
    components: List[List[float]] = field(default_factory=list)


@dataclass
class TrajectoryTimestep:
    """Single timestep in the trajectory."""
    t: float = 0.0
    state_2d: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
    curvature: float = 0.0
    effective_dim: float = 0.0


@dataclass
class Trajectory:
    """Full trajectory through state space."""
    timesteps: List[TrajectoryTimestep] = field(default_factory=list)


@dataclass
class ScalarTimestep:
    """Scalar token values at a timestep."""
    t: float = 0.0
    correctness: float = 0.0
    coherence: float = 0.0
    calibration: float = 0.0
    tradeoffs: float = 0.0
    clarity: float = 0.0


@dataclass
class ScalarTimeSeries:
    """Time series of scalar dimensions."""
    dimensions: List[str] = field(default_factory=lambda: [
        "correctness", "coherence", "calibration", "tradeoffs", "clarity"
    ])
    values: List[ScalarTimestep] = field(default_factory=list)


@dataclass
class EigenTimestep:
    """Eigenvalues at a timestep."""
    t: float = 0.0
    values: List[float] = field(default_factory=list)


@dataclass
class AnisotropyTimestep:
    """Anisotropy ratio at a timestep."""
    t: float = 0.0
    ratio: float = 0.0


@dataclass
class GeometryMetrics:
    """Eigenvalue and anisotropy time series."""
    eigenvalues: List[EigenTimestep] = field(default_factory=list)
    anisotropy: List[AnisotropyTimestep] = field(default_factory=list)


@dataclass
class ProfessorVector:
    """Professor position in latent space."""
    name: str = ""
    vector: List[float] = field(default_factory=list)
    holdout: bool = False


@dataclass
class EvaluatorGeometry:
    """Professor vectors for visualizing evaluator alignment."""
    latent_dim: int = 2
    professors: List[ProfessorVector] = field(default_factory=list)


@dataclass
class FailureEvent:
    """Detected failure during training."""
    t: float = 0.0
    category: str = ""
    severity: str = ""
    description: str = ""


@dataclass
class GeometryRun:
    """Complete geometry export for one training run."""
    schema_version: str = "1.0"
    run_metadata: RunMetadata = field(default_factory=RunMetadata)
    reduction: DimensionalityReduction = field(default_factory=DimensionalityReduction)
    trajectory: Trajectory = field(default_factory=Trajectory)
    scalars: ScalarTimeSeries = field(default_factory=ScalarTimeSeries)
    geometry: GeometryMetrics = field(default_factory=GeometryMetrics)
    evaluators: EvaluatorGeometry = field(default_factory=EvaluatorGeometry)
    failures: List[FailureEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "GeometryRun":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeometryRun":
        """Reconstruct from dictionary."""
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            run_metadata=RunMetadata(**data.get("run_metadata", {})),
            reduction=DimensionalityReduction(**data.get("reduction", {})),
            trajectory=Trajectory(
                timesteps=[
                    TrajectoryTimestep(**ts)
                    for ts in data.get("trajectory", {}).get("timesteps", [])
                ]
            ),
            scalars=ScalarTimeSeries(
                dimensions=data.get("scalars", {}).get("dimensions", []),
                values=[
                    ScalarTimestep(**sv)
                    for sv in data.get("scalars", {}).get("values", [])
                ]
            ),
            geometry=GeometryMetrics(
                eigenvalues=[
                    EigenTimestep(**ev)
                    for ev in data.get("geometry", {}).get("eigenvalues", [])
                ],
                anisotropy=[
                    AnisotropyTimestep(**av)
                    for av in data.get("geometry", {}).get("anisotropy", [])
                ]
            ),
            evaluators=EvaluatorGeometry(
                latent_dim=data.get("evaluators", {}).get("latent_dim", 2),
                professors=[
                    ProfessorVector(**pv)
                    for pv in data.get("evaluators", {}).get("professors", [])
                ]
            ),
            failures=[
                FailureEvent(**fe)
                for fe in data.get("failures", [])
            ]
        )


class GeometryExporter:
    """Builds geometry export from training run data."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        condition: str = "unknown",
        seed: int = 0,
    ):
        self.run = GeometryRun()
        self.run.run_metadata.run_id = run_id or datetime.now().isoformat()
        self.run.run_metadata.condition = condition
        self.run.run_metadata.seed = seed

        self._state_history: List[np.ndarray] = []
        self._scalar_history: List[Dict[str, float]] = []

    def set_metadata(
        self,
        training_items: int,
        cycles: int,
        holdout_professor: Optional[str] = None,
        conscience_tier: str = "UNKNOWN",
    ):
        """Set run metadata."""
        self.run.run_metadata.training_items = training_items
        self.run.run_metadata.cycles = cycles
        self.run.run_metadata.holdout_professor = holdout_professor
        self.run.run_metadata.conscience_tier = conscience_tier

    def record_state(self, state: np.ndarray, scalars: Dict[str, float]):
        """Record a state snapshot during training."""
        self._state_history.append(state.copy())
        self._scalar_history.append(scalars.copy())

    def set_professors(self, professors: List[Dict[str, Any]]):
        """Set professor vectors for evaluator geometry."""
        self.run.evaluators.professors = [
            ProfessorVector(
                name=p.get("name", ""),
                vector=p.get("vector", []),
                holdout=p.get("holdout", False),
            )
            for p in professors
        ]
        if professors:
            self.run.evaluators.latent_dim = len(professors[0].get("vector", []))

    def add_failure(
        self,
        t: float,
        category: str,
        severity: str,
        description: str,
    ):
        """Record a failure event."""
        self.run.failures.append(FailureEvent(
            t=t,
            category=category,
            severity=severity,
            description=description,
        ))

    def finalize(self) -> GeometryRun:
        """Compute derived quantities and finalize export."""
        if not self._state_history:
            return self.run

        states = np.array(self._state_history)
        n_steps = len(states)

        # Compute PCA for dimensionality reduction
        if states.shape[1] > 2:
            self._compute_pca(states)

        # Build trajectory
        self._build_trajectory(states)

        # Build scalar time series
        self._build_scalars()

        # Build geometry metrics
        self._build_geometry(states)

        return self.run

    def _compute_pca(self, states: np.ndarray):
        """Compute PCA for 2D projection."""
        centered = states - states.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store reduction info
        total_var = eigenvalues.sum()
        self.run.reduction.method = "PCA"
        self.run.reduction.input_dim = states.shape[1]
        self.run.reduction.output_dim = 2
        self.run.reduction.explained_variance = [
            float(eigenvalues[i] / total_var) if total_var > 0 else 0
            for i in range(min(2, len(eigenvalues)))
        ]
        self.run.reduction.components = [
            eigenvectors[:, i].tolist()
            for i in range(min(2, eigenvectors.shape[1]))
        ]

        # Store projection matrix for trajectory computation
        self._projection = eigenvectors[:, :2]

    def _build_trajectory(self, states: np.ndarray):
        """Build trajectory from state history."""
        n_steps = len(states)

        # Project to 2D if needed
        if hasattr(self, '_projection') and states.shape[1] > 2:
            centered = states - states.mean(axis=0)
            projected = centered @ self._projection
        else:
            projected = states[:, :2] if states.shape[1] >= 2 else states

        # Compute velocities
        velocities = np.zeros_like(projected)
        velocities[1:] = projected[1:] - projected[:-1]

        # Compute curvature (change in velocity direction)
        curvatures = np.zeros(n_steps)
        for i in range(2, n_steps):
            v1 = velocities[i - 1]
            v2 = velocities[i]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                curvatures[i] = np.arccos(cos_angle) / np.pi

        # Compute effective dimensionality over time
        window = max(10, n_steps // 10)
        eff_dims = np.zeros(n_steps)
        for i in range(n_steps):
            start = max(0, i - window)
            window_states = states[start:i + 1]
            if len(window_states) > 2:
                cov = np.cov(window_states.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.maximum(eigenvalues, 0)
                total = eigenvalues.sum()
                if total > 0:
                    normalized = eigenvalues / total
                    sum_sq = (normalized ** 2).sum()
                    eff_dims[i] = 1.0 / sum_sq if sum_sq > 0 else len(eigenvalues)
                else:
                    eff_dims[i] = 1.0

        # Build timesteps
        self.run.trajectory.timesteps = [
            TrajectoryTimestep(
                t=i / max(1, n_steps - 1),
                state_2d=projected[i].tolist(),
                velocity=velocities[i].tolist(),
                curvature=float(curvatures[i]),
                effective_dim=float(eff_dims[i]),
            )
            for i in range(n_steps)
        ]

    def _build_scalars(self):
        """Build scalar time series."""
        n_steps = len(self._scalar_history)
        self.run.scalars.values = [
            ScalarTimestep(
                t=i / max(1, n_steps - 1),
                correctness=self._scalar_history[i].get("correctness", 0),
                coherence=self._scalar_history[i].get("coherence", 0),
                calibration=self._scalar_history[i].get("calibration", 0),
                tradeoffs=self._scalar_history[i].get("tradeoffs", 0),
                clarity=self._scalar_history[i].get("clarity", 0),
            )
            for i in range(n_steps)
        ]

    def _build_geometry(self, states: np.ndarray):
        """Build geometry metrics over time."""
        n_steps = len(states)
        window = max(10, n_steps // 10)

        eigenvalue_series = []
        anisotropy_series = []

        for i in range(n_steps):
            start = max(0, i - window)
            window_states = states[start:i + 1]

            if len(window_states) > 2:
                cov = np.cov(window_states.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
                eigenvalues = np.maximum(eigenvalues, 0)

                # Top 5 eigenvalues
                top_eigen = eigenvalues[:5].tolist()

                # Anisotropy ratio
                if len(eigenvalues) > 1 and eigenvalues[-1] > 1e-10:
                    ratio = eigenvalues[0] / eigenvalues[-1]
                else:
                    ratio = 1.0
            else:
                top_eigen = [1.0]
                ratio = 1.0

            t = i / max(1, n_steps - 1)
            eigenvalue_series.append(EigenTimestep(t=t, values=top_eigen))
            anisotropy_series.append(AnisotropyTimestep(t=t, ratio=ratio))

        self.run.geometry.eigenvalues = eigenvalue_series
        self.run.geometry.anisotropy = anisotropy_series


def generate_sample_export(
    condition: str = "correlated_professors",
    n_steps: int = 100,
    seed: int = 42,
) -> GeometryRun:
    """Generate a sample geometry export for testing the visualizer."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    exporter = GeometryExporter(
        run_id=f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        condition=condition,
        seed=seed,
    )

    exporter.set_metadata(
        training_items=100,
        cycles=50,
        holdout_professor="Nuance" if "correlated" in condition else None,
        conscience_tier="STRONG" if "correlated" in condition else "WEAK",
    )

    # Set professors
    if "correlated" in condition:
        # Correlated professors - clustered vectors
        base = np.array([0.7, 0.3])
        professors = [
            {"name": "Rigor", "vector": (base + np.random.randn(2) * 0.1).tolist(), "holdout": False},
            {"name": "Nuance", "vector": (base + np.random.randn(2) * 0.1).tolist(), "holdout": True},
            {"name": "Holistic", "vector": (base + np.random.randn(2) * 0.1).tolist(), "holdout": False},
            {"name": "Pragmatist", "vector": (base + np.random.randn(2) * 0.1).tolist(), "holdout": False},
            {"name": "Theorist", "vector": (base + np.random.randn(2) * 0.1).tolist(), "holdout": False},
        ]
    else:
        # Orthogonal professors - spread out vectors
        professors = [
            {"name": "Accuracy", "vector": [1.0, 0.0], "holdout": False},
            {"name": "Clarity", "vector": [0.0, 1.0], "holdout": True},
            {"name": "Calibration", "vector": [-0.7, 0.7], "holdout": False},
        ]
    exporter.set_professors(professors)

    # Generate training trajectory
    state_dim = 10
    state = np.random.randn(state_dim) * 0.5

    for i in range(n_steps):
        t = i / n_steps

        # Simulate convergence dynamics
        if "correlated" in condition:
            # Clean spiral convergence (Path B)
            target = np.zeros(state_dim)
            target[0] = np.cos(t * 4 * np.pi) * (1 - t)
            target[1] = np.sin(t * 4 * np.pi) * (1 - t)
            noise = np.random.randn(state_dim) * 0.02 * (1 - t)
        else:
            # Chaotic multi-attractor (Path A)
            target = np.zeros(state_dim)
            target[0] = np.cos(t * 2 * np.pi) * np.cos(t * 5 * np.pi) * 0.5
            target[1] = np.sin(t * 3 * np.pi) * 0.5
            noise = np.random.randn(state_dim) * 0.1

        state = 0.9 * state + 0.1 * target + noise

        # Scalars - improving over time
        scalars = {
            "correctness": 0.3 + 0.5 * t + random.gauss(0, 0.05),
            "coherence": 0.4 + 0.4 * t + random.gauss(0, 0.05),
            "calibration": 0.35 + 0.45 * t + random.gauss(0, 0.05),
            "tradeoffs": 0.3 + 0.4 * t + random.gauss(0, 0.05),
            "clarity": 0.5 + 0.3 * t + random.gauss(0, 0.05),
        }
        scalars = {k: max(0, min(1, v)) for k, v in scalars.items()}

        exporter.record_state(state, scalars)

    # Add some failures
    if "orthogonal" in condition:
        exporter.add_failure(0.3, "EARLY_COLLAPSE", "expected", "Dimensional collapse before convergence")
        exporter.add_failure(0.7, "NO_TRANSFER", "unexpected", "Holdout transfer failed")

    return exporter.finalize()


if __name__ == "__main__":
    # Generate sample exports
    output_dir = Path("experiments/exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Correlated professors (Path B - success case)
    run_b = generate_sample_export("correlated_professors", n_steps=100, seed=42)
    run_b.save(output_dir / "sample_correlated.json")
    print(f"Saved: {output_dir / 'sample_correlated.json'}")

    # Orthogonal professors (Path A - failure case)
    run_a = generate_sample_export("orthogonal_professors", n_steps=100, seed=42)
    run_a.save(output_dir / "sample_orthogonal.json")
    print(f"Saved: {output_dir / 'sample_orthogonal.json'}")
