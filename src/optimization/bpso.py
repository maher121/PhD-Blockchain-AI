"""Pure, deterministic Binary Particle Swarm Optimization engine.

The optimizer operates only on indexed binary vectors and injected objective and
comparison callables. It has no dataset, classifier, metric, resource, energy,
or carbon semantics.

Generation 0 evaluates the initialized population and counts toward the budget.
For ``G`` configured evaluated generations, valid indices are ``0..G-1`` and
the hard request ceiling is ``particles * G``. The first velocity update creates
generation 1 with the initial inertia value; the final update creates generation
``G-1`` with the final inertia value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

import numpy as np


EvaluationT = TypeVar("EvaluationT")
Objective = Callable[[np.ndarray], EvaluationT]
Comparator = Callable[[EvaluationT, EvaluationT], bool]

STOP_MAX_GENERATIONS = "MAX_GENERATIONS"
STOP_EARLY_NO_IMPROVEMENT = "EARLY_STOP_NO_IMPROVEMENT"


class ComparatorContractError(ValueError):
    """Raised when an injected strict comparator prefers both operands."""


@dataclass(frozen=True)
class BPSOConfig:
    """Validated settings for one generic BPSO run."""

    dimensions: int
    particle_count: int
    evaluated_generations: int
    random_cardinalities: tuple[int, ...]
    velocity_initial_min: float = -1.0
    velocity_initial_max: float = 1.0
    velocity_clamp_min: float = -6.0
    velocity_clamp_max: float = 6.0
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    cognitive_coefficient: float = 2.0
    social_coefficient: float = 2.0
    minimum_selected_features: int = 1
    early_stopping_patience: int = 7
    early_stopping_min_generation: int = 10
    cache_enabled: bool = True
    transfer_function: str = "standard_logistic_sigmoid"
    optimizer_name: str = "BPSO"

    def __post_init__(self) -> None:
        if self.dimensions < 2:
            raise ValueError("BPSO dimensions must be at least two.")
        if self.particle_count < 2:
            raise ValueError("BPSO requires at least two particles for its anchors.")
        if self.evaluated_generations < 1:
            raise ValueError("BPSO must evaluate at least one generation.")
        if len(self.random_cardinalities) != self.particle_count - 2:
            raise ValueError(
                "Random cardinalities must define exactly particle_count - 2 masks."
            )
        if any(not 1 <= value <= self.dimensions for value in self.random_cardinalities):
            raise ValueError("Every initialization cardinality must be within the dimensions.")
        numeric = (
            self.velocity_initial_min,
            self.velocity_initial_max,
            self.velocity_clamp_min,
            self.velocity_clamp_max,
            self.inertia_start,
            self.inertia_end,
            self.cognitive_coefficient,
            self.social_coefficient,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("BPSO numeric parameters must be finite.")
        if self.velocity_initial_min >= self.velocity_initial_max:
            raise ValueError("Velocity initialization bounds must be increasing.")
        if self.velocity_clamp_min >= self.velocity_clamp_max:
            raise ValueError("Velocity clamp bounds must be increasing.")
        if (
            self.velocity_initial_min < self.velocity_clamp_min
            or self.velocity_initial_max > self.velocity_clamp_max
        ):
            raise ValueError("Initial velocities must lie within the velocity clamp.")
        if self.inertia_start < self.inertia_end:
            raise ValueError("The approved inertia schedule must be non-increasing.")
        if self.cognitive_coefficient < 0 or self.social_coefficient < 0:
            raise ValueError("BPSO cognitive and social coefficients must be nonnegative.")
        if self.minimum_selected_features != 1:
            raise ValueError("V0.8-A supports the approved minimum of exactly one feature.")
        if self.early_stopping_patience < 1:
            raise ValueError("Early-stopping patience must be positive.")
        if self.early_stopping_min_generation < 0:
            raise ValueError("The earliest stopping generation cannot be negative.")
        if not isinstance(self.cache_enabled, bool):
            raise TypeError("cache_enabled must be boolean.")
        if self.transfer_function != "standard_logistic_sigmoid":
            raise ValueError("V0.8-A permits only the standard logistic sigmoid.")
        if self.optimizer_name != "BPSO":
            raise ValueError("The optimizer name must remain BPSO.")

    @property
    def maximum_fitness_requests(self) -> int:
        return self.particle_count * self.evaluated_generations

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_name": self.optimizer_name,
            "dimensions": self.dimensions,
            "particle_count": self.particle_count,
            "evaluated_generations": self.evaluated_generations,
            "maximum_fitness_requests": self.maximum_fitness_requests,
            "random_cardinalities": list(self.random_cardinalities),
            "velocity_initialization": [
                self.velocity_initial_min,
                self.velocity_initial_max,
            ],
            "velocity_clamp": [self.velocity_clamp_min, self.velocity_clamp_max],
            "transfer_function": self.transfer_function,
            "inertia_schedule": {
                "kind": "linear_decay",
                "start": self.inertia_start,
                "end": self.inertia_end,
            },
            "cognitive_coefficient": self.cognitive_coefficient,
            "social_coefficient": self.social_coefficient,
            "minimum_selected_features": self.minimum_selected_features,
            "early_stopping": {
                "patience_evaluated_generations": self.early_stopping_patience,
                "not_before_generation_index": self.early_stopping_min_generation,
            },
            "cache_enabled": self.cache_enabled,
            "generation_index_origin": 0,
            "generation_semantics": (
                "generation 0 evaluates initialization and counts toward the configured "
                "evaluated-generation budget"
            ),
        }


@dataclass(frozen=True)
class GenerationRecord(Generic[EvaluationT]):
    """Governed state captured after one evaluated population."""

    generation_index: int
    best_mask: np.ndarray
    best_selected_feature_count: int
    best_evaluation: EvaluationT
    request_count: int
    cumulative_unique_evaluations: int
    cumulative_cache_hits: int
    population_diversity: float
    global_best_improved: bool
    global_best_changed: bool
    repair_count: int
    inertia: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_index": self.generation_index,
            "best_mask": to_json_compatible(self.best_mask),
            "best_selected_feature_count": self.best_selected_feature_count,
            "best_evaluation": to_json_compatible(self.best_evaluation),
            "request_count": self.request_count,
            "cumulative_unique_evaluations": self.cumulative_unique_evaluations,
            "cumulative_cache_hits": self.cumulative_cache_hits,
            "population_diversity": self.population_diversity,
            "global_best_improved": self.global_best_improved,
            "global_best_changed": self.global_best_changed,
            "repair_count": self.repair_count,
            "inertia": self.inertia,
        }


@dataclass(frozen=True)
class BPSOResult(Generic[EvaluationT]):
    """Complete reproducible result of one optimizer run."""

    optimizer_name: str
    optimizer_seed: int
    dimensions: int
    particle_count: int
    maximum_generations: int
    evaluated_generation_count: int
    stop_reason: str
    best_mask: np.ndarray
    best_selected_feature_count: int
    best_evaluation: EvaluationT
    best_generation: int
    total_fitness_requests: int
    unique_evaluations: int
    cache_hits: int
    initial_population: np.ndarray
    initial_velocities: np.ndarray
    final_population: np.ndarray
    final_velocities: np.ndarray
    personal_best_masks: np.ndarray
    personal_best_evaluations: tuple[EvaluationT, ...]
    convergence_history: tuple[GenerationRecord[EvaluationT], ...]
    repair_count: int
    configuration_snapshot: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "optimizer_name": self.optimizer_name,
            "optimizer_seed": self.optimizer_seed,
            "dimensions": self.dimensions,
            "particle_count": self.particle_count,
            "maximum_generations": self.maximum_generations,
            "evaluated_generation_count": self.evaluated_generation_count,
            "stop_reason": self.stop_reason,
            "best_mask": self.best_mask,
            "best_selected_feature_count": self.best_selected_feature_count,
            "best_evaluation": self.best_evaluation,
            "best_generation": self.best_generation,
            "total_fitness_requests": self.total_fitness_requests,
            "unique_evaluations": self.unique_evaluations,
            "cache_hits": self.cache_hits,
            "initial_population": self.initial_population,
            "initial_velocities": self.initial_velocities,
            "final_population": self.final_population,
            "final_velocities": self.final_velocities,
            "personal_best_masks": self.personal_best_masks,
            "personal_best_evaluations": self.personal_best_evaluations,
            "convergence_history": [item.to_dict() for item in self.convergence_history],
            "repair_count": self.repair_count,
            "configuration_snapshot": self.configuration_snapshot,
        }
        return to_json_compatible(payload)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize deterministically with strict finite-number handling."""
        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            allow_nan=False,
            indent=indent,
            separators=separators,
        )


def to_json_compatible(value: Any) -> Any:
    """Convert supported scientific values to stable JSON-compatible values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floats cannot be serialized safely.")
        return float(value)
    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())
    if isinstance(value, Enum):
        return to_json_compatible(value.value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_json_compatible(value.to_dict())
    if is_dataclass(value):
        return to_json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): to_json_compatible(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    raise TypeError(f"Unsupported JSON serialization type: {type(value).__name__}")


def stable_sigmoid(values: np.ndarray | Sequence[float] | float) -> np.ndarray:
    """Evaluate the logistic sigmoid without overflow."""
    array = np.asarray(values, dtype=float)
    output = np.empty_like(array, dtype=float)
    positive = array >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponent = np.exp(array[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def linear_inertia_schedule(
    evaluated_generations: int,
    start: float,
    end: float,
) -> tuple[float, ...]:
    """Return inertia values for the ``G-1`` updates between ``G`` evaluations."""
    if evaluated_generations < 1:
        raise ValueError("evaluated_generations must be positive.")
    if not all(math.isfinite(value) for value in (start, end)) or start < end:
        raise ValueError("Inertia endpoints must be finite and non-increasing.")
    update_count = evaluated_generations - 1
    if update_count == 0:
        return ()
    if update_count == 1:
        return (float(start),)
    return tuple(float(value) for value in np.linspace(start, end, update_count))


def canonical_mask_key(mask: np.ndarray | Sequence[int]) -> bytes:
    """Return a fixed-order binary cache key without lossy integer conversion."""
    array = _validated_mask(mask)
    return array.tobytes()


def build_initial_population(
    config: BPSOConfig,
    generator: np.random.Generator,
    *,
    k42_mask: np.ndarray | Sequence[int] | None = None,
    k42_inactive_index: int | None = None,
) -> np.ndarray:
    """Construct the two anchors and cardinality-stratified random masks."""
    _require_generator(generator)
    population = np.zeros((config.particle_count, config.dimensions), dtype=np.uint8)
    population[0, :] = 1
    if k42_mask is not None:
        anchor = _validated_mask(k42_mask, expected_length=config.dimensions)
        if int(anchor.sum()) != config.dimensions - 1:
            raise ValueError("The supplied K42-equivalent anchor must have D-1 active bits.")
        population[1] = anchor
    else:
        inactive = config.dimensions - 1 if k42_inactive_index is None else k42_inactive_index
        if not 0 <= inactive < config.dimensions:
            raise ValueError("The synthetic K42 inactive index is outside the dimensions.")
        population[1, :] = 1
        population[1, inactive] = 0
    for row, cardinality in enumerate(config.random_cardinalities, start=2):
        active = generator.choice(config.dimensions, size=cardinality, replace=False)
        population[row, active] = 1
    _validate_population(population, config)
    return population


def initialize_velocities(
    config: BPSOConfig, generator: np.random.Generator
) -> np.ndarray:
    """Initialize independent particle velocities from the configured uniform law."""
    _require_generator(generator)
    return generator.uniform(
        config.velocity_initial_min,
        config.velocity_initial_max,
        size=(config.particle_count, config.dimensions),
    )


def update_velocity(
    velocity: np.ndarray,
    position: np.ndarray,
    personal_best: np.ndarray,
    global_best: np.ndarray,
    *,
    inertia: float,
    cognitive_coefficient: float,
    social_coefficient: float,
    clamp_min: float,
    clamp_max: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Apply the standard PSO velocity update and hard clamp."""
    _require_generator(generator)
    velocity = np.asarray(velocity, dtype=float)
    position = _validated_mask(position, expected_length=len(velocity))
    personal_best = _validated_mask(personal_best, expected_length=len(velocity))
    global_best = _validated_mask(global_best, expected_length=len(velocity))
    if not math.isfinite(inertia) or clamp_min >= clamp_max:
        raise ValueError("Velocity update parameters are invalid.")
    r1 = generator.random(velocity.shape)
    r2 = generator.random(velocity.shape)
    updated = (
        inertia * velocity
        + cognitive_coefficient * r1 * (personal_best.astype(float) - position)
        + social_coefficient * r2 * (global_best.astype(float) - position)
    )
    return np.clip(updated, clamp_min, clamp_max)


def repair_empty_mask(
    position: np.ndarray | Sequence[int], probabilities: np.ndarray | Sequence[float]
) -> tuple[np.ndarray, bool, int | None]:
    """Activate the highest-probability coordinate when a mask is empty."""
    mask = _validated_mask(position).copy()
    values = np.asarray(probabilities, dtype=float)
    if values.shape != mask.shape or not np.isfinite(values).all():
        raise ValueError("Repair probabilities must be finite and match the mask.")
    if int(mask.sum()) > 0:
        return mask, False, None
    index = int(np.argmax(values))
    mask[index] = 1
    return mask, True, index


def sample_binary_position(
    velocity: np.ndarray,
    generator: np.random.Generator,
) -> tuple[np.ndarray, bool, int | None]:
    """Sample a binary position through the standard logistic transfer function."""
    _require_generator(generator)
    probabilities = stable_sigmoid(velocity)
    position = (generator.random(probabilities.shape) < probabilities).astype(np.uint8)
    return repair_empty_mask(position, probabilities)


def mean_pairwise_normalized_hamming(population: np.ndarray) -> float:
    """Return mean pairwise Hamming distance divided by dimensionality."""
    array = np.asarray(population)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("Population must be a nonempty two-dimensional matrix.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("Population positions must be binary.")
    if len(array) < 2:
        return 0.0
    distances = [
        float(np.count_nonzero(array[left] != array[right]) / array.shape[1])
        for left in range(len(array))
        for right in range(left + 1, len(array))
    ]
    return float(np.mean(distances))


class BinaryParticleSwarmOptimizer(Generic[EvaluationT]):
    """Generic BPSO controlled exclusively by an injected strict comparator."""

    def __init__(
        self,
        config: BPSOConfig,
        objective: Objective[EvaluationT],
        is_better: Comparator[EvaluationT],
    ) -> None:
        if not callable(objective) or not callable(is_better):
            raise TypeError("objective and is_better must be callable.")
        self.config = config
        self.objective = objective
        self.is_better = is_better

    def initialize(
        self,
        optimizer_seed: int,
        *,
        k42_mask: np.ndarray | Sequence[int] | None = None,
        k42_inactive_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return reproducible initial positions and velocities for one seed."""
        generator = np.random.Generator(np.random.PCG64(int(optimizer_seed)))
        population = build_initial_population(
            self.config,
            generator,
            k42_mask=k42_mask,
            k42_inactive_index=k42_inactive_index,
        )
        velocities = initialize_velocities(self.config, generator)
        return population, velocities

    def optimize(
        self,
        optimizer_seed: int,
        *,
        initial_population: np.ndarray | None = None,
        initial_velocities: np.ndarray | None = None,
        k42_mask: np.ndarray | Sequence[int] | None = None,
        k42_inactive_index: int | None = None,
    ) -> BPSOResult[EvaluationT]:
        """Execute one run with a run-local objective cache."""
        generator = np.random.Generator(np.random.PCG64(int(optimizer_seed)))
        if initial_population is None:
            positions = build_initial_population(
                self.config,
                generator,
                k42_mask=k42_mask,
                k42_inactive_index=k42_inactive_index,
            )
        else:
            positions = np.asarray(initial_population, dtype=np.uint8).copy()
            _validate_population(positions, self.config)
        if initial_velocities is None:
            velocities = initialize_velocities(self.config, generator)
        else:
            velocities = np.asarray(initial_velocities, dtype=float).copy()
            _validate_velocities(velocities, self.config)

        initial_positions_snapshot = positions.copy()
        initial_velocities_snapshot = velocities.copy()
        cache: dict[bytes, EvaluationT] = {}
        total_requests = 0
        unique_evaluations = 0
        cache_hits = 0

        def evaluate(mask: np.ndarray) -> EvaluationT:
            nonlocal total_requests, unique_evaluations, cache_hits
            total_requests += 1
            key = canonical_mask_key(mask)
            if self.config.cache_enabled and key in cache:
                cache_hits += 1
                return cache[key]
            evaluation = self.objective(mask.copy())
            unique_evaluations += 1
            if self.config.cache_enabled:
                cache[key] = evaluation
            return evaluation

        inertia_schedule = linear_inertia_schedule(
            self.config.evaluated_generations,
            self.config.inertia_start,
            self.config.inertia_end,
        )
        personal_best_masks: np.ndarray | None = None
        personal_best_evaluations: list[EvaluationT] = []
        global_best_mask: np.ndarray | None = None
        global_best_evaluation: EvaluationT | None = None
        best_generation = 0
        history: list[GenerationRecord[EvaluationT]] = []
        total_repairs = 0
        generation_repairs = 0
        applied_inertia: float | None = None
        stagnant_generations = 0
        stop_reason = STOP_MAX_GENERATIONS

        for generation in range(self.config.evaluated_generations):
            evaluations = [evaluate(position) for position in positions]
            if generation == 0:
                personal_best_masks = positions.copy()
                personal_best_evaluations = list(evaluations)
            else:
                assert personal_best_masks is not None
                for index, evaluation in enumerate(evaluations):
                    relation = self._evaluation_relation(
                        evaluation, personal_best_evaluations[index]
                    )
                    if relation < 0 or (
                        relation == 0
                        and _mask_tuple(positions[index])
                        < _mask_tuple(personal_best_masks[index])
                    ):
                        personal_best_masks[index] = positions[index].copy()
                        personal_best_evaluations[index] = evaluation

            generation_best = self._best_index(positions, evaluations)
            candidate_mask = positions[generation_best]
            candidate_evaluation = evaluations[generation_best]
            rank_improved = generation == 0
            best_changed = generation == 0
            if global_best_mask is None:
                global_best_mask = candidate_mask.copy()
                global_best_evaluation = candidate_evaluation
            else:
                assert global_best_evaluation is not None
                relation = self._evaluation_relation(
                    candidate_evaluation, global_best_evaluation
                )
                tie_preferred = (
                    relation == 0
                    and _mask_tuple(candidate_mask) < _mask_tuple(global_best_mask)
                )
                if relation < 0 or tie_preferred:
                    global_best_mask = candidate_mask.copy()
                    global_best_evaluation = candidate_evaluation
                    best_generation = generation
                    best_changed = True
                    rank_improved = relation < 0

            assert global_best_mask is not None and global_best_evaluation is not None
            history.append(
                GenerationRecord(
                    generation_index=generation,
                    best_mask=global_best_mask.copy(),
                    best_selected_feature_count=int(global_best_mask.sum()),
                    best_evaluation=global_best_evaluation,
                    request_count=self.config.particle_count,
                    cumulative_unique_evaluations=unique_evaluations,
                    cumulative_cache_hits=cache_hits,
                    population_diversity=mean_pairwise_normalized_hamming(positions),
                    global_best_improved=rank_improved,
                    global_best_changed=best_changed,
                    repair_count=generation_repairs,
                    inertia=applied_inertia,
                )
            )

            if generation > 0:
                stagnant_generations = 0 if rank_improved else stagnant_generations + 1
                if (
                    generation >= self.config.early_stopping_min_generation
                    and stagnant_generations >= self.config.early_stopping_patience
                ):
                    stop_reason = STOP_EARLY_NO_IMPROVEMENT
                    break
            if generation == self.config.evaluated_generations - 1:
                break

            assert personal_best_masks is not None
            applied_inertia = inertia_schedule[generation]
            next_velocities = np.empty_like(velocities)
            next_positions = np.empty_like(positions)
            generation_repairs = 0
            for index in range(self.config.particle_count):
                next_velocities[index] = update_velocity(
                    velocities[index],
                    positions[index],
                    personal_best_masks[index],
                    global_best_mask,
                    inertia=applied_inertia,
                    cognitive_coefficient=self.config.cognitive_coefficient,
                    social_coefficient=self.config.social_coefficient,
                    clamp_min=self.config.velocity_clamp_min,
                    clamp_max=self.config.velocity_clamp_max,
                    generator=generator,
                )
                next_positions[index], repaired, _ = sample_binary_position(
                    next_velocities[index], generator
                )
                generation_repairs += int(repaired)
            total_repairs += generation_repairs
            velocities = next_velocities
            positions = next_positions

        assert personal_best_masks is not None
        assert global_best_mask is not None and global_best_evaluation is not None
        if total_requests != unique_evaluations + cache_hits:
            raise RuntimeError("BPSO cache accounting invariant failed.")
        if total_requests > self.config.maximum_fitness_requests:
            raise RuntimeError("BPSO exceeded its configured request budget.")
        snapshot = self.config.to_dict()
        snapshot["optimizer_seed"] = int(optimizer_seed)
        return BPSOResult(
            optimizer_name=self.config.optimizer_name,
            optimizer_seed=int(optimizer_seed),
            dimensions=self.config.dimensions,
            particle_count=self.config.particle_count,
            maximum_generations=self.config.evaluated_generations,
            evaluated_generation_count=len(history),
            stop_reason=stop_reason,
            best_mask=global_best_mask.copy(),
            best_selected_feature_count=int(global_best_mask.sum()),
            best_evaluation=global_best_evaluation,
            best_generation=best_generation,
            total_fitness_requests=total_requests,
            unique_evaluations=unique_evaluations,
            cache_hits=cache_hits,
            initial_population=initial_positions_snapshot,
            initial_velocities=initial_velocities_snapshot,
            final_population=positions.copy(),
            final_velocities=velocities.copy(),
            personal_best_masks=personal_best_masks.copy(),
            personal_best_evaluations=tuple(personal_best_evaluations),
            convergence_history=tuple(history),
            repair_count=total_repairs,
            configuration_snapshot=snapshot,
        )

    def _evaluation_relation(self, left: EvaluationT, right: EvaluationT) -> int:
        left_better = bool(self.is_better(left, right))
        right_better = bool(self.is_better(right, left))
        if left_better and right_better:
            raise ComparatorContractError(
                "The injected comparator must be strict and asymmetric."
            )
        if left_better:
            return -1
        if right_better:
            return 1
        return 0

    def _best_index(
        self, positions: np.ndarray, evaluations: Sequence[EvaluationT]
    ) -> int:
        best = 0
        for index in range(1, len(evaluations)):
            relation = self._evaluation_relation(evaluations[index], evaluations[best])
            if relation < 0 or (
                relation == 0
                and _mask_tuple(positions[index]) < _mask_tuple(positions[best])
            ):
                best = index
        return best


def _require_generator(generator: np.random.Generator) -> None:
    if not isinstance(generator, np.random.Generator) or not isinstance(
        generator.bit_generator, np.random.PCG64
    ):
        raise TypeError("BPSO randomness requires numpy.random.Generator with PCG64.")


def _validated_mask(
    mask: np.ndarray | Sequence[int], *, expected_length: int | None = None
) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 1 or (expected_length is not None and len(array) != expected_length):
        raise ValueError("A binary mask must be one-dimensional with the expected length.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("Binary masks may contain only zero and one.")
    return array.astype(np.uint8, copy=False)


def _validate_population(population: np.ndarray, config: BPSOConfig) -> None:
    if population.shape != (config.particle_count, config.dimensions):
        raise ValueError("Initial population shape differs from the BPSO configuration.")
    if not np.isin(population, (0, 1)).all():
        raise ValueError("Every particle position must be binary.")
    if np.any(population.sum(axis=1) < config.minimum_selected_features):
        raise ValueError("Every initial particle must retain at least one dimension.")


def _validate_velocities(velocities: np.ndarray, config: BPSOConfig) -> None:
    if velocities.shape != (config.particle_count, config.dimensions):
        raise ValueError("Initial velocity shape differs from the BPSO configuration.")
    if not np.isfinite(velocities).all():
        raise ValueError("Initial velocities must be finite.")
    if (
        np.any(velocities < config.velocity_clamp_min)
        or np.any(velocities > config.velocity_clamp_max)
    ):
        raise ValueError("Initial velocities must lie inside the configured clamp.")


def _mask_tuple(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in mask)
