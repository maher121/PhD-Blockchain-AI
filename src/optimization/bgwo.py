"""Pure, deterministic Binary Grey Wolf Optimization engine.

The optimizer is generic: it operates on indexed binary masks and injected
objective/comparator callables only. It contains no dataset, model, or domain
semantics.

Iteration 0 evaluates the initialized population and counts toward the budget.
For ``I`` configured evaluated iterations, valid indices are ``0..I-1`` and the
hard request ceiling is ``wolves * I``.
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

STOP_MAX_ITERATIONS = "MAX_ITERATIONS"
STOP_EARLY_NO_IMPROVEMENT = "EARLY_STOP_NO_IMPROVEMENT"


class ComparatorContractError(ValueError):
    """Raised when an injected strict comparator prefers both operands."""


@dataclass(frozen=True)
class BGWOConfig:
    """Validated settings for one generic BGWO run."""

    dimensions: int
    wolf_count: int
    evaluated_iterations: int
    random_cardinalities: tuple[int, ...]
    sigmoid_clamp_min: float = -6.0
    sigmoid_clamp_max: float = 6.0
    control_parameter_start: float = 2.0
    control_parameter_end: float = 0.0
    minimum_selected_features: int = 1
    early_stopping_patience: int = 7
    early_stopping_min_iteration: int = 10
    cache_enabled: bool = True
    transfer_function: str = "standard_logistic_sigmoid"
    optimizer_name: str = "BGWO"

    def __post_init__(self) -> None:
        if self.dimensions < 2:
            raise ValueError("BGWO dimensions must be at least two.")
        if self.wolf_count < 3:
            raise ValueError("BGWO requires at least three wolves (alpha, beta, delta).")
        if self.evaluated_iterations < 1:
            raise ValueError("BGWO must evaluate at least one iteration.")
        if len(self.random_cardinalities) != self.wolf_count - 2:
            raise ValueError("Random cardinalities must define exactly wolf_count - 2 masks.")
        if any(not 1 <= value <= self.dimensions for value in self.random_cardinalities):
            raise ValueError("Every initialization cardinality must lie within the dimensions.")
        numeric = (
            self.sigmoid_clamp_min,
            self.sigmoid_clamp_max,
            self.control_parameter_start,
            self.control_parameter_end,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("BGWO numeric parameters must be finite.")
        if self.sigmoid_clamp_min >= self.sigmoid_clamp_max:
            raise ValueError("Sigmoid clamp bounds must be increasing.")
        if self.control_parameter_start < self.control_parameter_end:
            raise ValueError("Control parameter schedule must be non-increasing.")
        if self.minimum_selected_features != 1:
            raise ValueError("BGWO protocol requires exactly one minimum selected feature.")
        if self.early_stopping_patience < 1:
            raise ValueError("Early stopping patience must be positive.")
        if self.early_stopping_min_iteration < 0:
            raise ValueError("Earliest stopping iteration cannot be negative.")
        if not isinstance(self.cache_enabled, bool):
            raise TypeError("cache_enabled must be boolean.")
        if self.transfer_function != "standard_logistic_sigmoid":
            raise ValueError("BGWO permits only the standard logistic sigmoid transfer.")
        if self.optimizer_name != "BGWO":
            raise ValueError("Optimizer name must remain BGWO.")

    @property
    def maximum_candidate_requests(self) -> int:
        return self.wolf_count * self.evaluated_iterations

    @property
    def iterative_update_count(self) -> int:
        return self.evaluated_iterations - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_name": self.optimizer_name,
            "dimensions": self.dimensions,
            "wolf_count": self.wolf_count,
            "evaluated_iterations": self.evaluated_iterations,
            "maximum_candidate_requests": self.maximum_candidate_requests,
            "iterative_update_count": self.iterative_update_count,
            "random_cardinalities": list(self.random_cardinalities),
            "sigmoid_clamp": [self.sigmoid_clamp_min, self.sigmoid_clamp_max],
            "transfer_function": self.transfer_function,
            "control_parameter_schedule": {
                "kind": "linear_decay",
                "start": self.control_parameter_start,
                "end": self.control_parameter_end,
            },
            "minimum_selected_features": self.minimum_selected_features,
            "early_stopping": {
                "patience_evaluated_iterations": self.early_stopping_patience,
                "not_before_iteration_index": self.early_stopping_min_iteration,
            },
            "cache_enabled": self.cache_enabled,
            "iteration_index_origin": 0,
            "iteration_zero_semantics": (
                "iteration 0 evaluates initialization and counts toward the configured "
                "evaluated-iteration budget"
            ),
        }


@dataclass(frozen=True)
class IterationRecord(Generic[EvaluationT]):
    """Governed state captured after one evaluated population."""

    iteration_index: int
    best_mask: np.ndarray
    best_selected_feature_count: int
    best_evaluation: EvaluationT
    request_count: int
    cumulative_unique_evaluations: int
    cumulative_cache_hits: int
    population_diversity: float
    best_rank_improved: bool
    best_changed: bool
    repair_count: int
    control_parameter_a: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_index": self.iteration_index,
            "best_mask": to_json_compatible(self.best_mask),
            "best_selected_feature_count": self.best_selected_feature_count,
            "best_evaluation": to_json_compatible(self.best_evaluation),
            "request_count": self.request_count,
            "cumulative_unique_evaluations": self.cumulative_unique_evaluations,
            "cumulative_cache_hits": self.cumulative_cache_hits,
            "population_diversity": self.population_diversity,
            "best_rank_improved": self.best_rank_improved,
            "best_changed": self.best_changed,
            "repair_count": self.repair_count,
            "control_parameter_a": self.control_parameter_a,
        }


@dataclass(frozen=True)
class BGWOResult(Generic[EvaluationT]):
    """Complete reproducible result of one BGWO run."""

    optimizer_name: str
    optimizer_seed: int
    dimensions: int
    wolf_count: int
    maximum_iterations: int
    iterative_update_count: int
    evaluated_iteration_count: int
    stop_reason: str
    best_mask: np.ndarray
    best_selected_feature_count: int
    best_evaluation: EvaluationT
    best_iteration: int
    total_candidate_requests: int
    unique_evaluations: int
    cache_hits: int
    initial_population: np.ndarray
    initial_latent_positions: np.ndarray
    final_population: np.ndarray
    final_latent_positions: np.ndarray
    convergence_history: tuple[IterationRecord[EvaluationT], ...]
    repair_count: int
    configuration_snapshot: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "optimizer_name": self.optimizer_name,
            "optimizer_seed": self.optimizer_seed,
            "dimensions": self.dimensions,
            "wolf_count": self.wolf_count,
            "maximum_iterations": self.maximum_iterations,
            "iterative_update_count": self.iterative_update_count,
            "evaluated_iteration_count": self.evaluated_iteration_count,
            "stop_reason": self.stop_reason,
            "best_mask": self.best_mask,
            "best_selected_feature_count": self.best_selected_feature_count,
            "best_evaluation": self.best_evaluation,
            "best_iteration": self.best_iteration,
            "total_candidate_requests": self.total_candidate_requests,
            "unique_evaluations": self.unique_evaluations,
            "cache_hits": self.cache_hits,
            "initial_population": self.initial_population,
            "initial_latent_positions": self.initial_latent_positions,
            "final_population": self.final_population,
            "final_latent_positions": self.final_latent_positions,
            "convergence_history": [item.to_dict() for item in self.convergence_history],
            "repair_count": self.repair_count,
            "configuration_snapshot": self.configuration_snapshot,
        }
        return to_json_compatible(payload)

    def to_json(self, *, indent: int | None = None) -> str:
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


def canonical_mask_key(mask: np.ndarray | Sequence[int]) -> bytes:
    """Return a fixed-order binary cache key without lossy integer conversion."""
    array = _validated_mask(mask)
    return array.tobytes()


def exact_cardinality_mask(
    dimensions: int,
    cardinality: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Create one exact-cardinality binary mask for generic dimensions."""
    _require_generator(generator)
    if dimensions < 1:
        raise ValueError("dimensions must be positive.")
    if not 1 <= cardinality <= dimensions:
        raise ValueError("cardinality must satisfy 1 <= K <= dimensions.")
    mask = np.zeros(dimensions, dtype=np.uint8)
    active = generator.choice(dimensions, size=cardinality, replace=False)
    mask[active] = 1
    return mask


def build_initial_population(
    config: BGWOConfig,
    generator: np.random.Generator,
    *,
    k42_mask: np.ndarray | Sequence[int] | None = None,
    k42_inactive_index: int | None = None,
) -> np.ndarray:
    """Construct K-all, K-1, and exact-cardinality random initialization masks."""
    _require_generator(generator)
    population = np.zeros((config.wolf_count, config.dimensions), dtype=np.uint8)
    population[0, :] = 1
    if k42_mask is not None:
        anchor = _validated_mask(k42_mask, expected_length=config.dimensions)
        if int(anchor.sum()) != config.dimensions - 1:
            raise ValueError("Supplied K-1 anchor must have exactly dimensions-1 active bits.")
        population[1] = anchor
    else:
        inactive = config.dimensions - 1 if k42_inactive_index is None else k42_inactive_index
        if not 0 <= inactive < config.dimensions:
            raise ValueError("K-1 inactive index is outside dimensions.")
        population[1, :] = 1
        population[1, inactive] = 0
    for row, cardinality in enumerate(config.random_cardinalities, start=2):
        population[row] = exact_cardinality_mask(config.dimensions, cardinality, generator)
    _validate_population(population, config)
    return population


def linear_control_schedule(
    evaluated_iterations: int,
    start: float,
    end: float,
) -> tuple[float, ...]:
    """Return control-parameter values for the ``I-1`` BGWO updates."""
    if evaluated_iterations < 1:
        raise ValueError("evaluated_iterations must be positive.")
    if not all(math.isfinite(value) for value in (start, end)) or start < end:
        raise ValueError("Control-parameter endpoints must be finite and non-increasing.")
    update_count = evaluated_iterations - 1
    if update_count == 0:
        return ()
    if update_count == 1:
        return (float(start),)
    return tuple(float(value) for value in np.linspace(start, end, update_count))


def sigmoid_probabilities(
    latent: np.ndarray | Sequence[float],
    *,
    clamp_min: float,
    clamp_max: float,
) -> np.ndarray:
    """Compute logistic probabilities after deterministic clamp."""
    values = np.asarray(latent, dtype=float)
    clamped = np.clip(values, clamp_min, clamp_max)
    positive = clamped >= 0
    output = np.empty_like(clamped, dtype=float)
    output[positive] = 1.0 / (1.0 + np.exp(-clamped[positive]))
    exponent = np.exp(clamped[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def repair_empty_mask(
    mask: np.ndarray | Sequence[int],
    latent: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, bool, int | None]:
    """Activate one deterministic coordinate if a sampled mask is empty."""
    binary = _validated_mask(mask).copy()
    values = np.asarray(latent, dtype=float)
    if values.shape != binary.shape or not np.isfinite(values).all():
        raise ValueError("Repair latent vector must be finite and match the mask shape.")
    if int(binary.sum()) > 0:
        return binary, False, None
    magnitudes = np.abs(values)
    index = int(np.argmax(magnitudes))
    binary[index] = 1
    return binary, True, index


def sample_binary_mask(
    latent: np.ndarray | Sequence[float],
    *,
    clamp_min: float,
    clamp_max: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, bool, int | None, np.ndarray, np.ndarray]:
    """Sample one binary mask via the governed sigmoid transfer and repair policy."""
    _require_generator(generator)
    values = np.asarray(latent, dtype=float)
    clamped = np.clip(values, clamp_min, clamp_max)
    probabilities = sigmoid_probabilities(clamped, clamp_min=clamp_min, clamp_max=clamp_max)
    sampled = (generator.random(probabilities.shape) < probabilities).astype(np.uint8)
    repaired, repaired_flag, repaired_index = repair_empty_mask(sampled, clamped)
    return repaired, repaired_flag, repaired_index, probabilities, clamped


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


class BinaryGreyWolfOptimizer(Generic[EvaluationT]):
    """Generic deterministic BGWO with strict comparator-driven ranking."""

    def __init__(
        self,
        config: BGWOConfig,
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
        """Return reproducible initial binary and latent states for one seed."""
        generator = np.random.Generator(np.random.PCG64(int(optimizer_seed)))
        population = build_initial_population(
            self.config,
            generator,
            k42_mask=k42_mask,
            k42_inactive_index=k42_inactive_index,
        )
        latent = population.astype(float, copy=True)
        return population, latent

    def optimize(
        self,
        optimizer_seed: int,
        *,
        initial_population: np.ndarray | None = None,
        initial_latent_positions: np.ndarray | None = None,
        k42_mask: np.ndarray | Sequence[int] | None = None,
        k42_inactive_index: int | None = None,
    ) -> BGWOResult[EvaluationT]:
        """Execute one BGWO run with run-local duplicate-candidate cache."""
        generator = np.random.Generator(np.random.PCG64(int(optimizer_seed)))
        if initial_population is None:
            population = build_initial_population(
                self.config,
                generator,
                k42_mask=k42_mask,
                k42_inactive_index=k42_inactive_index,
            )
        else:
            population = np.asarray(initial_population, dtype=np.uint8).copy()
            _validate_population(population, self.config)
        if initial_latent_positions is None:
            latent_positions = population.astype(float, copy=True)
        else:
            latent_positions = np.asarray(initial_latent_positions, dtype=float).copy()
            _validate_latent(latent_positions, self.config)

        initial_population_snapshot = population.copy()
        initial_latent_snapshot = latent_positions.copy()
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

        a_schedule = linear_control_schedule(
            self.config.evaluated_iterations,
            self.config.control_parameter_start,
            self.config.control_parameter_end,
        )
        global_best_mask: np.ndarray | None = None
        global_best_evaluation: EvaluationT | None = None
        best_iteration = 0
        history: list[IterationRecord[EvaluationT]] = []
        total_repairs = 0
        update_repairs = 0
        applied_a: float | None = None
        stagnant_iterations = 0
        stop_reason = STOP_MAX_ITERATIONS

        for iteration in range(self.config.evaluated_iterations):
            evaluations = [evaluate(mask) for mask in population]
            ordered = self._rank_indices(population, evaluations)
            alpha_index, beta_index, delta_index = ordered[:3]

            candidate_mask = population[alpha_index]
            candidate_evaluation = evaluations[alpha_index]
            rank_improved = iteration == 0
            best_changed = iteration == 0
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
                    best_iteration = iteration
                    best_changed = True
                    rank_improved = relation < 0

            assert global_best_mask is not None and global_best_evaluation is not None
            history.append(
                IterationRecord(
                    iteration_index=iteration,
                    best_mask=global_best_mask.copy(),
                    best_selected_feature_count=int(global_best_mask.sum()),
                    best_evaluation=global_best_evaluation,
                    request_count=self.config.wolf_count,
                    cumulative_unique_evaluations=unique_evaluations,
                    cumulative_cache_hits=cache_hits,
                    population_diversity=mean_pairwise_normalized_hamming(population),
                    best_rank_improved=rank_improved,
                    best_changed=best_changed,
                    repair_count=update_repairs,
                    control_parameter_a=applied_a,
                )
            )

            if iteration > 0:
                stagnant_iterations = 0 if rank_improved else stagnant_iterations + 1
                if (
                    iteration >= self.config.early_stopping_min_iteration
                    and stagnant_iterations >= self.config.early_stopping_patience
                ):
                    stop_reason = STOP_EARLY_NO_IMPROVEMENT
                    break
            if iteration == self.config.evaluated_iterations - 1:
                break

            applied_a = a_schedule[iteration]
            alpha = latent_positions[alpha_index]
            beta = latent_positions[beta_index]
            delta = latent_positions[delta_index]

            next_latent = np.empty_like(latent_positions)
            next_population = np.empty_like(population)
            update_repairs = 0
            for idx in range(self.config.wolf_count):
                current = latent_positions[idx]

                r1_alpha = generator.random(self.config.dimensions)
                r2_alpha = generator.random(self.config.dimensions)
                r1_beta = generator.random(self.config.dimensions)
                r2_beta = generator.random(self.config.dimensions)
                r1_delta = generator.random(self.config.dimensions)
                r2_delta = generator.random(self.config.dimensions)

                a1 = 2.0 * applied_a * r1_alpha - applied_a
                c1 = 2.0 * r2_alpha
                a2 = 2.0 * applied_a * r1_beta - applied_a
                c2 = 2.0 * r2_beta
                a3 = 2.0 * applied_a * r1_delta - applied_a
                c3 = 2.0 * r2_delta

                d_alpha = np.abs(c1 * alpha - current)
                d_beta = np.abs(c2 * beta - current)
                d_delta = np.abs(c3 * delta - current)

                x1 = alpha - a1 * d_alpha
                x2 = beta - a2 * d_beta
                x3 = delta - a3 * d_delta
                updated = (x1 + x2 + x3) / 3.0
                next_latent[idx] = np.clip(
                    updated,
                    self.config.sigmoid_clamp_min,
                    self.config.sigmoid_clamp_max,
                )

                sampled, repaired, _, _, _ = sample_binary_mask(
                    next_latent[idx],
                    clamp_min=self.config.sigmoid_clamp_min,
                    clamp_max=self.config.sigmoid_clamp_max,
                    generator=generator,
                )
                next_population[idx] = sampled
                update_repairs += int(repaired)

            total_repairs += update_repairs
            latent_positions = next_latent
            population = next_population

        assert global_best_mask is not None and global_best_evaluation is not None
        if total_requests != unique_evaluations + cache_hits:
            raise RuntimeError("BGWO cache accounting invariant failed.")
        if total_requests > self.config.maximum_candidate_requests:
            raise RuntimeError("BGWO exceeded its configured candidate-request budget.")

        snapshot = self.config.to_dict()
        snapshot["optimizer_seed"] = int(optimizer_seed)
        return BGWOResult(
            optimizer_name=self.config.optimizer_name,
            optimizer_seed=int(optimizer_seed),
            dimensions=self.config.dimensions,
            wolf_count=self.config.wolf_count,
            maximum_iterations=self.config.evaluated_iterations,
            iterative_update_count=self.config.iterative_update_count,
            evaluated_iteration_count=len(history),
            stop_reason=stop_reason,
            best_mask=global_best_mask.copy(),
            best_selected_feature_count=int(global_best_mask.sum()),
            best_evaluation=global_best_evaluation,
            best_iteration=best_iteration,
            total_candidate_requests=total_requests,
            unique_evaluations=unique_evaluations,
            cache_hits=cache_hits,
            initial_population=initial_population_snapshot,
            initial_latent_positions=initial_latent_snapshot,
            final_population=population.copy(),
            final_latent_positions=latent_positions.copy(),
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

    def _rank_indices(
        self,
        population: np.ndarray,
        evaluations: Sequence[EvaluationT],
    ) -> tuple[int, ...]:
        ordered = list(range(len(evaluations)))
        for left in range(len(ordered)):
            best = left
            for right in range(left + 1, len(ordered)):
                idx_right = ordered[right]
                idx_best = ordered[best]
                relation = self._evaluation_relation(
                    evaluations[idx_right], evaluations[idx_best]
                )
                if relation < 0 or (
                    relation == 0
                    and _mask_tuple(population[idx_right])
                    < _mask_tuple(population[idx_best])
                ):
                    best = right
            ordered[left], ordered[best] = ordered[best], ordered[left]
        return tuple(ordered)


def _require_generator(generator: np.random.Generator) -> None:
    if not isinstance(generator, np.random.Generator) or not isinstance(
        generator.bit_generator, np.random.PCG64
    ):
        raise TypeError("BGWO randomness requires numpy.random.Generator with PCG64.")


def _validated_mask(
    mask: np.ndarray | Sequence[int], *, expected_length: int | None = None
) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 1 or (expected_length is not None and len(array) != expected_length):
        raise ValueError("A binary mask must be one-dimensional with the expected length.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("Binary masks may contain only zero and one.")
    return array.astype(np.uint8, copy=False)


def _validate_population(population: np.ndarray, config: BGWOConfig) -> None:
    if population.shape != (config.wolf_count, config.dimensions):
        raise ValueError("Initial population shape differs from the BGWO configuration.")
    if not np.isin(population, (0, 1)).all():
        raise ValueError("Every wolf position must be binary.")
    if np.any(population.sum(axis=1) < config.minimum_selected_features):
        raise ValueError("Every initial wolf must retain at least one selected dimension.")


def _validate_latent(latent: np.ndarray, config: BGWOConfig) -> None:
    if latent.shape != (config.wolf_count, config.dimensions):
        raise ValueError("Initial latent position shape differs from the BGWO configuration.")
    if not np.isfinite(latent).all():
        raise ValueError("Initial latent positions must be finite.")


def _mask_tuple(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in mask)
