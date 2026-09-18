"""Pure, deterministic GOVERNED hybrid BPSO+BGWO optimization engine (V1.0).

Implements the locked V1.0-A design ``BPSO_BGWO_SEQUENTIAL_50_50_ELITE3``:

    BPSO exploration phase  ->  Top-3 distinct elite transfer  ->  BGWO refinement

The engine is evaluator-agnostic: it operates on indexed binary masks and an
injected objective/comparator pair only. It carries no dataset, classifier,
metric, or carbon semantics of any kind.

Mechanics reuse the frozen, tested BPSO/BGWO primitives from ``bpso.py`` and
``bgwo.py`` without modifying them. The hybrid adds the pieces those standalone
engines cannot express: a single run-local cache shared across both phases,
elite knowledge transfer, and strict fixed-budget execution (no early
stopping).

Production arithmetic per the locked protocol (verified but never executed in
this module): 12 agents x (8 BPSO generations + 8 BGWO iterations) = 96 + 96 =
192 candidate requests per run; five future runs = 960 requests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

import numpy as np

from src.optimization.bpso import (
    linear_inertia_schedule,
    sample_binary_position,
    update_velocity,
)
from src.optimization.bgwo import (
    exact_cardinality_mask,
    linear_control_schedule,
    canonical_mask_key,
    mean_pairwise_normalized_hamming,
    sample_binary_mask,
    to_json_compatible,
)


EvaluationT = TypeVar("EvaluationT")
Objective = Callable[[np.ndarray], EvaluationT]
Comparator = Callable[[EvaluationT, EvaluationT], bool]

OPTIMIZER_IDENTIFIER = "GOVERNED_HYBRID_BPSO_BGWO"
DESIGN_IDENTIFIER = "BPSO_BGWO_SEQUENTIAL_50_50_ELITE3"
STOP_FIXED_BUDGET_EXHAUSTED = "FIXED_BUDGET_EXHAUSTED"

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "hybrid_v10.yaml"
)
DEFAULT_BPSO_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "bpso.yaml"
)
DEFAULT_BGWO_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "bgwo_v09.yaml"
)


class ComparatorContractError(ValueError):
    """Raised when an injected strict comparator prefers both operands."""


@dataclass(frozen=True)
class HybridConfig:
    """Validated settings for one governed hybrid BPSO+BGWO run."""

    dimensions: int
    population_size: int
    bpso_evaluated_generations: int
    bgwo_evaluated_iterations: int
    cardinality_pool: tuple[int, ...]
    elite_count: int = 3
    bgwo_phase_rng_offset: int = 10000
    velocity_initial_min: float = -1.0
    velocity_initial_max: float = 1.0
    velocity_clamp_min: float = -6.0
    velocity_clamp_max: float = 6.0
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    cognitive_coefficient: float = 2.0
    social_coefficient: float = 2.0
    sigmoid_clamp_min: float = -6.0
    sigmoid_clamp_max: float = 6.0
    control_parameter_start: float = 2.0
    control_parameter_end: float = 0.0
    minimum_selected_features: int = 1
    cache_enabled: bool = True
    optimizer_name: str = OPTIMIZER_IDENTIFIER
    design_identifier: str = DESIGN_IDENTIFIER

    def __post_init__(self) -> None:
        if self.dimensions < 2:
            raise ValueError("Hybrid dimensions must be at least two.")
        if self.population_size < 3:
            raise ValueError("Hybrid requires at least three agents (alpha/beta/delta).")
        if self.bpso_evaluated_generations < 1 or self.bgwo_evaluated_iterations < 1:
            raise ValueError("Both hybrid phases must evaluate at least once.")
        if not self.cardinality_pool:
            raise ValueError("The hybrid cardinality pool must be nonempty.")
        if any(not 1 <= value <= self.dimensions for value in self.cardinality_pool):
            raise ValueError("Every cardinality must lie within the dimensions.")
        if self.elite_count < 1:
            raise ValueError("The hybrid elite count must be at least one.")
        if self.bgwo_phase_rng_offset == 0:
            raise ValueError("The BGWO-phase RNG offset must be nonzero.")
        numeric = (
            self.velocity_initial_min,
            self.velocity_initial_max,
            self.velocity_clamp_min,
            self.velocity_clamp_max,
            self.inertia_start,
            self.inertia_end,
            self.cognitive_coefficient,
            self.social_coefficient,
            self.sigmoid_clamp_min,
            self.sigmoid_clamp_max,
            self.control_parameter_start,
            self.control_parameter_end,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("Hybrid numeric parameters must be finite.")
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
        if self.sigmoid_clamp_min >= self.sigmoid_clamp_max:
            raise ValueError("Sigmoid clamp bounds must be increasing.")
        if self.control_parameter_start < self.control_parameter_end:
            raise ValueError("Control parameter schedule must be non-increasing.")
        if self.cognitive_coefficient < 0 or self.social_coefficient < 0:
            raise ValueError("Hybrid cognitive/social coefficients must be nonnegative.")
        if self.minimum_selected_features != 1:
            raise ValueError("V1.0-A locks exactly one minimum selected feature.")
        if not self.cache_enabled:
            raise ValueError("The locked V1.0 hybrid requires cache_enabled=True.")
        if self.optimizer_name != OPTIMIZER_IDENTIFIER:
            raise ValueError("The hybrid optimizer identifier is locked.")
        if self.design_identifier != DESIGN_IDENTIFIER:
            raise ValueError("The hybrid design identifier is locked.")

    @property
    def bpso_request_allocation(self) -> int:
        """Exact BPSO-phase candidate-request allocation (fixed at execution)."""
        return self.population_size * self.bpso_evaluated_generations

    @property
    def bgwo_request_allocation(self) -> int:
        """Exact BGWO-phase candidate-request allocation (fixed at execution)."""
        return self.population_size * self.bgwo_evaluated_iterations

    @property
    def per_run_request_allocation(self) -> int:
        return self.bpso_request_allocation + self.bgwo_request_allocation

    @property
    def five_run_request_allocation(self) -> int:
        return 5 * self.per_run_request_allocation

    @property
    def phase_budget_ratio(self) -> tuple[int, int]:
        return (self.bpso_request_allocation, self.bgwo_request_allocation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_name": self.optimizer_name,
            "design_identifier": self.design_identifier,
            "dimensions": self.dimensions,
            "population_size": self.population_size,
            "bpso_evaluated_generations": self.bpso_evaluated_generations,
            "bgwo_evaluated_iterations": self.bgwo_evaluated_iterations,
            "bpso_request_allocation": self.bpso_request_allocation,
            "bgwo_request_allocation": self.bgwo_request_allocation,
            "per_run_request_allocation": self.per_run_request_allocation,
            "five_run_request_allocation": self.five_run_request_allocation,
            "elite_count": self.elite_count,
            "bgwo_phase_rng_offset": self.bgwo_phase_rng_offset,
            "cardinality_pool": list(self.cardinality_pool),
            "velocity_initialization": [
                self.velocity_initial_min,
                self.velocity_initial_max,
            ],
            "velocity_clamp": [self.velocity_clamp_min, self.velocity_clamp_max],
            "inertia_schedule": {
                "kind": "linear_decay",
                "start": self.inertia_start,
                "end": self.inertia_end,
            },
            "cognitive_coefficient": self.cognitive_coefficient,
            "social_coefficient": self.social_coefficient,
            "sigmoid_clamp": [self.sigmoid_clamp_min, self.sigmoid_clamp_max],
            "control_parameter_schedule": {
                "kind": "linear_decay",
                "start": self.control_parameter_start,
                "end": self.control_parameter_end,
            },
            "minimum_selected_features": self.minimum_selected_features,
            "cache_enabled": self.cache_enabled,
            "early_stopping": "disabled (fixed-budget execution locked by V1.0-A)",
            "iteration_zero_semantics": (
                "evaluated generation/iteration 0 evaluates the phase initialization "
                "and counts toward the phase allocation"
            ),
        }


@dataclass(frozen=True)
class HybridPhaseRecord(Generic[EvaluationT]):
    """Governed state captured after one evaluated population in either phase."""

    phase: str
    phase_index: int
    phase_best_mask: np.ndarray
    phase_best_selected_feature_count: int
    phase_best_evaluation: EvaluationT
    run_best_mask: np.ndarray
    run_best_selected_feature_count: int
    run_best_evaluation: EvaluationT
    run_best_improved: bool
    request_count: int
    cumulative_requests: int
    cumulative_unique_evaluations: int
    cumulative_cache_hits: int
    population_diversity: float
    repair_count: int
    kinetics_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "phase_index": self.phase_index,
            "phase_best_mask": to_json_compatible(self.phase_best_mask),
            "phase_best_selected_feature_count": self.phase_best_selected_feature_count,
            "phase_best_evaluation": to_json_compatible(self.phase_best_evaluation),
            "run_best_mask": to_json_compatible(self.run_best_mask),
            "run_best_selected_feature_count": self.run_best_selected_feature_count,
            "run_best_evaluation": to_json_compatible(self.run_best_evaluation),
            "run_best_improved": self.run_best_improved,
            "request_count": self.request_count,
            "cumulative_requests": self.cumulative_requests,
            "cumulative_unique_evaluations": self.cumulative_unique_evaluations,
            "cumulative_cache_hits": self.cumulative_cache_hits,
            "population_diversity": self.population_diversity,
            "repair_count": self.repair_count,
            "kinetics_value": self.kinetics_value,
        }


@dataclass(frozen=True)
class HybridResult(Generic[EvaluationT]):
    """Complete reproducible result of one hybrid BPSO+BGWO run."""

    optimizer_name: str
    design_identifier: str
    optimizer_seed: int
    dimensions: int
    population_size: int
    bpso_evaluated_generations: int
    bgwo_evaluated_iterations: int
    stop_reason: str
    best_mask: np.ndarray
    best_selected_feature_count: int
    best_evaluation: EvaluationT
    best_phase: str
    best_phase_index: int
    total_candidate_requests: int
    unique_evaluations: int
    cache_hits: int
    evaluator_calls: int
    bpso_requests: int
    bgwo_requests: int
    bpso_unique_evaluations: int
    bgwo_new_unique_evaluations: int
    bpso_cache_hits: int
    bgwo_cache_hits: int
    elite_masks: np.ndarray
    elite_hashes: tuple[str, ...]
    elite_count: int
    bgwo_initial_population: np.ndarray
    convergence_history: tuple[HybridPhaseRecord[EvaluationT], ...]
    reproducible_re_code: str
    best_feasible: bool | None
    best_normalized_violation: float | None
    repair_count: int
    configuration_snapshot: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "optimizer_name": self.optimizer_name,
            "design_identifier": self.design_identifier,
            "optimizer_seed": self.optimizer_seed,
            "dimensions": self.dimensions,
            "population_size": self.population_size,
            "bpso_evaluated_generations": self.bpso_evaluated_generations,
            "bgwo_evaluated_iterations": self.bgwo_evaluated_iterations,
            "stop_reason": self.stop_reason,
            "best_mask": self.best_mask,
            "best_selected_feature_count": self.best_selected_feature_count,
            "best_evaluation": self.best_evaluation,
            "best_phase": self.best_phase,
            "best_phase_index": self.best_phase_index,
            "total_candidate_requests": self.total_candidate_requests,
            "unique_evaluations": self.unique_evaluations,
            "cache_hits": self.cache_hits,
            "evaluator_calls": self.evaluator_calls,
            "bpso_requests": self.bpso_requests,
            "bgwo_requests": self.bgwo_requests,
            "bpso_unique_evaluations": self.bpso_unique_evaluations,
            "bgwo_new_unique_evaluations": self.bgwo_new_unique_evaluations,
            "bpso_cache_hits": self.bpso_cache_hits,
            "bgwo_cache_hits": self.bgwo_cache_hits,
            "elite_masks": self.elite_masks,
            "elite_hashes": list(self.elite_hashes),
            "elite_count": self.elite_count,
            "bgwo_initial_population": self.bgwo_initial_population,
            "convergence_history": [
                item.to_dict() for item in self.convergence_history
            ],
            "best_feasible": self.best_feasible,
            "best_normalized_violation": self.best_normalized_violation,
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


class RunLocalCache(Generic[EvaluationT]):
    """One run-local objective cache shared by the BPSO and BGWO phases.

    Every optimizer submission is a candidate request. A duplicate of an
    already-evaluated mask is a cache hit that reuses the recorded evaluation
    without calling the objective. ``evaluator_calls`` counts only real
    objective invocations and always equals ``unique_evaluations``.
    """

    def __init__(self, objective: Objective[EvaluationT], enabled: bool = True) -> None:
        if not callable(objective):
            raise TypeError("objective must be callable.")
        self._objective = objective
        self._enabled = bool(enabled)
        self._entries: dict[bytes, EvaluationT] = {}
        self._masks: dict[bytes, np.ndarray] = {}
        self.total_requests = 0
        self.unique_evaluations = 0
        self.cache_hits = 0
        self.evaluator_calls = 0

    def request(self, mask: np.ndarray | Sequence[int]) -> EvaluationT:
        """Submit one candidate mask; a duplicate resolves as a cache hit."""
        binary = _validated_mask(mask)
        self.total_requests += 1
        key = canonical_mask_key(binary)
        if self._enabled and key in self._entries:
            self.cache_hits += 1
            return self._entries[key]
        evaluation = self._objective(binary.copy())
        self.unique_evaluations += 1
        self.evaluator_calls += 1
        if self._enabled:
            self._entries[key] = evaluation
            self._masks[key] = binary.copy()
        return evaluation

    def distinct_mask_array(self) -> np.ndarray:
        """Return insertion-ordered distinct evaluated masks (uint8, M x D)."""
        if not self._enabled:
            return np.zeros((0, 0), dtype=np.uint8)
        if not self._masks:
            return np.zeros((0, 0), dtype=np.uint8)
        return np.asarray(list(self._masks.values()), dtype=np.uint8)

    def evaluation_for_mask(self, mask: np.ndarray | Sequence[int]) -> EvaluationT:
        _ = _validated_mask(mask)
        return self._entries[canonical_mask_key(mask)]

    def distinct_evaluations(self) -> list[EvaluationT]:
        return [self._entries[key] for key in self._masks]

    def snapshot(self) -> dict[str, int]:
        return {
            "total_requests": self.total_requests,
            "unique_evaluations": self.unique_evaluations,
            "cache_hits": self.cache_hits,
            "evaluator_calls": self.evaluator_calls,
        }


def select_elites(
    cache: RunLocalCache[EvaluationT],
    elite_count: int,
    is_better: Comparator[EvaluationT],
) -> np.ndarray:
    """Return the top-``elite_count`` DISTINCT evaluated masks by comparator.

    Performs zero evaluator calls: the elite set is a pure re-ranking of the
    already-evaluated run-local candidates. Feasible-first behavior is
    delegated to the injected strict comparator, matching the frozen ranking.
    """
    if not isinstance(cache, RunLocalCache):
        raise TypeError("Elite selection requires a RunLocalCache.")
    masks = cache.distinct_mask_array()
    if masks.ndim != 2 or masks.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    evaluations = cache.distinct_evaluations()
    ordered = _rank_indices(masks, evaluations, is_better)
    selected = [int(index) for index in ordered[:elite_count]]
    if not selected:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.asarray([masks[index] for index in selected], dtype=np.uint8)


def build_hybrid_initial_population(
    config: HybridConfig,
    generator: np.random.Generator,
) -> np.ndarray:
    """Build the BPSO-phase initialization: all-ones anchor + random exact-K.

    Wolf 0 is the canonical a-priori all-ones mask (K=dimensions). Wolves
    1..N-1 draw exact cardinalities with replacement from the locked pool and
    generate fresh exact-K masks under ``generator``. No frozen winner identity
    is injected.
    """
    _require_generator(generator)
    population = np.zeros((config.population_size, config.dimensions), dtype=np.uint8)
    population[0, :] = 1
    for row in range(1, config.population_size):
        cardinality = int(config.cardinality_pool[generator.integers(len(config.cardinality_pool))])
        population[row] = exact_cardinality_mask(config.dimensions, cardinality, generator)
    _validate_population(population)
    return population


def build_bgwo_initial_population(
    config: HybridConfig,
    elites: np.ndarray | Sequence[Sequence[int]],
    generator: np.random.Generator,
) -> np.ndarray:
    """Build the BGWO-phase initialization from the locked transfer rule.

    Wolf 0 remains the all-ones anchor. Wolves 1..elite_count are the
    transferred elites (or, if fewer elites are available, fresh random
    exact-K masks). The remaining wolves are governed random exact-K masks
    drawn under the BGWO-phase RNG.
    """
    _require_generator(generator)
    elite_array = np.asarray(elites, dtype=np.uint8)
    if elite_array.size == 0:
        elite_array = elite_array.reshape((0, config.dimensions))
    elif elite_array.ndim == 1:
        elite_array = elite_array.reshape((1, -1))
    available = elite_array.shape[0] if elite_array.ndim == 2 else 0
    if available:
        if elite_array.shape != (available, config.dimensions):
            raise ValueError("Elite masks must match the configured dimensions.")
        if not np.isin(elite_array, (0, 1)).all():
            raise ValueError("Elite masks must be binary.")
        if np.any(elite_array.sum(axis=1) < config.minimum_selected_features):
            raise ValueError("Every elite mask must keep at least one feature.")
    population = np.zeros((config.population_size, config.dimensions), dtype=np.uint8)
    population[0, :] = 1
    slot = 1
    for index in range(min(available, config.elite_count)):
        population[slot] = elite_array[index]
        slot += 1
    while slot < config.population_size:
        cardinality = int(config.cardinality_pool[generator.integers(len(config.cardinality_pool))])
        population[slot] = exact_cardinality_mask(config.dimensions, cardinality, generator)
        slot += 1
    _validate_population(population)
    return population


def hybrid_config_from_yaml(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    bpso_config_path: Path | str = DEFAULT_BPSO_CONFIG_PATH,
    bgwo_config_path: Path | str = DEFAULT_BGWO_CONFIG_PATH,
) -> HybridConfig:
    """Load the locked V1.0-A hybrid constants from the governed YAML files.

    Structural constants come from ``hybrid_v10.yaml``; the BPSO/BGWO phase
    mechanics constants come from the frozen ``bpso.yaml`` and ``bgwo_v09.yaml``
    so that no numeric value is silently invented.
    """
    try:
        import yaml  # local import keeps the pure optimizer evaluator-agnostic

        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        bpso_payload = yaml.safe_load(Path(bpso_config_path).read_text(encoding="utf-8"))
        bgwo_payload = yaml.safe_load(Path(bgwo_config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, AttributeError, yaml.YAMLError, TypeError) as exc:
        raise ValueError(f"Cannot load governed YAML for the hybrid protocol: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(bpso_payload, dict) or not isinstance(
        bgwo_payload, dict
    ):
        raise ValueError("Hybrid YAML payloads must be mappings.")

    dataset = payload.get("dataset", {})
    design = payload.get("hybrid_design", {})
    phases = design.get("phase_budgets", {})
    seeds = payload.get("seeds", {})
    transfer = payload.get("knowledge_transfer", {})
    initialization = payload.get("initialization", {})

    anchors = initialization.get("governed_anchors", [])
    pool: list[int] = []
    if isinstance(anchors, list):
        for anchor in anchors:
            match = re.search(r"\[([0-9,\s]+)\]", str(anchor))
            if match:
                pool = [int(value) for value in match.group(1).split(",")]
    if not pool:
        raise ValueError("The hybrid cardinality pool could not be parsed from the YAML.")

    bpso_optimizer = bpso_payload.get("optimizer", {})
    velocity = bpso_optimizer.get("velocity", {})
    inertia = bpso_optimizer.get("inertia", {})
    bgwo_optimizer = bgwo_payload.get("optimizer", {})
    transfer_cfg = bgwo_optimizer.get("binary_transfer", {})
    control = bgwo_optimizer.get("canonical_binary_gwo", {}).get("control_parameter_a", {})

    return HybridConfig(
        dimensions=int(dataset.get("feature_space_count", 43)),
        population_size=int(design.get("population_size_per_phase", 12)),
        bpso_evaluated_generations=int(phases.get("bpso_evaluated_generations_per_run", 8)),
        bgwo_evaluated_iterations=int(phases.get("bgwo_evaluated_iterations_per_run", 8)),
        cardinality_pool=tuple(pool),
        elite_count=int(transfer.get("elite_count", 3)),
        bgwo_phase_rng_offset=int(seeds.get("optimizer_seed_phase_rng_offset_bgwo", 10000)),
        velocity_initial_min=float(velocity.get("initialization_bounds", [-1.0, 1.0])[0]),
        velocity_initial_max=float(velocity.get("initialization_bounds", [-1.0, 1.0])[1]),
        velocity_clamp_min=float(velocity.get("clamp", [-6.0, 6.0])[0]),
        velocity_clamp_max=float(velocity.get("clamp", [-6.0, 6.0])[1]),
        inertia_start=float(inertia.get("start", 0.9)),
        inertia_end=float(inertia.get("end", 0.4)),
        cognitive_coefficient=float(bpso_optimizer.get("cognitive_coefficient", 2.0)),
        social_coefficient=float(bpso_optimizer.get("social_coefficient", 2.0)),
        sigmoid_clamp_min=float(transfer_cfg.get("input_clamp", [-6.0, 6.0])[0]),
        sigmoid_clamp_max=float(transfer_cfg.get("input_clamp", [-6.0, 6.0])[1]),
        control_parameter_start=float(control.get("start", 2.0)),
        control_parameter_end=float(control.get("end", 0.0)),
        minimum_selected_features=int(
            bgwo_optimizer.get("repair", {}).get("minimum_selected_features", 1)
        ),
        cache_enabled=True,
        optimizer_name=OPTIMIZER_IDENTIFIER,
        design_identifier=DESIGN_IDENTIFIER,
    )


class HybridBPSOBGWO(Generic[EvaluationT]):
    """Generic deterministic hybrid engine controlled by a strict comparator."""

    def __init__(
        self,
        config: HybridConfig,
        objective: Objective[EvaluationT],
        is_better: Comparator[EvaluationT],
    ) -> None:
        if not callable(objective) or not callable(is_better):
            raise TypeError("objective and is_better must be callable.")
        self.config = config
        self.objective = objective
        self.is_better = is_better

    def optimize(self, optimizer_seed: int) -> HybridResult[EvaluationT]:
        """Execute one fixed-budget hybrid BPSO -> elite transfer -> BGWO run."""
        config = self.config
        bpso_generator = np.random.Generator(np.random.PCG64(int(optimizer_seed)))
        cache: RunLocalCache[EvaluationT] = RunLocalCache(self.objective, config.cache_enabled)

        # ------------------------------------------------------------------
        # BPSO exploration phase (fixed budget, no early stopping)
        # ------------------------------------------------------------------
        positions = build_hybrid_initial_population(config, bpso_generator)
        velocities = bpso_generator.uniform(
            config.velocity_initial_min,
            config.velocity_initial_max,
            size=(config.population_size, config.dimensions),
        )
        inertia_schedule = linear_inertia_schedule(
            config.bpso_evaluated_generations,
            config.inertia_start,
            config.inertia_end,
        )
        personal_best_masks: np.ndarray | None = None
        personal_best_evaluations: list[EvaluationT] = []
        run_best_mask: np.ndarray | None = None
        run_best_evaluation: EvaluationT | None = None
        run_best_phase = ""
        run_best_phase_index = 0
        history: list[HybridPhaseRecord[EvaluationT]] = []
        bpso_repairs = 0
        generation_repairs = 0
        applied_inertia: float | None = None
        phase_requests_bound = config.population_size

        for generation in range(config.bpso_evaluated_generations):
            evaluations = [cache.request(mask) for mask in positions]
            if generation == 0:
                personal_best_masks = positions.copy()
                personal_best_evaluations = list(evaluations)
            else:
                assert personal_best_masks is not None
                for index, evaluation in enumerate(evaluations):
                    relation = _evaluation_relation(self.is_better, evaluation, personal_best_evaluations[index])
                    if relation < 0 or (
                        relation == 0
                        and _mask_tuple(positions[index]) < _mask_tuple(personal_best_masks[index])
                    ):
                        personal_best_masks[index] = positions[index].copy()
                        personal_best_evaluations[index] = evaluation

            phase_best_index = _best_index(positions, evaluations, self.is_better)
            candidate = positions[phase_best_index]
            candidate_evaluation = evaluations[phase_best_index]
            considered = "unknown"
            if run_best_mask is None:
                run_best_mask = candidate.copy()
                run_best_evaluation = candidate_evaluation
                run_best_phase = "BPSO"
                run_best_phase_index = generation
                considered = "BPSO"
            else:
                assert run_best_evaluation is not None
                relation = _evaluation_relation(self.is_better, candidate_evaluation, run_best_evaluation)
                if relation < 0 or (
                    relation == 0 and _mask_tuple(candidate) < _mask_tuple(run_best_mask)
                ):
                    run_best_mask = candidate.copy()
                    run_best_evaluation = candidate_evaluation
                    run_best_phase = "BPSO"
                    run_best_phase_index = generation
                    considered = "BPSO"

            history.append(
                HybridPhaseRecord(
                    phase="BPSO",
                    phase_index=generation,
                    phase_best_mask=candidate.copy(),
                    phase_best_selected_feature_count=int(candidate.sum()),
                    phase_best_evaluation=candidate_evaluation,
                    run_best_mask=run_best_mask.copy(),
                    run_best_selected_feature_count=int(run_best_mask.sum()),
                    run_best_evaluation=run_best_evaluation,
                    run_best_improved=considered == "BPSO",
                    request_count=phase_requests_bound,
                    cumulative_requests=cache.total_requests,
                    cumulative_unique_evaluations=cache.unique_evaluations,
                    cumulative_cache_hits=cache.cache_hits,
                    population_diversity=mean_pairwise_normalized_hamming(positions),
                    repair_count=generation_repairs,
                    kinetics_value=applied_inertia,
                )
            )
            if generation == config.bpso_evaluated_generations - 1:
                break

            assert personal_best_masks is not None
            assert run_best_mask is not None
            applied_inertia = inertia_schedule[generation]
            next_velocities = np.empty_like(velocities)
            next_positions = np.empty_like(positions)
            generation_repairs = 0
            for index in range(config.population_size):
                next_velocities[index] = update_velocity(
                    velocities[index],
                    positions[index],
                    personal_best_masks[index],
                    run_best_mask,
                    inertia=applied_inertia,
                    cognitive_coefficient=config.cognitive_coefficient,
                    social_coefficient=config.social_coefficient,
                    clamp_min=config.velocity_clamp_min,
                    clamp_max=config.velocity_clamp_max,
                    generator=bpso_generator,
                )
                next_positions[index], repaired, _ = sample_binary_position(
                    next_velocities[index], bpso_generator
                )
                generation_repairs += int(repaired)
            bpso_repairs += generation_repairs
            velocities = next_velocities
            positions = next_positions

        bpso_requests_snapshot = cache.total_requests
        bpso_unique_snapshot = cache.unique_evaluations
        bpso_hits_snapshot = cache.cache_hits
        final_bpso_positions = positions.copy()

        # ------------------------------------------------------------------
        # Elite selection (zero evaluator calls)
        # ------------------------------------------------------------------
        elite_masks = select_elites(cache, config.elite_count, self.is_better)
        placed_elites = int(min(elite_masks.shape[0], config.elite_count)) if elite_masks.ndim == 2 else 0
        elite_hashes = tuple(
            hashlib.sha256(elite_masks[index].tobytes()).hexdigest()
            for index in range(elite_masks.shape[0])
        )

        # ------------------------------------------------------------------
        # BGWO refinement phase (fixed budget, no early stopping)
        # ------------------------------------------------------------------
        bgwo_generator = np.random.Generator(
            np.random.PCG64(int(optimizer_seed) + config.bgwo_phase_rng_offset)
        )
        population = build_bgwo_initial_population(config, elite_masks, bgwo_generator)
        bgwo_initial_population_snapshot = population.copy()
        latent_positions = population.astype(float, copy=True)
        a_schedule = linear_control_schedule(
            config.bgwo_evaluated_iterations,
            config.control_parameter_start,
            config.control_parameter_end,
        )
        bgwo_repairs = 0
        update_repairs = 0
        applied_a: float | None = None

        for iteration in range(config.bgwo_evaluated_iterations):
            evaluations = [cache.request(mask) for mask in population]
            ordered = _rank_indices(population, evaluations, self.is_better)
            alpha_index, beta_index, delta_index = ordered[:3]
            candidate = population[alpha_index]
            candidate_evaluation = evaluations[alpha_index]
            considered = "unknown"
            assert run_best_mask is not None and run_best_evaluation is not None
            relation = _evaluation_relation(self.is_better, candidate_evaluation, run_best_evaluation)
            if relation < 0 or (
                relation == 0 and _mask_tuple(candidate) < _mask_tuple(run_best_mask)
            ):
                run_best_mask = candidate.copy()
                run_best_evaluation = candidate_evaluation
                run_best_phase = "BGWO"
                run_best_phase_index = iteration
                considered = "BGWO"

            history.append(
                HybridPhaseRecord(
                    phase="BGWO",
                    phase_index=iteration,
                    phase_best_mask=candidate.copy(),
                    phase_best_selected_feature_count=int(candidate.sum()),
                    phase_best_evaluation=candidate_evaluation,
                    run_best_mask=run_best_mask.copy(),
                    run_best_selected_feature_count=int(run_best_mask.sum()),
                    run_best_evaluation=run_best_evaluation,
                    run_best_improved=considered == "BGWO",
                    request_count=phase_requests_bound,
                    cumulative_requests=cache.total_requests,
                    cumulative_unique_evaluations=cache.unique_evaluations,
                    cumulative_cache_hits=cache.cache_hits,
                    population_diversity=mean_pairwise_normalized_hamming(population),
                    repair_count=update_repairs,
                    kinetics_value=applied_a,
                )
            )
            if iteration == config.bgwo_evaluated_iterations - 1:
                break

            applied_a = a_schedule[iteration]
            alpha = latent_positions[alpha_index]
            beta = latent_positions[beta_index]
            delta = latent_positions[delta_index]
            next_latent = np.empty_like(latent_positions)
            next_population = np.empty_like(population)
            update_repairs = 0
            for idx in range(config.population_size):
                current = latent_positions[idx]
                r1_alpha = bgwo_generator.random(config.dimensions)
                r2_alpha = bgwo_generator.random(config.dimensions)
                r1_beta = bgwo_generator.random(config.dimensions)
                r2_beta = bgwo_generator.random(config.dimensions)
                r1_delta = bgwo_generator.random(config.dimensions)
                r2_delta = bgwo_generator.random(config.dimensions)
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
                    config.sigmoid_clamp_min,
                    config.sigmoid_clamp_max,
                )
                sampled, repaired, _, _, _ = sample_binary_mask(
                    next_latent[idx],
                    clamp_min=config.sigmoid_clamp_min,
                    clamp_max=config.sigmoid_clamp_max,
                    generator=bgwo_generator,
                )
                next_population[idx] = sampled
                update_repairs += int(repaired)
            bgwo_repairs += update_repairs
            latent_positions = next_latent
            population = next_population

        bgwo_total_requests = cache.total_requests - bpso_requests_snapshot
        bgwo_unique = cache.unique_evaluations - bpso_unique_snapshot
        bgwo_hits = cache.cache_hits - bpso_hits_snapshot

        # ------------------------------------------------------------------
        # Invariant enforcement (fail closed)
        # ------------------------------------------------------------------
        if cache.total_requests != cache.unique_evaluations + cache.cache_hits:
            raise RuntimeError("Hybrid cache accounting invariant failed.")
        if cache.evaluator_calls != cache.unique_evaluations:
            raise RuntimeError("Hybrid evaluator-call invariant failed.")
        if cache.total_requests != config.per_run_request_allocation:
            raise RuntimeError("Hybrid must consume its complete fixed request budget.")
        if bpso_requests_snapshot != config.bpso_request_allocation:
            raise RuntimeError("BPSO phase must consume its complete fixed allocation.")
        if bgwo_total_requests != config.bgwo_request_allocation:
            raise RuntimeError("BGWO phase must consume its complete fixed allocation.")
        if bgwo_hits < placed_elites:
            raise RuntimeError("Transferred elite candidates must resolve as shared-cache hits.")
        assert run_best_mask is not None and run_best_evaluation is not None

        best_feasible = getattr(run_best_evaluation, "feasible", None)
        best_violation = getattr(run_best_evaluation, "normalized_violation", None)
        best_violation = float(best_violation) if best_violation is not None else None

        snapshot = config.to_dict()
        snapshot["optimizer_seed"] = int(optimizer_seed)
        snapshot["placed_elite_count"] = placed_elites
        snapshot["elite_count"] = config.elite_count
        result = HybridResult(
            optimizer_name=config.optimizer_name,
            design_identifier=config.design_identifier,
            optimizer_seed=int(optimizer_seed),
            dimensions=config.dimensions,
            population_size=config.population_size,
            bpso_evaluated_generations=config.bpso_evaluated_generations,
            bgwo_evaluated_iterations=config.bgwo_evaluated_iterations,
            stop_reason=STOP_FIXED_BUDGET_EXHAUSTED,
            best_mask=run_best_mask.copy(),
            best_selected_feature_count=int(run_best_mask.sum()),
            best_evaluation=run_best_evaluation,
            best_phase=run_best_phase,
            best_phase_index=run_best_phase_index,
            total_candidate_requests=cache.total_requests,
            unique_evaluations=cache.unique_evaluations,
            cache_hits=cache.cache_hits,
            evaluator_calls=cache.evaluator_calls,
            bpso_requests=bpso_requests_snapshot,
            bgwo_requests=bgwo_total_requests,
            bpso_unique_evaluations=bpso_unique_snapshot,
            bgwo_new_unique_evaluations=bgwo_unique,
            bpso_cache_hits=bpso_hits_snapshot,
            bgwo_cache_hits=bgwo_hits,
            elite_masks=elite_masks.copy(),
            elite_hashes=elite_hashes,
            elite_count=config.elite_count,
            bgwo_initial_population=bgwo_initial_population_snapshot,
            convergence_history=tuple(history),
            reproducible_re_code=_reproducible_re_code(config, int(optimizer_seed)),
            best_feasible=best_feasible,
            best_normalized_violation=best_violation,
            repair_count=bpso_repairs + bgwo_repairs,
            configuration_snapshot=snapshot,
        )
        return result


def _rank_indices(
    population: np.ndarray,
    evaluations: Sequence[EvaluationT],
    is_better: Comparator[EvaluationT],
) -> tuple[int, ...]:
    ordered = list(range(len(evaluations)))
    for left in range(len(ordered)):
        best = left
        for right in range(left + 1, len(ordered)):
            idx_right = ordered[right]
            idx_best = ordered[best]
            relation = _evaluation_relation(is_better, evaluations[idx_right], evaluations[idx_best])
            if relation < 0 or (
                relation == 0
                and _mask_tuple(population[idx_right]) < _mask_tuple(population[idx_best])
            ):
                best = right
        ordered[left], ordered[best] = ordered[best], ordered[left]
    return tuple(ordered)


def _best_index(
    positions: np.ndarray,
    evaluations: Sequence[EvaluationT],
    is_better: Comparator[EvaluationT],
) -> int:
    best = 0
    for index in range(1, len(evaluations)):
        relation = _evaluation_relation(is_better, evaluations[index], evaluations[best])
        if relation < 0 or (
            relation == 0 and _mask_tuple(positions[index]) < _mask_tuple(positions[best])
        ):
            best = index
    return best


def _evaluation_relation(
    is_better: Comparator[EvaluationT],
    left: EvaluationT,
    right: EvaluationT,
) -> int:
    left_better = bool(is_better(left, right))
    right_better = bool(is_better(right, left))
    if left_better and right_better:
        raise ComparatorContractError("The injected comparator must be strict and asymmetric.")
    if left_better:
        return -1
    if right_better:
        return 1
    return 0


def _require_generator(generator: np.random.Generator) -> None:
    if not isinstance(generator, np.random.Generator) or not isinstance(
        generator.bit_generator, np.random.PCG64
    ):
        raise TypeError("Hybrid randomness requires numpy.random.Generator with PCG64.")


def _validated_mask(
    mask: np.ndarray | Sequence[int], *, expected_length: int | None = None
) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 1 or (expected_length is not None and len(array) != expected_length):
        raise ValueError("A binary mask must be one-dimensional with the expected length.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("Binary masks may contain only zero and one.")
    return array.astype(np.uint8, copy=False)


def _validate_population(population: np.ndarray) -> None:
    if population.ndim != 2 or population.shape[1] < 2:
        raise ValueError("Population must be a nonempty two-dimensional matrix.")
    if not np.isin(population, (0, 1)).all():
        raise ValueError("Every agent position must be binary.")
    if np.any(population.sum(axis=1) < 1):
        raise ValueError("Every agent must retain at least one selected dimension.")


def _mask_tuple(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in mask)


def _reproducible_re_code(config: HybridConfig, optimizer_seed: int) -> str:
    """Return a compact reproducible run-identity code."""
    identity = {
        "optimizer": config.optimizer_name,
        "design": config.design_identifier,
        "seed": int(optimizer_seed),
        "dimensions": config.dimensions,
        "population": config.population_size,
        "bpso_generations": config.bpso_evaluated_generations,
        "bgwo_iterations": config.bgwo_evaluated_iterations,
        "rng_offset": config.bgwo_phase_rng_offset,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


__all__ = [
    "ComparatorContractError",
    "DESIGN_IDENTIFIER",
    "HybridBPSOBGWO",
    "HybridConfig",
    "HybridPhaseRecord",
    "HybridResult",
    "OPTIMIZER_IDENTIFIER",
    "RunLocalCache",
    "STOP_FIXED_BUDGET_EXHAUSTED",
    "build_bgwo_initial_population",
    "build_hybrid_initial_population",
    "hybrid_config_from_yaml",
    "select_elites",
]