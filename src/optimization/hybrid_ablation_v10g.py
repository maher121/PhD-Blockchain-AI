"""V1.0-G2 paired BPSO->BGWO elite-transfer ablation engine (implementation only).

This module implements the V1.0-G1 locked protocol
``V10G_BPSO_TO_BGWO_ELITE_TRANSFER_ABLATION`` as a pure, deterministic,
evaluator-agnostic paired engine. It carries no dataset, classifier, metric,
or TEST semantics: the objective and the strict comparator are injected, so
G2 tests can use synthetic/mock evaluators and toy dimensions without ever
touching production data or the production optimizer seeds 3042-3046.

The engine REUSES the frozen, tested primitives from ``bpso.py``, ``bgwo.py``
and ``hybrid_bpso_bgwo.py`` without modifying them:

* ``build_hybrid_initial_population`` for the BPSO phase initialization;
* ``update_velocity`` + ``sample_binary_position`` for BPSO updates;
* ``exact_cardinality_mask`` for every governed exact-K draw;
* ``RunLocalCache`` and ``select_elites`` for the run-local shared cache and
  the zero-call elite transfer;
* ``linear_inertia_schedule`` / ``linear_control_schedule`` for the frozen
  BPSO/BGWO schedules;
* ``mean_pairwise_normalized_hamming`` for population diversity records.

The single manipulated factor is the content of BGWO rows 1-3:

* WITH:  rows 1-3 = the top three DISTINCT BPSO-evaluated masks (elites);
         rows 4-11 = governed common random exact-K masks; the three shadow
         filler masks are generated and audited but NOT placed or evaluated.
* WITHOUT: rows 1-3 = the three deterministic shadow filler masks; rows 4-11
         = byte-identical governed common random exact-K masks; elite
         selection is never invoked and the filler construction never
         inspects BPSO masks, elites, fitness, cache, winners, or TEST.

The paired-run function asserts the locked invariants and FAILS CLOSED when
any of them is violated (paired BPSO outputs, BGWO row 0, BGWO rows 4-11, and
the pre-update common BGWO RNG state must be identical). Rows 1-3 collisions
are retained and reported, never rejected or redrawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
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
from src.optimization.hybrid_bpso_bgwo import (
    ComparatorContractError,
    RunLocalCache,
    build_hybrid_initial_population,
    select_elites,
)


EvaluationT = TypeVar("EvaluationT")
Objective = Callable[[np.ndarray], EvaluationT]
Comparator = Callable[[EvaluationT, EvaluationT], bool]

ABLATION_IDENTIFIER = "V10G_BPSO_TO_BGWO_ELITE_TRANSFER_ABLATION"
DESIGN_IDENTIFIER = "BPSO_BGWO_SEQUENTIAL_50_50_ELITE3"
STOP_FIXED_BUDGET_EXHAUSTED = "FIXED_BUDGET_EXHAUSTED"

# ---------------------------------------------------------------------------
# Frozen V1.0-G1 production constants (verified against the semantic lock).
# ---------------------------------------------------------------------------
PRODUCTION_DIMENSIONS = 43
PRODUCTION_POPULATION = 12
PRODUCTION_BPSO_GENERATIONS = 8
PRODUCTION_BGWO_ITERATIONS = 8
PRODUCTION_ELITE_COUNT = 3
BPSO_PHASE_RNG_OFFSET = 0
BGWO_PHASE_RNG_OFFSET = 10000
FILLER_PROTOCOL_TAG = 0x56313047
FILLER_STREAM_TAG = 0x46494C4C
CARDINALITY_POOL: tuple[int, ...] = (4, 8, 11, 14, 18, 22, 26, 30, 34, 38)
FILLER_ROW_COUNT = 3
ROW_ANCHOR_INDEX = 0
ROW_TREATMENT_SLICE = slice(1, 1 + FILLER_ROW_COUNT)
ROW_COMMON_SLICE = slice(1 + FILLER_ROW_COUNT, None)

PRODUCTION_OPTIMIZER_SEEDS: tuple[int, ...] = (3042, 3043, 3044, 3045, 3046)
PRODUCTION_MODEL_ATTACK_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
PRODUCTION_REQUESTS_PER_BPSO = PRODUCTION_POPULATION * PRODUCTION_BPSO_GENERATIONS
PRODUCTION_REQUESTS_PER_BGWO = PRODUCTION_POPULATION * PRODUCTION_BGWO_ITERATIONS
PRODUCTION_REQUESTS_PER_RUN = PRODUCTION_REQUESTS_PER_BPSO + PRODUCTION_REQUESTS_PER_BGWO
PRODUCTION_REQUESTS_PER_ARM = 5 * PRODUCTION_REQUESTS_PER_RUN
PRODUCTION_REQUESTS_ALL_ABLATION = 2 * PRODUCTION_REQUESTS_PER_ARM
PRODUCTION_FITS_PER_UNIQUE_EVALUATION = 5

# Frozen BPSO/BGWO mechanics (must equal the governed bpso.yaml / bgwo_v09.yaml
# values consumed by the frozen hybrid engine).
FROZEN_MECHANICS = {
    "velocity_initial_min": -1.0,
    "velocity_initial_max": 1.0,
    "velocity_clamp_min": -6.0,
    "velocity_clamp_max": 6.0,
    "inertia_start": 0.9,
    "inertia_end": 0.4,
    "cognitive_coefficient": 2.0,
    "social_coefficient": 2.0,
    "sigmoid_clamp_min": -6.0,
    "sigmoid_clamp_max": 6.0,
    "control_parameter_start": 2.0,
    "control_parameter_end": 0.0,
    "minimum_selected_features": 1,
    "cache_enabled": True,
}

EXPECTED_PRODUCTION_CONSTANTS: Mapping[str, Any] = {
    "dimensions": PRODUCTION_DIMENSIONS,
    "population_size": PRODUCTION_POPULATION,
    "bpso_evaluated_generations": PRODUCTION_BPSO_GENERATIONS,
    "bgwo_evaluated_iterations": PRODUCTION_BGWO_ITERATIONS,
    "elite_count": PRODUCTION_ELITE_COUNT,
    "cardinality_pool": CARDINALITY_POOL,
    "bpso_phase_rng_offset": BPSO_PHASE_RNG_OFFSET,
    "bgwo_phase_rng_offset": BGWO_PHASE_RNG_OFFSET,
    "filler_protocol_tag": FILLER_PROTOCOL_TAG,
    "filler_stream_tag": FILLER_STREAM_TAG,
    "filler_row_count": FILLER_ROW_COUNT,
    "requests_per_bpso": PRODUCTION_REQUESTS_PER_BPSO,
    "requests_per_bgwo": PRODUCTION_REQUESTS_PER_BGWO,
    "requests_per_run": PRODUCTION_REQUESTS_PER_RUN,
    "requests_per_arm": PRODUCTION_REQUESTS_PER_ARM,
    "requests_all_ablation": PRODUCTION_REQUESTS_ALL_ABLATION,
    "fits_per_unique_evaluation": PRODUCTION_FITS_PER_UNIQUE_EVALUATION,
    "optimizer_seeds": PRODUCTION_OPTIMIZER_SEEDS,
    "model_attack_seeds": PRODUCTION_MODEL_ATTACK_SEEDS,
    **FROZEN_MECHANICS,
}


class AblationVariant(str, Enum):
    WITH_ELITE_TRANSFER = "WITH_ELITE_TRANSFER"
    WITHOUT_ELITE_TRANSFER = "WITHOUT_ELITE_TRANSFER"


class AblationError(RuntimeError):
    """Base error for a fail-closed V1.0-G2 ablation execution."""


class AblationNoGoError(AblationError):
    """Raised when a mandatory G1 gate fails or a forbidden action is attempted."""


class AblationConfigError(ValueError):
    """Raised when an ablation configuration violates the frozen contract."""


class PairedInvariantViolation(AblationError):
    """Raised when a mandatory paired invariant fails (fail closed)."""


@dataclass(frozen=True)
class AblationConfig:
    """Validated settings for one paired or single ablation arm."""

    dimensions: int
    population_size: int
    bpso_evaluated_generations: int
    bgwo_evaluated_iterations: int
    cardinality_pool: tuple[int, ...]
    elite_count: int = PRODUCTION_ELITE_COUNT
    bgwo_phase_rng_offset: int = BGWO_PHASE_RNG_OFFSET
    bpso_phase_rng_offset: int = BPSO_PHASE_RNG_OFFSET
    filler_protocol_tag: int = FILLER_PROTOCOL_TAG
    filler_stream_tag: int = FILLER_STREAM_TAG
    filler_row_count: int = FILLER_ROW_COUNT
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

    def __post_init__(self) -> None:
        if self.dimensions < 2:
            raise AblationConfigError("Ablation dimensions must be at least two.")
        if self.population_size < 4:
            raise AblationConfigError("Ablation population must be at least four.")
        if self.bpso_evaluated_generations < 1 or self.bgwo_evaluated_iterations < 1:
            raise AblationConfigError("Both ablation phases must evaluate at least once.")
        if not 1 <= self.elite_count < self.population_size:
            raise AblationConfigError("Elite count must lie within the population.")
        if not 1 <= self.filler_row_count == self.elite_count:
            raise AblationConfigError(
                "The shadow filler block must match the elite count exactly."
            )
        if not self.cardinality_pool:
            raise AblationConfigError("The ablation cardinality pool must be nonempty.")
        if any(not 1 <= value <= self.dimensions for value in self.cardinality_pool):
            raise AblationConfigError("Every cardinality must lie within the dimensions.")
        if self.bpso_phase_rng_offset == self.bgwo_phase_rng_offset:
            raise AblationConfigError("BPSO and BGWO phase RNG offsets must differ.")
        if (
            self.filler_protocol_tag == 0
            or self.filler_stream_tag == 0
            or self.filler_protocol_tag == self.filler_stream_tag
        ):
            raise AblationConfigError("Filler stream tags must be nonzero and distinct.")
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
            raise AblationConfigError("Ablation numeric parameters must be finite.")
        if self.velocity_initial_min >= self.velocity_initial_max:
            raise AblationConfigError("Velocity initialization bounds must be increasing.")
        if self.velocity_clamp_min >= self.velocity_clamp_max:
            raise AblationConfigError("Velocity clamp bounds must be increasing.")
        if (
            self.velocity_initial_min < self.velocity_clamp_min
            or self.velocity_initial_max > self.velocity_clamp_max
        ):
            raise AblationConfigError("Initial velocities must lie within the velocity clamp.")
        if self.inertia_start < self.inertia_end:
            raise AblationConfigError("The frozen inertia schedule must be non-increasing.")
        if self.sigmoid_clamp_min >= self.sigmoid_clamp_max:
            raise AblationConfigError("Sigmoid clamp bounds must be increasing.")
        if self.control_parameter_start < self.control_parameter_end:
            raise AblationConfigError("Control parameter schedule must be non-increasing.")
        if self.cognitive_coefficient < 0 or self.social_coefficient < 0:
            raise AblationConfigError("Cognitive/social coefficients must be nonnegative.")
        if self.minimum_selected_features != 1:
            raise AblationConfigError("V1.0-A locks exactly one minimum selected feature.")
        if not self.cache_enabled:
            raise AblationConfigError("The locked ablation requires cache_enabled=True.")

    @property
    def bpso_request_allocation(self) -> int:
        return self.population_size * self.bpso_evaluated_generations

    @property
    def bgwo_request_allocation(self) -> int:
        return self.population_size * self.bgwo_evaluated_iterations

    @property
    def per_run_request_allocation(self) -> int:
        return self.bpso_request_allocation + self.bgwo_request_allocation

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_identifier": ABLATION_IDENTIFIER,
            "design_identifier": DESIGN_IDENTIFIER,
            "dimensions": self.dimensions,
            "population_size": self.population_size,
            "bpso_evaluated_generations": self.bpso_evaluated_generations,
            "bgwo_evaluated_iterations": self.bgwo_evaluated_iterations,
            "elite_count": self.elite_count,
            "cardinality_pool": list(self.cardinality_pool),
            "bpso_phase_rng_offset": self.bpso_phase_rng_offset,
            "bgwo_phase_rng_offset": self.bgwo_phase_rng_offset,
            "filler_protocol_tag": self.filler_protocol_tag,
            "filler_stream_tag": self.filler_stream_tag,
            "filler_row_count": self.filler_row_count,
            "bpso_request_allocation": self.bpso_request_allocation,
            "bgwo_request_allocation": self.bgwo_request_allocation,
            "per_run_request_allocation": self.per_run_request_allocation,
            "velocity_initialization": [self.velocity_initial_min, self.velocity_initial_max],
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
            "early_stopping": "disabled (fixed-budget execution locked by V1.0-G1)",
            "iteration_zero_semantics": (
                "evaluated generation/iteration 0 counts toward the phase allocation"
            ),
        }


def production_ablation_config() -> AblationConfig:
    """Return the exact frozen V1.0-G1 production ablation configuration."""
    config = AblationConfig(
        dimensions=PRODUCTION_DIMENSIONS,
        population_size=PRODUCTION_POPULATION,
        bpso_evaluated_generations=PRODUCTION_BPSO_GENERATIONS,
        bgwo_evaluated_iterations=PRODUCTION_BGWO_ITERATIONS,
        cardinality_pool=CARDINALITY_POOL,
        elite_count=PRODUCTION_ELITE_COUNT,
        bgwo_phase_rng_offset=BGWO_PHASE_RNG_OFFSET,
        bpso_phase_rng_offset=BPSO_PHASE_RNG_OFFSET,
        filler_protocol_tag=FILLER_PROTOCOL_TAG,
        filler_stream_tag=FILLER_STREAM_TAG,
        filler_row_count=FILLER_ROW_COUNT,
        **dict(FROZEN_MECHANICS),
    )
    validate_production_constants(config)
    return config


def validate_production_constants(config: AblationConfig) -> bool:
    """Verify a configuration matches every frozen G1 production constant."""
    observed = {
        "dimensions": config.dimensions,
        "population_size": config.population_size,
        "bpso_evaluated_generations": config.bpso_evaluated_generations,
        "bgwo_evaluated_iterations": config.bgwo_evaluated_iterations,
        "elite_count": config.elite_count,
        "cardinality_pool": config.cardinality_pool,
        "bpso_phase_rng_offset": config.bpso_phase_rng_offset,
        "bgwo_phase_rng_offset": config.bgwo_phase_rng_offset,
        "filler_protocol_tag": config.filler_protocol_tag,
        "filler_stream_tag": config.filler_stream_tag,
        "filler_row_count": config.filler_row_count,
        "requests_per_bpso": config.bpso_request_allocation,
        "requests_per_bgwo": config.bgwo_request_allocation,
        "requests_per_run": config.per_run_request_allocation,
        "requests_per_arm": PRODUCTION_REQUESTS_PER_ARM,
        "requests_all_ablation": PRODUCTION_REQUESTS_ALL_ABLATION,
        "fits_per_unique_evaluation": PRODUCTION_FITS_PER_UNIQUE_EVALUATION,
        "optimizer_seeds": PRODUCTION_OPTIMIZER_SEEDS,
        "model_attack_seeds": PRODUCTION_MODEL_ATTACK_SEEDS,
        "velocity_initial_min": config.velocity_initial_min,
        "velocity_initial_max": config.velocity_initial_max,
        "velocity_clamp_min": config.velocity_clamp_min,
        "velocity_clamp_max": config.velocity_clamp_max,
        "inertia_start": config.inertia_start,
        "inertia_end": config.inertia_end,
        "cognitive_coefficient": config.cognitive_coefficient,
        "social_coefficient": config.social_coefficient,
        "sigmoid_clamp_min": config.sigmoid_clamp_min,
        "sigmoid_clamp_max": config.sigmoid_clamp_max,
        "control_parameter_start": config.control_parameter_start,
        "control_parameter_end": config.control_parameter_end,
        "minimum_selected_features": config.minimum_selected_features,
        "cache_enabled": config.cache_enabled,
    }
    for key, expected in EXPECTED_PRODUCTION_CONSTANTS.items():
        if observed[key] != expected:
            raise AblationConfigError(
                f"Production constant {key} drifted from the frozen G1 value: "
                f"{observed[key]} != {expected}."
            )
    return True


# ---------------------------------------------------------------------------
# Shadow filler block (deterministic, independent of BPSO information).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowFillerBlock:
    """Three deterministic exact-K filler masks from the locked filler stream.

    Construction consumes ONLY the optimizer seed, dimensions, and the locked
    cardinality pool through ``PCG64(SeedSequence([seed, 0x56313047,
    0x46494C4C]))``. It never inspects elites, BPSO candidates, fitness values,
    cache contents, previous winners, or TEST information. Duplicate masks are
    retained and reported, never rejected or redrawn.
    """

    optimizer_seed: int
    dimensions: int
    cardinality_pool: tuple[int, ...]
    filler_protocol_tag: int
    filler_stream_tag: int
    masks: np.ndarray
    cardinalities: tuple[int, ...]
    mask_sha256: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return int(self.masks.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "shadow_filler_kind": "GOVERNED_DETERMINISTIC_EXACT_K",
            "optimizer_seed": int(self.optimizer_seed),
            "dimensions": int(self.dimensions),
            "cardinality_pool": list(self.cardinality_pool),
            "filler_stream_identity": (
                f"PCG64(SeedSequence([{int(self.optimizer_seed)}, "
                f"{self.filler_protocol_tag}, {self.filler_stream_tag}]))"
            ),
            "filler_protocol_tag": int(self.filler_protocol_tag),
            "filler_stream_tag": int(self.filler_stream_tag),
            "row_count": self.row_count,
            "cardinalities": list(self.cardinalities),
            "mask_sha256": list(self.mask_sha256),
            "masks": to_json_compatible(self.masks),
            "duplicate_rejection": False,
        }


def generate_shadow_filler_block(
    optimizer_seed: int,
    *,
    dimensions: int = PRODUCTION_DIMENSIONS,
    cardinality_pool: Sequence[int] = CARDINALITY_POOL,
    filler_protocol_tag: int = FILLER_PROTOCOL_TAG,
    filler_stream_tag: int = FILLER_STREAM_TAG,
    row_count: int = FILLER_ROW_COUNT,
) -> ShadowFillerBlock:
    """Generate the locked three-row deterministic shadow filler block."""
    if dimensions < 1:
        raise AblationConfigError("Filler dimensions must be positive.")
    pool = tuple(int(value) for value in cardinality_pool)
    if not pool or any(not 1 <= value <= dimensions for value in pool):
        raise AblationConfigError("Filler cardinality pool must lie within dimensions.")
    if row_count < 1:
        raise AblationConfigError("The shadow filler block must have at least one row.")
    generator = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [int(optimizer_seed), int(filler_protocol_tag), int(filler_stream_tag)]
            )
        )
    )
    masks: list[np.ndarray] = []
    cardinalities: list[int] = []
    for _ in range(int(row_count)):
        cardinality = int(pool[int(generator.integers(len(pool)))])
        masks.append(exact_cardinality_mask(dimensions, cardinality, generator))
        cardinalities.append(int(cardinality))
    array = np.asarray(masks, dtype=np.uint8)
    hashes = tuple(
        hashlib.sha256(array[index].tobytes()).hexdigest()
        for index in range(array.shape[0])
    )
    return ShadowFillerBlock(
        optimizer_seed=int(optimizer_seed),
        dimensions=int(dimensions),
        cardinality_pool=pool,
        filler_protocol_tag=int(filler_protocol_tag),
        filler_stream_tag=int(filler_stream_tag),
        masks=array,
        cardinalities=tuple(cardinalities),
        mask_sha256=hashes,
    )


# ---------------------------------------------------------------------------
# Records and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationPhaseRecord(Generic[EvaluationT]):
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
            "phase_index": int(self.phase_index),
            "phase_best_mask": to_json_compatible(self.phase_best_mask),
            "phase_best_selected_feature_count": int(self.phase_best_selected_feature_count),
            "phase_best_evaluation": to_json_compatible(self.phase_best_evaluation),
            "run_best_mask": to_json_compatible(self.run_best_mask),
            "run_best_selected_feature_count": int(self.run_best_selected_feature_count),
            "run_best_evaluation": to_json_compatible(self.run_best_evaluation),
            "run_best_improved": bool(self.run_best_improved),
            "request_count": int(self.request_count),
            "cumulative_requests": int(self.cumulative_requests),
            "cumulative_unique_evaluations": int(self.cumulative_unique_evaluations),
            "cumulative_cache_hits": int(self.cumulative_cache_hits),
            "population_diversity": float(self.population_diversity),
            "repair_count": int(self.repair_count),
            "kinetics_value": self.kinetics_value,
        }


@dataclass(frozen=True)
class AblationArmResult(Generic[EvaluationT]):
    """Complete reproducible result of ONE ablation arm."""

    variant: str
    optimizer_seed: int
    dimensions: int
    population_size: int
    bpso_evaluated_generations: int
    bgwo_evaluated_iterations: int
    elite_count: int
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
    elite_selection_invoked: bool
    placed_elite_count: int
    elite_masks: np.ndarray
    elite_hashes: tuple[str, ...]
    filler_block: ShadowFillerBlock | None
    shadow_filler_generated: bool
    shadow_filler_evaluated: bool
    shadow_filler_placed: bool
    bpso_submitted_outputs_sha256: str
    bpso_final_population_sha256: str
    bgwo_initial_population: np.ndarray
    bgwo_common_rng_state_before_updates: str
    bgwo_row0_sha256: str
    bgwo_rows_1_3_sha256: tuple[str, ...]
    bgwo_rows_4_11_sha256: tuple[str, ...]
    convergence_history: tuple[AblationPhaseRecord[EvaluationT], ...]
    repair_count: int
    best_feasible: bool | None
    best_normalized_violation: float | None
    configuration_snapshot: Mapping[str, Any]

    @property
    def final_population(self) -> np.ndarray:
        """The full composed BGWO initial population (row0..rows4-11), read-only."""
        return self.bgwo_initial_population.copy()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "variant": self.variant,
            "optimizer_seed": int(self.optimizer_seed),
            "dimensions": int(self.dimensions),
            "population_size": int(self.population_size),
            "bpso_evaluated_generations": int(self.bpso_evaluated_generations),
            "bgwo_evaluated_iterations": int(self.bgwo_evaluated_iterations),
            "elite_count": int(self.elite_count),
            "stop_reason": self.stop_reason,
            "best_mask": to_json_compatible(self.best_mask),
            "best_selected_feature_count": int(self.best_selected_feature_count),
            "best_evaluation": to_json_compatible(self.best_evaluation),
            "best_phase": self.best_phase,
            "best_phase_index": int(self.best_phase_index),
            "accounting": {
                "total_candidate_requests": int(self.total_candidate_requests),
                "unique_evaluations": int(self.unique_evaluations),
                "cache_hits": int(self.cache_hits),
                "evaluator_calls": int(self.evaluator_calls),
                "bpso_requests": int(self.bpso_requests),
                "bgwo_requests": int(self.bgwo_requests),
                "bpso_unique_evaluations": int(self.bpso_unique_evaluations),
                "bgwo_new_unique_evaluations": int(self.bgwo_new_unique_evaluations),
                "bpso_cache_hits": int(self.bpso_cache_hits),
                "bgwo_cache_hits": int(self.bgwo_cache_hits),
            },
            "elite_transfer": {
                "elite_selection_invoked": bool(self.elite_selection_invoked),
                "placed_elite_count": int(self.placed_elite_count),
                "elite_masks": to_json_compatible(self.elite_masks),
                "elite_hashes": list(self.elite_hashes),
            },
            "shadow_filler": {
                "generated": bool(self.shadow_filler_generated),
                "evaluated": bool(self.shadow_filler_evaluated),
                "placed": bool(self.shadow_filler_placed),
                "block": None if self.filler_block is None else self.filler_block.to_dict(),
                "duplicate_rejection": False,
            },
            "paired_identity": {
                "bpso_submitted_outputs_sha256": self.bpso_submitted_outputs_sha256,
                "bpso_final_population_sha256": self.bpso_final_population_sha256,
                "bgwo_common_rng_state_before_updates": self.bgwo_common_rng_state_before_updates,
                "bgwo_row0_sha256": self.bgwo_row0_sha256,
                "bgwo_rows_1_3_sha256": list(self.bgwo_rows_1_3_sha256),
                "bgwo_rows_4_11_sha256": list(self.bgwo_rows_4_11_sha256),
            },
            "bgwo_initial_population": to_json_compatible(self.bgwo_initial_population),
            "convergence_history": [item.to_dict() for item in self.convergence_history],
            "repair_count": int(self.repair_count),
            "best_feasible": self.best_feasible,
            "best_normalized_violation": self.best_normalized_violation,
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

    def deterministic_digest(self) -> str:
        """Stable SHA-256 over the canonical serialization of this arm."""
        fields = {
            "variant": self.variant,
            "optimizer_seed": int(self.optimizer_seed),
            "elite_selection_invoked": bool(self.elite_selection_invoked),
            "placed_elite_count": int(self.placed_elite_count),
            "elite_hashes": list(self.elite_hashes),
            "bpso_submitted_outputs_sha256": self.bpso_submitted_outputs_sha256,
            "bpso_final_population_sha256": self.bpso_final_population_sha256,
            "bgwo_row0_sha256": self.bgwo_row0_sha256,
            "bgwo_rows_1_3_sha256": list(self.bgwo_rows_1_3_sha256),
            "bgwo_rows_4_11_sha256": list(self.bgwo_rows_4_11_sha256),
            "bgwo_common_rng_state_before_updates": self.bgwo_common_rng_state_before_updates,
            "best_mask": to_json_compatible(self.best_mask),
            "best_selected_feature_count": int(self.best_selected_feature_count),
            "best_evaluation": to_json_compatible(self.best_evaluation),
            "best_phase": self.best_phase,
            "best_phase_index": int(self.best_phase_index),
            "accounting": self.to_dict()["accounting"],
            "repair_count": int(self.repair_count),
            "shadow_filler_evaluated": bool(self.shadow_filler_evaluated),
            "shadow_filler_placed": bool(self.shadow_filler_placed),
            "filler_hashes": (
                None if self.filler_block is None else list(self.filler_block.mask_sha256)
            ),
        }
        raw = json.dumps(
            to_json_compatible(fields),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PairedAblationResult(Generic[EvaluationT]):
    """One paired optimizer-seed run with WITH and WITHOUT arms plus evidence."""

    optimizer_seed: int
    filler_block: ShadowFillerBlock
    with_arm: AblationArmResult[EvaluationT]
    without_arm: AblationArmResult[EvaluationT]
    invariants: Mapping[str, Any]
    status: str

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_kind": "V10G_ABLATION_PAIRED_RUN",
            "schema_version": "v1.0-g-ablation-paired-run-2",
            "stage": "V1.0-G2",
            "variant_definitions": ["WITH_ELITE_TRANSFER", "WITHOUT_ELITE_TRANSFER"],
            "optimizer_seed": int(self.optimizer_seed),
            "filler_block": self.filler_block.to_dict(),
            "with_arm": self.with_arm.to_dict(),
            "without_arm": self.without_arm.to_dict(),
            "invariants": dict(self.invariants),
            "status": self.status,
            "test_accessed": False,
            "test_used_for_fitness": False,
            "test_used_for_selection": False,
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

    def deterministic_digest(self) -> str:
        raw = self.to_json(indent=None).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Frozen ranking helpers (mirror the frozen hybrid ranking: strict comparator
# plus canonical lexicographic mask tie-break; aggregate scores prohibited).
# ---------------------------------------------------------------------------


def evaluation_relation(
    is_better: Comparator[EvaluationT],
    left: EvaluationT,
    right: EvaluationT,
) -> int:
    left_better = bool(is_better(left, right))
    right_better = bool(is_better(right, left))
    if left_better and right_better:
        raise ComparatorContractError(
            "The injected comparator must be strict and asymmetric."
        )
    if left_better:
        return -1
    if right_better:
        return 1
    return 0


def rank_indices(
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
            relation = evaluation_relation(
                is_better, evaluations[idx_right], evaluations[idx_best]
            )
            if relation < 0 or (
                relation == 0
                and _mask_tuple(population[idx_right]) < _mask_tuple(population[idx_best])
            ):
                best = right
        ordered[left], ordered[best] = ordered[best], ordered[left]
    return tuple(ordered)


def best_index(
    population: np.ndarray,
    evaluations: Sequence[EvaluationT],
    is_better: Comparator[EvaluationT],
) -> int:
    best = 0
    for index in range(1, len(evaluations)):
        relation = evaluation_relation(is_better, evaluations[index], evaluations[best])
        if relation < 0 or (
            relation == 0 and _mask_tuple(population[index]) < _mask_tuple(population[best])
        ):
            best = index
    return best


# ---------------------------------------------------------------------------
# Paired arm execution
# ---------------------------------------------------------------------------


def run_ablation_arm(
    variant: AblationVariant | str,
    optimizer_seed: int,
    config: AblationConfig,
    objective: Objective[EvaluationT],
    is_better: Comparator[EvaluationT],
    filler_block: ShadowFillerBlock | None = None,
) -> AblationArmResult[EvaluationT]:
    """Execute one deterministic fixed-budget ablation arm.

    ``WITH_ELITE_TRANSFER`` transfers the top-three distinct BPSO elites into
    BGWO rows 1-3 (optionally recording a shadow filler block that is never
    placed or evaluated). ``WITHOUT_ELITE_TRANSFER`` places the shared shadow
    filler block in rows 1-3 and never invokes elite selection.
    """
    variant_enum = AblationVariant(variant)
    variant_name = variant_enum.value
    if variant_name not in (AblationVariant.WITH_ELITE_TRANSFER, AblationVariant.WITHOUT_ELITE_TRANSFER):
        raise AblationConfigError(f"Unknown ablation variant: {variant_name}")
    if not callable(objective) or not callable(is_better):
        raise TypeError("objective and is_better must be callable.")
    if config.filler_row_count != config.elite_count:
        raise AblationConfigError("Filler row count must equal the elite count.")

    if filler_block is None:
        filler_block = generate_shadow_filler_block(
            int(optimizer_seed),
            dimensions=config.dimensions,
            cardinality_pool=config.cardinality_pool,
            filler_protocol_tag=config.filler_protocol_tag,
            filler_stream_tag=config.filler_stream_tag,
            row_count=config.filler_row_count,
        )
    if int(filler_block.dimensions) != int(config.dimensions):
        raise AblationConfigError("Filler block dimensions must match the configuration.")
    if int(filler_block.row_count) != int(config.filler_row_count):
        raise AblationConfigError("Filler block row count must match the configuration.")

    bpso_generator = np.random.Generator(
        np.random.PCG64(int(optimizer_seed) + int(config.bpso_phase_rng_offset))
    )
    cache: RunLocalCache[EvaluationT] = RunLocalCache(objective, config.cache_enabled)

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
    history: list[AblationPhaseRecord[EvaluationT]] = []
    bpso_repairs = 0
    generation_repairs = 0
    applied_inertia: float | None = None
    phase_requests_bound = config.population_size
    submitted_bpso_masks: list[np.ndarray] = []

    for generation in range(config.bpso_evaluated_generations):
        evaluations = [cache.request(mask) for mask in positions]
        submitted_bpso_masks.extend(mask.copy() for mask in positions)
        if generation == 0:
            personal_best_masks = positions.copy()
            personal_best_evaluations = list(evaluations)
        else:
            assert personal_best_masks is not None
            for index, evaluation in enumerate(evaluations):
                relation = evaluation_relation(
                    is_better, evaluation, personal_best_evaluations[index]
                )
                if relation < 0 or (
                    relation == 0
                    and _mask_tuple(positions[index]) < _mask_tuple(personal_best_masks[index])
                ):
                    personal_best_masks[index] = positions[index].copy()
                    personal_best_evaluations[index] = evaluation

        phase_best_index = best_index(positions, evaluations, is_better)
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
            relation = evaluation_relation(is_better, candidate_evaluation, run_best_evaluation)
            if relation < 0 or (
                relation == 0 and _mask_tuple(candidate) < _mask_tuple(run_best_mask)
            ):
                run_best_mask = candidate.copy()
                run_best_evaluation = candidate_evaluation
                run_best_phase = "BPSO"
                run_best_phase_index = generation
                considered = "BPSO"

        history.append(
            AblationPhaseRecord(
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
    bpso_final_population = positions.copy()

    bpso_submitted_digest = hashlib.sha256(
        b"".join(mask.tobytes() for mask in submitted_bpso_masks)
    ).hexdigest()
    bpso_final_digest = hashlib.sha256(bpso_final_population.tobytes()).hexdigest()

    # ------------------------------------------------------------------
    # Elite selection (WITH only, zero evaluator calls)
    # ------------------------------------------------------------------
    elite_selection_invoked = variant_name == AblationVariant.WITH_ELITE_TRANSFER
    if elite_selection_invoked:
        elite_masks = select_elites(cache, config.elite_count, is_better)
        placed_elites = (
            int(min(elite_masks.shape[0], config.elite_count)) if elite_masks.ndim == 2 else 0
        )
    else:
        elite_masks = np.zeros((0, config.dimensions), dtype=np.uint8)
        placed_elites = 0
    elite_hashes = tuple(
        hashlib.sha256(elite_masks[index].tobytes()).hexdigest()
        for index in range(elite_masks.shape[0])
    )

    # ------------------------------------------------------------------
    # BGWO initialization (rows 0/1-3/4-11 built so common RNG draws match)
    # ------------------------------------------------------------------
    bgwo_generator = np.random.Generator(
        np.random.PCG64(int(optimizer_seed) + int(config.bgwo_phase_rng_offset))
    )
    population = np.zeros((config.population_size, config.dimensions), dtype=np.uint8)
    population[ROW_ANCHOR_INDEX, :] = 1
    slot = 1
    if elite_selection_invoked:
        for index in range(placed_elites):
            population[slot] = elite_masks[index]
            slot += 1
    else:
        for index in range(config.filler_row_count):
            population[slot] = filler_block.masks[index]
            slot += 1
    while slot < config.population_size:
        cardinality = int(
            config.cardinality_pool[bgwo_generator.integers(len(config.cardinality_pool))]
        )
        population[slot] = exact_cardinality_mask(config.dimensions, cardinality, bgwo_generator)
        slot += 1
    bgwo_initial_population_snapshot = population.copy()
    bgwo_common_rng_state_before_updates = _rng_state_json(bgwo_generator)

    row0 = population[ROW_ANCHOR_INDEX].copy()
    rows_1_3 = population[ROW_TREATMENT_SLICE].copy()
    rows_4_11 = population[ROW_COMMON_SLICE].copy()

    # ------------------------------------------------------------------
    # BGWO refinement phase (fixed budget, no early stopping)
    # ------------------------------------------------------------------
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
        ordered = rank_indices(population, evaluations, is_better)
        alpha_index, beta_index, delta_index = ordered[:3]
        candidate = population[alpha_index]
        candidate_evaluation = evaluations[alpha_index]
        considered = "unknown"
        assert run_best_mask is not None and run_best_evaluation is not None
        relation = evaluation_relation(is_better, candidate_evaluation, run_best_evaluation)
        if relation < 0 or (
            relation == 0 and _mask_tuple(candidate) < _mask_tuple(run_best_mask)
        ):
            run_best_mask = candidate.copy()
            run_best_evaluation = candidate_evaluation
            run_best_phase = "BGWO"
            run_best_phase_index = iteration
            considered = "BGWO"

        history.append(
            AblationPhaseRecord(
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
    # Invariant enforcement (fail closed on every mandatory invariant)
    # ------------------------------------------------------------------
    if cache.total_requests != cache.unique_evaluations + cache.cache_hits:
        raise AblationError("Ablation cache accounting invariant failed.")
    if cache.evaluator_calls != cache.unique_evaluations:
        raise AblationError("Ablation evaluator-call invariant failed.")
    if cache.total_requests != config.per_run_request_allocation:
        raise AblationError("An ablation arm must consume its complete fixed request budget.")
    if bpso_requests_snapshot != config.bpso_request_allocation:
        raise AblationError("BPSO phase must consume its complete fixed allocation.")
    if bgwo_total_requests != config.bgwo_request_allocation:
        raise AblationError("BGWO phase must consume its complete fixed allocation.")
    if elite_selection_invoked and bgwo_hits < placed_elites:
        raise AblationError(
            "Transferred elite candidates must resolve as shared-cache hits (WITH)."
        )
    if not elite_selection_invoked and placed_elites != 0:
        raise AblationError("The WITHOUT arm must never place elites.")
    if np.array_equal(population[ROW_ANCHOR_INDEX], 0):
        raise AblationError("BGWO row 0 must remain a nonzero anchor.")
    assert run_best_mask is not None and run_best_evaluation is not None

    best_feasible = getattr(run_best_evaluation, "feasible", None)
    best_violation = getattr(run_best_evaluation, "normalized_violation", None)
    best_violation = float(best_violation) if best_violation is not None else None

    snapshot = config.to_dict()
    snapshot["optimizer_seed"] = int(optimizer_seed)
    snapshot["variant"] = variant_name
    snapshot["placed_elite_count"] = placed_elites
    snapshot["shadow_filler_evaluated"] = not elite_selection_invoked

    shadow_evaluated = not elite_selection_invoked
    result = AblationArmResult(
        variant=variant_name,
        optimizer_seed=int(optimizer_seed),
        dimensions=config.dimensions,
        population_size=config.population_size,
        bpso_evaluated_generations=config.bpso_evaluated_generations,
        bgwo_evaluated_iterations=config.bgwo_evaluated_iterations,
        elite_count=config.elite_count,
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
        elite_selection_invoked=elite_selection_invoked,
        placed_elite_count=placed_elites,
        elite_masks=elite_masks.copy(),
        elite_hashes=elite_hashes,
        filler_block=filler_block,
        shadow_filler_generated=True,
        shadow_filler_evaluated=shadow_evaluated,
        shadow_filler_placed=not elite_selection_invoked,
        bpso_submitted_outputs_sha256=bpso_submitted_digest,
        bpso_final_population_sha256=bpso_final_digest,
        bgwo_initial_population=bgwo_initial_population_snapshot,
        bgwo_common_rng_state_before_updates=bgwo_common_rng_state_before_updates,
        bgwo_row0_sha256=hashlib.sha256(row0.tobytes()).hexdigest(),
        bgwo_rows_1_3_sha256=tuple(
            hashlib.sha256(rows_1_3[index].tobytes()).hexdigest()
            for index in range(rows_1_3.shape[0])
        ),
        bgwo_rows_4_11_sha256=tuple(
            hashlib.sha256(rows_4_11[index].tobytes()).hexdigest()
            for index in range(rows_4_11.shape[0])
        ),
        convergence_history=tuple(history),
        repair_count=bpso_repairs + bgwo_repairs,
        best_feasible=best_feasible,
        best_normalized_violation=best_violation,
        configuration_snapshot=snapshot,
    )
    return result


def _rng_state_json(generator: np.random.Generator) -> str:
    if not isinstance(generator, np.random.Generator) or not isinstance(
        generator.bit_generator, np.random.PCG64
    ):
        raise TypeError("Paired randomness requires numpy.random.Generator with PCG64.")
    return json.dumps(generator.bit_generator.state, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Paired run + fail-closed invariant verification
# ---------------------------------------------------------------------------


def verify_paired_invariants(paired: PairedAblationResult[EvaluationT]) -> dict[str, Any]:
    """Verify every locked paired invariant and fail closed on violation."""
    with_arm = paired.with_arm
    without_arm = paired.without_arm

    paired_bpso_identical = (
        with_arm.bpso_submitted_outputs_sha256 == without_arm.bpso_submitted_outputs_sha256
        and with_arm.bpso_final_population_sha256 == without_arm.bpso_final_population_sha256
    )
    row0_identical = with_arm.bgwo_row0_sha256 == without_arm.bgwo_row0_sha256
    rows_4_11_identical = (
        with_arm.bgwo_rows_4_11_sha256 == without_arm.bgwo_rows_4_11_sha256
    )
    bgwo_rng_identical = (
        with_arm.bgwo_common_rng_state_before_updates
        == without_arm.bgwo_common_rng_state_before_updates
    )
    with_rows_are_elites = (
        with_arm.placed_elite_count == len(with_arm.elite_hashes)
        and list(with_arm.elite_hashes) == list(with_arm.bgwo_rows_1_3_sha256)
    )
    without_rows_are_filler = _rows_match_filler(without_arm)
    with_filler_not_placed = not with_arm.shadow_filler_placed and not with_arm.shadow_filler_evaluated
    without_invokes_no_elites = (
        not without_arm.elite_selection_invoked and without_arm.placed_elite_count == 0
    )
    same_filler_block = (
        with_arm.filler_block is not None
        and without_arm.filler_block is not None
        and with_arm.filler_block.mask_sha256 == without_arm.filler_block.mask_sha256
        and with_arm.filler_block.to_dict() == without_arm.filler_block.to_dict()
    )
    shared_budget = (
        with_arm.bpso_requests == without_arm.bpso_requests
        and with_arm.bgwo_requests == without_arm.bgwo_requests
    )

    row_treatment_collisions = [
        left == right
        for left, right in zip(with_arm.bgwo_rows_1_3_sha256, without_arm.bgwo_rows_1_3_sha256)
    ]

    checks = {
        "paired_bpso_outputs_byte_identical": bool(paired_bpso_identical),
        "bgwo_row0_byte_identical": bool(row0_identical),
        "bgwo_rows_4_11_byte_identical": bool(rows_4_11_identical),
        "bgwo_common_rng_state_identical_before_updates": bool(bgwo_rng_identical),
        "with_rows_1_3_are_exactly_the_elites": bool(with_rows_are_elites),
        "without_rows_1_3_are_exactly_the_filler": bool(without_rows_are_filler),
        "with_shadow_filler_neither_placed_nor_evaluated": bool(with_filler_not_placed),
        "without_never_invokes_elite_selection": bool(without_invokes_no_elites),
        "shared_shadow_filler_block_identical": bool(same_filler_block),
        "shared_phase_request_budgets": bool(shared_budget),
        "treatment_rows_collision_reported_not_rejected": list(row_treatment_collisions),
    }
    mandatory = (
        "paired_bpso_outputs_byte_identical",
        "bgwo_row0_byte_identical",
        "bgwo_rows_4_11_byte_identical",
        "bgwo_common_rng_state_identical_before_updates",
        "with_rows_1_3_are_exactly_the_elites",
        "without_rows_1_3_are_exactly_the_filler",
        "with_shadow_filler_neither_placed_nor_evaluated",
        "without_never_invokes_elite_selection",
        "shared_shadow_filler_block_identical",
        "shared_phase_request_budgets",
    )
    failed = [name for name in mandatory if checks[name] is not True]
    if failed:
        raise PairedInvariantViolation(
            f"Paired invariant(s) failed; fail closed: {', '.join(failed)}"
        )
    return {
        "status": "PASS",
        "checks": checks,
        "treatment_rows_differ_bitwise": [not value for value in row_treatment_collisions],
    }


def _rows_match_filler(arm: AblationArmResult[EvaluationT]) -> bool:
    block = arm.filler_block
    if block is None:
        return False
    return list(block.mask_sha256) == list(arm.bgwo_rows_1_3_sha256)


def run_paired_seed(
    optimizer_seed: int,
    config: AblationConfig,
    objective: Objective[EvaluationT],
    is_better: Comparator[EvaluationT],
) -> PairedAblationResult[EvaluationT]:
    """Run the locked paired WITH/WITHOUT ablation for one optimizer seed.

    The same shadow filler block is generated exactly once and supplied to both
    arms. Mandatory paired invariants are verified and the run fails closed on
    any violation. This function performs NO filesystem I/O and never loads a
    real dataset or TEST split.
    """
    filler_block = generate_shadow_filler_block(
        int(optimizer_seed),
        dimensions=config.dimensions,
        cardinality_pool=config.cardinality_pool,
        filler_protocol_tag=config.filler_protocol_tag,
        filler_stream_tag=config.filler_stream_tag,
        row_count=config.filler_row_count,
    )
    with_arm = run_ablation_arm(
        AblationVariant.WITH_ELITE_TRANSFER,
        int(optimizer_seed),
        config,
        objective,
        is_better,
        filler_block=filler_block,
    )
    without_arm = run_ablation_arm(
        AblationVariant.WITHOUT_ELITE_TRANSFER,
        int(optimizer_seed),
        config,
        objective,
        is_better,
        filler_block=filler_block,
    )
    invariants = verify_paired_invariants(
        PairedAblationResult(
            optimizer_seed=int(optimizer_seed),
            filler_block=filler_block,
            with_arm=with_arm,
            without_arm=without_arm,
            invariants={},
            status="PENDING",
        )
    )
    return PairedAblationResult(
        optimizer_seed=int(optimizer_seed),
        filler_block=filler_block,
        with_arm=with_arm,
        without_arm=without_arm,
        invariants=invariants,
        status="PASS",
    )


def _mask_tuple(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in mask)


__all__ = [
    "ABLATION_IDENTIFIER",
    "AblationArmResult",
    "AblationConfig",
    "AblationConfigError",
    "AblationError",
    "AblationNoGoError",
    "AblationPhaseRecord",
    "AblationVariant",
    "BPSO_PHASE_RNG_OFFSET",
    "BGWO_PHASE_RNG_OFFSET",
    "CARDINALITY_POOL",
    "ComparatorContractError",
    "DESIGN_IDENTIFIER",
    "EXPECTED_PRODUCTION_CONSTANTS",
    "FILLER_PROTOCOL_TAG",
    "FILLER_ROW_COUNT",
    "FILLER_STREAM_TAG",
    "FROZEN_MECHANICS",
    "PairedAblationResult",
    "PairedInvariantViolation",
    "PRODUCTION_BGWO_ITERATIONS",
    "PRODUCTION_BPSO_GENERATIONS",
    "PRODUCTION_DIMENSIONS",
    "PRODUCTION_ELITE_COUNT",
    "PRODUCTION_FITS_PER_UNIQUE_EVALUATION",
    "PRODUCTION_MODEL_ATTACK_SEEDS",
    "PRODUCTION_OPTIMIZER_SEEDS",
    "PRODUCTION_POPULATION",
    "PRODUCTION_REQUESTS_ALL_ABLATION",
    "PRODUCTION_REQUESTS_PER_ARM",
    "PRODUCTION_REQUESTS_PER_BGWO",
    "PRODUCTION_REQUESTS_PER_BPSO",
    "PRODUCTION_REQUESTS_PER_RUN",
    "ROW_ANCHOR_INDEX",
    "ROW_COMMON_SLICE",
    "ROW_TREATMENT_SLICE",
    "STOP_FIXED_BUDGET_EXHAUSTED",
    "ShadowFillerBlock",
    "best_index",
    "evaluation_relation",
    "generate_shadow_filler_block",
    "production_ablation_config",
    "rank_indices",
    "run_ablation_arm",
    "run_paired_seed",
    "validate_production_constants",
]