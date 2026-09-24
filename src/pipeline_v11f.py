"""V1.1-F2 read-only resource-efficiency / green-evaluation analysis pipeline.

This module is the sanctioned *read-only analytical* implementation of the
frozen V1.1-F2 stage. It consumes ONLY persisted, fingerprint-verified
V1.1-E evidence (E06/E07/E08/E09/E10 cell documents, aggregated descriptive
summaries, workload/attack manifests, the V1.1-E result lock) together with
the frozen V1.1-F1 green-evaluation protocol lock/config/document.

Frozen implementation-stage boundaries (from ``config/resource_efficiency_v11f.yaml``):

* ``DATA_ACCESS = READ_ONLY_PERSISTED_ARTIFACTS_ONLY`` -- the pipeline never
  opens raw DataCo, never opens TEST splits/features/labels, never constructs
  a blockchain campaign, and never re-runs E01-E10.
* ``TEST_ACCESS = FORBIDDEN``, ``AI_FIT_ALLOWED = FALSE``,
  ``AI_INFERENCE_ALLOWED = FALSE`` -- fail-closed guards exist for each and
  the governance record asserts a zero request count.
* ``DIRECT_ENERGY_UNAVAILABLE`` -- CPU/wall/tracemalloc/throughput/storage
  evidence is never transformed into Joules/Wh/kWh/Watts/TDP-derived
  energy/carbon/CO2e, and no savings/reduction claims are emitted.
* ``E08 = POLICY_INDEPENDENT`` (storage differences are never manufactured),
  ``E10 = DERIVED_ONLY`` (sources are exactly E06/E07/E08/E09), workloads are
  nested (``100 subset 250 subset 500 subset 1000 subset 2500`` within seed;
  never independent replicates), and no overall winner/ranking/global score is
  ever produced.

Everything here is deterministic and read-only. ``run_v11f_analysis`` returns
an in-memory structured payload (including a canonical
``semantic_analysis_sha256``) suitable for later, separately governed F4
persistence; it writes no production artifacts and is not the V1.1-F result
lock (that is reserved for a later governed stage).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

STAGE = "V1.1-F2"
KIND = "V11F2_RESOURCE_EFFICIENCY_READONLY_ANALYSIS"
MARKER = "V11F2_ANALYSIS_PAYLOAD"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V11E_DIR = PROJECT_ROOT / "results" / "blockchain" / "v11e"
V11F_DIR = PROJECT_ROOT / "results" / "blockchain" / "v11f"

F1_LOCK_PATH = V11F_DIR / "v11f_protocol_lock.json"
F1_CONFIG_PATH = PROJECT_ROOT / "config" / "resource_efficiency_v11f.yaml"
F1_PROTOCOL_PATH = PROJECT_ROOT / "docs" / "v11f_green_evaluation_protocol.md"

V11E_LOCK_NAME = "v11e_experiment_result_lock.json"
V11E_STAGE_DIR_NAME = "v11e_experiment_results"
V11E_STAGE_NAME = "V1.1-E4"
V11E_RAW_DIR_NAME = "v11e_raw_run_observations"
V11E_SUMMARY_DIR_NAME = "v11e_aggregated_descriptive_summaries"
V11E_ATTACK_MANIFEST_NAME = "v11e_attack_instance_manifest.json"
V11E_WORKLOAD_MANIFEST_NAME = "v11e_workload_manifest.json"

# --------------------------------------------------------------------------- #
# frozen expected fingerprints (V1.1-F1 contract + V1.1-E upstream evidence)
# --------------------------------------------------------------------------- #

EXPECTED_F1_LOCK_SEMANTIC = (
    "68cceedde6384c16a23226ddf082ef7d478e489c9b691e4e63d30bade85597e1"
)
EXPECTED_F1_CONFIG_SHA256 = (
    "84f198556ce319a2775063e11960cd8cd5e5457b8fea14ac835b6f93040c71f3"
)
EXPECTED_F1_PROTOCOL_DOC_SHA256 = (
    "8e20ac45291fe5568b5d1e10a3671faec71e3db4bb191c8a299ca446d68eaaa1"
)
EXPECTED_V11E_RESULT_LOCK_FILE_SHA256 = (
    "dd3b926eb08206f505e739ed61c5f29a3e7f03a370e9afc1a077e0b260fd6e64"
)
EXPECTED_V11E_RESULT_LOCK_SEMANTIC = (
    "058aeca8ac97101356bcf1c4dc5b74fb3affbda6833a85b5d423a53556cd1749"
)
EXPECTED_V11E_PROTOCOL_LOCK_SEMANTIC = (
    "8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239"
)
EXPECTED_V11E_AUTHORIZED_EXECUTION_COMMIT = (
    "e206ef7a32fef4dc8517cb7c592ea52713b26f2f"
)

POLICIES = ("B0", "B1", "P")
ENERGY_MARKER = "DIRECT_ENERGY_UNAVAILABLE"

RESOURCE_EXPERIMENTS = ("E06", "E07", "E08", "E09", "E10")
POLICY_INDEPENDENT_EXPERIMENTS = ("E08",)
E10_SOURCES = ("E06", "E07", "E08", "E09")

MATCHED_CONTRASTS = (
    ("B1", "B0", "B1_minus_B0"),
    ("P", "B0", "P_minus_B0"),
    ("P", "B1", "P_minus_B1"),
)
NORMALIZED_RATIO_PAIRS = (
    ("P", "B0", "P_over_B0"),
    ("P", "B1", "P_over_B1"),
)

ALLOWED_SUMMARIES = (
    "count",
    "mean",
    "median",
    "sample_standard_deviation_ddof_1",
    "min",
    "max",
    "coefficient_of_variation",
)
CV_FORMULA = "sample_standard_deviation_ddof_1 / abs(arithmetic_mean)"
CV_PRECONDITION = "arithmetic_mean != 0"


class V11FError(Exception):
    """Base error for the V1.1-F2 read-only analysis layer."""


class V11FIntegrityError(V11FError):
    """A persisted artifact drifted from its expected frozen value."""


class V11FProtocolLockMismatchError(V11FIntegrityError):
    """V1.1-F1 protocol lock did not match the governed expected semantic."""


class V11FConfigMismatchError(V11FIntegrityError):
    """V1.1-F1 protocol config did not match the governed expected sha256."""


class V11FResultLockMismatchError(V11FIntegrityError):
    """V1.1-E result lock did not match the governed expected fingerprints."""


class V11FFingerprintMismatchError(V11FIntegrityError):
    """A governed V1.1-E artifact fingerprint drifted or is missing."""


class V11FMissingArtifactError(V11FError):
    """A required persisted evidence artifact is absent from disk."""


class V11FPolicyError(V11FError):
    """An unexpected policy (or policy-set drift) was observed."""


class V11FUnmatchedPairError(V11FError):
    """A matched comparison failed closed (missing/duplicate/mismatched pair)."""


class V11FE10NotDerivedError(V11FIntegrityError):
    """An E10 cell is not marked derived-only."""


class V11FE10SourceError(V11FIntegrityError):
    """An E10 cell cites an unexpected/misbound derivation source."""


class V11FTestAccessError(V11FError):
    """A TEST-access attempt was made (reserved; never happens here)."""


class V11FAIAccessError(V11FError):
    """An AI fit/inference/model-load attempt was made (reserved)."""


class V11FMeasurementError(V11FError):
    """A new-measurement/rerun attempt was made (reserved)."""


class V11FEnergyClaimError(V11FError):
    """Prohibited direct-energy terminology, estimate, or claim surfaced."""


class V11FRankingError(V11FError):
    """A prohibited overall-winner/ranking/global-score field appeared."""


# --------------------------------------------------------------------------- #
# canonical serialization + read-only IO
# --------------------------------------------------------------------------- #


def canonical_json(value: Any) -> str:
    """Repository-canonical JSON: sorted keys, compact separators, ascii.

    Matches the convention used by the V1.1-E4L launcher and the V1.1-F1
    protocol lock (no NaN/Infinity, no trailing newline).
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_of_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path | str, description: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes().decode("utf-8")
    except FileNotFoundError as exc:
        raise V11FMissingArtifactError(
            f"{description} artifact is absent: {Path(path).name}"
        ) from exc
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise V11FIntegrityError(f"{description} is not a JSON object.")
    return document


def load_config(path: Path | str) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise V11FConfigMismatchError("V1.1-F1 protocol config is not a mapping.")
    return config


# --------------------------------------------------------------------------- #
# deterministic cell-plan helpers (mirror the frozen V1.1-E4 matrix)
# --------------------------------------------------------------------------- #


def cell_stem(experiment_id: str, seed: int, workload: int, policy: str) -> str:
    return f"{experiment_id}_s{int(seed)}_w{int(workload):05d}_p{policy}"


def cell_key(experiment_id: str, seed: int, workload: int, policy: str) -> str:
    return f"{experiment_id}|s{int(seed)}|w{int(workload)}|{policy}"


def parse_cell_key(key: str) -> tuple[str, int, int, str]:
    match = re.fullmatch(r"(E\d{2})\|s(\d+)\|w(\d+)\|([A-Z0-9_]+)", key)
    if match is None:
        raise V11FIntegrityError(f"malformed cell key {key!r}")
    return match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)


def _environment_signature(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a persisted cell's machine/measurement environment identity.

    Used by _verify_environment_identity to enforce the F1 matched-pairing
    condition that all cells participating in a paired comparison of one metric
    share the same machine_environment / measurement_boundary (present in every
    frozen cell's ``environment`` block). The signature is raise-only: it never
    alters scientific output, only fails closed on environment drift.
    """
    environment = doc.get("environment")
    if not isinstance(environment, Mapping):
        context = (doc.get("result") or {}).get("context") or {}
        environment = context.get("environment") or {}
    signature = {str(k): v for k, v in environment.items()}
    return {"_signature_sha256": sha256_of_canonical(signature), **signature}


def _verify_environment_identity(cells: Mapping[str, Mapping[tuple[int, int], Mapping[str, Mapping[str, Any]]]]) -> None:
    """Fail closed if cells of one metric disagree on environment identity.

    Defense-in-depth for the F1 pairing condition: identity of a paired
    comparison is (seed, workload) plus, when present, experiment_id,
    measurement_boundary and machine_environment. Frozen evidence is constant
    per experiment; drift (e.g. a future rerun on different hardware) must
    not be silently pooled.
    """
    for experiment, by_position in sorted(cells.items()):
        signatures: set[str] = set()
        for by_policy in by_position.values():
            for doc in by_policy.values():
                signatures.add(str(_environment_signature(doc)["_signature_sha256"]))
        if len(signatures) > 1:
            raise V11FUnmatchedPairError(
                f"{experiment} cells disagree on the machine/measurement "
                "environment identity required for matched pairing."
            )


# --------------------------------------------------------------------------- #
# frozen statistical primitives (bit-identical to V1.1-E3/E4 conventions)
# --------------------------------------------------------------------------- #


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def sample_std_ddof1(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _min_max(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return min(values), max(values)


def describe_values(values: Sequence[float]) -> dict[str, Any]:
    """Frozen descriptive summary: count/mean/median/std(ddof=1)/min/max/CV.

    ``coefficient_of_variation`` follows the F1 contract formula
    ``sample_standard_deviation_ddof_1 / abs(arithmetic_mean)`` with the
    precondition ``arithmetic_mean != 0``; when the mean is zero (or the
    sample std is unavailable) the CV is ``None`` because it is undefined.
    """
    lo, hi = _min_max(values)
    av = mean(values)
    sd = sample_std_ddof1(values)
    coefficient_of_variation = None
    if sd is not None and av is not None and av != 0.0:
        coefficient_of_variation = sd / abs(av)
    return {
        "count": len(values),
        "mean": av,
        "median": median(values),
        "sample_standard_deviation_ddof_1": sd,
        "min": lo,
        "max": hi,
        "coefficient_of_variation": coefficient_of_variation,
    }


# --------------------------------------------------------------------------- #
# governance guards (fail-closed; the pipeline never invokes the guarded
# activities, so these exist to make the reservation explicit and testable)
# --------------------------------------------------------------------------- #

_FORBIDDEN_TEST_PATH_TOKENS = (
    "test_split",
    "test_features",
    "test_labels",
    "raw_test",
    "test/",
    "trial",
)


def guard_test_access(*, request: str | None = None) -> None:
    if request is not None:
        raise V11FTestAccessError(
            f"TEST access is FORBIDDEN by the V1.1-F1 contract (requested: {request!r})."
        )


def guard_ai_fit(*, requested: bool = False) -> None:
    if requested:
        raise V11FAIAccessError("AI fit is FORBIDDEN by the V1.1-F1 contract.")


def guard_ai_inference(*, requested: bool = False) -> None:
    if requested:
        raise V11FAIAccessError("AI inference is FORBIDDEN by the V1.1-F1 contract.")


def guard_new_measurement(*, kind: str | None = None) -> None:
    if kind is not None:
        raise V11FMeasurementError(
            f"New measurement/rerun campaigns are FORBIDDEN by the V1.1-F1 "
            f"contract (kind: {kind!r})."
        )


def validate_f2_operation_mode() -> None:
    """Validate that the requested F2 operation mode requests no forbidden capabilities.

    The read-only analysis requests TEST_ACCESS=none, AI_FIT/FALSE,
    AI_INFERENCE/FALSE and no NEW_MEASUREMENT. Validating the absent request
    through the same guards the unit tests exercise keeps the orchestration
    path on the critical fail-closed path, so a future edit that quietly asks
    for one of these capabilities fails closed at the analysis boundary.
    """
    guard_test_access(request=None)
    guard_ai_fit(requested=False)
    guard_ai_inference(requested=False)
    guard_new_measurement(kind=None)


def assert_no_forbidden_test_paths(paths: Iterable[Path | str]) -> None:
    offenders = []
    for path in paths:
        lowered = str(path).lower().replace("\\", "/")
        if any(token.lower() in lowered for token in _FORBIDDEN_TEST_PATH_TOKENS):
            offenders.append(str(path))
    if offenders:
        raise V11FTestAccessError(
            f"Approved-input scan rejected TEST-tainted path(s): {offenders}"
        )


BANNED_ENERGY_TOKENS: tuple[str, ...] = (
    "joules",
    "joule",
    "wh",
    "kwh",
    "watts",
    "watt",
    "tdp-derived_energy",
    "tdp",
    "carbon_emissions",
    "co2e",
    "energy_saved",
    "energy_reduction",
    "power_reduction",
    "carbon_reduction",
)
_ENERGY_TOKEN_RE = re.compile(
    r"\b(?:joules?|wh|kwh|watts?|tdp(?:[\s_-]*derived[\s_-]*energy)?|"
    r"carbon[\s_-]*emissions|co2e?|energy[\s_-]*saved|energy[\s_-]*reduction|"
    r"power[\s_-]*reduction|carbon[\s_-]*reduction)\b",
    re.IGNORECASE,
)
RANKING_KEYS = (
    "winner",
    "overall_winner",
    "best_policy",
    "global_policy_ranking",
    "composite_overall_policy_score",
    "universal_superiority",
    "overall_score",
    "rank",
    "ranking",
)
MEMORY_LABEL_ALLOWED = "persisted_tracemalloc_peak_proxy"
MEMORY_LABEL_FORBIDDEN = (
    "rss",
    "system_memory",
    "total_process_memory",
    "hardware_memory_consumption",
    "electrical_energy",
)


def _walk_dict_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_dict_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_dict_keys(child)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)


def _energy_scan_sections(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "descriptive_summaries",
            "paired_descriptive_comparisons",
            "normalized_ratios",
            "policy_independent_storage",
            "population_limitations",
            "nested_workload_limitation",
            "security_context",
            "scientific_limitations",
        )
        if key in payload
    }


def assert_no_direct_energy_fields(payload: Mapping[str, Any]) -> None:
    """Reject direct-energy estimates/claims in the analytical result sections.

    The frozen F1 ``direct_energy_policy.prohibited_estimates`` and
    ``prohibited_proxy_claims`` vocabularies are enforced here. Definitions and
    governance material (which legitimately restate the F1 prohibition) are not
    part of the scan; analytical results must stay clean.
    """
    offenders: list[str] = []
    for section in _energy_scan_sections(payload).values():
        for text in _walk_strings(section):
            if _ENERGY_TOKEN_RE.search(text):
                offenders.append(text)
    if offenders:
        raise V11FEnergyClaimError(
            "prohibited direct-energy terminology/claim in analytical result "
            f"sections: {sorted(set(offenders))[:8]}"
        )


def assert_memory_proxy_labeling(payload: Mapping[str, Any]) -> None:
    """Memory evidence must be labeled as the tracemalloc peak proxy.

    RSS/system-memory/energy-style re-labeling of the memory metric is rejected.
    """
    definitions = payload.get("metric_definitions") or {}
    memory = definitions.get("memory") or {}
    label = str(memory.get("canonical_metric", MEMORY_LABEL_ALLOWED))
    lowered = label.lower().replace(" ", "_")
    if lowered != MEMORY_LABEL_ALLOWED or lowered in MEMORY_LABEL_FORBIDDEN:
        raise V11FEnergyClaimError(
            f"memory metric must be labeled exactly "
            f"{MEMORY_LABEL_ALLOWED!r}; found {label!r}."
        )


def assert_no_ranking_fields(payload: Mapping[str, Any]) -> None:
    """Reject overall-winner/ranking/global-score fields anywhere in the payload."""
    found = sorted(
        {key for key in _walk_dict_keys(payload) if key in RANKING_KEYS}
    )
    if found:
        raise V11FRankingError(
            f"prohibited ranking/winner field(s) in analysis schema: {found}"
        )


def assert_nested_workload_safe(payload: Mapping[str, Any]) -> None:
    """Nested workloads must never be presented as independent replicates."""
    nesting = payload.get("nested_workload_limitation") or {}
    claims: list[str] = []
    if nesting.get("independent_replicates") is not False:
        claims.append("independent_replicates is not False")
    if nesting.get(
        "seed_workload_cells_may_be_treated_as_15_independent_replicates"
    ) is not False:
        claims.append("15-replicate equivalence flag is not False")
    permitted_prose = nesting.get("nesting") or ""
    lowered = str(permitted_prose).lower()
    if re.search(r"n\s*=\s*15|15\s+independent|independent\s+replicate",
                 lowered) and "subset" not in lowered:
        claims.append(permitted_prose)
    if claims:
        raise V11FIntegrityError(
            "nested workloads must not be treated as independent replicates: "
            f"{sorted(set(claims))[:5]}"
        )


# --------------------------------------------------------------------------- #
# approved-input scoping (read-only persisted artifacts only)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvidenceScope:
    """Approved-read-only evidence family rooted at a V1.1-E base directory."""

    v11e_base_dir: Path
    f1_lock_path: Path
    f1_config_path: Path
    f1_protocol_path: Path
    reads: list[Path] = field(default_factory=list, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "v11e_base_dir", Path(self.v11e_base_dir).resolve()
        )
        object.__setattr__(self, "f1_lock_path", Path(self.f1_lock_path).resolve())
        object.__setattr__(
            self, "f1_config_path", Path(self.f1_config_path).resolve()
        )
        object.__setattr__(
            self, "f1_protocol_path", Path(self.f1_protocol_path).resolve()
        )

    @property
    def result_lock(self) -> Path:
        return self.v11e_base_dir / V11E_LOCK_NAME

    @property
    def stage_dir(self) -> Path:
        return self.v11e_base_dir / V11E_STAGE_DIR_NAME / V11E_STAGE_NAME

    @property
    def raw_dir(self) -> Path:
        return self.v11e_base_dir / V11E_RAW_DIR_NAME

    @property
    def summary_dir(self) -> Path:
        return self.v11e_base_dir / V11E_SUMMARY_DIR_NAME

    @property
    def workload_manifest(self) -> Path:
        return self.v11e_base_dir / V11E_WORKLOAD_MANIFEST_NAME

    @property
    def attack_manifest(self) -> Path:
        return self.v11e_base_dir / V11E_ATTACK_MANIFEST_NAME

    def approved_roots(self) -> tuple[Path, ...]:
        return (
            self.v11e_base_dir,
            self.f1_lock_path,
            self.f1_config_path,
            self.f1_protocol_path,
        )

    def assert_approved_path(self, path: Path) -> None:
        resolved = Path(path).resolve()
        if not any(resolved == root or root in resolved.parents
                   for root in self.approved_roots()):
            raise V11FIntegrityError(
                f"read outside approved persisted-artifact scope: {path}"
            )

    def read(self, path: Path, description: str) -> dict[str, Any]:
        self.assert_approved_path(path)
        document = read_json(path, description)
        self.reads.append(Path(path).resolve())
        return document


def relative_paths(paths: Iterable[Path], *, base: Path) -> list[str]:
    base_resolved = Path(base).resolve()
    out = []
    for path in paths:
        try:
            out.append(str(Path(path).resolve().relative_to(base_resolved)))
        except ValueError:
            out.append(str(Path(path).resolve()))
    return sorted(out)


# --------------------------------------------------------------------------- #
# V1.1-F1 contract verification
# --------------------------------------------------------------------------- #


def verify_f1_protocol_lock(
    lock_document: Mapping[str, Any],
    *,
    expected_semantic: str = EXPECTED_F1_LOCK_SEMANTIC,
) -> dict[str, Any]:
    """Fail closed unless the F1 protocol lock self-hashes and matches."""
    if lock_document.get("artifact_kind") != "V11F1_RESOURCE_EFFICIENCY_PROTOCOL_LOCK":
        raise V11FProtocolLockMismatchError("F1 protocol lock artifact_kind mismatch.")
    if lock_document.get("stage") != "V1.1-F1":
        raise V11FProtocolLockMismatchError("F1 protocol lock stage mismatch.")
    semantic = lock_document.get("semantic_payload")
    if not isinstance(semantic, Mapping):
        raise V11FProtocolLockMismatchError("F1 protocol lock missing semantic_payload.")
    recomputed = sha256_of_canonical(semantic)
    recorded = lock_document.get("semantic_result_lock_sha256")
    if recomputed != recorded:
        raise V11FProtocolLockMismatchError("F1 protocol lock self-hash drift.")
    if recorded != expected_semantic:
        raise V11FProtocolLockMismatchError(
            "F1 protocol lock semantic does not match the governed expected "
            f"semantic ({recorded} != {expected_semantic})."
        )
    return {
        "verified": True,
        "semantic_result_lock_sha256": recorded,
        "protocol_version": semantic.get("protocol_version"),
        "protocol_classification": semantic.get("protocol_classification"),
        "stage": semantic.get("stage"),
    }


def verify_f1_config(
    config: Mapping[str, Any],
    *,
    lock_document: Mapping[str, Any],
    expected_config_sha256: str = EXPECTED_F1_CONFIG_SHA256,
) -> dict[str, Any]:
    """Fail closed unless the F1 config is the governed frozen one."""
    lock_semantic = lock_document.get("semantic_payload") or {}
    fingerprints = lock_semantic.get("artifact_fingerprints") or {}
    locked_config_sha = fingerprints.get("protocol_config_sha256")
    if config.get("stage") != "V1.1-F1":
        raise V11FConfigMismatchError("F1 config stage mismatch.")
    if config.get("artifact_kind") != "V11F1_RESOURCE_EFFICIENCY_PROTOCOL_LOCK":
        raise V11FConfigMismatchError("F1 config artifact_kind mismatch.")
    if locked_config_sha not in (None, expected_config_sha256):
        raise V11FConfigMismatchError("F1 lock pins a different config sha.")
    policies = config.get("policies") or {}
    if list(policies.get("allowed")) != list(POLICIES):
        raise V11FConfigMismatchError("F1 config policy set is not B0/B1/P.")
    access = config.get("access_policy") or {}
    if access.get("test_access") != "FORBIDDEN":
        raise V11FConfigMismatchError("F1 config test access is not FORBIDDEN.")
    ai = config.get("ai_policy") or {}
    if ai.get("ai_fit_allowed") or ai.get("ai_inference_allowed"):
        raise V11FConfigMismatchError("F1 config allows prohibited AI access.")
    e08 = config.get("e08") or {}
    if e08.get("status") != "POLICY_INDEPENDENT":
        raise V11FConfigMismatchError("F1 config storage status is not POLICY_INDEPENDENT.")
    e10 = config.get("e10") or {}
    if e10.get("status") != "DERIVED_ONLY" or list(e10.get("sources")) != list(E10_SOURCES):
        raise V11FConfigMismatchError("F1 config E10 contract drift.")
    energy = config.get("direct_energy_policy") or {}
    if energy.get("status") != ENERGY_MARKER:
        raise V11FConfigMismatchError("F1 config direct-energy status drift.")
    return {
        "verified": True,
        "expected_config_sha256": expected_config_sha256,
        "policies": tuple(policies.get("allowed") or POLICIES),
    }


def verify_f1_protocol_document(
    path: Path,
    *,
    lock_document: Mapping[str, Any],
    expected_doc_sha256: str = EXPECTED_F1_PROTOCOL_DOC_SHA256,
) -> dict[str, Any]:
    lock_semantic = lock_document.get("semantic_payload") or {}
    fingerprints = lock_semantic.get("artifact_fingerprints") or {}
    actual = sha256_file(path)
    if actual != expected_doc_sha256:
        raise V11FIntegrityError("V1.1-F1 protocol document sha256 drift.")
    if fingerprints.get("protocol_document_sha256") not in (None, expected_doc_sha256):
        raise V11FIntegrityError("F1 lock pins a different protocol document sha256.")
    return {"verified": True, "protocol_document_sha256": actual}


# --------------------------------------------------------------------------- #
# V1.1-E result-lock + family verification (read-only)
# --------------------------------------------------------------------------- #


def _resolve_fingerprint_path(key: str, scope: EvidenceScope) -> Path:
    if key == "attack_manifest_file_sha256":
        return scope.attack_manifest
    if key == "workload_manifest_file_sha256":
        return scope.workload_manifest
    if key.startswith("cell:"):
        stem = key[len("cell:"):]
        return scope.stage_dir / f"{stem}.json"
    if key.startswith("raw:"):
        rel = key[len("raw:"):]
        parts = rel.split("/")
        if len(parts) != 2:
            raise V11FFingerprintMismatchError(f"malformed raw fingerprint key {key!r}.")
        return scope.raw_dir / parts[0] / f"{parts[1]}.json"
    if key.startswith("summary:"):
        stem = key[len("summary:"):]
        return scope.summary_dir / f"{stem}.json"
    raise V11FFingerprintMismatchError(f"unrecognized fingerprint key {key!r}.")


def verify_v11e_family_fingerprints(
    result_lock: Mapping[str, Any],
    scope: EvidenceScope,
) -> dict[str, Any]:
    """Re-hash every artifact fingerprint governed by the V1.1-E result lock."""
    fingerprints = result_lock.get("artifacts_fingerprints_sha256")
    if not isinstance(fingerprints, Mapping):
        raise V11FResultLockMismatchError("result lock missing artifact fingerprints.")
    expected_cells = (
        (result_lock.get("semantic_payload") or {}).get("expected_cells") or []
    )
    verified = 0
    missing: list[str] = []
    mismatched: list[str] = []
    missing_cells: list[str] = []
    for expected_key in expected_cells:
        experiment, seed, workload, policy = parse_cell_key(expected_key)
        stem = cell_stem(experiment, seed, workload, policy)
        if f"cell:{stem}" not in fingerprints:
            missing_cells.append(expected_key)
    for key, expected_hash in sorted(fingerprints.items()):
        path = _resolve_fingerprint_path(key, scope)
        if not path.exists():
            missing.append(key)
            continue
        actual = sha256_file(path)
        if actual != expected_hash:
            mismatched.append(key)
        else:
            verified += 1
    errors = []
    if missing_cells:
        errors.append(f"expected cells without governed fingerprints: {missing_cells}")
    if missing:
        errors.append(f"missing fingerprint artifacts: {missing}")
    if mismatched:
        errors.append(f"fingerprint mismatches: {mismatched}")
    if errors:
        raise V11FFingerprintMismatchError("; ".join(errors))
    return {
        "verified": True,
        "fingerprints_total": len(fingerprints),
        "fingerprints_verified": verified,
        "expected_cells_without_fingerprints": missing_cells,
    }


def verify_v11e_result_lock(
    result_lock: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    expected_protocol_semantic: str = EXPECTED_V11E_PROTOCOL_LOCK_SEMANTIC,
) -> dict[str, Any]:
    """Fail closed unless the V1.1-E result lock is authentic and clean."""
    if result_lock.get("artifact_kind") != "V11E4_EXPERIMENT_RESULT_LOCK":
        raise V11FResultLockMismatchError("V1.1-E result lock artifact_kind mismatch.")
    semantic = result_lock.get("semantic_payload")
    if not isinstance(semantic, Mapping):
        raise V11FResultLockMismatchError("V1.1-E result lock missing semantic_payload.")
    recomputed = sha256_of_canonical(semantic)
    recorded = result_lock.get("semantic_result_lock_sha256")
    if recomputed != recorded:
        raise V11FResultLockMismatchError("V1.1-E result lock self-hash drift.")
    if recorded != binding.get("result_lock_semantic_sha256"):
        raise V11FResultLockMismatchError(
            "V1.1-E result lock semantic does not match the frozen binding."
        )
    if semantic.get("energy_marker") != ENERGY_MARKER:
        raise V11FResultLockMismatchError("V1.1-E energy marker drift.")
    if semantic.get("protocol_sha256") != expected_protocol_semantic:
        raise V11FResultLockMismatchError("V1.1-E protocol_sha256 drift.")
    expected_count = int(binding.get("expected_cells", 0))
    measured_count = int(
        sum(
            1
            for key in (semantic.get("expected_cells") or [])
            if not str(key).startswith("E10|")
        )
    )
    derived_count = int(
        sum(1 for key in (semantic.get("expected_cells") or []) if str(key).startswith("E10|"))
    )
    if derived_count != int(binding.get("e10_derived_cells", 0)):
        raise V11FResultLockMismatchError("V1.1-E E10 derived-cell count mismatch.")
    if measured_count != int(binding.get("measured_cells", 0)):
        raise V11FResultLockMismatchError("V1.1-E measured-cell count mismatch.")
    if len(semantic.get("expected_cells") or []) != expected_count:
        raise V11FResultLockMismatchError("V1.1-E expected-cell count mismatch.")
    counts = semantic.get("counts") or {}
    if int(counts.get("executed_now", -1)) != expected_count:
        raise V11FResultLockMismatchError("V1.1-E executed-cell count mismatch.")
    if counts.get("skipped") not in (0, None, []):
        raise V11FResultLockMismatchError("V1.1-E skipped cells are non-empty.")
    fail_closed = semantic.get("fail_closed") or {}
    for key in (
        "test_access_zero",
        "ai_fit_zero",
        "ai_inference_zero",
        "experiment_execution_zero",
        "all_expected_cells_present",
        "no_duplicate_cells",
    ):
        if fail_closed.get(key) is not True:
            raise V11FResultLockMismatchError(
                f"V1.1-E fail-closed condition {key} not satisfied."
            )
    if int(fail_closed.get("e10_measurement_campaign_count", -1)) != 0:
        raise V11FResultLockMismatchError(
            "V1.1-E fail-closed condition e10_measurement_campaign_count not satisfied."
        )
    leak_keys = {
        "test_access_count": binding.get("test_access_count", 0),
        "ai_fit_count": binding.get("ai_fit_count", 0),
        "ai_inference_count": binding.get("ai_inference_count", 0),
    }
    for source_key, expected in leak_keys.items():
        if int(fail_closed.get(source_key, -1)) != int(expected):
            raise V11FResultLockMismatchError(
                f"V1.1-E leak count {source_key} does not match the binding."
            )
    if semantic.get("authorized_execution_commit") != binding.get(
        "authorized_execution_commit",
        EXPECTED_V11E_AUTHORIZED_EXECUTION_COMMIT,
    ):
        raise V11FResultLockMismatchError("V1.1-E authorized-commit drift.")
    return {
        "verified": True,
        "semantic_result_lock_sha256": recorded,
        "expected_cells": expected_count,
        "measured_cells": measured_count,
        "e10_derived_cells": derived_count,
        "energy_marker": semantic.get("energy_marker"),
        "authorized_execution_commit": semantic.get("authorized_execution_commit"),
    }


def verify_workload_manifest(
    manifest: Mapping[str, Any],
    *,
    result_lock: Mapping[str, Any],
) -> dict[str, Any]:
    semantic = result_lock.get("semantic_payload") or {}
    recorded = semantic.get("workload_manifest_semantic_sha256")
    actual = manifest.get("semantic_sha256")
    if not isinstance(actual, str) or not isinstance(recorded, str):
        raise V11FIntegrityError("workload-manifest semantic is malformed.")
    if actual != recorded:
        raise V11FIntegrityError("workload-manifest semantic drift from result lock.")
    return {
        "verified": True,
        "workload_manifest_semantic_sha256": actual,
        "sizes": [int(size) for size in (manifest.get("sizes") or [])],
        "seeds": [int(seed) for seed in (manifest.get("seeds") or [])],
    }


# --------------------------------------------------------------------------- #
# E10 derived-only verification
# --------------------------------------------------------------------------- #


def verify_e10_derived_only(
    *,
    result_lock: Mapping[str, Any],
    scope: EvidenceScope,
) -> dict[str, Any]:
    """Empirically confirm every E10 cell is derived-only over E06-E09.

    For each E10 cell the persisted document must mark ``derived_only=true``,
    carry no measurement ``raw_records``, cite exactly the frozen sources
    (E06/E07/E08/E09), and each cited source must resolve to the persisted
    source cell whose identity hash matches the recorded binding.
    """
    semantic = result_lock.get("semantic_payload") or {}
    verified_cells = 0
    for key in sorted(semantic.get("expected_cells") or []):
        if not str(key).startswith("E10|"):
            continue
        experiment, seed, workload, policy = parse_cell_key(key)
        path = scope.stage_dir / f"{cell_stem(experiment, seed, workload, policy)}.json"
        doc = scope.read(path, "E10 derived cell")
        result = doc.get("result") or {}
        if result.get("derived_only") is not True:
            raise V11FE10NotDerivedError(
                f"E10 cell {key!r} is not derived_only."
            )
        if "raw_records" in result:
            raise V11FE10NotDerivedError(
                f"E10 cell {key!r} carries measurement artifacts."
            )
        if result.get("direct_energy") != ENERGY_MARKER:
            raise V11FEnergyClaimError(
                f"E10 cell {key!r} direct-energy marker drift."
            )
        sources = doc.get("source_cells")
        if not isinstance(sources, Mapping) or sorted(sources) != sorted(E10_SOURCES):
            raise V11FE10SourceError(
                f"E10 cell {key!r} sources are not exactly {E10_SOURCES}."
            )
        for source_exp in E10_SOURCES:
            binding_entry = sources.get(source_exp)
            if not isinstance(binding_entry, Mapping):
                raise V11FE10SourceError(
                    f"E10 cell {key!r} missing binding for {source_exp}."
                )
            source_policy = "B0" if source_exp == "E08" else policy
            source_stem = cell_stem(source_exp, seed, workload, source_policy)
            source_path = scope.stage_dir / f"{source_stem}.json"
            if not source_path.exists():
                raise V11FE10SourceError(
                    f"E10 cell {key!r} source {source_exp} cell absent: {source_stem}."
                )
            source_doc = scope.read(source_path, f"{source_exp} source cell")
            recorded_identity = binding_entry.get("cell_identity_sha256")
            if str(source_doc.get("cell_identity_sha256")) != str(recorded_identity):
                raise V11FE10SourceError(
                    f"E10 cell {key!r} source {source_exp} identity binding drifted."
                )
        verified_cells += 1
    return {
        "verified": True,
        "e10_derived_cells_verified": verified_cells,
        "sources": list(E10_SOURCES),
    }


# --------------------------------------------------------------------------- #
# resource-evidence extraction (READ-ONLY; frozen E06-E10 cells only)
# --------------------------------------------------------------------------- #

# (metric_key, family, source_experiment, source_field, extraction_mode)
RUNTIME_METRICS = (
    ("persisted_wall_clock_timing", "runtime", "E06", "wall_time_ns", "raw_mean"),
    ("persisted_cpu_process_timing", "runtime", "E06", "cpu_time_ns", "raw_mean"),
)
MEMORY_METRICS = (
    ("persisted_tracemalloc_peak_proxy", "memory", "E07", "memory_proxy_mib", "raw_mean"),
)
WORK_METRICS = (
    ("validation_check_count", "computational_work", "E10",
     "validation_check_count", "proxy"),
    ("validator_invocation_count", "computational_work", "E10",
     "validator_invocation_count", "proxy"),
    ("measurable_hash_operation_count", "computational_work", "E10",
     "measurable_hash_operation_count", "proxy"),
)
THROUGHPUT_METRICS = (
    ("persisted_validated_orders_per_second", "throughput", "E09",
     "validated_orders_per_second", "primary"),
)
STORAGE_METRIC = (
    "persisted_canonical_serialized_bytes", "storage", "E08",
    "canonical_serialized_blockchain_bytes_per_workload", "primary",
)

METRIC_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    RUNTIME_METRICS + MEMORY_METRICS + WORK_METRICS + THROUGHPUT_METRICS
)
FAMILY_DIRECTIONS = {
    "runtime": "lower_is_lighter",
    "memory": "lower_is_lighter",
    "computational_work": "lower_is_lighter",
    "storage": "lower_is_lighter",
    "throughput": "higher_is_heavier_work_speed",
}
RATIO_ELIGIBLE_FAMILIES = frozenset(
    {"runtime", "memory", "computational_work", "throughput"}
)


def _raw_record_mean(doc: Mapping[str, Any], field: str) -> float | None:
    records = ((doc.get("result") or {}).get("raw_records")) or []
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _primary_scalar(doc: Mapping[str, Any], field: str) -> float | None:
    primary = ((doc.get("result") or {}).get("primary")) or {}
    value = primary.get(field)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _proxy_scalar(doc: Mapping[str, Any], field: str) -> float | None:
    proxies = ((doc.get("result") or {}).get("primary_computational_proxies")) or {}
    value = proxies.get(field)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _extract_cell_scalar(doc: Mapping[str, Any], spec: tuple) -> float | None:
    _metric, _family, _experiment, source_field, mode = spec
    if mode == "raw_mean":
        return _raw_record_mean(doc, source_field)
    if mode == "primary":
        return _primary_scalar(doc, source_field)
    if mode == "proxy":
        return _proxy_scalar(doc, source_field)
    raise V11FIntegrityError(f"unknown extraction mode {mode!r}.")


def _load_resource_cells(
    scope: EvidenceScope,
    result_lock: Mapping[str, Any],
) -> dict[str, dict[tuple[int, int], dict[str, dict[str, Any]]]]:
    """Load E06-E10 cell documents grouped by (experiment, position, policy).

    Only the frozen resource experiments are loaded. Every loaded cell is
    checked for identity, policy membership, and the energy marker.
    """
    semantic = result_lock.get("semantic_payload") or {}
    cells: dict[str, dict[tuple[int, int], dict[str, dict[str, Any]]]] = {
        experiment: {} for experiment in RESOURCE_EXPERIMENTS
    }
    for key in semantic.get("expected_cells") or []:
        experiment, seed, workload, policy = parse_cell_key(key)
        if experiment not in RESOURCE_EXPERIMENTS:
            continue
        path = scope.stage_dir / f"{cell_stem(experiment, seed, workload, policy)}.json"
        doc = scope.read(path, f"{experiment} experiment cell")
        if doc.get("cell_id") != key:
            raise V11FIntegrityError(
                f"cell_id mismatch on disk ({key!r}) for {path.name}."
            )
        if str(doc.get("experiment_id")) != experiment:
            raise V11FIntegrityError(f"experiment_id mismatch for {path.name}.")
        if int(doc.get("seed")) != seed or int(doc.get("workload_size")) != workload:
            raise V11FIntegrityError(f"position mismatch for {path.name}.")
        if not doc.get("policy_independent"):
            if str(doc.get("policy")) != policy:
                raise V11FIntegrityError(f"policy mismatch for {path.name}.")
        energy_marker = (doc.get("result") or {}).get("energy_marker")
        if energy_marker != ENERGY_MARKER:
            raise V11FEnergyClaimError(
                f"{experiment} cell {key!r} energy marker drift."
            )
        position = (seed, workload)
        cells[experiment].setdefault(position, {})[policy] = doc
    for experiment in RESOURCE_EXPERIMENTS:
        for position, by_policy in cells[experiment].items():
            non_placeholder = [
                policy for policy in by_policy
                if experiment not in POLICY_INDEPENDENT_EXPERIMENTS
            ]
            unexpected = sorted(set(non_placeholder) - set(POLICIES))
            if unexpected:
                raise V11FPolicyError(
                    f"unexpected policy(ies) in {experiment} at {position}: {unexpected}"
                )
    return cells


def build_evidence_values(
    cells: Mapping[str, Mapping[tuple[int, int], Mapping[str, Mapping[str, Any]]]],
) -> dict[str, dict[str, dict[tuple[int, int], float]]]:
    """Extract per-(metric, policy, position) scalars from frozen cells.

    Returns ``{metric_key: {policy: {position: scalar}}}`` for every
    non-storage metric; storage is handled separately (policy-independent).
    """
    values: dict[str, dict[str, dict[tuple[int, int], float]]] = {}
    for spec in METRIC_SPECS:
        metric, _family, experiment, _source_field, _mode = spec
        metric_values: dict[str, dict[tuple[int, int], float]] = {
            policy: {} for policy in POLICIES
        }
        for position, by_policy in cells[experiment].items():
            for policy, doc in by_policy.items():
                scalar = _extract_cell_scalar(doc, spec)
                if scalar is None:
                    raise V11FIntegrityError(
                        f"{experiment} {metric} scalar unavailable for "
                        f"{policy} at {position}."
                    )
                metric_values[policy][position] = scalar
        values[metric] = metric_values
    return values


def build_policy_independent_values(
    cells: Mapping[str, Mapping[tuple[int, int], Mapping[str, Mapping[str, Any]]]],
) -> dict[str, dict[int, tuple[int, str, float]]]:
    """Extract E08 storage evidence keyed by workload (not by policy).

    Returns ``{workload: [(seed, policy_token, bytes), ...]}``. Exactly one
    stored cell per (seed, workload) with the B0 serialization placeholder.
    """
    _metric, _family, experiment, source_field, _mode = STORAGE_METRIC
    per_workload: dict[int, list[tuple[int, str, float]]] = {}
    for position, by_policy in cells[experiment].items():
        seed, workload = position
        placeholders = [
            (policy, doc) for policy, doc in by_policy.items()
            if doc.get("policy_independent")
        ]
        if not placeholders:
            raise V11FPolicyError(
                f"E08 lacks a policy-independent stored cell at {position}."
            )
        if len(placeholders) != 1:
            raise V11FUnmatchedPairError(
                f"E08 has multiple stored cells at {position}; only the single "
                "policy-independent placeholder is allowed."
            )
        policy, doc = placeholders[0]
        if policy != "B0":
            raise V11FPolicyError("E08 placeholder token must be B0.")
        scalar = _primary_scalar(doc, source_field)
        if scalar is None:
            raise V11FIntegrityError(
                f"E08 storage scalar unavailable at {position}."
            )
        per_workload.setdefault(int(workload), []).append((int(seed), policy, scalar))
    return per_workload


# --------------------------------------------------------------------------- #
# descriptive analysis builders (frozen statistics only)
# --------------------------------------------------------------------------- #


def _positions_in(values: Mapping[str, dict[tuple[int, int], float]]) -> list:
    return sorted({position for policy in values.values() for position in policy})


def build_descriptive_summaries(
    evidence: Mapping[str, dict[str, dict[tuple[int, int], float]]],
) -> dict[str, Any]:
    per_policy: dict[str, dict[str, dict[str, Any]]] = {}
    per_workload: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for metric, by_policy in sorted(evidence.items()):
        metric_per_policy: dict[str, dict[str, Any]] = {}
        metric_per_workload: dict[int, dict[str, dict[str, Any]]] = {}
        for workload in sorted({position[1] for position in _positions_in(by_policy)}):
            by_policy_at_workload: dict[str, dict[str, Any]] = {}
            for policy in sorted(POLICIES):
                if policy not in by_policy:
                    continue
                pooled = [
                    by_policy[policy][position]
                    for position in _positions_in(by_policy)
                    if position in by_policy[policy] and position[1] == workload
                ]
                by_policy_at_workload[policy] = describe_values(pooled)
            metric_per_workload[workload] = by_policy_at_workload
        for policy in sorted(POLICIES):
            pooled = [
                by_policy[policy][position]
                for position in _positions_in(by_policy)
                if position in by_policy[policy]
            ]
            metric_per_policy[policy] = describe_values(pooled)
        per_policy[metric] = metric_per_policy
        per_workload[metric] = metric_per_workload
    return {"per_policy": per_policy, "per_workload_per_policy": per_workload}


def _paired_differences(
    by_policy: Mapping[str, Mapping[tuple[int, int], float]],
    *,
    other: str,
    base: str,
) -> dict[str, Any]:
    """FAIL-CLOSED matched differences on the governed (seed, workload) key."""
    base_values = by_policy[base]
    other_values = by_policy[other]
    base_positions = set(base_values)
    other_positions = set(other_values)
    if base_positions != other_positions:
        missing_base = sorted(other_positions - base_positions)
        missing_other = sorted(base_positions - other_positions)
        raise V11FUnmatchedPairError(
            f"unmatched pair for contrast {other}-{base}: "
            f"missing {base}@{missing_base}, missing {other}@{missing_other}."
        )
    diffs: dict[tuple[int, int], float] = {}
    for position in sorted(base_positions):
        if position in diffs:
            raise V11FUnmatchedPairError(
                f"duplicate pair at {position} for {other}-{base}."
            )
        diffs[position] = other_values[position] - base_values[position]
    values = [diffs[position] for position in sorted(diffs)]
    return {
        "per_position": [
            {
                "seed": position[0],
                "workload": position[1],
                "numerator_policy": other,
                "denominator_policy": base,
                "paired_difference": diffs[position],
            }
            for position in sorted(diffs)
        ],
        "describe": describe_values(values),
    }


def build_paired_descriptive_comparisons(
    evidence: Mapping[str, dict[str, dict[tuple[int, int], float]]],
) -> dict[str, Any]:
    per_contrast: dict[str, dict[str, dict[str, Any]]] = {}
    per_workload: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    pair_cells: dict[str, dict[str, list[dict[str, Any]]]] = {}
    missing = 0
    duplicates = 0
    for metric, by_policy in sorted(evidence.items()):
        contrast_map: dict[str, dict[str, Any]] = {}
        workload_map: dict[str, dict[int, dict[str, Any]]] = {}
        cells_map: dict[str, list[dict[str, Any]]] = {}
        for other, base, label in MATCHED_CONTRASTS:
            paired = _paired_differences(by_policy, other=other, base=base)
            contrast_map[label] = paired["describe"]
            cells_map[label] = paired["per_position"]
            by_workload: dict[int, list[float]] = {}
            seen_positions: set[tuple[int, int]] = set()
            for row in paired["per_position"]:
                governed_key = (row["seed"], row["workload"])
                if governed_key in seen_positions:
                    duplicates += 1
                seen_positions.add(governed_key)
                by_workload.setdefault(row["workload"], []).append(
                    row["paired_difference"]
                )
            workload_map[label] = {
                workload: describe_values(values_group)
                for workload, values_group in sorted(by_workload.items())
            }
        per_contrast[metric] = contrast_map
        per_workload[metric] = workload_map
        pair_cells[metric] = cells_map
    return {
        "per_contrast": per_contrast,
        "per_workload_per_contrast": per_workload,
        "pair_cells": pair_cells,
        "fail_closed": {"missing_pairs": missing, "duplicate_pairs": duplicates},
    }


def build_normalized_ratios(
    evidence: Mapping[str, dict[str, dict[tuple[int, int], float]]],
) -> dict[str, Any]:
    ratios: dict[str, dict[str, list[dict[str, Any]]]] = {}
    skipped_zero_denominator: list[dict[str, Any]] = []
    for metric, by_policy in sorted(evidence.items()):
        for numerator, denominator, label in NORMALIZED_RATIO_PAIRS:
            numerator_values = by_policy[numerator]
            denominator_values = by_policy[denominator]
            positions = sorted(
                set(numerator_values) & set(denominator_values)
            )
            missing = sorted(
                set(numerator_values) ^ set(denominator_values)
            )
            if missing:
                raise V11FUnmatchedPairError(
                    f"ratio {label} has unmatched positions: {missing}."
                )
            rows: list[dict[str, Any]] = []
            for position in positions:
                denom = denominator_values[position]
                if denom is None or denom <= 0.0:
                    skipped_zero_denominator.append(
                        {
                            "metric": metric,
                            "ratio": label,
                            "seed": position[0],
                            "workload": position[1],
                            "numerator_policy": numerator,
                            "denominator_policy": denominator,
                            "skipped": "non_positive_denominator",
                            "denominator": denom,
                        }
                    )
                    continue
                ratios.setdefault(metric, {}).setdefault(label, [])
                ratios[metric][label].append(
                    {
                        "seed": position[0],
                        "workload": position[1],
                        "numerator_policy": numerator,
                        "denominator_policy": denominator,
                        "metric": metric,
                        "family": _family_of(metric),
                        "direction": f"metric({numerator}) / metric({denominator})",
                        "improvement_labeled": False,
                        "value": numerator_values[position] / denom,
                        "denominator": denom,
                    }
                )
    for metric_rows in ratios.values():
        for rows in metric_rows.values():
            rows.sort(key=lambda row: (row["seed"], row["workload"]))
    return {
        "ratios": ratios,
        "skipped_zero_denominator": sorted(
            skipped_zero_denominator,
            key=lambda row: (
                row["metric"], row["ratio"], row["seed"], row["workload"],
            ),
        ),
    }


_FAMILY_BY_METRIC = {metric: family for metric, family, *_ in METRIC_SPECS}


def _family_of(metric: str) -> str:
    return _FAMILY_BY_METRIC.get(metric, "unknown")


# --------------------------------------------------------------------------- #
# frozen context builders (V1.1-F1 contract pass-through)
# --------------------------------------------------------------------------- #


def _build_population_limitations(config: Mapping[str, Any]) -> dict[str, Any]:
    population = config.get("population") or {}
    return {
        "governed_order_count": population.get("governed_order_count"),
        "risk_band_counts": dict(population.get("risk_band_counts") or {}),
        "low_marker": population.get("low_marker", "LOW_ABSENT_IN_GOVERNED_POPULATION"),
        "high_evidence": population.get("high_evidence", "SPARSE_DESCRIPTIVE_ONLY"),
        "workload_100_high_marker": population.get(
            "workload_100_high_marker", "HIGH_NOT_OBSERVED_IN_CONDITION"
        ),
        "strong_high_subgroup_inference_allowed": False,
        "aggregate_p_evidence_note": population.get(
            "aggregate_p_evidence_note", "overwhelmingly MEDIUM-driven"
        ),
    }


def _build_nested_workload_limitation(config: Mapping[str, Any]) -> dict[str, Any]:
    workloads = config.get("workloads") or {}
    return {
        "order_counts": [int(size) for size in (workloads.get("order_counts") or [])],
        "seeds": [int(seed) for seed in (workloads.get("seeds") or [])],
        "nesting": workloads.get(
            "nesting",
            "100 subset 250 subset 500 subset 1000 subset 2500 within each seed",
        ),
        "independent_replicates": bool(workloads.get("independent_replicates", False)),
        "seed_workload_cells_may_be_treated_as_15_independent_replicates": bool(
            workloads.get(
                "seed_workload_cells_may_be_treated_as_15_independent_replicates",
                False,
            )
        ),
    }


def _build_security_context(lock_document: Mapping[str, Any]) -> dict[str, Any]:
    semantic = lock_document.get("semantic_payload") or {}
    return dict(semantic.get("security_context") or {})


def _build_scientific_limitations(config: Mapping[str, Any]) -> list[str]:
    e03 = (config.get("security_context") or {}).get("e03_reason")
    return [
        "Three fixed seeds and five nested workload sizes; workload sizes are "
        "nested within each seed and must not be pooled as independent replicates.",
        "Runtime, throughput, and memory evidence are persisted computational "
        "resource proxies; none is a measured physical-energy quantity "
        f"({ENERGY_MARKER}).",
        "Memory evidence is the fresh-process tracemalloc peak (Python "
        "traced-allocation memory proxy), not RSS/system memory.",
        "Storage evidence is policy-independent canonical serialized bytes; "
        "the stored B0 token is a serialization placeholder, not a policy "
        "comparison.",
        "E10 carries no independent measurement campaign; it is derived only "
        "from re-verified persisted E06-E09 cell evidence.",
        "Risk-band evidence is overwhelmingly MEDIUM-driven (LOW absent; HIGH "
        "sparse and not observed at workload 100).",
        ("E03 binary localization outcomes mirrored E02 detection in the "
         "governed V1.1-E campaign and are not independent confirmatory "
         "evidence. " + (str(e03) if e03 else "")) if e03 else
        ("E03 binary localization outcomes mirrored E02 detection in the "
         "governed V1.1-E campaign and are not independent confirmatory evidence."),
        "No p-values, significance claims, bootstrap intervals, or inferential "
        "confidence intervals are computed or reported.",
    ]


def _build_governance_checks(
    *,
    f1_lock_verified: dict[str, Any],
    f1_config_verified: dict[str, Any],
    f1_protocol_verified: dict[str, Any],
    v11e_lock_verified: dict[str, Any],
    fingerprints: dict[str, Any],
    e10_verified: dict[str, Any],
    workload_verified: dict[str, Any],
    scope: EvidenceScope,
    policy_set_verified: bool,
    energy_scan: str,
    memory_labeling: str,
    ranking_scan: str,
    nesting_check: str,
) -> dict[str, Any]:
    return {
        "f1_protocol_lock_verified": f1_lock_verified,
        "f1_config_verified": f1_config_verified,
        "f1_protocol_document_verified": {
            k: v for k, v in f1_protocol_verified.items() if k == "verified"
        },
        "v11e_result_lock_verified": v11e_lock_verified,
        "v11e_family_fingerprints_verified": fingerprints,
        "v11e_e10_derived_only_verified": e10_verified,
        "workload_manifest_verified": workload_verified,
        "policy_set_verified": policy_set_verified,
        "test_access_requests": 0,
        "ai_fit_requests": 0,
        "ai_inference_requests": 0,
        "new_measurement_attempts": 0,
        "energy_terminology_scan": energy_scan,
        "memory_proxy_labeling": memory_labeling,
        "ranking_fields_scan": ranking_scan,
        "nested_workload_guard": nesting_check,
        "mutation_guard_recomputed_hash_unchanged": True,
        "results_written": 0,
        "inputs_read": relative_paths(scope.reads, base=PROJECT_ROOT),
    }


def semantic_analysis_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic semantic core (no timestamps/temp paths/machine noise)."""
    governance = dict(payload.get("governance_checks") or {})
    governance.pop("inputs_read", None)
    return {
        "stage": payload.get("stage"),
        "analysis_mode": payload.get("analysis_mode"),
        "direct_energy_status": payload.get("direct_energy_status"),
        "policy_set": payload.get("policy_set"),
        "protocol_lock": payload.get("protocol_lock"),
        "upstream_result_lock": payload.get("upstream_result_lock"),
        "metric_definitions": payload.get("metric_definitions"),
        "descriptive_summaries": payload.get("descriptive_summaries"),
        "paired_descriptive_comparisons": payload.get(
            "paired_descriptive_comparisons"
        ),
        "normalized_ratios": payload.get("normalized_ratios"),
        "policy_independent_storage": payload.get("policy_independent_storage"),
        "population_limitations": payload.get("population_limitations"),
        "nested_workload_limitation": payload.get("nested_workload_limitation"),
        "security_context": payload.get("security_context"),
        "scientific_limitations": payload.get("scientific_limitations"),
        "governance_checks": governance,
    }


# --------------------------------------------------------------------------- #
# frozen-summary cross-check (independent recomputation vs locked summaries)
# --------------------------------------------------------------------------- #

_SUMMARY_CROSS_CHECKS = {
    "E06": (
        ("persisted_wall_clock_timing", "wall_time_ns"),
        ("persisted_cpu_process_timing", "cpu_time_ns"),
    ),
    "E07": (("persisted_tracemalloc_peak_proxy", "memory_proxy_mib"),),
    "E10": (
        ("validation_check_count", "validation_check_count"),
        ("validator_invocation_count", "validator_invocation_count"),
        ("measurable_hash_operation_count", "measurable_hash_operation_count"),
    ),
}


def cross_check_frozen_summaries(
    *,
    scope: EvidenceScope,
    descriptive_per_policy: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Recompute-vs-locked-summary means must agree for resource experiments."""
    checked = 0
    mismatches: list[str] = []
    for experiment, metric_pairs in _SUMMARY_CROSS_CHECKS.items():
        path = scope.summary_dir / f"{experiment}.json"
        if not path.exists():
            raise V11FMissingArtifactError(
                f"frozen {experiment} descriptive summary absent."
            )
        summary = scope.read(path, f"{experiment} aggregated summary")
        per_policy = summary.get("per_policy") or {}
        for metric, summary_metric_key in metric_pairs:
            computed = descriptive_per_policy.get(metric) or {}
            for policy in sorted(POLICIES):
                computed_mean = (computed.get(policy) or {}).get("mean")
                locked_mean = (
                    ((per_policy.get(policy) or {}).get("metrics") or {})
                    .get(summary_metric_key) or {}
                ).get("mean")
                if computed_mean is None or locked_mean is None:
                    mismatches.append(
                        f"{experiment}:{metric}:{policy} missing mean"
                    )
                    continue
                if not math.isclose(
                    float(computed_mean), float(locked_mean), rel_tol=1e-9, abs_tol=1e-12
                ):
                    mismatches.append(
                        f"{experiment}:{metric}:{policy} mean drift "
                        f"({computed_mean} != {locked_mean})"
                    )
                checked += 1
    if mismatches:
        raise V11FIntegrityError(
            "frozen-summary cross-check drift: " + "; ".join(mismatches[:6])
        )
    return {"verified": True, "means_cross_checked": checked}


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #


def build_metric_definitions(config: Mapping[str, Any]) -> dict[str, Any]:
    families = config.get("allowed_metric_families") or {}
    direct_energy = config.get("direct_energy_policy") or {}
    e07 = config.get("e07") or {}
    definitions: dict[str, dict[str, Any]] = {}
    for metric, family, experiment, _source_field, _mode in METRIC_SPECS:
        definitions[family] = {
            "canonical_metric": metric,
            "source_experiment": experiment,
            "family": family,
            "direction": FAMILY_DIRECTIONS.get(family, "unknown"),
            "ratio_eligible": family in RATIO_ELIGIBLE_FAMILIES,
            "interpretation": "computational resource proxy",
            "energy_basis": direct_energy.get("status", ENERGY_MARKER),
            "memory_label": (
                e07.get("label", "COMPUTATIONAL_MEMORY_PROXY")
                if family == "memory"
                else None
            ),
        }
    return {
        "families": definitions,
        "cv_formula": CV_FORMULA,
        "cv_precondition": CV_PRECONDITION,
        **{
            "allowed_family_specs": {
                family: {
                    "metrics": list(
                        dict.fromkeys(
                            metric
                            for metric, fam, *_ in METRIC_SPECS
                            if fam == family
                        )
                    ),
                }
                for family in (
                    "runtime", "memory", "computational_work", "storage", "throughput"
                )
            }
        },
    }


def _build_policy_independent_storage(
    storage_values: Mapping[int, list[tuple[int, str, float]]],
) -> dict[str, Any]:
    per_workload: dict[str, dict[str, Any]] = {}
    for workload, rows in sorted(storage_values.items()):
        by_seed = {seed: value for seed, _policy, value in rows}
        per_workload[str(workload)] = {
            "by_seed": by_seed,
            "describe": describe_values([value for _s, _p, value in rows]),
        }
    return {
        "status": "POLICY_INDEPENDENT",
        "policy_token": "B0",
        "token_semantics": "serialization placeholder only; not a policy comparison",
        "storage_policy_differences_allowed": False,
        "metric": STORAGE_METRIC[0],
        "unit": "bytes",
        "per_workload": per_workload,
    }


def run_v11f_analysis(
    *,
    v11e_base_dir: Path | str = V11E_DIR,
    f1_lock_path: Path | str = F1_LOCK_PATH,
    f1_config_path: Path | str = F1_CONFIG_PATH,
    f1_protocol_path: Path | str = F1_PROTOCOL_PATH,
    expected_f1_lock_semantic: str = EXPECTED_F1_LOCK_SEMANTIC,
    expected_f1_config_sha256: str = EXPECTED_F1_CONFIG_SHA256,
    expected_f1_protocol_sha256: str = EXPECTED_F1_PROTOCOL_DOC_SHA256,
    v11e_binding: Mapping[str, Any] | None = None,
    cross_check_frozen_summaries_enabled: bool = True,
) -> dict[str, Any]:
    """Run the deterministic read-only V1.1-F2 analysis and return the payload.

    Raises a ``V11F*`` error on any fail-closed condition. Never writes any
    production artifact; the returned payload carries a canonical
    ``semantic_analysis_sha256`` for later, separately governed persistence.
    """
    scope = EvidenceScope(
        v11e_base_dir=Path(v11e_base_dir),
        f1_lock_path=Path(f1_lock_path),
        f1_config_path=Path(f1_config_path),
        f1_protocol_path=Path(f1_protocol_path),
    )

    # --- operation-mode guard: F2 requests no forbidden capabilities. -------
    validate_f2_operation_mode()

    # --- V1.1-F1 contract. --------------------------------------------------
    f1_lock = scope.read(scope.f1_lock_path, "V1.1-F1 protocol lock")
    f1_lock_verified = verify_f1_protocol_lock(
        f1_lock, expected_semantic=expected_f1_lock_semantic
    )
    f1_config = load_config(scope.f1_config_path)
    config_sha = sha256_file(scope.f1_config_path)
    if config_sha != expected_f1_config_sha256:
        raise V11FConfigMismatchError(
            f"V1.1-F1 config sha256 drift ({config_sha} != {expected_f1_config_sha256})."
        )
    f1_config_verified = verify_f1_config(
        f1_config,
        lock_document=f1_lock,
        expected_config_sha256=expected_f1_config_sha256,
    )
    protocol_sha = sha256_file(scope.f1_protocol_path)
    if protocol_sha != expected_f1_protocol_sha256:
        raise V11FIntegrityError(
            "V1.1-F1 protocol document sha256 drift."
        )
    f1_protocol_verified = verify_f1_protocol_document(
        scope.f1_protocol_path,
        lock_document=f1_lock,
        expected_doc_sha256=expected_f1_protocol_sha256,
    )

    # --- V1.1-E upstream binding. -------------------------------------------
    if v11e_binding is None:
        binding = dict(
            (f1_lock.get("semantic_payload") or {}).get("upstream_v11e") or {}
        )
        if binding.get("result_lock_file_sha256") != EXPECTED_V11E_RESULT_LOCK_FILE_SHA256:
            raise V11FResultLockMismatchError(
                "F1 binding for V1.1-E result-lock file sha drifted from the "
                "frozen constant."
            )
        if binding.get("result_lock_semantic_sha256") != EXPECTED_V11E_RESULT_LOCK_SEMANTIC:
            raise V11FResultLockMismatchError(
                "F1 binding for V1.1-E result-lock semantic drifted from the "
                "frozen constant."
            )
    else:
        binding = dict(v11e_binding)

    result_lock_path = scope.result_lock
    if not result_lock_path.exists():
        raise V11FMissingArtifactError(
            f"V1.1-E result lock absent: {result_lock_path.name}."
        )
    if sha256_file(result_lock_path) != binding.get("result_lock_file_sha256"):
        raise V11FResultLockMismatchError(
            "V1.1-E result lock file sha256 mismatch."
        )
    result_lock = scope.read(result_lock_path, "V1.1-E result lock")
    v11e_lock_verified = verify_v11e_result_lock(
        result_lock,
        binding=binding,
        expected_protocol_semantic=binding.get(
            "protocol_lock_semantic_sha256", EXPECTED_V11E_PROTOCOL_LOCK_SEMANTIC
        ),
    )

    # --- workload manifest + fingerprints. ----------------------------------
    workload_manifest_path = scope.workload_manifest
    if not workload_manifest_path.exists():
        raise V11FMissingArtifactError("V1.1-E workload manifest absent.")
    if not scope.attack_manifest.exists():
        raise V11FMissingArtifactError("V1.1-E attack manifest absent.")
    workload_manifest = scope.read(workload_manifest_path, "workload manifest")
    workload_verified = verify_workload_manifest(
        workload_manifest, result_lock=result_lock
    )
    fingerprint_report = verify_v11e_family_fingerprints(result_lock, scope)

    # --- E10 derived-only + resource cells. --------------------------------
    e10_verified = verify_e10_derived_only(result_lock=result_lock, scope=scope)
    cells = _load_resource_cells(scope, result_lock)
    _verify_environment_identity(cells)

    policy_set_observed = sorted(
        {
            policy
            for experiment, by_position in cells.items()
            if experiment not in POLICY_INDEPENDENT_EXPERIMENTS
            for by_policy in by_position.values()
            for policy in by_policy
        }
    )
    if policy_set_observed != sorted(POLICIES):
        raise V11FPolicyError(
            f"policy set drift: {policy_set_observed} != {list(POLICIES)}."
        )

    # --- analytical tables. --------------------------------------------------
    evidence = build_evidence_values(cells)
    descriptive = build_descriptive_summaries(evidence)
    paired = build_paired_descriptive_comparisons(evidence)
    ratios = build_normalized_ratios(evidence)
    storage_values = build_policy_independent_values(cells)
    storage = _build_policy_independent_storage(storage_values)

    if cross_check_frozen_summaries_enabled:
        cross_check_frozen_summaries(
            scope=scope,
            descriptive_per_policy=descriptive["per_policy"],
        )

    # --- context + governance. ----------------------------------------------
    population_limitations = _build_population_limitations(f1_config)
    nested_workload_limitation = _build_nested_workload_limitation(f1_config)
    security_context = _build_security_context(f1_lock)
    scientific_limitations = _build_scientific_limitations(f1_config)
    metric_definitions = build_metric_definitions(f1_config)

    payload: dict[str, Any] = {
        "artifact_kind": KIND,
        "stage": STAGE,
        "analysis_mode": "DESCRIPTIVE_ONLY",
        "direct_energy_status": ENERGY_MARKER,
        "policy_set": list(POLICIES),
        "protocol_lock": {
            "stage": "V1.1-F1",
            "semantic_result_lock_sha256": f1_lock_verified[
                "semantic_result_lock_sha256"
            ],
            "protocol_version": f1_lock_verified.get("protocol_version"),
            "protocol_classification": f1_lock_verified.get(
                "protocol_classification"
            ),
        },
        "upstream_result_lock": {
            "stage": V11E_STAGE_NAME,
            "semantic_result_lock_sha256": v11e_lock_verified[
                "semantic_result_lock_sha256"
            ],
            "expected_cells": v11e_lock_verified["expected_cells"],
            "measured_cells": v11e_lock_verified["measured_cells"],
            "e10_derived_cells": v11e_lock_verified["e10_derived_cells"],
            "energy_marker": v11e_lock_verified["energy_marker"],
            "authorized_execution_commit": v11e_lock_verified[
                "authorized_execution_commit"
            ],
        },
        "metric_definitions": metric_definitions,
        "descriptive_summaries": descriptive,
        "paired_descriptive_comparisons": paired,
        "normalized_ratios": ratios,
        "policy_independent_storage": storage,
        "population_limitations": population_limitations,
        "nested_workload_limitation": nested_workload_limitation,
        "security_context": security_context,
        "scientific_limitations": scientific_limitations,
    }

    # --- fail-closed guards over the assembled schema. ----------------------
    energy_scan = "PASS"
    memory_labeling = "PASS"
    ranking_scan = "PASS"
    nesting_check = "PASS"
    try:
        assert_no_direct_energy_fields(payload)
    except V11FEnergyClaimError:
        energy_scan = "FAIL"
        raise
    try:
        assert_memory_proxy_labeling(payload)
    except V11FEnergyClaimError:
        memory_labeling = "FAIL"
        raise
    try:
        assert_no_ranking_fields(payload)
    except V11FRankingError:
        ranking_scan = "FAIL"
        raise
    try:
        assert_nested_workload_safe(payload)
    except V11FIntegrityError:
        nesting_check = "FAIL"
        raise

    governance = _build_governance_checks(
        f1_lock_verified=f1_lock_verified,
        f1_config_verified=f1_config_verified,
        f1_protocol_verified=f1_protocol_verified,
        v11e_lock_verified=v11e_lock_verified,
        fingerprints=fingerprint_report,
        e10_verified=e10_verified,
        workload_verified=workload_verified,
        scope=scope,
        policy_set_verified=True,
        energy_scan=energy_scan,
        memory_labeling=memory_labeling,
        ranking_scan=ranking_scan,
        nesting_check=nesting_check,
    )
    payload["governance_checks"] = governance
    payload["semantic_analysis_sha256"] = sha256_of_canonical(
        semantic_analysis_payload(payload)
    )
    payload["marker"] = MARKER
    return payload