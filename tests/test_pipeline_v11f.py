"""V1.1-F2 read-only resource-efficiency analysis pipeline unit tests.

Tests exercise the deterministic analytical layer over a small synthetic
V1.1-E family (temp directories only) plus optional read-only checks against
the real frozen V1.1-E artifacts. No governed experiment is ever executed and
no production output is ever written. The real V1.1-F1 protocol
lock/config/document are the authoritative F1 contract and are read-only.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

import src.pipeline_v11f as f2
from src.pipeline_v11f import (
    V11FAIAccessError,
    V11FConfigMismatchError,
    V11FEnergyClaimError,
    V11FE10NotDerivedError,
    V11FE10SourceError,
    V11FFingerprintMismatchError,
    V11FIntegrityError,
    V11FMeasurementError,
    V11FMissingArtifactError,
    V11FPolicyError,
    V11FProtocolLockMismatchError,
    V11FResultLockMismatchError,
    V11FRankingError,
    V11FTestAccessError,
    V11FUnmatchedPairError,
    canonical_json,
    describe_values,
    guard_ai_fit,
    guard_ai_inference,
    guard_new_measurement,
    guard_test_access,
    run_v11f_analysis,
    semantic_analysis_payload,
    sha256_of_canonical,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
F1_LOCK_PATH = PROJECT_ROOT / "results" / "blockchain" / "v11f" / "v11f_protocol_lock.json"
F1_CONFIG_PATH = PROJECT_ROOT / "config" / "resource_efficiency_v11f.yaml"
F1_PROTOCOL_PATH = PROJECT_ROOT / "docs" / "v11f_green_evaluation_protocol.md"
REAL_V11E_DIR = PROJECT_ROOT / "results" / "blockchain" / "v11e"

TEST_SEEDS = (522, 523)
TEST_WORKLOADS = (100, 250)
E2_SEMANTIC = "8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239"
EXEC_COMMIT = "e206ef7a32fef4dc8517cb7c592ea52713b26f2f"
ENERGY = "DIRECT_ENERGY_UNAVAILABLE"


# --------------------------------------------------------------------------- #
# synthetic V1.1-E family builder (temp dir only)
# --------------------------------------------------------------------------- #


def _record_fields(seed: int, workload: int, policy: str, index: int) -> dict:
    pi = f2.POLICIES.index(policy) if policy in f2.POLICIES else 7
    cpu = 100000 + seed * 1000 + workload * 40 + pi * 2000 + index * 7
    wall = cpu + 500 + index * 3
    memory = 1.2 + (seed + workload + pi + index) * 0.000000001
    storage = 1000 + seed + workload * 5
    return {"cpu": cpu, "wall": wall, "memory": memory, "throughput": 5000 + seed * 10 + workload * 2 + pi * 300, "storage": storage}


def _identity_sha(experiment: str, seed: int, workload: int, policy: str) -> str:
    return sha256_of_canonical(
        {
            "experiment_id": experiment,
            "seed": int(seed),
            "workload_size": int(workload),
            "policy": policy,
            "protocol_sha256": E2_SEMANTIC,
        }
    )


def _base_environment() -> dict:
    return {
        "energy_marker": ENERGY,
        "threads": "1",
        "timing_boundary": "validation_policy_call_only",
        "warmup": "one_unmeasured_synthetic_warmup_per_policy",
    }


def make_raw_record(
    experiment: str, seed: int, workload: int, policy: str, index: int
) -> dict:
    fields = _record_fields(seed, workload, policy, index)
    record = {
        "attack": None,
        "blocks": max(1, int(workload) // 20),
        "cpu_time_ns": fields["cpu"] if experiment in ("E06", "E09") else None,
        "wall_time_ns": fields["wall"] if experiment in ("E06", "E09") else None,
        "memory_proxy_mib": fields["memory"] if experiment == "E07" else None,
        "hash_operation_count": 1,
        "orders": workload,
        "policy": policy,
        "repetition": 1,
        "risk_counts": {"HIGH": 0, "LOW": 0, "MEDIUM": workload},
        "validation_outcome": True,
        "detection_outcome": None,
        "localization_outcome": None,
        "energy_marker": ENERGY,
        "experiment_id": experiment,
    }
    return record


def make_cell(
    experiment: str,
    seed: int,
    workload: int,
    policy: str,
    *,
    policy_independent: bool = False,
    derived_only: bool | None = None,
    override_policy: str | None = None,
    tamper_source_suffix: str | None = None,
) -> dict:
    actual_policy = override_policy if override_policy is not None else policy
    result: dict[str, object]
    if experiment == "E09":
        fields = _record_fields(seed, workload, policy, 0)
        result = {
            "context": {"environment": _base_environment(), "experiment_id": "E09"},
            "energy_marker": ENERGY,
            "experiment": "E09",
            "primary": {
                "total_validated_orders": workload,
                "total_validated_blocks": max(1, workflow_blocks(workload)),
                "validated_orders_per_second": fields["throughput"],
                "validated_blocks_per_second": fields["throughput"] * 5,
            },
            "raw_records": [
                make_raw_record("E09", seed, workload, policy, index)
                for index in range(4)
            ],
            "semantic_sha256": "",
        }
    elif experiment == "E08":
        fields = _record_fields(seed, workload, policy, 0)
        result = {
            "context": {"environment": _base_environment(), "experiment_id": "E08"},
            "energy_marker": ENERGY,
            "experiment": "E08",
            "note": "E08 policy-independent workload-level primary evidence only.",
            "policy_independent": True,
            "primary": {
                "bytes_per_order": max(1, int(workload) + seed),
                "bytes_per_block": max(1, int(workload) * 20 + seed),
                "canonical_serialized_blockchain_bytes_per_workload": fields["storage"],
            },
            "semantic_sha256": "",
        }
    elif experiment == "E10":
        fields = _record_fields(seed, workload, policy, 0)
        proxies = {
            "validation_check_count": int(workload) * 4,
            "validator_invocation_count": int(workload),
            "measurable_hash_operation_count": int(workload),
            "runtime_proxy": {"max": 1.0, "mean": 1.0, "min": 1.0,
                              "median": 1.0, "std_ddof_1": 0.0},
        }
        result = {
            "context": {"environment": _base_environment(), "experiment_id": "E10"},
            "derived_only": True if derived_only is None else derived_only,
            "direct_energy": ENERGY,
            "energy_marker": ENERGY,
            "experiment": "E10",
            "memory_evidence": {"max": 1.0, "mean": 1.0, "min": 1.0,
                                "median": 1.0, "std_ddof_1": 0.0},
            "storage_evidence": fields["storage"],
            "throughput_evidence": fields["throughput"],
            "primary_computational_proxies": proxies,
            "never_report": ["Joules", "joule", "Wh", "TDP", "kWh",
                             "physical_energy_savings"],
            "no_fourth_campaign": True,
        }
    else:  # E06 / E07
        raw_experiment = experiment
        result = {
            "context": {"environment": _base_environment(),
                        "experiment_id": experiment},
            "energy_marker": ENERGY,
            "experiment": experiment,
            "primary": {},
            "label": "COMPUTATIONAL_MEMORY_PROXY" if experiment == "E07" else None,
            "measurement": "tracemalloc_fresh_worker" if experiment == "E07" else None,
            "unit": "MiB" if experiment == "E07" else None,
            "timing_principal": True if experiment == "E06" else None,
            "timing_role": "PRINCIPAL" if experiment == "E06" else None,
            "raw_records": [
                make_raw_record(raw_experiment, seed, workload, policy, index)
                for index in range(4)
            ],
            "semantic_sha256": "",
        }

    doc = {
        "artifact_kind": "V11E4_EXPERIMENT_CELL",
        "experiment_id": experiment,
        "seed": int(seed),
        "workload_size": int(workload),
        "policy": actual_policy,
        "cell_id": f2.cell_key(experiment, seed, workload, policy),
        "cell_identity_sha256": _identity_sha(experiment, seed, workload, policy),
        "protocol_sha256": E2_SEMANTIC,
        "authorized_execution_commit": EXEC_COMMIT,
        "workload_manifest_semantic_sha256": "synthetic-workload-000",
        "policy_independent": policy_independent,
        "repetition": 1,
        "environment": _base_environment(),
        "result": result,
        "source_cells": {},
        "semantic_sha256": "",
        "version": 1,
    }
    if experiment == "E10":
        doc["source_cells"] = {
            source_exp: {
                "experiment_id": source_exp,
                "cell_id": f2.cell_key(
                    source_exp, seed, workload,
                    "B0" if source_exp == "E08" else policy,
                ),
                "cell_identity_sha256": _identity_sha(
                    source_exp, seed, workload,
                    "B0" if source_exp == "E08" else policy,
                ),
                "cell_semantic_sha256": "synthetic-source-000",
            }
            for source_exp in f2.E10_SOURCES
        }
        if tamper_source_suffix:
            doc["source_cells"][tamper_source_suffix] = {
                "experiment_id": tamper_source_suffix,
                "cell_id": "",
                "cell_identity_sha256": "synthetic-extra-000",
            }
    doc["semantic_sha256"] = sha256_of_canonical(
        {k: v for k, v in doc.items() if k != "semantic_sha256"}
    )
    return doc


def workflow_blocks(workload: int) -> int:
    return max(1, int(workload) // 20)


def make_raw_observation_document(cell_doc: dict) -> dict:
    records = ((cell_doc.get("result") or {}).get("raw_records")) or []
    included = {
        "artifact_kind": "V11E4_RAW_RUN_OBSERVATIONS",
        "experiment_id": cell_doc["experiment_id"],
        "seed": cell_doc["seed"],
        "workload_size": cell_doc["workload_size"],
        "policy": cell_doc["policy"],
        "cell_id": cell_doc["cell_id"],
        "cell_identity_sha256": cell_doc["cell_identity_sha256"],
        "protocol_sha256": E2_SEMANTIC,
        "authorized_execution_commit": EXEC_COMMIT,
        "record_count": len(records),
        "records": records,
        "note": "synthetic observation",
    }
    return {**included, "semantic_sha256": sha256_of_canonical(included)}


def _pooled_mean(cell_docs, metric_field: str) -> float | None:
    scalars = []
    for doc in cell_docs:
        values = [
            float(record[metric_field])
            for record in (doc.get("result") or {}).get("raw_records", [])
            if isinstance(record.get(metric_field), (int, float))
        ]
        if values:
            scalars.append(sum(values) / len(values))
    if not scalars:
        return None
    return sum(scalars) / len(scalars)


class SyntheticFamily:
    """A small, deterministic, internally consistent V1.1-E evidence family."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.base = self.root / "v11e"
        self.stage = self.base / "v11e_experiment_results" / "V1.1-E4"
        self.raw = self.base / "v11e_raw_run_observations"
        self.summaries = self.base / "v11e_aggregated_descriptive_summaries"
        for path in (self.stage, self.raw, self.summaries):
            path.mkdir(parents=True, exist_ok=True)
        self.expected_keys: list[str] = []
        self.cell_docs: dict[str, dict] = {}
        self._build()

    # -- construction ----------------------------------------------------- #
    def _build(self) -> None:
        for experiment in ("E06", "E07", "E09"):
            for seed in TEST_SEEDS:
                for workload in TEST_WORKLOADS:
                    for policy in f2.POLICIES:
                        doc = make_cell(experiment, seed, workload, policy)
                        self._write_cell(doc)
                        self._write_raw(doc)
                        self.expected_keys.append(doc["cell_id"])
        for seed in TEST_SEEDS:
            for workload in TEST_WORKLOADS:
                doc = make_cell("E08", seed, workload, "B0", policy_independent=True)
                self._write_cell(doc)
                self._write_raw(doc)
                self.expected_keys.append(doc["cell_id"])
                for policy in f2.POLICIES:
                    doc10 = make_cell("E10", seed, workload, policy)
                    self._write_cell(doc10)
                    self.expected_keys.append(doc10["cell_id"])
        self._write_manifests()
        self._write_summaries()
        self.recompute()

    def _write_cell(self, doc: dict, path: Path | None = None) -> Path:
        target = path or (
            self.stage / f"{doc['experiment_id']}_s{int(doc['seed'])}_w{int(doc['workload_size']):05d}_p{doc['policy']}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f2.canonical_json(doc), encoding="utf-8")
        self.cell_docs[doc["cell_id"]] = doc
        return target

    def _write_raw(self, doc: dict) -> Path:
        name = Path(doc["cell_id"].replace("|", "_")).stem
        sub = self.raw / f"{doc['experiment_id']}"
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{doc['experiment_id']}_s{int(doc['seed'])}_w{int(doc['workload_size']):05d}_p{doc['policy']}.json"
        path.write_text(f2.canonical_json(make_raw_observation_document(doc)), encoding="utf-8")
        return path

    def cell_path(self, experiment: str, seed: int, workload: int, policy: str) -> Path:
        return self.stage / (f2.cell_stem(experiment, seed, workload, policy) + ".json")

    # -- manifests -------------------------------------------------------- #
    def _write_manifests(self) -> None:
        workload = {
            "artifact_kind": "V11E4_WORKLOAD_MANIFEST",
            "stage": "V1.1-E4",
            "sizes": list(TEST_WORKLOADS),
            "seeds": list(TEST_SEEDS),
            "nested_prefixes": "100 subset 250 subset 500 subset 1000 subset 2500 within each seed",
            "no_replacement": True,
            "protocol_sha256": E2_SEMANTIC,
            "sampling": {"seed": 42},
        }
        workload["semantic_sha256"] = sha256_of_canonical(
            {k: v for k, v in workload.items() if k != "semantic_sha256"}
        )
        attack = {
            "artifact_kind": "V11E4_ATTACK_INSTANCE_MANIFEST",
            "stage": "V1.1-E4",
            "sizes": list(TEST_WORKLOADS),
            "seeds": list(TEST_SEEDS),
            "protocol_sha256": E2_SEMANTIC,
            "copy_only_target_blocks": True,
            "donor_read_only": True,
        }
        attack["semantic_sha256"] = sha256_of_canonical(
            {k: v for k, v in attack.items() if k != "semantic_sha256"}
        )
        (self.base / f2.V11E_WORKLOAD_MANIFEST_NAME).write_text(
            f2.canonical_json(workload), encoding="utf-8"
        )
        (self.base / f2.V11E_ATTACK_MANIFEST_NAME).write_text(
            f2.canonical_json(attack), encoding="utf-8"
        )
        self.workload_semantic = workload["semantic_sha256"]

    # -- summaries -------------------------------------------------------- #
    def _write_summaries(self) -> None:
        metrics_by_experiment = {
            "E06": ("wall_time_ns", "cpu_time_ns"),
            "E07": ("memory_proxy_mib",),
            "E09": ("wall_time_ns", "cpu_time_ns"),
        }
        for experiment, metric_names in metrics_by_experiment.items():
            per_policy: dict[str, dict] = {}
            for policy in f2.POLICIES:
                cells = [
                    self.cell_docs[f2.cell_key(experiment, seed, workload, policy)]
                    for seed in TEST_SEEDS
                    for workload in TEST_WORKLOADS
                ]
                metrics = {
                    name: describe_values(_cell_scalar_list(cells, name))
                    for name in metric_names
                }
                per_policy[policy] = {"metrics": metrics, "rates": {}}
            summary = {
                "artifact_kind": "V11E4_AGGREGATED_DESCRIPTIVE_SUMMARY",
                "experiment_id": experiment,
                "protocol_sha256": E2_SEMANTIC,
                "authorized_execution_commit": EXEC_COMMIT,
                "descriptive_statistics": "mean/median/sample_std_ddof1/min/max only",
                "per_policy": per_policy,
                "paired_differences": {},
                "energy_marker": ENERGY,
            }
            summary["semantic_sha256"] = sha256_of_canonical(
                {k: v for k, v in summary.items() if k != "semantic_sha256"}
            )
            (self.summaries / f"{experiment}.json").write_text(
                f2.canonical_json(summary), encoding="utf-8"
            )
        e10_per_policy: dict[str, dict] = {}
        for policy in f2.POLICIES:
            metrics = {}
            for metric in ("validation_check_count", "validator_invocation_count",
                           "measurable_hash_operation_count"):
                cells = [
                    self.cell_docs[f2.cell_key("E10", seed, workload, policy)]
                    for seed in TEST_SEEDS
                    for workload in TEST_WORKLOADS
                ]
                scalars = [
                    float(
                        ((doc.get("result") or {}).get("primary_computational_proxies") or {})[
                            metric
                        ]
                    )
                    for doc in cells
                ]
                metrics[metric] = describe_values(scalars)
            e10_per_policy[policy] = {"metrics": metrics, "rates": {}}
        e10 = {
            "artifact_kind": "V11E4_AGGREGATED_DESCRIPTIVE_SUMMARY",
            "experiment_id": "E10",
            "protocol_sha256": E2_SEMANTIC,
            "authorized_execution_commit": EXEC_COMMIT,
            "descriptive_statistics": "mean/median/sample_std_ddof1/min/max only",
            "per_policy": e10_per_policy,
            "paired_differences": {},
            "energy_marker": ENERGY,
        }
        e10["semantic_sha256"] = sha256_of_canonical(
            {k: v for k, v in e10.items() if k != "semantic_sha256"}
        )
        (self.summaries / "E10.json").write_text(
            f2.canonical_json(e10), encoding="utf-8"
        )

    # -- result lock + binding -------------------------------------------- #
    def recompute(self) -> None:
        """Recalc fingerprints + lock semantic against the current files."""
        keys = sorted(set(self.expected_keys))
        semantic_payload = {
            "stage": "V1.1-E4",
            "protocol_sha256": E2_SEMANTIC,
            "authorized_execution_commit": EXEC_COMMIT,
            "workload_manifest_semantic_sha256": self.workload_semantic,
            "expected_cells": keys,
            "executed_cells": keys,
            "skipped_cells": [],
            "counts": {
                "expected_total": len(keys),
                "executed_now": len(keys),
                "skipped": 0,
            },
            "fail_closed": {
                "expected_cell_count": len(keys),
                "persisted_cell_count": len(keys),
                "all_expected_cells_present": True,
                "no_duplicate_cells": True,
                "test_access_count": 0,
                "test_access_zero": True,
                "ai_fit_count": 0,
                "ai_fit_zero": True,
                "ai_inference_count": 0,
                "ai_inference_zero": True,
                "experiment_execution_count": 0,
                "experiment_execution_zero": True,
                "e10_measurement_campaign_count": 0,
                "all_cell_semantics_valid": True,
            },
            "energy_marker": ENERGY,
        }
        fingerprints: dict[str, str] = {}
        for key in keys:
            experiment, seed, workload, policy = f2.parse_cell_key(key)
            file_path = self.stage / (
                f2.cell_stem(experiment, seed, workload, policy) + ".json"
            )
            if not file_path.exists():
                continue
            fingerprints[f"cell:{file_path.stem}"] = f2.sha256_file(file_path)
        for sub in sorted(p for p in self.raw.iterdir() if p.is_dir()):
            for file_path in sorted(sub.glob("*.json")):
                fingerprints[f"raw:{sub.name}/{file_path.stem}"] = f2.sha256_file(
                    file_path
                )
        for file_path in sorted(self.summaries.glob("*.json")):
            fingerprints[f"summary:{file_path.stem}"] = f2.sha256_file(file_path)
        fingerprints["workload_manifest_file_sha256"] = f2.sha256_file(
            self.base / f2.V11E_WORKLOAD_MANIFEST_NAME
        )
        fingerprints["attack_manifest_file_sha256"] = f2.sha256_file(
            self.base / f2.V11E_ATTACK_MANIFEST_NAME
        )
        lock = {
            "artifact_kind": "V11E4_EXPERIMENT_RESULT_LOCK",
            "stage": "V1.1-E4",
            "semantic_payload": semantic_payload,
            "semantic_result_lock_sha256": sha256_of_canonical(semantic_payload),
            "artifacts_fingerprints_sha256": fingerprints,
            "execution_metadata": {
                "experiment_status": "EXECUTED",
                "generated_at_utc_iso": "2026-09-24T00:00:00Z",
            },
        }
        lock_path = self.base / f2.V11E_LOCK_NAME
        lock_path.write_text(f2.canonical_json(lock), encoding="utf-8")
        self.lock = lock
        measured = sum(1 for key in keys if not key.startswith("E10|"))
        derived = sum(1 for key in keys if key.startswith("E10|"))
        self.binding = {
            "stage": "V1.1-E4",
            "result_lock_semantic_sha256": lock["semantic_result_lock_sha256"],
            "result_lock_file_sha256": f2.sha256_file(lock_path),
            "protocol_lock_semantic_sha256": E2_SEMANTIC,
            "expected_cells": len(keys),
            "measured_cells": measured,
            "e10_derived_cells": derived,
            "missing_cells": 0,
            "duplicate_cells": 0,
            "test_access_count": 0,
            "ai_fit_count": 0,
            "ai_inference_count": 0,
            "authorized_execution_commit": EXEC_COMMIT,
            "immutable": True,
            "regeneration_allowed": False,
        }
        self.lock_path = lock_path

    # -- mutations -------------------------------------------------------- #
    def mutate_policy_extra(self) -> None:
        work = {"seed": 522, "workload": 100, "policy": "X"}
        doc = make_cell("E06", work["seed"], work["workload"], work["policy"],
                        override_policy="X")
        self._write_cell(doc, path=self.cell_path("E06", 522, 100, "X"))
        self.expected_keys.append("E06|s522|w100|X")
        self.recompute()

    def mutate_e10_derived_only(self) -> None:
        doc = make_cell("E10", 522, 100, "P", derived_only=False)
        self._write_cell(doc)
        self.recompute()

    def mutate_e10_extra_source(self) -> None:
        doc = make_cell("E10", 522, 100, "P", tamper_source_suffix="E00")
        self._write_cell(doc)
        self.recompute()

    def mutate_e08_duplicate_placeholder(self) -> None:
        doc = make_cell("E08", 522, 100, "B1", policy_independent=True,
                        override_policy="B1")
        self._write_cell(doc, path=self.cell_path("E08", 522, 100, "B1"))
        self.expected_keys.append("E08|s522|w100|B1")
        self.recompute()

    def mutate_drop_pair(self) -> None:
        for experiment in ("E07", "E10"):
            doc = self.cell_docs[f2.cell_key(experiment, 522, 100, "P")]
            path = self.cell_path(experiment, 522, 100, "P")
            path.unlink(missing_ok=True)
            self.expected_keys = [
                key for key in self.expected_keys
                if key != f2.cell_key(experiment, 522, 100, "P")
            ]
        self.recompute()

    def mutate_lock_duplicate_cell(self) -> None:
        """Insert a duplicated governed cell key into the lock semantic list.

        Duplicate positions collapse silently inside dict keys, so the
        FAIL-CLOSED duplicate guard lives at the lock level: the semantic
        list must reconcile exactly with the frozen counts. A duplicated
        key inflates both the list length and the derived-cell count, which
        must be caught by ``verify_v11e_result_lock``.
        """
        lock_path = self.base / f2.V11E_LOCK_NAME
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        semantic = lock["semantic_payload"]
        duplicate = "E10|s522|w100|P"
        occurrences = sum(1 for key in semantic["expected_cells"] if str(key) == duplicate)
        if occurrences < 2:
            semantic["expected_cells"].append(duplicate)
        lock["semantic_result_lock_sha256"] = sha256_of_canonical(semantic)
        lock_path.write_text(f2.canonical_json(lock), encoding="utf-8")
        self.binding = {
            **self.binding,
            "result_lock_file_sha256": f2.sha256_file(lock_path),
            "result_lock_semantic_sha256": lock["semantic_result_lock_sha256"],
        }

    def mutate_summary_mean(self, experiment: str, policy: str, metric: str,
                            delta: float) -> None:
        path = self.summaries / f"{experiment}.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        old = summary["per_policy"][policy]["metrics"][metric]
        summary["per_policy"][policy]["metrics"][metric] = {
            **old, "mean": float(old["mean"]) + delta,
        }
        summary["semantic_sha256"] = sha256_of_canonical(
            {k: v for k, v in summary.items() if k != "semantic_sha256"}
        )
        path.write_text(f2.canonical_json(summary), encoding="utf-8")
        self.recompute()

    def run(self, **kwargs) -> dict:
        return run_v11f_analysis(
            v11e_base_dir=self.base,
            v11e_binding=dict(self.binding),
            **kwargs,
        )


def _cell_scalar_list(cell_docs, field: str) -> list[float]:
    scalars = []
    for doc in cell_docs:
        values = [
            float(record[field])
            for record in (doc.get("result") or {}).get("raw_records", [])
            if isinstance(record.get(field), (int, float))
        ]
        if values:
            scalars.append(sum(values) / len(values))
    return scalars


@pytest.fixture()
def family(tmp_path: Path) -> SyntheticFamily:
    return SyntheticFamily(tmp_path)


def _has_frozen_checkpoint() -> bool:
    return (REAL_V11E_DIR / "v11e_experiment_result_lock.json").exists()


# --------------------------------------------------------------------------- #
# happy paths
# --------------------------------------------------------------------------- #


def test_happy_path_synthetic_payload(family: SyntheticFamily):
    payload = family.run()
    assert payload["marker"] == "V11F2_ANALYSIS_PAYLOAD"
    assert payload["stage"] == "V1.1-F2"
    assert payload["analysis_mode"] == "DESCRIPTIVE_ONLY"
    assert payload["direct_energy_status"] == ENERGY
    assert payload["policy_set"] == ["B0", "B1", "P"]
    assert payload["upstream_result_lock"]["expected_cells"] == 52
    assert payload["upstream_result_lock"]["measured_cells"] == 40
    assert payload["upstream_result_lock"]["e10_derived_cells"] == 12
    governance = payload["governance_checks"]
    assert governance["test_access_requests"] == 0
    assert governance["ai_fit_requests"] == 0
    assert governance["ai_inference_requests"] == 0
    assert governance["new_measurement_attempts"] == 0
    assert governance["results_written"] == 0
    assert governance["energy_terminology_scan"] == "PASS"
    assert governance["memory_proxy_labeling"] == "PASS"
    assert governance["ranking_fields_scan"] == "PASS"
    assert governance["nested_workload_guard"] == "PASS"
    assert governance["v11e_family_fingerprints_verified"]["fingerprints_verified"] > 0
    assert governance["v11e_family_fingerprints_verified"]["expected_cells_without_fingerprints"] == []


def test_happy_path_semantic_hash_matches_recompute(family: SyntheticFamily):
    payload = family.run()
    assert payload["semantic_analysis_sha256"] == sha256_of_canonical(
        semantic_analysis_payload(payload)
    )
    assert "inputs_read" not in semantic_analysis_payload(payload)


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_happy_path_real_frozen_artifacts():
    payload = run_v11f_analysis()
    assert payload["marker"] == "V11F2_ANALYSIS_PAYLOAD"
    assert payload["upstream_result_lock"]["expected_cells"] == 390
    assert payload["upstream_result_lock"]["measured_cells"] == 345
    assert payload["upstream_result_lock"]["e10_derived_cells"] == 45
    assert payload["governance_checks"]["v11e_family_fingerprints_verified"][
        "fingerprints_verified"
    ] == 747
    assert payload["descriptive_summaries"]["per_policy"][
        "persisted_wall_clock_timing"
    ]["B0"]["count"] == 15
    assert list(payload["policy_independent_storage"]["per_workload"]) == [
        "100", "250", "500", "1000", "2500",
    ]


# --------------------------------------------------------------------------- #
# fail-closed verification layers
# --------------------------------------------------------------------------- #


def test_f1_lock_semantic_mismatch(family: SyntheticFamily):
    with pytest.raises(V11FProtocolLockMismatchError):
        family.run(expected_f1_lock_semantic="0" * 64)


def test_f1_config_hash_mismatch(tmp_path: Path, family: SyntheticFamily):
    staged = tmp_path / "bad_config.yaml"
    staged.write_bytes(
        (PROJECT_ROOT / "config" / "resource_efficiency_v11f.yaml").read_bytes()
    )
    staged.write_bytes(staged.read_bytes() + b"\n")
    with pytest.raises(V11FConfigMismatchError):
        run_v11f_analysis(
            v11e_base_dir=family.base,
            v11e_binding=dict(family.binding),
            f1_config_path=staged,
        )


def test_v11e_result_lock_file_mismatch(family: SyntheticFamily):
    binding = dict(family.binding)
    binding["result_lock_file_sha256"] = "0" * 64
    with pytest.raises(V11FResultLockMismatchError):
        run_v11f_analysis(
            v11e_base_dir=family.base, v11e_binding=binding
        )


def test_v11e_result_lock_semantic_stale(family: SyntheticFamily):
    lock_path = family.base / f2.V11E_LOCK_NAME
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["semantic_result_lock_sha256"] = sha256_of_canonical(
        {**lock["semantic_payload"], "counts": {"expected_total": 0}}
    )
    lock_path.write_text(f2.canonical_json(lock), encoding="utf-8")
    binding = dict(family.binding)
    binding["result_lock_file_sha256"] = f2.sha256_file(lock_path)
    with pytest.raises(V11FResultLockMismatchError):
        run_v11f_analysis(v11e_base_dir=family.base, v11e_binding=binding)


def test_missing_result_lock(tmp_path: Path):
    base = tmp_path / "absent"
    base.mkdir()
    binding = {"result_lock_file_sha256": "0" * 64}
    with pytest.raises(V11FMissingArtifactError):
        run_v11f_analysis(v11e_base_dir=base, v11e_binding=binding)


def test_missing_cell_fingerprint(family: SyntheticFamily):
    family.cell_path("E06", 522, 100, "B0").unlink()
    family.recompute()
    with pytest.raises(V11FFingerprintMismatchError):
        family.run()
    family.cell_path("E06", 522, 100, "B0").write_text(
        f2.canonical_json(make_cell("E06", 522, 100, "B0")), encoding="utf-8"
    )
    family.recompute()
    assert family.run()["stage"] == "V1.1-F2"


def test_fingerprint_mismatch(family: SyntheticFamily):
    path = family.cell_path("E07", 523, 250, "B1")
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(V11FFingerprintMismatchError):
        family.run()


def test_policy_mismatch(family: SyntheticFamily):
    family.mutate_policy_extra()
    with pytest.raises(V11FPolicyError):
        family.run()


def test_e10_derived_only_violation(family: SyntheticFamily):
    family.mutate_e10_derived_only()
    with pytest.raises(V11FE10NotDerivedError):
        family.run()


def test_e10_unexpected_source(family: SyntheticFamily):
    family.mutate_e10_extra_source()
    with pytest.raises(V11FE10SourceError):
        family.run()


def test_missing_attack_manifest_fails_closed(family: SyntheticFamily):
    (family.base / f2.V11E_ATTACK_MANIFEST_NAME).unlink()
    with pytest.raises(V11FMissingArtifactError):
        family.run()


def test_missing_pair_fail_closed():
    by_policy = {
        "B0": {(522, 100): 1.0, (523, 100): 2.0},
        "B1": {(522, 100): 1.5, (523, 100): 3.0},
        "P": {(522, 100): 2.0},
    }
    with pytest.raises(V11FUnmatchedPairError):
        f2._paired_differences(by_policy, other="P", base="B0")


def test_duplicate_pair_fail_closed_lock(family: SyntheticFamily):
    # duplicate governed positions cannot hide behind dict key collision at
    # the lock level: an inflated semantic cell list fails closed.
    family.mutate_lock_duplicate_cell()
    with pytest.raises(V11FResultLockMismatchError):
        family.run()


def test_same_workload_different_seeds_not_duplicates():
    # governed pairing identity is (seed, workload): the same workload value
    # under distinct seeds is legitimate evidence, never a duplicate pair.
    evidence = {
        "persisted_wall_clock_timing": {
            "B0": {(522, 100): 1.0, (523, 100): 2.0, (522, 250): 3.0},
            "B1": {(522, 100): 1.5, (523, 100): 3.0, (522, 250): 4.0},
            "P": {(522, 100): 2.0, (523, 100): 4.0, (522, 250): 5.0},
        }
    }
    out = f2.build_paired_descriptive_comparisons(evidence)
    assert out["fail_closed"]["duplicate_pairs"] == 0
    per_workload = out["per_workload_per_contrast"]["persisted_wall_clock_timing"]
    # workload 100 spans two seeds -> two paired rows, one workload group
    assert per_workload["P_minus_B0"][100]["count"] == 2
    assert per_workload["P_minus_B0"][250]["count"] == 1
    seeds = {
        (row["seed"], row["workload"])
        for row in out["pair_cells"]["persisted_wall_clock_timing"]["P_minus_B0"]
    }
    assert seeds == {(522, 100), (523, 100), (522, 250)}


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_duplicate_pairs_zero_real_frozen_evidence():
    payload = run_v11f_analysis()
    fail_closed = payload["paired_descriptive_comparisons"]["fail_closed"]
    assert fail_closed["duplicate_pairs"] == 0
    assert fail_closed["missing_pairs"] == 0


def test_duplicate_storage_placeholder_fail_closed(family: SyntheticFamily):
    family.mutate_e08_duplicate_placeholder()
    with pytest.raises(V11FUnmatchedPairError):
        family.run()


def test_missing_pair_in_pipeline(family: SyntheticFamily):
    family.mutate_drop_pair()
    with pytest.raises(V11FUnmatchedPairError):
        family.run()


def test_zero_denominator_ratio_skipped():
    evidence = {
        "persisted_wall_clock_timing": {
            "B0": {(522, 100): 0.0, (523, 100): 10.0},
            "B1": {(522, 100): 5.0, (523, 100): 8.0},
            "P": {(522, 100): 2.0, (523, 100): 12.0},
        }
    }
    ratios = f2.build_normalized_ratios(evidence)
    skipped = ratios["skipped_zero_denominator"]
    assert any(
        row["ratio"] == "P_over_B0" and row["seed"] == 522
        and row["skipped"] == "non_positive_denominator"
        for row in skipped
    )
    p_over_b0 = ratios["ratios"]["persisted_wall_clock_timing"]["P_over_B0"]
    assert any(row["seed"] == 523 and row["value"] == 12.0 / 10.0 for row in p_over_b0)


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_real_upstream_immutability():
    before = {
        "f1_lock": f2.sha256_file(F1_LOCK_PATH),
        "f1_config": f2.sha256_file(F1_CONFIG_PATH),
        "f1_protocol": f2.sha256_file(F1_PROTOCOL_PATH),
        "e_lock": f2.sha256_file(REAL_V11E_DIR / "v11e_experiment_result_lock.json"),
    }
    run_v11f_analysis()
    after = {
        "f1_lock": f2.sha256_file(F1_LOCK_PATH),
        "f1_config": f2.sha256_file(F1_CONFIG_PATH),
        "f1_protocol": f2.sha256_file(F1_PROTOCOL_PATH),
        "e_lock": f2.sha256_file(REAL_V11E_DIR / "v11e_experiment_result_lock.json"),
    }
    assert before == after


def test_cross_check_summary_drift(family: SyntheticFamily):
    family.mutate_summary_mean("E06", "B0", "wall_time_ns", 1000.0)
    with pytest.raises(V11FIntegrityError):
        family.run()


# --------------------------------------------------------------------------- #
# descriptive statistics
# --------------------------------------------------------------------------- #


def test_describe_values_full_schema():
    desc = describe_values([1.0, 2.0, 3.0, 4.0])
    assert desc["count"] == 4
    assert desc["mean"] == 2.5
    assert desc["median"] == 2.5
    assert desc["sample_standard_deviation_ddof_1"] == pytest.approx(
        ((1.5 ** 2 + 0.5 ** 2 + 0.5 ** 2 + 1.5 ** 2) / 3) ** 0.5
    )
    assert desc["min"] == 1.0
    assert desc["max"] == 4.0
    assert desc["coefficient_of_variation"] == pytest.approx(
        desc["sample_standard_deviation_ddof_1"] / 2.5
    )


def test_describe_values_zero_mean_cv_is_none():
    desc = describe_values([0.0, 0.0, 0.0])
    assert desc["mean"] == 0.0
    assert desc["coefficient_of_variation"] is None


def test_no_inferential_fields(family: SyntheticFamily):
    payload = family.run()
    summaries = payload["descriptive_summaries"]["per_policy"]
    for metric, by_policy in summaries.items():
        for policy, row in by_policy.items():
            assert "p_value" not in row
            assert "confidence_interval" not in row
            assert "significance" not in row


def test_storage_is_policy_independent(family: SyntheticFamily):
    payload = family.run()
    storage = payload["policy_independent_storage"]
    assert storage["status"] == "POLICY_INDEPENDENT"
    assert storage["policy_token"] == "B0"
    assert storage["storage_policy_differences_allowed"] is False
    assert set(storage["per_workload"]) == {"100", "250"}
    assert sum(1 for metric in payload["normalized_ratios"]["ratios"]
               if metric == "persisted_canonical_serialized_bytes") == 0
    assert "persisted_canonical_serialized_bytes" not in payload[
        "paired_descriptive_comparisons"
    ]["per_contrast"]
    assert set(payload["paired_descriptive_comparisons"]["per_contrast"]) == {
        "persisted_wall_clock_timing",
        "persisted_cpu_process_timing",
        "persisted_tracemalloc_peak_proxy",
        "persisted_validated_orders_per_second",
        "validation_check_count",
        "validator_invocation_count",
        "measurable_hash_operation_count",
    }


def test_contradit_only_allowed_contrasts(family: SyntheticFamily):
    payload = family.run()
    for metric, contrasts in payload["paired_descriptive_comparisons"][
        "per_contrast"
    ].items():
        assert set(contrasts) == {"B1_minus_B0", "P_minus_B0", "P_minus_B1"}


def test_ratio_metadata(family: SyntheticFamily):
    payload = family.run()
    rows = payload["normalized_ratios"]["ratios"][
        "persisted_validated_orders_per_second"
    ]["P_over_B0"]
    assert rows
    row = rows[0]
    assert row["numerator_policy"] == "P"
    assert row["denominator_policy"] == "B0"
    assert row["metric"] == "persisted_validated_orders_per_second"
    assert row["family"] == "throughput"
    assert "P" in row["direction"] and "B0" in row["direction"]
    assert row["improvement_labeled"] is False
    assert row["denominator"] > 0


def test_direction_metadata(family: SyntheticFamily):
    payload = family.run()
    memory = payload["metric_definitions"]["families"]["memory"]
    assert memory["canonical_metric"] == "persisted_tracemalloc_peak_proxy"
    assert memory["direction"] == "lower_is_lighter"
    assert memory["memory_label"] == "COMPUTATIONAL_MEMORY_PROXY"


# --------------------------------------------------------------------------- #
# governance guards
# --------------------------------------------------------------------------- #


def test_test_access_forbidden():
    with pytest.raises(V11FTestAccessError):
        guard_test_access(request="test_split/features.csv")
    with pytest.raises(V11FTestAccessError):
        f2.assert_no_forbidden_test_paths(["results/raw_test_rows.json"])
    # the F2 operation mode requests no TEST access: the guard must accept
    # the absent request so the orchestrator can validate mode explicitly
    assert guard_test_access(request=None) is None


def test_ai_access_forbidden():
    with pytest.raises(V11FAIAccessError):
        guard_ai_fit(requested=True)
    with pytest.raises(V11FAIAccessError):
        guard_ai_inference(requested=True)
    # F2 requests neither capability
    assert guard_ai_fit(requested=False) is None
    assert guard_ai_inference(requested=False) is None


def test_new_measurement_forbidden():
    with pytest.raises(V11FMeasurementError):
        guard_new_measurement(kind="tracemalloc")
    assert guard_new_measurement(kind=None) is None


def test_validate_f2_operation_mode_passes():
    # the analysis mode requests TEST_ACCESS=none, AI_FIT/INFERENCE=FALSE and
    # NEW_MEASUREMENT=FALSE; requesting any capability raises (guard tests
    # above) while the F2-valid mode passes.
    f2.validate_f2_operation_mode()


def test_governance_zero_counters(family: SyntheticFamily):
    payload = family.run()
    governance = payload["governance_checks"]
    assert governance["test_access_requests"] == 0
    assert governance["ai_fit_requests"] == 0
    assert governance["ai_inference_requests"] == 0
    assert governance["new_measurement_attempts"] == 0
    assert governance["results_written"] == 0


def test_orchestrator_wires_governance_guards(family: SyntheticFamily, monkeypatch):
    # prove the orchestrator itself validates the operation mode through the
    # guards: a future edit that drops the wiring is caught by this test.
    calls: dict[str, dict] = {}

    def record(name):
        def wrapper(**kwargs):
            calls[name] = kwargs
        return wrapper

    monkeypatch.setattr(f2, "guard_test_access", record("guard_test_access"))
    monkeypatch.setattr(f2, "guard_ai_fit", record("guard_ai_fit"))
    monkeypatch.setattr(f2, "guard_ai_inference", record("guard_ai_inference"))
    monkeypatch.setattr(f2, "guard_new_measurement", record("guard_new_measurement"))
    family.run()
    assert calls == {
        "guard_test_access": {"request": None},
        "guard_ai_fit": {"requested": False},
        "guard_ai_inference": {"requested": False},
        "guard_new_measurement": {"kind": None},
    }


def test_environment_drift_fails_closed(family: SyntheticFamily):
    # matched pairing requires a single environment identity per experiment;
    # a cell drifted onto different hardware must not be pooled silently
    key = f2.cell_key("E06", 522, 100, "B0")
    doc = dict(family.cell_docs[key])
    doc["environment"] = {**doc["environment"], "machine_environment": "tampered-host"}
    family.cell_path("E06", 522, 100, "B0").write_text(
        f2.canonical_json(doc), encoding="utf-8"
    )
    family.cell_docs[key] = doc
    family.recompute()
    with pytest.raises(V11FUnmatchedPairError):
        family.run()


def test_approved_scope_only_reads(family: SyntheticFamily):
    payload = family.run()
    governance = payload["governance_checks"]
    scan = list(governance["inputs_read"])
    f2.assert_no_forbidden_test_paths(scan)
    for path in scan:
        resolved = Path(path).resolve()
        roots = (
            family.base,
            F1_LOCK_PATH,
            F1_CONFIG_PATH,
            F1_PROTOCOL_PATH,
        )
        assert any(
            resolved == root.resolve() or root.resolve() in resolved.parents
            for root in roots
        ), resolved


def test_energy_terminology_guard():
    with pytest.raises(V11FEnergyClaimError):
        f2.assert_no_direct_energy_fields(
            {"descriptive_summaries": {"note": "energy saved by 30 percent"}}
        )
    with pytest.raises(V11FEnergyClaimError):
        f2.assert_no_direct_energy_fields(
            {"normalized_ratios": {"note": "P reduces Joules versus B0"}}
    )


def test_memory_proxy_labeling_guard():
    good = {"metric_definitions": {"memory": {"canonical_metric": "persisted_tracemalloc_peak_proxy"}}}
    f2.assert_memory_proxy_labeling(good)
    for bad_label in ("RSS", "system_memory", "electrical_energy"):
        bad = {"metric_definitions": {"memory": {"canonical_metric": bad_label}}}
        with pytest.raises(V11FEnergyClaimError):
            f2.assert_memory_proxy_labeling(bad)


def test_e07_scientific_labeling_pinned(family: SyntheticFamily):
    payload = family.run()
    memory = payload["metric_definitions"]["families"]["memory"]
    assert memory["memory_label"] == "COMPUTATIONAL_MEMORY_PROXY"
    assert memory["canonical_metric"] == "persisted_tracemalloc_peak_proxy"
    assert memory["source_experiment"] == "E07"
    assert memory["energy_basis"] == ENERGY
    assert "computational resource proxy" in memory["interpretation"]
    # RSS / direct-energy memory cannot be emitted as a metric anywhere in
    # the analytical tables; only the proxy metric key may appear
    per_policy = payload["descriptive_summaries"]["per_policy"]
    paired = payload["paired_descriptive_comparisons"]
    ratios = payload["normalized_ratios"]
    assert "persisted_tracemalloc_peak_proxy" in per_policy
    assert list(paired["per_contrast"][
        "persisted_tracemalloc_peak_proxy"
    ].keys()) == ["B1_minus_B0", "P_minus_B0", "P_minus_B1"]
    assert "persisted_tracemalloc_peak_proxy" in ratios["ratios"]
    for bad in ("RSS", "system_memory", "electrical_energy"):
        assert bad not in f2.canonical_json(
            {
                "definitions": memory,
                "per_policy": per_policy,
                "paired": paired,
                "ratios": ratios,
            }
        )


def test_no_ranking_guard():
    f2.assert_no_ranking_fields({"descriptive_summaries": {"mean": 1.0}})
    with pytest.raises(V11FRankingError):
        f2.assert_no_ranking_fields({"overall_winner": "P"})
    with pytest.raises(V11FRankingError):
        f2.assert_no_ranking_fields({"descriptive_summaries": {"best_policy": "B1"}})


def test_pipeline_has_no_ranking_fields(family: SyntheticFamily):
    payload = family.run()
    f2.assert_no_ranking_fields(payload)
    assert "overall_score" not in canonical_json(payload)


def test_nested_workload_guard():
    good = {
        "nested_workload_limitation": {
            "independent_replicates": False,
            "seed_workload_cells_may_be_treated_as_15_independent_replicates": False,
            "nesting": "100 subset 250 subset 500 subset 1000 subset 2500 within each seed",
        }
    }
    f2.assert_nested_workload_safe(good)
    bad = copy.deepcopy(good)
    bad["nested_workload_limitation"]["independent_replicates"] = True
    with pytest.raises(V11FIntegrityError):
        f2.assert_nested_workload_safe(bad)
    bad2 = copy.deepcopy(good)
    bad2["nested_workload_limitation"][
        "seed_workload_cells_may_be_treated_as_15_independent_replicates"
    ] = True
    with pytest.raises(V11FIntegrityError):
        f2.assert_nested_workload_safe(bad2)


def test_pipeline_nested_workload_limitation(family: SyntheticFamily):
    payload = family.run()
    nesting = payload["nested_workload_limitation"]
    assert nesting["independent_replicates"] is False
    assert nesting["seed_workload_cells_may_be_treated_as_15_independent_replicates"] is False
    assert "_independent_replicates_claim" not in canonical_json(payload)


# --------------------------------------------------------------------------- #
# determinism + no-writes
# --------------------------------------------------------------------------- #


def test_deterministic_payload_and_hash(family: SyntheticFamily):
    first = family.run()
    second = family.run()
    assert canonical_json(semantic_analysis_payload(first)) == canonical_json(
        semantic_analysis_payload(second)
    )
    assert canonical_json(first) == canonical_json(second)
    assert first["semantic_analysis_sha256"] == second["semantic_analysis_sha256"]


def test_no_writes(family: SyntheticFamily):
    before = {p: p.stat().st_mtime_ns for p in family.base.rglob("*") if p.is_file()}
    family.run()
    after = {p: p.stat().st_mtime_ns for p in family.base.rglob("*") if p.is_file()}
    assert set(before) == set(after)
    assert before == after


def test_scope_rejects_outside_path():
    scope = f2.EvidenceScope(
        v11e_base_dir=Path("/tmp/opencode/v11f_test_scope_xyz/v11e"),
        f1_lock_path=F1_LOCK_PATH,
        f1_config_path=F1_CONFIG_PATH,
        f1_protocol_path=F1_PROTOCOL_PATH,
    )
    with pytest.raises(V11FIntegrityError):
        scope.assert_approved_path(Path("/etc/passwd"))