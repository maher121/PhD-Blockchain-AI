"""Focused infrastructure tests for V0.8-E2 resource benchmarking."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import src.pipeline_v08e as v08e


@pytest.fixture(scope="module")
def real_preconditions():
    return v08e.verify_v08e_prerequisites()


def _synthetic_observation(
    *,
    observation_id: str,
    classifier: str = "decision_tree",
    configuration_id: str = "K43",
    source_configuration_id: str = "none_natural",
    seed: int = 42,
    phase: str = "inference",
    repetition: int = 1,
    inner: int = 512,
    status: str = "SUCCESS",
) -> dict[str, object]:
    success = status == "SUCCESS"
    return {
        "observation_id": observation_id,
        "schema_version": v08e.V08E_SCHEMA_VERSION,
        "stage": v08e.V08E_STAGE,
        "environment_id": "env",
        "classifier": classifier,
        "configuration_id": configuration_id,
        "source_configuration_id": source_configuration_id,
        "seed": seed,
        "phase": phase,
        "outer_repetition": repetition,
        "inner_operation_count": inner,
        "measured_operation_count": inner if success else 0,
        "record_count": 6000 if phase == "inference" else 28000,
        "measured_record_count": (6000 if phase == "inference" else 28000) * (inner if success else 0),
        "feature_count": 43,
        "warmup_calls": v08e.WARMUP_CALLS,
        "worker_pid": 1234,
        "wall_time_sec": 1.0 if success else None,
        "process_cpu_time_sec": 0.9 if success else None,
        "start_rss_mib": 100.0 if success else None,
        "end_rss_mib": 101.0 if success else None,
        "absolute_peak_rss_mib": 102.0 if success else None,
        "incremental_peak_rss_mib": 2.0 if success else None,
        "input_dataframe_memory_bytes": 10000,
        "numpy_dense_nbytes": 9000,
        "serialized_model_bytes": 4096,
        "inference_latency_sec": (1.0 / inner) if success else None,
        "per_record_latency_sec": (1.0 / (inner * (6000 if phase == "inference" else 28000))) if success else None,
        "throughput_records_sec": (inner * (6000 if phase == "inference" else 28000)) if success else None,
        "time_unit": "seconds",
        "rss_unit": "MiB",
        "size_unit": "bytes",
        "throughput_unit": "records/second",
        "measurement_scope": (
            "prediction_only_model_loading_warmup_excluded"
            if phase == "inference"
            else "model.fit_only_data_loading_selection_serialization_excluded"
        ),
        "measurement_provenance": "DIRECT_COMPUTATIONAL",
        "derived_measurement_provenance": "DERIVED",
        "workload_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "configuration_features_sha256": "c" * 64,
        "v06_lock_sha256": "d" * 64,
        "v08c_semantic_lock_sha256": v08e.EXPECTED_K10["semantic_lock_sha256"],
        "v08d_semantic_result_lock_sha256": v08e.EXPECTED_V08D_RESULT_LOCK_SHA256,
        "status": status,
        "failure_type": None if success else "RuntimeError",
        "failure_stage": None if success else "measurement",
        "failure_reason": None if success else "synthetic failure",
    }


def _synthetic_checkpoint_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        _synthetic_observation(observation_id=f"obs-{index}", repetition=index)
        for index in range(1, 11)
    )


def test_01_exactly_four_configurations_locked() -> None:
    assert v08e.CONFIGURATION_IDS == ("K43", "K42", "K11", "BPSO-K10")


def test_02_exactly_two_classifiers_locked() -> None:
    assert v08e.CLASSIFIERS == ("decision_tree", "logistic_regression")


def test_03_exactly_five_seeds_locked() -> None:
    assert v08e.SEEDS == (42, 43, 44, 45, 46)


def test_04_exactly_two_phases_locked() -> None:
    assert v08e.PHASES == ("training", "inference")


def test_05_exactly_ten_repetitions_locked() -> None:
    assert v08e.OUTER_REPETITIONS == 10
    assert v08e.REPETITIONS == tuple(range(1, 11))


def test_06_manifest_expected_total_is_800(real_preconditions) -> None:
    manifest = v08e.build_run_manifest(real_preconditions, "env")
    assert manifest["expected_total_observations"] == 800
    v08e.validate_run_manifest(manifest)


def test_07_k10_identity_lock_is_exact(real_preconditions) -> None:
    k10 = real_preconditions["k10_identity"]
    assert k10["feature_count"] == 10
    assert k10["mask_sha256"] == v08e.EXPECTED_K10["mask_sha256"]
    assert k10["selected_features_sha256"] == v08e.EXPECTED_K10["selected_features_sha256"]


def test_08_k42_and_k43_identities_present(real_preconditions) -> None:
    identities = real_preconditions["configuration_identity"]
    assert identities["k43"]["feature_count"] == 43
    assert identities["k42"]["feature_count"] == 42
    assert len(identities["k42"]["mask_sha256"]) == 64


def test_09_k11_remains_seed_specific(real_preconditions) -> None:
    hashes = real_preconditions["configuration_identity"]["k11"]["seed_feature_hashes"]
    assert set(hashes) == {"42", "43", "44", "45", "46"}
    assert len(set(hashes.values())) > 1


def test_10_fresh_worker_contract_is_used() -> None:
    source = inspect.getsource(v08e._run_cell_measurement)
    assert "run_fresh_worker_protocol" in source
    assert "outer_repetitions=OUTER_REPETITIONS" in source


def test_11_warmup_locked_to_three() -> None:
    assert v08e.WARMUP_CALLS == 3


def test_12_inference_inner_operations_locked_to_512() -> None:
    assert v08e.INFERENCE_INNER_OPERATION_COUNT == 512


def test_13_training_inner_operations_locked_to_one() -> None:
    assert v08e.TRAINING_INNER_OPERATION_COUNT == 1


def test_14_measurement_boundaries_are_locked() -> None:
    config = v08e.load_v08e_protocol()
    measurement = config["measurement"]
    assert measurement["training_boundary"] == "model.fit_only_data_loading_selection_serialization_excluded"
    assert measurement["inference_boundary"] == "prediction_only_model_loading_warmup_excluded"


def test_15_checkpoint_roundtrip_resume(tmp_path: Path) -> None:
    rows = _synthetic_checkpoint_rows()
    cell = v08e.BenchmarkCell("decision_tree", "K43", "none_natural", 42, "inference", 512)
    path = tmp_path / "checkpoint.json"
    v08e.write_cell_checkpoint(
        path,
        cell=cell,
        protocol_hash="p" * 64,
        environment_id="env",
        rows=rows,
    )
    loaded = v08e.load_valid_checkpoint(
        path,
        cell=cell,
        protocol_hash="p" * 64,
        environment_id="env",
    )
    assert loaded == rows


def test_16_checkpoint_environment_mismatch_rejected(tmp_path: Path) -> None:
    rows = _synthetic_checkpoint_rows()
    cell = v08e.BenchmarkCell("decision_tree", "K43", "none_natural", 42, "inference", 512)
    path = tmp_path / "checkpoint.json"
    v08e.write_cell_checkpoint(
        path,
        cell=cell,
        protocol_hash="p" * 64,
        environment_id="env-a",
        rows=rows,
    )
    with pytest.raises(v08e.V08ECheckpointError):
        v08e.load_valid_checkpoint(
            path,
            cell=cell,
            protocol_hash="p" * 64,
            environment_id="env-b",
        )


def test_17_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    cell = v08e.BenchmarkCell("decision_tree", "K43", "none_natural", 42, "inference", 512)
    payload = {
        "stage": v08e.V08E_STAGE,
        "schema_version": v08e.V08E_SCHEMA_VERSION,
        "protocol_sha256": "p" * 64,
        "environment_id": "env",
        "expected_repetition_count": 10,
        "cell_identity": {
            "classifier": "decision_tree",
            "configuration_id": "K43",
            "seed": 42,
            "phase": "inference",
        },
        "rows": list(_synthetic_checkpoint_rows()[:-1]),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(v08e.V08ECheckpointError, match="complete"):
        v08e.load_valid_checkpoint(
            path,
            cell=cell,
            protocol_hash="p" * 64,
            environment_id="env",
        )


def test_18_observation_schema_and_uniqueness_enforced() -> None:
    rows = [_synthetic_observation(observation_id="dup"), _synthetic_observation(observation_id="dup")]
    with pytest.raises(v08e.V08EIntegrityError, match="unique"):
        v08e.validate_observation_rows(rows)


def test_19_provenance_labels_are_locked() -> None:
    labels = {
        "DIRECT_COMPUTATIONAL",
        "DERIVED",
        "IMPORTED_HISTORICAL",
        "DIRECT_ENERGY",
        "ESTIMATED",
    }
    manifest = v08e.build_run_manifest(v08e.verify_v08e_prerequisites(), "env")
    assert set(manifest["provenance_labels"]) == labels


def test_20_inference_workload_type_is_feature_only() -> None:
    fields = set(v08e.InferenceFeatureWorkload.__dataclass_fields__)
    assert "features" in fields
    assert "labels" not in fields
    assert "ground_truth" not in fields


def test_21_pipeline_source_does_not_call_predictive_metric_functions() -> None:
    source = inspect.getsource(v08e)
    forbidden = (
        "average_precision_score",
        "f1_score",
        "recall_score",
        "precision_score",
        "roc_auc_score",
        "confusion_matrix",
        "evaluate_detection",
    )
    assert all(marker not in source for marker in forbidden)


def test_22_pipeline_source_does_not_invoke_bpso() -> None:
    source = inspect.getsource(v08e)
    assert "BinaryParticleSwarmOptimizer" not in source
    assert "feature_fitness_is_better" not in source


def test_23_imported_optimizer_overhead_is_historical() -> None:
    imported = v08e.import_v08c_optimizer_overhead()
    assert imported["measurement_provenance"] == "IMPORTED_HISTORICAL"
    assert imported["optimizer_runs"] == 5
    assert imported["fitness_requests"] == 972
    assert imported["decision_tree_fits"] == 4860


def test_24_imported_predictive_evidence_is_historical() -> None:
    imported = v08e.import_v08d_predictive_evidence()
    assert imported["measurement_provenance"] == "IMPORTED_HISTORICAL"
    assert imported["source_semantic_result_lock_sha256"] == v08e.EXPECTED_V08D_RESULT_LOCK_SHA256
    assert imported["summaries"]


def test_25_direct_energy_unavailable_is_preserved() -> None:
    capability = v08e.resolve_energy_capability()
    assert capability["decision"] == "DIRECT_ENERGY_UNAVAILABLE"


def test_26_no_cpu_to_energy_conversion_utility_exists() -> None:
    source = inspect.getsource(v08e)
    assert "cpu_to_energy" not in source
    assert "wall_to_energy" not in source


def test_27_within_seed_aggregation_before_seed_pairing() -> None:
    rows = []
    for rep in range(1, 11):
        for classifier in v08e.CLASSIFIERS:
            rows.append(
                _synthetic_observation(
                    observation_id=f"{classifier}-k10-inf-{rep}",
                    classifier=classifier,
                    configuration_id="BPSO-K10",
                    source_configuration_id="v08c_locked_bpso_k10",
                    phase="inference",
                    repetition=rep,
                )
            )
            rows.append(
                _synthetic_observation(
                    observation_id=f"{classifier}-k43-inf-{rep}",
                    classifier=classifier,
                    configuration_id="K43",
                    source_configuration_id="none_natural",
                    phase="inference",
                    repetition=rep,
                )
            )
            rows.append(
                _synthetic_observation(
                    observation_id=f"{classifier}-k10-train-{rep}",
                    classifier=classifier,
                    configuration_id="BPSO-K10",
                    source_configuration_id="v08c_locked_bpso_k10",
                    phase="training",
                    inner=1,
                    repetition=rep,
                )
            )
            rows.append(
                _synthetic_observation(
                    observation_id=f"{classifier}-k43-train-{rep}",
                    classifier=classifier,
                    configuration_id="K43",
                    source_configuration_id="none_natural",
                    phase="training",
                    inner=1,
                    repetition=rep,
                )
            )
    summaries = v08e.summarize_within_seed(rows)
    assert all(row["repetition_count"] == 10 for row in summaries)
    paired = v08e.paired_seed_differences(
        [
            {**row, "seed": seed}
            for seed in v08e.SEEDS
            for row in summaries
        ],
        baseline_ids=("K43",),
    )
    assert all(row["n"] == 5 and row["df"] == 4 for row in paired)


def test_28_paired_t95_ci_formula_is_correct() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = v08e.paired_t95_interval(values)
    expected_margin = v08e.T_CRITICAL_95_DF4 * (1.5811388300841898) / (5 ** 0.5)
    assert low == pytest.approx(3.0 - expected_margin)
    assert high == pytest.approx(3.0 + expected_margin)


def test_29_break_even_positive_case() -> None:
    result = v08e.break_even_repetitions(
        overhead_value=100.0,
        recurring_saving=5.0,
        overhead_unit="seconds",
        saving_unit="seconds",
    )
    assert result["status"] == "APPLICABLE"
    assert result["break_even_repetitions"] == pytest.approx(20.0)


def test_30_break_even_zero_case() -> None:
    result = v08e.break_even_repetitions(
        overhead_value=100.0,
        recurring_saving=0.0,
        overhead_unit="seconds",
        saving_unit="seconds",
    )
    assert result["status"] == "NOT_APPLICABLE"
    assert result["break_even_repetitions"] is None


def test_31_break_even_negative_case() -> None:
    result = v08e.break_even_repetitions(
        overhead_value=100.0,
        recurring_saving=-1.0,
        overhead_unit="seconds",
        saving_unit="seconds",
    )
    assert result["status"] == "NOT_APPLICABLE"


def test_32_break_even_unit_mismatch_rejected() -> None:
    with pytest.raises(v08e.V08EIntegrityError, match="matching units"):
        v08e.break_even_repetitions(
            overhead_value=100.0,
            recurring_saving=5.0,
            overhead_unit="seconds",
            saving_unit="cpu-seconds",
        )


def test_33_protocol_hash_uses_canonical_lf(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes(b"a: 1\r\nb: 2\r\n")
    first = v08e.canonical_lf_sha256(path)
    path.write_bytes(b"a: 1\nb: 2\n")
    second = v08e.canonical_lf_sha256(path)
    assert first == second


def test_34_frozen_integrity_snapshot_contains_v08d_lock() -> None:
    snapshot = v08e.snapshot_immutable_paths()
    joined = "\n".join(snapshot)
    assert "v08d_final_test_lock.json" in joined
    assert "v08c_winner_lock.json" in joined
    assert "v07d_artifact_hashes.json" in joined


def test_35_build_schedule_is_exact_800() -> None:
    schedule = v08e.build_observation_schedule()
    assert len(schedule) == 800
    assert sum(row.phase == "training" for row in schedule) == 400
    assert sum(row.phase == "inference" for row in schedule) == 400


def test_36_schedule_cells_is_exact_80() -> None:
    cells = v08e.schedule_cells(v08e.build_observation_schedule())
    assert len(cells) == 80


def test_37_run_defaults_to_infrastructure_only(tmp_path: Path) -> None:
    result = v08e.run_v08e(output_dir=tmp_path, checkpoint_dir=tmp_path / ".cp", model_dir=tmp_path / "models")
    assert result["status"] == "INFRASTRUCTURE_READY"
    assert result["execute_matrix"] is False
    assert result["expected_total_observations"] == 800


def test_38_engineering_smoke_marker_is_isolated(tmp_path: Path) -> None:
    result = v08e.run_engineering_smoke_only(output_dir=tmp_path)
    assert result["label"] == "ENGINEERING_SMOKE_ONLY"
    assert result["merged_into_scientific_observations"] is False


def test_39_config_file_is_frozen() -> None:
    config = v08e.load_v08e_protocol()
    assert tuple(config["configurations"]) == v08e.CONFIGURATION_IDS
    assert tuple(config["classifiers"]) == v08e.CLASSIFIERS
    assert tuple(config["seeds"]) == v08e.SEEDS


def test_40_observation_fields_include_required_resource_metrics() -> None:
    required = {
        "wall_time_sec",
        "process_cpu_time_sec",
        "inference_latency_sec",
        "throughput_records_sec",
        "absolute_peak_rss_mib",
        "incremental_peak_rss_mib",
        "input_dataframe_memory_bytes",
        "numpy_dense_nbytes",
        "serialized_model_bytes",
        "feature_count",
    }
    assert required.issubset(set(v08e.OBSERVATION_FIELDS))


def test_41_stdin_launcher_path_breaks_fresh_worker_start() -> None:
    code = textwrap.dedent(
        """
        import json
        import multiprocessing as mp
        from src.green.measurement import MeasurementPhase, run_fresh_worker_protocol

        def prepare_state():
            return [0]

        def step(state):
            state[0] += 1
            return state[0]

        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        run = run_fresh_worker_protocol(
            step,
            prepare=prepare_state,
            inner_operation_count=1,
            phase=MeasurementPhase.INFERENCE.value,
            warmup_calls=0,
            outer_repetitions=1,
            records_per_operation=1,
            timeout_seconds=15.0,
            measurement_scope="stdin_worker_start_regression",
            start_method=method,
        )
        result = run.results[0]
        payload = {
            "status": result.status.value,
            "failure_type": None if result.failure is None else result.failure.error_type,
            "failure_stage": None if result.failure is None else result.failure.stage,
            "failure_message": None if result.failure is None else result.failure.message,
        }
        print(json.dumps(payload, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-"],
        input=code,
        text=True,
        capture_output=True,
        cwd=str(v08e.PROJECT_ROOT),
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] != "COMPLETED"
    assert payload["failure_type"] is not None
    assert payload["failure_stage"] in {"worker_start", "result_transport", "worker_timeout"}
    assert "<stdin>" in completed.stderr or "Broken pipe" in str(payload["failure_message"])


def test_42_real_file_launcher_smoke_starts_worker(tmp_path: Path) -> None:
    launcher = v08e.PROJECT_ROOT / "scripts" / "run_v08e_e3.py"
    assert launcher.is_file()
    completed = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--worker-smoke",
            "--smoke-output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        cwd=str(v08e.PROJECT_ROOT),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "COMPLETED"
    assert isinstance(payload["worker_pid"], int) and payload["worker_pid"] > 0
    assert payload["warmup_calls_completed"] == 3
    assert payload["measured_operation_count"] == 4
    assert payload["failure_type"] is None
    assert (tmp_path / "v08e_e3r_worker_smoke.json").is_file()
