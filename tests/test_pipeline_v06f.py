"""Synthetic-only tests for the V0.6-F locked final-test gate and evaluator."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.ai.model_utils import ProcessedSplit
from src.lightweight.models import LightweightDetector
from src.lightweight.resource_monitor import ResourceMeasurement
import src.pipeline_v06f as v06f
from src.pipeline_v06e import ValidationLockError, fingerprint_feature_names, semantic_payload_sha256
from src.security.experiment_data import (
    PreparedAttackSplit,
    fingerprint_attack_labels,
    fingerprint_feature_matrix,
    fingerprint_frame,
    fingerprint_mapping,
    fingerprint_row_ids,
)


FEATURES = ["feature_a", "feature_b"]
PARAMETERS = {"max_depth": 5, "min_samples_leaf": 20, "class_weight": "balanced"}


@pytest.fixture()
def signed_lock(tmp_path: Path) -> Path:
    results = tmp_path / "feature_selection"
    results.mkdir()
    attack_config = results / "synthetic.yaml"
    attack_config.write_text("version: synthetic\n", encoding="utf-8")
    (results / "run_metadata.json").write_text(
        json.dumps(
            {
                "attack_config_sha256": hashlib.sha256(
                    attack_config.read_bytes()
                ).hexdigest(),
                "processed_dataset_metadata_sha256": "1" * 64,
            }
        ),
        encoding="utf-8",
    )
    artifacts: dict[str, dict[str, Any]] = {}
    selected = {
        "baseline": FEATURES,
        "reduced": ["feature_a"],
    }
    training = pd.DataFrame(
        {
            "feature_a": [-2.0, -1.0, -0.5, 0.0, 0.2, 0.7, 1.0, 2.0] * 6,
            "feature_b": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 6,
        }
    )
    labels = pd.Series(([0, 0, 0, 0, 1, 1, 1, 1] * 6), dtype="int8")
    for configuration_id, feature_names in selected.items():
        rows: dict[str, Any] = {}
        for seed in v06f.EXPECTED_SEEDS:
            model = LightweightDetector(
                "decision_tree", feature_names, PARAMETERS, random_state=seed
            ).fit(training, labels)
            path = results / f"{configuration_id}_{seed}.joblib"
            model.save(path)
            payload = path.read_bytes()
            artifact = {
                "availability": "hashed",
                "reference": str(path),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "model_state_sha256": v06f._model_state_sha256(model),
                "serialized_model_bytes": len(payload),
            }
            rows[str(seed)] = {
                "selected_features": list(feature_names),
                "selected_features_sha256": fingerprint_feature_names(feature_names),
                "model_artifact": artifact,
            }
        artifacts[configuration_id] = {
            "configuration_id": configuration_id,
            "selected_features": {
                seed: row["selected_features"] for seed, row in rows.items()
            },
            "selected_feature_fingerprints": {
                seed: row["selected_features_sha256"] for seed, row in rows.items()
            },
            "seed_specific": rows,
            "metrics": {
                "average_precision": 0.7 if configuration_id == "baseline" else 0.69,
                "f1": 0.6,
                "recall": 0.5,
                "precision": 0.65,
                "roc_auc": 0.75,
            },
        }

    assignments = {
        "full_baseline": "baseline",
        "best_unsupervised": "reduced",
        "best_supervised": "reduced",
        "smallest_preserving": "reduced",
    }
    roles = {}
    for role, configuration_id in assignments.items():
        configuration = artifacts[configuration_id]
        roles[role] = {
            "configuration_id": configuration_id,
            "locked_configuration_ref": configuration_id,
            "selected_features": deepcopy(configuration["selected_features"]),
            "selected_feature_fingerprints": deepcopy(
                configuration["selected_feature_fingerprints"]
            ),
            "model_artifacts": {
                seed: deepcopy(row["model_artifact"])
                for seed, row in configuration["seed_specific"].items()
            },
        }
    lock = {
        "stage": "V0.6-E",
        "test_accessed": False,
        "seeds": list(v06f.EXPECTED_SEEDS),
        "attack_configuration": {
            "attack_type": "mixed",
            "attack_rate": 0.05,
            "attack_severity": "MEDIUM",
            "attack_config_path": str(attack_config),
        },
        "model_configuration": {
            "model_id": "decision_tree",
            "parameters": deepcopy(PARAMETERS),
        },
        "threshold": {"prediction_threshold": 0.5},
        "decision_threshold": 0.5,
        "candidate_manifest": {
            "features": FEATURES,
            "feature_count": len(FEATURES),
            "sha256": fingerprint_feature_names(FEATURES),
        },
        "candidate_manifest_hash": fingerprint_feature_names(FEATURES),
        "locked_configurations": artifacts,
        "roles": roles,
        "semantic_payload_sha256": None,
    }
    lock["semantic_payload_sha256"] = semantic_payload_sha256(lock)
    _write_signed_lock(results, lock)
    return results


def _write_signed_lock(results: Path, lock: dict[str, Any]) -> None:
    lock_path = results / "validation_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    file_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    (results / "validation_lock.sha256").write_text(
        f"{file_hash}  validation_lock.json\n", encoding="ascii"
    )
    required_checks = [
        "source_d_leakage_audit_pass",
        "source_d_test_accessed_false",
        "source_d_metadata_test_accessed_false",
        "source_hashes_verified",
        "source_semantics_reconstructed",
        "semantic_payload_verified",
        "sidecar_file_hash_verified",
    ]
    audit = {
        "stage": "V0.6-E",
        "status": "PASS",
        "test_accessed": False,
        "semantic_payload_sha256": lock["semantic_payload_sha256"],
        "validation_lock_file_sha256": file_hash,
        "checks": [{"name": name, "passed": True} for name in required_checks],
    }
    (results / "validation_lock_audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )


def _fake_verifier(lock_path: Path, **_: Any) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    return {
        "status": "PASS",
        "semantic_payload_sha256": lock["semantic_payload_sha256"],
        "file_sha256": hashlib.sha256(Path(lock_path).read_bytes()).hexdigest(),
        "source_hashes_verified": True,
        "source_semantics_reconstructed": True,
        "sidecar_verified": True,
        "test_accessed": False,
    }


def _gate(results: Path) -> v06f.IntegrityContext:
    return v06f.pre_test_integrity_gate(results, verifier=_fake_verifier)


def _resign(results: Path, mutate: Any) -> None:
    lock = json.loads((results / "validation_lock.json").read_text(encoding="utf-8"))
    mutate(lock)
    lock["semantic_payload_sha256"] = semantic_payload_sha256(lock)
    _write_signed_lock(results, lock)


def _synthetic_bundle() -> Any:
    features = pd.DataFrame(
        {
            "row_id": np.arange(10),
            "feature_a": np.linspace(-1.0, 1.0, 10),
            "feature_b": np.tile([0.0, 1.0], 5),
        }
    )
    metadata = pd.DataFrame({"row_id": np.arange(10), "source": ["x"] * 10})
    return SimpleNamespace(
        selected_features=list(FEATURES),
        splits={
            "train": ProcessedSplit("train", features.copy(), metadata.copy(), None),
            "test": ProcessedSplit("test", features.copy(), metadata.copy(), None),
        },
    )


def _preparer_factory(calls: list[int], objects: dict[int, PreparedAttackSplit]):
    def prepare(**kwargs: Any) -> PreparedAttackSplit:
        seed = kwargs["random_seed"]
        calls.append(seed)
        clean = kwargs["split"].features.copy(deep=True)
        attacked = clean.copy(deep=True)
        attacked.loc[[1, 7], "feature_a"] += 2.0
        labels = pd.Series([0, 1, 0, 0, 0, 0, 0, 1, 0, 0], name="is_attack", dtype="int8")
        ground_truth = pd.DataFrame(
            {"record_id": clean["row_id"], "is_attack": labels, "random_seed": seed}
        )
        manifest = pd.DataFrame(
            {"record_id": [1, 7], "random_seed": [seed, seed], "change": [2.0, 2.0]}
        )
        attack_metadata = {
            "attack_mode": "mixed",
            "configured_attack_rate": 0.05,
            "severity": "MEDIUM",
            "random_seed": seed,
        }
        prepared = PreparedAttackSplit(
            split_name="test",
            candidate_features=tuple(FEATURES),
            clean_features=clean,
            features=attacked,
            clean_metadata=kwargs["split"].metadata.copy(),
            metadata=kwargs["split"].metadata.copy(),
            labels=labels,
            ground_truth=ground_truth,
            manifest=manifest,
            attack_metadata=attack_metadata,
            row_ids_sha256=fingerprint_row_ids(attacked),
            clean_features_sha256=fingerprint_feature_matrix(clean, FEATURES),
            attacked_features_sha256=fingerprint_feature_matrix(attacked, FEATURES),
            labels_sha256=fingerprint_attack_labels(attacked, labels),
            ground_truth_sha256=fingerprint_frame(ground_truth),
            manifest_sha256=fingerprint_frame(manifest),
            attack_metadata_sha256=fingerprint_mapping(attack_metadata),
            test_authorization_id=kwargs["test_authorization_id"],
            test_authorization_capability=kwargs["test_authorization_capability"],
        )
        objects[seed] = prepared
        return prepared

    return prepare


def _inference(model: LightweightDetector, features: pd.DataFrame, **_: Any) -> Any:
    return SimpleNamespace(
        predictions=model.predict_frame(features),
        resources=ResourceMeasurement(0.01, 0.005, 50.0, None, 1024 * 1024, 0, 0),
    )


def _run(
    results: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    context: v06f.IntegrityContext | None = None,
    loader: Any = None,
) -> tuple[dict[str, Any], list[int], dict[int, PreparedAttackSplit]]:
    calls: list[int] = []
    objects: dict[int, PreparedAttackSplit] = {}
    monkeypatch.setattr(v06f, "_write_figures", lambda *_: {})
    result = v06f.run_v06f_final_test(
        results,
        context=context or _gate(results),
        loader=loader or (lambda *_args, **_kwargs: _synthetic_bundle()),
        attack_preparer=_preparer_factory(calls, objects),
        inference_runner=_inference,
        synthetic_test=True,
    )
    return result, calls, objects


def test_corrupted_lock_fails_before_loader(signed_lock: Path) -> None:
    lock_path = signed_lock / "validation_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["decision_threshold"] = 0.7
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    loader_calls = 0

    with pytest.raises(ValidationLockError, match="semantic payload"):
        v06f.pre_test_integrity_gate(signed_lock, verifier=_fake_verifier)
    assert loader_calls == 0


def test_wrong_lock_sidecar_sha_fails(signed_lock: Path) -> None:
    (signed_lock / "validation_lock.sha256").write_text(
        f"{'0' * 64}  validation_lock.json\n", encoding="ascii"
    )
    with pytest.raises(ValidationLockError, match="sidecar mismatch"):
        _gate(signed_lock)


def test_wrong_model_file_sha_fails(signed_lock: Path) -> None:
    def mutate(lock: dict[str, Any]) -> None:
        lock["locked_configurations"]["baseline"]["seed_specific"]["42"][
            "model_artifact"
        ]["file_sha256"] = "0" * 64
        lock["roles"]["full_baseline"]["model_artifacts"]["42"][
            "file_sha256"
        ] = "0" * 64

    _resign(signed_lock, mutate)
    with pytest.raises(v06f.V06FIntegrityError, match="hash or size"):
        _gate(signed_lock)


def test_selected_feature_fingerprint_mutation_fails(signed_lock: Path) -> None:
    def mutate(lock: dict[str, Any]) -> None:
        lock["locked_configurations"]["reduced"]["seed_specific"]["42"][
            "selected_features_sha256"
        ] = "0" * 64
        for role in ("best_unsupervised", "best_supervised", "smallest_preserving"):
            lock["roles"][role]["selected_feature_fingerprints"]["42"] = "0" * 64

    _resign(signed_lock, mutate)
    with pytest.raises(v06f.V06FIntegrityError, match="feature maps|fingerprint mismatch"):
        _gate(signed_lock)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "random_forest", "frozen V0.5 decision tree"),
        ("threshold", 0.4, "threshold"),
        ("attack", 0.2, "mixed/0.05/MEDIUM"),
    ],
)
def test_model_threshold_and_attack_mutations_fail(
    signed_lock: Path, field: str, value: Any, message: str
) -> None:
    def mutate(lock: dict[str, Any]) -> None:
        if field == "model":
            lock["model_configuration"]["model_id"] = value
        elif field == "threshold":
            lock["threshold"]["prediction_threshold"] = value
        else:
            lock["attack_configuration"]["attack_rate"] = value

    _resign(signed_lock, mutate)
    with pytest.raises(v06f.V06FIntegrityError, match=message):
        _gate(signed_lock)


def test_duplicate_roles_collapse_and_role_mapping_is_preserved(signed_lock: Path) -> None:
    context = _gate(signed_lock)
    assert len(context.plan.configurations) == 2
    assert v06f._role_mapping(context.plan) == {
        "full_baseline": "baseline",
        "best_unsupervised": "reduced",
        "best_supervised": "reduced",
        "smallest_preserving": "reduced",
    }


def test_seed_locked_features_remain_unchanged(
    signed_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _gate(signed_lock)
    before = {
        (configuration.configuration_id, seed.seed): seed.selected_features
        for configuration in context.plan.configurations
        for seed in configuration.seed_plans
    }
    _run(signed_lock, monkeypatch, context=context)
    after = {
        (configuration.configuration_id, seed.seed): seed.selected_features
        for configuration in context.plan.configurations
        for seed in configuration.seed_plans
    }
    assert after == before


def test_selector_fit_and_model_fit_are_never_called(
    signed_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _gate(signed_lock)
    monkeypatch.setattr(
        LightweightDetector,
        "fit",
        lambda *_args, **_kwargs: pytest.fail("training API reached final test"),
    )
    result, _, _ = _run(signed_lock, monkeypatch, context=context)
    assert result["failed_run_count"] == 0
    rows = pd.read_csv(signed_lock / "final_test" / "final_test_runs.csv")
    assert rows["selector_fit_time_sec"].isna().all()
    assert rows["model_fit_time_sec"].isna().all()


def test_labels_and_features_cannot_reach_a_training_api(
    signed_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _gate(signed_lock)
    import src.lightweight.training as training

    monkeypatch.setattr(training, "train_model", lambda *_a, **_k: pytest.fail("train_model called"))
    result, _, _ = _run(signed_lock, monkeypatch, context=context)
    assert result["completed_run_count"] == 10


def test_one_manifestation_per_seed_is_reused_for_every_unique_configuration(
    signed_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[int, set[int]] = {seed: set() for seed in v06f.EXPECTED_SEEDS}
    original = v06f._evaluate_locked_run

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seed_plan = args[2]
        prepared = args[4]
        seen[seed_plan.seed].add(id(prepared))
        return original(*args, **kwargs)

    monkeypatch.setattr(v06f, "_evaluate_locked_run", capture)
    _, calls, _ = _run(signed_lock, monkeypatch)
    assert calls == list(v06f.EXPECTED_SEEDS)
    assert all(len(object_ids) == 1 for object_ids in seen.values())


def test_execution_plan_mutation_is_rejected_before_loader(
    signed_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _gate(signed_lock)
    object.__setattr__(context.plan, "candidate_features", ("mutated",))
    loader_calls = 0

    def loader(*_: Any, **__: Any) -> Any:
        nonlocal loader_calls
        loader_calls += 1
        return _synthetic_bundle()

    with pytest.raises(v06f.V06FIntegrityError, match="execution plan mutated"):
        _run(signed_lock, monkeypatch, context=context, loader=loader)
    assert loader_calls == 0


def test_metric_naming_uses_explicit_trapezoidal_name_only(
    signed_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(signed_lock, monkeypatch)
    records = json.loads(
        (signed_lock / "final_test" / "final_test_runs.json").read_text(encoding="utf-8")
    )
    assert "pr_auc_trapezoidal" in records[0]
    assert all("pr_auc" not in record for record in records)


def test_access_and_audit_turn_true_only_after_gate_and_before_loader(
    signed_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _gate(signed_lock)
    output = signed_lock / "final_test"
    assert not output.exists()

    def loader(*_: Any, **__: Any) -> Any:
        state = json.loads((output / "test_access_state.json").read_text(encoding="utf-8"))
        assert state["test_accessed"] is True
        return _synthetic_bundle()

    _run(signed_lock, monkeypatch, context=context, loader=loader)
    audit = json.loads(
        (output / "final_test_governance_audit.json").read_text(encoding="utf-8")
    )
    assert audit["test_accessed"] is True
    assert audit["status"] == "PASS"


def test_no_reselection_occurs_after_test_metrics_exist(
    signed_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _gate(signed_lock)
    mapping_before = v06f._role_mapping(context.plan)
    _run(signed_lock, monkeypatch, context=context)
    rows = pd.read_csv(signed_lock / "final_test" / "final_test_runs.csv")
    preservation = pd.read_csv(signed_lock / "final_test" / "final_test_preservation.csv")
    assert not rows["reselection_performed"].any()
    assert not preservation["selection_performed"].any()
    assert v06f._role_mapping(context.plan) == mapping_before


def test_gate_failure_message_and_nonzero_exit(
    signed_lock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        v06f,
        "pre_test_integrity_gate",
        lambda *_a, **_k: (_ for _ in ()).throw(ValidationLockError("bad lock")),
    )
    monkeypatch.setattr(
        v06f,
        "run_v06f_final_test",
        lambda *_a, **_k: pytest.fail("loader-bearing runner called after gate failure"),
    )
    assert v06f.main(["--results-dir", str(signed_lock)]) == 1
    assert capsys.readouterr().out.strip() == v06f.PRETEST_FAILURE_MESSAGE
