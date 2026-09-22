from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "ai_risk_v11d.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "v11d_governed_ai_risk_protocol.md"
PROTOCOL_LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11d" / "v11d_ai_risk_protocol_lock.json"
)
V10D_LOCK_PATH = PROJECT_ROOT / "results" / "hybrid" / "v10d" / "v10d_winner_lock.json"
V10E_LOCK_PATH = PROJECT_ROOT / "results" / "hybrid" / "v10e" / "v10e_result_lock.json"
V11A_LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11a" / "v11a_protocol_lock.json"
)
V11C_LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11c" / "v11c_mapping_lock.json"
)

EXPECTED_FEATURES = [
    "order_item_quantity",
    "product_price",
    "order_item_total",
    "is_weekend",
    "Type_DEBIT",
    "Type_TRANSFER",
    "Type_CASH",
    "Market_LATAM",
    "Market_Pacific Asia",
    "Shipping Mode_First Class",
    "Department Name_Golf",
    "Department Name_Fitness",
    "Department Name_Health and Beauty",
]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict:
    return json.loads(PROTOCOL_LOCK_PATH.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_stage_is_protocol_only(cfg):
    assert cfg["stage"] == "V1.1-D1"
    assert cfg["kind"] == "SCIENTIFIC_DECISION_PROTOCOL_LOCK"
    identity = cfg["stage_identity"]
    assert identity["implementation_excluded"] is True
    assert identity["prediction_generation_excluded"] is True
    assert identity["blockchain_experiment_execution_excluded"] is True


def test_scope_forbids_scientific_execution_and_test(cfg):
    scope = cfg["scope"]
    forbidden = [
        "model_fit_allowed",
        "model_reconstruction_allowed",
        "prediction_allowed",
        "predict_proba_allowed",
        "decision_function_allowed",
        "risk_record_generation_allowed",
        "ai_risk_block_generation_allowed",
        "v11e_execution_allowed",
        "optimizer_execution_allowed",
        "feature_reselection_allowed",
        "test_access_allowed",
        "test_threshold_tuning_allowed",
    ]
    assert all(scope[name] is False for name in forbidden)


def test_hybrid_k13_identity_matches_v10d(cfg):
    frozen = cfg["feature_identity"]
    upstream = _read_json(V10D_LOCK_PATH)
    assert frozen["configuration_id"] == "HYBRID-K13"
    assert frozen["selected_feature_count"] == upstream["selected_feature_count"] == 13
    assert frozen["feature_space_dimension"] == upstream["feature_dimensions"] == 43
    assert frozen["winner_mask_sha256"] == upstream["mask_sha256"]
    assert frozen["canonical_feature_manifest_sha256"] == upstream["feature_manifest_sha256"]
    assert frozen["selected_feature_list_sha256"] == upstream["feature_list_sha256"]


def test_exact_feature_order_is_frozen(cfg):
    assert cfg["feature_identity"]["ordered_features"] == EXPECTED_FEATURES
    assert _read_json(V10D_LOCK_PATH)["ordered_selected_features"] == EXPECTED_FEATURES
    assert cfg["feature_identity"]["reselection_forbidden"] is True


def test_primary_classifier_and_parameters_are_frozen(cfg):
    classifier = cfg["classifier"]
    assert classifier["primary"] == "decision_tree"
    assert classifier["parameters"] == {
        "class_weight": "balanced",
        "max_depth": 5,
        "min_samples_leaf": 20,
    }
    assert classifier["hard_prediction_threshold"] == 0.5
    assert classifier["secondary_sensitivity_classifier"] == "logistic_regression"
    assert classifier["secondary_may_generate_primary_artifact"] is False


def test_primary_classifier_matches_frozen_v10e_configuration(cfg):
    v10e = _read_json(V10E_LOCK_PATH)["classifier_configurations"]["decision_tree"]
    assert cfg["classifier"]["parameters"] == v10e["parameters"]
    assert cfg["classifier"]["hard_prediction_threshold"] == v10e["prediction_threshold"]


def test_model_strategy_is_fixed_five_seed_mean(cfg):
    strategy = cfg["model_strategy"]
    assert strategy["type"] == "FIXED_FIVE_SEED_MEAN_ENSEMBLE"
    assert strategy["seeds"] == [42, 43, 44, 45, 46]
    assert strategy["member_count"] == 5
    assert strategy["member_classifier"] == "decision_tree"
    assert strategy["fixed_equal_weights"] is True
    assert strategy["selection_or_weight_learning_allowed"] is False


def test_score_semantics_and_positive_class_are_frozen(cfg):
    assert cfg["target"]["name"] == "is_attack"
    assert cfg["target"]["unit"] == "ROW"
    assert cfg["target"]["encoding"] == {"normal": 0, "attacked": 1}
    assert cfg["target"]["positive_class"] == 1
    assert cfg["target"]["late_delivery_risk_is_target"] is False
    score = cfg["risk_score"]
    assert score["semantics_version"] == "MODEL_DERIVED_ATTACK_RISK_V1"
    assert score["range"] == "[0,1]"
    assert "uncalibrated" in score["interpretation"]
    assert score["clipping_allowed"] is False


def test_row_to_order_rule_is_max_and_fail_closed(cfg):
    aggregation = cfg["row_to_order"]
    assert aggregation["rule"] == "MAX"
    assert aggregation["missing_row_policy"] == "FAIL_CLOSED"
    assert aggregation["omitted_row_policy"] == "FAIL_CLOSED"
    assert "maximum" in cfg["risk_score"]["order_score"]


def test_v11a_thresholds_are_retained_without_tuning(cfg):
    thresholds = cfg["thresholds"]
    assert thresholds["policy"] == "RETAIN_V11A_PREREGISTERED_BANDS"
    assert thresholds["low"] == {"lower_inclusive": 0.0, "upper_exclusive": 0.3333}
    assert thresholds["medium"] == {
        "lower_inclusive": 0.3333,
        "upper_exclusive": 0.6667,
    }
    assert thresholds["high"] == {"lower_inclusive": 0.6667, "upper_inclusive": 1.0}
    assert thresholds["validation_or_test_tuning_allowed"] is False
    assert thresholds["distribution_dependent_redefinition_allowed"] is False


def test_generation_partition_is_clean_validation_only(cfg):
    partition = cfg["generation_partition"]
    assert partition["primary"] == "VALIDATION"
    assert partition["rows"] == 6000
    assert partition["expected_unique_orders"] == 4588
    assert partition["input_view"] == "CLEAN_FROZEN_VALIDATION_FEATURES"
    assert partition["model_fit_partition"] == "TRAIN"
    assert partition["train_rows"] == 28000
    assert partition["train_and_validation_combination_for_primary_artifact_allowed"] is False
    assert partition["test_allowed"] is False


def test_preprocessing_reconstruction_is_train_only(cfg):
    preprocessing = cfg["preprocessing"]
    assert preprocessing["fit_partition"] == "TRAIN"
    assert preprocessing["fit_rows"] == 28000
    assert preprocessing["deterministic_reconstruction_required"] is True
    assert preprocessing["numeric_scaling"] == "StandardScaler fit on TRAIN only"
    assert "HYBRID-K13 projection" in preprocessing["output_feature_order"]


def test_training_reconstruction_binds_frozen_attack_workloads(cfg):
    reconstruction = cfg["training_reconstruction"]
    assert reconstruction["source_contract"] == \
        "src.pipeline_v08b.load_frozen_development_workloads"
    assert reconstruction["fit_split"] == "TRAIN"
    assert reconstruction["fit_rows"] == 28000
    assert reconstruction["attack_type"] == "mixed"
    assert reconstruction["attack_rate"] == 0.05
    assert reconstruction["attack_severity"] == "MEDIUM"
    assert reconstruction["attack_and_model_seeds"] == [42, 43, 44, 45, 46]
    assert "training_labels_sha256" in reconstruction["fingerprint_requirements"]
    assert "fails closed" in reconstruction["verification_rule"]


def test_risk_record_schema_and_digest_are_frozen(cfg):
    record = cfg["risk_record"]
    required = record["required_fields_in_order"]
    assert required[0] == "schema_version"
    assert required[-1] == "record_digest"
    for field in [
        "order_id",
        "risk_score",
        "risk_level",
        "classifier",
        "model_seeds",
        "aggregation_rule",
        "source_partition",
        "model_configuration_provenance",
        "hybrid_k13_provenance",
        "generation_stage_provenance",
    ]:
        assert field in required
    assert "except record_digest" in record["record_digest_rule"]
    assert record["unlock_before_use_forbidden"] is True


def test_blockchain_linkage_is_minimal_and_digest_bound(cfg):
    linkage = cfg["blockchain_linkage"]
    assert linkage["event_type"] == "AI_RISK_ASSESSED"
    assert linkage["risk_mode"] == "GOVERNED_AI_RISK"
    assert linkage["payload_policy"] == "DIGEST_MINIMAL_METADATA"
    assert linkage["append_position"] == "after DELIVERY_STATUS_RECORDED"
    assert "record_digest" in linkage["ai_risk_reference_fields"]
    assert "artifact_lock_sha256" in linkage["ai_risk_reference_fields"]
    assert linkage["raw_features_on_chain"] is False
    assert linkage["raw_labels_on_chain"] is False
    assert linkage["row_identifiers_on_chain"] is False


def test_alv_linkage_remains_v11a_policy(cfg):
    alv = cfg["alv_linkage"]
    assert alv["low"] == {"checks": ["c1", "c2", "c3", "c4"], "validator_count": 1}
    assert alv["medium"] == {
        "checks": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "validator_count": 1,
    }
    assert alv["high"] == {
        "checks": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
        "validator_count": 3,
    }
    assert alv["superiority_claim_before_v11e_allowed"] is False


def test_energy_governance_forbids_joules_and_tdp(cfg):
    policy = cfg["energy_governance"]
    assert policy["marker"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert policy["joules_claim_allowed"] is False
    assert policy["tdp_times_runtime_allowed"] is False
    assert "validation checks" in policy["permitted_future_proxies"]


def test_upstream_semantic_linkages_match_artifacts(cfg):
    upstream = cfg["upstream"]
    assert upstream["v10d_winner_semantic_sha256"] == _read_json(V10D_LOCK_PATH)[
        "semantic_lock_sha256"
    ]
    assert upstream["v10e_result_semantic_sha256"] == _read_json(V10E_LOCK_PATH)[
        "semantic_result_lock_sha256"
    ]
    assert upstream["v11a_protocol_semantic_sha256"] == _read_json(V11A_LOCK_PATH)[
        "semantic_result_lock_sha256"
    ]
    assert upstream["v11c_mapping_semantic_sha256"] == _read_json(V11C_LOCK_PATH)[
        "semantic_result_lock_sha256"
    ]


def test_protocol_lock_binds_files_and_semantic_payload(lock):
    payload = lock["semantic_payload"]
    fingerprints = payload["artifact_fingerprints"]
    assert fingerprints["protocol_config_sha256"] == _sha256_file(CONFIG_PATH)
    assert fingerprints["protocol_document_sha256"] == _sha256_file(PROTOCOL_PATH)
    assert lock["semantic_result_lock_sha256"] == _semantic_sha256(payload)
    assert lock["stage"] == "V1.1-D1"


def test_protocol_lock_records_no_execution(lock):
    metadata = lock["execution_metadata"]
    assert metadata["model_fit_executed"] is False
    assert metadata["predictions_generated"] is False
    assert metadata["risk_records_generated"] is False
    assert metadata["blockchain_experiments_executed"] is False
    assert metadata["optimizer_executed"] is False
    assert metadata["test_access_count"] == 0


def test_protocol_document_states_required_boundaries():
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    for marker in [
        "uncalibrated model-derived attack-risk score",
        "MAX",
        "VALIDATION",
        "DIGEST_MINIMAL_METADATA",
        "DIRECT_ENERGY_UNAVAILABLE",
        "TEST remains unopened",
    ]:
        assert marker in text
