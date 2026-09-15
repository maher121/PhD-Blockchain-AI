"""Cross-platform regression tests for canonical scientific hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.versioning import sha256_frame
from src.pipeline_v06e import fingerprint_feature_names as v06e_feature_fingerprint
from src.pipeline_v08c import winner_lock_semantic_hash
from src.pipeline_v08d import final_test_lock_semantic_hash
from src.security.experiment_data import fingerprint_feature_names


RESULTS = Path("results")
V06_MANIFEST_HASH = "5146fd08fe766979adaf443bf9f4f7d32ee3c94cfdaf0fd46ec0efd10e92a10d"
V08C_FEATURE_HASH = "5157d5bb6b17dc6c92b790671dac029e2b0d2b4454db9824870db1290a6c4121"
V08C_MASK_HASH = "5da981b5b87db97338ecdde9ca8a8b87db3a62771d03dc6a4ad901f6548a3299"
V08C_SEMANTIC_HASH = "0f356ccab422774b128ab82f5da8881f24816902d4ccc6a7cecc9760c6202ca8"
V08D_SEMANTIC_HASH = "ac19e5a0dba5d058f88485e6ab8ab9a96f97c30f697e95a744ccaa6563d2b657"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_frame(features):
    return pd.DataFrame(
        {
            "position": np.arange(len(features), dtype=int),
            "feature": list(features),
        }
    )


def test_canonical_dataframe_hash_uses_lf_not_host_or_crlf() -> None:
    frame = pd.DataFrame({"position": [0, 1], "feature": ["alpha", "beta"]})
    lf_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    crlf_bytes = frame.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
    assert b"\r\n" not in lf_bytes
    assert b"\n" in lf_bytes
    assert sha256_frame(frame) == hashlib.sha256(lf_bytes).hexdigest()
    assert sha256_frame(frame) != hashlib.sha256(crlf_bytes).hexdigest()


def test_historical_v06_manifest_hash_remains_authoritative() -> None:
    lock = _read_json(RESULTS / "feature_selection" / "validation_lock.json")
    features = lock["candidate_manifest"]["features"]
    crlf_hash = hashlib.sha256(
        _feature_frame(features)
        .to_csv(index=False, lineterminator="\r\n")
        .encode("utf-8")
    ).hexdigest()
    assert lock["candidate_manifest_hash"] == V06_MANIFEST_HASH
    assert lock["candidate_manifest"]["sha256"] == V06_MANIFEST_HASH
    assert fingerprint_feature_names(features) == V06_MANIFEST_HASH
    assert v06e_feature_fingerprint(features) == V06_MANIFEST_HASH
    assert crlf_hash != V06_MANIFEST_HASH


def test_v08c_feature_mask_and_semantic_hashes_remain_authoritative() -> None:
    lock = _read_json(RESULTS / "bpso" / "v08c_winner_lock.json")
    features = lock["ordered_selected_features"]
    mask = np.asarray(lock["mask"], dtype=np.uint8)
    crlf_hash = hashlib.sha256(
        _feature_frame(features)
        .to_csv(index=False, lineterminator="\r\n")
        .encode("utf-8")
    ).hexdigest()
    assert lock["selected_features_sha256"] == V08C_FEATURE_HASH
    assert fingerprint_feature_names(features) == V08C_FEATURE_HASH
    assert crlf_hash != V08C_FEATURE_HASH
    assert lock["mask_sha256"] == V08C_MASK_HASH
    assert hashlib.sha256(mask.tobytes()).hexdigest() == V08C_MASK_HASH
    assert lock["semantic_lock_sha256"] == V08C_SEMANTIC_HASH
    assert winner_lock_semantic_hash(lock) == V08C_SEMANTIC_HASH


def test_v08d_semantic_result_lock_remains_authoritative() -> None:
    lock = _read_json(RESULTS / "bpso" / "v08d_final_test_lock.json")
    assert lock["semantic_result_lock_sha256"] == V08D_SEMANTIC_HASH
    assert final_test_lock_semantic_hash(lock) == V08D_SEMANTIC_HASH
