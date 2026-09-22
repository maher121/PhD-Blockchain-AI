# V1.1-D1 - Governed AI-Blockchain Scientific Decision Lock

Protocol configuration: `config/ai_risk_v11d.yaml`
Protocol classification: `GOVERNED_AI_BLOCKCHAIN_SCIENTIFIC_DECISIONS_LOCKED`

## 1. Purpose and boundary

V1.1-D1 resolves only the scientific choices required to translate the frozen V1.0
`HYBRID-K13` evidence into a future per-order `GOVERNED_AI_RISK` artifact. It does not fit or
load a model, generate a prediction, create a risk record, append `AI_RISK_ASSESSED`, access
TEST, execute an optimizer, or run V1.1-E. V1.0 and V1.1-A/B/C remain immutable.

The design is preregistered before any governed score distribution or blockchain experiment is
observed. Later empirical outcomes may reject a hypothesis, but must not silently change this
protocol.

## 2. Frozen upstream evidence

| Evidence | Frozen value |
| --- | --- |
| Starting checkpoint | `e5c85aea6492431d37732b4214fef4974ad5c088` |
| AI configuration | `HYBRID-K13` (13 of 43 features) |
| Winner mask SHA-256 | `d1cc8c8b8643ce5d5daff3bdb6ff4c74dab6b1d78069a7f0052bcda10cb68c62` |
| Canonical 43-feature manifest SHA-256 | `5146fd08fe766979adaf443bf9f4f7d32ee3c94cfdaf0fd46ec0efd10e92a10d` |
| Ordered 13-feature list SHA-256 | `8be4b0818f6ca59a057ef91d47738fa692548a6ddbed983b4496c644296b906a` |
| V1.0-D semantic winner lock | `1c258dc1a90ad78197d25e62bbc9fca24904cbcb77edafd24b12726059e49d84` |
| V1.0-E semantic result lock | `ffca0957006bff3115705f9dfb0334e13a27c9abbc502f2c9d04f784a28141e9` |
| V1.1-A semantic protocol lock | `aaf3ac3139e8296fb7f976a8a7ed9f18eb317af2f73b10cd0208b6bb870ebd8d` |
| V1.1-C semantic mapping lock | `c992429b2165f18649d36b0339260d9a95e1ae8524d7dfd586a168504d3cc765` |

The exact selected features, in frozen order, are:

1. `order_item_quantity`
2. `product_price`
3. `order_item_total`
4. `is_weekend`
5. `Type_DEBIT`
6. `Type_TRANSFER`
7. `Type_CASH`
8. `Market_LATAM`
9. `Market_Pacific Asia`
10. `Shipping Mode_First Class`
11. `Department Name_Golf`
12. `Department Name_Fitness`
13. `Department Name_Health and Beauty`

Feature reselection is forbidden.

## 3. Decision table

| Decision | Options considered | Selected design | Scientific rationale | Potential limitation | Frozen before V1.1-E? |
| --- | --- | --- | --- | --- | --- |
| Primary classifier | DT; LR | Frozen depth-5 DT | It is the V1.0 feature-search classifier, has materially stronger frozen attack-detection evidence than LR, is bounded, lightweight, deterministic under a seed, and interpretable. The score is not called calibrated. | Tree probabilities are piecewise and uncalibrated; LR remains sensitivity-only. | YES |
| `risk_score` semantics | Literal probability; model-derived score; hard class | Uncalibrated model-derived attack-risk score in `[0,1]` | Preserves direction and class semantics without an unsupported real-world calibration claim. | Cannot be interpreted as attack incidence, causality, or certainty. | YES |
| Row-to-order aggregation | MAX; MEAN; NOISY-OR | `MAX` | One suspicious line should not be diluted by normal lines. MAX is deterministic, transparent, and O(n). | Sensitive to one extreme or erroneous row; fail-closed input checks are required. | YES |
| LOW/MEDIUM/HIGH bands | Retain V1.1-A thirds; TRAIN/VALIDATION relock | Retain `<0.3333`, `[0.3333,0.6667)`, `>=0.6667` | The score is finite and bounded; V1.1-A froze the bands independently of risk mode before outcomes. Retention avoids post-hoc tuning. | Bands may be imbalanced or empty; this is an empirical result, not grounds to retune. | YES |
| Generation partition | VALIDATION; TRAIN+VALIDATION; other non-TEST | Clean VALIDATION only | Avoids in-sample blockchain-risk evidence while providing 4,588 orders, enough for workloads up to 2,500. | Covers a subset, not all 25,881 V1.1-C development orders. | YES |
| Model strategy | One seed; one reconstruction; five-seed ensemble | Equal mean of five frozen-seed DTs (42-46) | Reduces arbitrary single-seed dependence without model selection. Five bounded trees remain small. | Approximately fivefold inference operations versus one tree. | YES |
| Risk-record schema | V1.1-A minimum; expanded minimal provenance | Minimum plus classifier, seeds, strategy, aggregation, partition, score version, row count/digest | Makes reconstruction and order aggregation auditable without storing raw features or labels. | Slightly larger off-chain records. | YES |
| `AI_RISK_ASSESSED` linkage | Raw payload; digest-only; minimal metadata plus digest | Minimal metadata plus authoritative record and artifact-lock digests | Proves identity, score, level, model lineage, and authoritative source while preserving `DIGEST_MINIMAL_METADATA`. | Re-verification requires the off-chain governed artifact. | YES |

## 4. Primary classifier

The primary governed classifier is the frozen V1.0 decision tree:

- `DecisionTreeClassifier`
- `class_weight="balanced"`
- `max_depth=5`
- `min_samples_leaf=20`
- `random_state` bound to each frozen model seed
- V1.0 hard-label threshold `0.5` retained as provenance, but hard labels do not define V1.1-D
  `risk_score`.

This is not a new model-selection result. The decision is based on consistency with the governed
V1.0 search objective, bounded computational complexity, interpretability, reproducibility, and
the materially stronger frozen attack-detection evidence relative to LR. LR remains available only
for a separately declared sensitivity analysis and must not create the primary governed artifact.

## 5. Model instances and reconstruction

The primary score uses a fixed equal-weight ensemble over seeds `[42, 43, 44, 45, 46]`. Each
member is the same frozen DT configuration. For a row `r` and seed `s`, let `q_s(r)` be the
classifier output associated with the class whose label is exactly `is_attack=1`. The row score is:

`row_attack_risk(r) = (q_42(r) + q_43(r) + q_44(r) + q_45(r) + q_46(r)) / 5`.

No learned ensemble weights, seed filtering, post-hoc member selection, or outcome-driven member
replacement is allowed.

The V1.0-F joblib files and manifest are reconstruction references, not automatic authority:
they are untracked resource-stage artifacts. A future authorized implementation must reconstruct
the five DTs deterministically from the frozen, fingerprint-verified TRAIN workloads loaded through
`src.pipeline_v08b.load_frozen_development_workloads`. The reconstruction recipe is the frozen
mixed attack task at rate `0.05`, severity `MEDIUM`, under seeds `42-46` and
`config/attack_scenarios.yaml`. For every seed, TRAIN row IDs, clean features, attacked features,
and generated `is_attack` labels must match the SHA-256 fingerprints in
`results/feature_selection/core_runs.csv` before fitting is authorized. It then applies the frozen
preprocessing, feature order, parameters, and corresponding model seed and provenance-locks the
reconstructed model state. This is exact governed reconstruction, not feature selection,
hyperparameter tuning, or retraining with a new scientific design. If any fingerprint or model
identity check fails, generation fails closed.

## 6. Target and score semantics

The frozen target is row-level `is_attack`:

- `0`: normal/control row;
- `1`: controlled experiment-generated attacked row;
- positive/risk class: exactly integer class `1`.

The V1.0 DataCo `Late_delivery_risk` column is not the V1.1-D target.

`risk_score` is an **uncalibrated model-derived attack-risk score**, not a calibrated probability
of real-world malicious activity. It is finite, lies in `[0,1]`, and increases as an order appears
more attack-like under the frozen model family and controlled V1.0 task. No clipping or rounding is
permitted before canonical serialization and hashing. The following claims are forbidden:

- real-world attack probability or incidence;
- causal risk;
- certainty that an order is malicious;
- calibration unless a future, separately preregistered calibration study establishes it.

## 7. Frozen preprocessing

The future implementation reconstructs the frozen `DataCoPreprocessor` from TRAIN only:

1. train-derived order aggregates under the V1.0 rules;
2. TRAIN numeric medians and categorical modes;
3. `StandardScaler` fitted on TRAIN numeric features only;
4. TRAIN-derived one-hot levels in frozen order;
5. canonical 43-feature ordering;
6. projection to the exact ordered HYBRID-K13 list.

VALIDATION is transformed only; TEST is never opened. A persisted fitted preprocessor may be used
only if its state and provenance are verified against the frozen specification.

## 8. Row-to-order aggregation

For canonical order `o`, let `R(o)` be all eligible VALIDATION rows whose retained `Order Id`
normalizes to `o`. The frozen aggregation is:

`risk_score(o) = max(row_attack_risk(r) for r in R(o))`.

MAX is selected because cybersecurity validation should remain sensitive to a single suspicious
item line. MEAN can dilute one attacked line in a multi-line order. NOISY-OR is rejected because
its probability interpretation requires conditional independence and calibrated member scores,
neither of which is established; it also rises mechanically with order size.

Every eligible row must contribute exactly once. Missing, duplicate, omitted, non-finite, or
out-of-range scores cause fail-closed rejection. Equal maxima require no row selection and yield the
same scalar under any row order.

## 9. Risk levels

The V1.1-A bands are retained exactly:

- LOW: `0 <= risk_score < 0.3333`
- MEDIUM: `0.3333 <= risk_score < 0.6667`
- HIGH: `0.6667 <= risk_score <= 1`

These are operational validation-intensity bands, not calibrated probability categories. V1.1-A
froze them independently of generation mode before V1.1-E outcomes. Their distribution must be
reported honestly. Empty or imbalanced bands do not authorize tuning on VALIDATION or TEST and do
not authorize threshold replacement after observing blockchain outcomes.

## 10. Generation partition

The primary future artifact covers the clean frozen VALIDATION partition only: 6,000 rows and
4,588 canonical orders. Model reconstruction uses TRAIN only (28,000 rows). TRAIN predictions do
not enter the primary artifact because they would be in-sample evidence. TEST remains unopened.

The clean VALIDATION view is scored without V1.1-D attack mutation. This preserves exact linkage
to V1.1-C source orders. The 4,588 eligible orders exceed the largest preregistered V1.1 workload
of 2,500 orders. Orders outside VALIDATION receive no governed risk record in this primary design;
the absence must never be silently filled with synthetic risk.

## 11. Governed risk-record schema

Each future off-chain authoritative record contains, in the frozen schema:

1. `schema_version`
2. `order_id`
3. `risk_score`
4. `risk_level`
5. `source_row_count`
6. `source_rows_digest`
7. `classifier`
8. `model_strategy`
9. `model_seeds`
10. `aggregation_rule`
11. `source_partition`
12. `score_semantics_version`
13. `model_configuration_provenance`
14. `hybrid_k13_provenance`
15. `generation_stage_provenance`
16. `record_digest`

`source_rows_digest` is SHA-256 over canonical JSON of sorted stable `row_id` strings. The row IDs
remain off-chain. `record_digest` is SHA-256 over canonical JSON of all preceding fields. Canonical
serialization follows V1.1: sorted keys, compact separators, ASCII escaping, finite numbers, UTF-8,
and no trailing newline.

## 12. Blockchain linkage

Only the 4,588 eligible V1.1-C validation orders may receive a governed `AI_RISK_ASSESSED` block
from this primary artifact. The event is appended after `DELIVERY_STATUS_RECORDED`. The event uses
the governed artifact's frozen `generated_at_utc` and must not precede the chain head timestamp.

The on-chain `ai_risk_reference` contains only:

- `mode = GOVERNED_AI_RISK`
- `stage = V1.1-D`
- canonical `order_id`
- `risk_score` and `risk_level`
- `configuration_id = HYBRID-K13`
- `classifier = decision_tree`
- `aggregation_rule = MAX`
- `record_digest`
- `artifact_ref`
- `artifact_lock_sha256`

Raw features, labels, row IDs, model objects, and attack manifests are never stored on-chain. The
off-chain risk artifact is authoritative. The block is accepted only after record-digest and
artifact-lock verification. This preserves the V1.1-A `DIGEST_MINIMAL_METADATA` policy.

## 13. Adaptive Lightweight Validation linkage

V1.1-A ALV remains unchanged:

| Level | Checks | Validators |
| --- | --- | ---: |
| LOW | `c1-c4` | 1 |
| MEDIUM | `c1-c6` | 1 |
| HIGH | `c1-c8` | 3 |

The AI-derived level selects validation intensity; it never changes block contents or acceptance
criteria within the selected check set. V1.1-E will test the preregistered hypothesis that
risk-adaptive validation can reduce computational validation cost while preserving required
integrity and tamper-detection behavior under the registered attack model. V1.1-D1 makes no
security-superiority claim.

## 14. Lightweight and green boundary

Direct energy remains `DIRECT_ENERGY_UNAVAILABLE`. Joules, measured energy savings, and
TDP-times-runtime inference are forbidden. V1.1-F may report wall-clock runtime, CPU time, hash
operations, validation checks, validator count, memory, throughput, and clearly labeled
computational proxies. Five-tree ensemble overhead must be measured, not omitted or described as
free. A computational proxy is never relabeled as energy.

## 15. Frozen architecture

```text
DataCo VALIDATION rows (clean; TEST forbidden)
    |
Frozen TRAIN-fitted preprocessing
    |
HYBRID-K13 projection
    |
Five frozen-seed depth-5 decision trees
    |
Mean row-level model-derived attack-risk score
    |
MAX over all rows of each canonical Order Id
    |
Order-level risk_score
    |
Frozen LOW / MEDIUM / HIGH mapping
    |
Governed V1.1-D off-chain risk artifact
    |
Record digest + artifact semantic lock
    |
AI_RISK_ASSESSED minimal reference
    |
Eligible V1.1-C OrderChain
    |
V1.1-E Adaptive Lightweight Validation
```

## 16. Future implementation gates

No later stage may generate or consume governed risk unless it verifies this protocol lock. Model
reconstruction requires explicit authorization, must remain TRAIN-only, and must not select
features, tune classifiers, alter seed weights, inspect TEST, or tune thresholds. The governed risk
artifact and its semantic lock must exist before any `AI_RISK_ASSESSED` block or principal V1.1-E
experiment consumes it.

V1.1-D1 status: protocol decisions frozen for scientific review; no implementation or experiment
has been executed.
