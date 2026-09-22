# V1.1-A — Lightweight Blockchain Scientific Protocol and Architecture Lock

Companion to `config/blockchain_v11.yaml`. This document is the authoritative, implementable
protocol for the blockchain component of the PhD study **"Lightweight and Green Blockchain–AI
Framework"**. Stage `V1.1-A` freezes design and governance only: no engine, validation mechanism,
or experiment is implemented, executed, or measured here.

Protocol-config SHA-256: `3b42e767b8101213896681e386dad2b5afa6a71016db5c960f1b27e301c44d9b`

**Protocol classification:** `BLOCKCHAIN_LIGHTWEIGHT_PROTOCOL_LOCKED`

---

## 1. Scope

- `V1.1-A` is a **design lock**: it specifies (a) an order-centric, hash-linked lightweight
  blockchain architecture, (b) a canonical serialization and SHA-256 hash scheme, (c) a frozen
  AI–blockchain reference contract over closed V1.0 outputs, (d) a threat model and bounded
  security claims, and (e) a preregistered Adaptive Lightweight Validation (ALV) policy with
  baselines, workloads, seeds, metrics, and experiment scenarios.
- Out of scope: any implementation, any execution, any tampering simulation, any training,
  re-training, or model reconstruction, any raw dataset access, and any energy measurement.
- A researcher must be able to implement V1.1-B and beyond from this document plus the config
  without redesign decisions.
- Starting checkpoint: `813a49b93db3271a275097e44c947ba96a09f763` (V1.0-I commit, origin/main).

## 2. Governance and immutability

- V1.0 is **closed and immutable**. The blockchain component consumes V1.0 artifacts
  read-only. No V1.0 pipeline, notebook, config, result lock, or test is re-run or mutated.
- Energy claims follow the governed policy (§24): `DIRECT_ENERGY_UNAVAILABLE`.
- This stage produces exactly four artifacts:
  1. `docs/v11_blockchain_protocol.md` (this document),
  2. `config/blockchain_v11.yaml`,
  3. `tests/test_protocol_v11a.py`,
  4. `results/blockchain/v11a/v11a_protocol_lock.json`.
- No commit or push is applied in V1.1-A; the stage is reviewed before any later checkpoint.

## 3. Upstream frozen evidence and AI reference

Frozen, closed V1.0 evidence referenced (never recomputed):

| Item | Value |
| --- | --- |
| V1.0 close checkpoint | `813a49b93db3271a275097e44c947ba96a09f763` (V1.0-I) |
| AI configuration | `HYBRID-K13` (13 selected features, 43-dim canonical space) |
| Feature manifest SHA-256 | `5146fd08fe766979adaf443bf9f4f7d32ee3c94cfdaf0fd46ec0efd10e92a10d` |
| V1.0-D winner lock semantic hash | `1c258dc1a90ad78197d25e62bbc9fca24904cbcb77edafd24b12726059e49d84` |
| V1.0-E result lock semantic hash | `ffca0957006bff3115705f9dfb0334e13a27c9abbc502f2c9d04f784a28141e9` |
| V1.0-F resource lock semantic hash | `f714b4a95070ee26718b3b2f8dc2ed820c5c2dadb7c5fb58fa1c09f95505b412` |
| V1.0-G ablation lock semantic hash | `e33414d491d50676bfb5dfc926dbe4ada1a16931fa083a0e8b914958b99fc4de` |

### Critical honesty note on the AI reference

The frozen V1.0 results are **aggregate** final-test metrics (average precision, F1, recall) for
the `HYBRID-K13` decision-tree and logistic-regression models. **No per-row / per-order AI
predictions exist** in the V1.0 artifacts — there is no persisted per-order AI risk output and no
inference service. This statement is kept: V1.0 provides aggregate evaluation evidence only.
Consequently the blockchain protocol never claims to hold per-order AI scores, and it defines two
distinct risk-input modes (§13.1): the **primary `GOVERNED_AI_RISK`** mode (per-order risk
produced, persisted, and provenance-locked in the later governed stage V1.1-D before ALV consumes
it) and the labeled **research-generated synthetic** mode (`SYNTHETIC_RISK`) for simulation,
ablation, and testing only.

## 4. Architecture: order-centric per-order logical blockchain

- **Primary classification:** `ORDER_CENTRIC_PER_ORDER_LOGICAL_BLOCKCHAIN`. Every distinct
  order gets its own hash-linked chain of blocks: `GENESIS → EVENT_1 → … → EVENT_k → FINAL`.
- Blocks of unrelated orders never co-occur in one shared chain. Order-level chaining provides:
  - **per-order traceability** of the frozen event sequence,
  - **tamper localization** (an anomaly is localizable to the affected order chain / block),
  - **fault isolation** (a corrupted order chain does not invalidate other orders),
  - **bounded per-order verification cost** (verification walks one chain, not a global ledger),
  - **independent verification** (each chain verifies standalone relative to its genesis anchor).
- These are **design motivations / testable hypotheses**, not empirically demonstrated
  superiority (any such claim requires the registered V1.1 experiments, §20).
- **Optional lightweight global summary layer** (`OPTIONAL_LIGHTWEIGHT_GLOBAL_SUMMARY_ABSTRACTION`):
  a hash-root index over completed per-order chains, proposed only to enable cheap cross-order
  consistency checks (e.g., detection of cross-order substitution, PA-08). It is explicitly a
  thin, non-mandatory abstraction — **not** a second complex blockchain — and is out of the V1.1-B
  core.

## 5. Data honesty: DataCo semantics (A / B / C classification)

DataCo SMART Supply Chain is a **historical, tabular** dataset. It is **not** a native blockchain
event log and has **no provenance DAG**. Every event type and every payload field carries exactly
one provenance class:

- **A — OBSERVED_ATTRIBUTE**: values recorded as attributes in the governed raw rows, e.g.
  `order date (DateOrders)`, `shipping date (DateOrders)`, `Delivery Status`, `Order Status`,
  `Type`. Usable as recorded observations only.
- **B — DETERMINISTICALLY_DERIVED**: representations reconstructed from observed attributes by a
  fixed, frozen rule — e.g. `ORDER_CREATED` from the order date, `SHIPMENT_RECORDED` from the
  shipping date, the canonical `order_id` from `Order Id`, and all digests/aggregates. These are
  **derived representations of observed attributes**, never claimed as native event logs.
- **C — RESEARCH_GENERATED**: scenario-only artifacts created for experiments and labeled as such —
  e.g. synthetic tampering payloads (§14) and the synthetic per-order risk reference (§18).

**Rule:** derived (B) and research-generated (C) items are never presented as observed (A)
blockchain events, and no claim of "blockchain data provenance" for the whole DataCo corpus is made.

## 6. Order identity and per-order chain identity

- **Canonical chain key:** `order_id`, derived from the governed raw `Order Id` column
  (present in `DATACO_IDENTIFIER_COLUMNS`; mapped `Order Id → order_id` under the 34-column
  canonical preprocessing).
- **Normalization:** canonical decimal integer string (no sign, no leading zeros).
- **Item lines:** each row of an order is a line identified by `transaction_id` (from
  `Order Item Id`; also canonical decimal integer string). Multi-row orders are allowed.
- **Uniqueness:** exactly one logical chain per distinct canonical `order_id` in the governed
  workload sample.
- **Order-level aggregation policy:** order payloads aggregate that order's item lines over
  canonical columns only, deterministically: `item_count` = number of lines; per-line canonical
  payload digests sorted by `transaction_id`; order payload digest is the nested hash over the
  ordered line digests. Train-derived order-level features from V1.0 (e.g. `item_count_per_order`,
  `order_total_value`) appear in research-generated payloads only with an explicit provenance note.

## 7. Block schema

`schema_version = "1"`. Each block is a JSON object with the following fields:

| Field | Type | Required | Provenance | Purpose |
| --- | --- | --- | --- | --- |
| `schema_version` | string | yes | protocol | schema self-identification |
| `order_id` | string | yes | derived | canonical chain identity |
| `block_index` | integer | yes | derived | chain position, ≥ 0, monotone |
| `event_type` | string | yes | per event type | taxonomy discriminator |
| `event_timestamp` | string | yes | observed/derived | canonical ISO-8601 UTC |
| `payload_digest` | string | yes | derived | payload integrity |
| `payload_ref` | string | no | per event type | off-chain canonical payload locator |
| `ai_risk_reference` | object / null | no | mode-dependent | provenance-locked AI risk record (governed V1.1-D, or labeled research-generated synthetic) |
| `validation_policy` | string | yes | derived | `AUTHORIZED_VALIDATOR_<variant>` |
| `previous_hash` | string / null | yes | derived | link to predecessor; null only for genesis |
| `block_hash` | string | yes | derived | `H(canonical(block_without_block_hash))` |

Constraints: `block_index` starts at 0 (genesis) and is strictly monotone per order; `event_timestamp`
is non-decreasing per order chain; `previous_hash` is null only in the genesis block.

## 8. Genesis rule

- Index `0`, `event_type = "ORDER_GENESIS"`, `previous_hash = null`.
- **Deterministic; no random nonce.** Content is the canonical JSON of
  `{"order_id": ..., "order_creation_canonical": ...}` (order creation derived from the governed
  order date).
- Self-hash: `H(canonical_serialization(block_without_block_hash))`. The genesis is reproducible
  from the canonical `order_id` alone.

## 9. Event taxonomy (static catalog, frozen)

| Event | Provenance class | Meaning |
| --- | --- | --- |
| `ORDER_GENESIS` | derived | deterministic anchor block |
| `ORDER_CREATED` | derived | from governed order date |
| `SHIPMENT_RECORDED` | derived | from governed shipping date |
| `DELIVERY_STATUS_RECORDED` | observed | recorded `Delivery Status` (operational outcome; **not** an AI feature) |
| `AI_RISK_ASSESSED` | mode-dependent | references a provenance-locked per-order AI risk record (governed V1.1-D, or labeled research-generated synthetic) |

Per-order chains append the static event sequence in frozen order
(`ORDER_GENESIS, ORDER_CREATED, SHIPMENT_RECORDED, DELIVERY_STATUS_RECORDED, AI_RISK_ASSESSED`).
Stage clocks are order-coherent, so `event_timestamp` is non-decreasing by construction.

## 10. Canonical serialization

One deterministic byte encoding for every hashed object (the same encoding used by all V1.0
semantic-hash locks):

- encoding `utf-8`
- `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`
- no trailing newline
- floats: finite, Python-round-trippable; NaN/Infinity forbidden
- times: pre-serialized ISO‑8601 UTC strings (`%Y-%m-%dT%H:%M:%SZ`); never native datetime objects
- sentinel: JSON `null` for absent/optional items (e.g. `previous_hash` in genesis)
- array order: preserved defined order (e.g. line digests sorted by `transaction_id`)
- semantic hash: `sha256(canonical_bytes)`; the compact canonical encoding is exactly
  `json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")`.

## 11. Hashing and chain linking

- Algorithm: **SHA-256**, used as an **integrity primitive only**. Hashing is **not encryption**;
  a digest alone provides **no confidentiality**.
- **Preimage:** `block_without_block_hash`, i.e. the block's canonical serialization with
  `block_hash` excluded (prevents circular self-reference). `block_hash` never participates in its
  own computation.
- **Linking:** `block_with_block_hash.previous_hash` equals the `block_hash` of the immediately
  preceding block of the same order chain. One strict link per chain; indexed by `block_index`.
- Security role: tamper evidence and deterministic verification; see §15 for what is **not**
  claimed.

## 12. Payload strategy

- **Primary:** `DIGEST_MINIMAL_METADATA` — on-chain blocks carry `payload_digest` plus minimal
  metadata; the authoritative full canonical payloads live in a versioned **off-chain canonical
  payload archive** keyed by `payload_ref`.
- **Tradeoff (documented):** digest-only blocks are small and cheap to verify, but content
  re-verification requires the archive. This is a reproducibility choice, **not** confidentiality
  or access control.
- **Full-payload variant:** optional experiment-only mode for storage-size measurement (E08);
  it changes on-chain size but never the hash scheme.

## 13. AI–Blockchain integration contract

Contract name: `AI-BLOCKCHAIN_frozen_reference_contract`.

- The blockchain component **reads frozen V1.0 outputs only** (never retrains, never reselects
  features, never reruns optimizers, never tunes with TEST outcomes). `HYBRID-K13` is immutable
  in V1.1.
- The `ai_risk_reference` object is the frozen evidence record reference:
  - `stage: V1.0-E`, `configuration_id: HYBRID-K13`,
  - classifiers `decision_tree` and `logistic_regression`,
  - `feature_space_dimension: 43`, `selected_feature_count: 13`,
  - `feature_manifest_sha256: 5146fd08fe766979adaf443bf9f4f7d32ee3c94cfdaf0fd46ec0efd10e92a10d`,
    evidence kind: aggregate frozen final-test metrics
    (average precision, F1, recall).
- **Kept statement:** V1.0 contains **aggregate evaluation evidence only** — there are **no
  persisted per-order AI risk predictions** among the frozen V1.0 artifacts
  (`per_order_risk_existence_in_v10_frozen_artifacts: false`). The protocol does not pretend
  otherwise.

### 13.1 Two distinct risk-input modes

- **A — `GOVERNED_AI_RISK` (intended primary integration mode).** Produced in a later authorized
  governed stage (V1.1-D, §26) from the frozen V1.0 AI configuration. `HYBRID-K13` remains
  immutable; no feature reselection, no optimizer execution, no TEST-based threshold tuning.
  Per-order risk records are **persisted and provenance-locked (record digest + stage
  fingerprint) before any ALV experiment consumes them**. Whether retraining is ever needed is a
  separate governed decision — this contract does **not** presume or decide it.
- **B — `SYNTHETIC_RISK` (simulation/ablation/testing only).** `RESEARCH_GENERATED`; a
  deterministic SHA-256-derived score over the canonical `order_id` normalized to `[0,1]`. It is
  **never presented as an AI prediction** and **never the sole evidence** for the thesis claim
  that AI adaptively controls blockchain validation. It is permitted only for functional tests,
  stress tests, controlled ablation, and validation-path coverage.

### 13.2 Per-order risk record contract

Each consumed risk record carries exactly these fields, deterministically serialized and digests:
`order_id` (canonical), `risk_score` (float in `[0,1]`), `risk_level` (`LOW|MEDIUM|HIGH`),
`model_configuration_provenance` (HYBRID-K13 classifier identity),
`hybrid_k13_provenance` (feature-manifest + mask + V1.0 semantic-lock digests),
`generation_stage_provenance` (V1.1-D for mode A; labeled research-generated for mode B), and
`record_digest` = SHA-256 over the canonical serialization of the six preceding fields.
**Unlocked consumption is forbidden:** a governed stage fingerprint over the persisted artifact
must be recorded before ALV may consume it.

### 13.3 Generation vs. mapping vs. validation policy (kept distinct)

1. **Risk-score generation** produces the `order_id`-level `risk_score`; its provenance depends on
   the mode (governed in V1.1-D, or labeled research-generated synthetic).
2. **Risk-level mapping** is a deterministic frozen rule from `risk_score` to `risk_level`
   (`< 0.3333 → LOW`; `[0.3333, 0.6667) → MEDIUM`; `≥ 0.6667 → HIGH`), preregistered here and
   independent of generation mode. It is **not** tuned on any validation outcome.
3. **Blockchain validation policy** maps `risk_level` to the check set and validator count (§18);
   this policy is frozen and identical regardless of which mode supplied the risk record.

### 13.4 Threshold preregistration

The LOW/MEDIUM/HIGH thresholds are **preregistered and frozen in this stage, before any ALV
evaluation**. They must never be tuned using final TEST outcomes or any validation outcome.

## 14. Threat model

Perspective: an off-chain actor (malicious or accidental) can reach the local archives and
tamper-editorialize copies. No full-node-compromise or adversarial-supermajority assumptions.

| ID | Scenario | Detection | Localization |
| --- | --- | --- | --- |
| PA-01 | payload modification | `payload_digest` mismatch at affected block | affected block |
| PA-02 | block deletion | `previous_hash` mismatch / index gap | first affected block |
| PA-03 | block insertion | `previous_hash` mismatch / index gap | insertion point |
| PA-04 | previous-hash modification | `block_hash` recomputation mismatch | modified block |
| PA-05 | order-id modification | genesis `order_id` prefix mismatch vs expected chain id | order chain |
| PA-06 | AI-risk record modification | `ai_risk_reference` digest verification failure | affected block/order |
| PA-07 | chain truncation | declared `block_count` vs chained reach | truncation point |
| PA-08 | cross-order substitution | per-order chain identity checks (summary layer) | substituted chain |
| PA-09 | cross-order replay | timestamp / order coherence + per-order uniqueness checks | replayed block |

**Execution rule:** attacks are run only on generated experiment copies, never on frozen V1.0
artifacts.

## 15. Security properties and claim boundaries

Demonstrated-scope properties (within the prototype): integrity (tamper evidence via chained
hashes), deterministic verification, traceability of per-order event order, order-level fault
isolation, AI-result provenance linkage via frozen reference digests.

**Explicitly NOT claimed:** confidentiality, anonymity/privacy, decentralization, BFT consensus,
non-repudiation, legal immutability. Any of these requires facilities this lightweight prototype
does not provide.

## 16. Terminology

Preferred: **lightweight research prototype**, **hash-linked order ledger**,
**blockchain-inspired integrity layer**, **per-order logical chain**.
Avoid: "full blockchain network", "mining layer", "consensus under adversarial nodes".

## 17. Validation: authorized validator, lightweight

- Mode: `AUTHORIZED_VALIDATOR_LIGHTWEIGHT` — validation is deterministic re-verification by an
  authorized validator; **no proof-of-work and no mining**.
- Core check set:
  - `c1` schema conformance
  - `c2` self `block_hash` recomputation
  - `c3` `previous_hash` linkage
  - `c4` event timestamp non-decreasing per order
  - `c5` order integrity (canonical `order_id` chain-prefix match)
  - `c6` `ai_risk_reference` present for AI-linked event blocks
  - `c7` `ai_risk_reference` digest re-verification
  - `c8` chain prefix re-verification (hash re-check of prior blocks)
- Acceptance: **all applied checks pass**; verdict list is deterministic; no probabilistic acceptance.

## 18. Adaptive Lightweight Validation (ALV)

`ADAPTIVE_LIGHTWEIGHT_VALIDATION` maps the frozen risk band to a check set and validator count:

| Band | Checks | Validators |
| --- | --- | --- |
| LOW | `c1 c2 c3 c4` | 1 |
| MEDIUM | `c1 c2 c3 c4 c5 c6` | 1 |
| HIGH | `c1 c2 c3 c4 c5 c6 c7 c8` | 3 |

- **Risk-mode policy:** the active risk-input mode (`GOVERNED_AI_RISK` or `SYNTHETIC_RISK`)
  determines only the **provenance** of the per-order risk record (§13.1). The `risk_level`
  mapping and the band→check policy above are **identical regardless of mode**. The principal
  evidence for AI-driven ALV behavior is always `GOVERNED_AI_RISK`; `SYNTHETIC_RISK` is
  simulation/ablation/testing only and never the sole evidence for the thesis claim.
- **Fail-safe:** missing or malformed risk reference escalates the block to MEDIUM validation.
- **Tie handling:** validators are deterministic functions; over the same block they cannot
  disagree. On any input discrepancy the block is **rejected with the reason recorded**.
- ALV is **defined mathematically/algorithmically here and not implemented** in V1.1-A. The
  generation, mapping, and validation-policy steps are kept distinct (§13.3) and all are
  preregistered and frozen; none is tuned on outcomes.

## 19. Baselines and fairness

Registered, preregistered baselines:
- **B0** — fixed single authorized validator; checks `c1…c4` on every block.
- **B1** — fixed stronger validation; quorum of 3 deterministic authorized validators on every block.
- **P** — proposed ALV; band-specific check sets and validator counts.

**Fairness rule:** B0, B1, and P evaluate the identical workload, serialization, hash scheme, block
schema, attack instances, machine, seeds, and measurement procedure; only the validation policy
differs.

## 20. Experiments (registered scenarios, NOT executed in V1.1-A)

- E01 functional correctness, E02 integrity / tamper detection, E03 tamper localization,
  E04 order-level fault isolation, E05 AI-linked adaptive behavior, E06 execution-time overhead,
  E07 memory overhead, E08 storage overhead, E09 throughput, E10 resource/green proxy.
- Status of every scenario in V1.1-A: `CANDIDATE_SCENARIO`. None is run. All run on generated
  copies; none touches frozen V1.0 artifacts.
- **Risk-mode requirements per scenario:** E01–E04 and E06–E10 may use either
  `GOVERNED_AI_RISK` or `SYNTHETIC_RISK`. **E05 (principal AI-driven ALV behavior) requires
  `GOVERNED_AI_RISK`** (produced, persisted, and provenance-locked in V1.1-D before the principal
  ALV experiment in V1.1-E consumes it);
  `SYNTHETIC_RISK` may support functional and stress pre-checks but is **never the sole evidence**
  for E05's conclusion. `SYNTHETIC_RISK` is otherwise permitted for functional tests, stress
  tests, controlled ablation, and validation-path coverage.
- No empirical results, no comparisons, and no superiority claims derive from V1.1-A.

## 21. Metrics

- Functional: `verification_pass_rate`, `detection_rate`, `localization_rate`, `fault_isolation_rate`.
- Performance: `wall_time_per_order_chain`, `memory_footprint_mib`, `storage_bytes_per_block`,
  `blocks_per_second`.
- Adaptive: `validation_level_distribution`, `validated_checks_per_block`, `quorum_frequency`.
- Green: `computational_proxy_checks_per_block`, `normalized_resource_proxy_vs_baselines`.
- **Proxy rule:** CPU/wall-time and resource counts are proxies only; they are never labeled or
  inferred as measured Joules (§24).

## 22. Workloads and scaling

- Basis: the frozen governed 40,000-row canonical sample (order-grouped split, cap and seeds as
  in `src/config.py`; `GLOBAL_SEED=42`). Workloads are drawn deterministically by canonical
  `order_id`; **no full-corpus processing**.
- Justification: raw DataCo ≈ 180,519 rows / ≈ 65,752 orders (~2.75 item lines per order); the
  40,000-row cap implies on the order of 14,000–15,000 unique orders, so even the largest workload
  stays well inside the governed sample.
- Order counts: `[100, 250, 500, 1000, 2500]`; repetitions: 3; deterministic sampling via
  canonical seeds.

## 23. Seeds and reproducibility

- Dedicated `blockchain_seed_family = [522, 523, 524]`, used for order sampling, synthetic risk
  scores, and per-order chain construction.
- **No overlap** with prior frozen families (`[42…46]`, `[1042…1046]`, `[2042…2046]`,
  `[3042…3046]`); the V1.1 family is recorded in the config and never reused for optimizers.
- Synthetic risk derivation is SHA-256-based and platform-independent; all randomness is seeded
  and integer-based.

## 24. Energy / green policy

- Direct physical energy measurement is unavailable: marker **`DIRECT_ENERGY_UNAVAILABLE`**.
- **Forbidden inference:** TDP-times-time, any "measured Joules" claim, any "measured energy
  reduction" claim.
- Permitted framing: computational proxy, resource proxy, "lightweight",
  green-computing-relevance wording only.

## 25. Implementation constraints (future stages)

- Local Python 3.12 only; **no** Ethereum, Hyperledger Fabric, Solidity, Docker, cloud, or mining.
- Reproducibility thread count = 1 where applicable.
- Upstream consumption is strictly read-only over frozen V1.0 artifacts.

## 26. Claim governance and future boundaries

V1.1-A makes **no empirical claims** of any kind. Corrected governed stage map:

| Stage | Scope |
| --- | --- |
| **V1.1-A** | Blockchain Protocol & Architecture Lock (this stage) |
| **V1.1-B** | Core Lightweight Blockchain Engine Implementation (canonical serialization, hashing, block schema, chaining, validation primitives) |
| **V1.1-C** | DataCo Order/Event Mapping and Blockchain Construction (canonical `order_id` chains built deterministically over the governed sample) |
| **V1.1-D** | Governed Per-Order AI Risk Artifact and Frozen-AI Integration — **produce, persist, provenance-lock (record digest + stage fingerprint), and freeze `GOVERNED_AI_RISK`** here, **before** any later stage consumes it; `HYBRID-K13` unchanged and immutable; whether retraining is ever needed is a separate governed decision |
| **V1.1-E** | AI-Driven Adaptive Lightweight Validation and Integrity/Tampering Experiments — **principal AI-driven ALV experiment consumes the frozen `GOVERNED_AI_RISK` artifact from V1.1-D; consumption without that provenance/lock is forbidden**; `SYNTHETIC_RISK` restricted to functional, stress, ablation, and validation-path-coverage runs |
| **V1.1-F** | Resource-Efficiency / Green Evaluation (computational-proxy rules, green wording, §24) |
| **V1.1-G** | Final Blockchain Scientific Analysis and Notebook (read-only over `results/`) |

- **Consumption precondition:** V1.1-E cannot consume the governed risk artifact unless the
  V1.1-D provenance/lock exists (checked by protocol tests in `tests/test_protocol_v11a.py`).
- The thesis claim that AI adaptively controls blockchain validation rests on `GOVERNED_AI_RISK`
  evidence from V1.1-E; any claim of ALV superiority requires the controlled B0/B1/P comparison
  (§19–§21) with `GOVERNED_AI_RISK` principal evidence.

## 27. Critical self-review: reviewer-resistant design questions and responses

The following concerns are explicitly challenged and answered by this protocol.

1. *"Is this just hashing?"* — The deliverable is a scientific protocol for a lightweight
   hash-linked order ledger plus a preregistered adaptive-validation research program (E01–E10).
   "Just hashing" is honestly acknowledged as the integrity primitive; the research contribution
   is the order-centric architecture choice, the A/B/C honesty discipline, the frozen AI-reference
   contract, and the ALV adaptive-validation policy, each evaluated explicitly in registered
   experiments — never overstated as a production chain.
2. *"Why per-order chains and not one shared ledger?"* — Order-centric chaining is justified by
   design goals (traceability, tamper localization, fault isolation, bounded per-order verification,
   independent verification). These are hypotheses to be tested by E01–E04, not asserted as proven.
3. *"Why is a global summary layer optional?"* — A monolithic secondary chain would add complexity
   without a demonstrated benefit for integrity; the optional hash-root index narrows cross-order
   checks only (PA-08). Honest about its absence from the V1.1-B core.
4. *"Off-chain payloads are not authenticated."* — Admitted. The off-chain archive is
   self-describing and digest-linked; authenticity is outside the demonstrated security scope and
   is not claimed (§15).
5. *"DataCo is not a blockchain dataset and has been over-interpreted."* — This protocol never
   claims native blockchain provenance for DataCo and enforces the A/B/C classification (§5); all
   lifecycle events are labeled derived, and risk bands are labeled research-generated (§13, §18).
6. *"Digest-only blocks cannot be verified by third parties."* — Correct and documented (§12); the
   full-payload variant exists only for storage-size experiments, and re-verification uses the
   canonical archive.
7. *"Risk bands imply AI scores the study does not have."* — Fully acknowledged (§13): frozen V1.0
   contains aggregate evaluation evidence only and **no persisted per-order AI risk predictions**.
   V1.1-A therefore defines two distinct modes: the **primary `GOVERNED_AI_RISK`** mode requires
   a later governed stage (V1.1-D) to produce, persist, and provenance-lock per-order risk records
   before ALV consumes them, and `SYNTHETIC_RISK` remains a labeled research-generated mode that is
   never presented as an AI prediction and never the sole evidence for the AI-driven ALV claim.
8. *"Validation without PoW provides no security under adversarial nodes."* — Accepted; the
   prototype targets deterministic authorized-validator integrity evidence and explicitly excludes
   BFT/decentralization claims (§15).
9. *"Adaptive validation is circular if risk is its own output."* — The risk reference is
   independent of the validation verdict (provenance-locked per-order risk record); ALV never tunes
   thresholds on outcomes (§13.4, §18).
10. *"Performance and green claims."* — No wall-clock/energy claim is made in V1.1-A; E06–E10 are
    registered scenarios and will be reported under the strict proxy rule and
    `DIRECT_ENERGY_UNAVAILABLE` marker (§21, §24).
11. *"The protocol is not implementable."* — Rejected as a concern: §4–§19 give field tables,
    exact canonical serialization, event catalog, threat table, check sets, and frozen workloads,
    seeds, and threshold constants; a fresh researcher can build V1.1-B from this document alone.
12. *"No empirical claim today, so the stage adds nothing."* — The stage's deliverable is a
    preregistered protocol: reproducibility and honesty of later measurements depend on locking
    decisions before any experiment runs, which is exactly what V1.1-A does.
13. *"Claims of 'lightweight' and 'green' are vague."* — Operationalized as
    computational-proxy metrics (checks per block, resource proxies vs. baselines) and governed
    wording (§21, §24); no Joules and no TDP-times-time inference anywhere.
14. *"Synthetic tampering is not real-world attack data."* — Accepted; attack scenarios (PA-01…
    PA-09) are laboratory scenarios on generated copies, disclosed as such, and used only to test
    the protocol's detection/localization logic.

---

*End of `docs/v11_blockchain_protocol.md`.*