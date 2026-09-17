# V0.9-A BGWO Scientific Protocol Lock

- **Stage:** V0.9-A (planning and protocol lock only)
- **Optimizer family:** Binary Grey Wolf Optimizer (BGWO)
- **Role in novelty sequence:** independent swarm baseline between V0.8 BPSO and V1.0 hybrid
- **Execution policy:** no optimizer implementation, no optimization runs, no new model-training experiments, no final-test access

## 1. Scope and Scientific Objective

Primary question:

> Can BGWO identify a compact feature subset that satisfies the same frozen predictive-preservation constraints used for BPSO while providing a useful alternative swarm-search baseline?

V0.9-A locks protocol, governance, and fairness constraints only. It does not execute the experiment.

## 2. Governed Preflight and Frozen Anchors

Preflight checks confirm V0.8 locks and frozen integrity remain valid:

- V0.8-C winner lock verified: `results/bpso/v08c_winner_lock.json`
- V0.8-D final-test lock verified: `results/bpso/v08d_final_test_lock.json`
- V0.8-E scientific result lock verified: `results/bpso/v08e_e4_analysis/v08e_scientific_result_lock.json`
- Frozen hash integrity verified across immutable historical artifacts (`53` locked hashes)

Frozen semantic identities used as BGWO comparability anchors:

- V0.8-D semantic result lock SHA-256: `ac19e5a0dba5d058f88485e6ab8ab9a96f97c30f697e95a744ccaa6563d2b657`
- V0.8-E semantic result lock SHA-256: `7307fda1cdb5f100cb6268dc0f4eff05b65bfcefadeb06707c8a3e13fd6888d5`
- V0.8 final classification: `STRUCTURAL_GAIN_TIMING_UNCERTAIN`

## 3. Fair-Comparison Lock (BPSO vs BGWO)

To keep BPSO-vs-BGWO scientifically controlled, BGWO inherits V0.8 governance where optimizer mechanics do not require differences:

- same governed DataCo pipeline and preprocessing
- same 40,000-row cap and grouped split (`28,000` train / `6,000` validation / `6,000` test)
- same 43-feature space
- same controlled attack protocol and ground truth (`is_attack`)
- same prohibited targets for cyber ground truth (`Late_delivery_risk`, `SUSPECTED_FRAUD`)
- same Decision Tree search classifier and threshold
- same predictive-preservation constraints and deterministic candidate ranking
- same model/attack seed namespace (`42..46`)
- same independent-run count (`5`)
- same search-budget philosophy (comparable, frozen before execution)
- same final-test governance (search blind to test)
- same seed-level statistical protocol (`n=5`, `df=4`, paired CI framing)

Only optimizer equations and optimizer-seed namespace differ by design.

## 4. Data and Ground-Truth Governance

Locked boundaries:

- **Search-visible data:** train + validation only
- **Search-forbidden data:** final test split (must remain untouched until winner lock)
- **Ground truth:** `is_attack` only
- **Forbidden as cyber labels:** `Late_delivery_risk`, `SUSPECTED_FRAUD`

## 5. Predictive Baseline and Preservation Constraints

V0.6 validation K43 baseline (frozen for V0.9):

- AP = `0.2306840611`
- F1 = `0.1977010726`
- Recall = `0.322`

Governance preservation thresholds (unchanged):

- AP relative loss <= `5%`
- F1 relative loss <= `5%`
- Recall relative loss <= `10%`

These are experiment-governance thresholds, not domain-clinical/security guarantees.

## 6. Optimization Objective and Ranking Semantics

BGWO objective remains identical to V0.8:

- **Constrained minimum-cardinality feature selection**
- **Primary objective:** smallest feasible subset meeting preservation constraints
- **Excluded from fitness:** timing, RSS, model size, throughput, energy, carbon

Deterministic ranking lock:

### 6.1 Feasible candidates
1. smaller `K`
2. higher AP
3. higher F1
4. higher Recall
5. canonical deterministic tie-break

### 6.2 Infeasible candidates
1. lower total normalized violation
2. higher AP
3. higher F1
4. higher Recall
5. smaller `K`
6. canonical deterministic tie-break

## 7. Search Classifier Lock

BGWO search uses **Decision Tree only**, frozen to the V0.8 governed configuration:

- `class_weight = balanced`
- `max_depth = 5`
- `min_samples_leaf = 20`
- prediction threshold = `0.5`

Logistic Regression remains post-lock comparative evaluation only.

## 8. BGWO Formulation Lock

V0.9 uses a canonical binary GWO baseline with continuous latent state and binary transfer:

Let `z_i(t)` be wolf `i` latent vector (`43` dimensions), and `b_i(t)` its binary mask.

Leadership wolves (`alpha`, `beta`, `delta`) are selected by the frozen deterministic comparator.

For each wolf and dimension:

- `D_alpha = |C1 * z_alpha - z_i|`
- `D_beta  = |C2 * z_beta  - z_i|`
- `D_delta = |C3 * z_delta - z_i|`
- `X1 = z_alpha - A1 * D_alpha`
- `X2 = z_beta  - A2 * D_beta`
- `X3 = z_delta - A3 * D_delta`
- `z_i(t+1) = (X1 + X2 + X3) / 3`

Exploration/exploitation schedule:

- `a(t) = 2 - 2 * t / T` (linear from `2` to `0`)
- `A = 2*a*r1 - a`
- `C = 2*r2`

Binary transfer and sampling:

- transfer: `S(x) = 1 / (1 + exp(-x))`
- clamp sigmoid input to `[-6, 6]`
- bit update: `b_ij(t+1) = 1 if u_ij < S(z_ij(t+1)) else 0`
- `u_ij`, `r1`, `r2` from seeded RNG

Minimum-one-feature repair:

- if mask is all-zero, set exactly one bit to `1`
- selected index = largest `|z_ij|`; deterministic tie-break = smallest feature index

Bounds/clamping:

- latent `z` values clamped to `[-6, 6]` before transfer

## 9. Search Budget and Run Count Lock

Budget is frozen before implementation and not tuned after outcomes.

- wolves (population) = `12`
- evaluated iterations including initialization = `20`
- initialization evaluation index = `0`
- iterative updates after initialization = `19`
- max fitness requests per run = `12 * 20 = 240`
- independent BGWO runs = `5`
- maximum total requests across five runs = `1200`

This preserves the same top-line budget envelope as V0.8 BPSO.

## 10. Initialization Fairness Lock

Initialization mirrors V0.8 coverage philosophy:

- anchor `K43` (all features active)
- anchor `K42` (governed pairwise-correlation 42-feature identity, bound during V0.9-C preflight)
- random exact-cardinality masks at:
  `4, 8, 11, 14, 18, 22, 26, 30, 34, 38`

Prohibited initialization leakage:

- do not seed with `BPSO-K10`
- do not seed with `MI-K11`
- do not inject BPSO winner identity into BGWO search

## 11. Seed Lock

- split seed remains `42`
- model/attack seeds remain `42, 43, 44, 45, 46`
- BGWO optimizer seeds frozen to: `2042, 2043, 2044, 2045, 2046`

## 12. Caching and Early-Stopping Lock

Duplicate/cache policy:

- cache enabled within one optimizer run
- key = canonical binary mask hash
- repeated masks reuse cached evaluation outputs
- report cache hits, unique candidates, and total requests

Early stopping policy (frozen):

- criterion: no deterministic-comparator improvement in current best
- not before evaluated iteration index `10`
- patience = `7` evaluated iterations
- otherwise stop at budget limit

## 13. Stability Reporting Lock

Across the five independent BGWO runs, report at minimum:

- selected `K` per run
- selected feature identities
- feature-selection frequency
- pairwise Jaccard
- Hamming distance summaries
- consensus and majority features
- convergence traces
- final feasibility/violation and rank descriptors
- runtime, request counts, unique candidates, cache behavior

Instability must be disclosed, not filtered.

## 14. Winner Selection and Winner Lock

Winner selection uses **validation-only** evidence and deterministic ranking.

Winner lock schema must include:

- stage and schema version
- optimizer identity (`BGWO`)
- source optimizer seed
- selected feature count `K`
- ordered feature identities
- binary mask and mask hash
- feature-list hash
- validation metrics (AP, F1, recall, etc.)
- feasibility status and normalized violation
- protocol/config identities and provenance hashes
- governance flags (no test access, no reselection)
- semantic winner lock hash

Final test is only allowed after winner lock creation and verification.

## 15. Final-Test Governance for Later V0.9 Stages

Final test campaign (outside V0.9-A) must compare locked configurations:

- `K43`
- `K42`
- `MI-K11`
- `BPSO-K10` (frozen, not rerun)
- `BGWO winner`

Final-test classifiers:

- Decision Tree
- Logistic Regression

Seed namespace remains `42..46`, and no post-test reselection is allowed.

## 16. Statistical Lock

Predictive comparison scientific unit:

- paired model/attack seed

Protocol:

- `n = 5`
- `df = 4`
- paired mean differences and 95% t confidence intervals
- no manufactured p-values

Measurement repetitions must not be treated as independent scientific samples.

## 17. Resource and Energy Planning Boundary

V0.9-A performs no resource execution. For the later V0.9 resource stage, governance must improve timing reliability relative to V0.8 sleep limitation:

- sleep disabled
- hibernate disabled
- stable AC power
- environment lock and metadata capture
- coarse campaign timestamps and observation-level (or sufficiently fine-grained) external timestamps outside timed boundaries
- fresh worker isolation
- fixed warmup/repetition protocol
- single-thread governance
- checkpoint/resume policy

Direct-energy policy remains conservative:

- default status: `DIRECT_ENERGY_UNAVAILABLE`
- no conversion of wall/CPU time to joules
- Green-AI claims use computational-resource proxy terminology unless direct backend validation passes

## 18. BPSO-vs-BGWO Comparison Plan

Comparison is metric-specific, not a single subjective score. Planned contrasts:

- winner cardinality (`K`)
- validation preservation behavior
- frozen final-test predictive behavior
- optimizer stability (Jaccard/Hamming/frequency/convergence)
- requests, unique candidates, cache behavior, runtime
- downstream structural/resource indicators in later stages

This is the scientific input to V1.0 hybrid design, not hybrid execution itself.

## 19. V1.0 Novelty Boundary (Locked)

V0.9 is an independent BGWO baseline only.

Prohibited in V0.9:

- injecting BPSO winner/population into BGWO search
- combining BGWO and BPSO trajectories
- hybrid handoff logic

Those belong to V1.0 only.

## 20. Governed V0.9 Stage Roadmap

- **V0.9-A:** BGWO scientific protocol lock (this document)
- **V0.9-B:** deterministic BGWO implementation + unit tests
- **V0.9-C:** real leakage-safe fitness integration + baseline reproduction + quarantined pilot
- **V0.9-D:** five full validation-only BGWO runs + winner lock
- **V0.9-E:** governed final-test predictive evaluation
- **V0.9-F:** controlled resource/overhead evaluation
- **V0.9-G:** notebook + figures + scientific synthesis + explicit BPSO-vs-BGWO comparison

## 21. Classification and Readiness

- **Final protocol classification:** `BGWO_PROTOCOL_LOCKED_FAIR_COMPARISON`
- **V0.9-B readiness intent:** GO after external review of this lock
- **No execution performed in V0.9-A**
