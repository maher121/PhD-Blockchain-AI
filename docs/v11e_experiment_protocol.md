# V1.1-E2B — Final Experiment Protocol Lock (E01–E10)

Stage: V1.1-E2B   Kind: V11E2_EXPERIMENT_PROTOCOL_LOCK
Protocol version: v1.1-e2b-final-experiment-protocol-lock-1
Upstream lock stage: V1.1-A  | Risk stage: V1.1-D3  | Integration readiness: V1.1-E1
Scientific matrix resolution: V1.1-E2A (FROZEN, GO)

This document freezes the E01–E10 mixed-reliability experiment protocol for
the governed V1.1 blockchain workload. It is a protocol/PREREgistration
artifact ONLY. It does not execute experiments, benchmark policies, generate
results, access TEST, fit/infer AI, or measure energy.

---

## 1. Frozen global design

Governed population: 4588 canonical governed orders.
Risk distribution (frozen, NOT rebalanced):
- LOW = 0
- MEDIUM = 4582
- HIGH = 6
Validation rows: 6000.
Seeds: blockchain seed family [522, 523, 524].
Workloads (order_counts, nested): [100, 250, 500, 1000, 2500].
Repetitions: 3 (mapped deterministically to the three blockchain seeds).
Sampling: deterministic, platform-independent, no replacement.

Do NOT:
- stratify by risk
- oversample HIGH
- synthesize LOW
- rebalance the governed distribution
- add 4588 as an additional workload
- change the risk aggregation rule (MAX), classifier, AI seeds, or ALV

---

## 2. Frozen sampling contract

For each blockchain seed, every canonical governed order_id is ranked by
SHA-256 over a canonical deterministic representation that includes the
seed and the order_idholmes, using the repository canonical serialization
rules. Digest ties are broken by canonical numeric order_id.

Select the first N order_ids WITHOUT replacement, for each N in
[100, 250, 500, 1000, 2500].

Workloads are nested prefixes of the same seed-specific ranking:
100 < 250 < 500 < 1000 < 2500.

The SAME {seed, N} sample is used across:
- B0, B1, P (matched policies)
- every experiment where that workload applies (E01–E05 functional cells)

No stratification by risk band; no oversampling of HIGH; no synthetic LOW.

---

## 3. Frozen policies

- B0 (lightweight baseline): validation checks c1–c4, single validator.
- B1 (full frozen validation): fixed quorum = 3, full frozen validation
  (c1–c8), 3 validators.
- P (adaptive AI-driven ALV, GOVERNED_AI_RISK):
  - LOW: c1–c4 / 1 validator
  - MEDIUM: c1–c6 / 1 validator
  - HIGH: c1–c8 / 3 validators

Fail-safe behavior and the c1–c8 check set remain as frozen upstream.
LOW is absent in the governed population (LOW = 0); the LOW band is reported
with the marker `LOW_ABSENT_IN_GOVERNED_POPULATION` and is never estimated,
imputed, or rebalanced. MEDIUM = 4582 dominates the governed population.
HIGH = 6 is used as naturally observed only; per-seed/per-workload exact
HIGH counts are persisted; if HIGH count == 0 the marker
`HIGH_NOT_OBSERVED_IN_CONDITION` is used and no HIGH-specific estimate is
made; if HIGH > 0 it is reported descriptively with its exact n.

---

## 4. Attacks (frozen upstream, reused)

PA-01 through PA-09 (frozen upstream attack scenarios) are reused verbatim.
For each {seed, workload, attack}, a deterministic attack instance is
generated once and the SAME attacked copies are reused across B0/B1/P where
applicableholmes. For PA-08/PA-09 the tamper targets a donor order that is
read-only; donors are never mutated.

---

## 5. E01–E10 experiment matrix

- E01 Functional correctness: B0/B1/P, all workloads, NO attacks. Primary:
  clean verification pass count/rate, valid-chain construction count.
  Secondary: deterministic hash reproduction, deterministic verdict
  reproduction, clean false-rejection count, policy-consistent check count,
  validator count. Timing: secondary descriptive. Integrity detection: NO.
- E02 Integrity/tamper detection: B0/B1/P, all workloads, PA-01–PA-09.
  Primary: policy-specific detected count + detection rate. Secondary:
  false-acceptance count, false-acceptance rate, failed-check distribution,
  validation time. Integrity detection: YES. Timing: secondary descriptive.
  Detection means THAT POLICY rejects the attacked chain (policy-specific
  verdict, not a generic PA harness).
- E03 Tamper localization: B0/B1/P, all workloads, PA-01–PA-09. Primary:
  unconditional correct-localization count + rate, with DENOMINATOR = ALL
  injected attacks (an undetected attack counts as NOT localized).
  Secondary: conditional localization among detected, localization
  granularity, affected-order identification, validation time. Integrity
  detection: YES. Timing: secondary descriptive.
- E04 Order-level fault isolation: B0/B1/P, all workloads, PA-01–PA-09.
  Each attack condition has exactly one affected target chain; PA-08/PA-09
  keep donor read-only and the target is the affected chain. Primary:
  unaffected-order preservation count + rate. Secondary: cross-order
  propagation count, target-chain verdict, validation time. Integrity
  detection: YES. Timing: secondary descriptive.
- E05 AI-linked adaptive behavior: policy P (principal), matched B0/B1
  reference, all workloads, NO attacks. Principal evidence MUST use
  governed AI risk. Primary: applied validation-check allocation by governed
  risk level, validator allocation by governed risk level, validation-level
  distribution. Secondary: matched runtime, check-count consequences,
  validator-count consequences, quorum-frequency consequences. Timing:
  secondary descriptive. Integrity detection: NO (E02 measures detection).
- E06 Execution-time overhead: B0/B1/P, all workloads, NO attacks. Primary:
  validation-only wall-clock runtime. Secondary: process CPU time,
  validation-check count, validator count, per-order validation time.
  Timing: YES (PRINCIPAL). Integrity detection: NO.
- E07 Memory overhead: B0/B1/P, all workloads, NO attacks. Primary: peak
  traced Python allocation during validation (tracemalloc, fresh worker
  process, MiB; labeled COMPUTATIONAL_MEMORY_PROXY, NOT whole-system RSS).
  Secondary: paired allocation increment vs B0, allocation per order.
  Timing: NO. Integrity detection: NO.
- E08 Storage overhead: policy-independent (validation policy does not
  mutate canonical chain representation). All workloads, NO attacks.
  Primary: canonical serialized blockchain bytes per workload. Secondary:
  bytes/order, bytes/block, separately identified off-chain artifact bytes.
  Exclude the optional full-payload variant from the principal measure.
  Timing: NO. Integrity detection: NO.
- E09 Throughput: B0/B1/P, all workloads, NO attacks. Primary: validated
  orders/second. Secondary: validated blocks/second, total validated
  orders, total validated blocks. Timing: YES. Integrity detection: NO.
- E10 Resource/green proxy: derived ONLY from E06–E09 plus deterministic
  operation counts; MUST NOT launch a fourth independent performance
  campaign. Primary computational proxies: runtime, validation-check count,
  validator-invocation count, measurable hash-operation count. Secondary:
  memory evidence (E07), storage evidence (E08), throughput evidence (E09),
  normalized proxy differences vs baselines. DIRECT_ENERGY_UNAVAILABLE is
  mandatory. Never report Joules/Wh/TDP×time/physical energy savings.

---

## 6. Timing boundary (frozen)

- Materialize chains, AI references, and attack instances before timing.
- Wall clock: time.perf_counter_ns. CPU: time.process_time_ns.
- START = immediately before calling the validation policy.
- STOP = immediately after the complete validation policy verdict is returned.
- EXCLUDE: chain construction, AI generation, AI inference, attack
  generation/injection, canonical serialization prep, result serialization,
  and result-file writing.
- One unmeasured synthetic warm-up per policy. One thread. Counterbalance
  policy execution order across seeds. Identical boundary for B0/B1/P.
- E01–E05: timing is secondary descriptive only. E06 is the principal
  timing experiment.

## 7. Result aggregation (frozen)

- Persist EVERY raw run observation before aggregation.
- Continuous measurements: arithmetic mean, median, sample std (ddof=1),
  min, max. Matched-policy raw paired differences: B1−B0, P−B0, P−B1.
- Rates: numerator + denominator per run; pooled descriptive numerator +
  denominator (no inference).
- NO confidence intervals, NO significance tests, NO bootstrap intervals,
  NO order-level inferential tests (only three fixed seed repetitions;
  avoid pseudoreplication).
- NEVER report Joules, Wh, TDP×time, or physical energy savings
  (`DIRECT_ENERGY_UNAVAILABLE`); E10 uses computational/resource proxies
  only.

## 8. Fairness contract

Matched B0/B1/P comparisons share: exact order IDs, exact chain content,
exact canonical serialization, exact schema, exact SHA-256 policy, exact
attack instance, exact seed, exact workload size, exact timing boundary,
same machine/environment. Only the validation policy differs.

## 9. Output schemas (frozen paths, future E3)

results/blockchain/v11e/
- Raw run observations: v11e_raw_run_observations/
- Attack instance manifest: v11e_attack_instance_manifest.json
- Workload manifest: v11e_workload_manifest.json
- Experiment results: v11e_experiment_results/<stage>/*.json
- Aggregated descriptive summaries: v11e_aggregated_summaries/
- Execution/result lock: v11e_experiment_result_lock.json (E3+)

## 10. Governance bindings (semantic SHAs)

- V1.1-A protocol lock semantic: aaf3ac3139e8296fb7f976a8a7ed9f18eb317af2f73b10cd0208b6bb870ebd8d
- V1.1-C mapping semantic: c992429b2165f18649d36b0339260d9a95e1ae8524d7dfd586a168504d3cc765
- V1.1-D3 result lock semantic: 440d351f4254cc74c2779b35fd161acecf452b071448a3592131d19ed87f5641
- V1.1-D3 artifact (semantic): ab7b1c8ef82035dc01005d8ac3ebaad379a77dd4bed743dbee5cae37b573186e
- V1.1-D3 artifact (disk): 70b1d74c800baaa5d060a5d22739b4d142a578523a847a094899cb9746239b7b
- V1.1-D3 summary (semantic): d91c0102273a55379bc498d82323b670be523044f49de74c9f8234e8559afc6c
- V1.1-D3 summary (disk): 99adcc25588e4bf431e61c52c68b6f06907bd55d7df32daf67852b39ece5007d
- V1.1-E1 integration readiness: stage V1.1-E1, marker
  V11E1_INTEGRATION_READINESS_COMPLETE_REVIEW_REQUIRED, guarded
  EXPECTED_HEAD 511490248cce0bd683c4beed21f6d9cb35fd70a8
- Starting checkpoint (E2B): HEAD == origin/main ==
  415e58ce260251dbbf1efbff9bbaab0a23c86a9c
- Scientific matrix resolution: V11E2A_SCIENTIFIC_MATRIX_RESOLVED_GO
