# V1.1-F1 - Resource-Efficiency / Green Evaluation Protocol Lock

Companion configuration: `config/resource_efficiency_v11f.yaml`  
Protocol classification: `RESOURCE_EFFICIENCY_GREEN_EVALUATION_PROTOCOL_LOCKED`  
Protocol version: `v1.1-f1-resource-efficiency-protocol-lock-1`  
Protocol-config SHA-256: `84f198556ce319a2775063e11960cd8cd5e5457b8fea14ac835b6f93040c71f3`

This is a protocol/preregistration artifact only. It freezes the future V1.1-F analysis but does
not implement, execute, or report that analysis.

## 1. Authoritative stage and purpose

The V1.1-A stage map defines **V1.1-F as Resource-Efficiency / Green Evaluation**. F is a read-only
analytical stage over frozen V1.1-E evidence. Its purpose is to evaluate the lightweight and
resource-efficiency characteristics of B0, B1, and P using persisted computational/resource
evidence without performing new measurements.

V1.1-F is not a new security experiment, blockchain campaign, AI training or inference stage,
TEST evaluation, optimizer, ablation, or direct electrical-energy experiment. F1 creates only this
document, its machine-readable config, focused protocol tests, and a semantic protocol lock. F2+
implementation, execution, analysis results, and reporting remain excluded.

## 2. Research question and claim boundary

> How resource-efficient/lightweight is adaptive policy P relative to fixed baselines B0 and B1
> on the governed V1.1 blockchain workload, using descriptive computational/resource proxies while
> making zero direct-energy claims?

Interpretation is endpoint-specific. No overall policy winner may be named. F must not construct a
composite score, globally rank policies, or claim universal superiority.

## 3. Frozen upstream checkpoint

Starting checkpoint: `HEAD == origin/main == d425cf1c89e391edd1a68fd2b14f3565ab487b88`.

V1.1-E is immutable upstream evidence:

- Result-lock semantic SHA-256:
  `058aeca8ac97101356bcf1c4dc5b74fb3affbda6833a85b5d423a53556cd1749`.
- Result-lock file SHA-256:
  `dd3b926eb08206f505e739ed61c5f29a3e7f03a370e9afc1a077e0b260fd6e64`.
- Experiment-protocol semantic SHA-256:
  `8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239`.
- Cells: 390 total = 345 measured + 45 E10-derived; missing=0; duplicates=0.
- TEST access=0; AI fit=0; AI inference=0.
- Energy state: `DIRECT_ENERGY_UNAVAILABLE`.

No V1.1-E artifact may be regenerated or mutated by V1.1-F.

## 4. Access and AI policy

`DATA_ACCESS = READ_ONLY_PERSISTED_ARTIFACTS_ONLY`. F may consume persisted V1.1-E artifacts. It
must not reopen raw DataCo merely to reconstruct the experiment. `TEST_ACCESS = FORBIDDEN` and the
required TEST-access count is zero.

`AI_FIT_ALLOWED = FALSE` and `AI_INFERENCE_ALLOWED = FALSE`. Model loading for new scoring,
retraining, and new risk prediction are forbidden. Existing governed AI-derived outcomes may be
referenced only as immutable upstream context.

## 5. No-new-measurement contract

All new campaigns are false: timing, memory, tracemalloc, storage, throughput, security, and energy.
E06, E07, E08, E09, and E10 must not be rerun. F derives evidence from the locked V1.1-E files;
it never invokes their measurement or experiment runners.

## 6. Policies and pairing

The only policies are B0, B1, and P. No baseline may be added, no policy may be tuned or changed,
and no policy execution occurs in F.

Matched comparisons require at minimum exact `seed` and `workload`; when present they also require
the same experiment, measurement boundary, and machine/environment. Every comparison must contain
exactly one observation for every required policy/key. A future F2 implementation must fail closed
on missing, duplicate, or unmatched pairs.

## 7. Allowed evidence

Allowed persisted evidence families are:

- Runtime: wall-clock timing and CPU process timing. Lower values mean lower runtime proxy.
- Computational work: `validation_check_count`, `validator_invocation_count`, and
  `measurable_hash_operation_count`. Lower values mean less counted computational work.
- Memory: persisted E07 fresh-process tracemalloc peak, labeled `COMPUTATIONAL_MEMORY_PROXY`.
- Storage: persisted E08 canonical serialized bytes.
- Throughput: persisted validated orders/second. Higher values mean higher throughput.

Every metric retains its original unit and boundary. Runtime and resource metrics are never
converted into physical-energy quantities.

## 8. E07 memory terminology

E07 is a **fresh-process tracemalloc peak** or **Python traced-allocation memory proxy**. It is not
RSS, total process memory, system memory, hardware memory consumption, or electrical energy.

## 9. E08 storage terminology

E08 is `POLICY_INDEPENDENT`: validation policy does not mutate the canonical chain representation.
The B0 token in the persisted representation is a serialization placeholder, not evidence that B0
uses less storage. F must not manufacture B0/B1/P storage differences or policy storage ratios.

## 10. E10 status

E10 is `DERIVED_ONLY` from E06, E07, E08, and E09. It is not an independent measurement. There is
no fourth campaign and no E10 rerun.

## 11. Direct-energy and green wording

The exact scientific state is `DIRECT_ENERGY_UNAVAILABLE`:

- CPU time != energy.
- Wall time != energy.
- Tracemalloc != energy.
- Throughput != energy.
- Storage != energy.

F must not estimate or report Joules, Wh, kWh, Watts, TDP-derived energy, carbon emissions, or CO2e.
Computational proxies cannot support claims of "energy saved", "energy reduction", "power
reduction", or "carbon reduction". A future separately governed stage with scientifically valid
direct instrumentation would be required to change this state.

In V1.1-F, "green" means resource/computational-efficiency evaluation under this limitation.
Permitted wording includes computational resource proxy, resource-efficiency evidence, lightweight
computational behavior, runtime evidence, throughput evidence, memory proxy, and storage evidence.

## 12. Governed population limitations

The frozen governed population has LOW=0, MEDIUM=4582, HIGH=6. LOW carries
`LOW_ABSENT_IN_GOVERNED_POPULATION`. HIGH evidence is sparse and the workload-100 condition carries
`HIGH_NOT_OBSERVED_IN_CONDITION`; no strong HIGH-subgroup inference is allowed. Aggregate P evidence
is overwhelmingly MEDIUM-driven.

## 13. Nested workload dependence

Within each seed, `100 subset 250 subset 500 subset 1000 subset 2500`. These sizes are nested
prefixes, not independent replicates. The 15 seed/workload cells must never be treated as 15
independent experimental replicates for inferential statistics.

## 14. Descriptive analysis plan

`ANALYTICAL_STATUS = DESCRIPTIVE_ONLY`. Allowed summaries are arithmetic mean, median, sample
standard deviation (`ddof=1`), minimum, maximum, and coefficient of variation. Coefficient of
variation is `sample_std_ddof_1 / abs(arithmetic_mean)` and is defined only when the arithmetic mean
is non-zero.

Allowed matched differences are calculated within the exact same seed/workload and metric:

- `B1_minus_B0 = metric(B1) - metric(B0)`;
- `P_minus_B0 = metric(P) - metric(B0)`;
- `P_minus_B1 = metric(P) - metric(B1)`.

A normalized policy ratio is allowed only as
`metric(numerator_policy) / metric(denominator_policy)` for an exact matched condition with a
strictly positive denominator. The numerator, denominator, unit, metric family, and beneficial
direction must be stated. E08 storage is excluded from policy ratios.

P-values, statistical-significance claims, bootstrap inference, order-level inference, and
confidence intervals presented as inferential evidence are forbidden because there are only three
fixed seeds, workloads are nested, and pseudoreplication risk is material.

## 15. Frozen security context

Security evidence remains V1.1-E evidence and is usable only as trade-off context:

- E02 detection: B0=30/135, B1=105/135, P=75/135.
- E04 unaffected-order preservation: 117315/117315 for each policy.
- E03 binary localization mirrored E02 detection and is not independent confirmatory evidence.

F must not become a new security evaluation.

## 16. Trade-off interpretation

The current frozen evidence descriptively indicates that B1 has greater governed attack-scenario
coverage with higher runtime and lower throughput; B0 has lower runtime and higher throughput with
lower governed attack coverage; and P exhibits intermediate/adaptive behavior on several endpoints.
These endpoint-specific observations do not establish an overall winner, universal superiority, or
a global policy ranking.

## 17. Future F2 fail-closed requirements

A future implementation must fail closed before producing results if any of the following occurs:

1. The expected V1.1-E result lock is absent or its semantic lock mismatches.
2. A required upstream artifact fingerprint mismatches or an upstream artifact is mutated.
3. TEST access, AI fitting, AI inference, or any new measurement campaign is attempted.
4. E10 is not `DERIVED_ONLY` from E06-E09.
5. Nested workloads are treated as independent replicates.
6. Direct-energy terminology or an unsupported energy/power/carbon claim is introduced.
7. An unexpected policy appears or the B0/B1/P contract changes.
8. A required matched comparison is absent, duplicated, or unmatched.

These are protocol requirements only; F1 does not implement F2.

## 18. Semantic protocol lock

`results/blockchain/v11f/v11f_protocol_lock.json` records the scientifically material decisions,
the document/config file fingerprints, and a canonical semantic SHA-256 using sorted keys, compact
separators, ASCII escaping, finite JSON values, UTF-8, and no trailing newline in the hashed
preimage. Unstable timestamps and machine-specific noise are excluded from the semantic payload.

## 19. F1 boundary

F1 produces exactly four protocol-layer artifacts: this document,
`config/resource_efficiency_v11f.yaml`, `tests/test_protocol_v11f.py`, and the semantic protocol
lock. It creates no `src/pipeline_v11f.py`, launcher, notebook, result summary, analysis output, or
V1.1-F execution evidence.
