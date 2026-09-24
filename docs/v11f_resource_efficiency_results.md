# V1.1-F Endpoint-Specific Resource-Efficiency Synthesis: Adaptive Policy P, Fixed Baselines B0/B1, and a Unified No-Energy-Claim Regime

Authoritative scientific synthesis for stage V1.1-F. Strict, governed descriptors
and artifact-**fingerprints** below are quoted verbatim from the frozen V1.1-F1/F2/F4
and upstream V1.1-E locks; all numeric **report values** in this document are copied
from the frozen analysis payload `results/blockchain/v11f/v11f_analysis_results.json`
or computed as descriptive mean/median/min/max over the persisted ratio arrays inside
that same payload. This document adds no measurement, no test split access, no fitted
model, no live model inference, and no new measurement campaign.

Energy evidence is declared unavailable: `DIRECT_ENERGY_UNAVAILABLE`. Therefore this
report makes no energy-consumption, no saved-energy, and no power-related claim of any
kind, and it never substitutes CPU/wall time or TDP-times-time arithmetic for any
measured physical-energy quantity.

## 1. Provenance and scope

Stage `V1.1-F5` closes the frozen V1.1-F resource-efficiency chain by (i) re-verifying
every upstream binding, (ii) reconciling each number stated here with the frozen
evidence, and (iii) recording the validation-closure artifact without creating a new
stage result lock. The F4 result lock is preserved as the only V1.1-F result lock.

The stage consumes only read-only persisted artifacts:
- upstream V1.1-E4 experiment result lock (semantic `058aeca8ac97101356bcf1c4dc5b74fb3affbda6833a85b5d423a53556cd1749`);
- protocol locks F1 (semantic `68cceedde6384c16a23226ddf082ef7d478e489c9b691e4e63d30bade85597e1`) and V1.1-E (semantic `8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239`);
- F2 analysis payload (`semantic_analysis_sha256` `0765ea4d79cdad956d09156983395c9c85fe748002cd91a084e2dc5ef6bf3899`, implementation commit `a33cb2fd7ceed03dde4be6094baa41089d707530`);
- F4 artifacts: `v11f_analysis_results.json` (`eb4ce1c69d56385a0bddc606bed40ec4d9dcc049e77a172c612a0414f2357d3a`), `v11f_execution_manifest.json` (`d1fb42d0bfb8b11267fc6572c2c333d7d77ca1b559533eff6309a128bfd599f8`), and result lock (semantic `d639be8970bd6662ed0efe688ec2c0375414942ebf636d2b958abd88d722bd1a`).

The F2 analysis `analysis_mode` is `DESCRIPTIVE_ONLY`; no inferential procedure is
applied or reported in this document.

## 2. Governed evidence base and read-only boundary

The underlying experiments are the frozen V1.1-E security/resource campaign
(seeds 522, 523, 524; workloads 100, 250, 500, 1000, 2500; policies B0, B1, P).
Each timing, memory, throughput, and storage observation is a persisted, cell-level
descriptive value aggregated from per-order run records; no measurement is repeated
here. The governing protocol forbids new timing/memory/storage/throughput/security
campaigns, forbids any TEST split access (`no TestSplit`), and forbids any AI fit or
prediction (`no fitted model`, `no live model inference`). Every governance counter in
the frozen analysis, manifest, and result lock is zero.

All values below are reported exactly as persisted (no round-trip conversion, no
unit arithmetic). The V1.1-E metric labels embed their unit identifiers:
`cpu_time_ns`/`wall_time_ns` (nanoseconds, per-order persisted runs),
`throughput_orders_per_second`, `memory_proxy_mib`, and `storage_bytes`/`canonical_serialized_bytes`
(bytes).

## 3. Metric families and proxy status

The F2 analysis defines four resource families on a shared 15-cell grid
(three seeds x five nested workloads):

| Family | Canonical metric | Source | Status |
|---|---|---|---|
| runtime | `persisted_wall_clock_timing`, `persisted_cpu_process_timing` | V1.1-E E06 | computational resource proxy |
| memory | `persisted_tracemalloc_peak_proxy` | V1.1-E E07 | `COMPUTATIONAL_MEMORY_PROXY` |
| throughput | `persisted_validated_orders_per_second` | V1.1-E E09 | computational resource proxy |
| storage | `persisted_canonical_serialized_bytes` | V1.1-E E08 | `POLICY_INDEPENDENT` (no policy comparison) |
| computational work | `measurable_hash_operation_count` | V1.1-E E10 | `DERIVED_ONLY` |

Memory evidence is the fresh-process tracemalloc peak (Python traced-allocation
memory proxy), not RSS/system memory. Storage evidence is policy-independent
canonical serialized bytes; the stored B0 token is a serialization placeholder, not
a policy comparison. E10 carries no independent measurement campaign; it is derived
only from re-verified persisted E06-E09 cell evidence.

## 4. Per-policy descriptive summaries (across the 15 governed cells)

Values are the persisted per-cell means/medians/min/max summarized across the 15 cells; timing metrics carry the V1.1-E nanosecond labels. `memory_proxy_mib` values are printed with six decimals; other metrics with three.
Metric: wall-clock process timing (`persisted_wall_clock_timing`)

| Policy                                                  | mean       | median     | min        | max        | n  |
| ------------------------------------------------------- | ---------- | ---------- | ---------- | ---------- | -- |
| B0 (fixed baseline A)                                   | 51063.607  | 49743.996  | 38510.940  | 95346.536  | 15 |
| B1 (fixed escalation / higher governed attack coverage) | 318159.839 | 315966.191 | 302736.120 | 353807.792 | 15 |
| P (adaptive)                                            | 71100.787  | 69758.644  | 60942.814  | 94350.536  | 15 |

Metric: CPU process timing (`persisted_cpu_process_timing`)

| Policy                                                  | mean       | median     | min        | max        | n  |
| ------------------------------------------------------- | ---------- | ---------- | ---------- | ---------- | -- |
| B0 (fixed baseline A)                                   | 53189.386  | 52082.356  | 40271.070  | 98935.146  | 15 |
| B1 (fixed escalation / higher governed attack coverage) | 331166.419 | 326487.428 | 314362.680 | 367318.672 | 15 |
| P (adaptive)                                            | 74081.177  | 72514.938  | 63299.350  | 98817.040  | 15 |

Metric: validated orders per second (`persisted_validated_orders_per_second`)

| Policy                                                  | mean      | median    | min       | max       | n  |
| ------------------------------------------------------- | --------- | --------- | --------- | --------- | -- |
| B0 (fixed baseline A)                                   | 26622.492 | 26408.379 | 14941.724 | 33150.911 | 15 |
| B1 (fixed escalation / higher governed attack coverage) | 4003.845  | 4167.242  | 2791.725  | 4646.619  | 15 |
| P (adaptive)                                            | 18016.101 | 18293.767 | 12370.732 | 22955.213 | 15 |

Metric: tracemalloc peak proxy (`persisted_tracemalloc_peak_proxy`)

| Policy                                                  | mean     | median   | min      | max      | n  |
| ------------------------------------------------------- | -------- | -------- | -------- | -------- | -- |
| B0 (fixed baseline A)                                   | 1.265201 | 1.265202 | 1.265200 | 1.265202 | 15 |
| B1 (fixed escalation / higher governed attack coverage) | 1.265740 | 1.265740 | 1.265738 | 1.265742 | 15 |
| P (adaptive)                                            | 1.265695 | 1.265695 | 1.265694 | 1.265696 | 15 |

Metric: hash-operation count (`measurable_hash_operation_count`)

| Policy                                                  | mean      | median   | min      | max       | n  |
| ------------------------------------------------------- | --------- | -------- | -------- | --------- | -- |
| B0 (fixed baseline A)                                   | 870.000   | 500.000  | 100.000  | 2500.000  | 15 |
| B1 (fixed escalation / higher governed attack coverage) | 15660.000 | 9000.000 | 1800.000 | 45000.000 | 15 |
| P (adaptive)                                            | 902.867   | 534.000  | 100.000  | 2585.000  | 15 |

Metric: validation-check count (`validation_check_count`)

| Policy                                                  | mean      | median    | min      | max       | n  |
| ------------------------------------------------------- | --------- | --------- | -------- | --------- | -- |
| B0 (fixed baseline A)                                   | 3480.000  | 2000.000  | 400.000  | 10000.000 | 15 |
| B1 (fixed escalation / higher governed attack coverage) | 20880.000 | 12000.000 | 2400.000 | 60000.000 | 15 |
| P (adaptive)                                            | 5254.800  | 3036.000  | 600.000  | 15090.000 | 15 |

Metric: validator-invocation count (`validator_invocation_count`)

| Policy                                                  | mean     | median   | min     | max      | n  |
| ------------------------------------------------------- | -------- | -------- | ------- | -------- | -- |
| B0 (fixed baseline A)                                   | 870.000  | 500.000  | 100.000 | 2500.000 | 15 |
| B1 (fixed escalation / higher governed attack coverage) | 2610.000 | 1500.000 | 300.000 | 7500.000 | 15 |
| P (adaptive)                                            | 873.867  | 504.000  | 100.000 | 2510.000 | 15 |

## 5. Paired descriptive contrasts (mean/median/min/max, n = 15 cells)

Metric: wall-clock process timing (`persisted_wall_clock_timing`)

| Contrast | mean        | median      | min         | max         | n  |
| -------- | ----------- | ----------- | ----------- | ----------- | -- |
| B1 − B0  | 267096.232  | 262632.651  | 212302.078  | 305155.177  | 15 |
| P − B0   | 20037.180   | 23289.640   | -24190.762  | 44606.540   | 15 |
| P − B1   | -247059.052 | -245358.500 | -288557.051 | -212533.508 | 15 |

Metric: CPU process timing (`persisted_cpu_process_timing`)

| Contrast | mean        | median      | min         | max         | n  |
| -------- | ----------- | ----------- | ----------- | ----------- | -- |
| B1 − B0  | 277977.033  | 273829.384  | 220517.690  | 316993.224  | 15 |
| P − B0   | 20891.791   | 24290.970   | -24968.432  | 46734.684   | 15 |
| P − B1   | -257085.242 | -256429.820 | -299758.358 | -219938.208 | 15 |

Metric: validated orders per second (`persisted_validated_orders_per_second`)

| Contrast | mean       | median     | min        | max        | n  |
| -------- | ---------- | ---------- | ---------- | ---------- | -- |
| B1 − B0  | -22618.646 | -22951.967 | -28736.270 | -10774.482 | 15 |
| P − B0   | -8606.390  | -9830.386  | -17957.612 | 5104.824   | 15 |
| P − B1   | 14012.256  | 14684.676  | 7909.552   | 18766.218  | 15 |

Metric: tracemalloc peak proxy (`persisted_tracemalloc_peak_proxy`)

| Contrast | mean      | median    | min       | max       | n  |
| -------- | --------- | --------- | --------- | --------- | -- |
| B1 − B0  | 0.000539  | 0.000539  | 0.000537  | 0.000541  | 15 |
| P − B0   | 0.000494  | 0.000494  | 0.000493  | 0.000496  | 15 |
| P − B1   | -0.000045 | -0.000045 | -0.000047 | -0.000043 | 15 |

Metric: hash-operation count (`measurable_hash_operation_count`)

| Contrast | mean       | median    | min        | max       | n  |
| -------- | ---------- | --------- | ---------- | --------- | -- |
| B1 − B0  | 14790.000  | 8500.000  | 1700.000   | 42500.000 | 15 |
| P − B0   | 32.867     | 34.000    | 0.000      | 85.000    | 15 |
| P − B1   | -14757.133 | -8466.000 | -42432.000 | -1700.000 | 15 |

Metric: validation-check count (`validation_check_count`)

| Contrast | mean       | median    | min        | max       | n  |
| -------- | ---------- | --------- | ---------- | --------- | -- |
| B1 − B0  | 17400.000  | 10000.000 | 2000.000   | 50000.000 | 15 |
| P − B0   | 1774.800   | 1036.000  | 200.000    | 5090.000  | 15 |
| P − B1   | -15625.200 | -8964.000 | -44928.000 | -1800.000 | 15 |

Metric: validator-invocation count (`validator_invocation_count`)

| Contrast | mean      | median   | min       | max      | n  |
| -------- | --------- | -------- | --------- | -------- | -- |
| B1 − B0  | 1740.000  | 1000.000 | 200.000   | 5000.000 | 15 |
| P − B0   | 3.867     | 4.000    | 0.000     | 10.000   | 15 |
| P − B1   | -1736.133 | -996.000 | -4992.000 | -200.000 | 15 |

## 6. Normalized ratio summaries (P / B0 and P / B1, descriptive)

Metric: wall-clock process timing (`persisted_wall_clock_timing`)

| Ratio  | mean   | median | min    | max    | n  |
| ------ | ------ | ------ | ------ | ------ | -- |
| P / B0 | 1.4507 | 1.5180 | 0.7463 | 1.8967 | 15 |
| P / B1 | 0.2241 | 0.2210 | 0.1783 | 0.2980 | 15 |

Metric: CPU process timing (`persisted_cpu_process_timing`)

| Ratio  | mean   | median | min    | max    | n  |
| ------ | ------ | ------ | ------ | ------ | -- |
| P / B0 | 1.4504 | 1.5145 | 0.7476 | 1.8973 | 15 |
| P / B1 | 0.2243 | 0.2207 | 0.1783 | 0.3004 | 15 |

Metric: validated orders per second (`persisted_validated_orders_per_second`)

| Ratio  | mean   | median | min    | max    | n  |
| ------ | ------ | ------ | ------ | ------ | -- |
| P / B0 | 0.7085 | 0.6655 | 0.4583 | 1.3416 | 15 |
| P / B1 | 4.5962 | 4.7126 | 2.7730 | 6.5905 | 15 |

Metric: tracemalloc peak proxy (`persisted_tracemalloc_peak_proxy`)

| Ratio  | mean   | median | min    | max    | n  |
| ------ | ------ | ------ | ------ | ------ | -- |
| P / B0 | 1.0004 | 1.0004 | 1.0004 | 1.0004 | 15 |
| P / B1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15 |

Metric: hash-operation count (`measurable_hash_operation_count`)

| Ratio  | mean   | median | min    | max    | n  |
| ------ | ------ | ------ | ------ | ------ | -- |
| P / B0 | 1.0392 | 1.0340 | 1.0000 | 1.0680 | 15 |
| P / B1 | 0.0577 | 0.0574 | 0.0556 | 0.0593 | 15 |

Metric: validation-check count (`validation_check_count`)

| Ratio  | mean   | median | min    | max    | n  |
| ------ | ------ | ------ | ------ | ------ | -- |
| P / B0 | 1.5104 | 1.5090 | 1.5000 | 1.5180 | 15 |
| P / B1 | 0.2517 | 0.2515 | 0.2500 | 0.2530 | 15 |

Metric: validator-invocation count (`validator_invocation_count`)

| Ratio  | mean   | median | min    | max    | n  |
| ------ | ------ | ------ | ------ | ------ | -- |
| P / B0 | 1.0046 | 1.0040 | 1.0000 | 1.0080 | 15 |
| P / B1 | 0.3349 | 0.3347 | 0.3333 | 0.3360 | 15 |

## 7. Policy-independent storage (mean canonical serialized bytes per workload)

Storage is independent of policy (E08, `POLICY_INDEPENDENT`)

| Workload (orders) | mean bytes |
| ----------------- | ---------- |
| 100               | 315028.7   |
| 250               | 787569.7   |
| 500               | 1575111.3  |
| 1000              | 3150197.7  |
| 2500              | 7875641.7  |

The stored token is a policy-independent serialization placeholder; it is not a policy comparison and carries no per-policy meaning.

## 8. Security context (frozen upstream trade-off context only)

E02 detection counts (denominator per policy = 135)

| Policy | detected |
| ------ | -------- |
| B0     | 30       |
| B1     | 105      |
| P      | 75       |

E04 unaffected-order preservation: 117315 / 117315 orders preserved. E03 binary localization outcomes mirrored E02 detection in the governed V1.1-E campaign and are not independent confirmatory evidence; detection/localization is therefore treated as a single trade-off axis.

## 9. Population composition and risk-band limitations

Governed population: 4588 orders. Risk-band counts: HIGH 6, MEDIUM 4582, LOW 0.
Risk-band evidence is overwhelmingly MEDIUM-driven. There is no LOW-band evidence in the governed population (`LOW_ABSENT_IN_GOVERNED_POPULATION`). HIGH-band evidence is sparse (six governed observations) and is treated as descriptive-only; HIGH was not observed in the workload-100 condition (`HIGH_NOT_OBSERVED_IN_CONDITION`).

## 10. Replication, nesting, and inferential scope

The 15 cell estimates share only three fixed seeds (522, 523, 524); the five workload sizes are nested within each seed (100, 250, 500, 1000, 2500) and must not be pooled as independent replicates. The analysis is `DESCRIPTIVE_ONLY`: only mean/median/min/max-style summaries are reported, no inferential procedure is applied or reported, no inferential statement is made, and no interval or resampling statement is made.

## 11. Direct-energy status and proxy constraints

Direct physical-energy measurement was unavailable (`DIRECT_ENERGY_UNAVAILABLE`).
No CPU/wall timing, no proxy count, and no TDP-times-time arithmetic is presented as
measured energy; nothing in this document asserts any saved-energy benefit, any
lower-energy use, or any energy-efficiency property. All runtime/throughput/memory
values are computational resource proxies (`COMPUTATIONAL_MEMORY_PROXY` for memory;
policy-independent
`POLICY_INDEPENDENT` serialization for storage; `DERIVED_ONLY` for computational work).

Declared limitations (each is a real constraint of the governed evidence):

1. No direct physical-energy measurement exists; the entire stage is energy-claim-free.
2. No direct-energy substitution is performed: CPU/wall time is never relabeled as
   a measured energy quantity, and TDP-times-time inference is prohibited.
3. Memory evidence is a `COMPUTATIONAL_MEMORY_PROXY` (fresh-process tracemalloc peak),
   not RSS/system memory; it is a traced-allocation proxy.
4. Storage evidence is `POLICY_INDEPENDENT`: canonical serialized bytes with a B0
   serialization placeholder token; it is not a policy comparison.
5. Computational-work evidence (E10) is `DERIVED_ONLY`: it reuses re-verified E06-E09
   persisted cells and carries no independent measurement campaign.
6. E03 binary localization outcomes mirrored E02 detection in the governed V1.1-E
   campaign and are not independent confirmatory evidence; detection/localization is a
   single axis.
7. The LOW risk band is absent from the governed population.
8. HIGH-band evidence is sparse (six governed observations) and descriptive-only;
   no HIGH-band subgroup-level inference is allowed.
9. HIGH was not observed in the workload-100 condition.
10. Workload sizes are nested within each seed and must not be pooled as independent
    replicates (15 cells are not 15 independent experiments).
11. Only three fixed seeds (522, 523, 524) were available; the shared seed/workload
    design is deliberate and finite.
12. The analysis is `DESCRIPTIVE_ONLY`: only descriptive summaries are reported; no
    inferential procedure, no probabilistic statement, no interval, and no resampling
    statement is made.
13. Results apply to the governed protocol design and environment (40,000-row cap,
    order-grouped 28k/6k/6k split, 43-feature canonical space, V1.1-E implementation,
    the B0/B1/P policy set, and the five nested workloads above).

## 12. Endpoint-specific conclusions and policy stance

The F1 protocol forbids a global policy selection, and this report honors that
bound: it does not name a single all-purpose preferred policy, does not produce any
ordering index or single-value aggregation, and never selects a policy. Instead it states,
for the governed population and the three resource families, which policy leads on
which specific endpoint while acknowledging the reverse on other endpoints:

- **Runtime lightness.** B0 leads: CPU process timing mean `53189.386` and wall-clock
  mean `51063.607` are the lightest; B1 is the heaviest (CPU mean `331166.419`, wall
  mean `318159.839`); P is intermediate (CPU mean `74081.177`, wall mean `71100.787`) and
  is lighter than B1 (CPU ratio P/B1 mean `0.2243`) yet not lighter than B0
  (CPU ratio P/B0 mean `1.4504`). A lighter runtime read on one fixed policy and
  a heavier one on the other is an endpoint-level trade-off, not a global result.
- **Throughput capacity.** B0 leads with mean `26622.492` validated orders per second;
  B1 is the lowest (mean `4003.845`); P is intermediate (mean `18016.101`; ratio P/B1
  mean `4.5962`, P/B0 mean `0.7085`). Throughput and runtime point the same
  way across the three policies.
- **Memory footprint.** The `COMPUTATIONAL_MEMORY_PROXY` values are nearly identical
  across policies (B0 `1.265201`, B1 `1.265740`, P `1.265695`); paired contrasts are
  three to four orders smaller than the level. Memory is effectively policy-flat in
  this governed design, so no memory-driven policy trade-off is claimed.
- **Computational load and validation work.** B1 carries the heavier load (hash
  sequence mean `15660.000` versus B0 `870.000`; paired mean `14790.000`), while
  P closely tracks B0 (P/B0 ratio mean `1.0392`). B1 consequently records the
  most validation checks and validator invocations; P records slightly more than B0.
- **Storage.** Storage is `POLICY_INDEPENDENT` and grows linearly with workload
  (mean `315028.7` bytes at 100 orders through `7875641.7` bytes at 2500 orders);
  there is no storage policy trade-off.
- **Security (frozen context).** B1 has the higher governed attack-scenario coverage
  (E02 detected `105` of `135`) versus B0 (`30` of `135`), with
  P intermediate (`75` of `135`); E04 order preservation is `117315` /
  `117315`.

In summary: no single policy leads on every endpoint category in the governed
population — B0 leads on runtime lightness and throughput while carrying lower
governed attack coverage; B1 leads on governed attack coverage while being the
heaviest and slowest; P sits intermediate on runtime/throughput/security and nearly
ties B0 on the memory proxy. These are endpoint-specific, descriptive observations
within the governed V1.1-E design; they are not evidence of any energy benefit,
not a global ordering, and not a policy recommendation. No final stage lock is created
(F4 result lock preserved).

## 13. Reproducibility, data policy, and follow-on governance

- Frozen identifiers: F1 protocol lock semantic `68cceedde6384c16a23226ddf082ef7d478e489c9b691e4e63d30bade85597e1`; F1 config `84f198556ce319a2775063e11960cd8cd5e5457b8fea14ac835b6f93040c71f3`; F1 protocol document `8e20ac45291fe5568b5d1e10a3671faec71e3db4bb191c8a299ca446d68eaaa1`; F2 implementation commit `a33cb2fd7ceed03dde4be6094baa41089d707530` (file `851478e2a5994543c15e5badd1614a7e6d6687a081f8d0fbd9d37dfe858e60a2`); F2 semantic analysis `0765ea4d79cdad956d09156983395c9c85fe748002cd91a084e2dc5ef6bf3899`; F4 analysis `eb4ce1c69d56385a0bddc606bed40ec4d9dcc049e77a172c612a0414f2357d3a`; F4 manifest `d1fb42d0bfb8b11267fc6572c2c333d7d77ca1b559533eff6309a128bfd599f8`; F4 result lock semantic `d639be8970bd6662ed0efe688ec2c0375414942ebf636d2b958abd88d722bd1a`; V1.1-E result lock file `dd3b926eb08206f505e739ed61c5f29a3e7f03a370e9afc1a077e0b260fd6e64` / semantic `058aeca8ac97101356bcf1c4dc5b74fb3affbda6833a85b5d423a53556cd1749`; V1.1-E protocol lock semantic `8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239`.
- This document is generated deterministically from the frozen analysis payload by
  `.venv/bin/python -m scripts.run_v11f5 report`. The validation-closure artifact
  `results/blockchain/v11f/v11f5_validation.json` records `report_file_sha256` and a
  semantic validation hash recomputed from the same pins.
- No new measurement was performed, no TEST split was accessed (`no TestSplit`), no
  fitted model exists in this stage, and no live model inference is made.
- Reproduction: re-running the report/verification/persistence commands at
  authorized commit `d66661248392cf775bfd6761590478fdf49641ea` must reproduce the
  exact report bytes and the exact semantic validation hash.
