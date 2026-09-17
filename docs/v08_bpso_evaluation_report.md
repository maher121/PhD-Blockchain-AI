# V0.8-F BPSO Evaluation Report

- **Project:** A Lightweight and Green Blockchain-AI Framework for Secure and Energy-Efficient Supply Chain Management
- **Stage:** V0.8-F (artifact-only notebook, publication figures/tables, and scientific synthesis)
- **Evidence scope:** frozen V0.8-A, V0.8-B, V0.8-C, V0.8-D, and V0.8-E artifacts
- **Execution policy:** presentation and synthesis only; no reruns of BPSO, predictive evaluation, or resource benchmark

## Abstract

This report synthesizes governed V0.8 evidence after completion of V0.8-E (`E4_COMPLETE`). The central question is whether the locked BPSO winner (`BPSO-K10`) provides defensible lightweight advantages while preserving predictive behavior under frozen evaluation governance. Across five governed optimizer runs, BPSO selected subsets with mean cardinality 11.8 (SD 1.483), and the locked winner from seed 1042 fixed a universal 10-feature subset.

Frozen V0.8-D final-test evidence shows Decision Tree preservation against the 43-feature baseline with small relative losses (AP about 1.11%, F1 about 0.92%, recall about 2.18%), while Logistic Regression remains weak in absolute predictive quality (for K10: AP about 0.05073, F1 about 0.08694, ROC-AUC about 0.48986). Frozen V0.8-E4 resource evidence shows major structural reductions for K10, including deterministic training input memory drop from 9,632,132 bytes (K43) to 2,240,132 bytes and inference input memory drop from 2,064,132 bytes to 480,132 bytes. BPSO-K10 is the smallest locked representation, but it is not universally superior on all metrics.

Timing evidence indicates potential downstream computational advantages (for example, Decision Tree training and inference means favor K10 vs K43), but timing interpretation remains indicative rather than definitive due to the documented Modern Standby limitation. The final governed classification therefore remains `STRUCTURAL_GAIN_TIMING_UNCERTAIN`.

## 1. Governance and Evidence Boundaries

V0.8-F uses only frozen artifacts. The following actions were not performed:

- no BPSO rerun
- no feature reselection
- no model retraining for new scientific evidence
- no rerun of V0.8-D governed final-test campaign
- no rerun of the V0.8-E benchmark campaign
- no outlier deletion
- no lock modification

Governed lock identities verified in V0.8-F preflight:

- `results/bpso/v08c_winner_lock.json`
- `results/bpso/v08d_final_test_lock.json`
- `results/bpso/v08e_e4_analysis/v08e_scientific_result_lock.json`

Verified semantic hashes:

- V0.8-D semantic result lock: `ac19e5a0dba5d058f88485e6ab8ab9a96f97c30f697e95a744ccaa6563d2b657`
- V0.8-E semantic result lock: `7307fda1cdb5f100cb6268dc0f4eff05b65bfcefadeb06707c8a3e13fd6888d5`

Frozen historical hash integrity checks remained valid.

## 2. V0.8 Pipeline Traceability

- **V0.8-A:** BPSO protocol validation (synthetic protocol checks, no scientific claims)
- **V0.8-B:** Real-fitness preflight and leakage governance
- **V0.8-C:** Five governed BPSO runs and winner lock
- **V0.8-D:** Governed final-test predictive evaluation and preservation lock
- **V0.8-E:** Resource benchmark + E4 analysis lock (`E4_COMPLETE`)
- **V0.8-F:** notebook + tables/figures + synthesis report (this document)

## 3. BPSO Optimization Outcomes (V0.8-C)

Five governed BPSO runs produced the following locked run-level outcomes:

- seed 1042: K10, AP about 0.230285, F1 about 0.203308, recall about 0.318667
- seed 1043: K14, AP about 0.236819, F1 about 0.215738, recall about 0.317333
- seed 1044: K12, AP about 0.234175, F1 about 0.189298, recall about 0.326667
- seed 1045: K12, AP about 0.236521, F1 about 0.208166, recall about 0.323333
- seed 1046: K11, AP about 0.229647, F1 about 0.205077, recall about 0.316000

Stability descriptors:

- mean selected K: 11.8
- SD selected K: 1.483
- mean pairwise Jaccard: 0.2198

These values indicate optimizer subset-selection variability, not optimizer failure.

Locked winner identity (seed 1042, K10) features:

1. `order_item_quantity`
2. `order_item_profit_ratio`
3. `order_item_total`
4. `hour`
5. `Type_PAYMENT`
6. `Market_LATAM`
7. `Shipping Mode_Same Day`
8. `Customer Segment_Corporate`
9. `Department Name_Apparel`
10. `Department Name_Health and Beauty`

## 4. Final-Test Predictive Evidence (V0.8-D)

Decision Tree means for K43 and K10:

- K43: AP 0.222084, F1 0.189005, recall 0.305333, precision 0.265385, ROC-AUC 0.603533
- K10: AP 0.219612, F1 0.187258, recall 0.298667, precision 0.245874, ROC-AUC 0.599909

Relative K10 losses vs K43 (Decision Tree):

- AP about 1.11%
- F1 about 0.92%
- recall about 2.18%

All frozen preservation criteria passed under the governed descriptive margins.

Logistic Regression (K10) remains weak in absolute predictive quality:

- AP about 0.05073
- F1 about 0.08694
- ROC-AUC about 0.48986

Therefore LR is interpreted as a robustness comparator for resource behavior, not strong standalone cybersecurity detection.

## 5. Structural Resource Results (V0.8-E4)

Locked feature counts:

- K43 = 43
- K42 = 42
- MI-K11 = 11
- BPSO-K10 = 10

Reduction levels:

- K10 vs K43: about 76.74%
- K10 vs K42: about 76.19%
- K10 vs K11: about 9.09%

Deterministic selected-input memory highlights:

- training input memory: K43 `9,632,132` bytes vs K10 `2,240,132` bytes
- inference input memory: K43 `2,064,132` bytes vs K10 `480,132` bytes

Model-size interpretation boundary:

- K10 does not uniformly minimize serialized model size.
- MI-K11 remains slightly smaller than K10 for serialized models in key cases.

## 6. Timing Results and Statistical Presentation

Scientific unit and interval protocol are fixed by E4:

- unit: seed-level means
- `n = 5`, `df = 4`, `t_critical = 2.776445`
- 95% paired confidence intervals only
- no new p-values

Timing evidence is interpreted as **indicative** (not definitive). Examples:

- DT training wall: K43 about 0.14042 s vs K10 about 0.04298 s (indicative reduction about 66.80%)
- DT inference latency: K43 about 0.002762 s/op vs K10 about 0.001823 s/op (indicative reduction about 31.74%)
- DT throughput: K43 about 2.364M records/s vs K10 about 3.385M records/s (indicative increase about 48.12%)
- LR training wall: indicative reduction about 79.64%
- LR inference direction remains uncertain where paired CI includes zero

Wording boundary used throughout:

- paired CI excludes zero
- paired CI includes zero
- direction consistent across seeds
- direction uncertain across seeds

## 7. RSS Interpretation

Absolute RSS and incremental RSS are presented separately:

- absolute RSS remained stable and is not invalidated solely by sleep events
- incremental RSS is sampling-sensitive and not overinterpreted

## 8. Optimization Overhead and Break-Even

Imported one-time V0.8-C overhead (not rerun):

- optimizer runs: 5
- fitness requests: 972
- Decision Tree fits: 4860
- core wall: `645.5028752319995 s`
- core CPU: `645.218273799 s`
- pipeline wall: `857.1113503500001 s`

Break-even results are represented as `DERIVED_DESCRIPTIVE_TIMING_LIMITED`:

- APPLICABLE rows: 40
- NOT_APPLICABLE rows: 8 (all LR K10-vs-K11)

These are descriptive planning indicators, not guaranteed operational thresholds.

## 9. Tradeoff and Green-AI Interpretation

Post-lock tradeoff analysis is descriptive and unweighted. No objective weighting or composite score was used.

Interpretation:

- BPSO-K10 provides strongest structural compactness.
- MI-K11 remains an important comparator with slightly stronger predictive behavior and slightly smaller model artifacts in some cases.
- Decision Tree remains the primary predictive classifier in this governed setting.

Direct-energy boundary is unchanged:

- `DIRECT_ENERGY_UNAVAILABLE`

Therefore this stage reports computational-resource efficiency and Green-AI proxy indicators only; it does not report Joules, Watts, kWh, CO2, or CO2e.

## 10. Sleep / Modern-Standby Limitation

Windows Modern Standby occurred during portions of the E3 campaign.

- 42 reconstructed cell-execution windows (conservatively 420 observations) intersected recorded Modern Standby intervals.
- confirmed observation-level suspend overlap was 0 because observation-level timestamps were unavailable.

Required interpretation statement:

> Host Modern Standby occurred during portions of the campaign, and observation-level timestamps were unavailable; therefore timing-based resource results are interpreted as indicative rather than definitive.

This does not imply that 420 measurements were corrupted; it constrains timing certainty.

## 11. Main Scientific Finding

The defensible central conclusion is:

**BPSO-K10 produced the smallest locked feature representation, reducing the 43-feature baseline by approximately 76.74%, while satisfying the frozen predictive-preservation criteria.**

Its strongest advantages are structural dimensionality and deterministic input-memory reduction. Decision Tree timing measurements indicate potential downstream computational advantages, but those timing results remain indicative rather than definitive because of the documented Modern Standby limitation. MI-K11 remains an important comparator because it provides slightly stronger predictive behavior and slightly smaller serialized models in some cases.

## 12. Limitations

1. Timing interpretation is constrained by the sleep-audit limitation.
2. Direct electrical energy is unavailable, so no measured energy/carbon claims are possible.
3. LR absolute predictive quality is weak and should not be overgeneralized.
4. Seed-level `n=5` supports descriptive CI reporting but limits uncertainty resolution.
5. Tradeoff analysis is descriptive post-lock and not a formal multi-objective optimization step.

## 13. Reproducibility and Artifacts

Notebook:

- `notebooks/v08_bpso_feature_selection_evaluation.ipynb`

Notebook-generated publication directories:

- `results/bpso/v08f_reporting/tables`
- `results/bpso/v08f_reporting/figures`

Companion report:

- `docs/v08_bpso_evaluation_report.md`

The notebook uses governed artifact loading and lock/hash verification, then generates publication tables/figures without invoking experimental reruns.

Representative figure links:

- ![BPSO run feature counts](../results/bpso/v08f_reporting/figures/01_bpso_selected_feature_count_by_run.png)
- ![BPSO convergence](../results/bpso/v08f_reporting/figures/02_bpso_convergence.png)
- ![Feature frequency](../results/bpso/v08f_reporting/figures/03_feature_selection_frequency.png)
- ![Feature reduction](../results/bpso/v08f_reporting/figures/04_feature_count_reduction_comparison.png)
- ![Predictive comparison](../results/bpso/v08f_reporting/figures/05_predictive_comparison.png)
- ![Input memory comparison](../results/bpso/v08f_reporting/figures/06_input_memory_comparison.png)
- ![Model-size comparison](../results/bpso/v08f_reporting/figures/07_model_size_comparison.png)
- ![Decision Tree resource comparison](../results/bpso/v08f_reporting/figures/08_decision_tree_resource_comparison.png)
- ![Timing variability](../results/bpso/v08f_reporting/figures/09_timing_variability.png)
- ![Predictive-resource tradeoff](../results/bpso/v08f_reporting/figures/10_predictive_resource_tradeoff.png)
