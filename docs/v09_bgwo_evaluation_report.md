# V0.9-G BGWO Evaluation Report

- **Project:** A Lightweight and Green Blockchain-AI Framework for Secure and Energy-Efficient Supply Chain Management
- **Stage:** V0.9-G (artifact-only notebook, publication figures/tables, and scientific synthesis)
- **Evidence scope:** frozen V0.8-C, V0.8-D, V0.9-D, V0.9-E, and V0.9-F artifacts
- **Execution policy:** presentation and synthesis only; no reruns of BGWO/BPSO, predictive evaluation, or resource benchmark

## Abstract

This report synthesizes governed V0.9 evidence for the BGWO feature-selection evaluation after completion of V0.9-F (`RESOURCE_RESULT_LOCKED`). The central question is whether the locked BGWO winner (`K=14`, seed 2042) provides defensible lightweight and predictive behavior under frozen evaluation governance, benchmarked against the frozen references `BPSO-K10`, `MI-K11`, `K42`, and `K43`.

Across five governed BGWO runs (seeds 2042-2046), selected subsets had cardinalities 14, 22, 25, 20, 22 (mean 20.6, SD 4.099, median 22, range 14-25) with mean pairwise Jaccard stability 0.3634 (min 0.1613). The locked winner fixed a universal 14-feature subset.

Frozen V0.9-E final-test evidence shows Decision Tree means closest to the winner: for BGWO, AP 0.226035, F1 0.179717, recall 0.308000. The only held-out paired comparison whose 95% CI excludes zero is BGWO vs BPSO-K10 on Decision Tree AP (−6.42e-3 ± CI [0.00131, 0.01154]), which favors BGWO; all other comparisons are direction-uncertain. Logistic Regression remains weak in absolute predictive quality for every configuration (BGWO: AP 0.050810, F1 0.088486, recall 0.468000).

Frozen V0.9-F evidence is computational only: `DIRECT_ENERGY_UNAVAILABLE`, no TDP-based Joule estimation, and timing is indicative rather than definitive (Modern Standby-limited). BGWO shows large deterministic structural reductions versus K43 (67.44% feature reduction, Decision Tree training input memory 3,136,132 bytes vs 6,635,332, Decision Tree serialized model 3,812 bytes vs 4,105), comparable to BPSO-K10, which remains the smallest locked representation.

The final governed classification remains `STRUCTURAL_GAIN_TIMING_UNCERTAIN_LOW_PREDICTIVE_DELTA`, reported per metric with no overall winner asserted.

## 1. Governance and Evidence Boundaries

V0.9-G uses only frozen artifacts. The following actions were not performed:

- no BGWO or BPSO rerun
- no feature reselection or tuning
- no model retraining for new scientific evidence
- no rerun of the V0.9-D governed validation runs
- no rerun of the V0.9-E governed final-test campaign
- no rerun of the V0.9-F resource campaign
- no outlier deletion
- no lock modification

Frozen source evidence paths verified in V0.9-G preflight (46 files snapshotted with sha256 before and after notebook execution; hashes unchanged):

- `results/bgwo/v09d_winner_lock.json`
- `results/bgwo/v09d_stability.json`
- `results/bgwo/v09d_convergence.json`, `v09d_execution_summary.json`, `v09d_validation_lock.sha256`
- `results/bgwo/v09e_result_lock.json`, `v09e_test_summary.json`, `v09e_test_metrics.csv`, `v09e_paired_statistics.csv`, `v09e_feature_overlap.json`, `v09e_preservation.json`
- `results/bgwo/v09f_result_lock.json`, `v09f_resource_summary.json`, `v09f_model_size.json`, `v09f_tradeoff.json`
- `results/bpso/v08c_winner_lock.json`, V0.6/V0.8 artifacts, `data/processed/dataset_metadata.json`

**Transparency note (frozen artifact quirk).** Runs 2043 and 2044 record identical best validation means (AP 0.235588513402709, F1 0.20743616729790348, recall 0.32066666666666666) on different masks (`c88d7354b053…` vs `ab272ed8bfeb…`) and different K (22 vs 25). This is the genuine frozen V0.9-D artifact data (verified as artifact content, not a V0.9-G loader artifact). It is reported as-is.

## 2. V0.9 Pipeline Traceability

- **V0.8-C:** governed BPSO winner lock (K10, seed 1042) - frozen reference
- **V0.9-D:** protocol validation, five governed BGWO runs, stability/convergence analysis, winner lock
- **V0.9-E:** governed final-test predictive evaluation, preservation and feature-overlap locks
- **V0.9-F:** resource benchmark + model-size/tradeoff analysis (`RESOURCE_RESULT_LOCKED`)
- **V0.9-G:** notebook + tables/figures + synthesis report (this document)

## 3. BGWO Optimization Outcomes (V0.9-D)

Five governed BGWO runs produced the following locked run-level outcomes:

- seed 2042: K14, AP 0.234618, F1 0.189644, recall 0.322667
- seed 2043: K22, AP 0.235589, F1 0.207436, recall 0.320667
- seed 2044: K25, AP 0.235589, F1 0.207436, recall 0.320667
- seed 2045: K20, AP 0.230799, F1 0.214693, recall 0.318667
- seed 2046: K22, AP 0.233952, F1 0.207636, recall 0.323333

Stability descriptors:

- mean selected K: 20.6; SD 4.099; median 22; range 14-25
- mean pairwise Jaccard: 0.3634 (min 0.1613); consensus used for winner selection
- feature-frequency table (43 rows) records each feature's selection count and selection frequency; features selected by all five runs (`selection_count = 5`): `order_item_total`, `Market_LATAM`, `Department Name_Outdoors`

Convergence aggregates (17 evaluated iterations) show feasible run count declining from 5 to 2 as runs early-stop (`EARLY_STOP_NO_IMPROVEMENT`); per-run best iteration 0, 9, 5, 4, 0; best-search fits 660, 1020, 780, 720, 660; candidate requests 132, 204, 156, 144, 132. Aggregate (`ALL`): 768 candidate requests, 3840 Decision Tree fits, stop reason `AGGREGATE_WINNER`.

These values indicate optimizer subset-selection variability, not optimizer failure.

Locked winner identity (seed 2042, K14) features:

1. `order_item_quantity`
2. `product_price`
3. `order_item_discount`
4. `order_item_discount_rate`
5. `order_item_total`
6. `days_schedule`
7. `is_weekend`
8. `item_count_per_order`
9. `Market_LATAM`
10. `Market_Pacific Asia`
11. `Market_USCA`
12. `Department Name_Outdoors`
13. `Department Name_Discs Shop`
14. `Department Name_Book Shop`

## 4. BPSO-K10 Reference Context (frozen V0.8-C)

Frozen reference values used throughout V0.9-G:

- BPSO-K10 (seed 1042): validation AP 0.230285, F1 0.203308, recall 0.318667

Language in this report is comparative and metric-specific; no overall winner, score, or ranking is asserted.

## 5. Feature Overlap (V0.9-E evidence)

- BGWO (K14) vs BPSO-K10: intersection 3, union 21, Jaccard 0.142857; shared `order_item_quantity`, `order_item_total`, `Market_LATAM`.
- BGWO (K14) vs MI-K11 (mean of the 5 MI seeds, seed-specific): mean intersection 6.4, mean union 18.6, union Jaccard 0.347826.

## 6. Final-Test Predictive Evaluation (V0.9-E)

Decision Tree final-test means (n=5 outcome/attack seeds):

- K43: AP 0.222084, F1 0.189005, recall 0.305333, ROC-AUC 0.603533
- K42: AP 0.222084, F1 0.189005, recall 0.305333, ROC-AUC 0.603533
- MI-K11: AP 0.224941, F1 0.192121, recall 0.300000, ROC-AUC 0.605495
- BPSO-K10: AP 0.219612, F1 0.187258, recall 0.298667, ROC-AUC 0.599909
- BGWO: AP 0.226035, F1 0.179717, recall 0.308000, ROC-AUC 0.605750

Logistic Regression final-test means:

- K43: AP 0.051150, F1 0.087348, recall 0.423333, ROC-AUC 0.494000
- K42: AP 0.051150, F1 0.087348, recall 0.423333, ROC-AUC 0.493999
- MI-K11: AP 0.051171, F1 0.092457, recall 0.510000, ROC-AUC 0.500645
- BPSO-K10: AP 0.050732, F1 0.086940, recall 0.468000, ROC-AUC 0.489860
- BGWO: AP 0.050810, F1 0.088486, recall 0.468000, ROC-AUC 0.490500

Decision Tree is the informative classifier for this controlled task (AP in the 0.22-0.23 range); Logistic Regression is a weak model for every configuration.

## 7. Paired Statistical Comparisons (V0.9-E)

Paired 95% CIs of (BGWO minus reference) across 5 seed pairs (n=5, df=4):

- Decision Tree, BGWO vs BPSO-K10, AP: difference 6.42e-3, 95% CI [0.001310, 0.011536] — excludes zero (favors BGWO)
- Decision Tree, BGWO vs BPSO-K10, F1: difference −7.54e-3, CI [−0.038912, 0.023829] — direction-uncertain
- Decision Tree, BGWO vs BPSO-K10, recall: difference 9.33e-3, CI [−0.020996, 0.039663] — direction-uncertain
- All other BGWO-vs-reference comparisons (vs K43, K42, MI-K11, BPSO-K10, per classifier and classifier metrics) are direction-uncertain

## 8. Validation vs Final-Test Preservation

Semantic preservation checks were evaluated after the final test was opened, without reselection or retuning. Margins (AP loss <= 5%, F1 loss <= 5%, recall loss <= 10%) are descriptive and operational: they are not cross-domain-validated, and the final test was never used to select or retune the winner. `winner_reselection_performed: false`. Both classifiers satisfy overall preservation under these margins; results are reported per metric and direction-uncertain bounds.

## 9. Resource Efficiency (V0.9-F)

`DIRECT_ENERGY_UNAVAILABLE`. All values are computational resource proxies (timing/CPU/RSS/memory), not energy measurements.

Decision Tree, BGWO (14 features):

- training input memory: 3,136,132 bytes; inference input memory: 672,132 bytes
- training wall time: 12.6238 s; training CPU time: 0.0420 s; peak RSS: 157.32 MiB
- inference throughput ~2.96e6 records/s; serialized model: 3,812 bytes

Logistic Regression, BGWO (14 features):

- training input memory: 3,136,132 bytes; inference input memory: 672,132 bytes
- training wall time: 12.2151 s; training CPU time: 0.0716 s; peak RSS: 164.83 MiB
- inference throughput ~3.13e6 records/s; serialized model: 1,494 bytes

Structural reductions vs K43 (Decision Tree): feature count −67.44% for BGWO-K14 (BPSO-K10 −76.74%), memory and serialized-size reductions of similar order. Fewer features do not guarantee a smaller serialized model in absolute terms.

## 10. One-Time Optimizer Overhead (imported, never rerun)

- BGWO (V0.9-D, IMPORTED_HISTORICAL): 768 candidate requests, 3840 fits, optimizer wall time 325.4254 s, CPU 325.2538 s, pipeline wall time 111.8911 s
- BPSO (V0.8-C, IMPORTED_HISTORICAL): 972 candidate requests, 4860 fits, optimizer wall time 645.5029 s, CPU 645.2183 s, pipeline wall time 857.1114 s

One-time search cost is reported separately from recurring resource use; it adds no per-inference energy claim.

## 11. Predictive-Resource Tradeoff (descriptive)

Post-lock, descriptive juxtaposition of feature count vs frozen final-test AP and measured resources. It is descriptive only; no multi-objective optimization or ranking is performed.

## 12. Energy, Timing, and Preservation Statements

- `DIRECT_ENERGY_UNAVAILABLE`; `joule_estimation_from_cpu_or_wall_performed: false`; `observation_level_timestamps_present: false`; TDP multiplication prohibited; timing is a proxy, not a Joule measurement.
- Preservation margins are operational, not domain-validated; direction-uncertain paired results are reported as uncertain.
- `Late_delivery_risk` is not cyber ground truth; `is_attack` is the controlled synthetic target.

## 13. Main Findings, Limitations, Conclusion

Main findings:

1. BGWO winner locked at K=14 (seed 2042); validation AP 0.234618, F1 0.189644, recall 0.322667.
2. BGWO stability: mean pairwise Jaccard 0.3634; mean K 20.6, median 22 (min runs selected as few as 14 features).
3. Final test: only Decision Tree AP (BGWO vs BPSO-K10) has a paired 95% CI excluding zero, favoring BGWO; all other pairwise differences are direction-uncertain.
4. Feature overlap: BGWO shares 3 features with BPSO-K10 (Jaccard 0.142857) and, per-seed, ~6.4 with MI-K11 (union Jaccard 0.347826).
5. Structural lightweight gains are large for both metaheuristics vs K43; BPSO-K10 remains the smallest locked representation.
6. V0.9-F reports computational proxies only; no direct or TDP-derived energy measurements.

Limitations (unchanged from frozen governance):

- No direct physical-energy measurement; TDP multiplication prohibited; Modern Standby-limited timing.
- Preservation margins are operational, not domain-validated.
- MI-K11 is seed-specific; its overlap with BGWO is summarized across 5 seeds.
- Runs 2043/2044 of V0.9-D recorded identical best validation means on different masks/K (frozen artifact quirk; reported as-is).

Neutral statement: metric-specific trade-offs are reported for every locked configuration; no single overall winner, score, or ranking is asserted. The final governed classification is `STRUCTURAL_GAIN_TIMING_UNCERTAIN_LOW_PREDICTIVE_DELTA`.

## 14. Reproducibility

- Notebook: `notebooks/v09_bgwo_feature_selection_evaluation.ipynb` (18 sections; top-to-bottom execution only, no experiment launch).
- Tables: `results/bgwo/v09g_reporting/tables/` (10 CSVs) and Figures: `results/bgwo/v09g_reporting/figures/` (12 PNG + 12 PDF).
- Reporting lock: `results/bgwo/v09g_reporting/v09g_reporting_lock.json` (created after notebook execution by tests; independently verifiable).
- Prior-stage locks (V0.8-C, V0.9-D/E/F) were not modified.