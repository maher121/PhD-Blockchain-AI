# V0.6-H Feature-Selection Research Report

- **Project:** A Lightweight and Green Blockchain-AI Framework for Secure and Energy-Efficient Supply Chain Management
- **Stage:** V0.6-H, research documentation and final feature-selection report
- **Evidence base:** Completed and locked V0.6-D/E/F/G artifacts
- **Analysis policy:** Artifact-only descriptive analysis; no experiment reruns or new model-selection decisions

## Abstract

V0.6 investigated whether systematic classical feature selection could reduce the dimensionality and computational cost of a supply-chain cybersecurity detection pipeline while preserving controlled attack-detection performance. The experiment evaluated 12 feature-selection configurations across five seeds on development validation, locked three unique configurations, evaluated those configurations once on the governed final test, and assessed robustness across 13 attack scenarios and two classifiers. Mutual-information selection with K=11 reduced the candidate feature space from 43 to 11 features, a 74.42% reduction, while satisfying the preregistered descriptive preservation margins on both validation and final test. The final-test paired confidence intervals for average precision, F1, and recall all crossed zero; therefore, the result supports preservation rather than statistical superiority. Absolute detection remained modest, robustness was scenario-dependent, and the selected MI feature sets varied across seeds. Attack visibility was necessary for attack-induced model response but was not sufficient for successful detection: timestamp manipulation changed model-visible features while remaining difficult to detect. V0.6 establishes a leakage-controlled classical baseline for later Green AI and metaheuristic stages without claiming production-ready cybersecurity effectiveness or measured electrical-energy savings.

## 1. Research Context and Question

V0.6 follows the V0.5 lightweight-model comparison and precedes optimization with BPSO, BGWO, and the proposed hybrid method. Its purpose is to establish a conventional, reproducible reference that later optimization stages cannot retrospectively weaken.

The research question was:

> Can systematic classical feature selection reduce the dimensionality and computational cost of the supply-chain cybersecurity detection pipeline while preserving acceptable controlled attack-detection performance?

The question was evaluated independently of future optimization. Development validation supported configuration decisions; the final test and robustness panels evaluated locked decisions without reopening selection.

## 2. PRD Traceability

| PhD PRD requirement | V0.6 contribution |
|---|---|
| AI/anomaly-detection pipeline support | Preserves a common detector interface, threshold policy, metrics, and model-artifact workflow |
| Cybersecurity evaluation | Evaluates controlled attack ground truth across families, rates, severities, and visibility conditions |
| Lightweight computing | Measures compact Decision Tree and Logistic Regression artifacts under reduced feature spaces |
| Feature reduction | Compares no selection, unsupervised filtering/ranking, supervised MI, and supervised ANOVA-F |
| Computational efficiency | Reports feature count, fit time, inference time, per-record latency, RSS, selected-input size, and serialized size |
| Reproducibility and leakage control | Uses five seeds, paired comparisons, confidence intervals, immutable locking, and training-only selector fitting |

Requirements intentionally deferred beyond V0.6 are direct Green/Energy measurement, BPSO, BGWO, hybrid optimization, integrated Blockchain-AI decision architecture, multi-objective optimization, deployment scalability, and final ablation studies. V0.6 supports these later stages but does not claim to implement them.

## 3. Experimental Design

### 3.1 Dataset and frozen split

The experiment uses the real DataCo Smart Supply Chain operational dataset after the frozen V0.2 preprocessing protocol. The fixed split contains 28,000 training rows, 6,000 development-validation rows, and 6,000 final-test rows. The 43-feature candidate manifest has SHA-256:

```text
5146fd08fe766979adaf443bf9f4f7d32ee3c94cfdaf0fd46ec0efd10e92a10d
```

All V0.6 uncertainty estimates describe five algorithm/attack seeds on this one fixed split. They do not quantify generalization across organizations, time periods, or independently sampled datasets.

### 3.2 Cybersecurity ground truth

DataCo does not provide native cybersecurity attack labels. V0.6 therefore uses controlled V0.4 attack manifestations and experiment-generated `is_attack` labels defined before inference. These labels support repeatable data-tampering experiments; they are not evidence of naturally occurring or operationally verified cyber intrusions.

The seven attack families are quantity manipulation, value manipulation, shipping/delivery manipulation, transaction-status manipulation, timestamp manipulation, geographic-route manipulation, and transaction-record tampering. The robustness panel additionally varies mixed attack rate and severity.

### 3.3 Leakage control

The V0.6-D leakage audit reports `PASS`. Unsupervised selectors received no labels. Supervised mutual-information and ANOVA-F selectors received only controlled training ground truth. Preprocessing and selector fitting were confined to training data. Validation supported locking; test data did not influence feature selection, model fitting, threshold selection, attack configuration, or candidate ranking.

Attack metadata, selected row identifiers, original and modified values, and ground-truth labels were excluded from model inputs. Final-test governance confirms that selector fitting, preprocessing refitting, model fitting, threshold tuning, consensus selection, and reselection were not performed after the lock.

### 3.4 Feature-selection methods and counts

The classical comparison included:

1. No-selection control.
2. Zero-variance threshold.
3. Pairwise-correlation filtering.
4. Correlation-redundancy ranking.
5. Supervised mutual information with `SelectKBest`.
6. Supervised ANOVA F-statistic with `SelectKBest`.

Fixed K values of 32, 22, and 11 approximate 75%, 50%, and 25% retention from the 43-feature manifest. Natural-count methods retained the count resulting from their rule. RFE was optional and was not included in the frozen core. No PSO, GWO, BPSO, BGWO, hybrid, NSGA-II, or MOPSO method was used.

### 3.5 Models

The primary model was a fixed Decision Tree with `max_depth=5`, `min_samples_leaf=20`, `class_weight="balanced"`, seed-specific `random_state`, and threshold 0.5. Keeping this model fixed isolated feature-selection effects.

V0.6-G added an untuned secondary Logistic Regression comparator with `solver="liblinear"`, `class_weight="balanced"`, `max_iter=500`, `C=1.0`, seed-specific `random_state`, and threshold 0.5. It was fitted once per seed/configuration on controlled training labels and reused across scenarios. It was not selected from final-test performance.

### 3.6 Experiment matrix

| Stage | Design | Completed runs |
|---|---|---:|
| V0.6-D validation | 12 configurations x 5 seeds x 1 Decision Tree | 60/60 |
| V0.6-F final test | 3 locked configurations x 5 seeds x 1 Decision Tree | 15/15 |
| V0.6-G robustness | 13 scenarios x 3 configurations x 2 classifiers x 5 seeds | 390/390 |

## 4. Validation Results

The primary descriptive preservation rule allowed at most 5% relative degradation in mean average precision and F1 and at most 10% relative degradation in mean recall. This was a preregistered margin screen, not a formal statistical non-inferiority test.

| Configuration | Features | Mean AP | Mean F1 | Mean recall | Validation status |
|---|---:|---:|---:|---:|---|
| Full baseline | 43 | 0.230684 | 0.197701 | 0.322000 | Reference |
| Pairwise correlation | 42 | 0.230684 | 0.197701 | 0.322000 | Preserving |
| MI K11 | 11 | 0.232628 | 0.205610 | 0.316667 | Preserving |

MI K11 reduced the feature count by:

```text
100 x (43 - 11) / 43 = 74.4186%
```

The paired MI K11 differences relative to the full baseline were:

| Metric | Mean difference | 95% CI |
|---|---:|---:|
| Average precision | +0.001944 | [-0.005582, +0.009470] |
| F1 | +0.007909 | [-0.048917, +0.064735] |
| Recall | -0.005333 | [-0.029924, +0.019257] |

Every interval crosses zero. The validation result therefore supports descriptive preservation but does not establish statistically significant improvement. Pairwise correlation produced exactly the same aggregate Decision Tree performance as the baseline while removing one feature.

## 5. Validation Lock

V0.6-E persisted the feature-selection decision before final-test access. Four semantic roles collapsed to three unique configurations:

| Semantic role | Locked configuration | Features |
|---|---|---:|
| Full baseline | `none_natural` | 43 |
| Best unsupervised | `pairwise_correlation_filter_natural` | 42 |
| Best supervised | `mutual_information_select_k_best_k11` | 11 |
| Smallest preserving | `mutual_information_select_k_best_k11` | 11 |

The lock has file SHA-256 `71dc92b8a34db1bb3c2f55a69e56f43733544329128ec60c5cdbbce2445d1604` and semantic SHA-256 `487b7aea89a9998b92e1d6bb6eac87d557a2328db82d82e7cbffb660de5de70c`. Its audit reports `PASS`. V0.6-E performed no fitting, inference, threshold tuning, or test access.

## 6. Final-Test Results

V0.6-F verified the lock before the first governed final-test access and then completed all 15 locked runs. Models and selected feature lists were loaded from the lock without fitting or reselection.

| Configuration | Features | AP | F1 | Recall | Precision | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Full baseline | 43 | 0.222084 | 0.189005 | 0.305333 | 0.265385 | 0.603533 |
| Pairwise correlation | 42 | 0.222084 | 0.189005 | 0.305333 | 0.265385 | 0.603533 |
| MI K11 | 11 | 0.224941 | 0.192121 | 0.300000 | 0.293470 | 0.605495 |

MI K11 satisfied all final-test descriptive preservation margins. Its recall degradation was 1.7467%, within the preregistered 10% margin. The paired differences were:

| Metric | Mean difference | 95% CI |
|---|---:|---:|
| Average precision | +0.002857 | [-0.007383, +0.013097] |
| F1 | +0.003116 | [-0.044833, +0.051065] |
| Recall | -0.005333 | [-0.037473, +0.026806] |

All intervals cross zero. MI K11 preserved performance under the specified rule, but V0.6 does not demonstrate superiority. Absolute AP, F1, and recall remained modest and do not support a production-ready detection claim.

## 7. Robustness Results

### 7.1 Attack-family behavior

Decision Tree full-baseline results at 5% MEDIUM illustrate strong family dependence:

| Attack family | Feature visibility | AP | Recall |
|---|---:|---:|---:|
| Quantity manipulation | 1.00 | 0.795512 | 0.846667 |
| Value manipulation | 1.00 | 0.086363 | 0.242000 |
| Shipping/delivery | 0.00 | 0.049837 | 0.149333 |
| Transaction status | 0.00 | 0.049837 | 0.149333 |
| Timestamp | 1.00 | 0.049837 | 0.149333 |
| Geographic route | 0.00 | 0.049837 | 0.149333 |
| Transaction-record tampering | 1.00 | 0.471151 | 0.563333 |

Quantity manipulation was comparatively detectable, transaction-record tampering was intermediate, and several families remained close to prevalence-level AP.

### 7.2 Attack-rate behavior

For the Decision Tree, mixed-scenario AP and F1 generally increased from 1% to 10% attacks, while recall stayed near 0.30-0.32. AP, precision, and F1 depend on prevalence; increasing values therefore do not by themselves show increased robustness.

| Rate | Baseline AP / F1 / recall | MI K11 AP / F1 / recall |
|---:|---|---|
| 1% | 0.1756 / 0.0750 / 0.3133 | 0.1734 / 0.0883 / 0.2967 |
| 3% | 0.2000 / 0.1527 / 0.3067 | 0.2031 / 0.1610 / 0.3000 |
| 5% | 0.2221 / 0.1890 / 0.3053 | 0.2249 / 0.1921 / 0.3000 |
| 10% | 0.2868 / 0.2616 / 0.3240 | 0.2894 / 0.2622 / 0.3187 |

### 7.3 Severity behavior

Severity response was not monotonic in AP. For the Decision Tree baseline, LOW, MEDIUM, and HIGH AP values were approximately 0.075, 0.222, and 0.214. Controlled severity labels define transformation magnitude or category-transition steps; they do not guarantee monotonic model separation.

### 7.4 Attack visibility

Visibility records whether a controlled source-field modification changes at least one selected engineered input. Shipping/delivery, transaction-status, and geographic-route attacks had zero visibility for the 43-feature baseline. These attacks cannot induce a response through information absent from the model matrix.

Visibility was nevertheless insufficient. Timestamp manipulation had full feature visibility but baseline Decision Tree AP of 0.049837 and recall of 0.149333, matching several invisible families. This demonstrates the central V0.6-G conclusion:

> Feature visibility is important and necessary for attack-induced model response, but it is not sufficient for successful detection.

### 7.5 Decision Tree versus Logistic Regression

At the primary mixed 5% MEDIUM scenario, Decision Tree AP was approximately 0.222-0.225. Logistic Regression AP was approximately 0.05. Logistic Regression sometimes achieved higher recall, but its precision remained near 0.05 and it generated many false positives. Higher recall alone was therefore not an across-the-board advantage.

### 7.6 Robustness preservation

Pairwise correlation descriptively preserved all 13 scenarios under both classifiers. MI K11 preserved the primary final-test case but did not preserve every robustness scenario. Decision Tree MI K11 failed overall descriptive preservation for shipping/delivery, transaction status, timestamp, geographic route, and mixed 5% LOW. Logistic Regression MI K11 had additional failures. No robustness result was used to change the lock.

## 8. Feature-Selection Stability

Deterministic unsupervised configurations had pairwise Jaccard similarity 1.0 across seeds. MI K11 had mean pairwise Jaccard similarity 0.567, ranging from 0.375 to 0.692. Only five features appeared in every seed-specific MI K11 set:

- `is_weekend`
- `order_item_quantity`
- `order_item_total`
- `product_price`
- `year`

No post hoc consensus selection was performed. MI K11 is therefore a locked configuration rule with seed-specific selected sets, not evidence for one universal 11-feature subset.

## 9. Computational Resource Analysis

### 9.1 Validation scope

V0.6-D training time includes selector and Decision Tree fitting. It represents development acquisition cost rather than only recurring model fitting.

| Configuration | Features | Combined training (s) | Inference (s) | Peak RSS (MiB) | Model bytes |
|---|---:|---:|---:|---:|---:|
| Full baseline | 43 | 0.176 | 0.00514 | 286.22 | 4,132.0 |
| Pairwise correlation | 42 | 1.397 | 0.00498 | 296.41 | 4,116.0 |
| MI K11 | 11 | 4.299 | 0.00527 | 296.66 | 3,402.4 |

MI K11 reduced serialized Decision Tree size but incurred the largest feature-selection acquisition cost. Validation inference differences were small and within a short, noisy timing regime.

### 9.2 Final-test inference scope

V0.6-F measured fresh-process repeated Decision Tree inference on 6,000 records, excluding model loading.

| Configuration | Features | Inference for 6,000 records (s) | Derived latency (microseconds/record) | Peak RSS (MiB) | Model bytes |
|---|---:|---:|---:|---:|---:|
| Full baseline | 43 | 0.02361 | 3.94 | 153.07 | 4,132.0 |
| Pairwise correlation | 42 | 0.03060 | 5.10 | 152.93 | 4,116.0 |
| MI K11 | 11 | 0.02692 | 4.49 | 150.80 | 3,402.4 |

Feature reduction did not produce lower measured final-test latency. MI K11 showed a smaller serialized model and slightly lower absolute peak RSS, but interpreter/library overhead dominated process memory.

### 9.3 Robustness scope and comparability

V0.6-G measured inference in a current process with models already loaded; it also records Logistic Regression fitting and selected-input bytes. Its approximately 800 MiB process RSS reflects the larger robustness process, not model-only memory. These measurements must not be pooled numerically with V0.6-D or V0.6-F.

The resource evidence supports lightweight computational analysis only. Wall time, CPU time, selected-input bytes, RSS, and serialized model size are **not electrical energy**. V0.7 uses a separate preregistered Green AI measurement protocol and will make direct-energy claims only if a valid hardware backend exists.

## 10. Key Figures

The executed research notebook regenerates presentation figures in memory from existing tables. The original locked pipeline figures remain available for cross-checking:

![Feature count versus average precision](../results/feature_selection/robustness/figures/feature_count_vs_average_precision.png)

![Performance by attack family](../results/feature_selection/robustness/figures/performance_by_attack_family.png)

![Average precision versus attack rate](../results/feature_selection/robustness/figures/average_precision_vs_attack_rate.png)

![Recall versus attack severity](../results/feature_selection/robustness/figures/recall_vs_attack_severity.png)

![Attack visibility](../results/feature_selection/robustness/figures/visibility_rate_by_attack_family.png)

![Decision Tree versus Logistic Regression](../results/feature_selection/robustness/figures/dt_vs_logistic_regression.png)

![Robustness preservation](../results/feature_selection/robustness/figures/robustness_preservation_heatmap.png)

![Final-test inference resources](../results/feature_selection/final_test/figures/inference_resources.png)

## 11. Experimental Facts

- V0.6-D completed 60/60 validation runs with a passing leakage audit.
- The lock contains 43-feature baseline, 42-feature pairwise, and 11-feature MI configurations.
- MI K11 reduced features by 74.4186%.
- V0.6-F completed 15/15 final-test runs without fitting or reselection.
- MI K11 met the preregistered final-test descriptive preservation margins.
- Final-test paired AP, F1, and recall confidence intervals all crossed zero.
- V0.6-G completed 390/390 robustness runs.
- Pairwise correlation preserved all robustness scenarios for both classifiers.
- MI K11 preservation depended on scenario and classifier.
- Several attack families were invisible to the frozen engineered feature space.
- Timestamp manipulation was feature-visible but difficult to detect.
- Computational measurements were proxies; electrical energy was not directly measured.

## 12. Scientific Interpretation

MI K11 is the strongest classical dimensionality-reduction baseline because it retained only one quarter of the candidate features while satisfying the predefined mean preservation margins. This result is relevant to later optimization because it establishes that substantial reduction is achievable without a metaheuristic.

The result does not demonstrate that MI K11 improved cybersecurity performance. The observed mean AP and F1 increases were small, confidence intervals crossed zero, and absolute metrics remained modest. The seed-dependent selected sets further caution against interpreting MI K11 as a unique explanatory feature subset.

Robustness results show that representational coverage is at least as important as feature count. Feature selection cannot recover attacks that do not alter the candidate feature matrix. Conversely, the timestamp example shows that feature exposure alone does not guarantee statistical separation. Better cybersecurity performance may require improved feature engineering, temporal/contextual modeling, or additional trustworthy data sources rather than selector optimization alone.

## 13. Limitations

1. Controlled data modifications are not equivalent to native cyber incidents.
2. DataCo is an operational supply-chain dataset rather than a dedicated cybersecurity corpus.
3. One fixed split and five seeds limit external and statistical generalization.
4. Preservation margins are descriptive and not formal non-inferiority tests.
5. Nominal t-intervals may extend outside metric bounds because `n=5` and no bounded transformation was used.
6. AP, precision, and F1 vary with attack prevalence, complicating rate comparisons.
7. The 43-feature baseline omits some source fields for leakage or availability reasons.
8. Feature visibility does not imply learnability or adequate detection.
9. MI K11 selected-feature identities varied across seeds.
10. The fixed Decision Tree and secondary untuned Logistic Regression do not represent all classifiers.
11. Resource scopes differ across V0.6-D/F/G and cannot be pooled.
12. Computational proxies cannot support electrical-energy or carbon claims.
13. Modest absolute performance prohibits claims of production readiness.

## 14. V0.6 Conclusions

V0.6 conditionally answers its research question in the affirmative. Systematic classical feature selection reduced the frozen 43-feature space to 11 features while preserving the preregistered validation and final-test mean performance margins. MI K11 is therefore a defensible classical reduced baseline for subsequent Green AI and metaheuristic stages.

The defensible conclusion is preservation, not improvement. Absolute detection remained modest, paired intervals did not establish superiority, feature identities varied across seeds, and preservation did not extend to every robustness scenario. Feature visibility emerged as necessary but insufficient for attack detection. V0.6 contributes a reproducible and leakage-controlled baseline together with an explicit account of its limits.

## 15. Future Work

V0.7 will evaluate Green AI under a separate measurement protocol and will distinguish direct hardware energy from computational proxies. V0.8 and V0.9 will compare BPSO and BGWO against both the 43-feature and MI K11 baselines. V1.0 will evaluate the proposed hybrid/multi-objective optimization, integrated Blockchain-AI framework, scalability, and final ablations.

Later stages must preserve V0.6 data, model, threshold, attack, and leakage controls; report optimizer-search cost; and avoid using final-test outcomes for selection.

## 16. Reproducibility and Artifact Provenance

The executable notebook is `notebooks/v06_feature_selection.ipynb`. It locates the repository root, fails clearly when a required result artifact is missing, verifies lock and run-count assertions, computes only descriptive summaries from existing tables, and generates plots in memory. It does not access raw test data or import an experiment pipeline.

Primary artifacts read include:

- `results/feature_selection/core_runs.csv`
- `results/feature_selection/validation_summary.csv`
- `results/feature_selection/candidate_selection.json`
- `results/feature_selection/preservation_primary.csv`
- `results/feature_selection/paired_baseline_comparisons.csv`
- `results/feature_selection/selected_set_stability.csv`
- `results/feature_selection/validation_lock.json`
- `results/feature_selection/leakage_audit.json`
- `results/feature_selection/final_test/final_test_runs.csv`
- `results/feature_selection/final_test/final_test_summary.csv`
- `results/feature_selection/final_test/final_test_paired_comparison.csv`
- `results/feature_selection/final_test/final_test_preservation.csv`
- `results/feature_selection/final_test/final_test_resource_summary.csv`
- `results/feature_selection/robustness/robustness_runs.csv`
- `results/feature_selection/robustness/attack_family_summary.csv`
- `results/feature_selection/robustness/attack_rate_summary.csv`
- `results/feature_selection/robustness/attack_severity_summary.csv`
- `results/feature_selection/robustness/visibility_analysis.csv`
- `results/feature_selection/robustness/classifier_comparison.csv`
- `results/feature_selection/robustness/robustness_preservation.csv`
- `results/feature_selection/robustness/resource_comparison.csv`

Execute the documentation notebook from the repository root without rerunning any scientific pipeline:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/v06_feature_selection.ipynb
```

The complete `results/feature_selection/` tree had identical SHA-256 digests before and after V0.6-H notebook execution:

```text
94854229e54dc8a7e170e255cf6c6d492739cfbaccbe7972d4397275d50a8edf
```
