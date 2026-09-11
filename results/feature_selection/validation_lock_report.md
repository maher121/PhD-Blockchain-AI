# V0.6-E Validation Configuration Lock

- Semantic payload SHA-256: `487b7aea89a9998b92e1d6bb6eac87d557a2328db82d82e7cbffb660de5de70c`
- Scope: stored development-validation artifacts only
- Test accessed: false
- No selector fit, model fit, threshold tuning, inference, or final-test execution was performed by V0.6-E.
- Ground truth is controlled experimental/synthetic cyber ground truth; it is not native DataCo cybersecurity labeling.
- Timing, RSS, and serialized-size measurements are computational proxies, not direct energy measurements.

## Locked Roles

### full_baseline

- Configuration: `none_natural`
- Selector: `none`
- Requested feature count: None
- Reason: full-feature validation baseline
- Preservation: preserving
- Mean actual feature count: 43
- Mean feature reduction: 0%
- Mean AP / F1 / recall: 0.23068406 / 0.19770107 / 0.322
- Mean precision / ROC-AUC: 0.285343 / 0.61961064
- Paired mean AP / F1 / recall differences: 0 / 0 / 0
- Mean model-size difference: 0 bytes
- Mean selector fit: 0 s
- Mean model fit: 0.1764273 s
- Mean combined fit: 0.1764273 s
- Mean validation inference: 0.0051401966 s
- Mean peak RSS: 286.21875 MiB
- Mean serialized model size: 4132 bytes
- Selected-set Jaccard mean / min / max: 1 / 1 / 1

### best_unsupervised

- Configuration: `pairwise_correlation_filter_natural`
- Selector: `pairwise_correlation_filter`
- Requested feature count: None
- Reason: best unsupervised per-method choice by frozen validation tie-break
- Preservation: preserving
- Mean actual feature count: 42
- Mean feature reduction: 2.3255814%
- Mean AP / F1 / recall: 0.23068406 / 0.19770107 / 0.322
- Mean precision / ROC-AUC: 0.285343 / 0.61961064
- Paired mean AP / F1 / recall differences: 0 / 0 / 0
- Mean model-size difference: -16 bytes
- Mean selector fit: 1.2845125 s
- Mean model fit: 0.11203446 s
- Mean combined fit: 1.3965469 s
- Mean validation inference: 0.0049837836 s
- Mean peak RSS: 296.41484 MiB
- Mean serialized model size: 4116 bytes
- Selected-set Jaccard mean / min / max: 1 / 1 / 1

### best_supervised

- Configuration: `mutual_information_select_k_best_k11`
- Selector: `mutual_information_select_k_best`
- Requested feature count: 11
- Reason: best supervised per-method choice by frozen validation tie-break
- Preservation: preserving
- Mean actual feature count: 11
- Mean feature reduction: 74.418605%
- Mean AP / F1 / recall: 0.23262793 / 0.20561017 / 0.31666667
- Mean precision / ROC-AUC: 0.3104048 / 0.6165193
- Paired mean AP / F1 / recall differences: 0.0019438704 / 0.0079090988 / -0.0053333333
- Mean model-size difference: -729.6 bytes
- Mean selector fit: 4.2281263 s
- Mean model fit: 0.070494104 s
- Mean combined fit: 4.2986204 s
- Mean validation inference: 0.005274279 s
- Mean peak RSS: 296.65547 MiB
- Mean serialized model size: 3402.4 bytes
- Selected-set Jaccard mean / min / max: 0.56709707 / 0.375 / 0.69230769

### smallest_preserving

- Configuration: `mutual_information_select_k_best_k11`
- Selector: `mutual_information_select_k_best`
- Requested feature count: 11
- Reason: fewest actual features among all primary-preserving nonbaseline configurations
- Preservation: preserving
- Mean actual feature count: 11
- Mean feature reduction: 74.418605%
- Mean AP / F1 / recall: 0.23262793 / 0.20561017 / 0.31666667
- Mean precision / ROC-AUC: 0.3104048 / 0.6165193
- Paired mean AP / F1 / recall differences: 0.0019438704 / 0.0079090988 / -0.0053333333
- Mean model-size difference: -729.6 bytes
- Mean selector fit: 4.2281263 s
- Mean model fit: 0.070494104 s
- Mean combined fit: 4.2986204 s
- Mean validation inference: 0.005274279 s
- Mean peak RSS: 296.65547 MiB
- Mean serialized model size: 3402.4 bytes
- Selected-set Jaccard mean / min / max: 0.56709707 / 0.375 / 0.69230769
