# V1.0-G Elite-Transfer Ablation Study — Final Results

**Stage:** V1.0-G6 (finalization and governed result lock)
**Status:** ABLATION_RESULT_LOCKED
**Protocol:** `docs/v10g_elite_transfer_ablation_protocol.md` (V1.0-G1)
**Config:** `config/ablation_v10g.yaml`
**Starting checkpoint:** `b302fa0b35c8f4b63f4ee1f27b886f733f941c6f` (G1/G2 commit, later superseded by the V1.0-G commit)
**Result lock:** `results/hybrid/v10g/v10g_result_lock.json`
**Semantic result-lock SHA-256:** `e33414d491d50676bfb5dfc926dbe4ada1a16931fa083a0e8b914958b99fc4de`

This document reports the outcome of the elite-transfer ablation run in the
frozen V1.0-G framework. It contains no new analysis; every number below is the
recorded, verified value of the governed V1.0-G4 campaign and the deterministic
V1.0-G5 analysis, as locked by V1.0-G6. G6 performed no experiment, invoked no
optimizer or fitted model, and never accessed the TEST split.

---

## 1. Research question

Does seeding each BPSO arm with the best solution ("elite") the BPSO optimizer
already produced on the immediately preceding arm of the same paired seed
(hereafter the *elite-transfer* arm, WITH) change the outcome relative to an
identical run that receives no such transfer (the control arm, WITHOUT)?

## 2. Hypothesis

The V1.0-G protocol hypothesized that elite transfer could improve the ablative
decision (fewer unique candidates / decision-tree fits for equal endpoint
values, or improved endpoints with equal budget). The protocol required
fiducial zero-equivalence: a resolved contribution requires the paired
posterior intervals for the primary endpoints to be strictly positive in favor
of the WITH arm.

## 3. Frozen design

- Framework: hybrid BGWO→BPSO protocol of V1.0-D (`HYBRID-K13` was the frozen
  winner); V1.0-F resource accounting; V1.0-G applies the ablation machinery
  only.
- Evaluator: production decision-tree classifier, balanced class weights,
  max_depth 5, min_samples_leaf 20, threshold 0.5, feasibility margins
  0.05/0.05/0.10, tolerance 1e-12.
- Dataset v08b, 43 candidate features; TRAIN 28 000 / VALIDATION 6 000
  (split seed 42). TEST locked during the entire ablation; `test_access_count =
  0` in every artifact.
- One optimizer rerun per arm could reuse cached evaluations; decision-tree
  fits counted at 5 per unique evaluation (the production fit strategy).
- AblationVariant enumeration `WITH_ELITE_TRANSFER` / `WITHOUT_ELITE_TRANSFER`;
  the capped elite came from the *other* adapter block at the same epoch
  boundary (epoch 11 BGWO → iterations 1–3 of BPSO) — see
  `src/optimization/hybrid_ablation_v10g.py`.

## 4. WITH vs WITHOUT definition

| Aspect | WITH (elite transfer) | WITHOUT (control) |
| --- | --- | --- |
| Elite placement | rows 1–3 of BPSO (the BGWO-block output of BPSO) replaced by the same seed's other arm's bgwo row 0 | rows 1–3 filled by the shadow filler block |
| RNG per arm | dedicated deterministic RNG (same law both arms) | same |
| Rest of code path | identical | identical |
| Budgets | 96 BGWO + 96 BPSO = 192 candidate requests | same |

Both arms therefore share *every* non-elite input; any endpoint difference is
attributable solely to the capped BGWO-block elite transfer.

## 5. Budget and seed design

- Paired seeds (alternating locked execution order): 3042, 3043, 3044, 3045,
  3046.
- Per arm: 96 BPSO + 96 BGWO = 192 candidate requests; per seed: 2 × 192 =
  384; campaign: 10 arms / 5 pairs / 1 920 requests, split 960 WITH and 960
  WITHOUT.
- Every arm finished with `FIXED_BUDGET_EXHAUSTED`; all finals feasible;
  final phase BPSO.

## 6. Primary endpoints (paired, n = 5)

Paired difference = WITH − WITHOUT on the arm-final validation metrics.

| Endpoint | Differences | Mean diff | SD | SE | 95% CI | all-zero |
| --- | --- | --- | --- | --- | --- | --- |
| Average Precision | 0, 0, 0, 0, 0 | 0.0 | 0.0 | 0.0 | [0.0000, 0.0000] | yes |
| F1 | 0, 0, 0, 0, 0 | 0.0 | 0.0 | 0.0 | [0.0000, 0.0000] | yes |

Precision and ROC-AUC differences are identically zero on all five seeds (the
paired finals are identical, so every derived metric coincides). No richer
p-value machinery is reported; by protocol, only fiducial CIs are used.

## 7. K (primary-supporting)

K = number of BPSO-evaluated candidate iterations that produced no fitness
gain.

| Seed | 3042 | 3043 | 3044 | 3045 | 3046 |
| --- | --- | --- | --- | --- | --- |
| K WITH | 16 | 13 | 14 | 13 | 15 |
| K WITHOUT | 16 | 13 | 14 | 13 | 15 |

Differences all zero; mean/median/min/max difference = 0.0/0.0/0.0/0.0;
negatives 0, zeros 5, positives 0; descriptive 95% CI [0.0, 0.0]. K is a
primary-*supporting* endpoint only: it cannot independently establish
inferential elite-transfer benefit.

## 8. Recall safeguard

Paired recall differences all exactly zero; safeguard status PASS;
`resolved_recall_harm_detected = false`. No elite-transfer pathology reduced
recall anywhere.

## 9. Stability

- Cross-arm paired mask Jaccard = 1.0 on every seed (WITH and WITHOUT reached
  byte-identical binary masks).
- Within-arm pairwise Jaccards are the same under WITH and WITHOUT, with values
  0.318, 0.200, 0.208, 0.292, 0.174, 0.130, 0.167, 0.286, 0.261, 0.167.

| Seed | mask WITH | mask WITHOUT | paired Jaccard |
| --- | --- | --- | --- |
| 3042 | identical | identical | 1.0 |
| 3043 | identical | identical | 1.0 |
| 3044 | identical | identical | 1.0 |
| 3045 | identical | identical | 1.0 |
| 3046 | identical | identical | 1.0 |

## 10. Feature frequency and consensus

Feature-selection frequency over the five arms, WITH vs WITHOUT, differs by
exactly zero for all 43 features.

- Strict consensus (selected by all 5 arms, both variants): `order_item_quantity`,
  `order_item_total`.
- Consensus ≥ 3 of 5: `order_item_quantity`, `order_item_product_price`,
  `order_item_total`, `Type_PAYMENT`, `Market_LATAM`,
  `Shipping Mode_Second Class`, `Department Name_Golf`.

| Feature | frequency WITH | frequency WITHOUT |
| --- | --- | --- |
| order_item_quantity | 5 | 5 |
| order_item_product_price | 4 | 4 |
| order_item_total | 5 | 5 |
| Type_PAYMENT | 4 | 4 |
| Market_LATAM | 3 | 3 |
| Shipping Mode_Second Class | 3 | 3 |
| Department Name_Golf | 4 | 4 |

## 11. Computational accounting

| Quantity | WITH | WITHOUT | Δ WITH − WITHOUT |
| --- | --- | --- | --- |
| Unique candidate evaluations (campaign) | 940 | 955 | −15 |
| Evaluator cache hits | 20 | 5 | +15 |
| Decision-tree fits (= unique × 5) | 4 700 | 4 775 | −75 |
| BGWO-block new unique evaluations | 460 | 475 | −15 |
| Nominal candidate requests | 960 | 960 | 0 |

Both arms used the full nominal budget (960 with / 960 without / 1 920 total);
the WITHOUT arm simply re-derived 15 unique evaluations (75 fits) that the WITH
arm obtained from cache. Every arm satisfies `unique + cache_hits = 192` and
`fits = unique × 5`.

## 12. Convergence and mechanistic interpretation

- Convergence histories differ: WITH skips the repeated re-evaluation of the
  bgwo row-0 elites and spends its BPSO evaluations earlier in the budget.
- On every seed, the first three WITH candidate rows equal the paired WITHOUT
  arm's bgwo row 0 (elite placement); the WITHOUT rows are the filler block;
  bgwo row 0 and BPSO rows 4+ are paired-identical; both arms share the common
  bgwo RNG stream.
- Net effect: elite transfer changed the *trajectory and the accounting*, not
  the destination — final masks, endpoints, and final-phase (BPSO) selections
  are identical WITH vs WITHOUT on all five seeds.

## 13. Scientific interpretation (frozen conclusion)

> **This experiment does not demonstrate a resolved elite-transfer contribution.**

All five paired differences for the primary endpoints and for K are exactly
zero, with zero-width 95% confidence intervals [0, 0]; cross-arm paired Jaccard
= 1.0 on every seed. The WITH arm saved 15 unique evaluations and 75 decision
tree fits over the campaign, but this is a computational/mechanistic difference,
not evidence of preference: the paired final solutions are identical and no
protocol-sanctioned improvement exists to claim. Per V1.0-G winner governance,
`HYBRID-K13` remains the frozen winner; this study replaces nothing.

## 14. Limitations

- Results are conditional on the two optimizer blocks, the capped one-elite
  transfer at the epoch-11 boundary, the production evaluator configuration,
  dataset v08b, and the five locked seeds; no generalization to other budgets
  or transfer policies is asserted.
- As a negative/zero-equivalent result within the protocol's fiducial framing,
  it bounds the effect size to zero for this configuration rather than
  excluding mechanistic effects in other regimes.
- The 15-evaluation / 75-fit saving is a single-campaign accounting fact and
  was not re-tested with repeated sampling.

## 15. Reproducibility

- G1 protocol lock recomputes exactly (`2914819f…`); G4 campaign artifacts,
  G5 analysis artifacts, and V10D/E/F locks byte-verified in the G6 validation
  (`results/hybrid/v10g/v10g6_validation.json`).
- V1.0-G5 outputs are deterministic (byte-identical rerun); V1.0-G6 artifacts
  are deterministic (byte-identical rerun, including from a fresh results
  state); the result-lock semantic SHA-256 recomputes to the stored value.
- Runners: `src/pipeline_v10g4.py` (production campaign),
  `src/pipeline_v10g5.py` (read-only analysis), `src/pipeline_v10g6.py`
  (read-only finalization + lock). Optimizer/fitness behavior is unchanged and
  TEST is fully isolated (`test_access_count = 0`).

## 16. Final V1.0-G status

| Item | Value |
| --- | --- |
| V1.0-G2 implementation tests | PASS (incl. lifecycle regression coverage) |
| V1.0-G4 campaign | COMPLETE (10 arms, 5 pairs, 1 920 requests, TEST 0) |
| V1.0-G5 deterministic analysis | V10G5_COMPLETE_GO |
| V1.0-G6 result lock | `e33414d491d50676bfb5dfc926dbe4ada1a16931fa083a0e8b914958b99fc4de` |
| Frozen winner | HYBRID-K13 (unchanged) |
| Conclusion | No demonstrated resolved elite-transfer contribution |