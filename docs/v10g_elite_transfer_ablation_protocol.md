# V1.0-G Elite-Transfer Ablation Protocol

## Protocol Identity

- Stage: `V1.0-G1`
- Protocol: `V10G_BPSO_TO_BGWO_ELITE_TRANSFER_ABLATION`
- Version: `v1.0-g-ablation-protocol-1`
- Status: `ABLATION_PROTOCOL_LOCKED`
- Starting checkpoint: `b086be7de998d18b6154301c0f29c5f43cb48d41`
- Frozen architecture: `BPSO_BGWO_SEQUENTIAL_50_50_ELITE3`
- Frozen V1.0-D winner: `HYBRID-K13`
- V1.0-D winner semantic lock: `1c258dc1a90ad78197d25e62bbc9fca24904cbcb77edafd24b12726059e49d84`
- V1.0-F semantic result lock: `f714b4a95070ee26718b3b2f8dc2ed820c5c2dadb7c5fb58fa1c09f95505b412`

This document and `config/ablation_v10g.yaml` define a validation-only paired
ablation. The YAML is the machine-readable authority. This stage persists and
locks the protocol only: it does not implement or execute either optimizer.

## Scientific Question

Does transferring the top three BPSO elites into BGWO initialization provide
measurable benefit compared with an otherwise identical hybrid pipeline whose
three transfer rows are filled by independent deterministic random exact-K
masks?

The manipulated factor is elite transfer present versus absent. The ablation
cannot select or replace the frozen V1.0-D winner.

## Experimental Arms

### WITH_ELITE_TRANSFER

- BGWO row 0 is the canonical K43 all-ones anchor.
- BGWO rows 1-3 are the top three distinct masks evaluated by the current
  run's BPSO phase under the frozen deterministic ranking.
- BGWO rows 4-11 are governed random exact-K masks from the common BGWO RNG.
- The independent three-row shadow filler block is generated and audited but
  is not placed or evaluated.

### WITHOUT_ELITE_TRANSFER

- BGWO row 0 is the same canonical K43 all-ones anchor.
- BGWO rows 1-3 are the independent deterministic shadow filler masks.
- BGWO rows 4-11 are byte-identical to the common random masks in the paired
  WITH run.
- Elite selection is not called.

The WITHOUT arm must not inspect elites, BPSO candidates, fitness values,
cache contents, previous winners, or TEST information when generating filler
masks. Accidental random collisions are retained and reported rather than
rejected or redrawn.

Both arms are executed fresh. Historical V1.0-D WITH results are replication
evidence only and cannot serve as the primary experimental arm.

## Search Budget

Each arm and optimizer seed uses:

| Phase | Population | Evaluated steps | Requests |
|---|---:|---:|---:|
| BPSO | 12 particles | 8 generations | 96 |
| BGWO | 12 wolves | 8 iterations | 96 |
| Total | | | 192 |

The optimizer seeds are `3042, 3043, 3044, 3045, 3046`. Each arm therefore
uses exactly 960 requests and the complete ablation uses exactly 1,920
requests. Iteration/generation zero counts toward the budget. Early stopping
is prohibited.

## Common-Random-Number Policy

For optimizer seed `s`, the locked PCG64 streams are:

- BPSO: `PCG64(s)`.
- BGWO common: `PCG64(s + 10000)`.
- Filler: `PCG64(SeedSequence([s, 0x56313047, 0x46494C4C]))`.

The filler stream uses the frozen cardinality pool
`[4, 8, 11, 14, 18, 22, 26, 30, 34, 38]`. Each row draws one cardinality and
then one exact-K mask over the canonical 43-feature space. Duplicate rejection
is prohibited.

Both arms generate the same shadow filler block. WITH does not evaluate it;
WITHOUT places it in rows 1-3. Before BGWO updates, paired runs must have
byte-identical BPSO outputs, row 0, rows 4-11, and BGWO common RNG state.
Later trajectories may diverge causally because rows 1-3 differ.

## Paired Execution And Cache Policy

The scientific unit is the paired optimizer seed. Runs are sequential and use
the locked order:

| Seed | First | Second |
|---:|---|---|
| 3042 | WITH_ELITE_TRANSFER | WITHOUT_ELITE_TRANSFER |
| 3043 | WITHOUT_ELITE_TRANSFER | WITH_ELITE_TRANSFER |
| 3044 | WITH_ELITE_TRANSFER | WITHOUT_ELITE_TRANSFER |
| 3045 | WITHOUT_ELITE_TRANSFER | WITH_ELITE_TRANSFER |
| 3046 | WITH_ELITE_TRANSFER | WITHOUT_ELITE_TRANSFER |

Caches are independent between arms and optimizer seeds. Within one arm/run,
the cache is shared between BPSO and BGWO. It is cleared after the run.

Accounting identities are:

- `requests = unique_evaluations + cache_hits`
- `evaluator_calls = unique_evaluations`
- `decision_tree_fits = unique_evaluations * 5`

Cache differences are outcomes and must not be normalized away by adding or
removing requests.

## Data Governance

Only TRAIN and VALIDATION features and labels may be loaded. TEST features and
TEST labels are inaccessible. There is no final-test evaluation, TEST-derived
ranking, threshold tuning, model tuning, feature reselection, or winner
replacement. The implementation must fail closed if TEST isolation cannot be
proved.

The frozen `HYBRID-K13` winner remains unchanged.

## Frozen Evaluator And Ranking

The search evaluator is Decision Tree only:

- `class_weight="balanced"`
- `max_depth=5`
- `min_samples_leaf=20`
- prediction threshold `0.5`
- model/attack seeds `42, 43, 44, 45, 46`

Feasibility requires:

- AP relative loss <= 5%.
- F1 relative loss <= 5%.
- Recall relative loss <= 10%.
- Boundary tolerance `1e-12`.

Feasible candidates outrank infeasible candidates. Feasible candidates are
ordered by smaller K, higher AP, higher F1, higher recall, and canonical
lexicographic mask tie-break. Infeasible candidates are ordered by lower total
normalized violation, higher AP, higher F1, higher recall, smaller K, and the
same tie-break. A scalar aggregate score is prohibited.

## Endpoint Hierarchy

### Primary Inferential Endpoints

1. Validation Average Precision.
2. Validation F1.

For each, the paired difference is `WITH - WITHOUT`; positive values favor
WITH. Report all five differences, their mean, standard deviation, and
two-sided 95% paired Student-t confidence interval. Interpret each endpoint
separately.

### Primary-Supporting Mechanistic/Parsimony Endpoint

3. Selected feature count K.

The paired difference is `K_WITH - K_WITHOUT`; negative values indicate fewer
features under WITH. Report all five integer differences, mean, median,
minimum, maximum, negative/zero/positive direction counts, and an approximate
paired 95% t interval as descriptive support only. K cannot independently
establish inferential benefit.

### Safeguard / Secondary Endpoint

4. Validation Recall.

Recall remains a safeguard because it is part of the frozen feasibility
contract. Precision, ROC-AUC, feasibility, normalized violation, accounting,
wall time, phase-best behavior, convergence, stability, and feature-frequency
results are additional secondary or descriptive endpoints.

## Statistical Contract

- Scientific unit: paired optimizer seed.
- `n = 5`.
- `df = 4`.
- Two-sided 95% Student-t interval.
- Critical value: `2.7764451051977987`.
- Paired difference: `WITH - WITHOUT`.
- Candidate evaluations are not scientific replicates.
- P-values are prohibited.
- The phrase "statistically significant" is prohibited.
- Allowed interval language: "CI excludes zero." or "Direction uncertain."

Feasibility is reported with paired transition counts, not a t interval.

## Stability And Mechanism Analysis

Report same-seed cross-arm final-mask Jaccard, all ten within-arm pairwise
Jaccards, per-feature selection frequency, arm-wise frequency differences,
and consensus features selected in at least three of five runs. Also report
the BPSO boundary best, BGWO phase best, best newly evaluated BGWO candidate,
final run best, BPSO-to-BGWO improvement, and first/final improvement
iterations.

These analyses are descriptive and cannot trigger feature reselection.

## Interpretation

Elite transfer receives endpoint-specific support when AP or F1 favors WITH
and its interval excludes zero, feasibility is preserved, and recall shows no
resolved harm. K may strengthen a conclusion but cannot establish it alone.

If an interval crosses zero, report direction uncertainty. If arms are
similar, conclude that the experiment does not demonstrate a resolved
elite-transfer contribution. If WITHOUT is better, report that directly. Do
not declare an overall algorithm winner.

## Relation To V1.0-F

V1.0-F remains frozen. V1.0-G will not repeat the 1,200-observation resource
campaign, use resource metrics as optimization objectives, or make physical
energy claims.

## Roadmap

- V1.0-G1: persist and lock this protocol.
- V1.0-G2: implementation and unit tests only.
- V1.0-G3: omitted by default.
- V1.0-G4: five fresh paired production runs.
- V1.0-G5: paired endpoint, stability, and convergence analyses.
- V1.0-G6: reporting and final ablation result lock.

Any pilot requires a reviewed protocol amendment and cannot use production
seeds.

## Fail-Closed Conditions

Stop if paired BPSO outputs differ; row 0 or rows 4-11 differ; the common BGWO
RNG states differ before updates; WITHOUT calls elite selection or inspects
BPSO information for filler construction; request counts drift; checkpoints
have the wrong protocol, arm, seed, or environment; TEST is loaded; frozen
evaluator/ranking/constraints/seeds drift; or V1.0-D/V1.0-F artifacts could be
overwritten.

## Semantic Lock

`results/hybrid/v10g/v10g_protocol_lock.json` hashes only its
`semantic_payload`. Canonicalization uses UTF-8 JSON with recursively sorted
keys, compact separators, ASCII escaping, finite numbers only, and no trailing
newline. Arrays are order-sensitive and paths are repository-relative POSIX
paths. Unknown semantic fields are prohibited.

The LF-normalized SHA256 values of this document and
`config/ablation_v10g.yaml` are inputs to the semantic payload. Neither
protocol artifact embeds the resulting semantic protocol-lock digest, avoiding
a circular hash dependency.
