# V1.0-A Hybrid BPSO+BGWO Feature-Selection Protocol Lock

- **Stage:** V1.0-A (protocol and design lock only)
- **Optimizer architecture:** sequential cooperative hybrid, BPSO exploration &rarr; elite knowledge transfer &rarr; BGWO refinement, within each run
- **Execution policy:** no hybrid optimizer implementation, no optimization execution, no model-training experiments, no resource benchmarking, no final-test access, no BPSO/BGWO reruns, no optimizer parameter tuning
- **Design classification:** `HYBRID_PROTOCOL_LOCKED_FAIR_BUDGET`

---

## 1. Frozen Evidence Verification (Artifact Source of Truth)

All values below were read directly from the frozen repository artifacts (not trusted from external summary), and will be asserted by `tests/test_protocol_v10a.py` against this document and `config/hybrid_v10.yaml`.

| Reference | Verified value | Source artifact |
|---|---|---|
| Predictive baseline AP | 0.23068406113411433 | `results/feature_selection/validation_lock.json` (V0.6-E) |
| Predictive baseline F1 | 0.1977010726398371 | same |
| Predictive baseline Recall | 0.322 | same |
| Preservation margins | AP &le; 5%, F1 &le; 5%, Recall &le; 10% relative loss | same (`preservation_rules.primary`) |
| Feature space | 43 features (manifest sha256 `5146fd08…`) | same |
| Model/attack seeds | 42, 43, 44, 45, 46 | same |
| Split seed | 42 (order-grouped 28k/6k/6k) | V0.6 governance |
| MI-K11 | smallest_preserving, K = 11 | `per_method_choices.mutual_information_select_k_best_k11` |
| BPSO winner seed / K | 1042 / 10 | `results/bpso/v08c_winner_lock.json` |
| BPSO budget | **972 candidate requests**, 972 unique evaluations, 4860 DT fits (5/request), optimizer wall &asymp; 645.503 s | `results/bpso/v08c_execution_summary.json` |
| BPSO per-run envelope | 132 / 228 / 192 / 216 / 132 &rarr; [132, 228] | `v08c_run_1042..1046.json` |
| BPSO config | 12 particles, cap 240 requests/run, 20 evaluated generations, early stop (patience 7, not before gen 10) | `v08c_run_1042.json` |
| BGWO winner seed / K | 2042 / 14 | `results/bgwo/v09d/v09d_winner_lock.json` |
| BGWO budget | **768 candidate requests**, 768 unique evaluations, 3840 DT fits (5/request), optimizer wall &asymp; 325.425 s | `results/bgwo/v09d/v09d_execution_summary.json` |
| BGWO per-run envelope | 132 / 204 / 156 / 144 / 132 &rarr; [132, 204] | `v09d_run_2042..2046.json` |
| BGWO config | 12 wolves, cap 240 requests/run, 20 evaluated iterations, early stop (patience 7, not before iter 10) | `v09d_run_2042.json` |
| Random cardinality anchors | [4, 8, 11, 14, 18, 22, 26, 30, 34, 38] | both execution summaries |
| Final-test participants (V0.9-E) | K43, K42, MI-K11, BPSO-K10, BGWO | `results/bgwo/v09e/v09e_result_lock.json` |
| Classifier locked params | DT (balanced, depth 5, min_samples_leaf 20, thr 0.5); LR (liblinear, C=1.0, balanced, max_iter 500, thr 0.5) | `v09e_result_lock.json` |

**Frozen-budget headline numbers (from artifacts):** BPSO = 972 total requests (4860 fits); BGWO = 768 total requests (3840 fits). Feature identity restrictions that remain frozen and out of reach of any search phase appear in Section 8.

---

## 2. Core Scientific Question

> Can a governed hybrid BPSO+BGWO cooperative search strategy obtain a more useful feature-selection trade-off than the standalone BPSO (V0.8) and BGWO (V0.9) searches under a controlled and comparable search budget, while preserving the frozen predictive-preservation constraints?

The question is answered **metric-specifically** (per-feasibility, per-K, per-predictive, per-resource metric), never by a single subjective headline score.

---

## 3. Hypotheses

- **H1 (feasibility).** The hybrid search identifies at least one feasible low-cardinality subset under the same frozen predictive-preservation constraints (AP &le; 5%, F1 &le; 5%, Recall &le; 10% relative loss vs V0.6 baseline).
- **H2 (budget-controlled benefit).** Any apparent benefit of the hybrid relative to BPSO and BGWO remains interpretable after controlling for the search/evaluation budget; primary production runs use a fixed, pre-locked budget so allocated equals consumed.
- **H3 (no universal superiority).** The hybrid may improve individual predictive, resource, or search metrics, but no universal superiority is assumed. Differences are reported metric-by-metric with matched statistics; unsupported superiority or "greener/less energy" claims are prohibited.

**Intent guard:** hypotheses are written so they cannot be satisfied merely by "no harm"; they are also written so that a modest hybrid result is not an experiment failure. There is no hypothesis asserting guaranteed improvement.

---

## 4. Frozen Objective, Margins, and Ranking

The objective is byte-identical to the frozen standalone objective (`constrained_minimum_cardinality`, `feature_fitness_is_better` ranking in `src/optimization/feature_fitness.py`). Timing, RSS, model size, energy, or throughput never enter optimizer fitness; they remain post-lock evaluation metrics only.

**Feasibility:** relative loss computed on five-seed validation means against the V0.6 baseline; feasible if all three losses are within margins (with the frozen `NUMERICAL_BOUNDARY_TOLERANCE`); `normalized_violation = sum(max(0, (loss−margin)/margin))`, zeroed when feasible.

| Tier | Rank order |
|---|---|
| Feasible | 1) minimum K &rarr; 2) higher AP &rarr; 3) higher F1 &rarr; 4) higher Recall &rarr; 5) canonical tie-break |
| Infeasible | 1) lower normalized feasibility violation &rarr; 2) higher AP &rarr; 3) higher F1 &rarr; 4) higher Recall &rarr; 5) lower K &rarr; 6) canonical tie-break |

Canonical tie-break = lexicographic comparison of the canonical binary mask over canonical feature indices 0..42 (matching `feature_fitness_is_better`).

---

## 5. Hybrid Design (One Locked Mechanism)

**Mechanism:** BPSO_BGWO_SEQUENTIAL_50_50_ELITE3 — a sequential cooperative hybrid executed **within a single run** under that run&rsquo;s optimizer seed:

```
BPSO exploration phase (96 requests)
        |
        v
controlled elite knowledge transfer (zero fitness evaluations)
        |
        v
BGWO refinement phase (96 requests)
```

This is a simple, defensible cooperative design. No new metaheuristic; each phase reuses the frozen, validated optimizer mechanics (BPSO and BGWO update equations, logistic transfer function, clamps) with phase RNG sub-streams and a fixed budget.

**BPSO phase role (exploration):** governed population search across the frozen cardinality anchors; its only deliverable is a deterministic ranking of the distinct masks it evaluated (the elite set).

**BGWO phase role (refinement):** standard alpha/beta/delta refinement whose initial population is elite-anchored (wolves 1..3) plus fresh governed random wolves (4..12) and the all-ones anchor (wolf 0).

---

## 6. Fair Search Budget — Primary Locked Value

**Total candidate-request budget (primary): 960 across the five production runs (192 per run).**

Derivation (mathematical):

1. Agent count = 12 per phase, matching both frozen standalone swarms (12 particles / 12 wolves) so per-evaluation parallelism and the per-iteration request shape are identical.
2. Phase ratio fixed at 50/50 for maximal methodological simplicity and symmetric fairness (no phase privileged by design).
3. Per-run budget R = 12 &times; (m + m) with integer phase iterations m. The feasible candidate values R &isin; {144, 168, 192} lie strictly inside the **common frozen per-run request envelope** [132, 204] = BPSO range [132, 228] &cap; BGWO range [132, 204].
4. R = 192 is the **largest** such value that is simultaneously (a) &le; the observed BPSO per-run mean (972/5 = 194.4) and (b) &ge; the observed BGWO per-run mean (768/5 = 153.6), and it decomposes cleanly into 96 BPSO + 96 BGWO requests (8 evaluated iterations per phase &times; 12 agents).
5. Five runs collide to a total of 960 requests, which is &le; the frozen BPSO total (972) and &gt; the frozen BGWO total (768). The fixed-budget rule makes **allocated == consumed**, forcing a clean, comparable budget interpretation and removing the early-stop confound present in the standalone runs (which consumed 132..228 per run).

| Budget quantity | Value |
|---|---|
| Total candidate requests (5 runs) | 960 |
| Requests per run | 192 |
| BPSO-phase requests per run | 96 |
| BGWO-phase requests per run | 96 |
| Population size per phase | 12 |
| BPSO evaluated generations per run | 8 (generation 0 = initialization evaluation) |
| BGWO evaluated iterations per run | 8 (iteration 0 = initialization evaluation) |
| Model fits per candidate | 5 (seeds 42..46) |
| Allocated DT fits total | 4800 (= 5 &times; 960); actual = 5 &times; unique evaluations after cache |

The phase ratio is **not** tuned using validation results. A ratio sensitivity experiment, if ever needed, is a separate secondary experiment and is excluded from primary winner selection.

---

## 7. Initialization

- **Wolf 0 (both phases):** canonical all-ones 43-mask (K=43); a-priori known, not data-derived.
- **Wolves 1..11 (initial population):** deterministic random exact-K masks; cardinalities drawn from the frozen set [4, 8, 11, 14, 18, 22, 26, 30, 34, 38]; feature identities generated under the phase RNG of the run (BPSO: `default_rng(seed)`; BGWO random init: `default_rng(seed + 10000)`).
- **Repair:** deterministic construction to exact cardinality; `all_zero_policy` = activate index with max absolute latent value then lowest-index tie-break; dry-run target repair count = 0.
- Cardinality anchors are **cardinality-only**. The K=11 and K=14 cardinality anchors are random masks drawn fresh under the run&rsquo;s own seed — never the MI-K11 or frozen BGWO-K14 feature identities.

---

## 8. Forbidden Injections (Explicit Denylist)

The search process must never receive:

1. BPSO-K10 frozen winner identity (`v08c_winner_lock`, mask_sha256 `5da981b5…`);
2. BGWO-K14 frozen winner identity (`v09d_winner_lock`, mask_sha256 `7ebb8233…`);
3. BGWO alpha/beta/delta latent positions or populations from any `v09d_run_*.json`;
4. BPSO initial population or velocity vectors from any `v08c_run_*.json`;
5. MI-K11 feature identity (`mutual_information_select_k_best_k11`);
6. the real K42 leave-one-out mask identity (`real_k42_mask_sha256`);
7. any TEST-derived information of any kind.

Only integrity metadata (hashes, counts) of frozen artifacts may be read by the search runner. A preflight dry-run must fail if any denylisted identity is resolvable from the search process.

---

## 9. Knowledge Transfer (Exact Semantics)

- **Elite count:** N = 3.
- **Elite ranking rule:** the frozen deterministic constrained ranking applied to **all distinct evaluated masks** of the BPSO phase of the same run.
- **Elite selection:** feasible-first by construction of the ranking; duplicate masks coalesced before ranking (distinct-mask set).
- **Fewer than N feasible elites:** elite slots are filled with the top distinct evaluated masks by the overall ranking (feasible-first semantics); any unfilled slot becomes a fresh governed random exact-K wolf.
- **Remaining wolves:** wolves 4..12 = fresh governed random exact-K masks under the BGWO phase RNG.
- **Duplicate handling:** later duplicate masks inside the BGWO phase are permitted; each is a counted candidate request whose evaluation is a cache hit (no new DT fits).
- **Budget accounting:** transferred elites occupy wolves 1..3, so their iteration-0 evaluation is counted inside the BGWO-phase 96-request allocation (3 requests/run). Transferred candidates **do** count against the evaluation budget.
- **Hidden evaluations:** zero fitness evaluations occur at the phase transition; elite selection is pure re-ranking of already-evaluated masks. No cross-phase, out-of-budget evaluator calls exist.

---

## 10. Cache and Accounting

- **Canonical mask representation:** uint8 binary mask over the canonical 43-feature manifest (index 0..42).
- **Mask hash:** `sha256(mask.tobytes())` — identical to the frozen `mask_sha256` convention.
- **Cache scope:** one optimizer run (both phases); cleared between runs; never shared across runs.
- **Cache-hit semantics:** request counted; no evaluator call; the previously recorded evaluation for that mask in the same run is reused.
- **Candidate request:** every mask the optimizer submits to the fitness layer — counted even when cached.
- **Unique evaluation:** a mask actually evaluated (fit-on-train / evaluate-on-validation) for the first time in the run; exactly 5 DT fits.
- **Decision-tree fit:** one fit of the frozen search classifier on train for one model/attack seed.

Expected audit result: 15 cache hits (3 elites &times; 5 runs) if no within-phase duplicates; actual values must reconcile as `unique = requests − cache_hits` and `fits = 5 × unique`.

---

## 11. Seeds

- **Optimizer family (primary):** 3042–3046 — verified free of repository conflict.
- **Optimizer family (ablation no-transfer arm):** same 3042–3046 (paired by seed across arms).
- **phase RNG offset:** BGWO random initialization uses `default_rng(seed + 10000)`.
- **Explicitly excluded:** BPSO 1042–1046, BGWO 2042–2046 (never used for any V1.0 run).
- **Model/attack seeds:** 42–46 (frozen). **Split seed:** 42 (frozen).

---

## 12. Early-Stopping Decision

**Decision: A — fixed-budget execution without early stopping.**

The standalone runs early-stopped at heterogeneous budgets (per-run 132..228 BPSO, 132..204 BGWO; totals 972 vs 768), which would confound a naive budget comparison. Fixed-budget execution makes **allocated == consumed** for every V1.0 primary run, so all three methods are compared on a clear evaluation budget. Both allocated and consumed budgets are reported anyway (they are equal in the primary experiment).

---

## 13. Validation-Only and TEST-Access Gate (Fail-Closed)

- Search may use TRAIN for fitting and VALIDATION for fitness only.
- TEST is inaccessible until: (a) all five hybrid runs finish; (b) winner selection completes; (c) the hybrid winner lock exists and verifies.
- Per-run audit fields: `test_accessed=false`, `test_authorized=false`, `test_used_for_winner_selection=false`; a `results/hybrid/v10d_test_access_audit.json` is written post-run.
- Preflight must record TEST identity hashes **without loading** TEST feature frames; the runner refuses to start if any gate flag is true.

---

## 14. Production Runs and Per-Run Report

Five independent hybrid runs (seeds 3042–3046), each reporting: seed, best K, AP, F1, Recall, feasible flag, normalized violation, requests, unique evaluations, cache hits, DT fits, phase-specific counts (BPSO-phase and BGWO-phase requests/unique/fits), runtime (per-phase and total), convergence (best-rank over evaluated iterations), and selected features. Artifact pattern: `results/hybrid/v10d_run_<seed>.json`.

---

## 15. Hybrid Winner Selection and Lock Schema

Winner selection uses validation only with the frozen deterministic ranking. The `results/hybrid/v10d_winner_lock.json` records: optimizer (`GOVERNED_HYBRID_BPSO_BGWO`), hybrid design (`BPSO_BGWO_SEQUENTIAL_50_50_ELITE3`), design version, seed, selected K, ordered feature list, mask sha256, selected-features sha256, validation metrics (AP/F1/Recall/precision/ROC-AUC/others), feasibility, normalized violation, budget (960 / 192 / 96 / 96), phase allocation, protocol hash (config SHA-256), and semantic winner lock sha256. No TEST access before this lock exists.

---

## 16. Final-Test Plan (V1.0-E — defined now, executed later)

Comparative one-time evaluation against: **K43, K42, MI-K11, BPSO-K10, BGWO-K14, Hybrid-V1.0 winner**.

- Classifiers: Decision Tree and Logistic Regression with the exact frozen locked parameters.
- Seeds 42–46 (paired by model/attack seed, shared split).
- Metrics: AP, F1, Recall, Precision, ROC-AUC (plus recorded accuracy, FPR, FNR, attack prevalence, PR-AUC).
- Paired seed statistics, n=5, df=4, &alpha;=0.05 (t_crit &asymp; 2.776), 95% CIs.
- TEST is **never** used to choose the hybrid winner.

---

## 17. Resource Plan (V1.0-F — after hybrid winner lock)

Primary comparison: K43, BPSO-K10, BGWO-K14, Hybrid-V1.0 winner; MI-K11 and K42 as secondary references. Reuses the accepted V0.9-F measurement methodology. Reported: training wall time, process CPU time, inference latency, throughput, RSS (absolute/incremental peak), input-memory footprint, serialized model size.

**Direct energy:** report Joules only if a valid measurement backend is available; otherwise emit `DIRECT_ENERGY_UNAVAILABLE`. Joules are never inferred from TDP &times; time.

---

## 18. Optimizer-Efficiency Comparison (V1.0-F)

V1.0 reporting must keep five constructs distinct and never conflate them: search quality, selected K, predictive performance, resource efficiency after selection, and optimizer search overhead. For BPSO, BGWO, and Hybrid, report: allocated requests, actual requests, unique evaluations, cache hits, DT fits, optimizer wall time.

---

## 19. Ablation Plan (V1.0-G — secondary, excluded from winner selection)

One controlled ablation: **hybrid without elite transfer** vs **hybrid with elite transfer**, under the identical total budget (960 candidate requests / five runs), paired by optimizer seed (3042–3046). The no-transfer arm runs the BPSO exploration phase identically but seeds the BGWO phase fully randomly (wolves 1..12, no elites). No hyperparameter sweep; validation-only; no TEST.

---

## 20. Statistics

- Predictive comparisons: paired by shared model/attack seeds (42..46), n=5, df=4, paired 95% CI, t_crit &asymp; 2.776.
- Optimizer-level metrics: BPSO (104x), BGWO (204x), and Hybrid (304x) use **different optimizer-seed families**, so cross-method optimizer comparisons are treated as **independent samples** — seeds are never falsely paired merely because every method has five runs.
- Ablation: paired by identical optimizer seeds across the two arms.
- No unsupported significance claims; reporting is metric-specific.

---

## 21. Claim Governance

Prohibited without directly supported, locked evidence: "Hybrid is universally superior", "Hybrid is greener", "Hybrid consumes less energy". Allowed language is metric-specific and tied to the locked statistical comparisons (e.g., "the hybrid winner reached feasible K=X with AP diff = +d (95% CI […]) on validation", "optimizer wall time of T vs T′ under fixed budget", "measured energy unavailable").

---

## 22. V1.0 Roadmap

1. **V1.0-A** — Protocol/design lock (this stage)
2. **V1.0-B** — Pure hybrid optimizer implementation + synthetic tests
3. **V1.0-C** — Real fitness integration + baseline reproduction + quarantined pilot
4. **V1.0-D** — Five production validation-only hybrid runs + winner lock
5. **V1.0-E** — Governed one-time final-test evaluation
6. **V1.0-F** — Resource efficiency + optimizer overhead
7. **V1.0-G** — Ablation
8. **V1.0-H** — Final notebook + integrated BPSO/BGWO/Hybrid synthesis

The prescribed order is retained. Reasoning: separating protocol (A) from implementation (B) before any real fitness contact (C), placing production (D) and final test (E) before resource (F) and ablation (G), and synthesizing last (H) isolates each selection decision from hindsight and prevents ablation/resource evidence from influencing production or winner choices.

---

## 23. Protocol Files and Hash

- Protocol document: `docs/v10_hybrid_bpso_bgwo_protocol.md` (this file)
- Implementation constants: `config/hybrid_v10.yaml`
- **Protocol-config SHA-256: `0ac491b50abeb60e65efc2295eb88c03b4c988a5182ff0e2e3ed729deb9367d8`**

The constant **`HYBRID_PROTOCOL_LOCKED_FAIR_BUDGET`** is declared only after the protocol passes all review checks (Section 24) — confirmed by `tests/test_protocol_v10a.py`.

---

## 24. Review Checklist (Audited in V1.0-A)

- no standalone (BPSO/BGWO/MI-K11/K42) winner injection: enforced by Section 8 denylist
- no TEST access: Section 13 fail-closed gate
- no optimizer execution, no code implementation: V1.0-A executes nothing but protocol checks
- matched/fair total search budget: 192/run &isin; common frozen envelope [132, 204]; total 960 &le; BPSO 972 and &gt; BGWO 768, fixed allocation (Section 6)
- exact phase budget: 96 BPSO + 96 BGWO per run (Section 6)
- no hidden evaluator calls: Section 9/10 accounting
- objective identical to standalone objective: Section 4
- same validation baseline and feasibility margins: Section 4
- new optimizer seeds: 3042–3046 (Section 11)
- deterministic ranking: Section 4
- deterministic winner-lock design: Section 15
- explicit optimizer accounting: Section 10/18
- explicit resource plan: Section 17
- honest energy policy: Section 17 (`DIRECT_ENERGY_UNAVAILABLE`; never TDP&times;time)
- ablation separated from winner selection: Section 19
- no subjective overall ranking: Sections 2/21

---

## 25. Classification

- **Final protocol classification:** `HYBRID_PROTOCOL_LOCKED_FAIR_BUDGET`
- **V1.0-B readiness intent:** GO after external review of this lock
- **No execution performed in V1.0-A**