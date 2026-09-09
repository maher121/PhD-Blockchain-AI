# V0.4 Controlled Cybersecurity Evaluation

## Scientific Scope

DataCo Smart Supply Chain supplies real operational transaction records. It does
not supply native cybersecurity attack labels. V0.4 therefore keeps three data
concepts separate:

1. Real operational records from the clean V0.2 test split.
2. Synthetic, controlled modifications made by the V0.4 attack generator.
3. Experimental ground-truth labels and manifests created before AI inference.

The resulting files are not a real cyber-attack dataset. They support controlled,
repeatable data-tampering experiments only.

## Experimental Order

```text
Clean DataCo data
  -> existing V0.2 train/validation/test split
  -> clean training data used by the V0.3 Isolation Forest
  -> controlled modification of clean test records
  -> experimental ground truth and attack manifest
  -> projection of eligible source changes into the exact V0.3 feature space
  -> V0.3 Isolation Forest inference
  -> supervised evaluation against experimental ground truth
```

AI predictions never select attacked records and never define ground truth.

## Implemented Scenarios

| Scenario | DataCo field used by default | V0.3 detector visibility |
|---|---|---|
| Quantity manipulation | `Order Item Quantity` | `order_item_quantity` |
| Order/product value manipulation | `Order Item Total` | `order_item_total` |
| Shipping/delivery manipulation | `Days for shipping (real)` | None; post-outcome field excluded by V0.2 |
| Transaction status manipulation | `Order Status` | None; outcome-adjacent field excluded by V0.2 |
| Timestamp manipulation | `order date (DateOrders)` | Six existing temporal features |
| Geographic/route manipulation | `Order Region` | None; traceability field excluded by V0.2 |
| Transaction record tampering | One eligible quantity/value field | Corresponding existing numeric feature |

A scenario is available only when at least one configured candidate field exists
and has eligible values. No absent DataCo column is synthesized. Changes to fields
that V0.2 excluded are intentionally not inserted into the AI feature matrix. Their
results measure the baseline's feature-bound visibility rather than model failure
on information it never receives.

## Severity Parameters

The authoritative configuration is `config/attack_scenarios.yaml`. Parameters are
expressed in source-field units.

| Scenario | LOW | MEDIUM | HIGH |
|---|---|---|---|
| Quantity | x0.90 or x1.10 | x0.60 or x1.50 | x0.25 or x3.00 |
| Monetary value | x0.90 or x1.10 | x0.70 or x1.35 | x0.40 or x2.00 |
| Shipping duration | -/+ 1 day | -/+ 3 days | -/+ 7 days |
| Status/route | one observed-state step | two observed-state steps | three observed-state steps |
| Timestamp | -/+ 1 hour | -/+ 24 hours | -/+ 168 hours |
| Generic record | x0.90 or x1.10 | x0.60 or x1.50 | x0.25 or x3.00 |

Quantity factors may produce fractional values; for DataCo item-count products,
that is an explicitly controlled quantity-domain inconsistency rather than a claim
of physical plausibility. Numerical shipping outputs remain non-negative.
Categorical changes use another state observed in the experiment frame. Because
status and route categories are nominal, their LOW/MEDIUM/HIGH step labels are
controlled transition variants, not an ordered claim that one state is more severe.
Timestamp outputs remain parseable. A high timestamp shift may deliberately
create an order/shipping inconsistency; this is a documented experimental scenario,
not a claim of physical plausibility.

## Rates and Selection

Configured rates are 1%, 3%, 5%, and 10% of eligible records. Selection uses
NumPy's deterministic generator with default seed 42. Reports distinguish eligible,
selected, and actually modified records and calculate the achieved rate against
eligible records. Mixed mode assigns distinct records across supported scenario
types, so one record has at most one experimental attack label per experiment.

## Traceability and Restoration

`src.security.attack_generator.generate_attack` returns the attacked copy, run
metadata, and a separate ground-truth frame. The manifest records experiment ID,
stable record ID, attack type, severity, source field, original value, modified
value, requested rate, and random seed. `restore_original` reverses supported
in-memory transformations from this manifest. The input frame is not mutated.

## Leakage Contract

The AI feature matrix is restricted to the exact 43 V0.3 feature names. It must
not contain `is_attack`, `attack_type`, `attack_severity`, `original_field`,
`original_value`, `modified_value`, `experiment_id`, `attack_rate`, or
`random_seed`. Both V0.3 feature governance and V0.4 runtime checks fail closed on
these names. Labels and manifests are saved separately from model inputs.

## AI and Blockchain Roles

The saved V0.3 Isolation Forest tests behavioral/statistical deviations in its
legitimate feature space. Precision, recall, F1, trapezoidal PR-AUC, average
precision, ROC-AUC, and the confusion matrix are valid in V0.4 because controlled
ground truth exists. Final-label metrics include records the baseline already
flagged before modification, so V0.4 also saves paired clean/attacked predictions,
newly induced detections, lost detections, and anomaly-score changes. Scenarios
that do not alter model inputs are marked as baseline overlap rather than credited
as attack-induced AI detection.

Direct numerical and temporal source edits are projected with affine parameters
recovered solely from aligned V0.2 **training** source/processed pairs. This
reproduces the existing train-fitted StandardScaler representation for those
features without fitting on test labels or attack outcomes. V0.2-excluded fields
are not projected.

The V0.1 ledger separately seals a clean transaction, verifies the chain, modifies
the transaction without recomputing hashes, and verifies again. Its expected result
is a valid clean chain and an invalid tampered chain. This demonstrates post-sealing
integrity evidence in the simulated hash chain. It does not provide consensus,
signatures, external anchoring, or protection against an actor able to recompute the
entire chain.

The combined experiment reports both signals without claiming that one mechanism
is universally superior and without introducing a decision engine.

## Reproducibility

Run from the repository root:

```bash
python -m src.pipeline_v02
python -m src.pipeline_v03
python -m src.pipeline_v04
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_cybersecurity_attack_scenarios.ipynb
python -m pytest tests/ -v
```

V0.4 requires the ignored/regenerable V0.2 processed splits and V0.3 model artifact,
so a fresh workspace must run the stages in the order shown above. The same source
data, configuration, attack rate, severity, experiment ID, and seed reproduce attack
selection, modifications, manifests, projected features, ground truth, and equivalent
model outputs. Runtime measurements may vary by machine and load. The current
experiment uses the V0.2 deterministic 40,000-row cap and its 6,000-row test split.

## Limitations

1. Synthetic attacks are not equivalent to real attacks.
2. Attack distributions are controlled by experiment parameters.
3. Isolation Forest remains an unsupervised baseline with its existing threshold.
4. DataCo is not a dedicated cybersecurity dataset.
5. Results describe controlled detection behavior, not universal cybersecurity effectiveness.
6. Some valid tampering scenarios affect fields intentionally excluded from AI for leakage safety.
7. The blockchain remains a lightweight single-authority research simulation.
