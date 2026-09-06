# Prototype V0.1 + V0.2

**A Lightweight and Green Blockchain–AI Framework for Secure and Energy-Efficient Supply Chain Management**

> **Status: Prototype V0.1 + V0.2** — a reproducible research foundation. V0.2
> integrates the **real DataCo SMART Supply-Chain dataset** with a rigorous,
> leakage-safe preprocessing pipeline. Later phases (NSGA-II/MOPSO/PSO feature
> optimisation, deep learning, PostgreSQL/FastAPI, real Ethereum/Hyperledger
> deployment, 1M-transaction experiments) are intentionally **not** implemented
> yet.

---

## 1. Research Objective

The overall project investigates a lightweight, energy-aware framework that
jointly uses **AI-based anomaly detection** and a **ledger** (blockchain) to
secure supply-chain transactions while minimising compute. Prototype V0.1 lays
the minimum reproducible skeleton on top of which the later optimisation and
deployment phases will build.

## 2. Prototype Scope (V0.1)

Implemented in V0.1:

- [x] **Source-aware data loading** — local CSV cache, else real DataCo
      SMART Supply-Chain dataset downloaded from Kaggle via `kagglehub`,
      else deterministic synthetic fallback
- [x] Basic preprocessing (deduplication, missing / non-positive handling)
- [x] Feature engineering (numeric matrix + derived features)
- [x] Baseline anomaly detection (**Isolation Forest**) with honest, optional
      label-based metrics
- [x] Lightweight **simulated** blockchain ledger (hash-chained blocks)
- [x] SHA-256 transaction hashing (`hashlib`)
- [x] Transaction / block / chain validation and integrity verification
- [x] Tamper detection (transaction edit, block rehash, broken link)
- [x] Execution-time, CPU and memory measurement
- [x] Clearly-separated **energy estimation** interface (Energy = Power × Time)
- [x] Reproducible experiment structure (fixed seeds, JSON reports)
- [x] Jupyter notebook driving `src/`
- [x] Unit tests (46 passing)

Explicitly **out of scope** for V0.1: NSGA-II, MOPSO, PSO/GWO feature
optimisation, deep learning, PostgreSQL, FastAPI, real Ethereum/Hyperledger,
smart contracts on a live chain, and 1M-transaction experiments.

## 2A. Prototype V0.2 — Real DataCo Dataset Pipeline

V0.2 replaces V0.1's synthetic sample with the **real DataCo SMART Supply-Chain
for Big Data Analysis** dataset (`shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`,
53 columns, 180,519 rows) under `data/raw/DataCoSupplyChainDataset.csv`.
The file is staged from the local `kagglehub` cache — V0.2 **never downloads or
fabricates data** and never auto-downloads at import time.

### What V0.2 does

- [x] **Discovery** (`src/data/dataset_discovery.py`) — signature-based detection
      of the real DataCo file (`DataCo` signatures in `src/config.py`), with a
      provenance trust chain (`raw_dir` / `kagglehub_cache` / `none`).
- [x] **Audit** (`src/data/audit.py`) — missing values, duplicates, dtypes,
      numeric/categorical/datetime/identifier roles, quality checks, leakage
      candidates; outputs JSON/CSV + PNG figures under `results/data_audit/`.
- [x] **Versioning** (`src/data/versioning.py`) — sha-256 of the raw file,
      processing environment, split config, identity-overlap checks and feature
      list → `data/processed/dataset_metadata.json`.
- [x] **Leakage-safe split** (`src/preprocessing/leakage.py`) — the split
      happens **before** any fitting. Default `order_grouped` keeps every
      `Order Id` entirely inside one split (no order straddles train/val/test).
      Alternatives: `random`, `chronological`.
- [x] **Fit-on-TRAIN-only preprocessing** (`src/preprocessing/dataco_pipeline.py`)
      — numeric medians, non-negative masking of impossible negatives,
      categorical one-hot (cardinality ≤ 15; higher-cardinality columns stay
      metadata-only), order-level aggregates derived from the train split only,
      `StandardScaler` fitted on train, constant/all-NaN columns reported and
      dropped; validation/test are **transformed** with train-fitted statistics.
- [x] **Feature engineering** (`src/preprocessing/feature_engineering_v2.py`)
      — 17 documented features (9 transaction/price, 6 temporal from the order
      date, 2 order-level) + one-hot of `Type`, `Market`, `Shipping Mode`,
      `Customer Segment`, `Department Name`.
- [x] **Versioned outputs** (`src/pipeline_v02.py`) — `features.csv`,
      `metadata.csv`, `target.csv` per split under `data/processed/{train,
      validation,test}/`, plus `docs/data_dictionary.md` and
      `results/dataco_pipeline_v02_report.json`.
- [x] **Deterministic, capped** at `DATACO_MAX_ROWS = 40000` (fixed seed 42)
      **before** the split — honest probabilities, reproducible bytes.
- [x] **Tests** — fixture-based unit tests (no real-file dependency) with an
      `integration`-marked real-dataset test (run with `RUN_DATACO_INTEGRATION=1`).

### Not in scope for V0.2

Purposely **excluded from V0.2** (documented, not silently done): cybersecurity
attack/anomaly generation, native attack labels (the real dataset has none),
advanced anomaly-detection modelling, feature optimisation, PSO, GWO, NSGA-II,
MOPSO, and advanced energy optimisation of the DataCo pipeline. `Late_delivery_risk`
is an **operational delivery outcome** (late vs on-time), not a cyber label.

### Run

```bash
python -m src.pipeline_v02            # orchestrates discovery→audit→split→preprocess→save
jupyter nbconvert --to notebook --execute --inplace notebooks/02_dataco_audit.ipynb
RUN_DATACO_INTEGRATION=1 python -m pytest tests/test_dataco_pipeline.py -v   # real-file integration test
```

## 3. Architecture

```
dataset resolution (DATA_SOURCE=auto)
  local CSV -> [src/data] loader
  | else Kaggle DataCo (kagglehub) -> normalized + cached locally
  \_ else synthetic generator (offline fallback)
then --> [src/preprocessing]  cleanup + feature engineering
     --> [src/ai]            Isolation Forest detector + optional metrics
     --> [src/blockchain]    transactions -> blocks -> chain
     --> [src/security]      integrity checks + tamper simulation
     --> [src/energy]        measured metrics + energy estimation
     --> [src/evaluation]    JSON experiment reports
```

A single orchestrator (`src/pipeline.py`) composes all modules into one
reproducible run; the notebook calls the same functions cell-by-cell.

## 4. Project Structure

```
data/             raw/, processed/, synthetic/ datasets (.csv)
src/
    config.py               constants, paths, seeds, schemas (+ V0.2 DataCo roles)
    data/                   generator + loading + dataset_discovery + audit + versioning
    preprocessing/          cleaning + feature engineering (+ leakage, dataco_pipeline)
    ai/                     anomaly_detection.py (Isolation Forest)
    blockchain/             crypto.py, core.py, validation.py
    security/               integrity.py (tamper detection facade)
    energy/                 measurement.py, estimation.py
    evaluation/             reporting.py (JSON results)
    pipeline.py             V0.1 orchestrator
    pipeline_v02.py         V0.2 orchestrator (real DataCo end-to-end)
notebooks/          01_prototype_pipeline.ipynb, 02_dataco_audit.ipynb
experiments/      (later phase experiment driver work)
models/           fitted estimators (.joblib, gitignored)
results/          JSON experiment + V0.2 audit reports (gitignored)
tests/            pytest suites (V0.1 46 + V0.2 45 = 91 passing, 1 skipped)
docs/             data_dictionary.md (generated), later-phase design docs
requirements.txt, README.md, .gitignore
```

> **Note on data/raw**: `data/raw/` and `data/processed/` are gitignored and
> regenerable; V0.2 stages the real DataCo CSV from the local `kagglehub` cache.

## 5. Installation

Requires Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: register the notebook kernel.

```bash
python -m ipykernel install --user --name prototype-v01 --display-name "Python 3 (prototype-v01)"
```

### Dataset access (optional, for the real Kaggle table)

With the default `DATA_SOURCE="auto"` the pipeline first looks for a local CSV
at `data/raw/supply_chain_raw.csv`. If none exists it downloads the real
*DataCo SMART Supply-Chain for Big Data Analysis* dataset
(`shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`) via
`kagglehub`, normalises it to the canonical schema and caches the result
under `data/raw/` so later runs are offline.

Public download usually needs no credentials. If Kaggle asks for consent or
credentials, either:

```bash
export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx    # from kaggle.com/settings/api
```

or place a `kaggle.json` (with `{"username": ..., "key": ...}`) in `~/.kaggle/`.

If the download fails (no network / no credentials), the pipeline falls back to
the deterministic synthetic generator and logs a clear warning — the framework
never hard-requires network access.

## 6. Running the Notebook

From the repository root:

```bash
jupyter notebook notebooks/01_prototype_pipeline.ipynb   # V0.1: AI + blockchain skeleton
jupyter notebook notebooks/02_dataco_audit.ipynb         # V0.2: real-dataset audit + split + features
```

or, if the `prototype-v01` kernel is registered, select *Python 3
(prototype-v01)* as the kernel. `01` demonstrates the whole V0.1 workflow
(data → preprocessing → features → Isolation Forest → blockchain →
verification → tamper detection → measurement → energy estimate → saved
results). `02` demonstrates the full V0.2 real-dataset workflow (discovery →
loading → audit → leakage analysis → split → feature engineering → save) in 16
sections and calls only `src/` functions.

Reproducibly headless-execute either notebook:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_prototype_pipeline.ipynb \
  --ExecutePreprocessor.kernel_name=prototype-v01
python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/02_dataco_audit.ipynb
```

## 7. Running Tests

```bash
python -m pytest tests/ -v
# or the V0.2 real-dataset integration test:
RUN_DATACO_INTEGRATION=1 python -m pytest tests/test_dataco_pipeline.py -v
```

The suite covers transaction hashing, transaction validation, block hashing,
block validation, blockchain verification and tamper detection, plus the data,
AI and energy modules, and the V0.2 data-discovery/audit/split/preprocessing/
pipeline modules (91 passing, 1 integration test skipped without the env flag).
A dedicated end-to-end test asserts that editing a transaction **after** block
creation causes integrity verification to fail.

## 8. Limitations

Honest boundaries of Prototype V0.1:

- **Blockchain is a research simulation.** No consensus, mining, propagation,
  digital signatures, node synchronisation or adversarial models. Hashes give
  *tamper-evidence*, not attacker resistance.
- **Energy estimates are not measurements.** `src/energy/estimation.py`
  converts *measured* wall-time into an energy *estimate* using a documented
  typical-TDP power convention (Energy = Power × Time). It is a research proxy.
  Measured metrics (time, CPU %, RSS memory) are always kept distinct and are
  never called "energy".
- **No cybersecurity attack labels exist.** Depending on the source, the label
  column is either `known_anomaly` (synthetic/controlled anomalies injected by
  the generator) or `natural_label` (the DataCo Kaggle table's real
  `Late_delivery_risk` flag). Every JSON report records `label_kind`
  (`controlled_synthetic`, `natural_late_delivery_risk` or `none`). Metric
  results describe recovery of that label source only.
- **Sampled features for the full Kaggle table.** The full 180k-row table is
  loaded and cleaned, but feature engineering (and the anomaly / blockchain
  stages) run on a fixed-seed sample (`PROCESS_MAX_ROWS=20000`) to keep
  notebook runtimes reasonable. Tune the cap in `src/config.py`.
- **Single-threaded, small scale.** No real datasets beyond the cached Kaggle
  table, no distributed storage.

## 9. Reproducibility

- Fixed random seed (`src/config.GLOBAL_SEED = 42`) across the synthetic
  generator, the isolation forest `random_state`, the Kaggle-table sampling,
  and the V0.2 cap/split/preprocessor.
- Deterministic timestamp/quantity bases in the synthetic generator.
- V0.2 stages only from the local `kagglehub` cache and records the raw file's
  sha-256 + provenance (`dataset_metadata.json`); no automatic downloads.
- The source used for every run is recorded (`data_source` /
  `label_kind` in the JSON report).
- Every run writes a self-describing JSON report (timestamp, Python/platform,
  metrics, measurement, energy estimate) under `results/`.
- `data/raw`, `data/processed`, `models/` and `results/` are gitignored because
  they are regenerable from `src` (or downloadable via `kagglehub`).

## 10. Future Development Phases

1. **V0.3 – Baseline AI anomaly detection:** train/evaluate detectors
   (Isolation Forest, autoencoders) on the V0.2 leakage-free features with
   honest-label reporting (delivery-outcome proxy).
2. **V0.4 – Optimisation & deeper AI:** NSGA-II / MOPSO / PSO/GWO feature
   selection and hyperparameter tuning; deep-learning detectors.
3. **V0.5 – Persistence & service:** PostgreSQL storage, FastAPI endpoints.
4. **V0.6 – Real ledger:** Ethereum/Hyperledger smart contracts and deployment.
5. **V1.0 – Scale:** 1M-transaction experiments and energy benchmarking at scale.

---

Prototypes V0.1 and V0.2 provide the clean, modular, reproducible foundation
these later phases will extend.