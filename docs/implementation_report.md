# Prototype V0.1 Implementation Report

- **Date:** 2026-09-05
- **Environment:** Python 3.12.3, Linux
- **Tests:** all passing (46)

## Delivered modules

| Area | Module | Responsibilities |
|------|--------|------------------|
| Config | `src/config.py` | paths, constants, seeds, schemas, Kaggle mapping |
| Data | `src/data/generator.py` | deterministic synthetic dataset (offline fallback) |
| Data | `src/data/loading.py` | source-aware loader: local CSV → Kaggle (`kagglehub`) → synthetic fallback; Kaggle column normalisation + local cache |
| Preprocessing | `src/preprocessing/preprocessing.py` `feature_engineering.py` | cleaning; numeric feature matrix |
| AI | `src/ai/anomaly_detection.py` | Isolation Forest wrapper; label-optional metrics |
| Blockchain | `src/blockchain/*` | crypto, core, validation |
| Security | `src/security/integrity.py` | integrity facade + tamper simulation |
| Energy | `src/energy/*` | measured metrics + energy estimation |
| Evaluation | `src/evaluation/reporting.py` | JSON result reports |
| Orchestration | `src/pipeline.py` | end-to-end V0.1 driver (source-aware) |

## Data-source behaviour

With `DATA_SOURCE="auto"` (the default):

1. If `data/raw/supply_chain_raw.csv` exists -> loaded locally (offline-safe).
2. Else the real DataCo *Smart Supply Chain for Big Data Analysis* dataset
   (`shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`,
   `DataCoSupplyChainDataset.csv`, 180,519 rows) is downloaded via
   `kagglehub` (Pandas adapter, latin-1 encoding), normalised onto the
   canonical `RAW_SCHEMA`, and cached to `data/raw/`.
3. If the download fails (no network / credentials), a clear warning is logged
   and the deterministic synthetic generator is used.

Verified live run on this machine: **180,519 rows** normalised; pipeline
completed with `data_source=kaggle` (cached thereafter as `local`).

## Label policy (scientific honesty)

- `known_anomaly` — synthetic/controlled anomalies injected by the generator.
- `natural_label` — the DataCo Kaggle table's real `Late_delivery_risk`
  (0/1) flag, preserved during normalisation.
- Neither kind is a cybersecurity attack label; reports record `label_kind`
  and metrics are only computed when a valid binary label exists.

## Key verifications

- 46 pytest tests pass, including the tamper-after-sealing integrity test and
  the offline (mocked) Kaggle-source tests: local reuse, download+cache,
  synthetic fallback, hard-fail on explicit `source="kaggle"`, label
  provenance through the pipeline.
- `notebooks/01_prototype_pipeline.ipynb` executes end-to-end with zero errors
  (42 cells), calling only `src/` functions. On the real Kaggle table:
  `label_kind=natural_late_delivery_risk`, 20k seeded feature sample from
  180,519 processed rows, Isolation Forest test metrics vs the natural label
  (e.g. F1 ≈ 0.56, ROC-AUC ≈ 0.55), 405 blocks / 3,230 flagged transactions.
- A tamper simulation (illegal quantity edit) is reliably detected by both
  `verify_chain` (False) and `detect_tampering` (True).
- Full run measurements are captured (e.g. wall-time, peak CPU %, peak RSS) and
  persisted; energy is reported as an **estimate** using `Energy = Power × Time`
  with a documented typical-TDP convention.

## Honesty / scientific boundaries

- Blockchain is a simulation (no consensus/signatures/network); provides
  tamper-evidence only.
- Energy values are estimates, never physical measurements.
- Labels are controlled/synthetic or the Kaggle `Late_delivery_risk` flag;
  never cybersecurity attack labels.

## Not implemented (deferred phases)

NSGA-II, MOPSO, PSO/GWO optimisation, deep learning, PostgreSQL, FastAPI, real
Ethereum/Hyperledger deployment, smart contracts, 1M-transaction experiments.