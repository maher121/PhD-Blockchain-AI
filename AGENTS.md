# AGENTS.md

Governed, staged PhD research repo: a supply-chain security/blockchain–AI study with strict
reproducibility artifacts. `README.md` documents only V0.1–V0.5; read `docs/PhD_PRD.md` and the
protocol docs under `docs/` for the governed V0.6–V1.0 stages, and trust `src/` + `tests/` +
`config/` as the source of truth when prose conflicts.

## Environment

- Two virtualenvs exist. Use `.venv/` on Linux/WSL, `.venv-win/` on Windows (prefer `.venv` here).
- Python 3.12. `requirements.txt` pins exact versions. `scipy`, `nbclient`, `joblib` are installed
  in `.venv` but are **not** pinned in `requirements.txt`.
- Run everything as `.venv/bin/python -m ...` — never the bare `python`/`pytest`.

## Commands

- Single/focused test file: `.venv/bin/python -m pytest tests/test_pipeline_v10d.py -q`
- Known-slow files (avoid unless the target is really there):
  - `tests/test_pipeline_v10c.py` — ~17.5 min at checkpoint `750cc9c` (68 tests, real pilot campaign)
  - Full suite at checkpoint `750cc9c` was **1222 passed / 1 skipped in ~33 min** — never run it as a
    routine check; these counts drift as stages are added.
- Real-dataset integration test is skipped unless gated:
  `RUN_DATACO_INTEGRATION=1 .venv/bin/python -m pytest tests/test_dataco_pipeline.py -v`
- No lint/typecheck/format tooling, pre-commit, or CI exists in this repo. `pytest` is the only gate.
- Headless notebook runs use nbformat + nbclient (e.g. `nbclient.NotebookClient`), which are
  installed; nbconvert is also available.
- Expensive experiment/campaign launchers live in `scripts/` (e.g. `run_v10f.py`) and require the
  real data staged under `data/raw/` (present). Do not launch them casually.

## Architecture

- Each stage is one commit (`V0.6` … `V1.0-H`) with a matching trio:
  `src/pipeline_vXX.py` orchestrator + `tests/test_pipeline_vXX.py` + artifacts under
  `results/<stage>/`. Pinned configs are in `config/*.yaml`.
- Pipelines are **HEAD-guarded**: many assert an exact `EXPECTED_HEAD`/`expected_head_sha256`
  and refuse to run from a different commit. Check the guard before running a pipeline.
- Canonical artifact naming: `*_result_lock.json` carries a `semantic_result_lock_sha256` computed
  over a canonical JSON encoding (sorted keys, compact separators, `ensure_ascii`, no trailing
  newline). Fingerprints of upstream artifacts are re-verified when locking.
- Data layout (frozen): 40,000-row cap (`DATACO_MAX_ROWS`), order-grouped split 28k/6k/6k,
  split seed 42, 43-feature canonical space, shared model/attack seeds 42–46. `src/config.py` is
  the single source of these constants (`GLOBAL_SEED=42`).
- `results/`, `data/raw/`, `data/processed/`, `models/**/*.joblib` are gitignored but, as of
  checkpoint `750cc9c`, 214 result artifacts are force-tracked. **New governed artifacts require
  `git add -f`** (a plain `git add` is silently ignored).

## Workflow conventions

- Do not run or re-run experiments from notebooks: `notebooks/v06`–`v10*.ipynb` are read-only
  synthesis notebooks that read `results/` plus `config/` and assert persisted values.
- Protected worktree entries — do not modify or stage them:
  - `notebooks/02_dataco_audit.ipynb` (currently `M`)
  - `h`, `session_plan.md`, `session_plan_v07.md` (untracked session material)
- Never `git add .` / `git add -A`, `git restore ./`, `git reset --hard`, or `git clean`; staged
  results need explicit `git add -f` on the exact files.
- Commit style is one-stage-per-commit with subject like
  `Complete V1.0-D hybrid validation search and winner lock`.
- Energy claims are strictly governed: direct energy measurement was unavailable in V1.0-F
  (`DIRECT_ENERGY_UNAVAILABLE`); CPU/wall time are never presented as measured Joules, and
  TDP-times-time inference is forbidden in prose.

## Verification loop

After changing a pipeline/test, validate with the narrowest relevant file(s), then re-run any
guarded reporting/locking stages in dependency order and confirm their fingerprints and
`expected_head` assertions still hold before committing. Notebooks referenced by a stage's results
should be re-executed headlessly if their inputs changed.
