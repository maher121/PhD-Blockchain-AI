#!/usr/bin/env python3
"""Real-file launcher for governed V0.8-E3 execution and worker smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROJECT_ROOT
from src.green.measurement import MeasurementPhase, run_fresh_worker_protocol
from src.pipeline_v08e import run_v08e


DEFAULT_CAMPAIGN_DIR = PROJECT_ROOT / "results" / "bpso" / "v08e_e3_campaign"
DEFAULT_CHECKPOINT_DIR = DEFAULT_CAMPAIGN_DIR / ".v08e_checkpoints"
DEFAULT_MODEL_DIR = DEFAULT_CAMPAIGN_DIR / "v08e_model_artifacts"
DEFAULT_SMOKE_DIR = PROJECT_ROOT / "results" / "bpso" / "v08e_e3r_engineering_smoke"


def _prepare_counter() -> list[int]:
    return [0]


def _increment_counter(state: list[int]) -> int:
    state[0] += 1
    return state[0]


def run_worker_smoke(*, output_dir: Path, start_method: str | None) -> dict[str, Any]:
    """Run one tiny fresh-worker observation in an isolated sandbox directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    run = run_fresh_worker_protocol(
        _increment_counter,
        prepare=_prepare_counter,
        inner_operation_count=4,
        phase=MeasurementPhase.INFERENCE.value,
        warmup_calls=3,
        outer_repetitions=1,
        records_per_operation=8,
        timeout_seconds=30.0,
        measurement_scope="engineering_worker_start_smoke",
        start_method=start_method,
    )
    result = run.results[0]
    payload = {
        "launcher_path": str(Path(__file__).resolve()),
        "status": result.status.value,
        "worker_pid": result.worker_pid,
        "warmup_calls_completed": result.warmup_calls_completed,
        "measured_operation_count": result.measured_operation_count,
        "failure_type": None if result.failure is None else result.failure.error_type,
        "failure_stage": None if result.failure is None else result.failure.stage,
        "failure_message": None if result.failure is None else result.failure.message,
    }
    (output_dir / "v08e_e3r_worker_smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-matrix", action="store_true", help="Run the governed 800-observation matrix resume.")
    parser.add_argument("--output-dir", default=str(DEFAULT_CAMPAIGN_DIR), help="V0.8-E3 campaign output directory.")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR), help="Checkpoint directory for cell-level resume.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Prepared model artifact directory.")
    parser.add_argument("--start-method", default=None, help="Optional multiprocessing start method override.")
    parser.add_argument(
        "--worker-smoke",
        action="store_true",
        help="Run one tiny fresh-worker engineering smoke test instead of the campaign.",
    )
    parser.add_argument(
        "--smoke-output-dir",
        default=str(DEFAULT_SMOKE_DIR),
        help="Output directory for the worker smoke artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_smoke:
        payload = run_worker_smoke(output_dir=Path(args.smoke_output_dir), start_method=args.start_method)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["status"] == "COMPLETED" else 1

    result = run_v08e(
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        model_dir=Path(args.model_dir),
        execute_matrix=bool(args.execute_matrix),
        start_method=args.start_method,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in {"INFRASTRUCTURE_READY", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
