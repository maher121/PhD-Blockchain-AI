#!/usr/bin/env python3
"""Real-file launcher for the governed V1.0-F Resource Efficiency and
Computational Overhead Evaluation campaign (120 cells / 1200 observations)."""

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
from src.pipeline_v10f import run_v10f


DEFAULT_CAMPAIGN_DIR = PROJECT_ROOT / "results" / "hybrid" / "v10f"
DEFAULT_CHECKPOINT_DIR = DEFAULT_CAMPAIGN_DIR / ".v10f_checkpoints"
DEFAULT_MODEL_DIR = DEFAULT_CAMPAIGN_DIR / "v10f_model_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-matrix", action="store_true", help="Run the governed 1200-observation matrix.")
    parser.add_argument("--output-dir", default=str(DEFAULT_CAMPAIGN_DIR), help="V1.0-F campaign output directory.")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR), help="Checkpoint directory for cell-level resume.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Prepared model artifact directory.")
    parser.add_argument("--start-method", default=None, help="Optional multiprocessing start method override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result: dict[str, Any] = run_v10f(
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