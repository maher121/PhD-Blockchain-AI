#!/usr/bin/env python3
"""Governed launcher for the V1.1-D3 governed AI risk generation campaign.

Runs the fail-closed D3 preflight, authorizes execution (the single sanctioned
producer of an execution capability), opens the module-level execution gate
ONLY after a fully passing preflight, executes the full deterministic
generation over the CLEAN frozen VALIDATION partition (4,588 canonical
orders), and prints the frozen 33-item final report ending in the
review-required marker. This launcher never commits or pushes.

Run with the repo virtualenv:
    .venv/bin/python scripts/run_v11d3.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai_risk.protocol import load_verified_protocol
import src.pipeline_v11d3 as pipeline


def main() -> int:
    pipeline.GOVERNED_RISK_GENERATION_AUTHORIZED = False
    preflight = pipeline.run_v11d3_preflight()
    print(f"[v11d3] preflight PASS: {preflight.semantic_sha256[:16]}...")

    protocol = load_verified_protocol()
    capability = pipeline.authorize_d3_execution(preflight)
    pipeline.GOVERNED_RISK_GENERATION_AUTHORIZED = True
    try:
        result = pipeline.run_governed_risk_generation(capability)
    except BaseException:
        pipeline.GOVERNED_RISK_GENERATION_AUTHORIZED = False
        raise
    pipeline.GOVERNED_RISK_GENERATION_AUTHORIZED = False

    summary = json.loads(pipeline.D3_SUMMARY_PATH.read_text(encoding="utf-8"))
    report = pipeline.build_final_report(
        preflight=preflight,
        protocol=protocol,
        summary=summary,
        record_count=result["record_count"],
        artifact_path=result["artifact_path"],
        artifact_file_sha256=result["artifact_file_sha256"],
        summary_path=result["summary_path"],
        summary_file_sha256=result["summary_file_sha256"],
        result_lock_path=result["result_lock_path"],
        result_lock_file_sha256=result["result_lock_file_sha256"],
        references_verified=result["references_verified"],
    )
    for item in report:
        print(
            f'  {item["code"]} [{item["status"]}] {item["label"]}: '
            f'{item["detail"]}'
        )
    print(f'  final report: {sum(1 for item in report if item["ok"])}'
          f'/{len(report)} PASS')
    print(pipeline.FINAL_REPORT_MARKER)
    return 0 if all(item["ok"] for item in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())