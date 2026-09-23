#!/usr/bin/env python3
"""Governed execution launcher + persistence CLI for the V1.1-E4 campaign.

Orchestrates the frozen V1.1-E3 machinery into the E2B-confirmed output family
under ``results/blockchain/v11e/``. Authorization is supplied at runtime:
``--authorized-commit`` must equal current HEAD == origin/main.

Modes:
  preflight  fail-closed readiness (authorization + data gates + family state)
  plan       deterministic cell plan + workload/attack manifests (no execution)
  execute    run/resume the full plan (idempotent; validated cells are skipped)
  resume     alias for execute (same fail-closed idempotent engine)
  verify     read-only verification of an existing output family/result lock

Synthetic provider (``--synthetic``) is for unit/dev coverage only and never
produces governed evidence. Governed execution additionally requires
``--allow-governed-execution``.

Run with the repo virtualenv:
    .venv/bin/python scripts/run_v11e4.py preflight --authorized-commit <sha>
    .venv/bin/python scripts/run_v11e4.py plan      --authorized-commit <sha>
    .venv/bin/python scripts/run_v11e4.py verify    --authorized-commit <sha>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.pipeline_v11e4 as pipeline


def _ints_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "plan", "execute", "resume", "verify"))
    parser.add_argument("--authorized-commit", required=True, help="40-hex git commit == HEAD == origin/main")
    parser.add_argument("--base-dir", default=str(pipeline.V11E_DIR))
    parser.add_argument("--sizes", default=",".join(str(size) for size in pipeline.p.WORKLOADS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in pipeline.p.BLOCKCHAIN_SEEDS))
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument("--governed", action="store_true", help="governed provider (default)")
    provider_group.add_argument("--synthetic", action="store_true", help="tiny synthetic fixture (tests only)")
    parser.add_argument("--synthetic-orders", type=int, default=12)
    parser.add_argument("--allow-governed-execution", action="store_true",
                        help="sanctioned operator path: permits governed E01-E10 execution")
    parser.add_argument("--no-git-guard", action="store_true",
                        help="skip git HEAD/origin checks (unit/CI temp-dir runs only)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    sizes = _ints_csv(args.sizes)
    seeds = _ints_csv(args.seeds)
    require_git = not args.no_git_guard

    if args.synthetic:
        if args.synthetic_orders < 4:
            parser.error("--synthetic-orders must be at least 4")
        order_ids = [str(index) for index in range(1, args.synthetic_orders + 1)]
        levels = ["LOW", "HIGH"] + ["MEDIUM"] * (args.synthetic_orders - 2)
        provider = pipeline.SyntheticDataProvider(order_ids, risk_levels=levels)
        print(f"[v11e4l] synthetic fixture: {len(order_ids)} orders (TEST-only)")
    else:
        provider = pipeline.GovernedDataProvider()

    if args.mode == "preflight":
        report = pipeline.run_launcher_preflight(
            authorized_commit=args.authorized_commit,
            base_dir=base_dir,
            sizes=sizes,
            seeds=seeds,
            require_git=require_git,
        )
        print(f"[v11e4l] preflight {report['marker']}")
        print(f"  authorization: {report['authorization']['authorized_execution_commit'][:16]}...")
        print(f"  leak counts: {report['leak_counts']} leaks_zero={report['leaks_zero']}")
        print(f"  plan total cells: {report['plan']['total_cells']}")
        print(f"  family_state: {report['family_state']}")
        return 0

    if args.mode == "plan":
        authorization = pipeline.verify_authorized_execution_commit(
            args.authorized_commit, require_git=require_git
        )
        _workload, _attack, paths = pipeline._fill_family_artifacts(
            base_dir,
            provider=provider,
            sizes=sizes,
            seeds=seeds,
            write_outputs=True,
            require_clean_write=False,
        )
        plan = pipeline.build_execution_plan(sizes=sizes, seeds=seeds)
        print(f"[v11e4l] plan mode (no execution) authorized={authorization['authorized_execution_commit'][:16]}...")
        print(f"  cells: {len(plan)} executed-measurement="
              f"{sum(1 for s in plan if s.experiment_id != 'E10')} derived-E10="
              f"{sum(1 for s in plan if s.experiment_id == 'E10')}")
        print(f"  workload manifest: {paths['workload_manifest']}")
        print(f"  attack manifest:   {paths['attack_manifest']}")
        return 0

    if args.mode in ("execute", "resume"):
        allow = args.allow_governed_execution
        result = pipeline.run_execution_session(
            provider=provider,
            authorized_commit=args.authorized_commit,
            base_dir=base_dir,
            allow_governed_execution=allow,
            sizes=sizes,
            seeds=seeds,
            require_git=require_git,
        )
        print(f"[v11e4l] {args.mode} {result['marker']}")
        print(f"  executed: {len(result['executed_cells'])} skipped: {len(result['skipped_cells'])}")
        print(f"  result lock:       {result['result_lock_path']}")
        for path in sorted(result["summary_files"]):
            print(f"  summary: {path}")
        return 0

    if args.mode == "verify":
        report = pipeline.verify_result_family(
            base_dir,
            authorized_commit=args.authorized_commit,
            sizes=sizes,
            seeds=seeds,
            require_git=require_git,
        )
        print(f"[v11e4l] verify {report['marker']}")
        print(f"  cells: {report['found_cells']}/{report['expected_cells']}"
              f" summary_files={len(report['family']['summary_files'])}")
        if report["errors"]:
            for error in report["errors"][:10]:
                print(f"  ERROR: {error}")
        return 0 if report["well_formed"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())