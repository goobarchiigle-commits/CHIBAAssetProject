"""
run_runtime_orchestrator.py
Phase 5A — Future Leader Runtime Orchestrator CLI.

Usage:
  python run_runtime_orchestrator.py                    # run today's pipeline
  python run_runtime_orchestrator.py --replay           # verify replay
  python run_runtime_orchestrator.py --verify-replay    # alias
  python run_runtime_orchestrator.py --force-soft-freeze
  python run_runtime_orchestrator.py --force-hard-freeze
  python run_runtime_orchestrator.py --reconcile
  python run_runtime_orchestrator.py --date 2026-05-26

PowerShell checkpoint inspection:
  Get-Content logs/runtime/runtime_orchestrator/runtime_checkpoints_*.jsonl
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.runtime.future_leader_runtime_orchestrator import cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())
