"""
src/runtime/determinism/invariant_validator.py

RuntimeInvariantValidator: minimal fail-closed invariant checker.
DeploymentGate:            always returns BLOCKED during validation phase.

Invariants validated:
1. append-only ledger integrity   (entry count never decreases)
2. idempotent rerun equality      (same run_id → same replay hash)
3. deployment authority isolation (gate always returns BLOCKED)
4. artifact lineage presence      (required artifact paths exist on disk)

On invariant failure:
- appends immutable report to artifacts/runtime_validation/invariant_reports.jsonl
- raises InvariantError (fail-closed: record then abort)

Module-level constants broadcast validation-phase block state:
  DEPLOYMENT_GATE_FORCE_BLOCK    = True
  LIVE_ORDER_SUBMISSION_DISABLED = True
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

INVARIANT_REPORTS_PATH: Path = Path("artifacts/runtime_validation/invariant_reports.jsonl")

DEPLOYMENT_GATE_FORCE_BLOCK:    bool = True
LIVE_ORDER_SUBMISSION_DISABLED: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────────────────────

class InvariantError(RuntimeError):
    def __init__(self, message: str, invariant: str, detail: Any = None) -> None:
        super().__init__(message)
        self.invariant = invariant
        self.detail    = detail


# ─────────────────────────────────────────────────────────────────────────────
# DeploymentGate
# ─────────────────────────────────────────────────────────────────────────────

class DeploymentGate:
    """
    Validation-phase deployment gate.
    ALWAYS returns BLOCKED. Live order submission is disabled.
    No parameters or conditions can unblock this gate.
    """

    DEPLOYMENT_GATE_FORCE_BLOCK:    bool = True
    LIVE_ORDER_SUBMISSION_DISABLED: bool = True

    def evaluate(self, *_args: Any, **_kwargs: Any) -> dict:
        return {
            "status":  "BLOCKED",
            "reason":  "validation_phase_force_block",
            "blocked": True,
        }

    def is_blocked(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeInvariantValidator
# ─────────────────────────────────────────────────────────────────────────────

class RuntimeInvariantValidator:
    """
    Minimal fail-closed validator for core runtime invariants.

    Each validate_* method is a standalone check.
    Any failure → report appended → InvariantError raised.
    """

    def __init__(self, reports_path: Path = INVARIANT_REPORTS_PATH) -> None:
        self._path = Path(reports_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── invariant checks ──────────────────────────────────────────────────────

    def validate_ledger_append_only(
        self,
        prior_count:   int,
        current_count: int,
        run_id:        str = "",
    ) -> None:
        """Ledger entry count must not decrease."""
        if current_count < prior_count:
            self._fail(
                invariant="append_only_ledger",
                message=(
                    f"ledger shrank: prior={prior_count} current={current_count}"
                ),
                run_id=run_id,
                detail={"prior_count": prior_count, "current_count": current_count},
            )

    def validate_idempotent_rerun(
        self,
        run_id: str,
        hash_a: str,
        hash_b: str,
    ) -> None:
        """Same run_id must produce identical replay hash."""
        if hash_a != hash_b:
            self._fail(
                invariant="idempotent_rerun_equality",
                message=(
                    f"hash divergence run_id={run_id}: "
                    f"{hash_a[:12]}… ≠ {hash_b[:12]}…"
                ),
                run_id=run_id,
                detail={"hash_a": hash_a, "hash_b": hash_b},
            )

    def validate_deployment_authority(self, gate: DeploymentGate) -> None:
        """Deployment gate must always return BLOCKED."""
        result = gate.evaluate()
        if not result.get("blocked", False) or result.get("status") != "BLOCKED":
            self._fail(
                invariant="deployment_authority_isolation",
                message="deployment gate did not return BLOCKED",
                run_id="",
                detail={"gate_result": result},
            )

    def validate_artifact_lineage(
        self,
        artifact_paths: list[str],
        run_id:         str = "",
    ) -> None:
        """All required artifact paths must exist on disk."""
        missing = [p for p in artifact_paths if not Path(p).exists()]
        if missing:
            self._fail(
                invariant="artifact_lineage_presence",
                message=f"missing artifacts: {missing}",
                run_id=run_id,
                detail={"missing": missing},
            )

    def load_reports(self) -> list[dict]:
        if not self._path.exists():
            return []
        results: list[dict] = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return results

    # ── internal ──────────────────────────────────────────────────────────────

    def _fail(
        self,
        invariant: str,
        message:   str,
        run_id:    str,
        detail:    Any = None,
    ) -> None:
        self._append_report(
            invariant=invariant, message=message, run_id=run_id, detail=detail
        )
        raise InvariantError(
            f"[INVARIANT] {invariant}: {message}",
            invariant=invariant,
            detail=detail,
        )

    def _append_report(
        self,
        invariant: str,
        message:   str,
        run_id:    str,
        detail:    Any,
    ) -> None:
        entry = json.dumps(
            {
                "ts":        time.time(),
                "invariant": invariant,
                "message":   message,
                "run_id":    run_id,
                "detail":    detail,
            },
            default=str,
        )
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")
