"""
src/runtime/determinism/freeze_guard.py

RuntimeFreezeGuard: validation-window manifest mutation detector.

Frozen sources (any mutation → fail-closed):
- schema_manifests
- governance_manifests
- feature_manifests
- orchestration_manifests

On violation:
- raises FreezeViolationError (fail-closed)
- appends immutable entry to artifacts/runtime_validation/freeze_violations.jsonl

No adaptive behavior. No retry. No silent pass.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict

FREEZE_VIOLATIONS_PATH: Path = Path("artifacts/runtime_validation/freeze_violations.jsonl")

FREEZE_SOURCES: frozenset[str] = frozenset({
    "schema_manifests",
    "governance_manifests",
    "feature_manifests",
    "orchestration_manifests",
})


class FreezeViolationError(RuntimeError):
    def __init__(self, message: str, key: str, baseline: Any, current: Any) -> None:
        super().__init__(message)
        self.key      = key
        self.baseline = baseline
        self.current  = current


class RuntimeFreezeGuard:
    """
    Validate manifests have not mutated since the freeze baseline was captured.

    Usage::
        guard = RuntimeFreezeGuard(baseline_manifests)
        guard.check(current_manifests)   # raises FreezeViolationError on mutation
    """

    def __init__(
        self,
        baseline:        Dict[str, Any],
        violations_path: Path = FREEZE_VIOLATIONS_PATH,
    ) -> None:
        self._baseline = copy.deepcopy(baseline)
        self._path     = Path(violations_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock     = threading.Lock()

    def check(self, current: Dict[str, Any]) -> None:
        """
        Compare current manifests against baseline.
        Raises FreezeViolationError on first detected mutation.
        Violation is persisted before raising (fail-closed: record then abort).
        """
        for key in sorted(FREEZE_SOURCES):
            baseline_val = self._baseline.get(key)
            current_val  = current.get(key)
            if baseline_val != current_val:
                self._record_violation(key=key, baseline=baseline_val, current=current_val)
                raise FreezeViolationError(
                    f"[FREEZE_GUARD] mutation detected key={key}",
                    key=key,
                    baseline=baseline_val,
                    current=current_val,
                )

    def load_violations(self) -> list[dict]:
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

    def _record_violation(self, key: str, baseline: Any, current: Any) -> None:
        entry = json.dumps(
            {
                "ts":       time.time(),
                "event":    "freeze_violation",
                "key":      key,
                "baseline": baseline,
                "current":  current,
            },
            default=str,
        )
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")
