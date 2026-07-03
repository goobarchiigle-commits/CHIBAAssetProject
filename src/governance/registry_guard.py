"""
src/governance/registry_guard.py
Strategy registry enforcement — fail closed.

Any strategy execution must be registered and active.
Unregistered or non-active strategies raise immediately.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
REGISTRY_FILE  = REPO_ROOT / "state" / "strategy_registry.json"

VALID_STATUSES    = frozenset({"active", "experimental", "deprecated", "archived"})
EXECUTABLE_STATUS = "active"


class StrategyNotRegisteredError(RuntimeError):
    """Strategy not found in state/strategy_registry.json."""


class StrategyStatusError(RuntimeError):
    """Strategy exists but status does not permit execution."""


class RegistryGuard:
    """
    Enforce that only registered, active strategies can execute.

    Usage:
        guard = RegistryGuard()
        entry = guard.require_active("fujiko_composite")   # raises if not active
        files = guard.get_active_source_files()
        bad   = guard.get_forbidden_source_files()
    """

    def __init__(self, registry_path: Path = REGISTRY_FILE) -> None:
        self._path      = registry_path
        self._cache: dict | None = None

    def _load(self) -> dict:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Strategy registry not found: {self._path}\n"
                "Create state/strategy_registry.json before executing strategies."
            )
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"strategy_registry.json is invalid JSON: {exc}") from exc
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def get_registry(self) -> dict:
        if self._cache is None:
            self._cache = self._load()
        return self._cache

    def invalidate_cache(self) -> None:
        self._cache = None

    # ── Enforcement ───────────────────────────────────────────────────────────

    def require_active(self, strategy_id: str) -> dict:
        """
        Return registry entry for strategy_id.
        Raises StrategyNotRegisteredError if not in registry.
        Raises StrategyStatusError if status != 'active'.
        """
        registry = self.get_registry()
        if strategy_id not in registry:
            raise StrategyNotRegisteredError(
                f"Strategy '{strategy_id}' is not registered. "
                f"Add it to state/strategy_registry.json with status='active' before executing."
            )
        entry  = registry[strategy_id]
        status = entry.get("status", "")
        if status != EXECUTABLE_STATUS:
            raise StrategyStatusError(
                f"Strategy '{strategy_id}' has status='{status}'. "
                f"Only status='active' strategies may execute in production. "
                f"Current status prevents execution."
            )
        logger.info(
            "[REGISTRY_GUARD] strategy=%s status=active oos_sharpe=%s last_validated=%s — ALLOWED",
            strategy_id,
            entry.get("oos_sharpe", "?"),
            entry.get("last_validated_at", "?"),
        )
        return entry

    def require_registered(self, strategy_id: str) -> dict:
        """Return entry for any registered strategy (any status). Raises if not found."""
        registry = self.get_registry()
        if strategy_id not in registry:
            raise StrategyNotRegisteredError(
                f"Strategy '{strategy_id}' is not registered in state/strategy_registry.json."
            )
        return registry[strategy_id]

    def assert_source_file_registered(self, source_file: str) -> None:
        """
        Raise StrategyNotRegisteredError if source_file is not the source of any registered strategy.
        Used by import validators.
        """
        registered = self.get_all_source_files()
        norm = source_file.replace("\\", "/")
        if norm not in registered:
            raise StrategyNotRegisteredError(
                f"Source file '{source_file}' is not registered in strategy_registry.json. "
                "Unregistered strategy files cannot be executed."
            )

    # ── Queries ───────────────────────────────────────────────────────────────

    def is_registered(self, strategy_id: str) -> bool:
        try:
            return strategy_id in self.get_registry()
        except Exception:
            return False

    def get_active_strategies(self) -> dict:
        return {k: v for k, v in self.get_registry().items()
                if v.get("status") == "active"}

    def get_active_source_files(self) -> set[str]:
        return {
            v["source_file"].replace("\\", "/")
            for v in self.get_active_strategies().values()
            if v.get("source_file")
        }

    def get_all_source_files(self) -> set[str]:
        return {
            v["source_file"].replace("\\", "/")
            for v in self.get_registry().values()
            if v.get("source_file")
        }

    def get_forbidden_source_files(self) -> set[str]:
        """Files from deprecated/archived strategies — must not be imported in production."""
        return {
            v["source_file"].replace("\\", "/")
            for v in self.get_registry().values()
            if v.get("status") in ("deprecated", "archived") and v.get("source_file")
        }

    def strategy_count_by_status(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(
            v.get("status", "unknown")
            for v in self.get_registry().values()
        ))


# Module-level singleton (pre-loaded against the real registry)
DEFAULT_GUARD = RegistryGuard()
