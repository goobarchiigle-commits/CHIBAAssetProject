"""
tests/test_governance_hardening.py
Comprehensive tests for the governance hardening layer.

Covers:
  - RegistryGuard: require_active, forbidden files, status queries
  - RuntimeMode: mode enum, enforcement, restrictions
  - StateGuard: live/research separation, get_state_path, contamination prevention
  - ConfigSnapshot: create, load, no-overwrite, integrity check
  - ValidateActiveImports: AST forbidden detection, registry check
  - ValidateOOS: hash check, staleness, missing artifact, null hash
  - GovernanceGate: end-to-end check orchestration
  - Research isolation: research/ path import prevention
"""
from __future__ import annotations

import json
import os
import sys
import hashlib
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

JST = timezone(timedelta(hours=9))


# ─── helpers ─────────────────────────────────────────────────────────────────

def _write(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_registry(tmp: Path, entries: dict | None = None) -> Path:
    if entries is None:
        entries = {
            "_meta": {"schema_version": "1.0"},
            "test_active": {
                "status": "active",
                "source_file": "src/backtest/composite_alpha_bt.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "live_entrypoints": [],
            },
        }
    path = tmp / "state" / "strategy_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _make_oos_registry(tmp: Path, artifacts: dict | None = None) -> Path:
    if artifacts is None:
        cfg  = tmp / "src" / "configs" / "strategy.yaml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("portfolio:\n  capital: 3000000\n", encoding="utf-8")
        art  = tmp / "backtests" / "oos_result.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text('{"oos_sharpe": 1.612}', encoding="utf-8")
        art_hash = hashlib.sha256(art.read_bytes()).hexdigest()
        cfg_hash = hashlib.sha256(cfg.read_bytes()).hexdigest()
        artifacts = {
            "test_oos_artifact": {
                "strategy_id": "test_active",
                "artifact_path": "backtests/oos_result.json",
                "artifact_hash": art_hash,
                "created_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "config_hash": cfg_hash,
                "validation_status": "valid",
                "max_age_days": 365,
            }
        }
    path = tmp / "state" / "oos_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"_meta": {}, "artifacts": artifacts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ═════════════════════════════════════════════════════════════════════════════
# TestRegistryGuard
# ═════════════════════════════════════════════════════════════════════════════

class TestRegistryGuard:

    def _make_guard(self, tmp: Path, entries: dict | None = None):
        from src.governance.registry_guard import RegistryGuard
        reg = _make_registry(tmp, entries)
        return RegistryGuard(registry_path=reg)

    def test_require_active_passes_for_active_strategy(self, tmp_path):
        guard = self._make_guard(tmp_path)
        entry = guard.require_active("test_active")
        assert entry["status"] == "active"

    def test_require_active_raises_for_unregistered(self, tmp_path):
        from src.governance.registry_guard import StrategyNotRegisteredError
        guard = self._make_guard(tmp_path)
        with pytest.raises(StrategyNotRegisteredError, match="not registered"):
            guard.require_active("nonexistent_strategy")

    def test_require_active_raises_for_experimental(self, tmp_path):
        from src.governance.registry_guard import StrategyStatusError
        guard = self._make_guard(tmp_path, {
            "_meta": {},
            "exp_strat": {
                "status": "experimental",
                "source_file": "src/backtest/composite_alpha_bt.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": "2026-01-01",
                "live_entrypoints": [],
            },
        })
        with pytest.raises(StrategyStatusError, match="status='experimental'"):
            guard.require_active("exp_strat")

    def test_require_active_raises_for_deprecated(self, tmp_path):
        from src.governance.registry_guard import StrategyStatusError
        guard = self._make_guard(tmp_path, {
            "_meta": {},
            "old_strat": {
                "status": "deprecated",
                "source_file": "src/backtest/composite_alpha_bt.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": "2026-01-01",
                "live_entrypoints": [],
            },
        })
        with pytest.raises(StrategyStatusError):
            guard.require_active("old_strat")

    def test_is_registered_true_for_known(self, tmp_path):
        guard = self._make_guard(tmp_path)
        assert guard.is_registered("test_active") is True

    def test_is_registered_false_for_unknown(self, tmp_path):
        guard = self._make_guard(tmp_path)
        assert guard.is_registered("totally_unknown") is False

    def test_get_active_strategies_returns_only_active(self, tmp_path):
        guard = self._make_guard(tmp_path, {
            "_meta": {},
            "active_a": {"status": "active", "source_file": "a.py", "active_config": "c.yaml", "last_validated_at": "2026-01-01", "live_entrypoints": []},
            "exp_b":    {"status": "experimental", "source_file": "b.py", "active_config": "c.yaml", "last_validated_at": "2026-01-01", "live_entrypoints": []},
        })
        result = guard.get_active_strategies()
        assert "active_a" in result
        assert "exp_b" not in result

    def test_get_active_source_files(self, tmp_path):
        guard = self._make_guard(tmp_path)
        files = guard.get_active_source_files()
        assert "src/backtest/composite_alpha_bt.py" in files

    def test_get_forbidden_source_files_returns_deprecated(self, tmp_path):
        guard = self._make_guard(tmp_path, {
            "_meta": {},
            "dep_strat": {
                "status": "deprecated",
                "source_file": "src/backtest/archive/old.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": "2026-01-01",
                "live_entrypoints": [],
            },
        })
        forbidden = guard.get_forbidden_source_files()
        assert "src/backtest/archive/old.py" in forbidden

    def test_require_registered_returns_any_status(self, tmp_path):
        guard = self._make_guard(tmp_path, {
            "_meta": {},
            "archived_strat": {
                "status": "archived",
                "source_file": "src/backtest/archive/old.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": "2026-01-01",
                "live_entrypoints": [],
            },
        })
        entry = guard.require_registered("archived_strat")
        assert entry["status"] == "archived"

    def test_strategy_count_by_status(self, tmp_path):
        guard = self._make_guard(tmp_path, {
            "_meta": {},
            "a": {"status": "active",       "source_file": "a.py", "active_config": "c.yaml", "last_validated_at": "2026-01-01", "live_entrypoints": []},
            "b": {"status": "experimental", "source_file": "b.py", "active_config": "c.yaml", "last_validated_at": "2026-01-01", "live_entrypoints": []},
            "c": {"status": "archived",     "source_file": "c.py", "active_config": "c.yaml", "last_validated_at": "2026-01-01", "live_entrypoints": []},
        })
        counts = guard.strategy_count_by_status()
        assert counts["active"] == 1
        assert counts["experimental"] == 1
        assert counts["archived"] == 1

    def test_missing_registry_file_raises(self, tmp_path):
        from src.governance.registry_guard import RegistryGuard
        guard = RegistryGuard(registry_path=tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            guard.require_active("anything")


# ═════════════════════════════════════════════════════════════════════════════
# TestRuntimeMode
# ═════════════════════════════════════════════════════════════════════════════

class TestRuntimeMode:

    def test_default_mode_is_research(self, monkeypatch):
        from src.governance.mode import get_current_mode, RuntimeMode
        monkeypatch.delenv("RUNTIME_MODE", raising=False)
        assert get_current_mode() == RuntimeMode.RESEARCH

    def test_production_mode_from_env(self, monkeypatch):
        from src.governance.mode import get_current_mode, RuntimeMode
        monkeypatch.setenv("RUNTIME_MODE", "production")
        assert get_current_mode() == RuntimeMode.PRODUCTION

    def test_paper_mode_from_env(self, monkeypatch):
        from src.governance.mode import get_current_mode, RuntimeMode
        monkeypatch.setenv("RUNTIME_MODE", "paper")
        assert get_current_mode() == RuntimeMode.PAPER

    def test_invalid_mode_raises(self, monkeypatch):
        from src.governance.mode import get_current_mode
        monkeypatch.setenv("RUNTIME_MODE", "invalid_xyz")
        with pytest.raises(ValueError, match="Invalid RUNTIME_MODE"):
            get_current_mode()

    def test_assert_production_mode_passes(self, monkeypatch):
        from src.governance.mode import assert_production_mode
        monkeypatch.setenv("RUNTIME_MODE", "production")
        assert_production_mode()  # should not raise

    def test_assert_production_mode_fails_in_research(self, monkeypatch):
        from src.governance.mode import assert_production_mode, ModeViolationError
        monkeypatch.setenv("RUNTIME_MODE", "research")
        with pytest.raises(ModeViolationError):
            assert_production_mode()

    def test_research_cannot_write_live_state(self):
        from src.governance.mode import (
            RuntimeMode, assert_not_research_writing_live_state, ModeViolationError
        )
        with pytest.raises(ModeViolationError):
            assert_not_research_writing_live_state(
                RuntimeMode.RESEARCH,
                Path("/project/state/live/capital_snapshot.json"),
            )

    def test_paper_cannot_write_live_state_for_capital(self):
        from src.governance.mode import (
            RuntimeMode, assert_not_paper_mutating_capital, ModeViolationError
        )
        with pytest.raises(ModeViolationError):
            assert_not_paper_mutating_capital(
                RuntimeMode.PAPER,
                Path("/project/state/live/capital_snapshot.json"),
            )

    def test_production_can_write_live_state(self):
        from src.governance.mode import (
            RuntimeMode, assert_not_research_writing_live_state
        )
        # Should not raise
        assert_not_research_writing_live_state(
            RuntimeMode.PRODUCTION,
            Path("/project/state/live/capital_snapshot.json"),
        )

    def test_assert_can_load_strategy_active_in_production(self, monkeypatch):
        from src.governance.mode import RuntimeMode, assert_can_load_strategy
        assert_can_load_strategy("active", RuntimeMode.PRODUCTION)  # should not raise

    def test_assert_can_load_strategy_experimental_blocked_in_production(self, monkeypatch):
        from src.governance.mode import RuntimeMode, assert_can_load_strategy, ModeViolationError
        with pytest.raises(ModeViolationError):
            assert_can_load_strategy("experimental", RuntimeMode.PRODUCTION)

    def test_assert_can_load_strategy_experimental_ok_in_paper(self):
        from src.governance.mode import RuntimeMode, assert_can_load_strategy
        assert_can_load_strategy("experimental", RuntimeMode.PAPER)  # should not raise

    def test_assert_can_load_strategy_any_ok_in_research(self):
        from src.governance.mode import RuntimeMode, assert_can_load_strategy
        for status in ("active", "experimental", "deprecated", "archived"):
            assert_can_load_strategy(status, RuntimeMode.RESEARCH)  # all OK in research


# ═════════════════════════════════════════════════════════════════════════════
# TestStateGuard
# ═════════════════════════════════════════════════════════════════════════════

class TestStateGuard:

    def test_production_gets_live_dir(self, tmp_path):
        from src.governance.mode import RuntimeMode
        from src.governance.state_guard import get_state_path
        p = get_state_path("capital_snapshot.json", RuntimeMode.PRODUCTION)
        assert "live" in str(p)

    def test_research_gets_research_dir(self, tmp_path):
        from src.governance.mode import RuntimeMode
        from src.governance.state_guard import get_state_path
        p = get_state_path("research_capital.json", RuntimeMode.RESEARCH)
        assert "research" in str(p)

    def test_research_cannot_access_live_only_file(self):
        from src.governance.mode import RuntimeMode, ModeViolationError
        from src.governance.state_guard import get_state_path
        with pytest.raises(ModeViolationError, match="live-only"):
            get_state_path("capital_snapshot.json", RuntimeMode.RESEARCH)

    def test_paper_cannot_access_live_only_file(self):
        from src.governance.mode import RuntimeMode, ModeViolationError
        from src.governance.state_guard import get_state_path
        with pytest.raises(ModeViolationError, match="live-only"):
            get_state_path("live_fills.jsonl", RuntimeMode.PAPER)

    def test_production_cannot_access_research_only_file(self):
        from src.governance.mode import RuntimeMode, ModeViolationError
        from src.governance.state_guard import get_state_path
        with pytest.raises(ModeViolationError, match="research-only"):
            get_state_path("research_capital.json", RuntimeMode.PRODUCTION)

    def test_assert_no_shared_capital_same_path_raises(self, tmp_path):
        from src.governance.state_guard import assert_no_shared_capital
        from src.governance.mode import ModeViolationError
        live_dir = tmp_path / "live"
        live_dir.mkdir()
        research_dir = live_dir  # same dir → same path
        with pytest.raises(ModeViolationError, match="same"):
            assert_no_shared_capital(live_dir, research_dir)

    def test_assert_no_shared_capital_different_paths_ok(self, tmp_path):
        from src.governance.state_guard import assert_no_shared_capital
        live_dir     = tmp_path / "live"
        research_dir = tmp_path / "research"
        live_dir.mkdir()
        research_dir.mkdir()
        assert_no_shared_capital(live_dir, research_dir)  # should not raise

    def test_no_live_files_in_research_dir_clean(self, tmp_path):
        from src.governance.state_guard import assert_no_live_files_in_research_dir
        live     = tmp_path / "live"
        research = tmp_path / "research"
        live.mkdir()
        research.mkdir()
        violations = assert_no_live_files_in_research_dir(live, research)
        assert violations == []

    def test_live_only_file_in_research_dir_detected(self, tmp_path):
        from src.governance.state_guard import (
            assert_no_live_files_in_research_dir, LIVE_ONLY_FILES
        )
        live     = tmp_path / "live"
        research = tmp_path / "research"
        live.mkdir()
        research.mkdir()
        live_name = next(iter(LIVE_ONLY_FILES))
        (research / live_name).write_text("bad", encoding="utf-8")
        violations = assert_no_live_files_in_research_dir(live, research)
        assert live_name in violations

    def test_research_only_file_in_live_dir_detected(self, tmp_path):
        from src.governance.state_guard import (
            assert_no_research_files_in_live_dir, RESEARCH_ONLY_FILES
        )
        live     = tmp_path / "live"
        research = tmp_path / "research"
        live.mkdir()
        research.mkdir()
        res_name = next(iter(RESEARCH_ONLY_FILES))
        (live / res_name).write_text("bad", encoding="utf-8")
        violations = assert_no_research_files_in_live_dir(live, research)
        assert res_name in violations


# ═════════════════════════════════════════════════════════════════════════════
# TestConfigSnapshot
# ═════════════════════════════════════════════════════════════════════════════

class TestConfigSnapshot:

    def _make_config(self, tmp: Path, content: str = "capital: 3000000") -> Path:
        cfg = tmp / "strategy.yaml"
        cfg.write_text(content, encoding="utf-8")
        return cfg

    def test_create_snapshot_returns_id(self, tmp_path):
        from src.governance.config_snapshot import create_snapshot
        cfg = self._make_config(tmp_path)
        sid = create_snapshot(cfg, snapshot_dir=tmp_path / "snapshots")
        assert isinstance(sid, str) and len(sid) > 0

    def test_snapshot_file_is_created(self, tmp_path):
        from src.governance.config_snapshot import create_snapshot
        cfg  = self._make_config(tmp_path)
        snap = tmp_path / "snapshots"
        sid  = create_snapshot(cfg, snapshot_dir=snap)
        yaml_files = list(snap.glob(f"{sid}.yaml"))
        assert len(yaml_files) == 1

    def test_manifest_is_created(self, tmp_path):
        from src.governance.config_snapshot import create_snapshot
        cfg  = self._make_config(tmp_path)
        snap = tmp_path / "snapshots"
        sid  = create_snapshot(cfg, snapshot_dir=snap)
        assert (snap / f"{sid}.json").exists()

    def test_snapshot_content_preserved(self, tmp_path):
        from src.governance.config_snapshot import create_snapshot, load_snapshot
        content = "portfolio:\n  capital: 3_000_000\n"
        cfg     = self._make_config(tmp_path, content)
        snap    = tmp_path / "snapshots"
        sid     = create_snapshot(cfg, snapshot_dir=snap)
        path, manifest = load_snapshot(sid, snapshot_dir=snap)
        assert path.read_text(encoding="utf-8") == content

    def test_no_overwrite_idempotent(self, tmp_path):
        from src.governance.config_snapshot import create_snapshot
        cfg  = self._make_config(tmp_path)
        snap = tmp_path / "snapshots"
        sid1 = create_snapshot(cfg, snapshot_dir=snap)
        # Calling with same content at a slightly different time gives a different ID (timestamp)
        # But same content at exactly same second gives same ID → idempotent
        # We test that multiple snapshots are never overwritten
        files_before = set(snap.glob("*.yaml"))
        # Create a second snapshot (different content)
        cfg.write_text("capital: 999999\n", encoding="utf-8")
        sid2 = create_snapshot(cfg, snapshot_dir=snap)
        files_after = set(snap.glob("*.yaml"))
        assert len(files_after) > len(files_before), "New snapshot should be added"
        assert sid1 != sid2 or (snap / f"{sid1}.yaml").exists(), "Old snapshot preserved"

    def test_research_mode_blocked(self, tmp_path):
        from src.governance.config_snapshot import create_snapshot
        from src.governance.mode import RuntimeMode, ModeViolationError
        cfg = self._make_config(tmp_path)
        with pytest.raises(ModeViolationError, match="Research mode"):
            create_snapshot(cfg, snapshot_dir=tmp_path / "snap", mode=RuntimeMode.RESEARCH)

    def test_missing_config_raises(self, tmp_path):
        from src.governance.config_snapshot import create_snapshot
        with pytest.raises(FileNotFoundError):
            create_snapshot(tmp_path / "nonexistent.yaml", snapshot_dir=tmp_path / "snap")

    def test_load_snapshot_integrity_check(self, tmp_path):
        from src.governance.config_snapshot import create_snapshot, load_snapshot
        cfg  = self._make_config(tmp_path, "original content")
        snap = tmp_path / "snapshots"
        sid  = create_snapshot(cfg, snapshot_dir=snap)
        # Tamper with the snapshot file
        snap_file = next(snap.glob(f"{sid}.yaml"))
        snap_file.write_text("TAMPERED CONTENT", encoding="utf-8")
        with pytest.raises(RuntimeError, match="integrity failure"):
            load_snapshot(sid, snapshot_dir=snap)

    def test_list_snapshots_empty(self, tmp_path):
        from src.governance.config_snapshot import list_snapshots
        assert list_snapshots(tmp_path / "empty_snap") == []

    def test_list_snapshots_returns_manifests(self, tmp_path):
        from src.governance.config_snapshot import create_snapshot, list_snapshots
        cfg  = self._make_config(tmp_path)
        snap = tmp_path / "snapshots"
        create_snapshot(cfg, snapshot_dir=snap)
        manifests = list_snapshots(snap)
        assert len(manifests) >= 1
        assert "snapshot_id" in manifests[0]


# ═════════════════════════════════════════════════════════════════════════════
# TestValidateActiveImports
# ═════════════════════════════════════════════════════════════════════════════

class TestValidateActiveImports:

    def _make_py(self, tmp: Path, name: str, content: str) -> Path:
        return _write(tmp / name, content)

    def test_clean_file_passes(self, tmp_path, monkeypatch):
        import validate_active_imports as vai
        monkeypatch.setattr(vai, "REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        _make_registry(tmp_path)

        f = self._make_py(tmp_path, "clean.py", "import os\nimport json\n")
        v, w = vai.validate_file(f, {}, strict=False)
        assert v == []

    def test_archive_import_is_violation(self, tmp_path, monkeypatch):
        import validate_active_imports as vai
        monkeypatch.setattr(vai, "REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")

        f = self._make_py(
            tmp_path, "bad.py",
            "from src.backtest.archive import clenow_momentum\n"
        )
        v, w = vai.validate_file(f, {}, strict=False)
        assert any("forbidden" in msg.lower() or "FORBIDDEN" in msg for msg in v)

    def test_research_import_is_violation(self, tmp_path, monkeypatch):
        import validate_active_imports as vai
        monkeypatch.setattr(vai, "REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")

        f = self._make_py(
            tmp_path, "bad.py",
            "from research.experimental import alpha_v99\n"
        )
        v, w = vai.validate_file(f, {}, strict=False)
        assert any("forbidden" in msg.lower() or "FORBIDDEN" in msg for msg in v)

    def test_infra_import_not_flagged(self, tmp_path, monkeypatch):
        import validate_active_imports as vai
        f = self._make_py(
            tmp_path, "infra.py",
            "from src.execution.journal import IntentJournal\n"
            "from src.common.position_normalizer import normalize\n"
        )
        v, w = vai.validate_file(f, {}, strict=False)
        assert v == []
        assert w == []

    def test_deprecated_strategy_import_is_violation(self, tmp_path, monkeypatch):
        import validate_active_imports as vai

        registry = {
            "old_strat": {
                "status": "deprecated",
                "source_file": "src/backtest/portfolio_full.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": "2026-01-01",
            }
        }
        f = self._make_py(
            tmp_path, "prod.py",
            "from src.backtest.portfolio_full import run\n"
        )
        v, w = vai.validate_file(f, registry, strict=False)
        assert any("deprecated" in msg.lower() or "DEPRECATED" in msg or "archived" in msg.lower() for msg in v)

    def test_module_to_rel_path(self):
        import validate_active_imports as vai
        assert vai.module_to_rel_path("src.backtest.archive.clenow") == \
               "src/backtest/archive/clenow.py"

    def test_classify_forbidden(self):
        import validate_active_imports as vai
        assert vai.classify_import("src/backtest/archive/old.py") == "forbidden"

    def test_classify_strategy(self):
        import validate_active_imports as vai
        assert vai.classify_import("src/backtest/composite_alpha_bt.py") == "strategy"

    def test_classify_infra(self):
        import validate_active_imports as vai
        assert vai.classify_import("src/execution/journal.py") == "infra"

    def test_extract_imports_from_file(self, tmp_path):
        import validate_active_imports as vai
        f = _write(tmp_path / "ex.py",
                   "import os\nfrom src.backtest.archive import old\nfrom src.execution.journal import J\n")
        mods = vai.extract_imports(f)
        assert "os" in mods
        assert "src.backtest.archive" in mods
        assert "src.execution.journal" in mods

    def test_syntax_error_returns_empty(self, tmp_path):
        import validate_active_imports as vai
        f = _write(tmp_path / "bad.py", "def broken(:\n    pass\n")
        mods = vai.extract_imports(f)
        assert mods == []

    def test_run_validation_no_targets_passes(self, tmp_path, monkeypatch):
        import validate_active_imports as vai
        monkeypatch.setattr(vai, "REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        _make_registry(tmp_path)  # no live_entrypoints
        v, w = vai.run_validation(targets=[])
        assert v == []


# ═════════════════════════════════════════════════════════════════════════════
# TestValidateOOS
# ═════════════════════════════════════════════════════════════════════════════

class TestValidateOOS:

    def _setup(self, tmp: Path, artifacts: dict | None = None):
        import validate_oos as vo
        reg_path = _make_registry(tmp)
        oos_path = _make_oos_registry(tmp, artifacts)
        return vo, tmp, oos_path

    def test_valid_artifact_passes(self, tmp_path, monkeypatch):
        import validate_oos as vo
        monkeypatch.setattr(vo, "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo, "REPO_ROOT",              tmp_path)
        _make_registry(tmp_path)
        _make_oos_registry(tmp_path)

        errors, warnings = vo.run_validation()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_artifact_file_is_error(self, tmp_path, monkeypatch):
        import validate_oos as vo
        monkeypatch.setattr(vo, "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo, "REPO_ROOT",              tmp_path)
        _make_registry(tmp_path)
        _make_oos_registry(tmp_path, {
            "bad_art": {
                "strategy_id": "test_active",
                "artifact_path": "backtests/nonexistent.json",
                "artifact_hash": "abc123",
                "created_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "validation_status": "valid",
                "max_age_days": 365,
            }
        })
        errors, _ = vo.run_validation()
        assert any("not found" in e for e in errors)

    def test_hash_mismatch_is_error(self, tmp_path, monkeypatch):
        import validate_oos as vo
        monkeypatch.setattr(vo, "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo, "REPO_ROOT",              tmp_path)
        _make_registry(tmp_path)

        art = tmp_path / "backtests" / "oos.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text('{"result": 1}', encoding="utf-8")

        _make_oos_registry(tmp_path, {
            "hash_mismatch_art": {
                "strategy_id": "test_active",
                "artifact_path": "backtests/oos.json",
                "artifact_hash": "0000000000000000000000000000000000000000000000000000000000000000",
                "created_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "validation_status": "valid",
                "max_age_days": 365,
            }
        })
        errors, _ = vo.run_validation()
        assert any("MISMATCH" in e or "mismatch" in e.lower() for e in errors)

    def test_null_hash_is_error(self, tmp_path, monkeypatch):
        import validate_oos as vo
        monkeypatch.setattr(vo, "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo, "REPO_ROOT",              tmp_path)
        _make_registry(tmp_path)

        art = tmp_path / "backtests" / "oos.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text('{"result": 1}', encoding="utf-8")

        _make_oos_registry(tmp_path, {
            "null_hash_art": {
                "strategy_id": "test_active",
                "artifact_path": "backtests/oos.json",
                "artifact_hash": None,
                "created_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "validation_status": "valid",
                "max_age_days": 365,
            }
        })
        errors, _ = vo.run_validation()
        assert any("null" in e.lower() or "unverified" in e.lower() or "init" in e.lower() for e in errors)

    def test_stale_oos_is_error(self, tmp_path, monkeypatch):
        import validate_oos as vo
        monkeypatch.setattr(vo, "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo, "REPO_ROOT",              tmp_path)
        _make_registry(tmp_path)

        art = tmp_path / "backtests" / "old.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text('{"old": true}', encoding="utf-8")
        art_hash = hashlib.sha256(art.read_bytes()).hexdigest()

        _make_oos_registry(tmp_path, {
            "stale_art": {
                "strategy_id": "test_active",
                "artifact_path": "backtests/old.json",
                "artifact_hash": art_hash,
                "created_at": "2020-01-01",   # very stale
                "validation_status": "valid",
                "max_age_days": 365,
            }
        })
        errors, _ = vo.run_validation()
        assert any("stale" in e.lower() or "STALE" in e for e in errors)

    def test_invalid_status_is_error(self, tmp_path, monkeypatch):
        import validate_oos as vo
        monkeypatch.setattr(vo, "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo, "REPO_ROOT",              tmp_path)
        _make_registry(tmp_path)

        art = tmp_path / "backtests" / "pend.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text('{}', encoding="utf-8")
        art_hash = hashlib.sha256(art.read_bytes()).hexdigest()

        _make_oos_registry(tmp_path, {
            "pending_art": {
                "strategy_id": "test_active",
                "artifact_path": "backtests/pend.json",
                "artifact_hash": art_hash,
                "created_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "validation_status": "pending",
                "max_age_days": 365,
            }
        })
        errors, _ = vo.run_validation()
        assert any("pending" in e or "valid" in e for e in errors)

    def test_unknown_strategy_id_is_error(self, tmp_path, monkeypatch):
        import validate_oos as vo
        monkeypatch.setattr(vo, "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo, "REPO_ROOT",              tmp_path)
        _make_registry(tmp_path)

        art = tmp_path / "backtests" / "ok.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text('{}', encoding="utf-8")
        art_hash = hashlib.sha256(art.read_bytes()).hexdigest()

        _make_oos_registry(tmp_path, {
            "unknown_strat_art": {
                "strategy_id": "nonexistent_strategy_xyz",
                "artifact_path": "backtests/ok.json",
                "artifact_hash": art_hash,
                "created_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "validation_status": "valid",
                "max_age_days": 365,
            }
        })
        errors, _ = vo.run_validation()
        assert any("not found" in e or "Unknown" in e or "lineage" in e for e in errors)

    def test_init_computes_null_hashes(self, tmp_path, monkeypatch):
        import validate_oos as vo
        monkeypatch.setattr(vo, "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo, "REPO_ROOT",              tmp_path)

        art = tmp_path / "backtests" / "uninit.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text('{"x": 1}', encoding="utf-8")

        oos_path = tmp_path / "state" / "oos_registry.json"
        oos_path.parent.mkdir(parents=True, exist_ok=True)
        oos_path.write_text(json.dumps({
            "_meta": {},
            "artifacts": {
                "uninit_art": {
                    "strategy_id": "test",
                    "artifact_path": "backtests/uninit.json",
                    "artifact_hash": None,
                    "created_at": datetime.now(JST).strftime("%Y-%m-%d"),
                    "validation_status": "valid",
                    "max_age_days": 365,
                }
            }
        }), encoding="utf-8")

        vo.init_hashes()

        updated = json.loads(oos_path.read_text(encoding="utf-8"))
        assert updated["artifacts"]["uninit_art"]["artifact_hash"] is not None

    def test_oos_registry_file_not_found_raises(self, tmp_path, monkeypatch):
        import validate_oos as vo
        monkeypatch.setattr(vo, "OOS_REGISTRY_FILE", tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            vo.run_validation()


# ═════════════════════════════════════════════════════════════════════════════
# TestGovernanceGate
# ═════════════════════════════════════════════════════════════════════════════

class TestGovernanceGate:

    def test_gate_passes_with_valid_state(self, tmp_path, monkeypatch):
        import governance_gate as gg
        import consistency_check as cc
        import validate_active_imports as vai
        import validate_oos as vo

        monkeypatch.setattr(cc,  "STRATEGY_REGISTRY_FILE", _make_registry(tmp_path))
        monkeypatch.setattr(cc,  "REPO_INVENTORY_FILE",    tmp_path / "state" / "repo_inventory.json")
        monkeypatch.setattr(cc,  "PORTFOLIO_STATE_FILE",   tmp_path / "runtime" / "portfolio_state.json")
        monkeypatch.setattr(cc,  "REPO_ROOT",              REPO_ROOT)
        monkeypatch.setattr(vai, "REGISTRY_FILE",          tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo,  "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo,  "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo,  "REPO_ROOT",              tmp_path)

        _make_oos_registry(tmp_path)

        errors, warnings = gg.run_all_checks(skip_import_check=True)
        assert not any(e for e in errors if "[CONSISTENCY]" in e or "[OOS]" in e)

    def test_gate_fails_on_oos_error(self, tmp_path, monkeypatch):
        import governance_gate as gg
        import consistency_check as cc
        import validate_active_imports as vai
        import validate_oos as vo

        monkeypatch.setattr(cc,  "STRATEGY_REGISTRY_FILE", _make_registry(tmp_path))
        monkeypatch.setattr(cc,  "REPO_INVENTORY_FILE",    tmp_path / "state" / "repo_inventory.json")
        monkeypatch.setattr(cc,  "PORTFOLIO_STATE_FILE",   tmp_path / "runtime" / "portfolio_state.json")
        monkeypatch.setattr(cc,  "REPO_ROOT",              REPO_ROOT)
        monkeypatch.setattr(vai, "REGISTRY_FILE",          tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo,  "OOS_REGISTRY_FILE",      tmp_path / "state" / "oos_registry.json")
        monkeypatch.setattr(vo,  "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(vo,  "REPO_ROOT",              tmp_path)

        # OOS with bad hash
        art = tmp_path / "backtests" / "r.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text("{}", encoding="utf-8")
        _make_oos_registry(tmp_path, {
            "bad_oos": {
                "strategy_id": "test_active",
                "artifact_path": "backtests/r.json",
                "artifact_hash": "0" * 64,
                "created_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "validation_status": "valid",
                "max_age_days": 365,
            }
        })

        errors, _ = gg.run_all_checks(skip_import_check=True)
        assert any("[OOS]" in e for e in errors)

    def test_generate_status_doc_contains_gate_result(self, tmp_path, monkeypatch):
        import governance_gate as gg
        monkeypatch.setattr(gg, "STATE_DIR", tmp_path / "state")
        monkeypatch.setattr(gg, "REPO_ROOT", tmp_path)
        _make_registry(tmp_path)
        _make_oos_registry(tmp_path)

        doc = gg.generate_status_doc([], [])
        assert "PASS" in doc or "FAIL" in doc
        assert "Active strategies" in doc

    def test_generate_status_doc_with_errors(self, tmp_path, monkeypatch):
        import governance_gate as gg
        monkeypatch.setattr(gg, "STATE_DIR", tmp_path / "state")
        monkeypatch.setattr(gg, "REPO_ROOT", tmp_path)
        _make_registry(tmp_path)
        _make_oos_registry(tmp_path)

        doc = gg.generate_status_doc(["[OOS] hash mismatch"], [])
        assert "FAIL" in doc
        assert "hash mismatch" in doc


# ═════════════════════════════════════════════════════════════════════════════
# TestResearchIsolation
# ═════════════════════════════════════════════════════════════════════════════

class TestResearchIsolation:
    """Verify that research/ path is treated as forbidden for production imports."""

    def test_research_path_classified_as_forbidden(self):
        import validate_active_imports as vai
        assert vai.classify_import("research/experimental_model.py") == "forbidden"

    def test_research_import_detected_in_validation(self, tmp_path):
        import validate_active_imports as vai
        f = _write(tmp_path / "prod.py", "from research.experimental import alpha\n")
        v, w = vai.validate_file(f, {}, strict=False)
        assert len(v) > 0

    def test_research_dir_exists_in_repo(self):
        research_dir = REPO_ROOT / "research"
        assert research_dir.exists(), "research/ directory must exist"

    def test_live_dir_exists_in_state(self):
        assert (REPO_ROOT / "state" / "live").exists()

    def test_research_dir_exists_in_state(self):
        assert (REPO_ROOT / "state" / "research").exists()

    def test_config_snapshots_dir_exists(self):
        assert (REPO_ROOT / "config" / "snapshots").exists()


# ═════════════════════════════════════════════════════════════════════════════
# TestRealOOSRegistry
# ═════════════════════════════════════════════════════════════════════════════

class TestRealOOSRegistry:
    """Tests against the actual state/oos_registry.json in the repo."""

    def test_oos_registry_file_exists(self):
        assert (REPO_ROOT / "state" / "oos_registry.json").exists()

    def test_oos_registry_valid_json(self):
        data = json.loads(
            (REPO_ROOT / "state" / "oos_registry.json").read_text(encoding="utf-8")
        )
        assert "artifacts" in data

    def test_all_oos_artifacts_have_hashes(self):
        data      = json.loads(
            (REPO_ROOT / "state" / "oos_registry.json").read_text(encoding="utf-8")
        )
        artifacts = data.get("artifacts", {})
        for name, art in artifacts.items():
            assert art.get("artifact_hash") is not None, \
                f"[{name}] artifact_hash is null — run validate_oos.py --init"

    def test_all_oos_artifact_files_exist(self):
        data      = json.loads(
            (REPO_ROOT / "state" / "oos_registry.json").read_text(encoding="utf-8")
        )
        artifacts = data.get("artifacts", {})
        for name, art in artifacts.items():
            path = REPO_ROOT / art.get("artifact_path", "")
            assert path.exists(), f"[{name}] artifact_path not found: {art.get('artifact_path')}"

    def test_all_oos_artifact_hashes_match(self):
        data      = json.loads(
            (REPO_ROOT / "state" / "oos_registry.json").read_text(encoding="utf-8")
        )
        artifacts = data.get("artifacts", {})
        for name, art in artifacts.items():
            stored_hash = art.get("artifact_hash")
            if stored_hash is None:
                continue
            path        = REPO_ROOT / art.get("artifact_path", "")
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual_hash == stored_hash, \
                f"[{name}] hash mismatch: stored={stored_hash[:16]}... actual={actual_hash[:16]}..."
