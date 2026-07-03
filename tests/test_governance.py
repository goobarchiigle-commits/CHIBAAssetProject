"""
tests/test_governance.py
Pytest suite for the repository governance layer.

Covers:
  - archive_policy: timestamp suffix, no overwrite, lineage log
  - repo_audit: hash correctness, ambiguous name detection, duplicate detection
  - consistency_check: valid registry passes, invalid fails with errors
  - strategy_registry: schema, valid statuses, required fields
"""
from __future__ import annotations

import json
import sys
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

# Make tools/ importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import archive_policy as ap
import repo_audit     as ra
import consistency_check as cc

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write(path: Path, content: str = "test content") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_registry(tmp: Path, entries: dict | None = None) -> Path:
    if entries is None:
        entries = {
            "_meta": {"schema_version": "1.0"},
            "test_strategy": {
                "status": "active",
                "source_file": "src/backtest/composite_alpha_bt.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "live_entrypoints": [],
                "owner": "test",
            },
        }
    reg_path = tmp / "state" / "strategy_registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return reg_path


# ─────────────────────────────────────────────────────────────────────────────
# TestArchivePolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestArchivePolicy:

    def test_archive_moves_file_with_timestamp(self, tmp_path):
        src = _write(tmp_path / "strategy.py")
        dest = ap.archive_file(src, archive_dir=tmp_path / "archive")
        assert not src.exists(), "Source should be gone after archiving"
        assert dest.exists(), "Destination should exist"

    def test_timestamp_suffix_in_dest_name(self, tmp_path):
        src = _write(tmp_path / "report.json")
        dest = ap.archive_file(src, archive_dir=tmp_path / "archive")
        today = datetime.now(JST).strftime("%Y-%m-%d")
        assert today in dest.name, "Timestamp should be in destination filename"

    def test_no_overwrite_on_collision(self, tmp_path):
        today = datetime.now(JST).strftime("%Y-%m-%d")
        archive = tmp_path / "archive"
        archive.mkdir()
        collision = archive / f"report_{today}.json"
        collision.write_text("existing", encoding="utf-8")

        src = _write(tmp_path / "report.json")
        dest = ap.archive_file(src, archive_dir=archive)
        assert dest != collision, "Should use unique name to avoid overwriting"
        assert dest.exists()
        assert collision.exists(), "Original collision file must be preserved"

    def test_source_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ap.archive_file(tmp_path / "nonexistent.py")

    def test_dry_run_does_not_move(self, tmp_path):
        src = _write(tmp_path / "report.json")
        dest = ap.archive_file(src, archive_dir=tmp_path / "archive", dry_run=True)
        assert src.exists(), "dry_run should not move file"
        assert not dest.exists(), "dry_run should not create destination"

    def test_lineage_log_appended(self, tmp_path, monkeypatch):
        log_path = tmp_path / "state" / "archive_log.jsonl"
        monkeypatch.setattr(ap, "ARCHIVE_LOG", log_path)
        monkeypatch.setattr(ap, "STATE_DIR", tmp_path / "state")

        src = _write(tmp_path / "old.py")
        ap.archive_file(src, archive_dir=tmp_path / "archive", reason="superseded")

        assert log_path.exists()
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["reason"] == "superseded"
        assert "old.py" in record["src"]

    def test_multiple_archives_produce_unique_names(self, tmp_path):
        archive = tmp_path / "archive"
        for i in range(3):
            src = _write(tmp_path / f"file{i}.json", f"content {i}")
            ap.archive_file(src, archive_dir=archive)
        archived = list(archive.glob("*.json"))
        assert len(archived) == 3

    def test_archive_preserves_content(self, tmp_path):
        content = "important trading logic"
        src = _write(tmp_path / "strategy.py", content)
        dest = ap.archive_file(src, archive_dir=tmp_path / "archive")
        assert dest.read_text(encoding="utf-8") == content


# ─────────────────────────────────────────────────────────────────────────────
# TestRepoAuditHashing
# ─────────────────────────────────────────────────────────────────────────────

class TestRepoAuditHashing:

    def test_sha256_is_deterministic(self, tmp_path):
        f = _write(tmp_path / "a.py", "hello world")
        h1 = ra.sha256_file(f)
        h2 = ra.sha256_file(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = _write(tmp_path / "a.py", "content A")
        f2 = _write(tmp_path / "b.py", "content B")
        assert ra.sha256_file(f1) != ra.sha256_file(f2)

    def test_same_content_same_hash(self, tmp_path):
        f1 = _write(tmp_path / "a.py", "identical")
        f2 = _write(tmp_path / "b.py", "identical")
        assert ra.sha256_file(f1) == ra.sha256_file(f2)

    def test_hash_is_64_hex_chars(self, tmp_path):
        f = _write(tmp_path / "a.py", "x")
        h = ra.sha256_file(f)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ─────────────────────────────────────────────────────────────────────────────
# TestAmbiguousNameDetection
# ─────────────────────────────────────────────────────────────────────────────

class TestAmbiguousNameDetection:

    @pytest.mark.parametrize("name,expected_pat", [
        ("strategy_final.py",    "final"),
        ("report_final2.json",   "final2"),
        ("model_backup.pkl",     "backup"),
        ("data_temp.csv",        "temp"),
        ("old_strategy.py",      "old"),
        ("fix_bug.py",           "fix"),
        ("report.bak",           "bak"),
        ("strategy_copy.py",     "copy"),
        ("orig_strategy.py",      "orig"),
        ("tmp_output.json",      "tmp"),
    ])
    def test_ambiguous_names_detected(self, name, expected_pat):
        is_amb, pat = ra._is_ambiguous(name)
        assert is_amb, f"{name!r} should be detected as ambiguous"
        assert pat == expected_pat

    @pytest.mark.parametrize("name", [
        "composite_alpha_bt.py",
        "strategy.yaml",
        "run_live_signal.py",
        "test_deferred_queue.py",
        "portfolio_engine.py",
        "fujiko_composite.py",
        "wf_dyn_rsr42.py",
    ])
    def test_clean_names_not_ambiguous(self, name):
        is_amb, _ = ra._is_ambiguous(name)
        assert not is_amb, f"{name!r} should NOT be ambiguous"


# ─────────────────────────────────────────────────────────────────────────────
# TestDuplicateDetection
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateDetection:

    def test_duplicate_filename_detected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(ra, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")

        _write(tmp_path / "a" / "report.json", "content a")
        _write(tmp_path / "b" / "report.json", "content b")

        inv = ra.scan(repo_root=tmp_path)
        dup_name = [i for i in inv["issues"] if i["type"] == "duplicate_filename"]
        assert any(i["name"] == "report.json" for i in dup_name)

    def test_duplicate_content_detected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(ra, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")

        _write(tmp_path / "a.py", "identical content")
        _write(tmp_path / "b.py", "identical content")

        inv = ra.scan(repo_root=tmp_path)
        dup_content = [i for i in inv["issues"] if i["type"] == "duplicate_content"]
        assert len(dup_content) >= 1

    def test_no_false_positive_for_unique_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(ra, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")

        _write(tmp_path / "a.py", "content A")
        _write(tmp_path / "b.py", "content B")

        inv = ra.scan(repo_root=tmp_path)
        dup_content = [i for i in inv["issues"] if i["type"] == "duplicate_content"]
        assert len(dup_content) == 0

    def test_skip_dirs_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(ra, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")

        _write(tmp_path / "__pycache__" / "file.pyc", "x")
        _write(tmp_path / ".git" / "config", "y")
        _write(tmp_path / "real.py", "z")

        inv = ra.scan(repo_root=tmp_path)
        paths = [f["path"] for f in inv["files"]]
        assert not any("__pycache__" in p for p in paths)
        assert not any(".git" in p for p in paths)
        assert any("real.py" in p for p in paths)

    def test_ambiguous_names_in_scan(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(ra, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")

        _write(tmp_path / "strategy_final.py", "final version")

        inv = ra.scan(repo_root=tmp_path)
        amb = [i for i in inv["issues"] if i["type"] == "ambiguous_name"]
        assert len(amb) >= 1

    def test_oversized_file_detected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(ra, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")
        monkeypatch.setattr(ra, "SIZE_WARN_MB", 0.001)  # 1KB threshold

        _write(tmp_path / "big.json", "x" * 2000)  # 2KB — exceeds threshold

        inv = ra.scan(repo_root=tmp_path)
        oversized = [i for i in inv["issues"] if i["type"] == "oversized_file"]
        assert any("big.json" in i["path"] for i in oversized)

    def test_summary_counts_match_issues(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(ra, "STRATEGY_REGISTRY_FILE", tmp_path / "state" / "strategy_registry.json")

        _write(tmp_path / "a" / "x.json", "same")
        _write(tmp_path / "b" / "x.json", "same")  # dup name + dup content

        inv = ra.scan(repo_root=tmp_path)
        s   = inv["_meta"]["summary"]
        assert s["total_issues"] == len(inv["issues"])
        assert s["duplicate_filename_count"] >= 1
        assert s["duplicate_content_count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TestConsistencyCheck
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyCheck:

    def _patch_paths(self, monkeypatch, tmp_path, registry_entries=None):
        reg_path = _make_registry(tmp_path, registry_entries)
        inv_path = tmp_path / "state" / "repo_inventory.json"
        monkeypatch.setattr(cc, "STRATEGY_REGISTRY_FILE", reg_path)
        monkeypatch.setattr(cc, "REPO_INVENTORY_FILE",    inv_path)
        monkeypatch.setattr(cc, "PORTFOLIO_STATE_FILE",   tmp_path / "runtime" / "portfolio_state.json")
        monkeypatch.setattr(cc, "REPO_ROOT",              REPO_ROOT)
        return reg_path

    def test_valid_registry_passes(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path)
        errors, warnings = cc.run_checks()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_registry_is_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cc, "STRATEGY_REGISTRY_FILE", tmp_path / "nonexistent.json")
        monkeypatch.setattr(cc, "REPO_INVENTORY_FILE",    tmp_path / "inv.json")
        monkeypatch.setattr(cc, "PORTFOLIO_STATE_FILE",   tmp_path / "port.json")
        errors, _ = cc.run_checks()
        assert any("not found" in e for e in errors)

    def test_invalid_json_is_error(self, tmp_path, monkeypatch):
        bad = tmp_path / "state" / "strategy_registry.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{invalid json}", encoding="utf-8")
        monkeypatch.setattr(cc, "STRATEGY_REGISTRY_FILE", bad)
        monkeypatch.setattr(cc, "REPO_INVENTORY_FILE",    tmp_path / "inv.json")
        monkeypatch.setattr(cc, "PORTFOLIO_STATE_FILE",   tmp_path / "port.json")
        errors, _ = cc.run_checks()
        assert any("JSON" in e for e in errors)

    def test_invalid_status_is_error(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path, {
            "_meta": {},
            "bad_strat": {
                "status": "unknown_status",
                "source_file": "src/backtest/composite_alpha_bt.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": "2026-04-01",
                "live_entrypoints": [],
            },
        })
        errors, _ = cc.run_checks()
        assert any("invalid status" in e for e in errors)

    def test_missing_source_file_is_error(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path, {
            "_meta": {},
            "strat": {
                "status": "active",
                "source_file": "src/backtest/nonexistent_strategy.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": "2026-04-01",
                "live_entrypoints": [],
            },
        })
        errors, _ = cc.run_checks()
        assert any("source_file not found" in e for e in errors)

    def test_missing_config_is_error(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path, {
            "_meta": {},
            "strat": {
                "status": "active",
                "source_file": "src/backtest/composite_alpha_bt.py",
                "active_config": "src/configs/nonexistent_config.yaml",
                "last_validated_at": "2026-04-01",
                "live_entrypoints": [],
            },
        })
        errors, _ = cc.run_checks()
        assert any("active_config not found" in e for e in errors)

    def test_stale_last_validated_active_is_error(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path, {
            "_meta": {},
            "strat": {
                "status": "active",
                "source_file": "src/backtest/composite_alpha_bt.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": "2020-01-01",   # very stale
                "live_entrypoints": [],
            },
        })
        errors, _ = cc.run_checks()
        assert any("last_validated_at" in e for e in errors)

    def test_stale_last_validated_experimental_is_warning(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path, {
            "_meta": {},
            "exp_strat": {
                "status": "experimental",
                "source_file": "src/backtest/composite_alpha_bt.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": "2020-01-01",   # very stale
                "live_entrypoints": [],
            },
        })
        errors, warnings = cc.run_checks()
        # Stale experimental → warning not error
        assert not any("last_validated_at" in e for e in errors)
        assert any("last_validated_at" in w for w in warnings)

    def test_missing_required_field_is_error(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path, {
            "_meta": {},
            "strat": {
                "status": "active",
                # missing source_file and active_config
                "last_validated_at": "2026-04-01",
            },
        })
        errors, _ = cc.run_checks()
        assert any("missing required field" in e for e in errors)

    def test_meta_key_not_treated_as_strategy(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path, {
            "_meta": {"schema_version": "1.0"},
            "real_strat": {
                "status": "active",
                "source_file": "src/backtest/composite_alpha_bt.py",
                "active_config": "src/configs/strategy.yaml",
                "last_validated_at": datetime.now(JST).strftime("%Y-%m-%d"),
                "live_entrypoints": [],
            },
        })
        errors, _ = cc.run_checks()
        assert errors == []

    def test_archived_strategy_skips_file_checks(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path, {
            "_meta": {},
            "old_strat": {
                "status": "archived",
                "source_file": "src/backtest/nonexistent.py",
                "active_config": "src/configs/nonexistent.yaml",
                "last_validated_at": "2020-01-01",
            },
        })
        errors, _ = cc.run_checks()
        # archived entries are skipped for file existence checks
        assert not any("source_file not found" in e for e in errors)
        assert not any("active_config not found" in e for e in errors)

    def test_inventory_missing_produces_warning(self, tmp_path, monkeypatch):
        self._patch_paths(monkeypatch, tmp_path)
        errors, warnings = cc.run_checks()
        assert not any("repo_inventory" in e for e in errors)
        assert any("repo_inventory" in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# TestStrategyRegistrySchema
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyRegistrySchema:

    def _load_registry(self) -> dict:
        reg_path = REPO_ROOT / "state" / "strategy_registry.json"
        assert reg_path.exists(), "strategy_registry.json must exist"
        return json.loads(reg_path.read_text(encoding="utf-8"))

    def test_registry_file_exists(self):
        assert (REPO_ROOT / "state" / "strategy_registry.json").exists()

    def test_registry_is_valid_json(self):
        raw = (REPO_ROOT / "state" / "strategy_registry.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_meta_key_present(self):
        data = self._load_registry()
        assert "_meta" in data

    def test_schema_version_present(self):
        data = self._load_registry()
        assert "schema_version" in data["_meta"]

    def test_all_statuses_valid(self):
        data = self._load_registry()
        valid = {"active", "experimental", "deprecated", "archived"}
        for name, entry in data.items():
            if name.startswith("_"):
                continue
            status = entry.get("status", "")
            assert status in valid, f"[{name}] invalid status={status!r}"

    def test_active_strategies_have_required_fields(self):
        data  = self._load_registry()
        required = {"status", "source_file", "active_config", "last_validated_at"}
        for name, entry in data.items():
            if name.startswith("_") or entry.get("status") != "active":
                continue
            for field in required:
                assert field in entry, f"[{name}] missing field: {field!r}"

    def test_fujiko_composite_is_active(self):
        data = self._load_registry()
        assert "fujiko_composite" in data, "fujiko_composite must be in registry"
        assert data["fujiko_composite"]["status"] == "active"

    def test_fujiko_oos_sharpe_correct(self):
        data = self._load_registry()
        sharpe = data["fujiko_composite"].get("oos_sharpe")
        assert sharpe == 1.612, f"OOS Sharpe should be 1.612, got {sharpe}"

    def test_active_strategy_source_file_exists(self):
        data = self._load_registry()
        for name, entry in data.items():
            if name.startswith("_") or entry.get("status") != "active":
                continue
            src = entry.get("source_file", "")
            if src:
                assert (REPO_ROOT / src).exists(), f"[{name}] source_file not found: {src}"

    def test_active_strategy_config_exists(self):
        data = self._load_registry()
        for name, entry in data.items():
            if name.startswith("_") or entry.get("status") != "active":
                continue
            cfg = entry.get("active_config", "")
            if cfg:
                assert (REPO_ROOT / cfg).exists(), f"[{name}] active_config not found: {cfg}"
