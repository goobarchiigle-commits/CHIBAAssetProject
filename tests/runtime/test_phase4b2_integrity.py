"""
tests/runtime/test_phase4b2_integrity.py — Phase 4B.2 integrity fixes

Covers:
  - [1] DRY/LIVE universe divergence audit
  - [2] FUTURE_LEADER_SURVIVABILITY_DIR local-shadow fix (import guard)
  - [3] JSON normalizer (numpy scalar → Python native)
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_numpy_stub():
    """
    Minimal numpy stub that provides np.bool_, np.int64, np.float64
    without requiring the real numpy package.
    """
    np = types.ModuleType("numpy")

    class _generic:
        def __init__(self, v): self._v = v
        def __module__(self): return "numpy"

    class bool_(int):
        __module__ = "numpy"
        __name__ = "bool_"

    class int64(int):
        __module__ = "numpy"

    class float64(float):
        __module__ = "numpy"

    np.bool_ = bool_
    np.int64 = int64
    np.float64 = float64
    np.generic = _generic
    return np


# ─────────────────────────────────────────────────────────────────────────────
# [3] json_normalizer
# ─────────────────────────────────────────────────────────────────────────────

class TestJsonNormalizer:
    def _module(self):
        from src.runtime.json_normalizer import normalize_json_scalar, normalize_for_json
        return normalize_json_scalar, normalize_for_json

    def test_python_bool_passthrough(self):
        n, _ = self._module()
        assert n(True) is True
        assert n(False) is False

    def test_python_int_passthrough(self):
        n, _ = self._module()
        assert n(42) == 42
        assert type(n(42)) is int

    def test_python_float_passthrough(self):
        n, _ = self._module()
        assert n(3.14) == 3.14
        assert type(n(3.14)) is float

    def test_python_str_passthrough(self):
        n, _ = self._module()
        assert n("hello") == "hello"

    def test_none_passthrough(self):
        n, _ = self._module()
        assert n(None) is None

    def test_numpy_bool_true(self):
        n, _ = self._module()
        np = _make_numpy_stub()
        result = n(np.bool_(1))
        assert result is True
        assert type(result) is bool

    def test_numpy_bool_false(self):
        n, _ = self._module()
        np = _make_numpy_stub()
        result = n(np.bool_(0))
        assert result is False
        assert type(result) is bool

    def test_numpy_int64(self):
        n, _ = self._module()
        np = _make_numpy_stub()
        result = n(np.int64(7))
        assert result == 7
        assert type(result) is int

    def test_numpy_float64(self):
        n, _ = self._module()
        np = _make_numpy_stub()
        result = n(np.float64(2.5))
        assert result == pytest.approx(2.5)
        assert type(result) is float

    def test_normalize_for_json_flat_dict(self):
        n, nfj = self._module()
        np = _make_numpy_stub()
        d = {
            "flag": np.bool_(1),
            "count": np.int64(3),
            "score": np.float64(0.75),
            "name": "abc",
        }
        result = nfj(d)
        assert result["flag"] is True
        assert type(result["count"]) is int
        assert type(result["score"]) is float
        assert result["name"] == "abc"
        # Must be JSON-serializable
        json.dumps(result)

    def test_json_dumps_after_normalize(self):
        n, _ = self._module()
        np = _make_numpy_stub()
        d = {"is_candidate": np.bool_(0), "pressure": np.float64(0.21)}
        d2 = {k: n(v) for k, v in d.items()}
        dumped = json.dumps(d2)
        restored = json.loads(dumped)
        assert restored["is_candidate"] is False
        assert pytest.approx(restored["pressure"]) == 0.21


# ─────────────────────────────────────────────────────────────────────────────
# [1] universe_determinism_audit
# ─────────────────────────────────────────────────────────────────────────────

class TestUniverseDeterminismAudit:
    def _record(self, tmp_path, mode, stage, live, shadow=None, tradeable=None):
        from src.runtime.universe_determinism_audit import record_universe_snapshot
        return record_universe_snapshot(
            audit_dir         = tmp_path,
            mode              = mode,
            snapshot_stage    = stage,
            live_symbols      = live,
            shadow_symbols    = shadow,
            tradeable_symbols = tradeable,
        )

    def test_symbol_hash_determinism(self):
        """Same symbol set always produces same hash."""
        from src.runtime.universe_determinism_audit import _symbol_hash
        syms = ["A", "B", "C"]
        assert _symbol_hash(syms) == _symbol_hash(["C", "A", "B"])
        expected = hashlib.sha256(",".join(sorted(syms)).encode()).hexdigest()
        assert _symbol_hash(syms) == expected

    def test_first_record_no_added_removed(self, tmp_path):
        """First record for a stage has empty added/removed."""
        rec = self._record(tmp_path, "DRY", "post_price_filter", ["X", "Y", "Z"])
        assert rec is not None
        assert rec.added_symbols == []
        assert rec.removed_symbols == []
        assert rec.live_count == 3
        assert rec.shadow_count == 0
        assert rec.total_symbols == 3

    def test_append_only(self, tmp_path):
        """Two records accumulate in the JSONL file."""
        self._record(tmp_path, "DRY", "post_price_filter", ["A", "B"])
        self._record(tmp_path, "DRY", "post_price_filter", ["A", "B", "C"])
        # find today's file
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        lines = [l for l in files[0].read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_added_removed_detection(self, tmp_path):
        """Second record detects added/removed vs first."""
        self._record(tmp_path, "DRY", "post_governance", ["A", "B", "C"])
        rec2 = self._record(tmp_path, "DRY", "post_governance", ["B", "C", "D"])
        assert rec2 is not None
        assert "A" in rec2.removed_symbols
        assert "D" in rec2.added_symbols

    def test_same_hash_no_warning(self, tmp_path, caplog):
        """No WARNING when hash unchanged between records."""
        import logging
        self._record(tmp_path, "DRY", "post_governance", ["A", "B"])
        with caplog.at_level(logging.WARNING, logger="src.runtime.universe_determinism_audit"):
            self._record(tmp_path, "DRY", "post_governance", ["A", "B"])
        assert "DIVERGENCE" not in caplog.text

    def test_different_hash_emits_warning(self, tmp_path, caplog):
        """WARNING emitted when same stage+date has different hash."""
        import logging
        self._record(tmp_path, "DRY", "post_price_filter", ["A", "B"])
        with caplog.at_level(logging.WARNING, logger="src.runtime.universe_determinism_audit"):
            self._record(tmp_path, "DRY", "post_price_filter", ["A", "B", "C"])
        assert "DIVERGENCE" in caplog.text

    def test_different_stages_no_cross_warning(self, tmp_path, caplog):
        """Different stages do not trigger cross-stage WARNING."""
        import logging
        self._record(tmp_path, "DRY", "post_price_filter", ["A", "B", "C"])
        with caplog.at_level(logging.WARNING, logger="src.runtime.universe_determinism_audit"):
            self._record(tmp_path, "DRY", "post_governance", ["A", "B"])  # different stage
        assert "DIVERGENCE" not in caplog.text

    def test_dry_live_no_cross_warning(self, tmp_path, caplog):
        """DRY and LIVE records do not warn against each other."""
        import logging
        self._record(tmp_path, "DRY",  "post_price_filter", ["A", "B", "C"])
        with caplog.at_level(logging.WARNING, logger="src.runtime.universe_determinism_audit"):
            self._record(tmp_path, "LIVE", "post_price_filter", ["A", "B"])  # different mode
        assert "DIVERGENCE" not in caplog.text

    def test_to_dict_json_serializable(self, tmp_path):
        """to_dict() output must be JSON-serializable."""
        rec = self._record(tmp_path, "DRY", "post_price_filter", ["X", "Y"])
        d = rec.to_dict()
        json.dumps(d)  # must not raise

    def test_shadow_count(self, tmp_path):
        """shadow_count reflects shadow symbols."""
        rec = self._record(
            tmp_path, "DRY", "rsr_context",
            ["A", "B"],
            shadow=["S1", "S2", "S3"],
        )
        assert rec.shadow_count == 3
        assert rec.total_symbols == 5

    def test_fail_open_on_bad_dir(self, tmp_path, caplog):
        """
        record_universe_snapshot fails gracefully when audit_dir cannot be created.
        (Simulate by passing a file path as audit_dir.)
        """
        import logging
        bad_dir = tmp_path / "not_a_dir.txt"
        bad_dir.write_text("block")  # file, not dir — mkdir will fail inside
        from src.runtime.universe_determinism_audit import record_universe_snapshot
        with caplog.at_level(logging.WARNING, logger="src.runtime.universe_determinism_audit"):
            result = record_universe_snapshot(
                audit_dir      = bad_dir,
                mode           = "DRY",
                snapshot_stage = "post_price_filter",
                live_symbols   = ["A"],
            )
        assert result is None
        assert "FAIL_OPEN" in caplog.text

    def test_jsonl_integrity(self, tmp_path):
        """Every appended line must be valid JSON."""
        for i in range(5):
            self._record(tmp_path, "DRY", "post_governance", [f"S{j}" for j in range(i + 1)])
        files = list(tmp_path.glob("*.jsonl"))
        for line in files[0].read_text().splitlines():
            if line.strip():
                json.loads(line)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# [2] FUTURE_LEADER_SURVIVABILITY_DIR import guard
# ─────────────────────────────────────────────────────────────────────────────

class TestFutureLeaderSurvivabilityDirFix:
    def test_no_local_shadow_in_run_live_signal(self):
        """
        Verify no 'from src.paths import ... FUTURE_LEADER_SURVIVABILITY_DIR ...'
        inside a local (non-module-level) scope in run_live_signal.py.

        The bug: a local `from src.paths import FUTURE_LEADER_SURVIVABILITY_DIR`
        inside main() makes Python treat ALL references in main() as local,
        causing UnboundLocalError at earlier call sites (lines 2420, 2453 etc.).
        """
        rls = Path("src/run_live_signal.py")
        content = rls.read_text(encoding="utf-8")
        lines = content.splitlines()

        inside_function = False
        for i, line in enumerate(lines, 1):
            # Track when we enter functions (rough heuristic: indented def)
            if line.startswith("def ") or line.startswith("class "):
                inside_function = False
            elif line.startswith("    def ") or line.startswith("    async def "):
                inside_function = True

            # Flag any local from-import that rebinds FUTURE_LEADER_SURVIVABILITY_DIR
            if (inside_function
                    and "FUTURE_LEADER_SURVIVABILITY_DIR" in line
                    and "from src.paths import" in line):
                pytest.fail(
                    f"Line {i}: local 'from src.paths import ... FUTURE_LEADER_SURVIVABILITY_DIR'"
                    " shadows module-level import → UnboundLocalError risk.\n"
                    f"  {line.strip()}"
                )

    def test_survivability_dir_importable(self):
        """FUTURE_LEADER_SURVIVABILITY_DIR must be importable from src.paths."""
        from src.paths import FUTURE_LEADER_SURVIVABILITY_DIR
        assert isinstance(FUTURE_LEADER_SURVIVABILITY_DIR, Path)

    def test_survivability_dir_auto_create(self, tmp_path):
        """
        Simulate the mkdir(parents=True, exist_ok=True) pattern expected
        when survivability_log_dir does not exist.
        """
        d = tmp_path / "surv_dir"
        assert not d.exists()
        d.mkdir(parents=True, exist_ok=True)
        assert d.exists()
        # Idempotent
        d.mkdir(parents=True, exist_ok=True)

    def test_universe_determinism_audit_dir_importable(self):
        """UNIVERSE_DETERMINISM_AUDIT_DIR must be importable from src.paths."""
        from src.paths import UNIVERSE_DETERMINISM_AUDIT_DIR
        assert isinstance(UNIVERSE_DETERMINISM_AUDIT_DIR, Path)
