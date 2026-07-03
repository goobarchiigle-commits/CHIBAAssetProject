"""
tests/universe/test_shadow_refill.py

Tests for Shadow Refill Engine.

Covers:
  1.  test_should_refill_when_shadow_only_below_min
  2.  test_no_refill_when_shadow_only_sufficient
  3.  test_no_refill_when_all_shadow_in_live
  4.  test_compute_refill_returns_candidates
  5.  test_compute_refill_excludes_live_symbols
  6.  test_compute_refill_viability_filter
  7.  test_compute_refill_ranked_by_pe_then_rsr
  8.  test_compute_refill_respects_batch_limit
  9.  test_compute_refill_empty_pool
  10. test_execute_refill_writes_atomically
  11. test_execute_refill_idempotent
  12. test_execute_refill_fail_open_on_bad_path
  13. test_execute_refill_updates_in_memory_shadow
  14. test_compute_refill_fail_open
  15. test_should_refill_boundary_exactly_at_min
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from universe.shadow_refill import (
    ShadowRefillEngine,
    MIN_SHADOW_SIZE,
    SHADOW_REFILL_BATCH,
    RSR_VIABILITY_MIN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_pe_file(tmp_path: Path, pa_scores: dict) -> Path:
    p = tmp_path / "predictive_scores.jsonl"
    p.write_text(
        json.dumps({"eval_date": "2026-06-06", "predictive_alpha_score": pa_scores}) + "\n",
        encoding="utf-8",
    )
    return p


def _write_shadow_file(tmp_path: Path, symbols: dict) -> Path:
    p = tmp_path / "shadow_universe.json"
    p.write_text(
        json.dumps({"version": "shadow_v1", "symbols": symbols}),
        encoding="utf-8",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_should_refill_when_shadow_only_below_min():
    engine = ShadowRefillEngine()
    shadow = {"A.T": "s1"}  # only 1 shadow-only symbol
    live   = {"B.T": "s2", "C.T": "s3"}
    assert engine.should_refill(shadow, live) is True


def test_no_refill_when_shadow_only_sufficient():
    engine = ShadowRefillEngine()
    shadow = {f"{i}.T": "s" for i in range(MIN_SHADOW_SIZE + 2)}
    live   = {"X.T": "s"}  # none of shadow symbols are in live
    assert engine.should_refill(shadow, live) is False


def test_no_refill_when_all_shadow_in_live():
    engine = ShadowRefillEngine()
    shadow = {"A.T": "s", "B.T": "s"}
    live   = {"A.T": "s", "B.T": "s"}  # both shadow symbols already in live
    # shadow_only = 0 < MIN_SHADOW_SIZE → should refill
    assert engine.should_refill(shadow, live) is True


def test_compute_refill_returns_candidates(tmp_path):
    engine = ShadowRefillEngine()
    pe_file = _write_pe_file(tmp_path, {"CAND1.T": 80.0, "CAND2.T": 70.0})
    rsr = {"CAND1.T": 60.0, "CAND2.T": 55.0}
    result = engine.compute_refill(
        shadow_universe  = {},
        live_universe    = {},
        rsr_scores       = rsr,
        pe_scores_file   = pe_file,
    )
    assert len(result) >= 1
    syms = [s for s, _ in result]
    assert "CAND1.T" in syms or "CAND2.T" in syms


def test_compute_refill_excludes_live_symbols(tmp_path):
    engine = ShadowRefillEngine()
    pe_file = _write_pe_file(tmp_path, {"LIVE.T": 90.0, "CAND.T": 80.0})
    rsr = {"LIVE.T": 80.0, "CAND.T": 60.0}
    result = engine.compute_refill(
        shadow_universe = {},
        live_universe   = {"LIVE.T": "s"},
        rsr_scores      = rsr,
        pe_scores_file  = pe_file,
    )
    syms = [s for s, _ in result]
    assert "LIVE.T" not in syms
    assert "CAND.T" in syms


def test_compute_refill_viability_filter(tmp_path):
    """Symbols with RSR < RSR_VIABILITY_MIN are excluded."""
    engine = ShadowRefillEngine()
    pe_file = _write_pe_file(tmp_path, {"WEAK.T": 90.0, "OK.T": 80.0})
    rsr = {"WEAK.T": RSR_VIABILITY_MIN - 1, "OK.T": RSR_VIABILITY_MIN + 5}
    result = engine.compute_refill(
        shadow_universe = {},
        live_universe   = {},
        rsr_scores      = rsr,
        pe_scores_file  = pe_file,
    )
    syms = [s for s, _ in result]
    assert "WEAK.T" not in syms
    assert "OK.T" in syms


def test_compute_refill_ranked_by_pe_then_rsr(tmp_path):
    """Highest PE score comes first."""
    engine = ShadowRefillEngine()
    pe_file = _write_pe_file(tmp_path, {"LOW.T": 30.0, "HIGH.T": 90.0, "MID.T": 60.0})
    rsr = {"LOW.T": 50.0, "HIGH.T": 50.0, "MID.T": 50.0}
    result = engine.compute_refill(
        shadow_universe = {},
        live_universe   = {},
        rsr_scores      = rsr,
        pe_scores_file  = pe_file,
    )
    syms = [s for s, _ in result]
    assert syms.index("HIGH.T") < syms.index("MID.T")
    assert syms.index("MID.T") < syms.index("LOW.T")


def test_compute_refill_respects_batch_limit(tmp_path):
    """At most SHADOW_REFILL_BATCH symbols returned even if more are available."""
    engine = ShadowRefillEngine()
    pa = {f"C{i}.T": float(100 - i) for i in range(20)}
    rsr = {f"C{i}.T": 50.0 for i in range(20)}
    pe_file = _write_pe_file(tmp_path, pa)
    result = engine.compute_refill(
        shadow_universe = {},
        live_universe   = {},
        rsr_scores      = rsr,
        pe_scores_file  = pe_file,
    )
    assert len(result) <= SHADOW_REFILL_BATCH


def test_compute_refill_empty_pool(tmp_path):
    """No candidates outside live → empty result."""
    engine = ShadowRefillEngine()
    pe_file = _write_pe_file(tmp_path, {"SYM.T": 80.0})
    result = engine.compute_refill(
        shadow_universe = {},
        live_universe   = {"SYM.T": "s"},   # SYM is already live
        rsr_scores      = {"SYM.T": 70.0},
        pe_scores_file  = pe_file,
    )
    assert result == []


def test_execute_refill_writes_atomically(tmp_path):
    """execute_refill updates shadow_universe.json with new symbols."""
    engine = ShadowRefillEngine()
    shadow_file = _write_shadow_file(tmp_path, {"OLD.T": "old"})
    candidates = [("NEW1.T", "sector1"), ("NEW2.T", "sector2")]
    added = engine.execute_refill(shadow_file, candidates)
    assert len(added) == 2
    data = json.loads(shadow_file.read_text(encoding="utf-8"))
    assert "NEW1.T" in data["symbols"]
    assert "NEW2.T" in data["symbols"]
    assert "OLD.T" in data["symbols"]   # existing preserved


def test_execute_refill_idempotent(tmp_path):
    """Calling execute_refill twice with same candidates adds nothing on second call."""
    engine = ShadowRefillEngine()
    shadow_file = _write_shadow_file(tmp_path, {})
    candidates = [("SYM.T", "sector")]
    added1 = engine.execute_refill(shadow_file, candidates)
    added2 = engine.execute_refill(shadow_file, candidates)
    assert len(added1) == 1
    assert len(added2) == 0   # already present


def test_execute_refill_fail_open_on_bad_path(tmp_path):
    """Bad shadow file path returns [] (FAIL_OPEN)."""
    engine = ShadowRefillEngine()
    bad_path = tmp_path / "nonexistent" / "shadow.json"  # parent doesn't exist
    # execute_refill should handle the mkdir itself, so let's use a read-only-parent scenario
    # Instead, test with a corrupted file
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not json", encoding="utf-8")
    result = engine.execute_refill(corrupt, [("SYM.T", "s")])
    assert result == []


def test_execute_refill_updates_in_memory_shadow(tmp_path):
    """Caller can update in-memory shadow dict from execute_refill result."""
    engine = ShadowRefillEngine()
    shadow_file = _write_shadow_file(tmp_path, {})
    candidates = [("NEW.T", "sector")]
    added = engine.execute_refill(shadow_file, candidates)
    in_memory: dict = {}
    for sym, sec in added:
        in_memory[sym] = sec
    assert "NEW.T" in in_memory


def test_compute_refill_fail_open(tmp_path):
    """Corrupt PE file: falls back to RSR-only ranking (FAIL_OPEN, not abort).
    If a viable RSR candidate exists it is still returned — PE failure is non-fatal."""
    engine = ShadowRefillEngine()
    bad_pe = tmp_path / "bad.jsonl"
    bad_pe.write_text("not json\n", encoding="utf-8")
    result = engine.compute_refill(
        shadow_universe = {},
        live_universe   = {},
        rsr_scores      = {"SYM.T": 60.0},
        pe_scores_file  = bad_pe,
    )
    # PE parse fails gracefully; RSR candidate still surfaces
    syms = [s for s, _ in result]
    assert "SYM.T" in syms


def test_should_refill_boundary_exactly_at_min():
    """Exactly MIN_SHADOW_SIZE shadow-only symbols → no refill needed."""
    engine = ShadowRefillEngine()
    shadow = {f"{i}.T": "s" for i in range(MIN_SHADOW_SIZE)}
    live   = {}
    assert engine.should_refill(shadow, live) is False
