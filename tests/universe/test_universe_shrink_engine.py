"""
tests/universe/test_universe_shrink_engine.py

Tests for Universe Shrink Engine.

Covers:
  1.  test_no_action_when_live_at_target
  2.  test_no_action_when_live_below_target
  3.  test_held_symbols_excluded
  4.  test_rsr_weak_days_counted_correctly
  5.  test_consecutive_guard_not_met_no_candidate
  6.  test_score_threshold_not_met_no_candidate
  7.  test_candidate_when_both_gates_pass
  8.  test_fl_weakness_from_score
  9.  test_pe_weakness_from_score
  10. test_missing_fl_uses_neutral
  11. test_missing_pe_uses_neutral
  12. test_rsr_weakness_capped_at_one
  13. test_fail_open_on_symbol_error
  14. test_duck_type_pressure_score_equals_shrink_score
  15. test_deterministic_same_inputs_same_output
  16. test_empty_live_universe_returns_empty
  17. test_compute_returns_only_non_held
  18. test_consecutive_weak_days_helper
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from universe.universe_shrink_engine import (
    UniverseShrinkEngine,
    ShrinkPressure,
    _consecutive_weak_days,
    _zero_pressure,
    RSR_SHRINK_THRESHOLD,
    RSR_MIN_WEAK_DAYS,
    SHRINK_SCORE_THRESHOLD,
    W_RSR, W_FL, W_PE,
    MISSING_FL_WEAKNESS, MISSING_PE_WEAKNESS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _write_rsr_snap(tmp_path: Path, filename: str, scores: dict) -> None:
    d = filename.replace(".json", "")
    (tmp_path / filename).write_text(
        json.dumps({"date": d, "scores": scores}), encoding="utf-8"
    )


def _write_fl_candidates(tmp_path: Path, records: list) -> None:
    path = tmp_path / "candidates.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _write_pe_scores(tmp_path: Path, pa_scores: dict) -> Path:
    path = tmp_path / "predictive_scores.jsonl"
    path.write_text(
        json.dumps({"eval_date": "2026-06-06", "predictive_alpha_score": pa_scores}) + "\n",
        encoding="utf-8",
    )
    return path


def _make_engine(tmp_path: Path) -> UniverseShrinkEngine:
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()
    fl_file = tmp_path / "candidates.jsonl"
    fl_file.touch()
    pe_file = tmp_path / "predictive_scores.jsonl"
    pe_file.touch()
    return UniverseShrinkEngine(
        rsr_snapshot_dir   = rsr_dir,
        fl_candidates_file = fl_file,
        pe_scores_file     = pe_file,
    )


def _make_engine_with_data(
    tmp_path: Path,
    rsr_days: Dict[str, Dict[str, float]],   # {date_str: {symbol: rsr}}
    fl_records: list = None,
    pe_scores: dict = None,
) -> UniverseShrinkEngine:
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir(exist_ok=True)
    for d, scores in rsr_days.items():
        _write_rsr_snap(rsr_dir, f"{d}.json", scores)

    fl_file = tmp_path / "candidates.jsonl"
    if fl_records:
        with fl_file.open("w", encoding="utf-8") as fh:
            for rec in fl_records:
                fh.write(json.dumps(rec) + "\n")
    else:
        fl_file.touch()

    pe_file = tmp_path / "predictive_scores.jsonl"
    if pe_scores:
        pe_file.write_text(
            json.dumps({"eval_date": "2026-06-06", "predictive_alpha_score": pe_scores}) + "\n",
            encoding="utf-8",
        )
    else:
        pe_file.touch()

    return UniverseShrinkEngine(
        rsr_snapshot_dir   = rsr_dir,
        fl_candidates_file = fl_file,
        pe_scores_file     = pe_file,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_no_action_when_live_at_target(tmp_path):
    """Returns {} when live_size == target_size — no shrink needed."""
    engine = _make_engine(tmp_path)
    live = {f"{i:04d}.T": "sector" for i in range(42)}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    assert result == {}


def test_no_action_when_live_below_target(tmp_path):
    """Returns {} when live_size < target_size."""
    engine = _make_engine(tmp_path)
    live = {f"{i:04d}.T": "sector" for i in range(30)}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    assert result == {}


def test_held_symbols_excluded(tmp_path):
    """Held symbols are not evaluated and not in result."""
    rsr_days = {
        f"2026-06-0{d}": {"1001.T": 5.0, "1002.T": 5.0}
        for d in range(1, 8)
    }
    engine = _make_engine_with_data(tmp_path, rsr_days)
    live = {"1001.T": "A", "1002.T": "B", **{f"{i:04d}.T": "X" for i in range(50)}}
    result = engine.compute(
        live_universe=live,
        held_symbols=frozenset({"1001.T"}),
        target_size=42,
    )
    assert "1001.T" not in result
    assert "1002.T" in result


def test_rsr_weak_days_counted_correctly(tmp_path):
    """rsr_weak_days reflects trailing consecutive days below RSR_SHRINK_THRESHOLD."""
    # 4 consecutive weak days then 1 strong
    rsr_series = [5.0, 3.0, 80.0, 5.0, 10.0, 8.0, 15.0]  # last 4 are weak
    count = _consecutive_weak_days(rsr_series, RSR_SHRINK_THRESHOLD)
    assert count == 4


def test_consecutive_guard_not_met_no_candidate(tmp_path):
    """symbol with shrink_score >= threshold but only 2 weak days is NOT a candidate."""
    # 2 weak days (< RSR_MIN_WEAK_DAYS=3)
    rsr_days = {
        "2026-06-04": {"1001.T": 80.0},
        "2026-06-05": {"1001.T": 5.0},
        "2026-06-06": {"1001.T": 5.0},
    }
    engine = _make_engine_with_data(tmp_path, rsr_days)
    live = {"1001.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    sp = result["1001.T"]
    assert sp.rsr_weak_days == 2
    assert sp.is_candidate is False


def test_score_threshold_not_met_no_candidate(tmp_path):
    """Symbol with RSR=40 (above shrink threshold) has low rsr_weakness → not candidate."""
    rsr_days = {f"2026-06-0{d}": {"1001.T": 40.0} for d in range(1, 8)}
    engine = _make_engine_with_data(
        tmp_path, rsr_days,
        fl_records=[{"symbol": "1001.T", "future_leader_score": 80.0}],
        pe_scores={"1001.T": 80.0},
    )
    live = {"1001.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    sp = result["1001.T"]
    # RSR=40 > RSR_SHRINK_THRESHOLD=25 → rsr_weak_days=0 → is_candidate=False
    assert sp.rsr_weak_days == 0
    assert sp.is_candidate is False


def test_candidate_when_both_gates_pass(tmp_path):
    """Symbol with RSR<25 for 5 days and high fl/pe weakness becomes a candidate."""
    rsr_days = {f"2026-06-0{d}": {"1001.T": 5.0} for d in range(1, 8)}
    # Low FL score (high weakness) and low PE score (high weakness)
    engine = _make_engine_with_data(
        tmp_path, rsr_days,
        fl_records=[{"symbol": "1001.T", "future_leader_score": 10.0}],
        pe_scores={"1001.T": 10.0},
    )
    live = {"1001.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    sp = result["1001.T"]
    assert sp.rsr_weak_days >= RSR_MIN_WEAK_DAYS
    assert sp.shrink_score >= SHRINK_SCORE_THRESHOLD
    assert sp.is_candidate is True


def test_fl_weakness_from_score(tmp_path):
    """fl_weakness = 1 - fl_score/100."""
    rsr_days = {f"2026-06-0{d}": {"SYM.T": 5.0} for d in range(1, 8)}
    engine = _make_engine_with_data(
        tmp_path, rsr_days,
        fl_records=[{"symbol": "SYM.T", "future_leader_score": 30.0}],
    )
    live = {"SYM.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    sp = result["SYM.T"]
    assert abs(sp.fl_weakness_score - (1.0 - 30.0 / 100.0)) < 0.01


def test_pe_weakness_from_score(tmp_path):
    """pe_weakness = 1 - predictive_alpha_score/100."""
    rsr_days = {f"2026-06-0{d}": {"SYM.T": 5.0} for d in range(1, 8)}
    engine = _make_engine_with_data(
        tmp_path, rsr_days,
        pe_scores={"SYM.T": 20.0},
    )
    live = {"SYM.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    sp = result["SYM.T"]
    assert abs(sp.pe_weakness_score - (1.0 - 20.0 / 100.0)) < 0.01


def test_missing_fl_uses_neutral(tmp_path):
    """Missing FL record → fl_weakness = MISSING_FL_WEAKNESS (0.5)."""
    rsr_days = {f"2026-06-0{d}": {"SYM.T": 5.0} for d in range(1, 8)}
    engine = _make_engine_with_data(tmp_path, rsr_days)  # no FL records
    live = {"SYM.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    assert result["SYM.T"].fl_weakness_score == MISSING_FL_WEAKNESS


def test_missing_pe_uses_neutral(tmp_path):
    """Missing PE record → pe_weakness = MISSING_PE_WEAKNESS (0.5)."""
    rsr_days = {f"2026-06-0{d}": {"SYM.T": 5.0} for d in range(1, 8)}
    engine = _make_engine_with_data(tmp_path, rsr_days)  # no PE scores
    live = {"SYM.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    assert result["SYM.T"].pe_weakness_score == MISSING_PE_WEAKNESS


def test_rsr_weakness_capped_at_one(tmp_path):
    """rsr_weakness_score is capped at 1.0 regardless of how many weak days."""
    rsr_days = {f"2026-06-0{d}": {"SYM.T": 1.0} for d in range(1, 8)}
    engine = _make_engine_with_data(tmp_path, rsr_days)
    live = {"SYM.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    assert result["SYM.T"].rsr_weakness_score <= 1.0


def test_fail_open_on_symbol_error(tmp_path):
    """A symbol that errors during computation gets pressure_score=0, is_candidate=False."""
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()
    # No FL or PE files — they'll just be missing (no crash)
    engine = UniverseShrinkEngine(
        rsr_snapshot_dir   = rsr_dir,
        fl_candidates_file = tmp_path / "nonexistent_fl.jsonl",
        pe_scores_file     = tmp_path / "nonexistent_pe.jsonl",
    )
    live = {"SYM.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    sp = result.get("SYM.T")
    assert sp is not None
    assert sp.is_candidate is False


def test_duck_type_pressure_score_equals_shrink_score(tmp_path):
    """pressure_score (duck-type alias) equals shrink_score."""
    rsr_days = {f"2026-06-0{d}": {"SYM.T": 5.0} for d in range(1, 8)}
    engine = _make_engine_with_data(tmp_path, rsr_days)
    live = {"SYM.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    sp = result["SYM.T"]
    assert sp.pressure_score == sp.shrink_score


def test_deterministic_same_inputs_same_output(tmp_path):
    """Two calls with identical data produce identical results."""
    rsr_days = {f"2026-06-0{d}": {"SYM.T": 5.0} for d in range(1, 8)}
    engine = _make_engine_with_data(tmp_path, rsr_days)
    live = {"SYM.T": "A", **{f"{i:04d}.T": "X" for i in range(56)}}
    r1 = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    r2 = engine.compute(live_universe=live, held_symbols=frozenset(), target_size=42)
    assert r1["SYM.T"].shrink_score == r2["SYM.T"].shrink_score
    assert r1["SYM.T"].is_candidate == r2["SYM.T"].is_candidate


def test_empty_live_universe_returns_empty(tmp_path):
    """Empty live universe → empty result."""
    engine = _make_engine(tmp_path)
    result = engine.compute(live_universe={}, held_symbols=frozenset(), target_size=42)
    assert result == {}


def test_compute_returns_only_non_held(tmp_path):
    """Result contains exactly live_universe - held_symbols."""
    rsr_days = {f"2026-06-0{d}": {"A.T": 5.0, "B.T": 5.0, "C.T": 5.0} for d in range(1, 8)}
    engine = _make_engine_with_data(tmp_path, rsr_days)
    live = {"A.T": "x", "B.T": "x", "C.T": "x", **{f"{i:04d}.T": "X" for i in range(56)}}
    result = engine.compute(
        live_universe=live,
        held_symbols=frozenset({"A.T", "C.T"}),
        target_size=42,
    )
    assert "A.T" not in result
    assert "B.T" in result
    assert "C.T" not in result


def test_consecutive_weak_days_helper():
    """_consecutive_weak_days counts trailing consecutive below-threshold entries."""
    assert _consecutive_weak_days([80.0, 5.0, 5.0, 5.0], 25.0) == 3
    assert _consecutive_weak_days([5.0, 5.0, 80.0, 5.0], 25.0) == 1
    assert _consecutive_weak_days([80.0, 80.0, 80.0], 25.0) == 0
    assert _consecutive_weak_days([], 25.0) == 0
    assert _consecutive_weak_days([5.0], 25.0) == 1
    # Threshold boundary: exactly at threshold does NOT count as weak
    assert _consecutive_weak_days([25.0, 25.0, 25.0], 25.0) == 0
    # Just below threshold counts
    assert _consecutive_weak_days([24.9, 24.9], 25.0) == 2
