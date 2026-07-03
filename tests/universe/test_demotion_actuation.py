"""
tests/universe/test_demotion_actuation.py

Tests for Phase 3A: Demotion Actuation Engine.

Covers:
  1.  test_demotion_candidate_creation
  2.  test_recovery_to_active
  3.  test_no_progress_without_replacement_candidate
  4.  test_pending_exit_when_position_exists
  5.  test_exit_confirmed_after_position_closed
  6.  test_removed_only_after_exit_confirmed
  7.  test_fail_open_on_storage_failure
  8.  test_no_sell_order_generated
  9.  test_state_recovery_after_restart
  10. test_duplicate_symbol_idempotent
  11. test_universe_size_not_reduced_without_replacement
  12. test_removed_is_terminal_state
  13. test_candidate_to_exit_confirmed_no_position
  14. test_metrics_counts_correct
  15. test_transitions_written_to_jsonl
  16. test_load_state_replays_correctly
  17. test_recovery_threshold_hysteresis
  18. test_multiple_symbols_independent
  19. test_write_state_append_only
  20. test_average_days_pending_exit
"""
import json
import sys
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from universe.demotion_actuation import (
    DemotionActuationEngine,
    DemotionActuationResult,
    DemotionRecord,
    STATE_ACTIVE,
    STATE_CANDIDATE,
    STATE_PENDING_EXIT,
    STATE_EXIT_CONFIRMED,
    STATE_REMOVED,
    DEMOTION_THRESHOLD,
    RECOVERY_THRESHOLD,
    _average_days_in_state,
    _now_utc,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FakePressure:
    """Minimal stand-in for DemotionPressure."""
    symbol: str
    pressure_score: float
    is_candidate: bool


def _make_pressure(symbol: str, score: float) -> _FakePressure:
    return _FakePressure(symbol=symbol, pressure_score=score,
                         is_candidate=score >= DEMOTION_THRESHOLD)


def _make_engine(tmp_path: Path) -> DemotionActuationEngine:
    return DemotionActuationEngine(
        pending_demotion_file=tmp_path / "pending_demotion.jsonl",
        metrics_file=tmp_path / "demotion_metrics.jsonl",
    )


def _portfolio_with_position(symbol: str, qty: int = 100) -> dict:
    if qty == 0:
        return {"position_entry_dates": {}, "positions": {}}
    return {
        "position_entry_dates": {symbol: "2026-05-01"},
        "positions": {symbol: {"quantity": qty}},
    }


def _empty_portfolio() -> dict:
    return {"position_entry_dates": {}, "positions": {}}


def _shadow(symbols) -> dict:
    return {s: "sector_X" for s in symbols}


def _live(symbols) -> dict:
    return {s: "sector_Y" for s in symbols}


# ─────────────────────────────────────────────────────────────────────────────
# 1. New DEMOTION_CANDIDATE created when pressure ≥ threshold
# ─────────────────────────────────────────────────────────────────────────────
def test_demotion_candidate_creation(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", DEMOTION_THRESHOLD)}
    result = eng.run(
        pressures=pressures,
        portfolio_state=_portfolio_with_position("7203"),
        shadow_universe=_shadow(["9984"]),
        live_universe=_live(["7203"]),
    )
    assert "7203" in result.demotion_candidates
    assert result.recovered_count == 0
    assert result.symbols_to_remove == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recovery: DEMOTION_CANDIDATE → ACTIVE when pressure < RECOVERY_THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────
def test_recovery_to_active(tmp_path):
    eng = _make_engine(tmp_path)
    # Seed DEMOTION_CANDIDATE state
    pressures_high = {"7203": _make_pressure("7203", DEMOTION_THRESHOLD)}
    r1 = eng.run(pressures_high, _portfolio_with_position("7203"),
                 _shadow(["9984"]), _live(["7203"]))
    eng.write_state(r1.transitions)
    assert "7203" in r1.demotion_candidates

    # Pressure drops below recovery threshold
    pressures_low = {"7203": _make_pressure("7203", RECOVERY_THRESHOLD - 0.05)}
    r2 = eng.run(pressures_low, _portfolio_with_position("7203"),
                 _shadow(["9984"]), _live(["7203"]))
    assert "7203" not in r2.demotion_candidates
    assert r2.recovered_count == 1
    assert r2.symbols_to_remove == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. No progress to PENDING_EXIT without a replacement candidate
# ─────────────────────────────────────────────────────────────────────────────
def test_no_progress_without_replacement_candidate(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    # shadow_universe is empty → no replacement
    result = eng.run(
        pressures=pressures,
        portfolio_state=_portfolio_with_position("7203"),
        shadow_universe={},
        live_universe=_live(["7203"]),
    )
    assert "7203" in result.demotion_candidates
    assert result.pending_exit_count == 0
    assert result.symbols_to_remove == []


# ─────────────────────────────────────────────────────────────────────────────
# 4. Advance to PENDING_EXIT when position exists AND replacement available
# ─────────────────────────────────────────────────────────────────────────────
def test_pending_exit_when_position_exists(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    result = eng.run(
        pressures=pressures,
        portfolio_state=_portfolio_with_position("7203", qty=200),
        shadow_universe=_shadow(["9984"]),  # replacement exists
        live_universe=_live(["7203"]),
    )
    # First run creates DEMOTION_CANDIDATE
    eng.write_state(result.transitions)

    # Second run with same high pressure + position → should advance to PENDING_EXIT
    result2 = eng.run(
        pressures=pressures,
        portfolio_state=_portfolio_with_position("7203", qty=200),
        shadow_universe=_shadow(["9984"]),
        live_universe=_live(["7203"]),
    )
    assert result2.pending_exit_count == 1
    assert "7203" not in result2.demotion_candidates


# ─────────────────────────────────────────────────────────────────────────────
# 5. PENDING_EXIT → EXIT_CONFIRMED when position closed
# ─────────────────────────────────────────────────────────────────────────────
def test_exit_confirmed_after_position_closed(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}

    # Step 1: create DEMOTION_CANDIDATE
    r1 = eng.run(pressures, _portfolio_with_position("7203"),
                 _shadow(["9984"]), _live(["7203"]))
    eng.write_state(r1.transitions)

    # Step 2: advance to PENDING_EXIT
    r2 = eng.run(pressures, _portfolio_with_position("7203"),
                 _shadow(["9984"]), _live(["7203"]))
    eng.write_state(r2.transitions)
    assert r2.pending_exit_count == 1

    # Step 3: position closed → EXIT_CONFIRMED
    r3 = eng.run(pressures, _empty_portfolio(),
                 _shadow(["9984"]), _live(["7203"]))
    assert r3.exit_confirmed_count == 1
    assert r3.pending_exit_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. REMOVED only after EXIT_CONFIRMED — not earlier
# ─────────────────────────────────────────────────────────────────────────────
def test_removed_only_after_exit_confirmed(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    live = _live(["7203"])
    shadow = _shadow(["9984"])

    # DEMOTION_CANDIDATE
    r1 = eng.run(pressures, _portfolio_with_position("7203"), shadow, live)
    eng.write_state(r1.transitions)
    assert r1.symbols_to_remove == []

    # PENDING_EXIT
    r2 = eng.run(pressures, _portfolio_with_position("7203"), shadow, live)
    eng.write_state(r2.transitions)
    assert r2.symbols_to_remove == []

    # EXIT_CONFIRMED
    r3 = eng.run(pressures, _empty_portfolio(), shadow, live)
    eng.write_state(r3.transitions)
    assert r3.symbols_to_remove == []    # EXIT_CONFIRMED, not yet REMOVED
    assert r3.exit_confirmed_count == 1

    # REMOVED — next run confirms it
    r4 = eng.run(pressures, _empty_portfolio(), shadow, live)
    assert "7203" in r4.symbols_to_remove
    assert r4.removed_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. FAIL_OPEN on storage failure — no exception raised
# ─────────────────────────────────────────────────────────────────────────────
def test_fail_open_on_storage_failure(tmp_path):
    # Use a path inside a non-existent deeply nested dir that can't be created
    bad_dir = tmp_path / "x" / "y" / "z"
    eng = DemotionActuationEngine(
        pending_demotion_file=bad_dir / "pending.jsonl",
        metrics_file=bad_dir / "metrics.jsonl",
    )
    pressures = {"7203": _make_pressure("7203", 0.80)}
    # run() should not raise even if write fails
    result = eng.run(pressures, _portfolio_with_position("7203"),
                     _shadow(["9984"]), _live(["7203"]))
    assert isinstance(result, DemotionActuationResult)
    # write_state and write_metrics should also not raise
    eng.write_state(result.transitions)
    eng.write_metrics(result)


# ─────────────────────────────────────────────────────────────────────────────
# 8. No sell order generated — symbols_to_remove ≠ sell signal
# ─────────────────────────────────────────────────────────────────────────────
def test_no_sell_order_generated(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    live = _live(["7203"])
    shadow = _shadow(["9984"])

    for _ in range(3):
        r = eng.run(pressures, _portfolio_with_position("7203"), shadow, live)
        eng.write_state(r.transitions)

    r = eng.run(pressures, _empty_portfolio(), shadow, live)
    eng.write_state(r.transitions)

    r_final = eng.run(pressures, _empty_portfolio(), shadow, live)
    # symbols_to_remove is just a list of strings — no execution fields
    for sym in r_final.symbols_to_remove:
        assert isinstance(sym, str)
    # result has no "side", "qty", "price", "order" fields
    result_dict = r_final.to_dict()
    for exec_field in ("side", "qty", "price", "order", "sell"):
        assert exec_field not in result_dict


# ─────────────────────────────────────────────────────────────────────────────
# 9. State recovery after restart (JSONL replay)
# ─────────────────────────────────────────────────────────────────────────────
def test_state_recovery_after_restart(tmp_path):
    state_file = tmp_path / "pending_demotion.jsonl"
    metrics_file = tmp_path / "metrics.jsonl"
    pressures = {"7203": _make_pressure("7203", 0.80)}

    eng1 = DemotionActuationEngine(state_file, metrics_file)
    r1 = eng1.run(pressures, _portfolio_with_position("7203"),
                  _shadow(["9984"]), _live(["7203"]))
    eng1.write_state(r1.transitions)

    # Simulate restart: new engine instance, same file
    eng2 = DemotionActuationEngine(state_file, metrics_file)
    state = eng2.load_state()
    assert "7203" in state
    assert state["7203"].state == STATE_CANDIDATE


# ─────────────────────────────────────────────────────────────────────────────
# 10. Idempotent: running twice with same input doesn't corrupt state
# ─────────────────────────────────────────────────────────────────────────────
def test_duplicate_symbol_idempotent(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    portfolio = _portfolio_with_position("7203")
    shadow = _shadow(["9984"])
    live = _live(["7203"])

    r1 = eng.run(pressures, portfolio, shadow, live)
    eng.write_state(r1.transitions)

    r2 = eng.run(pressures, portfolio, shadow, live)
    eng.write_state(r2.transitions)

    r3 = eng.run(pressures, portfolio, shadow, live)
    # 7203 should appear exactly once in candidates
    assert r3.demotion_candidates.count("7203") <= 1

    state = eng.load_state()
    assert list(state.keys()).count("7203") == 1


# ─────────────────────────────────────────────────────────────────────────────
# 11. Universe not reduced without replacement candidate
# ─────────────────────────────────────────────────────────────────────────────
def test_universe_size_not_reduced_without_replacement(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    live = _live(["7203"])

    for _ in range(5):
        r = eng.run(pressures, _empty_portfolio(), {}, live)  # empty shadow
        eng.write_state(r.transitions)

    assert r.symbols_to_remove == []


# ─────────────────────────────────────────────────────────────────────────────
# 12. REMOVED is terminal — stays REMOVED on subsequent runs
# ─────────────────────────────────────────────────────────────────────────────
def test_removed_is_terminal_state(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    live = _live(["7203"])
    shadow = _shadow(["9984"])

    # Advance through full lifecycle
    for _ in range(2):
        r = eng.run(pressures, _portfolio_with_position("7203"), shadow, live)
        eng.write_state(r.transitions)
    r = eng.run(pressures, _empty_portfolio(), shadow, live)
    eng.write_state(r.transitions)
    r = eng.run(pressures, _empty_portfolio(), shadow, live)
    eng.write_state(r.transitions)

    # Should be REMOVED now
    state = eng.load_state()
    if "7203" in state:
        assert state["7203"].state == STATE_REMOVED

    # Run again — stays REMOVED, not re-added
    r_after = eng.run(pressures, _empty_portfolio(), shadow, live)
    if "7203" in r_after.state_snapshot:
        assert r_after.state_snapshot["7203"].state == STATE_REMOVED
    assert r_after.symbols_to_remove == []  # already removed, no repeat


# ─────────────────────────────────────────────────────────────────────────────
# 13. DEMOTION_CANDIDATE → EXIT_CONFIRMED directly when no position
# ─────────────────────────────────────────────────────────────────────────────
def test_candidate_to_exit_confirmed_no_position(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    # No position → should skip PENDING_EXIT and go straight to EXIT_CONFIRMED
    result = eng.run(pressures, _empty_portfolio(),
                     _shadow(["9984"]), _live(["7203"]))
    assert result.exit_confirmed_count == 1
    assert result.pending_exit_count == 0
    assert "7203" not in result.demotion_candidates


# ─────────────────────────────────────────────────────────────────────────────
# 14. Metrics counts are correct
# ─────────────────────────────────────────────────────────────────────────────
def test_metrics_counts_correct(tmp_path):
    eng = _make_engine(tmp_path)
    # 3 candidates: A with position, B with position, C without position
    pressures = {
        "A": _make_pressure("A", 0.80),
        "B": _make_pressure("B", 0.80),
        "C": _make_pressure("C", 0.80),
    }
    portfolio = {
        "position_entry_dates": {"A": "2026-05-01", "B": "2026-05-01"},
        "positions": {"A": {"quantity": 100}, "B": {"quantity": 50}},
    }
    shadow = _shadow(["X", "Y"])
    live = _live(["A", "B", "C"])

    r = eng.run(pressures, portfolio, shadow, live)
    # C has no position → EXIT_CONFIRMED directly
    assert r.exit_confirmed_count == 1
    # A, B have positions → DEMOTION_CANDIDATE (first run, no prior state)
    assert len(r.demotion_candidates) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 15. Transitions written to JSONL correctly
# ─────────────────────────────────────────────────────────────────────────────
def test_transitions_written_to_jsonl(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    r = eng.run(pressures, _portfolio_with_position("7203"),
                _shadow(["9984"]), _live(["7203"]))
    eng.write_state(r.transitions)

    state_file = tmp_path / "pending_demotion.jsonl"
    assert state_file.exists()
    lines = [l for l in state_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    assert rec["symbol"] == "7203"
    assert rec["state"] == STATE_CANDIDATE


# ─────────────────────────────────────────────────────────────────────────────
# 16. load_state replays JSONL → correct latest state per symbol
# ─────────────────────────────────────────────────────────────────────────────
def test_load_state_replays_correctly(tmp_path):
    state_file = tmp_path / "pending_demotion.jsonl"
    metrics_file = tmp_path / "metrics.jsonl"
    eng = DemotionActuationEngine(state_file, metrics_file)

    # Write DEMOTION_CANDIDATE record
    rec_cand = DemotionRecord(
        symbol="7203", state=STATE_CANDIDATE,
        entered_at=_now_utc(), updated_at=_now_utc(),
        reason="test", demotion_pressure=0.75,
        replacement_delta=0.0, position_size=100,
    )
    eng.write_state([rec_cand])

    # Write PENDING_EXIT record (overrides CANDIDATE in replay)
    rec_pending = DemotionRecord(
        symbol="7203", state=STATE_PENDING_EXIT,
        entered_at=rec_cand.entered_at, updated_at=_now_utc(),
        reason="test2", demotion_pressure=0.80,
        replacement_delta=0.5, position_size=100,
    )
    eng.write_state([rec_pending])

    state = eng.load_state()
    assert state["7203"].state == STATE_PENDING_EXIT


# ─────────────────────────────────────────────────────────────────────────────
# 17. Hysteresis: pressure in band [RECOVERY_THRESHOLD, DEMOTION_THRESHOLD)
#     → stays in DEMOTION_CANDIDATE, does NOT recover
# ─────────────────────────────────────────────────────────────────────────────
def test_recovery_threshold_hysteresis(tmp_path):
    eng = _make_engine(tmp_path)
    pressures_high = {"7203": _make_pressure("7203", DEMOTION_THRESHOLD)}
    r1 = eng.run(pressures_high, _portfolio_with_position("7203"),
                 _shadow(["9984"]), _live(["7203"]))
    eng.write_state(r1.transitions)

    # Pressure drops into hysteresis band — neither recover nor advance
    mid_score = (RECOVERY_THRESHOLD + DEMOTION_THRESHOLD) / 2
    pressures_mid = {"7203": _FakePressure("7203", mid_score, is_candidate=False)}
    r2 = eng.run(pressures_mid, _portfolio_with_position("7203"),
                 _shadow(["9984"]), _live(["7203"]))
    # Still in DEMOTION_CANDIDATE (no recovery, no advancement without candidate)
    # With no replacement and mid pressure: stays CANDIDATE
    state = r2.state_snapshot
    assert "7203" in state
    assert r2.recovered_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 18. Multiple symbols processed independently
# ─────────────────────────────────────────────────────────────────────────────
def test_multiple_symbols_independent(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {
        "A": _make_pressure("A", 0.80),
        "B": _make_pressure("B", 0.70),
        "C": _make_pressure("C", 0.20),  # below threshold
    }
    portfolio = {
        "position_entry_dates": {"A": "2026-05-01", "B": "2026-05-01"},
        "positions": {"A": {"quantity": 100}, "B": {"quantity": 100}},
    }
    result = eng.run(pressures, portfolio, _shadow(["X"]), _live(["A", "B", "C"]))
    assert "A" in result.demotion_candidates or result.pending_exit_count >= 0
    assert "C" not in result.demotion_candidates  # below threshold
    assert len(result.demotion_candidates) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 19. write_state is append-only (does not truncate existing records)
# ─────────────────────────────────────────────────────────────────────────────
def test_write_state_append_only(tmp_path):
    eng = _make_engine(tmp_path)
    state_file = tmp_path / "pending_demotion.jsonl"

    for sym in ["A", "B", "C"]:
        eng.write_state([DemotionRecord(
            symbol=sym, state=STATE_CANDIDATE,
            entered_at=_now_utc(), updated_at=_now_utc(),
            reason="test", demotion_pressure=0.70,
            replacement_delta=0.0, position_size=100,
        )])

    lines = [l for l in state_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    symbols_written = [json.loads(l)["symbol"] for l in lines]
    assert set(symbols_written) == {"A", "B", "C"}


# ─────────────────────────────────────────────────────────────────────────────
# 20. average_days_pending_exit utility
# ─────────────────────────────────────────────────────────────────────────────
def test_average_days_pending_exit(tmp_path):
    now = _now_utc()
    five_days_ago = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()

    recs = [
        DemotionRecord(
            symbol="X", state=STATE_PENDING_EXIT,
            entered_at=five_days_ago, updated_at=now,
            reason="test", demotion_pressure=0.75,
            replacement_delta=0.5, position_size=100,
        )
    ]
    avg = _average_days_in_state(recs, now)
    assert 4.9 <= avg <= 5.1


# ─────────────────────────────────────────────────────────────────────────────
# Extra: empty pressures dict doesn't crash
# ─────────────────────────────────────────────────────────────────────────────
def test_empty_pressures_no_crash(tmp_path):
    eng = _make_engine(tmp_path)
    result = eng.run({}, _empty_portfolio(), {}, {})
    assert isinstance(result, DemotionActuationResult)
    assert result.demotion_candidates == []
    assert result.symbols_to_remove == []


# ─────────────────────────────────────────────────────────────────────────────
# Extra: has_replacement with EV scores
# ─────────────────────────────────────────────────────────────────────────────
def test_has_replacement_with_ev_scores(tmp_path):
    eng = _make_engine(tmp_path)
    pressures = {"7203": _make_pressure("7203", 0.80)}
    ev_scores = {"9984": 0.15, "7203": 0.05}

    # First run: create DEMOTION_CANDIDATE
    r1 = eng.run(pressures, _portfolio_with_position("7203"),
                 _shadow(["9984"]), _live(["7203"]), ev_scores=ev_scores)
    eng.write_state(r1.transitions)

    # Second run: 9984 EV (0.15) > 7203 EV (0.05) → replacement_delta > 0 → PENDING_EXIT
    r2 = eng.run(pressures, _portfolio_with_position("7203"),
                 _shadow(["9984"]), _live(["7203"]), ev_scores=ev_scores)
    assert r2.pending_exit_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Extra: ACTIVE recovery record in JSONL removes symbol from tracking
# ─────────────────────────────────────────────────────────────────────────────
def test_active_record_removes_from_tracking(tmp_path):
    eng = _make_engine(tmp_path)

    # Write CANDIDATE then ACTIVE (recovery)
    eng.write_state([DemotionRecord(
        symbol="7203", state=STATE_CANDIDATE,
        entered_at=_now_utc(), updated_at=_now_utc(),
        reason="high_pressure", demotion_pressure=0.75,
        replacement_delta=0.0, position_size=100,
    )])
    eng.write_state([DemotionRecord(
        symbol="7203", state=STATE_ACTIVE,
        entered_at=_now_utc(), updated_at=_now_utc(),
        reason="recovered", demotion_pressure=0.30,
        replacement_delta=0.0, position_size=100,
    )])

    state = eng.load_state()
    assert "7203" not in state
