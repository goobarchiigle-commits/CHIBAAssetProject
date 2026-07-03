"""
tests/universe/test_rotation_planner.py

Tests for Phase 3B: Rotation Planner.

Covers:
  1.  test_plan_creation_basic
  2.  test_target_universe_size_maintained
  3.  test_deficit_filled_with_best_candidates
  4.  test_stale_candidates_rejected
  5.  test_non_stale_candidates_accepted
  6.  test_plan_status_initial_proposed
  7.  test_plan_id_deterministic
  8.  test_rotation_score_computed
  9.  test_rotation_score_zero_no_candidates
  10. test_outcome_status_pending
  11. test_write_plan_atomic
  12. test_write_history_appends
  13. test_load_plan_roundtrip
  14. test_update_plan_status_lifecycle
  15. test_update_plan_status_terminal_blocked
  16. test_update_plan_status_bad_id_rejected
  17. test_replay_from_plan_identical_outputs
  18. test_no_shrink_below_current_size
  19. test_candidate_ranking_by_composite_score
  20. test_ev_scores_used_for_ranking
  21. test_rsr_scores_used_for_ranking
  22. test_fail_open_write_error
  23. test_decision_inputs_complete_for_replay
  24. test_no_duplicate_candidates
  25. test_empty_shadow_no_candidates
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from universe.rotation_planner import (
    RotationPlanner,
    RotationPlan,
    RotationPlanHistoryRecord,
    CandidateRecord,
    PLAN_STATUS_PROPOSED,
    PLAN_STATUS_APPROVED,
    PLAN_STATUS_EXECUTED,
    PLAN_STATUS_CANCELLED,
    OUTCOME_STATUS_PENDING,
    TARGET_UNIVERSE_SIZE,
    MAX_CANDIDATE_AGE_DAYS,
    _staleness,
    _now_utc,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_planner(tmp_path: Path, target: int = 5) -> RotationPlanner:
    return RotationPlanner(
        rotation_plan_file=tmp_path / "rotation_plan.json",
        rotation_plan_history_file=tmp_path / "rotation_plan_history.jsonl",
        target_universe_size=target,
        max_candidate_age_days=7,
    )


def _live(symbols) -> dict:
    return {s: "sector_A" for s in symbols}


def _shadow(symbols) -> dict:
    return {s: "sector_B" for s in symbols}


def _fresh_scored(symbols, offset_days: int = 0) -> dict:
    """scored_at = today minus offset_days."""
    dt = datetime.now(timezone.utc) - timedelta(days=offset_days)
    iso = dt.isoformat()
    return {s: iso for s in symbols}


def _stale_scored(symbols, age_days: int = 10) -> dict:
    """scored_at = today minus age_days (stale)."""
    return _fresh_scored(symbols, offset_days=age_days)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic plan creation
# ─────────────────────────────────────────────────────────────────────────────
def test_plan_creation_basic(tmp_path):
    pl = _make_planner(tmp_path, target=5)
    plan = pl.plan(
        live_universe=_live(["A", "B", "C"]),
        shadow_universe=_shadow(["X", "Y"]),
    )
    assert isinstance(plan, RotationPlan)
    assert plan.plan_status == PLAN_STATUS_PROPOSED
    assert plan.plan_id != ""
    assert plan.schema_version == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Universe size maintained: deficit filled to target
# ─────────────────────────────────────────────────────────────────────────────
def test_target_universe_size_maintained(tmp_path):
    pl = _make_planner(tmp_path, target=5)
    # live=3, remove=1 → post_removal=2, deficit=3
    plan = pl.plan(
        live_universe=_live(["A", "B", "C"]),
        shadow_universe=_shadow(["X", "Y", "Z", "W"]),
        symbols_to_remove=["C"],
    )
    assert plan.post_removal_size == 2
    assert plan.deficit == 3
    assert len(plan.symbols_to_add) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. Deficit filled with best-scoring candidates
# ─────────────────────────────────────────────────────────────────────────────
def test_deficit_filled_with_best_candidates(tmp_path):
    pl = _make_planner(tmp_path, target=4)
    # live=3, remove=1 → deficit=2
    plan = pl.plan(
        live_universe=_live(["A", "B", "C"]),
        shadow_universe=_shadow(["X", "Y", "Z"]),
        symbols_to_remove=["C"],
        ev_scores={"X": 0.10, "Y": 0.20, "Z": 0.05},
    )
    # Best 2: Y (0.20), X (0.10), skip Z (0.05)
    assert len(plan.symbols_to_add) == 2
    assert "Y" in plan.symbols_to_add
    assert "X" in plan.symbols_to_add
    assert "Z" not in plan.symbols_to_add


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stale candidates rejected
# ─────────────────────────────────────────────────────────────────────────────
def test_stale_candidates_rejected(tmp_path):
    pl = _make_planner(tmp_path, target=4)
    stale = _stale_scored(["X", "Y"], age_days=10)  # > 7 days
    plan = pl.plan(
        live_universe=_live(["A"]),
        shadow_universe=_shadow(["X", "Y"]),
        candidate_scored_at=stale,
    )
    assert len(plan.symbols_to_add) == 0  # all stale
    assert len(plan.rejected_candidates) == 2
    for c in plan.rejected_candidates:
        assert c.is_stale


# ─────────────────────────────────────────────────────────────────────────────
# 5. Non-stale candidates accepted
# ─────────────────────────────────────────────────────────────────────────────
def test_non_stale_candidates_accepted(tmp_path):
    pl = _make_planner(tmp_path, target=3)
    fresh = _fresh_scored(["X", "Y"], offset_days=1)  # 1 day old → fresh
    plan = pl.plan(
        live_universe=_live(["A"]),
        shadow_universe=_shadow(["X", "Y"]),
        candidate_scored_at=fresh,
    )
    assert len(plan.rejected_candidates) == 0
    assert len(plan.replacement_candidates) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 6. Initial plan_status is PROPOSED
# ─────────────────────────────────────────────────────────────────────────────
def test_plan_status_initial_proposed(tmp_path):
    pl = _make_planner(tmp_path)
    plan = pl.plan(live_universe=_live(["A"]), shadow_universe=_shadow(["X"]))
    assert plan.plan_status == PLAN_STATUS_PROPOSED


# ─────────────────────────────────────────────────────────────────────────────
# 7. plan_id is deterministic: same inputs → same plan_id
# ─────────────────────────────────────────────────────────────────────────────
def test_plan_id_deterministic(tmp_path):
    ref_date = "2026-05-31"
    live   = _live(["A", "B"])
    shadow = _shadow(["X", "Y"])
    ev     = {"X": 0.10, "Y": 0.20}
    rsr    = {"X": 70.0, "Y": 80.0}
    scored = _fresh_scored(["X", "Y"])

    pl1 = _make_planner(tmp_path, target=4)
    pl2 = _make_planner(tmp_path, target=4)

    plan1 = pl1.plan(live, shadow, ev_scores=ev, rsr_scores=rsr,
                     candidate_scored_at=scored, reference_date=ref_date)
    plan2 = pl2.plan(live, shadow, ev_scores=ev, rsr_scores=rsr,
                     candidate_scored_at=scored, reference_date=ref_date)

    assert plan1.plan_id == plan2.plan_id
    assert plan1.symbols_to_add == plan2.symbols_to_add
    assert plan1.rotation_score == plan2.rotation_score


# ─────────────────────────────────────────────────────────────────────────────
# 8. rotation_score is mean composite of selected candidates
# ─────────────────────────────────────────────────────────────────────────────
def test_rotation_score_computed(tmp_path):
    pl = _make_planner(tmp_path, target=3)
    # live=1, shadow=2 → deficit=2, both selected
    ev  = {"X": 0.10, "Y": 0.20}
    rsr = {"X": 60.0, "Y": 80.0}
    plan = pl.plan(
        live_universe=_live(["A"]),
        shadow_universe=_shadow(["X", "Y"]),
        ev_scores=ev, rsr_scores=rsr,
    )
    # composite(X) = 0.6*0.10 + 0.4*(60/100) = 0.06 + 0.24 = 0.30
    # composite(Y) = 0.6*0.20 + 0.4*(80/100) = 0.12 + 0.32 = 0.44
    # mean = (0.30 + 0.44) / 2 = 0.37
    assert abs(plan.rotation_score - 0.37) < 1e-4
    assert len(plan.symbols_to_add) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 9. rotation_score is 0.0 when no candidates selected
# ─────────────────────────────────────────────────────────────────────────────
def test_rotation_score_zero_no_candidates(tmp_path):
    pl = _make_planner(tmp_path, target=2)
    # live already at target, no deficit
    plan = pl.plan(
        live_universe=_live(["A", "B"]),
        shadow_universe=_shadow(["X"]),
    )
    assert plan.deficit == 0
    assert plan.rotation_score == 0.0
    assert plan.symbols_to_add == []


# ─────────────────────────────────────────────────────────────────────────────
# 10. History record has outcome_status=PENDING
# ─────────────────────────────────────────────────────────────────────────────
def test_outcome_status_pending(tmp_path):
    pl = _make_planner(tmp_path)
    plan = pl.plan(live_universe=_live(["A"]), shadow_universe=_shadow(["X"]))
    pl.write_history(plan)

    hist_file = tmp_path / "rotation_plan_history.jsonl"
    lines = [l for l in hist_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["outcome_status"] == OUTCOME_STATUS_PENDING
    assert "created_at" in rec
    assert "rotation_score" in rec


# ─────────────────────────────────────────────────────────────────────────────
# 11. write_plan writes atomically (tmp → rename)
# ─────────────────────────────────────────────────────────────────────────────
def test_write_plan_atomic(tmp_path):
    pl = _make_planner(tmp_path)
    plan = pl.plan(live_universe=_live(["A"]), shadow_universe=_shadow(["X"]))
    pl.write_plan(plan)

    plan_file = tmp_path / "rotation_plan.json"
    assert plan_file.exists()
    # .tmp should not remain
    assert not plan_file.with_suffix(".tmp").exists()

    loaded = json.loads(plan_file.read_text(encoding="utf-8"))
    assert loaded["plan_id"] == plan.plan_id
    assert loaded["plan_status"] == PLAN_STATUS_PROPOSED


# ─────────────────────────────────────────────────────────────────────────────
# 12. write_history appends without truncating
# ─────────────────────────────────────────────────────────────────────────────
def test_write_history_appends(tmp_path):
    pl = _make_planner(tmp_path)
    for i in range(3):
        plan = pl.plan(
            live_universe=_live(["A"]),
            shadow_universe=_shadow([f"X{i}"]),
            ev_scores={f"X{i}": float(i) * 0.1},
            reference_date=f"2026-06-0{i+1}",
        )
        pl.write_history(plan)

    hist_file = tmp_path / "rotation_plan_history.jsonl"
    lines = [l for l in hist_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 13. load_plan roundtrip: write then load gives identical plan
# ─────────────────────────────────────────────────────────────────────────────
def test_load_plan_roundtrip(tmp_path):
    pl = _make_planner(tmp_path, target=4)
    plan = pl.plan(
        live_universe=_live(["A", "B"]),
        shadow_universe=_shadow(["X", "Y", "Z"]),
        ev_scores={"X": 0.10, "Y": 0.20, "Z": 0.05},
        rsr_scores={"X": 70.0, "Y": 80.0, "Z": 50.0},
    )
    pl.write_plan(plan)

    loaded = pl.load_plan()
    assert loaded is not None
    assert loaded.plan_id == plan.plan_id
    assert loaded.symbols_to_add == plan.symbols_to_add
    assert loaded.rotation_score == plan.rotation_score
    assert loaded.deficit == plan.deficit


# ─────────────────────────────────────────────────────────────────────────────
# 14. Plan lifecycle: PROPOSED → APPROVED → EXECUTED
# ─────────────────────────────────────────────────────────────────────────────
def test_update_plan_status_lifecycle(tmp_path):
    pl = _make_planner(tmp_path)
    plan = pl.plan(live_universe=_live(["A"]), shadow_universe=_shadow(["X"]))
    pl.write_plan(plan)

    assert pl.update_plan_status(plan.plan_id, PLAN_STATUS_APPROVED)
    assert pl.load_plan().plan_status == PLAN_STATUS_APPROVED

    assert pl.update_plan_status(plan.plan_id, PLAN_STATUS_EXECUTED)
    assert pl.load_plan().plan_status == PLAN_STATUS_EXECUTED


# ─────────────────────────────────────────────────────────────────────────────
# 15. Terminal states (EXECUTED / CANCELLED) block further updates
# ─────────────────────────────────────────────────────────────────────────────
def test_update_plan_status_terminal_blocked(tmp_path):
    pl = _make_planner(tmp_path)
    plan = pl.plan(live_universe=_live(["A"]), shadow_universe=_shadow(["X"]))
    pl.write_plan(plan)

    pl.update_plan_status(plan.plan_id, PLAN_STATUS_EXECUTED)
    result = pl.update_plan_status(plan.plan_id, PLAN_STATUS_APPROVED)
    assert result is False
    assert pl.load_plan().plan_status == PLAN_STATUS_EXECUTED


# ─────────────────────────────────────────────────────────────────────────────
# 16. Wrong plan_id rejected on update
# ─────────────────────────────────────────────────────────────────────────────
def test_update_plan_status_bad_id_rejected(tmp_path):
    pl = _make_planner(tmp_path)
    plan = pl.plan(live_universe=_live(["A"]), shadow_universe=_shadow(["X"]))
    pl.write_plan(plan)

    result = pl.update_plan_status("wrong_plan_id", PLAN_STATUS_APPROVED)
    assert result is False
    assert pl.load_plan().plan_status == PLAN_STATUS_PROPOSED


# ─────────────────────────────────────────────────────────────────────────────
# 17. Replay: reconstructing from decision_inputs gives identical outputs
# ─────────────────────────────────────────────────────────────────────────────
def test_replay_from_plan_identical_outputs(tmp_path):
    ref = "2026-05-31"
    ev  = {"X": 0.15, "Y": 0.25}
    rsr = {"X": 70.0, "Y": 85.0}
    scored = _fresh_scored(["X", "Y"])

    pl = _make_planner(tmp_path, target=4)
    original = pl.plan(
        live_universe=_live(["A", "B"]),
        shadow_universe=_shadow(["X", "Y"]),
        ev_scores=ev, rsr_scores=rsr,
        candidate_scored_at=scored,
        reference_date=ref,
    )
    pl.write_plan(original)

    loaded = pl.load_plan()
    replayed = pl.replay_from_plan(loaded)

    # Core decision outputs must be identical
    assert replayed.symbols_to_add == original.symbols_to_add
    assert replayed.rotation_score == original.rotation_score
    assert replayed.deficit == original.deficit
    assert replayed.plan_id == original.plan_id  # same inputs → same plan_id


# ─────────────────────────────────────────────────────────────────────────────
# 18. No shrink: if live already >= target, no candidates needed
# ─────────────────────────────────────────────────────────────────────────────
def test_no_shrink_below_current_size(tmp_path):
    pl = _make_planner(tmp_path, target=3)
    # live=5 (already above target), no removal
    plan = pl.plan(
        live_universe=_live(["A", "B", "C", "D", "E"]),
        shadow_universe=_shadow(["X"]),
    )
    assert plan.deficit == 0
    assert plan.symbols_to_add == []


# ─────────────────────────────────────────────────────────────────────────────
# 19. Candidates ranked by composite score descending
# ─────────────────────────────────────────────────────────────────────────────
def test_candidate_ranking_by_composite_score(tmp_path):
    pl = _make_planner(tmp_path, target=10)
    ev = {"A": 0.05, "B": 0.30, "C": 0.15}
    plan = pl.plan(
        live_universe={},
        shadow_universe=_shadow(["A", "B", "C"]),
        ev_scores=ev,
    )
    scores = [c.composite_score for c in plan.replacement_candidates]
    assert scores == sorted(scores, reverse=True)
    assert plan.replacement_candidates[0].symbol == "B"


# ─────────────────────────────────────────────────────────────────────────────
# 20. EV scores drive ranking when RSR absent
# ─────────────────────────────────────────────────────────────────────────────
def test_ev_scores_used_for_ranking(tmp_path):
    pl = _make_planner(tmp_path, target=3)
    plan = pl.plan(
        live_universe=_live(["A"]),
        shadow_universe=_shadow(["X", "Y"]),
        ev_scores={"X": 0.05, "Y": 0.30},
    )
    assert plan.symbols_to_add[0] == "Y"  # higher EV selected first


# ─────────────────────────────────────────────────────────────────────────────
# 21. RSR scores contribute to ranking
# ─────────────────────────────────────────────────────────────────────────────
def test_rsr_scores_used_for_ranking(tmp_path):
    pl = _make_planner(tmp_path, target=3)
    # Equal EV, different RSR
    plan = pl.plan(
        live_universe=_live(["A"]),
        shadow_universe=_shadow(["X", "Y"]),
        ev_scores={"X": 0.10, "Y": 0.10},
        rsr_scores={"X": 40.0, "Y": 90.0},
    )
    # Y has higher RSR → higher composite → selected first
    assert plan.symbols_to_add[0] == "Y"


# ─────────────────────────────────────────────────────────────────────────────
# 22. FAIL_OPEN on write error — no exception raised
# ─────────────────────────────────────────────────────────────────────────────
def test_fail_open_write_error(tmp_path):
    bad = tmp_path / "no_such_dir" / "deep" / "nested"
    pl = RotationPlanner(
        rotation_plan_file=bad / "plan.json",
        rotation_plan_history_file=bad / "history.jsonl",
        target_universe_size=3,
    )
    plan = pl.plan(live_universe=_live(["A"]), shadow_universe=_shadow(["X"]))
    # Should not raise
    pl.write_plan(plan)
    pl.write_history(plan)
    assert isinstance(plan, RotationPlan)


# ─────────────────────────────────────────────────────────────────────────────
# 23. decision_inputs contains all fields needed for replay
# ─────────────────────────────────────────────────────────────────────────────
def test_decision_inputs_complete_for_replay(tmp_path):
    pl = _make_planner(tmp_path, target=4)
    plan = pl.plan(
        live_universe=_live(["A", "B"]),
        shadow_universe=_shadow(["X", "Y"]),
        ev_scores={"X": 0.10, "Y": 0.20},
        rsr_scores={"X": 70.0, "Y": 80.0},
        candidate_scored_at=_fresh_scored(["X", "Y"]),
        reference_date="2026-05-31",
    )
    di = plan.decision_inputs
    required_keys = {
        "live_universe_keys", "shadow_universe_keys", "symbols_to_remove",
        "target_universe_size", "max_candidate_age_days", "reference_date",
        "ev_scores", "rsr_scores", "candidate_scored_at", "scoring_weights",
        "schema_version",
    }
    assert required_keys.issubset(set(di.keys()))
    assert di["target_universe_size"] == 4
    assert "X" in di["ev_scores"]


# ─────────────────────────────────────────────────────────────────────────────
# 24. No duplicate candidates in output
# ─────────────────────────────────────────────────────────────────────────────
def test_no_duplicate_candidates(tmp_path):
    pl = _make_planner(tmp_path, target=10)
    plan = pl.plan(
        live_universe=_live(["A"]),
        shadow_universe=_shadow(["X", "Y", "Z"]),
    )
    all_symbols = [c.symbol for c in plan.replacement_candidates]
    assert len(all_symbols) == len(set(all_symbols))
    assert len(plan.symbols_to_add) == len(set(plan.symbols_to_add))


# ─────────────────────────────────────────────────────────────────────────────
# 25. Empty shadow universe → no candidates, deficit unfilled
# ─────────────────────────────────────────────────────────────────────────────
def test_empty_shadow_no_candidates(tmp_path):
    pl = _make_planner(tmp_path, target=5)
    plan = pl.plan(
        live_universe=_live(["A", "B"]),
        shadow_universe={},
    )
    assert plan.symbols_to_add == []
    assert plan.replacement_candidates == []
    assert plan.deficit > 0  # deficit exists but can't fill


# ─────────────────────────────────────────────────────────────────────────────
# Extra: mixed stale/fresh candidates → only fresh selected
# ─────────────────────────────────────────────────────────────────────────────
def test_mixed_stale_fresh_selection(tmp_path):
    pl = _make_planner(tmp_path, target=4)
    scored = {
        **_fresh_scored(["Y"], offset_days=2),   # fresh
        **_stale_scored(["X"], age_days=10),     # stale
    }
    plan = pl.plan(
        live_universe=_live(["A"]),
        shadow_universe=_shadow(["X", "Y"]),
        ev_scores={"X": 0.50, "Y": 0.10},  # X higher EV but stale
        candidate_scored_at=scored,
    )
    assert "Y" in plan.symbols_to_add
    assert "X" not in plan.symbols_to_add
    assert any(c.symbol == "X" for c in plan.rejected_candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Extra: staleness utility function
# ─────────────────────────────────────────────────────────────────────────────
def test_staleness_fresh():
    ref = datetime.now(timezone.utc)
    scored_iso = (ref - timedelta(days=3)).isoformat()
    age, stale = _staleness(scored_iso, ref, max_age_days=7)
    assert 2.9 <= age <= 3.1
    assert stale is False


def test_staleness_stale():
    ref = datetime.now(timezone.utc)
    scored_iso = (ref - timedelta(days=10)).isoformat()
    age, stale = _staleness(scored_iso, ref, max_age_days=7)
    assert age > 7
    assert stale is True


def test_staleness_empty_scored_at():
    ref = datetime.now(timezone.utc)
    age, stale = _staleness("", ref, max_age_days=7)
    assert age == 0.0
    assert stale is False


# ─────────────────────────────────────────────────────────────────────────────
# Extra: CandidateRecord roundtrip
# ─────────────────────────────────────────────────────────────────────────────
def test_candidate_record_roundtrip():
    rec = CandidateRecord(
        symbol="9984", sector="IT", ev_score=0.15, rsr_score=82.0,
        composite_score=0.418, scored_at="2026-05-31T09:00:00+09:00",
        age_days=0.5, is_stale=False,
    )
    d = rec.to_dict()
    rec2 = CandidateRecord.from_dict(d)
    assert rec2.symbol == rec.symbol
    assert rec2.composite_score == rec.composite_score
    assert rec2.is_stale == rec.is_stale


# ─────────────────────────────────────────────────────────────────────────────
# Extra: load_plan returns None when file absent
# ─────────────────────────────────────────────────────────────────────────────
def test_load_plan_returns_none_when_absent(tmp_path):
    pl = _make_planner(tmp_path)
    assert pl.load_plan() is None


# ─────────────────────────────────────────────────────────────────────────────
# Extra: CANCELLED plan blocked from further updates
# ─────────────────────────────────────────────────────────────────────────────
def test_cancelled_plan_blocked(tmp_path):
    pl = _make_planner(tmp_path)
    plan = pl.plan(live_universe=_live(["A"]), shadow_universe=_shadow(["X"]))
    pl.write_plan(plan)

    pl.update_plan_status(plan.plan_id, PLAN_STATUS_CANCELLED)
    result = pl.update_plan_status(plan.plan_id, PLAN_STATUS_APPROVED)
    assert result is False
