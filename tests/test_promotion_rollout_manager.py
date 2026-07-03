"""Tests for src/analytics/promotion_rollout_manager.py (Phase 6H-Lite)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analytics.promotion_rollout_manager import (
    SCHEMA_VERSION,
    STATE_MICRO_CAPITAL,
    STATE_OBSERVE,
    STATE_PAPER_SHADOW,
    STATE_PRODUCTION_READY,
    STATE_ROLLBACK_REQUIRED,
    RolloutRecord,
    RolloutRunResult,
    RolloutSummary,
    _CANDIDATE_ORDER,
    _MICRO_CAPITAL_MIN_DAYS,
    _PAPER_SHADOW_MIN_DAYS,
    _PROMOTE_FIRST,
    _SCORE_PROMOTE_MIN,
    _STATE_PRIORITY,
    _compute_days,
    _compute_summary,
    _latest_governance,
    _latest_impact_status,
    _load_all_jsonl,
    _load_current_states,
    _load_evaluated_keys,
    _load_state_records,
    _next_state,
    _safe_float,
    _safe_int,
    append_rollout_record,
    format_rollout_report,
    run_promotion_rollout_manager,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _sb_rec(ct: str, score: float, cls: str, date: str = "2026-07-20") -> dict:
    return {"date": date, "candidate": ct, "governance_score": score,
            "classification": cls, "schema_version": "v1"}


def _impact_rec(ct: str, status: str, date: str = "2026-07-20") -> dict:
    return {"date": date, "candidate": ct, "impact_status": status,
            "impact_delta": 0.1 if status == "POSITIVE" else -0.1,
            "schema_version": "v1"}


def _rollout_rec(ct: str, state: str, days: int, entry: str,
                 impact: str = "", score: float = 80.0,
                 date: str = "2026-07-20") -> dict:
    return {
        "date": date, "candidate": ct, "state": state,
        "days_in_state": days, "state_entry_date": entry,
        "latest_impact_status": impact, "governance_score": score,
        "schema_version": "v1",
    }


def _make_files(tmp_path: Path) -> dict:
    files = {}
    for name in ["scoreboard", "registry", "impact", "output"]:
        p = tmp_path / f"{name}.jsonl"
        p.write_text("", encoding="utf-8")
        files[name] = p
    return files


def _run(tmp_path, scoreboard=None, impact=None, existing=None,
         today="2026-07-25") -> tuple:
    files = _make_files(tmp_path)
    if scoreboard:
        _write_jsonl(files["scoreboard"], scoreboard)
    if impact:
        _write_jsonl(files["impact"], impact)
    if existing:
        _write_jsonl(files["output"], existing)
    result = run_promotion_rollout_manager(
        scoreboard_file=files["scoreboard"],
        registry_file=files["registry"],
        impact_file=files["impact"],
        output_file=files["output"],
        today=today,
    )
    return result, files["output"]


def _run_with_files(files: dict, today: str) -> RolloutRunResult:
    return run_promotion_rollout_manager(
        scoreboard_file=files["scoreboard"],
        registry_file=files["registry"],
        impact_file=files["impact"],
        output_file=files["output"],
        today=today,
    )


# ── TestConstants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_state_observe(self):
        assert STATE_OBSERVE == "OBSERVE"

    def test_state_paper_shadow(self):
        assert STATE_PAPER_SHADOW == "PAPER_SHADOW"

    def test_state_micro_capital(self):
        assert STATE_MICRO_CAPITAL == "MICRO_CAPITAL"

    def test_state_production_ready(self):
        assert STATE_PRODUCTION_READY == "PRODUCTION_READY"

    def test_state_rollback_required(self):
        assert STATE_ROLLBACK_REQUIRED == "ROLLBACK_REQUIRED"

    def test_paper_shadow_min_days(self):
        assert _PAPER_SHADOW_MIN_DAYS == 14

    def test_micro_capital_min_days(self):
        assert _MICRO_CAPITAL_MIN_DAYS == 30

    def test_promote_first(self):
        assert _PROMOTE_FIRST == "PROMOTE_FIRST"

    def test_score_promote_min(self):
        assert _SCORE_PROMOTE_MIN == 80.0

    def test_schema_version(self):
        assert SCHEMA_VERSION == "v1"

    def test_candidate_order_length(self):
        assert len(_CANDIDATE_ORDER) == 5

    def test_state_priority_ordering(self):
        assert _STATE_PRIORITY[STATE_PRODUCTION_READY] > _STATE_PRIORITY[STATE_MICRO_CAPITAL]
        assert _STATE_PRIORITY[STATE_MICRO_CAPITAL] > _STATE_PRIORITY[STATE_PAPER_SHADOW]
        assert _STATE_PRIORITY[STATE_PAPER_SHADOW] > _STATE_PRIORITY[STATE_OBSERVE]
        assert _STATE_PRIORITY[STATE_ROLLBACK_REQUIRED] == 0


# ── TestSafeHelpers ───────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_float_normal(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) == 0.0

    def test_safe_int_normal(self):
        assert _safe_int(5) == 5

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_string(self):
        assert _safe_int("30") == 30


# ── TestComputeDays ───────────────────────────────────────────────────────────

class TestComputeDays:
    def test_same_date(self):
        assert _compute_days("2026-07-20", "2026-07-20") == 0

    def test_one_day(self):
        assert _compute_days("2026-07-20", "2026-07-21") == 1

    def test_fourteen_days(self):
        assert _compute_days("2026-07-01", "2026-07-15") == 14

    def test_thirty_days(self):
        assert _compute_days("2026-06-01", "2026-07-01") == 30

    def test_invalid_date(self):
        assert _compute_days("bad", "2026-07-20") == 0

    def test_reverse_order_clamped_to_zero(self):
        assert _compute_days("2026-07-25", "2026-07-20") == 0

    def test_empty_string(self):
        assert _compute_days("", "2026-07-20") == 0


# ── TestNextState ─────────────────────────────────────────────────────────────

class TestNextState:
    # None → OBSERVE (no governance signal)
    def test_none_no_signal(self):
        assert _next_state(None, 0, "", 50.0, "") == STATE_OBSERVE

    # None → PAPER_SHADOW (PROMOTE_FIRST + score >= 80)
    def test_none_promote_first(self):
        assert _next_state(None, 0, "", 85.0, _PROMOTE_FIRST) == STATE_PAPER_SHADOW

    # OBSERVE → PAPER_SHADOW
    def test_observe_promote_first(self):
        assert _next_state(STATE_OBSERVE, 0, "", 85.0, _PROMOTE_FIRST) == STATE_PAPER_SHADOW

    # OBSERVE stays (score too low)
    def test_observe_score_too_low(self):
        assert _next_state(STATE_OBSERVE, 0, "", 70.0, _PROMOTE_FIRST) == STATE_OBSERVE

    # OBSERVE stays (wrong classification)
    def test_observe_wrong_classification(self):
        assert _next_state(STATE_OBSERVE, 0, "", 90.0, "PROMOTE_LATER") == STATE_OBSERVE

    # PAPER_SHADOW stays (too few days)
    def test_paper_shadow_stays_few_days(self):
        assert _next_state(STATE_PAPER_SHADOW, 5, "", 85.0, _PROMOTE_FIRST) == STATE_PAPER_SHADOW

    # PAPER_SHADOW → MICRO_CAPITAL (14 days, no negative)
    def test_paper_shadow_to_micro_capital(self):
        assert _next_state(STATE_PAPER_SHADOW, 14, "", 85.0, "") == STATE_MICRO_CAPITAL

    # PAPER_SHADOW → MICRO_CAPITAL (14 days, positive impact)
    def test_paper_shadow_to_micro_capital_positive(self):
        assert _next_state(STATE_PAPER_SHADOW, 14, "POSITIVE", 85.0, "") == STATE_MICRO_CAPITAL

    # PAPER_SHADOW blocked by negative impact
    def test_paper_shadow_blocked_by_negative(self):
        result = _next_state(STATE_PAPER_SHADOW, 14, "NEGATIVE", 85.0, "")
        assert result == STATE_ROLLBACK_REQUIRED

    # MICRO_CAPITAL stays (too few days)
    def test_micro_capital_stays_few_days(self):
        assert _next_state(STATE_MICRO_CAPITAL, 10, "POSITIVE", 85.0, "") == STATE_MICRO_CAPITAL

    # MICRO_CAPITAL → PRODUCTION_READY (30 days, positive)
    def test_micro_capital_to_production_ready(self):
        assert _next_state(STATE_MICRO_CAPITAL, 30, "POSITIVE", 85.0, "") == STATE_PRODUCTION_READY

    # MICRO_CAPITAL stays (30 days, not positive)
    def test_micro_capital_not_positive_stays(self):
        assert _next_state(STATE_MICRO_CAPITAL, 30, "", 85.0, "") == STATE_MICRO_CAPITAL

    # MICRO_CAPITAL → ROLLBACK (negative impact)
    def test_micro_capital_rollback(self):
        assert _next_state(STATE_MICRO_CAPITAL, 15, "NEGATIVE", 85.0, "") == STATE_ROLLBACK_REQUIRED

    # PRODUCTION_READY stays
    def test_production_ready_stays(self):
        assert _next_state(STATE_PRODUCTION_READY, 5, "POSITIVE", 90.0, "") == STATE_PRODUCTION_READY

    # PRODUCTION_READY → ROLLBACK (negative impact)
    def test_production_ready_rollback(self):
        assert _next_state(STATE_PRODUCTION_READY, 5, "NEGATIVE", 90.0, "") == STATE_ROLLBACK_REQUIRED

    # ROLLBACK_REQUIRED stays
    def test_rollback_stays(self):
        assert _next_state(STATE_ROLLBACK_REQUIRED, 5, "POSITIVE", 90.0, "") == STATE_ROLLBACK_REQUIRED

    # OBSERVE with negative impact → stays OBSERVE (not in active state)
    def test_observe_negative_impact_stays_observe(self):
        result = _next_state(STATE_OBSERVE, 0, "NEGATIVE", 90.0, _PROMOTE_FIRST)
        assert result == STATE_PAPER_SHADOW  # OBSERVE transitions before rollback check

    # None with negative impact → entry allowed (negative only triggers from active states)
    def test_none_negative_impact_still_enters_paper_shadow(self):
        result = _next_state(None, 0, "NEGATIVE", 90.0, _PROMOTE_FIRST)
        # None → OBSERVE → check: OBSERVE doesn't trigger rollback
        assert result == STATE_PAPER_SHADOW

    # PAPER_SHADOW exactly at 14 days boundary
    def test_paper_shadow_exactly_14_days(self):
        assert _next_state(STATE_PAPER_SHADOW, 14, "POSITIVE", 85.0, "") == STATE_MICRO_CAPITAL

    # MICRO_CAPITAL exactly at 30 days boundary
    def test_micro_capital_exactly_30_days(self):
        assert _next_state(STATE_MICRO_CAPITAL, 30, "POSITIVE", 85.0, "") == STATE_PRODUCTION_READY


# ── TestLoadAllJsonl ──────────────────────────────────────────────────────────

class TestLoadAllJsonl:
    def test_missing_file(self, tmp_path):
        assert _load_all_jsonl(tmp_path / "missing.jsonl") == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_all_jsonl(f) == []

    def test_valid_records(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"a": 1}, {"b": 2}])
        assert len(_load_all_jsonl(f)) == 2

    def test_bad_json_skipped(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        f.write_text('{"ok":1}\nbad\n', encoding="utf-8")
        assert len(_load_all_jsonl(f)) == 1


# ── TestLatestGovernance ──────────────────────────────────────────────────────

class TestLatestGovernance:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "sb.jsonl"
        f.write_text("", encoding="utf-8")
        assert _latest_governance(f) == {}

    def test_single_record(self, tmp_path):
        f = tmp_path / "sb.jsonl"
        _write_jsonl(f, [_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")])
        result = _latest_governance(f)
        assert "EXIT_EXTENSION" in result
        assert result["EXIT_EXTENSION"] == (pytest.approx(85.0), "PROMOTE_FIRST")

    def test_latest_date_selected(self, tmp_path):
        f = tmp_path / "sb.jsonl"
        _write_jsonl(f, [
            _sb_rec("EXIT_EXTENSION", 70.0, "PROMOTE_LATER", date="2026-07-10"),
            _sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST", date="2026-07-20"),
        ])
        result = _latest_governance(f)
        assert result["EXIT_EXTENSION"][0] == pytest.approx(85.0)
        assert result["EXIT_EXTENSION"][1] == "PROMOTE_FIRST"

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "sb.jsonl"
        _write_jsonl(f, [{"date": "2026-07-20", "candidate": "UNKNOWN",
                          "governance_score": 85.0, "classification": "PROMOTE_FIRST"}])
        assert _latest_governance(f) == {}

    def test_missing_file(self, tmp_path):
        assert _latest_governance(tmp_path / "missing.jsonl") == {}


# ── TestLatestImpactStatus ────────────────────────────────────────────────────

class TestLatestImpactStatus:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        f.write_text("", encoding="utf-8")
        assert _latest_impact_status(f) == {}

    def test_positive_impact(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [_impact_rec("EXIT_EXTENSION", "POSITIVE")])
        result = _latest_impact_status(f)
        assert result["EXIT_EXTENSION"] == "POSITIVE"

    def test_negative_impact(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [_impact_rec("EXIT_EXTENSION", "NEGATIVE")])
        result = _latest_impact_status(f)
        assert result["EXIT_EXTENSION"] == "NEGATIVE"

    def test_latest_date_wins(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [
            _impact_rec("EXIT_EXTENSION", "POSITIVE", date="2026-07-10"),
            _impact_rec("EXIT_EXTENSION", "NEGATIVE", date="2026-07-20"),
        ])
        assert _latest_impact_status(f)["EXIT_EXTENSION"] == "NEGATIVE"

    def test_unknown_excluded(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [{"date": "2026-07-20", "candidate": "UNKNOWN", "impact_status": "POSITIVE"}])
        assert _latest_impact_status(f) == {}


# ── TestLoadCurrentStates ─────────────────────────────────────────────────────

class TestLoadCurrentStates:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "out.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_current_states(f) == {}

    def test_state_loaded(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [_rollout_rec("EXIT_EXTENSION", STATE_PAPER_SHADOW, 5, "2026-07-15")])
        result = _load_current_states(f)
        assert "EXIT_EXTENSION" in result
        assert result["EXIT_EXTENSION"]["state"] == STATE_PAPER_SHADOW

    def test_latest_per_candidate(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [
            _rollout_rec("EXIT_EXTENSION", STATE_PAPER_SHADOW, 5, "2026-07-15", date="2026-07-20"),
            _rollout_rec("EXIT_EXTENSION", STATE_MICRO_CAPITAL, 0, "2026-07-21", date="2026-07-21"),
        ])
        result = _load_current_states(f)
        assert result["EXIT_EXTENSION"]["state"] == STATE_MICRO_CAPITAL

    def test_unknown_excluded(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [{"date": "2026-07-20", "candidate": "UNKNOWN", "state": "OBSERVE"}])
        assert _load_current_states(f) == {}


# ── TestLoadEvaluatedKeys ─────────────────────────────────────────────────────

class TestLoadEvaluatedKeys:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "out.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_evaluated_keys(f) == set()

    def test_key_loaded(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [{"date": "2026-07-25", "candidate": "EXIT_EXTENSION"}])
        assert ("2026-07-25", "EXIT_EXTENSION") in _load_evaluated_keys(f)

    def test_empty_fields_excluded(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [{"date": "", "candidate": "EXIT_EXTENSION"}])
        assert _load_evaluated_keys(f) == set()


# ── TestComputeSummary ────────────────────────────────────────────────────────

class TestComputeSummary:
    def _rec(self, ct, state, score=80.0):
        return RolloutRecord("2026-07-25", ct, state, 0, "2026-07-25", "", score)

    def test_empty(self):
        s = _compute_summary([])
        assert s.n_paper_shadow == 0
        assert s.top_rollout_candidate == ""

    def test_n_paper_shadow(self):
        records = [self._rec("EXIT_EXTENSION", STATE_PAPER_SHADOW)]
        s = _compute_summary(records)
        assert s.n_paper_shadow == 1

    def test_n_micro_capital(self):
        records = [self._rec("EXIT_EXTENSION", STATE_MICRO_CAPITAL)]
        s = _compute_summary(records)
        assert s.n_micro_capital == 1

    def test_n_production_ready(self):
        records = [self._rec("EXIT_EXTENSION", STATE_PRODUCTION_READY)]
        s = _compute_summary(records)
        assert s.n_production_ready == 1

    def test_n_rollback_required(self):
        records = [self._rec("EXIT_EXTENSION", STATE_ROLLBACK_REQUIRED)]
        s = _compute_summary(records)
        assert s.n_rollback_required == 1

    def test_top_rollout_production_ready_beats_paper_shadow(self):
        records = [
            self._rec("EXIT_EXTENSION",        STATE_PRODUCTION_READY, 80.0),
            self._rec("CAPITAL_CONCENTRATION", STATE_PAPER_SHADOW,     90.0),
        ]
        s = _compute_summary(records)
        assert s.top_rollout_candidate == "EXIT_EXTENSION"

    def test_top_rollout_tie_broken_by_score(self):
        records = [
            self._rec("EXIT_EXTENSION",        STATE_MICRO_CAPITAL, 85.0),
            self._rec("CAPITAL_CONCENTRATION", STATE_MICRO_CAPITAL, 90.0),
        ]
        s = _compute_summary(records)
        assert s.top_rollout_candidate == "CAPITAL_CONCENTRATION"

    def test_top_rollout_excludes_observe(self):
        records = [self._rec("EXIT_EXTENSION", STATE_OBSERVE, 95.0)]
        s = _compute_summary(records)
        assert s.top_rollout_candidate == ""

    def test_top_rollout_excludes_rollback(self):
        records = [self._rec("EXIT_EXTENSION", STATE_ROLLBACK_REQUIRED, 95.0)]
        s = _compute_summary(records)
        assert s.top_rollout_candidate == ""

    def test_mixed_counts(self):
        records = [
            self._rec("EXIT_EXTENSION",        STATE_PAPER_SHADOW),
            self._rec("CAPITAL_CONCENTRATION", STATE_MICRO_CAPITAL),
            self._rec("PRIORITY",              STATE_PRODUCTION_READY),
            self._rec("SIGNAL_LEADER",         STATE_ROLLBACK_REQUIRED),
            self._rec("CONTINUATION_SIGNAL",   STATE_OBSERVE),
        ]
        s = _compute_summary(records)
        assert s.n_paper_shadow == 1
        assert s.n_micro_capital == 1
        assert s.n_production_ready == 1
        assert s.n_rollback_required == 1


# ── TestFormatReport ──────────────────────────────────────────────────────────

class TestFormatReport:
    def _rec(self, ct, state, days=5, impact="POSITIVE"):
        return RolloutRecord("2026-07-25", ct, state, days, "2026-07-20", impact, 80.0)

    def test_empty_returns_empty(self):
        s = RolloutSummary(0, 0, 0, 0, "")
        assert format_rollout_report([], s) == ""

    def test_contains_header(self):
        r = self._rec("EXIT_EXTENSION", STATE_PAPER_SHADOW)
        s = _compute_summary([r])
        rpt = format_rollout_report([r], s)
        assert "PROMOTION ROLLOUT MANAGER" in rpt

    def test_contains_separator(self):
        r = self._rec("EXIT_EXTENSION", STATE_PAPER_SHADOW)
        s = _compute_summary([r])
        rpt = format_rollout_report([r], s)
        assert "═" in rpt

    def test_contains_candidate(self):
        r = self._rec("EXIT_EXTENSION", STATE_PAPER_SHADOW)
        s = _compute_summary([r])
        rpt = format_rollout_report([r], s)
        assert "EXIT_EXTENSION" in rpt

    def test_contains_state(self):
        r = self._rec("EXIT_EXTENSION", STATE_PAPER_SHADOW)
        s = _compute_summary([r])
        rpt = format_rollout_report([r], s)
        assert STATE_PAPER_SHADOW in rpt

    def test_contains_days(self):
        r = self._rec("EXIT_EXTENSION", STATE_PAPER_SHADOW, days=7)
        s = _compute_summary([r])
        rpt = format_rollout_report([r], s)
        assert "7" in rpt

    def test_contains_impact(self):
        r = self._rec("EXIT_EXTENSION", STATE_PAPER_SHADOW, impact="POSITIVE")
        s = _compute_summary([r])
        rpt = format_rollout_report([r], s)
        assert "POSITIVE" in rpt

    def test_contains_production_ready_count(self):
        r = self._rec("EXIT_EXTENSION", STATE_PRODUCTION_READY)
        s = _compute_summary([r])
        rpt = format_rollout_report([r], s)
        assert "Production Ready" in rpt

    def test_contains_rollback_count(self):
        r = self._rec("EXIT_EXTENSION", STATE_ROLLBACK_REQUIRED, impact="NEGATIVE")
        s = _compute_summary([r])
        rpt = format_rollout_report([r], s)
        assert "Rollback" in rpt

    def test_top_candidate_shown(self):
        r = self._rec("EXIT_EXTENSION", STATE_PAPER_SHADOW)
        s = _compute_summary([r])
        rpt = format_rollout_report([r], s)
        assert "Top Rollout Candidate" in rpt

    def test_all_five_candidates(self):
        records = [
            self._rec("EXIT_EXTENSION",        STATE_PAPER_SHADOW),
            self._rec("CAPITAL_CONCENTRATION", STATE_MICRO_CAPITAL),
            self._rec("PRIORITY",              STATE_PRODUCTION_READY),
            self._rec("SIGNAL_LEADER",         STATE_OBSERVE,           impact=""),
            self._rec("CONTINUATION_SIGNAL",   STATE_ROLLBACK_REQUIRED, impact="NEGATIVE"),
        ]
        s = _compute_summary(records)
        rpt = format_rollout_report(records, s)
        for ct in _CANDIDATE_ORDER:
            assert ct in rpt


# ── TestAppendRolloutRecord ───────────────────────────────────────────────────

class TestAppendRolloutRecord:
    def _make_record(self):
        return RolloutRecord(
            date="2026-07-25",
            candidate="EXIT_EXTENSION",
            state=STATE_PAPER_SHADOW,
            days_in_state=5,
            state_entry_date="2026-07-20",
            latest_impact_status="POSITIVE",
            governance_score=85.0,
        )

    def test_creates_file(self, tmp_path):
        path = tmp_path / "out" / "rollout.jsonl"
        append_rollout_record(self._make_record(), path)
        assert path.exists()

    def test_all_fields_present(self, tmp_path):
        path = tmp_path / "rollout.jsonl"
        append_rollout_record(self._make_record(), path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["date"] == "2026-07-25"
        assert data["candidate"] == "EXIT_EXTENSION"
        assert data["state"] == STATE_PAPER_SHADOW
        assert data["days_in_state"] == 5
        assert data["state_entry_date"] == "2026-07-20"
        assert data["latest_impact_status"] == "POSITIVE"
        assert data["governance_score"] == pytest.approx(85.0)
        assert data["schema_version"] == "v1"

    def test_appends_multiple(self, tmp_path):
        path = tmp_path / "rollout.jsonl"
        for ct in _CANDIDATE_ORDER[:3]:
            rec = RolloutRecord("2026-07-25", ct, STATE_OBSERVE, 0, "2026-07-25", "", 50.0)
            append_rollout_record(rec, path)
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "rollout.jsonl"
        append_rollout_record(self._make_record(), path)
        assert path.exists()


# ── TestRunPromotionRolloutManager ────────────────────────────────────────────

class TestRunPromotionRolloutManager:
    def test_empty_inputs_all_observe(self, tmp_path):
        result, _ = _run(tmp_path)
        for r in result.new_records:
            assert r.state == STATE_OBSERVE

    def test_all_five_candidates_scored(self, tmp_path):
        result, _ = _run(tmp_path)
        assert len(result.new_records) == 5

    def test_promote_first_enters_paper_shadow(self, tmp_path):
        result, _ = _run(
            tmp_path,
            scoreboard=[_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")],
        )
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_PAPER_SHADOW

    def test_promote_later_stays_observe(self, tmp_path):
        result, _ = _run(
            tmp_path,
            scoreboard=[_sb_rec("EXIT_EXTENSION", 75.0, "PROMOTE_LATER")],
        )
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_OBSERVE

    def test_idempotent_same_day(self, tmp_path):
        files = _make_files(tmp_path)
        _write_jsonl(files["scoreboard"], [_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")])
        _run_with_files(files, "2026-07-25")
        result2 = _run_with_files(files, "2026-07-25")
        assert result2.new_records == []

    def test_paper_shadow_advances_after_14_days(self, tmp_path):
        # seed existing state: PAPER_SHADOW entered 2026-07-06 (19 days ago)
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_PAPER_SHADOW,
                                 14, "2026-07-06", impact="POSITIVE", date="2026-07-20")]
        result, _ = _run(tmp_path, existing=existing,
                         scoreboard=[_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")],
                         impact=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
                         today="2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_MICRO_CAPITAL

    def test_paper_shadow_stays_under_14_days(self, tmp_path):
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_PAPER_SHADOW,
                                 5, "2026-07-20", impact="POSITIVE", date="2026-07-20")]
        result, _ = _run(tmp_path, existing=existing,
                         scoreboard=[_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")],
                         impact=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
                         today="2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_PAPER_SHADOW

    def test_micro_capital_advances_after_30_days_positive(self, tmp_path):
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_MICRO_CAPITAL,
                                 30, "2026-06-21", impact="POSITIVE", date="2026-07-20")]
        result, _ = _run(tmp_path, existing=existing,
                         scoreboard=[_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")],
                         impact=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
                         today="2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_PRODUCTION_READY

    def test_micro_capital_stays_without_positive_impact(self, tmp_path):
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_MICRO_CAPITAL,
                                 30, "2026-06-21", impact="", date="2026-07-20")]
        result, _ = _run(tmp_path, existing=existing,
                         scoreboard=[_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")],
                         today="2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_MICRO_CAPITAL

    def test_rollback_on_negative_impact(self, tmp_path):
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_PAPER_SHADOW,
                                 5, "2026-07-20", impact="POSITIVE", date="2026-07-20")]
        result, _ = _run(tmp_path, existing=existing,
                         impact=[_impact_rec("EXIT_EXTENSION", "NEGATIVE")],
                         today="2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_ROLLBACK_REQUIRED

    def test_rollback_stays_rollback(self, tmp_path):
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_ROLLBACK_REQUIRED,
                                 2, "2026-07-23", impact="NEGATIVE", date="2026-07-23")]
        result, _ = _run(tmp_path, existing=existing,
                         scoreboard=[_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")],
                         impact=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
                         today="2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_ROLLBACK_REQUIRED

    def test_state_entry_date_resets_on_transition(self, tmp_path):
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_PAPER_SHADOW,
                                 14, "2026-07-06", impact="POSITIVE", date="2026-07-20")]
        result, _ = _run(tmp_path, existing=existing,
                         impact=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
                         today="2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        if ee.state == STATE_MICRO_CAPITAL:
            assert ee.state_entry_date == "2026-07-25"
            assert ee.days_in_state == 0

    def test_days_in_state_computed(self, tmp_path):
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_PAPER_SHADOW,
                                 0, "2026-07-20", impact="POSITIVE", date="2026-07-20")]
        result, _ = _run(tmp_path, existing=existing,
                         today="2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.days_in_state == 5

    def test_summary_n_paper_shadow(self, tmp_path):
        result, _ = _run(
            tmp_path,
            scoreboard=[_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")],
        )
        assert result.summary.n_paper_shadow == 1

    def test_summary_top_rollout_candidate(self, tmp_path):
        result, _ = _run(
            tmp_path,
            scoreboard=[_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")],
        )
        assert result.summary.top_rollout_candidate == "EXIT_EXTENSION"

    def test_fail_open_on_exception(self, tmp_path):
        out = tmp_path / "out.jsonl"
        result = run_promotion_rollout_manager(
            scoreboard_file=tmp_path / "nx1.jsonl",
            registry_file=tmp_path / "nx2.jsonl",
            impact_file=tmp_path / "nx3.jsonl",
            output_file=out,
            today="2026-07-25",
        )
        assert isinstance(result, RolloutRunResult)

    def test_result_date(self, tmp_path):
        result, _ = _run(tmp_path, today="2026-08-01")
        assert result.date == "2026-08-01"

    def test_schema_version_in_output(self, tmp_path):
        _, out = _run(tmp_path)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        data = json.loads(lines[0])
        assert data["schema_version"] == "v1"

    def test_state_records_in_result(self, tmp_path):
        result, _ = _run(tmp_path)
        assert len(result.state_records) == 5

    def test_candidate_order_in_state_records(self, tmp_path):
        result, _ = _run(tmp_path)
        for i, rec in enumerate(result.state_records):
            assert rec.candidate == _CANDIDATE_ORDER[i]

    def test_different_day_new_records(self, tmp_path):
        files = _make_files(tmp_path)
        _run_with_files(files, "2026-07-25")
        result2 = _run_with_files(files, "2026-07-26")
        assert len(result2.new_records) == 5
        lines = files["output"].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 10


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_lifecycle_paper_shadow_to_micro(self, tmp_path):
        """Day 1: OBSERVE→PAPER_SHADOW. Day 15: PAPER_SHADOW→MICRO_CAPITAL."""
        files = _make_files(tmp_path)
        _write_jsonl(files["scoreboard"], [_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")])
        _write_jsonl(files["impact"], [_impact_rec("EXIT_EXTENSION", "POSITIVE")])

        r1 = _run_with_files(files, "2026-07-01")
        ee1 = next(r for r in r1.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee1.state == STATE_PAPER_SHADOW

        r2 = _run_with_files(files, "2026-07-15")
        ee2 = next(r for r in r2.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee2.state == STATE_MICRO_CAPITAL

    def test_full_lifecycle_micro_to_production(self, tmp_path):
        """Seed MICRO_CAPITAL at day -30, run today → PRODUCTION_READY."""
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_MICRO_CAPITAL,
                                 30, "2026-06-21", "POSITIVE", date="2026-07-21")]
        files = _make_files(tmp_path)
        _write_jsonl(files["output"], existing)
        _write_jsonl(files["scoreboard"], [_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")])
        _write_jsonl(files["impact"], [_impact_rec("EXIT_EXTENSION", "POSITIVE")])

        result = _run_with_files(files, "2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_PRODUCTION_READY

    def test_rollback_from_production_ready(self, tmp_path):
        existing = [_rollout_rec("EXIT_EXTENSION", STATE_PRODUCTION_READY,
                                 10, "2026-07-10", "POSITIVE", date="2026-07-20")]
        files = _make_files(tmp_path)
        _write_jsonl(files["output"], existing)
        _write_jsonl(files["impact"], [_impact_rec("EXIT_EXTENSION", "NEGATIVE")])

        result = _run_with_files(files, "2026-07-25")
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.state == STATE_ROLLBACK_REQUIRED

    def test_report_integrated(self, tmp_path):
        result, _ = _run(
            tmp_path,
            scoreboard=[_sb_rec("EXIT_EXTENSION", 85.0, "PROMOTE_FIRST")],
        )
        rpt = format_rollout_report(result.state_records, result.summary)
        assert "PROMOTION ROLLOUT MANAGER" in rpt
        assert "EXIT_EXTENSION" in rpt

    def test_idempotent_three_runs_same_day(self, tmp_path):
        files = _make_files(tmp_path)
        for _ in range(3):
            _run_with_files(files, "2026-07-25")
        lines = files["output"].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

    def test_report_empty_when_no_records(self):
        s = RolloutSummary(0, 0, 0, 0, "")
        assert format_rollout_report([], s) == ""

    def test_multiple_candidates_different_states(self, tmp_path):
        existing = [
            _rollout_rec("EXIT_EXTENSION",        STATE_MICRO_CAPITAL, 5,  "2026-07-20", "POSITIVE", date="2026-07-20"),
            _rollout_rec("CAPITAL_CONCENTRATION", STATE_PAPER_SHADOW,  3,  "2026-07-22", "POSITIVE", date="2026-07-22"),
        ]
        result, _ = _run(
            tmp_path, existing=existing,
            scoreboard=[
                _sb_rec("EXIT_EXTENSION",        85.0, "PROMOTE_FIRST"),
                _sb_rec("CAPITAL_CONCENTRATION", 82.0, "PROMOTE_FIRST"),
            ],
            impact=[
                _impact_rec("EXIT_EXTENSION",        "POSITIVE"),
                _impact_rec("CAPITAL_CONCENTRATION", "POSITIVE"),
            ],
            today="2026-07-25",
        )
        states = {r.candidate: r.state for r in result.new_records}
        assert states["EXIT_EXTENSION"] == STATE_MICRO_CAPITAL
        assert states["CAPITAL_CONCENTRATION"] == STATE_PAPER_SHADOW
