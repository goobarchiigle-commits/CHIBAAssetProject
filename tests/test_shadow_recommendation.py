"""
test_shadow_recommendation.py — Shadow Recommendation Engine

86 tests across 11 classes:
  TestSafeHelpers / TestLoadLatestRecord / TestMapStatus
  TestComputeShadowPriority / TestComputeShadowDuration / TestBuildRecommendation
  TestRunShadowRecommendation / TestAppendRecommendationRecord
  TestFormatRecommendationReport / TestComputeWeeklyGovernanceSummary / TestIntegration
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from src.analytics.shadow_recommendation import (
    CONTINUE_OBSERVE,
    DURATION_FAST,
    DURATION_MEDIUM,
    DURATION_SLOW,
    EP_INSUFFICIENT,
    EP_OBSERVE,
    EP_PROMOTABLE,
    SCORE_FAST_THRESHOLD,
    SCORE_MEDIUM_THRESHOLD,
    SHADOW_READY,
    WAIT_DATA,
    ShadowRecommendation,
    ShadowRecommendationOutput,
    WeeklyGovernanceSummary,
    _build_recommendation,
    _compute_shadow_duration,
    _compute_shadow_priority,
    _load_all_jsonl,
    _load_latest_record,
    _map_shadow_status,
    _safe_int,
    _safe_str,
    append_recommendation_record,
    compute_weekly_governance_summary,
    format_recommendation_report,
    run_shadow_recommendation,
)


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _ep_candidate(ct: str, status: str, score: int, src: str = "HDC") -> dict:
    return {
        "candidate_type": ct,
        "source_module": src,
        "status": status,
        "evidence_score": score,
        "supporting_kpis": {},
        "rules": [],
        "rules_met": [],
    }


def _ep_record(candidates: list[dict], date: str = "2026-05-30") -> dict:
    n_p = sum(1 for c in candidates if c["status"] == EP_PROMOTABLE)
    n_o = sum(1 for c in candidates if c["status"] == EP_OBSERVE)
    n_i = sum(1 for c in candidates if c["status"] == EP_INSUFFICIENT)
    top = next((c["candidate_type"] for c in candidates if c["status"] == EP_PROMOTABLE), "")
    return {
        "date": date,
        "n_promotable": n_p,
        "n_observe": n_o,
        "n_insufficient": n_i,
        "top_candidate": top,
        "candidates": candidates,
        "schema_version": "v1",
    }


def _all_promotable() -> list[dict]:
    return [
        _ep_candidate("EXIT_EXTENSION", EP_PROMOTABLE, 78, "HDC"),
        _ep_candidate("CAPITAL_CONCENTRATION", EP_PROMOTABLE, 65, "CCS"),
        _ep_candidate("PRIORITY", EP_PROMOTABLE, 82, "PCI"),
        _ep_candidate("SIGNAL_LEADER", EP_PROMOTABLE, 55, "SA"),
        _ep_candidate("CONTINUATION_SIGNAL", EP_PROMOTABLE, 70, "FCO"),
    ]


def _mixed_candidates() -> list[dict]:
    return [
        _ep_candidate("EXIT_EXTENSION", EP_PROMOTABLE, 78, "HDC"),
        _ep_candidate("CAPITAL_CONCENTRATION", EP_OBSERVE, 44, "CCS"),
        _ep_candidate("PRIORITY", EP_INSUFFICIENT, 0, "PCI"),
        _ep_candidate("SIGNAL_LEADER", EP_OBSERVE, 51, "SA"),
        _ep_candidate("CONTINUATION_SIGNAL", EP_PROMOTABLE, 62, "FCO"),
    ]


# ─── TestSafeHelpers ──────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_int_valid(self):
        assert _safe_int(42) == 42

    def test_safe_int_string(self):
        assert _safe_int("25") == 25

    def test_safe_int_none_fallback(self):
        assert _safe_int(None) == 0

    def test_safe_int_custom_fallback(self):
        assert _safe_int(None, 99) == 99

    def test_safe_str_valid(self):
        assert _safe_str("hello") == "hello"

    def test_safe_str_none_fallback(self):
        assert _safe_str(None) == ""

    def test_safe_str_int_converts(self):
        assert _safe_str(123) == "123"


# ─── TestLoadLatestRecord ─────────────────────────────────────────────────────

class TestLoadLatestRecord:
    def test_missing_returns_empty(self, tmp_path):
        result = _load_latest_record(tmp_path / "missing.jsonl")
        assert result == {}

    def test_returns_last_line(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _write_jsonl(p, [{"date": "2026-05-29"}, {"date": "2026-05-30"}])
        result = _load_latest_record(p)
        assert result["date"] == "2026-05-30"

    def test_skips_invalid_json(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        p.write_text('{"date": "2026-05-29"}\nNOT_JSON\n{"date": "2026-05-30"}\n', encoding="utf-8")
        result = _load_latest_record(p)
        assert result["date"] == "2026-05-30"

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        p.write_text("", encoding="utf-8")
        assert _load_latest_record(p) == {}

    def test_single_record(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _write_jsonl(p, [{"key": "only"}])
        assert _load_latest_record(p)["key"] == "only"


# ─── TestMapStatus ────────────────────────────────────────────────────────────

class TestMapStatus:
    def test_promotable_maps_shadow_ready(self):
        assert _map_shadow_status(EP_PROMOTABLE) == SHADOW_READY

    def test_observe_maps_continue_observe(self):
        assert _map_shadow_status(EP_OBSERVE) == CONTINUE_OBSERVE

    def test_insufficient_maps_wait_data(self):
        assert _map_shadow_status(EP_INSUFFICIENT) == WAIT_DATA

    def test_unknown_status_maps_wait_data(self):
        assert _map_shadow_status("UNKNOWN") == WAIT_DATA


# ─── TestComputeShadowPriority ────────────────────────────────────────────────

class TestComputeShadowPriority:
    def test_shadow_ready_preserves_score(self):
        assert _compute_shadow_priority(78, SHADOW_READY) == 78

    def test_shadow_ready_clamps_at_100(self):
        assert _compute_shadow_priority(150, SHADOW_READY) == 100

    def test_shadow_ready_clamps_at_0(self):
        assert _compute_shadow_priority(-10, SHADOW_READY) == 0

    def test_continue_observe_halves_score(self):
        assert _compute_shadow_priority(80, CONTINUE_OBSERVE) == 40

    def test_continue_observe_floors_at_0(self):
        assert _compute_shadow_priority(0, CONTINUE_OBSERVE) == 0

    def test_continue_observe_odd_score(self):
        # int(51 * 0.5) = 25
        assert _compute_shadow_priority(51, CONTINUE_OBSERVE) == 25

    def test_wait_data_always_0(self):
        assert _compute_shadow_priority(100, WAIT_DATA) == 0

    def test_wait_data_zero_score(self):
        assert _compute_shadow_priority(0, WAIT_DATA) == 0


# ─── TestComputeShadowDuration ────────────────────────────────────────────────

class TestComputeShadowDuration:
    def test_score_80_returns_fast(self):
        assert _compute_shadow_duration(SCORE_FAST_THRESHOLD) == DURATION_FAST

    def test_score_above_80_returns_fast(self):
        assert _compute_shadow_duration(95) == DURATION_FAST

    def test_score_100_returns_fast(self):
        assert _compute_shadow_duration(100) == DURATION_FAST

    def test_score_60_returns_medium(self):
        assert _compute_shadow_duration(SCORE_MEDIUM_THRESHOLD) == DURATION_MEDIUM

    def test_score_79_returns_medium(self):
        assert _compute_shadow_duration(79) == DURATION_MEDIUM

    def test_score_59_returns_slow(self):
        assert _compute_shadow_duration(59) == DURATION_SLOW

    def test_score_0_returns_slow(self):
        assert _compute_shadow_duration(0) == DURATION_SLOW

    def test_duration_values_correct(self):
        assert DURATION_FAST < DURATION_MEDIUM < DURATION_SLOW


# ─── TestBuildRecommendation ──────────────────────────────────────────────────

class TestBuildRecommendation:
    def test_promotable_becomes_shadow_ready(self):
        c = _ep_candidate("EXIT_EXTENSION", EP_PROMOTABLE, 78)
        r = _build_recommendation(c)
        assert r.shadow_status == SHADOW_READY

    def test_observe_becomes_continue_observe(self):
        c = _ep_candidate("PRIORITY", EP_OBSERVE, 45)
        r = _build_recommendation(c)
        assert r.shadow_status == CONTINUE_OBSERVE

    def test_insufficient_becomes_wait_data(self):
        c = _ep_candidate("SIGNAL_LEADER", EP_INSUFFICIENT, 0)
        r = _build_recommendation(c)
        assert r.shadow_status == WAIT_DATA

    def test_evidence_score_preserved(self):
        c = _ep_candidate("EXIT_EXTENSION", EP_PROMOTABLE, 82)
        r = _build_recommendation(c)
        assert r.evidence_score == 82

    def test_shadow_priority_score_for_ready(self):
        c = _ep_candidate("EXIT_EXTENSION", EP_PROMOTABLE, 82)
        r = _build_recommendation(c)
        assert r.shadow_priority_score == 82

    def test_shadow_priority_score_for_observe(self):
        c = _ep_candidate("PRIORITY", EP_OBSERVE, 60)
        r = _build_recommendation(c)
        assert r.shadow_priority_score == 30  # 60 × 0.5

    def test_shadow_duration_fast(self):
        c = _ep_candidate("EXIT_EXTENSION", EP_PROMOTABLE, 85)
        r = _build_recommendation(c)
        assert r.shadow_duration_days == DURATION_FAST

    def test_shadow_duration_slow_for_wait(self):
        c = _ep_candidate("SIGNAL_LEADER", EP_INSUFFICIENT, 0)
        r = _build_recommendation(c)
        assert r.shadow_duration_days == DURATION_SLOW

    def test_candidate_type_preserved(self):
        c = _ep_candidate("CAPITAL_CONCENTRATION", EP_PROMOTABLE, 70, "CCS")
        r = _build_recommendation(c)
        assert r.candidate_type == "CAPITAL_CONCENTRATION"

    def test_source_module_preserved(self):
        c = _ep_candidate("CAPITAL_CONCENTRATION", EP_PROMOTABLE, 70, "CCS")
        r = _build_recommendation(c)
        assert r.source_module == "CCS"

    def test_empty_candidate_does_not_raise(self):
        r = _build_recommendation({})
        assert isinstance(r, ShadowRecommendation)
        assert r.shadow_status == WAIT_DATA


# ─── TestRunShadowRecommendation ──────────────────────────────────────────────

class TestRunShadowRecommendation:
    def _run(self, tmp_path, candidates=None, date="2026-05-30"):
        ep_f  = tmp_path / "ep.jsonl"
        out_f = tmp_path / "sr.jsonl"
        if candidates is not None:
            _write_jsonl(ep_f, [_ep_record(candidates, date)])
        return run_shadow_recommendation(ep_file=ep_f, output_file=out_f, today=date)

    def test_returns_output_type(self, tmp_path):
        result = self._run(tmp_path, _mixed_candidates())
        assert isinstance(result, ShadowRecommendationOutput)

    def test_date_set(self, tmp_path):
        result = self._run(tmp_path, _mixed_candidates(), date="2026-05-30")
        assert result.date == "2026-05-30"

    def test_counts_correct(self, tmp_path):
        result = self._run(tmp_path, _mixed_candidates())
        # 2 PROMOTABLE → SHADOW_READY; 2 OBSERVE → CONTINUE_OBSERVE; 1 INSUFFICIENT → WAIT_DATA
        assert result.n_shadow_ready == 2
        assert result.n_continue_observe == 2
        assert result.n_wait_data == 1

    def test_counts_sum_to_candidates(self, tmp_path):
        result = self._run(tmp_path, _mixed_candidates())
        total = result.n_shadow_ready + result.n_continue_observe + result.n_wait_data
        assert total == len(result.recommendations)

    def test_all_promotable(self, tmp_path):
        result = self._run(tmp_path, _all_promotable())
        assert result.n_shadow_ready == 5
        assert result.n_wait_data == 0

    def test_top_shadow_candidate_set(self, tmp_path):
        result = self._run(tmp_path, _all_promotable())
        assert result.top_shadow_candidate != ""

    def test_top_shadow_candidate_is_highest_score(self, tmp_path):
        candidates = [
            _ep_candidate("EXIT_EXTENSION", EP_PROMOTABLE, 78, "HDC"),
            _ep_candidate("PRIORITY", EP_PROMOTABLE, 90, "PCI"),
        ]
        result = self._run(tmp_path, candidates)
        assert result.top_shadow_candidate == "PRIORITY"

    def test_top_empty_when_no_shadow_ready(self, tmp_path):
        candidates = [_ep_candidate("EXIT_EXTENSION", EP_OBSERVE, 30, "HDC")]
        result = self._run(tmp_path, candidates)
        assert result.top_shadow_candidate == ""

    def test_empty_ep_file_fail_open(self, tmp_path):
        result = self._run(tmp_path)  # no ep file written
        assert isinstance(result, ShadowRecommendationOutput)
        assert result.n_shadow_ready == 0

    def test_jsonl_written(self, tmp_path):
        ep_f  = tmp_path / "ep.jsonl"
        out_f = tmp_path / "sr.jsonl"
        _write_jsonl(ep_f, [_ep_record(_mixed_candidates())])
        run_shadow_recommendation(ep_file=ep_f, output_file=out_f, today="2026-05-30")
        assert out_f.exists()

    def test_corrupt_ep_file_fail_open(self, tmp_path):
        ep_f  = tmp_path / "ep.jsonl"
        out_f = tmp_path / "sr.jsonl"
        ep_f.write_text("NOTJSON\n", encoding="utf-8")
        result = run_shadow_recommendation(ep_file=ep_f, output_file=out_f, today="2026-05-30")
        assert isinstance(result, ShadowRecommendationOutput)


# ─── TestAppendRecommendationRecord ──────────────────────────────────────────

class TestAppendRecommendationRecord:
    def _make_output(self) -> ShadowRecommendationOutput:
        return ShadowRecommendationOutput(
            date="2026-05-30",
            n_shadow_ready=1,
            n_continue_observe=2,
            n_wait_data=2,
            top_shadow_candidate="EXIT_EXTENSION",
            recommendations=[
                ShadowRecommendation(
                    candidate_type="EXIT_EXTENSION",
                    source_module="HDC",
                    evidence_status=EP_PROMOTABLE,
                    shadow_status=SHADOW_READY,
                    evidence_score=78,
                    shadow_priority_score=78,
                    shadow_duration_days=20,
                )
            ],
        )

    def test_file_created(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        append_recommendation_record(self._make_output(), p)
        assert p.exists()

    def test_valid_json(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        append_recommendation_record(self._make_output(), p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["date"] == "2026-05-30"

    def test_two_appends_two_lines(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        append_recommendation_record(self._make_output(), p)
        append_recommendation_record(self._make_output(), p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_recommendations_serialised(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        append_recommendation_record(self._make_output(), p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["recommendations"][0]["candidate_type"] == "EXIT_EXTENSION"
        assert data["recommendations"][0]["shadow_status"] == SHADOW_READY

    def test_schema_version_present(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        append_recommendation_record(self._make_output(), p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert "schema_version" in data

    def test_fail_open_bad_path(self):
        bad = Path("/nonexistent_dir/sr.jsonl")
        append_recommendation_record(self._make_output(), bad)  # must not raise


# ─── TestFormatRecommendationReport ──────────────────────────────────────────

class TestFormatRecommendationReport:
    def _make_output(self) -> ShadowRecommendationOutput:
        return ShadowRecommendationOutput(
            date="2026-05-30",
            n_shadow_ready=2,
            n_continue_observe=2,
            n_wait_data=1,
            top_shadow_candidate="EXIT_EXTENSION",
            recommendations=[
                ShadowRecommendation("EXIT_EXTENSION", "HDC", EP_PROMOTABLE, SHADOW_READY, 78, 78, 20),
                ShadowRecommendation("CAPITAL_CONCENTRATION", "CCS", EP_OBSERVE, CONTINUE_OBSERVE, 44, 22, 45),
                ShadowRecommendation("PRIORITY", "PCI", EP_INSUFFICIENT, WAIT_DATA, 0, 0, 45),
                ShadowRecommendation("SIGNAL_LEADER", "SA", EP_OBSERVE, CONTINUE_OBSERVE, 51, 25, 45),
                ShadowRecommendation("CONTINUATION_SIGNAL", "FCO", EP_PROMOTABLE, SHADOW_READY, 62, 62, 30),
            ],
        )

    def test_header_present(self):
        rpt = format_recommendation_report(self._make_output())
        assert "SHADOW RECOMMENDATIONS" in rpt

    def test_shadow_ready_label(self):
        rpt = format_recommendation_report(self._make_output())
        assert "SHADOW_READY" in rpt

    def test_continue_observe_label(self):
        rpt = format_recommendation_report(self._make_output())
        assert "CONTINUE_OBSERVE" in rpt

    def test_wait_data_label(self):
        rpt = format_recommendation_report(self._make_output())
        assert "WAIT_DATA" in rpt

    def test_exit_extension_shown(self):
        rpt = format_recommendation_report(self._make_output())
        assert "EXIT_EXTENSION" in rpt

    def test_capital_concentration_shown(self):
        rpt = format_recommendation_report(self._make_output())
        assert "CAPITAL_CONCENTRATION" in rpt

    def test_priority_shown(self):
        rpt = format_recommendation_report(self._make_output())
        assert "PRIORITY" in rpt

    def test_signal_leader_shown(self):
        rpt = format_recommendation_report(self._make_output())
        assert "SIGNAL_LEADER" in rpt

    def test_continuation_signal_shown(self):
        rpt = format_recommendation_report(self._make_output())
        assert "CONTINUATION_SIGNAL" in rpt

    def test_top_candidate_shown(self):
        rpt = format_recommendation_report(self._make_output())
        assert "EXIT_EXTENSION" in rpt

    def test_priority_score_shown(self):
        rpt = format_recommendation_report(self._make_output())
        assert "priority=" in rpt

    def test_shadow_days_shown(self):
        rpt = format_recommendation_report(self._make_output())
        assert "shadow=" in rpt

    def test_empty_recommendations_returns_empty(self):
        output = ShadowRecommendationOutput(
            date="2026-05-30", n_shadow_ready=0, n_continue_observe=0,
            n_wait_data=0, top_shadow_candidate="", recommendations=[],
        )
        assert format_recommendation_report(output) == ""

    def test_counts_in_report(self):
        rpt = format_recommendation_report(self._make_output())
        assert "Shadow Ready: 2" in rpt


# ─── TestComputeWeeklyGovernanceSummary ──────────────────────────────────────

class TestComputeWeeklyGovernanceSummary:
    def _make_sr_record(self, date: str, ready_types: list[str], ready_scores: list[int]) -> dict:
        recs = [
            {
                "candidate_type": ct,
                "shadow_status": SHADOW_READY,
                "shadow_priority_score": score,
            }
            for ct, score in zip(ready_types, ready_scores)
        ]
        return {
            "date": date,
            "n_shadow_ready": len(recs),
            "recommendations": recs,
            "schema_version": "v1",
        }

    def test_returns_summary_type(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        summary = compute_weekly_governance_summary(p, today="2026-05-30")
        assert isinstance(summary, WeeklyGovernanceSummary)

    def test_empty_file_zero_state(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        summary = compute_weekly_governance_summary(p, today="2026-05-30")
        assert summary.shadow_ready_count == 0
        assert summary.top_shadow_candidate == ""
        assert summary.avg_shadow_priority == 0.0

    def test_counts_within_lookback(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        _write_jsonl(p, [
            self._make_sr_record("2026-05-28", ["EXIT_EXTENSION"], [78]),
            self._make_sr_record("2026-05-29", ["PRIORITY"], [82]),
        ])
        summary = compute_weekly_governance_summary(p, today="2026-05-30", lookback_days=7)
        assert summary.shadow_ready_count == 2

    def test_excludes_old_records(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        _write_jsonl(p, [
            self._make_sr_record("2026-05-01", ["EXIT_EXTENSION"], [78]),  # old
            self._make_sr_record("2026-05-29", ["PRIORITY"], [82]),         # recent
        ])
        summary = compute_weekly_governance_summary(p, today="2026-05-30", lookback_days=7)
        assert summary.shadow_ready_count == 1

    def test_top_candidate_most_common(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        _write_jsonl(p, [
            self._make_sr_record("2026-05-28", ["EXIT_EXTENSION", "EXIT_EXTENSION"], [78, 80]),
            self._make_sr_record("2026-05-29", ["PRIORITY"], [82]),
        ])
        summary = compute_weekly_governance_summary(p, today="2026-05-30", lookback_days=7)
        assert summary.top_shadow_candidate == "EXIT_EXTENSION"

    def test_avg_priority_correct(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        _write_jsonl(p, [
            self._make_sr_record("2026-05-29", ["EXIT_EXTENSION", "PRIORITY"], [60, 80]),
        ])
        summary = compute_weekly_governance_summary(p, today="2026-05-30", lookback_days=7)
        assert summary.avg_shadow_priority == pytest.approx(70.0)

    def test_lookback_days_attribute(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        summary = compute_weekly_governance_summary(p, today="2026-05-30", lookback_days=14)
        assert summary.lookback_days == 14

    def test_missing_file_fail_open(self, tmp_path):
        p = tmp_path / "missing_sr.jsonl"
        summary = compute_weekly_governance_summary(p, today="2026-05-30")
        assert isinstance(summary, WeeklyGovernanceSummary)

    def test_no_shadow_ready_in_window_zero_state(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        # record with only CONTINUE_OBSERVE / WAIT_DATA
        rec = {
            "date": "2026-05-29",
            "n_shadow_ready": 0,
            "recommendations": [
                {"candidate_type": "EXIT_EXTENSION", "shadow_status": CONTINUE_OBSERVE, "shadow_priority_score": 30},
            ],
            "schema_version": "v1",
        }
        _write_jsonl(p, [rec])
        summary = compute_weekly_governance_summary(p, today="2026-05-30", lookback_days=7)
        assert summary.shadow_ready_count == 0
        assert summary.avg_shadow_priority == 0.0

    def test_multiple_days_accumulated(self, tmp_path):
        p = tmp_path / "sr.jsonl"
        records = [
            self._make_sr_record("2026-05-25", ["EXIT_EXTENSION"], [70]),
            self._make_sr_record("2026-05-26", ["EXIT_EXTENSION"], [75]),
            self._make_sr_record("2026-05-27", ["EXIT_EXTENSION"], [80]),
            self._make_sr_record("2026-05-28", ["PRIORITY"], [85]),
        ]
        _write_jsonl(p, records)
        summary = compute_weekly_governance_summary(p, today="2026-05-30", lookback_days=7)
        assert summary.shadow_ready_count == 4


# ─── TestIntegration ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_end_to_end_promotable(self, tmp_path):
        ep_f  = tmp_path / "ep.jsonl"
        sr_f  = tmp_path / "sr.jsonl"
        _write_jsonl(ep_f, [_ep_record(_all_promotable())])
        result = run_shadow_recommendation(ep_file=ep_f, output_file=sr_f, today="2026-05-30")
        assert result.n_shadow_ready == 5
        assert sr_f.exists()

    def test_report_matches_counts(self, tmp_path):
        ep_f  = tmp_path / "ep.jsonl"
        sr_f  = tmp_path / "sr.jsonl"
        _write_jsonl(ep_f, [_ep_record(_mixed_candidates())])
        result = run_shadow_recommendation(ep_file=ep_f, output_file=sr_f, today="2026-05-30")
        rpt = format_recommendation_report(result)
        assert "SHADOW_READY" in rpt
        assert "CONTINUE_OBSERVE" in rpt
        assert "WAIT_DATA" in rpt

    def test_weekly_summary_after_run(self, tmp_path):
        ep_f  = tmp_path / "ep.jsonl"
        sr_f  = tmp_path / "sr.jsonl"
        _write_jsonl(ep_f, [_ep_record(_all_promotable())])
        run_shadow_recommendation(ep_file=ep_f, output_file=sr_f, today="2026-05-30")
        summary = compute_weekly_governance_summary(sr_f, today="2026-05-30", lookback_days=1)
        assert summary.shadow_ready_count == 5

    def test_shadow_priority_scores_reasonable(self, tmp_path):
        ep_f  = tmp_path / "ep.jsonl"
        sr_f  = tmp_path / "sr.jsonl"
        _write_jsonl(ep_f, [_ep_record(_all_promotable())])
        result = run_shadow_recommendation(ep_file=ep_f, output_file=sr_f, today="2026-05-30")
        for r in result.recommendations:
            assert 0 <= r.shadow_priority_score <= 100

    def test_duration_days_valid_set(self, tmp_path):
        ep_f  = tmp_path / "ep.jsonl"
        sr_f  = tmp_path / "sr.jsonl"
        _write_jsonl(ep_f, [_ep_record(_mixed_candidates())])
        result = run_shadow_recommendation(ep_file=ep_f, output_file=sr_f, today="2026-05-30")
        valid_durations = {DURATION_FAST, DURATION_MEDIUM, DURATION_SLOW}
        for r in result.recommendations:
            assert r.shadow_duration_days in valid_durations
