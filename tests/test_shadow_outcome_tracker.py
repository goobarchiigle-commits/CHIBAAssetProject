"""
test_shadow_outcome_tracker.py — Shadow Outcome Tracker (Phase 6A-Lite)

Tests (100% coverage):
  TestSafeHelpers / TestDaysElapsed / TestExtractKpi
  TestComputeDelta / TestComputeSuccess / TestBuildEpIndex
  TestLoadEvaluatedKeys / TestComputeSummary / TestFormatReport
  TestAppendOutcomeRecord / TestRunShadowOutcomeTracker / TestIntegration
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.analytics.shadow_outcome_tracker import (
    SCHEMA_VERSION,
    SHADOW_READY,
    ShadowOutcomeRecord,
    ShadowOutcomeRunResult,
    ShadowOutcomeSummary,
    _KPI_FIELD,
    _build_ep_index,
    _compute_delta,
    _compute_success,
    _days_elapsed,
    _extract_kpi,
    _load_all_jsonl,
    _load_evaluated_keys,
    _compute_summary,
    _safe_float,
    _safe_int,
    append_outcome_record,
    format_outcome_report,
    run_shadow_outcome_tracker,
)


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _ep_candidate(ct: str, kpi_field: str, kpi_value: float) -> dict:
    return {
        "candidate_type": ct,
        "source_module": "HDC",
        "status": "PROMOTABLE",
        "evidence_score": 75,
        "supporting_kpis": {kpi_field: kpi_value},
        "rules": [],
        "rules_met": [],
    }


def _ep_record(date: str, candidates: list[dict]) -> dict:
    return {
        "date": date,
        "n_promotable": 1,
        "n_observe": 0,
        "n_insufficient": 0,
        "top_candidate": "",
        "candidates": candidates,
        "schema_version": "v1",
    }


def _sr_recommendation(ct: str, duration: int = 20) -> dict:
    return {
        "candidate_type": ct,
        "source_module": "HDC",
        "evidence_status": "PROMOTABLE",
        "shadow_status": SHADOW_READY,
        "evidence_score": 80,
        "shadow_priority_score": 80,
        "shadow_duration_days": duration,
    }


def _sr_record(date: str, recommendations: list[dict]) -> dict:
    return {
        "date": date,
        "n_shadow_ready": len([r for r in recommendations if r.get("shadow_status") == SHADOW_READY]),
        "n_continue_observe": 0,
        "n_wait_data": 0,
        "top_shadow_candidate": "",
        "recommendations": recommendations,
        "schema_version": "v1",
    }


def _outcome_record(date: str, candidate: str, shadow_start: str, duration: int,
                    before: float, after: float, delta: float, success: bool) -> dict:
    return {
        "date": date,
        "candidate": candidate,
        "shadow_start_date": shadow_start,
        "shadow_duration_days": duration,
        "before_kpi": before,
        "after_kpi": after,
        "delta": delta,
        "success": success,
        "schema_version": SCHEMA_VERSION,
    }


# ─── TestSafeHelpers ─────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_float_valid(self):
        assert _safe_float(1.5) == pytest.approx(1.5)

    def test_safe_float_string(self):
        assert _safe_float("0.3") == pytest.approx(0.3)

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) == 0.0

    def test_safe_float_invalid(self):
        assert _safe_float("bad") == 0.0

    def test_safe_float_fallback(self):
        assert _safe_float(None, fallback=99.0) == pytest.approx(99.0)

    def test_safe_int_valid(self):
        assert _safe_int(20) == 20

    def test_safe_int_string(self):
        assert _safe_int("30") == 30

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_invalid(self):
        assert _safe_int("bad") == 0

    def test_safe_int_fallback(self):
        assert _safe_int(None, fallback=5) == 5


# ─── TestDaysElapsed ─────────────────────────────────────────────────────────

class TestDaysElapsed:
    def test_same_day(self):
        assert _days_elapsed("2026-05-30", "2026-05-30") == 0

    def test_twenty_days(self):
        assert _days_elapsed("2026-05-10", "2026-05-30") == 20

    def test_negative(self):
        assert _days_elapsed("2026-06-01", "2026-05-30") == -2

    def test_invalid_start(self):
        assert _days_elapsed("not-a-date", "2026-05-30") == 0

    def test_invalid_end(self):
        assert _days_elapsed("2026-05-01", "bad") == 0

    def test_both_invalid(self):
        assert _days_elapsed("x", "y") == 0

    def test_cross_month(self):
        assert _days_elapsed("2026-04-30", "2026-05-30") == 30


# ─── TestExtractKpi ──────────────────────────────────────────────────────────

class TestExtractKpi:
    def _make_ep(self, ct: str, field: str, value: float) -> dict:
        return _ep_record("2026-05-30", [_ep_candidate(ct, field, value)])

    def test_exit_extension(self):
        ep = self._make_ep("EXIT_EXTENSION", "suppression_return_delta", 0.023)
        assert _extract_kpi(ep, "EXIT_EXTENSION") == pytest.approx(0.023)

    def test_capital_concentration(self):
        ep = self._make_ep("CAPITAL_CONCENTRATION", "mean_concentration_alpha", 0.030)
        assert _extract_kpi(ep, "CAPITAL_CONCENTRATION") == pytest.approx(0.030)

    def test_priority(self):
        ep = self._make_ep("PRIORITY", "priority_monotonicity", 0.57)
        assert _extract_kpi(ep, "PRIORITY") == pytest.approx(0.57)

    def test_signal_leader(self):
        ep = self._make_ep("SIGNAL_LEADER", "top_signal_ir", 0.014)
        assert _extract_kpi(ep, "SIGNAL_LEADER") == pytest.approx(0.014)

    def test_continuation_signal(self):
        ep = self._make_ep("CONTINUATION_SIGNAL", "top_signal_ir", 0.016)
        assert _extract_kpi(ep, "CONTINUATION_SIGNAL") == pytest.approx(0.016)

    def test_unknown_candidate(self):
        ep = self._make_ep("EXIT_EXTENSION", "suppression_return_delta", 0.01)
        assert _extract_kpi(ep, "UNKNOWN") is None

    def test_candidate_not_in_ep(self):
        ep = _ep_record("2026-05-30", [])
        assert _extract_kpi(ep, "EXIT_EXTENSION") is None

    def test_missing_kpi_field(self):
        ep = _ep_record("2026-05-30", [{
            "candidate_type": "EXIT_EXTENSION",
            "supporting_kpis": {},
        }])
        assert _extract_kpi(ep, "EXIT_EXTENSION") is None

    def test_nan_value(self):
        ep = _ep_record("2026-05-30", [{
            "candidate_type": "PRIORITY",
            "supporting_kpis": {"priority_monotonicity": float("nan")},
        }])
        assert _extract_kpi(ep, "PRIORITY") is None

    def test_invalid_value(self):
        ep = _ep_record("2026-05-30", [{
            "candidate_type": "PRIORITY",
            "supporting_kpis": {"priority_monotonicity": "not-a-number"},
        }])
        assert _extract_kpi(ep, "PRIORITY") is None

    def test_empty_ep(self):
        assert _extract_kpi({}, "PRIORITY") is None


# ─── TestComputeDelta ─────────────────────────────────────────────────────────

class TestComputeDelta:
    def test_normal_positive(self):
        assert _compute_delta("EXIT_EXTENSION", 0.008, 0.023) == pytest.approx(0.015)

    def test_normal_negative(self):
        assert _compute_delta("PRIORITY", 0.57, 0.31) == pytest.approx(-0.26)

    def test_signal_leader_abs(self):
        # abs(0.014) - abs(0.006) = 0.008
        assert _compute_delta("SIGNAL_LEADER", 0.006, 0.014) == pytest.approx(0.008)

    def test_signal_leader_negative_to_positive(self):
        # abs(0.015) - abs(-0.010) = 0.005 (improvement in absolute IR)
        assert _compute_delta("SIGNAL_LEADER", -0.010, 0.015) == pytest.approx(0.005)

    def test_signal_leader_deterioration(self):
        # abs(0.005) - abs(-0.020) = -0.015 (deterioration)
        assert _compute_delta("SIGNAL_LEADER", -0.020, 0.005) == pytest.approx(-0.015)

    def test_capital_concentration(self):
        assert _compute_delta("CAPITAL_CONCENTRATION", 0.011, 0.030) == pytest.approx(0.019)

    def test_continuation_signal(self):
        assert _compute_delta("CONTINUATION_SIGNAL", 0.008, 0.016) == pytest.approx(0.008)


# ─── TestComputeSuccess ───────────────────────────────────────────────────────

class TestComputeSuccess:
    def test_exit_extension_success(self):
        assert _compute_success("EXIT_EXTENSION", 0.008, 0.023) is True

    def test_exit_extension_fail(self):
        assert _compute_success("EXIT_EXTENSION", 0.023, 0.008) is False

    def test_exit_extension_equal(self):
        assert _compute_success("EXIT_EXTENSION", 0.01, 0.01) is False

    def test_capital_concentration_success(self):
        assert _compute_success("CAPITAL_CONCENTRATION", 0.011, 0.030) is True

    def test_priority_success(self):
        assert _compute_success("PRIORITY", 0.31, 0.57) is True

    def test_priority_fail(self):
        assert _compute_success("PRIORITY", 0.57, 0.31) is False

    def test_signal_leader_abs_improvement(self):
        assert _compute_success("SIGNAL_LEADER", 0.006, 0.014) is True

    def test_signal_leader_abs_fail(self):
        assert _compute_success("SIGNAL_LEADER", 0.014, 0.006) is False

    def test_signal_leader_negative_improves(self):
        # |−0.015| > |−0.010| → improvement
        assert _compute_success("SIGNAL_LEADER", -0.010, -0.015) is True

    def test_signal_leader_direction_flip_improvement(self):
        # abs(0.015) > abs(-0.010) → True
        assert _compute_success("SIGNAL_LEADER", -0.010, 0.015) is True

    def test_continuation_signal_success(self):
        assert _compute_success("CONTINUATION_SIGNAL", 0.008, 0.016) is True

    def test_continuation_signal_fail(self):
        assert _compute_success("CONTINUATION_SIGNAL", 0.016, 0.008) is False


# ─── TestBuildEpIndex ─────────────────────────────────────────────────────────

class TestBuildEpIndex:
    def test_empty(self):
        assert _build_ep_index([]) == {}

    def test_single_record(self):
        rec = {"date": "2026-05-10", "candidates": []}
        idx = _build_ep_index([rec])
        assert "2026-05-10" in idx
        assert idx["2026-05-10"] is rec

    def test_multiple_dates(self):
        r1 = {"date": "2026-05-01"}
        r2 = {"date": "2026-05-10"}
        idx = _build_ep_index([r1, r2])
        assert len(idx) == 2

    def test_same_date_last_wins(self):
        r1 = {"date": "2026-05-10", "val": 1}
        r2 = {"date": "2026-05-10", "val": 2}
        idx = _build_ep_index([r1, r2])
        assert idx["2026-05-10"]["val"] == 2

    def test_missing_date_skipped(self):
        idx = _build_ep_index([{"no_date": True}])
        assert idx == {}


# ─── TestLoadEvaluatedKeys ────────────────────────────────────────────────────

class TestLoadEvaluatedKeys:
    def test_file_not_exist(self, tmp_path):
        keys = _load_evaluated_keys(tmp_path / "missing.jsonl")
        assert keys == set()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "out.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_evaluated_keys(f) == set()

    def test_single_record(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [_outcome_record("2026-06-20", "PRIORITY", "2026-05-31", 20, 0.31, 0.57, 0.26, True)])
        keys = _load_evaluated_keys(f)
        assert ("2026-05-31", "PRIORITY") in keys

    def test_multiple_records(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [
            _outcome_record("2026-06-20", "PRIORITY", "2026-05-31", 20, 0.31, 0.57, 0.26, True),
            _outcome_record("2026-06-25", "SIGNAL_LEADER", "2026-06-05", 20, 0.006, 0.014, 0.008, True),
        ])
        keys = _load_evaluated_keys(f)
        assert ("2026-05-31", "PRIORITY") in keys
        assert ("2026-06-05", "SIGNAL_LEADER") in keys

    def test_missing_fields_skipped(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [{"no_candidate": True}])
        assert _load_evaluated_keys(f) == set()


# ─── TestComputeSummary ───────────────────────────────────────────────────────

class TestComputeSummary:
    def test_empty(self):
        s = _compute_summary([])
        assert s.n_completed == 0
        assert s.success_rate == 0.0
        assert s.avg_delta == 0.0
        assert s.best_candidate == ""
        assert s.worst_candidate == ""

    def test_all_success(self):
        records = [
            _outcome_record("2026-06-20", "PRIORITY", "2026-05-31", 20, 0.31, 0.57, 0.26, True),
            _outcome_record("2026-06-20", "EXIT_EXTENSION", "2026-05-31", 20, 0.008, 0.023, 0.015, True),
        ]
        s = _compute_summary(records)
        assert s.n_completed == 2
        assert s.success_rate == pytest.approx(1.0)

    def test_all_fail(self):
        records = [
            _outcome_record("2026-06-20", "PRIORITY", "2026-05-31", 20, 0.57, 0.31, -0.26, False),
        ]
        s = _compute_summary(records)
        assert s.n_completed == 1
        assert s.success_rate == 0.0

    def test_mixed_success_rate(self):
        records = [
            _outcome_record("2026-06-20", "PRIORITY", "2026-05-31", 20, 0.31, 0.57, 0.26, True),
            _outcome_record("2026-06-20", "PRIORITY", "2026-06-01", 20, 0.57, 0.31, -0.26, False),
            _outcome_record("2026-06-20", "SIGNAL_LEADER", "2026-05-31", 20, 0.006, 0.014, 0.008, True),
            _outcome_record("2026-06-20", "SIGNAL_LEADER", "2026-06-01", 20, 0.014, 0.006, -0.008, False),
        ]
        s = _compute_summary(records)
        assert s.n_completed == 4
        assert s.success_rate == pytest.approx(0.5)

    def test_avg_delta(self):
        records = [
            _outcome_record("2026-06-20", "PRIORITY", "2026-05-31", 20, 0.31, 0.57, 0.26, True),
            _outcome_record("2026-06-20", "EXIT_EXTENSION", "2026-05-31", 20, 0.008, 0.023, 0.015, True),
        ]
        s = _compute_summary(records)
        assert s.avg_delta == pytest.approx((0.26 + 0.015) / 2, abs=1e-5)

    def test_best_worst_candidate(self):
        # PRIORITY: 2/2 success (100%), SIGNAL_LEADER: 0/2 success (0%)
        records = [
            _outcome_record("2026-06-20", "PRIORITY", "2026-05-31", 20, 0.31, 0.57, 0.26, True),
            _outcome_record("2026-06-21", "PRIORITY", "2026-06-01", 20, 0.31, 0.57, 0.26, True),
            _outcome_record("2026-06-20", "SIGNAL_LEADER", "2026-05-31", 20, 0.014, 0.006, -0.008, False),
            _outcome_record("2026-06-21", "SIGNAL_LEADER", "2026-06-01", 20, 0.014, 0.006, -0.008, False),
        ]
        s = _compute_summary(records)
        assert s.best_candidate == "PRIORITY"
        assert s.worst_candidate == "SIGNAL_LEADER"

    def test_single_candidate_best_worst_same(self):
        records = [
            _outcome_record("2026-06-20", "PRIORITY", "2026-05-31", 20, 0.31, 0.57, 0.26, True),
        ]
        s = _compute_summary(records)
        assert s.best_candidate == "PRIORITY"
        assert s.worst_candidate == "PRIORITY"

    def test_invalid_delta_skipped(self):
        records = [{"candidate": "PRIORITY", "delta": "bad", "success": True}]
        s = _compute_summary(records)
        assert s.avg_delta == 0.0


# ─── TestFormatReport ─────────────────────────────────────────────────────────

class TestFormatReport:
    def test_empty_returns_empty_string(self):
        s = ShadowOutcomeSummary(0, 0.0, 0.0, "", "")
        assert format_outcome_report(s) == ""

    def test_with_data(self):
        s = ShadowOutcomeSummary(8, 0.625, 0.014, "CAPITAL_CONCENTRATION", "SIGNAL_LEADER")
        report = format_outcome_report(s)
        assert "SHADOW OUTCOME VALIDATION" in report
        assert "Completed: 8" in report
        assert "62.5%" in report
        assert "+1.4%" in report
        assert "CAPITAL_CONCENTRATION" in report
        assert "SIGNAL_LEADER" in report

    def test_single_candidate_no_worst(self):
        s = ShadowOutcomeSummary(3, 1.0, 0.02, "PRIORITY", "PRIORITY")
        report = format_outcome_report(s)
        assert "Best: PRIORITY" in report
        # best == worst → no "Worst:" line
        assert "Worst:" not in report

    def test_no_best_candidate(self):
        s = ShadowOutcomeSummary(2, 0.5, 0.01, "", "")
        report = format_outcome_report(s)
        assert "Best:" not in report

    def test_separator_present(self):
        s = ShadowOutcomeSummary(1, 1.0, 0.01, "PRIORITY", "PRIORITY")
        report = format_outcome_report(s)
        assert "═" in report


# ─── TestAppendOutcomeRecord ──────────────────────────────────────────────────

class TestAppendOutcomeRecord:
    def test_creates_file(self, tmp_path):
        f = tmp_path / "out.jsonl"
        rec = ShadowOutcomeRecord(
            date="2026-06-20", candidate="PRIORITY", shadow_start_date="2026-05-31",
            shadow_duration_days=20, before_kpi=0.31, after_kpi=0.57,
            delta=0.26, success=True,
        )
        append_outcome_record(rec, f)
        assert f.exists()
        records = _load_all_jsonl(f)
        assert len(records) == 1
        assert records[0]["candidate"] == "PRIORITY"
        assert records[0]["success"] is True
        assert records[0]["schema_version"] == SCHEMA_VERSION

    def test_appends_multiple(self, tmp_path):
        f = tmp_path / "out.jsonl"
        for ct in ["PRIORITY", "EXIT_EXTENSION"]:
            rec = ShadowOutcomeRecord(
                date="2026-06-20", candidate=ct, shadow_start_date="2026-05-31",
                shadow_duration_days=20, before_kpi=0.1, after_kpi=0.2,
                delta=0.1, success=True,
            )
            append_outcome_record(rec, f)
        records = _load_all_jsonl(f)
        assert len(records) == 2
        assert records[0]["candidate"] == "PRIORITY"
        assert records[1]["candidate"] == "EXIT_EXTENSION"

    def test_all_fields_present(self, tmp_path):
        f = tmp_path / "out.jsonl"
        rec = ShadowOutcomeRecord(
            date="2026-06-20", candidate="SIGNAL_LEADER", shadow_start_date="2026-05-31",
            shadow_duration_days=30, before_kpi=0.006, after_kpi=0.014,
            delta=0.008, success=True,
        )
        append_outcome_record(rec, f)
        r = _load_all_jsonl(f)[0]
        assert r["date"] == "2026-06-20"
        assert r["shadow_start_date"] == "2026-05-31"
        assert r["shadow_duration_days"] == 30
        assert r["before_kpi"] == pytest.approx(0.006)
        assert r["after_kpi"] == pytest.approx(0.014)
        assert r["delta"] == pytest.approx(0.008)


# ─── TestLoadAllJsonl ─────────────────────────────────────────────────────────

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
        records = _load_all_jsonl(f)
        assert len(records) == 2
        assert records[0]["a"] == 1

    def test_skips_invalid_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a":1}\nNOT_JSON\n{"b":2}\n', encoding="utf-8")
        records = _load_all_jsonl(f)
        assert len(records) == 2

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
        records = _load_all_jsonl(f)
        assert len(records) == 2


# ─── TestRunShadowOutcomeTracker ──────────────────────────────────────────────

class TestRunShadowOutcomeTracker:
    TODAY = "2026-06-20"

    def test_empty_sr_file(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records == []
        assert result.summary.n_completed == 0

    def test_no_shadow_ready_rec(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [{
            "candidate_type": "PRIORITY",
            "shadow_status": "CONTINUE_OBSERVE",
            "shadow_duration_days": 20,
        }])])
        _write_jsonl(ep, [_ep_record("2026-05-31", [
            _ep_candidate("PRIORITY", "priority_monotonicity", 0.31)
        ])])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records == []

    def test_not_mature_yet(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        # Only 5 days elapsed but duration=20
        _write_jsonl(sr, [_sr_record("2026-06-15", [_sr_recommendation("PRIORITY", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-06-15", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records == []

    def test_mature_emits_record(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        # 20 days elapsed, duration=20 → mature
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert len(result.new_records) == 1
        rec = result.new_records[0]
        assert rec.candidate == "PRIORITY"
        assert rec.before_kpi == pytest.approx(0.31)
        assert rec.after_kpi == pytest.approx(0.57)
        assert rec.success is True
        assert rec.delta == pytest.approx(0.26)
        assert rec.shadow_start_date == "2026-05-31"

    def test_idempotent_second_run_skips(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        r1 = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        r2 = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert len(r1.new_records) == 1
        assert len(r2.new_records) == 0
        # Only one record in output file
        assert len(_load_all_jsonl(out)) == 1

    def test_missing_ep_before_date_skip(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", duration=20)])])
        # EP only has latest record (no record for 2026-05-31)
        _write_jsonl(ep, [
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records == []

    def test_missing_ep_latest_skip(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", duration=20)])])
        # EP has before record but candidate not in latest
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-20", [_ep_candidate("EXIT_EXTENSION", "suppression_return_delta", 0.02)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records == []

    def test_exit_extension_candidate(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("EXIT_EXTENSION", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("EXIT_EXTENSION", "suppression_return_delta", 0.008)]),
            _ep_record("2026-06-20", [_ep_candidate("EXIT_EXTENSION", "suppression_return_delta", 0.023)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert len(result.new_records) == 1
        rec = result.new_records[0]
        assert rec.candidate == "EXIT_EXTENSION"
        assert rec.success is True

    def test_capital_concentration_candidate(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("CAPITAL_CONCENTRATION", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("CAPITAL_CONCENTRATION", "mean_concentration_alpha", 0.011)]),
            _ep_record("2026-06-20", [_ep_candidate("CAPITAL_CONCENTRATION", "mean_concentration_alpha", 0.030)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records[0].success is True

    def test_signal_leader_candidate(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("SIGNAL_LEADER", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("SIGNAL_LEADER", "top_signal_ir", 0.006)]),
            _ep_record("2026-06-20", [_ep_candidate("SIGNAL_LEADER", "top_signal_ir", 0.014)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records[0].success is True

    def test_continuation_signal_candidate(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("CONTINUATION_SIGNAL", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("CONTINUATION_SIGNAL", "top_signal_ir", 0.008)]),
            _ep_record("2026-06-20", [_ep_candidate("CONTINUATION_SIGNAL", "top_signal_ir", 0.016)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records[0].success is True

    def test_failure_outcome(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records[0].success is False
        assert result.new_records[0].delta == pytest.approx(-0.26)

    def test_multiple_candidates_in_sr(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [
            _sr_recommendation("PRIORITY", duration=20),
            _sr_recommendation("EXIT_EXTENSION", duration=20),
        ])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [
                _ep_candidate("PRIORITY", "priority_monotonicity", 0.31),
                _ep_candidate("EXIT_EXTENSION", "suppression_return_delta", 0.008),
            ]),
            _ep_record("2026-06-20", [
                _ep_candidate("PRIORITY", "priority_monotonicity", 0.57),
                _ep_candidate("EXIT_EXTENSION", "suppression_return_delta", 0.023),
            ]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert len(result.new_records) == 2

    def test_multiple_sr_records_only_mature_evaluated(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        # Old record (mature) + new record (not mature)
        _write_jsonl(sr, [
            _sr_record("2026-05-31", [_sr_recommendation("PRIORITY", duration=20)]),
            _sr_record("2026-06-18", [_sr_recommendation("EXIT_EXTENSION", duration=20)]),  # only 2d elapsed
        ])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-18", [_ep_candidate("EXIT_EXTENSION", "suppression_return_delta", 0.008)]),
            _ep_record("2026-06-20", [
                _ep_candidate("PRIORITY", "priority_monotonicity", 0.57),
                _ep_candidate("EXIT_EXTENSION", "suppression_return_delta", 0.023),
            ]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        candidates = [r.candidate for r in result.new_records]
        assert "PRIORITY" in candidates
        assert "EXIT_EXTENSION" not in candidates

    def test_summary_reflects_all_history(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        # Pre-seed output with historical records
        _write_jsonl(out, [
            _outcome_record("2026-06-01", "PRIORITY", "2026-05-12", 20, 0.31, 0.57, 0.26, True),
            _outcome_record("2026-06-05", "EXIT_EXTENSION", "2026-05-16", 20, 0.008, 0.023, 0.015, True),
        ])
        # New mature record for this run
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("CAPITAL_CONCENTRATION", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("CAPITAL_CONCENTRATION", "mean_concentration_alpha", 0.011)]),
            _ep_record("2026-06-20", [_ep_candidate("CAPITAL_CONCENTRATION", "mean_concentration_alpha", 0.030)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        # 2 historical + 1 new = 3 completed
        assert result.summary.n_completed == 3

    def test_empty_ep_file(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", duration=20)])])
        # EP file empty
        ep.write_text("", encoding="utf-8")
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records == []

    def test_fail_open_on_exception(self, tmp_path):
        # Pass a non-existent directory as output → still FAIL_OPEN
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "nonexistent_dir" / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        # Should not raise; output dir is created by _append_jsonl internally
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert isinstance(result, ShadowOutcomeRunResult)

    def test_sr_record_without_date_skipped(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [{"no_date": True, "recommendations": [_sr_recommendation("PRIORITY", 20)]}])
        _write_jsonl(ep, [
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records == []

    def test_exactly_at_duration_boundary(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        # Exactly 20 days elapsed, duration=20 → mature (>=)
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert len(result.new_records) == 1

    def test_one_day_short(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        # 19 days elapsed, duration=20 → NOT mature
        _write_jsonl(sr, [_sr_record("2026-06-01", [_sr_recommendation("PRIORITY", duration=20)])])
        _write_jsonl(ep, [
            _ep_record("2026-06-01", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert result.new_records == []


# ─── TestIntegration ──────────────────────────────────────────────────────────

class TestIntegration:
    TODAY = "2026-06-20"

    def test_full_5_candidate_cycle(self, tmp_path):
        """All 5 candidate types evaluated in one run."""
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"

        candidates_sr = [
            _sr_recommendation("EXIT_EXTENSION", 20),
            _sr_recommendation("CAPITAL_CONCENTRATION", 20),
            _sr_recommendation("PRIORITY", 20),
            _sr_recommendation("SIGNAL_LEADER", 20),
            _sr_recommendation("CONTINUATION_SIGNAL", 20),
        ]
        _write_jsonl(sr, [_sr_record("2026-05-31", candidates_sr)])

        candidates_before = [
            _ep_candidate("EXIT_EXTENSION", "suppression_return_delta", 0.008),
            _ep_candidate("CAPITAL_CONCENTRATION", "mean_concentration_alpha", 0.011),
            _ep_candidate("PRIORITY", "priority_monotonicity", 0.31),
            _ep_candidate("SIGNAL_LEADER", "top_signal_ir", 0.006),
            _ep_candidate("CONTINUATION_SIGNAL", "top_signal_ir", 0.008),
        ]
        candidates_after = [
            _ep_candidate("EXIT_EXTENSION", "suppression_return_delta", 0.023),
            _ep_candidate("CAPITAL_CONCENTRATION", "mean_concentration_alpha", 0.030),
            _ep_candidate("PRIORITY", "priority_monotonicity", 0.57),
            _ep_candidate("SIGNAL_LEADER", "top_signal_ir", 0.014),
            _ep_candidate("CONTINUATION_SIGNAL", "top_signal_ir", 0.016),
        ]
        _write_jsonl(ep, [
            _ep_record("2026-05-31", candidates_before),
            _ep_record("2026-06-20", candidates_after),
        ])

        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        assert len(result.new_records) == 5
        assert all(r.success for r in result.new_records)
        assert result.summary.n_completed == 5
        assert result.summary.success_rate == pytest.approx(1.0)

    def test_report_format_integrated(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", 20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        result = run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        report = format_outcome_report(result.summary)
        assert "SHADOW OUTCOME VALIDATION" in report
        assert "Completed: 1" in report

    def test_schema_version_in_output(self, tmp_path):
        sr = tmp_path / "sr.jsonl"
        ep = tmp_path / "ep.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(sr, [_sr_record("2026-05-31", [_sr_recommendation("PRIORITY", 20)])])
        _write_jsonl(ep, [
            _ep_record("2026-05-31", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.31)]),
            _ep_record("2026-06-20", [_ep_candidate("PRIORITY", "priority_monotonicity", 0.57)]),
        ])
        run_shadow_outcome_tracker(sr, ep, out, self.TODAY)
        raw = _load_all_jsonl(out)
        assert raw[0]["schema_version"] == SCHEMA_VERSION

    def test_kpi_field_coverage(self):
        """All 5 candidate types have KPI field defined."""
        for ct in ["EXIT_EXTENSION", "CAPITAL_CONCENTRATION", "PRIORITY",
                   "SIGNAL_LEADER", "CONTINUATION_SIGNAL"]:
            assert ct in _KPI_FIELD, f"{ct} missing from _KPI_FIELD"
