"""Tests for src/analytics/promotion_impact_tracker.py (Phase 6F-Lite)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.analytics.promotion_impact_tracker import (
    IMPACT_NEGATIVE,
    IMPACT_NEUTRAL,
    IMPACT_POSITIVE,
    SCHEMA_VERSION,
    ImpactRecord,
    ImpactRunResult,
    ImpactSummary,
    _CANDIDATE_ORDER,
    _METRIC_FIELD,
    _METRIC_SOURCE,
    _SHADOW_STATUS_DEPLOYED,
    _append_jsonl,
    _classify_impact,
    _compute_summary,
    _extract_latest_metric,
    _extract_metric_at_or_before,
    _load_all_jsonl,
    _load_deployed_candidates,
    _load_evaluated_keys,
    _load_latest_impact_records,
    _safe_float,
    append_impact_record,
    format_impact_report,
    run_promotion_impact_tracker,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _shadow_rec(candidate: str, date: str = "2026-07-20",
                status: str = "DEPLOYED") -> dict:
    return {
        "date": date,
        "candidate": candidate,
        "planned_change": f"change_{candidate.lower()}",
        "shadow_status": status,
        "shadow_duration_days": 30,
        "schema_version": "v1",
    }


def _metric_rec(date: str, field: str, value: float) -> dict:
    return {"date": date, field: value}


def _make_source_files(tmp_path: Path) -> dict:
    """Return mapping of empty source file paths."""
    files = {}
    for name in ["hold_duration", "capital_concentration",
                 "priority_calibration", "signal_attribution", "forward_continuation"]:
        p = tmp_path / f"{name}.jsonl"
        p.write_text("", encoding="utf-8")
        files[name] = p
    return files


def _run(tmp_path, shadow_recs=None, metric_data=None, today="2026-07-25",
         existing_output=None):
    """Helper to run the tracker with minimal setup."""
    shadow_file = tmp_path / "shadow.jsonl"
    if shadow_recs:
        _write_jsonl(shadow_file, shadow_recs)
    else:
        shadow_file.write_text("", encoding="utf-8")

    src = _make_source_files(tmp_path)
    if metric_data:
        for key, recs in metric_data.items():
            _write_jsonl(src[key], recs)

    out = tmp_path / "impact.jsonl"
    if existing_output:
        _write_jsonl(out, existing_output)

    return run_promotion_impact_tracker(
        shadow_file=shadow_file,
        hold_duration_file=src["hold_duration"],
        capital_conc_file=src["capital_concentration"],
        priority_cal_file=src["priority_calibration"],
        signal_attr_file=src["signal_attribution"],
        fco_file=src["forward_continuation"],
        output_file=out,
        today=today,
    ), out


# ── TestSafeFloat ─────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_normal(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_fallback(self):
        assert _safe_float(None, 99.0) == 99.0

    def test_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf(self):
        assert _safe_float(float("inf")) == 0.0

    def test_string_number(self):
        assert _safe_float("7.5") == pytest.approx(7.5)

    def test_bad_string(self):
        assert _safe_float("bad") == 0.0


# ── TestConstants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_impact_positive(self):
        assert IMPACT_POSITIVE == "POSITIVE"

    def test_impact_negative(self):
        assert IMPACT_NEGATIVE == "NEGATIVE"

    def test_impact_neutral(self):
        assert IMPACT_NEUTRAL == "NEUTRAL"

    def test_schema_version(self):
        assert SCHEMA_VERSION == "v1"

    def test_shadow_deployed(self):
        assert _SHADOW_STATUS_DEPLOYED == "DEPLOYED"

    def test_candidate_order_length(self):
        assert len(_CANDIDATE_ORDER) == 5

    def test_candidate_order_values(self):
        assert _CANDIDATE_ORDER == [
            "EXIT_EXTENSION",
            "CAPITAL_CONCENTRATION",
            "PRIORITY",
            "SIGNAL_LEADER",
            "CONTINUATION_SIGNAL",
        ]

    def test_metric_field_exit_extension(self):
        assert _METRIC_FIELD["EXIT_EXTENSION"] == "suppression_return_delta"

    def test_metric_field_capital_concentration(self):
        assert _METRIC_FIELD["CAPITAL_CONCENTRATION"] == "mean_concentration_alpha"

    def test_metric_field_priority(self):
        assert _METRIC_FIELD["PRIORITY"] == "priority_monotonicity"

    def test_metric_field_signal_leader(self):
        assert _METRIC_FIELD["SIGNAL_LEADER"] == "top_signal_ir"

    def test_metric_field_continuation_signal(self):
        assert _METRIC_FIELD["CONTINUATION_SIGNAL"] == "top_signal_ir"

    def test_metric_source_all_keys(self):
        for ct in _CANDIDATE_ORDER:
            assert ct in _METRIC_SOURCE


# ── TestLoadAllJsonl ──────────────────────────────────────────────────────────

class TestLoadAllJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _load_all_jsonl(tmp_path / "missing.jsonl") == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_all_jsonl(f) == []

    def test_valid_lines_loaded(self, tmp_path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"a": 1}, {"b": 2}])
        assert len(_load_all_jsonl(f)) == 2

    def test_bad_json_skipped(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        f.write_text('{"ok": 1}\nnot_json\n{"ok": 2}\n', encoding="utf-8")
        assert len(_load_all_jsonl(f)) == 2

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "blanks.jsonl"
        f.write_text('\n{"x": 1}\n\n', encoding="utf-8")
        assert len(_load_all_jsonl(f)) == 1


# ── TestLoadDeployedCandidates ────────────────────────────────────────────────

class TestLoadDeployedCandidates:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_deployed_candidates(f) == {}

    def test_missing_file(self, tmp_path):
        assert _load_deployed_candidates(tmp_path / "missing.jsonl") == {}

    def test_deployed_candidate_included(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_shadow_rec("EXIT_EXTENSION", date="2026-07-20")])
        result = _load_deployed_candidates(f)
        assert "EXIT_EXTENSION" in result
        assert result["EXIT_EXTENSION"] == "2026-07-20"

    def test_non_deployed_excluded(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_shadow_rec("EXIT_EXTENSION", status="OTHER")])
        assert _load_deployed_candidates(f) == {}

    def test_latest_date_selected(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [
            _shadow_rec("EXIT_EXTENSION", date="2026-07-01"),
            _shadow_rec("EXIT_EXTENSION", date="2026-07-20"),
        ])
        result = _load_deployed_candidates(f)
        assert result["EXIT_EXTENSION"] == "2026-07-20"

    def test_multiple_candidates(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [
            _shadow_rec("EXIT_EXTENSION"),
            _shadow_rec("PRIORITY"),
        ])
        result = _load_deployed_candidates(f)
        assert set(result.keys()) == {"EXIT_EXTENSION", "PRIORITY"}

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_shadow_rec("UNKNOWN_CAND")])
        assert _load_deployed_candidates(f) == {}


# ── TestExtractMetricAtOrBefore ───────────────────────────────────────────────

class TestExtractMetricAtOrBefore:
    def test_empty_records(self):
        assert _extract_metric_at_or_before([], "field", "2026-07-20") is None

    def test_record_at_cutoff(self):
        recs = [{"date": "2026-07-20", "my_field": 0.5}]
        assert _extract_metric_at_or_before(recs, "my_field", "2026-07-20") == pytest.approx(0.5)

    def test_record_before_cutoff(self):
        recs = [{"date": "2026-07-15", "my_field": 0.3}]
        assert _extract_metric_at_or_before(recs, "my_field", "2026-07-20") == pytest.approx(0.3)

    def test_record_after_cutoff_excluded(self):
        recs = [{"date": "2026-07-25", "my_field": 0.8}]
        assert _extract_metric_at_or_before(recs, "my_field", "2026-07-20") is None

    def test_latest_at_or_before_selected(self):
        recs = [
            {"date": "2026-07-10", "f": 0.1},
            {"date": "2026-07-18", "f": 0.4},
            {"date": "2026-07-25", "f": 0.9},
        ]
        result = _extract_metric_at_or_before(recs, "f", "2026-07-20")
        assert result == pytest.approx(0.4)

    def test_missing_field_skipped(self):
        recs = [{"date": "2026-07-10", "other_field": 0.5}]
        assert _extract_metric_at_or_before(recs, "my_field", "2026-07-20") is None

    def test_negative_value(self):
        recs = [{"date": "2026-07-10", "f": -0.3}]
        assert _extract_metric_at_or_before(recs, "f", "2026-07-20") == pytest.approx(-0.3)

    def test_zero_value(self):
        recs = [{"date": "2026-07-10", "f": 0.0}]
        assert _extract_metric_at_or_before(recs, "f", "2026-07-20") == pytest.approx(0.0)


# ── TestExtractLatestMetric ───────────────────────────────────────────────────

class TestExtractLatestMetric:
    def test_empty_records(self):
        assert _extract_latest_metric([], "field") is None

    def test_single_record(self):
        recs = [{"date": "2026-07-20", "f": 0.7}]
        assert _extract_latest_metric(recs, "f") == pytest.approx(0.7)

    def test_latest_date_selected(self):
        recs = [
            {"date": "2026-07-10", "f": 0.1},
            {"date": "2026-07-25", "f": 0.9},
            {"date": "2026-07-15", "f": 0.5},
        ]
        assert _extract_latest_metric(recs, "f") == pytest.approx(0.9)

    def test_missing_field_skipped(self):
        recs = [
            {"date": "2026-07-10", "other": 0.5},
            {"date": "2026-07-20", "f": 0.7},
        ]
        assert _extract_latest_metric(recs, "f") == pytest.approx(0.7)

    def test_all_missing_field(self):
        recs = [{"date": "2026-07-10", "other": 0.5}]
        assert _extract_latest_metric(recs, "f") is None

    def test_negative_value(self):
        recs = [{"date": "2026-07-20", "f": -0.4}]
        assert _extract_latest_metric(recs, "f") == pytest.approx(-0.4)


# ── TestClassifyImpact ────────────────────────────────────────────────────────

class TestClassifyImpact:
    def test_positive(self):
        assert _classify_impact(0.01) == IMPACT_POSITIVE

    def test_negative(self):
        assert _classify_impact(-0.01) == IMPACT_NEGATIVE

    def test_zero(self):
        assert _classify_impact(0.0) == IMPACT_NEUTRAL

    def test_large_positive(self):
        assert _classify_impact(999.0) == IMPACT_POSITIVE

    def test_large_negative(self):
        assert _classify_impact(-999.0) == IMPACT_NEGATIVE

    def test_small_positive(self):
        assert _classify_impact(1e-10) == IMPACT_POSITIVE

    def test_small_negative(self):
        assert _classify_impact(-1e-10) == IMPACT_NEGATIVE


# ── TestLoadEvaluatedKeys ─────────────────────────────────────────────────────

class TestLoadEvaluatedKeys:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_evaluated_keys(f) == set()

    def test_missing_file(self, tmp_path):
        assert _load_evaluated_keys(tmp_path / "missing.jsonl") == set()

    def test_keys_loaded(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [{"date": "2026-07-25", "candidate": "EXIT_EXTENSION"}])
        keys = _load_evaluated_keys(f)
        assert ("2026-07-25", "EXIT_EXTENSION") in keys

    def test_multiple_keys(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [
            {"date": "2026-07-25", "candidate": "EXIT_EXTENSION"},
            {"date": "2026-07-25", "candidate": "PRIORITY"},
        ])
        keys = _load_evaluated_keys(f)
        assert len(keys) == 2

    def test_empty_fields_excluded(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [{"date": "", "candidate": "EXIT_EXTENSION"}])
        assert _load_evaluated_keys(f) == set()


# ── TestLoadLatestImpactRecords ───────────────────────────────────────────────

class TestLoadLatestImpactRecords:
    def _make_rec(self, date, ct, delta, status):
        return {
            "date": date,
            "candidate": ct,
            "metric_field": "f",
            "deployment_date": "2026-07-20",
            "before_metric": 0.1,
            "after_metric": 0.1 + delta,
            "impact_delta": delta,
            "impact_status": status,
            "schema_version": "v1",
        }

    def test_empty_file(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_latest_impact_records(f) == []

    def test_candidate_order_preserved(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [
            self._make_rec("2026-07-25", "PRIORITY", 0.1, "POSITIVE"),
            self._make_rec("2026-07-25", "EXIT_EXTENSION", 0.2, "POSITIVE"),
        ])
        result = _load_latest_impact_records(f)
        assert [r.candidate for r in result] == ["EXIT_EXTENSION", "PRIORITY"]

    def test_latest_per_candidate(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [
            self._make_rec("2026-07-24", "EXIT_EXTENSION", 0.1, "POSITIVE"),
            self._make_rec("2026-07-25", "EXIT_EXTENSION", 0.2, "POSITIVE"),
        ])
        result = _load_latest_impact_records(f)
        assert len(result) == 1
        assert result[0].date == "2026-07-25"

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "impact.jsonl"
        _write_jsonl(f, [self._make_rec("2026-07-25", "UNKNOWN", 0.1, "POSITIVE")])
        assert _load_latest_impact_records(f) == []


# ── TestComputeSummary ────────────────────────────────────────────────────────

class TestComputeSummary:
    def _rec(self, ct, delta, status):
        return ImpactRecord(
            date="2026-07-25",
            candidate=ct,
            metric_field="f",
            deployment_date="2026-07-20",
            before_metric=0.0,
            after_metric=delta,
            impact_delta=delta,
            impact_status=status,
        )

    def test_empty(self):
        s = _compute_summary([])
        assert s.n_deployed == 0
        assert s.n_positive == 0
        assert s.success_rate == 0.0
        assert s.best_candidate == ""

    def test_single_positive(self):
        s = _compute_summary([self._rec("EXIT_EXTENSION", 0.1, IMPACT_POSITIVE)])
        assert s.n_deployed == 1
        assert s.n_positive == 1
        assert s.success_rate == 100.0
        assert s.best_candidate == "EXIT_EXTENSION"

    def test_single_negative(self):
        s = _compute_summary([self._rec("EXIT_EXTENSION", -0.1, IMPACT_NEGATIVE)])
        assert s.n_negative == 1
        assert s.success_rate == 0.0

    def test_mixed(self):
        records = [
            self._rec("EXIT_EXTENSION",        0.2,  IMPACT_POSITIVE),
            self._rec("CAPITAL_CONCENTRATION", -0.1, IMPACT_NEGATIVE),
            self._rec("PRIORITY",               0.0,  IMPACT_NEUTRAL),
        ]
        s = _compute_summary(records)
        assert s.n_deployed == 3
        assert s.n_positive == 1
        assert s.n_negative == 1
        assert s.success_rate == pytest.approx(33.3, abs=0.2)

    def test_best_candidate_highest_delta(self):
        records = [
            self._rec("EXIT_EXTENSION",        0.5, IMPACT_POSITIVE),
            self._rec("CAPITAL_CONCENTRATION", 0.9, IMPACT_POSITIVE),
        ]
        s = _compute_summary(records)
        assert s.best_candidate == "CAPITAL_CONCENTRATION"
        assert s.best_impact_delta == pytest.approx(0.9)

    def test_success_rate_calculation(self):
        records = [
            self._rec("EXIT_EXTENSION",        0.1, IMPACT_POSITIVE),
            self._rec("CAPITAL_CONCENTRATION", 0.2, IMPACT_POSITIVE),
            self._rec("PRIORITY",             -0.1, IMPACT_NEGATIVE),
            self._rec("SIGNAL_LEADER",        -0.2, IMPACT_NEGATIVE),
        ]
        s = _compute_summary(records)
        assert s.success_rate == pytest.approx(50.0)


# ── TestFormatReport ──────────────────────────────────────────────────────────

class TestFormatReport:
    def _rec(self, ct, before, after, delta, status):
        return ImpactRecord(
            date="2026-07-25",
            candidate=ct,
            metric_field="suppression_return_delta",
            deployment_date="2026-07-20",
            before_metric=before,
            after_metric=after,
            impact_delta=delta,
            impact_status=status,
        )

    def test_empty_returns_empty(self):
        s = ImpactSummary(0, 0, 0, 0.0, "", 0.0)
        assert format_impact_report([], s) == ""

    def test_contains_header(self):
        r = self._rec("EXIT_EXTENSION", 0.1, 0.2, 0.1, IMPACT_POSITIVE)
        s = _compute_summary([r])
        rpt = format_impact_report([r], s)
        assert "PROMOTION IMPACT TRACKER" in rpt

    def test_contains_separator(self):
        r = self._rec("EXIT_EXTENSION", 0.1, 0.2, 0.1, IMPACT_POSITIVE)
        s = _compute_summary([r])
        rpt = format_impact_report([r], s)
        assert "═" in rpt

    def test_contains_candidate(self):
        r = self._rec("EXIT_EXTENSION", 0.1, 0.2, 0.1, IMPACT_POSITIVE)
        s = _compute_summary([r])
        rpt = format_impact_report([r], s)
        assert "EXIT_EXTENSION" in rpt

    def test_contains_status(self):
        r = self._rec("EXIT_EXTENSION", 0.1, 0.2, 0.1, IMPACT_POSITIVE)
        s = _compute_summary([r])
        rpt = format_impact_report([r], s)
        assert "POSITIVE" in rpt

    def test_contains_success_rate(self):
        r = self._rec("EXIT_EXTENSION", 0.1, 0.2, 0.1, IMPACT_POSITIVE)
        s = _compute_summary([r])
        rpt = format_impact_report([r], s)
        assert "Success Rate" in rpt

    def test_contains_best_candidate(self):
        r = self._rec("EXIT_EXTENSION", 0.1, 0.2, 0.1, IMPACT_POSITIVE)
        s = _compute_summary([r])
        rpt = format_impact_report([r], s)
        assert "Best" in rpt
        assert "EXIT_EXTENSION" in rpt

    def test_contains_table_header(self):
        r = self._rec("EXIT_EXTENSION", 0.1, 0.2, 0.1, IMPACT_POSITIVE)
        s = _compute_summary([r])
        rpt = format_impact_report([r], s)
        assert "Before" in rpt
        assert "After" in rpt
        assert "Delta" in rpt

    def test_negative_status_shown(self):
        r = self._rec("EXIT_EXTENSION", 0.2, 0.1, -0.1, IMPACT_NEGATIVE)
        s = _compute_summary([r])
        rpt = format_impact_report([r], s)
        assert "NEGATIVE" in rpt

    def test_all_five_candidates(self):
        records = [
            self._rec("EXIT_EXTENSION",        0.1, 0.2,  0.1, IMPACT_POSITIVE),
            self._rec("CAPITAL_CONCENTRATION", 0.0, 0.05, 0.05, IMPACT_POSITIVE),
            self._rec("PRIORITY",              0.5, 0.4, -0.1, IMPACT_NEGATIVE),
            self._rec("SIGNAL_LEADER",         0.3, 0.3,  0.0, IMPACT_NEUTRAL),
            self._rec("CONTINUATION_SIGNAL",   0.2, 0.25, 0.05, IMPACT_POSITIVE),
        ]
        s = _compute_summary(records)
        rpt = format_impact_report(records, s)
        for ct in _CANDIDATE_ORDER:
            assert ct in rpt


# ── TestAppendImpactRecord ────────────────────────────────────────────────────

class TestAppendImpactRecord:
    def _make_record(self):
        return ImpactRecord(
            date="2026-07-25",
            candidate="EXIT_EXTENSION",
            metric_field="suppression_return_delta",
            deployment_date="2026-07-20",
            before_metric=0.1,
            after_metric=0.2,
            impact_delta=0.1,
            impact_status=IMPACT_POSITIVE,
        )

    def test_creates_file(self, tmp_path):
        path = tmp_path / "out" / "impact.jsonl"
        append_impact_record(self._make_record(), path)
        assert path.exists()

    def test_all_fields_present(self, tmp_path):
        path = tmp_path / "impact.jsonl"
        append_impact_record(self._make_record(), path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["date"] == "2026-07-25"
        assert data["candidate"] == "EXIT_EXTENSION"
        assert data["metric_field"] == "suppression_return_delta"
        assert data["deployment_date"] == "2026-07-20"
        assert data["before_metric"] == pytest.approx(0.1)
        assert data["after_metric"] == pytest.approx(0.2)
        assert data["impact_delta"] == pytest.approx(0.1)
        assert data["impact_status"] == IMPACT_POSITIVE
        assert data["schema_version"] == "v1"

    def test_appends_multiple(self, tmp_path):
        path = tmp_path / "impact.jsonl"
        for ct in ["EXIT_EXTENSION", "PRIORITY"]:
            rec = ImpactRecord("2026-07-25", ct, "f", "2026-07-20", 0.1, 0.2, 0.1, IMPACT_POSITIVE)
            append_impact_record(rec, path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "dir" / "impact.jsonl"
        append_impact_record(self._make_record(), path)
        assert path.exists()


# ── TestRunPromotionImpactTracker ─────────────────────────────────────────────

class TestRunPromotionImpactTracker:
    def test_no_deployed_no_new(self, tmp_path):
        result, _ = _run(tmp_path)
        assert result.new_records == []

    def test_deployed_no_metric_no_new(self, tmp_path):
        # shadow has candidate but metric file is empty → no before/after
        result, _ = _run(tmp_path,
                         shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")])
        assert result.new_records == []

    def test_deployed_with_before_metric_only(self, tmp_path):
        # only before metric exists (before deployment date), no after
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [_metric_rec("2026-07-15", "suppression_return_delta", 0.1)],
            },
        )
        # before = 0.1 at 2026-07-15, after = same record (latest overall)
        assert len(result.new_records) == 1

    def test_deployed_with_both_metrics(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
            },
        )
        assert len(result.new_records) == 1
        rec = result.new_records[0]
        assert rec.before_metric == pytest.approx(0.1)
        assert rec.after_metric == pytest.approx(0.3)
        assert rec.impact_delta == pytest.approx(0.2)
        assert rec.impact_status == IMPACT_POSITIVE

    def test_negative_impact(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.5),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.2),
                ],
            },
        )
        rec = result.new_records[0]
        assert rec.impact_status == IMPACT_NEGATIVE
        assert rec.impact_delta < 0

    def test_neutral_impact(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-20", "suppression_return_delta", 0.3),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
            },
        )
        rec = result.new_records[0]
        assert rec.impact_status == IMPACT_NEUTRAL

    def test_idempotent_same_day(self, tmp_path):
        existing = [{
            "date": "2026-07-25",
            "candidate": "EXIT_EXTENSION",
            "metric_field": "suppression_return_delta",
            "deployment_date": "2026-07-20",
            "before_metric": 0.1,
            "after_metric": 0.3,
            "impact_delta": 0.2,
            "impact_status": IMPACT_POSITIVE,
            "schema_version": "v1",
        }]
        result, out = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
            },
            existing_output=existing,
        )
        assert result.new_records == []
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_capital_concentration_metric(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("CAPITAL_CONCENTRATION", date="2026-07-20")],
            metric_data={
                "capital_concentration": [
                    _metric_rec("2026-07-15", "mean_concentration_alpha", 0.02),
                    _metric_rec("2026-07-25", "mean_concentration_alpha", 0.05),
                ],
            },
        )
        assert len(result.new_records) == 1
        rec = result.new_records[0]
        assert rec.metric_field == "mean_concentration_alpha"
        assert rec.impact_status == IMPACT_POSITIVE

    def test_priority_metric(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("PRIORITY", date="2026-07-20")],
            metric_data={
                "priority_calibration": [
                    _metric_rec("2026-07-15", "priority_monotonicity", 0.4),
                    _metric_rec("2026-07-25", "priority_monotonicity", 0.7),
                ],
            },
        )
        assert len(result.new_records) == 1
        rec = result.new_records[0]
        assert rec.metric_field == "priority_monotonicity"

    def test_signal_leader_metric(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("SIGNAL_LEADER", date="2026-07-20")],
            metric_data={
                "signal_attribution": [
                    _metric_rec("2026-07-15", "top_signal_ir", 0.1),
                    _metric_rec("2026-07-25", "top_signal_ir", 0.3),
                ],
            },
        )
        rec = result.new_records[0]
        assert rec.metric_field == "top_signal_ir"

    def test_continuation_signal_metric(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("CONTINUATION_SIGNAL", date="2026-07-20")],
            metric_data={
                "forward_continuation": [
                    _metric_rec("2026-07-15", "top_signal_ir", 0.05),
                    _metric_rec("2026-07-25", "top_signal_ir", 0.15),
                ],
            },
        )
        rec = result.new_records[0]
        assert rec.metric_field == "top_signal_ir"

    def test_summary_n_deployed(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
            },
        )
        assert result.summary.n_deployed == 1

    def test_summary_n_positive(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
            },
        )
        assert result.summary.n_positive == 1

    def test_summary_success_rate(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
            },
        )
        assert result.summary.success_rate == pytest.approx(100.0)

    def test_summary_best_candidate(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.5),
                ],
            },
        )
        assert result.summary.best_candidate == "EXIT_EXTENSION"

    def test_fail_open_bad_shadow_file(self, tmp_path):
        src = _make_source_files(tmp_path)
        out = tmp_path / "impact.jsonl"
        result = run_promotion_impact_tracker(
            shadow_file=tmp_path / "nonexistent" / "shadow.jsonl",
            hold_duration_file=src["hold_duration"],
            capital_conc_file=src["capital_concentration"],
            priority_cal_file=src["priority_calibration"],
            signal_attr_file=src["signal_attribution"],
            fco_file=src["forward_continuation"],
            output_file=out,
            today="2026-07-25",
        )
        assert isinstance(result, ImpactRunResult)
        assert result.new_records == []

    def test_result_date(self, tmp_path):
        result, _ = _run(tmp_path, today="2026-08-01")
        assert result.date == "2026-08-01"

    def test_before_metric_at_deployment_date(self, tmp_path):
        # Record exactly at deployment date should be used as before
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-20", "suppression_return_delta", 0.25),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.40),
                ],
            },
        )
        rec = result.new_records[0]
        assert rec.before_metric == pytest.approx(0.25)

    def test_record_after_deployment_not_used_as_before(self, tmp_path):
        # Only post-deployment records available → no before → skip
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
            },
        )
        # before=None (no record at/before 2026-07-20) → skip
        assert result.new_records == []

    def test_multiple_candidates_multiple_new(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[
                _shadow_rec("EXIT_EXTENSION",        date="2026-07-20"),
                _shadow_rec("CAPITAL_CONCENTRATION", date="2026-07-20"),
            ],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
                "capital_concentration": [
                    _metric_rec("2026-07-15", "mean_concentration_alpha", 0.02),
                    _metric_rec("2026-07-25", "mean_concentration_alpha", 0.04),
                ],
            },
        )
        assert len(result.new_records) == 2

    def test_deployment_date_stored_in_record(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
            },
        )
        assert result.new_records[0].deployment_date == "2026-07-20"


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_cycle_all_candidates(self, tmp_path):
        shadow_file = tmp_path / "shadow.jsonl"
        _write_jsonl(shadow_file, [_shadow_rec(ct, date="2026-07-20") for ct in _CANDIDATE_ORDER])
        src = _make_source_files(tmp_path)
        for key, field in [
            ("hold_duration",        "suppression_return_delta"),
            ("capital_concentration","mean_concentration_alpha"),
            ("priority_calibration", "priority_monotonicity"),
            ("signal_attribution",   "top_signal_ir"),
            ("forward_continuation", "top_signal_ir"),
        ]:
            _write_jsonl(src[key], [
                _metric_rec("2026-07-15", field, 0.1),
                _metric_rec("2026-07-25", field, 0.3),
            ])
        out = tmp_path / "impact.jsonl"
        result = run_promotion_impact_tracker(
            shadow_file=shadow_file,
            hold_duration_file=src["hold_duration"],
            capital_conc_file=src["capital_concentration"],
            priority_cal_file=src["priority_calibration"],
            signal_attr_file=src["signal_attribution"],
            fco_file=src["forward_continuation"],
            output_file=out,
            today="2026-07-25",
        )
        assert len(result.new_records) == 5
        assert result.summary.n_positive == 5
        assert result.summary.success_rate == pytest.approx(100.0)

    def test_report_integrated(self, tmp_path):
        result, _ = _run(
            tmp_path,
            shadow_recs=[_shadow_rec("EXIT_EXTENSION", date="2026-07-20")],
            metric_data={
                "hold_duration": [
                    _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
                    _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
                ],
            },
        )
        rpt = format_impact_report(result.impact_records, result.summary)
        assert "PROMOTION IMPACT TRACKER" in rpt
        assert "EXIT_EXTENSION" in rpt

    def test_idempotent_three_runs(self, tmp_path):
        shadow_file = tmp_path / "shadow.jsonl"
        _write_jsonl(shadow_file, [_shadow_rec("EXIT_EXTENSION", date="2026-07-20")])
        src = _make_source_files(tmp_path)
        _write_jsonl(src["hold_duration"], [
            _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
            _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
        ])
        out = tmp_path / "impact.jsonl"
        for _ in range(3):
            run_promotion_impact_tracker(
                shadow_file=shadow_file,
                hold_duration_file=src["hold_duration"],
                capital_conc_file=src["capital_concentration"],
                priority_cal_file=src["priority_calibration"],
                signal_attr_file=src["signal_attribution"],
                fco_file=src["forward_continuation"],
                output_file=out,
                today="2026-07-25",
            )
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_candidate_order_in_impact_records(self, tmp_path):
        shadow_file = tmp_path / "shadow.jsonl"
        _write_jsonl(shadow_file, [_shadow_rec(ct, date="2026-07-20") for ct in _CANDIDATE_ORDER])
        src = _make_source_files(tmp_path)
        for key, field in [
            ("hold_duration",        "suppression_return_delta"),
            ("capital_concentration","mean_concentration_alpha"),
            ("priority_calibration", "priority_monotonicity"),
            ("signal_attribution",   "top_signal_ir"),
            ("forward_continuation", "top_signal_ir"),
        ]:
            _write_jsonl(src[key], [
                _metric_rec("2026-07-15", field, 0.1),
                _metric_rec("2026-07-25", field, 0.3),
            ])
        out = tmp_path / "impact.jsonl"
        result = run_promotion_impact_tracker(
            shadow_file=shadow_file,
            hold_duration_file=src["hold_duration"],
            capital_conc_file=src["capital_concentration"],
            priority_cal_file=src["priority_calibration"],
            signal_attr_file=src["signal_attribution"],
            fco_file=src["forward_continuation"],
            output_file=out,
            today="2026-07-25",
        )
        for i, rec in enumerate(result.impact_records):
            assert rec.candidate == _CANDIDATE_ORDER[i]

    def test_report_empty_when_no_impact(self):
        s = ImpactSummary(0, 0, 0, 0.0, "", 0.0)
        assert format_impact_report([], s) == ""

    def test_daily_accumulation(self, tmp_path):
        shadow_file = tmp_path / "shadow.jsonl"
        _write_jsonl(shadow_file, [_shadow_rec("EXIT_EXTENSION", date="2026-07-20")])
        src = _make_source_files(tmp_path)
        _write_jsonl(src["hold_duration"], [
            _metric_rec("2026-07-15", "suppression_return_delta", 0.1),
            _metric_rec("2026-07-25", "suppression_return_delta", 0.3),
            _metric_rec("2026-07-26", "suppression_return_delta", 0.35),
        ])
        out = tmp_path / "impact.jsonl"
        for day in ["2026-07-25", "2026-07-26"]:
            run_promotion_impact_tracker(
                shadow_file=shadow_file,
                hold_duration_file=src["hold_duration"],
                capital_conc_file=src["capital_concentration"],
                priority_cal_file=src["priority_calibration"],
                signal_attr_file=src["signal_attribution"],
                fco_file=src["forward_continuation"],
                output_file=out,
                today=day,
            )
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
