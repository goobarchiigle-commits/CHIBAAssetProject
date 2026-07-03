"""
tests/test_production_promotion_gate.py — Production Promotion Gate (Phase 6I-Lite)
"""
import json
import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.production_promotion_gate import (
    DECISION_APPROVED,
    DECISION_REVIEW_REQUIRED,
    DECISION_REJECTED,
    SCHEMA_VERSION,
    _APPROVAL_SCORE_MIN,
    _APPROVAL_DAYS_MIN,
    _STATE_PRODUCTION_READY,
    _IMPACT_POSITIVE,
    _IMPACT_NEGATIVE,
    _CANDIDATE_ORDER,
    GateRecord,
    GateSummary,
    GateRunResult,
    compute_risk_score,
    classify_candidate,
    compute_gate_records,
    compute_gate_summary,
    append_gate_result,
    format_gate_report,
    run_production_promotion_gate,
    _load_all_jsonl,
    _load_evaluated_dates,
    _latest_rollout_states,
    _latest_governance_scores,
    _latest_impact_statuses,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _rollout_rec(candidate, state, days_in_state=35, gov_score=84.5,
                 impact="POSITIVE", date="2026-05-30"):
    return {
        "date": date,
        "candidate": candidate,
        "state": state,
        "days_in_state": days_in_state,
        "state_entry_date": "2026-04-30",
        "latest_impact_status": impact,
        "governance_score": gov_score,
        "schema_version": "v1",
    }


def _sb_rec(candidate, gov_score=84.5, date="2026-05-30"):
    return {
        "date": date,
        "candidate": candidate,
        "governance_score": gov_score,
        "classification": "PROMOTE_FIRST",
    }


def _impact_rec(candidate, impact_status="POSITIVE", date="2026-05-30"):
    return {
        "date": date,
        "candidate": candidate,
        "impact_status": impact_status,
    }


# ── TestConstants ──────────────────────────────────────────────────────────────

class TestConstants:
    def test_decision_approved(self):
        assert DECISION_APPROVED == "APPROVED"

    def test_decision_review_required(self):
        assert DECISION_REVIEW_REQUIRED == "REVIEW_REQUIRED"

    def test_decision_rejected(self):
        assert DECISION_REJECTED == "REJECTED"

    def test_approval_score_min(self):
        assert _APPROVAL_SCORE_MIN == 80.0

    def test_approval_days_min(self):
        assert _APPROVAL_DAYS_MIN == 30

    def test_state_production_ready(self):
        assert _STATE_PRODUCTION_READY == "PRODUCTION_READY"

    def test_impact_positive(self):
        assert _IMPACT_POSITIVE == "POSITIVE"

    def test_impact_negative(self):
        assert _IMPACT_NEGATIVE == "NEGATIVE"

    def test_schema_version(self):
        assert SCHEMA_VERSION == "v1"

    def test_candidate_order_length(self):
        assert len(_CANDIDATE_ORDER) == 5

    def test_candidate_order_contents(self):
        assert "EXIT_EXTENSION" in _CANDIDATE_ORDER
        assert "CONTINUATION_SIGNAL" in _CANDIDATE_ORDER


# ── TestComputeRiskScore ───────────────────────────────────────────────────────

class TestComputeRiskScore:
    def test_score_80_gives_risk_20(self):
        assert compute_risk_score(80.0) == 20.0

    def test_score_100_gives_risk_0(self):
        assert compute_risk_score(100.0) == 0.0

    def test_score_0_gives_risk_100(self):
        assert compute_risk_score(0.0) == 100.0

    def test_score_50_gives_risk_50(self):
        assert compute_risk_score(50.0) == 50.0

    def test_clamp_negative_input(self):
        # governance_score > 100 → risk clamped to 0
        assert compute_risk_score(110.0) == 0.0

    def test_clamp_negative_risk(self):
        # governance_score < 0 → risk clamped to 100
        assert compute_risk_score(-10.0) == 100.0

    def test_decimal_precision(self):
        result = compute_risk_score(84.5)
        assert result == pytest.approx(15.5, abs=1e-9)

    def test_returns_float(self):
        assert isinstance(compute_risk_score(75.0), float)

    def test_score_84_5(self):
        assert compute_risk_score(84.5) == pytest.approx(15.5, abs=1e-9)

    def test_score_exact_80(self):
        assert compute_risk_score(80.0) == 20.0


# ── TestClassifyCandidate ──────────────────────────────────────────────────────

class TestClassifyCandidate:
    def test_all_conditions_met_approved(self):
        assert classify_candidate(85.0, "POSITIVE", 35) == DECISION_APPROVED

    def test_score_exactly_80_approved(self):
        assert classify_candidate(80.0, "POSITIVE", 30) == DECISION_APPROVED

    def test_days_exactly_30_approved(self):
        assert classify_candidate(90.0, "POSITIVE", 30) == DECISION_APPROVED

    def test_score_below_80_rejected(self):
        assert classify_candidate(79.9, "POSITIVE", 35) == DECISION_REJECTED

    def test_score_zero_rejected(self):
        assert classify_candidate(0.0, "POSITIVE", 35) == DECISION_REJECTED

    def test_impact_negative_rejected(self):
        assert classify_candidate(85.0, "NEGATIVE", 35) == DECISION_REJECTED

    def test_impact_negative_even_high_score_rejected(self):
        assert classify_candidate(99.0, "NEGATIVE", 100) == DECISION_REJECTED

    def test_days_below_30_review(self):
        assert classify_candidate(85.0, "POSITIVE", 29) == DECISION_REVIEW_REQUIRED

    def test_days_zero_review(self):
        assert classify_candidate(85.0, "POSITIVE", 0) == DECISION_REVIEW_REQUIRED

    def test_impact_neutral_review(self):
        assert classify_candidate(85.0, "NEUTRAL", 35) == DECISION_REVIEW_REQUIRED

    def test_impact_empty_review(self):
        assert classify_candidate(85.0, "", 35) == DECISION_REVIEW_REQUIRED

    def test_impact_positive_days_29_review(self):
        assert classify_candidate(80.0, "POSITIVE", 29) == DECISION_REVIEW_REQUIRED

    def test_score_below_80_and_negative_rejected(self):
        assert classify_candidate(50.0, "NEGATIVE", 35) == DECISION_REJECTED

    def test_score_exactly_79_9_rejected(self):
        assert classify_candidate(79.9, "POSITIVE", 30) == DECISION_REJECTED

    def test_impact_neutral_days_100_review(self):
        assert classify_candidate(95.0, "NEUTRAL", 100) == DECISION_REVIEW_REQUIRED

    def test_impact_positive_score_80_days_30_approved(self):
        assert classify_candidate(80.0, "POSITIVE", 30) == DECISION_APPROVED

    def test_unknown_impact_review_when_score_ok(self):
        assert classify_candidate(90.0, "UNKNOWN", 35) == DECISION_REVIEW_REQUIRED


# ── TestLoadAllJsonl ───────────────────────────────────────────────────────────

class TestLoadAllJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        assert _load_all_jsonl(p) == []

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert _load_all_jsonl(p) == []

    def test_valid_records_loaded(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"a": 1}, {"b": 2}])
        result = _load_all_jsonl(p)
        assert len(result) == 2

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"ok": 1}\nnot-json\n{"ok": 2}\n', encoding="utf-8")
        result = _load_all_jsonl(p)
        assert len(result) == 2

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"ok": 1}\n\n\n{"ok": 2}\n', encoding="utf-8")
        result = _load_all_jsonl(p)
        assert len(result) == 2


# ── TestLoadEvaluatedDates ─────────────────────────────────────────────────────

class TestLoadEvaluatedDates:
    def test_empty_file_returns_empty_set(self, tmp_path):
        p = tmp_path / "out.jsonl"
        assert _load_evaluated_dates(p) == set()

    def test_missing_file_returns_empty_set(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        assert _load_evaluated_dates(p) == set()

    def test_dates_collected(self, tmp_path):
        p = tmp_path / "out.jsonl"
        _write_jsonl(p, [{"date": "2026-05-30"}, {"date": "2026-05-31"}])
        result = _load_evaluated_dates(p)
        assert "2026-05-30" in result
        assert "2026-05-31" in result

    def test_duplicate_dates_deduplicated(self, tmp_path):
        p = tmp_path / "out.jsonl"
        _write_jsonl(p, [{"date": "2026-05-30"}, {"date": "2026-05-30"}])
        result = _load_evaluated_dates(p)
        assert len(result) == 1

    def test_record_without_date_ignored(self, tmp_path):
        p = tmp_path / "out.jsonl"
        _write_jsonl(p, [{"no_date": 1}])
        result = _load_evaluated_dates(p)
        assert len(result) == 0


# ── TestLatestRolloutStates ────────────────────────────────────────────────────

class TestLatestRolloutStates:
    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "rollout.jsonl"
        assert _latest_rollout_states(p) == {}

    def test_missing_file_returns_empty(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        assert _latest_rollout_states(p) == {}

    def test_single_record_returned(self, tmp_path):
        p = tmp_path / "rollout.jsonl"
        _write_jsonl(p, [_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY")])
        result = _latest_rollout_states(p)
        assert "EXIT_EXTENSION" in result
        assert result["EXIT_EXTENSION"]["state"] == "PRODUCTION_READY"

    def test_latest_date_wins(self, tmp_path):
        p = tmp_path / "rollout.jsonl"
        _write_jsonl(p, [
            _rollout_rec("EXIT_EXTENSION", "PAPER_SHADOW", date="2026-05-29"),
            _rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", date="2026-05-30"),
        ])
        result = _latest_rollout_states(p)
        assert result["EXIT_EXTENSION"]["state"] == "PRODUCTION_READY"

    def test_older_date_does_not_override(self, tmp_path):
        p = tmp_path / "rollout.jsonl"
        _write_jsonl(p, [
            _rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", date="2026-05-30"),
            _rollout_rec("EXIT_EXTENSION", "PAPER_SHADOW", date="2026-05-28"),
        ])
        result = _latest_rollout_states(p)
        assert result["EXIT_EXTENSION"]["state"] == "PRODUCTION_READY"

    def test_unknown_candidate_ignored(self, tmp_path):
        p = tmp_path / "rollout.jsonl"
        _write_jsonl(p, [{"date": "2026-05-30", "candidate": "UNKNOWN", "state": "PRODUCTION_READY"}])
        result = _latest_rollout_states(p)
        assert "UNKNOWN" not in result

    def test_multiple_candidates_returned(self, tmp_path):
        p = tmp_path / "rollout.jsonl"
        _write_jsonl(p, [
            _rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY"),
            _rollout_rec("CAPITAL_CONCENTRATION", "PAPER_SHADOW"),
        ])
        result = _latest_rollout_states(p)
        assert len(result) == 2


# ── TestLatestGovernanceScores ─────────────────────────────────────────────────

class TestLatestGovernanceScores:
    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "sb.jsonl"
        assert _latest_governance_scores(p) == {}

    def test_latest_score_selected(self, tmp_path):
        p = tmp_path / "sb.jsonl"
        _write_jsonl(p, [
            _sb_rec("EXIT_EXTENSION", gov_score=75.0, date="2026-05-29"),
            _sb_rec("EXIT_EXTENSION", gov_score=84.5, date="2026-05-30"),
        ])
        result = _latest_governance_scores(p)
        assert result["EXIT_EXTENSION"] == pytest.approx(84.5)

    def test_unknown_candidate_ignored(self, tmp_path):
        p = tmp_path / "sb.jsonl"
        _write_jsonl(p, [{"date": "2026-05-30", "candidate": "UNKNOWN", "governance_score": 90.0}])
        result = _latest_governance_scores(p)
        assert "UNKNOWN" not in result

    def test_missing_governance_score_defaults_to_zero(self, tmp_path):
        p = tmp_path / "sb.jsonl"
        _write_jsonl(p, [{"date": "2026-05-30", "candidate": "EXIT_EXTENSION"}])
        result = _latest_governance_scores(p)
        assert result["EXIT_EXTENSION"] == 0.0


# ── TestLatestImpactStatuses ───────────────────────────────────────────────────

class TestLatestImpactStatuses:
    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "impact.jsonl"
        assert _latest_impact_statuses(p) == {}

    def test_latest_status_selected(self, tmp_path):
        p = tmp_path / "impact.jsonl"
        _write_jsonl(p, [
            _impact_rec("EXIT_EXTENSION", "NEUTRAL", date="2026-05-29"),
            _impact_rec("EXIT_EXTENSION", "POSITIVE", date="2026-05-30"),
        ])
        result = _latest_impact_statuses(p)
        assert result["EXIT_EXTENSION"] == "POSITIVE"

    def test_unknown_candidate_ignored(self, tmp_path):
        p = tmp_path / "impact.jsonl"
        _write_jsonl(p, [{"date": "2026-05-30", "candidate": "UNKNOWN", "impact_status": "POSITIVE"}])
        result = _latest_impact_statuses(p)
        assert "UNKNOWN" not in result


# ── TestComputeGateRecords ─────────────────────────────────────────────────────

class TestComputeGateRecords:
    def _make_rollout(self, state="PRODUCTION_READY", days=35, gov=84.5, impact="POSITIVE"):
        return {
            "state": state, "days_in_state": days,
            "governance_score": gov, "latest_impact_status": impact,
        }

    def test_non_production_ready_excluded(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout(state="MICRO_CAPITAL")}
        result = compute_gate_records(rollout, {}, {})
        assert len(result) == 0

    def test_production_ready_included(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout()}
        result = compute_gate_records(rollout, {}, {})
        assert len(result) == 1
        assert result[0].candidate == "EXIT_EXTENSION"

    def test_decision_approved_all_conditions(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout(gov=85.0, impact="POSITIVE", days=35)}
        gov_scores = {"EXIT_EXTENSION": 85.0}
        impact_statuses = {"EXIT_EXTENSION": "POSITIVE"}
        result = compute_gate_records(rollout, gov_scores, impact_statuses)
        assert result[0].decision == DECISION_APPROVED

    def test_decision_rejected_low_score(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout(gov=70.0, impact="POSITIVE", days=35)}
        gov_scores = {"EXIT_EXTENSION": 70.0}
        impact_statuses = {"EXIT_EXTENSION": "POSITIVE"}
        result = compute_gate_records(rollout, gov_scores, impact_statuses)
        assert result[0].decision == DECISION_REJECTED

    def test_decision_rejected_negative_impact(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout(gov=90.0, impact="NEGATIVE", days=35)}
        gov_scores = {"EXIT_EXTENSION": 90.0}
        impact_statuses = {"EXIT_EXTENSION": "NEGATIVE"}
        result = compute_gate_records(rollout, gov_scores, impact_statuses)
        assert result[0].decision == DECISION_REJECTED

    def test_decision_review_required_days_short(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout(gov=85.0, impact="POSITIVE", days=20)}
        gov_scores = {"EXIT_EXTENSION": 85.0}
        impact_statuses = {"EXIT_EXTENSION": "POSITIVE"}
        result = compute_gate_records(rollout, gov_scores, impact_statuses)
        assert result[0].decision == DECISION_REVIEW_REQUIRED

    def test_risk_score_computed_correctly(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout(gov=84.5)}
        gov_scores = {"EXIT_EXTENSION": 84.5}
        result = compute_gate_records(rollout, gov_scores, {})
        assert result[0].risk_score == pytest.approx(15.5, abs=1e-9)

    def test_governance_score_from_scoreboard_overrides_rollout(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout(gov=70.0, impact="POSITIVE", days=35)}
        gov_scores = {"EXIT_EXTENSION": 90.0}
        impact_statuses = {"EXIT_EXTENSION": "POSITIVE"}
        result = compute_gate_records(rollout, gov_scores, impact_statuses)
        assert result[0].governance_score == pytest.approx(90.0)

    def test_impact_from_impact_file_overrides_rollout(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout(gov=85.0, impact="NEUTRAL", days=35)}
        gov_scores = {"EXIT_EXTENSION": 85.0}
        impact_statuses = {"EXIT_EXTENSION": "POSITIVE"}
        result = compute_gate_records(rollout, gov_scores, impact_statuses)
        assert result[0].impact_status == "POSITIVE"
        assert result[0].decision == DECISION_APPROVED

    def test_candidate_order_preserved(self):
        rollout = {
            ct: self._make_rollout() for ct in _CANDIDATE_ORDER
        }
        result = compute_gate_records(rollout, {}, {})
        names = [r.candidate for r in result]
        assert names == _CANDIDATE_ORDER

    def test_empty_rollout_returns_empty(self):
        result = compute_gate_records({}, {}, {})
        assert result == []

    def test_observe_state_excluded(self):
        rollout = {"EXIT_EXTENSION": self._make_rollout(state="OBSERVE")}
        result = compute_gate_records(rollout, {}, {})
        assert len(result) == 0


# ── TestComputeGateSummary ─────────────────────────────────────────────────────

class TestComputeGateSummary:
    def _make_record(self, candidate, decision, gov_score, risk_score):
        return GateRecord(
            candidate=candidate,
            state=_STATE_PRODUCTION_READY,
            governance_score=gov_score,
            risk_score=risk_score,
            days_in_state=35,
            impact_status="POSITIVE",
            decision=decision,
        )

    def test_empty_records_all_zeros(self):
        summary = compute_gate_summary([])
        assert summary.n_approved == 0
        assert summary.n_review_required == 0
        assert summary.n_rejected == 0
        assert summary.top_candidate == ""
        assert summary.lowest_risk_candidate == ""

    def test_counts_approved(self):
        records = [self._make_record("EXIT_EXTENSION", DECISION_APPROVED, 85.0, 15.0)]
        summary = compute_gate_summary(records)
        assert summary.n_approved == 1
        assert summary.n_review_required == 0
        assert summary.n_rejected == 0

    def test_counts_review_required(self):
        records = [self._make_record("EXIT_EXTENSION", DECISION_REVIEW_REQUIRED, 82.0, 18.0)]
        summary = compute_gate_summary(records)
        assert summary.n_review_required == 1

    def test_counts_rejected(self):
        records = [self._make_record("EXIT_EXTENSION", DECISION_REJECTED, 70.0, 30.0)]
        summary = compute_gate_summary(records)
        assert summary.n_rejected == 1

    def test_top_candidate_is_approved_highest_score(self):
        records = [
            self._make_record("EXIT_EXTENSION", DECISION_APPROVED, 85.0, 15.0),
            self._make_record("CAPITAL_CONCENTRATION", DECISION_APPROVED, 90.0, 10.0),
        ]
        summary = compute_gate_summary(records)
        assert summary.top_candidate == "CAPITAL_CONCENTRATION"

    def test_top_candidate_empty_when_no_approved(self):
        records = [self._make_record("EXIT_EXTENSION", DECISION_REVIEW_REQUIRED, 82.0, 18.0)]
        summary = compute_gate_summary(records)
        assert summary.top_candidate == ""

    def test_lowest_risk_candidate_min_risk(self):
        records = [
            self._make_record("EXIT_EXTENSION", DECISION_APPROVED, 85.0, 15.0),
            self._make_record("CAPITAL_CONCENTRATION", DECISION_REVIEW_REQUIRED, 90.0, 10.0),
        ]
        summary = compute_gate_summary(records)
        assert summary.lowest_risk_candidate == "CAPITAL_CONCENTRATION"

    def test_lowest_risk_empty_when_no_records(self):
        summary = compute_gate_summary([])
        assert summary.lowest_risk_candidate == ""

    def test_mixed_decisions_correct_counts(self):
        records = [
            self._make_record("EXIT_EXTENSION", DECISION_APPROVED, 85.0, 15.0),
            self._make_record("CAPITAL_CONCENTRATION", DECISION_REVIEW_REQUIRED, 82.0, 18.0),
            self._make_record("PRIORITY", DECISION_REJECTED, 70.0, 30.0),
        ]
        summary = compute_gate_summary(records)
        assert summary.n_approved == 1
        assert summary.n_review_required == 1
        assert summary.n_rejected == 1


# ── TestFormatGateReport ───────────────────────────────────────────────────────

class TestFormatGateReport:
    def _make_result(self, records, n_approved=0, top_candidate="", lowest=""):
        return GateRunResult(
            date="2026-05-30",
            gate_records=records,
            summary=GateSummary(
                n_approved=n_approved,
                n_review_required=0,
                n_rejected=0,
                lowest_risk_candidate=lowest,
                top_candidate=top_candidate,
            ),
        )

    def test_empty_records_returns_empty_string(self):
        result = self._make_result([])
        assert format_gate_report(result) == ""

    def test_report_contains_header(self):
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        result = self._make_result(records, n_approved=1, top_candidate="EXIT_EXTENSION", lowest="EXIT_EXTENSION")
        rpt = format_gate_report(result)
        assert "PRODUCTION PROMOTION GATE" in rpt

    def test_report_contains_candidate(self):
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        result = self._make_result(records, n_approved=1, top_candidate="EXIT_EXTENSION")
        rpt = format_gate_report(result)
        assert "EXIT_EXTENSION" in rpt

    def test_report_contains_decision(self):
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        result = self._make_result(records)
        rpt = format_gate_report(result)
        assert "APPROVED" in rpt

    def test_report_contains_gov_score(self):
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        result = self._make_result(records)
        rpt = format_gate_report(result)
        assert "85" in rpt

    def test_report_contains_risk_score(self):
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        result = self._make_result(records)
        rpt = format_gate_report(result)
        assert "15" in rpt

    def test_report_contains_top_candidate(self):
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        result = self._make_result(records, n_approved=1, top_candidate="EXIT_EXTENSION")
        rpt = format_gate_report(result)
        assert "Top Candidate" in rpt

    def test_report_contains_approved_count(self):
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        result = self._make_result(records, n_approved=1)
        rpt = format_gate_report(result)
        assert "Approved" in rpt

    def test_review_required_in_report(self):
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 82.0, 18.0, 20, "POSITIVE", "REVIEW_REQUIRED")]
        result = GateRunResult(
            date="2026-05-30",
            gate_records=records,
            summary=GateSummary(
                n_approved=0, n_review_required=1, n_rejected=0,
                lowest_risk_candidate="EXIT_EXTENSION", top_candidate="",
            ),
        )
        rpt = format_gate_report(result)
        assert "REVIEW_REQUIRED" in rpt

    def test_separator_lines_present(self):
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        result = self._make_result(records)
        rpt = format_gate_report(result)
        assert "═" in rpt


# ── TestAppendGateResult ───────────────────────────────────────────────────────

class TestAppendGateResult:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "gate" / "gate.jsonl"
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        summary = GateSummary(1, 0, 0, "EXIT_EXTENSION", "EXIT_EXTENSION")
        append_gate_result("2026-05-30", records, summary, p)
        assert p.exists()

    def test_appends_single_record(self, tmp_path):
        p = tmp_path / "gate.jsonl"
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        summary = GateSummary(1, 0, 0, "EXIT_EXTENSION", "EXIT_EXTENSION")
        append_gate_result("2026-05-30", records, summary, p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["date"] == "2026-05-30"
        assert data["n_approved"] == 1

    def test_gate_records_embedded_correctly(self, tmp_path):
        p = tmp_path / "gate.jsonl"
        records = [GateRecord("EXIT_EXTENSION", "PRODUCTION_READY", 85.0, 15.0, 35, "POSITIVE", "APPROVED")]
        summary = GateSummary(1, 0, 0, "EXIT_EXTENSION", "EXIT_EXTENSION")
        append_gate_result("2026-05-30", records, summary, p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert len(data["gate_records"]) == 1
        assert data["gate_records"][0]["candidate"] == "EXIT_EXTENSION"

    def test_schema_version_written(self, tmp_path):
        p = tmp_path / "gate.jsonl"
        summary = GateSummary(0, 0, 0, "", "")
        append_gate_result("2026-05-30", [], summary, p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["schema_version"] == SCHEMA_VERSION

    def test_second_append_adds_line(self, tmp_path):
        p = tmp_path / "gate.jsonl"
        summary = GateSummary(0, 0, 0, "", "")
        append_gate_result("2026-05-30", [], summary, p)
        append_gate_result("2026-05-31", [], summary, p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2


# ── TestRunProductionPromotionGate ─────────────────────────────────────────────

class TestRunProductionPromotionGate:
    def _setup_files(self, tmp_path, rollout_records=None, sb_records=None,
                     impact_records=None):
        rollout_file = tmp_path / "rollout.jsonl"
        sb_file      = tmp_path / "sb.jsonl"
        impact_file  = tmp_path / "impact.jsonl"
        out_file     = tmp_path / "gate.jsonl"
        _write_jsonl(rollout_file, rollout_records or [])
        _write_jsonl(sb_file, sb_records or [])
        _write_jsonl(impact_file, impact_records or [])
        return rollout_file, sb_file, impact_file, out_file

    def test_returns_run_result(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path)
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert isinstance(result, GateRunResult)

    def test_no_production_ready_empty_records(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path, rollout_records=[
            _rollout_rec("EXIT_EXTENSION", "MICRO_CAPITAL"),
        ])
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result.gate_records == []

    def test_production_ready_approved(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)],
            sb_records=[_sb_rec("EXIT_EXTENSION", gov_score=85.0)],
            impact_records=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
        )
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert len(result.gate_records) == 1
        assert result.gate_records[0].decision == DECISION_APPROVED

    def test_production_ready_rejected_low_score(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)],
            sb_records=[_sb_rec("EXIT_EXTENSION", gov_score=70.0)],
            impact_records=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
        )
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result.gate_records[0].decision == DECISION_REJECTED

    def test_production_ready_review_days_short(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=20)],
            sb_records=[_sb_rec("EXIT_EXTENSION", gov_score=85.0)],
            impact_records=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
        )
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result.gate_records[0].decision == DECISION_REVIEW_REQUIRED

    def test_appends_to_output_file(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path)
        run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert of.exists()

    def test_idempotent_same_day(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)],
            sb_records=[_sb_rec("EXIT_EXTENSION", gov_score=85.0)],
            impact_records=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
        )
        run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        lines = [l for l in of.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_idempotent_returns_skipped(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path)
        run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        result2 = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result2.skipped is True

    def test_different_days_create_two_records(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)],
            sb_records=[_sb_rec("EXIT_EXTENSION", gov_score=85.0)],
            impact_records=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
        )
        run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        run_production_promotion_gate(rf, sf, imf, of, "2026-05-31")
        lines = [l for l in of.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_summary_counts_correct(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)],
            sb_records=[_sb_rec("EXIT_EXTENSION", gov_score=85.0)],
            impact_records=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
        )
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result.summary.n_approved == 1
        assert result.summary.n_review_required == 0
        assert result.summary.n_rejected == 0

    def test_top_candidate_approved(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)],
            sb_records=[_sb_rec("EXIT_EXTENSION", gov_score=85.0)],
            impact_records=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
        )
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result.summary.top_candidate == "EXIT_EXTENSION"

    def test_not_skipped_on_first_run(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path)
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result.skipped is False

    def test_fail_open_bad_rollout_file(self, tmp_path):
        rf = tmp_path / "broken.jsonl"
        sf = tmp_path / "sb.jsonl"
        imf = tmp_path / "impact.jsonl"
        of  = tmp_path / "gate.jsonl"
        rf.write_text("not-json\nnot-json\n", encoding="utf-8")
        _write_jsonl(sf, [])
        _write_jsonl(imf, [])
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert isinstance(result, GateRunResult)

    def test_negative_impact_rejected(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)],
            sb_records=[_sb_rec("EXIT_EXTENSION", gov_score=90.0)],
            impact_records=[_impact_rec("EXIT_EXTENSION", "NEGATIVE")],
        )
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result.gate_records[0].decision == DECISION_REJECTED

    def test_multiple_production_ready_candidates(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[
                _rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35, gov_score=85.0),
                _rollout_rec("CAPITAL_CONCENTRATION", "PRODUCTION_READY", days_in_state=35, gov_score=90.0),
            ],
            sb_records=[
                _sb_rec("EXIT_EXTENSION", gov_score=85.0),
                _sb_rec("CAPITAL_CONCENTRATION", gov_score=90.0),
            ],
            impact_records=[
                _impact_rec("EXIT_EXTENSION", "POSITIVE"),
                _impact_rec("CAPITAL_CONCENTRATION", "POSITIVE"),
            ],
        )
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert len(result.gate_records) == 2
        assert result.summary.n_approved == 2
        assert result.summary.top_candidate == "CAPITAL_CONCENTRATION"

    def test_lowest_risk_candidate_set(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[
                _rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35, gov_score=85.0),
                _rollout_rec("CAPITAL_CONCENTRATION", "PRODUCTION_READY", days_in_state=35, gov_score=92.0),
            ],
            sb_records=[
                _sb_rec("EXIT_EXTENSION", gov_score=85.0),
                _sb_rec("CAPITAL_CONCENTRATION", gov_score=92.0),
            ],
            impact_records=[
                _impact_rec("EXIT_EXTENSION", "POSITIVE"),
                _impact_rec("CAPITAL_CONCENTRATION", "POSITIVE"),
            ],
        )
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result.summary.lowest_risk_candidate == "CAPITAL_CONCENTRATION"

    def test_idempotent_result_has_gate_records(self, tmp_path):
        rf, sf, imf, of = self._setup_files(tmp_path,
            rollout_records=[_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)],
            sb_records=[_sb_rec("EXIT_EXTENSION", gov_score=85.0)],
            impact_records=[_impact_rec("EXIT_EXTENSION", "POSITIVE")],
        )
        run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        result2 = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result2.summary.n_approved == 1

    def test_missing_all_files_returns_empty_result(self, tmp_path):
        rf  = tmp_path / "missing_rollout.jsonl"
        sf  = tmp_path / "missing_sb.jsonl"
        imf = tmp_path / "missing_impact.jsonl"
        of  = tmp_path / "gate.jsonl"
        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert isinstance(result, GateRunResult)
        assert result.gate_records == []


# ── TestIntegration ────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_approved_flow(self, tmp_path):
        rf  = tmp_path / "rollout.jsonl"
        sf  = tmp_path / "sb.jsonl"
        imf = tmp_path / "impact.jsonl"
        of  = tmp_path / "gate.jsonl"

        _write_jsonl(rf, [_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY",
                                       days_in_state=35, gov_score=85.0,
                                       impact="POSITIVE")])
        _write_jsonl(sf, [_sb_rec("EXIT_EXTENSION", gov_score=85.0)])
        _write_jsonl(imf, [_impact_rec("EXIT_EXTENSION", "POSITIVE")])

        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")

        assert result.gate_records[0].decision == DECISION_APPROVED
        assert result.summary.n_approved == 1
        assert result.summary.top_candidate == "EXIT_EXTENSION"
        assert result.summary.lowest_risk_candidate == "EXIT_EXTENSION"
        assert not result.skipped

        # Verify JSONL content
        data = json.loads(of.read_text(encoding="utf-8").strip())
        assert data["n_approved"] == 1
        assert data["top_candidate"] == "EXIT_EXTENSION"

    def test_full_rejected_flow(self, tmp_path):
        rf  = tmp_path / "rollout.jsonl"
        sf  = tmp_path / "sb.jsonl"
        imf = tmp_path / "impact.jsonl"
        of  = tmp_path / "gate.jsonl"

        _write_jsonl(rf, [_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY",
                                       days_in_state=35, gov_score=90.0,
                                       impact="NEGATIVE")])
        _write_jsonl(sf, [_sb_rec("EXIT_EXTENSION", gov_score=90.0)])
        _write_jsonl(imf, [_impact_rec("EXIT_EXTENSION", "NEGATIVE")])

        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")

        assert result.gate_records[0].decision == DECISION_REJECTED
        assert result.summary.n_rejected == 1
        assert result.summary.top_candidate == ""

    def test_multi_day_accumulation(self, tmp_path):
        rf  = tmp_path / "rollout.jsonl"
        sf  = tmp_path / "sb.jsonl"
        imf = tmp_path / "impact.jsonl"
        of  = tmp_path / "gate.jsonl"

        _write_jsonl(rf, [_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)])
        _write_jsonl(sf, [_sb_rec("EXIT_EXTENSION", gov_score=85.0)])
        _write_jsonl(imf, [_impact_rec("EXIT_EXTENSION", "POSITIVE")])

        for day in ["2026-05-30", "2026-05-31", "2026-06-01"]:
            run_production_promotion_gate(rf, sf, imf, of, day)

        lines = [l for l in of.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3

    def test_report_format_non_empty(self, tmp_path):
        rf  = tmp_path / "rollout.jsonl"
        sf  = tmp_path / "sb.jsonl"
        imf = tmp_path / "impact.jsonl"
        of  = tmp_path / "gate.jsonl"

        _write_jsonl(rf, [_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY", days_in_state=35)])
        _write_jsonl(sf, [_sb_rec("EXIT_EXTENSION", gov_score=85.0)])
        _write_jsonl(imf, [_impact_rec("EXIT_EXTENSION", "POSITIVE")])

        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        rpt = format_gate_report(result)
        assert len(rpt) > 0
        assert "PRODUCTION PROMOTION GATE" in rpt
        assert "EXIT_EXTENSION" in rpt

    def test_paths_constants_importable(self):
        from src.paths import (
            PRODUCTION_PROMOTION_GATE_DIR,
            PRODUCTION_PROMOTION_GATE_FILE,
        )
        assert PRODUCTION_PROMOTION_GATE_DIR.name == "production_promotion_gate"
        assert PRODUCTION_PROMOTION_GATE_FILE.suffix == ".jsonl"

    def test_no_execution_imports(self):
        import importlib, sys
        for mod_name in list(sys.modules.keys()):
            if "production_promotion_gate" in mod_name:
                del sys.modules[mod_name]
        import src.analytics.production_promotion_gate as m
        src_text = Path(m.__file__).read_text(encoding="utf-8")
        assert "signal_bridge" not in src_text
        assert "kabu" not in src_text.lower()
        assert "place_order" not in src_text

    def test_review_required_flow(self, tmp_path):
        rf  = tmp_path / "rollout.jsonl"
        sf  = tmp_path / "sb.jsonl"
        imf = tmp_path / "impact.jsonl"
        of  = tmp_path / "gate.jsonl"

        _write_jsonl(rf, [_rollout_rec("EXIT_EXTENSION", "PRODUCTION_READY",
                                       days_in_state=20, gov_score=85.0,
                                       impact="POSITIVE")])
        _write_jsonl(sf, [_sb_rec("EXIT_EXTENSION", gov_score=85.0)])
        _write_jsonl(imf, [_impact_rec("EXIT_EXTENSION", "POSITIVE")])

        result = run_production_promotion_gate(rf, sf, imf, of, "2026-05-30")
        assert result.gate_records[0].decision == DECISION_REVIEW_REQUIRED
        assert result.summary.n_review_required == 1
        assert result.summary.top_candidate == ""
