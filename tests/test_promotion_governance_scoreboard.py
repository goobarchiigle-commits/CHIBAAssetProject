"""Tests for src/analytics/promotion_governance_scoreboard.py (Phase 6G-Lite)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.analytics.promotion_governance_scoreboard import (
    CLASSIFICATION_OBSERVE,
    CLASSIFICATION_PROMOTE_FIRST,
    CLASSIFICATION_PROMOTE_LATER,
    SCHEMA_VERSION,
    SCORE_PROMOTE_FIRST_THRESHOLD,
    SCORE_PROMOTE_LATER_THRESHOLD,
    ScoreboardRecord,
    ScoreboardRunResult,
    ScoreboardSummary,
    _CANDIDATE_ORDER,
    _REGISTRY_STATUS_ACTIVE,
    _W_EVIDENCE,
    _W_IMPACT,
    _W_READINESS,
    _W_SUCCESS,
    _active_candidates,
    _classify,
    _compute_summary,
    _latest_per_candidate,
    _load_all_jsonl,
    _load_evaluated_keys,
    _load_latest_scoreboard,
    _normalize_impact,
    _success_rate_per_candidate,
    append_scoreboard_record,
    compute_governance_score,
    format_scoreboard_report,
    run_promotion_governance_scoreboard,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _readiness_rec(ct: str, score: float, date: str = "2026-07-20") -> dict:
    return {"date": date, "candidate": ct, "readiness_score": score,
            "success_rate": 75.0, "evidence_score": 70.0, "schema_version": "v1"}


def _evidence_rec(ct: str, score: int, date: str = "2026-07-20") -> dict:
    return {"date": date, "candidate": ct, "evidence_score": score,
            "schema_version": "v1"}


def _registry_rec(ct: str, status: str = "ACTIVE", date: str = "2026-07-20") -> dict:
    return {"date": date, "candidate": ct, "registry_status": status,
            "readiness_score": 80.0, "schema_version": "v1"}


def _shadow_rec(ct: str, success: bool, date: str = "2026-07-20") -> dict:
    return {"date": date, "candidate": ct, "success": success, "schema_version": "v1"}


def _impact_rec(ct: str, delta: float, date: str = "2026-07-20") -> dict:
    return {"date": date, "candidate": ct, "impact_delta": delta,
            "impact_status": "POSITIVE" if delta > 0 else "NEGATIVE",
            "schema_version": "v1"}


def _make_empty_files(tmp_path: Path) -> dict:
    """Return mapping of empty source file paths."""
    files = {}
    for name in ["readiness", "registry", "impact", "shadow", "evidence", "output"]:
        p = tmp_path / f"{name}.jsonl"
        p.write_text("", encoding="utf-8")
        files[name] = p
    return files


def _run_scoreboard(tmp_path, readiness=None, registry=None, impact=None,
                    shadow=None, evidence=None, existing_output=None,
                    today="2026-07-25") -> tuple:
    files = _make_empty_files(tmp_path)
    if readiness:
        _write_jsonl(files["readiness"], readiness)
    if registry:
        _write_jsonl(files["registry"], registry)
    if impact:
        _write_jsonl(files["impact"], impact)
    if shadow:
        _write_jsonl(files["shadow"], shadow)
    if evidence:
        _write_jsonl(files["evidence"], evidence)
    if existing_output:
        _write_jsonl(files["output"], existing_output)
    result = run_promotion_governance_scoreboard(
        readiness_file=files["readiness"],
        registry_file=files["registry"],
        impact_file=files["impact"],
        shadow_file=files["shadow"],
        evidence_file=files["evidence"],
        output_file=files["output"],
        today=today,
    )
    return result, files["output"]


# ── TestConstants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_promote_first(self):
        assert CLASSIFICATION_PROMOTE_FIRST == "PROMOTE_FIRST"

    def test_promote_later(self):
        assert CLASSIFICATION_PROMOTE_LATER == "PROMOTE_LATER"

    def test_observe(self):
        assert CLASSIFICATION_OBSERVE == "OBSERVE"

    def test_schema_version(self):
        assert SCHEMA_VERSION == "v1"

    def test_promote_first_threshold(self):
        assert SCORE_PROMOTE_FIRST_THRESHOLD == 80.0

    def test_promote_later_threshold(self):
        assert SCORE_PROMOTE_LATER_THRESHOLD == 60.0

    def test_weights_sum_to_one(self):
        assert _W_READINESS + _W_EVIDENCE + _W_SUCCESS + _W_IMPACT == pytest.approx(1.0)

    def test_candidate_order_length(self):
        assert len(_CANDIDATE_ORDER) == 5

    def test_candidate_order(self):
        assert _CANDIDATE_ORDER == [
            "EXIT_EXTENSION",
            "CAPITAL_CONCENTRATION",
            "PRIORITY",
            "SIGNAL_LEADER",
            "CONTINUATION_SIGNAL",
        ]

    def test_registry_active(self):
        assert _REGISTRY_STATUS_ACTIVE == "ACTIVE"


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
        f.write_text('{"ok": 1}\nnot_json\n', encoding="utf-8")
        assert len(_load_all_jsonl(f)) == 1

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "blanks.jsonl"
        f.write_text('\n{"x": 1}\n\n', encoding="utf-8")
        assert len(_load_all_jsonl(f)) == 1


# ── TestNormalizeImpact ───────────────────────────────────────────────────────

class TestNormalizeImpact:
    def test_positive_delta(self):
        assert _normalize_impact(0.5) == pytest.approx(50.0)

    def test_zero_delta(self):
        assert _normalize_impact(0.0) == pytest.approx(0.0)

    def test_negative_clamped_to_zero(self):
        assert _normalize_impact(-0.5) == pytest.approx(0.0)

    def test_large_delta_clamped_to_100(self):
        assert _normalize_impact(5.0) == pytest.approx(100.0)

    def test_exactly_one(self):
        assert _normalize_impact(1.0) == pytest.approx(100.0)

    def test_fractional(self):
        assert _normalize_impact(0.25) == pytest.approx(25.0)

    def test_very_small_positive(self):
        assert _normalize_impact(0.001) == pytest.approx(0.1)


# ── TestComputeGovernanceScore ────────────────────────────────────────────────

class TestComputeGovernanceScore:
    def test_all_max(self):
        score = compute_governance_score(100.0, 100.0, 1.0, 1.0)
        assert score == pytest.approx(100.0)

    def test_all_zero(self):
        score = compute_governance_score(0.0, 0.0, 0.0, 0.0)
        assert score == pytest.approx(0.0)

    def test_only_readiness(self):
        # 0.35 * 100 = 35
        score = compute_governance_score(100.0, 0.0, 0.0, 0.0)
        assert score == pytest.approx(35.0)

    def test_only_evidence(self):
        # 0.25 * 100 = 25
        score = compute_governance_score(0.0, 100.0, 0.0, 0.0)
        assert score == pytest.approx(25.0)

    def test_only_success_rate(self):
        # 0.20 * (1.0 * 100) = 20
        score = compute_governance_score(0.0, 0.0, 1.0, 0.0)
        assert score == pytest.approx(20.0)

    def test_only_impact(self):
        # 0.20 * clamp(1.0*100, 0, 100) = 20
        score = compute_governance_score(0.0, 0.0, 0.0, 1.0)
        assert score == pytest.approx(20.0)

    def test_weights_sum(self):
        # With all at 100: 0.35*100 + 0.25*100 + 0.20*100 + 0.20*100 = 100
        score = compute_governance_score(100.0, 100.0, 1.0, 1.0)
        assert score == pytest.approx(100.0, abs=0.01)

    def test_negative_impact_clamped(self):
        # negative impact → normalized to 0
        score = compute_governance_score(0.0, 0.0, 0.0, -1.0)
        assert score == pytest.approx(0.0)

    def test_promote_first_threshold_achievable(self):
        # achieve ≥ 80
        score = compute_governance_score(90.0, 90.0, 0.9, 0.8)
        assert score >= 80.0

    def test_result_clamped_to_100(self):
        score = compute_governance_score(200.0, 200.0, 2.0, 10.0)
        assert score <= 100.0

    def test_result_nonnegative(self):
        score = compute_governance_score(-50.0, -50.0, -1.0, -5.0)
        assert score >= 0.0


# ── TestClassify ──────────────────────────────────────────────────────────────

class TestClassify:
    def test_above_80(self):
        assert _classify(85.0) == CLASSIFICATION_PROMOTE_FIRST

    def test_exactly_80(self):
        assert _classify(80.0) == CLASSIFICATION_PROMOTE_FIRST

    def test_between_60_and_80(self):
        assert _classify(70.0) == CLASSIFICATION_PROMOTE_LATER

    def test_exactly_60(self):
        assert _classify(60.0) == CLASSIFICATION_PROMOTE_LATER

    def test_below_60(self):
        assert _classify(50.0) == CLASSIFICATION_OBSERVE

    def test_zero(self):
        assert _classify(0.0) == CLASSIFICATION_OBSERVE

    def test_100(self):
        assert _classify(100.0) == CLASSIFICATION_PROMOTE_FIRST

    def test_just_below_60(self):
        assert _classify(59.9) == CLASSIFICATION_OBSERVE

    def test_just_below_80(self):
        assert _classify(79.9) == CLASSIFICATION_PROMOTE_LATER


# ── TestLatestPerCandidate ────────────────────────────────────────────────────

class TestLatestPerCandidate:
    def test_empty_records(self):
        assert _latest_per_candidate([], "readiness_score") == {}

    def test_single_record(self, tmp_path):
        recs = [{"date": "2026-07-20", "candidate": "EXIT_EXTENSION", "readiness_score": 85.0}]
        result = _latest_per_candidate(recs, "readiness_score")
        assert result["EXIT_EXTENSION"] == pytest.approx(85.0)

    def test_latest_date_selected(self):
        recs = [
            {"date": "2026-07-10", "candidate": "EXIT_EXTENSION", "readiness_score": 70.0},
            {"date": "2026-07-20", "candidate": "EXIT_EXTENSION", "readiness_score": 85.0},
        ]
        result = _latest_per_candidate(recs, "readiness_score")
        assert result["EXIT_EXTENSION"] == pytest.approx(85.0)

    def test_unknown_candidate_excluded(self):
        recs = [{"date": "2026-07-20", "candidate": "UNKNOWN", "readiness_score": 80.0}]
        assert _latest_per_candidate(recs, "readiness_score") == {}

    def test_missing_field_excluded(self):
        recs = [{"date": "2026-07-20", "candidate": "EXIT_EXTENSION", "other": 80.0}]
        assert _latest_per_candidate(recs, "readiness_score") == {}

    def test_multiple_candidates(self):
        recs = [
            {"date": "2026-07-20", "candidate": "EXIT_EXTENSION",        "evidence_score": 80},
            {"date": "2026-07-20", "candidate": "CAPITAL_CONCENTRATION", "evidence_score": 75},
        ]
        result = _latest_per_candidate(recs, "evidence_score")
        assert result["EXIT_EXTENSION"] == pytest.approx(80.0)
        assert result["CAPITAL_CONCENTRATION"] == pytest.approx(75.0)


# ── TestActiveCandidates ──────────────────────────────────────────────────────

class TestActiveCandidates:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        f.write_text("", encoding="utf-8")
        assert _active_candidates(f) == set()

    def test_active_included(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_registry_rec("EXIT_EXTENSION")])
        assert "EXIT_EXTENSION" in _active_candidates(f)

    def test_non_active_excluded(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_registry_rec("EXIT_EXTENSION", status="INACTIVE")])
        assert _active_candidates(f) == set()

    def test_multiple_active(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [_registry_rec(ct) for ct in _CANDIDATE_ORDER])
        result = _active_candidates(f)
        assert result == set(_CANDIDATE_ORDER)

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "reg.jsonl"
        _write_jsonl(f, [{"date": "2026-07-20", "candidate": "UNKNOWN", "registry_status": "ACTIVE"}])
        assert _active_candidates(f) == set()


# ── TestSuccessRatePerCandidate ───────────────────────────────────────────────

class TestSuccessRatePerCandidate:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        f.write_text("", encoding="utf-8")
        assert _success_rate_per_candidate(f) == {}

    def test_all_success(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_shadow_rec("EXIT_EXTENSION", True) for _ in range(3)])
        result = _success_rate_per_candidate(f)
        assert result["EXIT_EXTENSION"] == pytest.approx(1.0)

    def test_no_success(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [_shadow_rec("EXIT_EXTENSION", False) for _ in range(3)])
        result = _success_rate_per_candidate(f)
        assert result["EXIT_EXTENSION"] == pytest.approx(0.0)

    def test_half_success(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [
            _shadow_rec("EXIT_EXTENSION", True),
            _shadow_rec("EXIT_EXTENSION", False),
        ])
        result = _success_rate_per_candidate(f)
        assert result["EXIT_EXTENSION"] == pytest.approx(0.5)

    def test_multiple_candidates(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [
            _shadow_rec("EXIT_EXTENSION", True),
            _shadow_rec("PRIORITY", False),
        ])
        result = _success_rate_per_candidate(f)
        assert result["EXIT_EXTENSION"] == pytest.approx(1.0)
        assert result["PRIORITY"] == pytest.approx(0.0)

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "shadow.jsonl"
        _write_jsonl(f, [{"date": "2026-07-20", "candidate": "UNKNOWN", "success": True}])
        assert _success_rate_per_candidate(f) == {}

    def test_missing_file(self, tmp_path):
        f = tmp_path / "missing.jsonl"
        assert _success_rate_per_candidate(f) == {}


# ── TestLoadEvaluatedKeys ─────────────────────────────────────────────────────

class TestLoadEvaluatedKeys:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "out.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_evaluated_keys(f) == set()

    def test_keys_loaded(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [{"date": "2026-07-25", "candidate": "EXIT_EXTENSION"}])
        assert ("2026-07-25", "EXIT_EXTENSION") in _load_evaluated_keys(f)

    def test_empty_date_excluded(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [{"date": "", "candidate": "EXIT_EXTENSION"}])
        assert _load_evaluated_keys(f) == set()

    def test_empty_candidate_excluded(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [{"date": "2026-07-25", "candidate": ""}])
        assert _load_evaluated_keys(f) == set()


# ── TestLoadLatestScoreboard ──────────────────────────────────────────────────

class TestLoadLatestScoreboard:
    def _sb_rec(self, ct, score, rank, cls, date="2026-07-25"):
        return {
            "date": date, "candidate": ct,
            "readiness_score": 80.0, "evidence_score": 75.0,
            "success_rate": 0.8, "impact_delta": 0.1,
            "governance_score": score, "rank": rank,
            "classification": cls, "schema_version": "v1",
        }

    def test_empty_file(self, tmp_path):
        f = tmp_path / "out.jsonl"
        f.write_text("", encoding="utf-8")
        assert _load_latest_scoreboard(f) == []

    def test_records_loaded(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [self._sb_rec("EXIT_EXTENSION", 85.0, 1, CLASSIFICATION_PROMOTE_FIRST)])
        result = _load_latest_scoreboard(f)
        assert len(result) == 1
        assert result[0].candidate == "EXIT_EXTENSION"

    def test_sorted_by_rank(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [
            self._sb_rec("PRIORITY",        70.0, 2, CLASSIFICATION_PROMOTE_LATER),
            self._sb_rec("EXIT_EXTENSION",  85.0, 1, CLASSIFICATION_PROMOTE_FIRST),
        ])
        result = _load_latest_scoreboard(f)
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_latest_per_candidate(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [
            self._sb_rec("EXIT_EXTENSION", 70.0, 2, CLASSIFICATION_PROMOTE_LATER, date="2026-07-24"),
            self._sb_rec("EXIT_EXTENSION", 85.0, 1, CLASSIFICATION_PROMOTE_FIRST, date="2026-07-25"),
        ])
        result = _load_latest_scoreboard(f)
        assert len(result) == 1
        assert result[0].governance_score == pytest.approx(85.0)

    def test_unknown_candidate_excluded(self, tmp_path):
        f = tmp_path / "out.jsonl"
        _write_jsonl(f, [{"date": "2026-07-25", "candidate": "UNKNOWN", "rank": 1,
                          "governance_score": 90.0, "classification": "PROMOTE_FIRST"}])
        assert _load_latest_scoreboard(f) == []


# ── TestComputeSummary ────────────────────────────────────────────────────────

class TestComputeSummary:
    def _rec(self, ct, score, cls, rank):
        return ScoreboardRecord(
            date="2026-07-25", candidate=ct,
            readiness_score=80.0, evidence_score=75.0,
            success_rate=0.8, impact_delta=0.1,
            governance_score=score, rank=rank, classification=cls,
        )

    def test_empty(self):
        s = _compute_summary([])
        assert s.top_candidate == ""
        assert s.top_governance_score == 0.0
        assert s.n_promote_first == 0

    def test_top_candidate_is_rank_1(self):
        records = [
            self._rec("EXIT_EXTENSION",       85.0, CLASSIFICATION_PROMOTE_FIRST, 1),
            self._rec("CAPITAL_CONCENTRATION", 70.0, CLASSIFICATION_PROMOTE_LATER, 2),
        ]
        s = _compute_summary(records)
        assert s.top_candidate == "EXIT_EXTENSION"

    def test_n_promote_first_counts(self):
        records = [
            self._rec("EXIT_EXTENSION",        85.0, CLASSIFICATION_PROMOTE_FIRST, 1),
            self._rec("CAPITAL_CONCENTRATION",  82.0, CLASSIFICATION_PROMOTE_FIRST, 2),
            self._rec("PRIORITY",               70.0, CLASSIFICATION_PROMOTE_LATER, 3),
        ]
        s = _compute_summary(records)
        assert s.n_promote_first == 2
        assert s.n_promote_later == 1

    def test_n_observe_counts(self):
        records = [self._rec("EXIT_EXTENSION", 40.0, CLASSIFICATION_OBSERVE, 1)]
        s = _compute_summary(records)
        assert s.n_observe == 1

    def test_recommendation_when_promote_first(self):
        records = [self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)]
        s = _compute_summary(records)
        assert "PROMOTE" in s.promotion_recommendation
        assert "EXIT_EXTENSION" in s.promotion_recommendation

    def test_no_recommendation_when_only_promote_later(self):
        records = [self._rec("EXIT_EXTENSION", 70.0, CLASSIFICATION_PROMOTE_LATER, 1)]
        s = _compute_summary(records)
        assert s.promotion_recommendation == ""

    def test_top_score_preserved(self):
        records = [self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)]
        s = _compute_summary(records)
        assert s.top_governance_score == pytest.approx(85.0)


# ── TestFormatReport ──────────────────────────────────────────────────────────

class TestFormatReport:
    def _rec(self, ct, score, cls, rank):
        return ScoreboardRecord(
            date="2026-07-25", candidate=ct,
            readiness_score=80.0, evidence_score=75.0,
            success_rate=0.8, impact_delta=0.1,
            governance_score=score, rank=rank, classification=cls,
        )

    def test_empty_returns_empty(self):
        s = ScoreboardSummary("", 0.0, 0, 0, 0, "")
        assert format_scoreboard_report([], s) == ""

    def test_contains_header(self):
        r = self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)
        s = _compute_summary([r])
        rpt = format_scoreboard_report([r], s)
        assert "PROMOTION GOVERNANCE SCOREBOARD" in rpt

    def test_contains_separator(self):
        r = self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)
        s = _compute_summary([r])
        rpt = format_scoreboard_report([r], s)
        assert "═" in rpt

    def test_contains_rank(self):
        r = self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)
        s = _compute_summary([r])
        rpt = format_scoreboard_report([r], s)
        assert "#1" in rpt

    def test_contains_candidate(self):
        r = self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)
        s = _compute_summary([r])
        rpt = format_scoreboard_report([r], s)
        assert "EXIT_EXTENSION" in rpt

    def test_contains_classification(self):
        r = self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)
        s = _compute_summary([r])
        rpt = format_scoreboard_report([r], s)
        assert CLASSIFICATION_PROMOTE_FIRST in rpt

    def test_contains_top_candidate(self):
        r = self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)
        s = _compute_summary([r])
        rpt = format_scoreboard_report([r], s)
        assert "Top Candidate" in rpt

    def test_recommendation_shown(self):
        r = self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)
        s = _compute_summary([r])
        rpt = format_scoreboard_report([r], s)
        assert "Promotion Recommendation" in rpt

    def test_no_recommendation_when_observe(self):
        r = self._rec("EXIT_EXTENSION", 40.0, CLASSIFICATION_OBSERVE, 1)
        s = _compute_summary([r])
        rpt = format_scoreboard_report([r], s)
        assert "Promotion Recommendation" not in rpt

    def test_counts_in_footer(self):
        r = self._rec("EXIT_EXTENSION", 85.0, CLASSIFICATION_PROMOTE_FIRST, 1)
        s = _compute_summary([r])
        rpt = format_scoreboard_report([r], s)
        assert "PROMOTE_FIRST" in rpt
        assert "OBSERVE" in rpt

    def test_all_five_candidates(self):
        records = [
            self._rec("EXIT_EXTENSION",        85.0, CLASSIFICATION_PROMOTE_FIRST, 1),
            self._rec("CAPITAL_CONCENTRATION",  75.0, CLASSIFICATION_PROMOTE_LATER, 2),
            self._rec("PRIORITY",               65.0, CLASSIFICATION_PROMOTE_LATER, 3),
            self._rec("SIGNAL_LEADER",          55.0, CLASSIFICATION_OBSERVE,       4),
            self._rec("CONTINUATION_SIGNAL",    45.0, CLASSIFICATION_OBSERVE,       5),
        ]
        s = _compute_summary(records)
        rpt = format_scoreboard_report(records, s)
        for ct in _CANDIDATE_ORDER:
            assert ct in rpt


# ── TestAppendScoreboardRecord ────────────────────────────────────────────────

class TestAppendScoreboardRecord:
    def _make_record(self):
        return ScoreboardRecord(
            date="2026-07-25",
            candidate="EXIT_EXTENSION",
            readiness_score=85.0,
            evidence_score=80.0,
            success_rate=0.9,
            impact_delta=0.2,
            governance_score=84.5,
            rank=1,
            classification=CLASSIFICATION_PROMOTE_FIRST,
        )

    def test_creates_file(self, tmp_path):
        path = tmp_path / "out" / "scoreboard.jsonl"
        append_scoreboard_record(self._make_record(), path)
        assert path.exists()

    def test_all_fields_present(self, tmp_path):
        path = tmp_path / "scoreboard.jsonl"
        append_scoreboard_record(self._make_record(), path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["date"] == "2026-07-25"
        assert data["candidate"] == "EXIT_EXTENSION"
        assert data["readiness_score"] == pytest.approx(85.0)
        assert data["evidence_score"] == pytest.approx(80.0)
        assert data["success_rate"] == pytest.approx(0.9)
        assert data["impact_delta"] == pytest.approx(0.2)
        assert data["governance_score"] == pytest.approx(84.5)
        assert data["rank"] == 1
        assert data["classification"] == CLASSIFICATION_PROMOTE_FIRST
        assert data["schema_version"] == "v1"

    def test_appends_multiple(self, tmp_path):
        path = tmp_path / "scoreboard.jsonl"
        for rank, ct in enumerate(_CANDIDATE_ORDER[:3], start=1):
            rec = ScoreboardRecord(
                "2026-07-25", ct, 80.0, 75.0, 0.8, 0.1,
                80.0 - rank * 5, rank, CLASSIFICATION_PROMOTE_FIRST,
            )
            append_scoreboard_record(rec, path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "dir" / "scoreboard.jsonl"
        append_scoreboard_record(self._make_record(), path)
        assert path.exists()


# ── TestRunPromotionGovernanceScoreboard ──────────────────────────────────────

class TestRunPromotionGovernanceScoreboard:
    def test_empty_inputs_no_new(self, tmp_path):
        result, _ = _run_scoreboard(tmp_path)
        # all metrics default to 0 → still creates records for all 5 candidates
        assert len(result.new_records) == 5

    def test_single_candidate_scored(self, tmp_path):
        result, _ = _run_scoreboard(
            tmp_path,
            readiness=[_readiness_rec("EXIT_EXTENSION", 85.0)],
            evidence=[_evidence_rec("EXIT_EXTENSION", 80)],
            shadow=[_shadow_rec("EXIT_EXTENSION", True)] * 4,
            impact=[_impact_rec("EXIT_EXTENSION", 0.2)],
        )
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.readiness_score == pytest.approx(85.0)
        assert ee.evidence_score == pytest.approx(80.0)
        assert ee.success_rate == pytest.approx(1.0)
        assert ee.impact_delta == pytest.approx(0.2)

    def test_governance_score_computed(self, tmp_path):
        result, _ = _run_scoreboard(
            tmp_path,
            readiness=[_readiness_rec("EXIT_EXTENSION", 100.0)],
            evidence=[_evidence_rec("EXIT_EXTENSION", 100)],
            shadow=[_shadow_rec("EXIT_EXTENSION", True)],
            impact=[_impact_rec("EXIT_EXTENSION", 1.0)],
        )
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.governance_score == pytest.approx(100.0)

    def test_classification_promote_first(self, tmp_path):
        result, _ = _run_scoreboard(
            tmp_path,
            readiness=[_readiness_rec("EXIT_EXTENSION", 100.0)],
            evidence=[_evidence_rec("EXIT_EXTENSION", 100)],
            shadow=[_shadow_rec("EXIT_EXTENSION", True)],
            impact=[_impact_rec("EXIT_EXTENSION", 1.0)],
        )
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        assert ee.classification == CLASSIFICATION_PROMOTE_FIRST

    def test_classification_observe(self, tmp_path):
        result, _ = _run_scoreboard(tmp_path)
        # all zeros → OBSERVE
        for r in result.new_records:
            assert r.classification == CLASSIFICATION_OBSERVE

    def test_ranking_descending(self, tmp_path):
        result, _ = _run_scoreboard(
            tmp_path,
            readiness=[
                _readiness_rec("EXIT_EXTENSION",        90.0),
                _readiness_rec("CAPITAL_CONCENTRATION", 70.0),
            ],
        )
        ranked = result.ranked_records
        for i in range(len(ranked) - 1):
            assert ranked[i].governance_score >= ranked[i + 1].governance_score

    def test_rank_starts_at_1(self, tmp_path):
        result, _ = _run_scoreboard(tmp_path)
        assert any(r.rank == 1 for r in result.new_records)

    def test_idempotent_same_day(self, tmp_path):
        result1, out = _run_scoreboard(tmp_path)
        # run again same day
        files = {
            "readiness": tmp_path / "readiness.jsonl",
            "registry":  tmp_path / "registry.jsonl",
            "impact":    tmp_path / "impact.jsonl",
            "shadow":    tmp_path / "shadow.jsonl",
            "evidence":  tmp_path / "evidence.jsonl",
        }
        result2 = run_promotion_governance_scoreboard(
            readiness_file=files["readiness"],
            registry_file=files["registry"],
            impact_file=files["impact"],
            shadow_file=files["shadow"],
            evidence_file=files["evidence"],
            output_file=out,
            today="2026-07-25",
        )
        assert result2.new_records == []
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

    def test_all_five_candidates_processed(self, tmp_path):
        result, _ = _run_scoreboard(tmp_path)
        assert len(result.new_records) == 5

    def test_summary_top_candidate(self, tmp_path):
        result, _ = _run_scoreboard(
            tmp_path,
            readiness=[
                _readiness_rec("EXIT_EXTENSION",        90.0),
                _readiness_rec("CAPITAL_CONCENTRATION", 60.0),
            ],
        )
        # EXIT_EXTENSION should score higher
        assert result.summary.top_candidate == "EXIT_EXTENSION"

    def test_summary_n_promote_first(self, tmp_path):
        result, _ = _run_scoreboard(
            tmp_path,
            readiness=[_readiness_rec("EXIT_EXTENSION", 100.0)],
            evidence=[_evidence_rec("EXIT_EXTENSION", 100)],
            shadow=[_shadow_rec("EXIT_EXTENSION", True)],
            impact=[_impact_rec("EXIT_EXTENSION", 1.0)],
        )
        assert result.summary.n_promote_first >= 1

    def test_fail_open_on_exception(self, tmp_path):
        # Pass non-existent paths for all inputs
        out = tmp_path / "out.jsonl"
        result = run_promotion_governance_scoreboard(
            readiness_file=tmp_path / "nx1.jsonl",
            registry_file=tmp_path / "nx2.jsonl",
            impact_file=tmp_path / "nx3.jsonl",
            shadow_file=tmp_path / "nx4.jsonl",
            evidence_file=tmp_path / "nx5.jsonl",
            output_file=out,
            today="2026-07-25",
        )
        assert isinstance(result, ScoreboardRunResult)

    def test_result_date(self, tmp_path):
        result, _ = _run_scoreboard(tmp_path, today="2026-08-01")
        assert result.date == "2026-08-01"

    def test_schema_version_in_output(self, tmp_path):
        _, out = _run_scoreboard(tmp_path)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        data = json.loads(lines[0])
        assert data["schema_version"] == "v1"

    def test_different_day_creates_new_records(self, tmp_path):
        files = _make_empty_files(tmp_path)
        out = files["output"]
        run_promotion_governance_scoreboard(
            readiness_file=files["readiness"],
            registry_file=files["registry"],
            impact_file=files["impact"],
            shadow_file=files["shadow"],
            evidence_file=files["evidence"],
            output_file=out,
            today="2026-07-25",
        )
        result2 = run_promotion_governance_scoreboard(
            readiness_file=files["readiness"],
            registry_file=files["registry"],
            impact_file=files["impact"],
            shadow_file=files["shadow"],
            evidence_file=files["evidence"],
            output_file=out,
            today="2026-07-26",
        )
        assert len(result2.new_records) == 5
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 10

    def test_ranked_records_in_result(self, tmp_path):
        result, _ = _run_scoreboard(tmp_path)
        assert len(result.ranked_records) == 5


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_cycle_high_scores(self, tmp_path):
        recs_r = [_readiness_rec(ct, 85.0) for ct in _CANDIDATE_ORDER]
        recs_e = [_evidence_rec(ct, 80) for ct in _CANDIDATE_ORDER]
        recs_s = [_shadow_rec(ct, True) for ct in _CANDIDATE_ORDER]
        recs_i = [_impact_rec(ct, 0.5) for ct in _CANDIDATE_ORDER]
        result, _ = _run_scoreboard(
            tmp_path,
            readiness=recs_r, evidence=recs_e, shadow=recs_s, impact=recs_i,
        )
        for r in result.new_records:
            assert r.governance_score >= 60.0  # at least PROMOTE_LATER
        assert result.summary.n_observe == 0

    def test_report_format_integrated(self, tmp_path):
        result, _ = _run_scoreboard(tmp_path)
        rpt = format_scoreboard_report(result.ranked_records, result.summary)
        assert "PROMOTION GOVERNANCE SCOREBOARD" in rpt

    def test_idempotent_three_runs(self, tmp_path):
        for day in ["2026-07-25", "2026-07-25", "2026-07-25"]:
            _run_scoreboard(tmp_path, today=day)
        _, out = _run_scoreboard(tmp_path, today="2026-07-25")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

    def test_candidate_order_invariant_in_ranking(self, tmp_path):
        # All same score → order should be deterministic
        result, _ = _run_scoreboard(tmp_path)
        candidates_with_rank_1 = [r for r in result.new_records if r.rank == 1]
        assert len(candidates_with_rank_1) == 1

    def test_score_formula_correctness(self, tmp_path):
        result, _ = _run_scoreboard(
            tmp_path,
            readiness=[_readiness_rec("EXIT_EXTENSION", 80.0)],
            evidence=[_evidence_rec("EXIT_EXTENSION", 60)],
            shadow=[_shadow_rec("EXIT_EXTENSION", True)] * 3 +
                   [_shadow_rec("EXIT_EXTENSION", False)],
            impact=[_impact_rec("EXIT_EXTENSION", 0.5)],
        )
        ee = next(r for r in result.new_records if r.candidate == "EXIT_EXTENSION")
        expected = compute_governance_score(
            readiness_score=80.0,
            evidence_score=60.0,
            success_rate=0.75,
            impact_delta=0.5,
        )
        assert ee.governance_score == pytest.approx(expected)

    def test_report_empty_when_no_records(self):
        s = ScoreboardSummary("", 0.0, 0, 0, 0, "")
        assert format_scoreboard_report([], s) == ""
