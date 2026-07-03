"""tests/reporting/test_predictive_expansion_report.py

predictive_expansion_report — unit tests
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from src.reporting.predictive_expansion_report import (
    HIGH_SCORE_SKIP_THRESHOLD,
    BucketStats,
    DecilePerformance,
    FalseExpansionStats,
    PredictiveExpansionReport,
    PredictiveExpansionReporter,
    SkippedHighScoreEntry,
    TopPredictiveLeader,
    _avg,
    _load_jsonl,
    _rate,
    _safe_float,
    _save_jsonl_line,
    _score_to_decile,
    _score_to_label,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _entry(
    symbol: str = "A",
    entry_ts: str = "2026-05-23T09:15:00+09:00",
    predictive_score: int = 75,
    expansion_type: str = "continuation",
    spike_confidence: float = 0.0,
    skip_reasons: Optional[list[str]] = None,
    sector_igniting: bool = False,
) -> dict:
    return {
        "record_type": "ENTRY",
        "symbol": symbol,
        "entry_ts": entry_ts,
        "predictive_score": predictive_score,
        "expansion_type": expansion_type,
        "spike_confidence": spike_confidence,
        "skip_reasons": skip_reasons or [],
        "sector_igniting": sector_igniting,
    }


def _fwd(
    symbol: str = "A",
    entry_ts: str = "2026-05-23T09:15:00+09:00",
    forward_eod: Optional[float] = 0.012,
    forward_5m: Optional[float] = 0.003,
    forward_15m: Optional[float] = 0.006,
    forward_30m: Optional[float] = 0.009,
    breakout_continued: Optional[bool] = True,
    is_false_expansion: Optional[bool] = False,
) -> dict:
    return {
        "record_type": "FORWARD_OBS",
        "symbol": symbol,
        "entry_ts": entry_ts,
        "forward_eod": forward_eod,
        "forward_5m": forward_5m,
        "forward_15m": forward_15m,
        "forward_30m": forward_30m,
        "breakout_continued": breakout_continued,
        "is_false_expansion": is_false_expansion,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _reporter() -> PredictiveExpansionReporter:
    return PredictiveExpansionReporter()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_float_nan_default():
    import math
    assert _safe_float(math.nan, -1.0) == -1.0


def test_avg_empty():
    assert _avg([]) is None


def test_avg_values():
    assert abs(_avg([0.1, 0.2, 0.3]) - 0.2) < 1e-9


def test_rate_zero_total():
    assert _rate(1, 0) is None


def test_rate_normal():
    assert abs(_rate(1, 4) - 0.25) < 1e-9


def test_score_to_decile_boundaries():
    assert _score_to_decile(0) == 1
    assert _score_to_decile(9) == 1
    assert _score_to_decile(10) == 2
    assert _score_to_decile(99) == 10
    assert _score_to_decile(100) == 10


def test_score_to_label():
    assert _score_to_label(-1) == "NO_VERDICT"
    assert _score_to_label(0) == "BLOCKED"
    assert _score_to_label(19) == "WEAK"
    assert _score_to_label(44) == "WEAK"
    assert _score_to_label(45) == "MODERATE"
    assert _score_to_label(69) == "MODERATE"
    assert _score_to_label(70) == "STRONG"
    assert _score_to_label(100) == "STRONG"


# ─────────────────────────────────────────────────────────────────────────────
# _load_jsonl / _save_jsonl_line
# ─────────────────────────────────────────────────────────────────────────────

def test_load_jsonl_normal(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [{"a": 1}, {"b": 2}])
    records = _load_jsonl(path)
    assert len(records) == 2


def test_load_jsonl_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert _load_jsonl(path) == []


def test_load_jsonl_nonexistent(tmp_path):
    assert _load_jsonl(tmp_path / "no.jsonl") == []


def test_load_jsonl_malformed_line_skipped(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text('{"ok": 1}\n{broken json\n{"ok": 2}\n', encoding="utf-8")
    records = _load_jsonl(path)
    assert len(records) == 2


def test_save_jsonl_line_appends(tmp_path):
    path = tmp_path / "out.jsonl"
    _save_jsonl_line({"x": 1}, path)
    _save_jsonl_line({"x": 2}, path)
    records = _load_jsonl(path)
    assert len(records) == 2
    assert records[0]["x"] == 1
    assert records[1]["x"] == 2


def test_save_jsonl_creates_parent(tmp_path):
    path = tmp_path / "sub" / "deep" / "out.jsonl"
    _save_jsonl_line({"v": 9}, path)
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# _load_shadow_records
# ─────────────────────────────────────────────────────────────────────────────

def test_load_shadow_records_separates(tmp_path):
    path = tmp_path / "shadow.jsonl"
    _write_jsonl(path, [
        _entry("A", "ts1"),
        _fwd("A", "ts1"),
        _entry("B", "ts2"),
    ])
    reporter = _reporter()
    entries, fwd = reporter._load_shadow_records(path)
    assert len(entries) == 2
    assert len(fwd) == 1


def test_load_shadow_records_nonexistent(tmp_path):
    reporter = _reporter()
    entries, fwd = reporter._load_shadow_records(tmp_path / "missing.jsonl")
    assert entries == []
    assert fwd == []


def test_load_shadow_records_ignores_unknown_type(tmp_path):
    path = tmp_path / "shadow.jsonl"
    _write_jsonl(path, [{"record_type": "UNKNOWN", "x": 1}, _entry("A", "ts1")])
    reporter = _reporter()
    entries, fwd = reporter._load_shadow_records(path)
    assert len(entries) == 1
    assert len(fwd) == 0


# ─────────────────────────────────────────────────────────────────────────────
# _compute_decile_performance
# ─────────────────────────────────────────────────────────────────────────────

def test_decile_performance_returns_10_buckets(tmp_path):
    reporter = _reporter()
    entries = [_entry("A", "t1", predictive_score=75)]
    obs_map = {("A", "t1"): _fwd("A", "t1")}
    result = reporter._compute_decile_performance(entries, obs_map)
    assert len(result) == 10


def test_decile_performance_correct_bucket(tmp_path):
    reporter = _reporter()
    entries = [_entry("A", "t1", predictive_score=85)]
    obs_map = {("A", "t1"): _fwd("A", "t1", forward_eod=0.02, breakout_continued=True)}
    result = reporter._compute_decile_performance(entries, obs_map)
    d9 = next(d for d in result if d.decile == 9)
    assert d9.n_entries == 1
    assert d9.avg_forward_eod is not None
    assert d9.continuation_rate == 1.0


def test_decile_performance_empty_bucket_none_stats(tmp_path):
    reporter = _reporter()
    entries = [_entry("A", "t1", predictive_score=85)]
    obs_map = {}
    result = reporter._compute_decile_performance(entries, obs_map)
    for d in result:
        if d.decile != 9:
            assert d.n_entries == 0
            assert d.avg_forward_eod is None


def test_decile_performance_no_forward_obs(tmp_path):
    reporter = _reporter()
    entries = [_entry("A", "t1", predictive_score=30)]
    obs_map = {}
    result = reporter._compute_decile_performance(entries, obs_map)
    d3 = next(d for d in result if d.decile == 4)  # score 30 → decile 4
    assert d3.n_entries == 1
    assert d3.continuation_rate is None
    assert d3.false_expansion_rate is None


def test_decile_performance_score_minus1_skipped(tmp_path):
    reporter = _reporter()
    entries = [_entry("A", "t1", predictive_score=-1)]
    obs_map = {}
    result = reporter._compute_decile_performance(entries, obs_map)
    total = sum(d.n_entries for d in result)
    assert total == 0


# ─────────────────────────────────────────────────────────────────────────────
# _compute_continuation_by_bucket
# ─────────────────────────────────────────────────────────────────────────────

def test_continuation_by_bucket_labels(tmp_path):
    reporter = _reporter()
    entries = []
    obs_map = {}
    result = reporter._compute_continuation_by_bucket(entries, obs_map)
    labels = {b.label for b in result}
    assert "STRONG" in labels
    assert "MODERATE" in labels
    assert "WEAK" in labels


def test_continuation_by_bucket_strong_high_continuation(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=80),
        _entry("B", "t2", predictive_score=85),
    ]
    obs_map = {
        ("A", "t1"): _fwd("A", "t1", breakout_continued=True),
        ("B", "t2"): _fwd("B", "t2", breakout_continued=True),
    }
    result = reporter._compute_continuation_by_bucket(entries, obs_map)
    strong = next(b for b in result if b.label == "STRONG")
    assert strong.n_entries == 2
    assert strong.continuation_rate == 1.0


def test_continuation_by_bucket_no_entries_none_stats(tmp_path):
    reporter = _reporter()
    result = reporter._compute_continuation_by_bucket([], {})
    for b in result:
        assert b.n_entries == 0
        assert b.continuation_rate is None


# ─────────────────────────────────────────────────────────────────────────────
# _compute_false_expansion
# ─────────────────────────────────────────────────────────────────────────────

def test_false_expansion_rate(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=80, expansion_type="continuation", spike_confidence=0.0),
        _entry("B", "t2", predictive_score=30, expansion_type="spike",        spike_confidence=0.9),
    ]
    obs_map = {
        ("A", "t1"): _fwd("A", "t1", is_false_expansion=False),
        ("B", "t2"): _fwd("B", "t2", is_false_expansion=True),
    }
    result = reporter._compute_false_expansion(entries, obs_map)
    assert result.n_total_obs == 2
    assert result.n_false_expansion == 1
    assert abs(result.false_expansion_rate - 0.5) < 1e-9


def test_false_expansion_by_score_range(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=85, expansion_type="continuation"),
    ]
    obs_map = {("A", "t1"): _fwd("A", "t1", is_false_expansion=False)}
    result = reporter._compute_false_expansion(entries, obs_map)
    high_range = next(r for r in result.by_score_range if r["range"] == "80-100")
    assert high_range["n"] == 1
    assert high_range["rate"] == 0.0


def test_false_expansion_by_type(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=10, expansion_type="spike"),
        _entry("B", "t2", predictive_score=80, expansion_type="continuation"),
    ]
    obs_map = {
        ("A", "t1"): _fwd("A", "t1", is_false_expansion=True),
        ("B", "t2"): _fwd("B", "t2", is_false_expansion=False),
    }
    result = reporter._compute_false_expansion(entries, obs_map)
    assert result.by_expansion_type.get("spike", 0.0) == 1.0
    assert result.by_expansion_type.get("continuation", 1.0) == 0.0


def test_false_expansion_high_confidence(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=10, spike_confidence=0.8),
        _entry("B", "t2", predictive_score=80, spike_confidence=0.0),
    ]
    obs_map = {
        ("A", "t1"): _fwd("A", "t1", is_false_expansion=True),
        ("B", "t2"): _fwd("B", "t2", is_false_expansion=False),
    }
    result = reporter._compute_false_expansion(entries, obs_map)
    assert result.high_confidence_false_rate == 1.0


def test_false_expansion_no_obs(tmp_path):
    reporter = _reporter()
    entries = [_entry("A", "t1")]
    result = reporter._compute_false_expansion(entries, {})
    assert result.n_total_obs == 0
    assert result.false_expansion_rate is None


# ─────────────────────────────────────────────────────────────────────────────
# _find_skipped_high_score
# ─────────────────────────────────────────────────────────────────────────────

def test_skipped_high_score_found(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=75, skip_reasons=["slot_blocked"]),
        _entry("B", "t2", predictive_score=80, skip_reasons=["capital_blocked"]),
        _entry("C", "t3", predictive_score=30, skip_reasons=["slot_blocked"]),  # low score
        _entry("D", "t4", predictive_score=70, skip_reasons=[]),                 # no skip
    ]
    result = reporter._find_skipped_high_score(entries)
    symbols = {s.symbol for s in result}
    assert "A" in symbols
    assert "B" in symbols
    assert "C" not in symbols   # below threshold
    assert "D" not in symbols   # not skipped


def test_skipped_high_score_weak_score_not_included(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=65, skip_reasons=["weak_predictive_score"]),
    ]
    result = reporter._find_skipped_high_score(entries)
    # weak_predictive_score is not in _SKIP_BLOCKED_REASONS
    assert len(result) == 0


def test_skipped_high_score_sorted_by_score_desc(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=65, skip_reasons=["slot_blocked"]),
        _entry("B", "t2", predictive_score=90, skip_reasons=["slot_blocked"]),
        _entry("C", "t3", predictive_score=75, skip_reasons=["capital_blocked"]),
    ]
    result = reporter._find_skipped_high_score(entries)
    scores = [r.predictive_score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_skipped_high_score_empty(tmp_path):
    reporter = _reporter()
    assert reporter._find_skipped_high_score([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# _compute_top_leaders
# ─────────────────────────────────────────────────────────────────────────────

def test_top_leaders_sorted_by_avg_score(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=90),
        _entry("B", "t2", predictive_score=50),
        _entry("C", "t3", predictive_score=70),
    ]
    obs_map = {}
    leaders = reporter._compute_top_leaders(entries, obs_map)
    assert leaders[0].symbol == "A"


def test_top_leaders_sector_ignition_rate(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=80, sector_igniting=True),
        _entry("A", "t2", predictive_score=80, sector_igniting=False),
    ]
    obs_map = {}
    leaders = reporter._compute_top_leaders(entries, obs_map)
    a = next(l for l in leaders if l.symbol == "A")
    assert abs(a.sector_ignition_rate - 0.5) < 1e-9


def test_top_leaders_continuation_rate_from_obs(tmp_path):
    reporter = _reporter()
    entries = [
        _entry("A", "t1", predictive_score=80),
        _entry("A", "t2", predictive_score=80),
    ]
    obs_map = {
        ("A", "t1"): _fwd("A", "t1", breakout_continued=True),
        ("A", "t2"): _fwd("A", "t2", breakout_continued=False),
    }
    leaders = reporter._compute_top_leaders(entries, obs_map)
    a = next(l for l in leaders if l.symbol == "A")
    assert a.continuation_rate == 0.5


def test_top_leaders_n_top_limit(tmp_path):
    reporter = _reporter()
    entries = [
        _entry(f"SYM{i:02d}", f"t{i}", predictive_score=90 - i)
        for i in range(20)
    ]
    leaders = reporter._compute_top_leaders(entries, {}, n_top=5)
    assert len(leaders) <= 5


def test_top_leaders_empty(tmp_path):
    reporter = _reporter()
    assert reporter._compute_top_leaders([], {}) == []


def test_top_leaders_no_forward_obs(tmp_path):
    reporter = _reporter()
    entries = [_entry("A", "t1", predictive_score=80)]
    leaders = reporter._compute_top_leaders(entries, {})
    assert leaders[0].continuation_rate is None
    assert leaders[0].false_expansion_rate is None


# ─────────────────────────────────────────────────────────────────────────────
# format_report_markdown
# ─────────────────────────────────────────────────────────────────────────────

def test_format_markdown_contains_sections(tmp_path):
    reporter = _reporter()
    entries = [_entry("A", "t1", predictive_score=80, skip_reasons=["slot_blocked"])]
    obs_map = {("A", "t1"): _fwd("A", "t1")}
    decile = reporter._compute_decile_performance(entries, obs_map)
    buckets = reporter._compute_continuation_by_bucket(entries, obs_map)
    false_exp = reporter._compute_false_expansion(entries, obs_map)
    skipped = reporter._find_skipped_high_score(entries)
    leaders = reporter._compute_top_leaders(entries, obs_map)

    report = PredictiveExpansionReport(
        report_date="2026-05-23",
        report_id="TEST",
        generated_at="2026-05-23T09:00:00+0900",
        n_shadow_entries=1,
        n_forward_obs=1,
        n_ranking_records=0,
        decile_performance=decile,
        continuation_by_bucket=buckets,
        false_expansion=false_exp,
        skipped_high_score=skipped,
        top_leaders=leaders,
        summary_lines=["test summary"],
    )
    md = reporter.format_report_markdown(report)
    assert "# Predictive Expansion Report" in md
    assert "## 1. Score Decile Performance" in md
    assert "## 2. Continuation Probability by Score Bucket" in md
    assert "## 3. False Expansion Analysis" in md
    assert "## 4. Skipped High-Score Opportunities" in md
    assert "## 5. Top Predictive Leaders" in md


def test_format_markdown_no_skipped(tmp_path):
    reporter = _reporter()
    entries = []
    obs_map = {}
    decile = reporter._compute_decile_performance(entries, obs_map)
    buckets = reporter._compute_continuation_by_bucket(entries, obs_map)
    false_exp = reporter._compute_false_expansion(entries, obs_map)
    report = PredictiveExpansionReport(
        report_date="2026-05-23",
        report_id="EMPTY",
        generated_at="2026-05-23T09:00:00+0900",
        n_shadow_entries=0,
        n_forward_obs=0,
        n_ranking_records=0,
        decile_performance=decile,
        continuation_by_bucket=buckets,
        false_expansion=false_exp,
        skipped_high_score=[],
        top_leaders=[],
        summary_lines=[],
    )
    md = reporter.format_report_markdown(report)
    assert "なし" in md


# ─────────────────────────────────────────────────────────────────────────────
# PredictiveExpansionReport.to_dict
# ─────────────────────────────────────────────────────────────────────────────

def test_report_to_dict_serializable(tmp_path):
    reporter = _reporter()
    entries = [_entry("A", "t1", predictive_score=75)]
    obs_map = {("A", "t1"): _fwd("A", "t1")}
    decile = reporter._compute_decile_performance(entries, obs_map)
    buckets = reporter._compute_continuation_by_bucket(entries, obs_map)
    false_exp = reporter._compute_false_expansion(entries, obs_map)
    skipped = reporter._find_skipped_high_score(entries)
    leaders = reporter._compute_top_leaders(entries, obs_map)

    report = PredictiveExpansionReport(
        report_date="2026-05-23",
        report_id="TEST",
        generated_at="2026-05-23T09:00:00+0900",
        n_shadow_entries=1,
        n_forward_obs=1,
        n_ranking_records=0,
        decile_performance=decile,
        continuation_by_bucket=buckets,
        false_expansion=false_exp,
        skipped_high_score=skipped,
        top_leaders=leaders,
        summary_lines=["test"],
    )
    d = report.to_dict()
    # JSON serializable
    serialized = json.dumps(d)
    assert "report_date" in serialized


# ─────────────────────────────────────────────────────────────────────────────
# generate_report — integration
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_report_empty_logs(tmp_path):
    reporter = _reporter()
    report, saved = reporter.generate_report(
        date="2026-05-23",
        run_id="EMPTY",
        shadow_log_path=tmp_path / "shadow.jsonl",
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=tmp_path / "reports",
    )
    assert report.n_shadow_entries == 0
    assert report.n_forward_obs == 0
    assert report.report_date == "2026-05-23"
    assert saved is not None
    assert saved.exists()


def test_generate_report_saves_markdown(tmp_path):
    shadow_path = tmp_path / "shadow.jsonl"
    _write_jsonl(shadow_path, [
        _entry("A", "t1", predictive_score=80),
        _fwd("A", "t1", breakout_continued=True),
    ])
    reporter = _reporter()
    _, saved = reporter.generate_report(
        date="2026-05-23",
        shadow_log_path=shadow_path,
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=tmp_path / "reports",
    )
    assert saved is not None
    content = saved.read_text(encoding="utf-8")
    assert "# Predictive Expansion Report" in content
    assert "2026-05-23" in content


def test_generate_report_appends_to_jsonl(tmp_path):
    reporter = _reporter()
    jsonl_path = tmp_path / "summary.jsonl"
    reporter.generate_report(
        date="2026-05-23",
        run_id="R1",
        shadow_log_path=tmp_path / "shadow.jsonl",
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=tmp_path / "reports",
        report_jsonl_path=jsonl_path,
    )
    reporter.generate_report(
        date="2026-05-24",
        run_id="R2",
        shadow_log_path=tmp_path / "shadow.jsonl",
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=tmp_path / "reports",
        report_jsonl_path=jsonl_path,
    )
    records = _load_jsonl(jsonl_path)
    assert len(records) == 2
    run_ids = {r["report_id"] for r in records}
    assert "R1" in run_ids
    assert "R2" in run_ids


def test_generate_report_does_not_overwrite_existing(tmp_path):
    reporter = _reporter()
    shadow_path = tmp_path / "shadow.jsonl"
    _write_jsonl(shadow_path, [_entry("A", "t1", predictive_score=80)])
    _, saved1 = reporter.generate_report(
        date="2026-05-23",
        shadow_log_path=shadow_path,
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=tmp_path / "reports",
    )
    # Second call for same date should overwrite (atomic replace)
    _, saved2 = reporter.generate_report(
        date="2026-05-23",
        shadow_log_path=shadow_path,
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=tmp_path / "reports",
    )
    assert saved1 == saved2  # same filename
    assert saved2.exists()


def test_generate_report_deterministic(tmp_path):
    shadow_path = tmp_path / "shadow.jsonl"
    _write_jsonl(shadow_path, [
        _entry("A", "t1", predictive_score=80),
        _entry("B", "t2", predictive_score=60),
        _fwd("A", "t1", breakout_continued=True, is_false_expansion=False),
    ])
    reporter = _reporter()
    r1, _ = reporter.generate_report(
        date="2026-05-23",
        shadow_log_path=shadow_path,
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=tmp_path / "r1",
    )
    r2, _ = reporter.generate_report(
        date="2026-05-23",
        shadow_log_path=shadow_path,
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=tmp_path / "r2",
    )
    assert r1.n_shadow_entries == r2.n_shadow_entries
    assert r1.n_forward_obs    == r2.n_forward_obs
    assert len(r1.top_leaders) == len(r2.top_leaders)
    if r1.top_leaders:
        assert r1.top_leaders[0].symbol == r2.top_leaders[0].symbol


def test_generate_report_with_skipped_entries(tmp_path):
    shadow_path = tmp_path / "shadow.jsonl"
    _write_jsonl(shadow_path, [
        _entry("A", "t1", predictive_score=75, skip_reasons=["slot_blocked"]),
        _entry("B", "t2", predictive_score=40, skip_reasons=["slot_blocked"]),  # below threshold
    ])
    reporter = _reporter()
    report, _ = reporter.generate_report(
        date="2026-05-23",
        shadow_log_path=shadow_path,
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=tmp_path / "reports",
    )
    skip_syms = {s.symbol for s in report.skipped_high_score}
    assert "A" in skip_syms
    assert "B" not in skip_syms


def test_generate_report_no_report_dir_created(tmp_path):
    reporter = _reporter()
    report_dir = tmp_path / "new_dir" / "nested"
    _, saved = reporter.generate_report(
        date="2026-05-23",
        shadow_log_path=tmp_path / "shadow.jsonl",
        ranking_log_path=tmp_path / "rankings.jsonl",
        report_dir=report_dir,
    )
    assert saved is not None
    assert report_dir.exists()
