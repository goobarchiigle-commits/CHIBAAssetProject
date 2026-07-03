"""
tests/analytics/test_future_leader_expectancy.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from src.analytics.future_leader_expectancy import (
    BucketStats,
    ExpectancySummary,
    PenaltyStats,
    _MIN_SAMPLE,
    _PENALTY_NAMES,
    _RSR_ZONES,
    compute_expectancy_summary,
    compute_penalty_attribution,
    compute_rsr_zone_analysis,
    compute_score_bucket_analysis,
    filter_clean_complete,
    format_future_leader_expectancy_section,
    load_candidate_records,
    load_survivability_records,
    render_expectancy_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _surv_rec(
    symbol: str = "A.T",
    obs_date: str = "2025-01-10",
    score: float = 70.0,
    rsr_zone: str = "emerging",
    dq_weight: float = 1.0,
    insuf: bool = False,
    fwd5: Optional[float] = 0.03,
    fwd10: Optional[float] = 0.05,
    mfe: Optional[float] = 0.08,
    mae: Optional[float] = -0.02,
) -> dict:
    return {
        "symbol": symbol,
        "observation_date": obs_date,
        "future_leader_score": score,
        "rsr_zone": rsr_zone,
        "data_quality_weight": dq_weight,
        "insufficient_forward_data": insuf,
        "forward_return_5d": fwd5,
        "forward_return_10d": fwd10,
        "mfe_10d": mfe,
        "mae_10d": mae,
    }


def _cand_rec(
    symbol: str = "A.T",
    obs_date: str = "2025-01-10",
    abs_pens: Optional[dict] = None,
) -> dict:
    return {
        "symbol": symbol,
        "observation_date": obs_date,
        "absolute_penalties": abs_pens or {},
    }


def _write_surv_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def _write_cand_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# filter_clean_complete
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_only_excludes_partial_failure():
    recs = [
        _surv_rec("A.T", dq_weight=1.0),   # CLEAN
        _surv_rec("B.T", dq_weight=0.5),   # PARTIAL_FAILURE
        _surv_rec("C.T", dq_weight=0.0),   # MAJOR_FAILURE
    ]
    filtered = filter_clean_complete(recs)
    assert len(filtered) == 1
    assert filtered[0]["symbol"] == "A.T"


def test_clean_only_excludes_insufficient_forward_data():
    recs = [
        _surv_rec("A.T", dq_weight=1.0, insuf=False),
        _surv_rec("B.T", dq_weight=1.0, insuf=True),
    ]
    filtered = filter_clean_complete(recs)
    assert len(filtered) == 1
    assert filtered[0]["symbol"] == "A.T"


def test_clean_only_both_conditions_required():
    recs = [_surv_rec("X.T", dq_weight=0.5, insuf=True)]
    assert filter_clean_complete(recs) == []


# ─────────────────────────────────────────────────────────────────────────────
# compute_expectancy_summary — mean/median correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_expectancy_summary_mean_median_correctness():
    recs = [
        _surv_rec(fwd10=0.04, fwd5=0.02),
        _surv_rec(fwd10=0.06, fwd5=0.04),
    ]
    s = compute_expectancy_summary(recs)
    assert s.record_count == 2
    assert s.mean_return_10d == pytest.approx(0.05, abs=1e-5)
    assert s.median_return_10d == pytest.approx(0.05, abs=1e-5)
    assert s.mean_return_5d == pytest.approx(0.03, abs=1e-5)


def test_expectancy_summary_empty():
    s = compute_expectancy_summary([])
    assert s.record_count == 0
    assert s.mean_return_10d is None
    assert s.median_return_10d is None
    assert s.median_mfe_10d is None


def test_expectancy_summary_none_values_ignored():
    recs = [
        _surv_rec(fwd10=0.05, mfe=None),
        _surv_rec(fwd10=0.03, mfe=0.08),
    ]
    s = compute_expectancy_summary(recs)
    assert s.mean_return_10d == pytest.approx(0.04, abs=1e-5)
    assert s.median_mfe_10d == pytest.approx(0.08, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# compute_score_bucket_analysis
# ─────────────────────────────────────────────────────────────────────────────

def test_score_bucket_classification():
    recs = [
        _surv_rec(score=75.0, fwd10=0.10),
        _surv_rec(score=65.0, fwd10=0.05),
        _surv_rec(score=55.0, fwd10=0.02),
        _surv_rec(score=40.0, fwd10=-0.01),
    ]
    buckets = compute_score_bucket_analysis(recs)
    labels = [b.label for b in buckets]
    assert labels == ["70+", "60-70", "50-60", "<50"]

    b70  = next(b for b in buckets if b.label == "70+")
    b60  = next(b for b in buckets if b.label == "60-70")
    b50  = next(b for b in buckets if b.label == "50-60")
    b_lt = next(b for b in buckets if b.label == "<50")

    assert b70.count == 1
    assert b60.count == 1
    assert b50.count == 1
    assert b_lt.count == 1


def test_score_bucket_boundary_70():
    recs = [_surv_rec(score=70.0)]
    buckets = compute_score_bucket_analysis(recs)
    b70 = next(b for b in buckets if b.label == "70+")
    b60 = next(b for b in buckets if b.label == "60-70")
    assert b70.count == 1
    assert b60.count == 0


def test_score_bucket_boundary_60():
    recs = [_surv_rec(score=60.0)]
    buckets = compute_score_bucket_analysis(recs)
    b60 = next(b for b in buckets if b.label == "60-70")
    b50 = next(b for b in buckets if b.label == "50-60")
    assert b60.count == 1
    assert b50.count == 0


def test_score_bucket_empty_dataset():
    buckets = compute_score_bucket_analysis([])
    for b in buckets:
        assert b.count == 0
        assert b.mean_return_10d is None


# ─────────────────────────────────────────────────────────────────────────────
# compute_penalty_attribution
# ─────────────────────────────────────────────────────────────────────────────

def test_penalty_attribution_basic():
    surv = [
        _surv_rec("A.T", "2025-01-10", fwd10=0.05),
        _surv_rec("B.T", "2025-01-11", fwd10=-0.03),
    ]
    cand_map = {
        ("A.T", "2025-01-10"): _cand_rec("A.T", "2025-01-10", {"low_drift": 0.70}),
        ("B.T", "2025-01-11"): _cand_rec("B.T", "2025-01-11", {"negative_accumulation": 0.40}),
    }
    penalty_stats = compute_penalty_attribution(surv, cand_map)
    na = next(p for p in penalty_stats if p.penalty_name == "negative_accumulation")
    ld = next(p for p in penalty_stats if p.penalty_name == "low_drift")
    assert na.triggered_count == 1
    assert na.mean_return_10d == pytest.approx(-0.03, abs=1e-5)
    assert ld.triggered_count == 1
    assert ld.mean_return_10d == pytest.approx(0.05, abs=1e-5)


def test_penalty_attribution_overlap_permitted():
    surv = [_surv_rec("A.T", "2025-01-10", fwd10=0.04)]
    cand_map = {
        ("A.T", "2025-01-10"): _cand_rec(
            "A.T", "2025-01-10",
            {"low_drift": 0.70, "negative_accumulation": 0.40},
        ),
    }
    penalty_stats = compute_penalty_attribution(surv, cand_map)
    na = next(p for p in penalty_stats if p.penalty_name == "negative_accumulation")
    ld = next(p for p in penalty_stats if p.penalty_name == "low_drift")
    # Same record appears in both groups (overlap permitted)
    assert na.triggered_count == 1
    assert ld.triggered_count == 1
    assert na.mean_return_10d == pytest.approx(0.04, abs=1e-5)
    assert ld.mean_return_10d == pytest.approx(0.04, abs=1e-5)


def test_penalty_attribution_no_match_returns_zero():
    surv = [_surv_rec("A.T", "2025-01-10")]
    cand_map = {}
    penalty_stats = compute_penalty_attribution(surv, cand_map)
    for p in penalty_stats:
        assert p.triggered_count == 0


def test_penalty_attribution_preserves_penalty_order():
    penalty_stats = compute_penalty_attribution([], {})
    names = [p.penalty_name for p in penalty_stats]
    assert names == _PENALTY_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# compute_rsr_zone_analysis
# ─────────────────────────────────────────────────────────────────────────────

def test_rsr_zone_aggregation():
    recs = [
        _surv_rec(rsr_zone="emerging", fwd10=0.07),
        _surv_rec(rsr_zone="emerging", fwd10=0.09),
        _surv_rec(rsr_zone="mature",   fwd10=0.02),
        _surv_rec(rsr_zone="weak",     fwd10=-0.01),
    ]
    zone_stats = compute_rsr_zone_analysis(recs)
    em = next(b for b in zone_stats if b.label == "emerging")
    ma = next(b for b in zone_stats if b.label == "mature")
    wk = next(b for b in zone_stats if b.label == "weak")

    assert em.count == 2
    assert ma.count == 1
    assert wk.count == 1
    assert em.mean_return_10d == pytest.approx(0.08, abs=1e-5)


def test_rsr_zone_order():
    zone_stats = compute_rsr_zone_analysis([])
    labels = [b.label for b in zone_stats]
    assert labels == _RSR_ZONES  # emerging, mature, weak


def test_rsr_zone_empty():
    zone_stats = compute_rsr_zone_analysis([])
    for b in zone_stats:
        assert b.count == 0
        assert b.mean_return_10d is None


# ─────────────────────────────────────────────────────────────────────────────
# render_expectancy_report
# ─────────────────────────────────────────────────────────────────────────────

def test_render_insufficient_sample():
    s = ExpectancySummary(
        record_count=2,
        mean_return_5d=None, median_return_5d=None,
        mean_return_10d=None, median_return_10d=None,
        median_mfe_10d=None, median_mae_10d=None,
    )
    empty_buckets = [BucketStats("70+", 0, None, None, None, None, None, None)]
    text = render_expectancy_report(s, empty_buckets, [], [])
    assert "No statistically meaningful sample yet" in text
    assert "CLEAN complete records: 2" in text


def test_render_full_report_sections():
    # Build enough records to exceed _MIN_SAMPLE
    recs = [_surv_rec(score=75.0, rsr_zone="emerging", fwd5=0.02, fwd10=0.05, mfe=0.08, mae=-0.01)] * 10
    clean = filter_clean_complete(recs)
    s = compute_expectancy_summary(clean)
    score_b = compute_score_bucket_analysis(clean)
    pen_s = compute_penalty_attribution(clean, {})
    zone_s = compute_rsr_zone_analysis(clean)
    text = render_expectancy_report(s, score_b, pen_s, zone_s)

    assert "=== FUTURE LEADER EXPECTANCY ===" in text
    assert "=== SCORE BUCKET EXPECTANCY ===" in text
    assert "=== PENALTY EXPECTANCY ===" in text
    assert "=== RSR ZONE EXPECTANCY ===" in text
    assert "+5.0%" in text   # fwd10=0.05 → +5.0%
    assert "+2.0%" in text   # fwd5=0.02 → +2.0%


def test_render_insufficient_penalty_sample_message():
    recs = [_surv_rec()] * 10
    clean = filter_clean_complete(recs)
    s = compute_expectancy_summary(clean)
    score_b = compute_score_bucket_analysis(clean)
    pen_s = compute_penalty_attribution(clean, {})  # no penalties triggered
    zone_s = compute_rsr_zone_analysis(clean)
    text = render_expectancy_report(s, score_b, pen_s, zone_s)
    assert "(insufficient penalty-triggered samples)" in text


# ─────────────────────────────────────────────────────────────────────────────
# load_survivability_records / load_candidate_records
# ─────────────────────────────────────────────────────────────────────────────

def test_load_survivability_records(tmp_path: Path):
    log_dir = tmp_path / "surv"
    _write_surv_jsonl(
        log_dir / "future_leader_survivability_20250201.jsonl",
        [_surv_rec("A.T"), _surv_rec("B.T")],
    )
    recs = load_survivability_records(log_dir)
    assert len(recs) == 2


def test_load_survivability_records_missing_dir(tmp_path: Path):
    recs = load_survivability_records(tmp_path / "nonexistent")
    assert recs == []


def test_load_candidate_records(tmp_path: Path):
    cands_path = tmp_path / "candidates.jsonl"
    _write_cand_jsonl(cands_path, [
        _cand_rec("A.T", "2025-01-10", {"low_drift": 0.70}),
        _cand_rec("B.T", "2025-01-11", {}),
    ])
    lookup = load_candidate_records(cands_path)
    assert ("A.T", "2025-01-10") in lookup
    assert lookup[("A.T", "2025-01-10")]["absolute_penalties"] == {"low_drift": 0.70}


def test_load_candidate_records_dedup_first_wins(tmp_path: Path):
    cands_path = tmp_path / "candidates.jsonl"
    cands_path.write_text(
        json.dumps(_cand_rec("A.T", "2025-01-10", {"low_drift": 0.70})) + "\n" +
        json.dumps(_cand_rec("A.T", "2025-01-10", {"negative_accumulation": 0.40})) + "\n",
        encoding="utf-8",
    )
    lookup = load_candidate_records(cands_path)
    assert lookup[("A.T", "2025-01-10")]["absolute_penalties"] == {"low_drift": 0.70}


def test_load_candidate_records_missing_file(tmp_path: Path):
    lookup = load_candidate_records(tmp_path / "missing.jsonl")
    assert lookup == {}


# ─────────────────────────────────────────────────────────────────────────────
# format_future_leader_expectancy_section — integration
# ─────────────────────────────────────────────────────────────────────────────

def test_format_expectancy_section_returns_string(tmp_path: Path):
    surv_dir   = tmp_path / "surv"
    cands_path = tmp_path / "candidates.jsonl"
    cands_path.write_text("", encoding="utf-8")
    text = format_future_leader_expectancy_section(surv_dir, cands_path)
    # No data → insufficient sample message
    assert "FUTURE LEADER EXPECTANCY" in text or text == ""


def test_format_expectancy_section_fail_open(tmp_path: Path):
    # Invalid path — should not raise
    text = format_future_leader_expectancy_section(
        tmp_path / "nonexistent",
        tmp_path / "nonexistent.jsonl",
    )
    assert isinstance(text, str)


# ─────────────────────────────────────────────────────────────────────────────
# observation_only: no execution path identifiers in module source
# ─────────────────────────────────────────────────────────────────────────────

def test_no_execution_path_identifiers():
    import inspect
    import src.analytics.future_leader_expectancy as mod
    source = inspect.getsource(mod)
    # Strip comment/docstring lines — check code-form only
    code_lines = [
        ln for ln in source.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
        and not ln.strip().startswith('"""')
        and not ln.strip().startswith("'")
    ]
    code_only = "\n".join(code_lines)
    forbidden = [
        "signal_bridge",
        "order_routing",
        "governance_promotion",
        "deployable_universe",
        "run_order",
        "place_order",
    ]
    for term in forbidden:
        assert term not in code_only, f"forbidden code term '{term}' found in expectancy module"
