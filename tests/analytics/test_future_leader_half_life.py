"""
tests/analytics/test_future_leader_half_life.py

Coverage:
  - half_life_day correctness (decay crosses 50% threshold)
  - half_life_day=None when alpha persists through all observed days
  - survival curve correctness (exact ratios at specific days)
  - decay slope sign (negative when e10 < e5, positive when e10 > e5)
  - CLEAN-only filtering (data_quality_weight < 1.0 excluded)
  - insufficient_forward_data exclusion (survivability filter)
  - no lookahead contamination (no execution/live imports)
  - duplicate persistence prevention (dedup by symbol+obs_date)
  - fail-open integration behavior (hook survives missing dirs / empty data)
  - compute_survival_days: failure=True → offset, failure=False → HALF_LIFE_WINDOW
  - compute_survival_days: failure=True + None offset → None
  - HoldingHalfLifeSummary: zero-sample edge case
  - clean_sample_ratio correctness
  - format_half_life_section: insufficient sample path
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

import src.analytics.future_leader_half_life as hl
from src.analytics.future_leader_half_life import (
    HALF_LIFE_WINDOW,
    HoldingHalfLifeSummary,
    compute_half_life_summary,
    compute_survival_curve,
    compute_survival_days,
    filter_clean_failure,
    filter_clean_survivability,
    format_half_life_section,
    run_future_leader_half_life_hook,
    _compute_decay_slope,
    _compute_expectancy_at_days,
    _compute_half_life_day,
    _dedup_by_key,
)

_OBS_DATE = "2026-05-11"   # Monday — valid business day


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic record factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_failure_rec(
    symbol: str = "TEST.T",
    obs_date: str = _OBS_DATE,
    failure_detected: bool = False,
    failure_day_offset: Optional[int] = None,
    dq: float = 1.0,
    fwd5: float = 0.04,
    fwd10: float = 0.02,
) -> dict:
    return {
        "symbol":                 symbol,
        "observation_date":       obs_date,
        "observation_ts":         f"{obs_date}T09:00:00+09:00",
        "future_leader_score":    65.0,
        "rsr_zone":               "emerging",
        "integrity_state":        "CLEAN" if dq >= 1.0 else "PARTIAL_FAILURE",
        "data_quality_weight":    dq,
        "failure_detected":       failure_detected,
        "primary_failure_reason": "weak_followthrough" if failure_detected else None,
        "secondary_failure_flags": [],
        "failure_day_offset":     failure_day_offset,
        "forward_return_5d":      fwd5,
        "forward_return_10d":     fwd10,
        "mfe_10d":                0.06,
        "mae_10d":                -0.01,
        "mfe_3d":                 0.02,
        "observation_only":       True,
    }


def _make_surv_rec(
    symbol: str = "TEST.T",
    obs_date: str = _OBS_DATE,
    dq: float = 1.0,
    fwd5: float = 0.04,
    fwd10: float = 0.02,
    insufficient: bool = False,
) -> dict:
    return {
        "symbol":                     symbol,
        "observation_date":           obs_date,
        "observation_ts":             f"{obs_date}T09:00:00+09:00",
        "future_leader_score":        65.0,
        "rsr_zone":                   "emerging",
        "data_quality_weight":        dq,
        "integrity_state":            "CLEAN" if dq >= 1.0 else "PARTIAL_FAILURE",
        "close_price_at_observation": 1000.0,
        "forward_return_5d":          fwd5,
        "forward_return_10d":         fwd10,
        "mfe_10d":                    0.06,
        "mae_10d":                    -0.01,
        "days_materialized":          10,
        "insufficient_forward_data":  insufficient,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 7a — compute_survival_days
# ─────────────────────────────────────────────────────────────────────────────

def test_survival_days_no_failure():
    rec = _make_failure_rec(failure_detected=False)
    assert compute_survival_days(rec) == HALF_LIFE_WINDOW


def test_survival_days_failure_with_offset():
    rec = _make_failure_rec(failure_detected=True, failure_day_offset=7)
    assert compute_survival_days(rec) == 7


def test_survival_days_failure_offset_1():
    rec = _make_failure_rec(failure_detected=True, failure_day_offset=1)
    assert compute_survival_days(rec) == 1


def test_survival_days_failure_no_offset_returns_none():
    rec = _make_failure_rec(failure_detected=True, failure_day_offset=None)
    assert compute_survival_days(rec) is None


# ─────────────────────────────────────────────────────────────────────────────
# Task 7b — survival curve correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_survival_curve_correctness():
    """
    3 records: no_fail (alive always), fail@day5, fail@day10.
    Expected:
      day_1..4:  2/3 alive  (fail@5 and fail@10 still alive; no_fail always alive)
    wait: at day_d, alive = failure_detected=False OR failure_day_offset > d
      day_1: no_fail + offset5>1 + offset10>1 = 3/3 = 1.0
      day_4: no_fail + offset5>4 + offset10>4 = 3/3 = 1.0
      day_5: no_fail + offset5>5? no (5>5=False) + offset10>5 = 2/3
      day_9: no_fail + offset5>9? no + offset10>9 = 2/3
      day_10: no_fail + offset10>10? no = 1/3
      day_11..20: no_fail only = 1/3
    """
    recs = [
        _make_failure_rec(symbol="A.T", failure_detected=False),
        _make_failure_rec(symbol="B.T", failure_detected=True, failure_day_offset=5),
        _make_failure_rec(symbol="C.T", failure_detected=True, failure_day_offset=10),
    ]
    curve = compute_survival_curve(recs)

    assert curve["day_1"]  == pytest.approx(1.0, abs=1e-4)
    assert curve["day_4"]  == pytest.approx(1.0, abs=1e-4)
    assert curve["day_5"]  == pytest.approx(2 / 3, abs=1e-4)
    assert curve["day_9"]  == pytest.approx(2 / 3, abs=1e-4)
    assert curve["day_10"] == pytest.approx(1 / 3, abs=1e-4)
    assert curve["day_11"] == pytest.approx(1 / 3, abs=1e-4)
    assert curve["day_20"] == pytest.approx(1 / 3, abs=1e-4)


def test_survival_curve_all_alive():
    recs = [_make_failure_rec(symbol=f"S{i}.T", failure_detected=False) for i in range(5)]
    curve = compute_survival_curve(recs)
    for d in range(1, HALF_LIFE_WINDOW + 1):
        assert curve[f"day_{d}"] == pytest.approx(1.0, abs=1e-4)


def test_survival_curve_all_fail_day1():
    recs = [
        _make_failure_rec(symbol=f"S{i}.T", failure_detected=True, failure_day_offset=1)
        for i in range(5)
    ]
    curve = compute_survival_curve(recs)
    # day_1: offset > 1? No → 0/5
    assert curve["day_1"] == pytest.approx(0.0, abs=1e-4)
    # day_0 doesn't exist; day_1 is 0
    for d in range(1, HALF_LIFE_WINDOW + 1):
        assert curve[f"day_{d}"] == pytest.approx(0.0, abs=1e-4)


def test_survival_curve_empty():
    curve = compute_survival_curve([])
    assert len(curve) == HALF_LIFE_WINDOW
    for d in range(1, HALF_LIFE_WINDOW + 1):
        assert curve[f"day_{d}"] == 0.0


def test_survival_curve_has_all_20_days():
    recs = [_make_failure_rec(failure_detected=False)]
    curve = compute_survival_curve(recs)
    assert len(curve) == 20
    assert "day_1"  in curve
    assert "day_20" in curve


# ─────────────────────────────────────────────────────────────────────────────
# Task 7c — half_life_day correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_half_life_day_detected_at_day10():
    """e5=+8%, e10=+3%  → peak=0.08, threshold=0.04 → e10(0.03)<=threshold → day10"""
    exp_map = {3: None, 5: 0.08, 10: 0.03, 20: None}
    assert _compute_half_life_day(exp_map) == 10


def test_half_life_day_none_when_sustained():
    """e5=+8%, e10=+5%  → peak=0.08, threshold=0.04 → e10(0.05)>threshold → None"""
    exp_map = {3: None, 5: 0.08, 10: 0.05, 20: None}
    assert _compute_half_life_day(exp_map) is None


def test_half_life_day_none_when_all_none():
    exp_map = {3: None, 5: None, 10: None, 20: None}
    assert _compute_half_life_day(exp_map) is None


def test_half_life_day_none_when_peak_nonpositive():
    """peak <= 0 → cannot compute half-life"""
    exp_map = {3: None, 5: -0.02, 10: -0.05, 20: None}
    assert _compute_half_life_day(exp_map) is None


def test_half_life_day_exactly_at_threshold():
    """e5=0.10, e10=0.05 → exactly at 50% → half_life_day=10"""
    exp_map = {3: None, 5: 0.10, 10: 0.05, 20: None}
    assert _compute_half_life_day(exp_map) == 10


# ─────────────────────────────────────────────────────────────────────────────
# Task 7d — decay slope sign
# ─────────────────────────────────────────────────────────────────────────────

def test_decay_slope_negative_when_declining():
    """e5 > e10 → slope is negative"""
    exp_map = {3: None, 5: 0.06, 10: 0.02, 20: None}
    slope = _compute_decay_slope(exp_map)
    assert slope is not None
    assert slope < 0


def test_decay_slope_positive_when_rising():
    """e5 < e10 → slope is positive"""
    exp_map = {3: None, 5: 0.01, 10: 0.05, 20: None}
    slope = _compute_decay_slope(exp_map)
    assert slope is not None
    assert slope > 0


def test_decay_slope_none_when_single_point():
    exp_map = {3: None, 5: 0.04, 10: None, 20: None}
    assert _compute_decay_slope(exp_map) is None


def test_decay_slope_none_when_all_none():
    exp_map = {3: None, 5: None, 10: None, 20: None}
    assert _compute_decay_slope(exp_map) is None


def test_decay_slope_exact_two_points():
    """slope = (e10 - e5) / (10 - 5) = (0.02 - 0.06) / 5 = -0.008"""
    exp_map = {3: None, 5: 0.06, 10: 0.02, 20: None}
    slope = _compute_decay_slope(exp_map)
    assert slope == pytest.approx(-0.008, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Task 7e — CLEAN-only filtering
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_failure_filter_excludes_partial():
    recs = [
        _make_failure_rec(symbol="A.T", dq=1.0),
        _make_failure_rec(symbol="B.T", dq=0.5),   # PARTIAL_FAILURE
        _make_failure_rec(symbol="C.T", dq=0.0),   # MAJOR_FAILURE
    ]
    clean = filter_clean_failure(recs)
    assert len(clean) == 1
    assert clean[0]["symbol"] == "A.T"


def test_clean_failure_filter_all_clean():
    recs = [_make_failure_rec(symbol=f"S{i}.T", dq=1.0) for i in range(5)]
    assert len(filter_clean_failure(recs)) == 5


def test_clean_failure_filter_empty():
    assert filter_clean_failure([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Task 7f — insufficient_forward_data exclusion
# ─────────────────────────────────────────────────────────────────────────────

def test_survivability_filter_excludes_insufficient():
    recs = [
        _make_surv_rec(symbol="A.T", dq=1.0, insufficient=False),
        _make_surv_rec(symbol="B.T", dq=1.0, insufficient=True),   # excluded
        _make_surv_rec(symbol="C.T", dq=0.5, insufficient=False),  # excluded (PARTIAL)
    ]
    clean = filter_clean_survivability(recs)
    assert len(clean) == 1
    assert clean[0]["symbol"] == "A.T"


def test_survivability_filter_all_complete():
    recs = [_make_surv_rec(symbol=f"S{i}.T") for i in range(5)]
    assert len(filter_clean_survivability(recs)) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Task 7g — no lookahead contamination (no execution/live imports)
# ─────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_IMPORTS = [
    "allocator",
    "signal_bridge",
    "order",
    "broker",
    "deployable_universe",
]


def test_no_execution_imports():
    """Module must not import from execution/broker/order/signal_bridge/allocator paths."""
    mod = importlib.import_module("src.analytics.future_leader_half_life")
    # Collect transitive imports of the module itself (not its dependencies)
    for name in list(sys.modules.keys()):
        for forbidden in _FORBIDDEN_IMPORTS:
            # Only flag if the module itself contains the import at top-level
            if name.startswith("src.") and forbidden in name:
                # Verify it was imported BY future_leader_half_life
                src_text = Path("src/analytics/future_leader_half_life.py").read_text(encoding="utf-8")
                assert forbidden not in src_text, (
                    f"forbidden import '{forbidden}' found in future_leader_half_life.py"
                )


def test_module_does_not_import_live_modules():
    """Verify no import statement in the module references forbidden execution modules."""
    src_text = Path("src/analytics/future_leader_half_life.py").read_text(encoding="utf-8")
    import_lines = [
        line.strip() for line in src_text.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    combined = "\n".join(import_lines)
    for forbidden in _FORBIDDEN_IMPORTS:
        assert forbidden not in combined, (
            f"forbidden import '{forbidden}' found in import statements of future_leader_half_life.py"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 7h — duplicate persistence prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_dedup_removes_duplicate_keys():
    recs = [
        _make_failure_rec(symbol="A.T", obs_date=_OBS_DATE),
        _make_failure_rec(symbol="A.T", obs_date=_OBS_DATE),  # duplicate
        _make_failure_rec(symbol="B.T", obs_date=_OBS_DATE),
    ]
    deduped = _dedup_by_key(recs)
    assert len(deduped) == 2
    symbols = {r["symbol"] for r in deduped}
    assert "A.T" in symbols
    assert "B.T" in symbols


def test_dedup_different_dates_kept():
    recs = [
        _make_failure_rec(symbol="A.T", obs_date="2026-05-11"),
        _make_failure_rec(symbol="A.T", obs_date="2026-05-12"),
    ]
    assert len(_dedup_by_key(recs)) == 2


def test_survival_curve_dedup_applied():
    """Duplicates should not double-count in survival curve via filter_clean_failure."""
    recs = [
        _make_failure_rec(symbol="A.T", failure_detected=False),
        _make_failure_rec(symbol="A.T", failure_detected=False),  # duplicate
    ]
    clean = filter_clean_failure(recs)
    # After dedup: only 1 record
    assert len(clean) == 1
    curve = compute_survival_curve(clean)
    assert curve["day_1"] == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Task 7i — fail-open integration behavior
# ─────────────────────────────────────────────────────────────────────────────

def test_hook_succeeds_with_missing_dirs(tmp_path):
    """Hook must not raise when log dirs don't exist."""
    missing_surv  = tmp_path / "no_surv"
    missing_fail  = tmp_path / "no_fail"
    reports       = tmp_path / "reports"

    run_future_leader_half_life_hook(
        survivability_log_dir=missing_surv,
        failure_log_dir=missing_fail,
        reports_dir=reports,
        today=date(2026, 5, 24),
    )
    # No exception — pass


def test_hook_succeeds_with_empty_logs(tmp_path):
    surv_dir  = tmp_path / "surv"
    fail_dir  = tmp_path / "fail"
    rep_dir   = tmp_path / "reports"
    surv_dir.mkdir()
    fail_dir.mkdir()

    run_future_leader_half_life_hook(
        survivability_log_dir=surv_dir,
        failure_log_dir=fail_dir,
        reports_dir=rep_dir,
        today=date(2026, 5, 24),
    )
    # No exception — pass


def test_hook_writes_report_section(tmp_path):
    """With sufficient CLEAN data, hook writes a section to the report file."""
    surv_dir = tmp_path / "surv"
    fail_dir = tmp_path / "fail"
    rep_dir  = tmp_path / "reports"
    surv_dir.mkdir()
    fail_dir.mkdir()

    today = date(2026, 5, 24)
    today_str = today.strftime("%Y%m%d")

    # Write 6 failure records (> _MIN_SAMPLE=5)
    fail_records = [
        _make_failure_rec(
            symbol=f"S{i:03d}.T",
            failure_detected=(i < 3),
            failure_day_offset=(i + 3) if i < 3 else None,
            fwd5=0.04,
            fwd10=0.02,
        )
        for i in range(6)
    ]
    fail_path = fail_dir / f"future_leader_failure_{today_str}.jsonl"
    with fail_path.open("w", encoding="utf-8") as f:
        for r in fail_records:
            f.write(json.dumps(r) + "\n")

    # Write 6 survivability records
    surv_records = [
        _make_surv_rec(symbol=f"S{i:03d}.T", fwd5=0.04, fwd10=0.02)
        for i in range(6)
    ]
    surv_path = surv_dir / f"future_leader_survivability_{today_str}.jsonl"
    with surv_path.open("w", encoding="utf-8") as f:
        for r in surv_records:
            f.write(json.dumps(r) + "\n")

    run_future_leader_half_life_hook(
        survivability_log_dir=surv_dir,
        failure_log_dir=fail_dir,
        reports_dir=rep_dir,
        today=today,
    )

    report_path = rep_dir / f"future_leader_report_{today_str}.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "HOLDING HALF-LIFE" in content
    assert "SURVIVAL CURVE" in content


# ─────────────────────────────────────────────────────────────────────────────
# compute_half_life_summary — integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_zero_sample():
    summary = compute_half_life_summary([], [], total_failure_count=0)
    assert summary.sample_size == 0
    assert summary.half_life_day is None
    assert summary.expectancy_decay_slope is None


def test_summary_clean_sample_ratio():
    clean = [_make_failure_rec(symbol=f"S{i}.T") for i in range(7)]
    summary = compute_half_life_summary(clean, [], total_failure_count=10)
    assert summary.clean_sample_ratio == pytest.approx(0.7, abs=1e-4)


def test_summary_median_survival_all_alive():
    """All no-failure → all survival_days = HALF_LIFE_WINDOW → median = 20"""
    recs = [_make_failure_rec(symbol=f"S{i}.T", failure_detected=False) for i in range(8)]
    surv = [_make_surv_rec(symbol=f"S{i}.T", fwd5=0.04, fwd10=0.02) for i in range(8)]
    summary = compute_half_life_summary(recs, surv, total_failure_count=8)
    assert summary.median_survival_days == pytest.approx(HALF_LIFE_WINDOW, abs=0.1)


def test_summary_median_survival_mixed():
    """4 no-fail (20d) + 4 fail@day5 → median of [5,5,5,5,20,20,20,20] = 12.5"""
    recs = (
        [_make_failure_rec(symbol=f"F{i}.T", failure_detected=True, failure_day_offset=5) for i in range(4)]
        + [_make_failure_rec(symbol=f"N{i}.T", failure_detected=False) for i in range(4)]
    )
    summary = compute_half_life_summary(recs, [], total_failure_count=8)
    assert summary.median_survival_days == pytest.approx(12.5, abs=0.1)


def test_summary_expectancy_from_survivability():
    """expectancy_day_5 and day_10 derived from survivability records."""
    fail_recs = [_make_failure_rec(symbol=f"S{i}.T", failure_detected=False) for i in range(6)]
    surv_recs = [_make_surv_rec(symbol=f"S{i}.T", fwd5=0.06, fwd10=0.03) for i in range(6)]
    summary = compute_half_life_summary(fail_recs, surv_recs, total_failure_count=6)
    assert summary.expectancy_day_5  == pytest.approx(0.06, abs=1e-5)
    assert summary.expectancy_day_10 == pytest.approx(0.03, abs=1e-5)
    assert summary.expectancy_day_3  is None
    assert summary.expectancy_day_20 is None


def test_summary_half_life_day_detected():
    """e5=0.08, e10=0.03 → peak=0.08, threshold=0.04 → day10 drops below → half_life=10"""
    fail_recs = [_make_failure_rec(symbol=f"S{i}.T", failure_detected=False) for i in range(6)]
    surv_recs = [_make_surv_rec(symbol=f"S{i}.T", fwd5=0.08, fwd10=0.03) for i in range(6)]
    summary = compute_half_life_summary(fail_recs, surv_recs, total_failure_count=6)
    assert summary.half_life_day == 10


def test_summary_half_life_day_none_persistent():
    """e5=0.05, e10=0.04 → peak=0.05, threshold=0.025 → e10(0.04) > threshold → None"""
    fail_recs = [_make_failure_rec(symbol=f"S{i}.T", failure_detected=False) for i in range(6)]
    surv_recs = [_make_surv_rec(symbol=f"S{i}.T", fwd5=0.05, fwd10=0.04) for i in range(6)]
    summary = compute_half_life_summary(fail_recs, surv_recs, total_failure_count=6)
    assert summary.half_life_day is None


def test_summary_decay_slope_negative():
    """e5=0.06, e10=0.02 → slope < 0"""
    fail_recs = [_make_failure_rec(symbol=f"S{i}.T", failure_detected=False) for i in range(6)]
    surv_recs = [_make_surv_rec(symbol=f"S{i}.T", fwd5=0.06, fwd10=0.02) for i in range(6)]
    summary = compute_half_life_summary(fail_recs, surv_recs, total_failure_count=6)
    assert summary.expectancy_decay_slope is not None
    assert summary.expectancy_decay_slope < 0


def test_summary_decay_slope_positive():
    """e5=0.01, e10=0.05 → slope > 0"""
    fail_recs = [_make_failure_rec(symbol=f"S{i}.T", failure_detected=False) for i in range(6)]
    surv_recs = [_make_surv_rec(symbol=f"S{i}.T", fwd5=0.01, fwd10=0.05) for i in range(6)]
    summary = compute_half_life_summary(fail_recs, surv_recs, total_failure_count=6)
    assert summary.expectancy_decay_slope is not None
    assert summary.expectancy_decay_slope > 0


# ─────────────────────────────────────────────────────────────────────────────
# format_half_life_section
# ─────────────────────────────────────────────────────────────────────────────

def test_format_insufficient_sample():
    summary = HoldingHalfLifeSummary(
        sample_size=2, median_survival_days=10.0, half_life_day=None,
        expectancy_day_3=None, expectancy_day_5=None, expectancy_day_10=None,
        expectancy_day_20=None, expectancy_decay_slope=None, clean_sample_ratio=1.0,
    )
    text = format_half_life_section(summary, {})
    assert "insufficient" in text.lower()


def test_format_full_section():
    summary = HoldingHalfLifeSummary(
        sample_size=10, median_survival_days=8.0, half_life_day=10,
        expectancy_day_3=None, expectancy_day_5=0.041, expectancy_day_10=0.020,
        expectancy_day_20=None, expectancy_decay_slope=-0.0042, clean_sample_ratio=0.9,
    )
    curve = {f"day_{d}": (20 - d) / 20 for d in range(1, 21)}
    text = format_half_life_section(summary, curve)
    assert "HOLDING HALF-LIFE" in text
    assert "sample: 10" in text
    assert "half-life day: 10d" in text
    assert "SURVIVAL CURVE" in text
    assert "day5" in text
    assert "day10" in text
    # day3 and day20 are None → not in output
    assert "day3" not in text
    assert "day20" not in text
