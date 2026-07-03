"""
tests/universe/test_data_freshness_guard.py

Tests for Data Freshness Guard.

Covers:
  1.  test_fresh_data_returns_true
  2.  test_stale_data_dry_mode_returns_false
  3.  test_stale_data_live_mode_raises
  4.  test_no_snapshots_live_raises
  5.  test_no_snapshots_dry_returns_false
  6.  test_missing_dir_live_raises
  7.  test_missing_dir_dry_returns_false
  8.  test_exactly_at_max_stale_days_is_fresh
  9.  test_one_over_max_stale_days_is_stale
  10. test_non_date_filename_skipped
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from universe.data_freshness_guard import (
    check_rsr_freshness,
    DataStalenessError,
    MAX_STALE_CALENDAR_DAYS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_snap(rsr_dir: Path, d: date) -> None:
    f = rsr_dir / f"{d.isoformat()}.json"
    f.write_text(json.dumps({"date": d.isoformat(), "scores": {"SYM.T": 50.0}}), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_fresh_data_returns_true(tmp_path):
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()
    ref = date(2026, 6, 6)
    _write_snap(rsr_dir, date(2026, 6, 5))   # 1 day old
    result = check_rsr_freshness(rsr_dir, is_live=False, reference_date=ref)
    assert result is True


def test_stale_data_dry_mode_returns_false(tmp_path):
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()
    ref = date(2026, 6, 6)
    _write_snap(rsr_dir, date(2026, 5, 25))   # very old
    result = check_rsr_freshness(rsr_dir, is_live=False, reference_date=ref)
    assert result is False


def test_stale_data_live_mode_raises(tmp_path):
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()
    ref = date(2026, 6, 6)
    _write_snap(rsr_dir, date(2026, 5, 25))   # very old
    with pytest.raises(DataStalenessError):
        check_rsr_freshness(rsr_dir, is_live=True, reference_date=ref)


def test_no_snapshots_live_raises(tmp_path):
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()  # empty directory
    with pytest.raises(DataStalenessError):
        check_rsr_freshness(rsr_dir, is_live=True, reference_date=date(2026, 6, 6))


def test_no_snapshots_dry_returns_false(tmp_path):
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()  # empty directory
    result = check_rsr_freshness(rsr_dir, is_live=False, reference_date=date(2026, 6, 6))
    assert result is False


def test_missing_dir_live_raises(tmp_path):
    rsr_dir = tmp_path / "nonexistent"
    with pytest.raises(DataStalenessError):
        check_rsr_freshness(rsr_dir, is_live=True, reference_date=date(2026, 6, 6))


def test_missing_dir_dry_returns_false(tmp_path):
    rsr_dir = tmp_path / "nonexistent"
    result = check_rsr_freshness(rsr_dir, is_live=False, reference_date=date(2026, 6, 6))
    assert result is False


def test_exactly_at_max_stale_days_is_fresh(tmp_path):
    """age == MAX_STALE_CALENDAR_DAYS is considered fresh."""
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()
    ref = date(2026, 6, 6)
    snap_date = ref - timedelta(days=MAX_STALE_CALENDAR_DAYS)
    _write_snap(rsr_dir, snap_date)
    result = check_rsr_freshness(rsr_dir, is_live=False, reference_date=ref)
    assert result is True


def test_one_over_max_stale_days_is_stale(tmp_path):
    """age == MAX_STALE_CALENDAR_DAYS + 1 is stale."""
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()
    ref = date(2026, 6, 6)
    snap_date = ref - timedelta(days=MAX_STALE_CALENDAR_DAYS + 1)
    _write_snap(rsr_dir, snap_date)
    result = check_rsr_freshness(rsr_dir, is_live=False, reference_date=ref)
    assert result is False


def test_non_date_filename_skipped(tmp_path):
    """Files with non-date stems (e.g. metadata.json) are skipped gracefully."""
    rsr_dir = tmp_path / "rsr"
    rsr_dir.mkdir()
    ref = date(2026, 6, 6)
    # Write a non-date file
    (rsr_dir / "metadata.json").write_text("{}", encoding="utf-8")
    # Write a valid date file (fresh)
    _write_snap(rsr_dir, date(2026, 6, 5))
    result = check_rsr_freshness(rsr_dir, is_live=False, reference_date=ref)
    assert result is True
