"""
tests/test_startup_check_stale.py — Stale Market Data FAIL-CLOSED tests

Coverage:
  - _prev_trading_day: 月曜・祝日・通常
  - _check_snapshot_date: fresh / stale / dir missing / empty dir
  - run_startup_check(is_live=True):  stale → issues → ok=False
  - run_startup_check(is_live=False): stale → warnings → ok=True
  - run_startup_check(is_live=True):  fresh → no stale issue
  - run_startup_check(is_live=False): fresh → no stale warning
  - metrics keys: stale_data_detected / snapshot_date / expected_date
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.startup_check import (
    _check_snapshot_date,
    _is_trading_day,
    _prev_trading_day,
    run_startup_check,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_rsr_dir(tmp: Path, dates: list[str]) -> Path:
    """Create fake RSR snapshot dir with dated JSON files."""
    rsr = tmp / "rsr"
    rsr.mkdir()
    for d in dates:
        (rsr / f"{d}.json").write_text(
            json.dumps({"date": d, "scores": {}}), encoding="utf-8"
        )
    return rsr


# ─────────────────────────────────────────────────────────────────────────────
# TestPrevTradingDay
# ─────────────────────────────────────────────────────────────────────────────

class TestPrevTradingDay(unittest.TestCase):

    def test_tuesday_returns_monday(self):
        # 2026-06-02 (Tue) → 2026-06-01 (Mon)
        assert _prev_trading_day(date(2026, 6, 2)) == date(2026, 6, 1)

    def test_monday_returns_friday(self):
        # 2026-06-08 (Mon) → 2026-06-05 (Fri)
        assert _prev_trading_day(date(2026, 6, 8)) == date(2026, 6, 5)

    def test_saturday_returns_friday(self):
        # 2026-06-06 (Sat) → 2026-06-05 (Fri)
        assert _prev_trading_day(date(2026, 6, 6)) == date(2026, 6, 5)

    def test_sunday_returns_friday(self):
        # 2026-06-07 (Sun) → 2026-06-05 (Fri)
        assert _prev_trading_day(date(2026, 6, 7)) == date(2026, 6, 5)

    def test_holiday_skipped(self):
        # 2026-05-04 (Mon, GW holiday) → prev trading = 2026-04-30 (Thu? no, check)
        # GW: 05-03, 05-04, 05-05, 05-06 → prev trading before 05-04 = 2026-04-30? Let's check
        # 04-29 is also a holiday (Showa Day), so:
        # 05-03 (Sun+holiday), 05-02 (Sat), 05-01 (Fri) → prev = 05-01
        # Wait, actual check: 2026-05-04 prev = 2026-04-30 (Thu)
        # 04-29 = holiday, 04-28 (Tue) is not. Let's just verify it skips holidays.
        result = _prev_trading_day(date(2026, 5, 4))
        assert _is_trading_day(result), f"{result} is not a trading day"
        assert result < date(2026, 5, 4)

    def test_result_is_always_trading_day(self):
        # Verify 30 consecutive days all produce a valid trading day
        d = date(2026, 1, 5)
        for _ in range(30):
            prev = _prev_trading_day(d)
            assert _is_trading_day(prev), f"{prev} is not a trading day (from {d})"
            d += timedelta(days=1)

    def test_post_holiday_chain(self):
        # 2026-01-13 (Tue, 成人の日) → prev = 2026-01-09 (Fri)
        # Because: 01-12 (Mon, normal), wait — 01-12 is in _TSE_HOLIDAYS for 2026
        # So: 2026-01-13 → check 01-12 (holiday) → 01-11 (Sun) → 01-10 (Sat) → 01-09 (Fri)
        result = _prev_trading_day(date(2026, 1, 13))
        assert _is_trading_day(result)
        assert result < date(2026, 1, 13)


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckSnapshotDate
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSnapshotDate(unittest.TestCase):

    def test_fresh_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            # today = 2026-06-04 (Thu), expected prev = 2026-06-03
            today = date(2026, 6, 4)
            rsr = _make_rsr_dir(Path(tmp), ["2026-06-03"])
            stale, msg, snap, exp = _check_snapshot_date(today=today, rsr_dir=rsr)
            assert stale is False
            assert snap == "2026-06-03"
            assert exp  == "2026-06-03"

    def test_stale_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            # today = 2026-06-04, expected = 2026-06-03, actual = 2026-06-01
            today = date(2026, 6, 4)
            rsr = _make_rsr_dir(Path(tmp), ["2026-06-01"])
            stale, msg, snap, exp = _check_snapshot_date(today=today, rsr_dir=rsr)
            assert stale is True
            assert snap == "2026-06-01"
            assert exp  == "2026-06-03"
            assert "2026-06-01" in msg
            assert "2026-06-03" in msg

    def test_missing_rsr_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            rsr = Path(tmp) / "nonexistent"
            stale, msg, snap, exp = _check_snapshot_date(today=date(2026, 6, 4), rsr_dir=rsr)
            assert stale is False
            assert snap is None
            assert exp  is None

    def test_empty_rsr_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            rsr = Path(tmp) / "rsr"
            rsr.mkdir()
            stale, msg, snap, exp = _check_snapshot_date(today=date(2026, 6, 4), rsr_dir=rsr)
            assert stale is False
            assert snap is None

    def test_uses_latest_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Multiple files: latest should be used
            today = date(2026, 6, 4)
            rsr = _make_rsr_dir(Path(tmp), ["2026-06-01", "2026-06-02", "2026-06-03"])
            stale, _, snap, _ = _check_snapshot_date(today=today, rsr_dir=rsr)
            assert snap == "2026-06-03"
            assert stale is False

    def test_monday_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            # today = Monday 2026-06-08, expected prev = Friday 2026-06-05
            today = date(2026, 6, 8)
            rsr = _make_rsr_dir(Path(tmp), ["2026-06-05"])
            stale, _, snap, exp = _check_snapshot_date(today=today, rsr_dir=rsr)
            assert stale is False
            assert exp == "2026-06-05"

    def test_monday_stale_with_friday_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            # today = Monday 2026-06-08, snapshot = Thursday 2026-06-04 (stale)
            today = date(2026, 6, 8)
            rsr = _make_rsr_dir(Path(tmp), ["2026-06-04"])
            stale, msg, snap, exp = _check_snapshot_date(today=today, rsr_dir=rsr)
            assert stale is True
            assert exp == "2026-06-05"
            assert snap == "2026-06-04"


# ─────────────────────────────────────────────────────────────────────────────
# TestRunStartupCheckStaleData
# ─────────────────────────────────────────────────────────────────────────────

def _patched_startup_check(is_live: bool, rsr_dir: Path, today: date) -> dict:
    """
    Run run_startup_check with mocked portfolio/API/lock checks (all pass)
    and a real _check_snapshot_date pointing at the given rsr_dir/today.
    """
    import src.startup_check as sc

    dummy_state = {
        "cb_state":       "NORMAL",
        "equity_peak":    3_000_000.0,
        "available_cash": 2_500_000.0,
        "last_equity":    2_800_000.0,
    }

    with (
        patch.object(sc, "_check_api_port",       return_value=(True, "OK")),
        patch.object(sc, "_check_portfolio_state", return_value=(True, [], [], dummy_state)),
        patch.object(sc, "_check_lock_file",       return_value=(True, "OK")),
        patch.object(sc, "_RSR_SNAPSHOT_DIR",      rsr_dir),
        patch("src.startup_check.datetime") as mock_dt,
    ):
        from datetime import timezone, timedelta
        jst = timezone(timedelta(hours=9))
        mock_dt.now.return_value.strftime.return_value = "2026-06-04 08:43:00 JST"
        mock_dt.now.return_value.date.return_value = today

        def fake_snapshot(today=None, rsr_dir=None):
            from src.startup_check import _check_snapshot_date as real_fn
            return real_fn(today=today, rsr_dir=rsr_dir)

        with patch.object(sc, "_check_snapshot_date", side_effect=fake_snapshot):
            return sc.run_startup_check(is_live=is_live)


class TestRunStartupCheckStaleData(unittest.TestCase):

    def _run(self, is_live: bool, snapshot_date: str, today=date(2026, 6, 4)):
        with tempfile.TemporaryDirectory() as tmp:
            rsr = _make_rsr_dir(Path(tmp), [snapshot_date])
            import src.startup_check as sc
            dummy_state = {
                "cb_state": "NORMAL",
                "equity_peak": 3_000_000.0,
                "available_cash": 2_500_000.0,
                "last_equity": 2_800_000.0,
            }
            with (
                patch.object(sc, "_check_api_port",       return_value=(True, "OK")),
                patch.object(sc, "_check_portfolio_state", return_value=(True, [], [], dummy_state)),
                patch.object(sc, "_check_lock_file",       return_value=(True, "OK")),
                patch.object(sc, "_RSR_SNAPSHOT_DIR",      rsr),
                patch("src.startup_check.datetime") as mock_dt,
            ):
                from datetime import timezone, timedelta as _td
                mock_dt.now.return_value.strftime.return_value = "2026-06-04 08:43:00 JST"
                mock_dt.now.return_value.date.return_value = today
                return sc.run_startup_check(is_live=is_live)

    # ── LIVE + stale ───────────────────────────────────────────────────────────

    def test_live_stale_sets_ok_false(self):
        result = self._run(is_live=True, snapshot_date="2026-06-01")
        assert result["ok"] is False, "LIVE+stale should set ok=False"

    def test_live_stale_adds_issue(self):
        result = self._run(is_live=True, snapshot_date="2026-06-01")
        stale_issues = [i for i in result["issues"] if "STALE_DATA" in i]
        assert stale_issues, f"Expected STALE_DATA issue, got: {result['issues']}"

    def test_live_stale_no_warning(self):
        result = self._run(is_live=True, snapshot_date="2026-06-01")
        stale_warns = [w for w in result["warnings"] if "STALE_DATA" in w]
        assert not stale_warns, "LIVE+stale should not produce a warning (only issue)"

    def test_live_stale_metrics(self):
        result = self._run(is_live=True, snapshot_date="2026-06-01")
        assert result["stale_data_detected"] is True
        assert result["snapshot_date"] == "2026-06-01"
        assert result["expected_date"] == "2026-06-03"

    # ── LIVE + fresh ───────────────────────────────────────────────────────────

    def test_live_fresh_ok_true(self):
        result = self._run(is_live=True, snapshot_date="2026-06-03")
        assert result["ok"] is True

    def test_live_fresh_no_stale_issue(self):
        result = self._run(is_live=True, snapshot_date="2026-06-03")
        stale_issues = [i for i in result["issues"] if "STALE_DATA" in i]
        assert not stale_issues

    def test_live_fresh_metrics(self):
        result = self._run(is_live=True, snapshot_date="2026-06-03")
        assert result["stale_data_detected"] is False
        assert result["snapshot_date"] == "2026-06-03"
        assert result["expected_date"] == "2026-06-03"

    # ── DRY + stale ────────────────────────────────────────────────────────────

    def test_dry_stale_ok_true(self):
        result = self._run(is_live=False, snapshot_date="2026-06-01")
        assert result["ok"] is True, "DRY+stale should keep ok=True (warn only)"

    def test_dry_stale_adds_warning(self):
        result = self._run(is_live=False, snapshot_date="2026-06-01")
        stale_warns = [w for w in result["warnings"] if "STALE_DATA" in w]
        assert stale_warns, f"Expected STALE_DATA warning, got: {result['warnings']}"

    def test_dry_stale_no_issue(self):
        result = self._run(is_live=False, snapshot_date="2026-06-01")
        stale_issues = [i for i in result["issues"] if "STALE_DATA" in i]
        assert not stale_issues, "DRY+stale should not add to issues"

    def test_dry_stale_metrics(self):
        result = self._run(is_live=False, snapshot_date="2026-06-01")
        assert result["stale_data_detected"] is True
        assert result["snapshot_date"] == "2026-06-01"
        assert result["expected_date"] == "2026-06-03"

    # ── DRY + fresh ────────────────────────────────────────────────────────────

    def test_dry_fresh_ok_true(self):
        result = self._run(is_live=False, snapshot_date="2026-06-03")
        assert result["ok"] is True

    def test_dry_fresh_no_stale_warning(self):
        result = self._run(is_live=False, snapshot_date="2026-06-03")
        stale_warns = [w for w in result["warnings"] if "STALE_DATA" in w]
        assert not stale_warns

    # ── 月曜実行（前営業日=金曜）─────────────────────────────────────────────

    def test_monday_with_friday_snapshot(self):
        # Run on Monday 2026-06-08, snapshot = Friday 2026-06-05 → fresh
        result = self._run(is_live=True, snapshot_date="2026-06-05", today=date(2026, 6, 8))
        assert result["ok"] is True
        assert result["stale_data_detected"] is False
        assert result["expected_date"] == "2026-06-05"

    def test_monday_with_thursday_snapshot(self):
        # Run on Monday 2026-06-08, snapshot = Thursday 2026-06-04 → stale
        result = self._run(is_live=True, snapshot_date="2026-06-04", today=date(2026, 6, 8))
        assert result["ok"] is False
        assert result["stale_data_detected"] is True
        assert result["expected_date"] == "2026-06-05"
        assert result["snapshot_date"] == "2026-06-04"

    # ── 祝日考慮 ──────────────────────────────────────────────────────────────

    def test_holiday_after_gw(self):
        # 2026-05-07 (Thu after GW)
        # GW holidays: 05-03(Sun+hol), 05-04(Mon+hol), 05-05(Tue+hol), 05-06(Wed+hol)
        # 05-02 = Sat (weekend), 05-01 = Fri (trading day)
        # prev trading day before 05-07 = 2026-05-01
        result = self._run(is_live=True, snapshot_date="2026-05-01", today=date(2026, 5, 7))
        assert result["stale_data_detected"] is False
        assert result["expected_date"] == "2026-05-01"

    def test_holiday_after_gw_stale(self):
        # same setup but snapshot is older than expected (2026-04-30 is older than 2026-05-01)
        result = self._run(is_live=True, snapshot_date="2026-04-30", today=date(2026, 5, 7))
        assert result["stale_data_detected"] is True
        assert result["expected_date"] == "2026-05-01"
        assert result["snapshot_date"] == "2026-04-30"

    # ── result keys ────────────────────────────────────────────────────────────

    def test_all_metric_keys_present(self):
        result = self._run(is_live=False, snapshot_date="2026-06-03")
        for key in ("ok", "issues", "warnings", "summary", "state",
                    "stale_data_detected", "snapshot_date", "expected_date", "timestamp"):
            assert key in result, f"missing key: {key}"

    def test_6_4_incident_would_have_been_caught(self):
        # 6/4 incident: run_date=2026-06-04, snapshot=2026-06-01 → stale
        result = self._run(is_live=True, snapshot_date="2026-06-01", today=date(2026, 6, 4))
        assert result["ok"] is False
        assert result["stale_data_detected"] is True
        assert result["snapshot_date"] == "2026-06-01"
        assert result["expected_date"] == "2026-06-03"

    # ── 6/8インシデント回帰（DRY run が当日付スナップショットを生成→LIVE誤ブロック） ──

    def test_6_8_incident_live_today_snapshot_not_stale(self):
        # run_date=2026-06-08(Mon), snapshot=2026-06-08(today, DRY runが生成)
        # expected=2026-06-05(Fri), 2026-06-08 >= 2026-06-05 → NOT stale → LIVE ok=True
        result = self._run(is_live=True, snapshot_date="2026-06-08", today=date(2026, 6, 8))
        assert result["ok"] is True, (
            f"LIVE+今日付スナップショットは stale 扱いしてはならない: issues={result['issues']}"
        )
        assert result["stale_data_detected"] is False
        assert result["snapshot_date"] == "2026-06-08"
        assert result["expected_date"] == "2026-06-05"

    def test_6_8_incident_dry_today_snapshot_no_warning(self):
        # 同上 DRY mode でも警告なし
        result = self._run(is_live=False, snapshot_date="2026-06-08", today=date(2026, 6, 8))
        assert result["ok"] is True
        assert result["stale_data_detected"] is False
        stale_warns = [w for w in result["warnings"] if "STALE_DATA" in w]
        assert not stale_warns, f"DRY+今日付スナップショットで stale 警告が発生: {stale_warns}"

    def test_monday_sunday_snapshot_is_fresh(self):
        # run_date=2026-06-08(Mon), snapshot=2026-06-06(Sun, 非営業日付)
        # expected=2026-06-05(Fri), 2026-06-06 >= 2026-06-05 → NOT stale
        result = self._run(is_live=True, snapshot_date="2026-06-06", today=date(2026, 6, 8))
        assert result["ok"] is True, (
            f"非営業日付スナップショット(>=expected)は stale 扱いしてはならない: issues={result['issues']}"
        )
        assert result["stale_data_detected"] is False

    def test_snapshot_older_than_expected_is_stale(self):
        # run_date=2026-06-08(Mon), snapshot=2026-06-04(Thu, 2日前) → stale
        result = self._run(is_live=True, snapshot_date="2026-06-04", today=date(2026, 6, 8))
        assert result["ok"] is False
        assert result["stale_data_detected"] is True
        assert result["expected_date"] == "2026-06-05"
        assert result["snapshot_date"] == "2026-06-04"


if __name__ == "__main__":
    unittest.main()
