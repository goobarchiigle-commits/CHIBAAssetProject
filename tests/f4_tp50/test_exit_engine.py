"""
tests/f4_tp50/test_exit_engine.py
Pure-function tests for src.f4_tp50.exit_engine — no I/O, no broker.

Lineage: fork of tests/f4_tp30/test_exit_engine.py. Trailing-only cases
(gap-down stop, intraday stop, highest-since-entry, split handling) are
numerically IDENTICAL to TP30's own tests, because TRAIL_PCT is unchanged
(15%) and does not depend on TARGET_PCT — those mechanics were already
independently verified against TradingView on 2026-08-16 (see
backtests/f4_tp30_tv_verification/) and are REUSED here without new TV
verification (task: separate what's reusable from TP30's TV validation vs
what needs new confirmation). Target-related cases use TARGET_PCT=0.50
(entry*1.50) and are NEW — the arithmetic is unit-tested exactly here, but
no dedicated TradingView spot-check has been performed for TP50's specific
50% target boundary (unlike TP30's 6/6 TV-verified real cases) — flagged as
a follow-up item, not fabricated as "TV-verified" in this file.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.f4_tp50 import exit_engine as ee


# ── target price (TP50-specific: entry * 1.50) ─────────────────────────────
def test_compute_target_price():
    assert ee.compute_target_price(100.0) == pytest.approx(150.0)
    assert ee.compute_target_price(390.89) == pytest.approx(586.335, abs=1e-2)
    assert ee.TARGET_PCT == pytest.approx(0.50)


# ── entry-day seed (TRAIL_PCT-only, identical mechanics to TP30) ───────────
def test_entry_day_seed_uses_max_of_entry_price_and_entry_day_high():
    assert ee.compute_entry_day_seed(entry_price=100.0, entry_day_adjusted_high=105.0) == 105.0
    assert ee.compute_entry_day_seed(entry_price=100.0, entry_day_adjusted_high=95.0) == 100.0


# ── highest-since-entry (TRAIL_PCT-only, identical mechanics to TP30) ──────
def test_highest_since_entry_uses_history_through_prior_day_only():
    idx = pd.date_range("2026-01-05", periods=6, freq="B")
    high = pd.Series([100.0, 110.0, 120.0, 90.0, 200.0, 130.0], index=idx)
    entry_date = idx[0]
    as_of = idx[3]
    result = ee.compute_highest_since_entry(high, entry_date, entry_price=95.0, as_of=as_of)
    assert result == pytest.approx(120.0)


def test_highest_since_entry_seeds_with_entry_price_if_history_is_lower():
    idx = pd.date_range("2026-01-05", periods=4, freq="B")
    high = pd.Series([50.0, 60.0, 55.0, 58.0], index=idx)
    result = ee.compute_highest_since_entry(high, idx[0], entry_price=100.0, as_of=idx[3])
    assert result == pytest.approx(100.0)


def test_highest_since_entry_on_entry_day_itself_uses_entry_day_seed():
    idx = pd.date_range("2026-01-05", periods=1, freq="B")
    high = pd.Series([105.0], index=idx)
    result = ee.compute_highest_since_entry(high, idx[0], entry_price=100.0, as_of=idx[0])
    assert result == pytest.approx(105.0)


def test_highest_since_entry_never_uses_data_before_entry_date_9344_regression():
    """
    2026-08-20 9344インシデントの回帰テスト。実際のシグナルは entry_date=2026-08-18
    (8/18 High=1418) を想定していたが、実約定は2026-08-19に発生した(真のentry_date)。
    entry_dateとして正しく2026-08-19が渡される限り、compute_highest_since_entry()は
    entry_date以前(8/18)のHighを絶対に混入させてはならない
    （8/18のHigh=1418を使うと誤ったstop_level=1205.30が算出され、スプリアスな
    trailing-stop SELLを引き起こした——バグの所在はこの関数ではなく、上流で
    entry_dateとして誤って8/18が渡されていたposition metadataだったが、この関数
    自体がentry_date境界を厳守することを明示的に固定化する）。
    """
    idx = pd.to_datetime(["2026-08-18", "2026-08-19", "2026-08-20"])
    high = pd.Series([1418.0, 1309.0, 1300.0], index=idx)  # 実測OHLC High
    entry_date = pd.Timestamp("2026-08-19")  # 正しいEntry日（実約定日）
    as_of = pd.Timestamp("2026-08-20")       # Exit判定対象日

    result = ee.compute_highest_since_entry(high, entry_date, entry_price=1224.0, as_of=as_of)

    assert result != pytest.approx(1418.0), "entry_date(8/19)より前の8/18 Highが混入している"
    assert result == pytest.approx(1309.0)  # 正しいseed = max(entry_price=1224, 8/19 High=1309)

    stop_level = result * (1.0 - ee.TRAIL_PCT)
    assert stop_level == pytest.approx(1112.65, abs=1e-2)
    assert stop_level != pytest.approx(1205.30, abs=1e-2)  # 汚染データによる誤ったstop_level


# ── gap-down stop (TRAIL_PCT=15%, identical to TP30) ────────────────────────
def test_gap_down_stop_fires_at_open():
    # entry=100, highest=100 -> stop=85. Today opens at 80 (gap below stop).
    d = ee.evaluate_exit("TEST", entry_price=100.0, highest_since_entry=100.0,
                         today_open=80.0, today_high=82.0, today_low=79.0)
    assert d is not None
    assert d.exit_reason == "trailing_gap_open"
    assert d.exit_fill_price == pytest.approx(80.0 * (1 - ee.SLIPPAGE))


# ── intraday stop (TRAIL_PCT=15%, identical to TP30) ────────────────────────
def test_intraday_stop_fires_at_stop_level_not_low():
    # entry=100, highest=100 -> stop=85. Open=90 (above stop, no gap), low=83 (touches stop).
    d = ee.evaluate_exit("TEST", entry_price=100.0, highest_since_entry=100.0,
                         today_open=90.0, today_high=91.0, today_low=83.0)
    assert d is not None
    assert d.exit_reason == "trailing_touch"
    assert d.exit_fill_price == pytest.approx(85.0 * (1 - ee.SLIPPAGE))


def test_no_exit_when_price_stays_within_band():
    # entry=100 -> target=150 (well above); highest=110 -> stop=93.5 (well below).
    d = ee.evaluate_exit("TEST", entry_price=100.0, highest_since_entry=110.0,
                         today_open=100.0, today_high=101.0, today_low=99.0)
    assert d is None


# ── gap-up target (TP50-specific: target=150) ───────────────────────────────
def test_gap_up_target_fires_at_open():
    # entry=100 -> target=150. Opens at 155 (gap above target).
    d = ee.evaluate_exit("TEST", entry_price=100.0, highest_since_entry=105.0,
                         today_open=155.0, today_high=156.0, today_low=154.0)
    assert d is not None
    assert d.exit_reason == "target_gap_open"
    assert d.exit_fill_price == pytest.approx(155.0 * (1 - ee.SLIPPAGE))


# ── intraday target (TP50-specific: target=150) ─────────────────────────────
def test_intraday_target_fires_at_target_level_not_high():
    # entry=100 -> target=150. Open=140 (below target, no gap), high=152 (touches target).
    d = ee.evaluate_exit("TEST", entry_price=100.0, highest_since_entry=105.0,
                         today_open=140.0, today_high=152.0, today_low=138.0)
    assert d is not None
    assert d.exit_reason == "target_touch"
    assert d.exit_fill_price == pytest.approx(150.0 * (1 - ee.SLIPPAGE))


def test_target_not_hit_but_trailing_open_position_stays_open():
    """target未到達時のtrailing: neither trailing nor (the far-off 50%) target is
    breached -> position remains open (None), not a spurious exit."""
    # entry=100 -> target=150. highest=115 -> stop=97.75. Today stays well inside the band.
    d = ee.evaluate_exit("TEST", entry_price=100.0, highest_since_entry=115.0,
                         today_open=112.0, today_high=114.0, today_low=110.0)
    assert d is None


def test_target_reached_fires_immediate_exit_same_bar():
    """target到達後の即時exit: as soon as High crosses the 50% target within a
    single bar, the position exits THAT bar (no lag / no confirmation delay)."""
    d = ee.evaluate_exit("TEST", entry_price=200.0, highest_since_entry=210.0,
                         today_open=290.0, today_high=301.0, today_low=288.0)
    # target = 200*1.5 = 300; high=301 >= 300 -> touch (open=290 < 300, no gap)
    assert d is not None
    assert d.exit_reason == "target_touch"
    assert d.exit_fill_price == pytest.approx(300.0 * (1 - ee.SLIPPAGE))


# ── same-day collision: trailing must still win over the new 50% target ────
def test_trailing_wins_same_day_collision_with_target():
    # entry=100, highest=170 -> stop=144.5, target=150 (fixed, TP50).
    # open=148 (between, no gap either way), low=140<=144.5 (touch), high=152>=150 (touch).
    d = ee.evaluate_exit("TEST", entry_price=100.0, highest_since_entry=170.0,
                         today_open=148.0, today_high=152.0, today_low=140.0)
    assert d is not None
    assert d.exit_reason == "trailing_touch"  # NOT target_touch


def test_trailing_gap_open_wins_over_target_gap_open_if_open_below_stop():
    # entry=100, highest=170 -> stop=144.5, target=150. Open itself (140) is below stop.
    d = ee.evaluate_exit("TEST", entry_price=100.0, highest_since_entry=170.0,
                         today_open=140.0, today_high=155.0, today_low=138.0)
    assert d is not None
    assert d.exit_reason == "trailing_gap_open"


# ── entry fill / cost model (identical to TP30 — not a strategy delta) ─────
def test_entry_fill_price_applies_buy_slippage():
    assert ee.compute_entry_fill_price(1000.0) == pytest.approx(1001.0)


def test_commission_is_percentage_above_minimum():
    assert ee.compute_commission(1_000_000.0) == pytest.approx(550.0)


def test_commission_floors_at_minimum_99_yen():
    assert ee.compute_commission(10_000.0) == pytest.approx(99.0)


# ── split handling: continuous ADJUSTED prices, orthogonal to TARGET_PCT ───
def test_split_continuous_adjusted_prices_no_spurious_trigger():
    """Mirrors TP30's TradingView Toyota-7203 5:1-split synthetic test — this
    mechanic is TRAIL_PCT-only and does not depend on TARGET_PCT, so it is
    reused unchanged from TP30's already-verified behavior."""
    idx = pd.date_range("2026-01-05", periods=10, freq="B")
    high = pd.Series([2000, 2020, 2077, 2073, 2052, 2073, 2080, 2090, 2100, 2110], index=idx, dtype=float)
    entry_date = idx[0]
    entry_price = 1977.4
    as_of_split_boundary = idx[4]
    highest = ee.compute_highest_since_entry(high, entry_date, entry_price, as_of_split_boundary)
    assert highest == pytest.approx(2077.0)
    stop_level = highest * (1 - ee.TRAIL_PCT)
    d = ee.evaluate_exit("7203", entry_price, highest, today_open=2052.0, today_high=2073.0, today_low=2050.0)
    assert d is None or d.exit_reason.startswith("target")
    assert stop_level < 2050.0


# ── frozen-constant guard: only TARGET_PCT may differ from TP30 ────────────
def test_only_target_pct_differs_from_tp30_exit_engine():
    from src.f4_tp30 import exit_engine as ee30
    assert ee.TRAIL_PCT == ee30.TRAIL_PCT
    assert ee.SLIPPAGE == ee30.SLIPPAGE
    assert ee.COMMISSION == ee30.COMMISSION
    assert ee.MIN_COMMISSION == ee30.MIN_COMMISSION
    assert ee.FIXED_LOT == ee30.FIXED_LOT
    assert ee.TARGET_PCT != ee30.TARGET_PCT
    assert ee.TARGET_PCT == pytest.approx(0.50)
    assert ee30.TARGET_PCT == pytest.approx(0.30)
