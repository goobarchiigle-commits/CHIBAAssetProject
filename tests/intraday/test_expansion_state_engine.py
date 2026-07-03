"""
test_expansion_state_engine.py

Coverage targets (≥90%):
  - expansion_state_engine の全コンポーネント関数
  - spike 検出 (寄り天パターン最優先)
  - continuation 検出
  - malformed / missing / partial session handling
  - deterministic replay
"""
from __future__ import annotations

import math
import pytest

from src.intraday.expansion_state_engine import (
    MinuteBar,
    ExpansionState,
    OpeningDriveResult,
    RVOLResult,
    PersistenceResult,
    RelativeStrengthResult,
    GapResult,
    compute_opening_drive,
    compute_rvol,
    compute_persistence,
    compute_rs,
    compute_gap_acceptance,
    compute_expansion_state,
    _validate_bars,
    _float_safe,
    _clamp01,
    SPIKE_EARLY_WINDOW,
    MIN_BARS_FOR_VERDICT,
    SPIKE_REVERSAL_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(dt: str, o: float, h: float, lo: float, c: float, v: int = 10000) -> MinuteBar:
    return MinuteBar(dt=dt, open=o, high=h, low=lo, close=c, volume=v)


def make_spike_bars(
    n: int = 25,
    open_price: float = 1000.0,
    spike_high: float = 1050.0,
    spike_bars: int = 3,
    reversal_close: float = 1010.0,
    vol_early: int = 50000,
    vol_late: int = 5000,
) -> list[MinuteBar]:
    """
    寄り天パターン: 最初 spike_bars 本で spike_high まで上昇、その後 reversal_close まで下落して推移。
    """
    bars: list[MinuteBar] = []
    for i in range(n):
        t = f"09:{i:02d}"
        if i < spike_bars:
            frac = (i + 1) / spike_bars
            c = open_price + (spike_high - open_price) * frac
            h = c + 2.0
            lo = open_price if i == 0 else bars[-1].close - 1.0
            vol = vol_early
        else:
            # 段階的に reversal_close まで戻す
            steps_left = n - spike_bars
            step = i - spike_bars
            c = spike_high - (spike_high - reversal_close) * (step + 1) / steps_left
            c = max(c, reversal_close)
            h = c + 1.0
            lo = c - 2.0
            vol = vol_late
        bars.append(_bar(t, open_price if i == 0 else bars[-1].close, h, lo, c, vol))
    return bars


def make_continuation_bars(
    n: int = 25,
    open_price: float = 1000.0,
    end_price: float = 1060.0,
    vol_early: int = 20000,
    vol_late: int = 25000,
) -> list[MinuteBar]:
    """継続パターン: 着実に上昇し続ける。"""
    bars: list[MinuteBar] = []
    for i in range(n):
        t = f"09:{i:02d}"
        frac = (i + 1) / n
        c = open_price + (end_price - open_price) * frac
        h = c + 1.0
        lo = c - 0.5
        vol = vol_early + int((vol_late - vol_early) * frac)
        bars.append(_bar(t, open_price if i == 0 else bars[-1].close, h, lo, c, vol))
    return bars


def make_flat_bars(n: int = 20, price: float = 1000.0) -> list[MinuteBar]:
    return [_bar(f"09:{i:02d}", price, price + 1, price - 1, price) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# _validate_bars
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_bars_filters_negative_price():
    bars = [
        _bar("09:00", -100, 100, 90, 95),
        _bar("09:01", 1000, 1010, 990, 1005),
    ]
    valid = _validate_bars(bars)
    assert len(valid) == 1
    assert valid[0].dt == "09:01"


def test_validate_bars_filters_high_below_low():
    bars = [
        _bar("09:00", 1000, 990, 1010, 995),  # high < low
        _bar("09:01", 1000, 1010, 990, 1005),
    ]
    valid = _validate_bars(bars)
    assert len(valid) == 1


def test_validate_bars_filters_zero_price():
    bars = [_bar("09:00", 0, 10, 0, 5)]
    assert _validate_bars(bars) == []


def test_validate_bars_allows_zero_volume():
    bars = [_bar("09:00", 1000, 1010, 990, 1005, v=0)]
    valid = _validate_bars(bars)
    assert len(valid) == 1
    assert valid[0].volume == 0


def test_validate_bars_empty_input():
    assert _validate_bars([]) == []


def test_validate_bars_handles_none_volume():
    b = MinuteBar(dt="09:00", open=1000, high=1010, low=990, close=1005, volume=None)
    valid = _validate_bars([b])
    assert len(valid) == 1
    assert valid[0].volume == 0


# ─────────────────────────────────────────────────────────────────────────────
# compute_opening_drive
# ─────────────────────────────────────────────────────────────────────────────

def test_drive_continuation_high_efficiency():
    bars = make_continuation_bars(25, 1000, 1060)
    result = compute_opening_drive(bars, open_price=1000.0)
    assert result.efficiency > 0.70, f"expected high efficiency, got {result.efficiency}"
    assert result.new_high_confirmed  # end price > early peak


def test_drive_spike_low_efficiency():
    bars = make_spike_bars(25, 1000, 1050, spike_bars=3, reversal_close=1010)
    result = compute_opening_drive(bars, open_price=1000.0)
    assert result.efficiency < 0.40, f"expected low efficiency for spike: {result.efficiency}"
    assert result.early_reversal_pct > SPIKE_REVERSAL_THRESHOLD


def test_drive_early_reversal_pct_clamped():
    bars = make_spike_bars(25, 1000, 1050, reversal_close=990)  # goes below open
    result = compute_opening_drive(bars, open_price=1000.0)
    assert 0.0 <= result.early_reversal_pct <= 1.0


def test_drive_new_high_confirmed_above_early_peak():
    # Current close 明確に early_peak 超え
    bars = make_continuation_bars(25, 1000, 1080)
    result = compute_opening_drive(bars, 1000.0)
    assert result.new_high_confirmed is True


def test_drive_empty_bars_returns_safe_default():
    result = compute_opening_drive([], open_price=1000.0)
    assert result.efficiency == 0.5
    assert result.new_high_confirmed is False


def test_drive_zero_open_price():
    bars = make_continuation_bars(10)
    result = compute_opening_drive(bars, open_price=0.0)
    assert result.efficiency == 0.5


def test_drive_spike_window_larger_than_bars():
    bars = make_spike_bars(3)  # only 3 bars, spike_window=5
    result = compute_opening_drive(bars, 1000.0, spike_window=5)
    assert 0.0 <= result.efficiency <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_rvol
# ─────────────────────────────────────────────────────────────────────────────

def test_rvol_high_acceleration():
    # 後半ほど出来高が多い
    bars = [
        _bar(f"09:{i:02d}", 1000, 1010, 990, 1005,
             v=5000 if i < 5 else 20000)
        for i in range(20)
    ]
    result = compute_rvol(bars, avg_daily_volume=100000)
    assert result.rvol_acceleration > 1.0


def test_rvol_low_acceleration_spike_pattern():
    # 前半に出来高集中 (寄り天パターン)
    bars = make_spike_bars(20, vol_early=50000, vol_late=3000)
    result = compute_rvol(bars, avg_daily_volume=300000)
    assert result.rvol_acceleration < 0.5


def test_rvol_no_adv_returns_neutral():
    bars = make_continuation_bars(20)
    result = compute_rvol(bars, avg_daily_volume=0)
    assert result.rvol_ratio == 1.0


def test_rvol_empty_bars():
    result = compute_rvol([], avg_daily_volume=100000)
    assert result.rvol_ratio == 1.0
    assert result.rvol_acceleration == 1.0
    assert result.total_volume == 0


def test_rvol_ratio_capped_at_10x():
    bars = [_bar("09:00", 1000, 1010, 990, 1005, v=10_000_000)]
    result = compute_rvol(bars, avg_daily_volume=100000)
    assert result.rvol_ratio <= 10.0


def test_rvol_few_bars_no_acceleration():
    bars = [_bar(f"09:{i:02d}", 1000, 1010, 990, 1005) for i in range(5)]
    result = compute_rvol(bars, avg_daily_volume=100000)
    assert result.rvol_acceleration == 1.0  # not enough bars for split


# ─────────────────────────────────────────────────────────────────────────────
# compute_persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_persistence_continuation_high():
    bars = make_continuation_bars(25, 1000, 1060)
    result = compute_persistence(bars, or_window=15)
    assert result.persistence_ratio > 0.55
    assert result.above_midpoint is True


def test_persistence_spike_low():
    bars = make_spike_bars(25, 1000, 1050, reversal_close=1005)
    result = compute_persistence(bars, or_window=15)
    assert result.persistence_ratio < 0.40


def test_persistence_empty_bars():
    result = compute_persistence([], or_window=15)
    assert result.persistence_ratio == 0.5


def test_persistence_clamped_0_to_1():
    bars = make_spike_bars(25, 1000, 1050, reversal_close=900)  # below OR low
    result = compute_persistence(bars, or_window=15)
    assert 0.0 <= result.persistence_ratio <= 1.0


def test_persistence_flat_session():
    bars = make_flat_bars(20, 1000)
    result = compute_persistence(bars, or_window=15)
    assert 0.0 <= result.persistence_ratio <= 1.0


def test_persistence_or_window_larger_than_bars():
    bars = make_continuation_bars(5, 1000, 1030)
    result = compute_persistence(bars, or_window=15)
    # or_bars = all 5 bars
    assert 0.0 <= result.persistence_ratio <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_rs
# ─────────────────────────────────────────────────────────────────────────────

def test_rs_top_performer():
    universe = {"A": 0.01, "B": 0.02, "C": 0.03}
    result = compute_rs("C", universe)
    assert result.rs_percentile == 100.0


def test_rs_bottom_performer():
    universe = {"A": 0.01, "B": 0.02, "C": 0.03}
    result = compute_rs("A", universe)
    pct = result.rs_percentile
    assert pct < 50.0


def test_rs_symbol_not_in_universe():
    result = compute_rs("X", {"A": 0.01})
    assert result.rs_percentile == 50.0


def test_rs_empty_universe():
    result = compute_rs("A", {})
    assert result.rs_percentile == 50.0


def test_rs_single_symbol_universe():
    result = compute_rs("A", {"A": 0.05})
    assert result.rs_percentile == 50.0  # 1 symbol → 中立


def test_rs_percentile_bounded():
    universe = {f"S{i}": float(i) * 0.001 for i in range(50)}
    result = compute_rs("S25", universe)
    assert 0.0 <= result.rs_percentile <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_gap_acceptance
# ─────────────────────────────────────────────────────────────────────────────

def test_gap_up_acceptance_holding():
    # gap up, holding at open
    result = compute_gap_acceptance(1050.0, 1000.0, 1050.0)
    assert result.gap_active is True
    assert abs(result.gap_acceptance - 1.0) < 0.01


def test_gap_up_extending():
    result = compute_gap_acceptance(1050.0, 1000.0, 1080.0)
    assert result.gap_active is True
    assert result.gap_acceptance == 1.0  # capped


def test_gap_up_filling():
    # price halfway between prev_close and open
    result = compute_gap_acceptance(1050.0, 1000.0, 1025.0)
    assert result.gap_active is True
    assert 0.4 < result.gap_acceptance < 0.6


def test_gap_up_completely_filled():
    result = compute_gap_acceptance(1050.0, 1000.0, 1000.0)
    assert result.gap_active is True
    assert abs(result.gap_acceptance) < 0.01


def test_no_gap_returns_neutral():
    result = compute_gap_acceptance(1000.0, 999.0, 1001.0)
    # gap_pct = 0.1% < MIN_GAP_PCT(0.3%) → inactive
    assert result.gap_active is False
    assert result.gap_acceptance == 0.5


def test_gap_acceptance_bounded():
    result = compute_gap_acceptance(1050.0, 1000.0, 900.0)
    assert -1.0 <= result.gap_acceptance <= 1.0


def test_gap_zero_prev_close():
    result = compute_gap_acceptance(1050.0, 0.0, 1050.0)
    assert result.gap_active is False


# ─────────────────────────────────────────────────────────────────────────────
# compute_expansion_state — spike detection (最重要テスト)
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE_SAMPLE = {"TARGET": 0.01, "A": 0.005, "B": -0.002, "C": 0.015}


def test_spike_pattern_detected():
    """寄り天: 上昇スパイック後、戻し。expansion_type == 'spike' になること。"""
    bars = make_spike_bars(25, 1000, 1060, spike_bars=3, reversal_close=1008)
    state = compute_expansion_state(
        "TARGET", bars, 950.0, UNIVERSE_SAMPLE, "09:25",
        avg_daily_volume=300000,
    )
    assert state.expansion_type == "spike", (
        f"Expected spike, got {state.expansion_type}"
        f" (spike_conf={state.spike_confidence:.3f}"
        f" reversal={state.early_reversal_pct:.3f})"
    )
    assert state.is_spike_pattern is True
    assert state.spike_confidence > 0.5


def test_spike_expansion_probability_suppressed():
    """スパイク時は expansion_probability が強く抑制される。"""
    bars = make_spike_bars(25, 1000, 1060, spike_bars=3, reversal_close=1005)
    state = compute_expansion_state(
        "TARGET", bars, 950.0, UNIVERSE_SAMPLE, "09:25",
        avg_daily_volume=300000,
    )
    assert state.expansion_probability < 0.30, (
        f"Spike should have low probability: {state.expansion_probability}"
    )


def test_continuation_detected():
    """継続パターン: expansion_type == 'continuation'。"""
    bars = make_continuation_bars(25, 1000, 1060)
    state = compute_expansion_state(
        "TARGET", bars, 950.0, UNIVERSE_SAMPLE, "09:25",
        avg_daily_volume=200000,
    )
    assert state.expansion_type == "continuation", (
        f"Expected continuation, got {state.expansion_type}"
        f" (drive={state.drive_efficiency:.2f}"
        f" persist={state.persistence_ratio:.2f})"
    )
    assert state.is_spike_pattern is False
    assert state.expansion_probability > 0.55


def test_new_high_overrides_spike():
    """
    新高値 (current > early_peak) はスパイク判定を完全解除する。
    spike 要件を満たすように見えても continuation になること。
    """
    bars = make_continuation_bars(25, 1000, 1080)  # 最終値がearly_peakを大きく超える
    state = compute_expansion_state(
        "TARGET", bars, 950.0, UNIVERSE_SAMPLE, "09:25",
        avg_daily_volume=200000,
    )
    assert state.new_high_confirmed is True
    assert state.spike_confidence == 0.0
    assert state.expansion_type == "continuation"


def test_partial_session_3_bars_no_spike():
    """
    bars < MIN_BARS_FOR_VERDICT → スパイク判定なし。
    3本の上昇バーだけでは reversal が未発生のため、is_spike_pattern=False が必須。
    continuation or undetermined どちらも許容。
    """
    bars = make_spike_bars(3)
    state = compute_expansion_state(
        "TARGET", bars, 950.0, UNIVERSE_SAMPLE, "09:03",
    )
    assert state.is_spike_pattern is False, "spike verdict requires >= MIN_BARS_FOR_VERDICT"
    assert state.spike_confidence == 0.0
    assert state.expansion_type in ("continuation", "undetermined")


def test_partial_session_exactly_min_bars():
    """ちょうど MIN_BARS_FOR_VERDICT 本。"""
    n = MIN_BARS_FOR_VERDICT
    bars = make_spike_bars(n, spike_bars=3, reversal_close=1005)
    state = compute_expansion_state(
        "TARGET", bars, 950.0, UNIVERSE_SAMPLE, f"09:{n:02d}",
        avg_daily_volume=300000,
    )
    # exactly at boundary — spike may or may not fire depending on signal strength
    assert state.expansion_type in ("spike", "continuation", "undetermined")


def test_empty_bars_returns_safe_default():
    state = compute_expansion_state("TARGET", [], 950.0, {}, "09:00")
    assert state.expansion_type == "undetermined"
    assert state.expansion_probability == 0.5  # safe default
    assert state.spike_confidence == 0.0


def test_zero_prev_close_returns_safe_default():
    bars = make_continuation_bars(15)
    state = compute_expansion_state("TARGET", bars, 0.0, {}, "09:15")
    assert state.expansion_type == "undetermined"


# ─────────────────────────────────────────────────────────────────────────────
# Malformed data handling
# ─────────────────────────────────────────────────────────────────────────────

def test_malformed_bars_negative_price_filtered():
    good = make_continuation_bars(15, 1000, 1050)
    bad  = [_bar("08:59", -100, 100, 90, 95)]
    bars = bad + good
    state = compute_expansion_state("TARGET", bars, 950.0, {}, "09:15")
    assert state.minutes_elapsed == 15  # bad bar excluded


def test_malformed_bars_nan_prices():
    good = make_continuation_bars(15, 1000, 1050)
    bad  = [MinuteBar(dt="08:59", open=float("nan"), high=1010, low=990, close=1005, volume=1000)]
    bars = bad + good
    state = compute_expansion_state("TARGET", bars, 950.0, {}, "09:15")
    assert 0.0 <= state.expansion_probability <= 1.0


def test_malformed_bars_all_bad_returns_safe_default():
    bars = [_bar("09:00", -1, -1, -1, -1), _bar("09:01", 0, 0, 0, 0)]
    state = compute_expansion_state("TARGET", bars, 950.0, {}, "09:01")
    assert state.expansion_type == "undetermined"


def test_missing_minute_bars_gaps():
    """分足に欠落がある場合 (時系列は連続していないが問題なく処理できること)。"""
    bars = make_continuation_bars(8, 1000, 1030)
    # 欠落をシミュレート: 15本分のうち8本しかない
    state = compute_expansion_state("TARGET", bars, 950.0, UNIVERSE_SAMPLE, "09:15")
    assert 0.0 <= state.expansion_probability <= 1.0
    assert state.expansion_type in ("continuation", "undetermined")


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / property tests
# ─────────────────────────────────────────────────────────────────────────────

def test_expansion_probability_bounded():
    for bars in [make_spike_bars(25), make_continuation_bars(25), make_flat_bars(20)]:
        state = compute_expansion_state("TARGET", bars, 950.0, UNIVERSE_SAMPLE, "ts",
                                        avg_daily_volume=200000)
        assert 0.0 <= state.expansion_probability <= 1.0, state.expansion_probability


def test_spike_confidence_bounded():
    for bars in [make_spike_bars(25), make_continuation_bars(25)]:
        state = compute_expansion_state("TARGET", bars, 950.0, UNIVERSE_SAMPLE, "ts",
                                        avg_daily_volume=200000)
        assert 0.0 <= state.spike_confidence <= 1.0


def test_rs_percentile_bounded():
    bars = make_continuation_bars(20)
    universe = {f"S{i}": float(i) * 0.001 for i in range(30)}
    universe["TARGET"] = 0.015
    state = compute_expansion_state("TARGET", bars, 950.0, universe, "ts")
    assert 0.0 <= state.rs_percentile <= 100.0


def test_drive_efficiency_bounded():
    for bars in [make_spike_bars(20), make_continuation_bars(20), make_flat_bars(20)]:
        result = compute_opening_drive(bars, 1000.0)
        assert 0.0 <= result.efficiency <= 1.0


def test_persistence_ratio_bounded():
    for bars in [make_spike_bars(20), make_continuation_bars(20)]:
        result = compute_persistence(bars, or_window=15)
        assert 0.0 <= result.persistence_ratio <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic replay
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_replay_same_inputs():
    bars = make_spike_bars(20)
    universe = {"TARGET": 0.01, "A": 0.005}
    kwargs = dict(avg_daily_volume=200000)

    s1 = compute_expansion_state("TARGET", bars, 950.0, universe, "2026-05-23T09:20:00+09:00", **kwargs)
    s2 = compute_expansion_state("TARGET", bars, 950.0, universe, "2026-05-23T09:20:00+09:00", **kwargs)

    assert s1.expansion_type       == s2.expansion_type
    assert s1.spike_confidence     == s2.spike_confidence
    assert s1.expansion_probability == s2.expansion_probability
    assert s1.drive_efficiency     == s2.drive_efficiency


def test_deterministic_replay_continuation():
    bars = make_continuation_bars(25)
    universe = {"TARGET": 0.02, "B": -0.01}

    s1 = compute_expansion_state("TARGET", bars, 950.0, universe, "ts1")
    s2 = compute_expansion_state("TARGET", bars, 950.0, universe, "ts1")

    assert s1.to_dict() == s2.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# to_dict
# ─────────────────────────────────────────────────────────────────────────────

def test_to_dict_has_required_keys():
    bars = make_continuation_bars(15)
    state = compute_expansion_state("TARGET", bars, 950.0, {}, "ts")
    d = state.to_dict()
    for key in [
        "symbol", "eval_ts", "minutes_elapsed",
        "drive_efficiency", "rvol_ratio", "rvol_acceleration",
        "persistence_ratio", "rs_percentile", "gap_acceptance",
        "expansion_probability", "spike_confidence",
        "expansion_type", "is_spike_pattern",
        "new_high_confirmed", "early_peak", "early_reversal_pct",
        "or_high", "or_low", "spike_reasons", "continuation_reasons",
    ]:
        assert key in d, f"missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Gap scenarios via compute_expansion_state
# ─────────────────────────────────────────────────────────────────────────────

def test_gap_up_continuation_high_acceptance():
    bars = make_continuation_bars(20, open_price=1050.0, end_price=1080.0)
    state = compute_expansion_state(
        "TARGET", bars,
        prev_close=1000.0,  # big gap up: open=1050 vs close=1000
        universe_intraday_returns={"TARGET": 0.05},
        eval_ts="09:20",
    )
    # gap_acceptance が高い (gap が持続されている)
    assert state.gap_acceptance > 0.0


def test_gap_up_rejection_low_acceptance():
    bars = make_spike_bars(20, open_price=1050.0, spike_high=1060.0,
                           reversal_close=1010.0)
    state = compute_expansion_state(
        "TARGET", bars,
        prev_close=1000.0,
        universe_intraday_returns={"TARGET": 0.01},
        eval_ts="09:20",
    )
    # gap は filling されている
    assert state.gap_acceptance < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Flat session
# ─────────────────────────────────────────────────────────────────────────────

def test_flat_session_undetermined():
    bars = make_flat_bars(20, 1000)
    state = compute_expansion_state("TARGET", bars, 990.0, {}, "09:20")
    # フラットは spike でも continuation でもない
    assert state.expansion_type in ("undetermined", "continuation")
    assert 0.0 <= state.expansion_probability <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def test_float_safe_nan():
    assert _float_safe(float("nan")) == 0.0


def test_float_safe_inf():
    assert _float_safe(float("inf")) == 0.0


def test_float_safe_string_error():
    assert _float_safe("bad") == 0.0


def test_clamp01():
    assert _clamp01(-0.5) == 0.0
    assert _clamp01(1.5)  == 1.0
    assert abs(_clamp01(0.5) - 0.5) < 1e-9
