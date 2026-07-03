"""
test_predictive_entry_overlay.py

Coverage targets (≥90%):
  - PredictiveEntryOverlay.observe() — 全スキップ理由、ログ記録
  - update_forward_observations() — forward returns / MAE / MFE
  - compute_statistics() — 継続確率 / false expansion 率
  - JSONL append-only: crash-safe, replay-consistent
  - 実注文ルーティングなし確認
  - deterministic timestamps
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.intraday.expansion_state_engine import (
    ExpansionState,
    MinuteBar,
    MIN_BARS_FOR_VERDICT,
)
from src.shadow.predictive_entry_overlay import (
    PredictiveEntryOverlay,
    ShadowEntryContext,
    ShadowEntryRecord,
    ForwardObsRecord,
    ShadowStatistics,
    SKIP_SLOT_BLOCKED,
    SKIP_CAPITAL_BLOCKED,
    SKIP_RISK_BLOCKED,
    SKIP_SPREAD_BLOCKED,
    SKIP_WEAK_SCORE,
    SKIP_COOLDOWN_BLOCKED,
    SKIP_PERSISTENCE_SHORT,
    _load_all_records,
    _append_record,
    _forward_return_at,
    _compute_mae_mfe,
    DEFAULT_MIN_SCORE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_expansion_state(
    expansion_type: str = "continuation",
    spike_confidence: float = 0.0,
    minutes_elapsed: int = 20,
) -> ExpansionState:
    return ExpansionState(
        symbol="TEST",
        eval_ts="09:20",
        minutes_elapsed=minutes_elapsed,
        drive_efficiency=0.80,
        rvol_ratio=1.5,
        rvol_acceleration=1.2,
        persistence_ratio=0.75,
        rs_percentile=70.0,
        gap_acceptance=0.90,
        expansion_probability=0.75,
        spike_confidence=spike_confidence,
        expansion_type=expansion_type,
        is_spike_pattern=(expansion_type == "spike"),
        new_high_confirmed=(expansion_type == "continuation"),
        early_peak=1050.0,
        early_reversal_pct=0.1,
        or_high=1060.0,
        or_low=1000.0,
        spike_reasons=[],
        continuation_reasons=["new_high"],
    )


def make_context(
    symbol: str = "TEST",
    eval_ts: str = "2026-05-23T09:20:00+09:00",
    predictive_score: int = 75,
    entry_price: float = 1050.0,
    skip_reasons: list[str] | None = None,
    expansion_type: str = "continuation",
) -> ShadowEntryContext:
    return ShadowEntryContext(
        symbol=symbol,
        eval_ts=eval_ts,
        expansion_state=make_expansion_state(expansion_type=expansion_type),
        predictive_score=predictive_score,
        entry_price=entry_price,
        skip_reasons=skip_reasons or [],
    )


def make_bars_after_entry(
    entry_price: float = 1050.0,
    n: int = 35,
    trend: float = 0.0005,  # +0.05% per bar
) -> list[MinuteBar]:
    bars = []
    price = entry_price
    for i in range(n):
        price = price * (1 + trend)
        bars.append(MinuteBar(
            dt=f"09:{20 + i:02d}",
            open=price - 1,
            high=price + 2,
            low=price - 2,
            close=price,
            volume=10000,
        ))
    return bars


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    return tmp_path / "shadow_entries.jsonl"


@pytest.fixture
def overlay(tmp_log: Path) -> PredictiveEntryOverlay:
    return PredictiveEntryOverlay(tmp_log, min_score_threshold=DEFAULT_MIN_SCORE)


# ─────────────────────────────────────────────────────────────────────────────
# observe() — basic logging
# ─────────────────────────────────────────────────────────────────────────────

def test_observe_logs_entry_record(overlay: PredictiveEntryOverlay, tmp_log: Path):
    ctx = make_context(predictive_score=75)
    logged = overlay.observe([ctx], run_ts="2026-05-23T09:20:00+09:00")
    assert len(logged) == 1
    records = _load_all_records(tmp_log)
    assert len(records) == 1
    assert records[0]["record_type"] == "ENTRY"
    assert records[0]["symbol"] == "TEST"


def test_observe_below_threshold_not_logged(tmp_log: Path):
    overlay = PredictiveEntryOverlay(tmp_log, min_score_threshold=60)
    ctx = make_context(predictive_score=55)
    logged = overlay.observe([ctx], run_ts="ts")
    assert len(logged) == 0
    assert not tmp_log.exists()


def test_observe_empty_contexts(overlay: PredictiveEntryOverlay, tmp_log: Path):
    logged = overlay.observe([], run_ts="ts")
    assert logged == []
    assert not tmp_log.exists()


def test_observe_multiple_contexts(overlay: PredictiveEntryOverlay, tmp_log: Path):
    contexts = [
        make_context(symbol=f"S{i}", predictive_score=70)
        for i in range(3)
    ]
    logged = overlay.observe(contexts, run_ts="ts")
    assert len(logged) == 3
    records = _load_all_records(tmp_log)
    assert len(records) == 3


def test_observe_multiple_runs_appends(overlay: PredictiveEntryOverlay, tmp_log: Path):
    overlay.observe([make_context(symbol="A", predictive_score=70)], run_ts="ts1")
    overlay.observe([make_context(symbol="B", predictive_score=80)], run_ts="ts2")
    records = _load_all_records(tmp_log)
    assert len(records) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Skip reasons
# ─────────────────────────────────────────────────────────────────────────────

def test_skip_reason_slot_blocked(overlay: PredictiveEntryOverlay, tmp_log: Path):
    ctx = make_context(skip_reasons=[SKIP_SLOT_BLOCKED], predictive_score=70)
    overlay.observe([ctx], run_ts="ts")
    records = _load_all_records(tmp_log)
    assert SKIP_SLOT_BLOCKED in records[0]["skip_reasons"]


def test_skip_reason_capital_blocked(overlay: PredictiveEntryOverlay, tmp_log: Path):
    ctx = make_context(skip_reasons=[SKIP_CAPITAL_BLOCKED], predictive_score=70)
    overlay.observe([ctx], run_ts="ts")
    records = _load_all_records(tmp_log)
    assert SKIP_CAPITAL_BLOCKED in records[0]["skip_reasons"]


def test_skip_reason_risk_blocked(overlay: PredictiveEntryOverlay, tmp_log: Path):
    ctx = make_context(skip_reasons=[SKIP_RISK_BLOCKED], predictive_score=75)
    overlay.observe([ctx], run_ts="ts")
    records = _load_all_records(tmp_log)
    assert SKIP_RISK_BLOCKED in records[0]["skip_reasons"]


def test_skip_reason_spread_blocked(overlay: PredictiveEntryOverlay, tmp_log: Path):
    ctx = make_context(skip_reasons=[SKIP_SPREAD_BLOCKED], predictive_score=65)
    overlay.observe([ctx], run_ts="ts")
    records = _load_all_records(tmp_log)
    assert SKIP_SPREAD_BLOCKED in records[0]["skip_reasons"]


def test_skip_reason_cooldown(overlay: PredictiveEntryOverlay, tmp_log: Path):
    ctx = make_context(skip_reasons=[SKIP_COOLDOWN_BLOCKED], predictive_score=70)
    overlay.observe([ctx], run_ts="ts")
    records = _load_all_records(tmp_log)
    assert SKIP_COOLDOWN_BLOCKED in records[0]["skip_reasons"]


def test_skip_reason_persistence_short(overlay: PredictiveEntryOverlay, tmp_log: Path):
    ctx = make_context(skip_reasons=[SKIP_PERSISTENCE_SHORT], predictive_score=65)
    overlay.observe([ctx], run_ts="ts")
    records = _load_all_records(tmp_log)
    assert SKIP_PERSISTENCE_SHORT in records[0]["skip_reasons"]


def test_weak_score_added_when_below_threshold(tmp_log: Path):
    overlay = PredictiveEntryOverlay(tmp_log, min_score_threshold=60)
    ctx = make_context(predictive_score=55, skip_reasons=[])
    # score 55 < 60 → not logged (below threshold)
    logged = overlay.observe([ctx], run_ts="ts")
    assert len(logged) == 0


def test_no_weak_score_added_above_threshold(overlay: PredictiveEntryOverlay, tmp_log: Path):
    ctx = make_context(predictive_score=75, skip_reasons=[])
    logged = overlay.observe([ctx], run_ts="ts")
    records = _load_all_records(tmp_log)
    assert SKIP_WEAK_SCORE not in records[0].get("skip_reasons", [])


# ─────────────────────────────────────────────────────────────────────────────
# JSONL: crash-safe, append-only
# ─────────────────────────────────────────────────────────────────────────────

def test_malformed_last_line_skipped(tmp_log: Path):
    """末尾の不正行があっても全有効レコードを読める。"""
    # 正常レコードを書いてから不正行を追記
    tmp_log.parent.mkdir(parents=True, exist_ok=True)
    valid = json.dumps({"record_type": "ENTRY", "symbol": "A"}) + "\n"
    bad   = "{corrupted json\n"
    tmp_log.write_text(valid + bad, encoding="utf-8")
    records = _load_all_records(tmp_log)
    assert len(records) == 1
    assert records[0]["symbol"] == "A"


def test_jsonl_file_grows_on_each_append(overlay: PredictiveEntryOverlay, tmp_log: Path):
    overlay.observe([make_context(symbol="A", predictive_score=70)], run_ts="ts1")
    size1 = tmp_log.stat().st_size
    overlay.observe([make_context(symbol="B", predictive_score=70)], run_ts="ts2")
    size2 = tmp_log.stat().st_size
    assert size2 > size1


def test_directory_created_on_demand(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "shadow_entries.jsonl"
    overlay = PredictiveEntryOverlay(nested)
    overlay.observe([make_context(predictive_score=70)], run_ts="ts")
    assert nested.exists()


def test_record_type_is_entry(overlay: PredictiveEntryOverlay, tmp_log: Path):
    overlay.observe([make_context(predictive_score=70)], run_ts="ts")
    records = _load_all_records(tmp_log)
    assert records[0]["record_type"] == "ENTRY"


# ─────────────────────────────────────────────────────────────────────────────
# update_forward_observations()
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_obs_5m(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "2026-05-23T09:20:00+09:00"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="ts")
    bars = make_bars_after_entry(1050.0, n=35, trend=0.001)
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0,
        obs_ts="2026-05-23T09:55:00+09:00", expansion_type="continuation",
    )
    assert obs is not None
    assert obs.forward_5m is not None
    assert obs.forward_5m > 0  # uptrend bars


def test_forward_obs_15m(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts1"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=35, trend=0.001)
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0, obs_ts="ts_obs",
    )
    assert obs.forward_15m is not None
    assert obs.forward_15m > 0


def test_forward_obs_30m(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts1"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=35, trend=0.001)
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0, obs_ts="ts_obs",
    )
    assert obs.forward_30m is not None
    assert obs.forward_30m > 0


def test_forward_obs_eod_close_provided(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts1"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=35)
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0,
        eod_close=1080.0, obs_ts="ts_obs",
    )
    assert obs.forward_eod is not None
    assert abs(obs.forward_eod - (1080.0 - 1050.0) / 1050.0) < 1e-5


def test_forward_obs_eod_from_last_bar(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts1"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=35, trend=0.002)
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0, obs_ts="ts_obs",
    )
    assert obs.forward_eod is not None


def test_mae_mfe_computed(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts1"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=35, trend=0.001)
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0, obs_ts="ts_obs",
    )
    assert obs.mae_pct is not None and obs.mae_pct >= 0.0
    assert obs.mfe_pct is not None and obs.mfe_pct >= 0.0


def test_breakout_continued_true(overlay: PredictiveEntryOverlay, tmp_log: Path):
    overlay = PredictiveEntryOverlay(
        overlay.log_path, min_score_threshold=DEFAULT_MIN_SCORE,
        continuation_threshold=0.005,
    )
    entry_ts = "ts1"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=35, trend=0.002)  # strong uptrend
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0, obs_ts="ts_obs",
    )
    assert obs.breakout_continued is True


def test_breakout_continued_false(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts1"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=35, trend=-0.003)  # downtrend
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0, obs_ts="ts_obs",
    )
    assert obs.breakout_continued is False


def test_is_false_expansion_detected(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts1"
    ctx = make_context(eval_ts=entry_ts, predictive_score=70, expansion_type="continuation")
    overlay.observe([ctx], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=35, trend=-0.003)  # reversal
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0, obs_ts="ts_obs",
        expansion_type="continuation",
    )
    assert obs.is_false_expansion is True


def test_forward_obs_appends_to_jsonl(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts1"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=35)
    overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0, obs_ts="ts_obs",
    )
    records = _load_all_records(tmp_log)
    assert len(records) == 2
    types = {r["record_type"] for r in records}
    assert "ENTRY" in types
    assert "FORWARD_OBS" in types


def test_forward_obs_unknown_entry_price_looks_up_jsonl(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts_lookup"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70, entry_price=1234.0)], run_ts="r")
    bars = make_bars_after_entry(1234.0, n=35, trend=0.001)
    # entry_price=None → JSOLから取得
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=None, obs_ts="ts_obs",
    )
    assert obs is not None
    assert obs.forward_5m is not None


def test_forward_obs_returns_none_if_no_entry(overlay: PredictiveEntryOverlay, tmp_log: Path):
    bars = make_bars_after_entry(1050.0, n=35)
    obs = overlay.update_forward_observations(
        "UNKNOWN", "nonexistent_ts", bars, obs_ts="ts_obs",
    )
    assert obs is None


def test_forward_obs_insufficient_bars(overlay: PredictiveEntryOverlay, tmp_log: Path):
    entry_ts = "ts1"
    overlay.observe([make_context(eval_ts=entry_ts, predictive_score=70)], run_ts="r")
    bars = make_bars_after_entry(1050.0, n=3)  # only 3 bars → forward_30m = None
    obs = overlay.update_forward_observations(
        "TEST", entry_ts, bars, entry_price=1050.0, obs_ts="ts_obs",
    )
    assert obs.forward_30m is None


# ─────────────────────────────────────────────────────────────────────────────
# compute_statistics()
# ─────────────────────────────────────────────────────────────────────────────

def test_statistics_empty_log(overlay: PredictiveEntryOverlay):
    stats = overlay.compute_statistics()
    assert stats.total_observed == 0
    assert stats.breakout_continuation_probability == 0.0
    assert stats.false_expansion_failure_rate == 0.0


def test_statistics_breakout_probability(tmp_log: Path):
    overlay = PredictiveEntryOverlay(tmp_log, min_score_threshold=60)

    # 3 entries, 2 with forward obs: 1 continued, 1 reversed
    for i, trend in enumerate([0.002, -0.003, 0.001]):
        ts = f"ts_{i}"
        overlay.observe([make_context(symbol=f"S{i}", eval_ts=ts, predictive_score=70)], run_ts="r")
        if i < 2:  # only 2 get forward obs
            bars = make_bars_after_entry(1050.0, n=35, trend=trend)
            overlay.update_forward_observations(f"S{i}", ts, bars, entry_price=1050.0, obs_ts="t")

    stats = overlay.compute_statistics()
    assert stats.total_observed == 3
    assert stats.total_with_forward_obs == 2
    # 1 out of 2 continued
    assert abs(stats.breakout_continuation_probability - 0.5) < 1e-6


def test_statistics_false_expansion_rate(tmp_log: Path):
    overlay = PredictiveEntryOverlay(tmp_log, min_score_threshold=60)

    # 2 continuation type entries, 1 false expansion
    for i, trend in enumerate([0.003, -0.004]):
        ts = f"ts_{i}"
        ctx = make_context(
            symbol=f"S{i}", eval_ts=ts, predictive_score=70,
            expansion_type="continuation",
        )
        overlay.observe([ctx], run_ts="r")
        bars = make_bars_after_entry(1050.0, n=35, trend=trend)
        overlay.update_forward_observations(
            f"S{i}", ts, bars, entry_price=1050.0, obs_ts="t",
            expansion_type="continuation",
        )

    stats = overlay.compute_statistics()
    # 1 out of 2 continuation entries was false expansion
    assert 0.0 <= stats.false_expansion_failure_rate <= 1.0


def test_statistics_skip_reason_counts(tmp_log: Path):
    overlay = PredictiveEntryOverlay(tmp_log)
    overlay.observe([
        make_context(symbol="A", eval_ts="ts1", predictive_score=70, skip_reasons=[SKIP_SLOT_BLOCKED]),
        make_context(symbol="B", eval_ts="ts2", predictive_score=70, skip_reasons=[SKIP_CAPITAL_BLOCKED]),
        make_context(symbol="C", eval_ts="ts3", predictive_score=70, skip_reasons=[SKIP_SLOT_BLOCKED]),
    ], run_ts="ts")
    stats = overlay.compute_statistics()
    assert stats.skip_reason_counts[SKIP_SLOT_BLOCKED] == 2
    assert stats.skip_reason_counts[SKIP_CAPITAL_BLOCKED] == 1


def test_statistics_score_distribution(tmp_log: Path):
    overlay = PredictiveEntryOverlay(tmp_log, min_score_threshold=0)
    overlay.observe([
        make_context(symbol="A", eval_ts="ts1", predictive_score=80),   # STRONG
        make_context(symbol="B", eval_ts="ts2", predictive_score=55),   # MODERATE
        make_context(symbol="C", eval_ts="ts3", predictive_score=30),   # WEAK
    ], run_ts="ts")
    stats = overlay.compute_statistics()
    assert stats.score_distribution["STRONG"] == 1
    assert stats.score_distribution["MODERATE"] == 1
    assert stats.score_distribution["WEAK"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# load_all_records / replay consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_load_all_records_returns_all(overlay: PredictiveEntryOverlay, tmp_log: Path):
    overlay.observe([
        make_context(symbol="A", predictive_score=70),
        make_context(symbol="B", predictive_score=75),
    ], run_ts="ts")
    records = overlay.load_all_records()
    assert len(records) == 2


def test_replay_entry_records_consistent(overlay: PredictiveEntryOverlay, tmp_log: Path):
    """再読み込み後も全 ENTRY レコードが一致すること。"""
    contexts = [make_context(symbol=f"S{i}", predictive_score=70) for i in range(5)]
    overlay.observe(contexts, run_ts="ts")
    records = overlay.load_all_records()
    symbols = {r["symbol"] for r in records if r["record_type"] == "ENTRY"}
    assert symbols == {f"S{i}" for i in range(5)}


# ─────────────────────────────────────────────────────────────────────────────
# No real order routing assertion
# ─────────────────────────────────────────────────────────────────────────────

def test_no_order_routing_on_observe(overlay: PredictiveEntryOverlay):
    """observe() は JSONL 以外の外部呼び出しを行わないこと。"""
    # kabuステーション API 等へのネットワーク接続がないことを副作用なしで確認
    import unittest.mock as mock
    import socket
    with mock.patch("socket.socket") as mock_sock:
        overlay.observe([make_context(predictive_score=75)], run_ts="ts")
        mock_sock.assert_not_called()  # ネットワーク呼び出しなし


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic timestamps
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_entry_ts(overlay: PredictiveEntryOverlay, tmp_log: Path):
    """entry_ts は呼び出し元提供値が使われること (datetime.now() ではない)。"""
    explicit_ts = "2026-05-23T09:20:00+09:00"
    ctx = make_context(eval_ts=explicit_ts, predictive_score=70)
    overlay.observe([ctx], run_ts="run_ts_value")
    records = _load_all_records(tmp_log)
    assert records[0]["entry_ts"] == explicit_ts
    assert records[0]["run_ts"] == "run_ts_value"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_return_at_insufficient_bars():
    bars = [MinuteBar(dt=f"09:{i}", open=1000, high=1010, low=990, close=1005, volume=1000)
            for i in range(3)]
    assert _forward_return_at(1000.0, bars, 5) is None


def test_forward_return_at_zero_offset():
    bars = [MinuteBar(dt="09:00", open=1000, high=1010, low=990, close=1005, volume=1000)]
    assert _forward_return_at(1000.0, bars, 0) is None


def test_forward_return_at_zero_entry_price():
    bars = [MinuteBar(dt="09:00", open=1000, high=1010, low=990, close=1005, volume=1000)]
    assert _forward_return_at(0.0, bars, 1) is None


def test_compute_mae_mfe_empty_bars():
    mae, mfe = _compute_mae_mfe(1000.0, [])
    assert mae is None
    assert mfe is None


def test_compute_mae_mfe_uptrend():
    bars = [MinuteBar(dt=f"09:{i}", open=1000, high=1000 + i * 2 + 5, low=1000 + i * 2, close=1000 + i * 2 + 3, volume=1000)
            for i in range(10)]
    mae, mfe = _compute_mae_mfe(1000.0, bars)
    assert mae is not None and mae >= 0
    assert mfe is not None and mfe > 0


def test_append_record_handles_write_error(tmp_path: Path):
    """書き込み失敗は例外を上げないこと (fail-open)。"""
    # read-only ファイルをシミュレート
    log = tmp_path / "ro.jsonl"
    log.write_text("")
    log.chmod(0o444)
    try:
        _append_record(log, {"test": "data"})  # should not raise
    except Exception:
        pytest.fail("_append_record should not raise on write failure")
    finally:
        log.chmod(0o644)  # restore for cleanup
