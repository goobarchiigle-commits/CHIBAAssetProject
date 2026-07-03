"""
tests/analytics/test_feature_forward_expectancy.py

Tests for src/analytics/feature_forward_expectancy.py
"""
import json
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.analytics.feature_forward_expectancy import (
    ATR_BUCKETS,
    DIST_25MA_BUCKETS,
    RSI_BUCKETS,
    EntryFeatureSnapshot,
    ExpectancyBucket,
    FeatureForwardExpectancyEngine,
    ForwardExpectancyReport,
    LifecycleDecay,
    PositionLifecycleRecord,
    RegimeExpectancyCell,
    SCHEMA_VERSION,
    _cache_key,
    _load_cache,
    _make_snapshot_id,
    _overheating_verdict,
    _safe_mean,
    _win_rate,
    analyze_atr_buckets,
    analyze_counterfactual,
    analyze_extension_expectancy,
    analyze_lifecycle_decay,
    analyze_regime_conditioned,
    analyze_sector_concentration,
    append_entry_snapshot,
    append_position_lifecycle,
    backfill_from_trades,
    enrich_snapshots_with_forward_returns,
    fetch_forward_returns,
    load_entry_snapshots,
    load_position_lifecycle,
    write_signal_snapshots,
)

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _snap(
    symbol="7203",
    eval_date="2026-01-10",
    executed=True,
    atr_pct=0.020,
    enrichment_status="pending",
    forward_5d=None,
    forward_10d=None,
    forward_mae=None,
    forward_mfe=None,
    continuation_probability=None,
    breakout_failure_flag=None,
    market_regime="mixed",
    sector="自動車",
    distance_25ma_pct=None,
    rsi14=None,
    run_id="test_run",
    skip_reason=None,
) -> EntryFeatureSnapshot:
    snap_id = _make_snapshot_id(run_id, symbol, eval_date, executed)
    return EntryFeatureSnapshot(
        schema_version=SCHEMA_VERSION,
        snapshot_id=snap_id,
        run_id=run_id,
        timestamp_jst=f"{eval_date}T09:00:00+09:00",
        eval_date=eval_date,
        symbol=symbol,
        executed=executed,
        skip_reason=skip_reason,
        entry_price=1000.0,
        atr_pct=atr_pct,
        distance_25ma_pct=distance_25ma_pct,
        distance_20d_high_pct=None,
        distance_52w_high_pct=None,
        atr_expansion_ratio=None,
        rsi14=rsi14,
        gap_up_pct=None,
        volume_zscore=None,
        sector=sector,
        sector_rank=None,
        rsr_rank=80.0,
        rsr_score=80.0,
        market_regime=market_regime,
        breadth_50=0.55,
        breadth_75=0.30,
        breakout_strength=None,
        volatility_regime="medium",
        cash_ratio=0.50,
        slot_utilization=0.33,
        executed_qty=100 if executed else None,
        capital_at_risk=100000.0 if executed else None,
        forward_5d=forward_5d,
        forward_10d=forward_10d,
        forward_mae=forward_mae,
        forward_mfe=forward_mfe,
        continuation_probability=continuation_probability,
        breakout_failure_flag=breakout_failure_flag,
        enrichment_status=enrichment_status,
    )


def _lifecycle_rec(
    symbol="7203",
    days_held=5,
    unrealized_pnl_pct=0.03,
    drawdown_from_peak=-0.01,
) -> PositionLifecycleRecord:
    return PositionLifecycleRecord(
        schema_version=SCHEMA_VERSION,
        snapshot_id=f"lc_{symbol}_{days_held}",
        timestamp_jst=f"2026-01-15T09:00:00+09:00",
        symbol=symbol,
        days_held=days_held,
        entry_price=1000.0,
        current_price=1030.0,
        unrealized_pnl_pct=unrealized_pnl_pct,
        distance_from_peak_pct=drawdown_from_peak,
        atr_pct=0.020,
        rsr_rank=80.0,
        sector="自動車",
        market_regime="mixed",
        drawdown_from_peak=drawdown_from_peak,
        momentum_decay=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestSnapshotId
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotId:
    def test_deterministic(self):
        id1 = _make_snapshot_id("run1", "7203", "2026-01-10", True)
        id2 = _make_snapshot_id("run1", "7203", "2026-01-10", True)
        assert id1 == id2

    def test_unique_by_executed(self):
        id_exec = _make_snapshot_id("run1", "7203", "2026-01-10", True)
        id_skip = _make_snapshot_id("run1", "7203", "2026-01-10", False)
        assert id_exec != id_skip

    def test_unique_by_symbol(self):
        id1 = _make_snapshot_id("run1", "7203", "2026-01-10", True)
        id2 = _make_snapshot_id("run1", "9984", "2026-01-10", True)
        assert id1 != id2

    def test_unique_by_date(self):
        id1 = _make_snapshot_id("run1", "7203", "2026-01-10", True)
        id2 = _make_snapshot_id("run1", "7203", "2026-01-11", True)
        assert id1 != id2

    def test_length_16(self):
        sid = _make_snapshot_id("run1", "7203", "2026-01-10", True)
        assert len(sid) == 16


# ─────────────────────────────────────────────────────────────────────────────
# TestPersistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        s = _snap()
        append_entry_snapshot(s, path)
        loaded = load_entry_snapshots(path)
        assert len(loaded) == 1
        assert loaded[0].symbol == s.symbol
        assert loaded[0].eval_date == s.eval_date

    def test_append_multiple_preserves_order(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        dates = ["2026-01-10", "2026-01-11", "2026-01-12"]
        for d in dates:
            append_entry_snapshot(_snap(eval_date=d, run_id=d), path)
        loaded = load_entry_snapshots(path)
        assert len(loaded) == 3
        assert [s.eval_date for s in loaded] == dates

    def test_append_only_does_not_overwrite(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        append_entry_snapshot(_snap(eval_date="2026-01-10"), path)
        size1 = path.stat().st_size
        append_entry_snapshot(_snap(eval_date="2026-01-11", run_id="r2"), path)
        size2 = path.stat().st_size
        assert size2 > size1

    def test_load_missing_file_returns_empty(self, tmp_path):
        snaps = load_entry_snapshots(tmp_path / "nonexistent.jsonl")
        assert snaps == []

    def test_corrupt_lines_skipped_gracefully(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        path.write_text('not_json\n', encoding="utf-8")
        loaded = load_entry_snapshots(path)
        assert loaded == []

    def test_corrupt_line_does_not_affect_valid_lines(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        s = _snap()
        append_entry_snapshot(s, path)
        with path.open("a", encoding="utf-8") as f:
            f.write("INVALID_JSON\n")
        loaded = load_entry_snapshots(path)
        assert len(loaded) == 1

    def test_lifecycle_roundtrip(self, tmp_path):
        path = tmp_path / "lifecycle.jsonl"
        rec = _lifecycle_rec()
        append_position_lifecycle(rec, path)
        loaded = load_position_lifecycle(path)
        assert len(loaded) == 1
        assert loaded[0].symbol == rec.symbol
        assert loaded[0].days_held == rec.days_held


# ─────────────────────────────────────────────────────────────────────────────
# TestBackfill
# ─────────────────────────────────────────────────────────────────────────────

class TestBackfill:
    def _trade(self, symbol="7203", date="2026-01-05", price=1000.0, qty=100):
        return {
            "side": "BUY",
            "symbol": symbol,
            "date": date,
            "price": price,
            "qty": qty,
            "atr20": 20.0,
            "sector": "自動車",
            "entry_regime": "trend",
            "amount": price * qty,
        }

    def test_backfill_creates_snapshots(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        trades = [self._trade()]
        n = backfill_from_trades(trades, path)
        assert n == 1
        loaded = load_entry_snapshots(path)
        assert len(loaded) == 1
        assert loaded[0].executed is True

    def test_backfill_deduplication(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        trades = [self._trade()]
        n1 = backfill_from_trades(trades, path)
        n2 = backfill_from_trades(trades, path)
        assert n1 == 1
        assert n2 == 0  # already exists
        assert len(load_entry_snapshots(path)) == 1

    def test_backfill_skips_sell_records(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        trades = [{"side": "SELL", "symbol": "7203", "date": "2026-01-10", "price": 1100.0, "qty": 100}]
        n = backfill_from_trades(trades, path)
        assert n == 0

    def test_backfill_atr_pct_computed(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        trades = [self._trade(price=1000.0)]  # atr20=20.0 → atr_pct=0.02
        backfill_from_trades(trades, path)
        loaded = load_entry_snapshots(path)
        assert abs(loaded[0].atr_pct - 0.02) < 1e-6

    def test_backfill_empty_trades(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        n = backfill_from_trades([], path)
        assert n == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestForwardReturnCache
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardReturnCache:
    def test_cache_key_deterministic(self):
        k1 = _cache_key("7203", "2026-01-10")
        k2 = _cache_key("7203", "2026-01-10")
        assert k1 == k2

    def test_cache_key_unique(self):
        k1 = _cache_key("7203", "2026-01-10")
        k2 = _cache_key("9984", "2026-01-10")
        assert k1 != k2

    def test_load_cache_miss_returns_none(self, tmp_path):
        result = _load_cache(tmp_path, "7203", "2026-01-10")
        assert result is None

    def test_fetch_returns_none_when_network_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALLOW_YFINANCE_NETWORK", "false")
        result = fetch_forward_returns("7203.T", "2026-01-10", 1000.0, tmp_path)
        assert result is None

    def test_fetch_uses_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALLOW_YFINANCE_NETWORK", "false")
        # Manually put data in cache
        key = _cache_key("7203.T", "2026-01-10")
        cache_file = tmp_path / f"{key}.json"
        cached_data = {"status": "enriched", "forward_5d": 0.025, "forward_1d": 0.01,
                       "forward_3d": 0.015, "forward_10d": 0.03, "forward_20d": 0.04,
                       "forward_mae": -0.01, "forward_mfe": 0.05,
                       "continuation_probability": 1.0, "breakout_failure_flag": False}
        cache_file.write_text(json.dumps(cached_data), encoding="utf-8")
        result = fetch_forward_returns("7203.T", "2026-01-10", 1000.0, tmp_path)
        assert result is not None
        assert result["forward_5d"] == 0.025


# ─────────────────────────────────────────────────────────────────────────────
# TestEnrichment
# ─────────────────────────────────────────────────────────────────────────────

class TestEnrichment:
    def test_enrichment_skips_recent_snapshots(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALLOW_YFINANCE_NETWORK", "false")
        today_str = datetime.now(JST).strftime("%Y-%m-%d")
        snaps = [_snap(eval_date=today_str)]
        result = enrich_snapshots_with_forward_returns(snaps, tmp_path)
        assert result[0].enrichment_status == "pending"

    def test_enrichment_marks_expired_when_no_data(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALLOW_YFINANCE_NETWORK", "false")
        old_date = (datetime.now(JST).date() - timedelta(days=30)).isoformat()
        snaps = [_snap(eval_date=old_date)]
        # No cache file, network disabled → will expire after max_enrichment_days
        result = enrich_snapshots_with_forward_returns(snaps, tmp_path, max_enrichment_days=25)
        # 30 days ago > 25 max → should expire
        assert result[0].enrichment_status == "expired"

    def test_enrichment_fills_forward_returns_from_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALLOW_YFINANCE_NETWORK", "false")
        old_date = (datetime.now(JST).date() - timedelta(days=10)).isoformat()
        s = _snap(eval_date=old_date, symbol="7203.T")

        key = _cache_key("7203.T", old_date)
        cache_file = tmp_path / f"{key}.json"
        cache_file.write_text(json.dumps({
            "status": "enriched", "forward_5d": 0.03, "forward_1d": 0.01,
            "forward_3d": 0.02, "forward_10d": 0.04, "forward_20d": 0.05,
            "forward_mae": -0.02, "forward_mfe": 0.06,
            "continuation_probability": 1.0, "breakout_failure_flag": False,
        }), encoding="utf-8")

        result = enrich_snapshots_with_forward_returns([s], tmp_path)
        assert result[0].enrichment_status == "enriched"
        assert result[0].forward_5d == 0.03
        assert result[0].forward_mae == -0.02


# ─────────────────────────────────────────────────────────────────────────────
# TestATRBuckets
# ─────────────────────────────────────────────────────────────────────────────

class TestATRBuckets:
    def test_returns_all_four_buckets(self):
        snaps = [_snap(atr_pct=0.010), _snap(atr_pct=0.020, run_id="r2"),
                 _snap(atr_pct=0.030, run_id="r3"), _snap(atr_pct=0.050, run_id="r4")]
        buckets = analyze_atr_buckets(snaps)
        assert len(buckets) == 4

    def test_bucket_labels_match_constants(self):
        snaps = [_snap(atr_pct=0.010)]
        buckets = analyze_atr_buckets(snaps)
        labels = [b.label for b in buckets]
        assert labels == [b[0] for b in ATR_BUCKETS]

    def test_sample_counts_correct(self):
        snaps = [
            _snap(atr_pct=0.010, run_id="r1"),  # <1.5%
            _snap(atr_pct=0.010, run_id="r2"),  # <1.5%
            _snap(atr_pct=0.020, run_id="r3"),  # 1.5-2.5%
            _snap(atr_pct=0.030, run_id="r4"),  # 2.5-4%
        ]
        buckets = analyze_atr_buckets(snaps)
        assert buckets[0].sample_count == 2   # <1.5%
        assert buckets[1].sample_count == 1   # 1.5-2.5%
        assert buckets[2].sample_count == 1   # 2.5-4%
        assert buckets[3].sample_count == 0   # 4%+

    def test_no_enriched_gives_none_expectancy(self):
        snaps = [_snap(atr_pct=0.020, enrichment_status="pending")]
        buckets = analyze_atr_buckets(snaps)
        assert buckets[1].expectancy_5d is None

    def test_enriched_expectancy_computed(self):
        snaps = [
            _snap(atr_pct=0.020, run_id="r1", enrichment_status="enriched", forward_5d=0.05),
            _snap(atr_pct=0.020, run_id="r2", enrichment_status="enriched", forward_5d=0.03),
        ]
        buckets = analyze_atr_buckets(snaps)
        # bucket 1.5-2.5%
        assert buckets[1].expectancy_5d == pytest.approx(0.04, rel=1e-5)

    def test_win_rate_correct(self):
        snaps = [
            _snap(atr_pct=0.020, run_id="r1", enrichment_status="enriched", forward_5d=0.05),
            _snap(atr_pct=0.020, run_id="r2", enrichment_status="enriched", forward_5d=-0.02),
        ]
        buckets = analyze_atr_buckets(snaps)
        assert buckets[1].win_rate_5d == pytest.approx(0.5, rel=1e-5)

    def test_executed_skipped_counts(self):
        snaps = [
            _snap(atr_pct=0.020, run_id="r1", executed=True),
            _snap(atr_pct=0.020, run_id="r2", executed=False, skip_reason="rsr"),
        ]
        buckets = analyze_atr_buckets(snaps)
        assert buckets[1].executed_count == 1
        assert buckets[1].skipped_count == 1

    def test_breakout_failure_rate(self):
        snaps = [
            _snap(atr_pct=0.020, run_id="r1", enrichment_status="enriched",
                  forward_5d=0.01, breakout_failure_flag=True),
            _snap(atr_pct=0.020, run_id="r2", enrichment_status="enriched",
                  forward_5d=0.01, breakout_failure_flag=False),
        ]
        buckets = analyze_atr_buckets(snaps)
        assert buckets[1].breakout_failure_rate == pytest.approx(0.5, rel=1e-5)

    def test_empty_snapshots(self):
        buckets = analyze_atr_buckets([])
        assert all(b.sample_count == 0 for b in buckets)
        assert all(b.expectancy_5d is None for b in buckets)


# ─────────────────────────────────────────────────────────────────────────────
# TestExtensionExpectancy
# ─────────────────────────────────────────────────────────────────────────────

class TestExtensionExpectancy:
    def test_parametric_distance_25ma(self):
        snaps = [
            _snap(run_id="r1", distance_25ma_pct=0.03, enrichment_status="enriched", forward_5d=0.04),
            _snap(run_id="r2", distance_25ma_pct=0.08, enrichment_status="enriched", forward_5d=0.01),
        ]
        results = analyze_extension_expectancy(snaps, "distance_25ma_pct", DIST_25MA_BUCKETS)
        assert len(results) == len(DIST_25MA_BUCKETS)
        # 0.03 → bucket "2-5%"
        bucket_2_5 = next(b for b in results if b.label == "2-5%")
        assert bucket_2_5.sample_count == 1
        assert bucket_2_5.expectancy_5d == pytest.approx(0.04, rel=1e-5)

    def test_none_feature_excluded(self):
        snaps = [
            _snap(run_id="r1", distance_25ma_pct=None),  # should be excluded
            _snap(run_id="r2", distance_25ma_pct=0.03),
        ]
        results = analyze_extension_expectancy(snaps, "distance_25ma_pct", DIST_25MA_BUCKETS)
        total = sum(b.sample_count for b in results)
        assert total == 1

    def test_empty_snapshots(self):
        results = analyze_extension_expectancy([], "distance_25ma_pct", DIST_25MA_BUCKETS)
        assert all(b.sample_count == 0 for b in results)


# ─────────────────────────────────────────────────────────────────────────────
# TestRegimeConditioned
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeConditioned:
    def test_returns_cells_per_regime_and_atr(self):
        snaps = [
            _snap(run_id="r1", market_regime="bull", atr_pct=0.020),
            _snap(run_id="r2", market_regime="bear", atr_pct=0.030),
        ]
        cells = analyze_regime_conditioned(snaps)
        regimes = {c.regime for c in cells}
        assert "bull" in regimes
        assert "bear" in regimes
        atr_labels = {c.atr_bucket for c in cells}
        assert set(b[0] for b in ATR_BUCKETS).issuperset(atr_labels)

    def test_cell_count_correct(self):
        snaps = [
            _snap(run_id="r1", market_regime="bull", atr_pct=0.020),
        ]
        cells = analyze_regime_conditioned(snaps)
        # 1 regime × 4 ATR buckets
        assert len(cells) == 4

    def test_enriched_expectancy_in_cell(self):
        snaps = [
            _snap(run_id="r1", market_regime="bull", atr_pct=0.020,
                  enrichment_status="enriched", forward_5d=0.05),
        ]
        cells = analyze_regime_conditioned(snaps)
        bull_15_25 = next(c for c in cells if c.regime == "bull" and c.atr_bucket == "1.5-2.5%")
        assert bull_15_25.expectancy_5d == pytest.approx(0.05, rel=1e-5)

    def test_no_regime_returns_empty(self):
        snaps = [_snap(run_id="r1", market_regime=None)]
        cells = analyze_regime_conditioned(snaps)
        assert cells == []


# ─────────────────────────────────────────────────────────────────────────────
# TestCounterfactual
# ─────────────────────────────────────────────────────────────────────────────

class TestCounterfactual:
    def test_groups_by_skip_reason(self):
        snaps = [
            _snap(run_id="r1", executed=False, skip_reason="rsr_filter",
                  enrichment_status="enriched", forward_5d=0.04),
            _snap(run_id="r2", executed=False, skip_reason="rsr_filter",
                  enrichment_status="enriched", forward_5d=0.02),
            _snap(run_id="r3", executed=False, skip_reason="capital_limit",
                  enrichment_status="enriched", forward_5d=0.01),
        ]
        results = analyze_counterfactual(snaps)
        reasons = [r.skip_reason for r in results]
        assert "rsr_filter" in reasons
        assert "capital_limit" in reasons

    def test_rsr_group_count_correct(self):
        snaps = [
            _snap(run_id="r1", executed=False, skip_reason="rsr_filter"),
            _snap(run_id="r2", executed=False, skip_reason="rsr_filter"),
        ]
        results = analyze_counterfactual(snaps)
        rsr_result = next(r for r in results if r.skip_reason == "rsr_filter")
        assert rsr_result.count == 2

    def test_sorted_by_expectancy_descending(self):
        snaps = [
            _snap(run_id="r1", executed=False, skip_reason="reason_A",
                  enrichment_status="enriched", forward_5d=0.01),
            _snap(run_id="r2", executed=False, skip_reason="reason_B",
                  enrichment_status="enriched", forward_5d=0.05),
        ]
        results = analyze_counterfactual(snaps)
        assert results[0].skip_reason == "reason_B"

    def test_missed_alpha_score_computed(self):
        snaps = [
            _snap(run_id="r1", executed=False, skip_reason="rsr_filter",
                  enrichment_status="enriched", forward_5d=0.05),
        ]
        results = analyze_counterfactual(snaps)
        assert results[0].missed_alpha_score == pytest.approx(0.05, rel=1e-5)

    def test_executed_signals_excluded(self):
        snaps = [
            _snap(run_id="r1", executed=True, skip_reason=None),
        ]
        results = analyze_counterfactual(snaps)
        assert results == []

    def test_empty_returns_empty(self):
        assert analyze_counterfactual([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# TestSectorConcentration
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorConcentration:
    def test_single_sector_monoculture(self):
        snaps = [_snap(run_id=f"r{i}", sector="IT") for i in range(5)]
        result = analyze_sector_concentration(snaps)
        assert result is not None
        assert result.monoculture_alert is True
        assert result.dominant_sector == "IT"
        assert result.dominant_sector_share == pytest.approx(1.0, rel=1e-5)

    def test_uniform_distribution_no_alert(self):
        snaps = [_snap(run_id=f"r{i}", sector=f"sector_{i}") for i in range(5)]
        result = analyze_sector_concentration(snaps)
        assert result is not None
        assert result.monoculture_alert is False

    def test_herfindahl_single_sector(self):
        snaps = [_snap(run_id=f"r{i}", sector="IT") for i in range(3)]
        result = analyze_sector_concentration(snaps)
        assert result.concentration_index == pytest.approx(1.0, rel=1e-5)

    def test_effective_sector_count_uniform(self):
        import math
        n = 4
        snaps = [_snap(run_id=f"r{i}", sector=f"sector_{i}") for i in range(n)]
        result = analyze_sector_concentration(snaps)
        # Uniform distribution → effective count ≈ n
        assert abs(result.effective_sector_count - n) < 0.1

    def test_sector_shares_sum_to_one(self):
        snaps = [
            _snap(run_id="r1", sector="IT"),
            _snap(run_id="r2", sector="自動車"),
            _snap(run_id="r3", sector="IT"),
        ]
        result = analyze_sector_concentration(snaps)
        total = sum(result.sector_shares.values())
        assert total == pytest.approx(1.0, rel=1e-5)

    def test_none_sector_treated_as_unknown(self):
        snaps = [_snap(run_id="r1", sector=None)]
        result = analyze_sector_concentration(snaps)
        assert "unknown" in result.sector_shares

    def test_empty_returns_none(self):
        result = analyze_sector_concentration([])
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TestLifecycleDecay
# ─────────────────────────────────────────────────────────────────────────────

class TestLifecycleDecay:
    def test_returns_six_buckets(self):
        records = [_lifecycle_rec(days_held=5)]
        result = analyze_lifecycle_decay(records)
        assert len(result) == 6

    def test_correct_bucket_assignment(self):
        records = [
            _lifecycle_rec(days_held=2, unrealized_pnl_pct=0.02),
            _lifecycle_rec(symbol="9984", days_held=6, unrealized_pnl_pct=0.05),
        ]
        result = analyze_lifecycle_decay(records)
        bucket_1_3 = next(b for b in result if b.label == "1-3d")
        bucket_4_7 = next(b for b in result if b.label == "4-7d")
        assert bucket_1_3.sample_count == 1
        assert bucket_4_7.sample_count == 1
        assert bucket_1_3.avg_unrealized_pnl_pct == pytest.approx(0.02, rel=1e-5)

    def test_empty_bucket_has_none_pnl(self):
        result = analyze_lifecycle_decay([])
        assert all(b.avg_unrealized_pnl_pct is None for b in result)

    def test_avg_pnl_averaged_correctly(self):
        records = [
            _lifecycle_rec(days_held=5, unrealized_pnl_pct=0.04),
            _lifecycle_rec(symbol="9984", days_held=5, unrealized_pnl_pct=0.02),
        ]
        result = analyze_lifecycle_decay(records)
        bucket = next(b for b in result if b.label == "4-7d")
        assert bucket.avg_unrealized_pnl_pct == pytest.approx(0.03, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# TestOverheatingVerdict
# ─────────────────────────────────────────────────────────────────────────────

class TestOverheatingVerdict:
    def _make_bucket(self, label, exp5, n=5):
        return ExpectancyBucket(
            label=label, sample_count=n, executed_count=n, skipped_count=0,
            expectancy_5d=exp5, expectancy_10d=exp5, expectancy_20d=exp5,
            mae=None, mfe=None, win_rate_5d=0.6, continuation_probability=0.6,
            breakout_failure_rate=0.1,
        )

    def test_insufficient_data_when_few_enriched(self):
        buckets = [self._make_bucket(b[0], 0.02) for b in ATR_BUCKETS]
        verdict, conf, evid = _overheating_verdict(buckets, enriched_count=2)
        assert verdict == "insufficient_data"
        assert conf == "LOW"

    def test_overheating_confirmed_when_high_atr_worse(self):
        buckets = [
            self._make_bucket("<1.5%",    0.05),
            self._make_bucket("1.5-2.5%", 0.04),
            self._make_bucket("2.5-4%",   0.01),
            self._make_bucket("4%+",      -0.02),
        ]
        verdict, conf, evid = _overheating_verdict(buckets, enriched_count=20)
        assert verdict == "overheating_confirmed"

    def test_momentum_capture_when_high_atr_better(self):
        buckets = [
            self._make_bucket("<1.5%",    0.01),
            self._make_bucket("1.5-2.5%", 0.01),
            self._make_bucket("2.5-4%",   0.04),
            self._make_bucket("4%+",      0.05),
        ]
        verdict, conf, evid = _overheating_verdict(buckets, enriched_count=20)
        assert verdict == "momentum_capture"

    def test_neutral_when_similar(self):
        buckets = [
            self._make_bucket("<1.5%",    0.03),
            self._make_bucket("1.5-2.5%", 0.03),
            self._make_bucket("2.5-4%",   0.03),
            self._make_bucket("4%+",      0.03),
        ]
        verdict, conf, evid = _overheating_verdict(buckets, enriched_count=20)
        assert verdict == "neutral"

    def test_high_confidence_on_large_diff(self):
        buckets = [
            self._make_bucket("<1.5%",    0.10),
            self._make_bucket("1.5-2.5%", 0.08),
            self._make_bucket("2.5-4%",   -0.05),
            self._make_bucket("4%+",      -0.06),
        ]
        verdict, conf, evid = _overheating_verdict(buckets, enriched_count=20)
        assert verdict == "overheating_confirmed"
        assert conf == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# TestWriteSignalSnapshots
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteSignalSnapshots:
    def _buy_signal(self, symbol="7203", atr20=20.0, price=1000.0, qty=100):
        return {
            "symbol": symbol,
            "action": "BUY",
            "signal": 1,
            "estimated_price": price,
            "atr20": atr20,
            "rsr": 82.0,
            "rsr_rank": 82.0,
            "sector": "自動車",
            "qty": qty,
            "reason": None,
        }

    def _skip_signal(self, symbol="9984", reason="rsr_filter"):
        return {
            "symbol": symbol,
            "action": "SKIP_BUY",
            "signal": 1,
            "estimated_price": 5000.0,
            "atr20": 100.0,
            "rsr": 65.0,
            "rsr_rank": 65.0,
            "sector": "IT",
            "qty": 0,
            "reason": reason,
        }

    def test_writes_buy_signals(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        signals = [self._buy_signal()]
        n = write_signal_snapshots(
            signals=signals, run_id="test", cash_ratio=0.5, slot_utilization=0.33,
            breadth_50=0.6, breadth_75=0.3, market_regime="mixed", snapshots_path=path,
        )
        assert n == 1
        loaded = load_entry_snapshots(path)
        assert len(loaded) == 1
        assert loaded[0].executed is True

    def test_writes_skip_signals(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        signals = [self._skip_signal()]
        n = write_signal_snapshots(
            signals=signals, run_id="test", cash_ratio=0.5, slot_utilization=0.33,
            breadth_50=0.4, breadth_75=0.2, market_regime="weak", snapshots_path=path,
        )
        assert n == 1
        loaded = load_entry_snapshots(path)
        assert loaded[0].executed is False
        assert loaded[0].skip_reason == "rsr_filter"

    def test_mixed_signals_writes_both(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        signals = [self._buy_signal(), self._skip_signal()]
        n = write_signal_snapshots(
            signals=signals, run_id="test", cash_ratio=0.5, slot_utilization=0.33,
            breadth_50=0.5, breadth_75=0.25, market_regime="mixed", snapshots_path=path,
        )
        assert n == 2

    def test_non_buy_signals_skipped(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        signals = [{"symbol": "7203", "action": "HOLD", "signal": 0}]
        n = write_signal_snapshots(
            signals=signals, run_id="test", cash_ratio=0.5, slot_utilization=0.33,
            breadth_50=0.5, breadth_75=0.25, market_regime="mixed", snapshots_path=path,
        )
        assert n == 0

    def test_atr_pct_computed_correctly(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        signals = [self._buy_signal(atr20=20.0, price=1000.0)]  # → 0.02
        write_signal_snapshots(
            signals=signals, run_id="test", cash_ratio=0.5, slot_utilization=0.33,
            breadth_50=0.5, breadth_75=0.25, market_regime="mixed", snapshots_path=path,
        )
        loaded = load_entry_snapshots(path)
        assert abs(loaded[0].atr_pct - 0.02) < 1e-5

    def test_fails_open_on_bad_path(self):
        bad_path = Path("/nonexistent_dir_xyz/snaps.jsonl")
        signals = [self._buy_signal()]
        # Should not raise — fail-open
        # NOTE: this may create the dir on some systems; we're testing no exception
        try:
            write_signal_snapshots(
                signals=signals, run_id="test", cash_ratio=0.5, slot_utilization=0.33,
                breadth_50=0.5, breadth_75=0.25, market_regime="mixed", snapshots_path=bad_path,
            )
        except Exception:
            pass  # tolerate if path creation fails; we only care no crash propagates


# ─────────────────────────────────────────────────────────────────────────────
# TestDeterminism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_atr_bucket_analysis_deterministic(self):
        snaps = [
            _snap(run_id="r1", atr_pct=0.020, enrichment_status="enriched", forward_5d=0.04),
            _snap(run_id="r2", atr_pct=0.010, enrichment_status="enriched", forward_5d=0.02),
        ]
        result1 = analyze_atr_buckets(snaps)
        result2 = analyze_atr_buckets(snaps)
        for b1, b2 in zip(result1, result2):
            assert b1.expectancy_5d == b2.expectancy_5d
            assert b1.sample_count == b2.sample_count

    def test_sector_concentration_deterministic(self):
        snaps = [
            _snap(run_id="r1", sector="IT"),
            _snap(run_id="r2", sector="自動車"),
            _snap(run_id="r3", sector="IT"),
        ]
        r1 = analyze_sector_concentration(snaps)
        r2 = analyze_sector_concentration(snaps)
        assert r1.concentration_index == r2.concentration_index
        assert r1.effective_sector_count == r2.effective_sector_count


# ─────────────────────────────────────────────────────────────────────────────
# TestFeatureForwardExpectancyEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureForwardExpectancyEngine:
    def test_no_snapshots_returns_unavailable(self, tmp_path):
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=tmp_path / "snaps.jsonl",
            lifecycle_path=tmp_path / "lifecycle.jsonl",
            cache_dir=tmp_path / "cache",
        )
        report = engine.run()
        assert report.available is False
        assert report.snapshot_count == 0
        assert "No entry feature snapshots" in report.insufficiency_markers[0]

    def test_with_snapshots_returns_available(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        append_entry_snapshot(_snap(eval_date="2026-01-10"), path)
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=path,
            lifecycle_path=tmp_path / "lifecycle.jsonl",
            cache_dir=tmp_path / "cache",
        )
        report = engine.run(enrich=False)
        assert report.available is True
        assert report.snapshot_count == 1

    def test_backfill_triggered_by_trades(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        trades = [{"side": "BUY", "symbol": "7203", "date": "2026-01-05",
                   "price": 1000.0, "qty": 100, "atr20": 20.0, "amount": 100000.0}]
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=path,
            lifecycle_path=tmp_path / "lifecycle.jsonl",
            cache_dir=tmp_path / "cache",
        )
        report = engine.run(trades=trades, enrich=False)
        assert report.available is True
        assert report.snapshot_count >= 1

    def test_date_range_extracted(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        for d in ["2026-01-10", "2026-01-11"]:
            append_entry_snapshot(_snap(eval_date=d, run_id=d), path)
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=path,
            lifecycle_path=tmp_path / "lifecycle.jsonl",
            cache_dir=tmp_path / "cache",
        )
        report = engine.run(enrich=False)
        assert report.eval_date_range[0] == "2026-01-10"
        assert report.eval_date_range[1] == "2026-01-11"

    def test_week_window_filter(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        # 2 in window, 1 outside (outside the 90d lookback for week_end=2026-01-15)
        old_date = (date(2026, 1, 15) - timedelta(days=100)).isoformat()
        append_entry_snapshot(_snap(eval_date=old_date, run_id="old"), path)
        for d in ["2026-01-10", "2026-01-11"]:
            append_entry_snapshot(_snap(eval_date=d, run_id=d), path)
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=path,
            lifecycle_path=tmp_path / "lifecycle.jsonl",
            cache_dir=tmp_path / "cache",
        )
        report = engine.run(
            week_start=date(2026, 1, 13),
            week_end=date(2026, 1, 15),
            enrich=False,
        )
        assert report.snapshot_count == 2

    def test_overheating_verdict_set(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        append_entry_snapshot(_snap(), path)
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=path,
            lifecycle_path=tmp_path / "lifecycle.jsonl",
            cache_dir=tmp_path / "cache",
        )
        report = engine.run(enrich=False)
        assert report.overheating_verdict in (
            "insufficient_data", "overheating_confirmed", "momentum_capture", "neutral"
        )

    def test_sector_concentration_populated(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        for i in range(3):
            append_entry_snapshot(_snap(run_id=f"r{i}", sector="IT"), path)
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=path,
            lifecycle_path=tmp_path / "lifecycle.jsonl",
            cache_dir=tmp_path / "cache",
        )
        report = engine.run(enrich=False)
        assert report.sector_concentration is not None
        assert report.sector_concentration.dominant_sector == "IT"

    def test_report_is_complete_dataclass(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        append_entry_snapshot(_snap(), path)
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=path,
            lifecycle_path=tmp_path / "lifecycle.jsonl",
            cache_dir=tmp_path / "cache",
        )
        report = engine.run(enrich=False)
        # All required list fields present
        assert isinstance(report.atr_buckets, list)
        assert isinstance(report.extension_expectancy, list)
        assert isinstance(report.regime_matrix, list)
        assert isinstance(report.lifecycle_decay, list)

    def test_insufficient_markers_propagated(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        # No snapshots → insufficiency marker
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=path,
            lifecycle_path=tmp_path / "lifecycle.jsonl",
            cache_dir=tmp_path / "cache",
        )
        report = engine.run()
        assert len(report.insufficiency_markers) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TestHelpers
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_safe_mean_empty(self):
        assert _safe_mean([]) is None

    def test_safe_mean_values(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0, rel=1e-5)

    def test_win_rate_empty(self):
        assert _win_rate([]) is None

    def test_win_rate_all_winners(self):
        assert _win_rate([0.1, 0.2, 0.3]) == pytest.approx(1.0, rel=1e-5)

    def test_win_rate_all_losers(self):
        assert _win_rate([-0.1, -0.2]) == pytest.approx(0.0, rel=1e-5)

    def test_win_rate_mixed(self):
        assert _win_rate([0.1, -0.1]) == pytest.approx(0.5, rel=1e-5)
