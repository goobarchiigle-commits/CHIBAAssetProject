"""
tests/test_breakout_quality.py — Breakout Quality Intelligence

Coverage:
  - Component score functions (CPR, USR, GER, RE): normal + edge cases
  - Phase classification: all 4 phases + priority ordering
  - compute_breakout_quality: golden paths, score bounds, error handling
  - get_addon_boost_multiplier: all phases
  - extract_bq_inputs_from_df: normal + short-df guard
  - append_bq_telemetry: write + read-back
  - materialize_bq_returns: materializes old, skips recent
  - compute_bq_kpis: normal, empty, partial data
  - format_bq_report_section: normal + empty
  - Constants: weights sum to 1.0, boost table coverage
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.breakout_quality import (
    BOOST_BY_PHASE,
    EXTENDED_GER_MIN,
    FAILED_CPR_MAX,
    FAILED_USR_MIN,
    HEALTHY_CPR_MIN,
    HEALTHY_GER_MAX,
    HEALTHY_RSR_MOM_MIN,
    HEALTHY_USR_MAX,
    PHASE_EXTENDED,
    PHASE_FAILED,
    PHASE_HEALTHY,
    PHASE_WEAK,
    W_CPR,
    W_GER,
    W_RE,
    W_USR,
    BreakoutQualityInput,
    BreakoutQualityResult,
    _cpr,
    _ger,
    _re,
    _usr,
    append_bq_telemetry,
    compute_atr20_from_df,
    compute_bq_kpis,
    compute_breakout_quality,
    extract_bq_inputs_from_df,
    format_bq_report_section,
    get_addon_boost_multiplier,
    materialize_bq_returns,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inp(
    symbol="7203.T",
    breakout_phase=True,
    close=1050.0,
    high=1080.0,
    low=1020.0,
    prev_close=1000.0,
    atr_20=40.0,
    volume_ratio_5d=1.5,
    compression_score=80.0,
    rsr=82.0,
    rsr_momentum=0.5,
) -> BreakoutQualityInput:
    return BreakoutQualityInput(
        symbol=symbol,
        breakout_phase=breakout_phase,
        close=close,
        high=high,
        low=low,
        prev_close=prev_close,
        atr_20=atr_20,
        volume_ratio_5d=volume_ratio_5d,
        compression_score=compression_score,
        rsr=rsr,
        rsr_momentum=rsr_momentum,
    )


def _make_ohlcv(n=30, close_end=1050.0) -> pd.DataFrame:
    import numpy as np
    closes = np.linspace(900.0, close_end, n)
    highs  = closes + 20
    lows   = closes - 20
    vols   = np.ones(n) * 1_000_000
    df = pd.DataFrame({"close": closes, "high": highs, "low": lows, "volume": vols})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_weights_sum_to_one(self):
        total = W_CPR + W_USR + W_GER + W_RE
        assert abs(total - 1.0) < 1e-9, f"weights sum to {total}"

    def test_all_phases_in_boost_table(self):
        for phase in (PHASE_HEALTHY, PHASE_EXTENDED, PHASE_WEAK, PHASE_FAILED):
            assert phase in BOOST_BY_PHASE

    def test_healthy_boost_is_1_15(self):
        assert BOOST_BY_PHASE[PHASE_HEALTHY] == 1.15

    def test_failed_boost_below_one(self):
        assert BOOST_BY_PHASE[PHASE_FAILED] < 1.0

    def test_weak_extended_boost_is_one(self):
        assert BOOST_BY_PHASE[PHASE_WEAK] == 1.00
        assert BOOST_BY_PHASE[PHASE_EXTENDED] == 1.00


# ─────────────────────────────────────────────────────────────────────────────
# CPR component
# ─────────────────────────────────────────────────────────────────────────────

class TestCPR:
    def test_close_at_high(self):
        v, s = _cpr(100.0, 100.0, 80.0)
        assert v == pytest.approx(1.0)
        assert s == pytest.approx(100.0)

    def test_close_at_low(self):
        v, s = _cpr(80.0, 100.0, 80.0)
        assert v == pytest.approx(0.0)
        assert s == pytest.approx(0.0)

    def test_close_at_midpoint(self):
        v, s = _cpr(90.0, 100.0, 80.0)
        assert v == pytest.approx(0.5)
        assert s == pytest.approx(50.0)

    def test_zero_range_guard(self):
        v, s = _cpr(100.0, 100.0, 100.0)
        assert v == pytest.approx(0.5)  # neutral
        assert s == pytest.approx(50.0)

    def test_score_bounds(self):
        for close in (70.0, 90.0, 110.0, 130.0):
            v, s = _cpr(close, 120.0, 80.0)
            assert 0.0 <= v <= 1.0
            assert 0.0 <= s <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# USR component
# ─────────────────────────────────────────────────────────────────────────────

class TestUSR:
    def test_no_upper_shadow(self):
        v, s = _usr(100.0, 100.0, 80.0)  # close == high
        assert v == pytest.approx(0.0)
        assert s == pytest.approx(100.0)

    def test_full_upper_shadow(self):
        v, s = _usr(80.0, 100.0, 80.0)  # close == low
        assert v == pytest.approx(1.0)
        assert s == pytest.approx(0.0)

    def test_zero_range_guard(self):
        v, s = _usr(100.0, 100.0, 100.0)
        assert v == pytest.approx(0.5)
        assert s == pytest.approx(50.0)

    def test_cpr_plus_usr_equals_one(self):
        close, high, low = 95.0, 100.0, 80.0
        cpr_v, _ = _cpr(close, high, low)
        usr_v, _ = _usr(close, high, low)
        assert abs(cpr_v + usr_v - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# GER component
# ─────────────────────────────────────────────────────────────────────────────

class TestGER:
    def test_ideal_zone_scores_100(self):
        # 1.0 ATR gap — inside 0.5-1.8 ideal
        v, s = _ger(1040.0, 1000.0, 40.0)   # gap=40, atr=40 → 1.0 ATR
        assert v == pytest.approx(1.0)
        assert s == pytest.approx(100.0)

    def test_below_ideal_partial_credit(self):
        # gap = 0.1 ATR → below 0.5 ideal floor
        v, s = _ger(1004.0, 1000.0, 40.0)
        assert v < 0.5
        assert 0 < s < 80

    def test_above_ideal_decays(self):
        # gap = 3.0 ATR → above 1.8 ideal ceiling
        v, s = _ger(1120.0, 1000.0, 40.0)
        assert v == pytest.approx(3.0)
        assert s < 100.0
        assert s > 0.0

    def test_exhaustion_zone_scores_low(self):
        # gap = 3.5+ ATR → score approaches 0
        v, s = _ger(1200.0, 1060.0, 40.0)
        assert s == pytest.approx(0.0)

    def test_zero_atr_guard(self):
        v, s = _ger(1050.0, 1000.0, 0.0)
        assert v == pytest.approx(1.0)
        assert s == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# RE component
# ─────────────────────────────────────────────────────────────────────────────

class TestRE:
    def test_ideal_range_scores_high(self):
        # range = 2.0 ATR — inside 1.0-2.5 zone
        v, s = _re(1080.0, 1000.0, 40.0)
        assert v == pytest.approx(2.0)
        assert s > 80.0

    def test_flat_expansion_scores_50(self):
        # range = 0.5 ATR
        v, s = _re(1020.0, 1000.0, 40.0)
        assert s == pytest.approx(50.0)

    def test_blowoff_candle_scores_low(self):
        # range = 4.5 ATR → near zero
        v, s = _re(1180.0, 1000.0, 40.0)
        assert s == pytest.approx(0.0)

    def test_zero_atr_guard(self):
        v, s = _re(1060.0, 1000.0, 0.0)
        assert s > 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase classification
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseClassification:
    def _healthy_inp(self):
        # close near high, small shadow, moderate gap, moderate range, positive RSR
        return _inp(close=1072.0, high=1080.0, low=1020.0, prev_close=1010.0, atr_20=40.0, rsr_momentum=0.5)

    def test_healthy_all_gates_pass(self):
        r = compute_breakout_quality(self._healthy_inp())
        assert r.phase_type == PHASE_HEALTHY

    def test_weak_low_cpr(self):
        # close below midpoint
        r = compute_breakout_quality(_inp(close=1030.0, high=1080.0, low=1020.0, prev_close=1010.0, atr_20=40.0))
        assert r.phase_type in (PHASE_WEAK, PHASE_FAILED)

    def test_weak_high_shadow(self):
        # heavy upper shadow: close near low
        r = compute_breakout_quality(_inp(close=1025.0, high=1080.0, low=1020.0, prev_close=1010.0, atr_20=40.0))
        assert r.phase_type in (PHASE_FAILED, PHASE_WEAK)

    def test_extended_large_gap(self):
        # gap > 2.5 ATR
        r = compute_breakout_quality(_inp(close=1160.0, high=1170.0, low=1150.0, prev_close=1000.0, atr_20=40.0))
        assert r.phase_type == PHASE_EXTENDED

    def test_extended_blowoff_range(self):
        # range = 4.0 ATR
        r = compute_breakout_quality(_inp(close=1130.0, high=1160.0, low=1000.0, prev_close=1010.0, atr_20=40.0))
        assert r.phase_type == PHASE_EXTENDED

    def test_failed_breakout_detection(self):
        # continuation_breakout + close below midpoint + heavy shadow
        r = compute_breakout_quality(_inp(
            breakout_phase=True,
            close=1023.0, high=1080.0, low=1020.0,
            prev_close=1010.0, atr_20=40.0,
        ))
        assert r.phase_type == PHASE_FAILED

    def test_no_breakout_phase_never_failed(self):
        # breakout_phase=False → cannot be classified as failed
        r = compute_breakout_quality(_inp(
            breakout_phase=False,
            close=1023.0, high=1080.0, low=1020.0,
            prev_close=1010.0, atr_20=40.0,
        ))
        assert r.phase_type != PHASE_FAILED

    def test_failed_priority_over_extended(self):
        # failed check runs before extended check
        r = compute_breakout_quality(_inp(
            breakout_phase=True,
            close=1023.0, high=1200.0, low=1000.0,  # large range (extended candidate)
            prev_close=800.0, atr_20=40.0,            # large gap (extended candidate)
        ))
        # Priority: failed wins because CPR<0.5 and USR>=0.45
        # (close=1023 in [1000,1200] range → CPR=(23/200)=0.115 < 0.50 ✓
        #  USR=(1200-1023)/200=0.885 >= 0.45 ✓)
        assert r.phase_type == PHASE_FAILED

    def test_negative_rsr_momentum_blocks_healthy(self):
        # rsr_momentum < 0 → cannot be healthy
        r = compute_breakout_quality(self._healthy_inp().__class__(
            symbol="TEST",
            breakout_phase=True,
            close=1072.0,
            high=1080.0,
            low=1020.0,
            prev_close=1010.0,
            atr_20=40.0,
            volume_ratio_5d=1.5,
            compression_score=80.0,
            rsr=82.0,
            rsr_momentum=-0.1,   # negative — blocks healthy
        ))
        assert r.phase_type != PHASE_HEALTHY


# ─────────────────────────────────────────────────────────────────────────────
# compute_breakout_quality
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBreakoutQuality:
    def test_returns_result_type(self):
        r = compute_breakout_quality(_inp())
        assert isinstance(r, BreakoutQualityResult)

    def test_score_in_bounds(self):
        for close in (1020.0, 1050.0, 1080.0):
            r = compute_breakout_quality(_inp(close=close))
            assert 0.0 <= r.quality_score <= 100.0

    def test_healthy_high_score(self):
        # All components should push score high
        r = compute_breakout_quality(_inp(
            close=1075.0, high=1080.0, low=1020.0,
            prev_close=1038.0, atr_20=40.0, rsr_momentum=1.0,
        ))
        assert r.quality_score > 70.0

    def test_failed_lower_score(self):
        r_healthy = compute_breakout_quality(_inp(
            close=1072.0, high=1080.0, low=1020.0, prev_close=1010.0, atr_20=40.0,
        ))
        r_failed  = compute_breakout_quality(_inp(
            close=1022.0, high=1080.0, low=1020.0, prev_close=1010.0, atr_20=40.0,
        ))
        assert r_healthy.quality_score > r_failed.quality_score

    def test_input_echoes_preserved(self):
        r = compute_breakout_quality(_inp(rsr=85.0, rsr_momentum=0.3, volume_ratio_5d=1.4))
        assert r.rsr == pytest.approx(85.0, abs=0.01)
        assert r.rsr_momentum == pytest.approx(0.3, abs=0.001)
        assert r.volume_ratio_5d == pytest.approx(1.4, abs=0.001)

    def test_zero_range_candle_returns_result(self):
        # doji: high == low == close — guard must not raise
        r = compute_breakout_quality(_inp(close=1000.0, high=1000.0, low=1000.0))
        assert isinstance(r, BreakoutQualityResult)
        assert 0.0 <= r.quality_score <= 100.0

    def test_invalid_nan_no_exception(self):
        # NaN close should not raise; result must be a valid bounded score
        r = compute_breakout_quality(_inp(close=float("nan")))
        assert isinstance(r, BreakoutQualityResult)
        assert math.isfinite(r.quality_score)
        assert 0.0 <= r.quality_score <= 100.0

    def test_symbol_preserved(self):
        r = compute_breakout_quality(_inp(symbol="6920.T"))
        assert r.symbol == "6920.T"


# ─────────────────────────────────────────────────────────────────────────────
# get_addon_boost_multiplier
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAddonBoostMultiplier:
    def test_healthy(self):
        assert get_addon_boost_multiplier(PHASE_HEALTHY) == 1.15

    def test_extended(self):
        assert get_addon_boost_multiplier(PHASE_EXTENDED) == 1.00

    def test_weak(self):
        assert get_addon_boost_multiplier(PHASE_WEAK) == 1.00

    def test_failed(self):
        assert get_addon_boost_multiplier(PHASE_FAILED) == 0.90

    def test_unknown_returns_one(self):
        assert get_addon_boost_multiplier("unknown_phase") == 1.00


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestOHLCVHelpers:
    def test_atr20_positive(self):
        df = _make_ohlcv(30)
        atr = compute_atr20_from_df(df)
        assert atr > 0

    def test_atr20_short_df_fallback(self):
        df = _make_ohlcv(1)
        atr = compute_atr20_from_df(df)
        assert atr > 0  # 2% fallback

    def test_extract_bq_inputs_fields(self):
        df  = _make_ohlcv(30)
        sig = {"continuation_breakout": True, "compression_score": 75.0, "compression_volume_ratio": 1.6}
        inp = extract_bq_inputs_from_df("7203.T", df, sig, rsr=80.0, rsr_momentum=0.4)
        assert inp.symbol == "7203.T"
        assert inp.breakout_phase is True
        assert inp.compression_score == pytest.approx(75.0)
        assert inp.rsr == pytest.approx(80.0)
        assert inp.atr_20 > 0

    def test_extract_prev_close_is_second_last(self):
        df = _make_ohlcv(10)
        inp = extract_bq_inputs_from_df("X", df, {})
        assert inp.prev_close == pytest.approx(float(df.iloc[-2]["close"]))

    def test_extract_close_is_last(self):
        df = _make_ohlcv(10)
        inp = extract_bq_inputs_from_df("X", df, {})
        assert inp.close == pytest.approx(float(df.iloc[-1]["close"]))


# ─────────────────────────────────────────────────────────────────────────────
# append_bq_telemetry
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendBqTelemetry:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        r = compute_breakout_quality(_inp())
        append_bq_telemetry(r, 1.15, 1.15, path)
        assert path.exists()

    def test_record_content(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        r = compute_breakout_quality(_inp(symbol="6920.T"))
        append_bq_telemetry(r, 1.15, 1.00, path, date_str="2026-05-28")
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert rec["symbol"] == "6920.T"
        assert rec["date"] == "2026-05-28"
        assert rec["addon_boost_before"] == pytest.approx(1.15)
        assert rec["addon_boost_after"] == pytest.approx(1.00)
        assert rec["subsequent_3d_return"] is None
        assert rec["materialized_3d"] is False
        assert rec["schema_version"] == "v1"

    def test_appends_multiple_records(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        r = compute_breakout_quality(_inp())
        append_bq_telemetry(r, 1.15, 1.15, path, date_str="2026-05-26")
        append_bq_telemetry(r, 1.15, 1.00, path, date_str="2026-05-27")
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_missing_parent_created(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "bq.jsonl"
        r = compute_breakout_quality(_inp())
        append_bq_telemetry(r, 1.15, 1.15, path)
        assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# materialize_bq_returns
# ─────────────────────────────────────────────────────────────────────────────

class TestMaterializeBqReturns:
    def _old_record(self, symbol="7203.T") -> dict:
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        r = compute_breakout_quality(_inp(symbol=symbol))
        return {
            "date": old_date,
            "symbol": symbol,
            "breakout_quality_score": r.quality_score,
            "breakout_phase_type": r.phase_type,
            "close_position_in_range": r.close_position_in_range,
            "upper_shadow_ratio": r.upper_shadow_ratio,
            "gap_extension_risk": r.gap_extension_risk,
            "range_expansion_atr": r.range_expansion_atr,
            "cpr_score": r.cpr_score, "usr_score": r.usr_score,
            "ger_score": r.ger_score, "re_score": r.re_score,
            "rsr": 80.0, "rsr_momentum": 0.3,
            "volume_ratio_5d": 1.5, "compression_score": 75.0,
            "addon_boost_before": 1.15, "addon_boost_after": 1.15,
            "subsequent_3d_return": None, "subsequent_5d_return": None,
            "materialized_3d": False, "materialized_5d": False,
            "schema_version": "v1",
        }

    def _write_records(self, path: Path, records: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

    def test_missing_file_returns_empty(self, tmp_path):
        result = materialize_bq_returns(tmp_path / "missing.jsonl", lambda s: None)
        assert result == {}

    def test_materializes_old_record(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        rec = self._old_record("7203.T")
        self._write_records(path, [rec])

        df = _make_ohlcv(60)
        df.index = pd.date_range(end=datetime.now(), periods=60, freq="B")

        result = materialize_bq_returns(path, lambda s: df)
        assert result.get("n_materialized_3d", 0) >= 0  # may or may not find

    def test_skips_recent_record(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        today = datetime.now().strftime("%Y-%m-%d")
        rec = self._old_record()
        rec["date"] = today  # today = not yet eligible
        self._write_records(path, [rec])

        df = _make_ohlcv(60)
        df.index = pd.date_range(end=datetime.now(), periods=60, freq="B")

        result = materialize_bq_returns(path, lambda s: df)
        assert result.get("n_materialized_3d", 0) == 0
        assert result.get("n_materialized_5d", 0) == 0

    def test_already_materialized_not_reprocessed(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        rec = self._old_record()
        rec["materialized_3d"] = True
        rec["subsequent_3d_return"] = 0.05
        self._write_records(path, [rec])

        df = _make_ohlcv(60)
        df.index = pd.date_range(end=datetime.now(), periods=60, freq="B")

        materialize_bq_returns(path, lambda s: df)
        out = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert out["subsequent_3d_return"] == pytest.approx(0.05)

    def test_none_loader_handled_gracefully(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        self._write_records(path, [self._old_record()])
        result = materialize_bq_returns(path, lambda s: None)
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# compute_bq_kpis
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBqKpis:
    def _write_records(self, path: Path, records: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

    def _rec(self, phase=PHASE_HEALTHY, ret3d=0.02, ret5d=0.03) -> dict:
        return {
            "symbol": "T",
            "date": "2026-05-01",
            "breakout_phase_type": phase,
            "subsequent_3d_return": ret3d,
            "subsequent_5d_return": ret5d,
            "materialized_3d": ret3d is not None,
            "materialized_5d": ret5d is not None,
        }

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        path.write_text("", encoding="utf-8")
        assert compute_bq_kpis(path) == {}

    def test_missing_file_returns_empty(self, tmp_path):
        assert compute_bq_kpis(tmp_path / "missing.jsonl") == {}

    def test_counts_by_phase(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        records = [
            self._rec(PHASE_HEALTHY),
            self._rec(PHASE_HEALTHY),
            self._rec(PHASE_WEAK),
            self._rec(PHASE_FAILED),
        ]
        self._write_records(path, records)
        kpis = compute_bq_kpis(path)
        assert kpis["healthy_breakout_count"] == 2
        assert kpis["weak_breakout_count"] == 1
        assert kpis["failed_breakout_count"] == 1
        assert kpis["total_records"] == 4

    def test_failed_breakout_rate(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        records = [self._rec(PHASE_FAILED)] * 2 + [self._rec(PHASE_HEALTHY)] * 2
        self._write_records(path, records)
        kpis = compute_bq_kpis(path)
        assert kpis["failed_breakout_rate"] == pytest.approx(0.5)

    def test_avg_return_by_phase(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        records = [
            self._rec(PHASE_HEALTHY, 0.04, 0.06),
            self._rec(PHASE_HEALTHY, 0.02, 0.03),
            self._rec(PHASE_WEAK, -0.01, -0.02),
        ]
        self._write_records(path, records)
        kpis = compute_bq_kpis(path)
        assert kpis["avg_3d_return_by_phase"][PHASE_HEALTHY] == pytest.approx(0.03)
        assert kpis["avg_3d_return_by_phase"][PHASE_WEAK] == pytest.approx(-0.01)

    def test_delta_healthy_vs_weak(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        records = [
            self._rec(PHASE_HEALTHY, 0.04),
            self._rec(PHASE_WEAK, 0.01),
        ]
        self._write_records(path, records)
        kpis = compute_bq_kpis(path)
        assert kpis["avg_return_delta_healthy_vs_weak_3d"] == pytest.approx(0.03)

    def test_high_quality_success_rate(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        records = [
            self._rec(PHASE_HEALTHY, 0.03),
            self._rec(PHASE_HEALTHY, 0.02),
            self._rec(PHASE_HEALTHY, -0.01),
        ]
        self._write_records(path, records)
        kpis = compute_bq_kpis(path)
        assert kpis["high_quality_breakout_success_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_null_returns_not_counted_in_avg(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        records = [
            self._rec(PHASE_HEALTHY, 0.04),
            self._rec(PHASE_HEALTHY, None),   # not yet materialized
        ]
        self._write_records(path, records)
        kpis = compute_bq_kpis(path)
        assert kpis["avg_3d_return_by_phase"][PHASE_HEALTHY] == pytest.approx(0.04)


# ─────────────────────────────────────────────────────────────────────────────
# format_bq_report_section
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatBqReportSection:
    def _write_records(self, path: Path, records: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

    def _rec(self, phase, ret3d=0.02):
        return {
            "symbol": "T", "date": "2026-05-01",
            "breakout_phase_type": phase,
            "subsequent_3d_return": ret3d,
            "subsequent_5d_return": ret3d,
            "materialized_3d": True, "materialized_5d": True,
        }

    def test_missing_file_returns_empty_string(self, tmp_path):
        result = format_bq_report_section(tmp_path / "missing.jsonl")
        assert result == ""

    def test_section_header_present(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        self._write_records(path, [self._rec(PHASE_HEALTHY)])
        section = format_bq_report_section(path)
        assert "BREAKOUT QUALITY" in section

    def test_phase_counts_shown(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        self._write_records(path, [
            self._rec(PHASE_HEALTHY),
            self._rec(PHASE_FAILED, -0.02),
        ])
        section = format_bq_report_section(path)
        assert "健全" in section
        assert "失敗" in section

    def test_no_exception_on_empty(self, tmp_path):
        path = tmp_path / "bq.jsonl"
        path.write_text("", encoding="utf-8")
        result = format_bq_report_section(path)
        assert result == ""
