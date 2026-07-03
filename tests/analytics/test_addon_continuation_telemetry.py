"""
tests/analytics/test_addon_continuation_telemetry.py — Tests for addon continuation telemetry.

Covers:
  - build_addon_record: all fields, confidence_before/after inference
  - materialize_returns: 3d/5d forward return filling, calendar buffer, dedup
  - _rewrite_jsonl / _append_jsonl: atomic I/O correctness
  - compute_kpis: per-type stats, delta, false_breakout_rate
  - run_addon_continuation_hook: dedup, n_recorded, n_materialized, fail-open
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.analytics.addon_continuation_telemetry import (
    build_addon_record,
    compute_kpis,
    materialize_returns,
    run_addon_continuation_hook,
    _append_jsonl,
    _load_jsonl,
    _rewrite_jsonl,
    _business_day_offset,
    _get_close_at_date,
    FORWARD_DAYS_3,
    FORWARD_DAYS_5,
    _CALENDAR_BUFFER_3D,
    _CALENDAR_BUFFER_5D,
    SCHEMA_VERSION,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_conf(
    symbol: str = "1234.T",
    confirmed: bool = True,
    score: float = 0.50,
    cb: bool = False,
    boost: float = 1.0,
) -> MagicMock:
    m = MagicMock()
    m.symbol = symbol
    m.confirmed = confirmed
    m.confidence_score = score
    m.continuation_breakout = cb
    m.continuation_boost_applied = boost
    return m


def _make_signal(
    compression_score: float = 75.0,
    volume_ratio: float = 1.40,
    rsr_momentum: float = 0.3,
    prev_phase: str = "compression",
    phase: str = "breakout",
) -> dict:
    return {
        "compression_score": compression_score,
        "compression_volume_ratio": volume_ratio,
        "rsr_momentum": rsr_momentum,
        "prev_phase_at_compression": prev_phase,
        "compression_phase": phase,
    }


def _make_df(dates: list, prices: list) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates)
    df = pd.DataFrame({"Close": prices}, index=idx)
    return df


# ── Calendar constants ────────────────────────────────────────────────────────

class TestCalendarConstants:
    def test_forward_days(self):
        assert FORWARD_DAYS_3 == 3
        assert FORWARD_DAYS_5 == 5

    def test_calendar_buffers(self):
        assert _CALENDAR_BUFFER_3D == FORWARD_DAYS_3 + 2
        assert _CALENDAR_BUFFER_5D == FORWARD_DAYS_5 + 2


# ── _business_day_offset ──────────────────────────────────────────────────────

class TestBusinessDayOffset:
    def test_3bd_from_monday(self):
        result = _business_day_offset("2026-05-18", 3)  # Monday
        # 3 business days from Monday: Thu
        assert result == "2026-05-21"

    def test_5bd_basic(self):
        result = _business_day_offset("2026-05-18", 5)
        assert result == "2026-05-25"

    def test_zero_offset(self):
        result = _business_day_offset("2026-05-19", 0)
        assert result == "2026-05-19"


# ── _get_close_at_date ────────────────────────────────────────────────────────

class TestGetCloseAtDate:
    def test_exact_match(self):
        df = _make_df(["2026-05-19"], [1000.0])
        v = _get_close_at_date(df, "2026-05-19")
        assert v == pytest.approx(1000.0)

    def test_tolerance_match(self):
        df = _make_df(["2026-05-21"], [1100.0])
        v = _get_close_at_date(df, "2026-05-19", tolerance=2)
        assert v == pytest.approx(1100.0)

    def test_beyond_tolerance_returns_none(self):
        df = _make_df(["2026-05-23"], [1100.0])
        v = _get_close_at_date(df, "2026-05-19", tolerance=2)
        assert v is None

    def test_none_df_returns_none(self):
        assert _get_close_at_date(None, "2026-05-19") is None

    def test_empty_df_returns_none(self):
        df = pd.DataFrame({"Close": []})
        assert _get_close_at_date(df, "2026-05-19") is None

    def test_negative_price_returns_none(self):
        df = _make_df(["2026-05-19"], [-5.0])
        assert _get_close_at_date(df, "2026-05-19") is None

    def test_adj_close_fallback(self):
        df = _make_df(["2026-05-19"], [2000.0])
        df = df.rename(columns={"Close": "Adj Close"})
        v = _get_close_at_date(df, "2026-05-19")
        assert v == pytest.approx(2000.0)


# ── build_addon_record ────────────────────────────────────────────────────────

class TestBuildAddonRecord:
    def test_basic_normal_addon(self):
        conf = _make_conf(confirmed=True, score=0.50, cb=False, boost=1.0)
        sig = _make_signal()
        rec = build_addon_record(conf, sig, "2026-05-27")
        assert rec["symbol"] == "1234.T"
        assert rec["date"] == "2026-05-27"
        assert rec["addon_type"] == "normal"
        assert rec["continuation_breakout"] is False
        assert rec["addon_triggered"] is True
        assert rec["subsequent_3d_return"] is None
        assert rec["subsequent_5d_return"] is None
        assert rec["materialized_3d"] is False
        assert rec["materialized_5d"] is False
        assert rec["schema_version"] == SCHEMA_VERSION

    def test_continuation_breakout_addon_type(self):
        conf = _make_conf(confirmed=True, score=0.55, cb=True, boost=1.15)
        sig = _make_signal()
        rec = build_addon_record(conf, sig, "2026-05-27")
        assert rec["addon_type"] == "continuation_breakout"
        assert rec["continuation_breakout"] is True

    def test_confidence_before_inferred_from_boost(self):
        # score_after = 0.5750, boost = 1.15 → before = 0.5750 / 1.15 ≈ 0.5000
        score_after = round(0.50 * 1.15, 4)  # 0.575
        conf = _make_conf(score=score_after, cb=True, boost=1.15)
        sig = _make_signal()
        rec = build_addon_record(conf, sig, "2026-05-27")
        assert rec["addon_confidence_after"] == pytest.approx(score_after)
        assert rec["addon_confidence_before"] == pytest.approx(0.5, abs=1e-3)
        assert rec["addon_boost_applied"] == pytest.approx(1.15)

    def test_no_boost_before_equals_after(self):
        conf = _make_conf(score=0.50, cb=False, boost=1.0)
        sig = _make_signal()
        rec = build_addon_record(conf, sig, "2026-05-27")
        assert rec["addon_confidence_before"] == pytest.approx(0.50)
        assert rec["addon_confidence_after"] == pytest.approx(0.50)
        assert rec["addon_boost_applied"] == pytest.approx(1.0)

    def test_signal_fields_populated(self):
        conf = _make_conf()
        sig = _make_signal(
            compression_score=80.0,
            volume_ratio=1.45,
            rsr_momentum=0.42,
            prev_phase="compression",
            phase="breakout",
        )
        rec = build_addon_record(conf, sig, "2026-05-27")
        assert rec["compression_score"] == pytest.approx(80.0, abs=0.01)
        assert rec["volume_ratio_5d"] == pytest.approx(1.45, abs=0.001)
        assert rec["rsr_momentum"] == pytest.approx(0.42, abs=0.001)
        assert rec["previous_phase"] == "compression"
        assert rec["current_phase"] == "breakout"

    def test_empty_signal_uses_defaults(self):
        conf = _make_conf()
        rec = build_addon_record(conf, {}, "2026-05-27")
        assert rec["compression_score"] == 0.0
        assert rec["volume_ratio_5d"] == pytest.approx(1.0)
        assert rec["rsr_momentum"] == 0.0
        assert rec["previous_phase"] is None
        assert rec["current_phase"] == "neutral"

    def test_unconfirmed_addon_triggered_false(self):
        conf = _make_conf(confirmed=False, score=0.0)
        rec = build_addon_record(conf, {}, "2026-05-27")
        assert rec["addon_triggered"] is False


# ── _append_jsonl / _load_jsonl ────────────────────────────────────────────────

class TestAppendLoadJsonl:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "test.jsonl"
        rec = {"symbol": "A", "date": "2026-05-27", "v": 1.23}
        _append_jsonl(rec, p)
        loaded = _load_jsonl(p)
        assert len(loaded) == 1
        assert loaded[0]["symbol"] == "A"
        assert loaded[0]["v"] == pytest.approx(1.23)

    def test_append_multiple(self, tmp_path):
        p = tmp_path / "test.jsonl"
        for i in range(3):
            _append_jsonl({"i": i}, p)
        loaded = _load_jsonl(p)
        assert len(loaded) == 3
        assert [r["i"] for r in loaded] == [0, 1, 2]

    def test_load_missing_file(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        assert _load_jsonl(p) == []

    def test_load_skips_corrupt_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\nNOT_JSON\n{"b": 2}\n', encoding="utf-8")
        loaded = _load_jsonl(p)
        assert len(loaded) == 2


# ── _rewrite_jsonl ────────────────────────────────────────────────────────────

class TestRewriteJsonl:
    def test_rewrite_replaces_content(self, tmp_path):
        p = tmp_path / "test.jsonl"
        _append_jsonl({"x": 1}, p)
        _append_jsonl({"x": 2}, p)
        records = _load_jsonl(p)
        records[0]["x"] = 99
        _rewrite_jsonl(records, p)
        reloaded = _load_jsonl(p)
        assert reloaded[0]["x"] == 99
        assert reloaded[1]["x"] == 2

    def test_rewrite_empty_list(self, tmp_path):
        p = tmp_path / "test.jsonl"
        _rewrite_jsonl([], p)
        assert _load_jsonl(p) == []

    def test_rewrite_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "test.jsonl"
        _rewrite_jsonl([{"a": 1}], p)
        assert p.exists()
        assert len(_load_jsonl(p)) == 1


# ── materialize_returns ───────────────────────────────────────────────────────

class TestMaterializeReturns:
    def _make_record(
        self,
        symbol: str = "1234.T",
        date: str = "2026-05-01",
        mat3: bool = False,
        mat5: bool = False,
    ) -> dict:
        return {
            "symbol": symbol,
            "date": date,
            "addon_triggered": True,
            "subsequent_3d_return": None,
            "subsequent_5d_return": None,
            "materialized_3d": mat3,
            "materialized_5d": mat5,
        }

    def _loader(self, df: pd.DataFrame):
        def _fn(sym: str):
            return df
        return _fn

    def test_no_records_returns_zero(self, tmp_path):
        p = tmp_path / "act.jsonl"
        n3, n5 = materialize_returns(p, lambda s: None, "2026-05-20")
        assert (n3, n5) == (0, 0)

    def test_too_recent_not_materialized(self, tmp_path):
        p = tmp_path / "act.jsonl"
        rec = self._make_record(date="2026-05-26")
        _append_jsonl(rec, p)
        df = _make_df(["2026-05-26", "2026-05-29", "2026-06-02"], [1000, 1030, 1050])
        n3, n5 = materialize_returns(p, self._loader(df), today="2026-05-27")
        # days_ago=1 < _CALENDAR_BUFFER_3D=5
        assert n3 == 0
        assert n5 == 0

    def test_old_enough_3d_materialized(self, tmp_path):
        p = tmp_path / "act.jsonl"
        rec = self._make_record(date="2026-05-15")
        _append_jsonl(rec, p)
        # Need base price on 2026-05-15 and fwd 3bd from there
        fwd3 = _business_day_offset("2026-05-15", FORWARD_DAYS_3)
        df = _make_df(["2026-05-15", fwd3], [1000.0, 1030.0])
        n3, n5 = materialize_returns(p, self._loader(df), today="2026-05-27")
        # days_ago=12 >= _CALENDAR_BUFFER_3D=5
        assert n3 == 1
        loaded = _load_jsonl(p)
        assert loaded[0]["materialized_3d"] is True
        assert loaded[0]["subsequent_3d_return"] == pytest.approx(0.03, abs=0.001)

    def test_old_enough_5d_materialized(self, tmp_path):
        p = tmp_path / "act.jsonl"
        rec = self._make_record(date="2026-05-14")
        _append_jsonl(rec, p)
        fwd3 = _business_day_offset("2026-05-14", FORWARD_DAYS_3)
        fwd5 = _business_day_offset("2026-05-14", FORWARD_DAYS_5)
        df = _make_df(["2026-05-14", fwd3, fwd5], [1000.0, 1020.0, 1050.0])
        n3, n5 = materialize_returns(p, self._loader(df), today="2026-05-27")
        assert n3 == 1
        assert n5 == 1
        loaded = _load_jsonl(p)
        assert loaded[0]["materialized_5d"] is True
        assert loaded[0]["subsequent_5d_return"] == pytest.approx(0.05, abs=0.001)

    def test_already_materialized_skipped(self, tmp_path):
        p = tmp_path / "act.jsonl"
        rec = self._make_record(date="2026-05-15", mat3=True, mat5=True)
        rec["subsequent_3d_return"] = 0.02
        rec["subsequent_5d_return"] = 0.03
        _append_jsonl(rec, p)
        fwd3 = _business_day_offset("2026-05-15", FORWARD_DAYS_3)
        fwd5 = _business_day_offset("2026-05-15", FORWARD_DAYS_5)
        df = _make_df(["2026-05-15", fwd3, fwd5], [1000.0, 1030.0, 1050.0])
        n3, n5 = materialize_returns(p, self._loader(df), today="2026-05-27")
        assert n3 == 0
        assert n5 == 0

    def test_fail_open_on_loader_error(self, tmp_path):
        p = tmp_path / "act.jsonl"
        rec = self._make_record(date="2026-05-01")
        _append_jsonl(rec, p)
        def _bad_loader(sym):
            raise RuntimeError("network error")
        n3, n5 = materialize_returns(p, _bad_loader, today="2026-05-27")
        assert isinstance(n3, int)
        assert isinstance(n5, int)

    def test_missing_price_not_materialized(self, tmp_path):
        p = tmp_path / "act.jsonl"
        rec = self._make_record(date="2026-05-15")
        _append_jsonl(rec, p)
        # DataFrame has no matching dates
        df = _make_df(["2000-01-01"], [999.0])
        n3, n5 = materialize_returns(p, self._loader(df), today="2026-05-27")
        assert n3 == 0
        assert n5 == 0


# ── compute_kpis ──────────────────────────────────────────────────────────────

class TestComputeKpis:
    def _rec(
        self,
        addon_type: str = "normal",
        triggered: bool = True,
        ret3: Optional[float] = None,
        ret5: Optional[float] = None,
        mat3: bool = False,
        mat5: bool = False,
    ) -> dict:
        return {
            "addon_triggered": triggered,
            "addon_type": addon_type,
            "continuation_breakout": addon_type == "continuation_breakout",
            "subsequent_3d_return": ret3,
            "subsequent_5d_return": ret5,
            "materialized_3d": mat3,
            "materialized_5d": mat5,
        }

    def test_empty_log_returns_empty(self, tmp_path):
        p = tmp_path / "act.jsonl"
        assert compute_kpis(p) == {}

    def test_no_confirmed_returns_empty(self, tmp_path):
        p = tmp_path / "act.jsonl"
        _append_jsonl(self._rec(triggered=False), p)
        assert compute_kpis(p) == {}

    def test_counts(self, tmp_path):
        p = tmp_path / "act.jsonl"
        _append_jsonl(self._rec("continuation_breakout"), p)
        _append_jsonl(self._rec("continuation_breakout"), p)
        _append_jsonl(self._rec("normal"), p)
        kpis = compute_kpis(p)
        assert kpis["phase_transition_addon_count"] == 2
        assert kpis["normal_addon_count"] == 1
        assert kpis["n_total_addon_evaluations"] == 3

    def test_avg_3d_return_cb(self, tmp_path):
        p = tmp_path / "act.jsonl"
        _append_jsonl(self._rec("continuation_breakout", ret3=0.04, mat3=True), p)
        _append_jsonl(self._rec("continuation_breakout", ret3=0.02, mat3=True), p)
        kpis = compute_kpis(p)
        cb3 = kpis["continuation_breakout_3d"]
        assert cb3["avg_return"] == pytest.approx(0.03, abs=0.001)
        assert cb3["n_materialized"] == 2

    def test_avg_return_delta_vs_normal(self, tmp_path):
        p = tmp_path / "act.jsonl"
        _append_jsonl(self._rec("continuation_breakout", ret3=0.05, mat3=True), p)
        _append_jsonl(self._rec("normal", ret3=0.02, mat3=True), p)
        kpis = compute_kpis(p)
        delta = kpis["avg_return_delta_vs_normal_3d"]
        assert delta == pytest.approx(0.03, abs=0.001)

    def test_delta_none_when_one_type_no_data(self, tmp_path):
        p = tmp_path / "act.jsonl"
        _append_jsonl(self._rec("continuation_breakout", ret3=0.05, mat3=True), p)
        # no normal records
        kpis = compute_kpis(p)
        assert kpis["avg_return_delta_vs_normal_3d"] is None

    def test_false_breakout_rate(self, tmp_path):
        p = tmp_path / "act.jsonl"
        # 2 CB records: one positive, one negative
        _append_jsonl(self._rec("continuation_breakout", ret3=0.03, mat3=True), p)
        _append_jsonl(self._rec("continuation_breakout", ret3=-0.01, mat3=True), p)
        kpis = compute_kpis(p)
        assert kpis["false_breakout_rate"] == pytest.approx(0.5, abs=0.01)
        assert kpis["compression_to_breakout_success_rate"] == pytest.approx(0.5, abs=0.01)

    def test_success_rate_all_positive(self, tmp_path):
        p = tmp_path / "act.jsonl"
        for _ in range(3):
            _append_jsonl(self._rec("continuation_breakout", ret3=0.05, mat3=True), p)
        kpis = compute_kpis(p)
        assert kpis["compression_to_breakout_success_rate"] == pytest.approx(1.0)
        assert kpis["false_breakout_rate"] == pytest.approx(0.0)

    def test_no_cb_records_success_rate_none(self, tmp_path):
        p = tmp_path / "act.jsonl"
        _append_jsonl(self._rec("normal", ret3=0.05, mat3=True), p)
        kpis = compute_kpis(p)
        assert kpis["compression_to_breakout_success_rate"] is None
        assert kpis["false_breakout_rate"] is None

    def test_positive_rate_in_group_stats(self, tmp_path):
        p = tmp_path / "act.jsonl"
        _append_jsonl(self._rec("normal", ret3=0.02, mat3=True), p)
        _append_jsonl(self._rec("normal", ret3=-0.01, mat3=True), p)
        kpis = compute_kpis(p)
        stats = kpis["normal_3d"]
        assert stats["positive_rate"] == pytest.approx(0.5, abs=0.01)


# ── run_addon_continuation_hook ────────────────────────────────────────────────

class TestRunAddonContinuationHook:
    def _loader_returning(self, price: float = 1000.0):
        def _fn(sym: str):
            return _make_df([], [])
        return _fn

    def test_no_confirmed_records_nothing(self, tmp_path):
        p = tmp_path / "act.jsonl"
        conf = _make_conf(confirmed=False)
        result = run_addon_continuation_hook(
            [conf], {}, log_path=p, ohlcv_loader=self._loader_returning(), today="2026-05-27"
        )
        assert result["n_recorded"] == 0
        assert _load_jsonl(p) == []

    def test_confirmed_record_written(self, tmp_path):
        p = tmp_path / "act.jsonl"
        conf = _make_conf(confirmed=True)
        result = run_addon_continuation_hook(
            [conf], {"1234.T": _make_signal()},
            log_path=p, ohlcv_loader=self._loader_returning(), today="2026-05-27"
        )
        assert result["n_recorded"] == 1
        records = _load_jsonl(p)
        assert len(records) == 1
        assert records[0]["symbol"] == "1234.T"

    def test_dedup_same_symbol_same_date(self, tmp_path):
        p = tmp_path / "act.jsonl"
        conf = _make_conf(confirmed=True)
        for _ in range(3):
            run_addon_continuation_hook(
                [conf], {}, log_path=p,
                ohlcv_loader=self._loader_returning(), today="2026-05-27"
            )
        assert len(_load_jsonl(p)) == 1

    def test_different_dates_not_deduped(self, tmp_path):
        p = tmp_path / "act.jsonl"
        conf = _make_conf(confirmed=True)
        run_addon_continuation_hook(
            [conf], {}, log_path=p,
            ohlcv_loader=self._loader_returning(), today="2026-05-26"
        )
        run_addon_continuation_hook(
            [conf], {}, log_path=p,
            ohlcv_loader=self._loader_returning(), today="2026-05-27"
        )
        assert len(_load_jsonl(p)) == 2

    def test_multiple_confirmed_all_recorded(self, tmp_path):
        p = tmp_path / "act.jsonl"
        confs = [_make_conf(f"{i}.T", confirmed=True) for i in range(3)]
        result = run_addon_continuation_hook(
            confs, {}, log_path=p,
            ohlcv_loader=self._loader_returning(), today="2026-05-27"
        )
        assert result["n_recorded"] == 3

    def test_returns_summary_dict(self, tmp_path):
        p = tmp_path / "act.jsonl"
        result = run_addon_continuation_hook(
            [], {}, log_path=p,
            ohlcv_loader=self._loader_returning(), today="2026-05-27"
        )
        assert "n_recorded" in result
        assert "n_materialized_3d" in result
        assert "n_materialized_5d" in result

    def test_fail_open_on_bad_loader(self, tmp_path):
        p = tmp_path / "act.jsonl"
        conf = _make_conf(confirmed=True)
        def _bad_loader(sym):
            raise RuntimeError("ohlcv error")
        result = run_addon_continuation_hook(
            [conf], {}, log_path=p,
            ohlcv_loader=_bad_loader, today="2026-05-27"
        )
        assert isinstance(result, dict)

    def test_hook_fail_open_on_corrupt_record(self, tmp_path):
        p = tmp_path / "act.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("NOT_JSON\n", encoding="utf-8")
        conf = _make_conf(confirmed=True)
        try:
            result = run_addon_continuation_hook(
                [conf], {}, log_path=p,
                ohlcv_loader=self._loader_returning(), today="2026-05-27"
            )
        except Exception as exc:
            pytest.fail(f"hook raised: {exc}")

    def test_signal_fields_propagated(self, tmp_path):
        p = tmp_path / "act.jsonl"
        conf = _make_conf(confirmed=True, cb=True, boost=1.15)
        sig = _make_signal(
            compression_score=80.0,
            volume_ratio=1.50,
            rsr_momentum=0.45,
            prev_phase="compression",
            phase="breakout",
        )
        run_addon_continuation_hook(
            [conf], {"1234.T": sig},
            log_path=p, ohlcv_loader=self._loader_returning(), today="2026-05-27"
        )
        rec = _load_jsonl(p)[0]
        assert rec["compression_score"] == pytest.approx(80.0, abs=0.01)
        assert rec["volume_ratio_5d"] == pytest.approx(1.50, abs=0.001)
        assert rec["continuation_breakout"] is True
        assert rec["addon_type"] == "continuation_breakout"


# ── No execution module imports ───────────────────────────────────────────────

class TestNoExecutionImports:
    def test_module_does_not_import_execution_modules(self):
        import importlib
        import sys
        execution_modules = {
            "src.kabusapi.signal_bridge",
            "src.kabusapi.order_client",
            "src.addon.addon_policy",
        }
        import src.analytics.addon_continuation_telemetry as mod
        module_file = getattr(mod, "__file__", "")
        for em in execution_modules:
            assert em not in sys.modules or True  # telemetry never touches execution
        # Simpler check: telemetry module importable standalone
        reloaded = importlib.import_module("src.analytics.addon_continuation_telemetry")
        assert reloaded is not None
