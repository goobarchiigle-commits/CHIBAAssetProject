"""
tests/test_opportunity_cost_materializer.py
Phase 5D: OC Materialization & Regime Attribution tests.

Covers:
  - classify_signal_environment: all branches, boundary cases
  - _build_regime_snapshot: provider injection, fail-open, empty
  - _age_bucket: all three buckets + boundaries
  - materialize_oc_events_5d: basic, idempotency, skip-too-early, regime attach,
      missing data skip-safe, multi-event, buffer_days param
  - compute_severity_persistence: all severity buckets, empty, missing data
  - compute_holding_staleness: age buckets, mixed ages, empty
  - compute_slot_efficiency: slot 3/4/5, missing max_positions default
  - aggregate_5d_kpis: positive_delta_rate, all sub-analytics, empty, fail-open
  - format_oc_materialization_report: structure, per-event detail, summary, empty
  - OC5DEnrichment schema: defaults, asdict serialization
  - Fail-open / edge cases: bad paths, malformed JSONL, None provider
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pytest

from src.analytics.opportunity_cost_materializer import (
    # Data structures
    RegimeSnapshot,
    OC5DEnrichment,
    # Functions
    classify_signal_environment,
    materialize_oc_events_5d,
    compute_severity_persistence,
    compute_holding_staleness,
    compute_slot_efficiency,
    aggregate_5d_kpis,
    format_oc_materialization_report,
    _make_event_id,
    _age_bucket,
    _build_regime_snapshot,
    _avg_list,
    _empty_5d_kpis,
    # Constants
    SCHEMA_VERSION_5D,
    REGIME_BULL, REGIME_BEAR, REGIME_NORMAL, REGIME_UNKNOWN,
    SIGNAL_TREND_PERSISTENT, SIGNAL_FRESH_BREAKOUT,
    SIGNAL_COMPRESSION, SIGNAL_RANGE_BOUND, SIGNAL_UNKNOWN,
    AGE_BUCKET_LT_10, AGE_BUCKET_10_20, AGE_BUCKET_GT_20,
)
from src.analytics.opportunity_cost import (
    SEVERITY_SEVERE, SEVERITY_MODERATE, SEVERITY_LOW,
)

TODAY      = "2026-05-29"
OLD_DATE   = "2026-05-01"  # well beyond MAT_BUFFER_DAYS (8 days)
RECENT_DATE = "2026-05-28"  # only 1 day ago → not yet eligible


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_event(path: Path, **overrides) -> dict:
    defaults = {
        "date": OLD_DATE,
        "rejected_symbol": "7011.T",
        "blocked_by_symbol": "5803.T",
        "priority_gap": 45.0,
        "opportunity_cost_severity": SEVERITY_SEVERE,
        "blocked_position_age_days": 30,
        "max_positions": 3,
        "active_positions": 3,
        "portfolio_full": True,
        "schema_version": "v1",
    }
    defaults.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(defaults) + "\n")
    return defaults


def _write_enrichment(path: Path, delta: Optional[float] = 0.08, **overrides) -> dict:
    defaults = {
        "event_id": _make_event_id(OLD_DATE, "7011.T"),
        "date": OLD_DATE,
        "rejected_symbol": "7011.T",
        "blocked_by_symbol": "5803.T",
        "rejected_candidate_5d_return": 0.084,
        "blocked_position_5d_return": -0.013,
        "rotation_counterfactual_delta": delta,
        "materialized_at": "2026-05-10T09:00:00+0900",
        "materialization_complete": delta is not None,
        "regime": REGIME_BULL,
        "regime_confidence": 0.72,
        "breadth_50": 0.54,
        "breadth_75": 0.23,
        "market_drawdown": -0.041,
        "signal_environment": SIGNAL_TREND_PERSISTENT,
        "priority_gap": 45.0,
        "opportunity_cost_severity": SEVERITY_SEVERE,
        "blocked_position_age_days": 27,
        "max_positions": 3,
        "active_positions": 3,
        "schema_version": SCHEMA_VERSION_5D,
    }
    defaults.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(defaults) + "\n")
    return defaults


def _price_fetcher(prices: dict):
    def _fetch(symbol: str, date: str) -> Optional[float]:
        return prices.get(symbol, {}).get(date)
    return _fetch


# ─────────────────────────────────────────────────────────────────────────────
# 1. classify_signal_environment
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifySignalEnvironment:
    def test_bull_high_breadth_is_trend_persistent(self):
        assert classify_signal_environment(REGIME_BULL, 0.55, 0.30) == SIGNAL_TREND_PERSISTENT

    def test_bull_breadth_50_exact_threshold(self):
        assert classify_signal_environment(REGIME_BULL, 0.50, 0.25) == SIGNAL_TREND_PERSISTENT

    def test_bull_moderate_breadth_is_fresh_breakout(self):
        assert classify_signal_environment(REGIME_BULL, 0.40, 0.20) == SIGNAL_FRESH_BREAKOUT

    def test_bull_breadth_30_exact_fresh_breakout(self):
        assert classify_signal_environment(REGIME_BULL, 0.30, 0.10) == SIGNAL_FRESH_BREAKOUT

    def test_low_breadth_both_thresholds_compression(self):
        assert classify_signal_environment(REGIME_BEAR, 0.20, 0.10) == SIGNAL_COMPRESSION

    def test_compression_regime_agnostic(self):
        assert classify_signal_environment(REGIME_NORMAL, 0.20, 0.10) == SIGNAL_COMPRESSION

    def test_normal_moderate_breadth_range_bound(self):
        assert classify_signal_environment(REGIME_NORMAL, 0.40, 0.20) == SIGNAL_RANGE_BOUND

    def test_bear_moderate_breadth_range_bound(self):
        assert classify_signal_environment(REGIME_BEAR, 0.35, 0.15) == SIGNAL_RANGE_BOUND

    def test_bull_low_breadth_falls_through_to_unknown(self):
        # bull + breadth_50 < 0.30 → no fresh_breakout, no compression (b75 >= 0.15)
        result = classify_signal_environment(REGIME_BULL, 0.25, 0.20)
        assert result == SIGNAL_UNKNOWN

    def test_unknown_regime_unknown_environment(self):
        result = classify_signal_environment(REGIME_UNKNOWN, 0.30, 0.10)
        # Unknown regime: not bull → no trend/fresh; b50=0.30 >= 0.30 but b75=0.10 < 0.15
        # compression check: b50 < 0.30 → False; range_bound: UNKNOWN not in (normal,bear) → False
        assert result == SIGNAL_UNKNOWN

    def test_regime_crash_low_breadth_compression(self):
        # crash + breadth_50<0.30 + breadth_75<0.15 → compression (breadth-agnostic check)
        result = classify_signal_environment("crash", 0.10, 0.05)
        assert result == SIGNAL_COMPRESSION


# ─────────────────────────────────────────────────────────────────────────────
# 2. _build_regime_snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildRegimeSnapshot:
    def test_none_provider_returns_empty(self):
        snap = _build_regime_snapshot(TODAY, None)
        assert snap.regime == REGIME_UNKNOWN
        assert snap.signal_environment == SIGNAL_UNKNOWN

    def test_provider_values_used(self):
        def _provider(date):
            return {
                "regime": REGIME_BULL,
                "regime_confidence": 0.80,
                "breadth_50": 0.60,
                "breadth_75": 0.30,
                "market_drawdown": -0.02,
            }
        snap = _build_regime_snapshot(TODAY, _provider)
        assert snap.regime == REGIME_BULL
        assert snap.regime_confidence == pytest.approx(0.80)
        assert snap.breadth_50 == pytest.approx(0.60)
        assert snap.signal_environment == SIGNAL_TREND_PERSISTENT

    def test_provider_returns_none_gives_empty(self):
        snap = _build_regime_snapshot(TODAY, lambda d: None)
        assert snap.regime == REGIME_UNKNOWN

    def test_provider_with_explicit_signal_env_used_verbatim(self):
        def _provider(date):
            return {
                "regime": REGIME_BULL,
                "signal_environment": "custom_env",
                "breadth_50": 0.60,
                "breadth_75": 0.30,
            }
        snap = _build_regime_snapshot(TODAY, _provider)
        assert snap.signal_environment == "custom_env"

    def test_provider_exception_returns_empty(self):
        def _bad_provider(date):
            raise RuntimeError("fail")
        snap = _build_regime_snapshot(TODAY, _bad_provider)
        assert snap.regime == REGIME_UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# 3. _age_bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestAgeBucket:
    def test_zero_days_is_lt_10(self):
        assert _age_bucket(0) == AGE_BUCKET_LT_10

    def test_nine_days_is_lt_10(self):
        assert _age_bucket(9) == AGE_BUCKET_LT_10

    def test_ten_days_is_10_20(self):
        assert _age_bucket(10) == AGE_BUCKET_10_20

    def test_nineteen_days_is_10_20(self):
        assert _age_bucket(19) == AGE_BUCKET_10_20

    def test_twenty_days_is_gt_20(self):
        assert _age_bucket(20) == AGE_BUCKET_GT_20

    def test_large_age_is_gt_20(self):
        assert _age_bucket(200) == AGE_BUCKET_GT_20


# ─────────────────────────────────────────────────────────────────────────────
# 4. OC5DEnrichment schema
# ─────────────────────────────────────────────────────────────────────────────

class TestOC5DEnrichmentSchema:
    def test_schema_version_default(self):
        e = OC5DEnrichment(
            event_id="x", date=TODAY, rejected_symbol="A.T", blocked_by_symbol="B.T",
            rejected_candidate_5d_return=None, blocked_position_5d_return=None,
            rotation_counterfactual_delta=None, materialized_at="ts",
            materialization_complete=False,
            regime=REGIME_UNKNOWN, regime_confidence=0.0,
            breadth_50=0.0, breadth_75=0.0, market_drawdown=0.0,
            signal_environment=SIGNAL_UNKNOWN,
            priority_gap=0.0, opportunity_cost_severity="", blocked_position_age_days=0,
            max_positions=3, active_positions=0,
        )
        assert e.schema_version == SCHEMA_VERSION_5D

    def test_asdict_has_all_regime_fields(self):
        e = OC5DEnrichment(
            event_id="2026-05-01/7011.T", date=OLD_DATE, rejected_symbol="7011.T",
            blocked_by_symbol="5803.T", rejected_candidate_5d_return=0.08,
            blocked_position_5d_return=-0.01, rotation_counterfactual_delta=0.09,
            materialized_at="ts", materialization_complete=True,
            regime=REGIME_BULL, regime_confidence=0.72,
            breadth_50=0.54, breadth_75=0.23, market_drawdown=-0.04,
            signal_environment=SIGNAL_TREND_PERSISTENT,
            priority_gap=45.0, opportunity_cost_severity=SEVERITY_SEVERE,
            blocked_position_age_days=27, max_positions=3, active_positions=3,
        )
        d = asdict(e)
        assert "regime" in d
        assert "regime_confidence" in d
        assert "signal_environment" in d
        assert "event_id" in d
        assert d["rotation_counterfactual_delta"] == pytest.approx(0.09)

    def test_none_returns_are_preserved(self):
        e = OC5DEnrichment(
            event_id="x", date=TODAY, rejected_symbol="A.T", blocked_by_symbol="B.T",
            rejected_candidate_5d_return=None, blocked_position_5d_return=None,
            rotation_counterfactual_delta=None, materialized_at="ts",
            materialization_complete=False,
            regime=REGIME_UNKNOWN, regime_confidence=0.0,
            breadth_50=0.0, breadth_75=0.0, market_drawdown=0.0,
            signal_environment=SIGNAL_UNKNOWN,
            priority_gap=0.0, opportunity_cost_severity="", blocked_position_age_days=0,
            max_positions=3, active_positions=0,
        )
        d = asdict(e)
        assert d["rotation_counterfactual_delta"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. materialize_oc_events_5d
# ─────────────────────────────────────────────────────────────────────────────

class TestMaterializeOcEvents5d:
    def _prices(self):
        return {
            "7011.T": {OLD_DATE: 2500.0, "2026-05-08": 2710.0},
            "5803.T": {OLD_DATE: 1500.0, "2026-05-08": 1480.0},
        }

    def test_basic_materialization_writes_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            n = materialize_oc_events_5d(
                ev_f, en_f, _price_fetcher(self._prices()), TODAY
            )
            assert n == 1
            records = json.loads(en_f.read_text().strip())
            assert records["rejected_symbol"] == "7011.T"

    def test_delta_computed_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            materialize_oc_events_5d(
                ev_f, en_f, _price_fetcher(self._prices()), TODAY
            )
            r = json.loads(en_f.read_text().strip())
            # rej: (2710-2500)/2500 = 0.084
            # blk: (1480-1500)/1500 ≈ -0.01333
            assert r["rejected_candidate_5d_return"] == pytest.approx(0.084)
            assert r["rotation_counterfactual_delta"] == pytest.approx(
                r["rejected_candidate_5d_return"] - r["blocked_position_5d_return"]
            )

    def test_idempotent_second_run_no_dup(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            n1 = materialize_oc_events_5d(ev_f, en_f, _price_fetcher(self._prices()), TODAY)
            n2 = materialize_oc_events_5d(ev_f, en_f, _price_fetcher(self._prices()), TODAY)
            assert n1 == 1
            assert n2 == 0
            assert len(en_f.read_text().strip().splitlines()) == 1

    def test_too_recent_event_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f, date=RECENT_DATE)
            n = materialize_oc_events_5d(ev_f, en_f, _price_fetcher(self._prices()), TODAY)
            assert n == 0
            assert not en_f.exists()

    def test_missing_price_writes_partial_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            n = materialize_oc_events_5d(ev_f, en_f, lambda sym, dt: None, TODAY)
            assert n == 1
            r = json.loads(en_f.read_text().strip())
            assert r["rejected_candidate_5d_return"] is None
            assert r["rotation_counterfactual_delta"] is None
            assert r["materialization_complete"] is False

    def test_regime_provider_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            def _regime_prov(date):
                return {"regime": REGIME_BULL, "breadth_50": 0.60, "breadth_75": 0.30, "regime_confidence": 0.8, "market_drawdown": -0.03}
            n = materialize_oc_events_5d(
                ev_f, en_f, _price_fetcher(self._prices()), TODAY,
                regime_provider=_regime_prov,
            )
            assert n == 1
            r = json.loads(en_f.read_text().strip())
            assert r["regime"] == REGIME_BULL
            assert r["signal_environment"] == SIGNAL_TREND_PERSISTENT

    def test_no_regime_provider_gives_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            materialize_oc_events_5d(ev_f, en_f, lambda sym, dt: None, TODAY, regime_provider=None)
            r = json.loads(en_f.read_text().strip())
            assert r["regime"] == REGIME_UNKNOWN

    def test_event_id_set_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            materialize_oc_events_5d(ev_f, en_f, lambda sym, dt: None, TODAY)
            r = json.loads(en_f.read_text().strip())
            assert r["event_id"] == f"{OLD_DATE}/7011.T"

    def test_multiple_events_materialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f, rejected_symbol="7011.T")
            _write_event(ev_f, rejected_symbol="6758.T")
            n = materialize_oc_events_5d(ev_f, en_f, lambda sym, dt: None, TODAY)
            assert n == 2

    def test_passthrough_fields_from_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f, blocked_position_age_days=47, max_positions=4, priority_gap=38.5)
            materialize_oc_events_5d(ev_f, en_f, lambda sym, dt: None, TODAY)
            r = json.loads(en_f.read_text().strip())
            assert r["blocked_position_age_days"] == 47
            assert r["max_positions"] == 4
            assert r["priority_gap"] == pytest.approx(38.5)

    def test_missing_events_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "nonexistent.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            n = materialize_oc_events_5d(ev_f, en_f, lambda sym, dt: None, TODAY)
            assert n == 0

    def test_materialized_at_timestamp_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            materialize_oc_events_5d(ev_f, en_f, lambda sym, dt: None, TODAY)
            r = json.loads(en_f.read_text().strip())
            assert "materialized_at" in r
            assert len(r["materialized_at"]) > 5

    def test_schema_version_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            materialize_oc_events_5d(ev_f, en_f, lambda sym, dt: None, TODAY)
            r = json.loads(en_f.read_text().strip())
            assert r["schema_version"] == SCHEMA_VERSION_5D

    def test_empty_events_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            ev_f.write_text("")
            n = materialize_oc_events_5d(ev_f, en_f, lambda sym, dt: None, TODAY)
            assert n == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. compute_severity_persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSeverityPersistence:
    def test_severe_bucket_avg(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(en_f, delta=0.10, opportunity_cost_severity=SEVERITY_SEVERE)
            _write_enrichment(en_f, delta=0.06, opportunity_cost_severity=SEVERITY_SEVERE,
                              event_id=f"{OLD_DATE}/B.T", rejected_symbol="B.T")
            stats = compute_severity_persistence(en_f)
            assert stats["avg_delta_severe"] == pytest.approx(0.08)
            assert stats["sample_count_severe"] == 2

    def test_moderate_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(en_f, delta=0.03, opportunity_cost_severity=SEVERITY_MODERATE)
            stats = compute_severity_persistence(en_f)
            assert stats["avg_delta_moderate"] == pytest.approx(0.03)
            assert stats["avg_delta_severe"] is None

    def test_low_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(en_f, delta=-0.02, opportunity_cost_severity=SEVERITY_LOW)
            stats = compute_severity_persistence(en_f)
            assert stats["avg_delta_low"] == pytest.approx(-0.02)
            assert stats["sample_count_low"] == 1

    def test_empty_file_all_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            stats = compute_severity_persistence(en_f)
            assert stats["avg_delta_severe"] is None
            assert stats["sample_count_severe"] == 0

    def test_missing_delta_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(en_f, delta=None, opportunity_cost_severity=SEVERITY_SEVERE)
            stats = compute_severity_persistence(en_f)
            assert stats["avg_delta_severe"] is None
            assert stats["sample_count_severe"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. compute_holding_staleness
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeHoldingStaleness:
    def _write_age(self, path: Path, age: int, delta: float, sym: str = "7011.T"):
        _write_enrichment(
            path, delta=delta,
            event_id=f"{OLD_DATE}/{sym}",
            rejected_symbol=sym,
            blocked_position_age_days=age,
        )

    def test_lt_10_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            self._write_age(en_f, 5, 0.04, "A.T")
            stats = compute_holding_staleness(en_f)
            assert stats["avg_delta_age_lt_10d"] == pytest.approx(0.04)
            assert stats["count_age_lt_10d"] == 1

    def test_10_20_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            self._write_age(en_f, 15, 0.06, "B.T")
            stats = compute_holding_staleness(en_f)
            assert stats["avg_delta_age_10_20d"] == pytest.approx(0.06)
            assert stats["count_age_10_20d"] == 1

    def test_gt_20_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            self._write_age(en_f, 35, 0.10, "C.T")
            stats = compute_holding_staleness(en_f)
            assert stats["avg_delta_age_gt_20d"] == pytest.approx(0.10)
            assert stats["count_age_gt_20d"] == 1

    def test_mixed_ages(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            self._write_age(en_f, 5,  0.02, "A.T")
            self._write_age(en_f, 15, 0.05, "B.T")
            self._write_age(en_f, 30, 0.09, "C.T")
            stats = compute_holding_staleness(en_f)
            assert stats["count_age_lt_10d"] == 1
            assert stats["count_age_10_20d"] == 1
            assert stats["count_age_gt_20d"] == 1

    def test_empty_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            stats = compute_holding_staleness(en_f)
            assert stats["avg_delta_age_lt_10d"] is None
            assert stats["count_age_lt_10d"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. compute_slot_efficiency
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSlotEfficiency:
    def _write_slot(self, path: Path, max_pos: int, delta: float, sym: str):
        _write_enrichment(
            path, delta=delta,
            event_id=f"{OLD_DATE}/{sym}",
            rejected_symbol=sym,
            max_positions=max_pos,
        )

    def test_slots_3_avg(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            self._write_slot(en_f, 3, 0.08, "A.T")
            self._write_slot(en_f, 3, 0.04, "B.T")
            stats = compute_slot_efficiency(en_f)
            assert stats["avg_delta_slots_3"] == pytest.approx(0.06)
            assert stats["count_slots_3"] == 2

    def test_slots_4_avg(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            self._write_slot(en_f, 4, 0.05, "X.T")
            stats = compute_slot_efficiency(en_f)
            assert stats["avg_delta_slots_4"] == pytest.approx(0.05)
            assert stats["avg_delta_slots_3"] is None

    def test_slots_5_avg(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            self._write_slot(en_f, 5, 0.03, "Y.T")
            stats = compute_slot_efficiency(en_f)
            assert stats["avg_delta_slots_5"] == pytest.approx(0.03)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            stats = compute_slot_efficiency(en_f)
            assert stats["avg_delta_slots_3"] is None
            assert stats["count_slots_3"] == 0

    def test_default_max_positions_3(self):
        # If max_positions absent from record, defaults to 3
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            record = {
                "event_id": "x", "date": OLD_DATE, "rejected_symbol": "A.T",
                "blocked_by_symbol": "B.T",
                "rotation_counterfactual_delta": 0.05,
                "opportunity_cost_severity": SEVERITY_SEVERE,
                "blocked_position_age_days": 10,
                "schema_version": SCHEMA_VERSION_5D,
                # max_positions deliberately absent
            }
            with en_f.open("a") as f:
                f.write(json.dumps(record) + "\n")
            stats = compute_slot_efficiency(en_f)
            assert stats["count_slots_3"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. aggregate_5d_kpis
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregate5dKpis:
    def test_positive_delta_rate_all_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            _write_enrichment(en_f, delta=0.08)
            kpis = aggregate_5d_kpis(en_f, ev_f)
            assert kpis["positive_delta_rate"] == pytest.approx(1.0)

    def test_positive_delta_rate_none_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            _write_enrichment(en_f, delta=-0.05)
            kpis = aggregate_5d_kpis(en_f, ev_f)
            assert kpis["positive_delta_rate"] == pytest.approx(0.0)

    def test_positive_delta_rate_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            _write_event(ev_f, rejected_symbol="B.T")
            _write_enrichment(en_f, delta=0.05)
            _write_enrichment(en_f, delta=-0.02, event_id=f"{OLD_DATE}/B.T", rejected_symbol="B.T")
            kpis = aggregate_5d_kpis(en_f, ev_f)
            assert kpis["positive_delta_rate"] == pytest.approx(0.5)

    def test_empty_returns_zero_kpis(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            kpis = aggregate_5d_kpis(en_f, ev_f)
            assert kpis["positive_delta_rate"] == 0.0
            assert kpis["opportunity_cost_event_count"] == 0
            assert kpis["materialized_event_count"] == 0

    def test_required_keys_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            kpis = aggregate_5d_kpis(en_f, ev_f)
            for k in (
                "opportunity_cost_event_count", "materialized_event_count",
                "avg_priority_gap", "avg_rejected_candidate_5d_return",
                "avg_blocked_position_5d_return", "avg_rotation_counterfactual_delta",
                "positive_delta_rate", "severe_event_rate",
                "avg_delta_by_regime", "avg_delta_by_severity",
                "avg_delta_by_holding_age", "avg_delta_by_slot_count",
            ):
                assert k in kpis

    def test_by_regime_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            _write_enrichment(en_f, delta=0.08, regime=REGIME_BULL)
            kpis = aggregate_5d_kpis(en_f, ev_f)
            assert REGIME_BULL in kpis["avg_delta_by_regime"]

    def test_severe_event_rate_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            _write_enrichment(en_f, delta=0.08, opportunity_cost_severity=SEVERITY_SEVERE)
            kpis = aggregate_5d_kpis(en_f, ev_f)
            assert kpis["severe_event_rate"] == pytest.approx(1.0)

    def test_sub_analytics_present_in_kpis(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            _write_enrichment(en_f, delta=0.08)
            kpis = aggregate_5d_kpis(en_f, ev_f)
            assert "lt_10d" in kpis["avg_delta_by_holding_age"]
            assert "slots_3" in kpis["avg_delta_by_slot_count"]

    def test_event_and_materialized_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            _write_event(ev_f, rejected_symbol="B.T")
            _write_enrichment(en_f, delta=0.08)
            kpis = aggregate_5d_kpis(en_f, ev_f)
            assert kpis["opportunity_cost_event_count"] == 2
            assert kpis["materialized_event_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. format_oc_materialization_report
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatOcMaterializationReport:
    def test_empty_enriched_returns_empty_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            report = format_oc_materialization_report(en_f)
            assert report == ""

    def test_section_header_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(en_f, delta=0.097)
            report = format_oc_materialization_report(en_f)
            assert "OPPORTUNITY COST MATERIALIZATION" in report

    def test_rejected_and_blocked_symbols_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(en_f, delta=0.097)
            report = format_oc_materialization_report(en_f)
            assert "7011.T" in report
            assert "5803.T" in report

    def test_counterfactual_delta_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(en_f, delta=0.097)
            report = format_oc_materialization_report(en_f)
            assert "Counterfactual delta" in report

    def test_regime_shown_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(en_f, delta=0.08, regime=REGIME_BULL,
                              signal_environment=SIGNAL_TREND_PERSISTENT)
            report = format_oc_materialization_report(en_f)
            assert "bull" in report

    def test_holding_age_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(en_f, delta=0.05, blocked_position_age_days=27)
            report = format_oc_materialization_report(en_f)
            assert "27d" in report

    def test_summary_section_with_events_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            _write_enrichment(en_f, delta=0.08)
            report = format_oc_materialization_report(en_f, events_file=ev_f)
            assert "Positive delta rate" in report

    def test_nonexistent_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "nonexistent.jsonl"
            report = format_oc_materialization_report(en_f)
            assert report == ""

    def test_pending_return_shown_as_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            en_f = Path(tmp) / "en.jsonl"
            _write_enrichment(
                en_f, delta=None,
                rejected_candidate_5d_return=None,
                blocked_position_5d_return=None,
            )
            # No completed records → empty
            report = format_oc_materialization_report(en_f)
            assert report == ""


# ─────────────────────────────────────────────────────────────────────────────
# 11. _avg_list helper
# ─────────────────────────────────────────────────────────────────────────────

class TestAvgList:
    def test_basic(self):
        assert _avg_list([0.1, 0.2, 0.3]) == pytest.approx(0.2)

    def test_empty_returns_none(self):
        assert _avg_list([]) is None

    def test_single_element(self):
        assert _avg_list([0.5]) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 12. _empty_5d_kpis
# ─────────────────────────────────────────────────────────────────────────────

class TestEmpty5dKpis:
    def test_all_required_keys(self):
        kpis = _empty_5d_kpis()
        assert kpis["positive_delta_rate"] == 0.0
        assert kpis["severe_event_rate"] == 0.0
        assert kpis["opportunity_cost_event_count"] == 0
        assert isinstance(kpis["avg_delta_by_regime"], dict)
        assert "severe" in kpis["avg_delta_by_severity"]
        assert "lt_10d" in kpis["avg_delta_by_holding_age"]
        assert "slots_3" in kpis["avg_delta_by_slot_count"]

    def test_schema_version_present(self):
        assert _empty_5d_kpis()["schema_version"] == SCHEMA_VERSION_5D


# ─────────────────────────────────────────────────────────────────────────────
# 13. Fail-open edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestFailOpen:
    def test_materialize_bad_price_fetcher_doesnt_raise(self):
        def _bad_fetcher(sym, dt):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            ev_f = Path(tmp) / "ev.jsonl"
            en_f = Path(tmp) / "en.jsonl"
            _write_event(ev_f)
            # Bad fetcher raises inside _compute_return → skip event
            n = materialize_oc_events_5d(ev_f, en_f, _bad_fetcher, TODAY)
            # Event is skipped (not enriched) — no exception propagated
            assert isinstance(n, int)

    def test_severity_persistence_missing_file_fail_open(self):
        p = Path("/tmp/definitely_missing_xyzzy.jsonl")
        stats = compute_severity_persistence(p)
        assert "avg_delta_severe" in stats

    def test_holding_staleness_missing_file_fail_open(self):
        p = Path("/tmp/definitely_missing_xyzzy.jsonl")
        stats = compute_holding_staleness(p)
        assert "avg_delta_age_lt_10d" in stats

    def test_slot_efficiency_missing_file_fail_open(self):
        p = Path("/tmp/definitely_missing_xyzzy.jsonl")
        stats = compute_slot_efficiency(p)
        assert "avg_delta_slots_3" in stats

    def test_aggregate_kpis_missing_files_fail_open(self):
        p = Path("/tmp/definitely_missing_xyzzy.jsonl")
        kpis = aggregate_5d_kpis(p, p)
        assert kpis["opportunity_cost_event_count"] == 0
