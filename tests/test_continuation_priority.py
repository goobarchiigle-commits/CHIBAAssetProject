"""
tests/test_continuation_priority.py — Continuation Priority Intelligence unit tests
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ── import target ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.continuation_priority import (
    BONUS_ACTIVE_BREAKOUT,
    BONUS_HEALTHY_BREAKOUT,
    BONUS_MFE_RETENTION,
    BONUS_RSR_STRONG,
    BONUS_SUPPRESSION,
    GIVEBACK_HIGH_THRESHOLD,
    HOLD_OVERSTAY_AT,
    MFE_FULL_AT,
    MFE_HALF_AT,
    MFE_RETENTION_MAX_GB,
    MFE_RETENTION_MIN_MFE,
    PENALTY_DISTRIBUTION,
    PENALTY_FAILED_BREAKOUT,
    PENALTY_HIGH_GIVEBACK,
    PENALTY_OVERSTAY,
    PENALTY_RSR_NEGATIVE_MOM,
    RSR_SCORE_CEIL,
    RSR_SCORE_FLOOR,
    RSR_STRONG_THRESHOLD,
    SCHEMA_VERSION,
    TIER_CORE,
    TIER_EXIT,
    TIER_HIGH,
    TIER_NEUTRAL,
    TIER_THRESHOLDS,
    W_BQ,
    W_COMPRESSION,
    W_HOLD,
    W_MFE,
    W_RSR,
    ContinuationPriorityInput,
    ContinuationPriorityResult,
    append_cp_telemetry,
    classify_tier,
    compute_continuation_priority,
    compute_cp_kpis,
    format_cp_kpi_section,
    format_priority_ranking_table,
    hold_quality_score,
    materialize_cp_returns,
    mfe_quality_score,
    rsr_strength_score,
)

_JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_inp(**kwargs) -> ContinuationPriorityInput:
    defaults = dict(
        symbol="1234",
        hold_days=10,
        unrealized_pnl_pct=0.05,
        compression_score=60.0,
        breakout_quality_score=70.0,
        rsr=82.0,
        rsr_momentum=0.5,
        mfe_pct=0.05,
        giveback_ratio=0.0,
        current_phase="healthy_breakout",
        suppression_eligible=False,
    )
    defaults.update(kwargs)
    return ContinuationPriorityInput(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_weights_sum_to_one(self):
        total = W_COMPRESSION + W_BQ + W_RSR + W_MFE + W_HOLD
        assert abs(total - 1.0) < 1e-9

    def test_tier_thresholds_all_defined(self):
        for t in (TIER_CORE, TIER_HIGH, TIER_NEUTRAL, TIER_EXIT):
            assert t in TIER_THRESHOLDS

    def test_tier_thresholds_descending(self):
        assert TIER_THRESHOLDS[TIER_CORE] > TIER_THRESHOLDS[TIER_HIGH]
        assert TIER_THRESHOLDS[TIER_HIGH] > TIER_THRESHOLDS[TIER_NEUTRAL]
        assert TIER_THRESHOLDS[TIER_NEUTRAL] > TIER_THRESHOLDS[TIER_EXIT]

    def test_exit_tier_threshold_zero(self):
        assert TIER_THRESHOLDS[TIER_EXIT] == 0.0

    def test_bonus_constants_positive(self):
        for b in (BONUS_HEALTHY_BREAKOUT, BONUS_SUPPRESSION, BONUS_MFE_RETENTION,
                  BONUS_RSR_STRONG, BONUS_ACTIVE_BREAKOUT):
            assert b > 0

    def test_penalty_constants_positive(self):
        for p in (PENALTY_DISTRIBUTION, PENALTY_FAILED_BREAKOUT,
                  PENALTY_HIGH_GIVEBACK, PENALTY_RSR_NEGATIVE_MOM, PENALTY_OVERSTAY):
            assert p > 0

    def test_schema_version_string(self):
        assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# rsr_strength_score
# ─────────────────────────────────────────────────────────────────────────────

class TestRsrStrengthScore:
    def test_at_floor_returns_zero(self):
        assert rsr_strength_score(RSR_SCORE_FLOOR) == 0.0

    def test_below_floor_returns_zero(self):
        assert rsr_strength_score(50.0) == 0.0

    def test_at_ceil_returns_100(self):
        assert rsr_strength_score(RSR_SCORE_CEIL) == 100.0

    def test_above_ceil_clamped_100(self):
        assert rsr_strength_score(110.0) == 100.0

    def test_midpoint_linear(self):
        mid = (RSR_SCORE_FLOOR + RSR_SCORE_CEIL) / 2
        assert abs(rsr_strength_score(mid) - 50.0) < 0.1

    def test_nan_returns_zero(self):
        assert rsr_strength_score(float("nan")) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# mfe_quality_score
# ─────────────────────────────────────────────────────────────────────────────

class TestMfeQualityScore:
    def test_zero_returns_zero(self):
        assert mfe_quality_score(0.0) == 0.0

    def test_negative_returns_zero(self):
        assert mfe_quality_score(-0.05) == 0.0

    def test_half_at_is_50(self):
        assert abs(mfe_quality_score(MFE_HALF_AT) - 50.0) < 0.1

    def test_full_at_is_100(self):
        assert abs(mfe_quality_score(MFE_FULL_AT) - 100.0) < 0.1

    def test_above_full_is_100(self):
        assert mfe_quality_score(0.5) == 100.0

    def test_midpoint_lower_segment(self):
        mid = MFE_HALF_AT / 2
        s = mfe_quality_score(mid)
        assert 0 < s < 50

    def test_midpoint_upper_segment(self):
        mid = (MFE_HALF_AT + MFE_FULL_AT) / 2
        s = mfe_quality_score(mid)
        assert 50 < s < 100

    def test_nan_returns_zero(self):
        assert mfe_quality_score(float("nan")) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# hold_quality_score
# ─────────────────────────────────────────────────────────────────────────────

class TestHoldQualityScore:
    def test_early_days(self):
        assert hold_quality_score(0) == 30.0
        assert hold_quality_score(1) == 30.0

    def test_ramp_phase(self):
        s3 = hold_quality_score(3)
        s4 = hold_quality_score(4)
        assert 30.0 <= s3 < s4

    def test_prime_start_is_60(self):
        assert abs(hold_quality_score(5) - 60.0) < 0.1

    def test_prime_end_is_90(self):
        assert abs(hold_quality_score(40) - 90.0) < 0.1

    def test_overstay_declines(self):
        assert hold_quality_score(50) < hold_quality_score(40)

    def test_floor_at_30(self):
        assert hold_quality_score(200) >= 30.0

    def test_negative_input(self):
        assert hold_quality_score(-5) == 30.0


# ─────────────────────────────────────────────────────────────────────────────
# classify_tier
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyTier:
    def test_core(self):
        assert classify_tier(80.0) == TIER_CORE
        assert classify_tier(100.0) == TIER_CORE

    def test_high(self):
        assert classify_tier(65.0) == TIER_HIGH
        assert classify_tier(79.9) == TIER_HIGH

    def test_neutral(self):
        assert classify_tier(45.0) == TIER_NEUTRAL
        assert classify_tier(64.9) == TIER_NEUTRAL

    def test_exit(self):
        assert classify_tier(0.0) == TIER_EXIT
        assert classify_tier(44.9) == TIER_EXIT

    def test_nan_maps_to_exit(self):
        assert classify_tier(float("nan")) == TIER_EXIT


# ─────────────────────────────────────────────────────────────────────────────
# compute_continuation_priority — golden paths
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeContinuationPriority:
    def test_returns_result_type(self):
        r = compute_continuation_priority(_make_inp())
        assert isinstance(r, ContinuationPriorityResult)

    def test_score_in_range(self):
        r = compute_continuation_priority(_make_inp())
        assert 0.0 <= r.priority_score <= 100.0

    def test_healthy_breakout_bonus_applied(self):
        r_healthy = compute_continuation_priority(_make_inp(current_phase="healthy_breakout"))
        r_neutral = compute_continuation_priority(_make_inp(current_phase="neutral"))
        assert r_healthy.priority_score > r_neutral.priority_score

    def test_suppression_eligible_bonus(self):
        r_sup = compute_continuation_priority(_make_inp(suppression_eligible=True))
        r_no  = compute_continuation_priority(_make_inp(suppression_eligible=False))
        assert r_sup.priority_score > r_no.priority_score

    def test_mfe_retention_bonus(self):
        r_good = compute_continuation_priority(_make_inp(
            mfe_pct=MFE_RETENTION_MIN_MFE + 0.01,
            giveback_ratio=MFE_RETENTION_MAX_GB - 0.01,
        ))
        r_bad  = compute_continuation_priority(_make_inp(mfe_pct=0.01, giveback_ratio=0.5))
        assert r_good.priority_score > r_bad.priority_score

    def test_rsr_strong_bonus(self):
        r_strong = compute_continuation_priority(_make_inp(rsr=RSR_STRONG_THRESHOLD + 1))
        r_weak   = compute_continuation_priority(_make_inp(rsr=RSR_STRONG_THRESHOLD - 1))
        assert r_strong.priority_score > r_weak.priority_score

    def test_active_breakout_bonus(self):
        r_brk = compute_continuation_priority(_make_inp(current_phase="breakout"))
        r_neu = compute_continuation_priority(_make_inp(current_phase="neutral"))
        assert r_brk.priority_score > r_neu.priority_score

    def test_distribution_penalty(self):
        r_dist = compute_continuation_priority(_make_inp(current_phase="distribution"))
        r_neu  = compute_continuation_priority(_make_inp(current_phase="neutral"))
        assert r_dist.priority_score < r_neu.priority_score

    def test_failed_breakout_penalty(self):
        r_fail = compute_continuation_priority(_make_inp(current_phase="failed_breakout"))
        r_neu  = compute_continuation_priority(_make_inp(current_phase="neutral"))
        assert r_fail.priority_score < r_neu.priority_score

    def test_high_giveback_penalty(self):
        r_hgb = compute_continuation_priority(_make_inp(giveback_ratio=GIVEBACK_HIGH_THRESHOLD + 0.1))
        r_ok  = compute_continuation_priority(_make_inp(giveback_ratio=0.0))
        assert r_hgb.priority_score < r_ok.priority_score

    def test_negative_rsr_momentum_penalty(self):
        r_neg = compute_continuation_priority(_make_inp(rsr_momentum=-0.5))
        r_pos = compute_continuation_priority(_make_inp(rsr_momentum=0.5))
        assert r_neg.priority_score < r_pos.priority_score

    def test_overstay_penalty(self):
        r_old = compute_continuation_priority(_make_inp(hold_days=HOLD_OVERSTAY_AT + 5))
        r_mid = compute_continuation_priority(_make_inp(hold_days=10))
        assert r_old.priority_score < r_mid.priority_score

    def test_components_sum_to_base(self):
        r = compute_continuation_priority(_make_inp())
        comp_sum = (r.compression_component + r.bq_component +
                    r.rsr_component + r.mfe_component + r.hold_component)
        assert abs(comp_sum - r.base_score) < 0.1

    def test_score_clamped_at_100(self):
        r = compute_continuation_priority(_make_inp(
            rsr=100.0, compression_score=100.0, breakout_quality_score=100.0,
            mfe_pct=1.0, hold_days=20,
            current_phase="healthy_breakout",
            suppression_eligible=True,
            rsr_momentum=1.0, giveback_ratio=0.0,
        ))
        assert r.priority_score <= 100.0

    def test_score_not_negative(self):
        r = compute_continuation_priority(_make_inp(
            rsr=60.0, compression_score=0.0, breakout_quality_score=0.0,
            mfe_pct=0.0, hold_days=100,
            current_phase="distribution",
            giveback_ratio=0.8, rsr_momentum=-1.0,
            suppression_eligible=False,
        ))
        assert r.priority_score >= 0.0

    def test_nan_input_no_raise(self):
        r = compute_continuation_priority(_make_inp(
            rsr=float("nan"),
            compression_score=float("nan"),
        ))
        assert isinstance(r, ContinuationPriorityResult)
        assert math.isfinite(r.priority_score)

    def test_symbol_preserved(self):
        r = compute_continuation_priority(_make_inp(symbol="9999"))
        assert r.symbol == "9999"


# ─────────────────────────────────────────────────────────────────────────────
# format_priority_ranking_table
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatPriorityRankingTable:
    def _result(self, sym="1234", score=75.0, tier=TIER_HIGH) -> ContinuationPriorityResult:
        return ContinuationPriorityResult(
            symbol=sym, priority_score=score, priority_tier=tier,
            base_score=score, bonus=0.0, penalty=0.0,
            compression_component=5.0, bq_component=5.0,
            rsr_component=4.0, mfe_component=3.0, hold_component=3.0,
        )

    def test_empty_returns_empty_string(self):
        assert format_priority_ranking_table([]) == ""

    def test_contains_symbol(self):
        sig = {"symbol": "1234", "unrealized_pnl_pct": 0.05, "rsr": 82.0, "hold_days": 10}
        s = format_priority_ranking_table([(sig, self._result("1234"))])
        assert "1234" in s

    def test_contains_header(self):
        sig = {"symbol": "1234", "unrealized_pnl_pct": 0.0, "rsr": 75.0, "hold_days": 5}
        s = format_priority_ranking_table([(sig, self._result())])
        assert "CONTINUATION PRIORITY" in s

    def test_multiple_symbols(self):
        pairs = [
            ({"symbol": "A", "unrealized_pnl_pct": 0.1, "rsr": 90.0, "hold_days": 20},
             self._result("A", 85.0, TIER_CORE)),
            ({"symbol": "B", "unrealized_pnl_pct": 0.0, "rsr": 70.0, "hold_days": 5},
             self._result("B", 40.0, TIER_EXIT)),
        ]
        s = format_priority_ranking_table(pairs)
        assert "A" in s and "B" in s

    def test_fail_open_on_bad_input(self):
        result = format_priority_ranking_table([(None, None)])
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# append_cp_telemetry
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendCpTelemetry:
    def _res(self, sym="1234") -> ContinuationPriorityResult:
        return ContinuationPriorityResult(
            symbol=sym, priority_score=72.0, priority_tier=TIER_HIGH,
            base_score=70.0, bonus=5.0, penalty=3.0,
            compression_component=15.0, bq_component=17.5,
            rsr_component=14.0, mfe_component=7.5, hold_component=9.0,
        )

    def test_creates_file(self, tmp_path):
        p = tmp_path / "cp_telem.jsonl"
        append_cp_telemetry(self._res(), _make_inp(), p, date_str="2026-01-01")
        assert p.exists()

    def test_record_readable(self, tmp_path):
        p = tmp_path / "cp_telem.jsonl"
        append_cp_telemetry(self._res("9876"), _make_inp(symbol="9876"), p, date_str="2026-05-01")
        line = p.read_text(encoding="utf-8").strip()
        rec = json.loads(line)
        assert rec["symbol"] == "9876"
        assert rec["priority_tier"] == TIER_HIGH
        assert rec["subsequent_5d_return"] is None
        assert rec["materialized_5d"] is False

    def test_appends_multiple_records(self, tmp_path):
        p = tmp_path / "cp_telem.jsonl"
        for i in range(3):
            append_cp_telemetry(self._res(), _make_inp(), p, date_str=f"2026-0{i+1}-01")
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3

    def test_schema_version_present(self, tmp_path):
        p = tmp_path / "cp_telem.jsonl"
        append_cp_telemetry(self._res(), _make_inp(), p, date_str="2026-01-01")
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert rec.get("schema_version") == SCHEMA_VERSION

    def test_fail_open_bad_path(self):
        bad = Path("/nonexistent_dir_xyz/cp.jsonl")
        append_cp_telemetry(self._res(), _make_inp(), bad, date_str="2026-01-01")


# ─────────────────────────────────────────────────────────────────────────────
# materialize_cp_returns
# ─────────────────────────────────────────────────────────────────────────────

class TestMaterializeCpReturns:
    def _build_record(self, sym="1234", date_str="2025-01-01") -> dict:
        return {
            "date": date_str, "symbol": sym,
            "priority_score": 72.0, "priority_tier": TIER_HIGH,
            "base_score": 70.0, "bonus": 0.0, "penalty": 0.0,
            "compression_score": 60.0, "breakout_quality_score": 70.0,
            "rsr": 82.0, "rsr_momentum": 0.5,
            "mfe_pct": 0.05, "giveback_ratio": 0.0,
            "hold_days": 10, "current_phase": "healthy_breakout",
            "suppression_eligible": False,
            "subsequent_5d_return": None, "materialized_5d": False,
            "schema_version": SCHEMA_VERSION,
        }

    def _make_ohlcv(self):
        import pandas as pd
        import numpy as np
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        closes = np.linspace(100, 130, 60)
        return pd.DataFrame({"close": closes}, index=dates)

    def test_missing_file_returns_empty(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        r = materialize_cp_returns(p, lambda s: None)
        assert r == {}

    def test_materializes_old_record(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        # record date inside the synthetic OHLCV range (2024-01-01 + 60 business days)
        old_rec = self._build_record(date_str="2024-01-15")
        p.write_text(json.dumps(old_rec) + "\n", encoding="utf-8")

        ohlcv = self._make_ohlcv()
        r = materialize_cp_returns(p, lambda s: ohlcv)
        assert r.get("n_materialized_5d", 0) >= 1

        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        rec = json.loads(lines[0])
        assert rec["materialized_5d"] is True
        assert rec["subsequent_5d_return"] is not None

    def test_skips_recent_record(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        rec = self._build_record(date_str=today)
        p.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        r = materialize_cp_returns(p, lambda s: self._make_ohlcv())
        assert r.get("n_materialized_5d", 0) == 0

    def test_already_materialized_skipped(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        rec = self._build_record(date_str="2025-01-10")
        rec["materialized_5d"] = True
        rec["subsequent_5d_return"] = 0.05
        p.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        r = materialize_cp_returns(p, lambda s: self._make_ohlcv())
        assert r.get("n_materialized_5d", 0) == 0


# ─────────────────────────────────────────────────────────────────────────────
# compute_cp_kpis
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCpKpis:
    def _write_records(self, path, recs):
        path.write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n",
            encoding="utf-8",
        )

    def _rec(self, tier, ret=None):
        return {
            "priority_tier": tier,
            "subsequent_5d_return": ret,
            "materialized_5d": ret is not None,
        }

    def test_missing_file_returns_empty(self, tmp_path):
        r = compute_cp_kpis(tmp_path / "missing.jsonl")
        assert r == {}

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        p.write_text("", encoding="utf-8")
        assert compute_cp_kpis(p) == {}

    def test_total_count(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        self._write_records(p, [self._rec(TIER_CORE), self._rec(TIER_HIGH), self._rec(TIER_EXIT)])
        kpis = compute_cp_kpis(p)
        assert kpis["total_records"] == 3

    def test_avg_5d_by_tier(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        recs = [
            self._rec(TIER_CORE, 0.10), self._rec(TIER_CORE, 0.20),
            self._rec(TIER_EXIT, -0.05), self._rec(TIER_EXIT, -0.03),
        ]
        self._write_records(p, recs)
        kpis = compute_cp_kpis(p)
        avg_by_tier = kpis["avg_5d_return_by_priority_tier"]
        assert abs(avg_by_tier[TIER_CORE] - 0.15) < 1e-5
        assert avg_by_tier[TIER_EXIT] < 0

    def test_delta_core_vs_exit(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        recs = [self._rec(TIER_CORE, 0.10), self._rec(TIER_EXIT, -0.05)]
        self._write_records(p, recs)
        kpis = compute_cp_kpis(p)
        delta = kpis["avg_return_delta_core_vs_exit_candidate"]
        assert delta is not None
        assert delta > 0

    def test_exit_underperformance_rate(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        recs = [self._rec(TIER_EXIT, -0.05), self._rec(TIER_EXIT, 0.03), self._rec(TIER_EXIT, -0.02)]
        self._write_records(p, recs)
        kpis = compute_cp_kpis(p)
        rate = kpis["exit_candidate_underperformance_rate"]
        assert abs(rate - 2 / 3) < 1e-3

    def test_null_returns_excluded_from_avg(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        recs = [self._rec(TIER_CORE, 0.10), self._rec(TIER_CORE, None)]
        self._write_records(p, recs)
        kpis = compute_cp_kpis(p)
        assert abs(kpis["avg_5d_return_by_priority_tier"][TIER_CORE] - 0.10) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# format_cp_kpi_section
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatCpKpiSection:
    def _write_records(self, path, recs):
        path.write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n",
            encoding="utf-8",
        )

    def test_missing_file_returns_empty(self, tmp_path):
        s = format_cp_kpi_section(tmp_path / "missing.jsonl")
        assert s == ""

    def test_contains_tier_labels(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        self._write_records(p, [
            {"priority_tier": TIER_CORE, "subsequent_5d_return": 0.05, "materialized_5d": True},
            {"priority_tier": TIER_EXIT, "subsequent_5d_return": -0.02, "materialized_5d": True},
        ])
        s = format_cp_kpi_section(p)
        assert "Core" in s

    def test_fail_open_on_corrupt_file(self, tmp_path):
        p = tmp_path / "cp.jsonl"
        p.write_text("not json\n", encoding="utf-8")
        s = format_cp_kpi_section(p)
        assert isinstance(s, str)
