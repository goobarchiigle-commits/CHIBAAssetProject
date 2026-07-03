"""
test_phase5a_live_integration.py — Phase 5A live allocation integration tests

Covers the 7 required failure modes:
  1. Stale allocation state detection
  2. Missing truth state fallback
  3. Concentration cap breach throttle
  4. Fallback sizing (passes through, no block)
  5. Audit log write failure isolation
  6. requested_size vs final_size divergence
  7. Replay schema stability (round-trip serialize/deserialize)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.allocation.intent_audit import (
    AllocationIntentRecord, append_intent_record, load_intent_records,
    SCHEMA_VERSION,
)
from src.allocation.sizing import (
    SizingResult, compute_exposure_aware_size,
    gate_on_truth_confidence, detect_stale_state,
    compute_allocation_cap, apply_concentration_throttle,
    compute_concentration_penalty,
    TRUTH_CONFIDENCE_GATE, STALE_STATE_AGE_SEC, MAX_CONCENTRATION_PCT,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_record(**kwargs) -> AllocationIntentRecord:
    defaults = dict(
        timestamp="2026-05-16T09:00:00+09:00",
        symbol="7203",
        signal_id="sig_001",
        strategy_id="fujiko_v2",
        requested_size=100,
        final_size=100,
        capital_efficiency_score=0.75,
        effective_independent_n=2.5,
        portfolio_concentration=0.10,
        concentration_penalty=1.0,
        truth_confidence=0.90,
        truth_regime="allowed",
        allocation_cap=100,
        throttle_applied=False,
        throttle_reason="",
        fallback_mode=False,
        fallback_reason="",
        stale_state_detected=False,
        allocation_state_age_sec=60.0,
        execution_allowed=True,
    )
    defaults.update(kwargs)
    return AllocationIntentRecord(**defaults)


def _sizing(
    requested_size: int = 100,
    truth_confidence: float = 0.90,
    effective_independent_n: float = 2.5,
    portfolio_concentration: float = 0.10,
    capital_efficiency_score: float = 0.75,
    portfolio_equity: float = 3_000_000.0,
    estimated_price: float = 2000.0,
    state_load_time_sec: float = 0.0,
    now_sec: float = 0.0,
) -> SizingResult:
    if state_load_time_sec == 0.0 and now_sec == 0.0:
        t = time.time()
        state_load_time_sec = t - 60  # 60s old = fresh
        now_sec = t
    return compute_exposure_aware_size(
        requested_size=requested_size,
        truth_confidence=truth_confidence,
        effective_independent_n=effective_independent_n,
        portfolio_concentration=portfolio_concentration,
        capital_efficiency_score=capital_efficiency_score,
        portfolio_equity=portfolio_equity,
        estimated_price=estimated_price,
        state_load_time_sec=state_load_time_sec,
        now_sec=now_sec,
    )


# ── 1. Stale allocation state detection ───────────────────────────────────────

class TestStaleStateDetection:
    def test_fresh_state_not_stale(self):
        now = time.time()
        is_stale, age = detect_stale_state(now - 30, now)
        assert not is_stale
        assert age < STALE_STATE_AGE_SEC

    def test_old_state_is_stale(self):
        now = time.time()
        is_stale, age = detect_stale_state(now - STALE_STATE_AGE_SEC - 1, now)
        assert is_stale

    def test_zero_load_time_is_stale(self):
        is_stale, age = detect_stale_state(0.0, time.time())
        assert is_stale
        assert age == STALE_STATE_AGE_SEC

    def test_stale_state_returns_fallback_mode_but_allows_execution(self):
        now = time.time()
        stale_time = now - STALE_STATE_AGE_SEC - 100
        result = compute_exposure_aware_size(
            requested_size=100,
            truth_confidence=0.95,
            effective_independent_n=2.0,
            portfolio_concentration=0.10,
            capital_efficiency_score=0.8,
            portfolio_equity=3_000_000.0,
            estimated_price=2000.0,
            state_load_time_sec=stale_time,
            now_sec=now,
        )
        assert result.fallback_mode is True
        assert result.stale_state_detected is True
        assert result.execution_allowed is True          # stale but still allows
        assert result.final_size == result.requested_size  # original size preserved

    def test_stale_reason_contains_age(self):
        now = time.time()
        result = compute_exposure_aware_size(
            requested_size=50,
            truth_confidence=0.95,
            effective_independent_n=2.0,
            portfolio_concentration=0.10,
            capital_efficiency_score=0.8,
            portfolio_equity=3_000_000.0,
            estimated_price=1000.0,
            state_load_time_sec=now - STALE_STATE_AGE_SEC - 200,
            now_sec=now,
        )
        assert "stale_state" in result.fallback_reason


# ── 2. Missing truth state fallback ───────────────────────────────────────────

class TestMissingTruthStateFallback:
    def test_truth_confidence_below_gate_blocks(self):
        result = _sizing(truth_confidence=TRUTH_CONFIDENCE_GATE - 0.01)
        assert result.execution_allowed is False
        assert result.final_size == 0
        assert result.fallback_mode is True

    def test_truth_confidence_at_gate_passes(self):
        result = _sizing(truth_confidence=TRUTH_CONFIDENCE_GATE)
        assert result.execution_allowed is True

    def test_truth_confidence_well_above_gate_passes(self):
        result = _sizing(truth_confidence=0.95)
        assert result.execution_allowed is True

    def test_zero_truth_confidence_blocks(self):
        result = _sizing(truth_confidence=0.0)
        assert result.execution_allowed is False
        assert result.final_size == 0

    def test_gate_reason_describes_failure(self):
        result = _sizing(truth_confidence=0.50)
        assert "truth_confidence" in result.fallback_reason
        assert "gate" in result.fallback_reason


# ── 3. Concentration cap breach throttle ──────────────────────────────────────

class TestConcentrationCapBreach:
    def test_concentration_at_limit_not_throttled(self):
        qty, applied, reason = apply_concentration_throttle(100, MAX_CONCENTRATION_PCT)
        assert not applied
        assert qty == 100

    def test_concentration_above_limit_throttled(self):
        over_conc = MAX_CONCENTRATION_PCT * 2.0
        qty, applied, reason = apply_concentration_throttle(100, over_conc)
        assert applied
        assert qty < 100
        assert qty > 0
        assert "concentration" in reason

    def test_cannot_increase_qty(self):
        # Even with very low concentration, throttle never increases qty
        qty, applied, _ = apply_concentration_throttle(100, 0.01)
        assert qty <= 100

    def test_concentration_exactly_double_limit_halves_qty(self):
        over_conc = MAX_CONCENTRATION_PCT * 2.0
        qty, applied, _ = apply_concentration_throttle(100, over_conc)
        assert applied
        assert qty == 50  # 0.25/0.50 = 0.5 → 100 * 0.5 = 50

    def test_sizing_applies_throttle_on_breach(self):
        portfolio_equity = 1_000_000.0
        estimated_price = 10_000.0
        # order_value = 100 * 10000 = 1_000_000 → concentration = 1.0 (100%)
        result = _sizing(
            requested_size=100,
            truth_confidence=0.95,
            portfolio_concentration=1.0,  # massively over-concentrated
            portfolio_equity=portfolio_equity,
            estimated_price=estimated_price,
        )
        assert result.throttle_applied is True
        assert result.final_size < 100


# ── 4. Fallback sizing ────────────────────────────────────────────────────────

class TestFallbackSizing:
    def test_stale_fallback_passes_original_size(self):
        now = time.time()
        result = compute_exposure_aware_size(
            requested_size=200,
            truth_confidence=0.95,
            effective_independent_n=2.5,
            portfolio_concentration=0.10,
            capital_efficiency_score=0.7,
            portfolio_equity=3_000_000.0,
            estimated_price=1500.0,
            state_load_time_sec=0.0,   # triggers stale
            now_sec=now,
        )
        assert result.fallback_mode is True
        assert result.final_size == 200
        assert result.execution_allowed is True

    def test_low_effective_n_applies_cap(self):
        # effective_n < MIN_EFFECTIVE_N_FULL → 80% cap
        result = _sizing(
            requested_size=100,
            truth_confidence=0.95,
            effective_independent_n=1.0,   # < 2.0 threshold
            portfolio_concentration=0.05,
            portfolio_equity=10_000_000.0,
            estimated_price=100.0,         # small price, conc cap won't bite
        )
        # Should get 80% cap from N
        assert result.allocation_cap <= 80
        assert result.final_size <= 80

    def test_normal_path_no_throttle_no_fallback(self):
        result = _sizing(
            requested_size=50,
            truth_confidence=0.95,
            effective_independent_n=3.0,
            portfolio_concentration=0.05,
            portfolio_equity=10_000_000.0,
            estimated_price=1000.0,
        )
        assert not result.fallback_mode
        assert not result.throttle_applied
        assert result.final_size == 50
        assert result.execution_allowed is True


# ── 5. Audit log write failure isolation ──────────────────────────────────────

class TestAuditLogWriteFailureIsolation:
    def test_write_to_nonexistent_parent_creates_dir(self, tmp_path):
        record = _make_record()
        deep = tmp_path / "a" / "b" / "c" / "audit.jsonl"
        append_intent_record(record, deep)   # must not raise
        assert deep.exists()
        records = load_intent_records(deep)
        assert len(records) == 1

    def test_write_failure_does_not_raise(self, tmp_path):
        # Write to a directory path (will fail I/O) — must not raise
        record = _make_record()
        dir_path = tmp_path / "dir_as_file"
        dir_path.mkdir()
        # append_intent_record swallows the error
        append_intent_record(record, dir_path)  # no exception

    def test_audit_write_does_not_affect_execution_logic(self, tmp_path):
        record = _make_record(execution_allowed=True)
        bad_path = tmp_path / "readonly_dir"
        bad_path.mkdir()
        # Even if write fails, execution_allowed is determined by sizing, not audit
        append_intent_record(record, bad_path)
        assert record.execution_allowed is True  # unchanged

    def test_empty_file_returns_empty_list(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.touch()
        assert load_intent_records(empty) == []

    def test_corrupt_line_skipped(self, tmp_path):
        p = tmp_path / "audit.jsonl"
        record = _make_record()
        append_intent_record(record, p)
        with p.open("a") as f:
            f.write("{ invalid json }\n")
        append_intent_record(_make_record(symbol="8035"), p)
        records = load_intent_records(p)
        assert len(records) == 2  # corrupt line skipped


# ── 6. requested_size vs final_size divergence ────────────────────────────────

class TestSizeDivergence:
    def test_truth_gate_produces_zero_final(self):
        result = _sizing(truth_confidence=0.50)
        assert result.requested_size == 100
        assert result.final_size == 0
        assert result.requested_size != result.final_size

    def test_throttle_reduces_final_below_requested(self):
        result = _sizing(
            requested_size=100,
            truth_confidence=0.95,
            portfolio_concentration=MAX_CONCENTRATION_PCT * 3,
        )
        assert result.final_size < result.requested_size

    def test_intent_record_captures_both_sizes(self, tmp_path):
        result = _sizing(
            requested_size=100,
            truth_confidence=0.95,
            portfolio_concentration=MAX_CONCENTRATION_PCT * 3,
        )
        record = _make_record(
            requested_size=result.requested_size,
            final_size=result.final_size,
            throttle_applied=result.throttle_applied,
        )
        p = tmp_path / "audit.jsonl"
        append_intent_record(record, p)
        loaded = load_intent_records(p)
        assert loaded[0].requested_size == 100
        assert loaded[0].final_size < 100
        assert loaded[0].throttle_applied is True

    def test_no_divergence_when_no_constraints_active(self):
        result = _sizing(
            requested_size=10,
            truth_confidence=1.0,
            effective_independent_n=5.0,
            portfolio_concentration=0.01,
            portfolio_equity=50_000_000.0,
            estimated_price=100.0,
        )
        assert result.final_size == result.requested_size


# ── 7. Replay schema stability ────────────────────────────────────────────────

class TestReplaySchemaStability:
    def test_round_trip_preserve_all_fields(self, tmp_path):
        record = _make_record(
            symbol="6758",
            requested_size=200,
            final_size=150,
            throttle_applied=True,
            throttle_reason="concentration=0.30 > max=0.25",
            fallback_mode=False,
            stale_state_detected=True,
            allocation_state_age_sec=120.5,
            execution_allowed=True,
        )
        p = tmp_path / "audit.jsonl"
        append_intent_record(record, p)
        loaded = load_intent_records(p)[0]

        assert loaded.symbol == record.symbol
        assert loaded.requested_size == record.requested_size
        assert loaded.final_size == record.final_size
        assert loaded.throttle_applied == record.throttle_applied
        assert loaded.throttle_reason == record.throttle_reason
        assert loaded.fallback_mode == record.fallback_mode
        assert loaded.stale_state_detected == record.stale_state_detected
        assert loaded.allocation_state_age_sec == record.allocation_state_age_sec
        assert loaded.execution_allowed == record.execution_allowed
        assert loaded.schema_version == SCHEMA_VERSION

    def test_sort_keys_produces_deterministic_json(self, tmp_path):
        record = _make_record()
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        append_intent_record(record, p1)
        append_intent_record(record, p2)
        assert p1.read_text() == p2.read_text()

    def test_missing_optional_fields_deserialize_safely(self, tmp_path):
        # Write minimal JSON (missing some fields)
        p = tmp_path / "partial.jsonl"
        minimal = {"timestamp": "2026-05-16T09:00:00", "symbol": "1234"}
        with p.open("w") as f:
            f.write(json.dumps(minimal) + "\n")
        records = load_intent_records(p)
        assert len(records) == 1
        assert records[0].symbol == "1234"
        assert records[0].requested_size == 0
        assert records[0].execution_allowed is True   # default

    def test_multiple_records_append_and_load_in_order(self, tmp_path):
        p = tmp_path / "audit.jsonl"
        syms = ["7203", "8035", "6758"]
        for sym in syms:
            append_intent_record(_make_record(symbol=sym), p)
        loaded = load_intent_records(p)
        assert [r.symbol for r in loaded] == syms


# ── Allocation cap ─────────────────────────────────────────────────────────────

class TestAllocationCap:
    def test_cap_never_exceeds_requested(self):
        cap = compute_allocation_cap(
            requested_size=100,
            effective_independent_n=5.0,
            portfolio_equity=3_000_000.0,
            estimated_price=5000.0,
        )
        assert cap <= 100

    def test_zero_equity_returns_requested(self):
        cap = compute_allocation_cap(100, 3.0, 0.0, 2000.0)
        assert cap == 100

    def test_zero_price_returns_requested(self):
        cap = compute_allocation_cap(100, 3.0, 3_000_000.0, 0.0)
        assert cap == 100

    def test_concentration_cap_limits_value(self):
        # MAX 25% of 1M = 250K. Price=10K → cap = 25 shares
        cap = compute_allocation_cap(
            requested_size=100,
            effective_independent_n=5.0,
            portfolio_equity=1_000_000.0,
            estimated_price=10_000.0,
        )
        assert cap == 25


# ── Concentration penalty ──────────────────────────────────────────────────────

class TestConcentrationPenalty:
    def test_at_zero_returns_one(self):
        assert compute_concentration_penalty(0.0) == 1.0

    def test_at_limit_returns_one(self):
        assert compute_concentration_penalty(MAX_CONCENTRATION_PCT) == 1.0

    def test_above_limit_returns_less_than_one(self):
        penalty = compute_concentration_penalty(MAX_CONCENTRATION_PCT * 2)
        assert 0.0 < penalty < 1.0

    def test_penalty_bounded_at_one(self):
        assert compute_concentration_penalty(0.001) <= 1.0
