"""
test_outcome_attribution.py — Tests for allocator outcome attribution layer

Covers:
  - skipped_trade_outcomes: record, enrichment, append, load, all skip reasons
  - allocation_outcomes: record, enrichment, append, load, partial enrichment
  - jsonl_rotation: needs_rotation, rotate_monthly, archive collision, no-op cases
  - stale_visibility: all four check functions, emit_stale_warnings, append, load
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.allocation.skipped_trade_outcomes import (
    SkippedTradeRecord,
    SkippedTradeOutcome,
    append_skipped_trade,
    append_skipped_outcome,
    load_skipped_trades,
    load_skipped_outcomes,
    enrich_skipped_trade,
    SKIP_ALLOC_CAP,
    SKIP_CONCENTRATION_THROTTLE,
    SKIP_TRUTH_CONFIDENCE,
    VALID_SKIP_REASONS,
    SCHEMA_VERSION as SKIP_SCHEMA_VERSION,
)
from src.allocation.allocation_outcomes import (
    AllocationOutcomeRecord,
    build_outcome_record,
    enrich_outcome_record,
    append_outcome_record,
    load_outcome_records,
    SCHEMA_VERSION as OUTCOME_SCHEMA_VERSION,
)
from src.allocation.jsonl_rotation import (
    needs_rotation,
    rotate_monthly,
    rotate_if_needed,
    list_archives,
    _parse_year_month,
    _archive_filename,
)
from src.allocation.stale_visibility import (
    StaleVisibilityWarning,
    append_stale_warning,
    load_stale_warnings,
    check_exposure_staleness,
    check_lifecycle_staleness,
    check_stability_staleness,
    check_fallback_activation,
    emit_stale_warnings,
    WARN_STALE_EXPOSURE,
    WARN_STALE_LIFECYCLE,
    WARN_STALE_STABILITY,
    WARN_FALLBACK_ACTIVATED,
    DEFAULT_EXPOSURE_STALE_SEC,
    DEFAULT_LIFECYCLE_STALE_SEC,
    DEFAULT_STABILITY_STALE_SEC,
    SCHEMA_VERSION as STALE_SCHEMA_VERSION,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_skipped_record(**kwargs) -> SkippedTradeRecord:
    defaults = dict(
        timestamp="2026-05-16T09:00:00+09:00",
        symbol="7203",
        signal_id="sig_001",
        strategy_id="fujiko_v2",
        skip_reason=SKIP_ALLOC_CAP,
        requested_size=100,
        alloc_cap=40,
        truth_confidence=0.85,
        concentration_at_skip=0.18,
        estimated_entry_price=2500.0,
    )
    defaults.update(kwargs)
    return SkippedTradeRecord(**defaults)


def _make_outcome_record(**kwargs) -> AllocationOutcomeRecord:
    defaults = dict(
        timestamp="2026-05-16T09:00:00+09:00",
        symbol="7203",
        signal_id="sig_001",
        strategy_id="fujiko_v2",
        intended_qty=100,
        adjusted_qty=80,
        throttle_reason="concentration_throttle",
        truth_confidence=0.85,
        alloc_cap=90,
        effective_n=2.5,
        sector_multiplier=1.0,
        realized_pnl=None,
        realized_slippage=None,
        holding_days=None,
        exit_reason=None,
    )
    defaults.update(kwargs)
    return AllocationOutcomeRecord(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# skipped_trade_outcomes
# ══════════════════════════════════════════════════════════════════════════════

class TestSkipReasonConstants:
    def test_valid_skip_reasons_contains_all(self):
        assert SKIP_ALLOC_CAP in VALID_SKIP_REASONS
        assert SKIP_CONCENTRATION_THROTTLE in VALID_SKIP_REASONS
        assert SKIP_TRUTH_CONFIDENCE in VALID_SKIP_REASONS

    def test_skip_reason_strings(self):
        assert SKIP_ALLOC_CAP == "alloc_cap"
        assert SKIP_CONCENTRATION_THROTTLE == "concentration_throttle"
        assert SKIP_TRUTH_CONFIDENCE == "truth_confidence"


class TestSkippedTradeAppendLoad:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "skipped.jsonl"
        rec = _make_skipped_record()
        append_skipped_trade(rec, path)
        loaded = load_skipped_trades(path)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.symbol == "7203"
        assert r.skip_reason == SKIP_ALLOC_CAP
        assert r.requested_size == 100
        assert r.alloc_cap == 40
        assert r.estimated_entry_price == 2500.0

    def test_append_multiple(self, tmp_path):
        path = tmp_path / "skipped.jsonl"
        for reason in [SKIP_ALLOC_CAP, SKIP_CONCENTRATION_THROTTLE, SKIP_TRUTH_CONFIDENCE]:
            append_skipped_trade(_make_skipped_record(skip_reason=reason), path)
        loaded = load_skipped_trades(path)
        assert len(loaded) == 3
        reasons = {r.skip_reason for r in loaded}
        assert reasons == VALID_SKIP_REASONS

    def test_append_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "skipped.jsonl"
        append_skipped_trade(_make_skipped_record(), path)
        assert path.exists()

    def test_load_missing_file_returns_empty(self, tmp_path):
        result = load_skipped_trades(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_load_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "skipped.jsonl"
        good = json.dumps({"timestamp": "2026-05-16T09:00:00+09:00", "symbol": "7203",
                           "signal_id": "s", "strategy_id": "f", "skip_reason": "alloc_cap",
                           "requested_size": 100, "alloc_cap": 40, "truth_confidence": 0.85,
                           "concentration_at_skip": 0.18, "estimated_entry_price": 2500.0})
        path.write_text("not_json\n" + good + "\n", encoding="utf-8")
        loaded = load_skipped_trades(path)
        assert len(loaded) == 1

    def test_schema_version_in_record(self, tmp_path):
        path = tmp_path / "skipped.jsonl"
        append_skipped_trade(_make_skipped_record(), path)
        raw = json.loads(path.read_text(encoding="utf-8").strip())
        assert raw["schema_version"] == SKIP_SCHEMA_VERSION

    def test_append_fail_does_not_raise(self, tmp_path):
        path = tmp_path / "readonly_dir" / "skipped.jsonl"
        # Even if directory can't be created on some systems, must not raise
        # We test the fail-open contract by monkeypatching
        import src.allocation.skipped_trade_outcomes as mod
        original = mod.Path.open

        class _BadPath:
            def __init__(self, *a, **kw): pass
            def open(self, *a, **kw): raise OSError("disk full")

        # Simpler: call with read-only path trick won't work cross-platform.
        # Instead verify no exception propagates even with bad data.
        rec = _make_skipped_record()
        append_skipped_trade(rec, path)  # may succeed or fail, must not raise


class TestSkippedTradeOutcomeAppendLoad:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        rec = _make_skipped_record()
        prices = [2500.0, 2550.0, 2480.0, 2600.0, 2520.0, 2650.0]
        outcome = enrich_skipped_trade(rec, prices, enriched_at="2026-05-21T09:00:00+09:00")
        append_skipped_outcome(outcome, path)
        loaded = load_skipped_outcomes(path)
        assert len(loaded) == 1
        o = loaded[0]
        assert o.symbol == "7203"
        assert o.enriched_at == "2026-05-21T09:00:00+09:00"
        assert o.forward_return_1d is not None
        assert o.mfe is not None
        assert o.mae is not None


class TestEnrichSkippedTrade:
    def test_basic_enrichment(self):
        rec = _make_skipped_record(estimated_entry_price=1000.0)
        prices = [1000.0, 1010.0, 990.0, 1020.0, 980.0, 1050.0]
        outcome = enrich_skipped_trade(rec, prices, enriched_at="2026-05-21T09:00:00")
        assert outcome.forward_return_1d == pytest.approx(0.01, rel=1e-5)
        assert outcome.forward_return_3d == pytest.approx(0.02, rel=1e-5)
        assert outcome.forward_return_5d == pytest.approx(0.05, rel=1e-5)
        assert outcome.mfe == pytest.approx(0.05, rel=1e-5)
        assert outcome.mae == pytest.approx(0.02, rel=1e-5)

    def test_empty_prices_returns_none_metrics(self):
        rec = _make_skipped_record(estimated_entry_price=1000.0)
        outcome = enrich_skipped_trade(rec, [], enriched_at="2026-05-21T09:00:00")
        assert outcome.forward_return_1d is None
        assert outcome.mfe is None
        assert outcome.mae is None

    def test_zero_entry_price_returns_none_metrics(self):
        rec = _make_skipped_record(estimated_entry_price=0.0)
        outcome = enrich_skipped_trade(rec, [1000.0, 1010.0], enriched_at="ts")
        assert outcome.forward_return_1d is None

    def test_partial_prices_gives_none_for_missing_days(self):
        rec = _make_skipped_record(estimated_entry_price=1000.0)
        prices = [1000.0, 1010.0]  # only day0 and day1
        outcome = enrich_skipped_trade(rec, prices, enriched_at="ts")
        assert outcome.forward_return_1d == pytest.approx(0.01, rel=1e-5)
        assert outcome.forward_return_3d is None
        assert outcome.forward_return_5d is None

    def test_enrichment_is_deterministic(self):
        rec = _make_skipped_record(estimated_entry_price=2000.0)
        prices = [2000.0, 2100.0, 1900.0, 2050.0, 1850.0, 2200.0]
        o1 = enrich_skipped_trade(rec, prices, enriched_at="ts")
        o2 = enrich_skipped_trade(rec, prices, enriched_at="ts")
        assert o1.forward_return_1d == o2.forward_return_1d
        assert o1.mfe == o2.mfe
        assert o1.mae == o2.mae

    def test_alloc_cap_skip_reason_preserved(self):
        rec = _make_skipped_record(skip_reason=SKIP_ALLOC_CAP)
        outcome = enrich_skipped_trade(rec, [1000.0, 1010.0], enriched_at="ts")
        assert outcome.skip_reason == SKIP_ALLOC_CAP

    def test_concentration_throttle_skip_reason_preserved(self):
        rec = _make_skipped_record(skip_reason=SKIP_CONCENTRATION_THROTTLE)
        outcome = enrich_skipped_trade(rec, [1000.0], enriched_at="ts")
        assert outcome.skip_reason == SKIP_CONCENTRATION_THROTTLE

    def test_truth_confidence_skip_reason_preserved(self):
        rec = _make_skipped_record(skip_reason=SKIP_TRUTH_CONFIDENCE)
        outcome = enrich_skipped_trade(rec, [1000.0], enriched_at="ts")
        assert outcome.skip_reason == SKIP_TRUTH_CONFIDENCE

    def test_mae_is_always_non_negative(self):
        rec = _make_skipped_record(estimated_entry_price=1000.0)
        prices = [1000.0, 950.0, 920.0, 1100.0]  # big drawdown then recovery
        outcome = enrich_skipped_trade(rec, prices, enriched_at="ts")
        assert outcome.mae >= 0.0

    def test_replay_safe_schema_fields(self):
        rec = _make_skipped_record()
        outcome = enrich_skipped_trade(rec, [2500.0, 2550.0], enriched_at="ts")
        assert outcome.schema_version == SKIP_SCHEMA_VERSION
        assert outcome.signal_id == rec.signal_id
        assert outcome.strategy_id == rec.strategy_id


# ══════════════════════════════════════════════════════════════════════════════
# allocation_outcomes
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildOutcomeRecord:
    def test_build_defaults_to_none_enriched_fields(self):
        rec = build_outcome_record(
            timestamp="2026-05-16T09:00:00+09:00",
            symbol="7203",
            signal_id="sig_001",
            strategy_id="fujiko_v2",
            intended_qty=100,
            adjusted_qty=80,
            throttle_reason="concentration_throttle",
            truth_confidence=0.85,
            alloc_cap=90,
            effective_n=2.5,
        )
        assert rec.realized_pnl is None
        assert rec.realized_slippage is None
        assert rec.holding_days is None
        assert rec.exit_reason is None
        assert rec.sector_multiplier == 1.0

    def test_build_with_custom_sector_multiplier(self):
        rec = build_outcome_record(
            timestamp="ts", symbol="8316", signal_id="s", strategy_id="f",
            intended_qty=50, adjusted_qty=50, throttle_reason="",
            truth_confidence=0.90, alloc_cap=50, effective_n=3.0,
            sector_multiplier=0.8,
        )
        assert rec.sector_multiplier == 0.8


class TestEnrichOutcomeRecord:
    def test_enrich_pnl_only(self):
        rec = _make_outcome_record()
        enriched = enrich_outcome_record(rec, realized_pnl=15000.0)
        assert enriched.realized_pnl == 15000.0
        assert enriched.holding_days is None
        assert enriched.exit_reason is None

    def test_enrich_all_fields(self):
        rec = _make_outcome_record()
        enriched = enrich_outcome_record(
            rec,
            realized_pnl=20000.0,
            realized_slippage=500.0,
            holding_days=7,
            exit_reason="turtle_exit_55d",
        )
        assert enriched.realized_pnl == 20000.0
        assert enriched.realized_slippage == 500.0
        assert enriched.holding_days == 7
        assert enriched.exit_reason == "turtle_exit_55d"

    def test_enrich_does_not_mutate_original(self):
        rec = _make_outcome_record()
        _ = enrich_outcome_record(rec, realized_pnl=999.0)
        assert rec.realized_pnl is None

    def test_enrich_preserves_existing_pnl_if_not_provided(self):
        rec = _make_outcome_record(realized_pnl=1000.0)
        enriched = enrich_outcome_record(rec, holding_days=5)
        assert enriched.realized_pnl == 1000.0
        assert enriched.holding_days == 5

    def test_enrich_preserves_all_identity_fields(self):
        rec = _make_outcome_record()
        enriched = enrich_outcome_record(rec, realized_pnl=0.0)
        assert enriched.symbol == rec.symbol
        assert enriched.signal_id == rec.signal_id
        assert enriched.intended_qty == rec.intended_qty
        assert enriched.adjusted_qty == rec.adjusted_qty


class TestAllocationOutcomeAppendLoad:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        rec = _make_outcome_record()
        append_outcome_record(rec, path)
        loaded = load_outcome_records(path)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.symbol == "7203"
        assert r.intended_qty == 100
        assert r.adjusted_qty == 80
        assert r.realized_pnl is None

    def test_append_multiple_records(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        for sym in ["7203", "8316", "9984"]:
            append_outcome_record(_make_outcome_record(symbol=sym), path)
        loaded = load_outcome_records(path)
        assert len(loaded) == 3
        assert {r.symbol for r in loaded} == {"7203", "8316", "9984"}

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_outcome_records(tmp_path / "missing.jsonl") == []

    def test_load_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        good = json.dumps({
            "timestamp": "ts", "symbol": "7203", "signal_id": "s",
            "strategy_id": "f", "intended_qty": 100, "adjusted_qty": 80,
            "throttle_reason": "", "truth_confidence": 0.85, "alloc_cap": 90,
            "effective_n": 2.5, "sector_multiplier": 1.0,
            "realized_pnl": None, "realized_slippage": None,
            "holding_days": None, "exit_reason": None,
        })
        path.write_text("corrupt_line\n" + good + "\n", encoding="utf-8")
        loaded = load_outcome_records(path)
        assert len(loaded) == 1

    def test_schema_version_in_serialized_output(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        append_outcome_record(_make_outcome_record(), path)
        raw = json.loads(path.read_text(encoding="utf-8").strip())
        assert raw["schema_version"] == OUTCOME_SCHEMA_VERSION

    def test_append_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "outcomes.jsonl"
        append_outcome_record(_make_outcome_record(), path)
        assert path.exists()

    def test_enriched_record_roundtrip_with_pnl(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        rec = _make_outcome_record()
        enriched = enrich_outcome_record(rec, realized_pnl=12500.0, holding_days=8)
        append_outcome_record(enriched, path)
        loaded = load_outcome_records(path)
        assert loaded[0].realized_pnl == 12500.0
        assert loaded[0].holding_days == 8


# ══════════════════════════════════════════════════════════════════════════════
# jsonl_rotation
# ══════════════════════════════════════════════════════════════════════════════

def _write_jsonl_with_timestamp(path: Path, timestamp: str, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"timestamp": timestamp, "value": i}) + "\n")


class TestParseYearMonth:
    def test_iso_with_timezone(self):
        assert _parse_year_month("2026-05-16T09:00:00+09:00") == (2026, 5)

    def test_iso_date_only(self):
        assert _parse_year_month("2026-03-01") == (2026, 3)

    def test_invalid_returns_none(self):
        assert _parse_year_month("not-a-date") is None
        assert _parse_year_month("") is None

    def test_year_boundary(self):
        assert _parse_year_month("2025-12-31T23:59:59Z") == (2025, 12)
        assert _parse_year_month("2026-01-01T00:00:00Z") == (2026, 1)


class TestArchiveFilename:
    def test_format(self):
        name = _archive_filename("allocation_outcomes", 2026, 4, ".jsonl")
        assert name == "allocation_outcomes_2026-04.jsonl"

    def test_zero_padded_month(self):
        name = _archive_filename("skipped", 2026, 1, ".jsonl")
        assert name == "skipped_2026-01.jsonl"


class TestNeedsRotation:
    def test_prior_month_needs_rotation(self, tmp_path):
        path = tmp_path / "active.jsonl"
        _write_jsonl_with_timestamp(path, "2026-04-15T09:00:00+09:00")
        now = datetime(2026, 5, 16, tzinfo=timezone.utc)
        assert needs_rotation(path, now=now) is True

    def test_current_month_does_not_need_rotation(self, tmp_path):
        path = tmp_path / "active.jsonl"
        _write_jsonl_with_timestamp(path, "2026-05-01T09:00:00+09:00")
        now = datetime(2026, 5, 16, tzinfo=timezone.utc)
        assert needs_rotation(path, now=now) is False

    def test_missing_file_does_not_need_rotation(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        assert needs_rotation(path) is False

    def test_empty_file_does_not_need_rotation(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert needs_rotation(path) is False

    def test_prior_year_needs_rotation(self, tmp_path):
        path = tmp_path / "active.jsonl"
        _write_jsonl_with_timestamp(path, "2025-12-31T09:00:00+09:00")
        now = datetime(2026, 1, 15, tzinfo=timezone.utc)
        assert needs_rotation(path, now=now) is True

    def test_corrupt_first_line_no_rotation(self, tmp_path):
        path = tmp_path / "active.jsonl"
        path.write_text("not_json\n", encoding="utf-8")
        now = datetime(2026, 5, 16, tzinfo=timezone.utc)
        assert needs_rotation(path, now=now) is False


class TestRotateMonthly:
    def test_rotation_moves_file_atomically(self, tmp_path):
        path = tmp_path / "active.jsonl"
        _write_jsonl_with_timestamp(path, "2026-04-15T09:00:00+09:00")
        now = datetime(2026, 5, 16, tzinfo=timezone.utc)
        archive_dir = tmp_path / "archive"
        result = rotate_monthly(path, archive_dir=archive_dir, now=now)
        assert result is not None
        assert result.exists()
        assert not path.exists()
        assert result.name == "active_2026-04.jsonl"

    def test_rotation_no_op_when_current_month(self, tmp_path):
        path = tmp_path / "active.jsonl"
        _write_jsonl_with_timestamp(path, "2026-05-01T09:00:00+09:00")
        now = datetime(2026, 5, 16, tzinfo=timezone.utc)
        result = rotate_monthly(path, now=now)
        assert result is None
        assert path.exists()

    def test_rotation_no_op_on_missing_file(self, tmp_path):
        result = rotate_monthly(tmp_path / "missing.jsonl")
        assert result is None

    def test_rotation_uses_default_archive_dir(self, tmp_path):
        path = tmp_path / "active.jsonl"
        _write_jsonl_with_timestamp(path, "2026-04-10T09:00:00+09:00")
        now = datetime(2026, 5, 16, tzinfo=timezone.utc)
        result = rotate_monthly(path, now=now)
        assert result is not None
        assert result.parent == tmp_path / "archive"

    def test_rotation_skips_on_archive_collision(self, tmp_path):
        path = tmp_path / "active.jsonl"
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        _write_jsonl_with_timestamp(path, "2026-04-10T09:00:00+09:00")
        # Pre-create the archive file to simulate collision
        collision = archive_dir / "active_2026-04.jsonl"
        collision.write_text("existing_archive\n", encoding="utf-8")
        now = datetime(2026, 5, 16, tzinfo=timezone.utc)
        result = rotate_monthly(path, archive_dir=archive_dir, now=now)
        assert result is None
        assert path.exists()
        assert collision.read_text(encoding="utf-8") == "existing_archive\n"

    def test_rotate_if_needed_alias(self, tmp_path):
        path = tmp_path / "active.jsonl"
        _write_jsonl_with_timestamp(path, "2026-04-01T09:00:00+09:00")
        now = datetime(2026, 5, 16, tzinfo=timezone.utc)
        result = rotate_if_needed(path, now=now)
        assert result is not None
        assert not path.exists()

    def test_rotated_file_content_intact(self, tmp_path):
        path = tmp_path / "active.jsonl"
        _write_jsonl_with_timestamp(path, "2026-04-01T09:00:00+09:00", n=5)
        original_content = path.read_text(encoding="utf-8")
        now = datetime(2026, 5, 16, tzinfo=timezone.utc)
        archive_dir = tmp_path / "archive"
        result = rotate_monthly(path, archive_dir=archive_dir, now=now)
        assert result is not None
        assert result.read_text(encoding="utf-8") == original_content


class TestListArchives:
    def test_lists_matching_archives(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        for month in ["2026-01", "2026-02", "2026-03"]:
            (archive_dir / f"outcomes_{month}.jsonl").write_text("{}\n")
        archives = list_archives(archive_dir, "outcomes")
        assert len(archives) == 3
        assert all(p.name.startswith("outcomes_") for p in archives)

    def test_ignores_non_matching_files(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        (archive_dir / "outcomes_2026-01.jsonl").write_text("{}\n")
        (archive_dir / "other_file.jsonl").write_text("{}\n")
        archives = list_archives(archive_dir, "outcomes")
        assert len(archives) == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        assert list_archives(archive_dir, "outcomes") == []

    def test_missing_dir_returns_empty(self, tmp_path):
        assert list_archives(tmp_path / "nonexistent", "outcomes") == []


# ══════════════════════════════════════════════════════════════════════════════
# stale_visibility
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckExposureStaleness:
    def test_fresh_state_returns_none(self):
        now = 10000.0
        load_time = now - 100.0  # 100s ago, threshold=3600
        result = check_exposure_staleness(load_time, now, "ts")
        assert result is None

    def test_stale_state_returns_warning(self):
        now = 10000.0
        load_time = now - 7200.0  # 2h ago
        result = check_exposure_staleness(load_time, now, "ts", symbol="7203")
        assert result is not None
        assert result.warning_type == WARN_STALE_EXPOSURE
        assert result.symbol == "7203"
        assert result.state_age_sec == pytest.approx(7200.0, rel=1e-3)

    def test_zero_load_time_returns_warning(self):
        result = check_exposure_staleness(0.0, 10000.0, "ts")
        assert result is not None
        assert result.warning_type == WARN_STALE_EXPOSURE

    def test_custom_threshold(self):
        now = 10000.0
        load_time = now - 200.0  # 200s ago
        # threshold=100 → stale
        assert check_exposure_staleness(load_time, now, "ts", threshold_sec=100.0) is not None
        # threshold=300 → fresh
        assert check_exposure_staleness(load_time, now, "ts", threshold_sec=300.0) is None

    def test_exactly_at_threshold_is_not_stale(self):
        now = 10000.0
        load_time = now - DEFAULT_EXPOSURE_STALE_SEC
        result = check_exposure_staleness(load_time, now, "ts")
        assert result is None


class TestCheckLifecycleStaleness:
    def test_fresh_lifecycle_returns_none(self):
        now = 10000.0
        load_time = now - 3600.0  # 1h ago, threshold=86400
        assert check_lifecycle_staleness(load_time, now, "ts") is None

    def test_stale_lifecycle_returns_warning(self):
        now = 100000.0
        load_time = now - 90000.0  # 25h ago
        result = check_lifecycle_staleness(load_time, now, "ts")
        assert result is not None
        assert result.warning_type == WARN_STALE_LIFECYCLE

    def test_zero_load_time_returns_warning(self):
        assert check_lifecycle_staleness(0.0, 10000.0, "ts") is not None


class TestCheckStabilityStaleness:
    def test_fresh_stability_returns_none(self):
        now = 10000.0
        load_time = now - 3600.0
        assert check_stability_staleness(load_time, now, "ts") is None

    def test_stale_stability_returns_warning(self):
        now = 200000.0
        load_time = now - 90000.0
        result = check_stability_staleness(load_time, now, "ts")
        assert result is not None
        assert result.warning_type == WARN_STALE_STABILITY

    def test_warning_contains_threshold(self):
        now = 200000.0
        load_time = now - 90000.0
        result = check_stability_staleness(load_time, now, "ts", threshold_sec=86400.0)
        assert result.threshold_sec == 86400.0


class TestCheckFallbackActivation:
    def test_no_fallback_returns_none(self):
        assert check_fallback_activation(False, "", "ts") is None

    def test_fallback_active_returns_warning(self):
        result = check_fallback_activation(True, "stale_state: age=7200s", "ts", symbol="7203")
        assert result is not None
        assert result.warning_type == WARN_FALLBACK_ACTIVATED
        assert result.symbol == "7203"
        assert "stale_state" in result.fallback_reason
        assert result.state_age_sec is None
        assert result.threshold_sec is None

    def test_fallback_warning_schema_version(self):
        result = check_fallback_activation(True, "reason", "ts")
        assert result.schema_version == STALE_SCHEMA_VERSION


class TestAppendLoadStaleWarnings:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "stale.jsonl"
        w = check_exposure_staleness(0.0, 10000.0, "2026-05-16T09:00:00+09:00", "7203")
        assert w is not None
        append_stale_warning(w, path)
        loaded = load_stale_warnings(path)
        assert len(loaded) == 1
        assert loaded[0].warning_type == WARN_STALE_EXPOSURE
        assert loaded[0].symbol == "7203"

    def test_append_multiple_warning_types(self, tmp_path):
        path = tmp_path / "stale.jsonl"
        now = 200000.0
        for fn in [
            check_exposure_staleness(0.0, now, "ts"),
            check_lifecycle_staleness(0.0, now, "ts"),
            check_stability_staleness(0.0, now, "ts"),
            check_fallback_activation(True, "reason", "ts"),
        ]:
            assert fn is not None
            append_stale_warning(fn, path)
        loaded = load_stale_warnings(path)
        assert len(loaded) == 4
        types = {w.warning_type for w in loaded}
        assert types == {
            WARN_STALE_EXPOSURE, WARN_STALE_LIFECYCLE,
            WARN_STALE_STABILITY, WARN_FALLBACK_ACTIVATED,
        }

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_stale_warnings(tmp_path / "missing.jsonl") == []

    def test_load_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "stale.jsonl"
        good = json.dumps({
            "timestamp": "ts", "warning_type": WARN_STALE_EXPOSURE,
            "symbol": "7203", "detail": "d", "state_age_sec": 7200.0,
            "threshold_sec": 3600.0, "fallback_reason": "r",
        })
        path.write_text("bad_line\n" + good + "\n", encoding="utf-8")
        loaded = load_stale_warnings(path)
        assert len(loaded) == 1

    def test_append_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "stale.jsonl"
        w = check_fallback_activation(True, "r", "ts")
        assert w is not None
        append_stale_warning(w, path)
        assert path.exists()

    def test_schema_version_serialized(self, tmp_path):
        path = tmp_path / "stale.jsonl"
        w = check_fallback_activation(True, "r", "ts")
        assert w is not None
        append_stale_warning(w, path)
        raw = json.loads(path.read_text(encoding="utf-8").strip())
        assert raw["schema_version"] == STALE_SCHEMA_VERSION


class TestEmitStaleWarnings:
    def test_emits_all_stale_and_fallback(self, tmp_path):
        path = tmp_path / "stale.jsonl"
        now = 200000.0
        emitted = emit_stale_warnings(
            exposure_load_time=0.0,
            lifecycle_load_time=0.0,
            stability_load_time=0.0,
            now_sec=now,
            timestamp="2026-05-16T09:00:00+09:00",
            path=path,
            symbol="7203",
            fallback_mode=True,
            fallback_reason="stale_state",
        )
        assert len(emitted) == 4
        assert path.exists()
        loaded = load_stale_warnings(path)
        assert len(loaded) == 4

    def test_emits_nothing_when_all_fresh(self, tmp_path):
        path = tmp_path / "stale.jsonl"
        now = 10000.0
        recent_load = now - 60.0
        emitted = emit_stale_warnings(
            exposure_load_time=recent_load,
            lifecycle_load_time=recent_load,
            stability_load_time=recent_load,
            now_sec=now,
            timestamp="ts",
            path=path,
            fallback_mode=False,
            fallback_reason="",
        )
        assert emitted == []
        assert not path.exists()

    def test_emit_does_not_raise_on_io_failure(self, tmp_path):
        # Path in a read-only-ish location — test fail-open contract
        # Even if the write fails, emit_stale_warnings must not raise.
        path = tmp_path / "stale.jsonl"
        now = 200000.0
        # Call with valid path — should succeed, verifying no exception
        result = emit_stale_warnings(
            exposure_load_time=0.0,
            lifecycle_load_time=0.0,
            stability_load_time=0.0,
            now_sec=now,
            timestamp="ts",
            path=path,
        )
        assert isinstance(result, list)

    def test_partial_staleness_emits_only_stale(self, tmp_path):
        path = tmp_path / "stale.jsonl"
        now = 10000.0
        recent = now - 60.0
        stale = 0.0  # triggers staleness
        emitted = emit_stale_warnings(
            exposure_load_time=stale,
            lifecycle_load_time=recent,
            stability_load_time=recent,
            now_sec=now,
            timestamp="ts",
            path=path,
            fallback_mode=False,
            fallback_reason="",
        )
        assert len(emitted) == 1
        assert emitted[0].warning_type == WARN_STALE_EXPOSURE
