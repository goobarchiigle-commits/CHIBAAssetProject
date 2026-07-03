"""
tests/analytics/test_missed_alpha.py
Tests for missed_alpha_registry and missed_alpha_enrichment modules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.analytics.missed_alpha_registry import (
    MissedAlphaReason,
    MissedAlphaRecord,
    append_missed_alpha,
    load_missed_alpha,
    record_missed_alpha_from_bridge,
)
from src.analytics.missed_alpha_enrichment import (
    EnrichedMissedAlphaSummary,
    enrich_missed_alpha_records,
    summarize_missed_alpha,
    run_missed_alpha_enrichment_hook,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_record(
    symbol: str = "7203.T",
    reason: MissedAlphaReason = MissedAlphaReason.GOVERNANCE_FROZEN,
    record_date: str = "2026-05-25",
) -> MissedAlphaRecord:
    return MissedAlphaRecord(
        record_date=record_date,
        symbol=symbol,
        reason=reason,
        future_leader_score=72.5,
        rsr_zone="weak",
        governance_hash="abc123",
        block_details="governance frozen: broker_timeout",
    )


def _write_survivability(log_dir: Path, symbol: str, obs_date: str, fwd5: float, fwd10: float, mfe: float) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = obs_date.replace("-", "")
    p = log_dir / f"future_leader_survivability_{date_str}.jsonl"
    record = {
        "symbol": symbol,
        "observation_date": obs_date,
        "fwd_ret_5d": fwd5,
        "fwd_ret_10d": fwd10,
        "fwd_mfe": mfe,
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Tests: MissedAlphaRecord ──────────────────────────────────────────────────

class TestMissedAlphaRecord:
    def test_to_dict_roundtrip(self):
        rec = _make_record()
        d = rec.to_dict()
        assert d["symbol"] == "7203.T"
        assert d["reason"] == "governance_frozen"
        assert d["enriched"] is False

    def test_from_dict_roundtrip(self):
        rec = _make_record()
        d = rec.to_dict()
        loaded = MissedAlphaRecord.from_dict(d)
        assert loaded.symbol == rec.symbol
        assert loaded.reason == rec.reason
        assert loaded.future_leader_score == rec.future_leader_score

    def test_unknown_reason_becomes_other(self):
        d = _make_record().to_dict()
        d["reason"] = "completely_unknown_reason"
        loaded = MissedAlphaRecord.from_dict(d)
        assert loaded.reason == MissedAlphaReason.OTHER

    def test_all_reasons_valid_enum_values(self):
        for r in MissedAlphaReason:
            d = _make_record(reason=r).to_dict()
            loaded = MissedAlphaRecord.from_dict(d)
            assert loaded.reason == r


class TestMissedAlphaJSONL:
    def test_append_creates_file(self, tmp_path):
        rec = _make_record()
        append_missed_alpha(rec, tmp_path)
        assert (tmp_path / "missed_alpha_20260525.jsonl").exists()

    def test_append_and_load_roundtrip(self, tmp_path):
        rec = _make_record()
        append_missed_alpha(rec, tmp_path)
        loaded = load_missed_alpha(tmp_path, date_str="20260525")
        assert len(loaded) == 1
        assert loaded[0].symbol == "7203.T"
        assert loaded[0].reason == MissedAlphaReason.GOVERNANCE_FROZEN

    def test_multiple_records_same_date(self, tmp_path):
        for sym in ["7203.T", "9984.T", "6758.T"]:
            append_missed_alpha(_make_record(symbol=sym), tmp_path)
        loaded = load_missed_alpha(tmp_path, date_str="20260525")
        assert len(loaded) == 3
        symbols = {r.symbol for r in loaded}
        assert symbols == {"7203.T", "9984.T", "6758.T"}

    def test_load_all_dates(self, tmp_path):
        for date in ["2026-05-24", "2026-05-25"]:
            append_missed_alpha(_make_record(record_date=date), tmp_path)
        all_recs = load_missed_alpha(tmp_path)
        assert len(all_recs) == 2

    def test_load_missing_dir_returns_empty(self, tmp_path):
        result = load_missed_alpha(tmp_path / "nonexistent")
        assert result == []


class TestRecordMissedAlphaFromBridge:
    def test_writes_records_for_each_symbol(self, tmp_path):
        records = record_missed_alpha_from_bridge(
            blocked_symbols=["7203.T", "9984.T"],
            block_reason=MissedAlphaReason.MAX_POSITIONS_HIT,
            block_details="max positions=1 reached",
            governance_hash="deadbeef",
            log_dir=tmp_path,
            evaluation_date="2026-05-25",
        )
        assert len(records) == 2
        loaded = load_missed_alpha(tmp_path, date_str="20260525")
        assert len(loaded) == 2

    def test_score_and_zone_preserved(self, tmp_path):
        records = record_missed_alpha_from_bridge(
            blocked_symbols=["7203.T"],
            block_reason=MissedAlphaReason.LOW_SCORE,
            block_details="score < 70",
            governance_hash=None,
            log_dir=tmp_path,
            evaluation_date="2026-05-25",
            scores={"7203.T": 65.0},
            rsr_zones={"7203.T": "weak"},
        )
        assert records[0].future_leader_score == 65.0
        assert records[0].rsr_zone == "weak"


# ── Tests: MissedAlphaEnrichment ─────────────────────────────────────────────

class TestEnrichMissedAlpha:
    def test_enriched_with_survivability_data(self, tmp_path):
        surv_dir = tmp_path / "survivability"
        _write_survivability(surv_dir, "7203.T", "2026-05-25", 0.03, 0.05, 0.08)

        rec = _make_record(symbol="7203.T", record_date="2026-05-25")
        enriched = enrich_missed_alpha_records([rec], surv_dir)

        assert len(enriched) == 1
        assert enriched[0].enriched is True
        assert enriched[0].fwd_ret_5d == pytest.approx(0.03)
        assert enriched[0].fwd_ret_10d == pytest.approx(0.05)
        assert enriched[0].fwd_mfe == pytest.approx(0.08)

    def test_no_survivability_data_stays_unenriched(self, tmp_path):
        rec = _make_record()
        enriched = enrich_missed_alpha_records([rec], tmp_path / "empty")
        assert len(enriched) == 1
        assert enriched[0].enriched is False

    def test_already_enriched_record_unchanged(self, tmp_path):
        from dataclasses import replace
        rec = _make_record()
        pre_enriched = replace(rec, enriched=True, fwd_ret_5d=0.10)
        result = enrich_missed_alpha_records([pre_enriched], tmp_path / "empty")
        assert result[0].fwd_ret_5d == pytest.approx(0.10)
        assert result[0].enriched is True

    def test_empty_input_returns_empty(self, tmp_path):
        result = enrich_missed_alpha_records([], tmp_path)
        assert result == []


class TestSummarizeMissedAlpha:
    def test_summary_with_enriched_records(self, tmp_path):
        surv_dir = tmp_path / "surv"
        records = []
        for sym, fwd5, fwd10, mfe in [
            ("7203.T", 0.03, 0.06, 0.10),
            ("9984.T", 0.01, 0.02, 0.03),
            ("6758.T", 0.05, 0.12, 0.15),
        ]:
            _write_survivability(surv_dir, sym, "2026-05-25", fwd5, fwd10, mfe)
            records.append(_make_record(symbol=sym, record_date="2026-05-25"))

        enriched = enrich_missed_alpha_records(records, surv_dir)
        summary = summarize_missed_alpha(enriched)

        assert summary.total_records == 3
        assert summary.enriched_count == 3
        assert summary.unenriched_count == 0
        # avg fwd_ret_10d = (0.06 + 0.02 + 0.12) / 3 = 0.0667
        assert summary.avg_fwd_ret_10d == pytest.approx(0.0667, rel=1e-3)
        # high cost: fwd_ret_10d >= 0.05 → 9984.T(0.02) no, 7203.T(0.06) yes, 6758.T(0.12) yes
        assert summary.high_cost_count == 2

    def test_summary_empty_records(self):
        summary = summarize_missed_alpha([])
        assert summary.total_records == 0
        assert summary.avg_fwd_ret_10d is None

    def test_reason_breakdown(self):
        records = [
            _make_record(reason=MissedAlphaReason.GOVERNANCE_FROZEN),
            _make_record(reason=MissedAlphaReason.GOVERNANCE_FROZEN),
            _make_record(reason=MissedAlphaReason.MAX_POSITIONS_HIT),
        ]
        summary = summarize_missed_alpha(records)
        assert summary.reason_breakdown["governance_frozen"] == 2
        assert summary.reason_breakdown["max_positions_hit"] == 1


class TestRunMissedAlphaEnrichmentHook:
    def test_hook_produces_summary(self, tmp_path):
        missed_dir = tmp_path / "missed"
        surv_dir   = tmp_path / "surv"
        enriched_dir = tmp_path / "enriched"

        rec = _make_record(record_date="2026-05-25")
        append_missed_alpha(rec, missed_dir)
        _write_survivability(surv_dir, "7203.T", "2026-05-25", 0.04, 0.08, 0.12)

        summary = run_missed_alpha_enrichment_hook(
            missed_alpha_log_dir=missed_dir,
            survivability_log_dir=surv_dir,
            enriched_log_dir=enriched_dir,
            evaluation_date="2026-05-25",
        )
        assert summary is not None
        assert summary.total_records == 1
        assert summary.enriched_count == 1
        assert (enriched_dir / "enriched_missed_alpha_20260525.jsonl").exists()

    def test_hook_no_records_returns_none(self, tmp_path):
        summary = run_missed_alpha_enrichment_hook(
            missed_alpha_log_dir=tmp_path / "empty",
            survivability_log_dir=tmp_path / "surv",
            enriched_log_dir=tmp_path / "enriched",
            evaluation_date="2026-05-25",
        )
        assert summary is None
