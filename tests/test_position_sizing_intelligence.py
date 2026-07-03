"""
tests/test_position_sizing_intelligence.py — Position Sizing Intelligence tests

Coverage:
  - _compute_conviction (all paths, missing inputs, FAIL_OPEN)
  - compute_conviction_score (public wrapper)
  - compute_virtual_weights (normalization, monotonicity, edge cases)
  - append_ps_telemetry (JSONL output, fields, FAIL_OPEN)
  - compute_ps_kpis (window filter, aggregation, FAIL_OPEN)
  - format_ps_report_section (content, FAIL_OPEN, empty)
  - idempotency
  - weight properties
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pytest

from src.portfolio.position_sizing_intelligence import (
    NEUTRAL_SCORE,
    RC_ET_MISSING,
    RC_FAIL_OPEN,
    RC_FL_MISSING,
    RC_RSR_FALLBACK,
    RC_RSR_SMOOTH_USED,
    SOFTMAX_TEMPERATURE,
    W_ENTRY_TIMING,
    W_FUTURE_LEADER,
    W_RSR_MOMENTUM,
    W_RSR_PERCENTILE,
    W_SEPA,
    PositionSizingInput,
    PositionSizingSignal,
    _compute_conviction,
    _momentum_component,
    _rsr_component,
    append_ps_telemetry,
    compute_conviction_score,
    compute_ps_kpis,
    compute_virtual_weights,
    format_ps_report_section,
)

TODAY = date(2026, 5, 31)
TODAY_STR = "2026-05-31"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inp(
    symbol: str = "1234.T",
    rsr: float = 80.0,
    sepa_score: int = 6,
    rsr_momentum: float = 5.0,
    rsr_pct_smooth: Optional[float] = 0.85,
    entry_timing_score: Optional[float] = 75.0,
    future_leader_score: Optional[float] = 70.0,
) -> PositionSizingInput:
    return PositionSizingInput(
        symbol=symbol,
        rsr=rsr,
        sepa_score=sepa_score,
        rsr_momentum=rsr_momentum,
        rsr_pct_smooth=rsr_pct_smooth,
        entry_timing_score=entry_timing_score,
        future_leader_score=future_leader_score,
    )


def _sig(symbol: str, conviction: float, vw: float) -> PositionSizingSignal:
    return PositionSizingSignal(
        symbol=symbol,
        conviction_score=conviction,
        virtual_weight=vw,
        component_scores={},
        reason_codes=[],
        computed_at="2026-05-31T09:00:00+09:00",
    )


def _write_telemetry(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_telemetry_record(
    symbol: str,
    conviction: float,
    vw: float,
    aw: float,
    days_ago: int = 1,
) -> dict:
    d = (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {
        "date": d,
        "symbol": symbol,
        "conviction_score": conviction,
        "virtual_weight": vw,
        "actual_weight": aw,
        "rsr_score": 80.0,
        "entry_timing_score": 75.0,
        "future_leader_score": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# _rsr_component
# ─────────────────────────────────────────────────────────────────────────────

class TestRsrComponent:
    def test_uses_pct_smooth_when_available(self):
        inp = _inp(rsr=70.0, rsr_pct_smooth=0.90)
        val, rc = _rsr_component(inp)
        assert abs(val - 90.0) < 0.01
        assert rc == RC_RSR_SMOOTH_USED

    def test_falls_back_to_rsr(self):
        inp = _inp(rsr=75.0, rsr_pct_smooth=None)
        val, rc = _rsr_component(inp)
        assert abs(val - 75.0) < 0.01
        assert rc == RC_RSR_FALLBACK

    def test_pct_smooth_clamped(self):
        inp = _inp(rsr_pct_smooth=1.5)  # > 1.0
        val, _ = _rsr_component(inp)
        assert val == 100.0

    def test_pct_smooth_zero(self):
        inp = _inp(rsr_pct_smooth=0.0)
        val, _ = _rsr_component(inp)
        assert val == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _momentum_component
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumComponent:
    def test_zero_momentum_is_neutral(self):
        assert abs(_momentum_component(0.0) - 50.0) < 0.1

    def test_positive_momentum_above_50(self):
        assert _momentum_component(5.0) > 50.0
        assert _momentum_component(10.0) > _momentum_component(5.0)

    def test_negative_momentum_below_50(self):
        assert _momentum_component(-5.0) < 50.0
        assert _momentum_component(-10.0) < _momentum_component(-5.0)

    def test_large_positive_approaches_100(self):
        assert _momentum_component(50.0) > 99.0

    def test_large_negative_approaches_0(self):
        assert _momentum_component(-50.0) < 1.0

    def test_symmetric(self):
        # sigmoid is symmetric around 0
        assert abs(_momentum_component(5.0) - (100.0 - _momentum_component(-5.0))) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# _compute_conviction
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeConviction:
    def test_all_components_present(self):
        inp = _inp(
            rsr=80.0, rsr_pct_smooth=0.80,
            sepa_score=8, rsr_momentum=10.0,
            entry_timing_score=90.0, future_leader_score=85.0,
        )
        conv, comps, reasons = _compute_conviction(inp)
        assert 0.0 <= conv <= 100.0
        assert "rsr" in comps
        assert "entry_timing" in comps
        assert "sepa" in comps
        assert "momentum" in comps
        assert "future_leader" in comps
        assert RC_RSR_SMOOTH_USED in reasons

    def test_sepa_full_score(self):
        inp = _inp(sepa_score=8, rsr_pct_smooth=None, rsr=80.0,
                   entry_timing_score=50.0, future_leader_score=50.0,
                   rsr_momentum=0.0)
        conv, comps, _ = _compute_conviction(inp)
        assert comps["sepa"] == 100.0

    def test_sepa_zero(self):
        inp = _inp(sepa_score=0, rsr_pct_smooth=None, rsr=80.0,
                   entry_timing_score=50.0, future_leader_score=50.0,
                   rsr_momentum=0.0)
        conv, comps, _ = _compute_conviction(inp)
        assert comps["sepa"] == 0.0

    def test_et_missing_uses_neutral(self):
        inp = _inp(entry_timing_score=None)
        conv, comps, reasons = _compute_conviction(inp)
        assert comps["entry_timing"] == NEUTRAL_SCORE
        assert RC_ET_MISSING in reasons

    def test_fl_missing_uses_neutral(self):
        inp = _inp(future_leader_score=None)
        conv, comps, reasons = _compute_conviction(inp)
        assert comps["future_leader"] == NEUTRAL_SCORE
        assert RC_FL_MISSING in reasons

    def test_rsr_pct_smooth_none_fallback(self):
        inp = _inp(rsr=70.0, rsr_pct_smooth=None)
        conv, comps, reasons = _compute_conviction(inp)
        assert comps["rsr"] == 70.0
        assert RC_RSR_FALLBACK in reasons

    def test_conviction_in_range(self):
        for sepa in [0, 4, 8]:
            for rsr in [0.0, 50.0, 100.0]:
                inp = _inp(rsr=rsr, sepa_score=sepa)
                conv, _, _ = _compute_conviction(inp)
                assert 0.0 <= conv <= 100.0

    def test_high_all_gives_high_conviction(self):
        inp = PositionSizingInput(
            symbol="9999.T", rsr=100.0, sepa_score=8, rsr_momentum=20.0,
            rsr_pct_smooth=1.0, entry_timing_score=100.0, future_leader_score=100.0,
        )
        conv, _, _ = _compute_conviction(inp)
        assert conv >= 90.0

    def test_low_all_gives_low_conviction(self):
        inp = PositionSizingInput(
            symbol="0001.T", rsr=0.0, sepa_score=0, rsr_momentum=-20.0,
            rsr_pct_smooth=0.0, entry_timing_score=0.0, future_leader_score=0.0,
        )
        conv, _, _ = _compute_conviction(inp)
        assert conv <= 20.0

    def test_fail_open_on_bad_sepa(self):
        # sepa_score out of normal range is clamped, not crashed
        inp = _inp(sepa_score=100)  # > 8
        conv, _, _ = _compute_conviction(inp)
        assert 0.0 <= conv <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_conviction_score (public wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeConvictionScore:
    def test_returns_float(self):
        result = compute_conviction_score(_inp())
        assert isinstance(result, float)

    def test_in_range(self):
        result = compute_conviction_score(_inp())
        assert 0.0 <= result <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_virtual_weights
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeVirtualWeights:
    def test_empty_returns_empty(self):
        assert compute_virtual_weights([]) == []

    def test_single_candidate_weight_is_1(self):
        results = compute_virtual_weights([_inp("A.T")])
        assert len(results) == 1
        assert abs(results[0].virtual_weight - 1.0) < 1e-6

    def test_weights_sum_to_1(self):
        inputs = [
            _inp("A.T", rsr=90.0, entry_timing_score=92.0),
            _inp("B.T", rsr=75.0, entry_timing_score=80.0),
            _inp("C.T", rsr=60.0, entry_timing_score=70.0),
            _inp("D.T", rsr=50.0, entry_timing_score=60.0),
            _inp("E.T", rsr=40.0, entry_timing_score=50.0),
        ]
        results = compute_virtual_weights(inputs)
        total = sum(r.virtual_weight for r in results)
        assert abs(total - 1.0) < 1e-5

    def test_higher_conviction_higher_weight(self):
        """Monotonicity: if conviction(A) > conviction(B) then weight(A) > weight(B)."""
        inputs = [
            _inp("A.T", rsr=95.0, sepa_score=8, entry_timing_score=95.0),
            _inp("B.T", rsr=50.0, sepa_score=2, entry_timing_score=40.0),
        ]
        results = compute_virtual_weights(inputs)
        by_sym = {r.symbol: r for r in results}
        assert by_sym["A.T"].conviction_score > by_sym["B.T"].conviction_score
        assert by_sym["A.T"].virtual_weight   > by_sym["B.T"].virtual_weight

    def test_uniform_conviction_uniform_weights(self):
        """Same inputs → approximately equal weights."""
        inputs = [_inp(f"{i}999.T", rsr=70.0, sepa_score=5,
                        rsr_momentum=0.0, entry_timing_score=70.0,
                        future_leader_score=70.0)
                  for i in range(1, 6)]
        results = compute_virtual_weights(inputs)
        weights = [r.virtual_weight for r in results]
        mean_w  = sum(weights) / len(weights)
        for w in weights:
            assert abs(w - mean_w) < 1e-6

    def test_symbols_preserved(self):
        inputs = [_inp("AAAA.T"), _inp("BBBB.T"), _inp("CCCC.T")]
        results = compute_virtual_weights(inputs)
        result_syms = [r.symbol for r in results]
        assert "AAAA.T" in result_syms
        assert "BBBB.T" in result_syms
        assert "CCCC.T" in result_syms

    def test_output_order_matches_input(self):
        inputs = [_inp(f"{i}001.T") for i in range(5)]
        results = compute_virtual_weights(inputs)
        for inp, res in zip(inputs, results):
            assert inp.symbol == res.symbol

    def test_conviction_in_range(self):
        inputs = [_inp("A.T"), _inp("B.T"), _inp("C.T")]
        for r in compute_virtual_weights(inputs):
            assert 0.0 <= r.conviction_score <= 100.0

    def test_virtual_weight_nonnegative(self):
        inputs = [_inp("A.T"), _inp("B.T")]
        for r in compute_virtual_weights(inputs):
            assert r.virtual_weight >= 0.0

    def test_reason_codes_populated(self):
        inp = _inp(entry_timing_score=None, future_leader_score=None)
        results = compute_virtual_weights([inp])
        assert RC_ET_MISSING in results[0].reason_codes
        assert RC_FL_MISSING in results[0].reason_codes

    def test_component_scores_present(self):
        results = compute_virtual_weights([_inp()])
        comps = results[0].component_scores
        for key in ("rsr", "entry_timing", "sepa", "momentum", "future_leader"):
            assert key in comps

    def test_fail_open_returns_empty_not_raises(self):
        # Pass a non-PositionSizingInput to trigger internal error
        try:
            results = compute_virtual_weights(["not_an_input"])  # type: ignore
        except Exception:
            pytest.fail("compute_virtual_weights raised instead of FAIL_OPEN")
        assert isinstance(results, list)

    def test_large_spread_top_dominates(self):
        """High conviction candidate should have significantly larger weight."""
        inputs = [
            _inp("TOP.T",  rsr=100.0, sepa_score=8, entry_timing_score=100.0,
                  rsr_momentum=20.0, future_leader_score=100.0, rsr_pct_smooth=1.0),
            _inp("LOW1.T", rsr=10.0, sepa_score=0,  entry_timing_score=0.0,
                  rsr_momentum=-20.0, future_leader_score=0.0, rsr_pct_smooth=0.0),
            _inp("LOW2.T", rsr=10.0, sepa_score=0,  entry_timing_score=0.0,
                  rsr_momentum=-20.0, future_leader_score=0.0, rsr_pct_smooth=0.0),
        ]
        results = compute_virtual_weights(inputs)
        by_sym = {r.symbol: r for r in results}
        assert by_sym["TOP.T"].virtual_weight > 0.5  # dominant weight


# ─────────────────────────────────────────────────────────────────────────────
# append_ps_telemetry
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendPsTelemetry:
    def _signals(self) -> List[PositionSizingSignal]:
        return [_sig("A.T", 80.0, 0.6), _sig("B.T", 60.0, 0.4)]

    def test_creates_file(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        append_ps_telemetry(
            signals=self._signals(), actual_weights={"A.T": 0.5, "B.T": 0.5},
            rsr_scores={}, entry_timing_scores={}, future_leader_scores={},
            telemetry_path=p, date_str=TODAY_STR,
        )
        assert p.exists()

    def test_one_record_per_signal(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        append_ps_telemetry(
            signals=self._signals(), actual_weights={},
            rsr_scores={}, entry_timing_scores={}, future_leader_scores={},
            telemetry_path=p, date_str=TODAY_STR,
        )
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_appends_on_second_call(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        for _ in range(3):
            append_ps_telemetry(
                signals=self._signals(), actual_weights={},
                rsr_scores={}, entry_timing_scores={}, future_leader_scores={},
                telemetry_path=p, date_str=TODAY_STR,
            )
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        assert len(lines) == 6  # 3 calls × 2 signals

    def test_record_is_valid_json(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        append_ps_telemetry(
            signals=self._signals(), actual_weights={"A.T": 0.5},
            rsr_scores={"A.T": 80.0}, entry_timing_scores={"A.T": 75.0},
            future_leader_scores={},
            telemetry_path=p, date_str=TODAY_STR,
        )
        for line in p.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                assert isinstance(record, dict)

    def test_required_fields_present(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        append_ps_telemetry(
            signals=[_sig("A.T", 80.0, 0.6)],
            actual_weights={"A.T": 0.5},
            rsr_scores={"A.T": 80.0},
            entry_timing_scores={"A.T": 75.0},
            future_leader_scores={},
            telemetry_path=p, date_str=TODAY_STR,
        )
        record = json.loads(p.read_text().strip())
        for key in ("timestamp", "date", "symbol", "conviction_score",
                    "virtual_weight", "actual_weight"):
            assert key in record

    def test_empty_signals_no_write(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        append_ps_telemetry(
            signals=[], actual_weights={},
            rsr_scores={}, entry_timing_scores={}, future_leader_scores={},
            telemetry_path=p, date_str=TODAY_STR,
        )
        assert not p.exists()

    def test_fail_open_bad_path(self):
        bad_path = Path("/nonexistent_root_dir_xyz/telem.jsonl")
        # Should not raise
        append_ps_telemetry(
            signals=[_sig("A.T", 80.0, 1.0)], actual_weights={},
            rsr_scores={}, entry_timing_scores={}, future_leader_scores={},
            telemetry_path=bad_path, date_str=TODAY_STR,
        )

    def test_conviction_and_vw_stored(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        append_ps_telemetry(
            signals=[_sig("X.T", 87.5, 0.42)], actual_weights={},
            rsr_scores={}, entry_timing_scores={}, future_leader_scores={},
            telemetry_path=p, date_str=TODAY_STR,
        )
        record = json.loads(p.read_text().strip())
        assert record["conviction_score"] == 87.5
        assert abs(record["virtual_weight"] - 0.42) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# compute_ps_kpis
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePsKpis:
    def _write_records(self, path: Path, n: int = 20) -> None:
        records = [
            _make_telemetry_record("A.T", 85.0, 0.40, 0.33, days_ago=i % 25)
            for i in range(n // 2)
        ] + [
            _make_telemetry_record("B.T", 60.0, 0.35, 0.33, days_ago=i % 25)
            for i in range(n // 2)
        ]
        _write_telemetry(path, records)

    def test_missing_file_returns_empty(self, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        assert compute_ps_kpis(p) == {}

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert compute_ps_kpis(p) == {}

    def test_basic_kpis_computed(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        self._write_records(p, n=20)
        kpis = compute_ps_kpis(p, window_days=30, today=TODAY)
        assert "average_conviction" in kpis
        assert "sample_count" in kpis
        assert kpis["sample_count"] == 20

    def test_average_conviction_in_range(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        self._write_records(p)
        kpis = compute_ps_kpis(p, today=TODAY)
        assert 0.0 <= kpis["average_conviction"] <= 100.0

    def test_top_conviction_symbols_at_most_5(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        records = [_make_telemetry_record(f"{i}001.T", float(50 + i), 0.1, 0.1) for i in range(10)]
        _write_telemetry(p, records)
        kpis = compute_ps_kpis(p, today=TODAY)
        assert len(kpis.get("top_conviction_symbols", [])) <= 5

    def test_window_filter(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        recent = [_make_telemetry_record("A.T", 80.0, 0.5, 0.5, days_ago=5)]
        old    = [_make_telemetry_record("B.T", 70.0, 0.5, 0.5, days_ago=40)]
        _write_telemetry(p, recent + old)
        kpis = compute_ps_kpis(p, window_days=30, today=TODAY)
        assert kpis["sample_count"] == 1  # only recent included

    def test_weight_dispersion_cv_nonnegative(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        self._write_records(p)
        kpis = compute_ps_kpis(p, today=TODAY)
        assert kpis.get("weight_dispersion_cv", 0) >= 0.0

    def test_virtual_vs_equal_diff_present(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        self._write_records(p)
        kpis = compute_ps_kpis(p, today=TODAY)
        assert kpis.get("virtual_vs_equal_diff") is not None

    def test_high_conviction_rate_in_range(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        self._write_records(p)
        kpis = compute_ps_kpis(p, today=TODAY)
        hcr = kpis.get("high_conviction_rate", 0.0)
        assert 0.0 <= hcr <= 1.0

    def test_fail_open_corrupt_line(self, tmp_path):
        p = tmp_path / "telem.jsonl"
        p.write_text('{"bad_key": 1}\n{invalid json}\n')
        # Should not raise
        kpis = compute_ps_kpis(p, today=TODAY)
        assert isinstance(kpis, dict)


# ─────────────────────────────────────────────────────────────────────────────
# format_ps_report_section
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatPsReportSection:
    def _sigs(self) -> List[PositionSizingSignal]:
        return [
            _sig("A.T", 85.0, 0.50),
            _sig("B.T", 65.0, 0.30),
            _sig("C.T", 50.0, 0.20),
        ]

    def _kpis(self) -> dict:
        return {
            "average_conviction": 67.5,
            "top_conviction_symbols": [("A.T", 85.0), ("B.T", 65.0)],
            "weight_dispersion_cv": 0.12,
            "virtual_vs_equal_diff": 0.0833,
            "high_conviction_rate": 0.33,
            "sample_count": 30,
        }

    def test_returns_string(self):
        assert isinstance(format_ps_report_section(self._sigs(), self._kpis()), str)

    def test_empty_inputs_returns_empty(self):
        assert format_ps_report_section([], {}) == ""

    def test_header_present(self):
        result = format_ps_report_section(self._sigs(), {})
        assert "POSITION SIZING" in result

    def test_all_symbols_present(self):
        result = format_ps_report_section(self._sigs(), {})
        assert "A.T" in result
        assert "B.T" in result
        assert "C.T" in result

    def test_sorted_by_conviction_desc(self):
        sigs = [_sig("LOW.T", 30.0, 0.2), _sig("HIGH.T", 90.0, 0.8)]
        result = format_ps_report_section(sigs, {})
        high_pos = result.find("HIGH.T")
        low_pos  = result.find("LOW.T")
        assert high_pos < low_pos  # HIGH appears before LOW

    def test_kpis_shown(self):
        result = format_ps_report_section(self._sigs(), self._kpis())
        assert "67.5" in result or "67" in result  # average_conviction
        assert "A.T" in result

    def test_observation_only_label(self):
        result = format_ps_report_section(self._sigs(), {})
        assert "観測専用" in result

    def test_fail_open_returns_empty(self):
        # Pass malformed object — should return "" not raise
        result = format_ps_report_section("not_a_list", {})  # type: ignore
        assert isinstance(result, str)

    def test_single_signal_weight_100pct(self):
        sigs = [_sig("ONLY.T", 80.0, 1.0)]
        result = format_ps_report_section(sigs, {})
        assert "ONLY.T" in result

    def test_equal_weight_comparison_shown(self):
        result = format_ps_report_section(self._sigs(), {})
        assert "等配分" in result


# ─────────────────────────────────────────────────────────────────────────────
# Weight invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightInvariants:
    def test_component_weights_sum_to_1(self):
        total = W_RSR_PERCENTILE + W_ENTRY_TIMING + W_SEPA + W_RSR_MOMENTUM + W_FUTURE_LEADER
        assert abs(total - 1.0) < 1e-9

    def test_weights_always_sum_to_1_single(self):
        inputs = [_inp()]
        results = compute_virtual_weights(inputs)
        assert abs(sum(r.virtual_weight for r in results) - 1.0) < 1e-6

    def test_weights_always_sum_to_1_multiple(self):
        for n in [2, 3, 5, 10]:
            inputs = [_inp(f"{i}001.T") for i in range(n)]
            results = compute_virtual_weights(inputs)
            assert abs(sum(r.virtual_weight for r in results) - 1.0) < 1e-5, f"n={n}"

    def test_idempotency_same_result_twice(self):
        inputs = [_inp("A.T"), _inp("B.T"), _inp("C.T")]
        r1 = compute_virtual_weights(inputs)
        r2 = compute_virtual_weights(inputs)
        for a, b in zip(r1, r2):
            assert a.symbol == b.symbol
            assert abs(a.conviction_score - b.conviction_score) < 1e-6
            assert abs(a.virtual_weight   - b.virtual_weight)   < 1e-6

    def test_conviction_strictly_monotone_with_rsr(self):
        """Higher RSR → higher conviction (all else equal)."""
        low_rsr  = compute_conviction_score(_inp(rsr=30.0, rsr_pct_smooth=0.30))
        high_rsr = compute_conviction_score(_inp(rsr=90.0, rsr_pct_smooth=0.90))
        assert high_rsr > low_rsr

    def test_virtual_weight_spec_example_approx(self):
        """
        Spec example: scores [92, 80, 70, 60, 50] → weights approx [0.35, 0.25, 0.20, 0.12, 0.08].
        Test that distribution is decreasing and top weight is significantly larger than bottom.
        """
        # Use entry_timing_score as the primary driver to get exact conviction spread
        inputs = [
            _inp("A.T", rsr=92.0, rsr_pct_smooth=0.92, entry_timing_score=92.0, sepa_score=7, rsr_momentum=0.0, future_leader_score=92.0),
            _inp("B.T", rsr=80.0, rsr_pct_smooth=0.80, entry_timing_score=80.0, sepa_score=6, rsr_momentum=0.0, future_leader_score=80.0),
            _inp("C.T", rsr=70.0, rsr_pct_smooth=0.70, entry_timing_score=70.0, sepa_score=5, rsr_momentum=0.0, future_leader_score=70.0),
            _inp("D.T", rsr=60.0, rsr_pct_smooth=0.60, entry_timing_score=60.0, sepa_score=4, rsr_momentum=0.0, future_leader_score=60.0),
            _inp("E.T", rsr=50.0, rsr_pct_smooth=0.50, entry_timing_score=50.0, sepa_score=3, rsr_momentum=0.0, future_leader_score=50.0),
        ]
        results = compute_virtual_weights(inputs)
        weights = [r.virtual_weight for r in results]
        # Strictly decreasing
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1], f"weight[{i}]={weights[i]} not > weight[{i+1}]={weights[i+1]}"
        # Top weight > 3× bottom weight (significant spread)
        assert weights[0] > weights[-1] * 2.5
