"""
tests/analytics/test_suppression_outcome_telemetry.py

Unit tests for suppression_outcome_telemetry.py.
Covers: outcome classification, suppression detection, lifecycle state,
exit resolution, MFE/MAE computation, phase transitions, attribution
metrics, sufficiency check, and the integration hook.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pandas as pd
import numpy as np

from src.analytics.suppression_outcome_telemetry import (
    ALPHA_EXT_THRESHOLD,
    LOSS_DELAY_THRESHOLD,
    SUFFICIENCY_MIN_ALPHA_EXT,
    SUFFICIENCY_MIN_PHASE_TRANSITIONS,
    SUFFICIENCY_MIN_SAMPLES,
    classify_outcome,
    compute_sufficiency_check,
    compute_suppression_attribution,
    detect_new_suppressions,
    detect_phase_transitions,
    run_suppression_outcome_hook,
    _compute_window_excursions,
    _get_close_at_date,
    _load_active_state,
    _load_jsonl,
    _run_internal,
    _safe,
    _save_active_state,
    _write_suppression_outcome,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(dates, closes, highs=None, lows=None):
    """Build minimal OHLCV DataFrame from arrays."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    n = len(closes)
    highs = highs or [c * 1.01 for c in closes]
    lows  = lows  or [c * 0.99 for c in closes]
    return pd.DataFrame({
        "Close":  closes,
        "High":   highs,
        "Low":    lows,
        "Volume": [100_000] * n,
    }, index=idx)


def _null_loader(sym: str) -> Optional[pd.DataFrame]:
    return None


# ── TestSafe ──────────────────────────────────────────────────────────────────

class TestSafe(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(_safe(1.5), 1.5)

    def test_nan_returns_fallback(self):
        self.assertEqual(_safe(float("nan")), 0.0)

    def test_inf_returns_fallback(self):
        self.assertEqual(_safe(float("inf")), 0.0)

    def test_none_returns_fallback(self):
        self.assertEqual(_safe(None, 99.0), 99.0)

    def test_string_returns_fallback(self):
        self.assertEqual(_safe("bad", -1.0), -1.0)


# ── TestClassifyOutcome ───────────────────────────────────────────────────────

class TestClassifyOutcome(unittest.TestCase):
    def test_alpha_extension(self):
        self.assertEqual(classify_outcome(0.05), "alpha_extension")

    def test_alpha_extension_exact_threshold(self):
        self.assertEqual(classify_outcome(ALPHA_EXT_THRESHOLD), "alpha_extension")

    def test_loss_delay(self):
        self.assertEqual(classify_outcome(-0.05), "loss_delay")

    def test_loss_delay_exact_threshold(self):
        self.assertEqual(classify_outcome(LOSS_DELAY_THRESHOLD), "loss_delay")

    def test_neutral_zero(self):
        self.assertEqual(classify_outcome(0.0), "neutral")

    def test_neutral_small_positive(self):
        self.assertEqual(classify_outcome(0.01), "neutral")

    def test_neutral_small_negative(self):
        self.assertEqual(classify_outcome(-0.01), "neutral")

    def test_just_below_alpha(self):
        self.assertEqual(classify_outcome(0.029), "neutral")

    def test_just_above_loss(self):
        self.assertEqual(classify_outcome(-0.029), "neutral")


# ── TestDetectNewSuppressions ─────────────────────────────────────────────────

class TestDetectNewSuppressions(unittest.TestCase):
    def _suppressed(self, sym="TEST.T", holding=True):
        return {
            "symbol": sym,
            "exit_orchestrator_applied": True,
            "action": "HOLD",
            "currently_holding": holding,
            "current_return": 0.05,
            "compression_phase": "compression",
        }

    def test_basic_detection(self):
        result = detect_new_suppressions([self._suppressed()])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "TEST.T")

    def test_non_suppressed_ignored(self):
        sig = self._suppressed()
        sig["exit_orchestrator_applied"] = False
        self.assertEqual(detect_new_suppressions([sig]), [])

    def test_sell_action_ignored(self):
        sig = self._suppressed()
        sig["action"] = "SELL"
        self.assertEqual(detect_new_suppressions([sig]), [])

    def test_not_holding_ignored(self):
        sig = self._suppressed(holding=False)
        self.assertEqual(detect_new_suppressions([sig]), [])

    def test_no_symbol_ignored(self):
        sig = self._suppressed()
        sig["symbol"] = ""
        self.assertEqual(detect_new_suppressions([sig]), [])

    def test_multiple_signals_filtered(self):
        sigs = [
            self._suppressed("A.T"),
            {"symbol": "B.T", "exit_orchestrator_applied": False,
             "action": "SELL", "currently_holding": True},
            self._suppressed("C.T"),
        ]
        result = detect_new_suppressions(sigs)
        self.assertEqual(len(result), 2)
        syms = [r["symbol"] for r in result]
        self.assertIn("A.T", syms)
        self.assertIn("C.T", syms)

    def test_empty_signals(self):
        self.assertEqual(detect_new_suppressions([]), [])


# ── TestLoadSaveActiveState ───────────────────────────────────────────────────

class TestLoadSaveActiveState(unittest.TestCase):
    def test_load_nonexistent_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            result = _load_active_state(Path(d) / "missing.json")
            self.assertEqual(result, {})

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.json"
            state = {"A.T": {"symbol": "A.T", "state": "active", "original_exit_date": "2026-05-27"}}
            _save_active_state(state, path)
            loaded = _load_active_state(path)
            self.assertEqual(loaded, state)

    def test_corrupted_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.json"
            path.write_text("not valid json", encoding="utf-8")
            result = _load_active_state(path)
            self.assertEqual(result, {})


# ── TestComputeWindowExcursions ───────────────────────────────────────────────

class TestComputeWindowExcursions(unittest.TestCase):
    def _make_df(self, dates, closes):
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
        return pd.DataFrame({"Close": closes}, index=idx)

    def test_mfe_mae_computed(self):
        df = self._make_df(
            ["2026-05-01", "2026-05-02", "2026-05-05", "2026-05-06", "2026-05-07"],
            [100, 103, 98, 106, 104],
        )
        mfe, mae = _compute_window_excursions(df, "2026-05-01", "2026-05-07")
        self.assertIsNotNone(mfe)
        self.assertIsNotNone(mae)
        self.assertAlmostEqual(mfe, 0.06, places=4)  # (106-100)/100
        self.assertAlmostEqual(mae, -0.02, places=4) # (98-100)/100

    def test_none_df_returns_none(self):
        mfe, mae = _compute_window_excursions(None, "2026-05-01", "2026-05-07")
        self.assertIsNone(mfe)
        self.assertIsNone(mae)

    def test_single_row_returns_none(self):
        df = self._make_df(["2026-05-01"], [100])
        mfe, mae = _compute_window_excursions(df, "2026-05-01", "2026-05-01")
        self.assertIsNone(mfe)
        self.assertIsNone(mae)

    def test_empty_df_returns_none(self):
        df = pd.DataFrame({"Close": []})
        mfe, mae = _compute_window_excursions(df, "2026-05-01", "2026-05-07")
        self.assertIsNone(mfe)
        self.assertIsNone(mae)

    def test_no_close_column(self):
        idx = pd.DatetimeIndex(pd.to_datetime(["2026-05-01", "2026-05-02"]))
        df = pd.DataFrame({"Open": [100, 102]}, index=idx)
        mfe, mae = _compute_window_excursions(df, "2026-05-01", "2026-05-02")
        # Falls back to Adj Close lookup — also missing, so None
        self.assertIsNone(mfe)

    def test_zero_start_price_returns_none(self):
        df = self._make_df(["2026-05-01", "2026-05-02"], [0.0, 100.0])
        mfe, mae = _compute_window_excursions(df, "2026-05-01", "2026-05-02")
        self.assertIsNone(mfe)

    def test_flat_prices(self):
        df = self._make_df(
            ["2026-05-01", "2026-05-02", "2026-05-05"],
            [100, 100, 100],
        )
        mfe, mae = _compute_window_excursions(df, "2026-05-01", "2026-05-05")
        self.assertAlmostEqual(mfe, 0.0, places=6)
        self.assertAlmostEqual(mae, 0.0, places=6)


# ── TestGetCloseAtDate ────────────────────────────────────────────────────────

class TestGetCloseAtDate(unittest.TestCase):
    def _make_df(self, dates, closes):
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
        return pd.DataFrame({"Close": closes}, index=idx)

    def test_exact_date(self):
        df = self._make_df(["2026-05-01", "2026-05-02"], [100.0, 105.0])
        v = _get_close_at_date(df, "2026-05-01")
        self.assertAlmostEqual(v, 100.0)

    def test_missing_date_within_tolerance(self):
        df = self._make_df(["2026-05-03"], [110.0])
        v = _get_close_at_date(df, "2026-05-01")  # within 2 day tolerance
        self.assertAlmostEqual(v, 110.0)

    def test_missing_date_outside_tolerance(self):
        df = self._make_df(["2026-05-10"], [110.0])
        v = _get_close_at_date(df, "2026-05-01")
        self.assertIsNone(v)

    def test_none_df_returns_none(self):
        v = _get_close_at_date(None, "2026-05-01")
        self.assertIsNone(v)


# ── TestWriteSuppressionOutcome ───────────────────────────────────────────────

class TestWriteSuppressionOutcome(unittest.TestCase):
    def _base_event(self):
        return {
            "symbol": "6920.T",
            "original_exit_date": "2026-05-27",
            "suppression_start_return": 0.045,
            "phase_at_suppression": "compression",
            "compression_score": 74.2,
            "hold_extension_days": 3,
            "original_exit_reason": "RSR_EXIT",
            "last_seen_return": 0.083,
            "last_seen_phase": "breakout",
            "state": "success_exit",
        }

    def test_alpha_extension_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            outcome_path = Path(d) / "outcomes.jsonl"
            _write_suppression_outcome(
                event=self._base_event(),
                final_return=0.083,
                phase_at_exit="breakout",
                today="2026-05-30",
                ohlcv_loader=_null_loader,
                outcome_path=outcome_path,
            )
            records = _load_jsonl(outcome_path)
            self.assertEqual(len(records), 1)
            r = records[0]
            self.assertEqual(r["suppression_outcome"], "alpha_extension")
            self.assertAlmostEqual(r["return_if_exited_original"], 0.045)
            self.assertAlmostEqual(r["return_after_suppression"], 0.083)
            self.assertEqual(r["phase_at_suppression"], "compression")
            self.assertEqual(r["phase_at_final_exit"], "breakout")
            self.assertEqual(r["suppression_days"], 3)

    def test_loss_delay_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            outcome_path = Path(d) / "outcomes.jsonl"
            event = self._base_event()
            event["suppression_start_return"] = 0.01
            _write_suppression_outcome(
                event=event,
                final_return=-0.04,
                phase_at_exit="distribution",
                today="2026-05-30",
                ohlcv_loader=_null_loader,
                outcome_path=outcome_path,
            )
            records = _load_jsonl(outcome_path)
            self.assertEqual(records[0]["suppression_outcome"], "loss_delay")

    def test_neutral_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            outcome_path = Path(d) / "outcomes.jsonl"
            event = self._base_event()
            event["suppression_start_return"] = 0.045
            _write_suppression_outcome(
                event=event,
                final_return=0.055,
                phase_at_exit="compression",
                today="2026-05-30",
                ohlcv_loader=_null_loader,
                outcome_path=outcome_path,
            )
            records = _load_jsonl(outcome_path)
            self.assertEqual(records[0]["suppression_outcome"], "neutral")

    def test_mfe_mae_none_without_ohlcv(self):
        with tempfile.TemporaryDirectory() as d:
            outcome_path = Path(d) / "outcomes.jsonl"
            _write_suppression_outcome(
                event=self._base_event(),
                final_return=0.083,
                phase_at_exit="breakout",
                today="2026-05-30",
                ohlcv_loader=_null_loader,
                outcome_path=outcome_path,
            )
            r = _load_jsonl(outcome_path)[0]
            self.assertIsNone(r["max_favorable_excursion_after_suppression"])
            self.assertIsNone(r["max_adverse_excursion_after_suppression"])

    def test_mfe_mae_computed_with_ohlcv(self):
        df = _make_ohlcv(
            ["2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30"],
            [1000, 1050, 980, 1030],
        )

        def loader(sym):
            return df

        with tempfile.TemporaryDirectory() as d:
            outcome_path = Path(d) / "outcomes.jsonl"
            _write_suppression_outcome(
                event=self._base_event(),
                final_return=0.083,
                phase_at_exit="breakout",
                today="2026-05-30",
                ohlcv_loader=loader,
                outcome_path=outcome_path,
            )
            r = _load_jsonl(outcome_path)[0]
            self.assertIsNotNone(r["max_favorable_excursion_after_suppression"])
            self.assertIsNotNone(r["max_adverse_excursion_after_suppression"])
            self.assertGreater(r["max_favorable_excursion_after_suppression"], 0)
            self.assertLess(r["max_adverse_excursion_after_suppression"], 0)

    def test_schema_version(self):
        with tempfile.TemporaryDirectory() as d:
            outcome_path = Path(d) / "outcomes.jsonl"
            _write_suppression_outcome(
                event=self._base_event(),
                final_return=0.083,
                phase_at_exit="breakout",
                today="2026-05-30",
                ohlcv_loader=_null_loader,
                outcome_path=outcome_path,
            )
            r = _load_jsonl(outcome_path)[0]
            self.assertEqual(r["schema_version"], "v1")


# ── TestDetectPhaseTransitions ────────────────────────────────────────────────

class TestDetectPhaseTransitions(unittest.TestCase):
    def _make_snapshots(self, records, path):
        for r in records:
            line = json.dumps(r) + "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    def test_compression_to_breakout(self):
        with tempfile.TemporaryDirectory() as d:
            snap_path = Path(d) / "snaps.jsonl"
            trans_path = Path(d) / "trans.jsonl"
            self._make_snapshots([
                {"symbol": "A.T", "date": "2026-05-25", "phase": "compression"},
                {"symbol": "A.T", "date": "2026-05-26", "phase": "compression"},
                {"symbol": "A.T", "date": "2026-05-27", "phase": "breakout"},
            ], snap_path)
            n = detect_phase_transitions(snap_path, trans_path, _null_loader, "2026-05-27")
            self.assertEqual(n, 1)
            records = _load_jsonl(trans_path)
            self.assertEqual(len(records), 1)
            r = records[0]
            self.assertEqual(r["from_phase"], "compression")
            self.assertEqual(r["to_phase"], "breakout")
            self.assertEqual(r["days_in_from_phase"], 2)

    def test_no_transition_same_phase(self):
        with tempfile.TemporaryDirectory() as d:
            snap_path = Path(d) / "snaps.jsonl"
            trans_path = Path(d) / "trans.jsonl"
            self._make_snapshots([
                {"symbol": "A.T", "date": "2026-05-25", "phase": "compression"},
                {"symbol": "A.T", "date": "2026-05-26", "phase": "compression"},
            ], snap_path)
            n = detect_phase_transitions(snap_path, trans_path, _null_loader, "2026-05-27")
            self.assertEqual(n, 0)

    def test_dedup_no_double_write(self):
        with tempfile.TemporaryDirectory() as d:
            snap_path = Path(d) / "snaps.jsonl"
            trans_path = Path(d) / "trans.jsonl"
            self._make_snapshots([
                {"symbol": "A.T", "date": "2026-05-25", "phase": "compression"},
                {"symbol": "A.T", "date": "2026-05-26", "phase": "breakout"},
            ], snap_path)
            detect_phase_transitions(snap_path, trans_path, _null_loader, "2026-05-27")
            n2 = detect_phase_transitions(snap_path, trans_path, _null_loader, "2026-05-28")
            self.assertEqual(n2, 0)

    def test_multiple_symbols(self):
        with tempfile.TemporaryDirectory() as d:
            snap_path = Path(d) / "snaps.jsonl"
            trans_path = Path(d) / "trans.jsonl"
            self._make_snapshots([
                {"symbol": "A.T", "date": "2026-05-25", "phase": "compression"},
                {"symbol": "A.T", "date": "2026-05-26", "phase": "breakout"},
                {"symbol": "B.T", "date": "2026-05-25", "phase": "neutral"},
                {"symbol": "B.T", "date": "2026-05-26", "phase": "distribution"},
            ], snap_path)
            n = detect_phase_transitions(snap_path, trans_path, _null_loader, "2026-05-27")
            self.assertEqual(n, 2)

    def test_empty_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            snap_path = Path(d) / "snaps.jsonl"
            trans_path = Path(d) / "trans.jsonl"
            n = detect_phase_transitions(snap_path, trans_path, _null_loader, "2026-05-27")
            self.assertEqual(n, 0)


# ── TestComputeSuppressionAttribution ────────────────────────────────────────

class TestComputeSuppressionAttribution(unittest.TestCase):
    def _write_outcomes(self, path, outcomes):
        for o in outcomes:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(o) + "\n")

    def test_empty_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as d:
            result = compute_suppression_attribution(Path(d) / "missing.jsonl")
            self.assertEqual(result, {})

    def test_all_alpha_extension(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "outcomes.jsonl"
            self._write_outcomes(path, [
                {
                    "suppression_outcome": "alpha_extension",
                    "return_if_exited_original": 0.02,
                    "return_after_suppression": 0.07,
                    "phase_at_suppression": "compression",
                    "suppression_days": 3,
                },
            ] * 3)
            result = compute_suppression_attribution(path)
            self.assertEqual(result["n_suppressions"], 3)
            self.assertEqual(result["alpha_extension_rate"], 1.0)
            self.assertEqual(result["loss_delay_rate"], 0.0)

    def test_mixed_outcomes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "outcomes.jsonl"
            self._write_outcomes(path, [
                {"suppression_outcome": "alpha_extension",
                 "return_if_exited_original": 0.02, "return_after_suppression": 0.06,
                 "phase_at_suppression": "compression", "suppression_days": 2},
                {"suppression_outcome": "loss_delay",
                 "return_if_exited_original": 0.02, "return_after_suppression": -0.04,
                 "phase_at_suppression": "neutral", "suppression_days": 4},
                {"suppression_outcome": "neutral",
                 "return_if_exited_original": 0.02, "return_after_suppression": 0.03,
                 "phase_at_suppression": "compression", "suppression_days": 1},
            ])
            result = compute_suppression_attribution(path)
            self.assertEqual(result["n_suppressions"], 3)
            self.assertAlmostEqual(result["alpha_extension_rate"], 1/3, places=3)
            self.assertAlmostEqual(result["loss_delay_rate"], 1/3, places=3)
            self.assertIsNotNone(result["avg_return_delta"])
            self.assertIn("compression", result["phase_breakdown"])

    def test_avg_return_delta_computed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "outcomes.jsonl"
            self._write_outcomes(path, [
                {"suppression_outcome": "alpha_extension",
                 "return_if_exited_original": 0.04, "return_after_suppression": 0.08,
                 "phase_at_suppression": "compression", "suppression_days": 3},
                {"suppression_outcome": "neutral",
                 "return_if_exited_original": 0.04, "return_after_suppression": 0.06,
                 "phase_at_suppression": "compression", "suppression_days": 2},
            ])
            result = compute_suppression_attribution(path)
            self.assertAlmostEqual(result["avg_return_delta"], 0.03, places=4)


# ── TestSufficiencyCheck ──────────────────────────────────────────────────────

class TestSufficiencyCheck(unittest.TestCase):
    def _write_outcomes(self, path, n_alpha, n_neutral, n_loss):
        for outcome, n in [("alpha_extension", n_alpha), ("neutral", n_neutral), ("loss_delay", n_loss)]:
            for _ in range(n):
                record = {
                    "suppression_outcome": outcome,
                    "return_if_exited_original": 0.0,
                    "return_after_suppression": 0.0,
                    "phase_at_suppression": "compression",
                    "suppression_days": 1,
                }
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

    def _write_transitions(self, path, n):
        for _ in range(n):
            record = {"symbol": "A.T", "transition_date": "2026-05-01",
                      "from_phase": "compression", "to_phase": "breakout",
                      "days_in_from_phase": 2, "subsequent_5d_return": None}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def test_all_criteria_met(self):
        with tempfile.TemporaryDirectory() as d:
            op = Path(d) / "outcomes.jsonl"
            tp = Path(d) / "trans.jsonl"
            self._write_outcomes(op, 5, 12, 3)
            self._write_transitions(tp, 10)
            result = compute_sufficiency_check(op, tp)
            self.assertTrue(result["sufficient"])
            self.assertEqual(result["reason"], "all criteria met")

    def test_insufficient_samples(self):
        with tempfile.TemporaryDirectory() as d:
            op = Path(d) / "outcomes.jsonl"
            tp = Path(d) / "trans.jsonl"
            self._write_outcomes(op, 2, 1, 1)  # only 4 total
            self._write_transitions(tp, 10)
            result = compute_sufficiency_check(op, tp)
            self.assertFalse(result["sufficient"])
            self.assertIn("n_suppressions", result["reason"])

    def test_insufficient_alpha_ext(self):
        with tempfile.TemporaryDirectory() as d:
            op = Path(d) / "outcomes.jsonl"
            tp = Path(d) / "trans.jsonl"
            self._write_outcomes(op, 2, 18, 0)  # only 2 alpha
            self._write_transitions(tp, 10)
            result = compute_sufficiency_check(op, tp)
            self.assertFalse(result["sufficient"])
            self.assertIn("n_alpha_extension", result["reason"])

    def test_insufficient_transitions(self):
        with tempfile.TemporaryDirectory() as d:
            op = Path(d) / "outcomes.jsonl"
            tp = Path(d) / "trans.jsonl"
            self._write_outcomes(op, 5, 12, 3)
            self._write_transitions(tp, 3)  # < 10
            result = compute_sufficiency_check(op, tp)
            self.assertFalse(result["sufficient"])
            self.assertIn("n_transitions", result["reason"])

    def test_details_structure(self):
        with tempfile.TemporaryDirectory() as d:
            op = Path(d) / "outcomes.jsonl"
            tp = Path(d) / "trans.jsonl"
            result = compute_sufficiency_check(op, tp)
            self.assertIn("details", result)
            self.assertIn("thresholds", result["details"])
            self.assertEqual(result["details"]["thresholds"]["min_samples"], SUFFICIENCY_MIN_SAMPLES)


# ── TestRunInternalLifecycle ──────────────────────────────────────────────────

class TestRunInternalLifecycle(unittest.TestCase):
    def _suppressed_signal(self, sym="6920.T", current_return=0.045):
        return {
            "symbol": sym,
            "exit_orchestrator_applied": True,
            "action": "HOLD",
            "currently_holding": True,
            "current_return": current_return,
            "compression_phase": "compression",
            "compression_score": 74.2,
            "hold_extension_days": 3,
            "exit_reason": "exit_orchestrator_suppress_premature_exit",
        }

    def _holding_signal(self, sym="6920.T", current_return=0.06):
        return {
            "symbol": sym,
            "currently_holding": True,
            "action": "HOLD",
            "exit_orchestrator_applied": False,
            "current_return": current_return,
            "compression_phase": "compression",
            "compression_score": 74.2,
        }

    def test_suppression_start_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            asp = Path(d) / "active.json"
            op  = Path(d) / "outcomes.jsonl"
            tp  = Path(d) / "trans.jsonl"
            sp  = Path(d) / "snaps.jsonl"

            result = _run_internal(
                signals=[self._suppressed_signal("A.T")],
                active_state_path=asp,
                outcome_path=op,
                transition_path=tp,
                compression_snapshots_path=sp,
                ohlcv_loader=_null_loader,
                today="2026-05-27",
            )
            self.assertEqual(result["n_suppression_starts"], 1)
            state = _load_active_state(asp)
            self.assertIn("A.T", state)
            self.assertEqual(state["A.T"]["state"], "active")

    def test_exit_detected_and_outcome_written(self):
        with tempfile.TemporaryDirectory() as d:
            asp = Path(d) / "active.json"
            op  = Path(d) / "outcomes.jsonl"
            tp  = Path(d) / "trans.jsonl"
            sp  = Path(d) / "snaps.jsonl"

            # Day 1: suppression fires
            _run_internal(
                signals=[self._suppressed_signal("B.T", current_return=0.045)],
                active_state_path=asp,
                outcome_path=op,
                transition_path=tp,
                compression_snapshots_path=sp,
                ohlcv_loader=_null_loader,
                today="2026-05-27",
            )

            # Day 2: still holding
            _run_internal(
                signals=[self._holding_signal("B.T", current_return=0.065)],
                active_state_path=asp,
                outcome_path=op,
                transition_path=tp,
                compression_snapshots_path=sp,
                ohlcv_loader=_null_loader,
                today="2026-05-28",
            )

            # Day 3: position exited (no longer in holding signals)
            result = _run_internal(
                signals=[],
                active_state_path=asp,
                outcome_path=op,
                transition_path=tp,
                compression_snapshots_path=sp,
                ohlcv_loader=_null_loader,
                today="2026-05-30",
            )

            self.assertEqual(result["n_exits_resolved"], 1)
            outcomes = _load_jsonl(op)
            self.assertEqual(len(outcomes), 1)
            r = outcomes[0]
            self.assertEqual(r["symbol"], "B.T")
            self.assertAlmostEqual(r["return_if_exited_original"], 0.045)
            self.assertAlmostEqual(r["return_after_suppression"], 0.065)  # last_seen_return
            self.assertEqual(r["suppression_days"], 3)

            # Active state should be cleared
            state = _load_active_state(asp)
            self.assertNotIn("B.T", state)

    def test_re_suppression_ignored_for_active_symbol(self):
        with tempfile.TemporaryDirectory() as d:
            asp = Path(d) / "active.json"
            op  = Path(d) / "outcomes.jsonl"
            tp  = Path(d) / "trans.jsonl"
            sp  = Path(d) / "snaps.jsonl"

            # Day 1: suppression fires
            _run_internal(
                signals=[self._suppressed_signal("C.T", current_return=0.04)],
                active_state_path=asp,
                outcome_path=op,
                transition_path=tp,
                compression_snapshots_path=sp,
                ohlcv_loader=_null_loader,
                today="2026-05-27",
            )

            # Day 2: suppression fires again for same symbol
            result = _run_internal(
                signals=[self._suppressed_signal("C.T", current_return=0.05)],
                active_state_path=asp,
                outcome_path=op,
                transition_path=tp,
                compression_snapshots_path=sp,
                ohlcv_loader=_null_loader,
                today="2026-05-28",
            )

            # Should NOT add a new start (already tracked)
            self.assertEqual(result["n_suppression_starts"], 0)
            state = _load_active_state(asp)
            # original_exit_date must remain the first day
            self.assertEqual(state["C.T"]["original_exit_date"], "2026-05-27")

    def test_fail_open_on_error(self):
        """Hook must not raise even if active state path is unwritable."""
        result = run_suppression_outcome_hook(
            signals=[],
            active_state_path=Path("/nonexistent_dir/active.json"),
            outcome_path=Path("/nonexistent_dir/outcomes.jsonl"),
            transition_path=Path("/nonexistent_dir/trans.jsonl"),
            compression_snapshots_path=Path("/nonexistent_dir/snaps.jsonl"),
            ohlcv_loader=_null_loader,
            today="2026-05-27",
        )
        # Should return {} (fail-open), not raise
        self.assertIsInstance(result, dict)


# ── TestOutcomeClassificationThresholds ──────────────────────────────────────

class TestOutcomeClassificationThresholds(unittest.TestCase):
    """Regression: threshold values match user specification."""

    def test_alpha_threshold_is_3pct(self):
        self.assertEqual(ALPHA_EXT_THRESHOLD, 0.03)

    def test_loss_threshold_is_minus_3pct(self):
        self.assertEqual(LOSS_DELAY_THRESHOLD, -0.03)

    def test_sufficiency_min_samples(self):
        self.assertEqual(SUFFICIENCY_MIN_SAMPLES, 20)

    def test_sufficiency_min_alpha_ext(self):
        self.assertEqual(SUFFICIENCY_MIN_ALPHA_EXT, 5)

    def test_sufficiency_min_transitions(self):
        self.assertEqual(SUFFICIENCY_MIN_PHASE_TRANSITIONS, 10)


# ── TestStateResolutionLabels ─────────────────────────────────────────────────

class TestStateResolutionLabels(unittest.TestCase):
    """Check that exited-in-window vs timeout labels are correct."""

    def _run_and_get_outcome(self, exit_day, hold_extension_days):
        with tempfile.TemporaryDirectory() as d:
            asp = Path(d) / "active.json"
            op  = Path(d) / "outcomes.jsonl"
            tp  = Path(d) / "trans.jsonl"
            sp  = Path(d) / "snaps.jsonl"

            # Suppression start
            _run_internal(
                signals=[{
                    "symbol": "X.T",
                    "exit_orchestrator_applied": True,
                    "action": "HOLD",
                    "currently_holding": True,
                    "current_return": 0.04,
                    "compression_phase": "compression",
                    "compression_score": 70.0,
                    "hold_extension_days": hold_extension_days,
                    "exit_reason": "exit_orchestrator_suppress_premature_exit",
                }],
                active_state_path=asp, outcome_path=op,
                transition_path=tp, compression_snapshots_path=sp,
                ohlcv_loader=_null_loader, today="2026-05-27",
            )

            # Exit on specified day
            _run_internal(
                signals=[],
                active_state_path=asp, outcome_path=op,
                transition_path=tp, compression_snapshots_path=sp,
                ohlcv_loader=_null_loader, today=exit_day,
            )

            records = _load_jsonl(op)
            return records[0]["state_at_resolution"] if records else None

    def test_success_exit_within_window(self):
        state = self._run_and_get_outcome("2026-05-29", hold_extension_days=3)  # 2 days
        self.assertEqual(state, "success_exit")

    def test_timeout_exit_beyond_window(self):
        state = self._run_and_get_outcome("2026-06-05", hold_extension_days=3)  # 9 days
        self.assertEqual(state, "timeout_exit")


if __name__ == "__main__":
    unittest.main()
