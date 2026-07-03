"""
test_pipeline_audit.py
Task 3: Strategy pipeline audit tests.
Verifies the filter funnel behavior, dyn filter fallback logic,
and the BUY_ZERO upstream diagnostic path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest


class TestDynFilterFallback(unittest.TestCase):
    """
    The dyn filter has two distinct zero-candidate paths:
      A) upstream already empty  → BUY_ZERO upstream log, NO fallback
      B) dyn removed all candidates → fallback to _buy_eligible_all
    """

    def _apply_dyn_filter(self, buy_eligible_all: list, dyn_active_syms: set) -> list:
        """Replicate the signal_bridge dyn filter logic."""
        if dyn_active_syms:
            buy_eligible = [
                (sc, sym) for sc, sym in buy_eligible_all if sym in dyn_active_syms
            ]
            if not buy_eligible:
                if not buy_eligible_all:
                    # upstream zero → no fallback
                    return []
                else:
                    # dyn removed all → fallback
                    return buy_eligible_all
            return buy_eligible
        else:
            return buy_eligible_all

    def test_upstream_zero_stays_zero_no_fallback(self):
        """When upstream is already empty, dyn filter must not invent candidates."""
        result = self._apply_dyn_filter([], dyn_active_syms={"1234.T", "5678.T"})
        self.assertEqual(result, [])

    def test_dyn_filter_removes_non_active_candidates(self):
        """Candidates not in dyn_active_syms must be excluded."""
        all_cands = [(80.0, "9104.T"), (75.0, "5401.T")]
        active    = {"9104.T"}   # 5401.T not active
        result    = self._apply_dyn_filter(all_cands, active)
        syms      = [sym for _, sym in result]
        self.assertIn("9104.T", syms)
        self.assertNotIn("5401.T", syms)

    def test_dyn_filter_fallback_when_all_removed(self):
        """If dyn removes ALL upstream candidates, fall back to full upstream list."""
        all_cands = [(80.0, "9104.T"), (75.0, "5401.T")]
        active    = {"9999.T"}   # neither candidate is active
        result    = self._apply_dyn_filter(all_cands, active)
        # fallback: returns original upstream list
        self.assertEqual(result, all_cands)

    def test_empty_dyn_active_syms_bypasses_filter(self):
        """Empty dyn_active_syms (calculation failure) → no filtering applied."""
        all_cands = [(80.0, "9104.T"), (75.0, "5401.T")]
        result    = self._apply_dyn_filter(all_cands, dyn_active_syms=set())
        self.assertEqual(result, all_cands)

    def test_full_overlap_preserves_all_candidates(self):
        """All candidates in active set → all preserved."""
        all_cands = [(80.0, "9104.T"), (75.0, "5401.T")]
        active    = {"9104.T", "5401.T", "9999.T"}
        result    = self._apply_dyn_filter(all_cands, active)
        self.assertEqual(result, all_cands)


class TestFilterFunnelLogic(unittest.TestCase):
    """
    Verify the BUY candidate funnel stages:
      signal==1 AND not currently_holding → _buy_eligible_all
      → dyn filter → top_k
    """

    def test_currently_holding_excluded_from_eligible(self):
        """Held symbols with signal==1 are HOLD_CONTINUE, not new BUY candidates."""
        signals = [
            {"symbol": "9104.T", "signal": 1, "currently_holding": True,  "rsr": 82.0, "rsr_momentum": 1.5},
            {"symbol": "5401.T", "signal": 1, "currently_holding": False, "rsr": 76.0, "rsr_momentum": 0.5},
        ]
        mom_weight = 0.3
        eligible = sorted(
            [(s["rsr"] + s["rsr_momentum"] * mom_weight, s["symbol"])
             for s in signals if s["signal"] == 1 and not s["currently_holding"]],
            reverse=True,
        )
        syms = [sym for _, sym in eligible]
        self.assertNotIn("9104.T", syms, "held symbol must not appear in eligible BUY list")
        self.assertIn("5401.T", syms, "non-held BUY signal must be eligible")

    def test_hold_continue_signals_separate_from_new_buy(self):
        """HOLD_CONTINUE and new BUY must be separate in print output logic."""
        signals = [
            {"symbol": "9104.T", "signal": 1, "currently_holding": True},
            {"symbol": "5401.T", "signal": 1, "currently_holding": False},
            {"symbol": "5411.T", "signal": 0, "currently_holding": False},
        ]
        hold_continue = [s for s in signals if s["signal"] == 1 and s.get("currently_holding")]
        new_buy       = [s for s in signals if s["signal"] == 1 and not s.get("currently_holding")]

        self.assertEqual(len(hold_continue), 1)
        self.assertEqual(hold_continue[0]["symbol"], "9104.T")
        self.assertEqual(len(new_buy), 1)
        self.assertEqual(new_buy[0]["symbol"], "5401.T")

    def test_sell_signal_for_held_not_confused_with_hold_continue(self):
        """SELL signal for held symbol must not appear as HOLD_CONTINUE."""
        signals = [
            {"symbol": "9104.T", "signal": -1, "currently_holding": True},
        ]
        hold_continue = [s for s in signals if s["signal"] == 1 and s.get("currently_holding")]
        self.assertEqual(hold_continue, [])


class TestBearRegimeFallback(unittest.TestCase):
    """
    In Bear mode with all rs_topix <= 0, the rs>0 filter falls back
    to score_bear.dropna() so that some active symbols are always selected.
    """

    def test_bear_fallback_when_all_rs_negative(self):
        """When all symbols have rs_topix <= 0, fallback keeps all symbols."""
        import pandas as pd
        rs_topix   = pd.Series({"A": -0.05, "B": -0.02, "C": -0.08})
        score_bear = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5})

        score_valid = score_bear[rs_topix > 0].dropna()
        if len(score_valid) == 0:
            score_valid = score_bear.dropna()   # fallback

        self.assertEqual(len(score_valid), 3, "fallback must keep all symbols when all rs<0")

    def test_bear_rs_filter_keeps_positive_only(self):
        """When some symbols have positive RS, only those are selected."""
        import pandas as pd
        rs_topix   = pd.Series({"A": 0.03, "B": -0.02, "C": 0.01})
        score_bear = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5})

        score_valid = score_bear[rs_topix > 0].dropna()

        self.assertIn("A", score_valid.index)
        self.assertIn("C", score_valid.index)
        self.assertNotIn("B", score_valid.index)
