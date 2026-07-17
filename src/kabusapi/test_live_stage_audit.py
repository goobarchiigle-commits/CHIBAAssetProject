"""
src/kabusapi/test_live_stage_audit.py

恒久Stage監査（2026-06-29 EVS RCA follow-up）の回帰テスト。
_build_orders() の audit_sink が実際のCAPACITY/CAPITAL/RISK/ORDER_BUILT判定を
正しく記録すること、かつ audit_sink=None（デフォルト）では一切挙動を
変えないことを検証する。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd  # noqa: F401 — pre-warm sys.modules (see test_shadow_capacity_guard.py)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _import_bridge_symbols():
    heavy = ["kabusapi", "kabu_station_api", "pandas_datareader"]
    mocks = {m: MagicMock() for m in heavy}
    with patch.dict("sys.modules", mocks):
        from src.kabusapi.signal_bridge import SignalBridge, StockSignal
    return SignalBridge, StockSignal


def _make_price_df(last_close: float, n: int = 40) -> pd.DataFrame:
    closes = [last_close * (1 + 0.001 * i) for i in range(n)]
    closes[-1] = last_close
    return pd.DataFrame({
        "Close": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes],
    })


def _make_bridge(SignalBridge):
    bridge = MagicMock(spec=SignalBridge)
    bridge.max_positions             = 3
    bridge.min_sectors                = 1
    bridge.capital                    = 3_000_000
    bridge.max_single_weight          = 0.25
    bridge.regime_sizing               = "none"
    bridge.bear_scale                  = 1.0
    bridge.max_new_positions_per_day  = 2
    bridge.top_k                      = 3
    bridge.universe_tickers            = {}
    bridge.entry_freeze_enabled         = False   # 2026-07-17 Entry Freeze Mode追加分（既定=無効）
    bridge.entry_freeze_reason          = "Research Freeze"
    bridge.pre_trade_risk_check        = MagicMock(return_value=True)
    bridge._build_orders = SignalBridge._build_orders.__get__(bridge, type(bridge))
    return bridge


class TestBuildOrdersStageAudit(unittest.TestCase):

    def setUp(self):
        self.SignalBridge, self.StockSignal = _import_bridge_symbols()

    def _sig(self, symbol, rsr, rank, holding=False):
        return self.StockSignal(
            symbol=symbol, sector="テスト", signal=1, rsr=rsr, rsr_rank=rank,
            sepa_score=8, rsr_mom=0.0, hold_days=0, currently_holding=holding,
            reason="test", strategy_type="fujiko",
        )

    def test_2026_06_29_capacity_full_recorded_for_all_candidates(self):
        """held=3=max_positionsの状態で2候補を評価 → 両方CAPACITY:FAILが記録される
        （2026-06-29の8035.T/6920.Tと同型の状況）。"""
        bridge = _make_bridge(self.SignalBridge)
        signals = [self._sig("8035.T", 96.4, 2), self._sig("6920.T", 94.6, 3)]
        current_positions = {"6981.T": {"qty": 100}, "5301.T": {"qty": 100}, "2802.T": {"qty": 100}}
        universe_raw = {
            "8035.T": {"df": _make_price_df(72320.0)},
            "6920.T": {"df": _make_price_df(3000.0)},
        }
        audit_sink: list = []
        bridge._build_orders(
            signals=signals, universe_raw=universe_raw, current_positions=current_positions,
            available_cash=1_872_291.0, cb_active=False, effective_max_pos=3,
            audit_sink=audit_sink,
        )
        capacity_fails = {d["symbol"] for d in audit_sink if d["stage"] == "CAPACITY" and not d["passed"]}
        self.assertEqual(capacity_fails, {"8035.T", "6920.T"})
        for d in audit_sink:
            if d["stage"] == "CAPACITY" and not d["passed"]:
                self.assertEqual(d["reason"], "position_full")
                self.assertEqual(d["held"], 3)
                self.assertEqual(d["max_positions"], 3)

    def test_successful_buy_records_full_stage_chain(self):
        """空き枠ありで正常に発注できた候補は CAPACITY/CAPITAL/RISK/ORDER_BUILT
        すべてpassed=Trueで記録される。"""
        bridge = _make_bridge(self.SignalBridge)
        signals = [self._sig("7203.T", 90.0, 1)]
        universe_raw = {"7203.T": {"df": _make_price_df(3000.0)}}
        audit_sink: list = []
        orders, *_ = bridge._build_orders(
            signals=signals, universe_raw=universe_raw, current_positions={},
            available_cash=3_000_000.0, cb_active=False, effective_max_pos=3,
            audit_sink=audit_sink,
        )
        self.assertEqual(len(orders), 1)
        stages_passed = {d["stage"] for d in audit_sink if d["symbol"] == "7203.T" and d["passed"]}
        self.assertIn("CAPACITY", stages_passed)
        self.assertIn("CAPITAL", stages_passed)
        self.assertIn("RISK", stages_passed)
        self.assertIn("ORDER_BUILT", stages_passed)

    def test_audit_sink_none_does_not_change_behavior(self):
        """audit_sink未指定（デフォルトNone）でも従来どおり動作し例外を出さない。"""
        bridge = _make_bridge(self.SignalBridge)
        signals = [self._sig("7203.T", 90.0, 1)]
        universe_raw = {"7203.T": {"df": _make_price_df(3000.0)}}
        result = bridge._build_orders(
            signals=signals, universe_raw=universe_raw, current_positions={},
            available_cash=3_000_000.0, cb_active=False, effective_max_pos=3,
        )
        self.assertEqual(len(result), 5)
        orders = result[0]
        self.assertEqual(len(orders), 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
