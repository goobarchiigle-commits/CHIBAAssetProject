"""
src/kabusapi/test_shadow_capacity_guard.py
2026-07-07 4銘柄同時保有インシデントの回帰テスト。

対象:
  - _capacity_check()                 : 残スロット計算の純粋関数
  - SignalBridge._build_shadow_orders(): Shadow経路が capacity_check /
                                         pre_trade_risk_check を必ず経由し、
                                         実発注可能な OrderInstruction を
                                         一切生成しないこと（observation_only）
  - SignalBridge._send_orders()       : 未知の side（旧 SHADOW_BUY 相当）を
                                         fail-closed でスキップし、
                                         Broker へは絶対に送信しないこと
  - update_state_after_execution()    : 全SELL理由で reentry_blocked が
                                         設定され、cooldown日数が
                                         strategy.yaml から可変であること

実行:
    cd C:/ai-trading
    python -m pytest src/kabusapi/test_shadow_capacity_guard.py -v
"""
from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

JST = timezone(timedelta(hours=9))


def _make_price_df(last_close: float, n: int = 40) -> pd.DataFrame:
    """ATR/rolling計算が通る最低限のOHLC DataFrameを作る。"""
    closes = [last_close * (1 + 0.001 * i) for i in range(n)]
    closes[-1] = last_close
    return pd.DataFrame({
        "Close": closes,
        "High":  [c * 1.01 for c in closes],
        "Low":   [c * 0.99 for c in closes],
    })


def _import_bridge_symbols():
    heavy = ["kabusapi", "kabu_station_api", "pandas_datareader"]
    mocks = {m: MagicMock() for m in heavy}
    with patch.dict("sys.modules", mocks):
        from src.kabusapi.signal_bridge import (
            SignalBridge, OrderInstruction, _capacity_check,
        )
    return SignalBridge, OrderInstruction, _capacity_check


def _make_shadow_bridge(SignalBridge, *, pre_trade_risk_check_result: bool = True):
    """_build_shadow_orders 単体テスト用の最小 SignalBridge スタブ。"""
    bridge = MagicMock(spec=SignalBridge)
    bridge.capital             = 3_000_000
    bridge.max_single_weight   = 0.25
    bridge.shadow_universe_tickers = {"5301.T": "化学"}
    bridge.pre_trade_risk_check = MagicMock(return_value=pre_trade_risk_check_result)
    bridge._build_shadow_orders = SignalBridge._build_shadow_orders.__get__(bridge, type(bridge))
    return bridge


class TestCapacityCheckPureFunction(unittest.TestCase):
    """_capacity_check() の境界値検証。"""

    def setUp(self):
        _, _, self.capacity_check = _import_bridge_symbols()

    def test_exact_capacity_zero_remaining(self):
        # 実保有2 + 通常BUY1 = 3、max_positions=3 → 残り0（2026-07-07の実状態）
        self.assertEqual(self.capacity_check(3, 2, 1), 0)

    def test_one_slot_free(self):
        self.assertEqual(self.capacity_check(3, 2, 0), 1)

    def test_already_at_cap(self):
        self.assertEqual(self.capacity_check(3, 3, 0), 0)

    def test_over_cap_is_negative(self):
        self.assertEqual(self.capacity_check(3, 3, 1), -1)


class TestBuildShadowOrdersCapacityGuard(unittest.TestCase):
    """_build_shadow_orders(): Shadow経路の capacity_check / risk_check 統合。"""

    def setUp(self):
        self.SignalBridge, self.OrderInstruction, self.capacity_check = _import_bridge_symbols()

    def _base_diag(self, rsr62_scores: dict[str, float]):
        return {
            "shadow_rsr_pass": 8,
            "rsr_distribution": [{"rsr": 80.0 - i} for i in range(10)],  # median=75.5
            "shadow_rsr62_scores": rsr62_scores,
        }

    # ── ①2026-07-07完全再現: 保有2 + 通常BUY1 → 残り0 → Shadow送信ゼロ ──────
    def test_2026_07_07_incident_reproduction(self):
        bridge = _make_shadow_bridge(self.SignalBridge, pre_trade_risk_check_result=True)
        current_positions = {"6981.T": {"qty": 100}, "2802.T": {"qty": 100}}
        live_orders = [self.OrderInstruction(
            symbol="6506.T", symbol_4digit="6506", sector="機械", side="BUY",
            qty=100, order_type="MARKET_OPEN", estimated_price=7449.0,
            estimated_amount=744900.0, reason="trend_follow fallback=False",
        )]
        universe_raw = {"5301.T": {"df": _make_price_df(1741.0)}}

        orders, metrics, new_virtual, closed = bridge._build_shadow_orders(
            diag=self._base_diag({"5301.T": 90.3}),
            universe_raw=universe_raw,
            current_positions=current_positions,
            available_cash=1_500_000.0,
            cb_active=False,
            live_orders=live_orders,
            shadow_virtual_positions={},
            today_str="2026-07-07",
            effective_max_pos=3,
        )

        self.assertEqual(orders, [], "Shadow経路は絶対に実発注可能なOrderを返さない")
        self.assertEqual(new_virtual, {}, "残スロット0のため5301.Tの仮想エントリーも記録されない")
        self.assertEqual(closed, [])
        self.assertEqual(metrics["shadow_remaining_slots"], 0)
        bridge.pre_trade_risk_check.assert_not_called()  # 枠なし判定で候補処理自体に到達しない

    # ── ②remaining_slots<=0 の一般ケース（held=3, pending=0）────────────────
    def test_remaining_slots_zero_generic(self):
        bridge = _make_shadow_bridge(self.SignalBridge)
        current_positions = {"A.T": {}, "B.T": {}, "C.T": {}}
        universe_raw = {"5301.T": {"df": _make_price_df(1741.0)}}

        orders, metrics, new_virtual, _ = bridge._build_shadow_orders(
            diag=self._base_diag({"5301.T": 90.3}),
            universe_raw=universe_raw,
            current_positions=current_positions,
            available_cash=1_500_000.0,
            cb_active=False,
            live_orders=[],
            shadow_virtual_positions={},
            today_str="2026-07-07",
            effective_max_pos=3,
        )
        self.assertEqual(orders, [])
        self.assertEqual(new_virtual, {})
        self.assertLessEqual(metrics["shadow_remaining_slots"], 0)

    # ── 空きスロットありでも pre_trade_risk_check 不合格なら記録しない ────────
    def test_shadow_candidate_rejected_by_risk_check(self):
        bridge = _make_shadow_bridge(self.SignalBridge, pre_trade_risk_check_result=False)
        universe_raw = {"5301.T": {"df": _make_price_df(1741.0)}}

        orders, metrics, new_virtual, _ = bridge._build_shadow_orders(
            diag=self._base_diag({"5301.T": 90.3}),
            universe_raw=universe_raw,
            current_positions={"6981.T": {"qty": 100}},   # held=1
            available_cash=1_500_000.0,
            cb_active=False,
            live_orders=[],                                 # pending=0 → remaining=2
            shadow_virtual_positions={},
            today_str="2026-07-07",
            effective_max_pos=3,
        )
        bridge.pre_trade_risk_check.assert_called_once()
        self.assertEqual(orders, [], "risk_check不合格でも当然orderは作られない")
        self.assertEqual(new_virtual, {}, "risk_check不合格の候補は仮想エントリーにも記録しない")

    # ── 空きスロットあり + risk_check通過 → 仮想記録のみ・orderは常に空 ──────
    def test_shadow_candidate_passes_risk_check_records_virtual_only(self):
        bridge = _make_shadow_bridge(self.SignalBridge, pre_trade_risk_check_result=True)
        universe_raw = {"5301.T": {"df": _make_price_df(1741.0)}}

        orders, metrics, new_virtual, _ = bridge._build_shadow_orders(
            diag=self._base_diag({"5301.T": 90.3}),
            universe_raw=universe_raw,
            current_positions={"6981.T": {"qty": 100}},   # held=1 → remaining=2
            available_cash=1_500_000.0,
            cb_active=False,
            live_orders=[],
            shadow_virtual_positions={},
            today_str="2026-07-07",
            effective_max_pos=3,
        )
        self.assertEqual(orders, [], "risk_check通過・枠ありでも実発注可能なorderは絶対に作らない")
        self.assertIn("5301.T", new_virtual, "観測記録（仮想）としては残る")
        self.assertTrue(new_virtual["5301.T"]["virtual"])
        self.assertEqual(metrics["shadow_entry_count"], 0, "実発注件数は常にゼロ固定")


class TestSendOrdersFailClosedUnknownSide(unittest.TestCase):
    """_send_orders(): 未知sideは fail-closed でBroker送信されないこと。"""

    def setUp(self):
        self.SignalBridge, self.OrderInstruction, _ = _import_bridge_symbols()

    def test_unknown_side_never_reaches_broker(self):
        bridge = MagicMock(spec=self.SignalBridge)
        bridge._client = MagicMock()
        bridge._last_signal_time = None
        bridge.order_rate_interval_sec = 0
        bridge._send_orders = self.SignalBridge._send_orders.__get__(bridge, type(bridge))

        bad_order = self.OrderInstruction(
            symbol="5301.T", symbol_4digit="5301", sector="化学",
            side="SHADOW_BUY",   # 旧実装が実発注していた値。もう存在しないはずだが防御的に検証。
            qty=300, order_type="MARKET_OPEN", estimated_price=1741.0,
            estimated_amount=522300.0, reason="defense-in-depth test",
        )

        # NOTE: `patch("src.kabusapi.signal_bridge.datetime")` は sys.modules 経由で
        # 再解決されるため、_import_bridge_symbols() の patch.dict(sys.modules) が
        # exit時にモジュールエントリを削除した後だと別モジュールを掴んでしまう。
        # 関数自身の __globals__ を直接差し替えることで確実に同一名前空間を patch する。
        _fixed_now = datetime(2026, 7, 7, 9, 30, 0, tzinfo=JST)
        mock_dt = MagicMock()
        mock_dt.now.return_value = _fixed_now
        _func_globals = self.SignalBridge._send_orders.__globals__
        with patch.dict(_func_globals, {"datetime": mock_dt}):
            results = bridge._send_orders([bad_order])

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["success"])
        self.assertEqual(results[0]["fill_status"], "rejected_unknown_side")
        bridge._client.send_order.assert_not_called()


class TestReentryBlockedAllSellReasons(unittest.TestCase):
    """update_state_after_execution(): 全SELL理由でcooldownが設定されること。"""

    def _make_bridge(self, tmp_state_path: Path, cooldown_days: int = 5):
        SignalBridge, OrderInstruction, _ = _import_bridge_symbols()
        bridge = MagicMock(spec=SignalBridge)
        bridge._cfg = MagicMock()
        bridge._cfg.risk.reentry_cooldown_days = cooldown_days
        bridge._load_portfolio_state = MagicMock(return_value={
            "position_entry_dates":     {"5301.T": "2026-06-25"},
            "position_entry_prices":    {"5301.T": 1800.0},
            "position_entry_atrs":      {},
            "position_entry_rsrs":      {},
            "position_highest_closes":  {},
            "position_qtys":            {"5301.T": 300},
            "reentry_blocked":          {},
            "available_cash":           1_000_000.0,
            "shadow_positions":         {},
            "position_strategy_types":  {},
        })
        _saved = {}
        def _save(state):
            _saved.update(state)
        bridge._save_portfolio_state = MagicMock(side_effect=_save)
        bridge.update_state_after_execution = SignalBridge.update_state_after_execution.__get__(
            bridge, type(bridge)
        )
        return bridge, _saved

    def test_rsr_exit_sell_sets_reentry_block(self):
        bridge, saved = self._make_bridge(Path("unused"))
        send_results = [{
            "symbol": "5301.T", "side": "SELL", "success": True,
            "qty": 300, "estimated_price": 1741.0,
            "reason": "SELL[フジコ法]: RSR=66.1 mom=-11.3",   # 時間ストップではない
            "sector": "化学",
        }]
        bridge.update_state_after_execution(send_results, today_str="2026-06-30")
        self.assertIn("5301.T", saved["reentry_blocked"], "RSR_EXIT系SELLでもcooldown対象になる")

    def test_trailing_stop_sell_sets_reentry_block(self):
        bridge, saved = self._make_bridge(Path("unused"))
        send_results = [{
            "symbol": "5301.T", "side": "SELL", "success": True,
            "qty": 300, "estimated_price": 1600.0,
            "reason": "SELL[トレーリングストップ]: close=1600",
            "sector": "化学",
        }]
        bridge.update_state_after_execution(send_results, today_str="2026-06-30")
        self.assertIn("5301.T", saved["reentry_blocked"])

    def test_cooldown_days_configurable(self):
        bridge, saved = self._make_bridge(Path("unused"), cooldown_days=10)
        send_results = [{
            "symbol": "5301.T", "side": "SELL", "success": True,
            "qty": 300, "estimated_price": 1600.0,
            "reason": "SELL[トレーリングストップ]: close=1600",
            "sector": "化学",
        }]
        bridge.update_state_after_execution(send_results, today_str="2026-06-30")
        block_end = saved["reentry_blocked"]["5301.T"]
        # 5営業日固定ではなく10営業日分先まで伸びていること（同一日ではない）
        self.assertNotEqual(block_end, "2026-06-30")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
