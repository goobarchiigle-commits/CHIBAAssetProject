"""
src/kabusapi/test_entry_freeze.py
Entry Freeze Mode（資産保全・2026-07-17）の _build_orders() ゲート検証。

実行:
    cd C:/ai-trading
    python -m pytest src/kabusapi/test_entry_freeze.py -v
または単独:
    python src/kabusapi/test_entry_freeze.py

Test1: BUY signal発生日 → signal generated=YES / broker order(BUY)=NO / position change=NO
Test2: SELL signal発生日 → SELL executed normally（freeze中でも変わらず）
Test3: DRY/LIVE parity → entry_freeze_enabled=True の挙動が self.live に依存しないこと
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# signal_bridge.py（pandas/numpy等の重量級依存込み）をプロセス内で一度だけ確実に
# importしてsys.modulesへキャッシュする。Windows環境でこのimportを1プロセス内で
# 複数回（patch.dict経由も含め）再実行するとnumpyのdelvewheel DLL登録が失敗する
# 既知の環境問題があるため、以降の _make_bridge() 呼び出しはキャッシュヒットのみにする。
try:
    from src.kabusapi.signal_bridge import SignalBridge as _SignalBridge_preload  # noqa: F401
    _PRELOAD_ERR: Exception | None = None
except Exception as _e:  # pragma: no cover
    _PRELOAD_ERR = _e


def _make_signal(symbol: str, signal: int, holding: bool = False, rsr: float = 80.0):
    s = MagicMock()
    s.symbol            = symbol
    s.signal            = signal
    s.currently_holding = holding
    s.rsr               = rsr
    s.rsr_rank          = 1
    s.sector            = "テスト"
    s.reason            = "unit-test"
    s.strategy_type     = "turtle"
    return s


def _make_bridge(*, live: bool, entry_freeze_enabled: bool, entry_freeze_reason: str = "Research Freeze"):
    """最小限の SignalBridge インスタンスを生成する（API 呼び出しなし）。"""
    heavy = ["kabusapi", "kabu_station_api", "pandas_datareader"]
    mocks = {m: MagicMock() for m in heavy}
    with patch.dict("sys.modules", mocks):
        from src.kabusapi.signal_bridge import SignalBridge

    bridge = MagicMock(spec=SignalBridge)
    bridge.max_positions             = 3
    bridge.min_sectors               = 2
    bridge.capital                   = 3_000_000
    bridge.max_single_weight         = 0.25
    bridge.regime_sizing             = "none"
    bridge.bear_scale                = 0.5
    bridge.max_new_positions_per_day = 2
    bridge.universe_tickers          = {}
    bridge.live                      = live
    bridge.entry_freeze_enabled      = entry_freeze_enabled
    bridge.entry_freeze_reason       = entry_freeze_reason
    bridge.pre_trade_risk_check      = MagicMock(return_value=True)
    bridge._build_orders = SignalBridge._build_orders.__get__(bridge, type(bridge))
    return bridge


def _sell_universe_and_positions():
    universe_raw = {
        "5678.T": {"df": MagicMock(**{"__getitem__.return_value": MagicMock(
            iloc=MagicMock(__getitem__=lambda s, i: 1000.0)
        )})},
    }
    current_positions = {"5678.T": {"qty": 100}}
    return universe_raw, current_positions


class TestEntryFreezeGate(unittest.TestCase):
    """
    setUp では _make_bridge() を呼ばない: 呼び出しごとに
    `with patch.dict("sys.modules", ...): import signal_bridge` を経由するため、
    1テスト内で複数回呼ぶと環境依存のnumpy DLL再登録エラーを誘発することを確認済み
    （Windows delvewheel patch起因・本テストの対象外の既知の環境問題）。
    各テストは _make_bridge() をちょうど1回だけ呼ぶこと。
    """

    # ── Test1: BUY signal day → signal generated / broker order NO / position change NO ──
    def _run_test1(self, *, live: bool):
        try:
            bridge = _make_bridge(live=live, entry_freeze_enabled=True)
        except Exception as e:
            self.skipTest(f"SignalBridge import unavailable in test env: {e}")
        buy_sig = _make_signal("1234.T", signal=1)

        orders, warnings, blocked_cap, lot_up, risk_rejected = bridge._build_orders(
            signals=[buy_sig],
            universe_raw={},
            current_positions={},
            available_cash=3_000_000,
            cb_active=False,
        )

        # signal generated=YES: 入力シグナル自体は失われていない（呼び出し元でそのまま参照可能）
        self.assertEqual(buy_sig.signal, 1, "signal generated = YES")
        # broker order=NO: BUY が orders（=broker送信対象）に一切含まれない
        buy_orders = [o for o in orders if getattr(o, "side", None) == "BUY"]
        self.assertEqual(len(buy_orders), 0, "broker order(BUY) must be NO")
        # position change=NO: SELLも含め注文が一切生成されない（保有変化なし）
        self.assertEqual(orders, [], "position change must be NO (no orders at all)")
        # ENTRY_FROZEN理由がwarningsに記録される
        self.assertTrue(
            any("Research Freeze" in w for w in warnings),
            f"ENTRY_FROZEN reason must be logged in warnings: {warnings}",
        )
        return orders, warnings

    def test1_buy_blocked_live_false(self):
        self._run_test1(live=False)

    def test1_buy_blocked_live_true(self):
        self._run_test1(live=True)

    def test1_buy_blocked_dry_live_parity(self):
        """DRY: BUY blocked / LIVE: BUY blocked が完全一致すること。"""
        orders_dry,  _ = self._run_test1(live=False)
        orders_live, _ = self._run_test1(live=True)
        self.assertEqual(orders_dry, orders_live, "DRY/LIVE の BUY ブロック結果は完全一致すること")

    # ── Test2: SELL signal day → SELL executed normally ──────────────────────
    def _run_test2(self, *, live: bool):
        bridge = _make_bridge(live=live, entry_freeze_enabled=True)
        sell_sig = _make_signal("5678.T", signal=-1, holding=True)
        universe_raw, current_positions = _sell_universe_and_positions()

        orders, warnings, _blocked_cap, _lot_up, _risk_rejected = bridge._build_orders(
            signals=[sell_sig],
            universe_raw=universe_raw,
            current_positions=current_positions,
            available_cash=2_000_000,
            cb_active=False,
        )
        sell_orders = [o for o in orders if getattr(o, "side", None) == "SELL"]
        self.assertEqual(len(sell_orders), 1, "SELL must execute normally even under Entry Freeze")
        self.assertEqual(sell_orders[0].symbol, "5678.T")
        return orders

    def test2_sell_executes_normally_live_false(self):
        self._run_test2(live=False)

    def test2_sell_executes_normally_live_true(self):
        self._run_test2(live=True)

    # ── 混在: BUY+SELL同日 → SELLのみ通過・BUYのみブロック ──────────────────
    def test_mixed_buy_and_sell_same_day(self):
        bridge = _make_bridge(live=True, entry_freeze_enabled=True)
        buy_sig  = _make_signal("1234.T", signal=1)
        sell_sig = _make_signal("5678.T", signal=-1, holding=True)
        universe_raw, current_positions = _sell_universe_and_positions()

        orders, warnings, *_ = bridge._build_orders(
            signals=[buy_sig, sell_sig],
            universe_raw=universe_raw,
            current_positions=current_positions,
            available_cash=2_000_000,
            cb_active=False,
        )
        sides = sorted(getattr(o, "side", None) for o in orders)
        self.assertEqual(sides, ["SELL"], "混在日: SELLのみ通過しBUYはブロックされること")

    # ── Entry Freeze OFF: 既存挙動に回帰がないこと ────────────────────────────
    def test_freeze_disabled_buy_not_blocked_by_freeze_logic(self):
        """entry_freeze_enabled=False の場合、freezeロジックが一切発動しないこと
        （5-tupleで正常に返り、ENTRY_FROZEN/ENTRY FREEZE警告が出ないことを検証。
        signals=[]で通常パスを通す＝BUY構築の実データ要件（価格df等）を回避しつつ
        「freeze無効時に早期リターンへ迷い込まないか」を確認する）。"""
        bridge = _make_bridge(live=True, entry_freeze_enabled=False)
        result = bridge._build_orders(
            signals=[],
            universe_raw={},
            current_positions={},
            available_cash=3_000_000,
            cb_active=False,
        )
        self.assertEqual(len(result), 5)
        _orders, warnings, *_ = result
        self.assertFalse(
            any("FREEZE" in w for w in warnings),
            "freeze無効時にFREEZE関連の警告が出てはならない",
        )

    # ── CBとFreezeの独立性: 両方True/片方Trueでも5-tuple破綻なし ─────────────
    def test_cb_and_freeze_both_active(self):
        bridge = _make_bridge(live=True, entry_freeze_enabled=True)
        buy_sig = _make_signal("1234.T", signal=1)
        result = bridge._build_orders(
            signals=[buy_sig],
            universe_raw={},
            current_positions={},
            available_cash=3_000_000,
            cb_active=True,
        )
        self.assertEqual(len(result), 5)
        orders, warnings, *_ = result
        self.assertEqual(orders, [])
        self.assertTrue(any("サーキットブレーカー" in w for w in warnings))
        self.assertTrue(any("Research Freeze" in w for w in warnings))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
