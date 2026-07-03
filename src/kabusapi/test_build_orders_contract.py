"""
src/kabusapi/test_build_orders_contract.py
_build_orders() 5-tuple 契約のリグレッションテスト。

実行:
    cd C:/ai-trading
    python -m pytest src/kabusapi/test_build_orders_contract.py -v
または単独:
    python src/kabusapi/test_build_orders_contract.py
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# プロジェクトルートを sys.path に追加
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── テスト用スタブ ──────────────────────────────────────────────────────────────
def _make_signal(symbol: str, signal: int, holding: bool = False, rsr: float = 80.0):
    s = MagicMock()
    s.symbol        = symbol
    s.signal        = signal
    s.currently_holding = holding
    s.rsr           = rsr
    s.rsr_rank      = 1
    s.sector        = "テスト"
    s.reason        = "unit-test"
    s.strategy_type = "turtle"
    return s


def _make_bridge():
    """最小限の SignalBridge インスタンスを生成する（API 呼び出しなし）。"""
    # 重いインポートをモックしてから本体をロード
    heavy = [
        "kabusapi", "kabu_station_api", "pandas_datareader",
    ]
    mocks = {m: MagicMock() for m in heavy}
    with patch.dict("sys.modules", mocks):
        from src.kabusapi.signal_bridge import SignalBridge, _validate_build_orders_contract

    bridge = MagicMock(spec=SignalBridge)
    bridge.max_positions           = 3
    bridge.min_sectors             = 2
    bridge.capital                 = 3_000_000
    bridge.max_single_weight       = 0.25
    bridge.regime_sizing           = "none"
    bridge.bear_scale              = 0.5
    bridge.max_new_positions_per_day = 2
    bridge.universe_tickers        = {}
    bridge.pre_trade_risk_check    = MagicMock(return_value=True)
    # 実メソッドを束縛
    bridge._build_orders = SignalBridge._build_orders.__get__(bridge, type(bridge))
    return bridge, _validate_build_orders_contract


class TestBuildOrdersContract(unittest.TestCase):

    def setUp(self):
        try:
            self.bridge, self.validate = _make_bridge()
        except Exception as e:
            self.skipTest(f"SignalBridge import unavailable in test env: {e}")

    # ── シナリオ 1: CB 発動 → BUY 全停止 ──────────────────────────────────────
    def test_cb_active_returns_4tuple(self):
        """CB 発動中でも _build_orders が 4 要素タプルを返すこと。"""
        buy_sig  = _make_signal("1234.T", signal=1)
        sell_sig = _make_signal("5678.T", signal=-1, holding=True)

        universe_raw = {
            "5678.T": {"df": MagicMock(**{"__getitem__.return_value": MagicMock(
                iloc=MagicMock(__getitem__=lambda s, i: 1000.0)
            )})},
        }
        current_positions = {
            "5678.T": {"qty": 100},
        }

        result = self.bridge._build_orders(
            signals=[buy_sig, sell_sig],
            universe_raw=universe_raw,
            current_positions=current_positions,
            available_cash=2_000_000,
            cb_active=True,
        )

        # 契約: 5-tuple
        self.assertIsInstance(result, tuple, "CB path must return a tuple")
        self.assertEqual(len(result), 5, "CB path must return exactly 5 values")

        orders, warnings, blocked_cap, lot_up, risk_rejected = result
        self.assertEqual(blocked_cap,    0, "CB path: blocked_alloc_cap must be 0")
        self.assertEqual(lot_up,         0, "CB path: lot_rounded_up must be 0")
        self.assertEqual(risk_rejected,  0, "CB path: risk_rejected must be 0")
        # BUY シグナルは含まれない
        buy_orders = [o for o in orders if getattr(o, "side", None) == "BUY"]
        self.assertEqual(len(buy_orders), 0, "CB path must emit no BUY orders")
        # CB 警告が含まれる
        self.assertTrue(any("サーキットブレーカー" in w for w in warnings))

    # ── シナリオ 2: 通常パス ────────────────────────────────────────────────────
    def test_normal_path_returns_5tuple(self):
        """通常パス（cb_active=False）でも 5 要素タプルを返すこと。"""
        result = self.bridge._build_orders(
            signals=[],
            universe_raw={},
            current_positions={},
            available_cash=3_000_000,
            cb_active=False,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

    # ── シナリオ 3: _validate_build_orders_contract ────────────────────────────
    def test_validator_rejects_4tuple(self):
        with self.assertRaises(ValueError):
            self.validate(([], [], 0, 0))  # 4要素 → ValueError

    def test_validator_rejects_non_tuple(self):
        with self.assertRaises(TypeError):
            self.validate([[], [], 0, 0, 0])  # list → TypeError

    def test_validator_accepts_5tuple(self):
        self.validate(([], [], 0, 0, 0))  # 正常 → 例外なし

    # ── シナリオ 4: DRY run 終了コード 0 シミュレーション ─────────────────────
    def test_cb_blocks_buy_dry_run_exits_zero(self):
        """CB 発動 + BUY シグナルあり → _build_orders が安全に返り、
        unpack クラッシュが発生しないことを検証する。"""
        buy_sigs = [_make_signal(f"{i:04d}.T", signal=1) for i in range(1, 4)]
        result = self.bridge._build_orders(
            signals=buy_sigs,
            universe_raw={},
            current_positions={},
            available_cash=3_000_000,
            cb_active=True,
        )
        # unpack が成功すれば ValueError は発生しない
        try:
            orders, order_warnings, _blocked, _lot, _risk = result
        except ValueError as e:
            self.fail(f"unpack raised ValueError: {e}")
        self.assertEqual(orders, [], "CB: no orders emitted")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
