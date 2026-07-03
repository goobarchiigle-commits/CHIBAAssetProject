"""
tests/test_sector_gate.py

セクター集中ゲートのテスト（pre_trade_risk_check 内の sector_concentration チェック）

背景:
  5401.T（日本製鉄）+ 5411.T（JFE）鉄鋼同時保有 → 相関0.85〜0.95で実質レバレッジ。
  max_names_per_sector=1 ゲートにより2銘柄目をブロックする。

実行:
  cd C:/ai-trading
  python -m unittest tests.test_sector_gate -v
"""
from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def _make_cfg(enabled: bool = True, max_names: int = 1, max_weight: float = 0.12):
    from src.config_loader import SectorConcentrationConfig, RiskControlsConfig
    sc = SectorConcentrationConfig(
        enabled=enabled,
        max_names_per_sector=max_names,
        max_weight_per_sector=max_weight,
    )
    rc = RiskControlsConfig(
        shock_exit_mode="composite",
        regime_sizing="none",
        bear_scale=0.25,
        dynamic_cap=False,
        symbol_cap=0.40,
        sector_cap=0.25,
        cluster_cap=0.35,
        bear_sector_cap=0.18,
        bear_cluster_cap=0.25,
        sector_concentration=sc,
    )
    cfg = MagicMock()
    cfg.risk_controls = rc
    return cfg


def _make_bridge(universe_tickers: dict, capital: float = 3_000_000, cfg=None):
    from src.kabusapi.signal_bridge import SignalBridge
    if cfg is None:
        cfg = _make_cfg()
    return SignalBridge(
        universe_tickers   = universe_tickers,
        fujiko_params      = {
            "min_sepa": 6, "min_rsr": 75.0, "mom_period": 21,
            "turtle_entry": 20, "turtle_exit": 55, "use_turtle_entry": True,
        },
        capital            = capital,
        max_positions      = 3,
        min_hold_days      = 3,
        emergency_exit_pct = 0.08,
        cfg                = cfg,
    )


def _make_order(symbol: str, sector: str, estimated_amount: float, side: str = "BUY"):
    from src.kabusapi.signal_bridge import OrderInstruction
    return OrderInstruction(
        symbol           = symbol,
        symbol_4digit    = symbol.replace(".T", ""),
        sector           = sector,
        side             = side,
        qty              = 100,
        order_type       = "MARKET_OPEN",
        estimated_price  = estimated_amount / 100,
        estimated_amount = estimated_amount,
        reason           = "test",
    )


# ------------------------------------------------------------------ #
# テストクラス
# ------------------------------------------------------------------ #
class TestSectorGate(unittest.TestCase):

    def test_sector_block_by_names(self):
        """同一セクター2銘柄目はブロックされる"""
        universe = {"5401.T": "鉄鋼", "5411.T": "鉄鋼", "9101.T": "海運"}
        bridge = _make_bridge(universe)
        current_positions = {"5401.T": {"qty": 500, "avg_price": 600.0}}
        order = _make_order("5411.T", "鉄鋼", 300_000)

        result = bridge.pre_trade_risk_check(
            order, current_positions=current_positions,
            universe_raw={}, above_ma200=True,
        )
        self.assertFalse(result, "同一セクター2銘柄目はブロックされるべき")

    def test_sector_block_by_weight(self):
        """セクターウェイト超過でブロックされる"""
        universe = {"5401.T": "鉄鋼", "5411.T": "鉄鋼"}
        cfg = _make_cfg(enabled=True, max_names=2, max_weight=0.12)
        bridge = _make_bridge(universe, capital=3_000_000, cfg=cfg)
        # 5401.T 300,000円保有（ウェイト10%）+ 5411.T 100,000円(3.3%) = 13.3% > 12%
        current_positions = {"5401.T": {"qty": 500, "avg_price": 600.0}}
        order = _make_order("5411.T", "鉄鋼", 100_000)

        result = bridge.pre_trade_risk_check(
            order, current_positions=current_positions,
            universe_raw={}, above_ma200=True,
        )
        self.assertFalse(result, "セクターウェイト超過でブロックされるべき")

    def test_sector_allow_different_sector(self):
        """異なるセクターは通過する"""
        universe = {"5401.T": "鉄鋼", "9101.T": "海運"}
        bridge = _make_bridge(universe)
        current_positions = {"5401.T": {"qty": 500, "avg_price": 600.0}}
        order = _make_order("9101.T", "海運", 300_000)

        result = bridge.pre_trade_risk_check(
            order, current_positions=current_positions,
            universe_raw={}, above_ma200=True,
        )
        self.assertTrue(result, "異なるセクターは通過するべき")

    def test_sector_gate_disabled(self):
        """enabled=False の場合はチェックしない"""
        universe = {"5401.T": "鉄鋼", "5411.T": "鉄鋼"}
        cfg = _make_cfg(enabled=False)
        bridge = _make_bridge(universe, cfg=cfg)
        current_positions = {"5401.T": {"qty": 500, "avg_price": 600.0}}
        order = _make_order("5411.T", "鉄鋼", 300_000)

        result = bridge.pre_trade_risk_check(
            order, current_positions=current_positions,
            universe_raw={}, above_ma200=True,
        )
        # enabled=False → sector_concentration スキップ
        # symbol_cap(0.40) / sector_cap(0.25) は 300k/3M=10% で超過しないので通過
        self.assertTrue(result, "enabled=False の場合はブロックされないべき")

    def test_sector_log_output(self):
        """ブロック時に sector_block ログが出力される"""
        universe = {"5401.T": "鉄鋼", "5411.T": "鉄鋼"}
        bridge = _make_bridge(universe)
        current_positions = {"5401.T": {"qty": 500, "avg_price": 600.0}}
        order = _make_order("5411.T", "鉄鋼", 300_000)

        logger = logging.getLogger("src.kabusapi.signal_bridge")
        with self.assertLogs(logger, level="INFO") as log_ctx:
            result = bridge.pre_trade_risk_check(
                order, current_positions=current_positions,
                universe_raw={}, above_ma200=True,
            )

        self.assertFalse(result)
        sector_block_msgs = [m for m in log_ctx.output if "sector_block" in m]
        self.assertGreaterEqual(
            len(sector_block_msgs), 1,
            f"sector_block ログが出力されていない。実際: {log_ctx.output}",
        )
        self.assertIn("5411.T", sector_block_msgs[0])
        self.assertIn("鉄鋼",   sector_block_msgs[0])

    def test_config_loader_sector_concentration(self):
        """strategy.yaml の sector_concentration が正しく読み込まれる"""
        from src.config_loader import load_strategy_config
        load_strategy_config.cache_clear()

        cfg = load_strategy_config()
        rc = cfg.risk_controls

        self.assertTrue(hasattr(rc, "sector_concentration"), "sector_concentration フィールドが存在しない")
        sc = rc.sector_concentration
        self.assertTrue(sc.enabled,                 f"enabled は True であるべき (got {sc.enabled})")
        self.assertEqual(sc.max_names_per_sector, 1, f"max_names_per_sector は 1 (got {sc.max_names_per_sector})")
        self.assertAlmostEqual(sc.max_weight_per_sector, 0.12, places=5,
                               msg=f"max_weight_per_sector は 0.12 (got {sc.max_weight_per_sector})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
