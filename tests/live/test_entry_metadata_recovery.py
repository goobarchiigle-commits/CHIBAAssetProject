"""
tests/live/test_entry_metadata_recovery.py

再起動後の entry_price/entry_atr/highest_close 復元（2026-07-07 follow-up
incident）の回帰テスト。対象: src/live/entry_metadata_recovery.py。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.live.entry_metadata_recovery import (
    recover_missing_entry_metadata, recover_missing_entry_rsr,
)


class TestEntryMetadataRecovery(unittest.TestCase):

    def _write_orders_log(self, logs_dir: Path, filename: str, orders: list, run_at="2026-07-07T08:44:05+0900"):
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / filename).write_text(
            json.dumps([{"run_at": run_at, "orders": orders, "send_results": []}], ensure_ascii=False),
            encoding="utf-8",
        )

    def test_recovers_entry_price_and_atr_from_order_log(self):
        """5301.T相当: portfolio_stateはentry_price=0.0/ATR欠落だが、
        当日の注文ログにestimated_price/atr20が残っていれば復元できる。"""
        tmp = Path(self.enterContext_tmpdir())
        logs_live = tmp / "logs" / "live"
        self._write_orders_log(logs_live, "20260707_084405_orders.json", orders=[{
            "symbol": "5301.T", "sector": "化学", "side": "SHADOW_BUY", "qty": 300,
            "estimated_price": 1758.5, "atr20": 45.27,
            "reason": "SHADOW_BUY: RSR62=90.3 (>87.5=live_top10_median) shadow_rsr_pass=8",
        }])

        state = {
            "position_qtys":           {"5301.T": 300},
            "position_entry_prices":   {"5301.T": 0.0},
            "position_entry_atrs":     {},
            "position_highest_closes": {"5301.T": 0.0},
            "position_entry_dates":    {"5301.T": "2026-07-07"},
            "position_strategy_types": {},
        }

        result = recover_missing_entry_metadata(
            state, logs_live_dir=logs_live, audit_log_path=tmp / "audit.jsonl",
        )

        self.assertEqual(len(result["recovered"]), 1)
        self.assertEqual(result["recovered"][0]["symbol"], "5301.T")
        self.assertEqual(state["position_entry_prices"]["5301.T"], 1758.5)
        self.assertEqual(state["position_highest_closes"]["5301.T"], 1758.5)
        self.assertEqual(state["position_entry_atrs"]["5301.T"], 45.27)
        self.assertNotIn("5301.T", state.get("entry_metadata_missing", {}))
        self.assertTrue((tmp / "audit.jsonl").exists())

    def test_unrecoverable_symbol_flagged_without_halting(self):
        """当日ログにも一致する注文が無い場合はentry_metadata_missingに記録するのみ
        （例外を投げない・他の処理は継続できる）。"""
        tmp = Path(self.enterContext_tmpdir())
        logs_live = tmp / "logs" / "live"
        logs_live.mkdir(parents=True, exist_ok=True)  # ログ自体が存在しないケース

        state = {
            "position_qtys":           {"9999.T": 100},
            "position_entry_prices":   {"9999.T": 0.0},
            "position_entry_atrs":     {},
            "position_highest_closes": {"9999.T": 0.0},
            "position_entry_dates":    {"9999.T": "2026-06-01"},
            "position_strategy_types": {},
        }

        result = recover_missing_entry_metadata(
            state, logs_live_dir=logs_live, audit_log_path=tmp / "audit.jsonl",
        )

        self.assertEqual(result["recovered"], [])
        self.assertEqual(len(result["unrecoverable"]), 1)
        self.assertIn("9999.T", state["entry_metadata_missing"],
                       "復元不可の銘柄は監査用レジストリに記録される")
        # 既存の0.0はそのまま（新たな捏造値へ書き換えない）。フラグのみで検知可能にする。
        self.assertEqual(state["position_entry_prices"]["9999.T"], 0.0)

    def test_atr_only_gap_recovered_via_yfinance_fallback(self):
        """entry_priceは注文ログから復元できるがATRが0.0の場合
        （旧SHADOW_BUY経路はatr20=0.0固定だった）、yfinance再計算を試みる。
        ネットワーク非依存にするためyfinance呼び出し自体はモックする。"""
        tmp = Path(self.enterContext_tmpdir())
        logs_live = tmp / "logs" / "live"
        self._write_orders_log(logs_live, "20260707_084405_orders.json", orders=[{
            "symbol": "5301.T", "sector": "化学", "side": "SHADOW_BUY", "qty": 300,
            "estimated_price": 1758.5, "atr20": 0.0,   # 旧経路の実測値どおり0.0
            "reason": "SHADOW_BUY: RSR62=90.3 (>87.5=live_top10_median) shadow_rsr_pass=8",
        }])

        state = {
            "position_qtys":           {"5301.T": 300},
            "position_entry_prices":   {"5301.T": 0.0},
            "position_entry_atrs":     {},
            "position_highest_closes": {"5301.T": 0.0},
            "position_entry_dates":    {"5301.T": "2026-07-07"},
            "position_strategy_types": {},
        }

        with patch(
            "src.live.entry_metadata_recovery._recompute_atr20_via_yfinance",
            return_value=45.275,
        ):
            recover_missing_entry_metadata(
                state, logs_live_dir=logs_live, audit_log_path=tmp / "audit.jsonl",
            )

        self.assertEqual(state["position_entry_prices"]["5301.T"], 1758.5)
        self.assertEqual(state["position_entry_atrs"]["5301.T"], 45.275)
        self.assertNotIn("5301.T", state.get("entry_metadata_missing", {}))

    def enterContext_tmpdir(self):
        import tempfile
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


class TestEntryRsrRecovery(unittest.TestCase):
    """
    2026-07-08 RCA 回帰テスト: run_morning_signal.py / 一部経路が
    update_state_after_execution() へ signal_rsr_map を渡していなかったため
    position_entry_rsrs が欠落し、Quality Replacement Engine が
    "entry_rsr missing (pre-v3 position)" として current RSR を proxy 使用し
    続けていた既存ポジション(6981.T/2802.T/6506.T 実例)の復元ロジック。
    """

    def _write_signal_json(self, signals_dir: Path, filename: str, *, symbol: str,
                            rsr: float, side: str = "BUY"):
        signals_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "LIVE",
            "orders": [{"symbol": symbol, "side": side, "qty": 100}],
            "signals": [{"symbol": symbol, "rsr": rsr, "signal": 1}],
        }
        (signals_dir / filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_recovers_entry_rsr_from_daily_signal_log(self):
        tmp = Path(self.enterContext_tmpdir())
        signals_dir = tmp / "data" / "signals"
        self._write_signal_json(
            signals_dir, "signal_20260623_084118_executed.json",
            symbol="2802.T", rsr=82.5,
        )
        state = {
            "position_qtys":        {"2802.T": 100},
            "position_entry_dates": {"2802.T": "2026-06-23"},
            "position_entry_rsrs":  {},
        }

        result = recover_missing_entry_rsr(
            state, signals_dir=signals_dir, audit_log_path=tmp / "rsr_audit.jsonl",
        )

        self.assertEqual(len(result["recovered"]), 1)
        self.assertEqual(state["position_entry_rsrs"]["2802.T"], 82.5)
        self.assertTrue((tmp / "rsr_audit.jsonl").exists())

    def test_already_recorded_entry_rsr_untouched(self):
        tmp = Path(self.enterContext_tmpdir())
        signals_dir = tmp / "data" / "signals"
        state = {
            "position_qtys":        {"5301.T": 300},
            "position_entry_dates": {"5301.T": "2026-07-07"},
            "position_entry_rsrs":  {"5301.T": 90.3},
        }

        result = recover_missing_entry_rsr(
            state, signals_dir=signals_dir, audit_log_path=tmp / "rsr_audit.jsonl",
        )

        self.assertEqual(result["recovered"], [])
        self.assertEqual(state["position_entry_rsrs"]["5301.T"], 90.3)

    def test_no_matching_buy_in_signal_log_leaves_unresolved_not_fabricated(self):
        """該当日のシグナルJSONにBUY記録が無ければ、推測でRSRを書き込まない
        （既存のQR proxyフォールバックが継続される）。"""
        tmp = Path(self.enterContext_tmpdir())
        signals_dir = tmp / "data" / "signals"
        # SELL のみ・BUY無し → recover対象外
        self._write_signal_json(
            signals_dir, "signal_20260428_130016_executed.json",
            symbol="6981.T", rsr=92.9, side="SELL",
        )
        state = {
            "position_qtys":        {"6981.T": 100},
            "position_entry_dates": {"6981.T": "2026-04-28"},
            "position_entry_rsrs":  {},
        }

        result = recover_missing_entry_rsr(
            state, signals_dir=signals_dir, audit_log_path=tmp / "rsr_audit.jsonl",
        )

        self.assertEqual(result["recovered"], [])
        self.assertEqual(len(result["unrecoverable"]), 1)
        self.assertNotIn("6981.T", state["position_entry_rsrs"])

    def enterContext_tmpdir(self):
        import tempfile
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


if __name__ == "__main__":
    unittest.main(verbosity=2)
