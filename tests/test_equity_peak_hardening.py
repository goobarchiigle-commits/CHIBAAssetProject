"""
tests/test_equity_peak_hardening.py
EQUITY_PEAK_HARDENING (2026-07-03) の回帰テスト。

対象:
  - _commit_equity_peak() が _update_cb_state() 以外からの呼び出しを拒否すること
  - broker_equity vs cash+market_value 乖離時に EQUITY_PEAK_REJECT すること
  - +10% 以上のジャンプが candidate_peak にステージングされ即時採用されないこと
  - 翌営業日の再確認で CONFIRMED / 基準未達で DISCARDED になること
  - 同日中は candidate_peak が据え置かれること
"""
import unittest
import pandas as pd

from src.kabusapi.signal_bridge import (
    SignalBridge,
    _commit_equity_peak,
    CANDIDATE_PEAK_JUMP_THRESHOLD,
)
from src.portfolio.state_store import BrokerSnapshot


def _make_bridge_stub(capital: float = 3_000_000.0, live: bool = True):
    bridge = object.__new__(SignalBridge)
    bridge.capital = capital
    bridge.live    = live
    return bridge


def _snapshot(cash: float, positions: dict[str, int], prices: dict[str, float]) -> BrokerSnapshot:
    return BrokerSnapshot(
        cash          = cash,
        positions     = positions,
        avg_costs     = prices,
        market_values = prices,
        equity        = 0.0,
        ts            = "2026-07-03T08:44:00+0900",
        source        = "broker",
        api_health    = {"positions_ok": True, "wallet_ok": True},
    )


def _next_trading_day(date_str: str) -> str:
    from src.kabusapi.signal_bridge import _add_trading_days
    return _add_trading_days(pd.Timestamp(date_str), 1).strftime("%Y-%m-%d")


class TestForbiddenWrite(unittest.TestCase):
    def test_direct_call_raises_runtime_error(self):
        """_update_cb_state() 以外から _commit_equity_peak() を呼ぶと RuntimeError。"""
        state = {"equity_peak": 3_000_000.0}
        with self.assertRaises(RuntimeError) as ctx:
            _commit_equity_peak(
                state, 4_000_000.0, 4_000_000.0,
                caller="test_direct_call", reason="new_high",
                broker_snapshot=None, today_str="2026-07-03", mode="live",
            )
        self.assertIn("EQUITY_PEAK_FORBIDDEN_WRITE", str(ctx.exception))
        self.assertEqual(state["equity_peak"], 3_000_000.0)  # 変更されていない


class TestBrokerConsistencyReject(unittest.TestCase):
    def test_diverged_broker_equity_rejects_update(self):
        """
        broker 生値と current_equity(cache併用計算値) が大きく乖離する場合、
        new_high 更新でも EQUITY_PEAK_REJECT となり peak は不変。
        """
        bridge = _make_bridge_stub()
        state = {
            "cb_state": "NORMAL", "equity_peak": 3_000_000.0, "safe_warn_count": 0,
            "cb_cooldown_end_date": None, "recovery_threshold": None,
            "last_equity": 3_000_000.0, "candidate_peak": None,
        }
        # broker 生値: cash 3,000,000 + position 0 = 3,000,000（乖離なし想定の生値）
        # current_equity（cache併用計算値）は broker値より遥かに高い異常値を模擬
        snap = _snapshot(cash=3_000_000.0, positions={}, prices={})
        bridge._update_cb_state(
            state, current_equity=4_200_000.0, today_str="2026-07-03",
            raw_equity=4_200_000.0, broker_snapshot=snap,
        )
        self.assertEqual(state["equity_peak"], 3_000_000.0)
        self.assertIsNone(state.get("candidate_peak"))

    def test_consistent_broker_equity_allows_new_high(self):
        """broker生値とcurrent_equityが一致(乖離小)なら通常通りnew_high更新される。"""
        bridge = _make_bridge_stub()
        state = {
            "cb_state": "NORMAL", "equity_peak": 3_000_000.0, "safe_warn_count": 0,
            "cb_cooldown_end_date": None, "recovery_threshold": None,
            "last_equity": 3_000_000.0, "candidate_peak": None,
        }
        snap = _snapshot(cash=3_050_000.0, positions={}, prices={})
        bridge._update_cb_state(
            state, current_equity=3_050_000.0, today_str="2026-07-03",
            raw_equity=3_050_000.0, broker_snapshot=snap,
        )
        self.assertEqual(state["equity_peak"], 3_050_000.0)

    def test_broker_snapshot_none_fail_open(self):
        """broker_snapshot=None（API部分失敗）は整合性チェックをスキップし通常更新される。"""
        bridge = _make_bridge_stub()
        state = {
            "cb_state": "NORMAL", "equity_peak": 3_000_000.0, "safe_warn_count": 0,
            "cb_cooldown_end_date": None, "recovery_threshold": None,
            "last_equity": 3_000_000.0, "candidate_peak": None,
        }
        bridge._update_cb_state(
            state, current_equity=3_050_000.0, today_str="2026-07-03",
            raw_equity=3_050_000.0, broker_snapshot=None,
        )
        self.assertEqual(state["equity_peak"], 3_050_000.0)


class TestCandidateStaging(unittest.TestCase):
    def _base_state(self, peak=3_000_000.0):
        return {
            "cb_state": "NORMAL", "equity_peak": peak, "safe_warn_count": 0,
            "cb_cooldown_end_date": None, "recovery_threshold": None,
            "last_equity": peak, "candidate_peak": None,
        }

    def test_jump_over_threshold_is_staged_not_applied(self):
        """前回peak比+10%以上のジャンプは即時採用されずcandidate_peakへ格納される。"""
        bridge = _make_bridge_stub()
        state  = self._base_state(peak=3_000_000.0)
        jumped_equity = 3_000_000.0 * (1 + CANDIDATE_PEAK_JUMP_THRESHOLD + 0.01)
        snap = _snapshot(cash=jumped_equity, positions={}, prices={})
        bridge._update_cb_state(
            state, current_equity=jumped_equity, today_str="2026-07-03",
            raw_equity=jumped_equity, broker_snapshot=snap,
        )
        self.assertEqual(state["equity_peak"], 3_000_000.0)  # 未反映
        self.assertIsNotNone(state["candidate_peak"])
        self.assertEqual(state["candidate_peak"]["value"], round(jumped_equity, 0))
        self.assertEqual(state["candidate_peak"]["staged_date"], "2026-07-03")

    def test_jump_under_threshold_applies_immediately(self):
        """+10%未満のジャンプはステージングされず即時反映される。"""
        bridge = _make_bridge_stub()
        state  = self._base_state(peak=3_000_000.0)
        small_jump_equity = 3_000_000.0 * 1.05
        snap = _snapshot(cash=small_jump_equity, positions={}, prices={})
        bridge._update_cb_state(
            state, current_equity=small_jump_equity, today_str="2026-07-03",
            raw_equity=small_jump_equity, broker_snapshot=snap,
        )
        self.assertEqual(state["equity_peak"], round(small_jump_equity, 0))
        self.assertIsNone(state["candidate_peak"])

    def test_same_day_candidate_left_pending(self):
        """ステージング当日中は候補が据え置かれ、確定も破棄もされない。"""
        bridge = _make_bridge_stub()
        state  = self._base_state(peak=3_000_000.0)
        state["candidate_peak"] = {
            "value": 3_400_000.0, "staged_date": "2026-07-03",
            "reason": "new_high", "current_equity_at_stage": 3_400_000.0,
        }
        bridge._update_cb_state(
            state, current_equity=3_400_000.0, today_str="2026-07-03",
            raw_equity=3_400_000.0, broker_snapshot=None,
        )
        self.assertEqual(state["equity_peak"], 3_000_000.0)
        self.assertIsNotNone(state["candidate_peak"])
        self.assertEqual(state["candidate_peak"]["value"], 3_400_000.0)

    def test_next_trading_day_confirmation_applies_candidate(self):
        """翌営業日、equityが候補値の許容下限以上なら確定してequity_peakへ反映される。"""
        bridge = _make_bridge_stub()
        staged_date = "2026-07-03"
        next_td     = _next_trading_day(staged_date)
        state = self._base_state(peak=3_000_000.0)
        state["candidate_peak"] = {
            "value": 3_400_000.0, "staged_date": staged_date,
            "reason": "new_high", "current_equity_at_stage": 3_400_000.0,
        }
        bridge._update_cb_state(
            state, current_equity=3_400_000.0, today_str=next_td,
            raw_equity=3_400_000.0, broker_snapshot=None,
        )
        self.assertEqual(state["equity_peak"], 3_400_000.0)
        self.assertIsNone(state["candidate_peak"])

    def test_next_trading_day_reconfirm_failure_discards_candidate(self):
        """翌営業日、equityが許容下限を下回ると候補は破棄されpeakは元のまま。"""
        bridge = _make_bridge_stub()
        staged_date = "2026-07-03"
        next_td     = _next_trading_day(staged_date)
        state = self._base_state(peak=3_000_000.0)
        state["candidate_peak"] = {
            "value": 3_400_000.0, "staged_date": staged_date,
            "reason": "new_high", "current_equity_at_stage": 3_400_000.0,
        }
        # 許容下限 = 3,400,000 * (1 - 0.02) = 3,332,000 を大きく下回る equity
        bridge._update_cb_state(
            state, current_equity=3_000_000.0, today_str=next_td,
            raw_equity=3_000_000.0, broker_snapshot=None,
        )
        self.assertEqual(state["equity_peak"], 3_000_000.0)
        self.assertIsNone(state["candidate_peak"])


if __name__ == "__main__":
    unittest.main()
