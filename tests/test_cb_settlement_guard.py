"""
tests/test_cb_settlement_guard.py

2026-07-19 更新: PendingOrderState機構（旧TASK-2 CB settlement-lag guard、
SOURCE1=state-based + SOURCE2=ledger-based の2ソース補償）はBroker-as-Sole-SSOT
の方針に基づき完全撤去された。根拠:
  - SOURCE1はstale portfolio_stateとの区別が原理的に不可能な構造的欠陥があり、
    実機検証で5日前に確定売却済みの3銘柄を「決済待ち」と誤認し¥1,748,850を
    誤算出した。
  - SOURCE2はrun_live_signal.pyがOrderLedger.check_and_record()を呼ばないため
    実質死コードだった。
  - 両インシデントの発生前提（run_morning_signal.pyとの並行スケジュール実行）は
    2026-07-15 SSOT統合で構造的に消滅済み。CB_ACTIVE誤発動時も既存の
    [CB_FAST_RECOVERY]/[CB_AUTO_RESTORE]機構により翌run（最大1日）で自己修復する。

旧`TestCBSettlementGuard`（ローカル再実装のsettlement-lag補償テスト）と
`TestCBGuardV2ExecutionLedger`（`_cb_guard_compensation`/`PendingOrderState`を
直接importするテスト）は検証対象が消滅したため本ファイルから削除した。
`TestCBStateStructuredLog`はpending_order_state引数を使わない形に書き換えた。

Run:
    cd C:/ai-trading
    python -m pytest tests/test_cb_settlement_guard.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.analytics.slot_pressure import (
    SlotPressureInput,
    compute_slot_pressure,
    build_slot_pressure_input_from_oc_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# TASK-1 tests — slot_pressure current_positions from portfolio_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestSlotPressureCurrentPositions(unittest.TestCase):
    """
    Verify that current_positions used for slot_pressure reflects
    actual holdings, not winner-confirmation count.
    """

    def _make_inp(self, current_positions: int, max_positions: int = 3) -> SlotPressureInput:
        return SlotPressureInput(
            max_positions=max_positions,
            current_positions=current_positions,
            holding_priorities=[],
            holding_tiers=[],
            avg_candidate_priority_5d=0.0,
            recent_rejected_candidates=0,
            recent_severe_opportunity_cost_events=0,
        )

    def test_two_holdings_one_open_slot(self):
        """
        2 actual holdings, max_positions=3 → open_slots MUST be 1.
        This reproduces the fixed bug (was 3 before fix).
        """
        inp    = self._make_inp(current_positions=2, max_positions=3)
        result = compute_slot_pressure(inp)
        self.assertEqual(result.open_slots, 1,
                         "open_slots must be max_positions - current_positions = 1")

    def test_zero_holdings_three_open_slots(self):
        """0 holdings, max_positions=3 → open_slots=3 (correct empty portfolio)."""
        inp    = self._make_inp(current_positions=0, max_positions=3)
        result = compute_slot_pressure(inp)
        self.assertEqual(result.open_slots, 3)

    def test_full_portfolio_zero_open_slots(self):
        """max_positions=3, current=3 → open_slots=0."""
        inp    = self._make_inp(current_positions=3, max_positions=3)
        result = compute_slot_pressure(inp)
        self.assertEqual(result.open_slots, 0)

    def test_portfolio_summary_value_used(self):
        """
        Simulates the exact fix: portfolio_summary["current_positions"] overrides
        len(_sp_held_list) when winner confirmations are zero.

        Before fix: current_positions=len([])=0 → open_slots=3
        After  fix: current_positions=portfolio_summary.get("current_positions", 0)=2 → open_slots=1
        """
        portfolio_summary = {"current_positions": 2}
        sp_held_list: list = []  # winner confirmations empty (both confirmed=False)

        # Replicate the fixed expression from run_live_signal.py
        effective_positions = portfolio_summary.get("current_positions", len(sp_held_list))
        self.assertEqual(effective_positions, 2)

        inp    = self._make_inp(current_positions=effective_positions, max_positions=3)
        result = compute_slot_pressure(inp)
        self.assertEqual(result.open_slots, 1)

    def test_fallback_to_held_list_when_summary_missing(self):
        """
        If portfolio_summary has no 'current_positions' key,
        fallback to len(_sp_held_list) is used.
        """
        portfolio_summary: dict = {}  # no current_positions key
        sp_held_list = [MagicMock(), MagicMock()]  # 2 confirmed winners

        effective_positions = portfolio_summary.get("current_positions", len(sp_held_list))
        self.assertEqual(effective_positions, 2)


# ─────────────────────────────────────────────────────────────────────────────
# CB_STATE structured log + last_equity fix
# ─────────────────────────────────────────────────────────────────────────────


def _make_bridge_stub(capital: float = 3_000_000.0):
    """
    _update_cb_state() をテストするための最小 SignalBridge スタブ。
    API / ファイル IO は一切呼ばない。
    """
    from src.kabusapi.signal_bridge import SignalBridge
    bridge = object.__new__(SignalBridge)
    bridge.capital = capital
    bridge.live    = True
    return bridge


class TestCBStateStructuredLog(unittest.TestCase):
    """
    _update_cb_state() が [CB_STATE] 構造化ログを正しく出力し、
    last_equity を上書きしないことを検証する。

    2026-07-19 Broker-as-Sole-SSOTリファクタ: PendingOrderState機構（決済ラグ
    補償によるCB発動抑制）を完全撤去。current_equityは常にraw broker equityを
    渡し、CB判定はこの値のみで行う（state/ledgerへの依存ゼロ）。撤去の根拠は
    _update_cb_state()のdocstring、および同セッションの設計レビューを参照。
    """

    PEAK = 4_089_109.0
    RAW  = 3_184_309.0

    def _state(self, cb="NORMAL", last_equity=None):
        return {
            "cb_state":             cb,
            "equity_peak":          self.PEAK,
            "safe_warn_count":      0,
            "cb_cooldown_end_date": None,
            "recovery_threshold":   None,
            "last_equity":          last_equity or self.RAW,
        }

    # ------------------------------------------------------------------
    # T1: raw DD が閾値を超えたら CB_ACTIVE が発火し WARNING が出ること
    # ------------------------------------------------------------------
    def test_raw_dd_trigger_fires_cb_and_logs_warning(self):
        """
        PendingOrderState撤去後、CB判定はraw broker equityのみで行われる。
        DD=-22.1% は閾値-15%を超えるため CB_ACTIVE が発火し
        [CB_STATE] reason=DRAWDOWN_TRIGGER が WARNING で出力されること。
        """
        bridge = _make_bridge_stub()
        state  = self._state()

        with self.assertLogs("src.kabusapi.signal_bridge", level="WARNING") as cm:
            bridge._update_cb_state(state, self.RAW, "2026-06-04")

        self.assertEqual(state["cb_state"], "CB_ACTIVE",
                         "raw DD=-22.1% must trigger CB_ACTIVE")

        combined = "\n".join(cm.output)
        self.assertIn("[CB_STATE]",          combined)
        self.assertIn("before=NORMAL",       combined)
        self.assertIn("after=CB_ACTIVE",     combined)
        self.assertIn("DRAWDOWN_TRIGGER",    combined)
        self.assertIn("dd=",                 combined)

    # ------------------------------------------------------------------
    # T2: last_equity が _update_cb_state() で上書きされないこと
    # ------------------------------------------------------------------
    def test_last_equity_not_overwritten(self):
        """
        commit_broker_snapshot() が raw equity を last_equity に書く。
        _update_cb_state() は状態遷移判定のみを行い、last_equity を
        上書きしてはならない。
        """
        bridge = _make_bridge_stub()
        state  = self._state(last_equity=self.RAW)

        bridge._update_cb_state(state, self.RAW, "2026-06-04")
        self.assertAlmostEqual(
            state["last_equity"], self.RAW, places=0,
            msg="last_equity must remain raw equity after _update_cb_state()"
        )

    # ------------------------------------------------------------------
    # T3: [CB_STATE] INFO ログ (遷移なし)
    # ------------------------------------------------------------------
    def test_no_transition_emits_info_log(self):
        """遷移がない場合は [CB_STATE] INFO が出ること (WARNING は出ない)。"""
        bridge = _make_bridge_stub()
        # equity == peak → DD=0%、遷移なし
        state  = self._state(last_equity=self.PEAK)

        with self.assertLogs("src.kabusapi.signal_bridge", level="INFO") as cm:
            bridge._update_cb_state(state, self.PEAK, "2026-06-04")

        self.assertEqual(state["cb_state"], "NORMAL")
        combined = "\n".join(cm.output)
        self.assertIn("[CB_STATE]", combined)
        self.assertIn("dd=",        combined)
        warning_lines = [l for l in cm.output if "WARNING" in l and "[CB_STATE]" in l]
        self.assertEqual(warning_lines, [],
                         "no [CB_STATE] WARNING expected when no state transition")

    # ------------------------------------------------------------------
    # T4: PEAK_ANOMALY は NORMAL から到達不能 (設計上の制約確認)
    # ------------------------------------------------------------------
    def test_peak_anomaly_unreachable_from_normal(self):
        """
        PEAK_ANOMALY_RATIO=1.25 かつ CB_DD_TRIGGER=0.15 の場合、
        ratio > 1.25 → DD < -20% → CB_DD_TRIGGER -15% を必ず超えるため
        SAFE_WARN は NORMAL 状態からは数学的に到達不能であることを確認する。

        ratio > 1.25 ⟹ equity/peak < 0.8 ⟹ DD = equity/peak - 1 < -0.2 = -20%
        -20% ≤ -15% (CB_DD_TRIGGER) ⟹ CB_ACTIVE が先に発火する。
        """
        bridge = _make_bridge_stub()
        equity = 3_000_000.0
        # ratio = 1.26: peak = equity × 1.26 → DD = -20.6%
        big_peak = equity * 1.26
        state = {
            "cb_state":             "NORMAL",
            "equity_peak":          big_peak,
            "safe_warn_count":      0,
            "cb_cooldown_end_date": None,
            "recovery_threshold":   None,
            "last_equity":          equity,
        }
        bridge._update_cb_state(state, equity, "2026-06-05")

        # ratio > 1.25 のとき DD > 20% → CB_ACTIVE が先に発火する
        self.assertEqual(state["cb_state"], "CB_ACTIVE",
                         "ratio>1.25 implies DD>20%>15% → CB_ACTIVE fires before SAFE_WARN")

    # ------------------------------------------------------------------
    # T5: CB_FAST_RECOVERY — 誤発動しても翌run相当のequity回復で自動復旧
    # ------------------------------------------------------------------
    def test_cb_active_fast_recovery_on_equity_rebound(self):
        """
        PendingOrderState撤去後の残存リスク（broker内部ラグ等によるDD誤検知）が
        自己修復することを確認する回帰テスト。CB_ACTIVE中でもequityが
        recovery_threshold(peak×98%)以上かつDD>-15%に戻れば、次のrun相当の
        呼び出しでNORMALへ即時復帰する（30営業日クールダウンを待たない）。
        """
        bridge = _make_bridge_stub()
        state = {
            "cb_state":             "CB_ACTIVE",
            "equity_peak":          self.PEAK,
            "safe_warn_count":      0,
            "cb_cooldown_end_date": "2026-08-15",
            "recovery_threshold":   round(self.PEAK * 0.98, 0),
            "last_equity":          self.RAW,
        }
        recovered_equity = self.PEAK  # 完全回復
        bridge._update_cb_state(state, recovered_equity, "2026-06-05")

        self.assertEqual(state["cb_state"], "NORMAL",
                         "equityがrecovery_threshold以上に回復すればCB_FAST_RECOVERYでNORMALへ復帰")
        self.assertIsNone(state["cb_cooldown_end_date"])


if __name__ == "__main__":
    unittest.main()
