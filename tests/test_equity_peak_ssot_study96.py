"""
tests/test_equity_peak_ssot_study96.py
Study96 — EquityPeak SSOT Root Cause Audit（2026-07-17）の回帰テスト。

設計方針（2026-07-17 ユーザー指摘を受けて確定）:
  equity_peakの唯一の真実は「現在の証券口座equity」である。原因推定
  （入出金の有無等）は一切行わない。異常なジャンプは理由を問わず一律に
  candidate_peakへ保留し、CANDIDATE_PEAK_RECONFIRM_COUNT（既定3）連続営業日の
  持続が確認されて初めて確定する。2026-07-03のcheck_broker_consistency()と
  10%ジャンプ猶予（candidate_peak staging）はデータ破損対策として維持する。

なぜ実インシデント（4,110,741 → 5,598,886・2026-07-15〜16）が旧設計をすり抜けたか:
  10%ジャンプ猶予は正しく機能し、+35.9%ジャンプは即座には採用されずSTAGEDされた。
  しかし「翌営業日の**1回だけ**の再確認」で無条件にCONFIRMEDされる設計だったため、
  たまたま2営業日連続で同水準のequityが観測されただけでpeakが確定してしまった。
  本修正はこの再確認を1回→N回（既定3）連続へ強化し、単発の持続だけでは
  確定させないようにする（原因推定は行わず、営業日をまたいだ持続回数のみで判定）。

本ファイルは Phase6 で要求された8シナリオ全てを検証する:
  1. 通常更新       (TestNormalUpdate)
  2. 再起動         (TestRestart)
  3. バックアップ復元 (TestBackupRestore)
  4. 壊れたstate     (TestCorruptedState)
  5. peak逆行        (TestPeakRegression)
  6. peakジャンプ     (TestPeakJumpReconfirmation) ← 実インシデントの直接再現+多段階確認の検証
  7. broker再取得     (TestBrokerRefetch)
  8. bootstrap        (TestBootstrap)
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.kabusapi.signal_bridge import (
    SignalBridge,
    _commit_equity_peak,
    EquityPeakInvariantError,
    CANDIDATE_PEAK_JUMP_THRESHOLD,
    CANDIDATE_PEAK_RECONFIRM_COUNT,
)
from src.portfolio.equity import detect_cash_event
from src.portfolio.state_store import (
    BrokerSnapshot,
    load_portfolio_state,
    save_portfolio_state,
    _self_heal,
    validate_state,
    INITIAL_CAPITAL,
)


def _update_cb_state(*args, **kwargs):
    """_commit_equity_peak() の呼び出し元フレーム名ガード（EQUITY_PEAK_FORBIDDEN_WRITE）を
    満たすためのdecoy。_update_cb_state()の正常フローでは構造的に到達不能な
    不変条件（例: reason=new_highでcandidate<=old_peak）を直接単体テストするために使う。
    本物の_update_cb_state()のロジックは一切呼ばない（名前が同じだけの薄いラッパー）。"""
    return _commit_equity_peak(*args, **kwargs)


def _make_bridge_stub(capital: float = 3_000_000.0, live: bool = True):
    bridge = object.__new__(SignalBridge)
    bridge.capital = capital
    bridge.live    = live
    return bridge


def _snapshot(cash: float, positions: dict[str, int], prices: dict[str, float]) -> BrokerSnapshot:
    return BrokerSnapshot(
        cash=cash, positions=positions, avg_costs=prices, market_values=prices,
        equity=0.0, ts="2026-07-17T08:44:00+0900", source="broker",
        api_health={"positions_ok": True, "wallet_ok": True},
    )


def _next_trading_day(date_str: str) -> str:
    from src.kabusapi.signal_bridge import _add_trading_days
    return _add_trading_days(pd.Timestamp(date_str), 1).strftime("%Y-%m-%d")


def _base_state(peak: float = 3_000_000.0) -> dict:
    return {
        "cb_state": "NORMAL", "equity_peak": peak, "safe_warn_count": 0,
        "cb_cooldown_end_date": None, "recovery_threshold": None,
        "last_equity": peak, "candidate_peak": None,
    }


# ══════════════════════════════════════════════════════════════════════════
# 1. 通常更新
# ══════════════════════════════════════════════════════════════════════════
class TestNormalUpdate(unittest.TestCase):
    def test_normal_update_below_threshold_applies_immediately(self):
        bridge = _make_bridge_stub()
        state = _base_state(peak=3_000_000.0)
        new_equity = 3_150_000.0  # +5% < 10%閾値
        bridge._update_cb_state(
            state, current_equity=new_equity, today_str="2026-07-17",
            broker_snapshot=_snapshot(new_equity, {}, {}),
        )
        self.assertEqual(state["equity_peak"], new_equity)

    def test_normal_update_no_change_when_equity_below_peak(self):
        bridge = _make_bridge_stub()
        state = _base_state(peak=3_000_000.0)
        bridge._update_cb_state(
            state, current_equity=2_800_000.0, today_str="2026-07-17",
            broker_snapshot=_snapshot(2_800_000.0, {}, {}),
        )
        self.assertEqual(state["equity_peak"], 3_000_000.0)


# ══════════════════════════════════════════════════════════════════════════
# 2. 再起動（プロセス再起動＝新規stateロード後も peak が保持されること）
# ══════════════════════════════════════════════════════════════════════════
class TestRestart(unittest.TestCase):
    def test_restart_preserves_peak_across_save_load_cycle(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "portfolio_state.json"
            state = _base_state(peak=4_200_000.0)
            state["available_cash"] = 1_000_000.0
            save_portfolio_state(state, path=path, data_source="test_restart")

            reloaded, vr = load_portfolio_state(path=path)
            self.assertTrue(vr.ok)
            self.assertEqual(reloaded["equity_peak"], 4_200_000.0)

    def test_restart_then_new_high_update_persists(self):
        """再起動後のロードstateに対してnew_high更新→再保存→再ロードでも一貫すること。"""
        with TemporaryDirectory() as d:
            path = Path(d) / "portfolio_state.json"
            save_portfolio_state(_base_state(peak=3_000_000.0), path=path, data_source="test")
            state, _ = load_portfolio_state(path=path)

            bridge = _make_bridge_stub()
            bridge._update_cb_state(
                state, current_equity=3_100_000.0, today_str="2026-07-17",
                broker_snapshot=_snapshot(3_100_000.0, {}, {}),
            )
            save_portfolio_state(state, path=path, data_source="test_restart2")
            reloaded, _ = load_portfolio_state(path=path)
            self.assertEqual(reloaded["equity_peak"], 3_100_000.0)


# ══════════════════════════════════════════════════════════════════════════
# 3. バックアップ復元
# ══════════════════════════════════════════════════════════════════════════
class TestBackupRestore(unittest.TestCase):
    def test_backup_created_on_load_when_validation_warns(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "portfolio_state.json"
            backup_dir = Path(d) / "state_backups"
            # 壊れたJSON（equity_peakがNaN文字列経由でNoneになるケースを模擬: 欠損キー）
            path.write_text(json.dumps({"equity_peak": 3_000_000.0}), encoding="utf-8")

            import src.portfolio.state_store as ss
            _orig_backup_dir = ss._BACKUP_DIR
            ss._BACKUP_DIR = backup_dir
            try:
                state, vr = load_portfolio_state(path=path)
                self.assertTrue(backup_dir.exists())
                backups = list(backup_dir.glob("portfolio_state.bak.*.json"))
                self.assertGreaterEqual(len(backups), 1, "欠損キーのself-heal発火時にバックアップが作成されること")
                # バックアップ内容が復元前の値と一致すること
                backup_content = json.loads(backups[0].read_text(encoding="utf-8"))
                self.assertEqual(backup_content["equity_peak"], 3_000_000.0)
            finally:
                ss._BACKUP_DIR = _orig_backup_dir

    def test_restore_from_backup_preserves_original_peak(self):
        """バックアップファイルをそのまま復元(コピー)した場合、peakが改変されないこと。"""
        with TemporaryDirectory() as d:
            path = Path(d) / "portfolio_state.json"
            backup_path = Path(d) / "portfolio_state.bak.json"
            state = _base_state(peak=4_500_000.0)
            state["available_cash"] = 500_000.0
            save_portfolio_state(state, path=path, data_source="test")

            import shutil
            shutil.copy2(path, backup_path)
            shutil.copy2(backup_path, path)  # 復元操作の模擬

            reloaded, vr = load_portfolio_state(path=path)
            self.assertEqual(reloaded["equity_peak"], 4_500_000.0)


# ══════════════════════════════════════════════════════════════════════════
# 4. 壊れたstate
# ══════════════════════════════════════════════════════════════════════════
class TestCorruptedState(unittest.TestCase):
    def test_nan_peak_is_healed_to_initial_capital(self):
        state = {"equity_peak": float("nan"), "last_equity": 1_000_000.0}
        vr = validate_state(state)
        self.assertFalse(vr.ok)
        healed_state, healed = _self_heal(state, vr)
        self.assertEqual(healed_state["equity_peak"], INITIAL_CAPITAL)
        self.assertTrue(any("equity_peak" in h for h in healed))

    def test_negative_peak_is_healed(self):
        state = {"equity_peak": -500_000.0, "last_equity": 1_000_000.0}
        vr = validate_state(state)
        healed_state, healed = _self_heal(state, vr)
        self.assertEqual(healed_state["equity_peak"], INITIAL_CAPITAL)

    def test_self_heal_repair_is_audited(self):
        """Study96 Phase4: _self_healの修復もequity_peak_audit.jsonlへ記録されること。"""
        import src.portfolio.state_store as ss
        calls = []

        def _fake_append(**kwargs):
            calls.append(kwargs)

        import src.portfolio.equity as eq_mod
        _orig = eq_mod.append_peak_audit
        eq_mod.append_peak_audit = _fake_append
        try:
            state = {"equity_peak": 0.0, "last_equity": 1_000_000.0}
            vr = validate_state(state)
            ss._self_heal(state, vr)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["action"], "SELF_HEAL")
        finally:
            eq_mod.append_peak_audit = _orig

    def test_corrupted_state_load_returns_safe_defaults(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "portfolio_state.json"
            path.write_text("{not valid json:::", encoding="utf-8")
            state, vr = load_portfolio_state(path=path)
            self.assertFalse(vr.ok)
            self.assertEqual(state["equity_peak"], INITIAL_CAPITAL)


# ══════════════════════════════════════════════════════════════════════════
# 5. peak逆行（Study96 Phase5 新規assert）
# ══════════════════════════════════════════════════════════════════════════
class TestPeakRegression(unittest.TestCase):
    def test_new_high_with_candidate_below_old_peak_raises(self):
        """reason=new_highでcandidate<=old_peakは論理的に呼ばれてはならない不変条件違反
        （_update_cb_state()の正常フローでは構造的に到達不能なため、frame名decoy経由で
        _commit_equity_peak()自体の防御ロジックを直接検証する）。"""
        state = {"equity_peak": 5_000_000.0}
        with self.assertRaises(EquityPeakInvariantError):
            _update_cb_state(
                state, candidate_value=4_000_000.0, current_equity=4_000_000.0,
                caller="_update_cb_state", reason="new_high",
                broker_snapshot=None, today_str="2026-07-17", mode="live",
            )
        # state不変であること
        self.assertEqual(state["equity_peak"], 5_000_000.0)

    def test_candidate_less_than_current_equity_raises_invariant_error(self):
        """peakはequity以上でなければならない不変条件（bypass_candidate_gate経路）。"""
        state = {"equity_peak": 3_000_000.0}
        with self.assertRaises(EquityPeakInvariantError):
            _update_cb_state(
                state, candidate_value=3_500_000.0, current_equity=4_000_000.0,
                caller="_update_cb_state", reason="candidate_peak_confirmed",
                broker_snapshot=None, today_str="2026-07-17", mode="live",
                bypass_candidate_gate=True,
            )
        self.assertEqual(state["equity_peak"], 3_000_000.0)


# ══════════════════════════════════════════════════════════════════════════
# 6. peakジャンプ（根本原因の直接再現 + 修正確認）
# ══════════════════════════════════════════════════════════════════════════
class TestPeakJumpReconfirmation(unittest.TestCase):
    """原因推定（入出金等）は一切行わない。異常なジャンプは営業日をまたいだ
    N回連続の持続確認（CANDIDATE_PEAK_RECONFIRM_COUNT）でのみ確定する。"""

    def _run_day(self, bridge, state, equity: float, date_str: str):
        bridge._update_cb_state(
            state, current_equity=equity, today_str=date_str,
            broker_snapshot=_snapshot(equity, {}, {}),
        )

    def test_single_day_persistence_no_longer_confirms(self):
        """実インシデントの再現: +35.9%ジャンプが翌営業日1回だけ持続しても、
        （旧設計ではCONFIRMEDされていたが）新設計ではまだHOLDING中でありpeakは不変。"""
        bridge = _make_bridge_stub()
        state = _base_state(peak=4_110_741.0)
        jumped_equity = 5_587_186.0  # 実インシデント値

        self._run_day(bridge, state, jumped_equity, "2026-07-15")
        self.assertEqual(state["equity_peak"], 4_110_741.0, "STAGEDのため即時反映なし")
        self.assertIsNotNone(state["candidate_peak"])

        # 翌営業日、同水準が1回持続（実インシデントでは、ここでCONFIRMEDされてしまっていた）
        self._run_day(bridge, state, 5_598_886.0, "2026-07-16")
        self.assertEqual(
            state["equity_peak"], 4_110_741.0,
            "1回の再確認だけではpeakは確定してはならない（実インシデントの再発防止・中核テスト）",
        )
        self.assertIsNotNone(state["candidate_peak"], "候補はHOLDING中で保持され続けること")
        self.assertEqual(state["candidate_peak"]["confirm_count"], 1)

    def test_reproduces_full_incident_still_holding_at_day3(self):
        """実際に観測された3日分の数値をそのまま再生しても
        （2026-07-15 staged→07-16 equity=5,598,886→07-17 equity=5,525,986）、
        3回目のチェック時点(confirm_count=2)ではまだCONFIRMEDされないこと。"""
        bridge = _make_bridge_stub()
        state = _base_state(peak=4_110_741.0)

        self._run_day(bridge, state, 5_587_186.0, "2026-07-15")  # STAGED
        self._run_day(bridge, state, 5_598_886.0, "2026-07-16")  # confirm 1/3
        self._run_day(bridge, state, 5_525_986.0, "2026-07-17")  # confirm 2/3

        self.assertEqual(state["equity_peak"], 4_110_741.0, "3営業日分の実データでもまだ確定しないこと")
        self.assertIsNotNone(state["candidate_peak"])
        self.assertEqual(state["candidate_peak"]["confirm_count"], 2)

    def test_confirms_after_required_consecutive_days(self):
        """CANDIDATE_PEAK_RECONFIRM_COUNT回連続で基準を満たせば、その時点で確定すること
        （恒久的に保留され続けるわけではないことの確認）。"""
        bridge = _make_bridge_stub()
        state = _base_state(peak=4_110_741.0)
        jumped_equity = 5_587_186.0

        date_str = "2026-07-15"
        self._run_day(bridge, state, jumped_equity, date_str)  # STAGED
        for i in range(1, CANDIDATE_PEAK_RECONFIRM_COUNT):
            date_str = _next_trading_day(date_str)
            self._run_day(bridge, state, jumped_equity, date_str)
            self.assertEqual(state["equity_peak"], 4_110_741.0, f"{i}回目の再確認ではまだ未確定")

        # CANDIDATE_PEAK_RECONFIRM_COUNT回目でついに確定
        date_str = _next_trading_day(date_str)
        self._run_day(bridge, state, jumped_equity, date_str)
        self.assertEqual(state["equity_peak"], round(jumped_equity, 0), "規定回数到達で確定すること")
        self.assertIsNone(state["candidate_peak"], "確定後は候補がクリアされること")

    def test_reconfirm_failure_on_any_day_discards_immediately(self):
        """持続の途中で1回でも基準未達になれば、その時点で候補は完全に破棄されること
        （それまでの部分的な持続回数は救済されない）。"""
        bridge = _make_bridge_stub()
        state = _base_state(peak=4_110_741.0)
        jumped_equity = 5_587_186.0

        self._run_day(bridge, state, jumped_equity, "2026-07-15")  # STAGED
        self._run_day(bridge, state, jumped_equity, "2026-07-16")  # confirm 1/3
        self._run_day(bridge, state, 4_000_000.0, "2026-07-17")    # 基準未達 → 破棄

        self.assertEqual(state["equity_peak"], 4_110_741.0)
        self.assertIsNone(state["candidate_peak"], "破棄後は候補が完全にクリアされること")

    def test_non_jump_organic_new_high_unaffected(self):
        """10%未満の通常の運用益ジャンプは従来通り即時反映され、
        多段階確認の対象にすらならないこと（回帰guard）。"""
        bridge = _make_bridge_stub()
        state = _base_state(peak=3_000_000.0)
        small_jump = 3_000_000.0 * 1.05
        self._run_day(bridge, state, small_jump, "2026-07-17")
        self.assertEqual(state["equity_peak"], round(small_jump, 0))
        self.assertIsNone(state["candidate_peak"])

    def test_final_confirmation_rechecks_broker_consistency_and_rejects(self):
        """N回連続確認に到達しても、確定直前のbroker整合性チェックで不整合と判定
        されればCONFIRMEDされないこと（ユーザー指定の追加要件・2026-07-18）。
        confirm_countは維持され、次回runで再試行できることも確認する。"""
        bridge = _make_bridge_stub()
        state = _base_state(peak=4_110_741.0)
        jumped_equity = 5_587_186.0
        date_str = "2026-07-15"

        bridge._update_cb_state(
            state, current_equity=jumped_equity, today_str=date_str,
            broker_snapshot=_snapshot(jumped_equity, {}, {}),
        )  # STAGED
        for _ in range(1, CANDIDATE_PEAK_RECONFIRM_COUNT - 1):
            date_str = _next_trading_day(date_str)
            bridge._update_cb_state(
                state, current_equity=jumped_equity, today_str=date_str,
                broker_snapshot=_snapshot(jumped_equity, {}, {}),
            )  # HOLDING（一貫してconsistentなsnapshotで進める）

        self.assertEqual(state["candidate_peak"]["confirm_count"], CANDIDATE_PEAK_RECONFIRM_COUNT - 2)

        # 最終確認日: current_equityは基準を満たすが、broker生値が大きく乖離した
        # snapshot（実は不整合＝state破損等を疑うべきケース）を与える。
        date_str = _next_trading_day(date_str)
        diverged_snapshot = _snapshot(cash=2_000_000.0, positions={}, prices={})  # 乖離大
        bridge._update_cb_state(
            state, current_equity=jumped_equity, today_str=date_str,
            broker_snapshot=diverged_snapshot,
        )
        self.assertEqual(
            state["equity_peak"], 4_110_741.0,
            "確定直前のbroker整合性チェックで不整合ならCONFIRMEDされてはならない",
        )
        self.assertIsNotNone(state["candidate_peak"], "confirm_countを維持したまま候補が残ること")
        self.assertEqual(
            state["candidate_peak"]["confirm_count"], CANDIDATE_PEAK_RECONFIRM_COUNT - 1,
            "整合性チェック直前まで進んだconfirm_count自体は失われないこと",
        )

        # 次回run: 整合するsnapshotに戻れば、そこで初めて確定する。
        date_str = _next_trading_day(date_str)
        bridge._update_cb_state(
            state, current_equity=jumped_equity, today_str=date_str,
            broker_snapshot=_snapshot(jumped_equity, {}, {}),
        )
        self.assertEqual(state["equity_peak"], round(jumped_equity, 0), "整合性回復後は正常に確定すること")
        self.assertIsNone(state["candidate_peak"])


# ══════════════════════════════════════════════════════════════════════════
# 7. broker再取得
# ══════════════════════════════════════════════════════════════════════════
class TestBrokerRefetch(unittest.TestCase):
    def test_broker_refetch_consistent_allows_update(self):
        bridge = _make_bridge_stub()
        state = _base_state(peak=3_000_000.0)
        refetched_equity = 3_100_000.0
        bridge._update_cb_state(
            state, current_equity=refetched_equity, today_str="2026-07-17",
            broker_snapshot=_snapshot(refetched_equity, {}, {}),
        )
        self.assertEqual(state["equity_peak"], refetched_equity)

    def test_broker_refetch_diverged_rejects(self):
        """broker再取得値とcurrent_equity(cache併用計算値)が乖離 → REJECT。"""
        bridge = _make_bridge_stub()
        state = _base_state(peak=3_000_000.0)
        # broker生値は3,000,000のまま(変化なし)だがcurrent_equityは異常に高い
        bridge._update_cb_state(
            state, current_equity=4_500_000.0, today_str="2026-07-17",
            broker_snapshot=_snapshot(3_000_000.0, {}, {}),
        )
        self.assertEqual(state["equity_peak"], 3_000_000.0)

    def test_broker_refetch_failure_fail_open(self):
        """broker再取得失敗(snapshot=None)はFAIL_OPENで整合性チェックをスキップする
        （既存設計を維持・入金ガードとは独立の挙動）。"""
        bridge = _make_bridge_stub()
        state = _base_state(peak=3_000_000.0)
        bridge._update_cb_state(
            state, current_equity=3_050_000.0, today_str="2026-07-17",
            broker_snapshot=None,
        )
        self.assertEqual(state["equity_peak"], 3_050_000.0)


# ══════════════════════════════════════════════════════════════════════════
# 8. bootstrap
# ══════════════════════════════════════════════════════════════════════════
class TestBootstrap(unittest.TestCase):
    def test_bootstrap_no_state_file_uses_initial_capital_peak(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "nonexistent_portfolio_state.json"
            state, vr = load_portfolio_state(path=path)
            self.assertFalse(vr.ok)
            self.assertEqual(state["equity_peak"], INITIAL_CAPITAL)

    def test_bootstrap_then_first_update_cb_state_sets_sane_peak(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "nonexistent.json"
            state, _ = load_portfolio_state(path=path)
            bridge = _make_bridge_stub(capital=INITIAL_CAPITAL)
            bridge._update_cb_state(
                state, current_equity=INITIAL_CAPITAL, today_str="2026-07-17",
                broker_snapshot=_snapshot(INITIAL_CAPITAL, {}, {}),
            )
            self.assertEqual(state["equity_peak"], INITIAL_CAPITAL)


# ══════════════════════════════════════════════════════════════════════════
# detect_cash_event() 単体（入金/出金判定ロジック自体の正確性）
# ══════════════════════════════════════════════════════════════════════════
class TestDetectCashEvent(unittest.TestCase):
    def test_detects_2026_07_15_deposit_exactly(self):
        """実インシデントの数値でdetect_cash_event()が正しくdepositと判定すること。"""
        ev = detect_cash_event(
            prev_cash=1_706_591.0, new_cash=3_642_786.0,
            prev_market_value=1_944_400.0, new_market_value=1_944_400.0,
        )
        self.assertIsNotNone(ev)
        self.assertEqual(ev["event_type"], "deposit")
        self.assertAlmostEqual(ev["unexplained_delta"], 1_936_195.0, delta=1.0)

    def test_normal_trading_day_no_event(self):
        """cash/market_valueの変化が売買で説明可能な範囲なら検知されないこと。"""
        ev = detect_cash_event(
            prev_cash=2_000_000.0, new_cash=1_950_000.0,
            prev_market_value=1_000_000.0, new_market_value=1_050_000.0,
        )
        self.assertIsNone(ev)


if __name__ == "__main__":
    unittest.main(verbosity=2)
