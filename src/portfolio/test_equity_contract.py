"""
src/portfolio/test_equity_contract.py
equity 計算の不変条件テスト。

目的:
    startup / watchdog / CB / dry-run / reconciliation の全パスが
    同一のポートフォリオ状態に対して同一の equity を返すことを保証する。

実行:
    cd C:/ai-trading
    python -m pytest src/portfolio/test_equity_contract.py -v
または:
    python src/portfolio/test_equity_contract.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.portfolio.equity import (
    PEAK_ANOMALY_RATIO,
    PEAK_CONSISTENCY_ABS_THRESHOLD,
    PEAK_CONSISTENCY_PCT_THRESHOLD,
    SAFE_WARN_CONFIRM_REQUIRED,
    BrokerEquityInvariantError,
    append_peak_audit,
    assert_broker_equity_invariant,
    check_broker_consistency,
    check_peak_anomaly,
    compute_live_equity,
)
from src.portfolio.state_store import BrokerSnapshot


# ── 共通フィクスチャ ──────────────────────────────────────────────────────────

_CASH = 1_500_000.0

_POSITIONS = {
    "7203.T": {"qty": 100, "avg_price": 2500.0},   # 評価額 ¥250,000
    "6758.T": {"qty": 200, "avg_price": 1200.0},   # 評価額 ¥240,000
}
# market_value = 250,000 + 240,000 = 490,000
# equity       = 1,500,000 + 490,000 = 1,990,000
_EXPECTED_EQUITY = 1_990_000.0

def _snapshot_from_positions(cash: float, positions: dict[str, dict]) -> BrokerSnapshot:
    """テスト用: {sym: {"qty","avg_price"}} 形式から BrokerSnapshot を組み立てる。
    market_values は avg_price と同値にする（broker CurrentPrice ≒ 取得単価の単純ケース）。"""
    return BrokerSnapshot(
        cash=cash,
        positions={s: int(p["qty"]) for s, p in positions.items()},
        avg_costs={s: float(p["avg_price"]) for s, p in positions.items()},
        market_values={s: float(p["avg_price"]) for s, p in positions.items()},
        equity=0.0, ts="2026-07-18T08:44:00+0900", source="broker",
        api_health={"positions_ok": True, "wallet_ok": True},
    )


class TestEquityInvariance(unittest.TestCase):
    """
    同一ポートフォリオ状態に対して全パスが同一 equity を返すこと
    （Broker-as-Sole-SSOT: 入力が BrokerSnapshot のみになったため、
    パス間の drift はそもそも構造的に発生し得ない）。
    """

    def _compute(self, mode: str) -> float:
        snap = _snapshot_from_positions(_CASH, _POSITIONS)
        return compute_live_equity(snapshot=snap, mode=mode, persist_snapshot=False)

    def test_live_equity(self):
        eq = self._compute("live")
        self.assertAlmostEqual(eq, _EXPECTED_EQUITY, places=0)

    def test_dry_equity(self):
        eq = self._compute("dry")
        self.assertAlmostEqual(eq, _EXPECTED_EQUITY, places=0)

    def test_startup_equity(self):
        eq = self._compute("startup")
        self.assertAlmostEqual(eq, _EXPECTED_EQUITY, places=0)

    def test_reconcile_equity(self):
        eq = self._compute("reconcile")
        self.assertAlmostEqual(eq, _EXPECTED_EQUITY, places=0)

    def test_cb_equity(self):
        eq = self._compute("cb")
        self.assertAlmostEqual(eq, _EXPECTED_EQUITY, places=0)

    def test_all_paths_identical(self):
        """全パスが数値的に同一であること（パス依存の drift なし）。"""
        modes = ["live", "dry", "startup", "reconcile", "cb", "watchdog", "shadow"]
        values = [self._compute(m) for m in modes]
        for i, (m, v) in enumerate(zip(modes, values)):
            with self.subTest(mode=m):
                self.assertAlmostEqual(
                    v, _EXPECTED_EQUITY, places=0,
                    msg=f"{m} path returned {v} != expected {_EXPECTED_EQUITY}",
                )

    def test_zero_cash(self):
        snap = _snapshot_from_positions(0.0, _POSITIONS)
        eq = compute_live_equity(snapshot=snap, persist_snapshot=False)
        self.assertAlmostEqual(eq, 490_000.0, places=0)

    def test_empty_positions(self):
        snap = _snapshot_from_positions(_CASH, {})
        eq = compute_live_equity(snapshot=snap, persist_snapshot=False)
        self.assertAlmostEqual(eq, _CASH, places=0)

    def test_zero_qty_excluded(self):
        positions_with_zero = {**_POSITIONS, "0000.T": {"qty": 0, "avg_price": 999.0}}
        snap = _snapshot_from_positions(_CASH, positions_with_zero)
        eq = compute_live_equity(snapshot=snap, persist_snapshot=False)
        self.assertAlmostEqual(eq, _EXPECTED_EQUITY, places=0)


class TestPeakAnomalyDetection(unittest.TestCase):
    """check_peak_anomaly() の動作確認。"""

    def test_normal_ratio_no_anomaly(self):
        is_anom, count, msg = check_peak_anomaly(
            equity_peak=2_000_000, current_equity=1_900_000
        )
        self.assertFalse(is_anom)
        self.assertEqual(count, 0)

    def test_exactly_at_threshold_no_anomaly(self):
        # ratio = 1.25 exactly should NOT trigger (> not >=)
        is_anom, count, _ = check_peak_anomaly(
            equity_peak=1_250_000, current_equity=1_000_000
        )
        self.assertFalse(is_anom)

    def test_above_threshold_triggers(self):
        # ratio = 1.30 > 1.25 → anomaly
        is_anom, count, msg = check_peak_anomaly(
            equity_peak=3_670_000, current_equity=2_800_000
        )
        self.assertTrue(is_anom)
        self.assertEqual(count, 1)
        self.assertIn("PEAK_ANOMALY", msg)

    def test_counter_increments(self):
        _, c1, _ = check_peak_anomaly(3_670_000, 2_800_000, safe_warn_count=0)
        _, c2, _ = check_peak_anomaly(3_670_000, 2_800_000, safe_warn_count=c1)
        _, c3, _ = check_peak_anomaly(3_670_000, 2_800_000, safe_warn_count=c2)
        self.assertEqual(c3, SAFE_WARN_CONFIRM_REQUIRED)

    def test_counter_resets_when_normal(self):
        _, count, _ = check_peak_anomaly(
            equity_peak=3_000_000, current_equity=3_000_000, safe_warn_count=2
        )
        self.assertEqual(count, 0)

    def test_fake_peak_scenario(self):
        """sync_positions が fake 3.67M peak を書いたシナリオ。"""
        is_anom, count, msg = check_peak_anomaly(
            equity_peak=3_670_000,  # 偽の peak
            current_equity=3_000_000,
        )
        # ratio = 3.67 / 3.0 = 1.223 — 1.25 未満なので WARN にならない場合もある
        # ただし 3.67 / 2.9M = 1.265 → 閾値超え
        ratio = 3_670_000 / 3_000_000
        if ratio > PEAK_ANOMALY_RATIO:
            self.assertTrue(is_anom)
        else:
            # ratio < 1.25 のケースは WARN 不要（通常 CB が先に発動）
            self.assertFalse(is_anom)

    def test_zero_equity_safe(self):
        """ゼロ除算しないこと。"""
        is_anom, count, _ = check_peak_anomaly(3_000_000, 0.0)
        self.assertFalse(is_anom)

    def test_zero_peak_safe(self):
        is_anom, count, _ = check_peak_anomaly(0.0, 3_000_000)
        self.assertFalse(is_anom)


class TestSafeWarnIntegration(unittest.TestCase):
    """SAFE_WARN 状態遷移の統合テスト（signal_bridge を使わずに equity.py 単体で検証）。"""

    def _simulate_n_runs(self, peak: float, equity: float, n: int) -> tuple[bool, int]:
        count = 0
        is_anom = False
        for _ in range(n):
            is_anom, count, _ = check_peak_anomaly(peak, equity, count)
            if not is_anom:
                break
        return is_anom, count

    def test_safe_warn_escalation_after_n_runs(self):
        """N 連続確認で SAFE_WARN → CB_ACTIVE 昇格が検出されること。"""
        # ratio = 4M / 2M = 2.0 > 1.25 → 毎回 anomaly
        is_anom, count = self._simulate_n_runs(4_000_000, 2_000_000, SAFE_WARN_CONFIRM_REQUIRED)
        self.assertTrue(is_anom)
        self.assertEqual(count, SAFE_WARN_CONFIRM_REQUIRED)

    def test_safe_warn_clears_on_recovery(self):
        """一旦 anomaly が検出されても比率が正常化したらリセットされること。"""
        _, count_before = self._simulate_n_runs(4_000_000, 2_000_000, 2)
        is_anom, count_after, _ = check_peak_anomaly(3_000_000, 3_000_000, count_before)
        self.assertFalse(is_anom)
        self.assertEqual(count_after, 0)


def _snapshot(cash: float, positions: dict[str, int], prices: dict[str, float]) -> BrokerSnapshot:
    return BrokerSnapshot(
        cash=cash, positions=positions, avg_costs=prices, market_values=prices,
        equity=0.0, ts="2026-07-03T08:44:00+0900", source="broker",
        api_health={"positions_ok": True, "wallet_ok": True},
    )


class TestBrokerConsistencyCheck(unittest.TestCase):
    """check_broker_consistency() — EQUITY_PEAK_HARDENING (2026-07-03)。"""

    def test_none_snapshot_fail_open(self):
        is_consistent, broker_equity, _ = check_broker_consistency(4_000_000.0, None)
        self.assertTrue(is_consistent)
        self.assertIsNone(broker_equity)

    def test_matching_values_consistent(self):
        snap = _snapshot(cash=1_000_000.0, positions={"7203.T": 100}, prices={"7203.T": 2500.0})
        # broker_equity = 1,000,000 + 100*2500 = 1,250,000
        is_consistent, broker_equity, _ = check_broker_consistency(1_250_000.0, snap)
        self.assertTrue(is_consistent)
        self.assertEqual(broker_equity, 1_250_000.0)

    def test_large_divergence_rejected(self):
        snap = _snapshot(cash=1_000_000.0, positions={}, prices={})
        # broker_equity=1,000,000 vs current_equity=2,000,000 (差額・比率とも閾値超)
        diverged_equity = 1_000_000.0 + PEAK_CONSISTENCY_ABS_THRESHOLD * 2
        is_consistent, broker_equity, msg = check_broker_consistency(diverged_equity, snap)
        self.assertFalse(is_consistent)
        self.assertIn("EQUITY_PEAK_REJECT", msg)

    def test_small_absolute_diff_within_pct_threshold_consistent(self):
        """絶対額が閾値超でも比率が小さければAND条件によりrejectしない。"""
        snap = _snapshot(cash=50_000_000.0, positions={}, prices={})
        # diff=400,000 (> ABS閾値) だが diff_pct = 400,000/50,000,000 = 0.8% (< PCT閾値)
        is_consistent, _, _ = check_broker_consistency(50_400_000.0, snap)
        self.assertTrue(is_consistent)


class TestBrokerEquityInvariant(unittest.TestCase):
    """assert_broker_equity_invariant() — 資産計算経路の再分岐検知（2026-07-19）。"""

    def test_matching_equity_does_not_raise(self):
        snap = _snapshot(cash=1_000_000.0, positions={"7203.T": 100}, prices={"7203.T": 2500.0})
        assert_broker_equity_invariant(snap, 1_250_000.0)  # 例外を投げないことを確認

    def test_mismatched_equity_raises(self):
        snap = _snapshot(cash=1_000_000.0, positions={"7203.T": 100}, prices={"7203.T": 2500.0})
        with self.assertRaises(BrokerEquityInvariantError):
            assert_broker_equity_invariant(snap, 1_883_200.0)  # 実インシデントの過大値相当

    def test_within_one_yen_rounding_tolerance_ok(self):
        snap = _snapshot(cash=1_000_000.0, positions={"7203.T": 100}, prices={"7203.T": 2500.0})
        assert_broker_equity_invariant(snap, 1_250_000.5)  # 丸め誤差の範囲内

    def test_empty_positions_matches_cash_only(self):
        snap = _snapshot(cash=3_642_786.0, positions={}, prices={})
        assert_broker_equity_invariant(snap, 3_642_786.0)

    def test_error_message_includes_both_values(self):
        snap = _snapshot(cash=1_000_000.0, positions={}, prices={})
        try:
            assert_broker_equity_invariant(snap, 5_000_000.0)
            self.fail("BrokerEquityInvariantError が発生しなかった")
        except BrokerEquityInvariantError as e:
            self.assertIn("1,000,000", str(e))
            self.assertIn("5,000,000", str(e))


class TestPeakAuditSink(unittest.TestCase):
    """append_peak_audit() — durable JSONL 監査シンク（EQUITY_PEAK_HARDENING）。"""

    def test_append_writes_jsonl_record(self):
        import json
        import tempfile
        from pathlib import Path
        import src.portfolio.equity as eq_mod

        with tempfile.TemporaryDirectory() as td:
            audit_file = Path(td) / "equity_peak_audit.jsonl"
            orig = eq_mod._PEAK_AUDIT_FILE
            eq_mod._PEAK_AUDIT_FILE = audit_file
            try:
                append_peak_audit(
                    action="APPLIED", old_peak=3_000_000.0, new_peak=3_500_000.0,
                    current_equity=3_500_000.0, broker_equity=3_499_000.0,
                    caller="_update_cb_state", reason="new_high", diag="consistent",
                    trading_date="2026-07-03", mode="live", pid=1234, run_id="test_run",
                )
                lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), 1)
                record = json.loads(lines[0])
                self.assertEqual(record["action"], "APPLIED")
                self.assertEqual(record["new_peak"], 3_500_000.0)
                self.assertEqual(record["mode"], "live")
            finally:
                eq_mod._PEAK_AUDIT_FILE = orig

    def test_append_failure_is_non_fatal(self):
        """書込み失敗時も例外を送出しない（house convention: FAIL_OPEN）。"""
        import src.portfolio.equity as eq_mod
        from pathlib import Path

        orig = eq_mod._PEAK_AUDIT_FILE
        # 存在しえないディレクトリ（ドライブレター混入）で書込み失敗を誘発
        eq_mod._PEAK_AUDIT_FILE = Path("\0invalid") / "audit.jsonl"
        try:
            append_peak_audit(
                action="APPLIED", old_peak=1.0, new_peak=2.0, current_equity=2.0,
                broker_equity=None, caller="x", reason="x", diag="x",
                trading_date="2026-07-03", mode="live", pid=1, run_id="x",
            )  # 例外を投げないことを確認するのみ
        finally:
            eq_mod._PEAK_AUDIT_FILE = orig


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
