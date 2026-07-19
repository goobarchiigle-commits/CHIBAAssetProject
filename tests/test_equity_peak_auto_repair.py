"""
tests/test_equity_peak_auto_repair.py

Phase 1-3 の自動修復テスト:
- rebuild_equity_peak()
- PEAK_ANOMALY 時の CB FAIL_SAFE (NORMAL 維持)
- CB_ACTIVE 時の自動修復
- 乖離警告閾値

2026-07-19 追記: _cb_guard_compensation()（settlement lag補償・PendingOrderState
機構）はBroker-as-Sole-SSOTの方針に基づき完全撤去された。同機構を検証していた
テスト3件（test_cb_guard_compensation_next_day_pending等）は検証対象の関数が
削除されたため本ファイルから削除した。詳細はtest_cb_settlement_guard.pyの
モジュールdocstringを参照。
"""
from __future__ import annotations
import sys, json, tempfile, os
sys.stdout.reconfigure(encoding="utf-8")

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────
# 1. rebuild_equity_peak — スナップショットから中央値を返す
# ─────────────────────────────────────────────────────────────────────

def test_rebuild_equity_peak_returns_median():
    """直近スナップショットの中央値が返る"""
    from src.portfolio.equity import rebuild_equity_peak
    import json

    snapshots = [
        {"equity": 4000000.0, "equity_peak": 4000000.0},
        {"equity": 4010000.0, "equity_peak": 4010000.0},
        {"equity": 4020000.0, "equity_peak": 4020000.0},
        {"equity": 4089109.0, "equity_peak": 4089109.0},  # 異常値(stale run)
        {"equity": 4015209.0, "equity_peak": 4015209.0},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        for s in snapshots:
            f.write(json.dumps(s) + "\n")
        tmp_path = Path(f.name)

    try:
        import src.portfolio.equity as eq_mod
        orig_file = eq_mod._SNAPSHOT_FILE
        eq_mod._SNAPSHOT_FILE = tmp_path

        result = rebuild_equity_peak(current_equity=3_184_309.0, n_entries=10)

        # 中央値は 4,015,209 or thereabouts (median of sorted [4M, 4.01M, 4.015M, 4.02M, 4.089M])
        assert result >= 3_184_309.0, "rebuild peak は current_equity 以上でなければならない"
        assert result < 4_089_109.0, "異常峰値 4,089,109 は中央値では採用されない"
    finally:
        eq_mod._SNAPSHOT_FILE = orig_file
        tmp_path.unlink(missing_ok=True)


def test_rebuild_equity_peak_fallback_on_empty():
    """スナップショット不足時は current_equity を返す"""
    from src.portfolio.equity import rebuild_equity_peak
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        f.write("")  # 空ファイル
        tmp_path = Path(f.name)

    try:
        import src.portfolio.equity as eq_mod
        orig_file = eq_mod._SNAPSHOT_FILE
        eq_mod._SNAPSHOT_FILE = tmp_path
        result = rebuild_equity_peak(current_equity=3_200_000.0)
        assert result == 3_200_000.0
    finally:
        eq_mod._SNAPSHOT_FILE = orig_file
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────
# 3. commit_broker_snapshot — settlement lag 検出 WARNING
# ─────────────────────────────────────────────────────────────────────

def test_commit_broker_snapshot_warns_on_orphaned_positions():
    """entry_dates に存在して broker にない銘柄を WARN する"""
    from src.portfolio.state_store import commit_broker_snapshot, BrokerSnapshot

    state = {
        "position_entry_dates":  {"6981.T": "2026-04-28", "5301.T": "2026-06-04"},
        "position_entry_prices": {"6981.T": 4864.0, "5301.T": 1852.5},
        "position_qtys":         {"6981.T": 100, "5301.T": 400},
        "snapshot_hash":         None,
    }
    snap = BrokerSnapshot(
        cash          = 1_427_109.0,
        positions     = {"6981.T": 100},  # 5301.T が broker に反映されていない
        avg_costs     = {"6981.T": 4864.0},
        market_values = {"6981.T": 4864.0},
        equity        = 1_913_509.0,
        ts            = "2026-06-05T08:43:00+0900",
        source        = "broker",
        api_health    = {"positions_ok": True, "wallet_ok": True},
    )

    import logging
    with patch("src.portfolio.state_store.logger") as mock_log:
        commit_broker_snapshot(state, snap)
        # SETTLEMENT_LAG_DETECTED の WARNING が発行されること
        warn_calls = [str(c) for c in mock_log.warning.call_args_list]
        assert any("SETTLEMENT_LAG_DETECTED" in c for c in warn_calls), (
            f"SETTLEMENT_LAG_DETECTED warning が出ていない。calls={warn_calls}"
        )


# ─────────────────────────────────────────────────────────────────────
# 4. 乖離警告閾値の数値確認
# ─────────────────────────────────────────────────────────────────────

def test_divergence_threshold_values():
    """乖離検出の閾値が正しく設定されている"""
    # signal_bridge.py の _DIVERGE_PCT / _DIVERGE_ABS の確認
    from pathlib import Path
    src = Path("src/kabusapi/signal_bridge.py").read_text(encoding="utf-8")
    assert "_DIVERGE_PCT" in src and "0.05" in src
    assert "_DIVERGE_ABS" in src and "300_000" in src
