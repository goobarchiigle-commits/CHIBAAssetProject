"""
tests/test_rsr_exit_fujiko_compat.py

回帰テスト: rsr_exit を fujiko_params に渡しても FujikoStrategy が TypeError を起こさないこと
Root Cause: signal_bridge._fujiko_params_live に rsr_exit が混入し
            FujikoStrategy(**_fujiko_params_live) で TypeError が発生していた。
Fix: _fujiko_params_live 構築時に rsr_exit を除外する。
"""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pytest
from unittest.mock import patch


_FUJIKO_PARAMS_WITH_RSR_EXIT = {
    "min_sepa": 6,
    "min_rsr": 75.0,
    "rsr_exit": 70.0,          # ← これが FujikoStrategy に渡ると TypeError
    "mom_period": 21,
    "turtle_entry": 20,
    "turtle_exit": 55,
    "use_turtle_entry": True,
}


# ─────────────────────────────────────────────────────────────────────
# 1. SignalBridge._fujiko_params_live に rsr_exit が含まれないこと
# ─────────────────────────────────────────────────────────────────────

def test_fujiko_params_live_excludes_rsr_exit():
    """_fujiko_params_live に rsr_exit が含まれない = FujikoStrategy に渡らない"""
    from src.kabusapi.signal_bridge import SignalBridge

    with patch("src.kabusapi.client.KabuClient"):
        sb = SignalBridge(
            universe_tickers={"6857.T": "電機精密"},
            fujiko_params=_FUJIKO_PARAMS_WITH_RSR_EXIT,
            capital=3_000_000,
            max_positions=3,
            min_hold_days=3,
            emergency_exit_pct=-0.08,
            live=False,
        )

    assert "rsr_exit" not in sb._fujiko_params_live, (
        "_fujiko_params_live に rsr_exit が混入している → FujikoStrategy で TypeError になる"
    )


# ─────────────────────────────────────────────────────────────────────
# 2. rsr_exit_threshold は正しく保持されること
# ─────────────────────────────────────────────────────────────────────

def test_rsr_exit_threshold_preserved():
    """_fujiko_params_live から除外しても rsr_exit_threshold は正しく設定される"""
    from src.kabusapi.signal_bridge import SignalBridge

    with patch("src.kabusapi.client.KabuClient"):
        sb = SignalBridge(
            universe_tickers={"6857.T": "電機精密"},
            fujiko_params=_FUJIKO_PARAMS_WITH_RSR_EXIT,
            capital=3_000_000,
            max_positions=3,
            min_hold_days=3,
            emergency_exit_pct=-0.08,
            live=False,
        )

    assert sb.rsr_exit_threshold == 70.0
    # min_rsr はそのまま残る (entry 用)
    assert sb._fujiko_params_live["min_rsr"] == 75.0


# ─────────────────────────────────────────────────────────────────────
# 3. FujikoStrategy が _fujiko_params_live を受け取っても TypeError にならないこと
# ─────────────────────────────────────────────────────────────────────

def test_fujiko_strategy_no_typeerror_with_live_params():
    """FujikoStrategy(**_fujiko_params_live) が TypeError を起こさない"""
    from src.kabusapi.signal_bridge import SignalBridge
    from src.backtest.fujiko_strategy import FujikoStrategy

    with patch("src.kabusapi.client.KabuClient"):
        sb = SignalBridge(
            universe_tickers={"6857.T": "電機精密"},
            fujiko_params=_FUJIKO_PARAMS_WITH_RSR_EXIT,
            capital=3_000_000,
            max_positions=3,
            min_hold_days=3,
            emergency_exit_pct=-0.08,
            live=False,
        )

    # これが TypeError を起こさなければ修正が正しい
    try:
        strat = FujikoStrategy(**sb._fujiko_params_live)
    except TypeError as e:
        pytest.fail(f"FujikoStrategy(**_fujiko_params_live) が TypeError: {e}")


# ─────────────────────────────────────────────────────────────────────
# 4. rsr_exit なしの fujiko_params でも動作すること（後方互換）
# ─────────────────────────────────────────────────────────────────────

def test_fujiko_params_without_rsr_exit_fallback():
    """rsr_exit を指定しない fujiko_params でも SignalBridge が正常動作"""
    from src.kabusapi.signal_bridge import SignalBridge

    params_no_exit = {k: v for k, v in _FUJIKO_PARAMS_WITH_RSR_EXIT.items() if k != "rsr_exit"}

    with patch("src.kabusapi.client.KabuClient"):
        sb = SignalBridge(
            universe_tickers={"6857.T": "電機精密"},
            fujiko_params=params_no_exit,
            capital=3_000_000,
            max_positions=3,
            min_hold_days=3,
            emergency_exit_pct=-0.08,
            live=False,
        )

    # fallback: min_rsr の値が使われる
    assert sb.rsr_exit_threshold == 75.0
    assert "rsr_exit" not in sb._fujiko_params_live


# ─────────────────────────────────────────────────────────────────────
# 5. logging の %,.0f が含まれないこと（ValueError 回帰防止）
# ─────────────────────────────────────────────────────────────────────

def test_no_invalid_logging_format_in_run_live_signal():
    """run_live_signal.py に %,.0f / %,d が残っていないこと"""
    from pathlib import Path
    src = Path("src/run_live_signal.py").read_text(encoding="utf-8")
    bad_patterns = ["%,.0f", "%,d", "%,.1f", "%,.2f"]
    found = [p for p in bad_patterns if p in src]
    assert not found, (
        f"run_live_signal.py に無効な logging フォーマット文字が残っている: {found}"
    )


def test_no_invalid_logging_format_in_live_equity_fetcher():
    """live_equity_fetcher.py に %,.0f が残っていないこと"""
    from pathlib import Path
    src = Path("src/broker/live_equity_fetcher.py").read_text(encoding="utf-8")
    bad_patterns = ["%,.0f", "%,d"]
    found = [p for p in bad_patterns if p in src]
    assert not found, (
        f"live_equity_fetcher.py に無効な logging フォーマット文字が残っている: {found}"
    )
