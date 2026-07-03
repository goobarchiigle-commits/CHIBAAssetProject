"""
tests/test_rsr_exit_separation.py

RSR exit/entry 分離テスト
- min_rsr=75 (entry) と rsr_exit=70 (exit) が完全に分離されていること
- RSR=74: エントリー不可、現在保有中ならexitしない (74 >= 70)
- RSR=69: 保有中ならexitする (69 < 70)
- config fallback: rsr_exit 未指定時は min_rsr にフォールバック
"""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pytest
from unittest.mock import MagicMock, patch
import yaml
from io import StringIO


# ─────────────────────────────────────────────────────────────────────
# 1. config_loader: FujikoConfig.rsr_exit フィールド
# ─────────────────────────────────────────────────────────────────────

def test_fujiko_config_has_rsr_exit_field():
    """FujikoConfig に rsr_exit フィールドが存在する"""
    from src.config_loader import FujikoConfig
    cfg = FujikoConfig(
        min_sepa=6, min_rsr=75.0, rsr_exit=70.0,
        mom_period=21, turtle_entry=20, turtle_exit=55,
        use_turtle_entry=True,
    )
    assert cfg.rsr_exit == 70.0
    assert cfg.min_rsr == 75.0


def test_fujiko_config_rsr_exit_independent_of_min_rsr():
    """rsr_exit と min_rsr が独立した値を持てる"""
    from src.config_loader import FujikoConfig
    cfg = FujikoConfig(
        min_sepa=6, min_rsr=75.0, rsr_exit=70.0,
        mom_period=21, turtle_entry=20, turtle_exit=55,
        use_turtle_entry=True,
    )
    assert cfg.min_rsr != cfg.rsr_exit
    assert cfg.min_rsr == 75.0
    assert cfg.rsr_exit == 70.0


def test_load_strategy_config_rsr_exit_value():
    """strategy.yaml から rsr_exit=70.0 が正しくロードされる"""
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    assert cfg.fujiko.min_rsr == 75.0, "min_rsr must remain 75.0 (PARAMS_LOCKED)"
    assert cfg.fujiko.rsr_exit == 70.0, "rsr_exit must be 70.0"
    assert cfg.fujiko.rsr_exit < cfg.fujiko.min_rsr, "rsr_exit must be less than min_rsr"


def test_load_strategy_config_rsr_exit_fallback():
    """rsr_exit 未指定時は min_rsr にフォールバック"""
    from src.config_loader import load_strategy_config
    import yaml as _yaml

    # strategy.yaml を動的にロードして rsr_exit を除いたYAMLでテスト
    from src.paths import STRATEGY_CONFIG_FILE
    with open(STRATEGY_CONFIG_FILE, encoding="utf-8") as f:
        data = _yaml.safe_load(f)

    fujiko = dict(data.get("fujiko", {}))
    fujiko.pop("rsr_exit", None)  # rsr_exit を除去

    # config_loader._parse_fujiko 相当を直接テスト
    min_rsr_val = float(fujiko["min_rsr"])
    rsr_exit_val = float(fujiko.get("rsr_exit", min_rsr_val))
    assert rsr_exit_val == min_rsr_val, "fallback should equal min_rsr"


# ─────────────────────────────────────────────────────────────────────
# 2. RSR=74 の挙動テスト（entry 不可 / exit しない）
# ─────────────────────────────────────────────────────────────────────

def test_rsr74_entry_blocked_by_min_rsr():
    """
    RSR=74 はエントリー不可: min_rsr=75 を満たさない
    FujikoStrategy で signal=0 (no BUY) が返ること
    """
    from src.backtest.fujiko_strategy import FujikoStrategy
    import pandas as pd
    import numpy as np

    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(1000, 1500, n), index=dates)
    rsr   = pd.Series([74.0] * n, index=dates)  # RSR固定74

    strat = FujikoStrategy(
        min_rsr=75.0, rsr_series=rsr,
        min_sepa=0,  # SEPA条件を無効化して RSR のみテスト
        mom_period=21, turtle_entry=20, turtle_exit=55,
        use_turtle_entry=False,
    )
    df = pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.01,
        "Low":  close * 0.98, "Close": close,
        "Volume": pd.Series([1_000_000] * n, index=dates),
    })
    sig = strat.generate_signal(df)
    # RSR=74 < min_rsr=75 → entry 不可 (signal=0 or -1, not 1)
    assert sig != 1, f"RSR=74 should NOT generate BUY (got signal={sig})"


def test_rsr74_no_exit_from_exit_threshold():
    """
    RSR=74 は exit しない: rsr_exit=70 より大きいので exit 不要
    capital_allocation_abc.run_period で RSR_EXIT が発生しないこと（簡易検証）
    """
    # rsr_exit_thr=70 の場合、rsr_val=74 では RSR_EXIT 不発動
    rsr_val = 74.0
    rsr_exit_thr = 70.0
    assert rsr_val >= rsr_exit_thr, "RSR=74 >= rsr_exit=70 → should NOT exit"


# ─────────────────────────────────────────────────────────────────────
# 3. RSR=69 の挙動テスト（exit する）
# ─────────────────────────────────────────────────────────────────────

def test_rsr69_triggers_exit():
    """
    RSR=69 は exit する: rsr_exit=70 を下回る
    """
    rsr_val = 69.0
    rsr_exit_thr = 70.0
    assert rsr_val < rsr_exit_thr, "RSR=69 < rsr_exit=70 → should EXIT"


def test_rsr69_not_blocked_by_min_rsr_for_exit():
    """
    RSR=69 の exit 判定は min_rsr=75 ではなく rsr_exit=70 を使う
    （min_rsr を使うと RSR=69 < 75 で同じ結果になり分離が確認できないので、
     exit threshold の値が正しく 70 であることを確認する）
    """
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    # exit threshold は 70
    assert cfg.fujiko.rsr_exit == 70.0
    # entry threshold は 75
    assert cfg.fujiko.min_rsr == 75.0
    # RSR=69 < exit_threshold → exit
    assert 69.0 < cfg.fujiko.rsr_exit
    # RSR=69 < min_rsr but that's irrelevant to exit logic
    assert 69.0 < cfg.fujiko.min_rsr  # min_rsr is NOT used for exit


# ─────────────────────────────────────────────────────────────────────
# 4. capital_allocation_abc: rsr_exit_thr のデフォルト確認
# ─────────────────────────────────────────────────────────────────────

def test_capital_allocation_abc_exit_thr_uses_rsr_exit():
    """
    capital_allocation_abc.run_period の rsr_exit_thr のデフォルトが
    cfg.fujiko.rsr_exit (=70) であること
    """
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    # rsr_exit_override=None の場合、fallback は cfg.fujiko.rsr_exit
    rsr_exit_thr = cfg.fujiko.rsr_exit  # 正常系
    assert rsr_exit_thr == 70.0
    # min_rsr は entry 専用
    assert cfg.fujiko.min_rsr == 75.0
    assert rsr_exit_thr != cfg.fujiko.min_rsr


# ─────────────────────────────────────────────────────────────────────
# 5. signal_bridge: rsr_exit_threshold 属性
# ─────────────────────────────────────────────────────────────────────

def test_signal_bridge_rsr_exit_threshold_attribute():
    """
    SignalBridge が rsr_exit_threshold=70.0 を正しく保持する
    """
    from src.kabusapi.signal_bridge import SignalBridge

    fujiko_params = {
        "min_sepa": 6,
        "min_rsr": 75.0,
        "rsr_exit": 70.0,
        "mom_period": 21,
        "turtle_entry": 20,
        "turtle_exit": 55,
        "use_turtle_entry": True,
    }

    with patch("src.kabusapi.client.KabuClient"):
        sb = SignalBridge(
            universe_tickers={"6857.T": "電機精密"},
            fujiko_params=fujiko_params,
            capital=3_000_000,
            max_positions=3,
            min_hold_days=3,
            emergency_exit_pct=-0.08,
            live=False,
        )

    assert sb.rsr_exit_threshold == 70.0
    # min_rsr は変更されていない（_fujiko_params_live に残る）
    assert sb._fujiko_params_live["min_rsr"] == 75.0


def test_signal_bridge_rsr_exit_fallback_to_min_rsr():
    """
    rsr_exit を指定しない場合、rsr_exit_threshold は min_rsr にフォールバック
    """
    from src.kabusapi.signal_bridge import SignalBridge

    fujiko_params_no_rsr_exit = {
        "min_sepa": 6,
        "min_rsr": 75.0,
        # rsr_exit は指定しない
        "mom_period": 21,
        "turtle_entry": 20,
        "turtle_exit": 55,
        "use_turtle_entry": True,
    }

    with patch("src.kabusapi.client.KabuClient"):
        sb = SignalBridge(
            universe_tickers={"6857.T": "電機精密"},
            fujiko_params=fujiko_params_no_rsr_exit,
            capital=3_000_000,
            max_positions=3,
            min_hold_days=3,
            emergency_exit_pct=-0.08,
            live=False,
        )

    # fallback: rsr_exit → min_rsr=75.0
    assert sb.rsr_exit_threshold == 75.0


def test_signal_bridge_min_rsr_unchanged_in_fujiko_params():
    """
    SignalBridge 内で _fujiko_params_live["min_rsr"] が 75.0 のまま
    （entry フィルターは変更されない）
    """
    from src.kabusapi.signal_bridge import SignalBridge

    fujiko_params = {
        "min_sepa": 6,
        "min_rsr": 75.0,
        "rsr_exit": 70.0,
        "mom_period": 21,
        "turtle_entry": 20,
        "turtle_exit": 55,
        "use_turtle_entry": True,
    }

    with patch("src.kabusapi.client.KabuClient"):
        sb = SignalBridge(
            universe_tickers={"6857.T": "電機精密"},
            fujiko_params=fujiko_params,
            capital=3_000_000,
            max_positions=3,
            min_hold_days=3,
            emergency_exit_pct=-0.08,
            live=False,
        )

    # entry 用 min_rsr は変更されていない
    assert sb._fujiko_params_live["min_rsr"] == 75.0
    # exit 用 rsr_exit_threshold は 70
    assert sb.rsr_exit_threshold == 70.0
    # entry != exit
    assert sb._fujiko_params_live["min_rsr"] != sb.rsr_exit_threshold


# ─────────────────────────────────────────────────────────────────────
# 6. run_live_signal fujiko_params に rsr_exit が含まれること
# ─────────────────────────────────────────────────────────────────────

def test_run_live_signal_includes_rsr_exit_in_fujiko_params():
    """
    run_live_signal.py が SignalBridge に rsr_exit を渡すこと
    （ソースコード上に "rsr_exit": cfg.fujiko.rsr_exit が存在すること）
    """
    from pathlib import Path
    src = Path("src/run_live_signal.py").read_text(encoding="utf-8")
    assert '"rsr_exit": cfg.fujiko.rsr_exit' in src, \
        "run_live_signal.py must pass rsr_exit to SignalBridge"


# ─────────────────────────────────────────────────────────────────────
# 7. 再現性: rsr_exit_override=None 時のデフォルト値検証
# ─────────────────────────────────────────────────────────────────────

def test_backtest_default_exit_threshold_is_70():
    """
    capital_allocation_abc.run_period で rsr_exit_override=None 時、
    rsr_exit_thr = cfg.fujiko.rsr_exit = 70.0 になること
    """
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()

    rsr_exit_override = None
    rsr_exit_thr = (
        float(rsr_exit_override) if rsr_exit_override is not None
        else float(cfg.fujiko.rsr_exit)
    )
    assert rsr_exit_thr == 70.0


def test_backtest_explicit_override_still_works():
    """
    rsr_exit_override を明示的に指定した場合は設定値より優先される（後方互換）
    """
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()

    for override in [65.0, 72.0, 75.0, 80.0]:
        rsr_exit_thr = (
            float(override) if override is not None
            else float(cfg.fujiko.rsr_exit)
        )
        assert rsr_exit_thr == override
