"""
tests/test_entry_timing_backtest_compat.py

Entry Timing バックテスト互換性テスト

保証: フラグOFF時 (et_enabled=False) は現行と完全一致

Run:
    cd C:/ai-trading
    python -m pytest tests/test_entry_timing_backtest_compat.py -v
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pytest
import numpy as np


# ─────────────────────────────────────────────────────────────────────
#  Smoke: ET import available
# ─────────────────────────────────────────────────────────────────────

def test_et_available():
    """Entry Timing Engine がインポート可能であること。"""
    from src.entry.entry_timing_engine import (
        build_entry_timing_input_from_df,
        compute_entry_timing,
        CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH,
    )
    assert CONFIDENCE_LOW    == "LOW"
    assert CONFIDENCE_MEDIUM == "MEDIUM"
    assert CONFIDENCE_HIGH   == "HIGH"


def test_et_bt_helper_importable():
    """バックテスト ET ヘルパーが capital_allocation_abc からインポート可能。"""
    from src.backtest.capital_allocation_abc import _compute_et_bt, _ET_AVAILABLE
    assert callable(_compute_et_bt)
    assert isinstance(_ET_AVAILABLE, bool)


def test_et_bt_helper_empty_input():
    """空の buy_cands は空の dict を返す。"""
    from src.backtest.capital_allocation_abc import _compute_et_bt
    import numpy as np, pandas as pd

    result = _compute_et_bt(
        buy_cands=[],
        signal_date=pd.Timestamp("2023-01-05"),
        universe_raw={},
        rsr_df=pd.DataFrame(),
        rsr_mat=np.zeros((1, 1), dtype=np.float32),
        i=0,
        sym_to_i={},
        active_syms=[],
    )
    assert result == {}


def test_et_bt_helper_unknown_symbol():
    """universe_raw に存在しない銘柄は FAIL_OPEN (空エントリー)。"""
    from src.backtest.capital_allocation_abc import _compute_et_bt
    import numpy as np, pandas as pd

    result = _compute_et_bt(
        buy_cands=[(80.0, "UNKNOWN.T")],
        signal_date=pd.Timestamp("2023-01-05"),
        universe_raw={},
        rsr_df=pd.DataFrame(),
        rsr_mat=np.array([[80.0]], dtype=np.float32),
        i=0,
        sym_to_i={"UNKNOWN.T": 0},
        active_syms=["UNKNOWN.T"],
    )
    # FAIL_OPEN: unknown symbol → not in result (graceful skip)
    assert "UNKNOWN.T" not in result


# ─────────────────────────────────────────────────────────────────────
#  run_period signature compatibility
# ─────────────────────────────────────────────────────────────────────

def test_run_period_accepts_et_params():
    """run_period が ET パラメータを受け付ける（TypeError なし）。"""
    import inspect
    from src.backtest.capital_allocation_abc import run_period
    sig = inspect.signature(run_period)
    params = sig.parameters
    assert "et_enabled"      in params, "et_enabled param missing"
    assert "et_block_low"    in params, "et_block_low param missing"
    assert "et_boost_weight" in params, "et_boost_weight param missing"


def test_run_period_defaults_are_off():
    """ET パラメータのデフォルトは全て OFF / ゼロであること。"""
    import inspect
    from src.backtest.capital_allocation_abc import run_period
    sig = inspect.signature(run_period)
    p = sig.parameters
    assert p["et_enabled"].default      is False,  "et_enabled default must be False"
    assert p["et_block_low"].default    is False,   "et_block_low default must be False"
    assert p["et_boost_weight"].default == 0.0,     "et_boost_weight default must be 0.0"


# ─────────────────────────────────────────────────────────────────────
#  signal_bridge block_low_confidence implementation
# ─────────────────────────────────────────────────────────────────────

def test_signal_bridge_has_block_low_confidence():
    """signal_bridge.py に block_low_confidence ロジックが実装されていること。"""
    with open("src/kabusapi/signal_bridge.py", encoding="utf-8") as f:
        content = f.read()
    assert "block_low_confidence" in content
    assert '_et_results[sym].confidence != "LOW"' in content or \
           "_et_results[sym].confidence != CONFIDENCE_LOW" in content


def test_block_low_default_off_in_strategy_yaml():
    """strategy.yaml で block_low_confidence がデフォルト false。"""
    with open("src/configs/strategy.yaml", encoding="utf-8") as f:
        content = f.read()
    assert "block_low_confidence: false" in content


# ─────────────────────────────────────────────────────────────────────
#  ET engine: score ranges and FAIL_OPEN
# ─────────────────────────────────────────────────────────────────────

def test_compute_et_score_range():
    """ET score は 0-100 の範囲内。"""
    from src.entry.entry_timing_engine import (
        EntryTimingInput, compute_entry_timing,
    )
    import random
    random.seed(42)

    inp = EntryTimingInput(
        symbol              = "TEST.T",
        rsr                 = 82.0,
        rsr_momentum        = 2.0,
        sepa_score          = 1,
        closes              = [100.0 + i * 0.5 for i in range(65)],
        highs               = [101.0 + i * 0.5 for i in range(65)],
        lows                = [99.0  + i * 0.5 for i in range(65)],
        volumes             = [1_000_000.0] * 65,
        rsr_series          = [80.0, 81.0, 82.0, 82.5, 83.0],
        trend_cluster_level = 0,
        universe_rsr_values = [70.0, 75.0, 80.0, 85.0, 90.0],
    )
    result = compute_entry_timing(inp)
    assert 0.0 <= result.score <= 100.0, f"score out of range: {result.score}"
    assert result.confidence in {"HIGH", "MEDIUM", "LOW"}
    assert result.action     in {"IMMEDIATE", "NORMAL", "WATCH"}


def test_compute_et_fail_open_on_bad_input():
    """不正な入力でも FAIL_OPEN (例外を投げない)。"""
    from src.entry.entry_timing_engine import EntryTimingInput, compute_entry_timing

    inp = EntryTimingInput(
        symbol="BAD.T",
        rsr=0.0, rsr_momentum=0.0, sepa_score=0,
        closes=[], highs=[], lows=[], volumes=[],
    )
    result = compute_entry_timing(inp)
    assert result is not None
    assert 0.0 <= result.score <= 100.0


def test_apply_entry_timing_boost_zero_weight():
    """boost_weight=0 では composite に変化なし。"""
    from src.entry.entry_timing_engine import (
        EntryTimingInput, compute_entry_timing, apply_entry_timing_boost,
    )
    inp = EntryTimingInput(
        symbol="A.T", rsr=80.0, rsr_momentum=1.0, sepa_score=0,
        closes=[100.0]*65, highs=[101.0]*65, lows=[99.0]*65, volumes=[1e6]*65,
    )
    result = compute_entry_timing(inp)
    base = 100.0
    boosted = apply_entry_timing_boost(base, result, boost_weight=0.0, enabled=True)
    assert boosted == base, "boost_weight=0 should not change composite"


def test_apply_entry_timing_boost_disabled():
    """et_enabled=False では composite に変化なし。"""
    from src.entry.entry_timing_engine import (
        EntryTimingInput, compute_entry_timing, apply_entry_timing_boost,
    )
    inp = EntryTimingInput(
        symbol="A.T", rsr=80.0, rsr_momentum=1.0, sepa_score=0,
        closes=[100.0]*65, highs=[101.0]*65, lows=[99.0]*65, volumes=[1e6]*65,
    )
    result = compute_entry_timing(inp)
    base = 100.0
    boosted = apply_entry_timing_boost(base, result, boost_weight=0.06, enabled=False)
    assert boosted == base, "enabled=False should not change composite"


# ─────────────────────────────────────────────────────────────────────
#  _compute_et_bt: FAIL_OPEN with real OHLCV-like data
# ─────────────────────────────────────────────────────────────────────

def test_et_bt_helper_with_synthetic_data():
    """合成データで _compute_et_bt が動作し EntryTimingResult を返す。"""
    from src.backtest.capital_allocation_abc import _compute_et_bt, _ET_AVAILABLE
    if not _ET_AVAILABLE:
        pytest.skip("Entry Timing engine not available")

    import numpy as np, pandas as pd
    from datetime import date, timedelta

    # 合成OHLCV (65 trading days)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close_vals = np.linspace(1000, 1200, n)
    df = pd.DataFrame({
        "Open":   close_vals * 0.995,
        "High":   close_vals * 1.01,
        "Low":    close_vals * 0.99,
        "Close":  close_vals,
        "Volume": [1_000_000] * n,
    }, index=dates)

    universe_raw = {"TEST.T": {"df": df}}

    n_days = len(dates)
    rsr_mat = np.full((n_days, 1), 80.0, dtype=np.float32)
    sym_to_i = {"TEST.T": 0}
    active_syms = ["TEST.T"]

    rsr_series = pd.Series(np.linspace(75, 82, n_days), index=dates)
    rsr_df = pd.DataFrame({"TEST.T": rsr_series})

    signal_date = dates[70]
    buy_cands = [(80.0, "TEST.T")]

    result = _compute_et_bt(
        buy_cands=buy_cands,
        signal_date=signal_date,
        universe_raw=universe_raw,
        rsr_df=rsr_df,
        rsr_mat=rsr_mat,
        i=70,
        sym_to_i=sym_to_i,
        active_syms=active_syms,
    )
    assert "TEST.T" in result
    r = result["TEST.T"]
    assert 0.0 <= r.score <= 100.0
    assert r.confidence in {"HIGH", "MEDIUM", "LOW"}


# ─────────────────────────────────────────────────────────────────────
#  entry_timing_ab_test imports
# ─────────────────────────────────────────────────────────────────────

def test_ab_test_module_importable():
    """entry_timing_ab_test.py がインポート可能であること。"""
    import src.backtest.entry_timing_ab_test as abt
    assert hasattr(abt, "MODES")
    assert set(abt.MODES.keys()) == {"A", "B", "C", "D"}


def test_ab_test_mode_a_is_baseline():
    """Mode A は et_enabled=False であること (baseline)。"""
    from src.backtest.entry_timing_ab_test import MODES
    assert MODES["A"]["et_enabled"]      is False
    assert MODES["A"]["et_block_low"]    is False
    assert MODES["A"]["et_boost_weight"] == 0.0


def test_ab_test_mode_c_block_only():
    """Mode C は block_low=True, boost=0.0 であること。"""
    from src.backtest.entry_timing_ab_test import MODES
    assert MODES["C"]["et_enabled"]      is True
    assert MODES["C"]["et_block_low"]    is True
    assert MODES["C"]["et_boost_weight"] == 0.0
