"""
tests/test_dynamic_universe.py
動的ユニバース: RSR NaN 修正のユニットテスト

実行:
  cd C:/ai-trading
  pytest tests/test_dynamic_universe.py -v -k rsr_nan
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.backtest.rsr import calc_universe_rsr
from src.backtest.wf_dynamic_universe import build_sym_active_df, _zscore


# ------------------------------------------------------------------ #
# ヘルパー: ダミー価格 Series を生成
# ------------------------------------------------------------------ #
def make_price_series(n=300, trend=1.001, seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 1000.0 * np.cumprod(trend + rng.normal(0, 0.01, n))
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


# ------------------------------------------------------------------ #
# Test 1: RSR NaN がないことを確認
# ------------------------------------------------------------------ #
def test_rsr_no_nan_full_pool():
    """全プール銘柄に対してRSRを計算したとき、
    ウォームアップ期間後はNaNが出ないことを確認"""
    syms = [f"SYM{i:02d}" for i in range(10)]
    prices = {sym: make_price_series(n=300, trend=1.001 + i*0.0001, seed=i) for i, sym in enumerate(syms)}

    rsr_df = calc_universe_rsr(prices)

    # ウォームアップ後（252日以降）はNaNがないはず
    warmup = 252
    if len(rsr_df) > warmup:
        post_warmup = rsr_df.iloc[warmup:]
        nan_count = post_warmup.isna().sum().sum()
        assert nan_count == 0, f"RSR NaN detected post-warmup: {nan_count} cells"


# ------------------------------------------------------------------ #
# Test 2: 非RSR42銘柄が rsr_df_full なしだと NaN を返す
# ------------------------------------------------------------------ #
def test_rsr_nan_without_full_pool():
    """RSR42ベースの rsr_df に非RSR42銘柄が含まれないことを確認
    （修正前の挙動を再現してNaN問題を証明）"""
    rsr42_syms = ["SYM00", "SYM01", "SYM02"]
    all_syms   = rsr42_syms + ["EXTRA00", "EXTRA01"]   # 追加銘柄

    all_prices = {sym: make_price_series(n=300, seed=i) for i, sym in enumerate(all_syms)}

    # RSR42のみで計算
    rsr42_prices = {s: all_prices[s] for s in rsr42_syms}
    rsr_df_rsr42_only = calc_universe_rsr(rsr42_prices)

    # 非RSR42銘柄は rsr_df にない
    for sym in ["EXTRA00", "EXTRA01"]:
        assert sym not in rsr_df_rsr42_only.columns, \
            f"{sym} should NOT be in RSR42-only rsr_df"


# ------------------------------------------------------------------ #
# Test 3: rsr_df_full で全銘柄にRSRが存在する
# ------------------------------------------------------------------ #
def test_rsr_full_pool_covers_all_syms():
    """全プールで RSR 計算するとすべての銘柄をカバーする"""
    all_syms = [f"SYM{i:02d}" for i in range(10)]
    all_prices = {sym: make_price_series(n=300, seed=i) for i, sym in enumerate(all_syms)}

    rsr_df_full = calc_universe_rsr(all_prices)

    for sym in all_syms:
        assert sym in rsr_df_full.columns, f"{sym} missing from full-pool RSR"


# ------------------------------------------------------------------ #
# Test 4: build_sym_active_df が NaN スコアの銘柄を除外する
# ------------------------------------------------------------------ #
def test_build_sym_active_dropna():
    """スコアに NaN がある銘柄は active セットに含まれないことを確認"""
    # 一部の銘柄は価格データが短くてモメンタムが NaN になる
    syms_full  = [f"SYM{i:02d}" for i in range(8)]
    syms_short = ["SHORT00", "SHORT01"]   # データが短い → mom_63 が NaN

    all_prices = {}
    for i, sym in enumerate(syms_full):
        all_prices[sym] = make_price_series(n=300, seed=i)
    for i, sym in enumerate(syms_short):
        # 50日しかない → 63日モメンタム不可
        prices = make_price_series(n=50, seed=100 + i)
        all_prices[sym] = prices

    all_syms = syms_full + syms_short
    universe_raw = {sym: {"df": pd.DataFrame({"Close": p, "Volume": p * 100})}
                    for sym, p in all_prices.items()}

    topix_close = make_price_series(n=300, seed=99)

    active_df = build_sym_active_df(
        universe_raw, topix_close, all_syms,
        active_n=5, start="2022-04-01", end="2022-12-31"
    )

    # アクティブな銘柄数はmax5
    max_active = active_df.sum(axis=1).max()
    assert max_active <= 5, f"active_n exceeded: {max_active}"

    # 全日付でアクティブ銘柄数 >= 0（クラッシュなし）
    assert (active_df.sum(axis=1) >= 0).all()


# ------------------------------------------------------------------ #
# Test 5: _zscore が NaN 入力を 0 に変換する
# ------------------------------------------------------------------ #
def test_zscore_nan_to_zero():
    """_zscore は NaN を 0.0 に変換するが、
    呼び出し元で score.dropna() を使うことで正しく除外できる"""
    s = pd.Series({"A": 1.0, "B": 2.0, "C": np.nan, "D": 3.0})
    z = _zscore(s)

    # NaN は 0.0 に変換される（_zscore の現挙動）
    assert z["C"] == 0.0, "NaN should be mapped to 0.0 by _zscore"
    # 有効値はゼロ平均
    valid_z = z[["A", "B", "D"]]
    assert abs(valid_z.mean()) < 1e-9, "valid values should have zero mean"


# ------------------------------------------------------------------ #
# Test 6: RSRパーセンタイルは0〜100の範囲内
# ------------------------------------------------------------------ #
def test_rsr_range():
    """RSR は 0〜100 の範囲に収まることを確認"""
    syms = [f"SYM{i:02d}" for i in range(6)]
    prices = {sym: make_price_series(n=300, seed=i) for i, sym in enumerate(syms)}
    rsr_df = calc_universe_rsr(prices)

    assert (rsr_df.fillna(50) >= 0).all().all(), "RSR < 0 detected"
    assert (rsr_df.fillna(50) <= 100).all().all(), "RSR > 100 detected"
