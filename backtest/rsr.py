"""
backtest/rsr.py
RSR（Relative Strength Rating）計算モジュール

フジコ投資法の中核指標:
  RSR      = ユニバース内での相対強度ランク（0〜100）
  RSRモメンタム = RSRの21日変化（正かつ上昇 → 買い、負かつ下降 → 売り）

算出方式（IBD式加重12ヶ月リターン）:
  直近3ヶ月(63日)リターン × 40%
  + 3〜6ヶ月前リターン    × 20%
  + 6〜9ヶ月前リターン    × 20%
  + 9〜12ヶ月前リターン   × 20%

先読みリーク防止（CLAUDE.md ルール1）:
  shift(1) を使って前日終値ベースでリターンを計算する。
  当日の終値は確定後のみ使用。
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
# 加重複合リターン（IBD式）
# ------------------------------------------------------------------ #
def calc_composite_return(prices: pd.Series) -> pd.Series:
    """
    IBD式の加重12ヶ月複合リターンを計算する。

    先読みリーク防止:
      prices[t] = t日の確定終値。shift()で過去のみ参照。

    Args:
        prices: Close価格の時系列（DatetimeIndex）

    Returns:
        加重複合リターンの時系列
    """
    r1 = prices / prices.shift(63)  - 1   # 直近3ヶ月
    r2 = prices.shift(63)  / prices.shift(126) - 1  # 3〜6ヶ月前
    r3 = prices.shift(126) / prices.shift(189) - 1  # 6〜9ヶ月前
    r4 = prices.shift(189) / prices.shift(252) - 1  # 9〜12ヶ月前
    return 0.4 * r1 + 0.2 * r2 + 0.2 * r3 + 0.2 * r4


# ------------------------------------------------------------------ #
# ユニバース内RSRランク計算（ポートフォリオバックテスト用）
# ------------------------------------------------------------------ #
def calc_universe_rsr(universe_prices: dict[str, pd.Series]) -> pd.DataFrame:
    """
    ユニバース内の全銘柄について、日次RSRを計算する。

    RSR = その日における複合リターンのユニバース内パーセンタイルランク × 100

    Args:
        universe_prices: {symbol: Close価格Series} の辞書

    Returns:
        各銘柄のRSRを列に持つDataFrame（0〜100スケール）
    """
    comp_dict = {
        sym: calc_composite_return(prices)
        for sym, prices in universe_prices.items()
    }
    comp_df = pd.DataFrame(comp_dict)

    # 各日付での横断的ランク（パーセンタイル → 0〜100）
    rsr_df = comp_df.rank(axis=1, pct=True) * 100
    return rsr_df.clip(0, 100)


# ------------------------------------------------------------------ #
# Clenow (2015) ボラティリティ調整モメンタムスコア
# ------------------------------------------------------------------ #
def calc_clenow_score(price: pd.Series, window: int = 90) -> float:
    """
    Clenow「Stocks on the Move」式スコアを計算する。

    Score = 年率換算指数回帰スロープ × R²

    特徴:
      - コツコツ上昇する銘柄（高R²）を優遇
      - 急騰・急落銘柄（大ギャップ）を除外
      - 下降トレンドは0を返す

    Args:
        price:  Close価格Series
        window: 回帰ウィンドウ（デフォルト90営業日）

    Returns:
        スコア（高いほど「スムーズな上昇トレンド」）
        除外条件に当てはまる場合は 0.0
    """
    if len(price) < window + 1:
        return 0.0

    p = price.iloc[-window:]

    # ギャップフィルター: 過去90日に15%超の単日変動があれば除外
    max_gap = float(p.pct_change().abs().max())
    if max_gap > 0.15:
        return 0.0

    # 指数回帰（対数価格に線形回帰 → 指数的トレンドを捉える）
    x = np.arange(len(p))
    y = np.log(p.values.astype(float))
    n = len(x)
    sx = x.sum(); sy = y.sum()
    sxx = (x * x).sum(); sxy = (x * y).sum()
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)

    # R²（決定係数）
    y_mean = sy / n
    y_hat  = slope * x + (sy - slope * sx) / n
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r_sq   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 年率換算スロープ（%）
    annualized_slope = (np.exp(slope * 252) - 1) * 100

    # 下降トレンドは0
    if annualized_slope <= 0:
        return 0.0

    return float(annualized_slope * r_sq)


def calc_universe_clenow(
    universe_prices: dict[str, pd.Series],
    window: int = 90,
) -> pd.DataFrame:
    """
    ユニバース全銘柄のClenowスコアをローリング計算する。

    Args:
        universe_prices: {symbol: Close価格Series}
        window:          回帰ウィンドウ（デフォルト90日）

    Returns:
        各銘柄のClenowスコアを列に持つDataFrame
    """
    result = {}
    for sym, prices in universe_prices.items():
        scores = []
        idx    = []
        for i in range(window, len(prices)):
            sub   = prices.iloc[:i + 1]
            score = calc_clenow_score(sub, window=window)
            scores.append(score)
            idx.append(prices.index[i])
        result[sym] = pd.Series(scores, index=idx)
    return pd.DataFrame(result).fillna(0.0)


# ------------------------------------------------------------------ #
# FIP (Frog in the Pan) フィルター — Quantitative Momentum (Gray/Vogel)
# ------------------------------------------------------------------ #
def calc_fip_ratio(price: pd.Series, window: int = 252) -> float:
    """
    FIP（Frog in the Pan）比率を計算する。

    Gray & Vogel「Quantitative Momentum」の知見:
      「コツコツと小さな上昇を積み重ねた銘柄（高FIP）は
       急騰銘柄（低FIP）より高いリスク調整後リターンを示す」

    FIP比率 = 過去window日間の正リターン日数 / 全取引日数

    判定目安:
      FIP > 0.52  → "コツコツ型" = 優良モメンタム銘柄
      FIP < 0.48  → "急騰型"     = 反転リスク高

    Args:
        price:  Close価格Series
        window: 計算ウィンドウ（デフォルト252営業日=12ヶ月）

    Returns:
        FIP比率（0〜1）。データ不足の場合は 0.5
    """
    if len(price) < window + 1:
        return 0.5
    returns = price.iloc[-window:].pct_change().dropna()
    if len(returns) == 0:
        return 0.5
    return float((returns > 0).sum() / len(returns))


def calc_universe_fip(
    universe_prices: dict[str, pd.Series],
    window: int = 252,
) -> pd.DataFrame:
    """
    ユニバース全銘柄のFIP比率をローリング計算する。

    Args:
        universe_prices: {symbol: Close価格Series}
        window:          計算ウィンドウ（デフォルト252日=12ヶ月）

    Returns:
        各銘柄のFIP比率を列に持つDataFrame（0〜1スケール）
    """
    result = {}
    for sym, prices in universe_prices.items():
        fip_list = []
        idx      = []
        for i in range(window, len(prices)):
            sub = prices.iloc[:i + 1]
            fip_list.append(calc_fip_ratio(sub, window=window))
            idx.append(prices.index[i])
        result[sym] = pd.Series(fip_list, index=idx)
    return pd.DataFrame(result).fillna(0.5)


# ------------------------------------------------------------------ #
# 単一銘柄RSR（ベンチマーク比較・単体分析用）
# ------------------------------------------------------------------ #
def calc_rsr_vs_benchmark(
    stock_prices:     pd.Series,
    benchmark_prices: pd.Series,
) -> pd.Series:
    """
    単一銘柄とベンチマーク（TOPIX等）の比較でRSRを計算する。

    スケーリング:
      超過リターン = 0%  → RSR = 50（ベンチマーク並み）
      超過リターン = +20% → RSR ≈ 70（アウトパフォーム）
      超過リターン = +40% → RSR ≈ 90（強いアウトパフォーム）

    Args:
        stock_prices:     銘柄のClose価格Series
        benchmark_prices: ベンチマークのClose価格Series

    Returns:
        RSR（0〜100スケール）
    """
    stock_comp = calc_composite_return(stock_prices)
    bench_comp = calc_composite_return(benchmark_prices)
    relative   = stock_comp - bench_comp
    rsr        = 50 + (relative * 100).clip(-50, 50)
    return rsr.clip(0, 100)


# ------------------------------------------------------------------ #
# RSRモメンタム
# ------------------------------------------------------------------ #
def calc_rsr_momentum(rsr: pd.Series, period: int = 21) -> pd.Series:
    """
    RSRモメンタム = RSR[t] - RSR[t - period]

    フジコ投資法でのシグナル解釈:
      > 0 かつ 上昇中  → 買いシグナル（水色→濃いピンク）
      > 0 だが 下降中  → 利確検討（RSR < 70 の場合）
      < 0 かつ 下降中  → 売りシグナル（必ず手放す）

    Args:
        rsr:    RSR時系列
        period: モメンタム計算期間（デフォルト: 21日 ≈ 1ヶ月）

    Returns:
        RSRモメンタムの時系列
    """
    return rsr - rsr.shift(period)


# ------------------------------------------------------------------ #
# SEPA 8条件スクリーナー
# ------------------------------------------------------------------ #
def calc_sepa(
    df:  pd.DataFrame,
    rsr: pd.Series,
) -> pd.DataFrame:
    """
    SEPA 8条件を計算して各条件の真偽とスコア（0〜8）を返す。

    Args:
        df:  OHLCV DataFrame（DatetimeIndex）
        rsr: RSR時系列（calc_universe_rsr または calc_rsr_vs_benchmark の結果）

    Returns:
        各条件の真偽列（c1〜c8）とスコア列（sepa_score）を持つDataFrame
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    ma50  = close.rolling(50,  min_periods=50).mean()
    ma150 = close.rolling(150, min_periods=150).mean()
    ma200 = close.rolling(200, min_periods=200).mean()

    high52 = close.rolling(252, min_periods=126).max()
    low52  = close.rolling(252, min_periods=126).min()

    # 1. 株価 > MA150 かつ > MA200
    c1 = (close > ma150) & (close > ma200)
    # 2. MA150 > MA200
    c2 = ma150 > ma200
    # 3. MA200が21日前より上（1ヶ月以上の上昇トレンド）
    c3 = ma200 > ma200.shift(21)
    # 4. MA50が上昇中
    c4 = ma50 > ma50.shift(1)
    # 5. 株価 > MA50
    c5 = close > ma50
    # 6. 52週安値から+30%以上
    c6 = close >= low52 * 1.30
    # 7. 52週高値から-25%以内
    c7 = close >= high52 * 0.75
    # 8. RSR >= 70
    c8 = rsr >= 70.0

    sepa_score = (
        c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int) +
        c5.astype(int) + c6.astype(int) + c7.astype(int) + c8.astype(int)
    )

    return pd.DataFrame({
        "c1": c1, "c2": c2, "c3": c3, "c4": c4,
        "c5": c5, "c6": c6, "c7": c7, "c8": c8,
        "sepa_score": sepa_score,
        "is_ace":  sepa_score >= 6,
        "is_king": sepa_score == 8,
    }, index=df.index)


# ------------------------------------------------------------------ #
# STARCバンド（ATRベースのボラティリティバンド）
# ------------------------------------------------------------------ #
def calc_starc_band(
    df:       pd.DataFrame,
    ma_len:   int   = 20,
    atr_len:  int   = 10,
    atr_mult: float = 1.5,
) -> pd.DataFrame:
    """
    STARCバンド = MA ± ATR × 乗数

    上出しATR（Upper > Close）: 上昇ボラ大 → トレンド強
    逆バンド  （Close < Lower）: 下降ボラ大 → 売りシグナル

    Args:
        df:       OHLCV DataFrame
        ma_len:   移動平均の期間
        atr_len:  ATRの期間
        atr_mult: ATRの乗数

    Returns:
        upper / middle / lower を列に持つDataFrame
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    ma    = close.rolling(ma_len, min_periods=ma_len).mean()

    # ATR計算
    hl    = high - low
    hc    = (high - close.shift(1)).abs()
    lc    = (low  - close.shift(1)).abs()
    tr    = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr   = tr.rolling(atr_len, min_periods=atr_len).mean()

    return pd.DataFrame({
        "starc_upper":  ma + atr * atr_mult,
        "starc_middle": ma,
        "starc_lower":  ma - atr * atr_mult,
    }, index=df.index)
