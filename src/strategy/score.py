"""
score.py — シグナルスコア計算・ランキング

formula:
    score = edge - λ_risk * risk - λ_cost * cost
    edge  = z * 0.6 + regime_ord * 0.3 + trend * 0.1

必須列: z
任意列: z_regime, trend, volatility, slippage
  → 列がなければ定数フォールバック（後述）。列が増えるほど精度向上。

────────────────────────────────────────────────────────────
CAGR 向上のための確認指標（優先順）
────────────────────────────────────────────────────────────
1. IC_mean (Spearman score vs realized ret) > 0.03
      → evaluate_signal_quality.py Section 3 で測定
      → 0.03 未満 = score に順位情報なし → weights 再調整 or 特徴量追加

2. Q4_hit_rate - Q1_hit_rate > 0.10
      → Section 4 バケット分析
      → 上位25%が下位25%を 10%pt 以上上回ること

3. precision@positive (score>0 時のヒット率) > 0.55
      → Section 3 fallback の「サイド方向精度」で近似確認
      → 低ければ score=0 閾値を上げる or λ_risk を強める

4. 偏回帰係数の方向確認
      → OLS(ret ~ z + regime_ord + trend) の係数が
        _W_Z=0.6 / _W_REGIME=0.3 / _W_TREND=0.1 と同符号であること
      → 同符号でなければ weight を実測値に合わせて更新

5. 特徴量カバレッジ
      → z        : 必須。なければ score=0 (warning 出力)
      → z_regime : 現データは "up"/"range" のみ。
                   "extreme" が出現すると +0.6 の最大寄与。
      → volatility: 列がなければ全銘柄 risk=1.0 で差異ゼロ
                    → latest_signals.csv に ATR 等を追加で有効化
      → trend     : 同上。中期モメンタム等を追加で有効化
────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# ── z_regime → ordinal (higher = stronger expected trend) ────────────────────
# "extreme" は現在データに未出現だが出現時に最大寄与するよう設計
_REGIME_ORD: dict[str, float] = {
    "extreme":  2.0,
    "up":       1.0,
    "range":    0.0,
    "down":    -0.5,
}

# ── Score formula weights ─────────────────────────────────────────────────────
_W_Z        = 0.6   # z の寄与
_W_REGIME   = 0.3   # regime ordinal の寄与
_W_TREND    = 0.1   # trend の寄与
_LAMBDA_RISK = 0.5  # risk ペナルティ係数
_LAMBDA_COST = 0.2  # cost ペナルティ係数


def enrich_with_z(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    date / symbol / open 列から z と z_regime を生成して返す。

    Parameters
    ----------
    df     : date, symbol, open 列を含む DataFrame（複数日・複数銘柄）
    window : open リターンの計算期間。1=1期間、5=5日、20=20日

    Algorithm
    ---------
    1. per-symbol open return: pct_change(window) grouped by symbol
    2. cross-sectional z-score at each date:
           z = (ret_i - mean(ret)) / std(ret)  across all symbols on that date
       → CS z-score で銘柄間の相対強度を標準化
    3. regime classification:
           |z| > 1.5  かつ z > 0 → "extreme"  (+2.0)
           z > 0.0               → "up"       (+1.0)
           z < -1.5              → "down"      (-0.5)
           z < 0.0               → "range"     (0.0)

    Notes
    -----
    - 同一日の銘柄数 < 3 は z=0 (CS z-score が不安定なため)
    - ret が NaN の行 (先頭 window 行) は z=0, z_regime="range"
    """
    required = {"date", "symbol", "open"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"enrich_with_z: 必須列なし: {missing}")

    df = df.copy().sort_values(["symbol", "date"])

    # per-symbol open return
    df["ret"] = df.groupby("symbol")["open"].pct_change(window)

    # cross-sectional z-score
    def _cs_zscore(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) < 3:
            return pd.Series(0.0, index=series.index)
        std = valid.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - valid.mean()) / std

    df["z"] = (
        df.groupby("date")["ret"]
        .transform(_cs_zscore)
        .fillna(0.0)
    )

    # regime classification (conditions evaluated in order — first match wins)
    z = df["z"]
    df["z_regime"] = np.select(
        [z > 1.5,    z > 0.0, z < -1.5,  z < 0.0],
        ["extreme", "up",    "down",    "range"],
        default="range",
    )

    return df


def compute_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    score 列を追加して返す（元 df を変更しない）。

    Parameters
    ----------
    df : DataFrame。必須列: z。任意列: z_regime, trend, volatility, slippage

    Returns
    -------
    DataFrame with columns: edge, risk, cost, score (追加)

    Notes
    -----
    - z 列がない場合 score=0.0 (ランキング無効、既存ロジックにフォールバック)
    - volatility 列がない場合 risk=1.0 (全銘柄同値 → cross-sectional 差異ゼロ)
    - slippage 列がない場合 cost=0.0
    """
    df = df.copy()

    if "z" not in df.columns:
        warnings.warn(
            "compute_score: 'z' 列が存在しないため score=0.0 (ランキング無効)。"
            "latest_signals.csv に z 列を追加してください。",
            stacklevel=2,
        )
        df["score"] = 0.0
        return df

    # regime ordinal — dict.map で安全に変換、未知値は 0.0
    if "z_regime" in df.columns:
        regime_ord = df["z_regime"].map(_REGIME_ORD).fillna(0.0)
    else:
        regime_ord = pd.Series(0.0, index=df.index)

    # optional columns (scalar fallback → cross-sectional 差異なし)
    trend      = df["trend"]      if "trend"      in df.columns else pd.Series(0.0, index=df.index)
    volatility = df["volatility"] if "volatility" in df.columns else pd.Series(1.0, index=df.index)
    slippage   = df["slippage"]   if "slippage"   in df.columns else pd.Series(0.0, index=df.index)

    df["edge"]  = df["z"] * _W_Z + regime_ord * _W_REGIME + trend * _W_TREND
    df["risk"]  = volatility
    df["cost"]  = slippage
    df["score"] = df["edge"] - _LAMBDA_RISK * df["risk"] - _LAMBDA_COST * df["cost"]

    return df


def select_top_signals(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    score を計算し、score > 0 の上位 top_k 件を返す。

    score > 0 の行がなければ元 df をそのまま返す（既存ロジックへのフォールバック）。
    """
    df = compute_score(df)
    positive = df[df["score"] > 0]
    if positive.empty:
        return df
    return positive.nlargest(top_k, "score").reset_index(drop=True)
