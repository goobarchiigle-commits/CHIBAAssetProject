"""
portfolio/volatility_allocator.py
ATR ベースのボラティリティ・パリティ配分エンジン

【設計思想】
  均等配分ではなく「リスク（ATR）が均等になるように」ウェイトを決める。
  ATR が小さい（穏やか）銘柄 → 多く配分
  ATR が大きい（荒い）銘柄   → 少なく配分

  weight_i ∝ 1 / ATR_i
  （Clenow "Stocks on the Move" の ATR ポジションサイジングと同系統）

【配分計算の詳細】
  1. ATR20（20 日 True Range の単純平均）を各銘柄で計算
  2. ATR を株価で割って「ボラティリティ率」に正規化（price normalization）
  3. 逆数を取って sum=1 になるよう正規化
  4. max_single_weight でキャップ → 残余をほかの銘柄に再配分

【バックテストへの組み込み例】

    from portfolio.volatility_allocator import VolatilityAllocator

    alloc = VolatilityAllocator(
        capital          = 2_000_000,
        max_single_weight= 0.25,
        atr_period       = 20,
        fallback_equal   = True,   # ATR 計算不能時は均等配分
    )

    # 各銘柄の OHLCV DataFrame を渡す
    weights = alloc.compute_weights({"8035.T": df1, "9101.T": df2, "6857.T": df3})
    # → {"8035.T": 0.28, "9101.T": 0.40, "6857.T": 0.32}

    # 配分額を計算
    allocs = alloc.compute_allocations(weights)
    # → {"8035.T": 560_000, "9101.T": 800_000, "6857.T": 640_000}
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# ATR 計算
# ------------------------------------------------------------------ #
def calc_atr(df: pd.DataFrame, period: int = 20) -> float:
    """
    ATR（Average True Range）を計算して最新値を返す。

    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    ATR = TR の period 日単純平均

    Args:
        df:     OHLCV DataFrame（High, Low, Close 列が必要）
        period: ATR の計算期間（デフォルト 20 日）

    Returns:
        ATR 値（float）。計算不能の場合は NaN。
    """
    if len(df) < period + 1:
        return float("nan")

    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        return float("nan")

    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values

    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - prev_close),
            np.abs(low[1:]  - prev_close),
        ),
    )
    return float(np.mean(tr[-period:]))


# ------------------------------------------------------------------ #
# 配分エンジン
# ------------------------------------------------------------------ #
class VolatilityAllocator:
    """
    ATR 逆数によるボラティリティ・パリティ配分を計算するクラス。

    Args:
        capital:           ポートフォリオ総資本（円）
        max_single_weight: 1 銘柄への最大ウェイト上限（デフォルト 0.25）
        atr_period:        ATR の計算期間（デフォルト 20 日）
        fallback_equal:    ATR 計算不能銘柄を均等配分にフォールバックするか
    """

    def __init__(
        self,
        capital:            float = 2_000_000,
        max_single_weight:  float = 0.25,
        atr_period:         int   = 20,
        fallback_equal:     bool  = True,
    ):
        self.capital           = capital
        self.max_single_weight = max_single_weight
        self.atr_period        = atr_period
        self.fallback_equal    = fallback_equal

    def compute_weights(
        self,
        symbol_dfs: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """
        銘柄ごとの OHLCV DataFrame からボラティリティ・パリティ・ウェイトを計算する。

        Args:
            symbol_dfs: {symbol: OHLCV DataFrame}

        Returns:
            {symbol: weight}（合計は 1.0 以下、max_single_weight 適用済み）
        """
        if not symbol_dfs:
            return {}

        symbols = list(symbol_dfs.keys())
        n       = len(symbols)

        # ── ATR（価格正規化済み）を計算 ───────────────────────────
        norm_atrs: dict[str, float] = {}
        for sym, df in symbol_dfs.items():
            atr   = calc_atr(df, self.atr_period)
            price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
            if np.isnan(atr) or atr <= 0 or price <= 0:
                norm_atrs[sym] = float("nan")
            else:
                norm_atrs[sym] = atr / price   # 価格正規化 ATR（= ボラ率）

        # ── フォールバック処理 ────────────────────────────────────
        valid   = {s: v for s, v in norm_atrs.items() if not np.isnan(v)}
        invalid = {s for s in norm_atrs if np.isnan(norm_atrs[s])}

        if not valid:
            # 全銘柄が計算不能 → 均等配分
            return {s: 1.0 / n for s in symbols}

        # ── 逆数ウェイト（ATR が小さいほど多く配分）────────────────
        inv_vol  = {s: 1.0 / v for s, v in valid.items()}
        total_iv = sum(inv_vol.values())
        raw_w    = {s: iv / total_iv for s, iv in inv_vol.items()}

        # ── max_single_weight キャップ（繰り返し正規化）─────────────
        weights = self._apply_weight_cap(raw_w)

        # ── 計算不能銘柄に均等配分（フォールバック）─────────────────
        if invalid and self.fallback_equal:
            # 均等配分分を確保してほかを再スケール
            eq_w     = 1.0 / n
            for sym in invalid:
                weights[sym] = min(eq_w, self.max_single_weight)
            # 合計を 1.0 に再正規化
            total = sum(weights.values())
            if total > 0:
                weights = {s: w / total for s, w in weights.items()}

        return weights

    def _apply_weight_cap(self, raw_w: dict[str, float]) -> dict[str, float]:
        """
        max_single_weight を超えるウェイトをキャップして残余を再配分する。
        収束するまで繰り返す（最大 10 回）。
        """
        w = dict(raw_w)
        cap = self.max_single_weight

        for _ in range(10):
            capped   = {s: min(v, cap) for s, v in w.items()}
            excess   = sum(v - cap for v in w.values() if v > cap)
            n_free   = sum(1 for v in w.values() if v < cap)

            if excess == 0 or n_free == 0:
                break

            # 超過分を cap 未満の銘柄に均等再配分
            bonus = excess / n_free
            w = {
                s: min(cap, v + (bonus if v < cap else 0))
                for s, v in capped.items()
            }

        # 最終正規化
        total = sum(w.values())
        if total > 0:
            w = {s: v / total for s, v in w.items()}

        return w

    def compute_allocations(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """
        ウェイトを配分額（円）に変換する。

        Returns:
            {symbol: 配分額（円）}
        """
        return {s: w * self.capital for s, w in weights.items()}

    def compute_qty(
        self,
        weights:     dict[str, float],
        prices:      dict[str, float],
        lot_size:    int = 100,
    ) -> dict[str, int]:
        """
        ウェイト → 発注株数（単元株単位に切り捨て）を計算する。

        Args:
            weights: compute_weights() の戻り値
            prices:  {symbol: 株価}
            lot_size: 単元株数（東証標準 100）

        Returns:
            {symbol: 株数（0 の場合は価格制約で買えない）}
        """
        allocs = self.compute_allocations(weights)
        qty: dict[str, int] = {}
        for sym, alloc in allocs.items():
            price = prices.get(sym, 0.0)
            if price <= 0:
                qty[sym] = 0
                continue
            one_lot = price * lot_size
            qty[sym] = int(alloc // one_lot) * lot_size
        return qty

    # ------------------------------------------------------------------ #
    # 比較ユーティリティ
    # ------------------------------------------------------------------ #
    @staticmethod
    def compare_with_equal(
        vol_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        ボラティリティ・パリティウェイトと均等ウェイトの差分を返す。
        正 = ボラパリの方が多く配分。負 = 均等の方が多く配分。
        """
        n = len(vol_weights)
        eq = 1.0 / n if n > 0 else 0.0
        return {s: w - eq for s, w in vol_weights.items()}
