"""
src/execution/bucket.py
O(N) インクリメンタルバケット統計。純粋関数ベース。副作用なし。

設計:
  - BucketAccumulator: per-key (sum, sq_sum, count) を O(1)/更新で維持
  - stats(): mean, std, sharpe, shrink, score をすべて有限値で返す
  - median_opportunity(): track_values=True 時のみ利用可能
  - resolve_best_bucket(): count > weight 優先、低サンプル高重み選択を禁止
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

_SHRINK_K: float = 5.0   # James-Stein スタイル縮退定数
_SQRT_252: float = math.sqrt(252.0)


def _safe(v: Any, fallback: float = 0.0) -> float:
    """NaN / inf / None → fallback。"""
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


class BucketAccumulator:
    """
    O(N) インクリメンタルバケット統計。
    update() で逐次追加、stats() で集計結果を取得する。
    """

    __slots__ = ("_sum", "_sq_sum", "_count", "_values")

    def __init__(self) -> None:
        self._sum:    dict[str, float]        = {}
        self._sq_sum: dict[str, float]        = {}
        self._count:  dict[str, int]          = {}
        self._values: dict[str, list[float]]  = {}

    def update(self, key: str, value: float, track_values: bool = False) -> None:
        """
        key のバケットに value を追加する（O(1)）。
        NaN / inf は 0.0 として扱う（伝播させない）。
        track_values=True の場合は後続の median_opportunity() が利用可能。
        """
        v = _safe(value)
        self._sum[key]    = self._sum.get(key,    0.0) + v
        self._sq_sum[key] = self._sq_sum.get(key, 0.0) + v * v
        self._count[key]  = self._count.get(key,  0)   + 1
        if track_values:
            if key not in self._values:
                self._values[key] = []
            self._values[key].append(v)

    def stats(self, key: str) -> dict[str, float]:
        """
        key の集計統計を返す。全出力が有限値であることを保証する。

        Returns:
            mean   : 標本平均
            std    : 標本標準偏差（n=1 → 0.0）
            sharpe : 年次化シャープ比（std=0 → 0.0）
            shrink : James-Stein 縮退推定量（mean * n/(n+K)）
            score  : sharpe * shrink_factor（小サンプルでゼロに収束）
            count  : 観測数
        """
        n = self._count.get(key, 0)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "sharpe": 0.0,
                    "shrink": 0.0, "score": 0.0, "count": 0}

        s  = self._sum.get(key, 0.0)
        sq = self._sq_sum.get(key, 0.0)
        mean = s / n

        # 数値安定な分散: max(0, (sq_sum - n*mean^2) / (n-1))
        if n > 1:
            raw_var = (sq - n * mean * mean) / (n - 1)
            std = math.sqrt(max(0.0, raw_var)) if math.isfinite(raw_var) else 0.0
        else:
            std = 0.0

        sharpe = (mean / std * _SQRT_252) if std > 1e-10 else 0.0

        shrink_factor = n / (n + _SHRINK_K)
        shrink = mean * shrink_factor
        score  = sharpe * shrink_factor

        return {
            "mean":   _safe(mean),
            "std":    _safe(std),
            "sharpe": _safe(sharpe),
            "shrink": _safe(shrink),
            "score":  _safe(score),
            "count":  n,
        }

    def median_opportunity(self, key: str) -> float | None:
        """
        track_values=True で update() した場合のみ利用可能な中央値。
        未追跡 / 空 → None。
        """
        vals = self._values.get(key)
        if not vals:
            return None
        arr = np.array([_safe(v) for v in vals], dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return None
        return float(np.median(finite))

    def all_keys(self) -> list[str]:
        return list(self._count.keys())

    def all_stats(self) -> dict[str, dict[str, float]]:
        """全 key の stats() を一括返却する。"""
        return {k: self.stats(k) for k in self.all_keys()}


def build_bucket_stats(
    df: pd.DataFrame,
    key_col: str,
    value_col: str,
    track_median: bool = False,
) -> dict[str, dict[str, float]]:
    """
    DataFrame から O(N) インクリメンタルバケット統計を構築する。

    Args:
        df           : 入力 DataFrame
        key_col      : バケットキー列名
        value_col    : 集計値列名
        track_median : True の場合 median_opportunity() が利用可能になる

    Returns:
        {key: {mean, std, sharpe, shrink, score, count}}
        全値が有限。df が空または列欠損の場合は {} を返す。
    """
    if df.empty or key_col not in df.columns or value_col not in df.columns:
        return {}

    acc = BucketAccumulator()
    for row in df[[key_col, value_col]].itertuples(index=False):
        k = str(getattr(row, key_col))
        v = _safe(getattr(row, value_col))
        acc.update(k, v, track_values=track_median)

    return acc.all_stats()


def resolve_best_bucket(
    stats: dict[str, dict[str, float]],
    min_count: int = 5,
    weight_key: str = "score",
) -> str | None:
    """
    バケット選択: count > weight 優先で最良バケットを返す。
    低サンプル・高重みバケットの選択を禁止する。

    Priority:
      1. count >= min_count のバケットが存在 → weight_key で最高スコアを選択
      2. 全バケットが min_count 未満 → count 最大（データ量優先）を選択

    Args:
        stats      : build_bucket_stats() / BucketAccumulator.all_stats() の出力
        min_count  : 信頼可能と見なす最小観測数
        weight_key : スコア比較に使うキー（default: "score"）

    Returns:
        最良バケットキー。stats が空なら None。
    """
    if not stats:
        return None

    # count >= min_count で weight_key 最大を選択
    qualified = {
        k: v for k, v in stats.items()
        if v.get("count", 0) >= min_count
    }
    if qualified:
        return max(qualified, key=lambda k: _safe(qualified[k].get(weight_key, 0.0)))

    # フォールバック: count 最大（weight による低サンプル選択を回避）
    return max(stats, key=lambda k: stats[k].get("count", 0))
