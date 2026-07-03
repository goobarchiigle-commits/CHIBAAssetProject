"""
strategy/cluster.py
Rolling correlation ベースのクラスター cap 実装

CLUSTER_MAP_DEFAULT : セクターをマクロ特性でグルーピングした静的クラスターマップ
get_cluster         : 銘柄 → クラスター名を返す
calc_cluster_weight : 現ポートフォリオのクラスター別ウェイトを計算
rolling_cluster_update : 直近リターン相関から動的にクラスターを更新（research 用）

確定設定（2026-04-07）:
  cluster_cap=0.35 (Bull), bear_cluster_cap=0.25 (Bear)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


# ------------------------------------------------------------------ #
# デフォルトクラスターマップ（静的・RSR42 セクター対応）
# ------------------------------------------------------------------ #
CLUSTER_MAP_DEFAULT: Dict[str, List[str]] = {
    "cyclical_macro": ["卸売業", "機械", "鉄鋼", "銀行業", "保険業", "輸送用機器"],
    "defensive":      ["情報・通信業", "医薬品", "食料品", "電気・ガス業"],
    "growth_tech":    ["電気機器", "精密機器", "サービス業"],
    "real_asset":     ["建設業", "不動産業", "金属製品", "化学"],
}

# composite_alpha_bt の SECTOR_STRATEGY で使われる略称 → CLUSTER_MAP 正式名
# rsr_universe_42.csv のセクター列が略称の場合に対応するためのエイリアステーブル
_SECTOR_ALIAS: Dict[str, str] = {
    "商社":     "卸売業",
    "機械":     "機械",
    "鉄鋼":     "鉄鋼",
    "銀行":     "銀行業",
    "保険":     "保険業",
    "輸送機器": "輸送用機器",
    "情報・通信業": "情報・通信業",
    "情報通信":     "情報・通信業",   # RSR42 CSV の略称
    "医薬品":       "医薬品",
    "食品":         "食料品",
    "ガス":         "電気・ガス業",
    "電機":     "電気機器",
    "電機精密": "電気機器",
    "精密":     "精密機器",
    "サービス": "サービス業",
    "建設":     "建設業",
    "不動産":   "不動産業",
    "金属":     "金属製品",
    "化学":     "化学",
    "海運":     "卸売業",
    "レジャー": "サービス業",
    "小売":     "サービス業",
    "ゲーム":   "サービス業",
}


def _normalize_sector(sector: str) -> str:
    """略称を CLUSTER_MAP_DEFAULT のキーで使われる正式セクター名に変換する。"""
    return _SECTOR_ALIAS.get(sector, sector)


def get_cluster(
    symbol: str,
    symbol_sector_map: Dict[str, str],
    cluster_map: Dict[str, List[str]],
) -> str:
    """
    symbol が属するクラスター名を返す。
    略称・正式名どちらのセクター名にも対応。
    どのクラスターにも属さない場合は "other" を返す。
    """
    sector = symbol_sector_map.get(symbol, "")
    sector_norm = _normalize_sector(sector)
    for cluster_name, sectors in cluster_map.items():
        if sector in sectors or sector_norm in sectors:
            return cluster_name
    return "other"


def calc_cluster_weight(
    portfolio_positions: Dict[str, float],
    symbol_sector_map: Dict[str, str],
    cluster_map: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    現在のポートフォリオのクラスター別ウェイトを計算する。

    Parameters
    ----------
    portfolio_positions : {symbol: weight}  ─ 資本比率（0.0〜1.0）
    symbol_sector_map   : {symbol: sector}
    cluster_map         : {cluster_name: [sector, ...]}

    Returns
    -------
    {cluster_name: total_weight}  ─ "other" を含む
    """
    cluster_weights: Dict[str, float] = {c: 0.0 for c in cluster_map}
    cluster_weights["other"] = 0.0

    for symbol, weight in portfolio_positions.items():
        sector = symbol_sector_map.get(symbol, "")
        matched = False
        for cluster, sectors in cluster_map.items():
            if sector in sectors:
                cluster_weights[cluster] += weight
                matched = True
                break
        if not matched:
            cluster_weights["other"] += weight

    return cluster_weights


def rolling_cluster_update(
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
    window: int = 60,
    corr_threshold: float = 0.60,
) -> Dict[str, List[str]]:
    """
    直近 window 営業日の日次リターン相関からセクタークラスターを動的更新。
    相関 >= corr_threshold のセクターを同一クラスターに分類。
    Union-Find アルゴリズムを使用。

    Parameters
    ----------
    returns        : 銘柄別日次リターン DataFrame（columns=symbol）
    sector_map     : {symbol: sector}
    window         : ルックバック期間（営業日）
    corr_threshold : 同一クラスター判定の相関閾値

    Returns
    -------
    {f"cluster_{i}": [sector, ...]}  ─ 静的 CLUSTER_MAP_DEFAULT と互換の形式
    先読みバイアスなし（呼び出し側で shift/eval_dt を管理すること）
    """
    # セクター別平均リターンを計算
    sector_returns: Dict[str, list] = {}
    for symbol in returns.columns:
        sector = sector_map.get(symbol, "other")
        sector_returns.setdefault(sector, []).append(returns[symbol])

    sector_avg = {
        s: pd.concat(v, axis=1).mean(axis=1)
        for s, v in sector_returns.items()
        if v
    }

    if len(sector_avg) < 2:
        return CLUSTER_MAP_DEFAULT

    sector_df  = pd.DataFrame(sector_avg).tail(window)
    corr_matrix = sector_df.corr()

    # Union-Find でクラスター形成
    sectors = list(corr_matrix.columns)
    parent: Dict[str, str] = {s: s for s in sectors}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for i, s1 in enumerate(sectors):
        for s2 in sectors[i + 1:]:
            if abs(corr_matrix.loc[s1, s2]) >= corr_threshold:
                union(s1, s2)

    clusters: Dict[str, List[str]] = {}
    for s in sectors:
        root = find(s)
        clusters.setdefault(root, []).append(s)

    return {f"cluster_{i}": v for i, v in enumerate(clusters.values())}
