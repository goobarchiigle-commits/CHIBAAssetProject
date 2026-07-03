"""
backtest/universe_builder.py
TOPIX100相当ユニバース定義 + データ取得

【銘柄数】
  74銘柄 / 17セクター（2024年時点のTOPIX100主要構成銘柄）

【生存者バイアスについて】
  現時点の構成銘柄を使っているため、期間中に降格・上場廃止した銘柄が
  除外されており、結果が若干楽観的になる可能性がある。
  TOPIX500などと比べればバイアスは小さいが、留意すること。

【使い方】
  from backtest.universe_builder import TOPIX100_TICKERS, download_universe
  universe_raw = download_universe(TOPIX100_TICKERS, start="2018-01-01", end="2024-12-31")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
from src.utils.memory import log_optimization, optimize_backtest_df

# ------------------------------------------------------------------ #
# スナップショット設定
# ------------------------------------------------------------------ #
from src.paths import (
    ALLOW_YFINANCE_NETWORK,
    BACKTEST_DATASET_DIR as SNAPSHOT_BASE,
    DEFAULT_DATA_VERSION,
)

DATA_VERSION: str = os.environ.get("DATA_VERSION", "").strip() or DEFAULT_DATA_VERSION


# ------------------------------------------------------------------ #
# TOPIX100 相当銘柄リスト（セクター付き）
# ------------------------------------------------------------------ #
TOPIX100_TICKERS: dict[str, str] = {
    "6758.T": "電機", "6861.T": "電機精密", "8035.T": "電機精密", "6954.T": "電機精密",
    "6645.T": "電機精密", "6702.T": "電機", "6501.T": "電機", "6503.T": "電機",
    "6752.T": "電機", "6762.T": "電機精密", "6971.T": "電機精密", "6723.T": "電機精密",
    "7741.T": "電機精密", "6920.T": "電機精密", "6902.T": "電機", "7751.T": "電機精密",
    "7203.T": "輸送機器", "7267.T": "輸送機器", "7201.T": "輸送機器", "7270.T": "輸送機器",
    "9984.T": "情報通信", "9432.T": "情報通信", "9433.T": "情報通信", "9613.T": "情報通信",
    "4307.T": "情報通信", "4704.T": "情報通信",
    "8306.T": "銀行", "8316.T": "銀行", "8411.T": "銀行", "8309.T": "銀行", "7182.T": "銀行",
    "8766.T": "保険", "8725.T": "保険",
    "8058.T": "商社", "8031.T": "商社", "8053.T": "商社", "8002.T": "商社", "8001.T": "商社",
    "4502.T": "医薬品", "4519.T": "医薬品", "4568.T": "医薬品", "4523.T": "医薬品", "4578.T": "医薬品", "4543.T": "医薬品",
    "4063.T": "化学", "4452.T": "化学", "3407.T": "化学", "4021.T": "化学", "4183.T": "化学", "4911.T": "化学", "4901.T": "化学",
    "2802.T": "食品", "2914.T": "食品", "2503.T": "食品", "2502.T": "食品",
    "6367.T": "機械", "6301.T": "機械", "6326.T": "機械", "7011.T": "機械",
    "8801.T": "不動産", "8802.T": "不動産", "8830.T": "不動産",
    "3382.T": "小売", "9983.T": "小売", "8267.T": "小売",
    "4661.T": "レジャー", "6098.T": "サービス", "7974.T": "ゲーム",
    "5401.T": "鉄鋼", "5411.T": "鉄鋼",
    "5020.T": "石油", "9531.T": "ガス",
    "9101.T": "海運", "9104.T": "海運",
    "9020.T": "陸運", "9021.T": "陸運", "9022.T": "陸運",
}


def _load_from_snapshot(sym: str, start: str, end: str, verbose: bool = False) -> pd.DataFrame | None:
    """
    スナップショットから銘柄データを読み込む。
    DATA_VERSION 未設定またはファイル不存在なら None を返す。
    """
    if not DATA_VERSION:
        return None
    path = SNAPSHOT_BASE / DATA_VERSION / f"{sym}.parquet"
    if not path.exists():
        return None
    _needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    import pyarrow.parquet as _pq
    schema_names = _pq.read_schema(path).names
    cols = [c for c in _needed if c in schema_names]
    df = pd.read_parquet(path, columns=cols)
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)
    df = df.loc[(df.index >= ts_start) & (df.index <= ts_end)]
    if "Adj Close" in df.columns and "Close" in df.columns:
        df = df.copy()
        df["Close"] = df["Adj Close"]
        df = df.drop(columns=["Adj Close"])
    optimized = optimize_backtest_df(df)
    log_optimization(df, optimized, label=f"{sym} snapshot", verbose=verbose)
    return optimized


def download_universe(
    tickers: dict[str, str],
    start: str,
    end: str,
    min_days: int = 500,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    指定ユニバースの株価データを一括取得する。
    """
    if DATA_VERSION and verbose:
        snap_dir = SNAPSHOT_BASE / DATA_VERSION
        meta_path = snap_dir / "_meta.json"
        snap_hash = "unknown"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                snap_hash = json.load(f).get("snapshot_hash", "unknown")
        print(f"  [SNAPSHOT] version={DATA_VERSION}  hash={snap_hash}")

    warnings.filterwarnings("ignore")
    result = {}
    failed = []
    too_short = []
    snap_hits = 0

    for sym, sector in tickers.items():
        try:
            df = _load_from_snapshot(sym, start, end, verbose=verbose)
            src = "SNAP"
            if df is not None:
                snap_hits += 1
            elif ALLOW_YFINANCE_NETWORK:
                df = yf.download(sym, start=start, end=end, progress=False)
                if df.empty:
                    failed.append(sym)
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(1, axis=1)
                optimized = optimize_backtest_df(df)
                log_optimization(df, optimized, label=f"{sym} yfinance", verbose=verbose)
                df = optimized
                src = "yf  "
            else:
                failed.append(sym)
                if verbose:
                    print(f"  ✗ {sym:<8} ローカルsnapshot不在・ネットワーク取得無効")
                continue

            if len(df) < min_days:
                too_short.append((sym, len(df)))
                continue

            result[sym] = {"df": df, "sector": sector}
            if verbose:
                print(f"  ✓ {sym:<8} ({sector:<8}) : {len(df):,} 日  [{src}]")

        except Exception as e:
            failed.append(sym)
            if verbose:
                print(f"  ✗ {sym:<8} 取得失敗: {e}")

    if verbose:
        print(f"\n取得成功: {len(result)} / {len(tickers)} 銘柄"
              f"  (snapshot={snap_hits} / yfinance={len(result)-snap_hits})", flush=True)
        if too_short:
            print(f"データ不足で除外: {[s for s, _ in too_short]}")
        if failed:
            print(f"取得失敗: {failed}")

    return result
