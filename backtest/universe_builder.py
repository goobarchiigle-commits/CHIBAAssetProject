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

# ------------------------------------------------------------------ #
# スナップショット設定
# ------------------------------------------------------------------ #
DATA_VERSION: str  = os.environ.get("DATA_VERSION", "")  # 空文字=スナップショット不使用
from paths import BACKTEST_DATASET_DIR as SNAPSHOT_BASE   # 絶対パス保証


# ------------------------------------------------------------------ #
# TOPIX100 相当銘柄リスト（セクター付き）
# ------------------------------------------------------------------ #
TOPIX100_TICKERS: dict[str, str] = {
    # ---- 電機・精密 ----
    "6758.T": "電機",        # ソニーグループ
    "6861.T": "電機精密",    # キーエンス
    "8035.T": "電機精密",    # 東京エレクトロン
    "6954.T": "電機精密",    # ファナック
    "6645.T": "電機精密",    # オムロン
    "6702.T": "電機",        # 富士通
    "6501.T": "電機",        # 日立製作所
    "6503.T": "電機",        # 三菱電機
    "6752.T": "電機",        # パナソニックHD
    "6762.T": "電機精密",    # TDK
    "6971.T": "電機精密",    # 京セラ
    "6723.T": "電機精密",    # ルネサスエレクトロニクス
    "7741.T": "電機精密",    # HOYA
    "6920.T": "電機精密",    # レーザーテック
    "6902.T": "電機",        # デンソー
    "7751.T": "電機精密",    # キヤノン
    # ---- 輸送機器 ----
    "7203.T": "輸送機器",    # トヨタ自動車
    "7267.T": "輸送機器",    # 本田技研工業
    "7201.T": "輸送機器",    # 日産自動車
    "7270.T": "輸送機器",    # SUBARU
    # ---- 情報通信 ----
    "9984.T": "情報通信",    # ソフトバンクグループ
    "9432.T": "情報通信",    # NTT
    "9433.T": "情報通信",    # KDDI
    "9613.T": "情報通信",    # NTTデータグループ
    "4307.T": "情報通信",    # 野村総合研究所
    "4704.T": "情報通信",    # トレンドマイクロ
    # ---- 銀行 ----
    "8306.T": "銀行",        # 三菱UFJフィナンシャル
    "8316.T": "銀行",        # 三井住友フィナンシャルグループ
    "8411.T": "銀行",        # みずほフィナンシャルグループ
    "8309.T": "銀行",        # 三井住友トラストHD
    "7182.T": "銀行",        # ゆうちょ銀行
    # ---- 保険 ----
    "8766.T": "保険",        # 東京海上HD
    "8725.T": "保険",        # MS&ADインシュアランスグループ
    # ---- 商社 ----
    "8058.T": "商社",        # 三菱商事
    "8031.T": "商社",        # 三井物産
    "8053.T": "商社",        # 住友商事
    "8002.T": "商社",        # 丸紅
    "8001.T": "商社",        # 伊藤忠商事
    # ---- 医薬品 ----
    "4502.T": "医薬品",      # 武田薬品工業
    "4519.T": "医薬品",      # 中外製薬
    "4568.T": "医薬品",      # 第一三共
    "4523.T": "医薬品",      # エーザイ
    "4578.T": "医薬品",      # 大塚HD
    "4543.T": "医薬品",      # テルモ
    # ---- 化学 ----
    "4063.T": "化学",        # 信越化学工業
    "4452.T": "化学",        # 花王
    "3407.T": "化学",        # 旭化成
    "4021.T": "化学",        # 日産化学
    "4183.T": "化学",        # 三井化学
    "4911.T": "化学",        # 資生堂
    "4901.T": "化学",        # 富士フイルムHD
    # ---- 食品・飲料 ----
    "2802.T": "食品",        # 味の素
    "2914.T": "食品",        # JT
    "2503.T": "食品",        # キリンHD
    "2502.T": "食品",        # アサヒグループHD
    # ---- 機械 ----
    "6367.T": "機械",        # ダイキン工業
    "6301.T": "機械",        # 小松製作所
    "6326.T": "機械",        # クボタ
    "7011.T": "機械",        # 三菱重工業
    # ---- 不動産 ----
    "8801.T": "不動産",      # 三井不動産
    "8802.T": "不動産",      # 三菱地所
    "8830.T": "不動産",      # 住友不動産
    # ---- 小売 ----
    "3382.T": "小売",        # セブン&アイHD
    "9983.T": "小売",        # ファーストリテイリング
    "8267.T": "小売",        # イオン
    # ---- レジャー・サービス ----
    "4661.T": "レジャー",    # オリエンタルランド
    "6098.T": "サービス",    # リクルートHD
    "7974.T": "ゲーム",      # 任天堂
    # ---- 鉄鋼 ----
    "5401.T": "鉄鋼",        # 日本製鉄
    "5411.T": "鉄鋼",        # JFEホールディングス
    # ---- 石油・ガス ----
    "5020.T": "石油",        # ENEOSホールディングス
    "9531.T": "ガス",        # 東京ガス
    # ---- 海運 ----
    "9101.T": "海運",        # 日本郵船
    "9104.T": "海運",        # 商船三井
    # ---- 陸運・インフラ ----
    "9020.T": "陸運",        # JR東日本
    "9021.T": "陸運",        # JR西日本
    "9022.T": "陸運",        # JR東海
}


# ------------------------------------------------------------------ #
# データ取得関数
# ------------------------------------------------------------------ #
def _load_from_snapshot(sym: str, start: str, end: str) -> pd.DataFrame | None:
    """
    スナップショットから銘柄データを読み込む。
    DATA_VERSION 未設定またはファイル不存在なら None を返す。
    """
    if not DATA_VERSION:
        return None
    path = SNAPSHOT_BASE / DATA_VERSION / f"{sym}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # 期間フィルタ（スナップショットは広い期間で保存されているため）
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)
    df = df.loc[(df.index >= ts_start) & (df.index <= ts_end)]
    # auto_adjust=False のスナップショットは "Adj Close" を Close として使う
    if "Adj Close" in df.columns and "Close" in df.columns:
        df = df.copy()
        df["Close"] = df["Adj Close"]   # RSR・シグナルは調整済み終値を使用
    return df


def download_universe(
    tickers: dict[str, str],
    start:   str,
    end:     str,
    min_days: int = 500,
    verbose:  bool = True,
) -> dict[str, dict]:
    """
    指定ユニバースの株価データを一括取得する。

    DATA_VERSION 環境変数が設定されていれば data/backtest_dataset/{DATA_VERSION}/
    のスナップショットを使用する（再現性保証）。未設定なら yfinance から取得。

    Args:
        tickers:  {symbol: sector} の辞書
        start:    取得開始日（YYYY-MM-DD）
        end:      取得終了日（YYYY-MM-DD）
        min_days: 最低取引日数（これを下回るデータは除外）
        verbose:  進捗を表示するか

    Returns:
        {symbol: {"df": OHLCV DataFrame, "sector": str}} の辞書
        取得失敗・データ不足の銘柄は除外される
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
    result     = {}
    failed     = []
    too_short  = []
    snap_hits  = 0

    for sym, sector in tickers.items():
        try:
            # ① スナップショット優先
            df = _load_from_snapshot(sym, start, end)
            if df is not None:
                snap_hits += 1
            else:
                # ② フォールバック: yfinance
                df = yf.download(sym, start=start, end=end, progress=False)
                if df.empty:
                    failed.append(sym)
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(1, axis=1)

            if len(df) < min_days:
                too_short.append((sym, len(df)))
                continue

            result[sym] = {"df": df, "sector": sector}
            if verbose:
                src = "SNAP" if snap_hits > len(result) - 1 else "yf  "
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
