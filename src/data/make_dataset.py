"""
src/data/make_dataset.py
OHLCVキャッシュにフィーチャーカラムを追加する。

【目的】
  signal_bridge._generate_all_signals() が要求する以下のカラムを
  スナップショット/ネットワークから取得したOHLCVに付与し、
  cache/ohlcv/ へ保存する。

  必須カラム:
    rsr_252        - IBD式12ヶ月複合リターンのクロスセクショナルパーセンタイル (0-100)
    rsr_63         - 63日リターンのクロスセクショナルパーセンタイル (0-100)
    ma20           - 20日単純移動平均
    ma50           - 50日単純移動平均
    ma20_slope     - MA20の5日変化量（正=上昇トレンド）
    avg_turnover_20d - 20日平均売買代金（Close × Volume）
    avg_volume_20d   - 20日平均出来高

【使い方】
  python src/data/make_dataset.py              # 全銘柄
  python src/data/make_dataset.py --force      # スナップショット強制再取得
  python src/data/make_dataset.py --symbols 8035.T 6702.T

【出力】
  cache/ohlcv/{symbol}.parquet  （フィーチャーカラム付き）

【注意】
  ALLOW_YFINANCE_NETWORK=true の場合、yfinance から最新データを取得してから
  フィーチャーを付与する（スナップショットより新鮮なデータになる）。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# プロジェクトルートをパスに追加
_HERE = Path(__file__).resolve()
_SRC  = _HERE.parents[1]           # src/
_ROOT = _HERE.parents[2]           # C:/ai-trading
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))

from src.paths import (
    RSR_UNIVERSE_FILE,
    BACKTEST_DATASET_DIR,
    DEFAULT_DATA_VERSION,
    CACHE_DIR,
    ALLOW_YFINANCE_NETWORK,
)

# ── 定数 ──────────────────────────────────────────────────────────────
JST             = timezone(timedelta(hours=9))
OHLCV_CACHE_DIR = CACHE_DIR / "ohlcv"
LOOKBACK_DAYS   = 600          # yfinance取得期間
NAN_RATE_LIMIT  = 0.20         # 検証時のNaN許容率

REQUIRED_FEATURES = [
    "rsr_252", "rsr_63",
    "ma20", "ma50", "ma20_slope",
    "avg_turnover_20d", "avg_volume_20d",
]


# ── ユニバース読み込み ─────────────────────────────────────────────────
def load_universe(symbols: list[str] | None = None) -> dict[str, str]:
    """RSRユニバースCSVを読み込む。symbols 指定時はフィルタ。"""
    if not RSR_UNIVERSE_FILE.exists():
        print(f"  ERROR: ユニバースファイルが見つかりません: {RSR_UNIVERSE_FILE}")
        sys.exit(1)
    df = pd.read_csv(RSR_UNIVERSE_FILE)
    universe = {row["symbol"]: row.get("sector", "不明") for _, row in df.iterrows()}
    if symbols:
        universe = {s: universe.get(s, "不明") for s in symbols}
    return universe


# ── OHLCV読み込み ─────────────────────────────────────────────────────
def _load_from_snapshot(symbol: str, version: str) -> pd.DataFrame:
    """スナップショットからOHLCVを読み込む。"""
    if not version:
        return pd.DataFrame()
    path = BACKTEST_DATASET_DIR / version / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df = df.dropna(subset=["Close"])
        return df if len(df) >= 252 else pd.DataFrame()
    except Exception as e:
        print(f"  WARN {symbol}: スナップショット読み込みエラー: {e}")
        return pd.DataFrame()


def _load_from_cache(symbol: str) -> pd.DataFrame:
    """既存キャッシュを読み込む（フィーチャー付与前の生データ）。"""
    path = OHLCV_CACHE_DIR / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df = df.dropna(subset=["Close"])
        return df if len(df) >= 252 else pd.DataFrame()
    except Exception as e:
        print(f"  WARN {symbol}: キャッシュ読み込みエラー: {e}")
        return pd.DataFrame()


def _download_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """yfinanceから1銘柄をダウンロード。"""
    try:
        import yfinance as yf
        warnings.filterwarnings("ignore")
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df = df.dropna(subset=["Close"])
        return df if len(df) >= 252 else pd.DataFrame()
    except Exception as e:
        print(f"  WARN {symbol}: yfinance エラー: {e}")
        return pd.DataFrame()


def load_ohlcv(symbol: str, version: str, force: bool = False) -> pd.DataFrame:
    """
    優先順位:
      1. yfinance（ALLOW_YFINANCE_NETWORK=true かつ not force のとき）
      2. スナップショット
      3. 既存キャッシュ
    """
    end_date   = datetime.now(JST).strftime("%Y-%m-%d")
    start_date = (datetime.now(JST) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    if ALLOW_YFINANCE_NETWORK:
        df = _download_yfinance(symbol, start_date, end_date)
        if not df.empty:
            return df

    df = _load_from_snapshot(symbol, version)
    if not df.empty:
        return df

    return _load_from_cache(symbol)


# ── RSR計算 ───────────────────────────────────────────────────────────
def _calc_composite_return(prices: pd.Series) -> pd.Series:
    """IBD式加重12ヶ月複合リターン。先読みリーク防止: shift()で過去のみ参照。"""
    r1 = prices / prices.shift(63) - 1
    r2 = prices.shift(63)  / prices.shift(126) - 1
    r3 = prices.shift(126) / prices.shift(189) - 1
    r4 = prices.shift(189) / prices.shift(252) - 1
    return 0.4 * r1 + 0.2 * r2 + 0.2 * r3 + 0.2 * r4


def calc_cross_sectional_rsr(
    prices_dict: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    全銘柄のクロスセクショナルRSRを計算する。

    Returns:
        (rsr_252_df, rsr_63_df) — 各列が銘柄、値は0〜100のスコア
    """
    # rsr_252: IBD式複合リターンのランク
    comp_dict = {sym: _calc_composite_return(p) for sym, p in prices_dict.items()}
    comp_df   = pd.DataFrame(comp_dict)
    rsr_252   = (comp_df.rank(axis=1, pct=True) * 100).clip(0, 100)

    # rsr_63: 63日単純リターンのランク
    r63_dict  = {sym: (p / p.shift(63) - 1) for sym, p in prices_dict.items()}
    r63_df    = pd.DataFrame(r63_dict)
    rsr_63    = (r63_df.rank(axis=1, pct=True) * 100).clip(0, 100)

    return rsr_252, rsr_63


# ── フィーチャー付与 ───────────────────────────────────────────────────
def enrich_df(
    df: pd.DataFrame,
    rsr_252_series: pd.Series,
    rsr_63_series:  pd.Series,
) -> pd.DataFrame:
    """OHLCVにフィーチャーカラムを付与して返す（元dfは変更しない）。"""
    df = df.copy()
    close  = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    df["ma20"]             = close.rolling(20, min_periods=1).mean()
    df["ma50"]             = close.rolling(50, min_periods=1).mean()
    df["ma20_slope"]       = df["ma20"].diff(5)
    df["avg_turnover_20d"] = (close * volume).rolling(20, min_periods=1).mean()
    df["avg_volume_20d"]   = volume.rolling(20, min_periods=1).mean()

    df["rsr_252"] = rsr_252_series.reindex(df.index).ffill().bfill().fillna(0.0)
    df["rsr_63"]  = rsr_63_series.reindex(df.index).ffill().bfill().fillna(0.0)

    return df


# ── 検証 ──────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame, symbol: str) -> tuple[bool, list[str]]:
    """必須カラムの存在とNaN率を検証する。"""
    issues = []
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            issues.append(f"{col}:missing")
            continue
        nan_rate = float(df[col].isna().mean())
        if nan_rate > NAN_RATE_LIMIT:
            issues.append(f"{col}:nan={nan_rate:.1%}")
    return (len(issues) == 0), issues


# ── メイン ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="OHLCVフィーチャーエンリッチメント")
    parser.add_argument("--force",   action="store_true", help="yfinance強制再取得")
    parser.add_argument("--symbols", nargs="*",           help="対象銘柄（省略=全銘柄）")
    args = parser.parse_args()

    version = DEFAULT_DATA_VERSION

    print("=" * 60)
    print("  make_dataset.py: OHLCVフィーチャーエンリッチメント")
    print(f"  スナップショット  : {version or '(なし)'}")
    print(f"  yfinanceネットワーク: {ALLOW_YFINANCE_NETWORK}")
    print(f"  出力先           : {OHLCV_CACHE_DIR}")
    print("=" * 60)

    # ── 1. ユニバース読み込み ─────────────────────────────────────
    universe = load_universe(args.symbols)
    print(f"\n  対象銘柄: {len(universe)} 銘柄\n")

    # ── 2. 全銘柄OHLCV取得 ───────────────────────────────────────
    raw_data: dict[str, pd.DataFrame] = {}
    for sym in universe:
        df = load_ohlcv(sym, version, force=args.force)
        if df.empty:
            print(f"  SKIP {sym:<10} データなし")
        else:
            raw_data[sym] = df
            print(f"  LOAD {sym:<10} rows={len(df):,}  last={df.index[-1].date()}")

    print(f"\n  読み込み成功: {len(raw_data)} / {len(universe)} 銘柄\n")

    if len(raw_data) < 5:
        print("  ERROR: データ不足（5銘柄未満）。")
        print("  build_dataset_snapshot.py を先に実行するか、")
        print("  ALLOW_YFINANCE_NETWORK=true を .env に設定してください。")
        sys.exit(1)

    # ── 3. クロスセクショナルRSR計算 ─────────────────────────────
    print("  RSR計算中（クロスセクショナル）...")
    prices_dict = {sym: df["Close"] for sym, df in raw_data.items()}
    try:
        rsr_252_df, rsr_63_df = calc_cross_sectional_rsr(prices_dict)
        print(f"  RSR計算完了: {len(rsr_252_df.columns)} 銘柄\n")
    except Exception as e:
        print(f"  ERROR: RSR計算失敗: {e}")
        sys.exit(1)

    # ── 4. フィーチャー付与 → 保存 ───────────────────────────────
    OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ok_list, ng_list = [], []

    for sym, df in raw_data.items():
        try:
            r252 = rsr_252_df[sym] if sym in rsr_252_df.columns else pd.Series(50.0, index=df.index)
            r63  = rsr_63_df[sym]  if sym in rsr_63_df.columns  else pd.Series(50.0, index=df.index)

            enriched = enrich_df(df, r252, r63)
            valid, issues = validate(enriched, sym)

            out_path = OHLCV_CACHE_DIR / f"{sym}.parquet"
            enriched.to_parquet(out_path)

            last = enriched.iloc[-1]
            status = "OK" if valid else "WARN"
            print(
                f"  {status} {sym:<10}"
                f"  rsr252={last['rsr_252']:5.1f}"
                f"  rsr63={last['rsr_63']:5.1f}"
                f"  ma20={last['ma20']:8.0f}"
                f"  slope={last['ma20_slope']:+8.0f}"
                f"  turnover={last['avg_turnover_20d']/1e8:6.1f}億"
                + (f"  ISSUES={issues}" if issues else "")
            )

            if valid:
                ok_list.append(sym)
            else:
                ng_list.append((sym, issues))

        except Exception as e:
            print(f"  NG  {sym:<10}  error={e}")
            ng_list.append((sym, [str(e)]))

    # ── 5. サマリー ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  完了: OK={len(ok_list)}  NG={len(ng_list)}  合計={len(raw_data)}")

    if ng_list:
        print(f"\n  問題銘柄:")
        for sym, issues in ng_list:
            print(f"    {sym}: {issues}")
        sys.exit(1)
    else:
        print("\n  全銘柄フィーチャー付与OK。run_morning_signal.py を実行できます。")


if __name__ == "__main__":
    main()
