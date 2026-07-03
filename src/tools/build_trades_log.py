"""
build_trades_log.py
複数銘柄OHLCVからMA5/MA20クロスでtrades_log.csvを生成・検証する
"""
from __future__ import annotations

import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import pandas as pd
import yfinance as yf

# ── パス設定 ─────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from src.paths import BASE_DIR

UNIVERSE_FILE    = BASE_DIR / "src" / "configs" / "rsr_universe_42.csv"
OHLCV_DIR        = BASE_DIR / "data" / "ohlcv"
OUT_PATH         = BASE_DIR / "data" / "trades_log.csv"
TRADE_REGIME_META_PATH = BASE_DIR / "logs" / "trade_regime_distribution.json"

START = "2018-01-01"
END   = "2024-12-31"
MIN_SYMBOLS = 10
MIN_TRADES  = 500

# trade_regime閾値 — 変更禁止
# trade_regime は「トレード損益ベースの分類」であり市場レジームではない
# up/down/range はトレードのret（entry→exit損益）に基づく区分
_TRADE_REGIME_UP_THRESHOLD   =  0.02   # ret >  2% → up
_TRADE_REGIME_DOWN_THRESHOLD = -0.02   # ret < -2% → down


# ── トレードレジーム分類 ───────────────────────────
def _label_trade_regime(ret: float) -> str:
    if ret > _TRADE_REGIME_UP_THRESHOLD:
        return "up"
    if ret < _TRADE_REGIME_DOWN_THRESHOLD:
        return "down"
    return "range"


def _check_trade_regime_distribution(df: pd.DataFrame) -> dict:
    counts    = df["trade_regime"].value_counts(normalize=True)
    ratio_up    = float(counts.get("up",    0.0))
    ratio_down  = float(counts.get("down",  0.0))
    ratio_range = float(counts.get("range", 0.0))
    imbalance   = bool((counts > 0.8).any())
    return {
        "trade_regime_ratio_up":       round(ratio_up,    4),
        "trade_regime_ratio_down":     round(ratio_down,  4),
        "trade_regime_ratio_range":    round(ratio_range, 4),
        "trade_regime_imbalance_flag": imbalance,
    }


# ── OHLCV取得 ────────────────────────────────────
def fetch_ohlcv(symbol: str, out_path: Path) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=START, end=END, interval="1d", auto_adjust=True)
    if df.empty:
        raise ValueError(f"{symbol}: データ取得失敗")

    df = df.reset_index()
    df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.dropna()
    df = df[df["volume"] > 0].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV未存在: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna()
    df = df[df["volume"] > 0].reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── バックテスト（MA5/MA20クロス・1銘柄・ロングのみ） ──────
def backtest_symbol(symbol: str, df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["ma5"]  = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df = df.dropna(subset=["ma5", "ma20"]).reset_index(drop=True)

    trades: list[dict] = []
    in_position = False
    entry_date: pd.Timestamp | None = None
    entry_price: float = 0.0

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if not in_position:
            # entry: close > MA20 and prev_close <= prev_MA20
            if curr["close"] > curr["ma20"] and prev["close"] <= prev["ma20"]:
                in_position = True
                entry_date  = curr["date"]
                entry_price = curr["close"]

        else:
            # exit: close < MA20 and prev_close >= prev_MA20
            if curr["close"] < curr["ma20"] and prev["close"] >= prev["ma20"]:
                exit_price = curr["close"]
                ret = (exit_price - entry_price) / entry_price
                trades.append({
                    "symbol":     symbol,
                    "entry_date": entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, "strftime") else str(entry_date)[:10],
                    "exit_date":  curr["date"].strftime("%Y-%m-%d") if hasattr(curr["date"], "strftime") else str(curr["date"])[:10],
                    "ret":        round(ret, 6),
                    "trade_regime": _label_trade_regime(ret),
                })
                in_position = False

    return trades


# ── 検証 ─────────────────────────────────────────
def validate(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n[VALIDATION]")
    print(f"  総trades数  : {n}")
    print(f"  ret min     : {df['ret'].min():.4f}")
    print(f"  ret max     : {df['ret'].max():.4f}")
    print(f"  ret mean    : {df['ret'].mean():.4f}")
    print(f"  ret std     : {df['ret'].std():.4f}")

    anomaly = df[df["ret"].abs() > 1]
    if not anomaly.empty:
        print(f"  [WARN] |ret|>1 異常値 {len(anomaly)}件:")
        print(anomaly[["symbol", "entry_date", "exit_date", "ret"]].to_string(index=False))

    if n < MIN_TRADES:
        print(f"  [WARN] trades数 {n} < {MIN_TRADES}（目標未達）")
    else:
        print(f"  [OK] trades数 {n} >= {MIN_TRADES}")


# ── メイン ────────────────────────────────────────
def main() -> None:
    # ユニバース読み込み
    uni = pd.read_csv(UNIVERSE_FILE)
    symbols = uni["symbol"].tolist()
    assert len(symbols) >= MIN_SYMBOLS, f"銘柄数 {len(symbols)} < {MIN_SYMBOLS}"
    print(f"ユニバース: {len(symbols)}銘柄")

    all_trades: list[dict] = []
    skipped: list[str] = []

    for sym in symbols:
        csv_path = OHLCV_DIR / f"{sym}.csv"

        # OHLCVデータ取得（キャッシュ利用）
        try:
            if csv_path.exists():
                df = load_ohlcv(csv_path)
                print(f"  {sym}: キャッシュ {len(df)}行")
            else:
                print(f"  {sym}: yfinance取得中...", end=" ", flush=True)
                df = fetch_ohlcv(sym, csv_path)
                print(f"{len(df)}行")
        except Exception as e:
            print(f"  {sym}: [ERROR] {e} → skip")
            skipped.append(sym)
            continue

        # バックテスト
        trades = backtest_symbol(sym, df)
        if not trades:
            print(f"  {sym}: trades=0 → skip")
            skipped.append(sym)
            continue

        print(f"  {sym}: {len(trades)}trades")
        all_trades.extend(trades)

    if not all_trades:
        raise RuntimeError("全銘柄でtrades=0")

    # trades_log生成
    df_out = pd.DataFrame(all_trades)
    df_out = df_out.sort_values(["entry_date", "symbol"]).reset_index(drop=True)

    # 指定列のみ出力（symbol は検証用に保持→最後に絞る）
    validate(df_out)

    # trade_regime分布チェック（損益ベース分類・市場レジームではない）
    tr_meta = _check_trade_regime_distribution(df_out)
    TRADE_REGIME_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRADE_REGIME_META_PATH.write_text(
        json.dumps(tr_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if tr_meta["trade_regime_imbalance_flag"]:
        print(
            f"\n[WARN] trade_regime偏り検出: "
            f"up={tr_meta['trade_regime_ratio_up']:.2%} "
            f"down={tr_meta['trade_regime_ratio_down']:.2%} "
            f"range={tr_meta['trade_regime_ratio_range']:.2%}"
        )

    df_out = df_out[["entry_date", "exit_date", "ret", "trade_regime"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    print(f"\n[OUTPUT] {OUT_PATH}")
    print(f"  行数   : {len(df_out)}")
    print(f"  期間   : {df_out['entry_date'].min()} 〜 {df_out['entry_date'].max()}")
    print(f"  trade_regime : up={tr_meta['trade_regime_ratio_up']:.2%} down={tr_meta['trade_regime_ratio_down']:.2%} range={tr_meta['trade_regime_ratio_range']:.2%}")
    if skipped:
        print(f"  skip銘柄: {skipped}")


if __name__ == "__main__":
    main()
