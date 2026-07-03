"""
entry_ok フィルター影響分析
trend_filter = (price > ma25) OR (ma25_slope > 0 over 5d)
momentum_filter = ret_5d > 0
entry_ok = trend_filter AND momentum_filter
"""
import sys
import json
import os
import glob
import warnings
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

SIGNALS_DIR = "C:/ai-trading/data/signals"
OHLCV_DIRS = [
    "C:/ai-trading/cache/ohlcv",
    "C:/ai-trading/src/cache/ohlcv",
]
TARGET_SYMBOLS = ["5401.T", "5411.T", "9104.T"]

# ── 1. 全BUYシグナル収集 ────────────────────────────────────────────
def load_buy_signals():
    records = []
    files = sorted(glob.glob(f"{SIGNALS_DIR}/signal_*.json"))
    if not files:
        print("ERROR: シグナルJSONが見つかりません:", SIGNALS_DIR)
        sys.exit(1)

    seen = set()  # (date, symbol) dedup
    for fp in files:
        try:
            with open(fp, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue
        date_str = d.get("data_as_of", "")
        if not date_str:
            continue
        for sig in d.get("signals", []):
            if sig.get("signal") != 1:
                continue
            sym = sig.get("symbol", "")
            key = (date_str, sym)
            if key in seen:
                continue
            seen.add(key)
            records.append({"date": pd.Timestamp(date_str), "symbol": sym})

    return pd.DataFrame(records)

# ── 2. OHLCVロード ─────────────────────────────────────────────────
_ohlcv_cache = {}

def load_ohlcv(symbol: str) -> pd.DataFrame | None:
    if symbol in _ohlcv_cache:
        return _ohlcv_cache[symbol]
    for d in OHLCV_DIRS:
        fp = os.path.join(d, f"{symbol}.parquet")
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            # MultiIndex列を flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            df = df.sort_index()
            _ohlcv_cache[symbol] = df
            return df
    _ohlcv_cache[symbol] = None
    return None

# ── 3. フィルター計算 ──────────────────────────────────────────────
def compute_filters(df: pd.DataFrame, date: pd.Timestamp) -> dict:
    """date時点でのフィルター値を返す"""
    result = dict(trend_filter=None, momentum_filter=None,
                  entry_ok=None, fwd_ret_5d=None, close=None)
    if df is None or date not in df.index:
        # 最寄り日を探す（±3営業日）
        if df is None:
            return result
        nearby = df.index[df.index <= date]
        if len(nearby) == 0:
            return result
        date = nearby[-1]

    loc = df.index.get_loc(date)

    # Close at entry
    close = df["Close"].iloc[loc]
    result["close"] = close

    # MA25
    if loc < 24:  # 25本未満
        return result
    window = df["Close"].iloc[max(0, loc - 24): loc + 1]
    ma25 = window.mean()

    # MA25 slope over 5d
    if loc >= 28:
        ma25_series = df["Close"].iloc[loc - 28: loc + 1].rolling(25).mean().dropna()
        if len(ma25_series) >= 5:
            recent5 = ma25_series.iloc[-5:].values
            x = np.arange(len(recent5))
            slope = np.polyfit(x, recent5, 1)[0]
        else:
            slope = 0.0
    else:
        slope = 0.0

    # ret_5d (過去5営業日リターン = モメンタム)
    if loc >= 5:
        close_5ago = df["Close"].iloc[loc - 5]
        ret_5d = (close - close_5ago) / close_5ago
    else:
        ret_5d = None

    # 翌5営業日リターン
    if loc + 5 < len(df):
        close_fwd = df["Close"].iloc[loc + 5]
        fwd_ret_5d = (close_fwd - close) / close
    else:
        fwd_ret_5d = None

    # フィルター
    trend_filter = (close > ma25) or (slope > 0)
    if ret_5d is not None:
        momentum_filter = ret_5d > 0
        entry_ok = trend_filter and momentum_filter
    else:
        momentum_filter = None
        entry_ok = None

    result.update(dict(
        trend_filter=trend_filter,
        momentum_filter=momentum_filter,
        entry_ok=entry_ok,
        fwd_ret_5d=fwd_ret_5d,
        ma25=ma25,
        slope=slope,
        ret_5d=ret_5d,
    ))
    return result

# ── Main ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("entry_ok フィルター影響分析")
    print("=" * 60)

    buy_signals = load_buy_signals()
    if buy_signals.empty:
        print("ERROR: BUYシグナル (signal=1) が存在しません")
        return

    print(f"BUYシグナル読み込み: {len(buy_signals)} 件 (重複除去済み)\n")

    rows = []
    missing_ohlcv = set()

    for _, row in buy_signals.iterrows():
        sym = row["symbol"]
        date = row["date"]
        df = load_ohlcv(sym)
        if df is None:
            missing_ohlcv.add(sym)
            continue
        res = compute_filters(df, date)
        if res["entry_ok"] is None:
            continue  # データ不足スキップ
        rows.append({
            "symbol": sym,
            "date": date,
            **res,
        })

    if missing_ohlcv:
        print(f"OHLCVなし({len(missing_ohlcv)}銘柄): {sorted(missing_ohlcv)[:10]} ...\n")

    result_df = pd.DataFrame(rows)
    n_total = len(result_df)

    if n_total == 0:
        print("ERROR: 計算可能なシグナルがありません")
        return

    ok_df = result_df[result_df["entry_ok"] == True]
    ng_df = result_df[result_df["entry_ok"] == False]

    n_ok = len(ok_df)
    n_ng = len(ng_df)

    ok_fwd = ok_df["fwd_ret_5d"].dropna() * 100
    ng_fwd = ng_df["fwd_ret_5d"].dropna() * 100

    print(f"総 BUY シグナル数 (計算可): {n_total}")
    print(f"entry_ok=True : {n_ok:4d} 件 ({n_ok/n_total*100:.1f}%)")
    print(f"entry_ok=False: {n_ng:4d} 件 ({n_ng/n_total*100:.1f}%)")
    print()

    def fmt_ret(s):
        if len(s) == 0:
            return "N/A"
        return f"{s.mean():+.2f}% (中央値 {s.median():+.2f}%, n={len(s)})"

    print(f"entry_ok=True  の翌5d平均リターン: {fmt_ret(ok_fwd)}")
    print(f"entry_ok=False の翌5d平均リターン: {fmt_ret(ng_fwd)}")
    print()

    # 銘柄別集計
    print("銘柄別 entry_ok 通過率 (上位10銘柄):")
    print(f"{'Symbol':<10} {'Total':>6} {'OK':>6} {'OK%':>7} {'OK fwd5d':>10} {'NG fwd5d':>10}")
    print("-" * 55)
    for sym, grp in sorted(result_df.groupby("symbol"), key=lambda x: -len(x[1])):
        ok = grp[grp["entry_ok"] == True]
        ng = grp[grp["entry_ok"] == False]
        ok_r = ok["fwd_ret_5d"].dropna().mean() * 100 if len(ok) else float("nan")
        ng_r = ng["fwd_ret_5d"].dropna().mean() * 100 if len(ng) else float("nan")
        ok_pct = len(ok) / len(grp) * 100
        ok_str = f"{ok_r:+.2f}%" if not np.isnan(ok_r) else "  N/A"
        ng_str = f"{ng_r:+.2f}%" if not np.isnan(ng_r) else "  N/A"
        print(f"{sym:<10} {len(grp):>6} {len(ok):>6} {ok_pct:>6.1f}% {ok_str:>10} {ng_str:>10}")
        # 上位10のみ (already sorted by count via lambda)
        break  # actually we need all, remove limit below

    # redo without break
    print()
    print("銘柄別 entry_ok 通過率 (全銘柄):")
    print(f"{'Symbol':<10} {'Total':>6} {'OK':>6} {'OK%':>7} {'OK fwd5d':>10} {'NG fwd5d':>10}")
    print("-" * 55)
    by_sym = (result_df.groupby("symbol")
              .apply(lambda g: pd.Series({
                  "total": len(g),
                  "ok": (g["entry_ok"] == True).sum(),
                  "ok_fwd": g.loc[g["entry_ok"] == True, "fwd_ret_5d"].mean() * 100,
                  "ng_fwd": g.loc[g["entry_ok"] == False, "fwd_ret_5d"].mean() * 100,
              }), include_groups=False)
              .sort_values("total", ascending=False))
    for sym, r in by_sym.iterrows():
        ok_str = f"{r.ok_fwd:+.2f}%" if not np.isnan(r.ok_fwd) else "  N/A"
        ng_str = f"{r.ng_fwd:+.2f}%" if not np.isnan(r.ng_fwd) else "  N/A"
        print(f"{sym:<10} {int(r.total):>6} {int(r.ok):>6} {r.ok/r.total*100:>6.1f}% {ok_str:>10} {ng_str:>10}")

    print()
    print("確認済みブロック (5401.T, 5411.T, 9104.T):")
    for sym in TARGET_SYMBOLS:
        subset = result_df[result_df["symbol"] == sym].sort_values("date")
        if subset.empty:
            print(f"  {sym}: BUYシグナルなし")
            continue
        for _, r in subset.iterrows():
            trend_s = "T" if r.trend_filter else "F"
            mom_s = "T" if r.momentum_filter else "F"
            ok_s = "T" if r.entry_ok else "F"
            fwd_s = f"{r.fwd_ret_5d*100:+.2f}%" if r.fwd_ret_5d is not None else "N/A"
            print(f"  {sym} [{r.date.date()}]: entry_ok={ok_s} (trend={trend_s}, mom={mom_s}) 翌5d={fwd_s}")

    print()
    print("=" * 60)
    print("分析完了")

if __name__ == "__main__":
    main()
