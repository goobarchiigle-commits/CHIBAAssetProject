import sys
sys.stdout.reconfigure(encoding='utf-8')

import json, glob, os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

PARQUET_DIR = Path("data/backtest_dataset/2026-03-28")
SIGNAL_DIR = Path("data/signals")
TRADES_FILE = Path("logs/trades.jsonl")

# ── 1. Load all parquet price data ──────────────────────────────────────────
print("Loading parquet files...")
price_cache = {}
for f in PARQUET_DIR.glob("*.parquet"):
    sym = f.stem  # e.g. "6920.T"
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    price_cache[sym] = df["Close"].sort_index()

print(f"  Loaded {len(price_cache)} symbols. Price data ends: {list(price_cache.values())[0].index.max().date()}")

# ── 2. Compute entry_ok for a symbol at a given date ────────────────────────
def compute_entry_ok(sym, sig_date):
    """Returns (entry_ok, price, ma25, slope_ok, mom_ok) or None if data missing."""
    if sym not in price_cache:
        return None
    price = price_cache[sym]
    sig_ts = pd.Timestamp(sig_date)
    avail = price[price.index <= sig_ts]
    if len(avail) < 26:
        return None
    ma25 = avail.rolling(25).mean()
    ma25_slope = ma25.diff(5)
    p = avail.iloc[-1]
    m = ma25.iloc[-1]
    sl = ma25_slope.iloc[-1]
    if len(avail) < 6:
        r5 = np.nan
    else:
        r5 = (avail.iloc[-1] / avail.iloc[-6]) - 1

    trend_filter = (p > m) or (sl > 0)
    momentum_filter = (not np.isnan(r5)) and (r5 > 0)
    entry_ok = trend_filter and momentum_filter
    return {
        "entry_ok": entry_ok,
        "price": p,
        "ma25": m,
        "above_ma25": p > m,
        "slope_pos": sl > 0,
        "ret_5d_in": r5,
        "trend_filter": trend_filter,
        "momentum_filter": momentum_filter,
    }

# ── 3. Compute forward returns ───────────────────────────────────────────────
def forward_returns(sym, sig_date):
    """5d/10d/20d forward return from sig_date."""
    if sym not in price_cache:
        return {}
    price = price_cache[sym]
    sig_ts = pd.Timestamp(sig_date)
    future = price[price.index > sig_ts]
    entry_price = price[price.index <= sig_ts]
    if entry_price.empty:
        return {}
    ep = entry_price.iloc[-1]
    result = {}
    for d in [5, 10, 20]:
        if len(future) >= d:
            result[f"ret_{d}d"] = (future.iloc[d - 1] / ep) - 1
        else:
            result[f"ret_{d}d"] = np.nan
    return result

# ── 4. Load and deduplicate BUY signals ─────────────────────────────────────
print("Loading signal files...")
seen = set()
buy_signals = []
for f in sorted(SIGNAL_DIR.glob("signal_*.json")):
    try:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        continue
    sig_date = d.get("data_as_of", "")
    if not sig_date:
        continue
    for s in d.get("signals", []):
        if s.get("signal", 0) == 1 and s.get("rsr", 0) >= 75:
            key = (sig_date, s["symbol"])
            if key not in seen:
                seen.add(key)
                buy_signals.append({
                    "date": sig_date,
                    "symbol": s["symbol"],
                    "rsr": s.get("rsr", 0),
                })

print(f"  Unique BUY signals (signal=1, RSR≥75): {len(buy_signals)}")

# ── 5. Main simulation ───────────────────────────────────────────────────────
print("Running entry_ok simulation...")
rows = []
for sig in buy_signals:
    sym = sig["symbol"]
    dt = sig["date"]
    eo = compute_entry_ok(sym, dt)
    if eo is None:
        continue
    fwd = forward_returns(sym, dt)
    rows.append({
        "date": dt,
        "symbol": sym,
        "rsr": sig["rsr"],
        **eo,
        **fwd,
    })

df = pd.DataFrame(rows)
print(f"  Computable rows: {len(df)}")

# ── TASK 4: Filter rate ──────────────────────────────────────────────────────
n_total = len(df)
n_blocked = (df["entry_ok"] == False).sum()
n_passed = (df["entry_ok"] == True).sum()
filter_rate = n_blocked / n_total * 100 if n_total > 0 else 0

print("\n" + "="*65)
print("TASK 4 — entry_ok FILTER RATE")
print("="*65)
print(f"  Total BUY signals computed : {n_total}")
print(f"  entry_ok=True  (PASS)      : {n_passed} ({100-filter_rate:.1f}%)")
print(f"  entry_ok=False (BLOCKED)   : {n_blocked} ({filter_rate:.1f}%)")
print(f"  Target: 30-60% blocked → {'✓ IN RANGE' if 30 <= filter_rate <= 60 else '✗ OUT OF RANGE'}")

# ── TASK 1: Forward return distribution by entry_ok ─────────────────────────
print("\n" + "="*65)
print("TASK 1 — FORWARD RETURN DISTRIBUTION (mean %)")
print("="*65)
for col in ["ret_5d", "ret_10d", "ret_20d"]:
    df[col + "_pct"] = df[col] * 100

passed = df[df["entry_ok"] == True]
blocked = df[df["entry_ok"] == False]

def stats(sub, col):
    s = sub[col].dropna()
    if len(s) == 0:
        return "N/A"
    pos = (s > 0).sum()
    return f"mean={s.mean():+.2f}% | med={s.median():+.2f}% | win%={pos/len(s)*100:.0f}% | n={len(s)}"

print(f"\n  entry_ok=TRUE (passed):")
for col in ["ret_5d_pct", "ret_10d_pct", "ret_20d_pct"]:
    print(f"    {col:12s}: {stats(passed, col)}")

print(f"\n  entry_ok=FALSE (blocked):")
for col in ["ret_5d_pct", "ret_10d_pct", "ret_20d_pct"]:
    print(f"    {col:12s}: {stats(blocked, col)}")

# Loss avoided = blocked signals with negative forward return
# Missed gain  = blocked signals with positive forward return
print(f"\n  BLOCKED breakdown (5d fwd):")
b5 = blocked["ret_5d_pct"].dropna()
print(f"    Losses avoided (ret<0): {(b5 < 0).sum()} signals | avg loss if entered: {b5[b5 < 0].mean():+.2f}%")
print(f"    Gains missed   (ret>0): {(b5 > 0).sum()} signals | avg gain if entered: {b5[b5 > 0].mean():+.2f}%")

b20 = blocked["ret_20d_pct"].dropna()
print(f"  BLOCKED breakdown (20d fwd):")
print(f"    Losses avoided (ret<0): {(b20 < 0).sum()} signals | avg loss if entered: {b20[b20 < 0].mean():+.2f}%")
print(f"    Gains missed   (ret>0): {(b20 > 0).sum()} signals | avg gain if entered: {b20[b20 > 0].mean():+.2f}%")

# ── TASK 2: Actual 3 trades ──────────────────────────────────────────────────
print("\n" + "="*65)
print("TASK 2 — ACTUAL TRADES: entry_ok STATUS AT ENTRY")
print("="*65)
actual_trades = [
    ("5401.T", "2026-03-16"),
    ("5411.T", "2026-03-16"),
    ("9104.T", "2026-04-15"),
]
for sym, dt in actual_trades:
    eo = compute_entry_ok(sym, dt)
    fwd = forward_returns(sym, dt)
    if eo is None:
        print(f"  {sym} @ {dt}: NO DATA")
        continue
    ok_str = "PASS ✓" if eo["entry_ok"] else "BLOCKED ✗"
    r5  = fwd.get("ret_5d",  np.nan)
    r10 = fwd.get("ret_10d", np.nan)
    r20 = fwd.get("ret_20d", np.nan)
    print(f"  {sym} @ {dt}: entry_ok={ok_str}")
    print(f"    above_ma25={eo['above_ma25']}, slope_pos={eo['slope_pos']}, ret_5d_in={eo['ret_5d_in']:+.3f}")
    print(f"    fwd_5d={r5:+.3f} ({r5*100:+.2f}%) | fwd_10d={r10:+.3f} ({r10*100:+.2f}%) | fwd_20d={r20:+.3f} ({r20*100:+.2f}%)" if not np.isnan(r5) else f"    fwd returns: limited data")

# ── TASK 3: RSR top-10 NOT traded, entry_ok=True ────────────────────────────
print("\n" + "="*65)
print("TASK 3 — MISSED OPPORTUNITIES (RSR≥75, entry_ok=True, NOT traded)")
print("="*65)
# Actual traded symbols
actual_symbols = {"5401.T", "5411.T", "9104.T", "7203.T"}
missed = passed[~passed["symbol"].isin(actual_symbols)].copy()
missed = missed.sort_values("rsr", ascending=False).drop_duplicates(subset=["symbol"])
missed_top = missed.head(15)
print(f"\n  {'Symbol':<10} {'Date':<12} {'RSR':>6} {'5d%':>8} {'10d%':>8} {'20d%':>8}")
print(f"  {'-'*10} {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
for _, row in missed_top.iterrows():
    r5  = f"{row.get('ret_5d_pct', np.nan):+.2f}" if not np.isnan(row.get('ret_5d_pct', np.nan)) else "  N/A"
    r10 = f"{row.get('ret_10d_pct', np.nan):+.2f}" if not np.isnan(row.get('ret_10d_pct', np.nan)) else "  N/A"
    r20 = f"{row.get('ret_20d_pct', np.nan):+.2f}" if not np.isnan(row.get('ret_20d_pct', np.nan)) else "  N/A"
    print(f"  {row['symbol']:<10} {row['date']:<12} {row['rsr']:>6.1f} {r5:>8} {r10:>8} {r20:>8}")

print(f"\n  Missed mean 20d return: {missed_top['ret_20d_pct'].dropna().mean():+.2f}%")
print(f"  Missed win% (20d):      {(missed_top['ret_20d_pct'].dropna() > 0).sum()}/{len(missed_top['ret_20d_pct'].dropna())}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"  Filter rate               : {filter_rate:.1f}% blocked")
print(f"  Passed mean 20d return    : {passed['ret_20d_pct'].dropna().mean():+.2f}%")
print(f"  Blocked mean 20d return   : {blocked['ret_20d_pct'].dropna().mean():+.2f}%")
print(f"  Filter edge (pass-block)  : {passed['ret_20d_pct'].dropna().mean() - blocked['ret_20d_pct'].dropna().mean():+.2f}%")
print(f"  Losses avoided / missed gains (20d): {(b20<0).sum()} / {(b20>0).sum()}")
