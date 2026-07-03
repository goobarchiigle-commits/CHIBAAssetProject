import sys
sys.stdout.reconfigure(encoding='utf-8')

import json, glob, os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

PARQUET_DIR = Path("data/backtest_dataset/2026-03-28")
SIGNAL_DIR  = Path("data/signals")

# ── Load all parquet price data ──────────────────────────────────────────────
price_cache = {}
last_date_cache = {}
for f in PARQUET_DIR.glob("*.parquet"):
    sym = f.stem
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    s = df["Close"].sort_index()
    price_cache[sym] = s
    last_date_cache[sym] = s.index.max()

print(f"Loaded {len(price_cache)} symbols")
print(f"Last date range in parquet: {min(last_date_cache.values()).date()} – {max(last_date_cache.values()).date()}")

# ── entry_ok computation ─────────────────────────────────────────────────────
def compute_entry_ok(sym, sig_date):
    if sym not in price_cache:
        return None
    price = price_cache[sym]
    sig_ts = pd.Timestamp(sig_date)
    avail = price[price.index <= sig_ts]
    if len(avail) < 26:
        return None
    ma25 = avail.rolling(25).mean()
    ma25_slope = ma25.diff(5)
    p   = float(avail.iloc[-1])
    m   = float(ma25.iloc[-1])
    sl  = float(ma25_slope.iloc[-1])
    r5  = float((avail.iloc[-1] / avail.iloc[-6]) - 1) if len(avail) >= 6 else np.nan

    above_ma  = p > m
    slope_pos = sl > 0
    trend_ok  = above_ma or slope_pos
    mom_ok    = (not np.isnan(r5)) and (r5 > 0)
    ok        = trend_ok and mom_ok
    return dict(entry_ok=ok, price=p, ma25=m, above_ma=above_ma,
                slope_pos=slope_pos, trend_ok=trend_ok, mom_ok=mom_ok, ret_5d_in=r5)

def forward_returns(sym, sig_date):
    if sym not in price_cache:
        return {}
    price = price_cache[sym]
    sig_ts = pd.Timestamp(sig_date)
    avail_at_entry = price[price.index <= sig_ts]
    future = price[price.index > sig_ts]
    if avail_at_entry.empty:
        return {}
    ep = float(avail_at_entry.iloc[-1])
    result = {"n_future_days": len(future), "ep": ep}
    for d in [5, 10, 20]:
        result[f"ret_{d}d"] = float((future.iloc[d-1] / ep) - 1) if len(future) >= d else np.nan
    return result

# ── Load signals ─────────────────────────────────────────────────────────────
# Deduplicate: keep signal=1 if any file for (date, symbol) has it
seen_pairs = {}  # (date, sym) -> best signal record
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
        sym = s.get("symbol", "")
        key = (sig_date, sym)
        sig_val = s.get("signal", 0)
        rsr_val = s.get("rsr", 0)
        if key not in seen_pairs or sig_val > seen_pairs[key]["signal"]:
            seen_pairs[key] = {"date": sig_date, "symbol": sym,
                               "signal": sig_val, "rsr": rsr_val}

all_signals = list(seen_pairs.values())
buy_sig1_rsr75 = [s for s in all_signals if s["signal"] == 1 and s["rsr"] >= 75]
rsr75_all      = [s for s in all_signals if s["rsr"] >= 75]

print(f"\nSignal inventory:")
print(f"  Total unique (date,symbol) pairs : {len(all_signals)}")
print(f"  signal=1, RSR≥75 (strict BUY)   : {len(buy_sig1_rsr75)}")
print(f"  RSR≥75 any signal (broad cands)  : {len(rsr75_all)}")

# ── Compute entry_ok + forward returns for broad RSR≥75 universe ─────────────
rows = []
for sig in rsr75_all:
    eo = compute_entry_ok(sig["symbol"], sig["date"])
    if eo is None:
        continue
    fwd = forward_returns(sig["symbol"], sig["date"])
    rows.append({**sig, **eo, **fwd})

df = pd.DataFrame(rows)
df_buy = df[df["signal"] == 1]  # strict BUY signals in the broad set

print(f"\n  RSR≥75 rows with computable entry_ok : {len(df)}")
print(f"  Of those, signal=1 rows              : {len(df_buy)}")

# ── TASK 4: Filter rate on strict BUY signals ────────────────────────────────
n_tot  = len(df_buy)
n_blk  = (df_buy["entry_ok"] == False).sum()
n_pass = (df_buy["entry_ok"] == True).sum()
rate   = n_blk / n_tot * 100 if n_tot > 0 else 0

print("\n" + "="*68)
print("TASK 4 — entry_ok FILTER RATE  (signal=1, RSR≥75 BUY signals only)")
print("="*68)
print(f"  Unique strict BUY signals (signal=1, RSR≥75) : {n_tot}")
print(f"  entry_ok=True  (PASS)                        : {n_pass} ({100-rate:.1f}%)")
print(f"  entry_ok=False (BLOCKED)                     : {n_blk}  ({rate:.1f}%)")
print(f"  Target 30-60% blocked: {'✓ IN RANGE' if 30<=rate<=60 else '✗ OUT OF RANGE'}")

# Also report on broader RSR≥75 universe
n_rsr = len(df)
n_rsr_pass = (df["entry_ok"] == True).sum()
n_rsr_blk  = (df["entry_ok"] == False).sum()
rate_rsr    = n_rsr_blk / n_rsr * 100 if n_rsr else 0
print(f"\n  Broad RSR≥75 universe (all signals)          : {n_rsr}")
print(f"  entry_ok=True  (PASS)                        : {n_rsr_pass} ({100-rate_rsr:.1f}%)")
print(f"  entry_ok=False (BLOCKED)                     : {n_rsr_blk}  ({rate_rsr:.1f}%)")
print(f"  Broad filter rate                            : {rate_rsr:.1f}% blocked")

# ── TASK 1: Forward return distribution ──────────────────────────────────────
print("\n" + "="*68)
print("TASK 1 — FORWARD RETURN DISTRIBUTION  (RSR≥75 universe, n=%d rows)" % len(df))
print("="*68)

def pct(v): return f"{v*100:+.2f}%" if not np.isnan(v) else "  N/A"
def stats_row(sub, col):
    s = sub[col].dropna() * 100
    if len(s) == 0:
        return f"n={len(sub)} (no fwd data)"
    win = (s > 0).sum()
    return (f"mean={s.mean():+.2f}% | med={s.median():+.2f}% | "
            f"win={win/len(s)*100:.0f}% | n={len(s)}")

passed  = df[df["entry_ok"] == True]
blocked = df[df["entry_ok"] == False]

print(f"\n  entry_ok=TRUE  (n={len(passed)}):")
for col, label in [("ret_5d","5d"), ("ret_10d","10d"), ("ret_20d","20d")]:
    print(f"    {label:4s}: {stats_row(passed, col)}")

print(f"\n  entry_ok=FALSE (n={len(blocked)}):")
for col, label in [("ret_5d","5d"), ("ret_10d","10d"), ("ret_20d","20d")]:
    print(f"    {label:4s}: {stats_row(blocked, col)}")

# Breakdown within blocked
print(f"\n  BLOCKED detail (5d fwd, n={blocked['ret_5d'].dropna().shape[0]}):")
b5 = blocked["ret_5d"].dropna() * 100
if len(b5):
    print(f"    Losses avoided (ret<0): n={( b5<0).sum()} | avg if entered: {b5[b5<0].mean():+.2f}%")
    print(f"    Gains missed  (ret>0) : n={(b5>0).sum()} | avg if entered: {b5[b5>0].mean():+.2f}%")
    print(f"    Neutral (ret=0)       : n={(b5==0).sum()}")

print(f"\n  BLOCKED detail (10d fwd, n={blocked['ret_10d'].dropna().shape[0]}):")
b10 = blocked["ret_10d"].dropna() * 100
if len(b10):
    print(f"    Losses avoided (ret<0): n={(b10<0).sum()} | avg if entered: {b10[b10<0].mean():+.2f}%")
    print(f"    Gains missed  (ret>0) : n={(b10>0).sum()} | avg if entered: {b10[b10>0].mean():+.2f}%")

# ── TASK 2: Actual 3 trades ──────────────────────────────────────────────────
print("\n" + "="*68)
print("TASK 2 — ACTUAL TRADES  (entry_ok status + available fwd returns)")
print("="*68)
ACTUAL = [
    ("5401.T", "2026-03-16", "JFE Steel — user entry, suspected loss in April tariff crash"),
    ("5411.T", "2026-03-16", "Nippon Steel — user entry, suspected loss in April tariff crash"),
    ("9104.T", "2026-04-15", "Mitsui O.S.K. — recent entry, no fwd data yet"),
]
for sym, dt, note in ACTUAL:
    eo  = compute_entry_ok(sym, dt)
    fwd = forward_returns(sym, dt)
    print(f"\n  {sym} @ {dt}  [{note}]")
    if eo is None:
        print("    ✗ NO DATA in parquet")
        continue
    ok_str = "PASS ✓" if eo["entry_ok"] else "BLOCKED ✗"
    print(f"    entry_ok    : {ok_str}")
    print(f"    above_ma25  : {eo['above_ma']}   slope_pos: {eo['slope_pos']}   5d_momentum: {pct(eo['ret_5d_in'])}")
    print(f"    Parquet fwd : {fwd.get('n_future_days',0)} trading days available after entry")
    ep = fwd.get("ep", np.nan)
    for d in [5, 10, 20]:
        r = fwd.get(f"ret_{d}d", np.nan)
        if not np.isnan(r):
            print(f"    fwd {d:2d}d : entry {ep:.1f} → {ep*(1+r):.1f} ({pct(r)})")
        else:
            print(f"    fwd {d:2d}d : N/A (parquet ends {last_date_cache.get(sym,'?').date() if sym in last_date_cache else '?'})")
    if eo["entry_ok"] == False:
        print(f"    ► Filter would have BLOCKED this entry")
    else:
        print(f"    ► Filter would have ALLOWED this entry")

# ── TASK 3: RSR≥75, entry_ok=True, NOT traded ────────────────────────────────
print("\n" + "="*68)
print("TASK 3 — MISSED OPPORTUNITIES  (RSR≥75, entry_ok=True, not traded)")
print("="*68)
TRADED = {"5401.T", "5411.T", "9104.T", "7203.T"}
missed = df[(df["entry_ok"] == True) & (~df["symbol"].isin(TRADED))].copy()
missed = missed.sort_values(["rsr","ret_20d"], ascending=[False, False])
missed_unique = missed.drop_duplicates(subset=["symbol"])

print(f"\n  {'Symbol':<10} {'Date':<12} {'RSR':>6} {'Sig':>4} {'5d':>8} {'10d':>8} {'20d':>8} {'N_days':>7}")
print(f"  {'-'*10} {'-'*12} {'-'*6} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
for _, row in missed_unique.head(20).iterrows():
    r5  = f"{row.get('ret_5d',np.nan)*100:+.2f}" if not np.isnan(row.get('ret_5d',np.nan)) else " N/A"
    r10 = f"{row.get('ret_10d',np.nan)*100:+.2f}" if not np.isnan(row.get('ret_10d',np.nan)) else " N/A"
    r20 = f"{row.get('ret_20d',np.nan)*100:+.2f}" if not np.isnan(row.get('ret_20d',np.nan)) else " N/A"
    nd  = int(row.get("n_future_days", 0))
    print(f"  {row['symbol']:<10} {row['date']:<12} {row['rsr']:>6.1f} {int(row['signal']):>4} {r5:>8} {r10:>8} {r20:>8} {nd:>7}")

m_10 = missed_unique["ret_10d"].dropna() * 100
m_20 = missed_unique["ret_20d"].dropna() * 100
if len(m_10):
    print(f"\n  Mean 10d return (missed, n={len(m_10)}): {m_10.mean():+.2f}%  win={( m_10>0).sum()}/{len(m_10)}")
if len(m_20):
    print(f"  Mean 20d return (missed, n={len(m_20)}): {m_20.mean():+.2f}%  win={(m_20>0).sum()}/{len(m_20)}")

# ── CONSOLIDATED SUMMARY ──────────────────────────────────────────────────────
print("\n" + "="*68)
print("CONSOLIDATED SUMMARY — entry_ok filter impact assessment")
print("="*68)
p_5  = passed["ret_5d"].dropna() * 100
p_10 = passed["ret_10d"].dropna() * 100
b5_  = blocked["ret_5d"].dropna() * 100
b10_ = blocked["ret_10d"].dropna() * 100

print(f"""
  DATA WINDOW  : 2026-03-04 – 2026-04-20 (parquet ends 2026-03-27 to 2026-04-03)
  UNIVERSE     : RSR≥75 candidates = {len(df)} (strict BUY signal=1+RSR≥75 = {len(df_buy)})

  ┌─────────────────────────────────────────────────────────┐
  │ FILTER RATE (strict BUY signals)                        │
  │   {n_pass}/{n_tot} passed ({100-rate:.0f}%)   {n_blk}/{n_tot} blocked ({rate:.0f}%)          │
  │   Broad RSR≥75: {n_rsr_pass}/{n_rsr} passed ({100-rate_rsr:.0f}%)  {n_rsr_blk}/{n_rsr} blocked ({rate_rsr:.0f}%)  │
  ├─────────────────────────────────────────────────────────┤
  │ PASSED SIGNALS (entry_ok=True)                          │
  │   5d  mean: {p_5.mean():+.2f}%   win: {(p_5>0).sum()}/{len(p_5)} ({(p_5>0).mean()*100:.0f}%)            │
  │   10d mean: {p_10.mean():+.2f}%   win: {(p_10>0).sum()}/{len(p_10)} ({(p_10>0).mean()*100:.0f}%)            │
  ├─────────────────────────────────────────────────────────┤
  │ BLOCKED SIGNALS (entry_ok=False)                        │
  │   5d  mean: {b5_.mean():+.2f}%   win: {(b5_>0).sum()}/{len(b5_)} ({(b5_>0).mean()*100:.0f}%)            │
  │   10d mean: {b10_.mean():+.2f}%   win: {(b10_>0).sum()}/{len(b10_)} ({(b10_>0).mean()*100:.0f}%)            │
  ├─────────────────────────────────────────────────────────┤
  │ FILTER EDGE (pass - block, 10d)                         │
  │   {p_10.mean()-b10_.mean():+.2f}% mean return advantage for passed signals      │
  ├─────────────────────────────────────────────────────────┤
  │ ACTUAL TRADES                                           │
  │   5401.T 2026-03-16: BLOCKED ✗  (8d in parquet: +2.5%, April crash →?)  │
  │   5411.T 2026-03-16: BLOCKED ✗  (8d in parquet: +0.4%, April crash →?)  │
  │   9104.T 2026-04-15: PASS ✓     (no parquet fwd data, after cutoff)      │
  └─────────────────────────────────────────────────────────┘

  VERDICT:
""")

if rate_rsr >= 30:
    print(f"  ✓ Filter is not too lenient (blocks {rate_rsr:.0f}% of RSR≥75 candidates)")
else:
    print(f"  ⚠ Filter may be too lenient (blocks only {rate_rsr:.0f}% of RSR≥75 candidates)")

if rate_rsr <= 60:
    print(f"  ✓ Filter is not too aggressive ({100-rate_rsr:.0f}% of RSR≥75 candidates pass)")
else:
    print(f"  ⚠ Filter may be too aggressive (only {100-rate_rsr:.0f}% of RSR≥75 candidates pass)")

edge = p_10.mean() - b10_.mean() if len(p_10) and len(b10_) else np.nan
if not np.isnan(edge):
    if edge > 0:
        print(f"  ✓ Passed signals outperform blocked by {edge:+.2f}% (10d), filter adds value")
    else:
        print(f"  ✗ Blocked signals actually performed better by {-edge:.2f}% (10d), filter hurts")

print(f"\n  SAMPLE SIZE WARNING: Only {len(df_buy)} strict BUY signals available.")
print(f"  Statistical confidence is LOW. Use broad RSR≥75 (n={len(df)}) for inference.")
print(f"  April crash (tariff shock) data missing from parquet — true loss-avoidance")
print(f"  for 5401.T/5411.T requires price data beyond 2026-03-27.")
