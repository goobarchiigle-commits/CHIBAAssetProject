"""
confirm_exit_1pt.py — Single-point confirmation backtest
Parameters: time_stop=4, trail_stop=0.025, rsr_exit=1.1
Run from C:/ai-trading/
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ── params ────────────────────────────────────────────────────────────────────
TIME_STOP   = 4
TRAIL_STOP  = 0.025
RSR_EXIT    = 1.1
ENTRY_THRESH = 1.0

# pass/fail thresholds
PASS_DD    = -0.12      # DD <= -12.00%
PASS_EXP   = 0.008      # expectancy >= 0.80%
PASS_PF    = 1.8        # profit_factor >= 1.8

# ── load data (identical to rsr_research.py main) ─────────────────────────────
price_path = r"C:\ai-trading\cache\ohlcv\5401.T.parquet"
if not os.path.exists(price_path):
    print(f"[ERROR] File not found: {price_path}")
    sys.exit(1)

price_df = pd.read_parquet(price_path)
price_df.columns = [c.lower() for c in price_df.columns]

close   = price_df["close"]
returns = close.pct_change(20)
vol     = close.pct_change().rolling(20).std()
raw     = returns / vol
rsr_z   = (raw - raw.rolling(60).mean()) / raw.rolling(60).std()
rsr_z.name = "rsr_z"

signal_mask = rsr_z > ENTRY_THRESH
entry_dates = price_df.index[signal_mask]

# ── simulate_trade (copied verbatim from rsr_research.py) ─────────────────────
def simulate_trade(price_df, rsr_z, entry_date, time_stop, trail_stop, rsr_exit_thresh):
    opens     = price_df["open"]
    highs     = price_df["high"]
    all_dates = price_df.index

    if entry_date not in all_dates:
        return None
    sig_pos   = all_dates.get_loc(entry_date)
    entry_pos = sig_pos + 1
    if entry_pos >= len(all_dates):
        return None

    entry_px = opens.iloc[entry_pos]
    if entry_px <= 0:
        return None

    hwm      = entry_px
    exit_px  = None
    exit_bar = None

    for bar in range(1, time_stop + 1):
        cur_pos = entry_pos + bar
        if cur_pos >= len(all_dates):
            exit_px  = opens.iloc[len(all_dates) - 1]
            exit_bar = bar
            break

        prev_high = highs.iloc[cur_pos - 1]
        hwm = max(hwm, prev_high)

        trail_dd = (opens.iloc[cur_pos] - hwm) / hwm
        if trail_dd <= -trail_stop:
            exit_px  = opens.iloc[cur_pos]
            exit_bar = bar
            break

        prev_date = all_dates[cur_pos - 1]
        if prev_date in rsr_z.index and rsr_z.loc[prev_date] < rsr_exit_thresh:
            exit_px  = opens.iloc[cur_pos]
            exit_bar = bar
            break

        if bar == time_stop:
            exit_px  = opens.iloc[cur_pos]
            exit_bar = bar

    if exit_px is None or exit_px <= 0:
        return None

    ret = (exit_px - entry_px) / entry_px
    return {"entry_date": all_dates[entry_pos], "exit_bar": exit_bar,
            "entry_px": entry_px, "exit_px": exit_px, "ret": ret}


# ── run confirmation ──────────────────────────────────────────────────────────
rets = []
for ed in entry_dates:
    r = simulate_trade(price_df, rsr_z, ed, TIME_STOP, TRAIL_STOP, RSR_EXIT)
    if r is not None:
        rets.append(r["ret"])

rets_arr = np.array(rets)
n        = len(rets_arr)

expectancy = float(np.mean(rets_arr))

cum         = np.cumprod(1.0 + rets_arr)
rolling_max = np.maximum.accumulate(cum)
max_dd      = float((cum / rolling_max - 1.0).min())

wins   = rets_arr[rets_arr > 0].sum()
losses = abs(rets_arr[rets_arr < 0].sum())
pf     = float(wins / losses) if losses > 0 else float("inf")

# ── output ────────────────────────────────────────────────────────────────────
print(f"\nConfirmation: time_stop={TIME_STOP}  trail_stop={TRAIL_STOP}  rsr_exit={RSR_EXIT}")
print("-" * 50)
print(f"  expectancy   : {expectancy*100:+.2f}%")
print(f"  max_drawdown : {max_dd*100:.2f}%")
print(f"  profit_factor: {pf:.3f}")
print(f"  n_trades     : {n}")
print("-" * 50)

p_dd  = "PASS" if max_dd >= PASS_DD  else "FAIL"
p_exp = "PASS" if expectancy >= PASS_EXP else "FAIL"
p_pf  = "PASS" if pf >= PASS_PF else "FAIL"

print(f"  DD  <= -12.00% : {p_dd}  ({max_dd*100:.2f}%)")
print(f"  EXP >= 0.80%   : {p_exp}  ({expectancy*100:.2f}%)")
print(f"  PF  >= 1.80    : {p_pf}  ({pf:.3f})")
print("-" * 50)

verdict = "PASS" if all(x == "PASS" for x in [p_dd, p_exp, p_pf]) else "FAIL"
print(f"  OVERALL        : {verdict}")
print()
