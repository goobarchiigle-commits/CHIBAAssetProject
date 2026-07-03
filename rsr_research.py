"""
rsr_research.py — RSR forward-return / exit-grid / portfolio-DD research
Run from C:/ai-trading/
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DEMO DATA FACTORY  (replace with your real price_df / rsr_z)
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_data(n: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic OHLC + rsr_z for offline testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)

    # geometric random walk
    log_ret = rng.normal(0.0003, 0.015, n)
    close = 1000.0 * np.exp(np.cumsum(log_ret))

    noise_hi = np.abs(rng.normal(0, 0.008, n))
    noise_lo = np.abs(rng.normal(0, 0.008, n))

    price_df = pd.DataFrame(
        {
            "open":  close * (1 + rng.normal(0, 0.004, n)),
            "high":  close * (1 + noise_hi),
            "low":   close * (1 - noise_lo),
            "close": close,
        },
        index=dates,
    )
    price_df = price_df.clip(lower=1.0)

    # synthetic RSR-Z: mean-reverting process with slow trend component
    rsr_raw = np.zeros(n)
    for i in range(1, n):
        rsr_raw[i] = 0.97 * rsr_raw[i - 1] + rng.normal(0, 0.3)
    rsr_z = pd.Series(rsr_raw, index=dates, name="rsr_z")

    return price_df, rsr_z


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FORWARD RETURN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

HORIZONS = [1, 3, 5, 10, 20]


def compute_forward_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execution price = next-day open (no lookahead bias).
    Entry signal fires on close[t]; execution at open[t+1].
    Forward return measured from open[t+1] to open[t+1+h].
    """
    exec_price = price_df["open"].shift(-1)           # open[t+1]
    fwd = {}
    for h in HORIZONS:
        exit_price = price_df["open"].shift(-(1 + h))  # open[t+1+h]
        fwd[f"fwd_{h}d"] = (exit_price - exec_price) / exec_price
    fwd_df = pd.DataFrame(fwd, index=price_df.index)
    return fwd_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRADE-LEVEL MFE / MAE / mfe_days
# ─────────────────────────────────────────────────────────────────────────────

def compute_mfe_mae(
    price_df: pd.DataFrame,
    entry_dates: pd.DatetimeIndex,
    max_hold: int = 20,
) -> pd.DataFrame:
    """
    For each entry date, scan up to max_hold bars after entry.
    Entry at open[t+1].  MFE/MAE use high/low within the hold window.
    mfe_days = first bar that achieved the MFE.
    """
    opens = price_df["open"]
    highs = price_df["high"]
    lows  = price_df["low"]
    all_dates = price_df.index

    records = []
    for sig_date in entry_dates:
        if sig_date not in all_dates:
            continue
        sig_pos = all_dates.get_loc(sig_date)
        entry_pos = sig_pos + 1                         # next-day open
        if entry_pos >= len(all_dates):
            continue
        entry_px = opens.iloc[entry_pos]
        if entry_px <= 0:
            continue

        end_pos = min(entry_pos + max_hold, len(all_dates))
        window_highs = highs.iloc[entry_pos:end_pos].values
        window_lows  = lows.iloc[entry_pos:end_pos].values

        mfe = (window_highs.max() - entry_px) / entry_px
        mae = (window_lows.min()  - entry_px) / entry_px   # negative = adverse
        mfe_day_idx = int(np.argmax(window_highs))

        records.append(
            {
                "entry_date":    all_dates[entry_pos],
                "entry_px":      entry_px,
                "mfe":           mfe,
                "mae":           mae,
                "mfe_days":      mfe_day_idx,
            }
        )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EXIT OPTIMISATION GRID
# ─────────────────────────────────────────────────────────────────────────────

TIME_STOPS   = [3, 4, 5]
TRAIL_STOPS  = [0.015, 0.02, 0.025]
RSR_EXITS    = [0.5, 1.0]


def simulate_trade(
    price_df: pd.DataFrame,
    rsr_z: pd.Series,
    entry_date: pd.Timestamp,
    time_stop: int,
    trail_stop: float,
    rsr_exit_thresh: float,
) -> dict | None:
    """
    Simulate a single long trade with three exit rules (first-hit):
      1. time_stop  : close after N bars
      2. trail_stop : high-water-mark drawdown triggers exit at next open
      3. rsr_exit   : rsr_z < rsr_exit_thresh triggers exit at next open
    Execution price = open of the bar after the signal.
    """
    opens = price_df["open"]
    highs = price_df["high"]
    all_dates = price_df.index

    if entry_date not in all_dates:
        return None
    sig_pos   = all_dates.get_loc(entry_date)
    entry_pos = sig_pos + 1
    if entry_pos >= len(all_dates):
        return None

    entry_px  = opens.iloc[entry_pos]
    if entry_px <= 0:
        return None

    hwm       = entry_px
    exit_px   = None
    exit_bar  = None

    for bar in range(1, time_stop + 1):
        cur_pos = entry_pos + bar
        if cur_pos >= len(all_dates):
            # hit end of data — exit at last available open
            exit_px  = opens.iloc[len(all_dates) - 1]
            exit_bar = bar
            break

        cur_date = all_dates[cur_pos]

        # update HWM using previous bar's high (no lookahead)
        prev_high = highs.iloc[cur_pos - 1]
        hwm = max(hwm, prev_high)

        # trailing stop: triggered if yesterday's close implied a drawdown
        # we exit at today's open
        trail_dd = (opens.iloc[cur_pos] - hwm) / hwm
        if trail_dd <= -trail_stop:
            exit_px  = opens.iloc[cur_pos]
            exit_bar = bar
            break

        # RSR exit: signal at end of previous bar, execute at today's open
        prev_date = all_dates[cur_pos - 1]
        if prev_date in rsr_z.index and rsr_z.loc[prev_date] < rsr_exit_thresh:
            exit_px  = opens.iloc[cur_pos]
            exit_bar = bar
            break

        # time stop (last iteration)
        if bar == time_stop:
            exit_px  = opens.iloc[cur_pos]
            exit_bar = bar

    if exit_px is None or exit_px <= 0:
        return None

    ret = (exit_px - entry_px) / entry_px
    return {
        "entry_date": all_dates[entry_pos],
        "exit_bar":   exit_bar,
        "entry_px":   entry_px,
        "exit_px":    exit_px,
        "ret":        ret,
    }


def _metrics(rets: np.ndarray) -> dict:
    if len(rets) == 0:
        return {"expectancy": np.nan, "max_drawdown": np.nan, "profit_factor": np.nan, "n_trades": 0}

    expectancy = float(np.mean(rets))

    # max drawdown on cumulative equity curve
    cum = np.cumprod(1.0 + rets)
    rolling_max = np.maximum.accumulate(cum)
    dd_series = cum / rolling_max - 1.0
    max_dd = float(dd_series.min())

    wins  = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = wins / losses if losses > 0 else np.inf

    return {
        "expectancy":    expectancy,
        "max_drawdown":  max_dd,
        "profit_factor": pf,
        "n_trades":      len(rets),
    }


def run_exit_grid(
    price_df: pd.DataFrame,
    rsr_z: pd.Series,
    entry_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows = []
    combos = list(product(TIME_STOPS, TRAIL_STOPS, RSR_EXITS))
    for ts, trail, rsr_ex in combos:
        rets = []
        for ed in entry_dates:
            result = simulate_trade(price_df, rsr_z, ed, ts, trail, rsr_ex)
            if result is not None:
                rets.append(result["ret"])
        m = _metrics(np.array(rets))
        rows.append(
            {
                "time_stop":   ts,
                "trail_stop":  trail,
                "rsr_exit":    rsr_ex,
                **m,
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PORTFOLIO DD CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def exposure_multiplier(dd: float) -> float:
    if dd > -0.05:
        return 1.0
    elif dd > -0.10:
        return 0.7
    elif dd > -0.15:
        return 0.4
    else:
        return 0.0


def apply_dd_control(trade_rets: pd.Series) -> pd.Series:
    """
    Walk through trades in chronological order.
    Equity curve starts at 1.0; each trade is scaled by exposure_multiplier.
    Returns adjusted return series.
    """
    equity   = 1.0
    hwm      = 1.0
    adjusted = []

    for raw_ret in trade_rets:
        dd = equity / hwm - 1.0
        mult = exposure_multiplier(dd)
        adj_ret = raw_ret * mult
        adjusted.append(adj_ret)
        equity *= 1.0 + adj_ret
        hwm = max(hwm, equity)

    return pd.Series(adjusted, index=trade_rets.index)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VALIDATION BLOCK
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    price_df: pd.DataFrame,
    rsr_z: pd.Series,
    fwd_df: pd.DataFrame,
    mfe_df: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION BLOCK")
    print("=" * 60)

    # date alignment
    price_dates = set(price_df.index)
    rsr_dates   = set(rsr_z.index)
    overlap     = price_dates & rsr_dates
    missing_in_price = rsr_dates - price_dates
    missing_in_rsr   = price_dates - rsr_dates
    print(f"\n[DATE ALIGNMENT]")
    print(f"  price rows   : {len(price_df)}")
    print(f"  rsr_z rows   : {len(rsr_z)}")
    print(f"  overlap      : {len(overlap)}")
    print(f"  in rsr, not price : {len(missing_in_price)}")
    print(f"  in price, not rsr : {len(missing_in_rsr)}")

    # max forward return per horizon
    print(f"\n[MAX FORWARD RETURN BY HORIZON]")
    for h in HORIZONS:
        col = f"fwd_{h}d"
        if col in fwd_df.columns:
            mx = fwd_df[col].max()
            print(f"  fwd_{h:>2}d max : {mx:+.4f}  ({mx*100:.2f}%)")

    # suspicious 1-day returns
    suspicious_1d = fwd_df["fwd_1d"][fwd_df["fwd_1d"].abs() > 0.30].dropna()
    print(f"\n[SUSPICIOUS: |fwd_1d| > 30%]  count={len(suspicious_1d)}")
    if len(suspicious_1d) > 0:
        print(suspicious_1d.to_string())

    # MFE stats
    if not mfe_df.empty:
        print(f"\n[MFE STATISTICS]")
        print(f"  mean MFE     : {mfe_df['mfe'].mean():+.4f}")
        print(f"  median MFE   : {mfe_df['mfe'].median():+.4f}")
        print(f"  mean MAE     : {mfe_df['mae'].mean():+.4f}")
        print(f"  mean mfe_days: {mfe_df['mfe_days'].mean():.1f}")

        large_mfe = mfe_df[mfe_df["mfe"] > 0.50]
        print(f"\n[SUSPICIOUS: MFE > 50%]  count={len(large_mfe)}")
        if len(large_mfe) > 0:
            print(large_mfe[["entry_date", "entry_px", "mfe", "mfe_days"]].to_string(index=False))

    # top 5 trades
    if not trade_log.empty and "ret" in trade_log.columns:
        top5 = trade_log.nlargest(5, "ret")[["entry_date", "entry_px", "exit_px", "ret", "exit_bar"]]
        print(f"\n[TOP 5 TRADES BY PnL]")
        print(top5.to_string(index=False))

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5b. VALIDATION AUDIT BLOCK
# ─────────────────────────────────────────────────────────────────────────────

def _root_cause_diagnostics(
    price_df: pd.DataFrame,
    rsr_z: pd.Series,
    fwd_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    flags: list[str],
) -> None:
    """
    Printed only when at least one suspicious flag was triggered.
    Six root-cause checks that help distinguish data errors from real alpha.
    """
    DIV = "-" * 60

    print("\n[ROOT-CAUSE DIAGNOSTICS]")
    print(DIV)

    # ── D1. index alignment ───────────────────────────────────────────────────
    print("\n  D1. INDEX ALIGNMENT")
    price_idx = price_df.index
    rsr_idx   = rsr_z.index
    fwd_idx   = fwd_df.index

    price_rsr_gap = price_idx.difference(rsr_idx)
    rsr_price_gap = rsr_idx.difference(price_idx)
    price_fwd_gap = price_idx.difference(fwd_idx)

    print(f"    price rows          : {len(price_idx)}")
    print(f"    rsr_z rows          : {len(rsr_idx)}")
    print(f"    fwd_df rows         : {len(fwd_idx)}")
    print(f"    in price, not rsr_z : {len(price_rsr_gap)}"
          + (f"  <- MISMATCH" if len(price_rsr_gap) else ""))
    print(f"    in rsr_z, not price : {len(rsr_price_gap)}"
          + (f"  <- MISMATCH" if len(rsr_price_gap) else ""))
    print(f"    in price, not fwd   : {len(price_fwd_gap)}"
          + (f"  <- MISMATCH" if len(price_fwd_gap) else ""))
    if len(price_rsr_gap) > 0:
        print(f"    first 5 price-only  : {list(price_rsr_gap[:5])}")
    if len(rsr_price_gap) > 0:
        print(f"    first 5 rsr_z-only  : {list(rsr_price_gap[:5])}")

    # freq inference
    inferred = pd.infer_freq(price_idx)
    print(f"    inferred price freq : {inferred!r}  "
          + ("(OK)" if inferred in ("B", "C") else "<- NOT BUSINESS-DAILY"))

    # ── D2. duplicated dates ──────────────────────────────────────────────────
    print(f"\n  D2. DUPLICATED DATES")
    dup_price = price_idx[price_idx.duplicated()]
    dup_rsr   = rsr_idx[rsr_idx.duplicated()]
    dup_fwd   = fwd_idx[fwd_idx.duplicated()]
    print(f"    price dups : {len(dup_price)}"
          + (f"  <- DUPLICATES FOUND" if len(dup_price) else ""))
    print(f"    rsr_z dups : {len(dup_rsr)}"
          + (f"  <- DUPLICATES FOUND" if len(dup_rsr) else ""))
    print(f"    fwd_df dups: {len(dup_fwd)}"
          + (f"  <- DUPLICATES FOUND" if len(dup_fwd) else ""))
    if len(dup_price):
        print(f"    first 5 dup price dates: {list(dup_price[:5])}")

    # ── D3. missing OHLC rows ─────────────────────────────────────────────────
    print(f"\n  D3. MISSING OHLC ROWS")
    for col in ["open", "high", "low", "close"]:
        if col not in price_df.columns:
            print(f"    {col:>5} : COLUMN MISSING")
            continue
        n_null = price_df[col].isna().sum()
        n_zero = (price_df[col] == 0).sum()
        n_neg  = (price_df[col] < 0).sum()
        issues = []
        if n_null: issues.append(f"{n_null} NaN")
        if n_zero: issues.append(f"{n_zero} zeros")
        if n_neg:  issues.append(f"{n_neg} negatives")
        status = "  <- " + ", ".join(issues) if issues else ""
        print(f"    {col:>5} : {price_df[col].count()} valid{status}")

    # high < low check
    if {"high", "low"}.issubset(price_df.columns):
        inverted = (price_df["high"] < price_df["low"]).sum()
        print(f"    high < low violations : {inverted}"
              + ("  <- OHLC INVERSION" if inverted else ""))

    # open outside high/low
    if {"open", "high", "low"}.issubset(price_df.columns):
        out_of_range = (
            (price_df["open"] > price_df["high"]) |
            (price_df["open"] < price_df["low"])
        ).sum()
        print(f"    open outside high/low : {out_of_range}"
              + ("  <- DATA ERROR" if out_of_range else ""))

    # ── D4. open-to-open gap distribution ────────────────────────────────────
    print(f"\n  D4. open-TO-open GAP DISTRIBUTION")
    if "open" in price_df.columns:
        o2o = price_df["open"].pct_change().dropna()
        abs_o2o = o2o.abs()
        print(f"    count   : {len(o2o)}")
        print(f"    mean    : {o2o.mean():+.4f} ({o2o.mean()*100:+.2f}%)")
        print(f"    std     : {o2o.std():.4f} ({o2o.std()*100:.2f}%)")
        print(f"    p1/p99  : {np.percentile(o2o,1):+.4f} / {np.percentile(o2o,99):+.4f}")
        for thresh, label in [(0.05, ">5%"), (0.10, ">10%"), (0.20, ">20%")]:
            n = (abs_o2o > thresh).sum()
            print(f"    |gap| {label:>4} : {n:>4} rows"
                  + ("  <- LARGE GAPS PRESENT" if n > 0 and thresh >= 0.10 else ""))

    # ── D5. top 5 largest overnight gaps ─────────────────────────────────────
    print(f"\n  D5. TOP 5 LARGEST OVERNIGHT GAPS (|open[t]/open[t-1] - 1|)")
    if "open" in price_df.columns:
        o2o_abs = price_df["open"].pct_change().abs().dropna()
        top5_gaps = o2o_abs.nlargest(5)
        for date, gap in top5_gaps.items():
            prev_date = price_df.index[price_df.index.get_loc(date) - 1]
            prev_open = price_df.loc[prev_date, "open"]
            curr_open = price_df.loc[date, "open"]
            print(f"    {date.date()}  prev_open={prev_open:.2f}  curr_open={curr_open:.2f}  gap={gap*100:+.2f}%")

    # ── D6. flagged-trade clustering ──────────────────────────────────────────
    print(f"\n  D6. FLAGGED-TRADE CLUSTERING")
    if trade_log.empty or "ret" not in trade_log.columns:
        print("    trade_log empty — skipping.")
    else:
        # identify suspicious trades: return > 15% in exit_bar days
        flagged = trade_log[trade_log["ret"] > 0.15].copy()
        print(f"    trades with ret > 15% : {len(flagged)}")

        if not flagged.empty and "entry_date" in flagged.columns:
            # date clustering: how many fall within 5-day windows of each other
            flagged_sorted = flagged.sort_values("entry_date")
            dates_arr = pd.to_datetime(flagged_sorted["entry_date"]).values
            if len(dates_arr) > 1:
                gaps_days = np.diff(dates_arr).astype("timedelta64[D]").astype(int)
                clustered = (gaps_days <= 5).sum()
                print(f"    consecutive pairs ≤5 biz days apart : {clustered}"
                      + ("  <- DATE CLUSTERING" if clustered > len(flagged) * 0.4 else ""))

            # entry_px clustering: identical entry prices suggest data repeat
            if "entry_px" in flagged.columns:
                dup_px = flagged["entry_px"].duplicated().sum()
                print(f"    duplicated entry_px values : {dup_px}"
                      + ("  <- POSSIBLE DATA REPEAT" if dup_px else ""))

            # show flagged trades
            cols = [c for c in ["entry_date", "exit_bar", "entry_px", "exit_px", "ret"]
                    if c in flagged.columns]
            print(f"    flagged trade detail:")
            print(flagged[cols].to_string(index=False, justify="right"))
        else:
            # no >15% trades; check for systematic win clustering by month
            if not trade_log.empty and "entry_date" in trade_log.columns:
                tl = trade_log.copy()
                tl["month"] = pd.to_datetime(tl["entry_date"]).dt.to_period("M")
                monthly_wr = tl.groupby("month").apply(
                    lambda g: (g["ret"] > 0).mean(), include_groups=False
                )
                hot_months = monthly_wr[monthly_wr > 0.80]
                print(f"    months with win-rate > 80% : {len(hot_months)}"
                      + ("  <- PERIOD CLUSTERING" if len(hot_months) >= 3 else ""))
                if not hot_months.empty:
                    for m, wr in hot_months.items():
                        print(f"      {m}  wr={wr*100:.1f}%")

    print(DIV)


def run_audit(
    fwd_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    price_df: pd.DataFrame | None = None,
    rsr_z: pd.Series | None = None,
) -> None:
    """
    Standalone audit block.  Flags suspicious statistics that indicate
    lookahead bias, data errors, or unrealistic backtest assumptions.
    When flags are triggered, delegates to _root_cause_diagnostics.

    Suspicious thresholds
    ---------------------
    - max 1-day forward return > 15%
    - mean forward return (any horizon) > 3%
    - win rate (any horizon) > 75%
    """
    SEP = "=" * 60
    FLAG = "*** SUSPICIOUS ***"

    print("\n" + SEP)
    print("VALIDATION AUDIT")
    print(SEP)

    # ── 1. mean forward return by horizon ────────────────────────────────────
    print("\n[1] MEAN FORWARD RETURN BY HORIZON")
    print(f"  {'horizon':>8}  {'mean':>9}  {'flag'}")
    for h in HORIZONS:
        col = f"fwd_{h}d"
        if col not in fwd_df.columns:
            continue
        vals = fwd_df[col].dropna()
        mean = vals.mean()
        flag = FLAG if mean > 0.03 else ""
        print(f"  fwd_{h:>2}d    {mean:+.4f} ({mean*100:+.2f}%)  {flag}")

    # ── 2. win rate by horizon ────────────────────────────────────────────────
    print("\n[2] WIN RATE BY HORIZON")
    print(f"  {'horizon':>8}  {'win_rate':>9}  {'n':>6}  {'flag'}")
    for h in HORIZONS:
        col = f"fwd_{h}d"
        if col not in fwd_df.columns:
            continue
        vals = fwd_df[col].dropna()
        if len(vals) == 0:
            continue
        wr   = (vals > 0).sum() / len(vals)
        flag = FLAG if wr > 0.75 else ""
        print(f"  fwd_{h:>2}d    {wr:.4f} ({wr*100:.1f}%)   {len(vals):>6}  {flag}")

    # ── 3. max 1-day forward return ───────────────────────────────────────────
    print("\n[3] MAX 1-DAY FORWARD RETURN")
    if "fwd_1d" in fwd_df.columns:
        v1d  = fwd_df["fwd_1d"].dropna()
        mx   = v1d.max()
        flag = FLAG if mx > 0.15 else ""
        worst = v1d.min()
        print(f"  max  : {mx:+.4f} ({mx*100:+.2f}%)  {flag}")
        print(f"  min  : {worst:+.4f} ({worst*100:+.2f}%)")
        if mx > 0.15:
            top_dates = v1d[v1d > 0.15].sort_values(ascending=False)
            print(f"  rows with >15%: {len(top_dates)}")
            print(top_dates.to_string())

    # ── 4. 99th-percentile forward return ────────────────────────────────────
    print("\n[4] 99th-PERCENTILE FORWARD RETURN BY HORIZON")
    print(f"  {'horizon':>8}  {'p99':>9}  {'p1':>9}")
    for h in HORIZONS:
        col = f"fwd_{h}d"
        if col not in fwd_df.columns:
            continue
        vals = fwd_df[col].dropna()
        if len(vals) < 10:
            continue
        p99 = float(np.percentile(vals, 99))
        p1  = float(np.percentile(vals, 1))
        print(f"  fwd_{h:>2}d    {p99:+.4f} ({p99*100:+.2f}%)  {p1:+.4f} ({p1*100:+.2f}%)")

    # ── 5. average holding days ───────────────────────────────────────────────
    print("\n[5] AVERAGE HOLDING DAYS (from trade log)")
    if trade_log.empty or "exit_bar" not in trade_log.columns:
        print("  trade_log is empty or missing exit_bar column.")
    else:
        avg_hold = trade_log["exit_bar"].mean()
        med_hold = trade_log["exit_bar"].median()
        dist     = trade_log["exit_bar"].value_counts().sort_index()
        print(f"  mean : {avg_hold:.2f} days")
        print(f"  median: {med_hold:.1f} days")
        print(f"  distribution:")
        for days, cnt in dist.items():
            bar = "#" * min(cnt, 40)
            print(f"    {days:>3}d : {cnt:>4}  {bar}")

    # ── 6. top 10 trades by PnL ───────────────────────────────────────────────
    print("\n[6] TOP 10 TRADES BY PnL")
    if trade_log.empty or "ret" not in trade_log.columns:
        print("  trade_log is empty or missing ret column.")
    else:
        cols = [c for c in ["entry_date", "exit_bar", "entry_px", "exit_px", "ret"] if c in trade_log.columns]
        top10 = trade_log.nlargest(10, "ret")[cols]
        # add exit_date if we can derive it
        if "entry_date" in trade_log.columns and "exit_bar" in trade_log.columns:
            top10 = top10.copy()
            top10["exit_date"] = top10.apply(
                lambda r: r["entry_date"] + pd.tseries.offsets.BusinessDay(int(r["exit_bar"])),
                axis=1,
            )
        print(top10.to_string(index=False))

    # ── collect flags ─────────────────────────────────────────────────────────
    flags = []
    if "fwd_1d" in fwd_df.columns:
        mx1d = fwd_df["fwd_1d"].dropna().max()
        if mx1d > 0.15:
            flags.append(f"max 1d return {mx1d*100:.1f}% > 15%")
    for h in HORIZONS:
        col = f"fwd_{h}d"
        if col in fwd_df.columns:
            vals = fwd_df[col].dropna()
            if vals.mean() > 0.03:
                flags.append(f"mean fwd_{h}d {vals.mean()*100:.2f}% > 3%")
            if len(vals) > 0 and (vals > 0).sum() / len(vals) > 0.75:
                flags.append(f"win rate fwd_{h}d {(vals>0).sum()/len(vals)*100:.1f}% > 75%")

    print("\n[AUDIT SUMMARY]")
    if flags:
        for f in flags:
            print(f"  {FLAG}  {f}")
        # root-cause block fires only when flags exist and price data is available
        if price_df is not None and rsr_z is not None:
            _root_cause_diagnostics(price_df, rsr_z, fwd_df, trade_log, flags)
        else:
            print("  (pass price_df and rsr_z to run_audit for root-cause diagnostics)")
    else:
        print("  All checks passed — no suspicious statistics detected.")

    print(SEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ALPHA DECAY (forward return by RSR quintile)
# ─────────────────────────────────────────────────────────────────────────────

def compute_alpha_decay(
    price_df: pd.DataFrame,
    rsr_z: pd.Series,
    fwd_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean forward return by RSR quintile and horizon — detects decay."""
    aligned_rsr = rsr_z.reindex(price_df.index)
    quintile     = pd.qcut(aligned_rsr.dropna(), 5, labels=False, duplicates="drop")
    quintile     = quintile.reindex(price_df.index)

    rows = []
    for q in range(5):
        mask = quintile == q
        row  = {"quintile": q}
        for h in HORIZONS:
            col = f"fwd_{h}d"
            if col in fwd_df.columns:
                row[col] = fwd_df.loc[mask, col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  BUILD TRADE LOG (confirmed exit defaults: time_stop=4, trail_stop=0.025, rsr_exit=1.1)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 7b. ABSOLUTE THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
_DEFAULT_TIME_STOP  = 4      # confirmed 2026-04-12: single-point BT PASS (DD=-11.88%, EXP=+0.83%, PF=2.396)
_DEFAULT_TRAIL_STOP = 0.025  # confirmed 2026-04-12
_DEFAULT_RSR_EXIT   = 1.1    # confirmed 2026-04-12


def _threshold_metrics(rets: np.ndarray, n_days: int) -> dict:
    """Trade-level metrics including CAGR proxy (annualised, 252 biz days)."""
    if len(rets) == 0:
        return {
            "n_trades":      0,
            "win_rate":      np.nan,
            "expectancy":    np.nan,
            "max_drawdown":  np.nan,
            "profit_factor": np.nan,
            "cagr_proxy":    np.nan,
        }

    expectancy = float(np.mean(rets))
    win_rate   = float((rets > 0).sum() / len(rets))

    cum = np.cumprod(1.0 + rets)
    rolling_max = np.maximum.accumulate(cum)
    dd_series   = cum / rolling_max - 1.0
    max_dd      = float(dd_series.min())

    wins   = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf     = float(wins / losses) if losses > 0 else np.inf

    # CAGR proxy: total return annualised over calendar span of data
    years  = max(n_days / 252.0, 1e-6)
    cagr   = float(cum[-1] ** (1.0 / years) - 1.0)

    return {
        "n_trades":      len(rets),
        "win_rate":      win_rate,
        "expectancy":    expectancy,
        "max_drawdown":  max_dd,
        "profit_factor": pf,
        "cagr_proxy":    cagr,
    }


def run_threshold_sweep(
    price_df: pd.DataFrame,
    rsr_z: pd.Series,
    thresholds: list[float] = THRESHOLDS,
    time_stop: int  = _DEFAULT_TIME_STOP,
    trail_stop: float = _DEFAULT_TRAIL_STOP,
    rsr_exit_thresh: float = _DEFAULT_RSR_EXIT,
) -> pd.DataFrame:
    """
    For each threshold t, entry_signal = rsr_z > t.
    Trades are simulated with simulate_trade (next-open execution, no lookahead).
    Returns one row per threshold with six performance metrics.
    """
    n_days = len(price_df)
    rows   = []
    for thresh in thresholds:
        signal_mask  = rsr_z > thresh
        entry_dates  = price_df.index[signal_mask.reindex(price_df.index, fill_value=False)]
        rets = []
        for ed in entry_dates:
            result = simulate_trade(price_df, rsr_z, ed, time_stop, trail_stop, rsr_exit_thresh)
            if result is not None:
                rets.append(result["ret"])
        m = _threshold_metrics(np.array(rets), n_days)
        rows.append({"threshold": thresh, **m})
    return pd.DataFrame(rows)


def build_trade_log(
    price_df: pd.DataFrame,
    rsr_z: pd.Series,
    entry_dates: pd.DatetimeIndex,
    time_stop: int = _DEFAULT_TIME_STOP,
    trail_stop: float = _DEFAULT_TRAIL_STOP,
    rsr_exit_thresh: float = _DEFAULT_RSR_EXIT,
) -> pd.DataFrame:
    records = []
    for ed in entry_dates:
        result = simulate_trade(price_df, rsr_z, ed, time_stop, trail_stop, rsr_exit_thresh)
        if result is not None:
            records.append(result)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "backtests")
    os.makedirs(out_dir, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────────
    price_path = r"C:\ai-trading\cache\ohlcv\5401.T.parquet"

    if not os.path.exists(price_path):
        print(f"[ERROR] Required file not found: {price_path}")
        sys.exit(1)

    print(f"[INFO] Loading price data from {price_path}")
    price_df = pd.read_parquet(price_path)
    price_df.columns = [c.lower() for c in price_df.columns]
    print(f"[INFO] price_df : {len(price_df)} rows  {price_df.index[0].date()} – {price_df.index[-1].date()}")

    # ── build rsr_z from close prices ────────────────────────────────────────
    close   = price_df["close"]
    returns = close.pct_change(20)
    vol     = close.pct_change().rolling(20).std()
    raw     = returns / vol
    rsr_z   = (raw - raw.rolling(60).mean()) / raw.rolling(60).std()
    rsr_z.name = "rsr_z"
    print(f"[INFO] rsr_z    : {rsr_z.notna().sum()} non-NaN rows")

    # ── entry signal: rsr_z > 1.0 (example long signal) ─────────────────────
    # Signal fires at end of bar t; execution at open[t+1]
    signal_mask  = rsr_z > 1.0
    entry_dates  = price_df.index[signal_mask]
    print(f"[INFO] Entry signals : {len(entry_dates)}")

    # ── 1. forward returns ────────────────────────────────────────────────────
    print("[STEP 1] Computing forward returns ...")
    fwd_df = compute_forward_returns(price_df)

    # ── 2. MFE / MAE ─────────────────────────────────────────────────────────
    print("[STEP 2] Computing MFE / MAE ...")
    mfe_df = compute_mfe_mae(price_df, entry_dates, max_hold=20)

    # ── 3. exit grid ─────────────────────────────────────────────────────────
    print("[STEP 3] Running exit optimisation grid ...")
    grid_df = run_exit_grid(price_df, rsr_z, entry_dates)

    # ── 4. trade log (default params) + DD control ───────────────────────────
    print("[STEP 4] Building trade log + applying DD control ...")
    trade_log = build_trade_log(price_df, rsr_z, entry_dates)

    if not trade_log.empty:
        raw_rets   = pd.Series(trade_log["ret"].values, index=trade_log.index)
        adj_rets   = apply_dd_control(raw_rets)
        trade_log["ret_adj"] = adj_rets.values

    # ── 5. validation ─────────────────────────────────────────────────────────
    run_validation(price_df, rsr_z, fwd_df, mfe_df, trade_log)

    # ── 5b. audit ─────────────────────────────────────────────────────────────
    run_audit(fwd_df, trade_log, price_df=price_df, rsr_z=rsr_z)

    # ── 6. alpha decay ────────────────────────────────────────────────────────
    print("[STEP 6] Computing alpha decay by RSR quintile ...")
    alpha_decay_df = compute_alpha_decay(price_df, rsr_z, fwd_df)

    # ── 7b. threshold sweep ───────────────────────────────────────────────────
    print("[STEP 7b] Running absolute threshold sweep ...")
    sweep_df = run_threshold_sweep(price_df, rsr_z)

    sweep_path = os.path.join(out_dir, "threshold_sweep.csv")

    # ── save outputs ──────────────────────────────────────────────────────────
    trades_path  = os.path.join(out_dir, "trades_log.csv")
    grid_path    = os.path.join(out_dir, "grid_results.csv")
    decay_path   = os.path.join(out_dir, "alpha_decay.csv")

    trade_log.to_csv(trades_path, index=False, encoding="utf-8")
    grid_df.to_csv(grid_path,   index=False, encoding="utf-8")
    alpha_decay_df.to_csv(decay_path, index=False, encoding="utf-8")
    sweep_df.to_csv(sweep_path, index=False, encoding="utf-8")

    print(f"\n[SAVED] {trades_path}")
    print(f"[SAVED] {grid_path}")
    print(f"[SAVED] {decay_path}")
    print(f"[SAVED] {sweep_path}")

    # ── 7. final summary ──────────────────────────────────────────────────────
    valid_grid = grid_df.dropna(subset=["expectancy"])
    if not valid_grid.empty:
        best_row = valid_grid.loc[valid_grid["expectancy"].idxmax()]
    else:
        best_row = None

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if best_row is not None:
        print(f"\n[BEST PARAMETER SET]")
        print(f"  time_stop   : {int(best_row['time_stop'])}")
        print(f"  trail_stop  : {best_row['trail_stop']:.2f}")
        print(f"  rsr_exit    : {best_row['rsr_exit']:.1f}")
        print(f"\n[BEST EXPECTANCY]     {best_row['expectancy']:+.4f}  ({best_row['expectancy']*100:.2f}%)")
        print(f"[BEST MAX DD]         {best_row['max_drawdown']:+.4f}  ({best_row['max_drawdown']*100:.2f}%)")
        print(f"[BEST PROFIT FACTOR]  {best_row['profit_factor']:.3f}")
        print(f"[N_TRADES]            {int(best_row['n_trades'])}")
    else:
        print("No valid grid results.")

    if not trade_log.empty and "ret_adj" in trade_log.columns:
        adj = trade_log["ret_adj"].values
        cum  = np.cumprod(1.0 + adj)
        hwm  = np.maximum.accumulate(cum)
        dd   = (cum / hwm - 1.0).min()
        print(f"\n[DD-CONTROLLED MAX DD] {dd:+.4f}  ({dd*100:.2f}%)")
        print(f"[DD-CONTROLLED FINAL EQUITY] {cum[-1]:.4f}")

    print("\n[ALPHA DECAY TABLE]")
    print(alpha_decay_df.to_string(index=False))

    # ── threshold sweep report ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP  (entry_signal = rsr_z > threshold)")
    print("=" * 60)
    fmt = "{:>10}  {:>8}  {:>10}  {:>12}  {:>14}  {:>10}  {:>8}"
    print(fmt.format("threshold", "n_trades", "win_rate",
                     "expectancy", "max_drawdown", "profit_fac", "cagr"))
    for _, row in sweep_df.iterrows():
        pf_str = f"{row['profit_factor']:.3f}" if np.isfinite(row['profit_factor']) else "  inf"
        print(fmt.format(
            f"{row['threshold']:.1f}",
            f"{int(row['n_trades'])}",
            f"{row['win_rate']*100:.1f}%"   if not np.isnan(row['win_rate'])      else "  n/a",
            f"{row['expectancy']*100:+.2f}%" if not np.isnan(row['expectancy'])    else "     n/a",
            f"{row['max_drawdown']*100:+.2f}%" if not np.isnan(row['max_drawdown']) else "       n/a",
            pf_str,
            f"{row['cagr_proxy']*100:+.1f}%" if not np.isnan(row['cagr_proxy'])   else " n/a",
        ))

    # best threshold: max expectancy subject to max_dd >= -0.15
    eligible = sweep_df[sweep_df["max_drawdown"] >= -0.15]
    print()
    if eligible.empty:
        print("[BEST THRESHOLD] No threshold passes max DD <= 15% constraint.")
    else:
        best = eligible.loc[eligible["expectancy"].idxmax()]
        print(f"[BEST THRESHOLD under max DD <= 15%]")
        print(f"  threshold    : {best['threshold']:.1f}")
        print(f"  n_trades     : {int(best['n_trades'])}")
        print(f"  win_rate     : {best['win_rate']*100:.1f}%")
        print(f"  expectancy   : {best['expectancy']*100:+.2f}%")
        print(f"  max_drawdown : {best['max_drawdown']*100:+.2f}%")
        pf_str = f"{best['profit_factor']:.3f}" if np.isfinite(best['profit_factor']) else "inf"
        print(f"  profit_factor: {pf_str}")
        print(f"  cagr_proxy   : {best['cagr_proxy']*100:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
