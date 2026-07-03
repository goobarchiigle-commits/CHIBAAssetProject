"""
backtest/entry_zone_audit.py

研究専用 / 実装禁止 / WF禁止

Phase A: 現行エントリー帯 RSR 65-84 品質監査 (selected=True のみ)
  A1 — 状態細分化マトリクス (5 states)
  A2 — 除外候補監査 (bottom 20% / 10% by fwd60)
  A3 — Oracle 実験 (理論改善上限)

Phase B: RSR 90-94 高RSR分解
  B1 — 成熟度グループ (NEW_BREAKOUT / MATURE / LATE)
  B2 — 4枚目適性 (capacity-blocked 候補の期待値 × 相関)

出力:
  reports/rsr65_80_state_surface.csv
  reports/entry_zone_audit.md

Run:
    cd C:/ai-trading
    python src/backtest/entry_zone_audit.py
"""

from __future__ import annotations

import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, _sector_ok, _execute_buy, Position, _take,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR  = Path("reports")
FULL_START   = "2018-01-01"
FULL_END     = "2025-12-31"
PERIOD_YEARS = 7.0

# ── 状態分類 ──────────────────────────────────────────────────────────
STALL_THR = 0.15   # RSR pts/day  (±0.75pp in 5d → STALL)

ENTRY_STATES = ["EARLY_UP", "STEADY_UP", "STALL", "EARLY_ROLL", "DOWN"]

RSR_BINS_A   = ["65-69", "70-74", "75-79", "80-84"]   # Phase A
RSR_BINS_B   = ["90-91", "92-93", "94"]                # Phase B2
MATURE_GROUPS= ["NEW_BREAKOUT", "MATURE", "LATE"]       # Phase B1


# ─────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def classify_entry_state(slope5: float, slope20: float) -> str:
    if np.isnan(slope5) or np.isnan(slope20):
        return "UNKNOWN"
    if abs(slope5) < STALL_THR:
        return "STALL"
    if slope20 > 0:
        if slope5 > slope20:
            return "EARLY_UP"
        elif slope5 > 0:
            return "STEADY_UP"
        else:
            return "EARLY_ROLL"
    return "DOWN"


def get_rsr_bin_a(rsr: float) -> str:
    if 65 <= rsr < 70: return "65-69"
    if 70 <= rsr < 75: return "70-74"
    if 75 <= rsr < 80: return "75-79"
    if 80 <= rsr < 85: return "80-84"
    return "other"


def get_rsr_bin_b(rsr: float) -> str:
    if 90 <= rsr < 92: return "90-91"
    if 92 <= rsr < 94: return "92-93"
    if 94 <= rsr < 95: return "94"
    return "other"


def days_since_cross(i: int, si: int, rsr_mat: np.ndarray,
                     threshold: float = 70.0, max_lookback: int = 252) -> int:
    if float(rsr_mat[i, si]) < threshold:
        return 0
    for j in range(i - 1, max(i - max_lookback - 1, -1), -1):
        if float(rsr_mat[j, si]) < threshold:
            return i - j - 1
    return min(i, max_lookback)


def atr20_pct(i: int, si: int,
              high_mat: np.ndarray, low_mat: np.ndarray,
              close_mat: np.ndarray, lookback: int = 20) -> float:
    if i < lookback:
        return np.nan
    trs = []
    for k in range(i - lookback + 1, i + 1):
        h = float(high_mat[k, si])
        l = float(low_mat[k, si])
        c_prev = float(close_mat[k - 1, si])
        if h <= 0 or l <= 0:
            continue
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    if not trs:
        return np.nan
    c = float(close_mat[i, si])
    return round(np.mean(trs) / c * 100, 2) if c > 0 else np.nan


def corr_with_existing(i: int, si: int,
                       held_syms: list[str],
                       sym_to_i: dict,
                       close_mat: np.ndarray,
                       lookback: int = 60) -> float:
    if not held_syms or i < lookback + 2:
        return np.nan
    c_raw = np.maximum(close_mat[i - lookback: i + 1, si], 1e-6)
    if len(c_raw) < lookback + 1:
        return np.nan
    ret_cand = np.diff(np.log(c_raw))
    corrs = []
    for h_sym in held_syms:
        h_si = sym_to_i.get(h_sym)
        if h_si is None:
            continue
        h_raw = np.maximum(close_mat[i - lookback: i + 1, h_si], 1e-6)
        if len(h_raw) < lookback + 1:
            continue
        ret_h = np.diff(np.log(h_raw))
        c = np.corrcoef(ret_cand, ret_h)[0, 1]
        if not np.isnan(c):
            corrs.append(float(c))
    return round(float(np.mean(corrs)), 3) if corrs else np.nan


def safe_mean(vals) -> float | None:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return round(float(np.mean(v)), 2) if v else None


def safe_pf(vals) -> float | None:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return None
    pos = sum(x for x in v if x > 0)
    neg = abs(sum(x for x in v if x <= 0))
    return round(pos / max(neg, 1e-6), 2) if neg > 0 else None


def hit_rate(vals) -> float | None:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return round(sum(1 for x in v if x > 0) / len(v) * 100, 1) if v else None


def pct_fmt(v, decimals=2) -> str:
    return f"{v:+.{decimals}f}%" if v is not None and not np.isnan(v) else "—"


def fmt(v, decimals=2) -> str:
    return f"{v:.{decimals}f}" if v is not None and not np.isnan(v) else "—"


MIN_N = 5


# ─────────────────────────────────────────────────────────────────────
#  SIGNAL COLLECTOR
# ─────────────────────────────────────────────────────────────────────

def collect_signals(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    start: str, end: str,
) -> list[dict]:
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)
    MAX_SINGLE_W = float(cfg.portfolio.max_single_weight)

    shock_market_thr = -0.05
    shock_sym_thr    = -0.08

    rc            = getattr(cfg, "risk_controls", None)
    MAX_SECTOR_W  = float(rc.sector_cap) if rc else 0.25
    gross_enabled = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gross_normal  = float(getattr(rc, "gross_cap_normal",       1.0)) if rc else 1.0
    gross_dd5     = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    gross_dd8     = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())

    strats = {}
    for sym, sector in trade_syms.items():
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        rule  = SECTOR_STRATEGY.get(sector, "fujiko")
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(
                min_rsr=cfg.fujiko.min_rsr, turtle_exit=cfg.fujiko.turtle_exit,
                rsr_series=rsr_s, min_sepa=cfg.fujiko.min_sepa,
                mom_period=cfg.fujiko.mom_period, turtle_entry=cfg.fujiko.turtle_entry,
                use_turtle_entry=cfg.fujiko.use_turtle_entry,
            )

    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    ts = pd.Timestamp(start); te = pd.Timestamp(end)
    common_dates = common_dates[(common_dates >= ts) & (common_dates <= te)]
    n_dates  = len(common_dates)
    n_syms   = len(active_syms)
    sym_to_i = {s: idx for idx, s in enumerate(active_syms)}

    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    high_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    low_mat   = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    sig_mat   = np.zeros((n_dates, n_syms), dtype=np.int8)
    sig_ready = np.zeros(n_syms, dtype=bool)

    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0):
            continue
        open_mat[:, si]  = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]
        high_mat[:, si]  = df_src["High"].to_numpy(dtype=np.float32)[row_idx]
        low_mat[:, si]   = df_src["Low"].to_numpy(dtype=np.float32)[row_idx]
        strat = strats[sym]
        if hasattr(strat, "precompute_signals"):
            required = 252 + getattr(strat, "mom_period", 21) + 2
            if len(df_src) >= required:
                sig_series = strat.precompute_signals(df_src)
                sig_mat[:, si] = sig_series.to_numpy(dtype=np.int8)[row_idx]
                sig_ready[si]  = True

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0,
    )
    sym_active_mat = None
    if sym_active_df is not None:
        sym_active_mat = _take(sym_active_df, common_dates, active_syms,
                               dtype=np.float32, fill_value=1.0)

    mkt_ret_arr = np.zeros(n_dates, dtype=np.float32)
    topix_ret20 = topix_ret60 = bear_arr = None
    if topix_close is not None:
        tr          = topix_close.pct_change()
        mkt_ret_arr = _take(tr, common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret20 = _take(topix_close.pct_change(20), common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret60 = _take(topix_close.pct_change(60), common_dates, dtype=np.float32, fill_value=0.0)
        ma200  = topix_close.rolling(200, min_periods=100).mean()
        bear_s = (topix_close < ma200).reindex(pd.DatetimeIndex(common_dates), method="ffill").fillna(False)
        bear_arr = bear_s.values.astype(bool)

    # ── Portfolio state ───────────────────────────────────────────────
    cash        = float(capital)
    positions: dict[str, Position] = {}
    peak_equity = float(capital)
    cb_active   = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    trades: list[dict] = []

    # hold_days tracking: sym → (entry_idx, signal_record_ref)
    entry_to_signal: dict[str, dict] = {}

    all_signals: list[dict] = []

    for i, date in enumerate(common_dates):
        if i < 21:
            continue

        invested = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
                       for s, pos in positions.items())
        cur_equity = cash + invested
        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        if not cb_active:
            if dd <= -max_dd_limit:
                cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0
        cb_scale = CB_SCALE if cb_active else 1.0

        gross_cap = gross_normal
        if gross_enabled and topix_ret20 is not None:
            r20 = float(topix_ret20[i]); r60 = float(topix_ret60[i])
            if r20 < -0.05:
                gross_cap = gross_dd5
            elif r60 < -0.08:
                gross_cap = gross_dd8

        is_bear     = bool(bear_arr[i]) if bear_arr is not None else False
        sec_cap_eff = 0.18 if is_bear else MAX_SECTOR_W

        mkt_shock   = float(mkt_ret_arr[i]) <= shock_market_thr
        sell_sigs: list[tuple] = []
        buy_cands: list[tuple] = []

        for sym in active_syms:
            si_sym      = sym_to_i[sym]
            is_holding  = sym in positions
            hold_idx    = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val     = float(rsr_mat[i, si_sym])
            close_t     = float(close_mat[i, si_sym])

            if mkt_shock:
                if not is_holding:
                    continue
                if i > 0:
                    prev_c = float(close_mat[i - 1, si_sym])
                    if prev_c > 0 and (close_t / prev_c - 1.0) <= shock_sym_thr:
                        sell_sigs.append((sym, "MARKET_SHOCK_EXIT")); continue
            if mkt_shock and not is_holding:
                continue

            if is_holding and max_hold is not None and hold_idx > max_hold:
                sell_sigs.append((sym, "TIME_STOP")); continue
            if is_holding and rsr_val < rsr_exit_thr and hold_idx >= min_hold:
                sell_sigs.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si_sym]) if sig_ready[si_sym] else 0
            if sig == -1 and is_holding and hold_idx >= min_hold:
                sell_sigs.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not is_holding:
                if i < reentry_ban.get(sym, -1):
                    continue
                if sym_active_mat is not None:
                    if float(sym_active_mat[i, si_sym]) < 0.5:
                        continue
                buy_cands.append((rsr_val, sym))

        if i + 1 >= n_dates:
            break
        next_i = i + 1

        # ── Sell processing ──────────────────────────────────────────
        for sym, reason in sell_sigs:
            if sym not in positions:
                continue
            pos     = positions[sym]
            sell_px = float(open_mat[next_i, sym_to_i[sym]])
            proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
            pnl     = (sell_px - pos.entry_price) * pos.qty
            cash   += proceeds
            hold_d  = i - pos.entry_idx
            # Update hold_days in the original signal record
            if sym in entry_to_signal:
                entry_to_signal[sym]["hold_days"] = hold_d
                del entry_to_signal[sym]
            trades.append({"side": "SELL", "symbol": sym, "pnl": pnl})
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if not buy_cands:
            continue

        buy_cands.sort(key=lambda x: -x[0])

        # Snapshot state AFTER sells
        n_pos_now   = len(positions)
        held_syms   = list(positions.keys())

        for rank_0, (rsr_val, sym) in enumerate(buy_cands):
            entry_rank  = rank_0 + 1
            si_sym      = sym_to_i[sym]

            # ── RSR features ─────────────────────────────────────────
            rsr_t   = float(rsr_mat[i, si_sym])
            rsr_t5  = float(rsr_mat[i-5,  si_sym]) if i >= 5  else np.nan
            rsr_t10 = float(rsr_mat[i-10, si_sym]) if i >= 10 else np.nan
            rsr_t20 = float(rsr_mat[i-20, si_sym]) if i >= 20 else np.nan

            slope5  = (rsr_t - rsr_t5)  / 5  if not np.isnan(rsr_t5)  else np.nan
            slope20 = (rsr_t - rsr_t20) / 20 if not np.isnan(rsr_t20) else np.nan
            accel   = (slope5 - slope20) if not (np.isnan(slope5) or np.isnan(slope20)) else np.nan

            state     = classify_entry_state(slope5, slope20)
            rsr_bin_a = get_rsr_bin_a(rsr_t)
            rsr_bin_b = get_rsr_bin_b(rsr_t)
            sector    = trade_syms.get(sym, "不明")
            strat_nm  = SECTOR_STRATEGY.get(sector, "fujiko")

            atr_pct   = atr20_pct(i, si_sym, high_mat, low_mat, close_mat)
            d_cross70 = days_since_cross(i, si_sym, rsr_mat, threshold=70.0)
            d_cross90 = days_since_cross(i, si_sym, rsr_mat, threshold=90.0)
            peak20    = float(np.nanmax(rsr_mat[i-20:i+1, si_sym]))
            drawdown  = rsr_t - peak20

            sig_record: dict = {
                "date":         str(date.date()),
                "ticker":       sym,
                "sector":       sector,
                "strategy":     strat_nm,
                "rsr_t":        round(rsr_t, 1),
                "slope5":       round(slope5, 4)  if not np.isnan(slope5)  else None,
                "slope20":      round(slope20, 4) if not np.isnan(slope20) else None,
                "accel":        round(accel, 4)   if not np.isnan(accel)   else None,
                "state":        state,
                "rsr_bin_a":    rsr_bin_a,
                "rsr_bin_b":    rsr_bin_b,
                "atr20_pct":    atr_pct,
                "days_cross70": d_cross70,
                "days_cross90": d_cross90,
                "drawdown_rsr": round(drawdown, 1),
                "entry_rank":   entry_rank,
                "n_cands":      len(buy_cands),
                "positions_held": n_pos_now,
                "held_syms":    list(held_syms),
                "selected":     False,
                "skip_reason":  "CAPACITY",
                "hold_days":    None,
                "entry_idx":    next_i,
                "fwd20":        None,
                "fwd60":        None,
                "fwd120":       None,
                "corr_existing": None,
            }

            # Correlation with existing positions
            if n_pos_now > 0:
                sig_record["corr_existing"] = corr_with_existing(
                    i, si_sym, held_syms, sym_to_i, close_mat
                )

            # ── Portfolio execution ───────────────────────────────────
            if cb_active:
                sig_record["skip_reason"] = "CB_ACTIVE"
                all_signals.append(sig_record)
                continue

            open_slots = max_pos - len(positions)
            if open_slots <= 0:
                sig_record["skip_reason"] = "CAPACITY"
                all_signals.append(sig_record)
                continue

            buy_px = float(open_mat[next_i, si_sym])
            if buy_px <= 0:
                sig_record["skip_reason"] = "NO_PRICE"
                all_signals.append(sig_record)
                continue
            if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                sig_record["skip_reason"] = "SECTOR"
                all_signals.append(sig_record)
                continue
            if gross_enabled:
                cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in positions.values()) / max(1.0, capital)
                if cur_gross + buy_px * LOT / max(1.0, capital) > gross_cap:
                    sig_record["skip_reason"] = "GROSS_CAP"
                    all_signals.append(sig_record)
                    continue

            n_rem  = sum(1 for _, s in buy_cands if s not in positions)
            alloc  = (cash / max(1, min(open_slots, n_rem))) * cb_scale
            alloc  = min(alloc, capital * MAX_SINGLE_W)
            qty    = int(alloc / buy_px / LOT) * LOT
            if qty <= 0:
                sig_record["skip_reason"] = "LOT_ZERO"
                all_signals.append(sig_record)
                continue

            _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
            cash -= qty * buy_px * (1 + COST_ONE_WAY)

            sig_record["selected"]    = True
            sig_record["skip_reason"] = None
            all_signals.append(sig_record)
            entry_to_signal[sym] = sig_record

    # ── Post-hoc: forward returns ──────────────────────────────────────
    for rec in all_signals:
        ei  = rec["entry_idx"]
        sym = rec["ticker"]
        if sym not in sym_to_i:
            continue
        si_sym = sym_to_i[sym]
        base   = float(close_mat[ei, si_sym]) if ei < n_dates and close_mat[ei, si_sym] > 0 else 0
        if base <= 0:
            continue
        for h, key in [(20, "fwd20"), (60, "fwd60"), (120, "fwd120")]:
            fi = ei + h
            if fi < n_dates and close_mat[fi, si_sym] > 0:
                rec[key] = round((float(close_mat[fi, si_sym]) / base - 1) * 100, 2)

    return all_signals


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def build_state_surface(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rsr_bin in RSR_BINS_A:
        for state in ENTRY_STATES:
            cell = df[(df["rsr_bin_a"] == rsr_bin) & (df["state"] == state)]
            f20  = cell["fwd20"].dropna().tolist()
            f60  = cell["fwd60"].dropna().tolist()
            f120 = cell["fwd120"].dropna().tolist()
            rows.append({
                "rsr_bin": rsr_bin, "state": state,
                "count":    len(cell),
                "avg_fwd20": safe_mean(f20),
                "avg_fwd60": safe_mean(f60),
                "avg_fwd120":safe_mean(f120),
                "pf":        safe_pf(f60),
                "hit_rate":  hit_rate(f60),
            })
    return pd.DataFrame(rows)


def compute_group_stats(f60_list) -> dict:
    if not f60_list:
        return {"n": 0, "avg": None, "pf": None, "hr": None}
    return {
        "n":   len(f60_list),
        "avg": safe_mean(f60_list),
        "pf":  safe_pf(f60_list),
        "hr":  hit_rate(f60_list),
    }


def oracle_delta(df: pd.DataFrame, pct: float = 0.10) -> dict:
    """
    Oracle 実験: bottom pct% を除いた場合の ΔCAGR / ΔPF 推定。
    fwd60 ベース。
    """
    f60_all = df["fwd60"].dropna().tolist()
    if not f60_all:
        return {}
    n_remove = max(1, int(len(f60_all) * pct))
    f60_sorted  = sorted(f60_all)
    f60_oracle  = f60_sorted[n_remove:]

    baseline_avg = np.mean(f60_all)
    oracle_avg   = np.mean(f60_oracle) if f60_oracle else baseline_avg

    n_total    = len(f60_all)
    n_per_year = n_total / PERIOD_YEARS
    cap_weight = 1.0 / 3.0   # max_pos=3 → each trade uses ~33%

    # ΔCAGR estimate (rough):
    # Before: N/yr trades × baseline_avg % × cap_weight
    # After:  N/yr×(1-pct) trades × oracle_avg % × cap_weight
    cagr_base   = n_per_year       * (baseline_avg / 100) * cap_weight
    cagr_oracle = n_per_year * (1 - pct) * (oracle_avg / 100) * cap_weight
    delta_cagr  = (cagr_oracle - cagr_base) * 100  # in pp

    # Worst trades (removed):
    removed  = f60_sorted[:n_remove]
    avg_loss = np.mean(removed) if removed else 0
    delta_dd = round(-avg_loss * cap_weight, 2)   # rough: fewer bad trades → less DD

    return {
        "n_all":        n_total,
        "n_removed":    n_remove,
        "baseline_avg": round(baseline_avg, 2),
        "oracle_avg":   round(oracle_avg, 2),
        "delta_avg":    round(oracle_avg - baseline_avg, 2),
        "baseline_pf":  safe_pf(f60_all),
        "oracle_pf":    safe_pf(f60_oracle),
        "baseline_hr":  hit_rate(f60_all),
        "oracle_hr":    hit_rate(f60_oracle),
        "delta_cagr_pp":round(delta_cagr, 2),
        "delta_dd_pp":  delta_dd,
        "avg_removed":  round(avg_loss, 2),
    }


# ─────────────────────────────────────────────────────────────────────
#  REPORT WRITER
# ─────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, surface: pd.DataFrame,
                 output_path: Path) -> str:
    L = []; w = L.append

    w("# エントリー帯品質監査: RSR 65-84 × RSR 90-94")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  期間: 2018-2025  |  研究専用 / 実装禁止 / WF禁止")

    sel = df[df["selected"] == True].copy()
    w(f"\n全シグナル: {len(df):,}件  |  selected={len(sel):,}件")

    # ── Phase A1: 状態細分化マトリクス ────────────────────────────────
    w("\n---\n## Phase A1: 状態細分化マトリクス (RSR 65-84, selected のみ)\n")

    sel_a = sel[sel["rsr_bin_a"].isin(RSR_BINS_A)].copy()
    w(f"RSR 65-84 selected 件数: **{len(sel_a)}件**\n")

    # avg_fwd60
    w("### avg_fwd60 (%)\n")
    w("| RSR bin |" + "".join(f" {s} |" for s in ENTRY_STATES))
    w("|---|" + "---|" * len(ENTRY_STATES))
    for rsr_bin in RSR_BINS_A:
        row = f"| **{rsr_bin}** |"
        for state in ENTRY_STATES:
            cr = surface[(surface["rsr_bin"] == rsr_bin) & (surface["state"] == state)]
            if len(cr) == 0:
                row += " — |"; continue
            cr = cr.iloc[0]
            n  = cr["count"]; v = cr["avg_fwd60"]
            row += (f" {v:+.1f}% (N={n}) |" if v is not None and n >= MIN_N else f" (N={n}) |")
        w(row)

    # hit_rate
    w("\n### hit_rate (fwd60>0 %)\n")
    w("| RSR bin |" + "".join(f" {s} |" for s in ENTRY_STATES))
    w("|---|" + "---|" * len(ENTRY_STATES))
    for rsr_bin in RSR_BINS_A:
        row = f"| **{rsr_bin}** |"
        for state in ENTRY_STATES:
            cr = surface[(surface["rsr_bin"] == rsr_bin) & (surface["state"] == state)]
            if len(cr) == 0:
                row += " — |"; continue
            cr = cr.iloc[0]
            n = cr["count"]; v = cr["hit_rate"]
            row += (f" {v:.0f}% |" if v is not None and n >= MIN_N else f" — |")
        w(row)

    # PF
    w("\n### Profit Factor\n")
    w("| RSR bin |" + "".join(f" {s} |" for s in ENTRY_STATES))
    w("|---|" + "---|" * len(ENTRY_STATES))
    for rsr_bin in RSR_BINS_A:
        row = f"| **{rsr_bin}** |"
        for state in ENTRY_STATES:
            cr = surface[(surface["rsr_bin"] == rsr_bin) & (surface["state"] == state)]
            if len(cr) == 0:
                row += " — |"; continue
            cr = cr.iloc[0]
            n = cr["count"]; v = cr["pf"]
            row += (f" {v:.2f} |" if v is not None and n >= MIN_N else f" — |")
        w(row)

    # 追加特徴量サマリ
    w("\n### 追加特徴量サマリ (RSR 65-84 selected)\n")
    w("| 特徴量 | 平均 | 中央値 | 備考 |")
    w("|---|---|---|---|")
    def stat_row(label, col, note=""):
        v = sel_a[col].dropna()
        if len(v) == 0:
            w(f"| {label} | — | — | {note} |")
        else:
            w(f"| {label} | {v.mean():.2f} | {v.median():.2f} | {note} |")
    stat_row("RSR_slope_5",       "slope5",      "RSR pts/day")
    stat_row("RSR_slope_20",      "slope20",     "RSR pts/day")
    stat_row("RSR_accel",         "accel",       "slope5−slope20")
    stat_row("ATR20_pct",         "atr20_pct",   "% of price")
    stat_row("days_since_RSR_cross70", "days_cross70", "日")
    stat_row("entry_rank",        "entry_rank",  "1=最高RSR候補")
    stat_row("positions_held",    "positions_held", "購入前ポジ数")

    # ── Phase A2: 除外候補監査 ─────────────────────────────────────────
    w("\n---\n## Phase A2: 除外候補監査\n")

    for pct_label, qtile in [("下位20%", 0.20), ("下位10%", 0.10)]:
        f60_all = sel_a["fwd60"].dropna()
        if len(f60_all) < MIN_N:
            w(f"\n### {pct_label}: データ不足\n")
            continue
        threshold = f60_all.quantile(qtile)
        bottom    = sel_a[sel_a["fwd60"] <= threshold].copy()
        rest      = sel_a[sel_a["fwd60"] > threshold].copy()

        w(f"\n### {pct_label} 除外候補 (N={len(bottom)}, fwd60 ≤ {threshold:.1f}%)\n")
        w(f"残存 N={len(rest)}\n")

        # State / RSR 分布
        w("**除外候補の State 分布:**\n")
        w("| State | N | % | avg_fwd60 |")
        w("|---|---|---|---|")
        for state in ENTRY_STATES:
            sub = bottom[bottom["state"] == state]
            if len(sub) == 0: continue
            f60_s = sub["fwd60"].dropna().tolist()
            w(f"| {state} | {len(sub)} | {len(sub)/max(len(bottom),1)*100:.1f}% "
              f"| {safe_mean(f60_s) or '—'} |")

        # RSR bin 分布
        w("\n**除外候補の RSR bin 分布:**\n")
        w("| RSR bin | N | % |")
        w("|---|---|---|")
        for b in RSR_BINS_A:
            sub = bottom[bottom["rsr_bin_a"] == b]
            if len(sub) == 0: continue
            w(f"| {b} | {len(sub)} | {len(sub)/max(len(bottom),1)*100:.1f}% |")

        # strategy 分布
        w("\n**除外候補の Strategy 分布:**\n")
        strat_cnt = bottom["strategy"].value_counts()
        w("| Strategy | N | % |")
        w("|---|---|---|")
        for s, cnt in strat_cnt.items():
            w(f"| {s} | {cnt} | {cnt/max(len(bottom),1)*100:.1f}% |")

        # vol・rank サマリ
        w("\n**除外候補 vs 残存 の特徴量比較:**\n")
        w("| 特徴量 | 除外候補 avg | 残存 avg |")
        w("|---|---|---|")
        for feat in ["rsr_t", "atr20_pct", "days_cross70", "entry_rank", "positions_held"]:
            b_v = bottom[feat].dropna()
            r_v = rest[feat].dropna()
            b_avg = f"{b_v.mean():.2f}" if len(b_v) else "—"
            r_avg = f"{r_v.mean():.2f}" if len(r_v) else "—"
            w(f"| {feat} | {b_avg} | {r_avg} |")

    # ── Phase A3: Oracle 実験 ─────────────────────────────────────────
    w("\n---\n## Phase A3: Oracle 実験 (理論改善上限)\n")
    w("_現行ルール変更なし。BOTTOM10%除外のみの仮想改善。_\n")

    for pct_label, pct in [("BOTTOM10%", 0.10), ("BOTTOM20%", 0.20)]:
        r = oracle_delta(sel_a, pct=pct)
        if not r:
            w(f"### {pct_label}: データ不足\n")
            continue
        w(f"### {pct_label} 除外 Oracle\n")
        w(f"| 指標 | ベースライン | Oracle | Delta |")
        w(f"|---|---|---|---|")
        w(f"| N | {r['n_all']} | {r['n_all'] - r['n_removed']} | -{r['n_removed']} |")
        w(f"| avg_fwd60 | {r['baseline_avg']:+.2f}% | {r['oracle_avg']:+.2f}% | {r['delta_avg']:+.2f}pp |")
        w(f"| PF | {r['baseline_pf'] or '—':.2f} | {r['oracle_pf'] or '—':.2f} | — |")
        w(f"| hit_rate | {r['baseline_hr']:.1f}% | {r['oracle_hr']:.1f}% | — |")
        w(f"| 除外平均fwd60 | — | {r['avg_removed']:+.2f}% | — |")
        w(f"\n**推定 ΔCAGR ≈ {r['delta_cagr_pp']:+.2f}pp** (粗い上限。資本1/3使用×回転率仮定)\n")
        w(f"**推定 ΔMaxDD ≈ {r['delta_dd_pp']:+.2f}pp** (除外損失トレード分)\n")

    # ── Phase B1: 成熟度グループ ──────────────────────────────────────
    w("\n---\n## Phase B1: RSR 90-94 成熟度グループ\n")

    df_b = df[df["rsr_t"].between(90, 95)].copy()
    w(f"RSR 90-94 全シグナル: {len(df_b)}件  "
      f"(selected={df_b['selected'].sum()}, not_selected={(~df_b['selected']).sum()})\n")

    def maturity_group(d: int) -> str:
        if d <= 5:  return "NEW_BREAKOUT"
        if d <= 15: return "MATURE"
        return "LATE"

    df_b["maturity"] = df_b["days_cross90"].apply(maturity_group)

    w("| Group | N | avg_fwd20 | avg_fwd60 | PF | hit_rate | avg_RSR |")
    w("|---|---|---|---|---|---|---|")
    for grp in MATURE_GROUPS:
        sub = df_b[df_b["maturity"] == grp]
        f20 = sub["fwd20"].dropna().tolist()
        f60 = sub["fwd60"].dropna().tolist()
        n   = len(f60)
        if n < MIN_N:
            w(f"| {grp} | {len(sub)} | — | — | — | — | — |")
            continue
        w(f"| {grp} | {len(sub)} "
          f"| {safe_mean(f20) or '—':>+.2f}% "
          f"| {safe_mean(f60):+.2f}% "
          f"| {safe_pf(f60) or '—':.2f} "
          f"| {hit_rate(f60):.1f}% "
          f"| {sub['rsr_t'].mean():.1f} |")

    # by state within 90-94
    w("\n**状態 × 成熟度**\n")
    w("| State | NEW_BREAKOUT avg_fwd60 | MATURE avg_fwd60 | LATE avg_fwd60 |")
    w("|---|---|---|---|")
    for state in ENTRY_STATES:
        row = f"| {state} |"
        for grp in MATURE_GROUPS:
            sub = df_b[(df_b["state"] == state) & (df_b["maturity"] == grp)]
            f60 = sub["fwd60"].dropna().tolist()
            row += (f" {safe_mean(f60):+.2f}%(N={len(f60)}) |"
                    if f60 and len(f60) >= MIN_N else f" (N={len(f60)}) |")
        w(row)

    # ── Phase B2: 4枚目適性 ──────────────────────────────────────────
    w("\n---\n## Phase B2: RSR 90-94 4枚目適性 (仮想比較)\n")
    w("_対象: 既存3ポジ保有中で CAPACITY blocked された RSR 90-94 候補のみ_\n")

    cap_b = df_b[(df_b["skip_reason"] == "CAPACITY") & df_b["rsr_t"].between(90, 95)].copy()
    w(f"Capacity-blocked RSR 90-94 候補: **{len(cap_b)}件**\n")

    w("| RSR sub | N | avg_fwd60 | PF | hit_rate | avg_corr_existing | avg_ATR% |")
    w("|---|---|---|---|---|---|---|")
    for b in RSR_BINS_B:
        sub = cap_b[cap_b["rsr_bin_b"] == b]
        f60 = sub["fwd60"].dropna().tolist()
        corrs = sub["corr_existing"].dropna().tolist()
        atrs  = sub["atr20_pct"].dropna().tolist()
        n     = len(f60)
        if n < MIN_N:
            w(f"| {b} | {len(sub)} | — | — | — | — | — |")
            continue
        w(f"| {b} | {len(sub)} "
          f"| {safe_mean(f60):+.2f}% "
          f"| {safe_pf(f60) or '—':.2f} "
          f"| {hit_rate(f60):.1f}% "
          f"| {safe_mean(corrs) or '—'} "
          f"| {safe_mean(atrs) or '—'} |")

    # B2 vs selected RSR 90-94
    sel_b = df_b[df_b["selected"] == True]
    cap_b_all = cap_b
    w("\n**選択済み vs キャパシティブロック 比較 (全90-94)**\n")
    w("| グループ | N | avg_fwd60 | PF | hit_rate |")
    w("|---|---|---|---|---|")
    for label, sub in [("selected", sel_b), ("CAPACITY blocked", cap_b_all)]:
        f60 = sub["fwd60"].dropna().tolist()
        n   = len(f60)
        if n < MIN_N:
            w(f"| {label} | {n} | — | — | — |")
        else:
            w(f"| {label} | {n} | {safe_mean(f60):+.2f}% | {safe_pf(f60):.2f} | {hit_rate(f60):.1f}% |")

    # ── 現行戦略エントリー帯まとめ ────────────────────────────────────
    w("\n---\n## 総括: 現行エントリー帯の強み・弱み\n")

    # State で最良・最悪
    state_stats = []
    for state in ENTRY_STATES:
        sub = sel_a[sel_a["state"] == state]
        f60 = sub["fwd60"].dropna().tolist()
        if len(f60) >= MIN_N:
            state_stats.append((state, safe_mean(f60), safe_pf(f60), hit_rate(f60), len(f60)))
    state_stats.sort(key=lambda x: -x[1])

    w("| State | N | avg_fwd60 | PF | hit_rate | 評価 |")
    w("|---|---|---|---|---|---|")
    for state, avg, pf_v, hr, n in state_stats:
        mark = "★" if avg >= 10 and pf_v is not None and pf_v >= 4 else ("⚠" if avg < 3 else "")
        w(f"| {state} | {n} | {avg:+.2f}% | {pf_v:.2f} | {hr:.1f}% | {mark} |")

    # ── 最終回答 ──────────────────────────────────────────────────────
    w("\n---\n## 最終回答\n")

    # Determine the answer
    # Compute key metrics
    a_oracle = oracle_delta(sel_a, pct=0.10)
    delta_cagr_10 = a_oracle.get("delta_cagr_pp", 0)
    delta_avg_fwd  = a_oracle.get("delta_avg", 0)

    # B: RSR 90-94 capacity-blocked quality
    cap_b_f60 = cap_b_all["fwd60"].dropna().tolist()
    cap_b_avg = safe_mean(cap_b_f60) or 0
    cap_b_pf  = safe_pf(cap_b_f60) or 0

    # A1 best/worst states
    best_state  = state_stats[0] if state_stats else ("—", 0, 0, 0, 0)
    worst_state = state_stats[-1] if len(state_stats) > 1 else ("—", 0, 0, 0, 0)

    state_spread = best_state[1] - worst_state[1]

    w("| 判定軸 | 値 | 基準 |")
    w("|---|---|---|")
    w(f"| BOTTOM10% ΔCAGR推定 | {delta_cagr_10:+.2f}pp | ≥+0.5pp → 改善研究価値あり |")
    w(f"| A1 State スプレッド | {state_spread:.1f}pp | ≥5pp → フィルタ価値あり |")
    w(f"| RSR90-94 blocked avg_fwd60 | {cap_b_avg:+.2f}% | >10% → 追加研究価値あり |")
    w(f"| RSR90-94 blocked PF | {cap_b_pf:.2f} | >3.0 → 追加研究価値あり |")
    w(f"| 最良状態 ({best_state[0]}) avg_fwd60 | {best_state[1]:+.2f}% | |")
    w(f"| 最悪状態 ({worst_state[0]}) avg_fwd60 | {worst_state[1]:+.2f}% | |")

    w("\n### 回答選択肢\n")
    w("```")
    w("A: 65-80改善を先に実装研究")
    w("B: 90-94追加研究継続")
    w("C: 両方価値薄")
    w("D: 改善余地<+0.5ppで研究終了")
    w("```\n")

    # Decision logic
    a_val = delta_cagr_10 >= 0.5 and state_spread >= 5
    b_val = cap_b_avg >= 10 and cap_b_pf >= 3.0 and len(cap_b_f60) >= MIN_N
    both_thin = delta_cagr_10 < 0.5 and cap_b_avg < 5

    if delta_cagr_10 < 0.5 and state_spread < 5 and cap_b_avg < 10:
        if delta_cagr_10 < 0.5:
            answer = "D"
            reason = (f"BOTTOM10%除外ΔCAGR={delta_cagr_10:+.2f}pp < 0.5pp 基準。"
                      f"状態スプレッド={state_spread:.1f}pp。"
                      "実装研究に値する改善余地が確認できない。")
        else:
            answer = "C"
            reason = "A1改善余地もB追加研究価値も閾値未満。両方価値薄。"
    elif a_val and not b_val:
        answer = "A"
        reason = (f"A1 state スプレッド={state_spread:.1f}pp、"
                  f"BOTTOM10% Δavg={delta_avg_fwd:+.2f}%。"
                  "RSR65-80フィルタリング改善を先行研究。")
    elif b_val and not a_val:
        answer = "B"
        reason = (f"RSR90-94 blocked候補: avg_fwd60={cap_b_avg:+.2f}%, PF={cap_b_pf:.2f}。"
                  "現行エントリー帯改善余地が薄い一方、高RSR帯に独立価値あり。")
    elif a_val and b_val:
        # Both have value; choose higher priority
        if delta_cagr_10 >= cap_b_avg / 30:  # heuristic comparison
            answer = "A"
            reason = (f"A・B両方に研究価値あるが、A の ΔCAGR推定={delta_cagr_10:+.2f}pp が優先。"
                      "現行65-80帯の状態フィルタを先行。")
        else:
            answer = "B"
            reason = (f"A・B両方に研究価値あるが、B の90-94高期待値(avg={cap_b_avg:+.2f}%)を優先。")
    else:
        answer = "D"
        reason = "いずれの閾値も満たさない。研究終了。"

    w(f"### 最終回答: **{answer}**\n")
    w(f"根拠: {reason}\n")

    # Implementation suggestion
    if answer == "A":
        w("**次ステップ候補 (要 ASK_FIRST + WF検証):**")
        w(f"- EARLY_ROLL / DOWN 状態の RSR 75-79 エントリーを slope20/slope5 フィルタで除外")
        w(f"- 最悪状態 {worst_state[0]} は avg_fwd60={worst_state[1]:+.2f}% → スキップ候補")
    elif answer == "B":
        w("**次ステップ候補 (要 ASK_FIRST + WF検証):**")
        w(f"- RSR 90-94 NEW_BREAKOUT (days_cross90 ≤ 5d) に絞った条件付き追加を検討")
        w(f"- Study 2 と異なり、WF フォールドを限定した再検証を推奨")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  MD レポート: {output_path}")
    return answer


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  エントリー帯品質監査: RSR 65-84 / RSR 90-94 分解")
    print("=" * 68 + "\n")

    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/4] シグナル収集 + ポートフォリオシミュレーション...")
    t0 = time.time()
    all_signals = collect_signals(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=FULL_START, end=FULL_END,
    )
    print(f"  完了: {len(all_signals):,}件  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(all_signals)
    sel = df[df["selected"] == True]
    sel_a = sel[sel["rsr_bin_a"].isin(RSR_BINS_A)]
    print(f"  selected={len(sel):,}  RSR65-84 selected={len(sel_a):,}")

    print("\n[3/4] State surface 構築...")
    surface = build_state_surface(sel_a)

    # Console preview
    print("\n  ★ A1 State Surface (avg_fwd60, selected RSR65-84)")
    hdr = f"  {'RSR':<10}"
    for s in ENTRY_STATES:
        hdr += f" {s:<15}"
    print(hdr)
    for rsr_bin in RSR_BINS_A:
        row_str = f"  {rsr_bin:<10}"
        for state in ENTRY_STATES:
            cr = surface[(surface["rsr_bin"] == rsr_bin) & (surface["state"] == state)]
            if len(cr) == 0:
                row_str += f" {'—':<15}"; continue
            cr = cr.iloc[0]
            n = cr["count"]; v = cr["avg_fwd60"]
            if n < MIN_N or v is None:
                row_str += f" {'(N='+str(n)+')':<15}"
            else:
                cell = f"{v:+.1f}%(N={n})"
                row_str += f" {cell:<15}"
        print(row_str)

    # Phase B console
    df_b = df[df["rsr_t"].between(90, 95)]
    cap_b = df_b[df_b["skip_reason"] == "CAPACITY"]
    f60_cb = cap_b["fwd60"].dropna().tolist()
    print(f"\n  RSR90-94 capacity-blocked: {len(cap_b)}件  "
          f"avg_fwd60={safe_mean(f60_cb) or 'N/A'}%  PF={safe_pf(f60_cb) or 'N/A'}")

    # Oracle quick view
    a3 = oracle_delta(sel_a, pct=0.10)
    if a3:
        print(f"\n  A3 Oracle BOTTOM10% ΔCAGR推定: {a3['delta_cagr_pp']:+.2f}pp  "
              f"Δavg_fwd60={a3['delta_avg']:+.2f}%")

    print("\n[4/4] CSV + レポート出力...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV: state surface
    surface_path = REPORTS_DIR / "rsr65_80_state_surface.csv"
    surface.to_csv(surface_path, index=False, encoding="utf-8-sig")
    print(f"  State surface CSV: {surface_path}")

    # MD report
    answer = write_report(df, surface, REPORTS_DIR / "entry_zone_audit.md")

    print(f"\n{'='*68}")
    print(f"  最終回答: {answer}")
    print(f"{'='*68}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
