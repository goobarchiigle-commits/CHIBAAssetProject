"""
backtest/missed_signal_alpha_audit.py

Missed Signal Alpha Audit
max_positions=3 制約によるアルファ漏出の定量検証

Analysis 1: Missed signal alpha (20d/40d/60d forward returns)
Analysis 2: Executed vs Missed signal comparison
Analysis 3: Missed reason by slot count
Analysis 4: RSR rank of missed signals vs held positions
Analysis 5: Swap-Lite counterfactual simulation

Output: reports/missed_signal_alpha_audit.md

Run:
    cd C:/ai-trading
    python src/backtest/missed_signal_alpha_audit.py
"""

from __future__ import annotations

import sys, json, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, calc_metrics, _sector_ok, _execute_buy, Position, _take,
    IS_START, IS_END, OOS_START, OOS_END,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR = Path("reports")
FULL_START  = IS_START   # 2018-01-01
FULL_END    = OOS_END    # 2025-12-31


# ─────────────────────────────────────────────────────────────────────
#  CORE RUNNER — Pattern A with missed/executed signal tracking
# ─────────────────────────────────────────────────────────────────────

def run_a_tracked(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    start: str, end: str,
    swap_mode: bool = False,    # if True: Swap-Lite (rotate worst when new>worst)
    swap_threshold: float = 0.0,
) -> dict:
    """
    Run Pattern A (or Swap-Lite) for a given period.
    Returns metrics + missed_records + executed_records.
    """
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
    if len(common_dates) == 0:
        return {}
    n_dates  = len(common_dates)
    n_syms   = len(active_syms)
    sym_to_i = {s: idx for idx, s in enumerate(active_syms)}

    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    sig_mat   = np.zeros((n_dates, n_syms), dtype=np.int8)
    sig_ready = np.zeros(n_syms, dtype=bool)

    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0):
            continue
        open_mat[:, si]  = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]
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

    cash         = float(capital)
    positions: dict[str, Position] = {}
    equity_curve  = []; exposure_list = []; trades = []
    peak_equity   = float(capital)
    cb_active     = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    exit_counts: dict[str, int] = {}

    missed_count = 0; blocked_gross = 0; swap_count = 0

    missed_records:   list[dict] = []
    executed_records: list[dict] = []

    for i, date in enumerate(common_dates):
        invested = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
                       for s, pos in positions.items())
        cur_equity = cash + invested
        equity_curve.append(cur_equity)
        exposure_list.append(invested / max(1.0, cur_equity))

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

        sell_sigs: list[tuple] = []
        buy_cands: list[tuple] = []
        mkt_shock = float(mkt_ret_arr[i]) <= shock_market_thr

        for sym in active_syms:
            si         = sym_to_i[sym]
            is_holding = sym in positions
            hold_idx   = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_mat[i, si])
            close_t    = float(close_mat[i, si])

            if mkt_shock:
                if not is_holding:
                    continue
                if i > 0:
                    prev_c = float(close_mat[i - 1, si])
                    if prev_c > 0 and (close_t / prev_c - 1.0) <= shock_sym_thr:
                        sell_sigs.append((sym, "MARKET_SHOCK_EXIT")); continue
            if mkt_shock and not is_holding:
                continue

            if is_holding and max_hold is not None and hold_idx > max_hold:
                sell_sigs.append((sym, "TIME_STOP")); continue
            if is_holding and rsr_val < rsr_exit_thr and hold_idx >= min_hold:
                sell_sigs.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si]) if sig_ready[si] else 0
            if sig == -1 and is_holding and hold_idx >= min_hold:
                sell_sigs.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not is_holding:
                if i < reentry_ban.get(sym, -1):
                    continue
                if sym_active_mat is not None:
                    if float(sym_active_mat[i, si]) < 0.5:
                        continue
                buy_cands.append((rsr_val, sym))

        if i + 1 >= n_dates:
            break
        next_i = i + 1

        for sym, reason in sell_sigs:
            if sym not in positions:
                continue
            pos      = positions[sym]
            sell_px  = float(open_mat[next_i, sym_to_i[sym]])
            proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
            pnl      = (sell_px - pos.entry_price) * pos.qty
            cash    += proceeds
            trades.append({"side":"SELL","symbol":sym,"entry":pos.entry_price,"exit":sell_px,
                           "qty":pos.qty,"pnl":pnl,"entry_idx":pos.entry_idx,"exit_idx":i,
                           "reason":reason})
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if cb_active or not buy_cands:
            continue

        buy_cands.sort(key=lambda x: -x[0])

        # compute held RSRs once
        held_rsrs_now = {s: float(rsr_mat[i, sym_to_i[s]]) for s in positions}
        held_min_rsr  = min(held_rsrs_now.values()) if held_rsrs_now else 0.0
        held_min_sym  = min(held_rsrs_now, key=held_rsrs_now.get) if held_rsrs_now else None

        n_held_start = len(positions)  # positions at start of buy loop this day

        for bc_idx, (rsr_val, sym) in enumerate(buy_cands):
            open_slots = max_pos - len(positions)

            if open_slots <= 0:
                if swap_mode and held_min_sym is not None:
                    # ── Swap-Lite: check if new > worst held ──────────
                    worst_hold_days = i - positions[held_min_sym].entry_idx
                    if (rsr_val > held_min_rsr + swap_threshold and
                            worst_hold_days >= min_hold):
                        # sell worst
                        w_pos    = positions[held_min_sym]
                        sell_px  = float(open_mat[next_i, sym_to_i[held_min_sym]])
                        proceeds = w_pos.qty * sell_px * (1 - COST_ONE_WAY)
                        pnl      = (sell_px - w_pos.entry_price) * w_pos.qty
                        cash    += proceeds
                        trades.append({"side":"SELL","symbol":held_min_sym,
                                       "entry":w_pos.entry_price,"exit":sell_px,
                                       "qty":w_pos.qty,"pnl":pnl,
                                       "entry_idx":w_pos.entry_idx,"exit_idx":i,
                                       "reason":"SWAP_OUT"})
                        exit_counts["SWAP_OUT"] = exit_counts.get("SWAP_OUT",0) + 1
                        del positions[held_min_sym]
                        swap_count += 1
                        # recalculate
                        held_rsrs_now = {s: float(rsr_mat[i, sym_to_i[s]]) for s in positions}
                        held_min_rsr  = min(held_rsrs_now.values()) if held_rsrs_now else 0.0
                        held_min_sym  = min(held_rsrs_now, key=held_rsrs_now.get) if held_rsrs_now else None
                        # buy new (fall through to normal buy below after break out of if)
                    else:
                        # record missed and skip
                        for rv2, s2 in buy_cands[bc_idx:]:
                            missed_records.append({
                                "date": str(date.date()), "date_idx": i,
                                "symbol": s2, "rsr": rv2,
                                "sector": trade_syms.get(s2, "不明"),
                                "rank": bc_idx + 1 + (buy_cands[bc_idx:].index((rv2, s2))),
                                "n_held_start": n_held_start,
                                "n_held_now": len(positions),
                                "held_min_rsr": held_min_rsr,
                                "rsr_vs_worst": rv2 - held_min_rsr,
                                "better_than_worst": rv2 > held_min_rsr,
                            })
                        missed_count += len(buy_cands) - bc_idx
                        break
                else:
                    # Pattern A: just record and break
                    for miss_i, (rv2, s2) in enumerate(buy_cands[bc_idx:]):
                        missed_records.append({
                            "date": str(date.date()), "date_idx": i,
                            "symbol": s2, "rsr": rv2,
                            "sector": trade_syms.get(s2, "不明"),
                            "rank": bc_idx + miss_i + 1,
                            "n_held_start": n_held_start,
                            "n_held_now": len(positions),
                            "held_min_rsr": held_min_rsr,
                            "rsr_vs_worst": rv2 - held_min_rsr,
                            "better_than_worst": rv2 > held_min_rsr,
                        })
                    missed_count += len(buy_cands) - bc_idx
                    break

            # Normal buy
            buy_px = float(open_mat[next_i, sym_to_i[sym]])
            if buy_px <= 0:
                continue
            if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                continue
            if gross_enabled:
                cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in positions.values()) / max(1.0, capital)
                if cur_gross + buy_px * LOT / max(1.0, capital) > gross_cap:
                    blocked_gross += 1; continue

            n_rem     = sum(1 for _, s in buy_cands if s not in positions)
            eff_slots = min(max_pos - len(positions), max(1, n_rem))
            alloc     = (cash / eff_slots) * cb_scale
            alloc     = min(alloc, capital * MAX_SINGLE_W)
            qty       = int(alloc / buy_px / LOT) * LOT
            if qty <= 0:
                continue
            _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
            cash -= qty * buy_px * (1 + COST_ONE_WAY)
            executed_records.append({
                "date": str(date.date()), "date_idx": i,
                "symbol": sym, "rsr": rsr_val,
                "sector": trade_syms.get(sym, "不明"),
                "rank": bc_idx + 1,
                "n_held_start": n_held_start,
                "entry_px": buy_px,
            })

    metrics = calc_metrics(equity_curve, trades, exposure_list, capital, list(common_dates))
    metrics["exit_reasons"]  = exit_counts
    metrics["missed_signals"]= missed_count
    metrics["swap_count"]    = swap_count
    return {
        "metrics":          metrics,
        "trades":           trades,
        "missed_records":   missed_records,
        "executed_records": executed_records,
        "common_dates":     common_dates,
        "equity_curve":     equity_curve,
        "exposure_list":    exposure_list,
    }


# ─────────────────────────────────────────────────────────────────────
#  FORWARD RETURN COMPUTATION
# ─────────────────────────────────────────────────────────────────────

def compute_fwd_returns(records: list[dict], universe_raw: dict) -> None:
    """Adds fwd20d/fwd40d/fwd60d (%) in-place to each record."""
    for rec in records:
        sym      = rec["symbol"]
        date_str = rec["date"]
        if sym not in universe_raw:
            continue
        close_s = universe_raw[sym]["df"]["Close"]
        try:
            sig_date = pd.Timestamp(date_str)
            if sig_date not in close_s.index:
                continue
            base  = float(close_s[sig_date])
            loc   = close_s.index.get_loc(sig_date)
            for N, key in [(20, "fwd20d"), (40, "fwd40d"), (60, "fwd60d")]:
                fl = loc + N
                if fl < len(close_s):
                    rec[key] = round((float(close_s.iloc[fl]) / base - 1) * 100, 3)
                else:
                    rec[key] = None
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
#  STATS HELPER
# ─────────────────────────────────────────────────────────────────────

def fwd_stats(records: list[dict], key: str) -> dict:
    vals = [r[key] for r in records if r.get(key) is not None]
    if not vals:
        return {"n": 0}
    arr = np.array(vals)
    wins = arr[arr > 0]
    return {
        "n":        len(vals),
        "mean":     round(float(np.mean(arr)), 3),
        "median":   round(float(np.median(arr)), 3),
        "std":      round(float(np.std(arr)), 3),
        "win_rate": round(len(wins) / len(arr) * 100, 1),
        "ir":       round(float(np.mean(arr) / np.std(arr)), 3) if np.std(arr) > 0 else 0.0,
        "p10":      round(float(np.percentile(arr, 10)), 3),
        "p25":      round(float(np.percentile(arr, 25)), 3),
        "p75":      round(float(np.percentile(arr, 75)), 3),
        "p90":      round(float(np.percentile(arr, 90)), 3),
    }


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 1: Missed signal alpha
# ─────────────────────────────────────────────────────────────────────

def analysis1_missed_alpha(missed: list[dict]) -> dict:
    """Forward return statistics for missed signals by period."""
    # Split by IS / OOS
    is_miss  = [r for r in missed if r["date"] < OOS_START]
    oos_miss = [r for r in missed if r["date"] >= OOS_START]

    out = {}
    for period_name, recs in [("IS_2018_2024", is_miss),
                               ("OOS_2025", oos_miss),
                               ("Full_2018_2025", missed)]:
        out[period_name] = {
            "n":   len(recs),
            "fwd20d": fwd_stats(recs, "fwd20d"),
            "fwd40d": fwd_stats(recs, "fwd40d"),
            "fwd60d": fwd_stats(recs, "fwd60d"),
        }
    return out


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 2: Executed vs Missed comparison
# ─────────────────────────────────────────────────────────────────────

def analysis2_exec_vs_missed(executed: list[dict], missed: list[dict]) -> dict:
    is_exec  = [r for r in executed if r["date"] < OOS_START]
    is_miss  = [r for r in missed   if r["date"] < OOS_START]
    oos_exec = [r for r in executed if r["date"] >= OOS_START]
    oos_miss = [r for r in missed   if r["date"] >= OOS_START]

    out = {}
    for pname, ex_r, mi_r in [("IS",  is_exec, is_miss),
                               ("OOS", oos_exec, oos_miss),
                               ("Full",executed, missed)]:
        out[pname] = {}
        for k in ["fwd20d", "fwd40d", "fwd60d"]:
            out[pname][k] = {
                "executed": fwd_stats(ex_r, k),
                "missed":   fwd_stats(mi_r, k),
            }
        # RSR comparison
        ex_rsrs = [r["rsr"] for r in ex_r if r.get("rsr", 0) > 0]
        mi_rsrs = [r["rsr"] for r in mi_r if r.get("rsr", 0) > 0]
        out[pname]["rsr"] = {
            "executed_mean":   round(np.mean(ex_rsrs),1) if ex_rsrs else 0,
            "executed_median": round(np.median(ex_rsrs),1) if ex_rsrs else 0,
            "missed_mean":     round(np.mean(mi_rsrs),1) if mi_rsrs else 0,
            "missed_median":   round(np.median(mi_rsrs),1) if mi_rsrs else 0,
        }
    return out


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 3: By slot count
# ─────────────────────────────────────────────────────────────────────

def analysis3_by_slot(missed: list[dict]) -> dict:
    """
    Group missed signals by how the day started (n_held_start).
    3→full: portfolio was already full at start of day's buy loop.
    <3→full: portfolio filled up during the day, then additional missed.
    """
    pre_full  = [r for r in missed if r.get("n_held_start", 3) >= 3]   # already full
    day_fill  = [r for r in missed if r.get("n_held_start", 3) < 3]    # filled during day

    out = {}
    for gname, grp in [("pre_full (3→full)", pre_full),
                        ("day_fill (<3→full)", day_fill)]:
        if not grp:
            out[gname] = {"n": 0}
            continue
        out[gname] = {
            "n": len(grp),
            "n_signals_per_day": {},
            "fwd20d": fwd_stats(grp, "fwd20d"),
            "fwd40d": fwd_stats(grp, "fwd40d"),
            "fwd60d": fwd_stats(grp, "fwd60d"),
            "rsr_mean":   round(np.mean([r["rsr"] for r in grp]),1),
            "rsr_median": round(np.median([r["rsr"] for r in grp]),1),
        }
    return out


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 4: RSR rank analysis
# ─────────────────────────────────────────────────────────────────────

def analysis4_rank(missed: list[dict]) -> dict:
    """
    Was the missed signal actually better than the worst held position?
    If better_than_worst: swapping out the weakest would have been justified.
    """
    better = [r for r in missed if r.get("better_than_worst", False)]
    worse  = [r for r in missed if not r.get("better_than_worst", True)]

    all_gaps = [r.get("rsr_vs_worst", 0) for r in missed]
    b_gaps   = [r.get("rsr_vs_worst", 0) for r in better]

    is_missed = [r for r in missed if r["date"] < OOS_START]
    oos_missed = [r for r in missed if r["date"] >= OOS_START]

    def pct_better(recs): return round(sum(1 for r in recs if r.get("better_than_worst")) / max(len(recs),1)*100,1)

    return {
        "total_missed":       len(missed),
        "better_than_worst":  len(better),
        "pct_better":         round(len(better)/max(len(missed),1)*100,1),
        "worse_than_worst":   len(worse),
        "rsr_vs_worst_mean":  round(np.mean(all_gaps),1) if all_gaps else 0,
        "rsr_vs_worst_p50":   round(np.median(all_gaps),1) if all_gaps else 0,
        "rsr_vs_worst_p75":   round(np.percentile(all_gaps,75),1) if all_gaps else 0,
        "rsr_gap_if_better":  round(np.mean(b_gaps),1) if b_gaps else 0,
        "IS_pct_better":      pct_better(is_missed),
        "OOS_pct_better":     pct_better(oos_missed),
        # Forward returns for better vs worse splits
        "better_fwd60d": fwd_stats(better, "fwd60d"),
        "worse_fwd60d":  fwd_stats(worse,  "fwd60d"),
        "all_fwd60d":    fwd_stats(missed,  "fwd60d"),
        # By rank (was the missed signal rank 4, 5, or 6+?)
        "by_rank": {
            "rank4": {"fwd60d": fwd_stats([r for r in missed if r.get("rank")==4], "fwd60d"),
                      "n": sum(1 for r in missed if r.get("rank")==4)},
            "rank5": {"fwd60d": fwd_stats([r for r in missed if r.get("rank")==5], "fwd60d"),
                      "n": sum(1 for r in missed if r.get("rank")==5)},
            "rank6+":{"fwd60d": fwd_stats([r for r in missed if r.get("rank",0)>=6], "fwd60d"),
                      "n": sum(1 for r in missed if r.get("rank",0)>=6)},
        },
    }


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 5: Swap-Lite simulation
# ─────────────────────────────────────────────────────────────────────

def analysis5_swap_lite(m_a_is, m_a_oos, m_swap_is, m_swap_oos) -> dict:
    def delta(key, base, new):
        bv = base.get(key, 0); nv = new.get(key, 0)
        return round(nv - bv, 3) if isinstance(bv,(int,float)) else "—"

    return {
        "IS": {
            "A":    m_a_is,
            "Swap": m_swap_is,
            "CAGR_delta":   delta("cagr",         m_a_is, m_swap_is),
            "Sharpe_delta": delta("sharpe",        m_a_is, m_swap_is),
            "MaxDD_delta":  delta("max_dd",        m_a_is, m_swap_is),
            "Exp_delta":    delta("avg_exposure",  m_a_is, m_swap_is),
            "Trades_delta": delta("n_trades_yr",   m_a_is, m_swap_is),
        },
        "OOS": {
            "A":    m_a_oos,
            "Swap": m_swap_oos,
            "CAGR_delta":   delta("cagr",         m_a_oos, m_swap_oos),
            "Sharpe_delta": delta("sharpe",        m_a_oos, m_swap_oos),
            "MaxDD_delta":  delta("max_dd",        m_a_oos, m_swap_oos),
            "Exp_delta":    delta("avg_exposure",  m_a_oos, m_swap_oos),
        },
    }


# ─────────────────────────────────────────────────────────────────────
#  MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(a1, a2, a3, a4, a5, missed, executed, output_path: Path) -> None:
    L = []
    def w(s=""): L.append(s)

    w("# Missed Signal Alpha Audit")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  対象期間: 2018-2025  |  max_positions=3\n")
    w("**問い:** Pattern A は max_positions=3 の制約により、構造的に高品質シグナルを捨てているか？\n")

    # ── Analysis 1 ────────────────────────────────────────────────────
    w("---\n## 分析1: Missed Signal Alpha (Forward Return)")
    w("\n### 期間別集計\n")
    w("| 期間 | N | fwd20d mean | fwd20d WR | fwd40d mean | fwd40d WR | fwd60d mean | fwd60d WR | IR(60d) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for pname, pdata in a1.items():
        s20 = pdata.get("fwd20d", {}); s40 = pdata.get("fwd40d", {}); s60 = pdata.get("fwd60d", {})
        w(f"| {pname} | {pdata['n']} "
          f"| {s20.get('mean','—'):+}% | {s20.get('win_rate','—')}% "
          f"| {s40.get('mean','—'):+}% | {s40.get('win_rate','—')}% "
          f"| {s60.get('mean','—'):+}% | {s60.get('win_rate','—')}% "
          f"| {s60.get('ir','—')} |")

    w("\n### IS 2018-2024 Missed Signal 詳細分布")
    s_is60 = a1.get("IS_2018_2024",{}).get("fwd60d",{})
    if s_is60.get("n",0) > 0:
        w(f"\n- N={s_is60['n']}  mean={s_is60['mean']:+}%  median={s_is60['median']:+}%")
        w(f"- Win Rate={s_is60['win_rate']}%  IR={s_is60['ir']}")
        w(f"- P10={s_is60['p10']:+}%  P25={s_is60['p25']:+}%  P75={s_is60['p75']:+}%  P90={s_is60['p90']:+}%")

    # ── Analysis 2 ────────────────────────────────────────────────────
    w("\n---\n## 分析2: Executed vs Missed Signal 比較\n")
    w("### IS 2018-2024\n")
    w("| | N | fwd20d mean | fwd20d WR | fwd40d mean | fwd40d WR | fwd60d mean | fwd60d WR |")
    w("|---|---|---|---|---|---|---|---|")
    for gname in ["executed","missed"]:
        for k in ["fwd20d","fwd40d","fwd60d"]:
            s = a2.get("IS",{}).get(k,{}).get(gname,{})
    # table rows
    a2_is = a2.get("IS",{})
    for gname in ["executed","missed"]:
        s20 = a2_is.get("fwd20d",{}).get(gname,{})
        s40 = a2_is.get("fwd40d",{}).get(gname,{})
        s60 = a2_is.get("fwd60d",{}).get(gname,{})
        w(f"| {gname} | {s20.get('n','—')} "
          f"| {s20.get('mean','—'):+}% | {s20.get('win_rate','—')}% "
          f"| {s40.get('mean','—'):+}% | {s40.get('win_rate','—')}% "
          f"| {s60.get('mean','—'):+}% | {s60.get('win_rate','—')}% |")

    w("\n### OOS 2025\n")
    w("| | N | fwd20d mean | fwd20d WR | fwd60d mean | fwd60d WR |")
    w("|---|---|---|---|---|---|")
    a2_oos = a2.get("OOS",{})
    for gname in ["executed","missed"]:
        s20 = a2_oos.get("fwd20d",{}).get(gname,{})
        s60 = a2_oos.get("fwd60d",{}).get(gname,{})
        w(f"| {gname} | {s20.get('n','—')} "
          f"| {s20.get('mean','—'):+}% | {s20.get('win_rate','—')}% "
          f"| {s60.get('mean','—'):+}% | {s60.get('win_rate','—')}% |")

    w("\n### RSR比較\n")
    w("| 期間 | Executed RSR mean | Executed RSR median | Missed RSR mean | Missed RSR median |")
    w("|---|---|---|---|---|")
    for pname in ["IS","OOS","Full"]:
        r = a2.get(pname,{}).get("rsr",{})
        w(f"| {pname} | {r.get('executed_mean','—')} | {r.get('executed_median','—')} "
          f"| {r.get('missed_mean','—')} | {r.get('missed_median','—')} |")

    # ── Analysis 3 ────────────────────────────────────────────────────
    w("\n---\n## 分析3: Missed理由別 (スロット状況)\n")
    w("| カテゴリ | N | fwd60d mean | fwd60d WR | RSR mean |")
    w("|---|---|---|---|---|")
    for gname, gdata in a3.items():
        if gdata.get("n",0) == 0:
            continue
        s60 = gdata.get("fwd60d",{})
        w(f"| {gname} | {gdata['n']} | {s60.get('mean','—'):+}% | {s60.get('win_rate','—')}% "
          f"| {gdata.get('rsr_mean','—')} |")

    # ── Analysis 4 ────────────────────────────────────────────────────
    w("\n---\n## 分析4: RSR Rank Analysis\n")
    w(f"- 全 missed signals: **{a4['total_missed']}件**")
    w(f"- うち worst held より RSR が高い: **{a4['better_than_worst']}件 ({a4['pct_better']}%)**")
    w(f"- RSR vs worst gap (mean): {a4['rsr_vs_worst_mean']:+}  (p50: {a4['rsr_vs_worst_p50']:+}  p75: {a4['rsr_vs_worst_p75']:+})")
    w(f"- IS期間 better%: {a4['IS_pct_better']}%  /  OOS期間 better%: {a4['OOS_pct_better']}%")

    w("\n### Better vs Worse than Worst — fwd60d比較\n")
    w("| グループ | N | mean | median | WR | IR |")
    w("|---|---|---|---|---|---|")
    for gname, key in [("better_than_worst","better_fwd60d"),("worse_than_worst","worse_fwd60d"),("全体","all_fwd60d")]:
        s = a4.get(key,{})
        if s.get("n",0) > 0:
            w(f"| {gname} | {s['n']} | {s['mean']:+}% | {s['median']:+}% | {s['win_rate']}% | {s['ir']} |")

    w("\n### Rank別 fwd60d\n")
    w("| Rank | N | mean | WR |")
    w("|---|---|---|---|")
    for rk in ["rank4","rank5","rank6+"]:
        s = a4.get("by_rank",{}).get(rk,{})
        sf = s.get("fwd60d",{})
        w(f"| {rk} | {s.get('n',0)} | {sf.get('mean','—'):+}% | {sf.get('win_rate','—')}% |")

    # ── Analysis 5 ────────────────────────────────────────────────────
    w("\n---\n## 分析5: Swap-Lite Counterfactual Simulation\n")
    w("ルール: 満杯時に新規シグナルRSR > 最低保有RSR + 0 なら最低保有を売却して新規取得\n")
    w("| 期間 | 指標 | Pattern A | Swap-Lite | Delta |")
    w("|---|---|---|---|---|")
    for period in ["IS","OOS"]:
        pdata = a5.get(period,{})
        m_a   = pdata.get("A",{})
        m_s   = pdata.get("Swap",{})
        for key, label in [("cagr","CAGR%"),("sharpe","Sharpe"),("max_dd","MaxDD%"),
                            ("avg_exposure","Exposure%"),("n_trades_yr","Trades/yr")]:
            av = m_a.get(key,"—"); sv = m_s.get(key,"—")
            dv = pdata.get(f"{label.split('%')[0]}_delta", pdata.get(key.replace("avg_","")+"_delta","—"))
            # simplify: compute delta directly
            if isinstance(av,(int,float)) and isinstance(sv,(int,float)):
                dv = f"{sv-av:+.2f}"
            else:
                dv = "—"
            a_s  = f"{av:.2f}" if isinstance(av,float) else str(av)
            s_s  = f"{sv:.2f}" if isinstance(sv,float) else str(sv)
            w(f"| {period} | {label} | {a_s} | {s_s} | {dv} |")

    w(f"\n- IS Swap実行回数: {a5.get('IS',{}).get('Swap',{}).get('swap_count',0)}")
    w(f"- OOS Swap実行回数: {a5.get('OOS',{}).get('Swap',{}).get('swap_count',0)}")

    # ── 最終結論 ──────────────────────────────────────────────────────
    w("\n---\n## 最終結論\n")

    # Determine verdict
    is_f60 = a1.get("IS_2018_2024",{}).get("fwd60d",{})
    is_f60_wr   = is_f60.get("win_rate",0)
    is_f60_mean = is_f60.get("mean",0)
    is_pct_b    = a4.get("IS_pct_better",0)
    swap_is_cagr_delta = (a5.get("IS",{}).get("Swap",{}).get("cagr",0) or 0) - (a5.get("IS",{}).get("A",{}).get("cagr",0) or 0)

    # Verdict A: 2025-only?  B: structural leakage?  C: room for improvement?  D: no improvement?
    if is_f60_wr >= 60 and is_f60_mean >= 5.0:
        if is_pct_b >= 40:
            verdict = "**B: 構造的アルファ漏出あり**"
            reason  = f"IS期間のmissed fwd60d win_rate={is_f60_wr}% / mean={is_f60_mean:+.2f}% かつ {is_pct_b}%のmissedシグナルが保有最下位より高RSR"
        else:
            verdict = "**C: スロット管理改善余地あり（限定的）**"
            reason  = f"IS期間fwd60d WR={is_f60_wr}%は高いが、better_than_worst={is_pct_b}%は低い"
    elif is_f60_wr >= 55:
        verdict = "**C: スロット管理改善余地あり（限定的）**"
        reason  = f"IS期間fwd60d WR={is_f60_wr}% / mean={is_f60_mean:+.2f}% — 弱いアルファ"
    else:
        verdict = "**D: 改善余地なし**"
        reason  = f"IS期間fwd60d WR={is_f60_wr}% / mean={is_f60_mean:+.2f}% — 有意なアルファなし"

    w(f"### 判定: {verdict}\n")
    w(f"根拠: {reason}\n")

    w("### Entry Timing vs Slot Optimization — どちらを優先すべきか\n")
    if swap_is_cagr_delta > 0.5:
        priority = "**Slot Optimization (Swap-Lite) を優先**"
        pri_reason = f"Swap-Lite IS CAGR改善 = {swap_is_cagr_delta:+.1f}pp > 0.5pp"
    else:
        priority = "**Entry Timing を優先**"
        pri_reason = f"Swap-Lite IS CAGR改善 = {swap_is_cagr_delta:+.1f}pp ≤ 0.5pp かつシグナル品質向上の方が安全"

    w(f"#### 優先順位判定: {priority}\n")
    w(f"根拠: {pri_reason}\n")
    w("\n| 項目 | Entry Timing | Slot Optimization |")
    w("|---|---|---|")
    w("| IS リスク | 低 (シグナル減少のみ) | 中 (回転コスト発生) |")
    w("| OOS 安定性 | 高 (品質向上は普遍的) | 低 (相場環境依存) |")
    w("| 実装複雑度 | 低 (1フラグ変更) | 中 (ロジック追加) |")
    w("| PARAMS_LOCKED影響 | なし | あり (max_positions変更不要だが動作変化) |")
    w("| 期待CAGR改善 | +3〜7pp (推定) | " + f"{swap_is_cagr_delta:+.1f}pp (実測IS) |")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート保存: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3

    print("=" * 68)
    print("  Missed Signal Alpha Audit — 2018-2025")
    print("=" * 68 + "\n")

    print("[1/6] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    def run_k(label, start, end, swap=False):
        print(f"[run] {label}  {start}〜{end}  swap={'ON' if swap else 'OFF'}")
        t0  = time.time()
        res = run_a_tracked(universe_raw, rsr_df, alpha_df, sym_active_df,
                            regime_df, topix_close, rsr_syms, cfg,
                            start=start, end=end, swap_mode=swap)
        m   = res["metrics"]
        print(f"      CAGR={m.get('cagr',0):+.1f}%  Sharpe={m.get('sharpe',0):.3f}  "
              f"MaxDD={m.get('max_dd',0):.1f}%  missed={m.get('missed_signals',0)}  "
              f"swap={m.get('swap_count',0)}  ({time.time()-t0:.1f}s)")
        return res

    # Full period for analyses 1-4
    print("\n[2/6] Full period Pattern A (2018-2025)...")
    full_a = run_k("A full", FULL_START, FULL_END)

    # Compute forward returns
    print("\n[3/6] Forward return計算...")
    compute_fwd_returns(full_a["missed_records"],   universe_raw)
    compute_fwd_returns(full_a["executed_records"], universe_raw)
    print(f"      missed: {len(full_a['missed_records'])}件  "
          f"executed: {len(full_a['executed_records'])}件")

    # IS and OOS separately for Analysis 5
    print("\n[4/6] IS/OOS Pattern A + Swap-Lite...")
    a_is    = run_k("A IS",     IS_START,  IS_END)
    a_oos   = run_k("A OOS",    OOS_START, OOS_END)
    swap_is = run_k("Swap IS",  IS_START,  IS_END,   swap=True)
    swap_oos= run_k("Swap OOS", OOS_START, OOS_END,  swap=True)

    print("\n[5/6] 分析実行...")
    a1 = analysis1_missed_alpha(full_a["missed_records"])
    a2 = analysis2_exec_vs_missed(full_a["executed_records"], full_a["missed_records"])
    a3 = analysis3_by_slot(full_a["missed_records"])
    a4 = analysis4_rank(full_a["missed_records"])
    a5 = analysis5_swap_lite(a_is["metrics"], a_oos["metrics"],
                              swap_is["metrics"], swap_oos["metrics"])
    a5["IS"]["Swap"]["swap_count"] = swap_is["metrics"].get("swap_count",0)
    a5["OOS"]["Swap"]["swap_count"]= swap_oos["metrics"].get("swap_count",0)

    print("\n[6/6] レポート生成...")
    out_path = REPORTS_DIR / "missed_signal_alpha_audit.md"
    write_report(a1, a2, a3, a4, a5,
                 full_a["missed_records"], full_a["executed_records"], out_path)

    # ── Console summary ────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  ★ コンソール要約")
    print("="*68)

    is_f  = a1.get("IS_2018_2024",{})
    oos_f = a1.get("OOS_2025",{})
    full_f= a1.get("Full_2018_2025",{})

    print(f"\n  missed signals: IS={is_f['n']}件  OOS={oos_f['n']}件  全体={full_f['n']}件")
    print(f"\n  [分析1] Missed Signal Forward Return")
    for pname, pdata in [("IS  ", a1.get("IS_2018_2024",{})),
                          ("OOS ", a1.get("OOS_2025",{})),
                          ("Full", a1.get("Full_2018_2025",{}))]:
        s20=pdata.get("fwd20d",{}); s40=pdata.get("fwd40d",{}); s60=pdata.get("fwd60d",{})
        print(f"  {pname}: fwd20d mean={s20.get('mean',0):+.2f}% WR={s20.get('win_rate',0):.1f}%  "
              f"fwd40d mean={s40.get('mean',0):+.2f}% WR={s40.get('win_rate',0):.1f}%  "
              f"fwd60d mean={s60.get('mean',0):+.2f}% WR={s60.get('win_rate',0):.1f}%  IR={s60.get('ir',0):.3f}")

    print(f"\n  [分析2] Executed vs Missed RSR")
    r_is = a2.get("IS",{}).get("rsr",{})
    print(f"  IS:  Executed RSR mean={r_is.get('executed_mean',0):.1f}  Missed RSR mean={r_is.get('missed_mean',0):.1f}")

    print(f"\n  [分析4] RSR rank")
    print(f"  better_than_worst: {a4['better_than_worst']}/{a4['total_missed']} ({a4['pct_better']}%)")
    print(f"  IS better%: {a4['IS_pct_better']}%  OOS better%: {a4['OOS_pct_better']}%")
    s_bt = a4.get("better_fwd60d",{}); s_wt = a4.get("worse_fwd60d",{})
    print(f"  better fwd60d: mean={s_bt.get('mean',0):+.2f}% WR={s_bt.get('win_rate',0):.1f}%")
    print(f"  worse  fwd60d: mean={s_wt.get('mean',0):+.2f}% WR={s_wt.get('win_rate',0):.1f}%")

    print(f"\n  [分析5] Swap-Lite vs A")
    d_is  = a5.get("IS",{})
    d_oos = a5.get("OOS",{})
    m_a_is  = d_is.get("A",{});   m_s_is  = d_is.get("Swap",{})
    m_a_oos = d_oos.get("A",{});  m_s_oos = d_oos.get("Swap",{})
    print(f"  IS:  A CAGR={m_a_is.get('cagr',0):+.1f}%  Swap CAGR={m_s_is.get('cagr',0):+.1f}%  "
          f"Δ={m_s_is.get('cagr',0)-m_a_is.get('cagr',0):+.1f}pp  "
          f"swap_count={a5.get('IS',{}).get('Swap',{}).get('swap_count',0)}")
    print(f"  OOS: A CAGR={m_a_oos.get('cagr',0):+.1f}%  Swap CAGR={m_s_oos.get('cagr',0):+.1f}%  "
          f"Δ={m_s_oos.get('cagr',0)-m_a_oos.get('cagr',0):+.1f}pp  "
          f"swap_count={a5.get('OOS',{}).get('Swap',{}).get('swap_count',0)}")

    print(f"\n  レポート → {out_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
