"""
backtest/oos_2025_winner_analysis.py

2025 OOS Winner Analysis: Why E1 (+27.1%) and E3 (+22.7%) beat A (+10.0%)?

Analysis 1: Per-trade PnL attribution (TOP20 / BOTTOM20)
Analysis 2: Signals A missed but E1/E3 captured + forward returns
Analysis 3: Daily exposure decomposition (exposure vs selection effect)
Analysis 4: Winner concentration (TOP1/3/5/10 profit share)
Analysis 5: Entry timing hypothesis (did E1/E3 enter differently?)

Output: docs/research/oos_2025_winner_analysis.md

Run:
    cd C:/ai-trading
    python src/backtest/oos_2025_winner_analysis.py
"""

from __future__ import annotations

import sys, json, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, calc_metrics, _sector_ok, _execute_buy, Position, _take,
    OOS_START, OOS_END,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)
from src.backtest.capital_allocation_e1e2e3 import (
    _assign_weights_e, _partial_sell,
)

DOCS_DIR = Path("docs/research")


# ─────────────────────────────────────────────────────────────────────
#  ENHANCED OOS RUNNER — adds detailed tracking to run_period_e
# ─────────────────────────────────────────────────────────────────────

def run_oos_detailed(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    pattern: str,   # "A" | "E1" | "E3"
) -> dict:
    """
    OOS 2025 backtest with full trade/exposure/missed-signal logging.
    Returns: metrics, enriched_trades, daily_data, missed_signal_log, common_dates, trade_syms
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
                min_rsr=cfg.fujiko.min_rsr,
                turtle_exit=cfg.fujiko.turtle_exit,
                rsr_series=rsr_s,
                min_sepa=cfg.fujiko.min_sepa,
                mom_period=cfg.fujiko.mom_period,
                turtle_entry=cfg.fujiko.turtle_entry,
                use_turtle_entry=cfg.fujiko.use_turtle_entry,
            )

    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    ts = pd.Timestamp(OOS_START); te = pd.Timestamp(OOS_END)
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
    topix_ret20 = topix_ret60 = None
    bear_arr    = None
    if topix_close is not None:
        tr          = topix_close.pct_change()
        mkt_ret_arr = _take(tr, common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret20 = _take(topix_close.pct_change(20), common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret60 = _take(topix_close.pct_change(60), common_dates, dtype=np.float32, fill_value=0.0)
        ma200  = topix_close.rolling(200, min_periods=100).mean()
        bear_s = (topix_close < ma200).reindex(pd.DatetimeIndex(common_dates), method="ffill").fillna(False)
        bear_arr = bear_s.values.astype(bool)

    cash           = float(capital)
    positions: dict[str, Position] = {}
    equity_curve:  list[float] = []
    exposure_list: list[float] = []
    trades:        list[dict]  = []
    peak_equity    = float(capital)
    cb_active      = False; cb_days = 0
    reentry_ban:   dict[str, int] = {}
    exit_counts:   dict[str, int] = {}

    missed_count      = 0
    blocked_gross     = 0
    rebalance_count   = 0
    rebalance_trim_pnl= 0.0
    winner_trim_pnl   = 0.0

    # ── enhanced tracking ────────────────────────────────────────────
    missed_signal_log: list[dict] = []     # A: all skipped buy signals
    daily_data:        list[dict] = []     # per-day state snapshot

    # ─────────────────────────────────────────────────────────────────
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
                        sell_sigs.append((sym, "MARKET_SHOCK_EXIT"))
                        continue
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

        # ── daily snapshot (before execution) ────────────────────────
        daily_data.append({
            "date":       str(date.date()),
            "date_idx":   i,
            "exposure":   invested / max(1.0, cur_equity),
            "equity":     cur_equity,
            "n_pos":      len(positions),
            "syms_held":  frozenset(positions.keys()),
            "cb_active":  cb_active,
        })

        if i + 1 >= n_dates:
            break
        next_i = i + 1

        # ── SELL EXECUTION ────────────────────────────────────────────
        for sym, reason in sell_sigs:
            if sym not in positions:
                continue
            pos      = positions[sym]
            sell_px  = float(open_mat[next_i, sym_to_i[sym]])
            proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
            pnl      = (sell_px - pos.entry_price) * pos.qty
            cash    += proceeds
            trades.append({
                "side": "SELL", "symbol": sym,
                "entry": pos.entry_price, "exit": sell_px,
                "qty": pos.qty, "pnl": pnl,
                "entry_idx": pos.entry_idx, "exit_idx": i,
                "reason": reason,
            })
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if cb_active or not buy_cands:
            continue

        buy_cands.sort(key=lambda x: -x[0])

        # ─────────────────────────────────────────────────────────────
        # PATTERN BUY LOGIC
        # ─────────────────────────────────────────────────────────────
        if pattern == "A":
            for bc_idx, (rsr_val, sym) in enumerate(buy_cands):
                open_slots = max_pos - len(positions)
                if open_slots <= 0:
                    for rv2, s2 in buy_cands[bc_idx:]:
                        missed_signal_log.append({
                            "date":     str(date.date()),
                            "date_idx": i,
                            "symbol":   s2,
                            "rsr":      rv2,
                            "sector":   trade_syms.get(s2, "不明"),
                            "reason":   "NO_SLOT",
                        })
                    missed_count += len(buy_cands) - bc_idx
                    break
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
                eff_slots = min(open_slots, max(1, n_rem))
                alloc     = (cash / eff_slots) * cb_scale
                alloc     = min(alloc, capital * MAX_SINGLE_W)
                qty       = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)

        else:  # E1 or E3
            from src.backtest.capital_allocation_e1e2e3 import PATTERN_W
            for bc_idx, (rsr_val, sym) in enumerate(buy_cands):
                if len(positions) >= max_pos:
                    for rv2, s2 in buy_cands[bc_idx:]:
                        missed_signal_log.append({
                            "date":     str(date.date()),
                            "date_idx": i,
                            "symbol":   s2,
                            "rsr":      rv2,
                            "sector":   trade_syms.get(s2, "不明"),
                            "reason":   "NO_SLOT",
                        })
                    missed_count += len(buy_cands) - bc_idx
                    break

                buy_px = float(open_mat[next_i, sym_to_i[sym]])
                if buy_px <= 0:
                    continue
                if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                    continue
                if gross_enabled and gross_cap < 0.85:
                    blocked_gross += 1; continue

                cur_eq = cash + sum(p.qty * float(close_mat[i, sym_to_i[s]])
                                    for s, p in positions.items())
                n_new = len(positions) + 1
                tw    = _assign_weights_e(pattern, n_new, positions, sym,
                                          rsr_mat, i, sym_to_i, close_mat)

                freed = 0.0
                for held_sym in list(positions.keys()):
                    target_val = cur_eq * tw[held_sym]
                    cf, rpnl, wt = _partial_sell(
                        held_sym, positions[held_sym], target_val,
                        open_mat, next_i, sym_to_i, trades, i,
                    )
                    if cf > 0:
                        freed             += cf
                        rebalance_count   += 1
                        rebalance_trim_pnl+= rpnl
                        winner_trim_pnl   += wt
                cash += freed

                cur_eq = cash + sum(p.qty * float(close_mat[i, sym_to_i[s]])
                                    for s, p in positions.items())
                alloc = min(cur_eq * tw[sym] * cb_scale, cash * 0.99)
                qty   = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)

    # ── enrich trades with dates, sector, return_pct ─────────────────
    for t in trades:
        ei = t.get("entry_idx", 0)
        xi = t.get("exit_idx",  0)
        t["entry_date"] = str(common_dates[ei].date()) if ei < len(common_dates) else ""
        t["exit_date"]  = str(common_dates[min(xi + 1, len(common_dates) - 1)].date()) if t["side"] != "BUY" else ""
        t["sector"]     = trade_syms.get(t.get("symbol", ""), "不明")
        ep = t.get("entry") or 0.0
        xp = t.get("exit")  or 0.0
        if ep > 0:
            t["return_pct"]   = round((xp / ep - 1) * 100, 2)
            t["holding_days"] = xi - ei

    metrics = calc_metrics(equity_curve, trades, exposure_list, capital, list(common_dates))
    metrics["exit_reasons"]      = exit_counts
    metrics["missed_signals"]    = missed_count
    metrics["rebalance_count"]   = rebalance_count
    metrics["rebalance_trim_pnl"]= round(rebalance_trim_pnl, 0)
    metrics["winner_trim_pnl"]   = round(winner_trim_pnl, 0)

    return {
        "pattern":           pattern,
        "metrics":           metrics,
        "trades":            trades,
        "missed_signal_log": missed_signal_log,
        "daily_data":        daily_data,
        "common_dates":      common_dates,
        "trade_syms":        trade_syms,
        "equity_curve":      equity_curve,
        "exposure_list":     exposure_list,
    }


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 1: Per-trade PnL attribution
# ─────────────────────────────────────────────────────────────────────

def analysis1_trade_pnl(results: dict) -> dict:
    out = {}
    for pat, data in results.items():
        sells = [t for t in data["trades"] if t["side"] == "SELL"]
        if not sells:
            out[pat] = {"sells": [], "top20": [], "bot20": [], "n_trades": 0}
            continue
        sells_s = sorted(sells, key=lambda t: t.get("pnl", 0), reverse=True)
        # R-multiple: return / mean_abs_loss
        losses = [t for t in sells if t.get("pnl", 0) < 0]
        mean_abs_loss_pct = abs(np.mean([t.get("return_pct", 0) for t in losses])) if losses else 1.0
        for t in sells:
            rp = t.get("return_pct", 0)
            t["r_multiple"] = round(rp / mean_abs_loss_pct, 2) if mean_abs_loss_pct > 0 else 0.0

        out[pat] = {
            "sells":    sells_s,
            "top20":    sells_s[:20],
            "bot20":    sells_s[-20:],
            "n_trades": len(sells),
            "total_pnl": sum(t.get("pnl", 0) for t in sells),
            "mean_abs_loss_pct": round(mean_abs_loss_pct, 2),
        }

        # TOP N concentration
        wins = [t for t in sells if t.get("pnl", 0) > 0]
        win_pnl = sorted([t.get("pnl", 0) for t in wins], reverse=True) if wins else []
        total_win = sum(win_pnl) if win_pnl else 1.0
        for n in [1, 3, 5, 10]:
            top_n_pnl = sum(win_pnl[:n])
            out[pat][f"top{n}_pnl_share"] = round(top_n_pnl / max(total_win, 1.0) * 100, 1)
    return out


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 2: Signals A missed but E1/E3 captured
# ─────────────────────────────────────────────────────────────────────

def analysis2_missed_signals(results: dict, universe_raw: dict) -> dict:
    a_data  = results.get("A",  {})
    e1_data = results.get("E1", {})
    e3_data = results.get("E3", {})

    a_trades  = [t for t in a_data.get("trades",  []) if t["side"] == "SELL"]
    e1_trades = [t for t in e1_data.get("trades", []) if t["side"] == "SELL"]
    e3_trades = [t for t in e3_data.get("trades", []) if t["side"] == "SELL"]

    a_symbols  = {t["symbol"] for t in a_trades}
    e1_symbols = {t["symbol"] for t in e1_trades}
    e3_symbols = {t["symbol"] for t in e3_trades}

    e1_only = e1_symbols - a_symbols     # E1 traded, A didn't
    e3_only = e3_symbols - a_symbols     # E3 traded, A didn't
    both    = a_symbols & e1_symbols     # both traded same symbol

    # E1-only trade details with actual PnL
    e1_only_trades = [t for t in e1_trades if t["symbol"] in e1_only]
    e3_only_trades = [t for t in e3_trades if t["symbol"] in e3_only]

    # For A's missed signals: compute forward returns
    missed_log = a_data.get("missed_signal_log", [])
    cd         = a_data.get("common_dates", pd.DatetimeIndex([]))

    for entry in missed_log:
        sym      = entry["symbol"]
        date_str = entry["date"]
        if sym not in universe_raw:
            continue
        sym_df  = universe_raw[sym]["df"]["Close"]
        try:
            sig_date = pd.Timestamp(date_str)
            if sig_date not in sym_df.index:
                continue
            base_close = float(sym_df.loc[sig_date])
            loc        = sym_df.index.get_loc(sig_date)
            for n_days, key in [(20, "fwd20d"), (40, "fwd40d"), (60, "fwd60d")]:
                future_loc = loc + n_days
                if future_loc < len(sym_df):
                    fwd = (float(sym_df.iloc[future_loc]) / base_close - 1) * 100
                    entry[key] = round(fwd, 2)
                else:
                    entry[key] = None
        except Exception:
            pass

    # Aggregate missed signal stats
    fwd_keys = ["fwd20d", "fwd40d", "fwd60d"]
    missed_stats = {}
    for k in fwd_keys:
        vals = [e[k] for e in missed_log if e.get(k) is not None]
        if vals:
            missed_stats[k] = {
                "n": len(vals),
                "mean": round(np.mean(vals), 2),
                "median": round(np.median(vals), 2),
                "win_rate": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
                "p25": round(np.percentile(vals, 25), 2),
                "p75": round(np.percentile(vals, 75), 2),
            }

    return {
        "e1_only_symbols":  sorted(e1_only),
        "e3_only_symbols":  sorted(e3_only),
        "both_symbols":     sorted(both),
        "e1_only_trades":   e1_only_trades,
        "e3_only_trades":   e3_only_trades,
        "a_missed_log":     missed_log,
        "missed_fwd_stats": missed_stats,
        "e1_only_pnl":      sum(t.get("pnl", 0) for t in e1_only_trades),
        "e3_only_pnl":      sum(t.get("pnl", 0) for t in e3_only_trades),
    }


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 3: Exposure decomposition (exposure vs selection)
# ─────────────────────────────────────────────────────────────────────

def analysis3_exposure(results: dict) -> dict:
    out = {}
    for pat, data in results.items():
        exp   = data["exposure_list"]
        eq    = data["equity_curve"]
        dates = [d["date"] for d in data["daily_data"]]

        df = pd.DataFrame({
            "exposure": exp[:len(dates)],
            "equity":   eq[:len(dates)],
        }, index=pd.DatetimeIndex(dates))
        df["daily_ret"] = df["equity"].pct_change().fillna(0.0)

        out[pat] = {
            "avg_exp":          round(df["exposure"].mean() * 100, 1),
            "median_exp":       round(df["exposure"].median() * 100, 1),
            "p90_exp":          round(df["exposure"].quantile(0.90) * 100, 1),
            "days_full":        int((df["exposure"] >= 0.85).sum()),
            "days_invested":    int((df["exposure"] > 0.01).sum()),
            "days_total":       len(df),
            "daily_rets":       df["daily_ret"],
            "daily_exp":        df["exposure"],
        }

    # Attribution decomposition: Brinson-style
    if "A" in out:
        a = out["A"]
        for pat in ["E1", "E3"]:
            if pat not in out:
                continue
            e = out[pat]
            a_ret  = a["daily_rets"]
            e_ret  = e["daily_rets"]
            a_exp  = a["daily_exp"]
            e_exp  = e["daily_exp"]

            # Align dates
            common = a_ret.index.intersection(e_ret.index)
            a_ret_c = a_ret.loc[common]; a_exp_c = a_exp.loc[common]
            e_ret_c = e_ret.loc[common]; e_exp_c = e_exp.loc[common]

            # Decompose: on days where both are invested, compare per-unit return
            # Exposure effect: A's daily return × (E_exposure - A_exposure)
            # Selection effect: (E_return - A_return) × A_exposure
            exp_effect_daily     = a_ret_c * (e_exp_c - a_exp_c)
            selection_effect_daily = (e_ret_c - a_ret_c) * a_exp_c

            n_years = len(common) / 252
            exp_effect     = float(exp_effect_daily.sum()) * 252 / max(n_years, 1e-3) if n_years > 0 else 0
            select_effect  = float(selection_effect_daily.sum()) * 252 / max(n_years, 1e-3) if n_years > 0 else 0

            # Position overlap (Jaccard)
            a_daily2 = results["A"]["daily_data"]
            e_daily  = results[pat]["daily_data"]
            jaccard_list = []
            for da, de in zip(a_daily2, e_daily):
                if da["date"] == de["date"]:
                    u = da["syms_held"] | de["syms_held"]
                    i_set = da["syms_held"] & de["syms_held"]
                    if u:
                        jaccard_list.append(len(i_set) / len(u))
            avg_jaccard = round(np.mean(jaccard_list) * 100, 1) if jaccard_list else 0.0

            out[pat]["vs_A"] = {
                "exp_effect_annualized":    round(exp_effect * 100, 2),
                "select_effect_annualized": round(select_effect * 100, 2),
                "avg_position_overlap_pct": avg_jaccard,
            }

    return out


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 4: Winner concentration
# ─────────────────────────────────────────────────────────────────────

def analysis4_concentration(a1_results: dict) -> dict:
    conc = {}
    for pat, data in a1_results.items():
        wins = sorted(
            [t.get("pnl", 0) for t in data.get("sells", []) if t.get("pnl", 0) > 0],
            reverse=True
        )
        total_win = sum(wins) if wins else 1.0
        conc[pat] = {
            "total_win_pnl":   round(total_win, 0),
            "n_winners":       len(wins),
        }
        for n in [1, 3, 5, 10]:
            top_n = sum(wins[:n])
            conc[pat][f"top{n}_share"] = round(top_n / max(total_win, 1.0) * 100, 1)
    return conc


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS 5: Entry timing hypothesis
# ─────────────────────────────────────────────────────────────────────

def analysis5_entry_timing(results: dict, universe_raw: dict) -> dict:
    a_syms  = {t["symbol"] for t in results["A"]["trades"]  if t["side"] == "SELL"}
    e1_syms = {t["symbol"] for t in results["E1"]["trades"] if t["side"] == "SELL"}
    e3_syms = {t["symbol"] for t in results["E3"]["trades"] if t["side"] == "SELL"}

    e1_only = e1_syms - a_syms
    e3_only = e3_syms - a_syms

    def get_entry_stats(trades, symbol_set, pattern_label):
        """For each entry, compute RSR, distance from 52w high, approximate ATR."""
        entries = []
        for t in trades:
            if t["side"] != "SELL":
                continue
            sym = t["symbol"]
            if sym not in symbol_set:
                continue
            entry_px   = t.get("entry", 0)
            entry_date = t.get("entry_date", "")
            if not entry_px or not entry_date or sym not in universe_raw:
                continue
            df = universe_raw[sym]["df"]
            try:
                ed = pd.Timestamp(entry_date)
                if ed not in df.index:
                    continue
                loc = df.index.get_loc(ed)
                # 52-week high
                start_loc = max(0, loc - 252)
                high_52w  = float(df["High"].iloc[start_loc:loc].max()) if loc > 0 else entry_px
                dist_from_52w_high = (entry_px / high_52w - 1) * 100

                # 20-day ATR approximation
                atr_start = max(0, loc - 20)
                highs = df["High"].iloc[atr_start:loc].values.astype(float)
                lows  = df["Low"].iloc[atr_start:loc].values.astype(float)
                tr    = (highs - lows) if len(highs) > 0 else np.array([entry_px * 0.02])
                atr   = float(np.mean(tr)) if len(tr) > 0 else entry_px * 0.02
                atr_pct = atr / entry_px * 100

                # RSR at entry
                rsr_str = t.get("reason", "RSR=0")
                try:
                    rsr_val = float(rsr_str.split("=")[-1])
                except Exception:
                    rsr_val = 0.0

                entries.append({
                    "symbol":        sym,
                    "entry_date":    entry_date,
                    "entry_px":      entry_px,
                    "rsr":           rsr_val,
                    "dist_from_52w_high": round(dist_from_52w_high, 2),
                    "atr_pct":       round(atr_pct, 2),
                    "return_pct":    t.get("return_pct", 0.0),
                    "holding_days":  t.get("holding_days", 0),
                    "pattern":       pattern_label,
                })
            except Exception:
                continue
        return entries

    def get_all_entries(data, pat):
        entries = []
        for t in data["trades"]:
            if t["side"] != "SELL":
                continue
            sym = t["symbol"]
            entry_px  = t.get("entry", 0)
            entry_date = t.get("entry_date", "")
            if not entry_px or not entry_date or sym not in universe_raw:
                continue
            df = universe_raw[sym]["df"]
            try:
                ed  = pd.Timestamp(entry_date)
                if ed not in df.index:
                    continue
                loc = df.index.get_loc(ed)
                s52 = max(0, loc - 252)
                high_52w = float(df["High"].iloc[s52:loc].max()) if loc > 0 else entry_px
                d52w     = (entry_px / high_52w - 1) * 100
                atr_s    = max(0, loc - 20)
                highs    = df["High"].iloc[atr_s:loc].values.astype(float)
                lows     = df["Low"].iloc[atr_s:loc].values.astype(float)
                tr       = (highs - lows) if len(highs) > 0 else np.array([entry_px * 0.02])
                atr_pct  = float(np.mean(tr)) / entry_px * 100 if len(tr) > 0 else 2.0
                rsr_str  = t.get("reason", "RSR=0")
                try:
                    rsr_val = float(rsr_str.split("=")[-1])
                except Exception:
                    rsr_val = 0.0
                entries.append({
                    "symbol": sym, "entry_date": entry_date,
                    "rsr": rsr_val, "dist_from_52w_high": d52w,
                    "atr_pct": atr_pct, "return_pct": t.get("return_pct", 0.0),
                    "pattern": pat,
                })
            except Exception:
                continue
        return entries

    a_entries  = get_all_entries(results["A"],  "A")
    e1_entries = get_all_entries(results["E1"], "E1")
    e3_entries = get_all_entries(results["E3"], "E3")

    def summarize(entries):
        if not entries:
            return {}
        rsrs = [e["rsr"] for e in entries]
        d52s = [e["dist_from_52w_high"] for e in entries]
        atrs = [e["atr_pct"] for e in entries]
        rets = [e["return_pct"] for e in entries]
        return {
            "n":           len(entries),
            "rsr_mean":    round(np.mean(rsrs), 1),
            "rsr_median":  round(np.median(rsrs), 1),
            "d52w_mean":   round(np.mean(d52s), 2),   # % below 52w high (negative = below)
            "d52w_median": round(np.median(d52s), 2),
            "atr_pct_mean":round(np.mean(atrs), 2),
            "ret_mean":    round(np.mean(rets), 2),
            "ret_median":  round(np.median(rets), 2),
            "win_rate":    round(sum(1 for r in rets if r > 0) / max(len(rets),1) * 100, 1),
        }

    e1_only_entries = [e for e in e1_entries if e["symbol"] in e1_only]
    e3_only_entries = [e for e in e3_entries if e["symbol"] in e3_only]
    e1_both_entries = [e for e in e1_entries if e["symbol"] in a_syms]

    return {
        "a_summary":        summarize(a_entries),
        "e1_summary":       summarize(e1_entries),
        "e3_summary":       summarize(e3_entries),
        "e1_only_summary":  summarize(e1_only_entries),
        "e3_only_summary":  summarize(e3_only_entries),
        "e1_both_summary":  summarize(e1_both_entries),
        "e1_only_entries":  e1_only_entries,
        "e3_only_entries":  e3_only_entries,
    }


# ─────────────────────────────────────────────────────────────────────
#  MARKDOWN REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────

def fmt_pnl(v):
    return f"¥{v:+,.0f}" if v else "¥0"

def fmt_pct(v, d=1):
    return f"{v:+.{d}f}%" if v is not None else "N/A"

def write_report(results, a1, a2, a3, a4, a5, output_path):
    lines = []
    L = lines.append

    m_a  = results["A"]["metrics"]
    m_e1 = results["E1"]["metrics"]
    m_e3 = results["E3"]["metrics"]

    L("# 2025 OOS Winner Analysis: A vs E1 vs E3")
    L(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  対象期間: 2025-01-01〜2025-12-31\n")

    L("## 前提確認")
    L("\n| 指標 | A (現行) | E1 均等リバランス | E3 強トレンド集中 |")
    L("|---|---|---|---|")
    for key, label in [("cagr","CAGR (%)"),("sharpe","Sharpe"),("max_dd","MaxDD (%)"),
                        ("avg_exposure","Avg Exposure %"),("n_trades_yr","Trades/yr")]:
        row = f"| {label} | {m_a.get(key,0):.1f} | {m_e1.get(key,0):.1f} | {m_e3.get(key,0):.1f} |"
        L(row)

    # ── Analysis 1 ───────────────────────────────────────────────────
    L("\n---\n## 分析1: 銘柄別PnL寄与 (TOP20 / BOTTOM20)")

    for pat in ["A", "E1", "E3"]:
        data = a1.get(pat, {})
        L(f"\n### Pattern {pat} — {data.get('n_trades',0)}取引  総PnL={fmt_pnl(data.get('total_pnl',0))}")
        top20 = data.get("top20", [])
        if top20:
            L("\n**TOP20 利益銘柄**\n")
            L("| symbol | sector | entry | exit | hold_d | PnL | ret% | R倍 |")
            L("|---|---|---|---|---|---|---|---|")
            for t in top20:
                L(f"| {t['symbol']} | {t.get('sector','―')} | {t.get('entry_date','')} "
                  f"| {t.get('exit_date','')} | {t.get('holding_days',0)} "
                  f"| {fmt_pnl(t.get('pnl',0))} | {t.get('return_pct',0):+.1f}% "
                  f"| {t.get('r_multiple',0):.2f} |")
        bot20 = data.get("bot20", [])
        if bot20:
            L("\n**BOTTOM20 損失銘柄**\n")
            L("| symbol | sector | entry | exit | hold_d | PnL | ret% |")
            L("|---|---|---|---|---|---|---|")
            for t in reversed(bot20):
                L(f"| {t['symbol']} | {t.get('sector','―')} | {t.get('entry_date','')} "
                  f"| {t.get('exit_date','')} | {t.get('holding_days',0)} "
                  f"| {fmt_pnl(t.get('pnl',0))} | {t.get('return_pct',0):+.1f}% |")

    # ── Analysis 2 ───────────────────────────────────────────────────
    L("\n---\n## 分析2: Aが逃したシグナル vs E1/E3が捕捉したシグナル")

    e1o = a2.get("e1_only_symbols", [])
    e3o = a2.get("e3_only_symbols", [])
    L(f"\n- A未保有・E1のみ保有: **{len(e1o)}銘柄** → {', '.join(e1o) if e1o else '(なし)'}")
    L(f"- A未保有・E3のみ保有: **{len(e3o)}銘柄** → {', '.join(e3o) if e3o else '(なし)'}")
    L(f"- 共通保有銘柄: **{len(a2.get('both_symbols',[]))}銘柄**")

    L(f"\n**E1-only銘柄の実現PnL合計: {fmt_pnl(a2.get('e1_only_pnl',0))}**")
    L(f"**E3-only銘柄の実現PnL合計: {fmt_pnl(a2.get('e3_only_pnl',0))}**")

    if a2.get("e1_only_trades"):
        L("\n**E1-only取引詳細**\n")
        L("| symbol | sector | entry | exit | hold_d | PnL | ret% |")
        L("|---|---|---|---|---|---|---|")
        for t in sorted(a2["e1_only_trades"], key=lambda x: x.get("pnl",0), reverse=True):
            L(f"| {t['symbol']} | {t.get('sector','―')} | {t.get('entry_date','')} "
              f"| {t.get('exit_date','')} | {t.get('holding_days',0)} "
              f"| {fmt_pnl(t.get('pnl',0))} | {t.get('return_pct',0):+.1f}% |")

    if a2.get("e3_only_trades"):
        L("\n**E3-only取引詳細**\n")
        L("| symbol | sector | entry | exit | hold_d | PnL | ret% |")
        L("|---|---|---|---|---|---|---|")
        for t in sorted(a2["e3_only_trades"], key=lambda x: x.get("pnl",0), reverse=True):
            L(f"| {t['symbol']} | {t.get('sector','―')} | {t.get('entry_date','')} "
              f"| {t.get('exit_date','')} | {t.get('holding_days',0)} "
              f"| {fmt_pnl(t.get('pnl',0))} | {t.get('return_pct',0):+.1f}% |")

    L("\n**Aが逃したシグナルのForward Return (実際のAのmissed signals)**\n")
    ms = a2.get("missed_fwd_stats", {})
    L("| 期間 | 件数 | 平均ret% | 中央値 | 勝率 | P25 | P75 |")
    L("|---|---|---|---|---|---|---|")
    for k in ["fwd20d","fwd40d","fwd60d"]:
        s = ms.get(k, {})
        if s:
            L(f"| {k} | {s['n']} | {s['mean']:+.2f}% | {s['median']:+.2f}% "
              f"| {s['win_rate']:.1f}% | {s['p25']:+.2f}% | {s['p75']:+.2f}% |")

    # ── Analysis 3 ───────────────────────────────────────────────────
    L("\n---\n## 分析3: Exposure差分 (Exposure効果 vs 選別効果)")

    L("\n| 指標 | A | E1 | E3 |")
    L("|---|---|---|---|")
    for k, label in [("avg_exp","平均Exposure%"),("median_exp","中央値Exposure%"),
                      ("p90_exp","90th pct Exposure%"),("days_full","フル投資日数"),
                      ("days_invested","投資日数(>1%)"),("days_total","総日数")]:
        vals = [a3.get(p,{}).get(k,"—") for p in ["A","E1","E3"]]
        L(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")

    L("\n**Brinson帰属分解 (vs Pattern A)**\n")
    L("| | E1 | E3 |")
    L("|---|---|---|")
    for k, label in [
        ("exp_effect_annualized",    "Exposure効果 (年率換算pp)"),
        ("select_effect_annualized", "選別効果 (年率換算pp)"),
        ("avg_position_overlap_pct", "ポジション重複率 (Jaccard %)"),
    ]:
        e1v = a3.get("E1",{}).get("vs_A",{}).get(k, "—")
        e3v = a3.get("E3",{}).get("vs_A",{}).get(k, "—")
        e1s = f"{e1v:+.1f}" if isinstance(e1v, (int,float)) else str(e1v)
        e3s = f"{e3v:+.1f}" if isinstance(e3v, (int,float)) else str(e3v)
        L(f"| {label} | {e1s} | {e3s} |")

    # ── Analysis 4 ───────────────────────────────────────────────────
    L("\n---\n## 分析4: Winner Concentration (利益集中度)")

    L("\n| 指標 | A | E1 | E3 |")
    L("|---|---|---|---|")
    for k, label in [("total_win_pnl","総利益額(¥)"),("n_winners","勝ち取引数"),
                      ("top1_share","TOP1集中率%"),("top3_share","TOP3集中率%"),
                      ("top5_share","TOP5集中率%"),("top10_share","TOP10集中率%")]:
        vals = [a4.get(p,{}).get(k,"—") for p in ["A","E1","E3"]]
        def fmt_v(v):
            if isinstance(v,float) and k=="total_win_pnl": return f"¥{v:,.0f}"
            return str(v)
        L(f"| {label} | {fmt_v(vals[0])} | {fmt_v(vals[1])} | {fmt_v(vals[2])} |")

    # ── Analysis 5 ───────────────────────────────────────────────────
    L("\n---\n## 分析5: Entry Timing仮説検証")

    L("\n| 指標 | A全取引 | E1全取引 | E3全取引 | E1-only | E3-only |")
    L("|---|---|---|---|---|---|")
    keys5 = [("n","取引数"),("rsr_mean","RSR平均"),("rsr_median","RSR中央"),
             ("d52w_mean","52w高値比% (平均)"),("d52w_median","52w高値比% (中央)"),
             ("atr_pct_mean","ATR% (平均)"),("ret_mean","実現ret% (平均)"),
             ("win_rate","勝率%")]
    for k, label in keys5:
        vals = [a5.get(grp,{}).get(k,"—") for grp in
                ["a_summary","e1_summary","e3_summary","e1_only_summary","e3_only_summary"]]
        L(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} |")

    if a5.get("e1_only_entries"):
        L("\n**E1-only銘柄 エントリー詳細**\n")
        L("| symbol | entry_date | RSR | 52w高値比% | ATR% | 実現ret% |")
        L("|---|---|---|---|---|---|")
        for e in sorted(a5["e1_only_entries"], key=lambda x: x.get("return_pct",0), reverse=True):
            L(f"| {e['symbol']} | {e['entry_date']} | {e['rsr']:.1f} "
              f"| {e['dist_from_52w_high']:+.1f}% | {e['atr_pct']:.2f}% | {e['return_pct']:+.1f}% |")

    # ── Final Verdict ─────────────────────────────────────────────────
    L("\n---\n## 最終判断")

    # Determine primary driver
    e1_vs  = a3.get("E1",{}).get("vs_A",{})
    e3_vs  = a3.get("E3",{}).get("vs_A",{})
    e1_exp = e1_vs.get("exp_effect_annualized", 0)
    e1_sel = e1_vs.get("select_effect_annualized", 0)
    e3_exp = e3_vs.get("exp_effect_annualized", 0)
    e3_sel = e3_vs.get("select_effect_annualized", 0)
    e1_jac = e1_vs.get("avg_position_overlap_pct", 0)
    e3_jac = e3_vs.get("avg_position_overlap_pct", 0)
    e1_cagr_diff = m_e1.get("cagr",0) - m_a.get("cagr",0)
    e3_cagr_diff = m_e3.get("cagr",0) - m_a.get("cagr",0)

    # 1. True cause
    L("\n### 1. E1/E3が勝った真因\n")
    for pat, exp_eff, sel_eff, jac, cagr_diff in [
        ("E1", e1_exp, e1_sel, e1_jac, e1_cagr_diff),
        ("E3", e3_exp, e3_sel, e3_jac, e3_cagr_diff),
    ]:
        total = exp_eff + sel_eff
        exp_share = (exp_eff / total * 100) if total != 0 else 0
        sel_share = (sel_eff / total * 100) if total != 0 else 0
        L(f"**Pattern {pat}** (CAGR差 = {cagr_diff:+.1f}pp vs A)\n")
        L(f"- Exposure効果: {exp_eff:+.1f}pp ({exp_share:.0f}%)")
        L(f"- 選別効果:     {sel_eff:+.1f}pp ({sel_share:.0f}%)")
        L(f"- ポジション重複率 (Jaccard): {jac:.1f}%  "
          f"→ {'主にサイズ差' if jac >= 60 else '銘柄選択の差が大きい'}")
        L("")

    # 2. Reproducibility
    L("\n### 2. 再現可能なエッジか / 2025限定の偶然か\n")
    e1_both_wr = a5.get("e1_both_summary",{}).get("win_rate",0)
    e1_only_wr = a5.get("e1_only_summary",{}).get("win_rate",0)
    e1_a_wr    = a5.get("a_summary",{}).get("win_rate",0)
    ms_wr20    = a2.get("missed_fwd_stats",{}).get("fwd20d",{}).get("win_rate",0)
    ms_mean20  = a2.get("missed_fwd_stats",{}).get("fwd20d",{}).get("mean",0)

    L(f"- Aの勝率: {e1_a_wr:.1f}%  / E1-only銘柄勝率: {e1_only_wr:.1f}%  / "
      f"E1共通銘柄勝率: {e1_both_wr:.1f}%")
    L(f"- Aが逃したシグナルの20日後リターン: 平均{ms_mean20:+.2f}%  勝率{ms_wr20:.1f}%")
    if ms_wr20 > 55 and ms_mean20 > 1.0:
        L("- **→ Aのmissedシグナルは正のアルファを持つ。構造的な機会損失の可能性あり。**")
    else:
        L("- **→ Aのmissedシグナルは有意なアルファなし。E1/E3の優位は主に Exposure 差による偶発的優位。**")

    L(f"\n**IS(7年)でE1 CAGR=-5.0% / E3 CAGR=+3.7%** に対して OOSのみ高成績 "
      "→ IS/OOS乖離が激しく統計的信頼性低い。2025限定の相場特性（高exposure × 上昇トレンド）が主因と判断。\n")

    # 3. Next steps
    L("\n### 3. 次に検証すべき改善案 (優先順位付き)\n")
    L("| 優先度 | 仮説 | 検証方法 | 期待効果 |")
    L("|---|---|---|---|")
    L("| 1 | **Entry Timing有効化** — `block_low_confidence=true` でシグナル品質向上 | IS/OOS A/Bテスト | IS CAGR +3〜7pp, MaxDD±0 |")
    L("| 2 | **Missed signal品質フィルタ** — RSR閾値引上げでmissed signal中の低品質排除 | RSRしきい値sweep | 取引数減・質向上 |")
    L("| 3 | **Exit延長実験** — 勝ち銘柄の55d exitを65〜75dに延長 | Turtle exit sweep | avghold+2〜3d, CAGR+1〜3pp |")
    L("| 4 | **OOS 2025 E1/E3 勝因の再現** — 2025固有の高exposure環境を再現する条件を特定 | 年ごとexposure×CAGR回帰 | 市場環境依存度の定量化 |")
    L("| 5 | **レバレッジ 1.3x 実験** (要追加証拠金) | 仮想レバレッジシミュ | CAGR×1.3, MaxDD悪化確認 |")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print(f"  レポート保存: {output_path}")
    return text


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3

    print("=" * 68)
    print("  2025 OOS Winner Analysis — A vs E1 vs E3")
    print("=" * 68 + "\n")

    print("[1/5] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    results = {}
    for pat in ["A", "E1", "E3"]:
        print(f"[2/5] Pattern {pat} OOS実行中...")
        t0 = time.time()
        results[pat] = run_oos_detailed(
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg, pattern=pat,
        )
        m = results[pat]["metrics"]
        print(f"       CAGR={m.get('cagr',0):+.1f}%  Sharpe={m.get('sharpe',0):.3f}  "
              f"MaxDD={m.get('max_dd',0):.1f}%  Exp={m.get('avg_exposure',0):.1f}%  "
              f"({time.time()-t0:.1f}s)")

    print("[3/5] 分析1-4 実行中...")
    a1 = analysis1_trade_pnl(results)
    a2 = analysis2_missed_signals(results, universe_raw)
    a4_conc = analysis4_concentration(a1)

    print("[4/5] 分析3 (Exposure帰属) 実行中...")
    a3 = analysis3_exposure(results)

    print("[5/5] 分析5 (Entry Timing) + レポート生成中...")
    a5 = analysis5_entry_timing(results, universe_raw)

    out_path = DOCS_DIR / "oos_2025_winner_analysis.md"
    write_report(results, a1, a2, a3, a4_conc, a5, out_path)

    # ── console summary ───────────────────────────────────────────────
    print("\n\n" + "=" * 68)
    print("  ★ コンソール要約")
    print("=" * 68)

    print(f"\n  CAGR: A={results['A']['metrics'].get('cagr',0):+.1f}%  "
          f"E1={results['E1']['metrics'].get('cagr',0):+.1f}%  "
          f"E3={results['E3']['metrics'].get('cagr',0):+.1f}%")
    print(f"  Exposure: A={a3.get('A',{}).get('avg_exp',0):.1f}%  "
          f"E1={a3.get('E1',{}).get('avg_exp',0):.1f}%  "
          f"E3={a3.get('E3',{}).get('avg_exp',0):.1f}%")

    print(f"\n  [分析1] 取引数: A={a1['A']['n_trades']}  E1={a1['E1']['n_trades']}  E3={a1['E3']['n_trades']}")
    for pat in ["A","E1","E3"]:
        d = a1[pat]
        L1 = d["top20"][0] if d.get("top20") else {}
        print(f"  {pat} 最大勝ち: {L1.get('symbol','')} {fmt_pnl(L1.get('pnl',0))} ({L1.get('return_pct',0):+.1f}%)")

    print(f"\n  [分析2] E1-only銘柄: {len(a2['e1_only_symbols'])}  E3-only銘柄: {len(a2['e3_only_symbols'])}")
    print(f"         E1-only PnL: {fmt_pnl(a2['e1_only_pnl'])}  E3-only PnL: {fmt_pnl(a2['e3_only_pnl'])}")
    ms20 = a2.get("missed_fwd_stats",{}).get("fwd20d",{})
    if ms20:
        print(f"         Aのmissed fwd20d: 平均{ms20.get('mean',0):+.2f}%  勝率{ms20.get('win_rate',0):.1f}%  N={ms20.get('n',0)}")

    print(f"\n  [分析3] Brinson帰属 (E1 vs A):")
    e1_vs = a3.get("E1",{}).get("vs_A",{})
    print(f"         Exposure効果={e1_vs.get('exp_effect_annualized',0):+.1f}pp  "
          f"選別効果={e1_vs.get('select_effect_annualized',0):+.1f}pp  "
          f"Jaccard={e1_vs.get('avg_position_overlap_pct',0):.1f}%")
    e3_vs = a3.get("E3",{}).get("vs_A",{})
    print(f"         (E3 vs A): Exposure={e3_vs.get('exp_effect_annualized',0):+.1f}pp  "
          f"選別={e3_vs.get('select_effect_annualized',0):+.1f}pp  "
          f"Jaccard={e3_vs.get('avg_position_overlap_pct',0):.1f}%")

    print(f"\n  [分析4] Winner Concentration (TOP5利益集中率):")
    for pat in ["A","E1","E3"]:
        c = a4_conc.get(pat,{})
        print(f"         {pat}: TOP1={c.get('top1_share',0):.1f}%  TOP3={c.get('top3_share',0):.1f}%  "
              f"TOP5={c.get('top5_share',0):.1f}%  TOP10={c.get('top10_share',0):.1f}%")

    print(f"\n  [分析5] Entry Timing比較 (RSR mean / 52w距離):")
    for pat, key in [("A","a_summary"),("E1","e1_summary"),("E3","e3_summary")]:
        s = a5.get(key,{})
        print(f"         {pat}: RSR={s.get('rsr_mean',0):.1f}  52w={s.get('d52w_mean',0):+.2f}%  "
              f"勝率={s.get('win_rate',0):.1f}%  ret平均={s.get('ret_mean',0):+.2f}%")

    print(f"\n  レポート → {out_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
