"""
backtest/capital_allocation_e1e2e3.py

Dynamic Rebalancing Allocation Study — E1 / E2 / E3 vs Baseline A

Signal gen / entry / exit / RSR ranking: identical to Pattern A
Changed: capital allocation logic only (rebalancing on new signal only)

A   Baseline             現行 (no rebalance)
E1  Equal Rebalancing    N=1:90%  N=2:45/45      N=3:30/30/30
E2  Winner Rebalancing   N=1:90%  N=2:60/30      N=3:50/25/15
E3  Trend Concentration  N=1:90%  N=2:60/30      N=3:60/20/10

Run:
    cd C:/ai-trading
    python src/backtest/capital_allocation_e1e2e3.py
"""

from __future__ import annotations

import sys, json, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, calc_metrics, _sector_ok, _execute_buy, Position, _take,
    IS_START, IS_END, OOS_START, OOS_END,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS, OUTPUT_DIR,
)

# ── E-pattern allocation weight tables ────────────────────────────────
# key = n_positions after new entry; value = weights list (rank0 = top priority)
E1_WEIGHTS = {1: [0.90],        2: [0.45, 0.45],       3: [0.30, 0.30, 0.30]}
E2_WEIGHTS = {1: [0.90],        2: [0.60, 0.30],       3: [0.50, 0.25, 0.15]}
E3_WEIGHTS = {1: [0.90],        2: [0.60, 0.30],       3: [0.60, 0.20, 0.10]}
PATTERN_W  = {"E1": E1_WEIGHTS, "E2": E2_WEIGHTS, "E3": E3_WEIGHTS}


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

def _assign_weights_e(
    pattern: str,
    n: int,
    positions: dict,
    new_sym: str,
    rsr_mat: np.ndarray,
    i: int,
    sym_to_i: dict,
    close_mat: np.ndarray,
) -> dict[str, float]:
    """
    Return {sym: target_weight} for all (existing + new_sym).
    E1: equal weight.
    E2: rank existing by unrealized PnL%; new_sym always lowest rank.
    E3: rank ALL (including new_sym) by current RSR.
    """
    base_w   = PATTERN_W[pattern][n]
    all_syms = list(positions.keys()) + [new_sym]

    if pattern == "E1":
        return {s: base_w[0] for s in all_syms}

    if pattern == "E2":
        def score_e2(sym: str) -> float:
            if sym == new_sym:
                return -1e9          # new entry: always lowest rank
            pos = positions[sym]
            px  = float(close_mat[i, sym_to_i[sym]])
            return (px - pos.entry_price) / max(pos.entry_price, 1.0)
        ranked = sorted(all_syms, key=score_e2, reverse=True)
        return {s: base_w[j] for j, s in enumerate(ranked)}

    # E3: rank by RSR (new_sym competes equally)
    def score_e3(sym: str) -> float:
        si = sym_to_i.get(sym, -1)
        return float(rsr_mat[i, si]) if si >= 0 else 0.0
    ranked = sorted(all_syms, key=score_e3, reverse=True)
    return {s: base_w[j] for j, s in enumerate(ranked)}


def _partial_sell(
    sym: str,
    pos: Position,
    target_val: float,
    open_mat: np.ndarray,
    next_i: int,
    sym_to_i: dict,
    trades: list,
    cur_day_i: int,
) -> tuple[float, float, float]:
    """
    Trim pos to target_val at next open.
    Returns (cash_freed, realized_pnl, winner_trim_pnl).
    Keeps at least 1 LOT. Returns zeros if trim < 1 LOT.
    """
    px = float(open_mat[next_i, sym_to_i[sym]])
    if px <= 0:
        return 0.0, 0.0, 0.0

    current_val = pos.qty * px
    if current_val <= target_val:
        return 0.0, 0.0, 0.0

    excess    = current_val - target_val
    lots_sell = int(excess / px / LOT) * LOT
    max_sell  = pos.qty - LOT             # keep at least 1 LOT
    if max_sell < LOT:
        return 0.0, 0.0, 0.0
    lots_sell = min(lots_sell, max_sell)
    if lots_sell < LOT:
        return 0.0, 0.0, 0.0

    proceeds = lots_sell * px * (1.0 - COST_ONE_WAY)
    rpnl     = (px - pos.entry_price) * lots_sell
    win_trim = rpnl if rpnl > 0 else 0.0

    trades.append({
        "side": "SELL_PARTIAL", "symbol": sym,
        "entry": pos.entry_price, "exit": px,
        "qty": lots_sell, "pnl": rpnl,
        "entry_idx": pos.entry_idx, "exit_idx": cur_day_i,
        "reason": "REBALANCE_TRIM", "partial": True,
    })
    pos.qty -= lots_sell
    return proceeds, rpnl, win_trim


# ─────────────────────────────────────────────────────────────────────
#  CORE BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────

def run_period_e(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    start: str, end: str,
    pattern: str,        # "A" | "E1" | "E2" | "E3"
) -> dict:
    """
    Single-period backtest for one allocation pattern.
    A = baseline (no rebalance, cash/slots sizing).
    E1/E2/E3 = dynamic rebalancing on new signal arrival.
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

    # ── strategy objects ─────────────────────────────────────────────
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

    # ── common date range ─────────────────────────────────────────────
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

    # ── price / signal matrices ────────────────────────────────────────
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

    # TOPIX arrays
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

    # ── state ─────────────────────────────────────────────────────────
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
    new_signal_entries= 0
    position_conc: list[float] = []

    # ─────────────────────────────────────────────────────────────────
    for i, date in enumerate(common_dates):
        invested = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
                       for s, pos in positions.items())
        cur_equity = cash + invested
        equity_curve.append(cur_equity)
        exposure_list.append(invested / max(1.0, cur_equity))

        # daily max-position concentration
        if positions:
            pos_vals = [pos.qty * float(close_mat[i, sym_to_i[s]])
                        for s, pos in positions.items() if s in sym_to_i]
            position_conc.append(max(pos_vals) / max(cur_equity, 1.0) if pos_vals else 0.0)
        else:
            position_conc.append(0.0)

        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        # CB
        if not cb_active:
            if dd <= -max_dd_limit:
                cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0
        cb_scale = CB_SCALE if cb_active else 1.0

        # gross cap
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
        # PATTERN-SPECIFIC BUY LOGIC
        # ─────────────────────────────────────────────────────────────

        if pattern == "A":
            # Baseline: cash/effective_slots, max_single_weight cap
            for rsr_val, sym in buy_cands:
                open_slots = max_pos - len(positions)
                if open_slots <= 0:
                    missed_count += len(buy_cands) - buy_cands.index((rsr_val, sym))
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

        else:
            # E1 / E2 / E3: rebalance existing on new signal
            for bc_idx, (rsr_val, sym) in enumerate(buy_cands):
                if len(positions) >= max_pos:
                    missed_count += len(buy_cands) - bc_idx
                    break

                buy_px = float(open_mat[next_i, sym_to_i[sym]])
                if buy_px <= 0:
                    continue
                if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                    continue

                # Under market stress: skip if stress gross_cap < 85% target
                if gross_enabled and gross_cap < 0.85:
                    blocked_gross += 1
                    continue

                # Current equity before rebalance
                cur_eq = cash + sum(p.qty * float(close_mat[i, sym_to_i[s]])
                                    for s, p in positions.items())

                # Compute target weights (new N = existing + 1)
                n_new = len(positions) + 1
                tw    = _assign_weights_e(pattern, n_new, positions, sym,
                                          rsr_mat, i, sym_to_i, close_mat)

                # Partial sell existing positions that exceed their new target
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

                # Equity after partial sells
                cur_eq = cash + sum(p.qty * float(close_mat[i, sym_to_i[s]])
                                    for s, p in positions.items())

                # Buy new at target weight
                alloc = min(cur_eq * tw[sym] * cb_scale, cash * 0.99)
                qty   = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)
                new_signal_entries += 1

    # ── metrics ───────────────────────────────────────────────────────
    metrics = calc_metrics(equity_curve, trades, exposure_list, capital,
                           list(common_dates))
    metrics["exit_reasons"]       = exit_counts
    metrics["missed_signals"]     = missed_count
    metrics["blocked_gross"]      = blocked_gross
    metrics["rebalance_count"]    = rebalance_count
    metrics["rebalance_trim_pnl"] = round(rebalance_trim_pnl, 0)
    metrics["winner_trim_pnl"]    = round(winner_trim_pnl, 0)
    metrics["new_signal_entries"] = new_signal_entries
    metrics["avg_concentration"]  = (
        round(float(np.mean(position_conc)) * 100, 1) if position_conc else 0.0
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3, \
        f"max_positions={cfg.portfolio.max_positions} — must be 3 (PARAMS_LOCKED)"

    print("=" * 72)
    print("  Dynamic Rebalancing Allocation Study — E1 / E2 / E3 vs A")
    print(f"  IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
    print(f"  max_pos={cfg.portfolio.max_positions}  capital=¥{cfg.portfolio.capital:,.0f}")
    print("=" * 72 + "\n")

    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    patterns = ["A", "E1", "E2", "E3"]
    labels   = {
        "A":  "A Baseline            (現行)",
        "E1": "E1 Equal Rebalancing  (均等リバランス)",
        "E2": "E2 Winner Rebalancing (勝者優遇)",
        "E3": "E3 Trend Concentration(強トレンド集中)",
    }

    results: dict[str, dict] = {}

    for pattern in patterns:
        print(f"\n{'='*60}")
        print(f"  パターン {labels[pattern]}")
        print(f"{'='*60}")

        print("  [IS 2018-2024] 実行中...")
        t0    = time.time()
        is_r  = run_period_e(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                             topix_close, rsr_syms, cfg,
                             start=IS_START, end=IS_END, pattern=pattern)
        is_t  = time.time() - t0

        print("  [OOS 2025]     実行中...")
        t0    = time.time()
        oos_r = run_period_e(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                             topix_close, rsr_syms, cfg,
                             start=OOS_START, end=OOS_END, pattern=pattern)
        oos_t = time.time() - t0

        results[pattern] = {"IS": is_r, "OOS": oos_r}

        def _p(label, key, fmt="{:.1f}"):
            iv = is_r.get(key, "-");  ov = oos_r.get(key, "-")
            iv_s = fmt.format(iv) if isinstance(iv, (int, float)) else str(iv)
            ov_s = fmt.format(ov) if isinstance(ov, (int, float)) else str(ov)
            print(f"    {label:<26} IS={iv_s:>8}   OOS={ov_s:>8}")

        print(f"\n  ── 収益性 ─────────────────────────────────────────────")
        _p("CAGR (%)",         "cagr")
        _p("Profit Factor",    "profit_factor", "{:.3f}")
        _p("Win Rate (%)",     "win_rate")
        print(f"  ── リスク ─────────────────────────────────────────────")
        _p("Sharpe",           "sharpe", "{:.3f}")
        _p("Sortino",          "sortino", "{:.3f}")
        _p("MaxDD (%)",        "max_dd")
        _p("Calmar",           "calmar", "{:.3f}")
        print(f"  ── 資本効率 ───────────────────────────────────────────")
        _p("Avg Exposure (%)", "avg_exposure")
        _p("Avg Concentration","avg_concentration")
        _p("Idle Cash (%)",    "idle_cash")
        _p("Turnover/yr",      "turnover_yr", "{:.2f}")
        print(f"  ── 売買構造 ───────────────────────────────────────────")
        _p("Trades/yr",        "n_trades_yr")
        _p("Avg Hold Days",    "avg_hold_days")
        print(f"    {'Rebalance events':<26} IS={is_r.get('rebalance_count',0):>8}   "
              f"OOS={oos_r.get('rebalance_count',0):>8}")
        print(f"    {'Missed signals':<26} IS={is_r.get('missed_signals',0):>8}   "
              f"OOS={oos_r.get('missed_signals',0):>8}")
        print(f"    {'New entries via rebal':<26} IS={is_r.get('new_signal_entries',0):>8}   "
              f"OOS={oos_r.get('new_signal_entries',0):>8}")
        print(f"    {'Rebal trim PnL (¥)':<26} IS={is_r.get('rebalance_trim_pnl',0):>8,.0f}   "
              f"OOS={oos_r.get('rebalance_trim_pnl',0):>8,.0f}")
        print(f"    {'Winner trim PnL (¥)':<26} IS={is_r.get('winner_trim_pnl',0):>8,.0f}   "
              f"OOS={oos_r.get('winner_trim_pnl',0):>8,.0f}")

        print(f"\n  IS 年次リターン:")
        for yr in sorted(is_r.get("annual_returns", {}).keys()):
            print(f"    {yr}: {is_r['annual_returns'][yr]:+.1f}%")
        print(f"  ({is_t:.1f}s IS / {oos_t:.1f}s OOS)")

    # ─────────────────────────────────────────────────────────────────
    # COMPARATIVE SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  ★ COMPARATIVE SUMMARY")
    print("=" * 80)

    hdr = f"  {'指標':<28} {'A':>10} {'E1':>10} {'E2':>10} {'E3':>10}"
    print(hdr)
    print("  " + "-" * 70)

    def row(label, key, period, fmt="{:.1f}"):
        vals = []
        for p in patterns:
            v = results[p][period].get(key, "-")
            vals.append(fmt.format(v) if isinstance(v, (int, float)) else str(v))
        print(f"  {label:<28} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    print(f"\n  IS 2018-2024")
    row("CAGR (%)",            "cagr",            "IS")
    row("Sharpe",              "sharpe",           "IS", "{:.3f}")
    row("Sortino",             "sortino",          "IS", "{:.3f}")
    row("MaxDD (%)",           "max_dd",           "IS")
    row("Calmar",              "calmar",           "IS", "{:.3f}")
    row("Profit Factor",       "profit_factor",    "IS", "{:.3f}")
    row("Win Rate (%)",        "win_rate",         "IS")
    row("Avg Hold Days",       "avg_hold_days",    "IS")
    row("Avg Exposure (%)",    "avg_exposure",     "IS")
    row("Avg Concentration (%)", "avg_concentration","IS")
    row("Idle Cash (%)",       "idle_cash",        "IS")
    row("Turnover/yr",         "turnover_yr",      "IS", "{:.2f}")
    row("Trades/yr",           "n_trades_yr",      "IS")
    row("Rebalance events",    "rebalance_count",  "IS", "{:.0f}")
    row("Missed signals",      "missed_signals",   "IS", "{:.0f}")
    row("New entries (rebal)", "new_signal_entries","IS","{:.0f}")

    print(f"\n  OOS 2025")
    row("CAGR (%)",            "cagr",            "OOS")
    row("Sharpe",              "sharpe",          "OOS", "{:.3f}")
    row("MaxDD (%)",           "max_dd",          "OOS")
    row("Avg Exposure (%)",    "avg_exposure",    "OOS")
    row("Rebalance events",    "rebalance_count", "OOS", "{:.0f}")

    # ─────────────────────────────────────────────────────────────────
    # IS ANNUAL RETURNS TABLE
    # ─────────────────────────────────────────────────────────────────
    print(f"\n  IS 年次リターン (%)")
    print(f"  {'年':<8} {'A':>10} {'E1':>10} {'E2':>10} {'E3':>10}")
    print("  " + "-" * 48)
    all_years = sorted(results["A"]["IS"].get("annual_returns", {}).keys())
    for yr in all_years:
        vals = [results[p]["IS"]["annual_returns"].get(yr, 0.0) for p in patterns]
        print(f"  {yr:<8} {vals[0]:>+10.1f} {vals[1]:>+10.1f} {vals[2]:>+10.1f} {vals[3]:>+10.1f}")

    # ─────────────────────────────────────────────────────────────────
    # DELTA ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  ★ 差分分析 (vs Pattern A)")
    print("=" * 80)

    def delta(key, pa, pb, period, fmt="{:+.1f}"):
        va = results[pa][period].get(key, 0)
        vb = results[pb][period].get(key, 0)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            return fmt.format(vb - va)
        return "n/a"

    for e_pat, e_label in [("E1","① A→E1"), ("E2","② A→E2"), ("E3","③ A→E3")]:
        print(f"\n  {e_label}  ({labels[e_pat].split('(')[0].strip()})")
        print(f"    IS  CAGR:          {delta('cagr',         'A', e_pat, 'IS')} pp")
        print(f"    IS  Sharpe:        {delta('sharpe',       'A', e_pat, 'IS', '{:+.3f}')}")
        print(f"    IS  MaxDD:         {delta('max_dd',       'A', e_pat, 'IS')} pp")
        print(f"    IS  Calmar:        {delta('calmar',       'A', e_pat, 'IS', '{:+.3f}')}")
        print(f"    IS  Exposure:      {delta('avg_exposure', 'A', e_pat, 'IS')} pp")
        print(f"    IS  Concentration: {delta('avg_concentration','A',e_pat,'IS')} pp")
        print(f"    IS  Turnover:      {delta('turnover_yr',  'A', e_pat, 'IS', '{:+.2f}')}")
        print(f"    OOS CAGR:          {delta('cagr',         'A', e_pat, 'OOS')} pp")
        print(f"    OOS Sharpe:        {delta('sharpe',       'A', e_pat, 'OOS', '{:+.3f}')}")
        print(f"    OOS MaxDD:         {delta('max_dd',       'A', e_pat, 'OOS')} pp")

        # CAGR per 1pp exposure (efficiency metric)
        exp_gain_is  = results[e_pat]["IS"].get("avg_exposure",0) - results["A"]["IS"].get("avg_exposure",0)
        cagr_gain_is = results[e_pat]["IS"].get("cagr",0)         - results["A"]["IS"].get("cagr",0)
        exp_gain_oos = results[e_pat]["OOS"].get("avg_exposure",0)- results["A"]["OOS"].get("avg_exposure",0)
        cagr_gain_oos= results[e_pat]["OOS"].get("cagr",0)        - results["A"]["OOS"].get("cagr",0)

        if abs(exp_gain_is) > 0.1:
            eff_is = cagr_gain_is / exp_gain_is
            print(f"    IS  CAGR/Exp 1pp:  {eff_is:+.3f}  (Exp{exp_gain_is:+.1f}pp → CAGR{cagr_gain_is:+.1f}pp)")
        if abs(exp_gain_oos) > 0.1:
            eff_oos = cagr_gain_oos / exp_gain_oos
            print(f"    OOS CAGR/Exp 1pp:  {eff_oos:+.3f}  (Exp{exp_gain_oos:+.1f}pp → CAGR{cagr_gain_oos:+.1f}pp)")

    # ─────────────────────────────────────────────────────────────────
    # REBALANCE ANALYTICS
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  ★ リバランス分析")
    print("=" * 80)
    print(f"\n  {'指標':<36} {'E1':>10} {'E2':>10} {'E3':>10}")
    print("  " + "-" * 66)

    def rrow(label, key, period, fmt="{:.0f}"):
        vals = [results[p][period].get(key, 0) for p in ["E1","E2","E3"]]
        fmts = [fmt.format(v) if isinstance(v,(int,float)) else str(v) for v in vals]
        print(f"  {label:<36} {fmts[0]:>10} {fmts[1]:>10} {fmts[2]:>10}")

    print(f"\n  IS 2018-2024")
    rrow("Rebalance events (total)",     "rebalance_count",   "IS")
    rrow("Rebalance events/yr",          "rebalance_count",   "IS",  "{:.1f}")  # raw count / IS years
    rrow("Trim realized PnL (¥)",        "rebalance_trim_pnl","IS", "{:,.0f}")
    rrow("Winner trim PnL (¥)",          "winner_trim_pnl",   "IS", "{:,.0f}")
    rrow("New entries via rebalance",    "new_signal_entries","IS")
    rrow("Missed signals",               "missed_signals",    "IS")

    # winner trimming ratio
    print(f"\n  Winner Trim Ratio (IS) — 利益確定トリム / 全トリム (%)")
    for e_pat in ["E1","E2","E3"]:
        r    = results[e_pat]["IS"]
        trim = r.get("rebalance_trim_pnl", 0)
        wt   = r.get("winner_trim_pnl", 0)
        ratio = (wt / trim * 100) if trim > 0 else 0.0
        print(f"    {e_pat}: trim_pnl={trim:+,.0f}¥  winner_part={wt:+,.0f}¥  ratio={ratio:.1f}%")

    # ─────────────────────────────────────────────────────────────────
    # CONCLUSIONS
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  ★ 結論")
    print("=" * 80)

    is_cagr   = {p: results[p]["IS"].get("cagr",   -999) for p in patterns}
    is_sharpe = {p: results[p]["IS"].get("sharpe",  -999) for p in patterns}
    is_calmar = {p: results[p]["IS"].get("calmar",  -999) for p in patterns}
    oos_cagr  = {p: results[p]["OOS"].get("cagr",  -999) for p in patterns}
    oos_sh    = {p: results[p]["OOS"].get("sharpe", -999) for p in patterns}

    best_is_cagr   = max(is_cagr,   key=is_cagr.get)
    best_is_sharpe = max(is_sharpe, key=is_sharpe.get)
    best_is_calmar = max(is_calmar, key=is_calmar.get)
    best_oos_cagr  = max(oos_cagr,  key=oos_cagr.get)

    print(f"\n  IS 最優秀:")
    print(f"    CAGR最大  : {best_is_cagr}  ({is_cagr[best_is_cagr]:+.1f}%)")
    print(f"    Sharpe最大: {best_is_sharpe}  ({is_sharpe[best_is_sharpe]:.3f})")
    print(f"    Calmar最大: {best_is_calmar}  ({is_calmar[best_is_calmar]:.3f})")
    print(f"  OOS 最優秀:")
    print(f"    CAGR最大  : {best_oos_cagr}  ({oos_cagr[best_oos_cagr]:+.1f}%)")

    print(f"\n  1. 待機資金削減の効果")
    any_beats_a = any(
        results[p]["IS"].get("cagr", -999) > results["A"]["IS"].get("cagr", 0) and
        results[p]["IS"].get("calmar", -999) > results["A"]["IS"].get("calmar", 0)
        for p in ["E1","E2","E3"]
    )
    if any_beats_a:
        print("     → IS CAGR・Calmar 双方でAを上回るパターンが存在")
    else:
        print("     → IS CAGR・Calmar 双方でAを上回るパターンは存在しない")
        a_cagr = results["A"]["IS"].get("cagr",0)
        for e_pat in ["E1","E2","E3"]:
            diff = results[e_pat]["IS"].get("cagr",0) - a_cagr
            print(f"     {e_pat}: CAGR diff = {diff:+.1f}pp  MaxDD diff = "
                  f"{results[e_pat]['IS'].get('max_dd',0) - results['A']['IS'].get('max_dd',0):+.1f}pp")

    print(f"\n  2. 最も効率的な配分方式 (IS CAGR/Exposure 効率)")
    for e_pat in ["E1","E2","E3"]:
        exp_g  = results[e_pat]["IS"].get("avg_exposure",0) - results["A"]["IS"].get("avg_exposure",0)
        cagr_g = results[e_pat]["IS"].get("cagr",0)         - results["A"]["IS"].get("cagr",0)
        eff    = (cagr_g / exp_g) if abs(exp_g) > 0.1 else 0.0
        print(f"     {e_pat}: Exposure {exp_g:+.1f}pp → CAGR {cagr_g:+.1f}pp  効率={eff:+.3f}")

    print(f"\n  3. A を上回るモデルの存在")
    for e_pat in ["E1","E2","E3"]:
        is_better_cagr  = results[e_pat]["IS"].get("cagr",0) > results["A"]["IS"].get("cagr",0)
        is_better_sh    = results[e_pat]["IS"].get("sharpe",0) > results["A"]["IS"].get("sharpe",0)
        is_better_calmar= results[e_pat]["IS"].get("calmar",0) > results["A"]["IS"].get("calmar",0)
        oos_better      = results[e_pat]["OOS"].get("cagr",0) > results["A"]["OOS"].get("cagr",0)
        flags = (
            ("IS_CAGR+" if is_better_cagr else "IS_CAGR-") + " " +
            ("IS_Sharpe+" if is_better_sh else "IS_Sharpe-") + " " +
            ("IS_Calmar+" if is_better_calmar else "IS_Calmar-") + " " +
            ("OOS_CAGR+" if oos_better else "OOS_CAGR-")
        )
        print(f"     {e_pat}: {flags}")

    print(f"\n  4. CAGR 30% 到達への寄与見込み")
    a_is_cagr = results["A"]["IS"].get("cagr", 0)
    for e_pat in ["E1","E2","E3"]:
        e_is_cagr = results[e_pat]["IS"].get("cagr", 0)
        gap_to_30 = 30.0 - a_is_cagr
        contrib   = e_is_cagr - a_is_cagr
        coverage  = (contrib / gap_to_30 * 100) if gap_to_30 > 0 else 0.0
        print(f"     {e_pat}: +{contrib:.1f}pp → 30%ギャップ({gap_to_30:.1f}pp)の{coverage:.0f}%をカバー")

    print(f"\n  5. 採用評価 (REJECT / RESEARCH / PROMOTE)")
    for e_pat in ["E1","E2","E3"]:
        r_is   = results[e_pat]["IS"]
        r_oos  = results[e_pat]["OOS"]
        a_is   = results["A"]["IS"]
        a_oos  = results["A"]["OOS"]

        cagr_ok  = r_is.get("cagr",0)   >= a_is.get("cagr",0)
        calmar_ok= r_is.get("calmar",0) >= a_is.get("calmar",0) * 0.9  # within 10%
        oos_ok   = r_oos.get("cagr",0)  >= a_oos.get("cagr",0) * 0.8   # within 20%
        dd_ok    = abs(r_is.get("max_dd",0)) <= abs(a_is.get("max_dd",0)) * 1.20  # max 20% worse

        if cagr_ok and calmar_ok and oos_ok and dd_ok:
            verdict = "★ PROMOTE"
        elif cagr_ok or (calmar_ok and oos_ok):
            verdict = "△ RESEARCH"
        else:
            verdict = "✕ REJECT"

        print(f"     {e_pat}: {verdict}  "
              f"(IS_CAGR={r_is.get('cagr',0):+.1f}% "
              f"Calmar={r_is.get('calmar',0):.3f} "
              f"MaxDD={r_is.get('max_dd',0):.1f}% "
              f"OOS_CAGR={r_oos.get('cagr',0):+.1f}%)")

    # ─────────────────────────────────────────────────────────────────
    # SAVE JSON
    # ─────────────────────────────────────────────────────────────────
    today    = time.strftime("%Y-%m-%d")
    out_path = OUTPUT_DIR / f"capital_allocation_e1e2e3_{today}.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_data = {
        f"pattern_{p}": {
            "IS":  {k: v for k, v in results[p]["IS"].items()  if k != "equity_curve"},
            "OOS": {k: v for k, v in results[p]["OOS"].items() if k != "equity_curve"},
        }
        for p in patterns
    }
    out_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  結果保存: {out_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
