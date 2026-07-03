"""
backtest/conditional_4th_position_audit.py

Conditional 4th Position Audit
仮説: 4枚目失敗は品質低下が原因 → 高品質4枚目のみに絞れば改善するか？

Step 1: max_pos=4 全トレード抽出（4枚目を特定）
Step 2: 品質別集計 (RSR tier / conviction quartile / entry timing)
Step 3: 2021失敗モード分析
Step 4: Counterfactual A/B/C/D/E シミュレーション

Output: reports/conditional_4th_position_audit.md

Run:
    cd C:/ai-trading
    python src/backtest/conditional_4th_position_audit.py
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
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR = Path("reports")

# ── Variant definitions ───────────────────────────────────────────────
VARIANTS = {
    "A": dict(max_pos=3, rsr_4th=0.0,  conv_mode="none",             label="A max_pos=3 (現行)"),
    "B": dict(max_pos=4, rsr_4th=0.0,  conv_mode="none",             label="B max_pos=4 フィルタなし"),
    "C": dict(max_pos=4, rsr_4th=80.0, conv_mode="none",             label="C max_pos=4 + 4th RSR≥80"),
    "D": dict(max_pos=4, rsr_4th=85.0, conv_mode="none",             label="D max_pos=4 + 4th RSR≥85"),
    "E": dict(max_pos=4, rsr_4th=80.0, conv_mode="portfolio_median", label="E max_pos=4 + 4th RSR≥80 + RSR≥portfolio_median"),
}


# ─────────────────────────────────────────────────────────────────────
#  CORE RUNNER  (configurable max_pos + 4th-position filter)
# ─────────────────────────────────────────────────────────────────────

def run_variant(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    start: str, end: str,
    max_pos: int = 3,
    rsr_4th: float = 0.0,           # min RSR for 4th slot (0 = no filter)
    conv_mode: str = "none",        # "none" | "portfolio_median"
) -> dict:
    """
    Run backtest with variable max_pos and optional 4th-slot quality filter.
    Tracks n_at_entry for every BUY so 4th positions can be identified.
    Also computes entry_timing_score (proximity to 20d breakout high).
    """
    capital      = float(cfg.portfolio.capital)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)
    MAX_SINGLE_W = float(cfg.portfolio.max_single_weight)
    turtle_entry = int(cfg.fujiko.turtle_entry)    # 20
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

    cash            = float(capital)
    positions: dict[str, Position] = {}
    pos_meta: dict[str, dict] = {}   # sym → {n_at_entry, rsr, entry_timing_score, entry_date}
    equity_curve    = []; exposure_list = []; trades = []
    peak_equity     = float(capital)
    cb_active       = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    exit_counts: dict[str, int] = {}

    missed_count    = 0; blocked_4th = 0
    n4_days         = 0   # days with 4 positions

    for i, date in enumerate(common_dates):
        invested = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
                       for s, pos in positions.items())
        cur_equity = cash + invested
        equity_curve.append(cur_equity)
        exposure_list.append(invested / max(1.0, cur_equity))
        if len(positions) == 4:
            n4_days += 1

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
            meta     = pos_meta.get(sym, {})
            trades.append({
                "side": "SELL", "symbol": sym,
                "entry": pos.entry_price, "exit": sell_px,
                "qty": pos.qty, "pnl": pnl,
                "entry_idx": pos.entry_idx, "exit_idx": i,
                "reason": reason,
                "entry_date": meta.get("entry_date",""),
                "exit_date":  str(date.date()),
                "sector":     trade_syms.get(sym,"不明"),
                "n_at_entry": meta.get("n_at_entry", 0),
                "rsr_at_entry": meta.get("rsr", 0.0),
                "entry_timing_score": meta.get("entry_timing_score", 1.0),
                "holding_days": i - pos.entry_idx,
                "return_pct":   round((sell_px / pos.entry_price - 1) * 100, 2),
            })
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
            del positions[sym]
            del pos_meta[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if cb_active or not buy_cands:
            continue

        buy_cands.sort(key=lambda x: -x[0])

        for rsr_val, sym in buy_cands:
            open_slots = max_pos - len(positions)
            if open_slots <= 0:
                missed_count += 1
                continue

            buy_px = float(open_mat[next_i, sym_to_i[sym]])
            if buy_px <= 0:
                continue
            if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                continue
            if gross_enabled:
                cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in positions.values()) / max(1.0, capital)
                if cur_gross + buy_px * LOT / max(1.0, capital) > gross_cap:
                    continue

            # ── 4th-position quality filter ───────────────────────────
            n_before = len(positions)   # number held BEFORE this entry
            if n_before == max_pos - 1 and max_pos == 4:
                # Would be the 4th slot → apply quality gate
                if rsr_val < rsr_4th:
                    blocked_4th += 1
                    continue
                if conv_mode == "portfolio_median":
                    held_rsrs = [float(rsr_mat[i, sym_to_i[s]]) for s in positions]
                    if held_rsrs and rsr_val < float(np.median(held_rsrs)):
                        blocked_4th += 1
                        continue

            # sizing (cash / effective_slots, max_single_weight cap)
            n_rem     = sum(1 for _, s in buy_cands if s not in positions)
            eff_slots = min(open_slots, max(1, n_rem))
            alloc     = (cash / eff_slots) * cb_scale
            alloc     = min(alloc, capital * MAX_SINGLE_W)
            qty       = int(alloc / buy_px / LOT) * LOT
            if qty <= 0:
                continue

            # entry timing score: buy_px vs 20-day close high (turtle breakout proximity)
            if i >= turtle_entry:
                si_idx = sym_to_i[sym]
                high20 = float(np.nanmax(close_mat[i - turtle_entry:i, si_idx]))
                ets    = round(buy_px / high20, 4) if high20 > 0 else 1.0
            else:
                ets = 1.0

            _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
            cash -= qty * buy_px * (1 + COST_ONE_WAY)

            pos_meta[sym] = {
                "n_at_entry":         n_before,
                "rsr":                rsr_val,
                "entry_timing_score": ets,
                "entry_date":         str(date.date()),
            }

    metrics = calc_metrics(equity_curve, trades, exposure_list, capital, list(common_dates))
    metrics["exit_reasons"]  = exit_counts
    metrics["missed_signals"]= missed_count
    metrics["blocked_4th"]   = blocked_4th
    metrics["n4_days"]       = n4_days
    metrics["n4_days_pct"]   = round(n4_days / max(len(common_dates), 1) * 100, 1)

    sells = [t for t in trades if t["side"] == "SELL"]
    n4th_trades = [t for t in sells if t.get("n_at_entry", 0) >= max_pos - 1 and max_pos == 4]
    metrics["n4th_trades"]   = len(n4th_trades)

    return {
        "metrics":    metrics,
        "trades":     trades,
        "common_dates": common_dates,
        "trade_syms": trade_syms,
        "equity_curve": equity_curve,
    }


# ─────────────────────────────────────────────────────────────────────
#  STEP 1: Extract 4th-position trades
# ─────────────────────────────────────────────────────────────────────

def extract_4th_trades(trades: list) -> list:
    return [t for t in trades
            if t["side"] == "SELL" and t.get("n_at_entry", 0) == 3]


# ─────────────────────────────────────────────────────────────────────
#  STEP 2: Quality breakdown
# ─────────────────────────────────────────────────────────────────────

def quality_stats(trades: list) -> dict:
    if not trades:
        return {}
    wins   = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    rets   = [t.get("return_pct", 0) for t in trades]
    gross_p= sum(t.get("pnl",0) for t in wins)
    gross_l= abs(sum(t.get("pnl",0) for t in losses))
    return {
        "n":          len(trades),
        "win_rate":   round(len(wins) / max(len(trades),1) * 100, 1),
        "avg_return": round(float(np.mean(rets)), 2),
        "median_ret": round(float(np.median(rets)), 2),
        "pf":         round(gross_p / max(gross_l, 1.0), 3),
        "total_pnl":  round(sum(t.get("pnl",0) for t in trades), 0),
    }


def step2_quality_breakdown(fourth_trades: list) -> dict:
    out = {}

    # RSR tiers
    tiers = [("75-80", 75, 80), ("80-85", 80, 85), ("85-90", 85, 90), ("90+", 90, 999)]
    out["by_rsr_tier"] = {}
    for label, lo, hi in tiers:
        grp = [t for t in fourth_trades if lo <= t.get("rsr_at_entry",0) < hi]
        out["by_rsr_tier"][label] = quality_stats(grp)

    # Conviction quartile (using RSR as conviction proxy)
    rsrs = [t.get("rsr_at_entry",0) for t in fourth_trades if t.get("rsr_at_entry",0) > 0]
    if rsrs:
        q25, q50, q75 = np.percentile(rsrs, [25, 50, 75])
        out["conviction_quartiles"] = {"q25": round(q25,1), "q50": round(q50,1), "q75": round(q75,1)}
        for qlabel, qlo, qhi in [("Q1 (low)", 0, q25), ("Q2", q25, q50),
                                   ("Q3", q50, q75), ("Q4 (high)", q75, 999)]:
            grp = [t for t in fourth_trades if qlo <= t.get("rsr_at_entry",0) < qhi]
            out[f"conviction_{qlabel}"] = quality_stats(grp)

    # Entry timing quartile
    ets_vals = [t.get("entry_timing_score",1.0) for t in fourth_trades]
    if ets_vals:
        et25, et50, et75 = np.percentile(ets_vals, [25, 50, 75])
        out["ets_quartiles"] = {"q25": round(et25,4), "q50": round(et50,4), "q75": round(et75,4)}
        # Lower ETS = tighter to breakout = better for turtle
        for qlabel, qlo, qhi in [("Q1 tight≤", 0, et25), ("Q2", et25, et50),
                                   ("Q3", et50, et75), ("Q4 loose>", et75, 999)]:
            grp = [t for t in fourth_trades if qlo <= t.get("entry_timing_score",1.0) < qhi]
            out[f"ets_{qlabel}"] = quality_stats(grp)

    return out


# ─────────────────────────────────────────────────────────────────────
#  STEP 3: 2021 failure mode
# ─────────────────────────────────────────────────────────────────────

def step3_2021_analysis(trades: list, fourth_trades: list) -> dict:
    y2021_all   = [t for t in trades        if t["side"]=="SELL" and t.get("exit_date","").startswith("2021")]
    y2021_4th   = [t for t in fourth_trades if t.get("exit_date","").startswith("2021")]
    y2021_other = [t for t in y2021_all     if t.get("n_at_entry",0) < 3]

    # Top losses among 4th positions in 2021
    y2021_4th_sorted = sorted(y2021_4th, key=lambda x: x.get("pnl",0))

    # Sector breakdown
    sector_loss_4th = {}
    for t in y2021_4th:
        s = t.get("sector","不明")
        sector_loss_4th[s] = sector_loss_4th.get(s, 0) + t.get("pnl", 0)

    # RSR distribution of 2021 4th positions
    rsrs_2021 = [t.get("rsr_at_entry",0) for t in y2021_4th]

    return {
        "all_2021_trades":   quality_stats(y2021_all),
        "4th_2021_trades":   quality_stats(y2021_4th),
        "other_2021_trades": quality_stats(y2021_other),
        "top_losses":        y2021_4th_sorted[:5],
        "sector_pnl":        dict(sorted(sector_loss_4th.items(), key=lambda x: x[1])),
        "rsr_mean":          round(np.mean(rsrs_2021),1) if rsrs_2021 else 0,
        "rsr_min":           round(min(rsrs_2021),1)     if rsrs_2021 else 0,
        "n_below_80":        sum(1 for r in rsrs_2021 if r < 80),
        "n_below_85":        sum(1 for r in rsrs_2021 if r < 85),
    }


# ─────────────────────────────────────────────────────────────────────
#  MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────

def fmt(v, d=1):
    if isinstance(v, float):
        return f"{v:+.{d}f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)

def write_report(variant_results, s2, s3, fourth_trades_B, output_path: Path) -> None:
    L = []; w = L.append

    w("# Conditional 4th Position Audit")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  IS: 2018-2024  |  OOS: 2025")
    w("\n**仮説:** 4枚目失敗は「4枚目自体」ではなく「低品質4枚目の混入」が原因\n")

    # ── Step 1 ─────────────────────────────────────────────────────────
    w("---\n## Step 1: 4枚目ポジション全トレード (Variant B — max_pos=4 フィルタなし)\n")
    if fourth_trades_B:
        qs = quality_stats(fourth_trades_B)
        w(f"- 4枚目 SELL trades: **{qs['n']}件**")
        w(f"- 勝率: {qs['win_rate']}%  |  平均return: {qs['avg_return']:+.2f}%  |  中央値: {qs['median_ret']:+.2f}%")
        w(f"- Profit Factor: {qs['pf']:.3f}  |  総PnL: ¥{qs['total_pnl']:+,.0f}\n")

        w("| symbol | sector | entry | exit | hold_d | RSR | ETS | ret% | PnL |")
        w("|---|---|---|---|---|---|---|---|---|")
        for t in sorted(fourth_trades_B, key=lambda x: x.get("pnl",0)):
            w(f"| {t['symbol']} | {t.get('sector','―')} | {t.get('entry_date','')} "
              f"| {t.get('exit_date','')} | {t.get('holding_days',0)} "
              f"| {t.get('rsr_at_entry',0):.1f} | {t.get('entry_timing_score',0):.3f} "
              f"| {t.get('return_pct',0):+.1f}% | ¥{t.get('pnl',0):+,.0f} |")
    else:
        w("_(4枚目トレードなし)_")

    # ── Step 2 ─────────────────────────────────────────────────────────
    w("\n---\n## Step 2: 品質別集計\n")

    w("### RSR Tier別\n")
    w("| RSR Tier | N | 勝率 | avg ret% | median ret% | PF | 総PnL |")
    w("|---|---|---|---|---|---|---|")
    for tier, data in s2.get("by_rsr_tier",{}).items():
        if not data:
            continue
        w(f"| {tier} | {data['n']} | {data['win_rate']}% | {data['avg_return']:+.2f}% "
          f"| {data['median_ret']:+.2f}% | {data['pf']:.3f} | ¥{data['total_pnl']:+,.0f} |")

    qs_all = s2.get("conviction_quartiles",{})
    if qs_all:
        w(f"\n### Conviction Quartile (RSR proxy) — Q1≤{qs_all['q25']}  Q2≤{qs_all['q50']}  Q3≤{qs_all['q75']}  Q4+\n")
        w("| Quartile | N | 勝率 | avg ret% | PF |")
        w("|---|---|---|---|---|")
        for qlabel in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
            d = s2.get(f"conviction_{qlabel}", {})
            if not d:
                continue
            w(f"| {qlabel} | {d['n']} | {d['win_rate']}% | {d['avg_return']:+.2f}% | {d['pf']:.3f} |")

    ets_all = s2.get("ets_quartiles",{})
    if ets_all:
        w(f"\n### Entry Timing Score (Turtle Breakout近接度: 1.0=ブレイクアウト直後) "
          f"— Q1≤{ets_all['q25']}  Q2≤{ets_all['q50']}  Q3≤{ets_all['q75']}  Q4+\n")
        w("| ETS Quartile | N | 勝率 | avg ret% | PF |")
        w("|---|---|---|---|---|")
        for qlabel in ["Q1 tight≤", "Q2", "Q3", "Q4 loose>"]:
            d = s2.get(f"ets_{qlabel}", {})
            if not d:
                continue
            w(f"| {qlabel} | {d['n']} | {d['win_rate']}% | {d['avg_return']:+.2f}% | {d['pf']:.3f} |")

    # ── Step 3 ─────────────────────────────────────────────────────────
    w("\n---\n## Step 3: 2021年 失敗モード分析\n")
    w("| グループ | N | 勝率 | avg ret% | PF |")
    w("|---|---|---|---|---|")
    for gname, key in [("全取引(2021)","all_2021_trades"),
                        ("4枚目のみ(2021)","4th_2021_trades"),
                        ("1-3枚目(2021)","other_2021_trades")]:
        d = s3.get(key,{})
        if not d:
            w(f"| {gname} | 0 | — | — | — |"); continue
        w(f"| {gname} | {d['n']} | {d['win_rate']}% | {d['avg_return']:+.2f}% | {d['pf']:.3f} |")

    w(f"\n**2021 4枚目 RSR統計**: mean={s3.get('rsr_mean',0):.1f}  min={s3.get('rsr_min',0):.1f}")
    w(f"- RSR < 80: {s3.get('n_below_80',0)}件  |  RSR < 85: {s3.get('n_below_85',0)}件\n")

    w("**2021 4枚目 Top損失**\n")
    w("| symbol | sector | entry | exit | hold_d | RSR | ret% | PnL |")
    w("|---|---|---|---|---|---|---|---|")
    for t in s3.get("top_losses",[]):
        w(f"| {t['symbol']} | {t.get('sector','―')} | {t.get('entry_date','')} "
          f"| {t.get('exit_date','')} | {t.get('holding_days',0)} "
          f"| {t.get('rsr_at_entry',0):.1f} | {t.get('return_pct',0):+.1f}% "
          f"| ¥{t.get('pnl',0):+,.0f} |")

    w("\n**セクター別PnL (2021 4枚目)**\n")
    for sec, pnl in s3.get("sector_pnl",{}).items():
        w(f"- {sec}: ¥{pnl:+,.0f}")

    # ── Step 4 ─────────────────────────────────────────────────────────
    w("\n---\n## Step 4: Counterfactual Simulation\n")
    w("| Variant | IS CAGR | IS Sharpe | IS MaxDD | IS Calmar | IS Exposure | IS Missed | IS 4th利用日% | OOS CAGR | OOS Sharpe | OOS MaxDD |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for vk, vdata in VARIANTS.items():
        is_m  = variant_results.get(vk,{}).get("IS",{})
        oos_m = variant_results.get(vk,{}).get("OOS",{})
        w(f"| {vk} {vdata['label'].split('(')[0].strip()} "
          f"| {is_m.get('cagr',0):+.1f}% "
          f"| {is_m.get('sharpe',0):.3f} "
          f"| {is_m.get('max_dd',0):.1f}% "
          f"| {is_m.get('calmar',0):.3f} "
          f"| {is_m.get('avg_exposure',0):.1f}% "
          f"| {is_m.get('missed_signals',0)} "
          f"| {is_m.get('n4_days_pct',0):.1f}% "
          f"| {oos_m.get('cagr',0):+.1f}% "
          f"| {oos_m.get('sharpe',0):.3f} "
          f"| {oos_m.get('max_dd',0):.1f}% |")

    # Delta vs A
    a_is = variant_results.get("A",{}).get("IS",{})
    w("\n**Δ vs Pattern A (IS)**\n")
    w("| Variant | ΔCAGR | ΔSharpe | ΔMaxDD | ΔCalmar | Δblocked_4th |")
    w("|---|---|---|---|---|---|")
    for vk in ["B","C","D","E"]:
        m = variant_results.get(vk,{}).get("IS",{})
        if not m:
            continue
        w(f"| {vk} "
          f"| {m.get('cagr',0)-a_is.get('cagr',0):+.1f}pp "
          f"| {m.get('sharpe',0)-a_is.get('sharpe',0):+.3f} "
          f"| {m.get('max_dd',0)-a_is.get('max_dd',0):+.1f}pp "
          f"| {m.get('calmar',0)-a_is.get('calmar',0):+.3f} "
          f"| {m.get('blocked_4th',0)} blocked |")

    # IS annual returns table
    w("\n**IS 年次リターン (%)**\n")
    yr_keys = sorted(variant_results.get("A",{}).get("IS",{}).get("annual_returns",{}).keys())
    header = "| 年 |" + "".join(f" {vk} |" for vk in VARIANTS)
    w(header)
    w("|---|" + "---|"*len(VARIANTS))
    for yr in yr_keys:
        row = f"| {yr} |"
        for vk in VARIANTS:
            v = variant_results.get(vk,{}).get("IS",{}).get("annual_returns",{}).get(yr,0)
            row += f" {v:+.1f}% |"
        w(row)

    # ── Final verdict ──────────────────────────────────────────────────
    w("\n---\n## 最終結論\n")

    # Determine verdict
    b_is  = variant_results.get("B",{}).get("IS",{})
    c_is  = variant_results.get("C",{}).get("IS",{})
    d_is  = variant_results.get("D",{}).get("IS",{})
    a_cagr= a_is.get("cagr",0)
    b_cagr= b_is.get("cagr",0)
    c_cagr= c_is.get("cagr",0)
    d_cagr= d_is.get("cagr",0)
    c_calmar=c_is.get("calmar",0)
    a_calmar=a_is.get("calmar",0)

    if c_cagr > a_cagr and c_calmar >= a_calmar * 0.90:
        verdict = "**2. 高品質4枚目のみ有効 (RSR≥80)**"
        detail  = f"Variant C IS CAGR={c_cagr:+.1f}% (Δ={c_cagr-a_cagr:+.1f}pp) かつ Calmar={c_calmar:.3f} (A比{c_calmar/max(a_calmar,0.01)*100:.0f}%)"
    elif d_cagr > a_cagr and d_is.get("calmar",0) >= a_calmar * 0.90:
        verdict = "**2. 高品質4枚目のみ有効 (RSR≥85)**"
        detail  = f"Variant D IS CAGR={d_cagr:+.1f}% (Δ={d_cagr-a_cagr:+.1f}pp)"
    elif b_cagr < a_cagr and c_cagr <= a_cagr:
        verdict = "**1. 4枚目自体が不要 (フィルタでも改善しない)**"
        detail  = f"Variant B CAGR={b_cagr:+.1f}% Variant C CAGR={c_cagr:+.1f}% — いずれもA比改善なし"
    else:
        verdict = "**判断保留 — 追加検証要**"
        detail  = f"Variant B={b_cagr:+.1f}% C={c_cagr:+.1f}% D={d_cagr:+.1f}% vs A={a_cagr:+.1f}%"

    w(f"### 判定: {verdict}\n")
    w(f"根拠: {detail}\n")

    # 4th slot util summary
    b4_util = b_is.get("n4_days_pct",0)
    c4_util = c_is.get("n4_days_pct",0)
    d4_util = d_is.get("n4_days_pct",0)
    w(f"**4枚目スロット利用率 (IS)**")
    w(f"- B (無制限): {b4_util:.1f}%  |  C (RSR≥80): {c4_util:.1f}%  |  D (RSR≥85): {d4_util:.1f}%")
    w(f"- C blocked: {c_is.get('blocked_4th',0)}件  |  D blocked: {d_is.get('blocked_4th',0)}件\n")

    # Is the 4th position intrinsically bad?
    step2_75_80 = s2.get("by_rsr_tier",{}).get("75-80",{})
    step2_80_85 = s2.get("by_rsr_tier",{}).get("80-85",{})
    step2_85_90 = s2.get("by_rsr_tier",{}).get("85-90",{})
    w("**品質フィルタの効果（RSR tier PF）**")
    w(f"- RSR 75-80: PF={step2_75_80.get('pf',0):.3f}  勝率={step2_75_80.get('win_rate',0):.1f}%  (問題ゾーン)")
    w(f"- RSR 80-85: PF={step2_80_85.get('pf',0):.3f}  勝率={step2_80_85.get('win_rate',0):.1f}%")
    w(f"- RSR 85-90: PF={step2_85_90.get('pf',0):.3f}  勝率={step2_85_90.get('win_rate',0):.1f}%")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート保存: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  Conditional 4th Position Audit")
    print("=" * 68 + "\n")

    print("[1/3] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    def run_v(label, start, end, max_pos, rsr_4th=0.0, conv_mode="none"):
        print(f"  [{label}] max_pos={max_pos}  rsr_4th≥{rsr_4th}  conv={conv_mode}  {start}〜{end}")
        t0  = time.time()
        res = run_variant(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                          topix_close, rsr_syms, cfg,
                          start=start, end=end,
                          max_pos=max_pos, rsr_4th=rsr_4th, conv_mode=conv_mode)
        m   = res.get("metrics",{})
        print(f"         CAGR={m.get('cagr',0):+.1f}%  Sharpe={m.get('sharpe',0):.3f}  "
              f"MaxDD={m.get('max_dd',0):.1f}%  Calmar={m.get('calmar',0):.3f}  "
              f"4th_days={m.get('n4_days_pct',0):.1f}%  blocked={m.get('blocked_4th',0)}  "
              f"({time.time()-t0:.1f}s)")
        return res

    print("\n[2/3] バリアント実行 (IS + OOS)...")
    variant_results: dict[str, dict] = {}
    for vk, vdef in VARIANTS.items():
        kw = {k: v for k, v in vdef.items() if k != "label"}
        res_is  = run_v(f"{vk} IS",  IS_START,  IS_END,  **kw)
        res_oos = run_v(f"{vk} OOS", OOS_START, OOS_END, **kw)
        variant_results[vk] = {
            "IS":     res_is.get("metrics", {}),   # metrics only (for report table)
            "OOS":    res_oos.get("metrics", {}),
            "IS_run": res_is,                       # full run (for trades extraction)
        }

    print("\n[3/3] 分析 + レポート生成...")

    # Extract B IS data for steps 2, 3
    b_is_trades     = variant_results["B"]["IS_run"]["trades"]
    fourth_trades_B = extract_4th_trades(b_is_trades)
    print(f"  4枚目 IS trades: {len(fourth_trades_B)}件")

    s2 = step2_quality_breakdown(fourth_trades_B)
    s3 = step3_2021_analysis(b_is_trades, fourth_trades_B)

    out_path = REPORTS_DIR / "conditional_4th_position_audit.md"
    write_report(variant_results, s2, s3, fourth_trades_B, out_path)

    # ── Console summary ────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  ★ コンソール要約")
    print("="*68)

    print(f"\n  IS Counterfactual 結果:")
    for vk in ["A","B","C","D","E"]:
        m = variant_results.get(vk,{}).get("IS",{})
        print(f"  {vk}: CAGR={m.get('cagr',0):+.1f}%  Sharpe={m.get('sharpe',0):.3f}  "
              f"MaxDD={m.get('max_dd',0):.1f}%  Calmar={m.get('calmar',0):.3f}  "
              f"Exp={m.get('avg_exposure',0):.1f}%  4th_util={m.get('n4_days_pct',0):.1f}%  "
              f"missed={m.get('missed_signals',0)}")

    print(f"\n  OOS Counterfactual 結果:")
    for vk in ["A","B","C","D","E"]:
        m = variant_results.get(vk,{}).get("OOS",{})
        print(f"  {vk}: CAGR={m.get('cagr',0):+.1f}%  Sharpe={m.get('sharpe',0):.3f}  "
              f"MaxDD={m.get('max_dd',0):.1f}%  Calmar={m.get('calmar',0):.3f}")

    print(f"\n  4枚目 品質別 PF (IS):")
    for tier in ["75-80","80-85","85-90","90+"]:
        d = s2.get("by_rsr_tier",{}).get(tier,{})
        print(f"  RSR {tier}: N={d.get('n',0)}  WR={d.get('win_rate',0):.1f}%  "
              f"avg={d.get('avg_return',0):+.2f}%  PF={d.get('pf',0):.3f}")

    print(f"\n  2021 4枚目: {s3.get('4th_2021_trades',{}).get('n',0)}件  "
          f"RSR mean={s3.get('rsr_mean',0):.1f}  below_80={s3.get('n_below_80',0)}件  "
          f"below_85={s3.get('n_below_85',0)}件")

    print(f"\n  レポート → {out_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
