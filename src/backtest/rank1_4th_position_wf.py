"""
backtest/rank1_4th_position_wf.py

研究専用 / 実装変更禁止

仮説:
  既存3ポジ保有中でも、その日の全候補中 rank1 が CAPACITY-blocked になった場合のみ、
  余剰キャッシュ（equity×0.25 上限）で一時的4枚目を追加すると期待値改善するか。

Baseline: max_pos=3 (現行)
Variant:  max_pos=3 + rank1-CAPACITY時のみ4枚目（余剰キャッシュ限定）

Winner定義:
  buy_cands をRSR降順ソート → candidate_rank = 全候補中順位
  rank1 かつ reason==CAPACITY（当日処理開始時点でpositions==3）のみ許可
  rank2以下は絶対不可

4枚目サイズ:
  min(available_cash, equity × MAX_SINGLE_W)
  既存3ポジのサイズ・構成は一切変更しない

WF: 5-Fold (test: 2021/2022/2023/2024/2025)

Output: reports/rank1_4th_position_wf.md

Run:
    cd C:/ai-trading
    python src/backtest/rank1_4th_position_wf.py
"""

from __future__ import annotations

import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, calc_metrics, _sector_ok, _execute_buy, Position, _take,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR = Path("reports")

# WF folds
FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2021弱年"),
    ("Fold2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2022Bear"),
    ("Fold3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2023強気"),
    ("Fold4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2024"),
    ("Fold5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2025"),
]

# 採用基準
ADOPT_TRIGGER_MAX  = 12   # /year
ADOPT_DCAGR_MIN    = 1.0  # pp
ADOPT_DMAXDD_MAX   = 2.0  # pp (absolute delta, positive = worse)
ADOPT_GPT_MIN      = 0.0  # gain_per_trigger > 0
WF_PASS_MIN        = 4    # /5


# ─────────────────────────────────────────────────────────────────────
#  CORE RUNNER
# ─────────────────────────────────────────────────────────────────────

def run_backtest(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    start: str, end: str,
    enable_rank1_4th: bool = False,
) -> dict:
    """
    enable_rank1_4th=False  → Baseline (max_pos=3 only)
    enable_rank1_4th=True   → Variant  (rank1-CAPACITY → 4th pos using available cash)
    """
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)   # always 3
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)
    MAX_SINGLE_W = float(cfg.portfolio.max_single_weight)  # 0.25
    turtle_entry = int(cfg.fujiko.turtle_entry)

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
    pos_meta: dict[str, dict] = {}
    equity_curve    = []
    exposure_list   = []
    trades          = []
    triggers        = []   # 4th-position trigger log
    peak_equity     = float(capital)
    cb_active       = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    exit_counts: dict[str, int] = {}

    # Track 4th-position trades separately (n_at_entry==3)
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
            meta     = pos_meta.get(sym, {})
            trades.append({
                "side": "SELL", "symbol": sym,
                "entry": pos.entry_price, "exit": sell_px,
                "qty": pos.qty, "pnl": pnl,
                "entry_idx": pos.entry_idx, "exit_idx": i,
                "reason": reason,
                "entry_date": meta.get("entry_date", ""),
                "exit_date":  str(date.date()),
                "sector":     trade_syms.get(sym, "不明"),
                "n_at_entry": meta.get("n_at_entry", 0),
                "rsr_at_entry": meta.get("rsr", 0.0),
                "holding_days": i - pos.entry_idx,
                "return_pct":   round((sell_px / pos.entry_price - 1) * 100, 2),
                "is_4th": meta.get("is_4th", False),
            })
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
            del positions[sym]
            del pos_meta[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if cb_active or not buy_cands:
            continue

        # Sort by RSR descending: rank1 = index 0
        buy_cands.sort(key=lambda x: -x[0])

        # Snapshot: n_positions at start of buy processing (after sells)
        n_before_buys = len(positions)

        rank1_4th_done = False   # at most one 4th position per day

        for global_rank_0, (rsr_val, sym) in enumerate(buy_cands):
            global_rank = global_rank_0 + 1  # 1-indexed
            open_slots  = max_pos - len(positions)

            if open_slots > 0:
                # ── Normal buy ─────────────────────────────────────────
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

                n_rem  = sum(1 for _, s in buy_cands if s not in positions)
                eff_slots = min(open_slots, max(1, n_rem))
                alloc  = (cash / eff_slots) * cb_scale
                alloc  = min(alloc, capital * MAX_SINGLE_W)
                qty    = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue

                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)
                pos_meta[sym] = {
                    "n_at_entry": len(positions) - 1,
                    "rsr":        rsr_val,
                    "entry_date": str(date.date()),
                    "is_4th":     False,
                }

            else:
                # ── CAPACITY blocked ───────────────────────────────────
                if (
                    enable_rank1_4th
                    and global_rank == 1          # rank1 globally
                    and n_before_buys == max_pos  # all 3 slots were full BEFORE today's buys
                    and not rank1_4th_done        # one 4th position per day max
                ):
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

                    # 4th position sizing: min(available_cash, equity × MAX_SINGLE_W)
                    invested_now = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                       for p in positions.values())
                    cur_eq = cash + invested_now
                    alloc_4th = min(cash, cur_eq * MAX_SINGLE_W) * cb_scale
                    qty_4th   = int(alloc_4th / buy_px / LOT) * LOT
                    if qty_4th <= 0:
                        continue

                    # Capture weakest held position (lowest RSR)
                    held_rsrs = {s: float(rsr_mat[i, sym_to_i[s]]) for s in positions}
                    weakest_sym = min(held_rsrs, key=held_rsrs.get) if held_rsrs else ""
                    weakest_rsr = held_rsrs.get(weakest_sym, 0.0)

                    existing_rsrs_sorted = sorted(held_rsrs.values(), reverse=True)

                    trigger = {
                        "date":           str(date.date()),
                        "ticker":         sym,
                        "candidate_rsr":  round(rsr_val, 1),
                        "existing_rsrs":  [round(r, 1) for r in existing_rsrs_sorted],
                        "available_cash": round(cash, 0),
                        "alloc":          round(alloc_4th, 0),
                        "entry_idx":      next_i,
                        "accepted":       True,
                        "weakest_sym":    weakest_sym,
                        "weakest_rsr":    round(weakest_rsr, 1),
                        # forward returns filled post-run
                        "future_20d":     None,
                        "future_60d":     None,
                        "future_120d":    None,
                        "weakest_future_20d":  None,
                        "weakest_future_60d":  None,
                        "weakest_future_120d": None,
                    }

                    _execute_buy(sym, buy_px, qty_4th, i, next_i, trade_syms, trades, positions, rsr_val)
                    cash -= qty_4th * buy_px * (1 + COST_ONE_WAY)
                    pos_meta[sym] = {
                        "n_at_entry": 3,
                        "rsr":        rsr_val,
                        "entry_date": str(date.date()),
                        "is_4th":     True,
                    }
                    triggers.append(trigger)
                    rank1_4th_done = True

                # rank2+ are never eligible → no else-branch needed

    # ── Post-run: compute forward returns for triggers ─────────────────
    for trig in triggers:
        entry_idx = trig["entry_idx"]
        sym = trig["ticker"]
        wsym = trig["weakest_sym"]
        if sym in sym_to_i:
            si = sym_to_i[sym]
            for h in [20, 60, 120]:
                fi = entry_idx + h
                if fi < n_dates and close_mat[entry_idx, si] > 0:
                    fwd = float(close_mat[fi, si]) / float(close_mat[entry_idx, si]) - 1
                    trig[f"future_{h}d"] = round(fwd * 100, 2)
        if wsym and wsym in sym_to_i:
            wsi = sym_to_i[wsym]
            for h in [20, 60, 120]:
                fi = entry_idx + h
                if fi < n_dates and close_mat[entry_idx, wsi] > 0:
                    fwd = float(close_mat[fi, wsi]) / float(close_mat[entry_idx, wsi]) - 1
                    trig[f"weakest_future_{h}d"] = round(fwd * 100, 2)

    metrics = calc_metrics(equity_curve, trades, exposure_list, capital, list(common_dates))
    metrics["exit_reasons"]   = exit_counts
    metrics["trigger_count"]  = len(triggers)
    metrics["n_dates"]        = n_dates

    sells_4th = [t for t in trades if t.get("is_4th") and t["side"] == "SELL"]
    if sells_4th:
        rets_4th = [t["return_pct"] for t in sells_4th]
        pnls_4th = [t["pnl"] for t in sells_4th]
        metrics["n_4th_trades"]    = len(sells_4th)
        metrics["winner_avg_ret"]  = round(float(np.mean(rets_4th)), 2)
        metrics["total_4th_pnl"]   = round(sum(pnls_4th), 0)
        metrics["gain_per_trigger"]= round(sum(pnls_4th) / max(1, len(triggers)), 0)
    else:
        metrics["n_4th_trades"]    = 0
        metrics["winner_avg_ret"]  = 0.0
        metrics["total_4th_pnl"]   = 0.0
        metrics["gain_per_trigger"]= 0.0

    return {
        "metrics":       metrics,
        "trades":        trades,
        "triggers":      triggers,
        "equity_curve":  equity_curve,
        "common_dates":  common_dates,
    }


# ─────────────────────────────────────────────────────────────────────
#  WF EVALUATION
# ─────────────────────────────────────────────────────────────────────

def compare_winner_vs_weakest(triggers: list) -> dict:
    """4枚目return vs 保有中最低RSRポジreturnの比較 (future_20d base)."""
    valid = [t for t in triggers
             if t.get("future_20d") is not None and t.get("weakest_future_20d") is not None]
    if not valid:
        return {}
    diffs = [t["future_20d"] - t["weakest_future_20d"] for t in valid]
    winner_above = sum(1 for d in diffs if d > 0)
    return {
        "n_valid":       len(valid),
        "avg_winner_ret_20d":  round(float(np.mean([t["future_20d"] for t in valid])), 2),
        "avg_weakest_ret_20d": round(float(np.mean([t["weakest_future_20d"] for t in valid])), 2),
        "avg_diff_20d":        round(float(np.mean(diffs)), 2),
        "winner_above_weakest_pct": round(winner_above / len(valid) * 100, 1),
        "n_winner_above":      winner_above,
    }


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(fold_rows: list, all_triggers: list, output_path: Path) -> None:
    L = []; w = L.append

    w("# Rank1 条件付き4枚目ポジション WF 研究")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  WF 5-Fold  |  研究専用（実装変更なし）")
    w("""
**仮説**: 既存3ポジ満員でも、その日の全候補中 rank1 が CAPACITY-blocked の場合のみ
余剰キャッシュ（equity×25%上限）で4枚目を追加すると期待値改善するか。

**Winner定義**: candidate_rank==1 かつ reason==CAPACITY（当日開始時点でpositions==3）
**rank2以下は絶対不可** / **RSR絶対閾値は使わない**（相場環境依存のため）
""")

    # ── Section 1: WF 結果表 ────────────────────────────────────────
    w("---\n## 1. WF 結果（5-Fold）\n")
    w("| Fold | 年 | A CAGR | F CAGR | ΔCAGR | ΔCalmar | ΔMaxDD | trigger/y | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|")

    wf_pass = 0
    for row in fold_rows:
        a = row["A"]; f = row["F"]
        dcagr  = f["cagr"]   - a["cagr"]
        dcal   = f["calmar"] - a["calmar"]
        dmaxdd = f["max_dd"] - a["max_dd"]   # negative = better DD
        trig_y = round(row["trigger_count"] / max(row["test_years"], 0.01), 1)

        # WF pass: ΔCAGR >= 0 AND ΔMaxDD <= +2pp
        pass_mark = dcagr >= 0 and dmaxdd <= 2.0
        if pass_mark:
            wf_pass += 1
        judge = "✅" if pass_mark else "❌"

        w(f"| {row['fold']} | {row['label']} "
          f"| {a['cagr']:+.1f}% "
          f"| {f['cagr']:+.1f}% "
          f"| {dcagr:+.1f}pp "
          f"| {dcal:+.3f} "
          f"| {dmaxdd:+.1f}pp "
          f"| {trig_y:.1f} "
          f"| {judge} |")

    w(f"\n**WF通過: {wf_pass}/5**")

    # ── Section 2: 評価指標サマリ ─────────────────────────────────
    w("\n---\n## 2. 評価指標サマリ（全Fold集計）\n")

    total_triggers = sum(r["trigger_count"] for r in fold_rows)
    total_years    = sum(r["test_years"] for r in fold_rows)
    avg_dcagr   = float(np.mean([r["F"]["cagr"]   - r["A"]["cagr"]   for r in fold_rows]))
    avg_dcalmar = float(np.mean([r["F"]["calmar"] - r["A"]["calmar"] for r in fold_rows]))
    avg_dmaxdd  = float(np.mean([r["F"]["max_dd"] - r["A"]["max_dd"] for r in fold_rows]))

    total_4th_pnl = sum(r["F"].get("total_4th_pnl", 0) for r in fold_rows)
    avg_trigger_y = round(total_triggers / max(total_years, 0.01), 1)

    w(f"| 指標 | 値 | 採用基準 | 判定 |")
    w(f"|---|---|---|---|")
    w(f"| trigger_count (全期間) | {total_triggers}件 | — | — |")
    w(f"| trigger/year (平均) | {avg_trigger_y:.1f}回/年 | ≤{ADOPT_TRIGGER_MAX}回/年 | {'✅' if avg_trigger_y <= ADOPT_TRIGGER_MAX else '❌'} |")
    w(f"| 平均ΔCAGR | {avg_dcagr:+.2f}pp | ≥+{ADOPT_DCAGR_MIN}pp | {'✅' if avg_dcagr >= ADOPT_DCAGR_MIN else '❌'} |")
    w(f"| 平均ΔMaxDD | {avg_dmaxdd:+.2f}pp | ≤+{ADOPT_DMAXDD_MAX}pp | {'✅' if avg_dmaxdd <= ADOPT_DMAXDD_MAX else '❌'} |")
    w(f"| 平均ΔCalmar | {avg_dcalmar:+.3f} | — | — |")
    w(f"| total_4th_pnl (全期間) | ¥{total_4th_pnl:+,.0f} | — | — |")

    # gain_per_trigger
    all_gpt = [r["F"].get("gain_per_trigger", 0) for r in fold_rows]
    avg_gpt = float(np.mean(all_gpt))
    w(f"| gain_per_trigger (平均) | ¥{avg_gpt:+,.0f}/件 | >0 | {'✅' if avg_gpt > 0 else '❌'} |")

    w(f"\n**WF通過: {wf_pass}/5**  (採用基準: ≥{WF_PASS_MIN}/5)")

    # ── Section 3: Fold別 詳細 ────────────────────────────────────
    w("\n---\n## 3. Fold別 詳細\n")
    w("| Fold | A Calmar | F Calmar | A 4th% | F 4th% | winner_avg_ret | 4th trades |")
    w("|---|---|---|---|---|---|---|")
    for row in fold_rows:
        a = row["A"]; f = row["F"]
        w(f"| {row['fold']} {row['label']} "
          f"| {a['calmar']:.3f} "
          f"| {f['calmar']:.3f} "
          f"| {a.get('idle_cash', 0):.1f}% idle "
          f"| {f.get('idle_cash', 0):.1f}% idle "
          f"| {f.get('winner_avg_ret', 0):+.2f}% "
          f"| {f.get('n_4th_trades', 0)} |")

    # ── Section 4: cash utilization ───────────────────────────────
    w("\n---\n## 4. 現金利用率の変化\n")
    w("| Fold | A idle_cash | F idle_cash | Δidle | 追加投資額(avg) |")
    w("|---|---|---|---|---|")
    for row in fold_rows:
        a = row["A"]; f = row["F"]
        did = f.get("idle_cash", 0) - a.get("idle_cash", 0)
        avg_alloc = sum(t.get("alloc", 0) for t in row.get("triggers", [])) / max(1, len(row.get("triggers", [])))
        w(f"| {row['fold']} "
          f"| {a.get('idle_cash',0):.1f}% "
          f"| {f.get('idle_cash',0):.1f}% "
          f"| {did:+.1f}pp "
          f"| ¥{avg_alloc:,.0f} |")

    # ── Section 5: Winner vs Weakest 比較 ─────────────────────────
    w("\n---\n## 5. 4枚目 vs 保有中最低RSPポジション 比較（future_20d）\n")
    wvw = compare_winner_vs_weakest(all_triggers)
    if wvw:
        judge_wvw = "✅" if wvw.get("avg_diff_20d", -1) > 0 else "❌"
        w(f"- 有効サンプル数: {wvw['n_valid']}件")
        w(f"- 4枚目 avg future_20d: **{wvw['avg_winner_ret_20d']:+.2f}%**")
        w(f"- 最低RSRポジ avg future_20d: **{wvw['avg_weakest_ret_20d']:+.2f}%**")
        w(f"- 差分 (winner − weakest): **{wvw['avg_diff_20d']:+.2f}%** {judge_wvw}")
        w(f"- winner > weakest の割合: **{wvw['winner_above_weakest_pct']:.1f}%** "
          f"({wvw['n_winner_above']}/{wvw['n_valid']}件)")
        w(f"\n採用基準: **winner_avg_return > weakest_return** → {judge_wvw}")
    else:
        w("_(サンプル不足 — 4枚目ポジションが発生しなかったか forward data 不足)_")

    # ── Section 6: 観測ログ (全Fold) ──────────────────────────────
    w("\n---\n## 6. 観測ログ（全Foldトリガー一覧）\n")
    if all_triggers:
        w("| date | ticker | candidate_RSR | 既存[rank1/2/3] | avail_cash | alloc "
          "| weakest_RSR | 4th_20d | weakest_20d | diff |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for t in all_triggers:
            existing = "/".join(str(r) for r in t.get("existing_rsrs", []))
            diff_20d = (
                round(t["future_20d"] - t["weakest_future_20d"], 2)
                if t.get("future_20d") is not None and t.get("weakest_future_20d") is not None
                else None
            )
            w(f"| {t['date']} "
              f"| {t['ticker']} "
              f"| {t['candidate_rsr']:.1f} "
              f"| {existing} "
              f"| ¥{t['available_cash']:,.0f} "
              f"| ¥{t['alloc']:,.0f} "
              f"| {t['weakest_rsr']:.1f}({t.get('weakest_sym','')}) "
              f"| {t['future_20d'] if t['future_20d'] is not None else '-'} "
              f"| {t['weakest_future_20d'] if t['weakest_future_20d'] is not None else '-'} "
              f"| {diff_20d if diff_20d is not None else '-'} |")
    else:
        w("_(トリガーなし)_")

    # ── Section 7: 年次リターン比較 ───────────────────────────────
    w("\n---\n## 7. 年次リターン比較 (%)\n")
    all_years = sorted({y for r in fold_rows for y in r["A"].get("annual_returns", {}).keys()})
    w("| 年 | A (Baseline) | F (Rank1-4th) | ΔCAGR |")
    w("|---|---|---|---|")
    yr_seen: set = set()
    for row in fold_rows:
        test_yr = row.get("test_year", "")
        if test_yr in yr_seen:
            continue
        yr_seen.add(test_yr)
        a_yr = row["A"].get("annual_returns", {}).get(test_yr, 0)
        f_yr = row["F"].get("annual_returns", {}).get(test_yr, 0)
        w(f"| {test_yr} | {a_yr:+.1f}% | {f_yr:+.1f}% | {f_yr - a_yr:+.1f}pp |")

    # ── Section 8: 採用判定 ────────────────────────────────────────
    w("\n---\n## 8. 採用判定\n")

    criteria = {
        "WF通過": wf_pass >= WF_PASS_MIN,
        "trigger/year": avg_trigger_y <= ADOPT_TRIGGER_MAX,
        "ΔCAGR≥+1pp": avg_dcagr >= ADOPT_DCAGR_MIN,
        "ΔMaxDD≤+2pp": avg_dmaxdd <= ADOPT_DMAXDD_MAX,
        "gain_per_trigger>0": avg_gpt > 0,
        "winner>weakest": wvw.get("avg_diff_20d", -1) > 0 if wvw else False,
    }
    all_pass = all(criteria.values())
    verdict = "**ADOPT (採用候補)** ✅" if all_pass else "**REJECT (破棄)** ❌"

    w(f"### 判定: {verdict}\n")
    w("| 条件 | 基準 | 実測値 | 判定 |")
    w("|---|---|---|---|")
    labels = {
        "WF通過":          f"≥{WF_PASS_MIN}/5",
        "trigger/year":    f"≤{ADOPT_TRIGGER_MAX}回/年",
        "ΔCAGR≥+1pp":      "≥+1pp",
        "ΔMaxDD≤+2pp":     "≤+2pp",
        "gain_per_trigger>0": ">0",
        "winner>weakest":  "winner_ret > weakest_ret",
    }
    values = {
        "WF通過":          f"{wf_pass}/5",
        "trigger/year":    f"{avg_trigger_y:.1f}回/年",
        "ΔCAGR≥+1pp":      f"{avg_dcagr:+.2f}pp",
        "ΔMaxDD≤+2pp":     f"{avg_dmaxdd:+.2f}pp",
        "gain_per_trigger>0": f"¥{avg_gpt:+,.0f}",
        "winner>weakest":  (f"diff={wvw.get('avg_diff_20d',0):+.2f}%" if wvw else "データ不足"),
    }
    for k, ok in criteria.items():
        w(f"| {k} | {labels[k]} | {values[k]} | {'✅' if ok else '❌'} |")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート保存: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  Rank1 条件付き4枚目ポジション WF 研究")
    print("=" * 68 + "\n")

    print("[1/3] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    def run(label, start, end, enable):
        mode = "Variant" if enable else "Baseline"
        print(f"  [{label} {mode}] {start}〜{end}", end=" ", flush=True)
        t0  = time.time()
        res = run_backtest(
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
            start=start, end=end,
            enable_rank1_4th=enable,
        )
        m = res.get("metrics", {})
        tc = m.get("trigger_count", 0)
        print(f"  CAGR={m.get('cagr',0):+.1f}%  Calmar={m.get('calmar',0):.3f}  "
              f"MaxDD={m.get('max_dd',0):.1f}%  triggers={tc}  ({time.time()-t0:.1f}s)")
        return res

    print("\n[2/3] WF 5-Fold 実行...")
    fold_rows = []
    all_triggers = []

    for fold_name, tr_s, tr_e, te_s, te_e, label in FOLDS:
        print(f"\n  {fold_name} [{label}] Test: {te_s}〜{te_e}")
        res_a = run(fold_name, te_s, te_e, enable=False)
        res_f = run(fold_name, te_s, te_e, enable=True)

        m_a = res_a.get("metrics", {})
        m_f = res_f.get("metrics", {})
        trig = res_f.get("triggers", [])
        all_triggers.extend(trig)

        test_years = m_a.get("n_dates", 252) / 252

        fold_rows.append({
            "fold":          fold_name,
            "label":         label,
            "test_year":     te_s[:4],
            "test_years":    round(test_years, 2),
            "A":             m_a,
            "F":             m_f,
            "trigger_count": len(trig),
            "triggers":      trig,
        })

    print(f"\n[3/3] 分析 + レポート生成...")
    print(f"  全Fold トリガー合計: {len(all_triggers)}件")

    out_path = REPORTS_DIR / "rank1_4th_position_wf.md"
    write_report(fold_rows, all_triggers, out_path)

    # ── Console summary ────────────────────────────────────────────
    print("\n" + "="*68)
    print("  ★ WF サマリ")
    print("="*68)
    print(f"{'Fold':<10} {'A CAGR':>8} {'F CAGR':>8} {'ΔCAGR':>7} {'ΔMaxDD':>7} {'trig/y':>7} {'判定':>4}")
    print("-"*60)

    wf_pass = 0
    for row in fold_rows:
        a = row["A"]; f = row["F"]
        dcagr  = f["cagr"]   - a["cagr"]
        dmaxdd = f["max_dd"] - a["max_dd"]
        trig_y = round(row["trigger_count"] / max(row["test_years"], 0.01), 1)
        ok     = dcagr >= 0 and dmaxdd <= 2.0
        if ok:
            wf_pass += 1
        print(f"{row['fold']:<10} {a['cagr']:>+7.1f}% {f['cagr']:>+7.1f}%"
              f" {dcagr:>+6.1f}pp {dmaxdd:>+6.1f}pp {trig_y:>6.1f}  {'✅' if ok else '❌'}")

    print("-"*60)
    print(f"WF通過: {wf_pass}/5  全トリガー: {len(all_triggers)}件")

    wvw = compare_winner_vs_weakest(all_triggers)
    if wvw:
        print(f"winner_20d={wvw['avg_winner_ret_20d']:+.2f}%  "
              f"weakest_20d={wvw['avg_weakest_ret_20d']:+.2f}%  "
              f"diff={wvw['avg_diff_20d']:+.2f}%  "
              f"winner>weakest: {wvw['winner_above_weakest_pct']:.1f}%")

    avg_dcagr = float(np.mean([r["F"]["cagr"] - r["A"]["cagr"] for r in fold_rows]))
    avg_dmaxdd = float(np.mean([r["F"]["max_dd"] - r["A"]["max_dd"] for r in fold_rows]))
    avg_gpt = float(np.mean([r["F"].get("gain_per_trigger", 0) for r in fold_rows]))

    all_ok = (
        wf_pass >= WF_PASS_MIN
        and (len(all_triggers) / max(sum(r["test_years"] for r in fold_rows), 1)) <= ADOPT_TRIGGER_MAX
        and avg_dcagr >= ADOPT_DCAGR_MIN
        and avg_dmaxdd <= ADOPT_DMAXDD_MAX
        and avg_gpt > 0
        and (wvw.get("avg_diff_20d", -1) > 0 if wvw else False)
    )
    print(f"\n最終判定: {'ADOPT ✅' if all_ok else 'REJECT ❌'}")
    print(f"  ΔCAGR={avg_dcagr:+.2f}pp  ΔMaxDD={avg_dmaxdd:+.2f}pp  GPT=¥{avg_gpt:+,.0f}")
    print(f"\n  レポート → {out_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
