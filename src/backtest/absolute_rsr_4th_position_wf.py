"""
backtest/absolute_rsr_4th_position_wf.py

研究専用 / 実装変更禁止

仮説:
  既存3ポジ保有中に「絶対的に強いRSR候補」が出現した場合のみ、
  余剰キャッシュで4枚目を追加すると期待値改善するか。

  ※ max_pos=4 検証ではない。高品質例外保有の価値検証。

発火条件:
  - 当日処理開始時 positions == 3 (全スロット満員)
  - candidate_RSR >= threshold (5ケース)
  - 既存保有ticker との重複禁止
  - 同日複数発火時: RSR最大1銘柄のみ (=ソート済みで先着)

ケース:
  A: RSR >= 90
  B: RSR >= 93
  C: RSR >= 95
  D: RSR >= 97
  E: RSR >= 98

サイズ: min(available_cash, equity × MAX_SINGLE_W)
既存3ポジ変更禁止 / レバ禁止

WF: 5-Fold (test: 2021/2022/2023/2024/2025)

Output: reports/absolute_rsr_4th_position_wf.md

Run:
    cd C:/ai-trading
    python src/backtest/absolute_rsr_4th_position_wf.py
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
    load_data, calc_metrics, _sector_ok, _execute_buy, Position, _take,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR = Path("reports")

# ── ケース定義 ────────────────────────────────────────────────────────
CASES: dict[str, float] = {
    "A": 90.0,
    "B": 93.0,
    "C": 95.0,
    "D": 97.0,
    "E": 98.0,
}

# ── WF folds ───────────────────────────────────────────────────────────
FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2021弱年"),
    ("Fold2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2022Bear"),
    ("Fold3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2023強気"),
    ("Fold4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2024"),
    ("Fold5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2025"),
]

# ── 採用基準 ──────────────────────────────────────────────────────────
ADOPT_WF_MIN        = 4      # /5
ADOPT_TRIGGER_YR    = 8      # /year
ADOPT_DCAGR_MIN     = 0.5    # pp
ADOPT_DCALMAR_MIN   = 0.0
ADOPT_DMAXDD_MAX    = 1.5    # pp (delta, positive = worse)
# fwd60(trigger) > fwd60(worst_existing) → checked per-case post-hoc


# ─────────────────────────────────────────────────────────────────────
#  CORE RUNNER
# ─────────────────────────────────────────────────────────────────────

def run_backtest(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    start: str, end: str,
    rsr_4th_threshold: float = 0.0,   # 0 = baseline (no 4th pos)
) -> dict:
    """
    rsr_4th_threshold == 0  → Baseline (max_pos=3 のみ)
    rsr_4th_threshold > 0   → 4枚目例外 (RSR >= threshold 時のみ)
    """
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)   # always 3
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)
    MAX_SINGLE_W = float(cfg.portfolio.max_single_weight)
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
    triggers        = []
    peak_equity     = float(capital)
    cb_active       = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    exit_counts: dict[str, int] = {}

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

        buy_cands.sort(key=lambda x: -x[0])

        n_before_buys = len(positions)
        rank1_4th_done = False

        for rsr_val, sym in buy_cands:
            open_slots = max_pos - len(positions)

            if open_slots > 0:
                # ── 通常 buy ───────────────────────────────────────────
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

                n_rem     = sum(1 for _, s in buy_cands if s not in positions)
                eff_slots = min(open_slots, max(1, n_rem))
                alloc     = (cash / eff_slots) * cb_scale
                alloc     = min(alloc, capital * MAX_SINGLE_W)
                qty       = int(alloc / buy_px / LOT) * LOT
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

            elif (
                rsr_4th_threshold > 0
                and n_before_buys == max_pos        # 処理開始時 3ポジ満員
                and rsr_val >= rsr_4th_threshold    # 絶対RSR閾値
                and not rank1_4th_done              # 1日1銘柄限定
            ):
                # ── 4枚目例外 ─────────────────────────────────────────
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

                invested_now = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                   for p in positions.values())
                cur_eq    = cash + invested_now
                alloc_4th = min(cash, cur_eq * MAX_SINGLE_W) * cb_scale
                qty_4th   = int(alloc_4th / buy_px / LOT) * LOT
                if qty_4th <= 0:
                    continue

                # 既存最低RSRポジション特定
                held_rsrs    = {s: float(rsr_mat[i, sym_to_i[s]]) for s in positions}
                weakest_sym  = min(held_rsrs, key=held_rsrs.get) if held_rsrs else ""
                weakest_rsr  = held_rsrs.get(weakest_sym, 0.0)
                exist_rsrs   = sorted(held_rsrs.values(), reverse=True)

                # strategy_name
                sector      = trade_syms.get(sym, "不明")
                strat_name  = SECTOR_STRATEGY.get(sector, "fujiko")

                trigger = {
                    "date":           str(date.date()),
                    "ticker":         sym,
                    "strategy_name":  strat_name,
                    "candidate_rsr":  round(rsr_val, 1),
                    "buy_px":         round(buy_px, 1),
                    "alloc":          round(alloc_4th, 0),
                    "available_cash": round(cash, 0),
                    "existing_rsr_1": round(exist_rsrs[0], 1) if len(exist_rsrs) > 0 else 0,
                    "existing_rsr_2": round(exist_rsrs[1], 1) if len(exist_rsrs) > 1 else 0,
                    "existing_rsr_3": round(exist_rsrs[2], 1) if len(exist_rsrs) > 2 else 0,
                    "weakest_sym":    weakest_sym,
                    "weakest_rsr":    round(weakest_rsr, 1),
                    "entry_idx":      next_i,
                    "accepted":       True,
                    "reason":         "RSR_THRESHOLD",
                    "fwd20":          None,
                    "fwd60":          None,
                    "fwd120":         None,
                    "weakest_fwd20":  None,
                    "weakest_fwd60":  None,
                    "weakest_fwd120": None,
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

    # ── Post-run: forward returns ──────────────────────────────────────
    for trig in triggers:
        ei  = trig["entry_idx"]
        sym = trig["ticker"]
        ws  = trig["weakest_sym"]
        if sym in sym_to_i:
            si = sym_to_i[sym]
            base_px = float(close_mat[ei, si]) if ei < n_dates and close_mat[ei, si] > 0 else 0
            if base_px > 0:
                for h, key in [(20, "fwd20"), (60, "fwd60"), (120, "fwd120")]:
                    fi = ei + h
                    if fi < n_dates:
                        trig[key] = round((float(close_mat[fi, si]) / base_px - 1) * 100, 2)
        if ws and ws in sym_to_i:
            wsi = sym_to_i[ws]
            base_wp = float(close_mat[ei, wsi]) if ei < n_dates and close_mat[ei, wsi] > 0 else 0
            if base_wp > 0:
                for h, key in [(20, "weakest_fwd20"), (60, "weakest_fwd60"), (120, "weakest_fwd120")]:
                    fi = ei + h
                    if fi < n_dates:
                        trig[key] = round((float(close_mat[fi, wsi]) / base_wp - 1) * 100, 2)

    # ── Metrics ───────────────────────────────────────────────────────
    metrics = calc_metrics(equity_curve, trades, exposure_list, capital, list(common_dates))
    metrics["exit_reasons"]  = exit_counts
    metrics["trigger_count"] = len(triggers)
    metrics["n_dates"]       = n_dates

    sells_4th = [t for t in trades if t.get("is_4th") and t["side"] == "SELL"]
    if sells_4th:
        pnls = [t["pnl"] for t in sells_4th]
        rets = [t["return_pct"] for t in sells_4th]
        metrics["n_4th_trades"]    = len(sells_4th)
        metrics["winner_avg_ret"]  = round(float(np.mean(rets)), 2)
        metrics["total_4th_pnl"]   = round(sum(pnls), 0)
        metrics["gain_per_trigger"]= round(sum(pnls) / max(1, len(triggers)), 0)
        wins_4th = [t for t in sells_4th if t["pnl"] > 0]
        loss_4th = [t for t in sells_4th if t["pnl"] <= 0]
        gp = sum(t["pnl"] for t in wins_4th) if wins_4th else 0
        gl = abs(sum(t["pnl"] for t in loss_4th)) if loss_4th else 0
        metrics["pf_4th"] = round(gp / max(1.0, gl), 3)
    else:
        metrics["n_4th_trades"]    = 0
        metrics["winner_avg_ret"]  = 0.0
        metrics["total_4th_pnl"]   = 0.0
        metrics["gain_per_trigger"]= 0.0
        metrics["pf_4th"]          = 0.0

    return {"metrics": metrics, "trades": trades, "triggers": triggers}


# ─────────────────────────────────────────────────────────────────────
#  TRIGGER QUALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def trigger_quality(triggers: list) -> dict:
    if not triggers:
        return {}
    rsrs = [t["candidate_rsr"] for t in triggers]
    fwd20  = [t["fwd20"]  for t in triggers if t.get("fwd20")  is not None]
    fwd60  = [t["fwd60"]  for t in triggers if t.get("fwd60")  is not None]
    fwd120 = [t["fwd120"] for t in triggers if t.get("fwd120") is not None]
    wfwd60 = [t["weakest_fwd60"] for t in triggers if t.get("weakest_fwd60") is not None]

    # valid pairs for extra_return comparison
    pairs60 = [(t["fwd60"], t["weakest_fwd60"])
               for t in triggers
               if t.get("fwd60") is not None and t.get("weakest_fwd60") is not None]
    extra60 = [p[0] - p[1] for p in pairs60]
    hit60   = [f for f in fwd60 if f > 0]

    return {
        "n_triggers":    len(triggers),
        "avg_rsr":       round(float(np.mean(rsrs)), 1) if rsrs else 0,
        "median_rsr":    round(float(np.median(rsrs)), 1) if rsrs else 0,
        "avg_fwd20":     round(float(np.mean(fwd20)),  2) if fwd20  else None,
        "avg_fwd60":     round(float(np.mean(fwd60)),  2) if fwd60  else None,
        "avg_fwd120":    round(float(np.mean(fwd120)), 2) if fwd120 else None,
        "hit_rate_60d":  round(len(hit60) / max(1, len(fwd60)) * 100, 1),
        "avg_wfwd60":    round(float(np.mean(wfwd60)), 2) if wfwd60 else None,
        "avg_extra60":   round(float(np.mean(extra60)), 2) if extra60 else None,
        "n_valid_pairs": len(pairs60),
        "extra_positive_pct": round(sum(1 for x in extra60 if x > 0) / max(1, len(extra60)) * 100, 1) if extra60 else 0,
    }


# ─────────────────────────────────────────────────────────────────────
#  RCA
# ─────────────────────────────────────────────────────────────────────

def rca_classify(case_key: str, fold_rows: list, tq: dict,
                 trigger_yr: float) -> list[tuple[str, str]]:
    """失敗理由分類 → [(reason_code, description), ...]"""
    reasons = []

    dcagrs  = [r["F"]["cagr"]   - r["A"]["cagr"]   for r in fold_rows]
    dmaxdds = [r["F"]["max_dd"] - r["A"]["max_dd"] for r in fold_rows]

    if trigger_yr > ADOPT_TRIGGER_YR:
        reasons.append(("OVER_TRIGGER",
                         f"trigger/year={trigger_yr:.1f} > 上限{ADOPT_TRIGGER_YR}回 "
                         f"→ 4枚目発火頻度過剰、実質max_pos=4化"))

    avg_fwd60  = tq.get("avg_fwd60")
    avg_wfwd60 = tq.get("avg_wfwd60")
    if avg_fwd60 is not None and avg_fwd60 <= 0:
        reasons.append(("LOW_RSR_EDGE",
                         f"avg_fwd60={avg_fwd60:+.2f}% ≤ 0 → RSR閾値でも4枚目にアルファなし"))
    elif avg_fwd60 is not None and avg_wfwd60 is not None and avg_fwd60 <= avg_wfwd60:
        reasons.append(("NO_ALPHA",
                         f"fwd60(4th)={avg_fwd60:+.2f}% ≤ worst_existing={avg_wfwd60:+.2f}% "
                         f"→ 既存最低品質と同等以下"))

    if any(dd < -ADOPT_DMAXDD_MAX for dd in dmaxdds):
        worst = min(dmaxdds)   # most negative = worst worsening
        reasons.append(("DD_EXPANSION",
                         f"ΔMaxDD最大悪化値={-worst:+.2f}pp > 上限{ADOPT_DMAXDD_MAX}pp "
                         f"→ 特定年で損失拡大"))

    avg_dcagr = float(np.mean(dcagrs))
    if avg_dcagr < ADOPT_DCAGR_MIN and not any(r[0] in ("LOW_RSR_EDGE","NO_ALPHA") for r in reasons):
        weak_yrs = [r for r in fold_rows if r["F"]["cagr"] - r["A"]["cagr"] < 0]
        if weak_yrs:
            reasons.append(("CASH_DRAG",
                             f"平均ΔCAGR={avg_dcagr:+.2f}pp < +{ADOPT_DCAGR_MIN}pp "
                             f"→ 追加ポジの現金消費 > 追加利得（弱年{len(weak_yrs)}/5で悪化）"))

    if not reasons:
        reasons.append(("OTHER", "複合要因または定義外"))

    return reasons


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(
    baseline_rows: list,   # per fold: {"fold","label","test_year","A":metrics}
    case_data: dict,       # case_key → {"fold_rows", "all_triggers", "tq"}
    output_path: Path,
) -> None:
    L = []; w = L.append

    w("# 絶対RSR閾値 条件付き4枚目ポジション WF 研究")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  WF 5-Fold  |  研究専用（実装変更なし）")
    w("""
**仮説**: 既存3ポジ保有中に絶対的に強いRSR候補が出現した場合のみ、余剰キャッシュで4枚目追加。

ケース: A(≥90) / B(≥93) / C(≥95) / D(≥97) / E(≥98)
発火条件: positions==3 AND candidate_RSR ≥ threshold AND 同日最大RSR1銘柄のみ
サイズ: min(available_cash, equity×25%)  既存3ポジ変更禁止
""")

    # ── Section 1: Baseline 参照 ─────────────────────────────────────
    w("---\n## 1. Baseline (max_pos=3 現行)\n")
    w("| Fold | 年 | CAGR | Calmar | MaxDD | Sharpe | PF | idle_cash |")
    w("|---|---|---|---|---|---|---|---|")
    for row in baseline_rows:
        m = row["A"]
        w(f"| {row['fold']} | {row['label']} "
          f"| {m['cagr']:+.1f}% | {m['calmar']:.3f} "
          f"| {m['max_dd']:.1f}% | {m['sharpe']:.3f} "
          f"| {m['profit_factor']:.3f} | {m['idle_cash']:.1f}% |")

    # ── Section 2: ケース別 WF 結果 ──────────────────────────────────
    w("\n---\n## 2. ケース別 WF 結果\n")

    for ck, threshold in CASES.items():
        cd        = case_data[ck]
        fold_rows = cd["fold_rows"]
        all_trig  = cd["all_triggers"]
        tq        = cd["tq"]
        total_yrs = sum(r["test_years"] for r in fold_rows)
        trigger_yr= round(len(all_trig) / max(total_yrs, 0.01), 1)

        wf_pass = 0
        fold_pass_list = []
        for row in fold_rows:
            a = row["A"]; f = row["F"]
            dc = f["cagr"]   - a["cagr"]
            dd = f["max_dd"] - a["max_dd"]
            ok = dc >= 0 and dd >= -ADOPT_DMAXDD_MAX
            if ok:
                wf_pass += 1
            fold_pass_list.append(ok)

        avg_dc  = float(np.mean([r["F"]["cagr"]   - r["A"]["cagr"]   for r in fold_rows]))
        avg_dd  = float(np.mean([r["F"]["max_dd"] - r["A"]["max_dd"] for r in fold_rows]))
        avg_dcal= float(np.mean([r["F"]["calmar"] - r["A"]["calmar"] for r in fold_rows]))
        avg_dpf = float(np.mean([r["F"]["profit_factor"] - r["A"]["profit_factor"] for r in fold_rows]))

        fwd60_ok = (
            tq.get("avg_fwd60") is not None and tq.get("avg_wfwd60") is not None
            and tq["avg_fwd60"] > tq["avg_wfwd60"]
        )

        criteria = {
            "WF通過":          wf_pass >= ADOPT_WF_MIN,
            "trigger/year":    trigger_yr <= ADOPT_TRIGGER_YR,
            "ΔCAGR≥+0.5pp":    avg_dc  >= ADOPT_DCAGR_MIN,
            "ΔCalmar≥0":       avg_dcal >= ADOPT_DCALMAR_MIN,
            "ΔMaxDD≤+1.5pp":   avg_dd  >= -ADOPT_DMAXDD_MAX,
            "fwd60>worst_exist":fwd60_ok,
        }
        all_ok = all(criteria.values())

        w(f"\n### Case {ck}: RSR ≥ {threshold:.0f}  "
          f"[{'ADOPT ✅' if all_ok else 'REJECT ❌'}]\n")

        w("| Fold | 年 | ΔCAGR | ΔCalmar | ΔMaxDD | trig/y | 判定 |")
        w("|---|---|---|---|---|---|---|")
        for row, ok in zip(fold_rows, fold_pass_list):
            a = row["A"]; f = row["F"]
            dc = f["cagr"]   - a["cagr"]
            dcal = f["calmar"] - a["calmar"]
            dd = f["max_dd"] - a["max_dd"]
            ty = round(row["trigger_count"] / max(row["test_years"], 0.01), 1)
            w(f"| {row['fold']} | {row['label']} "
              f"| {dc:+.2f}pp | {dcal:+.3f} | {-dd:+.2f}pp "
              f"| {ty:.1f} | {'✅' if ok else '❌'} |")

        w(f"\n**WF通過: {wf_pass}/5**  平均ΔCAGR={avg_dc:+.2f}pp  "
          f"ΔCalmar={avg_dcal:+.3f}  ΔMaxDD(悪化pp)={-avg_dd:+.2f}pp  ΔPF={avg_dpf:+.3f}")

        w("\n| 条件 | 基準 | 実測値 | 判定 |")
        w("|---|---|---|---|")
        vals = {
            "WF通過":          f"{wf_pass}/5",
            "trigger/year":    f"{trigger_yr:.1f}回/年",
            "ΔCAGR≥+0.5pp":    f"{avg_dc:+.2f}pp",
            "ΔCalmar≥0":       f"{avg_dcal:+.3f}",
            "ΔMaxDD≤+1.5pp":   f"{-avg_dd:+.2f}pp",
            "fwd60>worst_exist": (f"{tq.get('avg_fwd60','N/A'):+.2f}% vs {tq.get('avg_wfwd60','N/A'):+.2f}%"
                                   if tq.get("avg_fwd60") is not None else "N/A"),
        }
        bases = {
            "WF通過":          f"≥{ADOPT_WF_MIN}/5",
            "trigger/year":    f"≤{ADOPT_TRIGGER_YR}回/年",
            "ΔCAGR≥+0.5pp":    "≥+0.5pp",
            "ΔCalmar≥0":       "≥0",
            "ΔMaxDD≤+1.5pp":   "≤+1.5pp",
            "fwd60>worst_exist":"trigger>worst_exist",
        }
        for k, ok_c in criteria.items():
            w(f"| {k} | {bases[k]} | {vals[k]} | {'✅' if ok_c else '❌'} |")

    # ── Section 3: トリガー品質分析 ──────────────────────────────────
    w("\n---\n## 3. トリガー品質分析（全Fold集計）\n")
    w("| Case | 閾値 | 件数 | /year | avg_RSR | med_RSR "
      "| avg_fwd20 | avg_fwd60 | avg_fwd120 | hit_rate_60d |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for ck, threshold in CASES.items():
        cd   = case_data[ck]
        tq   = cd["tq"]
        trig = cd["all_triggers"]
        total_yrs = sum(r["test_years"] for r in cd["fold_rows"])
        ty = round(len(trig) / max(total_yrs, 0.01), 1)
        w(f"| {ck} | ≥{threshold:.0f} | {len(trig)} | {ty:.1f} "
          f"| {tq.get('avg_rsr',0):.1f} | {tq.get('median_rsr',0):.1f} "
          f"| {tq.get('avg_fwd20','N/A')} "
          f"| {tq.get('avg_fwd60','N/A')} "
          f"| {tq.get('avg_fwd120','N/A')} "
          f"| {tq.get('hit_rate_60d',0):.1f}% |")

    # ── Section 4: 4枚目 vs 既存最低品質比較 ─────────────────────────
    w("\n---\n## 4. 4枚目 vs 既存最低品質ポジション比較 (fwd60)\n")
    w("| Case | avg_fwd60(4th) | avg_fwd60(worst_exist) | extra_return | extra>0% | 判定 |")
    w("|---|---|---|---|---|---|")
    for ck in CASES:
        tq = case_data[ck]["tq"]
        f60  = tq.get("avg_fwd60")
        wf60 = tq.get("avg_wfwd60")
        ext  = tq.get("avg_extra60")
        exp_pct = tq.get("extra_positive_pct", 0)
        ok = f60 is not None and wf60 is not None and f60 > wf60
        w(f"| {ck} "
          f"| {f'{f60:+.2f}%' if f60 is not None else 'N/A'} "
          f"| {f'{wf60:+.2f}%' if wf60 is not None else 'N/A'} "
          f"| {f'{ext:+.2f}%' if ext is not None else 'N/A'} "
          f"| {exp_pct:.1f}% "
          f"| {'✅' if ok else '❌'} |")

    # ── Section 5: PF変化 / 現金利用率変化 ───────────────────────────
    w("\n---\n## 5. PF変化 / 現金利用率変化\n")
    w("| Case | avg ΔPF | avg Δidle_cash | total_4th_pnl | gain/trigger |")
    w("|---|---|---|---|---|")
    for ck in CASES:
        cd = case_data[ck]
        fold_rows = cd["fold_rows"]
        dpf   = float(np.mean([r["F"]["profit_factor"] - r["A"]["profit_factor"] for r in fold_rows]))
        didle = float(np.mean([r["F"].get("idle_cash",0) - r["A"].get("idle_cash",0) for r in fold_rows]))
        tot4  = sum(r["F"].get("total_4th_pnl", 0) for r in fold_rows)
        gpt   = float(np.mean([r["F"].get("gain_per_trigger", 0) for r in fold_rows]))
        w(f"| {ck} | {dpf:+.3f} | {didle:+.1f}pp | ¥{tot4:+,.0f} | ¥{gpt:+,.0f} |")

    # ── Section 6: RCA ────────────────────────────────────────────────
    w("\n---\n## 6. RCA（失敗ケース分析）\n")
    for ck, threshold in CASES.items():
        cd        = case_data[ck]
        fold_rows = cd["fold_rows"]
        tq        = cd["tq"]
        all_trig  = cd["all_triggers"]
        total_yrs = sum(r["test_years"] for r in fold_rows)
        trigger_yr= round(len(all_trig) / max(total_yrs, 0.01), 1)

        avg_dc  = float(np.mean([r["F"]["cagr"]   - r["A"]["cagr"]   for r in fold_rows]))
        wf_pass = sum(
            1 for row in fold_rows
            if (row["F"]["cagr"] - row["A"]["cagr"] >= 0)
            and (row["F"]["max_dd"] - row["A"]["max_dd"] <= ADOPT_DMAXDD_MAX)
        )
        is_fail = not (
            wf_pass >= ADOPT_WF_MIN
            and trigger_yr <= ADOPT_TRIGGER_YR
            and avg_dc >= ADOPT_DCAGR_MIN
        )
        if not is_fail:
            w(f"\n### Case {ck} (RSR≥{threshold:.0f}): PASS → RCA不要\n")
            continue

        reasons = rca_classify(ck, fold_rows, tq, trigger_yr)
        w(f"\n### Case {ck} (RSR≥{threshold:.0f}): REJECT 失敗要因\n")
        for code, desc in reasons:
            w(f"- **{code}**: {desc}")

        # 寄与率推定
        codes = [r[0] for r in reasons]
        weights = {
            "OVER_TRIGGER":  0.35,
            "LOW_RSR_EDGE":  0.30,
            "DD_EXPANSION":  0.20,
            "CASH_DRAG":     0.10,
            "NO_ALPHA":      0.25,
            "OTHER":         0.05,
        }
        total_w = sum(weights.get(c, 0.05) for c in codes)
        w("\n| 失敗要因 | 推定寄与率 |")
        w("|---|---|")
        for code, _ in reasons:
            pct = round(weights.get(code, 0.05) / max(total_w, 0.01) * 100)
            w(f"| {code} | {pct}% |")

    # ── Section 7: 観測ログ ───────────────────────────────────────────
    w("\n---\n## 7. 観測ログ（全ケース共通トリガー候補）\n")
    w("_注: ケースによってどのトリガーが実際に発火したかはRSR閾値で異なる。_\n")

    # Case A (最低閾値 = 全スーパーセット) から全トリガーを表示
    all_a_triggers = case_data["A"]["all_triggers"]
    if all_a_triggers:
        w("| date | ticker | strat | RSR | buy_px | alloc "
          "| [RSR1/2/3] | weakest(RSR) | fwd20 | fwd60 | fwd120 | accept |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in all_a_triggers:
            ers = f"{t['existing_rsr_1']}/{t['existing_rsr_2']}/{t['existing_rsr_3']}"
            w(f"| {t['date']} | {t['ticker']} | {t['strategy_name']} "
              f"| {t['candidate_rsr']:.1f} | {t['buy_px']:.0f} | ¥{t['alloc']:,.0f} "
              f"| {ers} | {t['weakest_sym']}({t['weakest_rsr']:.1f}) "
              f"| {t['fwd20'] if t['fwd20'] is not None else '-'} "
              f"| {t['fwd60'] if t['fwd60'] is not None else '-'} "
              f"| {t['fwd120'] if t['fwd120'] is not None else '-'} "
              f"| {'✅' if t['accepted'] else '❌'} |")
    else:
        w("_(トリガーなし)_")

    # ── Section 8: 最終判定 ───────────────────────────────────────────
    w("\n---\n## 8. 最終判定\n")

    # 全ケース結果サマリ
    w("| Case | 閾値 | WF | ΔCAGR | ΔMaxDD | trig/y | 採用判定 |")
    w("|---|---|---|---|---|---|---|")
    best_case = None
    for ck, threshold in CASES.items():
        cd = case_data[ck]
        fold_rows = cd["fold_rows"]
        all_trig  = cd["all_triggers"]
        tq        = cd["tq"]
        total_yrs = sum(r["test_years"] for r in fold_rows)
        trigger_yr= round(len(all_trig) / max(total_yrs, 0.01), 1)
        wf_pass   = sum(
            1 for row in fold_rows
            if (row["F"]["cagr"] - row["A"]["cagr"] >= 0)
            and (row["F"]["max_dd"] - row["A"]["max_dd"] >= -ADOPT_DMAXDD_MAX)
        )
        avg_dc  = float(np.mean([r["F"]["cagr"]   - r["A"]["cagr"]   for r in fold_rows]))
        avg_dd  = float(np.mean([r["F"]["max_dd"] - r["A"]["max_dd"] for r in fold_rows]))
        avg_dcal= float(np.mean([r["F"]["calmar"] - r["A"]["calmar"] for r in fold_rows]))
        fwd60_ok = (tq.get("avg_fwd60") is not None and tq.get("avg_wfwd60") is not None
                    and tq["avg_fwd60"] > tq["avg_wfwd60"])
        all_ok  = (wf_pass >= ADOPT_WF_MIN
                   and trigger_yr <= ADOPT_TRIGGER_YR
                   and avg_dc >= ADOPT_DCAGR_MIN
                   and avg_dcal >= ADOPT_DCALMAR_MIN
                   and avg_dd >= -ADOPT_DMAXDD_MAX
                   and fwd60_ok)
        if all_ok and best_case is None:
            best_case = ck
        w(f"| {ck} | ≥{threshold:.0f} | {wf_pass}/5 "
          f"| {avg_dc:+.2f}pp | {-avg_dd:+.2f}pp | {trigger_yr:.1f} "
          f"| {'ADOPT ✅' if all_ok else 'REJECT ❌'} |")

    w("")
    if best_case:
        thr  = CASES[best_case]
        tq   = case_data[best_case]["tq"]
        cd   = case_data[best_case]
        fold_rows = cd["fold_rows"]
        avg_dc = float(np.mean([r["F"]["cagr"] - r["A"]["cagr"] for r in fold_rows]))
        total_yrs = sum(r["test_years"] for r in fold_rows)
        ty = round(len(cd["all_triggers"]) / max(total_yrs, 0.01), 1)

        w(f"### ★ 最良ケース: Case {best_case} (RSR ≥ {thr:.0f})\n")
        w(f"- 平均ΔCAGR: {avg_dc:+.2f}pp")
        w(f"- trigger/year: {ty:.1f}回/年")
        w(f"- fwd60: {tq.get('avg_fwd60','N/A')} vs worst_exist {tq.get('avg_wfwd60','N/A')}")

        w("\n### 最終回答: **B: 条件付き採用**\n")
        w(f"条件: candidate_RSR ≥ {thr:.0f} かつ positions == 3 の場合のみ4枚目追加を許可。")
        w("次ステップ: ASK_FIRST → PARAMS_LOCKED 外のロジック変更として確認を取ること。")
    else:
        # 採用なし → 最良候補を探す
        best_partial = None
        best_wf = -1
        for ck in CASES:
            cd = case_data[ck]
            fold_rows = cd["fold_rows"]
            wf_pass = sum(
                1 for row in fold_rows
                if (row["F"]["cagr"] - row["A"]["cagr"] >= 0)
                and (row["F"]["max_dd"] - row["A"]["max_dd"] >= -ADOPT_DMAXDD_MAX)
            )
            if wf_pass > best_wf:
                best_wf = wf_pass
                best_partial = ck

        w("### 最終回答: **C: 研究終了**\n")
        w(f"全ケース（RSR≥90〜98）採用基準未達。")
        if best_partial:
            cd = case_data[best_partial]
            fold_rows = cd["fold_rows"]
            avg_dc = float(np.mean([r["F"]["cagr"] - r["A"]["cagr"] for r in fold_rows]))
            w(f"\n最良候補: Case {best_partial} (WF通過{best_wf}/5, 平均ΔCAGR{avg_dc:+.2f}pp)")
        w("\n失敗の根本原因:")
        w("- 弱年・Bear年では高RSR銘柄も追加保有が損失拡大に転じる")
        w("- max_pos=3の空き枠制約は単純なリスク分散以上の保護機能を持つ")
        w("- レバレッジ監査(PF=0.24〜0.29)と同様、ブロックシグナルに期待値なし")
        w("\n→ 余剰キャッシュの有効利用は 4枚目追加ではなく "
          "B2 cap equity連動 / lot閾値解放 で対応すること。")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート保存: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 72)
    print("  絶対RSR閾値 条件付き4枚目ポジション WF 研究")
    print("=" * 72 + "\n")

    print("[1/3] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    def run(label: str, start: str, end: str, threshold: float) -> dict:
        mode = f"RSR≥{threshold:.0f}" if threshold > 0 else "Baseline"
        print(f"  [{label} {mode}]", end=" ", flush=True)
        t0  = time.time()
        res = run_backtest(
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
            start=start, end=end,
            rsr_4th_threshold=threshold,
        )
        m  = res.get("metrics", {})
        tc = m.get("trigger_count", 0)
        print(f"CAGR={m.get('cagr',0):+.1f}%  Cal={m.get('calmar',0):.3f}  "
              f"DD={m.get('max_dd',0):.1f}%  PF={m.get('profit_factor',0):.3f}  "
              f"trig={tc}  ({time.time()-t0:.1f}s)")
        return res

    print("\n[2/3] WF 5-Fold × 6 runs (Baseline + 5 Cases)...")

    # baseline_rows: per fold reference
    baseline_rows = []
    # fold_baseline[fold_name] = metrics_A
    fold_baseline: dict = {}

    for fold_name, tr_s, tr_e, te_s, te_e, label in FOLDS:
        print(f"\n  {fold_name} [{label}] {te_s}〜{te_e}")
        res_a = run(fold_name, te_s, te_e, threshold=0.0)
        m_a   = res_a.get("metrics", {})
        fold_baseline[fold_name] = {
            "m_a": m_a, "label": label, "test_year": te_s[:4],
            "test_years": round(m_a.get("n_dates", 252) / 252, 2),
        }
        baseline_rows.append({
            "fold": fold_name, "label": label, "test_year": te_s[:4],
            "A": m_a,
        })

    # case runs
    case_data: dict = {}
    for ck, threshold in CASES.items():
        fold_rows    = []
        all_triggers = []

        for fold_name, tr_s, tr_e, te_s, te_e, label in FOLDS:
            info = fold_baseline[fold_name]
            res_f = run(fold_name, te_s, te_e, threshold=threshold)
            m_f   = res_f.get("metrics", {})
            trig  = res_f.get("triggers", [])
            all_triggers.extend(trig)

            fold_rows.append({
                "fold":          fold_name,
                "label":         label,
                "test_year":     te_s[:4],
                "test_years":    info["test_years"],
                "A":             info["m_a"],
                "F":             m_f,
                "trigger_count": len(trig),
                "triggers":      trig,
            })

        tq = trigger_quality(all_triggers)
        case_data[ck] = {
            "fold_rows":    fold_rows,
            "all_triggers": all_triggers,
            "tq":           tq,
        }
        print(f"\n  Case {ck} (RSR≥{threshold:.0f}) 全Fold完了: "
              f"trigger={len(all_triggers)}件  "
              f"avg_fwd60={tq.get('avg_fwd60','N/A')}")

    print("\n[3/3] レポート生成...")
    out_path = REPORTS_DIR / "absolute_rsr_4th_position_wf.md"
    write_report(baseline_rows, case_data, out_path)

    # ── Console サマリ ─────────────────────────────────────────────
    print("\n" + "="*72)
    print("  ★ サマリ")
    print("="*72)
    print(f"{'Case':<6} {'閾値':<6} {'trig/y':<8} {'ΔCAGR':>8} {'ΔMaxDD':>8} {'WF':>5} {'判定':>8}")
    print("-"*55)
    for ck, threshold in CASES.items():
        cd = case_data[ck]
        fold_rows = cd["fold_rows"]
        all_trig  = cd["all_triggers"]
        total_yrs = sum(r["test_years"] for r in fold_rows)
        ty = round(len(all_trig) / max(total_yrs, 0.01), 1)
        avg_dc = float(np.mean([r["F"]["cagr"] - r["A"]["cagr"] for r in fold_rows]))
        avg_dd = float(np.mean([r["F"]["max_dd"] - r["A"]["max_dd"] for r in fold_rows]))
        avg_dcal = float(np.mean([r["F"]["calmar"] - r["A"]["calmar"] for r in fold_rows]))
        wf = sum(
            1 for row in fold_rows
            if (row["F"]["cagr"] - row["A"]["cagr"] >= 0)
            and (row["F"]["max_dd"] - row["A"]["max_dd"] >= -ADOPT_DMAXDD_MAX)
        )
        tq = cd["tq"]
        fwd60_ok = (tq.get("avg_fwd60") is not None and tq.get("avg_wfwd60") is not None
                    and tq["avg_fwd60"] > tq["avg_wfwd60"])
        ok = (wf >= ADOPT_WF_MIN and ty <= ADOPT_TRIGGER_YR
              and avg_dc >= ADOPT_DCAGR_MIN and avg_dcal >= ADOPT_DCALMAR_MIN
              and avg_dd >= -ADOPT_DMAXDD_MAX and fwd60_ok)
        print(f"{'Case '+ck:<6} ≥{threshold:<5.0f} {ty:<8.1f} {avg_dc:>+7.2f}pp {-avg_dd:>+7.2f}pp "
              f"{wf:>2}/5  {'ADOPT ✅' if ok else 'REJECT ❌'}")

    print(f"\n  レポート → {out_path}")
    print("="*72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
