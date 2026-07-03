"""
backtest/entry_timing_predictive_power_audit.py

Entry Timing Predictive Power Audit
ET score が将来リターンを予測しているかを検証する。

Analysis 1: Score distribution
Analysis 2: Decile analysis (fwd20d/40d/60d)
Analysis 3: Predictive power (Spearman / Pearson correlation)
Analysis 4: Winner vs Loser score comparison
Analysis 5: Threshold sweep (IS CAGR/Sharpe/MaxDD/Calmar)

Output: reports/entry_timing_predictive_power_audit.md

Run:
    cd C:/ai-trading
    python src/backtest/entry_timing_predictive_power_audit.py
"""

from __future__ import annotations

import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, run_period, _compute_et_bt, _ET_AVAILABLE,
    IS_START, IS_END, OOS_START, OOS_END,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy

REPORTS_DIR = Path("reports")
FULL_START  = IS_START
FULL_END    = OOS_END


# ─────────────────────────────────────────────────────────────────────
#  DATA COLLECTION: run Pattern A, record ET scores for ALL buy_cands
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ETRecord:
    date:       str
    symbol:     str
    rsr:        float
    sector:     str
    et_score:   Optional[float]
    confidence: Optional[str]
    executed:   bool                  # was this signal actually traded?
    period:     str                   # "IS" | "OOS"
    fwd20d:     Optional[float] = None
    fwd40d:     Optional[float] = None
    fwd60d:     Optional[float] = None


def collect_et_records(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    start: str, end: str,
    period_label: str,
) -> list[ETRecord]:
    """
    Run Pattern A for the given period and collect ET scores for ALL buy_cands.
    Does NOT filter by ET — only collects scores for analysis.
    Returns list of ETRecord (one per buy candidate per signal day).
    """
    from src.backtest.capital_allocation_abc import (
        Position, _take, _sector_ok, _execute_buy,
    )

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
    gross_normal  = float(getattr(rc, "gross_cap_normal", 1.0))       if rc else 1.0
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
        return []
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
    reentry_ban: dict[str, int] = {}
    peak_equity  = float(capital)
    cb_active    = False; cb_days = 0

    records: list[ETRecord] = []

    for i, date in enumerate(common_dates):
        invested   = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
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
        mkt_shock = float(mkt_ret_arr[i]) <= -0.05

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
                    if prev_c > 0 and (close_t / prev_c - 1.0) <= -0.08:
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
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
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
            cash    += proceeds
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if cb_active or not buy_cands:
            continue

        buy_cands.sort(key=lambda x: -x[0])

        # ── Compute ET scores for ALL buy candidates (no filtering) ───
        et_scores: dict = {}
        if _ET_AVAILABLE:
            et_scores = _compute_et_bt(
                buy_cands, date, universe_raw, rsr_df,
                rsr_mat, i, sym_to_i, active_syms,
            )

        # ── Execute Pattern A (no ET filter) — track which are executed ─
        executed_syms: set[str] = set()
        for rsr_val, sym in buy_cands:
            open_slots = max_pos - len(positions)
            if open_slots <= 0:
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
                    continue
            n_rem     = sum(1 for _, s in buy_cands if s not in positions)
            eff_slots = min(open_slots, max(1, n_rem))
            alloc     = (cash / eff_slots)
            alloc     = min(alloc, capital * MAX_SINGLE_W)
            qty       = int(alloc / buy_px / LOT) * LOT
            if qty <= 0:
                continue
            _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, [], positions, rsr_val)
            cash -= qty * buy_px * (1 + COST_ONE_WAY)
            executed_syms.add(sym)

        # ── Record all buy candidates with their ET scores ─────────────
        for rsr_val, sym in buy_cands:
            et_r = et_scores.get(sym)
            records.append(ETRecord(
                date       = str(date.date()),
                symbol     = sym,
                rsr        = rsr_val,
                sector     = trade_syms.get(sym, "不明"),
                et_score   = et_r.score      if et_r else None,
                confidence = et_r.confidence if et_r else None,
                executed   = sym in executed_syms,
                period     = period_label,
            ))

    return records


# ─────────────────────────────────────────────────────────────────────
#  FORWARD RETURN COMPUTATION
# ─────────────────────────────────────────────────────────────────────

def add_forward_returns(records: list[ETRecord], universe_raw: dict) -> None:
    """Add fwd20d/40d/60d in-place. Signal date close = baseline."""
    for rec in records:
        sym = rec.symbol
        if sym not in universe_raw:
            continue
        close_s = universe_raw[sym]["df"]["Close"]
        try:
            sig_date = pd.Timestamp(rec.date)
            if sig_date not in close_s.index:
                continue
            base = float(close_s[sig_date])
            if base <= 0:
                continue
            loc  = close_s.index.get_loc(sig_date)
            for n_days, attr in [(20, "fwd20d"), (40, "fwd40d"), (60, "fwd60d")]:
                fl = loc + n_days
                if fl < len(close_s):
                    setattr(rec, attr, round((float(close_s.iloc[fl]) / base - 1) * 100, 3))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
#  ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────

def _fwd_stats(vals: list) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "win_rate": None, "ir": None, "pf": None}
    arr  = np.array(vals, dtype=float)
    wins = arr[arr > 0]; loss = arr[arr <= 0]
    gp   = float(wins.sum()) if len(wins) else 0.0
    gl   = abs(float(loss.sum())) if len(loss) else 0.0
    return {
        "n":       len(vals),
        "mean":    round(float(arr.mean()), 3),
        "median":  round(float(np.median(arr)), 3),
        "std":     round(float(arr.std()),  3),
        "win_rate":round(len(wins) / len(arr) * 100, 1),
        "ir":      round(float(arr.mean() / arr.std()), 3) if arr.std() > 0 else 0.0,
        "pf":      round(gp / max(gl, 0.001), 3),
    }


# ─────────────────────────────────────────────────────────────────────
#  ANALYSES
# ─────────────────────────────────────────────────────────────────────

def analysis1_distribution(recs: list[ETRecord]) -> dict:
    scored = [r for r in recs if r.et_score is not None]
    scores = [r.et_score for r in scored]
    if not scores:
        return {}
    arr = np.array(scores)
    bins = list(range(0, 101, 10))
    hist = {}
    for lo in range(0, 100, 10):
        hi  = lo + 10
        cnt = int(((arr >= lo) & (arr < hi)).sum())
        hist[f"{lo}-{hi}"] = cnt
    by_conf = {}
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        grp = [r.et_score for r in scored if r.confidence == conf]
        by_conf[conf] = _fwd_stats(grp) if grp else {"n": 0}
    return {
        "n_total":   len(recs),
        "n_scored":  len(scored),
        "pct_scored":round(len(scored)/max(len(recs),1)*100,1),
        "mean":      round(float(arr.mean()), 2),
        "median":    round(float(np.median(arr)), 2),
        "std":       round(float(arr.std()), 2),
        "min":       round(float(arr.min()), 2),
        "max":       round(float(arr.max()), 2),
        "pct_high":  round((arr >= 75).sum() / len(arr) * 100, 1),
        "pct_med":   round(((arr >= 50) & (arr < 75)).sum() / len(arr) * 100, 1),
        "pct_low":   round((arr < 50).sum() / len(arr) * 100, 1),
        "histogram": hist,
        "by_confidence": by_conf,
    }


def analysis2_decile(recs: list[ETRecord]) -> dict:
    scored = [r for r in recs if r.et_score is not None and r.fwd60d is not None]
    if len(scored) < 10:
        return {}
    scores = np.array([r.et_score for r in scored])
    decile_bounds = np.percentile(scores, list(range(0, 110, 10)))
    deciles = {}
    for d in range(10):
        lo = decile_bounds[d]
        hi = decile_bounds[d + 1]
        grp = [r for r in scored
               if (lo <= r.et_score <= hi if d == 9
                   else lo <= r.et_score < hi)]
        label = f"D{d+1} ({lo:.0f}-{hi:.0f})"
        deciles[label] = {
            "score_range": (round(lo, 1), round(hi, 1)),
            "n":       len(grp),
            "fwd20d":  _fwd_stats([r.fwd20d for r in grp if r.fwd20d is not None]),
            "fwd40d":  _fwd_stats([r.fwd40d for r in grp if r.fwd40d is not None]),
            "fwd60d":  _fwd_stats([r.fwd60d for r in grp if r.fwd60d is not None]),
        }
    return deciles


def analysis3_correlation(recs: list[ETRecord]) -> dict:
    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    out = {}
    for fwd_attr, label in [("fwd20d","fwd20d"),("fwd40d","fwd40d"),("fwd60d","fwd60d")]:
        scored = [(r.et_score, getattr(r, fwd_attr))
                  for r in recs
                  if r.et_score is not None and getattr(r, fwd_attr) is not None]
        if len(scored) < 10:
            out[label] = {"n": len(scored)}
            continue
        x = np.array([s for s,_ in scored])
        y = np.array([v for _,v in scored])
        pearson  = round(float(np.corrcoef(x, y)[0, 1]), 4)
        if HAS_SCIPY:
            sp, sp_p = scipy_stats.spearmanr(x, y)
            spearman = round(float(sp),   4)
            spear_p  = round(float(sp_p), 4)
            pr, pr_p = scipy_stats.pearsonr(x, y)
            pearson  = round(float(pr),   4)
            pearson_p= round(float(pr_p), 4)
        else:
            spearman = spear_p = pearson_p = None
        out[label] = {
            "n":        len(scored),
            "pearson":  pearson,
            "pearson_p":pearson_p,
            "spearman": spearman,
            "spearman_p": spear_p,
        }
    return out


def analysis4_winner_loser(recs: list[ETRecord]) -> dict:
    scored = [r for r in recs if r.et_score is not None and r.fwd60d is not None]
    winners = [r for r in scored if r.fwd60d >  0]
    losers  = [r for r in scored if r.fwd60d <= 0]
    def _grp_stats(grp):
        if not grp:
            return {}
        sc = [r.et_score for r in grp]
        conf_cnt = {c: sum(1 for r in grp if r.confidence == c)
                    for c in ["HIGH","MEDIUM","LOW"]}
        return {
            "n":            len(grp),
            "score_mean":   round(float(np.mean(sc)), 2),
            "score_median": round(float(np.median(sc)), 2),
            "score_std":    round(float(np.std(sc)), 2),
            "pct_high":     round(conf_cnt.get("HIGH",0)/max(len(grp),1)*100,1),
            "pct_med":      round(conf_cnt.get("MEDIUM",0)/max(len(grp),1)*100,1),
            "pct_low":      round(conf_cnt.get("LOW",0)/max(len(grp),1)*100,1),
            "fwd60d_mean":  round(float(np.mean([r.fwd60d for r in grp])), 2),
        }
    return {
        "winners": _grp_stats(winners),
        "losers":  _grp_stats(losers),
        "all":     _grp_stats(scored),
    }


def analysis5_threshold_sweep(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
) -> dict:
    thresholds = [0, 10, 20, 30, 40, 50, 60]
    results = {}
    for thr in thresholds:
        label = f"score≥{thr}" if thr > 0 else "no_filter(A)"
        print(f"    [{label}]", end=" ", flush=True)
        t0 = time.time()
        m = run_period(
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
            start=IS_START, end=IS_END,
            pattern="A",
            et_enabled=(thr > 0),
            et_min_score=float(thr),
        )
        results[thr] = m
        print(f"CAGR={m.get('cagr',0):+.1f}%  Sharpe={m.get('sharpe',0):.3f}  "
              f"Calmar={m.get('calmar',0):.3f}  MaxDD={m.get('max_dd',0):.1f}%  "
              f"Missed={m.get('missed_signals',0)}  ({time.time()-t0:.1f}s)")
    return results


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(a1, a2, a3, a4, a5, records, output_path: Path) -> None:
    L = []; w = L.append

    w("# Entry Timing Predictive Power Audit")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  対象: {FULL_START}〜{FULL_END}  |  Pattern A 全BUYシグナル")
    w(f"\n**注意: バックテストではSEPA=0 / trend_cluster=0 で近似**\n")

    # ── Analysis 1 ─────────────────────────────────────────────────────
    w("---\n## 分析1: ET Score Distribution\n")
    w(f"- 全BUYシグナル: {a1.get('n_total',0)}件")
    w(f"- ETスコア取得成功: {a1.get('n_scored',0)}件 ({a1.get('pct_scored',0):.1f}%)")
    w(f"- Score  mean={a1.get('mean',0):.2f}  median={a1.get('median',0):.2f}  "
      f"std={a1.get('std',0):.2f}  range=[{a1.get('min',0):.1f}, {a1.get('max',0):.1f}]")
    w(f"- HIGH(≥75): {a1.get('pct_high',0):.1f}%  "
      f"MEDIUM(50-75): {a1.get('pct_med',0):.1f}%  "
      f"LOW(<50): {a1.get('pct_low',0):.1f}%\n")

    w("### ヒストグラム\n")
    w("| Score range | Count | % |")
    w("|---|---|---|")
    total_scored = a1.get("n_scored", 1)
    for rng, cnt in a1.get("histogram", {}).items():
        pct = cnt / total_scored * 100
        w(f"| {rng} | {cnt} | {pct:.1f}% |")

    w("\n### Confidence別統計\n")
    w("| Confidence | N | score mean | score median | score std |")
    w("|---|---|---|---|---|")
    for conf in ["HIGH","MEDIUM","LOW"]:
        d = a1.get("by_confidence",{}).get(conf,{})
        w(f"| {conf} | {d.get('n',0)} | {d.get('mean','—')} | {d.get('median','—')} | {d.get('std','—')} |")

    # ── Analysis 2 ─────────────────────────────────────────────────────
    w("\n---\n## 分析2: Decile Analysis\n")
    w("| Decile | Score Range | N | fwd20d mean | fwd20d WR | fwd40d mean | fwd40d WR | fwd60d mean | fwd60d WR | fwd60d IR |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for dlabel, ddata in a2.items():
        lo, hi = ddata.get("score_range", (0, 0))
        n    = ddata.get("n", 0)
        s20  = ddata.get("fwd20d", {})
        s40  = ddata.get("fwd40d", {})
        s60  = ddata.get("fwd60d", {})
        w(f"| {dlabel} | {lo:.0f}-{hi:.0f} | {n} "
          f"| {s20.get('mean','—'):+}% | {s20.get('win_rate','—')}% "
          f"| {s40.get('mean','—'):+}% | {s40.get('win_rate','—')}% "
          f"| {s60.get('mean','—'):+}% | {s60.get('win_rate','—')}% "
          f"| {s60.get('ir','—')} |")

    # ── Analysis 3 ─────────────────────────────────────────────────────
    w("\n---\n## 分析3: Predictive Power (相関分析)\n")
    w("| Forward Period | N | Pearson r | Pearson p | Spearman ρ | Spearman p |")
    w("|---|---|---|---|---|---|")
    for fwd_k in ["fwd20d","fwd40d","fwd60d"]:
        d = a3.get(fwd_k, {})
        w(f"| {fwd_k} | {d.get('n','—')} "
          f"| {d.get('pearson','—')} | {d.get('pearson_p','—')} "
          f"| {d.get('spearman','—')} | {d.get('spearman_p','—')} |")

    w("\n**解釈ガイド**: |r| ≥ 0.10 = 弱い相関  |r| ≥ 0.20 = 中程度  |r| ≥ 0.30 = 実用的  p < 0.05 = 有意\n")

    # ── Analysis 4 ─────────────────────────────────────────────────────
    w("\n---\n## 分析4: Winner vs Loser Score比較\n")
    w("| グループ | N | score mean | score median | HIGH% | MED% | LOW% | fwd60d mean |")
    w("|---|---|---|---|---|---|---|---|")
    for gname, gkey in [("Winners (fwd60d>0)","winners"),("Losers (fwd60d≤0)","losers"),("全体","all")]:
        d = a4.get(gkey, {})
        if not d:
            continue
        w(f"| {gname} | {d.get('n',0)} "
          f"| {d.get('score_mean','—')} | {d.get('score_median','—')} "
          f"| {d.get('pct_high','—')}% | {d.get('pct_med','—')}% | {d.get('pct_low','—')}% "
          f"| {d.get('fwd60d_mean','—'):+}% |")

    # ── Analysis 5 ─────────────────────────────────────────────────────
    w("\n---\n## 分析5: Threshold Sweep (IS 2018-2024)\n")
    w("| Threshold | IS CAGR | IS Sharpe | IS MaxDD | IS Calmar | IS Exposure | IS Missed | ΔCAGR vs A |")
    w("|---|---|---|---|---|---|---|---|")
    a_base = a5.get(0, {})
    for thr in sorted(a5.keys()):
        m = a5[thr]
        label = f"score≥{thr}" if thr > 0 else "no_filter (**A**)"
        dc = m.get("cagr",0) - a_base.get("cagr",0)
        w(f"| {label} "
          f"| {m.get('cagr',0):+.1f}% "
          f"| {m.get('sharpe',0):.3f} "
          f"| {m.get('max_dd',0):.1f}% "
          f"| {m.get('calmar',0):.3f} "
          f"| {m.get('avg_exposure',0):.1f}% "
          f"| {m.get('missed_signals',0)} "
          f"| {dc:+.1f}pp |")

    # ── 最終結論 ────────────────────────────────────────────────────────
    w("\n---\n## 最終結論\n")

    # Determine verdict
    corr_60 = a3.get("fwd60d", {})
    sp_rho   = corr_60.get("spearman", 0) or 0
    sp_p     = corr_60.get("spearman_p", 1.0) or 1.0
    d_w      = a4.get("winners",{}).get("score_mean", 50) or 50
    d_l      = a4.get("losers", {}).get("score_mean", 50) or 50
    score_gap = d_w - d_l

    # Find best threshold
    best_calmar_thr = max(a5.keys(), key=lambda k: a5[k].get("calmar", -999))
    best_cagr_thr   = max(a5.keys(), key=lambda k: a5[k].get("cagr", -999))
    a_calmar = a_base.get("calmar", 0)
    best_calmar_val = a5.get(best_calmar_thr, {}).get("calmar", 0)

    if abs(sp_rho) >= 0.20 and sp_p < 0.05:
        if best_calmar_val > a_calmar and best_calmar_thr > 0:
            verdict = "C: 予測力あり、threshold最適化余地あり"
        else:
            verdict = "B: 予測力ありだが弱い"
    elif abs(sp_rho) >= 0.08 and sp_p < 0.10:
        verdict = "B: 予測力ありだが弱い"
    elif best_calmar_val > a_calmar and best_calmar_thr > 0:
        verdict = "D: block_lowの閾値設計だけが失敗"
    else:
        verdict = "A: ET scoreに予測力なし"

    w(f"### 判定: **{verdict}**\n")
    w(f"- Spearman ρ (fwd60d): {sp_rho:+.4f}  p={sp_p:.4f}")
    w(f"- Winner vs Loser score差: {score_gap:+.2f}pt  (Winner={d_w:.1f}, Loser={d_l:.1f})")
    w(f"- Threshold sweep ベスト Calmar: score≥{best_calmar_thr}  ({best_calmar_val:.3f}  vs A={a_calmar:.3f})")

    # Recommendation
    w("\n### 推奨: ranking補助 vs entry filter\n")
    w("| 用途 | 判定 | 根拠 |")
    w("|---|---|---|")

    if abs(sp_rho) >= 0.10:
        ranking_verdict = "✅ 有効 (スコアでランク順位に意味あり)"
    else:
        ranking_verdict = "⚠️ 限定的 (スコア差が小さい)"

    if best_calmar_val > a_calmar * 1.05 and best_calmar_thr > 0:
        filter_verdict = f"✅ 有効 (score≥{best_calmar_thr} で Calmar改善)"
    elif best_calmar_val > a_calmar:
        filter_verdict = f"⚠️ 要検証 (score≥{best_calmar_thr} で僅かに改善, WF必要)"
    else:
        filter_verdict = "❌ 無効 (どの閾値でもA以下)"

    w(f"| ranking補助 (boost_weight調整) | {ranking_verdict} | Spearman ρ={sp_rho:+.4f} |")
    w(f"| entry filter (threshold) | {filter_verdict} | Calmar {a_calmar:.3f}→{best_calmar_val:.3f} at thr={best_calmar_thr} |")

    w("\n### 次のアクション\n")
    if "C" in verdict or "B" in verdict:
        w(f"1. **score≥{best_calmar_thr} でIS/OOS Walk-Forward検証** (5-Fold)")
        w(f"2. **SEPA・trend_clusterを正しく計算**してスコア精度向上")
        w(f"3. **boost_weight を0.15〜0.20に引上げ** (現行0.06は効果なし)")
    elif "D" in verdict:
        w(f"1. **block_low の替わりに score≥{best_calmar_thr} で再テスト**")
        w(f"2. **LOW判定の見直し** — 閾値score<50は広すぎる可能性")
    else:
        w("1. **SEPA・trend_clusterを正しく計算した上で再検証** (現在の近似が精度を落とした可能性)")
        w("2. **ET scoreの改善** (追加ファクター、ウェイト見直し)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート保存: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if not _ET_AVAILABLE:
        print("ERROR: Entry Timing engine not available. Check src/entry/entry_timing_engine.py")
        return 1

    cfg = load_strategy_config()

    print("=" * 68)
    print("  Entry Timing Predictive Power Audit")
    print(f"  Period: {FULL_START}〜{FULL_END}  (IS+OOS)")
    print("=" * 68)

    print("\n[1/5] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/5] ETスコア収集 (IS + OOS, 全BUYシグナル)...")
    records: list[ETRecord] = []
    for period_label, start, end in [
        ("IS",  IS_START,  IS_END),
        ("OOS", OOS_START, OOS_END),
    ]:
        print(f"  {period_label} {start}〜{end} ...", end=" ", flush=True)
        t0 = time.time()
        recs = collect_et_records(
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
            start=start, end=end, period_label=period_label,
        )
        records.extend(recs)
        print(f"{len(recs)}件  ({time.time()-t0:.1f}s)")

    print(f"  総計: {len(records)}件  (スコア取得済み: {sum(1 for r in records if r.et_score is not None)}件)")

    print("\n[3/5] Forward return計算中...")
    add_forward_returns(records, universe_raw)
    scored = [r for r in records if r.et_score is not None and r.fwd60d is not None]
    print(f"  fwd60d あり: {len(scored)}件")

    print("\n[4/5] 分析1-4 実行中...")
    a1 = analysis1_distribution(records)
    a2 = analysis2_decile(scored)
    a3 = analysis3_correlation(records)
    a4 = analysis4_winner_loser(records)

    print("\n[5/5] Threshold Sweep (Analysis 5)...")
    a5 = analysis5_threshold_sweep(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
    )

    out_path = REPORTS_DIR / "entry_timing_predictive_power_audit.md"
    write_report(a1, a2, a3, a4, a5, records, out_path)

    # ── Console summary ────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  ★ 結果サマリー")
    print("="*68)

    print(f"\n  Score分布: mean={a1.get('mean',0):.1f}  median={a1.get('median',0):.1f}  "
          f"std={a1.get('std',0):.1f}")
    print(f"  HIGH={a1.get('pct_high',0):.1f}%  MED={a1.get('pct_med',0):.1f}%  LOW={a1.get('pct_low',0):.1f}%")

    print(f"\n  相関 (fwd60d): Spearman={a3.get('fwd60d',{}).get('spearman','?')}  "
          f"p={a3.get('fwd60d',{}).get('spearman_p','?')}  "
          f"Pearson={a3.get('fwd60d',{}).get('pearson','?')}")

    print(f"\n  Winner vs Loser score:")
    print(f"    Winners (fwd60d>0): score_mean={a4.get('winners',{}).get('score_mean','?')}")
    print(f"    Losers  (fwd60d≤0): score_mean={a4.get('losers',{}).get('score_mean','?')}")

    print(f"\n  Threshold Sweep:")
    a_base = a5.get(0, {})
    for thr in sorted(a5.keys()):
        m = a5[thr]
        dc = m.get("cagr",0) - a_base.get("cagr",0)
        marker = " ← best Calmar" if m.get("calmar",0) == max(a5[k].get("calmar",0) for k in a5) else ""
        print(f"    score≥{thr:2d}: CAGR={m.get('cagr',0):+.1f}%  Calmar={m.get('calmar',0):.3f}  "
              f"MaxDD={m.get('max_dd',0):.1f}%  Missed={m.get('missed_signals',0)}  {marker}")

    print(f"\n  レポート → {out_path}")
    print("="*68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
