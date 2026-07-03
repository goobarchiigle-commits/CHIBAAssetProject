"""
backtest/ranking_quality_audit.py

研究専用 / 実装変更禁止

RSR単独順位 vs 複合スコア順位 の質改善測定。
除外なし・position追加なし・exit変更なし。
「Top3候補の並び替えのみ」でCAGR改善できるか。

Phase1: 特徴量収集（全買いシグナル 2018-2025）
Phase2: 単変量 rank quality
Phase3: 複合スコア品質（ISデータのみ / WFなし研究）
Phase4: WF Replay Test（5-fold OOS）

スコア定義:
  S1 = rsr                                (ベースライン)
  S2 = rsr + z(slope20)
  S3 = rsr - 0.3*z(days_cross70)
  S4 = rsr + 0.4*z(slope5) - 0.2*z(days_cross70)
  S5 = rsr + 0.5*z(state_score)

採用条件: WF≥4/5 AND ΔCAGR≥+1pp AND ΔDD≤+1pp AND swap≤15%

Run:
    cd C:/ai-trading
    python src/backtest/ranking_quality_audit.py
"""

from __future__ import annotations

import sys, time, csv, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, _sector_ok, _execute_buy, Position, _take, calc_metrics,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"
STALL_THR   = 0.15
MIN_RSR_COL = 70.0   # collect signals from RSR>=70 (slightly below strategy threshold)

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31", "is_end": "2020-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31", "is_end": "2021-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31", "is_end": "2022-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31", "is_end": "2023-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31", "is_end": "2024-12-31"},
]

SCORES = ["S1", "S2", "S3", "S4", "S5"]
SCORE_LABELS = {
    "S1": "RSR only (baseline)",
    "S2": "RSR + z(slope20)",
    "S3": "RSR - 0.3*z(d70)",
    "S4": "RSR + 0.4*z(slope5) - 0.2*z(d70)",
    "S5": "RSR + 0.5*z(state_score)",
}

STATE_SCORE = {
    "EARLY_UP":   1.5,
    "STEADY_UP":  1.0,
    "FLAT":       0.75,
    "STALL":      0.5,
    "DOWN":       0.0,
    "EARLY_ROLL": -0.5,
    "UNKNOWN":    0.0,
}

ADOPT_WF_MIN   = 4
ADOPT_DCAGR    = 1.0   # pp
ADOPT_DMAXDD   = 1.0   # pp (stricter than Study 5)
ADOPT_SWAP_MAX = 15.0  # %


# ─────────────────────────────────────────────────────────────────────
#  FEATURE HELPERS
# ─────────────────────────────────────────────────────────────────────

def classify_entry_state(s5: float, s20: float) -> str:
    if np.isnan(s5) or np.isnan(s20):
        return "UNKNOWN"
    if abs(s5) < STALL_THR:
        return "STALL"
    if s20 > 0:
        if s5 > s20:  return "EARLY_UP"
        if s5 > 0:    return "STEADY_UP"
        return "EARLY_ROLL"
    return "DOWN"


def compute_cross_mat(rsr_mat: np.ndarray, threshold: float) -> np.ndarray:
    """Consecutive days RSR >= threshold (value on day i = count up-to day i-1)."""
    n_dates, n_syms = rsr_mat.shape
    out     = np.zeros((n_dates, n_syms), dtype=np.int32)
    running = np.zeros(n_syms, dtype=np.int32)
    for i in range(n_dates):
        out[i] = running
        above   = rsr_mat[i] >= threshold
        running[above] += 1
        running[~above] = 0
    return out


def compute_slope_mats(rsr_mat: np.ndarray):
    s5  = np.zeros_like(rsr_mat)
    s20 = np.zeros_like(rsr_mat)
    s5[5:]   = (rsr_mat[5:]   - rsr_mat[:-5])  / 5.0
    s20[20:] = (rsr_mat[20:]  - rsr_mat[:-20]) / 20.0
    return s5, s20


def compute_atr20_mat(high_mat, low_mat, close_mat, lookback=20) -> np.ndarray:
    n_dates, n_syms = high_mat.shape
    tr = np.zeros_like(high_mat)
    tr[1:] = np.maximum(
        np.maximum(high_mat[1:] - low_mat[1:],
                   np.abs(high_mat[1:] - close_mat[:-1])),
        np.abs(low_mat[1:] - close_mat[:-1])
    )
    cs = np.zeros((n_dates + 1, n_syms), dtype=np.float64)
    cs[1:] = np.cumsum(tr, axis=0)
    atr_raw = np.zeros_like(high_mat, dtype=np.float64)
    atr_raw[lookback:] = (cs[lookback + 1:] - cs[1: n_dates - lookback + 1]) / lookback
    safe_c = np.where(close_mat > 0, close_mat, np.nan)
    return np.where(close_mat > 0, atr_raw / safe_c * 100.0, 0.0).astype(np.float32)


def compute_rsr_peak_dist(rsr_mat: np.ndarray, lookback: int = 20) -> np.ndarray:
    """peak_dist[i,si] = rsr[i,si] - rolling_max(rsr, lookback)[i,si]  (<=0)."""
    rsr_df = pd.DataFrame(rsr_mat)
    peak   = rsr_df.rolling(lookback, min_periods=1).max().to_numpy(dtype=np.float32)
    return (rsr_mat - peak).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
#  SIGNAL COLLECTION (Phase 1)
# ─────────────────────────────────────────────────────────────────────

def collect_signals(
    close_mat, sig_mat, sig_ready, rsr_mat,
    cross70_mat, cross80_mat, slope5_mat, slope20_mat,
    atr20_mat, peak_dist_mat,
    active_syms, sym_to_i, common_dates,
    min_rsr: float = MIN_RSR_COL,
    fwd_days: int = 60,
) -> pd.DataFrame:
    n_dates  = len(common_dates)
    n_syms   = len(active_syms)
    records  = []

    for i in range(n_dates - fwd_days):
        date_str = str(common_dates[i].date())
        day_sigs = []

        for si in range(n_syms):
            if not sig_ready[si]:
                continue
            if int(sig_mat[i, si]) != 1:
                continue
            rsr_v = float(rsr_mat[i, si])
            if rsr_v < min_rsr:
                continue

            s5  = float(slope5_mat[i, si])
            s20 = float(slope20_mat[i, si])
            d70 = int(cross70_mat[i, si])
            d80 = int(cross80_mat[i, si])
            atr = float(atr20_mat[i, si])
            pk  = float(peak_dist_mat[i, si])
            st  = classify_entry_state(s5, s20)
            ss  = STATE_SCORE.get(st, 0.0)

            c0  = float(close_mat[i, si])
            c60 = float(close_mat[i + fwd_days, si])
            f60 = (c60 / c0 - 1) * 100 if c0 > 0 and c60 > 0 else np.nan

            day_sigs.append({
                "date":        date_str,
                "symbol":      active_syms[si],
                "si":          si,
                "date_idx":    i,
                "rsr":         round(rsr_v, 2),
                "slope5":      round(s5, 3),
                "slope20":     round(s20, 3),
                "days_cross70":d70,
                "days_cross80":d80,
                "atr20_pct":   round(atr, 2),
                "peak_dist":   round(pk, 2),
                "state":       st,
                "state_score": ss,
                "fwd60":       round(f60, 2) if not np.isnan(f60) else np.nan,
            })

        if not day_sigs:
            continue

        # rsr_rank within same day (1 = highest RSR)
        day_sigs_sorted = sorted(day_sigs, key=lambda x: -x["rsr"])
        for rank, rec in enumerate(day_sigs_sorted, start=1):
            rec["rsr_rank"] = rank
        records.extend(day_sigs)

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────────────────────────────
#  PHASE 2: Single-feature rank quality
# ─────────────────────────────────────────────────────────────────────

def phase2_rank_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["fwd60"])
    features = ["rsr", "slope5", "slope20", "days_cross70", "days_cross80",
                "atr20_pct", "peak_dist", "state_score", "rsr_rank"]

    rows = []
    for feat in features:
        col = df[feat].to_numpy(dtype=float)
        tgt = df["fwd60"].to_numpy(dtype=float)
        mask = ~(np.isnan(col) | np.isnan(tgt))
        col, tgt = col[mask], tgt[mask]
        if len(col) < 10:
            continue

        rho, p_val  = sp_stats.spearmanr(col, tgt)
        n           = len(col)
        q10         = np.percentile(col, 10)
        q90         = np.percentile(col, 90)
        top_mask    = col >= q90
        bot_mask    = col <= q10
        top_avg     = float(np.mean(tgt[top_mask])) if top_mask.sum() > 0 else np.nan
        bot_avg     = float(np.mean(tgt[bot_mask])) if bot_mask.sum() > 0 else np.nan
        overall_avg = float(np.mean(tgt))
        lift        = top_avg / abs(overall_avg) if overall_avg != 0 else np.nan

        rows.append({
            "feature":      feat,
            "n":            n,
            "spearman_rho": round(rho, 4),
            "p_value":      round(p_val, 4),
            "top_decile_avg": round(top_avg, 2) if not np.isnan(top_avg) else None,
            "bot_decile_avg": round(bot_avg, 2) if not np.isnan(bot_avg) else None,
            "overall_avg":  round(overall_avg, 2),
            "lift":         round(lift, 3) if not np.isnan(lift) else None,
        })

    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)


# ─────────────────────────────────────────────────────────────────────
#  PHASE 3: Composite score quality (all data, research)
# ─────────────────────────────────────────────────────────────────────

def _z_norm(vals: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(vals)
    if mask.sum() < 2:
        return np.zeros_like(vals)
    m, s = np.mean(vals[mask]), np.std(vals[mask])
    return np.where(mask, (vals - m) / max(s, 1e-6), 0.0)


def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add S1-S5 columns to df using global z-norms (research only)."""
    z_s20 = _z_norm(df["slope20"].to_numpy(float))
    z_s5  = _z_norm(df["slope5"].to_numpy(float))
    z_d70 = _z_norm(df["days_cross70"].to_numpy(float))
    z_ss  = _z_norm(df["state_score"].to_numpy(float))

    df = df.copy()
    df["S1"] = df["rsr"]
    df["S2"] = df["rsr"] + z_s20
    df["S3"] = df["rsr"] - 0.3 * z_d70
    df["S4"] = df["rsr"] + 0.4 * z_s5 - 0.2 * z_d70
    df["S5"] = df["rsr"] + 0.5 * z_ss
    return df


def phase3_score_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_all_scores(df)
    df_valid = df.dropna(subset=["fwd60"]).copy()

    rows = []
    for s_id in SCORES:
        grp_records = []
        top1_fwds = []; top3_fwds = []; swap_vs_s1 = 0; contested = 0; wins = 0

        for date_str, grp in df_valid.groupby("date"):
            grp = grp.sort_values(s_id, ascending=False).reset_index(drop=True)
            n = len(grp)
            fwds = grp["fwd60"].to_numpy()

            if n == 0 or all(np.isnan(fwds)):
                continue

            top1_fwds.append(fwds[0])

            top3_f = fwds[:3]
            top3_fwds.extend([v for v in top3_f if not np.isnan(v)])

            if n >= 2:
                contested += 1
                # winner = highest fwd60 in this group
                valid_fwds = fwds[~np.isnan(fwds)]
                if len(valid_fwds) >= 2:
                    best_fwd = np.max(valid_fwds)
                    if abs(fwds[0] - best_fwd) < 0.01:
                        wins += 1

                # swap vs S1: top symbol differs?
                top_s1  = grp.sort_values("S1", ascending=False).iloc[0]["symbol"]
                top_sid = grp.iloc[0]["symbol"]
                if top_s1 != top_sid:
                    swap_vs_s1 += 1

            grp_records.append({"date": date_str, "n": n, "top1_fwd": fwds[0]})

        rows.append({
            "score":          s_id,
            "label":          SCORE_LABELS[s_id],
            "top1_avg_fwd60": round(float(np.nanmean(top1_fwds)), 2) if top1_fwds else np.nan,
            "top3_avg_fwd60": round(float(np.nanmean(top3_fwds)), 2) if top3_fwds else np.nan,
            "winner_rate_pct":round(wins / max(1, contested) * 100, 1),
            "swap_rate_pct":  round(swap_vs_s1 / max(1, contested) * 100, 1),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  SCORE MATRIX BUILDER (for WF replay)
# ─────────────────────────────────────────────────────────────────────

def _z_params_from_is(
    feature_mat: np.ndarray, sig_mat: np.ndarray, sig_ready: np.ndarray,
    rsr_mat: np.ndarray, common_dates, is_end_str: str, min_rsr: float = 70.0
):
    """Compute mean/std from IS buy signals only."""
    vals = []
    for i, d in enumerate(common_dates):
        if str(d.date()) > is_end_str:
            break
        for si in range(len(sig_ready)):
            if not sig_ready[si]:
                continue
            if int(sig_mat[i, si]) != 1:
                continue
            if float(rsr_mat[i, si]) < min_rsr:
                continue
            v = float(feature_mat[i, si])
            if not np.isnan(v):
                vals.append(v)
    if len(vals) < 2:
        return {"mean": 0.0, "std": 1.0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)) or 1.0}


def build_score_mat(
    score_id: str, rsr_mat, slope5_mat, slope20_mat,
    cross70_mat, state_score_mat,
    z_params: dict,
    n_dates: int, n_syms: int,
) -> np.ndarray:
    """Build composite score matrix using IS-calibrated z-params."""
    def z(mat, key):
        p = z_params[key]
        return (mat.astype(np.float32) - p["mean"]) / max(p["std"], 1e-6)

    if score_id == "S1":
        return rsr_mat.astype(np.float32)
    elif score_id == "S2":
        return rsr_mat.astype(np.float32) + z(slope20_mat, "slope20")
    elif score_id == "S3":
        return rsr_mat.astype(np.float32) - 0.3 * z(cross70_mat.astype(np.float32), "d70")
    elif score_id == "S4":
        return (rsr_mat.astype(np.float32)
                + 0.4 * z(slope5_mat, "slope5")
                - 0.2 * z(cross70_mat.astype(np.float32), "d70"))
    elif score_id == "S5":
        return rsr_mat.astype(np.float32) + 0.5 * z(state_score_mat, "state_score")
    else:
        return rsr_mat.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
#  PHASE 4: WF REPLAY BACKTEST
# ─────────────────────────────────────────────────────────────────────

def run_fold_with_score(
    open_mat, close_mat, sig_mat, sig_ready, rsr_mat,
    score_mat,           # pre-computed composite score matrix
    active_syms, sym_to_i, trade_syms, cfg,
    common_dates,
    oos_start_str: str, oos_end_str: str,
) -> dict:
    """Run one WF fold using score_mat for candidate ranking (all other logic identical)."""
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)
    MAX_SINGLE_W = float(cfg.portfolio.max_single_weight)
    rc           = getattr(cfg, "risk_controls", None)
    MAX_SECTOR_W = float(rc.sector_cap) if rc else 0.25
    gross_enabled= bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gross_normal = float(getattr(rc, "gross_cap_normal", 1.0)) if rc else 1.0

    n_dates = len(common_dates)
    cash    = float(capital)
    positions: dict[str, Position] = {}
    peak_equity = float(capital)
    cb_active   = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    trades: list[dict] = []

    equity_curve: list[float] = []
    exposure_list:list[float] = []
    date_list:    list        = []

    oos_start_i = None; oos_end_i = None
    n_swap = 0; n_contested = 0   # swap tracking

    for i, date in enumerate(common_dates):
        date_str = str(date.date())

        if date_str >= oos_start_str and oos_start_i is None:
            oos_start_i = i
        if date_str > oos_end_str and oos_end_i is None:
            oos_end_i = i

        invested    = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
                         for s, pos in positions.items())
        cur_equity  = cash + invested
        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        equity_curve.append(cur_equity)
        exposure_list.append(invested / max(1.0, cur_equity))
        date_list.append(date)

        if not cb_active:
            if dd <= -max_dd_limit:
                cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0

        if i + 1 >= n_dates:
            break
        next_i = i + 1

        sell_sigs: list[tuple] = []
        buy_cands: list[tuple] = []

        for sym in active_syms:
            si_sym     = sym_to_i[sym]
            is_holding = sym in positions
            hold_idx   = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_mat[i, si_sym])

            if is_holding:
                if rsr_val < rsr_exit_thr and hold_idx >= min_hold:
                    sell_sigs.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si_sym]) if sig_ready[si_sym] else 0
            if sig == -1 and is_holding and hold_idx >= min_hold:
                sell_sigs.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not is_holding:
                if i < reentry_ban.get(sym, -1):
                    continue
                sc_val = float(score_mat[i, si_sym])
                buy_cands.append((sc_val, rsr_val, sym))

        # Sell
        for sym, reason in sell_sigs:
            if sym not in positions:
                continue
            pos      = positions[sym]
            sell_px  = float(open_mat[next_i, sym_to_i[sym]])
            proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
            pnl      = (sell_px - pos.entry_price) * pos.qty
            cash    += proceeds
            trades.append({
                "side": "SELL", "symbol": sym, "pnl": pnl,
                "entry": pos.entry_price, "exit": sell_px,
                "qty": pos.qty, "entry_idx": pos.entry_idx, "exit_idx": i,
                "date": date_str,
            })
            del positions[sym]

        if not buy_cands or cb_active:
            continue

        # Sort by composite score (descending)
        buy_cands.sort(key=lambda x: -x[0])

        # Swap tracking (OOS only, when 2+ candidates)
        in_oos = (oos_start_i is not None and i >= oos_start_i
                  and (oos_end_i is None or i < oos_end_i))
        if in_oos and len(buy_cands) >= 2:
            n_contested += 1
            top_score_sym = buy_cands[0][2]
            # What would RSR (S1) rank pick?
            rsr_sorted = sorted(buy_cands, key=lambda x: -x[1])
            top_rsr_sym = rsr_sorted[0][2]
            if top_score_sym != top_rsr_sym:
                n_swap += 1

        for sc_val, rsr_val, sym in buy_cands:
            si_sym     = sym_to_i[sym]
            open_slots = max_pos - len(positions)
            if open_slots <= 0:
                break

            buy_px = float(open_mat[next_i, si_sym])
            if buy_px <= 0:
                continue
            if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, MAX_SECTOR_W):
                continue
            if gross_enabled:
                cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in positions.values()) / max(1.0, capital)
                if cur_gross + buy_px * LOT / max(1.0, capital) > gross_normal:
                    continue

            n_rem  = sum(1 for _, _, s in buy_cands if s not in positions)
            alloc  = (cash / max(1, min(open_slots, n_rem)))
            alloc  = min(alloc, capital * MAX_SINGLE_W)
            qty    = int(alloc / buy_px / LOT) * LOT
            if qty <= 0:
                continue

            _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
            cash -= qty * buy_px * (1 + COST_ONE_WAY)
            trades[-1].update({"date": date_str})

    # OOS metrics
    oos_si = oos_start_i or 0
    oos_ei = oos_end_i   or len(equity_curve)
    oos_eq     = equity_curve[oos_si:oos_ei]
    oos_exp    = exposure_list[oos_si:oos_ei]
    oos_dates  = list(common_dates[oos_si:oos_ei])
    oos_trades = [t for t in trades
                  if oos_start_str <= t.get("date", "") <= oos_end_str]

    if not oos_eq:
        return {"cagr": 0, "max_dd": 0, "calmar": 0, "sharpe": 0,
                "n_trades": 0, "swap_rate": 0}

    m = calc_metrics(oos_eq, oos_trades, oos_exp, oos_eq[0], oos_dates)
    swap_rate = n_swap / max(1, n_contested) * 100

    return {
        "cagr":      m.get("cagr", 0),
        "max_dd":    m.get("max_dd", 0),
        "calmar":    m.get("calmar", 0),
        "sharpe":    m.get("sharpe", 0),
        "n_trades":  m.get("n_trades", 0),
        "turnover":  m.get("turnover_yr", 0),
        "swap_rate": round(swap_rate, 1),
    }


# ─────────────────────────────────────────────────────────────────────
#  WF EVALUATOR
# ─────────────────────────────────────────────────────────────────────

def evaluate_wf(baseline_folds: list[dict], scored_folds: list[dict]) -> dict:
    dcagrs=[]; ddds=[]; dcalmars=[]; swaps=[]; dturn=[]
    n_pass = 0

    for bf, sf in zip(baseline_folds, scored_folds):
        dc  = sf["cagr"]   - bf["cagr"]
        dd  = -(sf["max_dd"] - bf["max_dd"])   # positive = worse (already %)
        dcal= sf["calmar"] - bf["calmar"]
        sw  = sf["swap_rate"]
        dt  = sf["turnover"] - bf["turnover"]

        dcagrs.append(dc); ddds.append(dd); dcalmars.append(dcal)
        swaps.append(sw);  dturn.append(dt)

        if dc >= 0 and dd <= ADOPT_DMAXDD:
            n_pass += 1

    avg_dc   = float(np.mean(dcagrs))
    avg_dd   = float(np.mean(ddds))
    avg_dcal = float(np.mean(dcalmars))
    avg_swap = float(np.mean(swaps))
    avg_dt   = float(np.mean(dturn))

    adopted  = (n_pass >= ADOPT_WF_MIN
                and avg_dc >= ADOPT_DCAGR
                and avg_dd <= ADOPT_DMAXDD
                and avg_swap <= ADOPT_SWAP_MAX)

    return {
        "wf_score":    n_pass,
        "avg_dcagr":   round(avg_dc, 2),
        "avg_ddd":     round(avg_dd, 2),
        "avg_dcalmar": round(avg_dcal, 3),
        "avg_swap":    round(avg_swap, 1),
        "avg_dturn":   round(avg_dt, 3),
        "fold_dcagrs": [round(x, 2) for x in dcagrs],
        "fold_ddds":   [round(x, 2) for x in ddds],
        "fold_swaps":  [round(x, 1) for x in swaps],
        "adopted":     adopted,
    }


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(
    p2_df: pd.DataFrame,
    p3_df: pd.DataFrame,
    wf_results: dict,   # score_id -> {"wf": ..., "folds": []}
    baseline_folds: list[dict],
    output_path: Path,
) -> None:
    L = []; w = L.append

    w("# RSR Ranking Quality Audit")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n採用条件: WF≥{ADOPT_WF_MIN}/5, ΔCAGR≥+{ADOPT_DCAGR}pp, ΔDD≤+{ADOPT_DMAXDD}pp, swap≤{ADOPT_SWAP_MAX}%\n")

    adopted_scores = [sid for sid, r in wf_results.items() if r["wf"].get("adopted")]
    w(f"**採用スコア**: {len(adopted_scores)}件  |  評価スコア: {len(SCORES)}件\n")

    # ── S1. ベースライン ──────────────────────────────────────────────
    w("---\n## 1. ベースライン (S1=RSR)\n")
    w("| Fold | CAGR | MaxDD | Calmar | n_trades |")
    w("|---|---|---|---|---|")
    for i, fd in enumerate(baseline_folds):
        w(f"| Fold{i+1} | {fd['cagr']:+.1f}% | {abs(fd['max_dd']):.1f}% | {fd['calmar']:.2f} | {fd['n_trades']} |")
    w(f"| **avg** | **{np.mean([f['cagr'] for f in baseline_folds]):+.1f}%** "
      f"| **{np.mean([abs(f['max_dd']) for f in baseline_folds]):.1f}%** "
      f"| **{np.mean([f['calmar'] for f in baseline_folds]):.2f}** | — |")

    # ── S2. Phase 2 ───────────────────────────────────────────────────
    w("\n---\n## 2. Phase 2: 単変量 Rank Quality\n")
    w("| Feature | n | Spearman ρ | p-value | top-decile avg | bot-decile avg | overall avg | lift |")
    w("|---|---|---|---|---|---|---|---|")
    for _, row in p2_df.iterrows():
        sig_mark = "**" if float(row["p_value"]) < 0.05 else ""
        w(f"| {row['feature']} "
          f"| {row['n']} "
          f"| {sig_mark}{row['spearman_rho']:+.4f}{sig_mark} "
          f"| {row['p_value']:.4f} "
          f"| {row['top_decile_avg'] if row['top_decile_avg'] is not None else '—':>6} "
          f"| {row['bot_decile_avg'] if row['bot_decile_avg'] is not None else '—':>6} "
          f"| {row['overall_avg']:+.2f}% "
          f"| {row['lift'] if row['lift'] is not None else '—'} |")
    w("\n_**bold** = p<0.05_")

    # ── S3. Phase 3 ───────────────────────────────────────────────────
    w("\n---\n## 3. Phase 3: 複合スコア品質 (全データ / 研究参考)\n")
    w("\n> ⚠ Phase 3 は IS/OOS 分割なし。研究参考値のみ。WF結果 (Phase 4) が採否の根拠。\n")
    w("| Score | 説明 | top1 avg_fwd60 | top3 avg_fwd60 | winner_rate | swap_rate |")
    w("|---|---|---|---|---|---|")
    for _, row in p3_df.iterrows():
        w(f"| {row['score']} | {row['label']} "
          f"| {row['top1_avg_fwd60']:+.2f}% "
          f"| {row['top3_avg_fwd60']:+.2f}% "
          f"| {row['winner_rate_pct']:.1f}% "
          f"| {row['swap_rate_pct']:.1f}% |")

    # ── S4. Phase 4 WF サマリ ─────────────────────────────────────────
    w("\n---\n## 4. Phase 4: WF Replay Test サマリ\n")
    w("| Score | 説明 | WF | ΔCAGR | ΔDD | ΔCalmar | avg_swap | 採用 |")
    w("|---|---|---|---|---|---|---|---|")
    for sid in SCORES:
        r = wf_results[sid]
        wf = r["wf"]
        mark = "**✅**" if wf["adopted"] else "❌"
        w(f"| {sid} | {SCORE_LABELS[sid]} "
          f"| {wf['wf_score']}/5 "
          f"| {wf['avg_dcagr']:+.2f}pp "
          f"| {wf['avg_ddd']:+.2f}pp "
          f"| {wf['avg_dcalmar']:+.3f} "
          f"| {wf['avg_swap']:.1f}% "
          f"| {mark} |")

    # ── S5. 採用スコア詳細 ────────────────────────────────────────────
    if adopted_scores:
        w("\n---\n## 5. 採用スコア 詳細\n")
        for sid in adopted_scores:
            r  = wf_results[sid]
            wf = r["wf"]
            w(f"\n### ✅ {sid}: {SCORE_LABELS[sid]}\n")
            w(f"- WF: **{wf['wf_score']}/5**  |  avg ΔCAGR: **{wf['avg_dcagr']:+.2f}pp**")
            w(f"- avg ΔDD: **{wf['avg_ddd']:+.2f}pp**  |  avg ΔCalmar: {wf['avg_dcalmar']:+.3f}")
            w(f"- avg swap_rate: {wf['avg_swap']:.1f}%  |  avg Δturnover: {wf['avg_dturn']:+.3f}")
            w("\n| Fold | ΔCAGR | ΔDD | swap | 判定 |")
            w("|---|---|---|---|---|")
            for i, (dc, dd, sw) in enumerate(zip(
                wf["fold_dcagrs"], wf["fold_ddds"], wf["fold_swaps"]
            )):
                ok = "✅" if dc >= 0 and dd <= ADOPT_DMAXDD else "❌"
                w(f"| Fold{i+1} | {dc:+.2f}pp | {dd:+.2f}pp | {sw:.1f}% | {ok} |")

    # ── S6. 仮説検証 ──────────────────────────────────────────────────
    w("\n---\n## 6. 仮説検証サマリ\n")
    w("| 仮説 | 検証方法 | 結果 |")
    w("|---|---|---|")

    # H1: RSR × 状態 × 鮮度 複合 > RSR 単独
    best_sid  = max(SCORES, key=lambda s: wf_results[s]["wf"]["avg_dcagr"])
    best_wf   = wf_results[best_sid]["wf"]
    h1_result = "✅ CONFIRM" if adopted_scores else f"❌ REJECT (best={best_wf['avg_dcagr']:+.2f}pp, WF={best_wf['wf_score']}/5)"
    w(f"| 複合スコア順位 > RSR 単独順位 | WF ΔCAGR ≥ +1pp | {h1_result} |")

    # H2: slope20 が最有効 (from Phase 2)
    if not p2_df.empty:
        best_feat = p2_df.iloc[0]["feature"]
        best_rho  = p2_df.iloc[0]["spearman_rho"]
        w(f"| 最有効特徴量 の予測力 | Spearman ρ | {best_feat}: ρ={best_rho:+.4f} |")

    # H3: swap rate <= 15% (低侵襲)
    swap_h3 = all(
        wf_results[s]["wf"]["avg_swap"] <= ADOPT_SWAP_MAX for s in SCORES
    )
    min_swap = min(wf_results[s]["wf"]["avg_swap"] for s in SCORES)
    max_swap = max(wf_results[s]["wf"]["avg_swap"] for s in SCORES)
    w(f"| swap ≤ 15% (低侵襲) | OOS avg_swap | {'✅' if swap_h3 else '⚠'} range: {min_swap:.1f}%-{max_swap:.1f}% |")

    # ── S7. 結論 ──────────────────────────────────────────────────────
    w("\n---\n## 7. 結論\n")
    if adopted_scores:
        w(f"**採用スコア: {', '.join(adopted_scores)}**\n")
        best = max(adopted_scores, key=lambda s: wf_results[s]["wf"]["avg_dcagr"])
        bwf  = wf_results[best]["wf"]
        w(f"推奨 1位: **{best}** ({SCORE_LABELS[best]})")
        w(f"- ΔCAGR={bwf['avg_dcagr']:+.2f}pp, WF={bwf['wf_score']}/5, swap={bwf['avg_swap']:.1f}%\n")
        w("実装前確認事項 (ASK_FIRST):")
        w("- signal_bridge.py の候補ソート変更に該当")
        w("- IS z-score 更新頻度 (月次 or 四半期) を定義")
    else:
        w("**採用スコアなし。RSR単独順位が現状最適。**\n")
        w(f"- 最良 ΔCAGR: {best_wf['avg_dcagr']:+.2f}pp (WF={best_wf['wf_score']}/5)  ← 採用基準未満")
        w("- 順位改善効果はあっても OOS での再現性が不十分")
        w("- Phase 2 の Spearman ρ が有意でも、ポートフォリオ全体 CAGR には非線形に作用")
        w("\n次ステップ候補:")
        w("- G: 2025 低調年 (Fold5 CAGR +1.6%) の原因分析")
        w("- H: risk_pct 感度 WF (既存 CONDITIONAL 判定の再評価)")
        w("- I: Exit RSR 70 本番適用 WF (B grade: WF 4/5)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  RSR Ranking Quality Audit")
    print("  除外なし・順位改善のみ × WF 5-fold")
    print("=" * 68 + "\n")

    # ── Data load ────────────────────────────────────────────────────
    print("[1/6] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: idx for idx, s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    common_dates = common_dates[
        (common_dates >= pd.Timestamp(IS_START)) &
        (common_dates <= pd.Timestamp("2025-12-31"))
    ]
    n_dates = len(common_dates)

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

        sector = trade_syms.get(sym, "")
        rule   = SECTOR_STRATEGY.get(sector, "fujiko")
        rsr_s  = rsr_df[sym] if sym in rsr_df.columns else None
        if rule == "mean_rev":
            st = MeanReversionStrategy(**MR_PARAMS)
        else:
            st = FujikoStrategy(
                min_rsr=cfg.fujiko.min_rsr, turtle_exit=cfg.fujiko.turtle_exit,
                rsr_series=rsr_s, min_sepa=cfg.fujiko.min_sepa,
                mom_period=cfg.fujiko.mom_period, turtle_entry=cfg.fujiko.turtle_entry,
                use_turtle_entry=cfg.fujiko.use_turtle_entry,
            )
        required = 252 + getattr(st, "mom_period", 21) + 2
        if hasattr(st, "precompute_signals") and len(df_src) >= required:
            sig_series = st.precompute_signals(df_src)
            sig_mat[:, si] = sig_series.to_numpy(dtype=np.int8)[row_idx]
            sig_ready[si]  = True

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0,
    )

    # ── Feature matrices ─────────────────────────────────────────────
    print("[2/6] 特徴量マトリクス計算...")
    cross70_mat          = compute_cross_mat(rsr_mat, 70.0)
    cross80_mat          = compute_cross_mat(rsr_mat, 80.0)
    slope5_mat, slope20_mat = compute_slope_mats(rsr_mat)
    atr20_mat            = compute_atr20_mat(high_mat, low_mat, close_mat)
    peak_dist_mat        = compute_rsr_peak_dist(rsr_mat, lookback=20)

    # state_score_mat
    state_score_mat = np.zeros((n_dates, n_syms), dtype=np.float32)
    for i in range(n_dates):
        s5_row  = slope5_mat[i]
        s20_row = slope20_mat[i]
        for si in range(n_syms):
            st = classify_entry_state(float(s5_row[si]), float(s20_row[si]))
            state_score_mat[i, si] = STATE_SCORE.get(st, 0.0)

    print("  特徴量計算完了")

    # ── Phase 1: Signal collection ───────────────────────────────────
    print("[3/6] Phase 1: シグナル収集...")
    df_signals = collect_signals(
        close_mat, sig_mat, sig_ready, rsr_mat,
        cross70_mat, cross80_mat, slope5_mat, slope20_mat,
        atr20_mat, peak_dist_mat,
        active_syms, sym_to_i, common_dates,
        min_rsr=MIN_RSR_COL,
    )
    print(f"  収集シグナル数: {len(df_signals)} 件  (dates={df_signals['date'].nunique()}日)")

    # CSV保存
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_signals.to_csv(REPORTS_DIR / "ranking_signals.csv", index=False, encoding="utf-8-sig")

    # ── Phase 2: Rank quality ────────────────────────────────────────
    print("[4/6] Phase 2: 単変量 rank quality...")
    p2_df = phase2_rank_quality(df_signals)
    p2_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    print("  特徴量別 Spearman ρ:")
    for _, row in p2_df.iterrows():
        sig = "*" if float(row["p_value"]) < 0.05 else " "
        print(f"    {row['feature']:<20} ρ={row['spearman_rho']:+.4f}{sig}  "
              f"top={row['top_decile_avg']!r:>7}  bot={row['bot_decile_avg']!r:>7}")

    # ── Phase 3: Composite score quality ────────────────────────────
    print("[5/6] Phase 3: 複合スコア品質...")
    p3_df = phase3_score_quality(df_signals)
    print("  Score quality (全データ参考):")
    for _, row in p3_df.iterrows():
        print(f"    {row['score']}: top1={row['top1_avg_fwd60']:+.2f}%  "
              f"top3={row['top3_avg_fwd60']:+.2f}%  "
              f"winner={row['winner_rate_pct']:.1f}%  "
              f"swap={row['swap_rate_pct']:.1f}%")

    # ── Phase 4: WF Replay ───────────────────────────────────────────
    print(f"\n[6/6] Phase 4: WF Replay ({len(SCORES)} スコア × 5 fold)...")

    all_wf_results = {}

    for sid in SCORES:
        print(f"\n  === {sid}: {SCORE_LABELS[sid]} ===")
        fold_results = []

        for fold in FOLDS:
            # IS z-params
            z_params = {
                "slope20":     _z_params_from_is(slope20_mat, sig_mat, sig_ready, rsr_mat,
                                                  common_dates, fold["is_end"]),
                "slope5":      _z_params_from_is(slope5_mat, sig_mat, sig_ready, rsr_mat,
                                                  common_dates, fold["is_end"]),
                "d70":         _z_params_from_is(cross70_mat.astype(np.float32),
                                                  sig_mat, sig_ready, rsr_mat,
                                                  common_dates, fold["is_end"]),
                "state_score": _z_params_from_is(state_score_mat, sig_mat, sig_ready, rsr_mat,
                                                  common_dates, fold["is_end"]),
            }

            score_mat = build_score_mat(
                sid, rsr_mat, slope5_mat, slope20_mat,
                cross70_mat, state_score_mat, z_params, n_dates, n_syms,
            )

            fd = run_fold_with_score(
                open_mat, close_mat, sig_mat, sig_ready, rsr_mat,
                score_mat,
                active_syms, sym_to_i, trade_syms, cfg,
                common_dates,
                oos_start_str=fold["oos_start"],
                oos_end_str=fold["oos_end"],
            )
            fold_results.append(fd)

            print(f"    Fold{fold['id']} OOS: CAGR={fd['cagr']:+.1f}%  "
                  f"DD={abs(fd['max_dd']):.1f}%  swap={fd['swap_rate']:.1f}%")

        # S1 baseline
        if sid == "S1":
            baseline_folds = fold_results

        all_wf_results[sid] = {
            "folds": fold_results,
            "wf":    {},  # filled below
        }

    # Evaluate WF relative to S1 baseline
    for sid in SCORES:
        wf_stats = evaluate_wf(baseline_folds, all_wf_results[sid]["folds"])
        all_wf_results[sid]["wf"] = wf_stats

    # Console summary
    print(f"\n{'='*68}")
    print(f"  S1 (baseline) avg CAGR: {np.mean([f['cagr'] for f in baseline_folds]):+.1f}%\n")
    for sid in SCORES[1:]:
        wf = all_wf_results[sid]["wf"]
        ok = "✅" if wf["adopted"] else "❌"
        print(f"  {ok} {sid}: WF={wf['wf_score']}/5  ΔCAGR={wf['avg_dcagr']:+.2f}pp  "
              f"ΔDD={wf['avg_ddd']:+.2f}pp  swap={wf['avg_swap']:.1f}%")
    print(f"{'='*68}")

    # ── Reports ──────────────────────────────────────────────────────
    write_report(p2_df, p3_df, all_wf_results, baseline_folds,
                 REPORTS_DIR / "ranking_quality_audit.md")

    # Detail CSV
    csv_rows = []
    for sid in SCORES:
        wf = all_wf_results[sid]["wf"]
        for i, fd in enumerate(all_wf_results[sid]["folds"]):
            bl = baseline_folds[i]
            csv_rows.append({
                "score":       sid,
                "fold":        i + 1,
                "cagr":        fd["cagr"],
                "max_dd":      fd["max_dd"],
                "calmar":      fd["calmar"],
                "n_trades":    fd["n_trades"],
                "swap_rate":   fd["swap_rate"],
                "dcagr":       round(fd["cagr"] - bl["cagr"], 2),
                "ddd":         round(-(fd["max_dd"] - bl["max_dd"]), 2),
            })

    with open(REPORTS_DIR / "ranking_quality_detail.csv", "w",
              newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader(); writer.writerows(csv_rows)
    print(f"  CSV: reports/ranking_quality_detail.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
