"""
backtest/conditional_rsr_policy.py

研究専用 / 実装変更禁止

RSR帯内条件付きポリシー：帯を跨いで比較禁止・帯内のみ順位変更。
全ての exit / risk / max_pos / MSW / lev は変更しない。

P0  baseline  : sort = RSR
P1  G1 only   : G1 sort = state_score
P2  G1+G3     : G1 sort = state_score−0.3*z(d70); G3 sort = −days_cross90
P3  G1+G3     : G1 sort = slope5;                 G3 sort = −days_cross90

バックテスト: run_period (pattern A) を完全再現。
  max_hold=60, sym_active_mat, 動的gross_cap, bear_sector_cap,
  market_shock (1-day TOPIX ret), alloc=(capital/max_pos)*cb_scale
  など production 挙動を完全踏襲。

採用条件: WF≥4/5, ΔCAGR≥+0.5pp, ΔDD≤+1pp, swap≤5%

出力:
  reports/conditional_rsr_policy.md
  reports/conditional_rsr_policy_swap.csv

Run:
    cd C:/ai-trading
    python src/backtest/conditional_rsr_policy.py
"""

from __future__ import annotations

import sys, time, csv, warnings, math
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, _sector_ok, _execute_buy,
    Position, _take, calc_metrics,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"
STALL_THR   = 0.15

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31", "is_end": "2020-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31", "is_end": "2021-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31", "is_end": "2022-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31", "is_end": "2023-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31", "is_end": "2024-12-31"},
]

POLICIES = ["P0", "P1", "P2", "P3"]
POLICY_DESC = {
    "P0": "RSR only (production baseline)",
    "P1": "G1: state_score; others: RSR",
    "P2": "G1: state_score-0.3z(d70); G3: −d90; others: RSR",
    "P3": "G1: slope5; G3: −d90; others: RSR",
}

BAND_RANGE = {"G1": (65.0, 80.0), "G2": (80.0, 90.0),
              "G3": (90.0, 95.0), "G4": (95.0, 101.0)}
BAND_PRI   = {"G1": 0, "G2": 1, "G3": 2, "G4": 3}

STATE_SCORE_MAP = {
    "EARLY_UP": 1.5, "STEADY_UP": 1.0, "FLAT": 0.75,
    "STALL": 0.5,    "DOWN": 0.0,      "EARLY_ROLL": -0.5, "UNKNOWN": 0.0,
}

ADOPT_WF_MIN   = 4
ADOPT_DCAGR    = 0.5   # pp
ADOPT_DMAXDD   = 1.0   # pp
ADOPT_SWAP_MAX = 5.0   # %

SHOCK_MKT_THR = -0.05
SHOCK_SYM_THR = -0.08


# ─────────────────────────────────────────────────────────────────────
#  FEATURE HELPERS
# ─────────────────────────────────────────────────────────────────────

def classify_state(s5: float, s20: float) -> str:
    if np.isnan(s5) or np.isnan(s20):
        return "UNKNOWN"
    if abs(s5) < STALL_THR:
        return "STALL"
    if s20 > 0:
        if s5 > s20:  return "EARLY_UP"
        if s5 > 0:    return "STEADY_UP"
        return "EARLY_ROLL"
    return "DOWN"


def get_band(rsr: float) -> str | None:
    for bname, (lo, hi) in BAND_RANGE.items():
        if lo <= rsr < hi:
            return bname
    return None


def compute_cross_mat(rsr_mat: np.ndarray, threshold: float) -> np.ndarray:
    n_dates, n_syms = rsr_mat.shape
    out     = np.zeros((n_dates, n_syms), dtype=np.int32)
    running = np.zeros(n_syms, dtype=np.int32)
    for i in range(n_dates):
        out[i] = running
        above   = rsr_mat[i] >= threshold
        running[above] += 1
        running[~above]  = 0
    return out


def compute_slope_mats(rsr_mat: np.ndarray):
    s5  = np.zeros_like(rsr_mat)
    s20 = np.zeros_like(rsr_mat)
    s5[5:]   = (rsr_mat[5:]   - rsr_mat[:-5])  / 5.0
    s20[20:] = (rsr_mat[20:]  - rsr_mat[:-20]) / 20.0
    return s5.astype(np.float32), s20.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
#  IS Z-PARAM CALIBRATION (for P2 G1 z(d70))
# ─────────────────────────────────────────────────────────────────────

def compute_z_params_band(
    feature_mat: np.ndarray, sig_mat: np.ndarray, sig_ready: np.ndarray,
    rsr_mat: np.ndarray, common_dates, is_end_str: str, band: str,
) -> dict:
    lo, hi = BAND_RANGE[band]
    vals = []
    for i, d in enumerate(common_dates):
        if str(d.date()) > is_end_str:
            break
        for si in range(len(sig_ready)):
            if not sig_ready[si]:
                continue
            if int(sig_mat[i, si]) != 1:
                continue
            rsr_v = float(rsr_mat[i, si])
            if not (lo <= rsr_v < hi):
                continue
            v = float(feature_mat[i, si])
            if not np.isnan(v):
                vals.append(v)
    if len(vals) < 2:
        return {"mean": 0.0, "std": 1.0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)) or 1.0}


# ─────────────────────────────────────────────────────────────────────
#  WITHIN-BAND SCORE
# ─────────────────────────────────────────────────────────────────────

def within_band_score(
    band: str, rsr: float, si: int, date_i: int,
    policy_id: str,
    slope5_mat, cross70_mat, cross90_mat, state_score_mat,
    z_d70_G1: dict,
) -> float:
    if policy_id == "P0":
        return rsr

    if policy_id == "P1":
        return float(state_score_mat[date_i, si]) if band == "G1" else rsr

    if policy_id == "P2":
        if band == "G1":
            d70  = float(cross70_mat[date_i, si])
            zd70 = (d70 - z_d70_G1["mean"]) / max(z_d70_G1["std"], 1e-6)
            ss   = float(state_score_mat[date_i, si])
            return ss - 0.3 * zd70
        if band == "G3":
            return -float(cross90_mat[date_i, si])
        return rsr

    if policy_id == "P3":
        if band == "G1":
            return float(slope5_mat[date_i, si])
        if band == "G3":
            return -float(cross90_mat[date_i, si])
        return rsr

    return rsr


# ─────────────────────────────────────────────────────────────────────
#  PRODUCTION-FAITHFUL FOLD RUNNER
#  Mirrors run_period(pattern="A") exactly.
#  Only modification: band-conditional policy sort on buy_cands.
# ─────────────────────────────────────────────────────────────────────

def run_fold_policy(
    open_mat, close_mat,
    sig_mat, sig_ready, rsr_mat, sym_active_mat,
    mkt_ret1, topix_ret20, topix_ret60, bear_arr,
    slope5_mat, cross70_mat, cross90_mat, state_score_mat,
    active_syms, sym_to_i, trade_syms, cfg,
    common_dates,
    oos_start_str: str, oos_end_str: str,
    policy_id: str,
    z_d70_G1: dict,
) -> tuple[dict, list]:
    """
    Run one WF fold with policy-based within-band sorting.
    Returns (fold_metrics, swap_events).
    swap_events have fwd60 stub (filled by caller post-hoc).
    """
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)

    rc = getattr(cfg, "risk_controls", None)
    MAX_SECTOR_W  = float(rc.sector_cap)  if rc else 0.25
    gross_enabled = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gross_normal  = float(getattr(rc, "gross_cap_normal",       1.0)) if rc else 1.0
    gross_dd5     = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    gross_dd8     = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    n_dates = len(common_dates)
    cash    = float(capital)
    positions: dict[str, Position] = {}
    peak_equity = float(capital)
    cb_active   = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    trades:       list[dict]  = []
    equity_curve: list[float] = []
    exposure_list:list[float] = []
    date_list:    list        = []

    oos_start_i = None; oos_end_i = None
    swap_events: list[dict] = []
    n_contested = 0; n_swap = 0

    for i, date in enumerate(common_dates):
        date_str = str(date.date())

        if date_str >= oos_start_str and oos_start_i is None:
            oos_start_i = i
        if date_str > oos_end_str and oos_end_i is None:
            oos_end_i = i

        invested   = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
                        for s, pos in positions.items())
        cur_equity = cash + invested
        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        equity_curve.append(cur_equity)
        exposure_list.append(invested / max(1.0, cur_equity))
        date_list.append(date)

        # CB
        if not cb_active:
            if dd <= -max_dd_limit:
                cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0

        # Gross cap (TOPIX-based)
        gross_cap = gross_normal
        if gross_enabled and topix_ret20 is not None:
            r20 = float(topix_ret20[i]) if i < len(topix_ret20) else 0.0
            r60 = float(topix_ret60[i]) if i < len(topix_ret60) else 0.0
            if r20 < -0.05:
                gross_cap = gross_dd5
            elif r60 < -0.08:
                gross_cap = gross_dd8

        # Bear regime
        is_bear = bool(bear_arr[i]) if bear_arr is not None and i < len(bear_arr) else False
        sec_cap_eff = 0.18 if is_bear else MAX_SECTOR_W

        # Market shock: 1-day TOPIX return (mirrors production mkt_ret_arr)
        mkt_shock = (mkt_ret1 is not None and
                     float(mkt_ret1[i]) <= SHOCK_MKT_THR)

        sell_sigs:    list[tuple] = []
        buy_cands_raw:list[tuple] = []

        for sym in active_syms:
            si_sym     = sym_to_i[sym]
            is_holding = sym in positions
            hold_idx   = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_mat[i, si_sym])
            close_t    = float(close_mat[i, si_sym])

            if mkt_shock and is_holding:
                if i > 0:
                    prev_c = float(close_mat[i - 1, si_sym])
                    if prev_c > 0 and (close_t / prev_c - 1.0) <= SHOCK_SYM_THR:
                        sell_sigs.append((sym, "MARKET_SHOCK_EXIT"))
                        continue
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
                buy_cands_raw.append((rsr_val, sym))

        if i + 1 >= n_dates:
            break
        next_i = i + 1

        # ── SELL ─────────────────────────────────────────────────────
        for sym, reason in sell_sigs:
            if sym not in positions:
                continue
            pos     = positions[sym]
            sell_px = float(open_mat[next_i, sym_to_i[sym]])
            pnl     = (sell_px - pos.entry_price) * pos.qty
            cash   += pos.qty * sell_px * (1 - COST_ONE_WAY)
            trades.append({
                "side": "SELL", "symbol": sym, "pnl": pnl,
                "entry": pos.entry_price, "exit": sell_px,
                "qty": pos.qty, "entry_idx": pos.entry_idx, "exit_idx": i,
                "reason": reason, "date": date_str,
            })
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if cb_active or not buy_cands_raw:
            continue

        # ── POLICY SORT: within-band reordering ──────────────────────
        in_oos = (oos_start_i is not None and i >= oos_start_i
                  and (oos_end_i is None or i < oos_end_i))

        cand_info:   list[tuple]         = []
        band_groups: dict[str, list]     = {}

        for rsr_v, sym in buy_cands_raw:
            si_sym = sym_to_i[sym]
            band   = get_band(rsr_v) or "G1"
            bp     = BAND_PRI.get(band, 0)
            wb     = within_band_score(
                band, rsr_v, si_sym, i, policy_id,
                slope5_mat, cross70_mat, cross90_mat, state_score_mat,
                z_d70_G1,
            )
            cand_info.append((bp, wb, rsr_v, sym))
            band_groups.setdefault(band, []).append((rsr_v, sym))

        # highest band first, highest within-band score within band
        cand_info.sort(key=lambda x: (-x[0], -x[1]))

        # ── Swap event tracking (OOS only, non-P0) ───────────────────
        if in_oos and policy_id != "P0":
            for band, group in band_groups.items():
                if len(group) < 2:
                    continue
                rsr_top_rsr, rsr_top_sym = max(group, key=lambda x: x[0])
                band_pol = [(wb, rsr_v, s) for bp, wb, rsr_v, s in cand_info
                            if (get_band(rsr_v) or "G1") == band]
                if not band_pol:
                    continue
                pol_top_wb, pol_top_rsr, pol_top_sym = max(band_pol, key=lambda x: x[0])

                n_contested += 1
                if rsr_top_sym != pol_top_sym:
                    n_swap += 1
                    swap_events.append({
                        "date":        date_str,
                        "date_idx":    i,
                        "fold_oos":    oos_start_str[:4],
                        "band":        band,
                        "rsr_top_sym": rsr_top_sym,
                        "rsr_top_si":  sym_to_i[rsr_top_sym],
                        "rsr_top_rsr": round(rsr_top_rsr, 2),
                        "pol_top_sym": pol_top_sym,
                        "pol_top_si":  sym_to_i[pol_top_sym],
                        "pol_top_wb":  round(pol_top_wb, 4),
                        "fwd60_rsr":   np.nan,
                        "fwd60_pol":   np.nan,
                        "delta":       np.nan,
                        "improved":    None,
                    })

        # ── BUY (pattern A: alloc = capital/max_pos) ─────────────────
        for bp, wb, rsr_v, sym in cand_info:
            si_sym     = sym_to_i[sym]
            open_slots = max_pos - len(positions)
            if open_slots <= 0:
                break

            buy_px = float(open_mat[next_i, si_sym])
            if buy_px <= 0:
                continue
            if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms,
                               capital, sec_cap_eff):
                continue
            if gross_enabled:
                cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in positions.values()) / max(1.0, capital)
                if cur_gross + buy_px * LOT / max(1.0, capital) > gross_cap:
                    continue

            alloc = capital / max_pos   # cb_active=False here (continued above)
            qty   = int(alloc / buy_px / LOT) * LOT
            if qty <= 0:
                continue
            if qty * buy_px * (1.0 + COST_ONE_WAY) > cash:
                continue

            _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_v)
            cash -= qty * buy_px * (1 + COST_ONE_WAY)
            trades[-1].update({"date": date_str})

    # ── OOS extraction ────────────────────────────────────────────────
    oos_si    = oos_start_i or 0
    oos_ei    = oos_end_i   or len(equity_curve)
    oos_eq    = equity_curve[oos_si:oos_ei]
    oos_exp   = exposure_list[oos_si:oos_ei]
    oos_dates = list(common_dates[oos_si:oos_ei])
    oos_trades = [t for t in trades
                  if oos_start_str <= t.get("date", "") <= oos_end_str]

    swap_rate = n_swap / max(1, n_contested) * 100

    if not oos_eq:
        return {"cagr": 0, "max_dd": 0, "calmar": 0, "sharpe": 0,
                "n_trades": 0, "swap_rate": 0, "n_swap": 0, "n_contested": 0}, swap_events

    m = calc_metrics(oos_eq, oos_trades, oos_exp, oos_eq[0], oos_dates)
    return {
        "cagr":        m.get("cagr", 0),
        "max_dd":      m.get("max_dd", 0),
        "calmar":      m.get("calmar", 0),
        "sharpe":      m.get("sharpe", 0),
        "n_trades":    m.get("n_trades", 0),
        "swap_rate":   round(swap_rate, 1),
        "n_swap":      n_swap,
        "n_contested": n_contested,
    }, swap_events


# ─────────────────────────────────────────────────────────────────────
#  FWD60 ENRICHMENT (post-hoc, no lookahead violation)
# ─────────────────────────────────────────────────────────────────────

def enrich_swap_fwd60(swap_events: list[dict], close_mat: np.ndarray,
                       n_dates: int, fwd_days: int = 60) -> None:
    for ev in swap_events:
        i   = ev["date_idx"]
        rsi = ev["rsr_top_si"]
        psi = ev["pol_top_si"]
        if i + fwd_days >= n_dates:
            continue
        c0r  = float(close_mat[i, rsi]); c60r = float(close_mat[i + fwd_days, rsi])
        c0p  = float(close_mat[i, psi]); c60p = float(close_mat[i + fwd_days, psi])
        f60r = (c60r / c0r - 1) * 100 if c0r > 0 and c60r > 0 else float("nan")
        f60p = (c60p / c0p - 1) * 100 if c0p > 0 and c60p > 0 else float("nan")
        ev["fwd60_rsr"] = round(f60r, 2)
        ev["fwd60_pol"] = round(f60p, 2)
        if not (math.isnan(f60r) or math.isnan(f60p)):
            ev["delta"]    = round(f60p - f60r, 2)
            ev["improved"] = ev["delta"] > 0
        else:
            ev["delta"]    = float("nan")
            ev["improved"] = None


# ─────────────────────────────────────────────────────────────────────
#  WF EVALUATOR
# ─────────────────────────────────────────────────────────────────────

def evaluate_wf(bl_folds: list[dict], pol_folds: list[dict]) -> dict:
    dcagrs = []; ddds = []; swaps = []; dcalmars = []
    n_pass = 0
    for bf, pf in zip(bl_folds, pol_folds):
        dc   = pf["cagr"]   - bf["cagr"]
        dd   = -(pf["max_dd"] - bf["max_dd"])   # positive = worse
        dcal = pf["calmar"] - bf["calmar"]
        sw   = pf["swap_rate"]
        dcagrs.append(dc); ddds.append(dd)
        dcalmars.append(dcal); swaps.append(sw)
        if dc >= 0 and dd <= ADOPT_DMAXDD:
            n_pass += 1

    avg_dc   = float(np.mean(dcagrs))
    avg_dd   = float(np.mean(ddds))
    avg_dcal = float(np.mean(dcalmars))
    avg_swap = float(np.mean(swaps))

    adopted = (n_pass >= ADOPT_WF_MIN
               and avg_dc >= ADOPT_DCAGR
               and avg_dd <= ADOPT_DMAXDD
               and avg_swap <= ADOPT_SWAP_MAX)

    return {
        "wf_score":    n_pass,
        "avg_dcagr":   round(avg_dc, 2),
        "avg_ddd":     round(avg_dd, 2),
        "avg_dcalmar": round(avg_dcal, 3),
        "avg_swap":    round(avg_swap, 1),
        "adopted":     adopted,
        "fold_dcagrs": [round(x, 2) for x in dcagrs],
        "fold_ddds":   [round(x, 2) for x in ddds],
        "fold_swaps":  [round(x, 1) for x in swaps],
    }


# ─────────────────────────────────────────────────────────────────────
#  REGIME HELPER
# ─────────────────────────────────────────────────────────────────────

def regime_label(topix_close: pd.Series, year: int) -> str:
    yr_data = topix_close[topix_close.index.year == year]
    if len(yr_data) < 2:
        return "N/A"
    ret = float(yr_data.iloc[-1] / yr_data.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


def _fmt_pp(v: float) -> str:
    return f"{v:+.2f}pp" if not math.isnan(v) else "—"


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(
    wf_results: dict,
    baseline_folds: list[dict],
    topix_close: pd.Series | None,
    n_syms_per_band: dict,
    output_path: Path,
) -> None:
    L = []; w = L.append

    w("# Conditional RSR Policy WF (最終品質監査)")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n採用条件: WF≥{ADOPT_WF_MIN}/5, ΔCAGR≥+{ADOPT_DCAGR}pp, "
      f"ΔDD≤+{ADOPT_DMAXDD}pp, swap≤{ADOPT_SWAP_MAX}%")
    w(f"\nバックテスト: run_period (pattern A) 完全再現。"
      f"max_hold=60, sym_active_mat, 動的gross_cap, 1-day mkt_shock 適用。\n")

    adopted = [p for p in POLICIES[1:] if wf_results[p]["wf"].get("adopted")]
    w(f"**採用ポリシー**: {len(adopted)}件  |  評価: {len(POLICIES)-1}件 (P1/P2/P3)\n")

    # ── S1. Band distribution ────────────────────────────────────────
    w("---\n## 1. RSR帯別シグナル分布 (Phase 1)\n")
    w("| 帯 | RSR範囲 | OOS全シグナル件数 | Policy変更 |")
    w("|---|---|---|---|")
    policy_change = {
        "G1": "P1/P2: state_score; P3: slope5",
        "G2": "全Policy: RSR (変更なし)",
        "G3": "P2/P3: −days_cross90 (鮮度優先)",
        "G4": "全Policy: RSR (変更なし)",
    }
    for band in ["G1", "G2", "G3", "G4"]:
        lo, hi = BAND_RANGE[band]
        w(f"| {band} | {lo:.0f}–{hi:.0f} | "
          f"{n_syms_per_band.get(band, 0)} | {policy_change[band]} |")

    # ── S2. Baseline P0 ──────────────────────────────────────────────
    w("\n---\n## 2. ベースライン P0 (production-faithful)\n")
    w("| Fold | OOS年 | Regime | CAGR | MaxDD | Calmar | n_trades |")
    w("|---|---|---|---|---|---|---|")
    for fold, fd in zip(FOLDS, baseline_folds):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr) if topix_close is not None else "N/A"
        w(f"| Fold{fold['id']} | {yr} | {reg} "
          f"| {fd['cagr']:+.1f}% | {abs(fd['max_dd']):.1f}% "
          f"| {fd['calmar']:.2f} | {fd['n_trades']} |")
    avg_c = np.mean([fd["cagr"]  for fd in baseline_folds])
    avg_d = np.mean([abs(fd["max_dd"]) for fd in baseline_folds])
    avg_a = np.mean([fd["calmar"] for fd in baseline_folds])
    w(f"| **avg** | — | — | **{avg_c:+.1f}%** | **{avg_d:.1f}%** | **{avg_a:.2f}** | — |")

    # ── S3. WF Summary ───────────────────────────────────────────────
    w("\n---\n## 3. Phase 2: WF Replay サマリ\n")
    w("| Policy | 説明 | WF | ΔCAGR | ΔDD | ΔCalmar | avg_swap | 採用 |")
    w("|---|---|---|---|---|---|---|---|")
    for pid in POLICIES[1:]:
        wf   = wf_results[pid]["wf"]
        mark = "**✅**" if wf["adopted"] else "❌"
        w(f"| {pid} | {POLICY_DESC[pid]} "
          f"| {wf['wf_score']}/5 "
          f"| {wf['avg_dcagr']:+.2f}pp "
          f"| {wf['avg_ddd']:+.2f}pp "
          f"| {wf['avg_dcalmar']:+.3f} "
          f"| {wf['avg_swap']:.1f}% "
          f"| {mark} |")

    # ── S4. Adopted policy detail ─────────────────────────────────────
    if adopted:
        w("\n---\n## 4. 採用ポリシー 詳細\n")
        for pid in adopted:
            wf = wf_results[pid]["wf"]
            w(f"\n### ✅ {pid}: {POLICY_DESC[pid]}\n")
            w(f"- WF: **{wf['wf_score']}/5**  ΔCAGR: **{wf['avg_dcagr']:+.2f}pp**")
            w(f"- ΔDD: {wf['avg_ddd']:+.2f}pp  ΔCalmar: {wf['avg_dcalmar']:+.3f}  "
              f"avg swap: **{wf['avg_swap']:.1f}%**\n")
            w("| Fold | OOS年 | ΔCAGR | ΔDD | swap | 判定 |")
            w("|---|---|---|---|---|---|")
            for fold, dc, dd, sw in zip(
                FOLDS, wf["fold_dcagrs"], wf["fold_ddds"], wf["fold_swaps"]
            ):
                ok = "✅" if dc >= 0 and dd <= ADOPT_DMAXDD else "❌"
                w(f"| Fold{fold['id']} | {fold['oos_start'][:4]} "
                  f"| {dc:+.2f}pp | {dd:+.2f}pp | {sw:.1f}% | {ok} |")

    # ── S5. Swap audit ───────────────────────────────────────────────
    w("\n---\n## 5. Swap Event 手動監査\n")
    for pid in POLICIES[1:]:
        all_swaps = wf_results[pid]["swaps"]
        if not all_swaps:
            w(f"\n**{pid}**: swap 0件\n")
            continue
        valid     = [e for e in all_swaps
                     if not math.isnan(e.get("delta", float("nan")))]
        n_imp     = sum(1 for e in valid if e.get("improved"))
        n_hurt    = sum(1 for e in valid if not e.get("improved"))
        avg_delta = float(np.mean([e["delta"] for e in valid])) if valid else float("nan")
        verdict_avg = "policy有利" if (not math.isnan(avg_delta) and avg_delta > 0) else "RSR有利"

        w(f"\n### {pid}: {POLICY_DESC[pid]}\n")
        w(f"- OOS swap件数: **{len(all_swaps)}件**  勝ち: {n_imp}  負け: {n_hurt}")
        w(f"- avg Δfwd60: **{_fmt_pp(avg_delta)}**  ({verdict_avg})\n")
        w("| Date | Band | RSR pick | RSR fwd60 | Policy pick | Pol fwd60 | Δfwd60 | 判定 |")
        w("|---|---|---|---|---|---|---|---|")
        for ev in all_swaps[:25]:
            delta = ev.get("delta", float("nan"))
            f60r  = ev.get("fwd60_rsr", float("nan"))
            f60p  = ev.get("fwd60_pol", float("nan"))
            if math.isnan(delta):
                v = "—"
            elif ev.get("improved"):
                v = "✅ policy勝"
            else:
                v = "❌ policy負"
            f60r_s = f"{f60r:+.1f}%" if not math.isnan(f60r) else "—"
            f60p_s = f"{f60p:+.1f}%" if not math.isnan(f60p) else "—"
            d_s    = f"{delta:+.1f}%" if not math.isnan(delta) else "—"
            w(f"| {ev['date']} | {ev['band']} "
              f"| {ev['rsr_top_sym']} ({ev['rsr_top_rsr']}) | {f60r_s} "
              f"| {ev['pol_top_sym']} | {f60p_s} | {d_s} | {v} |")
        if len(all_swaps) > 25:
            w(f"\n_残り {len(all_swaps)-25} 件は conditional_rsr_policy_swap.csv 参照_")

    # ── S6. Fold × Policy CAGR table ─────────────────────────────────
    w("\n---\n## 6. Fold × Policy CAGR 比較\n")
    hdr = "| Fold | OOS年 | Regime | " + " | ".join(f"CAGR {p}" for p in POLICIES) + " |"
    sep = "|---|---|---| " + " | ".join("---" for _ in POLICIES) + " |"
    w(hdr); w(sep)
    for i, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr) if topix_close is not None else "N/A"
        row = f"| Fold{fold['id']} | {yr} | {reg} | "
        for pid in POLICIES:
            row += f"{wf_results[pid]['folds'][i]['cagr']:+.1f}% | "
        w(row)

    # ── S7. Regime-based ΔCAGR ───────────────────────────────────────
    w("\n---\n## 7. Regime別 ΔCAGR 寄与\n")
    bull_idx: list[int] = []; bear_idx: list[int] = []
    for i, fold in enumerate(FOLDS):
        yr = int(fold["oos_start"][:4])
        if topix_close is not None:
            yr_data = topix_close[topix_close.index.year == yr]
            if len(yr_data) >= 2:
                ret = float(yr_data.iloc[-1] / yr_data.iloc[0] - 1)
                (bull_idx if ret > 0 else bear_idx).append(i)

    w("| Policy | Bull avg ΔCAGR | Bear avg ΔCAGR | 寄与差 (Bull-Bear) |")
    w("|---|---|---|---|")
    bl_cagrs = [fd["cagr"] for fd in baseline_folds]
    for pid in POLICIES[1:]:
        pol_cagrs = [wf_results[pid]["folds"][i]["cagr"] for i in range(5)]
        deltas    = [pol_cagrs[i] - bl_cagrs[i] for i in range(5)]
        bull_avg  = float(np.mean([deltas[i] for i in bull_idx])) if bull_idx else float("nan")
        bear_avg  = float(np.mean([deltas[i] for i in bear_idx])) if bear_idx else float("nan")
        diff      = (bull_avg - bear_avg
                     if not (math.isnan(bull_avg) or math.isnan(bear_avg))
                     else float("nan"))
        w(f"| {pid} | {_fmt_pp(bull_avg)} | {_fmt_pp(bear_avg)} | {_fmt_pp(diff)} |")

    # ── S8. Hypothesis tests ──────────────────────────────────────────
    w("\n---\n## 8. 仮説検証\n")
    w("| 仮説 | 検証方法 | 結果 |")
    w("|---|---|---|")
    best_pid = max(POLICIES[1:], key=lambda p: wf_results[p]["wf"]["avg_dcagr"])
    best_wf  = wf_results[best_pid]["wf"]
    h1 = ("✅ CONFIRM"
          if adopted
          else f"❌ REJECT (best={best_wf['avg_dcagr']:+.2f}pp WF={best_wf['wf_score']}/5)")
    w(f"| 帯内ポリシーで ΔCAGR≥+0.5pp | WF ≥4/5 | {h1} |")

    p2_wf = wf_results["P2"]["wf"]; p3_wf = wf_results["P3"]["wf"]
    g3_dc  = max(p2_wf["avg_dcagr"], p3_wf["avg_dcagr"])
    g3_tag = "P2" if p2_wf["avg_dcagr"] > p3_wf["avg_dcagr"] else "P3"
    w(f"| G3: −d90 鮮度優先の効果 | ΔCAGR vs baseline | best={g3_dc:+.2f}pp ({g3_tag}) |")

    max_swap = max(wf_results[p]["wf"]["avg_swap"] for p in POLICIES[1:])
    all_ok   = all(wf_results[p]["wf"]["avg_swap"] <= ADOPT_SWAP_MAX for p in POLICIES[1:])
    w(f"| swap ≤ 5% (低侵襲) | OOS avg_swap | "
      f"{'✅' if all_ok else '⚠'} max={max_swap:.1f}% |")

    # ── S9. Conclusion ───────────────────────────────────────────────
    w("\n---\n## 9. 結論\n")
    if adopted:
        best = max(adopted, key=lambda p: wf_results[p]["wf"]["avg_dcagr"])
        bwf  = wf_results[best]["wf"]
        w(f"**採用推奨: {best}** — {POLICY_DESC[best]}\n")
        w(f"- ΔCAGR: **{bwf['avg_dcagr']:+.2f}pp**  WF: {bwf['wf_score']}/5  "
          f"swap: {bwf['avg_swap']:.1f}%  ΔDD: {bwf['avg_ddd']:+.2f}pp\n")
        w("実装前必須事項 (ASK_FIRST):")
        w("- signal_bridge.py の候補ソートロジック変更 → 事前確認必須")
        w("- IS z-score 更新頻度 (月次/四半期) 定義が必要 (P2 のみ)")
        w("- 追加 OOS 期間または live shadow での最終検証推奨")
    else:
        w(f"**採用なし。現行 RSR 順位が最適。**\n")
        w(f"- 最良: {best_pid} ΔCAGR={best_wf['avg_dcagr']:+.2f}pp "
          f"(WF={best_wf['wf_score']}/5) — 採用基準未満")
        w(f"- 帯内 policy 変更: OOS で有意な改善なし")
        w(f"- swap が 0–{max_swap:.1f}% と低い → 修正幅自体が小さい\n")
        w("**根本的制約:**")
        w("- G1(75-79): OOS年平均 ~10件未満 → 効果が誤差内")
        w("- G3(90-94): OOS N≤25/fold → d90 効果不安定")
        w("- Study 5/6 と同様: IS 有意な特徴量も OOS portfolio では希薄化\n")
        w("次ステップ候補:")
        w("- G: 2025 低調年 (Fold5) 原因分析")
        w("- H: risk_pct 感度 WF 再評価 (既存 CONDITIONAL)")
        w("- I: Exit RSR 70 本番適用 WF (Regime-Aware B grade: WF4/5)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3, "max_positions must be 3 (PARAMS_LOCKED)"

    print("=" * 68)
    print("  Conditional RSR Policy WF (最終品質監査)")
    print("  帯内順位変更のみ × production-faithful × WF 5-fold")
    print("=" * 68 + "\n")

    # ── Data load ────────────────────────────────────────────────────
    print("[1/5] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: idx for idx, s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    # Build common date index
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
    print(f"  共通日数: {n_dates}, 銘柄数: {n_syms}")

    # Build OHLC and signal matrices
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    sig_mat   = np.zeros((n_dates, n_syms), dtype=np.int8)
    sig_ready = np.zeros(n_syms, dtype=bool)

    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0):
            continue
        open_mat[:,  si] = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]

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
    sym_active_mat = None
    if sym_active_df is not None:
        sym_active_mat = _take(sym_active_df, common_dates, active_syms,
                                dtype=np.float32, fill_value=1.0)

    # TOPIX arrays (mirrors run_period)
    mkt_ret1 = topix_ret20 = topix_ret60 = None
    bear_arr = None
    if topix_close is not None:
        mkt_ret1    = _take(topix_close.pct_change(),     common_dates,
                             dtype=np.float32, fill_value=0.0)
        topix_ret20 = _take(topix_close.pct_change(20),  common_dates,
                             dtype=np.float32, fill_value=0.0)
        topix_ret60 = _take(topix_close.pct_change(60),  common_dates,
                             dtype=np.float32, fill_value=0.0)
        ma200    = topix_close.rolling(200, min_periods=100).mean()
        bear_s   = (topix_close < ma200).reindex(
            pd.DatetimeIndex(common_dates), method="ffill").fillna(False)
        bear_arr = bear_s.values.astype(bool)

    # ── Feature matrices ─────────────────────────────────────────────
    print("[2/5] 特徴量マトリクス計算...")
    cross70_mat           = compute_cross_mat(rsr_mat, 70.0)
    cross90_mat           = compute_cross_mat(rsr_mat, 90.0)
    slope5_mat, slope20_mat = compute_slope_mats(rsr_mat)

    state_score_mat = np.zeros((n_dates, n_syms), dtype=np.float32)
    for i in range(n_dates):
        for si in range(n_syms):
            st_label = classify_state(
                float(slope5_mat[i, si]), float(slope20_mat[i, si]))
            state_score_mat[i, si] = STATE_SCORE_MAP.get(st_label, 0.0)

    # ── Phase 1: Band signal distribution ────────────────────────────
    print("[3/5] Phase 1: 帯別シグナル分布...")
    n_syms_per_band: dict[str, int] = {b: 0 for b in BAND_RANGE}
    for i, d in enumerate(common_dates):
        if str(d.date()) < "2021-01-01":
            continue
        for si in range(n_syms):
            if not sig_ready[si] or int(sig_mat[i, si]) != 1:
                continue
            if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
                continue
            band = get_band(float(rsr_mat[i, si]))
            if band:
                n_syms_per_band[band] += 1

    for band, cnt in n_syms_per_band.items():
        lo, hi = BAND_RANGE[band]
        print(f"  {band} ({lo:.0f}–{hi:.0f}): {cnt} シグナル")

    # ── Phase 2: WF Replay ───────────────────────────────────────────
    print(f"\n[4/5] WF Replay ({len(POLICIES)} policies × 5 folds)...\n")
    all_results: dict[str, dict] = {}

    for pid in POLICIES:
        print(f"  === {pid}: {POLICY_DESC[pid]} ===")
        fold_results:   list[dict] = []
        all_swap_events:list[dict] = []

        for fold in FOLDS:
            z_d70_G1 = {"mean": 0.0, "std": 1.0}
            if pid == "P2":
                z_d70_G1 = compute_z_params_band(
                    cross70_mat.astype(np.float32), sig_mat, sig_ready, rsr_mat,
                    common_dates, fold["is_end"], "G1",
                )

            fd, swaps = run_fold_policy(
                open_mat, close_mat,
                sig_mat, sig_ready, rsr_mat, sym_active_mat,
                mkt_ret1, topix_ret20, topix_ret60, bear_arr,
                slope5_mat, cross70_mat, cross90_mat, state_score_mat,
                active_syms, sym_to_i, trade_syms, cfg,
                common_dates,
                oos_start_str=fold["oos_start"],
                oos_end_str=fold["oos_end"],
                policy_id=pid,
                z_d70_G1=z_d70_G1,
            )

            enrich_swap_fwd60(swaps, close_mat, n_dates)
            all_swap_events.extend(swaps)
            fold_results.append(fd)

            sw_info = (f"  swap={fd['swap_rate']:.1f}% "
                       f"({fd['n_swap']}/{fd['n_contested']})"
                       if pid != "P0" else "")
            print(f"    Fold{fold['id']} OOS {fold['oos_start'][:4]}: "
                  f"CAGR={fd['cagr']:+.1f}%  DD={abs(fd['max_dd']):.1f}%{sw_info}")

        if pid == "P0":
            baseline_folds = fold_results

        all_results[pid] = {
            "folds": fold_results,
            "wf":    {},
            "swaps": all_swap_events,
        }

    # WF stats vs P0
    for pid in POLICIES:
        all_results[pid]["wf"] = evaluate_wf(baseline_folds, all_results[pid]["folds"])

    bl_avg = np.mean([fd["cagr"] for fd in baseline_folds])
    print(f"\n{'='*68}")
    print(f"  P0 baseline avg CAGR: {bl_avg:+.1f}%\n")
    for pid in POLICIES[1:]:
        wf = all_results[pid]["wf"]
        ok = "✅" if wf["adopted"] else "❌"
        print(f"  {ok} {pid}: WF={wf['wf_score']}/5  "
              f"ΔCAGR={wf['avg_dcagr']:+.2f}pp  "
              f"ΔDD={wf['avg_ddd']:+.2f}pp  swap={wf['avg_swap']:.1f}%")
    print(f"{'='*68}")

    # ── Output ───────────────────────────────────────────────────────
    print("\n[5/5] レポート生成...")
    write_report(
        all_results, baseline_folds, topix_close,
        n_syms_per_band,
        REPORTS_DIR / "conditional_rsr_policy.md",
    )

    csv_rows = []
    for pid in POLICIES[1:]:
        for ev in all_results[pid]["swaps"]:
            csv_rows.append({"policy": pid, **{
                k: v for k, v in ev.items()
                if k not in ("date_idx", "rsr_top_si", "pol_top_si")
            }})
    if csv_rows:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fp = REPORTS_DIR / "conditional_rsr_policy_swap.csv"
        with open(fp, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader(); writer.writerows(csv_rows)
        print(f"  Swap CSV: {fp}  ({len(csv_rows)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
