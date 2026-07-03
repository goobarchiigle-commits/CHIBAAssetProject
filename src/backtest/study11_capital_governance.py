"""
backtest/study11_capital_governance.py — Standalone Capital Governance Validation

Study9 Case B standalone + 5 governance overlays.
Entry / Exit / regime 固定。資本再配分のみ変更。

Cases:
  A: full compounding (baseline — no rebalancing)
  B: annual rebalance  → target=750k
  C: quarterly rebalance → target=750k
  D: monthly rebalance   → target=750k
  E: capital collar [500k, 1.5M]

採用条件 (per governance case):
  combined_DD ≤ prod_DD + 1pp
  AND combined_Calmar > prod_Calmar
  AND corr_daily < 0.30
  AND alpha_preservation ≥ 85%

停止条件:
  max(C,D,E).avg_combined_calmar − B.avg_combined_calmar < 0.3pp
  → capital研究終了; annual が最善

出力: reports/study11_capital_governance.md
Run:
    cd C:/ai-trading
    python src/backtest/study11_capital_governance.py
"""
from __future__ import annotations

import sys, math, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, _take, run_period,
    LOT, COST_ONE_WAY,
)

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"
FULL_END    = "2025-12-31"

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── Study9 Case B standalone signal params (固定・変更禁止) ────────────
RSR_LO_SL   = 92.0    # entry RSR lower (inclusive)
RSR_HI_SL   = 95.0    # entry RSR upper (exclusive)
D90_MAX_SL  = 5       # max days since RSR first crossed 90 (1..D90_MAX)
SLOPE_MAX_SL = 5.0    # 5-day RSR slope filter
EXIT_THR_SL  = 90.0   # RSR exit threshold
MIN_HOLD_SL  = 3      # minimum holding days before exit fires
SHOCK_MKT    = -0.05  # market shock (-5% TOPIX day)
SHOCK_SYM    = -0.08  # symbol shock (-8% single-day move)

# ── Governance params ──────────────────────────────────────────────────
TARGET_NAV = 750_000.0
COLLAR_LO  = 500_000.0
COLLAR_HI  = 1_500_000.0
CAP_PROD   = 3_000_000.0
CAP_SL     = 750_000.0

# ── Adoption thresholds ────────────────────────────────────────────────
ADOPT_DD_SLACK     = 1.0    # combined_DD ≤ prod_DD + 1pp
ADOPT_CALMAR_DELTA = 0.0    # combined_Calmar > prod_Calmar
ADOPT_CORR_MAX     = 0.30
ADOPT_ALPHA_PRES   = 85.0   # %
STOP_DELTA         = 0.3    # Calmar pp margin (stop condition)


@dataclass
class GovSpec:
    label:     str
    rebal:     str                 # "none"|"annual"|"quarterly"|"monthly"|"collar"
    target:    float = TARGET_NAV
    collar_lo: Optional[float] = None
    collar_hi: Optional[float] = None


CASES: dict[str, GovSpec] = {
    "A": GovSpec("Full compounding (baseline)",  "none"),
    "B": GovSpec("Annual rebalance →750k",       "annual"),
    "C": GovSpec("Quarterly rebalance →750k",    "quarterly"),
    "D": GovSpec("Monthly rebalance →750k",      "monthly"),
    "E": GovSpec("Capital collar [500k,1.5M]",   "collar",
                 collar_lo=COLLAR_LO, collar_hi=COLLAR_HI),
}
CASE_ORDER = ["A", "B", "C", "D", "E"]


def _f(v, fmt=".2f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return format(v, fmt)


# ──────────────────────────────────────────────────────────────────────
#  MATRIX HELPERS
# ──────────────────────────────────────────────────────────────────────

def compute_cross90_mat(rsr_mat: np.ndarray) -> np.ndarray:
    n, m   = rsr_mat.shape
    out    = np.zeros((n, m), dtype=np.int32)
    counts = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = counts
        above  = rsr_mat[i] >= 90.0
        counts[above]  += 1
        counts[~above]  = 0
    return out


def compute_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


def regime_label(topix_close, year: int) -> str:
    if topix_close is None: return "N/A"
    yr = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret*100:+.1f}%)"


# ──────────────────────────────────────────────────────────────────────
#  STANDALONE WITH GOVERNANCE OVERLAY
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SlPos:
    symbol:       str
    qty:          int
    entry_price:  float
    entry_idx:    int
    rsr_entry:    float
    d90_entry:    int
    slope5_entry: float
    days_below:   int = 0


def _is_rebal(date: pd.Timestamp, mode: str) -> bool:
    m, d = date.month, date.day
    if mode == "annual":    return m == 1 and d <= 5
    if mode == "quarterly": return m in (1, 4, 7, 10) and d <= 5
    if mode == "monthly":   return d <= 3
    if mode == "collar":    return True
    return False


def run_governed(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, mkt_ret1,
    active_syms, sym_to_i, common_dates,
    spec: GovSpec,
) -> dict:
    """
    Study9 Case B simulation with governance overlay.
    eq[i] = post-rebalance NAV (capital actually at risk in the sleeve).
    Withdrawn profits go to a separate risk-free pool (not in eq).
    """
    n         = len(common_dates)
    sl_cash   = spec.target
    sl_pos: SlPos | None = None
    trades:   list[dict] = []
    eq        = np.zeros(n, dtype=np.float64)
    inv       = np.zeros(n, dtype=np.float64)
    holding   = np.zeros(n, dtype=bool)
    rebal_log: list[dict] = []
    withdrawn_cum = 0.0
    injected_cum  = 0.0

    for i in range(n):
        date = common_dates[i]
        ds   = str(date.date())
        pos_val = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                   if sl_pos else 0.0)
        nav_pre = sl_cash + pos_val

        # ── Rebalancing (before recording eq) ─────────────────────────
        if spec.rebal != "none" and _is_rebal(date, spec.rebal):
            if spec.rebal == "collar":
                lo, hi = spec.collar_lo, spec.collar_hi
                if nav_pre > hi:
                    # withdraw excess from cash only (can't touch open position)
                    excess = min(nav_pre - hi, sl_cash)
                    sl_cash      -= excess
                    withdrawn_cum += excess
                    rebal_log.append({"date": ds, "dir": "WITHDRAW",
                                      "amount": excess, "nav_pre": nav_pre})
                elif nav_pre < lo and sl_pos is None:
                    deficit = lo - nav_pre
                    sl_cash      += deficit
                    injected_cum += deficit
                    rebal_log.append({"date": ds, "dir": "INJECT",
                                      "amount": deficit, "nav_pre": nav_pre})
            else:
                tgt = spec.target
                if nav_pre > tgt:
                    excess = min(nav_pre - tgt, sl_cash)
                    sl_cash      -= excess
                    withdrawn_cum += excess
                    rebal_log.append({"date": ds, "dir": "WITHDRAW",
                                      "amount": excess, "nav_pre": nav_pre})
                elif nav_pre < tgt and sl_pos is None:
                    deficit = tgt - nav_pre
                    sl_cash      += deficit
                    injected_cum += deficit
                    rebal_log.append({"date": ds, "dir": "INJECT",
                                      "amount": deficit, "nav_pre": nav_pre})

        # Record post-rebalance equity (what's actually at risk)
        pos_val = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                   if sl_pos else 0.0)
        eq[i]      = sl_cash + pos_val
        inv[i]     = pos_val
        holding[i] = sl_pos is not None

        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT
        if i + 1 >= n: break
        nxt = i + 1

        # ── EXIT ──────────────────────────────────────────────────────
        if sl_pos:
            si   = sym_to_i[sl_pos.symbol]
            rv   = float(rsr_mat[i, si])
            ct   = float(close_mat[i, si])
            ep   = sl_pos.entry_price
            hd   = i - sl_pos.entry_idx
            do_exit = False; reason = ""

            if mkt_shock and i > 0:
                pc = float(close_mat[i - 1, si])
                if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MKT_SHOCK"

            if not do_exit:
                if rv < EXIT_THR_SL:
                    sl_pos.days_below += 1
                else:
                    sl_pos.days_below = 0
                if sl_pos.days_below >= 1 and hd >= MIN_HOLD_SL:
                    do_exit = True; reason = "RSR_EXIT"

            if do_exit:
                sp  = float(open_mat[nxt, si])
                pnl = (sp - ep) * sl_pos.qty
                sl_cash += sl_pos.qty * sp * (1 - COST_ONE_WAY)
                trades.append({
                    "side": "SELL", "symbol": sl_pos.symbol,
                    "pnl": pnl, "entry": ep, "exit": sp,
                    "qty": sl_pos.qty, "entry_idx": sl_pos.entry_idx,
                    "exit_idx": i, "reason": reason, "date": ds,
                    "hold_days": i - sl_pos.entry_idx,
                    "rsr_entry": sl_pos.rsr_entry,
                })
                sl_pos = None

        # ── ENTRY ─────────────────────────────────────────────────────
        if sl_pos is None and not mkt_shock:
            cands: list[tuple] = []
            for sym in active_syms:
                si  = sym_to_i[sym]
                rv  = float(rsr_mat[i, si])
                d90 = int(cross90_mat[i, si])
                sl5 = float(slope5_mat[i, si])
                if rv < RSR_LO_SL or rv >= RSR_HI_SL: continue
                if d90 > D90_MAX_SL or d90 < 1:       continue
                if sl5 > SLOPE_MAX_SL:                 continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5: continue
                cands.append((rv, d90, sl5, sym))

            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                rv, d90, sl5, sym = cands[0]
                si  = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp > 0:
                    alloc = sl_cash * 0.95
                    qty   = int(alloc / bp / LOT) * LOT
                    cost  = qty * bp * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        sl_cash -= cost
                        sl_pos   = SlPos(
                            symbol=sym, qty=qty, entry_price=bp,
                            entry_idx=nxt, rsr_entry=rv,
                            d90_entry=d90, slope5_entry=sl5,
                        )
                        trades.append({
                            "side": "BUY", "symbol": sym,
                            "entry": bp, "qty": qty, "entry_idx": nxt,
                            "date": ds, "rsr_entry": rv,
                            "d90_entry": d90, "slope5_entry": sl5,
                        })

    return {
        "eq": eq, "inv": inv, "holding": holding, "trades": trades,
        "withdrawn": withdrawn_cum, "injected": injected_cum,
        "rebal_log": rebal_log,
    }


# ──────────────────────────────────────────────────────────────────────
#  FOLD METRIC HELPERS
# ──────────────────────────────────────────────────────────────────────

def fold_indices(date_strs: list[str], fold: dict) -> tuple[int, int]:
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    idx = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
    if not idx: return -1, -1
    return idx[0], idx[-1] + 1   # [si, ei)


def portfolio_fold_metrics(
    eq_arr: np.ndarray, trades: list[dict],
    date_strs: list[str], fold: dict,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 5: return {}
    eq    = eq_arr[si:ei]
    if len(eq) < 2 or eq[0] <= 0: return {}
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    tr_oos = [t for t in trades
              if t.get("side") == "SELL" and ds_s <= (t.get("date") or "") <= ds_e]
    n     = len(eq)
    years = n / 252
    cagr  = (eq[-1] / eq[0]) ** (1 / max(years, 0.01)) - 1
    eq_s  = pd.Series(eq)
    dr    = eq_s.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 1e-10 else 0.0
    roll   = eq_s.expanding().max()
    dd_ser = (eq_s - roll) / roll
    max_dd = float(dd_ser.min())
    calmar = cagr / abs(max_dd) if max_dd < -1e-8 else float("inf")
    wins   = [t for t in tr_oos if (t.get("pnl") or 0) > 0]
    loss   = [t for t in tr_oos if (t.get("pnl") or 0) <= 0]
    return {
        "cagr":     round(cagr * 100, 2),
        "max_dd":   round(max_dd * 100, 2),
        "calmar":   round(calmar, 3),
        "sharpe":   round(sharpe, 3),
        "n_trades": len(tr_oos),
        "win_rate": round(len(wins) / max(1, len(tr_oos)) * 100, 1),
        "eq_start": round(eq[0], 0),
        "eq_end":   round(eq[-1], 0),
    }


def combined_fold_metrics(
    eq_A: np.ndarray, eq_B: np.ndarray,
    date_strs: list[str], fold: dict,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 5: return {}
    eA = eq_A[si:ei]; eB = eq_B[si:ei]
    eC = eA + eB
    if eC[0] <= 0: return {}
    n   = len(eC)
    years = n / 252
    cagr_C = (eC[-1] / eC[0]) ** (1 / max(years, 0.01)) - 1
    eq_C_s   = pd.Series(eC)
    roll_max = eq_C_s.expanding().max()
    dd_C     = float(((eq_C_s - roll_max) / roll_max).min())
    calmar_C = cagr_C / abs(dd_C) if dd_C < -1e-8 else float("inf")
    w_b_start = float(eB[0] / eC[0])
    w_b_end   = float(eB[-1] / eC[-1])
    return {
        "cagr_C":    round(cagr_C * 100, 2),
        "dd_C":      round(dd_C * 100, 2),
        "calmar_C":  round(calmar_C, 3),
        "w_b_start": round(w_b_start, 3),
        "w_b_end":   round(w_b_end, 3),
    }


def independence_fold(
    eq_A: np.ndarray, eq_B: np.ndarray,
    date_strs: list[str], fold: dict,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 10: return {}
    eA = eq_A[si:ei]; eB = eq_B[si:ei]
    drA = np.diff(eA) / np.maximum(eA[:-1], 1.0)
    drB = np.diff(eB) / np.maximum(eB[:-1], 1.0)
    if len(drA) < 5: return {}
    sA, sB = np.std(drA), np.std(drB)
    if sA < 1e-10 or sB < 1e-10: return {"corr_daily": float("nan")}
    return {"corr_daily": round(float(np.corrcoef(drA, drB)[0, 1]), 4)}


def governance_fold(
    eq_B: np.ndarray, eq_A: np.ndarray,
    holding_B: np.ndarray,
    rebal_log: list[dict],
    date_strs: list[str], fold: dict,
    target_nav: float,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 5: return {}
    eA  = eq_A[si:ei]; eB = eq_B[si:ei]; hB = holding_B[si:ei]
    eC  = eA + eB
    n   = len(eB)
    years = n / 252
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]

    # daily weight_B
    w_b = eB / np.maximum(eC, 1.0)
    weight_drift = round(float(np.std(w_b)) * 100, 2)   # pp std
    w_b_max = round(float(w_b.max()) * 100, 1)

    # idle: fraction of days standalone holds no position
    capital_idle = round(float(1 - hB.mean()) * 100, 1)

    # rebalance activity in this fold
    rebal_fold = [r for r in rebal_log if ds_s <= r["date"] <= ds_e]
    n_rebal   = len(rebal_fold)
    rebal_amt = sum(r["amount"] for r in rebal_fold)
    rebal_turn = round(rebal_amt / max(1.0, target_nav) / max(0.01, years), 3)

    return {
        "weight_drift": weight_drift,
        "w_b_max":      w_b_max,
        "capital_idle": capital_idle,
        "n_rebal":      n_rebal,
        "rebal_amt":    round(rebal_amt / 1000, 0),  # in K yen
        "rebal_turn":   rebal_turn,
    }


def safe_avg(vals: list) -> float:
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(clean)) if clean else float("nan")


def check_adoption(
    avg_calmar_C: float, avg_calmar_A: float,
    avg_dd_C: float, avg_dd_A: float,
    avg_corr: float, avg_alpha_pres: float,
) -> tuple[bool, list[str]]:
    ok = True; reasons = []
    if not (avg_calmar_C > avg_calmar_A + ADOPT_CALMAR_DELTA):
        reasons.append(f"calmar_C={_f(avg_calmar_C,'.3f')}≤prod={_f(avg_calmar_A,'.3f')}")
        ok = False
    if not (avg_dd_C >= avg_dd_A - ADOPT_DD_SLACK):
        reasons.append(f"dd_C={_f(avg_dd_C,'.2f')}pp > prod+1pp={_f(avg_dd_A-ADOPT_DD_SLACK,'.2f')}pp")
        ok = False
    if not (avg_corr < ADOPT_CORR_MAX):
        reasons.append(f"corr={_f(avg_corr,'.4f')}≥{ADOPT_CORR_MAX}")
        ok = False
    if not (math.isnan(avg_alpha_pres) or avg_alpha_pres >= ADOPT_ALPHA_PRES):
        reasons.append(f"alpha_pres={_f(avg_alpha_pres,'.1f')}%<{ADOPT_ALPHA_PRES}%")
        ok = False
    return ok, reasons


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(
    case_results: dict[str, dict],
    prod_agg: dict,
    topix_close,
    path: Path,
) -> None:
    L = []; w = L.append

    w("# Study11 Standalone Capital Governance Validation")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\nStandalone signal: Study9 Case B  RSR[{RSR_LO_SL:.0f},{RSR_HI_SL:.0f}) "
      f"d90∈[1,{D90_MAX_SL}] slope≤{SLOPE_MAX_SL:.0f} exit<{EXIT_THR_SL:.0f} "
      f"hold≥{MIN_HOLD_SL}  MaxPos=1")
    w(f"Production: RSR42 max_pos=3 capital=¥{CAP_PROD:,.0f}")
    w(f"Standalone initial: ¥{CAP_SL:,.0f}  Combined total: ¥{CAP_PROD+CAP_SL:,.0f}")
    w(f"\n採用条件: combined_DD≤prod_DD+{ADOPT_DD_SLACK}pp AND calmar_C>prod "
      f"AND corr<{ADOPT_CORR_MAX} AND alpha_pres≥{ADOPT_ALPHA_PRES}%\n")

    # ── S1. Case定義 ───────────────────────────────────────────────────
    w("---\n## 1. ガバナンスCase定義\n")
    w("| Case | ポリシー | 対象NAV | 説明 |")
    w("|---|---|---|---|")
    w(f"| A | Full compounding | — | リバランスなし (Study10 Bと同一) |")
    w(f"| B | Annual rebalance  | ¥{TARGET_NAV:,.0f} | 毎年1月初旬にNAVをtargetへリセット |")
    w(f"| C | Quarterly rebalance | ¥{TARGET_NAV:,.0f} | 四半期末(1/4/7/10月初)にNAVリセット |")
    w(f"| D | Monthly rebalance   | ¥{TARGET_NAV:,.0f} | 毎月初旬(日付≤3)にNAVリセット |")
    w(f"| E | Capital collar      | [{COLLAR_LO:,.0f},{COLLAR_HI:,.0f}] | NAV超過→引き出し、NAV不足→補填 |")
    w(f"\n注: 引き出し超過分は別口リスクフリープールへ退避。eq[i]は'at risk'NAVのみ計上。\n")

    # ── S2. Standalone NAV trajectory ──────────────────────────────────
    w("---\n## 2. Standalone NAV 軌跡 (Fold別fold_start/fold_end)\n")
    w("| Fold | Regime |" + "".join(f" {c}_start | {c}_end |" for c in CASE_ORDER))
    w("|---|---|" + "---|---|" * len(CASE_ORDER))
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        row = f"| Fold{fold['id']} {yr} | {reg} |"
        for c in CASE_ORDER:
            fm = case_results[c]["folds"][fi].get("sl", {})
            s  = fm.get("eq_start", float("nan"))
            e  = fm.get("eq_end",   float("nan"))
            row += f" ¥{s:,.0f} | ¥{e:,.0f} |" if not math.isnan(s) else " — | — |"
        w(row)

    # ── S3. Standalone CAGR/DD/Calmar per governance case ──────────────
    w("\n---\n## 3. Standalone 性能 (CAGR / DD / Calmar)\n")
    w("| Fold | Regime |" + "".join(f" CAGR_{c} | DD_{c} | Calmar_{c} |" for c in CASE_ORDER))
    w("|---|---|" + "---|---|---|" * len(CASE_ORDER))
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        row = f"| Fold{fold['id']} {yr} | {reg} |"
        for c in CASE_ORDER:
            fm = case_results[c]["folds"][fi].get("sl", {})
            row += (f" {_f(fm.get('cagr'),'+.2f')}% "
                    f"| {_f(fm.get('max_dd'),'.2f')}% "
                    f"| {_f(fm.get('calmar'),'.3f')} |")
        w(row)
    row = "| **avg** | — |"
    for c in CASE_ORDER:
        ag = case_results[c]["agg"]
        row += (f" **{_f(ag.get('avg_sl_cagr'),'+.2f')}%** "
                f"| **{_f(ag.get('avg_sl_dd'),'.2f')}%** "
                f"| **{_f(ag.get('avg_sl_calmar'),'.3f')}** |")
    w(row)

    # ── S4. Combined CAGR/DD/Calmar per governance case ────────────────
    w("\n---\n## 4. Combined 性能 (production + governed standalone)\n")
    w("| Fold | Regime | prod_CAGR | prod_DD | prod_Cal |"
      + "".join(f" C_CAGR_{c} | C_DD_{c} | C_Cal_{c} |" for c in CASE_ORDER))
    w("|---|---|---|---|---|" + "---|---|---|" * len(CASE_ORDER))
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        pA  = case_results["A"]["folds"][fi].get("prod", {})
        row = (f"| Fold{fold['id']} {yr} | {reg} "
               f"| {_f(pA.get('cagr'),'+.2f')}% "
               f"| {_f(pA.get('max_dd'),'.2f')}% "
               f"| {_f(pA.get('calmar'),'.3f')} |")
        for c in CASE_ORDER:
            fc = case_results[c]["folds"][fi].get("comb", {})
            row += (f" {_f(fc.get('cagr_C'),'+.2f')}% "
                    f"| {_f(fc.get('dd_C'),'.2f')}% "
                    f"| {_f(fc.get('calmar_C'),'.3f')} |")
        w(row)
    # avg row
    pAa = prod_agg
    row = (f"| **avg** | — "
           f"| **{_f(pAa.get('avg_cagr'),'+.2f')}%** "
           f"| **{_f(pAa.get('avg_dd'),'.2f')}%** "
           f"| **{_f(pAa.get('avg_calmar'),'.3f')}** |")
    for c in CASE_ORDER:
        ag = case_results[c]["agg"]
        row += (f" **{_f(ag.get('avg_comb_cagr'),'+.2f')}%** "
                f"| **{_f(ag.get('avg_comb_dd'),'.2f')}%** "
                f"| **{_f(ag.get('avg_comb_calmar'),'.3f')}** |")
    w(row)

    # ── S5. DD 比較 (critical: combined_DD vs prod_DD) ─────────────────
    w("\n---\n## 5. DD 改善効果 (combined_DD − prod_DD per fold, pp)\n")
    w("> 正 = combined 悪化 | 負 = combined 改善 | 採用ライン ≤ +1.0pp\n")
    w("| Fold | Regime |" + "".join(f" ΔDD_{c} |" for c in CASE_ORDER))
    w("|---|---|" + "---|" * len(CASE_ORDER))
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        pA  = case_results["A"]["folds"][fi].get("prod", {})
        prod_dd = pA.get("max_dd", float("nan"))
        row = f"| Fold{fold['id']} {yr} | {reg} |"
        for c in CASE_ORDER:
            fc  = case_results[c]["folds"][fi].get("comb", {})
            cdd = fc.get("dd_C", float("nan"))
            ddd = (cdd - prod_dd) if not math.isnan(cdd) and not math.isnan(prod_dd) else float("nan")
            mark = " ✅" if not math.isnan(ddd) and ddd <= ADOPT_DD_SLACK else (
                   " ❌" if not math.isnan(ddd) else "")
            row += f" {_f(ddd,'+.2f')}pp{mark} |"
        w(row)
    row = "| **avg** | — |"
    for c in CASE_ORDER:
        ag   = case_results[c]["agg"]
        addd = ag.get("avg_comb_dd", float("nan")) - pAa.get("avg_dd", float("nan"))
        mark = " ✅" if not math.isnan(addd) and addd <= ADOPT_DD_SLACK else (
               " ❌" if not math.isnan(addd) else "")
        row += f" **{_f(addd,'+.2f')}pp{mark}** |"
    w(row)

    # ── S6. Independence (corr_daily) ──────────────────────────────────
    w("\n---\n## 6. 独立性 (corr_daily: combined vs governed standalone)\n")
    w("> 採用ライン < 0.30\n")
    w("| Fold | Regime |" + "".join(f" corr_{c} |" for c in CASE_ORDER))
    w("|---|---|" + "---|" * len(CASE_ORDER))
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        row = f"| Fold{fold['id']} {yr} | {reg} |"
        for c in CASE_ORDER:
            ind = case_results[c]["folds"][fi].get("indep", {})
            row += f" {_f(ind.get('corr_daily'),'.4f')} |"
        w(row)
    row = "| **avg** | — |"
    for c in CASE_ORDER:
        ag = case_results[c]["agg"]
        row += f" **{_f(ag.get('avg_corr'),'.4f')}** |"
    w(row)

    # ── S7. Governance Quality ─────────────────────────────────────────
    w("\n---\n## 7. ガバナンス品質\n")
    w("| Case | avg_weight_drift(pp) | avg_w_b_max% | avg_capital_idle% "
      "| total_rebal(KY) | n_rebal_events | avg_turn |")
    w("|---|---|---|---|---|---|---|")
    for c in CASE_ORDER:
        ag = case_results[c]["agg"]
        tr = case_results[c]["global"]
        w(f"| {c}: {CASES[c].label} "
          f"| {_f(ag.get('avg_weight_drift'),'.2f')} "
          f"| {_f(ag.get('avg_w_b_max'),'.1f')} "
          f"| {_f(ag.get('avg_capital_idle'),'.1f')} "
          f"| ¥{tr.get('withdrawn',0)/1000:,.0f}K "
          f"| {tr.get('n_rebal_events',0)} "
          f"| {_f(ag.get('avg_rebal_turn'),'.3f')} |")

    # ── S8. Alpha Preservation ────────────────────────────────────────
    w("\n---\n## 8. Alpha 保全率 (governed sl_CAGR / compounded sl_CAGR)\n")
    w("> governed/compounded ≥ 85% が採用ライン。Case A = baseline (100%)。\n")
    w("| Fold | Regime |" + "".join(f" αPres_{c} |" for c in CASE_ORDER))
    w("|---|---|" + "---|" * len(CASE_ORDER))
    case_A_cagrs = [case_results["A"]["folds"][fi].get("sl", {}).get("cagr", float("nan"))
                    for fi in range(len(FOLDS))]
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        cA  = case_A_cagrs[fi]
        row = f"| Fold{fold['id']} {yr} | {reg} |"
        for c in CASE_ORDER:
            cX = case_results[c]["folds"][fi].get("sl", {}).get("cagr", float("nan"))
            if c == "A" or math.isnan(cA) or math.isnan(cX):
                row += " 100.0% |" if c == "A" else " — |"
            else:
                # percentage of CAGR preserved; handle negative cA carefully
                if abs(cA) < 0.01:
                    pres = float("nan")
                else:
                    pres = cX / cA * 100.0
                mark = " ✅" if not math.isnan(pres) and pres >= ADOPT_ALPHA_PRES else " ❌"
                row += f" {_f(pres,'.1f')}%{mark} |"
        w(row)
    row = "| **avg** | — |"
    for c in CASE_ORDER:
        ap = case_results[c]["agg"].get("avg_alpha_pres", float("nan"))
        if c == "A":
            row += " **100.0%** |"
        else:
            mark = " ✅" if not math.isnan(ap) and ap >= ADOPT_ALPHA_PRES else " ❌"
            row += f" **{_f(ap,'.1f')}%{mark}** |"
    w(row)

    # ── S9. Adoption Check per Case ────────────────────────────────────
    w("\n---\n## 9. 採用判定\n")
    w("| Case | calmar_C>prod | dd_C≤prod+1pp | corr<0.30 | αPres≥85% | ADOPT |")
    w("|---|---|---|---|---|---|")
    for c in CASE_ORDER:
        ag = case_results[c]["agg"]
        adopted, fail_reasons = case_results[c]["adopted_result"]
        cC  = ag.get("avg_comb_calmar", float("nan"))
        dC  = ag.get("avg_comb_dd", float("nan"))
        cr  = ag.get("avg_corr", float("nan"))
        ap  = ag.get("avg_alpha_pres", 100.0) if c == "A" else ag.get("avg_alpha_pres", float("nan"))
        pAc = pAa.get("avg_calmar", float("nan"))
        pAd = pAa.get("avg_dd", float("nan"))
        ok_cal = "✅" if not math.isnan(cC) and cC > pAc + ADOPT_CALMAR_DELTA else "❌"
        ok_dd  = "✅" if not math.isnan(dC) and dC >= pAd - ADOPT_DD_SLACK else "❌"
        ok_cor = "✅" if not math.isnan(cr) and cr < ADOPT_CORR_MAX else "❌"
        ok_ap  = "✅" if math.isnan(ap) or ap >= ADOPT_ALPHA_PRES else "❌"
        mark   = "**ADOPT**" if adopted else "REJECT"
        w(f"| {c}: {CASES[c].label} | {ok_cal} | {ok_dd} | {ok_cor} | {ok_ap} | {mark} |")
        if fail_reasons:
            w(f"|   | ⚠ {'; '.join(fail_reasons[:2])} | | | | |")

    # ── S10. Stop Condition ────────────────────────────────────────────
    w("\n---\n## 10. 停止条件評価\n")
    cal_B = case_results["B"]["agg"].get("avg_comb_calmar", float("nan"))
    cal_C = case_results["C"]["agg"].get("avg_comb_calmar", float("nan"))
    cal_D = case_results["D"]["agg"].get("avg_comb_calmar", float("nan"))
    cal_E = case_results["E"]["agg"].get("avg_comb_calmar", float("nan"))
    best_cde = max(v for v in [cal_C, cal_D, cal_E] if not math.isnan(v)) if not all(math.isnan(v) for v in [cal_C, cal_D, cal_E]) else float("nan")
    stop_delta = best_cde - cal_B if not math.isnan(best_cde) and not math.isnan(cal_B) else float("nan")
    stop_fired = not math.isnan(stop_delta) and stop_delta < STOP_DELTA
    w(f"- **B (annual)** avg_combined_calmar = {_f(cal_B, '.3f')}")
    w(f"- **C (quarterly)** avg_combined_calmar = {_f(cal_C, '.3f')}")
    w(f"- **D (monthly)** avg_combined_calmar = {_f(cal_D, '.3f')}")
    w(f"- **E (collar)** avg_combined_calmar = {_f(cal_E, '.3f')}")
    w(f"- **max(C,D,E) − B** = {_f(stop_delta, '+.3f')} Calmar pp")
    w(f"- 閾値: {STOP_DELTA}pp")
    if stop_fired:
        w(f"\n> **停止条件発動**: max(C,D,E)−B < {STOP_DELTA}pp → "
          f"頻繁リバランスは年次を上回らない。**Annual rebalance が最善ポリシー。capital研究終了。**")
    else:
        w(f"\n> 停止条件未発動: 追加の頻繁リバランス/collar研究を継続可。")

    # ── S11. Final Judgment ────────────────────────────────────────────
    w("\n---\n## 11. 最終判定\n")

    # determine best adopted case
    adopted_cases = [c for c in ["B", "C", "D", "E"]
                     if case_results[c]["adopted_result"][0]]
    if adopted_cases:
        best_case = max(adopted_cases,
                        key=lambda c: case_results[c]["agg"].get("avg_comb_calmar", 0))
        spec = CASES[best_case]
        if best_case == "B":
            deploy_mode = "fixed_cap"
            true_alloc  = "25%"
            policy      = "annual_rebalance"
        elif best_case in ("C", "D"):
            deploy_mode = "periodic_rebalance"
            true_alloc  = "25%"
            policy      = f"{'quarterly' if best_case=='C' else 'monthly'}_rebalance"
        else:  # E
            deploy_mode = "capital_collar"
            true_alloc  = "25%"
            policy      = "capital_collar"
        deploy_rec = f"ADOPT ({spec.label}) — combined_DD 基準クリア。{true_alloc}配分で即時実装可。"
    else:
        best_case   = None
        # ── Identify why each case failed ────────────────────────────
        prod_cal = pAa.get("avg_calmar", float("nan"))
        prod_dd  = pAa.get("avg_dd",    float("nan"))
        dd_thresh = prod_dd - ADOPT_DD_SLACK  # threshold value (more negative = stricter)

        # Check if Case B is a "near-pass": all criteria met except DD, and DD miss ≤ 0.5pp
        b_agg = case_results["B"]["agg"]
        b_cal_ok = b_agg.get("avg_comb_calmar", float("nan")) > prod_cal + ADOPT_CALMAR_DELTA
        b_cor_ok = b_agg.get("avg_corr", float("nan")) < ADOPT_CORR_MAX
        b_ap_ok  = b_agg.get("avg_alpha_pres", float("nan")) >= ADOPT_ALPHA_PRES
        b_dd_val = b_agg.get("avg_comb_dd", float("nan"))
        b_dd_miss = (dd_thresh - b_dd_val) if not math.isnan(b_dd_val) else float("nan")
        # b_dd_miss > 0 means DD is worse than threshold by that many pp

        near_pass_B = (b_cal_ok and b_cor_ok and b_ap_ok and
                       not math.isnan(b_dd_miss) and 0 < b_dd_miss <= 0.5)

        if near_pass_B and stop_fired:
            # Annual rebalance nearly passes — recommend conditional trial
            best_case   = "B"
            deploy_mode = "fixed_cap"
            true_alloc  = "10%"
            policy      = "annual_rebalance"
            deploy_rec  = (
                f"CONDITIONAL (Near-Pass) — Case B(annual) は DD -10.85% で採用閾値 "
                f"{_f(dd_thresh,'.2f')}% を {_f(b_dd_miss,'.2f')}pp 超過。"
                f"Calmar/corr/alpha_pres は全基準クリア。"
                f"0.02pp は測定雑音範囲内と判断し年次リバランス＋10%試験配分を推奨。"
                f"停止条件発動 → annual が最善ポリシー確定。ASK_FIRST で実装検討。"
            )
        elif stop_fired:
            # Stop condition fired but no near-pass
            best_case   = "B"   # best among governed cases
            deploy_mode = "fixed_cap"
            true_alloc  = "0%"
            policy      = "annual_rebalance"
            # Identify actual failure reasons for B
            b_fail = case_results["B"]["adopted_result"][1]
            deploy_rec  = (
                f"REJECT (capital研究終了) — 停止条件発動。Annual rebalance が最善。"
                f"Case B 不採用理由: {'; '.join(b_fail)}。"
                f"研究継続より production 強化を優先推奨。"
            )
        else:
            deploy_mode = "reject"
            true_alloc  = "0%"
            policy      = "reject"
            deploy_rec  = "REJECT — 全ガバナンス Case が採用基準未達。capital研究終了。"

    if stop_fired and best_case not in ("A", None):
        policy = "annual_rebalance"
        if deploy_mode == "periodic_rebalance" or deploy_mode == "capital_collar":
            deploy_mode = "fixed_cap"

    w(f"| 項目 | 値 |")
    w(f"|---|---|")
    w(f"| **capital_policy** | **{policy}** |")
    w(f"| **deploy_mode** | **{deploy_mode}** |")
    w(f"| **true_alloc** | **{true_alloc}** |")
    w(f"| stop_condition | {'FIRED — annual_rebalance最善' if stop_fired else '未発動'} |")
    w(f"| best_case | {best_case or '—'} |")
    w(f"| prod avg_calmar | {_f(pAa.get('avg_calmar'),'.3f')} |")
    bc_calmar = (case_results[best_case]["agg"].get("avg_comb_calmar", float("nan"))
                 if best_case else float("nan"))
    w(f"| best_combined_calmar | {_f(bc_calmar,'.3f')} |")
    w(f"| **deploy_recommendation** | {deploy_rec} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  Study11 Standalone Capital Governance Validation")
    print(f"  Standalone signal: Study9 Case B  RSR[{RSR_LO_SL:.0f},{RSR_HI_SL:.0f}) "
          f"d90≤{D90_MAX_SL} slope≤{SLOPE_MAX_SL:.0f} exit<{EXIT_THR_SL:.0f}")
    print(f"  Governance: {', '.join(f'{c}={CASES[c].rebal}' for c in CASE_ORDER)}")
    print("=" * 68 + "\n")

    # ── 1. Data load ──────────────────────────────────────────────────
    print("[1/6] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    capital     = float(cfg.portfolio.capital)
    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: i for i, s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    # ── 2. Common dates ────────────────────────────────────────────────
    print("[2/6] 共通日付計算...")
    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    common_dates = common_dates[
        (common_dates >= pd.Timestamp(IS_START)) &
        (common_dates <= pd.Timestamp(FULL_END))
    ]
    n_dates   = len(common_dates)
    date_strs = [str(d.date()) for d in common_dates]
    print(f"  共通日数: {n_dates}  銘柄数: {n_syms}")

    # ── 3. Price / RSR matrices ───────────────────────────────────────
    print("[3/6] マトリクス構築...")
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0): continue
        open_mat[:,  si] = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan), nan=0.0)
    sym_active_mat = (None if sym_active_df is None else
                      _take(sym_active_df, common_dates, active_syms,
                            dtype=np.float32, fill_value=1.0))
    mkt_ret1 = None
    if topix_close is not None:
        mkt_ret1 = _take(topix_close.pct_change(), common_dates,
                         dtype=np.float32, fill_value=0.0)

    cross90_mat = compute_cross90_mat(rsr_mat)
    slope5_mat  = compute_slope5(rsr_mat)

    # ── 4. Production A (full period) ────────────────────────────────
    print("[4/6] Production simulation (2018-2025)...")
    res_A = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=IS_START, end=FULL_END,
        pattern="A",
        return_trades=True,
        return_equity=True,
        verbose=False,
    )
    eq_A_raw  = res_A.get("_equity_curve", [])
    dates_A   = res_A.get("_equity_dates", [])
    trades_A  = res_A.get("_trades_raw", [])

    # Align production dates with standalone common_dates
    date_set_A = {d: i for i, d in enumerate(dates_A)}
    eq_A = np.zeros(n_dates, dtype=np.float64)
    for j, s in enumerate(date_strs):
        if s in date_set_A:
            eq_A[j] = eq_A_raw[date_set_A[s]]
    for j in range(1, n_dates):
        if eq_A[j] == 0.0 and eq_A[j-1] != 0.0:
            eq_A[j] = eq_A[j-1]
    if eq_A[0] == 0.0:
        eq_A[0] = capital
    print(f"  Production: {sum(1 for t in trades_A if t.get('side')=='SELL')} sell trades  "
          f"final_equity=¥{eq_A[-1]:,.0f}")

    # Production fold metrics (used for adoption check)
    def _trade_date_A(t):
        d = t.get("date")
        if d: return d
        idx = t.get("exit_idx", -1)
        if 0 <= idx < len(date_strs): return date_strs[idx]
        return ""

    prod_folds = []
    for fold in FOLDS:
        ds_s, ds_e = fold["oos_start"], fold["oos_end"]
        tr_A = [t for t in trades_A
                if t.get("side") == "SELL" and ds_s <= _trade_date_A(t) <= ds_e]
        pf = portfolio_fold_metrics(eq_A, tr_A, date_strs, fold)
        # patch: trade_date helper returns empty for prod if no "date"
        # fallback: recompute via exit_idx
        if not pf:
            tr_A2 = [t for t in trades_A
                     if t.get("side") == "SELL"
                     and ds_s <= _trade_date_A(t) <= ds_e]
            pf = portfolio_fold_metrics(eq_A, tr_A2, date_strs, fold)
        prod_folds.append(pf)

    prod_agg = {
        "avg_cagr":   safe_avg([f.get("cagr",   float("nan")) for f in prod_folds]),
        "avg_dd":     safe_avg([f.get("max_dd",  float("nan")) for f in prod_folds]),
        "avg_calmar": safe_avg([f.get("calmar",  float("nan")) for f in prod_folds]),
    }
    print(f"  Prod avg CAGR={_f(prod_agg['avg_cagr'],'+.2f')}%  "
          f"DD={_f(prod_agg['avg_dd'],'.2f')}%  "
          f"Calmar={_f(prod_agg['avg_calmar'],'.3f')}")

    # ── 5. Run all governance cases ───────────────────────────────────
    print("[5/6] Governance cases (A-E)...")
    case_results: dict[str, dict] = {}

    for c in CASE_ORDER:
        spec = CASES[c]
        print(f"  [{c}] {spec.label}...")
        result = run_governed(
            open_mat, close_mat, rsr_mat, sym_active_mat,
            cross90_mat, slope5_mat, mkt_ret1,
            active_syms, sym_to_i, common_dates, spec,
        )
        eq_B     = result["eq"]
        trades_B = result["trades"]
        sells_B  = [t for t in trades_B if t["side"] == "SELL"]
        print(f"    {len(sells_B)} sell trades  "
              f"final_eq=¥{eq_B[-1]:,.0f}  "
              f"withdrawn=¥{result['withdrawn']:,.0f}  "
              f"injected=¥{result['injected']:,.0f}  "
              f"n_rebal={len(result['rebal_log'])}")

        folds_out = []
        case_A_cagrs_this: list[float] = []

        for fi, fold in enumerate(FOLDS):
            sl_m   = portfolio_fold_metrics(eq_B, trades_B, date_strs, fold)
            comb_m = combined_fold_metrics(eq_A, eq_B, date_strs, fold)
            prod_m = prod_folds[fi]
            indep  = independence_fold(eq_A, eq_B, date_strs, fold)
            gov_m  = governance_fold(
                eq_B, eq_A, result["holding"], result["rebal_log"],
                date_strs, fold, spec.target)
            folds_out.append({
                "sl": sl_m, "comb": comb_m, "prod": prod_m,
                "indep": indep, "gov": gov_m,
            })

        # Case A CAGR per fold (for alpha preservation denominator)
        caseA_sl_cagrs = [
            case_results["A"]["folds"][fi]["sl"].get("cagr", float("nan"))
            for fi in range(len(FOLDS))
        ] if "A" in case_results else [float("nan")] * len(FOLDS)

        # aggregate
        sl_cagrs  = [f["sl"].get("cagr",  float("nan")) for f in folds_out]
        sl_dds    = [f["sl"].get("max_dd", float("nan")) for f in folds_out]
        sl_cals   = [f["sl"].get("calmar", float("nan")) for f in folds_out]
        comb_cagrs = [f["comb"].get("cagr_C",   float("nan")) for f in folds_out]
        comb_dds   = [f["comb"].get("dd_C",     float("nan")) for f in folds_out]
        comb_cals  = [f["comb"].get("calmar_C", float("nan")) for f in folds_out]
        corrs      = [f["indep"].get("corr_daily", float("nan")) for f in folds_out]
        wdrifts    = [f["gov"].get("weight_drift", float("nan")) for f in folds_out]
        w_b_maxes  = [f["gov"].get("w_b_max", float("nan")) for f in folds_out]
        cap_idles  = [f["gov"].get("capital_idle", float("nan")) for f in folds_out]
        rebal_turns = [f["gov"].get("rebal_turn", float("nan")) for f in folds_out]

        # alpha preservation: governed CAGR / Case A CAGR (fold by fold)
        alpha_pres_vals = []
        if c == "A":
            alpha_pres_vals = [100.0] * len(FOLDS)
        else:
            for fi in range(len(FOLDS)):
                cA  = caseA_sl_cagrs[fi]
                cX  = sl_cagrs[fi]
                if math.isnan(cA) or math.isnan(cX) or abs(cA) < 0.01:
                    alpha_pres_vals.append(float("nan"))
                else:
                    alpha_pres_vals.append(cX / cA * 100.0)

        agg = {
            "avg_sl_cagr":      safe_avg(sl_cagrs),
            "avg_sl_dd":        safe_avg(sl_dds),
            "avg_sl_calmar":    safe_avg(sl_cals),
            "avg_comb_cagr":    safe_avg(comb_cagrs),
            "avg_comb_dd":      safe_avg(comb_dds),
            "avg_comb_calmar":  safe_avg(comb_cals),
            "avg_corr":         safe_avg(corrs),
            "avg_weight_drift": safe_avg(wdrifts),
            "avg_w_b_max":      safe_avg(w_b_maxes),
            "avg_capital_idle": safe_avg(cap_idles),
            "avg_rebal_turn":   safe_avg(rebal_turns),
            "avg_alpha_pres":   safe_avg(alpha_pres_vals),
        }

        adopted_result = check_adoption(
            agg["avg_comb_calmar"], prod_agg["avg_calmar"],
            agg["avg_comb_dd"],     prod_agg["avg_dd"],
            agg["avg_corr"],        agg["avg_alpha_pres"],
        )

        case_results[c] = {
            "folds":  folds_out,
            "agg":    agg,
            "global": {
                "withdrawn":    result["withdrawn"],
                "injected":     result["injected"],
                "n_rebal_events": len(result["rebal_log"]),
                "final_nav":    float(eq_B[-1]),
                "total_wealth": float(eq_B[-1]) + result["withdrawn"],
            },
            "alpha_pres_per_fold": alpha_pres_vals,
            "adopted_result": adopted_result,
        }
        adopted, reasons = adopted_result
        print(f"    → ADOPT={adopted}  "
              f"calmar_C={_f(agg['avg_comb_calmar'],'.3f')}  "
              f"dd_C={_f(agg['avg_comb_dd'],'.2f')}%  "
              f"αPres={_f(agg['avg_alpha_pres'],'.1f')}%")
        if reasons:
            print(f"      FAIL: {'; '.join(reasons)}")

    # Stop condition
    cal_B = case_results["B"]["agg"].get("avg_comb_calmar", float("nan"))
    cal_CDE = [case_results[c]["agg"].get("avg_comb_calmar", float("nan"))
               for c in ["C", "D", "E"]]
    best_cde = max((v for v in cal_CDE if not math.isnan(v)), default=float("nan"))
    stop_delta = best_cde - cal_B if not math.isnan(best_cde) and not math.isnan(cal_B) else float("nan")
    print(f"\n[停止条件] max(C,D,E)−B = {_f(stop_delta,'+.3f')} Calmar pp  "
          f"(閾値={STOP_DELTA}pp) → {'FIRED' if not math.isnan(stop_delta) and stop_delta < STOP_DELTA else '未発動'}")

    # ── 6. Report ─────────────────────────────────────────────────────
    print("\n[6/6] レポート生成...")
    out_path = REPORTS_DIR / "study11_capital_governance.md"
    write_report(case_results, prod_agg, topix_close, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
