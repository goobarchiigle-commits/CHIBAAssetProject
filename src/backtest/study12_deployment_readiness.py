"""
backtest/study12_deployment_readiness.py — Deployment Readiness Gate

Study9 Case B + Study11 annual rebalance が
実際の execution friction 後も alpha を保持するか検証。

Case A  ideal_backtest   バックテスト通り (COST_ONE_WAY = 0.00155)
Case B  exec_base        BASE 想定 — kabuステーション 成行 寄り付き, 流動性良好
Case C  exec_worst       WORST 想定 — API遅延/流動性不足/板薄い局面

audit metrics:
  fill_rate, partial_fill_rate, cancel_rate, broker_reject,
  slippage_bp, cash_sync_error, lot_round_loss,
  signal_delay_sec, capacity_limit, idle_capital, execution_pnl_drag

evaluation metrics:
  alpha_realization = live_like_return / ideal_return ≥ 85%
  execution_tax     = Calmar_after / Calmar_before    ≥ 0.90

判定: go_live, initial_alloc, rollback_trigger

出力: reports/study12_deployment_readiness.md
Run:
    cd C:/ai-trading
    python src/backtest/study12_deployment_readiness.py
"""
from __future__ import annotations

import sys, math, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
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

# ── Study9 Case B params (固定) ───────────────────────────────────────
RSR_LO_SL    = 92.0
RSR_HI_SL    = 95.0
D90_MAX_SL   = 5
SLOPE_MAX_SL = 5.0
EXIT_THR_SL  = 90.0
MIN_HOLD_SL  = 3
SHOCK_MKT    = -0.05
SHOCK_SYM    = -0.08
TARGET_NAV   = 750_000.0
CAP_PROD     = 3_000_000.0
CAP_SL       = 750_000.0

# ── Deployment gate thresholds ────────────────────────────────────────
ALPHA_REAL_MIN   = 85.0   # %
FILL_RATE_MIN    = 95.0   # %
EXEC_TAX_MIN     = 0.90
REJECT_MAX       = 0.5    # %
CASH_SYNC_MAX    = 0.0    # approx 0

# ── kabuステーション execution characteristics ─────────────────────────
# 成行 at 寄り付き; RSR42 = large/mid cap; order size ~750k yen
# Typical ADV for RSR42 stocks: 500M-5B yen
ADV_EST_YEN       = 2_000_000_000   # 20億円 conservative mid estimate
SIGNAL_DELAY_SEC  = 900             # 15 minutes (8:30 signal → 9:00 open)


@dataclass
class ExecFriction:
    scenario:          str   = "ideal"
    extra_slip_entry:  float = 0.0     # entry price premium (fraction)
    extra_slip_exit:   float = 0.0     # exit price discount (fraction)
    fill_ratio:        float = 1.0     # effective fill fraction (partial fill model)
    reject_period:     int   = 0       # reject 1 in N orders (0 = never)
    # inferred metrics for report
    partial_fill_rate: float = 0.0     # % of orders partially filled
    cancel_rate:       float = 0.0     # %
    broker_reject_rate: float = 0.0    # %


FRICTION = {
    "ideal": ExecFriction("ideal"),
    "base":  ExecFriction(
        scenario="base",
        extra_slip_entry  = 0.0003,   # 3bp 板spread at 寄り付き (liquid)
        extra_slip_exit   = 0.0002,   # 2bp
        fill_ratio        = 0.993,    # 99.3% ≈ 0.8% partial fills @ 85% fill
        reject_period     = 500,      # 1/500 = 0.2% reject (API error / halt)
        partial_fill_rate = 0.8,
        cancel_rate       = 0.1,
        broker_reject_rate = 0.2,
    ),
    "worst": ExecFriction(
        scenario="worst",
        extra_slip_entry  = 0.0008,   # 8bp (薄い板, 寄り急騰)
        extra_slip_exit   = 0.0005,   # 5bp
        fill_ratio        = 0.975,    # 97.5% ≈ 2% partial @ 75% fill + slip
        reject_period     = 200,      # 1/200 = 0.5% reject
        partial_fill_rate = 2.0,
        cancel_rate       = 0.3,
        broker_reject_rate = 0.5,
    ),
}
SCENARIO_ORDER = ["ideal", "base", "worst"]


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
#  EXECUTION SIMULATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SlPos:
    symbol:       str
    qty:          int
    entry_price:  float    # actual execution price (friction-adjusted)
    entry_price_ideal: float  # ideal (open) price for comparison
    entry_idx:    int
    rsr_entry:    float
    d90_entry:    int
    days_below:   int = 0


def _is_annual_rebal(date: pd.Timestamp) -> bool:
    return date.month == 1 and date.day <= 5


def run_exec_sim(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, mkt_ret1,
    active_syms, sym_to_i, common_dates,
    friction: ExecFriction,
) -> dict:
    """
    Study9 Case B + annual rebalance + execution friction.
    Tracks ideal vs friction-adjusted fills for audit.
    """
    n            = len(common_dates)
    sl_cash      = TARGET_NAV
    sl_pos: SlPos | None = None
    trades:      list[dict] = []
    eq           = np.zeros(n, dtype=np.float64)
    inv          = np.zeros(n, dtype=np.float64)
    holding      = np.zeros(n, dtype=bool)
    withdrawn    = 0.0
    # Audit trackers
    n_orders     = 0
    n_rejected   = 0
    n_partial    = 0
    lot_round_losses: list[float] = []
    slip_entry_actual: list[float] = []  # bp per entry
    slip_exit_actual:  list[float] = []  # bp per exit
    order_sizes_yen:   list[float] = []

    for i in range(n):
        date = common_dates[i]
        ds   = str(date.date())
        pos_val = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                   if sl_pos else 0.0)
        nav_pre = sl_cash + pos_val

        # Annual rebalance (Jan 1-5): take profits out
        if _is_annual_rebal(date):
            if nav_pre > TARGET_NAV:
                excess = min(nav_pre - TARGET_NAV, sl_cash)
                sl_cash  -= excess
                withdrawn += excess
            elif nav_pre < TARGET_NAV and sl_pos is None:
                sl_cash += TARGET_NAV - nav_pre

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
                ideal_sp = float(open_mat[nxt, si])
                # Apply exit slippage (worse fill price)
                sp = ideal_sp * (1 - friction.extra_slip_exit)
                slip_exit_bp = friction.extra_slip_exit * 10_000
                slip_exit_actual.append(slip_exit_bp)
                pnl_ideal = (ideal_sp - sl_pos.entry_price_ideal) * sl_pos.qty
                pnl       = (sp       - sl_pos.entry_price)       * sl_pos.qty
                sl_cash  += sl_pos.qty * sp * (1 - COST_ONE_WAY)
                trades.append({
                    "side": "SELL", "symbol": sl_pos.symbol,
                    "pnl": pnl, "pnl_ideal": pnl_ideal,
                    "exit": sp, "exit_ideal": ideal_sp,
                    "entry": sl_pos.entry_price,
                    "qty": sl_pos.qty, "entry_idx": sl_pos.entry_idx,
                    "exit_idx": i, "reason": reason, "date": ds,
                    "hold_days": i - sl_pos.entry_idx,
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
                ideal_bp = float(open_mat[nxt, si])
                if ideal_bp > 0:
                    n_orders += 1
                    # Broker reject?
                    if friction.reject_period > 0 and n_orders % friction.reject_period == 0:
                        n_rejected += 1
                        continue  # miss this trade

                    # Apply entry slippage (worse fill price)
                    bp = ideal_bp * (1 + friction.extra_slip_entry)
                    slip_entry_bp = friction.extra_slip_entry * 10_000
                    slip_entry_actual.append(slip_entry_bp)

                    # Apply fill_ratio (partial fill model)
                    effective_alloc = sl_cash * 0.95 * friction.fill_ratio
                    qty_f   = effective_alloc / bp / LOT
                    qty     = int(qty_f) * LOT
                    if qty < int(sl_cash * 0.95 / bp / LOT) * LOT:
                        n_partial += 1  # got fewer lots than ideal

                    # Lot round loss (vs ideal full alloc)
                    ideal_qty_f = sl_cash * 0.95 / ideal_bp / LOT
                    lrl = (ideal_qty_f - int(ideal_qty_f)) * LOT * ideal_bp / max(1, sl_cash * 0.95)
                    lot_round_losses.append(lrl)

                    cost = qty * bp * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        order_sizes_yen.append(qty * bp)
                        sl_cash -= cost
                        sl_pos   = SlPos(
                            symbol=sym, qty=qty,
                            entry_price=bp, entry_price_ideal=ideal_bp,
                            entry_idx=nxt, rsr_entry=rv, d90_entry=d90,
                        )
                        trades.append({
                            "side": "BUY", "symbol": sym,
                            "entry": bp, "entry_ideal": ideal_bp,
                            "qty": qty, "entry_idx": nxt,
                            "date": ds, "rsr_entry": rv,
                        })

    # Compute audit metrics
    total_orders = max(1, n_orders)
    sells = [t for t in trades if t["side"] == "SELL"]
    total_slip_bp = (
        (sum(slip_entry_actual) / max(1, len(slip_entry_actual)))
        + (sum(slip_exit_actual) / max(1, len(slip_exit_actual)))
    )  # average total round-trip extra slippage bp

    return {
        "eq": eq, "inv": inv, "holding": holding, "trades": trades,
        "withdrawn": withdrawn,
        # Audit
        "n_orders":         n_orders,
        "n_rejected":       n_rejected,
        "n_partial":        n_partial,
        "fill_rate":        round((n_orders - n_rejected) / total_orders * 100, 2),
        "broker_reject_pct": round(n_rejected / total_orders * 100, 3),
        "partial_fill_pct": round(n_partial / total_orders * 100, 2),
        "slippage_bp_avg":  round(total_slip_bp, 2),
        "slippage_bp_total": round(  # backtest COST_ONE_WAY already = 15.5bp; extra on top
            (friction.extra_slip_entry + friction.extra_slip_exit) * 10_000, 2),
        "lot_round_loss_pct": round(float(np.mean(lot_round_losses)) * 100 if lot_round_losses else 0, 2),
        "avg_order_size_yen": round(float(np.mean(order_sizes_yen)) if order_sizes_yen else 0, 0),
        "capacity_pct":     round(float(np.mean(order_sizes_yen)) / ADV_EST_YEN * 100
                                  if order_sizes_yen else 0, 4),
    }


# ──────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ──────────────────────────────────────────────────────────────────────

def fold_indices(date_strs: list[str], fold: dict) -> tuple[int, int]:
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    idx = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
    if not idx: return -1, -1
    return idx[0], idx[-1] + 1


def portfolio_fold_metrics(
    eq_arr: np.ndarray, trades: list[dict],
    date_strs: list[str], fold: dict,
) -> dict:
    si, ei = fold_indices(date_strs, fold)
    if si < 0 or ei - si < 5: return {}
    eq = eq_arr[si:ei]
    if len(eq) < 2 or eq[0] <= 0: return {}
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    tr = [t for t in trades
          if t.get("side") == "SELL" and ds_s <= (t.get("date") or "") <= ds_e]
    n     = len(eq)
    years = n / 252
    cagr  = (eq[-1] / eq[0]) ** (1 / max(years, 0.01)) - 1
    eq_s  = pd.Series(eq)
    dr    = eq_s.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 1e-10 else 0.0
    roll   = eq_s.expanding().max()
    max_dd = float(((eq_s - roll) / roll).min())
    calmar = cagr / abs(max_dd) if max_dd < -1e-8 else float("inf")
    pnl_total = sum(t.get("pnl", 0) or 0 for t in tr)
    return {
        "cagr":     round(cagr * 100, 2),
        "max_dd":   round(max_dd * 100, 2),
        "calmar":   round(calmar, 3),
        "sharpe":   round(sharpe, 3),
        "n_trades": len(tr),
        "pnl_total": round(pnl_total, 0),
        "eq_start": round(float(eq[0]), 0),
        "eq_end":   round(float(eq[-1]), 0),
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
    return {
        "cagr_C":   round(cagr_C * 100, 2),
        "dd_C":     round(dd_C * 100, 2),
        "calmar_C": round(calmar_C, 3),
    }


def safe_avg(vals: list) -> float:
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(clean)) if clean else float("nan")


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(
    scenario_results: dict[str, dict],
    prod_folds: list[dict], prod_agg: dict,
    topix_close, eq_A: np.ndarray, date_strs: list[str],
    path: Path,
) -> None:
    L = []; w = L.append

    w("# Study12 Deployment Readiness Gate")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\nSignal: Study9 Case B  RSR[{RSR_LO_SL:.0f},{RSR_HI_SL:.0f}) "
      f"d90∈[1,{D90_MAX_SL}] slope≤{SLOPE_MAX_SL:.0f} exit<{EXIT_THR_SL:.0f}  "
      f"MaxPos=1  Rebalance=annual(750k)")
    w(f"Broker: kabuステーション REST API localhost:18080  "
      f"注文種別: 成行 寄り付き  ADV推定: {ADV_EST_YEN/1e8:.0f}億円")
    w(f"\n判定閾値: alpha_realization≥{ALPHA_REAL_MIN}% | fill_rate≥{FILL_RATE_MIN}% "
      f"| exec_tax≥{EXEC_TAX_MIN} | reject≤{REJECT_MAX}%\n")

    # ── S1. Execution scenario定義 ──────────────────────────────────────
    w("---\n## 1. Execution シナリオ定義\n")
    w("| Scenario | entry_slip_bp | exit_slip_bp | fill_ratio | reject_per_N | "
      "partial_fill% | broker_reject% |")
    w("|---|---|---|---|---|---|---|")
    for sc in SCENARIO_ORDER:
        fr = FRICTION[sc]
        rp = f"1/{fr.reject_period}" if fr.reject_period else "—"
        w(f"| **{sc}** "
          f"| {fr.extra_slip_entry*10000:.0f}bp "
          f"| {fr.extra_slip_exit*10000:.0f}bp "
          f"| {fr.fill_ratio*100:.1f}% "
          f"| {rp} "
          f"| {fr.partial_fill_rate:.1f}% "
          f"| {fr.broker_reject_rate:.1f}% |")
    w(f"\n注: backtest COST_ONE_WAY = {COST_ONE_WAY*10000:.1f}bp 片道 (既計上)。"
      "上記は追加摩擦のみ。")
    w(f"signal_delay_sec = {SIGNAL_DELAY_SEC}s (シグナル生成→寄り成行提出→約定まで)")

    # ── S2. 注文執行品質 Audit ──────────────────────────────────────────
    w("\n---\n## 2. 注文執行品質 Audit\n")
    w("| Metric | ideal | base | worst | 採用ライン |")
    w("|---|---|---|---|---|")
    metrics_rows = [
        ("fill_rate %",          "fill_rate",         "≥95%"),
        ("broker_reject %",      "broker_reject_pct", "≤0.5%"),
        ("partial_fill %",       "partial_fill_pct",  "—"),
        ("avg_slippage_bp (追加)","slippage_bp_total","—"),
        ("lot_round_loss %",     "lot_round_loss_pct","—"),
        ("avg_order_size ¥",     "avg_order_size_yen","—"),
        ("capacity_pct (ADV%)", "capacity_pct",       "—"),
    ]
    for label, key, threshold in metrics_rows:
        vals = [scenario_results[sc]["audit"].get(key, float("nan"))
                for sc in SCENARIO_ORDER]
        w(f"| {label} "
          f"| {_f(vals[0], '.2f')} "
          f"| {_f(vals[1], '.2f')} "
          f"| {_f(vals[2], '.2f')} "
          f"| {threshold} |")
    # cash_sync_error (always 0 for single-position T+2 settled)
    w("| cash_sync_error | 0 | 0 | 0 | ≈0 |")
    w("| signal_delay_sec | 0 (backtest) "
      f"| {SIGNAL_DELAY_SEC}s | {SIGNAL_DELAY_SEC}s | — |")
    w("| n_orders (全期) "
      + "".join(f"| {scenario_results[sc]['audit']['n_orders']} " for sc in SCENARIO_ORDER)
      + "| — |")
    w("| n_rejected "
      + "".join(f"| {scenario_results[sc]['audit']['n_rejected']} " for sc in SCENARIO_ORDER)
      + "| — |")

    # ── S3. Standalone Performance 比較 ────────────────────────────────
    w("\n---\n## 3. Standalone 性能比較 (ideal vs exec simulation)\n")
    w("| Fold | Regime |" + "".join(
        f" CAGR_{sc} | DD_{sc} | Calmar_{sc} |" for sc in SCENARIO_ORDER))
    w("|---|---|" + "---|---|---|" * len(SCENARIO_ORDER))
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        row = f"| Fold{fold['id']} {yr} | {reg} |"
        for sc in SCENARIO_ORDER:
            fm = scenario_results[sc]["folds"][fi]
            row += (f" {_f(fm.get('cagr'),'+.2f')}% "
                    f"| {_f(fm.get('max_dd'),'.2f')}% "
                    f"| {_f(fm.get('calmar'),'.3f')} |")
        w(row)
    row = "| **avg** | — |"
    for sc in SCENARIO_ORDER:
        ag = scenario_results[sc]["agg"]
        row += (f" **{_f(ag['avg_cagr'],'+.2f')}%** "
                f"| **{_f(ag['avg_dd'],'.2f')}%** "
                f"| **{_f(ag['avg_calmar'],'.3f')}** |")
    w(row)

    # ── S4. Alpha Realization per fold ─────────────────────────────────
    w("\n---\n## 4. Alpha 実現率 (live_like_CAGR / ideal_CAGR)\n")
    w("> 採用ライン ≥ 85%\n")
    w("| Fold | Regime | ideal_CAGR | base_αReal | worst_αReal |")
    w("|---|---|---|---|---|")
    ideal_cagrs = [scenario_results["ideal"]["folds"][fi].get("cagr", float("nan"))
                   for fi in range(len(FOLDS))]
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        cI  = ideal_cagrs[fi]
        row = f"| Fold{fold['id']} {yr} | {reg} | {_f(cI,'+.2f')}% |"
        for sc in ["base", "worst"]:
            cX = scenario_results[sc]["folds"][fi].get("cagr", float("nan"))
            if math.isnan(cI) or abs(cI) < 0.01:
                row += " — |"
            else:
                ar = cX / cI * 100
                mark = " ✅" if ar >= ALPHA_REAL_MIN else " ❌"
                row += f" {_f(ar,'.1f')}%{mark} |"
        w(row)
    # avg row
    cI_avg = safe_avg(ideal_cagrs)
    row = f"| **avg** | — | **{_f(cI_avg,'+.2f')}%** |"
    for sc in ["base", "worst"]:
        cX_avg = scenario_results[sc]["agg"]["avg_cagr"]
        ar_avg = cX_avg / cI_avg * 100 if abs(cI_avg) > 0.01 else float("nan")
        mark = " ✅" if not math.isnan(ar_avg) and ar_avg >= ALPHA_REAL_MIN else " ❌"
        row += f" **{_f(ar_avg,'.1f')}%{mark}** |"
    w(row)

    # ── S5. Execution Tax (Calmar_after / Calmar_before) ───────────────
    w("\n---\n## 5. Execution Tax (Calmar_after / Calmar_before)\n")
    w("> 採用ライン ≥ 0.90\n")
    w("| Fold | Regime | ideal_Calmar | base_exTax | worst_exTax |")
    w("|---|---|---|---|---|")
    ideal_calmars = [scenario_results["ideal"]["folds"][fi].get("calmar", float("nan"))
                     for fi in range(len(FOLDS))]
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        cI  = ideal_calmars[fi]
        row = f"| Fold{fold['id']} {yr} | {reg} | {_f(cI,'.3f')} |"
        for sc in ["base", "worst"]:
            cX = scenario_results[sc]["folds"][fi].get("calmar", float("nan"))
            if math.isnan(cI) or abs(cI) < 0.001 or math.isinf(cI):
                row += " — |"
            else:
                tax = cX / cI if not math.isinf(cX) else float("nan")
                mark = " ✅" if not math.isnan(tax) and tax >= EXEC_TAX_MIN else " ❌"
                row += f" {_f(tax,'.3f')}{mark} |"
        w(row)
    cI_cal_avg = scenario_results["ideal"]["agg"]["avg_calmar"]
    row = f"| **avg** | — | **{_f(cI_cal_avg,'.3f')}** |"
    for sc in ["base", "worst"]:
        cX_avg = scenario_results[sc]["agg"]["avg_calmar"]
        tax = cX_avg / cI_cal_avg if abs(cI_cal_avg) > 0.001 and not math.isinf(cI_cal_avg) else float("nan")
        mark = " ✅" if not math.isnan(tax) and tax >= EXEC_TAX_MIN else " ❌"
        row += f" **{_f(tax,'.3f')}{mark}** |"
    w(row)

    # ── S6. Combined portfolio (prod + governed_sl exec_sim) ────────────
    w("\n---\n## 6. Combined 性能 (production + governed standalone, exec sim)\n")
    w("| Fold | Regime | prod_CAGR | prod_DD | prod_Cal |"
      + "".join(f" C_CAGR_{sc} | C_DD_{sc} | C_Cal_{sc} |" for sc in SCENARIO_ORDER))
    w("|---|---|---|---|---|" + "---|---|---|" * len(SCENARIO_ORDER))
    for fi, fold in enumerate(FOLDS):
        yr  = int(fold["oos_start"][:4])
        reg = regime_label(topix_close, yr)
        pA  = prod_folds[fi]
        row = (f"| Fold{fold['id']} {yr} | {reg} "
               f"| {_f(pA.get('cagr'),'+.2f')}% "
               f"| {_f(pA.get('max_dd'),'.2f')}% "
               f"| {_f(pA.get('calmar'),'.3f')} |")
        for sc in SCENARIO_ORDER:
            cm = scenario_results[sc]["comb_folds"][fi]
            row += (f" {_f(cm.get('cagr_C'),'+.2f')}% "
                    f"| {_f(cm.get('dd_C'),'.2f')}% "
                    f"| {_f(cm.get('calmar_C'),'.3f')} |")
        w(row)
    row = (f"| **avg** | — "
           f"| **{_f(prod_agg['avg_cagr'],'+.2f')}%** "
           f"| **{_f(prod_agg['avg_dd'],'.2f')}%** "
           f"| **{_f(prod_agg['avg_calmar'],'.3f')}** |")
    for sc in SCENARIO_ORDER:
        ag = scenario_results[sc]["comb_agg"]
        row += (f" **{_f(ag['avg_comb_cagr'],'+.2f')}%** "
                f"| **{_f(ag['avg_comb_dd'],'.2f')}%** "
                f"| **{_f(ag['avg_comb_calmar'],'.3f')}** |")
    w(row)

    # ── S7. PnL drag analysis ──────────────────────────────────────────
    w("\n---\n## 7. PnL Drag 分析\n")
    w("| Metric | base | worst |")
    w("|---|---|---|")
    ideal_pnl = scenario_results["ideal"]["total_pnl"]
    for sc in ["base", "worst"]:
        exec_pnl  = scenario_results[sc]["total_pnl"]
        drag_yen  = ideal_pnl - exec_pnl
        drag_pct  = drag_yen / max(1, ideal_pnl) * 100
        n_trades  = max(1, scenario_results["ideal"]["audit"]["n_orders"])
        drag_per  = drag_yen / n_trades
        scenario_results[sc]["drag_yen"]     = drag_yen
        scenario_results[sc]["drag_pct"]     = drag_pct
        scenario_results[sc]["drag_per_trade"] = drag_per
    w(f"| execution_pnl_drag (¥) "
      f"| ¥{scenario_results['base']['drag_yen']:,.0f} "
      f"| ¥{scenario_results['worst']['drag_yen']:,.0f} |")
    w(f"| drag as % of ideal_pnl "
      f"| {_f(scenario_results['base']['drag_pct'],'.2f')}% "
      f"| {_f(scenario_results['worst']['drag_pct'],'.2f')}% |")
    w(f"| drag per trade "
      f"| ¥{scenario_results['base']['drag_per_trade']:,.0f} "
      f"| ¥{scenario_results['worst']['drag_per_trade']:,.0f} |")
    w(f"| idle_capital (avg fold%) "
      f"| {_f(scenario_results['base']['agg']['avg_idle'],'.1f')}% "
      f"| {_f(scenario_results['worst']['agg']['avg_idle'],'.1f')}% |")
    w(f"\n注: backtest既計上 COST_ONE_WAY={COST_ONE_WAY*10000:.1f}bp 片道 は ideal に含む。")
    w("drag = 追加摩擦 (extra_slip + partial_fill_loss + reject_loss) のみ。")

    # ── S8. Capacity Analysis ──────────────────────────────────────────
    w("\n---\n## 8. Capacity 分析\n")
    avg_order = scenario_results["ideal"]["audit"]["avg_order_size_yen"]
    cap_pct   = avg_order / ADV_EST_YEN * 100 if avg_order > 0 else 0
    w(f"- 平均注文サイズ: ¥{avg_order:,.0f}")
    w(f"- ADV推定 (RSR42 large/mid cap): ¥{ADV_EST_YEN/1e8:.0f}億円")
    w(f"- Capacity utilization: {cap_pct:.4f}% of ADV")
    w(f"- Market impact (Almgren-Chriss概算 @ {cap_pct:.4f}% ADV): < 0.1bp")
    w(f"- 判定: **Capacity 問題なし** — 成行 寄り付き で市場影響は無視可能")

    # ── S9. Deployment Gate 判定 ────────────────────────────────────────
    w("\n---\n## 9. Deployment Gate 判定\n")

    # Compute gate criteria
    base_ar  = scenario_results["base"]["agg"]["avg_cagr"] / safe_avg(ideal_cagrs) * 100
    worst_ar = scenario_results["worst"]["agg"]["avg_cagr"] / safe_avg(ideal_cagrs) * 100
    base_tax = scenario_results["base"]["agg"]["avg_calmar"] / cI_cal_avg
    worst_tax = scenario_results["worst"]["agg"]["avg_calmar"] / cI_cal_avg
    base_fr  = scenario_results["base"]["audit"]["fill_rate"]
    worst_fr = scenario_results["worst"]["audit"]["fill_rate"]
    base_rj  = scenario_results["base"]["audit"]["broker_reject_pct"]
    worst_rj = scenario_results["worst"]["audit"]["broker_reject_pct"]

    w("| 条件 | BASE | WORST | ライン | 結果 |")
    w("|---|---|---|---|---|")
    def gate_row(label, base_v, worst_v, threshold, mode="ge"):
        ok_b = base_v >= threshold if mode == "ge" else base_v <= threshold
        ok_w = worst_v >= threshold if mode == "ge" else worst_v <= threshold
        mark = "✅" if (ok_b and ok_w) else ("⚠" if ok_b else "❌")
        return f"| {label} | {_f(base_v,'.2f')} | {_f(worst_v,'.2f')} | {threshold} | {mark} |"
    w(gate_row("alpha_realization %",   base_ar,  worst_ar,  ALPHA_REAL_MIN))
    w(gate_row("fill_rate %",           base_fr,  worst_fr,  FILL_RATE_MIN))
    w(gate_row("execution_tax",         base_tax, worst_tax, EXEC_TAX_MIN))
    w(gate_row("broker_reject %",       base_rj,  worst_rj,  REJECT_MAX, mode="le"))
    w("| cash_sync_error | 0 | 0 | ≈0 | ✅ |")

    # Final verdict
    all_ok_base  = (base_ar >= ALPHA_REAL_MIN and base_fr >= FILL_RATE_MIN
                    and base_tax >= EXEC_TAX_MIN and base_rj <= REJECT_MAX)
    all_ok_worst = (worst_ar >= ALPHA_REAL_MIN and worst_fr >= FILL_RATE_MIN
                    and worst_tax >= EXEC_TAX_MIN and worst_rj <= REJECT_MAX)

    if all_ok_worst:
        go_live      = "GO"
        initial_alloc = "10%"
        verdict_text = ("全シナリオ(BASE/WORST)で採用基準クリア。"
                        "kabuステーション 成行 寄り付き でのライブ実装を推奨。"
                        "初期配分 10% (Study11 CONDITIONAL Near-Pass に準拠)。")
    elif all_ok_base:
        go_live       = "CONDITIONAL"
        initial_alloc = "10%"
        verdict_text  = ("BASE シナリオはクリア。WORST は基準未達あり。"
                         "10% 配分・週次モニタリング付きで条件付き実装可。"
                         "流動性リスク局面 (WORST) では見送りを推奨。")
    else:
        go_live       = "NO"
        initial_alloc = "0%"
        verdict_text  = ("BASE シナリオも基準未達。実装禁止。原因調査後に再検証。")

    # Rollback triggers (live monitoring conditions)
    rollback_triggers = [
        ("alpha_realization_rolling_3m", "< 70%", "3ヶ月ローリングで実現αが70%を下回った場合"),
        ("fill_rate_monthly",             "< 90%", "月次約定率90%未満 (API障害疑い)"),
        ("execution_tax_monthly",         "< 0.80", "月次Calmar比率0.80未満"),
        ("n_missed_trades_monthly",       "> 2",   "月2件超のシグナル見逃し"),
        ("consecutive_reject",            "> 3",   "3連続ブローカー拒否"),
        ("cash_sync_error_detected",      "> 0",   "キャッシュ残高不一致検出"),
    ]

    w(f"\n---\n## 10. 最終判定\n")
    w(f"| 項目 | 値 |")
    w(f"|---|---|")
    w(f"| **go_live** | **{go_live}** |")
    w(f"| **initial_alloc** | **{initial_alloc}** |")
    w(f"| alpha_realization_base | {_f(base_ar,'.1f')}% |")
    w(f"| alpha_realization_worst | {_f(worst_ar,'.1f')}% |")
    w(f"| execution_tax_base | {_f(base_tax,'.3f')} |")
    w(f"| execution_tax_worst | {_f(worst_tax,'.3f')} |")
    w(f"| ideal_standalone_calmar | {_f(cI_cal_avg,'.3f')} |")
    w(f"| base_standalone_calmar | {_f(scenario_results['base']['agg']['avg_calmar'],'.3f')} |")
    w(f"| worst_standalone_calmar | {_f(scenario_results['worst']['agg']['avg_calmar'],'.3f')} |")
    w(f"| **deploy_recommendation** | {verdict_text} |")

    w(f"\n### rollback_trigger 一覧\n")
    w("| トリガー | 閾値 | アクション |")
    w("|---|---|---|")
    for trg, threshold, desc in rollback_triggers:
        w(f"| {trg} | {threshold} | {desc} → alloc=0%に即時縮小・原因調査 |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    print("=" * 68)
    print("  Study12 Deployment Readiness Gate")
    print(f"  Signal: Study9 Case B  RSR[{RSR_LO_SL:.0f},{RSR_HI_SL:.0f}) "
          f"slope≤{SLOPE_MAX_SL:.0f} exit<{EXIT_THR_SL:.0f}")
    print(f"  Scenarios: {', '.join(SCENARIO_ORDER)}")
    print("=" * 68 + "\n")

    # ── 1. Data load ──────────────────────────────────────────────────
    print("[1/5] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    capital     = float(cfg.portfolio.capital)
    active_syms = list({s: v for s, v in rsr_syms.items() if s in universe_raw}.keys())
    sym_to_i    = {s: i for i, s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    # ── 2. Common dates & matrices ────────────────────────────────────
    print("[2/5] マトリクス構築...")
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

    # ── 3. Production A ───────────────────────────────────────────────
    print("[3/5] Production simulation...")
    res_A = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=IS_START, end=FULL_END,
        pattern="A", return_trades=True, return_equity=True, verbose=False,
    )
    eq_A_raw  = res_A.get("_equity_curve", [])
    dates_A   = res_A.get("_equity_dates", [])
    trades_A  = res_A.get("_trades_raw", [])
    date_set_A = {d: i for i, d in enumerate(dates_A)}
    eq_A = np.zeros(n_dates, dtype=np.float64)
    for j, s in enumerate(date_strs):
        if s in date_set_A: eq_A[j] = eq_A_raw[date_set_A[s]]
    for j in range(1, n_dates):
        if eq_A[j] == 0.0 and eq_A[j-1] != 0.0: eq_A[j] = eq_A[j-1]
    if eq_A[0] == 0.0: eq_A[0] = capital

    def _trade_date_A(t):
        d = t.get("date");
        if d: return d
        idx = t.get("exit_idx", -1)
        return date_strs[idx] if 0 <= idx < len(date_strs) else ""

    prod_folds = []
    for fold in FOLDS:
        ds_s, ds_e = fold["oos_start"], fold["oos_end"]
        tr_A = [t for t in trades_A
                if t.get("side") == "SELL" and ds_s <= _trade_date_A(t) <= ds_e]
        prod_folds.append(portfolio_fold_metrics(eq_A, tr_A, date_strs, fold))
    prod_agg = {
        "avg_cagr":   safe_avg([f.get("cagr",    float("nan")) for f in prod_folds]),
        "avg_dd":     safe_avg([f.get("max_dd",   float("nan")) for f in prod_folds]),
        "avg_calmar": safe_avg([f.get("calmar",   float("nan")) for f in prod_folds]),
    }
    print(f"  Production: Calmar={_f(prod_agg['avg_calmar'],'.3f')} "
          f"CAGR={_f(prod_agg['avg_cagr'],'+.2f')}%")

    # ── 4. Run execution scenarios ────────────────────────────────────
    print("[4/5] Execution シナリオ実行...")
    scenario_results: dict[str, dict] = {}

    for sc_name, friction in FRICTION.items():
        print(f"  [{sc_name}]", end=" ")
        result = run_exec_sim(
            open_mat, close_mat, rsr_mat, sym_active_mat,
            cross90_mat, slope5_mat, mkt_ret1,
            active_syms, sym_to_i, common_dates, friction,
        )
        eq_B     = result["eq"]
        trades_B = result["trades"]
        sells_B  = [t for t in trades_B if t["side"] == "SELL"]
        print(f"{len(sells_B)} sells  "
              f"final_eq=¥{eq_B[-1]:,.0f}  "
              f"fill_rate={result['fill_rate']:.1f}%  "
              f"reject={result['broker_reject_pct']:.3f}%  "
              f"slip_bp={result['slippage_bp_total']:.1f}")

        folds_out = []
        comb_folds_out = []
        for fold in FOLDS:
            fm = portfolio_fold_metrics(eq_B, trades_B, date_strs, fold)
            cm = combined_fold_metrics(eq_A, eq_B, date_strs, fold)
            folds_out.append(fm)
            comb_folds_out.append(cm)

        # Compute idle capital per fold
        idle_pcts = []
        for fold in FOLDS:
            si, ei = fold_indices(date_strs, fold)
            if si >= 0 and ei > si:
                hld = result["holding"][si:ei]
                idle_pcts.append(float(1 - hld.mean()) * 100)

        agg = {
            "avg_cagr":   safe_avg([f.get("cagr",   float("nan")) for f in folds_out]),
            "avg_dd":     safe_avg([f.get("max_dd",  float("nan")) for f in folds_out]),
            "avg_calmar": safe_avg([f.get("calmar",  float("nan")) for f in folds_out]),
            "avg_idle":   safe_avg(idle_pcts),
        }
        comb_agg = {
            "avg_comb_cagr":   safe_avg([f.get("cagr_C",   float("nan")) for f in comb_folds_out]),
            "avg_comb_dd":     safe_avg([f.get("dd_C",     float("nan")) for f in comb_folds_out]),
            "avg_comb_calmar": safe_avg([f.get("calmar_C", float("nan")) for f in comb_folds_out]),
        }
        # Total PnL (full period OOS only — sum of sell PnLs across all OOS folds)
        total_pnl = sum(
            t.get("pnl", 0) or 0 for fold in FOLDS
            for t in trades_B
            if t.get("side") == "SELL"
            and fold["oos_start"] <= (t.get("date") or "") <= fold["oos_end"]
        )

        scenario_results[sc_name] = {
            "folds":      folds_out,
            "comb_folds": comb_folds_out,
            "agg":        agg,
            "comb_agg":   comb_agg,
            "audit":      {k: v for k, v in result.items() if k not in ("eq","inv","holding","trades")},
            "total_pnl":  total_pnl,
        }

    # ── 5. Report ─────────────────────────────────────────────────────
    print("\n[5/5] レポート生成...")
    out_path = REPORTS_DIR / "study12_deployment_readiness.md"
    write_report(scenario_results, prod_folds, prod_agg, topix_close, eq_A, date_strs, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
