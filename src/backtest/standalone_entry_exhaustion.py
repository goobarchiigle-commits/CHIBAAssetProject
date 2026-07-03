"""
backtest/standalone_entry_exhaustion.py  —  Standalone Entry Exhaustion WF

仮説: RSR95到達直後は残存alphaより過熱リスクが大きい

固定:
  EXIT  = RSR<90 (即時・単日)
  EXIT consec = 1 (min_hold=3 は維持)

比較:
  A: RSR [92,95)  d90<=5               (baseline)
  B: RSR [92,94]  d90<=5
  C: RSR [92,94]  d90<=3
  D: RSR [92,94]  d90<=5  + 過去5日RSR上昇<=+6
  E: RSR [92,94]  + 3日累積ret<=15%
  F: RSR [92,94]  + ATR_zscore<1.5

追加監査:
  entry_decay        hold日別 avg return
  entry_exhaustion   entry時 RSR>=93 の割合
  post_entry_fwd10   entry後 10日 forward return
  post_entry_fwd20   entry後 20日 forward return
  95_touch_rate      hold中 RSR>=95 到達率

採用: WF>=4/5, Calmar改善(vs Case A), fold CAGR 分散縮小

出力: reports/standalone_entry_exhaustion.md
最後に出力: late_momentum_penalty

Run:
    cd C:/ai-trading
    python src/backtest/standalone_entry_exhaustion.py
"""
from __future__ import annotations

import sys, math, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, _take, calc_metrics,
    LOT, COST_ONE_WAY,
)

REPORTS_DIR = Path("reports")
IS_START    = "2018-01-01"

FOLDS = [
    {"id": 1, "oos_start": "2021-01-01", "oos_end": "2021-12-31"},
    {"id": 2, "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    {"id": 3, "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    {"id": 4, "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    {"id": 5, "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
]

# ── Fixed exit params ──────────────────────────────────────────────────
EXIT_THR     = 90.0   # RSR<90 → exit
MIN_HOLD     = 3      # min hold days before exit
SLEEVE_FRAC  = 0.25   # sleeve capital = capital * 0.25
SHOCK_MKT    = -0.05
SHOCK_SYM    = -0.08

# ── ATR window ─────────────────────────────────────────────────────────
ATR_PERIOD   = 14
ATR_ZSCORE_W = 252    # z-score lookback window

# ── Adoption ──────────────────────────────────────────────────────────
ADOPT_WF     = 4


# ──────────────────────────────────────────────────────────────────────
#  ENTRY SPEC
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EntrySpec:
    label:           str
    rsr_lo:          float         # inclusive
    rsr_hi:          float         # see hi_excl
    hi_excl:         bool          # True = strict upper (< hi), False = inclusive (<= hi)
    d90_max:         int
    slope5_max:      float | None  # RSR change 5d; None = no constraint
    ret3d_max:       float | None  # 3d cumulative ret (fraction); None = no constraint
    atr_zscore_max:  float | None  # ATR z-score upper bound; None = no constraint


CASES: dict[str, EntrySpec] = {
    "A": EntrySpec("RSR[92,95) d90≤5",                       92, 95, True,  5, None, None, None),
    "B": EntrySpec("RSR[92,94] d90≤5",                       92, 94, False, 5, None, None, None),
    "C": EntrySpec("RSR[92,94] d90≤3",                       92, 94, False, 3, None, None, None),
    "D": EntrySpec("RSR[92,94] d90≤5 slope5≤+6",             92, 94, False, 5,  6.0, None, None),
    "E": EntrySpec("RSR[92,94] d90≤5 ret3d≤15%",             92, 94, False, 5, None, 0.15, None),
    "F": EntrySpec("RSR[92,94] d90≤5 ATR_z<1.5",             92, 94, False, 5, None, None,  1.5),
}


def entry_ok(spec: EntrySpec, rv: float, d90: int,
             slope5: float, ret3d: float, atr_z: float) -> bool:
    if rv < spec.rsr_lo:                        return False
    if spec.hi_excl and rv >= spec.rsr_hi:      return False
    if not spec.hi_excl and rv > spec.rsr_hi:   return False
    if d90 > spec.d90_max:                       return False
    if spec.slope5_max is not None and slope5 > spec.slope5_max: return False
    if spec.ret3d_max  is not None and ret3d  > spec.ret3d_max:  return False
    if spec.atr_zscore_max is not None and atr_z >= spec.atr_zscore_max: return False
    return True


# ──────────────────────────────────────────────────────────────────────
#  PRE-COMPUTE MATRICES
# ──────────────────────────────────────────────────────────────────────

def compute_cross_mat(rsr_mat: np.ndarray, threshold: float) -> np.ndarray:
    n_dates, n_syms = rsr_mat.shape
    out = np.zeros((n_dates, n_syms), dtype=np.int32)
    running = np.zeros(n_syms, dtype=np.int32)
    for i in range(n_dates):
        out[i] = running
        above = rsr_mat[i] >= threshold
        running[above] += 1
        running[~above] = 0
    return out


def compute_rsr_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


def compute_ret3d(close_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(close_mat, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(close_mat[:-3] > 0, close_mat[3:] / close_mat[:-3] - 1, 0.0)
    out[3:] = r.astype(np.float32)
    return out


def compute_atr_zscore(high_mat, low_mat, close_mat) -> np.ndarray:
    n, m = close_mat.shape
    atr_mat = np.zeros((n, m), dtype=np.float32)
    for i in range(1, n):
        tr = np.maximum(
            high_mat[i] - low_mat[i],
            np.maximum(
                np.abs(high_mat[i] - close_mat[i - 1]),
                np.abs(low_mat[i]  - close_mat[i - 1]),
            )
        )
        atr_mat[i] = tr
    # rolling ATR (simple avg)
    atr_smooth = np.zeros_like(atr_mat)
    for i in range(ATR_PERIOD, n):
        atr_smooth[i] = atr_mat[i - ATR_PERIOD + 1:i + 1].mean(axis=0)

    # z-score of ATR vs rolling window
    z_mat = np.zeros_like(atr_smooth)
    for i in range(ATR_ZSCORE_W, n):
        window = atr_smooth[i - ATR_ZSCORE_W:i]
        mu  = window.mean(axis=0)
        sig = window.std(axis=0)
        sig = np.where(sig < 1e-8, 1e-8, sig)
        z_mat[i] = (atr_smooth[i] - mu) / sig
    return z_mat


# ──────────────────────────────────────────────────────────────────────
#  STANDALONE SIMULATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SlPos:
    symbol:      str
    qty:         int
    entry_price: float
    entry_idx:   int
    rsr_entry:   float
    d90_entry:   int
    days_below:  int = 0


def run_standalone(
    open_mat, close_mat, rsr_mat, sym_active_mat,
    cross90_mat, slope5_mat, ret3d_mat, atr_z_mat,
    mkt_ret1,
    active_syms, sym_to_i,
    common_dates,
    capital: float,
    spec: EntrySpec,
) -> dict:
    n         = len(common_dates)
    sl_cash   = capital * SLEEVE_FRAC
    sl_pos: SlPos | None = None
    trades: list[dict]   = []
    eq  = np.zeros(n, dtype=np.float64)
    inv = np.zeros(n, dtype=np.float64)

    for i, date in enumerate(common_dates):
        ds    = str(date.date())
        s_inv = (sl_pos.qty * float(close_mat[i, sym_to_i[sl_pos.symbol]])
                 if sl_pos else 0.0)
        s_eq  = sl_cash + s_inv
        eq[i]  = s_eq
        inv[i] = s_inv

        mkt_shock = mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT

        if i + 1 >= n: break
        nxt = i + 1

        # ── EXIT ──────────────────────────────────────────────────────
        if sl_pos:
            si   = sym_to_i[sl_pos.symbol]
            rv   = float(rsr_mat[i, si])
            ct   = float(close_mat[i, si])
            hd   = i - sl_pos.entry_idx
            do_exit = False; reason = ""

            if mkt_shock and i > 0:
                pc = float(close_mat[i - 1, si])
                if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MARKET_SHOCK"

            if not do_exit:
                if rv < EXIT_THR:
                    sl_pos.days_below += 1
                else:
                    sl_pos.days_below = 0
                if sl_pos.days_below >= 1 and hd >= MIN_HOLD:
                    do_exit = True; reason = "RSR_EXIT_90"

            if do_exit:
                sp  = float(open_mat[nxt, si])
                pnl = (sp - sl_pos.entry_price) * sl_pos.qty
                hd_f = i - sl_pos.entry_idx
                sl_cash += sl_pos.qty * sp * (1 - COST_ONE_WAY)
                trades.append({
                    "side": "SELL", "symbol": sl_pos.symbol,
                    "pnl": pnl, "entry": sl_pos.entry_price, "exit": sp,
                    "qty": sl_pos.qty, "entry_idx": sl_pos.entry_idx,
                    "exit_idx": i, "reason": reason, "date": ds,
                    "hold_days": hd_f, "rsr_entry": sl_pos.rsr_entry,
                    "d90_entry": sl_pos.d90_entry,
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
                r3  = float(ret3d_mat[i, si])
                az  = float(atr_z_mat[i, si])
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
                    continue
                if entry_ok(spec, rv, d90, sl5, r3, az):
                    cands.append((rv, d90, sym))

            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))  # high RSR, low d90
                rv, d90, sym = cands[0]
                si  = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp > 0:
                    alloc = sl_cash * 0.95   # keep 5% buffer
                    qty   = int(alloc / bp / LOT) * LOT
                    cost  = qty * bp * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        sl_cash -= cost
                        sl_pos = SlPos(
                            symbol=sym, qty=qty, entry_price=bp,
                            entry_idx=nxt, rsr_entry=rv, d90_entry=d90,
                        )
                        trades.append({
                            "side": "BUY", "symbol": sym,
                            "entry": bp, "exit": None, "pnl": None,
                            "qty": qty, "entry_idx": nxt, "exit_idx": None,
                            "reason": f"RSR={rv:.1f} d90={d90}", "date": ds,
                            "hold_days": None, "rsr_entry": rv, "d90_entry": d90,
                        })

    return {"eq": eq, "inv": inv, "trades": trades}


# ──────────────────────────────────────────────────────────────────────
#  POST-HOC ANALYTICS (no lookahead in simulation; lookforward here)
# ──────────────────────────────────────────────────────────────────────

def compute_exhaustion_analytics(
    trades: list[dict],
    rsr_mat: np.ndarray,
    close_mat: np.ndarray,
    sym_to_i: dict,
    n_dates: int,
) -> dict:
    sells = [t for t in trades if t["side"] == "SELL"]
    if not sells:
        return {
            "post_entry_fwd10": float("nan"),
            "post_entry_fwd20": float("nan"),
            "95_touch_rate":    0.0,
            "entry_exhaustion": 0.0,
            "entry_decay":      {},
        }

    fwd10, fwd20 = [], []
    touch95      = []
    exhausted    = []
    decay_by_day: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}

    for t in sells:
        sym  = t["symbol"]
        ei   = t["entry_idx"]
        xi   = t["exit_idx"]
        si   = sym_to_i.get(sym)
        if si is None or ei is None or xi is None: continue

        ep   = t["entry"]
        rv_e = t["rsr_entry"]

        # forward returns from entry (post-hoc; analytics only)
        for fwd, bucket in [(10, fwd10), (20, fwd20)]:
            fi = ei + fwd
            if fi < n_dates and ep > 0:
                fp = float(close_mat[fi, si])
                if fp > 0:
                    bucket.append((fp / ep - 1) * 100)

        # did RSR touch >=95 during hold?
        if ei < n_dates and xi < n_dates:
            rsr_hold = rsr_mat[ei:xi + 1, si]
            touch95.append(bool(np.any(rsr_hold >= 95.0)))

        # entry exhaustion: RSR >= 93 at entry
        exhausted.append(rv_e >= 93.0)

        # hold-day decay
        for d in [1, 3, 5, 10]:
            fi = ei + d
            if fi < n_dates and ep > 0:
                fp = float(close_mat[fi, si])
                if fp > 0:
                    decay_by_day[d].append((fp / ep - 1) * 100)

    decay_avg = {f"d{d}": (round(float(np.mean(v)), 2) if v else float("nan"))
                 for d, v in decay_by_day.items()}

    return {
        "post_entry_fwd10": round(float(np.mean(fwd10)), 2) if fwd10 else float("nan"),
        "post_entry_fwd20": round(float(np.mean(fwd20)), 2) if fwd20 else float("nan"),
        "95_touch_rate":    round(float(np.mean(touch95)) * 100, 1) if touch95 else 0.0,
        "entry_exhaustion": round(float(np.mean(exhausted)) * 100, 1) if exhausted else 0.0,
        "entry_decay":      decay_avg,
    }


# ──────────────────────────────────────────────────────────────────────
#  FOLD METRICS
# ──────────────────────────────────────────────────────────────────────

def compute_fold_metrics(
    result: dict,
    common_dates,
    fold: dict,
    capital: float,
    calmar_A: float,
) -> dict:
    ds_s, ds_e = fold["oos_start"], fold["oos_end"]
    date_strs   = [str(d.date()) for d in common_dates]
    oos_idx     = [i for i, s in enumerate(date_strs) if ds_s <= s <= ds_e]
    if not oos_idx: return {}

    si, ei    = oos_idx[0], oos_idx[-1] + 1
    oos_dates = list(common_dates[si:ei])
    n_days    = len(oos_dates)
    n_years   = n_days / 252.0

    s_eq  = result["eq"][si:ei].tolist()
    s_inv = result["inv"][si:ei].tolist()
    if len(s_eq) < 5: return {}

    trades_oos = [t for t in result["trades"]
                  if ds_s <= (t.get("date") or "") <= ds_e]

    exp_list = [vi / max(1.0, ve) for vi, ve in zip(s_inv, s_eq)]
    m = calc_metrics(s_eq, trades_oos, exp_list, s_eq[0], oos_dates)

    sells = [t for t in trades_oos if t["side"] == "SELL"]
    n_exits  = len(sells)
    trig_yr  = n_exits / max(0.01, n_years)
    hds      = [t["hold_days"] for t in sells if (t.get("hold_days") or 0) > 0]
    avg_hold = float(np.mean(hds)) if hds else 0.0
    cap_util = sum(1 for v in s_inv if v > 0) / max(1, n_days) * 100

    calmar  = m.get("calmar", 0.0)
    fold_pass = calmar >= calmar_A

    return {
        "cagr":      m.get("cagr", 0.0),
        "max_dd":    m.get("max_dd", 0.0),
        "calmar":    m.get("calmar", 0.0),
        "pf":        m.get("profit_factor", 0.0),
        "hit_rate":  m.get("win_rate", 0.0),
        "sharpe":    m.get("sharpe", 0.0),
        "n_trades":  n_exits,
        "trig_yr":   round(trig_yr, 1),
        "avg_hold":  round(avg_hold, 1),
        "cap_util":  round(cap_util, 1),
        "fold_pass": fold_pass,
        "calmar_A":  calmar_A,
    }


def aggregate_wf(fold_results: list[dict]) -> dict:
    valid = [f for f in fold_results if f]
    if not valid: return {}
    n_pass = sum(1 for f in valid if f.get("fold_pass"))

    def avg(k): return float(np.mean([f[k] for f in valid]))

    calmar_vals = [f["calmar"] for f in valid]

    adopted = (
        n_pass >= ADOPT_WF
        and avg("calmar") > avg("calmar_A")
    )

    return {
        "n_pass":      n_pass,
        "avg_cagr":    round(avg("cagr"), 2),
        "avg_dd":      round(avg("max_dd"), 2),
        "avg_calmar":  round(avg("calmar"), 3),
        "calmar_std":  round(float(np.std(calmar_vals)), 3),
        "avg_pf":      round(avg("pf"), 3),
        "avg_hit":     round(avg("hit_rate"), 1),
        "avg_trig":    round(avg("trig_yr"), 1),
        "avg_hold":    round(avg("avg_hold"), 1),
        "avg_util":    round(avg("cap_util"), 1),
        "avg_calmar_A":round(avg("calmar_A"), 3),
        "calmar_lift": round(avg("calmar") - avg("calmar_A"), 3),
        "adopted":     adopted,
    }


# ──────────────────────────────────────────────────────────────────────
#  LATE MOMENTUM PENALTY
# ──────────────────────────────────────────────────────────────────────

def compute_late_momentum_penalty(
    trades_A: list[dict],
    close_mat: np.ndarray,
    sym_to_i: dict,
    n_dates: int,
) -> dict:
    """
    RSR at entry を early [92,93) / late [93,95) に分割し
    fwd20 return の差を late_momentum_penalty として計算。
    penalty > 0 → late entry は early entry より fwd return が低い
    """
    sells = [t for t in trades_A if t["side"] == "SELL"]
    early_fwd, late_fwd = [], []

    for t in sells:
        sym  = t["symbol"]
        si   = sym_to_i.get(sym)
        ei   = t.get("entry_idx")
        ep   = t.get("entry", 0)
        rv_e = t.get("rsr_entry", 0)
        if si is None or ei is None or ep <= 0: continue
        fi = ei + 20
        if fi >= n_dates: continue
        fp = float(close_mat[fi, si])
        if fp <= 0: continue
        ret20 = (fp / ep - 1) * 100
        if rv_e < 93.0:
            early_fwd.append(ret20)
        else:
            late_fwd.append(ret20)

    avg_early = float(np.mean(early_fwd)) if early_fwd else float("nan")
    avg_late  = float(np.mean(late_fwd))  if late_fwd  else float("nan")

    if math.isnan(avg_early) or math.isnan(avg_late):
        penalty = float("nan")
    else:
        penalty = avg_early - avg_late   # positive = late entry is worse

    return {
        "avg_fwd20_early_rsr": round(avg_early, 2),
        "avg_fwd20_late_rsr":  round(avg_late, 2),
        "late_momentum_penalty": round(penalty, 2) if not math.isnan(penalty) else float("nan"),
        "n_early": len(early_fwd),
        "n_late":  len(late_fwd),
    }


# ──────────────────────────────────────────────────────────────────────
#  REGIME LABEL
# ──────────────────────────────────────────────────────────────────────

def regime_label(topix_close, year: int) -> str:
    if topix_close is None: return "N/A"
    yr  = topix_close[topix_close.index.year == year]
    if len(yr) < 2: return "N/A"
    ret = float(yr.iloc[-1] / yr.iloc[0] - 1)
    return f"{'Bull' if ret > 0 else 'Bear'} ({ret * 100:+.1f}%)"


# ──────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────

def write_report(
    case_results: dict,
    topix_close,
    penalty_info: dict,
    path: Path,
) -> None:
    L = []; w = L.append

    adopted = [c for c in CASES if case_results[c]["wf"].get("adopted")]
    verdict = "ADOPT" if adopted else "REJECT"

    def pct(v):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
        return f"{v:+.2f}%"

    def fp(v, fmt=".3f"):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
        return format(v, fmt)

    w(f"# Standalone Entry Exhaustion WF")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  研究専用 / 実装変更禁止")
    w(f"\n仮説: RSR95到達直後は残存alphaより過熱リスクが大きい")
    w(f"\n固定: EXIT=RSR<90(即時)  MIN_HOLD={MIN_HOLD}d  SLEEVE={SLEEVE_FRAC*100:.0f}%\n")
    w(f"採用条件: WF>={ADOPT_WF}/5  Calmar改善(vs Case A)  fold CAGR 分散縮小\n")
    w(f"**採用: {len(adopted)}件**  最終判定: **{verdict}**\n")

    # ── S1. Executive Summary ─────────────────────────────────────────
    w("---\n## 1. Executive Summary\n")
    wf_A = case_results["A"]["wf"]
    w(f"- ベースライン (Case A): CAGR={wf_A.get('avg_cagr',0):+.2f}%  "
      f"DD={wf_A.get('avg_dd',0):.2f}%  Calmar={wf_A.get('avg_calmar',0):.3f}  "
      f"WF={wf_A.get('n_pass',0)}/5")
    pen = penalty_info.get("late_momentum_penalty", float("nan"))
    pen_s = f"{pen:+.2f}pp" if not math.isnan(pen) else "—"
    w(f"- late_momentum_penalty: **{pen_s}**  "
      f"(early_fwd20={pct(penalty_info.get('avg_fwd20_early_rsr'))}  "
      f"late_fwd20={pct(penalty_info.get('avg_fwd20_late_rsr'))}  "
      f"n_early={penalty_info.get('n_early',0)}  n_late={penalty_info.get('n_late',0)})")
    w(f"- 採用: **{len(adopted)}件**{(' — ' + ', '.join(adopted)) if adopted else ''}\n")

    # ── S2. WF Summary ────────────────────────────────────────────────
    w("---\n## 2. WF サマリ (6 Cases)\n")
    w("| Case | Filter | WF | CAGR | DD | Calmar | Calmar_lift | PF | hit% | trig/yr | util% | 採用 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c, spec in CASES.items():
        wf   = case_results[c]["wf"]
        mark = "**ADOPT**" if wf.get("adopted") else "REJECT"
        w(f"| {c} | {spec.label} "
          f"| {wf.get('n_pass',0)}/5 "
          f"| {wf.get('avg_cagr',0):+.2f}% "
          f"| {wf.get('avg_dd',0):.2f}% "
          f"| {fp(wf.get('avg_calmar'))} "
          f"| {fp(wf.get('calmar_lift'),'+.3f')} "
          f"| {fp(wf.get('avg_pf'),'.3f')} "
          f"| {fp(wf.get('avg_hit'),'.1f')} "
          f"| {fp(wf.get('avg_trig'),'.1f')} "
          f"| {fp(wf.get('avg_util'),'.1f')} "
          f"| {mark} |")

    # ── S3. Fold Detail ───────────────────────────────────────────────
    w("\n---\n## 3. Fold 詳細\n")
    w("| Case | Fold | Regime | CAGR | DD | Calmar | Calmar_A | PF | hit% | avg_hold | trig/yr | pass |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CASES:
        for fold, fm in zip(FOLDS, case_results[c]["folds"]):
            if not fm: continue
            yr  = int(fold["oos_start"][:4])
            reg = regime_label(topix_close, yr)
            ok  = "✅" if fm.get("fold_pass") else "❌"
            w(f"| {c} | Fold{fold['id']} | {reg} "
              f"| {fm['cagr']:+.2f}% "
              f"| {fm['max_dd']:.2f}% "
              f"| {fp(fm['calmar'])} "
              f"| {fp(fm['calmar_A'])} "
              f"| {fp(fm['pf'],'.3f')} "
              f"| {fm['hit_rate']:.1f}% "
              f"| {fm['avg_hold']:.1f}d "
              f"| {fm['trig_yr']:.1f} "
              f"| {ok} |")

    # ── S4. Exhaustion Analytics ──────────────────────────────────────
    w("\n---\n## 4. Entry Exhaustion Analytics\n")
    w("| Case | fwd10 | fwd20 | 95_touch% | exhaustion% | d1 | d3 | d5 | d10 | n_trades |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for c in CASES:
        ea = case_results[c]["exhaustion"]
        dc = ea.get("entry_decay", {})
        sells = [t for t in case_results[c]["trades"] if t["side"] == "SELL"]
        w(f"| {c} "
          f"| {pct(ea.get('post_entry_fwd10'))} "
          f"| {pct(ea.get('post_entry_fwd20'))} "
          f"| {ea.get('95_touch_rate',0):.1f}% "
          f"| {ea.get('entry_exhaustion',0):.1f}% "
          f"| {pct(dc.get('d1'))} "
          f"| {pct(dc.get('d3'))} "
          f"| {pct(dc.get('d5'))} "
          f"| {pct(dc.get('d10'))} "
          f"| {len(sells)} |")

    # ── S5. Calmar Dispersion ─────────────────────────────────────────
    w("\n---\n## 5. Calmar 分散 (fold間)\n")
    w("| Case | avg_Calmar | std_Calmar | Calmar_lift | fold分散改善 |")
    w("|---|---|---|---|---|")
    std_A = case_results["A"]["wf"].get("calmar_std", 9999)
    for c in CASES:
        wf  = case_results[c]["wf"]
        std = wf.get("calmar_std", 0)
        imp = "✅" if std < std_A else "❌"
        w(f"| {c} "
          f"| {fp(wf.get('avg_calmar'))} "
          f"| {fp(std)} "
          f"| {fp(wf.get('calmar_lift'),'+.3f')} "
          f"| {imp} |")

    # ── S6. Late Momentum Penalty ─────────────────────────────────────
    w("\n---\n## 6. Late Momentum Penalty\n")
    w("> Case A エントリーを RSR帯で early[92,93) / late[93,95) に分割したfwd20差分\n")
    pen = penalty_info.get("late_momentum_penalty", float("nan"))
    pen_s = f"{pen:+.2f}pp" if not math.isnan(pen) else "—"
    w(f"| metric | 値 |")
    w(f"|---|---|")
    w(f"| avg_fwd20 early(RSR<93)  | {pct(penalty_info.get('avg_fwd20_early_rsr'))} |")
    w(f"| avg_fwd20 late(RSR>=93)  | {pct(penalty_info.get('avg_fwd20_late_rsr'))} |")
    w(f"| **late_momentum_penalty**| **{pen_s}** |")
    w(f"| n_early                  | {penalty_info.get('n_early',0)} |")
    w(f"| n_late                   | {penalty_info.get('n_late',0)} |")
    if not math.isnan(pen):
        if pen > 2.0:
            w("\n→ **仮説支持**: late entry は early entry より fwd20 が有意に低い\n")
        elif pen > 0:
            w("\n→ **弱支持**: late entry の penalty 軽微 (≤2pp)\n")
        else:
            w("\n→ **仮説棄却**: late entry で penalty なし\n")

    # ── S7. Failure Analysis ──────────────────────────────────────────
    w("\n---\n## 7. Failure Analysis\n")
    for c, spec in CASES.items():
        wf = case_results[c]["wf"]
        calmar_A_avg = wf.get("avg_calmar_A", 0)
        fails = []
        if wf.get("n_pass", 0) < ADOPT_WF:
            fails.append(f"WF={wf['n_pass']}/5 < {ADOPT_WF}")
        if wf.get("avg_calmar", 0) <= calmar_A_avg:
            fails.append(f"Calmar={wf.get('avg_calmar',0):.3f} ≤ A={calmar_A_avg:.3f}")
        if not fails:
            w(f"\n**{c}** ({spec.label}): ✅ 採用")
        else:
            w(f"\n**{c}** ({spec.label}): REJECT — " + " / ".join(fails))

    # ── S8. Final Recommendation ──────────────────────────────────────
    w("\n---\n## 8. Final Recommendation\n")
    w(f"### **{verdict}**\n")

    if adopted:
        best = max(adopted, key=lambda c: case_results[c]["wf"].get("avg_calmar", 0))
        bwf  = case_results[best]["wf"]
        bsp  = CASES[best]
        w(f"**採用推奨: Case {best}** — {bsp.label}\n")
        w(f"- CAGR: {bwf['avg_cagr']:+.2f}%  DD: {bwf['avg_dd']:.2f}%  Calmar: {bwf.get('avg_calmar',0):.3f}")
        w(f"- Calmar_lift: {bwf.get('calmar_lift',0):+.3f}  WF: {bwf['n_pass']}/5")
        w(f"- trig: {bwf['avg_trig']:.1f}/yr  avg_hold: {bwf['avg_hold']:.1f}d  util: {bwf['avg_util']:.1f}%\n")
        ea_best = case_results[best]["exhaustion"]
        w(f"- post_entry_fwd20: {pct(ea_best.get('post_entry_fwd20'))}")
        w(f"- 95_touch_rate: {ea_best.get('95_touch_rate',0):.1f}%")
        w(f"- entry_exhaustion: {ea_best.get('entry_exhaustion',0):.1f}%\n")
        w(f"**late_momentum_penalty: {pen_s}**\n")
        w("実装優先度 (ASK_FIRST 必須):")
        w("1. signal_bridge.py エントリー条件追加")
        w("2. live dry-run 30日以上 → 本番移行")
    else:
        best = max(CASES, key=lambda c: case_results[c]["wf"].get("avg_calmar", 0))
        bwf  = case_results[best]["wf"]
        bsp  = CASES[best]
        w(f"全6 Case 採用基準未達。最良: **Case {best}** ({bsp.label})\n")
        w(f"- CAGR={bwf['avg_cagr']:+.2f}%  DD={bwf['avg_dd']:.2f}%  "
          f"Calmar={bwf.get('avg_calmar',0):.3f}  WF={bwf['n_pass']}/5\n")

        fails = []
        if bwf.get("n_pass", 0) < ADOPT_WF:
            fails.append(f"WF={bwf['n_pass']}/5")
        if bwf.get("avg_calmar", 0) <= bwf.get("avg_calmar_A", 0):
            fails.append(f"Calmar≤A baseline")
        w("**バインディング制約**: " + " / ".join(fails) + "\n")
        w("**構造的発見:**")
        w(f"- late_momentum_penalty: {pen_s}")
        ea_A = case_results["A"]["exhaustion"]
        w(f"- Case A entry_exhaustion={ea_A.get('entry_exhaustion',0):.1f}%  "
          f"95_touch_rate={ea_A.get('95_touch_rate',0):.1f}%")
        w(f"- post_entry_fwd20 (Case A): {pct(ea_A.get('post_entry_fwd20'))}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  レポート: {path}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3

    print("=" * 68)
    print("  Standalone Entry Exhaustion WF")
    print(f"  EXIT=RSR<{EXIT_THR:.0f}(即時)  MIN_HOLD={MIN_HOLD}d  SLEEVE={SLEEVE_FRAC*100:.0f}%")
    print("=" * 68 + "\n")

    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: i for i, s in enumerate(active_syms)}
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
    print(f"  共通日数: {n_dates}, 銘柄数: {n_syms}")

    print("[2/4] マトリクス構築...")
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    high_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    low_mat   = np.full((n_dates, n_syms), np.nan, dtype=np.float32)

    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0): continue
        open_mat[:,  si] = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]
        if "High" in df_src.columns:
            high_mat[:, si] = df_src["High"].to_numpy(dtype=np.float32)[row_idx]
        if "Low" in df_src.columns:
            low_mat[:, si]  = df_src["Low"].to_numpy(dtype=np.float32)[row_idx]

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0)
    sym_active_mat = (None if sym_active_df is None else
                      _take(sym_active_df, common_dates, active_syms,
                             dtype=np.float32, fill_value=1.0))

    mkt_ret1 = None
    if topix_close is not None:
        mkt_ret1 = _take(topix_close.pct_change(), common_dates,
                          dtype=np.float32, fill_value=0.0)

    print("  補助マトリクス計算中...")
    cross90_mat = compute_cross_mat(rsr_mat, 90.0)
    slope5_mat  = compute_rsr_slope5(rsr_mat)
    ret3d_mat   = compute_ret3d(close_mat)
    # Use high/low=close where unavailable
    h_mat = np.where(np.isnan(high_mat), close_mat, high_mat)
    l_mat = np.where(np.isnan(low_mat),  close_mat, low_mat)
    atr_z_mat   = compute_atr_zscore(h_mat, l_mat, close_mat)

    capital = float(cfg.portfolio.capital)

    print("[3/4] Case シミュレーション (6 cases)...\n")

    case_results: dict = {}
    calmar_A_per_fold: list[float] = []

    # ── Pass 1: Case A to get per-fold calmar baseline ─────────────
    spec_A = CASES["A"]
    print(f"  [A pass-1] {spec_A.label}")
    res_A = run_standalone(
        open_mat, close_mat, rsr_mat, sym_active_mat,
        cross90_mat, slope5_mat, ret3d_mat, atr_z_mat,
        mkt_ret1, active_syms, sym_to_i, common_dates, capital, spec_A,
    )
    # collect per-fold calmar for A (use 0.0 as placeholder on first pass)
    for fold in FOLDS:
        fm = compute_fold_metrics(res_A, common_dates, fold, capital, calmar_A=0.0)
        calmar_A_per_fold.append(fm.get("calmar", 0.0) if fm else 0.0)

    # ── All cases with correct baseline ────────────────────────────
    for c, spec in CASES.items():
        print(f"  [{c}] {spec.label}")
        if c == "A":
            res = res_A
        else:
            res = run_standalone(
                open_mat, close_mat, rsr_mat, sym_active_mat,
                cross90_mat, slope5_mat, ret3d_mat, atr_z_mat,
                mkt_ret1, active_syms, sym_to_i, common_dates, capital, spec,
            )

        folds_m = []
        for fi, fold in enumerate(FOLDS):
            fm = compute_fold_metrics(
                res, common_dates, fold, capital,
                calmar_A=calmar_A_per_fold[fi],
            )
            folds_m.append(fm)

        wf = aggregate_wf(folds_m)
        ea = compute_exhaustion_analytics(
            res["trades"], rsr_mat, close_mat, sym_to_i, n_dates)

        case_results[c] = {
            "folds":      folds_m,
            "wf":         wf,
            "exhaustion": ea,
            "trades":     res["trades"],
        }

        for fold, fm in zip(FOLDS, folds_m):
            if fm:
                ok = "✅" if fm.get("fold_pass") else "❌"
                print(f"    Fold{fold['id']} {fold['oos_start'][:4]}: "
                      f"CAGR={fm['cagr']:+.2f}%  DD={fm['max_dd']:.2f}%  "
                      f"Calmar={fm['calmar']:.3f}  hit={fm['hit_rate']:.1f}%  {ok}")
        ok_s = "✅ ADOPTED" if wf.get("adopted") else "❌ REJECT"
        print(f"    → WF={wf.get('n_pass',0)}/5  CAGR={wf.get('avg_cagr',0):+.2f}%  "
              f"Calmar={wf.get('avg_calmar',0):.3f}  "
              f"lift={wf.get('calmar_lift',0):+.3f}  {ok_s}\n")

    # ── Late momentum penalty ──────────────────────────────────────
    print("[4/4] Late Momentum Penalty 計算...")
    penalty_info = compute_late_momentum_penalty(
        case_results["A"]["trades"], close_mat, sym_to_i, n_dates)

    print(f"\n  avg_fwd20 early (RSR<93):  {penalty_info['avg_fwd20_early_rsr']:+.2f}%  "
          f"n={penalty_info['n_early']}")
    print(f"  avg_fwd20 late  (RSR>=93): {penalty_info['avg_fwd20_late_rsr']:+.2f}%  "
          f"n={penalty_info['n_late']}")
    pen = penalty_info.get("late_momentum_penalty", float("nan"))
    pen_s = f"{pen:+.2f}pp" if not math.isnan(pen) else "—"
    print(f"\n  ★ late_momentum_penalty = {pen_s}\n")

    adopted = [c for c in CASES if case_results[c]["wf"].get("adopted")]
    print("=" * 68)
    print(f"  採用: {len(adopted)}件  |  {', '.join(adopted) if adopted else 'なし'}")
    for c in CASES:
        wf = case_results[c]["wf"]
        ok = "✅" if wf.get("adopted") else "❌"
        print(f"  {ok} Case {c}: WF={wf.get('n_pass',0)}/5  "
              f"CAGR={wf.get('avg_cagr',0):+.2f}%  "
              f"Calmar={wf.get('avg_calmar',0):.3f}  "
              f"lift={wf.get('calmar_lift',0):+.3f}")
    print("=" * 68 + "\n")

    write_report(
        case_results, topix_close, penalty_info,
        REPORTS_DIR / "standalone_entry_exhaustion.md",
    )

    print(f"\nlate_momentum_penalty = {pen_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
