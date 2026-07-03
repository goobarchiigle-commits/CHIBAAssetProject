"""
src/backtest/study27_access_preserving_risk_activation_audit.py
Study27 — Access-Preserving Risk Activation Audit

Objective: with Entry/Exit/Signal/Execution/Authority/Capital FROZEN (Study9 Case B,
¥1,800,000, max_pos=1), determine whether Calmar improvement room remains by changing
ONLY the *risk-activation firing condition* (when a new entry is admitted), NOT position
size and NOT the candidate-selection rule. This is the explicit pivot away from Study25's
finding: size-shrink overlays destroy lot access on a single-slot book
(PORTFOLIO_GEOMETRY_EXHAUSTED, best Calmar_delta=+0.088<0.10, Case C/D/F destroyed by
lot-rounding cascade). Study27 asks: instead of changing HOW MUCH to size, can changing
WHEN to admit a new entry preserve alpha + lot access while still cutting drawdown?

Research lineage: Study14(lot feasibility) -> Study15B(access bottleneck) ->
Study19(activation optimum) -> Study25(size reduction destroys access) -> Study27(this).

Fixed configuration (FROZEN, reused verbatim from Study9 Case B / Study20 / Study24 / Study25):
  Strategy : Study9 Case B  (RSR[92,95), days_cross90<=5, slope5<=5, exit RSR<90)
  Capital  : ¥1,800,000   (efficient_capital ¥1.5M per Study15B)
  max_pos  : 1 (single slot)
  Exit     : verbatim copy of Study20/24/25's exit block -- UNCHANGED
  Entry candidate selection rule (top RSR among RSR[92,95) d90<=5 slope5<=5): UNCHANGED

Change scope: risk_activation_only -- i.e. WHEN a new entry is admitted. Sizing of an
admitted entry is either UNCHANGED (Cases D/E/F: mult=1.0 always, "size change forbidden")
or is the exact Study25 Case E reproduction (Cases B/C, included only as a sizing-baseline
reference to contrast against the activation-only cases).

Cases:
  A  Baseline                  : no change (mult=1.0, never blocked) -- identical to Study25 Case A
  B  Volatility-aware Shrink   : Study25 Case E reproduction. New entries only:
                                  size = base * clip(VOL_REF/realized_vol_20d, VOL_MULT_MIN, 1.0).
                                  Included as the known sizing-overlay reference point.
  C  Volatility-aware Shrink
     + Activation Floor        : same shrink as B, but if the shrunk size would round down
                                  to 0 lots, force a minimum-1-LOT entry instead of skipping
                                  it (when affordable) -- "lot_access优先". Tests whether
                                  guaranteeing lot access repairs Study25's destructive
                                  lot-rounding cascade for the sizing approach.
  D  Drawdown Throttle         : NO size change (mult=1.0 always). While rolling_dd at
                                  decision time exceeds DD_THROTTLE_THRESHOLD (=0.05, reused
                                  from Study25's DD_DECAY_START -- not refit), a new entry is
                                  only admitted after an extra DD_THROTTLE_EXTRA_DAYS
                                  (=MIN_HOLD=3, PARAMS_LOCKED min_hold -- structurally tied,
                                  not fit) beyond the normal next-day re-entry gap, measured
                                  from the most recent exit. Pure entry-TIMING intervention.
  E  Heat Budget                : NO size change (mult=1.0 always). Portfolio "heat" =
                                  rolling 20-day stdev of the equity curve's own daily
                                  returns (HEAT_VOL_LOOKBACK=20, reused from Study25's
                                  VOL_LOOKBACK; HEAT_VOL_THRESHOLD=0.025, reused from
                                  Study25's VOL_REF -- both structural reuses, not refit).
                                  While heat > threshold, new entries are suppressed
                                  entirely; size of an admitted entry is never touched.
                                  Distinct from D: heat is a portfolio-level realized-vol
                                  signal, not a peak-to-trough drawdown signal.
  F  Combined Activation        : D's throttle OR E's heat (whichever fires) gates
                                  admission; size reduction is explicitly FORBIDDEN
                                  (mult=1.0 always) per task spec ("サイズ縮小は禁止" /
                                  "lot access优先") -- by construction lot access for F is
                                  identical to baseline whenever an entry is admitted at all.

All constants are single fixed values chosen once from structural reasoning (tied to
existing PARAMS_LOCKED/Study25 values) -- no sweep, no fitting to this run's outcome
(OVERFIT_GUARD: param_sweep_limit=bounded, single_metric_optimization=forbid).

Prohibited (per task spec): Entry change, Exit change, RSR threshold change,
days_cross90 change, slope change, Signal change, Execution change, Authority change,
Capital change, Production code change. No improvement is implemented in production --
this script only measures whether improvement room exists.

Adoption thresholds (per task spec):
  meets_adoption (per case)   : alpha_retention>=98% AND lot_access>=95% AND
                                 dd_reduction_pp>=2.0pp AND calmar_delta>=0.10
  DEPLOY_CANDIDATE (strict)   : alpha_retention>=98% AND lot_access>=98% AND
                                 calmar_delta>=0.10 AND dd_reduction_pp>=2.0pp
  PROMISING_BUT_INSUFFICIENT : calmar_delta>0 AND alpha_retention>=95%
  EXHAUSTED                  : best_calmar_delta<0.10 OR access_loss_ratio(best)>2%

Run:
    cd C:/ai-trading
    python src/backtest/study27_access_preserving_risk_activation_audit.py
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from src.backtest.capital_allocation_abc import load_data, _take, calc_metrics, LOT
from src.backtest.study20_limited_live_risk_envelope import (
    _cross90, _slope5,
    RSR_LO, RSR_HI, D90_MIN, D90_MAX, SLOPE_MAX, EXIT_THR, MIN_HOLD,
    SHOCK_MKT, SHOCK_SYM, SLIPPAGE, COMMISSION, COST, R_UNIT_PCT,
)
from src.config_loader import load_strategy_config

OBS_START = "2018-01-01"
OBS_END   = "2025-12-31"
CAPITAL   = 1_800_000.0   # FROZEN, Study9 Case B / Study20-25 config

# ── Risk Activation constants (THIS STUDY ONLY -- audit, not production) ──────────
VOL_LOOKBACK         = 20     # Case B/C: trailing daily-return stdev window (days) -- Study25 reuse
VOL_REF              = 0.025  # Case B/C: fixed reference daily vol (2.5%) -- Study25 reuse
VOL_MULT_MIN         = 0.50   # Case B/C: floor on size multiplier -- Study25 reuse
DD_THROTTLE_THRESHOLD = 0.05  # Case D/F: rolling_dd above this triggers extra wait -- Study25 DD_DECAY_START reuse
DD_THROTTLE_EXTRA_DAYS = MIN_HOLD   # Case D/F: extra days beyond normal 1-day gap -- PARAMS_LOCKED min_hold reuse
HEAT_VOL_LOOKBACK    = 20     # Case E/F: rolling stdev window on equity-curve returns -- Study25 VOL_LOOKBACK reuse
HEAT_VOL_THRESHOLD   = 0.025  # Case E/F: portfolio heat threshold -- Study25 VOL_REF reuse

REPORT_DIR    = Path("reports")
REPORT_MD     = REPORT_DIR / "study27_access_preserving_risk_activation_audit.md"
TRADE_CSV     = REPORT_DIR / "study27_case_trades.csv"
LOG_DIR       = Path("logs")
TELEMETRY_LOG = LOG_DIR / "study27_telemetry.jsonl"

CASES = ["A", "B", "C", "D", "E", "F"]
CASE_LABELS = {
    "A": "Baseline (現行)",
    "B": "Volatility-aware Shrink (Study25 Case E再現)",
    "C": "Volatility-aware Shrink + Activation Floor (最小1lot保証)",
    "D": "Drawdown Throttle (サイズ不変、DD上昇時のEntry間隔延長)",
    "E": "Heat Budget (サイズ不変、リスク過熱時のEntry抑制)",
    "F": "Combined Activation (D∨E、サイズ縮小禁止)",
}
# decision_complexity: number of new free parameters introduced relative to baseline
DECISION_COMPLEXITY = {"A": 0, "B": 1, "C": 2, "D": 1, "E": 1, "F": 2}

ADOPT_ALPHA_RET_MIN = 0.98
ADOPT_LOT_ACCESS_MIN = 0.95
ADOPT_CALMAR_DELTA_MIN = 0.10
ADOPT_DD_REDUCTION_MIN_PP = 2.0

DEPLOY_LOT_ACCESS_MIN = 0.98
PROMISING_ALPHA_RET_MIN = 0.95
EXHAUSTED_CALMAR_DELTA = 0.10
EXHAUSTED_ACCESS_LOSS_MAX = 0.02


@dataclass
class CaseMetrics:
    case: str
    label: str
    trade_count: int = 0
    cagr: float = 0.0
    calmar: float = 0.0
    mar: float = 0.0
    max_dd: float = 0.0
    sharpe: float = 0.0
    alpha_retention: float = 1.0
    lot_access: float = 1.0
    activation_ratio: float = 1.0
    capital_activation_ratio: float = 0.0
    capital_efficiency: float = 1.0
    slot_utilization: float = 0.0
    cash_idle_ratio: float = 0.0
    p95_trade_loss: float = 0.0
    p95_r: float = 0.0
    recovery_days: float = 0.0
    recovery_speed: float = 0.0
    dd_reduction_pp: float = 0.0
    calmar_delta: float = 0.0
    return_preservation: float = 1.0
    access_loss_ratio: float = 0.0
    activation_loss_ratio: float = 0.0
    opportunity_loss_R: float = 0.0
    tail_capture_change: float = 0.0
    signal_preservation: float = 1.0
    decision_complexity: int = 0
    risk_activation_frequency: float = 0.0
    activation_trigger_effectiveness: float = 0.0
    n_dd_blocked: int = 0
    n_heat_blocked: int = 0
    n_lot_infeasible: int = 0
    n_floor_rescued: int = 0
    meets_adoption: bool = False


class _Pos:
    __slots__ = ("symbol", "qty", "entry_price", "entry_idx", "db", "entry_decision_date",
                 "size_mult_at_entry")

    def __init__(self, symbol, qty, entry_price, entry_idx):
        self.symbol = symbol; self.qty = qty
        self.entry_price = entry_price; self.entry_idx = entry_idx
        self.db = 0
        self.entry_decision_date = ""
        self.size_mult_at_entry = 1.0


def _vol_mult(close_mat: np.ndarray, si: int, i: int) -> float:
    lo = max(0, i - VOL_LOOKBACK)
    path = close_mat[lo:i + 1, si]
    path = path[~np.isnan(path)]
    if len(path) < 5:
        return 1.0
    rets = np.diff(path) / path[:-1]
    vol = float(np.std(rets))
    if vol <= 0:
        return 1.0
    return float(np.clip(VOL_REF / vol, VOL_MULT_MIN, 1.0))


def _heat_vol(eq: np.ndarray, i: int) -> float:
    lo = max(0, i - HEAT_VOL_LOOKBACK)
    path = eq[lo:i + 1]
    if len(path) < 5:
        return 0.0
    denom = np.where(path[:-1] != 0, path[:-1], 1.0)
    rets = np.diff(path) / denom
    return float(np.std(rets))


def simulate_case(
    case: str,
    common_dates: pd.DatetimeIndex,
    active_syms: List[str],
    sym_to_i: Dict[str, int],
    open_mat: np.ndarray,
    close_mat: np.ndarray,
    rsr_mat: np.ndarray,
    sym_active_mat: Optional[np.ndarray],
    cross90_mat: np.ndarray,
    slope5_mat: np.ndarray,
    mkt_ret1: Optional[np.ndarray],
    capital: float,
) -> dict:
    n_dates = len(common_dates)
    cash = capital
    pos: Optional[_Pos] = None
    trades: List[dict] = []
    eq  = np.zeros(n_dates, dtype=np.float64)
    inv = np.zeros(n_dates, dtype=np.float64)

    last_exit_idx = -10_000   # decision-day index of most recent exit (never blocks first entry)

    n_trigger_days   = 0   # days a valid top candidate existed while flat
    n_dd_blocked      = 0   # Case D/F: admission denied by drawdown-throttle gap requirement
    n_heat_blocked    = 0   # Case E/F: admission denied by portfolio heat budget
    n_lot_infeasible  = 0   # admitted day, but resulting size rounded to 0 lots (entry skipped)
    n_floor_rescued   = 0   # Case C: floor rule rescued an otherwise lot_infeasible day
    blocked_dates: List[str] = []          # admission-denied dates (D/E/F only)

    for i, date in enumerate(common_dates):
        ds = str(date.date())
        s_inv = (pos.qty * float(close_mat[i, sym_to_i[pos.symbol]]) if pos else 0.0)
        eq[i]  = cash + s_inv
        inv[i] = s_inv
        if i + 1 >= n_dates:
            break
        nxt = i + 1
        mkt_shock = (mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT)

        # ── EXIT (verbatim copy of Study20/24/25's exit block -- Exit UNCHANGED) ──
        if pos is not None:
            si  = sym_to_i[pos.symbol]
            rv  = float(rsr_mat[i, si])
            ct  = float(close_mat[i, si])
            hd  = i - pos.entry_idx
            do_exit = False; reason = ""
            if mkt_shock and i > 0:
                pc = float(close_mat[i - 1, si])
                if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                    do_exit = True; reason = "MKT_SHOCK"
            if not do_exit:
                pos.db = (pos.db + 1) if rv < EXIT_THR else 0
                if pos.db >= 1 and hd >= MIN_HOLD:
                    do_exit = True; reason = "RSR_EXIT"
            if do_exit:
                sp  = float(open_mat[nxt, si])
                entry_fill = pos.entry_price * (1.0 + SLIPPAGE)
                exit_fill  = sp * (1.0 - SLIPPAGE)
                cost_basis = entry_fill * (1.0 + COMMISSION)
                proceeds   = exit_fill  * (1.0 - COMMISSION)
                pnl_per_share_ret = proceeds / cost_basis - 1.0
                pnl = pos.qty * pos.entry_price * pnl_per_share_ret
                cash += pos.qty * sp * (1 - COST)
                trades.append({
                    "side": "SELL", "symbol": pos.symbol, "entry": pos.entry_price,
                    "exit": sp, "pnl": pnl, "qty": pos.qty,
                    "entry_idx": pos.entry_idx, "exit_idx": i, "date": ds,
                    "hold_days": hd, "reason": reason,
                    "realized_R": pnl_per_share_ret / R_UNIT_PCT,
                    "entry_date": pos.entry_decision_date,
                    "size_mult": getattr(pos, "size_mult_at_entry", 1.0),
                })
                pos = None
                last_exit_idx = i

        # ── ENTRY (selection rule UNCHANGED; only admission timing / Case B/C sizing differ) ──
        if pos is None and not mkt_shock:
            cands: List[Tuple[float, int, float, str]] = []
            for sym in active_syms:
                si   = sym_to_i[sym]
                rv   = float(rsr_mat[i, si])
                d90  = int(cross90_mat[i, si])
                sl5v = float(slope5_mat[i, si])
                if not (RSR_LO <= rv < RSR_HI): continue
                if not (D90_MIN <= d90 <= D90_MAX): continue
                if sl5v > SLOPE_MAX: continue
                if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
                    continue
                cands.append((rv, d90, sl5v, sym))

            chosen = None
            chosen_mult = 1.0
            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                top = cands[0]
                n_trigger_days += 1

                # rolling DD at decision time (uses eq[0..i] only -- no lookahead)
                peak_so_far = float(np.max(eq[: i + 1]))
                dd_i = (peak_so_far - eq[i]) / peak_so_far if peak_so_far > 0 else 0.0

                blocked = False
                block_reason = None
                if case in ("D", "F"):
                    gap = i - last_exit_idx
                    required_gap = 1 + (DD_THROTTLE_EXTRA_DAYS if dd_i > DD_THROTTLE_THRESHOLD else 0)
                    if gap < required_gap:
                        blocked = True; block_reason = "DD_THROTTLE"
                if not blocked and case in ("E", "F"):
                    heat = _heat_vol(eq, i)
                    if heat > HEAT_VOL_THRESHOLD:
                        blocked = True; block_reason = "HEAT_BUDGET"

                if blocked:
                    if block_reason == "DD_THROTTLE":
                        n_dd_blocked += 1
                    else:
                        n_heat_blocked += 1
                    blocked_dates.append(ds)
                else:
                    chosen = top
                    chosen_mult = 1.0
                    if case in ("B", "C"):
                        chosen_mult = _vol_mult(close_mat, sym_to_i[top[3]], i)

            if chosen is not None:
                _, _, _, sym = chosen
                si = sym_to_i[sym]
                bp = float(open_mat[nxt, si])
                if bp > 0:
                    alloc = cash * 0.95 * chosen_mult
                    qty   = int(alloc / bp / LOT) * LOT
                    floor_applied = False
                    if qty <= 0 and case == "C":
                        floor_qty  = LOT
                        floor_cost = floor_qty * bp * (1 + COST)
                        if floor_cost <= cash:
                            qty = floor_qty
                            floor_applied = True
                            n_floor_rescued += 1
                    cost = qty * bp * (1 + COST)
                    if qty <= 0 or cost > cash:
                        if qty <= 0:
                            n_lot_infeasible += 1
                    if qty > 0 and cost <= cash:
                        cash -= cost
                        pos = _Pos(sym, qty, bp, nxt)
                        pos.entry_decision_date = ds
                        pos.size_mult_at_entry = 1.0 if floor_applied else chosen_mult
                        trades.append({
                            "side": "BUY", "symbol": sym, "entry": bp, "exit": None,
                            "pnl": None, "qty": qty, "entry_idx": nxt,
                            "date": ds, "entry_date": ds,
                            "size_mult": pos.size_mult_at_entry,
                        })

    n_days_in_pos = int(np.sum(inv > 0))
    return {
        "eq": eq, "inv": inv, "trades": trades,
        "n_trigger_days": n_trigger_days,
        "n_dd_blocked": n_dd_blocked,
        "n_heat_blocked": n_heat_blocked,
        "n_lot_infeasible": n_lot_infeasible,
        "n_floor_rescued": n_floor_rescued,
        "blocked_dates": blocked_dates,
        "n_days_in_pos": n_days_in_pos,
        "n_days_total": n_dates,
    }


def _sell_records(trades: List[dict]) -> List[dict]:
    return [t for t in trades if t["side"] == "SELL"]


def _recovery_stats(eq: np.ndarray) -> Tuple[float, float, int]:
    n = len(eq)
    if n < 2:
        return 0.0, 0.0, 0
    cur_peak = eq[0]
    trough_val: Optional[float] = None
    trough_idx: Optional[int] = None
    in_dd = False
    episodes: List[Tuple[float, int]] = []
    for i in range(1, n):
        if eq[i] >= cur_peak:
            if in_dd and trough_idx is not None and cur_peak > 0:
                depth = (cur_peak - trough_val) / cur_peak
                rec_days = i - trough_idx
                if rec_days > 0:
                    episodes.append((depth, rec_days))
            cur_peak = eq[i]
            in_dd = False
            trough_val = None
            trough_idx = None
        else:
            in_dd = True
            if trough_val is None or eq[i] < trough_val:
                trough_val = eq[i]
                trough_idx = i
    if not episodes:
        return 0.0, 0.0, 0
    avg_days  = float(np.mean([d for _, d in episodes]))
    avg_speed = float(np.mean([depth / d for depth, d in episodes]))
    return avg_days, avg_speed, len(episodes)


def compute_case_metrics(
    case: str,
    sim: dict,
    common_dates: pd.DatetimeIndex,
    capital: float,
    baseline_sim: dict,
    baseline_entry_R_by_date: Dict[str, float],
    n_years: float,
) -> CaseMetrics:
    sells = _sell_records(sim["trades"])
    m = calc_metrics(sim["eq"].tolist(), sim["trades"], (sim["inv"] / np.maximum(sim["eq"], 1.0)).tolist(),
                      capital, list(common_dates))

    cm = CaseMetrics(case=case, label=CASE_LABELS[case])
    cm.trade_count = len(sells)
    cm.cagr     = m.get("cagr", 0.0)
    cm.calmar   = m.get("calmar", 0.0)
    cm.mar      = m.get("calmar", 0.0)   # same full-period formula; no separate windowing in this run
    cm.max_dd   = m.get("max_dd", 0.0)
    cm.sharpe   = m.get("sharpe", 0.0)
    cm.decision_complexity = DECISION_COMPLEXITY[case]
    cm.n_dd_blocked = sim["n_dd_blocked"]
    cm.n_heat_blocked = sim["n_heat_blocked"]
    cm.n_lot_infeasible = sim["n_lot_infeasible"]
    cm.n_floor_rescued = sim["n_floor_rescued"]

    base_sells = _sell_records(baseline_sim["trades"])
    base_total_R = sum(t["realized_R"] for t in base_sells) or 1e-9
    case_total_R = sum(t["realized_R"] for t in sells)
    cm.alpha_retention = round(case_total_R / base_total_R, 4)

    base_keys = {(t["symbol"], t.get("entry_date", "")) for t in base_sells}
    case_keys = {(t["symbol"], t.get("entry_date", "")) for t in sells}
    cm.signal_preservation = round(
        len(base_keys & case_keys) / max(1, len(base_keys)), 4
    )

    admission_blocked_days = sim["n_dd_blocked"] + sim["n_heat_blocked"]
    cm.activation_ratio = round(
        1.0 - admission_blocked_days / max(1, sim["n_trigger_days"]), 4
    )
    cm.activation_loss_ratio = round(1.0 - cm.activation_ratio, 4)
    cm.risk_activation_frequency = round(admission_blocked_days / max(n_years, 0.01), 2)

    admitted_days = max(1, sim["n_trigger_days"] - admission_blocked_days)
    cm.access_loss_ratio = round(sim["n_lot_infeasible"] / admitted_days, 4)
    cm.lot_access = round(1.0 - cm.access_loss_ratio, 4)

    avg_exp_frac = m.get("avg_exposure", 0.0) / 100.0
    cm.capital_activation_ratio = round(avg_exp_frac, 4)
    cm.cash_idle_ratio = round(1.0 - avg_exp_frac, 4)
    cm.slot_utilization = round(sim["n_days_in_pos"] / max(1, sim["n_days_total"]), 4)

    rec_days, rec_speed, _ = _recovery_stats(sim["eq"])
    cm.recovery_days = round(rec_days, 1)
    cm.recovery_speed = round(rec_speed, 5)

    if sells:
        pnls = [t["pnl"] for t in sells]
        rs   = [t["realized_R"] for t in sells]
        cm.p95_trade_loss = round(float(np.percentile(pnls, 5)), 0)
        cm.p95_r = round(float(np.percentile(rs, 5)), 3)

    base_m = calc_metrics(
        baseline_sim["eq"].tolist(), baseline_sim["trades"],
        (baseline_sim["inv"] / np.maximum(baseline_sim["eq"], 1.0)).tolist(),
        capital, list(common_dates),
    )
    cm.dd_reduction_pp = round(abs(base_m.get("max_dd", 0.0)) - abs(cm.max_dd), 4)
    cm.return_preservation = round(
        cm.cagr / base_m.get("cagr", 1e-9), 4
    ) if base_m.get("cagr", 0.0) != 0 else 1.0

    base_act = base_m.get("avg_exposure", 0.0) / 100.0
    base_cagr_per_act = (base_m.get("cagr", 0.0) / base_act) if base_act > 0 else 0.0
    case_cagr_per_act = (cm.cagr / avg_exp_frac) if avg_exp_frac > 0 else 0.0
    cm.capital_efficiency = round(
        case_cagr_per_act / base_cagr_per_act, 4
    ) if base_cagr_per_act != 0 else 1.0

    cm.opportunity_loss_R = round(
        sum(baseline_entry_R_by_date.get(d, 0.0) for d in sim["blocked_dates"]), 4
    )

    base_winners = [t for t in base_sells if t["realized_R"] >= 0]
    top3_base = sorted(base_winners, key=lambda t: -t["realized_R"])[:3]
    preserved = sum(1 for t in top3_base if (t["symbol"], t.get("entry_date", "")) in case_keys)
    tail_capture = preserved / max(1, len(top3_base))
    cm.tail_capture_change = round(tail_capture - 1.0, 4)   # vs baseline's trivial 100%

    cm.calmar_delta = round(cm.calmar - base_m.get("calmar", 0.0), 4)
    cm.activation_trigger_effectiveness = round(
        cm.calmar_delta / max(1, admission_blocked_days), 5
    )

    cm.meets_adoption = (
        cm.alpha_retention >= ADOPT_ALPHA_RET_MIN and
        cm.lot_access >= ADOPT_LOT_ACCESS_MIN and
        cm.dd_reduction_pp >= ADOPT_DD_REDUCTION_MIN_PP and
        cm.calmar_delta >= ADOPT_CALMAR_DELTA_MIN
    )
    return cm


def write_trade_csv(all_trades: Dict[str, List[dict]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, trades in all_trades.items():
        for t in _sell_records(trades):
            row = dict(t); row["case"] = case
            rows.append(row)
    if not rows:
        return
    fieldnames = ["case"] + [k for k in rows[0].keys() if k != "case"]
    with TRADE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[CSV] {TRADE_CSV}")


def _deploy_candidates(metrics: Dict[str, CaseMetrics]) -> List[str]:
    out = []
    for c in CASES:
        if c == "A":
            continue
        m = metrics[c]
        if (m.alpha_retention >= ADOPT_ALPHA_RET_MIN and
                m.lot_access >= DEPLOY_LOT_ACCESS_MIN and
                m.calmar_delta >= ADOPT_CALMAR_DELTA_MIN and
                m.dd_reduction_pp >= ADOPT_DD_REDUCTION_MIN_PP):
            out.append(c)
    return out


def write_md_report(
    metrics: Dict[str, CaseMetrics],
    research_status: str,
    best_case: str,
    root_constraint: str,
    geometry_interaction: str,
    access_preservation: str,
    risk_activation_effect: str,
    recommend_policy: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    w = L.append

    w("# Study27 Access-Preserving Risk Activation Audit")
    w("")
    w("作成日: 2026-06-24  |  リスク発火条件のみ変更（risk_activation_only）/ "
      "Entry・Exit・Signal・RSR閾値・days_cross90・slope・Execution・Authority・Capital・Production変更禁止 / "
      "サイズ変更ではなく発火タイミングだけでの改善余地判定")
    w("")
    w("**Strategy**: Study9 Case B (FROZEN)  **Entry/Exit/Signal/Execution/Authority**: "
      "現行（無変更）  **Capital**: ¥1,800,000  **max_pos**: 1")
    w("")
    w(f"**観測ウィンドウ**: {OBS_START} → {OBS_END}（Study24/25と同期）")
    w("")
    w("**最重要原則**: 本研究は「勝ち方を変える研究」ではない。"
      "既存alphaを維持したまま、Accessを守りながら、"
      "Risk Activationだけで改善余地が残るかを検証する。")
    w("")

    w("---")
    w("## Case別結果")
    w("")
    w("| Case | 説明 | trade_count | CAGR | Calmar | MAR | maxDD | alpha_retention | lot_access |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.label} | {m.trade_count} | {m.cagr:+.2f}% | {m.calmar:.3f} | "
          f"{m.mar:.3f} | {m.max_dd:.2f}% | {m.alpha_retention:.1%} | {m.lot_access:.1%} |")
    w("")
    w("| Case | activation_ratio | capital_activation_ratio | capital_efficiency | "
      "slot_utilization | cash_idle_ratio |")
    w("|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.activation_ratio:.1%} | {m.capital_activation_ratio:.1%} | "
          f"{m.capital_efficiency:.2f}x | {m.slot_utilization:.1%} | {m.cash_idle_ratio:.1%} |")
    w("")
    w("| Case | P95_trade_loss | P95_R | recovery_days | recovery_speed |")
    w("|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | ¥{m.p95_trade_loss:,.0f} | {m.p95_r:.3f}R | {m.recovery_days:.1f}d | "
          f"{m.recovery_speed:.4f}/d |")
    w("")

    w("### 追加監査")
    w("")
    w("| Case | access_loss_ratio | activation_loss_ratio | opportunity_loss(R) | "
      "tail_capture_change | signal_preservation | decision_complexity | "
      "risk_activation_frequency(/yr) | trigger_effectiveness |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.access_loss_ratio:.1%} | {m.activation_loss_ratio:.1%} | "
          f"{m.opportunity_loss_R:+.4f}R | {m.tail_capture_change:+.1%} | "
          f"{m.signal_preservation:.1%} | {m.decision_complexity} | "
          f"{m.risk_activation_frequency:.2f} | {m.activation_trigger_effectiveness:+.5f} |")
    w("")
    w("注（lot_access / access_loss_ratio）: admission（D/E/Fのdd_throttle・heat_budget）"
      "を通過した日のうち、サイズが縮小され1単元未満に"
      "切り下げられてentryが事実上消失した割合（Study25既知のlot feasibility問題）。"
      "Case D/E/Fはサイズ不変なので構造上 lot_access≈100%となる。")
    w("")
    w("注（activation_ratio / activation_loss_ratio）: trigger_daysのうち、"
      "admissionそのものが拒否された日の割合。"
      "B/Cは一度もadmissionを拒否しない設計（サイズのみ変更）のため activation_ratio=100%。")
    w("")

    w("---")
    w("## 採用条件判定")
    w("")
    w(f"基準: alpha_retention≥{ADOPT_ALPHA_RET_MIN:.0%} AND lot_access≥{ADOPT_LOT_ACCESS_MIN:.0%} AND "
      f"DD改善≥{ADOPT_DD_REDUCTION_MIN_PP:.1f}pp AND Calmar改善≥{ADOPT_CALMAR_DELTA_MIN:.2f}")
    w("")
    w("| Case | calmar_delta | dd_delta | alpha_retention | lot_access | meets_adoption |")
    w("|---|---|---|---|---|---|")
    for c in CASES:
        if c == "A":
            continue
        m = metrics[c]
        w(f"| {c} | {m.calmar_delta:+.4f} | {m.dd_reduction_pp:+.2f}pp | "
          f"{m.alpha_retention:.1%} | {m.lot_access:.1%} | "
          f"{'✅ YES' if m.meets_adoption else '❌ no'} |")
    w("")

    w("---")
    w("## 最終出力")
    w("")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| **best_case** | {best_case} |")
    w(f"| best_case_label | {CASE_LABELS[best_case]} |")
    w(f"| **root_constraint** | {root_constraint} |")
    w(f"| geometry_interaction | {geometry_interaction} |")
    w(f"| access_preservation | {access_preservation} |")
    w(f"| risk_activation_effect | {risk_activation_effect} |")
    w(f"| calmar_delta (best) | {metrics[best_case].calmar_delta:+.4f} |")
    w(f"| dd_delta (best) | {metrics[best_case].dd_reduction_pp:+.2f}pp |")
    w(f"| **recommend_policy** | {recommend_policy} |")
    w(f"| **research_status** | **{research_status}** |")
    w("")
    w("制約: 本研究は改善案の実装・Entry/Exit/Signal/Production変更を一切行わない"
      "（改善余地の存在判定のみ）。")
    w("")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[MD] {REPORT_MD}")


def append_telemetry(metrics: Dict[str, CaseMetrics], research_status: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "study": "study27",
            "research_status": research_status,
            "cases": {c: asdict(m) for c, m in metrics.items()},
        }
        with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def main() -> int:
    print("[Study27] Access-Preserving Risk Activation Audit")
    print("=" * 68)

    print("[1/4] データロード中...")
    cfg = load_strategy_config()
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
        (common_dates >= pd.Timestamp(OBS_START)) &
        (common_dates <= pd.Timestamp(OBS_END))
    ]
    n_years = (common_dates[-1] - common_dates[0]).days / 365.25
    print(f"[1/4] 共通日数={len(common_dates)}  銘柄={n_syms}  期間={n_years:.2f}年")

    print("[2/4] 価格・RSRマトリクス構築...")
    n_dates = len(common_dates)
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src = universe_raw[sym]["df"]
        ri = df_src.index.get_indexer(common_dates)
        valid = ri >= 0
        if valid.any():
            open_mat[valid, si]  = df_src["Open"].to_numpy(dtype=np.float32)[ri[valid]]
            close_mat[valid, si] = df_src["Close"].to_numpy(dtype=np.float32)[ri[valid]]

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan), nan=0.0)
    sym_active_mat = (
        None if sym_active_df is None
        else _take(sym_active_df, common_dates, active_syms, dtype=np.float32, fill_value=1.0)
    )
    mkt_ret1 = (
        None if topix_close is None
        else _take(topix_close.pct_change(), common_dates, dtype=np.float32, fill_value=0.0)
    )
    cross90_mat = _cross90(rsr_mat)
    slope5_mat  = _slope5(rsr_mat)

    print("[3/4] Case A〜F シミュレーション実行...")
    sims: Dict[str, dict] = {}
    for case in CASES:
        sims[case] = simulate_case(
            case, common_dates, active_syms, sym_to_i, open_mat, close_mat, rsr_mat,
            sym_active_mat, cross90_mat, slope5_mat, mkt_ret1, CAPITAL,
        )
        s = sims[case]
        print(f"  [{case}] trades={len(_sell_records(s['trades']))}  "
              f"trigger_days={s['n_trigger_days']}  dd_blocked={s['n_dd_blocked']}  "
              f"heat_blocked={s['n_heat_blocked']}  lot_infeasible={s['n_lot_infeasible']}  "
              f"floor_rescued={s['n_floor_rescued']}")

    base_sells = _sell_records(sims["A"]["trades"])
    baseline_entry_R_by_date = {t["entry_date"]: t["realized_R"] for t in base_sells}

    print("[4/4] メトリクス計算 + 判定...")
    metrics: Dict[str, CaseMetrics] = {}
    for case in CASES:
        metrics[case] = compute_case_metrics(
            case, sims[case], common_dates, CAPITAL, sims["A"], baseline_entry_R_by_date, n_years,
        )

    non_baseline = [c for c in CASES if c != "A"]
    best_by_calmar = max(non_baseline, key=lambda c: metrics[c].calmar_delta)
    best_delta_calmar = metrics[best_by_calmar].calmar_delta

    deploy_cands = _deploy_candidates(metrics)

    if deploy_cands:
        best_case = max(deploy_cands, key=lambda c: metrics[c].calmar_delta)
        research_status = "DEPLOY_CANDIDATE"
    elif (best_delta_calmar < EXHAUSTED_CALMAR_DELTA or
          metrics[best_by_calmar].access_loss_ratio > EXHAUSTED_ACCESS_LOSS_MAX):
        best_case = best_by_calmar
        research_status = "EXHAUSTED"
    elif (metrics[best_by_calmar].calmar_delta > 0 and
          metrics[best_by_calmar].alpha_retention >= PROMISING_ALPHA_RET_MIN):
        best_case = best_by_calmar
        research_status = "PROMISING_BUT_INSUFFICIENT"
    else:
        best_case = best_by_calmar
        research_status = "EXHAUSTED"

    bm = metrics[best_case]

    # root_constraint: dominant binding constraint, derived from the actual case comparison
    # (not a static guess) -- compare the pure-timing levers (D/E/F, mult=1.0 always) against
    # the sizing lever (B/C) to see which one actually carries the achievable Calmar delta.
    timing_cases = ["D", "E", "F"]
    sizing_cases = ["B", "C"]
    best_timing_calmar = max(metrics[c].calmar_delta for c in timing_cases)
    best_sizing_calmar = max(metrics[c].calmar_delta for c in sizing_cases)
    heat_destructive = metrics["E"].alpha_retention < 0.5 or metrics["F"].alpha_retention < 0.5

    heat_note = (
        f" Secondary finding: pure-timing Heat Budget (E/F) is actively destructive "
        f"(E alpha_retention={metrics['E'].alpha_retention:.1%}, F={metrics['F'].alpha_retention:.1%}) "
        f"-- portfolio realized-vol heat budget blocks the strategy's own alpha-bearing momentum "
        f"continuation; equity-curve volatility IS the alpha signature here, not an independent risk "
        f"signal that can be gated separately." if heat_destructive else ""
    )

    if bm.access_loss_ratio > EXHAUSTED_ACCESS_LOSS_MAX:
        root_constraint = "LOT_ACCESS (size-shrink overlays still destroy lot access on a single-slot book)" + heat_note
    elif best_case in sizing_cases:
        root_constraint = (
            f"EFFECT_SIZE_INSUFFICIENT (sizing-side lever dominates pure-timing levers: best sizing "
            f"Δcalmar={best_sizing_calmar:+.4f} ({max(sizing_cases, key=lambda c: metrics[c].calmar_delta)}) "
            f"vs best pure-timing Δcalmar={best_timing_calmar:+.4f} "
            f"({max(timing_cases, key=lambda c: metrics[c].calmar_delta)}) -- risk_activation TIMING alone "
            f"captures less DD-reduction headroom than size modulation, and even the best sizing lever "
            f"falls short of the adoption bar ({ADOPT_CALMAR_DELTA_MIN:.2f}))." + heat_note
        )
    elif best_case in ("E", "F") and heat_destructive:
        root_constraint = (
            f"HEAT_SIGNAL_CONFOUND (portfolio realized-vol heat budget (E: alpha_retention="
            f"{metrics['E'].alpha_retention:.1%}, F: {metrics['F'].alpha_retention:.1%}) blocks the "
            f"strategy's own alpha-bearing momentum continuation -- equity-curve volatility IS the "
            f"alpha signature here, not an independent risk signal that can be gated separately)"
        )
    elif bm.calmar_delta < EXHAUSTED_CALMAR_DELTA:
        root_constraint = "EFFECT_SIZE_INSUFFICIENT (best case calmar_delta below adoption bar with no single dominant mechanism identified)." + heat_note
    else:
        root_constraint = "NONE_BINDING (adoption thresholds met)"

    geometry_interaction = (
        f"Case B/C(sizing) vs D/E/F(activation-only) lot_access comparison: "
        f"B={metrics['B'].lot_access:.1%} / C={metrics['C'].lot_access:.1%} / "
        f"D={metrics['D'].lot_access:.1%} / E={metrics['E'].lot_access:.1%} / F={metrics['F'].lot_access:.1%}. "
        f"{'サイズ縮小を含むCaseのみ lot_access低下→Study25と同様のLOTカスケード再現' if metrics['B'].lot_access < 0.98 or metrics['C'].lot_access < 0.98 else '全Caseでlot_access高水準保持'}"
    )
    access_preservation = f"{best_case}: lot_access={bm.lot_access:.1%}, alpha_retention={bm.alpha_retention:.1%}"
    risk_activation_effect = (
        f"{best_case}: dd_reduction={bm.dd_reduction_pp:+.2f}pp via "
        f"{bm.n_dd_blocked}dd_blocks+{bm.n_heat_blocked}heat_blocks "
        f"({bm.risk_activation_frequency:.2f}/yr), calmar_delta={bm.calmar_delta:+.4f}"
    )
    if research_status == "DEPLOY_CANDIDATE":
        recommend_policy = f"PROCEED_TO_WALKFORWARD ({best_case}: {CASE_LABELS[best_case]})"
    elif research_status == "PROMISING_BUT_INSUFFICIENT":
        recommend_policy = f"CONTINUE_RESEARCH ({best_case} は正のCalmar改善だが採用閾値未達。追加検証候補)"
    else:
        recommend_policy = "NO_CHANGE_RECOMMENDED (risk_activation_only軸も新たな改善余地なし)"

    print(f"  best_case          = {best_case}")
    print(f"  research_status    = {research_status}")
    print(f"  root_constraint    = {root_constraint}")

    write_trade_csv({c: sims[c]["trades"] for c in CASES})
    write_md_report(
        metrics, research_status, best_case, root_constraint, geometry_interaction,
        access_preservation, risk_activation_effect, recommend_policy,
    )
    append_telemetry(metrics, research_status)

    print()
    print("=" * 68)
    print(f"research_status = {research_status}  (best_case={best_case})")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
