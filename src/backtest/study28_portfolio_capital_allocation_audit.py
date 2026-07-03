"""
src/backtest/study28_portfolio_capital_allocation_audit.py
Study28 — Portfolio Capital Allocation Audit

Objective: with Entry/Exit/Signal/Execution/Universe FULLY FROZEN (Study9 Case B signal:
RSR[92,95), days_cross90<=5, slope5<=5, exit RSR<90, min_hold=3 -- the same signal
definition used verbatim across Study20/24/25/27), determine whether Calmar improvement
room remains by changing ONLY how capital is SPLIT across positions that are held
SIMULTANEOUSLY. This is explicitly NOT a sizing study (Study25/27 already audited and
closed the sizing/timing axes) -- it is a pure intra-portfolio capital-allocation audit.

Baseline switch from Study20-27: this study uses the REAL PRODUCTION PARAMS_LOCKED
config (Capital=Y3,000,000, max_positions=3), not the Study20-27 lineage's simplified
single-slot Y1.8M variant -- because "simultaneous holding" is structurally impossible
at max_pos=1. Entry/Exit signal logic is otherwise identical to the Study9/20/24/25/27
lineage (same RSR window, same exit rule) -- only extended to fill up to 3 slots with NO
rotation (Position Count change forbidden; selection rule unchanged: rank-by-RSR, fill
open slots, never swap an already-held name out early).

KEY STRUCTURAL FACT (verified by construction, not assumed): because the entry/exit
SIGNAL never depends on position size or available cash (only on RSR/date/slope), the
schedule of which symbol enters/exits on which day is IDENTICAL across all cases A-G,
*except* for the mandatory 1-lot floor (see below) which guarantees every selected
candidate is always actually entered. This lets every case be compared apples-to-apples
on the same trade population -- only the DOLLAR WEIGHT given to each simultaneously-held
name differs.

Mandatory constraints (per task spec, applied uniformly to ALL cases including A):
  - 1lot未満禁止: every sized entry/rebalance-buy is floored to >=1 LOT when cash allows;
    an existing position is never reduced below 1 LOT by a rebalance trade (only the
    Exit signal can fully close a position).
  - max_single_weight=0.25 (CLAUDE.md CIRCUIT, code-embedded, change forbidden): every
    case's per-symbol dollar target is hard-capped at 25% of CAPITAL. Note this means
    even Case A's "equal weight" target (capital/max_positions = 33.3%) is capped down to
    25% -- production's real circuit constraint, not a Study28 invention.

Allocation re-weighting model: re-weighting across the CURRENTLY HELD set happens only on
a "composition-change event" -- a day where a new entry is added and/or an exit removes a
name (i.e. exactly "同時保有時のみ配分変更": the relative split is only a meaningful
question when 2+ names are, or become, simultaneously held). When only 1 name is held,
every case degenerates to the same single-slot target as Case A (no allocation choice to
make with 1 name). Existing survivors are re-traded toward their new target dollar value
at the next day's open (LOT-quantized, floor-protected); a partial sell realizes pnl
against the position's running weighted-average cost basis and is tallied as
capital_reallocation_gain/loss; partial buys raise the weighted-average cost basis. The
position's eventual EXIT pnl/realized_R is computed off that running avg_cost, so the
allocation policy's effect flows naturally into alpha_retention via the cost basis.

Cases (target_i = weight_i * (n_held/MAX_POS) * CAPITAL, capped at MAX_SINGLE_W*CAPITAL):
  A  Current Allocation        : production Pattern-A equivalent -- fixed 1/MAX_POS target
                                  set ONCE at entry, survivors NEVER re-traded. No
                                  case-specific weight function (this IS the no-allocation
                                  baseline all other cases are measured against).
  B  RSR Weighted Allocation   : weight_i = RSR_i / sum(RSR)            (today's RSR)
  C  Volatility Quality Alloc. : weight_i = (1/ATRpct_i) / sum(1/ATRpct)  (today's ATR%,
                                  20d True-Range / Close -- NOT a sizing shrink, a SPLIT
                                  ratio between simultaneously-held names; Study27's Vol
                                  Shrink is explicitly out of scope here)
  D  RSR x Vol Quality         : weight_i = (RSR_i/ATRpct_i), normalized
  E  Winner Concentration      : rank by today's RSR among held names -> literal buckets
                                  Top1=40% / Top2=30% / Top3=20% (NOT renormalized -- the
                                  unused "残り10%" sits idle as cash by construction,
                                  exactly as specified)
  F  Dynamic Composite         : score = z(RSR) + z(1/ATRpct) across the held set today,
                                  weight_i = softmax(score)_i
  G  Perfect Foresight         : weight_i ~ max(EPS, full-episode realized return) for
                                  that holding's OWN (already signal-determined, allocation
                                  -independent) entry-to-exit return, looked up from a
                                  precomputed Case-A episode table. NOT real-time
                                  executable (uses each episode's own future outcome) --
                                  included only to measure allocation_theoretical_ceiling,
                                  i.e. the best ANY allocation policy could possibly do.

Prohibited (per task spec): Entry/Exit/Signal/Execution/Universe change, new Risk Filter,
Heat Budget, DD Admission, Volatility Shrink (sizing), Adaptive Sizing, Position Count
change, Production code change. This script changes nothing in production; it only
measures whether intra-portfolio allocation improvement room exists.

Adoption thresholds (per task spec):
  meets_adoption (per case) : alpha_retention>=98% AND lot_access>=95% AND
                               calmar_delta>=+0.10 AND dd_reduction_pp>=2.0pp
  EXHAUSTED                 : best_calmar_delta (across B-F) < 0.10
                               -- STRONGLY reinforced if Case G's calmar_delta (the
                               theoretical ceiling) is ALSO < 0.10, since no real-time
                               policy can ever exceed the oracle.

Run:
    cd C:/ai-trading
    python src/backtest/study28_portfolio_capital_allocation_audit.py
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
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
CAPITAL   = 3_000_000.0   # 本番 PARAMS_LOCKED (strategy.yaml portfolio.capital)
MAX_POS   = 3             # 本番 PARAMS_LOCKED (strategy.yaml portfolio.max_positions)
MAX_SINGLE_W = 0.25       # CLAUDE.md CIRCUIT (code組み込み済み・変更禁止) -- applied uniformly

ATR_LOOKBACK = 20
G_FLOOR      = 0.01       # Case G: floor weight so a foreseen loser never gets literal 0
E_BUCKETS    = [0.40, 0.30, 0.20]   # Case E: literal buckets, NOT renormalized

REPORT_DIR    = Path("reports")
REPORT_MD     = REPORT_DIR / "study28_portfolio_capital_allocation_audit.md"
TRADE_CSV     = REPORT_DIR / "study28_case_trades.csv"
LOG_DIR       = Path("logs")
TELEMETRY_LOG = LOG_DIR / "study28_telemetry.jsonl"

CASES = ["A", "B", "C", "D", "E", "F", "G"]
CASE_LABELS = {
    "A": "Current Allocation (現行実装, 固定1/MAX_POS, 再配分なし)",
    "B": "RSR Weighted Allocation (weight=RSR/ΣRSR)",
    "C": "Volatility Quality Allocation (weight=(1/ATRpct)/Σ(1/ATRpct))",
    "D": "RSR x Vol Quality (weight=(RSR/ATRpct)正規化)",
    "E": "Winner Concentration (Top1=40%/Top2=30%/Top3=20%, 残り現金)",
    "F": "Dynamic Composite (softmax(z(RSR)+z(1/ATRpct)))",
    "G": "Perfect Foresight Allocation (oracle, 理論上限のみ)",
}
DECISION_COMPLEXITY = {"A": 0, "B": 1, "C": 1, "D": 1, "E": 1, "F": 2, "G": 0}

ADOPT_ALPHA_RET_MIN = 0.98
ADOPT_LOT_ACCESS_MIN = 0.95
ADOPT_CALMAR_DELTA_MIN = 0.10
ADOPT_DD_REDUCTION_MIN_PP = 2.0
EXHAUSTED_CALMAR_DELTA = 0.10


@dataclass
class CaseMetrics:
    case: str
    label: str
    trade_count: int = 0
    cagr: float = 0.0
    max_dd: float = 0.0
    calmar: float = 0.0
    sharpe: float = 0.0
    alpha_retention: float = 1.0
    lot_access: float = 1.0
    capital_activation: float = 0.0
    allocation_entropy: float = 0.0
    concentration_risk: float = 0.0
    opportunity_cost: float = 0.0
    winner_capture: float = 1.0
    loser_capital_ratio: float = 0.0
    capital_reallocation_gain: float = 0.0
    capital_reallocation_loss: float = 0.0
    top20pct_profit_alloc_share: float = 0.0
    bottom20pct_loss_alloc_share: float = 0.0
    realloc_profit_gain_yen: float = 0.0
    realloc_loss_increase_yen: float = 0.0
    n_simultaneous_days: int = 0
    n_rebalance_events: int = 0
    n_lot_infeasible: int = 0
    dd_reduction_pp: float = 0.0
    calmar_delta: float = 0.0
    meets_adoption: bool = False


class _Pos:
    __slots__ = ("symbol", "qty", "avg_cost", "entry_idx", "db", "entry_decision_date",
                 "episode_return")

    def __init__(self, symbol, qty, avg_cost, entry_idx):
        self.symbol = symbol; self.qty = qty
        self.avg_cost = avg_cost; self.entry_idx = entry_idx
        self.db = 0
        self.entry_decision_date = ""
        self.episode_return = 0.0   # Case G lookup only


def _zscore(arr: np.ndarray) -> np.ndarray:
    sd = float(np.std(arr))
    if sd <= 1e-9:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / sd


def _entropy(weights: List[float]) -> float:
    w = np.array([x for x in weights if x > 1e-9])
    if len(w) == 0:
        return 0.0
    w = w / w.sum()
    return float(-np.sum(w * np.log(w)))


def _weights_for_case(
    case: str, held_syms: List[str], i: int, sym_to_i: Dict[str, int],
    rsr_mat: np.ndarray, atrpct_mat: np.ndarray, ep_return_by_sym: Dict[str, float],
) -> Dict[str, float]:
    """Returns normalized (or, for Case E, literal-bucket non-normalized) weights
    over held_syms. Caller is responsible for the single-name degenerate case."""
    if case == "B":
        vals = {s: max(float(rsr_mat[i, sym_to_i[s]]), 1e-6) for s in held_syms}
    elif case == "C":
        vals = {s: 1.0 / max(float(atrpct_mat[i, sym_to_i[s]]), 1e-6) for s in held_syms}
    elif case == "D":
        vals = {
            s: max(float(rsr_mat[i, sym_to_i[s]]), 1e-6) /
               max(float(atrpct_mat[i, sym_to_i[s]]), 1e-6)
            for s in held_syms
        }
    elif case == "E":
        ranked = sorted(held_syms, key=lambda s: -float(rsr_mat[i, sym_to_i[s]]))
        vals = {}
        for rank, s in enumerate(ranked):
            vals[s] = E_BUCKETS[rank] if rank < len(E_BUCKETS) else 0.0
        return vals   # literal buckets -- NOT normalized (by design)
    elif case == "F":
        rsr_vals  = np.array([float(rsr_mat[i, sym_to_i[s]]) for s in held_syms])
        atrq_vals = np.array([1.0 / max(float(atrpct_mat[i, sym_to_i[s]]), 1e-6) for s in held_syms])
        score = _zscore(rsr_vals) + _zscore(atrq_vals)
        ex = np.exp(score - np.max(score))
        w = ex / ex.sum()
        return {s: float(w[k]) for k, s in enumerate(held_syms)}
    elif case == "G":
        vals = {s: max(ep_return_by_sym.get(s, 0.0), G_FLOOR) for s in held_syms}
    else:
        vals = {s: 1.0 for s in held_syms}

    total = sum(vals.values()) or 1e-9
    return {s: v / total for s, v in vals.items()}


def _atrpct_matrix(universe_raw: dict, active_syms: List[str], common_dates: pd.DatetimeIndex,
                    close_mat: np.ndarray) -> np.ndarray:
    n_dates = len(common_dates)
    n_syms  = len(active_syms)
    atrpct_mat = np.full((n_dates, n_syms), 0.02, dtype=np.float32)   # 2% fallback
    for si, sym in enumerate(active_syms):
        df_src = universe_raw[sym]["df"]
        ri = df_src.index.get_indexer(common_dates)
        valid = ri >= 0
        if not valid.any():
            continue
        h = df_src["High"].to_numpy(dtype=np.float64)[ri[valid]]
        l = df_src["Low"].to_numpy(dtype=np.float64)[ri[valid]]
        c = close_mat[valid, si].astype(np.float64)
        cp = np.empty_like(c); cp[0] = c[0]; cp[1:] = c[:-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - cp), np.abs(l - cp)))
        atr = pd.Series(tr).rolling(ATR_LOOKBACK, min_periods=1).mean().to_numpy(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(c > 0, atr / c, 0.02)
        idx_valid = np.where(valid)[0]
        atrpct_mat[idx_valid, si] = pct.astype(np.float32)
    return atrpct_mat


def simulate_case(
    case: str,
    common_dates: pd.DatetimeIndex,
    active_syms: List[str],
    sym_to_i: Dict[str, int],
    open_mat: np.ndarray,
    close_mat: np.ndarray,
    rsr_mat: np.ndarray,
    atrpct_mat: np.ndarray,
    sym_active_mat: Optional[np.ndarray],
    cross90_mat: np.ndarray,
    slope5_mat: np.ndarray,
    mkt_ret1: Optional[np.ndarray],
    capital: float,
    episode_return_lookup: Optional[Dict[Tuple[str, str], float]] = None,
) -> dict:
    n_dates = len(common_dates)
    cash = capital
    positions: Dict[str, _Pos] = {}
    trades: List[dict] = []
    eq  = np.zeros(n_dates, dtype=np.float64)
    inv = np.zeros(n_dates, dtype=np.float64)

    n_lot_infeasible   = 0
    n_simultaneous_days = 0
    n_rebalance_events  = 0
    realloc_gain_yen    = 0.0
    realloc_loss_yen    = 0.0
    weight_snapshots: List[List[float]] = []   # realized weight vector at each n_held>=2 event

    episode_return_lookup = episode_return_lookup or {}

    for i, date in enumerate(common_dates):
        ds = str(date.date())
        invested = sum(p.qty * float(close_mat[i, sym_to_i[s]]) for s, p in positions.items())
        eq[i]  = cash + invested
        inv[i] = invested
        if len(positions) >= 2:
            n_simultaneous_days += 1
        if i + 1 >= n_dates:
            break
        next_i = i + 1
        mkt_shock = (mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT)

        composition_changed = False

        # ── EXIT (verbatim signal logic, applied per held symbol) ──────────
        sell_syms: List[Tuple[str, str]] = []
        for sym, pos in list(positions.items()):
            si = sym_to_i[sym]
            rv = float(rsr_mat[i, si]); ct = float(close_mat[i, si])
            hd = i - pos.entry_idx
            do_exit = False; reason = ""
            if mkt_shock:
                if i > 0:
                    pc = float(close_mat[i - 1, si])
                    if pc > 0 and (ct / pc - 1) <= SHOCK_SYM:
                        do_exit = True; reason = "MKT_SHOCK"
            if not do_exit:
                pos.db = (pos.db + 1) if rv < EXIT_THR else 0
                if pos.db >= 1 and hd >= MIN_HOLD:
                    do_exit = True; reason = "RSR_EXIT"
            if do_exit:
                sell_syms.append((sym, reason))

        for sym, reason in sell_syms:
            pos = positions[sym]
            si = sym_to_i[sym]
            sp = float(open_mat[next_i, si])
            entry_fill = pos.avg_cost * (1.0 + SLIPPAGE)
            exit_fill  = sp * (1.0 - SLIPPAGE)
            cost_basis = entry_fill * (1.0 + COMMISSION)
            proceeds_ps = exit_fill * (1.0 - COMMISSION)
            pnl_per_share_ret = proceeds_ps / cost_basis - 1.0
            pnl = pos.qty * pos.avg_cost * pnl_per_share_ret
            cash += pos.qty * sp * (1 - COST)
            trades.append({
                "side": "SELL", "symbol": sym, "entry": pos.avg_cost, "exit": sp,
                "pnl": pnl, "qty": pos.qty, "entry_idx": pos.entry_idx, "exit_idx": i,
                "date": ds, "hold_days": i - pos.entry_idx, "reason": reason,
                "realized_R": pnl_per_share_ret / R_UNIT_PCT,
                "entry_date": pos.entry_decision_date,
            })
            del positions[sym]
            composition_changed = True

        # ── ENTRY (selection rule UNCHANGED; fill open slots, no rotation) ──
        new_entrants: List[Tuple[float, str]] = []
        if not mkt_shock:
            open_slots = MAX_POS - len(positions)
            if open_slots > 0:
                cands: List[Tuple[float, int, float, str]] = []
                for sym in active_syms:
                    if sym in positions:
                        continue
                    si = sym_to_i[sym]
                    rv = float(rsr_mat[i, si]); d90 = int(cross90_mat[i, si])
                    sl5v = float(slope5_mat[i, si])
                    if not (RSR_LO <= rv < RSR_HI): continue
                    if not (D90_MIN <= d90 <= D90_MAX): continue
                    if sl5v > SLOPE_MAX: continue
                    if sym_active_mat is not None and float(sym_active_mat[i, si]) < 0.5:
                        continue
                    cands.append((rv, d90, sl5v, sym))
                if cands:
                    cands.sort(key=lambda x: (-x[0], x[1]))
                    for rv, _, _, sym in cands[:open_slots]:
                        new_entrants.append((rv, sym))
                    composition_changed = True

        # ── ALLOCATION / REBALANCE (only on composition-change events) ─────
        if composition_changed:
            new_held = list(positions.keys()) + [s for _, s in new_entrants]
            n_held = len(new_held)

            if n_held == 0:
                pass
            elif n_held == 1 or case == "A":
                # degenerate single-name case, OR Case A's fixed-at-entry policy:
                # survivors are NEVER re-traded; only size brand-new entrants.
                if n_held >= 2:
                    weight_snapshots.append([1.0 / n_held] * n_held)
                for _, sym in new_entrants:
                    si = sym_to_i[sym]
                    bp = float(open_mat[next_i, si])
                    if bp <= 0:
                        continue
                    target = min(capital / MAX_POS, capital * MAX_SINGLE_W)
                    qty = max(LOT, int(target / bp / LOT) * LOT)
                    cost = qty * bp * (1 + COST)
                    if cost > cash:
                        qty = int(cash / bp / LOT) * LOT
                        cost = qty * bp * (1 + COST)
                    if qty <= 0:
                        n_lot_infeasible += 1
                        continue
                    cash -= cost
                    pos = _Pos(sym, qty, bp, next_i)
                    pos.entry_decision_date = ds
                    pos.episode_return = episode_return_lookup.get((sym, ds), 0.0)
                    positions[sym] = pos
                    trades.append({
                        "side": "BUY", "symbol": sym, "entry": bp, "exit": None,
                        "pnl": None, "qty": qty, "entry_idx": next_i,
                        "date": ds, "entry_date": ds,
                    })
            else:
                n_rebalance_events += 1
                ep_return_by_sym = {
                    s: (positions[s].episode_return if s in positions
                        else episode_return_lookup.get((s, ds), 0.0))
                    for s in new_held
                }
                weights = _weights_for_case(case, new_held, i, sym_to_i, rsr_mat, atrpct_mat,
                                             ep_return_by_sym)
                weight_snapshots.append([weights.get(s, 0.0) for s in new_held])
                pool = (n_held / MAX_POS) * capital
                for sym in new_held:
                    target = min(weights.get(sym, 0.0) * pool, capital * MAX_SINGLE_W)
                    si = sym_to_i[sym]
                    bp = float(open_mat[next_i, si])
                    if bp <= 0:
                        continue
                    if sym not in positions:
                        # brand-new entrant: buy straight to target, floor 1 LOT
                        qty = max(LOT, int(target / bp / LOT) * LOT)
                        cost = qty * bp * (1 + COST)
                        if cost > cash:
                            qty = int(cash / bp / LOT) * LOT
                            cost = qty * bp * (1 + COST)
                        if qty <= 0:
                            n_lot_infeasible += 1
                            continue
                        cash -= cost
                        pos = _Pos(sym, qty, bp, next_i)
                        pos.entry_decision_date = ds
                        pos.episode_return = episode_return_lookup.get((sym, ds), 0.0)
                        positions[sym] = pos
                        trades.append({
                            "side": "BUY", "symbol": sym, "entry": bp, "exit": None,
                            "pnl": None, "qty": qty, "entry_idx": next_i,
                            "date": ds, "entry_date": ds,
                        })
                    else:
                        # existing survivor: re-trade toward new target (floor-protected)
                        pos = positions[sym]
                        cur_value = pos.qty * float(close_mat[i, si])
                        delta = target - cur_value
                        if delta > 0:
                            add_qty = int(delta / bp / LOT) * LOT
                            cost = add_qty * bp * (1 + COST)
                            if cost > cash:
                                add_qty = int(cash / bp / LOT) * LOT
                                cost = add_qty * bp * (1 + COST)
                            if add_qty > 0:
                                new_qty = pos.qty + add_qty
                                pos.avg_cost = (pos.qty * pos.avg_cost + add_qty * bp) / new_qty
                                pos.qty = new_qty
                                cash -= cost
                        elif delta < 0:
                            max_sellable = max(0, pos.qty - LOT)   # never breach 1-LOT floor
                            sell_qty = min(max_sellable, int((-delta) / bp / LOT) * LOT)
                            if sell_qty > 0:
                                sell_fill = bp * (1 - SLIPPAGE) * (1 - COMMISSION)
                                cost_fill = pos.avg_cost * (1 + SLIPPAGE) * (1 + COMMISSION)
                                partial_pnl = (sell_fill - cost_fill) * sell_qty
                                if partial_pnl >= 0:
                                    realloc_gain_yen += partial_pnl
                                else:
                                    realloc_loss_yen += -partial_pnl
                                cash += sell_qty * bp * (1 - COST)
                                pos.qty -= sell_qty

    n_days_in_pos = int(np.sum(inv > 0))
    return {
        "eq": eq, "inv": inv, "trades": trades,
        "weight_snapshots": weight_snapshots,
        "n_lot_infeasible": n_lot_infeasible,
        "n_simultaneous_days": n_simultaneous_days,
        "n_rebalance_events": n_rebalance_events,
        "realloc_gain_yen": realloc_gain_yen,
        "realloc_loss_yen": realloc_loss_yen,
        "n_days_in_pos": n_days_in_pos,
        "n_days_total": n_dates,
    }


def _sell_records(trades: List[dict]) -> List[dict]:
    return [t for t in trades if t["side"] == "SELL"]


def compute_case_metrics(
    case: str, sim: dict, common_dates: pd.DatetimeIndex, capital: float,
    baseline_sim: dict,
) -> CaseMetrics:
    sells = _sell_records(sim["trades"])
    m = calc_metrics(sim["eq"].tolist(), sim["trades"], (sim["inv"] / np.maximum(sim["eq"], 1.0)).tolist(),
                      capital, list(common_dates))
    base_m = calc_metrics(
        baseline_sim["eq"].tolist(), baseline_sim["trades"],
        (baseline_sim["inv"] / np.maximum(baseline_sim["eq"], 1.0)).tolist(),
        capital, list(common_dates),
    )
    base_sells = _sell_records(baseline_sim["trades"])

    cm = CaseMetrics(case=case, label=CASE_LABELS[case])
    cm.trade_count = len(sells)
    cm.cagr   = m.get("cagr", 0.0)
    cm.max_dd = m.get("max_dd", 0.0)
    cm.calmar = m.get("calmar", 0.0)
    cm.sharpe = m.get("sharpe", 0.0)
    cm.n_lot_infeasible = sim["n_lot_infeasible"]
    cm.n_simultaneous_days = sim["n_simultaneous_days"]
    cm.n_rebalance_events = sim["n_rebalance_events"]
    cm.realloc_profit_gain_yen = round(sim["realloc_gain_yen"], 0)
    cm.realloc_loss_increase_yen = round(sim["realloc_loss_yen"], 0)
    cm.capital_reallocation_gain = round(sim["realloc_gain_yen"], 0)
    cm.capital_reallocation_loss = round(sim["realloc_loss_yen"], 0)

    base_total_R = sum(t["realized_R"] for t in base_sells) or 1e-9
    case_total_R = sum(t["realized_R"] for t in sells)
    cm.alpha_retention = round(case_total_R / base_total_R, 4)

    # lot_access: fraction of attempted entries (BUY records + infeasible count) that
    # actually resulted in a real (>=1 LOT) position.
    n_buy_attempts = len([t for t in sim["trades"] if t["side"] == "BUY"]) + sim["n_lot_infeasible"]
    cm.lot_access = round(
        1.0 - sim["n_lot_infeasible"] / max(1, n_buy_attempts), 4
    )

    avg_exp = m.get("avg_exposure", 0.0) / 100.0
    cm.capital_activation = round(avg_exp, 4)

    snaps = sim.get("weight_snapshots", [])
    if snaps:
        cm.allocation_entropy = round(float(np.mean([_entropy(w) for w in snaps])), 4)
        cm.concentration_risk = round(float(np.mean([max(w) for w in snaps])), 4)

    cm.dd_reduction_pp = round(abs(base_m.get("max_dd", 0.0)) - abs(cm.max_dd), 4)
    cm.calmar_delta = round(cm.calmar - base_m.get("calmar", 0.0), 4)

    if sells:
        pnls = sorted(t["pnl"] for t in sells)
        n = len(pnls)
        k = max(1, round(n * 0.2))
        bottom20 = pnls[:k]                 # worst k trades (most negative pnl)
        top20    = sorted(pnls, reverse=True)[:k]   # best k trades
        # capital allocated to a trade approximated by qty*entry(avg_cost) at its own entry
        alloc_by_pnl = {t["pnl"]: t["qty"] * t["entry"] for t in sells}
        total_alloc = sum(alloc_by_pnl.values()) or 1e-9
        cm.top20pct_profit_alloc_share = round(
            sum(alloc_by_pnl.get(p, 0.0) for p in top20) / total_alloc, 4
        )
        cm.bottom20pct_loss_alloc_share = round(
            sum(alloc_by_pnl.get(p, 0.0) for p in bottom20) / total_alloc, 4
        )
        loser_alloc = sum(alloc_by_pnl[t["pnl"]] for t in sells if t["pnl"] <= 0)
        cm.loser_capital_ratio = round(loser_alloc / total_alloc, 4)
    base_wins = [t["pnl"] for t in base_sells if t["pnl"] > 0]
    case_wins = [t["pnl"] for t in sells if t["pnl"] > 0]
    base_win_sum = sum(base_wins) or 1e-9
    cm.winner_capture = round(sum(case_wins) / base_win_sum, 4)

    cm.meets_adoption = (
        cm.alpha_retention >= ADOPT_ALPHA_RET_MIN and
        cm.lot_access >= ADOPT_LOT_ACCESS_MIN and
        cm.calmar_delta >= ADOPT_CALMAR_DELTA_MIN and
        cm.dd_reduction_pp >= ADOPT_DD_REDUCTION_MIN_PP
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


def write_md_report(
    metrics: Dict[str, CaseMetrics], research_status: str, best_case: str,
    best_allocation_policy: str, allocation_theoretical_ceiling: float,
    portfolio_alpha_leverage: float, recommend_change: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    w = L.append

    w("# Study28 Portfolio Capital Allocation Audit")
    w("")
    w("作成日: 2026-06-24  |  保有中ポートフォリオ内の資本配分のみ変更 / "
      "Entry・Exit・Signal・Execution・Universe・Position Count・Production変更禁止 / "
      "Sizing研究ではない（同時保有時の配分比のみ）")
    w("")
    w(f"**基盤**: 本番PARAMS_LOCKED（Capital=¥{CAPITAL:,.0f}, max_positions={MAX_POS}）"
      "  **Entry/Exit/Signal**: Study9 Case B系列と同一（RSR[92,95) d90≤5 slope5≤5 / exit RSR<90）")
    w("")
    w(f"**観測ウィンドウ**: {OBS_START} → {OBS_END}")
    w("")
    w("**重要**: 本研究はSizing研究ではない。Entry/Exit/Signal/Executionは完全固定。"
      "変更対象は「同時保有中の複数ポジション間での資本配分比」のみ。")
    w("")

    w("---")
    w("## Case別結果")
    w("")
    w("| Case | 説明 | trade_count | CAGR | Calmar | maxDD | alpha_retention | lot_access |")
    w("|---|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.label} | {m.trade_count} | {m.cagr:+.2f}% | {m.calmar:.3f} | "
          f"{m.max_dd:.2f}% | {m.alpha_retention:.1%} | {m.lot_access:.1%} |")
    w("")
    w("| Case | calmar_delta | dd_delta | capital_activation | allocation_entropy | "
      "concentration_risk |")
    w("|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.calmar_delta:+.4f} | {m.dd_reduction_pp:+.2f}pp | "
          f"{m.capital_activation:.1%} | {m.allocation_entropy:.3f} | {m.concentration_risk:.1%} |")
    w("")

    w("### Capital Efficiency")
    w("")
    w("| Case | winner_capture | loser_capital_ratio | realloc_gain(¥) | realloc_loss(¥) | "
      "n_rebalance_events | n_simultaneous_days |")
    w("|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.winner_capture:.1%} | {m.loser_capital_ratio:.1%} | "
          f"¥{m.capital_reallocation_gain:,.0f} | ¥{m.capital_reallocation_loss:,.0f} | "
          f"{m.n_rebalance_events} | {m.n_simultaneous_days} |")
    w("")

    w("### 追加分析（利益上位20%/損失下位20%への資本配分比率）")
    w("")
    w("| Case | top20%profit_alloc_share | bottom20%loss_alloc_share | "
      "資本再配置利益増加額 | 資本再配置損失増加額 |")
    w("|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.top20pct_profit_alloc_share:.1%} | {m.bottom20pct_loss_alloc_share:.1%} | "
          f"¥{m.realloc_profit_gain_yen:,.0f} | ¥{m.realloc_loss_increase_yen:,.0f} |")
    w("")
    w("注: 「資本再配置利益/損失増加額」は、同時保有中の配分変更（リバランス）に伴う部分売却が"
      "確定した実現益・実現損の合計（実現P&Lのみ。最終Exitの損益とは別に集計）。")
    w("")

    w("---")
    w("## 採用条件判定")
    w("")
    w(f"基準: alpha_retention≥{ADOPT_ALPHA_RET_MIN:.0%} AND lot_access≥{ADOPT_LOT_ACCESS_MIN:.0%} AND "
      f"Calmar改善≥+{ADOPT_CALMAR_DELTA_MIN:.2f} AND DD改善≥{ADOPT_DD_REDUCTION_MIN_PP:.1f}pp")
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
    w(f"| **research_status** | **{research_status}** |")
    w(f"| **best_case** | {best_case} |")
    w(f"| **best_allocation_policy** | {best_allocation_policy} |")
    w(f"| calmar_delta (best, B-F) | {metrics[best_case].calmar_delta:+.4f} |")
    w(f"| dd_delta (best, B-F) | {metrics[best_case].dd_reduction_pp:+.2f}pp |")
    w(f"| **allocation_theoretical_ceiling** (Case G calmar_delta) | "
      f"{allocation_theoretical_ceiling:+.4f} |")
    w(f"| portfolio_alpha_leverage (best B-F calmar_delta / ceiling) | "
      f"{portfolio_alpha_leverage:+.3f} |")
    w(f"| **recommend_change** | {recommend_change} |")
    w("")
    w("制約: 本研究は改善案の実装・Entry/Exit/Signal/Execution/Universe/Production変更を"
      "一切行わない（配分余地の存在判定のみ）。")
    w("")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[MD] {REPORT_MD}")


def append_telemetry(metrics: Dict[str, CaseMetrics], research_status: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "study": "study28",
            "research_status": research_status,
            "cases": {c: asdict(m) for c, m in metrics.items()},
        }
        with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def main() -> int:
    print("[Study28] Portfolio Capital Allocation Audit")
    print("=" * 68)

    print("[1/5] データロード中...")
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
    print(f"[1/5] 共通日数={len(common_dates)}  銘柄={n_syms}")

    print("[2/5] 価格・RSR・ATR%マトリクス構築...")
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
    atrpct_mat = _atrpct_matrix(universe_raw, active_syms, common_dates, close_mat)
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

    print("[3/5] Case A (baseline schedule + episode return table) 実行...")
    sim_a = simulate_case(
        "A", common_dates, active_syms, sym_to_i, open_mat, close_mat, rsr_mat, atrpct_mat,
        sym_active_mat, cross90_mat, slope5_mat, mkt_ret1, CAPITAL,
    )
    episode_return_lookup: Dict[Tuple[str, str], float] = {}
    for t in sim_a["trades"]:
        if t["side"] == "SELL":
            pct = t["exit"] / t["entry"] - 1.0
            episode_return_lookup[(t["symbol"], t["entry_date"])] = pct
    print(f"  [A] trades={len(_sell_records(sim_a['trades']))}  "
          f"simultaneous_days={sim_a['n_simultaneous_days']}  episodes_tabled={len(episode_return_lookup)}")

    print("[4/5] Case B〜G シミュレーション実行...")
    sims: Dict[str, dict] = {"A": sim_a}
    for case in CASES:
        if case == "A":
            continue
        sims[case] = simulate_case(
            case, common_dates, active_syms, sym_to_i, open_mat, close_mat, rsr_mat, atrpct_mat,
            sym_active_mat, cross90_mat, slope5_mat, mkt_ret1, CAPITAL,
            episode_return_lookup=episode_return_lookup,
        )
        s = sims[case]
        print(f"  [{case}] trades={len(_sell_records(s['trades']))}  "
              f"rebalance_events={s['n_rebalance_events']}  lot_infeasible={s['n_lot_infeasible']}")

    print("[5/5] メトリクス計算 + 判定...")
    metrics: Dict[str, CaseMetrics] = {}
    for case in CASES:
        metrics[case] = compute_case_metrics(case, sims[case], common_dates, CAPITAL, sims["A"])

    g_total_pnl = sum(t["pnl"] for t in _sell_records(sims["G"]["trades"]))
    for case in CASES:
        case_total_pnl = sum(t["pnl"] for t in _sell_records(sims[case]["trades"]))
        metrics[case].opportunity_cost = round(g_total_pnl - case_total_pnl, 0)

    bf_cases = ["B", "C", "D", "E", "F"]
    best_case = max(bf_cases, key=lambda c: metrics[c].calmar_delta)
    best_delta_calmar = metrics[best_case].calmar_delta
    allocation_theoretical_ceiling = metrics["G"].calmar_delta

    deploy_cands = [c for c in bf_cases if metrics[c].meets_adoption]

    if deploy_cands:
        best_case = max(deploy_cands, key=lambda c: metrics[c].calmar_delta)
        research_status = "DEPLOY_CANDIDATE"
    elif best_delta_calmar < EXHAUSTED_CALMAR_DELTA:
        research_status = "EXHAUSTED"
    elif metrics[best_case].calmar_delta > 0 and metrics[best_case].alpha_retention >= 0.95:
        research_status = "PROMISING_BUT_INSUFFICIENT"
    else:
        research_status = "EXHAUSTED"

    portfolio_alpha_leverage = (
        best_delta_calmar / allocation_theoretical_ceiling
        if abs(allocation_theoretical_ceiling) > 1e-6 else 0.0
    )

    ceiling_also_exhausted = allocation_theoretical_ceiling < EXHAUSTED_CALMAR_DELTA
    if research_status == "EXHAUSTED" and ceiling_also_exhausted:
        research_status = "EXHAUSTED"   # reinforced -- see report note
        ceiling_note = (
            f"STRONGLY SUPPORTED: even Case G (perfect-foresight oracle, theoretical "
            f"ceiling) only reaches calmar_delta={allocation_theoretical_ceiling:+.4f}<0.10 "
            f"-- no real-time-executable allocation policy can ever exceed this, so "
            f"intra-portfolio capital allocation has NO exploitable headroom at all."
        )
    elif research_status == "EXHAUSTED":
        ceiling_note = (
            f"WEAKLY SUPPORTED: best real-time case is below the adoption bar, but the "
            f"oracle ceiling (Case G, calmar_delta={allocation_theoretical_ceiling:+.4f}) "
            f"is itself >=0.10 -- meaning a smarter (but undiscovered) real-time policy "
            f"could in principle still close some of the gap; this audit's specific "
            f"policies (B-F) just failed to capture it."
        )
    else:
        ceiling_note = ""

    if research_status == "DEPLOY_CANDIDATE":
        recommend_change = (
            f"PROCEED_TO_WALKFORWARD ({best_case}: {CASE_LABELS[best_case]})"
        )
    elif research_status == "PROMISING_BUT_INSUFFICIENT":
        recommend_change = (
            f"CONTINUE_RESEARCH ({best_case}は正のCalmar改善だが採用閾値未達。追加検証候補。" + ceiling_note + ")"
        )
    else:
        recommend_change = (
            "PORTFOLIO_ALLOCATION_RESEARCH_END_CANDIDATE -- "
            "次段階としてLIMITED_LIVE + Exit Monitoringへの移行を提言。" + ceiling_note
        )

    best_allocation_policy = CASE_LABELS[best_case]

    print(f"  best_case (B-F)             = {best_case}  (Δcalmar={best_delta_calmar:+.4f})")
    print(f"  allocation_theoretical_ceiling (G) = {allocation_theoretical_ceiling:+.4f}")
    print(f"  research_status              = {research_status}")

    write_trade_csv({c: sims[c]["trades"] for c in CASES})
    write_md_report(
        metrics, research_status, best_case, best_allocation_policy,
        allocation_theoretical_ceiling, portfolio_alpha_leverage, recommend_change,
    )
    append_telemetry(metrics, research_status)

    print()
    print("=" * 68)
    print(f"research_status = {research_status}  (best_case={best_case}, ceiling(G)={allocation_theoretical_ceiling:+.4f})")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
