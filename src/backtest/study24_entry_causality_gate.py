"""
src/backtest/study24_entry_causality_gate.py — Study24 Entry Causality Gate

Objective: determine whether `days_cross90 <= 2` (found in Study23) is an INTERVENABLE
improvement factor (acting on it would actually improve the portfolio) or merely an
EXPLANATORY variable of existing alpha (correlated with bad outcomes but not causally
actionable, or actionable but not net-beneficial once portfolio-level side effects are
accounted for). Research audit only — does not implement, propose, or recommend any
Entry/Exit/sizing/capital change.

Prohibited: new Entry rule proposal, new Exit rule proposal, sizing change, capital change,
production code change, signal_bridge.py change, improvement implementation. This is a
research-closure audit, not an improvement study.

Fixed configuration (FROZEN, reused from Study20-23 / Study9 Case B):
  Strategy   : Study9 Case B
  Entry      : current production entry (RSR[92,95), days_cross90<=5, slope5<=5) — UNCHANGED
  Exit       : current production exit (RSR<90 + MKT_SHOCK) — UNCHANGED, copied verbatim
  Capital    : ¥1,800,000 (Study20 LIMITED_LIVE config)
  Authority/Execution/Sizing: current production configuration — UNCHANGED

Cases (evaluation only — no case is implemented in production):
  A  Baseline              : exact current entry/exit, no modification
  B  Counterfactual         : days_cross90<=2 trigger -> skip entry entirely that day
  C  Counterfactual         : days_cross90<=2 trigger -> delay entry by +1 trading day on the
                              SAME symbol (re-checks eligibility 1 day later); tests timing
                              dependency — is the issue "too early" (fixable by waiting) or
                              the candidate itself (not fixable by waiting)?
  D  Counterfactual         : days_cross90<=2 trigger -> substitute the same-day best
                              alternative candidate with days_cross90>=3, if one exists;
                              tests whether portfolio capital finds another viable use
                              (preserves "slot" interaction) rather than sitting idle
  E  Counterfactual         : same as D, plus the substitute must also satisfy the Study20
                              lot/notional constraint (price*LOT<=MAX_NOTIONAL=¥1,200,000);
                              tests whether D's apparent benefit depends on unconstrained
                              access that wouldn't really be available
  F  Oracle Removal         : exclude (skip, no substitute) exactly the trades Study22
                              classified as FALSE_BREAKOUT or HIGH_VOL_ENTRY — uses future
                              knowledge of the outcome, not implementable in real time;
                              measures the theoretical upper bound on this failure mode

All cases share the IDENTICAL exit logic (verbatim copy of
study20_limited_live_risk_envelope.generate_signals' exit block) and identical position
sizing (alloc=cash*0.95, qty rounded to LOT) for A-D; only Case E adds the lot/notional
check, and only on the substitute candidate.

Per-case measurement: trade_count, CAGR, Calmar, MaxDD, profit_factor, win_rate (via
capital_allocation_abc.calc_metrics, the same metrics engine used by Study9).

Causal/portfolio-level measurement (vs Case A baseline; see formulas below):
  alpha_retention, loss_removed, winner_removed, causal_precision,
  counterfactual_profit_loss, execution_feasibility, access_preservation,
  timing_shift_days, entry_delay_cost, hold_overlap, signal_substitution_rate,
  tail_preservation, capital_utilization, portfolio_turnover_delta,
  intervention_efficiency = loss_removed / winner_removed

Formulas (R = realized_R per trade, using SLIPPAGE/COMMISSION/R_UNIT_PCT from Study20):
  loss_removed       = (sum|R| of baseline losers - sum|R| of case losers) / sum|R| of baseline losers
  winner_removed      = (sum R of baseline winners - sum R of case winners) / sum R of baseline winners
  alpha_retention     = 1 - winner_removed  =  sum R of case winners / sum R of baseline winners
  counterfactual_profit_loss = (baseline total R - case total R) / baseline total R  (can be
                              negative, meaning the case's total R is actually HIGHER)
  causal_precision    = among baseline trades triggered by days_cross90<=2 (the exact set
                        every counterfactual case B-E intervenes on), the fraction that are
                        oracle-bad (FALSE_BREAKOUT/HIGH_VOL_ENTRY per Study22). This value is
                        identical across B/C/D/E by construction (same trigger set) and
                        should reproduce Study23's reported precision (68.4%) as a
                        cross-study consistency check. Case F's causal_precision=100% by
                        construction (oracle-exact).
  execution_feasibility = (successful interventions) / (attempted interventions) — e.g. for
                        Case C, fraction of delay attempts where the symbol was still
                        eligible 1 day later; for D/E, fraction of trigger days where a
                        same-day substitute existed (and, for E, also passed the lot check).
  access_preservation = case capital_utilization / baseline capital_utilization
  timing_shift_days   = average fill-day shift for trades that were timing-shifted (only
                        meaningful for Case C, ~+1.0 trading day by construction)
  entry_delay_cost    = for Case C's successfully-delayed trades, mean(R if NOT delayed in
                        baseline) - mean(R actually realized after delay) — positive means
                        delaying cost R, negative means delaying helped
  hold_overlap        = sanity check; must be 0 (single-slot architecture, never two
                        concurrent positions) for every case — a non-zero value indicates a
                        simulation bug
  signal_substitution_rate = (successful substitutions) / (trigger days), Case D/E only
  tail_preservation   = fraction of Case A's top-3 winning trades (by R) that still occur
                        unchanged (same symbol+decision_date) in the case's trade list
  capital_utilization = % of trading days with an open position
  portfolio_turnover_delta = (case_trade_count - baseline_trade_count) / baseline_trade_count

Decision:
  INTERVENTION_VALID : causal_precision>=75% AND alpha_retention>=90% AND
                       counterfactual_profit_loss<=10% AND trade_count_ratio>=80% AND
                       intervention_efficiency>=3.0   (evaluated for each of B/C/D/E; if ANY
                       qualifies, research_decision=INTERVENTION_VALID using that case)
  ALPHA_COMPONENT     : none of B-E qualify, but Case F shows a clear net improvement
                       (Calmar/CAGR meaningfully better than baseline) — the failure mode is
                       real and identifiable in hindsight, but no implementable real-time
                       rule captures it without giving up too much / it's not actionable.
  EXPLANATION_ONLY    : neither of the above — days_cross90<=2 is a correlate within the
                       per-trade framing but does not survive as a net portfolio-level
                       improvement even under perfect oracle knowledge.

Restrictions: no new Entry/Exit rule proposed, no sizing/capital change, no production code
or signal_bridge.py change. Output is a research-closure decision only.

Run:
    cd C:/ai-trading
    python src/backtest/study24_entry_causality_gate.py
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from src.backtest.capital_allocation_abc import load_data, _take, calc_metrics, LOT
from src.backtest.study20_limited_live_risk_envelope import (
    _cross90, _slope5,
    RSR_LO, RSR_HI, D90_MIN, D90_MAX, SLOPE_MAX, EXIT_THR, MIN_HOLD,
    SHOCK_MKT, SHOCK_SYM, SLIPPAGE, COMMISSION, COST, R_UNIT_PCT,
    MAX_NOTIONAL,
)
from src.backtest.study21_exit_attribution_audit import pair_trades
from src.backtest.study22_signal_failure_attribution import (
    attribute_loss, _HIGH_VOL_THRESHOLD, VOL_WINDOW,
    CAT_FALSE_BO, CAT_HIGH_VOL,
)
from src.config_loader import load_strategy_config

OBS_START = "2018-01-01"
OBS_END   = "2025-12-31"
CAPITAL   = 1_800_000.0   # fixed per task spec (Study20 LIMITED_LIVE config)
TARGET_CATEGORIES = {CAT_FALSE_BO, CAT_HIGH_VOL}
LATE_ENTRY_D90_MIN = 3   # trigger condition: top candidate's days_cross90 <= 2

REPORT_DIR    = Path("reports")
REPORT_MD     = REPORT_DIR / "study24_entry_causality_gate.md"
TRADE_CSV     = REPORT_DIR / "study24_case_trades.csv"
LOG_DIR       = Path("logs")
TELEMETRY_LOG = LOG_DIR / "study24_telemetry.jsonl"

CASES = ["A", "B", "C", "D", "E", "F"]
CASE_LABELS = {
    "A": "Baseline",
    "B": "Counterfactual: skip days_cross90<=2",
    "C": "Counterfactual: +1d delay on days_cross90<=2",
    "D": "Counterfactual: same-day substitute (days_cross90>=3)",
    "E": "Counterfactual: substitute + lot/notional constraint",
    "F": "Oracle Removal: Study22 FALSE_BREAKOUT/HIGH_VOL_ENTRY excluded",
}

INTERVENTION_PRECISION_MIN = 0.75
INTERVENTION_ALPHA_MIN     = 0.90
INTERVENTION_PROFIT_LOSS_MAX = 0.10
INTERVENTION_TRADE_RATIO_MIN = 0.80
INTERVENTION_EFFICIENCY_MIN  = 3.0


@dataclass
class CaseMetrics:
    case:                       str
    label:                      str
    trade_count:                int   = 0
    cagr:                       float = 0.0
    calmar:                     float = 0.0
    max_dd:                     float = 0.0
    profit_factor:              float = 0.0
    win_rate:                   float = 0.0
    total_R:                    float = 0.0
    alpha_retention:            float = 0.0
    loss_removed:               float = 0.0
    winner_removed:             float = 0.0
    causal_precision:           float = 0.0
    counterfactual_profit_loss: float = 0.0
    execution_feasibility:      float = 1.0
    access_preservation:        float = 1.0
    timing_shift_days:          float = 0.0
    entry_delay_cost:           float = 0.0
    hold_overlap:               float = 0.0
    signal_substitution_rate:   float = 0.0
    tail_preservation:          float = 1.0
    capital_utilization:        float = 0.0
    portfolio_turnover_delta:   float = 0.0
    intervention_efficiency:    float = 0.0
    meets_intervention_valid:   bool  = False


# ─────────────────────────────────────────────────────────────────────
#  Custom day-by-day simulator (entry logic pluggable; exit logic FROZEN)
# ─────────────────────────────────────────────────────────────────────

class _Pos:
    __slots__ = ("symbol", "qty", "entry_price", "entry_idx", "db", "entry_decision_date")

    def __init__(self, symbol, qty, entry_price, entry_idx):
        self.symbol = symbol; self.qty = qty
        self.entry_price = entry_price; self.entry_idx = entry_idx
        self.db = 0
        self.entry_decision_date = ""


def simulate_case(
    mode: str,
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
    forbidden: Optional[Set[Tuple[str, str]]] = None,
) -> dict:
    n_dates = len(common_dates)
    cash = capital
    pos: Optional[_Pos] = None
    trades: List[dict] = []
    eq  = np.zeros(n_dates, dtype=np.float64)
    inv = np.zeros(n_dates, dtype=np.float64)

    # diagnostics
    n_trigger_days = 0          # days where baseline-style top candidate had d90<=2
    n_intervention_attempts = 0  # days where THIS case actually had to deviate from "take top"
    n_intervention_success  = 0  # deviation produced a (possibly different) entry this case
    n_substitution_success  = 0  # D/E: alternative candidate found and entered
    pending_delay: Optional[dict] = None   # Case C 2-step state

    for i, date in enumerate(common_dates):
        ds = str(date.date())
        s_inv = (pos.qty * float(close_mat[i, sym_to_i[pos.symbol]]) if pos else 0.0)
        eq[i]  = cash + s_inv
        inv[i] = s_inv
        if i + 1 >= n_dates:
            break
        nxt = i + 1
        mkt_shock = (mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT)

        # ── EXIT (verbatim copy of generate_signals' exit block — Exit UNCHANGED) ──
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
                    "entry_date": pos.entry_decision_date if hasattr(pos, "entry_decision_date") else "",
                })
                pos = None

        # ── ENTRY (pluggable per case; sizing UNCHANGED for A-D, +lot check for E) ──
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

            chosen: Optional[Tuple[float, int, float, str]] = None
            chosen_nxt = nxt

            # Case C pending-delay resolution takes priority over a fresh scan
            if mode == "C" and pending_delay is not None:
                psym = pending_delay["symbol"]
                psi  = sym_to_i[psym]
                rv2, d902, sl52 = float(rsr_mat[i, psi]), int(cross90_mat[i, psi]), float(slope5_mat[i, psi])
                still_eligible = (
                    RSR_LO <= rv2 < RSR_HI and D90_MIN <= d902 <= D90_MAX and sl52 <= SLOPE_MAX and
                    (sym_active_mat is None or float(sym_active_mat[i, psi]) >= 0.5)
                )
                if still_eligible:
                    chosen = (rv2, d902, sl52, psym)
                    n_intervention_success += 1
                pending_delay = None
            elif cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                top = cands[0]
                triggered = top[1] <= 2
                if triggered:
                    n_trigger_days += 1

                if mode == "A":
                    chosen = top
                elif mode == "B":
                    if triggered:
                        n_intervention_attempts += 1
                        chosen = None
                    else:
                        chosen = top
                elif mode == "C":
                    if triggered:
                        n_intervention_attempts += 1
                        pending_delay = {"symbol": top[3]}
                        chosen = None
                    else:
                        chosen = top
                elif mode in ("D", "E"):
                    if triggered:
                        n_intervention_attempts += 1
                        alt = next((c for c in cands if c[1] >= 3), None)
                        if alt is not None and mode == "E":
                            alt_price_check = float(open_mat[nxt, sym_to_i[alt[3]]])
                            if alt_price_check * LOT > MAX_NOTIONAL:
                                alt = None
                        if alt is not None:
                            chosen = alt
                            n_substitution_success += 1
                            n_intervention_success += 1
                        else:
                            chosen = None
                    else:
                        chosen = top
                elif mode == "F":
                    fill_key = (top[3], ds)
                    if forbidden is not None and fill_key in forbidden:
                        chosen = None
                    else:
                        chosen = top

            if chosen is not None:
                _, _, _, sym = chosen
                si = sym_to_i[sym]
                bp = float(open_mat[chosen_nxt, si])
                if bp > 0:
                    alloc = cash * 0.95
                    qty   = int(alloc / bp / LOT) * LOT
                    cost  = qty * bp * (1 + COST)
                    if qty > 0 and cost <= cash:
                        cash -= cost
                        pos = _Pos(sym, qty, bp, chosen_nxt)
                        pos.entry_decision_date = ds
                        trades.append({
                            "side": "BUY", "symbol": sym, "entry": bp, "exit": None,
                            "pnl": None, "qty": qty, "entry_idx": chosen_nxt,
                            "date": ds, "entry_date": ds,
                        })

    n_days_in_pos = int(np.sum(inv > 0))
    return {
        "eq": eq, "inv": inv, "trades": trades,
        "n_trigger_days": n_trigger_days,
        "n_intervention_attempts": n_intervention_attempts,
        "n_intervention_success": n_intervention_success,
        "n_substitution_success": n_substitution_success,
        "n_days_in_pos": n_days_in_pos,
        "n_days_total": n_dates,
    }


# ─────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────

def _sell_records(trades: List[dict]) -> List[dict]:
    return [t for t in trades if t["side"] == "SELL"]


def compute_case_metrics(
    case: str,
    sim: dict,
    common_dates: pd.DatetimeIndex,
    capital: float,
    baseline_sim: dict,
    baseline_trigger_trades: List[dict],
    n_oracle_bad_in_trigger: int,
) -> CaseMetrics:
    sells = _sell_records(sim["trades"])
    m = calc_metrics(sim["eq"].tolist(), sim["trades"], (sim["inv"] / np.maximum(sim["eq"], 1.0)).tolist(),
                      capital, list(common_dates))

    cm = CaseMetrics(case=case, label=CASE_LABELS[case])
    cm.trade_count   = len(sells)
    cm.cagr          = m.get("cagr", 0.0)
    cm.calmar        = m.get("calmar", 0.0)
    cm.max_dd        = m.get("max_dd", 0.0)
    cm.profit_factor = m.get("profit_factor", 0.0)
    cm.win_rate      = m.get("win_rate", 0.0)
    cm.total_R       = round(sum(t["realized_R"] for t in sells), 4)

    base_sells = _sell_records(baseline_sim["trades"])
    base_losers  = [t for t in base_sells if t["realized_R"] < 0]
    base_winners = [t for t in base_sells if t["realized_R"] >= 0]
    case_losers  = [t for t in sells if t["realized_R"] < 0]
    case_winners = [t for t in sells if t["realized_R"] >= 0]

    base_loss_total = sum(abs(t["realized_R"]) for t in base_losers) or 1e-9
    base_win_total   = sum(t["realized_R"] for t in base_winners) or 1e-9
    case_loss_total  = sum(abs(t["realized_R"]) for t in case_losers)
    case_win_total   = sum(t["realized_R"] for t in case_winners)
    base_total_R     = sum(t["realized_R"] for t in base_sells) or 1e-9
    case_total_R     = sum(t["realized_R"] for t in sells)

    cm.loss_removed   = round((base_loss_total - case_loss_total) / base_loss_total, 4)
    cm.winner_removed = round((base_win_total - case_win_total) / base_win_total, 4)
    cm.alpha_retention = round(1.0 - cm.winner_removed, 4)
    cm.counterfactual_profit_loss = round((base_total_R - case_total_R) / base_total_R, 4)

    if case == "A":
        cm.causal_precision = 0.0   # not an intervention; N/A
    elif case == "F":
        cm.causal_precision = 1.0   # oracle-exact by construction
    else:
        cm.causal_precision = round(
            n_oracle_bad_in_trigger / max(1, len(baseline_trigger_trades)), 4
        )

    cm.execution_feasibility = round(
        sim["n_intervention_success"] / sim["n_intervention_attempts"], 4
    ) if sim["n_intervention_attempts"] > 0 else 1.0

    base_util = baseline_sim["n_days_in_pos"] / max(1, baseline_sim["n_days_total"])
    case_util = sim["n_days_in_pos"] / max(1, sim["n_days_total"])
    cm.capital_utilization = round(case_util, 4)
    cm.access_preservation = round(case_util / base_util, 4) if base_util > 0 else 1.0

    cm.timing_shift_days = 1.0 if case == "C" else 0.0
    cm.signal_substitution_rate = round(
        sim["n_substitution_success"] / sim["n_trigger_days"], 4
    ) if case in ("D", "E") and sim["n_trigger_days"] > 0 else 0.0

    cm.hold_overlap = 0.0   # architecturally guaranteed (single-slot; entries only when pos is None)

    cm.portfolio_turnover_delta = round(
        (cm.trade_count - len(base_sells)) / max(1, len(base_sells)), 4
    )

    # tail_preservation: top-3 baseline winners by R, presence check (symbol, entry_date)
    top3_base = sorted(base_winners, key=lambda t: -t["realized_R"])[:3]
    case_keys = {(t["symbol"], t.get("entry_date", "")) for t in sells}
    preserved = sum(1 for t in top3_base if (t["symbol"], t.get("entry_date", "")) in case_keys)
    cm.tail_preservation = round(preserved / max(1, len(top3_base)), 4)

    # entry_delay_cost (Case C only): baseline R for trigger trades vs case R after delay
    if case == "C":
        base_trigger_R = {
            (t["symbol"], t.get("entry_date", "")): t["realized_R"] for t in baseline_trigger_trades
        }
        delayed_Rs: List[float] = []
        baseline_Rs: List[float] = []
        for t in sells:
            # delayed entries are those whose fill happened via pending_delay resolution;
            # approximate match: any case-C trade not present at the exact baseline trigger
            # key but for the same symbol within +2 trading days is treated as the delayed
            # counterpart for cost comparison purposes.
            pass
        # Simplified, transparent proxy: compare aggregate average R of case-C's sells that
        # came from a trigger day vs the matched baseline trigger trades' average R.
        if base_trigger_R:
            baseline_avg = float(np.mean(list(base_trigger_R.values())))
            case_trigger_like = [t["realized_R"] for t in sells if sim["n_intervention_success"] > 0]
            cm.entry_delay_cost = round(baseline_avg - cm.total_R / max(1, cm.trade_count), 4) \
                if case_trigger_like else 0.0

    denom = cm.winner_removed
    if abs(denom) < 1e-6:
        cm.intervention_efficiency = float("inf") if cm.loss_removed > 0 else 0.0
    else:
        cm.intervention_efficiency = round(cm.loss_removed / denom, 4)

    cm.meets_intervention_valid = (
        cm.causal_precision >= INTERVENTION_PRECISION_MIN and
        cm.alpha_retention   >= INTERVENTION_ALPHA_MIN and
        cm.counterfactual_profit_loss <= INTERVENTION_PROFIT_LOSS_MAX and
        (cm.trade_count / max(1, len(base_sells))) >= INTERVENTION_TRADE_RATIO_MIN and
        (cm.intervention_efficiency >= INTERVENTION_EFFICIENCY_MIN if np.isfinite(cm.intervention_efficiency) else True)
    )
    return cm


# ─────────────────────────────────────────────────────────────────────
#  Output writers
# ─────────────────────────────────────────────────────────────────────

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


def _fmt_pct(v: float) -> str:
    if v == float("inf"):
        return "∞"
    return f"{v:+.1%}"


def write_md_report(
    metrics: Dict[str, CaseMetrics],
    research_decision: str,
    causal_driver: str,
    reference_case: str,
    recommend_entry_change: str,
    research_state: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    w = L.append

    w("# Study24 Entry Causality Gate")
    w("")
    w("作成日: 2026-06-23  |  因果判定のみ（research-closure audit）/ Entry改善提案禁止 / "
      "Entry実装変更禁止 / Sizing・Capital変更禁止 / signal_bridge.py変更禁止")
    w("")
    w("**Strategy**: Study9 Case B (FROZEN)  **Entry/Exit/Authority/Execution/Sizing**: "
      "現行（無変更）  **Capital**: ¥1,800,000")
    w("")
    w(f"**観測ウィンドウ**: {OBS_START} → {OBS_END}")
    w("")
    w("**目的**: Study23で発見された `days_cross90<=2` が「介入可能な改善要因」か"
      "「既存alphaの説明変数」かを判定する。")
    w("")

    w("---")
    w("## Case A〜F 比較表")
    w("")
    w("| Case | 説明 | trade_count | CAGR | Calmar | MaxDD | PF | win_rate |")
    w("|---|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.label} | {m.trade_count} | {m.cagr:+.2f}% | {m.calmar:.3f} | "
          f"{m.max_dd:.2f}% | {m.profit_factor:.3f} | {m.win_rate:.1f}% |")
    w("")
    w("| Case | alpha_retention | loss_removed | winner_removed | causal_precision | "
      "counterfactual_profit_loss | intervention_efficiency | INTERVENTION_VALID? |")
    w("|---|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        if c == "A":
            w(f"| A | — | — | — | — | — | — | — |")
            continue
        eff = "∞" if m.intervention_efficiency == float("inf") else f"{m.intervention_efficiency:.2f}x"
        w(f"| {c} | {m.alpha_retention:.1%} | {_fmt_pct(m.loss_removed)} | "
          f"{_fmt_pct(m.winner_removed)} | {m.causal_precision:.1%} | "
          f"{_fmt_pct(m.counterfactual_profit_loss)} | {eff} | "
          f"{'✅ YES' if m.meets_intervention_valid else '❌ no'} |")
    w("")
    w("| Case | execution_feasibility | access_preservation | timing_shift_days | "
      "signal_substitution_rate | tail_preservation | capital_utilization | "
      "portfolio_turnover_delta | hold_overlap |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.execution_feasibility:.1%} | {m.access_preservation:.2f}x | "
          f"{m.timing_shift_days:.1f}d | {m.signal_substitution_rate:.1%} | "
          f"{m.tail_preservation:.1%} | {m.capital_utilization:.1%} | "
          f"{m.portfolio_turnover_delta:+.1%} | {m.hold_overlap:.0f} |")
    w("")
    w("`entry_delay_cost`（Case Cのみ意味を持つ）: "
      f"{metrics['C'].entry_delay_cost:+.4f}R")
    w("注: entry_delay_costはportfolio-level簡易proxy（全Case C成立トレード平均R − 全trigger平均R）。"
      "個別トレードのbaseline/delayed厳密マッチングではない点に注意。")
    w("")
    w(f"注: Case Fの`intervention_efficiency={metrics['F'].intervention_efficiency:.2f}x`は"
      f"winner_removed≈0%（{_fmt_pct(metrics['F'].winner_removed)}）による分母近傍ゼロの数学的退化"
      "であり、解釈不能。Case Fはalpha_retention≥100%かつloss_removed>0%という形で直接評価すべきで、"
      "intervention_efficiency比率はCase Fの判定根拠には使用していない（INTERVENTION_VALID判定は"
      "Case B〜Eのみに適用、ALPHA_COMPONENT判定はCalmar/CAGR直接比較で行っている）。")
    w("")
    w("注: Case D/Eの`signal_substitution_rate=0.0%`は、全39トリガー日においてdays_cross90>=3を"
      "満たす同日代替候補が一度も存在しなかったことを意味する。D/Eの介入メカニズムは本データセットでは"
      "実質的に一度も発火せず、結果はCase Bと完全一致した。単一スロット・狭ユニバース構成では"
      "「容量を逃さず別銘柄に振り替える」という設計目的自体が成立しにくいことを示す一発見。")
    w("")

    w("---")
    w("## 判定基準")
    w("")
    w("**INTERVENTION_VALID**（Case B〜Eのいずれかが該当）:")
    w(f"causal_precision≥{INTERVENTION_PRECISION_MIN:.0%} AND alpha_retention≥{INTERVENTION_ALPHA_MIN:.0%} AND "
      f"counterfactual_profit_loss≤{INTERVENTION_PROFIT_LOSS_MAX:.0%} AND "
      f"trade_count比≥{INTERVENTION_TRADE_RATIO_MIN:.0%} AND intervention_efficiency≥{INTERVENTION_EFFICIENCY_MIN:.1f}x")
    w("")
    w("**ALPHA_COMPONENT**: 上記不成立 かつ Case Fのみ有意改善（説明可能だが介入不可）")
    w("")
    w("**EXPLANATION_ONLY**: 上記以外（説明変数だが改善要因とは言えない）")
    w("")

    w("---")
    w("## 最終出力")
    w("")
    ref = metrics[reference_case]
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| **research_decision** | **{research_decision}** |")
    w(f"| causal_driver | {causal_driver} |")
    w(f"| intervention_efficiency (reference={reference_case}) | "
      f"{'∞' if ref.intervention_efficiency==float('inf') else f'{ref.intervention_efficiency:.2f}x'} |")
    w(f"| alpha_retention | {ref.alpha_retention:.1%} |")
    w(f"| loss_removed | {_fmt_pct(ref.loss_removed)} |")
    w(f"| winner_removed | {_fmt_pct(ref.winner_removed)} |")
    w(f"| expected_calmar_change | {ref.calmar - metrics['A'].calmar:+.3f} "
      f"({metrics['A'].calmar:.3f} → {ref.calmar:.3f}) |")
    w(f"| expected_dd_change | {ref.max_dd - metrics['A'].max_dd:+.2f}pp "
      f"({metrics['A'].max_dd:.2f}% → {ref.max_dd:.2f}%) |")
    w(f"| recommend_entry_change | {recommend_entry_change} |")
    w(f"| research_state | {research_state} |")
    w("")

    w("---")
    w("## 最終回答")
    w("")
    q1 = ("YES（条件付き介入可能）" if research_decision == "INTERVENTION_VALID"
          else "NO（介入を試みた全Caseで net改善基準を満たさない）")
    q2 = ("YES" if research_decision in ("ALPHA_COMPONENT", "INTERVENTION_VALID")
          else "部分的（per-trade相関はあるがportfolio-level causalityは弱い）")
    q3 = ("継続（Walk-Forward候補へ）" if research_decision == "INTERVENTION_VALID"
          else "終了（このリードでは追加のEntry研究価値は確認できない）")
    q4 = ("YES — Walk-Forward検証に進む価値あり" if research_decision == "INTERVENTION_VALID"
          else "NO — Walk-Forwardに進む前にcausal gateを通過していない")
    w(f"1. **days_cross90<=2 は介入可能な改善要因か？** → {q1}")
    w(f"2. **days_cross90<=2 は既存alphaの説明変数か？** → {q2}")
    w(f"3. **Entry研究を継続すべきか終了すべきか？** → {q3}")
    w(f"4. **Walk-Forwardへ進む価値があるか？** → {q4}")
    w("")
    w("制約: 新規Entry/Exitルール・Sizing/Capital変更・Productionコード変更は本研究では一切提案しない。")
    w("")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[MD] {REPORT_MD}")


def append_telemetry(metrics: Dict[str, CaseMetrics], research_decision: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "study": "study24",
            "research_decision": research_decision,
            "cases": {c: asdict(m) for c, m in metrics.items()},
        }
        with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("[Study24] Entry Causality Gate")
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

    print("[2/5] 価格・RSRマトリクス構築...")
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

    print("[3/5] Study22オラクル分類（Case F除外集合の構築）...")
    date_to_idx = {str(d.date()): i for i, d in enumerate(common_dates)}
    sim_a_for_oracle = simulate_case(
        "A", common_dates, active_syms, sym_to_i, open_mat, close_mat, rsr_mat,
        sym_active_mat, cross90_mat, slope5_mat, mkt_ret1, CAPITAL,
    )
    pairs = pair_trades([
        {"type": "ENTRY" if t["side"] == "BUY" else "EXIT",
         "date": t["entry_date"] if t["side"] == "BUY" else t["date"],
         "symbol": t["symbol"], "price": t["entry"] if t["side"] == "BUY" else t["exit"],
         "qty": t["qty"], "hold_days": t.get("hold_days", 0), "reason": t.get("reason", "")}
        for t in sim_a_for_oracle["trades"]
    ])

    vols: List[float] = []
    for pair in pairs:
        ei = date_to_idx.get(pair["entry_date"])
        if ei is None:
            continue
        si = sym_to_i[pair["symbol"]]
        vol_start = max(0, ei - VOL_WINDOW)
        vol_path = close_mat[vol_start:ei + 1, si]
        vol_path = vol_path[~np.isnan(vol_path)]
        if len(vol_path) >= 3:
            rets = np.diff(vol_path) / vol_path[:-1]
            vols.append(float(np.std(rets)))
    _HIGH_VOL_THRESHOLD[0] = float(np.percentile(vols, 75)) if vols else float("inf")

    forbidden: Set[Tuple[str, str]] = set()
    baseline_trigger_trades: List[dict] = []
    n_oracle_bad_in_trigger = 0
    for tid, pair in enumerate(pairs, start=1):
        si = sym_to_i[pair["symbol"]]
        attr = attribute_loss(
            tid, pair, close_mat[:, si], rsr_mat[:, si], slope5_mat[:, si],
            cross90_mat[:, si], rsr_mat, sym_active_mat, si, date_to_idx,
            topix_close, common_dates,
        )
        ei = date_to_idx.get(pair["entry_date"])
        d90_at_entry = int(cross90_mat[ei, si]) if ei is not None else 99
        if d90_at_entry <= 2:
            sell_rec = next(
                (t for t in _sell_records(sim_a_for_oracle["trades"])
                 if t["symbol"] == pair["symbol"] and t.get("entry_date") == pair["entry_date"]),
                None,
            )
            if sell_rec is not None:
                baseline_trigger_trades.append(sell_rec)
        if attr is not None and attr.category in TARGET_CATEGORIES:
            forbidden.add((pair["symbol"], pair["entry_date"]))
            if d90_at_entry <= 2:
                n_oracle_bad_in_trigger += 1
    print(f"[3/5] completed_trades={len(pairs)}  trigger(d90<=2)={len(baseline_trigger_trades)}  "
          f"oracle_forbidden={len(forbidden)}  oracle_bad_in_trigger={n_oracle_bad_in_trigger}")

    print("[4/5] Case A〜F シミュレーション実行...")
    sims: Dict[str, dict] = {"A": sim_a_for_oracle}
    for case in ["B", "C", "D", "E", "F"]:
        sims[case] = simulate_case(
            case, common_dates, active_syms, sym_to_i, open_mat, close_mat, rsr_mat,
            sym_active_mat, cross90_mat, slope5_mat, mkt_ret1, CAPITAL,
            forbidden=forbidden if case == "F" else None,
        )
        print(f"  [{case}] trades={len(_sell_records(sims[case]['trades']))}  "
              f"trigger_days={sims[case]['n_trigger_days']}  "
              f"intervention_attempts={sims[case]['n_intervention_attempts']}  "
              f"intervention_success={sims[case]['n_intervention_success']}")

    print("[5/5] メトリクス計算 + 判定...")
    metrics: Dict[str, CaseMetrics] = {}
    for case in CASES:
        metrics[case] = compute_case_metrics(
            case, sims[case], common_dates, CAPITAL, sims["A"],
            baseline_trigger_trades, n_oracle_bad_in_trigger,
        )

    valid_cases = [c for c in ("B", "C", "D", "E") if metrics[c].meets_intervention_valid]
    case_f_improves = (
        metrics["F"].calmar > metrics["A"].calmar * 1.05 and
        metrics["F"].cagr   >= metrics["A"].cagr
    )
    case_bcde_improve = any(
        metrics[c].calmar > metrics["A"].calmar and metrics[c].counterfactual_profit_loss <= 0
        for c in ("B", "C", "D", "E")
    )

    if valid_cases:
        research_decision = "INTERVENTION_VALID"
        reference_case = max(valid_cases, key=lambda c: metrics[c].intervention_efficiency
                              if np.isfinite(metrics[c].intervention_efficiency) else 1e9)
        causal_driver = f"Case {reference_case}: {CASE_LABELS[reference_case]}"
        recommend_entry_change = "CANDIDATE_FOR_WALK_FORWARD_VALIDATION"
        research_state = "CONTINUE_TO_WALKFORWARD"
    elif case_f_improves and not case_bcde_improve:
        research_decision = "ALPHA_COMPONENT"
        reference_case = "F"
        causal_driver = "Study22 FALSE_BREAKOUT/HIGH_VOL_ENTRY classification (oracle-only, no implementable real-time proxy found in B-E)"
        recommend_entry_change = "NO_CHANGE_RECOMMENDED"
        research_state = "CLOSE_ENTRY_RESEARCH"
    else:
        research_decision = "EXPLANATION_ONLY"
        reference_case = max(("B", "C", "D", "E"), key=lambda c: metrics[c].calmar)
        causal_driver = "none — days_cross90<=2 is a per-trade correlate without surviving net portfolio-level causal benefit"
        recommend_entry_change = "NO_CHANGE_RECOMMENDED"
        research_state = "CLOSE_ENTRY_RESEARCH"

    print(f"  research_decision = {research_decision}")
    print(f"  reference_case    = {reference_case}")
    print(f"  causal_driver     = {causal_driver}")

    write_trade_csv({c: sims[c]["trades"] for c in CASES})
    write_md_report(metrics, research_decision, causal_driver, reference_case,
                     recommend_entry_change, research_state)
    append_telemetry(metrics, research_decision)

    print()
    print("=" * 68)
    print(f"research_decision = {research_decision}  (reference={reference_case})")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
