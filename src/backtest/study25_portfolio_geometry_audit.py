"""
src/backtest/study25_portfolio_geometry_audit.py — Study25 Portfolio Geometry Audit

Objective: with Entry/Exit/Signal/Execution/Authority/Capital FROZEN (Study9 Case B,
¥1,800,000, max_pos=1), determine whether Calmar improvement room remains by changing
PORTFOLIO GEOMETRY ONLY (capital reserve / exposure decay / drawdown-aware admission /
volatility-aware sizing). This is a structure audit, not a signal-improvement study.
Profit maximization is explicitly out of scope; the only objective is
CAGR maintained + MaxDD reduced + Calmar improved.

Prohibited (per task spec): Entry change, Exit change, Signal change, RSR threshold
change, days_cross90 change, slope change, Execution change, Authority change, live
production code change. No improvement is implemented here — this script only measures
whether improvement room exists.

Fixed configuration (FROZEN, reused from Study9 Case B / Study20 / Study24):
  Strategy : Study9 Case B  (RSR[92,95), days_cross90<=5, slope5<=5, exit RSR<90)
  Capital  : ¥1,800,000
  max_pos  : 1 (single slot, architecturally identical to Study20/24)
  Exit     : verbatim copy of Study20/24's exit block — UNCHANGED

Cases (portfolio geometry only — entry SELECTION rule is always "top RSR candidate",
identical to baseline; only entry-time SIZE / ADMISSION differs):
  A  Baseline           : no geometry change (mult=1.0, never blocked)
  B  Capital Reserve    : always hold RESERVE_FRAC=0.20 of cash back; deployable=0.80
                          (task spec gives this constant directly: "常時20%キャッシュ保持")
  C  Exposure Decay     : new_position_size = base_size * f(rolling_dd); f=1.0 below
                          DD_DECAY_START, linearly -> 0.0 at DD_DECAY_FULL. DD_DECAY_FULL
                          is pinned to CLAUDE.md CIRCUIT max_dd_limit=0.15 so the decay
                          fully exhausts exactly at the existing circuit-breaker boundary
                          (not re-derived from this run's results). Existing holdings are
                          never resized — only the size of a NEW entry is affected.
  D  Drawdown-aware Admission : new_entry blocked entirely while rolling_dd >
                          DD_ADMISSION_BLOCK=0.10 (structural midpoint between C's decay
                          onset 0.05 and the circuit breaker 0.15). Existing holdings are
                          held to their normal (unchanged) exit logic.
  E  Volatility-aware Sizing  : new entries only — risk_unit = base_size *
                          clip(VOL_REF / realized_vol_20d, VOL_MULT_MIN, 1.0). VOL_REF and
                          VOL_MULT_MIN are fixed external constants (not fit to this run);
                          the multiplier is capped at 1.0 so volatility can only ever
                          shrink size, never lever up (profit maximization is out of
                          scope). Sizing of an already-open position never changes.
  F  Combined           : B + C + D + E stacked (admission gate from D checked first;
                          if not blocked, multiplier = reserve_mult * decay_mult * vol_mult)

All geometry constants are single fixed values chosen once from structural reasoning
(ties to existing PARAMS_LOCKED/CIRCUIT values or the task's own stated figure) — there is
no sweep and no fitting to this run's outcome, consistent with OVERFIT_GUARD
(param_sweep_limit=bounded, single_metric_optimization=forbid).

IMPORTANT (read before interpreting alpha_retention): realized_R is a per-share return
ratio, independent of position size. Therefore for sizing-only cases (B, C while
mult>0, E) alpha_retention (R-based) is ~100% BY CONSTRUCTION whenever no entry is fully
skipped — CAGR/Calmar/MaxDD changes for those cases flow entirely through dollar-weighted
capital efficiency, not through the per-trade signal. alpha_retention only drops below
100% when an entry is actually skipped (D's admission block, or C/F's decay hitting 0).

Decision rule (per task spec):
  ADOPTION  : alpha_retention>=95% AND calmar_delta(vs A)>0.15 AND
              dd_reduction_pp(vs A)>=1.0pp   (evaluated per case B-F)
  research_decision = PORTFOLIO_GEOMETRY_EXHAUSTED  if best_delta_calmar < 0.10
                     = ADOPTION_CANDIDATE_FOUND      if any case meets ADOPTION
                     = MARGINAL_IMPROVEMENT_OBSERVED otherwise (0.10<=best<adoption bar)

Run:
    cd C:/ai-trading
    python src/backtest/study25_portfolio_geometry_audit.py
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
CAPITAL   = 1_800_000.0   # FROZEN, Study9 Case B / Study20-24 config

# ── Portfolio Geometry constants (THIS STUDY ONLY — audit, not production) ─────────
RESERVE_FRAC        = 0.20   # Case B: task spec value ("常時20%キャッシュ保持")
DD_DECAY_START       = 0.05   # Case C: decay begins at rolling_dd > 5%
DD_DECAY_FULL        = 0.15   # Case C: decay reaches 0 at CIRCUIT max_dd_limit=0.15
DD_ADMISSION_BLOCK   = 0.10   # Case D: block new entries while rolling_dd > 10%
VOL_LOOKBACK         = 20     # Case E: trailing daily-return stdev window (days)
VOL_REF              = 0.025  # Case E: fixed reference daily vol (2.5%), external constant
VOL_MULT_MIN         = 0.50   # Case E: floor on size multiplier (never below 50%)

REPORT_DIR    = Path("reports")
REPORT_MD     = REPORT_DIR / "study25_portfolio_geometry_audit.md"
TRADE_CSV     = REPORT_DIR / "study25_case_trades.csv"
LOG_DIR       = Path("logs")
TELEMETRY_LOG = LOG_DIR / "study25_telemetry.jsonl"

CASES = ["A", "B", "C", "D", "E", "F"]
CASE_LABELS = {
    "A": "Baseline (現行)",
    "B": "Capital Reserve (常時20%キャッシュ保持)",
    "C": "Exposure Decay (rolling DD上昇時のみ size↓)",
    "D": "Drawdown-aware Admission (DD>10%でnew_entry blocked)",
    "E": "Volatility-aware Sizing (新規Entryのみ risk_unit=base×f(vol))",
    "F": "Combined (B+C+D+E)",
}
DECISION_COMPLEXITY = {"A": 0, "B": 1, "C": 2, "D": 1, "E": 3, "F": 7}

ADOPT_ALPHA_RET_MIN = 0.95
ADOPT_CALMAR_DELTA_MIN = 0.15
ADOPT_DD_REDUCTION_MIN_PP = 1.0
EXHAUSTED_CALMAR_DELTA = 0.10


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
    capital_activation_ratio: float = 0.0
    slot_utilization: float = 0.0
    cash_idle_ratio: float = 0.0
    exposure_utilization: float = 0.0
    recovery_days: float = 0.0
    recovery_speed: float = 0.0
    risk_adjusted_return: float = 0.0
    dd_reduction_pp: float = 0.0
    return_preservation: float = 1.0
    capital_efficiency: float = 1.0
    activation_loss_pp: float = 0.0
    opportunity_loss_R: float = 0.0
    tail_capture_change: float = 0.0
    decision_complexity: int = 0
    calmar_delta: float = 0.0
    meets_adoption: bool = False
    n_lot_infeasible: int = 0


class _Pos:
    __slots__ = ("symbol", "qty", "entry_price", "entry_idx", "db", "entry_decision_date",
                 "size_mult_at_entry")

    def __init__(self, symbol, qty, entry_price, entry_idx):
        self.symbol = symbol; self.qty = qty
        self.entry_price = entry_price; self.entry_idx = entry_idx
        self.db = 0
        self.entry_decision_date = ""
        self.size_mult_at_entry = 1.0


def _exposure_decay(dd: float) -> float:
    if dd <= DD_DECAY_START:
        return 1.0
    span = DD_DECAY_FULL - DD_DECAY_START
    return float(max(0.0, 1.0 - (dd - DD_DECAY_START) / span))


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

    n_trigger_days = 0       # days a valid top candidate existed while flat
    n_blocked_days = 0       # days geometry blocked/decayed-to-zero the entry
    n_lot_infeasible = 0     # days geometry's reduced size rounded down to qty=0 (1 LOT)
    blocked_dates: List[str] = []

    for i, date in enumerate(common_dates):
        ds = str(date.date())
        s_inv = (pos.qty * float(close_mat[i, sym_to_i[pos.symbol]]) if pos else 0.0)
        eq[i]  = cash + s_inv
        inv[i] = s_inv
        if i + 1 >= n_dates:
            break
        nxt = i + 1
        mkt_shock = (mkt_ret1 is not None and float(mkt_ret1[i]) <= SHOCK_MKT)

        # ── EXIT (verbatim copy of Study20/24's exit block — Exit UNCHANGED) ──
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

        # ── ENTRY (selection rule UNCHANGED; geometry only affects size/admission) ──
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

            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                top = cands[0]
                n_trigger_days += 1

                # rolling DD at decision time (uses eq[0..i] only — no lookahead)
                peak_so_far = float(np.max(eq[: i + 1]))
                dd_i = (peak_so_far - eq[i]) / peak_so_far if peak_so_far > 0 else 0.0

                blocked = False
                mult = 1.0
                if case in ("D", "F") and dd_i > DD_ADMISSION_BLOCK:
                    blocked = True
                if not blocked:
                    if case in ("B", "F"):
                        mult *= (1.0 - RESERVE_FRAC)
                    if case in ("C", "F"):
                        mult *= _exposure_decay(dd_i)
                    if case in ("E", "F"):
                        mult *= _vol_mult(close_mat, sym_to_i[top[3]], i)
                    if mult <= 1e-6:
                        blocked = True

                if blocked:
                    n_blocked_days += 1
                    blocked_dates.append(ds)
                    chosen = None
                else:
                    chosen = top
                    chosen_mult = mult
            else:
                chosen = None

            if chosen is not None:
                _, _, _, sym = chosen
                si = sym_to_i[sym]
                bp = float(open_mat[nxt, si])
                if bp > 0:
                    alloc = cash * 0.95 * chosen_mult
                    qty   = int(alloc / bp / LOT) * LOT
                    cost  = qty * bp * (1 + COST)
                    if qty <= 0:
                        n_lot_infeasible += 1
                    if qty > 0 and cost <= cash:
                        cash -= cost
                        pos = _Pos(sym, qty, bp, nxt)
                        pos.entry_decision_date = ds
                        pos.size_mult_at_entry = chosen_mult
                        trades.append({
                            "side": "BUY", "symbol": sym, "entry": bp, "exit": None,
                            "pnl": None, "qty": qty, "entry_idx": nxt,
                            "date": ds, "entry_date": ds, "size_mult": chosen_mult,
                        })

    n_days_in_pos = int(np.sum(inv > 0))
    return {
        "eq": eq, "inv": inv, "trades": trades,
        "n_trigger_days": n_trigger_days,
        "n_blocked_days": n_blocked_days,
        "n_lot_infeasible": n_lot_infeasible,
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
    cm.risk_adjusted_return = cm.sharpe
    cm.decision_complexity = DECISION_COMPLEXITY[case]
    cm.n_lot_infeasible = sim["n_lot_infeasible"]

    base_sells = _sell_records(baseline_sim["trades"])
    base_total_R = sum(t["realized_R"] for t in base_sells) or 1e-9
    case_total_R = sum(t["realized_R"] for t in sells)
    cm.alpha_retention = round(case_total_R / base_total_R, 4)

    avg_exp_frac = m.get("avg_exposure", 0.0) / 100.0
    cm.capital_activation_ratio = round(avg_exp_frac, 4)
    cm.cash_idle_ratio = round(1.0 - avg_exp_frac, 4)
    cm.slot_utilization = round(sim["n_days_in_pos"] / max(1, sim["n_days_total"]), 4)
    cm.exposure_utilization = round(
        avg_exp_frac / cm.slot_utilization, 4
    ) if cm.slot_utilization > 0 else 0.0

    rec_days, rec_speed, _ = _recovery_stats(sim["eq"])
    cm.recovery_days = round(rec_days, 1)
    cm.recovery_speed = round(rec_speed, 5)

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

    cm.activation_loss_pp = round((base_act - avg_exp_frac) * 100, 2)

    cm.opportunity_loss_R = round(
        sum(baseline_entry_R_by_date.get(d, 0.0) for d in sim["blocked_dates"]), 4
    )

    base_winners = [t for t in base_sells if t["realized_R"] >= 0]
    top3_base = sorted(base_winners, key=lambda t: -t["realized_R"])[:3]
    case_keys = {(t["symbol"], t.get("entry_date", "")) for t in sells}
    preserved = sum(1 for t in top3_base if (t["symbol"], t.get("entry_date", "")) in case_keys)
    tail_capture = preserved / max(1, len(top3_base))
    cm.tail_capture_change = round(tail_capture - 1.0, 4)   # vs baseline's trivial 100%

    cm.calmar_delta = round(cm.calmar - base_m.get("calmar", 0.0), 4)
    cm.meets_adoption = (
        cm.alpha_retention >= ADOPT_ALPHA_RET_MIN and
        cm.calmar_delta > ADOPT_CALMAR_DELTA_MIN and
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
    metrics: Dict[str, CaseMetrics],
    research_decision: str,
    best_case: str,
    best_delta_calmar: float,
    final_research_state: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    w = L.append

    w("# Study25 Portfolio Geometry Audit")
    w("")
    w("作成日: 2026-06-23  |  ポートフォリオ構造監査のみ（research audit）/ "
      "Entry・Exit・Signal・RSR閾値・days_cross90・slope・Execution・Authority変更禁止 / "
      "改善実装禁止（改善余地の存在判定のみ）")
    w("")
    w("**Strategy**: Study9 Case B (FROZEN)  **Entry/Exit/Signal/Execution/Authority**: "
      "現行（無変更）  **Capital**: ¥1,800,000  **max_pos**: 1")
    w("")
    w(f"**観測ウィンドウ**: {OBS_START} → {OBS_END}（Study24と同一）")
    w("")
    w("**目的**: 既存alphaを変更せず、資本配置とリスク配分（Portfolio Geometry）だけで "
      "CAGR維持 + MaxDD削減 + Calmar改善の余地が残るかを判定する。利益最大化は目的外。")
    w("")

    w("---")
    w("## Case別結果")
    w("")
    w("| Case | 説明 | trade_count | CAGR | Calmar | MAR | maxDD | alpha_retention | "
      "capital_activation_ratio |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.label} | {m.trade_count} | {m.cagr:+.2f}% | {m.calmar:.3f} | "
          f"{m.mar:.3f} | {m.max_dd:.2f}% | {m.alpha_retention:.1%} | "
          f"{m.capital_activation_ratio:.1%} |")
    w("")
    w("| Case | slot_utilization | cash_idle_ratio | exposure_utilization | "
      "recovery_days | recovery_speed | risk_adjusted_return |")
    w("|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.slot_utilization:.1%} | {m.cash_idle_ratio:.1%} | "
          f"{m.exposure_utilization:.1%} | {m.recovery_days:.1f}d | "
          f"{m.recovery_speed:.4f}/d | {m.risk_adjusted_return:.3f} |")
    w("")
    w("### 追加監査")
    w("")
    w("| Case | DD_reduction | return_preservation | capital_efficiency | "
      "activation_loss | opportunity_loss(R) | tail_capture_change | decision_complexity | "
      "lot_infeasible_days |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in CASES:
        m = metrics[c]
        w(f"| {c} | {m.dd_reduction_pp:+.2f}pp | {m.return_preservation:.2f}x | "
          f"{m.capital_efficiency:.2f}x | {m.activation_loss_pp:+.2f}pp | "
          f"{m.opportunity_loss_R:+.4f}R | {m.tail_capture_change:+.1%} | "
          f"{m.decision_complexity} | {m.n_lot_infeasible} |")
    w("")
    w("注（lot_infeasible_days）: geometryによる縮小サイズが1単元(LOT)未満に切り下げられ、"
      "entryが事実上スキップされた日数。Case C/D/Fでtrade_countが大きく減少しているのは、"
      "サイズ縮小が高価格銘柄で1単元購入不能になる「lot feasibility」問題（Study14既知の論点）と"
      "連動して発生しており、単純なdrawdown抑制効果だけでなく執行制約も同時に効いている。")
    w("")
    w("注: alpha_retentionはR単位（1株あたりリターン比率）であり、ポジションサイズに非依存。"
      "従って Case B/C/E（資金量のみ変更・entryをスキップしない限り）はalpha_retention≈100%が"
      "構造的に当然となる — CAGR/Calmar/maxDDの変化は全てdollar-weightedなcapital_efficiency経由で"
      "発生しており、シグナル自体の変化ではない。alpha_retentionが95%を下回るのはentryが実際に"
      "スキップされた場合（Case Dのadmission block、またはCase C/Fのdecay=0到達時）のみ。")
    w("")

    w("---")
    w("## 採用条件判定")
    w("")
    w(f"基準: alpha_retention≥{ADOPT_ALPHA_RET_MIN:.0%} AND "
      f"Calmar > baseline+{ADOPT_CALMAR_DELTA_MIN} AND maxDD ≤ baseline-{ADOPT_DD_REDUCTION_MIN_PP}pp")
    w("")
    w("| Case | calmar_delta | dd_delta | meets_adoption |")
    w("|---|---|---|---|")
    for c in CASES:
        if c == "A":
            continue
        m = metrics[c]
        w(f"| {c} | {m.calmar_delta:+.4f} | {m.dd_reduction_pp:+.2f}pp | "
          f"{'✅ YES' if m.meets_adoption else '❌ no'} |")
    w("")

    w("---")
    w("## 最終出力")
    w("")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| **best_case** | {best_case} |")
    w(f"| geometry_effect | {CASE_LABELS[best_case]} |")
    w(f"| calmar_delta (best) | {best_delta_calmar:+.4f} |")
    w(f"| dd_delta (best) | {metrics[best_case].dd_reduction_pp:+.2f}pp |")
    w(f"| **research_decision** | **{research_decision}** |")
    w(f"| recommend_change | "
      f"{'NO_CHANGE_RECOMMENDED' if research_decision != 'ADOPTION_CANDIDATE_FOUND' else f'CANDIDATE_FOR_WALK_FORWARD ({best_case})'} |")
    w(f"| final_research_state | {final_research_state} |")
    w("")
    w("制約: 本研究は改善案の実装・Entry/Exit/Signal/Production変更を一切行わない"
      "（改善余地の存在判定のみ）。")
    w("")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[MD] {REPORT_MD}")


def append_telemetry(metrics: Dict[str, CaseMetrics], research_decision: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "study": "study25",
            "research_decision": research_decision,
            "cases": {c: asdict(m) for c, m in metrics.items()},
        }
        with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def main() -> int:
    print("[Study25] Portfolio Geometry Audit")
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
    print(f"[1/4] 共通日数={len(common_dates)}  銘柄={n_syms}")

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
        print(f"  [{case}] trades={len(_sell_records(sims[case]['trades']))}  "
              f"trigger_days={sims[case]['n_trigger_days']}  "
              f"blocked_days={sims[case]['n_blocked_days']}  "
              f"lot_infeasible_days={sims[case]['n_lot_infeasible']}")

    base_sells = _sell_records(sims["A"]["trades"])
    baseline_entry_R_by_date = {t["entry_date"]: t["realized_R"] for t in base_sells}

    print("[4/4] メトリクス計算 + 判定...")
    metrics: Dict[str, CaseMetrics] = {}
    for case in CASES:
        metrics[case] = compute_case_metrics(
            case, sims[case], common_dates, CAPITAL, sims["A"], baseline_entry_R_by_date,
        )

    non_baseline = [c for c in CASES if c != "A"]
    best_case = max(non_baseline, key=lambda c: metrics[c].calmar_delta)
    best_delta_calmar = metrics[best_case].calmar_delta
    adoption_cases = [c for c in non_baseline if metrics[c].meets_adoption]

    if adoption_cases:
        research_decision = "ADOPTION_CANDIDATE_FOUND"
        best_case = max(adoption_cases, key=lambda c: metrics[c].calmar_delta)
        best_delta_calmar = metrics[best_case].calmar_delta
        final_research_state = "PROCEED_TO_WALKFORWARD"
    elif best_delta_calmar < EXHAUSTED_CALMAR_DELTA:
        research_decision = "PORTFOLIO_GEOMETRY_EXHAUSTED"
        final_research_state = "CLOSE_GEOMETRY_RESEARCH"
    else:
        research_decision = "MARGINAL_IMPROVEMENT_OBSERVED"
        final_research_state = "FURTHER_RESEARCH_WARRANTED"

    print(f"  best_case          = {best_case}")
    print(f"  best_delta_calmar   = {best_delta_calmar:+.4f}")
    print(f"  research_decision   = {research_decision}")

    write_trade_csv({c: sims[c]["trades"] for c in CASES})
    write_md_report(metrics, research_decision, best_case, best_delta_calmar, final_research_state)
    append_telemetry(metrics, research_decision)

    print()
    print("=" * 68)
    print(f"research_decision = {research_decision}  (best_case={best_case})")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
