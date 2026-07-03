"""
run_study13_pilot.py — Study13 Deployment Pilot & Operational Readiness

Usage:
  python run_study13_pilot.py --backtest 2025 --days 30
      30-day shadow replay using 2025 historical data (Phase 1 validation)

  python run_study13_pilot.py --gate-check
      Evaluate Phase 1 gate from saved state

  python run_study13_pilot.py --report
      Generate daily pilot report

CRITICAL: --live mode requires kabuステーション API + 明示的な --yes
          signal_bridge.py 本番統合は Phase 2 以降・ASK_FIRST
"""
from __future__ import annotations

import argparse, math, sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, _take, run_period,
    LOT, COST_ONE_WAY,
)
from src.pilot.shadow_state import (
    PilotState, ShadowPosition,
    PILOT_STATE_FILE, PILOT_LOG_DIR, PILOT_REPORT_DIR,
    PHASE1, PHASE2, PHASE3, STOPPED,
    BACKTEST_CALMAR,
)
from src.pilot.shadow_engine import (
    AuditLogger, step_day,
    RSR_LO, RSR_HI, D90_MAX, D90_MIN, SLOPE_MAX,
    EXIT_THR, MIN_HOLD, TARGET_NAV,
)
from src.pilot.phase_gate import (
    check_phase1_gate, check_phase1_stop, check_phase3_gate,
)

IS_START  = "2018-01-01"
FULL_END  = "2025-12-31"


# ── matrix helpers ─────────────────────────────────────────────────────

def compute_cross90_mat(rsr_mat: np.ndarray) -> np.ndarray:
    n, m   = rsr_mat.shape
    out    = np.zeros((n, m), dtype=np.int32)
    counts = np.zeros(m, dtype=np.int32)
    for i in range(n):
        out[i] = counts.copy()
        above  = rsr_mat[i] >= 90.0
        counts[above]  += 1
        counts[~above]  = 0
    return out


def compute_slope5(rsr_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rsr_mat, dtype=np.float32)
    out[5:] = rsr_mat[5:] - rsr_mat[:-5]
    return out


def build_matrices(universe_raw, rsr_df, sym_active_df, topix_close,
                   active_syms, date_range_start, date_range_end):
    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    common_dates = common_dates[
        (common_dates >= pd.Timestamp(date_range_start)) &
        (common_dates <= pd.Timestamp(date_range_end))
    ]
    n_dates  = len(common_dates)
    n_syms   = len(active_syms)
    sym_to_i = {s: i for i, s in enumerate(active_syms)}
    date_strs = [str(d.date()) for d in common_dates]

    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        valid   = row_idx >= 0
        if valid.any():
            open_mat[valid,  si] = df_src["Open"].to_numpy(dtype=np.float32)[row_idx[valid]]
            close_mat[valid, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx[valid]]

    rsr_mat    = np.nan_to_num(_take(rsr_df, common_dates, active_syms,
                                     dtype=np.float32, fill_value=np.nan), nan=0.0)
    sym_act_m  = (_take(sym_active_df, common_dates, active_syms,
                        dtype=np.float32, fill_value=1.0)
                  if sym_active_df is not None else None)
    mkt_ret1   = (_take(topix_close.pct_change(), common_dates,
                        dtype=np.float32, fill_value=0.0)
                  if topix_close is not None else np.zeros(n_dates, dtype=np.float32))
    cross90    = compute_cross90_mat(rsr_mat)
    slope5     = compute_slope5(rsr_mat)

    return (common_dates, date_strs, sym_to_i,
            open_mat, close_mat, rsr_mat, sym_act_m, mkt_ret1, cross90, slope5)


# ── Ideal backtest equity (Study11 Case B replay) for tracking error ──

def run_ideal_shadow(open_mat, close_mat, rsr_mat, sym_act_m, cross90, slope5,
                     mkt_ret1, active_syms, sym_to_i, common_dates, date_strs,
                     pilot_indices: np.ndarray | None = None) -> np.ndarray:
    """
    Replay Study9 Case B + annual rebalance → ideal equity series.

    pilot_indices: if provided, run fresh from pilot_indices[0] with cash=750k.
                   Returns array indexed by position in pilot_indices.
    """
    from src.pilot.shadow_engine import _is_annual_rebal

    if pilot_indices is not None:
        # Fresh simulation from pilot start — same initial state as shadow
        indices   = pilot_indices
        sl_cash   = TARGET_NAV
        sl_pos    = None
        eq        = np.zeros(len(indices), dtype=np.float64)
        day_counter = 0  # local hold-day counter (matches shadow engine's total_days)

        for local_i, i in enumerate(indices):
            date = common_dates[i]
            pos_val = (sl_pos["qty"] * float(close_mat[i, sym_to_i[sl_pos["symbol"]]])
                       if sl_pos else 0.0)
            nav = sl_cash + pos_val
            if _is_annual_rebal(date):
                if nav > TARGET_NAV:
                    excess = min(nav - TARGET_NAV, sl_cash)
                    sl_cash -= excess
                elif nav < TARGET_NAV and sl_pos is None:
                    sl_cash += TARGET_NAV - nav
            pos_val = (sl_pos["qty"] * float(close_mat[i, sym_to_i[sl_pos["symbol"]]])
                       if sl_pos else 0.0)
            eq[local_i] = sl_cash + pos_val

            nxt = i + 1
            if nxt >= len(common_dates): day_counter += 1; continue
            mkt_shock = float(mkt_ret1[i]) <= -0.05

            if sl_pos:
                si  = sym_to_i[sl_pos["symbol"]]
                rv  = float(rsr_mat[i, si])
                hd  = day_counter - sl_pos["entry_idx"]
                sl_pos.setdefault("days_below", 0)
                if mkt_shock:
                    ct = float(close_mat[i, si])
                    ep = sl_pos["entry_price"]
                    if ep > 0 and ct / ep - 1 <= -0.08:
                        sp = float(open_mat[nxt, si])
                        sl_cash += sl_pos["qty"] * sp * (1 - COST_ONE_WAY)
                        sl_pos = None; day_counter += 1; continue
                if rv < EXIT_THR: sl_pos["days_below"] += 1
                else:              sl_pos["days_below"]  = 0
                if sl_pos["days_below"] >= 1 and hd >= MIN_HOLD:
                    sp = float(open_mat[nxt, si])
                    sl_cash += sl_pos["qty"] * sp * (1 - COST_ONE_WAY)
                    sl_pos = None

            if sl_pos is None and not mkt_shock:
                cands = []
                for sym in active_syms:
                    si  = sym_to_i[sym]
                    rv  = float(rsr_mat[i, si])
                    d90 = int(cross90[i, si])
                    sl5 = float(slope5[i, si])
                    if rv < RSR_LO or rv >= RSR_HI: continue
                    if d90 < D90_MIN or d90 > D90_MAX: continue
                    if sl5 > SLOPE_MAX: continue
                    if sym_act_m is not None and float(sym_act_m[i, si]) < 0.5: continue
                    cands.append((rv, d90, sl5, sym))
                if cands:
                    cands.sort(key=lambda x: (-x[0], x[1]))
                    sym = cands[0][3]; si = sym_to_i[sym]
                    bp  = float(open_mat[nxt, si])
                    if bp > 0:
                        qty = int(sl_cash * 0.95 / bp / LOT) * LOT
                        cost = qty * bp * (1 + COST_ONE_WAY)
                        if qty > 0 and cost <= sl_cash:
                            sl_cash -= cost
                            sl_pos = {"symbol": sym, "qty": qty, "entry_price": bp,
                                      "entry_idx": day_counter, "days_below": 0}
            day_counter += 1
        return eq

    # Full-period version (2018-2025, for reference)
    n       = len(common_dates)
    sl_cash = TARGET_NAV
    sl_pos  = None
    eq      = np.zeros(n, dtype=np.float64)
    for i in range(n):
        date = common_dates[i]
        pos_val = (sl_pos["qty"] * float(close_mat[i, sym_to_i[sl_pos["symbol"]]])
                   if sl_pos else 0.0)
        nav = sl_cash + pos_val
        if _is_annual_rebal(date):
            if nav > TARGET_NAV:
                excess = min(nav - TARGET_NAV, sl_cash)
                sl_cash -= excess
            elif nav < TARGET_NAV and sl_pos is None:
                sl_cash += TARGET_NAV - nav
        pos_val = (sl_pos["qty"] * float(close_mat[i, sym_to_i[sl_pos["symbol"]]])
                   if sl_pos else 0.0)
        eq[i] = sl_cash + pos_val
        if i + 1 >= n: break
        nxt = i + 1
        mkt_shock = float(mkt_ret1[i]) <= -0.05
        if sl_pos:
            si  = sym_to_i[sl_pos["symbol"]]
            rv  = float(rsr_mat[i, si])
            hd  = i - sl_pos["entry_idx"]
            sl_pos.setdefault("days_below", 0)
            if mkt_shock:
                ct = float(close_mat[i, si])
                ep = sl_pos["entry_price"]
                if ep > 0 and ct / ep - 1 <= -0.08:
                    sp = float(open_mat[nxt, si])
                    sl_cash += sl_pos["qty"] * sp * (1 - COST_ONE_WAY)
                    sl_pos = None; continue
            if rv < EXIT_THR: sl_pos["days_below"] += 1
            else:              sl_pos["days_below"]  = 0
            if sl_pos["days_below"] >= 1 and hd >= MIN_HOLD:
                sp = float(open_mat[nxt, si])
                sl_cash += sl_pos["qty"] * sp * (1 - COST_ONE_WAY)
                sl_pos = None
        if sl_pos is None and not mkt_shock:
            cands = []
            for sym in active_syms:
                si  = sym_to_i[sym]
                rv  = float(rsr_mat[i, si])
                d90 = int(cross90[i, si])
                sl5 = float(slope5[i, si])
                if rv < RSR_LO or rv >= RSR_HI: continue
                if d90 < D90_MIN or d90 > D90_MAX: continue
                if sl5 > SLOPE_MAX: continue
                if sym_act_m is not None and float(sym_act_m[i, si]) < 0.5: continue
                cands.append((rv, d90, sl5, sym))
            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                sym = cands[0][3]; si = sym_to_i[sym]
                bp  = float(open_mat[nxt, si])
                if bp > 0:
                    qty = int(sl_cash * 0.95 / bp / LOT) * LOT
                    cost = qty * bp * (1 + COST_ONE_WAY)
                    if qty > 0 and cost <= sl_cash:
                        sl_cash -= cost
                        sl_pos = {"symbol": sym, "qty": qty, "entry_price": bp,
                                  "entry_idx": nxt, "days_below": 0}
    return eq


# ── Backtest-mode runner ───────────────────────────────────────────────

def run_backtest_pilot(year: int = 2025, n_days: int = 30, verbose: bool = True) -> PilotState:
    cfg = load_strategy_config()
    if verbose:
        print("=" * 68)
        print(f"  Study13 Pilot — Phase 1 Shadow Replay  [{year}  {n_days}d]")
        print("=" * 68 + "\n")

    print("[1/4] データロード...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    active_syms = list({s: v for s, v in rsr_syms.items() if s in universe_raw}.keys())
    start_dt = f"{year}-01-01"
    end_dt   = f"{year}-12-31"

    print("[2/4] マトリクス構築...")
    (common_dates, date_strs, sym_to_i,
     open_mat, close_mat, rsr_mat, sym_act_m, mkt_ret1,
     cross90, slope5) = build_matrices(
        universe_raw, rsr_df, sym_active_df, topix_close,
        active_syms, IS_START, FULL_END,
    )
    # Filter to target year
    year_mask  = np.array([s.startswith(str(year)) for s in date_strs])
    year_idx   = np.where(year_mask)[0]
    if len(year_idx) < n_days:
        print(f"  WARNING: {year} にデータが {len(year_idx)} 日しかありません ({n_days}日要求)")
        n_days = len(year_idx)
    pilot_idx  = year_idx[:n_days]
    print(f"  Shadow期間: {date_strs[pilot_idx[0]]} → {date_strs[pilot_idx[-1]]}  "
          f"({len(pilot_idx)}営業日)")

    print("[3/4] Ideal equity (tracking error base) 計算...")
    # Run fresh from pilot start (same initial state as shadow: cash=750k, no pos)
    ideal_eq_pilot = run_ideal_shadow(
        open_mat, close_mat, rsr_mat, sym_act_m, cross90, slope5,
        mkt_ret1, active_syms, sym_to_i, common_dates, date_strs,
        pilot_indices=pilot_idx,
    )
    # ideal_eq_pilot[j] = ideal equity on j-th pilot day (indexed 0..n_days-1)

    print("[4/4] Phase 1 Shadow 実行...")
    state  = PilotState(mode="backtest")
    audit  = AuditLogger(PILOT_LOG_DIR)

    for local_i, idx in enumerate(pilot_idx):
        date = common_dates[idx]
        ds   = date_strs[idx]

        # Build per-day signal dicts
        rsr_d   = {sym: float(rsr_mat[idx, si]) for sym, si in sym_to_i.items()}
        d90_d   = {sym: int(cross90[idx, si])   for sym, si in sym_to_i.items()}
        sl5_d   = {sym: float(slope5[idx, si])  for sym, si in sym_to_i.items()}
        close_d = {sym: float(close_mat[idx, si]) for sym, si in sym_to_i.items()}
        mkt_r   = float(mkt_ret1[idx])

        # Next-day open for virtual fills
        nxt = idx + 1
        if nxt < len(date_strs):
            nopen_d = {sym: float(open_mat[nxt, si]) for sym, si in sym_to_i.items()}
        else:
            nopen_d = close_d.copy()

        # Filter to active symbols
        act = [sym for sym in active_syms
               if sym_act_m is None or float(sym_act_m[idx, sym_to_i[sym]]) >= 0.5]

        ideal_eq = float(ideal_eq_pilot[local_i])

        state, stop_reason = step_day(
            state=state, date=date, ds=ds,
            rsr_by_sym=rsr_d, d90_by_sym=d90_d, slope5_by_sym=sl5_d,
            close_by_sym=close_d, mkt_ret=mkt_r,
            active_syms=act, next_open_by_sym=nopen_d,
            ideal_eq_today=ideal_eq, audit=audit,
        )

        # Check immediate stop conditions
        stop_gate = check_phase1_stop(state, stop_reason)
        if stop_gate.stopped:
            state.phase = STOPPED
            audit.critical(ds, "PILOT_STOPPED", stop_gate.stop_reason or "")
            print(f"\n  !! PILOT STOPPED: {stop_gate.stop_reason}")
            break

        if verbose and (state.total_days % 5 == 0 or state.total_days == 1):
            eq_now = state.equity_history[-1] if state.equity_history else TARGET_NAV
            print(f"  day {state.trading_days:>3}  {ds}  eq=¥{eq_now:,.0f}  "
                  f"pos={'YES' if state.position else 'no':3}  "
                  f"orders={state.n_orders}")

    state.save(PILOT_STATE_FILE)
    return state


# ── Gate check ────────────────────────────────────────────────────────

def run_gate_check(verbose: bool = True) -> None:
    state = PilotState.load(PILOT_STATE_FILE)
    gate  = check_phase1_gate(state)
    if verbose:
        print("\n" + "=" * 68)
        print(f"  Study13 Phase 1 Gate Check  ({state.last_date})")
        print("=" * 68)
        for c in gate.conditions:
            mark = "✅" if c.passed else "❌"
            print(f"  {mark} {c.name:<42} 実績={c.actual}  目標={c.target}")
        print(f"\n  → {gate.summary}")


# ── Report generator ──────────────────────────────────────────────────

def write_report(state: PilotState, path: Path) -> None:
    import time as _time
    L = []; w = L.append

    eq_h = state.equity_history
    eq_arr = np.array(eq_h, dtype=float) if eq_h else np.array([TARGET_NAV])
    final_eq  = float(eq_arr[-1]) if len(eq_arr) > 0 else TARGET_NAV
    n_years   = state.trading_days / 252
    cagr      = ((final_eq / TARGET_NAV) ** (1 / max(n_years, 0.01)) - 1) * 100 if n_years > 0 else 0.0
    roll      = pd.Series(eq_arr).expanding().max()
    max_dd    = float(((pd.Series(eq_arr) - roll) / roll).min()) * 100 if len(eq_arr) > 1 else 0.0
    calmar    = cagr / abs(max_dd) if abs(max_dd) > 0.01 else float("inf")
    te_max    = state.tracking_error_max() * 100
    rj_rate   = state.broker_reject_rate() * 100
    lat_p99   = state.latency_p99()

    w(f"# Study13 Deployment Pilot — Phase 1 Shadow Report")
    w(f"\n作成日: {_time.strftime('%Y-%m-%d')}  |  運用専用 / signal_bridge.py 統合禁止")
    w(f"Shadow期間: {state.start_date} → {state.last_date}  ({state.trading_days}営業日)")
    w(f"フェーズ: **{state.phase}**\n")

    w("---\n## 1. Shadow 性能サマリー\n")
    w("| 指標 | 値 | 評価 |")
    w("|---|---|---|")
    w(f"| shadow_CAGR       | {cagr:+.2f}%  | — |")
    w(f"| shadow_MaxDD      | {max_dd:.2f}% | — |")
    w(f"| shadow_Calmar     | {calmar:.3f}  | 昇格閾値 ≥ {BACKTEST_CALMAR*0.90:.3f} |")
    w(f"| ideal_Calmar (BT) | {BACKTEST_CALMAR:.3f} | Study12 ideal avg |")
    w(f"| shadow_equity     | ¥{final_eq:,.0f} | 開始 ¥{TARGET_NAV:,.0f} |")
    w(f"| n_orders          | {state.n_orders}    | — |")
    w(f"| n_filled          | {state.n_filled}    | — |")

    w("\n---\n## 2. 運用品質 Audit\n")
    w("| 指標 | 実績 | 採用ライン | 判定 |")
    w("|---|---|---|---|")

    def row(name, val, target_str, ok):
        mark = "✅" if ok else "❌"
        return f"| {name} | {val} | {target_str} | {mark} |"

    w(row("broker_reject %",      f"{rj_rate:.3f}%",   "< 0.5%",  rj_rate < 0.5))
    w(row("latency_p99 (ms)",     f"{lat_p99:.1f}ms",  "< 5000ms", lat_p99 < 5000))
    w(row("tracking_error_max %", f"{te_max:.3f}%",    "< 1.0%",   te_max < 1.0))
    w(row("position_diff_days",   str(state.n_position_diff_days), "== 0",
          state.n_position_diff_days == 0))
    w(row("cash_drift_events",    str(state.n_cash_drift_events),  "== 0",
          state.n_cash_drift_events == 0))
    w(row("manual_intervention",  str(state.n_manual_intervention),"== 0",
          state.n_manual_intervention == 0))
    w(row("rollback_counter",     str(state.n_rollback),           "== 0",
          state.n_rollback == 0))
    w(row("duplicate_guard",      str(state.n_duplicate_guard),    "== 0",
          state.n_duplicate_guard == 0))
    w(row("state_hash_break",     str(state.n_state_hash_break),   "== 0",
          state.n_state_hash_break == 0))

    w("\n---\n## 3. Tracking Error 履歴\n")
    if state.tracking_errors:
        te_arr = np.array(state.tracking_errors) * 100
        w(f"- mean: {te_arr.mean():.4f}%")
        w(f"- max:  {te_arr.max():.4f}%")
        w(f"- p95:  {np.percentile(te_arr, 95):.4f}%")

    w("\n---\n## 4. Phase 1 昇格ゲート\n")
    gate = check_phase1_gate(state)
    w("| 条件 | 実績 | 目標 | 判定 |")
    w("|---|---|---|---|")
    for c in gate.conditions:
        mark = "✅" if c.passed else "❌"
        w(f"| {c.name} | {c.actual} | {c.target} | {mark} |")
    w(f"\n**判定: {gate.summary}**")

    w("\n---\n## 5. Phase 2 以降 設計\n")
    w("```")
    w("Phase 2 — Limited Live")
    w("  mode:            live")
    w("  alloc:           10%  (fixed_cap)")
    w("  rebalance:       annual target=750k")
    w("  isolation:       独立口座 (production 資金共有禁止)")
    w("")
    w("  kill_switch:     rolling_DD > prod_DD + 2pp → alloc=0%")
    w("  freeze:          consec_reject ≥ 3 → 新規注文停止")
    w("  rollback:        alpha_real_30d < 80% → Phase1_Shadow 再開")
    w("  degrade:         fill_rate < 95% → 注意モード")
    w("")
    w("Phase 3 — Production Promotion 条件")
    w("  30営業日連続正常 AND Calmar改善維持 AND 障害0件")
    w("  deploy_decision: recommended_alloc / post_launch_monitor")
    w("```")

    w("\n---\n## 6. Daily 出力定義\n")
    w("Phase 2 移行後、毎日出力必須 (REPORT_ALWAYS=true):")
    w("| 項目 | 計測方法 |")
    w("|---|---|")
    w("| realized_alpha    | shadow_equity / ideal_equity × 100% |")
    w("| execution_tax     | shadow_Calmar / ideal_Calmar |")
    w("| cash_integrity    | n_cash_drift_events == 0 |")
    w("| allocation_weight | shadow_equity / (prod_equity + shadow_equity) |")
    w("| rollback_state    | NORMAL / FREEZE / ROLLBACK / KILL |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Study13 Deployment Pilot")
    p.add_argument("--backtest", type=int, default=2025, metavar="YEAR",
                   help="Historical replay year (default: 2025)")
    p.add_argument("--days",     type=int, default=30, metavar="N",
                   help="Number of shadow trading days (default: 30)")
    p.add_argument("--gate-check", action="store_true",
                   help="Check Phase 1 gate from saved state")
    p.add_argument("--report",    action="store_true",
                   help="Generate report from saved state")
    p.add_argument("--live",      action="store_true",
                   help="Live shadow mode (requires --yes + kabu API)")
    p.add_argument("--yes",       action="store_true",
                   help="Confirm live execution")
    args = p.parse_args()

    if args.live:
        if not args.yes:
            print("ERROR: --live は --yes が必要。ASK_FIRST ルール確認後に実行。")
            return 1
        print("WARN: live shadow mode は未実装 (Phase 2 は ASK_FIRST 後に実装)。")
        return 1

    if args.gate_check:
        run_gate_check(verbose=True)
        return 0

    # Backtest shadow replay
    state = run_backtest_pilot(year=args.backtest, n_days=args.days)

    # Gate check
    run_gate_check(verbose=True)

    # Report
    report_path = PILOT_REPORT_DIR / f"{state.last_date}_phase1.md"
    write_report(state, report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
