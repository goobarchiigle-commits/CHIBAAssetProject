"""
study53_opportunity_loss_analysis.py
Opportunity Loss Analysis — Production D_ATR_EQ (Study52)

平均Exposure約29%の根本原因を定量化し、資本効率改善余地を評価する。

Sections:
  1. 日次統計 (cash_ratio / exposure / holdings / open_slots / candidates)
  2. BUY候補記録 (採用/不採用/理由/RSR/Rank/ATR%/Composite Score)
  3. 不採用理由ランキング (件数/割合/累積)
  4. 不採用銘柄のForward Return (5d/10d/20d/60d)
  5. Counterfactual BT (max_positions=10, 実質無制限)
  6. Opportunity Loss (実利益 vs 仮想利益)
  7. 診断 (資本不足 / フィルター過剰 / エントリー不足)

設定: Production D_ATR_EQ (ATR Extension + EQ_SCALE, VOL_ADJ 除外)
期間: IS 2018-2024 Full
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab
from src.backtest.wf_dynamic_universe import WF_SEGS
from src.config_loader import load_strategy_config

TODAY_STR   = date.today().strftime("%Y-%m-%d")
CAPITAL     = 3_000_000
IS_START    = "2018-01-01"
IS_END      = "2024-12-31"
DATA_END    = "2025-12-31"
MIN_HOLD    = 3
FWD_WINDOWS = [5, 10, 20, 60]

# D_ATR_EQ production params
EP_EXIT     = "A"       # ATR Extension
EP_ADDON    = "D"       # EQ_SCALE
ADDON_ATR_MULT   = 1.0
ADDON_SIZE_FRAC  = 0.25

# Counterfactual: remove position cap constraint
CF_MAX_POSITIONS = 10


def get_active(ds: dict, start: str, end: str) -> pd.DataFrame:
    cfg = load_strategy_config()
    bc  = cfg.risk_controls.bear_universe_filter
    be  = list(bc.excluded_sectors) if bc.enabled else None
    all_syms = list(ds["trade_syms"].keys())
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"],
        topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"],
        all_syms=all_syms,
        start=start, end=end,
        bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds: dict, sym_active_df, start: str, end: str,
           max_positions_override=None) -> dict:
    return cab.run_scenario(
        scenario               = "BASELINE",
        universe_raw           = ds["universe_raw"],
        rsr_df                 = ds["rsr_df"],
        alpha_df               = None,
        regime_df              = ds["regime_df"],
        trade_syms             = ds["trade_syms"],
        rsr_syms               = ds["rsr_syms"],
        cfg                    = ds["base_cfg"],
        start                  = start, end=end, verbose=False,
        tech_matrices          = ds["tech_matrices"],
        breadth_series         = ds["breadth_series"],
        capital                = CAPITAL,
        min_hold               = MIN_HOLD,
        topix_close            = ds["topix_close"],
        market_shock_mode      = "composite",
        rsr_exit_threshold     = 70.0,
        sym_active_df          = sym_active_df,
        enable_simple_rsr_exit = True,
        enable_atr_trailing_prod = True,
        enable_multilayer_rsr  = True,
        enable_atr_risk_sizing = False,
        enable_mtf_filter      = False,
        sizing_mode            = "existing",
        exit_policy            = EP_EXIT,
        exit_policy_atr_mult   = ADDON_ATR_MULT,
        exit_policy_defer_days = 5,
        max_positions_ts       = None,
        addon_policy           = EP_ADDON,
        addon_atr_mult         = ADDON_ATR_MULT,
        addon_stage2_mult      = 2.0,
        addon_max_per_pos      = 1,
        addon_size_frac        = ADDON_SIZE_FRAC,
        max_positions_override = max_positions_override,
    )


def calc_forward_returns(
    candidates: list[dict],
    universe_raw: dict,
    label: str,
) -> dict:
    """candidates: [{date, symbol, ...}, ...]
    Returns summary of forward returns at FWD_WINDOWS days.
    """
    if not candidates:
        return {}

    records: list[dict] = []
    for cand in candidates:
        sym  = cand.get("symbol")
        dstr = cand.get("date")
        if not sym or not dstr or sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close = df_c["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        try:
            d = pd.Timestamp(dstr)
        except Exception:
            continue
        # find reference close (nearest at or before date)
        avail = close[close.index <= d]
        if avail.empty:
            continue
        ref_px = float(avail.iloc[-1])
        if ref_px <= 0:
            continue
        row = {
            "date": dstr, "symbol": sym,
            "rsr": cand.get("rsr", 0.0),
            "atr_pct": cand.get("atr_pct", 0.0),
            "rank": cand.get("rank", 0),
        }
        for w in FWD_WINDOWS:
            # find close at approximately w trading days later
            future = close[close.index > d]
            if len(future) < w:
                row[f"fwd{w}d"] = None
            else:
                fwd_px = float(future.iloc[w - 1])
                row[f"fwd{w}d"] = round((fwd_px - ref_px) / ref_px * 100, 2) if ref_px > 0 else None
        records.append(row)

    if not records:
        return {"n": 0}

    summary: dict = {"n": len(records), "label": label}
    for w in FWD_WINDOWS:
        key = f"fwd{w}d"
        vals = [r[key] for r in records if r.get(key) is not None]
        if not vals:
            summary[key] = {}
            continue
        arr = np.array(vals)
        summary[key] = {
            "mean":   round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "win_rate": round(float((arr > 0).mean() * 100), 1),
            "top10pct": round(float(np.percentile(arr, 90)), 2),
            "bottom10pct": round(float(np.percentile(arr, 10)), 2),
            "n": len(arr),
        }
    return summary


def print_section(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print('='*72)


def main() -> None:
    print("="*72)
    print("  Study53 — Opportunity Loss Analysis (Production D_ATR_EQ)")
    print(f"  Date: {TODAY_STR}   IS: {IS_START}~{IS_END}   Capital: ¥{CAPITAL:,}")
    print("="*72)

    # ── 1. データセット ──────────────────────────────────────────────────
    print(f"\n[Data] データセット構築中 (end={DATA_END})...")
    ds       = build_common_dataset(DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  {len(all_syms)} シンボル")

    # ── 2. Production BT (D_ATR_EQ, IS 2018-2024) ────────────────────
    print(f"\n[BT] D_ATR_EQ IS run ({IS_START}~{IS_END})...")
    active_is = get_active(ds, IS_START, IS_END)
    res = run_bt(ds, active_is, IS_START, IS_END)

    # 基本指標
    cagr       = res.get("cagr", 0.0)
    sharpe     = res.get("sharpe", 0.0)
    max_dd     = res.get("max_dd", 0.0)
    calmar     = res.get("calmar", 0.0)
    n_trades   = res.get("n_trades", 0)
    avg_exp    = res.get("avg_exposure", 0.0)
    print(f"  CAGR={cagr:+.2f}%  Sharpe={sharpe:.3f}  MaxDD={max_dd:.2f}%  Calmar={calmar:.3f}")
    print(f"  Trades={n_trades}  AvgExposure={avg_exp:.1f}%")

    # ── Section 1: 日次統計 ──────────────────────────────────────────
    print_section("Section 1: 日次統計")

    eq        = res.get("equity_curve", pd.Series(dtype=float))
    long_not  = res.get("long_notional", pd.Series(dtype=float))
    pos_ser   = res.get("pos_series", pd.Series(dtype=int))
    cand_ser  = res.get("cand_series", pd.Series(dtype=int))

    exposure_ser   = (long_not / eq.clip(lower=1.0)) * 100
    cash_ratio_ser = 100.0 - exposure_ser
    open_slots_ser = 3 - pos_ser  # max_positions=3

    # 日次分布
    days_total   = len(eq)
    days_zero_exp = int((exposure_ser < 1.0).sum())  # 実質フルキャッシュ
    days_full_cap = int((pos_ser >= 3).sum())          # max_pos到達
    days_no_cand  = int((cand_ser == 0).sum())          # BUY候補ゼロ
    days_idle_w_cand = int(((cash_ratio_ser > 10.0) & (cand_ser > 0)).sum())  # 資本遊休+候補あり

    avg_cash_pct = float(cash_ratio_ser.mean())
    avg_open_slots = float(open_slots_ser.mean())
    avg_cand_daily = float(cand_ser.mean())

    print(f"  総営業日数:              {days_total:,}")
    print(f"  平均Exposure:            {avg_exp:.1f}%")
    print(f"  平均キャッシュ比率:      {avg_cash_pct:.1f}%")
    print(f"  平均日次候補数:          {avg_cand_daily:.2f} 件/日")
    print(f"  平均空きスロット:        {avg_open_slots:.2f} スロット")
    print(f"  ──────────────────────────────────────────")
    print(f"  候補ゼロ日:              {days_no_cand:,} 日 ({days_no_cand/max(1,days_total)*100:.1f}%)")
    print(f"  max_pos到達日:           {days_full_cap:,} 日 ({days_full_cap/max(1,days_total)*100:.1f}%)")
    print(f"  フルキャッシュ日(exp<1%): {days_zero_exp:,} 日 ({days_zero_exp/max(1,days_total)*100:.1f}%)")
    print(f"  資本遊休+候補あり日:      {days_idle_w_cand:,} 日 ({days_idle_w_cand/max(1,days_total)*100:.1f}%)")

    # 年別exposure
    print(f"\n  年別平均Exposure:")
    if not eq.empty:
        exp_annual = exposure_ser.groupby(exposure_ser.index.year).mean()
        for yr, v in exp_annual.items():
            print(f"    {yr}: {v:.1f}%")

    # ── Section 2: BUY候補サマリー ──────────────────────────────────
    print_section("Section 2: BUY候補記録サマリー")

    missed_cands   = res.get("_missed_cands",       [])
    lot_rej_detail = res.get("_rejected_by_lot_detail", [])
    skip_detail    = res.get("_skip_detail",         [])

    total_max_pos   = res.get("missed_by_cap_count", 0)
    total_lot_rej   = res.get("rejected_by_lot_count", 0)
    skip_stats      = res.get("skip_stats", {})
    total_sec_cap   = skip_stats.get("sector_cap", 0)
    total_cls_cap   = skip_stats.get("cluster_cap", 0)
    total_gross_exp = skip_stats.get("gross_exposure", 0)

    total_rejected = total_max_pos + total_lot_rej + total_sec_cap + total_cls_cap + total_gross_exp

    # admitted = n_trades (BUY side only - note n_trades counts SELL side)
    # approximate: number of actual buy executions
    trades_list = res.get("_trades", [])
    n_admitted  = len(trades_list)  # SELL trades ≈ entries

    print(f"  採用済みトレード(SELL件数): {n_admitted}")
    print(f"  不採用合計:                 {total_rejected}")
    print(f"    - MAX_POS:              {total_max_pos}")
    print(f"    - LOT_REJECT:           {total_lot_rej}")
    print(f"    - SECTOR_CAP:           {total_sec_cap}")
    print(f"    - CLUSTER_CAP:          {total_cls_cap}")
    print(f"    - GROSS_EXPOSURE:       {total_gross_exp}")

    # Accepted RSR/ATR distribution from trades
    if trades_list:
        # RSR from trades reason field
        print(f"\n  採用トレードの日次候補中の相対ランク分布:")
        # We don't have rank of admitted trades, use avg candidates as proxy
        print(f"    (avg日次候補数={avg_cand_daily:.2f}、採用は通常上位{3}件)")

    # ── Section 3: 不採用理由ランキング ─────────────────────────────
    print_section("Section 3: 不採用理由ランキング")

    reasons = [
        ("MAX_POS (ポジション上限)",          total_max_pos),
        ("LOT_REJECT (資本不足/lot切捨て)",   total_lot_rej),
        ("SECTOR_CAP (セクター集中制限)",      total_sec_cap),
        ("CLUSTER_CAP (クラスター集中制限)",   total_cls_cap),
        ("GROSS_EXPOSURE (総Exposure制限)",    total_gross_exp),
    ]
    reasons_sorted = sorted(reasons, key=lambda x: -x[1])
    cumul = 0
    print(f"  {'理由':<40} {'件数':>8} {'割合':>8} {'累積':>8}")
    print(f"  {'─'*68}")
    for name, cnt in reasons_sorted:
        pct  = cnt / max(1, total_rejected) * 100
        cumul += pct
        print(f"  {name:<40} {cnt:>8,} {pct:>7.1f}% {cumul:>7.1f}%")
    print(f"  {'─'*68}")
    print(f"  {'合計':<40} {total_rejected:>8,}  100.0%")

    # MAX_POS rejected RSR分布
    if missed_cands:
        rsr_vals = [c.get("rsr", 0) for c in missed_cands if c.get("rsr")]
        scores   = [c.get("composite_score", 0) for c in missed_cands if c.get("composite_score")]
        atr_vals = [c.get("atr_pct", 0) for c in missed_cands if c.get("atr_pct")]
        print(f"\n  MAX_POS候補の特性 (n={len(missed_cands)}/{total_max_pos}):")
        if rsr_vals:
            print(f"    RSR平均: {np.mean(rsr_vals):.1f}  中央値: {np.median(rsr_vals):.1f}  "
                  f"最小: {min(rsr_vals):.1f}  最大: {max(rsr_vals):.1f}")
        if scores:
            print(f"    CompositeScore平均: {np.mean(scores):.4f}  中央値: {np.median(scores):.4f}")
        if atr_vals:
            print(f"    ATR%平均: {np.mean(atr_vals):.2f}%  中央値: {np.median(atr_vals):.2f}%")
        ranks = [c.get("rank", 0) for c in missed_cands if c.get("rank") is not None]
        if ranks:
            print(f"    Rank(0=最高)平均: {np.mean(ranks):.1f}  中央値: {np.median(ranks):.1f}")

    # LOT_REJECT分布
    if lot_rej_detail:
        px_vals  = [c.get("buy_px", 0) for c in lot_rej_detail if c.get("buy_px")]
        rsr_rej  = [c.get("rsr", 0) for c in lot_rej_detail if c.get("rsr")]
        print(f"\n  LOT_REJECT候補の特性 (n={len(lot_rej_detail)}/{total_lot_rej}):")
        if px_vals:
            print(f"    株価平均: ¥{int(np.mean(px_vals)):,}  中央値: ¥{int(np.median(px_vals)):,}  "
                  f"最小: ¥{int(min(px_vals)):,}  最大: ¥{int(max(px_vals)):,}")
        if rsr_rej:
            print(f"    RSR平均: {np.mean(rsr_rej):.1f}  中央値: {np.median(rsr_rej):.1f}")

    # ── Section 4: 不採用銘柄のForward Return ────────────────────────
    print_section("Section 4: 不採用銘柄のForward Return")

    print("  [4a] MAX_POS (ポジション上限) 不採用候補...")
    fwd_max_pos = calc_forward_returns(missed_cands, ds["universe_raw"], "MAX_POS")
    print(f"  n={fwd_max_pos.get('n', 0)}")

    print("  [4b] LOT_REJECT (資本不足) 不採用候補...")
    fwd_lot_rej = calc_forward_returns(lot_rej_detail, ds["universe_raw"], "LOT_REJECT")
    print(f"  n={fwd_lot_rej.get('n', 0)}")

    print("  [4c] SECTOR/CLUSTER CAP 不採用候補...")
    fwd_skip = calc_forward_returns(skip_detail, ds["universe_raw"], "SECTOR/CLUSTER/GROSS")
    print(f"  n={fwd_skip.get('n', 0)}")

    def print_fwd_table(label: str, fwd_data: dict) -> None:
        if not fwd_data or fwd_data.get("n", 0) == 0:
            print(f"  {label}: データなし")
            return
        print(f"\n  {label} (n={fwd_data['n']}):")
        print(f"  {'期間':<10} {'平均':>8} {'中央値':>8} {'勝率':>8} {'上位10%':>10} {'下位10%':>10}")
        print(f"  {'─'*56}")
        for w in FWD_WINDOWS:
            key = f"fwd{w}d"
            d   = fwd_data.get(key, {})
            if not d:
                print(f"  {key:<10} {'N/A':>8}")
                continue
            print(f"  {key:<10} {d['mean']:>+8.2f}% {d['median']:>+8.2f}%"
                  f" {d['win_rate']:>7.1f}% {d['top10pct']:>+10.2f}% {d['bottom10pct']:>+10.2f}%")

    print_fwd_table("MAX_POS不採用", fwd_max_pos)
    print_fwd_table("LOT_REJECT不採用", fwd_lot_rej)
    print_fwd_table("SECTOR/CLUSTER/GROSS不採用", fwd_skip)

    # ── Section 5: Counterfactual BT ────────────────────────────────
    print_section(f"Section 5: Counterfactual BT (max_positions={CF_MAX_POSITIONS})")

    print(f"  D_ATR_EQ設定でmax_positions={CF_MAX_POSITIONS}として実行...")
    res_cf = run_bt(ds, active_is, IS_START, IS_END, max_positions_override=CF_MAX_POSITIONS)

    cagr_cf   = res_cf.get("cagr", 0.0)
    sharpe_cf = res_cf.get("sharpe", 0.0)
    max_dd_cf = res_cf.get("max_dd", 0.0)
    calmar_cf = res_cf.get("calmar", 0.0)
    n_tr_cf   = res_cf.get("n_trades", 0)
    avg_exp_cf = res_cf.get("avg_exposure", 0.0)

    sep = "─"*56
    print(f"\n  {sep}")
    print(f"  {'指標':<18} {'実際(max=3)':>14} {'Counterfactual(max={:d})'.format(CF_MAX_POSITIONS):>18} {'Δ':>10}")
    print(f"  {sep}")
    metrics_cf = [
        ("CAGR%",    cagr,    cagr_cf,    cagr_cf    - cagr),
        ("MaxDD%",   max_dd,  max_dd_cf,  max_dd_cf  - max_dd),
        ("Sharpe",   sharpe,  sharpe_cf,  sharpe_cf  - sharpe),
        ("Calmar",   calmar,  calmar_cf,  calmar_cf  - calmar),
        ("Trades",   n_trades,n_tr_cf,    n_tr_cf    - n_trades),
        ("AvgExp%",  avg_exp, avg_exp_cf, avg_exp_cf - avg_exp),
    ]
    for name, v_act, v_cf, delta in metrics_cf:
        print(f"  {name:<18} {str(v_act):>14} {str(v_cf):>22} {delta:>+10.2f}")
    print(f"  {sep}")

    # ── Section 6: Opportunity Loss ──────────────────────────────────
    print_section("Section 6: Opportunity Loss 定量評価")

    eq_cf = res_cf.get("equity_curve", pd.Series(dtype=float))

    # Actual final equity vs initial
    if not eq.empty:
        start_eq  = float(eq.iloc[0])
        end_eq    = float(eq.iloc[-1])
        actual_profit = end_eq - start_eq
    else:
        actual_profit = 0.0

    if not eq_cf.empty:
        cf_end_eq     = float(eq_cf.iloc[-1])
        cf_start_eq   = float(eq_cf.iloc[0])
        cf_profit     = cf_end_eq - cf_start_eq
    else:
        cf_profit = 0.0

    opportunity_loss    = cf_profit - actual_profit
    opportunity_loss_pct = opportunity_loss / max(1.0, CAPITAL) * 100

    years_is = (pd.Timestamp(IS_END) - pd.Timestamp(IS_START)).days / 365.25

    print(f"  期間: {IS_START} ~ {IS_END} ({years_is:.1f}年)")
    print(f"  初期資本: ¥{CAPITAL:,}")
    print(f"\n  実際の利益 (D_ATR_EQ, max=3):          ¥{int(actual_profit):>12,}  ({actual_profit/CAPITAL*100:+.1f}%)")
    print(f"  仮想利益 (Counterfactual, max={CF_MAX_POSITIONS}):       ¥{int(cf_profit):>12,}  ({cf_profit/CAPITAL*100:+.1f}%)")
    print(f"  差額 (機会損失):                         ¥{int(opportunity_loss):>12,}  ({opportunity_loss_pct:+.1f}%)")

    # ── Section 7: 診断 ─────────────────────────────────────────────
    print_section("Section 7: 診断 — 資本効率29%の根本原因")

    total_constraint = total_rejected
    if total_constraint == 0:
        total_constraint = 1

    pct_max_pos  = total_max_pos   / total_constraint * 100
    pct_lot_rej  = total_lot_rej   / total_constraint * 100
    pct_caps     = (total_sec_cap + total_cls_cap + total_gross_exp) / total_constraint * 100

    print(f"\n  ── 不採用原因の構成 ──")
    print(f"    ポジション上限 (MAX_POS):      {pct_max_pos:.1f}%  → max_positions=3 が主因?")
    print(f"    資本不足 (LOT_REJECT):         {pct_lot_rej:.1f}%  → ¥3M + lot=100 が主因?")
    print(f"    集中制限 (SEC/CLUS/GROSS):     {pct_caps:.1f}%  → フィルター過剰?")

    print(f"\n  ── Exposure 29% の分解 ──")
    print(f"    avg_exposure actual:          {avg_exp:.1f}%")
    print(f"    avg_exposure counterfactual:  {avg_exp_cf:.1f}%")
    exp_gain_from_cap_lift = avg_exp_cf - avg_exp
    print(f"    max_positions解除によるΔExp:  {exp_gain_from_cap_lift:+.1f}pp")

    print(f"\n  ── 候補不足の評価 ──")
    print(f"    候補ゼロ日の割合:             {days_no_cand/max(1,days_total)*100:.1f}%")
    print(f"    平均日次候補数:               {avg_cand_daily:.2f} 件/日")
    print(f"    (参考) RSR75+ブレイク銘柄は常時少数 → エントリー頻度が構造的制限)")

    print(f"\n  ── 主診断 ──")
    if pct_max_pos >= 40:
        diag = "MAX_POS_DOMINANT: ポジション上限(max=3)が支配的。cap解除でExposure向上余地あり。"
    elif pct_lot_rej >= 40:
        diag = "CAPITAL_DOMINANT: 資本不足(¥3M/lot=100)が支配的。増資またはlot緩和で改善余地あり。"
    elif days_no_cand / max(1, days_total) >= 0.50:
        diag = "ENTRY_DEFICIT: 候補ゼロ日が50%超。シグナル不足がボトルネック。"
    else:
        diag = "MIXED: 複合要因。MAX_POS/資本不足/フィルターが並存。"

    print(f"  {diag}")

    # 機会損失 vs 理論改善余地の結論
    cagr_gain_cf = cagr_cf - cagr
    print(f"\n  ── 資本効率改善余地 ──")
    print(f"    max_positions 3→{CF_MAX_POSITIONS}でのΔCAGR:   {cagr_gain_cf:+.2f}pp")
    print(f"    この改善は max_positions 解除の理論上限を示す。")
    print(f"    実際の¥3M制約下ではlot拒否が継続するため一部しか実現しない。")

    if cagr_gain_cf < 1.0:
        cap_verdict = "→ ポジション上限解除の効果は限定的(+{:.2f}pp)。エントリー不足が真のボトルネック。".format(cagr_gain_cf)
    elif cagr_gain_cf < 3.0:
        cap_verdict = "→ ポジション上限解除で+{:.2f}pp改善。ただし¥3M資本ではlot拒否も同時に発生。".format(cagr_gain_cf)
    else:
        cap_verdict = "→ ポジション上限解除で+{:.2f}pp改善見込み。資本増強と組み合わせると有効。".format(cagr_gain_cf)

    print(f"  {cap_verdict}")

    # ── 保存 ─────────────────────────────────────────────────────────
    out = ROOT / "backtests" / f"study53_opportunity_loss_{TODAY_STR}.json"

    payload = {
        "study": "Study53_OpportunityLossAnalysis",
        "date":  TODAY_STR,
        "config": "D_ATR_EQ (ATR_Extension + EQ_SCALE, VOL_ADJ除外)",
        "period": {"is_start": IS_START, "is_end": IS_END},
        "capital": CAPITAL,
        # Section 1
        "daily_stats": {
            "days_total":         days_total,
            "avg_exposure_pct":   round(avg_exp, 1),
            "avg_cash_ratio_pct": round(avg_cash_pct, 1),
            "avg_candidates":     round(avg_cand_daily, 2),
            "avg_open_slots":     round(avg_open_slots, 2),
            "days_no_cand":       days_no_cand,
            "days_no_cand_pct":   round(days_no_cand / max(1, days_total) * 100, 1),
            "days_full_cap":      days_full_cap,
            "days_full_cap_pct":  round(days_full_cap / max(1, days_total) * 100, 1),
            "days_zero_exp":      days_zero_exp,
            "days_idle_w_cand":   days_idle_w_cand,
        },
        # Section 2 — actual BT metrics
        "actual_bt": {
            "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
            "calmar": calmar, "n_trades": n_trades, "avg_exposure": avg_exp,
        },
        # Section 3 — rejection reasons
        "rejection_reasons": {
            "total_rejected":  total_rejected,
            "MAX_POS":         total_max_pos,
            "LOT_REJECT":      total_lot_rej,
            "SECTOR_CAP":      total_sec_cap,
            "CLUSTER_CAP":     total_cls_cap,
            "GROSS_EXPOSURE":  total_gross_exp,
        },
        "rejection_pct": {
            "MAX_POS":   round(pct_max_pos, 1),
            "LOT_REJECT": round(pct_lot_rej, 1),
            "CAPS":       round(pct_caps, 1),
        },
        # Section 4 — forward returns
        "forward_returns": {
            "max_pos": fwd_max_pos,
            "lot_rej": fwd_lot_rej,
            "skip":    fwd_skip,
        },
        # Section 5 — counterfactual
        "counterfactual_bt": {
            "max_positions": CF_MAX_POSITIONS,
            "cagr": cagr_cf, "sharpe": sharpe_cf, "max_dd": max_dd_cf,
            "calmar": calmar_cf, "n_trades": n_tr_cf, "avg_exposure": avg_exp_cf,
            "delta_cagr":   round(cagr_cf - cagr, 2),
            "delta_max_dd": round(max_dd_cf - max_dd, 2),
            "delta_calmar": round(calmar_cf - calmar, 3),
            "delta_exp":    round(avg_exp_cf - avg_exp, 1),
        },
        # Section 6 — opportunity loss
        "opportunity_loss": {
            "actual_profit_jpy":   int(actual_profit),
            "cf_profit_jpy":       int(cf_profit),
            "gap_jpy":             int(opportunity_loss),
            "gap_pct_capital":     round(opportunity_loss_pct, 1),
        },
        # Section 7 — diagnosis
        "diagnosis": {
            "primary":         diag,
            "cap_verdict":     cap_verdict,
            "pct_max_pos":     round(pct_max_pos, 1),
            "pct_lot_rej":     round(pct_lot_rej, 1),
            "pct_caps":        round(pct_caps, 1),
            "exp_gain_from_cap_lift_pp": round(exp_gain_from_cap_lift, 1),
        },
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Save] {out}")

    print(f"\n{'='*72}")
    print("  Study53 COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
