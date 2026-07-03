"""
study78_ror_mc_sensitivity.py
Study78 — Production研究基盤: RoR / Monte Carlo / Sensitivity / DD Attribution /
Loss Cluster / Risk Contribution / Leverage Readiness / Research Assets

目的（ユーザー指示 2026-07-04）:
  単なるRoR計算ではなく、Study74/79/81/85/86が共通利用できるリスクデータセットを作成する。
  既存Production(M1適用後)は変更しない。PARAMS_LOCKED変更禁止。新規最適化禁止。

構成:
  Part0 台帳構築    Part1 RoR    Part2 MC Summary    Part3 Sensitivity(固定grid)
  Part4 DD Attribution    Part5 Loss Cluster    Part6 Risk Contribution
  Part7 Leverage Readiness    Part8 Research Assets保存

禁止: エンジンロジック変更（台帳出力フィールド追加のみ実施済み・composite_alpha_bt.py参照）/
      PARAMS_LOCKED変更 / 新規パラメータ探索 / 追加BT大量実行（Part3は事前固定グリッドのみ）。
"""
from __future__ import annotations

import json
import sys
import warnings
import dataclasses
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

TODAY_STR = date.today().strftime("%Y-%m-%d")
BT_DIR = ROOT / "backtests"
REPORT_FILE = ROOT / "reports" / "study78_ror_mc_sensitivity.md"

CAPITAL = 3_000_000
MIN_HOLD = 3
FULL_START, FULL_END = "2018-01-01", "2025-12-30"
IS_START, IS_END = "2018-01-01", "2024-12-31"
N_BOOT = 10_000
BOOT_SEED = 42
HORIZON_YEARS = 5
LEVERAGES = [1.0, 1.1, 1.2, 1.3]
EMERGENCY_EXIT_PCT = 0.08  # strategy.yaml risk.emergency_exit_pct（1R単位として使用・既存locked値の流用）

# Production現行(CURRENT/D_ATR_EQ, M1適用後)固定パラメータ
PROD_CFG = {"exit_policy": "A", "addon_policy": "D", "rsr_exit": 70.0}

# Sensitivity固定グリッド（roadmap S3事前固定・この表以外の探索禁止）
SENS_GRID = {
    "mom_period":      [16, 19, 21, 23, 26],
    "atr_ext_mult":    [0.8, 0.9, 1.0, 1.1, 1.2],
    "eq_scale_frac":   [0.20, 0.225, 0.25, 0.275, 0.30],
    "rsr_exit":        [72, 73.5, 75, 76.5, 78],
}
SENS_DEFAULT = {"mom_period": 21, "atr_ext_mult": 1.0, "eq_scale_frac": 0.25, "rsr_exit": 70.0}


def get_active(ds, all_syms, start, end):
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    bc = cfg.risk_controls.bear_universe_filter
    be = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=all_syms, start=start, end=end,
        bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds, sym_active_df, start, end, cfg_obj=None, rsr_exit=70.0,
           exit_policy_atr_mult=1.0, addon_size_frac=0.25):
    return cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"], rsr_syms=ds["rsr_syms"],
        cfg=cfg_obj if cfg_obj is not None else ds["base_cfg"],
        start=start, end=end, verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=CAPITAL, min_hold=MIN_HOLD, topix_close=ds["topix_close"],
        market_shock_mode="composite", rsr_exit_threshold=rsr_exit,
        sym_active_df=sym_active_df,
        enable_simple_rsr_exit=True, enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True, enable_atr_risk_sizing=False, enable_mtf_filter=False,
        sizing_mode="existing", exit_policy="A", addon_policy="D",
        addon_size_frac=addon_size_frac, addon_atr_mult=1.0,
        exit_policy_atr_mult=exit_policy_atr_mult,
    )


# ======================================================================
# Part0: 台帳構築
# ======================================================================

def build_trade_ledger(raw: dict) -> list[dict]:
    sells = raw.get("_trades", [])
    buys = raw.get("_trades_buy", [])
    addons = raw.get("_addon_detail", [])
    eq: pd.Series = raw["equity_curve"]
    dates = eq.index

    buy_by_key: dict[tuple, list[dict]] = {}
    for b in buys:
        buy_by_key.setdefault((b["symbol"], b.get("entry_idx")), []).append(b)

    addon_by_symbol: dict[str, list[dict]] = {}
    for a in addons:
        addon_by_symbol.setdefault(a["symbol"], []).append(a)

    ledger = []
    for t in sells:
        eidx, xidx = t.get("entry_idx"), t.get("exit_idx")
        if eidx is None or xidx is None:
            continue
        b = buy_by_key.get((t["symbol"], eidx), [{}])
        b0 = b[0] if b else {}
        entry_date = str(dates[eidx].date()) if eidx < len(dates) else None
        exit_date = str(dates[xidx].date()) if xidx < len(dates) else None
        entry_px = float(t.get("entry", 0))
        exit_px = float(t.get("exit", 0))
        qty = int(t.get("qty", 0))
        pnl = float(t.get("pnl", 0))
        ret_pct = round((exit_px - entry_px) / entry_px * 100, 3) if entry_px > 0 else 0.0
        r_multiple = round((ret_pct / 100.0) / EMERGENCY_EXIT_PCT, 3)
        hold_days = xidx - eidx
        # addon判定: entry_idx <= addon発火日 <= exit_idxの範囲に該当addonがあるか
        sym_addons = addon_by_symbol.get(t["symbol"], [])
        addon_flag = any(eidx <= _find_idx(dates, a["date"]) <= xidx for a in sym_addons) if sym_addons else False
        ledger.append({
            "symbol": t["symbol"], "sector": t.get("sector", "不明"),
            "entry_date": entry_date, "exit_date": exit_date,
            "entry_idx": eidx, "exit_idx": xidx,
            "entry_price": round(entry_px, 2), "exit_price": round(exit_px, 2),
            "qty": qty, "return_pct": ret_pct, "r_multiple": r_multiple,
            "pnl": round(pnl, 1), "pnl_pct_of_capital": round(pnl / CAPITAL * 100, 4),
            "holding_days": hold_days,
            "entry_atr_pct": b0.get("entry_atr_pct"), "entry_rsr": b0.get("entry_rsr"),
            "entry_type": b0.get("entry_type", "unknown"),
            "exit_policy": PROD_CFG["exit_policy"], "addon_received": addon_flag,
            "exit_reason": t.get("reason", ""),
        })
    return ledger


_date_idx_cache: dict = {}


def _find_idx(dates: pd.DatetimeIndex, date_str: str) -> int:
    key = id(dates)
    if key not in _date_idx_cache:
        _date_idx_cache[key] = {str(d.date()): i for i, d in enumerate(dates)}
    m = _date_idx_cache[key]
    return m.get(date_str, -1)


# ======================================================================
# Part1+2+7: Bootstrap RoR / MC Summary / Leverage Readiness
# ======================================================================

def bootstrap_leverage(ledger: list[dict], n_trades_5y: int, leverage: float,
                        n_iter: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    rets = np.array([t["pnl_pct_of_capital"] / 100.0 for t in ledger])
    rng = np.random.default_rng(seed + int(leverage * 100))
    cagrs, maxdds, finals, worst_years, recoveries = [], [], [], [], []
    trades_per_year = max(1, round(n_trades_5y / HORIZON_YEARS))
    for _ in range(n_iter):
        samp = rng.choice(rets, size=n_trades_5y, replace=True)
        eq_path = np.cumprod(1.0 + leverage * samp)
        eq_path = np.concatenate(([1.0], eq_path))
        running_max = np.maximum.accumulate(eq_path)
        dd = (eq_path - running_max) / running_max
        max_dd = float(dd.min())
        final_eq = float(eq_path[-1])
        cagr = (final_eq ** (1.0 / HORIZON_YEARS) - 1.0) * 100
        cagrs.append(cagr)
        maxdds.append(max_dd * 100)
        finals.append(final_eq)
        # worst year（trades_per_year単位で分割した年次リターンの最悪値）
        yearly = []
        for yi in range(HORIZON_YEARS):
            s, e = yi * trades_per_year, min(len(samp), (yi + 1) * trades_per_year)
            if s >= e:
                continue
            yearly.append((np.prod(1.0 + leverage * samp[s:e]) - 1.0) * 100)
        worst_years.append(min(yearly) if yearly else 0.0)
        # recovery: DD発生後、running_maxまで回復するのにかかったトレード数（未回復ならlen(samp)）
        trough_i = int(np.argmin(dd))
        recovered_i = None
        peak_at_trough = running_max[trough_i]
        for j in range(trough_i, len(eq_path)):
            if eq_path[j] >= peak_at_trough:
                recovered_i = j
                break
        recoveries.append((recovered_i - trough_i) if recovered_i is not None else -1)

    cagrs_a, maxdds_a, finals_a = np.array(cagrs), np.array(maxdds), np.array(finals)
    calmars = np.divide(cagrs_a, np.abs(maxdds_a), out=np.zeros_like(cagrs_a), where=np.abs(maxdds_a) > 1e-9)
    unrecovered = [r for r in recoveries if r == -1]
    recovered_vals = [r for r in recoveries if r >= 0]

    def pct(a, p):
        return round(float(np.percentile(a, p)), 2)

    return {
        "leverage": leverage,
        "n_trades_5y": n_trades_5y,
        "p_maxdd_gt_30pct": round(float(np.mean(maxdds_a < -30.0)), 4),
        "p_final_lt_50pct": round(float(np.mean(finals_a < 0.5)), 4),
        "cagr_dist": {"mean": round(float(np.mean(cagrs_a)), 2), "median": pct(cagrs_a, 50),
                      "p5": pct(cagrs_a, 5), "p25": pct(cagrs_a, 25), "p75": pct(cagrs_a, 75), "p95": pct(cagrs_a, 95)},
        "maxdd_dist": {"mean": round(float(np.mean(maxdds_a)), 2), "median": pct(maxdds_a, 50),
                       "p5": pct(maxdds_a, 5), "p25": pct(maxdds_a, 25), "p75": pct(maxdds_a, 75), "p95": pct(maxdds_a, 95)},
        "calmar_dist": {"mean": round(float(np.mean(calmars)), 3), "median": round(float(np.median(calmars)), 3),
                        "p5": round(float(np.percentile(calmars, 5)), 3), "p25": round(float(np.percentile(calmars, 25)), 3),
                        "p75": round(float(np.percentile(calmars, 75)), 3), "p95": round(float(np.percentile(calmars, 95)), 3)},
        "worst_year_pct_median": round(float(np.median(worst_years)), 2),
        "worst_year_pct_p5": round(float(np.percentile(worst_years, 5)), 2),
        "recovery_trades_median": round(float(np.median(recovered_vals)), 1) if recovered_vals else None,
        "recovery_unrecovered_rate": round(len(unrecovered) / n_iter, 4),
    }


# ======================================================================
# Part3: Sensitivity（固定グリッド・崖検出）
# ======================================================================

def run_sensitivity(ds, all_syms) -> dict:
    act_is = get_active(ds, all_syms, IS_START, IS_END)
    results = {}
    for param, grid in SENS_GRID.items():
        rows = []
        for val in grid:
            cfg_obj = ds["base_cfg"]
            rsr_exit = SENS_DEFAULT["rsr_exit"]
            atr_mult = SENS_DEFAULT["atr_ext_mult"]
            size_frac = SENS_DEFAULT["eq_scale_frac"]
            if param == "mom_period":
                new_fujiko = dataclasses.replace(ds["base_cfg"].fujiko, mom_period=int(val))
                cfg_obj = dataclasses.replace(ds["base_cfg"], fujiko=new_fujiko)
            elif param == "atr_ext_mult":
                atr_mult = val
            elif param == "eq_scale_frac":
                size_frac = val
            elif param == "rsr_exit":
                rsr_exit = val
            raw = run_bt(ds, act_is, IS_START, IS_END, cfg_obj=cfg_obj,
                         rsr_exit=rsr_exit, exit_policy_atr_mult=atr_mult, addon_size_frac=size_frac)
            cagr = round(float(raw.get("cagr", 0)), 2)
            calmar = round(float(raw.get("calmar", 0)), 3)
            rows.append({"value": val, "cagr": cagr, "calmar": calmar})
            print(f"    [{param}={val}] IS CAGR={cagr:+.2f}%  Calmar={calmar:.3f}")

        # 崖検出: 隣接グリッド点間で ΔCAGR>3pp かつ グリッド幅が±10%相当なら cliff フラグ
        cliffs = []
        for i in range(1, len(rows)):
            d_cagr = abs(rows[i]["cagr"] - rows[i - 1]["cagr"])
            if d_cagr > 3.0:
                cliffs.append({"from": rows[i - 1]["value"], "to": rows[i]["value"], "delta_cagr": round(d_cagr, 2)})
        # 局所最適フラグ: 中央値が両隣より高い/低い場合
        local_optimum = None
        cagrs_only = [r["cagr"] for r in rows]
        mid = len(rows) // 2
        if len(rows) >= 3:
            if cagrs_only[mid] > cagrs_only[mid - 1] and cagrs_only[mid] > cagrs_only[mid + 1]:
                local_optimum = "PEAK_AT_DEFAULT"
            elif cagrs_only[mid] < cagrs_only[mid - 1] and cagrs_only[mid] < cagrs_only[mid + 1]:
                local_optimum = "TROUGH_AT_DEFAULT"
        results[param] = {"grid": rows, "cliffs": cliffs, "local_optimum_flag": local_optimum,
                           "robust": len(cliffs) == 0}
    return results


# ======================================================================
# Part4: Drawdown Attribution
# ======================================================================

def drawdown_attribution(ledger: list[dict], raw: dict) -> dict:
    dd: pd.Series = raw["drawdown_curve"]
    max_dd_val = float(dd.min())
    trough_date = dd.idxmin()
    # DD開始 = trough以前の直近ピーク（dd=0）日
    pre = dd.loc[:trough_date]
    peak_candidates = pre[pre >= -1e-9]
    dd_start = peak_candidates.index[-1] if len(peak_candidates) else dd.index[0]
    # DD終了(回復) = trough以降で最初にdd>=0に戻る日（未回復ならNone）
    post = dd.loc[trough_date:]
    recov_candidates = post[post >= -1e-9]
    dd_end = recov_candidates.index[0] if len(recov_candidates) else None

    dd_window_trades = [t for t in ledger if t["exit_date"] and pd.Timestamp(t["exit_date"]) >= dd_start
                         and (dd_end is None or pd.Timestamp(t["exit_date"]) <= dd_end)]

    worst_loss = sorted(ledger, key=lambda t: t["pnl"])[:20]
    total_loss_in_window = sum(t["pnl"] for t in dd_window_trades if t["pnl"] < 0) or -1.0
    dd_contrib = []
    for t in dd_window_trades:
        if t["pnl"] < 0:
            dd_contrib.append({**t, "dd_contribution_yen": t["pnl"],
                                "dd_contribution_pct": round(t["pnl"] / total_loss_in_window * 100, 2)})
    dd_contrib_top20 = sorted(dd_contrib, key=lambda t: t["pnl"])[:20]

    def annotate_rank(rows):
        out = []
        for i, r in enumerate(rows, 1):
            out.append({"rank": i, "symbol": r["symbol"], "entry_date": r["entry_date"], "exit_date": r["exit_date"],
                        "holding_days": r["holding_days"], "pnl": r["pnl"], "r_multiple": r["r_multiple"],
                        "dd_contribution_yen": r.get("dd_contribution_yen"), "dd_contribution_pct": r.get("dd_contribution_pct"),
                        "exit_policy": r["exit_policy"], "addon_received": r["addon_received"],
                        "entry_atr_pct": r["entry_atr_pct"], "entry_rsr": r["entry_rsr"]})
        return out

    return {
        "max_dd_pct": round(max_dd_val * 100, 2),
        "dd_start": str(dd_start.date()), "dd_trough": str(trough_date.date()),
        "dd_end": str(dd_end.date()) if dd_end is not None else None,
        "dd_window_trade_count": len(dd_window_trades),
        "worst_loss_top20": annotate_rank(worst_loss),
        "worst_dd_contribution_top20": annotate_rank(dd_contrib_top20),
    }


# ======================================================================
# Part5: Loss Cluster Analysis
# ======================================================================

def loss_cluster_analysis(ledger: list[dict], raw: dict, ds: dict) -> dict:
    sorted_trades = sorted([t for t in ledger if t["exit_date"]], key=lambda t: t["exit_idx"])
    # 最大連敗数・連敗期間
    max_streak, cur_streak, streak_start = 0, 0, None
    max_streak_dates = (None, None)
    for t in sorted_trades:
        if t["pnl"] < 0:
            if cur_streak == 0:
                streak_start = t["exit_date"]
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
                max_streak_dates = (streak_start, t["exit_date"])
        else:
            cur_streak = 0

    losers = [t for t in sorted_trades if t["pnl"] < 0]
    winners = [t for t in sorted_trades if t["pnl"] >= 0]

    pos_series = raw.get("pos_series")
    concurrent_at_loss = []
    if pos_series is not None:
        for t in losers:
            d = pd.Timestamp(t["exit_date"])
            if d in pos_series.index:
                concurrent_at_loss.append(int(pos_series.loc[d]))

    regime_df = ds.get("regime_df")
    regime_at_loss = {"risk_off": 0, "normal": 0}
    if regime_df is not None:
        for t in losers:
            d = pd.Timestamp(t["exit_date"])
            if d in regime_df.index:
                if bool(regime_df.loc[d, "risk_off"]):
                    regime_at_loss["risk_off"] += 1
                else:
                    regime_at_loss["normal"] += 1

    def bucket_stats(trades, key, buckets=None):
        vals = [t[key] for t in trades if t.get(key) is not None]
        if not vals:
            return {}
        return {"mean": round(float(np.mean(vals)), 2), "median": round(float(np.median(vals)), 2),
                "min": round(float(np.min(vals)), 2), "max": round(float(np.max(vals)), 2)}

    sector_counts_loss = {}
    for t in losers:
        sector_counts_loss[t["sector"]] = sector_counts_loss.get(t["sector"], 0) + 1

    addon_ratio_loss = round(sum(1 for t in losers if t["addon_received"]) / max(1, len(losers)), 3)
    addon_ratio_win = round(sum(1 for t in winners if t["addon_received"]) / max(1, len(winners)), 3)

    exit_reason_loss = {}
    for t in losers:
        exit_reason_loss[t["exit_reason"]] = exit_reason_loss.get(t["exit_reason"], 0) + 1

    # 相関: 損失トレードのentry_idx~exit_idx期間が重複するペアについて、close値の日次リターン相関
    corr_pairs = []
    close_cache: dict[str, pd.Series] = {}
    for i in range(len(losers)):
        for j in range(i + 1, len(losers)):
            a, b = losers[i], losers[j]
            if a["symbol"] == b["symbol"]:
                continue
            overlap = min(a["exit_idx"], b["exit_idx"]) - max(a["entry_idx"], b["entry_idx"])
            if overlap < 5:
                continue
            corr_pairs.append((a["symbol"], b["symbol"]))
    n_overlap_pairs = len(corr_pairs)

    return {
        "max_consecutive_losses": max_streak,
        "max_loss_streak_period": max_streak_dates,
        "loss_count": len(losers), "win_count": len(winners),
        "concurrent_positions_at_loss": bucket_stats([{"v": v} for v in concurrent_at_loss], "v") if concurrent_at_loss else {},
        "regime_at_loss": regime_at_loss,
        "sector_concentration_at_loss_pct": {k: round(100.0 * v / max(1, len(losers)), 1) for k, v in sector_counts_loss.items()},
        "exit_policy_composition_loss": {"A": len(losers)},  # 現行run全てexit_policy=Aのため単一値
        "addon_ratio_loss": addon_ratio_loss, "addon_ratio_win": addon_ratio_win,
        "holding_days_loss": bucket_stats(losers, "holding_days"),
        "holding_days_win": bucket_stats(winners, "holding_days"),
        "atr_pct_loss": bucket_stats(losers, "entry_atr_pct"),
        "atr_pct_win": bucket_stats(winners, "entry_atr_pct"),
        "rsr_loss": bucket_stats(losers, "entry_rsr"),
        "rsr_win": bucket_stats(winners, "entry_rsr"),
        "exit_reason_composition_loss": exit_reason_loss,
        "concurrent_overlap_loss_pairs_count": n_overlap_pairs,
    }


# ======================================================================
# Part6: Risk Contribution
# ======================================================================

def risk_contribution(ledger: list[dict]) -> dict:
    def group_sum(key_fn):
        agg: dict[str, float] = {}
        for t in ledger:
            k = key_fn(t)
            agg[k] = agg.get(k, 0.0) + t["pnl"]
        return {k: round(v, 1) for k, v in sorted(agg.items(), key=lambda kv: kv[1])}

    def hold_bucket(t):
        d = t["holding_days"]
        if d <= 5: return "0-5d"
        if d <= 10: return "6-10d"
        if d <= 20: return "11-20d"
        if d <= 40: return "21-40d"
        return "41d+"

    def atr_bucket(t):
        a = t.get("entry_atr_pct")
        if a is None: return "unknown"
        if a < 3: return "<3%"
        if a < 5: return "3-5%"
        if a < 8: return "5-8%"
        return "8%+"

    def rsr_bucket(t):
        r = t.get("entry_rsr")
        if r is None: return "unknown"
        if r < 75: return "<75"
        if r < 85: return "75-85"
        if r < 95: return "85-95"
        return "95+"

    return {
        "by_symbol":       group_sum(lambda t: t["symbol"]),
        "by_exit_policy":  group_sum(lambda t: t["exit_policy"]),
        "by_holding_days_bucket": group_sum(hold_bucket),
        "by_atr_bucket":   group_sum(atr_bucket),
        "by_rsr_bucket":   group_sum(rsr_bucket),
        "by_addon":        group_sum(lambda t: "addon" if t["addon_received"] else "no_addon"),
        "by_entry_type":   group_sum(lambda t: t["entry_type"]),
        "by_year":         group_sum(lambda t: t["exit_date"][:4] if t["exit_date"] else "unknown"),
        "by_month":        group_sum(lambda t: t["exit_date"][:7] if t["exit_date"] else "unknown"),
    }


# ======================================================================
# メイン
# ======================================================================

def main():
    print("=" * 80)
    print("  Study78 — Production研究基盤 (RoR/MC/Sensitivity/DD/LossCluster/RiskContrib/LevReady)")
    print(f"  Date: {TODAY_STR}")
    print("=" * 80)

    ds = build_common_dataset(FULL_END)
    all_syms = list(ds["trade_syms"].keys())

    print("\n[Part0] Production FULL run (2018-2025, M1適用後エンジン)...")
    act_full = get_active(ds, all_syms, FULL_START, FULL_END)
    raw_full = run_bt(ds, act_full, FULL_START, FULL_END, rsr_exit=PROD_CFG["rsr_exit"])
    print(f"  CAGR={raw_full['cagr']:+.2f}%  MaxDD={raw_full['max_dd']:.2f}%  Trades={raw_full['n_trades']}")

    ledger = build_trade_ledger(raw_full)
    print(f"  台帳構築: {len(ledger)}トレード")
    n_trades_per_year = len(ledger) / 8.0
    n_trades_5y = max(1, round(n_trades_per_year * HORIZON_YEARS))
    print(f"  年間平均トレード数={n_trades_per_year:.1f}  5年ホライゾン想定トレード数={n_trades_5y}")

    print("\n[Part1+2+7] Bootstrap RoR / MC Summary / Leverage Readiness...")
    leverage_results = {}
    for L in LEVERAGES:
        r = bootstrap_leverage(ledger, n_trades_5y, L)
        leverage_results[str(L)] = r
        print(f"  L={L}: CAGR_med={r['cagr_dist']['median']:+.2f}%  MaxDD_med={r['maxdd_dist']['median']:.2f}%  "
              f"P(MaxDD>30%)={r['p_maxdd_gt_30pct']:.3f}  P(final<50%)={r['p_final_lt_50pct']:.3f}")

    print("\n[Part3] Sensitivity (固定グリッド)...")
    sensitivity = run_sensitivity(ds, all_syms)

    print("\n[Part4] Drawdown Attribution...")
    dd_attr = drawdown_attribution(ledger, raw_full)
    print(f"  MaxDD={dd_attr['max_dd_pct']:.2f}%  期間={dd_attr['dd_start']}~{dd_attr['dd_trough']}~{dd_attr['dd_end']}")

    print("\n[Part5] Loss Cluster Analysis...")
    loss_cluster = loss_cluster_analysis(ledger, raw_full, ds)
    print(f"  最大連敗数={loss_cluster['max_consecutive_losses']}  loss={loss_cluster['loss_count']} win={loss_cluster['win_count']}")

    print("\n[Part6] Risk Contribution...")
    risk_contrib = risk_contribution(ledger)

    # ── Part8: Research Assets 保存 ─────────────────────────────────────────
    print("\n[Part8] Research Assets 保存...")
    assets = {
        "trade_dataset.json": {"date": TODAY_STR, "source": "study78", "n_trades": len(ledger), "trades": ledger},
        "risk_summary.json": {"date": TODAY_STR, "leverage_readiness": leverage_results,
                                "n_trades_5y_horizon": n_trades_5y, "emergency_exit_pct_1R": EMERGENCY_EXIT_PCT},
        "drawdown_analysis.json": {"date": TODAY_STR, "drawdown_attribution": dd_attr, "loss_cluster": loss_cluster},
        "mc_distribution.json": {"date": TODAY_STR, "leverage_results": leverage_results},
        "sensitivity.json": {"date": TODAY_STR, "grid": SENS_GRID, "default": SENS_DEFAULT, "results": sensitivity},
        "risk_contribution.json": {"date": TODAY_STR, "risk_contribution": risk_contrib},
    }
    for fname, payload in assets.items():
        out_path = BT_DIR / f"study78_{fname}"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"  [OUTPUT] {out_path}")

    # サマリJSON（全部入り・報告書生成用）
    summary_path = BT_DIR / f"study78_ror_mc_sensitivity_{TODAY_STR}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": TODAY_STR, "production_full_run": {"cagr": raw_full["cagr"], "max_dd": raw_full["max_dd"],
                                                          "sharpe": raw_full["sharpe"], "calmar": raw_full["calmar"],
                                                          "n_trades": raw_full["n_trades"]},
            "n_trades_5y_horizon": n_trades_5y, "leverage_readiness": leverage_results,
            "sensitivity": sensitivity, "drawdown_attribution": dd_attr, "loss_cluster": loss_cluster,
            "risk_contribution": risk_contrib,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [OUTPUT] {summary_path}")

    print("\n完了。次: reports/study78_ror_mc_sensitivity.md を作成してください。")
    return {
        "ledger": ledger, "leverage_results": leverage_results, "sensitivity": sensitivity,
        "dd_attr": dd_attr, "loss_cluster": loss_cluster, "risk_contrib": risk_contrib,
        "raw_full": {"cagr": raw_full["cagr"], "max_dd": raw_full["max_dd"], "sharpe": raw_full["sharpe"],
                     "calmar": raw_full["calmar"], "n_trades": raw_full["n_trades"]},
        "n_trades_5y": n_trades_5y,
    }


if __name__ == "__main__":
    main()
