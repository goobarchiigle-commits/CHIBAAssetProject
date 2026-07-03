"""
study70_addon_portfolio_economic_audit.py
Study70 — Add-on Portfolio Economic Audit

目的: Study65観測値(+3.16pp realistic)がポートフォリオ視点でも成立するか検証。
     資本拘束コスト / 機会費用 / max_positions=3制約 / 再投資遅延を全て含む。

Phase0: Integrity
Phase1: 4シナリオBT (NO_ADDON / 0.25 / 0.50 / 1.00)
Phase2: Add-on Attribution (addon trade P&L / BW vs NonBW)
Phase3: Opportunity Cost Analysis (displaced entries)
Phase4: Portfolio NEV (Add-on Gain - Opportunity Cost)
Phase5: Unit Size Comparison (0.25/0.50/1.00 vs NO_ADDON)
Phase6: Regime Analysis
Phase7: Verdict (Shadow Add-on価値判定)

禁止: 売買ルール作成 / 閾値最適化 / Production変更 / Lookahead
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
from scipy.stats import mannwhitneyu

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
from src.backtest.study61_return_distribution_anatomy import get_active
import src.backtest.composite_alpha_bt as cab

TODAY_STR  = date.today().strftime("%Y-%m-%d")
CAPITAL    = 3_000_000
IS_START   = "2018-01-01"
IS_END     = "2024-12-31"
OOS_START  = "2025-01-01"
DATA_END   = "2025-12-31"
MIN_HOLD   = 3
MAX_POS    = 3

EP_EXIT        = "A"
EP_ADDON       = "D"
ADDON_ATR_MULT = 1.0

# Study参照値
S65_REALISTIC_PP = 3.16    # Study65 Add-on realistic expected ΔCAGR
S64_ADDON_PP     = 6.78    # Study64 BW-only add-on oracle
S67_NEV_PORTFOLIO = -313_271  # Study67 portfolio NEV

RSR_REASONS = {"RSR_EXIT", "RSR_MOMENTUM_EXIT"}
ADDON_KEYWORDS = {"ADDON", "ADD_ON", "WINNER_ADDON", "ADDON_ENTRY", "ADD"}

OUT_FILE = ROOT / "backtests" / f"study70_addon_portfolio_economic_audit_{TODAY_STR}.json"


# ======================================================================
# ユーティリティ
# ======================================================================

def _ss(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    return v


def _stats_arr(arr, label=""):
    a = np.array([x for x in arr if x is not None], dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"label": label, "n": 0}
    return {
        "label":  label,
        "n":      int(len(a)),
        "mean":   _ss(float(np.mean(a))),
        "median": _ss(float(np.median(a))),
        "std":    _ss(float(np.std(a, ddof=1))) if len(a) > 1 else 0.0,
        "min":    _ss(float(np.min(a))),
        "max":    _ss(float(np.max(a))),
    }


# ======================================================================
# BT実行
# ======================================================================

def _run_bt_scenario(ds, sym_active_df, start, end,
                     addon_max_per_pos: int, addon_size_frac: float) -> dict:
    return cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"], rsr_syms=ds["rsr_syms"],
        cfg=ds["base_cfg"], start=start, end=end, verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=CAPITAL, min_hold=MIN_HOLD, topix_close=ds["topix_close"],
        market_shock_mode="composite", rsr_exit_threshold=70.0,
        sym_active_df=sym_active_df,
        enable_simple_rsr_exit=True, enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True, enable_atr_risk_sizing=False,
        enable_mtf_filter=False, sizing_mode="existing",
        exit_policy=EP_EXIT, exit_policy_atr_mult=ADDON_ATR_MULT,
        exit_policy_defer_days=5, max_positions_ts=None,
        addon_policy=EP_ADDON,
        addon_atr_mult=ADDON_ATR_MULT,
        addon_stage2_mult=2.0,
        addon_max_per_pos=addon_max_per_pos,
        addon_size_frac=addon_size_frac,
    )


def _extract_metrics(bt_result: dict) -> dict:
    """BT結果からポートフォリオ指標を抽出。"""
    stats = bt_result.get("stats") or bt_result
    cagr = (stats.get("cagr")
            or stats.get("annual_return")
            or stats.get("cagr_pct")
            or stats.get("annual_return_pct"))
    calmar = (stats.get("calmar")
              or stats.get("calmar_ratio"))
    max_dd = (stats.get("max_dd")
              or stats.get("max_drawdown")
              or stats.get("max_dd_pct"))
    sharpe = stats.get("sharpe") or stats.get("sharpe_ratio")
    n_trades = len(bt_result.get("_trades", []))
    return {
        "cagr":     _ss(cagr),
        "calmar":   _ss(calmar),
        "max_dd":   _ss(max_dd),
        "sharpe":   _ss(sharpe),
        "n_trades": n_trades,
    }


def _extract_trades(bt_result: dict, calendar_dates, rsr_df) -> list:
    trades = bt_result.get("_trades", [])
    out = []
    for t in trades:
        ei   = int(t.get("entry_idx", 0))
        xi   = int(t.get("exit_idx", ei))
        sym  = t.get("symbol", "")
        if not sym or ei >= len(calendar_dates):
            continue
        entry_date = calendar_dates[ei]
        exit_date  = calendar_dates[min(xi, len(calendar_dates) - 1)]
        reason     = t.get("reason", "UNKNOWN")
        is_addon   = any(kw in reason.upper() for kw in ADDON_KEYWORDS)
        out.append({
            "symbol":     sym,
            "entry_date": pd.Timestamp(entry_date),
            "exit_date":  pd.Timestamp(exit_date),
            "reason":     reason,
            "is_addon":   is_addon,
            "pnl":        float(t.get("pnl", 0)),
            "qty":        float(t.get("qty", 0)),
            "entry_px":   float(t.get("entry", 0)),
            "exit_px":    float(t.get("exit", 0)),
        })
    out.sort(key=lambda x: x["entry_date"])
    return out


# ======================================================================
# 銘柄fwd return
# ======================================================================

def _fwd_return(ds, sym, from_date, n_days):
    df_c = ds["universe_raw"].get(sym, {}).get("df")
    if df_c is None or "Close" not in df_c.columns:
        return np.nan
    close = df_c["Close"].dropna()
    close.index = pd.to_datetime(close.index)
    base   = close[close.index <= from_date]
    future = close[close.index > from_date]
    if base.empty or len(future) < n_days:
        return np.nan
    bp = float(base.iloc[-1])
    if bp <= 0:
        return np.nan
    return (float(future.iloc[n_days - 1]) / bp - 1.0) * 100.0


# ======================================================================
# 全データセット構築
# ======================================================================

def build_full_dataset(ds, all_trades_raw):
    """Study67/68と同一の291件フィルタで全取引を構築。"""
    records = []
    universe_raw = ds["universe_raw"]
    for tr in all_trades_raw:
        sym        = tr["symbol"]
        entry_date = tr["entry_date"]
        exit_date  = tr["exit_date"]
        reason     = tr["reason"]
        if sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close = df_c["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        entry_base   = close[close.index <= entry_date]
        entry_future = close[close.index > entry_date]
        if entry_base.empty or len(entry_future) < 60:
            continue
        entry_px = float(entry_base.iloc[-1])
        if entry_px <= 0:
            continue
        fwd60 = (float(entry_future.iloc[59]) / entry_px - 1.0) * 100.0
        exit_base = close[close.index <= exit_date]
        ex_px = float(exit_base.iloc[-1]) if not exit_base.empty else tr.get("exit_px", 0)
        pos_val = ex_px * tr.get("qty", 0)
        records.append({
            "symbol":     sym,
            "entry_date": entry_date,
            "exit_date":  exit_date,
            "reason":     reason,
            "pnl":        tr.get("pnl", 0),
            "fwd60d":     _ss(fwd60),
            "pos_val":    _ss(pos_val),
            "qty":        tr.get("qty", 0),
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    p90 = float(df["fwd60d"].dropna().quantile(0.90))
    df["is_big_winner"] = df["fwd60d"] >= p90
    df["is_rsr_exit"]   = df["reason"].isin(RSR_REASONS)
    return df


# ======================================================================
# Phase0: Integrity
# ======================================================================

def phase0_integrity(df: pd.DataFrame) -> dict:
    n_total = len(df)
    n_bw    = int(df["is_big_winner"].sum())
    n_rsr   = int(df["is_rsr_exit"].sum())
    n_bw_rsr = int(df[df["is_rsr_exit"]]["is_big_winner"].sum())
    return {
        "n_total":         n_total,
        "n_match_291":     n_total == 291,
        "n_bw":            n_bw,
        "n_bw_match_30":   n_bw == 30,
        "n_rsr":           n_rsr,
        "n_rsr_match_248": n_rsr == 248,
        "n_bw_rsr":        n_bw_rsr,
        "lookahead":       0,
    }


# ======================================================================
# Phase1: 4シナリオBT
# ======================================================================

UNIT_CONFIGS = {
    "NO_ADDON": {"addon_max_per_pos": 0, "addon_size_frac": 0.00},
    "UNIT_025": {"addon_max_per_pos": 1, "addon_size_frac": 0.25},
    "UNIT_050": {"addon_max_per_pos": 1, "addon_size_frac": 0.50},
    "UNIT_100": {"addon_max_per_pos": 1, "addon_size_frac": 1.00},
}


def phase1_scenarios(ds, all_dates) -> dict:
    """4シナリオのIS+OOS BT実行。"""
    results = {}
    for label, cfg in UNIT_CONFIGS.items():
        print(f"  {label} IS BT...")
        sym_is = get_active(ds, IS_START, IS_END)
        bt_is  = _run_bt_scenario(ds, sym_is, IS_START, IS_END,
                                   cfg["addon_max_per_pos"], cfg["addon_size_frac"])
        is_dates = all_dates[(all_dates >= IS_START) & (all_dates <= IS_END)]
        tr_is    = _extract_trades(bt_is, is_dates, ds["rsr_df"])

        print(f"  {label} OOS BT...")
        sym_oos = get_active(ds, OOS_START, DATA_END)
        bt_oos  = _run_bt_scenario(ds, sym_oos, OOS_START, DATA_END,
                                    cfg["addon_max_per_pos"], cfg["addon_size_frac"])
        oos_dates = all_dates[(all_dates >= OOS_START) & (all_dates <= DATA_END)]
        tr_oos    = _extract_trades(bt_oos, oos_dates, ds["rsr_df"])

        m_is   = _extract_metrics(bt_is)
        m_oos  = _extract_metrics(bt_oos)
        results[label] = {
            "config":     cfg,
            "is_metrics": m_is,
            "oos_metrics": m_oos,
            "is_trades":  tr_is,
            "oos_trades": tr_oos,
        }
        print(f"    IS: CAGR={m_is['cagr']}  Calmar={m_is['calmar']}  MaxDD={m_is['max_dd']}")
        print(f"    OOS: CAGR={m_oos['cagr']}  Calmar={m_oos['calmar']}  MaxDD={m_oos['max_dd']}")

    return results


# ======================================================================
# Phase2: Add-on Attribution
# ======================================================================

def phase2_addon_attribution(scenarios: dict, df_base: pd.DataFrame, ds) -> dict:
    """UNIT_025のadd-on取引をBW/NonBW別に解剖。"""
    unit025 = scenarios["UNIT_025"]
    all_trades = unit025["is_trades"] + unit025["oos_trades"]

    # add-on識別 (reason or relative entry timing)
    addon_trades  = [t for t in all_trades if t["is_addon"]]
    base_trades   = [t for t in all_trades if not t["is_addon"]]

    # base tradesとBW status照合
    bw_set = set()
    for _, row in df_base[df_base["is_big_winner"]].iterrows():
        bw_set.add((row["symbol"], pd.Timestamp(row["entry_date"]).date()))

    # add-on取引の親ポジションBW判定 (同symbol・entry_date最近接)
    addon_bw, addon_nonbw = [], []
    for t in addon_trades:
        sym  = t["symbol"]
        # 同symbolで直近のbase trade entryを探す (add-on entryの前)
        parent_date = None
        for bt in sorted(base_trades, key=lambda x: x["entry_date"], reverse=True):
            if bt["symbol"] == sym and bt["entry_date"] <= t["entry_date"]:
                parent_date = bt["entry_date"].date()
                break
        if parent_date and (sym, parent_date) in bw_set:
            addon_bw.append(t)
        else:
            addon_nonbw.append(t)

    def _trade_stats(tlist, label):
        if not tlist:
            return {"label": label, "n": 0, "total_pnl": 0,
                    "pnl_mean": None, "pnl_positive_pct": None}
        pnls = [t["pnl"] for t in tlist]
        return {
            "label":           label,
            "n":               len(tlist),
            "total_pnl":       _ss(sum(pnls)),
            "pnl_mean":        _ss(float(np.mean(pnls))),
            "pnl_std":         _ss(float(np.std(pnls))) if len(pnls) > 1 else None,
            "pnl_positive_pct": _ss(sum(1 for p in pnls if p > 0) / len(pnls) * 100),
        }

    return {
        "note_addon_identification": (
            "reason中のADON keyword。0件の場合はBT側がadon reasonを使わない可能性あり。"
            "その場合はPhase1 ΔCAGR比較を主評価とする。"
        ),
        "all_addon_trades":   _trade_stats(addon_trades,  "ALL_ADDON"),
        "bw_addon_trades":    _trade_stats(addon_bw,      "BW_ADDON"),
        "nonbw_addon_trades": _trade_stats(addon_nonbw,   "NONBW_ADDON"),
        "n_addon_raw":        len(addon_trades),
        "n_base_raw":         len(base_trades),
        "addon_pnl_total":    _ss(sum(t["pnl"] for t in addon_trades)),
        "base_pnl_total":     _ss(sum(t["pnl"] for t in base_trades)),
    }


# ======================================================================
# Phase3: Opportunity Cost Analysis
# ======================================================================

def phase3_opportunity_cost(scenarios: dict, ds) -> dict:
    """
    NO_ADDON vs UNIT_025の取引リスト比較。
    NO_ADDONに存在してUNIT_025に存在しない新規エントリ = displaced entries。
    """
    no_addon_trades = scenarios["NO_ADDON"]["is_trades"] + scenarios["NO_ADDON"]["oos_trades"]
    addon_trades    = scenarios["UNIT_025"]["is_trades"] + scenarios["UNIT_025"]["oos_trades"]

    # base (non-addon) trades only
    no_addon_base = [t for t in no_addon_trades if not t["is_addon"]]
    addon_base    = [t for t in addon_trades    if not t["is_addon"]]

    # エントリをmatch: (symbol, entry_date) で照合
    def _entry_key(t):
        return (t["symbol"], t["entry_date"].date())

    no_addon_keys = set(_entry_key(t) for t in no_addon_base)
    addon_keys    = set(_entry_key(t) for t in addon_base)

    displaced_keys = no_addon_keys - addon_keys  # NO_ADDONのみに存在

    displaced_trades = [t for t in no_addon_base if _entry_key(t) in displaced_keys]

    # displaced tradesのfwd40d return (exit点からではなく entry点からの40d)
    displaced_fwd40 = []
    displaced_pnl   = []
    for t in displaced_trades:
        fwd = _fwd_return(ds, t["symbol"], t["entry_date"], 40)
        displaced_fwd40.append(_ss(fwd))
        displaced_pnl.append(t["pnl"])

    # 資本拘束コスト: displaced entries × avg_position_size × avg_fwd40d
    avg_pos_size = CAPITAL / MAX_POS  # ¥1,000,000
    valid_fwd = [f for f in displaced_fwd40 if f is not None]
    avg_fwd40 = float(np.mean(valid_fwd)) if valid_fwd else 0.0
    opportunity_cost_jpy = len(displaced_trades) * avg_pos_size * avg_fwd40 / 100

    # 再投資遅延 (add-on exit → 次の新規エントリまでの日数)
    addon_only = [t for t in addon_trades if t["is_addon"]]
    redeploy_delays = []
    for ao in addon_only:
        sym = ao["symbol"]
        ao_exit = ao["exit_date"]
        next_entry = None
        for t in sorted(addon_base, key=lambda x: x["entry_date"]):
            if t["entry_date"] > ao_exit:
                next_entry = t["entry_date"]
                break
        if next_entry:
            delay_days = (next_entry - ao_exit).days
            redeploy_delays.append(delay_days)

    return {
        "no_addon_base_n":     len(no_addon_base),
        "addon_base_n":        len(addon_base),
        "displaced_entries_n": len(displaced_trades),
        "displaced_keys_n":    len(displaced_keys),
        "displaced_pnl_total": _ss(sum(displaced_pnl)),
        "displaced_pnl_stats": _stats_arr(displaced_pnl, "displaced_entry_pnl"),
        "displaced_fwd40_stats": _stats_arr(
            [f for f in displaced_fwd40 if f is not None], "displaced_fwd40d_%"
        ),
        "avg_position_size_jpy": _ss(avg_pos_size),
        "opportunity_cost_jpy":  _ss(opportunity_cost_jpy),
        "redeploy_delay_stats":  _stats_arr(redeploy_delays, "redeploy_delay_days"),
        "note": (
            "displaced = trades appearing in NO_ADDON but absent in UNIT_025. "
            "Opportunity cost = n_displaced × avg_pos × avg_fwd40d."
        ),
    }


# ======================================================================
# Phase4: Portfolio NEV
# ======================================================================

def phase4_portfolio_nev(p2: dict, p3: dict, scenarios: dict) -> dict:
    """
    NEV = Add-on Gain - Opportunity Cost
    ΔCAGR / ΔCalmar / ΔMaxDD は直接BT比較で算出。
    """
    addon_gain   = p2.get("addon_pnl_total") or 0.0
    opp_cost_jpy = p3.get("opportunity_cost_jpy") or 0.0
    nev_jpy      = addon_gain - opp_cost_jpy

    # 直接BT比較
    def _delta(key, base_key="NO_ADDON", cmp_key="UNIT_025", period="is"):
        base_m = scenarios[base_key].get(f"{period}_metrics", {})
        cmp_m  = scenarios[cmp_key].get(f"{period}_metrics", {})
        bv = base_m.get(key)
        cv = cmp_m.get(key)
        if bv is None or cv is None:
            return None
        return _ss(float(cv) - float(bv))

    is_dcagr   = _delta("cagr",   period="is")
    is_dcalmar = _delta("calmar", period="is")
    is_ddd     = _delta("max_dd", period="is")
    oos_dcagr  = _delta("cagr",   period="oos")
    oos_dcalmar= _delta("calmar", period="oos")
    oos_ddd    = _delta("max_dd", period="oos")

    # Study65参照値との比較
    dcagr_vs_s65 = _ss(is_dcagr - S65_REALISTIC_PP) if is_dcagr is not None else None
    dcagr_vs_s64 = _ss(is_dcagr - S64_ADDON_PP)     if is_dcagr is not None else None

    return {
        "addon_gain_jpy":    _ss(addon_gain),
        "opportunity_cost_jpy": _ss(opp_cost_jpy),
        "portfolio_nev_jpy": _ss(nev_jpy),
        "nev_positive":      nev_jpy > 0,
        "is_comparison": {
            "delta_cagr_pp":   is_dcagr,
            "delta_calmar":    is_dcalmar,
            "delta_max_dd_pp": is_ddd,
        },
        "oos_comparison": {
            "delta_cagr_pp":   oos_dcagr,
            "delta_calmar":    oos_dcalmar,
            "delta_max_dd_pp": oos_ddd,
        },
        "reference_comparison": {
            "s65_realistic_pp":  S65_REALISTIC_PP,
            "s64_oracle_bw_pp":  S64_ADDON_PP,
            "dcagr_vs_s65":      dcagr_vs_s65,
            "dcagr_vs_s64":      dcagr_vs_s64,
        },
    }


# ======================================================================
# Phase5: Unit Size Comparison
# ======================================================================

def phase5_unit_comparison(scenarios: dict) -> dict:
    """0.25 / 0.50 / 1.00 vs NO_ADDON のΔCAGR/ΔCalmar/ΔMaxDD。"""
    baseline_is  = scenarios["NO_ADDON"]["is_metrics"]
    baseline_oos = scenarios["NO_ADDON"]["oos_metrics"]

    rows = []
    for label in ["UNIT_025", "UNIT_050", "UNIT_100"]:
        sc  = scenarios[label]
        cfg = sc["config"]
        is_m  = sc["is_metrics"]
        oos_m = sc["oos_metrics"]

        def _d(key, base, cmp):
            bv = base.get(key); cv = cmp.get(key)
            return _ss(float(cv) - float(bv)) if (bv is not None and cv is not None) else None

        rows.append({
            "label":          label,
            "addon_size_frac": cfg["addon_size_frac"],
            "is_cagr":         is_m.get("cagr"),
            "is_dcagr":        _d("cagr",   baseline_is,  is_m),
            "is_dcalmar":      _d("calmar", baseline_is,  is_m),
            "is_ddd":          _d("max_dd", baseline_is,  is_m),
            "oos_cagr":        oos_m.get("cagr"),
            "oos_dcagr":       _d("cagr",   baseline_oos, oos_m),
            "oos_dcalmar":     _d("calmar", baseline_oos, oos_m),
            "oos_ddd":         _d("max_dd", baseline_oos, oos_m),
            "is_n_trades":     is_m.get("n_trades"),
            "oos_n_trades":    oos_m.get("n_trades"),
        })

    return {
        "no_addon_is_cagr":   baseline_is.get("cagr"),
        "no_addon_oos_cagr":  baseline_oos.get("cagr"),
        "unit_comparison":    rows,
        "optimal_unit": max(rows, key=lambda r: r["is_dcagr"] or -1e9).get("label")
                         if rows else None,
    }


# ======================================================================
# Phase6: Regime Analysis
# ======================================================================

def phase6_regime_analysis(scenarios: dict, ds) -> dict:
    """
    TOPIX年間リターンでBull/Bear/Sidewaysに分類してunit_025 Add-on効果を比較。
    """
    # TOPIX年間リターン計算
    topix = ds.get("topix_close")
    if topix is None:
        return {"note": "topix_close not available"}
    topix.index = pd.to_datetime(topix.index)

    regime_by_year = {}
    for yr in range(2018, 2026):
        yr_data = topix[topix.index.year == yr]
        if len(yr_data) < 20:
            continue
        ret = (float(yr_data.iloc[-1]) / float(yr_data.iloc[0]) - 1.0) * 100
        regime = "Bull" if ret > 10 else ("Bear" if ret < -10 else "Sideways")
        regime_by_year[yr] = {"topix_ret_pct": _ss(ret), "regime": regime}

    # 年別にUNIT_025 add-on取引のPNLを集計
    all_addon_trades = scenarios["UNIT_025"]["is_trades"] + scenarios["UNIT_025"]["oos_trades"]
    addon_only = [t for t in all_addon_trades if t["is_addon"]]

    regime_pnl: dict = {}
    for t in addon_only:
        yr = t["entry_date"].year
        reg_info = regime_by_year.get(yr, {})
        regime = reg_info.get("regime", "UNKNOWN")
        if regime not in regime_pnl:
            regime_pnl[regime] = []
        regime_pnl[regime].append(t["pnl"])

    regime_summary = {}
    for regime, pnls in regime_pnl.items():
        regime_summary[regime] = {
            "n":           len(pnls),
            "total_pnl":   _ss(sum(pnls)),
            "pnl_mean":    _ss(float(np.mean(pnls))),
            "positive_pct": _ss(sum(1 for p in pnls if p > 0) / len(pnls) * 100) if pnls else None,
        }

    # 年別ΔCAGR (NO_ADDON vs UNIT_025) — 直接計算は困難なのでtrade PNL比較
    # no_addon vs addon の年別PNL差
    no_addon_by_yr: dict = {}
    addon_base_by_yr: dict = {}
    for t in scenarios["NO_ADDON"]["is_trades"] + scenarios["NO_ADDON"]["oos_trades"]:
        yr = t["entry_date"].year
        no_addon_by_yr.setdefault(yr, []).append(t["pnl"])
    for t in scenarios["UNIT_025"]["is_trades"] + scenarios["UNIT_025"]["oos_trades"]:
        yr = t["entry_date"].year
        addon_base_by_yr.setdefault(yr, []).append(t["pnl"])

    year_delta = {}
    for yr in sorted(set(list(no_addon_by_yr.keys()) + list(addon_base_by_yr.keys()))):
        na_pnl = sum(no_addon_by_yr.get(yr, []))
        a_pnl  = sum(addon_base_by_yr.get(yr, []))
        year_delta[str(yr)] = {
            "no_addon_pnl": _ss(na_pnl),
            "addon_pnl":    _ss(a_pnl),
            "delta_pnl":    _ss(a_pnl - na_pnl),
            "regime":       regime_by_year.get(yr, {}).get("regime", "UNKNOWN"),
        }

    return {
        "regime_by_year":  regime_by_year,
        "addon_by_regime": regime_summary,
        "year_delta_pnl":  year_delta,
    }


# ======================================================================
# Phase7: Verdict
# ======================================================================

def phase7_verdict(p4: dict, p5: dict, p2: dict) -> dict:
    nev_pos      = p4.get("nev_positive", False)
    is_dcagr     = (p4.get("is_comparison") or {}).get("delta_cagr_pp")
    oos_dcagr    = (p4.get("oos_comparison") or {}).get("delta_cagr_pp")
    is_dcalmar   = (p4.get("is_comparison") or {}).get("delta_calmar")
    n_addon      = p2.get("n_addon_raw", 0)

    # ポートフォリオ経済価値判定 (Study67と同基準)
    # ① IS ΔCAGR > 1pp → meaningful
    # ② OOS ΔCAGR > 0  → out-of-sample positive
    # ③ NEV > 0
    # ④ ΔCalmar >= 0   → リスク調整後も改善

    cond_is_positive    = (is_dcagr  is not None and is_dcagr > 0)
    cond_is_meaningful  = (is_dcagr  is not None and is_dcagr > 1.0)
    cond_oos_positive   = (oos_dcagr is not None and oos_dcagr > 0)
    cond_nev_positive   = nev_pos
    cond_calmar_improve = (is_dcalmar is not None and is_dcalmar >= 0)

    n_pass = sum([cond_is_meaningful, cond_oos_positive, cond_nev_positive, cond_calmar_improve])

    if n_pass >= 4:
        verdict = "SHADOW_WORTHY"
        reason  = "全条件クリア → Shadow Add-on実装を推奨"
    elif n_pass >= 2 and cond_oos_positive:
        verdict = "CONDITIONAL"
        reason  = "IS/OOS正だがNEVまたはCalmar条件不足 → 追加監査推奨"
    elif cond_is_meaningful and not cond_oos_positive:
        verdict = "IS_ONLY"
        reason  = "IS正だがOOS負 → 過学習疑い / Shadow不推奨"
    else:
        verdict = "REJECT"
        reason  = "経済価値不十分 → Add-on拡大は不推奨"

    # 最適unitサイズ
    optimal_unit = (p5.get("optimal_unit") or "不明")

    return {
        "①_is_dcagr_pp":     is_dcagr,
        "②_oos_dcagr_pp":    oos_dcagr,
        "③_portfolio_nev_jpy": p4.get("portfolio_nev_jpy"),
        "④_is_dcalmar":      is_dcalmar,
        "⑤_conditions": {
            "is_meaningful_1pp": cond_is_meaningful,
            "oos_positive":      cond_oos_positive,
            "nev_positive":      cond_nev_positive,
            "calmar_improve":    cond_calmar_improve,
            "n_pass":            n_pass,
        },
        "⑥_optimal_unit":  optimal_unit,
        "⑦_s65_validation": {
            "s65_realistic_pp":   S65_REALISTIC_PP,
            "portfolio_dcagr_pp": is_dcagr,
            "s65_confirmed":      (is_dcagr is not None and is_dcagr >= S65_REALISTIC_PP * 0.80),
        },
        "final_verdict": verdict,
        "reason":         reason,
    }


# ======================================================================
# main
# ======================================================================

def main():
    print("Study70: Add-on Portfolio Economic Audit")
    print("=" * 60)

    print("\nデータ構築中...")
    ds = build_common_dataset(DATA_END)
    all_dates = ds["rsr_df"].index.sort_values()

    # --- 全トレードの基準データセット構築 (Study67と同一) ---
    print("  基準BT (UNIT_025 IS)...")
    sym_is  = get_active(ds, IS_START, IS_END)
    bt_ref_is  = _run_bt_scenario(ds, sym_is, IS_START, IS_END, 1, 0.25)
    is_dates   = all_dates[(all_dates >= IS_START)  & (all_dates <= IS_END)]
    tr_ref_is  = _extract_trades(bt_ref_is, is_dates, ds["rsr_df"])

    sym_oos = get_active(ds, OOS_START, DATA_END)
    bt_ref_oos = _run_bt_scenario(ds, sym_oos, OOS_START, DATA_END, 1, 0.25)
    oos_dates  = all_dates[(all_dates >= OOS_START) & (all_dates <= DATA_END)]
    tr_ref_oos = _extract_trades(bt_ref_oos, oos_dates, ds["rsr_df"])

    all_raw = tr_ref_is + tr_ref_oos
    df_all  = build_full_dataset(ds, all_raw)
    print(f"  全取引: {len(df_all)}件")

    # --- Phase0 ---
    print("\nPhase0: Integrity...")
    p0 = phase0_integrity(df_all)
    print(f"  n={p0['n_total']}({p0['n_match_291']}), BW={p0['n_bw']}({p0['n_bw_match_30']}), "
          f"RSR={p0['n_rsr']}({p0['n_rsr_match_248']})")

    # --- Phase1: 4シナリオ ---
    print("\nPhase1: 4シナリオBT...")
    scenarios = phase1_scenarios(ds, all_dates)

    # NO_ADDONの基準値確認
    na_is = scenarios["NO_ADDON"]["is_metrics"]
    u25_is = scenarios["UNIT_025"]["is_metrics"]
    print(f"\n  IS NO_ADDON: CAGR={na_is['cagr']}  n={na_is['n_trades']}")
    print(f"  IS UNIT_025: CAGR={u25_is['cagr']}  n={u25_is['n_trades']}")

    # --- Phase2: Add-on Attribution ---
    print("\nPhase2: Add-on Attribution...")
    p2 = phase2_addon_attribution(scenarios, df_all, ds)
    print(f"  addon trades raw: {p2['n_addon_raw']}, base: {p2['n_base_raw']}")
    print(f"  addon P&L total: ¥{p2['addon_pnl_total']:,.0f}" if p2['addon_pnl_total'] else "  addon P&L: N/A")
    print(f"  {p2['note_addon_identification']}")

    # --- Phase3: Opportunity Cost ---
    print("\nPhase3: Opportunity Cost Analysis...")
    p3 = phase3_opportunity_cost(scenarios, ds)
    print(f"  displaced entries: {p3['displaced_entries_n']}件")
    print(f"  opportunity cost: ¥{p3['opportunity_cost_jpy']:,.0f}" if p3['opportunity_cost_jpy'] else "  opp cost: ¥0")
    print(f"  redeploy delay: mean={p3['redeploy_delay_stats'].get('mean')}d")

    # --- Phase4: Portfolio NEV ---
    print("\nPhase4: Portfolio NEV...")
    p4 = phase4_portfolio_nev(p2, p3, scenarios)
    print(f"  Add-on Gain: ¥{p4['addon_gain_jpy']:,.0f}" if p4['addon_gain_jpy'] else "  addon gain: N/A")
    print(f"  Opportunity Cost: ¥{p4['opportunity_cost_jpy']:,.0f}" if p4['opportunity_cost_jpy'] else "  opp cost: N/A")
    print(f"  Portfolio NEV: ¥{p4['portfolio_nev_jpy']:,.0f}" if p4['portfolio_nev_jpy'] else "  NEV: N/A")
    print(f"  IS  ΔCAGR={p4['is_comparison']['delta_cagr_pp']}pp  ΔCalmar={p4['is_comparison']['delta_calmar']}  ΔMaxDD={p4['is_comparison']['delta_max_dd_pp']}pp")
    print(f"  OOS ΔCAGR={p4['oos_comparison']['delta_cagr_pp']}pp  ΔCalmar={p4['oos_comparison']['delta_calmar']}  ΔMaxDD={p4['oos_comparison']['delta_max_dd_pp']}pp")
    print(f"  Study65 reference: +{S65_REALISTIC_PP}pp | delta_vs_s65={p4['reference_comparison']['dcagr_vs_s65']}")

    # --- Phase5: Unit Comparison ---
    print("\nPhase5: Unit Size Comparison...")
    p5 = phase5_unit_comparison(scenarios)
    print(f"  NO_ADDON IS CAGR: {p5['no_addon_is_cagr']}")
    for row in p5["unit_comparison"]:
        print(f"  {row['label']}: IS_ΔCAGR={row['is_dcagr']}pp  OOS_ΔCAGR={row['oos_dcagr']}pp  "
              f"IS_ΔCalmar={row['is_dcalmar']}  IS_ΔMaxDD={row['is_ddd']}")
    print(f"  Optimal unit: {p5['optimal_unit']}")

    # --- Phase6: Regime ---
    print("\nPhase6: Regime Analysis...")
    p6 = phase6_regime_analysis(scenarios, ds)
    for reg, s in p6.get("addon_by_regime", {}).items():
        print(f"  {reg}: n={s['n']}, total_pnl={s['total_pnl']}, pos%={s['positive_pct']}")

    # --- Phase7: Verdict ---
    print("\nPhase7: Verdict...")
    p7 = phase7_verdict(p4, p5, p2)
    print(f"  ① IS  ΔCAGR: {p7['①_is_dcagr_pp']}pp")
    print(f"  ② OOS ΔCAGR: {p7['②_oos_dcagr_pp']}pp")
    print(f"  ③ Portfolio NEV: ¥{p7['③_portfolio_nev_jpy']:,.0f}" if p7['③_portfolio_nev_jpy'] else "  ③ NEV: N/A")
    print(f"  ④ IS ΔCalmar: {p7['④_is_dcalmar']}")
    print(f"  ⑤ Conditions: {p7['⑤_conditions']}")
    print(f"  ⑥ Optimal unit: {p7['⑥_optimal_unit']}")
    print(f"  ⑦ S65 confirmed: {p7['⑦_s65_validation']['s65_confirmed']}")
    print(f"  最終判定: {p7['final_verdict']} — {p7['reason']}")

    # --- 保存 ---
    # scenariosからtradeリストを省いてJSON化
    scenarios_summary = {}
    for label, sc in scenarios.items():
        scenarios_summary[label] = {
            "config":      sc["config"],
            "is_metrics":  sc["is_metrics"],
            "oos_metrics": sc["oos_metrics"],
        }

    output = {
        "study":  "Study70",
        "title":  "Add-on Portfolio Economic Audit",
        "date":   TODAY_STR,
        "params": {
            "capital":   CAPITAL,
            "max_pos":   MAX_POS,
            "s65_realistic_pp": S65_REALISTIC_PP,
            "s64_oracle_bw_pp": S64_ADDON_PP,
        },
        "phase0_integrity":         p0,
        "phase1_scenarios_summary": scenarios_summary,
        "phase2_addon_attribution": p2,
        "phase3_opportunity_cost":  p3,
        "phase4_portfolio_nev":     p4,
        "phase5_unit_comparison":   p5,
        "phase6_regime_analysis":   p6,
        "phase7_verdict":           p7,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n結果保存: {OUT_FILE.name}")
    print("======== Study70 COMPLETE ========")
    return output


if __name__ == "__main__":
    main()
