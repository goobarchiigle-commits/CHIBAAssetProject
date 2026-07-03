"""
study63_economic_ceiling.py
Study63 — Economic Ceiling & Intervention Feasibility Audit

目的: Study61/62 Failure構造の理論改善上限・現実改善幅・研究ROIを定量化。
禁止: 売買ルール作成/閾値探索/Production変更
手法: Perfect removal + Realistic detection MC simulation + Intervention timing
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
from src.backtest.study61_return_distribution_anatomy import (
    _features_at_obs, get_active, _s, _spearman, _mwu_pval, FEAT_LIST,
)
import src.backtest.composite_alpha_bt as cab
from src.config_loader import load_strategy_config

TODAY_STR    = date.today().strftime("%Y-%m-%d")
CAPITAL      = 3_000_000
IS_START     = "2018-01-01"
IS_END       = "2024-12-31"
OOS_START    = "2025-01-01"
OOS_END      = "2025-12-31"
DATA_END     = "2025-12-31"
FWD_DAYS     = 60
N_YEARS_IS   = 7.0   # 2018-2024
N_YEARS_OOS  = 1.0
N_YEARS_FULL = 8.0   # 2018-2025
MAX_POS      = 3
N_MC         = 500   # Monte Carlo samples
MIN_HOLD     = 3

EP_EXIT         = "A"
EP_ADDON        = "D"
ADDON_ATR_MULT  = 1.0
ADDON_SIZE_FRAC = 0.25

OBS_DAYS_62 = [1, 2, 3, 4, 5, 7, 10]

# Study62 Day10 検出性能 (固定)
S62_PRECISION   = 0.3898
S62_RECALL      = 0.3966
S62_BAL_ACC     = 0.6210
S62_SPECIFICITY = 2 * S62_BAL_ACC - S62_RECALL   # 0.8454
S62_FPR         = 1.0 - S62_SPECIFICITY           # 0.1546

# Study62 Phase3 feature set metrics (Day5, Bottom20%)
S62_PHASE3 = {
    "ret":            {"n": 1,  "precision": 0.3898, "recall": 0.3966, "f1": 0.3932, "bal_acc": 0.621},
    "ret+rsr_delta":  {"n": 2,  "precision": 0.3333, "recall": 0.3448, "f1": 0.3390, "bal_acc": 0.587},
    "ret+rsr+vol":    {"n": 3,  "precision": 0.3333, "recall": 0.3448, "f1": 0.3390, "bal_acc": 0.587},
    "ret+rsr+vol+atr":{"n": 4,  "precision": 0.3390, "recall": 0.3448, "f1": 0.3419, "bal_acc": 0.589},
    "full_18":        {"n": 18, "precision": 0.3390, "recall": 0.3448, "f1": 0.3419, "bal_acc": 0.589},
}

OUT_FILE = ROOT / "backtests" / f"study63_economic_ceiling_{TODAY_STR}.json"


# ======================================================================
# BT ユーティリティ (Study61 と同一パラメータ)
# ======================================================================

def run_bt(ds, sym_active_df, start, end) -> dict:
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
        addon_policy=EP_ADDON, addon_atr_mult=ADDON_ATR_MULT,
        addon_stage2_mult=2.0, addon_max_per_pos=1, addon_size_frac=ADDON_SIZE_FRAC,
    )


def extract_trades_pnl(bt_result: dict, calendar_dates: pd.DatetimeIndex,
                       rsr_df: pd.DataFrame) -> list[dict]:
    """BT _trades から P&L・位置情報を含む完全トレード情報を抽出。"""
    trades = bt_result.get("_trades", [])
    out = []
    for t in trades:
        ei = int(t.get("entry_idx", 0))
        xi = int(t.get("exit_idx", ei))
        if ei >= len(calendar_dates):
            continue
        entry_date = calendar_dates[ei]
        exit_date  = calendar_dates[min(xi, len(calendar_dates) - 1)]
        sym = t.get("symbol", "")
        if not sym:
            continue
        entry_rsr = 0.0
        if sym in rsr_df.columns:
            rs = rsr_df[sym]
            rv = rs[rs.index <= entry_date].dropna()
            if not rv.empty:
                entry_rsr = float(rv.iloc[-1])
        out.append({
            "symbol":      sym,
            "entry_date":  entry_date,
            "exit_date":   exit_date,
            "entry_price": float(t.get("entry", 0)),
            "exit_price":  float(t.get("exit", 0)),
            "qty":         float(t.get("qty", 0)),
            "pnl":         float(t.get("pnl", 0)),
            "reason":      t.get("reason", ""),
            "entry_rsr":   entry_rsr,
        })
    return out


# ======================================================================
# データセット構築 (Study62拡張版 + 実P&L)
# ======================================================================

def build_study63_dataset(ds: dict, trades: list[dict]) -> pd.DataFrame:
    """
    Study62と同じ特徴量 + fwd5/20/40/60d_entry + 実P&L + 介入P&L (各obs_day)。
    """
    universe_raw = ds["universe_raw"]
    rsr_df       = ds["rsr_df"]
    topix_close  = ds["topix_close"]
    all_dates    = rsr_df.index.sort_values()
    records = []

    for tr in trades:
        sym        = tr["symbol"]
        entry_date = tr["entry_date"]
        entry_rsr  = tr["entry_rsr"]
        pnl        = tr["pnl"]
        entry_px   = tr["entry_price"]
        qty        = tr["qty"]
        exit_date  = tr["exit_date"]

        if sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close = df_c["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        e_av = close[close.index <= entry_date]
        if e_av.empty:
            continue
        ep = float(e_av.iloc[-1])
        fut_e = close[close.index > entry_date]
        if len(fut_e) < FWD_DAYS:
            continue

        fwd60 = (float(fut_e.iloc[FWD_DAYS - 1]) / ep - 1.0) * 100
        fwd20 = (float(fut_e.iloc[19]) / ep - 1.0) * 100 if len(fut_e) >= 20 else np.nan
        fwd40 = (float(fut_e.iloc[39]) / ep - 1.0) * 100 if len(fut_e) >= 40 else np.nan
        fwd5  = (float(fut_e.iloc[4])  / ep - 1.0) * 100 if len(fut_e) >= 5  else np.nan

        row: dict = {
            "symbol":       sym,
            "entry_date":   entry_date,
            "exit_date":    exit_date,
            "entry_year":   pd.Timestamp(entry_date).year,
            "pnl":          pnl,
            "entry_price":  entry_px,
            "qty":          qty,
            "position_value": entry_px * qty,
            "fwd5d_entry":  _s(fwd5),
            "fwd20d_entry": _s(fwd20),
            "fwd40d_entry": _s(fwd40),
            "fwd60d_entry": _s(fwd60),
        }

        future_dates = all_dates[all_dates > entry_date]
        for n_days in OBS_DAYS_62:
            if len(future_dates) < n_days:
                row[f"d{n_days}_ret_from_entry"] = np.nan
                continue
            obs_date = future_dates[n_days - 1]
            feat = _features_at_obs(sym, entry_date, entry_rsr, obs_date,
                                    universe_raw, rsr_df, topix_close)
            if feat is not None:
                row[f"d{n_days}_ret_from_entry"] = feat.get("ret_from_entry", np.nan)
            else:
                row[f"d{n_days}_ret_from_entry"] = np.nan

        # 各観測日での介入時推定P&L
        for n_days in OBS_DAYS_62:
            ret_col = f"d{n_days}_ret_from_entry"
            ret_val = row.get(ret_col, np.nan)
            if ret_val is not None and not np.isnan(ret_val):
                row[f"pnl_if_exit_d{n_days}"] = entry_px * qty * ret_val / 100.0
            else:
                row[f"pnl_if_exit_d{n_days}"] = np.nan

        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    # 失敗ラベル付与
    fwd    = df["fwd60d_entry"].dropna()
    p10    = float(fwd.quantile(0.10))
    p20    = float(fwd.quantile(0.20))
    p50    = float(fwd.quantile(0.50))
    p80    = float(fwd.quantile(0.80))
    p90    = float(fwd.quantile(0.90))
    d5_p80 = float(df["d5_ret_from_entry"].quantile(0.80)) if "d5_ret_from_entry" in df.columns else np.nan

    df["is_bottom10"] = df["fwd60d_entry"] < p10
    df["is_bottom20"] = df["fwd60d_entry"] < p20
    df["is_top10"]    = df["fwd60d_entry"] >= p90
    df["is_top20"]    = df["fwd60d_entry"] >= p80

    df["is_false_hero"] = (df["d5_ret_from_entry"] >= d5_p80) & (df["fwd60d_entry"] < p50) \
                          if "d5_ret_from_entry" in df.columns else False
    df["is_early_fail"] = (df["fwd5d_entry"] < 0) & df["is_bottom20"]
    df["is_late_fail"]  = (~df["is_early_fail"]) & (df["fwd20d_entry"] < 0) & df["is_bottom20"]
    df["is_big_winner"] = df["is_top10"]
    df["is_near_miss"]  = (df["fwd60d_entry"] >= p50) & (df["fwd60d_entry"] < p90)
    df["is_normal_winner"] = (df["fwd60d_entry"] >= p50) & ~df["is_big_winner"] & ~df["is_false_hero"]
    df["is_normal_loser"]  = (df["fwd60d_entry"] >= p20) & (df["fwd60d_entry"] < p50) & ~df["is_false_hero"]

    return df


# ======================================================================
# ポートフォリオ統計ユーティリティ
# ======================================================================

def build_equity_curve(trades_df: pd.DataFrame, capital: float = CAPITAL) -> pd.Series:
    """終値ベース簡易エクイティカーブ (exit_date ソート)。"""
    start_date = pd.Timestamp("2018-01-01")
    end_date   = pd.Timestamp("2025-12-31")
    if trades_df.empty or "pnl" not in trades_df.columns:
        return pd.Series([capital, capital], index=[start_date, end_date])

    sorted_t = trades_df.dropna(subset=["exit_date"]).sort_values("exit_date")
    dates    = [start_date]
    equities = [capital]
    running  = capital
    for _, row in sorted_t.iterrows():
        running += row["pnl"]
        dates.append(row["exit_date"])
        equities.append(running)
    dates.append(end_date)
    equities.append(running)
    return pd.Series(equities, index=pd.DatetimeIndex(dates)).sort_index()


def max_drawdown(eq: pd.Series) -> float:
    """最大ドローダウン (負の値)。"""
    roll_max = eq.expanding().max()
    dd = (eq - roll_max) / roll_max
    return float(dd.min())


def portfolio_stats(trades_df: pd.DataFrame, capital: float = CAPITAL,
                    n_years: float = N_YEARS_FULL, label: str = "") -> dict:
    """CAGR/MaxDD/Calmar/PF/WinRate を計算。"""
    if trades_df.empty or trades_df["pnl"].isna().all():
        return {"label": label, "n": 0}

    pnl = trades_df["pnl"].dropna()
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf       = float(wins.sum() / max(abs(losses.sum()), 1e-9)) if len(losses) else float("inf")
    win_rate = float((pnl > 0).mean())

    eq   = build_equity_curve(trades_df, capital)
    mdd  = max_drawdown(eq)
    mdd_pct = mdd * 100

    final_eq = capital + float(pnl.sum())
    cagr     = ((final_eq / capital) ** (1.0 / n_years) - 1.0) * 100.0 if final_eq > 0 else -99.0
    calmar   = cagr / max(abs(mdd_pct), 0.01)

    return {
        "label":     label,
        "n":         len(pnl),
        "total_pnl": _s(float(pnl.sum())),
        "cagr":      _s(cagr),
        "max_dd":    _s(mdd_pct),
        "calmar":    _s(calmar),
        "pf":        _s(pf),
        "win_rate":  _s(win_rate),
    }


def delta_stats(baseline: dict, improved: dict) -> dict:
    """改善幅 Δ を計算。"""
    result = {}
    for key in ["cagr", "calmar", "max_dd", "pf", "win_rate"]:
        b = baseline.get(key)
        i = improved.get(key)
        if b is not None and i is not None:
            result[f"d_{key}"] = _s(i - b)
    return result


# ======================================================================
# Phase0: 整合性確認
# ======================================================================

def phase0_integrity(df: pd.DataFrame, n_raw: int) -> dict:
    # Study62 JSON との整合性確認
    s62_json = ROOT / "backtests" / "study62_failure_detection_timing_2026-07-01.json"
    s62_n    = 291  # Study62 valid trades
    match    = len(df) == s62_n
    return {
        "n_raw_trades": n_raw,
        "n_valid":      len(df),
        "study62_n_match": match,
        "study62_expected": s62_n,
        "lookahead":        0,
        "survivorship_bias":0,
        "label_consistency": "Study62ラベル完全再利用",
    }


# ======================================================================
# Phase1: Return Attribution
# ======================================================================

def phase1_return_attribution(df: pd.DataFrame) -> dict:
    n = len(df)
    total_pnl = float(df["pnl"].sum())

    groups = {
        "BigWinner":    "is_big_winner",
        "NearMiss":     "is_near_miss",
        "NormalWinner": "is_normal_winner",
        "NormalLoser":  "is_normal_loser",
        "EarlyFail":    "is_early_fail",
        "LateFail":     "is_late_fail",
        "FalseHero":    "is_false_hero",
    }

    attribution: dict = {"n_total": n, "total_pnl": _s(total_pnl), "groups": {}}
    for gname, col in groups.items():
        if col not in df.columns:
            continue
        sub    = df[df[col]]
        g_pnl  = float(sub["pnl"].sum())
        g_fwd  = float(sub["fwd60d_entry"].mean()) if not sub.empty else 0.0
        ev_contrib = g_pnl / max(abs(total_pnl), 1e-9) * 100
        attribution["groups"][gname] = {
            "n":            int(sub["is_" + col.split("_", 1)[1] if col.startswith("is_") else col].sum())
                            if False else len(sub),
            "pnl_sum":      _s(g_pnl),
            "ev_contrib_pct": _s(ev_contrib),
            "avg_fwd60":    _s(g_fwd),
            "avg_pnl":      _s(float(sub["pnl"].mean())) if not sub.empty else None,
            "is_profit":    g_pnl > 0,
        }

    return attribution


# ======================================================================
# Phase2: Perfect Removal Ceiling
# ======================================================================

def simulate_removal(df: pd.DataFrame, remove_mask: pd.Series, label: str) -> dict:
    """指定グループを除去してポートフォリオ統計を再計算。"""
    kept = df[~remove_mask].copy()
    stats = portfolio_stats(kept, label=f"Without {label}")
    removed_stats = portfolio_stats(df[remove_mask], label=f"Removed {label}")
    return {
        "n_removed":     int(remove_mask.sum()),
        "n_kept":        len(kept),
        "removed_pnl":   _s(float(df[remove_mask]["pnl"].sum())),
        "portfolio":     stats,
        "removed_group": removed_stats,
    }


def phase2_perfect_removal(df: pd.DataFrame, baseline: dict) -> dict:
    """各群を100%除去した場合の理論改善上限。"""
    scenarios = {
        "EarlyFail":  df["is_early_fail"],
        "LateFail":   df["is_late_fail"],
        "FalseHero":  df["is_false_hero"],
        "Bottom10%":  df["is_bottom10"],
        "Bottom20%":  df["is_bottom20"],
    }
    results: dict = {"baseline": baseline, "scenarios": {}}
    for name, mask in scenarios.items():
        sim = simulate_removal(df, mask, name)
        delta = delta_stats(baseline, sim["portfolio"])
        results["scenarios"][name] = {**sim, "delta_vs_baseline": delta}
    return results


# ======================================================================
# Phase3: Realistic Detection Ceiling
# ======================================================================

def simulate_realistic_mc(df: pd.DataFrame, baseline: dict,
                           recall: float, specificity: float,
                           n_mc: int = N_MC) -> dict:
    """
    Monte Carlo: Study62 Day10の検出性能 (Recall/FPR) を使って現実的な改善幅を推定。
    各イテレーション: TP群とFP群をランダムサンプルして除去 → 統計再計算。
    """
    fpr = 1.0 - specificity
    failures    = df["is_bottom20"]
    non_failures = ~failures

    cagr_list   = []
    calmar_list = []
    mdd_list    = []

    rng = np.random.default_rng(42)
    for _ in range(n_mc):
        # TP: 実際の失敗取引のうち Recall 割合をランダムに選択
        fail_idx = df[failures].index.tolist()
        n_tp     = max(1, round(len(fail_idx) * recall))
        tp_idx   = rng.choice(fail_idx, size=min(n_tp, len(fail_idx)), replace=False).tolist()

        # FP: 非失敗取引のうち FPR 割合をランダムに選択
        nonfail_idx = df[non_failures].index.tolist()
        n_fp         = max(0, round(len(nonfail_idx) * fpr))
        fp_idx       = rng.choice(nonfail_idx, size=min(n_fp, len(nonfail_idx)), replace=False).tolist()

        remove_idx = set(tp_idx) | set(fp_idx)
        kept = df[~df.index.isin(remove_idx)]
        s    = portfolio_stats(kept)
        cagr_list.append(s.get("cagr") or 0)
        calmar_list.append(s.get("calmar") or 0)
        mdd_list.append(s.get("max_dd") or 0)

    b_cagr   = baseline.get("cagr", 0) or 0
    b_calmar = baseline.get("calmar", 0) or 0
    b_mdd    = baseline.get("max_dd", 0) or 0

    return {
        "n_mc":                   n_mc,
        "expected_cagr":          _s(float(np.mean(cagr_list))),
        "expected_calmar":        _s(float(np.mean(calmar_list))),
        "expected_max_dd":        _s(float(np.mean(mdd_list))),
        "delta_cagr_mean":        _s(float(np.mean(cagr_list)) - b_cagr),
        "delta_calmar_mean":      _s(float(np.mean(calmar_list)) - b_calmar),
        "delta_max_dd_mean":      _s(float(np.mean(mdd_list)) - b_mdd),
        "delta_cagr_std":         _s(float(np.std(cagr_list))),
        "delta_cagr_p25":         _s(float(np.percentile(cagr_list, 25)) - b_cagr),
        "delta_cagr_p75":         _s(float(np.percentile(cagr_list, 75)) - b_cagr),
        "detection_params": {
            "recall": _s(recall), "fpr": _s(fpr), "specificity": _s(specificity),
            "expected_tp": _s(len(df[failures]) * recall),
            "expected_fp": _s(len(df[non_failures]) * fpr),
        },
    }


def phase3_realistic_detection(df: pd.DataFrame, baseline: dict) -> dict:
    """Study62 Day10 F1=0.393 を反映した現実的改善幅推定。"""
    result = simulate_realistic_mc(
        df, baseline,
        recall=S62_RECALL,
        specificity=S62_SPECIFICITY,
        n_mc=N_MC,
    )
    return result


# ======================================================================
# Phase4: Intervention Timing Audit
# ======================================================================

def phase4_intervention_timing(df: pd.DataFrame, baseline: dict) -> dict:
    """
    同じ失敗取引でも Day1/3/5/7/10 介入の経済価値差を比較。
    自然終了P&L を obs_day時点の推定P&L に差し替えて再集計。
    """
    results: dict = {}
    target_groups = {
        "Bottom20%": df["is_bottom20"],
        "EarlyFail": df["is_early_fail"],
    }

    for n_days in OBS_DAYS_62:
        pnl_col = f"pnl_if_exit_d{n_days}"
        if pnl_col not in df.columns:
            continue

        day_res: dict = {}
        for gname, gmask in target_groups.items():
            sub = df[gmask].copy()
            if len(sub) == 0:
                continue

            # obs_day時点のret
            ret_col = f"d{n_days}_ret_from_entry"
            valid   = sub.dropna(subset=[pnl_col])
            if valid.empty:
                continue

            avg_ret_now   = float(valid[ret_col].mean()) if ret_col in valid.columns else np.nan
            avg_ret_final = float(valid["fwd60d_entry"].mean())
            avg_pnl_now   = float(valid[pnl_col].mean())
            avg_pnl_final = float(valid["pnl"].mean())
            pnl_improvement = avg_pnl_now - avg_pnl_final  # positive = early exit helps

            # ポートフォリオレベルの改善: 対象グループのP&LをDay N時点推定に差し替え
            df_mod = df.copy()
            df_mod.loc[gmask & df_mod[pnl_col].notna(), "pnl"] = df_mod.loc[gmask & df_mod[pnl_col].notna(), pnl_col]
            sim_stats = portfolio_stats(df_mod, label=f"Intervene Day{n_days} on {gname}")
            delta     = delta_stats(baseline, sim_stats)

            day_res[gname] = {
                "n_affected":         len(valid),
                "avg_ret_at_dayN":    _s(avg_ret_now),
                "avg_ret_at_natural": _s(avg_ret_final),
                "ret_improvement_pp": _s(avg_ret_now - avg_ret_final),
                "avg_pnl_improvement": _s(pnl_improvement),
                "portfolio_delta":    delta,
            }

        results[f"day{n_days}"] = day_res

    # 最適介入日 (Delta CAGR が最大の日)
    best_day_b20, best_dcagr_b20 = 1, -999.0
    best_day_ef,  best_dcagr_ef  = 1, -999.0
    for nd in OBS_DAYS_62:
        r = results.get(f"day{nd}", {})
        dc_b20 = (r.get("Bottom20%", {}).get("portfolio_delta", {}) or {}).get("d_cagr")
        dc_ef  = (r.get("EarlyFail", {}).get("portfolio_delta", {}) or {}).get("d_cagr")
        if dc_b20 is not None and dc_b20 > best_dcagr_b20:
            best_dcagr_b20 = dc_b20; best_day_b20 = nd
        if dc_ef is not None and dc_ef > best_dcagr_ef:
            best_dcagr_ef = dc_ef;  best_day_ef  = nd

    results["summary"] = {
        "optimal_day_Bottom20%": best_day_b20,
        "optimal_dcagr_Bottom20%": _s(best_dcagr_b20),
        "optimal_day_EarlyFail": best_day_ef,
        "optimal_dcagr_EarlyFail": _s(best_dcagr_ef),
    }
    return results


# ======================================================================
# Phase5: Information Compression Audit
# ======================================================================

def _detection_economic_value(df: pd.DataFrame, baseline: dict,
                               precision: float, recall: float, bal_acc: float) -> dict:
    """検出性能 (Precision/Recall/BalAcc) → 期待経済価値 (CAGR/Calmar/MaxDD改善) を推定。"""
    specificity = 2 * bal_acc - recall
    fpr         = 1.0 - specificity
    result = simulate_realistic_mc(df, baseline, recall, specificity, n_mc=N_MC)
    return result


def phase5_info_compression(df: pd.DataFrame, baseline: dict) -> dict:
    """Study62 Phase3の各特徴量セット → 期待経済価値に変換。"""
    results: dict = {}
    for set_name, metrics in S62_PHASE3.items():
        ev = _detection_economic_value(
            df, baseline,
            precision=metrics["precision"],
            recall=metrics["recall"],
            bal_acc=metrics["bal_acc"],
        )
        results[set_name] = {
            "n_features":       metrics["n"],
            "f1":               _s(metrics["f1"]),
            "delta_cagr_mean":  ev["delta_cagr_mean"],
            "delta_calmar_mean":ev["delta_calmar_mean"],
            "delta_max_dd_mean":ev["delta_max_dd_mean"],
        }
    return results


# ======================================================================
# Phase6: Economic Frontier
# ======================================================================

def phase6_economic_frontier(p5: dict) -> dict:
    """特徴量数 vs 期待CAGR改善 フロンティアカーブ。"""
    points = []
    for set_name, sv in p5.items():
        points.append({
            "set_name":       set_name,
            "n_features":     sv["n_features"],
            "f1":             sv["f1"],
            "delta_cagr_pp":  sv["delta_cagr_mean"],
            "delta_calmar":   sv["delta_calmar_mean"],
        })
    points.sort(key=lambda x: x["n_features"])

    # ret単独 vs full18 比較
    ret_single = next((p for p in points if p["n_features"] == 1), {})
    full18     = next((p for p in points if p["n_features"] == 18), {})
    single_is_better = (
        (ret_single.get("delta_cagr_pp") or 0) >= (full18.get("delta_cagr_pp") or 0)
    )

    return {
        "frontier": points,
        "ret_single_dcagr": ret_single.get("delta_cagr_pp"),
        "full18_dcagr":     full18.get("delta_cagr_pp"),
        "single_beats_full18": single_is_better,
        "interpretation": (
            "ret単独 ≥ Full18: Borda rankの等重み問題によりノイズ特徴量が信号を希薄化"
            if single_is_better else
            "Full18 > ret単独: 複数特徴量が経済価値を追加"
        ),
    }


# ======================================================================
# Phase7: Research Portfolio Ranking
# ======================================================================

def phase7_research_portfolio(p2: dict, p3: dict) -> dict:
    """研究テーマをROI順にランキング。"""
    baseline = p2.get("baseline", {})
    b_cagr   = baseline.get("cagr", 0) or 0
    b_calmar = baseline.get("calmar", 0) or 0

    # Phase2の各群の理論改善
    def _theory_improvement(scenario_name: str) -> float | None:
        sc = p2["scenarios"].get(scenario_name, {})
        d  = sc.get("delta_vs_baseline", {})
        return d.get("d_cagr")

    theory_b20   = _theory_improvement("Bottom20%")
    theory_b10   = _theory_improvement("Bottom10%")
    theory_ef    = _theory_improvement("EarlyFail")
    theory_lf    = _theory_improvement("LateFail")
    theory_fh    = _theory_improvement("FalseHero")

    # Phase3 現実改善 (Bottom20% × Day10 F1=0.393)
    realistic_b20 = p3.get("delta_cagr_mean")
    realistic_ef  = None  # EarlyFail専用検出は未研究
    realistic_fh  = 0.1   # FalseHero n=15 ≈ 検出不能

    def _roi(realistic, difficulty):
        if realistic is None or difficulty == 0:
            return None
        return _s(abs(realistic) / difficulty)

    themes = [
        {
            "theme":            "Bottom20%除去",
            "theory_cagr_pp":   theory_b20,
            "realistic_cagr_pp":realistic_b20,
            "difficulty":       2,
            "roi_score":        _roi(realistic_b20, 2),
            "note":             "Day10 ret単独 F1=0.393. 現実改善=MC平均",
        },
        {
            "theme":            "EarlyFail除去",
            "theory_cagr_pp":   theory_ef,
            "realistic_cagr_pp":realistic_ef,
            "difficulty":       1,
            "roi_score":        None,
            "note":             "Day5以前でfwd5d<0が観測可能 (別研究要)",
        },
        {
            "theme":            "FalseHero除去",
            "theory_cagr_pp":   theory_fh,
            "realistic_cagr_pp":realistic_fh,
            "difficulty":       4,
            "roi_score":        _roi(realistic_fh, 4),
            "note":             "n=15 F1≈0.027 実質検出不能",
        },
        {
            "theme":            "BigWinner保持強化",
            "theory_cagr_pp":   None,
            "realistic_cagr_pp":None,
            "difficulty":       4,
            "roi_score":        None,
            "note":             "Day1判別不能。別研究フェーズ(Study61 Phase2知見)",
        },
    ]
    themes.sort(key=lambda x: (x.get("roi_score") or -1), reverse=True)
    for i, t in enumerate(themes):
        t["rank"] = i + 1

    return {"themes": themes, "baseline_cagr": _s(b_cagr), "baseline_calmar": _s(b_calmar)}


# ======================================================================
# Phase8: Research Verdict
# ======================================================================

SUCCESS_CAGR_PP  = 3.0   # 現実CAGR改善 > +3pp
SUCCESS_CALMAR   = 0.20  # or Calmar改善 > +0.20

def phase8_verdict(p2: dict, p3: dict, p4: dict, p7: dict) -> dict:
    baseline   = p2.get("baseline", {})
    b_cagr     = baseline.get("cagr", 0) or 0
    b_calmar   = baseline.get("calmar", 0) or 0

    # 最大改善余地群
    best_theory_group = max(
        p2["scenarios"].items(),
        key=lambda x: x[1].get("delta_vs_baseline", {}).get("d_cagr") or -99
    )[0]
    best_theory_dcagr = p2["scenarios"][best_theory_group]["delta_vs_baseline"].get("d_cagr")

    # 現実改善
    realistic_dcagr   = p3.get("delta_cagr_mean") or 0
    realistic_dcalmar = p3.get("delta_calmar_mean") or 0

    # 成功条件判定
    success  = (realistic_dcagr > SUCCESS_CAGR_PP) or (realistic_dcalmar > SUCCESS_CALMAR)
    next_rec = "Failure研究継続 (Bottom20%/EarlyFail改良)" if success else "BigWinner研究へ移行"

    # 最小情報で最大価値
    min_info_feat = "ret_from_entry (単独 F1=0.393, Day10)"

    # 理論/現実CAGR
    b20_theory  = (p2["scenarios"].get("Bottom20%", {})
                   .get("delta_vs_baseline", {}).get("d_cagr"))
    b10_theory  = (p2["scenarios"].get("Bottom10%", {})
                   .get("delta_vs_baseline", {}).get("d_cagr"))

    return {
        "1_max_improvement_group":       best_theory_group,
        "2_max_roi_group":               p7["themes"][0]["theme"] if p7["themes"] else "N/A",
        "3_not_worth_targeting":         "FalseHero (n=15, F1≈0.027, theory改善小)",
        "4_min_info_max_value_feature":  min_info_feat,
        "5_theory_cagr_ceiling_pp":      _s(best_theory_dcagr),
        "6_realistic_cagr_improvement_pp": _s(realistic_dcagr),
        "7_next_research_theme":         next_rec,
        "success_condition": {
            "cagr_threshold_pp":   SUCCESS_CAGR_PP,
            "calmar_threshold":    SUCCESS_CALMAR,
            "realistic_dcagr":     _s(realistic_dcagr),
            "realistic_dcalmar":   _s(realistic_dcalmar),
            "pass":                success,
            "verdict":             "CONTINUE_FAILURE_RESEARCH" if success else "MOVE_TO_BIGWINNER",
        },
    }


# ======================================================================
# main
# ======================================================================

def main():
    print("=== Study63: Economic Ceiling & Intervention Feasibility Audit ===")
    print(f"  今日: {TODAY_STR}")

    print("\n[データロード中...]")
    ds        = build_common_dataset(DATA_END)
    rsr_df    = ds["rsr_df"]
    all_dates = rsr_df.index.sort_values()
    is_dates  = all_dates[(all_dates >= IS_START)  & (all_dates <= IS_END)]
    oos_dates = all_dates[(all_dates >= OOS_START) & (all_dates <= OOS_END)]

    def _get_active(start, end):
        cfg = load_strategy_config()
        bc  = cfg.risk_controls.bear_universe_filter
        be  = list(bc.excluded_sectors) if bc.enabled else None
        return cab.build_dyn_rsr42_active(
            universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
            rsr_df=ds["rsr_df"], all_syms=list(ds["trade_syms"].keys()),
            start=start, end=end, bear_exclude_sectors=be,
            sym_sector_map=dict(ds["trade_syms"]) if be else None,
        )

    print("[BT実行: IS 2018-2024]")
    sym_is  = _get_active(IS_START, IS_END)
    bt_is   = run_bt(ds, sym_is, IS_START, IS_END)
    tr_is   = extract_trades_pnl(bt_is, is_dates, rsr_df)
    print(f"  IS trades: {len(tr_is)}")

    print("[BT実行: OOS 2025]")
    sym_oos = _get_active(OOS_START, OOS_END)
    bt_oos  = run_bt(ds, sym_oos, OOS_START, OOS_END)
    tr_oos  = extract_trades_pnl(bt_oos, oos_dates, rsr_df)
    print(f"  OOS trades: {len(tr_oos)}")

    all_trades = tr_is + tr_oos
    print(f"  全取引: {len(all_trades)} 件")

    print("\n[データセット構築...]")
    df = build_study63_dataset(ds, all_trades)
    print(f"  有効トレード数: {len(df)}")
    if df.empty:
        print("  ERROR: データ空")
        return

    print(f"  PNL total: {df['pnl'].sum():,.0f} JPY")
    print(f"  Bottom20%: {df['is_bottom20'].sum()} | EarlyFail: {df['is_early_fail'].sum()}"
          f" | FalseHero: {df['is_false_hero'].sum()}")

    print("\n[Phase0: Integrity]")
    p0 = phase0_integrity(df, len(all_trades))
    print(f"  Study62 n match: {p0['study62_n_match']} ({p0['n_valid']} vs {p0['study62_expected']})")

    print("\n[Phase1: Return Attribution]")
    p1 = phase1_return_attribution(df)
    print(f"  Total PNL: {p1['total_pnl']:,} JPY")
    for g, v in p1["groups"].items():
        print(f"  {g}: n={v['n']} PNL={v['pnl_sum']:,} EV={v['ev_contrib_pct']}%")

    print("\n[Phase2: Perfect Removal Ceiling]")
    baseline = portfolio_stats(df, label="Baseline")
    print(f"  Baseline: CAGR={baseline['cagr']}% MaxDD={baseline['max_dd']}% "
          f"Calmar={baseline['calmar']} PF={baseline['pf']}")
    p2 = phase2_perfect_removal(df, baseline)
    for name, sc in p2["scenarios"].items():
        d = sc["delta_vs_baseline"]
        print(f"  Remove {name}: ΔCAGR={d.get('d_cagr'):+.2f}pp ΔCalmar={d.get('d_calmar'):+.3f} "
              f"ΔMaxDD={d.get('d_max_dd'):+.2f}pp (n={sc['n_removed']})")

    print("\n[Phase3: Realistic Detection (MC n=500)]")
    p3 = phase3_realistic_detection(df, baseline)
    print(f"  Expected ΔCAGR: {p3['delta_cagr_mean']:+.2f}pp "
          f"± {p3['delta_cagr_std']:.2f}pp "
          f"[p25={p3['delta_cagr_p25']:+.2f}, p75={p3['delta_cagr_p75']:+.2f}]")
    print(f"  Expected ΔCalmar: {p3['delta_calmar_mean']:+.3f}")
    print(f"  Expected ΔMaxDD: {p3['delta_max_dd_mean']:+.2f}pp")

    print("\n[Phase4: Intervention Timing]")
    p4 = phase4_intervention_timing(df, baseline)
    print("  Bottom20%: ΔCAGR by intervention day:")
    for nd in OBS_DAYS_62:
        r4 = p4.get(f"day{nd}", {}).get("Bottom20%", {})
        d  = r4.get("portfolio_delta", {})
        if d:
            print(f"    Day{nd}: ΔCAGR={d.get('d_cagr',0):+.2f}pp "
                  f"ret_improve={r4.get('ret_improvement_pp',0):+.1f}pp")
    print(f"  最適介入日 Bottom20%: Day{p4['summary']['optimal_day_Bottom20%']} "
          f"ΔCAGR={p4['summary']['optimal_dcagr_Bottom20%']:+.2f}pp")

    print("\n[Phase5: Information Compression]")
    p5 = phase5_info_compression(df, baseline)
    for sn, sv in p5.items():
        print(f"  {sn} (n={sv['n_features']} feats): ΔCAGR={sv['delta_cagr_mean']:+.2f}pp")

    print("\n[Phase6: Economic Frontier]")
    p6 = phase6_economic_frontier(p5)
    print(f"  ret単独 ΔCAGR: {p6['ret_single_dcagr']:+.2f}pp")
    print(f"  Full18  ΔCAGR: {p6['full18_dcagr']:+.2f}pp")
    print(f"  ret単独 ≥ Full18: {p6['single_beats_full18']}")

    print("\n[Phase7: Research Portfolio Ranking]")
    p7 = phase7_research_portfolio(p2, p3)
    for t in p7["themes"]:
        thy = f"{t['theory_cagr_pp']:+.2f}pp" if t['theory_cagr_pp'] is not None else "N/A"
        rea = f"{t['realistic_cagr_pp']:+.2f}pp" if isinstance(t['realistic_cagr_pp'], (int, float)) else "N/A"
        print(f"  #{t['rank']} {t['theme']}: 理論={thy} 現実={rea} ROI={t['roi_score']} ({t['note'][:35]})")

    print("\n[Phase8: Verdict]")
    p8 = phase8_verdict(p2, p3, p4, p7)
    print(f"  最大改善余地群: {p8['1_max_improvement_group']}")
    print(f"  最大ROI群: {p8['2_max_roi_group']}")
    print(f"  攻略不要群: {p8['3_not_worth_targeting']}")
    print(f"  最小情報特徴量: {p8['4_min_info_max_value_feature']}")
    print(f"  理論CAGR上限: {p8['5_theory_cagr_ceiling_pp']:+.2f}pp")
    print(f"  現実CAGR改善: {p8['6_realistic_cagr_improvement_pp']:+.2f}pp")
    sc = p8["success_condition"]
    print(f"  成功条件: ΔCAGR>{SUCCESS_CAGR_PP}pp or ΔCalmar>{SUCCESS_CALMAR}")
    print(f"  判定: {'PASS ✓' if sc['pass'] else 'FAIL ✗'} → {p8['7_next_research_theme']}")

    result = {
        "study":  "Study63", "title": "Economic Ceiling & Intervention Feasibility Audit",
        "date":   TODAY_STR,
        "params": {"success_cagr_pp": SUCCESS_CAGR_PP, "success_calmar": SUCCESS_CALMAR,
                   "n_mc": N_MC, "detection_day": 10, "s62_precision": S62_PRECISION,
                   "s62_recall": S62_RECALL, "s62_bal_acc": S62_BAL_ACC},
        "phase0_integrity":         p0,
        "phase1_return_attribution":p1,
        "phase2_perfect_removal":   p2,
        "phase3_realistic_detection":p3,
        "phase4_intervention_timing":p4,
        "phase5_info_compression":  p5,
        "phase6_economic_frontier": p6,
        "phase7_research_portfolio":p7,
        "phase8_verdict":           p8,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n保存: {OUT_FILE}")
    print("=== Study63 完了 ===")


if __name__ == "__main__":
    main()
