"""
study69_bigwinner_protection_stability_audit.py
Study69 — BigWinner Protection Stability Audit

目的: Study68の情報価値(Oracle比35.3%, ΔCAGR+2.25pp)が
     期間依存でなくWalk Forwardでも再現するか監査。
主評価: 経済価値の再現性 (特徴量識別性能ではなく)
禁止: 売買ルール / 閾値最適化 / Production変更 / Lookahead

Phase0: Integrity
Phase1: Walk Forward (6Fold: Train2年→Test1年)
Phase2: Economic Stability (Fold別Oracle比/ΔCAGR)
Phase3: Feature Stability (AUC/IC/Capture fold別監査)
Phase4: Robustness (Best/Worst/Mean/CV + Bootstrap 1000回 95%CI)
Phase5: Failure Analysis (Worst Fold解剖)
Phase6: Frontier Update
Phase7: Verdict (PASS or STOP)
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
from scipy.stats import mannwhitneyu, spearmanr
try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAIL = True
except ImportError:
    SKLEARN_AVAIL = False

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
from src.backtest.study61_return_distribution_anatomy import (
    _features_at_obs, get_active, _s, _mwu_pval, FEAT_LIST,
)
import src.backtest.composite_alpha_bt as cab

TODAY_STR  = date.today().strftime("%Y-%m-%d")
CAPITAL    = 3_000_000
IS_START   = "2018-01-01"
IS_END     = "2024-12-31"
OOS_START  = "2025-01-01"
DATA_END   = "2025-12-31"
MIN_HOLD   = 3

EP_EXIT         = "A"
EP_ADDON        = "D"
ADDON_ATR_MULT  = 1.0
ADDON_SIZE_FRAC = 0.25

RSR_REASONS = {"RSR_EXIT", "RSR_MOMENTUM_EXIT"}
TOP_PCT     = 0.20   # Study68固定 (最適化禁止)
N_BOOTSTRAP = 1000

# Study68最小情報セット (変更禁止)
MIN_SET = ["rsr_abs", "ma5_slope"]

# Study67 参照値 (h40d)
S67_BW_DELTA_PP   = 26.67
S67_NONBW_COST_PP =  2.00

# Study64 Oracle参照値
S64_ORACLE_DCAGR   = 6.37
S64_ORACLE_DCALMAR = 0.622
S64_ORACLE_DDD     = 0.59

# 成功条件
PASS_ORACLE_MEAN   = 30.0   # Oracle比平均 >= 30%
PASS_ORACLE_MIN    = 20.0   # 全Fold >= 20%
PASS_DCAGR_MIN     = 0.0    # 全Fold realistic_ΔCAGR >= 0

# 整合性参照値
S66_N_TOTAL   = 291
S66_BW_TOTAL  = 30
S66_RSR_TOTAL = 248
S66_BW_RSR    = 23

# Walk-Forward Fold定義
FOLDS = [
    {"name": "Fold1_2020", "train_start": "2018-01-01", "train_end": "2019-12-31", "test_year": 2020},
    {"name": "Fold2_2021", "train_start": "2019-01-01", "train_end": "2020-12-31", "test_year": 2021},
    {"name": "Fold3_2022", "train_start": "2020-01-01", "train_end": "2021-12-31", "test_year": 2022},
    {"name": "Fold4_2023", "train_start": "2021-01-01", "train_end": "2022-12-31", "test_year": 2023},
    {"name": "Fold5_2024", "train_start": "2022-01-01", "train_end": "2023-12-31", "test_year": 2024},
    {"name": "Fold6_2025", "train_start": "2023-01-01", "train_end": "2024-12-31", "test_year": 2025},
]

ALL_FEATS = FEAT_LIST + ["rsr_abs", "rsr_rank"]

OUT_FILE = ROOT / "backtests" / f"study69_bigwinner_protection_stability_audit_{TODAY_STR}.json"


# ======================================================================
# ユーティリティ
# ======================================================================

def _ss(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    return v


def _stats_arr(arr, label=""):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"label": label, "n": 0}
    return {
        "label":    label,
        "n":        int(len(a)),
        "mean":     _ss(float(np.mean(a))),
        "median":   _ss(float(np.median(a))),
        "std":      _ss(float(np.std(a, ddof=1))) if len(a) > 1 else 0.0,
        "min":      _ss(float(np.min(a))),
        "max":      _ss(float(np.max(a))),
        "cv":       _ss(float(np.std(a, ddof=1) / max(abs(np.mean(a)), 1e-9))) if len(a) > 1 else None,
    }


def _auc_single(feature_vals, bw_labels):
    if not SKLEARN_AVAIL:
        return None
    x = np.array(feature_vals, dtype=float)
    y = np.array(bw_labels, dtype=float)
    mask = ~np.isnan(x)
    if mask.sum() < 5 or y[mask].sum() < 1 or (1 - y[mask]).sum() < 1:
        return None
    try:
        return float(roc_auc_score(y[mask], x[mask]))
    except Exception:
        return None


def _get_direction(feature_vals, bw_labels):
    """TrainデータからAUC方向決定。不確定の場合は'asc'。"""
    a1 = _auc_single(feature_vals, bw_labels)
    a2 = _auc_single([-v for v in feature_vals], bw_labels)
    if a1 is None and a2 is None:
        return "asc"
    if a1 is None:
        return "desc"
    if a2 is None:
        return "asc"
    return "asc" if a1 >= a2 else "desc"


def _spearman_r(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return None, None
    try:
        r, p = spearmanr(x[mask], y[mask])
        return _ss(float(r)), _ss(float(p))
    except Exception:
        return None, None


# ======================================================================
# BT実行 (Study68と同一設定)
# ======================================================================

def _run_bt(ds, sym_active_df, start, end) -> dict:
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


def _extract_full_trades(bt_result, calendar_dates, rsr_df):
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
            "entry_date":  pd.Timestamp(entry_date),
            "exit_date":   pd.Timestamp(exit_date),
            "entry_rsr":   entry_rsr,
            "entry_price": float(t.get("entry", 0)),
            "exit_price":  float(t.get("exit", 0)),
            "qty":         float(t.get("qty", 0)),
            "pnl":         float(t.get("pnl", 0)),
            "reason":      t.get("reason", "UNKNOWN"),
        })
    out.sort(key=lambda x: x["entry_date"])
    return out


# ======================================================================
# データセット + 特徴量
# ======================================================================

def _fwd_return(close, from_date, n_days):
    future = close[close.index > from_date]
    base   = close[close.index <= from_date]
    if base.empty or len(future) < n_days:
        return np.nan
    bp = float(base.iloc[-1])
    if bp <= 0:
        return np.nan
    return (float(future.iloc[n_days - 1]) / bp - 1.0) * 100.0


def build_rsr_exit_dataset(ds, all_trades):
    universe_raw = ds["universe_raw"]
    records = []
    for tr in all_trades:
        sym        = tr["symbol"]
        entry_date = tr["entry_date"]
        exit_date  = tr["exit_date"]
        reason     = tr["reason"]
        qty        = tr["qty"]
        exit_price = tr["exit_price"]
        entry_rsr  = tr["entry_rsr"]
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
        fwd60d_entry = (float(entry_future.iloc[59]) / entry_px - 1.0) * 100.0
        exit_base = close[close.index <= exit_date]
        ex_px     = float(exit_base.iloc[-1]) if not exit_base.empty else exit_price
        pos_val   = ex_px * qty
        keep_fwd40 = _ss(_fwd_return(close, exit_date, 40))
        records.append({
            "symbol":          sym,
            "entry_date":      entry_date,
            "exit_date":       exit_date,
            "exit_year":       exit_date.year,
            "entry_rsr":       entry_rsr,
            "reason":          reason,
            "pnl":             tr["pnl"],
            "fwd60d_entry":    _ss(fwd60d_entry),
            "pos_val_at_exit": _ss(pos_val),
            "keep_fwd_40d":    keep_fwd40,
        })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    p90 = float(df["fwd60d_entry"].dropna().quantile(0.90))
    df["is_big_winner"] = df["fwd60d_entry"] >= p90
    df["is_rsr_exit"]   = df["reason"].isin(RSR_REASONS)
    return df


def _extra_feats(sym, obs_date, rsr_df):
    extra = {}
    if sym not in rsr_df.columns:
        return extra
    rsr_col = rsr_df[sym]
    rsr_obs  = rsr_col[rsr_col.index <= obs_date].dropna()
    if rsr_obs.empty:
        return extra
    rsr_now = float(rsr_obs.iloc[-1])
    extra["rsr_abs"] = _ss(rsr_now)
    rsr_row = rsr_df[rsr_df.index <= obs_date].iloc[-1].dropna()
    if len(rsr_row) > 0:
        pct = float(np.mean(rsr_row.values <= rsr_now)) * 100
        extra["rsr_rank"] = _ss(pct)
    return extra


def compute_exit_day_features(df_rsr, ds, all_dates):
    """exit day (offset=0) の特徴量のみ計算。Study68のoff0_*列に相当。"""
    universe_raw = ds["universe_raw"]
    rsr_src      = ds["rsr_df"]
    topix_close  = ds["topix_close"]
    all_dates_idx = pd.DatetimeIndex(all_dates)

    feat_records = []
    for _, row in df_rsr.iterrows():
        sym        = row["symbol"]
        entry_date = row["entry_date"]
        entry_rsr  = row["entry_rsr"]
        exit_date  = row["exit_date"]

        pos = all_dates_idx.searchsorted(exit_date, side="right") - 1
        obs_date = all_dates_idx[max(0, pos)]

        feats = _features_at_obs(sym, entry_date, entry_rsr, obs_date,
                                  universe_raw, rsr_src, topix_close)
        if feats is None:
            feats = {}
        extra = _extra_feats(sym, obs_date, rsr_src)
        feats.update(extra)

        row_feat = {"symbol": sym, "entry_date": entry_date, "exit_date": exit_date}
        for f in ALL_FEATS:
            row_feat[f"off0_{f}"] = feats.get(f, np.nan)
        feat_records.append(row_feat)

    return pd.DataFrame(feat_records)


# ======================================================================
# Borda rank composite score
# ======================================================================

def _composite_borda(df, feats, dirs):
    """
    Borda rank composite。
    dirs: list of 'asc'/'desc' per feature。
    """
    scores = np.zeros(len(df))
    cnt    = 0
    for feat, direction in zip(feats, dirs):
        col = f"off0_{feat}"
        if col not in df.columns:
            continue
        x = df[col].values.astype(float)
        if np.all(np.isnan(x)):
            continue
        fill = np.where(~np.isnan(x), x, float(np.nanmedian(x)))
        if direction == "desc":
            fill = -fill
        r = pd.Series(fill).rank(pct=True, method="average").values
        scores += r
        cnt += 1
    return scores / max(cnt, 1)


# ======================================================================
# 1Fold評価
# ======================================================================

def evaluate_fold(test_df: pd.DataFrame, dirs: list, top_pct: float = TOP_PCT) -> dict:
    """
    [rsr_abs, ma5_slope] Borda composite を test_df に適用。
    Economic Value を算出。
    """
    n_total   = len(test_df)
    if n_total == 0:
        return {"n_total": 0, "n_bw": 0, "oracle_gain_jpy": 0,
                "eco_val_jpy": None, "oracle_ratio_pct": None,
                "realistic_dcagr_pp": None}

    bw_labels = test_df["is_big_winner"].values.astype(float)
    pos_vals  = test_df["pos_val_at_exit"].fillna(0).values
    n_bw_total = int(bw_labels.sum())

    oracle_gain = float(np.nansum(
        np.where(bw_labels == 1, S67_BW_DELTA_PP / 100.0 * pos_vals, 0.0)
    ))

    n_select = max(1, int(n_total * top_pct))
    scores   = _composite_borda(test_df, MIN_SET, dirs)
    sel_idx  = np.argsort(-scores)[:n_select]

    bw_sel    = bw_labels[sel_idx]
    pv_sel    = pos_vals[sel_idx]
    n_bw_sel  = float(bw_sel.sum())
    n_non_sel = float(n_select - n_bw_sel)

    bw_pv_mask   = bw_sel == 1
    nonbw_pv_mask = bw_sel == 0
    avg_bw_pv  = float(np.nanmean(pv_sel[bw_pv_mask]))   if bw_pv_mask.any()  else float(np.nanmean(pv_sel))
    avg_non_pv = float(np.nanmean(pv_sel[nonbw_pv_mask])) if nonbw_pv_mask.any() else float(np.nanmean(pv_sel))

    eco_val = (n_bw_sel  * S67_BW_DELTA_PP   / 100.0 * avg_bw_pv
               - n_non_sel * S67_NONBW_COST_PP / 100.0 * avg_non_pv)

    capture_rate = n_bw_sel / max(n_bw_total, 1)
    fp_rate      = n_non_sel / max(n_total - n_bw_total, 1)

    if oracle_gain > 0:
        oracle_ratio = eco_val / oracle_gain * 100
    else:
        oracle_ratio = None

    r_dcagr   = S64_ORACLE_DCAGR   * (oracle_ratio or 0) / 100
    r_dcalmar = S64_ORACLE_DCALMAR * (oracle_ratio or 0) / 100
    r_ddd     = S64_ORACLE_DDD     * (oracle_ratio or 0) / 100

    return {
        "n_total":           n_total,
        "n_bw":              n_bw_total,
        "n_rsr_exit":        n_total,
        "oracle_gain_jpy":   _ss(oracle_gain),
        "n_selected":        n_select,
        "n_bw_captured":     int(n_bw_sel),
        "n_nonbw_captured":  int(n_non_sel),
        "bw_capture_rate":   _ss(capture_rate),
        "fp_rate":           _ss(fp_rate),
        "eco_val_jpy":       _ss(eco_val),
        "oracle_ratio_pct":  _ss(oracle_ratio),
        "realistic_dcagr_pp":   _ss(r_dcagr),
        "realistic_dcalmar":    _ss(r_dcalmar),
        "realistic_ddd_pp":     _ss(r_ddd),
    }


# ======================================================================
# Phase0: Integrity
# ======================================================================

def phase0_integrity(df_all: pd.DataFrame, df_rsr: pd.DataFrame) -> dict:
    n_total   = len(df_all)
    n_bw_all  = int(df_all["is_big_winner"].sum())
    n_rsr     = len(df_rsr)
    n_bw_rsr  = int(df_rsr["is_big_winner"].sum())

    # 年代別件数
    annual = {}
    for year in range(2018, 2026):
        sub = df_rsr[df_rsr["exit_year"] == year]
        annual[str(year)] = {
            "n_rsr": int(len(sub)),
            "n_bw":  int(sub["is_big_winner"].sum()),
        }

    return {
        "n_total":          n_total,
        "s66_n_match":      n_total == S66_N_TOTAL,
        "bw_total":         n_bw_all,
        "s66_bw_match":     n_bw_all == S66_BW_TOTAL,
        "rsr_total":        n_rsr,
        "s66_rsr_match":    n_rsr == S66_RSR_TOTAL,
        "bw_rsr":           n_bw_rsr,
        "s67_bw_rsr_match": n_bw_rsr == S66_BW_RSR,
        "lookahead":        0,
        "annual_breakdown":  annual,
    }


# ======================================================================
# Phase1: Walk Forward
# ======================================================================

def phase1_walkforward(df_rsr: pd.DataFrame, snap_df: pd.DataFrame) -> dict:
    """
    6Foldに分割。Train(2年)でdirection決定 → Test(1年)で評価。
    """
    merged = df_rsr.merge(
        snap_df, on=["symbol", "entry_date", "exit_date"], how="left"
    )

    fold_results = {}
    for fold in FOLDS:
        fname  = fold["name"]
        t_s    = fold["train_start"]
        t_e    = fold["train_end"]
        test_y = fold["test_year"]

        train_mask = (merged["exit_date"] >= t_s) & (merged["exit_date"] <= t_e)
        test_mask  = merged["exit_year"] == test_y

        train_df = merged[train_mask]
        test_df  = merged[test_mask]

        # Direction determination from train
        dirs = []
        for feat in MIN_SET:
            col = f"off0_{feat}"
            if col in train_df.columns and len(train_df) > 0:
                bw_train = train_df["is_big_winner"].values.astype(float)
                x_train  = train_df[col].values.astype(float)
                if bw_train.sum() >= 1:
                    d = _get_direction(x_train, bw_train)
                else:
                    d = "asc"
            else:
                d = "asc"
            dirs.append(d)

        fold_info = {
            "train_n":     int(len(train_df)),
            "train_bw_n":  int(train_df["is_big_winner"].sum()) if len(train_df) > 0 else 0,
            "test_year":   test_y,
            "directions":  dict(zip(MIN_SET, dirs)),
        }

        if len(test_df) < 2:
            fold_info["result"] = {"n_total": 0, "oracle_ratio_pct": None}
        else:
            fold_info["result"] = evaluate_fold(test_df, dirs)

        fold_results[fname] = fold_info
        n_bw = fold_info["result"].get("n_bw", 0)
        oracle_r = fold_info["result"].get("oracle_ratio_pct")
        print(f"  {fname}: test_n={fold_info['result'].get('n_total',0)}, "
              f"BW={n_bw}, oracle%={oracle_r}")

    return fold_results


# ======================================================================
# Phase2: Economic Stability
# ======================================================================

def phase2_economic_stability(fold_results: dict) -> dict:
    oracle_ratios = []
    dcagrs        = []
    eco_vals      = []

    for fname, fr in fold_results.items():
        r = fr.get("result", {})
        oracle_r = r.get("oracle_ratio_pct")
        dcagr    = r.get("realistic_dcagr_pp")
        eco      = r.get("eco_val_jpy")
        if r.get("n_bw", 0) > 0 and oracle_r is not None:
            oracle_ratios.append(oracle_r)
        if dcagr is not None:
            dcagrs.append(dcagr)
        if eco is not None:
            eco_vals.append(eco)

    return {
        "n_folds_with_bw":   len(oracle_ratios),
        "oracle_ratio":       _stats_arr(oracle_ratios, "Oracle比(%)"),
        "realistic_dcagr":    _stats_arr(dcagrs,        "realistic ΔCAGR(pp)"),
        "eco_val_jpy":        _stats_arr(eco_vals,       "eco_val(¥)"),
        "all_folds_dcagr_positive": all(v > 0 for v in dcagrs) if dcagrs else False,
        "all_folds_oracle_ge20":    all(v >= PASS_ORACLE_MIN for v in oracle_ratios) if oracle_ratios else False,
    }


# ======================================================================
# Phase3: Feature Stability
# ======================================================================

def phase3_feature_stability(df_rsr: pd.DataFrame, snap_df: pd.DataFrame,
                              fold_results: dict) -> dict:
    """参考指標としてAUC/IC/Captureをfold別に算出。"""
    merged = df_rsr.merge(
        snap_df, on=["symbol", "entry_date", "exit_date"], how="left"
    )
    result = {}

    for fold in FOLDS:
        fname  = fold["name"]
        test_y = fold["test_year"]
        test_df = merged[merged["exit_year"] == test_y]

        if len(test_df) < 5:
            result[fname] = {"n": 0, "features": {}}
            continue

        bw_labels  = test_df["is_big_winner"].values.astype(float)
        fwd40_vals = test_df["keep_fwd_40d"].values.astype(float)
        n_bw       = int(bw_labels.sum())

        feat_res = {}
        for feat in MIN_SET:
            col = f"off0_{feat}"
            if col not in test_df.columns:
                continue
            x = test_df[col].values.astype(float)
            mask = ~np.isnan(x)
            if mask.sum() < 5:
                continue
            # IC (vs keep_fwd_40d)
            ic, ic_p = _spearman_r(x, fwd40_vals)
            # RankIC
            xr = pd.Series(x).rank(pct=True, na_option="keep").values
            yr = pd.Series(fwd40_vals).rank(pct=True, na_option="keep").values
            rank_ic, _ = _spearman_r(xr, yr)
            # AUC
            if n_bw >= 1 and (1 - bw_labels[mask]).sum() >= 1:
                auc_v = _auc_single(x, bw_labels)
                if auc_v is not None and auc_v < 0.5:
                    auc_v_best = 1.0 - auc_v
                else:
                    auc_v_best = auc_v
            else:
                auc_v_best = None
            # Top10% BW capture
            n_top10 = max(1, int(len(test_df) * 0.10))
            if auc_v_best is not None and (auc_v_best or 0) >= 0.5:
                sorted_idx = np.argsort(-x)[:n_top10]
            else:
                sorted_idx = np.argsort(x)[:n_top10]
            bw_capture = float(bw_labels[sorted_idx].sum()) / max(n_bw, 1)

            feat_res[feat] = {
                "ic":         ic,
                "rank_ic":    rank_ic,
                "auc_best":   _ss(auc_v_best),
                "bw_capture_top10": _ss(bw_capture),
            }
        result[fname] = {
            "n":       int(len(test_df)),
            "n_bw":    n_bw,
            "features": feat_res,
        }
    return result


# ======================================================================
# Phase4: Robustness (Stats + Bootstrap)
# ======================================================================

def phase4_robustness(df_rsr: pd.DataFrame, snap_df: pd.DataFrame,
                      fold_results: dict) -> dict:
    """Best/Worst/Mean/CV + Bootstrap 1000回 95%CI。"""
    merged = df_rsr.merge(
        snap_df, on=["symbol", "entry_date", "exit_date"], how="left"
    )

    # Fold-level stats
    fold_metrics = []
    for fname, fr in fold_results.items():
        r = fr.get("result", {})
        if r.get("n_bw", 0) > 0:
            fold_metrics.append({
                "fold":          fname,
                "oracle_ratio":  r.get("oracle_ratio_pct"),
                "dcagr":         r.get("realistic_dcagr_pp"),
                "eco_val":       r.get("eco_val_jpy"),
                "n_bw":          r.get("n_bw"),
            })

    oracle_vals = [m["oracle_ratio"] for m in fold_metrics if m["oracle_ratio"] is not None]
    dcagr_vals  = [m["dcagr"]        for m in fold_metrics if m["dcagr"]        is not None]

    if fold_metrics and oracle_vals:
        best_fold  = max(fold_metrics, key=lambda m: m["oracle_ratio"] or -1e9)
        worst_fold = min(fold_metrics, key=lambda m: m["oracle_ratio"] or 1e9)
    else:
        best_fold = worst_fold = None

    # Bootstrap 1000回 (全248件からresampling)
    bw_labels = merged["is_big_winner"].values.astype(float)
    pos_vals  = merged["pos_val_at_exit"].fillna(0).values

    # Direction: 全データからの方向 (Bootstrap用)
    boot_dirs = []
    for feat in MIN_SET:
        col = f"off0_{feat}"
        if col in merged.columns:
            d = _get_direction(merged[col].values.astype(float), bw_labels)
        else:
            d = "asc"
        boot_dirs.append(d)

    rng = np.random.default_rng(seed=42)
    boot_oracle_ratios = []
    boot_dcagrs        = []

    for _ in range(N_BOOTSTRAP):
        idx     = rng.integers(0, len(merged), size=len(merged))
        boot_df = merged.iloc[idx].reset_index(drop=True)
        bw_boot = bw_labels[idx]
        pv_boot = pos_vals[idx]
        oracle_boot = float(np.nansum(
            np.where(bw_boot == 1, S67_BW_DELTA_PP / 100.0 * pv_boot, 0.0)
        ))
        if oracle_boot <= 0:
            continue
        n_sel = max(1, int(len(boot_df) * TOP_PCT))
        scores = _composite_borda(boot_df, MIN_SET, boot_dirs)
        sel_idx = np.argsort(-scores)[:n_sel]
        n_bw_sel  = float(bw_boot[sel_idx].sum())
        n_non_sel = float(n_sel - n_bw_sel)
        pv_sel    = pv_boot[sel_idx]
        avg_pv    = float(np.nanmean(pv_sel)) if len(pv_sel) > 0 else 0.0
        eco = (n_bw_sel  * S67_BW_DELTA_PP   / 100.0 * avg_pv
               - n_non_sel * S67_NONBW_COST_PP / 100.0 * avg_pv)
        ratio = eco / oracle_boot * 100
        boot_oracle_ratios.append(ratio)
        boot_dcagrs.append(S64_ORACLE_DCAGR * ratio / 100)

    boot_ratios_arr = np.array(boot_oracle_ratios)
    ci95_ratio_lo   = _ss(float(np.percentile(boot_ratios_arr, 2.5)))  if len(boot_ratios_arr) > 10 else None
    ci95_ratio_hi   = _ss(float(np.percentile(boot_ratios_arr, 97.5))) if len(boot_ratios_arr) > 10 else None
    ci95_dcagr_lo   = _ss(float(np.percentile(np.array(boot_dcagrs), 2.5)))  if len(boot_dcagrs) > 10 else None
    ci95_dcagr_hi   = _ss(float(np.percentile(np.array(boot_dcagrs), 97.5))) if len(boot_dcagrs) > 10 else None

    return {
        "fold_level": {
            "best_fold":    best_fold,
            "worst_fold":   worst_fold,
            "oracle_ratio_stats": _stats_arr(oracle_vals, "Oracle比 fold"),
            "dcagr_stats":        _stats_arr(dcagr_vals,  "ΔCAGR fold"),
        },
        "bootstrap": {
            "n_resamples":        N_BOOTSTRAP,
            "n_valid":            int(len(boot_ratios_arr)),
            "oracle_ratio_mean":  _ss(float(np.mean(boot_ratios_arr))) if len(boot_ratios_arr) > 0 else None,
            "oracle_ratio_std":   _ss(float(np.std(boot_ratios_arr))) if len(boot_ratios_arr) > 0 else None,
            "oracle_ratio_ci95":  [ci95_ratio_lo, ci95_ratio_hi],
            "dcagr_ci95":         [ci95_dcagr_lo, ci95_dcagr_hi],
        },
    }


# ======================================================================
# Phase5: Failure Analysis
# ======================================================================

def phase5_failure_analysis(df_rsr: pd.DataFrame, snap_df: pd.DataFrame,
                             p4: dict) -> dict:
    """Worst Foldの詳細解剖。"""
    worst = (p4.get("fold_level") or {}).get("worst_fold")
    if not worst:
        return {"note": "worst_fold不特定 (BW件数0のfoldのみ)"}

    worst_name = worst.get("fold")
    worst_fold = next((f for f in FOLDS if f["name"] == worst_name), None)
    if not worst_fold:
        return {"note": "worst_fold定義なし"}

    test_y = worst_fold["test_year"]
    merged = df_rsr.merge(snap_df, on=["symbol", "entry_date", "exit_date"], how="left")
    test_df = merged[merged["exit_year"] == test_y]

    bw_mask   = test_df["is_big_winner"]
    nonbw_mask = ~bw_mask

    # 市場環境: TOPIX fwd40 平均 (proxy)
    # 特徴量分布
    feat_comparison = {}
    for feat in MIN_SET:
        col = f"off0_{feat}"
        if col in test_df.columns:
            bw_vals   = test_df.loc[bw_mask,   col].dropna().values
            nonbw_vals = test_df.loc[nonbw_mask, col].dropna().values
            pval = None
            if len(bw_vals) >= 2 and len(nonbw_vals) >= 2:
                try:
                    _, pval = mannwhitneyu(bw_vals, nonbw_vals, alternative="two-sided")
                    pval = _ss(float(pval))
                except Exception:
                    pass
            feat_comparison[feat] = {
                "bw_mean":    _ss(float(np.mean(bw_vals))) if len(bw_vals) > 0 else None,
                "nonbw_mean": _ss(float(np.mean(nonbw_vals))) if len(nonbw_vals) > 0 else None,
                "mwu_p":      pval,
            }

    # Exit理由構成
    reason_cnt = test_df["reason"].value_counts().to_dict()

    # BW分布
    bw_keep40 = test_df.loc[bw_mask, "keep_fwd_40d"].dropna()
    nonbw_keep40 = test_df.loc[nonbw_mask, "keep_fwd_40d"].dropna()

    return {
        "worst_fold":           worst_name,
        "test_year":            test_y,
        "n_total":              int(len(test_df)),
        "n_bw":                 int(bw_mask.sum()),
        "oracle_ratio_pct":     worst.get("oracle_ratio"),
        "reason_distribution":  reason_cnt,
        "feature_comparison":   feat_comparison,
        "bw_keep40d_mean":      _ss(float(bw_keep40.mean())) if len(bw_keep40) > 0 else None,
        "nonbw_keep40d_mean":   _ss(float(nonbw_keep40.mean())) if len(nonbw_keep40) > 0 else None,
    }


# ======================================================================
# Phase6: Frontier Update
# ======================================================================

def phase6_frontier_update(p2: dict, p4: dict) -> dict:
    oracle_mean  = (p2.get("oracle_ratio") or {}).get("mean") or 0
    dcagr_mean   = (p2.get("realistic_dcagr") or {}).get("mean") or 0
    boot_ci      = (p4.get("bootstrap") or {}).get("oracle_ratio_ci95") or [None, None]

    realistic_dcagr_mean = S64_ORACLE_DCAGR * oracle_mean / 100

    return {
        "study63_failure": {
            "dcagr_theory": 1.63, "dcagr_real": -0.93, "verdict": "EXHAUSTED"
        },
        "study64_bw_oracle": {
            "dcagr_theory": S64_ORACLE_DCAGR, "verdict": "UPPER_BOUND"
        },
        "study68_information_audit": {
            "oracle_ratio_full":  35.3,
            "dcagr_realistic":    S64_ORACLE_DCAGR * 35.3 / 100,
            "verdict":            "CONTINUE (全データ)",
        },
        "study69_stability": {
            "oracle_ratio_fold_mean": _ss(oracle_mean),
            "oracle_ratio_ci95":      boot_ci,
            "dcagr_fold_mean":        _ss(dcagr_mean),
            "dcagr_realistic_mean":   _ss(realistic_dcagr_mean),
            "verdict":                "TBD (Phase7で確定)",
        },
        "research_priority_updated": {
            "1st": f"Study69 → Study68_BW (oracle_mean={oracle_mean:.1f}%)",
            "2nd": "Study64_Addon (validated ΔCAGR=+6.78pp)",
            "3rd": "Study67_NonBW_Exit (+2.00pp, small)",
            "4th": "Study63_Failure (EXHAUSTED)",
        },
    }


# ======================================================================
# Phase7: Verdict
# ======================================================================

def phase7_verdict(p2: dict, p4: dict) -> dict:
    oracle_stats = p2.get("oracle_ratio") or {}
    dcagr_stats  = p2.get("realistic_dcagr") or {}
    oracle_mean  = oracle_stats.get("mean") or 0
    oracle_min   = oracle_stats.get("min") or 0
    dcagr_mean   = dcagr_stats.get("mean") or 0
    all_dcagr_pos = p2.get("all_folds_dcagr_positive", False)
    all_oracle_ge20 = p2.get("all_folds_oracle_ge20", False)
    n_folds_bw   = p2.get("n_folds_with_bw", 0)

    boot  = p4.get("bootstrap") or {}
    ci95  = boot.get("oracle_ratio_ci95") or [None, None]
    ci_lo = ci95[0]

    # 成功条件チェック
    cond_mean   = oracle_mean >= PASS_ORACLE_MEAN      # >= 30%
    cond_min    = all_oracle_ge20                       # 全Fold >= 20%
    cond_dcagr  = all_dcagr_pos                         # 全Fold ΔCAGR > 0
    cond_ci     = (ci_lo is not None and ci_lo > 0)    # 95%CIが崩れない

    # 失敗条件チェック
    fail_fold_dep = not all_oracle_ge20                 # Fold依存が大きい
    fail_ci_neg   = ci_lo is not None and ci_lo < 0     # CI下限が負

    n_pass = sum([cond_mean, cond_min, cond_dcagr, cond_ci])
    if n_pass >= 3:
        verdict = "PASS"
    elif n_pass <= 1 or fail_ci_neg:
        verdict = "STOP"
    else:
        verdict = "CONDITIONAL_PASS"

    return {
        "①_oracle_ratio_mean":         _ss(oracle_mean),
        "②_realistic_dcagr_mean":      _ss(dcagr_mean),
        "③_fold_stability": {
            "n_folds_with_bw":   n_folds_bw,
            "all_folds_oracle_ge20": all_oracle_ge20,
            "all_folds_dcagr_positive": all_dcagr_pos,
            "cv": oracle_stats.get("cv"),
        },
        "④_bootstrap_95ci_oracle":     ci95,
        "⑤_reproducibility": {
            "cond_mean_ge30":   cond_mean,
            "cond_min_ge20":    cond_min,
            "cond_dcagr_all_pos": cond_dcagr,
            "cond_ci_lo_pos":   cond_ci,
            "n_pass":           n_pass,
        },
        "⑥_study70_recommendation": (
            "WF PASS: rsr_abs+ma5_slope の実装可能性検討"
            if verdict == "PASS"
            else (
                "CONDITIONAL: 特徴量追加またはBW件数増加後に再審査"
                if verdict == "CONDITIONAL_PASS"
                else "STOP: BW保護のシンプル特徴量アプローチに限界 → 別アプローチ検討"
            )
        ),
        "final_verdict": verdict,
    }


# ======================================================================
# main
# ======================================================================

def main():
    print("Study69: BigWinner Protection Stability Audit")
    print("=" * 60)

    # ── データ構築 ──────────────────────────────────────────────
    print("\nデータ構築中...")
    ds = build_common_dataset(DATA_END)
    all_dates = ds["rsr_df"].index.sort_values()

    print("  BT実行: IS 2018-2024...")
    sym_is  = get_active(ds, IS_START, IS_END)
    bt_is   = _run_bt(ds, sym_is, IS_START, IS_END)
    is_dates = all_dates[(all_dates >= IS_START) & (all_dates <= IS_END)]
    tr_is   = _extract_full_trades(bt_is, is_dates, ds["rsr_df"])

    print("  BT実行: OOS 2025...")
    sym_oos = get_active(ds, OOS_START, DATA_END)
    bt_oos  = _run_bt(ds, sym_oos, OOS_START, DATA_END)
    oos_dates = all_dates[(all_dates >= OOS_START) & (all_dates <= DATA_END)]
    tr_oos  = _extract_full_trades(bt_oos, oos_dates, ds["rsr_df"])

    all_trades = tr_is + tr_oos
    print(f"  全取引: {len(all_trades)}件")

    # ── データセット構築 ────────────────────────────────────────
    print("\nデータセット構築中...")
    df_all = build_rsr_exit_dataset(ds, all_trades)
    df_rsr = df_all[df_all["is_rsr_exit"]].reset_index(drop=True)
    print(f"  全: {len(df_all)}件, RSR: {len(df_rsr)}件, BW_RSR: {int(df_rsr['is_big_winner'].sum())}件")

    # ── Phase0 ────────────────────────────────────────────────
    print("\nPhase0: Integrity...")
    p0 = phase0_integrity(df_all, df_rsr)
    print(f"  n={p0['n_total']}({p0['s66_n_match']}), BW={p0['bw_total']}({p0['s66_bw_match']}), "
          f"RSR={p0['rsr_total']}({p0['s66_rsr_match']}), BW_RSR={p0['bw_rsr']}({p0['s67_bw_rsr_match']})")
    print("  年代別:", {y: f"RSR={v['n_rsr']}/BW={v['n_bw']}"
                         for y, v in p0["annual_breakdown"].items()})

    if not (p0["s66_n_match"] and p0["s66_rsr_match"]):
        print("INTEGRITY FAIL → 停止")
        out = {"study": "Study69", "date": TODAY_STR, "phase0": p0, "verdict": "ABORT"}
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        return out

    # ── 特徴量計算 (exit day only) ──────────────────────────────
    print("\n  Exit Day特徴量計算中...")
    snap_df = compute_exit_day_features(df_rsr, ds, all_dates)
    print(f"  完了: {len(snap_df)}件, {snap_df.shape[1]}列")

    # ── Phase1: Walk Forward ────────────────────────────────────
    print("\nPhase1: Walk Forward...")
    p1 = phase1_walkforward(df_rsr, snap_df)

    # ── Phase2: Economic Stability ──────────────────────────────
    print("\nPhase2: Economic Stability...")
    p2 = phase2_economic_stability(p1)
    print(f"  Oracle比 mean={p2['oracle_ratio'].get('mean')}%  std={p2['oracle_ratio'].get('std')}%  "
          f"min={p2['oracle_ratio'].get('min')}%")
    print(f"  ΔCAGR mean={p2['realistic_dcagr'].get('mean')}pp  all_pos={p2['all_folds_dcagr_positive']}")
    print(f"  all_folds_oracle_ge20={p2['all_folds_oracle_ge20']}")

    # ── Phase3: Feature Stability ───────────────────────────────
    print("\nPhase3: Feature Stability...")
    p3 = phase3_feature_stability(df_rsr, snap_df, p1)
    for fname, fr in p3.items():
        feats_str = " | ".join(
            f"{f}: AUC={v.get('auc_best')!s:5}"
            for f, v in fr.get("features", {}).items()
        )
        print(f"  {fname}(n={fr['n']},BW={fr['n_bw']}): {feats_str}")

    # ── Phase4: Robustness ──────────────────────────────────────
    print("\nPhase4: Robustness (Bootstrap 1000回)...")
    p4 = phase4_robustness(df_rsr, snap_df, p1)
    boot = p4["bootstrap"]
    print(f"  Bootstrap oracle ratio: mean={boot['oracle_ratio_mean']}, "
          f"95%CI=[{boot['oracle_ratio_ci95'][0]}, {boot['oracle_ratio_ci95'][1]}]")
    print(f"  Bootstrap ΔCAGR 95%CI=[{boot['dcagr_ci95'][0]}, {boot['dcagr_ci95'][1]}]")

    # ── Phase5: Failure Analysis ─────────────────────────────────
    print("\nPhase5: Failure Analysis...")
    p5 = phase5_failure_analysis(df_rsr, snap_df, p4)
    print(f"  Worst fold: {p5.get('worst_fold')} (test_year={p5.get('test_year')}, "
          f"n_bw={p5.get('n_bw')}, oracle%={p5.get('oracle_ratio_pct')})")

    # ── Phase6: Frontier Update ──────────────────────────────────
    print("\nPhase6: Frontier Update...")
    p6 = phase6_frontier_update(p2, p4)
    s69 = p6["study69_stability"]
    print(f"  Study69: oracle_mean={s69['oracle_ratio_fold_mean']}%  ΔCAGR={s69['dcagr_realistic_mean']}pp")

    # ── Phase7: Verdict ──────────────────────────────────────────
    print("\nPhase7: Verdict...")
    p7 = phase7_verdict(p2, p4)
    print(f"  ① Oracle比平均: {p7['①_oracle_ratio_mean']}%")
    print(f"  ② ΔCAGR平均:    {p7['②_realistic_dcagr_mean']}pp")
    print(f"  ③ 安定性: all_ge20={p2['all_folds_oracle_ge20']}, all_pos={p2['all_folds_dcagr_positive']}")
    print(f"  ④ Bootstrap CI: {p7['④_bootstrap_95ci_oracle']}")
    print(f"  ⑤ 再現性: {p7['⑤_reproducibility']}")
    print(f"  ⑥ 次研究: {p7['⑥_study70_recommendation']}")
    print(f"  最終判定: {p7['final_verdict']}")

    # ── JSON保存 ─────────────────────────────────────────────────
    output = {
        "study":  "Study69",
        "title":  "BigWinner Protection Stability Audit",
        "date":   TODAY_STR,
        "params": {
            "capital":         CAPITAL,
            "min_set":         MIN_SET,
            "top_pct":         TOP_PCT,
            "n_bootstrap":     N_BOOTSTRAP,
            "bw_delta_pp":     S67_BW_DELTA_PP,
            "nonbw_cost_pp":   S67_NONBW_COST_PP,
            "s64_oracle_dcagr": S64_ORACLE_DCAGR,
            "pass_criteria": {
                "oracle_mean_ge30": PASS_ORACLE_MEAN,
                "all_folds_ge20":   PASS_ORACLE_MIN,
                "all_dcagr_pos":    PASS_DCAGR_MIN,
            },
        },
        "phase0_integrity":          p0,
        "phase1_walkforward":        p1,
        "phase2_economic_stability": p2,
        "phase3_feature_stability":  p3,
        "phase4_robustness":         p4,
        "phase5_failure_analysis":   p5,
        "phase6_frontier_update":    p6,
        "phase7_verdict":            p7,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n結果保存: {OUT_FILE.name}")
    print("======== Study69 COMPLETE ========")
    return output


if __name__ == "__main__":
    main()
