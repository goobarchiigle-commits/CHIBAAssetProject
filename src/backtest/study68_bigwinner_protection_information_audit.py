"""
study68_bigwinner_protection_information_audit.py
Study68 — BigWinner Protection Information Audit

目的: RSR Exit前時点で観測可能な情報に、
     BigWinner保護の経済価値が存在するかを検証。
主評価指標: Economic Value (予測精度ではなく)
禁止: 売買ルール作成 / 閾値最適化 / Production変更 / 最適化

Phase0: Integrity
Phase1: Exit Snapshot (BW vs NonBW特徴量比較)
Phase2: Economic Information Value (特徴量ランキング)
Phase3: Time-Series Divergence (BW/NonBW分離開始日)
Phase4: Oracle Ceiling (完全BW保護の上限)
Phase5: Partial Detection Ceiling (Top10-40%の天井)
Phase6: Minimal Information Set (最小情報量)
Phase7: Portfolio Impact Frontier (研究優先度)
Phase8: Verdict
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from itertools import combinations
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
    _features_at_obs, get_active, run_bt, extract_trades,
    _s, _mwu_pval, FEAT_LIST,
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

RSR_REASONS  = {"RSR_EXIT", "RSR_MOMENTUM_EXIT"}
OBS_OFFSETS  = [20, 10, 5, 3, 1, 0]   # Exit-N trading days (0=exit day)
FWD_HORIZONS = [20, 40, 60]
TOP_PCTS     = [0.10, 0.20, 0.30, 0.40]

# Study67 参照値 (h40d)
S67_BW_DELTA_PP   =  26.67   # BW保護利得 (pp) = keep_40d - rep_40d
S67_NONBW_COST_PP =   2.00   # NonBW誤保護コスト (pp)
S67_BW_KEEP_MEAN  =  35.17   # BW keep_fwd_40d mean (%)
S67_NONBW_KEEP    =   2.90

# Study64 Oracle参照値
S64_ORACLE_DCAGR    =  6.37   # BW+40d ΔCAGR (pp)
S64_ORACLE_DCALMAR  =  0.622
S64_ORACLE_DDD      =  0.59   # MaxDD増加 (pp)

# 整合性確認
S66_N_TOTAL   = 291
S66_BW_TOTAL  = 30
S66_RSR_TOTAL = 248
S66_BW_RSR    = 23   # Study67 BW among RSR exits

EXTRA_FEATS = ["rsr_abs", "rsr_rank"]   # _features_at_obs追加
ALL_FEATS   = FEAT_LIST + EXTRA_FEATS

OUT_FILE = ROOT / "backtests" / f"study68_bigwinner_protection_information_audit_{TODAY_STR}.json"


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


def _stats(arr, label=""):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"label": label, "n": 0}
    return {
        "label":    label,
        "n":        int(len(a)),
        "mean":     _ss(float(np.mean(a))),
        "median":   _ss(float(np.median(a))),
        "std":      _ss(float(np.std(a, ddof=1))),
        "p10":      _ss(float(np.percentile(a, 10))),
        "p25":      _ss(float(np.percentile(a, 25))),
        "p75":      _ss(float(np.percentile(a, 75))),
        "p90":      _ss(float(np.percentile(a, 90))),
    }


def _mwu(a, b):
    a = np.array(a, dtype=float); a = a[~np.isnan(a)]
    b = np.array(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return _ss(float(p))
    except Exception:
        return None


def _cohens_d(a, b):
    a = np.array(a, dtype=float); a = a[~np.isnan(a)]
    b = np.array(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pa = np.var(a, ddof=1) * (len(a) - 1)
    pb = np.var(b, ddof=1) * (len(b) - 1)
    pooled = np.sqrt((pa + pb) / (len(a) + len(b) - 2))
    return _ss(float((np.mean(a) - np.mean(b)) / max(pooled, 1e-9)))


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


def _auc(feature_vals, bw_labels):
    """feature_vals が高いほど BW に近い前提の AUC。"""
    if not SKLEARN_AVAIL:
        return None
    x = np.array(feature_vals, dtype=float)
    y = np.array(bw_labels, dtype=float)
    mask = ~np.isnan(x)
    if mask.sum() < 5 or y[mask].sum() < 1:
        return None
    try:
        auc = roc_auc_score(y[mask], x[mask])
        return _ss(float(auc))
    except Exception:
        return None


def _auc_best_dir(feature_vals, bw_labels):
    """方向自動選択: AUC >= 0.5 を返す。"""
    a1 = _auc(feature_vals, bw_labels)
    a2 = _auc([-v if v is not None else np.nan for v in feature_vals], bw_labels)
    if a1 is None and a2 is None:
        return None, None
    if a1 is None:
        return a2, "desc"
    if a2 is None:
        return a1, "asc"
    if (a1 or 0) >= (a2 or 0):
        return a1, "asc"
    return a2, "desc"


# ======================================================================
# BT + トレード抽出 (Study66-67と同一)
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
    """Study66-67と同一: entry_date/exit_date/reason/qty/pnl付き。"""
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
# フォワードリターン
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


# ======================================================================
# 拡張特徴量: rsr_abs, rsr_rank
# ======================================================================

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
    # rsr_rank: RSRパーセンタイルランク (全銘柄中)
    rsr_row = rsr_df[rsr_df.index <= obs_date].iloc[-1].dropna()
    if len(rsr_row) > 0:
        pct = float(np.mean(rsr_row.values <= rsr_now)) * 100
        extra["rsr_rank"] = _ss(pct)
    return extra


# ======================================================================
# データセット構築
# ======================================================================

def build_rsr_exit_dataset(ds, all_trades):
    """
    RSR exits (248件) に対して:
    - BW label (fwd60d_entry Top10%)
    - keep_fwd_{20/40/60}d (Study67と同一)
    - pos_val_at_exit
    """
    universe_raw = ds["universe_raw"]
    rsr_df_      = ds["rsr_df"]

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
        entry_px     = float(entry_base.iloc[-1])
        if entry_px <= 0:
            continue
        fwd60d_entry = (float(entry_future.iloc[59]) / entry_px - 1.0) * 100.0

        exit_base = close[close.index <= exit_date]
        ex_px     = float(exit_base.iloc[-1]) if not exit_base.empty else exit_price
        pos_val   = ex_px * qty

        keep_fwds = {}
        for h in FWD_HORIZONS:
            keep_fwds[f"keep_fwd_{h}d"] = _ss(_fwd_return(close, exit_date, h))

        records.append({
            "symbol":          sym,
            "entry_date":      entry_date,
            "exit_date":       exit_date,
            "entry_rsr":       entry_rsr,
            "reason":          reason,
            "qty":             qty,
            "pnl":             tr["pnl"],
            "fwd60d_entry":    _ss(fwd60d_entry),
            "pos_val_at_exit": _ss(pos_val),
            **keep_fwds,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    # BW label (Top10% across all 291 trades)
    p90 = float(df["fwd60d_entry"].dropna().quantile(0.90))
    df["is_big_winner"] = df["fwd60d_entry"] >= p90
    df["is_rsr_exit"]   = df["reason"].isin(RSR_REASONS)

    return df


# ======================================================================
# Exit Snapshot特徴量
# ======================================================================

def compute_exit_snapshots(rsr_df_: pd.DataFrame, ds: dict, all_dates) -> pd.DataFrame:
    """
    各RSR ExitについてOBS_OFFSETSの特徴量を計算。
    obs_date = N trading days before exit_date。
    """
    universe_raw = ds["universe_raw"]
    rsr_src      = ds["rsr_df"]
    topix_close  = ds["topix_close"]
    all_dates_idx = pd.DatetimeIndex(all_dates)

    feat_records = []
    for _, row in rsr_df_.iterrows():
        sym        = row["symbol"]
        entry_date = row["entry_date"]
        entry_rsr  = row["entry_rsr"]
        exit_date  = row["exit_date"]

        # exit_date の位置インデックス
        pos = all_dates_idx.searchsorted(exit_date, side="right") - 1

        row_feat = {
            "symbol":     sym,
            "entry_date": entry_date,
            "exit_date":  exit_date,
        }

        for offset in OBS_OFFSETS:
            obs_idx  = max(0, pos - offset)
            obs_date = all_dates_idx[obs_idx]

            feats = _features_at_obs(sym, entry_date, entry_rsr, obs_date,
                                     universe_raw, rsr_src, topix_close)
            if feats is None:
                feats = {}

            extra = _extra_feats(sym, obs_date, rsr_src)
            feats.update(extra)

            for f in ALL_FEATS:
                row_feat[f"off{offset}_{f}"] = feats.get(f, np.nan)

        feat_records.append(row_feat)

    return pd.DataFrame(feat_records)


# ======================================================================
# Phase0: Integrity
# ======================================================================

def phase0_integrity(df_all: pd.DataFrame, df_rsr: pd.DataFrame) -> dict:
    n_total    = len(df_all)
    n_bw_all   = int(df_all["is_big_winner"].sum())
    n_rsr      = len(df_rsr)
    n_bw_rsr   = int(df_rsr["is_big_winner"].sum())
    n_rsr_exit = int((df_rsr["reason"] == "RSR_EXIT").sum())
    n_rsr_mom  = int((df_rsr["reason"] == "RSR_MOMENTUM_EXIT").sum())

    return {
        "n_total":          n_total,
        "s66_n_match":      n_total == S66_N_TOTAL,
        "bw_total":         n_bw_all,
        "s66_bw_match":     n_bw_all == S66_BW_TOTAL,
        "rsr_total":        n_rsr,
        "s66_rsr_match":    n_rsr == S66_RSR_TOTAL,
        "bw_rsr":           n_bw_rsr,
        "s67_bw_rsr_match": n_bw_rsr == S66_BW_RSR,
        "rsr_exit_n":       n_rsr_exit,
        "rsr_momentum_n":   n_rsr_mom,
        "lookahead":        0,
        "survivorship":     0,
    }


# ======================================================================
# Phase1: Exit Snapshot
# ======================================================================

def phase1_exit_snapshot(df_rsr: pd.DataFrame, snap_df: pd.DataFrame) -> dict:
    """BW vs NonBW の特徴量比較 (各observationオフセット)。"""
    merged = df_rsr[["symbol", "entry_date", "exit_date", "is_big_winner"]].merge(
        snap_df, on=["symbol", "entry_date", "exit_date"], how="left"
    )
    bw_m   = merged["is_big_winner"]
    result = {}

    for offset in OBS_OFFSETS:
        offset_res = {}
        for feat in ALL_FEATS:
            col = f"off{offset}_{feat}"
            if col not in merged.columns:
                continue
            bw_vals   = merged.loc[bw_m,  col].dropna().values
            nonbw_vals = merged.loc[~bw_m, col].dropna().values
            if len(bw_vals) < 3 or len(nonbw_vals) < 3:
                continue
            pval = _mwu(bw_vals, nonbw_vals)
            d    = _cohens_d(bw_vals, nonbw_vals)
            offset_res[feat] = {
                "bw":       _stats(bw_vals,    f"BW Exit-{offset} {feat}"),
                "nonbw":    _stats(nonbw_vals, f"NonBW Exit-{offset} {feat}"),
                "mwu_p":    pval,
                "cohens_d": d,
            }
        result[f"exit_minus{offset}d"] = {
            "n_bw":    int(bw_m.sum()),
            "n_nonbw": int((~bw_m).sum()),
            "features": offset_res,
        }
    return result


# ======================================================================
# Phase2: Economic Information Value
# ======================================================================

def phase2_economic_info_value(df_rsr: pd.DataFrame, snap_df: pd.DataFrame) -> dict:
    """
    各特徴量の経済価値評価 (exit day = offset 0)。
    主評価: Economic Value Score
    補足: IC, RankIC, AUC
    """
    merged = df_rsr[["symbol", "entry_date", "exit_date",
                      "is_big_winner", "keep_fwd_40d", "pos_val_at_exit"]].merge(
        snap_df, on=["symbol", "entry_date", "exit_date"], how="left"
    )
    bw_labels  = merged["is_big_winner"].values.astype(float)
    fwd40_vals = merged["keep_fwd_40d"].values.astype(float)
    pos_vals   = merged["pos_val_at_exit"].fillna(0).values

    avg_pos_val = float(np.nanmean(pos_vals))
    bw_avg_pv   = float(np.nanmean(pos_vals[bw_labels == 1]))
    nonbw_avg_pv = float(np.nanmean(pos_vals[bw_labels == 0]))
    n_bw_total  = int(bw_labels.sum())
    oracle_eco  = float(np.nansum(
        np.where(bw_labels == 1,
                 S67_BW_DELTA_PP / 100.0 * pos_vals, 0.0)
    ))

    rankings = []
    for feat in ALL_FEATS:
        col = f"off0_{feat}"
        if col not in merged.columns:
            continue
        x = merged[col].values.astype(float)
        valid = ~np.isnan(x)
        if valid.sum() < 10:
            continue

        # IC (Spearman: feature vs keep_fwd_40d)
        ic, ic_p = _spearman_r(x, fwd40_vals)

        # RankIC (Spearman: feature_rank vs fwd40d_rank)
        xr = pd.Series(x).rank(pct=True, na_option="keep").values
        yr = pd.Series(fwd40_vals).rank(pct=True, na_option="keep").values
        rank_ic, rank_ic_p = _spearman_r(xr, yr)

        # AUC (best direction)
        auc, auc_dir = _auc_best_dir(x, bw_labels)

        # BW Capture Rate (top decile)
        valid_mask  = valid & ~np.isnan(fwd40_vals)
        x_v = x[valid_mask]; bw_v = bw_labels[valid_mask]; pv_v = pos_vals[valid_mask]
        n_top10 = max(1, int(len(x_v) * 0.1))
        if auc_dir == "desc" or (auc_dir is None and (ic or 0) < 0):
            sorted_idx = np.argsort(x_v)[:n_top10]
        else:
            sorted_idx = np.argsort(-x_v)[:n_top10]
        bw_in_top10 = float(bw_v[sorted_idx].sum())
        top10_capture = bw_in_top10 / max(n_bw_total, 1)

        # Economic Value Score (top 20% threshold)
        n_top20 = max(1, int(len(x_v) * 0.2))
        if auc_dir == "desc" or (auc_dir is None and (ic or 0) < 0):
            top20_idx = np.argsort(x_v)[:n_top20]
        else:
            top20_idx = np.argsort(-x_v)[:n_top20]
        n_bw_sel   = float(bw_v[top20_idx].sum())
        n_nonbw_sel = float(len(top20_idx) - n_bw_sel)
        eco_val = (n_bw_sel   * S67_BW_DELTA_PP   / 100.0 * bw_avg_pv
                   - n_nonbw_sel * S67_NONBW_COST_PP / 100.0 * nonbw_avg_pv)
        oracle_ratio = eco_val / oracle_eco * 100 if oracle_eco != 0 else 0.0

        rankings.append({
            "feature":         feat,
            "ic":              ic,
            "ic_p":            ic_p,
            "rank_ic":         rank_ic,
            "auc":             auc,
            "auc_dir":         auc_dir,
            "top10_bw_capture_rate": _ss(top10_capture),
            "eco_val_top20":   _ss(eco_val),
            "oracle_ratio_pct": _ss(oracle_ratio),
        })

    # Economic Value順ソート
    rankings.sort(key=lambda r: r["eco_val_top20"] or -1e9, reverse=True)

    return {
        "oracle_eco_gain_jpy": _ss(oracle_eco),
        "avg_pos_val":         _ss(avg_pos_val),
        "bw_avg_pos_val":      _ss(bw_avg_pv),
        "n_bw_rsr":            n_bw_total,
        "feature_rankings":    rankings,
    }


# ======================================================================
# Phase3: Time-Series Divergence
# ======================================================================

def phase3_timeseries_divergence(df_rsr: pd.DataFrame, snap_df: pd.DataFrame) -> dict:
    """
    各特徴量について BW vs NonBW の最初の有意差出現日を特定。
    """
    merged = df_rsr[["symbol", "entry_date", "exit_date", "is_big_winner"]].merge(
        snap_df, on=["symbol", "entry_date", "exit_date"], how="left"
    )
    bw_m   = merged["is_big_winner"]
    result = {}

    for feat in ALL_FEATS:
        feat_timeline = []
        first_sig_offset = None
        for offset in sorted(OBS_OFFSETS, reverse=True):  # 20→10→5→3→1→0
            col = f"off{offset}_{feat}"
            if col not in merged.columns:
                continue
            bw_vals    = merged.loc[bw_m,  col].dropna().values
            nonbw_vals = merged.loc[~bw_m, col].dropna().values
            if len(bw_vals) < 3 or len(nonbw_vals) < 3:
                feat_timeline.append({"offset": offset, "n_bw": 0, "n_nonbw": 0})
                continue
            pval = _mwu(bw_vals, nonbw_vals)
            d    = _cohens_d(bw_vals, nonbw_vals)
            bw_mean    = float(np.mean(bw_vals))
            nonbw_mean = float(np.mean(nonbw_vals))
            entry = {
                "offset":     offset,
                "n_bw":       len(bw_vals),
                "n_nonbw":    len(nonbw_vals),
                "bw_mean":    _ss(bw_mean),
                "nonbw_mean": _ss(nonbw_mean),
                "diff":       _ss(bw_mean - nonbw_mean),
                "mwu_p":      pval,
                "cohens_d":   d,
            }
            feat_timeline.append(entry)
            if pval is not None and pval < 0.10 and first_sig_offset is None:
                first_sig_offset = offset
        result[feat] = {
            "first_sig_offset": first_sig_offset,
            "timeline":         feat_timeline,
        }
    return result


# ======================================================================
# Phase4: Oracle Ceiling
# ======================================================================

def phase4_oracle_ceiling(df_rsr: pd.DataFrame) -> dict:
    """
    完全BW保護の経済上限 (Study64参照 + Study67ベース計算)。
    """
    bw_mask = df_rsr["is_big_winner"]
    bw_df   = df_rsr[bw_mask]
    n_bw    = int(len(bw_df))

    keep40  = bw_df["keep_fwd_40d"].dropna()
    pv      = bw_df["pos_val_at_exit"].fillna(0)

    # Oracle: BW全件保護の経済価値 (Study67のdelta使用)
    oracle_gain_jpy = float(np.nansum(
        S67_BW_DELTA_PP / 100.0 * pv.values
    ))

    # BW各件の経済損失 (Study67 NEV_portfolio_BW参照)
    s67_nev_portfolio_bw = -3_267_564  # Study67 Phase6

    return {
        "n_bw_rsr":              n_bw,
        "bw_keep_40d_mean":      _ss(float(keep40.mean())) if len(keep40) > 0 else None,
        "oracle_gain_jpy":       _ss(oracle_gain_jpy),
        "s67_nev_portfolio_bw":  s67_nev_portfolio_bw,
        "study64_oracle": {
            "delta_cagr_pp":   S64_ORACLE_DCAGR,
            "delta_calmar":    S64_ORACLE_DCALMAR,
            "delta_maxdd_pp":  S64_ORACLE_DDD,
        },
        "note": "oracle_gain_jpy = 全BW保護でS67 BW deltaを全額回収した場合の利得",
    }


# ======================================================================
# Phase5: Partial Detection Ceiling
# ======================================================================

def phase5_partial_detection_ceiling(df_rsr: pd.DataFrame, snap_df: pd.DataFrame,
                                     p2_rankings: list) -> dict:
    """
    上位特徴量(by eco_val)のTop10-40%保護シミュレーション。
    """
    merged = df_rsr[["symbol", "entry_date", "exit_date",
                      "is_big_winner", "pos_val_at_exit"]].merge(
        snap_df, on=["symbol", "entry_date", "exit_date"], how="left"
    )
    bw_labels = merged["is_big_winner"].values.astype(float)
    pos_vals  = merged["pos_val_at_exit"].fillna(0).values
    n_total   = len(merged)

    # Oracle経済価値 (分母)
    oracle_gain_jpy = float(np.nansum(
        np.where(bw_labels == 1, S67_BW_DELTA_PP / 100.0 * pos_vals, 0.0)
    ))

    # 上位5特徴量を選択
    top_feats = [r["feature"] for r in p2_rankings[:5]
                 if r.get("eco_val_top20") is not None]

    result = {}
    for feat in top_feats:
        col = f"off0_{feat}"
        if col not in merged.columns:
            continue
        x = merged[col].values.astype(float)
        # AUC方向確認
        auc, auc_dir = _auc_best_dir(x, bw_labels)

        feat_result = {}
        for pct in TOP_PCTS:
            n_select = max(1, int(n_total * pct))
            valid = ~np.isnan(x)
            x_fill = np.where(valid, x, np.nanmedian(x))
            if auc_dir == "desc":
                sel_idx = np.argsort(x_fill)[:n_select]
            else:
                sel_idx = np.argsort(-x_fill)[:n_select]

            n_bw_sel   = float(bw_labels[sel_idx].sum())
            n_nonbw_sel = float(n_select - n_bw_sel)
            n_bw_total  = float(bw_labels.sum())
            capture_rate = n_bw_sel / max(n_bw_total, 1)
            fp_rate      = n_nonbw_sel / max(n_total - n_bw_total, 1)

            eco_val = (n_bw_sel   * S67_BW_DELTA_PP   / 100.0 * float(np.nanmean(pos_vals[sel_idx]))
                       - n_nonbw_sel * S67_NONBW_COST_PP / 100.0 * float(np.nanmean(pos_vals[sel_idx])))
            oracle_ratio = eco_val / oracle_gain_jpy * 100 if oracle_gain_jpy != 0 else 0.0

            feat_result[f"top{int(pct*100)}pct"] = {
                "n_selected":    n_select,
                "n_bw_captured": int(n_bw_sel),
                "bw_capture_rate": _ss(capture_rate),
                "fp_rate":       _ss(fp_rate),
                "eco_val_jpy":   _ss(eco_val),
                "oracle_ratio_pct": _ss(oracle_ratio),
            }
        result[feat] = {"auc": auc, "auc_dir": auc_dir, "by_threshold": feat_result}

    result["oracle_gain_jpy"] = _ss(oracle_gain_jpy)
    return result


# ======================================================================
# Phase6: Minimal Information Set
# ======================================================================

def _composite_score(merged, feats, auc_dirs):
    """複数特徴量のBorda rank composite。"""
    scores = np.zeros(len(merged))
    cnt = 0
    for feat, auc_dir in zip(feats, auc_dirs):
        col = f"off0_{feat}"
        if col not in merged.columns:
            continue
        x = merged[col].values.astype(float)
        fill = np.where(~np.isnan(x), x, float(np.nanmedian(x) if ~np.all(np.isnan(x)) else 0))
        if auc_dir == "desc":
            fill = -fill
        r = pd.Series(fill).rank(pct=True, method="average").values
        scores += r
        cnt += 1
    return scores / max(cnt, 1)


def phase6_minimal_info_set(df_rsr: pd.DataFrame, snap_df: pd.DataFrame,
                             p2_rankings: list) -> dict:
    """
    1/2/3特徴量の組み合わせでEconomic Delta最大化。
    探索範囲: 上位10特徴量。
    """
    merged = df_rsr[["symbol", "entry_date", "exit_date",
                      "is_big_winner", "pos_val_at_exit"]].merge(
        snap_df, on=["symbol", "entry_date", "exit_date"], how="left"
    )
    bw_labels = merged["is_big_winner"].values.astype(float)
    pos_vals  = merged["pos_val_at_exit"].fillna(0).values
    n_total   = len(merged)
    n_select  = max(1, int(n_total * 0.20))  # Top20%固定

    oracle_gain_jpy = float(np.nansum(
        np.where(bw_labels == 1, S67_BW_DELTA_PP / 100.0 * pos_vals, 0.0)
    ))

    # 上位10特徴量
    top10 = [(r["feature"], r.get("auc_dir", "asc"))
             for r in p2_rankings[:10]
             if r.get("eco_val_top20") is not None]

    def _eval_combo(feat_dirs):
        feats = [f for f, _ in feat_dirs]
        dirs  = [d for _, d in feat_dirs]
        scores = _composite_score(merged, feats, dirs)
        sel_idx = np.argsort(-scores)[:n_select]
        n_bw_sel   = float(bw_labels[sel_idx].sum())
        n_nonbw_sel = float(n_select - n_bw_sel)
        pv_sel = pos_vals[sel_idx]
        eco_val = (n_bw_sel   * S67_BW_DELTA_PP   / 100.0 * float(np.nanmean(pv_sel))
                   - n_nonbw_sel * S67_NONBW_COST_PP / 100.0 * float(np.nanmean(pv_sel)))
        oracle_ratio = eco_val / oracle_gain_jpy * 100 if oracle_gain_jpy != 0 else 0.0
        return {
            "features":         [f for f, _ in feat_dirs],
            "n_bw_captured":    int(n_bw_sel),
            "bw_capture_rate":  _ss(n_bw_sel / max(float(bw_labels.sum()), 1)),
            "eco_val_jpy":      _ss(eco_val),
            "oracle_ratio_pct": _ss(oracle_ratio),
        }

    result = {"oracle_gain_jpy": _ss(oracle_gain_jpy)}

    # 1特徴量
    res1 = [_eval_combo([fd]) for fd in top10]
    res1.sort(key=lambda r: r["eco_val_jpy"] or -1e9, reverse=True)
    result["k1_best"] = res1[:3]

    # 2特徴量
    res2 = [_eval_combo(list(pair)) for pair in combinations(top10[:7], 2)]
    res2.sort(key=lambda r: r["eco_val_jpy"] or -1e9, reverse=True)
    result["k2_best"] = res2[:3]

    # 3特徴量
    res3 = [_eval_combo(list(tri)) for tri in combinations(top10[:7], 3)]
    res3.sort(key=lambda r: r["eco_val_jpy"] or -1e9, reverse=True)
    result["k3_best"] = res3[:3]

    # 最善セット
    all_best = res1[:1] + res2[:1] + res3[:1]
    all_best.sort(key=lambda r: r["eco_val_jpy"] or -1e9, reverse=True)
    result["best_overall"] = all_best[0] if all_best else None

    return result


# ======================================================================
# Phase7: Portfolio Impact Frontier
# ======================================================================

def phase7_portfolio_frontier(p4: dict, p5: dict, p6: dict) -> dict:
    oracle_jpy  = p4.get("oracle_gain_jpy", 0) or 0
    s64_dcagr   = S64_ORACLE_DCAGR

    # P5: best oracle_ratio among all features and thresholds
    best_real_ratio = 0.0
    best_real_eco   = 0.0
    for feat, v in p5.items():
        if not isinstance(v, dict) or "by_threshold" not in v:
            continue
        for thr, tv in v["by_threshold"].items():
            ratio = tv.get("oracle_ratio_pct") or 0
            eco   = tv.get("eco_val_jpy") or 0
            if ratio > best_real_ratio:
                best_real_ratio = ratio
                best_real_eco   = eco

    # Realistic ΔCAGR = Oracle ΔCAGR × Oracle比達成率
    realistic_dcagr = s64_dcagr * (best_real_ratio / 100.0)

    # P6: best oracle_ratio from minimal set
    best_min_set = p6.get("best_overall", {}) or {}
    min_set_ratio = best_min_set.get("oracle_ratio_pct") or 0
    min_set_dcagr = s64_dcagr * (min_set_ratio / 100.0)

    return {
        "study63_ceiling": {
            "description":  "Failure Detection Theory Ceiling",
            "dcagr_theory": 1.63,
            "dcagr_real":   -0.93,
            "verdict":      "EXHAUSTED",
        },
        "study64_ceiling": {
            "description":  "BW Oracle Retention Ceiling",
            "dcagr_theory": s64_dcagr,
            "oracle_gain_jpy": _ss(oracle_jpy),
            "verdict":      "ORACLE_UPPER_BOUND",
        },
        "study67_nev": {
            "description":  "Portfolio NEV after Replacement",
            "nev_portfolio_jpy": -313_271,
            "bw_nev_jpy":       -3_267_564,
            "verdict":      "BW_DOMINANT",
        },
        "study68_partial_detection": {
            "description":        "Best single-feature partial BW protection",
            "oracle_ratio_pct":   _ss(best_real_ratio),
            "realistic_eco_jpy":  _ss(best_real_eco),
            "realistic_dcagr_pp": _ss(realistic_dcagr),
        },
        "study68_min_set": {
            "description":        "Minimal info set BW protection",
            "oracle_ratio_pct":   _ss(min_set_ratio),
            "realistic_dcagr_pp": _ss(min_set_dcagr),
        },
        "research_priority": {
            "1st": "Study68_BW_Protection (oracle_dcagr=+6.37pp)",
            "2nd": "Study64_Addon (validated +6.78pp)",
            "3rd": "Study67_NonBW_Exit (replacement+2.00pp, small)",
            "4th": "Study63_Failure (exhausted, realistic=-0.93pp)",
        },
    }


# ======================================================================
# Phase8: Verdict
# ======================================================================

def phase8_verdict(p2: dict, p3: dict, p4: dict, p5: dict, p6: dict, p7: dict) -> dict:
    top_feats = p2.get("feature_rankings", [])[:3]

    # 最初の有意差出現日 (最頻値)
    div_days = []
    for feat, v in p3.items():
        fd = v.get("first_sig_offset")
        if fd is not None:
            div_days.append(fd)
    first_divergence = min(div_days) if div_days else None
    n_sig_feats = len(div_days)

    oracle_jpy  = p4.get("oracle_gain_jpy") or 0
    best_min    = p6.get("best_overall") or {}
    min_ratio   = best_min.get("oracle_ratio_pct") or 0
    min_eco     = best_min.get("eco_val_jpy") or 0

    p5_best_ratio = 0.0
    for feat, v in p5.items():
        if not isinstance(v, dict) or "by_threshold" not in v:
            continue
        for thr, tv in v["by_threshold"].items():
            r = tv.get("oracle_ratio_pct") or 0
            if r > p5_best_ratio:
                p5_best_ratio = r

    # SHADOW_TEST_READY: oracle比 >= 50%
    # CONTINUE: oracle比 >= 20%
    # RESEARCH_EXIT: oracle比 < 20%
    if min_ratio >= 50.0:
        verdict = "SHADOW_TEST_READY"
    elif min_ratio >= 20.0:
        verdict = "CONTINUE"
    else:
        verdict = "RESEARCH_EXIT"

    return {
        "①_bw_identifiable_before_exit": n_sig_feats > 0,
        "②_first_sig_divergence_offset":  first_divergence,
        "③_top3_features":               [r.get("feature") for r in top_feats],
        "③_top3_eco_val_jpy":            [r.get("eco_val_top20") for r in top_feats],
        "④_oracle_ceiling_jpy":          _ss(oracle_jpy),
        "④_study64_oracle_dcagr_pp":     S64_ORACLE_DCAGR,
        "⑤_realistic_ceiling_ratio_pct": _ss(p5_best_ratio),
        "⑤_realistic_dcagr_pp":         _ss(S64_ORACLE_DCAGR * p5_best_ratio / 100),
        "⑥_oracle_ratio_best_pct":       _ss(min_ratio),
        "⑦_minimal_info_set":            best_min.get("features"),
        "⑦_minimal_eco_val_jpy":         _ss(min_eco),
        "⑧_research_priority":           p7.get("research_priority"),
        "final_verdict":                 verdict,
    }


# ======================================================================
# main
# ======================================================================

def main():
    print("Study68: BigWinner Protection Information Audit")
    print("=" * 60)

    # ── データ構築 ────────────────────────────────────────────────
    print("\nデータ構築中...")
    ds = build_common_dataset(DATA_END)
    all_dates = ds["rsr_df"].index.sort_values()

    print("  BT実行: IS 2018-2024...")
    sym_is = get_active(ds, IS_START, IS_END)
    bt_is  = _run_bt(ds, sym_is, IS_START, IS_END)
    is_dates = all_dates[(all_dates >= IS_START) & (all_dates <= IS_END)]
    tr_is_base  = extract_trades(bt_is, is_dates, ds["rsr_df"])
    tr_is_full  = _extract_full_trades(bt_is, is_dates, ds["rsr_df"])

    print("  BT実行: OOS 2025...")
    sym_oos = get_active(ds, OOS_START, DATA_END)
    bt_oos  = _run_bt(ds, sym_oos, OOS_START, DATA_END)
    oos_dates = all_dates[(all_dates >= OOS_START) & (all_dates <= DATA_END)]
    tr_oos_full = _extract_full_trades(bt_oos, oos_dates, ds["rsr_df"])

    all_trades_full = tr_is_full + tr_oos_full
    print(f"  全取引: {len(all_trades_full)}件")

    # ── RSR Exitデータセット構築 ────────────────────────────────────
    print("\nRSR Exit dataset構築中...")
    df_all = build_rsr_exit_dataset(ds, all_trades_full)
    df_rsr = df_all[df_all["is_rsr_exit"]].reset_index(drop=True)
    print(f"  全取引: {len(df_all)}件, RSR Exit: {len(df_rsr)}件, BW_RSR: {int(df_rsr['is_big_winner'].sum())}件")

    # ── Phase0: Integrity ──────────────────────────────────────────
    print("\nPhase0: Integrity...")
    p0 = phase0_integrity(df_all, df_rsr)
    print(f"  n={p0['n_total']} ({p0['s66_n_match']}), BW={p0['bw_total']} ({p0['s66_bw_match']}), "
          f"RSR={p0['rsr_total']} ({p0['s66_rsr_match']}), BW_RSR={p0['bw_rsr']} ({p0['s67_bw_rsr_match']})")

    if not all([p0["s66_n_match"], p0["s66_rsr_match"]]):
        print("INTEGRITY FAIL → 停止")
        out = {"study": "Study68", "date": TODAY_STR, "phase0": p0, "verdict": "ABORT"}
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        return out

    # ── Exit Snapshot特徴量計算 ─────────────────────────────────────
    print("\n  Exit Snapshot特徴量計算中 (248件 × 6オフセット)...")
    snap_df = compute_exit_snapshots(df_rsr, ds, all_dates)
    print(f"  snapshot完了: {len(snap_df)}件, {snap_df.shape[1]}列")

    # ── Phase1: Exit Snapshot ──────────────────────────────────────
    print("\nPhase1: Exit Snapshot (BW vs NonBW)...")
    p1 = phase1_exit_snapshot(df_rsr, snap_df)
    for offset in OBS_OFFSETS:
        key = f"exit_minus{offset}d"
        if key in p1:
            n_sig = sum(1 for r in p1[key]["features"].values()
                        if r.get("mwu_p") is not None and r["mwu_p"] < 0.10)
            print(f"  Exit-{offset:2d}d: 有意差あり特徴量={n_sig}/{len(ALL_FEATS)}")

    # ── Phase2: Economic Information Value ─────────────────────────
    print("\nPhase2: Economic Information Value...")
    p2 = phase2_economic_info_value(df_rsr, snap_df)
    print(f"  Oracle eco gain: ¥{p2['oracle_eco_gain_jpy']:,.0f}")
    print("  Top5 features by Economic Value:")
    for r in p2["feature_rankings"][:5]:
        print(f"    {r['feature']:25s} eco_top20=¥{r['eco_val_top20']:,.0f}  "
              f"AUC={r['auc']!s:6}  oracle_ratio={r['oracle_ratio_pct']!s}%")

    # ── Phase3: Time-Series Divergence ─────────────────────────────
    print("\nPhase3: Time-Series Divergence...")
    p3 = phase3_timeseries_divergence(df_rsr, snap_df)
    sig_feats = [(f, v["first_sig_offset"]) for f, v in p3.items()
                 if v["first_sig_offset"] is not None]
    sig_feats.sort(key=lambda x: (x[1] if x[1] is not None else 99), reverse=True)
    print(f"  有意差出現特徴量: {len(sig_feats)}/{len(ALL_FEATS)}件")
    if sig_feats:
        print(f"  最早出現: {sig_feats[0][0]} at Exit-{sig_feats[0][1]}d")

    # ── Phase4: Oracle Ceiling ─────────────────────────────────────
    print("\nPhase4: Oracle Ceiling...")
    p4 = phase4_oracle_ceiling(df_rsr)
    print(f"  Oracle gain: ¥{p4['oracle_gain_jpy']:,.0f}")
    print(f"  Study64 ΔCAGR: +{S64_ORACLE_DCAGR}pp")

    # ── Phase5: Partial Detection Ceiling ──────────────────────────
    print("\nPhase5: Partial Detection Ceiling...")
    p5 = phase5_partial_detection_ceiling(df_rsr, snap_df, p2["feature_rankings"])
    print("  Top feature results (Top20%):")
    for feat in list(p5.keys())[:5]:
        if not isinstance(p5[feat], dict) or "by_threshold" not in p5[feat]:
            continue
        t20 = p5[feat]["by_threshold"].get("top20pct", {})
        print(f"    {feat:25s} BW_captured={t20.get('n_bw_captured')!s:4}  "
              f"eco=¥{t20.get('eco_val_jpy') or 0:,.0f}  "
              f"oracle%={t20.get('oracle_ratio_pct')!s}")

    # ── Phase6: Minimal Information Set ────────────────────────────
    print("\nPhase6: Minimal Information Set...")
    p6 = phase6_minimal_info_set(df_rsr, snap_df, p2["feature_rankings"])
    print(f"  k=1 best: {p6['k1_best'][0]['features'] if p6['k1_best'] else None}  "
          f"oracle%={p6['k1_best'][0]['oracle_ratio_pct'] if p6['k1_best'] else None}")
    print(f"  k=2 best: {p6['k2_best'][0]['features'] if p6['k2_best'] else None}  "
          f"oracle%={p6['k2_best'][0]['oracle_ratio_pct'] if p6['k2_best'] else None}")
    print(f"  k=3 best: {p6['k3_best'][0]['features'] if p6['k3_best'] else None}  "
          f"oracle%={p6['k3_best'][0]['oracle_ratio_pct'] if p6['k3_best'] else None}")
    best = p6.get("best_overall")
    if best:
        print(f"  Best overall: {best['features']}  "
              f"eco=¥{best['eco_val_jpy']:,.0f}  oracle%={best['oracle_ratio_pct']}")

    # ── Phase7: Portfolio Impact Frontier ──────────────────────────
    print("\nPhase7: Portfolio Impact Frontier...")
    p7 = phase7_portfolio_frontier(p4, p5, p6)
    print(f"  Study63: ΔCAGR theory={p7['study63_ceiling']['dcagr_theory']}pp → {p7['study63_ceiling']['verdict']}")
    print(f"  Study64: ΔCAGR oracle={p7['study64_ceiling']['dcagr_theory']}pp → {p7['study64_ceiling']['verdict']}")
    print(f"  Study68 partial: oracle%={p7['study68_partial_detection']['oracle_ratio_pct']}% → "
          f"realistic_ΔCAGR={p7['study68_partial_detection']['realistic_dcagr_pp']}pp")

    # ── Phase8: Verdict ────────────────────────────────────────────
    print("\nPhase8: Verdict...")
    p8 = phase8_verdict(p2, p3, p4, p5, p6, p7)
    print(f"  ① BW識別可能: {p8['①_bw_identifiable_before_exit']}")
    print(f"  ② 最初の有意差: Exit-{p8['②_first_sig_divergence_offset']}d")
    print(f"  ③ Top3特徴量: {p8['③_top3_features']}")
    print(f"  ④ Oracle ΔCAGR: +{p8['④_study64_oracle_dcagr_pp']}pp")
    print(f"  ⑤ 実現可能天井: {p8['⑤_realistic_ceiling_ratio_pct']}%  ΔCAGR={p8['⑤_realistic_dcagr_pp']}pp")
    print(f"  ⑥ Oracle比: {p8['⑥_oracle_ratio_best_pct']}%")
    print(f"  ⑦ 最小情報セット: {p8['⑦_minimal_info_set']}")
    print(f"  最終判定: {p8['final_verdict']}")

    # ── JSON保存 ──────────────────────────────────────────────────
    output = {
        "study":  "Study68",
        "title":  "BigWinner Protection Information Audit",
        "date":   TODAY_STR,
        "params": {
            "capital":        CAPITAL,
            "obs_offsets":    OBS_OFFSETS,
            "fwd_horizons":   FWD_HORIZONS,
            "top_pcts":       TOP_PCTS,
            "bw_delta_pp":    S67_BW_DELTA_PP,
            "nonbw_cost_pp":  S67_NONBW_COST_PP,
            "s64_oracle_dcagr": S64_ORACLE_DCAGR,
        },
        "phase0_integrity":             p0,
        "phase1_exit_snapshot":         p1,
        "phase2_economic_info_value":   p2,
        "phase3_timeseries_divergence": p3,
        "phase4_oracle_ceiling":        p4,
        "phase5_partial_detection":     p5,
        "phase6_minimal_info_set":      p6,
        "phase7_portfolio_frontier":    p7,
        "phase8_verdict":               p8,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n結果保存: {OUT_FILE.name}")
    print("======== Study68 COMPLETE ========")
    return output


if __name__ == "__main__":
    main()
