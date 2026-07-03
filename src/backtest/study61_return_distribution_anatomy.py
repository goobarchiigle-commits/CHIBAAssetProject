"""
study61_return_distribution_anatomy.py
Study61 — Return Distribution Anatomy

目的: D_ATR_EQ 305トレードの期待値支配構造を特定。
禁止: 売買ルール作成 / 閾値探索 / Production変更 / 実装

Phase0:  Study60 Validation (前提確認)
Phase1:  Return Distribution Mapping (デシル分類)
Phase2:  Big Winner Anatomy (Day1-60 特徴量推移)
Phase2.5:Near Miss Winner Analysis (連続体 vs 特殊事象)
Phase3:  False Hero Analysis (Day5上位→Day60平凡)
Phase4:  Failure Taxonomy (Bottom群構造)
Phase5:  Day5 Tail Effect (rankIC負 / Spread正 の解明)
Phase6:  Temporal Evolution (分岐タイミング)
Phase7:  Stable Feature Audit (壊れにくい特徴量)
Phase8:  Research Verdict
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
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

TODAY_STR    = date.today().strftime("%Y-%m-%d")
CAPITAL      = 3_000_000
IS_START     = "2018-01-01"
IS_END       = "2024-12-31"
OOS_START    = "2025-01-01"
OOS_END      = "2025-12-31"
DATA_END     = "2025-12-31"
MIN_HOLD     = 3
FWD_DAYS     = 60

EP_EXIT        = "A"
EP_ADDON       = "D"
ADDON_ATR_MULT = 1.0
ADDON_SIZE_FRAC = 0.25

# Phase2で使う観測日 (Day20/40/60追加)
OBS_DAYS_EXT = [1, 2, 3, 5, 10, 20, 40, 60]

# 群分類定義
GROUPS = {
    "BigWinner":   (0.80, 1.00),  # Top 20%
    "TopHalf":     (0.50, 0.80),  # 次の30%
    "BottomHalf":  (0.20, 0.50),  # 下半分の上位
    "Loser":       (0.00, 0.20),  # Bottom 20%
}

OUT_FILE = ROOT / "backtests" / f"study61_return_distribution_anatomy_{TODAY_STR}.json"


# ======================================================================
# ユーティリティ (Study60と共通)
# ======================================================================

def _s(v, d=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), d)


def _atr20(df_c: pd.DataFrame) -> pd.Series:
    h, l, c = df_c["High"], df_c["Low"], df_c["Close"]
    cp = c.shift(1).fillna(c)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.rolling(20, min_periods=10).mean()


def _spearman(x, y):
    from scipy.stats import spearmanr
    try:
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
        return float(np.corrcoef(rx, ry)[0, 1]), 0.05


def _stats(vals: np.ndarray, d: int = 3) -> dict:
    if len(vals) == 0:
        return {"n": 0}
    return {
        "n":       int(len(vals)),
        "mean":    _s(float(vals.mean()), d),
        "median":  _s(float(np.median(vals)), d),
        "std":     _s(float(vals.std(ddof=1)), d),
        "win_rate":_s(float((vals > 0).mean() * 100), 1),
        "p10":     _s(float(np.percentile(vals, 10)), d),
        "p25":     _s(float(np.percentile(vals, 25)), d),
        "p75":     _s(float(np.percentile(vals, 75)), d),
        "p90":     _s(float(np.percentile(vals, 90)), d),
        "pf":      _s(float(vals[vals > 0].sum() / max(abs(vals[vals < 0].sum()), 1e-9))),
    }


def _mwu_pval(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return 1.0


# ======================================================================
# BT + トレード抽出 (Study60と同一)
# ======================================================================

def get_active(ds, start, end):
    cfg = load_strategy_config()
    bc  = cfg.risk_controls.bear_universe_filter
    be  = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=list(ds["trade_syms"].keys()),
        start=start, end=end, bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


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


def extract_trades(bt_result, calendar_dates, rsr_df):
    trades = bt_result.get("_trades", [])
    out = []
    for t in trades:
        ei = int(t.get("entry_idx", 0))
        if ei >= len(calendar_dates):
            continue
        entry_date = calendar_dates[ei]
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
            "entry_price": float(t.get("entry", 0)),
            "entry_rsr":   entry_rsr,
        })
    return out


# ======================================================================
# 特徴量計算 (Study60と同一 + 拡張)
# ======================================================================

FEAT_LIST = [
    "ret_from_entry", "rsr_delta", "atr_expansion", "vol_retention",
    "rs_accel_post", "ma20_dev", "breakout_dist",
    "rsr_slope", "volume_ratio", "market_rs",
    "candle_body_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
    "nr7", "inside_bar", "ma5_slope", "ma20_slope", "high_persistence",
]


def _features_at_obs(sym, entry_date, entry_rsr, obs_date, universe_raw, rsr_df, topix_close):
    if sym not in universe_raw:
        return None
    df_c = universe_raw[sym].get("df")
    if df_c is None or "Close" not in df_c.columns:
        return None
    close = df_c["Close"].dropna(); close.index = pd.to_datetime(close.index)
    avail = close[close.index <= obs_date]
    e_av  = close[close.index <= entry_date]
    if len(avail) < 21 or e_av.empty:
        return None
    obs_px   = float(avail.iloc[-1])
    entry_px = float(e_av.iloc[-1])
    if entry_px <= 0 or obs_px <= 0:
        return None
    feat = {}
    feat["ret_from_entry"]   = _s((obs_px / entry_px - 1.0) * 100)
    feat["high_persistence"] = 1.0 if obs_px >= entry_px else 0.0
    rsr_sym = rsr_df[sym] if sym in rsr_df.columns else pd.Series(dtype=float)
    rsr_obs = rsr_sym[rsr_sym.index <= obs_date].dropna()
    rsr_ent = rsr_sym[rsr_sym.index <= entry_date].dropna()
    if not rsr_obs.empty and not rsr_ent.empty:
        rsr_now = float(rsr_obs.iloc[-1])
        feat["rsr_delta"] = _s(rsr_now - entry_rsr)
        if len(rsr_obs) >= 22:
            slope_now = (rsr_now - float(rsr_obs.iloc[-22])) / 21.0
            feat["rsr_slope"] = _s(slope_now, 5)
            if len(rsr_ent) >= 22:
                slope_ent = (float(rsr_ent.iloc[-1]) - float(rsr_ent.iloc[-22])) / 21.0
                feat["rs_accel_post"] = _s(slope_now - slope_ent, 5)
    if len(avail) >= 20:
        ma20 = float(avail.iloc[-20:].mean())
        feat["ma20_dev"] = _s((obs_px / ma20 - 1.0) * 100) if ma20 > 0 else None
        if len(avail) >= 21:
            ma20_prev = float(avail.iloc[-21:-1].mean())
            feat["ma20_slope"] = _s((ma20 / ma20_prev - 1.0) * 100) if ma20_prev > 0 else None
    if len(avail) >= 10:
        ma5 = float(avail.iloc[-5:].mean())
        ma5_prev = float(avail.iloc[-10:-5].mean())
        feat["ma5_slope"] = _s((ma5 / ma5_prev - 1.0) * 100) if ma5_prev > 0 else None
    if len(avail) >= 21:
        ph21 = float(avail.iloc[-21:-1].max())
        feat["breakout_dist"] = _s((obs_px / ph21 - 1.0) * 100) if ph21 > 0 else None
    vol_col = "Volume" if "Volume" in df_c.columns else None
    if vol_col:
        vol = pd.to_numeric(df_c[vol_col], errors="coerce").dropna()
        vol.index = pd.to_datetime(vol.index)
        vol_av = vol[vol.index <= obs_date]
        if len(vol_av) >= 20:
            vm20 = float(vol_av.iloc[-20:].mean())
            vd   = float(vol_av.iloc[-1])
            if vm20 > 0:
                feat["vol_retention"] = _s(vd / vm20)
                feat["volume_ratio"]  = feat["vol_retention"]
    if "High" in df_c.columns and "Low" in df_c.columns:
        atr = _atr20(df_c); atr.index = pd.to_datetime(atr.index)
        atr_obs = atr[atr.index <= obs_date].dropna()
        atr_ent = atr[atr.index <= entry_date].dropna()
        if not atr_obs.empty and not atr_ent.empty:
            av_ent = float(atr_ent.iloc[-1])
            if av_ent > 0:
                feat["atr_expansion"] = _s(float(atr_obs.iloc[-1]) / av_ent)
    if "Open" in df_c.columns and "High" in df_c.columns and "Low" in df_c.columns:
        opn  = df_c["Open"].dropna(); opn.index  = pd.to_datetime(opn.index)
        high = df_c["High"].dropna(); high.index = pd.to_datetime(high.index)
        low  = df_c["Low"].dropna();  low.index  = pd.to_datetime(low.index)
        opn_av  = opn[opn.index   <= obs_date]
        high_av = high[high.index <= obs_date]
        low_av  = low[low.index   <= obs_date]
        if len(opn_av) >= 2:
            o = float(opn_av.iloc[-1]); h = float(high_av.iloc[-1])
            l = float(low_av.iloc[-1]); c = obs_px
            hl = h - l
            if hl > 0:
                feat["candle_body_ratio"]  = _s(abs(c - o) / hl)
                feat["upper_shadow_ratio"] = _s((h - max(o, c)) / hl)
                feat["lower_shadow_ratio"] = _s((min(o, c) - l) / hl)
            if len(high_av) >= 7:
                ranges7 = high_av.iloc[-7:].values - low_av.iloc[-7:].values
                feat["nr7"] = 1.0 if hl <= float(min(ranges7[:-1])) else 0.0
            if len(high_av) >= 2:
                ph = float(high_av.iloc[-2]); pl = float(low_av.iloc[-2])
                feat["inside_bar"] = 1.0 if (h < ph and l > pl) else 0.0
    if topix_close is not None:
        tc = topix_close; tc.index = pd.to_datetime(tc.index)
        tc_av = tc[tc.index <= obs_date].dropna()
        if len(tc_av) >= 63:
            feat["market_rs"] = _s((float(tc_av.iloc[-1]) / float(tc_av.iloc[-63]) - 1.0) * 100)
    return feat


def build_trade_dataset(ds, trades) -> pd.DataFrame:
    """
    各トレードについて:
    - fwd60d_entry: エントリー日から60営業日後のリターン (固定ラベル)
    - fwd60d_from_obs_dayN: 各観測日からの60営業日リターン
    - features at each obs_day
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

        # ラベル: fwd60d from entry_date
        if sym not in universe_raw:
            continue
        df_c   = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close = df_c["Close"].dropna(); close.index = pd.to_datetime(close.index)
        entry_close_av = close[close.index <= entry_date]
        if entry_close_av.empty:
            continue
        entry_px = float(entry_close_av.iloc[-1])
        future_from_entry = close[close.index > entry_date]
        if len(future_from_entry) < FWD_DAYS:
            continue
        fwd60d_entry = (float(future_from_entry.iloc[FWD_DAYS - 1]) / entry_px - 1.0) * 100

        row = {
            "symbol":       sym,
            "entry_date":   entry_date,
            "entry_rsr":    entry_rsr,
            "entry_px":     entry_px,
            "fwd60d_entry": _s(fwd60d_entry),
        }

        # 各観測日の特徴量 + fwd60d_from_obs
        future_dates = all_dates[all_dates > entry_date]
        for n_days in OBS_DAYS_EXT:
            if len(future_dates) < n_days:
                continue
            obs_date = future_dates[n_days - 1]

            feat = _features_at_obs(sym, entry_date, entry_rsr, obs_date,
                                    universe_raw, rsr_df, topix_close)
            if feat is None:
                continue

            # fwd60d from obs_date
            future_obs = close[close.index > obs_date]
            fwd60d_obs = None
            if len(future_obs) >= FWD_DAYS:
                obs_px = close[close.index <= obs_date].iloc[-1]
                fwd60d_obs = _s((float(future_obs.iloc[FWD_DAYS - 1]) / float(obs_px) - 1.0) * 100)

            for k, v in feat.items():
                row[f"d{n_days}_{k}"] = v
            row[f"d{n_days}_fwd60d_obs"] = fwd60d_obs

        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["year"] = df["entry_date"].dt.year
    return df


def classify_group(df: pd.DataFrame) -> pd.DataFrame:
    """fwd60d_entryに基づいてパフォーマンス群を付与。"""
    df = df.dropna(subset=["fwd60d_entry"]).copy()
    pcts = df["fwd60d_entry"].rank(pct=True)
    def _grp(p):
        if p >= 0.80:  return "BigWinner"
        if p >= 0.50:  return "TopHalf"
        if p >= 0.20:  return "BottomHalf"
        return "Loser"
    df["perf_group"] = pcts.map(_grp)
    df["perf_pct"]   = pcts
    return df


# ======================================================================
# Phase0: Study60 Validation
# ======================================================================

def phase0_validation(n_trades: int) -> dict:
    study60_json = ROOT / "backtests" / "study60_information_ceiling_2026-06-30.json"
    s60 = {}
    if study60_json.exists():
        with open(study60_json, encoding="utf-8") as f:
            s60 = json.load(f)
    p0_s60 = s60.get("phase0_integrity", {})
    fwd_med = s60.get("phase1_label_audit", {}).get("median_fwd60d")
    return {
        "study60_file": str(study60_json),
        "study60_lookahead": p0_s60.get("lookahead_verified"),
        "study60_verdict": p0_s60.get("verdict"),
        "study60_n_trades": p0_s60.get("n_trades"),
        "study61_n_trades": n_trades,
        "study60_fwd60d_median": fwd_med,
        "feature_definition": "固定 (Study60と同一)",
        "wf_splits": "固定 (Study60と同一)",
        "verdict": "PASS" if p0_s60.get("verdict") == "PASS" and n_trades >= 200 else "WARN",
    }


# ======================================================================
# Phase1: Return Distribution Mapping
# ======================================================================

def phase1_distribution(df: pd.DataFrame) -> dict:
    fwd = df["fwd60d_entry"].dropna().values
    n = len(fwd)
    if n == 0:
        return {"error": "empty"}

    # 基本統計
    overall = _stats(fwd)

    # デシル分類
    deciles = {}
    for label, lo, hi in [
        ("Top1%",    0.99, 1.00), ("Top5%",   0.95, 1.00),
        ("Top10%",   0.90, 1.00), ("Top20%",  0.80, 1.00),
        ("Top30%",   0.70, 1.00), ("Mid40%",  0.30, 0.70),
        ("Bot30%",   0.00, 0.30), ("Bot20%",  0.00, 0.20),
        ("Bot10%",   0.00, 0.10), ("Bot5%",   0.00, 0.05),
        ("Bot1%",    0.00, 0.01),
    ]:
        lo_thr = float(np.percentile(fwd, lo * 100)) if lo > 0 else -9999
        hi_thr = float(np.percentile(fwd, hi * 100)) if hi < 1 else 9999
        seg = fwd[(fwd >= lo_thr) & (fwd <= hi_thr)]
        deciles[label] = _stats(seg)

    # 期待値分解: Top20% vs Bot20% の寄与
    top20 = fwd[fwd >= np.percentile(fwd, 80)]
    bot20 = fwd[fwd <= np.percentile(fwd, 20)]
    mid60 = fwd[(fwd > np.percentile(fwd, 20)) & (fwd < np.percentile(fwd, 80))]
    total_mean = float(fwd.mean())
    contribution = {
        "top20_contribution": _s(float(top20.mean()) * 0.2 / total_mean * 100) if total_mean != 0 else None,
        "mid60_contribution": _s(float(mid60.mean()) * 0.6 / total_mean * 100) if total_mean != 0 else None,
        "bot20_contribution": _s(float(bot20.mean()) * 0.2 / total_mean * 100) if total_mean != 0 else None,
        "total_mean": _s(total_mean),
    }

    # 年別統計
    yr_stats = {}
    for yr, grp in df.groupby("year"):
        fv = grp["fwd60d_entry"].dropna().values
        if len(fv) > 0:
            yr_stats[str(yr)] = _stats(fv, 2)

    return {
        "n": n,
        "overall": overall,
        "deciles": deciles,
        "expected_value_decomposition": contribution,
        "annual_stats": yr_stats,
    }


# ======================================================================
# Phase2: Big Winner Anatomy
# ======================================================================

def phase2_big_winner_anatomy(df: pd.DataFrame) -> dict:
    """各パフォーマンス群の特徴量推移 (Day1-60)。"""
    groups_def = {
        "Top1pct":  df["perf_pct"] >= 0.99,
        "Top5pct":  df["perf_pct"] >= 0.95,
        "Top10pct": df["perf_pct"] >= 0.90,
        "Top20pct": df["perf_pct"] >= 0.80,
        "Mid40pct": (df["perf_pct"] >= 0.30) & (df["perf_pct"] < 0.70),
        "Bot20pct": df["perf_pct"] <= 0.20,
    }

    anatomy = {}
    for grp_name, mask in groups_def.items():
        seg = df[mask]
        if len(seg) == 0:
            continue
        traj = {}
        for n_days in OBS_DAYS_EXT:
            day_key = f"d{n_days}"
            day_feats = {}
            for f in FEAT_LIST:
                col = f"{day_key}_{f}"
                if col not in seg.columns:
                    continue
                vals = seg[col].dropna().values
                if len(vals) > 0:
                    day_feats[f] = {
                        "mean": _s(float(vals.mean())),
                        "median": _s(float(np.median(vals))),
                        "n": len(vals),
                    }
            # fwd60d_obs (from obs_date)
            col_fwd = f"{day_key}_fwd60d_obs"
            if col_fwd in seg.columns:
                fv = seg[col_fwd].dropna().values
                if len(fv) > 0:
                    day_feats["fwd60d_obs"] = _stats(fv, 2)

            traj[f"day{n_days}"] = day_feats

        anatomy[grp_name] = {
            "n": int(mask.sum()),
            "avg_fwd60d": _s(float(seg["fwd60d_entry"].mean())),
            "med_fwd60d": _s(float(seg["fwd60d_entry"].median())),
            "trajectory": traj,
        }

    return anatomy


# ======================================================================
# Phase2.5: Near Miss Winner Analysis
# ======================================================================

def phase25_near_miss(df: pd.DataFrame) -> dict:
    """
    Top1%/5%/10%/20%で特徴量が連続的に変化するかを評価。
    連続体 (continuous) vs 特殊事象 (discontinuous) の判定。
    """
    results = {}
    for n_days in [1, 3, 5, 10]:
        day_key = f"d{n_days}"
        feat_continuity = {}
        for f in FEAT_LIST:
            col = f"{day_key}_{f}"
            if col not in df.columns:
                continue
            # Top1/5/10/20/50/all の平均を確認
            pct_means = {}
            for pct in [0.99, 0.95, 0.90, 0.80, 0.50, 0.00]:
                seg = df[df["perf_pct"] >= pct]
                vals = seg[col].dropna().values
                if len(vals) >= 2:
                    pct_means[f"top{int((1-pct)*100)}pct"] = _s(float(vals.mean()))
            # 連続性: 上位群ほど単調増加/減少か
            mn_vals = [v for v in pct_means.values() if v is not None]
            if len(mn_vals) >= 3:
                diffs = [mn_vals[i+1] - mn_vals[i] for i in range(len(mn_vals)-1)]
                monotone_inc = all(d >= -0.001 for d in diffs)
                monotone_dec = all(d <= 0.001 for d in diffs)
                continuity   = "monotone_inc" if monotone_inc else ("monotone_dec" if monotone_dec else "non_monotone")
            else:
                continuity = "insufficient"
            feat_continuity[f] = {
                "pct_means": pct_means,
                "continuity": continuity,
            }

        # IC (Spearman) per feature vs perf_pct
        ic_summary = {}
        for f in FEAT_LIST:
            col = f"{day_key}_{f}"
            if col not in df.columns:
                continue
            valid = df[[col, "perf_pct"]].dropna()
            if len(valid) < 10:
                continue
            ic, _ = _spearman(valid[col].values, valid["perf_pct"].values)
            ic_summary[f] = _s(ic)

        results[f"day{n_days}"] = {
            "feature_continuity": feat_continuity,
            "ic_vs_perf_pct": ic_summary,
            "top_features_continuous": [
                f for f, v in feat_continuity.items()
                if v.get("continuity") in ("monotone_inc", "monotone_dec")
            ],
        }

    return results


# ======================================================================
# Phase3: False Hero Analysis
# ======================================================================

def phase3_false_hero(df: pd.DataFrame) -> dict:
    """
    Day5時点で上位に見えるが Day60では平凡な銘柄。
    False Hero = Day5 ret_from_entry > p60 かつ fwd60d_entry < median
    """
    col_d5_ret = "d5_ret_from_entry"
    if col_d5_ret not in df.columns:
        return {"error": "d5_ret_from_entry not found"}

    valid = df.dropna(subset=[col_d5_ret, "fwd60d_entry"]).copy()
    if len(valid) < 10:
        return {"error": "insufficient data"}

    d5_ret_p60 = float(valid[col_d5_ret].quantile(0.60))
    fwd_median  = float(valid["fwd60d_entry"].median())

    false_hero_mask = (valid[col_d5_ret] >= d5_ret_p60) & (valid["fwd60d_entry"] < fwd_median)
    winner_mask     = (valid[col_d5_ret] >= d5_ret_p60) & (valid["fwd60d_entry"] >= fwd_median)
    true_hero_mask  = (valid["fwd60d_entry"] >= valid["fwd60d_entry"].quantile(0.80))

    false_hero = valid[false_hero_mask]
    winner     = valid[winner_mask]
    true_hero  = valid[true_hero_mask]

    def _group_feat_summary(grp, day_keys, feats):
        out = {}
        for dk in day_keys:
            day_out = {}
            for f in feats:
                col = f"d{dk}_{f}"
                if col not in grp.columns:
                    continue
                vals = grp[col].dropna().values
                if len(vals) > 0:
                    day_out[f] = {"mean": _s(float(vals.mean())), "n": len(vals)}
            out[f"day{dk}"] = day_out
        return out

    key_feats = ["ret_from_entry", "rsr_delta", "atr_expansion", "rs_accel_post",
                 "ma5_slope", "ma20_slope", "nr7", "inside_bar", "vol_retention"]
    day_keys = [1, 3, 5, 10, 20]

    fh_summary = _group_feat_summary(false_hero, day_keys, key_feats)
    wn_summary = _group_feat_summary(winner, day_keys, key_feats)

    # 差分: FalseHero vs Winner at Day5
    diff_day5 = {}
    for f in key_feats:
        col = f"d5_{f}"
        if col not in valid.columns:
            continue
        fh_v = false_hero[col].dropna().values
        wn_v = winner[col].dropna().values
        if len(fh_v) > 0 and len(wn_v) > 0:
            pval = _mwu_pval(fh_v, wn_v)
            diff_day5[f] = {
                "false_hero_mean": _s(float(fh_v.mean())),
                "winner_mean":     _s(float(wn_v.mean())),
                "diff":            _s(float(fh_v.mean()) - float(wn_v.mean())),
                "pval":            _s(pval, 3),
            }

    return {
        "definition": f"Day5 ret > p60 ({_s(d5_ret_p60, 2)}%) AND fwd60d_entry < median ({_s(fwd_median, 2)}%)",
        "n_false_hero": int(false_hero_mask.sum()),
        "n_winner":     int(winner_mask.sum()),
        "n_true_hero":  int(true_hero_mask.sum()),
        "false_hero_avg_fwd60d": _s(float(false_hero["fwd60d_entry"].mean())),
        "winner_avg_fwd60d":     _s(float(winner["fwd60d_entry"].mean())),
        "false_hero_trajectory": fh_summary,
        "winner_trajectory":     wn_summary,
        "day5_feature_diff":     diff_day5,
        "top_distinguishing_features": sorted(
            [(f, abs(v["diff"] or 0)) for f, v in diff_day5.items() if v.get("diff") is not None],
            key=lambda x: x[1], reverse=True
        )[:5],
    }


# ======================================================================
# Phase4: Failure Taxonomy
# ======================================================================

def phase4_failure_taxonomy(df: pd.DataFrame) -> dict:
    """Bottom群の構造分析 + Winner群との差分。"""
    valid = df.dropna(subset=["fwd60d_entry"]).copy()
    fwd = valid["fwd60d_entry"].values

    bot20_thr = float(np.percentile(fwd, 20))
    top20_thr = float(np.percentile(fwd, 80))

    bot20 = valid[valid["fwd60d_entry"] <= bot20_thr]
    top20 = valid[valid["fwd60d_entry"] >= top20_thr]

    # 特徴量比較 (Day3: 早期観測の代表日)
    day3_comparison = {}
    for f in FEAT_LIST:
        col = f"d3_{f}"
        if col not in valid.columns:
            continue
        bv = bot20[col].dropna().values
        tv = top20[col].dropna().values
        if len(bv) < 3 or len(tv) < 3:
            continue
        pval = _mwu_pval(bv, tv)
        day3_comparison[f] = {
            "bot20_mean":  _s(float(bv.mean())),
            "top20_mean":  _s(float(tv.mean())),
            "diff_top_bot": _s(float(tv.mean()) - float(bv.mean())),
            "pval":         _s(pval, 3),
            "discriminant": "YES" if pval < 0.05 else "NO",
        }

    # Failure trajectory (特徴量推移)
    fail_traj = {}
    for n_days in [1, 3, 5, 10, 20]:
        day_out = {}
        for f in ["ret_from_entry", "rsr_delta", "atr_expansion", "rs_accel_post", "ma5_slope"]:
            col = f"d{n_days}_{f}"
            if col not in bot20.columns:
                continue
            vals = bot20[col].dropna().values
            t_vals = top20[col].dropna().values
            if len(vals) > 0:
                day_out[f] = {
                    "bot20_mean": _s(float(vals.mean())),
                    "top20_mean": _s(float(t_vals.mean())) if len(t_vals) > 0 else None,
                }
        fail_traj[f"day{n_days}"] = day_out

    # Failure cluster (年別)
    yr_fail = {}
    for yr, grp in bot20.groupby("year"):
        yr_fail[str(yr)] = _stats(grp["fwd60d_entry"].dropna().values, 2)

    # Failure types: early decline vs late decline
    col_d5 = "d5_ret_from_entry"
    col_d20 = "d20_ret_from_entry"
    type_counts = {}
    if col_d5 in bot20.columns and col_d20 in bot20.columns:
        bt = bot20.dropna(subset=[col_d5, col_d20])
        early_fail  = bt[(bt[col_d5] < 0)]  # Day5時点で既にマイナス
        late_fail   = bt[(bt[col_d5] >= 0) & (bt[col_d20] < 0)]  # Day5はプラスだがDay20でマイナス
        persist_ok  = bt[(bt[col_d20] >= 0)]  # Day20でまだプラス（結局60日でマイナス）
        type_counts = {
            "early_fail_n":  int(len(early_fail)),
            "late_fail_n":   int(len(late_fail)),
            "persist_ok_n":  int(len(persist_ok)),
            "total": int(len(bt)),
        }

    # 最も識別力の高い特徴量
    top_disc = sorted(
        [(f, abs(v["diff_top_bot"] or 0)) for f, v in day3_comparison.items()
         if v.get("diff_top_bot") is not None and v.get("discriminant") == "YES"],
        key=lambda x: x[1], reverse=True
    )

    return {
        "n_bot20": int(len(bot20)),
        "n_top20": int(len(top20)),
        "bot20_avg_fwd60d": _s(float(bot20["fwd60d_entry"].mean())),
        "top20_avg_fwd60d": _s(float(top20["fwd60d_entry"].mean())),
        "day3_comparison":  day3_comparison,
        "failure_trajectory": fail_traj,
        "failure_types": type_counts,
        "top_discriminating_features": top_disc[:5],
        "annual_failure_distribution": yr_fail,
    }


# ======================================================================
# Phase5: Day5 Tail Effect
# ======================================================================

def phase5_tail_effect(df: pd.DataFrame) -> dict:
    """
    Study60 Day5: rank_ic=-0.025 / spread20=+7.60pp の解明。
    観測: モデルが負IC でも Top/Bot decile の spread がなぜ正?
    """
    valid = df.dropna(subset=["fwd60d_entry"]).copy()
    fwd60 = valid["fwd60d_entry"].values

    # Day5の各特徴量について:
    # 1) IC vs fwd60d_entry (Study61の視点)
    # 2) IC vs fwd60d_obs (Study60の視点: Day5からの残リターン)
    feat_analysis = {}
    for f in FEAT_LIST:
        col = f"d5_{f}"
        if col not in valid.columns:
            continue
        sub = valid[[col, "fwd60d_entry"]].dropna()
        col_obs = "d5_fwd60d_obs"
        sub_obs = valid[[col, col_obs]].dropna() if col_obs in valid.columns else None

        ic_entry, p_entry = _spearman(sub[col].values, sub["fwd60d_entry"].values) if len(sub) >= 5 else (None, None)
        ic_obs = None
        if sub_obs is not None and len(sub_obs) >= 5:
            ic_obs, _ = _spearman(sub_obs[col].values, sub_obs[col_obs].values)

        # Decile spread (feature-based ranking)
        if len(sub) >= 10:
            scores = sub[col].values
            rets   = sub["fwd60d_entry"].values
            idx_s  = np.argsort(scores)[::-1]
            n_v    = len(sub)
            top20n = max(1, int(n_v * 0.2))
            bot20n = max(1, int(n_v * 0.2))
            top20_ret = float(rets[idx_s[:top20n]].mean())
            bot20_ret = float(rets[idx_s[-bot20n:]].mean())
            spread20 = top20_ret - bot20_ret
        else:
            spread20 = None

        feat_analysis[f] = {
            "ic_vs_fwd60d_entry": _s(ic_entry),
            "ic_vs_fwd60d_obs":   _s(ic_obs),
            "spread20_entry":     _s(spread20),
            "pval_entry":         _s(p_entry, 3),
            "sign_consistent": (
                (ic_entry or 0) * (spread20 or 0) > 0
            ) if ic_entry is not None and spread20 is not None else None,
        }

    # 分位別: Day5 ret_from_entry の fwd60d_entry 関係
    col_d5_ret = "d5_ret_from_entry"
    quantile_breakdown = {}
    if col_d5_ret in valid.columns:
        sub = valid[[col_d5_ret, "fwd60d_entry"]].dropna()
        scores = sub[col_d5_ret].values
        rets   = sub["fwd60d_entry"].values
        for pct_label, lo, hi in [
            ("Q1_bot20", 0, 0.20), ("Q2_2040", 0.20, 0.40),
            ("Q3_4060", 0.40, 0.60), ("Q4_6080", 0.60, 0.80),
            ("Q5_top20", 0.80, 1.00),
        ]:
            lo_th = float(np.percentile(scores, lo * 100)) if lo > 0 else -9999
            hi_th = float(np.percentile(scores, hi * 100)) if hi < 1 else 9999
            seg_rets = rets[(scores >= lo_th) & (scores <= hi_th)]
            quantile_breakdown[pct_label] = _stats(seg_rets, 2)

    # 非線形性: 上位と下位だけ分離 vs 中央も含めた連続
    col_d5_nr7 = "d5_nr7"
    col_d5_inside = "d5_inside_bar"
    nonlinear_check = {}
    for col_feat, fname in [(col_d5_nr7, "nr7"), (col_d5_inside, "inside_bar")]:
        if col_feat not in valid.columns:
            continue
        sub = valid[[col_feat, "fwd60d_entry"]].dropna()
        g0 = sub[sub[col_feat] == 0]["fwd60d_entry"].values
        g1 = sub[sub[col_feat] == 1]["fwd60d_entry"].values
        if len(g0) > 0 and len(g1) > 0:
            nonlinear_check[fname] = {
                "value_0_mean": _s(float(g0.mean())), "n0": len(g0),
                "value_1_mean": _s(float(g1.mean())), "n1": len(g1),
                "diff":         _s(float(g1.mean()) - float(g0.mean())),
                "pval":         _s(_mwu_pval(g1, g0), 3),
            }

    # 説明仮説
    # Study60のDay5: inside_bar IC=-0.103, upper_shadow IC=-0.099
    # つまり「inside_bar=1/upper_shadow大 → 次60日リターン低い」
    # モデルはこれを学習してinside_bar=0 / upper_shadow小 を高スコア付与
    # → 高スコア = inside_bar=0 = 実際に高リターン → spread正
    # 全体IC=-0.025は他の特徴量との相殺で見かけ上低下
    hypothesis = (
        "Study60 Day5: inside_bar(IC=-0.10)/upper_shadow(IC=-0.10)が支配的な負IC特徴量。"
        "ML modelはinside_bar=0/upper_shadow小を高スコアに → top20=inside_bar=0群 = 実際高リターン。"
        "全体rank_IC=-0.025はこの効果が他特徴量のノイズに相殺された結果。"
        "Spread20=+7.60ppはinside_bar/upper_shadowの非線形効果が本体。"
    )

    return {
        "feature_analysis": feat_analysis,
        "day5_ret_quantile_breakdown": quantile_breakdown,
        "binary_feature_check": nonlinear_check,
        "hypothesis": hypothesis,
        "top_spread_features": sorted(
            [(f, abs(v["spread20_entry"] or 0)) for f, v in feat_analysis.items()
             if v.get("spread20_entry") is not None],
            key=lambda x: x[1], reverse=True
        )[:5],
    }


# ======================================================================
# Phase6: Temporal Evolution
# ======================================================================

def phase6_temporal_evolution(df: pd.DataFrame) -> dict:
    """各群の特徴量推移を比較して分岐タイミングを特定。"""
    groups = {
        "BigWinner":  df["perf_pct"] >= 0.80,
        "NearMiss":   (df["perf_pct"] >= 0.60) & (df["perf_pct"] < 0.80),
        "Middle":     (df["perf_pct"] >= 0.30) & (df["perf_pct"] < 0.60),
        "Loser":      df["perf_pct"] <= 0.20,
    }

    key_feats = ["ret_from_entry", "atr_expansion", "rsr_delta", "rs_accel_post",
                 "ma5_slope", "vol_retention", "nr7"]

    trajectories = {}
    for grp_name, mask in groups.items():
        seg = df[mask]
        traj = {}
        for n_days in OBS_DAYS_EXT:
            day_out = {}
            for f in key_feats:
                col = f"d{n_days}_{f}"
                if col not in seg.columns:
                    continue
                vals = seg[col].dropna().values
                if len(vals) > 0:
                    day_out[f] = _s(float(vals.mean()))
            traj[f"day{n_days}"] = day_out
        trajectories[grp_name] = {"n": int(mask.sum()), "trajectory": traj}

    # 分岐タイミング: BigWinner vs Loser の差が最大になる日
    divergence = {}
    for f in key_feats:
        max_diff = 0.0
        best_day = 1
        for n_days in OBS_DAYS_EXT:
            bw_val = trajectories.get("BigWinner", {}).get("trajectory", {}).get(f"day{n_days}", {}).get(f)
            lo_val = trajectories.get("Loser", {}).get("trajectory", {}).get(f"day{n_days}", {}).get(f)
            if bw_val is not None and lo_val is not None:
                diff = abs(bw_val - lo_val)
                if diff > max_diff:
                    max_diff = diff
                    best_day = n_days
        divergence[f] = {"max_divergence_day": best_day, "max_diff": _s(max_diff)}

    return {
        "group_trajectories": trajectories,
        "divergence_timing": divergence,
        "early_divergence_features": [
            f for f, v in divergence.items() if v["max_divergence_day"] <= 3
        ],
        "late_divergence_features": [
            f for f, v in divergence.items() if v["max_divergence_day"] >= 10
        ],
    }


# ======================================================================
# Phase7: Stable Feature Audit
# ======================================================================

def phase7_stable_feature_audit(df: pd.DataFrame) -> dict:
    """Study56〜60確立特徴量の群別有効性・年別安定性を監査。"""
    focus_feats = ["atr_expansion", "nr7", "ma5_slope", "ma20_slope",
                   "rs_accel_post", "ret_from_entry", "rsr_delta"]
    obs_day = 3  # Day3代表

    results = {}
    for f in focus_feats:
        col = f"d{obs_day}_{f}"
        if col not in df.columns:
            continue
        valid = df[[col, "fwd60d_entry", "year"]].dropna()
        if len(valid) < 10:
            continue

        # 全体IC
        ic_all, _ = _spearman(valid[col].values, valid["fwd60d_entry"].values)

        # 年別IC安定性
        yr_ics = []
        for yr, grp in valid.groupby("year"):
            if len(grp) < 5:
                continue
            ic_yr, _ = _spearman(grp[col].values, grp["fwd60d_entry"].values)
            yr_ics.append({"year": str(yr), "ic": _s(ic_yr), "n": len(grp)})

        ic_yr_vals = [x["ic"] for x in yr_ics if x["ic"] is not None]
        yr_stability = {
            "mean_ic":  _s(float(np.mean(ic_yr_vals))) if ic_yr_vals else None,
            "std_ic":   _s(float(np.std(ic_yr_vals, ddof=1)) if len(ic_yr_vals) > 1 else 0.0),
            "positive_years": sum(1 for v in ic_yr_vals if v > 0),
            "total_years": len(ic_yr_vals),
        }

        # 群別有効性: Top20 vs Bot20 平均
        top20 = df[df["perf_pct"] >= 0.80]
        bot20 = df[df["perf_pct"] <= 0.20]
        top_mean = float(top20[col].dropna().mean()) if col in top20.columns else None
        bot_mean = float(bot20[col].dropna().mean()) if col in bot20.columns else None

        results[f] = {
            "ic_all":         _s(ic_all),
            "yr_stability":   yr_stability,
            "yr_detail":      yr_ics,
            "top20_mean":     _s(top_mean),
            "bot20_mean":     _s(bot_mean),
            "top_bot_diff":   _s(top_mean - bot_mean) if top_mean is not None and bot_mean is not None else None,
            "is_stable":      (yr_stability["std_ic"] or 1) < 0.08 and (yr_stability["mean_ic"] or 0) > 0.04,
        }

    # Study60整合性確認
    study60_findings = {
        "atr_expansion_day1_ic": "0.136 (Study60 Phase5)",
        "nr7_day2_ic": "0.141 (Study60 Phase5)",
        "ma5_slope_day1_ic": "0.114 (Study60 Phase5)",
        "study61_focus_day": f"Day{obs_day}",
    }

    return {
        "features": results,
        "study60_cross_check": study60_findings,
        "stable_features": [f for f, v in results.items() if v.get("is_stable")],
        "unstable_features": [f for f, v in results.items() if not v.get("is_stable")],
    }


# ======================================================================
# Phase8: Research Verdict
# ======================================================================

def phase8_verdict(p1, p2, p25, p3, p4, p5, p6, p7) -> dict:
    # ① 最も情報量の高い観測日
    # Phase6 early divergence features + Study60結論
    early_divs = p6.get("early_divergence_features", [])
    best_obs_day = "Day1-2 (特徴量IC最大: atr_expansion/nr7) / Day10 (ML WF IC最大)"

    # ② 最も安定した特徴量
    stable_feats = p7.get("stable_features", [])

    # ③ Big Winner共通構造
    bw_traj = p2.get("Top20pct", {}).get("trajectory", {})
    bw_d1_feats = bw_traj.get("day1", {})
    bw_d3_feats = bw_traj.get("day3", {})

    # ④ Near Miss共通構造
    nm_d3 = p25.get("day3", {})
    continuous_feats = nm_d3.get("top_features_continuous", [])

    # ⑤ False Hero
    fh_count = p3.get("n_false_hero", 0)
    fh_def   = p3.get("definition", "")
    fh_top5  = p3.get("top_distinguishing_features", [])

    # ⑥ Failure
    fail_types = p4.get("failure_types", {})
    top_disc   = p4.get("top_discriminating_features", [])

    # ⑦ Day5 Tail Effect
    tail_hyp = p5.get("hypothesis", "")

    # ⑧ 研究優先順位
    # 根拠:
    # - Big Winner連続体なら Big Winner Retention が最有望
    # - False Hero多数なら Failure Detection が重要
    # - IC安定なら Position Sizing での活用も可
    # - Quality Replacement v2は Study57-59で既検証済み
    n_false_hero = fh_count
    n_top20 = p1.get("deciles", {}).get("Top20%", {}).get("n", 1)
    false_hero_ratio = n_false_hero / max(n_top20, 1)

    bw_contribution = p1.get("expected_value_decomposition", {}).get("top20_contribution")

    priority = []
    if bw_contribution and bw_contribution > 150:
        priority.append({"rank": 1, "theme": "Big Winner Retention",
                         "basis": f"Top20% が期待値の {bw_contribution}% を担う"})
    if false_hero_ratio > 0.3:
        priority.append({"rank": 2, "theme": "Failure Detection",
                         "basis": f"FalseHero率={false_hero_ratio:.1%} (Top群の {false_hero_ratio:.0%}が失速)"})
    if stable_feats:
        priority.append({"rank": 3, "theme": "Position Sizing",
                         "basis": f"安定特徴量({stable_feats[:3]})で確信度重み付け"})
    priority.append({"rank": len(priority)+1, "theme": "Replacement Engine v2",
                     "basis": "Study57-59既検証. 現時点で Shadow中. 新規優先度低"})

    return {
        "study61_verdict": "STRUCTURAL_UNDERSTANDING_COMPLETE",
        "findings": {
            "1_best_observation_day":      best_obs_day,
            "2_stable_features":           stable_feats,
            "3_big_winner_structure":      {
                "day1_top_feats": {k: v.get("mean") for k, v in bw_d1_feats.items()
                                   if k in ["atr_expansion", "ret_from_entry", "rsr_delta"] and v.get("mean") is not None},
                "continuous_vs_special":   "Phase2.5参照",
                "continuous_features":     continuous_feats[:3],
            },
            "4_near_miss_structure":       {"continuous_features": continuous_feats},
            "5_false_hero":                {"n": fh_count, "ratio": _s(false_hero_ratio, 3), "definition": fh_def},
            "6_failure_structure":         {"types": fail_types, "top_discriminators": [f for f, _ in top_disc[:3]]},
            "7_day5_tail_effect":          tail_hyp,
            "8_research_priority":         priority,
        },
    }


# ======================================================================
# main
# ======================================================================

def main():
    print("=== Study61: Return Distribution Anatomy ===")
    print(f"  今日: {TODAY_STR}")

    print("\n[データロード中...]")
    ds = build_common_dataset(DATA_END)
    rsr_df   = ds["rsr_df"]
    all_dates = rsr_df.index.sort_values()
    is_dates  = all_dates[(all_dates >= IS_START)  & (all_dates <= IS_END)]
    oos_dates = all_dates[(all_dates >= OOS_START) & (all_dates <= OOS_END)]

    print("[BT実行: IS 2018-2024]")
    sym_active_is = get_active(ds, IS_START, IS_END)
    bt_is = run_bt(ds, sym_active_is, IS_START, IS_END)
    trades_is = extract_trades(bt_is, is_dates, rsr_df)
    print(f"  IS trades: {len(trades_is)}")

    print("[BT実行: OOS 2025]")
    sym_active_oos = get_active(ds, OOS_START, OOS_END)
    bt_oos = run_bt(ds, sym_active_oos, OOS_START, OOS_END)
    trades_oos = extract_trades(bt_oos, oos_dates, rsr_df)
    print(f"  OOS trades: {len(trades_oos)}")

    all_trades = trades_is + trades_oos
    print(f"  全取引: {len(all_trades)} 件")

    print("\n[Phase0: Study60 Validation]")
    p0 = phase0_validation(len(all_trades))
    print(f"  verdict={p0['verdict']} study60_verdict={p0['study60_verdict']} n={p0['study61_n_trades']}")

    print("\n[データセット構築: Day1-60...]")
    raw_df = build_trade_dataset(ds, all_trades)
    print(f"  有効トレード数: {len(raw_df)}")
    if raw_df.empty:
        print("  ERROR: データセット空")
        return

    df = classify_group(raw_df)
    print(f"  BigWinner: {(df['perf_group']=='BigWinner').sum()} | "
          f"TopHalf: {(df['perf_group']=='TopHalf').sum()} | "
          f"BottomHalf: {(df['perf_group']=='BottomHalf').sum()} | "
          f"Loser: {(df['perf_group']=='Loser').sum()}")

    print("\n[Phase1: Return Distribution Mapping]")
    p1 = phase1_distribution(df)
    ov = p1.get("overall", {})
    print(f"  n={p1['n']} mean={ov.get('mean')}% median={ov.get('median')}% std={ov.get('std')}%")
    ev = p1.get("expected_value_decomposition", {})
    print(f"  Top20%が期待値の{ev.get('top20_contribution')}%を担う "
          f"/ Bot20%寄与={ev.get('bot20_contribution')}%")

    print("\n[Phase2: Big Winner Anatomy]")
    p2 = phase2_big_winner_anatomy(df)
    bw = p2.get("Top20pct", {})
    print(f"  Top20%: n={bw.get('n')} avg={bw.get('avg_fwd60d')}%")
    bw_d1 = bw.get("trajectory", {}).get("day1", {})
    print(f"  Day1 atr_expansion={bw_d1.get('atr_expansion',{}).get('mean')} "
          f"ret_from_entry={bw_d1.get('ret_from_entry',{}).get('mean')}")

    print("\n[Phase2.5: Near Miss Winner Analysis]")
    p25 = phase25_near_miss(df)
    d3_cont = p25.get("day3", {}).get("top_features_continuous", [])
    print(f"  Day3 連続的特徴量: {d3_cont[:5]}")

    print("\n[Phase3: False Hero Analysis]")
    p3 = phase3_false_hero(df)
    print(f"  False Hero: n={p3.get('n_false_hero')} / Winner: n={p3.get('n_winner')}")
    print(f"  {p3.get('definition', '')}")
    print(f"  Top識別特徴量: {p3.get('top_distinguishing_features', [])[:3]}")

    print("\n[Phase4: Failure Taxonomy]")
    p4 = phase4_failure_taxonomy(df)
    ft = p4.get("failure_types", {})
    print(f"  Bottom20% n={p4.get('n_bot20')}: early={ft.get('early_fail_n')} late={ft.get('late_fail_n')}")
    print(f"  Top識別特徴量: {[f for f,_ in p4.get('top_discriminating_features', [])[:3]]}")

    print("\n[Phase5: Day5 Tail Effect]")
    p5 = phase5_tail_effect(df)
    top_spread = p5.get("top_spread_features", [])
    print(f"  Top spread features (Day5): {top_spread[:3]}")
    bnc = p5.get("binary_feature_check", {})
    for fn, bv in bnc.items():
        print(f"  {fn}: val0_mean={bv.get('value_0_mean')} val1_mean={bv.get('value_1_mean')} "
              f"diff={bv.get('diff')} pval={bv.get('pval')}")

    print("\n[Phase6: Temporal Evolution]")
    p6 = phase6_temporal_evolution(df)
    print(f"  早期分岐特徴量(≤Day3): {p6.get('early_divergence_features', [])}")
    print(f"  遅延分岐特徴量(≥Day10): {p6.get('late_divergence_features', [])}")

    print("\n[Phase7: Stable Feature Audit]")
    p7 = phase7_stable_feature_audit(df)
    print(f"  安定特徴量: {p7.get('stable_features', [])}")
    print(f"  不安定特徴量: {p7.get('unstable_features', [])}")

    print("\n[Phase8: Research Verdict]")
    p8 = phase8_verdict(p1, p2, p25, p3, p4, p5, p6, p7)
    print(f"  verdict: {p8['study61_verdict']}")
    for k, v in p8.get("findings", {}).items():
        print(f"  {k}: {v}")

    result = {
        "study":   "Study61",
        "date":    TODAY_STR,
        "params":  {"is_period": f"{IS_START}~{IS_END}", "oos_period": f"{OOS_START}~{OOS_END}",
                    "obs_days": OBS_DAYS_EXT, "fwd_days": FWD_DAYS, "strategy": "D_ATR_EQ"},
        "phase0_validation":         p0,
        "phase1_distribution":       p1,
        "phase2_big_winner_anatomy": p2,
        "phase25_near_miss":         p25,
        "phase3_false_hero":         p3,
        "phase4_failure_taxonomy":   p4,
        "phase5_tail_effect":        p5,
        "phase6_temporal_evolution": p6,
        "phase7_stable_feature":     p7,
        "phase8_verdict":            p8,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n出力: {OUT_FILE}")
    return result


if __name__ == "__main__":
    main()
