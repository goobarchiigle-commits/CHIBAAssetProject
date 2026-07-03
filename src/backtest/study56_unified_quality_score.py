"""
study56_unified_quality_score.py
Unified Quality Score Research

Phase1: Quality Score Construction
  - Top features from Study55: atr_expansion/ret_from_entry/vol_retention/rsr_delta
  - IC, Spearman, Decile Spread, Stability per feature
  - VIF / mutual correlation
  - Weight determination → Quality Score v1 formula

Phase2: Quality Score Validation (全EXECUTED trades)
  - Decile: Top10/20/30% / Bottom10/20/30%
  - fwd_rem, winrate, PF, holding_days, runup
  - Lift / Spread / Monotonicity

Phase3: MAX_POS Attribution
  - 423件候補 vs 実際保有銘柄 Quality Score 比較
  - Selection Edge: Score-based swap precision vs oracle

Phase4: Exit Readiness Assessment
  - Day3/5/10 Score trajectory: Winner vs Loser divergence
  - Value: HIGH / MEDIUM / LOW

禁止: 新ルール実装 / バックテスト変更 / 閾値最適化 / 過剰最適化 / WF情報漏洩
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
from src.config_loader import load_strategy_config

TODAY_STR   = date.today().strftime("%Y-%m-%d")
CAPITAL     = 3_000_000
IS_START    = "2018-01-01"
IS_END      = "2024-12-31"
DATA_END    = "2025-12-31"
MIN_HOLD    = 3
MIN_RSR     = 75.0

EP_EXIT         = "A"
EP_ADDON        = "D"
ADDON_ATR_MULT  = 1.0
ADDON_SIZE_FRAC = 0.25

OBS_DAYS = [3, 5, 10]

# Study55確認済みIC (残リターンとのSpearman、Day3/5/10平均)
STUDY55_IC = {
    "atr_expansion":  0.115,
    "ret_from_entry": 0.098,
    "vol_retention":  0.096,
    "rsr_delta":      0.072,
    "ma20_dev":       0.064,
}

# Quality Score v1 に使う特徴量セット
SCORE_FEATURES = ["atr_expansion", "ret_from_entry", "vol_retention", "rsr_delta"]

OUT_FILE = ROOT / "backtests" / f"study56_unified_quality_score_{TODAY_STR}.json"


# ================================================================== #
#  ユーティリティ
# ================================================================== #

def _safe(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), d)


def _atr20(df_c: pd.DataFrame) -> pd.Series:
    h = df_c["High"]; l = df_c["Low"]; c = df_c["Close"]
    cp = c.shift(1).fillna(c)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.rolling(20, min_periods=10).mean()


def _spearman(x, y) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        rx = pd.Series(x).rank().values
        ry = pd.Series(y).rank().values
        r = float(np.corrcoef(rx, ry)[0, 1])
        return r, 0.05


def _summary(vals: list) -> dict:
    if not vals:
        return {"n": 0}
    a = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if len(a) == 0:
        return {"n": 0}
    return {
        "n":          int(len(a)),
        "mean":       _safe(float(a.mean())),
        "median":     _safe(float(np.median(a))),
        "std":        _safe(float(a.std(ddof=1))),
        "win_rate":   _safe(float((a > 0).mean() * 100), 1),
        "p10":        _safe(float(np.percentile(a, 10))),
        "p25":        _safe(float(np.percentile(a, 25))),
        "p75":        _safe(float(np.percentile(a, 75))),
        "p90":        _safe(float(np.percentile(a, 90))),
    }


def _fwd_return(close: pd.Series, ref_date: pd.Timestamp, ref_px: float, n: int):
    fut = close[close.index > ref_date]
    if len(fut) < n:
        return None
    return _safe((float(fut.iloc[n - 1]) / ref_px - 1.0) * 100)


def _maxmin(close: pd.Series, ref_date: pd.Timestamp, ref_px: float, h: int = 60):
    fut = close[close.index > ref_date].iloc[:h]
    if fut.empty:
        return None, None
    return _safe((float(fut.max()) / ref_px - 1) * 100), _safe((float(fut.min()) / ref_px - 1) * 100)


# ================================================================== #
#  BT 実行
# ================================================================== #

def get_active(ds: dict, start: str, end: str) -> pd.DataFrame:
    cfg = load_strategy_config()
    bc  = cfg.risk_controls.bear_universe_filter
    be  = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=list(ds["trade_syms"].keys()),
        start=start, end=end, bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds: dict, sym_active_df, start: str, end: str) -> dict:
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


# ================================================================== #
#  特徴量計算 (post-entry)
# ================================================================== #

POST_FEATURES = [
    "atr_expansion", "ret_from_entry", "vol_retention",
    "rsr_delta", "ma20_dev", "breakout_dist", "rs_accel_post",
]


def _features_at_dayN(
    sym: str,
    entry_date: pd.Timestamp,
    entry_idx: int,
    n_days: int,
    is_dates: pd.DatetimeIndex,
    universe_raw: dict,
    rsr_df: pd.DataFrame,
    topix_close: pd.Series | None,
    entry_rsr: float,
) -> dict | None:
    obs_idx = entry_idx + n_days
    if obs_idx >= len(is_dates):
        return None
    obs_date = is_dates[obs_idx]

    if sym not in universe_raw:
        return None
    df_c = universe_raw[sym].get("df")
    if df_c is None or "Close" not in df_c.columns:
        return None

    close = df_c["Close"].dropna()
    close.index = pd.to_datetime(close.index)
    avail = close[close.index <= obs_date]
    if len(avail) < 20:
        return None
    obs_px = float(avail.iloc[-1])
    entry_av = close[close.index <= entry_date]
    if entry_av.empty:
        return None
    entry_close = float(entry_av.iloc[-1])
    if entry_close <= 0:
        return None

    feat: dict = {}
    feat["ret_from_entry"] = _safe((obs_px / entry_close - 1.0) * 100)

    rsr_sym = rsr_df[sym] if sym in rsr_df.columns else pd.Series(dtype=float)
    rsr_avail = rsr_sym[rsr_sym.index <= obs_date].dropna()
    if not rsr_avail.empty:
        rsr_now = float(rsr_avail.iloc[-1])
        feat["rsr_now"] = _safe(rsr_now, 1)
        feat["rsr_delta"] = _safe(rsr_now - entry_rsr)
        if len(rsr_avail) >= 22:
            slope_now = (rsr_now - float(rsr_avail.iloc[-22])) / 21.0
            entry_rsr_av = rsr_sym[rsr_sym.index <= entry_date].dropna()
            if len(entry_rsr_av) >= 22:
                slope_entry = (float(entry_rsr_av.iloc[-1]) - float(entry_rsr_av.iloc[-22])) / 21.0
                feat["rs_accel_post"] = _safe(slope_now - slope_entry, 4)

    if len(avail) >= 20:
        ma20 = float(avail.iloc[-20:].mean())
        feat["ma20_dev"] = _safe((obs_px / ma20 - 1.0) * 100) if ma20 > 0 else None

    if len(avail) >= 21:
        ph20 = float(avail.iloc[-21:-1].max())
        feat["breakout_dist"] = _safe((obs_px / ph20 - 1.0) * 100) if ph20 > 0 else None

    vol_col = "Volume" if "Volume" in df_c.columns else ("volume" if "volume" in df_c.columns else None)
    if vol_col:
        vol = pd.to_numeric(df_c[vol_col], errors="coerce").dropna()
        vol.index = pd.to_datetime(vol.index)
        vol_av = vol[vol.index <= obs_date]
        if len(vol_av) >= 20:
            vol_ma20 = float(vol_av.iloc[-20:].mean())
            vol_now  = float(vol_av.iloc[-1])
            feat["vol_retention"] = _safe(vol_now / vol_ma20) if vol_ma20 > 0 else None

    if "High" in df_c.columns and "Low" in df_c.columns:
        atr = _atr20(df_c)
        atr_av_obs   = atr[atr.index <= obs_date].dropna()
        atr_av_entry = atr[atr.index <= entry_date].dropna()
        if not atr_av_obs.empty and not atr_av_entry.empty:
            atr_entry_val = float(atr_av_entry.iloc[-1])
            feat["atr_expansion"] = _safe(float(atr_av_obs.iloc[-1]) / atr_entry_val) \
                if atr_entry_val > 0 else None

    return feat


def _entry_features(
    sym: str,
    sig_date: pd.Timestamp,
    universe_raw: dict,
    rsr_df: pd.DataFrame,
) -> dict:
    """Signal date (Day0) での銘柄品質特徴量。MAX_POS候補ランキング用。"""
    feat: dict = {}
    if sym not in universe_raw:
        return feat
    df_c = universe_raw[sym].get("df")
    if df_c is None or "Close" not in df_c.columns:
        return feat

    close = df_c["Close"].dropna()
    close.index = pd.to_datetime(close.index)
    avail = close[close.index <= sig_date]
    if len(avail) < 21:
        return feat
    px = float(avail.iloc[-1])
    if px <= 0:
        return feat

    # RSR at signal
    rsr_sym = rsr_df[sym] if sym in rsr_df.columns else pd.Series(dtype=float)
    rsr_av = rsr_sym[rsr_sym.index <= sig_date].dropna()
    if not rsr_av.empty:
        feat["rsr"] = _safe(float(rsr_av.iloc[-1]), 1)
        if len(rsr_av) >= 22:
            feat["rsr_slope"] = _safe((float(rsr_av.iloc[-1]) - float(rsr_av.iloc[-22])) / 21.0, 4)

    # MA20 deviation
    ma20 = float(avail.iloc[-20:].mean())
    feat["ma20_dev"] = _safe((px / ma20 - 1.0) * 100) if ma20 > 0 else None

    # Breakout distance from 20d prev high (no lookahead: shift)
    ph20 = float(avail.iloc[-21:-1].max())
    feat["breakout_dist"] = _safe((px / ph20 - 1.0) * 100) if ph20 > 0 else None

    # Volume ratio
    vol_col = "Volume" if "Volume" in df_c.columns else ("volume" if "volume" in df_c.columns else None)
    if vol_col:
        vol = pd.to_numeric(df_c[vol_col], errors="coerce").dropna()
        vol.index = pd.to_datetime(vol.index)
        vol_av = vol[vol.index <= sig_date]
        if len(vol_av) >= 20:
            vm20 = float(vol_av.iloc[-20:].mean())
            feat["vol_ratio"] = _safe(float(vol_av.iloc[-1]) / vm20) if vm20 > 0 else None

    # ATR pct of price
    if "High" in df_c.columns and "Low" in df_c.columns:
        atr = _atr20(df_c)
        atr_av = atr[atr.index <= sig_date].dropna()
        if not atr_av.empty:
            feat["atr_pct"] = _safe(float(atr_av.iloc[-1]) / px * 100) if px > 0 else None

    return feat


# ================================================================== #
#  Quality Score 計算
# ================================================================== #

def _compute_quality_score(feat_df: pd.DataFrame, features: list[str], weights: dict[str, float]) -> pd.Series:
    """
    各特徴量をZ-score化し重み付き合成。スコアをpercentiileランク(0-100)に変換。
    """
    z_scores = pd.DataFrame(index=feat_df.index)
    for f in features:
        if f not in feat_df.columns:
            continue
        col = feat_df[f].dropna()
        mu = col.mean()
        sigma = col.std(ddof=1)
        if sigma > 0:
            z_scores[f] = (feat_df[f] - mu) / sigma
        else:
            z_scores[f] = 0.0

    raw_score = pd.Series(0.0, index=feat_df.index)
    total_w = 0.0
    for f, w in weights.items():
        if f in z_scores.columns:
            raw_score += w * z_scores[f].fillna(0.0)
            total_w += w
    if total_w > 0:
        raw_score /= total_w

    # Percentile rank 0-100
    pct_score = raw_score.rank(pct=True) * 100
    return pct_score


def _ic_weighted_weights(features: list[str]) -> dict[str, float]:
    total = sum(STUDY55_IC.get(f, 0) for f in features)
    if total <= 0:
        return {f: 1.0 for f in features}
    return {f: STUDY55_IC.get(f, 0) / total for f in features}


def _vif(df: pd.DataFrame) -> dict[str, float]:
    """VIF (Variance Inflation Factor) 各特徴量について計算。"""
    result = {}
    cols = [c for c in df.columns if df[c].notna().sum() > 5]
    if len(cols) < 2:
        return result
    for target in cols:
        others = [c for c in cols if c != target]
        mat = df[cols].dropna()
        if mat.empty:
            continue
        X = mat[others].values
        y = mat[target].values
        try:
            from numpy.linalg import lstsq
            coef, _, _, _ = lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
            yhat = np.column_stack([np.ones(len(X)), X]) @ coef
            ss_res = ((y - yhat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
            result[target] = _safe(vif)
        except Exception:
            result[target] = None
    return result


# ================================================================== #
#  Phase1: Quality Score Construction
# ================================================================== #

def phase1_score_construction(feat_records_by_day: dict[str, list[dict]]) -> dict:
    """
    全Day特徴量レコードを統合してIC/VIF/相関/安定性を計算し
    Quality Score v1 の定義を出力する。
    """
    # 全 Day の feat_df を結合
    all_records = []
    for day_key, recs in feat_records_by_day.items():
        for r in recs:
            row = dict(r)
            row["_day"] = int(day_key.replace("day", ""))
            all_records.append(row)

    if not all_records:
        return {"error": "no records"}

    full_df = pd.DataFrame(all_records)

    # --- IC (Spearman) 各特徴量 vs remaining_return, by day ---
    ic_by_day: dict[str, dict] = {}
    ic_stability: dict[str, list] = defaultdict(list)

    for day_key, recs in feat_records_by_day.items():
        if not recs:
            continue
        df_d = pd.DataFrame(recs)
        ic_day = {}
        for f in POST_FEATURES:
            if f not in df_d.columns:
                continue
            valid = df_d[["rem_ret", f]].dropna()
            if len(valid) < 5:
                continue
            ic, pval = _spearman(valid["rem_ret"].values, valid[f].values)
            ic_day[f] = {"ic": _safe(ic), "n": len(valid), "pval": _safe(pval)}
            ic_stability[f].append(abs(ic))
        ic_by_day[day_key] = ic_day

    # IC stability (std across days, lower = more stable)
    ic_stability_summary = {}
    for f, vals in ic_stability.items():
        ic_stability_summary[f] = {
            "mean_abs_ic": _safe(np.mean(vals)),
            "std_abs_ic":  _safe(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            "n_days":      len(vals),
        }

    # --- Mutual correlation (Spearman) ---
    score_df = full_df[SCORE_FEATURES].dropna()
    corr_matrix = {}
    for f1 in SCORE_FEATURES:
        corr_matrix[f1] = {}
        for f2 in SCORE_FEATURES:
            if f1 == f2:
                corr_matrix[f1][f2] = 1.0
                continue
            valid = full_df[[f1, f2]].dropna()
            if len(valid) < 5:
                corr_matrix[f1][f2] = None
                continue
            r, _ = _spearman(valid[f1].values, valid[f2].values)
            corr_matrix[f1][f2] = _safe(r)

    # --- VIF ---
    vif_result = _vif(full_df[SCORE_FEATURES])

    # --- Decile Spread (per feature, vs rem_ret) ---
    decile_spread = {}
    for f in SCORE_FEATURES:
        valid = full_df[["rem_ret", f]].dropna()
        if len(valid) < 20:
            continue
        top30 = valid[valid[f] >= valid[f].quantile(0.70)]["rem_ret"].mean()
        bot30 = valid[valid[f] <= valid[f].quantile(0.30)]["rem_ret"].mean()
        decile_spread[f] = {
            "top30_rem_ret": _safe(top30),
            "bot30_rem_ret": _safe(bot30),
            "spread":        _safe(top30 - bot30),
        }

    # --- IC-weighted quality score weights ---
    weights = _ic_weighted_weights(SCORE_FEATURES)

    # --- Feature ranking by IC stability and magnitude ---
    feature_rank = sorted(
        ic_stability_summary.items(),
        key=lambda x: x[1]["mean_abs_ic"] or 0,
        reverse=True,
    )

    return {
        "n_total_records": len(all_records),
        "features_analyzed": SCORE_FEATURES,
        "ic_by_day": ic_by_day,
        "ic_stability": ic_stability_summary,
        "mutual_correlation": corr_matrix,
        "vif": vif_result,
        "decile_spread": decile_spread,
        "quality_score_v1": {
            "formula": "IC-weighted Z-score combination → percentile rank 0-100",
            "features": SCORE_FEATURES,
            "weights": {f: _safe(w, 4) for f, w in weights.items()},
            "weight_basis": "Study55 avg IC vs remaining return",
        },
        "feature_ranking": [
            {"rank": i + 1, "feature": f, "mean_abs_ic": s["mean_abs_ic"],
             "std_abs_ic": s["std_abs_ic"], "vif": vif_result.get(f)}
            for i, (f, s) in enumerate(feature_rank)
            if f in SCORE_FEATURES
        ],
    }


# ================================================================== #
#  Phase2: Quality Score Validation
# ================================================================== #

def phase2_score_validation(
    feat_records_by_day: dict[str, list[dict]],
    all_trades_df: pd.DataFrame,
) -> dict:
    """
    Day3特徴量で Quality Score を算出し Decile分析 / Lift / Monotonicity を出力。
    全309 (EXECUTED) トレードのうち Day3 観測可能なものが対象。
    """
    weights = _ic_weighted_weights(SCORE_FEATURES)

    results_by_day: dict = {}
    for day_key, recs in feat_records_by_day.items():
        if not recs:
            results_by_day[day_key] = {"n": 0}
            continue
        df_d = pd.DataFrame(recs)

        # Quality Score計算
        score = _compute_quality_score(df_d, SCORE_FEATURES, weights)
        df_d["quality_score"] = score
        df_d = df_d.dropna(subset=["quality_score", "rem_ret"])
        n = len(df_d)
        if n < 10:
            results_by_day[day_key] = {"n": n}
            continue

        # --- Decile分析 ---
        decile_res = {}
        for label, lo_pct, hi_pct in [
            ("Top10%", 90, 100), ("Top20%", 80, 100), ("Top30%", 70, 100),
            ("Mid40%", 30, 70),
            ("Bot30%", 0, 30), ("Bot20%", 0, 20), ("Bot10%", 0, 10),
        ]:
            lo_thr = float(df_d["quality_score"].quantile(lo_pct / 100)) if lo_pct > 0 else -9999
            hi_thr = float(df_d["quality_score"].quantile(hi_pct / 100)) if hi_pct < 100 else 9999
            seg = df_d[(df_d["quality_score"] >= lo_thr) & (df_d["quality_score"] <= hi_thr)]
            if seg.empty:
                continue
            rem_vals = seg["rem_ret"].dropna().tolist()
            tot_vals = seg["total_ret"].dropna().tolist()
            pf = None
            pos_mean = np.mean([v for v in rem_vals if v > 0]) if any(v > 0 for v in rem_vals) else 0
            neg_mean = abs(np.mean([v for v in rem_vals if v < 0])) if any(v < 0 for v in rem_vals) else None
            if neg_mean and neg_mean > 0:
                pf = _safe(pos_mean / neg_mean)
            hold_vals = seg["hold_days"].dropna().tolist() if "hold_days" in seg.columns else []
            decile_res[label] = {
                "n":             len(seg),
                "score_mean":    _safe(float(seg["quality_score"].mean()), 1),
                "rem_ret":       _summary(rem_vals),
                "total_ret_mean": _safe(float(np.mean(tot_vals))) if tot_vals else None,
                "winrate_rem":   _safe(float((np.array(rem_vals) > 0).mean() * 100), 1) if rem_vals else None,
                "PF_rem":        pf,
                "hold_days_mean":_safe(float(np.mean(hold_vals))) if hold_vals else None,
            }

        # --- Lift vs overall ---
        overall_mean = float(df_d["rem_ret"].mean())
        top10_mean = decile_res.get("Top10%", {}).get("rem_ret", {}).get("mean")
        top20_mean = decile_res.get("Top20%", {}).get("rem_ret", {}).get("mean")
        lift_10 = _safe(top10_mean / overall_mean) if top10_mean and overall_mean != 0 else None
        lift_20 = _safe(top20_mean / overall_mean) if top20_mean and overall_mean != 0 else None

        # --- Spread (top30% - bot30%) ---
        top30_mean = decile_res.get("Top30%", {}).get("rem_ret", {}).get("mean")
        bot30_mean = decile_res.get("Bot30%", {}).get("rem_ret", {}).get("mean")
        spread = _safe(top30_mean - bot30_mean) if top30_mean is not None and bot30_mean is not None else None

        # --- Monotonicity check (score decile vs rem_ret mean) ---
        decile_labels = ["Bot10%", "Bot20%", "Bot30%", "Mid40%", "Top30%", "Top20%", "Top10%"]
        decile_means = [decile_res.get(l, {}).get("rem_ret", {}).get("mean") for l in decile_labels]
        decile_means_clean = [(l, v) for l, v in zip(decile_labels, decile_means) if v is not None]
        monotone_pairs = 0
        total_pairs = max(1, len(decile_means_clean) - 1)
        for i in range(len(decile_means_clean) - 1):
            if decile_means_clean[i + 1][1] >= decile_means_clean[i][1]:
                monotone_pairs += 1
        monotonicity = _safe(monotone_pairs / total_pairs * 100, 1)

        # --- IC of Quality Score itself ---
        valid_ic = df_d[["rem_ret", "quality_score"]].dropna()
        ic_score, _ = _spearman(valid_ic["rem_ret"].values, valid_ic["quality_score"].values) if len(valid_ic) >= 5 else (None, None)

        results_by_day[day_key] = {
            "n":              n,
            "overall_mean_rem_ret": _safe(overall_mean),
            "ic_score_vs_rem": _safe(ic_score),
            "lift_top10":     lift_10,
            "lift_top20":     lift_20,
            "spread_top30_bot30": spread,
            "monotonicity_pct":   monotonicity,
            "decile_analysis":    decile_res,
        }

    # --- Cross-day summary ---
    cross_ic = [v.get("ic_score_vs_rem") for v in results_by_day.values() if v.get("ic_score_vs_rem") is not None]
    cross_spread = [v.get("spread_top30_bot30") for v in results_by_day.values() if v.get("spread_top30_bot30") is not None]

    verdict = "PASS" if (cross_ic and np.mean(cross_ic) >= 0.05 and any(s and s > 0 for s in cross_spread)) else "FAIL"

    return {
        "quality_score_v1_weights": {f: _safe(w, 4) for f, w in weights.items()},
        "by_day":          results_by_day,
        "cross_day_summary": {
            "mean_ic_across_days":    _safe(np.mean(cross_ic)) if cross_ic else None,
            "mean_spread_across_days": _safe(np.mean(cross_spread)) if cross_spread else None,
        },
        "validation_verdict": verdict,
        "verdict_basis": "IC≥0.05 AND spread>0 across Day3/5/10",
    }


# ================================================================== #
#  Phase3: MAX_POS Attribution
# ================================================================== #

def phase3_maxpos_attribution(
    missed_cands: list[dict],
    sell_trades: list[dict],
    is_dates: pd.DatetimeIndex,
    universe_raw: dict,
    rsr_df: pd.DataFrame,
    trade_syms: dict,
) -> dict:
    """
    423件の MAX_POS 候補 vs 保有銘柄の Score 比較。
    Selection Edge = Score-based swap decision の正解率（vs oracle delta>0）。
    """
    # --- 保有銘柄 tracker (entry_idx → exit_idx) ---
    position_map: dict[int, list[dict]] = defaultdict(list)
    for t in sell_trades:
        ei = t.get("entry_idx", -1)
        xi = t.get("exit_idx", -1)
        sym = t.get("symbol")
        if ei < 0 or xi < 0 or not sym:
            continue
        entry_rsr = t.get("entry_rsr")
        for i in range(ei, xi):
            position_map[i].append({
                "sym": sym, "entry_idx": ei, "entry_rsr": entry_rsr,
                "entry_px": t.get("entry", 0.0), "exit_px": t.get("exit", 0.0),
            })

    date_to_idx = {str(d.date()): i for i, d in enumerate(is_dates)}
    rsr_is = rsr_df.loc[rsr_df.index.isin(is_dates)]

    records = []
    for cand in missed_cands:
        sym  = cand.get("symbol")
        dstr = cand.get("date")
        if not sym or not dstr:
            continue
        d_idx = date_to_idx.get(dstr)
        if d_idx is None:
            continue

        holdings = position_map.get(d_idx, [])
        seen = set()
        unique_holdings = []
        for h in holdings:
            if h["sym"] not in seen and h["sym"] != sym:
                seen.add(h["sym"])
                unique_holdings.append(h)
        if not unique_holdings:
            continue

        sig_date = pd.Timestamp(dstr)

        # Candidate: oracle fwd60
        if sym not in universe_raw:
            continue
        df_cand = universe_raw[sym].get("df")
        if df_cand is None or "Close" not in df_cand.columns:
            continue
        close_c = df_cand["Close"].dropna()
        close_c.index = pd.to_datetime(close_c.index)
        avail_c = close_c[close_c.index <= sig_date]
        if avail_c.empty:
            continue
        ref_px_c = float(avail_c.iloc[-1])
        cand_fwd60 = _fwd_return(close_c, sig_date, ref_px_c, 60)
        if cand_fwd60 is None:
            continue

        # Candidate: entry quality score features
        cand_entry_feat = _entry_features(sym, sig_date, universe_raw, rsr_is)

        # Holdings: oracle fwd60 + continuation quality
        holding_data = []
        for h in unique_holdings:
            h_sym = h["sym"]
            if h_sym not in universe_raw:
                continue
            df_h = universe_raw[h_sym].get("df")
            if df_h is None or "Close" not in df_h.columns:
                continue
            close_h = df_h["Close"].dropna()
            close_h.index = pd.to_datetime(close_h.index)
            avail_h = close_h[close_h.index <= sig_date]
            if avail_h.empty:
                continue
            ref_px_h = float(avail_h.iloc[-1])
            h_fwd60 = _fwd_return(close_h, sig_date, ref_px_h, 60)
            if h_fwd60 is None:
                continue

            # Post-hold quality (features at time of rejection)
            h_entry_idx = h["entry_idx"]
            days_held = d_idx - h_entry_idx if d_idx > h_entry_idx else 0
            h_entry_date = is_dates[h_entry_idx] if h_entry_idx < len(is_dates) else sig_date
            h_entry_rsr = h.get("entry_rsr") or 0.0
            h_feat = {}
            if days_held > 0:
                h_feat = _features_at_dayN(
                    sym=h_sym, entry_date=h_entry_date,
                    entry_idx=h_entry_idx, n_days=days_held,
                    is_dates=is_dates, universe_raw=universe_raw,
                    rsr_df=rsr_is, topix_close=None,
                    entry_rsr=float(h_entry_rsr),
                ) or {}

            holding_data.append({
                "sym": h_sym, "fwd60": h_fwd60,
                "days_held": days_held, "feat": h_feat,
            })

        if not holding_data:
            continue

        weakest = min(holding_data, key=lambda x: x["fwd60"])

        # --- Score comparison: candidate RSR vs weakest RSR ---
        cand_rsr = cand_entry_feat.get("rsr") or cand.get("rsr") or 0.0
        weak_rsr = weakest["feat"].get("rsr_now") or 0.0

        # --- Oracle delta ---
        delta_fwd60 = _safe(cand_fwd60 - weakest["fwd60"])

        records.append({
            "date":          dstr,
            "cand_sym":      sym,
            "cand_rsr":      _safe(cand_rsr, 1),
            "cand_fwd60":    _safe(cand_fwd60),
            "cand_entry_feat": {k: _safe(v) for k, v in cand_entry_feat.items()},
            "weakest_sym":   weakest["sym"],
            "weakest_rsr_now": _safe(weak_rsr, 1),
            "weakest_fwd60": _safe(weakest["fwd60"]),
            "weakest_days_held": weakest["days_held"],
            "delta_fwd60":   delta_fwd60,
            "oracle_swap":   bool((delta_fwd60 or 0) > 0),
            "score_says_swap": bool(cand_rsr > weak_rsr),
        })

    if not records:
        return {"n_events": 0}

    df_r = pd.DataFrame(records)
    n = len(df_r)

    oracle_swap = df_r["oracle_swap"]
    score_says = df_r["score_says_swap"]

    tp = int(((score_says == True) & (oracle_swap == True)).sum())
    fp = int(((score_says == True) & (oracle_swap == False)).sum())
    fn = int(((score_says == False) & (oracle_swap == True)).sum())
    tn = int(((score_says == False) & (oracle_swap == False)).sum())

    precision = _safe(tp / (tp + fp) * 100) if (tp + fp) > 0 else None
    recall    = _safe(tp / (tp + fn) * 100) if (tp + fn) > 0 else None
    accuracy  = _safe((tp + tn) / n * 100)

    # IC between cand_rsr and delta_fwd60
    valid_ic = df_r[["cand_rsr", "delta_fwd60"]].dropna()
    ic_rsr_delta, _ = _spearman(valid_ic["cand_rsr"].values, valid_ic["delta_fwd60"].values) if len(valid_ic) >= 5 else (None, None)

    # Decile by cand_rsr: does higher RSR → better delta?
    df_sorted = df_r.dropna(subset=["cand_rsr", "delta_fwd60"]).sort_values("cand_rsr")
    n_dec = max(1, len(df_sorted) // 5)
    rsr_delta_by_quintile = []
    for i in range(5):
        seg = df_sorted.iloc[i * n_dec: (i + 1) * n_dec]
        rsr_delta_by_quintile.append({
            "quintile": i + 1,
            "n": len(seg),
            "cand_rsr_mean": _safe(float(seg["cand_rsr"].mean()), 1),
            "delta_mean":    _safe(float(seg["delta_fwd60"].mean())),
            "oracle_swap_rate": _safe(float(seg["oracle_swap"].mean() * 100), 1),
        })

    oracle_swap_rate = _safe(float(oracle_swap.mean() * 100), 1)

    return {
        "n_events":              n,
        "oracle_swap_rate":      oracle_swap_rate,
        "rsr_score_selection": {
            "accuracy_pct":   accuracy,
            "precision_pct":  precision,
            "recall_pct":     recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "ic_rsr_vs_delta": _safe(ic_rsr_delta),
        },
        "rsr_quintile_analysis": rsr_delta_by_quintile,
        "selection_edge_verdict": (
            "HIGH"   if (ic_rsr_delta or 0) >= 0.10 and (accuracy or 0) >= 60 else
            "MEDIUM" if (ic_rsr_delta or 0) >= 0.05 and (accuracy or 0) >= 55 else
            "LOW"
        ),
        "verdict_basis": "IC(RSR vs delta_fwd60) + swap accuracy vs oracle",
        "delta_fwd60_summary": _summary(df_r["delta_fwd60"].dropna().tolist()),
    }


# ================================================================== #
#  Phase4: Exit Readiness Assessment
# ================================================================== #

def phase4_exit_readiness(feat_records_by_day: dict[str, list[dict]]) -> dict:
    """
    Day3/5/10 の Quality Score 軌跡を Winner vs Loser で比較。
    Winner/Loser 乖離が大きければ Quality Exit 研究 → HIGH。
    """
    weights = _ic_weighted_weights(SCORE_FEATURES)

    by_day: dict = {}
    trajectory: dict[str, dict] = {}  # sym+entry → {day: score}

    for day_key, recs in feat_records_by_day.items():
        if not recs:
            by_day[day_key] = {"n": 0}
            continue
        df_d = pd.DataFrame(recs)
        score = _compute_quality_score(df_d, SCORE_FEATURES, weights)
        df_d["quality_score"] = score

        winners = df_d[df_d["label_winner"] == 1]
        losers  = df_d[df_d["label_loser"]  == 1]

        w_scores = winners["quality_score"].dropna().tolist()
        l_scores = losers["quality_score"].dropna().tolist()

        # Cohen's d
        cohens_d = None
        if len(w_scores) > 1 and len(l_scores) > 1:
            pooled_std = np.sqrt((np.var(w_scores, ddof=1) + np.var(l_scores, ddof=1)) / 2)
            if pooled_std > 0:
                cohens_d = _safe((np.mean(w_scores) - np.mean(l_scores)) / pooled_std)

        # IC of quality score vs rem_ret
        valid = df_d[["rem_ret", "quality_score"]].dropna()
        ic, _ = _spearman(valid["rem_ret"].values, valid["quality_score"].values) if len(valid) >= 5 else (None, None)

        # Score spread W vs L
        score_spread = _safe(np.mean(w_scores) - np.mean(l_scores)) if w_scores and l_scores else None

        by_day[day_key] = {
            "n":            len(df_d),
            "n_winners":    len(winners),
            "n_losers":     len(losers),
            "winner_score_mean": _safe(float(np.mean(w_scores))) if w_scores else None,
            "loser_score_mean":  _safe(float(np.mean(l_scores))) if l_scores else None,
            "score_spread_w_l":  score_spread,
            "cohens_d":          cohens_d,
            "ic_score_vs_rem":   _safe(ic),
        }

        # Track trajectory
        for _, row in df_d.iterrows():
            key = f"{row.get('symbol','?')}_{row.get('entry_date','?')}"
            if key not in trajectory:
                trajectory[key] = {"total_ret": row.get("total_ret"), "label_winner": row.get("label_winner")}
            trajectory[key][day_key] = _safe(float(row.get("quality_score", 0)))

    # --- Trajectory: Winner vs Loser score progression ---
    traj_summary: dict = {}
    for day_key in [f"day{n}" for n in OBS_DAYS]:
        w_traj = [v[day_key] for v in trajectory.values() if v.get("label_winner") == 1 and v.get(day_key) is not None]
        l_traj = [v[day_key] for v in trajectory.values() if v.get("label_winner") == 0 and v.get("label_loser") == 1 and v.get(day_key) is not None]
        traj_summary[day_key] = {
            "winner_mean_score": _safe(float(np.mean(w_traj))) if w_traj else None,
            "loser_mean_score":  _safe(float(np.mean(l_traj))) if l_traj else None,
        }

    # --- Overall exit readiness value ---
    ic_vals = [v.get("ic_score_vs_rem") for v in by_day.values() if v.get("ic_score_vs_rem") is not None]
    d_vals  = [v.get("cohens_d") for v in by_day.values() if v.get("cohens_d") is not None]
    mean_ic = float(np.mean(ic_vals)) if ic_vals else 0.0
    mean_d  = float(np.mean(d_vals)) if d_vals else 0.0

    exit_readiness = (
        "HIGH"   if mean_ic >= 0.08 and mean_d >= 0.4 else
        "MEDIUM" if mean_ic >= 0.05 or mean_d >= 0.3 else
        "LOW"
    )

    return {
        "by_day":        by_day,
        "trajectory_summary": traj_summary,
        "cross_day": {
            "mean_ic":      _safe(mean_ic),
            "mean_cohens_d": _safe(mean_d),
        },
        "exit_readiness_value": exit_readiness,
        "verdict_basis": "IC≥0.08 AND Cohen's d≥0.4 → HIGH",
    }


# ================================================================== #
#  共通: feat_records の構築 (全EXECUTED trades at Day N)
# ================================================================== #

def build_feat_records(
    sell_trades: list[dict],
    is_dates: pd.DatetimeIndex,
    ds: dict,
) -> dict[str, list[dict]]:
    universe_raw = ds["universe_raw"]
    rsr_df       = ds["rsr_df"].loc[ds["rsr_df"].index.isin(is_dates)]
    topix_close  = ds.get("topix_close")
    trade_syms   = ds["trade_syms"]

    base = []
    for t in sell_trades:
        ei = t.get("entry_idx", -1)
        if ei < 1 or ei >= len(is_dates):
            continue
        sym      = t["symbol"]
        entry_px = t.get("entry", 0.0)
        exit_px  = t.get("exit", 0.0)
        exit_idx = t.get("exit_idx", -1)
        hold_days = exit_idx - ei if exit_idx >= 0 else -1
        total_ret = (exit_px / entry_px - 1.0) * 100 if entry_px > 0 else None

        entry_date = is_dates[ei]
        sig_date   = is_dates[ei - 1]

        rsr_val = None
        if sym in rsr_df.columns and sig_date in rsr_df.index:
            rsr_val = float(rsr_df.loc[sig_date, sym])

        base.append({
            "symbol": sym, "sector": trade_syms.get(sym, "不明"),
            "entry_idx": ei, "exit_idx": exit_idx,
            "entry_date": entry_date, "sig_date": sig_date,
            "entry_px": entry_px, "exit_px": exit_px,
            "total_ret": total_ret, "hold_days": hold_days,
            "entry_rsr": rsr_val,
        })

    base_df = pd.DataFrame(base)
    if base_df.empty or base_df["total_ret"].dropna().empty:
        return {f"day{n}": [] for n in OBS_DAYS}

    top_thr = float(base_df["total_ret"].quantile(0.75))
    bot_thr = float(base_df["total_ret"].quantile(0.25))

    records_by_day: dict[str, list[dict]] = {}

    for n_days in OBS_DAYS:
        recs = []
        for _, row in base_df.iterrows():
            if row["hold_days"] < n_days or row["total_ret"] is None:
                continue
            if pd.isna(row.get("entry_rsr")):
                continue

            feat = _features_at_dayN(
                sym=row["symbol"], entry_date=row["entry_date"],
                entry_idx=int(row["entry_idx"]), n_days=n_days,
                is_dates=is_dates, universe_raw=universe_raw,
                rsr_df=rsr_df, topix_close=topix_close,
                entry_rsr=float(row["entry_rsr"]),
            )
            if feat is None:
                continue

            # remaining return
            rem_ret = None
            df_c = universe_raw.get(row["symbol"], {}).get("df")
            if df_c is not None and "Close" in df_c.columns:
                close = df_c["Close"].dropna()
                close.index = pd.to_datetime(close.index)
                obs_date = is_dates[int(row["entry_idx"]) + n_days]
                obs_av = close[close.index <= obs_date]
                if not obs_av.empty and row["exit_px"] > 0:
                    obs_px = float(obs_av.iloc[-1])
                    if obs_px > 0:
                        rem_ret = (row["exit_px"] / obs_px - 1.0) * 100

            rec = {
                "symbol":    row["symbol"],
                "entry_date": str(row["entry_date"].date()),
                "total_ret": row["total_ret"],
                "rem_ret":   rem_ret,
                "hold_days": row["hold_days"],
                "label_winner": int(row["total_ret"] >= top_thr),
                "label_loser":  int(row["total_ret"] <= bot_thr),
            }
            rec.update(feat)
            recs.append(rec)
        records_by_day[f"day{n_days}"] = recs

    return records_by_day


# ================================================================== #
#  Final Deliverables サマリー
# ================================================================== #

def final_deliverables(p1: dict, p2: dict, p3: dict, p4: dict) -> dict:
    """
    6項目の最終アウトプット + 研究優先順位推奨。
    """
    weights = p1.get("quality_score_v1", {}).get("weights", {})
    feat_rank = p1.get("feature_ranking", [])
    val_verdict = p2.get("validation_verdict", "?")
    mean_ic_val = p2.get("cross_day_summary", {}).get("mean_ic_across_days")
    mean_spread = p2.get("cross_day_summary", {}).get("mean_spread_across_days")
    sel_edge    = p3.get("selection_edge_verdict", "?")
    sel_acc     = p3.get("rsr_score_selection", {}).get("accuracy_pct")
    exit_val    = p4.get("exit_readiness_value", "?")
    exit_mean_ic = p4.get("cross_day", {}).get("mean_ic")
    exit_mean_d  = p4.get("cross_day", {}).get("mean_cohens_d")

    # Priority recommendation
    # A: Quality Exit WF → 実施判断
    # B: MAX_POS Selection → 実施判断
    # C: Cluster Allocation WF → 既知
    # D: Early Entry → 既存DEFER維持

    qe_priority = "A" if exit_val == "HIGH" else ("B" if exit_val == "MEDIUM" else "C")
    mp_priority = "B" if sel_edge in ("HIGH", "MEDIUM") else "C"

    priority_list = [
        {"priority": "A",
         "research": "Quality Exit WF",
         "value": exit_val,
         "basis": f"IC={exit_mean_ic} / Cohen's_d={exit_mean_d}"},
        {"priority": "B",
         "research": "MAX_POS Selection Score",
         "value": sel_edge,
         "basis": f"accuracy={sel_acc}%"},
        {"priority": "C",
         "research": "Cluster Allocation WF",
         "value": "HIGH",
         "basis": "Study55確認済 alpha_gap=+1.98pp DISTRIBUTED"},
        {"priority": "D",
         "research": "Early Entry",
         "value": "LOW",
         "basis": "Study54確認済 Group A -2.96pp (DEFER維持)"},
    ]

    return {
        "1_quality_score_definition": {
            "formula": "IC-weighted Z-score → percentile rank 0-100",
            "weights": weights,
        },
        "2_feature_importance": [
            {"rank": r["rank"], "feature": r["feature"],
             "mean_abs_ic": r["mean_abs_ic"], "vif": r["vif"]}
            for r in feat_rank
        ],
        "3_decile_analysis": {
            "validation_verdict": val_verdict,
            "mean_ic_cross_day":  mean_ic_val,
            "mean_spread_cross_day": mean_spread,
            "note": "詳細は phase2_validation.by_day 参照",
        },
        "4_maxpos_selection_edge": {
            "verdict":    sel_edge,
            "accuracy_pct": sel_acc,
        },
        "5_quality_exit_value": {
            "verdict": exit_val,
            "mean_ic": exit_mean_ic,
            "mean_cohens_d": exit_mean_d,
        },
        "6_priority_vs_cluster": {
            "note": "Cluster Allocation (Study55確認済)との比較",
            "final_priority_list": priority_list,
        },
        "next_recommended_research": priority_list[0]["research"],
    }


# ================================================================== #
#  メイン
# ================================================================== #

def main() -> None:
    print("=" * 72)
    print("  Study56 Unified Quality Score Research")
    print(f"  Date: {TODAY_STR}   IS: {IS_START}~{IS_END}")
    print("=" * 72)

    print("\n[Data] データセット構築中...")
    ds = build_common_dataset(DATA_END)
    print(f"  {len(ds['trade_syms'])} シンボル")

    print("\n[BT] D_ATR_EQ IS run (Study55同一設定)...")
    active_is = get_active(ds, IS_START, IS_END)
    res = run_bt(ds, active_is, IS_START, IS_END)
    print(f"  CAGR={res['cagr']:+.2f}%  Trades={res['n_trades']}")

    is_dates = ds["rsr_df"].index[
        (ds["rsr_df"].index >= IS_START) & (ds["rsr_df"].index <= IS_END)
    ]

    sell_trades  = [t for t in res.get("_trades", []) if t.get("side") == "SELL"]
    missed_cands = res.get("_missed_cands", [])

    print(f"\n[Info] EXECUTED sell_trades={len(sell_trades)} / MAX_POS candidates={len(missed_cands)}")

    # ── 共通特徴量レコード構築 ────────────────────────────────────────
    print("\n[Feat] 全trades 特徴量計算 (Day3/5/10)...")
    feat_records_by_day = build_feat_records(sell_trades, is_dates, ds)
    for dk, recs in feat_records_by_day.items():
        print(f"  {dk}: n={len(recs)}")

    # ── Phase1 ────────────────────────────────────────────────────────
    print("\n[Phase1] Quality Score Construction...")
    p1 = phase1_score_construction(feat_records_by_day)
    print(f"  VIF: { {k: v for k, v in (p1.get('vif') or {}).items()} }")
    print(f"  Feature ranking: {[r['feature'] for r in p1.get('feature_ranking', [])]}")

    # ── Phase2 ────────────────────────────────────────────────────────
    print("\n[Phase2] Quality Score Validation...")
    p2 = phase2_score_validation(feat_records_by_day, pd.DataFrame())
    print(f"  Verdict: {p2.get('validation_verdict')}")
    for dk, dv in p2.get("by_day", {}).items():
        print(f"  {dk}: IC={dv.get('ic_score_vs_rem')}  spread={dv.get('spread_top30_bot30')}  mono={dv.get('monotonicity_pct')}%")

    # ── Phase3 ────────────────────────────────────────────────────────
    print("\n[Phase3] MAX_POS Attribution...")
    p3 = phase3_maxpos_attribution(
        missed_cands, sell_trades, is_dates,
        ds["universe_raw"],
        ds["rsr_df"].loc[ds["rsr_df"].index.isin(is_dates)],
        ds["trade_syms"],
    )
    print(f"  n_events={p3.get('n_events')}  oracle_swap={p3.get('oracle_swap_rate')}%")
    print(f"  Selection Edge: {p3.get('selection_edge_verdict')}")
    acc = p3.get("rsr_score_selection", {})
    print(f"  accuracy={acc.get('accuracy_pct')}%  IC={acc.get('ic_rsr_vs_delta')}")

    # ── Phase4 ────────────────────────────────────────────────────────
    print("\n[Phase4] Exit Readiness Assessment...")
    p4 = phase4_exit_readiness(feat_records_by_day)
    print(f"  Exit Readiness: {p4.get('exit_readiness_value')}")
    for dk, dv in p4.get("by_day", {}).items():
        print(f"  {dk}: IC={dv.get('ic_score_vs_rem')}  d={dv.get('cohens_d')}  W_score={dv.get('winner_score_mean')} vs L_score={dv.get('loser_score_mean')}")

    # ── Final Deliverables ────────────────────────────────────────────
    print("\n[Final] Deliverables...")
    final = final_deliverables(p1, p2, p3, p4)
    for item in final.get("6_priority_vs_cluster", {}).get("final_priority_list", []):
        print(f"  Priority {item['priority']}: {item['research']} → {item['value']}")

    # ── 保存 ─────────────────────────────────────────────────────────
    out = {
        "study":  "Study56_UnifiedQualityScore",
        "date":   TODAY_STR,
        "config": "D_ATR_EQ",
        "period": {"is_start": IS_START, "is_end": IS_END},
        "n_sell_trades":   len(sell_trades),
        "n_missed_cands":  len(missed_cands),
        "phase1_construction":  p1,
        "phase2_validation":    p2,
        "phase3_maxpos":        p3,
        "phase4_exit_readiness": p4,
        "final_deliverables":   final,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 保存: {OUT_FILE}")
    print("=" * 72)
    print(f"  Study56 完了")
    print(f"  Quality Score Validation: {p2.get('validation_verdict')}")
    print(f"  Exit Readiness: {p4.get('exit_readiness_value')}")
    print(f"  MAX_POS Selection Edge: {p3.get('selection_edge_verdict')}")
    print(f"  Next: {final.get('next_recommended_research')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
