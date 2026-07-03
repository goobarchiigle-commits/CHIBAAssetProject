"""
study54_entry_alpha_attribution.py
Entry Alpha Attribution — D_ATR_EQ Production

Phase 1: Entry Pool Attribution
  A = RSR未達  (RSR<75, 20d breakout, post-hoc reconstruction)
  B = MAX_POS  (RSR≥75, position cap)
  C = SECTOR_CAP
  D = CLUSTER_CAP
  E = LOT_REJECT
  F = EXECUTED

Phase 2: Winner Attribution (Top20%/Mid60%/Bottom20%)
Phase 3: Feature Screening (AUC, IC, Lift, Coverage)
Deliverable 9: False Negative Analysis

禁止: 閾値最適化 / パラメータ変更 / ルール追加 / エンジン改変
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
from src.config_loader import load_strategy_config

TODAY_STR   = date.today().strftime("%Y-%m-%d")
CAPITAL     = 3_000_000
IS_START    = "2018-01-01"
IS_END      = "2024-12-31"
DATA_END    = "2025-12-31"
MIN_HOLD    = 3
MIN_RSR     = 75.0
FWD_WINDOWS = [20, 60, 120]

# D_ATR_EQ production params
EP_EXIT          = "A"
EP_ADDON         = "D"
ADDON_ATR_MULT   = 1.0
ADDON_SIZE_FRAC  = 0.25

OUT_FILE = ROOT / "backtests" / f"study54_entry_alpha_attribution_{TODAY_STR}.json"


# ================================================================== #
#  ユーティリティ
# ================================================================== #

def _safe_float(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), decimals)


def _atr20_series(df_c: pd.DataFrame) -> pd.Series:
    h = df_c["High"]
    l = df_c["Low"]
    c = df_c["Close"]
    c_prev = c.shift(1).fillna(c)
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    return tr.rolling(20, min_periods=10).mean()


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman 相関係数と p 値 (scipy fallback なし版)"""
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        # 手実装: rank correlate
        rx = pd.Series(x).rank().values
        ry = pd.Series(y).rank().values
        r = float(np.corrcoef(rx, ry)[0, 1])
        n = len(x)
        t = r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-12))
        from scipy.special import betainc as _bi
        p = 0.05  # approximation
        return r, p


def _auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U → AUC (sklearn不要版)"""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return float(u / (len(pos) * len(neg)))


# ================================================================== #
#  BT 実行
# ================================================================== #

def get_active(ds: dict, start: str, end: str) -> pd.DataFrame:
    cfg = load_strategy_config()
    bc = cfg.risk_controls.bear_universe_filter
    be = list(bc.excluded_sectors) if bc.enabled else None
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


def run_bt(ds: dict, sym_active_df, start: str, end: str) -> dict:
    return cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"],
        rsr_df=ds["rsr_df"],
        alpha_df=None,
        regime_df=ds["regime_df"],
        trade_syms=ds["trade_syms"],
        rsr_syms=ds["rsr_syms"],
        cfg=ds["base_cfg"],
        start=start, end=end, verbose=False,
        tech_matrices=ds["tech_matrices"],
        breadth_series=ds["breadth_series"],
        capital=CAPITAL,
        min_hold=MIN_HOLD,
        topix_close=ds["topix_close"],
        market_shock_mode="composite",
        rsr_exit_threshold=70.0,
        sym_active_df=sym_active_df,
        enable_simple_rsr_exit=True,
        enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True,
        enable_atr_risk_sizing=False,
        enable_mtf_filter=False,
        sizing_mode="existing",
        exit_policy=EP_EXIT,
        exit_policy_atr_mult=ADDON_ATR_MULT,
        exit_policy_defer_days=5,
        max_positions_ts=None,
        addon_policy=EP_ADDON,
        addon_atr_mult=ADDON_ATR_MULT,
        addon_stage2_mult=2.0,
        addon_max_per_pos=1,
        addon_size_frac=ADDON_SIZE_FRAC,
    )


# ================================================================== #
#  Group A: RSR未達ブレイクアウト (post-hoc)
# ================================================================== #

def compute_group_a(ds: dict, sym_active_df: pd.DataFrame) -> list[dict]:
    """
    RSR<75 かつ 20日高値ブレイクアウトのイベントを post-hoc 再構築。
    SEPA・RSRモメンタムは省略（conservative 推計）。
    """
    rsr_df      = ds["rsr_df"]
    universe_raw = ds["universe_raw"]
    trade_syms  = ds["trade_syms"]

    mask = (rsr_df.index >= pd.Timestamp(IS_START)) & (rsr_df.index <= pd.Timestamp(IS_END))
    rsr_is   = rsr_df.loc[mask]
    is_dates = rsr_is.index

    events: list[dict] = []

    for sym in rsr_is.columns:
        if sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue

        close = df_c["Close"].dropna()
        close.index = pd.to_datetime(close.index)

        # 20d prev high (先読みなし: shift(1))
        prev_20d = close.rolling(20, min_periods=10).max().shift(1)

        close_al  = close.reindex(is_dates)
        prev_al   = prev_20d.reindex(is_dates)
        rsr_sym   = rsr_is[sym] if sym in rsr_is.columns else pd.Series(dtype=float)

        if sym_active_df is not None and sym in sym_active_df.columns:
            active_al = sym_active_df[sym].reindex(is_dates, fill_value=0.0)
        else:
            active_al = pd.Series(1.0, index=is_dates)

        rsr_lt75 = (rsr_sym < MIN_RSR) & rsr_sym.notna() & (rsr_sym > 0)
        breakout = close_al.notna() & prev_al.notna() & (prev_al > 0) & (close_al > prev_al)
        is_act   = active_al >= 0.5

        comb = rsr_lt75 & breakout & is_act

        for d in is_dates[comb]:
            events.append({
                "date":   str(d.date()),
                "symbol": sym,
                "rsr":    round(float(rsr_sym[d]), 1),
                "sector": trade_syms.get(sym, "不明"),
                "group":  "A",
            })

    return events


# ================================================================== #
#  Forward Returns (全グループ共通)
# ================================================================== #

def _path_extremes(close: pd.Series, ref_date: pd.Timestamp, ref_px: float, horizon: int = 60
                   ) -> tuple[float | None, float | None]:
    """
    ref_date 翌日から horizon 日後までの最大/最小リターン (%)。
    max_runup, max_drawdown を返す。
    """
    future = close[close.index > ref_date]
    if future.empty:
        return None, None
    seg = future.iloc[:horizon]
    if seg.empty:
        return None, None
    max_r = float(seg.max() / ref_px - 1) * 100
    min_r = float(seg.min() / ref_px - 1) * 100
    return round(max_r, 2), round(min_r, 2)


def calc_forward_returns(candidates: list[dict], universe_raw: dict) -> dict:
    """候補リスト → Forward Return サマリー + MaxRunup / MaxDrawdown"""
    if not candidates:
        return {"n": 0}

    rows: list[dict] = []
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
        avail = close[close.index <= d]
        if avail.empty:
            continue
        ref_px = float(avail.iloc[-1])
        if ref_px <= 0:
            continue

        row: dict = {"date": dstr, "symbol": sym,
                     "rsr": cand.get("rsr"), "sector": cand.get("sector")}
        future = close[close.index > d]
        for w in FWD_WINDOWS:
            row[f"fwd{w}d"] = (round((float(future.iloc[w - 1]) - ref_px) / ref_px * 100, 2)
                               if len(future) >= w else None)
        mr, md = _path_extremes(close, d, ref_px, horizon=60)
        row["max_runup60"]   = mr
        row["max_drawdown60"] = md
        rows.append(row)

    if not rows:
        return {"n": 0}

    df = pd.DataFrame(rows)
    n  = len(df)
    out: dict = {"n": n}

    for w in FWD_WINDOWS:
        k  = f"fwd{w}d"
        vs = df[k].dropna().values
        if len(vs) == 0:
            out[k] = {}
            continue
        out[k] = {
            "mean":          round(float(vs.mean()), 2),
            "median":        round(float(np.median(vs)), 2),
            "win_rate":      round(float((vs > 0).mean() * 100), 1),
            "top10pct":      round(float(np.percentile(vs, 90)), 2),
            "bottom10pct":   round(float(np.percentile(vs, 10)), 2),
            "n":             int(len(vs)),
        }

    for k in ("max_runup60", "max_drawdown60"):
        vs = df[k].dropna().values
        out[k + "_avg"] = round(float(vs.mean()), 2) if len(vs) > 0 else None

    # RSR avg
    rsr_vs = df["rsr"].dropna().values
    out["avg_rsr"] = round(float(rsr_vs.mean()), 1) if len(rsr_vs) > 0 else None

    return out


# ================================================================== #
#  Phase 2: Winner Attribution — 特徴量計算
# ================================================================== #

def _compute_features_single(
    sym: str,
    sig_date: pd.Timestamp,
    universe_raw: dict,
    rsr_df: pd.DataFrame,
    topix_close: pd.Series | None,
    is_dates: pd.DatetimeIndex,
    all_rsr_day: pd.Series,
) -> dict:
    """1トレードの signal_date における特徴量を返す。"""
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

    close_val = float(avail.iloc[-1])

    # RSR + slope + acceleration
    rsr_sym = rsr_df[sym] if sym in rsr_df.columns else pd.Series(dtype=float)
    rsr_avail = rsr_sym[rsr_sym.index <= sig_date].dropna()
    if len(rsr_avail) >= 22:
        rsr_val = float(rsr_avail.iloc[-1])
        rsr_21  = float(rsr_avail.iloc[-22])
        rsr_slope = (rsr_val - rsr_21) / 21.0
        rsr_val_prev = float(rsr_avail.iloc[-2])
        rsr_21_prev  = float(rsr_avail.iloc[-23]) if len(rsr_avail) >= 23 else rsr_21
        rsr_slope_prev = (rsr_val_prev - rsr_21_prev) / 21.0
        rsr_accel = rsr_slope - rsr_slope_prev
        feat["rsr"]       = round(rsr_val, 1)
        feat["rsr_slope"] = round(rsr_slope, 3)
        feat["rs_accel"]  = round(rsr_accel, 4)
    elif len(rsr_avail) >= 2:
        feat["rsr"] = round(float(rsr_avail.iloc[-1]), 1)

    # RSR rank (RSR≥75 のアクティブ銘柄中のパーセンタイル)
    if "rsr" in feat and not all_rsr_day.empty:
        q75_syms = all_rsr_day[all_rsr_day >= MIN_RSR]
        if len(q75_syms) > 0:
            feat["rsr_rank_pct"] = round(float((q75_syms < feat["rsr"]).mean() * 100), 1)

    # MA20 deviation
    if len(avail) >= 20:
        ma20 = float(avail.iloc[-20:].mean())
        feat["ma20_dev_pct"] = round((close_val / ma20 - 1.0) * 100, 2) if ma20 > 0 else None

    # 20d prev-high ブレイクアウト距離
    if len(avail) >= 21:
        ph20 = float(avail.iloc[-21:-1].max())
        feat["breakout_dist_pct"] = round((close_val / ph20 - 1.0) * 100, 2) if ph20 > 0 else None

    # ATR
    if "High" in df_c.columns and "Low" in df_c.columns:
        atr = _atr20_series(df_c)
        atr_avail = atr[atr.index <= sig_date].dropna()
        if not atr_avail.empty:
            atr_now = float(atr_avail.iloc[-1])
            feat["atr_pct"] = round(atr_now / close_val * 100, 2) if close_val > 0 else None
            # ATR compression: atr_now / 90d median (shift 1)
            atr90_med_full = atr.rolling(90, min_periods=45).median().shift(1)
            atr90_avail = atr90_med_full[atr90_med_full.index <= sig_date].dropna()
            if not atr90_avail.empty:
                med90 = float(atr90_avail.iloc[-1])
                feat["atr_compression"] = round(atr_now / med90, 3) if med90 > 0 else None

    # Volume expansion
    vol_col = "Volume" if "Volume" in df_c.columns else ("volume" if "volume" in df_c.columns else None)
    if vol_col:
        vol = pd.to_numeric(df_c[vol_col], errors="coerce").dropna()
        vol.index = pd.to_datetime(vol.index)
        vol_avail = vol[vol.index <= sig_date]
        if len(vol_avail) >= 20:
            vol_ma20 = float(vol_avail.iloc[-20:].mean())
            vol_now  = float(vol_avail.iloc[-1])
            feat["vol_expansion"] = round(vol_now / vol_ma20, 3) if vol_ma20 > 0 else None

    # Market Relative Strength (sym vs TOPIX, 20d)
    if topix_close is not None and len(topix_close) > 0:
        topix_avail = topix_close[topix_close.index <= sig_date].dropna()
        if len(topix_avail) >= 21 and len(avail) >= 21:
            sym_ret20  = float(avail.iloc[-1] / avail.iloc[-21] - 1)
            tp_ret20   = float(topix_avail.iloc[-1] / topix_avail.iloc[-21] - 1)
            feat["mkt_rs_20d"] = round((sym_ret20 - tp_ret20) * 100, 2)

    return feat


def compute_trade_features(
    trades: list[dict],
    ds: dict,
    is_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """全 EXECUTED トレード × 特徴量 の DataFrame を返す。"""
    rsr_df      = ds["rsr_df"]
    universe_raw = ds["universe_raw"]
    topix_close = ds.get("topix_close")
    trade_syms  = ds["trade_syms"]

    # IS 期間の RSR スライス (日付 → Series)
    rsr_is = rsr_df.loc[rsr_df.index.isin(is_dates)]

    records = []
    for t in trades:
        if t.get("side") != "SELL":
            continue
        sym        = t["symbol"]
        entry_idx  = t.get("entry_idx", -1)
        exit_idx   = t.get("exit_idx", -1)
        entry_px   = t.get("entry", 0.0)
        exit_px    = t.get("exit", 0.0)
        return_pct = (exit_px / entry_px - 1.0) * 100 if entry_px > 0 else None

        # Signal date = entry_idx - 1 (BUY は翌日 open)
        sig_idx = entry_idx - 1
        if sig_idx < 0 or sig_idx >= len(is_dates):
            continue
        sig_date   = is_dates[sig_idx]
        entry_date = is_dates[entry_idx] if entry_idx < len(is_dates) else None
        hold_days  = exit_idx - entry_idx if exit_idx >= 0 and entry_idx >= 0 else None

        # その日の全銘柄 RSR (RSR rank 計算用)
        all_rsr_day = rsr_is.loc[sig_date] if sig_date in rsr_is.index else pd.Series(dtype=float)

        feat = _compute_features_single(
            sym, sig_date, universe_raw, rsr_is, topix_close, is_dates, all_rsr_day
        )

        row = {
            "symbol":      sym,
            "sector":      trade_syms.get(sym, "不明"),
            "signal_date": str(sig_date.date()),
            "entry_date":  str(entry_date.date()) if entry_date is not None else None,
            "return_pct":  _safe_float(return_pct, 2),
            "pnl":         _safe_float(t.get("pnl"), 0),
            "hold_days":   hold_days,
            "exit_reason": t.get("reason", ""),
        }
        row.update(feat)
        records.append(row)

    return pd.DataFrame(records) if records else pd.DataFrame()


# ================================================================== #
#  Phase 2: Winner Attribution
# ================================================================== #

FEATURE_LIST = [
    "rsr", "rsr_rank_pct", "rsr_slope", "rs_accel",
    "vol_expansion", "atr_compression", "atr_pct",
    "breakout_dist_pct", "ma20_dev_pct", "mkt_rs_20d",
]


def winner_attribution(trade_df: pd.DataFrame) -> dict:
    df = trade_df.dropna(subset=["return_pct"]).copy()
    if len(df) < 10:
        return {"error": "n<10"}

    n      = len(df)
    top_thr = np.percentile(df["return_pct"], 80)
    bot_thr = np.percentile(df["return_pct"], 20)

    top = df[df["return_pct"] >= top_thr]
    mid = df[(df["return_pct"] > bot_thr) & (df["return_pct"] < top_thr)]
    bot = df[df["return_pct"] <= bot_thr]

    result: dict = {
        "n_total": n,
        "top20pct": {
            "n": len(top), "threshold_pct": _safe_float(top_thr, 2),
            "avg_return": _safe_float(float(top["return_pct"].mean()), 2),
            "avg_hold":   _safe_float(float(top["hold_days"].mean()), 1) if "hold_days" in top else None,
        },
        "mid60pct": {
            "n": len(mid),
            "avg_return": _safe_float(float(mid["return_pct"].mean()), 2),
        },
        "bot20pct": {
            "n": len(bot), "threshold_pct": _safe_float(bot_thr, 2),
            "avg_return": _safe_float(float(bot["return_pct"].mean()), 2),
            "avg_hold":   _safe_float(float(bot["hold_days"].mean()), 1) if "hold_days" in bot else None,
        },
        "feature_comparison":  {},
        "rank_correlations":   {},
        "effect_sizes":        {},
    }

    for feat in FEATURE_LIST:
        if feat not in df.columns:
            continue
        top_v = top[feat].dropna()
        bot_v = bot[feat].dropna()
        all_v = df[feat].dropna()

        result["feature_comparison"][feat] = {
            "top_mean":   _safe_float(float(top_v.mean()), 3) if len(top_v) else None,
            "bot_mean":   _safe_float(float(bot_v.mean()), 3) if len(bot_v) else None,
            "all_mean":   _safe_float(float(all_v.mean()), 3) if len(all_v) else None,
            "top_vs_bot": _safe_float(float(top_v.mean() - bot_v.mean()), 3) if len(top_v) and len(bot_v) else None,
            "coverage_pct": round(float(len(all_v) / n * 100), 1),
        }

        valid = df[["return_pct", feat]].dropna()
        if len(valid) >= 5:
            corr, p_val = _spearman_corr(valid["return_pct"].values, valid[feat].values)
            result["rank_correlations"][feat] = {
                "spearman": _safe_float(corr, 3),
                "p_value":  _safe_float(p_val, 4),
                "n":        len(valid),
            }

            if len(top_v) > 1 and len(bot_v) > 1:
                pooled = np.sqrt((top_v.var(ddof=1) + bot_v.var(ddof=1)) / 2)
                d = (float(top_v.mean()) - float(bot_v.mean())) / pooled if pooled > 0 else 0.0
                result["effect_sizes"][feat] = _safe_float(d, 3)

    return result


# ================================================================== #
#  Phase 3: Feature Screening
# ================================================================== #

def feature_screening(trade_df: pd.DataFrame) -> dict:
    df = trade_df.dropna(subset=["return_pct"]).copy()
    if len(df) < 10:
        return {"error": "n<10"}

    # binary: top20% = 1, bottom20% = 0 (extremes only)
    top_thr = np.percentile(df["return_pct"], 80)
    bot_thr = np.percentile(df["return_pct"], 20)
    df_ext = df[(df["return_pct"] >= top_thr) | (df["return_pct"] <= bot_thr)].copy()
    df_ext["label"] = (df_ext["return_pct"] >= top_thr).astype(int)

    base_return = float(df["return_pct"].mean())

    results: dict = {}

    for feat in FEATURE_LIST:
        if feat not in df.columns:
            continue

        n_valid  = int(df[feat].notna().sum())
        coverage = round(n_valid / len(df) * 100, 1)

        valid = df[["return_pct", feat]].dropna()
        if len(valid) < 5:
            results[feat] = {"coverage_pct": coverage, "n_valid": n_valid}
            continue

        ic, p_ic = _spearman_corr(valid["return_pct"].values, valid[feat].values)

        # AUC (top20% vs bottom20%)
        df_ext_f = df_ext[[feat, "label"]].dropna(subset=[feat])
        auc = None
        if len(df_ext_f) >= 10 and df_ext_f["label"].nunique() > 1:
            auc = _safe_float(_auc_binary(df_ext_f[feat].values, df_ext_f["label"].values), 3)

        # Top decile vs Bottom decile (feature → return)
        q90 = df[feat].quantile(0.9)
        q10 = df[feat].quantile(0.1)
        top_feat_ret = df.loc[df[feat] >= q90, "return_pct"].mean()
        bot_feat_ret = df.loc[df[feat] <= q10, "return_pct"].mean()
        lift = (top_feat_ret / base_return) if base_return != 0 and not np.isnan(top_feat_ret) else None

        results[feat] = {
            "coverage_pct":              coverage,
            "n_valid":                   n_valid,
            "ic_spearman":               _safe_float(ic, 3),
            "p_value":                   _safe_float(p_ic, 4),
            "auc":                       auc,
            "top_feat_decile_avg_ret":   _safe_float(top_feat_ret, 2),
            "bot_feat_decile_avg_ret":   _safe_float(bot_feat_ret, 2),
            "base_avg_return":           _safe_float(base_return, 2),
            "lift":                      _safe_float(lift, 2) if lift is not None else None,
        }

    # ランク: |IC| 降順
    ranked = sorted(
        [(k, v) for k, v in results.items() if "ic_spearman" in v and v["ic_spearman"] is not None],
        key=lambda x: abs(x[1]["ic_spearman"]),
        reverse=True,
    )
    top5 = [
        {"rank": i + 1, "feature": k, "ic": v["ic_spearman"], "auc": v.get("auc"),
         "lift": v.get("lift"), "coverage_pct": v.get("coverage_pct")}
        for i, (k, v) in enumerate(ranked[:5])
    ]

    return {"by_feature": results, "ranked_by_ic": [r[0] for r in ranked], "top5_candidates": top5}


# ================================================================== #
#  Alpha Leakage Ranking (Phase 1 から導出)
# ================================================================== #

def alpha_leakage_ranking(phase1: dict, group_f_fwd60: float | None) -> list[dict]:
    """
    各グループの fwd60d_mean vs EXECUTED(F) を比較して機会損失をランク。
    """
    base = group_f_fwd60 or 0.0
    rows = []
    for grp in ("A", "B", "C", "D", "E"):
        g = phase1.get(f"group_{grp}", {})
        n = g.get("n", 0)
        if n == 0:
            continue
        fwd60 = g.get("fwd60d", {}).get("mean")
        if fwd60 is None:
            continue
        alpha_vs_exec = round(fwd60 - base, 2)
        rows.append({
            "group":         grp,
            "n":             n,
            "fwd60d_mean":   fwd60,
            "alpha_vs_exec": alpha_vs_exec,
            "win_rate60d":   g.get("fwd60d", {}).get("win_rate"),
        })
    rows.sort(key=lambda x: x["fwd60d_mean"], reverse=True)
    for i, r in enumerate(rows):
        r["alpha_rank"] = i + 1
    return rows


# ================================================================== #
#  Deliverable 9: False Negative Analysis
# ================================================================== #

def false_negative_analysis(
    missed_cands: list[dict],
    executed_fwd60_med: float,
    universe_raw: dict,
    is_dates: pd.DatetimeIndex,
    rsr_df: pd.DataFrame,
    trade_syms: dict,
) -> list[dict]:
    """
    MAX_POS 不採用候補のうち、fwd60d > EXECUTED 中央値の銘柄を全抽出。
    """
    results = []
    for cand in missed_cands:
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
        avail = close[close.index <= d]
        if avail.empty:
            continue
        ref_px = float(avail.iloc[-1])
        if ref_px <= 0:
            continue
        future = close[close.index > d]
        fwd60  = (float(future.iloc[59]) / ref_px - 1) * 100 if len(future) >= 60 else None
        if fwd60 is None or fwd60 <= executed_fwd60_med:
            continue

        # Rank on the skip date (was it the #1 candidate?)
        rsr_val  = cand.get("rsr", 0.0)
        rank_idx = cand.get("rank", -1)  # 0=1位

        # 追加特徴量
        sig_avail = close[close.index <= d]
        ma20_dev = None
        if len(sig_avail) >= 20:
            ma20 = float(sig_avail.iloc[-20:].mean())
            ma20_dev = round((ref_px / ma20 - 1) * 100, 2) if ma20 > 0 else None

        results.append({
            "date":           dstr,
            "symbol":         sym,
            "sector":         trade_syms.get(sym, "不明"),
            "rsr":            rsr_val,
            "rank_in_pool":   rank_idx,
            "fwd60d":         round(fwd60, 2),
            "fwd60_vs_exec_median": round(fwd60 - executed_fwd60_med, 2),
            "ma20_dev_pct":   ma20_dev,
            "atr_pct":        cand.get("atr_pct"),
        })

    results.sort(key=lambda x: x["fwd60d"], reverse=True)
    return results


# ================================================================== #
#  Study55 / Study56 判定
# ================================================================== #

def _study55_recommend(winner_attr: dict, phase3: dict) -> str:
    """Quality Exit (Study55) の実施推奨を判定。"""
    lines = []
    # 勝者 hold_days が長い → exit 延命の効果を評価する価値あり
    top_hold = winner_attr.get("top20pct", {}).get("avg_hold")
    bot_hold = winner_attr.get("bot20pct", {}).get("avg_hold")
    if top_hold and bot_hold and top_hold > bot_hold * 1.3:
        lines.append(f"TOP20% avg_hold={top_hold:.1f}d > BOT20% {bot_hold:.1f}d (×1.3): 保有延長で追加利益示唆")
    # RSR slope が強い場合 → exit 判断にも RSR momentum を活かせるか
    fc = winner_attr.get("feature_comparison", {})
    if "rsr_slope" in fc:
        top_slope = fc["rsr_slope"].get("top_mean") or 0
        if abs(top_slope) > 0.05:
            lines.append(f"rsr_slope TOP={top_slope:.3f}: exit 判断への組込み余地あり")

    if lines:
        return "RECOMMEND: " + " / ".join(lines)
    return "DEFER: 現状データでは exit 改善余地の根拠不十分 (Study40-47で概ね枯渇)"


def _study56_recommend(group_a: dict, phase3: dict) -> str:
    """Early Entry (Study56) の実施推奨を判定。"""
    n_a    = group_a.get("n", 0)
    fwd60  = group_a.get("fwd60d", {}).get("mean")
    wrate  = group_a.get("fwd60d", {}).get("win_rate")
    fn60   = None  # will be set by caller

    if n_a == 0:
        return "DEFER: Group A イベント不足"
    if fwd60 is None:
        return "DEFER: Group A fwd60d データ不足"

    lines = []
    if fwd60 >= 5.0 and wrate and wrate >= 55.0:
        lines.append(f"Group A fwd60d={fwd60:.2f}%, WR={wrate:.1f}% → 早期エントリーで追加アルファ示唆")
    if fwd60 < 3.0:
        lines.append(f"Group A fwd60d={fwd60:.2f}% 低水準 → RSR<75 での早期参入は期待値低い")

    # rsr_slope の IC が高ければ → RSR slope が 75 到達前に使える可能性
    top5 = phase3.get("top5_candidates", [])
    if any(c["feature"] == "rsr_slope" for c in top5[:3]):
        lines.append("rsr_slope が Phase3 Top3 に入る → RSR slope ゲートで早期参入候補絞り込み可能")

    if lines:
        return "RECOMMEND: " + " / ".join(lines)
    return "DEFER: Group A の期待値が EXECUTED を大幅に下回る or データ不足"


# ================================================================== #
#  メイン
# ================================================================== #

def main() -> None:
    print("=" * 72)
    print("  Study54 — Entry Alpha Attribution (D_ATR_EQ)")
    print(f"  Date: {TODAY_STR}   IS: {IS_START}~{IS_END}   Capital: ¥{CAPITAL:,}")
    print("=" * 72)

    # ── データセット ──────────────────────────────────────────────────
    print("\n[Data] データセット構築中...")
    ds = build_common_dataset(DATA_END)
    print(f"  {len(ds['trade_syms'])} シンボル")

    # ── BT 実行 (D_ATR_EQ) ─────────────────────────────────────────
    print("\n[BT] D_ATR_EQ IS run...")
    active_is = get_active(ds, IS_START, IS_END)
    res = run_bt(ds, active_is, IS_START, IS_END)
    print(f"  CAGR={res['cagr']:+.2f}%  Sharpe={res['sharpe']:.3f}  MaxDD={res['max_dd']:.2f}%  Trades={res['n_trades']}")

    # IS 期間の共通日付インデックスを rsr_df から再構築
    rsr_df   = ds["rsr_df"]
    is_dates = rsr_df.index[(rsr_df.index >= IS_START) & (rsr_df.index <= IS_END)]
    print(f"  IS dates={len(is_dates)}")

    # ── グループ分類 ────────────────────────────────────────────────
    trades_f      = [t for t in res.get("_trades", []) if t.get("side") == "SELL"]
    missed_cands  = res.get("_missed_cands", [])    # B: MAX_POS
    skip_detail   = res.get("_skip_detail", [])      # C: SECTOR_CAP, D: CLUSTER_CAP, GROSS
    lot_rej       = res.get("_rejected_by_lot_detail", [])  # E: LOT_REJECT

    group_b = [c for c in missed_cands]
    group_c = [c for c in skip_detail if c.get("reason") == "SECTOR_CAP"]
    group_d = [c for c in skip_detail if c.get("reason") == "CLUSTER_CAP"]
    group_e = lot_rej

    # Group F (EXECUTED): signal_date に変換
    group_f_cands = []
    for t in trades_f:
        ei = t.get("entry_idx", -1)
        si = ei - 1
        if 0 <= si < len(is_dates):
            group_f_cands.append({
                "date":   str(is_dates[si].date()),
                "symbol": t["symbol"],
                "rsr":    None,
                "sector": t.get("sector"),
            })

    print("\n[Phase1] Group A reconstruction (RSR<75 + 20d breakout)...")
    group_a_events = compute_group_a(ds, active_is)
    print(f"  Group A: {len(group_a_events)} events")

    print(f"  Group B (MAX_POS):     {len(group_b)}")
    print(f"  Group C (SECTOR_CAP):  {len(group_c)}")
    print(f"  Group D (CLUSTER_CAP): {len(group_d)}")
    print(f"  Group E (LOT_REJECT):  {len(group_e)}")
    print(f"  Group F (EXECUTED):    {len(group_f_cands)}")

    # ── Forward Returns (全グループ) ────────────────────────────────
    print("\n[FwdRet] Forward return 計算中...")
    univ = ds["universe_raw"]

    # Group A の RSR を cand に付与済み (compute_group_a が rsr を含む)
    fr_a = calc_forward_returns(group_a_events, univ)
    fr_b = calc_forward_returns(group_b, univ)
    fr_c = calc_forward_returns(group_c, univ)
    fr_d = calc_forward_returns(group_d, univ)
    fr_e = calc_forward_returns(group_e, univ)
    fr_f = calc_forward_returns(group_f_cands, univ)

    print(f"  A fwd60d: {fr_a.get('fwd60d', {}).get('mean', 'N/A')}")
    print(f"  B fwd60d: {fr_b.get('fwd60d', {}).get('mean', 'N/A')}")
    print(f"  C fwd60d: {fr_c.get('fwd60d', {}).get('mean', 'N/A')}")
    print(f"  D fwd60d: {fr_d.get('fwd60d', {}).get('mean', 'N/A')}")
    print(f"  E fwd60d: {fr_e.get('fwd60d', {}).get('mean', 'N/A')}")
    print(f"  F fwd60d: {fr_f.get('fwd60d', {}).get('mean', 'N/A')}")

    phase1 = {
        "group_A": {"label": "RSR未達", "n": len(group_a_events), **fr_a},
        "group_B": {"label": "MAX_POS",     "n": len(group_b), "avg_rsr": _safe_float(
            np.mean([c.get("rsr", 0) for c in group_b if c.get("rsr")]), 1), **fr_b},
        "group_C": {"label": "SECTOR_CAP",  "n": len(group_c), **fr_c},
        "group_D": {"label": "CLUSTER_CAP", "n": len(group_d), **fr_d},
        "group_E": {"label": "LOT_REJECT",  "n": len(group_e), **fr_e},
        "group_F": {"label": "EXECUTED",    "n": len(group_f_cands), **fr_f},
    }

    exec_fwd60_med = fr_f.get("fwd60d", {}).get("median", 0.0) or 0.0
    exec_fwd60_mean = fr_f.get("fwd60d", {}).get("mean", 0.0) or 0.0

    alpha_leak = alpha_leakage_ranking(phase1, exec_fwd60_mean)

    # ── Phase 2: Winner Attribution ──────────────────────────────────
    print("\n[Phase2] Feature computation for EXECUTED trades...")
    trade_feat_df = compute_trade_features(trades_f, ds, is_dates)
    print(f"  Trades with features: {len(trade_feat_df)}")

    w_attr = winner_attribution(trade_feat_df)
    print(f"  Top20% avg_return: {w_attr.get('top20pct', {}).get('avg_return')}")
    print(f"  Bot20% avg_return: {w_attr.get('bot20pct', {}).get('avg_return')}")

    # ── Phase 3: Feature Screening ───────────────────────────────────
    print("\n[Phase3] Feature screening...")
    phase3 = feature_screening(trade_feat_df)
    print(f"  Top feature by |IC|: {phase3.get('ranked_by_ic', ['N/A'])[0]}")

    # ── Deliverable 9: False Negative Analysis ───────────────────────
    print("\n[FN] False negative analysis...")
    fn_list = false_negative_analysis(
        missed_cands=group_b,
        executed_fwd60_med=exec_fwd60_med,
        universe_raw=univ,
        is_dates=is_dates,
        rsr_df=rsr_df,
        trade_syms=ds["trade_syms"],
    )
    print(f"  False negatives (fwd60d > exec median {exec_fwd60_med:.2f}%): {len(fn_list)}")

    # ── Study55/56 判定 ──────────────────────────────────────────────
    rec55 = _study55_recommend(w_attr, phase3)
    rec56 = _study56_recommend(fr_a, phase3)

    # ── 結果出力 ─────────────────────────────────────────────────────
    output = {
        "study":  "Study54_EntryAlphaAttribution",
        "date":   TODAY_STR,
        "config": "D_ATR_EQ (ATR_Extension + EQ_SCALE, VOL_ADJ除外)",
        "period": {"is_start": IS_START, "is_end": IS_END},
        "capital": CAPITAL,
        "bt_summary": {
            "cagr": res["cagr"], "sharpe": res["sharpe"],
            "max_dd": res["max_dd"], "n_trades": res["n_trades"],
        },
        "phase1_entry_pool": phase1,
        "alpha_leakage_ranking": alpha_leak,
        "phase2_winner_attribution": w_attr,
        "phase3_feature_screening": phase3,
        "false_negatives": fn_list[:50],  # top-50
        "false_negatives_total": len(fn_list),
        "study55_recommend": rec55,
        "study56_recommend": rec56,
    }

    OUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] 結果保存: {OUT_FILE}")

    # ── コンソール サマリー ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Phase1: Entry Pool Attribution")
    print("=" * 72)
    headers = ["Group", "N", "avgRSR", "fwd20d", "fwd60d", "fwd120d", "WR60d", "MaxRU60", "MaxDD60"]
    print(f"  {'  '.join(headers)}")
    for grp, label in [("A","RSR未達"),("B","MAX_POS"),("C","SEC_CAP"),("D","CLS_CAP"),("E","LOT_REJ"),("F","EXECUTED")]:
        g = phase1[f"group_{grp}"]
        n = g["n"]
        ar = g.get("avg_rsr", "—")
        f20 = g.get("fwd20d", {}).get("mean", "—")
        f60 = g.get("fwd60d", {}).get("mean", "—")
        f120= g.get("fwd120d",{}).get("mean", "—")
        wr  = g.get("fwd60d", {}).get("win_rate", "—")
        mr  = g.get("max_runup60_avg", "—")
        md  = g.get("max_drawdown60_avg", "—")
        print(f"  {grp}({label:10s}) n={n:4d}  RSR={ar}  "
              f"f20={f20}  f60={f60}  f120={f120}  WR={wr}  MR={mr}  MD={md}")

    print("\n  Alpha Leakage Ranking (fwd60d vs EXECUTED):")
    for r in alpha_leak:
        print(f"    #{r['alpha_rank']} {r['group']}: fwd60={r['fwd60d_mean']:.2f}% "
              f"Δ={r['alpha_vs_exec']:+.2f}pp  n={r['n']}")

    print("\n" + "=" * 72)
    print("  Phase2: Winner vs Loser (feature avg)")
    print("=" * 72)
    fc = w_attr.get("feature_comparison", {})
    rc = w_attr.get("rank_correlations", {})
    es = w_attr.get("effect_sizes", {})
    for feat in FEATURE_LIST:
        if feat not in fc:
            continue
        tm = fc[feat].get("top_mean", "—")
        bm = fc[feat].get("bot_mean", "—")
        sp = rc.get(feat, {}).get("spearman", "—")
        ef = es.get(feat, "—")
        cov= fc[feat].get("coverage_pct", "—")
        print(f"  {feat:22s}  TOP={tm}  BOT={bm}  IC={sp}  d={ef}  cov={cov}%")

    print("\n" + "=" * 72)
    print("  Phase3: Feature Importance (|IC| rank)")
    print("=" * 72)
    top5 = phase3.get("top5_candidates", [])
    for c in top5:
        print(f"  #{c['rank']} {c['feature']:22s}  IC={c['ic']}  AUC={c['auc']}  Lift={c['lift']}  cov={c['coverage_pct']}%")

    print("\n" + "=" * 72)
    print(f"  Study55(Quality Exit): {rec55}")
    print(f"  Study56(Early Entry):  {rec56}")
    print("=" * 72)

    print(f"\n  False Negatives (fwd60 > exec_median {exec_fwd60_med:.2f}%): {len(fn_list)}件")
    for fn in fn_list[:10]:
        print(f"    {fn['date']} {fn['symbol']} ({fn['sector']}) RSR={fn['rsr']} "
              f"rank={fn['rank_in_pool']} fwd60={fn['fwd60d']:.2f}% Δ={fn['fwd60_vs_exec_median']:+.2f}pp")


if __name__ == "__main__":
    main()
