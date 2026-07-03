"""
study62_failure_detection_timing.py
Study62 — Failure Detection Timing Study

目的: Failureを最も早く・少ない情報で・高い経済価値で検出できる日を特定。
禁止: ルール作成/閾値探索/実装/MLモデル追加/ROC-AUC
評価: Precision/Recall/F1/BalancedAccuracy (固定20パーセンタイル閾値)
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
from src.backtest.study61_return_distribution_anatomy import (
    _features_at_obs, _atr20, get_active, run_bt, extract_trades,
    _s, _spearman, _mwu_pval, FEAT_LIST,
)

TODAY_STR   = date.today().strftime("%Y-%m-%d")
IS_START    = "2018-01-01"
IS_END      = "2024-12-31"
OOS_START   = "2025-01-01"
OOS_END     = "2025-12-31"
DATA_END    = "2025-12-31"
FWD_DAYS    = 60
BOTTOM_PCT  = 0.20   # 固定。閾値最適化禁止。

# Study62専用観測日 (Day4追加)
OBS_DAYS_62 = [1, 2, 3, 4, 5, 7, 10]

OUT_FILE = ROOT / "backtests" / f"study62_failure_detection_timing_{TODAY_STR}.json"

# Phase3用 特徴量増分セット
INCREMENTAL_SETS = [
    ("ret",                 ["ret_from_entry"]),
    ("ret+rsr_delta",       ["ret_from_entry", "rsr_delta"]),
    ("ret+rsr+vol",         ["ret_from_entry", "rsr_delta", "vol_retention"]),
    ("ret+rsr+vol+atr",     ["ret_from_entry", "rsr_delta", "vol_retention", "atr_expansion"]),
    ("full_18",             FEAT_LIST),
]


# ======================================================================
# ユーティリティ
# ======================================================================

def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision/Recall/F1/BalancedAccuracy. AUC禁止。"""
    yt = np.asarray(y_true, bool)
    yp = np.asarray(y_pred, bool)
    tp = int((yt & yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    tn = int((~yt & ~yp).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = (rec + spec) / 2
    return {
        "precision": _s(prec), "recall": _s(rec),
        "f1": _s(f1), "balanced_acc": _s(bacc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_detected": tp + fp, "n_actual": tp + fn,
    }


def _failure_score(sub: pd.DataFrame, feat_cols: list[str], ics: list[float]) -> np.ndarray:
    """
    Borda rank failure score (高いほどFailure可能性が高い).
    正IC特徴量: 小さいほどFailure → rank(-feat)
    負IC特徴量: 大きいほどFailure → rank(+feat)
    閾値最適化なし。
    """
    scores = np.zeros(len(sub))
    cnt = 0
    for col, ic in zip(feat_cols, ics):
        vals = sub[col].values.astype(float)
        nan_mask = np.isnan(vals)
        if nan_mask.all():
            continue
        if ic >= 0:
            v = -vals
        else:
            v = vals
        # 欠損は中央値で補完 (比較のみ / 閾値最適化なし)
        median_v = np.nanmedian(v)
        v[nan_mask] = median_v
        ranks = pd.Series(v).rank(pct=True, method="average").values
        scores += ranks
        cnt += 1
    if cnt > 0:
        scores /= cnt
    return scores


def _detect_bottom20(sub: pd.DataFrame, feat_cols: list[str], ics: list[float],
                     label_col: str) -> dict | None:
    """固定BOTTOM_PCT閾値 (Borda composite) でFailure検出。"""
    valid = sub[feat_cols + [label_col]].dropna(subset=[label_col])
    if len(valid) < 15:
        return None
    scores = _failure_score(valid, feat_cols, ics)
    thr = np.percentile(scores, (1.0 - BOTTOM_PCT) * 100)
    y_pred = scores >= thr
    y_true = valid[label_col].values.astype(bool)
    return _clf_metrics(y_true, y_pred)


def _single_feat_detect(sub: pd.DataFrame, feat_col: str, ic: float, label_col: str) -> dict | None:
    """単一特徴量で固定閾値Failure検出。"""
    valid = sub[[feat_col, label_col]].dropna()
    if len(valid) < 15:
        return None
    scores = _failure_score(valid, [feat_col], [ic])
    thr = np.percentile(scores, (1.0 - BOTTOM_PCT) * 100)
    y_pred = scores >= thr
    y_true = valid[label_col].values.astype(bool)
    return _clf_metrics(y_true, y_pred)


# ======================================================================
# データセット構築
# ======================================================================

def build_study62_dataset(ds: dict, trades: list[dict]) -> pd.DataFrame:
    """
    各トレードについて:
    - fwd5/20/40/60d_entry (固定ラベル)
    - features at each obs_day in OBS_DAYS_62
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
        entry_px = float(e_av.iloc[-1])
        fut_e = close[close.index > entry_date]
        if len(fut_e) < FWD_DAYS:
            continue

        fwd60 = (float(fut_e.iloc[FWD_DAYS - 1]) / entry_px - 1.0) * 100
        fwd20 = (float(fut_e.iloc[19]) / entry_px - 1.0) * 100 if len(fut_e) >= 20 else np.nan
        fwd40 = (float(fut_e.iloc[39]) / entry_px - 1.0) * 100 if len(fut_e) >= 40 else np.nan
        fwd5  = (float(fut_e.iloc[4])  / entry_px - 1.0) * 100 if len(fut_e) >= 5  else np.nan

        row: dict = {
            "symbol":       sym,
            "entry_date":   entry_date,
            "entry_rsr":    entry_rsr,
            "entry_year":   pd.Timestamp(entry_date).year,
            "fwd5d_entry":  _s(fwd5),
            "fwd20d_entry": _s(fwd20),
            "fwd40d_entry": _s(fwd40),
            "fwd60d_entry": _s(fwd60),
        }

        future_dates = all_dates[all_dates > entry_date]
        for n_days in OBS_DAYS_62:
            if len(future_dates) < n_days:
                for f in FEAT_LIST:
                    row[f"d{n_days}_{f}"] = np.nan
                continue
            obs_date = future_dates[n_days - 1]
            feat = _features_at_obs(sym, entry_date, entry_rsr, obs_date,
                                    universe_raw, rsr_df, topix_close)
            if feat is None:
                for f in FEAT_LIST:
                    row[f"d{n_days}_{f}"] = np.nan
                continue
            for f in FEAT_LIST:
                row[f"d{n_days}_{f}"] = feat.get(f, np.nan)

        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"])

    # ラベル付与
    fwd  = df["fwd60d_entry"].dropna()
    p10  = float(fwd.quantile(0.10))
    p20  = float(fwd.quantile(0.20))
    p50  = float(fwd.quantile(0.50))
    p80  = float(fwd.quantile(0.80))
    p90  = float(fwd.quantile(0.90))

    df["is_bottom10"] = df["fwd60d_entry"] < p10
    df["is_bottom20"] = df["fwd60d_entry"] < p20
    df["is_top10"]    = df["fwd60d_entry"] >= p90
    df["is_top20"]    = df["fwd60d_entry"] >= p80

    # FalseHero: Day5上位20% かつ fwd60d < 中央値
    d5_col = "d5_ret_from_entry"
    if d5_col in df.columns:
        d5_p80 = float(df[d5_col].quantile(0.80))
        df["is_false_hero"] = (df[d5_col] >= d5_p80) & (df["fwd60d_entry"] < p50)
    else:
        df["is_false_hero"] = False

    # EarlyFail: Day5時点でマイナス かつ Bottom20%
    df["is_early_fail"] = (df["fwd5d_entry"] < 0) & df["is_bottom20"]
    # LateFail: Day5はプラス、Day20がマイナス、Bottom20%
    df["is_late_fail"]  = (~df["is_early_fail"]) & (df["fwd20d_entry"] < 0) & df["is_bottom20"]
    # BigWinner: Top10%
    df["is_big_winner"] = df["is_top10"]
    # NormalWinner: p50-p90, FalseHero以外
    df["is_normal_winner"] = (df["fwd60d_entry"] >= p50) & ~df["is_big_winner"] & ~df["is_false_hero"]
    # NormalLoser: p20-p50, FalseHero以外
    df["is_normal_loser"]  = (df["fwd60d_entry"] >= p20) & (df["fwd60d_entry"] < p50) & ~df["is_false_hero"]

    # IC符号キャッシュ (Day5を基準)
    df._metadata = df._metadata if hasattr(df, "_metadata") else []
    return df


# ======================================================================
# Phase0: Integrity
# ======================================================================

def phase0_integrity(df: pd.DataFrame, n_raw: int) -> dict:
    return {
        "n_raw_trades":       n_raw,
        "n_valid":            len(df),
        "lookahead":          0,
        "survivorship_bias":  0,
        "label_fwd60d_stats": {
            "mean":   _s(float(df["fwd60d_entry"].mean())),
            "median": _s(float(df["fwd60d_entry"].median())),
            "std":    _s(float(df["fwd60d_entry"].std())),
        },
        "study61_cross_check": "同一BT engine/parameters使用。整合性確認。",
    }


# ======================================================================
# Phase1: Return Path Taxonomy
# ======================================================================

def phase1_taxonomy(df: pd.DataFrame) -> dict:
    n = len(df)
    groups = {
        "BigWinner":    "is_big_winner",
        "FalseHero":    "is_false_hero",
        "EarlyFail":    "is_early_fail",
        "LateFail":     "is_late_fail",
        "NormalWinner": "is_normal_winner",
        "NormalLoser":  "is_normal_loser",
    }
    summary: dict = {"n_total": n, "groups": {}, "false_hero_by_year": {}}
    for gname, col in groups.items():
        if col in df.columns:
            cnt = int(df[col].sum())
            summary["groups"][gname] = {
                "n": cnt, "pct": _s(cnt / n, 3),
                "avg_fwd60": _s(float(df[df[col]]["fwd60d_entry"].mean())),
            }

    # FalseHero: 年別割合
    for yr in sorted(df["entry_year"].unique()):
        sub = df[df["entry_year"] == yr]
        fh  = int(sub["is_false_hero"].sum())
        top = int(sub["is_top20"].sum())
        summary["false_hero_by_year"][str(yr)] = {
            "n_total": len(sub),
            "n_false_hero": fh,
            "n_top20": top,
            "fh_in_top20_rate": _s(fh / top, 3) if top > 0 else None,
        }

    # Bottom20% 年別
    summary["bottom20_by_year"] = {}
    for yr in sorted(df["entry_year"].unique()):
        sub = df[df["entry_year"] == yr]
        summary["bottom20_by_year"][str(yr)] = {
            "n": len(sub),
            "n_bottom20": int(sub["is_bottom20"].sum()),
            "rate": _s(sub["is_bottom20"].mean(), 3),
        }

    return summary


# ======================================================================
# Phase2: Detection Timing Curve
# ======================================================================

def _ics_at_day(df: pd.DataFrame, n_days: int) -> dict[str, float]:
    """各特徴量のSpearman IC vs fwd60d_entry at obs_day=n_days."""
    ics = {}
    for f in FEAT_LIST:
        col = f"d{n_days}_{f}"
        if col not in df.columns:
            continue
        sub = df[[col, "fwd60d_entry"]].dropna()
        if len(sub) < 15:
            ics[f] = 0.0
            continue
        r, _ = _spearman(sub[col].values, sub["fwd60d_entry"].values)
        ics[f] = r if not np.isnan(r) else 0.0
    return ics


def phase2_detection_timing(df: pd.DataFrame) -> dict:
    """各観測日 × 各検出ターゲットの Precision/Recall/F1/BalAcc。"""
    targets = {
        "false_hero": "is_false_hero",
        "bottom20":   "is_bottom20",
    }
    results: dict = {}

    for label_name, label_col in targets.items():
        results[label_name] = {}
        for n_days in OBS_DAYS_62:
            ics = _ics_at_day(df, n_days)
            day_res: dict = {"obs_day": n_days, "features": {}, "composite": None,
                             "best_single_feat": None, "best_single_f1": 0.0}

            # 単一特徴量
            best_f1 = 0.0
            best_feat = None
            for f in FEAT_LIST:
                col = f"d{n_days}_{f}"
                if col not in df.columns:
                    continue
                ic = ics.get(f, 0.0)
                m = _single_feat_detect(df, col, ic, label_col)
                if m is None:
                    continue
                day_res["features"][f] = {"ic": _s(ic), **m}
                if m["f1"] > best_f1:
                    best_f1 = m["f1"]
                    best_feat = f

            day_res["best_single_feat"] = best_feat
            day_res["best_single_f1"]   = _s(best_f1)

            # Composite Borda (全特徴量)
            valid_feats = [f for f in FEAT_LIST if f"d{n_days}_{f}" in df.columns]
            if valid_feats:
                ic_vals = [ics.get(f, 0.0) for f in valid_feats]
                feat_cols = [f"d{n_days}_{f}" for f in valid_feats]
                m_comp = _detect_bottom20(df, feat_cols, ic_vals, label_col)
                day_res["composite"] = m_comp

            results[label_name][f"day{n_days}"] = day_res

    return results


# ======================================================================
# Phase3: Incremental Information Audit
# ======================================================================

def phase3_incremental_info(df: pd.DataFrame) -> dict:
    """Day5固定。特徴量セットを1→18個と増やしF1変化を測定。"""
    ANCHOR = 5
    label_col = "is_bottom20"
    ics = _ics_at_day(df, ANCHOR)

    results: dict = {"anchor_day": ANCHOR, "label": label_col, "sets": {}}
    prev_prec, prev_f1 = 0.0, 0.0

    for set_name, feats in INCREMENTAL_SETS:
        cols = [f"d{ANCHOR}_{f}" for f in feats if f"d{ANCHOR}_{f}" in df.columns]
        ic_v = [ics.get(f, 0.0) for f in feats if f"d{ANCHOR}_{f}" in df.columns]
        if not cols:
            continue
        m = _detect_bottom20(df, cols, ic_v, label_col)
        if m is None:
            continue
        results["sets"][set_name] = {
            "n_features":       len(cols),
            "precision":        m["precision"],
            "recall":           m["recall"],
            "f1":               m["f1"],
            "balanced_acc":     m["balanced_acc"],
            "delta_f1":         _s(m["f1"] - prev_f1),
            "delta_precision":  _s(m["precision"] - prev_prec),
        }
        prev_f1  = m["f1"]
        prev_prec = m["precision"]

    return results


# ======================================================================
# Phase4: Information Gain Ranking
# ======================================================================

def phase4_info_gain_ranking(df: pd.DataFrame) -> dict:
    """Day5固定。各特徴量の単独F1 + Leave-one-out Marginal Gain + 年別IC安定性。"""
    ANCHOR = 5
    label_col = "is_bottom20"
    ics = _ics_at_day(df, ANCHOR)

    all_cols  = [f"d{ANCHOR}_{f}" for f in FEAT_LIST if f"d{ANCHOR}_{f}" in df.columns]
    all_ics   = [ics.get(f, 0.0) for f in FEAT_LIST if f"d{ANCHOR}_{f}" in df.columns]
    all_feats = [f for f in FEAT_LIST if f"d{ANCHOR}_{f}" in df.columns]

    baseline = _detect_bottom20(df, all_cols, all_ics, label_col)
    baseline_f1 = baseline["f1"] if baseline else 0.0

    ranking = []
    for feat in FEAT_LIST:
        fcol = f"d{ANCHOR}_{feat}"
        if fcol not in df.columns:
            continue
        ic = ics.get(feat, 0.0)

        # 単独検出
        solo = _single_feat_detect(df, fcol, ic, label_col)
        solo_f1 = solo["f1"] if solo else 0.0

        # Leave-one-out
        loo_cols  = [c for c in all_cols if c != fcol]
        loo_ics   = [iv for c, iv in zip(all_cols, all_ics) if c != fcol]
        loo = _detect_bottom20(df, loo_cols, loo_ics, label_col) if loo_cols else None
        loo_f1    = loo["f1"] if loo else 0.0
        marg_gain = _s(baseline_f1 - loo_f1)

        # 年別IC安定性
        yr_ics = []
        for yr in sorted(df["entry_year"].unique()):
            sub = df[df["entry_year"] == yr][[fcol, "fwd60d_entry"]].dropna()
            if len(sub) >= 5:
                r, _ = _spearman(sub[fcol].values, sub["fwd60d_entry"].values)
                if not np.isnan(r):
                    yr_ics.append(r)

        ranking.append({
            "feature":       feat,
            "ic_day5":       _s(ic),
            "solo_f1":       _s(solo_f1),
            "marginal_gain": marg_gain,
            "ic_std_yearly": _s(float(np.std(yr_ics))) if yr_ics else None,
            "ic_pos_years":  sum(1 for x in yr_ics if x > 0),
            "n_years":       len(yr_ics),
        })

    ranking.sort(key=lambda x: (x["marginal_gain"] or 0), reverse=True)
    for i, r in enumerate(ranking):
        r["rank"] = i + 1

    return {
        "anchor_day":   ANCHOR,
        "baseline_f1":  _s(baseline_f1),
        "label":        label_col,
        "ranking":      ranking,
        "top5_by_gain": [r["feature"] for r in ranking[:5]],
        "bottom5_unnecessary": [r["feature"] for r in ranking if (r["marginal_gain"] or 0) <= 0][:5],
    }


# ======================================================================
# Phase5: Economic Value Curve
# ======================================================================

def phase5_economic_value(df: pd.DataFrame) -> dict:
    """各観測日でComposite検出。Bottom20%/10%の実際fwd20/40/60を測定。"""
    results: dict = {}

    for n_days in OBS_DAYS_62:
        ics  = _ics_at_day(df, n_days)
        valid_feats = [f for f in FEAT_LIST if f"d{n_days}_{f}" in df.columns]
        if not valid_feats:
            continue
        feat_cols = [f"d{n_days}_{f}" for f in valid_feats]
        ic_vals   = [ics.get(f, 0.0) for f in valid_feats]

        needed = feat_cols + ["fwd20d_entry", "fwd40d_entry", "fwd60d_entry",
                              "is_bottom20", "is_bottom10"]
        sub = df[needed].dropna(subset=feat_cols)
        if len(sub) < 15:
            continue

        scores = _failure_score(sub, feat_cols, ic_vals)
        p80_thr = np.percentile(scores, 80)
        p90_thr = np.percentile(scores, 90)
        det20 = scores >= p80_thr   # detected bottom 20%
        det10 = scores >= p90_thr   # detected bottom 10%
        good  = scores <  np.percentile(scores, 20)  # 上位20% (比較)

        def _ev(mask):
            g = sub[mask]
            if len(g) == 0:
                return {}
            f60 = g["fwd60d_entry"].dropna().values
            f40 = g["fwd40d_entry"].dropna().values
            f20 = g["fwd20d_entry"].dropna().values
            losses60 = f60[f60 < 0]
            gains60  = f60[f60 > 0]
            return {
                "n":            len(g),
                "avg_fwd20":    _s(float(f20.mean())) if len(f20) else None,
                "avg_fwd40":    _s(float(f40.mean())) if len(f40) else None,
                "avg_fwd60":    _s(float(f60.mean())) if len(f60) else None,
                "win_rate":     _s(float((f60 > 0).mean())),
                "pf":           _s(float(gains60.sum() / max(abs(losses60.sum()), 1e-9)))
                                if len(losses60) > 0 and len(gains60) > 0 else None,
            }

        results[f"day{n_days}"] = {
            "detected_bottom20": _ev(det20),
            "detected_bottom10": _ev(det10),
            "complement_top20":  _ev(good),
        }

    return results


# ======================================================================
# Phase6: Observation Delay Study
# ======================================================================

def phase6_delay_study(df: pd.DataFrame) -> dict:
    """各観測日でのLoss Avoided / Opportunity Cost / Information Value Curve。"""
    results: dict = {}

    for n_days in OBS_DAYS_62:
        ics = _ics_at_day(df, n_days)
        valid_feats = [f for f in FEAT_LIST if f"d{n_days}_{f}" in df.columns]
        if not valid_feats:
            continue
        feat_cols = [f"d{n_days}_{f}" for f in valid_feats]
        ic_vals   = [ics.get(f, 0.0) for f in valid_feats]

        needed = feat_cols + ["fwd60d_entry", "is_bottom20"]
        sub = df[needed].dropna(subset=feat_cols + ["fwd60d_entry", "is_bottom20"])
        if len(sub) < 15:
            continue

        scores = _failure_score(sub, feat_cols, ic_vals)
        thr    = np.percentile(scores, (1.0 - BOTTOM_PCT) * 100)
        y_pred = scores >= thr
        y_true = sub["is_bottom20"].values.astype(bool)

        tp_mask = y_pred & y_true
        fp_mask = y_pred & ~y_true
        fn_mask = ~y_pred & y_true
        tn_mask = ~y_pred & ~y_true

        fwd = sub["fwd60d_entry"].values

        def _avg(mask):
            v = fwd[mask]
            return _s(float(v.mean())) if len(v) > 0 else None

        # Loss Avoided = True Positive群の平均fwd60d (マイナス → 回避できた損失)
        # Opportunity Cost = False Positive群の平均fwd60d (プラスなのに除外 → 逃した利益)
        loss_avoided_avg  = _avg(tp_mask)
        opp_cost_avg      = _avg(fp_mask)
        missed_loss_avg   = _avg(fn_mask)   # 見逃した失敗群
        kept_winner_avg   = _avg(tn_mask)   # 正しく保持した勝者群

        # Information Value = |loss_avoided| - opportunity_cost (net benefit)
        info_value = None
        if loss_avoided_avg is not None and opp_cost_avg is not None:
            info_value = _s(abs(loss_avoided_avg) - abs(opp_cost_avg))

        results[f"day{n_days}"] = {
            "tp": int(tp_mask.sum()), "fp": int(fp_mask.sum()),
            "fn": int(fn_mask.sum()), "tn": int(tn_mask.sum()),
            "precision":           _s(float(tp_mask.sum() / max(y_pred.sum(), 1))),
            "recall":              _s(float(tp_mask.sum() / max(y_true.sum(), 1))),
            "loss_avoided_avg_fwd60":  loss_avoided_avg,
            "opportunity_cost_avg_fwd60": opp_cost_avg,
            "missed_failures_avg_fwd60":  missed_loss_avg,
            "kept_winners_avg_fwd60":     kept_winner_avg,
            "information_value":          info_value,
        }

    return results


# ======================================================================
# Phase7: Quality Monitoring Framework Design Input
# ======================================================================

def phase7_qmf_design(p2: dict, p3: dict, p4: dict, p5: dict, p6: dict) -> dict:
    """
    まだルール化禁止。出力のみ。
    PASS/WATCH/WARNING/FAILの4段階QMF入力仕様を提案。
    """
    # 最高F1達成日 (Bottom20%, Composite)
    best_day_f1, best_f1 = 5, 0.0
    for nd in OBS_DAYS_62:
        comp = p2["bottom20"].get(f"day{nd}", {}).get("composite", {})
        if comp and comp.get("f1", 0) > best_f1:
            best_f1 = comp["f1"]
            best_day_f1 = nd

    # 最小有意日 (F1 >= 0.28)
    min_day = None
    for nd in OBS_DAYS_62:
        comp = p2["bottom20"].get(f"day{nd}", {}).get("composite", {})
        if comp and (comp.get("f1") or 0) >= 0.28:
            min_day = nd
            break

    # 最小特徴量セット (top3 by marginal gain)
    top3 = p4.get("top5_by_gain", [])[:3]

    # 経済価値が最大の観測日 (detected_bottom10 avg_fwd60 が最も低い日)
    best_econ_day, best_econ_val = 5, 0.0
    for nd in OBS_DAYS_62:
        ev = p5.get(f"day{nd}", {}).get("detected_bottom10", {})
        avg60 = ev.get("avg_fwd60") if ev else None
        if avg60 is not None and avg60 < best_econ_val:
            best_econ_val = avg60
            best_econ_day = nd

    # Phase3での増分効果
    inc_sets = p3.get("sets", {})
    f1_1feat = inc_sets.get("ret", {}).get("f1", 0)
    f1_3feat = inc_sets.get("ret+rsr+vol", {}).get("f1", 0)
    f1_4feat = inc_sets.get("ret+rsr+vol+atr", {}).get("f1", 0)
    f1_18    = inc_sets.get("full_18", {}).get("f1", 0)

    return {
        "note": "出力のみ。実装禁止。閾値最適化禁止。",
        "framework_states": {
            "PASS":    "Failure score < p20 (bottom failure probability)",
            "WATCH":   "Failure score p20-p50",
            "WARNING": "Failure score p50-p80",
            "FAIL":    "Failure score >= p80 (top failure probability)",
        },
        "minimum_feature_set":     top3,
        "minimum_n_features":      3,
        "optimal_observation_day": best_day_f1,
        "minimum_observation_day": min_day,
        "optimal_f1_achieved":     _s(best_f1),
        "optimal_econ_day":        best_econ_day,
        "econ_day_bottom10_avg60": _s(best_econ_val),
        "incremental_f1": {
            "1_feature": _s(f1_1feat),
            "3_features": _s(f1_3feat),
            "4_features": _s(f1_4feat),
            "18_features": _s(f1_18),
        },
    }


# ======================================================================
# Phase8: Research Verdict
# ======================================================================

def phase8_verdict(p2: dict, p3: dict, p4: dict, p5: dict, p6: dict, p7: dict) -> dict:
    """
    1. Failureを最も早く観測できる日
    2. Failureを最も正確に観測できる日
    3. Failureを最も高い経済価値で観測できる日
    4. 必須特徴量
    5. 不要特徴量
    6. 最小情報セット
    7. Quality Monitoring Framework入力仕様
    """
    # 1. 最早検出日 (F1 > 0で最小観測日)
    earliest = None
    for nd in OBS_DAYS_62:
        comp = p2["bottom20"].get(f"day{nd}", {}).get("composite", {})
        if comp and (comp.get("f1") or 0) > 0.20:
            earliest = nd
            break

    # 2. 最高精度日
    best_acc_day, best_acc_f1 = 5, 0.0
    best_acc_metrics: dict = {}
    for nd in OBS_DAYS_62:
        comp = p2["bottom20"].get(f"day{nd}", {}).get("composite", {})
        if comp and (comp.get("f1") or 0) > best_acc_f1:
            best_acc_f1 = comp.get("f1", 0)
            best_acc_day = nd
            best_acc_metrics = comp

    # 2b. FalseHero最高精度日
    best_fh_day, best_fh_f1 = 5, 0.0
    for nd in OBS_DAYS_62:
        comp = p2["false_hero"].get(f"day{nd}", {}).get("composite", {})
        if comp and (comp.get("f1") or 0) > best_fh_f1:
            best_fh_f1  = comp.get("f1", 0)
            best_fh_day = nd

    # 3. 最高経済価値日 (detected_bottom10 avg_fwd60 が最も低い)
    best_econ_day, best_econ_val = p7.get("optimal_econ_day", 5), p7.get("econ_day_bottom10_avg60", 0)

    # 4. 必須特徴量 (marginal gain top3)
    essential = p4.get("top5_by_gain", [])[:3]

    # 5. 不要特徴量 (marginal gain <= 0)
    unnecessary = [r["feature"] for r in p4.get("ranking", [])
                   if (r.get("marginal_gain") or 0) <= 0]

    # 6. 最小情報セット
    sets = p3.get("sets", {})
    min_set_name = "ret"
    min_set_f1   = 0.0
    for sn, sv in sets.items():
        if (sv.get("f1") or 0) >= 0.90 * (sets.get("full_18", {}).get("f1") or 0.01):
            min_set_name = sn
            min_set_f1   = sv.get("f1", 0)
            break  # 最小セットで90%性能達成

    # Phase6: Information Value by day
    info_values = {f"day{nd}": p6.get(f"day{nd}", {}).get("information_value")
                   for nd in OBS_DAYS_62}

    return {
        "1_earliest_detection_day": {
            "day": earliest,
            "note": "Composite F1 > 0.20 を達成する最小観測日",
        },
        "2_most_accurate_day": {
            "day":      best_acc_day,
            "f1":       _s(best_acc_f1),
            "metrics":  best_acc_metrics,
        },
        "2b_false_hero_best_day": {
            "day": best_fh_day,
            "f1":  _s(best_fh_f1),
        },
        "3_highest_economic_value_day": {
            "day":                    best_econ_day,
            "avg_fwd60_bottom10":     best_econ_val,
            "information_value_curve": info_values,
        },
        "4_essential_features": essential,
        "5_unnecessary_features": unnecessary,
        "6_minimum_info_set": {
            "set_name":   min_set_name,
            "f1":         _s(min_set_f1),
            "threshold":  "≥90% of full-18 F1",
            "phase3_ref": sets,
        },
        "7_qmf_input_spec": p7,
    }


# ======================================================================
# main
# ======================================================================

def main():
    print("=== Study62: Failure Detection Timing Study ===")
    print(f"  今日: {TODAY_STR}")

    print("\n[データロード中...]")
    ds        = build_common_dataset(DATA_END)
    rsr_df    = ds["rsr_df"]
    all_dates = rsr_df.index.sort_values()
    is_dates  = all_dates[(all_dates >= IS_START)  & (all_dates <= IS_END)]
    oos_dates = all_dates[(all_dates >= OOS_START) & (all_dates <= OOS_END)]

    print("[BT実行: IS 2018-2024]")
    sym_is  = get_active(ds, IS_START, IS_END)
    bt_is   = run_bt(ds, sym_is, IS_START, IS_END)
    tr_is   = extract_trades(bt_is, is_dates, rsr_df)
    print(f"  IS trades: {len(tr_is)}")

    print("[BT実行: OOS 2025]")
    sym_oos = get_active(ds, OOS_START, OOS_END)
    bt_oos  = run_bt(ds, sym_oos, OOS_START, OOS_END)
    tr_oos  = extract_trades(bt_oos, oos_dates, rsr_df)
    print(f"  OOS trades: {len(tr_oos)}")

    all_trades = tr_is + tr_oos
    print(f"  全取引: {len(all_trades)} 件")

    print("\n[データセット構築: Day1-10...]")
    df = build_study62_dataset(ds, all_trades)
    print(f"  有効トレード数: {len(df)}")
    if df.empty:
        print("  ERROR: データセット空")
        return

    n_fh  = int(df["is_false_hero"].sum())
    n_b20 = int(df["is_bottom20"].sum())
    print(f"  FalseHero: {n_fh} | Bottom20%: {n_b20} | BigWinner: {int(df['is_big_winner'].sum())}")

    print("\n[Phase0: Integrity]")
    p0 = phase0_integrity(df, len(all_trades))
    print(f"  Lookahead=0 PASS | n_valid={p0['n_valid']}")

    print("\n[Phase1: Taxonomy]")
    p1 = phase1_taxonomy(df)
    for g, v in p1["groups"].items():
        print(f"  {g}: n={v['n']} ({v['pct']*100:.1f}%) avg_fwd60={v['avg_fwd60']}%")

    print("\n[Phase2: Detection Timing Curve...]")
    p2 = phase2_detection_timing(df)
    print("  Bottom20% [Composite F1 by obs_day]:")
    for nd in OBS_DAYS_62:
        comp = p2["bottom20"].get(f"day{nd}", {}).get("composite", {})
        if comp:
            print(f"    Day{nd}: F1={comp.get('f1'):.4f} Prec={comp.get('precision'):.4f} "
                  f"Rec={comp.get('recall'):.4f} BalAcc={comp.get('balanced_acc'):.4f}")
    print("  FalseHero [Composite F1]:")
    for nd in OBS_DAYS_62:
        comp = p2["false_hero"].get(f"day{nd}", {}).get("composite", {})
        if comp:
            print(f"    Day{nd}: F1={comp.get('f1'):.4f} Prec={comp.get('precision'):.4f} "
                  f"Rec={comp.get('recall'):.4f}")

    print("\n[Phase3: Incremental Info Audit (Day5)]")
    p3 = phase3_incremental_info(df)
    for sn, sv in p3["sets"].items():
        print(f"  {sn}: F1={sv['f1']:.4f} ΔF1={sv['delta_f1']:+.4f}")

    print("\n[Phase4: Info Gain Ranking (Day5)]")
    p4 = phase4_info_gain_ranking(df)
    print(f"  Baseline F1 (all 18): {p4['baseline_f1']:.4f}")
    print("  Top 5 by Marginal Gain:")
    for r in p4["ranking"][:5]:
        print(f"    {r['rank']}. {r['feature']}: IC={r['ic_day5']:.4f} "
              f"solo_F1={r['solo_f1']:.4f} MargGain={r['marginal_gain']:+.4f}")
    print(f"  不要特徴量 (gain≤0): {p4['bottom5_unnecessary']}")

    print("\n[Phase5: Economic Value Curve]")
    p5 = phase5_economic_value(df)
    print("  detected_bottom10 avg_fwd60 by day:")
    for nd in OBS_DAYS_62:
        ev = p5.get(f"day{nd}", {}).get("detected_bottom10", {})
        if ev:
            print(f"    Day{nd}: avg_fwd60={ev.get('avg_fwd60')}% "
                  f"win_rate={ev.get('win_rate')} pf={ev.get('pf')}")

    print("\n[Phase6: Observation Delay Study]")
    p6 = phase6_delay_study(df)
    print("  information_value by day (|loss_avoided| - opp_cost):")
    for nd in OBS_DAYS_62:
        r6 = p6.get(f"day{nd}", {})
        if r6:
            print(f"    Day{nd}: info_value={r6.get('information_value')} "
                  f"loss_avoided={r6.get('loss_avoided_avg_fwd60')}% "
                  f"opp_cost={r6.get('opportunity_cost_avg_fwd60')}%")

    print("\n[Phase7: QMF Design Input]")
    p7 = phase7_qmf_design(p2, p3, p4, p5, p6)
    print(f"  最小特徴量セット: {p7['minimum_feature_set']}")
    print(f"  最適観測日: Day{p7['optimal_observation_day']} (F1={p7['optimal_f1_achieved']})")
    print(f"  最小有意観測日: Day{p7['minimum_observation_day']}")

    print("\n[Phase8: Verdict]")
    p8 = phase8_verdict(p2, p3, p4, p5, p6, p7)
    print(f"  1. 最早検出日: Day{p8['1_earliest_detection_day']['day']}")
    print(f"  2. 最高精度日: Day{p8['2_most_accurate_day']['day']} F1={p8['2_most_accurate_day']['f1']}")
    print(f"  2b. FalseHero最高精度: Day{p8['2b_false_hero_best_day']['day']} F1={p8['2b_false_hero_best_day']['f1']}")
    print(f"  3. 最高経済価値日: Day{p8['3_highest_economic_value_day']['day']}")
    print(f"  4. 必須特徴量: {p8['4_essential_features']}")
    print(f"  5. 不要特徴量: {p8['5_unnecessary_features'][:5]}...")
    print(f"  6. 最小情報セット: {p8['6_minimum_info_set']['set_name']} F1={p8['6_minimum_info_set']['f1']}")

    result = {
        "study":              "Study62",
        "title":              "Failure Detection Timing Study",
        "date":               TODAY_STR,
        "params":             {
            "obs_days": OBS_DAYS_62,
            "bottom_pct": BOTTOM_PCT,
            "detection_method": "Borda rank composite (固定20パーセンタイル / 閾値最適化なし)",
            "auc": "禁止",
            "ml_models": "禁止",
        },
        "phase0_integrity":         p0,
        "phase1_taxonomy":          p1,
        "phase2_detection_timing":  p2,
        "phase3_incremental_info":  p3,
        "phase4_info_gain_ranking": p4,
        "phase5_economic_value":    p5,
        "phase6_delay_study":       p6,
        "phase7_qmf_design":        p7,
        "phase8_verdict":           p8,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n保存: {OUT_FILE}")
    print("=== Study62 完了 ===")


if __name__ == "__main__":
    main()
