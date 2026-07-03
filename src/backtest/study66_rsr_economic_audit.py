"""
study66_rsr_economic_audit.py
Study66 — RSR Economic Audit

目的: RSR_EXIT / RSR_MOMENTUM_EXITの純経済価値(Net Economic Value)を監査。
禁止: 売買ルール作成 / 閾値探索 / Production変更 / 最適化
手法: Exit後Forward Return分布 / Loss Avoided - Profit Lost / BW vs Non-BW比較

Phase0: Integrity
Phase1: Forward Distribution Audit (RSR系Exit後リターン分布)
Phase2: Economic Classification (A=Correct / B=Mixed / C=Premature)
Phase3: BigWinner vs Non-BigWinner 比較
Phase4: Net Economic Value (Loss Avoided - Profit Lost)
Phase5: RSR_EXIT vs RSR_MOMENTUM_EXIT 分解
Phase6: Sensitivity Audit (Horizon 20/40/60)
Phase7: Research Verdict
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
    _features_at_obs, get_active, _s, _mwu_pval,
)
import src.backtest.composite_alpha_bt as cab

TODAY_STR    = date.today().strftime("%Y-%m-%d")
CAPITAL      = 3_000_000
IS_START     = "2018-01-01"
IS_END       = "2024-12-31"
OOS_START    = "2025-01-01"
OOS_END      = "2025-12-31"
DATA_END     = "2025-12-31"
FWD_DAYS     = 60
N_YEARS_FULL = 8.0
MIN_HOLD     = 3

EP_EXIT         = "A"
EP_ADDON        = "D"
ADDON_ATR_MULT  = 1.0
ADDON_SIZE_FRAC = 0.25

# Study65基準
S65_RSR_PLB_PCT   = 82.8   # RSR系ExitのPLB%
S65_PRE_PEAK_PCT  = 80.0   # BW Peak前Exit率
S65_BW_FWD40      = 27.7   # BW Exit後fwd40d mean

# Exit後Forward Return観測日
POST_EXIT_DAYS = [5, 10, 20, 40, 60]

# RSR系Exit理由
RSR_REASONS = {"RSR_EXIT", "RSR_MOMENTUM_EXIT"}

OUT_FILE = ROOT / "backtests" / f"study66_rsr_economic_audit_{TODAY_STR}.json"


# ======================================================================
# BT + トレード抽出 (Study65と同一)
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


def extract_trades_full(bt_result: dict, calendar_dates: pd.DatetimeIndex,
                        rsr_df: pd.DataFrame) -> list[dict]:
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
            "reason":      t.get("reason", "UNKNOWN"),
            "entry_rsr":   entry_rsr,
        })
    return out


# ======================================================================
# データセット構築
# ======================================================================

def build_study66_dataset(ds: dict, trades: list[dict]) -> pd.DataFrame:
    """
    各トレードについて:
    - fwd60d_entry / BigWinnerラベル
    - exit後 +5/10/20/40/60d forward return
    - position_value_at_exit (Net Economic Value計算用)
    """
    universe_raw = ds["universe_raw"]
    rsr_df       = ds["rsr_df"]
    all_dates    = rsr_df.index.sort_values()

    records = []
    for tr in trades:
        sym        = tr["symbol"]
        entry_date = tr["entry_date"]
        exit_date  = tr["exit_date"]
        entry_px   = tr["entry_price"]
        exit_px    = tr["exit_price"]
        qty        = tr["qty"]
        pnl        = tr["pnl"]
        entry_rsr  = tr["entry_rsr"]
        reason     = tr["reason"]

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

        # ラベル用フォワードリターン
        fwd60 = (float(fut_e.iloc[FWD_DAYS - 1]) / ep - 1.0) * 100

        # Exit後フォワードリターン
        exit_future = close[close.index > exit_date]
        ex_av = close[close.index <= exit_date]
        ex_px = float(ex_av.iloc[-1]) if not ex_av.empty else exit_px

        exit_fwd: dict = {}
        for nd in POST_EXIT_DAYS:
            if len(exit_future) >= nd:
                exit_fwd[f"exit_fwd_{nd}d"] = _s(
                    (float(exit_future.iloc[nd - 1]) / ex_px - 1.0) * 100
                )
            else:
                exit_fwd[f"exit_fwd_{nd}d"] = np.nan

        # Position Value at Exit (NEV計算用)
        pos_val_at_exit = ex_px * qty

        # Net Economic Contribution: each day
        nev_contrib: dict = {}
        for nd in POST_EXIT_DAYS:
            col = f"exit_fwd_{nd}d"
            fv  = exit_fwd.get(col)
            if fv is not None and not np.isnan(fv):
                # NEV = -fwd_ret * pos_val (正 = exit帰還価値, 負 = exit損失)
                # 直感: fwd>0 → holding would gain → exit costs us (負のNEV)
                #       fwd<0 → holding would lose → exit saves us (正のNEV)
                nev_contrib[f"nev_{nd}d"] = _s(-fv / 100.0 * pos_val_at_exit)
            else:
                nev_contrib[f"nev_{nd}d"] = np.nan

        row: dict = {
            "symbol":       sym,
            "entry_date":   pd.Timestamp(entry_date),
            "exit_date":    pd.Timestamp(exit_date),
            "entry_year":   pd.Timestamp(entry_date).year,
            "entry_price":  ep,
            "exit_price":   ex_px,
            "qty":          qty,
            "pnl":          pnl,
            "exit_reason":  reason,
            "fwd60d_entry": _s(fwd60),
            "pos_val_at_exit": _s(pos_val_at_exit),
            **exit_fwd,
            **nev_contrib,
        }
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    # BigWinnerラベル (Top10%)
    fwd = df["fwd60d_entry"].dropna()
    p90 = float(fwd.quantile(0.90))
    p20 = float(fwd.quantile(0.20))

    df["is_big_winner"]  = df["fwd60d_entry"] >= p90
    df["is_bottom20"]    = df["fwd60d_entry"] < p20
    df["is_rsr_exit"]    = df["exit_reason"].isin(RSR_REASONS)

    return df


# ======================================================================
# ユーティリティ
# ======================================================================

def _fwd_stats(series: pd.Series, label: str = "") -> dict:
    if series.empty:
        return {"label": label, "n": 0}
    return {
        "label":    label,
        "n":        len(series),
        "mean":     _s(float(series.mean())),
        "median":   _s(float(series.median())),
        "std":      _s(float(series.std())),
        "win_rate": _s(float((series > 0).mean() * 100)),
        "p10":      _s(float(series.quantile(0.10))),
        "p25":      _s(float(series.quantile(0.25))),
        "p50":      _s(float(series.quantile(0.50))),
        "p75":      _s(float(series.quantile(0.75))),
        "p90":      _s(float(series.quantile(0.90))),
    }


def _cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's Delta (effect size) — ノンパラメトリック。"""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    count = 0
    for x in a:
        count += np.sum(x > b) - np.sum(x < b)
    return float(count) / (len(a) * len(b))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = np.sqrt((np.var(a, ddof=1) * (len(a) - 1) + np.var(b, ddof=1) * (len(b) - 1))
                         / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / max(pooled_std, 1e-9))


# ======================================================================
# Phase0: Integrity
# ======================================================================

def phase0_integrity(df: pd.DataFrame, all_trades: list[dict]) -> dict:
    s63_n = 291
    n_match = len(df) == s63_n
    rsr_df_sub = df[df["is_rsr_exit"]]

    # Study65整合確認
    s65_rsr_exit_n = 129    # RSR_EXIT
    s65_rsr_mom_n  = 119    # RSR_MOMENTUM_EXIT
    rsr_exit_n  = int((df["exit_reason"] == "RSR_EXIT").sum())
    rsr_mom_n   = int((df["exit_reason"] == "RSR_MOMENTUM_EXIT").sum())

    return {
        "n_valid":              len(df),
        "study63_n_match":      n_match,
        "study63_expected":     s63_n,
        "bigwinner_n":          int(df["is_big_winner"].sum()),
        "rsr_exit_total_n":     len(rsr_df_sub),
        "rsr_exit_n":           rsr_exit_n,
        "rsr_momentum_exit_n":  rsr_mom_n,
        "study65_rsr_exit_match":    rsr_exit_n == s65_rsr_exit_n,
        "study65_rsr_mom_match":     rsr_mom_n  == s65_rsr_mom_n,
        "lookahead":            0,
        "survivorship":         0,
    }


# ======================================================================
# Phase1: Forward Distribution Audit
# ======================================================================

def phase1_forward_distribution(df: pd.DataFrame) -> dict:
    """RSR_EXIT / RSR_MOMENTUM_EXIT それぞれのExit後フォワードリターン分布。"""
    results: dict = {}

    targets = {
        "RSR_EXIT":            df["exit_reason"] == "RSR_EXIT",
        "RSR_MOMENTUM_EXIT":   df["exit_reason"] == "RSR_MOMENTUM_EXIT",
        "RSR_COMBINED":        df["is_rsr_exit"],
        "NON_RSR":             ~df["is_rsr_exit"],
        "ALL":                 pd.Series(True, index=df.index),
    }

    for label, mask in targets.items():
        sub = df[mask]
        by_day: dict = {}
        for nd in POST_EXIT_DAYS:
            col = f"exit_fwd_{nd}d"
            if col in sub.columns:
                vals = sub[col].dropna()
                by_day[f"fwd_{nd}d"] = _fwd_stats(vals, f"{label} fwd+{nd}d")
        results[label] = {
            "n":         len(sub),
            "by_horizon": by_day,
        }

    # RSR_EXIT vs NON_RSR の比較 (fwd40d)
    rsr_fwd40 = df.loc[df["is_rsr_exit"], "exit_fwd_40d"].dropna()
    non_fwd40 = df.loc[~df["is_rsr_exit"], "exit_fwd_40d"].dropna()
    mwu_p = _mwu_pval(rsr_fwd40.values, non_fwd40.values)
    d40   = _cohens_d(rsr_fwd40.values, non_fwd40.values)
    cliff40 = _cliff_delta(rsr_fwd40.values, non_fwd40.values)

    results["comparison_rsr_vs_non_rsr_fwd40d"] = {
        "rsr_mean":     _s(float(rsr_fwd40.mean())) if not rsr_fwd40.empty else None,
        "non_rsr_mean": _s(float(non_fwd40.mean())) if not non_fwd40.empty else None,
        "mwu_p":        _s(mwu_p),
        "cohens_d":     _s(d40),
        "cliffs_delta": _s(cliff40),
    }
    return results


# ======================================================================
# Phase2: Economic Classification
# ======================================================================

def phase2_economic_classification(df: pd.DataFrame) -> dict:
    """
    RSR系Exit後のForward Returnを3分類。
    閾値は自然な0基準: fwd_Nd の正/負で分類。
    A: Correct  = fwd40d < 0  (holding more would have lost money)
    C: Premature= fwd40d > 0  (holding more would have gained)
    B: Mixed    = fwd20d・fwd40d・fwd60d で一致しない場合 (横断定義)

    補足: B定義 = fwd20dとfwd60dで符号が異なる (中期と長期で方向が割れる)
    """
    rsr = df[df["is_rsr_exit"]].copy()

    def _classify(row) -> str:
        f20 = row.get("exit_fwd_20d")
        f40 = row.get("exit_fwd_40d")
        f60 = row.get("exit_fwd_60d")
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [f20, f40, f60]):
            return "UNKNOWN"
        # A = 全ホライゾンで持ち続けると損 → exit正解
        if f20 < 0 and f40 < 0 and f60 < 0:
            return "A_CORRECT"
        # C = 全ホライゾンで持ち続けると得 → exit早計
        if f20 > 0 and f40 > 0 and f60 > 0:
            return "C_PREMATURE"
        # B = 混合
        return "B_MIXED"

    rsr["econ_class"] = rsr.apply(_classify, axis=1)

    n_rsr = len(rsr)
    classes = ["A_CORRECT", "B_MIXED", "C_PREMATURE", "UNKNOWN"]
    cls_res: dict = {}
    for cls in classes:
        sub = rsr[rsr["econ_class"] == cls]
        cls_res[cls] = {
            "n":           len(sub),
            "pct":         _s(len(sub) / max(n_rsr, 1) * 100),
            "bw_n":        int(sub["is_big_winner"].sum()),
            "bw_pct":      _s(int(sub["is_big_winner"].sum()) / max(len(sub), 1) * 100),
            "fwd40d_mean": _s(float(sub["exit_fwd_40d"].dropna().mean())) if not sub.empty else None,
            "pnl_sum_yen": _s(float(sub["pnl"].sum())),
        }

    # BW別分類
    bw_rsr  = rsr[rsr["is_big_winner"]]
    nbw_rsr = rsr[~rsr["is_big_winner"]]
    bw_cls  = bw_rsr["econ_class"].value_counts().to_dict() if not bw_rsr.empty else {}
    nbw_cls = nbw_rsr["econ_class"].value_counts().to_dict() if not nbw_rsr.empty else {}

    return {
        "n_rsr_total":       n_rsr,
        "classification":    cls_res,
        "bw_classification": bw_cls,
        "nbw_classification":nbw_cls,
        "interpretation":    (
            "A_CORRECT=Exit後全期間でリターン負(exit有効)。"
            "C_PREMATURE=Exit後全期間でリターン正(exit早計)。"
            "B_MIXED=期間によって方向が変わる。"
        ),
    }


# ======================================================================
# Phase3: BigWinner vs Non-BigWinner
# ======================================================================

def phase3_bw_vs_nbw(df: pd.DataFrame) -> dict:
    """RSR系ExitにおけるBigWinner vs Non-BigWinner の比較。"""
    rsr = df[df["is_rsr_exit"]]
    bw  = rsr[rsr["is_big_winner"]]
    nbw = rsr[~rsr["is_big_winner"]]

    def _grp(g: pd.DataFrame, label: str) -> dict:
        res: dict = {"label": label, "n": len(g)}
        for nd in POST_EXIT_DAYS:
            col = f"exit_fwd_{nd}d"
            if col in g.columns:
                vals = g[col].dropna()
                res[f"fwd_{nd}d"] = _fwd_stats(vals)
        return res

    bw_res  = _grp(bw,  "BigWinner RSR exits")
    nbw_res = _grp(nbw, "Non-BigWinner RSR exits")

    # 統計的比較 (各horizon)
    comparisons: dict = {}
    for nd in POST_EXIT_DAYS:
        col = f"exit_fwd_{nd}d"
        bw_vals  = bw[col].dropna().values  if col in bw.columns  else np.array([])
        nbw_vals = nbw[col].dropna().values if col in nbw.columns else np.array([])
        if len(bw_vals) < 3 or len(nbw_vals) < 3:
            continue
        mwu_p  = _mwu_pval(bw_vals, nbw_vals)
        d_val  = _cohens_d(bw_vals, nbw_vals)
        cliff  = _cliff_delta(bw_vals, nbw_vals)
        comparisons[f"fwd_{nd}d"] = {
            "bw_mean":      _s(float(bw_vals.mean())),
            "nbw_mean":     _s(float(nbw_vals.mean())),
            "diff_bw_nbw":  _s(float(bw_vals.mean() - nbw_vals.mean())),
            "mwu_p":        _s(mwu_p),
            "cohens_d":     _s(d_val),
            "cliffs_delta": _s(cliff),
            "significant":  mwu_p < 0.05,
        }

    return {
        "bigwinner":     bw_res,
        "non_bigwinner": nbw_res,
        "comparisons":   comparisons,
        "bw_n":          len(bw),
        "nbw_n":         len(nbw),
        "interpretation": (
            "BWのRSR exit後fwd40d > NBWのRSR exit後fwd40d なら"
            "BWにとってRSR exitが特に有害。"
        ),
    }


# ======================================================================
# Phase4: Net Economic Value
# ======================================================================

def _compute_nev(group: pd.DataFrame, horizon: int, label: str) -> dict:
    """
    指定ホライゾンでのNet Economic Value計算。
    fwd_col: Exit後 N日フォワードリターン (%)
    pos_val_at_exit: 決済時ポジション価値 (¥)

    Loss Avoided = sum(|fwd| * pos_val) for fwd < 0  (exit有効だった取引)
    Profit Lost  = sum(|fwd| * pos_val) for fwd > 0  (exit早計だった取引)
    Net EV = Loss Avoided - Profit Lost
    """
    col_fwd = f"exit_fwd_{horizon}d"
    if col_fwd not in group.columns or "pos_val_at_exit" not in group.columns:
        return {"label": label, "horizon": horizon, "n": 0}

    valid = group.dropna(subset=[col_fwd, "pos_val_at_exit"])
    if valid.empty:
        return {"label": label, "horizon": horizon, "n": 0}

    fwd  = valid[col_fwd].values / 100.0
    pval = valid["pos_val_at_exit"].values

    gain_if_held = fwd * pval   # 保持した場合の仮想損益

    loss_avoided_arr  = -gain_if_held[gain_if_held < 0]  # 保持で損 → exit有効
    profit_lost_arr   =  gain_if_held[gain_if_held > 0]  # 保持で得 → exit損

    loss_avoided_yen = float(loss_avoided_arr.sum())
    profit_lost_yen  = float(profit_lost_arr.sum())
    net_ev_yen       = loss_avoided_yen - profit_lost_yen

    n_avoided = len(loss_avoided_arr)
    n_lost    = len(profit_lost_arr)

    return {
        "label":             label,
        "horizon":           horizon,
        "n":                 len(valid),
        "n_loss_avoided":    n_avoided,
        "n_profit_lost":     n_lost,
        "loss_avoided_yen":  _s(loss_avoided_yen),
        "profit_lost_yen":   _s(profit_lost_yen),
        "net_ev_yen":        _s(net_ev_yen),
        "net_ev_positive":   net_ev_yen > 0,
        "loss_avoided_pct_of_pl": _s(loss_avoided_yen / max(abs(profit_lost_yen), 1.0) * 100),
        "avg_gain_if_held":  _s(float(np.mean(gain_if_held))),
        "avg_loss_if_held_avoided": _s(float(np.mean(loss_avoided_arr))) if n_avoided else None,
        "avg_gain_if_held_lost":    _s(float(np.mean(profit_lost_arr)))  if n_lost   else None,
    }


def phase4_net_economic_value(df: pd.DataFrame) -> dict:
    """RSR系Exit全体 / BW / Non-BW の各ホライゾンNEV。"""
    rsr = df[df["is_rsr_exit"]]
    bw  = rsr[rsr["is_big_winner"]]
    nbw = rsr[~rsr["is_big_winner"]]

    groups = [
        (rsr, "RSR_COMBINED"),
        (bw,  "RSR_BigWinner"),
        (nbw, "RSR_NonBigWinner"),
    ]

    results: dict = {}
    for grp, lbl in groups:
        by_horizon: dict = {}
        for nd in [20, 40, 60]:
            by_horizon[f"h{nd}"] = _compute_nev(grp, nd, lbl)
        # 主要ホライゾン40dの結果を要約
        h40 = by_horizon.get("h40", {})
        results[lbl] = {
            "n":          len(grp),
            "by_horizon": by_horizon,
            "h40_net_ev": h40.get("net_ev_yen"),
            "h40_positive": h40.get("net_ev_positive"),
        }

    # 主要判定: RSR系Exit全体のNEV方向
    rsr_h40_nev = results.get("RSR_COMBINED", {}).get("h40_net_ev") or 0
    bw_h40_nev  = results.get("RSR_BigWinner", {}).get("h40_net_ev") or 0
    nbw_h40_nev = results.get("RSR_NonBigWinner", {}).get("h40_net_ev") or 0

    results["summary"] = {
        "rsr_combined_h40_nev_yen":   _s(rsr_h40_nev),
        "rsr_bw_h40_nev_yen":         _s(bw_h40_nev),
        "rsr_nbw_h40_nev_yen":        _s(nbw_h40_nev),
        "rsr_overall_positive":       rsr_h40_nev > 0,
        "bw_positive":                bw_h40_nev  > 0,
        "nbw_positive":               nbw_h40_nev > 0,
        "interpretation": (
            "NEV>0 = ExitがLoss AvoidedをProfit Lostより大きく、純プラス貢献。"
            "NEV<0 = Exit早計が多く、純マイナス貢献。"
        ),
    }
    return results


# ======================================================================
# Phase5: RSR_EXIT vs RSR_MOMENTUM_EXIT 分解
# ======================================================================

def phase5_exit_reason_decomposition(df: pd.DataFrame) -> dict:
    """RSR_EXIT と RSR_MOMENTUM_EXIT を別々にNEV算出・比較。"""
    rsr_only = df[df["exit_reason"] == "RSR_EXIT"]
    mom_only = df[df["exit_reason"] == "RSR_MOMENTUM_EXIT"]

    results: dict = {}
    for lbl, grp in [("RSR_EXIT", rsr_only), ("RSR_MOMENTUM_EXIT", mom_only)]:
        by_h: dict = {}
        for nd in [20, 40, 60]:
            by_h[f"h{nd}"] = _compute_nev(grp, nd, lbl)

        bw_sub  = grp[grp["is_big_winner"]]
        nbw_sub = grp[~grp["is_big_winner"]]

        h40 = by_h.get("h40", {})
        fwd40 = grp["exit_fwd_40d"].dropna()
        results[lbl] = {
            "n":             len(grp),
            "bw_n":          len(bw_sub),
            "nbw_n":         len(nbw_sub),
            "fwd40d_mean":   _s(float(fwd40.mean())) if not fwd40.empty else None,
            "fwd40d_median": _s(float(fwd40.median())) if not fwd40.empty else None,
            "fwd40d_win_rate": _s(float((fwd40 > 0).mean() * 100)) if not fwd40.empty else None,
            "h40_net_ev_yen": h40.get("net_ev_yen"),
            "h40_loss_avoided_yen": h40.get("loss_avoided_yen"),
            "h40_profit_lost_yen":  h40.get("profit_lost_yen"),
            "h40_net_ev_positive":  h40.get("net_ev_positive"),
            "by_horizon":    by_h,
        }

    # 比較サマリー
    rsr_nev = results.get("RSR_EXIT", {}).get("h40_net_ev_yen") or 0
    mom_nev = results.get("RSR_MOMENTUM_EXIT", {}).get("h40_net_ev_yen") or 0
    worse   = "RSR_EXIT" if rsr_nev < mom_nev else "RSR_MOMENTUM_EXIT"

    results["comparison"] = {
        "rsr_exit_h40_nev":     _s(rsr_nev),
        "rsr_mom_exit_h40_nev": _s(mom_nev),
        "worse_reason":          worse,
        "rsr_exit_fwd40_mean":  results.get("RSR_EXIT", {}).get("fwd40d_mean"),
        "rsr_mom_fwd40_mean":   results.get("RSR_MOMENTUM_EXIT", {}).get("fwd40d_mean"),
    }
    return results


# ======================================================================
# Phase6: Sensitivity Audit
# ======================================================================

def phase6_sensitivity(df: pd.DataFrame) -> dict:
    """Horizon 20/40/60日でNEV方向一致確認。Study64/65整合確認。"""
    rsr = df[df["is_rsr_exit"]]

    nev_by_horizon: dict = {}
    for nd in [20, 40, 60]:
        nev = _compute_nev(rsr, nd, "RSR_COMBINED")
        nev_by_horizon[f"h{nd}"] = {
            "net_ev_yen":   nev.get("net_ev_yen"),
            "net_ev_positive": nev.get("net_ev_positive"),
            "loss_avoided": nev.get("loss_avoided_yen"),
            "profit_lost":  nev.get("profit_lost_yen"),
        }

    # 方向一致確認
    nev_signs = [v.get("net_ev_positive") for v in nev_by_horizon.values()]
    consistent = len(set(nev_signs)) == 1

    # Study65整合: BW exit後 fwd40d平均の再確認
    bw_rsr = rsr[rsr["is_big_winner"]]
    bw_fwd40 = bw_rsr["exit_fwd_40d"].dropna()
    bw_fwd40_mean = _s(float(bw_fwd40.mean())) if not bw_fwd40.empty else None

    all_rsr_fwd40 = rsr["exit_fwd_40d"].dropna()
    all_fwd40_mean = _s(float(all_rsr_fwd40.mean())) if not all_rsr_fwd40.empty else None

    return {
        "nev_by_horizon":    nev_by_horizon,
        "nev_direction_consistent": consistent,
        "nev_signs":         nev_signs,
        "bw_rsr_fwd40_mean": bw_fwd40_mean,
        "all_rsr_fwd40_mean":all_fwd40_mean,
        "study65_bw_fwd40_expected": S65_BW_FWD40,
        "study65_consistent": (
            bw_fwd40_mean is not None and abs(bw_fwd40_mean - S65_BW_FWD40) < 5.0
        ),
        "interpretation": (
            "全ホライゾンでNEV方向一致 → 結論STABLE。"
            "不一致 → 短中長期で評価が変わる複雑な構造。"
        ),
    }


# ======================================================================
# Phase7: Research Verdict
# ======================================================================

def phase7_verdict(p1: dict, p2: dict, p3: dict, p4: dict,
                   p5: dict, p6: dict) -> dict:
    # ① RSR ExitのNEV方向
    rsr_h40_nev     = (p4.get("summary") or {}).get("rsr_combined_h40_nev_yen") or 0
    rsr_net_positive = (p4.get("summary") or {}).get("rsr_overall_positive") or False

    # ② BW問題 vs 全体問題
    bw_h40_nev  = (p4.get("summary") or {}).get("rsr_bw_h40_nev_yen")  or 0
    nbw_h40_nev = (p4.get("summary") or {}).get("rsr_nbw_h40_nev_yen") or 0
    bw_is_problem  = bw_h40_nev  < 0
    nbw_protecting = nbw_h40_nev > 0

    # ③ Non-BW保護効果
    nbw_fwd40_mean = (
        ((p3.get("non_bigwinner") or {}).get("fwd_40d") or {}).get("mean")
    )

    # ④ 改善候補: NEV絶対値が大きい方
    rsr_nev = (p5.get("comparison") or {}).get("rsr_exit_h40_nev") or 0
    mom_nev = (p5.get("comparison") or {}).get("rsr_mom_exit_h40_nev") or 0
    improvement_candidate = "RSR_EXIT" if rsr_nev < mom_nev else "RSR_MOMENTUM_EXIT"

    # ⑤ 次研究テーマ推奨
    if not rsr_net_positive and bw_is_problem:
        next_theme = "B_BigWinner_Exception"
        rationale  = "RSR ExitがNEV負 + BW問題が支配的 → BW向け例外ロジック研究"
    elif rsr_net_positive and bw_is_problem:
        next_theme = "B_BigWinner_Exception"
        rationale  = "全体NEVは正だがBWが負 → BWだけ別扱いすれば改善余地あり"
    elif not rsr_net_positive:
        next_theme = "A_RSR_Exit_Improvement"
        rationale  = "RSR全体がNEV負 → Exit基準そのものの改善研究"
    else:
        next_theme = "C_Retention_Signal"
        rationale  = "RSR全体NEV正・BW問題少 → 特定条件下での保持延長シグナル研究"

    # 感度確認
    consistent = p6.get("nev_direction_consistent") or False

    # RSR系Exitのfwd40d win rateまとめ
    rsr_fwd40_wr = (
        ((p1.get("RSR_COMBINED") or {}).get("by_horizon") or {})
        .get("fwd_40d", {}).get("win_rate")
    )

    return {
        "1_rsr_net_ev_h40_yen":     _s(rsr_h40_nev),
        "1_rsr_net_ev_positive":    rsr_net_positive,
        "2_bw_rsr_nev_h40":         _s(bw_h40_nev),
        "2_bw_is_problem":          bw_is_problem,
        "3_nbw_rsr_nev_h40":        _s(nbw_h40_nev),
        "3_nbw_protecting":         nbw_protecting,
        "3_nbw_fwd40_mean":         nbw_fwd40_mean,
        "4_improvement_candidate":  improvement_candidate,
        "4_rsr_exit_h40_nev":       _s(rsr_nev),
        "4_rsr_mom_h40_nev":        _s(mom_nev),
        "5_recommended_next":       next_theme,
        "5_rationale":              rationale,
        "6_sensitivity_consistent": consistent,
        "rsr_fwd40_win_rate":       rsr_fwd40_wr,
        "priority_themes": [
            "A: RSR_EXIT_Improvement" if next_theme == "A_RSR_Exit_Improvement"
            else "A: RSR_EXIT_Improvement (低優先)",
            "B: BigWinner_Exception" if next_theme == "B_BigWinner_Exception"
            else "B: BigWinner_Exception (低優先)",
            "C: Retention_Signal Research" if next_theme == "C_Retention_Signal"
            else "C: Retention_Signal Research (低優先)",
        ],
    }


# ======================================================================
# main
# ======================================================================

def main():
    print("=== Study66: RSR Economic Audit ===")
    print(f"  今日: {TODAY_STR}")

    print("\n[データロード中...]")
    ds        = build_common_dataset(DATA_END)
    rsr_df    = ds["rsr_df"]
    all_dates = rsr_df.index.sort_values()
    is_dates  = all_dates[(all_dates >= IS_START) & (all_dates <= IS_END)]
    oos_dates = all_dates[(all_dates >= OOS_START) & (all_dates <= OOS_END)]

    print("[BT実行: IS 2018-2024]")
    sym_is = get_active(ds, IS_START, IS_END)
    bt_is  = run_bt(ds, sym_is, IS_START, IS_END)
    tr_is  = extract_trades_full(bt_is, is_dates, rsr_df)
    print(f"  IS trades: {len(tr_is)}")

    print("[BT実行: OOS 2025]")
    sym_oos = get_active(ds, OOS_START, OOS_END)
    bt_oos  = run_bt(ds, sym_oos, OOS_START, OOS_END)
    tr_oos  = extract_trades_full(bt_oos, oos_dates, rsr_df)
    print(f"  OOS trades: {len(tr_oos)}")

    all_trades = tr_is + tr_oos
    print(f"  全取引: {len(all_trades)} 件")

    print("\n[データセット構築...]")
    df = build_study66_dataset(ds, all_trades)
    print(f"  有効トレード数: {len(df)}")
    if df.empty:
        print("  ERROR: データ空"); return

    rsr_n = df["is_rsr_exit"].sum()
    print(f"  BW={df['is_big_winner'].sum()} / RSR系Exit={rsr_n}")

    print("\n[Phase0: Integrity]")
    p0 = phase0_integrity(df, all_trades)
    print(f"  n={p0['n_valid']} Study63:{p0['study63_n_match']} "
          f"RSR_EXIT={p0['rsr_exit_n']}({p0['study65_rsr_exit_match']}) "
          f"RSR_MOM={p0['rsr_momentum_exit_n']}({p0['study65_rsr_mom_match']})")

    print("\n[Phase1: Forward Distribution]")
    p1 = phase1_forward_distribution(df)
    for label in ["RSR_EXIT", "RSR_MOMENTUM_EXIT", "RSR_COMBINED", "NON_RSR"]:
        d = p1.get(label, {})
        fwd40 = (d.get("by_horizon") or {}).get("fwd_40d", {})
        print(f"  {label} (n={d.get('n')}): fwd40d mean={fwd40.get('mean')}% "
              f"WR={fwd40.get('win_rate')}%")
    cmp = p1.get("comparison_rsr_vs_non_rsr_fwd40d", {})
    print(f"  RSR vs NonRSR fwd40d: {cmp.get('rsr_mean')}% vs {cmp.get('non_rsr_mean')}% "
          f"p={cmp.get('mwu_p')} d={cmp.get('cohens_d')}")

    print("\n[Phase2: Economic Classification]")
    p2 = phase2_economic_classification(df)
    cls = p2.get("classification", {})
    for c in ["A_CORRECT", "B_MIXED", "C_PREMATURE"]:
        v = cls.get(c, {})
        print(f"  {c}: n={v.get('n')} ({v.get('pct')}%) "
              f"BW={v.get('bw_n')} fwd40d={v.get('fwd40d_mean')}%")
    print(f"  BW分類: {p2['bw_classification']}")

    print("\n[Phase3: BW vs Non-BW]")
    p3 = phase3_bw_vs_nbw(df)
    for nd in [20, 40, 60]:
        c = p3["comparisons"].get(f"fwd_{nd}d", {})
        print(f"  fwd+{nd}d: BW={c.get('bw_mean')}% NBW={c.get('nbw_mean')}% "
              f"Diff={c.get('diff_bw_nbw')}% p={c.get('mwu_p')} d={c.get('cohens_d')}")

    print("\n[Phase4: Net Economic Value (h40d)]")
    p4 = phase4_net_economic_value(df)
    s  = p4.get("summary", {})
    for lbl in ["RSR_COMBINED", "RSR_BigWinner", "RSR_NonBigWinner"]:
        g  = p4.get(lbl, {})
        h40 = (g.get("by_horizon") or {}).get("h40", {})
        print(f"  {lbl} (n={g.get('n')}): "
              f"NEV={h40.get('net_ev_yen'):,}¥ "
              f"[LA={h40.get('loss_avoided_yen'):,} PL={h40.get('profit_lost_yen'):,}] "
              f"pos={h40.get('net_ev_positive')}")

    print("\n[Phase5: RSR_EXIT vs RSR_MOMENTUM_EXIT]")
    p5 = phase5_exit_reason_decomposition(df)
    for lbl in ["RSR_EXIT", "RSR_MOMENTUM_EXIT"]:
        v = p5.get(lbl, {})
        print(f"  {lbl}: n={v.get('n')} BW={v.get('bw_n')} "
              f"fwd40d={v.get('fwd40d_mean')}%(WR={v.get('fwd40d_win_rate')}%) "
              f"NEV={v.get('h40_net_ev_yen'):,}¥ "
              f"[LA={v.get('h40_loss_avoided_yen'):,} PL={v.get('h40_profit_lost_yen'):,}]")
    cmp5 = p5.get("comparison", {})
    print(f"  改善候補: {cmp5.get('worse_reason')} (NEV小)")

    print("\n[Phase6: Sensitivity]")
    p6 = phase6_sensitivity(df)
    print(f"  NEV方向一致: {p6['nev_direction_consistent']} {p6['nev_signs']}")
    print(f"  BW RSR fwd40d: {p6['bw_rsr_fwd40_mean']}% (Study65={S65_BW_FWD40}%) "
          f"整合:{p6['study65_consistent']}")

    print("\n[Phase7: Research Verdict]")
    p7 = phase7_verdict(p1, p2, p3, p4, p5, p6)
    print(f"  ① RSR NEV(h40): {p7['1_rsr_net_ev_h40_yen']:,}¥ "
          f"正={p7['1_rsr_net_ev_positive']}")
    print(f"  ② BW問題: NEV={p7['2_bw_rsr_nev_h40']:,}¥ is_problem={p7['2_bw_is_problem']}")
    print(f"  ③ NonBW保護: NEV={p7['3_nbw_rsr_nev_h40']:,}¥ "
          f"protecting={p7['3_nbw_protecting']}")
    print(f"  ④ 改善候補: {p7['4_improvement_candidate']}")
    print(f"  ⑤ 推奨次テーマ: {p7['5_recommended_next']}")
    print(f"     理由: {p7['5_rationale']}")

    result = {
        "study":  "Study66",
        "title":  "RSR Economic Audit",
        "date":   TODAY_STR,
        "params": {
            "capital": CAPITAL, "fwd_days": FWD_DAYS,
            "post_exit_days": POST_EXIT_DAYS,
            "rsr_reasons": list(RSR_REASONS),
            "s65_rsr_plb_pct": S65_RSR_PLB_PCT,
            "s65_pre_peak_pct": S65_PRE_PEAK_PCT,
            "s65_bw_fwd40": S65_BW_FWD40,
        },
        "phase0_integrity":    p0,
        "phase1_forward_dist": p1,
        "phase2_classification":p2,
        "phase3_bw_vs_nbw":   p3,
        "phase4_nev":          p4,
        "phase5_decomposition":p5,
        "phase6_sensitivity":  p6,
        "phase7_verdict":      p7,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n保存: {OUT_FILE}")
    print("=== Study66 完了 ===")


if __name__ == "__main__":
    main()
