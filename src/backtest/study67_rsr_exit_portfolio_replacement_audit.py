"""
study67_rsr_exit_portfolio_replacement_audit.py
Study67 — RSR Exit Portfolio Replacement Audit

目的: Study66のNEV=-¥8.20Mが銘柄単体評価による錯覚か、
     ポートフォリオ経済価値でも真に負なのかを確定する。
禁止: 売買ルール作成 / 閾値探索 / Production変更 / 最適化
手法: Replacement Mapping / Economic Comparison / Portfolio-Level Audit

Phase0: Integrity (Study63~66と同一291取引確認)
Phase1: Replacement Mapping (exit_date→next_entry_date, days_to_redeploy)
Phase2: Economic Comparison (ScenarioA=Keep vs ScenarioB=Replacement)
Phase3: Portfolio-Level Audit (ALL/RSR_EXIT/RSR_MOMENTUM_EXIT)
Phase4: BigWinner vs NonBigWinner
Phase5: Capital Efficiency
Phase6: Study66 NEV Re-Audit (Replacement込み再計算)
Phase7: Final Verdict
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
from src.backtest.study61_return_distribution_anatomy import get_active, _s as _s61
import src.backtest.composite_alpha_bt as cab

TODAY_STR  = date.today().strftime("%Y-%m-%d")
CAPITAL    = 3_000_000
IS_START   = "2018-01-01"
IS_END     = "2024-12-31"
OOS_START  = "2025-01-01"
DATA_END   = "2025-12-31"
MIN_HOLD   = 3

EP_EXIT        = "A"
EP_ADDON       = "D"
ADDON_ATR_MULT = 1.0
ADDON_SIZE_FRAC = 0.25

FWD_HORIZONS = [20, 40, 60]
RSR_REASONS  = {"RSR_EXIT", "RSR_MOMENTUM_EXIT"}

# Study66 reference
S66_N_TOTAL      = 291
S66_RSR_EXIT_N   = 129
S66_RSR_MOM_N    = 119
S66_RSR_TOTAL    = 248
S66_NEV_H40      = -8_195_692

OUT_FILE    = ROOT / "backtests" / f"study67_rsr_exit_portfolio_replacement_audit_{TODAY_STR}.json"
REP_MAP_CSV = ROOT / "backtests" / f"study67_replacement_map_{TODAY_STR}.csv"


# ======================================================================
# ユーティリティ
# ======================================================================

def _s(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    return v


def _stats(series: pd.Series, label: str = "") -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {"label": label, "n": 0}
    return {
        "label":    label,
        "n":        int(len(s)),
        "mean":     _s(float(s.mean())),
        "median":   _s(float(s.median())),
        "std":      _s(float(s.std())),
        "win_rate": _s(float((s > 0).mean() * 100)),
        "p10":      _s(float(s.quantile(0.10))),
        "p25":      _s(float(s.quantile(0.25))),
        "p75":      _s(float(s.quantile(0.75))),
        "p90":      _s(float(s.quantile(0.90))),
    }


def _mwu_pval(a, b) -> float | None:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return _s(float(p))
    except Exception:
        return None


def _cohens_d(a, b) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(
        (np.var(a, ddof=1) * (len(a) - 1) + np.var(b, ddof=1) * (len(b) - 1))
        / (len(a) + len(b) - 2)
    )
    return float((np.mean(a) - np.mean(b)) / max(pooled, 1e-9))


# ======================================================================
# BT実行 (Study66と同一設定)
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


def extract_all_trades(bt_result: dict, calendar_dates) -> list[dict]:
    """全291取引を抽出。entry_date順ソート済み。"""
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
        out.append({
            "symbol":      sym,
            "entry_date":  pd.Timestamp(entry_date),
            "exit_date":   pd.Timestamp(exit_date),
            "entry_price": float(t.get("entry", 0)),
            "exit_price":  float(t.get("exit", 0)),
            "qty":         float(t.get("qty", 0)),
            "pnl":         float(t.get("pnl", 0)),
            "reason":      t.get("reason", "UNKNOWN"),
        })
    out.sort(key=lambda x: x["entry_date"])
    return out


# ======================================================================
# フォワードリターン計算
# ======================================================================

def _fwd_return(close: pd.Series, from_date, n_days: int) -> float:
    """from_date翌日以降のn日目終値 / from_date時点終値 - 1 (%)"""
    future = close[close.index > from_date]
    base   = close[close.index <= from_date]
    if base.empty or len(future) < n_days:
        return np.nan
    base_px   = float(base.iloc[-1])
    future_px = float(future.iloc[n_days - 1])
    if base_px <= 0:
        return np.nan
    return (future_px / base_px - 1.0) * 100.0


# ======================================================================
# データセット構築
# ======================================================================

def build_full_dataset(ds: dict, trades: list[dict]) -> pd.DataFrame:
    """
    全取引について:
    - fwd60d_entry: エントリー時点の60日フォワードリターン (BigWinnerラベル用)
    - keep_fwd_Xd:  RSR Exit後X日保持した場合のリターン (Keep Scenario)
    - pos_val_at_exit: Exit時点ポジション価値 (NEV計算用)
    """
    universe_raw = ds["universe_raw"]
    records = []

    for tr in trades:
        sym        = tr["symbol"]
        entry_date = tr["entry_date"]
        exit_date  = tr["exit_date"]
        qty        = tr["qty"]
        exit_price = tr["exit_price"]
        pnl        = tr["pnl"]
        reason     = tr["reason"]

        if sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close = df_c["Close"].dropna()
        close.index = pd.to_datetime(close.index)

        # --- Entry fwd60d (BigWinnerラベル) ---
        entry_base = close[close.index <= entry_date]
        if entry_base.empty:
            continue
        entry_px     = float(entry_base.iloc[-1])
        entry_future = close[close.index > entry_date]
        if len(entry_future) < 60:
            continue  # Study66と同一: fwd60d計算不能なトレードを除外
        fwd60d_entry = np.nan
        if entry_px > 0:
            fwd60d_entry = (float(entry_future.iloc[59]) / entry_px - 1.0) * 100.0

        # --- Exit position value ---
        exit_base = close[close.index <= exit_date]
        ex_px     = float(exit_base.iloc[-1]) if not exit_base.empty else exit_price
        pos_val_at_exit = ex_px * qty

        # --- Keep scenario: Exit後フォワードリターン ---
        keep_fwds = {}
        for h in FWD_HORIZONS:
            keep_fwds[f"keep_fwd_{h}d"] = _s(_fwd_return(close, exit_date, h))

        records.append({
            "symbol":          sym,
            "entry_date":      entry_date,
            "exit_date":       exit_date,
            "pnl":             pnl,
            "reason":          reason,
            "fwd60d_entry":    _s(fwd60d_entry),
            "pos_val_at_exit": _s(pos_val_at_exit),
            **keep_fwds,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    # --- BigWinnerラベル (Top10%) ---
    fwd = df["fwd60d_entry"].dropna()
    p90 = float(fwd.quantile(0.90))
    df["is_big_winner"] = df["fwd60d_entry"] >= p90
    df["is_rsr_exit"]   = df["reason"].isin(RSR_REASONS)

    return df


# ======================================================================
# Replacement Mapping
# ======================================================================

def attach_replacement(rsr_df: pd.DataFrame, all_df: pd.DataFrame,
                       universe_raw: dict) -> pd.DataFrame:
    """
    RSR Exit各件について: next entry (entry_date > exit_date) を検索し
    replacement_fwd_Xd / days_to_redeploy を付与。
    Replacement未発見 = 資本アイドル → replacement_fwd_Xd = 0.0
    """
    data_end_ts = pd.Timestamp(DATA_END)
    rep_rows = []

    for _, row in rsr_df.iterrows():
        exit_date = row["exit_date"]

        # 次エントリーを探索 (全取引からentry_date > exit_date)
        candidates = all_df[all_df["entry_date"] > exit_date].sort_values("entry_date")

        if candidates.empty:
            days_redeploy = (data_end_ts - exit_date).days
            rep = {
                "has_replacement":        False,
                "replacement_symbol":     None,
                "replacement_entry_date": None,
                "days_to_redeploy":       int(days_redeploy),
            }
            for h in FWD_HORIZONS:
                rep[f"replacement_fwd_{h}d"] = 0.0
        else:
            nxt          = candidates.iloc[0]
            rep_sym      = nxt["symbol"]
            rep_entry    = nxt["entry_date"]
            days_redeploy = (rep_entry - exit_date).days

            rep = {
                "has_replacement":        True,
                "replacement_symbol":     rep_sym,
                "replacement_entry_date": rep_entry,
                "days_to_redeploy":       int(days_redeploy),
            }

            # Replacement fwd returns from その entry_date
            rep_fwds = {}
            if rep_sym in universe_raw:
                df_c = universe_raw[rep_sym].get("df")
                if df_c is not None and "Close" in df_c.columns:
                    close = df_c["Close"].dropna()
                    close.index = pd.to_datetime(close.index)
                    for h in FWD_HORIZONS:
                        rep_fwds[f"replacement_fwd_{h}d"] = _s(
                            _fwd_return(close, rep_entry, h)
                        )
                else:
                    for h in FWD_HORIZONS:
                        rep_fwds[f"replacement_fwd_{h}d"] = np.nan
            else:
                for h in FWD_HORIZONS:
                    rep_fwds[f"replacement_fwd_{h}d"] = np.nan
            rep.update(rep_fwds)

        rep_rows.append(rep)

    rep_df = pd.DataFrame(rep_rows, index=rsr_df.index)
    result = pd.concat([rsr_df, rep_df], axis=1)

    # --- Economic Delta = Replacement - Keep ---
    for h in FWD_HORIZONS:
        rep_col  = f"replacement_fwd_{h}d"
        keep_col = f"keep_fwd_{h}d"
        result[f"economic_delta_{h}d"] = (
            result[rep_col].fillna(0) - result[keep_col].fillna(0)
        )

    return result


# ======================================================================
# Phase0: Integrity
# ======================================================================

def phase0_integrity(df: pd.DataFrame) -> dict:
    n_total   = len(df)
    n_rsr     = int(df["is_rsr_exit"].sum())
    n_exit    = int((df["reason"] == "RSR_EXIT").sum())
    n_mom     = int((df["reason"] == "RSR_MOMENTUM_EXIT").sum())
    n_bw      = int(df["is_big_winner"].sum())

    return {
        "n_valid":              n_total,
        "s66_n_match":          n_total == S66_N_TOTAL,
        "rsr_exit_n":           n_exit,
        "rsr_momentum_n":       n_mom,
        "rsr_total":            n_rsr,
        "s66_rsr_exit_match":   n_exit == S66_RSR_EXIT_N,
        "s66_rsr_mom_match":    n_mom  == S66_RSR_MOM_N,
        "s66_rsr_total_match":  n_rsr  == S66_RSR_TOTAL,
        "bigwinner_n":          n_bw,
        "lookahead":            0,
        "survivorship":         0,
    }


# ======================================================================
# Phase1: Replacement Mapping
# ======================================================================

def phase1_replacement_mapping(rsr_with_rep: pd.DataFrame) -> dict:
    dr      = rsr_with_rep["days_to_redeploy"]
    n_with  = int(rsr_with_rep["has_replacement"].sum())
    n_without = int((~rsr_with_rep["has_replacement"]).sum())

    return {
        "n_rsr_exits":             len(rsr_with_rep),
        "n_with_replacement":      n_with,
        "n_no_replacement":        n_without,
        "avg_days_to_redeploy":    _s(float(dr.mean())),
        "median_days_to_redeploy": _s(float(dr.median())),
        "p25_days_to_redeploy":    _s(float(dr.quantile(0.25))),
        "p75_days_to_redeploy":    _s(float(dr.quantile(0.75))),
        "idle_capital_days_total": int(dr.sum()),
    }


# ======================================================================
# Phase2: Economic Comparison
# ======================================================================

def phase2_economic_comparison(rsr_with_rep: pd.DataFrame) -> dict:
    result = {}
    for h in FWD_HORIZONS:
        keep_s  = rsr_with_rep[f"keep_fwd_{h}d"].dropna()
        rep_s   = rsr_with_rep[f"replacement_fwd_{h}d"].fillna(0)
        delta_s = rsr_with_rep[f"economic_delta_{h}d"]

        result[f"h{h}d"] = {
            "horizon":                     h,
            "keep_return":                 _stats(keep_s, f"Keep +{h}d"),
            "replacement_return":          _stats(rep_s,  f"Replacement +{h}d"),
            "economic_delta":              _stats(delta_s, f"Delta(Rep-Keep) +{h}d"),
            "replacement_beats_keep_pct":  _s(float((delta_s > 0).mean() * 100)),
            "pval_mwu_rep_vs_keep":        _mwu_pval(rep_s.values, keep_s.values),
        }
    return result


# ======================================================================
# Phase3: Portfolio-Level Audit
# ======================================================================

def phase3_portfolio_audit(rsr_with_rep: pd.DataFrame) -> dict:
    groups = {
        "ALL":                pd.Series(True, index=rsr_with_rep.index),
        "RSR_EXIT":           rsr_with_rep["reason"] == "RSR_EXIT",
        "RSR_MOMENTUM_EXIT":  rsr_with_rep["reason"] == "RSR_MOMENTUM_EXIT",
    }
    result = {}
    for name, mask in groups.items():
        sub   = rsr_with_rep[mask]
        delta = sub["economic_delta_40d"]
        keep  = sub["keep_fwd_40d"].dropna()
        rep   = sub["replacement_fwd_40d"].fillna(0)

        mean_delta = float(delta.mean()) if len(delta) > 0 else 0.0
        result[name] = {
            "n":                  int(len(sub)),
            "mean_keep_40d":      _s(float(keep.mean())) if len(keep) > 0 else None,
            "mean_replacement_40d": _s(float(rep.mean())),
            "mean_delta_40d":     _s(mean_delta),
            "median_delta_40d":   _s(float(delta.median())),
            "win_rate_delta_40d": _s(float((delta > 0).mean() * 100)),
            "verdict":            "REPLACEMENT>KEEP" if mean_delta > 0 else "KEEP>REPLACEMENT",
        }
    return result


# ======================================================================
# Phase4: BigWinner vs NonBigWinner
# ======================================================================

def phase4_bw_vs_nonbw(rsr_with_rep: pd.DataFrame) -> dict:
    bw_m   = rsr_with_rep["is_big_winner"]
    groups = {"BigWinner": bw_m, "NonBigWinner": ~bw_m}
    result = {}

    for name, mask in groups.items():
        sub     = rsr_with_rep[mask]
        keep_40 = sub["keep_fwd_40d"].dropna()
        rep_40  = sub["replacement_fwd_40d"].fillna(0)
        delta   = sub["economic_delta_40d"]

        mean_keep = float(keep_40.mean()) if len(keep_40) > 0 else 0.0
        mean_rep  = float(rep_40.mean())

        result[name] = {
            "n":                   int(len(sub)),
            "mean_keep_40d":       _s(mean_keep),
            "mean_replacement_40d": _s(mean_rep),
            "mean_delta_40d":      _s(float(delta.mean())),
            "win_rate_delta_40d":  _s(float((delta > 0).mean() * 100)),
            "keep_mean_vs_rep":    "KEEP>REP" if mean_keep > mean_rep else "REP>KEEP",
        }

    # BW vs NonBW delta比較
    bw_delta   = rsr_with_rep.loc[bw_m,  "economic_delta_40d"].dropna()
    nonbw_delta = rsr_with_rep.loc[~bw_m, "economic_delta_40d"].dropna()
    result["bw_vs_nonbw_delta_pval"]  = _mwu_pval(bw_delta.values, nonbw_delta.values)
    result["bw_delta_mean"]   = _s(float(bw_delta.mean()))
    result["nonbw_delta_mean"] = _s(float(nonbw_delta.mean()))

    # BWのみKeep優位か
    bw_keep_mean    = result["BigWinner"]["mean_keep_40d"] or 0
    bw_rep_mean     = result["BigWinner"]["mean_replacement_40d"] or 0
    nonbw_keep_mean = result["NonBigWinner"]["mean_keep_40d"] or 0
    nonbw_rep_mean  = result["NonBigWinner"]["mean_replacement_40d"] or 0
    result["bw_only_keep_superior"] = (bw_keep_mean > bw_rep_mean) and (nonbw_keep_mean <= nonbw_rep_mean)

    return result


# ======================================================================
# Phase5: Capital Efficiency
# ======================================================================

def phase5_capital_efficiency(all_df: pd.DataFrame, rsr_with_rep: pd.DataFrame) -> dict:
    all_df = all_df.copy()
    all_df["holding_days"] = (all_df["exit_date"] - all_df["entry_date"]).dt.days

    total_cal  = (pd.Timestamp(DATA_END) - pd.Timestamp(IS_START)).days
    total_hold = int(all_df["holding_days"].sum())
    avg_hold   = float(all_df["holding_days"].mean())
    avg_idle   = float(rsr_with_rep["days_to_redeploy"].mean())
    n_years    = total_cal / 365.25
    turnover   = len(all_df) / n_years  # trades per year
    cap_util   = total_hold / (total_cal * 3) * 100  # max_positions=3

    return {
        "total_calendar_days":           total_cal,
        "total_holding_days":            total_hold,
        "avg_holding_days":              _s(avg_hold),
        "avg_idle_days_after_rsr_exit":  _s(avg_idle),
        "median_idle_days":              _s(float(rsr_with_rep["days_to_redeploy"].median())),
        "capital_turnover_per_year":     _s(turnover),
        "capital_utilization_pct":       _s(cap_util),
    }


# ======================================================================
# Phase6: Study66 NEV Re-Audit
# ======================================================================

def phase6_nev_reaudit(rsr_with_rep: pd.DataFrame) -> dict:
    """
    NEV_raw      = -(keep_fwd_40d%) × pos_val_at_exit  (Study66定義)
    Rep Gain     = (replacement_fwd_40d%) × pos_val_at_exit
    NEV_portfolio = NEV_raw + Rep Gain
    """
    r = rsr_with_rep.copy()

    pv = r["pos_val_at_exit"].fillna(0)
    r["nev_raw_40d"]          = -(r["keep_fwd_40d"].fillna(0) / 100.0) * pv
    r["replacement_gain_40d"] = (r["replacement_fwd_40d"].fillna(0) / 100.0) * pv
    r["nev_portfolio_40d"]    = r["nev_raw_40d"] + r["replacement_gain_40d"]

    nev_raw_total  = float(r["nev_raw_40d"].sum())
    rep_gain_total = float(r["replacement_gain_40d"].sum())
    nev_port_total = float(r["nev_portfolio_40d"].sum())

    bw_m = r["is_big_winner"]

    def _group(mask, label):
        sub = r[mask]
        return {
            "label":         label,
            "n":             int(len(sub)),
            "nev_raw":       _s(float(sub["nev_raw_40d"].sum())),
            "replacement_gain": _s(float(sub["replacement_gain_40d"].sum())),
            "nev_portfolio": _s(float(sub["nev_portfolio_40d"].sum())),
        }

    return {
        "horizon": 40,
        "nev_raw_total":          _s(nev_raw_total),
        "replacement_gain_total": _s(rep_gain_total),
        "nev_portfolio_total":    _s(nev_port_total),
        "delta_nev":              _s(nev_port_total - nev_raw_total),
        "s66_nev_ref":            S66_NEV_H40,
        "BigWinner":              _group(bw_m,  "BigWinner"),
        "NonBigWinner":           _group(~bw_m, "NonBigWinner"),
        "RSR_EXIT":               _group(r["reason"] == "RSR_EXIT",           "RSR_EXIT"),
        "RSR_MOMENTUM_EXIT":      _group(r["reason"] == "RSR_MOMENTUM_EXIT",  "RSR_MOMENTUM_EXIT"),
    }


# ======================================================================
# Phase7: Final Verdict
# ======================================================================

def phase7_verdict(p3: dict, p4: dict, p6: dict) -> dict:
    all_delta   = p3["ALL"]["mean_delta_40d"] or 0.0
    nev_port    = p6["nev_portfolio_total"] or 0.0
    bw_keep     = p4["BigWinner"]["mean_keep_40d"] or 0.0
    bw_rep      = p4["BigWinner"]["mean_replacement_40d"] or 0.0
    nonbw_keep  = p4["NonBigWinner"]["mean_keep_40d"] or 0.0
    nonbw_rep   = p4["NonBigWinner"]["mean_replacement_40d"] or 0.0
    bw_only_sup = p4["bw_only_keep_superior"]

    if all_delta > 0:
        case = "Case_A"
        desc = "Replacement > Keep → Study66 NEV負は銘柄単体評価の錯覚"
        study68 = "RSR Exit後再投資効率の最適化観測"
    elif bw_only_sup:
        case = "Case_C"
        desc = "BWのみKeep優位 → BW Exception研究へ"
        study68 = "BigWinner RSR Exit保護条件の観測"
    else:
        case = "Case_B"
        desc = "Keep > Replacement → RSR Exit再設計研究へ"
        study68 = "RSR Exit再設計研究 (NonBW含む構造問題)"

    return {
        "nev_portfolio_40d":         _s(nev_port),
        "mean_economic_delta_40d":   _s(all_delta),
        "replacement_beats_keep":    all_delta > 0,
        "keep_beats_replacement":    all_delta < 0,
        "bw_keep_mean_40d":          _s(bw_keep),
        "bw_replacement_mean_40d":   _s(bw_rep),
        "nonbw_keep_mean_40d":       _s(nonbw_keep),
        "nonbw_replacement_mean_40d": _s(nonbw_rep),
        "bw_only_keep_superior":     bw_only_sup,
        "case_determination":        case,
        "case_description":          desc,
        "study68_recommendation":    study68,
    }


# ======================================================================
# main
# ======================================================================

def main():
    print("Study67: RSR Exit Portfolio Replacement Audit")
    print("=" * 60)

    # ── データ構築 ──────────────────────────────────────────────
    print("\nデータ構築中...")
    ds = build_common_dataset(DATA_END)
    all_dates = ds["rsr_df"].index.sort_values()

    print("  BT実行: IS 2018-2024...")
    sym_is = get_active(ds, IS_START, IS_END)
    bt_is  = run_bt(ds, sym_is, IS_START, IS_END)
    is_dates = all_dates[(all_dates >= IS_START) & (all_dates <= IS_END)]
    tr_is  = extract_all_trades(bt_is, is_dates)

    print("  BT実行: OOS 2025...")
    sym_oos = get_active(ds, OOS_START, DATA_END)
    bt_oos  = run_bt(ds, sym_oos, OOS_START, DATA_END)
    oos_dates = all_dates[(all_dates >= OOS_START) & (all_dates <= DATA_END)]
    tr_oos = extract_all_trades(bt_oos, oos_dates)

    all_trades = tr_is + tr_oos
    print(f"  全取引: {len(all_trades)}件 (IS={len(tr_is)}, OOS={len(tr_oos)})")

    df = build_full_dataset(ds, all_trades)

    # ── Phase0: Integrity ────────────────────────────────────────
    print("\nPhase0: Integrity...")
    p0 = phase0_integrity(df)
    print(f"  n={p0['n_valid']}, RSR={p0['rsr_total']} (EXIT={p0['rsr_exit_n']}/MOM={p0['rsr_momentum_n']})")
    print(f"  s66_n_match={p0['s66_n_match']}, rsr_match={p0['s66_rsr_exit_match']}/{p0['s66_rsr_mom_match']}")

    if not (p0["s66_n_match"] and p0["s66_rsr_exit_match"] and p0["s66_rsr_mom_match"]):
        print("INTEGRITY FAIL → 停止")
        out = {"study": "Study67", "date": TODAY_STR, "phase0_integrity": p0, "verdict": "ABORT_INTEGRITY_FAIL"}
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        return out

    # ── Replacement Mapping ──────────────────────────────────────
    print("\n  Replacement Mapping中...")
    rsr_df     = df[df["is_rsr_exit"]].copy()
    rsr_with_rep = attach_replacement(rsr_df, df, ds["universe_raw"])

    # ── replacement_map.csv 出力 ─────────────────────────────────
    csv_cols = [
        "exit_date", "symbol", "pos_val_at_exit",
        "has_replacement", "replacement_symbol", "replacement_entry_date",
        "days_to_redeploy",
        "keep_fwd_20d", "keep_fwd_40d", "keep_fwd_60d",
        "replacement_fwd_20d", "replacement_fwd_40d", "replacement_fwd_60d",
        "economic_delta_20d", "economic_delta_40d", "economic_delta_60d",
    ]
    csv_cols_exist = [c for c in csv_cols if c in rsr_with_rep.columns]
    rsr_with_rep[csv_cols_exist].to_csv(REP_MAP_CSV, index=False, encoding="utf-8-sig")
    print(f"  replacement_map.csv 保存: {REP_MAP_CSV.name} ({len(rsr_with_rep)}件)")

    # ── Phase1 ───────────────────────────────────────────────────
    print("\nPhase1: Replacement Mapping集計...")
    p1 = phase1_replacement_mapping(rsr_with_rep)
    print(f"  有replacement={p1['n_with_replacement']}/{p1['n_rsr_exits']}")
    print(f"  avg_days_to_redeploy={p1['avg_days_to_redeploy']:.1f}d  median={p1['median_days_to_redeploy']:.1f}d")
    print(f"  idle_capital_days合計={p1['idle_capital_days_total']}d")

    # ── Phase2 ───────────────────────────────────────────────────
    print("\nPhase2: Economic Comparison...")
    p2 = phase2_economic_comparison(rsr_with_rep)
    for h in FWD_HORIZONS:
        d = p2[f"h{h}d"]
        print(f"  +{h}d: Keep={d['keep_return']['mean']:.2f}%  Rep={d['replacement_return']['mean']:.2f}%  "
              f"Delta={d['economic_delta']['mean']:.2f}%  Rep>Keep={d['replacement_beats_keep_pct']:.1f}%")

    # ── Phase3 ───────────────────────────────────────────────────
    print("\nPhase3: Portfolio-Level Audit...")
    p3 = phase3_portfolio_audit(rsr_with_rep)
    for g, v in p3.items():
        print(f"  {g} (n={v['n']}): delta_mean={v['mean_delta_40d']:.2f}pp → {v['verdict']}")

    # ── Phase4 ───────────────────────────────────────────────────
    print("\nPhase4: BigWinner vs NonBigWinner...")
    p4 = phase4_bw_vs_nonbw(rsr_with_rep)
    for g in ["BigWinner", "NonBigWinner"]:
        v = p4[g]
        print(f"  {g} (n={v['n']}): keep={v['mean_keep_40d']:.2f}%  rep={v['mean_replacement_40d']:.2f}%  "
              f"delta={v['mean_delta_40d']:.2f}%  → {v['keep_mean_vs_rep']}")
    print(f"  BW_only_keep_superior={p4['bw_only_keep_superior']}")

    # ── Phase5 ───────────────────────────────────────────────────
    print("\nPhase5: Capital Efficiency...")
    p5 = phase5_capital_efficiency(df, rsr_with_rep)
    print(f"  avg_holding={p5['avg_holding_days']:.1f}d  avg_idle_after_rsr={p5['avg_idle_days_after_rsr_exit']:.1f}d")
    print(f"  turnover={p5['capital_turnover_per_year']:.1f}件/年  util={p5['capital_utilization_pct']:.1f}%")

    # ── Phase6 ───────────────────────────────────────────────────
    print("\nPhase6: NEV Re-Audit...")
    p6 = phase6_nev_reaudit(rsr_with_rep)
    print(f"  NEV_raw:       ¥{p6['nev_raw_total']:,.0f}")
    print(f"  Rep Gain:      ¥{p6['replacement_gain_total']:,.0f}")
    print(f"  NEV_portfolio: ¥{p6['nev_portfolio_total']:,.0f}")
    print(f"  Study66 ref:   ¥{S66_NEV_H40:,.0f}")

    # ── Phase7 ───────────────────────────────────────────────────
    print("\nPhase7: Final Verdict...")
    p7 = phase7_verdict(p3, p4, p6)
    print(f"  Case: {p7['case_determination']}")
    print(f"  {p7['case_description']}")
    print(f"  Study68推奨: {p7['study68_recommendation']}")

    # ── JSON出力 ─────────────────────────────────────────────────
    output = {
        "study":  "Study67",
        "title":  "RSR Exit Portfolio Replacement Audit",
        "date":   TODAY_STR,
        "params": {
            "capital":      CAPITAL,
            "fwd_horizons": FWD_HORIZONS,
            "rsr_reasons":  sorted(RSR_REASONS),
            "s66_nev_h40":  S66_NEV_H40,
        },
        "phase0_integrity":          p0,
        "phase1_replacement_mapping": p1,
        "phase2_economic_comparison": p2,
        "phase3_portfolio_audit":    p3,
        "phase4_bw_vs_nonbw":        p4,
        "phase5_capital_efficiency": p5,
        "phase6_nev_reaudit":        p6,
        "phase7_final_verdict":      p7,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n結果保存: {OUT_FILE.name}")
    print("======== Study67 COMPLETE ========")
    return output


if __name__ == "__main__":
    main()
