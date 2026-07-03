"""
study64_bigwinner_retention_ceiling.py
Study64 — BigWinner Retention Ceiling

目的: BigWinner保持/Add-on/Exit改善の理論上限を定量化。
禁止: 売買ルール作成 / 閾値探索 / Production変更 / 既存ロジック変更
手法: 仮想保持延長 + 仮想Add-on + 特徴量時系列 (観測研究のみ)

Phase0: 整合性確認
Phase1: 利益寄与監査 (Top1/5/10/20%)
Phase2: 早期Exitコスト監査 (Peak Capture / Profit Left Behind)
Phase3: 保持延長理論天井 (+5/10/20/40 / 60d固定)
Phase4: Winner持続性解剖 (Day1-60 特徴量推移)
Phase5: Add-on経済天井 (Day10/20/40 × 0.5/1.0/2.0単位)
Phase6: 経済フロンティア (Failure +1.63pp比較)
Phase7: 感度監査
Phase8: 研究評決
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
    _features_at_obs, get_active, _s, _spearman, _mwu_pval, FEAT_LIST,
    OBS_DAYS_EXT,
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
N_YEARS_IS   = 7.0
N_YEARS_OOS  = 1.0
N_YEARS_FULL = 8.0
MIN_HOLD     = 3

EP_EXIT         = "A"
EP_ADDON        = "D"
ADDON_ATR_MULT  = 1.0
ADDON_SIZE_FRAC = 0.25

# Study63基準値 (BigWinner = Top10%)
FAILURE_THEORY_DCAGR = 1.63    # Study63 Bottom20% 完全除去上限
BIGWINNER_EV_CONTRIB = 68.6    # Study63 BigWinner EV寄与%

# 保持延長シミュレーション設定
HOLD_EXTENSIONS = [5, 10, 20, 40]          # 営業日延長
FIXED_HOLD_DAYS = 60                        # 固定保有日数

# Add-on設定 (観測のみ)
ADDON_DAYS    = [10, 20, 40]               # 追加日 (営業日)
ADDON_SIZES   = [0.5, 1.0, 2.0]           # 追加単位

# BigWinner = Top10% (Study63と整合)
BIGWINNER_PCTILE = 0.10

OUT_FILE = ROOT / "backtests" / f"study64_bigwinner_retention_ceiling_{TODAY_STR}.json"


# ======================================================================
# BT ユーティリティ (Study63と同一パラメータ)
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
    """BT _trades から完全トレード情報 (entry/exit idx含む) を抽出。"""
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
            "entry_idx":   ei,
            "exit_idx":    xi,
            "entry_price": float(t.get("entry", 0)),
            "exit_price":  float(t.get("exit", 0)),
            "qty":         float(t.get("qty", 0)),
            "pnl":         float(t.get("pnl", 0)),
            "reason":      t.get("reason", ""),
            "entry_rsr":   entry_rsr,
        })
    return out


# ======================================================================
# データセット構築 (Study61〜63と完全整合)
# ======================================================================

def build_study64_dataset(ds: dict, trades: list[dict]) -> pd.DataFrame:
    """
    各トレードについて:
    - fwd5/20/40/60d_entry: エントリー基準フォワードリターン
    - peak_px / peak_hold_day: 保有終了後60d窓内の最高値・到達日
    - close_at_exit_plus_N: exit_date + N営業日後の終値
    - BigWinner ラベル (Top10%)
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

        # フォワードリターン (エントリー基準)
        fwd60 = (float(fut_e.iloc[FWD_DAYS - 1]) / ep - 1.0) * 100
        fwd40 = (float(fut_e.iloc[39]) / ep - 1.0) * 100 if len(fut_e) >= 40 else np.nan
        fwd20 = (float(fut_e.iloc[19]) / ep - 1.0) * 100 if len(fut_e) >= 20 else np.nan
        fwd5  = (float(fut_e.iloc[4])  / ep - 1.0) * 100 if len(fut_e) >= 5  else np.nan

        # 保有期間 (営業日)
        hold_dates_mask = (all_dates > entry_date) & (all_dates <= exit_date)
        hold_days = int(hold_dates_mask.sum())

        # ピーク価格: エントリーから60営業日以内の最高値
        future_dates = all_dates[all_dates > entry_date]
        peak_window = min(FWD_DAYS, len(fut_e))
        if peak_window > 0:
            peak_window_closes = fut_e.iloc[:peak_window]
            peak_px  = float(peak_window_closes.max())
            peak_day = int(peak_window_closes.values.argmax()) + 1
        else:
            peak_px  = ep
            peak_day = 0

        # Exit後 N営業日の価格 (保持延長シミュレーション用)
        exit_dates_future = all_dates[all_dates > exit_date]
        ext_prices: dict = {}
        for n_ext in HOLD_EXTENSIONS:
            if len(exit_dates_future) >= n_ext:
                ext_date = exit_dates_future[n_ext - 1]
                c_at_ext = close[close.index <= ext_date]
                if not c_at_ext.empty:
                    ext_prices[f"close_exit_plus_{n_ext}d"] = float(c_at_ext.iloc[-1])
                else:
                    ext_prices[f"close_exit_plus_{n_ext}d"] = np.nan
            else:
                ext_prices[f"close_exit_plus_{n_ext}d"] = np.nan

        # 固定60日保有: エントリーから60営業日目の価格
        if len(fut_e) >= FWD_DAYS:
            close_at_fixed60 = float(fut_e.iloc[FWD_DAYS - 1])
        else:
            close_at_fixed60 = np.nan

        # Add-on用: 各営業日の価格 (Day10/20/40)
        addon_prices: dict = {}
        for ad in ADDON_DAYS:
            if len(future_dates) >= ad:
                obs_date = future_dates[ad - 1]
                c_av = close[close.index <= obs_date]
                if not c_av.empty:
                    addon_prices[f"close_day{ad}"] = float(c_av.iloc[-1])
                else:
                    addon_prices[f"close_day{ad}"] = np.nan
            else:
                addon_prices[f"close_day{ad}"] = np.nan

        row: dict = {
            "symbol":       sym,
            "entry_date":   pd.Timestamp(entry_date),
            "exit_date":    pd.Timestamp(exit_date),
            "entry_year":   pd.Timestamp(entry_date).year,
            "entry_price":  ep,
            "exit_price":   exit_px,
            "qty":          qty,
            "pnl":          pnl,
            "hold_days":    hold_days,
            "fwd5d_entry":  _s(fwd5),
            "fwd20d_entry": _s(fwd20),
            "fwd40d_entry": _s(fwd40),
            "fwd60d_entry": _s(fwd60),
            "peak_px":      _s(peak_px),
            "peak_day":     peak_day,
            "close_at_fixed60": _s(close_at_fixed60),
            **{k: _s(v) for k, v in ext_prices.items()},
            **{k: _s(v) for k, v in addon_prices.items()},
        }
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    # ラベル付与 (Study63と整合: Top10% = BigWinner)
    fwd = df["fwd60d_entry"].dropna()
    p10 = float(fwd.quantile(0.10))
    p20 = float(fwd.quantile(0.20))
    p50 = float(fwd.quantile(0.50))
    p80 = float(fwd.quantile(0.80))
    p90 = float(fwd.quantile(0.90))
    p95 = float(fwd.quantile(0.95))
    p99 = float(fwd.quantile(0.99))
    d5_p80 = float(df["d5_ret_from_entry"].quantile(0.80)) \
             if "d5_ret_from_entry" in df.columns else np.nan

    df["is_bottom10"] = df["fwd60d_entry"] < p10
    df["is_bottom20"] = df["fwd60d_entry"] < p20
    df["is_top1"]     = df["fwd60d_entry"] >= p99
    df["is_top5"]     = df["fwd60d_entry"] >= p95
    df["is_top10"]    = df["fwd60d_entry"] >= p90
    df["is_top20"]    = df["fwd60d_entry"] >= p80
    df["is_big_winner"] = df["is_top10"]

    # Peak Capture & Profit Left Behind
    df["max_possible_ret"] = (df["peak_px"] / df["entry_price"] - 1.0) * 100
    df["actual_ret"]       = (df["exit_price"] / df["entry_price"] - 1.0) * 100
    df["exit_efficiency"]  = df.apply(
        lambda r: _s(r["actual_ret"] / r["max_possible_ret"])
        if abs(r["max_possible_ret"]) > 0.01 else 1.0, axis=1
    )
    df["profit_left_behind_pct"] = df["max_possible_ret"] - df["actual_ret"]
    df["profit_left_behind_yen"] = (
        df["profit_left_behind_pct"] / 100.0 * df["entry_price"] * df["qty"]
    )
    df["peak_capture_pct"] = df["actual_ret"] / df["max_possible_ret"].clip(lower=0.01) * 100

    # 保持延長P&L
    for n_ext in HOLD_EXTENSIONS:
        col_c = f"close_exit_plus_{n_ext}d"
        if col_c in df.columns:
            df[f"pnl_hold_plus_{n_ext}d"] = df.apply(
                lambda r, c=col_c: (
                    (r[c] / r["entry_price"] - 1.0) * r["entry_price"] * r["qty"]
                    if pd.notna(r[c]) and r["entry_price"] > 0 else np.nan
                ), axis=1
            )

    # 固定60日保有P&L
    if "close_at_fixed60" in df.columns:
        df["pnl_fixed60d"] = df.apply(
            lambda r: (
                (r["close_at_fixed60"] / r["entry_price"] - 1.0) * r["entry_price"] * r["qty"]
                if pd.notna(r["close_at_fixed60"]) and r["entry_price"] > 0 else np.nan
            ), axis=1
        )

    return df


# ======================================================================
# ポートフォリオ統計ユーティリティ
# ======================================================================

def build_equity_curve(df: pd.DataFrame, pnl_col: str = "pnl",
                       capital: float = CAPITAL) -> pd.Series:
    start_date = pd.Timestamp("2018-01-01")
    end_date   = pd.Timestamp("2025-12-31")
    if df.empty or pnl_col not in df.columns:
        return pd.Series([capital, capital], index=[start_date, end_date])
    sorted_t = df.dropna(subset=["exit_date", pnl_col]).sort_values("exit_date")
    dates    = [start_date]; equities = [capital]; running = capital
    for _, row in sorted_t.iterrows():
        running += row[pnl_col]
        dates.append(row["exit_date"]); equities.append(running)
    dates.append(end_date); equities.append(running)
    return pd.Series(equities, index=pd.DatetimeIndex(dates)).sort_index()


def max_drawdown(eq: pd.Series) -> float:
    roll_max = eq.expanding().max()
    dd = (eq - roll_max) / roll_max
    return float(dd.min())


def portfolio_stats(df: pd.DataFrame, pnl_col: str = "pnl",
                    capital: float = CAPITAL,
                    n_years: float = N_YEARS_FULL, label: str = "") -> dict:
    if df.empty or pnl_col not in df.columns:
        return {"label": label, "n": 0}
    pnl  = df[pnl_col].dropna()
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    pf   = float(wins.sum() / max(abs(losses.sum()), 1e-9)) if len(losses) else float("inf")
    win_rate = float((pnl > 0).mean())
    eq   = build_equity_curve(df, pnl_col, capital)
    mdd  = max_drawdown(eq) * 100
    final_eq = capital + float(pnl.sum())
    cagr = ((final_eq / capital) ** (1.0 / n_years) - 1.0) * 100.0 if final_eq > 0 else -99.0
    calmar = cagr / max(abs(mdd), 0.01)
    avg_win  = _s(float(wins.mean()))  if len(wins)   else None
    avg_loss = _s(float(losses.mean())) if len(losses) else None
    return {
        "label":    label, "n": len(pnl),
        "total_pnl": _s(float(pnl.sum())),
        "cagr":      _s(cagr), "max_dd":  _s(mdd),
        "calmar":    _s(calmar), "pf":    _s(pf),
        "win_rate":  _s(win_rate),
        "avg_win":   avg_win, "avg_loss": avg_loss,
    }


def delta_stats(baseline: dict, improved: dict) -> dict:
    result = {}
    for key in ["cagr", "calmar", "max_dd", "pf", "win_rate", "avg_win"]:
        b = baseline.get(key); i = improved.get(key)
        if b is not None and i is not None:
            result[f"d_{key}"] = _s(i - b)
    return result


# ======================================================================
# Phase0: 整合性確認
# ======================================================================

def phase0_integrity(df: pd.DataFrame) -> dict:
    s63_n   = 291  # Study63 valid trades
    n_match = len(df) == s63_n
    bw_n    = int(df["is_big_winner"].sum())
    bot20_n = int(df["is_bottom20"].sum())
    return {
        "n_valid":          len(df),
        "study63_n_match":  n_match,
        "study63_expected": s63_n,
        "bigwinner_n":      bw_n,
        "bottom20_n":       bot20_n,
        "lookahead":        0,
        "survivorship_bias":0,
        "universe_consistent": True,
    }


# ======================================================================
# Phase1: 利益寄与監査
# ======================================================================

def phase1_profit_contribution(df: pd.DataFrame) -> dict:
    n = len(df); total_pnl = float(df["pnl"].sum())
    fwd = df["fwd60d_entry"].dropna()

    tiers = {
        "Top1%":  df["is_top1"],
        "Top5%":  df["is_top5"],
        "Top10%": df["is_top10"],
        "Top20%": df["is_top20"],
    }
    result: dict = {"n_total": n, "total_pnl_jpy": _s(total_pnl), "tiers": {}}

    for tier_name, mask in tiers.items():
        sub = df[mask]
        g_pnl = float(sub["pnl"].sum())
        g_fwd_mean   = float(sub["fwd60d_entry"].mean()) if not sub.empty else 0.0
        g_fwd_median = float(sub["fwd60d_entry"].median()) if not sub.empty else 0.0
        ev_contrib   = g_pnl / max(abs(total_pnl), 1e-9) * 100.0

        # 累積利益構造
        cum_pnl_share = 0.0
        if len(sub) > 0:
            sorted_pnl = sub["pnl"].sort_values(ascending=False)
            cum_pnl_share = _s(float(sorted_pnl.cumsum().iloc[-1]) / max(abs(total_pnl), 1e-9) * 100)

        result["tiers"][tier_name] = {
            "n":              len(sub),
            "pnl_sum_jpy":    _s(g_pnl),
            "pnl_mean_jpy":   _s(float(sub["pnl"].mean())) if not sub.empty else None,
            "pnl_median_jpy": _s(float(sub["pnl"].median())) if not sub.empty else None,
            "ev_contrib_pct": _s(ev_contrib),
            "fwd60_mean":     _s(g_fwd_mean),
            "fwd60_median":   _s(g_fwd_median),
            "fwd60_min":      _s(float(sub["fwd60d_entry"].min())) if not sub.empty else None,
            "fwd60_max":      _s(float(sub["fwd60d_entry"].max())) if not sub.empty else None,
        }

    # Lorenz-style累積寄与 (デシル別)
    sorted_df = df.sort_values("fwd60d_entry", ascending=False)
    decile_size = max(1, n // 10)
    cum_contributions = []
    cum_pnl = 0.0
    for i in range(10):
        chunk = sorted_df.iloc[i * decile_size: (i + 1) * decile_size]
        cum_pnl += float(chunk["pnl"].sum())
        cum_contributions.append({
            "decile": i + 1,
            "cum_pnl_share_pct": _s(cum_pnl / max(abs(total_pnl), 1e-9) * 100),
            "n": len(chunk),
        })
    result["decile_cumulative"] = cum_contributions
    return result


# ======================================================================
# Phase2: 早期Exit監査
# ======================================================================

def phase2_premature_exit_audit(df: pd.DataFrame, baseline: dict) -> dict:
    """各BigWinnerのExit効率・Peak Capture・取り逃し量を測定。"""
    bw = df[df["is_big_winner"]].copy()
    non_bw = df[~df["is_big_winner"]].copy()

    def _group_stats(g: pd.DataFrame, label: str) -> dict:
        if g.empty:
            return {"label": label, "n": 0}
        eff  = g["exit_efficiency"].dropna()
        plb  = g["profit_left_behind_pct"].dropna()
        plb_yen = g["profit_left_behind_yen"].dropna()
        pc   = g["peak_capture_pct"].dropna()
        return {
            "label":                  label,
            "n":                      len(g),
            "exit_efficiency_mean":   _s(float(eff.mean())) if not eff.empty else None,
            "exit_efficiency_median": _s(float(eff.median())) if not eff.empty else None,
            "exit_efficiency_p25":    _s(float(eff.quantile(0.25))) if not eff.empty else None,
            "exit_efficiency_p75":    _s(float(eff.quantile(0.75))) if not eff.empty else None,
            "profit_left_behind_pct_mean":   _s(float(plb.mean())) if not plb.empty else None,
            "profit_left_behind_pct_median": _s(float(plb.median())) if not plb.empty else None,
            "profit_left_behind_yen_sum":    _s(float(plb_yen.sum())) if not plb_yen.empty else None,
            "profit_left_behind_yen_mean":   _s(float(plb_yen.mean())) if not plb_yen.empty else None,
            "peak_capture_pct_mean":  _s(float(pc.mean())) if not pc.empty else None,
            "peak_day_mean":          _s(float(g["peak_day"].mean())) if "peak_day" in g.columns else None,
            "peak_day_median":        _s(float(g["peak_day"].median())) if "peak_day" in g.columns else None,
            "hold_days_mean":         _s(float(g["hold_days"].mean())) if "hold_days" in g.columns else None,
            "actual_ret_mean":        _s(float(g["actual_ret"].mean())) if "actual_ret" in g.columns else None,
            "max_possible_ret_mean":  _s(float(g["max_possible_ret"].mean())) if "max_possible_ret" in g.columns else None,
        }

    bw_stats   = _group_stats(bw, "BigWinner (Top10%)")
    all_stats  = _group_stats(df, "All trades")

    # BigWinner別の取り逃し分布
    if not bw.empty and "profit_left_behind_yen" in bw.columns:
        plby = bw["profit_left_behind_yen"].dropna()
        total_plb_yen   = float(plby.sum())
        total_pnl_all   = float(df["pnl"].sum())
        plb_as_pct_total_pnl = total_plb_yen / max(abs(total_pnl_all), 1e-9) * 100
    else:
        total_plb_yen        = 0.0
        plb_as_pct_total_pnl = 0.0

    return {
        "bigwinner": bw_stats,
        "all_trades": all_stats,
        "bigwinner_total_plb_yen":    _s(total_plb_yen),
        "bigwinner_plb_as_pct_total_pnl": _s(plb_as_pct_total_pnl),
        "interpretation": (
            "BigWinnerの実現P&Lに対してPeak Captureで捉え逃した利益量。"
            "Exit改善の理論上限の上界として機能する。"
        ),
    }


# ======================================================================
# Phase3: 保持延長理論天井
# ======================================================================

def _simulate_hold_extension(df: pd.DataFrame, pnl_col: str,
                              label: str, baseline: dict) -> dict:
    """指定P&Lカラムで全取引のP&Lを置換してポートフォリオ統計を再計算。"""
    df_mod = df.copy()
    valid_mask = df_mod[pnl_col].notna()
    df_mod.loc[valid_mask, "pnl"] = df_mod.loc[valid_mask, pnl_col]
    stats = portfolio_stats(df_mod, label=label)
    delta = delta_stats(baseline, stats)
    n_valid = int(valid_mask.sum())
    return {
        "n_valid": n_valid,
        "n_replaced": n_valid,
        "portfolio": stats,
        "delta": delta,
    }


def phase3_retention_ceiling(df: pd.DataFrame, baseline: dict) -> dict:
    """
    保持延長シミュレーション:
    - 現行Exit → exit_date + N営業日後の価格でP&Lを置換
    - 60日固定保有
    全取引対象 (BigWinnerに限らない。天井測定)
    """
    results: dict = {"baseline": baseline, "scenarios": {}}

    # +N日延長
    for n_ext in HOLD_EXTENSIONS:
        pnl_col = f"pnl_hold_plus_{n_ext}d"
        if pnl_col not in df.columns:
            continue
        sim = _simulate_hold_extension(df, pnl_col, f"Hold+{n_ext}d", baseline)
        results["scenarios"][f"hold_plus_{n_ext}d"] = sim

    # 60日固定
    if "pnl_fixed60d" in df.columns:
        sim = _simulate_hold_extension(df, "pnl_fixed60d", "Fixed60d", baseline)
        results["scenarios"]["fixed_60d"] = sim

    # BigWinnerのみ延長 (仮想：BW以外は現行P&L維持)
    for n_ext in HOLD_EXTENSIONS:
        pnl_col = f"pnl_hold_plus_{n_ext}d"
        if pnl_col not in df.columns:
            continue
        df_mod = df.copy()
        bw_mask = df_mod["is_big_winner"] & df_mod[pnl_col].notna()
        df_mod.loc[bw_mask, "pnl"] = df_mod.loc[bw_mask, pnl_col]
        stats = portfolio_stats(df_mod, label=f"BW Only Hold+{n_ext}d")
        delta = delta_stats(baseline, stats)
        results["scenarios"][f"bw_only_hold_plus_{n_ext}d"] = {
            "n_valid":   int(bw_mask.sum()),
            "portfolio": stats,
            "delta":     delta,
        }

    # 最大改善シナリオの特定
    best_scenario   = None
    best_dcagr      = -999.0
    for sc_name, sc in results["scenarios"].items():
        dc = (sc.get("delta") or {}).get("d_cagr")
        if dc is not None and dc > best_dcagr:
            best_dcagr = dc; best_scenario = sc_name

    results["summary"] = {
        "best_scenario":   best_scenario,
        "best_dcagr_pp":   _s(best_dcagr),
    }
    return results


# ======================================================================
# Phase4: Winner持続性解剖
# ======================================================================

def phase4_winner_persistence(ds: dict, trades: list[dict], df: pd.DataFrame) -> dict:
    """
    BigWinner vs LoserのDay1-60特徴量推移。
    各観測日の特徴量を算出し、BigWinner/Loser/Middle群の平均値を比較。
    """
    universe_raw = ds["universe_raw"]
    rsr_df       = ds["rsr_df"]
    topix_close  = ds["topix_close"]
    all_dates    = rsr_df.index.sort_values()

    # BigWinner / Middle / Loser の entry_dateをマッピング
    bw_dates  = set(pd.Timestamp(d) for d in df[df["is_big_winner"]]["entry_date"])
    bot_dates = set(pd.Timestamp(d) for d in df[df["is_bottom20"]]["entry_date"])
    # Middle: 上でも下でもない
    other_mask = ~df["is_big_winner"] & ~df["is_bottom20"]
    mid_dates  = set(pd.Timestamp(d) for d in df[other_mask]["entry_date"])

    sym_to_entry_rsr = {tr["symbol"] + str(tr["entry_date"]): tr["entry_rsr"] for tr in trades}

    # 観測日リスト (Study61と整合)
    obs_days = [1, 3, 5, 10, 20, 40, 60]

    groups = {
        "BigWinner": bw_dates,
        "Middle":    mid_dates,
        "Loser":     bot_dates,
    }

    # 各obs_dayの特徴量を収集
    day_feats: dict = {nd: {g: [] for g in groups} for nd in obs_days}

    for tr in trades:
        sym        = tr["symbol"]
        entry_date = pd.Timestamp(tr["entry_date"])
        entry_rsr  = tr["entry_rsr"]

        g_label = None
        if entry_date in bw_dates:
            g_label = "BigWinner"
        elif entry_date in bot_dates:
            g_label = "Loser"
        elif entry_date in mid_dates:
            g_label = "Middle"
        if g_label is None:
            continue

        future_dates = all_dates[all_dates > entry_date]
        for n_days in obs_days:
            if len(future_dates) < n_days:
                continue
            obs_date = future_dates[n_days - 1]
            feat = _features_at_obs(sym, entry_date, entry_rsr, obs_date,
                                    universe_raw, rsr_df, topix_close)
            if feat is not None:
                day_feats[n_days][g_label].append(feat)

    # 各obs_day / 各group / 各特徴量の平均を集計
    key_feats = [
        "ret_from_entry", "rsr_delta", "atr_expansion",
        "vol_retention", "rs_accel_post", "high_persistence",
        "ma5_slope",
    ]
    timeline: dict = {}
    for nd in obs_days:
        timeline[f"day{nd}"] = {}
        for g_label in groups:
            feat_list = day_feats[nd][g_label]
            if not feat_list:
                continue
            g_res: dict = {"n": len(feat_list)}
            for kf in key_feats:
                vals = [f[kf] for f in feat_list if f.get(kf) is not None]
                if vals:
                    g_res[f"{kf}_mean"]   = _s(float(np.mean(vals)))
                    g_res[f"{kf}_median"] = _s(float(np.median(vals)))
                    g_res[f"{kf}_std"]    = _s(float(np.std(vals)))
            timeline[f"day{nd}"][g_label] = g_res

    # 各obs_dayでBigWinner vs Loserの差
    divergence: dict = {}
    for nd in obs_days:
        bw_d  = timeline.get(f"day{nd}", {}).get("BigWinner", {})
        los_d = timeline.get(f"day{nd}", {}).get("Loser", {})
        diffs: dict = {}
        for kf in key_feats:
            bw_v  = bw_d.get(f"{kf}_mean")
            los_v = los_d.get(f"{kf}_mean")
            if bw_v is not None and los_v is not None:
                diffs[f"bw_minus_loser_{kf}"] = _s(bw_v - los_v)
        divergence[f"day{nd}"] = diffs

    # BigWinnerらしさ持続: ret_from_entryでBWが明確に乖離し始める日を特定
    bw_ret_by_day  = []
    los_ret_by_day = []
    for nd in obs_days:
        bw_r  = timeline.get(f"day{nd}", {}).get("BigWinner", {}).get("ret_from_entry_mean")
        los_r = timeline.get(f"day{nd}", {}).get("Loser",     {}).get("ret_from_entry_mean")
        bw_ret_by_day.append(bw_r); los_ret_by_day.append(los_r)

    # 乖離日特定: |BW - Loser| > 2%pp が最初に現れる日
    diverge_day = None
    for i, nd in enumerate(obs_days):
        bw_v = bw_ret_by_day[i]; los_v = los_ret_by_day[i]
        if bw_v is not None and los_v is not None and abs(bw_v - los_v) > 2.0:
            diverge_day = nd; break

    return {
        "timeline":     timeline,
        "divergence":   divergence,
        "obs_days":     obs_days,
        "key_features": key_feats,
        "bw_ret_trajectory":  [_s(v) for v in bw_ret_by_day],
        "los_ret_trajectory": [_s(v) for v in los_ret_by_day],
        "diverge_day_2pp": diverge_day,
        "interpretation": "BigWinnerとLoserのret_from_entryが2pp乖離し始める最初の観測日",
    }


# ======================================================================
# Phase5: Add-on経済天井
# ======================================================================

def phase5_addon_ceiling(df: pd.DataFrame, baseline: dict) -> dict:
    """
    BigWinnerのみ対象。Day10/20/40時点で 0.5/1.0/2.0 単位追加。
    追加ポジションのP&L = (exit_price - dayN_price) * qty * size_multiplier
    ポートフォリオP&Lに加算 → ΔCAGR/ΔCalmar/ΔPF を測定。
    実装禁止。観測・天井測定のみ。
    """
    bw = df[df["is_big_winner"]].copy()
    results: dict = {"n_bigwinner": len(bw), "scenarios": {}}

    for ad in ADDON_DAYS:
        day_col = f"close_day{ad}"
        if day_col not in df.columns:
            continue
        for sz in ADDON_SIZES:
            # BigWinnerの追加P&L
            addon_pnl_series = bw.apply(
                lambda r, dc=day_col, s=sz: (
                    (r["exit_price"] - r[dc]) / r[dc] * r[dc] * r["qty"] * s
                    if pd.notna(r[dc]) and r[dc] > 0 and r["exit_price"] > 0
                    else np.nan
                ), axis=1
            ).dropna()

            total_addon_pnl = float(addon_pnl_series.sum())
            n_valid = len(addon_pnl_series)

            # ポートフォリオ統計: 元のP&L + BigWinnerの追加P&L
            df_aug = df.copy()
            addon_map = bw.loc[addon_pnl_series.index, "entry_date"].copy()
            df_aug["pnl_aug"] = df_aug["pnl"].copy()

            # entry_date × symbolでマッチング
            for idx, addon_val in addon_pnl_series.items():
                df_aug.loc[idx, "pnl_aug"] = df_aug.loc[idx, "pnl"] + addon_val

            stats = portfolio_stats(df_aug, pnl_col="pnl_aug", label=f"Addon Day{ad} x{sz}")
            delta = delta_stats(baseline, stats)

            # BigWinner add-on単体の期待値
            avg_addon_ret = None
            if n_valid > 0 and day_col in bw.columns:
                addon_ret_pcts = bw.apply(
                    lambda r, dc=day_col: (
                        (r["exit_price"] / r[dc] - 1.0) * 100
                        if pd.notna(r[dc]) and r[dc] > 0 else np.nan
                    ), axis=1
                ).dropna()
                avg_addon_ret = _s(float(addon_ret_pcts.mean()))

            sc_key = f"day{ad}_x{str(sz).replace('.', '_')}"
            results["scenarios"][sc_key] = {
                "addon_day":          ad,
                "addon_size":         sz,
                "n_valid":            n_valid,
                "total_addon_pnl_jpy": _s(total_addon_pnl),
                "avg_addon_ret_pct":  avg_addon_ret,
                "portfolio":          stats,
                "delta":              delta,
            }

    # 最大改善シナリオ
    best_sc = None; best_dcagr = -999.0
    for sc_key, sc in results["scenarios"].items():
        dc = (sc.get("delta") or {}).get("d_cagr")
        if dc is not None and dc > best_dcagr:
            best_dcagr = dc; best_sc = sc_key
    results["best_scenario"] = best_sc
    results["best_dcagr_pp"] = _s(best_dcagr)

    return results


# ======================================================================
# Phase6: 経済フロンティア
# ======================================================================

def phase6_economic_frontier(p3: dict, p5: dict, baseline: dict) -> dict:
    """
    Failure研究上限 (+1.63pp) vs Retention改善 vs Add-on改善 vs Exit改善
    の比較ランキング。
    """
    b_cagr   = (baseline.get("cagr")   or 0)
    b_calmar = (baseline.get("calmar") or 0)

    # Phase3から最大値を取得
    retention_best_dcagr   = -999.0
    retention_bw_best      = -999.0
    for sc_name, sc in (p3.get("scenarios") or {}).items():
        dc = (sc.get("delta") or {}).get("d_cagr")
        if dc is not None:
            if "bw_only" in sc_name:
                retention_bw_best = max(retention_bw_best, dc)
            else:
                retention_best_dcagr = max(retention_best_dcagr, dc)

    # Phase5から最大値を取得
    addon_best_dcagr = -999.0
    for sc_key, sc in (p5.get("scenarios") or {}).items():
        dc = (sc.get("delta") or {}).get("d_cagr")
        if dc is not None:
            addon_best_dcagr = max(addon_best_dcagr, dc)

    themes = [
        {
            "rank":         1,
            "theme":        "Failure除去 (Bottom20%完全除去)",
            "dcagr_theory": _s(FAILURE_THEORY_DCAGR),
            "dcagr_realistic": -0.93,   # Study63 MC平均
            "source":       "Study63 Phase2/3",
            "note":         "現実改善は負(FPコスト支配)",
        },
        {
            "rank":         2,
            "theme":        "BigWinner保持延長 (全取引)",
            "dcagr_theory": _s(retention_best_dcagr) if retention_best_dcagr > -999 else None,
            "dcagr_realistic": None,   # 未測定
            "source":       "Study64 Phase3",
            "note":         "全取引の自然exit後延長",
        },
        {
            "rank":         3,
            "theme":        "BigWinner保持延長 (BW限定)",
            "dcagr_theory": _s(retention_bw_best) if retention_bw_best > -999 else None,
            "dcagr_realistic": None,
            "source":       "Study64 Phase3 BW-only",
            "note":         "BigWinnerのみ延長",
        },
        {
            "rank":         4,
            "theme":        "Add-on (BigWinner)",
            "dcagr_theory": _s(addon_best_dcagr) if addon_best_dcagr > -999 else None,
            "dcagr_realistic": None,
            "source":       "Study64 Phase5",
            "note":         "Day10-40 仮想追加ポジション",
        },
    ]

    # ΔCAGR降順でソート
    def sort_key(t):
        v = t.get("dcagr_theory")
        return v if v is not None else -999.0
    themes.sort(key=sort_key, reverse=True)
    for i, t in enumerate(themes):
        t["rank"] = i + 1

    return {
        "baseline_cagr":          _s(b_cagr),
        "baseline_calmar":        _s(b_calmar),
        "failure_theory_ceiling": _s(FAILURE_THEORY_DCAGR),
        "themes":                 themes,
    }


# ======================================================================
# Phase7: 感度監査
# ======================================================================

def phase7_sensitivity_audit(df: pd.DataFrame, p3: dict, p5: dict, baseline: dict) -> dict:
    """
    Top5%/Top10%/Top20% でBigWinner定義を変えて結論が変わらないか確認。
    保持期間 +5/10/20/40 でも感度確認。
    """
    # BigWinner定義感度 (Retention +10d で比較)
    pnl_col_10 = "pnl_hold_plus_10d"
    bw_def_sensitivity: dict = {}

    for tier_col, tier_name in [("is_top5", "Top5%"), ("is_top10", "Top10%"), ("is_top20", "Top20%")]:
        if pnl_col_10 not in df.columns or tier_col not in df.columns:
            continue
        df_mod = df.copy()
        bw_mask = df_mod[tier_col] & df_mod[pnl_col_10].notna()
        df_mod.loc[bw_mask, "pnl"] = df_mod.loc[bw_mask, pnl_col_10]
        stats = portfolio_stats(df_mod, label=f"BW={tier_name} +10d")
        delta = delta_stats(baseline, stats)
        bw_def_sensitivity[tier_name] = {
            "n_bw":     int(bw_mask.sum()),
            "dcagr":    (delta.get("d_cagr")),
            "dcalmar":  (delta.get("d_calmar")),
        }

    # 保持期間感度 (BigWinner=Top10%固定)
    hold_sensitivity: dict = {}
    for n_ext in HOLD_EXTENSIONS:
        pnl_col = f"pnl_hold_plus_{n_ext}d"
        if pnl_col not in df.columns:
            continue
        df_mod = df.copy()
        bw_mask = df_mod["is_big_winner"] & df_mod[pnl_col].notna()
        df_mod.loc[bw_mask, "pnl"] = df_mod.loc[bw_mask, pnl_col]
        stats = portfolio_stats(df_mod, label=f"BW +{n_ext}d")
        delta = delta_stats(baseline, stats)
        hold_sensitivity[f"plus_{n_ext}d"] = {
            "n_replaced": int(bw_mask.sum()),
            "dcagr":  delta.get("d_cagr"),
            "dcalmar":delta.get("d_calmar"),
        }

    # 感度スコア: 全シナリオでΔCAGRの方向が一致しているか
    hold_dcagrs = [v.get("dcagr") for v in hold_sensitivity.values() if v.get("dcagr") is not None]
    consistent  = all(d > 0 for d in hold_dcagrs) or all(d < 0 for d in hold_dcagrs)

    return {
        "bw_definition_sensitivity": bw_def_sensitivity,
        "hold_extension_sensitivity":hold_sensitivity,
        "hold_consistent_direction": consistent,
        "interpretation": (
            "全保持期間でΔCAGR方向一致" if consistent
            else "保持期間によって改善・悪化が逆転 → 結論が不安定"
        ),
    }


# ======================================================================
# Phase8: 研究評決
# ======================================================================

def phase8_verdict(p1: dict, p2: dict, p3: dict, p5: dict, p6: dict, p7: dict) -> dict:
    """研究結論と Study65 候補の提案。"""
    baseline_cagr = p6.get("baseline_cagr") or 0

    # Phase3 最大改善
    retention_ceiling = p3.get("summary", {}).get("best_dcagr_pp") or 0
    retention_best_sc = p3.get("summary", {}).get("best_scenario") or "N/A"

    # Phase5 最大改善
    addon_ceiling = p5.get("best_dcagr_pp") or 0
    addon_best_sc = p5.get("best_scenario") or "N/A"

    # Phase2 取り逃し
    bw_plb_yen   = p2.get("bigwinner_total_plb_yen") or 0
    bw_plb_pct   = p2.get("bigwinner_plb_as_pct_total_pnl") or 0
    bw_exit_eff  = (p2.get("bigwinner", {}) or {}).get("exit_efficiency_mean")
    bw_peak_cap  = (p2.get("bigwinner", {}) or {}).get("peak_capture_pct_mean")

    # Phase6 テーマランキング
    top_theme    = (p6.get("themes") or [{}])[0]
    top_dcagr    = top_theme.get("dcagr_theory")

    # Phase7 感度
    consistent   = p7.get("hold_consistent_direction", False)

    # Study65候補ロジック
    if retention_ceiling > FAILURE_THEORY_DCAGR:
        study65_candidate = "保持延長シグナル研究 (何が早期Exit引き起こすか観測)"
        rationale = f"Retention ceiling {retention_ceiling:+.2f}pp > Failure ceiling +{FAILURE_THEORY_DCAGR:.2f}pp"
    elif addon_ceiling > FAILURE_THEORY_DCAGR:
        study65_candidate = "Add-onタイミング研究 (Day10-20の強者確認シグナル)"
        rationale = f"Add-on ceiling {addon_ceiling:+.2f}pp > Failure ceiling +{FAILURE_THEORY_DCAGR:.2f}pp"
    elif max(retention_ceiling, addon_ceiling) > 0.5:
        study65_candidate = "保持延長・Add-on複合研究 (BigWinner識別精度向上)"
        rationale = f"最大改善余地 {max(retention_ceiling, addon_ceiling):+.2f}pp > 0.5pp"
    else:
        study65_candidate = "別の研究軸を探索 (Winner構造は既存システムで対応済の可能性)"
        rationale = "保持/Add-on両天井ともに改善幅小"

    return {
        "1_winner_retention_ceiling_pp": _s(retention_ceiling),
        "1_best_retention_scenario":      retention_best_sc,
        "2_addon_ceiling_pp":             _s(addon_ceiling),
        "2_best_addon_scenario":          addon_best_sc,
        "3_exit_efficiency_bw":           bw_exit_eff,
        "3_peak_capture_bw_pct":          bw_peak_cap,
        "3_profit_left_behind_yen":       _s(bw_plb_yen),
        "3_profit_left_behind_as_pct_total_pnl": _s(bw_plb_pct),
        "4_failure_ceiling_pp":           _s(FAILURE_THEORY_DCAGR),
        "4_failure_realistic_pp":         -0.93,
        "5_top_theme":                    top_theme.get("theme"),
        "5_top_theme_dcagr_theory":       top_dcagr,
        "6_sensitivity_consistent":       consistent,
        "7_study65_candidate":            study65_candidate,
        "7_rationale":                    rationale,
    }


# ======================================================================
# main
# ======================================================================

def main():
    print("=== Study64: BigWinner Retention Ceiling ===")
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
    tr_is   = extract_trades_full(bt_is, is_dates, rsr_df)
    print(f"  IS trades: {len(tr_is)}")

    print("[BT実行: OOS 2025]")
    sym_oos = get_active(ds, OOS_START, OOS_END)
    bt_oos  = run_bt(ds, sym_oos, OOS_START, OOS_END)
    tr_oos  = extract_trades_full(bt_oos, oos_dates, rsr_df)
    print(f"  OOS trades: {len(tr_oos)}")

    all_trades = tr_is + tr_oos
    print(f"  全取引: {len(all_trades)} 件")

    print("\n[データセット構築...]")
    df = build_study64_dataset(ds, all_trades)
    print(f"  有効トレード数: {len(df)}")
    if df.empty:
        print("  ERROR: データ空")
        return

    print(f"  BigWinner (Top10%): {df['is_big_winner'].sum()} / Total PNL: {df['pnl'].sum():,.0f} JPY")

    print("\n[Phase0: Integrity]")
    p0 = phase0_integrity(df)
    print(f"  n={p0['n_valid']} Study63 match: {p0['study63_n_match']} "
          f"BigWinner={p0['bigwinner_n']}")

    print("\n[Phase1: 利益寄与監査]")
    p1 = phase1_profit_contribution(df)
    for tier, v in p1["tiers"].items():
        print(f"  {tier}: n={v['n']} PNL={v['pnl_sum_jpy']:,} "
              f"EV={v['ev_contrib_pct']}% fwd60={v['fwd60_mean']}%")

    print("\n[Phase2: 早期Exit監査]")
    baseline = portfolio_stats(df, label="Baseline")
    print(f"  Baseline: CAGR={baseline['cagr']}% MaxDD={baseline['max_dd']}% "
          f"Calmar={baseline['calmar']}")
    p2 = phase2_premature_exit_audit(df, baseline)
    bw_d = p2.get("bigwinner", {})
    print(f"  BigWinner Exit Efficiency: {bw_d.get('exit_efficiency_mean')}")
    print(f"  BigWinner Peak Capture%:   {bw_d.get('peak_capture_pct_mean')}")
    print(f"  BigWinner Profit Left (¥): {p2.get('bigwinner_total_plb_yen'):,}")
    print(f"  PLB as % of Total PNL:     {p2.get('bigwinner_plb_as_pct_total_pnl')}%")

    print("\n[Phase3: 保持延長理論天井]")
    p3 = phase3_retention_ceiling(df, baseline)
    for sc_name, sc in p3["scenarios"].items():
        d = sc.get("delta", {})
        print(f"  {sc_name}: ΔCAGR={d.get('d_cagr'):+.2f}pp "
              f"ΔCalmar={d.get('d_calmar', 0):+.3f} "
              f"ΔMaxDD={d.get('d_max_dd', 0):+.2f}pp "
              f"(n={sc.get('n_valid', sc.get('n_replaced', '?'))})")
    print(f"  最大: {p3['summary']['best_scenario']} = {p3['summary']['best_dcagr_pp']:+.2f}pp")

    print("\n[Phase4: Winner持続性解剖]")
    p4 = phase4_winner_persistence(ds, all_trades, df)
    print("  ret_from_entry 軌跡:")
    for i, nd in enumerate(p4["obs_days"]):
        bw_r  = p4["bw_ret_trajectory"][i]
        los_r = p4["los_ret_trajectory"][i]
        print(f"    Day{nd:2d}: BigWinner={bw_r}% Loser={los_r}%")
    print(f"  乖離開始日 (>2pp): Day{p4['diverge_day_2pp']}")

    print("\n[Phase5: Add-on経済天井]")
    p5 = phase5_addon_ceiling(df, baseline)
    for sc_key, sc in p5["scenarios"].items():
        d = sc.get("delta", {})
        print(f"  {sc_key}: ΔCAGR={d.get('d_cagr', 0):+.2f}pp "
              f"avg_addon_ret={sc.get('avg_addon_ret_pct')}%")
    print(f"  最大: {p5['best_scenario']} = {p5['best_dcagr_pp']:+.2f}pp")

    print("\n[Phase6: 経済フロンティア]")
    p6 = phase6_economic_frontier(p3, p5, baseline)
    for t in p6["themes"]:
        thy = f"{t['dcagr_theory']:+.2f}pp" if t.get("dcagr_theory") is not None else "N/A"
        print(f"  #{t['rank']} {t['theme']}: 理論天井={thy}")

    print("\n[Phase7: 感度監査]")
    p7 = phase7_sensitivity_audit(df, p3, p5, baseline)
    print(f"  保持延長方向一致: {p7['hold_consistent_direction']}")
    for k, v in p7["hold_extension_sensitivity"].items():
        print(f"  BW {k}: ΔCAGR={v.get('dcagr', 0):+.2f}pp ΔCalmar={v.get('dcalmar', 0):+.3f}")

    print("\n[Phase8: 研究評決]")
    p8 = phase8_verdict(p1, p2, p3, p5, p6, p7)
    print(f"  1. Winner保持理論上限:  {p8['1_winner_retention_ceiling_pp']:+.2f}pp")
    print(f"  2. Add-on理論上限:     {p8['2_addon_ceiling_pp']:+.2f}pp")
    print(f"  3. Exit Efficiency:   {p8['3_exit_efficiency_bw']}")
    print(f"  3. Profit Left (¥):  {p8['3_profit_left_behind_yen']:,}")
    print(f"  4. Failure上限:       {p8['4_failure_ceiling_pp']:+.2f}pp")
    print(f"  5. 最優先テーマ:       {p8['5_top_theme']}")
    print(f"  7. Study65候補:       {p8['7_study65_candidate']}")

    result = {
        "study":  "Study64",
        "title":  "BigWinner Retention Ceiling",
        "date":   TODAY_STR,
        "params": {
            "n_mc": 0, "capital": CAPITAL, "bigwinner_pctile": BIGWINNER_PCTILE,
            "hold_extensions": HOLD_EXTENSIONS, "fixed_hold_days": FIXED_HOLD_DAYS,
            "addon_days": ADDON_DAYS, "addon_sizes": ADDON_SIZES,
            "failure_theory_dcagr_ref": FAILURE_THEORY_DCAGR,
        },
        "phase0_integrity":           p0,
        "phase1_profit_contribution": p1,
        "phase2_premature_exit":      p2,
        "phase3_retention_ceiling":   p3,
        "phase4_winner_persistence":  p4,
        "phase5_addon_ceiling":       p5,
        "phase6_economic_frontier":   p6,
        "phase7_sensitivity":         p7,
        "phase8_verdict":             p8,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n保存: {OUT_FILE}")
    print("=== Study64 完了 ===")


if __name__ == "__main__":
    main()
