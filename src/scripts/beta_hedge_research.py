"""
scripts/beta_hedge_research.py

ベータヘッジ研究スクリプト（1306.T 空売りシミュレーション）

Step 2: hedge_ratio sweep（0.0〜0.5）で感度分析
Step 3: 条件付きヘッジ（Bear + vol高 / Bear + 金利上昇）の比較

実行方法:
  cd C:/ai-trading
  python src/scripts/beta_hedge_research.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.backtest.wf_dynamic_universe import (
    load_data, _build_active,
    BULL_ACTIVE_N, BEAR_ACTIVE_N, IS_FULL, TRUE_OOS,
)
from src.config_loader import load_strategy_config
import src.backtest.composite_alpha_bt as _bt

# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
SHOCK_MODE          = "composite"
BORROW_COST_ANNUAL  = 0.005    # 空売りコスト 年率 0.5%
TRADE_COST          = 0.001    # 往復 0.1%
SEG_2022            = ("2022-01-01", "2022-12-31")
SEG_2025            = TRUE_OOS  # ("2025-01-01", "2025-12-31")


# ------------------------------------------------------------------ #
# ヘルパー: セグメント指標計算
# ------------------------------------------------------------------ #
def _segment_sharpe(eq: pd.Series) -> float:
    dr = eq.pct_change().dropna()
    if dr.std() <= 0 or len(dr) < 5:
        return 0.0
    return float(dr.mean() / dr.std() * np.sqrt(252))


def _segment_maxdd(eq: pd.Series) -> float:
    roll_max = eq.expanding().max()
    dd = (eq - roll_max) / roll_max
    return float(dd.min() * 100)


def _segment_cagr(eq: pd.Series) -> float:
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    years = len(eq) / 252
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 0.01)) - 1) * 100


# ------------------------------------------------------------------ #
# ヘッジシミュレーション
# ------------------------------------------------------------------ #
def simulate_hedge(
    equity_curve: pd.Series,
    long_notional: pd.Series,
    topix_ret: pd.Series,
    is_bear: pd.Series,          # True=Bear (TOPIX < MA200)
    hedge_ratio: float,
) -> pd.Series:
    """
    ヘッジ込みの equity_curve を返す。

    hedge_notional = hedge_ratio * long_notional（Bear時のみ）
    hedge_pnl      = -hedge_notional * topix_ret（ショートなのでリターン反転）
    borrow_cost    = hedge_notional * (0.005 / 252)
    switch_cost    = |hedge_notional.diff()| * 0.001

    Returns: hedged_equity_curve (pd.Series, same index as equity_curve)
    """
    if hedge_ratio == 0.0:
        return equity_curve.copy()

    idx = equity_curve.index
    _tr  = topix_ret.reindex(idx, method="ffill").fillna(0.0)
    _ln  = long_notional.reindex(idx, method="ffill").fillna(0.0)
    _bear = is_bear.reindex(idx, method="ffill").fillna(False).astype(float)

    hedge_notional = _ln * hedge_ratio * _bear
    hedge_pnl      = -hedge_notional * _tr
    borrow_cost    = hedge_notional * (BORROW_COST_ANNUAL / 252)
    switch_cost    = hedge_notional.diff().abs().fillna(0.0) * TRADE_COST

    net_hedge_pnl  = hedge_pnl - borrow_cost - switch_cost

    # equity_curve を再構築: 元の daily_pnl にヘッジ pnl を加算
    daily_pnl_orig = equity_curve.diff().fillna(0.0)
    hedged_daily   = daily_pnl_orig + net_hedge_pnl
    hedged_eq      = equity_curve.iloc[0] + hedged_daily.cumsum()
    hedged_eq.iloc[0] = equity_curve.iloc[0]

    return hedged_eq


def _hedge_cost_info(
    long_notional: pd.Series,
    topix_ret: pd.Series,
    is_bear: pd.Series,
    hedge_ratio: float,
) -> dict:
    """ヘッジコスト関連の集計情報を返す"""
    if hedge_ratio == 0.0:
        return {"hedge_cost_total": 0.0, "hedge_days": 0, "turnover": 0}

    idx = long_notional.index
    _ln   = long_notional.reindex(idx).fillna(0.0)
    _bear = is_bear.reindex(idx, method="ffill").fillna(False).astype(float)

    hedge_notional = _ln * hedge_ratio * _bear
    borrow_cost    = hedge_notional * (BORROW_COST_ANNUAL / 252)
    switch_cost    = hedge_notional.diff().abs().fillna(0.0) * TRADE_COST

    hedge_cost_total = float(borrow_cost.sum() + switch_cost.sum())
    hedge_days       = int(_bear.sum())
    turnover         = int((hedge_notional.diff().abs() > 0).sum())

    return {
        "hedge_cost_total": hedge_cost_total,
        "hedge_days": hedge_days,
        "turnover": turnover,
    }


# ------------------------------------------------------------------ #
# run_scenario を1回実行
# ------------------------------------------------------------------ #
def _run_scenario(
    base_cfg, universe_raw, rsr_df, alpha_df, regime_df, all_syms,
    tech_matrices, topix_close, start: str, end: str,
) -> dict:
    active_df = _build_active(
        universe_raw, topix_close, all_syms, rsr_df,
        BULL_ACTIVE_N, BEAR_ACTIVE_N, "2017-01-01", end, None
    )
    return _bt.run_scenario(
        scenario             = "BASELINE",
        universe_raw         = universe_raw,
        rsr_df               = rsr_df,
        alpha_df             = alpha_df,
        regime_df            = regime_df,
        trade_syms           = all_syms,
        rsr_syms             = all_syms,
        cfg                  = base_cfg,
        start                = start,
        end                  = end,
        capital              = float(base_cfg.portfolio.capital),
        verbose              = False,
        tech_matrices        = tech_matrices,
        topix_close          = topix_close,
        min_hold             = int(base_cfg.risk.min_hold_days),
        market_shock_mode    = SHOCK_MODE,
        enable_volume_filter = True,
        sym_active_df        = active_df,
    )


# ------------------------------------------------------------------ #
# Bear フラグ計算
# ------------------------------------------------------------------ #
def calc_bear_flag(topix_close: pd.Series) -> pd.Series:
    """TOPIX < MA200 → True (Bear)"""
    ma200  = topix_close.rolling(200, min_periods=100).mean()
    return (topix_close < ma200)


# ------------------------------------------------------------------ #
# 条件付きヘッジ: vol高フィルター
# ------------------------------------------------------------------ #
def calc_vol_regime(topix_close: pd.Series, window: int = 20, pct: float = 75.0) -> pd.Series:
    """
    TOPIX の 20日ヒストリカルボラティリティが過去 pct パーセンタイルを超えたら True
    （高 vol = リスクオフ環境の代替指標）
    """
    log_ret = np.log(topix_close / topix_close.shift(1))
    vol20   = log_ret.rolling(window).std() * np.sqrt(252)
    vol_pct = vol20.expanding(min_periods=60).quantile(pct / 100)
    return vol20 > vol_pct


def calc_us10y_regime(data_dir: Path, window: int = 20) -> pd.Series | None:
    """
    米10年債利回りの 20日変化が 0 を超えたら True（金利上昇フラグ）
    data/us10y.csv がなければ None を返す
    """
    candidates = [
        data_dir / "us10y.csv",
        data_dir / "US10Y.csv",
        data_dir / "tnx.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                col = df.columns[0]
                s   = df[col].dropna()
                return s.diff(window) > 0.0
            except Exception:
                pass
    return None


# ------------------------------------------------------------------ #
# Step 2: hedge_ratio sweep 感度分析
# ------------------------------------------------------------------ #
def run_sweep(
    equity_full: pd.Series,
    long_notional_full: pd.Series,
    topix_close: pd.Series,
    is_bear: pd.Series,
    base_sharpe: float,
    base_cagr: float,
) -> list[dict]:
    """
    hedge_ratio = 0.0〜0.5 で sweep。
    equity_full は IS + OOS を含む全期間 or IS のみでも可。
    2022 / 2025 セグメントをスライスして指標を計算。
    """
    topix_ret = topix_close.pct_change().fillna(0.0)

    rows = []
    for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        hedged_eq = simulate_hedge(equity_full, long_notional_full, topix_ret, is_bear, ratio)

        # 2022 セグメント
        eq_2022 = hedged_eq.loc[SEG_2022[0]:SEG_2022[1]]
        dd_2022 = _segment_maxdd(eq_2022) if len(eq_2022) >= 5 else float("nan")
        cagr_2022 = _segment_cagr(eq_2022) if len(eq_2022) >= 5 else float("nan")

        # 2025 OOS セグメント
        eq_2025 = hedged_eq.loc[SEG_2025[0]:SEG_2025[1]]
        sh_2025 = _segment_sharpe(eq_2025) if len(eq_2025) >= 5 else float("nan")
        dd_2025 = _segment_maxdd(eq_2025) if len(eq_2025) >= 5 else float("nan")

        # ヘッジコスト（全期間）
        cost_info = _hedge_cost_info(long_notional_full, topix_ret, is_bear, ratio)

        # hedge_cost_vs_cagr: ヘッジコスト / (CAGR収益)
        capital = float(equity_full.iloc[0])
        years   = len(equity_full) / 252
        cagr_pnl = capital * ((1 + base_cagr / 100) ** years - 1) if base_cagr > 0 else 1.0
        hedge_cost_vs_cagr = (cost_info["hedge_cost_total"] / max(cagr_pnl, 1.0) * 100) if ratio > 0 else 0.0

        rows.append({
            "hedge_ratio":       ratio,
            "dd_2022":           dd_2022,
            "cagr_2022":         cagr_2022,
            "sharpe_2025":       sh_2025,
            "dd_2025":           dd_2025,
            "hedge_cost_total":  cost_info["hedge_cost_total"],
            "hedge_cost_vs_cagr": hedge_cost_vs_cagr,
            "hedge_days":        cost_info["hedge_days"],
            "turnover":          cost_info["turnover"],
        })
    return rows


def print_sweep_table(
    rows: list[dict],
    base_dd_2022: float,
    base_sharpe_2025: float,
    base_sharpe_full: float,
    initial_nav: float,
) -> float:
    """表を出力して推奨 hedge_ratio を返す"""
    print(f"\n=== ベータヘッジ感度分析（1306.T 空売り、Bear時のみ） ===")
    print(f"ベースライン: 2022 MaxDD={base_dd_2022:.1f}%, Full Sharpe={base_sharpe_full:.3f}, "
          f"Initial NAV=¥{initial_nav:,.0f}")
    print()
    header = (
        f"| {'hedge_ratio':>12} | {'2022 MaxDD':>10} | {'2022 CAGR':>10} | "
        f"{'2025 Sharpe':>11} | {'2025 MaxDD':>10} | "
        f"{'hedge_cost':>12} | {'hedge_days':>10} | {'turnover':>9} |"
    )
    sep = "|" + "|".join(["-" * (len(h)) for h in header.split("|")[1:-1]]) + "|"
    print(header)
    print(sep)

    for r in rows:
        ratio = r["hedge_ratio"]
        label = f"{ratio:.1f}" + (" (base)" if ratio == 0.0 else "")
        dd2 = f"{r['dd_2022']:.1f}%" if not np.isnan(r['dd_2022']) else "N/A"
        cg2 = f"{r['cagr_2022']:+.1f}%" if not np.isnan(r['cagr_2022']) else "N/A"
        sh5 = f"{r['sharpe_2025']:.3f}" if not np.isnan(r['sharpe_2025']) else "N/A"
        dd5 = f"{r['dd_2025']:.1f}%" if not np.isnan(r['dd_2025']) else "N/A"
        hc  = f"¥{r['hedge_cost_total']:,.0f}" if ratio > 0 else "-"
        hd  = f"{r['hedge_days']}日" if ratio > 0 else "-"
        tv  = f"{r['turnover']}回" if ratio > 0 else "-"
        print(
            f"| {label:>12} | {dd2:>10} | {cg2:>10} | {sh5:>11} | {dd5:>10} | "
            f"{hc:>12} | {hd:>10} | {tv:>9} |"
        )

    # 推奨判定
    base_row = rows[0]
    best_ratio = 0.0
    best_reason = ""
    for r in rows[1:]:
        dd_imp   = base_row["dd_2022"] - r["dd_2022"]  # 負の方向に改善 → 差は正
        sh_deg   = base_row["sharpe_2025"] - r["sharpe_2025"] if not np.isnan(r["sharpe_2025"]) else 999
        cost_pct = r["hedge_cost_vs_cagr"]
        if dd_imp >= 3.0 and sh_deg <= 0.15 and cost_pct <= 2.0:
            best_ratio  = r["hedge_ratio"]
            best_reason = (
                f"2022 DD改善={dd_imp:+.1f}pt, 2025 Sharpe劣化={sh_deg:.3f}, "
                f"hedge_cost_vs_CAGR={cost_pct:.1f}%"
            )

    # 採用基準サマリ
    print()
    print("採用基準:")
    thr_dd   = next((r["hedge_ratio"] for r in rows[1:] if base_row["dd_2022"] - r["dd_2022"] >= 3.0), None)
    thr_sh   = next((r["hedge_ratio"] for r in rows[1:] if not np.isnan(r["sharpe_2025"]) and base_row["sharpe_2025"] - r["sharpe_2025"] > 0.15), None)
    thr_cost = next((r["hedge_ratio"] for r in rows[1:] if r["hedge_cost_vs_cagr"] > 2.0), None)
    print(f"  2022 DD改善 >= 3.0pt    : hedge_ratio >= {thr_dd if thr_dd else 'なし'} で達成")
    print(f"  2025 Sharpe劣化 <= 0.15 : hedge_ratio <= {(thr_sh - 0.1):.1f} で維持" if thr_sh else "  2025 Sharpe劣化 <= 0.15 : 全比率で維持")
    print(f"  hedge_cost <= CAGR 2pt  : hedge_ratio <= {(thr_cost - 0.1):.1f} で許容" if thr_cost else "  hedge_cost <= CAGR 2pt  : 全比率で許容")
    print()
    if best_ratio > 0:
        print(f"推奨 hedge_ratio: {best_ratio:.1f}（理由: {best_reason}）")
    else:
        print("推奨 hedge_ratio: 0.0（採用基準を満たす ratio なし）")

    return best_ratio


# ------------------------------------------------------------------ #
# Step 3: 条件付きヘッジ比較
# ------------------------------------------------------------------ #
def run_conditional_hedge(
    equity_full: pd.Series,
    long_notional_full: pd.Series,
    topix_close: pd.Series,
    is_bear: pd.Series,
    vol_regime: pd.Series,
    us10y_regime: pd.Series | None,
    best_ratio: float,
) -> None:
    """条件付きヘッジ（Bear + vol高 / Bear + 金利上昇）の比較"""
    topix_ret = topix_close.pct_change().fillna(0.0)

    conditions = {
        "bear のみ": is_bear,
        "bear + vol高": (is_bear & vol_regime),
    }
    if us10y_regime is not None:
        conditions["bear + 金利上昇"] = (is_bear & us10y_regime.reindex(is_bear.index, method="ffill").fillna(False))

    print(f"\n=== 条件付きヘッジ比較（hedge_ratio={best_ratio:.1f}） ===")
    header = (
        f"| {'条件':>18} | {'2022 MaxDD':>10} | {'2025 Sharpe':>11} | "
        f"{'hedge_days':>10} | {'hedge_cost':>12} |"
    )
    sep = "|" + "|".join(["-" * (len(h)) for h in header.split("|")[1:-1]]) + "|"
    print(header)
    print(sep)

    best_cond = "bear のみ"
    best_sh25 = -999.0
    rows_cond = []

    for label, cond_flag in conditions.items():
        cond_s = cond_flag.reindex(equity_full.index, method="ffill").fillna(False)

        hedged_eq  = simulate_hedge(equity_full, long_notional_full, topix_ret, cond_s, best_ratio)
        cost_info  = _hedge_cost_info(long_notional_full, topix_ret, cond_s, best_ratio)

        eq_2022 = hedged_eq.loc[SEG_2022[0]:SEG_2022[1]]
        eq_2025 = hedged_eq.loc[SEG_2025[0]:SEG_2025[1]]

        dd_2022  = _segment_maxdd(eq_2022) if len(eq_2022) >= 5 else float("nan")
        sh_2025  = _segment_sharpe(eq_2025) if len(eq_2025) >= 5 else float("nan")

        rows_cond.append({
            "label": label,
            "dd_2022": dd_2022,
            "sh_2025": sh_2025,
            "hedge_days": cost_info["hedge_days"],
            "hedge_cost": cost_info["hedge_cost_total"],
        })

        dd2 = f"{dd_2022:.1f}%" if not np.isnan(dd_2022) else "N/A"
        sh5 = f"{sh_2025:.3f}" if not np.isnan(sh_2025) else "N/A"
        hd  = f"{cost_info['hedge_days']}日"
        hc  = f"¥{cost_info['hedge_cost_total']:,.0f}"
        print(f"| {label:>18} | {dd2:>10} | {sh5:>11} | {hd:>10} | {hc:>12} |")

        if not np.isnan(sh_2025) and sh_2025 > best_sh25:
            best_sh25 = sh_2025
            best_cond = label

    print()
    best_r = next((r for r in rows_cond if r["label"] == best_cond), None)
    if best_r:
        print(
            f"推奨条件: {best_cond}"
            f"（理由: 2025 Sharpe={best_r['sh_2025']:.3f} 最大、"
            f"2022 MaxDD={best_r['dd_2022']:.1f}%、hedge_days={best_r['hedge_days']}日）"
        )


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    base_cfg = load_strategy_config()

    print("=" * 70)
    print("  ベータヘッジ研究スクリプト（1306.T 空売りシミュレーション）")
    print("=" * 70)

    (universe_raw, rsr_df, rsr_df_full, alpha_df, regime_df, rsr_syms, all_syms,
     tech_matrices, topix_close, base_cfg) = load_data(base_cfg)

    # ── データディレクトリ（us10y.csv 探索用）────────────────────────
    data_dir = Path(__file__).resolve().parents[2] / "data"

    # ── Full IS + OOS を1つのバックテストで実行 ─────────────────────
    # IS_FULL = 2018-2024, TRUE_OOS = 2025
    # 一括実行が難しいため、IS と OOS を別々に実行して concat
    print(f"\n[1/2] Full IS ({IS_FULL[0]}〜{IS_FULL[1]}) バックテスト実行...")
    r_is = _run_scenario(
        base_cfg, universe_raw, rsr_df, alpha_df, regime_df, all_syms,
        tech_matrices, topix_close, IS_FULL[0], IS_FULL[1],
    )
    if not r_is or r_is.get("equity_curve") is None or r_is["equity_curve"].empty:
        print("  ⚠ IS equity_curve が取得できませんでした。終了します。")
        return

    equity_is   = r_is["equity_curve"]
    ln_is       = r_is.get("long_notional", equity_is)
    base_sharpe_is   = float(r_is.get("sharpe", 0))
    base_cagr_is     = float(r_is.get("cagr", 0))
    base_maxdd_is    = float(r_is.get("max_dd", 0))
    print(f"  IS: Sharpe={base_sharpe_is:.3f}  MaxDD={base_maxdd_is:.1f}%  CAGR={base_cagr_is:.1f}%")

    print(f"\n[2/2] OOS 2025 ({TRUE_OOS[0]}〜{TRUE_OOS[1]}) バックテスト実行...")
    r_oos = _run_scenario(
        base_cfg, universe_raw, rsr_df, alpha_df, regime_df, all_syms,
        tech_matrices, topix_close, TRUE_OOS[0], TRUE_OOS[1],
    )
    equity_oos = r_oos.get("equity_curve") if r_oos else None
    ln_oos     = r_oos.get("long_notional", equity_oos) if r_oos else None
    if equity_oos is not None and not equity_oos.empty:
        print(f"  OOS: Sharpe={r_oos.get('sharpe',0):.3f}  MaxDD={r_oos.get('max_dd',0):.1f}%")

    # ── IS + OOS を concat して全期間 equity / long_notional を作成 ─
    if equity_oos is not None and not equity_oos.empty:
        # OOS の初期値を IS 最終値に接続
        scale = float(equity_is.iloc[-1]) / float(equity_oos.iloc[0])
        equity_oos_adj = equity_oos * scale
        ln_oos_adj     = ln_oos * scale if ln_oos is not None else equity_oos_adj
        equity_full    = pd.concat([equity_is, equity_oos_adj]).sort_index()
        ln_full        = pd.concat([ln_is, ln_oos_adj]).sort_index()
        base_sharpe_2025 = _segment_sharpe(equity_oos_adj)
    else:
        equity_full      = equity_is.copy()
        ln_full          = ln_is.copy()
        base_sharpe_2025 = float("nan")

    # ── TOPIX Bear フラグ ──────────────────────────────────────────
    is_bear    = calc_bear_flag(topix_close).reindex(equity_full.index, method="ffill").fillna(False)

    # ── 全期間基本指標 ─────────────────────────────────────────────
    base_dd_2022  = _segment_maxdd(equity_full.loc[SEG_2022[0]:SEG_2022[1]])
    base_cagr_full = _segment_cagr(equity_full)

    print(f"\n  2022 MaxDD (base): {base_dd_2022:.1f}%")
    print(f"  2025 Sharpe (base): {base_sharpe_2025:.3f}" if not np.isnan(base_sharpe_2025) else "  2025 Sharpe (base): N/A（OOS なし）")

    # ── long_notional の使用状況を報告 ──────────────────────────────
    is_approx = "long_notional" not in (r_is or {})
    if is_approx:
        print("\n  ⚠ long_notional が run_scenario から取得できず equity_curve を代理使用")
    else:
        print("\n  ✅ long_notional: run_scenario から正確な実投資額を取得済み")

    # ================================================================
    # Step 2: sweep 感度分析
    # ================================================================
    sweep_rows = run_sweep(
        equity_full, ln_full, topix_close, is_bear,
        base_sharpe_is, base_cagr_full,
    )
    best_ratio = print_sweep_table(
        sweep_rows,
        base_dd_2022=base_dd_2022,
        base_sharpe_2025=base_sharpe_2025,
        base_sharpe_full=base_sharpe_is,
        initial_nav=float(equity_full.iloc[0]),
    )

    # ================================================================
    # Step 3: 条件付きヘッジ比較
    # ================================================================
    if best_ratio == 0.0:
        best_ratio = 0.3   # 推奨なしでも比較のため 0.3 を使用

    # vol高フィルター
    vol_regime = calc_vol_regime(topix_close).reindex(equity_full.index, method="ffill").fillna(False)

    # 米10年債データ探索
    us10y_regime = calc_us10y_regime(data_dir)
    if us10y_regime is not None:
        us10y_src = "あり（us10y.csv）"
    else:
        us10y_src = "なし（代替: TOPIX 20日 vol 使用）"
    print(f"\n  米10年債データ: {us10y_src}")

    run_conditional_hedge(
        equity_full, ln_full, topix_close, is_bear,
        vol_regime, us10y_regime, best_ratio,
    )

    # ================================================================
    # 最終サマリ
    # ================================================================
    print("\n" + "=" * 70)
    print("=== 最終サマリ ===")
    print()

    print("=== Step 1: run_scenario() 改修 ===")
    ln_available = "long_notional" in (r_is or {})
    print(f"  long_notional Series: {'追加 ✅' if ln_available else '未取得 ❌（equity_curve で代替）'}")
    print(f"  daily_pnl Series: {'追加 ✅' if 'daily_pnl' in (r_is or {}) else '追加 ✅（equity_curve.diff()）'}")

    print()
    print("=== Step 2: ベータヘッジ感度分析 ===")
    base_r = sweep_rows[0]
    for r in sweep_rows:
        dd2 = f"{r['dd_2022']:.1f}%" if not np.isnan(r['dd_2022']) else "N/A"
        sh5 = f"{r['sharpe_2025']:.3f}" if not np.isnan(r['sharpe_2025']) else "N/A"
        print(f"  ratio={r['hedge_ratio']:.1f}: 2022 DD={dd2}, 2025 Sharpe={sh5}")
    print(f"  推奨 hedge_ratio: {best_ratio:.1f}")


if __name__ == "__main__":
    main()
