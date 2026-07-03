"""
scripts/risk_regime_report.py

ショック年イベントのDD分析を自動出力するレポートスクリプト。
毎回のWF検証に加えて、このレポートを必ず実行すること。

対象イベント（固定）:
- 2020-02〜2020-04: コロナショック
- 2022-01〜2022-06: 金利ショック
- 2024-07〜2024-08: 日銀利上げ
- 2025-04〜2025-05: 関税ショック（2025 OOS）

実行:
  cd C:/ai-trading
  python src/scripts/risk_regime_report.py

使用方法（外部から呼び出す場合）:
  from src.scripts.risk_regime_report import run_risk_regime_report
  run_risk_regime_report(
      scenario_name = "pattern_A",
      returns       = returns_series,    # 日次リターン（pct）
      topix_ret     = topix_ret_series,  # TOPIX 日次リターン
      # オプション
      positions_df  = None,              # 日次ポジション DataFrame（未実装時は None）
      skip_log_df   = None,              # skip_stats DataFrame（未実装時は None）
  )

returns の取り出し方（run_scenario() 戻り値から）:
  result = run_scenario(...)
  equity = result["equity_curve"]        # pd.Series（日次資産額）
  returns = equity.pct_change().dropna() # 日次リターン Series
  topix_ret = topix_close.pct_change().reindex(returns.index).dropna()
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# 固定ショックイベント定義
# ------------------------------------------------------------------ #
SHOCK_EVENTS: dict[str, tuple[str, str]] = {
    "2020_covid":  ("2020-02-01", "2020-04-30"),
    "2022_rates":  ("2022-01-01", "2022-06-30"),
    "2024_boj":    ("2024-07-01", "2024-08-31"),
    "2025_tariff": ("2025-04-01", "2025-05-31"),
}

EVENT_LABELS = {
    "2020_covid":  "2020コロナ",
    "2022_rates":  "2022金利",
    "2024_boj":    "2024日銀",
    "2025_tariff": "2025関税",
}


# ------------------------------------------------------------------ #
# 単一ショックイベント分析
# ------------------------------------------------------------------ #
def analyze_shock_event(
    returns: pd.Series,
    topix_ret: pd.Series,
    event_name: str,
    start: str,
    end: str,
    positions_df: "pd.DataFrame | None" = None,
    skip_log_df:  "pd.DataFrame | None" = None,
) -> dict:
    """
    各ショックイベント期間中の指標を計算。

    引数:
        returns     : 日次リターン Series（index = DatetimeIndex）
        topix_ret   : TOPIX 日次リターン Series
        event_name  : イベント識別子
        start / end : 期間文字列 "YYYY-MM-DD"
        positions_df: 日次ポジション DataFrame（列=銘柄、値=評価額）。None 可
        skip_log_df : skip 統計 DataFrame。None 可

    戻り値:
        peak_to_trough_dd : ピークからトラフまでの DD (%)
        recovery_days     : DDからの回復日数（次のピーク更新まで）
        exposure_avg      : イベント期間の平均グロスエクスポージャー（positions_df 使用時）
        skip_ratio        : イベント期間のskip率（skip_log_df 使用時）
        topix_dd          : 同期間の TOPIX DD（対比用）
    """
    mask = (returns.index >= start) & (returns.index <= end)
    r    = returns[mask]

    result = {
        "event":             event_name,
        "label":             EVENT_LABELS.get(event_name, event_name),
        "start":             start,
        "end":               end,
        "peak_to_trough_dd": None,
        "recovery_days":     None,
        "exposure_avg":      None,
        "skip_ratio":        None,
        "topix_dd":          None,
    }

    if len(r) == 0:
        result["error"] = "no data in period"
        return result

    cum   = (1 + r).cumprod()
    peak  = cum.cummax()
    dd    = (cum / peak - 1)
    peak_to_trough = float(dd.min())
    result["peak_to_trough_dd"] = round(peak_to_trough * 100, 2)

    # 回復日数: トラフ以降に cum がトラフ時点の値を更新するまでの日数
    trough_idx = dd.idxmin()
    post_trough = cum[trough_idx:]
    if len(post_trough) > 1:
        trough_val = float(post_trough.iloc[0])
        recovered  = post_trough[post_trough >= peak[trough_idx]]
        if len(recovered) > 0:
            recovery_date = recovered.index[0]
            result["recovery_days"] = (recovery_date - trough_idx).days
        else:
            result["recovery_days"] = None  # 未回復

    # TOPIX DD（同期間）
    mask_tx = (topix_ret.index >= start) & (topix_ret.index <= end)
    tx = topix_ret[mask_tx]
    if len(tx) > 0:
        cum_tx   = (1 + tx).cumprod()
        peak_tx  = cum_tx.cummax()
        dd_tx    = (cum_tx / peak_tx - 1)
        result["topix_dd"] = round(float(dd_tx.min()) * 100, 2)

    # exposure_avg（positions_df が利用可能な場合）
    if positions_df is not None:
        mask_p = (positions_df.index >= start) & (positions_df.index <= end)
        pf = positions_df[mask_p]
        if not pf.empty:
            result["exposure_avg"] = round(float(pf.sum(axis=1).mean()), 0)

    # skip_ratio（skip_log_df が利用可能な場合）
    if skip_log_df is not None:
        mask_s = (skip_log_df.index >= start) & (skip_log_df.index <= end)
        sf = skip_log_df[mask_s]
        if not sf.empty and "total_skip" in sf.columns and "total_attempt" in sf.columns:
            total_skip    = sf["total_skip"].sum()
            total_attempt = sf["total_attempt"].sum()
            if total_attempt > 0:
                result["skip_ratio"] = round(total_skip / total_attempt * 100, 1)

    return result


# ------------------------------------------------------------------ #
# メインレポート生成
# ------------------------------------------------------------------ #
def run_risk_regime_report(
    scenario_name: str,
    returns: pd.Series,
    topix_ret: pd.Series,
    positions_df: "pd.DataFrame | None" = None,
    skip_log_df:  "pd.DataFrame | None" = None,
) -> list[dict]:
    """
    メイン実行関数。全ショックイベントを分析してテーブル出力する。

    引数:
        scenario_name : 識別用ラベル（例: "pattern_A"）
        returns       : 日次リターン Series（equity_curve.pct_change()）
        topix_ret     : TOPIX 日次リターン Series
        positions_df  : 日次ポジション評価額 DataFrame（None 可）
        skip_log_df   : skip 統計 DataFrame（None 可）
    """
    results = []
    for event_name, (start, end) in SHOCK_EVENTS.items():
        r = analyze_shock_event(
            returns, topix_ret, event_name, start, end,
            positions_df=positions_df,
            skip_log_df=skip_log_df,
        )
        results.append(r)

    print(f"\n=== Risk Regime Report: {scenario_name} ===")
    print(f"{'イベント':<16} {'期間':>20} {'P→T DD':>10} {'TOPIX DD':>10}"
          f" {'回復日数':>10} {'avg exp':>10} {'skip率':>8}")
    print("-" * 88)

    for r in results:
        if r.get("error"):
            print(f"  {r['label']:<14} データなし")
            continue

        dd_str   = f"{r['peak_to_trough_dd']:.1f}%" if r["peak_to_trough_dd"] is not None else "N/A"
        tx_str   = f"{r['topix_dd']:.1f}%"           if r["topix_dd"]          is not None else "N/A"
        rec_str  = f"{r['recovery_days']}日"          if r["recovery_days"]     is not None else "未回復"
        exp_str  = f"¥{r['exposure_avg']:,.0f}"       if r["exposure_avg"]      is not None else "N/A"
        skip_str = f"{r['skip_ratio']:.1f}%"          if r["skip_ratio"]        is not None else "N/A"
        period   = f"{r['start']}〜{r['end']}"

        print(f"  {r['label']:<14} {period:>20} {dd_str:>10} {tx_str:>10}"
              f" {rec_str:>10} {exp_str:>10} {skip_str:>8}")

    # サマリー
    valid_dds = [r["peak_to_trough_dd"] for r in results if r.get("peak_to_trough_dd") is not None]
    if valid_dds:
        worst = min(valid_dds)
        worst_ev = next(r for r in results if r.get("peak_to_trough_dd") == worst)
        print(f"\n  最悪ショック: {worst_ev['label']} ({worst:.1f}%)")
        print(f"  平均ショックDD: {np.mean(valid_dds):.1f}%")

    return results


# ------------------------------------------------------------------ #
# スタンドアロン実行
# ------------------------------------------------------------------ #
def main():
    """
    composite_alpha_bt.py のパターンA（固定 cap）で Full IS を実行し、
    ショックイベントレポートを出力する。
    """
    from src.backtest.wf_dynamic_universe import (
        load_data, _build_active,
        BULL_ACTIVE_N, BEAR_ACTIVE_N, IS_FULL,
    )
    import src.backtest.composite_alpha_bt as _bt

    base_cfg = load_strategy_config()

    print("=" * 70)
    print("  Risk Regime Report（ショックイベントDD分析）")
    print("=" * 70)

    (universe_raw, rsr_df, rsr_df_full, alpha_df, regime_df, rsr_syms, all_syms,
     tech_matrices, topix_close, base_cfg) = load_data(base_cfg)

    # Full IS（2018-2024）で通常パターン（パターンA相当）を実行
    print(f"\n  Full IS ({IS_FULL[0]}〜{IS_FULL[1]}) でバックテスト実行中...")
    active_full = _build_active(
        universe_raw, topix_close, all_syms, rsr_df,
        BULL_ACTIVE_N, BEAR_ACTIVE_N, "2017-01-01", IS_FULL[1], None
    )
    result = _bt.run_scenario(
        scenario             = "BASELINE",
        universe_raw         = universe_raw,
        rsr_df               = rsr_df,
        alpha_df             = alpha_df,
        regime_df            = regime_df,
        trade_syms           = all_syms,
        rsr_syms             = all_syms,
        cfg                  = base_cfg,
        start                = IS_FULL[0],
        end                  = IS_FULL[1],
        capital              = float(base_cfg.portfolio.capital),
        verbose              = False,
        tech_matrices        = tech_matrices,
        topix_close          = topix_close,
        min_hold             = int(base_cfg.risk.min_hold_days),
        market_shock_mode    = "composite",
        enable_volume_filter = True,
        sym_active_df        = active_full,
    )

    equity = result.get("equity_curve")
    if equity is None or equity.empty:
        print("  ⚠ equity_curve が取得できませんでした。")
        return

    returns   = equity.pct_change().dropna()
    topix_ret = topix_close.pct_change().reindex(returns.index).fillna(0.0)

    print(f"  Full IS: Sharpe={result.get('sharpe', 0):.3f}"
          f"  MaxDD={result.get('max_dd', 0):.1f}%")

    # OOS 2025 も実行
    print(f"\n  OOS 2025 (2025-01-01〜2025-12-31) でバックテスト実行中...")
    try:
        active_2025 = _build_active(
            universe_raw, topix_close, all_syms, rsr_df,
            BULL_ACTIVE_N, BEAR_ACTIVE_N, "2017-01-01", "2025-12-31", None
        )
        result_oos = _bt.run_scenario(
            scenario             = "BASELINE",
            universe_raw         = universe_raw,
            rsr_df               = rsr_df,
            alpha_df             = alpha_df,
            regime_df            = regime_df,
            trade_syms           = all_syms,
            rsr_syms             = all_syms,
            cfg                  = base_cfg,
            start                = "2025-01-01",
            end                  = "2025-12-31",
            capital              = float(base_cfg.portfolio.capital),
            verbose              = False,
            tech_matrices        = tech_matrices,
            topix_close          = topix_close,
            min_hold             = int(base_cfg.risk.min_hold_days),
            market_shock_mode    = "composite",
            enable_volume_filter = True,
            sym_active_df        = active_2025,
        )
        equity_oos = result_oos.get("equity_curve")
        if equity_oos is not None and not equity_oos.empty:
            returns_oos   = equity_oos.pct_change().dropna()
            topix_ret_oos = topix_close.pct_change().reindex(returns_oos.index).fillna(0.0)
            # Full IS + OOS 2025 を結合
            combined_ret    = pd.concat([returns, returns_oos]).sort_index()
            combined_topix  = pd.concat([topix_ret, topix_ret_oos]).sort_index()
            combined_ret    = combined_ret[~combined_ret.index.duplicated()]
            combined_topix  = combined_topix[~combined_topix.index.duplicated()]
        else:
            combined_ret   = returns
            combined_topix = topix_ret
    except Exception as e:
        print(f"  ⚠ OOS 2025 実行エラー: {e}")
        combined_ret   = returns
        combined_topix = topix_ret

    # レポート出力
    run_risk_regime_report(
        scenario_name = "baseline_pattern_A",
        returns       = combined_ret,
        topix_ret     = combined_topix,
    )

    print("\n使用方法（外部から呼び出す場合）:")
    print("  from src.scripts.risk_regime_report import run_risk_regime_report")
    print("  returns = result['equity_curve'].pct_change().dropna()")
    print("  topix_ret = topix_close.pct_change().reindex(returns.index).fillna(0)")
    print("  run_risk_regime_report('pattern_D', returns, topix_ret)")


if __name__ == "__main__":
    main()
