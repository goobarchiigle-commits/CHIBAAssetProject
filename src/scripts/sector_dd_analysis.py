"""
scripts/sector_dd_analysis.py

セクター別 DD 分解レポート（2022 年 + 2025 年 OOS）

実行:
  python -m scripts.sector_dd_analysis
"""

from __future__ import annotations
import sys
import json
import os
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# プロジェクトルートを sys.path に追加
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.paths import CONFIGS_DIR, BACKTEST_DATASET_DIR, DEFAULT_DATA_VERSION, RESULTS_DIR
from src.backtest.composite_alpha_bt import (
    _load_rsr_universe,
    _download_topix,
    run_scenario,
    START, END,
)
from src.backtest.rsr import calc_universe_rsr, calc_composite_alpha_matrix
from src.backtest.universe_builder import download_universe
from src.strategy.universe import build_dyn_rsr42_active
from src.config_loader import load_strategy_config
from src.backtest.composite_alpha_bt import _calc_regime, _precompute_tech_matrices, _calc_breadth


# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
PERIODS = {
    "2022": {"warm_start": "2021-01-01", "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    "2025": {"warm_start": "2024-01-01", "oos_start": "2025-01-01", "oos_end": "2025-03-31"},
}
OUT_MD  = _ROOT / "docs" / "research" / "sector_dd_2022_2025.md"
OUT_JSON = _ROOT / "backtests" / "sector_dd_2022_2025.json"


# ------------------------------------------------------------------ #
# セクター別メトリクス計算
# ------------------------------------------------------------------ #
def compute_sector_metrics(
    trades: list[dict],
    trade_syms: dict[str, str],
    total_max_dd_pct: float,
) -> pd.DataFrame:
    """SELL トレードからセクター別集計を返す。"""
    rows = []
    for t in trades:
        sym = t["symbol"]
        sector = trade_syms.get(sym, "不明")
        hold_days = t.get("exit_idx", 0) - t.get("entry_idx", 0)
        rows.append({
            "sector": sector,
            "pnl": t["pnl"],
            "hold_days": hold_days,
            "win": 1 if t["pnl"] > 0 else 0,
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    grp = df.groupby("sector")

    total_pnl = df["pnl"].sum()

    records = []
    for sector, g in grp:
        sector_pnl   = g["pnl"].sum()
        loss_sum     = g[g["pnl"] < 0]["pnl"].sum()
        dd_contrib   = (loss_sum / (total_max_dd_pct * 1_000_000)) * 100 if total_max_dd_pct != 0 else 0.0
        records.append({
            "sector":         sector,
            "total_pnl":      round(sector_pnl),
            "pnl_pct":        round(sector_pnl / max(abs(total_pnl), 1) * 100, 1),
            "dd_contribution": round(abs(loss_sum / max(abs(total_max_dd_pct * 1_000_000), 1)) * 100, 1),
            "avg_hold_days":  round(g["hold_days"].mean(), 1),
            "win_rate":       round(g["win"].mean() * 100, 1),
            "trade_count":    len(g),
        })

    result = pd.DataFrame(records).sort_values("total_pnl")
    return result


# ------------------------------------------------------------------ #
# セクター日次リターン相関行列
# ------------------------------------------------------------------ #
def compute_sector_correlation(
    universe_raw: dict,
    trade_syms: dict[str, str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """各セクターの等ウェイト日次リターンから相関行列を計算する。"""
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)

    # セクター別銘柄群の Close を集計
    sector_map: dict[str, list] = {}
    for sym, sector in trade_syms.items():
        if sym not in universe_raw:
            continue
        sector_map.setdefault(sector, []).append(sym)

    sector_returns: dict[str, pd.Series] = {}
    for sector, syms in sector_map.items():
        closes = [
            universe_raw[s]["df"]["Close"]
            for s in syms
            if s in universe_raw
        ]
        if not closes:
            continue
        combined = pd.concat(closes, axis=1).dropna(how="all")
        combined = combined.loc[
            (combined.index >= ts_start) & (combined.index <= ts_end)
        ]
        if combined.empty:
            continue
        ret = combined.pct_change().mean(axis=1)
        sector_returns[sector] = ret

    if len(sector_returns) < 2:
        return pd.DataFrame()

    ret_df = pd.DataFrame(sector_returns).dropna(how="all")
    return ret_df.corr().round(3)


# ------------------------------------------------------------------ #
# 1 期間の分析実行
# ------------------------------------------------------------------ #
def run_period(
    label: str,
    warm_start: str,
    oos_start: str,
    oos_end: str,
    cfg,
    trade_syms: dict[str, str],
    rsr_syms: dict[str, str],
) -> dict:
    print(f"\n{'='*60}")
    print(f"  期間: {label}  ({oos_start} 〜 {oos_end})")
    print(f"  データ取得期間: {warm_start} 〜 {oos_end}")
    print(f"{'='*60}")

    # 価格データ
    print("[1] 価格データ取得中...")
    all_syms = {**rsr_syms, **trade_syms}
    universe_raw = download_universe(
        all_syms,
        start=warm_start,
        end=oos_end,
        min_days=50,
        verbose=False,
    )
    topix_close = _download_topix(warm_start, oos_end, verbose=False)
    print(f"  取得完了: {len(universe_raw)}銘柄, TOPIX={len(topix_close)}日")

    # RSR 計算
    print("[2] RSR計算中...")
    rsr42_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in rsr_syms if sym in universe_raw
    }
    rsr_df = calc_universe_rsr(rsr42_prices)

    # 動的ユニバース
    print("[3] 動的ユニバース構築中（dyn_rsr42_bear_rs0）...")
    sym_active_df = build_dyn_rsr42_active(
        universe_raw=universe_raw,
        topix_close=topix_close,
        rsr_df=rsr_df,
        all_syms=list(trade_syms.keys()),
        start=warm_start,
        end=oos_end,
    )

    # Composite Alpha
    print("[4] Composite Alpha計算中...")
    trade_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in trade_syms if sym in universe_raw
    }
    alpha_df = calc_composite_alpha_matrix(trade_prices, window=90)
    alpha_df = alpha_df.shift(1)

    # レジーム
    regime_df = _calc_regime(topix_close)
    tech_matrices = _precompute_tech_matrices(universe_raw, list(trade_syms.keys()))
    breadth_series = _calc_breadth(universe_raw)

    # run_scenario（OOS 期間のみ）
    print(f"[5] run_scenario 実行中（{oos_start}〜{oos_end}）...")
    res = run_scenario(
        scenario       = "BASELINE",
        universe_raw   = universe_raw,
        rsr_df         = rsr_df,
        alpha_df       = alpha_df,
        regime_df      = regime_df,
        trade_syms     = trade_syms,
        rsr_syms       = rsr_syms,
        cfg            = cfg,
        start          = oos_start,
        end            = oos_end,
        capital        = float(cfg.portfolio.capital),
        verbose        = False,
        tech_matrices  = tech_matrices,
        breadth_series = breadth_series,
        topix_close    = topix_close,
        sym_active_df  = sym_active_df,
    )

    trades   = res.get("_trades", [])
    max_dd   = res.get("max_dd", 0.0)   # % (e.g. -7.8)
    sharpe   = res.get("sharpe", 0.0)
    cagr     = res.get("cagr", 0.0)

    print(f"  CAGR={cagr:+.1f}%  Sharpe={sharpe:.3f}  MaxDD={max_dd:.1f}%  取引={len(trades)}")

    # セクター別集計
    sector_df = compute_sector_metrics(trades, trade_syms, max_dd / 100.0)

    # セクター相関
    corr_df = compute_sector_correlation(universe_raw, trade_syms, oos_start, oos_end)

    return {
        "label":      label,
        "oos_start":  oos_start,
        "oos_end":    oos_end,
        "cagr":       cagr,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "n_trades":   len(trades),
        "sector_df":  sector_df,
        "corr_df":    corr_df,
    }


# ------------------------------------------------------------------ #
# Markdown 出力
# ------------------------------------------------------------------ #
def render_markdown(results: dict[str, dict]) -> str:
    lines = [
        "# セクター別 DD 分解レポート",
        f"生成日: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]
    for label, r in results.items():
        lines += [
            f"## {label} 年 OOS（{r['oos_start']} 〜 {r['oos_end']}）",
            "",
            f"- CAGR: {r['cagr']:+.1f}%  Sharpe: {r['sharpe']:.3f}  MaxDD: {r['max_dd']:.1f}%  取引数: {r['n_trades']}",
            "",
        ]

        sdf = r["sector_df"]
        if sdf.empty:
            lines.append("（取引なし）\n")
        else:
            # セクター別テーブル
            lines.append("### セクター別集計（PnL 昇順）")
            lines.append("")
            lines.append("| セクター | 累積PnL(円) | PnL寄与% | DD寄与% | 平均保有日 | 勝率% | 取引数 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for _, row in sdf.iterrows():
                lines.append(
                    f"| {row['sector']} | {row['total_pnl']:,} | {row['pnl_pct']:+.1f}% "
                    f"| {row['dd_contribution']:.1f}% | {row['avg_hold_days']:.1f}日 "
                    f"| {row['win_rate']:.1f}% | {int(row['trade_count'])} |"
                )
            lines.append("")

        cdf = r["corr_df"]
        if not cdf.empty:
            lines.append("### セクター間日次リターン相関行列")
            lines.append("")
            sectors = list(cdf.columns)
            lines.append("| セクター | " + " | ".join(sectors) + " |")
            lines.append("|---|" + "---|" * len(sectors))
            for s in sectors:
                row_vals = " | ".join(f"{cdf.loc[s, c]:.2f}" for c in sectors)
                lines.append(f"| {s} | {row_vals} |")
            lines.append("")

            # 特徴的な相関（|r| > 0.6 のペア）
            pairs = []
            for i, s1 in enumerate(sectors):
                for s2 in sectors[i+1:]:
                    v = cdf.loc[s1, s2]
                    if abs(v) > 0.60:
                        pairs.append((s1, s2, v))
            if pairs:
                lines.append("**高相関ペア（|r| > 0.60）**")
                lines.append("")
                for s1, s2, v in sorted(pairs, key=lambda x: -abs(x[2])):
                    lines.append(f"- {s1} × {s2}: r = {v:.2f}")
                lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main() -> int:
    cfg        = load_strategy_config()
    rsr_syms   = _load_rsr_universe(verbose=False)
    trade_syms = rsr_syms

    results: dict[str, dict] = {}
    for label, p in PERIODS.items():
        r = run_period(
            label       = label,
            warm_start  = p["warm_start"],
            oos_start   = p["oos_start"],
            oos_end     = p["oos_end"],
            cfg         = cfg,
            trade_syms  = trade_syms,
            rsr_syms    = rsr_syms,
        )
        results[label] = r

    # ── コンソール表形式表示 ──────────────────────────────────────────
    for label, r in results.items():
        print(f"\n{'='*60}")
        print(f"  {label} 年 OOS セクター別集計")
        print(f"{'='*60}")
        sdf = r["sector_df"]
        if sdf.empty:
            print("  （取引なし）")
            continue
        print(sdf.to_string(index=False))

        cdf = r["corr_df"]
        if not cdf.empty:
            print(f"\n  {label} 年 セクター相関行列（抜粋: |r|>0.6）")
            sectors = list(cdf.columns)
            for i, s1 in enumerate(sectors):
                for s2 in sectors[i+1:]:
                    v = cdf.loc[s1, s2]
                    if abs(v) > 0.60:
                        print(f"    {s1} × {s2}: {v:+.2f}")

    # ── 保存 ─────────────────────────────────────────────────────────
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(results)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"\n→ Markdown 保存: {OUT_MD}")

    # JSON 保存（sector_df と corr_df を dict 化）
    json_out: dict = {}
    for label, r in results.items():
        sdf = r["sector_df"]
        cdf = r["corr_df"]
        json_out[label] = {
            "cagr":      r["cagr"],
            "sharpe":    r["sharpe"],
            "max_dd":    r["max_dd"],
            "n_trades":  r["n_trades"],
            "sector_pnl": sdf.set_index("sector")["total_pnl"].to_dict() if not sdf.empty else {},
            "sector_metrics": sdf.to_dict(orient="records") if not sdf.empty else [],
            "sector_correlation": cdf.to_dict() if not cdf.empty else {},
        }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ JSON 保存: {OUT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
