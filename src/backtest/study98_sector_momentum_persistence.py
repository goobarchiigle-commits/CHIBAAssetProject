"""
src/backtest/study98_sector_momentum_persistence.py
Study98 — TOPIX17セクターモメンタム持続性（factor-level存在確認）

正典: ユーザー指示「Study98」（2026-07-15）。FUJIKO 2.0ロードマップ（Study95と同一H0/H1系列の
一部・reports/fujiko_r2_research_roadmap.md）のStudy88（セクターモメンタム持続性・H1）の
より厳密な定式化を兼ねる。

目的:
  TOPIX17各セクターについて
    P(R_sector(t+1~3m) > R_TOPIX(t+1~3m) | R_sector(t) > R_TOPIX(t))
  を算出する。「牽引セクター（TOPIXを上回っているセクター）は今後もTOPIXを上回り続けるか」の
  factor-level検証。銘柄レベル（セクター内牽引銘柄）の検証はStudy89（別途起案）に委ねる。

データ:
  database/market/index/prices/{0000,0040-0056}.parquet
    公式TOPIX-17業種別指数（J-Quants /v2/indices/bars/daily・2026-07-15新規取得）。
    DIY等ウェイトproxyではなく公式指数を使用（Study77で記録された
    "TOPIX17セクターETF価格データは既存パイプラインに存在せず"というギャップを解消）。
  database/market/master/companies.parquet: Sector17Code→名称の参照のみ（指数取得には不使用）。

禁止（タスク仕様）:
  戦略BT禁止。本スクリプトはfactor-level統計のみ（発注・ポジションサイジング・
  BTエンジン一切不使用）。

設計（トレーリング条件とフォワード結果を同一horizonで対称に測定）:
  horizon X ∈ {1M(21営業日), 3M(63営業日)}
  条件: excess_trailing_X(t) = [sector(t)/sector(t-X) - 1] - [topix(t)/topix(t-X) - 1] > 0
  結果: excess_forward_X(t)  = [sector(t+X)/sector(t) - 1] - [topix(t+X)/topix(t) - 1] > 0
  月次リバランス日（各月最初の営業日）でサンプリング。

追加分析:
  1. 上位3限定: 条件を「その月のexcess_trailingが17セクター中上位3位以内」に変更
  2. TOPIX超過全セクター: 条件「excess_trailing > 0」（プライマリと同一・全量ベースライン）
  3. regime別: TOPIX > MA200（既存canon Market Regime定義）でBull/Bear分割

出力: 遷移確率・excess return・t統計量・confusion matrix・サブ期間(2016-2020/2021-2025)

出力ファイル:
  backtests/study98_sector_momentum_persistence.json
  reports/study98_sector_momentum_persistence.md
  reports/study98_transition_chart.png
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date, datetime, timezone, timedelta

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy import stats

from src.database.index_prices import SECTOR17_NAMES, load_all_topix17_series, load_index_series
from src.paths import DATABASE_MARKET_DIR, REPORTS_DIR, RESULTS_DIR

rcParams["font.family"] = "MS Gothic"

_JST = timezone(timedelta(hours=9))

HORIZONS: dict[str, int] = {"1M": 21, "3M": 63}
REGIME_MA_PERIOD = 200
SUBPERIODS: dict[str, tuple[str, str]] = {
    "2016-2020": ("2016-01-01", "2020-12-31"),
    "2021-2025": ("2021-01-01", "2025-12-31"),
}
TOP_N = 3

OUT_JSON  = RESULTS_DIR / "study98_sector_momentum_persistence.json"
OUT_MD    = REPORTS_DIR / "study98_sector_momentum_persistence.md"
OUT_CHART = REPORTS_DIR / "study98_transition_chart.png"


# ======================================================================
# 1. データ読込・パネル構築
# ======================================================================
def load_panel() -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """
    Returns:
        topix_close: Series（全期間の日次Close）
        sector_close: {sector17_code: Series}
    """
    topix = load_index_series("0000")["Close"].dropna().sort_index()
    sector_series = load_all_topix17_series()
    sector_close = {k: v["Close"].dropna().sort_index() for k, v in sector_series.items()}
    return topix, sector_close


def monthly_formation_dates(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """各月最初の営業日（Study95と同一の月次リバランス規約）。"""
    s = pd.Series(calendar, index=calendar)
    ym = calendar.to_period("M")
    firsts = s.groupby(ym).min()
    return sorted(firsts.tolist())


def calc_regime(topix: pd.Series) -> pd.Series:
    """TOPIX > MA200（既存canon Market Regime定義。Study76/95と同一）。"""
    sma = topix.rolling(REGIME_MA_PERIOD, min_periods=REGIME_MA_PERIOD).mean()
    return topix >= sma


# ======================================================================
# 2. パネル構築（rebalance_date x sector17_code の観測行列）
# ======================================================================
def build_observation_panel(
    topix: pd.Series, sector_close: dict[int, pd.Series],
) -> pd.DataFrame:
    """
    各月次formation date・各セクターについて、horizon={1M,3M}ごとの
    excess_trailing / excess_forward を算出したロング形式パネルを返す。

    列: rebalance_date, sector17_code, horizon, excess_trailing, excess_forward,
        condition (excess_trailing>0), outcome (excess_forward>0),
        trailing_rank（その月のexcess_trailing降順順位・1=最高）, regime_bull
    """
    calendar = topix.index
    formation_dates = monthly_formation_dates(calendar)
    calendar_pos = {d: i for i, d in enumerate(calendar)}
    regime_bull = calc_regime(topix)
    regime_defined_dates = set(regime_bull.dropna().index)

    # セクター系列を共通カレンダーへreindex（前方補完なし・存在しない日はNaN）
    sector_aligned: dict[int, pd.Series] = {
        k: v.reindex(calendar) for k, v in sector_close.items()
    }

    rows: list[dict] = []
    for rb_date in formation_dates:
        if rb_date not in calendar_pos:
            continue
        base_pos = calendar_pos[rb_date]

        # その月の全セクターexcess_trailingを先に計算（順位付け用）
        trailing_by_sector: dict[str, dict[int, float]] = {"1M": {}, "3M": {}}
        for label, h in HORIZONS.items():
            if base_pos - h < 0:
                continue
            topix_now = topix.iloc[base_pos]
            topix_prev = topix.iloc[base_pos - h]
            if topix_prev <= 0 or pd.isna(topix_now) or pd.isna(topix_prev):
                continue
            topix_trailing_ret = topix_now / topix_prev - 1.0
            for sc, ser in sector_aligned.items():
                s_now = ser.iloc[base_pos]
                s_prev = ser.iloc[base_pos - h]
                if pd.isna(s_now) or pd.isna(s_prev) or s_prev <= 0:
                    continue
                sector_trailing_ret = s_now / s_prev - 1.0
                trailing_by_sector[label][sc] = sector_trailing_ret - topix_trailing_ret

        is_bull = bool(regime_bull.get(rb_date, False)) if rb_date in regime_defined_dates else None

        for label, h in HORIZONS.items():
            n = len(calendar)
            if base_pos - h < 0 or base_pos + h >= n:
                continue
            _trailing_map = trailing_by_sector[label]
            if not _trailing_map:
                continue
            ranked = sorted(_trailing_map.items(), key=lambda kv: kv[1], reverse=True)
            rank_of = {sc: i + 1 for i, (sc, _) in enumerate(ranked)}

            topix_now = topix.iloc[base_pos]
            topix_fwd = topix.iloc[base_pos + h]
            if topix_now <= 0 or pd.isna(topix_now) or pd.isna(topix_fwd):
                continue
            topix_forward_ret = topix_fwd / topix_now - 1.0

            for sc, excess_trailing in _trailing_map.items():
                ser = sector_aligned[sc]
                s_now = ser.iloc[base_pos]
                s_fwd = ser.iloc[base_pos + h]
                if pd.isna(s_now) or pd.isna(s_fwd) or s_now <= 0:
                    continue
                sector_forward_ret = s_fwd / s_now - 1.0
                excess_forward = sector_forward_ret - topix_forward_ret

                rows.append({
                    "rebalance_date": rb_date.strftime("%Y-%m-%d"),
                    "sector17_code": sc,
                    "sector_name": SECTOR17_NAMES.get(sc, "?"),
                    "horizon": label,
                    "excess_trailing": excess_trailing,
                    "excess_forward": excess_forward,
                    "condition": bool(excess_trailing > 0),
                    "outcome": bool(excess_forward > 0),
                    "trailing_rank": rank_of[sc],
                    "top3": bool(rank_of[sc] <= TOP_N),
                    "regime_bull": is_bull,
                })

    return pd.DataFrame(rows)


# ======================================================================
# 3. 統計量
# ======================================================================
def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def _two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_val)


def compute_transition_stats(df: pd.DataFrame, condition_col: str = "condition") -> dict:
    """
    P(outcome=True | condition=True) vs P(outcome=True | condition=False)。
    confusion matrix・excess return比較・t/z統計量を返す。
    """
    sub = df.dropna(subset=[condition_col, "outcome", "excess_forward"])
    cond_true  = sub[sub[condition_col] == True]   # noqa: E712
    cond_false = sub[sub[condition_col] == False]  # noqa: E712

    n1, x1 = len(cond_true), int(cond_true["outcome"].sum())
    n2, x2 = len(cond_false), int(cond_false["outcome"].sum())
    p_cond_true  = x1 / n1 if n1 > 0 else float("nan")
    p_cond_false = x2 / n2 if n2 > 0 else float("nan")
    z, z_p = _two_proportion_z(x1, n1, x2, n2)

    er_t, er_p = _welch_t(
        cond_true["excess_forward"].to_numpy(dtype=float),
        cond_false["excess_forward"].to_numpy(dtype=float),
    )

    cm = {
        "condition_true_outcome_true":   int(((sub[condition_col] == True) & (sub["outcome"] == True)).sum()),   # noqa: E712
        "condition_true_outcome_false":  int(((sub[condition_col] == True) & (sub["outcome"] == False)).sum()),  # noqa: E712
        "condition_false_outcome_true":  int(((sub[condition_col] == False) & (sub["outcome"] == True)).sum()),  # noqa: E712
        "condition_false_outcome_false": int(((sub[condition_col] == False) & (sub["outcome"] == False)).sum()), # noqa: E712
    }

    return {
        "n_condition_true": n1, "n_condition_false": n2,
        "transition_prob_given_condition_true":  p_cond_true,
        "transition_prob_given_condition_false": p_cond_false,
        "unconditional_prob": float(sub["outcome"].mean()) if len(sub) else float("nan"),
        "z_stat_proportion_diff": z, "z_pvalue": z_p,
        "mean_excess_forward_given_condition_true":  float(cond_true["excess_forward"].mean()) if n1 else float("nan"),
        "mean_excess_forward_given_condition_false": float(cond_false["excess_forward"].mean()) if n2 else float("nan"),
        "t_stat_excess_forward_diff": er_t, "t_pvalue": er_p,
        "confusion_matrix": cm,
        "n_total": len(sub),
    }


def compute_per_sector_stats(df: pd.DataFrame) -> dict:
    out = {}
    for sc in sorted(df["sector17_code"].unique()):
        sub = df[df["sector17_code"] == sc]
        for label in HORIZONS:
            h_sub = sub[sub["horizon"] == label]
            if len(h_sub) < 10:
                continue
            stats_h = compute_transition_stats(h_sub)
            out.setdefault(SECTOR17_NAMES.get(sc, str(sc)), {})[label] = stats_h
    return out


# ======================================================================
# 4. メイン
# ======================================================================
def main() -> int:
    print("[1/5] TOPIX-17指数データ読込...")
    topix, sector_close = load_panel()
    print(f"  TOPIX: {len(topix)}日  セクター数: {len(sector_close)}")

    print("[2/5] 観測パネル構築（月次formation・horizon={1M,3M}）...")
    panel = build_observation_panel(topix, sector_close)
    print(f"  panel rows={len(panel):,}  期間={panel['rebalance_date'].min()}〜{panel['rebalance_date'].max()}")

    print("[3/5] 集計（プライマリ／上位3限定／regime別／サブ期間）...")
    results = aggregate(panel)

    print("[4/5] 出力生成...")
    write_outputs(results, panel)

    print("[5/5] 完了")
    print(f"  JSON: {OUT_JSON}")
    print(f"  Markdown: {OUT_MD}")
    print(f"  Chart: {OUT_CHART}")
    return 0


def aggregate(panel: pd.DataFrame) -> dict:
    results: dict = {"n_panel_rows": len(panel)}

    # 2. TOPIX超過全セクター（プライマリ・ベースライン）
    for label in HORIZONS:
        h_panel = panel[panel["horizon"] == label]
        results.setdefault("primary_all_qualifying", {})[label] = compute_transition_stats(h_panel)

    # 1. 上位3限定
    for label in HORIZONS:
        h_panel = panel[panel["horizon"] == label]
        results.setdefault("top3_only", {})[label] = compute_transition_stats(h_panel, condition_col="top3")

    # per-sector breakdown
    results["per_sector"] = compute_per_sector_stats(panel)

    # 3. regime別
    for regime_key, regime_val in (("bull", True), ("bear", False)):
        r_panel = panel[panel["regime_bull"] == regime_val]
        for label in HORIZONS:
            h_panel = r_panel[r_panel["horizon"] == label]
            results.setdefault("regime", {}).setdefault(regime_key, {})[label] = compute_transition_stats(h_panel)

    # subperiod
    for sp_key, (sp_start, sp_end) in SUBPERIODS.items():
        sp_panel = panel[(panel["rebalance_date"] >= sp_start) & (panel["rebalance_date"] <= sp_end)]
        for label in HORIZONS:
            h_panel = sp_panel[sp_panel["horizon"] == label]
            results.setdefault("subperiod", {}).setdefault(sp_key, {})[label] = compute_transition_stats(h_panel)

    return results


# ======================================================================
# 5. 出力
# ======================================================================
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return str(o)


def write_outputs(results: dict, panel: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    full = {
        "study": "Study98",
        "title": "TOPIX17 Sector Momentum Persistence (Factor-Level)",
        "generated_at": datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_source": "J-Quants official TOPIX-17 sub-indices (database/market/index/prices/)",
        "prohibited_confirmed": {"bt_engine_used": False, "strategy_backtest": False},
        "params": {"horizons_trading_days": HORIZONS, "top_n": TOP_N, "regime_ma_period": REGIME_MA_PERIOD},
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(full, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    _write_chart(results)
    _write_markdown(full, panel)


def _write_chart(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label in zip(axes, HORIZONS):
        primary = results["primary_all_qualifying"][label]
        top3 = results["top3_only"][label]
        labels = ["全量\n(超過セクター)", "上位3限定"]
        p_true = [primary["transition_prob_given_condition_true"], top3["transition_prob_given_condition_true"]]
        p_base = [primary["unconditional_prob"], primary["unconditional_prob"]]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, [p * 100 for p in p_true], width=0.4, label="条件付き遷移確率", color="#2980b9")
        ax.bar(x + 0.2, [p * 100 for p in p_base], width=0.4, label="無条件base rate", color="#95a5a6")
        ax.axhline(50, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("P(forward outperform TOPIX) %")
        ax.set_title(f"horizon={label}")
        ax.legend(fontsize=8)
    fig.suptitle("Study98 — セクターモメンタム持続 遷移確率")
    fig.tight_layout()
    fig.savefig(OUT_CHART, dpi=140)
    plt.close(fig)


def _fmt_pct(v, digits=1):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{digits}f}%"


def _fmt(v, digits=3):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def _stats_row(label: str, s: dict) -> str:
    return (
        f"| {label} | {s['n_condition_true']} | {_fmt_pct(s['transition_prob_given_condition_true'])} "
        f"| {_fmt_pct(s['unconditional_prob'])} | {_fmt(s['z_stat_proportion_diff'])} "
        f"| {_fmt_pct(s['mean_excess_forward_given_condition_true'])} "
        f"| {_fmt(s['t_stat_excess_forward_diff'])} |"
    )


def _write_markdown(full: dict, panel: pd.DataFrame) -> None:
    r = full["results"]
    today_str = date.today().strftime("%Y-%m-%d")
    lines = [
        f"# Study98 — TOPIX17セクターモメンタム持続性（Factor-Level・{today_str}）",
        "",
        "目的: P(R_sector(t+1~3m)>R_TOPIX | R_sector(t)>R_TOPIX) をTOPIX-17公式指数で検証。"
        "戦略BT不使用・factor-level統計のみ。",
        "",
        f"データ: J-Quants公式TOPIX-17業種別指数（`database/market/index/prices/`）・"
        f"panel行数={len(panel):,}・期間={panel['rebalance_date'].min()}〜{panel['rebalance_date'].max()}",
        "",
        "## サマリー（プライマリ: 全量TOPIX超過セクター vs 上位3限定）",
        "",
        "| 区分 | N(条件成立) | 条件付き遷移確率 | 無条件base rate | z統計量 | 平均excess forward | t統計量 |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in HORIZONS:
        lines.append(_stats_row(f"全量・{label}", r["primary_all_qualifying"][label]))
        lines.append(_stats_row(f"上位3限定・{label}", r["top3_only"][label]))
    lines += ["", "判定基準（参考）: 遷移確率が無条件base rateを有意に上回れば（z>2目安）"
              "セクターモメンタム持続の証拠。t統計量はexcess forward returnの差の有意性。", ""]

    lines += ["## Confusion Matrix（全量・horizon別）", "",
              "| horizon | 条件T→結果T | 条件T→結果F | 条件F→結果T | 条件F→結果F |",
              "|---|---|---|---|---|"]
    for label in HORIZONS:
        cm = r["primary_all_qualifying"][label]["confusion_matrix"]
        lines.append(
            f"| {label} | {cm['condition_true_outcome_true']} | {cm['condition_true_outcome_false']} "
            f"| {cm['condition_false_outcome_true']} | {cm['condition_false_outcome_false']} |"
        )

    lines += ["", "## Regime別（TOPIX>MA200・全量ベース）", "",
              "| Regime | horizon | N | 条件付き遷移確率 | 無条件base rate | z統計量 | 平均excess forward | t統計量 |",
              "|---|---|---|---|---|---|---|---|"]
    for regime_key in ("bull", "bear"):
        for label in HORIZONS:
            s = r["regime"][regime_key][label]
            lines.append(
                f"| {regime_key} | {label} | {s['n_condition_true']} "
                f"| {_fmt_pct(s['transition_prob_given_condition_true'])} | {_fmt_pct(s['unconditional_prob'])} "
                f"| {_fmt(s['z_stat_proportion_diff'])} | {_fmt_pct(s['mean_excess_forward_given_condition_true'])} "
                f"| {_fmt(s['t_stat_excess_forward_diff'])} |"
            )

    lines += ["", "## サブ期間（2016-2020 / 2021-2025・全量ベース）", "",
              "| 期間 | horizon | N | 条件付き遷移確率 | 無条件base rate | z統計量 | 平均excess forward | t統計量 |",
              "|---|---|---|---|---|---|---|---|"]
    for sp_key in SUBPERIODS:
        for label in HORIZONS:
            s = r["subperiod"][sp_key][label]
            lines.append(
                f"| {sp_key} | {label} | {s['n_condition_true']} "
                f"| {_fmt_pct(s['transition_prob_given_condition_true'])} | {_fmt_pct(s['unconditional_prob'])} "
                f"| {_fmt(s['z_stat_proportion_diff'])} | {_fmt_pct(s['mean_excess_forward_given_condition_true'])} "
                f"| {_fmt(s['t_stat_excess_forward_diff'])} |"
            )

    lines += ["", "## セクター別内訳（1M horizon・全量ベース・N<10は除外）", "",
              "| セクター | N | 条件付き遷移確率 | 無条件base rate | z統計量 | 平均excess forward | t統計量 |",
              "|---|---|---|---|---|---|---|"]
    for sector_name, by_horizon in sorted(r["per_sector"].items()):
        if "1M" not in by_horizon:
            continue
        s = by_horizon["1M"]
        lines.append(
            f"| {sector_name} | {s['n_condition_true']} "
            f"| {_fmt_pct(s['transition_prob_given_condition_true'])} | {_fmt_pct(s['unconditional_prob'])} "
            f"| {_fmt(s['z_stat_proportion_diff'])} | {_fmt_pct(s['mean_excess_forward_given_condition_true'])} "
            f"| {_fmt(s['t_stat_excess_forward_diff'])} |"
        )

    lines += [
        "",
        "![transition chart](study98_transition_chart.png)",
        "",
        "---",
        "",
        f"*生成: Study98自動分析パイプライン, {today_str}。戦略BT不使用（factor-level統計のみ）。"
        "データソース=J-Quants公式TOPIX-17業種別指数（database/market/index/prices/、2026-07-15新規取得）。*",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
