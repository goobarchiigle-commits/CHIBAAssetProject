"""
src/backtest/study95e_momentum_reverse_decomposition.py
Study95E — Momentum Reverse Decomposition（post-mortem decomposition・Study82Eと同型フレーム）

正典: reports/study82e_root_cause_audit.md（Study82型分解フレームの前例・Size/Liquidity/Period/
      Regime/Horizon軸・root cause ranking手法をそのまま踏襲）
      backtests/study95_cs_momentum_factor_level.json（Study95本体・KILL判定=FAIL_ZERO_SPREAD起点）
実装承認: ユーザー"Study95E Momentum Reverse Decomposition"タスク指示（2026-07-22）

目的（狭く固定）:
  Study82（PEAD）逆転の構造がcross-sectional momentumにも一般化するかを判定する。
  NOT alpha search. NOT optimization. Post-mortem decomposition only。

方法（Study95本体の再利用・変更なし）:
  factor定義（12-1モメンタム・skip21d・lookback252d）・Universe C・価格ソース・
  Newey-West HAC t検定・block bootstrap CIは`study95_cs_momentum_factor_level.py`から
  そのままimportして再利用する（パラメータ変更・ロジック分岐なし＝alpha探索ではない）。
  新規追加はSize proxy（ShOutFY×Close、既存fins_summaryキャッシュ結合のみ）と
  bucket層別Q10-Q1スプレッド集計のみ。Clenow slope×R²は本分解の対象外
  （タイトル通りmomentum一本に絞りスコープを固定・別途要望があれば追加分解可能）。

禁止事項（厳守）: 新規アルファ探索・最適化・新規データ取得は一切行わない。

データ源（既存キャッシュのみ・Study95と同一 + fins_summaryのみ新規結合）:
  - backtests/study75_rule_universe.json （Universe C・Study95と同一）
  - data/jquants/processed/{code}.parquet （価格・Study95と同一）
  - data/jquants/processed/TOPIX.parquet （regime・Study95と同一）
  - database/market/master/companies.parquet （sector・本分解では不使用だが同一ソース確認済み）
  - backtests/study75_universe_diagnostics.parquet （ADV20・Study95と同一＝Liquidity軸に再利用）
  - data/jquants/cache/fins_summary/{code}.json （Study82Eと同一キャッシュ・Size proxy用に新規結合）

Tier軸: Size(tercile/月次CS) / Liquidity ADV(tercile/月次CS) / Time period(3区分) /
        Market regime(TOPIX>200MA) / Holding horizon(Study95本体の1M/3M/6M/12Mをそのまま転用)

セル最小サンプル: min_periods=5（Study95本体`_q10_q1_spread`と同一の閾値を踏襲）。
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats as sstats

try:
    from src.paths import JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR, BASE_DIR
    import src.backtest.study95_cs_momentum_factor_level as s95
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.paths import JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR, BASE_DIR
    import src.backtest.study95_cs_momentum_factor_level as s95

RUN_DATE = "2026-07-22"
FINS_CACHE_DIR = BASE_DIR / "data" / "jquants" / "cache" / "fins_summary"
STUDY95_JSON = RESULTS_DIR / "study95_cs_momentum_factor_level.json"
OUT_JSON = RESULTS_DIR / f"study95e_momentum_reverse_decomposition_{RUN_DATE}.json"
OUT_PANEL_CSV = RESULTS_DIR / f"study95e_panel_enriched_{RUN_DATE}.csv"
OUT_MD = REPORTS_DIR / "study95e_momentum_reverse_decomposition.md"

FIN_STATEMENT_DOCTYPE_RE = re.compile(r"FinancialStatements", re.IGNORECASE)
MIN_PERIODS = 5  # Study95本体 _q10_q1_spread と同一閾値


# ---------------------------------------------------------------- Size proxy（既存fins_summaryのみ）
def load_shoutfy_asof(codes: list[str], rebalance_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """各codeについてShOutFY(発行済株式数FY)をrebalance_date時点にasof(backward)結合。
    PIT安全（DiscDate<=rebalance_dateの直近開示値のみ使用）。新規API呼び出しなし。"""
    rb_df = pd.DataFrame({"rebalance_date": sorted(rebalance_dates)})
    cols: dict[str, np.ndarray] = {}
    for code in codes:
        fp = FINS_CACHE_DIR / f"{code}.json"
        if not fp.exists():
            continue
        recs = json.loads(fp.read_text(encoding="utf-8"))
        fin_recs = [r for r in recs if FIN_STATEMENT_DOCTYPE_RE.search(str(r.get("DocType", "")))]
        if not fin_recs:
            continue
        df = pd.DataFrame(fin_recs)
        df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
        df["ShOutFY"] = pd.to_numeric(df.get("ShOutFY"), errors="coerce")
        df = df.dropna(subset=["DiscDate", "ShOutFY"]).sort_values("DiscDate")
        if df.empty:
            continue
        merged = pd.merge_asof(rb_df, df[["DiscDate", "ShOutFY"]],
                                left_on="rebalance_date", right_on="DiscDate", direction="backward")
        cols[code] = merged["ShOutFY"].to_numpy()
    return pd.DataFrame(cols, index=rb_df["rebalance_date"].to_numpy())


# ---------------------------------------------------------------- 汎用Q10-Q1スプレッド（mask拡張版）
def masked_q10_q1_spread(panel: pd.DataFrame, decile_col: str, mask: pd.Series | None = None) -> dict:
    sub = panel[mask] if mask is not None else panel
    out = {}
    for label, h in s95.HORIZONS.items():
        months = s95.HORIZON_MONTHS[label]
        spreads = []
        for rb_date, g in sub.groupby("rebalance_date"):
            g = g.dropna(subset=[decile_col, f"fwd_{label}"])
            q10 = g.loc[g[decile_col] == s95.N_DECILES, f"fwd_{label}"]
            q1 = g.loc[g[decile_col] == 1, f"fwd_{label}"]
            if len(q10) == 0 or len(q1) == 0:
                continue
            spreads.append(float(q10.mean() - q1.mean()))
        spreads_arr = np.array(spreads)
        if len(spreads_arr) < MIN_PERIODS:
            out[label] = {"n_periods": int(len(spreads_arr)), "mean_spread": None, "annualized_spread": None,
                          "newey_west_t": None, "hit_ratio": None,
                          "note": f"n_periods<{MIN_PERIODS}: report only, no significance test"}
            continue
        mean_spread, t_stat = s95.newey_west_tstat(spreads_arr, lag=max(0, months - 1))
        ann = s95.annualize(mean_spread, months)
        hit_ratio = float((spreads_arr > 0).mean())
        out[label] = {"n_periods": int(len(spreads_arr)), "mean_spread": mean_spread,
                      "annualized_spread": ann, "newey_west_t": t_stat, "hit_ratio": hit_ratio, "note": None}
    return out


def bucketed_axis(panel: pd.DataFrame, axis_col: str, order: list[str], decile_col: str = "mom_decile") -> dict:
    cells = {}
    for bucket in order:
        mask = panel[axis_col] == bucket
        cells[bucket] = masked_q10_q1_spread(panel, decile_col, mask=mask)
    ann_12m = [cells[b]["12M"]["annualized_spread"] for b in order
               if cells[b]["12M"]["annualized_spread"] is not None]
    monotonic = None
    spearman_rho = None
    if len(ann_12m) == len(order) and len(order) >= 3:
        rho, _p = sstats.spearmanr(list(range(len(order))), ann_12m)
        spearman_rho = round(float(rho), 4) if rho == rho else None
        diffs = np.diff(ann_12m)
        monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    return {"order": order, "cells": cells, "monotonic": monotonic, "spread_rank_spearman_12m": spearman_rho,
            "spread_range_12m_ann": round(max(ann_12m) - min(ann_12m), 4) if ann_12m else None}


def main() -> None:
    print("Study95E — Momentum Reverse Decomposition（post-mortem・Study95本体のfactor定義をそのまま再利用）")

    print("[1/6] Universe C / TOPIXカレンダー読込（Study95本体と同一関数）...")
    monthly_universe = s95.load_universe()
    rebalance_dates = sorted(pd.Timestamp(k) for k in monthly_universe)
    topix = s95.load_topix_calendar()
    calendar = topix.index
    topix_sma200 = topix.rolling(s95.REGIME_MA_PERIOD, min_periods=s95.REGIME_MA_PERIOD).mean()
    regime_bull = topix >= topix_sma200

    print(f"[2/6] 価格パネル読込（Study95本体と同一関数・{len(rebalance_dates)}ヶ月）...")
    all_codes = sorted({c for v in monthly_universe.values() for c in v})
    close_df, missing_codes = s95.load_close_panel(all_codes, calendar)
    adv20_df = s95.load_adv20_map()
    adv20_df["date"] = adv20_df["date"].astype(str)
    print(f"  loaded={close_df.shape[1]}  missing={len(missing_codes)}")

    print("[3/6] Size proxy（ShOutFY asof結合・既存fins_summaryキャッシュのみ・新規取得ゼロ）...")
    shoutfy_df = load_shoutfy_asof(all_codes, rebalance_dates)
    print(f"  ShOutFYカバレッジ: {shoutfy_df.notna().any().sum()}/{len(all_codes)}銘柄")

    print("[4/6] 月次factor・forward return・bucket計算（Study95本体と同一ロジック + Size/Liquidity/Period追加）...")
    calendar_pos = {d: i for i, d in enumerate(calendar)}
    records: list[dict] = []

    def period_bucket(d: pd.Timestamp) -> str | float:
        y = d.year
        if 2016 <= y <= 2019:
            return "2016-2019"
        if 2020 <= y <= 2022:
            return "2020-2022"
        if 2023 <= y <= 2026:
            return "2023-2026"
        return np.nan

    for rb_date in rebalance_dates:
        rb_str = rb_date.strftime("%Y-%m-%d")
        universe_codes = monthly_universe[rb_str]
        if rb_date not in calendar_pos:
            later = calendar[calendar >= rb_date]
            if len(later) == 0:
                continue
            rb_date_eff = later[0]
        else:
            rb_date_eff = rb_date
        base_pos = calendar_pos[rb_date_eff]
        snap_pos = base_pos - 1
        if snap_pos < 0:
            continue

        universe_mask = pd.Series(False, index=close_df.columns)
        universe_mask.loc[[c for c in universe_codes if c in close_df.columns]] = True

        mom = s95.calc_momentum_row(close_df, snap_pos).where(universe_mask)
        fwd = s95.calc_forward_returns_row(close_df, base_pos)
        mom_decile = s95.assign_deciles(mom)

        snap_date = calendar[snap_pos]
        is_bull = bool(regime_bull.get(snap_date, False)) if snap_date in regime_bull.index else None

        adv_lookup = adv20_df.loc[adv20_df["date"] == rb_str].set_index("code")["adv20"] \
            if (adv20_df["date"] == rb_str).any() else pd.Series(dtype=float)

        snap_close = close_df.iloc[snap_pos]
        mktcap = shoutfy_df.loc[rb_date] * snap_close if rb_date in shoutfy_df.index else pd.Series(dtype=float)

        # 月次クロスセクション内tercile（Size・Liquidity）
        mktcap_u = mktcap.reindex(universe_codes).dropna()
        size_tercile = pd.Series(np.nan, index=universe_codes, dtype=object)
        if len(mktcap_u) >= 30:
            edges = mktcap_u.quantile([0, 1/3, 2/3, 1.0]).to_numpy().copy()
            edges[0] -= 1
            size_tercile.loc[mktcap_u.index] = pd.cut(
                mktcap_u, bins=edges, labels=["Small", "Mid", "Large"]).astype(object)

        adv_u = adv_lookup.reindex(universe_codes).dropna()
        adv_u = pd.to_numeric(adv_u, errors="coerce").dropna()
        liq_tercile = pd.Series(np.nan, index=universe_codes, dtype=object)
        if len(adv_u) >= 30:
            edges = adv_u.quantile([0, 1/3, 2/3, 1.0]).to_numpy().copy()
            edges[0] -= 1
            liq_tercile.loc[adv_u.index] = pd.cut(
                adv_u, bins=edges, labels=["Low", "Mid", "High"]).astype(object)

        pbucket = period_bucket(rb_date)

        for code in universe_codes:
            if code not in close_df.columns:
                continue
            row = {
                "rebalance_date": rb_str, "code": code,
                "mom_value": float(mom.get(code, np.nan)), "mom_decile": mom_decile.get(code, np.nan),
                "regime_bull": is_bull,
                "adv20": float(adv_u.get(code, np.nan)) if code in adv_u.index else np.nan,
                "mktcap_proxy": float(mktcap_u.get(code, np.nan)) if code in mktcap_u.index else np.nan,
                "size_bucket": size_tercile.get(code, np.nan),
                "liquidity_bucket": liq_tercile.get(code, np.nan),
                "period_bucket": pbucket,
            }
            for label in s95.HORIZONS:
                row[f"fwd_{label}"] = float(fwd[label].get(code, np.nan))
            records.append(row)

    panel = pd.DataFrame(records)
    panel["regime_bucket"] = panel["regime_bull"].map({True: "Above200MA", False: "Below200MA"})
    print(f"  panel rows={len(panel):,}")
    panel.to_csv(OUT_PANEL_CSV, index=False, encoding="utf-8")
    print(f"  Enriched panel CSV: {OUT_PANEL_CSV}")

    print("[5/6] Tier軸別Q10-Q1スプレッド集計...")
    result: dict = {"run_at": datetime.now(timezone.utc).isoformat(), "n_panel_rows": int(len(panel)),
                     "n_rebalance_dates": len(rebalance_dates)}

    result["axis_size"] = bucketed_axis(panel, "size_bucket", ["Small", "Mid", "Large"])
    result["axis_liquidity"] = bucketed_axis(panel, "liquidity_bucket", ["Low", "Mid", "High"])
    result["axis_period"] = bucketed_axis(panel, "period_bucket", ["2016-2019", "2020-2022", "2023-2026"])
    result["axis_regime"] = bucketed_axis(panel, "regime_bucket", ["Above200MA", "Below200MA"])

    if STUDY95_JSON.exists():
        s95_full = json.loads(STUDY95_JSON.read_text(encoding="utf-8"))
        result["axis_holding_horizon_from_study95_original"] = \
            s95_full["results"]["mom"]["q10_q1_spread"]

    ranking = []
    for name, axis_key in (("size", "axis_size"), ("liquidity", "axis_liquidity"),
                            ("time_period", "axis_period"), ("market_regime", "axis_regime")):
        v = result[axis_key].get("spread_range_12m_ann")
        if v is not None:
            ranking.append((name, v))
    ranking.sort(key=lambda kv: kv[1], reverse=True)
    result["root_cause_ranking_by_spread_range_12m"] = ranking

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[6/6] JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
