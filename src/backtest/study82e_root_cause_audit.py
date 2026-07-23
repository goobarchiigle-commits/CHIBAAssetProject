"""
src/backtest/study82e_root_cause_audit.py
Study82E — PEAD Reverse Root Cause Audit（post-mortem decomposition・実装）

正典: reports/study82e_proposal.md（起案書・実現可能性監査済み）
      reports/study82_phase_d_pead_alpha.md（Study82 Phase D実測・FAIL・スプレッド逆転の起点）
実装承認: ユーザー"Study82E Implementation"タスク指示（2026-07-22・本ファイルはその直接反映）

目的（狭く固定）:
  WHY did PEAD reverse in Study82 Phase D?
  NOT alpha research. NOT successor strategy design. Post-mortem decomposition only.

禁止事項（厳守・本ファイルはこれを一切行わない）:
  新規データ取得・新規API呼び出し・最適化・パラメータ探索・新規アルファ実装・
  ロードマップ変更・root cause確定前のsuccessor提案。

データ源（既存キャッシュのみ・新規取得ゼロ）:
  - backtests/study82_phase_d_pead_events_2026-07-21.csv （既存30,952件イベント台帳・再利用のみ）
  - data/jquants/processed/{code}.parquet （既存価格キャッシュ・Phase Dと同一ソース・Volume列も既存）
  - data/jquants/cache/fins_summary/{code}.json （既存財務キャッシュ・Phase Dと同一ソース）
  - data/jquants/processed/TOPIX.parquet （既存TOPIXキャッシュ）

Tier1（正式判定基準）: Size(3) / Liquidity ADV(3) / Time period(3) / Market regime(2) / Gap absorption(4窓)
Tier2（診断専用・survival宣言に使用禁止）: Surprise strength / Quarter type / pseudo F-score / pseudo GP/A

セル最小サンプル: min_trade_required=5（CLAUDE.md VALIDATION）。n<5は報告のみ・有意性検定なし。
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
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.paths import JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR, BASE_DIR

RUN_DATE = "2026-07-22"
EVENTS_CSV = RESULTS_DIR / "study82_phase_d_pead_events_2026-07-21.csv"
FINS_CACHE_DIR = BASE_DIR / "data" / "jquants" / "cache" / "fins_summary"
TOPIX_PARQUET = JQUANTS_PROCESSED_DIR / "TOPIX.parquet"
OUT_JSON = RESULTS_DIR / f"study82e_root_cause_audit_{RUN_DATE}.json"
OUT_ENRICHED_CSV = RESULTS_DIR / f"study82e_events_enriched_{RUN_DATE}.csv"
OUT_MD = REPORTS_DIR / "study82e_root_cause_audit.md"

HOLDING_DAYS = 40  # Phase Dと同一（変更禁止・パラメータ探索ではない）
ADV_LOOKBACK_DAYS = 60
MIN_TRADE_REQUIRED = 5  # CLAUDE.md VALIDATION trade_min
FIN_STATEMENT_DOCTYPE_RE = re.compile(r"FinancialStatements", re.IGNORECASE)
YOY_GAP_MIN_DAYS, YOY_GAP_MAX_DAYS = 330, 400  # Phase Dと同一の年次ペア照合窓

GAP_WINDOWS = [("r0_1", 0, 1), ("r2_5", 1, 5), ("r6_20", 5, 20), ("r21_40", 20, 40)]


# ---------------------------------------------------------------- 価格キャッシュ
_price_cache: dict[str, pd.DataFrame | None] = {}


def load_price_series(code: str) -> pd.DataFrame | None:
    if code in _price_cache:
        return _price_cache[code]
    fp = JQUANTS_PROCESSED_DIR / f"{code}.parquet"
    if not fp.exists():
        _price_cache[code] = None
        return None
    df = pd.read_parquet(fp)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    _price_cache[code] = df
    return df


# ---------------------------------------------------------------- 財務キャッシュ（既存JSON再利用）
_fins_cache: dict[str, pd.DataFrame | None] = {}


def load_fins_records(code: str) -> pd.DataFrame | None:
    if code in _fins_cache:
        return _fins_cache[code]
    fp = FINS_CACHE_DIR / f"{code}.json"
    if not fp.exists():
        _fins_cache[code] = None
        return None
    recs = json.loads(fp.read_text(encoding="utf-8"))
    fin_recs = [r for r in recs if FIN_STATEMENT_DOCTYPE_RE.search(str(r.get("DocType", "")))]
    if not fin_recs:
        _fins_cache[code] = None
        return None
    df = pd.DataFrame(fin_recs)
    df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    df["CurPerEn"] = pd.to_datetime(df["CurPerEn"], errors="coerce")
    for col in ("EPS", "NP", "TA", "CFO", "Sales", "OP", "ShOutFY"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df = df.dropna(subset=["DiscDate", "CurPerEn"]).sort_values("CurPerEn").reset_index(drop=True)
    _fins_cache[code] = df
    return df


def match_fins_record(fins: pd.DataFrame, cur_per_end: pd.Timestamp) -> pd.Series | None:
    hit = fins[fins["CurPerEn"] == cur_per_end]
    if hit.empty:
        return None
    return hit.iloc[-1]


def match_prior_yoy_record(fins: pd.DataFrame, cur: pd.Series) -> pd.Series | None:
    mask = (
        (fins["CurPerType"] == cur["CurPerType"])
        & ((cur["CurPerEn"] - fins["CurPerEn"]).dt.days >= YOY_GAP_MIN_DAYS)
        & ((cur["CurPerEn"] - fins["CurPerEn"]).dt.days <= YOY_GAP_MAX_DAYS)
    )
    cand = fins[mask]
    if cand.empty:
        return None
    return cand.iloc[-1]


# ---------------------------------------------------------------- TOPIX regime（既存キャッシュのみ）
def build_topix_regime() -> pd.Series:
    df = pd.read_parquet(TOPIX_PARQUET)
    df.index = pd.to_datetime(df.index)
    close = df["Close"].sort_index()
    ma200 = close.rolling(200, min_periods=200).mean()  # トレイリングのみ・先読みなし
    regime = pd.Series(np.where(close > ma200, "Above200MA", "Below200MA"), index=close.index)
    regime[ma200.isna()] = np.nan
    return regime


def topix_regime_at(regime: pd.Series, disc_date: pd.Timestamp) -> str | None:
    idx = regime.index
    at_or_before = idx[idx <= disc_date]
    if len(at_or_before) == 0:
        return None
    v = regime.loc[at_or_before[-1]]
    return v if isinstance(v, str) else None


# ---------------------------------------------------------------- イベント単位のエンリッチ
def enrich_event(row: pd.Series, topix_regime: pd.Series) -> dict:
    code = str(row["code"])
    disc_date = row["disc_date"]
    out: dict = {}

    price = load_price_series(code)
    if price is not None:
        idx = price.index
        after = idx[idx > disc_date]
        if len(after) > 0:
            entry_pos = idx.get_loc(after[0])
            exit_pos = entry_pos + HOLDING_DAYS
            if exit_pos < len(idx):
                # --- Liquidity (ADV, JPY turnover, entry前ADV_LOOKBACK_DAYS営業日・PIT安全) ---
                lb_start = max(0, entry_pos - ADV_LOOKBACK_DAYS)
                lb_slice = price.iloc[lb_start:entry_pos]
                if len(lb_slice) >= 20:
                    turnover = (lb_slice["Volume"] * lb_slice["Close"]).replace([np.inf, -np.inf], np.nan).dropna()
                    if len(turnover) >= 20:
                        out["adv_jpy"] = float(turnover.mean())

                # --- Size proxy (ShOutFY × entry前Close) ---
                entry_close = price["Close"].iloc[entry_pos]
                out["_entry_close_for_size"] = float(entry_close) if pd.notna(entry_close) else np.nan

                # --- Gap absorption decomposition (telescoping・raw_returnと合致するはず) ---
                pts = {}
                ok = True
                for _, a, b in GAP_WINDOWS:
                    if entry_pos + b >= len(idx):
                        ok = False
                        break
                    pts[a] = float(price["Close"].iloc[entry_pos + a]) if a > 0 else float(price["Open"].iloc[entry_pos])
                    pts[b] = float(price["Close"].iloc[entry_pos + b])
                if ok:
                    for label, a, b in GAP_WINDOWS:
                        base = pts[a]
                        end = pts[b]
                        if base and base > 0 and pd.notna(base) and pd.notna(end):
                            out[label] = float(end / base - 1.0)

    # --- Market regime (TOPIX 200MA at disc_date) ---
    out["topix_regime"] = topix_regime_at(topix_regime, disc_date)

    # --- Pseudo quality (fins_summary既存キャッシュのみ・COGS/流動資産負債内訳なし=簡易版) ---
    fins = load_fins_records(code)
    if fins is not None:
        cur = match_fins_record(fins, row["cur_per_end"])
        if cur is not None:
            ta, np_, cfo, op_ = cur.get("TA"), cur.get("NP"), cur.get("CFO"), cur.get("OP")
            if pd.notna(row.get("eps")) and pd.notna(cur.get("ShOutFY")) and cur.get("ShOutFY", 0) not in (0, None):
                pass  # ShOutFYはmarket cap proxyでのみ使用（下でentry_closeと結合）
            out["fins_shoutfy"] = float(cur["ShOutFY"]) if pd.notna(cur.get("ShOutFY")) else np.nan

            roa = (np_ / ta) if (pd.notna(ta) and ta not in (0, None) and pd.notna(np_)) else np.nan
            prior = match_prior_yoy_record(fins, cur)
            droa = np.nan
            if prior is not None and pd.notna(prior.get("TA")) and prior.get("TA", 0) not in (0, None) and pd.notna(prior.get("NP")):
                prior_roa = prior["NP"] / prior["TA"]
                if pd.notna(roa):
                    droa = roa - prior_roa

            score = 0
            n_avail = 0
            for cond, avail in (
                (roa if pd.notna(roa) else None, pd.notna(roa)),
                (droa if pd.notna(droa) else None, pd.notna(droa)),
                (cfo if pd.notna(cfo) else None, pd.notna(cfo)),
                ((cfo - np_) if (pd.notna(cfo) and pd.notna(np_)) else None, pd.notna(cfo) and pd.notna(np_)),
                (op_ if pd.notna(op_) else None, pd.notna(op_)),
            ):
                if avail:
                    n_avail += 1
                    if cond is not None and cond > 0:
                        score += 1
            out["pseudo_f_score"] = score if n_avail == 5 else np.nan  # 5点満点全項目揃った場合のみ採用
            out["pseudo_f_score_n_avail"] = n_avail
            out["pseudo_gpa_proxy"] = float(op_ / ta) if (pd.notna(op_) and pd.notna(ta) and ta not in (0, None)) else np.nan

    return out


# ---------------------------------------------------------------- 統計エンジン
def group_stats(sub: pd.DataFrame, ret_col: str = "cost_adj_return") -> dict:
    g = {}
    for grp in ("Positive", "Negative"):
        s = sub.loc[sub["surprise_group"] == grp, ret_col].dropna()
        n = len(s)
        if n == 0:
            g[grp] = {"n": 0, "mean_return_pct": None, "t_stat": None, "hit_ratio": None, "note": "no data"}
            continue
        mean = float(s.mean())
        hit = float((s > 0).mean())
        if n < MIN_TRADE_REQUIRED:
            g[grp] = {"n": n, "mean_return_pct": round(mean * 100, 3), "t_stat": None,
                      "hit_ratio": round(hit, 3), "note": f"n<{MIN_TRADE_REQUIRED}: report only, no significance test"}
            continue
        se = float(s.std(ddof=1) / np.sqrt(n))
        t = float(mean / se) if se > 0 else None
        g[grp] = {"n": n, "mean_return_pct": round(mean * 100, 3),
                  "t_stat": round(t, 2) if t is not None else None, "hit_ratio": round(hit, 3), "note": None}

    pos, neg = g["Positive"], g["Negative"]
    spread = None
    spread_t = None
    if pos["mean_return_pct"] is not None and neg["mean_return_pct"] is not None:
        spread = round(pos["mean_return_pct"] - neg["mean_return_pct"], 3)
        if pos["n"] >= MIN_TRADE_REQUIRED and neg["n"] >= MIN_TRADE_REQUIRED:
            s_pos = sub.loc[sub["surprise_group"] == "Positive", ret_col].dropna()
            s_neg = sub.loc[sub["surprise_group"] == "Negative", ret_col].dropna()
            tt = sstats.ttest_ind(s_pos, s_neg, equal_var=False)
            spread_t = round(float(tt.statistic), 2)

    ic = None
    s_all = sub.dropna(subset=["surprise_pct", ret_col])
    if len(s_all) >= MIN_TRADE_REQUIRED:
        rho, _p = sstats.spearmanr(s_all["surprise_pct"], s_all[ret_col])
        if rho == rho:
            ic = round(float(rho), 4)

    return {"Positive": pos, "Negative": neg, "spread_pct": spread, "spread_t_stat": spread_t, "ic_spearman": ic,
            "n_total": int(len(sub))}


def bucketed_axis(df: pd.DataFrame, axis_col: str, order: list[str]) -> dict:
    cells = {}
    for bucket in order:
        sub = df[df[axis_col] == bucket]
        cells[bucket] = group_stats(sub)
    spreads = [cells[b]["spread_pct"] for b in order if cells[b]["spread_pct"] is not None]
    monotonic = None
    spearman_rho = None
    if len(spreads) == len(order) and len(order) >= 3:
        ranks = list(range(len(order)))
        rho, _p = sstats.spearmanr(ranks, spreads)
        spearman_rho = round(float(rho), 4) if rho == rho else None
        diffs = np.diff(spreads)
        monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    return {"order": order, "cells": cells, "monotonic": monotonic, "spread_rank_spearman": spearman_rho,
            "spread_range_pct": round(max(spreads) - min(spreads), 3) if spreads else None}


def main() -> None:
    print("Study82E — PEAD Reverse Root Cause Audit（post-mortem decomposition・既存キャッシュのみ）")

    events = pd.read_csv(EVENTS_CSV, encoding="utf-8")
    events["disc_date"] = pd.to_datetime(events["disc_date"])
    events["cur_per_end"] = pd.to_datetime(events["cur_per_end"])
    events["code"] = events["code"].astype(str)
    events = events[events["surprise_group"].isin(["Positive", "Negative"])].reset_index(drop=True)
    print(f"既存イベント台帳ロード: {len(events)}件（新規取得ゼロ・{EVENTS_CSV.name}）")

    print("TOPIXレジーム構築中（既存TOPIX.parquetのみ・200MAトレイリング）...")
    topix_regime = build_topix_regime()

    print("イベント単位エンリッチ中（既存価格/財務キャッシュ結合のみ）...")
    enriched_rows = []
    for i, row in events.iterrows():
        enriched_rows.append(enrich_event(row, topix_regime))
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(events)}]", flush=True)
    enriched = pd.DataFrame(enriched_rows)
    df = pd.concat([events.reset_index(drop=True), enriched.reset_index(drop=True)], axis=1)

    # --- Size proxy = ShOutFY × entry前Close ---
    df["mktcap_proxy_jpy"] = df["fins_shoutfy"] * df["_entry_close_for_size"]

    # --- 整合性チェック: gap窓のtelescoping積 vs 既存raw_return ---
    have_all = df[["r0_1", "r2_5", "r6_20", "r21_40"]].notna().all(axis=1)
    if have_all.any():
        chained = ((1 + df.loc[have_all, "r0_1"]) * (1 + df.loc[have_all, "r2_5"])
                   * (1 + df.loc[have_all, "r6_20"]) * (1 + df.loc[have_all, "r21_40"])) - 1
        diff = (chained - df.loc[have_all, "raw_return"]).abs()
        print(f"整合性チェック（gap窓チェイン積 vs 既存raw_return）: "
              f"n={have_all.sum()}, max_abs_diff={diff.max():.6f}, mean_abs_diff={diff.mean():.6f}")

    # --- Tier1 bucketing ---
    valid_mktcap = df["mktcap_proxy_jpy"].dropna()
    if len(valid_mktcap) >= 30:
        edges = valid_mktcap.quantile([0, 1/3, 2/3, 1.0]).to_numpy().copy()
        edges[0] -= 1
        df["size_bucket"] = pd.cut(df["mktcap_proxy_jpy"], bins=edges, labels=["Small", "Mid", "Large"])
    else:
        df["size_bucket"] = np.nan

    valid_adv = df["adv_jpy"].dropna()
    if len(valid_adv) >= 30:
        edges = valid_adv.quantile([0, 1/3, 2/3, 1.0]).to_numpy().copy()
        edges[0] -= 1
        df["liquidity_bucket"] = pd.cut(df["adv_jpy"], bins=edges, labels=["Low", "Mid", "High"])
    else:
        df["liquidity_bucket"] = np.nan

    def period_bucket(d: pd.Timestamp) -> str | float:
        y = d.year
        if 2016 <= y <= 2019:
            return "2016-2019"
        if 2020 <= y <= 2022:
            return "2020-2022"
        if 2023 <= y <= 2026:
            return "2023-2026"
        return np.nan

    df["period_bucket"] = df["disc_date"].apply(period_bucket)

    # --- Tier2 bucketing (diagnostic only) ---
    def strength_bucket(pct: float, direction_series: pd.Series) -> str | float:
        return np.nan  # placeholder, replaced below per-direction

    df["abs_surprise"] = df["surprise_pct"].abs()
    df["surprise_strength_bucket"] = "Other"
    for grp in ("Positive", "Negative"):
        mask = df["surprise_group"] == grp
        if mask.sum() < 30:
            continue
        s = df.loc[mask, "abs_surprise"]
        q70, q90, q95, q99 = s.quantile([0.70, 0.90, 0.95, 0.99])
        bucket = pd.Series("Other", index=s.index)
        bucket[s >= q70] = "Top30%"
        bucket[s >= q90] = "Top10%"
        bucket[s >= q95] = "Top5%"
        bucket[s >= q99] = "Top1%"
        df.loc[mask, "surprise_strength_bucket"] = bucket

    df.to_csv(OUT_ENRICHED_CSV, index=False, encoding="utf-8")
    print(f"エンリッチ済みイベントCSV保存: {OUT_ENRICHED_CSV}")

    result: dict = {"run_at": datetime.now(timezone.utc).isoformat(), "n_events": int(len(df)),
                     "config": {"holding_days": HOLDING_DAYS, "adv_lookback_days": ADV_LOOKBACK_DAYS,
                                "min_trade_required": MIN_TRADE_REQUIRED, "gap_windows": GAP_WINDOWS}}

    print("\n--- Tier1: Size ---")
    result["tier1_size"] = bucketed_axis(df, "size_bucket", ["Small", "Mid", "Large"])
    print("--- Tier1: Liquidity (ADV terciles) ---")
    result["tier1_liquidity"] = bucketed_axis(df, "liquidity_bucket", ["Low", "Mid", "High"])
    print("--- Tier1: Time period ---")
    result["tier1_period"] = bucketed_axis(df, "period_bucket", ["2016-2019", "2020-2022", "2023-2026"])
    print("--- Tier1: Market regime ---")
    result["tier1_regime"] = bucketed_axis(df, "topix_regime", ["Above200MA", "Below200MA"])

    print("--- Tier1: Gap absorption decomposition ---")
    gap_result = {}
    for label, _a, _b in GAP_WINDOWS:
        gap_result[label] = group_stats(df.dropna(subset=[label]), ret_col=label)
    result["tier1_gap_absorption"] = gap_result

    # Gap absorption × regime（decision tree Case5判定に必要な追加切り口）
    gap_by_regime = {}
    for regime_v in ("Above200MA", "Below200MA"):
        sub_r = df[df["topix_regime"] == regime_v]
        gap_by_regime[regime_v] = {label: group_stats(sub_r.dropna(subset=[label]), ret_col=label)
                                    for label, _a, _b in GAP_WINDOWS}
    result["tier1_gap_absorption_by_regime"] = gap_by_regime

    print("--- Tier2 (diagnostic only): Surprise strength ---")
    result["tier2_surprise_strength"] = bucketed_axis(
        df, "surprise_strength_bucket", ["Other", "Top30%", "Top10%", "Top5%", "Top1%"])
    print("--- Tier2 (diagnostic only): Quarter type ---")
    qtypes = sorted(df["cur_per_type"].dropna().unique().tolist())
    result["tier2_quarter_type"] = bucketed_axis(df, "cur_per_type", qtypes)

    print("--- Tier2 (diagnostic only): pseudo quality ---")
    valid_pf = df["pseudo_f_score"].dropna()
    if len(valid_pf) >= 30:
        df["pseudo_f_bucket"] = pd.cut(df["pseudo_f_score"], bins=[-0.5, 1.5, 3.5, 5.5],
                                        labels=["Low(0-1)", "Mid(2-3)", "High(4-5)"])
        result["tier2_pseudo_f_score"] = bucketed_axis(df, "pseudo_f_bucket", ["Low(0-1)", "Mid(2-3)", "High(4-5)"])
    else:
        result["tier2_pseudo_f_score"] = {"note": f"n={len(valid_pf)} insufficient for bucketing"}

    valid_gpa = df["pseudo_gpa_proxy"].dropna()
    if len(valid_gpa) >= 30:
        edges = valid_gpa.quantile([0, 1/3, 2/3, 1.0]).to_numpy().copy()
        edges[0] -= 1
        df["pseudo_gpa_bucket"] = pd.cut(df["pseudo_gpa_proxy"], bins=edges, labels=["Low", "Mid", "High"])
        result["tier2_pseudo_gpa"] = bucketed_axis(df, "pseudo_gpa_bucket", ["Low", "Mid", "High"])
    else:
        result["tier2_pseudo_gpa"] = {"note": f"n={len(valid_gpa)} insufficient for bucketing"}

    # --- Root cause ranking (spread_range_pct ベース・機械算出・恣意的重み付けなし) ---
    ranking_candidates = [
        ("size", result["tier1_size"].get("spread_range_pct")),
        ("liquidity", result["tier1_liquidity"].get("spread_range_pct")),
        ("time_period", result["tier1_period"].get("spread_range_pct")),
        ("market_regime", result["tier1_regime"].get("spread_range_pct")),
    ]
    ranking_candidates = [(k, v) for k, v in ranking_candidates if v is not None]
    ranking_candidates.sort(key=lambda kv: kv[1], reverse=True)
    result["root_cause_ranking_by_spread_range"] = ranking_candidates

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nJSON: {OUT_JSON}")
    print(f"Enriched events CSV: {OUT_ENRICHED_CSV}")


if __name__ == "__main__":
    main()
