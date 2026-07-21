"""
src/backtest/study82_phase_d_pead_alpha.py
Study82 Phase D — PEAD (Post-Earnings-Announcement Drift) existence study

正典: reports/study82_phase0_report.md（Phase0=PASS・PEAD研究可能と確定済み）
      roadmap_v15_governance_layer.md §8A-8 Phase D「PEAD期待値の実測」
      alternative_architectures_5x_2026-07-03.md §ARCH-B（40d保有・翌営業日寄付エントリー原典）

目的（狭く固定・ユーザータスク仕様）:
  Determine whether PEAD retains investable alpha after realistic costs.
  最小・PIT安全・禁止: 最適化・パラメータスイープ・ポートフォリオ構築・overlay・ML。

固定仕様（事前固定・実行後の変更禁止）:
  - Universe: Study75 PIT universe のみ（backtests/study75_rule_universe.json monthly_universe）。
    イベントはDiscDateの月における直近スナップショット月でのUniverse-A所属で判定（PIT安全・
    hindsight無し）。
  - サプライズ定義: YoY EPS変化の符号のみ（同一CurPerTypeの1年前開示とのEPS比較）。
    パーセンタイル・閾値スイープは行わない（単純な符号2群分類のみ）。
  - エントリー: 発表翌営業日始値（canon「発表翌営業日寄付エントリー」・場中/引後の区別による
    執行タイミング分岐はしない=最小実装として単純化）。
  - 保有: 40営業日固定（canon例示値をそのまま採用）。
  - コスト: PARAMS_LOCKED（slippage=0.001・commission=0.00055）を往復適用。パラメータ変更禁止。
  - 価格調整: AdjustmentFactorによる分割調整は未実施（生Close使用）。保有期間中の日次リターンが
    ±40%を超えるイベントは分割等のコーポレートアクション疑いとして除外（事前固定の健全性フィルタ・
    最適化ではない）。
  - CAGR推定: 単一スロット逐次複利（252/40≈6.3サイクル/年）。複数スロット・資金配分は
    「ポートフォリオ構築」に該当するため実施しない。
  - Capacity: 既存Study75A ADV20資産（backtests/study75_universe_diagnostics.parquet）を再利用
    （新規計算・新規収集なし）。

PASS/FAIL判定（事前固定・結果を見た後の変更禁止）:
  PASS ⇔ A(年間イベント数≥30 [positive surprise群]) ∧ B(コスト後平均CAR≥+1.5%/40d [positive群])
         ∧ C(PEADスプレッド[positive-negative、コスト後]>0) ∧ D(単一スロットCAGR推定>TOPIX B&H CAGR)
  FAIL ⇔ 上記いずれか不成立

PASS: Proceed to portfolio design.
FAIL: Freeze PEAD research and promote Cross-sectional Momentum Proposal.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

try:
    from src.paths import JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR, BASE_DIR
    from src.jquants.provider import JQuantsProvider
    from src.jquants.exceptions import JQuantsAPIError, JQuantsConfigError
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.paths import JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR, BASE_DIR
    from src.jquants.provider import JQuantsProvider
    from src.jquants.exceptions import JQuantsAPIError, JQuantsConfigError

RUN_DATE = "2026-07-21"
UNIVERSE_FILE = RESULTS_DIR / "study75_rule_universe.json"
DIAGNOSTICS_PARQUET = RESULTS_DIR / "study75_universe_diagnostics.parquet"
TOPIX_PARQUET = JQUANTS_PROCESSED_DIR / "TOPIX.parquet"
CACHE_DIR = BASE_DIR / "data" / "jquants" / "cache" / "fins_summary"
OUT_JSON = RESULTS_DIR / f"study82_phase_d_pead_alpha_{RUN_DATE}.json"
OUT_EVENTS_CSV = RESULTS_DIR / f"study82_phase_d_pead_events_{RUN_DATE}.csv"
OUT_MD = REPORTS_DIR / "study82_phase_d_pead_alpha.md"

HOLDING_DAYS = 40
YOY_GAP_MIN_DAYS, YOY_GAP_MAX_DAYS = 330, 400  # ~1年±tolerance（固定・スイープなし）
CORP_ACTION_GUARD_ABS_RET = 0.40  # 保有中日次リターンの異常検知（分割等の疑い・固定閾値）
SLIPPAGE, COMMISSION = 0.001, 0.00055  # PARAMS_LOCKED（変更禁止）
ROUND_TRIP_COST = 2 * (SLIPPAGE + COMMISSION)  # 往復
TRADING_DAYS_PER_YEAR = 252
SLOT_CYCLES_PER_YEAR = TRADING_DAYS_PER_YEAR / HOLDING_DAYS  # 単一スロット逐次複利
FIN_STATEMENT_DOCTYPE_RE = re.compile(r"FinancialStatements", re.IGNORECASE)

PASS_GATE = {"min_events_per_year": 30, "min_car_pct": 1.5}


def load_universe_membership() -> dict[str, list[str]]:
    d = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    return d["monthly_universe"]  # {"YYYY-MM-01": [code, ...], ...}


def month_key(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-01")


FINS_SUMMARY_THROTTLE_SEC = 1.2  # 実測: 高速連投で/v2/fins/summaryが持続的429となるため保守的に設定
MAX_RETRY_PASSES = 2


def fetch_fins_summary_once(provider: JQuantsProvider, code: str) -> list[dict] | None:
    """
    成功時のみリストを返す。リトライ上限到達等の失敗はNone（呼び出し側はキャッシュしないこと
    — 空リストと失敗を区別しないと「データなし」と「取得失敗」が混同されキャッシュが汚染される）。
    """
    try:
        records, _path = provider.get_fins_summary(code=code)
        return records
    except JQuantsAPIError as e:
        print(f"    code={code}: 取得失敗（キャッシュしない・後続パスで再試行）: {e}", flush=True)
        return None


def collect_all_disclosures(codes: list[str]) -> dict[str, list[dict]]:
    provider = JQuantsProvider()
    out: dict[str, list[dict]] = {}
    t0 = time.time()
    n_cached_hit = 0
    failed_codes: list[str] = []

    for i, code in enumerate(codes):
        cache_file = CACHE_DIR / f"{code}.json"
        if cache_file.exists():
            out[code] = json.loads(cache_file.read_text(encoding="utf-8"))
            n_cached_hit += 1
        else:
            time.sleep(FINS_SUMMARY_THROTTLE_SEC)
            recs = fetch_fins_summary_once(provider, code)
            if recs is None:
                failed_codes.append(code)
                out[code] = []  # 集計上は空扱い（キャッシュには書かない）
            else:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
                out[code] = recs
        if (i + 1) % 100 == 0 or (i + 1) == len(codes):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(codes)}] cached_hit={n_cached_hit} failed={len(failed_codes)} "
                  f"elapsed={elapsed:.0f}s avg={elapsed/(i+1):.2f}s/code", flush=True)

    for retry_pass in range(MAX_RETRY_PASSES):
        if not failed_codes:
            break
        print(f"\n再試行パス{retry_pass+1}/{MAX_RETRY_PASSES}: {len(failed_codes)}件", flush=True)
        still_failed: list[str] = []
        for code in failed_codes:
            time.sleep(FINS_SUMMARY_THROTTLE_SEC * 2)
            recs = fetch_fins_summary_once(provider, code)
            if recs is None:
                still_failed.append(code)
            else:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                (CACHE_DIR / f"{code}.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
                out[code] = recs
        failed_codes = still_failed

    if failed_codes:
        print(f"\n⚠ 最終的に取得失敗（データなしと未取得の区別のため0件scan対象から除外）: "
              f"{len(failed_codes)}件 = {failed_codes[:20]}{'...' if len(failed_codes) > 20 else ''}", flush=True)
        (RESULTS_DIR / f"study82_phase_d_failed_codes_{RUN_DATE}.json").write_text(
            json.dumps(failed_codes, ensure_ascii=False), encoding="utf-8")
    return out


def build_surprise_events(disclosures: dict[str, list[dict]], membership: dict[str, list[str]]) -> pd.DataFrame:
    # PIT membership lookup: month -> set(codes)
    month_sets = {m: set(c) for m, c in membership.items()}
    sorted_months = sorted(month_sets.keys())

    def nearest_month_at_or_before(dt: pd.Timestamp) -> str | None:
        mk = month_key(dt)
        candidates = [m for m in sorted_months if m <= mk]
        return candidates[-1] if candidates else None

    rows = []
    for code, recs in disclosures.items():
        fin_recs = [r for r in recs if FIN_STATEMENT_DOCTYPE_RE.search(str(r.get("DocType", "")))]
        if len(fin_recs) < 5:
            continue
        df = pd.DataFrame(fin_recs)
        df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
        df["CurPerEn"] = pd.to_datetime(df["CurPerEn"], errors="coerce")
        df["EPS"] = pd.to_numeric(df["EPS"], errors="coerce")
        df = df.dropna(subset=["DiscDate", "CurPerEn"]).sort_values("CurPerEn").reset_index(drop=True)

        for i in range(len(df)):
            cur = df.iloc[i]
            if pd.isna(cur["EPS"]) or cur["EPS"] == 0:
                continue
            prior_mask = (
                (df["CurPerType"] == cur["CurPerType"])
                & ((cur["CurPerEn"] - df["CurPerEn"]).dt.days >= YOY_GAP_MIN_DAYS)
                & ((cur["CurPerEn"] - df["CurPerEn"]).dt.days <= YOY_GAP_MAX_DAYS)
            )
            prior_candidates = df[prior_mask]
            if prior_candidates.empty:
                continue
            prior = prior_candidates.iloc[-1]
            if pd.isna(prior["EPS"]) or prior["EPS"] == 0:
                continue

            month = nearest_month_at_or_before(cur["DiscDate"])
            if month is None or code not in month_sets.get(month, set()):
                continue  # PIT: Study75 universe非所属月の開示は除外

            surprise_pct = (cur["EPS"] - prior["EPS"]) / abs(prior["EPS"])
            rows.append({
                "code": code, "disc_date": cur["DiscDate"], "cur_per_end": cur["CurPerEn"],
                "cur_per_type": cur["CurPerType"], "eps": cur["EPS"], "eps_prior_yoy": prior["EPS"],
                "surprise_pct": surprise_pct,
                "surprise_group": "Positive" if surprise_pct > 0 else ("Negative" if surprise_pct < 0 else "Zero"),
                "universe_month": month,
            })
    return pd.DataFrame(rows)


def load_price_series(code: str) -> pd.DataFrame | None:
    fp = JQUANTS_PROCESSED_DIR / f"{code}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def compute_event_return(price: pd.DataFrame, disc_date: pd.Timestamp) -> tuple[float, float] | None:
    """entry=翌営業日始値・exit=entryからHOLDING_DAYS営業日後の終値。異常リターンは除外(None)。"""
    idx = price.index
    after = idx[idx > disc_date]
    if len(after) == 0:
        return None
    entry_date = after[0]
    entry_pos = idx.get_loc(entry_date)
    exit_pos = entry_pos + HOLDING_DAYS
    if exit_pos >= len(idx):
        return None
    entry_px = price["Open"].iloc[entry_pos]
    exit_px = price["Close"].iloc[exit_pos]
    if entry_px <= 0 or pd.isna(entry_px) or pd.isna(exit_px):
        return None

    window_close = price["Close"].iloc[entry_pos:exit_pos + 1]
    daily_ret = window_close.pct_change().dropna()
    if (daily_ret.abs() > CORP_ACTION_GUARD_ABS_RET).any():
        return None  # コーポレートアクション疑い・除外（固定ガード）

    raw_ret = float(exit_px / entry_px - 1.0)
    cost_adj_ret = raw_ret - ROUND_TRIP_COST
    return raw_ret, cost_adj_ret


def topix_baseline_cagr() -> float:
    df = pd.read_parquet(TOPIX_PARQUET)
    close = df["Close"].sort_index()
    n_years = len(close) / TRADING_DAYS_PER_YEAR
    return float((close.iloc[-1] / close.iloc[0]) ** (1.0 / n_years) - 1.0)


def capacity_estimate(events: pd.DataFrame) -> dict:
    if not DIAGNOSTICS_PARQUET.exists() or events.empty:
        return {"available": False}
    diag = pd.read_parquet(DIAGNOSTICS_PARQUET)
    diag["date"] = pd.to_datetime(diag["date"])
    diag["adv20"] = pd.to_numeric(diag["adv20"], errors="coerce")
    ev = events.copy()
    ev["month_dt"] = pd.to_datetime(ev["universe_month"])
    merged = ev.merge(diag, left_on=["code", "month_dt"], right_on=["code", "date"], how="left")
    adv = merged["adv20"].dropna()
    if adv.empty:
        return {"available": False}
    return {
        "available": True, "n_matched": int(len(adv)),
        "median_adv20_jpy": float(adv.median()), "p25_adv20_jpy": float(adv.quantile(0.25)),
        "note": "ADV20の中央値・p25。10%participation上限を仮定した場合の1イベントあたり執行可能額の目安="
                "ADV20×10%（既存Study75A資産の再利用・新規計算なし）。",
        "implied_per_event_capacity_jpy_at_10pct_participation": float(adv.median() * 0.10),
    }


def main() -> None:
    print("Study82 Phase D — PEAD existence study（PIT-safe・最小実装・禁止事項: 最適化/スイープ/"
          "ポートフォリオ構築/overlay/ML）")
    print(f"目的: PEADはリアリスティックなコスト控除後も投資可能なアルファを持つか\n")

    membership = load_universe_membership()
    all_codes = sorted({c for codes in membership.values() for c in codes})
    print(f"Study75 PITユニバース: {len(membership)}ヶ月・ユニーク銘柄{len(all_codes)}件")

    print("\n決算開示データ収集中（キャッシュ済み銘柄はスキップ）...")
    disclosures = collect_all_disclosures(all_codes)
    n_with_data = sum(1 for v in disclosures.values() if v)
    print(f"開示データ取得済み銘柄: {n_with_data}/{len(all_codes)}")

    print("\nYoYサプライズイベント構築中（PIT所属フィルタ適用）...")
    events = build_surprise_events(disclosures, membership)
    events = events[events["surprise_group"] != "Zero"].reset_index(drop=True)
    print(f"サプライズイベント（Positive/Negative）: {len(events)}件")

    print("\n価格データ照合・リターン計算中...")
    price_cache: dict[str, pd.DataFrame | None] = {}
    raw_rets, cost_rets = [], []
    for i, row in events.iterrows():
        code = row["code"]
        if code not in price_cache:
            price_cache[code] = load_price_series(code)
        price = price_cache[code]
        if price is None:
            raw_rets.append(np.nan); cost_rets.append(np.nan); continue
        result = compute_event_return(price, row["disc_date"])
        if result is None:
            raw_rets.append(np.nan); cost_rets.append(np.nan)
        else:
            raw_rets.append(result[0]); cost_rets.append(result[1])
    events["raw_return"] = raw_rets
    events["cost_adj_return"] = cost_rets
    events_valid = events.dropna(subset=["cost_adj_return"]).copy()
    print(f"価格照合・健全性フィルタ通過: {len(events_valid)}/{len(events)}件")

    events_valid.to_csv(OUT_EVENTS_CSV, index=False, encoding="utf-8")

    date_span_years = (events_valid["disc_date"].max() - events_valid["disc_date"].min()).days / 365.25 \
        if len(events_valid) else np.nan

    summary = {}
    for grp in ("Positive", "Negative"):
        g = events_valid[events_valid["surprise_group"] == grp]
        n = len(g)
        n_per_year = n / date_span_years if date_span_years and date_span_years > 0 else np.nan
        mean_raw = float(g["raw_return"].mean()) if n else np.nan
        mean_cost = float(g["cost_adj_return"].mean()) if n else np.nan
        se_cost = float(g["cost_adj_return"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        t_stat = float(mean_cost / se_cost) if se_cost and se_cost > 0 else np.nan
        win_rate = float((g["cost_adj_return"] > 0).mean()) if n else np.nan
        summary[grp] = {
            "n_events": n, "n_events_per_year": round(n_per_year, 1) if n_per_year == n_per_year else None,
            "mean_raw_return_pct": round(mean_raw * 100, 3) if mean_raw == mean_raw else None,
            "mean_cost_adj_return_pct": round(mean_cost * 100, 3) if mean_cost == mean_cost else None,
            "se_cost_adj_pct": round(se_cost * 100, 3) if se_cost == se_cost else None,
            "t_stat": round(t_stat, 2) if t_stat == t_stat else None,
            "win_rate": round(win_rate, 3) if win_rate == win_rate else None,
        }

    pead_spread_pct = None
    if summary["Positive"]["mean_cost_adj_return_pct"] is not None and summary["Negative"]["mean_cost_adj_return_pct"] is not None:
        pead_spread_pct = round(summary["Positive"]["mean_cost_adj_return_pct"] - summary["Negative"]["mean_cost_adj_return_pct"], 3)

    pos_mean_cost = summary["Positive"]["mean_cost_adj_return_pct"]
    cagr_estimate = None
    if pos_mean_cost is not None:
        cagr_estimate = float((1.0 + pos_mean_cost / 100.0) ** SLOT_CYCLES_PER_YEAR - 1.0)

    topix_cagr = topix_baseline_cagr()
    cap = capacity_estimate(events_valid)

    gate_a = (summary["Positive"]["n_events_per_year"] or 0) >= PASS_GATE["min_events_per_year"]
    gate_b = (pos_mean_cost or -999) >= PASS_GATE["min_car_pct"]
    gate_c = (pead_spread_pct or -999) > 0
    gate_d = (cagr_estimate if cagr_estimate is not None else -999) > topix_cagr
    verdict = "PASS" if (gate_a and gate_b and gate_c and gate_d) else "FAIL"

    result = {
        "config": {
            "holding_days": HOLDING_DAYS, "round_trip_cost_pct": ROUND_TRIP_COST * 100,
            "slot_cycles_per_year": round(SLOT_CYCLES_PER_YEAR, 2),
            "corp_action_guard_abs_ret": CORP_ACTION_GUARD_ABS_RET,
            "yoy_gap_days": [YOY_GAP_MIN_DAYS, YOY_GAP_MAX_DAYS],
            "universe_source": str(UNIVERSE_FILE),
        },
        "universe_stats": {"n_months": len(membership), "n_unique_codes": len(all_codes),
                            "n_codes_with_disclosure_data": n_with_data},
        "event_counts": {"n_surprise_events_total": len(events), "n_valid_priced_events": len(events_valid),
                          "date_span_years": round(date_span_years, 2) if date_span_years == date_span_years else None},
        "group_stats": summary,
        "pead_spread_cost_adj_pct": pead_spread_pct,
        "cagr_estimate_single_slot": round(cagr_estimate, 4) if cagr_estimate is not None else None,
        "topix_baseline_cagr": round(topix_cagr, 4),
        "capacity_estimate": cap,
        "pass_fail_gates": {
            "A_events_per_year_ge_30": gate_a, "B_positive_car_ge_1.5pct": gate_b,
            "C_pead_spread_positive": gate_c, "D_cagr_beats_topix": gate_d,
        },
        "final_verdict": verdict,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print("\n===== Study82 Phase D 結果 =====")
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    print(f"\n最終判定: {verdict}")
    print(f"JSON: {OUT_JSON}")
    print(f"Events CSV: {OUT_EVENTS_CSV}")


if __name__ == "__main__":
    main()
