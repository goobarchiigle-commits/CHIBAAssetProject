"""
src/backtest/study75c_e1_bootstrap.py
Study75C E1 — PITブートストラップ・ユニバースによる生存者+選定バイアスの単離。

設計正典: reports/study75c_interpretation.md §5.3

要点:
  - 2017-12-29（IS開始のT-1）スナップショットでPIT適格候補プールを構築
    （生存中・上場60営業日以上・lot feasible・ADV20がRSR42同日レンジ内）。
  - K=20個の42銘柄ユニバースを無作為抽出（プールサイズ42固定→パーセンタイル・
    セマンティクスがU0と構造同一）。
  - 各ドローの「生存者置換ツイン」（死亡銘柄をADV最近傍の生存銘柄で置換=2026年に
    リストを構築した場合の再現）も対で実行。
  - 分解: 生存者バイアス≈median(ツイン)−median(PIT) / 選定バイアス≈U0'−median(ツイン)。
  - セクター情報なし（enrichment未実行）のため bear_universe_filter のセクター除外は
    ドロー・U0'再アンカーの双方で無効化して整合（U0公式値8.71%と併記）。

エンジン(composite_alpha_bt.py)無改変。Study75Bのシナリオ構築機構を再利用。
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import src.backtest.composite_alpha_bt as cab
from src.backtest.study75b_survivorship_bias import (
    IS_END, IS_START, OOS_END, OOS_START,
    build_scenario_dataset, run_bt, window_dataset, and_masks, summarize,
)
from src.paths import JQUANTS_PROCESSED_DIR, RESULTS_DIR

_JST = timezone(timedelta(hours=9))

K_DRAWS = 20
SEED = 42
LOT_SIZE = 100
LOT_COST_THRESHOLD = 3_000_000 * 0.30
MIN_LISTING_TRADING_DAYS = 60
SNAPSHOT_DATE = "2017-12-29"  # IS開始(2018-01-04)のT-1営業日
U0_JQUANTS_OFFICIAL = {"IS": 8.71, "OOS": -0.98}  # Study75B U0（bearセクター除外あり・参考）

OUT_FILE = RESULTS_DIR / f"study75c_e1_bootstrap_{datetime.now(_JST).strftime('%Y-%m-%d')}.json"


def get_active_no_sector_filter(ds: dict, start: str, end: str) -> pd.DataFrame:
    """dyn_rsr42_bear_rs0（bearセクター除外なし・ドロー/U0'共通の整合条件）。"""
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=list(ds["trade_syms"].keys()), start=start, end=end,
        bear_exclude_sectors=None, sym_sector_map=None,
    )


def run_universe(name: str, symbols: list[str]) -> dict:
    """
    42銘柄ユニバース1本のIS/OOS実行（Study75Bと同一wiring・bearセクター除外なし）。

    セクターは各銘柄固有の擬似セクター（sector=自銘柄コード）を割り当てる。
    エンジンのsector cap（composite_alpha_bt.py L1431-1445・セクター合計ウェイト上限）は
    全銘柄が同一セクター('不明')だとポートフォリオ全体を1セクター扱いで約25%に制限してしまい
    実質1ポジション戦略に退化する（初回E1実行で発覚）。銘柄固有セクターなら
    「セクター内1銘柄」の公式挙動と同一になり、capは既存の銘柄別上限と同水準で非拘束。
    """
    sectors = {s: s for s in symbols}
    ds = build_scenario_dataset(sectors)
    out = {"effective_size": len(ds["trade_syms"]), "short_history_excluded": len(ds["short_history"]),
           "missing": len(ds["missing"])}
    for wname, s, e, years in (("IS", IS_START, IS_END, 7.0), ("OOS", OOS_START, OOS_END, 1.0)):
        try:
            dsw = window_dataset(ds, s, e)
            act = get_active_no_sector_filter(dsw, s, e)
            act = and_masks(act, ds["alive_df"])
            raw = run_bt(dsw, act, s, e)
            out[wname] = summarize(raw, years)
        except Exception as exc:  # noqa: BLE001
            out[wname] = {"error": str(exc)}
        cagr = out[wname].get("cagr")
        print(f"  [{name}] {wname}: CAGR={cagr}")
    return out


def build_eligible_pool() -> pd.DataFrame:
    """
    2017-12-29スナップショットのPIT適格候補プール。
    daily_bars_2017（バルク・実バーのみ）からtrailing 20営業日のADV20・終値を全銘柄分計算する。
    """
    df = pd.read_parquet("data/jquants/processed/daily_bars_2017.parquet",
                         columns=["Date", "Code", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    snap = pd.Timestamp(SNAPSHOT_DATE)
    window = df.loc[(df["Date"] > snap - pd.Timedelta(days=40)) & (df["Date"] <= snap)].copy()
    window["TradedValue"] = window["Close"] * window["Volume"]
    agg = window.groupby("Code").agg(
        adv20=("TradedValue", "mean"), n_days=("Date", "count"), last_close=("Close", "last"),
    )
    agg = agg.loc[agg["n_days"] >= 15]
    agg["lot_cost"] = agg["last_close"] * LOT_SIZE

    # 上場60営業日以上: first_seen（イベントログ由来）が2017-12-29の60営業日以上前。
    # データセット開始(2016-07-11)からの左検閲銘柄はfirst_seen=開始日≒17ヶ月前なので自動的に通過。
    ref = pd.read_parquet(JQUANTS_PROCESSED_DIR / "universe.parquet")
    ref["first_seen_date"] = pd.to_datetime(ref["first_seen_date"])
    first_seen = ref.set_index("code")["first_seen_date"]
    listing_cutoff = snap - pd.Timedelta(days=MIN_LISTING_TRADING_DAYS * 1.45)  # 営業日→暦日概算
    agg = agg.join(first_seen, how="left")
    agg = agg.loc[agg["first_seen_date"].notna() & (agg["first_seen_date"] <= listing_cutoff)]

    agg = agg.loc[agg["lot_cost"] <= LOT_COST_THRESHOLD]
    agg["is_currently_listed"] = agg.index.map(ref.set_index("code")["is_currently_listed"])
    return agg


def main() -> int:
    started = datetime.now(_JST)
    rng = np.random.default_rng(SEED)

    # ── RSR42の同日ADV20レンジ（流動性プロファイル帯）────────────────────
    official = cab._load_rsr_universe(verbose=False)
    rsr42_syms = list(official.keys())
    pool = build_eligible_pool()

    rsr42_codes = [s.split(".")[0] + "0" for s in rsr42_syms]
    rsr42_in_pool = pool.loc[pool.index.isin(rsr42_codes)]
    adv_lo, adv_hi = rsr42_in_pool["adv20"].min(), rsr42_in_pool["adv20"].max()
    print(f"RSR42 ADV20 range at {SNAPSHOT_DATE}: [{adv_lo/1e6:.0f}M, {adv_hi/1e6:.0f}M] "
          f"(matched {len(rsr42_in_pool)}/42 in pool)")

    eligible = pool.loc[(pool["adv20"] >= adv_lo) & (pool["adv20"] <= adv_hi)].copy()
    survivors = eligible.loc[eligible["is_currently_listed"] == True]  # noqa: E712
    eventually_dead = eligible.loc[eligible["is_currently_listed"] == False]  # noqa: E712
    print(f"eligible pool={len(eligible)} (survivors={len(survivors)} eventually_dead={len(eventually_dead)} "
          f"→ PIT死亡率 {100*len(eventually_dead)/len(eligible):.1f}%)")

    # ── U0' 再アンカー（bearセクター除外なし・ドローと同一条件）──────────
    print("\n[U0'] official RSR42 without bear-sector-exclusion (consistency anchor)...")
    u0_prime = run_universe("U0'", rsr42_syms)

    # ── K本のPITドロー + 生存者置換ツイン ─────────────────────────────────
    eligible_codes = eligible.index.to_list()
    survivor_by_adv = survivors.sort_values("adv20")
    draws = []
    for k in range(K_DRAWS):
        draw_codes = list(rng.choice(eligible_codes, size=42, replace=False))
        dead_in_draw = [c for c in draw_codes if c in eventually_dead.index]

        # ツイン: 死亡銘柄をADV最近傍の生存銘柄（未使用のもの）で置換
        twin_codes = list(draw_codes)
        used = set(twin_codes)
        for dc in dead_in_draw:
            target_adv = eligible.loc[dc, "adv20"]
            cands = survivor_by_adv.loc[~survivor_by_adv.index.isin(used)]
            nearest = (cands["adv20"] - target_adv).abs().idxmin()
            twin_codes[twin_codes.index(dc)] = nearest
            used.add(nearest)

        print(f"\n[draw {k:02d}] dead_in_draw={len(dead_in_draw)}")
        pit_res = run_universe(f"pit{k:02d}", draw_codes)
        twin_res = run_universe(f"twin{k:02d}", twin_codes)
        draws.append({
            "k": k, "n_dead": len(dead_in_draw),
            "dead_codes": dead_in_draw,
            "pit": pit_res, "twin": twin_res,
        })

    # ── 分布統計と分解 ────────────────────────────────────────────────────
    def dist_stats(vals: list[float]) -> dict:
        a = np.array([v for v in vals if v is not None])
        if len(a) == 0:
            return {}
        return {
            "n": int(len(a)), "mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
            "std": round(float(a.std()), 2),
            "p5": round(float(np.percentile(a, 5)), 2), "p25": round(float(np.percentile(a, 25)), 2),
            "p75": round(float(np.percentile(a, 75)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "values": [round(float(v), 2) for v in a],
        }

    summary = {}
    for w in ("IS", "OOS"):
        pit_cagrs = [d["pit"].get(w, {}).get("cagr") for d in draws]
        twin_cagrs = [d["twin"].get(w, {}).get("cagr") for d in draws]
        pit_stats = dist_stats(pit_cagrs)
        twin_stats = dist_stats(twin_cagrs)
        u0p = u0_prime.get(w, {}).get("cagr")

        pit_arr = np.array([v for v in pit_cagrs if v is not None])
        u0_pctile = round(float((pit_arr < u0p).mean() * 100), 1) if (u0p is not None and len(pit_arr)) else None

        paired = [
            (d["twin"][w]["cagr"] - d["pit"][w]["cagr"])
            for d in draws
            if d["pit"].get(w, {}).get("cagr") is not None and d["twin"].get(w, {}).get("cagr") is not None
        ]
        survivorship_bias = round(float(np.mean(paired)), 2) if paired else None

        summary[w] = {
            "pit_distribution": pit_stats,
            "twin_distribution": twin_stats,
            "u0_prime_cagr": u0p,
            "u0_percentile_in_pit_distribution": u0_pctile,
            "survivorship_bias_pp_paired": survivorship_bias,  # mean(twin - pit)
            "selection_bias_pp": round(u0p - twin_stats["median"], 2) if (u0p is not None and twin_stats) else None,
            "combined_bias_pp": round(u0p - pit_stats["median"], 2) if (u0p is not None and pit_stats) else None,
        }

    out = {
        "study": "Study75C_E1_pit_bootstrap",
        "generated_at": datetime.now(_JST).isoformat(),
        "started_at": started.isoformat(),
        "config": {
            "k_draws": K_DRAWS, "seed": SEED, "snapshot_date": SNAPSHOT_DATE,
            "adv_band_jpy": [float(adv_lo), float(adv_hi)],
            "eligible_pool": len(eligible), "eligible_survivors": len(survivors),
            "eligible_eventually_dead": len(eventually_dead),
            "bear_sector_exclusion": "disabled_for_consistency（セクター情報なし・U0'にも同一適用）",
        },
        "u0_jquants_reference_with_sector_filter": U0_JQUANTS_OFFICIAL,
        "u0_prime": u0_prime,
        "summary": summary,
        "draws": draws,
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}")

    for w in ("IS", "OOS"):
        s = summary[w]
        print(f"\n=== {w} ===")
        print(f"PIT dist: median={s['pit_distribution'].get('median')} "
              f"[p5={s['pit_distribution'].get('p5')}, p95={s['pit_distribution'].get('p95')}]")
        print(f"U0'={s['u0_prime_cagr']} (percentile in PIT dist: {s['u0_percentile_in_pit_distribution']}%)")
        print(f"survivorship_bias(paired twin-pit)={s['survivorship_bias_pp_paired']}pp")
        print(f"selection_bias(U0'-twin_median)={s['selection_bias_pp']}pp")
        print(f"combined_bias(U0'-pit_median)={s['combined_bias_pp']}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
