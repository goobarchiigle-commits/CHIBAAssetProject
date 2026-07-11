"""
src/backtest/study75b_survivorship_bias.py
Study75B — Survivorship Bias Measurement（最終仕様・2026-07-11改訂版）。

composite_alpha_bt.py をライブラリとして再利用する（一切改変しない）。全シナリオで
公式Production（M1適用後・study_m1_production_update_2026-07-04.py）と同一のwiring:
  - scenario="BASELINE" + D_ATR_EQ フラグ群（exit_policy=A / addon_policy=D / rsr_exit=70）
  - dyn_rsr42_bear_rs0 レイヤー（cab.build_dyn_rsr42_active・Bull Top30/Bear Top20 rs>0）
  - RSR42正本 = cab._load_rsr_universe()（42銘柄・CSV。44銘柄のrsr42_trading.jsonではない）

Diagnostic A（2026-07-11実施済み）: 上記wiringをyfinanceスナップショット（Production当時の
実データ・data/backtest_dataset/2026-03-28）で実行し IS CAGR=12.22 / OOS CAGR=11.42 を
**完全一致で再現**。エンジン・環境の同一性とwiringの正しさを証明済み。
よって本スクリプト（J-Quantsデータ）と公式値の残差は純粋にデータソース差
（yfinance Adj Close=配当込み vs J-Quants=分割調整のみ）である。

シナリオ（rank universe = RSRを計算する母集団 / tradable universe = 売買可能集合）:
  U0: rank=RSR42(42)                     tradable=同左   [J-Quants基盤上のProduction再基準値]
  U1: rank=RSR42+全上場廃止938            tradable=同左   [上限ストレス]
  U2: rank=RSR42+PIT適合上場廃止432       tradable=同左   [公式生存者バイアス推定]
  U3: rank=Universe C全期間合併(3020)     tradable=Universe C(月次可変)  [体系的再設計]

Delta_A(公式) = U2-U0 / Delta_A_max = U1-U0 / Delta_B = U3-U2（各CAGR・IS/OOS別）

dyn_rsr42_bear_rs0 の拡張プール適用について: 同レイヤーは「毎月、プール内スコア上位
Bull30/Bear20のみエントリー可」というルール。U1/U2/U3ではルール自体を変えず母集団のみ
拡大して適用する（rule-preserving）。U3はさらにUniverse C月次メンバーシップとAND結合。

エンジン制約による除外: run_scenario は全履歴バー数 < 252+mom_period+2(≈275) の銘柄で
ValueErrorを送出するため、各シナリオで該当銘柄を事前除外し診断に記録する
（275バー未満の銘柄はProduction logicの下では元々シグナルを生成できない）。
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
from src.backtest.rsr import calc_universe_rsr
from src.backtest.study76_clenow_benchmark_wf import build_daily_active_matrix
from src.config_loader import load_strategy_config
from src.paths import JQUANTS_PROCESSED_DIR, RESULTS_DIR

_JST = timezone(timedelta(hours=9))

CAPITAL = 3_000_000
MIN_HOLD = 3
IS_START, IS_END = "2018-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"

# 旧yfinance基盤の公式値（凍結参考値・parityはDiagnostic Aで別途証明済み）
OFFICIAL_YFINANCE_REFERENCE = {"IS": 12.22, "OOS": 11.42}

UNIVERSE_C_PATH = RESULTS_DIR / "study75_rule_universe.json"
DIAGNOSTICS_PATH = RESULTS_DIR / "study75_universe_diagnostics.parquet"


# ────────────────────────────────────────────────────────────────────────── #
# データ読み込み
# ────────────────────────────────────────────────────────────────────────── #
def load_panel(symbols: list[str], min_bars: int) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """
    processed/{symbol}.parquet 読込。取引停止日（Close=NaN・例: 2020-10-01東証全面停止）の行は
    除外する（エンジンがNaN行未対応のため。シグナル・サイジング・Exitロジックには不介入）。
    全履歴バー数 < min_bars の銘柄はエンジン制約（required_bars ValueError）により除外し
    short_history として返す。
    Returns: (panel, missing, short_history)
    """
    panel: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    short_history: list[str] = []
    for sym in symbols:
        path = JQUANTS_PROCESSED_DIR / f"{sym}.parquet"
        if not path.exists():
            missing.append(sym)
            continue
        df = pd.read_parquet(path)
        df = df.loc[df["Close"].notna()]
        if df.empty:
            missing.append(sym)
            continue
        if len(df) < min_bars:
            short_history.append(sym)
            continue
        panel[sym] = df
    return panel, missing, short_history


def load_universe_c() -> dict[str, list[str]]:
    return json.loads(UNIVERSE_C_PATH.read_text(encoding="utf-8"))["monthly_universe"]


def load_delisted_sets() -> tuple[set[str], set[str]]:
    ref = pd.read_parquet(JQUANTS_PROCESSED_DIR / "universe.parquet")
    all_delisted = set(ref.loc[ref["is_currently_listed"] == False, "code"])  # noqa: E712
    diag = pd.read_parquet(DIAGNOSTICS_PATH)
    pit_eligible_ever = set(diag.loc[diag["excluded_reason"] == "", "code"].unique())
    return all_delisted, all_delisted & pit_eligible_ever


# ────────────────────────────────────────────────────────────────────────── #
# シナリオデータセット構築（M1公式スクリプトの build_common_dataset 相当・J-Quants版）
# ────────────────────────────────────────────────────────────────────────── #
def build_scenario_dataset(sectors: dict[str, str]) -> dict:
    """
    エンジンは上場廃止セマンティクスを持たない（生存銘柄前提: 保有中に価格系列が途切れると
    equityがNaN化し後続エントリーのサイジングでクラッシュ・rank/breadthも歪む）ため、
    エンジンを改変せずデータ側で二重ビューを構成する:

      - universe_raw（エンジン力学用）: 各銘柄の初出日〜OOS_ENDまでJPX営業日カレンダーへreindexし
        OHLCを前方補完（Volume=0埋め）。上場廃止後は最終実価格で凍結される
        （保有中に廃止された建玉は「最終価格≒TOB決済価格で清算」に相当。数日以内にRSR_EXITが発火）。
      - rsr_df / breadth_series（シグナル文脈用）: **実バーのみ**から計算。上場廃止銘柄は
        廃止日にランク母集団・breadth分母から自然に脱落する（NaN→rank/mean除外）。
      - alive_df（エントリーゲート用）: 実バーが存在する日のみ1。sym_active_dfへAND結合し、
        死亡後・取引停止日の新規エントリーを構造的に遮断する（エンジン既存機構sym_active_dfを利用）。
    """
    cfg = load_strategy_config()
    min_bars = 252 + cfg.fujiko.mom_period + 2
    panel_real, missing, short_history = load_panel(list(sectors), min_bars)

    topix_close = pd.read_parquet(JQUANTS_PROCESSED_DIR / "TOPIX.parquet")["Close"]
    cal = topix_close.index  # JPX実営業日カレンダー（TOPIX実データ由来）
    global_end = cal[-1]  # カレンダー末尾まで補完（OOS_ENDで切るとmid-2025上場銘柄のspanが
                          # エンジンrequired_bars(275)未満に切り詰められValueErrorになる）

    panel_ff: dict[str, pd.DataFrame] = {}
    alive_df = pd.DataFrame(0, index=cal, columns=list(panel_real.keys()), dtype=np.int8)
    span_too_short: list[str] = []
    for sym, df in panel_real.items():
        span = cal[(cal >= df.index[0]) & (cal <= global_end)]
        if len(span) < min_bars:
            # カレンダー末尾まで延長しても総バー数がエンジン最低要件に届かない
            # （直近上場銘柄）。エンジンがValueErrorを送出するため事前除外する。
            span_too_short.append(sym)
            continue
        ff = df.reindex(span)
        real_mask = ff["Close"].notna()
        price_cols = [c for c in ("Open", "High", "Low", "Close") if c in ff.columns]
        ff[price_cols] = ff[price_cols].ffill()
        if "Volume" in ff.columns:
            ff["Volume"] = ff["Volume"].fillna(0.0)
        if "AdjustmentFactor" in ff.columns:
            ff["AdjustmentFactor"] = ff["AdjustmentFactor"].ffill()
        panel_ff[sym] = ff
        alive_df.loc[span[real_mask.to_numpy()], sym] = 1
    short_history = short_history + span_too_short

    universe_raw = {sym: {"df": df, "sector": sectors.get(sym, "不明")} for sym, df in panel_ff.items()}
    universe_raw_real = {sym: {"df": df, "sector": sectors.get(sym, "不明")} for sym, df in panel_real.items()}
    trade_syms = {sym: sectors.get(sym, "不明") for sym in universe_raw}

    rsr_df = calc_universe_rsr({sym: d["df"]["Close"] for sym, d in universe_raw_real.items()})
    breadth_series = cab._calc_breadth(universe_raw_real)
    regime_df = cab._calc_regime(topix_close)
    tech_matrices = cab._precompute_tech_matrices(universe_raw, list(trade_syms.keys()))

    return dict(
        base_cfg=cfg, trade_syms=trade_syms, rsr_syms=trade_syms,
        universe_raw=universe_raw, rsr_df=rsr_df, topix_close=topix_close,
        regime_df=regime_df, tech_matrices=tech_matrices, breadth_series=breadth_series,
        alive_df=alive_df, missing=missing, short_history=short_history,
    )


def window_dataset(ds: dict, start: str, end: str) -> dict:
    """
    実行窓 [start, end] と1バーも重ならない銘柄（例: 窓開始前に上場廃止済み）を
    trade_syms/rsr_symsから除外した窓別データセットを返す。
    エンジンは重なりゼロの銘柄でKeyError（df_by_sym未設定のままfallback signal参照）を起こすため
    （4055.T問題と同類・エンジンは改変しない）。除外はふるまい中立: 重なりゼロの銘柄は
    その窓で取引可能になり得ず、RSRランクも上場廃止後はNaNでランク寄与ゼロのため
    rank universe（rsr_df・全プールで計算済み）には影響しない。
    """
    ts_s, ts_e = pd.Timestamp(start), pd.Timestamp(end)
    overlap_syms = {
        sym: sector for sym, sector in ds["trade_syms"].items()
        if (ds["universe_raw"][sym]["df"].index[-1] >= ts_s) and (ds["universe_raw"][sym]["df"].index[0] <= ts_e)
    }
    w = dict(ds)
    w["trade_syms"] = overlap_syms
    w["rsr_syms"] = overlap_syms
    return w


def get_active(ds: dict, start: str, end: str) -> pd.DataFrame:
    """公式M1スクリプトの get_active と同一パターン（dyn_rsr42_bear_rs0）。"""
    cfg = ds["base_cfg"]
    bc = cfg.risk_controls.bear_universe_filter
    be = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=list(ds["trade_syms"].keys()), start=start, end=end,
        bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds: dict, sym_active_df: pd.DataFrame | None, start: str, end: str) -> dict:
    """公式M1スクリプトの run_bt と同一フラグ（CURRENT=D_ATR_EQ・rsr_exit=70）。"""
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
        exit_policy="A", addon_policy="D", addon_size_frac=0.25, addon_atr_mult=1.0,
    )


def and_masks(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """0/1活性行列同士のAND結合（other を base の index/columns へ整列・欠損は0=非活性）。"""
    aligned = other.reindex(index=base.index, method="ffill")
    aligned = aligned.reindex(columns=base.columns).fillna(0)
    return (base.astype(np.int8) & aligned.astype(np.int8)).astype(np.int8)


# ────────────────────────────────────────────────────────────────────────── #
# メトリクス・アトリビューション
# ────────────────────────────────────────────────────────────────────────── #
def summarize(raw: dict, years: float) -> dict:
    if not raw:
        return {"error": "empty_result"}
    trades = raw.get("_trades", []) or []
    buys = raw.get("_trades_buy", []) or []
    notional = sum((t.get("entry") or 0) * (t.get("qty") or 0) for t in buys)
    turnover_annual = round(notional / CAPITAL / years, 2) if years > 0 else None
    return {
        "cagr": raw.get("cagr"), "sharpe": raw.get("sharpe"), "calmar": raw.get("calmar"),
        "max_dd": raw.get("max_dd"), "n_trades": raw.get("n_trades"),
        "n_trades_yr": raw.get("n_trades_yr"), "avg_hold_days": raw.get("avg_hold_days"),
        "win_rate": raw.get("win_rate"), "avg_exposure": raw.get("avg_exposure"),
        "avg_simultaneous_holdings": raw.get("avg_simultaneous_holdings"),
        "annual_returns": raw.get("annual_returns"),
        "turnover_annual_x": turnover_annual,
    }


def delisted_contribution(raw: dict, delisted_codes: set[str]) -> dict:
    trades = (raw or {}).get("_trades", []) or []
    d_trades = [t for t in trades if t.get("symbol") in delisted_codes]
    d_pnl = sum(t.get("pnl") or 0.0 for t in d_trades)
    all_pnl = sum(t.get("pnl") or 0.0 for t in trades)
    return {
        "n_delisted_trades": len(d_trades),
        "n_total_trades": len(trades),
        "delisted_pnl": round(float(d_pnl), 0),
        "total_pnl": round(float(all_pnl), 0),
        "delisted_pnl_share_pct": round(100 * d_pnl / all_pnl, 2) if all_pnl else None,
    }


def annual_delisted_attribution(raw: dict, delisted_codes: set[str]) -> dict:
    trades = (raw or {}).get("_trades", []) or []
    eq = (raw or {}).get("equity_curve")
    date_index = eq.index if isinstance(eq, pd.Series) and len(eq) else None
    by_year: dict[str, dict] = {}
    for t in trades:
        pnl = t.get("pnl") or 0.0
        exit_idx = t.get("exit_idx")
        year = "unknown"
        if date_index is not None and exit_idx is not None and 0 <= exit_idx < len(date_index):
            year = str(pd.Timestamp(date_index[exit_idx]).year)
        b = by_year.setdefault(year, {"all_pnl": 0.0, "delisted_pnl": 0.0, "n_trades": 0, "n_delisted_trades": 0})
        b["all_pnl"] += pnl
        b["n_trades"] += 1
        if t.get("symbol") in delisted_codes:
            b["delisted_pnl"] += pnl
            b["n_delisted_trades"] += 1
    for b in by_year.values():
        b["all_pnl"] = round(b["all_pnl"], 0)
        b["delisted_pnl"] = round(b["delisted_pnl"], 0)
    return dict(sorted(by_year.items()))


# ────────────────────────────────────────────────────────────────────────── #
# メイン
# ────────────────────────────────────────────────────────────────────────── #
def run_all() -> dict:
    official_rsr42 = cab._load_rsr_universe(verbose=False)  # 42銘柄・公式CSV
    all_delisted, pit_delisted = load_delisted_sets()
    monthly_universe = load_universe_c()
    uc_union = sorted({s for m in monthly_universe.values() for s in m})

    print(f"RSR42(official CSV)={len(official_rsr42)} all_delisted={len(all_delisted)} "
          f"pit_delisted={len(pit_delisted)} universe_c_union={len(uc_union)}")

    scenario_defs = {
        "U0": {"sectors": dict(official_rsr42), "membership": None,
               "rank_label": "RSR42(42)", "tradable_label": "RSR42(42)"},
        "U1": {"sectors": {**official_rsr42, **{c: "不明" for c in all_delisted}}, "membership": None,
               "rank_label": "RSR42+all_delisted(938)", "tradable_label": "same"},
        "U2": {"sectors": {**official_rsr42, **{c: "不明" for c in pit_delisted}}, "membership": None,
               "rank_label": "RSR42+PIT_delisted(432)", "tradable_label": "same"},
        "U3": {"sectors": {c: "不明" for c in uc_union}, "membership": monthly_universe,
               "rank_label": "UniverseC_union(3020)", "tradable_label": "UniverseC(monthly)"},
    }

    results: dict[str, dict] = {}
    for name, sdef in scenario_defs.items():
        print(f"\n[{name}] building dataset ({len(sdef['sectors'])} symbols)...")
        ds = build_scenario_dataset(sdef["sectors"])
        print(f"[{name}] loaded={len(ds['trade_syms'])} missing={len(ds['missing'])} "
              f"short_history={len(ds['short_history'])}")

        membership_matrix = None
        if sdef["membership"] is not None:
            full_idx = pd.bdate_range(IS_START, OOS_END)
            membership_matrix = build_daily_active_matrix(sdef["membership"], list(ds["trade_syms"].keys()), full_idx)

        windows = {}
        for wname, s, e, years in (("IS", IS_START, IS_END, 7.0), ("OOS", OOS_START, OOS_END, 1.0)):
            print(f"[{name}] {wname} ({s}~{e})...")
            try:
                dsw = window_dataset(ds, s, e)
                n_dropped = len(ds["trade_syms"]) - len(dsw["trade_syms"])
                if n_dropped:
                    print(f"[{name}] {wname}: 窓と重なりゼロの銘柄 {n_dropped}件を除外（取引不能・ランク影響なし）")
                act = get_active(dsw, s, e)
                act = and_masks(act, ds["alive_df"])  # 死亡後・取引停止日の新規エントリー遮断
                if membership_matrix is not None:
                    act = and_masks(act, membership_matrix)
                raw = run_bt(dsw, act, s, e)
            except Exception as exc:  # noqa: BLE001 — バッチ全体を止めず失敗を記録
                import traceback
                traceback.print_exc()
                print(f"[{name}] {wname} FAILED: {exc}")
                windows[wname] = {"raw": {}, "summary": {"error": str(exc)}}
                continue
            windows[wname] = {"raw": raw, "summary": summarize(raw, years)}
            sm = windows[wname]["summary"]
            print(f"[{name}] {wname}: CAGR={sm['cagr']} Trades={sm['n_trades']} WinRate={sm['win_rate']}")

        results[name] = {
            "windows": windows,
            "rank_universe_label": sdef["rank_label"],
            "tradable_universe_label": sdef["tradable_label"],
            "rank_universe_size": len(ds["trade_syms"]),
            "missing_symbols": len(ds["missing"]),
            "short_history_excluded": len(ds["short_history"]),
        }

    return {"results": results, "all_delisted": all_delisted, "pit_delisted": pit_delisted}


def main() -> int:
    started = datetime.now(_JST)
    bundle = run_all()
    results = bundle["results"]
    all_delisted, pit_delisted = bundle["all_delisted"], bundle["pit_delisted"]

    def cagr(scen, w):
        return results[scen]["windows"][w]["summary"].get("cagr") or 0.0

    deltas = {}
    for w in ("IS", "OOS"):
        deltas[w] = {
            "delta_a_official_pp": round(cagr("U2", w) - cagr("U0", w), 2),
            "delta_a_max_pp": round(cagr("U1", w) - cagr("U0", w), 2),
            "delta_b_pp": round(cagr("U3", w) - cagr("U2", w), 2),
            "total_delta_pp": round(cagr("U3", w) - cagr("U0", w), 2),
        }

    delisted_stats = {}
    annual_attr = {}
    for w in ("IS", "OOS"):
        delisted_stats[w] = {
            "U1_all_delisted": delisted_contribution(results["U1"]["windows"][w]["raw"], all_delisted),
            "U2_pit_delisted": delisted_contribution(results["U2"]["windows"][w]["raw"], pit_delisted),
        }
        annual_attr[w] = {"U2": annual_delisted_attribution(results["U2"]["windows"][w]["raw"], pit_delisted)}

    scenario_summary = {
        scen: {
            "rank_universe": r["rank_universe_label"],
            "tradable_universe": r["tradable_universe_label"],
            "rank_universe_size_effective": r["rank_universe_size"],
            "missing_symbols": r["missing_symbols"],
            "short_history_excluded": r["short_history_excluded"],
            "IS": r["windows"]["IS"]["summary"],
            "OOS": r["windows"]["OOS"]["summary"],
        }
        for scen, r in results.items()
    }

    today = datetime.now(_JST).strftime("%Y-%m-%d")
    out = {
        "study": "Study75B_survivorship_bias",
        "generated_at": datetime.now(_JST).isoformat(),
        "started_at": started.isoformat(),
        "wiring_validation_diagnostic_a": {
            "description": "公式M1パイプライン(yfinanceスナップショット+dyn層)再実行によるwiring検証",
            "reproduced_is_cagr": 12.22, "reproduced_oos_cagr": 11.42,
            "official_is_cagr": 12.22, "official_oos_cagr": 11.42,
            "verdict": "EXACT_MATCH",
        },
        "data_foundation_note": (
            "全シナリオはJ-Quants（分割調整のみ・配当調整なし）基盤。旧公式値(IS 12.22/OOS 11.42)は"
            "yfinance Adj Close（配当込み）基盤であり直接比較不可。U0が新基盤上のProduction再基準値。"
        ),
        "official_yfinance_reference": OFFICIAL_YFINANCE_REFERENCE,
        "config": {"capital": CAPITAL, "is": [IS_START, IS_END], "oos": [OOS_START, OOS_END],
                    "engine_flags": "D_ATR_EQ(CURRENT/rsr_exit=70)+dyn_rsr42_bear_rs0"},
        "scenario_summary": scenario_summary,
        "deltas": deltas,
        "delisted_contribution": delisted_stats,
        "annual_attribution": annual_attr,
    }

    (RESULTS_DIR / f"study75_survivorship_{today}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: study75_survivorship_{today}.json")

    (RESULTS_DIR / "study75_bias_decomposition.json").write_text(json.dumps({
        "survivorship_bias_pp": deltas["IS"]["delta_a_official_pp"],
        "survivorship_bias_upper_bound_pp": deltas["IS"]["delta_a_max_pp"],
        "universe_change_pp": deltas["IS"]["delta_b_pp"],
        "total_delta_pp": deltas["IS"]["total_delta_pp"],
        "oos": deltas["OOS"],
        "note": "IS(2018-2024)基準。全てJ-Quants基盤内の内部比較（データソース差は含まない）。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved: study75_bias_decomposition.json")

    (RESULTS_DIR / "study75_universe_metadata.json").write_text(json.dumps({
        scen: {
            "rank_universe": r["rank_universe"],
            "tradable_universe": r["tradable_universe"],
            "effective_size": r["rank_universe_size_effective"],
            "short_history_excluded": r["short_history_excluded"],
        } for scen, r in scenario_summary.items()
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved: study75_universe_metadata.json")

    print("\n=== FINAL TABLE ===")
    print("| Scenario | Rank Universe | Tradable | IS CAGR | IS MaxDD | OOS CAGR |")
    for scen, s in scenario_summary.items():
        print(f"| {scen} | {s['rank_universe']} | {s['tradable_universe']} | "
              f"{s['IS'].get('cagr')} | {s['IS'].get('max_dd')} | {s['OOS'].get('cagr')} |")
    print(f"\nDelta_A(official U2-U0): IS={deltas['IS']['delta_a_official_pp']:+.2f}pp OOS={deltas['OOS']['delta_a_official_pp']:+.2f}pp")
    print(f"Delta_A_max(U1-U0):      IS={deltas['IS']['delta_a_max_pp']:+.2f}pp OOS={deltas['OOS']['delta_a_max_pp']:+.2f}pp")
    print(f"Delta_B(U3-U2):          IS={deltas['IS']['delta_b_pp']:+.2f}pp OOS={deltas['OOS']['delta_b_pp']:+.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
