"""
src/backtest/study101_legacy_fujiko_true_expectancy.py
Study101 — Legacy Fujiko True Expectancy Audit（2026-07-16）

正典: ユーザー指示「Study101」（2026-07-16）。Study100 Phase1（Universe監査FATAL）の帰結として、
selection alphaを除去したPITユニバース上で旧フジコ法の真の期待値を測定する。

ユニバース（月次PIT・hindsight選定なし）:
  全上場銘柄（study75_universe_diagnostics.parquet・excluded_reason==""のPIT適格集合）
  → ADV20上位500 → 月末t-1のIBD式12M加重複合リターン上位200 = 当月メンバーシップ。
  月末t時点で利用可能な情報のみ使用（compositeはshift済み・月境界はT-1末値参照）。

戦略: 旧フジコ法完全固定（PARAMS_LOCKED無変更・Study75B run_btと同一のproduction parityフラグ:
  BASELINE/rsr_exit=70/multilayer/ATR trailing prod/exit_policy=A/addon D/min_hold=3/capital=3M）。

バリアント（2x2・事前固定・スイープなし）:
  A: min_rsr=75 パーセンタイル版。RSR文脈=当月ADV500プール内挿入ランク
     （全銘柄の系列が連続 → Study76Dで確定したffill/warm-upアーティファクトを構造的に排除。
      union固定プール型パーセンタイルはStudy75C U3で汚染確認済みのため不使用）。
  B: 絶対スコア版。rsr_abs = clip(50 + 100*composite, 0, 100)。
     entry閾値75 ⇔ 12M加重複合リターン+25% / exit閾値70 ⇔ +20%。
     プールサイズ非依存（統治原則3・仮説4対応）。関数系は本実行前に固定・スイープ禁止。
  フィルタ: none / 25MA乖離率<20%（t-1参照・sym_active_dfへAND注入=新規エントリーのみ遮断）。

出力: IS(2018-2024)/OOS(2025) CAGR・Calmar・Sharpe・MaxDD・Exposure・Trade count・
      Bootstrap CI（月次リターンiid bootstrap 2000回・CAGR 95%CI）・TOPIX比較。
判定: GREEN CAGR>8%∧Calmar>0.6 / YELLOW 5-8% / RED <5%。

出力ファイル:
  backtests/study101_legacy_fujiko_true_expectancy.json
  reports/study101_legacy_fujiko_true_expectancy.md
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

from src.backtest.rsr import calc_composite_return
from src.backtest.study75b_survivorship_bias import (
    IS_END, IS_START, OOS_END, OOS_START, and_masks, build_scenario_dataset,
    run_bt, window_dataset,
)
from src.backtest.study76_datr_eq_universe_c_rebaseline import load_sector_map
from src.paths import JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR

_JST = timezone(timedelta(hours=9))

ADV_TOP_N = 500
RSR_TOP_N = 200
MA_DEV_MAX_PCT = 20.0
MA_DEV_PERIOD = 25
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
ABS_OFFSET, ABS_SCALE = 50.0, 100.0  # rsr_abs = clip(50 + 100*comp, 0, 100)

DIAGNOSTICS_PATH = RESULTS_DIR / "study75_universe_diagnostics.parquet"
OUT_JSON = RESULTS_DIR / "study101_legacy_fujiko_true_expectancy.json"
OUT_MD = REPORTS_DIR / "study101_legacy_fujiko_true_expectancy.md"


# ────────────────────────────────────────────────────────────────────────── #
# 1) ユニバース構築（月次PIT: ADV20 Top500 → composite Top200）
# ────────────────────────────────────────────────────────────────────────── #
def build_adv500_by_month() -> dict[str, list[str]]:
    diag = pd.read_parquet(DIAGNOSTICS_PATH, columns=["date", "code", "adv20", "excluded_reason"])
    diag = diag[diag["excluded_reason"] == ""]
    out: dict[str, list[str]] = {}
    for dt, g in diag.groupby("date"):
        g = g.dropna(subset=["adv20"]).sort_values("adv20", ascending=False)
        out[pd.Timestamp(dt).strftime("%Y-%m-%d")] = g["code"].head(ADV_TOP_N).tolist()
    return out


def load_close_map(codes: list[str]) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for c in codes:
        p = JQUANTS_PROCESSED_DIR / f"{c}.parquet"
        if not p.exists():
            continue
        s = pd.read_parquet(p, columns=["Close"])["Close"].dropna()
        s = s[~s.index.duplicated(keep="last")]
        if len(s) > 260:
            out[c] = s
    return out


def build_membership(adv500: dict[str, list[str]],
                     comp_ctx: dict[str, pd.Series]) -> dict[str, list[str]]:
    """各月T-1末のcomposite上位RSR_TOP_N（当月ADV500内）。build_dynamic_rsr42_membershipと同一時点規約。"""
    result: dict[str, list[str]] = {}
    for month_key in sorted(adv500.keys()):
        month_ts = pd.Timestamp(month_key)
        scores: dict[str, float] = {}
        for sym in adv500[month_key]:
            comp = comp_ctx.get(sym)
            if comp is None:
                continue
            hist = comp.loc[comp.index < month_ts]
            if hist.empty or pd.isna(hist.iloc[-1]):
                continue
            scores[sym] = float(hist.iloc[-1])
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:RSR_TOP_N]
        result[month_key] = [s for s, _ in top]
    return result


# ────────────────────────────────────────────────────────────────────────── #
# 2) RSR行列（バリアントA: 月次ADV500文脈への挿入ランク・全銘柄連続系列）
# ────────────────────────────────────────────────────────────────────────── #
def build_rsr_insertion_rank(
    comp_union_df: pd.DataFrame,     # index=日次カレンダー, columns=union syms（実バーcomposite）
    comp_ctx_df: pd.DataFrame,       # index=同, columns=ADV500-union syms
    adv500: dict[str, list[str]],
) -> pd.DataFrame:
    """
    月mの各日dについて: rsr[d,s] = 当月ADV500メンバーのcomposite分布への挿入位置percentile。
    メンバーは通常のランクと一致し、非メンバーも毎日値を持つ（系列連続=warm-upアーティキャクト排除）。
    """
    months = sorted(adv500.keys())
    cal = comp_union_df.index
    out = np.full(comp_union_df.shape, np.nan)
    un_mat = comp_union_df.to_numpy()

    for i, key in enumerate(months):
        m_start = pd.Timestamp(key)
        m_end = pd.Timestamp(months[i + 1]) if i + 1 < len(months) else cal.max() + pd.Timedelta(days=1)
        day_idx = np.where((cal >= m_start) & (cal < m_end))[0]
        if len(day_idx) == 0:
            continue
        members = [s for s in adv500[key] if s in comp_ctx_df.columns]
        if len(members) < 50:
            continue
        ctx_block = comp_ctx_df[members].to_numpy()
        for di in day_idx:
            ctx_row = ctx_block[di]
            v = np.sort(ctx_row[~np.isnan(ctx_row)])
            if len(v) < 50:
                continue
            row = un_mat[di]
            valid = ~np.isnan(row)
            ranks = np.searchsorted(v, row[valid], side="right") / len(v) * 100.0
            out_row = out[di]
            out_row[valid] = np.clip(ranks, 0.0, 100.0)
    return pd.DataFrame(out, index=cal, columns=comp_union_df.columns)


# ────────────────────────────────────────────────────────────────────────── #
# 3) 活性マスク
# ────────────────────────────────────────────────────────────────────────── #
def build_membership_active(membership: dict[str, list[str]],
                            cal: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    act = pd.DataFrame(0, index=cal, columns=cols, dtype=np.int8)
    months = sorted(membership.keys())
    for i, key in enumerate(months):
        m_start = pd.Timestamp(key)
        m_end = pd.Timestamp(months[i + 1]) if i + 1 < len(months) else cal.max() + pd.Timedelta(days=1)
        mask = (cal >= m_start) & (cal < m_end)
        syms = [s for s in membership[key] if s in act.columns]
        if syms:
            act.loc[mask, syms] = 1
    return act


def build_ma_dev_mask(universe_raw: dict, cal: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    """25MA乖離率 < MA_DEV_MAX_PCT（t-1参照・shift(1)）。乖離不明(初期)は不許可=0。"""
    close_df = pd.DataFrame({s: universe_raw[s]["df"]["Close"] for s in cols if s in universe_raw})
    close_df = close_df.reindex(cal)
    ma = close_df.rolling(MA_DEV_PERIOD, min_periods=MA_DEV_PERIOD).mean()
    dev_pct = (close_df / ma - 1.0) * 100.0
    ok = (dev_pct < MA_DEV_MAX_PCT).shift(1).fillna(False)
    return ok.reindex(columns=cols).fillna(False).astype(np.int8)


# ────────────────────────────────────────────────────────────────────────── #
# 4) メトリクス・bootstrap
# ────────────────────────────────────────────────────────────────────────── #
def bootstrap_cagr_ci(equity: pd.Series, n_boot: int = N_BOOTSTRAP,
                      seed: int = BOOTSTRAP_SEED) -> dict:
    """月次リターンのiid bootstrapによるCAGR 95%CI（自己相関未補正・注記付き参考値）。"""
    eq = equity.dropna()
    if len(eq) < 60:
        return {"ci_lo": None, "ci_hi": None, "n_months": 0}
    monthly = eq.resample("ME").last().pct_change().dropna().to_numpy()
    n = len(monthly)
    if n < 6:
        return {"ci_lo": None, "ci_hi": None, "n_months": n}
    rng = np.random.default_rng(seed)
    cagr_samples = np.empty(n_boot)
    for b in range(n_boot):
        draw = monthly[rng.integers(0, n, n)]
        total = np.prod(1.0 + draw)
        cagr_samples[b] = total ** (12.0 / n) - 1.0
    lo, hi = np.percentile(cagr_samples, [2.5, 97.5])
    return {"ci_lo": round(float(lo) * 100, 2), "ci_hi": round(float(hi) * 100, 2), "n_months": n}


def extract(raw: dict, equity_key: str = "equity_curve") -> dict:
    eq = raw.get(equity_key)
    boot = bootstrap_cagr_ci(eq) if isinstance(eq, pd.Series) else {"ci_lo": None, "ci_hi": None}
    return {
        "cagr": raw.get("cagr"), "sharpe": raw.get("sharpe"), "calmar": raw.get("calmar"),
        "max_dd": raw.get("max_dd"), "n_trades": raw.get("n_trades"),
        "avg_exposure": raw.get("avg_exposure"),
        "bootstrap_cagr_ci95": boot,
    }


def topix_cagr(topix: pd.Series, start: str, end: str) -> float:
    w = topix.loc[start:end].dropna()
    years = (w.index[-1] - w.index[0]).days / 365.25
    return round((float(w.iloc[-1] / w.iloc[0]) ** (1 / years) - 1) * 100, 2)


def decide(is_m: dict, oos_m: dict) -> str:
    """判定はIS/OOSの弱い方（保守側）のCAGRとISのCalmarで機械適用。"""
    cagrs = [v for v in (is_m.get("cagr"), oos_m.get("cagr")) if v is not None]
    if not cagrs:
        return "NO_DATA"
    worst = min(cagrs)
    calmar = is_m.get("calmar") or 0.0
    if worst > 8.0 and calmar > 0.6:
        return "GREEN"
    if worst >= 5.0:
        return "YELLOW"
    return "RED"


# ────────────────────────────────────────────────────────────────────────── #
# 5) メイン
# ────────────────────────────────────────────────────────────────────────── #
def main() -> int:
    t0 = datetime.now(_JST)
    print("Study101 — Legacy Fujiko True Expectancy Audit")

    # ── ユニバース ──
    adv500 = build_adv500_by_month()
    adv_union = sorted({c for v in adv500.values() for c in v})
    print(f"ADV500: months={len(adv500)} union={len(adv_union)}")

    close_ctx = load_close_map(adv_union)
    comp_ctx = {s: calc_composite_return(cl) for s, cl in close_ctx.items()}
    membership = build_membership(adv500, comp_ctx)
    membership = {k: v for k, v in membership.items() if v}
    union = sorted({c for v in membership.values() for c in v})
    n_per_month = [len(v) for v in membership.values()]
    print(f"membership(Top{RSR_TOP_N}): months={len(membership)} union={len(union)} "
          f"median_n={int(np.median(n_per_month))}")

    # ── データセット（Study75B流儀: 上場廃止セマンティクス込み） ──
    sectors = load_sector_map(union)
    ds = build_scenario_dataset(sectors)
    engine_syms = list(ds["trade_syms"].keys())
    print(f"dataset: engine_syms={len(engine_syms)} missing={len(ds['missing'])} "
          f"short_history={len(ds['short_history'])}")
    cal = ds["topix_close"].index

    # ── RSR行列（A: 月次ADV500挿入ランク / B: 絶対スコア） ──
    comp_union_df = pd.DataFrame(
        {s: comp_ctx[s] for s in engine_syms if s in comp_ctx}).reindex(cal)
    comp_ctx_df = pd.DataFrame(comp_ctx).reindex(cal)
    print("RSR行列A（挿入ランク）計算中...")
    rsr_a = build_rsr_insertion_rank(comp_union_df, comp_ctx_df, adv500)
    rsr_b = ((comp_union_df * ABS_SCALE) + ABS_OFFSET).clip(0.0, 100.0)
    print(f"rsr_a valid率={float(rsr_a.notna().mean().mean()):.2f} "
          f"rsr_b valid率={float(rsr_b.notna().mean().mean()):.2f}")

    # ── 活性マスク ──
    base_active = build_membership_active(membership, cal, engine_syms)
    ma_mask = build_ma_dev_mask(ds["universe_raw"], cal, engine_syms)
    active_none = and_masks(base_active, ds["alive_df"])
    active_ma = and_masks(and_masks(base_active, ma_mask), ds["alive_df"])
    print(f"active率: none={float(active_none.mean().mean()):.3f} "
          f"ma20={float(active_ma.mean().mean()):.3f}")

    # ── 4構成 × IS/OOS ──
    configs = {
        "A_pct75_none":  (rsr_a, active_none),
        "A_pct75_ma20":  (rsr_a, active_ma),
        "B_abs_none":    (rsr_b, active_none),
        "B_abs_ma20":    (rsr_b, active_ma),
    }
    windows = {"IS": (IS_START, IS_END), "OOS": (OOS_START, OOS_END)}
    results: dict = {}
    for name, (rsr_df, act) in configs.items():
        ds_v = dict(ds)
        ds_v["rsr_df"] = rsr_df
        results[name] = {}
        for wname, (s, e) in windows.items():
            print(f"[run] {name} {wname} {s}..{e} ...", flush=True)
            dsw = window_dataset(ds_v, s, e)
            raw = run_bt(dsw, act, s, e)
            results[name][wname] = extract(raw)
            m = results[name][wname]
            print(f"      CAGR={m['cagr']} Calmar={m['calmar']} MaxDD={m['max_dd']} "
                  f"trades={m['n_trades']} exp={m['avg_exposure']}")
        results[name]["decision"] = decide(results[name]["IS"], results[name]["OOS"])

    tpx = {w: topix_cagr(ds["topix_close"], *windows[w]) for w in windows}

    payload = {
        "study": "study101_legacy_fujiko_true_expectancy",
        "generated_at": datetime.now(_JST).isoformat(),
        "universe": {
            "method": "monthly PIT: all-JP eligible (diagnostics excluded_reason=='') -> ADV20 top500 -> composite top200 (T-1 EOM)",
            "months": len(membership), "union": len(union),
            "median_members": int(np.median(n_per_month)),
        },
        "strategy": "legacy fujiko unchanged (Study75B run_bt production-parity flags, PARAMS_LOCKED)",
        "variants": {
            "A": "min_rsr=75 percentile, context=monthly ADV500 insertion-rank (continuous series)",
            "B": f"absolute score rsr_abs=clip({ABS_OFFSET}+{ABS_SCALE}*composite,0,100); entry75<=>+25% / exit70<=>+20% 12M weighted return (pre-fixed, no sweep)",
            "filters": ["none", f"25MA deviation < {MA_DEV_MAX_PCT}% (t-1)"],
        },
        "windows": {"IS": list(windows["IS"]), "OOS": list(windows["OOS"])},
        "topix_cagr": tpx,
        "results": results,
        "runtime_min": round((datetime.now(_JST) - t0).total_seconds() / 60, 1),
    }

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=_default),
                        encoding="utf-8")

    # ── Markdown ──
    md = ["# Study101 — Legacy Fujiko True Expectancy Audit（2026-07-16）\n"]
    md.append("selection alpha除去後の旧フジコ法真の期待値。ユニバース=月次PIT（全上場適格→ADV20 Top500→"
              f"composite Top{RSR_TOP_N}）・union={len(union)}銘柄・{len(membership)}ヶ月。戦略=旧フジコ法完全固定"
              "（production parityフラグ・PARAMS_LOCKED無変更）。\n")
    md.append(f"TOPIX比較: IS(2018-2024)={tpx['IS']}% / OOS(2025)={tpx['OOS']}%\n")
    md.append("| 構成 | 窓 | CAGR% | Calmar | Sharpe | MaxDD% | Exposure% | Trades | CAGR 95%CI |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for name, res in results.items():
        for w in ("IS", "OOS"):
            m = res[w]
            ci = m["bootstrap_cagr_ci95"]
            ci_s = f"[{ci['ci_lo']}, {ci['ci_hi']}]" if ci.get("ci_lo") is not None else "-"
            md.append(f"| {name} | {w} | {m['cagr']} | {m['calmar']} | {m['sharpe']} | "
                      f"{m['max_dd']} | {m['avg_exposure']} | {m['n_trades']} | {ci_s} |")
        md.append(f"| {name} | **判定** | **{res['decision']}** | | | | | | |")
    md.append("\n判定規則: GREEN=min(IS,OOS) CAGR>8%∧IS Calmar>0.6 / YELLOW=5-8% / RED=<5%（保守側=弱い方の窓）\n")
    md.append("注記: bootstrap CIは月次リターンiid再抽出（自己相関未補正・参考値）。"
              "バリアントAのRSR文脈は月次ADV500への挿入ランク＝全系列連続で"
              "Study76D warm-upアーティファクト非該当。Bの絶対閾値は実行前固定・スイープなし。\n")
    md.append(f"---\n*生成: Study101, {datetime.now(_JST).strftime('%Y-%m-%d %H:%M')} JST。"
              f"runtime={payload['runtime_min']}min。*")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"出力: {OUT_JSON}")
    print(f"出力: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
