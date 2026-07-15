"""
src/scripts/study75c_e1_validity_audit.py
Study75C E1 妥当性監査 — E1ブートストラップ・ユニバースはRSR42と真に比較可能か。

方針: バックテストエンジン(composite_alpha_bt.py)は一切呼ばない（ユーザー指示: "Do NOT rerun
backtests initially"）。E1のRNGドロー（乱数シード固定・study75c_e1_bootstrap.py と同一ロジック）
のみを再現し、42銘柄構成（ticker一覧）を復元する。これはE1のJSON成果物に保存されていない
（集計統計のみ保存）ため、監査に必須の前提再構築である。P&Lシミュレーションは一切行わない。

計算する6指標（ユニバース構成の経済的比較可能性）:
  1. 時価総額プロキシ（真の時価総額データなし・ScaleCategory/last_close/ADV20で代替）
  2. ADV20（2017-12-29スナップショット・E1と同一ウィンドウ）
  3. セクター構成（Sector33・universe_events ADD時点ラベルでPIT整合）
  4. ボラティリティ（trailing 90営業日・年率化realized vol）
  5. モメンタム/トレンド（Clenow流 slope90d×R²・log価格OLS）
  6. 出来高代金特性（ADV20水準 + 変動係数CV）

統計: Mann-Whitney U（銘柄プールレベル）・percentile位置（ユニバースレベル、E1と同一手法）・
Clopper-Pearson exact CI（K=20の順位統計の不確実性）・bootstrap-of-bootstrap（中央値/percentile
位置のCI）・Spearman相関（ドロー特性とドローCAGRの関係、探索的）。
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
from scipy import stats

import src.backtest.composite_alpha_bt as cab
import src.backtest.study75c_e1_bootstrap as e1
from src.paths import CONFIGS_DIR, DATABASE_MASTER_DIR, JQUANTS_PROCESSED_DIR, RESULTS_DIR

_JST = timezone(timedelta(hours=9))
SNAPSHOT_DATE = e1.SNAPSHOT_DATE  # "2017-12-29"
MOMENTUM_WINDOW = 90  # 営業日
VOL_WINDOW = 90

# E1本体JSON（backtests/study75c_e1_bootstrap_2026-07-11.json）から転記した固定値。
# 監査対象そのもの（再計算しない・再現性チェックはdraw構成の再現のみで行う）。
PIT_CAGR_IS = [5.91, -2.95, 1.72, -1.63, -5.53, -2.82, -4.06, -8.76, -3.90, 0.97,
               1.56, 8.98, 3.93, -2.14, 4.40, -11.83, -4.46, -7.34, -12.64, -1.84]
TWIN_MEDIAN_IS = -3.44
U0_PRIME_CAGR_IS = 8.82
SELECTION_BIAS_PP = 12.26
COMBINED_BIAS_PP = 11.30


# ────────────────────────────────────────────────────────────────────────── #
# 1) E1ドロー構成の再現（RNGのみ・バックテストエンジン呼び出しなし）
# ────────────────────────────────────────────────────────────────────────── #
def reproduce_draws() -> dict:
    """study75c_e1_bootstrap.main() のRNGサンプリング部のみを再現する（run_universe呼び出しなし）。"""
    rng = np.random.default_rng(e1.SEED)
    official = cab._load_rsr_universe(verbose=False)
    rsr42_syms = list(official.keys())
    pool = e1.build_eligible_pool()

    rsr42_codes = [s.split(".")[0] + "0" for s in rsr42_syms]
    rsr42_in_pool = pool.loc[pool.index.isin(rsr42_codes)]
    adv_lo, adv_hi = rsr42_in_pool["adv20"].min(), rsr42_in_pool["adv20"].max()
    eligible = pool.loc[(pool["adv20"] >= adv_lo) & (pool["adv20"] <= adv_hi)].copy()
    survivors = eligible.loc[eligible["is_currently_listed"] == True]  # noqa: E712
    eventually_dead = eligible.loc[eligible["is_currently_listed"] == False]  # noqa: E712

    eligible_codes = eligible.index.to_list()
    survivor_by_adv = survivors.sort_values("adv20")
    draws = []
    for k in range(e1.K_DRAWS):
        draw_codes = list(rng.choice(eligible_codes, size=42, replace=False))
        dead_in_draw = [c for c in draw_codes if c in eventually_dead.index]
        twin_codes = list(draw_codes)
        used = set(twin_codes)
        for dc in dead_in_draw:
            target_adv = eligible.loc[dc, "adv20"]
            cands = survivor_by_adv.loc[~survivor_by_adv.index.isin(used)]
            nearest = (cands["adv20"] - target_adv).abs().idxmin()
            twin_codes[twin_codes.index(dc)] = nearest
            used.add(nearest)
        draws.append({"k": k, "draw_codes": draw_codes, "twin_codes": twin_codes, "dead_codes": dead_in_draw})

    # 再現性チェック: dead_codes件数がE1本体JSONのn_deadと一致するはず（構成監査の第一関門）。
    reproduced_n_dead = [len(d["dead_codes"]) for d in draws]
    return {
        "rsr42_syms": rsr42_syms, "rsr42_codes": rsr42_codes,
        "adv_band": (float(adv_lo), float(adv_hi)),
        "eligible": eligible, "survivors": survivors, "eventually_dead": eventually_dead,
        "draws": draws, "reproduced_n_dead": reproduced_n_dead,
    }


def verify_reproduction_against_json() -> dict:
    """再現したdead_codes件数を、実際のE1出力JSONのn_deadと突き合わせる（改ざん・非決定性の検出）。"""
    json_path = RESULTS_DIR / "study75c_e1_bootstrap_2026-07-11.json"
    original = json.loads(json_path.read_text(encoding="utf-8"))
    original_n_dead = [d["n_dead"] for d in original["draws"]]
    original_dead_codes = [sorted(d["dead_codes"]) for d in original["draws"]]
    return {"original_n_dead": original_n_dead, "original_dead_codes": original_dead_codes}


# ────────────────────────────────────────────────────────────────────────── #
# 2) RSR42定義（凍結バックテストCSV・現行ライブJSON）
# ────────────────────────────────────────────────────────────────────────── #
def load_rsr42_frozen() -> pd.DataFrame:
    df = pd.read_csv(CONFIGS_DIR / "rsr_universe_42.csv")
    df["code"] = df["symbol"].str.split(".").str[0] + "0"
    return df[["symbol", "code", "sector"]]


def load_rsr42_live() -> list[str]:
    data = json.loads((CONFIGS_DIR / "universe" / "rsr42_trading.json").read_text(encoding="utf-8"))
    syms = data["symbols"]
    if isinstance(syms, dict):
        syms = list(syms.keys())
    return [s.split(".")[0] + "0" for s in syms]


# ────────────────────────────────────────────────────────────────────────── #
# 3) 銘柄別特性量（2017-12-29スナップショット時点・PIT整合）
# ────────────────────────────────────────────────────────────────────────── #
def compute_stock_metrics(codes: list[str]) -> pd.DataFrame:
    """
    codes（重複除く）について、snapshot時点のADV20・ボラ・モメンタム・出来高CVを計算する。
    ルックアヘッド防止: 全ウィンドウは Date <= snapshot に厳密限定。
    """
    codes = sorted(set(codes))
    df = pd.read_parquet(JQUANTS_PROCESSED_DIR / "daily_bars_2017.parquet",
                          columns=["Date", "Code", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    snap = pd.Timestamp(SNAPSHOT_DATE)
    df = df.loc[(df["Code"].isin(codes)) & (df["Date"] <= snap)].sort_values(["Code", "Date"])

    rows = []
    for code, g in df.groupby("Code"):
        g = g.tail(max(MOMENTUM_WINDOW, VOL_WINDOW) + 5)
        if len(g) < 20:  # データ不足銘柄はスキップ（適格プール条件で既にn_days>=15保証されているはず）
            continue
        close = g["Close"].to_numpy()
        vol = g["Volume"].to_numpy()
        traded_value = close * vol

        adv20 = float(traded_value[-20:].mean())
        turnover_cv = float(np.std(traded_value[-VOL_WINDOW:]) / max(1.0, np.mean(traded_value[-VOL_WINDOW:])))

        ret = np.diff(np.log(np.clip(close[-VOL_WINDOW:], 1e-6, None)))
        vol90d_annualized = float(np.std(ret) * np.sqrt(252)) if len(ret) >= 10 else np.nan

        mom_close = close[-MOMENTUM_WINDOW:]
        if len(mom_close) >= 30:
            x = np.arange(len(mom_close))
            y = np.log(np.clip(mom_close, 1e-6, None))
            slope, intercept = np.polyfit(x, y, 1)
            fitted = slope * x + intercept
            ss_res = np.sum((y - fitted) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            momentum_score = float(slope * 252 * r2)  # 年率化slope×R²（Clenow流）
        else:
            momentum_score = np.nan

        rows.append({
            "code": code, "last_close": float(close[-1]), "adv20": adv20,
            "turnover_cv": turnover_cv, "vol90d_annualized": vol90d_annualized,
            "momentum_score": momentum_score,
        })
    return pd.DataFrame(rows).set_index("code")


def load_sector33_pit_safe(codes: list[str]) -> pd.Series:
    """
    セクター源: universe_events.parquet のADD時点sector_33_nameは大半が空欄（enrichment未実行の
    レガシー・実測確認済み）のため使えない。代わりに database/market/master/companies.parquet
    （現在の分類スナップショット・2026-07時点）のSector33CodeNameを使う。
    制限（明示）: 2017年時点の真のPIT分類ではなく現在分類の遡及適用（セクター再分類は稀なため近似
    として妥当だが、上場廃止銘柄（現在分類が存在しない）は分類不能となる — 本監査ではその銘柄数を
    別途報告し、重み計算から除外する）。
    """
    companies = pd.read_parquet(DATABASE_MASTER_DIR / "companies.parquet", columns=["Code", "Sector33CodeName"])
    sector_map = companies.dropna(subset=["Sector33CodeName"]).set_index("Code")["Sector33CodeName"]
    sector_map = sector_map[sector_map != ""]
    sector = pd.Series(codes, index=codes).map(sector_map)
    return sector


# ────────────────────────────────────────────────────────────────────────── #
# 4) ユニバース要約 + 統計検定
# ────────────────────────────────────────────────────────────────────────── #
def universe_summary(codes: list[str], metrics: pd.DataFrame, sectors: pd.Series) -> dict:
    sub = metrics.reindex(codes).dropna(how="all")
    sec = sectors.reindex(codes)
    n_sector_missing = int(sec.isna().sum())
    sec_known = sec.dropna()
    sec_weights = (sec_known.value_counts(normalize=True)).to_dict() if len(sec_known) else {}
    hhi = float(sum(w ** 2 for w in sec_weights.values())) if sec_weights else float("nan")
    out = {"n": len(sub), "n_sector_missing": n_sector_missing, "n_sector_known": len(sec_known),
           "sector_weights": sec_weights, "sector_hhi": hhi}
    for col in ("last_close", "adv20", "turnover_cv", "vol90d_annualized", "momentum_score"):
        vals = sub[col].dropna()
        out[col] = {"median": float(vals.median()), "mean": float(vals.mean())} if len(vals) else {}
    return out


def percentile_position(rsr42_value: float, draw_values: list[float]) -> float:
    arr = np.array(draw_values)
    return float((arr < rsr42_value).mean() * 100)


def clopper_pearson_ci(k_exceeded: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """RSR42がn本中k_exceeded本を上回った、という二項比率のexact CI。"""
    lo = stats.beta.ppf(alpha / 2, k_exceeded, n - k_exceeded + 1) if k_exceeded > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k_exceeded + 1, n - k_exceeded) if k_exceeded < n else 1.0
    return float(lo * 100), float(hi * 100)


def bootstrap_of_bootstrap(draw_values: list[float], rsr42_value: float | None, n_resamples: int = 10000,
                            seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    arr = np.array(draw_values)
    medians = np.empty(n_resamples)
    pctiles = np.empty(n_resamples) if rsr42_value is not None else None
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        medians[i] = np.median(sample)
        if rsr42_value is not None:
            pctiles[i] = (sample < rsr42_value).mean() * 100
    out = {
        "median_point": float(np.median(arr)),
        "median_ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
    }
    if pctiles is not None:
        out["percentile_position_ci95"] = [float(np.percentile(pctiles, 2.5)), float(np.percentile(pctiles, 97.5))]
    return out


def mann_whitney(rsr42_vals: pd.Series, pooled_draw_vals: pd.Series) -> dict:
    a, b = rsr42_vals.dropna().to_numpy(), pooled_draw_vals.dropna().to_numpy()
    if len(a) < 3 or len(b) < 3:
        return {}
    u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)  # 効果量
    return {"u_stat": float(u_stat), "p_value": float(p), "rank_biserial_effect": float(rank_biserial),
            "rsr42_median": float(np.median(a)), "pooled_pit_median": float(np.median(b)),
            "n_rsr42": n1, "n_pooled": n2}


# ────────────────────────────────────────────────────────────────────────── #
# main
# ────────────────────────────────────────────────────────────────────────── #
def main() -> int:
    started = datetime.now(_JST)
    print("[1/6] E1ドロー構成を再現中（RNGのみ・バックテストなし）...")
    rep = reproduce_draws()
    verify = verify_reproduction_against_json()
    match_check = rep["reproduced_n_dead"] == [len(d) for d in verify["original_dead_codes"]]
    reproduced_dead_sorted = [sorted(d["dead_codes"]) for d in rep["draws"]]
    exact_match = reproduced_dead_sorted == verify["original_dead_codes"]
    print(f"  n_dead一致: {match_check} / dead_codes完全一致: {exact_match}")

    print("[2/6] RSR42定義を読込中（凍結CSV・ライブJSON）...")
    rsr42_frozen = load_rsr42_frozen()
    rsr42_live_codes = load_rsr42_live()
    overlap = set(rsr42_frozen["code"]) & set(rsr42_live_codes)
    print(f"  frozen(42) vs live({len(rsr42_live_codes)}): overlap={len(overlap)}")

    print("[3/6] 銘柄別特性量を計算中（ADV20・ボラ・モメンタム・出来高CV）...")
    all_codes = set(rsr42_frozen["code"]) | set(rsr42_live_codes)
    for d in rep["draws"]:
        all_codes |= set(d["draw_codes"]) | set(d["twin_codes"])
    metrics = compute_stock_metrics(list(all_codes))
    sectors = load_sector33_pit_safe(list(all_codes))

    print("[4/6] ユニバース要約を構築中...")
    rsr42_summary = universe_summary(rsr42_frozen["code"].tolist(), metrics, sectors)
    rsr42_live_summary = universe_summary(rsr42_live_codes, metrics, sectors)
    draw_summaries = [universe_summary(d["draw_codes"], metrics, sectors) for d in rep["draws"]]
    twin_summaries = [universe_summary(d["twin_codes"], metrics, sectors) for d in rep["draws"]]

    print("[5/6] 統計検定・percentile位置・CI・相関を計算中...")
    metric_cols = ["adv20", "vol90d_annualized", "momentum_score", "turnover_cv"]
    per_metric = {}
    pooled_pit_stock_metrics = metrics.reindex(
        [c for d in rep["draws"] for c in d["draw_codes"]]
    )
    for col in metric_cols:
        draw_medians = [s[col]["median"] for s in draw_summaries if s.get(col)]
        rsr42_val = rsr42_summary[col]["median"]
        pct = percentile_position(rsr42_val, draw_medians)
        n_exceeded = int(round(pct / 100 * len(draw_medians)))
        ci = clopper_pearson_ci(n_exceeded, len(draw_medians))
        boot = bootstrap_of_bootstrap(draw_medians, rsr42_val)
        mwu = mann_whitney(metrics.reindex(rsr42_frozen["code"].tolist())[col], pooled_pit_stock_metrics[col])
        spearman = stats.spearmanr(draw_medians, PIT_CAGR_IS)
        per_metric[col] = {
            "rsr42_universe_median": rsr42_val,
            "pit_draw_medians_20": draw_medians,
            "rsr42_percentile_in_pit_dist": pct,
            "percentile_clopper_pearson_ci95": ci,
            "bootstrap_of_bootstrap": boot,
            "stock_level_mann_whitney": mwu,
            "spearman_draw_characteristic_vs_draw_cagr": {
                "rho": float(spearman.statistic), "p_value": float(spearman.pvalue),
            },
        }

    # CAGR自体のK=20頑健性チェック（E1本体の主張の再検証）
    pit_arr = np.array(PIT_CAGR_IS)
    n_exceeded_cagr = int((pit_arr < U0_PRIME_CAGR_IS).sum())
    cagr_ci = clopper_pearson_ci(n_exceeded_cagr, len(pit_arr))
    cagr_boot = bootstrap_of_bootstrap(PIT_CAGR_IS, U0_PRIME_CAGR_IS)

    # セクター比較: RSR42 vs プールした20ドロー平均セクター重み（Jensen-Shannon divergence）
    def js_divergence(p: dict, q: dict) -> float:
        keys = sorted(set(p) | set(q))
        pv = np.array([p.get(k, 0.0) for k in keys])
        qv = np.array([q.get(k, 0.0) for k in keys])
        pv, qv = pv / pv.sum(), qv / qv.sum()
        m = 0.5 * (pv + qv)

        def kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
        return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)

    pooled_sector_weights: dict[str, float] = {}
    for s in draw_summaries:
        for sec, w in s["sector_weights"].items():
            pooled_sector_weights[sec] = pooled_sector_weights.get(sec, 0.0) + w / len(draw_summaries)
    sector_js = js_divergence(rsr42_summary["sector_weights"], pooled_sector_weights)
    draw_hhis = [s["sector_hhi"] for s in draw_summaries]
    sector_hhi_pctile = percentile_position(rsr42_summary["sector_hhi"], draw_hhis)

    # モメンタムに対する特性調整回帰（n=20・単変量のみ・過学習回避）
    reg_results = {}
    for col in ("momentum_score", "adv20"):
        x = np.array([s[col]["median"] for s in draw_summaries])
        y = np.array(PIT_CAGR_IS)
        slope, intercept, r, p, se = stats.linregress(x, y)
        rsr42_x = rsr42_summary[col]["median"]
        predicted_cagr = float(slope * rsr42_x + intercept)
        adjusted_selection_bias = float(U0_PRIME_CAGR_IS - predicted_cagr)
        reg_results[col] = {
            "slope": float(slope), "r_squared": float(r ** 2), "p_value": float(p),
            "rsr42_characteristic_value": float(rsr42_x),
            "predicted_cagr_at_rsr42_characteristic": predicted_cagr,
            "characteristics_adjusted_residual_bias_pp": round(adjusted_selection_bias, 2),
        }

    print("[6/6] 結果を保存中...")
    out = {
        "study": "Study75C_E1_validity_audit",
        "generated_at": datetime.now(_JST).isoformat(),
        "started_at": started.isoformat(),
        "reproduction_check": {
            "n_dead_match": match_check, "dead_codes_exact_match": exact_match,
            "note": "backtestエンジン非呼び出し・RNGサンプリングのみ再現",
        },
        "rsr42_vintage_drift": {
            "frozen_n": len(rsr42_frozen), "live_n": len(rsr42_live_codes),
            "overlap_n": len(overlap), "overlap_ratio": round(len(overlap) / 42, 3),
        },
        "rsr42_summary": rsr42_summary,
        "rsr42_live_summary": rsr42_live_summary,
        "draw_summaries": draw_summaries,
        "twin_summaries": twin_summaries,
        "per_metric_comparison": per_metric,
        "cagr_robustness_recheck": {
            "n_exceeded_of_20": n_exceeded_cagr,
            "percentile_clopper_pearson_ci95": cagr_ci,
            "bootstrap_of_bootstrap": cagr_boot,
        },
        "sector_comparison": {
            "rsr42_sector_weights": rsr42_summary["sector_weights"],
            "pooled_pit_sector_weights": pooled_sector_weights,
            "jensen_shannon_divergence": sector_js,
            "rsr42_sector_hhi": rsr42_summary["sector_hhi"],
            "pit_draw_sector_hhi_distribution": draw_hhis,
            "rsr42_hhi_percentile_in_pit_dist": sector_hhi_pctile,
        },
        "characteristics_adjusted_regression": reg_results,
    }
    out_path = RESULTS_DIR / f"study75c_e1_validity_audit_{datetime.now(_JST).strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
