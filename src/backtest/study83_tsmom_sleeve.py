"""
src/backtest/study83_tsmom_sleeve.py
Study83 — 指数TSMOM（時系列モメンタム）実測評価

正典: reports/study83_proposal.md（起案書・Proposal only段階）
      reports/roadmap_v15_governance_layer.md §8A-8（CP3までの流れ・Study83実装段階）
      final_research_roadmap_2026-07-04.md L9 Study83定義:
        成功=Sharpe≥0.8 ∧ 現行Core相関<0.5 ∧ ロールコスト補正後正。固定グリッド{20,60,120d}のみ。
      alternative_architectures_5x_2026-07-03.md §ARCH-C（TSMOM原典設計）

目的（狭く固定・ユーザータスク指定）:
  Study103で採用したTSMOM仮定（study103_design.md §3.4）が現実データで生存するかを測定する。
  30%到達研究でも新規アルファ探索でもない — 「仮定が楽観的/現実的/保守的のいずれか」の判定のみ。

禁止（厳守）:
  - 新規アルファアイデア・Route C研究・新規overlay・ポートフォリオ再最適化・追加ガバナンス変更
  - PEADアルファ推定（Study82 Phase D未実施のため、PEAD相関は測定不能=N/Aとして扱う。
    代替値の推定・仮置きは禁止）

固定仕様（事前固定・実行後の変更禁止 — 変更が必要な場合はStudy83B新版）:
  - シグナル: 絶対モメンタム sign(TOPIX Close[t-1]/Close[t-1-N] - 1)（N∈{20,60,120}固定グリッド・
    canon「アーム追加・閾値スイープ禁止」準拠）
  - PIT安全性: シグナル・ボラティリティ推定とも shift(1)（前日終値までの情報のみ使用）
  - vol-target: 目標ボラ=22.5%（Study103 Base仮定の中央値・study103_design.md §3.4より流用。
    結果を見てからの逆算的パラメータ選択ではない）。realized_vol=63日ローリング年率化・
    レバ上限2.0x（Study103自動RED境界 required_leverage>2x と整合させた上限）
  - コスト: 0.05%（5bp）/ポジション変化量単位（実先物ロールコストの実測なし・保守的近似と明記）
  - データ: data/jquants/processed/TOPIX.parquet（canon Study76/95/99と同一ソース・2016-07-11〜
    2026-07-09・2,442営業日）。単一指数（TOPIX）のみ — 日経225先物・セクターETFは本Studyのスコープ外
    （「minimal PIT-safe TSMOM sleeve」の実装として単一指数へ絞り込み・スコープ限定を明記）
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

try:
    from src.paths import JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.paths import JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR

RUN_DATE = "2026-07-21"
OUT_JSON = RESULTS_DIR / f"study83_tsmom_sleeve_{RUN_DATE}.json"
OUT_MD = REPORTS_DIR / "study83_tsmom_sleeve.md"

TOPIX_PARQUET = JQUANTS_PROCESSED_DIR / "TOPIX.parquet"
CORE_TRADE_DATASET = RESULTS_DIR / "study78_trade_dataset.json"

LOOKBACKS = (20, 60, 120)  # 固定グリッド（canon・変更禁止）
TARGET_VOL = 0.225          # Study103 Base仮定の中央値（study103_design.md §3.4）
VOL_WINDOW = 63
MAX_LEVERAGE = 2.0          # Study103自動RED境界 required_leverage>2x と整合
COST_BP_PER_TURNOVER_UNIT = 0.0005  # 5bp・実ロールコスト未実測の保守的近似
TRADING_DAYS = 252

# Study103仮定表（比較用・凍結値・study103_design.md §3.4を転記）
ASSUMPTIONS = {
    "Conservative": {"cagr": 0.15, "vol": 0.25, "maxdd": 0.20, "core_corr": 0.35},
    "Base":         {"cagr": 0.18, "vol": 0.225, "maxdd": 0.15, "core_corr": 0.25},
    "Optimistic":   {"cagr": 0.25, "vol": 0.20, "maxdd": 0.12, "core_corr": 0.15},
}
SUCCESS_GATE = {"sharpe_min": 0.8, "core_corr_max": 0.5}  # canon成功条件（final_research_roadmap L9）


def load_topix() -> pd.Series:
    df = pd.read_parquet(TOPIX_PARQUET)
    df.index = pd.to_datetime(df.index)
    return df["Close"].sort_index()


def run_arm(close: pd.Series, lookback: int) -> pd.DataFrame:
    daily_ret = close.pct_change()
    mom = close.shift(1) / close.shift(1 + lookback) - 1.0  # PIT: t-1時点までの情報のみ
    signal = np.sign(mom)

    realized_vol = daily_ret.shift(1).rolling(VOL_WINDOW).std() * np.sqrt(TRADING_DAYS)
    size = (TARGET_VOL / realized_vol).clip(upper=MAX_LEVERAGE).fillna(0.0)
    exposure = (signal * size).fillna(0.0)

    turnover_amt = exposure.diff().abs().fillna(0.0)
    cost = turnover_amt * COST_BP_PER_TURNOVER_UNIT

    gross_ret = exposure * daily_ret
    net_ret = (gross_ret - cost).fillna(0.0)

    out = pd.DataFrame({
        "close": close, "daily_ret": daily_ret, "mom": mom, "signal": signal,
        "realized_vol": realized_vol, "size": size, "exposure": exposure,
        "turnover_amt": turnover_amt, "cost": cost, "gross_ret": gross_ret, "net_ret": net_ret,
    })
    # ウォームアップ期間（vol推定+モメンタム算出に必要な最大ラグ）を除外
    warmup = lookback + VOL_WINDOW + 2
    return out.iloc[warmup:]


def perf_metrics(net_ret: pd.Series) -> dict:
    wealth = (1.0 + net_ret).cumprod()
    n_years = len(net_ret) / TRADING_DAYS
    cagr = float(wealth.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else float("nan")
    ann_vol = float(net_ret.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(cagr / ann_vol) if ann_vol > 1e-9 else float("nan")
    peak = wealth.cummax()
    dd = 1.0 - wealth / peak
    maxdd = float(dd.max())
    calmar = float(cagr / maxdd) if maxdd > 1e-9 else float("nan")
    return {"cagr": round(cagr, 4), "ann_vol": round(ann_vol, 4), "sharpe": round(sharpe, 3),
            "maxdd": round(maxdd, 4), "calmar": round(calmar, 3), "n_years": round(n_years, 2)}


def core_monthly_returns() -> pd.Series:
    """
    study78_trade_dataset.json（309トレード・Core Production台帳）から月次リターン近似系列を構築。
    手法: 各トレードのpnl_pct_of_capitalをexit_date所属月へ帰属・月次合算。
    注意: 真のdaily mark-to-marketエクイティカーブではなく「決済月にPnLを認識する」近似
    （日次エクイティ曲線の永続化資産が現状存在しないため — Study78観測基盤の外側）。
    相関の符号・大まかな強度の把握には十分だが、厳密な日次相関ではない点を明記する。
    """
    d = json.loads(CORE_TRADE_DATASET.read_text(encoding="utf-8"))
    trades = d["trades"]
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["month"] = df["exit_date"].dt.to_period("M")
    monthly = df.groupby("month")["pnl_pct_of_capital"].sum() / 100.0  # pnl_pct_of_capitalは%表記
    return monthly  # index: pandas Period[M]（月末/月初表記ゆれを避けるためPeriod型のまま返す）


def tsmom_monthly_returns(net_ret: pd.Series) -> pd.Series:
    m = (1.0 + net_ret).resample("ME").apply(lambda x: x.prod() - 1.0)
    m.index = m.index.to_period("M")  # Core側とPeriod型で厳密一致させる（月初/月末ズレの排除）
    return m


def main() -> None:
    print("Study83 — 指数TSMOM実測評価（PIT-safe minimal sleeve・固定グリッド20/60/120d）")
    print("目的: Study103 TSMOM仮定は現実データで生存するか（新規アルファ探索ではない）\n")

    close = load_topix()
    print(f"TOPIX: {close.index[0].date()} 〜 {close.index[-1].date()}（{len(close)}営業日）")

    core_m = core_monthly_returns()
    print(f"Core月次系列（study78trade帰属近似）: {core_m.index[0]} 〜 {core_m.index[-1]}（{len(core_m)}ヶ月）")

    results: dict = {"config": {
        "lookbacks": LOOKBACKS, "target_vol": TARGET_VOL, "vol_window": VOL_WINDOW,
        "max_leverage": MAX_LEVERAGE, "cost_bp_per_turnover_unit": COST_BP_PER_TURNOVER_UNIT,
        "data_source": str(TOPIX_PARQUET), "core_reference": str(CORE_TRADE_DATASET),
    }, "arms": {}, "pead_correlation": "N/A — Study82 Phase D（PEADアルファ実測）未実施のため測定不能。"
                                        "代替値の推定は禁止事項（PEAD alpha estimation）に該当するため実施していない。"}

    for lb in LOOKBACKS:
        arm = run_arm(close, lb)
        perf = perf_metrics(arm["net_ret"])

        n_years = perf["n_years"]
        n_flips = int((np.sign(arm["exposure"]).diff().fillna(0) != 0).sum())
        turnover_per_year = round(n_flips / n_years, 2) if n_years > 0 else None
        avg_gross_exposure = round(float(arm["exposure"].abs().mean()), 3)
        pct_flat = round(float((arm["exposure"].abs() < 1e-6).mean()), 4)
        pct_long = round(float((arm["exposure"] > 1e-6).mean()), 4)
        pct_short = round(float((arm["exposure"] < -1e-6).mean()), 4)

        tsmom_m = tsmom_monthly_returns(arm["net_ret"])
        aligned = pd.concat([tsmom_m.rename("tsmom"), core_m.rename("core")], axis=1, join="inner").dropna()
        core_corr = round(float(aligned["tsmom"].corr(aligned["core"])), 3) if len(aligned) >= 6 else None

        gate_sharpe = perf["sharpe"] >= SUCCESS_GATE["sharpe_min"]
        gate_corr = (core_corr is not None) and (core_corr < SUCCESS_GATE["core_corr_max"])
        gate_cost_positive = perf["cagr"] > 0.0
        canon_success = bool(gate_sharpe and gate_corr and gate_cost_positive)

        vs_assumption = {}
        for level, a in ASSUMPTIONS.items():
            vs_assumption[level] = {
                "cagr_delta_pp": round((perf["cagr"] - a["cagr"]) * 100, 2),
                "vol_delta_pp": round((perf["ann_vol"] - a["vol"]) * 100, 2),
                "maxdd_delta_pp": round((perf["maxdd"] - a["maxdd"]) * 100, 2),
                "core_corr_delta": round((core_corr - a["core_corr"]), 3) if core_corr is not None else None,
            }
        closest_level = min(ASSUMPTIONS, key=lambda lv: abs(perf["cagr"] - ASSUMPTIONS[lv]["cagr"]))

        results["arms"][f"N{lb}"] = {
            "perf": perf, "turnover_per_year": turnover_per_year,
            "avg_gross_exposure": avg_gross_exposure, "pct_flat": pct_flat,
            "pct_long": pct_long, "pct_short": pct_short,
            "core_correlation_monthly": core_corr, "n_months_aligned": len(aligned),
            "canon_success_gate": {"sharpe_ge_0.8": gate_sharpe, "core_corr_lt_0.5": gate_corr,
                                    "cost_adjusted_positive": gate_cost_positive, "overall": canon_success},
            "vs_study103_assumption": vs_assumption, "closest_assumption_level": closest_level,
        }
        print(f"[N={lb}d] CAGR={perf['cagr']:+.2%} Vol={perf['ann_vol']:.2%} Sharpe={perf['sharpe']:.2f} "
              f"MaxDD={perf['maxdd']:.2%} Calmar={perf['calmar']:.2f} CoreCorr={core_corr} "
              f"Turnover/yr={turnover_per_year} canon_success={canon_success} "
              f"closest={closest_level}")

    n_arms_pass = sum(1 for a in results["arms"].values() if a["canon_success_gate"]["overall"])
    results["summary"] = {
        "n_arms_pass_canon_gate": n_arms_pass, "n_arms_total": len(LOOKBACKS),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n{n_arms_pass}/{len(LOOKBACKS)} arms pass canon success gate (Sharpe>=0.8 & Corr<0.5 & cost-adj positive)")
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
