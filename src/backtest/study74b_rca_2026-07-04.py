"""
study74b_rca_2026-07-04.py
Study74B-RCA — CAP_MISS矛盾（見送り候補=採用候補と同品質なのにmax_positions緩和はCAGR悪化）の
Root Cause Analysis。

**新規BT禁止**（ユーザー指示・厳守）。既存Research Assetsのみ使用:
  - backtests/study78_trade_dataset.json（309採用トレード全件）
  - backtests/study74b_candidate_shortage_2026-07-04.json（見送り候補の集計統計）
  - backtests/study74_capital_scaling_2026-07-04.json（制約waterfall・参照用）

本スクリプトが行うのは「backtest実行(run_scenario呼び出し)」ではなく、
既にキャッシュ済みの価格データ(build_common_dataset内のuniverse_raw/regime_df)を
読み込んでの記述統計・データ結合のみ（戦略シミュレーションは一切実行しない）。

⚠ 重要な制約: 見送り候補(CAP_MISS 449件)の個別(date, symbol)一覧は
Study74B実行時にメモリ上でのみ存在し、集計統計(mean/median/percentile等)のみが
JSON化されていた。個別レコードは永続化されていないため、
解析2(日次ペアリング)・解析3(Opportunity Cost)は本スクリプトでは実行不可能
（新規BTで_missed_candsを再取得すれば可能だが、本タスクでは禁止されている）。
この制約は隠さず明記し、"未解決"として結果に含める。
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest.snapshot_archaeology_202606 import build_common_dataset

TODAY_STR = date.today().strftime("%Y-%m-%d")
BT_DIR = ROOT / "backtests"

TRADE_DATASET = BT_DIR / "study78_trade_dataset.json"
SHORTAGE_JSON = BT_DIR / "study74b_candidate_shortage_2026-07-04.json"
CAPSCALE_JSON = BT_DIR / "study74_capital_scaling_2026-07-04.json"


def load_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stats(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    a = np.array(vals, dtype=float)
    return {"n": len(a), "mean": round(float(np.mean(a)), 3), "median": round(float(np.median(a)), 3),
            "p25": round(float(np.percentile(a, 25)), 3), "p75": round(float(np.percentile(a, 75)), 3),
            "std": round(float(np.std(a)), 3)}


def main():
    print("=" * 80)
    print("  Study74B-RCA — CAP_MISS矛盾の根本原因分析（新規BT禁止・既存データのみ）")
    print(f"  Date: {TODAY_STR}")
    print("=" * 80)

    trades = load_json(TRADE_DATASET)["trades"]
    shortage = load_json(SHORTAGE_JSON)["analysis"]
    capscale = load_json(CAPSCALE_JSON)

    print(f"\n[LOAD] 採用トレード: {len(trades)}件（既存 study78_trade_dataset.json）")
    print(f"[LOAD] 見送り候補集計: {SHORTAGE_JSON.name}")
    print(f"[LOAD] 制約waterfall: {CAPSCALE_JSON.name}")

    # 価格データ・regimeのみ読み込み（run_scenario＝戦略シミュレーションは一切実行しない）
    print("\n[DATA] キャッシュ済み価格・regimeデータ読み込み中（BTではない）...")
    ds = build_common_dataset("2025-12-31")
    universe_raw = ds["universe_raw"]
    regime_df = ds["regime_df"]
    trade_syms = ds["trade_syms"]

    # ── 解析1: 採用トレードの完全記述統計（MFE/MAE含む・価格データから直接計算） ──
    print("\n[解析1] 採用トレードの記述統計を計算中（MFE/MAE含む）...")
    enriched = []
    for t in trades:
        sym = t["symbol"]
        if sym not in universe_raw or not t["entry_date"] or not t["exit_date"]:
            continue
        df = universe_raw[sym]["df"]
        try:
            window = df.loc[t["entry_date"]:t["exit_date"]]
        except Exception:
            window = pd.DataFrame()
        if len(window) == 0:
            mfe = mae = None
        else:
            entry_px = t["entry_price"]
            mfe = round((float(window["High"].max()) - entry_px) / entry_px * 100, 2) if entry_px > 0 else None
            mae = round((float(window["Low"].min()) - entry_px) / entry_px * 100, 2) if entry_px > 0 else None
        exit_d = pd.Timestamp(t["exit_date"])
        regime = "unknown"
        if exit_d in regime_df.index:
            regime = "risk_off" if bool(regime_df.loc[exit_d, "risk_off"]) else "normal"
        enriched.append({**t, "mfe_pct": mfe, "mae_pct": mae, "exit_regime": regime})

    winners = [t for t in enriched if t["pnl"] >= 0]
    losers = [t for t in enriched if t["pnl"] < 0]
    win_rate = round(len(winners) / max(1, len(enriched)) * 100, 1)
    avg_profit = round(float(np.mean([t["pnl"] for t in winners])), 1) if winners else 0.0
    avg_loss = round(float(np.mean([t["pnl"] for t in losers])), 1) if losers else 0.0
    gross_profit = sum(t["pnl"] for t in winners)
    gross_loss = abs(sum(t["pnl"] for t in losers))
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")
    expectancy = round(float(np.mean([t["pnl"] for t in enriched])), 1) if enriched else 0.0

    adopted_profile = {
        "n_trades": len(enriched), "win_rate_pct": win_rate,
        "avg_profit_yen": avg_profit, "avg_loss_yen": avg_loss,
        "profit_factor": profit_factor, "expectancy_yen": expectancy,
        "entry_atr_pct": stats([t["entry_atr_pct"] for t in enriched]),
        "entry_rsr": stats([t["entry_rsr"] for t in enriched]),
        "holding_days": stats([t["holding_days"] for t in enriched]),
        "mfe_pct": stats([t["mfe_pct"] for t in enriched]),
        "mae_pct": stats([t["mae_pct"] for t in enriched]),
        "by_regime": {r: len([t for t in enriched if t["exit_regime"] == r]) for r in ["risk_off", "normal", "unknown"]},
        "by_sector": {},
        "by_entry_type": {},
    }
    for t in enriched:
        adopted_profile["by_sector"][t["sector"]] = adopted_profile["by_sector"].get(t["sector"], 0) + 1
        adopted_profile["by_entry_type"][t["entry_type"]] = adopted_profile["by_entry_type"].get(t["entry_type"], 0) + 1

    print(f"  採用トレード: win_rate={win_rate}%  PF={profit_factor}  expectancy={expectancy}円  "
          f"MFE中央値={adopted_profile['mfe_pct'].get('median')}%  MAE中央値={adopted_profile['mae_pct'].get('median')}%")

    missed_profile = {
        "note": "見送り候補(CAP_MISS 449件)の個別(date,symbol)一覧は永続化されておらず、"
                "集計統計のみ既存JSON(study74b_candidate_shortage)から参照可能。"
                "momentum/holding_days/win_rate/profit/loss/PF/expectancy/MFE/MAEは"
                "個別レコードと事後の価格追跡が必須のため計算不可能（新規BT要）。",
        "entry_rsr": shortage["candidate_quality"]["missed_candidates_rsr"],
        "alpha_score": shortage["candidate_quality"]["missed_candidates_alpha"],
        "rank_histogram": shortage["candidate_competition_rank_histogram"],
        "by_regime": shortage["missed_by_regime"],
        "by_year": shortage["missed_by_year"],
        "by_sector": shortage["missed_by_sector"],
        "entry_atr_pct": "N/A（未永続化）", "momentum": "N/A（未永続化）",
        "holding_days": "N/A（トレードされていないため概念自体が不成立）",
        "win_rate": "N/A（要フォワード価格追跡・新規BT）", "avg_profit": "N/A（同上）", "avg_loss": "N/A（同上）",
        "profit_factor": "N/A（同上）", "expectancy": "N/A（同上）",
        "mfe_pct": "N/A（同上）", "mae_pct": "N/A（同上）",
    }

    analysis1 = {"adopted_profile": adopted_profile, "missed_profile": missed_profile,
                 "comparison_available_axes": {
                     "entry_rsr": {"adopted_median": adopted_profile["entry_rsr"].get("median"),
                                   "missed_median": missed_profile["entry_rsr"].get("median"),
                                   "verdict": "ほぼ同一（品質差なし）"},
                 },
                 "comparison_blocked_axes": ["momentum", "sector別win_rate", "holding_days(見送り側)",
                                             "win_rate(見送り側)", "avg_profit/avg_loss(見送り側)",
                                             "profit_factor(見送り側)", "expectancy(見送り側)",
                                             "MFE/MAE(見送り側)"]}

    # ── 解析2: CAP_MISS日次ペアリング — ブロック（データ欠落） ──────────────
    print("\n[解析2] 日次ペアリング（採用3銘柄 vs 見送り候補）...")
    print("  ⚠ ブロック: 見送り候補の個別(date,symbol)が未永続化のため実行不可能。年次集計のみ可能。")
    cap_miss_pairs = {
        "status": "PARTIAL_BLOCKED",
        "reason": "CAP_MISS発生日ごとの個別(date,symbol)ペアが study74b 実行時に永続化されていない"
                  "（集計統計のみJSON化）。日次の厳密ペアリングには新規BTでの_missed_cands再取得が必須だが、"
                  "本タスクでは新規BT禁止のため実行不可能。",
        "best_effort_year_level_pairing": [],
        "unresolved": True,
    }
    # 年次レベルでの粗いペアリング（見送り件数 vs その年の採用トレード数・成績）
    trades_by_year: dict[str, list] = {}
    for t in enriched:
        if not t["exit_date"]:
            continue
        yr = t["exit_date"][:4]
        trades_by_year.setdefault(yr, []).append(t)
    for yr, missed_n in shortage["missed_by_year"].items():
        yr_trades = trades_by_year.get(yr, [])
        yr_pnl = sum(t["pnl"] for t in yr_trades)
        yr_win_rate = round(100 * len([t for t in yr_trades if t["pnl"] >= 0]) / max(1, len(yr_trades)), 1)
        cap_miss_pairs["best_effort_year_level_pairing"].append({
            "year": yr, "missed_candidate_count": missed_n,
            "adopted_trade_count_exiting_that_year": len(yr_trades),
            "adopted_total_pnl_yen": round(yr_pnl, 1), "adopted_win_rate_pct": yr_win_rate,
        })
        print(f"    {yr}: missed={missed_n}件  adopted_exits={len(yr_trades)}件  "
              f"pnl={yr_pnl:+.0f}円  win_rate={yr_win_rate}%")

    # ── 解析3: Opportunity Cost — ブロック（データ欠落） ─────────────────
    print("\n[解析3] Opportunity Cost（見送り候補のその後の値動き）...")
    print("  ⚠ ブロック: 見送り候補のsymbol/date個別識別子が無いためフォワード価格追跡が不可能。")
    opportunity_cost = {
        "status": "BLOCKED", "unresolved": True,
        "reason": "見送り候補(449件)の個別symbol・発生dateが永続化されていないため、"
                  "見送り後Xシテ日以内の値動き(フォワードリターン)を追跡できない。"
                  "追跡には新規BTで_missed_candsの個別レコード(date,symbol,rsr,atr_pct,rank)を"
                  "再取得し、その後の価格データと突合する必要があるが、本タスクでは新規BT禁止。",
        "what_would_be_needed": [
            "①_missed_candsを個別レコードのままJSON永続化する新規BT(1回)",
            "②各レコードのsymbol×date+N営業日(5/10/20/40日)のforward returnを価格データから計算",
            "③同期間の採用銘柄のforward returnと比較しΔを可視化",
        ],
        "partial_context_available": {
            "reference": "Study74 Part A waterfall（既存）: max_positions緩和時のCAGR/MaxDD変化",
            "relax_maxpos_delta_cagr_by_capital": capscale["part_a_constraint_waterfall"],
        },
    }

    # ── 解析4: Hidden Factor探索（既存集計データからの仮説構築） ──────────
    print("\n[解析4] Hidden Factor探索（既存データからの仮説抽出）...")
    rank_hist = shortage["candidate_competition_rank_histogram"]
    total_missed = sum(rank_hist.values())
    rank0_share = round(rank_hist.get("rank0(最上位)", 0) / max(1, total_missed) * 100, 1)

    hidden_factor_analysis = {
        "observed_facts": {
            "rsr_equivalence": "見送りRSR中央値81.0 = 採用RSR中央値81.0（完全一致・品質差なし）",
            "alpha_near_zero": f"見送り候補のalpha score: mean=median=p25=p75=0.0（{shortage['candidate_quality']['missed_candidates_alpha']}）",
            "rank0_dominance": f"見送りの{rank0_share}%が「その日の最上位候補」（rank0）",
            "maxpos_relax_hurts_at_scale": "Study74 Part A実測: ¥20M以降はmax_positions緩和がCAGRを悪化させる"
                                            f"（delta_cagr={[capscale['part_a_constraint_waterfall'][c]['delta_cagr']['max_positions'] for c in ['3000000','10000000','20000000','30000000']]}）",
        },
        "hypothesis": {
            "name": "ポートフォリオ状態依存仮説（timing/slot-availability hypothesis）",
            "statement": "見送り候補の「質」はRSR・alpha双方で採用候補と統計的に区別できない。"
                         "差を生んでいるのは候補自身の属性ではなく「候補が到着した時点で既存ポジションが"
                         "何日目のどの局面にあったか」というポートフォリオ側の状態（＝隠れ因子）である可能性が高い。"
                         "rank0候補(最上位)が56%も見送られているのは、候補の質の問題ではなく、"
                         "既存3ポジションの入れ替わりタイミングと新規候補の到着タイミングが噛み合っていないことを示す。",
            "supporting_evidence": ["rsr_equivalence", "rank0_dominance"],
            "why_maxpos_relax_still_hurts": "候補自体は良質でも、それを「同時に」保有することで"
                                            "既存の少数精鋭ポジション（集中投資）が持つ複利効果・比重が希薄化する。"
                                            "また高候補発生期（2023年など）は市場全体のモメンタムが強く、"
                                            "同時多発的に類似セクター・類似要因の銘柄が候補化しやすいため、"
                                            "見かけ上の分散が実質的な相関の高い集中（隠れた集中リスク）になっている可能性がある。",
            "confidence": "仮説（部分的示唆はあるが解析2/3のデータ欠落により因果の完全証明はできない）",
        },
        "additional_hidden_factor_candidates_untested": [
            "ATR順位（見送り候補のATR%分布は本データでは取得不可・未検証）",
            "Momentum順位（momentumフィールド自体が未永続化・未検証）",
            "Composite Score順位差（rank0でも僅差 vs 大差だったかは不明・未検証）",
            "Entry日の曜日・月内位置（季節性、未検証）",
        ],
    }
    print(f"  rank0(最上位)候補の見送り比率: {rank0_share}%")
    print(f"  仮説: ポートフォリオ状態依存仮説（候補品質ではなくタイミングが差を生む） — 信頼度=仮説止まり")

    # ── 保存（ユーザー指定の固定ファイル名・日付サフィックスなし） ──────────
    with open(BT_DIR / "cap_miss_pairs.json", "w", encoding="utf-8") as f:
        json.dump(cap_miss_pairs, f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / "opportunity_cost.json", "w", encoding="utf-8") as f:
        json.dump(opportunity_cost, f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / "hidden_factor_analysis.json", "w", encoding="utf-8") as f:
        json.dump(hidden_factor_analysis, f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / f"study74b_rca_analysis1_{TODAY_STR}.json", "w", encoding="utf-8") as f:
        json.dump(analysis1, f, ensure_ascii=False, indent=2, default=str)

    print("\n[OUTPUT] backtests/cap_miss_pairs.json")
    print("[OUTPUT] backtests/opportunity_cost.json")
    print("[OUTPUT] backtests/hidden_factor_analysis.json")
    print(f"[OUTPUT] backtests/study74b_rca_analysis1_{TODAY_STR}.json")


if __name__ == "__main__":
    main()
