"""
src/backtest/study29_exit_monitoring_extension.py — Study29 Exit Monitoring Extension
(Research Freeze Phase)

背景: Study24(Entry)/Study25(Sizing/Geometry)/Study27(Risk Activation/Timing)/Study28
(Portfolio Allocation) が全てEXHAUSTED確定（Study28 oracle ceiling=+0.069<+0.10で最も
強い停止理由）。新規最適化研究は凍結し、現行Exitの品質を大サンプルで再評価する
"Exit Monitoring" フェーズへ移行する。

監査対象: 現在のExitのみ（RSR<90 + MKT_SHOCK構造的Exit, 現行production）。
変更禁止: Entry/Signal/Universe/Sizing/Allocation/Capital/Execution/Authority/Exit Logic
         （本スクリプトはこれらを一切変更しない。観測専用）。

基盤: Study28 Case A と同一（本番PARAMS_LOCKED: Capital=Y3,000,000, max_positions=3,
固定1/MAX_POS, 再配分なし）。Study21(単一スロットY1.8M, n=35)よりサンプル数を増やすため、
"同時保有が頻発する"（Study28で確認済み: 観測1320日中829日が2銘柄以上同時保有）
production-faithfulな複数ポジション構成を採用する。Entry/Exit信号自体はStudy9 Case B /
Study20-28系列と完全に同一（RSR[92,95) d90<=5 slope5<=5 / exit RSR<90, min_hold=3）。

per-trade counterfactual attribution method (tail_capture / profit_left /
loss_avoided_ratio / in_trade_mfe) is reused VERBATIM from
src/backtest/study21_exit_attribution_audit.py (attribute_trade) — no methodology change,
only sample-size extension + additional aggregate views (exit_efficiency,
holding_period_distribution, rolling_20_trade_metrics, exit_reason_distribution).

収集対象 (per task spec):
  1. tail_capture        : 全利益トレード / 直近20件 / 直近40件 / 全期間
  2. profit_left          : Exit後に取り逃した利益率
  3. loss_avoided         : Exitによって回避できた追加損失率
  4. exit_efficiency       : 実現利益 / 保有期間中最大含み益(in_trade_mfe)
  5. holding_period_distribution
  6. rolling_20_trade_metrics (tail_capture/profit_left/exit_efficiency, 20トレード移動窓)
  7. exit_reason_distribution (RSR Exit / Stop Exit / Other Exit 比率)
     注: 本戦略には「Stop Exit」という独立カテゴリは存在しない（CIRCUIT max_dd_limitは
     BUY_STOP警告のみで自動撤退なし）。生成される exit reason は RSR_EXIT と MKT_SHOCK の
     2種類のみ。MKT_SHOCK（市場急落時の強制ディフェンシブExit）をタスク仕様の
     "Stop Exit" 相当として分類する（構造的に最も近い概念のため）。Other Exit は
     観測上常に0件になる想定（コード上他のreasonは存在しない）。

判定基準 (per task spec):
  KEEP_EXIT    : tail_capture>=75% AND profit_left<=10% AND exit_efficiency 安定
  RESEARCH_EXIT: tail_capture<70% OR profit_left>15%
  REPLACE_EXIT : tail_capture<65% AND profit_left>20%
  (exit_efficiency 安定 の運用定義: rolling_20窓間のexit_efficiency標準偏差<=0.20。
   タスク仕様は定性的記述のため、本研究で採用した操作的定義として明記する。)

停止条件: 利益確定済みトレード数(tail_capture定義可能件数)>=60 OR
          いずれかの判定が境界から5pp以上の余裕(confidence=HIGH)で確定。

Run:
    cd C:/ai-trading
    python src/backtest/study29_exit_monitoring_extension.py
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from src.backtest.capital_allocation_abc import load_data, _take, LOT
from src.backtest.study20_limited_live_risk_envelope import _cross90, _slope5
from src.backtest.study28_portfolio_capital_allocation_audit import (
    simulate_case, _atrpct_matrix, CAPITAL, MAX_POS,
)
from src.backtest.study21_exit_attribution_audit import attribute_trade, TradeAttribution
from src.config_loader import load_strategy_config

OBS_START = "2018-01-01"
OBS_END   = "2025-12-31"

REPORT_DIR    = Path("reports")
REPORT_MD     = REPORT_DIR / "study29_exit_monitoring_extension.md"
TRADE_CSV     = REPORT_DIR / "study29_trade_log.csv"
ROLLING_CSV   = REPORT_DIR / "study29_rolling_20_trade_metrics.csv"
LOG_DIR       = Path("logs")
TELEMETRY_LOG = LOG_DIR / "study29_telemetry.jsonl"

# ── Decision thresholds (per task spec) ─────────────────────────────────
KEEP_EXIT_TAIL_MIN        = 0.75
KEEP_EXIT_PROFIT_LEFT_MAX = 0.10
RESEARCH_TAIL_MAX         = 0.70
RESEARCH_PROFIT_LEFT_MIN  = 0.15
REPLACE_TAIL_MAX          = 0.65
REPLACE_PROFIT_LEFT_MIN   = 0.20
EXIT_EFFICIENCY_STABILITY_STD_MAX = 0.20   # 操作的定義（"安定"の数値化）
STOP_TRADE_COUNT_MIN      = 60
CONFIDENCE_MARGIN_MIN     = 0.05            # 全境界からの最小距離(5pp) -> HIGH confidence
ROLLING_WINDOW            = 20

REASON_LABELS = {"RSR_EXIT": "RSR Exit", "MKT_SHOCK": "Stop Exit"}


@dataclass
class ExitTradeRecord:
    trade_id:        int
    symbol:          str
    entry_date:      str
    exit_date:       str
    hold_days:       int
    actual_return:    float
    actual_R:         float
    exit_reason:      str
    tail_capture:     Optional[float]
    profit_left:      float
    loss_avoided_ratio: Optional[float]
    in_trade_mfe:     float
    exit_efficiency:  Optional[float]


def build_trades(sells: List[dict], close_mat: np.ndarray, sym_to_i: Dict[str, int],
                  date_to_idx: Dict[str, int]) -> List[ExitTradeRecord]:
    records: List[ExitTradeRecord] = []
    for tid, t in enumerate(sells, start=1):
        pair = {
            "symbol": t["symbol"], "entry_date": t["entry_date"], "exit_date": t["date"],
            "entry_price": t["entry"], "exit_price": t["exit"], "hold_days": t["hold_days"],
            "reason": t["reason"],
        }
        close_series = close_mat[:, sym_to_i[pair["symbol"]]]
        attr = attribute_trade(tid, pair, close_series, date_to_idx)
        if attr is None:
            continue
        eff = (
            round(attr.actual_return / attr.in_trade_mfe, 4)
            if attr.in_trade_mfe > 1e-6 else None
        )
        records.append(ExitTradeRecord(
            trade_id=attr.trade_id, symbol=attr.symbol, entry_date=attr.entry_date,
            exit_date=attr.exit_date, hold_days=attr.hold_days,
            actual_return=attr.actual_return, actual_R=attr.actual_R,
            exit_reason=t["reason"], tail_capture=attr.tail_capture,
            profit_left=attr.profit_left, loss_avoided_ratio=attr.loss_avoided_ratio,
            in_trade_mfe=attr.in_trade_mfe, exit_efficiency=eff,
        ))
    return records


def _mean(vals: List[float]) -> Optional[float]:
    return round(float(np.mean(vals)), 4) if vals else None


def _median(vals: List[float]) -> Optional[float]:
    return round(float(np.median(vals)), 4) if vals else None


def rolling_trend(rolling: List[dict], key: str, q: int = 4) -> Optional[dict]:
    """early-quartile vs late-quartile mean comparison (degradation/trend check)."""
    vals = [w[key] for w in rolling if w[key] is not None]
    if len(vals) < q * 2:
        return None
    chunk = max(1, len(vals) // q)
    early = vals[:chunk]
    late = vals[-chunk:]
    return {
        "early_mean": _mean(early), "late_mean": _mean(late),
        "delta": round(_mean(late) - _mean(early), 4),
    }


def tail_capture_views(records: List[ExitTradeRecord]) -> Dict[str, Optional[float]]:
    winners = [r for r in records if r.tail_capture is not None]
    return {
        "all_winning_trades": _mean([r.tail_capture for r in winners]),
        "last_20":            _mean([r.tail_capture for r in winners[-20:]]),
        "last_40":             _mean([r.tail_capture for r in winners[-40:]]),
        "full_period":         _mean([r.tail_capture for r in winners]),
        "n_winning_trades":   len(winners),
    }


def holding_period_distribution(records: List[ExitTradeRecord]) -> dict:
    hd = [r.hold_days for r in records]
    if not hd:
        return {"n": 0}
    buckets = {"<=5d": 0, "6-10d": 0, "11-20d": 0, "21-40d": 0, "41-60d": 0, ">60d": 0}
    for d in hd:
        if d <= 5: buckets["<=5d"] += 1
        elif d <= 10: buckets["6-10d"] += 1
        elif d <= 20: buckets["11-20d"] += 1
        elif d <= 40: buckets["21-40d"] += 1
        elif d <= 60: buckets["41-60d"] += 1
        else: buckets[">60d"] += 1
    return {
        "n": len(hd), "mean": round(float(np.mean(hd)), 2),
        "median": round(float(np.median(hd)), 2), "std": round(float(np.std(hd)), 2),
        "min": int(np.min(hd)), "max": int(np.max(hd)), "buckets": buckets,
    }


def rolling_20_trade_metrics(records: List[ExitTradeRecord]) -> List[dict]:
    out: List[dict] = []
    n = len(records)
    if n < ROLLING_WINDOW:
        return out
    for end in range(ROLLING_WINDOW, n + 1):
        window = records[end - ROLLING_WINDOW:end]
        tc = [r.tail_capture for r in window if r.tail_capture is not None]
        pl = [r.profit_left for r in window]
        ee = [r.exit_efficiency for r in window if r.exit_efficiency is not None]
        out.append({
            "window_end_trade_id": window[-1].trade_id,
            "window_end_date": window[-1].exit_date,
            "tail_capture": _mean(tc), "profit_left": _mean(pl), "exit_efficiency": _mean(ee),
        })
    return out


def exit_reason_distribution(records: List[ExitTradeRecord]) -> dict:
    n = len(records)
    counts: Dict[str, int] = {}
    for r in records:
        counts[r.exit_reason] = counts.get(r.exit_reason, 0) + 1
    out = {}
    for raw, label in REASON_LABELS.items():
        c = counts.get(raw, 0)
        out[label] = {"count": c, "pct": round(c / n, 4) if n else 0.0}
    other = n - sum(v["count"] for v in out.values())
    out["Other Exit"] = {"count": other, "pct": round(other / n, 4) if n else 0.0}
    return out


def evaluate_decision(
    tc_views: dict, profit_left_all: float, loss_avoided_all: Optional[float],
    exit_eff_all: Optional[float], rolling: List[dict], n_records: int,
) -> dict:
    tail_capture_all = tc_views["all_winning_trades"]
    n_winners = tc_views["n_winning_trades"]

    ee_vals = [w["exit_efficiency"] for w in rolling if w["exit_efficiency"] is not None]
    ee_stability_std = round(float(np.std(ee_vals)), 4) if len(ee_vals) >= 2 else None
    ee_stable = (ee_stability_std is not None and ee_stability_std <= EXIT_EFFICIENCY_STABILITY_STD_MAX)

    if tail_capture_all is None:
        decision = "KEEP_MONITOR"
        reason = "tail_capture undefined (no winning trades)"
    elif tail_capture_all < REPLACE_TAIL_MAX and profit_left_all > REPLACE_PROFIT_LEFT_MIN:
        decision = "REPLACE_EXIT"
        reason = (f"tail_capture={tail_capture_all:.1%}<{REPLACE_TAIL_MAX:.0%} AND "
                   f"profit_left={profit_left_all:.1%}>{REPLACE_PROFIT_LEFT_MIN:.0%}")
    elif tail_capture_all < RESEARCH_TAIL_MAX or profit_left_all > RESEARCH_PROFIT_LEFT_MIN:
        decision = "RESEARCH_EXIT"
        fired = []
        if tail_capture_all < RESEARCH_TAIL_MAX:
            fired.append(f"tail_capture={tail_capture_all:.1%}<{RESEARCH_TAIL_MAX:.0%}")
        if profit_left_all > RESEARCH_PROFIT_LEFT_MIN:
            fired.append(f"profit_left={profit_left_all:.1%}>{RESEARCH_PROFIT_LEFT_MIN:.0%}")
        reason = " OR ".join(fired)
    elif (tail_capture_all >= KEEP_EXIT_TAIL_MIN and profit_left_all <= KEEP_EXIT_PROFIT_LEFT_MAX
          and ee_stable):
        decision = "KEEP_EXIT"
        reason = (f"tail_capture={tail_capture_all:.1%}>={KEEP_EXIT_TAIL_MIN:.0%} AND "
                   f"profit_left={profit_left_all:.1%}<={KEEP_EXIT_PROFIT_LEFT_MAX:.0%} AND "
                   f"exit_efficiency_stability_std={ee_stability_std}<={EXIT_EFFICIENCY_STABILITY_STD_MAX}")
    else:
        decision = "KEEP_MONITOR"
        reason = (f"borderline: tail_capture={tail_capture_all:.1%} in "
                  f"[{RESEARCH_TAIL_MAX:.0%},{KEEP_EXIT_TAIL_MIN:.0%}) or profit_left "
                  f"in ({KEEP_EXIT_PROFIT_LEFT_MAX:.0%},{RESEARCH_PROFIT_LEFT_MIN:.0%}] "
                  f"or exit_efficiency not yet stable(std={ee_stability_std})")

    margins = []
    if tail_capture_all is not None:
        margins += [abs(tail_capture_all - b) for b in (REPLACE_TAIL_MAX, RESEARCH_TAIL_MAX, KEEP_EXIT_TAIL_MIN)]
        margins += [abs(profit_left_all - b) for b in (KEEP_EXIT_PROFIT_LEFT_MAX, RESEARCH_PROFIT_LEFT_MIN, REPLACE_PROFIT_LEFT_MIN)]
    decision_margin = round(min(margins), 4) if margins else 0.0
    confidence = "HIGH" if decision_margin >= CONFIDENCE_MARGIN_MIN else "LOW"

    stop_fired = (n_winners >= STOP_TRADE_COUNT_MIN) or (confidence == "HIGH" and decision != "KEEP_MONITOR")

    return {
        "production_decision": decision, "decision_reason": reason,
        "confidence": confidence, "decision_margin": decision_margin,
        "exit_efficiency_stability_std": ee_stability_std, "exit_efficiency_stable": ee_stable,
        "stop_condition_fired": stop_fired,
        "stop_reason": (
            f"n_winning_trades={n_winners}>={STOP_TRADE_COUNT_MIN}" if n_winners >= STOP_TRADE_COUNT_MIN
            else (f"confidence=HIGH (margin={decision_margin:.1%}>={CONFIDENCE_MARGIN_MIN:.0%})"
                  if stop_fired else
                  f"NOT MET: n_winning_trades={n_winners}<{STOP_TRADE_COUNT_MIN} AND confidence=LOW(margin={decision_margin:.1%})")
        ),
    }


# ─────────────────────────────────────────────────────────────────────
#  Output writers
# ─────────────────────────────────────────────────────────────────────

def write_csvs(records: List[ExitTradeRecord], rolling: List[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if records:
        with TRADE_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            w.writeheader()
            for r in records:
                w.writerow(asdict(r))
        print(f"[CSV] {TRADE_CSV}")
    if rolling:
        with ROLLING_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rolling[0].keys()))
            w.writeheader()
            for row in rolling:
                w.writerow(row)
        print(f"[CSV] {ROLLING_CSV}")


def write_md_report(
    records: List[ExitTradeRecord], tc_views: dict, profit_left_all: float,
    loss_avoided_all: Optional[float], exit_eff_all: Optional[float],
    exit_eff_median: Optional[float], hold_dist: dict, reason_dist: dict,
    rolling: List[dict], decision: dict, tc_trend: Optional[dict], ee_trend: Optional[dict],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    w = L.append

    w("# Study29 Exit Monitoring Extension (Research Freeze Phase)")
    w("")
    w("作成日: 2026-06-24  |  観測専用 / Entry・Signal・Universe・Sizing・Allocation・"
      "Capital・Execution・Authority・Exit Logic 変更禁止 / Production変更なし")
    w("")
    w("**背景**: Study24(Entry)/Study25(Sizing)/Study27(Risk Activation)/Study28(Allocation) "
      "全てEXHAUSTED確定（Study28 oracle ceiling=+0.069<+0.10）。新規最適化研究を凍結し、"
      "現行Exit品質の継続監査(サンプル拡大)へ移行。")
    w("")
    w(f"**基盤**: Study28 Case A 同一設定（本番PARAMS_LOCKED: Capital=¥{CAPITAL:,.0f}, "
      f"max_positions={MAX_POS}, 固定1/MAX_POS, 再配分なし） / Entry・Exit信号は "
      "Study9 Case B / Study20-28系列と完全同一（RSR[92,95) d90<=5 slope5<=5 / exit RSR<90, min_hold=3）")
    w("")
    w(f"**観測ウィンドウ**: {OBS_START} → {OBS_END}（全完了トレード, "
      "counterfactual attributionはStudy21 attribute_trade()を無変更で再利用）")
    w("")

    w("---")
    w("## 1. tail_capture")
    w("")
    w("| 集計対象 | tail_capture |")
    w("|---|---|")
    w(f"| 全利益トレード (n={tc_views['n_winning_trades']}) | "
      f"{_fmt_pct(tc_views['all_winning_trades'])} |")
    w(f"| 直近20件 | {_fmt_pct(tc_views['last_20'])} |")
    w(f"| 直近40件 | {_fmt_pct(tc_views['last_40'])} |")
    w(f"| 全期間 | {_fmt_pct(tc_views['full_period'])} |")
    w("")
    w("注: 本監査は単発フルヒストリー実行（ライブの段階的蓄積ではない）のため、"
      "「全利益トレード」と「全期間」は定義上一致する（同じ母集団）。直近20件/40件は"
      "tail_capture劣化トレンドの有無を確認するための補助指標。")
    w("")

    w("---")
    w("## 2-4. profit_left / loss_avoided / exit_efficiency")
    w("")
    w("| 指標 | 値 | 定義 |")
    w("|---|---|---|")
    w(f"| profit_left (avg, 全{len(records)}トレード) | {_fmt_pct(profit_left_all)} | "
      "Exit後20営業日以内に残っていた上昇余地（counterfactual_max_gain） |")
    w(f"| loss_avoided (avg) | {_fmt_pct(loss_avoided_all)} | "
      "Exitによって回避できた追加損失率（future_loss_after_exit/max_future_loss） |")
    w(f"| exit_efficiency (avg) | {_fmt_pct(exit_eff_all)} | "
      "実現利益 / 保有期間中最大含み益(in_trade_mfe)。in_trade_mfe<=0のトレードは対象外 |")
    w(f"| exit_efficiency (median) | {_fmt_pct(exit_eff_median)} | "
      "平均値はin_trade_mfeが極小のトレードで比率が発散しやすいため、median を頑健性参考値として併記 |")
    w(f"| exit_efficiency_stability_std (rolling20間の標準偏差) | "
      f"{decision['exit_efficiency_stability_std']} | 操作的安定性閾値<= {EXIT_EFFICIENCY_STABILITY_STD_MAX} |")
    w("")
    w("注: exit_efficiency=実現利益/in_trade_mfeは分母(in_trade_mfe)が極小(例: 0.1%)のトレードで"
      "比率が±数倍に発散しうる不安定な指標。本研究ではタスク仕様の定義をそのまま適用し平均値を"
      "主指標として採用するが、median（頑健性参考値）も併記し、解釈時はrolling_20の時系列トレンド"
      "（下記7節寄り）を優先して判断する。")
    w("")

    w("---")
    w("## 5. holding_period_distribution")
    w("")
    if hold_dist.get("n", 0) > 0:
        w(f"n={hold_dist['n']}  mean={hold_dist['mean']}d  median={hold_dist['median']}d  "
          f"std={hold_dist['std']}d  min={hold_dist['min']}d  max={hold_dist['max']}d")
        w("")
        w("| bucket | count |")
        w("|---|---|")
        for b, c in hold_dist["buckets"].items():
            w(f"| {b} | {c} |")
    else:
        w("n=0 (no trades)")
    w("")

    w("---")
    w("## 6. rolling_20_trade_metrics")
    w("")
    if rolling:
        w(f"window数={len(rolling)}（trade_id基準, 20トレード移動窓） / "
          f"先頭window={_fmt_pct(rolling[0]['tail_capture'])}tail_capture → "
          f"末尾window={_fmt_pct(rolling[-1]['tail_capture'])}tail_capture")
        w("")
        w("詳細は `reports/study29_rolling_20_trade_metrics.csv` 参照（全window収録）。")
        w("")
        w("| window_end | tail_capture | profit_left | exit_efficiency |")
        w("|---|---|---|---|")
        step = max(1, len(rolling) // 15)
        for row in rolling[::step]:
            w(f"| #{row['window_end_trade_id']} ({row['window_end_date']}) | "
              f"{_fmt_pct(row['tail_capture'])} | {_fmt_pct(row['profit_left'])} | "
              f"{_fmt_pct(row['exit_efficiency'])} |")
    else:
        w(f"trade_count={len(records)} < {ROLLING_WINDOW}（rolling window未計算）")
    w("")
    if tc_trend is not None:
        w(f"**トレンド検査（前半1/4窓 vs 後半1/4窓平均）**: "
          f"tail_capture {_fmt_pct(tc_trend['early_mean'])} → {_fmt_pct(tc_trend['late_mean'])} "
          f"(Δ{tc_trend['delta']:+.1%})  /  "
          f"exit_efficiency {_fmt_pct(ee_trend['early_mean'])} → {_fmt_pct(ee_trend['late_mean'])} "
          f"(Δ{ee_trend['delta']:+.1%})")
        w("")
        if tc_trend["delta"] < -0.05:
            w("⚠ tail_captureが後半窓で5pp以上悪化 — Exit品質の経時劣化の疑いあり"
              "（KEEP_EXIT判定には不利な追加根拠）。")
            w("")

    w("---")
    w("## 7. exit_reason_distribution")
    w("")
    w("注: 本戦略には独立した「Stop Exit」カテゴリは存在しない。MKT_SHOCK"
      "（市場急落時の強制ディフェンシブExit, CIRCUIT max_dd_limitとは別の構造的Exit）を"
      "タスク仕様の Stop Exit 相当として分類。")
    w("")
    w("| 区分 | count | pct |")
    w("|---|---|---|")
    for label, v in reason_dist.items():
        w(f"| {label} | {v['count']} | {v['pct']:.1%} |")
    w("")

    w("---")
    w("## 判定基準")
    w("")
    w("| 判定 | 条件 |")
    w("|---|---|")
    w(f"| KEEP_EXIT | tail_capture>={KEEP_EXIT_TAIL_MIN:.0%} AND profit_left<={KEEP_EXIT_PROFIT_LEFT_MAX:.0%} AND exit_efficiency安定(std<={EXIT_EFFICIENCY_STABILITY_STD_MAX}) |")
    w(f"| RESEARCH_EXIT | tail_capture<{RESEARCH_TAIL_MAX:.0%} OR profit_left>{RESEARCH_PROFIT_LEFT_MIN:.0%} |")
    w(f"| REPLACE_EXIT | tail_capture<{REPLACE_TAIL_MAX:.0%} AND profit_left>{REPLACE_PROFIT_LEFT_MIN:.0%} |")
    w("")
    w(f"停止条件: 利益確定済みトレード数>={STOP_TRADE_COUNT_MIN} OR 判定境界から"
      f"{CONFIDENCE_MARGIN_MIN:.0%}以上の余裕(confidence=HIGH)")
    w("")

    w("---")
    w("## 最終出力")
    w("")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| trade_count (全完了トレード) | {len(records)} |")
    w(f"| n_winning_trades (利益確定済み, tail_capture定義可能) | {tc_views['n_winning_trades']} |")
    w(f"| tail_capture (全期間) | {_fmt_pct(tc_views['full_period'])} |")
    w(f"| profit_left | {_fmt_pct(profit_left_all)} |")
    w(f"| loss_avoided | {_fmt_pct(loss_avoided_all)} |")
    w(f"| exit_efficiency | {_fmt_pct(exit_eff_all)} |")
    w(f"| **production_decision** | **{decision['production_decision']}** |")
    w(f"| decision_reason | {decision['decision_reason']} |")
    w(f"| confidence | {decision['confidence']} (margin={decision['decision_margin']:.1%}) |")
    w(f"| stop_condition_fired | {decision['stop_condition_fired']} ({decision['stop_reason']}) |")
    w("")
    if decision["production_decision"] == "KEEP_EXIT" and decision["stop_condition_fired"]:
        w("**recommend_next_step**: KEEP_EXIT確定。現行Exit(RSR<90)を維持。"
          "新規最適化研究は引き続き凍結。LIMITED_LIVE運用継続+定期再評価(90日サイクル)へ。")
    elif decision["production_decision"] == "RESEARCH_EXIT":
        w("**recommend_next_step**: RESEARCH_EXIT。Exit品質に改善余地の疑いあり。"
          "Exit再設計の研究フェーズへの移行を検討（ただしREPLACE_EXIT条件未達のため"
          "段階的検証=WF必須、即時変更は禁止）。")
    elif decision["production_decision"] == "REPLACE_EXIT":
        w("**recommend_next_step**: REPLACE_EXIT。現行Exitロジックの置換検討に進む根拠あり。"
          "ASK_FIRST: Exit Logic変更はCLAUDE.md PERMISSIONにより必ずユーザー確認が必要。")
    else:
        w("**recommend_next_step**: KEEP_MONITOR。サンプル蓄積を継続し、"
          f"n_winning_trades>={STOP_TRADE_COUNT_MIN}またはconfidence=HIGHに到達後に再評価。"
          "現時点でExit変更提案は行わない。")
    w("")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[MD] {REPORT_MD}")


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if v is not None else "—"


def append_telemetry(tc_views: dict, profit_left_all: float, loss_avoided_all: Optional[float],
                      exit_eff_all: Optional[float], decision: dict, n_trades: int) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "study": "study29", "trade_count": n_trades,
            "exit_monitoring_result": {
                "tail_capture": tc_views, "profit_left": profit_left_all,
                "loss_avoided": loss_avoided_all, "exit_efficiency": exit_eff_all,
            },
            **decision,
        }
        with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("[Study29] Exit Monitoring Extension (Research Freeze Phase)")
    print("=" * 68)

    print("[1/4] データロード中...")
    cfg = load_strategy_config()
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())
    sym_to_i    = {s: i for i, s in enumerate(active_syms)}
    n_syms      = len(active_syms)

    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    common_dates = common_dates[
        (common_dates >= pd.Timestamp(OBS_START)) &
        (common_dates <= pd.Timestamp(OBS_END))
    ]
    date_to_idx = {str(d.date()): i for i, d in enumerate(common_dates)}
    print(f"[1/4] 共通日数={len(common_dates)}  銘柄={n_syms}")

    print("[2/4] 価格・RSRマトリクス構築...")
    n_dates = len(common_dates)
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src = universe_raw[sym]["df"]
        ri = df_src.index.get_indexer(common_dates)
        valid = ri >= 0
        if valid.any():
            open_mat[valid, si]  = df_src["Open"].to_numpy(dtype=np.float32)[ri[valid]]
            close_mat[valid, si] = df_src["Close"].to_numpy(dtype=np.float32)[ri[valid]]

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan), nan=0.0)
    atrpct_mat = _atrpct_matrix(universe_raw, active_syms, common_dates, close_mat)
    sym_active_mat = (
        None if sym_active_df is None
        else _take(sym_active_df, common_dates, active_syms, dtype=np.float32, fill_value=1.0)
    )
    mkt_ret1 = (
        None if topix_close is None
        else _take(topix_close.pct_change(), common_dates, dtype=np.float32, fill_value=0.0)
    )
    cross90_mat = _cross90(rsr_mat)
    slope5_mat  = _slope5(rsr_mat)

    print("[3/4] Study28 Case A (production-faithful, max_pos=3, FROZEN) シミュレーション...")
    sim = simulate_case(
        "A", common_dates, active_syms, sym_to_i, open_mat, close_mat, rsr_mat, atrpct_mat,
        sym_active_mat, cross90_mat, slope5_mat, mkt_ret1, CAPITAL,
    )
    sells = [t for t in sim["trades"] if t["side"] == "SELL"]
    print(f"[3/4] completed_trades={len(sells)}")

    print("[4/4] Per-trade counterfactual attribution + 拡張メトリクス集計...")
    records = build_trades(sells, close_mat, sym_to_i, date_to_idx)

    tc_views = tail_capture_views(records)
    profit_left_all = _mean([r.profit_left for r in records]) or 0.0
    loss_avoided_all = _mean([r.loss_avoided_ratio for r in records if r.loss_avoided_ratio is not None])
    eff_vals = [r.exit_efficiency for r in records if r.exit_efficiency is not None]
    exit_eff_all = _mean(eff_vals)
    exit_eff_median = _median(eff_vals)
    hold_dist = holding_period_distribution(records)
    rolling = rolling_20_trade_metrics(records)
    reason_dist = exit_reason_distribution(records)
    decision = evaluate_decision(tc_views, profit_left_all, loss_avoided_all, exit_eff_all,
                                  rolling, len(records))
    tc_trend = rolling_trend(rolling, "tail_capture")
    ee_trend = rolling_trend(rolling, "exit_efficiency")

    print(f"  trade_count           = {len(records)}")
    print(f"  n_winning_trades      = {tc_views['n_winning_trades']}")
    print(f"  tail_capture(全期間)   = {_fmt_pct(tc_views['full_period'])}")
    print(f"  profit_left           = {_fmt_pct(profit_left_all)}")
    print(f"  loss_avoided          = {_fmt_pct(loss_avoided_all)}")
    print(f"  exit_efficiency       = {_fmt_pct(exit_eff_all)}")
    print(f"  production_decision   = {decision['production_decision']}")
    print(f"  confidence            = {decision['confidence']} (margin={decision['decision_margin']:.1%})")
    print(f"  stop_condition_fired  = {decision['stop_condition_fired']}")

    write_csvs(records, rolling)
    write_md_report(records, tc_views, profit_left_all, loss_avoided_all, exit_eff_all,
                     exit_eff_median, hold_dist, reason_dist, rolling, decision, tc_trend, ee_trend)
    append_telemetry(tc_views, profit_left_all, loss_avoided_all, exit_eff_all, decision, len(records))

    print()
    print("=" * 68)
    print(f"production_decision = {decision['production_decision']}  "
          f"(trade_count={len(records)}, n_winning={tc_views['n_winning_trades']})")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
