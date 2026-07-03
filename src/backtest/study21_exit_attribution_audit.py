"""
src/backtest/study21_exit_attribution_audit.py — Study21 Exit Attribution Audit

Objective: Quantify what the current Exit policy (RSR < 90) is actually doing.
Does NOT modify Exit / Entry / sizing / authority / execution.
Does NOT optimize profit. Observation-only attribution study.

Fixed configuration (FROZEN, reused from Study20/Study9 Case B):
  Strategy : Study9 Case B  (src.backtest.study20_limited_live_risk_envelope.generate_signals)
  Entry    : current production entry  RSR[92,95) d90<=5, slope5<=5
  Exit     : RSR<90 (+ MKT_SHOCK structural exit, unchanged)
  Capital  : current production configuration (CAPITAL/LOT/SLIPPAGE/COMMISSION from Study20)
  Scope    : all completed trades, 2018-01-01..2025-12-31 (full available observation window)

Research question:
  Does the current Exit preserve the profit tail while acting as a risk-control mechanism?

Methodology (see report "Methodology" section for full definitions):
  - Per-trade counterfactual replay: forward price path for LOOKAHEAD=20 trading days
    after the actual exit (raw close, no slippage/commission — theoretical path).
  - tail_capture        = actual_return / max_future_profit  (winning trades only)
        max_future_profit = max(actual_return, max gain from entry over [entry, exit+20d])
        losing trades have no profit tail to capture -> tail_capture=None (excluded)
  - profit_left         = counterfactual_max_gain (upside left on the table after exit)
  - loss_avoided_ratio  = future_loss_after_exit / max_future_loss
        future_loss_after_exit = decline that occurred AFTER exit (avoided by exiting)
        max_future_loss        = worst decline from entry over the full extended window
  - exit_quality_score  = 0.5*tail_capture + 0.5*loss_avoided_ratio (when both defined)
  - early_exit_cost     = mean(profit_left) across all trades (upside given up by exiting)
  - late_exit_cost      = mean(max(0, in_trade_MFE - actual_return)) (peak give-back before exit)

Restrictions: REPLACE_EXIT prohibited. No Exit redesign proposed here.

Run:
    cd C:/ai-trading
    python src/backtest/study21_exit_attribution_audit.py
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
from src.backtest.study20_limited_live_risk_envelope import (
    generate_signals,
    SLIPPAGE, COMMISSION, COST, R_UNIT_PCT, STOP_LOSS_R,
)
from src.config_loader import load_strategy_config

OBS_START = "2018-01-01"
OBS_END   = "2025-12-31"
LOOKAHEAD = 20   # trading days post-exit counterfactual horizon

REPORT_DIR    = Path("reports")
REPORT_MD     = REPORT_DIR / "study21_exit_attribution_audit.md"
TRADE_CSV     = REPORT_DIR / "study21_trade_attribution.csv"
LOG_DIR       = Path("logs")
TELEMETRY_LOG = LOG_DIR / "study21_telemetry.jsonl"

# ── Decision thresholds (per task spec) ─────────────────────────────────
KEEP_EXIT_TAIL_MIN     = 0.80
KEEP_EXIT_PROFIT_LEFT_MAX = 0.15
KEEP_EXIT_TRADE_MIN    = 20
MONITOR_TAIL_LO        = 0.60
MONITOR_TAIL_HI        = 0.80
RESEARCH_TAIL_MAX      = 0.60
RESEARCH_PROFIT_LEFT_MIN = 0.20


@dataclass
class TradeAttribution:
    trade_id:                  int
    symbol:                    str
    entry_date:                str
    exit_date:                 str
    hold_days:                 int
    entry_price:                float
    exit_price:                 float
    actual_return:               float
    actual_R:                    float
    counterfactual_ret_5d:        Optional[float]
    counterfactual_ret_10d:       Optional[float]
    counterfactual_ret_20d:       Optional[float]
    counterfactual_max_gain:      float
    counterfactual_max_loss:      float
    counterfactual_peak_day:      int
    counterfactual_stop_breach_R: float
    tail_capture:                 Optional[float]
    profit_left:                  float
    loss_avoided_ratio:           Optional[float]
    exit_quality_score:           Optional[float]
    in_trade_mfe:                 float
    late_exit_giveback:           float


@dataclass
class Study21Result:
    trade_count:                int   = 0
    tail_capture:                float = 0.0
    profit_left:                 float = 0.0
    loss_avoided_ratio:          float = 0.0
    exit_quality_score:          float = 0.0
    counterfactual_peak_day:     float = 0.0
    counterfactual_stop_breach_R: float = 0.0
    stop_breach_rate:            float = 0.0
    early_exit_cost:             float = 0.0
    late_exit_cost:              float = 0.0
    n_tail_capture_defined:      int   = 0
    n_loss_avoided_defined:      int   = 0
    production_decision:         str   = ""
    decision_reason:             str   = ""


# ─────────────────────────────────────────────────────────────────────
#  Trade pairing (ENTRY -> next EXIT, single-slot sleeve)
# ─────────────────────────────────────────────────────────────────────

def pair_trades(all_signals: List[dict]) -> List[dict]:
    pairs: List[dict] = []
    pending: Optional[dict] = None
    for sig in all_signals:
        if sig["type"] == "ENTRY":
            pending = sig
        elif sig["type"] == "EXIT" and pending is not None:
            pairs.append({
                "symbol":     pending["symbol"],
                "entry_date": pending["date"],
                "exit_date":  sig["date"],
                "entry_price": pending["price"],
                "exit_price":  sig["price"],
                "hold_days":   sig["hold_days"],
                "reason":      sig.get("reason", ""),
            })
            pending = None
    return pairs


# ─────────────────────────────────────────────────────────────────────
#  Per-trade counterfactual attribution
# ─────────────────────────────────────────────────────────────────────

def attribute_trade(
    trade_id: int,
    pair: dict,
    close_series: np.ndarray,
    date_to_idx: Dict[str, int],
) -> Optional[TradeAttribution]:
    ei = date_to_idx.get(pair["entry_date"])
    xi = date_to_idx.get(pair["exit_date"])
    if ei is None or xi is None or xi <= ei:
        return None

    ep = pair["entry_price"]
    xp = pair["exit_price"]
    if ep <= 0 or xp <= 0:
        return None

    # Fills with production cost model (Study20 convention)
    entry_fill = ep * (1.0 + SLIPPAGE)
    exit_fill  = xp * (1.0 - SLIPPAGE)
    cost_basis = entry_fill * (1.0 + COMMISSION)
    proceeds   = exit_fill  * (1.0 - COMMISSION)
    actual_return = proceeds / cost_basis - 1.0
    actual_R      = actual_return / R_UNIT_PCT

    n = len(close_series)
    post_end = min(n, xi + LOOKAHEAD + 1)
    post_window = close_series[xi:post_end]   # includes exit day itself (idx xi)
    post_window = post_window[~np.isnan(post_window)]

    def _ret_at(offset: int) -> Optional[float]:
        idx = xi + offset
        if idx >= n or np.isnan(close_series[idx]):
            return None
        return round(float(close_series[idx] / xp - 1.0), 6)

    cf_5d  = _ret_at(5)
    cf_10d = _ret_at(10)
    cf_20d = _ret_at(20)

    if len(post_window) > 0:
        max_post = float(np.max(post_window))
        min_post = float(np.min(post_window))
        peak_offset = int(np.argmax(post_window))
    else:
        max_post = xp
        min_post = xp
        peak_offset = 0

    cf_max_gain = round(max_post / xp - 1.0, 6)
    cf_max_loss = round(min_post / xp - 1.0, 6)
    cf_peak_day = peak_offset

    # Hypothetical worst R if the position had stayed open through the post-exit window
    cf_stop_breach_R = round((min_post - ep) / ep / R_UNIT_PCT, 4)

    # Extended window: entry -> exit+LOOKAHEAD (for tail_capture denominator)
    ext_end = min(n, xi + LOOKAHEAD + 1)
    ext_window = close_series[ei:ext_end]
    ext_window = ext_window[~np.isnan(ext_window)]
    max_ext = float(np.max(ext_window)) if len(ext_window) > 0 else xp
    min_ext = float(np.min(ext_window)) if len(ext_window) > 0 else ep

    max_gain_from_entry = (max_ext - ep) / ep
    max_future_profit = max(max_gain_from_entry, actual_return)
    # tail_capture answers "did the exit preserve the profit tail" — only meaningful for
    # WINNING trades (actual_return > 0). For losing trades there is no profit tail to
    # capture; the relevant question for those is loss_avoided_ratio instead. Computing
    # tail_capture on losers (loss / incidental small rally elsewhere) produces unstable,
    # unbounded ratios with no interpretable meaning, so they are excluded (None).
    tail_capture = (
        round(actual_return / max_future_profit, 4)
        if (actual_return > 0 and max_future_profit > 1e-6) else None
    )

    future_loss_after_exit = max(0.0, -cf_max_loss)            # avoided by exiting (post-exit decline)
    max_future_loss = max(0.0, (ep - min_ext) / ep)            # worst decline from entry over full window
    loss_avoided_ratio = (
        round(future_loss_after_exit / max_future_loss, 4)
        if max_future_loss > 1e-6 else None
    )

    if tail_capture is not None and loss_avoided_ratio is not None:
        exit_quality_score = round(0.5 * tail_capture + 0.5 * loss_avoided_ratio, 4)
    elif tail_capture is not None:
        exit_quality_score = tail_capture
    elif loss_avoided_ratio is not None:
        exit_quality_score = loss_avoided_ratio
    else:
        exit_quality_score = None

    # In-trade MFE / late-exit giveback (entry -> exit, actual hold only)
    hold_window = close_series[ei:xi + 1]
    hold_window = hold_window[~np.isnan(hold_window)]
    in_trade_mfe = (float(np.max(hold_window)) - ep) / ep if len(hold_window) > 0 else 0.0
    late_exit_giveback = round(max(0.0, in_trade_mfe - actual_return), 6)

    return TradeAttribution(
        trade_id=trade_id,
        symbol=pair["symbol"],
        entry_date=pair["entry_date"],
        exit_date=pair["exit_date"],
        hold_days=pair["hold_days"],
        entry_price=round(ep, 2),
        exit_price=round(xp, 2),
        actual_return=round(actual_return, 6),
        actual_R=round(actual_R, 4),
        counterfactual_ret_5d=cf_5d,
        counterfactual_ret_10d=cf_10d,
        counterfactual_ret_20d=cf_20d,
        counterfactual_max_gain=cf_max_gain,
        counterfactual_max_loss=cf_max_loss,
        counterfactual_peak_day=cf_peak_day,
        counterfactual_stop_breach_R=cf_stop_breach_R,
        tail_capture=tail_capture,
        profit_left=round(max(0.0, cf_max_gain), 6),
        loss_avoided_ratio=loss_avoided_ratio,
        exit_quality_score=exit_quality_score,
        in_trade_mfe=round(in_trade_mfe, 6),
        late_exit_giveback=late_exit_giveback,
    )


# ─────────────────────────────────────────────────────────────────────
#  Aggregation + decision
# ─────────────────────────────────────────────────────────────────────

def aggregate(trades: List[TradeAttribution]) -> Study21Result:
    r = Study21Result()
    r.trade_count = len(trades)
    if r.trade_count == 0:
        r.production_decision = "KEEP_MONITOR"
        r.decision_reason = "trade_count=0"
        return r

    tail_vals = [t.tail_capture for t in trades if t.tail_capture is not None]
    loss_vals = [t.loss_avoided_ratio for t in trades if t.loss_avoided_ratio is not None]
    eqs_vals  = [t.exit_quality_score for t in trades if t.exit_quality_score is not None]

    r.tail_capture = round(float(np.mean(tail_vals)), 4) if tail_vals else 0.0
    r.profit_left  = round(float(np.mean([t.profit_left for t in trades])), 4)
    r.loss_avoided_ratio = round(float(np.mean(loss_vals)), 4) if loss_vals else 0.0
    r.exit_quality_score = round(float(np.mean(eqs_vals)), 4) if eqs_vals else 0.0
    r.counterfactual_peak_day = round(float(np.mean([t.counterfactual_peak_day for t in trades])), 2)
    r.counterfactual_stop_breach_R = round(float(np.mean([t.counterfactual_stop_breach_R for t in trades])), 4)
    r.stop_breach_rate = round(
        sum(1 for t in trades if t.counterfactual_stop_breach_R < -STOP_LOSS_R) / r.trade_count, 4
    )
    r.early_exit_cost = round(float(np.mean([t.profit_left for t in trades])), 4)
    r.late_exit_cost  = round(float(np.mean([t.late_exit_giveback for t in trades])), 4)
    r.n_tail_capture_defined = len(tail_vals)
    r.n_loss_avoided_defined = len(loss_vals)

    if (r.tail_capture >= KEEP_EXIT_TAIL_MIN and
            r.profit_left <= KEEP_EXIT_PROFIT_LEFT_MAX and
            r.trade_count >= KEEP_EXIT_TRADE_MIN):
        r.production_decision = "KEEP_EXIT"
        r.decision_reason = (
            f"tail_capture={r.tail_capture:.1%}>={KEEP_EXIT_TAIL_MIN:.0%} AND "
            f"profit_left={r.profit_left:.1%}<={KEEP_EXIT_PROFIT_LEFT_MAX:.0%} AND "
            f"trade_count={r.trade_count}>={KEEP_EXIT_TRADE_MIN}"
        )
    elif r.tail_capture < RESEARCH_TAIL_MAX and r.profit_left > RESEARCH_PROFIT_LEFT_MIN:
        r.production_decision = "RESEARCH_EXIT"
        r.decision_reason = (
            f"tail_capture={r.tail_capture:.1%}<{RESEARCH_TAIL_MAX:.0%} AND "
            f"profit_left={r.profit_left:.1%}>{RESEARCH_PROFIT_LEFT_MIN:.0%}"
        )
    else:
        r.production_decision = "KEEP_MONITOR"
        if r.trade_count < KEEP_EXIT_TRADE_MIN:
            r.decision_reason = f"trade_count={r.trade_count}<{KEEP_EXIT_TRADE_MIN}"
        else:
            r.decision_reason = (
                f"tail_capture={r.tail_capture:.1%} in [{MONITOR_TAIL_LO:.0%},{MONITOR_TAIL_HI:.0%}] "
                f"or borderline (sample accumulation continues)"
            )
    return r


# ─────────────────────────────────────────────────────────────────────
#  Output writers
# ─────────────────────────────────────────────────────────────────────

def write_trade_csv(trades: List[TradeAttribution]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not trades:
        return
    with TRADE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys()))
        w.writeheader()
        for t in trades:
            w.writerow(asdict(t))
    print(f"[CSV] {TRADE_CSV}")


def write_md_report(result: Study21Result, trades: List[TradeAttribution]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    w = L.append

    w("# Study21 Exit Attribution Audit")
    w("")
    w("作成日: 2026-06-23  |  観測専用 / Exit変更禁止 / 収益最適化禁止")
    w("")
    w("**Strategy**: Study9 Case B (FROZEN)  **Entry**: 現行production Entry  "
      "**Exit**: RSR<90  **Capital/Authority/Execution**: 現行production configuration  "
      "**Governance**: annual_rebalance")
    w("")
    w(f"**観測ウィンドウ**: {OBS_START} → {OBS_END}  (全完了トレード, lookahead={LOOKAHEAD}営業日)")
    w("")
    w("**Research Question**: 現行ExitはProfit tailを保持しつつリスク制御機構として機能しているか？")
    w("")
    w("---")
    w("## Methodology（定義）")
    w("")
    w("- `tail_capture` = actual_return / max_future_profit（勝ちトレードのみ定義）")
    w("  - max_future_profit = max(actual_return, entry→exit+20d 区間の最大ゲイン)")
    w("  - 負けトレードはprofit tailが存在しないため tail_capture=None（集計から除外、"
      "loss_avoided_ratioで評価）")
    w("- `profit_left` = counterfactual_max_gain（exit後20d以内に残っていた上昇余地、負値は0扱い）")
    w("- `loss_avoided_ratio` = future_loss_after_exit / max_future_loss")
    w("  - future_loss_after_exit = exit後20d以内に実際に発生した下落（exitで回避できた分）")
    w("  - max_future_loss = entry→exit+20d 区間で発生しえた最大下落（entry比）")
    w("  - 注: 分子はexit価格基準、分母はentry価格基準のため、exit時点で含み益が大きいトレードでは"
      "loss_avoided_ratio>100%になり得る（exit後の戻しがentry比の最悪下落より大きい場合。算出エラーではない）")
    w("- `exit_quality_score` = 0.5×tail_capture + 0.5×loss_avoided_ratio（両方定義されている場合）")
    w("- `counterfactual_stop_breach_R` = exit後にholdを継続した場合の最悪R（entry基準, R_UNIT=2%）")
    w("- `early_exit_cost` = 全トレード平均のprofit_left（早すぎるExitで失った上昇）")
    w("- `late_exit_cost` = 全トレード平均のlate_exit_giveback = max(0, in_trade_MFE - actual_return)"
      "（Exit前にpeakから戻された分 = 遅すぎるExitのコスト）")
    w("- 反事実パスはraw close（slippage/commission未適用の理論パス）。実際の約定はSLIPPAGE/COMMISSION適用済み。")
    w("")
    w("**Restrictions**: REPLACE_EXIT prohibited. Exit redesignはここでは提案しない。")
    w("")

    w("---")
    w("## Executive Summary")
    w("")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| trade_count | {result.trade_count} |")
    w(f"| tail_capture (avg, n={result.n_tail_capture_defined}) | {result.tail_capture:.1%} |")
    w(f"| profit_left (avg) | {result.profit_left:.1%} |")
    w(f"| loss_avoided_ratio (avg, n={result.n_loss_avoided_defined}) | {result.loss_avoided_ratio:.1%} |")
    w(f"| exit_quality_score (avg) | {result.exit_quality_score:.3f} |")
    w(f"| counterfactual_peak_day (avg) | {result.counterfactual_peak_day:.1f}d |")
    w(f"| counterfactual_stop_breach_R (avg) | {result.counterfactual_stop_breach_R:+.3f}R |")
    w(f"| stop_breach_rate (< -{STOP_LOSS_R}R) | {result.stop_breach_rate:.1%} |")
    w(f"| early_exit_cost | {result.early_exit_cost:.1%} |")
    w(f"| late_exit_cost | {result.late_exit_cost:.1%} |")
    w("")
    w(f"**production_decision: {result.production_decision}**")
    w("")
    w(f"判定理由: {result.decision_reason}")
    w("")

    w("---")
    w("## Decision Rules（適用基準）")
    w("")
    w("| 判定 | 条件 |")
    w("|---|---|")
    w(f"| KEEP_EXIT | tail_capture≥{KEEP_EXIT_TAIL_MIN:.0%} AND profit_left≤{KEEP_EXIT_PROFIT_LEFT_MAX:.0%} AND trade_count≥{KEEP_EXIT_TRADE_MIN} |")
    w(f"| KEEP_MONITOR | tail_capture {MONITOR_TAIL_LO:.0%}–{MONITOR_TAIL_HI:.0%} OR trade_count<{KEEP_EXIT_TRADE_MIN} |")
    w(f"| RESEARCH_EXIT | tail_capture<{RESEARCH_TAIL_MAX:.0%} AND profit_left>{RESEARCH_PROFIT_LEFT_MIN:.0%} |")
    w("")

    w("---")
    w("## Trade Log（全件）")
    w("")
    w("| # | 銘柄 | Entry | Exit | hold | actual_ret | actual_R | tail_capture | "
      "profit_left | loss_avoided | EQS | peak_day | stop_breach_R |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in trades:
        tc = f"{t.tail_capture:.1%}" if t.tail_capture is not None else "—"
        lar = f"{t.loss_avoided_ratio:.1%}" if t.loss_avoided_ratio is not None else "—"
        eqs = f"{t.exit_quality_score:.3f}" if t.exit_quality_score is not None else "—"
        w(f"| {t.trade_id} | {t.symbol} | {t.entry_date} | {t.exit_date} | {t.hold_days}d | "
          f"{t.actual_return:+.2%} | {t.actual_R:+.2f}R | {tc} | {t.profit_left:.1%} | "
          f"{lar} | {eqs} | {t.counterfactual_peak_day}d | {t.counterfactual_stop_breach_R:+.2f}R |")
    w("")

    w("---")
    w("## 最終判定")
    w("")
    w(f"| 指標 | 値 |")
    w(f"|---|---|")
    w(f"| trade_count | {result.trade_count} |")
    w(f"| tail_capture | {result.tail_capture:.1%} |")
    w(f"| profit_left | {result.profit_left:.1%} |")
    w(f"| loss_avoided_ratio | {result.loss_avoided_ratio:.1%} |")
    w(f"| exit_quality_score | {result.exit_quality_score:.3f} |")
    w(f"| **production_decision** | **{result.production_decision}** |")
    w("")
    if result.production_decision == "KEEP_EXIT":
        w("→ 90営業日継続運用。次回再評価まで現行Exit維持。")
    elif result.production_decision == "KEEP_MONITOR":
        w("→ サンプル蓄積継続。trade_count/tail_captureが閾値を超えるまでExit変更検討せず。")
    else:
        w("→ Study22 Soft Stop Audit へ。REPLACE_EXIT提案はattribution evidence収集完了まで禁止。")
    w("")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[MD] {REPORT_MD}")


def append_telemetry(result: Study21Result) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {"study": "study21", **asdict(result)}
        with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("[Study21] Exit Attribution Audit")
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

    print("[2/4] 価格マトリクス構築...")
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
    sym_active_mat = (
        None if sym_active_df is None
        else _take(sym_active_df, common_dates, active_syms, dtype=np.float32, fill_value=1.0)
    )
    mkt_ret1 = (
        None if topix_close is None
        else _take(topix_close.pct_change(), common_dates, dtype=np.float32, fill_value=0.0)
    )

    print("[3/4] Study9 Case B シグナル生成 (FROZEN, 観測のみ)...")
    all_signals = generate_signals(
        common_dates, active_syms, sym_to_i,
        open_mat, close_mat, rsr_mat, sym_active_mat, mkt_ret1,
    )
    pairs = pair_trades(all_signals)
    print(f"[3/4] completed_trades={len(pairs)}")

    print("[4/4] Per-trade counterfactual attribution...")
    trades: List[TradeAttribution] = []
    for tid, pair in enumerate(pairs, start=1):
        si = sym_to_i[pair["symbol"]]
        close_series = close_mat[:, si]
        attr = attribute_trade(tid, pair, close_series, date_to_idx)
        if attr is not None:
            trades.append(attr)

    result = aggregate(trades)

    print(f"  trade_count             = {result.trade_count}")
    print(f"  tail_capture            = {result.tail_capture:.1%}")
    print(f"  profit_left             = {result.profit_left:.1%}")
    print(f"  loss_avoided_ratio      = {result.loss_avoided_ratio:.1%}")
    print(f"  exit_quality_score      = {result.exit_quality_score:.3f}")
    print(f"  counterfactual_peak_day = {result.counterfactual_peak_day:.1f}d")
    print(f"  stop_breach_rate        = {result.stop_breach_rate:.1%}")
    print(f"  early_exit_cost         = {result.early_exit_cost:.1%}")
    print(f"  late_exit_cost          = {result.late_exit_cost:.1%}")
    print(f"  production_decision     = {result.production_decision}")
    print(f"  decision_reason         = {result.decision_reason}")

    write_trade_csv(trades)
    write_md_report(result, trades)
    append_telemetry(result)

    print()
    print("=" * 68)
    print(f"production_decision = {result.production_decision}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
