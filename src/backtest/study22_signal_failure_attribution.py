"""
src/backtest/study22_signal_failure_attribution.py — Study22 Signal Failure Attribution

Objective: quantify whether losing trades (actual_R < 0) are NORMAL_LOSS (structural,
expected statistical variance) or AVOIDABLE_LOSS (signal-timing related: REVERSAL_LOSS /
FALSE_BREAKOUT / HIGH_VOL_ENTRY / LATE_ENTRY) or REGIME_SHIFT (macro-driven).

Accountability-only study. Does NOT modify Entry / Exit / sizing / authority / execution.
Does NOT optimize profit. No Entry or Exit redesign is proposed here regardless of finding.

Fixed configuration (FROZEN, reused from Study21/Study20/Study9 Case B):
  Strategy : Study9 Case B  (src.backtest.study20_limited_live_risk_envelope.generate_signals)
  Entry    : RSR in [92,95), days_cross90<=5, rsr_slope_5d<=5  (current production entry)
  Exit     : RSR<90 (+ MKT_SHOCK structural exit, unchanged)
  Capital / Authority / Execution / Governance: current production configuration
  Scope    : all trades with actual_R<0, 2018-01-01..2025-12-31 (full observation window)

Per-trade measurements:
  symbol, entry_date, exit_date, actual_R, hold_days, days_to_loss, entry_rsr, entry_slope,
  entry_rank, market_regime, peak_before_loss, max_drawdown_before_exit,
  counterfactual_hold_5d/10d/20d, recovery_probability

Definitions:
  - days_to_loss            = first trading day offset (from entry) where unrealized
                              return turns negative; hold_days if it never went negative
                              before exit (i.e. loss only realized at the exit fill).
  - entry_rank              = cross-sectional RSR rank (1=strongest) among all active
                              universe symbols on the entry date (lookahead-free).
  - market_regime           = "Bear" if is_bear_regime(topix_close up to entry_date) else
                              "Bull" (src.strategy.universe.is_bear_regime, point-in-time).
  - peak_before_loss        = max unrealized gain (%) reached at any point during the
                              hold (entry->exit), i.e. in-trade MFE.
  - max_drawdown_before_exit= max decline (%) from that peak to the exit, within the hold
                              (0 if the peak was at/after the exit day itself).
  - counterfactual_hold_Nd  = hypothetical total trade return (%, from entry_price) had
                              the position been held N trading days past the actual exit
                              (raw close, no slippage/commission — theoretical path).
  - recovery_probability    = fraction of {5d,10d,20d} checkpoints where
                              counterfactual_hold_Nd > actual_return (staying in would
                              have been better than the actual exit, price-wise).

Classification (priority order, exactly one label per trade):
  1. REGIME_SHIFT    : exit_reason==MKT_SHOCK, OR market regime flipped Bull->Bear during
                       the hold (macro-driven, not an entry/exit signal quality issue).
  2. HIGH_VOL_ENTRY   : entry-day volatility (10d realized std of daily returns) is in the
                       top quartile across all entries in this sample (chased an
                       overextended/volatile move).
  3. LATE_ENTRY       : days_cross90 at entry >= 4 (entered late within the allowed [1,5]
                       window — the move was already mature).
  4. FALSE_BREAKOUT   : peak_before_loss<=2% and hold_days<=5 (signal fired, no follow-
                       through ever materialized — a pure false signal).
  5. REVERSAL_LOSS    : peak_before_loss>3% and max_drawdown_before_exit>50% (trade moved
                       favorably then gave it back — a round-trip reversal).
  6. NORMAL_LOSS      : fallback — ordinary statistical variance, no identifiable failure
                       mode above.

Aggregate ratios (the 6 categories partition exactly; ratios sum to 100% of losers):
  structural_loss_ratio = NORMAL_LOSS / n_losers
  regime_loss_ratio     = REGIME_SHIFT / n_losers
  avoidable_loss_ratio  = (REVERSAL_LOSS+FALSE_BREAKOUT+HIGH_VOL_ENTRY+LATE_ENTRY)/n_losers
  signal_failure_rate   = (FALSE_BREAKOUT+HIGH_VOL_ENTRY+LATE_ENTRY)/n_losers
                          (entry-quality subset of avoidable; excludes REVERSAL_LOSS which
                          is an exit-timing-given-a-good-entry issue, not an entry failure)
  recovery_rate         = mean(recovery_probability) across all losers

Decision (recommend_entry_change — accountability label only, no redesign proposed):
  KEEP_ENTRY     : avoidable_loss_ratio<=40% AND n_losers>=15
  RESEARCH_ENTRY : avoidable_loss_ratio>=60% AND signal_failure_rate>=40% AND n_losers>=15
  MONITOR_ENTRY  : otherwise (including insufficient sample)

Restrictions: Entry redesign prohibited. Exit redesign prohibited. Output is
accountability/attribution only.

Run:
    cd C:/ai-trading
    python src/backtest/study22_signal_failure_attribution.py
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

from src.backtest.capital_allocation_abc import load_data, _take
from src.backtest.study20_limited_live_risk_envelope import (
    generate_signals, _cross90, _slope5,
    SLIPPAGE, COMMISSION, R_UNIT_PCT,
)
from src.backtest.study21_exit_attribution_audit import pair_trades
from src.strategy.universe import is_bear_regime
from src.config_loader import load_strategy_config

OBS_START = "2018-01-01"
OBS_END   = "2025-12-31"
LOOKAHEAD = 20   # trading days post-exit counterfactual horizon
VOL_WINDOW = 10  # trading days for entry-volatility proxy

REPORT_DIR    = Path("reports")
REPORT_MD     = REPORT_DIR / "study22_signal_failure_attribution.md"
TRADE_CSV     = REPORT_DIR / "study22_loss_attribution.csv"
LOG_DIR       = Path("logs")
TELEMETRY_LOG = LOG_DIR / "study22_telemetry.jsonl"

CAT_REGIME_SHIFT  = "REGIME_SHIFT"
CAT_HIGH_VOL      = "HIGH_VOL_ENTRY"
CAT_LATE_ENTRY    = "LATE_ENTRY"
CAT_FALSE_BO      = "FALSE_BREAKOUT"
CAT_REVERSAL      = "REVERSAL_LOSS"
CAT_NORMAL        = "NORMAL_LOSS"
ALL_CATEGORIES = [CAT_REGIME_SHIFT, CAT_HIGH_VOL, CAT_LATE_ENTRY,
                   CAT_FALSE_BO, CAT_REVERSAL, CAT_NORMAL]

LATE_ENTRY_D90_MIN     = 4      # of [1,5] allowed window
FALSE_BO_PEAK_MAX      = 0.02
FALSE_BO_HOLD_MAX      = 5
REVERSAL_PEAK_MIN       = 0.03
REVERSAL_DRAWDOWN_MIN   = 0.50

KEEP_AVOIDABLE_MAX        = 0.40
RESEARCH_AVOIDABLE_MIN    = 0.60
RESEARCH_SIGNAL_FAIL_MIN  = 0.40
MIN_SAMPLE                = 15


@dataclass
class LossAttribution:
    trade_id:                  int
    symbol:                    str
    entry_date:                str
    exit_date:                 str
    actual_R:                  float
    hold_days:                 int
    days_to_loss:               int
    entry_rsr:                  float
    entry_slope:                float
    entry_rank:                 int
    market_regime:               str
    peak_before_loss:             float
    max_drawdown_before_exit:     float
    counterfactual_hold_5d:        Optional[float]
    counterfactual_hold_10d:       Optional[float]
    counterfactual_hold_20d:       Optional[float]
    recovery_probability:          float
    entry_vol_10d:                 float
    exit_reason:                   str
    category:                      str = ""


@dataclass
class Study22Result:
    trade_count:              int = 0   # losers analyzed
    avoidable_loss_ratio:     float = 0.0
    structural_loss_ratio:    float = 0.0
    regime_loss_ratio:        float = 0.0
    signal_failure_rate:      float = 0.0
    recovery_rate:            float = 0.0
    loss_source_breakdown:    Dict[str, int] = field(default_factory=dict)
    recommend_entry_change:   str = ""
    decision_reason:          str = ""


# ─────────────────────────────────────────────────────────────────────
#  Per-trade attribution
# ─────────────────────────────────────────────────────────────────────

def attribute_loss(
    trade_id: int,
    pair: dict,
    close_series: np.ndarray,
    rsr_series: np.ndarray,
    slope5_series: np.ndarray,
    cross90_series: np.ndarray,
    rsr_mat: np.ndarray,
    sym_active_mat: Optional[np.ndarray],
    si: int,
    date_to_idx: Dict[str, int],
    topix_close: Optional[pd.Series],
    common_dates: pd.DatetimeIndex,
) -> Optional[LossAttribution]:
    ei = date_to_idx.get(pair["entry_date"])
    xi = date_to_idx.get(pair["exit_date"])
    if ei is None or xi is None or xi <= ei:
        return None

    ep = pair["entry_price"]
    xp = pair["exit_price"]
    if ep <= 0 or xp <= 0:
        return None

    entry_fill = ep * (1.0 + SLIPPAGE)
    exit_fill  = xp * (1.0 - SLIPPAGE)
    cost_basis = entry_fill * (1.0 + COMMISSION)
    proceeds   = exit_fill  * (1.0 - COMMISSION)
    actual_return = proceeds / cost_basis - 1.0
    actual_R      = actual_return / R_UNIT_PCT

    if actual_R >= 0:
        return None   # scope: actual_R < 0 only

    hold_days = pair["hold_days"]

    # ── In-hold path (entry -> exit inclusive) ──────────────────────────
    hold_path = close_series[ei:xi + 1]
    valid_mask = ~np.isnan(hold_path)
    hold_path_v = hold_path[valid_mask]
    hold_rets = (hold_path_v - ep) / ep if len(hold_path_v) > 0 else np.array([0.0])

    # days_to_loss: first offset where unrealized return < 0
    days_to_loss = hold_days
    for off, ret in enumerate(hold_rets):
        if ret < 0:
            days_to_loss = off
            break

    peak_idx_in_hold = int(np.argmax(hold_rets))
    peak_before_loss = float(hold_rets[peak_idx_in_hold])
    peak_price = float(hold_path_v[peak_idx_in_hold]) if len(hold_path_v) > 0 else ep

    # drawdown from that peak to the exit (0 if peak is at/after exit day)
    if peak_idx_in_hold < len(hold_path_v) - 1:
        trough_after_peak = float(np.min(hold_path_v[peak_idx_in_hold:]))
        max_drawdown_before_exit = (
            (peak_price - trough_after_peak) / peak_price if peak_price > 0 else 0.0
        )
    else:
        max_drawdown_before_exit = 0.0

    # ── Entry-time features ─────────────────────────────────────────────
    entry_rsr   = float(rsr_series[ei])
    entry_slope = float(slope5_series[ei])

    if sym_active_mat is not None:
        active_row = sym_active_mat[ei] >= 0.5
    else:
        active_row = np.ones(rsr_mat.shape[1], dtype=bool)
    rsr_today = rsr_mat[ei]
    eligible_rsr = rsr_today[active_row]
    entry_rank = int(np.sum(eligible_rsr > entry_rsr) + 1) if len(eligible_rsr) > 0 else 1

    # entry-day volatility proxy: 10d realized std of daily returns ending at entry
    vol_start = max(0, ei - VOL_WINDOW)
    vol_path = close_series[vol_start:ei + 1]
    vol_path = vol_path[~np.isnan(vol_path)]
    if len(vol_path) >= 3:
        rets = np.diff(vol_path) / vol_path[:-1]
        entry_vol_10d = float(np.std(rets))
    else:
        entry_vol_10d = 0.0

    # market regime at entry (point-in-time, no lookahead)
    market_regime = "Bull"
    regime_shift = False
    if topix_close is not None:
        entry_dt = common_dates[ei]
        exit_dt  = common_dates[xi]
        topix_at_entry = topix_close.loc[:entry_dt]
        topix_at_exit  = topix_close.loc[:exit_dt]
        bear_at_entry = is_bear_regime(topix_at_entry) if len(topix_at_entry) > 0 else False
        bear_at_exit  = is_bear_regime(topix_at_exit)  if len(topix_at_exit)  > 0 else False
        market_regime = "Bear" if bear_at_entry else "Bull"
        regime_shift = (not bear_at_entry) and bear_at_exit

    exit_reason = pair.get("reason", "")

    # ── Counterfactual forward path (raw close, post actual exit) ──────
    n = len(close_series)

    def _cf_hold(offset: int) -> Optional[float]:
        idx = xi + offset
        if idx >= n or np.isnan(close_series[idx]):
            return None
        return round(float(close_series[idx] / ep - 1.0), 6)

    cf_5d  = _cf_hold(5)
    cf_10d = _cf_hold(10)
    cf_20d = _cf_hold(20)
    checkpoints = [v for v in (cf_5d, cf_10d, cf_20d) if v is not None]
    recovery_probability = (
        round(sum(1 for v in checkpoints if v > actual_return) / len(checkpoints), 4)
        if checkpoints else 0.0
    )

    # ── Classification (priority order, single label) ──────────────────
    category = CAT_NORMAL
    if exit_reason == "MKT_SHOCK" or regime_shift:
        category = CAT_REGIME_SHIFT
    elif entry_vol_10d >= _HIGH_VOL_THRESHOLD[0]:
        category = CAT_HIGH_VOL
    elif int(cross90_series[ei]) >= LATE_ENTRY_D90_MIN:
        category = CAT_LATE_ENTRY
    elif peak_before_loss <= FALSE_BO_PEAK_MAX and hold_days <= FALSE_BO_HOLD_MAX:
        category = CAT_FALSE_BO
    elif peak_before_loss > REVERSAL_PEAK_MIN and max_drawdown_before_exit > REVERSAL_DRAWDOWN_MIN:
        category = CAT_REVERSAL

    return LossAttribution(
        trade_id=trade_id,
        symbol=pair["symbol"],
        entry_date=pair["entry_date"],
        exit_date=pair["exit_date"],
        actual_R=round(actual_R, 4),
        hold_days=hold_days,
        days_to_loss=days_to_loss,
        entry_rsr=round(entry_rsr, 2),
        entry_slope=round(entry_slope, 2),
        entry_rank=entry_rank,
        market_regime=market_regime,
        peak_before_loss=round(peak_before_loss, 6),
        max_drawdown_before_exit=round(max_drawdown_before_exit, 6),
        counterfactual_hold_5d=cf_5d,
        counterfactual_hold_10d=cf_10d,
        counterfactual_hold_20d=cf_20d,
        recovery_probability=recovery_probability,
        entry_vol_10d=round(entry_vol_10d, 6),
        exit_reason=exit_reason,
        category=category,
    )


# Module-level mutable holder so the top-quartile volatility threshold (computed from the
# full sample after a first pass) is available inside attribute_loss without re-plumbing
# the signature. Set once per run before classification, read-only thereafter.
_HIGH_VOL_THRESHOLD = [float("inf")]


# ─────────────────────────────────────────────────────────────────────
#  Aggregation + decision
# ─────────────────────────────────────────────────────────────────────

def aggregate(trades: List[LossAttribution]) -> Study22Result:
    r = Study22Result()
    r.trade_count = len(trades)
    r.loss_source_breakdown = {c: 0 for c in ALL_CATEGORIES}
    if r.trade_count == 0:
        r.recommend_entry_change = "MONITOR_ENTRY"
        r.decision_reason = "trade_count=0"
        return r

    for t in trades:
        r.loss_source_breakdown[t.category] += 1

    n = r.trade_count
    n_normal   = r.loss_source_breakdown[CAT_NORMAL]
    n_regime   = r.loss_source_breakdown[CAT_REGIME_SHIFT]
    n_high_vol = r.loss_source_breakdown[CAT_HIGH_VOL]
    n_late     = r.loss_source_breakdown[CAT_LATE_ENTRY]
    n_false_bo = r.loss_source_breakdown[CAT_FALSE_BO]
    n_reversal = r.loss_source_breakdown[CAT_REVERSAL]

    r.structural_loss_ratio = round(n_normal / n, 4)
    r.regime_loss_ratio     = round(n_regime / n, 4)
    r.avoidable_loss_ratio  = round((n_high_vol + n_late + n_false_bo + n_reversal) / n, 4)
    r.signal_failure_rate   = round((n_high_vol + n_late + n_false_bo) / n, 4)
    r.recovery_rate         = round(float(np.mean([t.recovery_probability for t in trades])), 4)

    if r.trade_count >= MIN_SAMPLE and r.avoidable_loss_ratio <= KEEP_AVOIDABLE_MAX:
        r.recommend_entry_change = "KEEP_ENTRY"
        r.decision_reason = (
            f"avoidable_loss_ratio={r.avoidable_loss_ratio:.1%}<={KEEP_AVOIDABLE_MAX:.0%} "
            f"AND trade_count={r.trade_count}>={MIN_SAMPLE}"
        )
    elif (r.trade_count >= MIN_SAMPLE and
          r.avoidable_loss_ratio >= RESEARCH_AVOIDABLE_MIN and
          r.signal_failure_rate >= RESEARCH_SIGNAL_FAIL_MIN):
        r.recommend_entry_change = "RESEARCH_ENTRY"
        r.decision_reason = (
            f"avoidable_loss_ratio={r.avoidable_loss_ratio:.1%}>={RESEARCH_AVOIDABLE_MIN:.0%} "
            f"AND signal_failure_rate={r.signal_failure_rate:.1%}>={RESEARCH_SIGNAL_FAIL_MIN:.0%}"
        )
    else:
        r.recommend_entry_change = "MONITOR_ENTRY"
        if r.trade_count < MIN_SAMPLE:
            r.decision_reason = f"trade_count={r.trade_count}<{MIN_SAMPLE}"
        else:
            r.decision_reason = (
                f"avoidable_loss_ratio={r.avoidable_loss_ratio:.1%} borderline "
                f"(between KEEP {KEEP_AVOIDABLE_MAX:.0%} and RESEARCH {RESEARCH_AVOIDABLE_MIN:.0%})"
            )
    return r


# ─────────────────────────────────────────────────────────────────────
#  Output writers
# ─────────────────────────────────────────────────────────────────────

def write_trade_csv(trades: List[LossAttribution]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not trades:
        return
    with TRADE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys()))
        w.writeheader()
        for t in trades:
            w.writerow(asdict(t))
    print(f"[CSV] {TRADE_CSV}")


def write_md_report(result: Study22Result, trades: List[LossAttribution]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    w = L.append

    w("# Study22 Signal Failure Attribution")
    w("")
    w("作成日: 2026-06-23  |  説明責任のみ（accountability only）/ Entry変更禁止 / Exit変更禁止 / 収益最適化禁止")
    w("")
    w("**Strategy**: Study9 Case B (FROZEN)  **Entry**: RSR∈[92,95), days_cross90≤5, "
      "rsr_slope_5d≤5  **Exit**: RSR<90  **Capital/Authority/Execution**: 現行production "
      "configuration  **Governance**: annual_rebalance")
    w("")
    w(f"**観測ウィンドウ**: {OBS_START} → {OBS_END}（actual_R<0 全トレード, lookahead={LOOKAHEAD}営業日）")
    w("")
    w("**目的**: 負けトレードが「正常損失」か「回避可能損失」かを定量化する（Entry/Exit再設計は提案しない）")
    w("")

    w("---")
    w("## 分類ロジック（優先順位順・排他的1ラベル）")
    w("")
    w("| 優先 | カテゴリ | 条件 |")
    w("|---|---|---|")
    w("| 1 | REGIME_SHIFT | exit_reason=MKT_SHOCK または hold中にBull→Bear転換 |")
    w(f"| 2 | HIGH_VOL_ENTRY | entry時10d実現volが全トレード中の上位25%（閾値={_HIGH_VOL_THRESHOLD[0]:.4f}） |")
    w(f"| 3 | LATE_ENTRY | entry時days_cross90 ≥ {LATE_ENTRY_D90_MIN}（許容範囲[1,5]の後半） |")
    w(f"| 4 | FALSE_BREAKOUT | peak_before_loss≤{FALSE_BO_PEAK_MAX:.0%} AND hold_days≤{FALSE_BO_HOLD_MAX}d（追随なし） |")
    w(f"| 5 | REVERSAL_LOSS | peak_before_loss>{REVERSAL_PEAK_MIN:.0%} AND drawdown>{REVERSAL_DRAWDOWN_MIN:.0%}（往復負け） |")
    w("| 6 | NORMAL_LOSS | 上記非該当（通常の統計的分散） |")
    w("")
    w("**集計区分**: structural=NORMAL_LOSS / regime=REGIME_SHIFT / "
      "avoidable=REVERSAL+FALSE_BO+HIGH_VOL+LATE（3区分は排他的に合計100%）。"
      "signal_failure_rate=FALSE_BO+HIGH_VOL+LATE（avoidableのうちEntry品質起因のサブセット、"
      "REVERSAL_LOSSは「良いEntryからのExitタイミング」問題として除外）。")
    w("")

    w("---")
    w("## Executive Summary")
    w("")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| trade_count（losers） | {result.trade_count} |")
    w(f"| avoidable_loss_ratio | {result.avoidable_loss_ratio:.1%} |")
    w(f"| structural_loss_ratio | {result.structural_loss_ratio:.1%} |")
    w(f"| regime_loss_ratio | {result.regime_loss_ratio:.1%} |")
    w(f"| signal_failure_rate | {result.signal_failure_rate:.1%} |")
    w(f"| recovery_rate | {result.recovery_rate:.1%} |")
    w("")
    w("**loss_source_breakdown**")
    w("")
    w("| カテゴリ | 件数 | 割合 |")
    w("|---|---|---|")
    for c in ALL_CATEGORIES:
        cnt = result.loss_source_breakdown.get(c, 0)
        pct = cnt / max(1, result.trade_count) * 100
        w(f"| {c} | {cnt} | {pct:.1f}% |")
    w("")
    w(f"**recommend_entry_change: {result.recommend_entry_change}**")
    w("")
    w(f"判定理由: {result.decision_reason}")
    w("")

    w("---")
    w("## Decision Rules（適用基準）")
    w("")
    w("| 判定 | 条件 |")
    w("|---|---|")
    w(f"| KEEP_ENTRY | avoidable_loss_ratio≤{KEEP_AVOIDABLE_MAX:.0%} AND trade_count≥{MIN_SAMPLE} |")
    w(f"| RESEARCH_ENTRY | avoidable_loss_ratio≥{RESEARCH_AVOIDABLE_MIN:.0%} AND signal_failure_rate≥{RESEARCH_SIGNAL_FAIL_MIN:.0%} AND trade_count≥{MIN_SAMPLE} |")
    w(f"| MONITOR_ENTRY | 上記以外（サンプル不足含む） |")
    w("")

    w("---")
    w("## Trade Log（全件）")
    w("")
    w("| # | 銘柄 | Entry | Exit | hold | days_to_loss | actual_R | entry_rsr | entry_slope | "
      "rank | regime | peak | DD | cf5d | cf10d | cf20d | recovery_p | category |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in trades:
        c5  = f"{t.counterfactual_hold_5d:+.2%}"  if t.counterfactual_hold_5d  is not None else "—"
        c10 = f"{t.counterfactual_hold_10d:+.2%}" if t.counterfactual_hold_10d is not None else "—"
        c20 = f"{t.counterfactual_hold_20d:+.2%}" if t.counterfactual_hold_20d is not None else "—"
        w(f"| {t.trade_id} | {t.symbol} | {t.entry_date} | {t.exit_date} | {t.hold_days}d | "
          f"{t.days_to_loss}d | {t.actual_R:+.2f}R | {t.entry_rsr:.1f} | {t.entry_slope:+.1f} | "
          f"{t.entry_rank} | {t.market_regime} | {t.peak_before_loss:+.2%} | "
          f"{t.max_drawdown_before_exit:.1%} | {c5} | {c10} | {c20} | "
          f"{t.recovery_probability:.0%} | {t.category} |")
    w("")

    w("---")
    w("## 最終判定")
    w("")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| trade_count | {result.trade_count} |")
    w(f"| avoidable_loss_ratio | {result.avoidable_loss_ratio:.1%} |")
    w(f"| structural_loss_ratio | {result.structural_loss_ratio:.1%} |")
    w(f"| signal_failure_rate | {result.signal_failure_rate:.1%} |")
    w(f"| loss_source_breakdown | {result.loss_source_breakdown} |")
    w(f"| **recommend_entry_change** | **{result.recommend_entry_change}** |")
    w("")
    w("研究目的は説明責任のみ。Entry/Exit再設計はこのレポートでは提案しない。")
    w("")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[MD] {REPORT_MD}")


def append_telemetry(result: Study22Result) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {"study": "study22", **asdict(result)}
        with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("[Study22] Signal Failure Attribution")
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

    print("[3/4] Study9 Case B シグナル生成 (FROZEN, 観測のみ)...")
    all_signals = generate_signals(
        common_dates, active_syms, sym_to_i,
        open_mat, close_mat, rsr_mat, sym_active_mat, mkt_ret1,
    )
    pairs = pair_trades(all_signals)
    print(f"[3/4] completed_trades={len(pairs)}")

    print("[4/4] 損失アトリビューション（actual_R<0 のみ）...")
    # Pass 1: compute raw entry-volatility for every completed trade to derive the
    # top-quartile threshold (point-in-time features only, no lookahead into outcomes).
    vols: List[float] = []
    for pair in pairs:
        ei = date_to_idx.get(pair["entry_date"])
        if ei is None:
            continue
        si = sym_to_i[pair["symbol"]]
        vol_start = max(0, ei - VOL_WINDOW)
        vol_path = close_mat[vol_start:ei + 1, si]
        vol_path = vol_path[~np.isnan(vol_path)]
        if len(vol_path) >= 3:
            rets = np.diff(vol_path) / vol_path[:-1]
            vols.append(float(np.std(rets)))
    _HIGH_VOL_THRESHOLD[0] = float(np.percentile(vols, 75)) if vols else float("inf")

    trades: List[LossAttribution] = []
    for tid, pair in enumerate(pairs, start=1):
        si = sym_to_i[pair["symbol"]]
        attr = attribute_loss(
            tid, pair, close_mat[:, si], rsr_mat[:, si], slope5_mat[:, si],
            cross90_mat[:, si], rsr_mat, sym_active_mat, si, date_to_idx,
            topix_close, common_dates,
        )
        if attr is not None:
            trades.append(attr)

    result = aggregate(trades)

    print(f"  trade_count           = {result.trade_count}")
    print(f"  avoidable_loss_ratio  = {result.avoidable_loss_ratio:.1%}")
    print(f"  structural_loss_ratio = {result.structural_loss_ratio:.1%}")
    print(f"  regime_loss_ratio     = {result.regime_loss_ratio:.1%}")
    print(f"  signal_failure_rate   = {result.signal_failure_rate:.1%}")
    print(f"  recovery_rate         = {result.recovery_rate:.1%}")
    print(f"  loss_source_breakdown = {result.loss_source_breakdown}")
    print(f"  recommend_entry_change= {result.recommend_entry_change}")
    print(f"  decision_reason       = {result.decision_reason}")

    write_trade_csv(trades)
    write_md_report(result, trades)
    append_telemetry(result)

    print()
    print("=" * 68)
    print(f"recommend_entry_change = {result.recommend_entry_change}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
