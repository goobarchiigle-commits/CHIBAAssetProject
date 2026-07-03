"""
src/backtest/study23_signal_failure_decomposition.py — Study23 Signal Failure Decomposition

Objective: verify whether the FALSE_BREAKOUT and HIGH_VOL_ENTRY failure modes detected in
Study22 are explainable using EXISTING features alone. Accountability-only study.

Prohibited: new signal design, Entry change, Exit change, sizing change, capital change,
improvement proposals. Output is explainability classification only.

Fixed configuration (FROZEN, reused from Study22/Study21/Study20/Study9 Case B):
  Strategy : Study9 Case B  (generate_signals imported, not reimplemented)
  Entry    : RSR in [92,95), days_cross90<=5, rsr_slope_5d<=5
  Exit     : RSR<90 (+ MKT_SHOCK structural exit, unchanged)
  Capital / Authority / Execution / Governance: current production configuration

Scope: trades classified FALSE_BREAKOUT or HIGH_VOL_ENTRY in Study22 (label=1, "bad"),
       plus ALL winning trades (actual_R>=0) from the same Study9 Case B sample (label=0).
       (NORMAL_LOSS / LATE_ENTRY / REGIME_SHIFT / REVERSAL_LOSS trades are out of scope —
       Study22 already attributed those to causes other than the two failure modes audited
       here.)

Candidate features (existing, no new feature engineering beyond standard transforms):
  entry_rsr, rsr_slope_5d, days_cross90, atr_z, gap_pct, volume_z, market_regime, sector, rank
  - atr_z   = (ATR14/Close) z-scored against its own trailing 60d history (per symbol)
  - gap_pct = (entry-day Open - prior Close) / prior Close (overnight gap at the fill date)
  - volume_z= Volume z-scored against its own trailing 20d history (per symbol)
  - rank    = entry_rank (cross-sectional RSR rank among active universe), from Study22

Per-feature measurement:
  - bin_breakdown (tercile or category bins): conditional_loss_rate per bin = n_bad/n
  - lift                = max bin conditional_loss_rate / overall base rate
  - mutual_information  = I(bin; label) on the tercile/category binning
  - best single-condition rule (threshold or category, whichever direction maximizes
    precision over the scoped sample): precision, recall, coverage
    (conditional_loss_rate of that best rule == its precision, reported as such)

Aggregate:
  - top_predictors      = features ranked by mutual_information (desc), top 3
  - feature_interactions= AND-combination of the top-2 features' best rules vs each alone
  - best_rule           = whichever of {each feature's best single rule, the 2-way
                          interaction rule} maximizes precision (ties: higher coverage) —
                          used for the counterfactual removal calc and final decision
  - counterfactual_removed_loss   = R-weighted recall: sum(|R| of flagged bad trades) /
                                    sum(|R| of all bad trades)            ("loss_explainability")
  - counterfactual_removed_profit = R-weighted false-positive cost: sum(R of flagged
                                    winners) / sum(R of all winners)
  - alpha_retention      = 1 - counterfactual_removed_profit             ("profit_explainability")

Decision (research_decision — explainability label only, no fix proposed):
  EXPLAINABLE            : precision>=70% AND coverage>=50% AND
                            counterfactual_removed_profit<=20% AND alpha_retention>=80%
  PARTIALLY_EXPLAINABLE  : precision>=60% AND coverage>=40%
  NEW_SIGNAL_REQUIRED    : otherwise (existing features insufficient — does NOT prescribe
                            what a new signal should be; that is explicitly prohibited here)

Restrictions: no new Entry/Exit rule proposed. No improvement proposal generated.
Accountability/explainability only.

Run:
    cd C:/ai-trading
    python src/backtest/study23_signal_failure_decomposition.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from src.backtest.study22_signal_failure_attribution import (
    attribute_loss, _HIGH_VOL_THRESHOLD, VOL_WINDOW,
    CAT_FALSE_BO, CAT_HIGH_VOL,
)
from src.config_loader import load_strategy_config

OBS_START = "2018-01-01"
OBS_END   = "2025-12-31"
TARGET_CATEGORIES = {CAT_FALSE_BO, CAT_HIGH_VOL}

# Anti-overfitting floor: without this, the precision-maximizing search degenerates to
# picking single-point splits (e.g. one symbol's data artifact) that trivially hit 100%
# precision on 1-2 trades out of 27. A rule must cover a non-trivial share of the sample
# to be considered a real explanatory split, not curve-fitting to noise.
MIN_RULE_COVERAGE = 0.15

REPORT_DIR    = Path("reports")
REPORT_MD     = REPORT_DIR / "study23_signal_failure_decomposition.md"
TRADE_CSV     = REPORT_DIR / "study23_feature_dataset.csv"
LOG_DIR       = Path("logs")
TELEMETRY_LOG = LOG_DIR / "study23_telemetry.jsonl"

CONTINUOUS_FEATURES = ["entry_rsr", "rsr_slope_5d", "days_cross90", "atr_z",
                         "gap_pct", "volume_z", "rank"]
CATEGORICAL_FEATURES = ["market_regime", "sector"]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

N_BINS = 3   # tercile binning for conditional_loss_rate / mutual_information

EXPLAINABLE_PRECISION_MIN = 0.70
EXPLAINABLE_COVERAGE_MIN  = 0.50
EXPLAINABLE_PROFIT_MAX    = 0.20
EXPLAINABLE_ALPHA_MIN     = 0.80
PARTIAL_PRECISION_MIN     = 0.60
PARTIAL_COVERAGE_MIN      = 0.40


@dataclass
class TradeFeatureRow:
    trade_id:       int
    symbol:         str
    entry_date:     str
    exit_date:      str
    actual_R:       float
    label:          int     # 1 = bad (FALSE_BREAKOUT/HIGH_VOL_ENTRY), 0 = winner
    category:       str     # Study22 category, or "WINNER"
    entry_rsr:      float
    rsr_slope_5d:   float
    days_cross90:   int
    atr_z:          float
    gap_pct:        float
    volume_z:       float
    market_regime:  str
    sector:         str
    rank:           int


@dataclass
class FeatureProfile:
    feature:               str
    feature_type:           str
    mutual_information:     float
    lift:                   float
    bin_breakdown:           List[dict]
    best_rule_desc:          str
    precision:               float
    recall:                  float
    coverage:                float
    conditional_loss_rate:   float


@dataclass
class Study23Result:
    trade_count:                 int = 0   # scoped (bad + winners)
    n_bad:                       int = 0
    n_winners:                   int = 0
    best_rule_desc:               str = ""
    precision:                    float = 0.0
    recall:                       float = 0.0
    coverage:                     float = 0.0
    loss_explainability:           float = 0.0
    profit_explainability:         float = 0.0
    counterfactual_removed_loss:   float = 0.0
    counterfactual_removed_profit: float = 0.0
    alpha_retention:               float = 0.0
    top_predictors:                List[str] = field(default_factory=list)
    research_decision:             str = ""
    decision_reason:                str = ""


# ─────────────────────────────────────────────────────────────────────
#  Feature engineering (existing-feature transforms only)
# ─────────────────────────────────────────────────────────────────────

def build_symbol_feature_series(df: pd.DataFrame) -> Dict[str, pd.Series]:
    high, low, close_s, open_s, vol = df["High"], df["Low"], df["Close"], df["Open"], df["Volume"]
    prev_close = close_s.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_pct = atr14 / close_s
    atr_z = (atr_pct - atr_pct.rolling(60).mean()) / atr_pct.rolling(60).std()
    gap_pct = (open_s - prev_close) / prev_close
    vol_z = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    return {"atr_z": atr_z, "gap_pct": gap_pct, "volume_z": vol_z}


# ─────────────────────────────────────────────────────────────────────
#  Per-feature evaluation
# ─────────────────────────────────────────────────────────────────────

def _mutual_information(bin_ids: List[int], labels: List[int]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    py = {0: 0, 1: 0}
    for l in labels:
        py[l] += 1
    px: Dict[int, int] = {}
    pxy: Dict[Tuple[int, int], int] = {}
    for b, l in zip(bin_ids, labels):
        px[b] = px.get(b, 0) + 1
        pxy[(b, l)] = pxy.get((b, l), 0) + 1
    mi = 0.0
    for (b, l), cnt in pxy.items():
        p_xy = cnt / n
        p_x = px[b] / n
        p_y = py[l] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return round(mi, 4)


def _tercile_bins(values: List[float]) -> List[int]:
    arr = np.array(values, dtype=float)
    try:
        edges = np.quantile(arr, [1 / 3, 2 / 3])
    except Exception:
        return [0] * len(values)
    bins = []
    for v in values:
        if v <= edges[0]:
            bins.append(0)
        elif v <= edges[1]:
            bins.append(1)
        else:
            bins.append(2)
    return bins


def evaluate_feature(
    feature: str,
    values: List,
    labels: List[int],
    is_categorical: bool,
) -> FeatureProfile:
    n = len(labels)
    n_bad = sum(labels)
    base_rate = n_bad / n if n else 0.0

    # ── bin breakdown + lift + mutual_information ───────────────────
    if is_categorical:
        bin_ids = values
    else:
        bin_ids = _tercile_bins([float(v) for v in values])

    bin_breakdown: List[dict] = []
    uniq_bins = sorted(set(bin_ids), key=lambda x: str(x))
    for b in uniq_bins:
        idx = [i for i, bv in enumerate(bin_ids) if bv == b]
        n_b = len(idx)
        n_bad_b = sum(labels[i] for i in idx)
        bin_breakdown.append({
            "bin": str(b), "n": n_b, "n_bad": n_bad_b,
            "conditional_loss_rate": round(n_bad_b / n_b, 4) if n_b else 0.0,
        })
    max_cond_rate = max((b["conditional_loss_rate"] for b in bin_breakdown), default=0.0)
    lift = round(max_cond_rate / base_rate, 4) if base_rate > 0 else 0.0
    mi = _mutual_information(bin_ids, labels)

    # ── best single-condition rule (maximize precision, tie-break coverage) ──
    candidates: List[Tuple[List[bool], str]] = []
    if is_categorical:
        for v in sorted(set(values), key=str):
            candidates.append(([x == v for x in values], f"{feature} == {v}"))
    else:
        fvals = [float(v) for v in values]
        for v in sorted(set(fvals)):
            candidates.append(([x >= v for x in fvals], f"{feature} >= {v:.3g}"))
            candidates.append(([x <= v for x in fvals], f"{feature} <= {v:.3g}"))

    min_cov_n = max(1, math.ceil(MIN_RULE_COVERAGE * n))
    best = {"precision": 0.0, "recall": 0.0, "coverage": 0.0, "desc": "(no valid split)"}
    for mask, desc in candidates:
        cov_n = sum(mask)
        if cov_n < min_cov_n or cov_n == n:
            continue
        tp = sum(1 for m, l in zip(mask, labels) if m and l == 1)
        precision = tp / cov_n
        recall = tp / n_bad if n_bad else 0.0
        coverage = cov_n / n
        score = (round(precision, 6), round(coverage, 6))
        if score > (best["precision"], best["coverage"]):
            best = {"precision": precision, "recall": recall, "coverage": coverage, "desc": desc}

    return FeatureProfile(
        feature=feature,
        feature_type="categorical" if is_categorical else "continuous",
        mutual_information=mi,
        lift=lift,
        bin_breakdown=bin_breakdown,
        best_rule_desc=best["desc"],
        precision=round(best["precision"], 4),
        recall=round(best["recall"], 4),
        coverage=round(best["coverage"], 4),
        conditional_loss_rate=round(best["precision"], 4),
    )


def _apply_rule_mask(rule_desc: str, row_lookup: Dict[str, dict]) -> Dict[str, bool]:
    """Re-evaluate a `feature OP value` rule description against each trade row."""
    parts = rule_desc.split(" ")
    feat, op, val = parts[0], parts[1], " ".join(parts[2:])
    out: Dict[str, bool] = {}
    for tid, row in row_lookup.items():
        fv = row[feat]
        if op == "==":
            out[tid] = (str(fv) == val)
        elif op == ">=":
            out[tid] = (float(fv) >= float(val))
        elif op == "<=":
            out[tid] = (float(fv) <= float(val))
        else:
            out[tid] = False
    return out


# ─────────────────────────────────────────────────────────────────────
#  Aggregation + decision
# ─────────────────────────────────────────────────────────────────────

def aggregate(
    rows: List[TradeFeatureRow],
    profiles: Dict[str, FeatureProfile],
) -> Tuple[Study23Result, dict]:
    r = Study23Result()
    r.trade_count = len(rows)
    r.n_bad     = sum(1 for x in rows if x.label == 1)
    r.n_winners = sum(1 for x in rows if x.label == 0)

    top_predictors = sorted(profiles.values(), key=lambda p: -p.mutual_information)[:3]
    r.top_predictors = [p.feature for p in top_predictors]

    row_lookup = {str(x.trade_id): asdict(x) for x in rows}
    labels = {str(x.trade_id): x.label for x in rows}
    actual_R = {str(x.trade_id): x.actual_R for x in rows}

    # ── candidate rules: each feature's best single rule + top-2 interaction ──
    candidate_rules: List[Tuple[str, Dict[str, bool]]] = []
    for p in profiles.values():
        if p.coverage > 0:
            mask = _apply_rule_mask(p.best_rule_desc, row_lookup)
            candidate_rules.append((p.best_rule_desc, mask))

    interaction_info: Optional[dict] = None
    if len(top_predictors) >= 2:
        p1, p2 = top_predictors[0], top_predictors[1]
        m1 = _apply_rule_mask(p1.best_rule_desc, row_lookup)
        m2 = _apply_rule_mask(p2.best_rule_desc, row_lookup)
        combo_desc = f"({p1.best_rule_desc}) AND ({p2.best_rule_desc})"
        combo_mask = {k: (m1[k] and m2[k]) for k in m1}
        cov_n = sum(combo_mask.values())
        if 0 < cov_n < len(rows):
            tp = sum(1 for k, v in combo_mask.items() if v and labels[k] == 1)
            precision = tp / cov_n
            recall = tp / max(1, r.n_bad)
            coverage = cov_n / len(rows)
            interaction_info = {
                "rule": combo_desc, "precision": round(precision, 4),
                "recall": round(recall, 4), "coverage": round(coverage, 4),
                "feature1": p1.feature, "feature1_precision": p1.precision,
                "feature2": p2.feature, "feature2_precision": p2.precision,
                "improves_on_best_single": precision > max(p1.precision, p2.precision),
            }
            candidate_rules.append((combo_desc, combo_mask))

    # ── select best_rule across all candidates ──────────────────────────
    # Same anti-overfitting floor as evaluate_feature: a candidate must cover a
    # non-trivial share of the sample, else degenerate 1-2-trade splits would win.
    # Selection targets the best DECISION TIER the data supports (EXPLAINABLE >
    # PARTIALLY_EXPLAINABLE > best-effort), not raw precision alone — otherwise a
    # narrow-but-precise rule would always beat a broader rule that actually clears
    # the joint precision+coverage gate, making EXPLAINABLE/PARTIALLY_EXPLAINABLE
    # structurally unreachable even when a qualifying rule exists.
    min_cov_n = max(1, math.ceil(MIN_RULE_COVERAGE * len(rows)))
    scored: List[Tuple[str, Dict[str, bool], float, float, float]] = []
    for desc, mask in candidate_rules:
        cov_n = sum(mask.values())
        if cov_n < min_cov_n:
            continue
        tp = sum(1 for k, v in mask.items() if v and labels[k] == 1)
        precision = tp / cov_n
        recall = tp / max(1, r.n_bad)
        coverage = cov_n / len(rows)
        scored.append((desc, mask, precision, recall, coverage))

    explainable_tier = [c for c in scored if c[2] >= EXPLAINABLE_PRECISION_MIN and c[4] >= EXPLAINABLE_COVERAGE_MIN]
    partial_tier      = [c for c in scored if c[2] >= PARTIAL_PRECISION_MIN and c[4] >= PARTIAL_COVERAGE_MIN]

    if explainable_tier:
        pool = explainable_tier
    elif partial_tier:
        pool = partial_tier
    else:
        pool = scored

    best_choice = max(pool, key=lambda c: (c[2], c[4])) if pool else ("(none)", {}, 0.0, 0.0, 0.0)

    r.best_rule_desc, best_mask, r.precision, r.recall, r.coverage = (
        best_choice[0], best_choice[1], round(best_choice[2], 4),
        round(best_choice[3], 4), round(best_choice[4], 4),
    )

    # ── R-weighted counterfactual removal ───────────────────────────────
    bad_R_total     = sum(abs(actual_R[k]) for k, l in labels.items() if l == 1)
    winner_R_total  = sum(actual_R[k] for k, l in labels.items() if l == 0)
    flagged_bad_R   = sum(abs(actual_R[k]) for k in best_mask
                          if best_mask.get(k) and labels[k] == 1)
    flagged_win_R   = sum(actual_R[k] for k in best_mask
                          if best_mask.get(k) and labels[k] == 0)

    r.counterfactual_removed_loss   = round(flagged_bad_R / bad_R_total, 4) if bad_R_total > 0 else 0.0
    r.counterfactual_removed_profit = round(flagged_win_R / winner_R_total, 4) if winner_R_total > 0 else 0.0
    r.alpha_retention                = round(1.0 - r.counterfactual_removed_profit, 4)
    r.loss_explainability            = r.counterfactual_removed_loss
    r.profit_explainability          = r.alpha_retention

    if (r.precision >= EXPLAINABLE_PRECISION_MIN and r.coverage >= EXPLAINABLE_COVERAGE_MIN and
            r.counterfactual_removed_profit <= EXPLAINABLE_PROFIT_MAX and
            r.alpha_retention >= EXPLAINABLE_ALPHA_MIN):
        r.research_decision = "EXPLAINABLE"
        r.decision_reason = (
            f"precision={r.precision:.1%}>={EXPLAINABLE_PRECISION_MIN:.0%} AND "
            f"coverage={r.coverage:.1%}>={EXPLAINABLE_COVERAGE_MIN:.0%} AND "
            f"removed_profit={r.counterfactual_removed_profit:.1%}<={EXPLAINABLE_PROFIT_MAX:.0%} AND "
            f"alpha_retention={r.alpha_retention:.1%}>={EXPLAINABLE_ALPHA_MIN:.0%}"
        )
    elif r.precision >= PARTIAL_PRECISION_MIN and r.coverage >= PARTIAL_COVERAGE_MIN:
        r.research_decision = "PARTIALLY_EXPLAINABLE"
        r.decision_reason = (
            f"precision={r.precision:.1%}>={PARTIAL_PRECISION_MIN:.0%} AND "
            f"coverage={r.coverage:.1%}>={PARTIAL_COVERAGE_MIN:.0%} "
            f"(EXPLAINABLE full gate not met)"
        )
    else:
        r.research_decision = "NEW_SIGNAL_REQUIRED"
        r.decision_reason = (
            f"best_rule precision={r.precision:.1%} / coverage={r.coverage:.1%} — "
            f"existing features insufficient to explain FALSE_BREAKOUT/HIGH_VOL_ENTRY"
        )

    return r, {"interaction": interaction_info}


# ─────────────────────────────────────────────────────────────────────
#  Output writers
# ─────────────────────────────────────────────────────────────────────

def write_dataset_csv(rows: List[TradeFeatureRow]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with TRADE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for x in rows:
            w.writerow(asdict(x))
    print(f"[CSV] {TRADE_CSV}")


def write_md_report(
    result: Study23Result,
    rows: List[TradeFeatureRow],
    profiles: Dict[str, FeatureProfile],
    extra: dict,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    w = L.append

    w("# Study23 Signal Failure Decomposition")
    w("")
    w("作成日: 2026-06-23  |  説明責任のみ（accountability only）/ 新規シグナル設計禁止 / "
      "Entry・Exit・Sizing・Capital変更禁止 / 改善案生成禁止")
    w("")
    w("**Strategy**: Study9 Case B (FROZEN)  **Entry**: RSR∈[92,95), days_cross90≤5, "
      "rsr_slope_5d≤5  **Exit**: RSR<90  **Capital/Authority/Execution**: 現行production "
      "configuration  **Governance**: annual_rebalance")
    w("")
    w(f"**観測ウィンドウ**: {OBS_START} → {OBS_END}")
    w("")
    w(f"**目的**: Study22で検出されたFALSE_BREAKOUT/HIGH_VOL_ENTRYが既存特徴量のみで説明可能か検証する")
    w("")
    w(f"**対象**: FALSE_BREAKOUT/HIGH_VOL_ENTRY（label=1, n={result.n_bad}）+ 全勝ちトレード"
      f"（label=0, n={result.n_winners}）= 計{result.trade_count}件")
    w("")
    w("⚠ 小サンプル注意: n=27前後（Study21/22由来）。per-feature分割は過学習リスクが高く、"
      "本研究はexplainability評価専用（新規ルール採用は禁止）。")
    w("")
    w("⚠ データ品質注意: 9104.T（#3/#4/#5）でgap_pctが+45〜+55%という異常値。"
      "OHLCVデータの分割調整不整合が疑われる（real economic gapではない可能性が高い）。"
      "gap_pctはbest_ruleとして選定されなかったため最終判定への影響はないが、"
      "gap_pct単独の precision/lift 数値（本レポートFeature Profiles表）は9104.Tの寄与分だけ"
      "noisy である点に注意。データパイプラインの修正は本研究の範囲外。")
    w("")

    w("---")
    w("## Feature Profiles")
    w("")
    w("| feature | type | MI | lift | best_rule | precision | recall | coverage |")
    w("|---|---|---|---|---|---|---|---|")
    for feat in ALL_FEATURES:
        p = profiles[feat]
        w(f"| {p.feature} | {p.feature_type} | {p.mutual_information:.4f} | {p.lift:.2f}x | "
          f"`{p.best_rule_desc}` | {p.precision:.1%} | {p.recall:.1%} | {p.coverage:.1%} |")
    w("")

    w("### Bin Breakdown（tercile / category別 conditional_loss_rate）")
    w("")
    for feat in ALL_FEATURES:
        p = profiles[feat]
        w(f"**{feat}**: " + " / ".join(
            f"bin{b['bin']}(n={b['n']}, loss_rate={b['conditional_loss_rate']:.1%})"
            for b in p.bin_breakdown
        ))
    w("")

    w("---")
    w("## Top Predictors（mutual_information降順 上位3）")
    w("")
    for i, feat in enumerate(result.top_predictors, 1):
        p = profiles[feat]
        w(f"{i}. **{feat}** — MI={p.mutual_information:.4f}, lift={p.lift:.2f}x, "
          f"precision={p.precision:.1%}, coverage={p.coverage:.1%}")
    w("")

    w("---")
    w("## Feature Interactions")
    w("")
    inter = extra.get("interaction")
    if inter:
        w(f"- Rule: `{inter['rule']}`")
        w(f"- 単独: {inter['feature1']}(precision={inter['feature1_precision']:.1%}) / "
          f"{inter['feature2']}(precision={inter['feature2_precision']:.1%})")
        w(f"- 組合せ: precision={inter['precision']:.1%}  recall={inter['recall']:.1%}  "
          f"coverage={inter['coverage']:.1%}")
        w(f"- 相乗効果: {'あり（単独最大より改善）' if inter['improves_on_best_single'] else 'なし（単独と同等以下）'}")
    else:
        w("(top predictorが2件未満のため評価不可)")
    w("")

    w("---")
    w("## Executive Summary")
    w("")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| best_rule | `{result.best_rule_desc}` |")
    w(f"| precision | {result.precision:.1%} |")
    w(f"| recall | {result.recall:.1%} |")
    w(f"| coverage | {result.coverage:.1%} |")
    w(f"| loss_explainability (counterfactual_removed_loss, R-weighted) | {result.loss_explainability:.1%} |")
    w(f"| profit_explainability (alpha_retention) | {result.profit_explainability:.1%} |")
    w(f"| counterfactual_removed_loss | {result.counterfactual_removed_loss:.1%} |")
    w(f"| counterfactual_removed_profit | {result.counterfactual_removed_profit:.1%} |")
    w(f"| alpha_retention | {result.alpha_retention:.1%} |")
    w(f"| top_predictors | {', '.join(result.top_predictors)} |")
    w("")
    w(f"**research_decision: {result.research_decision}**")
    w("")
    w(f"判定理由: {result.decision_reason}")
    w("")

    w("---")
    w("## 判定基準")
    w("")
    w("| 判定 | 条件 |")
    w("|---|---|")
    w(f"| EXPLAINABLE | precision≥{EXPLAINABLE_PRECISION_MIN:.0%} AND coverage≥{EXPLAINABLE_COVERAGE_MIN:.0%} AND "
      f"removed_profit≤{EXPLAINABLE_PROFIT_MAX:.0%} AND alpha_retention≥{EXPLAINABLE_ALPHA_MIN:.0%} |")
    w(f"| PARTIALLY_EXPLAINABLE | precision≥{PARTIAL_PRECISION_MIN:.0%} AND coverage≥{PARTIAL_COVERAGE_MIN:.0%} |")
    w("| NEW_SIGNAL_REQUIRED | 上記未達 |")
    w("")

    w("---")
    w("## Trade Dataset（全件）")
    w("")
    w("| # | 銘柄 | Entry | label | category | actual_R | entry_rsr | slope5 | d90 | "
      "atr_z | gap% | vol_z | regime | sector | rank |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        w(f"| {x.trade_id} | {x.symbol} | {x.entry_date} | {x.label} | {x.category} | "
          f"{x.actual_R:+.2f}R | {x.entry_rsr:.1f} | {x.rsr_slope_5d:+.1f} | {x.days_cross90} | "
          f"{x.atr_z:+.2f} | {x.gap_pct:+.2%} | {x.volume_z:+.2f} | {x.market_regime} | "
          f"{x.sector} | {x.rank} |")
    w("")

    w("---")
    w("## 最終出力")
    w("")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| loss_explainability | {result.loss_explainability:.1%} |")
    w(f"| profit_explainability | {result.profit_explainability:.1%} |")
    w(f"| top_predictors | {result.top_predictors} |")
    w(f"| counterfactual_removed_loss | {result.counterfactual_removed_loss:.1%} |")
    w(f"| counterfactual_removed_profit | {result.counterfactual_removed_profit:.1%} |")
    w(f"| alpha_retention | {result.alpha_retention:.1%} |")
    w(f"| **research_decision** | **{result.research_decision}** |")
    w("")
    w("研究目的は説明責任のみ。新規Entry/Exitルール・改善案はこのレポートでは提案しない。")
    w("")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[MD] {REPORT_MD}")


def append_telemetry(result: Study23Result) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {"study": "study23", **asdict(result)}
        with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("[Study23] Signal Failure Decomposition")
    print("=" * 68)

    print("[1/5] データロード中...")
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
    print(f"[1/5] 共通日数={len(common_dates)}  銘柄={n_syms}")

    print("[2/5] 価格・RSR・特徴量マトリクス構築...")
    n_dates = len(common_dates)
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    atr_z_mat   = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    gap_pct_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    vol_z_mat   = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    for si, sym in enumerate(active_syms):
        df_src = universe_raw[sym]["df"]
        ri = df_src.index.get_indexer(common_dates)
        valid = ri >= 0
        if valid.any():
            open_mat[valid, si]  = df_src["Open"].to_numpy(dtype=np.float32)[ri[valid]]
            close_mat[valid, si] = df_src["Close"].to_numpy(dtype=np.float32)[ri[valid]]
        feats = build_symbol_feature_series(df_src)
        for name, mat in (("atr_z", atr_z_mat), ("gap_pct", gap_pct_mat), ("volume_z", vol_z_mat)):
            series = feats[name].reindex(common_dates, method="ffill")
            mat[:, si] = series.to_numpy(dtype=np.float32)

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

    print("[3/5] Study9 Case B シグナル生成 + Study22 損失分類 (FROZEN, 観測のみ)...")
    all_signals = generate_signals(
        common_dates, active_syms, sym_to_i,
        open_mat, close_mat, rsr_mat, sym_active_mat, mkt_ret1,
    )
    pairs = pair_trades(all_signals)
    print(f"[3/5] completed_trades={len(pairs)}")

    # Study22 high-vol threshold pass (must match study22's own computation exactly)
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

    loss_attrs = {}
    for tid, pair in enumerate(pairs, start=1):
        si = sym_to_i[pair["symbol"]]
        attr = attribute_loss(
            tid, pair, close_mat[:, si], rsr_mat[:, si], slope5_mat[:, si],
            cross90_mat[:, si], rsr_mat, sym_active_mat, si, date_to_idx,
            topix_close, common_dates,
        )
        if attr is not None:
            loss_attrs[tid] = attr

    print(f"[3/5] losers={len(loss_attrs)}  "
          f"target({'/'.join(TARGET_CATEGORIES)})="
          f"{sum(1 for a in loss_attrs.values() if a.category in TARGET_CATEGORIES)}")

    print("[4/5] 特徴量データセット構築（target losers + 全勝ちトレード）...")
    rows: List[TradeFeatureRow] = []
    for tid, pair in enumerate(pairs, start=1):
        ei = date_to_idx.get(pair["entry_date"])
        if ei is None:
            continue
        sym = pair["symbol"]
        si  = sym_to_i[sym]
        ep, xp = pair["entry_price"], pair["exit_price"]
        entry_fill = ep * (1.0 + SLIPPAGE)
        exit_fill  = xp * (1.0 - SLIPPAGE)
        cost_basis = entry_fill * (1.0 + COMMISSION)
        proceeds   = exit_fill  * (1.0 - COMMISSION)
        actual_return = proceeds / cost_basis - 1.0
        actual_R = actual_return / R_UNIT_PCT

        is_winner = actual_R >= 0
        attr = loss_attrs.get(tid)
        category = "WINNER" if is_winner else (attr.category if attr else "UNKNOWN")
        if not is_winner and category not in TARGET_CATEGORIES:
            continue   # out of scope (Study22 attributed this loss elsewhere)
        label = 0 if is_winner else 1

        if sym_active_mat is not None:
            active_row = sym_active_mat[ei] >= 0.5
        else:
            active_row = np.ones(rsr_mat.shape[1], dtype=bool)
        rsr_today = rsr_mat[ei]
        entry_rsr_val = float(rsr_mat[ei, si])
        eligible_rsr = rsr_today[active_row]
        entry_rank = int(np.sum(eligible_rsr > entry_rsr_val) + 1) if len(eligible_rsr) > 0 else 1

        bear_at_entry = False
        if topix_close is not None:
            from src.strategy.universe import is_bear_regime
            entry_dt = common_dates[ei]
            topix_slice = topix_close.loc[:entry_dt]
            bear_at_entry = is_bear_regime(topix_slice) if len(topix_slice) > 0 else False

        rows.append(TradeFeatureRow(
            trade_id=tid, symbol=sym, entry_date=pair["entry_date"],
            exit_date=pair["exit_date"], actual_R=round(actual_R, 4),
            label=label, category=category,
            entry_rsr=round(entry_rsr_val, 2),
            rsr_slope_5d=round(float(slope5_mat[ei, si]), 2),
            days_cross90=int(cross90_mat[ei, si]),
            atr_z=round(float(atr_z_mat[ei, si]) if not np.isnan(atr_z_mat[ei, si]) else 0.0, 4),
            gap_pct=round(float(gap_pct_mat[ei, si]) if not np.isnan(gap_pct_mat[ei, si]) else 0.0, 6),
            volume_z=round(float(vol_z_mat[ei, si]) if not np.isnan(vol_z_mat[ei, si]) else 0.0, 4),
            market_regime="Bear" if bear_at_entry else "Bull",
            sector=trade_syms.get(sym, "不明"),
            rank=entry_rank,
        ))

    print(f"[4/5] scoped_trades={len(rows)}  (bad={sum(1 for r in rows if r.label==1)}  "
          f"winners={sum(1 for r in rows if r.label==0)})")

    print("[5/5] Feature evaluation + decision...")
    profiles: Dict[str, FeatureProfile] = {}
    labels = [r.label for r in rows]
    for feat in ALL_FEATURES:
        values = [getattr(r, feat) for r in rows]
        profiles[feat] = evaluate_feature(feat, values, labels, feat in CATEGORICAL_FEATURES)

    result, extra = aggregate(rows, profiles)

    print(f"  trade_count            = {result.trade_count}  (bad={result.n_bad}, winners={result.n_winners})")
    print(f"  best_rule              = {result.best_rule_desc}")
    print(f"  precision/coverage     = {result.precision:.1%} / {result.coverage:.1%}")
    print(f"  loss_explainability    = {result.loss_explainability:.1%}")
    print(f"  profit_explainability  = {result.profit_explainability:.1%}")
    print(f"  alpha_retention        = {result.alpha_retention:.1%}")
    print(f"  top_predictors         = {result.top_predictors}")
    print(f"  research_decision      = {result.research_decision}")
    print(f"  decision_reason        = {result.decision_reason}")

    write_dataset_csv(rows)
    write_md_report(result, rows, profiles, extra)
    append_telemetry(result)

    print()
    print("=" * 68)
    print(f"research_decision = {result.research_decision}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
