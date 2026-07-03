"""
predictive_expansion_score.py — Intraday Predictive Expansion Scorer

ExpansionState → 0-100 整数スコア に変換する。

設計原則:
  - deterministic: 同一入力 → 同一出力
  - pure function: I/O なし
  - no ML: 線形加重のみ
  - explainable: 全コンポーネントの寄与度を公開

スパイク抑制:
  - expansion_type == "spike" → スコア上限 = MAX_SCORE_WHEN_SPIKE (10)
  - spike_confidence が高いほど score を乗数で削減
  - MIN_BARS_FOR_SCORE 未満 → SCORE_NO_VERDICT (-1) を返す

スコア区分:
  STRONG   ≥ 70
  MODERATE ≥ 45
  WEAK     ≥ 20
  BLOCKED  < 20 または spike
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .expansion_state_engine import ExpansionState, MIN_BARS_FOR_VERDICT

logger = logging.getLogger(__name__)

# ── Score weights ────────────────────────────────────────────────────────────
SCORE_WEIGHTS: dict[str, float] = {
    "drive_efficiency":  0.30,
    "persistence_ratio": 0.25,
    "rvol_acceleration": 0.20,
    "rs_percentile":     0.15,
    "gap_acceptance":    0.10,
}

assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9, "SCORE_WEIGHTS must sum to 1.0"

# ── Thresholds ────────────────────────────────────────────────────────────────
MAX_SCORE_WHEN_SPIKE: int   = 10   # confirmed spike は上限10
SPIKE_SUPPRESSION: float    = 0.90  # score *= (1 - 0.90 * spike_confidence)
MIN_BARS_FOR_SCORE: int     = MIN_BARS_FOR_VERDICT
SCORE_NO_VERDICT: int       = -1    # 判定不能 (bars 不足)

# ── Score labels ─────────────────────────────────────────────────────────────
LABEL_STRONG:   int = 70
LABEL_MODERATE: int = 45
LABEL_WEAK:     int = 20


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpansionScoreResult:
    """PredictiveExpansionScorer.score() の出力。"""
    score: int                           # 0-100; -1 = 判定不能
    score_label: str                     # "STRONG" | "MODERATE" | "WEAK" | "BLOCKED" | "NO_VERDICT"
    component_scores: dict[str, float]   # 各コンポーネントの 0-100 換算値
    component_weights: dict[str, float]  # 使用したウェイト (SCORE_WEIGHTS のコピー)
    spike_penalty_applied: float         # 適用したスパイクペナルティ乗数 [0,1]
    explanation: list[str]               # 人間可読な判断根拠

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "score_label": self.score_label,
            "component_scores": self.component_scores,
            "component_weights": self.component_weights,
            "spike_penalty_applied": self.spike_penalty_applied,
            "explanation": self.explanation,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Component → [0, 100] 変換
# ─────────────────────────────────────────────────────────────────────────────

def _score_drive(v: float) -> float:
    """drive_efficiency [0,1] → [0,100]"""
    return float(np.clip(v * 100.0, 0.0, 100.0))


def _score_persistence(v: float) -> float:
    """persistence_ratio [0,1] → [0,100]"""
    return float(np.clip(v * 100.0, 0.0, 100.0))


def _score_rvol_accel(v: float) -> float:
    """
    rvol_acceleration → [0,100]
    0.0 →  0  (出来高消滅)
    1.0 → 50  (中立)
    2.0 → 100 (倍速)
    """
    return float(np.clip(v / 2.0 * 100.0, 0.0, 100.0))


def _score_rs(v: float) -> float:
    """rs_percentile [0,100] → [0,100] (直接)"""
    return float(np.clip(v, 0.0, 100.0))


def _score_gap(gap_acceptance: float, gap_active: bool) -> float:
    """
    gap_acceptance [-1,1] → [0,100]
    gap_active=False → 50 (中立)
    """
    if not gap_active:
        return 50.0
    mapped = (gap_acceptance + 1.0) / 2.0 * 100.0
    return float(np.clip(mapped, 0.0, 100.0))


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring function (pure, no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def score_expansion(
    state: ExpansionState,
    weights: dict[str, float] | None = None,
    min_bars: int = MIN_BARS_FOR_SCORE,
) -> ExpansionScoreResult:
    """
    ExpansionState を 0-100 スコアに変換する。

    Args:
        state:   compute_expansion_state() の出力
        weights: カスタムウェイト (デフォルト: SCORE_WEIGHTS)
        min_bars: この本数未満なら SCORE_NO_VERDICT を返す

    Returns:
        ExpansionScoreResult — 例外時もデフォルト値で返す
    """
    try:
        return _score_inner(state, weights, min_bars)
    except Exception as exc:
        logger.error("[EXPANSION_SCORE] %s: %s", state.symbol, exc)
        return ExpansionScoreResult(
            score=0,
            score_label="BLOCKED",
            component_scores={},
            component_weights=weights or SCORE_WEIGHTS.copy(),
            spike_penalty_applied=0.0,
            explanation=["error_in_scoring"],
        )


def _score_inner(
    state: ExpansionState,
    weights: dict[str, float] | None,
    min_bars: int,
) -> ExpansionScoreResult:
    w = weights if weights is not None else SCORE_WEIGHTS.copy()
    explanation: list[str] = []

    # ── 1. 判定不能: bars 不足 ────────────────────────────────────────────────
    if state.minutes_elapsed < min_bars:
        explanation.append(f"no_verdict: elapsed={state.minutes_elapsed}<{min_bars}")
        return ExpansionScoreResult(
            score=SCORE_NO_VERDICT,
            score_label="NO_VERDICT",
            component_scores={},
            component_weights=w,
            spike_penalty_applied=0.0,
            explanation=explanation,
        )

    # ── 2. コンポーネントスコア [0,100] ──────────────────────────────────────
    gap_active = abs(state.gap_acceptance - 0.5) > 0.001 or state.gap_acceptance != 0.5
    # gap_active を ExpansionState から再導出できないのでheuristic: 0.5は中立(inactive)
    # より正確には expansion_state_engine 側で gap_active をフィールドとして公開する
    # 現状: gap_acceptance が 0.5 のとき gap_active=False と仮定 (safe default)
    inferred_gap_active = state.gap_acceptance != 0.5

    cs: dict[str, float] = {
        "drive_efficiency":  _score_drive(state.drive_efficiency),
        "persistence_ratio": _score_persistence(state.persistence_ratio),
        "rvol_acceleration": _score_rvol_accel(state.rvol_acceleration),
        "rs_percentile":     _score_rs(state.rs_percentile),
        "gap_acceptance":    _score_gap(state.gap_acceptance, inferred_gap_active),
    }

    raw_score = sum(cs[k] * w.get(k, 0.0) for k in cs)

    # ── 3. スパイク抑制 ───────────────────────────────────────────────────────
    spike_penalty = state.spike_confidence * SPIKE_SUPPRESSION
    post_spike    = raw_score * (1.0 - spike_penalty)

    if state.spike_confidence > 0.01:
        explanation.append(
            f"spike_penalty={spike_penalty:.2f}"
            f" (confidence={state.spike_confidence:.2f})"
        )

    # ── 4. スパイク Hard Cap ──────────────────────────────────────────────────
    if state.expansion_type == "spike":
        final_score = min(int(round(post_spike)), MAX_SCORE_WHEN_SPIKE)
        explanation.append(f"spike_hard_cap={MAX_SCORE_WHEN_SPIKE}")
    else:
        final_score = int(round(float(np.clip(post_spike, 0.0, 100.0))))

    # ── 5. 継続シグナル根拠 ───────────────────────────────────────────────────
    for reason in state.continuation_reasons:
        explanation.append(reason)
    for reason in state.spike_reasons:
        explanation.append(reason)

    # ── 6. ラベル ─────────────────────────────────────────────────────────────
    if state.expansion_type == "spike":
        label = "BLOCKED"
    elif final_score < 0:
        label = "NO_VERDICT"
    elif final_score >= LABEL_STRONG:
        label = "STRONG"
    elif final_score >= LABEL_MODERATE:
        label = "MODERATE"
    elif final_score >= LABEL_WEAK:
        label = "WEAK"
    else:
        label = "BLOCKED"

    return ExpansionScoreResult(
        score=max(0, final_score),
        score_label=label,
        component_scores=cs,
        component_weights=w,
        spike_penalty_applied=round(spike_penalty, 4),
        explanation=explanation,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Class API
# ─────────────────────────────────────────────────────────────────────────────

class PredictiveExpansionScorer:
    """
    ExpansionState → 0-100 スコアの計算器。

    デフォルトウェイトは SCORE_WEIGHTS。
    weights 引数でカスタマイズ可能だが sum == 1.0 を要求する。
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        min_bars: int = MIN_BARS_FOR_SCORE,
    ) -> None:
        if weights is not None:
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"weights must sum to 1.0, got {total:.6f}"
                )
        self._weights = weights or SCORE_WEIGHTS.copy()
        self._min_bars = min_bars

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def score(self, state: ExpansionState) -> ExpansionScoreResult:
        """ExpansionState を 0-100 スコアに変換する。決定論的・副作用なし。"""
        return score_expansion(state, weights=self._weights, min_bars=self._min_bars)

    def score_batch(
        self, states: list[ExpansionState]
    ) -> list[tuple[str, ExpansionScoreResult]]:
        """複数銘柄を一括スコアリング。[(symbol, result), ...]"""
        return [(s.symbol, self.score(s)) for s in states]

    def top_k(
        self, states: list[ExpansionState], k: int = 5
    ) -> list[tuple[str, int]]:
        """
        上位k銘柄を (symbol, score) で返す。
        score=-1 (NO_VERDICT) と type="spike" は除外。
        決定論的タイブレーク: alphabetical。
        """
        results = self.score_batch(states)
        valid = [
            (sym, r.score)
            for sym, r in results
            if r.score >= 0 and r.score_label != "BLOCKED"
        ]
        ranked = sorted(valid, key=lambda x: (-x[1], x[0]))
        return ranked[:k]
