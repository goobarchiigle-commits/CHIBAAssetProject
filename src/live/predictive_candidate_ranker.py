"""
predictive_candidate_ranker.py — Predictive Expansion Integration Layer

既存の dry-run 候補リストに predictive expansion score を統合し、
「どの銘柄を優先するか」を改善する。

役割:
  Phase 5A (efficiency_ranker) + Predictive Expansion (predictive_expansion.py) を
  統合した combined_score で候補銘柄を再ランキングする。

設計原則:
  - deterministic: 同一入力 → 同一出力
  - pure ranking: I/O なし (ログ書き込みは呼び出し元が担当)
  - fail-safe fallback: predictive_result が空でも既存 efficiency_score で動く
  - no live order routing changes

Predictive タグ:
  PREDICTIVE_EXPANSION_HIGH   predictive_alpha_score > 70
  STRONG_CONTINUATION         sector_ignition または leader/follower + score > 60
  SPIKE_RISK                  expansion without volume (breakout_expansion 高 + vol_ratio 低)
  FALSE_EXPANSION_RISK        shadow stats の false expansion 率が高い

Combined score:
  combined_score =
    W_EFF  * efficiency_rank_normalized   (Phase 5A 順位を [0,1] に正規化)
    + W_PRED * predictive_alpha_score/100
    + W_CONT * sector_bonus
  乗じる: spike_risk_multiplier = 1 - SPIKE_RISK_PENALTY * spike_risk
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))
_LOCK = threading.Lock()

# ── Composite weights ─────────────────────────────────────────────────────────
W_EFF:  float = 0.50   # Phase 5A efficiency (liquidity, correlation, DD)
W_PRED: float = 0.35   # predictive alpha (early expansion detection)
W_CONT: float = 0.15   # sector ignition / follower boost

assert abs(W_EFF + W_PRED + W_CONT - 1.0) < 1e-9

# ── Tag thresholds ────────────────────────────────────────────────────────────
TAG_PRED_HIGH_THRESHOLD: float     = 70.0  # predictive_alpha_score > この値
TAG_CONT_SCORE_MIN: float          = 60.0  # STRONG_CONTINUATION の最低スコア
TAG_SPIKE_BREAKOUT_MIN: float      = 1.50  # breakout_expansion > この値
TAG_SPIKE_VOL_MAX: float           = 0.90  # volume_ratio_5d < この値 (出来高サポートなし)
TAG_FALSE_EXP_RATE_MAX: float      = 0.30  # false_expansion_rate > 30% → WARNING

# ── Spike risk penalty ────────────────────────────────────────────────────────
SPIKE_RISK_PENALTY: float = 0.30   # combined_score *= (1 - 0.30 * spike_risk)

# ── Sector bonus ─────────────────────────────────────────────────────────────
SECTOR_IGNITION_BONUS: float = 0.15
FOLLOWER_BONUS: float        = 0.10

# ── Tag names (定数) ─────────────────────────────────────────────────────────
TAG_PRED_HIGH     = "PREDICTIVE_EXPANSION_HIGH"
TAG_STRONG_CONT   = "STRONG_CONTINUATION"
TAG_SPIKE_RISK    = "SPIKE_RISK"
TAG_FALSE_EXP     = "FALSE_EXPANSION_RISK"

ALL_TAGS: frozenset[str] = frozenset({
    TAG_PRED_HIGH, TAG_STRONG_CONT, TAG_SPIKE_RISK, TAG_FALSE_EXP,
})


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictiveTag:
    """1銘柄に付与される predictive タグ。"""
    name: str          # TAG_* 定数
    value: float       # トリガーとなった指標値
    note: str          # 人間可読な説明


@dataclass
class RankedCandidate:
    """
    Predictive + Efficiency を統合した候補ランキング結果。
    combined_rank=1 が最優先候補。
    """
    symbol: str
    combined_rank: int
    combined_score: float             # [0,1]
    base_efficiency_score: float      # Phase 5A から (0 = 不明)
    base_efficiency_rank: int         # Phase 5A 内のランク (0 = 不明)
    predictive_alpha_score: float     # 0-100 (0 = 不明)
    sector_igniting: bool
    is_leader: bool
    is_follower: bool
    spike_risk: float                 # [0,1]
    continuation_probability: float   # [0,1] heuristic or shadow stats
    tags: list[PredictiveTag] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = [{"name": t.name, "value": t.value, "note": t.note} for t in self.tags]
        return d


@dataclass
class CandidateRankingResult:
    """rank_with_predictive() の出力。JSONL ログ用。"""
    run_ts: str
    run_id: str
    n_candidates: int
    ranked: list[RankedCandidate]
    predictive_available: bool
    efficiency_available: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_ts": self.run_ts,
            "run_id": self.run_id,
            "n_candidates": self.n_candidates,
            "ranked": [r.to_dict() for r in self.ranked],
            "predictive_available": self.predictive_available,
            "efficiency_available": self.efficiency_available,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _normalize_scores(
    raw: dict[str, float],
    fallback: float = 0.5,
) -> dict[str, float]:
    """
    raw スコアをバッチ内でパーセンタイル正規化して [0,1] に変換する。
    1銘柄以下 / 全スコア同値の場合は fallback を返す。
    タイは同一ランク (unique-value 位置) を付与して一貫性を保つ。
    """
    if len(raw) < 2:
        return {sym: fallback for sym in raw}
    items = list(raw.items())
    values = np.array([v for _, v in items], dtype=float)
    unique_vals = np.unique(values)
    if len(unique_vals) == 1:
        return {sym: fallback for sym in raw}
    # 各値のユニーク値内での位置 → [0, 1] に正規化
    inv = np.searchsorted(unique_vals, values)
    normalized = inv / (len(unique_vals) - 1)
    return {sym: float(normalized[i]) for i, (sym, _) in enumerate(items)}


def _compute_spike_risk(
    sym: str,
    scores_dict: dict,
) -> float:
    """
    Spike risk [0,1] を計算する。
    breakout_expansion が高いかつ volume_ratio が低い = 出来高サポートなし = spike リスク。
    """
    bo_exp  = _safe_float(scores_dict.get("breakout_expansion", {}).get(sym, 1.0), 1.0)
    vol_r5  = _safe_float(scores_dict.get("volume_ratio_5d",    {}).get(sym, 1.0), 1.0)

    has_breakout   = bo_exp  > TAG_SPIKE_BREAKOUT_MIN
    no_vol_support = vol_r5  < TAG_SPIKE_VOL_MAX

    if has_breakout and no_vol_support:
        # 比例リスク: 超過分を0-1にスケール
        bo_excess = min((bo_exp  - TAG_SPIKE_BREAKOUT_MIN) / 1.0, 1.0)
        vol_def   = min((TAG_SPIKE_VOL_MAX - vol_r5)      / TAG_SPIKE_VOL_MAX, 1.0)
        return _safe_float(bo_excess * 0.5 + vol_def * 0.5, 0.0)

    return 0.0


def _build_tags(
    sym: str,
    pred_score: float,
    sector_igniting: bool,
    is_follower: bool,
    spike_risk: float,
    false_exp_rate: float,
) -> tuple[list[PredictiveTag], list[str]]:
    tags: list[PredictiveTag] = []
    expl: list[str] = []

    if pred_score > TAG_PRED_HIGH_THRESHOLD:
        tags.append(PredictiveTag(
            name=TAG_PRED_HIGH,
            value=pred_score,
            note=f"predictive_alpha={pred_score:.1f}>{TAG_PRED_HIGH_THRESHOLD:.0f}",
        ))
        expl.append(f"{TAG_PRED_HIGH}({pred_score:.0f})")

    if pred_score > TAG_CONT_SCORE_MIN and (sector_igniting or is_follower):
        role = "sector_igniting" if sector_igniting else "follower"
        tags.append(PredictiveTag(
            name=TAG_STRONG_CONT,
            value=pred_score,
            note=f"{role} + score={pred_score:.1f}>{TAG_CONT_SCORE_MIN:.0f}",
        ))
        expl.append(f"{TAG_STRONG_CONT}({role})")

    if spike_risk > 0.1:
        tags.append(PredictiveTag(
            name=TAG_SPIKE_RISK,
            value=spike_risk,
            note=f"spike_risk={spike_risk:.2f} (breakout_high+vol_low)",
        ))
        expl.append(f"{TAG_SPIKE_RISK}({spike_risk:.2f})")

    if false_exp_rate > TAG_FALSE_EXP_RATE_MAX:
        tags.append(PredictiveTag(
            name=TAG_FALSE_EXP,
            value=false_exp_rate,
            note=f"false_expansion_rate={false_exp_rate:.0%}>{TAG_FALSE_EXP_RATE_MAX:.0%}",
        ))
        expl.append(f"{TAG_FALSE_EXP}({false_exp_rate:.0%})")

    return tags, expl


def _heuristic_continuation_probability(
    pred_score: float,
    sector_igniting: bool,
    is_follower: bool,
    spike_risk: float,
) -> float:
    """
    Shadow stats が利用できない場合のヒューリスティック継続確率。
    predictive_alpha_score を基準に補正する。
    """
    base = _safe_float(pred_score / 100.0, 0.5)
    if sector_igniting:
        base = min(1.0, base + 0.10)
    if is_follower:
        base = min(1.0, base + 0.05)
    base = max(0.0, base - spike_risk * 0.30)
    return round(base, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def rank_with_predictive(
    buy_signals: list[dict],
    predictive_result: dict,
    efficiency_scores: dict[str, float],
    *,
    false_expansion_rates: dict[str, float] | None = None,
    shadow_continuation_probs: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    run_ts: str = "",
    run_id: str = "",
) -> CandidateRankingResult:
    """
    BUY シグナルを predictive + efficiency を統合してランキングする。

    Args:
        buy_signals:               action="BUY" のシグナル dict リスト
        predictive_result:         run_predictive_expansion_hook() の戻り値
        efficiency_scores:         {symbol: capital_efficiency_score} (Phase 5A)
        false_expansion_rates:     {symbol: false_expansion_rate} (shadow stats から)
        shadow_continuation_probs: {symbol: continuation_probability} (shadow stats から)
        weights:                   {"eff": float, "pred": float, "cont": float} (合計1.0)
        run_ts:                    タイムスタンプ (呼び出し元提供)
        run_id:                    ランID

    Returns:
        CandidateRankingResult — ranked に combined_rank=1 が最優先候補
    """
    warnings: list[str] = []

    # ── Weight 設定 ───────────────────────────────────────────────────────────
    if weights:
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            warnings.append(f"weights_sum={total:.4f} != 1.0; using defaults")
            weights = None
    w_eff  = weights.get("eff",  W_EFF)  if weights else W_EFF
    w_pred = weights.get("pred", W_PRED) if weights else W_PRED
    w_cont = weights.get("cont", W_CONT) if weights else W_CONT

    # ── Predictive scores 取得 ───────────────────────────────────────────────
    scores_dict: dict = predictive_result.get("scores", {}) if predictive_result else {}
    pred_alpha:   dict[str, float] = scores_dict.get("predictive_alpha_score", {})
    igniting_map: dict[str, bool]  = scores_dict.get("sector_ignition_active", {})
    is_leader_map: dict[str, bool] = scores_dict.get("is_leader", {})
    is_follower_map: dict[str, bool] = scores_dict.get("is_follower", {})
    predictive_available = bool(pred_alpha)

    # ── Efficiency スコアの正規化 ─────────────────────────────────────────────
    eff_available = bool(efficiency_scores)
    eff_normalized = _normalize_scores(efficiency_scores) if eff_available else {}

    # ── Predictive スコアの正規化 ─────────────────────────────────────────────
    pred_normalized = _normalize_scores(pred_alpha) if predictive_available else {}

    # ── 各候補を計算 ─────────────────────────────────────────────────────────
    items: list[tuple[str, float, RankedCandidate]] = []

    for sig in buy_signals:
        sym = sig.get("symbol", "") if isinstance(sig, dict) else getattr(sig, "symbol", "")
        if not sym:
            continue

        pred_score = _safe_float(pred_alpha.get(sym, 0.0), 0.0)
        pred_norm  = _safe_float(pred_normalized.get(sym, 0.5), 0.5)
        eff_score  = _safe_float(efficiency_scores.get(sym, 0.0), 0.0)
        eff_norm   = _safe_float(eff_normalized.get(sym, 0.5), 0.5)

        sector_igniting = bool(igniting_map.get(sym, False))
        is_leader       = bool(is_leader_map.get(sym, False))
        is_follower     = bool(is_follower_map.get(sym, False))

        spike_risk = _compute_spike_risk(sym, scores_dict) if predictive_available else 0.0

        false_exp_rate = _safe_float(
            (false_expansion_rates or {}).get(sym, 0.0), 0.0
        )

        # Sector bonus
        sector_bonus: float = 0.0
        if sector_igniting:
            sector_bonus += SECTOR_IGNITION_BONUS
        if is_follower:
            sector_bonus += FOLLOWER_BONUS
        sector_bonus = min(1.0, sector_bonus)

        # Combined score
        raw_combined = (
            w_eff  * eff_norm
            + w_pred * pred_norm
            + w_cont * sector_bonus
        )
        spike_mult = 1.0 - SPIKE_RISK_PENALTY * spike_risk
        combined = _safe_float(raw_combined * spike_mult, 0.0)
        combined = float(np.clip(combined, 0.0, 1.0))

        # Continuation probability
        if shadow_continuation_probs and sym in shadow_continuation_probs:
            cont_prob = _safe_float(shadow_continuation_probs[sym], 0.5)
        else:
            cont_prob = _heuristic_continuation_probability(
                pred_score, sector_igniting, is_follower, spike_risk
            )

        tags, expl = _build_tags(
            sym, pred_score, sector_igniting, is_follower, spike_risk, false_exp_rate
        )

        candidate = RankedCandidate(
            symbol=sym,
            combined_rank=0,          # set below after sorting
            combined_score=round(combined, 6),
            base_efficiency_score=round(eff_score, 6),
            base_efficiency_rank=0,   # set below
            predictive_alpha_score=round(pred_score, 2),
            sector_igniting=sector_igniting,
            is_leader=is_leader,
            is_follower=is_follower,
            spike_risk=round(spike_risk, 4),
            continuation_probability=cont_prob,
            tags=tags,
            explanation=expl,
        )
        items.append((sym, combined, candidate))

    # ── ランク付け (高スコア優先、タイは alphabetical) ─────────────────────
    items.sort(key=lambda x: (-x[1], x[0]))

    # Phase 5A 内ランク付け (efficiency_score 降順)
    eff_rank_order = sorted(
        [sym for sym, _, _ in items],
        key=lambda s: (-_safe_float(efficiency_scores.get(s, 0.0)), s),
    )
    eff_rank_map = {sym: i + 1 for i, sym in enumerate(eff_rank_order)}

    ranked: list[RankedCandidate] = []
    for rank, (sym, _, cand) in enumerate(items, 1):
        cand.combined_rank    = rank
        cand.base_efficiency_rank = eff_rank_map.get(sym, 0)
        ranked.append(cand)

    if not predictive_available:
        warnings.append("predictive_result empty; using efficiency_score only")
    if not eff_available:
        warnings.append("efficiency_scores empty; using predictive_score only")

    ts = run_ts or datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")

    return CandidateRankingResult(
        run_ts=ts,
        run_id=run_id,
        n_candidates=len(ranked),
        ranked=ranked,
        predictive_available=predictive_available,
        efficiency_available=eff_available,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL append helper
# ─────────────────────────────────────────────────────────────────────────────

def append_ranking_record(result: CandidateRankingResult, path: Path) -> None:
    """ランキング結果を JSONL に追記する (fail-open)。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
        with _LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
    except Exception as exc:
        logger.warning("[PRED_RANKER] JSONL append failed: %s", exc)


def load_ranking_records(path: Path) -> list[dict]:
    """JSONL から全ランキング記録を読み込む (crash-safe)。"""
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("[PRED_RANKER] malformed line skipped")
    except Exception as exc:
        logger.warning("[PRED_RANKER] JSONL read failed: %s", exc)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run display helper
# ─────────────────────────────────────────────────────────────────────────────

def format_predictive_display(result: CandidateRankingResult) -> str:
    """
    dry-run 出力用のフォーマット済み文字列を返す。
    predictive score / continuation probability / spike risk を含む。
    """
    if not result.ranked:
        return "  (predictive ranking: 候補なし)"

    lines = [
        "",
        "  ── [PREDICTIVE_RANKING] 統合ランキング ───────────────────────────",
        f"  {'#':<3} {'銘柄':<10} {'Combined':>8} {'Pred':>6} {'ContP':>6}"
        f" {'SpikeR':>7} {'Tags'}",
        f"  {'-'*3} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*20}",
    ]

    for c in result.ranked[:10]:
        tag_str = " ".join(t.name.replace("_", "").replace("PREDICTIVE", "PRED")
                           for t in c.tags) or "-"
        lines.append(
            f"  {c.combined_rank:<3}"
            f" {c.symbol:<10}"
            f" {c.combined_score:>8.4f}"
            f" {c.predictive_alpha_score:>6.1f}"
            f" {c.continuation_probability:>6.2f}"
            f" {c.spike_risk:>7.3f}"
            f" {tag_str}"
        )

    if result.warnings:
        lines.append(f"  [WARN] {' | '.join(result.warnings)}")

    lines.append("")
    return "\n".join(lines)
