"""
src/execution/deployability.py
Capital-aware deployability scoring layer.

Scores each candidate BEFORE final selection so the allocator processes
deployable symbols first. Prevents idle capital accumulation caused by
repeated ranking of undeployable high-alpha candidates.

deployability_score = alpha_score
                    * capital_fit_multiplier
                    * sector_capacity_multiplier
                    * lot_feasibility_multiplier
                    * alloc_survival_probability

All functions are pure / deterministic: same input → same output.
No allocator state mutation occurs here.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOT_SIZE:        int   = 100
MIN_SCORE_RATIO: float = 0.05   # score < MIN_SCORE_RATIO * alpha_score → undeployable


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeployabilityResult:
    """Deployability scoring for a single candidate."""
    symbol:                     str
    deployability_score:        float   # 0.0 – alpha_score
    alpha_score:                float
    capital_fit_multiplier:     float   # 0.0 – 1.0
    sector_capacity_multiplier: float   # 0.0 – 1.0
    lot_feasibility_multiplier: float   # 0.0 or 1.0
    alloc_survival_probability: float   # 0.0 – 1.0
    is_deployable:              bool
    rejection_reason:           str     # "" if deployable


@dataclass
class DeployabilityDiagnostics:
    """Aggregate metrics across one ranking pass."""
    total_candidates:           int   = 0
    deployable_count:           int   = 0
    undeployable_count:         int   = 0
    undeployable_alpha_count:   int   = 0
    alloc_survival_rate:        float = 0.0
    idle_cash_ratio:            float = 0.0
    capital_efficiency:         float = 0.0
    rejection_reason_breakdown: dict  = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Core scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate(
    symbol:             str,
    alpha_score:        float,
    price:              float,
    qty_requested:      int,
    available_cash:     float,
    max_alloc:          float,        # yen — max single-position allocation
    sector:             str,
    sector_weight:      float,        # current sector weight fraction (0..1)
    sector_cap:         float,        # sector cap fraction (e.g. 0.25)
    capital:            float,        # total equity in yen
    existing_positions: int   = 0,
    max_positions:      int   = 3,
    reject_threshold:   float = 0.25,
) -> DeployabilityResult:
    """
    Compute deployability score for one candidate. Deterministic. Read-only.

    Multiplier semantics:
      lot_feasibility     — 1.0 if ≥1 lot fits in available_cash, else 0.0
      capital_fit         — fraction of qty_requested that available cash covers
      sector_capacity     — fraction of requested sector weight that still fits
      alloc_survival      — estimated probability allocator will not reject
    """
    rejection_reason = ""

    # ── guard: zero price ────────────────────────────────────────────────────
    if price <= 0 or qty_requested <= 0:
        return DeployabilityResult(
            symbol=symbol, deployability_score=0.0, alpha_score=alpha_score,
            capital_fit_multiplier=0.0, sector_capacity_multiplier=0.0,
            lot_feasibility_multiplier=0.0, alloc_survival_probability=0.0,
            is_deployable=False, rejection_reason="price_or_qty_zero",
        )

    # ── guard: max positions reached ─────────────────────────────────────────
    if existing_positions >= max_positions:
        return DeployabilityResult(
            symbol=symbol, deployability_score=0.0, alpha_score=alpha_score,
            capital_fit_multiplier=0.0, sector_capacity_multiplier=0.0,
            lot_feasibility_multiplier=0.0, alloc_survival_probability=0.0,
            is_deployable=False, rejection_reason="max_positions_reached",
        )

    # ── 1. lot feasibility ────────────────────────────────────────────────────
    min_lot_cost = LOT_SIZE * price
    lot_ok = available_cash >= min_lot_cost
    lot_feasibility = 1.0 if lot_ok else 0.0
    if not lot_ok:
        rejection_reason = "insufficient_cash_for_1_lot"

    # ── 2. capital fit ────────────────────────────────────────────────────────
    affordable_cash  = min(available_cash, max_alloc)
    max_affordable   = (affordable_cash // (LOT_SIZE * price)) * LOT_SIZE
    capital_fit      = min(1.0, max_affordable / qty_requested) if qty_requested > 0 else 0.0
    if capital_fit == 0.0 and not rejection_reason:
        rejection_reason = "capital_insufficient"

    # ── 3. sector capacity ────────────────────────────────────────────────────
    remaining_sector = max(0.0, sector_cap - sector_weight)
    effective_cap    = max(1.0, capital)
    requested_weight = (qty_requested * price) / effective_cap
    if requested_weight > 0:
        sector_mult = min(1.0, remaining_sector / requested_weight)
    else:
        sector_mult = 0.0
    if sector_mult < reject_threshold and not rejection_reason:
        rejection_reason = (
            f"sector_cap_insufficient("
            f"rem={remaining_sector:.3f} req={requested_weight:.3f})"
        )

    # ── 4. alloc survival probability ─────────────────────────────────────────
    # Heuristic: product of individual feasibility signals
    alloc_survival = (
        min(1.0, sector_mult / max(reject_threshold, 0.01))
        * lot_feasibility
        * min(1.0, capital_fit * 2.0)   # partial fill ≥ 0.5 capital_fit still survives
    )
    alloc_survival = min(1.0, max(0.0, alloc_survival))

    # ── final score ───────────────────────────────────────────────────────────
    score = (
        alpha_score
        * capital_fit
        * sector_mult
        * lot_feasibility
        * alloc_survival
    )
    score = max(0.0, min(alpha_score, score))

    is_deployable = (
        lot_feasibility > 0.0
        and capital_fit  > 0.0
        and sector_mult >= reject_threshold
        and score >= MIN_SCORE_RATIO * max(alpha_score, 1e-9)
    )
    if not is_deployable and not rejection_reason:
        rejection_reason = f"low_deployability_score({score:.4f})"

    return DeployabilityResult(
        symbol                     = symbol,
        deployability_score        = round(score, 6),
        alpha_score                = alpha_score,
        capital_fit_multiplier     = round(capital_fit,    4),
        sector_capacity_multiplier = round(sector_mult,    4),
        lot_feasibility_multiplier = round(lot_feasibility, 4),
        alloc_survival_probability = round(alloc_survival, 4),
        is_deployable              = is_deployable,
        rejection_reason           = rejection_reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch ranker
# ─────────────────────────────────────────────────────────────────────────────

def rank_by_deployability(
    candidates:         list[dict],
    available_cash:     float,
    capital:            float,
    sector_weights:     dict[str, float],   # sector → current weight fraction
    sector_cap:         float,
    max_alloc:          float,              # yen
    existing_positions: int,
    max_positions:      int   = 3,
    reject_threshold:   float = 0.25,
) -> tuple[list[dict], DeployabilityDiagnostics]:
    """
    Re-rank candidates by deployability_score desc.

    Each candidate dict must contain:
        symbol       str
        alpha_score  float   (RSR or equivalent ranking signal)
        price        float
        qty          int
        sector       str

    Returns (ranked_candidates, diagnostics).
    ranked_candidates: input dicts augmented with deployability fields, sorted by
        (deployability_score desc, alpha_score desc, symbol asc) for determinism.
    """
    diag = DeployabilityDiagnostics(total_candidates=len(candidates))
    scored: list[tuple[float, float, str, dict]] = []

    for cand in candidates:
        sym    = cand["symbol"]
        alpha  = float(cand.get("alpha_score", 1.0))
        price  = float(cand.get("price", 0.0))
        qty    = int(cand.get("qty", 0))
        sector = cand.get("sector", "不明")
        sw     = sector_weights.get(sector, 0.0)

        result = score_candidate(
            symbol             = sym,
            alpha_score        = alpha,
            price              = price,
            qty_requested      = qty,
            available_cash     = available_cash,
            max_alloc          = max_alloc,
            sector             = sector,
            sector_weight      = sw,
            sector_cap         = sector_cap,
            capital            = capital,
            existing_positions = existing_positions,
            max_positions      = max_positions,
            reject_threshold   = reject_threshold,
        )

        cand_out = dict(cand)
        cand_out["deployability_score"]        = result.deployability_score
        cand_out["rejection_reason"]           = result.rejection_reason
        cand_out["is_deployable"]              = result.is_deployable
        cand_out["capital_fit_multiplier"]     = result.capital_fit_multiplier
        cand_out["sector_cap_multiplier"]      = result.sector_capacity_multiplier
        cand_out["lot_feasibility_multiplier"] = result.lot_feasibility_multiplier
        cand_out["alloc_survival_probability"] = result.alloc_survival_probability

        scored.append((-result.deployability_score, -alpha, sym, cand_out))

        if result.is_deployable:
            diag.deployable_count += 1
        else:
            diag.undeployable_count += 1
            if alpha > 0:
                diag.undeployable_alpha_count += 1
            reason_key = result.rejection_reason or "unknown"
            diag.rejection_reason_breakdown[reason_key] = (
                diag.rejection_reason_breakdown.get(reason_key, 0) + 1
            )

        logger.info(
            "[DEPLOYABILITY] symbol=%s alpha=%.4f deploy=%.4f "
            "cap_fit=%.3f sec_cap=%.3f lot_ok=%s alloc_surv=%.3f "
            "deployable=%s reason=%s",
            sym, alpha, result.deployability_score,
            result.capital_fit_multiplier,
            result.sector_capacity_multiplier,
            result.lot_feasibility_multiplier > 0,
            result.alloc_survival_probability,
            result.is_deployable,
            result.rejection_reason,
        )

    # Deterministic sort: deployability_score desc → alpha desc → symbol asc
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    ranked = [entry[3] for entry in scored]

    # Aggregate
    n = max(1, diag.total_candidates)
    diag.alloc_survival_rate = diag.deployable_count / n
    # idle_cash: fraction of capital not covered by available_cash
    diag.idle_cash_ratio = max(0.0, 1.0 - available_cash / max(1.0, capital))

    logger.info(
        "[DEPLOYABILITY_DIAG] total=%d deployable=%d undeployable=%d "
        "alloc_survival=%.3f undeployable_alpha=%d breakdown=%s",
        diag.total_candidates, diag.deployable_count, diag.undeployable_count,
        diag.alloc_survival_rate, diag.undeployable_alpha_count,
        diag.rejection_reason_breakdown,
    )

    return ranked, diag


# ─────────────────────────────────────────────────────────────────────────────
# Metrics emitter
# ─────────────────────────────────────────────────────────────────────────────

def emit_deployability_metrics(
    diag:           DeployabilityDiagnostics,
    capital:        float,
    available_cash: float,
    committed_yen:  float,
    log_path:       Optional[Path] = None,
) -> dict:
    """
    Build deployability metrics dict; optionally append to JSONL log.
    Returns the metrics dict for downstream governance integration.
    """
    idle_cash    = max(0.0, available_cash - committed_yen)
    idle_ratio   = idle_cash / max(1.0, capital)
    cap_eff      = committed_yen / max(1.0, available_cash) if available_cash > 0 else 0.0

    metrics = {
        "ts":                         time.time(),
        "total_candidates":           diag.total_candidates,
        "deployable_count":           diag.deployable_count,
        "undeployable_count":         diag.undeployable_count,
        "undeployable_alpha_count":   diag.undeployable_alpha_count,
        "alloc_survival_rate":        round(diag.alloc_survival_rate,  4),
        "capital_efficiency":         round(min(1.0, cap_eff),         4),
        "idle_cash_ratio":            round(idle_ratio,                4),
        "rejection_reason_breakdown": diag.rejection_reason_breakdown,
    }

    if log_path is not None:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as exc:
            logger.warning("[DEPLOYABILITY] metrics log failed: %s", exc)

    return metrics
