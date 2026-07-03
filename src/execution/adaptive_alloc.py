"""
src/execution/adaptive_alloc.py
Adaptive portfolio allocator v2.

Replaces binary sector-cap rejection with continuous degradation:
    multiplier = min(1.0, remaining_exposure / requested_exposure)
    reject only if multiplier < REJECT_THRESHOLD (0.25)

Invariants preserved:
    - no sector overflow  (allocated weight never exceeds cap)
    - deterministic       (same input → same output, ranking order preserved)
    - board-lot safety    (degraded qty rounded down to nearest LOT_SIZE)
    - max position constraints (enforced by caller)

v2.1 additions:
    - pre_rank_candidates(): deployability pre-ranking before final selection
      Wraps src.execution.deployability to expose the scoring layer via the
      allocator interface so callers get a single integration point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

REJECT_THRESHOLD: float = 0.25
LOT_SIZE:         int   = 100


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectorDegradeResult:
    """Outcome of a single AdaptiveAllocator.apply() call."""
    multiplier:    float   # 0.0 – 1.0
    degraded_qty:  int     # lot-rounded; 0 when reason == "rejected"
    original_qty:  int
    sector:        str
    reason:        str     # "ok" | "degraded" | "rejected"


@dataclass
class ForecastFrame:
    """One candidate in the pre-allocation sector exposure forecast."""
    symbol:           str
    sector:           str
    rsr_rank:         int
    requested_weight: float
    multiplier:       float
    effective_weight: float
    rejected:         bool


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveAllocator
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveAllocator:
    """
    Stateful sector-exposure tracker with continuous degradation.

    Usage per _build_orders() call:
        1. alloc = AdaptiveAllocator(capital, sector_cap, ...)
        2. alloc.set_existing_exposure(positions, universe_tickers, price_fn)
        3. frames = alloc.forecast(candidates)          # diagnostics
        4. for each candidate (in rsr_rank order):
               result = alloc.apply(sector, qty, price)
               if result.reason == "rejected": skip
               # use result.degraded_qty for the order
               alloc.commit(sector, result.degraded_qty, price)
        5. logger.info("[ALLOCATOR] %s", alloc.summary(...))
    """

    def __init__(
        self,
        capital:          float,
        sector_cap:       float = 0.25,
        bear_sector_cap:  float = 0.18,
        is_bear:          bool  = False,
        reject_threshold: float = REJECT_THRESHOLD,
    ) -> None:
        self._capital          = max(1.0, capital)
        self._effective_cap    = bear_sector_cap if is_bear else sector_cap
        self._reject_threshold = reject_threshold
        self._is_bear          = is_bear

        # running state
        self._sector_weight: dict[str, float] = {}
        self._committed_count: int   = 0
        self._degraded_count:  int   = 0
        self._rejected_count:  int   = 0
        self._total_allocated: float = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def set_existing_exposure(
        self,
        positions:        dict,
        universe_tickers: dict[str, str],
        price_fn,
    ) -> None:
        """Seed sector weights from already-held positions (excluding sells)."""
        for sym, pos in positions.items():
            sector = universe_tickers.get(sym, "不明")
            price  = price_fn(sym)
            if price > 0:
                w = pos.get("qty", 0) * price / self._capital
                self._sector_weight[sector] = (
                    self._sector_weight.get(sector, 0.0) + w
                )

    def apply(self, sector: str, qty: int, price: float) -> SectorDegradeResult:
        """
        Compute degradation for one candidate.
        Does NOT mutate state — call commit() after the order is accepted.
        """
        if qty <= 0 or price <= 0:
            return SectorDegradeResult(
                multiplier=0.0, degraded_qty=0, original_qty=qty,
                sector=sector, reason="rejected",
            )

        requested_weight = qty * price / self._capital
        current_weight   = self._sector_weight.get(sector, 0.0)
        remaining        = max(0.0, self._effective_cap - current_weight)
        multiplier       = min(1.0, remaining / requested_weight)

        if multiplier < self._reject_threshold:
            logger.info(
                "[SECTOR_DEGRADE] sector=%s requested=%.3f remaining=%.3f "
                "multiplier=%.3f action=rejected cap=%.3f(bear=%s)",
                sector, requested_weight, remaining, multiplier,
                self._effective_cap, self._is_bear,
            )
            return SectorDegradeResult(
                multiplier=multiplier, degraded_qty=0, original_qty=qty,
                sector=sector, reason="rejected",
            )

        if multiplier < 1.0:
            degraded_qty = _round_lot(int(qty * multiplier))
            if degraded_qty == 0:
                logger.info(
                    "[SECTOR_DEGRADE] sector=%s multiplier=%.3f → 0 lots after rounding, rejected",
                    sector, multiplier,
                )
                return SectorDegradeResult(
                    multiplier=multiplier, degraded_qty=0, original_qty=qty,
                    sector=sector, reason="rejected",
                )
            logger.info(
                "[SECTOR_DEGRADE] sector=%s requested=%.3f remaining=%.3f "
                "multiplier=%.3f qty=%d→%d action=degraded cap=%.3f(bear=%s)",
                sector, requested_weight, remaining, multiplier, qty, degraded_qty,
                self._effective_cap, self._is_bear,
            )
            return SectorDegradeResult(
                multiplier=multiplier, degraded_qty=degraded_qty, original_qty=qty,
                sector=sector, reason="degraded",
            )

        # multiplier == 1.0: full allocation fits
        return SectorDegradeResult(
            multiplier=1.0, degraded_qty=qty, original_qty=qty,
            sector=sector, reason="ok",
        )

    def commit(self, sector: str, qty: int, price: float) -> None:
        """Record an accepted (non-zero) allocation."""
        weight = qty * price / self._capital
        self._sector_weight[sector] = (
            self._sector_weight.get(sector, 0.0) + weight
        )
        self._total_allocated += qty * price
        self._committed_count += 1

    def record_degraded(self) -> None:
        self._degraded_count += 1

    def record_rejected(self) -> None:
        self._rejected_count += 1

    # ── diagnostics ──────────────────────────────────────────────────────────

    def sector_utilization(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self._sector_weight.items()}

    def capital_efficiency(self, available_cash: float) -> float:
        if available_cash <= 0:
            return 0.0
        return round(min(1.0, self._total_allocated / available_cash), 4)

    def summary(self, available_cash: float = 0.0, missed_entries: int = 0) -> str:
        return (
            f"capital_efficiency={self.capital_efficiency(available_cash):.3f} "
            f"committed={self._committed_count} "
            f"degraded={self._degraded_count} "
            f"rejected={self._rejected_count} "
            f"missed_entries={missed_entries} "
            f"sector_utilization={self.sector_utilization()}"
        )

    # ── forecast ─────────────────────────────────────────────────────────────

    def forecast(
        self,
        candidates: list[tuple[str, str, int, int, float]],
    ) -> list[ForecastFrame]:
        """
        Forward-scan across ranked candidates to predict sector exposure trajectory.

        candidates: [(symbol, sector, rsr_rank, qty, price), ...]  in rank order.
        Read-only — does not mutate self._sector_weight.
        """
        snap: dict[str, float] = dict(self._sector_weight)
        frames: list[ForecastFrame] = []

        for sym, sector, rsr_rank, qty, price in candidates:
            if qty <= 0 or price <= 0:
                frames.append(ForecastFrame(
                    symbol=sym, sector=sector, rsr_rank=rsr_rank,
                    requested_weight=0.0, multiplier=0.0,
                    effective_weight=0.0, rejected=True,
                ))
                continue

            requested_weight = qty * price / self._capital
            current_weight   = snap.get(sector, 0.0)
            remaining        = max(0.0, self._effective_cap - current_weight)
            multiplier       = min(1.0, remaining / requested_weight)

            if multiplier < self._reject_threshold:
                frames.append(ForecastFrame(
                    symbol=sym, sector=sector, rsr_rank=rsr_rank,
                    requested_weight=round(requested_weight, 4),
                    multiplier=round(multiplier, 4),
                    effective_weight=0.0,
                    rejected=True,
                ))
                continue

            degraded_qty = _round_lot(int(qty * multiplier))
            if degraded_qty == 0:
                frames.append(ForecastFrame(
                    symbol=sym, sector=sector, rsr_rank=rsr_rank,
                    requested_weight=round(requested_weight, 4),
                    multiplier=round(multiplier, 4),
                    effective_weight=0.0,
                    rejected=True,
                ))
                continue

            eff_w = degraded_qty * price / self._capital
            snap[sector] = current_weight + eff_w
            frames.append(ForecastFrame(
                symbol=sym, sector=sector, rsr_rank=rsr_rank,
                requested_weight=round(requested_weight, 4),
                multiplier=round(multiplier, 4),
                effective_weight=round(eff_w, 4),
                rejected=False,
            ))

        return frames


    # ── deployability pre-ranking ─────────────────────────────────────────────

    def pre_rank_candidates(
        self,
        candidates:         list[dict],
        available_cash:     float,
        max_alloc:          float,
        existing_positions: int,
        max_positions:      int,
    ) -> tuple[list[dict], object]:
        """
        Re-rank candidates by deployability_score BEFORE the main alloc loop.

        Wraps src.execution.deployability.rank_by_deployability using current
        sector exposure state from this allocator instance.

        candidates: list of dicts with keys:
            symbol, alpha_score, price, qty, sector

        Returns (ranked_candidates, DeployabilityDiagnostics).
        Deployable candidates surface before undeployable ones, reducing missed
        entries and idle capital accumulation.

        Read-only — does NOT mutate allocator sector state.
        """
        from src.execution.deployability import rank_by_deployability
        return rank_by_deployability(
            candidates         = candidates,
            available_cash     = available_cash,
            capital            = self._capital,
            sector_weights     = dict(self._sector_weight),
            sector_cap         = self._effective_cap,
            max_alloc          = max_alloc,
            existing_positions = existing_positions,
            max_positions      = max_positions,
            reject_threshold   = self._reject_threshold,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _round_lot(qty: int) -> int:
    return max(0, (qty // LOT_SIZE) * LOT_SIZE)
