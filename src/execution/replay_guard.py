"""
src/execution/replay_guard.py
Deferred replay storm protection.

Applied as a second pass AFTER evaluate_retry_candidates().
Enforces four independent guards in this order:

  1. Auction window  — block all retries during 08:59:00–09:00:30 JST
  2. Cooldown        — block entries whose last retry failed < 30 min ago
  3. Sector replay   — max 1 new deferred entry per sector per cycle
  4. Cycle throttle  — max 2 deferred retries admitted per cycle

Guards are evaluated per entry in signal-score-descending order (highest first).
Throttled / blocked entries stay in the queue — apply_retry_result is NOT called
for them, so retry_count is not incremented.

Logs:
  [DEFERRED_THROTTLE]     cycle quota exhausted
  [DEFERRED_SECTOR_BLOCK] sector quota exhausted this cycle
  [DEFERRED_COOLDOWN]     entry within cooldown window
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from typing import Optional

from src.execution.deferred_retry import RetryEvaluation

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReplayGuardConfig:
    max_retries_per_cycle:           int = 2
    max_entries_per_sector_per_cycle: int = 1
    cooldown_minutes:                int = 30
    # Open auction block window (JST HH, MM, SS)
    auction_block_start: tuple[int, int, int] = (8, 59,  0)
    auction_block_end:   tuple[int, int, int] = (9,  0, 30)


DEFAULT_CONFIG = ReplayGuardConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StormProtectionResult:
    """
    Partition of a RetryEvaluation list after storm protection is applied.

    admitted         — cleared for execution this cycle
    throttled        — cycle quota (max_retries_per_cycle) reached; stay queued
    blocked_cooldown — last_failed_at within cooldown window; stay queued
    blocked_sector   — sector quota (max_entries_per_sector) reached; stay queued
    blocked_auction  — open auction window active; all eligible entries held back
    non_eligible     — should_retry=False; caller must call apply_retry_result to drop
    """
    admitted:         list[RetryEvaluation] = field(default_factory=list)
    throttled:        list[RetryEvaluation] = field(default_factory=list)
    blocked_cooldown: list[RetryEvaluation] = field(default_factory=list)
    blocked_sector:   list[RetryEvaluation] = field(default_factory=list)
    blocked_auction:  list[RetryEvaluation] = field(default_factory=list)
    non_eligible:     list[RetryEvaluation] = field(default_factory=list)

    # ── Metric aliases ────────────────────────────────────────────────────────

    @property
    def deferred_replay_skipped(self) -> int:
        """Entries skipped due to open auction window."""
        return len(self.blocked_auction)

    @property
    def deferred_replay_throttled(self) -> int:
        """Entries throttled by cycle quota."""
        return len(self.throttled)

    @property
    def deferred_sector_blocked(self) -> int:
        """Entries blocked by per-sector cycle quota."""
        return len(self.blocked_sector)

    @property
    def total_suppressed(self) -> int:
        return (
            len(self.throttled)
            + len(self.blocked_cooldown)
            + len(self.blocked_sector)
            + len(self.blocked_auction)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────

class ReplayGuard:
    """
    Stateless storm protection filter.

    Usage:
        guard = ReplayGuard(config)
        result = guard.filter(evaluations, now=datetime.now(JST))

        for ev in result.admitted:
            executed = attempt_order(ev.entry)
            apply_retry_result(queue, ev, executed=executed)

        for ev in result.non_eligible:
            apply_retry_result(queue, ev, executed=False)

        # throttled / blocked_* — no action; entries remain in queue unchanged
    """

    def __init__(self, config: ReplayGuardConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config

    def filter(
        self,
        evaluations: list[RetryEvaluation],
        now:         Optional[datetime] = None,
    ) -> StormProtectionResult:
        """
        Apply all storm protection guards to an ordered evaluation list.

        Args:
            evaluations: output of evaluate_retry_candidates()
                         must be sorted: eligible first, highest score first
            now:         current JST datetime (injectable for tests; defaults to now)

        Returns:
            StormProtectionResult partitioning all evaluations
        """
        if now is None:
            now = datetime.now(JST)
        # Normalise to JST for all comparisons
        if now.tzinfo is None:
            now = now.replace(tzinfo=JST)

        result = StormProtectionResult()
        in_auction = self._in_auction_window(now)

        # Per-cycle state
        admitted_count:  int             = 0
        sector_admitted: dict[str, int]  = {}

        for ev in evaluations:
            # ── Non-eligible: pass through for caller to drop ────────────────
            if not ev.should_retry:
                result.non_eligible.append(ev)
                continue

            # ── Guard 1: open auction window — hold all eligible entries ─────
            if in_auction:
                result.blocked_auction.append(ev)
                logger.info(
                    "[DEFERRED_THROTTLE] entry_id=%s symbol=%s "
                    "reason=auction_window now=%s",
                    ev.entry.entry_id, ev.entry.symbol,
                    now.strftime("%H:%M:%S"),
                )
                continue

            # ── Guard 2: cooldown ─────────────────────────────────────────────
            cooldown_remaining = self._cooldown_remaining(ev, now)
            if cooldown_remaining > 0.0:
                result.blocked_cooldown.append(ev)
                logger.info(
                    "[DEFERRED_COOLDOWN] entry_id=%s symbol=%s "
                    "last_failed_at=%s cooldown_remaining=%.1fmin",
                    ev.entry.entry_id, ev.entry.symbol,
                    ev.entry.last_failed_at, cooldown_remaining,
                )
                continue

            # ── Guard 3: sector replay cap ────────────────────────────────────
            sector = ev.entry.sector
            if sector_admitted.get(sector, 0) >= self._cfg.max_entries_per_sector_per_cycle:
                result.blocked_sector.append(ev)
                logger.info(
                    "[DEFERRED_SECTOR_BLOCK] entry_id=%s symbol=%s sector=%s "
                    "sector_admitted=%d cap=%d",
                    ev.entry.entry_id, ev.entry.symbol, sector,
                    sector_admitted.get(sector, 0),
                    self._cfg.max_entries_per_sector_per_cycle,
                )
                continue

            # ── Guard 4: cycle throttle ───────────────────────────────────────
            if admitted_count >= self._cfg.max_retries_per_cycle:
                result.throttled.append(ev)
                logger.info(
                    "[DEFERRED_THROTTLE] entry_id=%s symbol=%s "
                    "admitted_count=%d max=%d",
                    ev.entry.entry_id, ev.entry.symbol,
                    admitted_count, self._cfg.max_retries_per_cycle,
                )
                continue

            # ── Admitted ──────────────────────────────────────────────────────
            result.admitted.append(ev)
            admitted_count += 1
            sector_admitted[sector] = sector_admitted.get(sector, 0) + 1

        logger.info(
            "[DEFERRED_THROTTLE] cycle_summary admitted=%d throttled=%d "
            "blocked_cooldown=%d blocked_sector=%d blocked_auction=%d "
            "non_eligible=%d",
            len(result.admitted), len(result.throttled),
            len(result.blocked_cooldown), len(result.blocked_sector),
            len(result.blocked_auction), len(result.non_eligible),
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _in_auction_window(self, now: datetime) -> bool:
        h, m, s  = self._cfg.auction_block_start
        start    = time(h, m, s)
        h2, m2, s2 = self._cfg.auction_block_end
        end      = time(h2, m2, s2)
        t        = now.astimezone(JST).time()
        return start <= t <= end

    def _cooldown_remaining(
        self,
        ev:  RetryEvaluation,
        now: datetime,
    ) -> float:
        """Returns remaining cooldown in minutes. 0.0 means cooldown has passed."""
        ts = ev.entry.last_failed_at
        if not ts:
            return 0.0
        try:
            failed_at    = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            cooldown_end = failed_at + timedelta(minutes=self._cfg.cooldown_minutes)
            remaining    = (cooldown_end - now).total_seconds() / 60.0
            return max(0.0, remaining)
        except Exception:
            return 0.0
