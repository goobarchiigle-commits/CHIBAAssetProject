"""
src/execution/deferred_queue.py
Deferred entry queue — persistent storage for unexecuted entry intents.

STRICT INVARIANT:
  DeferredEntry is NOT a position.
  It MUST NOT affect: cash, equity, pnl, drawdown, exposure, holdings.
  It represents ONLY unexecuted execution intent.

Queue file: runtime/deferred_entries.json
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

JST             = timezone(timedelta(hours=9))
DEFAULT_TTL_DAYS = 3
MAX_RETRY_COUNT  = 5

ALLOWED_REJECT_REASONS: frozenset[str] = frozenset({
    "sector_cap",
    "lot_rounding",
    "alloc_cap",
    "insufficient_cash",
})


@dataclass
class DeferredEntry:
    """
    Unexecuted entry intent. NOT a position. NOT a virtual holding.

    accumulated_demand: fractional lot accumulator — observational only.
                        Has NO effect on cash, equity, or exposure.
    """
    entry_id:           str
    symbol:             str
    sector:             str
    strategy:           str
    signal_score:       float
    desired_weight:     float
    degraded_weight:    float
    reject_reason:      str
    created_at:         str
    ttl_days:           int
    retry_count:        int   = 0
    accumulated_demand: float = 0.0
    updated_at:         str   = ""
    last_failed_at:     str   = ""   # ISO timestamp of last failed retry (for cooldown)
    # Creation context — optional; None means "not captured at creation time"
    rsr_rank_at_creation:     int | None = None   # universe RSR rank (1-based, lower=better)
    sector_rank_at_creation:  int | None = None   # sector rank (1-based, lower=better)
    market_regime_at_creation: str | None = None  # MarketRegime value string

    def __post_init__(self) -> None:
        if self.reject_reason not in ALLOWED_REJECT_REASONS:
            raise ValueError(
                f"Invalid reject_reason={self.reject_reason!r}. "
                f"Allowed: {sorted(ALLOWED_REJECT_REASONS)}"
            )
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DeferredEntry:
        def _opt_int(v) -> int | None:
            return int(v) if v is not None else None

        return cls(
            entry_id                  = d["entry_id"],
            symbol                    = d["symbol"],
            sector                    = d["sector"],
            strategy                  = d["strategy"],
            signal_score              = float(d["signal_score"]),
            desired_weight            = float(d["desired_weight"]),
            degraded_weight           = float(d["degraded_weight"]),
            reject_reason             = d["reject_reason"],
            created_at                = d["created_at"],
            ttl_days                  = int(d["ttl_days"]),
            retry_count               = int(d.get("retry_count", 0)),
            accumulated_demand        = float(d.get("accumulated_demand", 0.0)),
            updated_at                = d.get("updated_at", d["created_at"]),
            last_failed_at            = d.get("last_failed_at", ""),
            rsr_rank_at_creation      = _opt_int(d.get("rsr_rank_at_creation")),
            sector_rank_at_creation   = _opt_int(d.get("sector_rank_at_creation")),
            market_regime_at_creation = d.get("market_regime_at_creation"),
        )

    @property
    def is_expired(self) -> bool:
        try:
            created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
            now     = datetime.now(JST)
            return (now - created).days >= self.ttl_days
        except Exception:
            return True

    @property
    def is_retryable(self) -> bool:
        return self.retry_count < MAX_RETRY_COUNT and not self.is_expired


def make_entry_id(symbol: str, strategy: str, created_at: str) -> str:
    payload = f"{symbol}|{strategy}|{created_at}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_deferred_entry(
    *,
    symbol:                    str,
    sector:                    str,
    strategy:                  str,
    signal_score:              float,
    desired_weight:            float,
    degraded_weight:           float,
    reject_reason:             str,
    ttl_days:                  int       = DEFAULT_TTL_DAYS,
    rsr_rank_at_creation:      int | None = None,
    sector_rank_at_creation:   int | None = None,
    market_regime_at_creation: str | None = None,
) -> DeferredEntry:
    now      = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
    entry_id = make_entry_id(symbol, strategy, now)
    return DeferredEntry(
        entry_id                  = entry_id,
        symbol                    = symbol,
        sector                    = sector,
        strategy                  = strategy,
        signal_score              = signal_score,
        desired_weight            = desired_weight,
        degraded_weight           = degraded_weight,
        reject_reason             = reject_reason,
        created_at                = now,
        ttl_days                  = ttl_days,
        updated_at                = now,
        rsr_rank_at_creation      = rsr_rank_at_creation,
        sector_rank_at_creation   = sector_rank_at_creation,
        market_regime_at_creation = market_regime_at_creation,
    )


class DeferredQueue:
    """
    Persistent deferred entry queue backed by JSON file.

    Thread-safety: single-process only (file writes are atomic via .tmp replace).

    INVARIANT: This queue has ZERO effect on portfolio state.
               Reads/writes are pure storage operations only.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load(self) -> list[DeferredEntry]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [DeferredEntry.from_dict(d) for d in data]
        except Exception as exc:
            logger.error("[DEFERRED_QUEUE] load failed: %s", exc)
            return []

    def save(self, entries: list[DeferredEntry]) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps([e.to_dict() for e in entries],
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception as exc:
            logger.error("[DEFERRED_QUEUE] save failed: %s", exc)
            raise

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add(self, entry: DeferredEntry) -> None:
        entries    = self.load()
        existing   = {e.entry_id for e in entries}
        if entry.entry_id in existing:
            logger.warning(
                "[DEFERRED_ADD] duplicate entry_id=%s symbol=%s — skipped",
                entry.entry_id, entry.symbol,
            )
            return
        entries.append(entry)
        self.save(entries)
        logger.info(
            "[DEFERRED_ADD] entry_id=%s symbol=%s sector=%s reason=%s "
            "desired_weight=%.4f degraded_weight=%.4f ttl_days=%d",
            entry.entry_id, entry.symbol, entry.sector, entry.reject_reason,
            entry.desired_weight, entry.degraded_weight, entry.ttl_days,
        )

    def update(self, entry: DeferredEntry) -> None:
        entries = self.load()
        for i, e in enumerate(entries):
            if e.entry_id == entry.entry_id:
                entries[i] = entry
                self.save(entries)
                return
        logger.warning("[DEFERRED_QUEUE] update: entry_id=%s not found", entry.entry_id)

    def remove(self, entry_id: str, reason: str = "") -> None:
        entries = self.load()
        before  = len(entries)
        entries = [e for e in entries if e.entry_id != entry_id]
        if len(entries) < before:
            logger.info("[DEFERRED_DROP] entry_id=%s reason=%s", entry_id, reason)
        self.save(entries)

    def remove_symbol(self, symbol: str, reason: str = "") -> None:
        entries = self.load()
        before  = len(entries)
        entries = [e for e in entries if e.symbol != symbol]
        dropped = before - len(entries)
        if dropped:
            logger.info("[DEFERRED_DROP] symbol=%s count=%d reason=%s",
                        symbol, dropped, reason)
        self.save(entries)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_active(self) -> list[DeferredEntry]:
        """Return non-expired, retryable entries."""
        return [e for e in self.load() if e.is_retryable]

    def expire_stale(self) -> list[DeferredEntry]:
        """Remove expired / exhausted entries. Returns list of dropped entries."""
        entries = self.load()
        active, expired = [], []
        for e in entries:
            if e.is_expired or e.retry_count >= MAX_RETRY_COUNT:
                expired.append(e)
                logger.info(
                    "[DEFERRED_DROP] entry_id=%s symbol=%s reason=stale "
                    "is_expired=%s retry_count=%d ttl_days=%d",
                    e.entry_id, e.symbol, e.is_expired, e.retry_count, e.ttl_days,
                )
            else:
                active.append(e)
        if expired:
            self.save(active)
        return expired
