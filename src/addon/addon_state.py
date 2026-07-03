"""
addon_state.py — Add-on execution state persistence

Two-tier persistence:
  addon_state.json     — current per-symbol state (counts, cooldowns).
                         Atomic overwrite on every change.
  addon_decisions.jsonl — append-only decision log (replay-compatible).

Thread-safety: all file I/O protected by module-level lock.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

JST = timezone(timedelta(hours=9))
logger = logging.getLogger(__name__)
_LOCK = threading.Lock()


@dataclass
class AddonSymbolState:
    """Per-symbol add-on execution state."""
    addon_count: int = 0
    last_addon_date: str = ""         # ISO date "YYYY-MM-DD" of last approved add-on
    total_addon_shares: int = 0
    last_confirmation_score: float = 0.0


@dataclass
class AddonDecisionRecord:
    """One append-only record written to addon_decisions.jsonl."""
    ts: str
    run_id: str
    symbol: str
    decision: str                     # "ADDON_APPROVED" | "ADDON_BLOCKED"
    block_reason: str
    addon_count_before: int
    addon_count_after: int
    addon_shares: int
    confirmation_score: float
    unrealized_pnl_pct: float
    hold_days: int
    rsr_rank: int
    rsr: float
    estimated_cost: float


def load_addon_state(path: Path) -> dict[str, AddonSymbolState]:
    """Load per-symbol add-on state. Returns empty dict if missing/corrupt."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, AddonSymbolState] = {}
        for sym, data in raw.items():
            if isinstance(data, dict):
                result[sym] = AddonSymbolState(**{
                    k: data.get(k, v)
                    for k, v in asdict(AddonSymbolState()).items()
                })
        return result
    except Exception as e:
        logger.warning("[ADDON_STATE] load failed (%s) — starting fresh", e)
        return {}


def save_addon_state(state: dict[str, AddonSymbolState], path: Path) -> None:
    """Atomically overwrite addon state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    data = {sym: asdict(s) for sym, s in state.items()}
    with _LOCK:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def append_decision_record(record: AddonDecisionRecord, path: Path) -> None:
    """Append one decision record to the JSONL log (fail-open)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
        with _LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception as e:
        logger.warning("[ADDON_STATE] decision log write failed (%s)", e)


def purge_exited_symbols(
    state: dict[str, AddonSymbolState],
    held_symbols: set[str],
) -> dict[str, AddonSymbolState]:
    """Remove state for symbols no longer held. Returns pruned copy."""
    return {sym: s for sym, s in state.items() if sym in held_symbols}
