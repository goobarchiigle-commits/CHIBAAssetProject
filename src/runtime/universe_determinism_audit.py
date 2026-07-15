"""
src/runtime/universe_determinism_audit.py — DRY/LIVE universe divergence recorder.

Purpose: deterministically record WHY universe state diverged between DRY and LIVE,
or between snapshot stages within the same run.

Design:
  - append-only JSONL per calendar date
  - O(n) set diff only — no deep diff, no pandas merge
  - FAIL_OPEN: audit write failures never abort execution
  - WARNING emitted only when: same stage + same date + different symbol_hash
  - No global mutable state; all state derived from JSONL on each call

Storage: runtime/determinism/universe_audit/YYYY-MM-DD.jsonl
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Collection

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UniverseDeterminismAudit:
    ts: str                    # ISO-8601 UTC
    mode: str                  # "DRY" | "LIVE"
    snapshot_stage: str        # e.g. "post_price_filter", "post_governance", "rsr_context"
    total_symbols: int
    live_count: int
    shadow_count: int
    tradeable_count: int
    symbol_hash: str           # sha256(",".join(sorted(symbols)))
    added_symbols: list[str]   # vs most recent prior record for same mode+stage
    removed_symbols: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        # ensure Python native types only
        d["total_symbols"]   = int(d["total_symbols"])
        d["live_count"]      = int(d["live_count"])
        d["shadow_count"]    = int(d["shadow_count"])
        d["tradeable_count"] = int(d["tradeable_count"])
        return d


def _symbol_hash(symbols: Collection[str]) -> str:
    payload = ",".join(sorted(str(s) for s in symbols))
    return hashlib.sha256(payload.encode()).hexdigest()


def _today_log(audit_dir: Path) -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return audit_dir / f"{date_str}.jsonl"


def _load_prior(log_path: Path, mode: str, snapshot_stage: str) -> str | None:
    """Return symbol_hash of most recent prior record matching mode+stage, or None."""
    if not log_path.exists():
        return None
    last_hash: str | None = None
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("mode") == mode and rec.get("snapshot_stage") == snapshot_stage:
                    last_hash = rec.get("symbol_hash")
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return last_hash


def record_universe_snapshot(
    *,
    audit_dir: Path,
    mode: str,
    snapshot_stage: str,
    live_symbols: Collection[str],
    shadow_symbols: Collection[str] | None = None,
    tradeable_symbols: Collection[str] | None = None,
) -> UniverseDeterminismAudit | None:
    """
    Append one audit record for the given universe snapshot.

    Emits WARNING when same mode+stage already appears today with a different hash
    (divergence detected).

    Returns the audit record, or None on failure (FAIL_OPEN).
    """
    try:
        audit_dir.mkdir(parents=True, exist_ok=True)
        log_path = _today_log(audit_dir)

        live_set      = set(str(s) for s in live_symbols)
        shadow_set    = set(str(s) for s in (shadow_symbols or []))
        tradeable_set = set(str(s) for s in (tradeable_symbols or live_symbols))
        all_symbols   = live_set | shadow_set

        current_hash  = _symbol_hash(all_symbols)
        prior_hash    = _load_prior(log_path, mode, snapshot_stage)

        added   : list[str] = []
        removed : list[str] = []
        if prior_hash is not None:
            # Re-derive prior symbol sets from previous record for set diff
            prior_record = _find_prior_record(log_path, mode, snapshot_stage)
            if prior_record is not None:
                prior_live     = set(prior_record.get("_live_symbols",   []))
                prior_shadow   = set(prior_record.get("_shadow_symbols", []))
                prior_all      = prior_live | prior_shadow
                added   = sorted(all_symbols - prior_all)
                removed = sorted(prior_all   - all_symbols)

            if prior_hash != current_hash:
                logger.warning(
                    "[UNIVERSE_AUDIT] DIVERGENCE detected  mode=%s stage=%s  "
                    "prior_hash=%s...  current_hash=%s...  +%d -%d",
                    mode, snapshot_stage,
                    prior_hash[:12], current_hash[:12],
                    len(added), len(removed),
                )

        record = UniverseDeterminismAudit(
            ts              = datetime.now(timezone.utc).isoformat(),
            mode            = mode,
            snapshot_stage  = snapshot_stage,
            total_symbols   = len(all_symbols),
            live_count      = len(live_set),
            shadow_count    = len(shadow_set),
            tradeable_count = len(tradeable_set),
            symbol_hash     = current_hash,
            added_symbols   = added,
            removed_symbols = removed,
        )

        payload = record.to_dict()
        # Store live/shadow sets for future diff (not in dataclass to keep it clean)
        payload["_live_symbols"]   = sorted(live_set)
        payload["_shadow_symbols"] = sorted(shadow_set)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        logger.debug(
            "[UNIVERSE_AUDIT] recorded  mode=%s stage=%s total=%d hash=%s...",
            mode, snapshot_stage, record.total_symbols, current_hash[:12],
        )
        return record

    except Exception as exc:
        logger.warning("[UNIVERSE_AUDIT] record failed (FAIL_OPEN): %s", exc)
        return None


def _find_prior_record(log_path: Path, mode: str, snapshot_stage: str) -> dict | None:
    """Return the most recent JSONL record matching mode+stage."""
    if not log_path.exists():
        return None
    last: dict | None = None
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("mode") == mode and rec.get("snapshot_stage") == snapshot_stage:
                    last = rec
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return last
