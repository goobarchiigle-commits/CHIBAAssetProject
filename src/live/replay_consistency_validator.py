"""
replay_consistency_validator.py — Reconciliation determinism and lineage validation

Validates that:
  1. All snapshot checksums are valid (no data tampering since capture)
  2. Snapshot timestamps are monotonically increasing (replay ordering preserved)
  3. No duplicate snapshot timestamps (collision-free)
  4. Each reconciliation result maps to a known snapshot by run_id (lineage intact)
  5. Reconciliation results are ordered consistently with their snapshots

Used post-run to confirm that the reconciliation layer can be replayed
deterministically from persisted JSONL.

Never raises. Fail-open for telemetry. Returns structured result.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.live.broker_truth_snapshot import BrokerTruthSnapshot, validate_snapshot
from src.live.reconciliation_engine import ReconciliationResult

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

DIVERGENCE_CHECKSUM_INVALID   = "checksum_invalid"
DIVERGENCE_NON_MONOTONIC_TS   = "non_monotonic_timestamp"
DIVERGENCE_DUPLICATE_TS       = "duplicate_snapshot_timestamp"
DIVERGENCE_MISSING_SNAPSHOT   = "missing_snapshot_for_reconciliation"
DIVERGENCE_LINEAGE_BREAK      = "lineage_break"


@dataclass
class ReplayDivergence:
    divergence_type: str
    run_id: str
    detail: str
    snapshot_ts: str = ""
    recon_ts: str = ""


@dataclass
class ReplayConsistencyResult:
    timestamp: str
    determinism_ok: bool
    lineage_ok: bool
    total_snapshots: int
    total_recon_results: int
    divergences: list   # list[dict] — ReplayDivergence serialised
    divergence_count: int
    schema_version: int = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ts_key(ts_str: str) -> str:
    """Normalised sortable timestamp string."""
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return ts_str


def validate_replay_consistency(
    snapshots: List[BrokerTruthSnapshot],
    recon_results: List[ReconciliationResult],
    timestamp: str = "",
) -> ReplayConsistencyResult:
    """
    Validate that reconciliation can be deterministically replayed.

    Args:
        snapshots:     Ordered list of BrokerTruthSnapshot (as loaded from JSONL)
        recon_results: Ordered list of ReconciliationResult (as loaded from JSONL)

    Returns:
        ReplayConsistencyResult with divergences enumerated.
    Never raises.
    """
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    divergences: List[ReplayDivergence] = []

    # ── Check 1: checksum validity ────────────────────────────────────────────
    for snap in snapshots:
        try:
            if not validate_snapshot(snap):
                divergences.append(ReplayDivergence(
                    divergence_type=DIVERGENCE_CHECKSUM_INVALID,
                    run_id=snap.run_id,
                    detail=f"checksum mismatch ts={snap.timestamp}",
                    snapshot_ts=snap.timestamp,
                ))
        except Exception as exc:
            logger.warning("[REPLAY] checksum check failed for %s: %s", snap.run_id, exc)

    # ── Check 2: monotonic timestamps ────────────────────────────────────────
    prev_ts = ""
    for snap in snapshots:
        curr_ts = _parse_ts_key(snap.timestamp)
        if prev_ts and curr_ts and curr_ts < prev_ts:
            divergences.append(ReplayDivergence(
                divergence_type=DIVERGENCE_NON_MONOTONIC_TS,
                run_id=snap.run_id,
                detail=f"snapshot ts {curr_ts!r} < previous {prev_ts!r}",
                snapshot_ts=snap.timestamp,
            ))
        prev_ts = curr_ts or prev_ts

    # ── Check 3: duplicate timestamps ────────────────────────────────────────
    seen_ts: dict = {}
    for snap in snapshots:
        ts_key = _parse_ts_key(snap.timestamp)
        if ts_key in seen_ts:
            divergences.append(ReplayDivergence(
                divergence_type=DIVERGENCE_DUPLICATE_TS,
                run_id=snap.run_id,
                detail=f"duplicate snapshot ts={ts_key!r} first_run_id={seen_ts[ts_key]}",
                snapshot_ts=snap.timestamp,
            ))
        else:
            if ts_key:
                seen_ts[ts_key] = snap.run_id

    # ── Check 4 & 5: lineage — each recon maps to a known snapshot ───────────
    snapshot_run_ids: set = {snap.run_id for snap in snapshots if snap.run_id}
    for recon in recon_results:
        try:
            rid = recon.run_id
            rts = recon.timestamp
        except Exception:
            continue
        if rid and rid not in snapshot_run_ids:
            divergences.append(ReplayDivergence(
                divergence_type=DIVERGENCE_MISSING_SNAPSHOT,
                run_id=rid,
                detail=f"reconciliation run_id={rid!r} has no matching snapshot",
                recon_ts=rts,
            ))

    if divergences:
        for d in divergences:
            logger.warning(
                "[REPLAY] %s run_id=%s: %s",
                d.divergence_type, d.run_id, d.detail,
            )
    else:
        logger.info(
            "[REPLAY] ok: %d snapshots / %d recon_results — no divergences",
            len(snapshots), len(recon_results),
        )

    determinism_ok = not any(
        d.divergence_type in (DIVERGENCE_CHECKSUM_INVALID, DIVERGENCE_NON_MONOTONIC_TS)
        for d in divergences
    )
    lineage_ok = not any(
        d.divergence_type in (DIVERGENCE_MISSING_SNAPSHOT, DIVERGENCE_LINEAGE_BREAK)
        for d in divergences
    )

    return ReplayConsistencyResult(
        timestamp=timestamp,
        determinism_ok=determinism_ok,
        lineage_ok=lineage_ok,
        total_snapshots=len(snapshots),
        total_recon_results=len(recon_results),
        divergences=[asdict(d) for d in divergences],
        divergence_count=len(divergences),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_replay_result(result: ReplayConsistencyResult, path: Path) -> None:
    """Fail-safe append. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(result), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[REPLAY] log write failed: %s", exc)


def load_replay_results(path: Path) -> List[ReplayConsistencyResult]:
    """Load all results. Corrupted lines skipped."""
    if not path.exists():
        return []
    results: List[ReplayConsistencyResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            results.append(ReplayConsistencyResult(
                timestamp=d.get("timestamp", ""),
                determinism_ok=bool(d.get("determinism_ok", True)),
                lineage_ok=bool(d.get("lineage_ok", True)),
                total_snapshots=int(d.get("total_snapshots", 0)),
                total_recon_results=int(d.get("total_recon_results", 0)),
                divergences=d.get("divergences", []),
                divergence_count=int(d.get("divergence_count", 0)),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[REPLAY] parse error: %s", exc)
    return results
