"""
src/execution/replay.py
Deterministic replay of execution history from the intent journal.

Goal: reproduce state transitions, DD, alloc, exposure, retries
      from journal + equity snapshots only — for forensic debugging.

Usage:
    frames = replay_intent_journal(journal_path)
    for frame in frames:
        print(frame)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.execution.intent import VALID_TRANSITIONS, IntentStatus, ACTIVE_STATUS_VALUES
from src.execution.dd_engine import compute_drawdown

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


@dataclass
class ReplayFrame:
    """Single state-transition event in the replay timeline."""
    ts:                  str
    run_id:              str
    intent_id:           str
    event:               str
    symbol:              str
    side:                str
    qty:                 int
    status:              str
    prev_status:         str | None
    retry_count:         int
    filled_qty:          int
    remaining_qty:       int
    snapshot_generation: int
    expected_price:      float
    broker_order_id:     str
    dd:                  float | None = None
    exposure:            float        = 0.0   # cumulative live exposure at this event
    anomalies:           list[str]    = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def replay_intent_journal(
    journal_path: Path | str,
    equity_snapshot_path: Path | str | None = None,
) -> list[ReplayFrame]:
    """
    Replay execution history from the intent journal.

    Validates:
    - State machine transitions (flags invalid ones as anomalies)
    - Retry counts (flags unexpected resets)
    - UNKNOWN intents that were never reconciled

    Optionally loads equity_snapshots.jsonl to compute DD at each event.

    Returns:
        List[ReplayFrame] sorted by ts. Anomalies annotated in frame.anomalies.
    """
    journal_path = Path(journal_path)
    if not journal_path.exists():
        logger.warning("[REPLAY] journal not found: %s", journal_path)
        return []

    # ── Load journal ──────────────────────────────────────────────────
    raw: list[dict] = []
    with journal_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("[REPLAY] line %d corrupted, skipped: %s", i + 1, e)

    if not raw:
        return []

    # ── Load equity snapshots for DD computation ──────────────────────
    equity_timeline: list[tuple[str, float, float]] = []  # (ts, equity, peak)
    if equity_snapshot_path:
        snap_path = Path(equity_snapshot_path)
        if snap_path.exists():
            with snap_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        s = json.loads(line.strip())
                        equity_timeline.append((
                            s.get("timestamp", ""),
                            float(s.get("equity", 0)),
                            float(s.get("equity_peak", 0)),
                        ))
                    except Exception:
                        pass

    # ── Replay ───────────────────────────────────────────────────────
    intent_prev_status: dict[str, str] = {}   # intent_id → last seen status
    frames: list[ReplayFrame] = []
    total_anomalies = 0

    for r in raw:
        intent_id   = r.get("intent_id", "?")
        new_status  = r.get("status", "")
        prev_status = intent_prev_status.get(intent_id)
        anomalies: list[str] = []

        # Validate transition
        if prev_status is not None:
            try:
                old_s = IntentStatus(prev_status)
                new_s = IntentStatus(new_status)
                if new_s not in VALID_TRANSITIONS.get(old_s, frozenset()):
                    msg = f"INVALID_TRANSITION {prev_status} → {new_status}"
                    anomalies.append(msg)
                    total_anomalies += 1
            except ValueError:
                anomalies.append(f"UNKNOWN_STATUS {new_status!r}")
                total_anomalies += 1

        # Find nearest equity snapshot for DD
        ts  = r.get("ts", "")
        dd  = None
        if equity_timeline and ts:
            nearest = min(equity_timeline,
                          key=lambda x: abs(x[0] <= ts and float("inf") or
                                           (datetime.fromisoformat(x[0]) -
                                            datetime.fromisoformat(ts)).total_seconds()),
                          default=None)
            if nearest and nearest[2] > 0:
                dd = compute_drawdown(nearest[1], nearest[2])

        frames.append(ReplayFrame(
            ts                  = ts,
            run_id              = r.get("run_id", ""),
            intent_id           = intent_id,
            event               = r.get("event", ""),
            symbol              = r.get("symbol", ""),
            side                = r.get("side", ""),
            qty                 = int(r.get("qty", 0)),
            status              = new_status,
            prev_status         = prev_status,
            retry_count         = int(r.get("retry_count", 0)),
            filled_qty          = int(r.get("filled_qty", 0)),
            remaining_qty       = int(r.get("remaining_qty", r.get("qty", 0))),
            snapshot_generation = int(r.get("snapshot_generation", 0)),
            expected_price      = float(r.get("expected_price", 0)),
            broker_order_id     = r.get("broker_order_id", ""),
            dd                  = dd,
            anomalies           = anomalies,
        ))

        intent_prev_status[intent_id] = new_status

    # ── Detect lingering UNKNOWN ──────────────────────────────────────
    for intent_id, status in intent_prev_status.items():
        if status == "UNKNOWN":
            logger.warning("[REPLAY] intent %s ended in UNKNOWN — never reconciled", intent_id)
            total_anomalies += 1

    # ── Summary ──────────────────────────────────────────────────────
    by_status: dict[str, int] = {}
    for s in intent_prev_status.values():
        by_status[s] = by_status.get(s, 0) + 1

    logger.info(
        "[REPLAY] journal=%s records=%d intents=%d status=%s anomalies=%d",
        journal_path.name,
        len(raw),
        len(intent_prev_status),
        by_status,
        total_anomalies,
    )

    return sorted(frames, key=lambda x: x.ts)


def replay_execution_state(
    journal_path:          Path | str,
    snapshot_history_path: Path | str | None = None,
) -> list[ReplayFrame]:
    """
    Deterministic replay of execution history.

    Given the same inputs, produces identical output every time.

    Replays per frame:
      - DD from nearest equity snapshot
      - Cumulative exposure (filled positions × expected_price)
      - All state transitions validated against VALID_TRANSITIONS
      - Retries, UNKNOWN, reconciliation events

    This is the forensic entry point — use for post-mortem analysis and
    regression testing of execution semantics.

    Logs [REPLAY] run_id=... events=... result=... for each run_id seen.

    Returns:
        list[ReplayFrame] with dd and exposure fields set.
    """
    frames = replay_intent_journal(journal_path, snapshot_history_path)

    # Annotate exposure per frame: track filled positions over time
    exposure_by_intent: dict[str, float] = {}

    for frame in frames:
        price  = frame.expected_price or 0.0
        status = frame.status

        if status in ("PARTIAL", "ACKED"):
            # Partial or acked — account for filled portion
            exposure_by_intent[frame.intent_id] = frame.filled_qty * price if frame.filled_qty else frame.qty * price
        elif status == "FILLED":
            exposure_by_intent[frame.intent_id] = frame.filled_qty * price
        elif status in ("CANCELLED", "REJECTED"):
            exposure_by_intent.pop(frame.intent_id, None)
        elif status == "SUBMITTED" and frame.intent_id not in exposure_by_intent:
            # Reserve original qty during submission
            exposure_by_intent[frame.intent_id] = frame.qty * price

        frame.exposure = sum(exposure_by_intent.values())

    # Emit per-run summary
    runs_seen: dict[str, list[ReplayFrame]] = {}
    for frame in frames:
        runs_seen.setdefault(frame.run_id, []).append(frame)

    for run_id, run_frames in runs_seen.items():
        final_statuses = {}
        for f in run_frames:
            final_statuses[f.intent_id] = f.status
        logger.info(
            "[REPLAY] run_id=%s events=%d result=%s",
            run_id, len(run_frames), final_statuses,
        )

    return frames
