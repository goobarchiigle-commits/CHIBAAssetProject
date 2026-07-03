"""ExecutionRealityArtifacts — routes artifact writes to artifacts/execution_reality/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.research.analysis.artifacts import (
    write_immutable, append_jsonl, read_jsonl, ImmutabilityError,
)


class ExecutionRealityArtifacts:
    """Handles persistence routing for execution reality governance artifacts."""

    SUBDIRS = (
        "snapshots",
        "divergence",
        "regime",
        "slippage",
        "fill_quality",
        "latency",
        "liquidity",
        "comparator",
        "decay",
        "decisions",
        "replay",
    )

    def __init__(self, base_path: Path = Path("artifacts/execution_reality")):
        self._base = base_path

    def _ensure(self, subdir: str) -> Path:
        p = self._base / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Snapshots ──────────────────────────────────────────────────────────────

    def write_snapshot(self, snapshot_id: str, payload: dict) -> None:
        p = self._ensure("snapshots") / f"{snapshot_id}.json"
        write_immutable(p, payload)

    def write_snapshot_index(self, session_id: str, payload: dict) -> None:
        p = self._ensure("snapshots") / f"index_{session_id}.json"
        write_immutable(p, payload)

    # ── Divergence ─────────────────────────────────────────────────────────────

    def append_divergence_record(self, session_id: str, record: dict) -> None:
        p = self._ensure("divergence") / f"{session_id}.jsonl"
        append_jsonl(p, record)

    def read_divergence_records(self, session_id: str) -> List[dict]:
        p = self._base / "divergence" / f"{session_id}.jsonl"
        return read_jsonl(p)

    def write_divergence_summary(self, session_id: str, payload: dict) -> None:
        p = self._ensure("divergence") / f"summary_{session_id}.json"
        write_immutable(p, payload)

    # ── Regime ─────────────────────────────────────────────────────────────────

    def write_regime_classification(self, classification_id: str, payload: dict) -> None:
        p = self._ensure("regime") / f"{classification_id}.json"
        write_immutable(p, payload)

    # ── Slippage ───────────────────────────────────────────────────────────────

    def write_slippage_summary(self, session_id: str, payload: dict) -> None:
        p = self._ensure("slippage") / f"summary_{session_id}.json"
        write_immutable(p, payload)

    def append_slippage_observation(self, session_id: str, record: dict) -> None:
        p = self._ensure("slippage") / f"{session_id}.jsonl"
        append_jsonl(p, record)

    # ── Fill Quality ───────────────────────────────────────────────────────────

    def write_fill_quality_summary(self, session_id: str, payload: dict) -> None:
        p = self._ensure("fill_quality") / f"summary_{session_id}.json"
        write_immutable(p, payload)

    # ── Latency ────────────────────────────────────────────────────────────────

    def write_latency_summary(self, session_id: str, payload: dict) -> None:
        p = self._ensure("latency") / f"summary_{session_id}.json"
        write_immutable(p, payload)

    # ── Liquidity ──────────────────────────────────────────────────────────────

    def write_liquidity_summary(self, session_id: str, payload: dict) -> None:
        p = self._ensure("liquidity") / f"summary_{session_id}.json"
        write_immutable(p, payload)

    # ── Comparator ─────────────────────────────────────────────────────────────

    def write_comparison_summary(self, session_id: str, payload: dict) -> None:
        p = self._ensure("comparator") / f"summary_{session_id}.json"
        write_immutable(p, payload)

    # ── Decay ──────────────────────────────────────────────────────────────────

    def write_decay_summary(self, session_id: str, payload: dict) -> None:
        p = self._ensure("decay") / f"summary_{session_id}.json"
        write_immutable(p, payload)

    # ── Decisions ──────────────────────────────────────────────────────────────

    def write_decision(self, decision_id: str, payload: dict) -> None:
        p = self._ensure("decisions") / f"{decision_id}.json"
        write_immutable(p, payload)

    def append_decision(self, session_id: str, record: dict) -> None:
        p = self._ensure("decisions") / f"{session_id}.jsonl"
        append_jsonl(p, record)

    def read_decisions(self, session_id: str) -> List[dict]:
        p = self._base / "decisions" / f"{session_id}.jsonl"
        return read_jsonl(p)

    # ── Replay ─────────────────────────────────────────────────────────────────

    def write_replay_result(self, result_id: str, payload: dict) -> None:
        p = self._ensure("replay") / f"{result_id}.json"
        write_immutable(p, payload)

    def write_replay_summary(self, session_id: str, payload: dict) -> None:
        p = self._ensure("replay") / f"summary_{session_id}.json"
        write_immutable(p, payload)
