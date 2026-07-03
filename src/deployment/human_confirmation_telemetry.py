"""
src/deployment/human_confirmation_telemetry.py
Phase 4B.1 — Human Confirmation Outcome Telemetry.

Observation only. No execution mutation. No auto-approval.

Purpose:
  Record the outcome of each governance-gate evaluation in HUMAN_CONFIRMED
  mode — whether the intent was approved, rejected, or modified externally.

Enables post-hoc analysis of:
  - human override frequency
  - governance distrust frequency
  - missed-alpha caused by human rejection
  - operator-induced drift

INVARIANTS:
  - append-only JSONL: records never mutated or deleted
  - approved and rejected are mutually exclusive
  - modified may coexist with approved (quantity/price change by human)
  - observation only: no execution path modification

JSONL: runtime/deployment/human_confirmation/human_confirmation_YYYYMMDD.jsonl
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


@dataclass(slots=True)
class HumanConfirmationOutcome:
    """
    One human-confirmation event at a HUMAN_CONFIRMED governance gate.

    Semantics:
      approved:  intent submitted as-is
      rejected:  intent suppressed (freeze active, or explicit rejection)
      modified:  intent submitted with qty/price/symbol change
      rejection_reason: populated when rejected=True

    Invariant: not (approved and rejected).
    modified may coexist with approved.
    """
    ts:               str
    execution_mode:   str
    symbol:           str
    intended_action:  str
    governance_hash:  str
    approved:         bool
    rejected:         bool
    modified:         bool
    rejection_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.approved and self.rejected:
            raise ValueError(
                "HumanConfirmationOutcome: approved and rejected are mutually exclusive"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts":               self.ts,
            "execution_mode":   self.execution_mode,
            "symbol":           self.symbol,
            "intended_action":  self.intended_action,
            "governance_hash":  self.governance_hash,
            "approved":         self.approved,
            "rejected":         self.rejected,
            "modified":         self.modified,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HumanConfirmationOutcome":
        return cls(
            ts=d["ts"],
            execution_mode=d.get("execution_mode", "human_confirmed"),
            symbol=d.get("symbol", ""),
            intended_action=d.get("intended_action", ""),
            governance_hash=d.get("governance_hash", ""),
            approved=bool(d.get("approved", False)),
            rejected=bool(d.get("rejected", False)),
            modified=bool(d.get("modified", False)),
            rejection_reason=d.get("rejection_reason"),
        )


# ── I/O helpers ───────────────────────────────────────────────────────────────

def append_human_confirmation(outcome: HumanConfirmationOutcome, conf_dir: Path) -> None:
    """
    Append HumanConfirmationOutcome to dated JSONL.
    Raises IOError on failure (telemetry integrity).
    """
    conf_dir.mkdir(parents=True, exist_ok=True)
    try:
        date_str = outcome.ts[:10].replace("-", "")
    except Exception:
        date_str = datetime.now(JST).strftime("%Y%m%d")
    out_path = conf_dir / f"human_confirmation_{date_str}.jsonl"
    line = json.dumps(outcome.to_dict(), ensure_ascii=False) + "\n"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line)


def load_human_confirmations(
    conf_dir: Path,
    date_str: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load human confirmation records from JSONL.
    date_str: YYYYMMDD for specific day; None loads all.
    """
    if not conf_dir.exists():
        return []
    records: List[Dict[str, Any]] = []
    if date_str:
        paths = [conf_dir / f"human_confirmation_{date_str}.jsonl"]
    else:
        paths = sorted(conf_dir.glob("human_confirmation_*.jsonl"))
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ── Convenience constructors ──────────────────────────────────────────────────

def make_approved_outcome(
    symbol:          str,
    intended_action: str,
    governance_hash: str,
    execution_mode:  str  = "human_confirmed",
    modified:        bool = False,
) -> HumanConfirmationOutcome:
    """Intent submitted as-is (or with modification)."""
    return HumanConfirmationOutcome(
        ts=datetime.now(JST).isoformat(),
        execution_mode=execution_mode,
        symbol=symbol,
        intended_action=intended_action,
        governance_hash=governance_hash,
        approved=True,
        rejected=False,
        modified=modified,
    )


def make_rejected_outcome(
    symbol:           str,
    intended_action:  str,
    governance_hash:  str,
    rejection_reason: str,
    execution_mode:   str = "human_confirmed",
) -> HumanConfirmationOutcome:
    """Intent suppressed — freeze active or explicit rejection."""
    return HumanConfirmationOutcome(
        ts=datetime.now(JST).isoformat(),
        execution_mode=execution_mode,
        symbol=symbol,
        intended_action=intended_action,
        governance_hash=governance_hash,
        approved=False,
        rejected=True,
        modified=False,
        rejection_reason=rejection_reason,
    )
