"""
src/universe/decision_context_capture.py — Phase 4C-lite: Decision Context Capture

Records a full environmental snapshot at each major governance decision point.
Does NOT change any decision logic. Observation and audit only.

Decision types captured:
  PROMOTION    — after run_universe_governance()
  DEMOTION     — after DemotionActuationEngine.run()
  ROTATION     — after RotationPlanner.plan()
  GOVERNANCE   — after GovernanceHealthEngine.run()
  AUTO_FREEZE  — after AutoFreezeEngine.run()

Persistence:
  runtime/universe/decision_context_history.jsonl (append-only)

Replay:
  replay_decision_context(date) → list of DecisionContext for that date
  daily_summary(date) → DailySummary with counts per decision_type
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# ── Decision type constants ────────────────────────────────────────────────────
DECISION_PROMOTION  = "PROMOTION"
DECISION_DEMOTION   = "DEMOTION"
DECISION_ROTATION   = "ROTATION"
DECISION_GOVERNANCE = "GOVERNANCE"
DECISION_AUTO_FREEZE = "AUTO_FREEZE"

ALL_DECISION_TYPES = [
    DECISION_PROMOTION,
    DECISION_DEMOTION,
    DECISION_ROTATION,
    DECISION_GOVERNANCE,
    DECISION_AUTO_FREEZE,
]


# ── Context dataclasses ────────────────────────────────────────────────────────

@dataclass
class MarketContext:
    market_regime: Optional[str] = None
    volatility_regime: Optional[str] = None
    turnover_rate: Optional[float] = None


@dataclass
class PortfolioContext:
    open_positions: Optional[int] = None
    portfolio_heat: Optional[float] = None
    cash_ratio: Optional[float] = None


@dataclass
class UniverseContext:
    live_universe_size: Optional[int] = None
    shadow_universe_size: Optional[int] = None
    pending_exit_count: Optional[int] = None


@dataclass
class GovernanceContext:
    health_score: Optional[float] = None
    governance_mode: Optional[str] = None
    auto_freeze_mode: Optional[str] = None
    effective_mode: Optional[str] = None


@dataclass
class DecisionContext:
    record_id: str
    timestamp: str
    decision_type: str
    symbol: Optional[str]
    market_context: Dict[str, Any]
    portfolio_context: Dict[str, Any]
    universe_context: Dict[str, Any]
    governance_context: Dict[str, Any]
    decision_inputs: Dict[str, Any]
    decision_result: Dict[str, Any]
    schema_version: int = SCHEMA_VERSION


@dataclass
class DailySummary:
    date: str
    decision_records: int = 0
    promotion_context_records: int = 0
    demotion_context_records: int = 0
    rotation_context_records: int = 0
    governance_context_records: int = 0
    auto_freeze_context_records: int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_serialize(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types to JSON-safe equivalents."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_safe_serialize(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _safe_serialize(asdict(obj))
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    try:
        return float(obj)
    except (TypeError, ValueError):
        pass
    return str(obj)


def _make_record_id(timestamp: str, decision_type: str, symbol: Optional[str]) -> str:
    key = f"{timestamp}:{decision_type}:{symbol or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _coerce_context(ctx: Any) -> Dict[str, Any]:
    """Convert a dataclass, dict, or None to a plain dict."""
    if ctx is None:
        return {}
    if isinstance(ctx, dict):
        return _safe_serialize(ctx)
    if hasattr(ctx, "__dataclass_fields__"):
        return _safe_serialize(asdict(ctx))
    return {}


# ── Main class ────────────────────────────────────────────────────────────────

class DecisionContextCapture:
    """Append-only recorder for governance decision contexts.

    FAIL_OPEN: any internal error is logged; the calling pipeline is never blocked.
    """

    def __init__(self, history_file: Path) -> None:
        self._history_file = Path(history_file)

    # ── Public API ─────────────────────────────────────────────────────────────

    def capture(
        self,
        decision_type: str,
        *,
        symbol: Optional[str] = None,
        market_context: Any = None,
        portfolio_context: Any = None,
        universe_context: Any = None,
        governance_context: Any = None,
        decision_inputs: Any = None,
        decision_result: Any = None,
    ) -> Optional[DecisionContext]:
        """Record one decision context. Returns the record or None on error."""
        try:
            ts = datetime.now(timezone.utc).isoformat()
            record = DecisionContext(
                record_id=_make_record_id(ts, decision_type, symbol),
                timestamp=ts,
                decision_type=decision_type,
                symbol=symbol,
                market_context=_coerce_context(market_context),
                portfolio_context=_coerce_context(portfolio_context),
                universe_context=_coerce_context(universe_context),
                governance_context=_coerce_context(governance_context),
                decision_inputs=_safe_serialize(decision_inputs) if decision_inputs is not None else {},
                decision_result=_safe_serialize(decision_result) if decision_result is not None else {},
                schema_version=SCHEMA_VERSION,
            )
            self._append(record)
            return record
        except Exception as exc:
            logger.warning("[DCC] capture failed (FAIL_OPEN): %s", exc)
            return None

    def replay_decision_context(self, date: str) -> List[DecisionContext]:
        """Return all DecisionContext records for the given date (YYYY-MM-DD)."""
        results: List[DecisionContext] = []
        try:
            if not self._history_file.exists():
                return results
            for line in self._history_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("timestamp", "").startswith(date):
                        results.append(DecisionContext(**d))
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("[DCC] replay failed (FAIL_OPEN): %s", exc)
        return results

    def daily_summary(self, date: str) -> DailySummary:
        """Return counts of each decision type recorded on date (YYYY-MM-DD)."""
        summary = DailySummary(date=date)
        try:
            records = self.replay_decision_context(date)
            summary.decision_records = len(records)
            for r in records:
                dt = r.decision_type
                if dt == DECISION_PROMOTION:
                    summary.promotion_context_records += 1
                elif dt == DECISION_DEMOTION:
                    summary.demotion_context_records += 1
                elif dt == DECISION_ROTATION:
                    summary.rotation_context_records += 1
                elif dt == DECISION_GOVERNANCE:
                    summary.governance_context_records += 1
                elif dt == DECISION_AUTO_FREEZE:
                    summary.auto_freeze_context_records += 1
        except Exception as exc:
            logger.warning("[DCC] daily_summary failed (FAIL_OPEN): %s", exc)
        return summary

    # ── Internal ───────────────────────────────────────────────────────────────

    def _append(self, record: DecisionContext) -> None:
        """Atomically append one JSON line to the history file."""
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._history_file.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                # Replay-safe: append to existing content
                if self._history_file.exists():
                    f.write(self._history_file.read_text(encoding="utf-8"))
                f.write(line)
            os.replace(tmp_path, self._history_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
