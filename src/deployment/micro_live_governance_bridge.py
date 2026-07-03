"""
src/deployment/micro_live_governance_bridge.py
Phase 4B — Deterministic Micro Live Governance Bridge.

Purpose:
  Connect upstream governance layers to minimal live capital with
  deterministic telemetry. NOT execution automation.

  Answers:
    - What would be submitted to broker?
    - Does governance allow submission?
    - Does rollback/freeze block submission?
    - Is reconciliation deterministic?

Scope (Phase 4B ONLY produces):
  "deterministic broker intent telemetry"
  i.e., what WOULD be submitted, NOT automated execution.

HARD CONSTRAINTS (変更禁止):
  MAX_LIVE_POSITIONS     = 1
  MAX_LIVE_CAPITAL_PCT   = 0.02   (2% of total capital)
  MAX_LIVE_DAILY_ORDERS  = 1      (intent-level limit per day)
  MIN_FREEZE_DAYS        = 3
  DEFAULT_MODE           = HUMAN_CONFIRMED

Architecture:
  Upstream governance state
  ↓ MicroCapitalConstraints (hard limits validation)
  ↓ FreezeEvaluator (freeze persistence validation)
  ↓ ReconciliationChecker (position mismatch detection)
  ↓ BrokerIntentGenerator (deterministic intent with governance hash)
  ↓ GovernanceAuditRecord (append-only JSONL)
  → (HUMAN_CONFIRMED) BrokerInterface.submit_order()

禁止:
  - broker execution engine
  - async order management / retry orchestration / websocket
  - multiprocessing / thread addition / heavy pandas merge
  - new OHLCV fetch / new broker polling loop
  - runtime > 300ms target
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .connectors.broker_interface import (
    BrokerAccountState,
    BrokerInterface,
    BrokerOrderResult,
    BrokerPosition,
    LiveExecutionMode,
    DEFAULT_EXECUTION_MODE,
)
from .governance_audit import (
    FreezeEvaluator,
    GovernanceAuditRecord,
    GovernanceFreezeState,
    ReconciliationStatus,
    append_governance_audit,
    compute_governance_hash,
    load_freeze_state,
    save_freeze_state,
)

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ── Hard constraints (変更禁止) ───────────────────────────────────────────────
MAX_LIVE_POSITIONS:    int   = 1
MAX_LIVE_CAPITAL_PCT:  float = 0.02
MAX_LIVE_DAILY_ORDERS: int   = 1

# Reconciliation tolerance
_RECON_WEIGHT_TOLERANCE: float = 0.005   # 0.5% weight difference

# Replay hash mismatch triggers freeze
_REPLAY_HASH_FILE: str = "governance_replay_hash.json"


# ── Upstream governance state (read-only inputs) ──────────────────────────────

@dataclass(frozen=True)
class UpstreamGovernanceState:
    """
    Read-only snapshot of all upstream governance layer states.
    Each hash is a deterministic identifier for that layer's current state.

    current_live_symbols: symbols already active in live — used for reconciliation.
                          Empty if no live positions yet (normal for first-time entry).
    intended_symbols:     symbols governance wants to enter next cycle.
    """
    universe_version:          str
    deployment_manifest_hash:  str
    rollback_state:            str     # stable / watchlist / soft_rollback / full_rollback / suspended
    rollback_pressure_score:   float   # [0.0, 1.0]
    eligibility_gate_passed:   bool
    intended_symbols:          List[str]   # symbols governance wants to enter (new)
    intended_weights:          Dict[str, float]   # symbol → target_weight (sum ≤ MAX_LIVE_CAPITAL_PCT)
    upstream_hashes:           Dict[str, str]   # layer_name → hash
    current_live_symbols:      List[str] = field(default_factory=list)  # already-live positions
    current_live_weights:      Dict[str, float] = field(default_factory=dict)  # already-live weights
    broker_timeout:            bool = False
    api_integrity_unknown:     bool = False


@dataclass(frozen=True)
class ReconciliationResult:
    """
    Reconciliation between intended positions and broker positions.
    mismatch=True triggers LIVE_ROLLBACK candidate.
    """
    status:           str    # clean / mismatch / unknown / skipped
    mismatched_symbols: List[str]
    intended_count:   int
    broker_count:     int
    details:          str


@dataclass(frozen=True)
class BrokerIntent:
    """
    Deterministic broker intent for this evaluation cycle.
    order_eligible=True only when action=="enter" and governance allows.
    actual submission requires HUMAN_CONFIRMED mode.
    """
    symbol:           Optional[str]
    action:           str    # idle / enter / hold / reduce / exit / rollback / freeze
    target_weight:    float
    quantity_estimate: int   # estimated shares (informational, not binding)
    governance_hash:  str
    governance_blocked: bool
    block_reasons:    List[str]
    order_eligible:   bool   # True only when action=="enter" AND governance allows


@dataclass
class BridgeEvaluationReport:
    """
    Full output of one MicroLiveGovernanceBridge evaluation cycle.
    Written to append-only JSONL via GovernanceAuditRecord.
    """
    evaluation_date:        str
    live_mode:              str
    upstream:               UpstreamGovernanceState
    reconciliation:         ReconciliationResult
    freeze_state:           GovernanceFreezeState
    freeze_decisions:       List[str]
    broker_intents:         List[BrokerIntent]
    governance_audit:       GovernanceAuditRecord
    daily_order_count:      int
    max_daily_orders_hit:   bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_date":       self.evaluation_date,
            "live_mode":             self.live_mode,
            "freeze_active":         self.freeze_state.frozen,
            "freeze_day_count":      self.freeze_state.freeze_day_count,
            "reconciliation_status": self.reconciliation.status,
            "mismatched_symbols":    self.reconciliation.mismatched_symbols,
            "broker_intents": [
                {
                    "symbol":            bi.symbol,
                    "action":            bi.action,
                    "target_weight":     bi.target_weight,
                    "quantity_estimate": bi.quantity_estimate,
                    "order_eligible":    bi.order_eligible,
                    "governance_blocked": bi.governance_blocked,
                    "block_reasons":     bi.block_reasons,
                    "governance_hash":   bi.governance_hash,
                }
                for bi in self.broker_intents
            ],
            "daily_order_count":    self.daily_order_count,
            "max_daily_orders_hit": self.max_daily_orders_hit,
            "governance_audit":     self.governance_audit.to_dict(),
        }


# ── MicroCapitalConstraints validator ─────────────────────────────────────────

class MicroCapitalConstraints:
    """
    Hard constraint validation. Fail-closed on violations.
    All checks are deterministic and stateless.
    """

    @staticmethod
    def validate(
        intended_weights: Dict[str, float],
        daily_order_count: int,
    ) -> Tuple[bool, List[str]]:
        """
        Returns (passed, violations).
        Any violation → governance blocks the intent.
        """
        violations: List[str] = []

        total_weight = sum(intended_weights.values())
        symbol_count = len([w for w in intended_weights.values() if w > 0])

        if symbol_count > MAX_LIVE_POSITIONS:
            violations.append(
                f"MAX_LIVE_POSITIONS exceeded: {symbol_count} > {MAX_LIVE_POSITIONS}"
            )
        if total_weight > MAX_LIVE_CAPITAL_PCT + 1e-9:
            violations.append(
                f"MAX_LIVE_CAPITAL_PCT exceeded: {total_weight:.4f} > {MAX_LIVE_CAPITAL_PCT}"
            )
        if daily_order_count >= MAX_LIVE_DAILY_ORDERS:
            violations.append(
                f"MAX_LIVE_DAILY_ORDERS reached: {daily_order_count} >= {MAX_LIVE_DAILY_ORDERS}"
            )

        return len(violations) == 0, violations


# ── ReconciliationChecker ─────────────────────────────────────────────────────

class ReconciliationChecker:
    """
    Compare intended positions with broker positions.
    Mismatch → LIVE_ROLLBACK candidate.
    """

    def check(
        self,
        intended_symbols: List[str],
        broker_positions: List[BrokerPosition],
        total_capital:    float,
        intended_weights: Dict[str, float],
    ) -> ReconciliationResult:
        if not broker_positions and not intended_symbols:
            return ReconciliationResult(
                status=ReconciliationStatus.CLEAN,
                mismatched_symbols=[],
                intended_count=0,
                broker_count=0,
                details="both_empty",
            )

        broker_syms = {p.symbol for p in broker_positions if p.quantity > 0}
        intended_syms = set(intended_symbols)
        mismatches: List[str] = []

        # Symbols in broker but not intended (unexpected position)
        unexpected = broker_syms - intended_syms
        if unexpected:
            mismatches.extend(f"unexpected:{s}" for s in sorted(unexpected))

        # Symbols intended but not in broker (missing position)
        missing = intended_syms - broker_syms
        if missing:
            mismatches.extend(f"missing:{s}" for s in sorted(missing))

        # Weight mismatch for symbols in both
        if total_capital > 0:
            for pos in broker_positions:
                sym = pos.symbol
                if sym not in intended_weights:
                    continue
                actual_weight = pos.market_value / total_capital
                target_weight = intended_weights[sym]
                if abs(actual_weight - target_weight) > _RECON_WEIGHT_TOLERANCE:
                    mismatches.append(
                        f"weight_mismatch:{sym}:"
                        f"actual={actual_weight:.4f}:intended={target_weight:.4f}"
                    )

        status = ReconciliationStatus.MISMATCH if mismatches else ReconciliationStatus.CLEAN
        return ReconciliationResult(
            status=status,
            mismatched_symbols=mismatches,
            intended_count=len(intended_syms),
            broker_count=len(broker_syms),
            details="; ".join(mismatches) if mismatches else "ok",
        )


# ── Replay hash manager ────────────────────────────────────────────────────────

class ReplayHashManager:
    """
    Maintains rolling replay hash for mismatch detection.
    Compares today's governance hash with stored hash for same inputs.
    Mismatch on replay → freeze candidate.
    """

    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / _REPLAY_HASH_FILE

    def check_and_update(self, evaluation_date: str, governance_hash: str) -> bool:
        """
        Returns True if replay mismatch detected.
        Updates stored hash for this date.
        """
        stored: Dict[str, str] = {}
        if self._path.exists():
            try:
                stored = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                stored = {}

        existing = stored.get(evaluation_date)
        mismatch = existing is not None and existing != governance_hash

        if mismatch:
            logger.warning(
                "[REPLAY] mismatch on %s: stored=%s current=%s",
                evaluation_date, existing, governance_hash,
            )

        # Update stored hash
        stored[evaluation_date] = governance_hash
        # Keep only last 30 days to avoid unbounded growth
        if len(stored) > 30:
            oldest_keys = sorted(stored.keys())[:-30]
            for k in oldest_keys:
                del stored[k]

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("[REPLAY] hash save failed: %s", exc)

        return mismatch


# ── MicroLiveGovernanceBridge ─────────────────────────────────────────────────

class MicroLiveGovernanceBridge:
    """
    Deterministic Micro Live Governance Bridge.

    Usage (one call per trading day):
        bridge = MicroLiveGovernanceBridge(
            state_dir=Path("runtime/deployment/governance_audit"),
            audit_dir=Path("runtime/deployment/governance_audit"),
            mode=LiveExecutionMode.HUMAN_CONFIRMED,
        )
        report = bridge.evaluate(upstream_state, broker, total_capital, daily_orders_today)

    Outputs:
        - BridgeEvaluationReport (returned)
        - GovernanceAuditRecord  (appended to JSONL)
        - GovernanceFreezeState  (persisted to JSON)
    """

    def __init__(
        self,
        state_dir:  Path,
        audit_dir:  Path,
        mode:       LiveExecutionMode = DEFAULT_EXECUTION_MODE,
    ) -> None:
        self.state_dir  = state_dir
        self.audit_dir  = audit_dir
        self.mode       = mode
        self._freeze_path = state_dir / "freeze_state.json"
        self._freeze_evaluator  = FreezeEvaluator()
        self._recon_checker     = ReconciliationChecker()
        self._replay_manager    = ReplayHashManager(state_dir)
        self._constraint_validator = MicroCapitalConstraints()

    def evaluate(
        self,
        upstream:           UpstreamGovernanceState,
        broker:             Optional[BrokerInterface],
        total_capital:      float,
        daily_order_count:  int,
        evaluation_date:    Optional[str] = None,
    ) -> BridgeEvaluationReport:
        """
        One deterministic evaluation cycle.
        Runtime target: < 300ms.
        No external network calls beyond broker.get_positions() / get_account_state().
        """
        if evaluation_date is None:
            evaluation_date = datetime.now(JST).strftime("%Y-%m-%d")

        ts = datetime.now(JST).isoformat()

        # 1. Load freeze state
        freeze_state = load_freeze_state(self._freeze_path)

        # 2. Get broker state (fail-open for account_state, critical for positions)
        broker_positions: List[BrokerPosition] = []
        api_reachable = False
        if broker is not None:
            try:
                account = broker.get_account_state()
                broker_positions = account.positions
                api_reachable = account.api_reachable
            except Exception as exc:
                logger.warning("[BRIDGE] broker.get_account_state failed: %s", exc)

        # 3. Reconciliation check (against current_live_symbols, NOT intended new entries)
        reconciliation = self._recon_checker.check(
            intended_symbols=upstream.current_live_symbols,
            broker_positions=broker_positions,
            total_capital=total_capital,
            intended_weights=upstream.current_live_weights,
        )

        # 4. Constraint validation
        constraints_ok, constraint_violations = self._constraint_validator.validate(
            intended_weights=upstream.intended_weights,
            daily_order_count=daily_order_count,
        )
        max_daily_orders_hit = daily_order_count >= MAX_LIVE_DAILY_ORDERS

        # 5. Preliminary governance hash (before freeze decision so hash is stable)
        pre_hash = compute_governance_hash(
            universe_version=upstream.universe_version,
            deployment_manifest_hash=upstream.deployment_manifest_hash,
            rollback_state=upstream.rollback_state,
            freeze_state_frozen=freeze_state.frozen,
            governance_decisions=constraint_violations,
            live_mode=self.mode.value,
            upstream_hashes=upstream.upstream_hashes,
        )

        # 6. Replay mismatch detection
        replay_mismatch = self._replay_manager.check_and_update(evaluation_date, pre_hash)

        # 7. Freeze evaluation
        new_freeze_state, freeze_decisions = self._freeze_evaluator.evaluate(
            state=freeze_state,
            evaluation_date=evaluation_date,
            broker_timeout=upstream.broker_timeout,
            api_integrity_unknown=upstream.api_integrity_unknown,
            replay_mismatch=replay_mismatch,
            reconciliation_status=reconciliation.status,
            api_reachable=api_reachable,
            rollback_pressure_score=upstream.rollback_pressure_score,
        )

        # 8. Final governance decisions
        governance_decisions: List[str] = []
        rollback_decisions:   List[str] = []

        if not upstream.eligibility_gate_passed:
            governance_decisions.append("ELIGIBILITY_GATE_FAILED")
        if constraint_violations:
            governance_decisions.extend(constraint_violations)
        if upstream.rollback_state in ("soft_rollback", "full_rollback", "suspended"):
            rollback_decisions.append(f"ROLLBACK_ACTIVE: state={upstream.rollback_state}")
        if reconciliation.status == ReconciliationStatus.MISMATCH:
            rollback_decisions.append(f"RECONCILIATION_MISMATCH: {reconciliation.details}")

        # 9. Build broker intents
        broker_intents = self._build_intents(
            upstream=upstream,
            freeze_active=new_freeze_state.frozen,
            constraints_ok=constraints_ok,
            constraint_violations=constraint_violations,
            rollback_active=upstream.rollback_state in ("soft_rollback", "full_rollback", "suspended"),
            governance_decisions=governance_decisions,
            rollback_decisions=rollback_decisions,
            total_capital=total_capital,
            max_daily_orders_hit=max_daily_orders_hit,
        )

        # 10. Final deterministic hash (includes freeze decision outcome)
        all_decisions = governance_decisions + rollback_decisions + freeze_decisions
        final_hash = compute_governance_hash(
            universe_version=upstream.universe_version,
            deployment_manifest_hash=upstream.deployment_manifest_hash,
            rollback_state=upstream.rollback_state,
            freeze_state_frozen=new_freeze_state.frozen,
            governance_decisions=all_decisions,
            live_mode=self.mode.value,
            upstream_hashes=upstream.upstream_hashes,
        )

        # 11. Build GovernanceAuditRecord
        audit_record = GovernanceAuditRecord(
            evaluation_date=evaluation_date,
            upstream_hashes=dict(upstream.upstream_hashes),
            eligibility_gate_passed=upstream.eligibility_gate_passed,
            governance_decisions=governance_decisions,
            rollback_decisions=rollback_decisions,
            freeze_decisions=freeze_decisions,
            reconciliation_status=reconciliation.status,
            deterministic_hash=final_hash,
            live_mode=self.mode.value,
            freeze_active=new_freeze_state.frozen,
            freeze_day_count=new_freeze_state.freeze_day_count,
            consecutive_clean=new_freeze_state.consecutive_clean_reconciliations,
            timestamp=ts,
        )

        # 12. Persist freeze state and audit record (fail-closed)
        try:
            save_freeze_state(new_freeze_state, self._freeze_path)
        except Exception as exc:
            logger.error("[BRIDGE] freeze state save failed: %s", exc)
            raise

        try:
            append_governance_audit(audit_record, self.audit_dir)
        except Exception as exc:
            logger.error("[BRIDGE] audit record append failed: %s", exc)
            raise

        return BridgeEvaluationReport(
            evaluation_date=evaluation_date,
            live_mode=self.mode.value,
            upstream=upstream,
            reconciliation=reconciliation,
            freeze_state=new_freeze_state,
            freeze_decisions=freeze_decisions,
            broker_intents=broker_intents,
            governance_audit=audit_record,
            daily_order_count=daily_order_count,
            max_daily_orders_hit=max_daily_orders_hit,
        )

    def _build_intents(
        self,
        upstream:             UpstreamGovernanceState,
        freeze_active:        bool,
        constraints_ok:       bool,
        constraint_violations: List[str],
        rollback_active:      bool,
        governance_decisions: List[str],
        rollback_decisions:   List[str],
        total_capital:        float,
        max_daily_orders_hit: bool,
    ) -> List[BrokerIntent]:
        """Build deterministic broker intents. No side effects."""
        intents: List[BrokerIntent] = []
        all_block_reasons: List[str] = []

        if freeze_active:
            all_block_reasons.append("GOVERNANCE_FROZEN")
        if not upstream.eligibility_gate_passed:
            all_block_reasons.append("ELIGIBILITY_GATE_FAILED")
        if rollback_active:
            all_block_reasons.append(f"ROLLBACK:{upstream.rollback_state}")
        if constraint_violations:
            all_block_reasons.extend(constraint_violations)
        if max_daily_orders_hit:
            all_block_reasons.append("MAX_DAILY_ORDERS_HIT")

        governance_blocked = len(all_block_reasons) > 0

        # Determine action
        if freeze_active:
            action = "freeze"
        elif rollback_active and upstream.rollback_state == "full_rollback":
            action = "rollback"
        elif not upstream.eligibility_gate_passed:
            action = "idle"
        elif governance_blocked:
            action = "idle"
        elif upstream.intended_symbols:
            action = "enter"
        else:
            action = "idle"

        # Build intent per symbol
        for sym in (upstream.intended_symbols or [None]):
            weight = upstream.intended_weights.get(sym, 0.0) if sym else 0.0
            qty_est = 0
            if sym and total_capital > 0 and weight > 0:
                # Rough lot estimate (informational only — no actual price data)
                qty_est = max(1, int(total_capital * weight / 1000))  # assume ¥1000/share floor

            intent_hash = compute_governance_hash(
                universe_version=upstream.universe_version,
                deployment_manifest_hash=upstream.deployment_manifest_hash,
                rollback_state=upstream.rollback_state,
                freeze_state_frozen=freeze_active,
                governance_decisions=all_block_reasons,
                live_mode=self.mode.value,
                upstream_hashes={**(upstream.upstream_hashes), "symbol": sym or ""},
            )

            intents.append(BrokerIntent(
                symbol=sym,
                action=action,
                target_weight=weight,
                quantity_estimate=qty_est,
                governance_hash=intent_hash,
                governance_blocked=governance_blocked,
                block_reasons=list(all_block_reasons),
                order_eligible=(
                    action == "enter"
                    and not governance_blocked
                    and sym is not None
                    and self.mode == LiveExecutionMode.HUMAN_CONFIRMED
                ),
            ))

        if not intents:
            intent_hash = compute_governance_hash(
                universe_version=upstream.universe_version,
                deployment_manifest_hash=upstream.deployment_manifest_hash,
                rollback_state=upstream.rollback_state,
                freeze_state_frozen=freeze_active,
                governance_decisions=all_block_reasons,
                live_mode=self.mode.value,
                upstream_hashes=upstream.upstream_hashes,
            )
            intents.append(BrokerIntent(
                symbol=None,
                action=action,
                target_weight=0.0,
                quantity_estimate=0,
                governance_hash=intent_hash,
                governance_blocked=governance_blocked,
                block_reasons=list(all_block_reasons),
                order_eligible=False,
            ))

        return intents


# ── Convenience run hook ───────────────────────────────────────────────────────

def run_governance_bridge_hook(
    state_dir:         Path,
    audit_dir:         Path,
    upstream:          UpstreamGovernanceState,
    broker:            Optional[BrokerInterface],
    total_capital:     float,
    daily_order_count: int = 0,
    mode:              LiveExecutionMode = DEFAULT_EXECUTION_MODE,
    evaluation_date:   Optional[str] = None,
) -> BridgeEvaluationReport:
    """
    Stateless convenience wrapper for run_live_signal.py integration.
    Instantiates bridge, evaluates, returns report.
    """
    bridge = MicroLiveGovernanceBridge(
        state_dir=state_dir,
        audit_dir=audit_dir,
        mode=mode,
    )
    return bridge.evaluate(
        upstream=upstream,
        broker=broker,
        total_capital=total_capital,
        daily_order_count=daily_order_count,
        evaluation_date=evaluation_date,
    )
