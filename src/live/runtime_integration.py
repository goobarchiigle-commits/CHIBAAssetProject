"""
runtime_integration.py — Deterministic runtime integration stabilization layer

Eliminates orchestration drift, invalid runtime assumptions, and broker/runtime
state inconsistencies in both DRY and LIVE execution modes.

Fail policy:
  execution-critical:  FAIL_CLOSED (raises RuntimeContractError)
  analytics-only:      FAIL_OPEN (structured warnings, persisted)

Artifacts: runtime/integration/ (append-only JSONL, replay-compatible)
Schema version: 1
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1

# ─────────────────────────────────────────────────────────────────────────────
# Stage constants (must match StagedSupervisor stage names)
# ─────────────────────────────────────────────────────────────────────────────

STAGE_API_CHECK         = "api_check"
STAGE_BOOTSTRAP         = "bootstrap"
STAGE_BROKER_SYNC       = "broker_sync"
STAGE_RECONCILIATION    = "reconciliation"
STAGE_SIGNAL_GENERATION = "signal_generation"
STAGE_MARKET_DATA       = "market_data"
STAGE_ALLOCATION        = "allocation"
STAGE_ORDER_EXECUTION   = "order_execution"
STAGE_CLEANUP           = "cleanup"

# Directed edges: stage → required prior stages
STAGE_PRECONDITIONS: Dict[str, List[str]] = {
    STAGE_API_CHECK:         [],
    STAGE_BOOTSTRAP:         [],
    STAGE_BROKER_SYNC:       [STAGE_API_CHECK],
    STAGE_MARKET_DATA:       [STAGE_BOOTSTRAP],
    STAGE_SIGNAL_GENERATION: [STAGE_MARKET_DATA],
    STAGE_RECONCILIATION:    [STAGE_BROKER_SYNC],
    STAGE_ALLOCATION:        [STAGE_SIGNAL_GENERATION, STAGE_BROKER_SYNC],
    STAGE_ORDER_EXECUTION:   [STAGE_ALLOCATION, STAGE_RECONCILIATION],
    STAGE_CLEANUP:           [STAGE_ORDER_EXECUTION],
}


# ─────────────────────────────────────────────────────────────────────────────
# Exception hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class RuntimeContractError(RuntimeError):
    """Base class for all runtime contract violations (fail-closed)."""

    def __init__(self, message: str, violation_type: str = "contract_violation") -> None:
        super().__init__(message)
        self.violation_type = violation_type


class NullStateAccessError(RuntimeContractError):
    """Critical state is None when accessed."""

    def __init__(self, state_name: str) -> None:
        super().__init__(
            f"Required runtime state '{state_name}' is None — cannot proceed",
            violation_type="null_state_access",
        )
        self.state_name = state_name


class DependencyMissingError(RuntimeContractError):
    """A required runtime dependency is not registered/initialized."""

    def __init__(self, dependency: str, required_by: str) -> None:
        super().__init__(
            f"Runtime dependency '{dependency}' required by '{required_by}' is missing",
            violation_type="dependency_missing",
        )
        self.dependency   = dependency
        self.required_by  = required_by


class StagePreconditionError(RuntimeContractError):
    """Stage executed before its precondition stages completed."""

    def __init__(self, stage: str, missing_preconditions: List[str]) -> None:
        super().__init__(
            f"Stage '{stage}' preconditions not met: {missing_preconditions}",
            violation_type="stage_precondition_violation",
        )
        self.stage                = stage
        self.missing_preconditions = missing_preconditions


class InitializationOrderError(RuntimeContractError):
    """Module initialized out of required topological order."""

    def __init__(self, module: str, unresolved_deps: List[str]) -> None:
        super().__init__(
            f"Module '{module}' initialized before dependencies: {unresolved_deps}",
            violation_type="initialization_order_error",
        )
        self.module          = module
        self.unresolved_deps = unresolved_deps


class InterfaceSignatureMismatchError(RuntimeContractError):
    """Injected dependency has an incompatible interface signature."""

    def __init__(self, component: str, missing_attrs: List[str]) -> None:
        super().__init__(
            f"Component '{component}' missing required attributes: {missing_attrs}",
            violation_type="interface_signature_mismatch",
        )
        self.component     = component
        self.missing_attrs = missing_attrs


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeStateRegistry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegistryEntry:
    name:          str
    value:         Any
    registered_at: str
    expected_type: Optional[str]   # qualified type name string (for JSON serialization)
    is_critical:   bool
    ttl_sec:       Optional[float]


class RuntimeStateRegistry:
    """
    Tracks registered runtime state objects.

    - Critical entries: None value raises NullStateAccessError
    - Non-critical entries: None value emits a warning
    - Stale entries (TTL exceeded): warning, never blocks
    - Type mismatch: warning only (not blocking by design)
    """

    def __init__(self) -> None:
        self._entries: Dict[str, RegistryEntry] = {}
        self._access_log: List[Dict[str, Any]] = []

    def register(
        self,
        name:          str,
        value:         Any,
        expected_type: Optional[Type] = None,
        is_critical:   bool = True,
        ttl_sec:       Optional[float] = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        type_name = f"{expected_type.__module__}.{expected_type.__qualname__}" if expected_type else None
        entry = RegistryEntry(
            name=name,
            value=value,
            registered_at=ts,
            expected_type=type_name,
            is_critical=is_critical,
            ttl_sec=ttl_sec,
        )
        self._entries[name] = entry
        logger.debug("[REGISTRY] registered: %s is_critical=%s", name, is_critical)

    def get(self, name: str, now_sec: Optional[float] = None) -> Any:
        entry = self._entries.get(name)
        if entry is None:
            raise RuntimeContractError(
                f"State '{name}' not registered in RuntimeStateRegistry",
                violation_type="unregistered_state_access",
            )

        ts = datetime.now(timezone.utc).isoformat()
        access_record = {"name": name, "accessed_at": ts, "value_is_none": entry.value is None}

        if entry.value is None:
            access_record["null_access"] = True
            self._access_log.append(access_record)
            if entry.is_critical:
                raise NullStateAccessError(name)
            logger.warning("[REGISTRY] non-critical state '%s' is None", name)
            return None

        if entry.ttl_sec is not None and now_sec is not None:
            try:
                registered_ts = datetime.fromisoformat(entry.registered_at)
                if registered_ts.tzinfo is None:
                    registered_ts = registered_ts.replace(tzinfo=timezone.utc)
                age = now_sec - registered_ts.timestamp()
                if age > entry.ttl_sec:
                    logger.warning(
                        "[REGISTRY] stale state '%s': age=%.0fs > ttl=%.0fs",
                        name, age, entry.ttl_sec,
                    )
                    access_record["stale"] = True
                    access_record["age_sec"] = age
            except Exception:
                pass

        self._access_log.append(access_record)
        return entry.value

    def validate_all(self, now_sec: Optional[float] = None) -> List[str]:
        """Return list of violation strings. Empty = ok."""
        violations: List[str] = []
        if now_sec is None:
            now_sec = time.time()

        for name, entry in self._entries.items():
            if entry.value is None:
                severity = "CRITICAL" if entry.is_critical else "WARN"
                violations.append(
                    f"[REGISTRY:{severity}] state '{name}' is None "
                    f"(critical={entry.is_critical})"
                )
                continue

            if entry.ttl_sec is not None:
                try:
                    registered_ts = datetime.fromisoformat(entry.registered_at)
                    if registered_ts.tzinfo is None:
                        registered_ts = registered_ts.replace(tzinfo=timezone.utc)
                    age = now_sec - registered_ts.timestamp()
                    if age > entry.ttl_sec:
                        violations.append(
                            f"[REGISTRY:WARN] state '{name}' stale: "
                            f"age={age:.0f}s > ttl={entry.ttl_sec:.0f}s"
                        )
                except Exception:
                    pass

        return violations

    def dump_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "entry_count": len(self._entries),
            "entries": {
                name: {
                    "registered_at": e.registered_at,
                    "expected_type": e.expected_type,
                    "is_critical":   e.is_critical,
                    "ttl_sec":       e.ttl_sec,
                    "value_is_none": e.value is None,
                }
                for name, e in self._entries.items()
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# IntegrationDependencyGraph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DependencyNode:
    name:           str
    dependencies:   List[str]
    initialized:    bool = False
    initialized_at: Optional[str] = None


class IntegrationDependencyGraph:
    """
    Directed acyclic dependency graph for runtime module initialization.

    - declare(): register a module with its dependencies
    - mark_initialized(): record a module as ready
    - validate(): return violations (missing deps, uninitialized required nodes)
    - topological_order(): deterministic BFS ordering for manifest generation
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, DependencyNode] = {}

    def declare(self, name: str, dependencies: Optional[List[str]] = None) -> None:
        deps = list(dependencies or [])
        if name in self._nodes:
            self._nodes[name].dependencies = deps
        else:
            self._nodes[name] = DependencyNode(name=name, dependencies=deps)

    def mark_initialized(
        self,
        name: str,
        ts: Optional[str] = None,
        strict_order: bool = False,
    ) -> None:
        """
        Record a module as initialized.

        If strict_order=True, raises InitializationOrderError when any declared
        dependency is not yet initialized.
        """
        if name not in self._nodes:
            self._nodes[name] = DependencyNode(name=name, dependencies=[])

        if strict_order:
            unresolved = [
                dep for dep in self._nodes[name].dependencies
                if dep in self._nodes and not self._nodes[dep].initialized
            ]
            if unresolved:
                raise InitializationOrderError(name, unresolved)

        self._nodes[name].initialized    = True
        self._nodes[name].initialized_at = ts or datetime.now(timezone.utc).isoformat()

    def validate(self) -> List[str]:
        """Return list of violations. Empty = dependency graph satisfied."""
        violations: List[str] = []

        for name, node in self._nodes.items():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    violations.append(
                        f"[DEPGRAPH] '{name}' requires undeclared dependency '{dep}'"
                    )
                elif not self._nodes[dep].initialized:
                    violations.append(
                        f"[DEPGRAPH] '{name}' dependency '{dep}' not initialized"
                    )

        return violations

    def topological_order(self) -> List[str]:
        """Return BFS topological ordering for deterministic manifest output."""
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node.name] = in_degree.get(node.name, 0) + 1

        queue = sorted(n for n, d in in_degree.items() if d == 0)
        result: List[str] = []
        adj: Dict[str, List[str]] = {n: [] for n in self._nodes}
        for n, node in self._nodes.items():
            for dep in node.dependencies:
                if dep in adj:
                    adj[dep].append(n)

        while queue:
            node_name = queue.pop(0)
            result.append(node_name)
            for neighbor in sorted(adj.get(node_name, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        remaining = sorted(set(self._nodes) - set(result))
        result.extend(remaining)
        return result

    def dump_manifest(self) -> Dict[str, Any]:
        order = self.topological_order()
        return {
            "schema_version": SCHEMA_VERSION,
            "node_count": len(self._nodes),
            "topological_order": order,
            "nodes": {
                name: {
                    "dependencies":   self._nodes[name].dependencies,
                    "initialized":    self._nodes[name].initialized,
                    "initialized_at": self._nodes[name].initialized_at,
                }
                for name in order
                if name in self._nodes
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# StagePreconditionValidator
# ─────────────────────────────────────────────────────────────────────────────

class StagePreconditionValidator:
    """
    Validates stage execution ordering against declared preconditions.

    Completed stages are recorded by record_stage_complete().
    validate_preconditions() returns violations without raising.
    assert_preconditions() raises StagePreconditionError on any violation.
    """

    def __init__(
        self,
        preconditions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._preconditions = preconditions if preconditions is not None else dict(STAGE_PRECONDITIONS)
        self._completed: Dict[str, str] = {}  # stage → ISO timestamp

    def record_stage_complete(self, stage: str, ts: Optional[str] = None) -> None:
        self._completed[stage] = ts or datetime.now(timezone.utc).isoformat()

    def validate_preconditions(self, stage: str) -> List[str]:
        """Return list of unmet preconditions. Empty = ok to proceed."""
        required = self._preconditions.get(stage, [])
        return [req for req in required if req not in self._completed]

    def assert_preconditions(self, stage: str) -> None:
        """Raises StagePreconditionError if any precondition is unmet."""
        missing = self.validate_preconditions(stage)
        if missing:
            raise StagePreconditionError(stage, missing)

    def completed_stages(self) -> Dict[str, str]:
        return dict(self._completed)

    def dump_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version":   SCHEMA_VERSION,
            "completed_stages": self._completed,
            "preconditions":    self._preconditions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeCoherenceValidator
# ─────────────────────────────────────────────────────────────────────────────

# Required attributes per well-known component interface
_ALLOCATOR_REQUIRED_ATTRS: List[str] = ["return_buffers"]
_CAP_STATE_REQUIRED_ATTRS: List[str] = [
    "actual_equity", "effective_capital", "risk_adjusted_capital",
]
_EXPOSURE_STATE_REQUIRED_ATTRS: List[str] = ["return_buffers"]

# Required attributes for compute_exposure_report()-compatible allocators
_EXPOSURE_REPORT_REQUIRED_ATTRS: List[str] = ["return_buffers"]


class RuntimeCoherenceValidator:
    """
    Validates coherence of runtime state objects:
      - None checks
      - Required attribute presence
      - Type compatibility
      - Allocator integration interface correctness

    All checks return violation lists (fail-open).
    Use assert_coherent() for fail-closed enforcement.
    """

    def check_none(
        self,
        name:     str,
        value:    Any,
        critical: bool = True,
    ) -> List[str]:
        if value is None:
            severity = "CRITICAL" if critical else "WARN"
            return [f"[COHERENCE:{severity}] '{name}' is None (critical={critical})"]
        return []

    def check_type(
        self,
        name:          str,
        value:         Any,
        expected_type: Type,
        critical:      bool = False,
    ) -> List[str]:
        if value is not None and not isinstance(value, expected_type):
            severity = "CRITICAL" if critical else "WARN"
            actual = type(value).__name__
            expected = expected_type.__name__
            return [
                f"[COHERENCE:{severity}] '{name}' type={actual} != expected={expected}"
            ]
        return []

    def check_required_attrs(
        self,
        name:     str,
        obj:      Any,
        attrs:    List[str],
        critical: bool = True,
    ) -> List[str]:
        if obj is None:
            return []
        missing = [a for a in attrs if not hasattr(obj, a)]
        if missing:
            severity = "CRITICAL" if critical else "WARN"
            return [
                f"[COHERENCE:{severity}] '{name}' missing attributes: {missing}"
            ]
        return []

    def validate_cap_state(self, cap_state: Any) -> List[str]:
        violations: List[str] = []
        violations.extend(self.check_none("cap_state", cap_state, critical=True))
        if cap_state is None:
            return violations
        violations.extend(
            self.check_required_attrs("cap_state", cap_state, _CAP_STATE_REQUIRED_ATTRS)
        )
        if hasattr(cap_state, "actual_equity"):
            if cap_state.actual_equity is None:
                violations.append("[COHERENCE:CRITICAL] cap_state.actual_equity is None")
            elif cap_state.actual_equity < 0:
                violations.append(
                    f"[COHERENCE:WARN] cap_state.actual_equity={cap_state.actual_equity} < 0"
                )
        return violations

    def validate_exposure_state(self, exposure_state: Any) -> List[str]:
        violations: List[str] = []
        violations.extend(self.check_none("exposure_state", exposure_state, critical=False))
        if exposure_state is None:
            return violations
        violations.extend(
            self.check_required_attrs(
                "exposure_state", exposure_state,
                _EXPOSURE_STATE_REQUIRED_ATTRS,
                critical=False,
            )
        )
        return violations

    def validate_allocator_integration(
        self,
        allocator:      Any,
        cap_state:      Any,
        exposure_state: Any,
    ) -> List[str]:
        """
        Validate that allocator, cap_state, and exposure_state are mutually
        compatible for compute_exposure_report() integration.
        """
        violations: List[str] = []

        violations.extend(self.validate_cap_state(cap_state))
        violations.extend(self.validate_exposure_state(exposure_state))

        if allocator is None:
            return violations

        if exposure_state is not None and allocator is not exposure_state:
            exp_buffers = getattr(exposure_state, "return_buffers", None)
            alloc_buffers = getattr(allocator, "return_buffers", None)
            if exp_buffers is not None and alloc_buffers is not None:
                if type(exp_buffers) != type(alloc_buffers):
                    violations.append(
                        "[COHERENCE:WARN] allocator.return_buffers type mismatch "
                        f"with exposure_state.return_buffers: "
                        f"{type(alloc_buffers).__name__} vs {type(exp_buffers).__name__}"
                    )

        missing = [a for a in _EXPOSURE_REPORT_REQUIRED_ATTRS if not hasattr(allocator, a)]
        if missing:
            violations.append(
                f"[COHERENCE:CRITICAL] allocator missing compute_exposure_report "
                f"interface attributes: {missing}"
            )

        return violations

    def validate_registry(
        self,
        registry:   RuntimeStateRegistry,
        now_sec:    Optional[float] = None,
    ) -> List[str]:
        return registry.validate_all(now_sec=now_sec)

    def assert_coherent(self, violations: List[str]) -> None:
        """Raise RuntimeContractError if any CRITICAL violation is present."""
        critical = [v for v in violations if ":CRITICAL]" in v]
        if critical:
            raise RuntimeContractError(
                f"Runtime coherence violations ({len(critical)} critical): {critical[0]}",
                violation_type="coherence_violation",
            )


# ─────────────────────────────────────────────────────────────────────────────
# BrokerRuntimeConsistencyValidator
# ─────────────────────────────────────────────────────────────────────────────

CONSISTENCY_QTY_DRIFT          = "qty_drift"
CONSISTENCY_ORPHAN_RUNTIME     = "orphan_runtime_position"
CONSISTENCY_ORPHAN_BROKER      = "orphan_broker_position"
CONSISTENCY_INTENT_QTY_MISMATCH = "intent_qty_mismatch"

CSEV_BLOCKING = "blocking"
CSEV_WARNING  = "warning"

_BLOCKING_CONSISTENCY_TYPES: frozenset = frozenset({
    CONSISTENCY_QTY_DRIFT,
    CONSISTENCY_ORPHAN_BROKER,
})


@dataclass
class ConsistencyViolation:
    violation_type: str
    symbol:         str
    broker_qty:     Optional[int]
    runtime_qty:    Optional[int]
    detail:         str
    severity:       str
    attribution:    Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerRuntimeConsistencyReport:
    timestamp:          str
    run_id:             str
    violations:         List[Dict[str, Any]]   # serialized ConsistencyViolation
    broker_positions:   Dict[str, int]
    runtime_positions:  Dict[str, int]
    consistent:         bool
    blocking_count:     int
    schema_version:     int = SCHEMA_VERSION


class BrokerRuntimeConsistencyValidator:
    """
    Validates broker vs runtime position consistency beyond qty mismatch:

      - qty_drift:            runtime_qty != broker_qty (broker authoritative)
      - orphan_runtime:       symbol in runtime with no broker record (qty=0 both sides)
      - orphan_broker:        broker has position with no runtime record (blocking)
      - intent_qty_mismatch:  open intent qty doesn't reconcile with known positions

    Broker truth is authoritative. Blocking violations prevent live order placement.
    """

    def validate(
        self,
        broker_positions:  Dict[str, int],
        runtime_positions: Dict[str, int],
        intent_map:        Optional[Dict[str, int]] = None,
    ) -> List[ConsistencyViolation]:
        """
        Returns list of ConsistencyViolation.
        Empty = fully consistent.
        """
        violations: List[ConsistencyViolation] = []
        all_symbols: Set[str] = set(broker_positions) | set(runtime_positions)

        for sym in sorted(all_symbols):
            bq = broker_positions.get(sym, 0)
            rq = runtime_positions.get(sym, 0)

            if bq == rq:
                continue

            if bq == 0 and rq > 0:
                violations.append(ConsistencyViolation(
                    violation_type=CONSISTENCY_ORPHAN_RUNTIME,
                    symbol=sym,
                    broker_qty=0,
                    runtime_qty=rq,
                    detail=(
                        f"runtime has qty={rq} for '{sym}' "
                        "but broker shows 0 — stale runtime state"
                    ),
                    severity=CSEV_WARNING,
                    attribution={"source": "runtime_only", "broker_authoritative": True},
                ))

            elif bq > 0 and rq == 0:
                violations.append(ConsistencyViolation(
                    violation_type=CONSISTENCY_ORPHAN_BROKER,
                    symbol=sym,
                    broker_qty=bq,
                    runtime_qty=0,
                    detail=(
                        f"broker has qty={bq} for '{sym}' "
                        "with no runtime record — unknown position origin"
                    ),
                    severity=CSEV_BLOCKING,
                    attribution={"source": "broker_only", "broker_authoritative": True},
                ))

            else:
                violations.append(ConsistencyViolation(
                    violation_type=CONSISTENCY_QTY_DRIFT,
                    symbol=sym,
                    broker_qty=bq,
                    runtime_qty=rq,
                    detail=(
                        f"runtime_qty={rq} != broker_qty={bq} "
                        f"for '{sym}' (broker authoritative)"
                    ),
                    severity=CSEV_BLOCKING,
                    attribution={
                        "broker_qty":  bq,
                        "runtime_qty": rq,
                        "drift":       bq - rq,
                        "broker_authoritative": True,
                    },
                ))

        if intent_map:
            for sym, intent_qty in sorted(intent_map.items()):
                bq = broker_positions.get(sym, 0)
                if bq != intent_qty and intent_qty > 0:
                    violations.append(ConsistencyViolation(
                        violation_type=CONSISTENCY_INTENT_QTY_MISMATCH,
                        symbol=sym,
                        broker_qty=bq,
                        runtime_qty=intent_qty,
                        detail=(
                            f"open intent qty={intent_qty} for '{sym}' "
                            f"does not reconcile with broker_qty={bq}"
                        ),
                        severity=CSEV_WARNING,
                        attribution={"intent_qty": intent_qty, "broker_qty": bq},
                    ))

        return violations

    def build_report(
        self,
        violations:         List[ConsistencyViolation],
        broker_positions:   Dict[str, int],
        runtime_positions:  Dict[str, int],
        run_id:             str = "",
        timestamp:          Optional[str] = None,
    ) -> BrokerRuntimeConsistencyReport:
        blocking = sum(1 for v in violations if v.severity == CSEV_BLOCKING)
        return BrokerRuntimeConsistencyReport(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            violations=[asdict(v) for v in violations],
            broker_positions=dict(broker_positions),
            runtime_positions=dict(runtime_positions),
            consistent=(len(violations) == 0),
            blocking_count=blocking,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Artifact persistence (append-only JSONL, replay-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a JSON record to a JSONL file. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:
        logger.warning("[INTEGRATION] artifact write failed %s: %s", path, exc)


def write_dependency_manifest(
    graph:  IntegrationDependencyGraph,
    path:   Path,
    run_id: str,
) -> None:
    record = {
        "artifact_type": "dependency_manifest",
        "run_id":        run_id,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        **graph.dump_manifest(),
    }
    _safe_append_jsonl(path, record)


def write_initialization_lineage(
    registry: RuntimeStateRegistry,
    path:     Path,
    run_id:   str,
) -> None:
    record = {
        "artifact_type": "initialization_lineage",
        "run_id":        run_id,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        **registry.dump_manifest(),
    }
    _safe_append_jsonl(path, record)


def write_state_validation_report(
    violations: List[str],
    path:       Path,
    run_id:     str,
    stage:      str,
) -> None:
    record = {
        "artifact_type":    "state_validation_report",
        "run_id":           run_id,
        "stage":            stage,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "violation_count":  len(violations),
        "violations":       violations,
        "schema_version":   SCHEMA_VERSION,
    }
    _safe_append_jsonl(path, record)


def write_coherence_diagnostic(
    violations:  List[str],
    stage_state: Dict[str, Any],
    path:        Path,
    run_id:      str,
) -> None:
    record = {
        "artifact_type":    "coherence_diagnostic",
        "run_id":           run_id,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "violation_count":  len(violations),
        "violations":       violations,
        "stage_state":      stage_state,
        "schema_version":   SCHEMA_VERSION,
    }
    _safe_append_jsonl(path, record)


def write_broker_consistency_report(
    report: BrokerRuntimeConsistencyReport,
    path:   Path,
) -> None:
    record = asdict(report)
    record["artifact_type"] = "broker_runtime_consistency_report"
    _safe_append_jsonl(path, record)


def write_failure_attribution(
    error:  RuntimeContractError,
    path:   Path,
    run_id: str,
    stage:  str,
) -> None:
    record = {
        "artifact_type":  "integration_failure_attribution",
        "run_id":         run_id,
        "stage":          stage,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "violation_type": error.violation_type,
        "message":        str(error),
        "schema_version": SCHEMA_VERSION,
    }
    _safe_append_jsonl(path, record)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level integration validation entry point
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntegrationValidationResult:
    run_id:                  str
    timestamp:               str
    stage:                   str
    registry_violations:     List[str]
    dep_graph_violations:    List[str]
    precondition_violations: List[str]
    coherence_violations:    List[str]
    consistency_violations:  List[Dict[str, Any]]
    all_violations:          List[str]
    execution_allowed:       bool
    blocking_count:          int
    schema_version:          int = SCHEMA_VERSION


def validate_runtime_integration(
    run_id:            str,
    stage:             str,
    registry:          Optional[RuntimeStateRegistry]           = None,
    dep_graph:         Optional[IntegrationDependencyGraph]     = None,
    precondition_val:  Optional[StagePreconditionValidator]     = None,
    coherence_val:     Optional[RuntimeCoherenceValidator]      = None,
    broker_positions:  Optional[Dict[str, int]]                 = None,
    runtime_positions: Optional[Dict[str, int]]                 = None,
    intent_map:        Optional[Dict[str, int]]                 = None,
    cap_state:         Any                                      = None,
    exposure_state:    Any                                      = None,
    allocator:         Any                                      = None,
    artifact_dir:      Optional[Path]                           = None,
    is_live:           bool                                     = False,
    now_sec:           Optional[float]                          = None,
) -> IntegrationValidationResult:
    """
    Full runtime integration validation pass.

    Fail policy:
      is_live=True:   raises RuntimeContractError on any CRITICAL violation
      is_live=False:  logs all violations, returns result (observable)

    Artifacts written to artifact_dir (runtime/integration/) if provided.
    """
    if now_sec is None:
        now_sec = time.time()
    ts = datetime.now(timezone.utc).isoformat()

    reg_violations:   List[str] = []
    dep_violations:   List[str] = []
    pre_violations:   List[str] = []
    coh_violations:   List[str] = []
    con_violations:   List[Dict[str, Any]] = []
    first_error: Optional[RuntimeContractError] = None

    # 1. Registry state validation
    if registry is not None:
        try:
            reg_violations = registry.validate_all(now_sec=now_sec)
            if artifact_dir:
                write_initialization_lineage(registry, artifact_dir / "initialization_lineage.jsonl", run_id)
        except Exception as exc:
            logger.warning("[INTEGRATION] registry validation error: %s", exc)

    # 2. Dependency graph validation
    if dep_graph is not None:
        try:
            dep_violations = dep_graph.validate()
            if artifact_dir:
                write_dependency_manifest(dep_graph, artifact_dir / "dependency_manifests.jsonl", run_id)
        except Exception as exc:
            logger.warning("[INTEGRATION] dep_graph validation error: %s", exc)

    # 3. Stage precondition validation
    if precondition_val is not None:
        try:
            pre_violations = [
                f"[PRECONDITION] stage='{stage}' missing: '{p}'"
                for p in precondition_val.validate_preconditions(stage)
            ]
        except Exception as exc:
            logger.warning("[INTEGRATION] precondition validation error: %s", exc)

    # 4. Runtime coherence validation
    _coh = coherence_val if coherence_val is not None else RuntimeCoherenceValidator()
    try:
        if cap_state is not None or exposure_state is not None or allocator is not None:
            coh_violations.extend(
                _coh.validate_allocator_integration(allocator, cap_state, exposure_state)
            )
        if registry is not None:
            coh_violations.extend(_coh.validate_registry(registry, now_sec=now_sec))
    except Exception as exc:
        logger.warning("[INTEGRATION] coherence validation error: %s", exc)

    # 5. Broker/runtime consistency validation
    _br_val = BrokerRuntimeConsistencyValidator()
    _br_report: Optional[BrokerRuntimeConsistencyReport] = None
    if broker_positions is not None and runtime_positions is not None:
        try:
            _con_list = _br_val.validate(
                broker_positions=broker_positions,
                runtime_positions=runtime_positions,
                intent_map=intent_map,
            )
            con_violations = [asdict(v) for v in _con_list]
            _br_report = _br_val.build_report(
                violations=_con_list,
                broker_positions=broker_positions,
                runtime_positions=runtime_positions,
                run_id=run_id,
                timestamp=ts,
            )
            if artifact_dir:
                write_broker_consistency_report(
                    _br_report, artifact_dir / "broker_runtime_consistency_reports.jsonl"
                )
        except Exception as exc:
            logger.warning("[INTEGRATION] broker consistency validation error: %s", exc)

    # Aggregate all violations
    all_viol: List[str] = []
    all_viol.extend(reg_violations)
    all_viol.extend(dep_violations)
    all_viol.extend(pre_violations)
    all_viol.extend(coh_violations)
    all_viol.extend(v.get("detail", str(v)) for v in con_violations)

    # Count blocking
    critical_viol = [v for v in (reg_violations + dep_violations + pre_violations + coh_violations)
                     if ":CRITICAL]" in v or "[PRECONDITION]" in v or "[DEPGRAPH]" in v]
    broker_blocking = sum(1 for v in con_violations if v.get("severity") == CSEV_BLOCKING)
    blocking_count = len(critical_viol) + broker_blocking
    execution_allowed = blocking_count == 0

    # Write validation report artifact
    if artifact_dir:
        try:
            write_state_validation_report(
                all_viol, artifact_dir / "state_validation_reports.jsonl", run_id, stage
            )
            write_coherence_diagnostic(
                coh_violations,
                {
                    "cap_state_loaded":      cap_state is not None,
                    "exposure_state_loaded": exposure_state is not None,
                    "allocator_loaded":      allocator is not None,
                    "stage":                 stage,
                },
                artifact_dir / "runtime_coherence_diagnostics.jsonl",
                run_id,
            )
        except Exception as exc:
            logger.warning("[INTEGRATION] report write error: %s", exc)

    # Log summary
    if all_viol:
        log_fn = logger.error if (blocking_count > 0 and is_live) else logger.warning
        log_fn(
            "[INTEGRATION] %d violations (%d blocking) stage=%s run_id=%s",
            len(all_viol), blocking_count, stage, run_id,
        )
        for v in all_viol:
            logger.warning("[INTEGRATION] %s", v)
    else:
        logger.info("[INTEGRATION] ok: 0 violations stage=%s run_id=%s", stage, run_id)

    result = IntegrationValidationResult(
        run_id=run_id,
        timestamp=ts,
        stage=stage,
        registry_violations=reg_violations,
        dep_graph_violations=dep_violations,
        precondition_violations=pre_violations,
        coherence_violations=coh_violations,
        consistency_violations=con_violations,
        all_violations=all_viol,
        execution_allowed=execution_allowed,
        blocking_count=blocking_count,
    )

    # Fail-closed enforcement for live mode
    if is_live and not execution_allowed and first_error is None:
        # Build a RuntimeContractError from the first blocking violation
        if critical_viol:
            first_error = RuntimeContractError(
                critical_viol[0], violation_type="runtime_contract_violation"
            )
        elif broker_blocking > 0 and con_violations:
            v = con_violations[0]
            first_error = RuntimeContractError(
                v.get("detail", "broker/runtime consistency violation"),
                violation_type=v.get("violation_type", "consistency_violation"),
            )

    if first_error is not None and is_live:
        if artifact_dir:
            try:
                write_failure_attribution(
                    first_error,
                    artifact_dir / "integration_failure_attribution.jsonl",
                    run_id,
                    stage,
                )
            except Exception:
                pass
        raise first_error

    return result
