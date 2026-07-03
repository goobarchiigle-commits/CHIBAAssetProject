"""
bootstrap — Runtime initialization integrity and dependency validation layer

Modules:
  initialization_graph      — 9-phase graph, PhaseStatus, failure propagation
  dependency_validator      — ordering + completeness enforcement (fail-closed)
  contract_verifier         — per-phase state contract verification
  root_cause_compressor     — failure chain → compressed human-readable summary
  state_coherence_validator — cross-phase coherence checks
  bootstrap_report          — atomic JSON report generation
"""
from src.bootstrap.initialization_graph import (
    ALL_PHASES,
    CRITICAL_PHASES,
    DEFAULT_PHASE_DEPENDENCIES,
    NON_CRITICAL_PHASES,
    PHASE_ALLOCATION,
    PHASE_ANALYTICS_REGISTRY,
    PHASE_BROKER_INTEGRATION,
    PHASE_CAPITAL_STATE,
    PHASE_CONFIGURATION,
    PHASE_GOVERNANCE,
    PHASE_INFLIGHT_REGISTRY,
    PHASE_RECONCILIATION,
    PHASE_RUNTIME_SUPERVISOR,
    PhaseRecord,
    PhaseStatus,
    RuntimeInitializationGraph,
)
from src.bootstrap.dependency_validator import (
    BootstrapDependencyValidator,
    BootstrapIntegrityError,
)
from src.bootstrap.contract_verifier import InitializationContractVerifier
from src.bootstrap.root_cause_compressor import RootCauseCompressor
from src.bootstrap.state_coherence_validator import StateCoherenceValidator
from src.bootstrap.bootstrap_report import (
    BootstrapReport,
    build_bootstrap_report,
    run_bootstrap_validation,
    write_bootstrap_report,
)

__all__ = [
    "ALL_PHASES",
    "CRITICAL_PHASES",
    "DEFAULT_PHASE_DEPENDENCIES",
    "NON_CRITICAL_PHASES",
    "PHASE_ALLOCATION",
    "PHASE_ANALYTICS_REGISTRY",
    "PHASE_BROKER_INTEGRATION",
    "PHASE_CAPITAL_STATE",
    "PHASE_CONFIGURATION",
    "PHASE_GOVERNANCE",
    "PHASE_INFLIGHT_REGISTRY",
    "PHASE_RECONCILIATION",
    "PHASE_RUNTIME_SUPERVISOR",
    "PhaseRecord",
    "PhaseStatus",
    "RuntimeInitializationGraph",
    "BootstrapDependencyValidator",
    "BootstrapIntegrityError",
    "InitializationContractVerifier",
    "RootCauseCompressor",
    "StateCoherenceValidator",
    "BootstrapReport",
    "build_bootstrap_report",
    "run_bootstrap_validation",
    "write_bootstrap_report",
]
