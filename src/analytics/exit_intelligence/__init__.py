"""
src/analytics/exit_intelligence — Exit Intelligence OS

Unified deterioration-aware capital lifecycle management.
Observation → Deterioration → Exit Pressure → Capital Lifecycle → Research Dataset.

All analytics are append-only, immutable, and replay-deterministic.
No execution mutation. No ML. No adaptive parameter tuning.
"""
from .models import (
    ExitObservationRecord,
    ExitPathSnapshot,
    ExecutionExitObservation,
    PortfolioExitPressureSnapshot,
    ExitVelocityScore,
    MomentumDecayMetrics,
    StructuralFailureMetrics,
    PeakDecayVelocityMetrics,
    ConvexityRetentionMetrics,
    PersistenceMetrics,
    PartialExitDecision,
    CounterfactualExitOutcome,
    CounterfactualPartialExitRecord,
    LoserRetentionRecord,
    ConditionalExpectancyPoint,
    ExitResearchRecord,
    SCHEMA_VERSION,
)
from .observation import (
    ExitObservationStore,
    ExitPathSnapshotStore,
    ExecutionExitObservationStore,
    PortfolioExitPressureStore,
    build_exit_path_snapshot,
)
from .deterioration import (
    MomentumDecayAnalyzer,
    StructuralFailureDetector,
    PeakDecayVelocityAnalyzer,
    ConvexityRetentionAnalyzer,
    PersistenceExtensionLayer,
)
from .exit_velocity import ExitVelocityScorer, ExitVelocityStore
from .lifecycle import (
    PartialExitAllocator,
    CounterfactualExitAnalyzer,
    CounterfactualPartialExitAnalytics,
    LoserRetentionAnalysis,
    ConditionalExpectancySurface,
    LifecycleStore,
)
from .dataset import ExitResearchDatasetBuilder, ExitResearchDatasetStore
from .validator import IntegrityValidator, ValidationReport, ValidationViolation
from .hook import run_exit_intelligence_hook

__all__ = [
    "ExitObservationRecord",
    "ExitPathSnapshot",
    "ExecutionExitObservation",
    "PortfolioExitPressureSnapshot",
    "ExitVelocityScore",
    "MomentumDecayMetrics",
    "StructuralFailureMetrics",
    "PeakDecayVelocityMetrics",
    "ConvexityRetentionMetrics",
    "PersistenceMetrics",
    "PartialExitDecision",
    "CounterfactualExitOutcome",
    "CounterfactualPartialExitRecord",
    "LoserRetentionRecord",
    "ConditionalExpectancyPoint",
    "ExitResearchRecord",
    "ExitObservationStore",
    "ExitPathSnapshotStore",
    "ExecutionExitObservationStore",
    "PortfolioExitPressureStore",
    "build_exit_path_snapshot",
    "MomentumDecayAnalyzer",
    "StructuralFailureDetector",
    "PeakDecayVelocityAnalyzer",
    "ConvexityRetentionAnalyzer",
    "PersistenceExtensionLayer",
    "ExitVelocityScorer",
    "ExitVelocityStore",
    "PartialExitAllocator",
    "CounterfactualExitAnalyzer",
    "CounterfactualPartialExitAnalytics",
    "LoserRetentionAnalysis",
    "ConditionalExpectancySurface",
    "LifecycleStore",
    "ExitResearchDatasetBuilder",
    "ExitResearchDatasetStore",
    "IntegrityValidator",
    "ValidationReport",
    "ValidationViolation",
    "run_exit_intelligence_hook",
    "SCHEMA_VERSION",
]
