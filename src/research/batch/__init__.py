"""Batch experimentation layer — high-throughput deterministic alpha iteration."""
from src.research.batch.batch import ExperimentBatch, BatchArtifacts
from src.research.batch.sweep import ParameterSweep
from src.research.batch.generator import ExperimentGenerator
from src.research.batch.runner import BatchRunner, BatchExecutionLog
from src.research.batch.summary import BatchSummary
from src.research.batch.correlation import compute_pairwise_correlation, flag_near_duplicates, rolling_correlation_stability
from src.research.batch.capacity import compute_capacity_proxy, CapacityProxy

__all__ = [
    "ExperimentBatch",
    "BatchArtifacts",
    "ParameterSweep",
    "ExperimentGenerator",
    "BatchRunner",
    "BatchExecutionLog",
    "BatchSummary",
    "compute_pairwise_correlation",
    "flag_near_duplicates",
    "rolling_correlation_stability",
    "compute_capacity_proxy",
    "CapacityProxy",
]
