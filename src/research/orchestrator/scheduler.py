"""Deterministic experiment scheduler.

Ordering guarantee: given the same list of experiments and seed,
schedule_experiments always returns the same ordering.
No random, no time-based, no external state.
"""
from __future__ import annotations

from typing import List

from src.research.contracts.experiment import ExperimentContract


def schedule_experiments(
    experiments: List[ExperimentContract],
    seed: int = 0,
) -> List[ExperimentContract]:
    """Sort experiments into a deterministic execution order.

    Primary key: (strategy_id, parameter_hash, seed, experiment_id).
    seed parameter is reserved for future stratified ordering; currently unused
    to keep determinism pure.
    """
    return sorted(
        experiments,
        key=lambda e: (e.strategy_id, e.parameter_hash, e.seed, e.experiment_id),
    )


def batch(
    experiments: List[ExperimentContract],
    max_concurrency: int,
) -> List[List[ExperimentContract]]:
    """Partition a scheduled list into bounded-concurrency batches.

    Batches are deterministic: same input → same batches.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    ordered = schedule_experiments(experiments)
    return [ordered[i:i + max_concurrency] for i in range(0, len(ordered), max_concurrency)]
