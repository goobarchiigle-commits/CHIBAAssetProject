"""ExperimentGenerator — creates ExperimentContracts from ParameterSweeps.

Deterministic: same sweep + dataset_hash + code_version → same experiment IDs.
Duplicate experiments (same ID already in registry) are rejected, not re-registered.
Each generated experiment carries: experiment_id, parameter_hash, seed, batch_id linkage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.research.contracts.experiment import ExperimentContract
from src.research.batch.sweep import ParameterSweep, param_hash
from src.research.batch.batch import ExperimentBatch
from src.research.orchestrator.registry import ExperimentRegistry


@dataclass(frozen=True)
class GenerationResult:
    batch: ExperimentBatch
    new_experiments: Tuple[ExperimentContract, ...]
    duplicates: Tuple[str, ...]         # experiment_ids already in registry
    parameter_map: Dict[str, dict]      # experiment_id → parameter dict


class ExperimentGenerator:
    """Generates ExperimentContracts deterministically from a ParameterSweep."""

    def generate(
        self,
        sweep: ParameterSweep,
        dataset_hash: str,
        code_version: str,
        registry: ExperimentRegistry,
        description: str = "",
        timestamp: Optional[str] = None,
    ) -> GenerationResult:
        """Generate all sweep combinations, register non-duplicates.

        Returns GenerationResult with immutable batch + experiment list.
        Duplicate detection is by experiment_id (deterministic content hash).
        Order: same sweep inputs → same order, always.
        """
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d")
        combos = sweep.combinations()          # deterministic order
        space_hash = sweep.parameter_space_hash()

        batch_id = ExperimentBatch.make_id(
            sweep.strategy_id, space_hash, sweep.seed, timestamp=ts
        )

        new_experiments: List[ExperimentContract] = []
        duplicates: List[str] = []
        parameter_map: Dict[str, dict] = {}

        for params in combos:
            ph = param_hash(params)
            exp_id = ExperimentContract.make_id(
                sweep.strategy_id, ph, dataset_hash, sweep.seed, timestamp=ts
            )

            if registry.exists(exp_id):
                duplicates.append(exp_id)
                continue

            exp = ExperimentContract(
                experiment_id=exp_id,
                strategy_id=sweep.strategy_id,
                parameter_hash=ph,
                dataset_hash=dataset_hash,
                code_version=code_version,
                seed=sweep.seed,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            registry.register(exp)
            new_experiments.append(exp)
            parameter_map[exp_id] = params

        all_ids = tuple(e.experiment_id for e in new_experiments)

        batch = ExperimentBatch(
            batch_id=batch_id,
            strategy_id=sweep.strategy_id,
            description=description,
            seed=sweep.seed,
            created_at=datetime.now(timezone.utc).isoformat(),
            experiment_ids=all_ids,
            parameter_space_hash=space_hash,
        )

        return GenerationResult(
            batch=batch,
            new_experiments=tuple(new_experiments),
            duplicates=tuple(duplicates),
            parameter_map=parameter_map,
        )
