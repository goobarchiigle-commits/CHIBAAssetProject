"""CounterfactualReplay — deterministic scenario replay for execution analysis."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SCENARIO_REDUCED_PARTICIPATION = "REDUCED_PARTICIPATION"
SCENARIO_DELAYED_ENTRY = "DELAYED_ENTRY"
SCENARIO_AUCTION_AVOIDANCE = "AUCTION_AVOIDANCE"
SCENARIO_ALTERNATE_THROTTLING = "ALTERNATE_THROTTLING"
SCENARIO_ALTERNATE_LIQUIDITY_CAP = "ALTERNATE_LIQUIDITY_CAP"

ALL_SCENARIOS = (
    SCENARIO_REDUCED_PARTICIPATION,
    SCENARIO_DELAYED_ENTRY,
    SCENARIO_AUCTION_AVOIDANCE,
    SCENARIO_ALTERNATE_THROTTLING,
    SCENARIO_ALTERNATE_LIQUIDITY_CAP,
)


@dataclass(frozen=True)
class ReplayScenario:
    scenario_id: str        # "scn_" + SHA256[:12]
    scenario_type: str
    parameters: Tuple[Tuple[str, str], ...]     # sorted (key, str_value) pairs
    description: str

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "parameters": list(self.parameters),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReplayScenario":
        return cls(
            scenario_id=d["scenario_id"],
            scenario_type=d["scenario_type"],
            parameters=tuple(tuple(x) for x in d["parameters"]),
            description=d["description"],
        )


@dataclass(frozen=True)
class ReplayResult:
    result_id: str          # "rply_" + SHA256[:12]
    session_id: str
    scenario: ReplayScenario
    baseline_slippage_bps: float
    counterfactual_slippage_bps: float
    baseline_fill_ratio: float
    counterfactual_fill_ratio: float
    baseline_latency_ns: int
    counterfactual_latency_ns: int
    slippage_improvement_bps: float     # baseline - counterfactual
    fill_ratio_improvement: float       # counterfactual - baseline
    replayed_at: str
    schema_version: str
    checksum: str

    def is_improvement(self) -> bool:
        return self.slippage_improvement_bps > 0.0 or self.fill_ratio_improvement > 0.0

    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "session_id": self.session_id,
            "scenario": self.scenario.to_dict(),
            "baseline_slippage_bps": self.baseline_slippage_bps,
            "counterfactual_slippage_bps": self.counterfactual_slippage_bps,
            "baseline_fill_ratio": self.baseline_fill_ratio,
            "counterfactual_fill_ratio": self.counterfactual_fill_ratio,
            "baseline_latency_ns": self.baseline_latency_ns,
            "counterfactual_latency_ns": self.counterfactual_latency_ns,
            "slippage_improvement_bps": self.slippage_improvement_bps,
            "fill_ratio_improvement": self.fill_ratio_improvement,
            "replayed_at": self.replayed_at,
            "schema_version": self.schema_version,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReplayResult":
        return cls(
            result_id=d["result_id"],
            session_id=d["session_id"],
            scenario=ReplayScenario.from_dict(d["scenario"]),
            baseline_slippage_bps=d["baseline_slippage_bps"],
            counterfactual_slippage_bps=d["counterfactual_slippage_bps"],
            baseline_fill_ratio=d["baseline_fill_ratio"],
            counterfactual_fill_ratio=d["counterfactual_fill_ratio"],
            baseline_latency_ns=d["baseline_latency_ns"],
            counterfactual_latency_ns=d["counterfactual_latency_ns"],
            slippage_improvement_bps=d["slippage_improvement_bps"],
            fill_ratio_improvement=d["fill_ratio_improvement"],
            replayed_at=d["replayed_at"],
            schema_version=d["schema_version"],
            checksum=d["checksum"],
        )


def _scenario_id(scenario_type: str, parameters: tuple) -> str:
    raw = json.dumps({"scenario_type": scenario_type, "parameters": list(parameters)},
                     sort_keys=True, separators=(",", ":"))
    return "scn_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _result_id(session_id: str, scenario_id: str, replayed_at: str) -> str:
    raw = json.dumps({"session_id": session_id, "scenario_id": scenario_id, "replayed_at": replayed_at},
                     sort_keys=True, separators=(",", ":"))
    return "rply_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _checksum(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_scenario(
    scenario_type: str,
    parameters: Dict[str, str],
    description: str,
) -> ReplayScenario:
    if scenario_type not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario type: {scenario_type!r}")
    params_tuple = tuple(sorted((k, str(v)) for k, v in parameters.items()))
    scenario_id = _scenario_id(scenario_type, params_tuple)
    return ReplayScenario(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        parameters=params_tuple,
        description=description,
    )


class CounterfactualReplay:
    """Deterministic replay engine. Applies scenario modifiers to baseline observations."""

    SCENARIO_MODIFIERS: Dict[str, Dict[str, float]] = {
        SCENARIO_REDUCED_PARTICIPATION: {"slippage_mult": 0.70, "fill_ratio_mult": 0.90, "latency_mult": 1.10},
        SCENARIO_DELAYED_ENTRY: {"slippage_mult": 1.20, "fill_ratio_mult": 0.95, "latency_mult": 1.50},
        SCENARIO_AUCTION_AVOIDANCE: {"slippage_mult": 0.85, "fill_ratio_mult": 0.85, "latency_mult": 0.90},
        SCENARIO_ALTERNATE_THROTTLING: {"slippage_mult": 0.90, "fill_ratio_mult": 1.00, "latency_mult": 1.20},
        SCENARIO_ALTERNATE_LIQUIDITY_CAP: {"slippage_mult": 0.75, "fill_ratio_mult": 0.80, "latency_mult": 1.05},
    }

    def __init__(self, schema_version: str = "1.0.0"):
        self._schema_version = schema_version
        self._results: List[ReplayResult] = []

    def replay(
        self,
        session_id: str,
        scenario: ReplayScenario,
        baseline_slippage_bps: float,
        baseline_fill_ratio: float,
        baseline_latency_ns: int,
        replayed_at: str,
    ) -> ReplayResult:
        mods = self.SCENARIO_MODIFIERS[scenario.scenario_type]
        cf_slippage = baseline_slippage_bps * mods["slippage_mult"]
        cf_fill = min(1.0, baseline_fill_ratio * mods["fill_ratio_mult"])
        cf_latency = int(baseline_latency_ns * mods["latency_mult"])
        slip_imp = baseline_slippage_bps - cf_slippage
        fill_imp = cf_fill - baseline_fill_ratio
        result_id = _result_id(session_id, scenario.scenario_id, replayed_at)
        payload = {
            "session_id": session_id,
            "scenario_id": scenario.scenario_id,
            "cf_slippage": cf_slippage,
            "cf_fill": cf_fill,
        }
        checksum = _checksum(payload)
        result = ReplayResult(
            result_id=result_id,
            session_id=session_id,
            scenario=scenario,
            baseline_slippage_bps=baseline_slippage_bps,
            counterfactual_slippage_bps=cf_slippage,
            baseline_fill_ratio=baseline_fill_ratio,
            counterfactual_fill_ratio=cf_fill,
            baseline_latency_ns=baseline_latency_ns,
            counterfactual_latency_ns=cf_latency,
            slippage_improvement_bps=slip_imp,
            fill_ratio_improvement=fill_imp,
            replayed_at=replayed_at,
            schema_version=self._schema_version,
            checksum=checksum,
        )
        self._results.append(result)
        return result

    def all_results(self) -> List[ReplayResult]:
        return sorted(self._results, key=lambda r: r.replayed_at)

    def result_count(self) -> int:
        return len(self._results)

    def best_scenario(self) -> Optional[ReplayResult]:
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.slippage_improvement_bps + r.fill_ratio_improvement * 10)

    def to_summary(self) -> dict:
        return {
            "result_count": len(self._results),
            "schema_version": self._schema_version,
            "best_scenario": self.best_scenario().scenario.scenario_type if self._results else None,
        }
