"""Shared fixtures and helpers for governance tests."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.governance.ledger import ExplorationLedger
from src.research.governance.negative import NegativeKnowledgeRegistry, FAILURE_DEAD_TOPOLOGY, FAILURE_REGIME_FRAGILE
from src.research.governance.topology import TopologyFailureRegistry, TOPOLOGY_CLIFF_EDGE, TOPOLOGY_NARROW_PLATEAU
from src.research.governance.persistence import FamilyPersistenceTracker
from src.research.governance.ecology import HypothesisEcologyTracker
from src.research.governance.memory import ResearchMemory

TS = "2026-01-01T00:00:00.000000Z"
TS2 = "2026-01-02T00:00:00.000000Z"
TS3 = "2026-01-03T00:00:00.000000Z"

FAM_A = "fam_aaaaaaaaaaaa"
FAM_B = "fam_bbbbbbbbbbbb"
FAM_C = "fam_cccccccccccc"
GH_A = "gh_aaaaaaaaaaaaaaaa"
GH_B = "gh_bbbbbbbbbbbbbbbb"
HYP_A = "hyp_aaaaaaaaaaaa"
HYP_B = "hyp_bbbbbbbbbbbb"
TMPL = "tmpl_aaaaaaaaaaaa"
PARAM_H = "paramhash0000001"
TAX = "trend"


def make_ledger(ledger_id: str = "led_test") -> ExplorationLedger:
    return ExplorationLedger(ledger_id)


def make_negative() -> NegativeKnowledgeRegistry:
    return NegativeKnowledgeRegistry()


def make_topology() -> TopologyFailureRegistry:
    return TopologyFailureRegistry()


def make_persistence() -> FamilyPersistenceTracker:
    return FamilyPersistenceTracker()


def make_ecology() -> HypothesisEcologyTracker:
    return HypothesisEcologyTracker()


def make_memory(memory_id: str = "mem_test") -> ResearchMemory:
    return ResearchMemory(memory_id)


def populated_ledger() -> ExplorationLedger:
    ledger = make_ledger()
    ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
    ledger.record_rejected(FAM_B, HYP_B, TMPL, PARAM_H, GH_B, TAX, timestamp=TS2)
    return ledger


def populated_negative() -> NegativeKnowledgeRegistry:
    neg = make_negative()
    neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
    neg.register_failure(FAM_B, GH_B, FAILURE_REGIME_FRAGILE, timestamp=TS)
    return neg


def populated_topology() -> TopologyFailureRegistry:
    topo = make_topology()
    topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
    topo.register_topology_failure(GH_B, FAM_B, TOPOLOGY_NARROW_PLATEAU, 0.5, timestamp=TS)
    return topo


def populated_persistence(n_survived: int = 3, n_failed: int = 1) -> FamilyPersistenceTracker:
    pt = make_persistence()
    for i in range(n_survived):
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
    for i in range(n_failed):
        pt.record_observation(FAM_A, survived=False, timestamp=TS2)
    return pt
