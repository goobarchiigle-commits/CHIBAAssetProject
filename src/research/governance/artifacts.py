"""GovernanceArtifacts — deterministic immutable storage for governance outputs.

Subdirectories under artifacts/governance/:
    ledgers/        — exploration ledger manifests + JSONL entries
    negative/       — negative knowledge manifests
    topology/       — topology failure reports
    ecology/        — ecology summaries
    persistence/    — family persistence reports
    budget/         — exploration budget reports
    memory/         — research memory manifests
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.research.analysis.artifacts import (
    ImmutabilityError,
    append_jsonl,
    compute_checksum,
    read_artifact,
    read_jsonl,
    write_immutable,
    utc_now_iso,
)

GOVERNANCE_ROOT = Path("artifacts/governance")

_SUBDIRS = ("ledgers", "negative", "topology", "ecology", "persistence", "budget", "memory")


class GovernanceArtifacts:
    """Thin routing layer for governance artifact storage.

    All writes are immutable (raises ImmutabilityError on collision).
    All JSONL appends are append-only.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self.root = Path(base_path) if base_path is not None else GOVERNANCE_ROOT

    def _ensure_dirs(self) -> None:
        for sub in _SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ── Ledger ─────────────────────────────────────────────────────────────────

    def write_ledger_manifest(self, ledger_id: str, manifest: dict) -> Path:
        self._ensure_dirs()
        path = self.root / "ledgers" / f"{ledger_id}.json"
        write_immutable(path, manifest)
        return path

    def append_ledger_entry(self, ledger_id: str, entry: dict) -> None:
        self._ensure_dirs()
        append_jsonl(self.root / "ledgers" / f"{ledger_id}.jsonl", entry)

    def read_ledger_entries(self, ledger_id: str) -> list:
        return read_jsonl(self.root / "ledgers" / f"{ledger_id}.jsonl")

    # ── Negative knowledge ─────────────────────────────────────────────────────

    def write_negative_manifest(self, manifest_id: str, manifest: dict) -> Path:
        self._ensure_dirs()
        path = self.root / "negative" / f"{manifest_id}.json"
        write_immutable(path, manifest)
        return path

    # ── Topology ───────────────────────────────────────────────────────────────

    def write_topology_report(self, report_id: str, report: dict) -> Path:
        self._ensure_dirs()
        path = self.root / "topology" / f"{report_id}.json"
        write_immutable(path, report)
        return path

    # ── Ecology ────────────────────────────────────────────────────────────────

    def write_ecology_summary(self, summary_id: str, summary: dict) -> Path:
        self._ensure_dirs()
        path = self.root / "ecology" / f"{summary_id}.json"
        write_immutable(path, summary)
        return path

    # ── Persistence ────────────────────────────────────────────────────────────

    def write_persistence_report(self, report_id: str, report: dict) -> Path:
        self._ensure_dirs()
        path = self.root / "persistence" / f"{report_id}.json"
        write_immutable(path, report)
        return path

    # ── Budget ─────────────────────────────────────────────────────────────────

    def write_budget_report(self, report_id: str, report: dict) -> Path:
        self._ensure_dirs()
        path = self.root / "budget" / f"{report_id}.json"
        write_immutable(path, report)
        return path

    # ── Memory ─────────────────────────────────────────────────────────────────

    def write_memory_manifest(self, memory_id: str, manifest: dict) -> Path:
        self._ensure_dirs()
        path = self.root / "memory" / f"{memory_id}.json"
        write_immutable(path, manifest)
        return path

    def append_memory_entry(self, memory_id: str, entry: dict) -> None:
        self._ensure_dirs()
        append_jsonl(self.root / "memory" / f"{memory_id}.jsonl", entry)

    def read_memory_entries(self, memory_id: str) -> list:
        return read_jsonl(self.root / "memory" / f"{memory_id}.jsonl")
