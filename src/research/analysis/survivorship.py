"""SurvivorshipLedger — immutable append-only survivorship audit artifacts.

Artifacts stored in artifacts/analysis/survivorship/ (relative to BASE_DIR).
Each artifact is an immutable JSON file with a deterministic checksum.
A manifest.jsonl indexes all entries (append-only).

Fail-closed: attempting to overwrite an existing artifact raises ImmutabilityError.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.research.analysis.artifacts import (
    ImmutabilityError,
    append_jsonl,
    compute_checksum,
    read_artifact,
    read_jsonl,
    utc_now_iso,
    verify_checksum,
    write_immutable,
)
from src.research.analysis.contract import (
    SCHEMA_VERSION,
    FamilyRecord,
    RegimeResilienceResult,
    SurvivorRecord,
    SurvivorshipLedgerEntry,
    TopologyResult,
)


class SurvivorshipLedger:
    """Append-only survivorship audit ledger.

    Args:
        artifact_dir: directory for storing artifacts (created on first write).
    """

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._manifest_path = self.artifact_dir / "manifest.jsonl"

    # ── Public interface ───────────────────────────────────────────────────────

    def record(
        self,
        *,
        records: List[SurvivorRecord],
        families: List[FamilyRecord],
        topology: Optional[Dict[str, TopologyResult]] = None,
        regime_resilience: Optional[RegimeResilienceResult] = None,
        extra_metadata: Optional[dict] = None,
    ) -> SurvivorshipLedgerEntry:
        """Build and persist an immutable ledger entry.

        Args:
            records:           SurvivorRecord list for this batch.
            families:          FamilyRecord list from ExperimentFamilyGraph.
            topology:          Optional topology results (exp_id → TopologyResult).
            regime_resilience: Optional regime resilience result.
            extra_metadata:    Optional dict of additional context (must be JSON-serializable).

        Returns:
            SurvivorshipLedgerEntry with validated checksum.

        Raises:
            ImmutabilityError: if an artifact with this ID already exists.
        """
        generated_at = utc_now_iso()

        payload = self._build_payload(
            records, families, topology, regime_resilience, extra_metadata
        )
        checksum = compute_checksum(payload)
        ledger_id = self._make_ledger_id(checksum, generated_at)

        entry = SurvivorshipLedgerEntry(
            ledger_id=ledger_id,
            generated_at=generated_at,
            schema_version=SCHEMA_VERSION,
            experiment_count=len(records),
            survivor_count=sum(1 for r in records if r.survived),
            family_count=len(families),
            checksum=checksum,
            payload=payload,
        )

        artifact_path = self.artifact_dir / f"{ledger_id}.json"
        write_immutable(artifact_path, entry.to_dict())

        manifest_record = {
            "ledger_id": ledger_id,
            "generated_at": generated_at,
            "schema_version": SCHEMA_VERSION,
            "artifact_file": f"{ledger_id}.json",
            "experiment_count": entry.experiment_count,
            "survivor_count": entry.survivor_count,
            "family_count": entry.family_count,
            "checksum": checksum,
        }
        append_jsonl(self._manifest_path, manifest_record)

        return entry

    def verify(self, ledger_id: str) -> bool:
        """Verify the checksum of a stored artifact.

        Returns True if checksum matches, False if artifact missing or corrupted.
        """
        artifact_path = self.artifact_dir / f"{ledger_id}.json"
        if not artifact_path.exists():
            return False
        try:
            d = read_artifact(artifact_path)
            return verify_checksum(d["payload"], d["checksum"])
        except (KeyError, json.JSONDecodeError, ValueError):
            return False

    def verify_all(self) -> Dict[str, bool]:
        """Verify all artifacts listed in the manifest.

        Returns Dict[ledger_id, is_valid].
        """
        manifest_entries = read_jsonl(self._manifest_path)
        return {
            entry["ledger_id"]: self.verify(entry["ledger_id"])
            for entry in manifest_entries
        }

    def load(self, ledger_id: str) -> SurvivorshipLedgerEntry:
        """Load and validate a stored ledger entry."""
        artifact_path = self.artifact_dir / f"{ledger_id}.json"
        d = read_artifact(artifact_path)
        if not verify_checksum(d["payload"], d["checksum"]):
            raise ValueError(f"Checksum mismatch for ledger_id={ledger_id}")
        return SurvivorshipLedgerEntry.from_dict(d)

    def list_entries(self) -> List[dict]:
        """Return manifest entries sorted by generated_at ascending."""
        entries = read_jsonl(self._manifest_path)
        return sorted(entries, key=lambda e: e["generated_at"])

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_payload(
        records: List[SurvivorRecord],
        families: List[FamilyRecord],
        topology: Optional[Dict[str, TopologyResult]],
        regime_resilience: Optional[RegimeResilienceResult],
        extra_metadata: Optional[dict],
    ) -> dict:
        """Build canonically-ordered payload dict."""
        # Canonical sort: experiment_id for records, family_id for families
        records_sorted = sorted(records, key=lambda r: r.experiment_id)
        families_sorted = sorted(families, key=lambda f: f.family_id)

        payload: dict = {
            "families": [f.to_dict() for f in families_sorted],
            "records": [r.to_dict() for r in records_sorted],
        }

        if topology is not None:
            payload["topology"] = {
                eid: topo.to_dict()
                for eid, topo in sorted(topology.items())
            }

        if regime_resilience is not None:
            payload["regime_resilience"] = regime_resilience.to_dict()

        if extra_metadata is not None:
            payload["extra_metadata"] = dict(sorted(extra_metadata.items()))

        return payload

    @staticmethod
    def _make_ledger_id(checksum: str, generated_at: str) -> str:
        """Deterministic ledger ID from checksum + timestamp (microseconds for uniqueness)."""
        compact_ts = (
            generated_at
            .replace("-", "").replace(":", "").replace("T", "").replace("Z", "").replace(".", "")
        )
        return f"surv_{compact_ts}_{checksum[:8]}"
