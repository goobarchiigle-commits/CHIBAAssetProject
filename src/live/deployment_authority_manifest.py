"""
deployment_authority_manifest.py — Immutable deployment authorization manifests

Each deployment decision is captured in an immutable manifest that links:
  - reconciliation epoch (authority provenance)
  - broker snapshot hash (broker-authoritative linkage)
  - authority level (full/partial/blocked/quarantined)
  - integrity status
  - quarantine status
  - pending authority divergences
  - authorization reason

Execution gates:
  - require_manifest(): raises RuntimeError if manifest is missing or not authorized
  - Deployment blocked on: quarantine, integrity failure, pending divergences,
    missing epoch IDs, blocked authority level

Append-only JSONL. Deterministic serialization. Replay-compatible.
manifest_id is sha256 of all fields — tamper-evident.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

AUTHORITY_LEVEL_FULL        = "full"
AUTHORITY_LEVEL_PARTIAL     = "partial"
AUTHORITY_LEVEL_BLOCKED     = "blocked"
AUTHORITY_LEVEL_QUARANTINED = "quarantined"

MANIFEST_AUTHORIZED  = "authorized"
MANIFEST_BLOCKED     = "blocked"
MANIFEST_QUARANTINED = "quarantined"


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _compute_deployment_status(
    authority_level: str,
    integrity_ok: bool,
    is_quarantined: bool,
    pending_divergences: int,
    reconciliation_epoch_id: str,
    deployment_epoch_id: str,
) -> str:
    """Deterministic deployment status derivation from inputs."""
    if is_quarantined:
        return MANIFEST_QUARANTINED
    if not integrity_ok:
        return MANIFEST_BLOCKED
    if authority_level in (AUTHORITY_LEVEL_BLOCKED, AUTHORITY_LEVEL_QUARANTINED):
        return MANIFEST_BLOCKED
    if pending_divergences > 0:
        return MANIFEST_BLOCKED
    if not reconciliation_epoch_id or not deployment_epoch_id:
        return MANIFEST_BLOCKED
    if authority_level in (AUTHORITY_LEVEL_FULL, AUTHORITY_LEVEL_PARTIAL):
        return MANIFEST_AUTHORIZED
    return MANIFEST_BLOCKED


# ─────────────────────────────────────────────────────────────────────────────
# DeploymentAuthorityManifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeploymentAuthorityManifest:
    """
    Immutable authorization manifest for a single deployment decision.
    manifest_id = sha256 of canonical JSON of all fields (tamper-evident).
    """
    manifest_id: str
    run_id: str
    reconciliation_epoch_id: str
    deployment_epoch_id: str
    broker_snapshot_hash: str
    authority_level: str
    integrity_ok: bool
    integrity_violation_count: int
    is_quarantined: bool
    quarantine_id: Optional[str]
    pending_divergences: int
    deployment_status: str
    authorization_reason: str
    lineage_entry_hash: str
    created_at: str
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def _fields_for_id(
        run_id: str,
        reconciliation_epoch_id: str,
        deployment_epoch_id: str,
        broker_snapshot_hash: str,
        authority_level: str,
        integrity_ok: bool,
        integrity_violation_count: int,
        is_quarantined: bool,
        quarantine_id: Optional[str],
        pending_divergences: int,
        deployment_status: str,
        authorization_reason: str,
        lineage_entry_hash: str,
        created_at: str,
        schema_version: int,
    ) -> dict:
        return {
            "authority_level": authority_level,
            "authorization_reason": authorization_reason,
            "broker_snapshot_hash": broker_snapshot_hash,
            "created_at": created_at,
            "deployment_epoch_id": deployment_epoch_id,
            "deployment_status": deployment_status,
            "integrity_ok": integrity_ok,
            "integrity_violation_count": integrity_violation_count,
            "is_quarantined": is_quarantined,
            "lineage_entry_hash": lineage_entry_hash,
            "pending_divergences": pending_divergences,
            "quarantine_id": quarantine_id,
            "reconciliation_epoch_id": reconciliation_epoch_id,
            "run_id": run_id,
            "schema_version": schema_version,
        }

    @staticmethod
    def create(
        run_id: str,
        reconciliation_epoch_id: str,
        deployment_epoch_id: str,
        broker_snapshot_hash: str,
        authority_level: str,
        integrity_ok: bool,
        integrity_violation_count: int,
        is_quarantined: bool,
        quarantine_id: Optional[str],
        pending_divergences: int,
        authorization_reason: str,
        lineage_entry_hash: str = "",
        created_at: str = "",
    ) -> "DeploymentAuthorityManifest":
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()

        deployment_status = _compute_deployment_status(
            authority_level=authority_level,
            integrity_ok=integrity_ok,
            is_quarantined=is_quarantined,
            pending_divergences=pending_divergences,
            reconciliation_epoch_id=reconciliation_epoch_id,
            deployment_epoch_id=deployment_epoch_id,
        )

        fields = DeploymentAuthorityManifest._fields_for_id(
            run_id=run_id,
            reconciliation_epoch_id=reconciliation_epoch_id,
            deployment_epoch_id=deployment_epoch_id,
            broker_snapshot_hash=broker_snapshot_hash,
            authority_level=authority_level,
            integrity_ok=integrity_ok,
            integrity_violation_count=integrity_violation_count,
            is_quarantined=is_quarantined,
            quarantine_id=quarantine_id,
            pending_divergences=pending_divergences,
            deployment_status=deployment_status,
            authorization_reason=authorization_reason,
            lineage_entry_hash=lineage_entry_hash,
            created_at=created_at,
            schema_version=SCHEMA_VERSION,
        )
        manifest_id = _sha256(_canonical_json(fields))

        return DeploymentAuthorityManifest(
            manifest_id=manifest_id,
            run_id=run_id,
            reconciliation_epoch_id=reconciliation_epoch_id,
            deployment_epoch_id=deployment_epoch_id,
            broker_snapshot_hash=broker_snapshot_hash,
            authority_level=authority_level,
            integrity_ok=integrity_ok,
            integrity_violation_count=integrity_violation_count,
            is_quarantined=is_quarantined,
            quarantine_id=quarantine_id,
            pending_divergences=pending_divergences,
            deployment_status=deployment_status,
            authorization_reason=authorization_reason,
            lineage_entry_hash=lineage_entry_hash,
            created_at=created_at,
        )

    def is_authorized(self) -> bool:
        return self.deployment_status == MANIFEST_AUTHORIZED

    def validate(self) -> bool:
        """Recompute manifest_id from stored fields and verify match."""
        try:
            fields = self._fields_for_id(
                run_id=self.run_id,
                reconciliation_epoch_id=self.reconciliation_epoch_id,
                deployment_epoch_id=self.deployment_epoch_id,
                broker_snapshot_hash=self.broker_snapshot_hash,
                authority_level=self.authority_level,
                integrity_ok=self.integrity_ok,
                integrity_violation_count=self.integrity_violation_count,
                is_quarantined=self.is_quarantined,
                quarantine_id=self.quarantine_id,
                pending_divergences=self.pending_divergences,
                deployment_status=self.deployment_status,
                authorization_reason=self.authorization_reason,
                lineage_entry_hash=self.lineage_entry_hash,
                created_at=self.created_at,
                schema_version=self.schema_version,
            )
            return _sha256(_canonical_json(fields)) == self.manifest_id
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builder
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest(
    run_id: str,
    reconciliation_epoch_id: str,
    deployment_epoch_id: str,
    broker_snapshot_hash: str,
    integrity_ok: bool,
    integrity_violation_count: int,
    is_quarantined: bool,
    quarantine_id: Optional[str],
    pending_divergences: int,
    authority_level: Optional[str] = None,
    authorization_reason: str = "",
    lineage_entry_hash: str = "",
) -> DeploymentAuthorityManifest:
    """
    Build manifest, deriving authority_level and authorization_reason if not provided.
    """
    if authority_level is None:
        if is_quarantined:
            authority_level = AUTHORITY_LEVEL_QUARANTINED
        elif not integrity_ok or pending_divergences > 0:
            authority_level = AUTHORITY_LEVEL_BLOCKED
        else:
            authority_level = AUTHORITY_LEVEL_FULL

    if not authorization_reason:
        if is_quarantined:
            authorization_reason = "quarantine active — deployment blocked"
        elif not integrity_ok:
            authorization_reason = f"integrity violations: {integrity_violation_count}"
        elif pending_divergences > 0:
            authorization_reason = f"pending authority divergences: {pending_divergences}"
        else:
            authorization_reason = "all checks passed"

    return DeploymentAuthorityManifest.create(
        run_id=run_id,
        reconciliation_epoch_id=reconciliation_epoch_id,
        deployment_epoch_id=deployment_epoch_id,
        broker_snapshot_hash=broker_snapshot_hash,
        authority_level=authority_level,
        integrity_ok=integrity_ok,
        integrity_violation_count=integrity_violation_count,
        is_quarantined=is_quarantined,
        quarantine_id=quarantine_id,
        pending_divergences=pending_divergences,
        authorization_reason=authorization_reason,
        lineage_entry_hash=lineage_entry_hash,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Execution gate
# ─────────────────────────────────────────────────────────────────────────────

def require_manifest(
    path: Path,
    run_id: str,
) -> "DeploymentAuthorityManifest":
    """
    Return latest authorized manifest for run_id.
    Raises RuntimeError if manifest is missing or not authorized.
    Call at execution entry points to enforce the manifest requirement.
    """
    manifest = get_latest_manifest(path, run_id)
    if manifest is None:
        raise RuntimeError(
            f"[MANIFEST] No deployment authority manifest for run_id={run_id!r}."
            " Execution blocked: manifest required before deployment."
        )
    if not manifest.is_authorized():
        raise RuntimeError(
            f"[MANIFEST] Deployment not authorized for run_id={run_id!r}:"
            f" status={manifest.deployment_status!r}"
            f" reason={manifest.authorization_reason!r}"
        )
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_manifest(manifest: DeploymentAuthorityManifest, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning("[MANIFEST] write failed: %s", exc)


def load_manifests(path: Path) -> List[DeploymentAuthorityManifest]:
    if not path.exists():
        return []
    result: List[DeploymentAuthorityManifest] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            result.append(DeploymentAuthorityManifest(
                manifest_id=d.get("manifest_id", ""),
                run_id=d.get("run_id", ""),
                reconciliation_epoch_id=d.get("reconciliation_epoch_id", ""),
                deployment_epoch_id=d.get("deployment_epoch_id", ""),
                broker_snapshot_hash=d.get("broker_snapshot_hash", ""),
                authority_level=d.get("authority_level", AUTHORITY_LEVEL_BLOCKED),
                integrity_ok=bool(d.get("integrity_ok", False)),
                integrity_violation_count=int(d.get("integrity_violation_count", 0)),
                is_quarantined=bool(d.get("is_quarantined", False)),
                quarantine_id=d.get("quarantine_id"),
                pending_divergences=int(d.get("pending_divergences", 0)),
                deployment_status=d.get("deployment_status", MANIFEST_BLOCKED),
                authorization_reason=d.get("authorization_reason", ""),
                lineage_entry_hash=d.get("lineage_entry_hash", ""),
                created_at=d.get("created_at", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[MANIFEST] parse error: %s", exc)
    return result


def get_latest_manifest(
    path: Path,
    run_id: str,
) -> Optional[DeploymentAuthorityManifest]:
    """Return the most recent manifest for run_id, or None."""
    manifests = [m for m in load_manifests(path) if m.run_id == run_id]
    if not manifests:
        return None
    return sorted(manifests, key=lambda m: m.created_at)[-1]
