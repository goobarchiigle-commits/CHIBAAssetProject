"""
execution_epoch.py — Reconciliation and deployment epoch governance

Extends the runtime ExecutionEpoch concept with epochs scoped to:
  - ReconciliationEpoch: single reconciliation pass, linked to broker snapshot
  - DeploymentEpoch: single deployment decision, linked to reconciliation epoch
  - AuthorityEpochChain: ordered chain linking broker reality to deployment

Epoch IDs are deterministic: same inputs always produce same epoch_id.
All epochs are immutable. Append-only JSONL persistence.
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


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _append_jsonl(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[EPOCH] write failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# ReconciliationEpoch
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReconciliationEpoch:
    """Epoch scoped to a single reconciliation pass."""
    epoch_id: str
    run_id: str
    broker_snapshot_hash: str
    reconciliation_ok: bool
    blocking_mismatches: int
    created_at: str
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def create(
        run_id: str,
        broker_snapshot_hash: str,
        reconciliation_ok: bool,
        blocking_mismatches: int,
        created_at: str = "",
    ) -> "ReconciliationEpoch":
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()
        payload = _canonical_json({
            "blocking_mismatches": blocking_mismatches,
            "broker_snapshot_hash": broker_snapshot_hash,
            "created_at": created_at,
            "reconciliation_ok": reconciliation_ok,
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
        })
        return ReconciliationEpoch(
            epoch_id=_sha256(payload),
            run_id=run_id,
            broker_snapshot_hash=broker_snapshot_hash,
            reconciliation_ok=reconciliation_ok,
            blocking_mismatches=blocking_mismatches,
            created_at=created_at,
        )

    def validate(self) -> bool:
        try:
            expected = ReconciliationEpoch.create(
                run_id=self.run_id,
                broker_snapshot_hash=self.broker_snapshot_hash,
                reconciliation_ok=self.reconciliation_ok,
                blocking_mismatches=self.blocking_mismatches,
                created_at=self.created_at,
            )
            return expected.epoch_id == self.epoch_id
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# DeploymentEpoch
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DeploymentEpoch:
    """Epoch scoped to a single deployment decision."""
    epoch_id: str
    run_id: str
    reconciliation_epoch_id: str
    broker_snapshot_hash: str
    deployment_allowed: bool
    authority_level: str
    created_at: str
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def create(
        run_id: str,
        reconciliation_epoch_id: str,
        broker_snapshot_hash: str,
        deployment_allowed: bool,
        authority_level: str,
        created_at: str = "",
    ) -> "DeploymentEpoch":
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()
        payload = _canonical_json({
            "authority_level": authority_level,
            "broker_snapshot_hash": broker_snapshot_hash,
            "created_at": created_at,
            "deployment_allowed": deployment_allowed,
            "reconciliation_epoch_id": reconciliation_epoch_id,
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
        })
        return DeploymentEpoch(
            epoch_id=_sha256(payload),
            run_id=run_id,
            reconciliation_epoch_id=reconciliation_epoch_id,
            broker_snapshot_hash=broker_snapshot_hash,
            deployment_allowed=deployment_allowed,
            authority_level=authority_level,
            created_at=created_at,
        )

    def validate(self) -> bool:
        try:
            expected = DeploymentEpoch.create(
                run_id=self.run_id,
                reconciliation_epoch_id=self.reconciliation_epoch_id,
                broker_snapshot_hash=self.broker_snapshot_hash,
                deployment_allowed=self.deployment_allowed,
                authority_level=self.authority_level,
                created_at=self.created_at,
            )
            return expected.epoch_id == self.epoch_id
        except Exception:
            return False

    def is_epoch_consistent_with(self, reconciliation: ReconciliationEpoch) -> bool:
        """Verify this deployment epoch references the given reconciliation epoch."""
        return (
            self.reconciliation_epoch_id == reconciliation.epoch_id
            and self.broker_snapshot_hash == reconciliation.broker_snapshot_hash
        )


# ─────────────────────────────────────────────────────────────────────────────
# AuthorityEpochChain
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuthorityEpochChain:
    """
    Ordered chain linking broker snapshot → reconciliation → deployment.
    chain_hash provides cryptographic proof that all three are consistent.
    """
    run_id: str
    broker_snapshot_hash: str
    reconciliation_epoch_id: str
    deployment_epoch_id: str
    chain_hash: str
    created_at: str
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def create(
        run_id: str,
        broker_snapshot_hash: str,
        reconciliation_epoch_id: str,
        deployment_epoch_id: str,
        created_at: str = "",
    ) -> "AuthorityEpochChain":
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()
        chain_payload = _canonical_json({
            "broker_snapshot_hash": broker_snapshot_hash,
            "deployment_epoch_id": deployment_epoch_id,
            "reconciliation_epoch_id": reconciliation_epoch_id,
        })
        return AuthorityEpochChain(
            run_id=run_id,
            broker_snapshot_hash=broker_snapshot_hash,
            reconciliation_epoch_id=reconciliation_epoch_id,
            deployment_epoch_id=deployment_epoch_id,
            chain_hash=_sha256(chain_payload),
            created_at=created_at,
        )

    def validate(self) -> bool:
        try:
            chain_payload = _canonical_json({
                "broker_snapshot_hash": self.broker_snapshot_hash,
                "deployment_epoch_id": self.deployment_epoch_id,
                "reconciliation_epoch_id": self.reconciliation_epoch_id,
            })
            return _sha256(chain_payload) == self.chain_hash
        except Exception:
            return False

    def is_consistent_with(
        self,
        reconciliation: ReconciliationEpoch,
        deployment: DeploymentEpoch,
    ) -> bool:
        return (
            self.reconciliation_epoch_id == reconciliation.epoch_id
            and self.deployment_epoch_id == deployment.epoch_id
            and self.broker_snapshot_hash == reconciliation.broker_snapshot_hash
            and self.broker_snapshot_hash == deployment.broker_snapshot_hash
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build a complete authority chain from reconciliation + deployment
# ─────────────────────────────────────────────────────────────────────────────

def build_authority_chain(
    reconciliation: ReconciliationEpoch,
    deployment: DeploymentEpoch,
) -> AuthorityEpochChain:
    """
    Build and validate an AuthorityEpochChain from a reconciliation + deployment epoch pair.
    Raises ValueError if the two epochs are inconsistent.
    """
    if not deployment.is_epoch_consistent_with(reconciliation):
        raise ValueError(
            f"Epoch inconsistency: deployment references"
            f" reconciliation_epoch_id={deployment.reconciliation_epoch_id!r}"
            f" but got reconciliation epoch_id={reconciliation.epoch_id!r}"
        )
    return AuthorityEpochChain.create(
        run_id=reconciliation.run_id,
        broker_snapshot_hash=reconciliation.broker_snapshot_hash,
        reconciliation_epoch_id=reconciliation.epoch_id,
        deployment_epoch_id=deployment.epoch_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_reconciliation_epoch(epoch: ReconciliationEpoch, path: Path) -> None:
    _append_jsonl(path, asdict(epoch))


def append_deployment_epoch(epoch: DeploymentEpoch, path: Path) -> None:
    _append_jsonl(path, asdict(epoch))


def append_authority_chain(chain: AuthorityEpochChain, path: Path) -> None:
    _append_jsonl(path, asdict(chain))


def load_reconciliation_epochs(path: Path) -> List[ReconciliationEpoch]:
    if not path.exists():
        return []
    epochs: List[ReconciliationEpoch] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            epochs.append(ReconciliationEpoch(
                epoch_id=d["epoch_id"],
                run_id=d.get("run_id", ""),
                broker_snapshot_hash=d.get("broker_snapshot_hash", ""),
                reconciliation_ok=bool(d.get("reconciliation_ok", True)),
                blocking_mismatches=int(d.get("blocking_mismatches", 0)),
                created_at=d.get("created_at", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[EPOCH] reconciliation parse error: %s", exc)
    return epochs


def load_deployment_epochs(path: Path) -> List[DeploymentEpoch]:
    if not path.exists():
        return []
    epochs: List[DeploymentEpoch] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            epochs.append(DeploymentEpoch(
                epoch_id=d["epoch_id"],
                run_id=d.get("run_id", ""),
                reconciliation_epoch_id=d.get("reconciliation_epoch_id", ""),
                broker_snapshot_hash=d.get("broker_snapshot_hash", ""),
                deployment_allowed=bool(d.get("deployment_allowed", False)),
                authority_level=d.get("authority_level", AUTHORITY_LEVEL_BLOCKED),
                created_at=d.get("created_at", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[EPOCH] deployment parse error: %s", exc)
    return epochs


def load_authority_chains(path: Path) -> List[AuthorityEpochChain]:
    if not path.exists():
        return []
    chains: List[AuthorityEpochChain] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            chains.append(AuthorityEpochChain(
                run_id=d.get("run_id", ""),
                broker_snapshot_hash=d.get("broker_snapshot_hash", ""),
                reconciliation_epoch_id=d.get("reconciliation_epoch_id", ""),
                deployment_epoch_id=d.get("deployment_epoch_id", ""),
                chain_hash=d.get("chain_hash", ""),
                created_at=d.get("created_at", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[EPOCH] chain parse error: %s", exc)
    return chains
