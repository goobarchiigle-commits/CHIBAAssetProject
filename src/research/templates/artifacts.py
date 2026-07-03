"""Template artifact storage — append-only, immutable, checksum-validated.

Stores:
    artifacts/templates/
      manifest.jsonl          — append-only index of all generation manifests
      templates/              — immutable template definitions
      generations/            — immutable generation manifests
      fingerprints/           — structural fingerprint reports
      spaces/                 — space snapshots
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

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
from src.research.templates import SCHEMA_VERSION


class TemplateArtifacts:
    """Append-only artifact store for hypothesis templates and generation manifests.

    Args:
        base_dir: Base directory for all template artifacts.
                  Defaults to artifacts/templates/ relative to CWD.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path("artifacts/templates")
        self._manifest_path = self.base_dir / "manifest.jsonl"
        self._templates_dir = self.base_dir / "templates"
        self._generations_dir = self.base_dir / "generations"
        self._fingerprints_dir = self.base_dir / "fingerprints"
        self._spaces_dir = self.base_dir / "spaces"

    # ── Template artifacts ─────────────────────────────────────────────────────

    def write_template(self, template_dict: dict) -> Path:
        """Persist an immutable template definition. Returns artifact path."""
        template_id = template_dict["template_id"]
        path = self._templates_dir / f"{template_id}.json"
        checksum = compute_checksum(template_dict)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "template",
            "checksum": checksum,
            "payload": template_dict,
        }
        write_immutable(path, artifact)
        self._append_manifest("template", template_id, path.name, checksum)
        return path

    def load_template(self, template_id: str) -> dict:
        """Load and validate a template artifact."""
        path = self._templates_dir / f"{template_id}.json"
        d = read_artifact(path)
        if not verify_checksum(d["payload"], d["checksum"]):
            raise ValueError(f"Checksum mismatch for template {template_id}")
        return d["payload"]

    # ── Generation manifest artifacts ──────────────────────────────────────────

    def write_generation_manifest(self, manifest_dict: dict) -> Path:
        """Persist an immutable generation manifest."""
        manifest_id = manifest_dict["manifest_id"]
        path = self._generations_dir / f"{manifest_id}.json"
        checksum = compute_checksum(manifest_dict)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "generation_manifest",
            "checksum": checksum,
            "payload": manifest_dict,
        }
        write_immutable(path, artifact)
        self._append_manifest("generation_manifest", manifest_id, path.name, checksum)
        return path

    def load_generation_manifest(self, manifest_id: str) -> dict:
        path = self._generations_dir / f"{manifest_id}.json"
        d = read_artifact(path)
        if not verify_checksum(d["payload"], d["checksum"]):
            raise ValueError(f"Checksum mismatch for manifest {manifest_id}")
        return d["payload"]

    # ── Fingerprint report artifacts ───────────────────────────────────────────

    def write_fingerprint_report(self, report: dict, report_id: str) -> Path:
        """Persist a structural fingerprint report."""
        path = self._fingerprints_dir / f"{report_id}.json"
        checksum = compute_checksum(report)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "fingerprint_report",
            "checksum": checksum,
            "payload": report,
        }
        write_immutable(path, artifact)
        self._append_manifest("fingerprint_report", report_id, path.name, checksum)
        return path

    # ── Space snapshot artifacts ───────────────────────────────────────────────

    def write_space_snapshot(self, snapshot: dict) -> Path:
        """Persist a hypothesis space snapshot."""
        space_id = snapshot["space_id"]
        space_hash = snapshot["space_hash"]
        artifact_name = f"{space_id}_{space_hash}.json"
        path = self._spaces_dir / artifact_name
        checksum = compute_checksum(snapshot)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "space_snapshot",
            "checksum": checksum,
            "payload": snapshot,
        }
        write_immutable(path, artifact)
        self._append_manifest("space_snapshot", space_id, artifact_name, checksum)
        return path

    # ── Manifest index ─────────────────────────────────────────────────────────

    def list_manifests(self) -> List[dict]:
        """Return all manifest entries, sorted by generated_at."""
        entries = read_jsonl(self._manifest_path)
        return sorted(entries, key=lambda e: e.get("generated_at", ""))

    def list_templates(self) -> List[str]:
        """Return sorted list of stored template IDs."""
        if not self._templates_dir.exists():
            return []
        return sorted(
            p.stem for p in self._templates_dir.iterdir() if p.suffix == ".json"
        )

    def list_generation_manifests(self) -> List[str]:
        """Return sorted list of stored generation manifest IDs."""
        if not self._generations_dir.exists():
            return []
        return sorted(
            p.stem for p in self._generations_dir.iterdir() if p.suffix == ".json"
        )

    # ── Private ────────────────────────────────────────────────────────────────

    def _append_manifest(
        self,
        artifact_type: str,
        artifact_id: str,
        filename: str,
        checksum: str,
    ) -> None:
        record = {
            "artifact_type": artifact_type,
            "artifact_id": artifact_id,
            "filename": filename,
            "checksum": checksum,
            "generated_at": utc_now_iso(),
        }
        append_jsonl(self._manifest_path, record)
