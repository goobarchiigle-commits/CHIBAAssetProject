"""
src/runtime/determinism/cross_run_validator.py

CrossRunValidator: deterministic cross-run replay equivalence verifier.

Detects runtime determinism drift: same logical inputs must produce the
same replay hash across every repeated execution.

Design constraints enforced here:
- equivalence_key represents ONLY expected logical inputs (no runtime outputs)
- replay_hash is provided by the upstream authority (ReplayEquivalenceHash)
- sequences are NEVER auto-sorted; ordering drift triggers divergence
- comparison scope is bounded by COMPARISON_WINDOW per equivalence_key
- persisted artifact contains NO wall-clock field as comparison authority
- append-only JSONL; no record is ever mutated or deleted

equivalence_key = sha256(canonical_json(
    epoch_manifest + orchestration_manifest +
    dependency_manifest + governance_snapshot
))

IMPORTANT: runtime_decision_ledger is EXCLUDED from the equivalence_key.
It is part of the replay_hash authority, not the logical input grouping.
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .replay_hash import _canonical_json

VALIDATOR_VERSION:         str  = "1.0.0"
CANONICALIZATION_VERSION:  str  = "1.0.0"
COMPARISON_WINDOW:         int  = 20

CROSS_RUN_PATH: Path = Path("artifacts/runtime_validation/cross_run_validation.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# ValidationReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Immutable result of a single cross-run equivalence check."""
    validator_version:        str
    canonicalization_version: str
    validation_sequence_id:   str
    comparison_window:        int
    equivalence_key:          str
    authoritative_hash:       str
    compared_runs:            list[str]
    divergent_runs:           list[str]
    replay_hashes:            list[str]
    replay_stable:            bool
    divergence_detected:      bool
    quarantine_required:      bool
    severity:                 str   # "INFO" | "CRITICAL"
    invariant_status:         str   # "PASSED" | "FAILED"

    def to_record(self) -> dict:
        """Canonical dict suitable for JSONL persistence."""
        return {
            "validator_version":        self.validator_version,
            "canonicalization_version": self.canonicalization_version,
            "validation_sequence_id":   self.validation_sequence_id,
            "comparison_window":        self.comparison_window,
            "equivalence_key":          self.equivalence_key,
            "authoritative_hash":       self.authoritative_hash,
            "compared_runs":            list(self.compared_runs),
            "divergent_runs":           list(self.divergent_runs),
            "replay_hashes":            list(self.replay_hashes),
            "replay_stable":            self.replay_stable,
            "divergence_detected":      self.divergence_detected,
            "quarantine_required":      self.quarantine_required,
            "severity":                 self.severity,
            "invariant_status":         self.invariant_status,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CrossRunValidator
# ─────────────────────────────────────────────────────────────────────────────

class CrossRunValidator:
    """
    Deterministic cross-run replay equivalence verifier.

    Groups runs by equivalence_key (logical inputs only).
    For each group, all replay_hash values must match.
    Any mismatch → divergence_detected=True, severity=CRITICAL,
    quarantine_required=True, invariant_status=FAILED.

    Sequence safety: dict keys are canonically sorted; list/sequence ordering
    is PRESERVED. Reordering a list in any manifest changes the equivalence_key.

    Usage::
        validator = CrossRunValidator()
        report = validator.validate(
            run_id="run_001",
            replay_hash="<from upstream ReplayEquivalenceHash>",
            epoch_manifest={...},
            orchestration_manifest={...},
            dependency_manifest={...},
            governance_snapshot={...},
        )
        if report.quarantine_required:
            ...  # caller handles quarantine

    On init, reconstructs in-memory registry from existing JSONL so
    comparisons survive process restarts.
    """

    def __init__(
        self,
        output_path: Path = CROSS_RUN_PATH,
        window:      int  = COMPARISON_WINDOW,
    ) -> None:
        self._path   = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._window = window
        self._lock   = threading.Lock()
        # registry: equivalence_key → [(run_id, replay_hash), ...] (capped to window)
        self._registry: dict[str, list[tuple[str, str]]] = self._rebuild()

    # ── public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        run_id:                  str,
        replay_hash:             str,
        epoch_manifest:          Any,
        orchestration_manifest:  Any,
        dependency_manifest:     Any,
        governance_snapshot:     Any,
    ) -> ValidationReport:
        """
        Register run and check equivalence against prior runs sharing the same
        logical inputs. Returns ValidationReport; never raises on divergence
        (caller owns quarantine logic).

        replay_hash is treated as upstream authority — not recomputed here.
        """
        eq_key = self._compute_equivalence_key(
            epoch_manifest, orchestration_manifest,
            dependency_manifest, governance_snapshot,
        )

        with self._lock:
            # Take at most (window-1) prior entries so current fits in window.
            # Guard: when window==1, n_prior==0 and -0 == 0 in Python (full slice).
            n_prior = self._window - 1
            stored  = self._registry.get(eq_key, [])
            prior: list[tuple[str, str]] = (
                list(stored[-n_prior:]) if n_prior > 0 else []
            )

            # First recorded hash for this equivalence_key is authoritative
            authoritative_hash = prior[0][1] if prior else replay_hash

            # All runs in comparison scope (ordered: prior → current)
            all_runs: list[tuple[str, str]] = prior + [(run_id, replay_hash)]

            divergent = [
                (r_id, r_hash)
                for r_id, r_hash in all_runs
                if r_hash != authoritative_hash
            ]
            divergence_detected = bool(divergent)

            report = ValidationReport(
                validator_version        = VALIDATOR_VERSION,
                canonicalization_version = CANONICALIZATION_VERSION,
                validation_sequence_id   = uuid.uuid4().hex,
                comparison_window        = self._window,
                equivalence_key          = eq_key,
                authoritative_hash       = authoritative_hash,
                compared_runs            = [r[0] for r in all_runs],
                divergent_runs           = [r[0] for r in divergent],
                replay_hashes            = [r[1] for r in all_runs],
                replay_stable            = not divergence_detected,
                divergence_detected      = divergence_detected,
                quarantine_required      = divergence_detected,
                severity                 = "CRITICAL" if divergence_detected else "INFO",
                invariant_status         = "FAILED"  if divergence_detected else "PASSED",
            )

            # Update registry (always cap to window)
            entries = list(self._registry.get(eq_key, []))
            entries.append((run_id, replay_hash))
            self._registry[eq_key] = entries[-self._window:]

            self._persist(report)

        return report

    def load_records(self) -> list[dict]:
        """Read all persisted validation records in append order."""
        if not self._path.exists():
            return []
        results: list[dict] = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return results

    # ── equivalence key ───────────────────────────────────────────────────────

    def _compute_equivalence_key(
        self,
        epoch_manifest:         Any,
        orchestration_manifest: Any,
        dependency_manifest:    Any,
        governance_snapshot:    Any,
    ) -> str:
        """
        sha256 of canonical JSON of the four logical input manifests.

        Canonicalization:
        - dict keys: recursively sorted (insertion order irrelevant)
        - list/sequence: ORDER PRESERVED (sequence drift changes the key)

        runtime_decision_ledger and replay_hash are intentionally excluded.
        """
        payload = _canonical_json({
            "epoch_manifest":         epoch_manifest,
            "orchestration_manifest": orchestration_manifest,
            "dependency_manifest":    dependency_manifest,
            "governance_snapshot":    governance_snapshot,
        })
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── persistence ───────────────────────────────────────────────────────────

    def _persist(self, report: ValidationReport) -> None:
        """Append one canonical UTF-8 JSON line. Called within self._lock."""
        entry = json.dumps(
            report.to_record(),
            separators=(",", ":"),
            ensure_ascii=True,
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")

    # ── registry reconstruction ───────────────────────────────────────────────

    def _rebuild(self) -> dict[str, list[tuple[str, str]]]:
        """
        Rebuild in-memory registry from existing JSONL on startup.

        From each persisted record, the current run is the last entry in
        compared_runs / replay_hashes (guaranteed by _persist ordering).
        Registry is capped to self._window entries per equivalence_key.
        """
        registry: dict[str, list[tuple[str, str]]] = {}
        if not self._path.exists():
            return registry
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eq_key       = rec.get("equivalence_key", "")
                compared     = rec.get("compared_runs", [])
                hashes       = rec.get("replay_hashes", [])
                if not eq_key or not compared or not hashes:
                    continue
                # Last item is the run that triggered this validation record
                run_id      = compared[-1]
                r_hash      = hashes[-1]
                if eq_key not in registry:
                    registry[eq_key] = []
                registry[eq_key].append((run_id, r_hash))
        return {k: v[-self._window:] for k, v in registry.items()}
