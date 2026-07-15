import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ValidationResult:
    valid: bool
    reason: str
    checksum: Optional[str] = None

class ArtifactCompletionValidator:
    def validate(self, artifact_path: str,
                 expected_checksum: Optional[str] = None,
                 min_bytes: int = 1,
                 epoch_id: Optional[str] = None,
                 artifact_epoch_id: Optional[str] = None) -> ValidationResult:
        p = Path(artifact_path)
        if not p.exists():
            return ValidationResult(False, "artifact_not_found")
        size = p.stat().st_size
        if size < min_bytes:
            return ValidationResult(False, f"partial_write_detected:size={size}<{min_bytes}")
        checksum = hashlib.sha256(p.read_bytes()).hexdigest()
        if expected_checksum and checksum != expected_checksum:
            return ValidationResult(False, f"checksum_mismatch:got={checksum}")
        if epoch_id and artifact_epoch_id and epoch_id != artifact_epoch_id:
            return ValidationResult(False, f"epoch_inconsistency")
        return ValidationResult(True, "ok", checksum=checksum)
