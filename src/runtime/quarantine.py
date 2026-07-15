import json, time, shutil, hashlib
from pathlib import Path

class RuntimeQuarantineZone:
    def __init__(self, quarantine_dir: str = "artifacts/runtime/quarantine",
                 manifest_path: str = "artifacts/runtime/quarantine_manifest.jsonl"):
        self._dir = Path(quarantine_dir)
        self._manifest = Path(manifest_path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest.parent.mkdir(parents=True, exist_ok=True)

    def quarantine(self, artifact_path: str, reason: str,
                   run_id: str, epoch_id: str) -> str:
        src = Path(artifact_path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        ts = int(time.time())
        dest = self._dir / f"{ts}_{run_id}_{src.name}"
        shutil.copy2(str(src), str(dest))
        checksum = hashlib.sha256(dest.read_bytes()).hexdigest()
        entry = {"ts": ts, "run_id": run_id, "epoch_id": epoch_id,
                 "reason": reason, "original_path": str(artifact_path),
                 "quarantine_path": str(dest), "checksum": checksum}
        with self._manifest.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return str(dest)

    def list_quarantined(self) -> list:
        if not self._manifest.exists():
            return []
        with self._manifest.open("r") as f:
            return [json.loads(l) for l in f if l.strip()]
