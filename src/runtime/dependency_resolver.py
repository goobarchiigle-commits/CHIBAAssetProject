import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class DependencyCheckResult:
    satisfied: bool
    violations: List[str]

class DependencyResolver:
    def check(self, epoch_id: str, required_artifacts: List[str],
              feature_manifest_path: Optional[str] = None,
              max_dataset_age_hours: float = 25.0) -> DependencyCheckResult:
        violations = []
        for art in required_artifacts:
            p = Path(art)
            if not p.exists():
                violations.append(f"missing_artifact:{art}")
            elif (time.time() - p.stat().st_mtime) / 3600 > max_dataset_age_hours:
                violations.append(f"stale_artifact:{art}")
        if feature_manifest_path and not Path(feature_manifest_path).exists():
            violations.append(f"missing_feature_manifest:{feature_manifest_path}")
        return DependencyCheckResult(satisfied=not violations, violations=violations)
