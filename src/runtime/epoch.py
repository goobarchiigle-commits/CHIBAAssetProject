import hashlib, json, time
from dataclasses import dataclass

@dataclass(frozen=True)
class ExecutionEpoch:
    epoch_id: str
    dataset_manifest: str
    feature_manifest: str
    governance_state: str
    schema_version: str
    template_state: str
    reality_schema_version: str
    created_at: float

    @staticmethod
    def create(dataset_manifest, feature_manifest, governance_state,
               schema_version, template_state, reality_schema_version):
        payload = json.dumps({
            "dataset_manifest": dataset_manifest,
            "feature_manifest": feature_manifest,
            "governance_state": governance_state,
            "schema_version": schema_version,
            "template_state": template_state,
            "reality_schema_version": reality_schema_version,
        }, sort_keys=True)
        epoch_id = hashlib.sha256(payload.encode()).hexdigest()
        return ExecutionEpoch(
            epoch_id=epoch_id,
            dataset_manifest=dataset_manifest,
            feature_manifest=feature_manifest,
            governance_state=governance_state,
            schema_version=schema_version,
            template_state=template_state,
            reality_schema_version=reality_schema_version,
            created_at=time.time(),
        )

    def validate_consistency(self, other: "ExecutionEpoch") -> bool:
        return self.epoch_id == other.epoch_id
