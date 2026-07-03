"""src/research/features — deterministic reusable feature cache layer."""
from src.research.features.contract import FeatureContract, FeatureMetadata
from src.research.features.registry import FeatureRegistry, FeatureRegistryError
from src.research.features.cache import (
    FeatureArtifacts,
    FeatureCache,
    CacheCollisionError,
    CacheChecksumError,
)
from src.research.features.pipeline import FeaturePipeline, FeatureSpec, FeaturePipelineError
from src.research.features.snapshot import FeatureSnapshot

__all__ = [
    "FeatureContract",
    "FeatureMetadata",
    "FeatureRegistry",
    "FeatureRegistryError",
    "FeatureArtifacts",
    "FeatureCache",
    "CacheCollisionError",
    "CacheChecksumError",
    "FeaturePipeline",
    "FeatureSpec",
    "FeaturePipelineError",
    "FeatureSnapshot",
]
