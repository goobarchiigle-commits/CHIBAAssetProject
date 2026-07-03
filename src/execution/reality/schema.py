"""Schema versioning for execution reality artifacts."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

CURRENT_MAJOR = 1
CURRENT_MINOR = 0
CURRENT_PATCH = 0


@dataclass(frozen=True)
class SchemaVersion:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        return self.major == other.major

    def to_dict(self) -> dict:
        return {"major": self.major, "minor": self.minor, "patch": self.patch}

    @classmethod
    def from_str(cls, s: str) -> "SchemaVersion":
        parts = s.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid schema version: {s!r}")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))

    @classmethod
    def from_dict(cls, d: dict) -> "SchemaVersion":
        return cls(d["major"], d["minor"], d["patch"])


@dataclass(frozen=True)
class ExecutionRealitySchemaVersion:
    version: SchemaVersion
    component: str
    schema_id: str          # 16-char SHA256 of (component, str(version))

    @classmethod
    def for_component(cls, component: str) -> "ExecutionRealitySchemaVersion":
        version = SchemaVersion(CURRENT_MAJOR, CURRENT_MINOR, CURRENT_PATCH)
        raw = f"{component}:{version}"
        schema_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return cls(version=version, component=component, schema_id=schema_id)

    def to_dict(self) -> dict:
        return {
            "version": str(self.version),
            "component": self.component,
            "schema_id": self.schema_id,
        }
