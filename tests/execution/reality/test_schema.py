"""Tests: schema.py — SchemaVersion, ExecutionRealitySchemaVersion."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.schema import (
    SchemaVersion, ExecutionRealitySchemaVersion,
    CURRENT_MAJOR, CURRENT_MINOR, CURRENT_PATCH,
)


class TestSchemaVersion:
    def test_str(self):
        sv = SchemaVersion(1, 0, 0)
        assert str(sv) == "1.0.0"

    def test_frozen(self):
        sv = SchemaVersion(1, 0, 0)
        with pytest.raises(Exception):
            sv.major = 2  # type: ignore

    def test_compatible_same_major(self):
        a = SchemaVersion(1, 0, 0)
        b = SchemaVersion(1, 5, 3)
        assert a.is_compatible_with(b)

    def test_incompatible_different_major(self):
        a = SchemaVersion(1, 0, 0)
        b = SchemaVersion(2, 0, 0)
        assert not a.is_compatible_with(b)

    def test_to_dict(self):
        sv = SchemaVersion(1, 2, 3)
        d = sv.to_dict()
        assert d == {"major": 1, "minor": 2, "patch": 3}

    def test_from_str(self):
        sv = SchemaVersion.from_str("1.2.3")
        assert sv.major == 1 and sv.minor == 2 and sv.patch == 3

    def test_from_str_invalid_raises(self):
        with pytest.raises(ValueError):
            SchemaVersion.from_str("1.0")

    def test_from_dict(self):
        sv = SchemaVersion.from_dict({"major": 2, "minor": 1, "patch": 0})
        assert sv == SchemaVersion(2, 1, 0)

    def test_roundtrip(self):
        sv = SchemaVersion(1, 0, 0)
        assert SchemaVersion.from_dict(sv.to_dict()) == sv


class TestExecutionRealitySchemaVersion:
    def test_for_component(self):
        erv = ExecutionRealitySchemaVersion.for_component("snapshot")
        assert erv.component == "snapshot"
        assert str(erv.version) == f"{CURRENT_MAJOR}.{CURRENT_MINOR}.{CURRENT_PATCH}"

    def test_frozen(self):
        erv = ExecutionRealitySchemaVersion.for_component("ledger")
        with pytest.raises(Exception):
            erv.component = "other"  # type: ignore

    def test_schema_id_16_chars(self):
        erv = ExecutionRealitySchemaVersion.for_component("snapshot")
        assert len(erv.schema_id) == 16

    def test_schema_id_deterministic(self):
        a = ExecutionRealitySchemaVersion.for_component("snapshot")
        b = ExecutionRealitySchemaVersion.for_component("snapshot")
        assert a.schema_id == b.schema_id

    def test_schema_id_differs_by_component(self):
        a = ExecutionRealitySchemaVersion.for_component("snapshot")
        b = ExecutionRealitySchemaVersion.for_component("ledger")
        assert a.schema_id != b.schema_id

    def test_to_dict_keys(self):
        erv = ExecutionRealitySchemaVersion.for_component("regime")
        d = erv.to_dict()
        assert "version" in d and "component" in d and "schema_id" in d

    def test_version_is_current(self):
        erv = ExecutionRealitySchemaVersion.for_component("x")
        assert erv.version.major == CURRENT_MAJOR
