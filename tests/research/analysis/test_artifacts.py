"""Tests: artifacts.py — checksums, immutability, append-only JSONL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.analysis.artifacts import (
    ImmutabilityError,
    append_jsonl,
    compute_checksum,
    read_artifact,
    read_jsonl,
    stable_hash,
    verify_checksum,
    write_immutable,
)


class TestChecksum:
    def test_deterministic(self):
        payload = {"a": 1, "b": [1, 2, 3]}
        assert compute_checksum(payload) == compute_checksum(payload)

    def test_key_order_invariant(self):
        p1 = {"b": 2, "a": 1}
        p2 = {"a": 1, "b": 2}
        assert compute_checksum(p1) == compute_checksum(p2)

    def test_different_payloads_differ(self):
        assert compute_checksum({"a": 1}) != compute_checksum({"a": 2})

    def test_verify_valid(self):
        p = {"x": 42}
        cs = compute_checksum(p)
        assert verify_checksum(p, cs) is True

    def test_verify_invalid(self):
        p = {"x": 42}
        assert verify_checksum(p, "deadbeef" * 8) is False

    def test_full_sha256_length(self):
        cs = compute_checksum({"a": 1})
        assert len(cs) == 64  # full SHA256 hex


class TestStableHash:
    def test_length(self):
        assert len(stable_hash({"x": 1})) == 16

    def test_deterministic(self):
        assert stable_hash({"a": [1, 2]}) == stable_hash({"a": [1, 2]})

    def test_key_order_invariant(self):
        assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


class TestWriteImmutable:
    def test_write_and_read(self, tmp_path):
        p = tmp_path / "artifact.json"
        write_immutable(p, {"key": "value"})
        d = read_artifact(p)
        assert d["key"] == "value"

    def test_overwrite_raises(self, tmp_path):
        p = tmp_path / "artifact.json"
        write_immutable(p, {"v": 1})
        with pytest.raises(ImmutabilityError):
            write_immutable(p, {"v": 2})

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "file.json"
        write_immutable(p, {"ok": True})
        assert p.exists()

    def test_sorted_keys_in_file(self, tmp_path):
        p = tmp_path / "a.json"
        write_immutable(p, {"b": 2, "a": 1})
        text = p.read_text(encoding="utf-8")
        # "a" must appear before "b" in the file
        assert text.index('"a"') < text.index('"b"')

    def test_read_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_artifact(tmp_path / "missing.json")


class TestAppendJsonl:
    def test_appends_multiple(self, tmp_path):
        p = tmp_path / "log.jsonl"
        append_jsonl(p, {"i": 1})
        append_jsonl(p, {"i": 2})
        lines = read_jsonl(p)
        assert len(lines) == 2
        assert lines[0]["i"] == 1
        assert lines[1]["i"] == 2

    def test_read_jsonl_empty_missing(self, tmp_path):
        assert read_jsonl(tmp_path / "nonexistent.jsonl") == []

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "deep" / "log.jsonl"
        append_jsonl(p, {"x": 1})
        assert p.exists()

    def test_records_sorted_keys(self, tmp_path):
        p = tmp_path / "log.jsonl"
        append_jsonl(p, {"z": 3, "a": 1})
        text = p.read_text(encoding="utf-8")
        assert text.index('"a"') < text.index('"z"')
