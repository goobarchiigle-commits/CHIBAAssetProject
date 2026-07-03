"""
Tests for intent_snapshot.py — SHA-256 hash chain audit log.
"""
import json
import pytest
from src.truth.intent_snapshot import (
    IntentSnapshot, GENESIS_HASH,
    make_snapshot, finalize_snapshot, append_snapshot,
    load_chain, verify_chain, replay_chain, get_tail_hash,
    _compute_hash, _canonical_json,
)


def make_snap(
    intent_id="i-001",
    symbol="1234",
    prev_state="CREATED",
    new_state="SUBMITTED",
    prev_hash=GENESIS_HASH,
    **kwargs,
) -> IntentSnapshot:
    return make_snapshot(
        intent_id=intent_id,
        symbol=symbol,
        side="BUY",
        requested_qty=5,
        prev_state=prev_state,
        new_state=new_state,
        reason="test",
        trading_day="2026-05-16",
        strategy="turtle",
        prev_hash=prev_hash,
        **kwargs,
    )


# ── Hash computation ──────────────────────────────────────────────────────────

class TestHashComputation:
    def test_entry_hash_computed(self):
        snap = make_snap()
        assert snap.entry_hash != ""
        assert len(snap.entry_hash) == 64

    def test_entry_hash_excludes_itself(self):
        snap = make_snap()
        # canonical JSON must not contain entry_hash
        canonical = _canonical_json(snap)
        data = json.loads(canonical)
        assert "entry_hash" not in data

    def test_hash_deterministic(self):
        snap = make_snap()
        h1 = _compute_hash(snap)
        h2 = _compute_hash(snap)
        assert h1 == h2

    def test_different_content_different_hash(self):
        s1 = make_snap(symbol="1234")
        s2 = make_snap(symbol="5678")
        assert s1.entry_hash != s2.entry_hash

    def test_first_entry_uses_genesis_hash(self):
        snap = make_snap()
        assert snap.prev_hash == GENESIS_HASH


# ── Chain verification ────────────────────────────────────────────────────────

class TestVerifyChain:
    def test_empty_chain_valid(self):
        ok, violations = verify_chain([])
        assert ok
        assert violations == []

    def test_single_entry_valid(self):
        snap = make_snap()
        ok, violations = verify_chain([snap])
        assert ok

    def test_two_entry_chain_valid(self):
        s1 = make_snap(new_state="SUBMITTED")
        s2 = make_snap(prev_state="SUBMITTED", new_state="ACKED",
                       prev_hash=s1.entry_hash)
        ok, violations = verify_chain([s1, s2])
        assert ok, violations

    def test_broken_prev_hash_detected(self):
        s1 = make_snap()
        s2 = make_snap(prev_state="SUBMITTED", new_state="ACKED",
                       prev_hash="wrong" + "0" * 60)
        ok, violations = verify_chain([s1, s2])
        assert not ok
        assert any("prev_hash" in v for v in violations)

    def test_tampered_entry_hash_detected(self):
        import dataclasses
        snap = make_snap()
        tampered = dataclasses.replace(snap, entry_hash="a" * 64)
        ok, violations = verify_chain([tampered])
        assert not ok
        assert any("entry_hash" in v or "tampered" in v for v in violations)


# ── Append and load ───────────────────────────────────────────────────────────

class TestAppendLoad:
    def test_append_and_load_single(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        snap = make_snap()
        append_snapshot(snap, path)
        loaded = load_chain(path)
        assert len(loaded) == 1
        assert loaded[0].intent_id == snap.intent_id
        assert loaded[0].entry_hash == snap.entry_hash

    def test_append_multiple_and_verify(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        s1 = make_snap(new_state="SUBMITTED")
        append_snapshot(s1, path)
        s2 = make_snap(intent_id="i-001", prev_state="SUBMITTED",
                       new_state="ACKED", prev_hash=s1.entry_hash)
        append_snapshot(s2, path)
        loaded = load_chain(path)
        ok, violations = verify_chain(loaded)
        assert ok, violations

    def test_load_missing_file_returns_empty(self, tmp_path):
        loaded = load_chain(tmp_path / "missing.jsonl")
        assert loaded == []

    def test_get_tail_hash_empty(self, tmp_path):
        h = get_tail_hash(tmp_path / "missing.jsonl")
        assert h == GENESIS_HASH

    def test_get_tail_hash_returns_last(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        s1 = make_snap(new_state="SUBMITTED")
        s2 = make_snap(prev_state="SUBMITTED", new_state="ACKED",
                       prev_hash=s1.entry_hash)
        append_snapshot(s1, path)
        append_snapshot(s2, path)
        assert get_tail_hash(path) == s2.entry_hash


# ── Replay ────────────────────────────────────────────────────────────────────

class TestReplayChain:
    def test_replay_single_transition(self):
        snap = make_snap(prev_state="CREATED", new_state="SUBMITTED")
        records = replay_chain([snap])
        assert "i-001" in records
        assert records["i-001"].state == "SUBMITTED"

    def test_replay_full_lifecycle(self):
        s1 = make_snap(new_state="SUBMITTED")
        s2 = make_snap(intent_id="i-001", prev_state="SUBMITTED",
                       new_state="ACKED", prev_hash=s1.entry_hash)
        s3 = make_snap(intent_id="i-001", prev_state="ACKED",
                       new_state="FILLED", prev_hash=s2.entry_hash,
                       filled_qty=5, fill_price=1000.0)
        records = replay_chain([s1, s2, s3])
        assert records["i-001"].state == "FILLED"
        assert records["i-001"].filled_qty == 5

    def test_replay_preserves_unknown_state(self):
        s1 = make_snap(new_state="SUBMITTED")
        s2 = make_snap(intent_id="i-001", prev_state="SUBMITTED",
                       new_state="UNKNOWN", prev_hash=s1.entry_hash)
        records = replay_chain([s1, s2])
        # UNKNOWN is preserved — never auto-resolved
        assert records["i-001"].state == "UNKNOWN"

    def test_replay_deterministic(self):
        snaps = [make_snap(new_state="SUBMITTED")]
        r1 = replay_chain(snaps)
        r2 = replay_chain(snaps)
        assert r1["i-001"].state == r2["i-001"].state

    def test_replay_multiple_intents(self):
        s1 = make_snap(intent_id="i-001", new_state="SUBMITTED")
        s2 = make_snap(intent_id="i-002", new_state="SUBMITTED",
                       prev_hash=s1.entry_hash)
        records = replay_chain([s1, s2])
        assert "i-001" in records
        assert "i-002" in records

    def test_replay_idempotent_same_state(self):
        # If snapshot says new_state == current state, skip (idempotent)
        s1 = make_snap(new_state="SUBMITTED")
        s_dup = make_snap(new_state="SUBMITTED", prev_hash=s1.entry_hash)
        records = replay_chain([s1, s_dup])
        assert records["i-001"].state == "SUBMITTED"
