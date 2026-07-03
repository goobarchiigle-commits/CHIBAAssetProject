"""
Tests for deterministic replay — same snapshot sequence always produces same state.
"""
import pytest
from src.truth.intent_snapshot import (
    make_snapshot, append_snapshot, load_chain,
    replay_chain, verify_chain, GENESIS_HASH,
)
from src.truth.execution_truth import TERMINAL_STATES, UNKNOWN_STATES


def build_chain(*transitions, base_id="i-001"):
    """
    Build an in-memory snapshot sequence.
    transitions: list of (prev_state, new_state[, {extra kwargs}])
    """
    snaps = []
    prev_hash = GENESIS_HASH
    for item in transitions:
        if len(item) == 3:
            prev_state, new_state, extra = item
        else:
            prev_state, new_state = item
            extra = {}
        snap = make_snapshot(
            intent_id=base_id,
            symbol="1234",
            side="BUY",
            requested_qty=10,
            prev_state=prev_state,
            new_state=new_state,
            reason="test",
            trading_day="2026-05-16",
            strategy="turtle",
            prev_hash=prev_hash,
            **extra,
        )
        snaps.append(snap)
        prev_hash = snap.entry_hash
    return snaps


# ── Determinism ───────────────────────────────────────────────────────────────

class TestReplayDeterminism:
    def test_same_input_same_output(self):
        snaps = build_chain(
            ("CREATED", "SUBMITTED"),
            ("SUBMITTED", "ACKED"),
            ("ACKED", "FILLED"),
        )
        r1 = replay_chain(snaps)
        r2 = replay_chain(snaps)
        assert r1["i-001"].state == r2["i-001"].state
        assert r1["i-001"].filled_qty == r2["i-001"].filled_qty

    def test_order_matters_different_sequence_different_result(self):
        # CREATED→SUBMITTED→ACKED vs CREATED→SUBMITTED→UNKNOWN
        chain_a = build_chain(("CREATED", "SUBMITTED"), ("SUBMITTED", "ACKED"))
        chain_b = build_chain(("CREATED", "SUBMITTED"), ("SUBMITTED", "UNKNOWN"))
        ra = replay_chain(chain_a)
        rb = replay_chain(chain_b)
        assert ra["i-001"].state != rb["i-001"].state

    def test_repeated_replay_stable(self):
        snaps = build_chain(
            ("CREATED", "SUBMITTED"),
            ("SUBMITTED", "ACKED"),
        )
        results = [replay_chain(snaps)["i-001"].state for _ in range(5)]
        assert all(r == "ACKED" for r in results)


# ── UNKNOWN preservation ──────────────────────────────────────────────────────

class TestUnknownPreservation:
    def test_unknown_not_auto_resolved(self):
        snaps = build_chain(
            ("CREATED", "SUBMITTED"),
            ("SUBMITTED", "UNKNOWN"),
        )
        records = replay_chain(snaps)
        assert records["i-001"].state == "UNKNOWN"
        assert "i-001" in records
        # should not be resolved automatically
        assert records["i-001"].state not in TERMINAL_STATES

    def test_unknown_resolves_only_via_explicit_snapshot(self):
        snaps = build_chain(
            ("CREATED", "SUBMITTED"),
            ("SUBMITTED", "UNKNOWN"),
            ("UNKNOWN", "RESOLVED_FILLED"),
        )
        records = replay_chain(snaps)
        assert records["i-001"].state == "RESOLVED_FILLED"

    def test_unknown_not_in_terminal_mid_chain(self):
        snaps = build_chain(
            ("CREATED", "SUBMITTED"),
            ("SUBMITTED", "UNKNOWN"),
        )
        records = replay_chain(snaps)
        # UNKNOWN itself is not terminal — record is still "active" (uncertain)
        assert not records["i-001"].is_terminal
        assert records["i-001"].is_unknown


# ── Multi-intent replay ───────────────────────────────────────────────────────

class TestMultiIntentReplay:
    def test_two_intents_independent(self):
        snaps_a = build_chain(
            ("CREATED", "SUBMITTED"),
            ("SUBMITTED", "ACKED"),
            ("ACKED", "FILLED"),
            base_id="i-001",
        )
        snaps_b = build_chain(
            ("CREATED", "SUBMITTED"),
            ("SUBMITTED", "UNKNOWN"),
            base_id="i-002",
        )
        # Merge: one FILLED, one UNKNOWN
        all_snaps = snaps_a + snaps_b
        records = replay_chain(all_snaps)
        assert records["i-001"].state == "FILLED"
        assert records["i-002"].state == "UNKNOWN"

    def test_interleaved_intents(self):
        s1a = make_snapshot("i-001", "1234", "BUY", 5, "CREATED", "SUBMITTED",
                            "test", "2026-05-16", "t", prev_hash=GENESIS_HASH)
        prev_hash_1a = s1a.entry_hash

        s1b = make_snapshot("i-002", "5678", "BUY", 3, "CREATED", "SUBMITTED",
                            "test", "2026-05-16", "t", prev_hash=prev_hash_1a)
        prev_hash_1b = s1b.entry_hash

        s2a = make_snapshot("i-001", "1234", "BUY", 5, "SUBMITTED", "ACKED",
                            "test", "2026-05-16", "t", prev_hash=prev_hash_1b)

        records = replay_chain([s1a, s1b, s2a])
        assert records["i-001"].state == "ACKED"
        assert records["i-002"].state == "SUBMITTED"


# ── Idempotent replay ─────────────────────────────────────────────────────────

class TestIdempotentReplay:
    def test_duplicate_snapshot_idempotent(self):
        s1 = make_snapshot("i-001", "1234", "BUY", 5, "CREATED", "SUBMITTED",
                           "test", "2026-05-16", "t", prev_hash=GENESIS_HASH)
        s2 = make_snapshot("i-001", "1234", "BUY", 5, "SUBMITTED", "SUBMITTED",
                           "dup", "2026-05-16", "t", prev_hash=s1.entry_hash)
        # s2 has new_state == SUBMITTED == current state → should be skipped
        records = replay_chain([s1, s2])
        assert records["i-001"].state == "SUBMITTED"


# ── File roundtrip ────────────────────────────────────────────────────────────

def test_file_roundtrip_produces_same_state(tmp_path):
    path = tmp_path / "snaps.jsonl"
    snaps = build_chain(
        ("CREATED", "SUBMITTED"),
        ("SUBMITTED", "ACKED"),
        ("ACKED", "PARTIAL_FILL", {"filled_qty": 3, "fill_price": 1000.0}),
    )
    for s in snaps:
        append_snapshot(s, path)
    loaded = load_chain(path)
    ok, violations = verify_chain(loaded)
    assert ok, violations

    records = replay_chain(loaded)
    assert records["i-001"].state == "PARTIAL_FILL"
