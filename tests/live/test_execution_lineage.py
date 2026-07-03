"""
tests/live/test_execution_lineage.py

Tests for the execution lineage and epoch governance layer:
  - src/live/execution_epoch.py
  - src/live/deployment_lineage.py
  - src/live/pending_order_authority.py
  - src/live/quarantine_governance.py
  - src/live/deployment_authority_manifest.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

import pytest

# ── execution_epoch ──────────────────────────────────────────────────────────
from src.live.execution_epoch import (
    AUTHORITY_LEVEL_BLOCKED,
    AUTHORITY_LEVEL_FULL,
    AUTHORITY_LEVEL_PARTIAL,
    AUTHORITY_LEVEL_QUARANTINED,
    AuthorityEpochChain,
    DeploymentEpoch,
    ReconciliationEpoch,
    append_authority_chain,
    append_deployment_epoch,
    append_reconciliation_epoch,
    build_authority_chain,
    load_authority_chains,
    load_deployment_epochs,
    load_reconciliation_epochs,
)

# ── deployment_lineage ────────────────────────────────────────────────────────
from src.live.deployment_lineage import (
    GENESIS_HASH,
    ORDERED_STEPS,
    STEP_AUTHORITY_DECISION,
    STEP_BROKER_ACKNOWLEDGMENT,
    STEP_BROKER_SNAPSHOT,
    STEP_DEPLOYMENT_DECISION,
    STEP_INTEGRITY_VALIDATION,
    STEP_RECONCILIATION,
    STEP_REPLAY_ARTIFACT,
    DeploymentLineage,
    LineageEntry,
    append_lineage_step,
    build_lineage,
    create_lineage_entry,
    load_lineage_entries,
    persist_lineage,
    persist_lineage_entry,
    reconstruct_lineage,
)

# ── pending_order_authority ───────────────────────────────────────────────────
from src.live.pending_order_authority import (
    AUTH_ACKNOWLEDGED,
    AUTH_CANCELLED,
    AUTH_FILLED,
    AUTH_PARTIAL_FILL,
    AUTH_REJECTED,
    AUTH_STALE,
    AUTH_SUBMITTED,
    AUTH_UNKNOWN,
    AUTHORITY_TTL_SEC,
    DIVERGENCE_DUPLICATE_AUTHORITY,
    DIVERGENCE_STALE_AUTHORITY,
    AuthorityDivergenceReport,
    AuthorityTransitionEvent,
    PendingOrderAuthority,
    append_authority,
    append_divergence_report,
    append_transition_event,
    check_stale_authority,
    detect_authority_divergences,
    load_authorities,
    load_transition_events,
    transition_authority,
)

# ── quarantine_governance ─────────────────────────────────────────────────────
from src.live.quarantine_governance import (
    QSTATE_QUARANTINED,
    QUARANTINE_AUTHORITY_FAILURE,
    QUARANTINE_EPOCH_INCONSISTENCY,
    QUARANTINE_INTEGRITY_VIOLATION,
    QUARANTINE_RECONCILIATION_MISMATCH,
    QUARANTINE_REPLAY_DIVERGENCE,
    RECOVERY_PREREQ_EPOCH_NORMALIZED,
    RECOVERY_PREREQ_INTEGRITY_VALID,
    RECOVERY_PREREQ_RECONCILIATION_OK,
    QuarantineEntry,
    QuarantineExitAttempt,
    QuarantineStatus,
    append_exit_attempt,
    append_quarantine_entry,
    evaluate_quarantine_exit,
    get_quarantine_status,
    load_exit_attempts,
    load_quarantine_entries,
)

# ── deployment_authority_manifest ─────────────────────────────────────────────
from src.live.deployment_authority_manifest import (
    AUTHORITY_LEVEL_BLOCKED as DAM_BLOCKED,
    AUTHORITY_LEVEL_FULL as DAM_FULL,
    AUTHORITY_LEVEL_PARTIAL as DAM_PARTIAL,
    AUTHORITY_LEVEL_QUARANTINED as DAM_QUARANTINED,
    MANIFEST_AUTHORIZED,
    MANIFEST_BLOCKED,
    MANIFEST_QUARANTINED as MANIFEST_QUARANTINED_STATUS,
    DeploymentAuthorityManifest,
    append_manifest,
    build_manifest,
    get_latest_manifest,
    load_manifests,
    require_manifest,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

FIXED_TS = "2026-05-17T09:00:00+00:00"
FIXED_HASH = "a" * 64
RUN_ID = "run-test-001"


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


def _make_reconciliation_epoch(
    run_id: str = RUN_ID,
    broker_hash: str = FIXED_HASH,
    ok: bool = True,
    blocking: int = 0,
    ts: str = FIXED_TS,
) -> ReconciliationEpoch:
    return ReconciliationEpoch.create(
        run_id=run_id,
        broker_snapshot_hash=broker_hash,
        reconciliation_ok=ok,
        blocking_mismatches=blocking,
        created_at=ts,
    )


def _make_deployment_epoch(
    r_epoch: ReconciliationEpoch,
    allowed: bool = True,
    level: str = AUTHORITY_LEVEL_FULL,
    ts: str = FIXED_TS,
) -> DeploymentEpoch:
    return DeploymentEpoch.create(
        run_id=r_epoch.run_id,
        reconciliation_epoch_id=r_epoch.epoch_id,
        broker_snapshot_hash=r_epoch.broker_snapshot_hash,
        deployment_allowed=allowed,
        authority_level=level,
        created_at=ts,
    )


def _make_authority(
    order_id: str = "order-001",
    run_id: str = RUN_ID,
    symbol: str = "1234",
    side: str = "BUY",
    epoch_id: str = "epoch-001",
    submitted_at: str = FIXED_TS,
) -> PendingOrderAuthority:
    return PendingOrderAuthority.create(
        order_id=order_id,
        run_id=run_id,
        symbol=symbol,
        side=side,
        requested_qty=100,
        authority_epoch_id=epoch_id,
        submitted_at=submitted_at,
    )


def _make_quarantine_entry(
    run_id: str = RUN_ID,
    reason: str = QUARANTINE_INTEGRITY_VIOLATION,
    epoch_id: str = "epoch-001",
    ts: str = FIXED_TS,
) -> QuarantineEntry:
    return QuarantineEntry.create(
        run_id=run_id,
        entry_reason=reason,
        entry_epoch_id=epoch_id,
        broker_snapshot_hash=FIXED_HASH,
        reconciliation_epoch_id="rec-epoch-001",
        integrity_violation_detail="checksum mismatch",
        created_at=ts,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. EPOCH DETERMINISM
# ══════════════════════════════════════════════════════════════════════════════

class TestEpochDeterminism:
    def test_reconciliation_epoch_same_inputs_same_id(self):
        e1 = _make_reconciliation_epoch()
        e2 = _make_reconciliation_epoch()
        assert e1.epoch_id == e2.epoch_id

    def test_reconciliation_epoch_different_hash_different_id(self):
        e1 = _make_reconciliation_epoch(broker_hash="a" * 64)
        e2 = _make_reconciliation_epoch(broker_hash="b" * 64)
        assert e1.epoch_id != e2.epoch_id

    def test_reconciliation_epoch_different_ok_flag_different_id(self):
        e1 = _make_reconciliation_epoch(ok=True)
        e2 = _make_reconciliation_epoch(ok=False)
        assert e1.epoch_id != e2.epoch_id

    def test_reconciliation_epoch_validates(self):
        e = _make_reconciliation_epoch()
        assert e.validate() is True

    def test_reconciliation_epoch_tampered_id_fails_validate(self):
        e = _make_reconciliation_epoch()
        import dataclasses
        bad = dataclasses.replace(e, epoch_id="tampered" + "0" * 57)
        assert bad.validate() is False

    def test_deployment_epoch_same_inputs_same_id(self):
        r = _make_reconciliation_epoch()
        d1 = _make_deployment_epoch(r)
        d2 = _make_deployment_epoch(r)
        assert d1.epoch_id == d2.epoch_id

    def test_deployment_epoch_different_level_different_id(self):
        r = _make_reconciliation_epoch()
        d1 = _make_deployment_epoch(r, level=AUTHORITY_LEVEL_FULL)
        d2 = _make_deployment_epoch(r, level=AUTHORITY_LEVEL_BLOCKED)
        assert d1.epoch_id != d2.epoch_id

    def test_deployment_epoch_validates(self):
        r = _make_reconciliation_epoch()
        d = _make_deployment_epoch(r)
        assert d.validate() is True

    def test_deployment_epoch_consistent_with_reconciliation(self):
        r = _make_reconciliation_epoch()
        d = _make_deployment_epoch(r)
        assert d.is_epoch_consistent_with(r) is True

    def test_deployment_epoch_inconsistent_on_wrong_reconciliation(self):
        r1 = _make_reconciliation_epoch(ts=FIXED_TS)
        r2 = _make_reconciliation_epoch(broker_hash="b" * 64, ts=FIXED_TS)
        d = _make_deployment_epoch(r1)
        assert d.is_epoch_consistent_with(r2) is False

    def test_authority_chain_same_inputs_same_hash(self):
        r = _make_reconciliation_epoch()
        d = _make_deployment_epoch(r)
        c1 = build_authority_chain(r, d)
        c2 = build_authority_chain(r, d)
        assert c1.chain_hash == c2.chain_hash

    def test_authority_chain_validates(self):
        r = _make_reconciliation_epoch()
        d = _make_deployment_epoch(r)
        c = build_authority_chain(r, d)
        assert c.validate() is True

    def test_authority_chain_is_consistent_with_epochs(self):
        r = _make_reconciliation_epoch()
        d = _make_deployment_epoch(r)
        c = build_authority_chain(r, d)
        assert c.is_consistent_with(r, d) is True

    def test_authority_chain_inconsistent_on_epoch_mismatch(self):
        r1 = _make_reconciliation_epoch(ts=FIXED_TS)
        r2 = _make_reconciliation_epoch(broker_hash="b" * 64, ts=FIXED_TS)
        d = _make_deployment_epoch(r1)
        c = build_authority_chain(r1, d)
        assert c.is_consistent_with(r2, d) is False

    def test_build_authority_chain_raises_on_inconsistent_pair(self):
        r1 = _make_reconciliation_epoch(ts=FIXED_TS)
        r2 = _make_reconciliation_epoch(broker_hash="b" * 64, ts=FIXED_TS)
        d = _make_deployment_epoch(r1)
        import dataclasses
        # Swap broker_snapshot_hash to mismatch
        bad_d = dataclasses.replace(d, broker_snapshot_hash="b" * 64)
        with pytest.raises(ValueError, match="Epoch inconsistency"):
            build_authority_chain(r1, bad_d)


# ══════════════════════════════════════════════════════════════════════════════
# 2. EPOCH PERSISTENCE & REPLAY
# ══════════════════════════════════════════════════════════════════════════════

class TestEpochPersistence:
    def test_reconciliation_epoch_roundtrip(self, tmp: Path):
        path = tmp / "rec_epochs.jsonl"
        e = _make_reconciliation_epoch()
        append_reconciliation_epoch(e, path)
        loaded = load_reconciliation_epochs(path)
        assert len(loaded) == 1
        assert loaded[0].epoch_id == e.epoch_id
        assert loaded[0].broker_snapshot_hash == e.broker_snapshot_hash
        assert loaded[0].validate() is True

    def test_deployment_epoch_roundtrip(self, tmp: Path):
        path = tmp / "dep_epochs.jsonl"
        r = _make_reconciliation_epoch()
        d = _make_deployment_epoch(r)
        append_deployment_epoch(d, path)
        loaded = load_deployment_epochs(path)
        assert len(loaded) == 1
        assert loaded[0].epoch_id == d.epoch_id
        assert loaded[0].validate() is True

    def test_authority_chain_roundtrip(self, tmp: Path):
        path = tmp / "chains.jsonl"
        r = _make_reconciliation_epoch()
        d = _make_deployment_epoch(r)
        c = build_authority_chain(r, d)
        append_authority_chain(c, path)
        loaded = load_authority_chains(path)
        assert len(loaded) == 1
        assert loaded[0].chain_hash == c.chain_hash
        assert loaded[0].validate() is True

    def test_multiple_epochs_appended_and_loaded(self, tmp: Path):
        path = tmp / "rec_epochs.jsonl"
        import dataclasses
        for i in range(5):
            ts = f"2026-05-17T09:0{i}:00+00:00"
            e = _make_reconciliation_epoch(ts=ts)
            append_reconciliation_epoch(e, path)
        loaded = load_reconciliation_epochs(path)
        assert len(loaded) == 5
        ids = [e.epoch_id for e in loaded]
        assert len(set(ids)) == 5

    def test_load_from_missing_file_returns_empty(self, tmp: Path):
        assert load_reconciliation_epochs(tmp / "missing.jsonl") == []
        assert load_deployment_epochs(tmp / "missing.jsonl") == []
        assert load_authority_chains(tmp / "missing.jsonl") == []

    def test_corrupted_line_skipped(self, tmp: Path):
        path = tmp / "rec_epochs.jsonl"
        e = _make_reconciliation_epoch()
        append_reconciliation_epoch(e, path)
        with path.open("a") as f:
            f.write("not json\n")
        loaded = load_reconciliation_epochs(path)
        assert len(loaded) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. LINEAGE CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

class TestLineageConsistency:
    def test_empty_lineage_tail_is_genesis(self):
        lin = build_lineage(RUN_ID)
        assert lin.tail_hash() == GENESIS_HASH

    def test_first_entry_has_genesis_prev_hash(self):
        lin = build_lineage(RUN_ID)
        entry = append_lineage_step(lin, STEP_BROKER_SNAPSHOT, {"snap_hash": "abc"}, timestamp=FIXED_TS)
        assert entry.prev_hash == GENESIS_HASH

    def test_second_entry_links_to_first(self):
        lin = build_lineage(RUN_ID)
        e1 = append_lineage_step(lin, STEP_BROKER_SNAPSHOT, {"snap": "a"}, timestamp=FIXED_TS)
        e2 = append_lineage_step(lin, STEP_RECONCILIATION, {"ok": True}, timestamp=FIXED_TS)
        assert e2.prev_hash == e1.entry_hash

    def test_full_chain_verifies(self):
        lin = build_lineage(RUN_ID)
        contents = [
            {"snap_hash": FIXED_HASH},
            {"reconciliation_ok": True, "blocking": 0},
            {"integrity_ok": True, "violations": 0},
            {"authority_level": "full"},
            {"deployment_allowed": True},
            {"broker_ack": "ok"},
            {"replay_hash": "xyz"},
        ]
        for step, content in zip(ORDERED_STEPS, contents):
            append_lineage_step(lin, step, content, timestamp=FIXED_TS)
        assert lin.is_complete() is True
        assert lin.verify_chain() is True

    def test_partial_chain_not_complete(self):
        lin = build_lineage(RUN_ID)
        append_lineage_step(lin, STEP_BROKER_SNAPSHOT, {}, timestamp=FIXED_TS)
        assert lin.is_complete() is False

    def test_chain_broken_by_hash_tamper_fails_verify(self):
        lin = build_lineage(RUN_ID)
        e = append_lineage_step(lin, STEP_BROKER_SNAPSHOT, {"a": 1}, timestamp=FIXED_TS)
        import dataclasses
        tampered = dataclasses.replace(e, entry_hash="bad" + "0" * 61)
        lin.entries[0] = tampered
        assert lin.verify_chain() is False

    def test_chain_broken_by_content_tamper_fails_verify(self):
        lin = build_lineage(RUN_ID)
        append_lineage_step(lin, STEP_BROKER_SNAPSHOT, {"snap": "a"}, timestamp=FIXED_TS)
        lin.entries[0].content["snap"] = "TAMPERED"
        assert lin.verify_chain() is False

    def test_step_ordering_enforced(self):
        lin = build_lineage(RUN_ID)
        with pytest.raises(ValueError, match="Step ordering violation"):
            append_lineage_step(lin, STEP_RECONCILIATION, {}, timestamp=FIXED_TS)

    def test_unknown_step_raises(self):
        with pytest.raises(ValueError, match="Unknown lineage step"):
            create_lineage_entry(RUN_ID, "invalid_step", {}, GENESIS_HASH, FIXED_TS)

    def test_step_index_correct(self):
        lin = build_lineage(RUN_ID)
        for i, step in enumerate(ORDERED_STEPS):
            e = append_lineage_step(lin, step, {}, timestamp=FIXED_TS)
            assert e.step_index == i

    def test_get_step_returns_correct_entry(self):
        lin = build_lineage(RUN_ID)
        append_lineage_step(lin, STEP_BROKER_SNAPSHOT, {"snap": "x"}, timestamp=FIXED_TS)
        append_lineage_step(lin, STEP_RECONCILIATION, {"ok": True}, timestamp=FIXED_TS)
        step = lin.get_step(STEP_RECONCILIATION)
        assert step is not None
        assert step.content["ok"] is True

    def test_get_step_missing_returns_none(self):
        lin = build_lineage(RUN_ID)
        assert lin.get_step(STEP_RECONCILIATION) is None

    def test_deterministic_same_inputs_same_entry_hash(self):
        e1 = create_lineage_entry(RUN_ID, STEP_BROKER_SNAPSHOT, {"snap": "abc"}, GENESIS_HASH, FIXED_TS)
        e2 = create_lineage_entry(RUN_ID, STEP_BROKER_SNAPSHOT, {"snap": "abc"}, GENESIS_HASH, FIXED_TS)
        assert e1.entry_hash == e2.entry_hash
        assert e1.content_hash == e2.content_hash

    def test_different_content_different_entry_hash(self):
        e1 = create_lineage_entry(RUN_ID, STEP_BROKER_SNAPSHOT, {"snap": "abc"}, GENESIS_HASH, FIXED_TS)
        e2 = create_lineage_entry(RUN_ID, STEP_BROKER_SNAPSHOT, {"snap": "xyz"}, GENESIS_HASH, FIXED_TS)
        assert e1.entry_hash != e2.entry_hash


# ══════════════════════════════════════════════════════════════════════════════
# 4. LINEAGE PERSISTENCE & REPLAY
# ══════════════════════════════════════════════════════════════════════════════

class TestLineagePersistence:
    def test_persist_and_load_single_entry(self, tmp: Path):
        path = tmp / "lineage.jsonl"
        lin = build_lineage(RUN_ID)
        e = append_lineage_step(lin, STEP_BROKER_SNAPSHOT, {"snap": FIXED_HASH}, timestamp=FIXED_TS)
        persist_lineage_entry(e, path)
        loaded = load_lineage_entries(path)
        assert len(loaded) == 1
        assert loaded[0].entry_hash == e.entry_hash
        assert loaded[0].step == STEP_BROKER_SNAPSHOT

    def test_persist_full_lineage_and_reconstruct(self, tmp: Path):
        path = tmp / "lineage.jsonl"
        lin = build_lineage(RUN_ID)
        for step in ORDERED_STEPS:
            append_lineage_step(lin, step, {"step": step}, timestamp=FIXED_TS)
        persist_lineage(lin, path)

        reconstructed = reconstruct_lineage(path, RUN_ID)
        assert reconstructed.is_complete() is True
        assert reconstructed.verify_chain() is True
        assert len(reconstructed.entries) == len(ORDERED_STEPS)

    def test_reconstruct_filters_by_run_id(self, tmp: Path):
        path = tmp / "lineage.jsonl"
        for run_id in ("run-A", "run-B"):
            lin = build_lineage(run_id)
            append_lineage_step(lin, STEP_BROKER_SNAPSHOT, {}, timestamp=FIXED_TS)
            persist_lineage(lin, path)

        rec_a = reconstruct_lineage(path, "run-A")
        rec_b = reconstruct_lineage(path, "run-B")
        assert all(e.run_id == "run-A" for e in rec_a.entries)
        assert all(e.run_id == "run-B" for e in rec_b.entries)

    def test_corrupted_line_skipped_in_load(self, tmp: Path):
        path = tmp / "lineage.jsonl"
        lin = build_lineage(RUN_ID)
        e = append_lineage_step(lin, STEP_BROKER_SNAPSHOT, {}, timestamp=FIXED_TS)
        persist_lineage_entry(e, path)
        with path.open("a") as f:
            f.write("{bad json\n")
        loaded = load_lineage_entries(path)
        assert len(loaded) == 1

    def test_replay_consistency_same_data_same_hashes(self, tmp: Path):
        path = tmp / "lineage.jsonl"
        lin = build_lineage(RUN_ID)
        for step in ORDERED_STEPS[:3]:
            append_lineage_step(lin, step, {"k": step}, timestamp=FIXED_TS)
        persist_lineage(lin, path)

        reconstructed = reconstruct_lineage(path, RUN_ID)
        for orig, loaded in zip(lin.entries, reconstructed.entries):
            assert orig.entry_hash == loaded.entry_hash
            assert orig.content_hash == loaded.content_hash


# ══════════════════════════════════════════════════════════════════════════════
# 5. PENDING-ORDER LIFECYCLE & DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

class TestPendingOrderAuthority:
    def test_create_initial_state_is_submitted(self):
        auth = _make_authority()
        assert auth.authority_state == AUTH_SUBMITTED

    def test_authority_id_deterministic(self):
        a1 = _make_authority()
        a2 = _make_authority()
        assert a1.authority_id == a2.authority_id

    def test_valid_transition_submitted_to_acknowledged(self):
        auth = _make_authority()
        updated, event = transition_authority(auth, AUTH_ACKNOWLEDGED, "broker ack", FIXED_TS)
        assert updated.authority_state == AUTH_ACKNOWLEDGED
        assert event.prev_state == AUTH_SUBMITTED
        assert event.new_state == AUTH_ACKNOWLEDGED

    def test_valid_full_lifecycle_to_filled(self):
        auth = _make_authority()
        auth, _ = transition_authority(auth, AUTH_ACKNOWLEDGED, "ack", FIXED_TS)
        auth, _ = transition_authority(auth, AUTH_PARTIAL_FILL, "partial", FIXED_TS)
        auth, _ = transition_authority(auth, AUTH_FILLED, "filled", FIXED_TS)
        assert auth.authority_state == AUTH_FILLED

    def test_valid_transition_submitted_to_cancelled(self):
        auth = _make_authority()
        updated, _ = transition_authority(auth, AUTH_CANCELLED, "user cancel", FIXED_TS)
        assert updated.authority_state == AUTH_CANCELLED

    def test_valid_transition_submitted_to_stale(self):
        auth = _make_authority()
        updated, _ = transition_authority(auth, AUTH_STALE, "ttl expired", FIXED_TS)
        assert updated.authority_state == AUTH_STALE

    def test_valid_transition_submitted_to_unknown(self):
        auth = _make_authority()
        updated, _ = transition_authority(auth, AUTH_UNKNOWN, "broker unresponsive", FIXED_TS)
        assert updated.authority_state == AUTH_UNKNOWN

    def test_invalid_transition_filled_to_submitted_raises(self):
        auth = _make_authority()
        auth, _ = transition_authority(auth, AUTH_ACKNOWLEDGED, "ack", FIXED_TS)
        auth, _ = transition_authority(auth, AUTH_FILLED, "filled", FIXED_TS)
        with pytest.raises(ValueError, match="Invalid authority transition"):
            transition_authority(auth, AUTH_SUBMITTED, "bad", FIXED_TS)

    def test_invalid_transition_cancelled_to_anything_raises(self):
        auth = _make_authority()
        auth, _ = transition_authority(auth, AUTH_CANCELLED, "cancel", FIXED_TS)
        with pytest.raises(ValueError, match="Invalid authority transition"):
            transition_authority(auth, AUTH_ACKNOWLEDGED, "bad", FIXED_TS)

    def test_invalid_transition_filled_to_partial_fill_raises(self):
        auth = _make_authority()
        auth, _ = transition_authority(auth, AUTH_ACKNOWLEDGED, "ack", FIXED_TS)
        auth, _ = transition_authority(auth, AUTH_FILLED, "fill", FIXED_TS)
        with pytest.raises(ValueError):
            transition_authority(auth, AUTH_PARTIAL_FILL, "partial after fill", FIXED_TS)

    def test_event_id_deterministic(self):
        auth = _make_authority()
        _, ev1 = transition_authority(auth, AUTH_ACKNOWLEDGED, "ack", FIXED_TS)
        _, ev2 = transition_authority(auth, AUTH_ACKNOWLEDGED, "ack", FIXED_TS)
        assert ev1.event_id == ev2.event_id


class TestStaleAuthority:
    def test_non_terminal_beyond_ttl_is_stale(self):
        old_ts = "2020-01-01T00:00:00+00:00"
        auth = _make_authority(submitted_at=old_ts)
        assert check_stale_authority(auth, now_sec=time.time()) is True

    def test_terminal_state_never_stale(self):
        old_ts = "2020-01-01T00:00:00+00:00"
        auth = _make_authority(submitted_at=old_ts)
        auth, _ = transition_authority(auth, AUTH_CANCELLED, "cancel", old_ts)
        assert check_stale_authority(auth, now_sec=time.time()) is False

    def test_fresh_non_terminal_not_stale(self):
        now = datetime.now(timezone.utc).isoformat()
        auth = _make_authority(submitted_at=now)
        assert check_stale_authority(auth, now_sec=time.time(), ttl_sec=AUTHORITY_TTL_SEC) is False

    def test_detect_stale_in_divergence_report(self):
        old_ts = "2020-01-01T00:00:00+00:00"
        auth = _make_authority(submitted_at=old_ts)
        report = detect_authority_divergences([auth], now_sec=time.time())
        assert auth.order_id in report.stale_authorities
        assert report.blocks_deployment is True

    def test_detect_duplicate_authority(self):
        now = datetime.now(timezone.utc).isoformat()
        a1 = _make_authority(order_id="order-dup", submitted_at=now)
        a2 = _make_authority(order_id="order-dup", submitted_at=now)
        report = detect_authority_divergences([a1, a2], now_sec=time.time())
        assert "order-dup" in report.duplicate_authorities
        assert report.blocks_deployment is True

    def test_no_divergences_does_not_block(self):
        now = datetime.now(timezone.utc).isoformat()
        auth = _make_authority(submitted_at=now)
        auth, _ = transition_authority(auth, AUTH_ACKNOWLEDGED, "ack", now)
        auth, _ = transition_authority(auth, AUTH_FILLED, "filled", now)
        report = detect_authority_divergences([auth], now_sec=time.time())
        assert report.blocks_deployment is False
        assert report.total_divergences == 0

    def test_unresolved_pending_not_stale(self):
        now = datetime.now(timezone.utc).isoformat()
        auth = _make_authority(submitted_at=now)
        auth, _ = transition_authority(auth, AUTH_ACKNOWLEDGED, "ack", now)
        report = detect_authority_divergences([auth], now_sec=time.time(), ttl_sec=AUTHORITY_TTL_SEC)
        assert auth.order_id in report.unresolved_pending
        assert report.blocks_deployment is False  # unresolved alone doesn't block

    def test_divergence_report_run_id(self):
        now = datetime.now(timezone.utc).isoformat()
        auth = _make_authority(run_id="run-xyz", submitted_at=now)
        report = detect_authority_divergences([auth], now_sec=time.time())
        assert report.run_id == "run-xyz"


class TestAuthorityPersistence:
    def test_authority_roundtrip(self, tmp: Path):
        path = tmp / "authorities.jsonl"
        auth = _make_authority()
        append_authority(auth, path)
        loaded = load_authorities(path)
        assert len(loaded) == 1
        assert loaded[0].authority_id == auth.authority_id
        assert loaded[0].order_id == auth.order_id

    def test_transition_event_roundtrip(self, tmp: Path):
        path = tmp / "events.jsonl"
        auth = _make_authority()
        _, event = transition_authority(auth, AUTH_ACKNOWLEDGED, "ack", FIXED_TS)
        append_transition_event(event, path)
        loaded = load_transition_events(path)
        assert len(loaded) == 1
        assert loaded[0].event_id == event.event_id
        assert loaded[0].new_state == AUTH_ACKNOWLEDGED

    def test_divergence_report_roundtrip(self, tmp: Path):
        path = tmp / "divergences.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        auth = _make_authority(submitted_at=now)
        report = detect_authority_divergences([auth], now_sec=time.time())
        append_divergence_report(report, path)
        # Just verify write doesn't raise; no load function needed
        assert path.exists()


# ══════════════════════════════════════════════════════════════════════════════
# 6. QUARANTINE ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

class TestQuarantineAttribution:
    def test_quarantine_id_deterministic(self):
        e1 = _make_quarantine_entry()
        e2 = _make_quarantine_entry()
        assert e1.quarantine_id == e2.quarantine_id

    def test_different_reason_different_id(self):
        e1 = _make_quarantine_entry(reason=QUARANTINE_INTEGRITY_VIOLATION)
        e2 = _make_quarantine_entry(reason=QUARANTINE_RECONCILIATION_MISMATCH)
        assert e1.quarantine_id != e2.quarantine_id

    def test_different_epoch_different_id(self):
        e1 = _make_quarantine_entry(epoch_id="epoch-A")
        e2 = _make_quarantine_entry(epoch_id="epoch-B")
        assert e1.quarantine_id != e2.quarantine_id

    def test_validate_id_passes(self):
        e = _make_quarantine_entry()
        assert e.validate_id() is True

    def test_validate_id_fails_on_tamper(self):
        e = _make_quarantine_entry()
        import dataclasses
        bad = dataclasses.replace(e, quarantine_id="tampered" * 8)
        assert bad.validate_id() is False

    def test_initial_state_is_quarantined(self):
        e = _make_quarantine_entry()
        assert e.quarantine_state == QSTATE_QUARANTINED

    def test_all_entry_reasons_accepted(self):
        reasons = [
            QUARANTINE_INTEGRITY_VIOLATION,
            QUARANTINE_RECONCILIATION_MISMATCH,
            QUARANTINE_AUTHORITY_FAILURE,
            QUARANTINE_REPLAY_DIVERGENCE,
            QUARANTINE_EPOCH_INCONSISTENCY,
        ]
        for i, reason in enumerate(reasons):
            ts = f"2026-05-17T09:0{i}:00+00:00"
            entry = _make_quarantine_entry(reason=reason, ts=ts)
            assert entry.entry_reason == reason

    def test_unmet_prerequisites_all_when_none_pass(self):
        e = _make_quarantine_entry()
        unmet = e.unmet_prerequisites(
            reconciliation_ok=False,
            epoch_normalized=False,
            integrity_valid=False,
        )
        assert RECOVERY_PREREQ_RECONCILIATION_OK in unmet
        assert RECOVERY_PREREQ_EPOCH_NORMALIZED in unmet
        assert RECOVERY_PREREQ_INTEGRITY_VALID in unmet

    def test_unmet_prerequisites_empty_when_all_pass(self):
        e = _make_quarantine_entry()
        unmet = e.unmet_prerequisites(
            reconciliation_ok=True,
            epoch_normalized=True,
            integrity_valid=True,
        )
        assert unmet == []


class TestQuarantineExit:
    def test_exit_granted_when_all_prerequisites_pass(self):
        entry = _make_quarantine_entry()
        attempt = evaluate_quarantine_exit(
            entry,
            reconciliation_ok=True,
            epoch_normalized=True,
            integrity_valid=True,
            new_epoch_id="new-epoch-001",
        )
        assert attempt.exit_granted is True

    def test_exit_denied_when_reconciliation_fails(self):
        entry = _make_quarantine_entry()
        attempt = evaluate_quarantine_exit(
            entry,
            reconciliation_ok=False,
            epoch_normalized=True,
            integrity_valid=True,
        )
        assert attempt.exit_granted is False
        assert "reconciliation" in attempt.exit_reason

    def test_exit_denied_when_epoch_not_normalized(self):
        entry = _make_quarantine_entry()
        attempt = evaluate_quarantine_exit(
            entry,
            reconciliation_ok=True,
            epoch_normalized=False,
            integrity_valid=True,
        )
        assert attempt.exit_granted is False
        assert "epoch" in attempt.exit_reason

    def test_exit_denied_when_integrity_fails(self):
        entry = _make_quarantine_entry()
        attempt = evaluate_quarantine_exit(
            entry,
            reconciliation_ok=True,
            epoch_normalized=True,
            integrity_valid=False,
        )
        assert attempt.exit_granted is False
        assert "integrity" in attempt.exit_reason

    def test_attempt_id_deterministic(self):
        entry = _make_quarantine_entry()
        a1 = QuarantineExitAttempt.create(
            quarantine_id=entry.quarantine_id,
            run_id=RUN_ID,
            reconciliation_ok=True,
            epoch_normalized=True,
            integrity_valid=True,
            attempted_at=FIXED_TS,
        )
        a2 = QuarantineExitAttempt.create(
            quarantine_id=entry.quarantine_id,
            run_id=RUN_ID,
            reconciliation_ok=True,
            epoch_normalized=True,
            integrity_valid=True,
            attempted_at=FIXED_TS,
        )
        assert a1.attempt_id == a2.attempt_id

    def test_exit_reason_auto_generated_on_success(self):
        entry = _make_quarantine_entry()
        attempt = evaluate_quarantine_exit(entry, True, True, True)
        assert "prerequisites" in attempt.exit_reason

    def test_exit_reason_auto_generated_on_all_failures(self):
        entry = _make_quarantine_entry()
        attempt = evaluate_quarantine_exit(entry, False, False, False)
        assert "reconciliation" in attempt.exit_reason
        assert "epoch" in attempt.exit_reason
        assert "integrity" in attempt.exit_reason


class TestQuarantineStatus:
    def test_no_entries_not_quarantined(self):
        status = get_quarantine_status(RUN_ID, [], [])
        assert status.is_quarantined is False
        assert status.active_quarantine_id is None

    def test_active_entry_no_exit_is_quarantined(self):
        entry = _make_quarantine_entry()
        status = get_quarantine_status(RUN_ID, [entry], [])
        assert status.is_quarantined is True
        assert status.active_quarantine_id == entry.quarantine_id
        assert status.entry_reason == QUARANTINE_INTEGRITY_VIOLATION

    def test_active_entry_with_failed_exit_still_quarantined(self):
        entry = _make_quarantine_entry()
        attempt = evaluate_quarantine_exit(entry, False, True, True)
        status = get_quarantine_status(RUN_ID, [entry], [attempt])
        assert status.is_quarantined is True

    def test_active_entry_with_successful_exit_not_quarantined(self):
        entry = _make_quarantine_entry()
        attempt = evaluate_quarantine_exit(entry, True, True, True)
        status = get_quarantine_status(RUN_ID, [entry], [attempt])
        assert status.is_quarantined is False

    def test_recovery_prerequisites_listed(self):
        entry = _make_quarantine_entry()
        status = get_quarantine_status(RUN_ID, [entry], [])
        assert RECOVERY_PREREQ_RECONCILIATION_OK in status.recovery_prerequisites
        assert RECOVERY_PREREQ_EPOCH_NORMALIZED in status.recovery_prerequisites
        assert RECOVERY_PREREQ_INTEGRITY_VALID in status.recovery_prerequisites

    def test_filters_by_run_id(self):
        e_a = _make_quarantine_entry(run_id="run-A", ts="2026-05-17T09:00:00+00:00")
        e_b = _make_quarantine_entry(run_id="run-B", ts="2026-05-17T09:00:01+00:00")
        s_a = get_quarantine_status("run-A", [e_a, e_b], [])
        s_b = get_quarantine_status("run-B", [e_a, e_b], [])
        assert s_a.is_quarantined is True
        assert s_a.active_quarantine_id == e_a.quarantine_id
        assert s_b.is_quarantined is True
        assert s_b.active_quarantine_id == e_b.quarantine_id


class TestQuarantinePersistence:
    def test_entry_roundtrip(self, tmp: Path):
        path = tmp / "quarantine.jsonl"
        entry = _make_quarantine_entry()
        append_quarantine_entry(entry, path)
        loaded = load_quarantine_entries(path)
        assert len(loaded) == 1
        assert loaded[0].quarantine_id == entry.quarantine_id
        assert loaded[0].validate_id() is True

    def test_exit_attempt_roundtrip(self, tmp: Path):
        path = tmp / "exits.jsonl"
        entry = _make_quarantine_entry()
        attempt = evaluate_quarantine_exit(entry, True, True, True)
        append_exit_attempt(attempt, path)
        loaded = load_exit_attempts(path)
        assert len(loaded) == 1
        assert loaded[0].attempt_id == attempt.attempt_id
        assert loaded[0].exit_granted is True

    def test_status_consistent_after_persistence(self, tmp: Path):
        entry_path = tmp / "quarantine.jsonl"
        exit_path = tmp / "exits.jsonl"
        entry = _make_quarantine_entry()
        append_quarantine_entry(entry, entry_path)

        attempt = evaluate_quarantine_exit(entry, True, True, True, new_epoch_id="new-ep")
        append_exit_attempt(attempt, exit_path)

        entries = load_quarantine_entries(entry_path)
        exits = load_exit_attempts(exit_path)
        status = get_quarantine_status(RUN_ID, entries, exits)
        assert status.is_quarantined is False


# ══════════════════════════════════════════════════════════════════════════════
# 7. DEPLOYMENT AUTHORIZATION REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestDeploymentAuthorityManifest:
    def test_manifest_id_deterministic(self):
        m1 = DeploymentAuthorityManifest.create(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            authority_level=DAM_FULL,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
            authorization_reason="all checks passed",
            created_at=FIXED_TS,
        )
        m2 = DeploymentAuthorityManifest.create(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            authority_level=DAM_FULL,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
            authorization_reason="all checks passed",
            created_at=FIXED_TS,
        )
        assert m1.manifest_id == m2.manifest_id

    def test_manifest_validates(self):
        m = DeploymentAuthorityManifest.create(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            authority_level=DAM_FULL,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
            authorization_reason="ok",
            created_at=FIXED_TS,
        )
        assert m.validate() is True

    def test_tampered_manifest_fails_validate(self):
        m = DeploymentAuthorityManifest.create(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            authority_level=DAM_FULL,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
            authorization_reason="ok",
            created_at=FIXED_TS,
        )
        import dataclasses
        bad = dataclasses.replace(m, manifest_id="tampered" * 8)
        assert bad.validate() is False

    def test_authorized_when_all_checks_pass(self):
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
        )
        assert m.is_authorized() is True
        assert m.deployment_status == MANIFEST_AUTHORIZED

    def test_blocked_when_quarantined(self):
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=True,
            quarantine_id="qid-001",
            pending_divergences=0,
        )
        assert m.is_authorized() is False
        assert m.deployment_status == MANIFEST_QUARANTINED_STATUS

    def test_blocked_when_integrity_fails(self):
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=False,
            integrity_violation_count=2,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
        )
        assert m.is_authorized() is False
        assert m.deployment_status == MANIFEST_BLOCKED

    def test_blocked_when_pending_divergences(self):
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=2,
        )
        assert m.is_authorized() is False
        assert m.deployment_status == MANIFEST_BLOCKED

    def test_blocked_when_missing_epoch_ids(self):
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="",
            deployment_epoch_id="",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
        )
        assert m.is_authorized() is False

    def test_blocked_authority_level_blocks(self):
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
            authority_level=DAM_BLOCKED,
        )
        assert m.is_authorized() is False

    def test_partial_authority_still_authorized(self):
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
            authority_level=DAM_PARTIAL,
        )
        assert m.is_authorized() is True

    def test_authorization_reason_auto_generated(self):
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=False,
            integrity_violation_count=3,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
        )
        assert "integrity" in m.authorization_reason

    def test_quarantine_overrides_integrity_ok(self):
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=True,
            quarantine_id="qid-001",
            pending_divergences=0,
        )
        assert m.deployment_status == MANIFEST_QUARANTINED_STATUS


class TestManifestPersistence:
    def test_manifest_roundtrip(self, tmp: Path):
        path = tmp / "manifests.jsonl"
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
        )
        append_manifest(m, path)
        loaded = load_manifests(path)
        assert len(loaded) == 1
        assert loaded[0].manifest_id == m.manifest_id
        assert loaded[0].validate() is True

    def test_get_latest_manifest_returns_most_recent(self, tmp: Path):
        path = tmp / "manifests.jsonl"
        for i in range(3):
            ts = f"2026-05-17T09:0{i}:00+00:00"
            m = DeploymentAuthorityManifest.create(
                run_id=RUN_ID,
                reconciliation_epoch_id=f"rec-00{i}",
                deployment_epoch_id=f"dep-00{i}",
                broker_snapshot_hash=FIXED_HASH,
                authority_level=DAM_FULL,
                integrity_ok=True,
                integrity_violation_count=0,
                is_quarantined=False,
                quarantine_id=None,
                pending_divergences=0,
                authorization_reason="ok",
                created_at=ts,
            )
            append_manifest(m, path)
        latest = get_latest_manifest(path, RUN_ID)
        assert latest is not None
        assert latest.reconciliation_epoch_id == "rec-002"

    def test_get_latest_manifest_returns_none_for_unknown_run(self, tmp: Path):
        path = tmp / "manifests.jsonl"
        assert get_latest_manifest(path, "unknown-run") is None

    def test_require_manifest_raises_when_missing(self, tmp: Path):
        path = tmp / "manifests.jsonl"
        with pytest.raises(RuntimeError, match="No deployment authority manifest"):
            require_manifest(path, RUN_ID)

    def test_require_manifest_raises_when_not_authorized(self, tmp: Path):
        path = tmp / "manifests.jsonl"
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=False,
            integrity_violation_count=1,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
        )
        append_manifest(m, path)
        with pytest.raises(RuntimeError, match="Deployment not authorized"):
            require_manifest(path, RUN_ID)

    def test_require_manifest_returns_authorized_manifest(self, tmp: Path):
        path = tmp / "manifests.jsonl"
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
        )
        append_manifest(m, path)
        result = require_manifest(path, RUN_ID)
        assert result.manifest_id == m.manifest_id

    def test_filters_by_run_id(self, tmp: Path):
        path = tmp / "manifests.jsonl"
        for run_id in ("run-A", "run-B"):
            m = build_manifest(
                run_id=run_id,
                reconciliation_epoch_id="rec-001",
                deployment_epoch_id="dep-001",
                broker_snapshot_hash=FIXED_HASH,
                integrity_ok=True,
                integrity_violation_count=0,
                is_quarantined=False,
                quarantine_id=None,
                pending_divergences=0,
            )
            append_manifest(m, path)
        m_a = get_latest_manifest(path, "run-A")
        m_b = get_latest_manifest(path, "run-B")
        assert m_a is not None and m_a.run_id == "run-A"
        assert m_b is not None and m_b.run_id == "run-B"
        assert m_a.manifest_id != m_b.manifest_id


# ══════════════════════════════════════════════════════════════════════════════
# 8. END-TO-END: FULL DEPLOYMENT CYCLE
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndDeploymentCycle:
    def test_complete_cycle_authorized(self, tmp: Path):
        """Happy path: broker snap → recon → integrity → authority → deployment."""
        # 1. Broker snapshot
        snap_hash = "a" * 64

        # 2. Reconciliation epoch
        rec_epoch = ReconciliationEpoch.create(
            run_id=RUN_ID,
            broker_snapshot_hash=snap_hash,
            reconciliation_ok=True,
            blocking_mismatches=0,
            created_at=FIXED_TS,
        )

        # 3. Deployment epoch
        dep_epoch = DeploymentEpoch.create(
            run_id=RUN_ID,
            reconciliation_epoch_id=rec_epoch.epoch_id,
            broker_snapshot_hash=snap_hash,
            deployment_allowed=True,
            authority_level=AUTHORITY_LEVEL_FULL,
            created_at=FIXED_TS,
        )

        # 4. Authority chain
        chain = build_authority_chain(rec_epoch, dep_epoch)
        assert chain.validate() is True
        assert chain.is_consistent_with(rec_epoch, dep_epoch) is True

        # 5. Lineage
        lineage = build_lineage(RUN_ID)
        for step, content in zip(ORDERED_STEPS, [
            {"snap_hash": snap_hash},
            {"reconciliation_ok": True},
            {"integrity_ok": True},
            {"authority_level": "full"},
            {"deployment_allowed": True},
            {"broker_ack": "ok"},
            {"replay_hash": "xyz"},
        ]):
            append_lineage_step(lineage, step, content, timestamp=FIXED_TS)
        assert lineage.verify_chain() is True

        # 6. Manifest
        authority_entry = lineage.get_step(STEP_AUTHORITY_DECISION)
        assert authority_entry is not None
        manifest = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id=rec_epoch.epoch_id,
            deployment_epoch_id=dep_epoch.epoch_id,
            broker_snapshot_hash=snap_hash,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
            lineage_entry_hash=authority_entry.entry_hash,
        )
        assert manifest.is_authorized() is True
        assert manifest.validate() is True

    def test_quarantine_blocks_entire_cycle(self, tmp: Path):
        """Quarantine overrides deployment authorization."""
        rec_epoch = ReconciliationEpoch.create(
            run_id=RUN_ID,
            broker_snapshot_hash=FIXED_HASH,
            reconciliation_ok=False,
            blocking_mismatches=2,
            created_at=FIXED_TS,
        )
        entry = QuarantineEntry.create(
            run_id=RUN_ID,
            entry_reason=QUARANTINE_RECONCILIATION_MISMATCH,
            entry_epoch_id=rec_epoch.epoch_id,
            broker_snapshot_hash=FIXED_HASH,
            reconciliation_epoch_id=rec_epoch.epoch_id,
            reconciliation_mismatch_detail="2 blocking mismatches",
            created_at=FIXED_TS,
        )
        status = get_quarantine_status(RUN_ID, [entry], [])
        assert status.is_quarantined is True

        manifest = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id=rec_epoch.epoch_id,
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=True,
            quarantine_id=entry.quarantine_id,
            pending_divergences=0,
        )
        assert manifest.is_authorized() is False
        assert manifest.deployment_status == MANIFEST_QUARANTINED_STATUS

    def test_stale_authority_blocks_manifest(self, tmp: Path):
        """Stale pending order authority prevents deployment."""
        old_ts = "2020-01-01T00:00:00+00:00"
        auth = _make_authority(submitted_at=old_ts)
        report = detect_authority_divergences([auth], now_sec=time.time())
        assert report.blocks_deployment is True

        manifest = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="rec-001",
            deployment_epoch_id="dep-001",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=report.total_divergences,
        )
        assert manifest.is_authorized() is False

    def test_require_manifest_gates_execution(self, tmp: Path):
        """require_manifest raises without a valid manifest."""
        manifest_path = tmp / "manifests.jsonl"
        with pytest.raises(RuntimeError):
            require_manifest(manifest_path, RUN_ID)

        # Write a blocked manifest — still raises
        m = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id="",
            deployment_epoch_id="",
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=False,
            integrity_violation_count=1,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
        )
        append_manifest(m, manifest_path)
        with pytest.raises(RuntimeError, match="Deployment not authorized"):
            require_manifest(manifest_path, RUN_ID)

    def test_full_persistence_and_replay(self, tmp: Path):
        """Persist all artifacts, reload, verify invariants hold."""
        rec_path = tmp / "rec_epochs.jsonl"
        dep_path = tmp / "dep_epochs.jsonl"
        chain_path = tmp / "chains.jsonl"
        lineage_path = tmp / "lineage.jsonl"
        manifest_path = tmp / "manifests.jsonl"
        quarantine_path = tmp / "quarantine.jsonl"

        rec = ReconciliationEpoch.create(RUN_ID, FIXED_HASH, True, 0, FIXED_TS)
        dep = DeploymentEpoch.create(RUN_ID, rec.epoch_id, FIXED_HASH, True, AUTHORITY_LEVEL_FULL, FIXED_TS)
        chain = build_authority_chain(rec, dep)

        append_reconciliation_epoch(rec, rec_path)
        append_deployment_epoch(dep, dep_path)
        append_authority_chain(chain, chain_path)

        lineage = build_lineage(RUN_ID)
        for step in ORDERED_STEPS:
            append_lineage_step(lineage, step, {"step": step}, timestamp=FIXED_TS)
        persist_lineage(lineage, lineage_path)

        manifest = build_manifest(
            run_id=RUN_ID,
            reconciliation_epoch_id=rec.epoch_id,
            deployment_epoch_id=dep.epoch_id,
            broker_snapshot_hash=FIXED_HASH,
            integrity_ok=True,
            integrity_violation_count=0,
            is_quarantined=False,
            quarantine_id=None,
            pending_divergences=0,
        )
        append_manifest(manifest, manifest_path)

        # Replay all
        loaded_recs = load_reconciliation_epochs(rec_path)
        loaded_deps = load_deployment_epochs(dep_path)
        loaded_chains = load_authority_chains(chain_path)
        loaded_lineage = reconstruct_lineage(lineage_path, RUN_ID)
        loaded_manifests = load_manifests(manifest_path)

        assert loaded_recs[0].validate() is True
        assert loaded_deps[0].validate() is True
        assert loaded_chains[0].validate() is True
        assert loaded_lineage.verify_chain() is True
        assert loaded_manifests[0].validate() is True
        assert loaded_manifests[0].is_authorized() is True
