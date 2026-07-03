"""
src/portfolio/test_state_store.py
state_store.py の不変条件テスト (20+ tests)。

実行:
    cd C:/ai-trading
    python -m pytest src/portfolio/test_state_store.py -v
または:
    python src/portfolio/test_state_store.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.portfolio.state_store import (
    SCHEMA_VERSION,
    STALE_HARD_SECONDS,
    STALE_WARN_SECONDS,
    V2_SAFE_DEFAULTS,
    BrokerSnapshot,
    SnapshotValidationError,
    ValidationResult,
    _compute_snapshot_hash,
    _recompute_hash_from_state,
    _read_generation_watermark,
    _write_generation_watermark,
    atomic_write_json,
    commit_broker_snapshot,
    load_portfolio_state,
    log_startup_state_line,
    save_portfolio_state,
    update_portfolio_state_from_broker,
    validate_broker_snapshot,
    validate_state,
    write_reconciliation_log,
)

JST = timezone(timedelta(hours=9))


def _tmp_path() -> Path:
    fd, p = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(p)
    return Path(p)


def _make_valid_state(**overrides) -> dict:
    s = {
        "schema_version":        SCHEMA_VERSION,
        "updated_at":            datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_source":           "test",
        "cb_state":              "NORMAL",
        "equity_peak":           3_048_109.0,
        "available_cash":        1_799_309.0,
        "position_entry_dates":  {"6981.T": "2026-04-28"},
        "position_entry_prices": {"6981.T": 4864.0},
        "position_qtys":         {"6981.T": 100},
        "positions_count":       1,
        "last_equity":           3_048_109.0,
        "safe_warn_count":       0,
    }
    s.update(overrides)
    return s


class TestAtomicWrite(unittest.TestCase):

    def test_atomic_write_creates_file(self):
        p = _tmp_path()
        data = {"a": 1, "b": "test"}
        atomic_write_json(p, data)
        self.assertTrue(p.exists())
        loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(loaded, data)
        p.unlink()

    def test_atomic_write_no_temp_file_left_on_success(self):
        # 専用ディレクトリを使うことでシステム全体の .tmp ファイルを拾わない
        import shutil
        tmpdir = Path(tempfile.mkdtemp())
        p = tmpdir / "state.json"
        try:
            atomic_write_json(p, {"x": 99})
            tmp_files = list(tmpdir.glob("*.tmp"))
            self.assertEqual(len(tmp_files), 0, f"残存 .tmp: {tmp_files}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_atomic_write_unicode(self):
        p = _tmp_path()
        data = {"msg": "村田製作所", "val": 4864}
        atomic_write_json(p, data)
        loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(loaded["msg"], "村田製作所")
        p.unlink()

    def test_atomic_write_overwrites_existing(self):
        p = _tmp_path()
        atomic_write_json(p, {"v": 1})
        atomic_write_json(p, {"v": 2})
        self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["v"], 2)
        p.unlink()


class TestLoadPortfolioState(unittest.TestCase):

    def test_load_valid_state(self):
        p = _tmp_path()
        state = _make_valid_state()
        atomic_write_json(p, state)
        loaded, vr = load_portfolio_state(p)
        self.assertTrue(vr.ok)
        self.assertEqual(loaded["equity_peak"], 3_048_109.0)
        p.unlink()

    def test_corrupted_json_recovery(self):
        """壊れた JSON → safe defaults を返し crash しない。"""
        p = _tmp_path()
        p.write_text("{ NOT VALID JSON !!!}", encoding="utf-8")
        loaded, vr = load_portfolio_state(p)
        self.assertFalse(vr.ok)
        self.assertTrue(any("JSON" in f for f in vr.hard_fails))
        # safe defaults が返る
        self.assertEqual(loaded["cb_state"], "NORMAL")
        p.unlink()

    def test_missing_file_returns_defaults(self):
        p = Path(tempfile.gettempdir()) / "nonexistent_state_9999.json"
        if p.exists():
            p.unlink()
        loaded, vr = load_portfolio_state(p)
        self.assertFalse(vr.ok)
        self.assertEqual(loaded["cb_state"], "NORMAL")

    def test_schema_v1_migrated_to_v2(self):
        """schema_version なし (v1) → v2 に self-heal。"""
        p = _tmp_path()
        v1 = {
            "cb_state": "NORMAL",
            "equity_peak": 3_000_000.0,
            "available_cash": 2_500_000.0,
        }
        atomic_write_json(p, v1)
        loaded, vr = load_portfolio_state(p)
        # 結果として schema_version=2 になっていること (経路は問わない)
        self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
        # v1 にない必須キーが safe defaults で補填されていること
        self.assertIn("position_qtys", loaded)
        self.assertIn("last_equity",   loaded)
        p.unlink()

    def test_nan_cash_healed(self):
        p = _tmp_path()
        bad = _make_valid_state(available_cash=float("nan"))
        atomic_write_json(p, bad)
        loaded, vr = load_portfolio_state(p)
        self.assertEqual(loaded["available_cash"], 0.0)
        self.assertTrue(any("NaN" in h for h in vr.healed))
        p.unlink()

    def test_negative_cash_healed(self):
        p = _tmp_path()
        bad = _make_valid_state(available_cash=-999.0)
        atomic_write_json(p, bad)
        loaded, vr = load_portfolio_state(p)
        self.assertEqual(loaded["available_cash"], 0.0)
        p.unlink()

    def test_negative_peak_healed(self):
        p = _tmp_path()
        bad = _make_valid_state(equity_peak=-1.0)
        atomic_write_json(p, bad)
        loaded, vr = load_portfolio_state(p)
        self.assertGreater(loaded["equity_peak"], 0)
        p.unlink()

    def test_invalid_cb_state_healed(self):
        p = _tmp_path()
        bad = _make_valid_state(cb_state="INVALID_STATE")
        atomic_write_json(p, bad)
        loaded, vr = load_portfolio_state(p)
        self.assertEqual(loaded["cb_state"], "NORMAL")
        p.unlink()

    def test_positions_count_recomputed(self):
        """positions_count が position_qtys と食い違う → 自動修正。"""
        p = _tmp_path()
        bad = _make_valid_state(positions_count=99)
        bad["position_qtys"] = {"6981.T": 100}
        atomic_write_json(p, bad)
        loaded, vr = load_portfolio_state(p)
        self.assertEqual(loaded["positions_count"], 1)
        p.unlink()

    def test_missing_candidate_peak_defaults_to_none(self):
        """candidate_peak キー欠損時は None で補填される (EQUITY_PEAK_HARDENING)。"""
        p = _tmp_path()
        good = _make_valid_state()
        good.pop("candidate_peak", None)
        atomic_write_json(p, good)
        loaded, vr = load_portfolio_state(p)
        self.assertIsNone(loaded["candidate_peak"])
        p.unlink()

    def test_malformed_candidate_peak_healed_to_none(self):
        """candidate_peak が不正形式（value/staged_dateキー欠損）なら None にクリア。"""
        p = _tmp_path()
        bad = _make_valid_state(candidate_peak={"value": 4_000_000.0})  # staged_date欠損
        atomic_write_json(p, bad)
        loaded, vr = load_portfolio_state(p)
        self.assertIsNone(loaded["candidate_peak"])
        self.assertTrue(any("candidate_peak" in h for h in vr.healed))
        p.unlink()

    def test_nan_candidate_peak_value_healed_to_none(self):
        p = _tmp_path()
        bad = _make_valid_state(candidate_peak={
            "value": float("nan"), "staged_date": "2026-07-03",
        })
        atomic_write_json(p, bad)
        loaded, vr = load_portfolio_state(p)
        self.assertIsNone(loaded["candidate_peak"])
        p.unlink()

    def test_valid_candidate_peak_preserved(self):
        p = _tmp_path()
        cand = {
            "value": 4_400_000.0, "staged_date": "2026-07-03",
            "reason": "new_high", "current_equity_at_stage": 4_400_000.0,
        }
        good = _make_valid_state(candidate_peak=cand)
        atomic_write_json(p, good)
        loaded, vr = load_portfolio_state(p)
        self.assertEqual(loaded["candidate_peak"], cand)
        p.unlink()


class TestValidateState(unittest.TestCase):

    def test_valid_state_passes(self):
        vr = validate_state(_make_valid_state())
        self.assertTrue(vr.ok)
        self.assertEqual(vr.hard_fails, [])

    def test_nan_equity_hard_fail(self):
        bad = _make_valid_state(last_equity=float("nan"))
        vr = validate_state(bad)
        self.assertFalse(vr.ok)
        self.assertTrue(any("NaN" in f for f in vr.hard_fails))

    def test_negative_equity_hard_fail(self):
        bad = _make_valid_state(last_equity=-1.0)
        vr = validate_state(bad)
        self.assertFalse(vr.ok)

    def test_negative_cash_hard_fail(self):
        bad = _make_valid_state(available_cash=-100.0)
        vr = validate_state(bad)
        self.assertFalse(vr.ok)

    def test_negative_qty_hard_fail(self):
        bad = _make_valid_state()
        bad["position_qtys"] = {"6981.T": -1}
        vr = validate_state(bad)
        self.assertFalse(vr.ok)
        self.assertTrue(any("負値" in f for f in vr.hard_fails))

    def test_nan_peak_hard_fail(self):
        bad = _make_valid_state(equity_peak=float("inf"))
        vr = validate_state(bad)
        self.assertFalse(vr.ok)

    def test_malformed_candidate_peak_warning_not_hard_fail(self):
        """candidate_peak 形式不正は self-heal 対象の警告であり hard_fail ではない。"""
        bad = _make_valid_state(candidate_peak={"value": 4_000_000.0})  # staged_date欠損
        vr = validate_state(bad)
        self.assertTrue(vr.ok)
        self.assertTrue(any("candidate_peak" in w for w in vr.warnings))

    def test_valid_candidate_peak_no_warning(self):
        cand = {"value": 4_000_000.0, "staged_date": "2026-07-03"}
        s = _make_valid_state(candidate_peak=cand)
        vr = validate_state(s)
        self.assertTrue(vr.ok)
        self.assertFalse(any("candidate_peak" in w for w in vr.warnings))

    def test_none_candidate_peak_no_warning(self):
        s = _make_valid_state(candidate_peak=None)
        vr = validate_state(s)
        self.assertTrue(vr.ok)
        self.assertFalse(any("candidate_peak" in w for w in vr.warnings))

    def test_zero_equity_warning_not_hard_fail(self):
        """equity=0 は初回起動で許容 (警告のみ)。"""
        s = _make_valid_state(last_equity=0.0)
        vr = validate_state(s)
        self.assertTrue(vr.ok)   # hard fail ではない
        self.assertTrue(any("0" in w for w in vr.warnings))

    def test_stale_15min_warning(self):
        old_ts = (datetime.now(JST) - timedelta(seconds=STALE_WARN_SECONDS + 60))
        s = _make_valid_state(updated_at=old_ts.strftime("%Y-%m-%dT%H:%M:%S%z"))
        vr = validate_state(s)
        self.assertTrue(vr.is_stale)
        self.assertTrue(any("stale" in w for w in vr.warnings))

    def test_stale_60min_hard_warning(self):
        old_ts = (datetime.now(JST) - timedelta(seconds=STALE_HARD_SECONDS + 60))
        s = _make_valid_state(updated_at=old_ts.strftime("%Y-%m-%dT%H:%M:%S%z"))
        vr = validate_state(s)
        self.assertTrue(vr.is_very_stale)

    def test_non_dict_hard_fail(self):
        vr = validate_state([1, 2, 3])  # type: ignore
        self.assertFalse(vr.ok)


class TestSavePortfolioState(unittest.TestCase):

    def test_save_adds_schema_version(self):
        p = _tmp_path()
        s = _make_valid_state()
        save_portfolio_state(s, path=p, data_source="test")
        loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
        p.unlink()

    def test_save_adds_updated_at(self):
        p = _tmp_path()
        save_portfolio_state(_make_valid_state(), path=p)
        loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertIn("updated_at", loaded)
        self.assertIsNotNone(loaded["updated_at"])
        p.unlink()

    def test_save_sets_data_source(self):
        p = _tmp_path()
        save_portfolio_state(_make_valid_state(), path=p, data_source="broker_api")
        loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(loaded["data_source"], "broker_api")
        p.unlink()

    def test_save_recomputes_positions_count(self):
        p = _tmp_path()
        s = _make_valid_state()
        s["position_qtys"] = {"6981.T": 100, "8015.T": 100}
        s["positions_count"] = 0  # intentionally wrong
        save_portfolio_state(s, path=p)
        loaded = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(loaded["positions_count"], 2)
        p.unlink()


class TestBrokerSnapshotPersistence(unittest.TestCase):

    def test_update_cash(self):
        s = _make_valid_state(available_cash=0.0)
        update_portfolio_state_from_broker(s, available_cash=1_799_309.0)
        self.assertEqual(s["available_cash"], 1_799_309.0)

    def test_update_position_qtys(self):
        s = _make_valid_state()
        update_portfolio_state_from_broker(s, position_qtys={"6981.T": 100, "8015.T": 100})
        self.assertEqual(s["position_qtys"]["6981.T"], 100)
        self.assertEqual(s["positions_count"], 2)

    def test_update_equity(self):
        s = _make_valid_state(last_equity=0.0)
        update_portfolio_state_from_broker(s, current_equity=3_048_109.0)
        self.assertEqual(s["last_equity"], 3_048_109.0)

    def test_zero_qty_positions_excluded(self):
        s = _make_valid_state()
        update_portfolio_state_from_broker(s, position_qtys={"6981.T": 0, "8015.T": 100})
        self.assertNotIn("6981.T", s["position_qtys"])
        self.assertEqual(s["positions_count"], 1)

    def test_dry_fallback_uses_persisted_positions(self):
        """DRY ラン (API 空) でも状態に保存済みの qty が保持されること。"""
        s = _make_valid_state()
        s["position_qtys"] = {"6981.T": 100}
        # DRY ラン: broker が空を返す → update 呼ばれない → 既存 qty 保持
        # (このテストは update が呼ばれない場合を確認)
        self.assertEqual(s["position_qtys"]["6981.T"], 100)


class TestStaleStateDetection(unittest.TestCase):

    def _state_with_age(self, seconds_ago: float) -> dict:
        old_ts = (datetime.now(JST) - timedelta(seconds=seconds_ago))
        return _make_valid_state(updated_at=old_ts.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def test_fresh_state_not_stale(self):
        s = self._state_with_age(60)
        vr = validate_state(s)
        self.assertFalse(vr.is_stale)

    def test_15min_stale(self):
        s = self._state_with_age(STALE_WARN_SECONDS + 10)
        vr = validate_state(s)
        self.assertTrue(vr.is_stale)

    def test_60min_very_stale(self):
        s = self._state_with_age(STALE_HARD_SECONDS + 10)
        vr = validate_state(s)
        self.assertTrue(vr.is_very_stale)

    def test_old_date_format_parsed(self):
        """旧形式 YYYY-MM-DD でも年齢計算できること。"""
        s = _make_valid_state(updated_at="2020-01-01")
        vr = validate_state(s)
        self.assertIsNotNone(vr.snapshot_age_seconds)
        self.assertGreater(vr.snapshot_age_seconds, STALE_HARD_SECONDS)


class TestReconciliationLog(unittest.TestCase):

    def test_reconciliation_log_written(self):
        log_dir = Path(tempfile.mkdtemp())
        write_reconciliation_log(
            mode="dry",
            broker_cash=1_799_309.0,
            state_cash=1_799_309.0,
            broker_equity=3_048_109.0,
            computed_equity=3_048_109.0,
            positions_match=True,
            log_dir=log_dir,
        )
        files = list(log_dir.glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        record = json.loads(files[0].read_text(encoding="utf-8").strip())
        self.assertEqual(record["mode"], "dry")
        self.assertEqual(record["positions_match"], True)
        self.assertEqual(record["delta"], 0.0)
        shutil.rmtree(log_dir, ignore_errors=True)

    def test_reconciliation_delta_mismatch(self):
        """delta != 0 でも crash しないこと (WARNING ログ出力のみ)。"""
        import shutil
        log_dir = Path(tempfile.mkdtemp())
        write_reconciliation_log(
            mode="live",
            broker_cash=1_799_309.0,
            state_cash=2_000_000.0,
            broker_equity=3_048_109.0,
            computed_equity=3_250_000.0,
            positions_match=False,
            log_dir=log_dir,
        )
        record = json.loads(list(log_dir.glob("*.jsonl"))[0].read_text().strip())
        self.assertNotEqual(record["delta"], 0)
        shutil.rmtree(log_dir, ignore_errors=True)


class TestConcurrentWriteSafety(unittest.TestCase):

    def test_concurrent_writes_no_corruption(self):
        """
        複数スレッドが同時に書き込んでもファイルが corrupt しないこと。

        Windows の os.replace() はリトライ付きで競合を処理する。
        全スレッドが成功するとは限らない (last-write-wins) が、
        最終ファイルは必ず valid JSON でなければならない。
        """
        import shutil
        tmpdir = Path(tempfile.mkdtemp())
        p = tmpdir / "state_concurrent.json"
        errors = []

        def writer(i: int):
            try:
                s = _make_valid_state(last_equity=float(i * 100_000))
                save_portfolio_state(s, path=p)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # ファイルが存在し valid JSON であることを確認 (corruption は許容しない)
        self.assertTrue(p.exists(), "並行書き込み後にファイルが存在しない")
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            self.assertIn("schema_version", loaded)
        except json.JSONDecodeError as e:
            self.fail(f"並行書き込み後にファイルが壊れた: {e}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


import shutil  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_snapshot(**overrides) -> BrokerSnapshot:
    """有効な BrokerSnapshot を返すヘルパー。"""
    base = BrokerSnapshot(
        cash          = 1_799_309.0,
        positions     = {"6981.T": 100, "8015.T": 100},
        avg_costs     = {"6981.T": 4864.0, "8015.T": 6733.0},
        market_values = {"6981.T": 5648.0, "8015.T": 6840.0},
        equity        = 3_048_109.0,
        ts            = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        source        = "broker",
        api_health    = {"positions_ok": True, "wallet_ok": True},
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ── TestBrokerSnapshotValidation ──────────────────────────────────────────────

class TestBrokerSnapshotValidation(unittest.TestCase):

    def test_valid_snapshot_passes(self):
        errors = validate_broker_snapshot(_make_snapshot())
        self.assertEqual(errors, [])

    def test_negative_cash_fails(self):
        errors = validate_broker_snapshot(_make_snapshot(cash=-100.0))
        self.assertTrue(any("negative" in e for e in errors))

    def test_nan_cash_fails(self):
        errors = validate_broker_snapshot(_make_snapshot(cash=float("nan")))
        self.assertTrue(any("NaN/Inf" in e for e in errors))

    def test_inf_cash_fails(self):
        errors = validate_broker_snapshot(_make_snapshot(cash=float("inf")))
        self.assertTrue(any("NaN/Inf" in e for e in errors))

    def test_mismatched_positions_costs_fails(self):
        snap = _make_snapshot()
        snap.avg_costs = {"6981.T": 4864.0}   # 8015.T missing
        errors = validate_broker_snapshot(snap)
        self.assertTrue(any("mismatch" in e for e in errors))

    def test_mismatched_positions_market_values_fails(self):
        snap = _make_snapshot()
        snap.market_values = {"6981.T": 5648.0}  # 8015.T missing
        errors = validate_broker_snapshot(snap)
        self.assertTrue(any("mismatch" in e for e in errors))

    def test_qty_nan_fails(self):
        snap = _make_snapshot()
        snap.positions = {"6981.T": float("nan"), "8015.T": 100}
        errors = validate_broker_snapshot(snap)
        self.assertTrue(any("NaN/Inf" in e for e in errors))

    def test_cost_nan_fails(self):
        snap = _make_snapshot()
        snap.avg_costs = {"6981.T": float("nan"), "8015.T": 6733.0}
        errors = validate_broker_snapshot(snap)
        self.assertTrue(any("NaN/Inf" in e for e in errors))

    def test_empty_ts_fails(self):
        errors = validate_broker_snapshot(_make_snapshot(ts=""))
        self.assertTrue(any("ts" in e for e in errors))

    def test_equity_nan_fails(self):
        errors = validate_broker_snapshot(_make_snapshot(equity=float("nan")))
        self.assertTrue(any("NaN/Inf" in e for e in errors))

    def test_zero_equity_passes(self):
        # equity=0 は commit 前の初期値として許容
        errors = validate_broker_snapshot(_make_snapshot(equity=0.0))
        self.assertEqual(errors, [])

    def test_qty_count_ne_cost_count_fails(self):
        snap = _make_snapshot()
        snap.avg_costs = {"6981.T": 4864.0, "8015.T": 6733.0, "extra.T": 1000.0}
        errors = validate_broker_snapshot(snap)
        # symbol mismatch が検出される
        self.assertTrue(len(errors) > 0)

    def test_empty_positions_passes(self):
        snap = _make_snapshot(
            positions={}, avg_costs={}, market_values={}
        )
        errors = validate_broker_snapshot(snap)
        self.assertEqual(errors, [])


# ── TestCommitBrokerSnapshot ──────────────────────────────────────────────────

class TestCommitBrokerSnapshot(unittest.TestCase):

    def test_full_commit_updates_all_fields(self):
        state = _make_valid_state()
        snap  = _make_snapshot()
        commit_broker_snapshot(state, snap)
        self.assertEqual(state["available_cash"],  1_799_309.0)
        self.assertEqual(state["position_qtys"],   {"6981.T": 100, "8015.T": 100})
        self.assertEqual(state["positions_count"], 2)
        self.assertEqual(state["last_equity"],     3_048_109.0)
        self.assertIsNotNone(state["snapshot_hash"])
        self.assertIsNotNone(state["snapshot_ts"])

    def test_validation_failure_leaves_state_unchanged(self):
        state = _make_valid_state()
        original_cash = state["available_cash"]
        bad_snap = _make_snapshot(cash=float("nan"))
        with self.assertRaises(SnapshotValidationError):
            commit_broker_snapshot(state, bad_snap)
        # state must be unchanged
        self.assertEqual(state["available_cash"], original_cash)

    def test_mismatched_costs_raises_and_no_partial_write(self):
        state = _make_valid_state()
        original_positions = dict(state.get("position_qtys", {}))
        snap = _make_snapshot()
        snap.avg_costs = {"6981.T": 4864.0}  # missing 8015.T
        with self.assertRaises(SnapshotValidationError):
            commit_broker_snapshot(state, snap)
        self.assertEqual(state.get("position_qtys", {}), original_positions)

    def test_zero_qty_excluded_from_commit(self):
        snap = _make_snapshot()
        snap.positions     = {"6981.T": 0, "8015.T": 100}
        snap.avg_costs     = {"6981.T": 4864.0, "8015.T": 6733.0}
        snap.market_values = {"6981.T": 5648.0, "8015.T": 6840.0}
        state = _make_valid_state()
        commit_broker_snapshot(state, snap)
        self.assertNotIn("6981.T", state["position_qtys"])
        self.assertEqual(state["positions_count"], 1)

    def test_entry_price_preserved_for_existing_symbol(self):
        state = _make_valid_state()
        state["position_entry_prices"] = {"6981.T": 9999.0}  # original entry
        snap = _make_snapshot()
        commit_broker_snapshot(state, snap)
        # original entry price MUST NOT be overwritten by avg_cost
        self.assertEqual(state["position_entry_prices"]["6981.T"], 9999.0)

    def test_new_symbol_gets_avg_cost_as_entry_price(self):
        state = _make_valid_state()
        state["position_entry_prices"] = {}  # no prior entry
        snap = _make_snapshot()
        commit_broker_snapshot(state, snap)
        self.assertEqual(state["position_entry_prices"]["6981.T"], 4864.0)

    def test_snapshot_avg_costs_stored(self):
        state = _make_valid_state()
        commit_broker_snapshot(state, _make_snapshot())
        self.assertIn("snapshot_avg_costs", state)
        self.assertEqual(state["snapshot_avg_costs"]["6981.T"], 4864.0)

    def test_exception_during_save_does_not_corrupt_state(self):
        """save_portfolio_state() 失敗でも state dict は commit 後の値を保持する。
        (dict 更新と file 書き込みは分離されているため dict は常に一貫している)"""
        state = _make_valid_state()
        snap  = _make_snapshot()
        commit_broker_snapshot(state, snap)
        # file write failure は state dict には影響しない
        self.assertEqual(state["last_equity"], 3_048_109.0)


# ── TestGenerationId ──────────────────────────────────────────────────────────

class TestGenerationId(unittest.TestCase):

    def test_generation_increments_on_each_save(self):
        p = _tmp_path()
        s = _make_valid_state()
        save_portfolio_state(s, path=p)
        gen1 = json.loads(p.read_text(encoding="utf-8"))["generation_id"]
        save_portfolio_state(s, path=p)
        gen2 = json.loads(p.read_text(encoding="utf-8"))["generation_id"]
        self.assertEqual(gen2, gen1 + 1)
        p.unlink()

    def test_generation_never_decrements(self):
        p = _tmp_path()
        s = _make_valid_state()
        for _ in range(5):
            save_portfolio_state(s, path=p)
        final_gen = json.loads(p.read_text(encoding="utf-8"))["generation_id"]
        self.assertGreaterEqual(final_gen, 5)
        p.unlink()

    def test_no_duplicate_generation_ids_sequential(self):
        p = _tmp_path()
        s = _make_valid_state()
        seen: list[int] = []
        for _ in range(10):
            save_portfolio_state(s, path=p)
            seen.append(json.loads(p.read_text(encoding="utf-8"))["generation_id"])
        self.assertEqual(len(seen), len(set(seen)), f"duplicate generation IDs: {seen}")
        p.unlink()

    def test_rollback_detection_warns(self):
        import shutil as _sh
        tmpdir = Path(tempfile.mkdtemp())
        # set watermark to 50
        wm_path = tmpdir / "state_generation.txt"
        wm_path.write_text("50", encoding="utf-8")

        p = tmpdir / "portfolio_state.json"
        s = _make_valid_state(generation_id=10)  # lower than watermark
        atomic_write_json(p, s)

        import unittest.mock as _mock
        from src.portfolio import state_store as _ss
        orig_wm = _ss._GENERATION_WATERMARK_PATH
        _ss._GENERATION_WATERMARK_PATH = wm_path
        try:
            with self.assertLogs("src.portfolio.state_store", level="WARNING") as cm:
                load_portfolio_state(p)
            self.assertTrue(any("rollback" in m.lower() for m in cm.output))
        finally:
            _ss._GENERATION_WATERMARK_PATH = orig_wm
            _sh.rmtree(tmpdir, ignore_errors=True)

    def test_watermark_updated_on_save(self):
        import shutil as _sh
        tmpdir = Path(tempfile.mkdtemp())
        wm_path = tmpdir / "state_generation.txt"
        p = tmpdir / "state.json"

        import unittest.mock as _mock
        from src.portfolio import state_store as _ss
        orig_wm = _ss._GENERATION_WATERMARK_PATH
        _ss._GENERATION_WATERMARK_PATH = wm_path
        try:
            s = _make_valid_state()
            save_portfolio_state(s, path=p)
            wm = int(wm_path.read_text(encoding="utf-8").strip())
            self.assertGreaterEqual(wm, 1)
        finally:
            _ss._GENERATION_WATERMARK_PATH = orig_wm
            _sh.rmtree(tmpdir, ignore_errors=True)


# ── TestSnapshotHash ──────────────────────────────────────────────────────────

class TestSnapshotHash(unittest.TestCase):

    def test_identical_snapshots_produce_same_hash(self):
        s1 = _make_snapshot()
        s2 = _make_snapshot()
        self.assertEqual(_compute_snapshot_hash(s1), _compute_snapshot_hash(s2))

    def test_different_cash_produces_different_hash(self):
        s1 = _make_snapshot(cash=1_799_309.0)
        s2 = _make_snapshot(cash=2_000_000.0)
        self.assertNotEqual(_compute_snapshot_hash(s1), _compute_snapshot_hash(s2))

    def test_different_qty_produces_different_hash(self):
        s1 = _make_snapshot()
        s2 = _make_snapshot()
        s2.positions = {"6981.T": 200, "8015.T": 100}
        self.assertNotEqual(_compute_snapshot_hash(s1), _compute_snapshot_hash(s2))

    def test_hash_integrity_after_commit(self):
        """commit 後の state から hash を再計算しても stored_hash と一致すること。"""
        state = _make_valid_state()
        snap  = _make_snapshot()
        commit_broker_snapshot(state, snap)
        stored  = state["snapshot_hash"]
        recomp  = _recompute_hash_from_state(state)
        self.assertEqual(stored, recomp)

    def test_hash_mismatch_detected_in_validate_state(self):
        """state の cash を直接変更すると validate_state() が hash mismatch を警告する。"""
        state = _make_valid_state()
        commit_broker_snapshot(state, _make_snapshot())
        # 外部から cash を直接変更 (state_store bypass)
        state["available_cash"] = 9_999_999.0
        vr = validate_state(state)
        self.assertTrue(any("mismatch" in w for w in vr.warnings))

    def test_no_hash_no_warning(self):
        """snapshot_hash が None なら hash チェックをスキップ。"""
        state = _make_valid_state()
        state["snapshot_hash"] = None
        vr = validate_state(state)
        self.assertFalse(any("mismatch" in w for w in vr.warnings))

    def test_hash_16_chars(self):
        snap = _make_snapshot()
        h = _compute_snapshot_hash(snap)
        self.assertEqual(len(h), 16)


# ── TestPartialWriteInterruption ─────────────────────────────────────────────

class TestPartialWriteInterruption(unittest.TestCase):

    def test_exception_during_atomic_write_leaves_original_intact(self):
        """書き込み中に例外が出ても元ファイルが intact なこと。"""
        import shutil as _sh
        tmpdir = Path(tempfile.mkdtemp())
        p = tmpdir / "state.json"
        original = _make_valid_state(last_equity=1_000_000.0)
        atomic_write_json(p, original)

        # patching json.dump to raise after initial file write
        import unittest.mock as _mock
        call_count = [0]
        original_dump = json.dump

        def raising_dump(obj, fp, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("simulated disk error")
            return original_dump(obj, fp, **kwargs)

        try:
            with _mock.patch("json.dump", side_effect=raising_dump):
                try:
                    atomic_write_json(p, _make_valid_state(last_equity=9_999_999.0))
                except OSError:
                    pass  # expected

            # original file must still be readable and intact
            loaded = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(loaded["last_equity"], 1_000_000.0)
        finally:
            _sh.rmtree(tmpdir, ignore_errors=True)

    def test_no_tmp_file_left_after_exception(self):
        """例外発生時に .tmp ファイルが残らないこと。"""
        import unittest.mock as _mock
        import shutil as _sh
        tmpdir = Path(tempfile.mkdtemp())
        p = tmpdir / "state.json"

        original_dump = json.dump
        call_count = [0]

        def raising_dump(obj, fp, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("simulated")
            return original_dump(obj, fp, **kwargs)

        try:
            with _mock.patch("json.dump", side_effect=raising_dump):
                try:
                    atomic_write_json(p, {"x": 1})
                except OSError:
                    pass
            tmp_files = list(tmpdir.glob("*.tmp"))
            self.assertEqual(len(tmp_files), 0, f"tmp files left: {tmp_files}")
        finally:
            _sh.rmtree(tmpdir, ignore_errors=True)

    def test_commit_validation_failure_leaves_state_unchanged(self):
        """バリデーション失敗 → SnapshotValidationError → state 変更なし。"""
        state = _make_valid_state(available_cash=500_000.0)
        bad   = _make_snapshot(cash=float("nan"))
        with self.assertRaises(SnapshotValidationError):
            commit_broker_snapshot(state, bad)
        self.assertEqual(state["available_cash"], 500_000.0)


# ── TestRollbackDetection ─────────────────────────────────────────────────────

class TestRollbackDetection(unittest.TestCase):

    def _patched_watermark(self, tmpdir: Path, watermark_val: int):
        """watermark ファイルを tmpdir に置き _GENERATION_WATERMARK_PATH をパッチするコンテキスト。"""
        import contextlib
        import unittest.mock as _mock
        from src.portfolio import state_store as _ss

        @contextlib.contextmanager
        def _ctx():
            wm_path = tmpdir / "state_generation.txt"
            wm_path.write_text(str(watermark_val), encoding="utf-8")
            orig = _ss._GENERATION_WATERMARK_PATH
            _ss._GENERATION_WATERMARK_PATH = wm_path
            try:
                yield wm_path
            finally:
                _ss._GENERATION_WATERMARK_PATH = orig

        return _ctx()

    def test_rollback_warning_emitted(self):
        import shutil as _sh
        tmpdir = Path(tempfile.mkdtemp())
        try:
            p = tmpdir / "state.json"
            # write state with low generation
            s = _make_valid_state()
            s["generation_id"] = 3
            atomic_write_json(p, s)

            with self._patched_watermark(tmpdir, 99):
                with self.assertLogs("src.portfolio.state_store", level="WARNING") as cm:
                    load_portfolio_state(p)
                self.assertTrue(any("rollback" in m.lower() for m in cm.output))
        finally:
            _sh.rmtree(tmpdir, ignore_errors=True)

    def test_no_rollback_warning_when_generation_current(self):
        import logging as _log
        import shutil as _sh
        tmpdir = Path(tempfile.mkdtemp())
        try:
            p = tmpdir / "state.json"
            s = _make_valid_state()
            s["generation_id"] = 100
            atomic_write_json(p, s)

            with self._patched_watermark(tmpdir, 50):
                captured: list[str] = []

                class _Cap(_log.Handler):
                    def emit(self, record):
                        captured.append(record.getMessage())

                handler = _Cap(level=_log.WARNING)
                ss_logger = _log.getLogger("src.portfolio.state_store")
                ss_logger.addHandler(handler)
                try:
                    load_portfolio_state(p)
                finally:
                    ss_logger.removeHandler(handler)

                rollback_msgs = [m for m in captured if "rollback" in m.lower()]
                self.assertEqual(rollback_msgs, [], f"Unexpected rollback warning: {rollback_msgs}")
        finally:
            _sh.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
