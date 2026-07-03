"""
tests/test_monitoring.py
monitoring 層 全機能テスト
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import pytest

from src.monitoring.audit import compute_metrics
from src.monitoring.controller import (
    COOLDOWN_SEC, DRIFT, MAX_DD_HARD, MIN_ACTIVITY, MIN_MULTIPLIER,
    RISK_HARD, RISK_SOFT, RISK_SOFT_RECOV, SLIP_SOFT,
    apply_controller, get_system_status, make_default_state,
)
from src.monitoring.gate import RISK_HARD_CAP, SLIP_HARD_CAP, pre_trade_gate


# ── ヘルパー ─────────────────────────────────────────────────────────────────

def mk_df(pnl_list, risk=0.015, slip_r=0.001, slip_e=0.001):
    n = len(pnl_list)
    return pd.DataFrame({
        "pnl":           pnl_list,
        "expected_risk": [risk] * n,
        "slippage_real": [slip_r] * n,
        "slippage_est":  [slip_e] * n,
    })


def ok_metrics(**overrides):
    m = compute_metrics(mk_df([0.01] * 5))
    m.update(overrides)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 1. audit
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudit:
    def test_ok(self):
        m = compute_metrics(mk_df([0.01, 0.02, -0.005, 0.015]))
        assert m["status"] == "OK"
        assert m["n_trades"] == 4
        assert m["max_dd"] <= 0
        assert m["slip_diff"] == 0.0
        assert m["risk_ratio"] > 0

    def test_nan_halt_corrupt(self):
        df = mk_df([0.01, float("nan"), 0.02])
        m = compute_metrics(df)
        assert m["status"] == "HALT_CORRUPT"
        assert "pnl" in m["corrupt_reason"]

    def test_missing_col_halt_corrupt(self):
        df = pd.DataFrame({"pnl": [0.01], "expected_risk": [0.01]})
        m = compute_metrics(df)
        assert m["status"] == "HALT_CORRUPT"

    def test_zero_risk_guard(self):
        m = compute_metrics(mk_df([0.01, 0.02], risk=0.0))
        assert m["risk_ratio"] == 0.0

    def test_slip_diff(self):
        m = compute_metrics(mk_df([0.01], slip_r=0.002, slip_e=0.001))
        assert abs(m["slip_diff"] - 0.001) < 1e-9

    def test_empty_df(self):
        m = compute_metrics(mk_df([]))
        assert m["n_trades"] == 0
        assert m["status"] == "OK"

    def test_max_dd_negative(self):
        m = compute_metrics(mk_df([0.02, 0.01, -0.05, 0.01]))
        assert m["max_dd"] < 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. controller
# ═══════════════════════════════════════════════════════════════════════════════

class TestController:
    def test_halt_latch_max_dd(self):
        s = make_default_state()
        m = ok_metrics(max_dd=-0.20)
        s2 = apply_controller(s, m, ts=1000.0)
        assert s2["halted"] is True
        assert s2["global_multiplier"] == 0.0
        assert s2["halt_reason"] is not None

    def test_halt_irreversible(self):
        s = make_default_state()
        m_bad = ok_metrics(max_dd=-0.20)
        s_halt = apply_controller(s, m_bad, ts=1000.0)
        s_try  = apply_controller(s_halt, ok_metrics(), ts=99999.0)
        assert s_try["halted"] is True
        assert s_try["global_multiplier"] == 0.0

    def test_halt_latch_risk_hard(self):
        s = make_default_state()
        m = ok_metrics(risk_ratio=RISK_HARD + 0.1)
        s2 = apply_controller(s, m, ts=1000.0)
        assert s2["halted"] is True

    def test_halt_corrupt(self):
        s = make_default_state()
        m = {"status": "HALT_CORRUPT", "corrupt_reason": "NaN", "max_dd": 0.0,
             "risk_ratio": 0.0, "slip_diff": 0.0, "n_trades": 0}
        s2 = apply_controller(s, m, ts=1000.0)
        assert s2["halted"] is True
        assert s2["mode"] == "HALT_CORRUPT"

    def test_degraded(self):
        s = make_default_state()
        m = ok_metrics(risk_ratio=RISK_SOFT + 0.1)
        s2 = apply_controller(s, m, ts=1000.0)
        assert s2["mode"] == "DEGRADED"
        assert s2["global_multiplier"] < 1.0
        assert s2["risk_scale"] < 1.0

    def test_adjust(self):
        s = make_default_state()
        m = ok_metrics(slip_diff=SLIP_SOFT + 0.001)
        s2 = apply_controller(s, m, ts=1000.0)
        assert s2["mode"] == "ADJUST"
        assert s2["impact_coeff"] > 1.0

    def test_hysteresis_recovery(self):
        s = make_default_state()
        m_deg  = ok_metrics(risk_ratio=RISK_SOFT + 0.1)
        s_deg  = apply_controller(s, m_deg, ts=1000.0)
        s_free = dict(s_deg, last_adjust_ts=0.0)
        m_rec  = ok_metrics(risk_ratio=RISK_SOFT_RECOV - 0.01)
        s_ok   = apply_controller(s_free, m_rec, ts=99999.0)
        assert s_ok["mode"] == "OK"

    def test_cooldown(self):
        s = make_default_state()
        m_adj = ok_metrics(slip_diff=SLIP_SOFT + 0.001)
        s_adj = apply_controller(s, m_adj, ts=1000.0)
        s_cool = apply_controller(s_adj, ok_metrics(), ts=1000.0 + 60)
        assert s_cool["mode"] == "COOLDOWN"

    def test_drift_clamp(self):
        s = dict(make_default_state(), risk_scale=0.1)  # far below base*(1-DRIFT)=0.7
        s2 = apply_controller(s, ok_metrics(), ts=99999.0)
        assert s2["risk_scale"] >= (1.0 * (1 - DRIFT))

    def test_min_activity_guarantee(self):
        s = dict(make_default_state(), global_multiplier=0.1)
        m = ok_metrics(n_trades=MIN_ACTIVITY - 1)
        s2 = apply_controller(s, m, ts=99999.0)
        assert s2["global_multiplier"] >= MIN_MULTIPLIER

    def test_mean_revert(self):
        s = dict(make_default_state(), risk_scale=0.8)
        s2 = apply_controller(s, ok_metrics(), ts=99999.0)
        assert s2["risk_scale"] > 0.8  # 1.0 方向に引き寄せられる


# ═══════════════════════════════════════════════════════════════════════════════
# 3. gate
# ═══════════════════════════════════════════════════════════════════════════════

class TestGate:
    def _s(self): return make_default_state()

    def test_ok(self):
        ok, _ = pre_trade_gate(self._s(), ok_metrics())
        assert ok is True

    def test_halted_blocks(self):
        s = dict(self._s(), halted=True, halt_reason="test")
        ok, reason = pre_trade_gate(s, ok_metrics())
        assert ok is False
        assert "HALT" in reason

    def test_risk_hard_cap_blocks(self):
        ok, r = pre_trade_gate(self._s(), ok_metrics(risk_ratio=RISK_HARD_CAP + 0.1))
        assert ok is False
        assert "RISK_HARD_CAP" in r

    def test_slip_hard_cap_blocks(self):
        ok, r = pre_trade_gate(self._s(), ok_metrics(slip_diff=SLIP_HARD_CAP + 0.001))
        assert ok is False
        assert "SLIP_HARD_CAP" in r

    def test_soft_passes(self):
        ok, _ = pre_trade_gate(
            self._s(),
            ok_metrics(risk_ratio=RISK_SOFT + 0.1, slip_diff=SLIP_SOFT + 0.001),
        )
        assert ok is True

    def test_halt_corrupt_blocks(self):
        m = {"status": "HALT_CORRUPT", "corrupt_reason": "NaN", "max_dd": 0.0,
             "risk_ratio": 0.0, "slip_diff": 0.0, "n_trades": 0}
        ok, r = pre_trade_gate(self._s(), m)
        assert ok is False
        assert "HALT_CORRUPT" in r


# ═══════════════════════════════════════════════════════════════════════════════
# 4. watchdog import
# ═══════════════════════════════════════════════════════════════════════════════

class TestWatchdog:
    def test_importable(self):
        from src.monitoring.watchdog import run_watchdog, _log_age_sec, _read_n_trades, _write_kill
        assert callable(run_watchdog)

    def test_log_age_nonexistent(self, tmp_path):
        from src.monitoring.watchdog import _log_age_sec
        assert _log_age_sec(tmp_path / "no_file.csv") is None

    def test_log_age_existing(self, tmp_path):
        import time
        from src.monitoring.watchdog import _log_age_sec
        p = tmp_path / "log.csv"
        p.write_text("dummy")
        age = _log_age_sec(p)
        assert age is not None and age >= 0

    def test_read_n_trades_missing(self, tmp_path):
        from src.monitoring.watchdog import _read_n_trades
        assert _read_n_trades(tmp_path / "missing.json") is None

    def test_read_n_trades_ok(self, tmp_path):
        import json
        from src.monitoring.watchdog import _read_n_trades
        p = tmp_path / "state.json"
        p.write_text(json.dumps({"n_trades": 7}))
        assert _read_n_trades(p) == 7

    def test_write_kill(self, tmp_path):
        import json
        from src.monitoring.watchdog import _write_kill
        p = tmp_path / "state.json"
        _write_kill(p, "TEST_REASON")
        data = json.loads(p.read_text())
        assert data["halted"] is True
        assert data["global_multiplier"] == 0.0
        assert "TEST_REASON" in data["halt_reason"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. get_system_status 全 7 状態
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemStatus:
    @pytest.mark.parametrize("mode,halted,expected", [
        ("OK",           False, "OK"),
        ("ADJUST",       False, "ADJUST"),
        ("DEGRADED",     False, "DEGRADED"),
        ("COOLDOWN",     False, "COOLDOWN"),
        ("HALT",         True,  "HALT"),
        ("HALT_CORRUPT", True,  "HALT_CORRUPT"),
    ])
    def test_all_modes(self, mode, halted, expected):
        s = {**make_default_state(), "mode": mode, "halted": halted}
        assert get_system_status(s) == expected

    def test_watchdog_stale_maps_to_halt(self):
        s = {**make_default_state(), "mode": "HALT", "halted": True,
             "halt_reason": "watchdog: STALE 3700s"}
        assert get_system_status(s) == "HALT"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. pipeline.py 統合: インポート確認
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    def test_monitoring_symbols_in_pipeline(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pipeline", Path(__file__).resolve().parents[1] / "pipeline.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # ソースを読んでキーワード確認（実行はしない）
        src = (Path(__file__).resolve().parents[1] / "pipeline.py").read_text(encoding="utf-8")
        assert "_MON_AVAILABLE" in src
        assert "pre_trade_gate" in src
        assert "compute_metrics" in src
        assert "apply_controller" in src
        assert "save_state" in src

    def test_monitoring_import_flag(self):
        import pipeline as _pl
        assert hasattr(_pl, "_MON_AVAILABLE")
        assert _pl._MON_AVAILABLE is True


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Config-driven controller / gate  (追加テスト)
# ═══════════════════════════════════════════════════════════════════════════════

from src.monitoring.controller import can_recover, sanitize_state
from src.monitoring.audit import detect_strategy_degradation, sanitize_metrics


def test_recovery_block_time():
    state = {"halted": True, "halt_ts": 0}
    metrics = {"risk_ratio": 0.5, "max_dd": -0.01, "slip_diff": 0.0}
    cfg = {"RECOVERY_RISK": 1.0, "RECOVERY_DD": 0.1, "RECOVERY_SLIP": 0.01, "MIN_HALTED_TIME": 100}
    assert not can_recover(state, metrics, 50, cfg)


def test_recovery_success():
    state = {"halted": True, "halt_ts": 0}
    metrics = {"risk_ratio": 0.5, "max_dd": -0.01, "slip_diff": 0.0}
    cfg = {"RECOVERY_RISK": 1.0, "RECOVERY_DD": 0.1, "RECOVERY_SLIP": 0.01, "MIN_HALTED_TIME": 10}
    assert can_recover(state, metrics, 20, cfg)


def test_degradation_detection():
    df = pd.DataFrame({"pnl": [-1, -1, -1, 0]})
    cfg = {"PNL_FLOOR": 0.1, "MIN_WIN_RATE": 0.5}
    assert detect_strategy_degradation(df, cfg)


def test_gate_rate_limit():
    order = {"side": "buy", "price": 100, "mid_price": 100}
    state = {"orders_last_min": 10}
    cfg = {"MAX_ORDERS_PER_MIN": 5}
    ok, reason = pre_trade_gate(order, state, {}, cfg)
    assert not ok and reason == "RATE_LIMIT"


def test_gate_bad_price():
    order = {"side": "buy", "price": 120, "mid_price": 100}
    state: dict = {}
    cfg = {"MAX_PRICE_DEVIATION": 0.1}
    ok, reason = pre_trade_gate(order, state, {}, cfg)
    assert not ok and reason == "BAD_PRICE"


def test_gate_overexposure():
    order = {"side": "buy", "price": 100, "mid_price": 100}
    state = {"last_side": "buy", "same_side_count": 10}
    cfg = {"MAX_SAME_SIDE": 5}
    ok, reason = pre_trade_gate(order, state, {}, cfg)
    assert not ok and reason == "OVEREXPOSURE"


def test_halt_sets_multiplier_zero():
    state: dict = {"halted": False, "global_multiplier": 1.0}
    metrics = {"risk_ratio": 10.0, "max_dd": -1.0, "slip_diff": 0.0}
    cfg = {"MAX_DD_HARD": 0.2, "RISK_HARD": 5.0}
    state, mode = apply_controller(state, metrics, 0, cfg)
    assert state["global_multiplier"] == 0.0


def test_recover_sets_reduced_multiplier():
    state: dict = {"halted": True, "halt_ts": 0}
    metrics = {"risk_ratio": 0.1, "max_dd": -0.01, "slip_diff": 0.0}
    cfg = {
        "RECOVERY_RISK": 1.0, "RECOVERY_DD": 0.1, "RECOVERY_SLIP": 0.01,
        "MIN_HALTED_TIME": 1, "RECOVERY_MULTIPLIER": 0.2,
    }
    state, mode = apply_controller(state, metrics, 10, cfg)
    assert state["global_multiplier"] <= cfg["RECOVERY_MULTIPLIER"]


def test_multiplier_bounds():
    state: dict = {"halted": False, "global_multiplier": 2.0}
    metrics = {"risk_ratio": 0.1, "max_dd": -0.01, "slip_diff": 0.0}
    cfg = {"RISK_SOFT": 1.0, "SLIP_SOFT": 1.0}
    state, _ = apply_controller(state, metrics, 0, cfg)
    assert 0.0 <= state["global_multiplier"] <= 1.0


def test_state_corruption_halts():
    state: dict = {"halted": False, "global_multiplier": float("nan")}
    metrics = {"risk_ratio": 0.1, "max_dd": -0.01, "slip_diff": 0.0}
    cfg = {"RISK_SOFT": 1.0, "SLIP_SOFT": 1.0}
    state, mode = apply_controller(state, metrics, 0, cfg)
    assert state["halted"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
