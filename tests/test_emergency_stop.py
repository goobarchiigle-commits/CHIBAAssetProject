from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.live.order_ledger as order_ledger_module
import src.live.position_sync as position_sync_module
import src.run_morning_signal as morning_signal
from src.kabusapi.signal_bridge import AbortError, SignalBridge


def _workspace_path(name: str) -> Path:
    base_dir = Path(__file__).resolve().parents[1] / ".codex_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / name
    if path.exists():
        path.unlink()
    return path


def _result_with_orders(*orders):
    return SimpleNamespace(
        data_as_of="2026-04-17",
        n_universe=1,
        portfolio_summary={
            "current_positions": 0,
            "max_positions": 1,
            "open_slots": 1,
            "available_cash": 1000000,
        },
        signals=[],
        orders=list(orders),
        warnings=[],
        to_json=lambda: "{}",
    )


def _order(symbol: str, side: str = "BUY") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        side=side,
        qty=100,
        estimated_price=1000.0,
        estimated_amount=100000.0,
        reason="test",
    )


def _patch_live_main(monkeypatch, bridge: MagicMock, shadow_mode: bool = False):
    monkeypatch.setattr(
        morning_signal,
        "parse_args",
        lambda: Namespace(live=True, yes=True, no_save=True, output_dir="runtime", reconcile=False),
    )
    monkeypatch.setattr(morning_signal, "print_banner", lambda live: None)
    monkeypatch.setattr(morning_signal, "load_live_universe", lambda: {"7203.T": "AUTO"})
    monkeypatch.setattr(morning_signal, "check_api_connection", lambda: True)
    monkeypatch.setattr(morning_signal, "sync_portfolio_from_api", lambda: {"7203.T": 100})
    monkeypatch.setattr(morning_signal, "print_signals", MagicMock())
    monkeypatch.setattr(morning_signal, "save_signal_json", MagicMock())
    monkeypatch.setattr(morning_signal, "assert_live_mode_env", lambda: None)
    monkeypatch.setattr(morning_signal, "_fetch_broker_position_qty", lambda: {})
    monkeypatch.setattr(morning_signal, "mark_ordered", MagicMock())
    monkeypatch.setattr(morning_signal, "run_morning_smoke_test", lambda **kwargs: None)
    monkeypatch.setattr(
        morning_signal,
        "cfg",
        SimpleNamespace(
            live_execution=SimpleNamespace(shadow_mode=shadow_mode, shadow_mode_reason=""),
            fujiko=morning_signal.cfg.fujiko,
            mean_reversion=morning_signal.cfg.mean_reversion,
            portfolio=morning_signal.cfg.portfolio,
            risk=morning_signal.cfg.risk,
            risk_controls=morning_signal.cfg.risk_controls,
        ),
    )
    assert morning_signal.cfg.live_execution.shadow_mode is shadow_mode
    monkeypatch.setattr(
        "src.kabusapi.client.KabuClient",
        MagicMock(return_value=MagicMock(get_wallet_cash=MagicMock(return_value={"cash": 1000000}))),
    )
    monkeypatch.setenv("LIVE_MODE", "true")
    monkeypatch.setattr("src.kabusapi.signal_bridge.SignalBridge", MagicMock(return_value=bridge))
    monkeypatch.setattr(
        "src.live.execution_guard.check_execution_preconditions",
        lambda **kwargs: {"status": "OK"},
    )


def test_position_sync_fail_raises_abort():
    bridge = MagicMock()
    bridge.get_broker_positions.side_effect = ConnectionError("down")

    with pytest.raises(AbortError, match="failed to sync positions"):
        position_sync_module.sync_and_validate_state(bridge)


def test_position_sync_fail_no_order_call(monkeypatch):
    bridge = MagicMock()
    bridge.run.return_value = (_result_with_orders(_order("7203.T")), [_order("7203.T")])
    bridge._send_orders = MagicMock()
    generate_signals = bridge.run
    send_order = bridge._send_orders
    _patch_live_main(monkeypatch, bridge)
    monkeypatch.setattr(
        position_sync_module,
        "sync_and_validate_state",
        MagicMock(side_effect=AbortError("position_sync_fail", "sync failed")),
    )

    def fail_if_called(*args, **kwargs):
        pytest.fail("send_order should never be called after abort")

    send_order.side_effect = fail_if_called

    assert morning_signal.main() == 1
    assert send_order.call_count == 0
    assert generate_signals.call_count == 1


def test_duplicate_buy_raises_abort(monkeypatch):
    monkeypatch.setattr(order_ledger_module, "LEDGER_PATH", _workspace_path("emergency_stop_duplicate_buy.json"))
    ledger = order_ledger_module.OrderLedger()
    ledger.check_and_record("7203.T", "BUY", qty=100)

    with pytest.raises(AbortError, match="duplicate order detected: 7203.T BUY"):
        ledger.check_and_record("7203.T", "BUY", qty=100)


def test_duplicate_sell_raises_abort(monkeypatch):
    monkeypatch.setattr(order_ledger_module, "LEDGER_PATH", _workspace_path("emergency_stop_duplicate_sell.json"))
    ledger = order_ledger_module.OrderLedger()
    ledger.check_and_record("7203.T", "SELL", qty=100)

    with pytest.raises(AbortError, match="duplicate order detected: 7203.T SELL"):
        ledger.check_and_record("7203.T", "SELL", qty=100)


def test_duplicate_order_no_api_call(monkeypatch):
    bridge = MagicMock()
    first_order = _order("7203.T", "BUY")
    bridge.run.return_value = (_result_with_orders(first_order), [first_order])
    bridge._send_orders = MagicMock()
    generate_signals = bridge.run
    send_order = bridge._send_orders
    _patch_live_main(monkeypatch, bridge)
    ledger = MagicMock()
    ledger.daily_count.return_value = 0
    ledger.check_and_record.side_effect = AbortError("duplicate_order_detected", "duplicate buy")
    monkeypatch.setattr("src.live.order_ledger.OrderLedger", MagicMock(return_value=ledger))
    monkeypatch.setattr(position_sync_module, "sync_and_validate_state", lambda bridge: {"synced": False})

    def fail_if_called(*args, **kwargs):
        pytest.fail("send_order should never be called after abort")

    send_order.side_effect = fail_if_called

    assert morning_signal.main() == 1
    assert send_order.call_count == 0
    assert generate_signals.call_count == 1


def test_duplicate_order_stops_batch(monkeypatch):
    bridge = MagicMock()
    orders = [_order("7203.T", "BUY"), _order("6758.T", "BUY")]
    bridge.run.return_value = (_result_with_orders(*orders), orders)
    bridge._send_orders = MagicMock()
    generate_signals = bridge.run
    send_order = bridge._send_orders
    _patch_live_main(monkeypatch, bridge)
    ledger = MagicMock()
    ledger.daily_count.return_value = 0
    ledger.check_and_record.side_effect = AbortError("duplicate_order_detected", "duplicate buy")
    monkeypatch.setattr("src.live.order_ledger.OrderLedger", MagicMock(return_value=ledger))
    monkeypatch.setattr(position_sync_module, "sync_and_validate_state", lambda bridge: {"synced": False})

    assert morning_signal.main() == 1
    assert ledger.check_and_record.call_count == 1
    assert generate_signals.call_count == 1
    send_order.assert_not_called()
    assert send_order.call_count == 0


def test_portfolio_state_missing_raises_abort(tmp_path):
    bridge = SignalBridge.__new__(SignalBridge)
    bridge._state_file = tmp_path / "missing_portfolio_state.json"

    with pytest.raises(AbortError, match="portfolio state file is missing"):
        bridge._load_portfolio_state()


def test_portfolio_state_missing_no_fallback(tmp_path):
    bridge = SignalBridge.__new__(SignalBridge)
    bridge._state_file = tmp_path / "missing_portfolio_state.json"

    with pytest.raises(AbortError):
        bridge._load_portfolio_state()


def test_portfolio_state_missing_no_signal_call(monkeypatch):
    bridge = MagicMock()
    bridge.run = MagicMock()
    bridge._send_orders = MagicMock()
    generate_signals = bridge.run
    send_order = bridge._send_orders
    _patch_live_main(monkeypatch, bridge)
    monkeypatch.setattr(
        morning_signal,
        "sync_portfolio_from_api",
        MagicMock(side_effect=AbortError("portfolio_state_missing", "missing state")),
    )

    def fail_if_called(*args, **kwargs):
        pytest.fail("send_order should never be called after abort")

    send_order.side_effect = fail_if_called

    assert morning_signal.main() == 1
    assert generate_signals.call_count == 0
    assert send_order.call_count == 0


def test_no_fallback_after_abort(monkeypatch):
    bridge = MagicMock()
    bridge.run = MagicMock(side_effect=AbortError("portfolio_state_missing", "missing state"))
    bridge._send_orders = MagicMock()
    generate_signals = bridge.run
    send_order = bridge._send_orders
    _patch_live_main(monkeypatch, bridge)
    print_signals = MagicMock()
    monkeypatch.setattr(morning_signal, "print_signals", print_signals)

    def fail_if_called(*args, **kwargs):
        pytest.fail("send_order should never be called after abort")

    send_order.side_effect = fail_if_called

    assert morning_signal.main() == 1
    assert generate_signals.call_count == 1
    assert send_order.call_count == 0
    print_signals.assert_not_called()
    assert print_signals.call_count == 0


def test_live_mode_executes_orders(monkeypatch):
    bridge = MagicMock()
    bridge.run.return_value = (_result_with_orders(_order("7203.T")), [_order("7203.T")])
    bridge._send_orders = MagicMock(return_value=[
        {"success": True, "symbol": "7203.T", "side": "BUY", "qty": 100, "order_id": "TEST001"}
    ])
    bridge.pre_trade_risk_check.return_value = True

    _patch_live_main(monkeypatch, bridge, shadow_mode=False)

    ledger = MagicMock()
    ledger.daily_count.return_value = 0
    monkeypatch.setattr("src.live.order_ledger.OrderLedger", MagicMock(return_value=ledger))
    monkeypatch.setattr(position_sync_module, "sync_and_validate_state", lambda bridge: None)

    assert morning_signal.main() == 0
    assert bridge._send_orders.call_count == 1
