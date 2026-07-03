"""
tests/broker/test_live_equity_fetcher.py

LiveEquitySnapshot と fetch_live_equity の単体テスト。
KabuClient はモックに置き換える（ブローカーAPI非依存）。
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from src.broker.live_equity_fetcher import LiveEquitySnapshot, fetch_live_equity


# ─────────────────────────────────────────────────────────────────────────────
# ヘルパー: モック client 生成
# ─────────────────────────────────────────────────────────────────────────────

def _make_client(
    wallet_response: dict | Exception,
    positions_response: list | Exception,
) -> MagicMock:
    client = MagicMock()
    if isinstance(wallet_response, Exception):
        client.get_wallet_cash.side_effect = wallet_response
    else:
        client.get_wallet_cash.return_value = wallet_response
    if isinstance(positions_response, Exception):
        client.get_positions.side_effect = positions_response
    else:
        client.get_positions.return_value = positions_response
    return client


def _pos(symbol: str, qty: float, current_price: float = 0.0, avg_price: float = 0.0) -> dict:
    return {
        "Symbol": symbol,
        "LeavesQty": qty,
        "CurrentPrice": current_price,
        "Price": avg_price,
    }


# ─────────────────────────────────────────────────────────────────────────────
# T-1: 両API成功 → actual_equity = cash + market_value
# ─────────────────────────────────────────────────────────────────────────────

def test_both_api_success():
    client = _make_client(
        wallet_response={"StockAccountWallet": 1_000_000},
        positions_response=[
            _pos("7203", 100, current_price=3000.0),   # 300,000
            _pos("9984", 200, current_price=2500.0),   # 500,000
        ],
    )
    snap = fetch_live_equity(client)
    assert snap is not None
    assert snap.cash == 1_000_000.0
    assert snap.market_value == pytest.approx(800_000.0)
    assert snap.actual_equity == pytest.approx(1_800_000.0)
    assert snap.position_count == 2
    assert snap.source == "broker"


# ─────────────────────────────────────────────────────────────────────────────
# T-2: wallet API 失敗 → cash=0, source="positions_only"
# ─────────────────────────────────────────────────────────────────────────────

def test_wallet_fail_positions_ok():
    client = _make_client(
        wallet_response=RuntimeError("Connection refused"),
        positions_response=[_pos("7203", 100, current_price=3000.0)],
    )
    snap = fetch_live_equity(client)
    assert snap is not None
    assert snap.cash == 0.0
    assert snap.market_value == pytest.approx(300_000.0)
    assert snap.actual_equity == pytest.approx(300_000.0)
    assert snap.source == "positions_only"


# ─────────────────────────────────────────────────────────────────────────────
# T-3: positions API 失敗 → market_value=0, source="broker_partial"
# ─────────────────────────────────────────────────────────────────────────────

def test_positions_fail_wallet_ok():
    client = _make_client(
        wallet_response={"StockAccountWallet": 2_500_000},
        positions_response=RuntimeError("Timeout"),
    )
    snap = fetch_live_equity(client)
    assert snap is not None
    assert snap.cash == 2_500_000.0
    assert snap.market_value == 0.0
    assert snap.actual_equity == pytest.approx(2_500_000.0)
    assert snap.source == "broker_partial"


# ─────────────────────────────────────────────────────────────────────────────
# T-4: 両API失敗 → None 返却 (FAIL_OPEN)
# ─────────────────────────────────────────────────────────────────────────────

def test_both_api_fail_returns_none():
    client = _make_client(
        wallet_response=RuntimeError("wallet down"),
        positions_response=RuntimeError("positions down"),
    )
    snap = fetch_live_equity(client)
    assert snap is None


# ─────────────────────────────────────────────────────────────────────────────
# T-5: CurrentPrice=0 → Price(約定単価) にフォールバック
# ─────────────────────────────────────────────────────────────────────────────

def test_position_price_fallback_to_avg():
    client = _make_client(
        wallet_response={"StockAccountWallet": 0},
        positions_response=[
            _pos("7203", 100, current_price=0.0, avg_price=2800.0),
        ],
    )
    snap = fetch_live_equity(client)
    assert snap is not None
    assert snap.market_value == pytest.approx(280_000.0)


# ─────────────────────────────────────────────────────────────────────────────
# T-6: LeavesQty=0 の行はカウントしない
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_qty_excluded():
    client = _make_client(
        wallet_response={"StockAccountWallet": 1_000_000},
        positions_response=[
            _pos("7203", 0, current_price=3000.0),    # LeavesQty=0 → 除外
            _pos("9984", 100, current_price=2500.0),  # 250,000
        ],
    )
    snap = fetch_live_equity(client)
    assert snap is not None
    assert snap.position_count == 1
    assert snap.market_value == pytest.approx(250_000.0)


# ─────────────────────────────────────────────────────────────────────────────
# T-7: ポジションなし（現金のみ）→ market_value=0, position_count=0
# ─────────────────────────────────────────────────────────────────────────────

def test_no_positions():
    client = _make_client(
        wallet_response={"StockAccountWallet": 3_000_000},
        positions_response=[],
    )
    snap = fetch_live_equity(client)
    assert snap is not None
    assert snap.market_value == 0.0
    assert snap.position_count == 0
    assert snap.actual_equity == pytest.approx(3_000_000.0)
    assert snap.source == "broker"


# ─────────────────────────────────────────────────────────────────────────────
# T-8: StockAccountWallet フィールドが欠損 → wallet 失敗扱い
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_stock_account_wallet_field():
    client = _make_client(
        wallet_response={"MarginAccountWallet": 1_000_000},  # 違うフィールド名
        positions_response=[_pos("7203", 100, current_price=3000.0)],
    )
    snap = fetch_live_equity(client)
    # StockAccountWallet が None → ValueError → positions_only
    assert snap is not None
    assert snap.source == "positions_only"
    assert snap.cash == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# T-9: fetched_at フィールドに JST タイムスタンプが格納される
# ─────────────────────────────────────────────────────────────────────────────

def test_fetched_at_is_populated():
    client = _make_client(
        wallet_response={"StockAccountWallet": 1_000_000},
        positions_response=[],
    )
    snap = fetch_live_equity(client)
    assert snap is not None
    assert snap.fetched_at != ""
    assert "+09:00" in snap.fetched_at or "+0900" in snap.fetched_at


# ─────────────────────────────────────────────────────────────────────────────
# T-10: dataclass フィールド型チェック
# ─────────────────────────────────────────────────────────────────────────────

def test_snapshot_fields_are_correct_types():
    snap = LiveEquitySnapshot(
        cash=1_000_000.0,
        market_value=500_000.0,
        actual_equity=1_500_000.0,
        position_count=2,
        source="broker",
        fetched_at="2026-06-02T09:00:00+09:00",
    )
    assert isinstance(snap.cash, float)
    assert isinstance(snap.market_value, float)
    assert isinstance(snap.actual_equity, float)
    assert isinstance(snap.position_count, int)
    assert isinstance(snap.source, str)
    assert isinstance(snap.fetched_at, str)


# ─────────────────────────────────────────────────────────────────────────────
# T-11: get_positions が None を返す（空ポジション扱い）
# ─────────────────────────────────────────────────────────────────────────────

def test_positions_returns_none_treated_as_empty():
    """get_positions() が None を返す場合、`or []` で空リスト扱いになり market_value=0 で成功。"""
    client = _make_client(
        wallet_response={"StockAccountWallet": 2_000_000},
        positions_response=None,
    )
    snap = fetch_live_equity(client)
    # None → `raw_positions or []` → 空リスト → positions_ok=True, market_value=0
    assert snap is not None
    assert snap.cash == 2_000_000.0
    assert snap.market_value == 0.0
    assert snap.actual_equity == pytest.approx(2_000_000.0)
    assert snap.source == "broker"


# ─────────────────────────────────────────────────────────────────────────────
# T-12: 大きな入金後でも actual_equity が正確に計算される
# ─────────────────────────────────────────────────────────────────────────────

def test_large_deposit_reflected_in_actual_equity():
    client = _make_client(
        wallet_response={"StockAccountWallet": 4_000_000},  # 1M 入金後
        positions_response=[_pos("7203", 100, current_price=3000.0)],
    )
    snap = fetch_live_equity(client)
    assert snap is not None
    assert snap.actual_equity == pytest.approx(4_300_000.0)
    # eff_capital は rate-limit で制限されるが actual_equity は制限なし


# ─────────────────────────────────────────────────────────────────────────────
# T-13: _virtual_available_cash で deployable_capital が優先される
# ─────────────────────────────────────────────────────────────────────────────

def test_virtual_cash_prefers_deployable_capital():
    """signal_bridge._virtual_available_cash は deployable_capital > 0 時にそれを使う。"""
    from src.kabusapi.signal_bridge import SignalBridge
    import types

    bridge = object.__new__(SignalBridge)
    bridge.capital = 3_000_000
    bridge.deployable_capital = 4_500_000  # 実資産ベース
    bridge.max_positions = 3

    # 2ポジション保有 → 1スロット空き
    cash = bridge._virtual_available_cash({"7203.T": {}, "9984.T": {}})
    expected = 4_500_000 / 3 * 1
    assert cash == pytest.approx(expected)


def test_virtual_cash_falls_back_to_capital_when_deployable_zero():
    """deployable_capital=0 の場合は capital にフォールバックする。"""
    from src.kabusapi.signal_bridge import SignalBridge

    bridge = object.__new__(SignalBridge)
    bridge.capital = 3_000_000
    bridge.deployable_capital = 0.0
    bridge.max_positions = 3

    cash = bridge._virtual_available_cash({})  # 0 ポジション → 3 スロット空き
    expected = 3_000_000 / 3 * 3
    assert cash == pytest.approx(expected)
