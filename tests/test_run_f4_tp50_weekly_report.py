"""
tests/test_run_f4_tp50_weekly_report.py
src/run_f4_tp50_weekly_report.py の純粋関数・レポート生成を検証する。
kabuステーションAPIには一切接続しない（stub clientのみ使用）。sendorderは絶対に呼ばない
（.claude/skills/notification-design.md §10 テスト方針準拠）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.run_f4_tp50_weekly_report as wr


class _StubClient:
    """get_orders/get_board/get_positionsのみ実装。send_order等の発注メソッドは
    一切持たない（実装しないことで、誤って呼ばれればAttributeErrorで即座に検知できる）。"""

    def __init__(self, orders=None, board_names=None, positions=None):
        self._orders = orders or []
        self._board_names = board_names or {}
        self._positions = positions or []

    def get_orders(self, only_open=False):
        return self._orders

    def get_board(self, symbol4):
        class _Board:
            pass
        b = _Board()
        b.symbol_name = self._board_names.get(symbol4, "")
        return b

    def get_positions(self):
        return self._positions


def _order_with_fill(order_id, price, qty):
    return {
        "ID": order_id,
        "Details": [
            {"SeqNum": 1, "RecType": 1, "Price": 0.0},
            {"SeqNum": 8, "RecType": 8, "Price": price, "Qty": qty, "ExecutionDay": "2026-08-20T09:38:17+09:00"},
        ],
    }


def _run(live=True, exits_detail=None, funded_detail=None, order_submission_results=None,
         decisions=None, run_started_at=None, scheduled_trigger_hhmm=None,
         last_equity=None, metadata_warnings=None, cash_source="broker_live",
         risk_gate=None):
    return {
        "live": live, "signal_date": "2026-08-17", "entry_date": "2026-08-18",
        "exits_intended": len(exits_detail or []),
        "sizing_breakdown": {"funded_total": len(funded_detail or []), "capital_exhausted_skip": 0,
                              "already_held_skip": 0},
        "exits_detail": exits_detail or [], "funded_detail": funded_detail or [],
        "order_submission_results": order_submission_results,
        "score_replacement": {"enabled": True, "decisions": decisions or []},
        "entry_freeze_enabled": False, "risk_gate": risk_gate or {"recommendation": "NORMAL", "dd": -0.02},
        "run_started_at": run_started_at, "scheduled_trigger_hhmm": scheduled_trigger_hhmm,
        "last_equity": last_equity, "metadata_warnings": metadata_warnings or [], "cash_source": cash_source,
    }


def _day_row(date, run):
    rep = wr._fmt_run(run)
    rep["date"] = date
    rep["n_runs"] = 1
    return rep


# ── formatting helpers ──────────────────────────────────────────────────
def test_fmt_yen_and_pct():
    assert wr._fmt_yen(1234) == "¥1,234"
    assert wr._fmt_yen(-500, signed=True) == "¥-500"
    assert wr._fmt_yen(500, signed=True) == "+¥500"
    assert wr._fmt_yen(None) == "N/A"
    assert wr._fmt_pct(0.0231) == "+2.31%"
    assert wr._fmt_pct(-0.05) == "-5.00%"
    assert wr._fmt_pct(None) == "N/A"


def test_exit_reason_bucket_mapping():
    assert wr._exit_reason_bucket("trailing_touch") == "T15 STOP"
    assert wr._exit_reason_bucket("trailing_gap_open") == "T15 STOP"
    assert wr._exit_reason_bucket("target_touch") == "TP50 TARGET"
    assert wr._exit_reason_bucket("score_replacement") == "REPLACEMENT"
    assert wr._exit_reason_bucket("something_else") == "Other"


# ── _fetch_actual_fill: GET専用、sendorderは絶対に呼ばれない ──────────────
def test_fetch_actual_fill_never_calls_send_order():
    client = _StubClient(orders=[_order_with_fill("ORD-1", 1232.0, 100.0)])
    assert not hasattr(client, "send_order")  # stubにsendorder系メソッドが存在しないことの保証
    price, qty = wr._fetch_actual_fill(client, "ORD-1")
    assert price == 1232.0
    assert qty == 100.0


def test_fetch_actual_fill_returns_none_for_unconfirmed():
    client = _StubClient(orders=[])
    price, qty = wr._fetch_actual_fill(client, "ORD-MISSING")
    assert price is None and qty is None


def test_fetch_actual_fill_returns_none_for_dry_run_simulated():
    client = _StubClient(orders=[_order_with_fill("DRY_RUN_SIMULATED", 100.0, 100.0)])
    price, qty = wr._fetch_actual_fill(client, "DRY_RUN_SIMULATED")
    assert price is None and qty is None


# ── collect_weekly_trades: 実約定ベースのP/L集計 ────────────────────────
def test_collect_weekly_trades_computes_realized_pnl_from_actual_fill():
    run = _run(
        exits_detail=[{"code": "93440", "exit_reason": "trailing_touch", "entry_price": 1224.0, "qty": 100}],
        order_submission_results=[
            {"symbol": "93440", "side": "SELL", "qty": 100, "success": True, "order_id": "ORD-SELL-1"},
        ],
    )
    client = _StubClient(orders=[_order_with_fill("ORD-SELL-1", 1232.0, 100.0)])
    week = wr.collect_weekly_trades([_day_row("2026-08-20", run)], client)
    assert week["sell_success_count"] == 1
    assert len(week["closed_trades"]) == 1
    t = week["closed_trades"][0]
    assert t["pnl"] == pytest.approx(800.0)  # (1232-1224)*100
    assert week["exit_bucket_counts"]["T15 STOP"] == 1


def test_collect_weekly_trades_excludes_trade_when_entry_price_missing():
    """旧形式ログ(entry_price未収録)由来のSELLはEXIT BREAKDOWNには数えるが、
    P/L集計（closed_trades）からは除外する（推測値で埋めない）。"""
    run = _run(
        exits_detail=[],  # 旧形式: entry_priceを含むexits_detailが存在しない
        order_submission_results=[
            {"symbol": "93440", "side": "SELL", "qty": 100, "success": True, "order_id": "ORD-SELL-2"},
        ],
    )
    client = _StubClient(orders=[_order_with_fill("ORD-SELL-2", 1232.0, 100.0)])
    week = wr.collect_weekly_trades([_day_row("2026-08-20", run)], client)
    assert week["sell_success_count"] == 1
    assert len(week["closed_trades"]) == 0  # entry_price不明のため除外


def test_collect_weekly_trades_excludes_trade_when_actual_fill_unconfirmed():
    run = _run(
        exits_detail=[{"code": "93440", "exit_reason": "trailing_touch", "entry_price": 1224.0, "qty": 100}],
        order_submission_results=[
            {"symbol": "93440", "side": "SELL", "qty": 100, "success": True, "order_id": "ORD-MISSING"},
        ],
    )
    client = _StubClient(orders=[])  # 約定確認不能
    week = wr.collect_weekly_trades([_day_row("2026-08-20", run)], client)
    assert len(week["closed_trades"]) == 0


def test_collect_weekly_trades_counts_replacement_bucket():
    run = _run(
        exits_detail=[{"code": "93440", "exit_reason": "trailing_touch", "entry_price": 1000.0, "qty": 100}],
        order_submission_results=[
            {"symbol": "93440", "side": "SELL", "qty": 100, "success": True, "order_id": "ORD-R1"},
            {"symbol": "48260", "side": "BUY", "qty": 100, "success": True, "order_id": "ORD-R2"},
        ],
        decisions=[{"sold_code": "93440", "candidate_code": "48260", "decision": "BUY_FILLED",
                    "candidate_score": 71.5, "holding_score": 48.2}],
    )
    client = _StubClient(orders=[_order_with_fill("ORD-R1", 1000.0, 100.0)])
    week = wr.collect_weekly_trades([_day_row("2026-08-20", run)], client)
    assert week["replacement_count"] == 1
    assert week["exit_bucket_counts"]["REPLACEMENT"] == 1
    assert week["buy_count"] == 1


def test_collect_weekly_trades_dry_run_days_are_ignored():
    run = _run(live=False, exits_detail=[{"code": "1301", "exit_reason": "trailing_touch",
                                          "entry_price": 100.0, "qty": 100}])
    client = _StubClient()
    week = wr.collect_weekly_trades([_day_row("2026-08-20", run)], client)
    assert week["sell_success_count"] == 0
    assert week["buy_count"] == 0


def test_collect_weekly_trades_counts_order_failures_and_metadata_warnings():
    run = _run(
        order_submission_results=[{"symbol": "1301", "side": "BUY", "qty": 100, "success": False,
                                    "order_id": None, "error": "HTTP 400"}],
        metadata_warnings=["metadata mismatch: X"],
        cash_source="unavailable_dry_run_degraded",
    )
    week = wr.collect_weekly_trades([_day_row("2026-08-20", run)], _StubClient())
    assert week["order_fail_count"] == 1
    assert week["metadata_warning_count"] == 1
    assert week["api_error_count"] == 1


def test_scheduler_delayed_detection():
    delayed = _run(run_started_at="2026-08-20 09:36:11", scheduled_trigger_hhmm="08:49")
    on_time = _run(run_started_at="2026-08-20 08:50:00", scheduled_trigger_hhmm="08:49")
    assert wr._run_was_scheduler_delayed(delayed) is True
    assert wr._run_was_scheduler_delayed(on_time) is False


# ── _week_equity_bounds: データ不足時はNoneを返す（捏造しない） ────────────
def test_week_equity_bounds_returns_none_when_no_data():
    rows = [_day_row("2026-08-18", _run(last_equity=None))]
    assert wr._week_equity_bounds(rows) == (None, None, None)


def test_week_equity_bounds_uses_first_and_last_available():
    rows = [
        _day_row("2026-08-18", _run(last_equity=3000000.0)),
        _day_row("2026-08-19", _run(last_equity=None)),
        _day_row("2026-08-20", _run(last_equity=3100000.0, risk_gate={"recommendation": "NORMAL", "dd": -0.03})),
    ]
    start, end, dd = wr._week_equity_bounds(rows)
    assert start == 3000000.0 and end == 3100000.0
    assert dd == pytest.approx(-0.03)


# ── _topix_return: 実データ不使用（tmp_path上の合成parquetでpatch） ─────────
def test_topix_return_computes_close_to_close(tmp_path, monkeypatch):
    idx = pd.date_range("2026-08-17", "2026-08-20", freq="B")
    df = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                        "Close": [4200.0, 4180.0, 4150.0, 4030.0]}, index=idx)
    p = tmp_path / "0000.parquet"
    df.to_parquet(p)
    monkeypatch.setattr(wr, "TOPIX_PARQUET", p)
    ret = wr._topix_return("2026-08-17", "2026-08-20")
    assert ret == pytest.approx((4030.0 - 4200.0) / 4200.0)


def test_topix_return_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "TOPIX_PARQUET", tmp_path / "does_not_exist.parquet")
    assert wr._topix_return("2026-08-17", "2026-08-20") is None


# ── _current_holdings: kabu API + portfolio_state（stub、実接続なし） ──────
def test_current_holdings_matches_tp50_tagged_positions_only(monkeypatch):
    class _FakeState:
        def get(self, key, default=None):
            return {
                "position_strategy_types": {"93440": "f4_tp50", "99990": "f4_tp30"},
                "position_entry_dates": {"93440": "2026-08-19"},
                "position_entry_prices": {"93440": 1224.0},
            }.get(key, default)

    def _fake_load_portfolio_state():
        return _FakeState(), None

    monkeypatch.setattr("src.portfolio.state_store.load_portfolio_state", _fake_load_portfolio_state)
    client = _StubClient(positions=[
        {"Symbol": "9344", "SymbolName": "アクシスコンサルティング", "LeavesQty": 100.0, "CurrentPrice": 1299.0},
        {"Symbol": "9999", "SymbolName": "TP30の別銘柄", "LeavesQty": 100.0, "CurrentPrice": 500.0},
    ])
    holdings = wr._current_holdings(client)
    assert len(holdings) == 1
    assert holdings[0]["code"] == "93440"
    assert holdings[0]["pnl_pct"] == pytest.approx((1299.0 - 1224.0) / 1224.0)


def test_current_holdings_empty_when_client_none():
    assert wr._current_holdings(None) == []


# ── build_report: セクション構成・0件時の崩れなし ───────────────────────
def test_build_report_zero_activity_all_sections_present(monkeypatch):
    monkeypatch.setattr(wr, "_topix_return", lambda a, b: None)
    monkeypatch.setattr(wr, "_current_holdings", lambda client: [])
    daily_runs = {"2026-08-20": [_run(live=True, order_submission_results=[])]}
    full_md, email_text, stats = wr.build_report(daily_runs, 7, client=None)
    for section in ("【PERFORMANCE】", "【BENCHMARK】", "【TRADING】", "【TOP CONTRIBUTORS】",
                     "【TOP LOSERS】", "【CURRENT HOLDINGS】", "【EXIT BREAKDOWN】",
                     "【SCORE REPLACEMENT】", "【RISK】", "【SYSTEM HEALTH】"):
        assert section in email_text
    assert "SELL            0" in email_text
    assert "なし" in email_text  # TOP CONTRIBUTORS/LOSERS/HOLDINGSのいずれか


def test_build_report_sell_count_matches_exit_breakdown_total(monkeypatch):
    """SELL件数（TRADING）とEXIT BREAKDOWNの合計は必ず一致する
    （P/L算出不能な取引がいてもEXIT BREAKDOWN側の分類は行うため）。"""
    monkeypatch.setattr(wr, "_topix_return", lambda a, b: None)
    monkeypatch.setattr(wr, "_current_holdings", lambda client: [])
    run = _run(
        exits_detail=[],  # 旧形式扱い（entry_price不明）
        order_submission_results=[{"symbol": "93440", "side": "SELL", "qty": 100,
                                    "success": True, "order_id": "ORD-X"}],
    )
    daily_runs = {"2026-08-20": [run]}
    full_md, email_text, stats = wr.build_report(daily_runs, 7, client=_StubClient(orders=[]))
    assert "SELL            1" in email_text
    total_bucket = sum(int(line.split()[-1]) for line in email_text.splitlines()
                        if any(line.startswith(b) for b in ("T15 STOP", "TP50 TARGET", "REPLACEMENT", "Other")))
    assert total_bucket == 1
    assert "P/L集計対象外" in email_text


def test_build_report_benchmark_shows_unavailable_indices(monkeypatch):
    monkeypatch.setattr(wr, "_topix_return", lambda a, b: -0.0231)
    monkeypatch.setattr(wr, "_current_holdings", lambda client: [])
    daily_runs = {"2026-08-20": [_run(live=True, order_submission_results=[])]}
    full_md, email_text, stats = wr.build_report(daily_runs, 7, client=None)
    assert "TOPIX          -2.31%" in email_text
    assert "日経平均       データ未取得" in email_text
    assert "S&P500         データ未取得" in email_text


def test_build_report_top_contributors_and_losers_sorted(monkeypatch):
    monkeypatch.setattr(wr, "_topix_return", lambda a, b: None)
    monkeypatch.setattr(wr, "_current_holdings", lambda client: [])
    run = _run(
        exits_detail=[
            {"code": "1111", "exit_reason": "target_touch", "entry_price": 1000.0, "qty": 100},
            {"code": "2222", "exit_reason": "trailing_touch", "entry_price": 1000.0, "qty": 100},
        ],
        order_submission_results=[
            {"symbol": "1111", "side": "SELL", "qty": 100, "success": True, "order_id": "ORD-WIN"},
            {"symbol": "2222", "side": "SELL", "qty": 100, "success": True, "order_id": "ORD-LOSS"},
        ],
    )
    client = _StubClient(orders=[
        _order_with_fill("ORD-WIN", 1500.0, 100.0),   # +¥50,000
        _order_with_fill("ORD-LOSS", 800.0, 100.0),   # -¥20,000
    ])
    daily_runs = {"2026-08-20": [run]}
    full_md, email_text, stats = wr.build_report(daily_runs, 7, client=client)
    assert stats["win_rate"] == pytest.approx(0.5)
    contributors_block = email_text.split("【TOP CONTRIBUTORS】")[1].split("【TOP LOSERS】")[0]
    losers_block = email_text.split("【TOP LOSERS】")[1].split("【CURRENT HOLDINGS】")[0]
    assert "1111" in contributors_block
    assert "2222" in losers_block


def test_build_report_never_sends_real_orders(monkeypatch):
    """stub clientはsend_orderを持たない = 実装レベルで発注不可能であることの保証。"""
    monkeypatch.setattr(wr, "_topix_return", lambda a, b: None)
    monkeypatch.setattr(wr, "_current_holdings", lambda client: [])
    client = _StubClient(orders=[])
    assert not hasattr(client, "send_order")
    daily_runs = {"2026-08-20": [_run(live=True, order_submission_results=[])]}
    wr.build_report(daily_runs, 7, client=client)  # must not raise / must not attempt to order
