"""
tests/test_run_live_signal_f4_tp50.py
src/run_live_signal_f4_tp50.py の純粋関数（Exit判定・100株固定sizing・重複防止判定・
entry metadata永続化）を検証する。kabuステーションAPI・実ファイルI/Oには一切依存しない。

Lineage: fork of tests/test_run_live_signal_f4_tp30.py — same test structure,
STRATEGY_TYPE="f4_tp50", src.f4_tp50 modules.

NOTE: src/run_live_signal_f4_tp50.py はモジュールトップレベルで assert_execution_context()
を呼ぶ。既存のTP30/E5テストと同じ手法で、import前に sys.argv[0] を一時的にスプーフィングする。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _import_tp50_module():
    _orig_argv0 = sys.argv[0]
    sys.argv[0] = str(_ROOT / "run_live_signal_f4_tp50.py")
    try:
        import src.run_live_signal_f4_tp50 as tp50
        return tp50
    finally:
        sys.argv[0] = _orig_argv0


_tp50 = _import_tp50_module()


@pytest.fixture(autouse=True)
def _isolate_audit_sidecar(tmp_path, monkeypatch):
    """apply_fill_metadata_updates() は _append_entry_fill_audit() 経由で
    AUDIT_DIR/entry_fill_audit.jsonl に追記する。AUDIT_DIRを本番runtime/f4_tp50への
    参照のまま放置すると、テスト実行のたびに本番監査サイドカーへfixtureデータ
    (symbol=1301, broker_order_id=ORDER-1等)が混入する
    (2026-08-20 Production Gate監査で発見)。全テストでtmp_pathへ隔離する。"""
    monkeypatch.setattr(_tp50, "AUDIT_DIR", tmp_path)


FIXED_LOT_SIZE = _tp50.FIXED_LOT_SIZE
STRATEGY_TYPE = _tp50.STRATEGY_TYPE
OrderInstruction = _tp50.OrderInstruction
evaluate_buy_sizing = _tp50.evaluate_buy_sizing
evaluate_exits = _tp50.evaluate_exits
apply_fill_metadata_updates = _tp50.apply_fill_metadata_updates
compute_order_submission_timeout_sec = _tp50.compute_order_submission_timeout_sec
compute_asof_staleness_bdays = _tp50.compute_asof_staleness_bdays
should_block_buy_for_stale_asof = _tp50.should_block_buy_for_stale_asof

from src.f4_tp50 import exit_engine as ee
from src.f4_tp50.entry_pipeline import TP50LiveData
from src.live.inflight_registry import InflightRegistry
from src.live.staged_supervisor import StageError, StageTimeout


def _synthetic_data(codes=("11110",)) -> TP50LiveData:
    calendar = pd.date_range("2026-01-05", periods=6, freq="B")
    open_df = pd.DataFrame(100.0, index=calendar, columns=list(codes))
    high_df = open_df + 5
    low_df = open_df - 5
    close_df = open_df + 1
    mats_adj = {"open": open_df, "high": high_df, "low": low_df, "close": close_df}
    mats_raw = {k: v.copy() for k, v in mats_adj.items()}
    cond_b = pd.DataFrame(False, index=calendar, columns=list(codes))
    pit = pd.DataFrame(True, index=calendar, columns=list(codes))
    sum_df = pd.DataFrame({"Code": [], "DiscDate": []})
    return TP50LiveData(calendar=calendar, codes=list(codes), mats_adj=mats_adj, mats_raw=mats_raw,
                        pit=pit, cond_b=cond_b, sum_df=sum_df)


# ======================================================================
# STRATEGY_TYPE / constants
# ======================================================================
def test_strategy_type_is_f4_tp50():
    assert STRATEGY_TYPE == "f4_tp50"
    assert FIXED_LOT_SIZE == 100


# ======================================================================
# evaluate_exits: reconciliation (broker qty authoritative, TP50 tag only)
# ======================================================================
def test_evaluate_exits_ignores_non_tp50_positions():
    """E5・TP30がタグ付けした建玉はTP50のExit判定対象にしない
    （existing position reconciliation — 他戦略の建玉に一切触れない）。"""
    data = _synthetic_data(("11110",))
    as_of = data.calendar[-1]
    entry_date = data.calendar[0]
    data.mats_adj["low"].loc[as_of, "11110"] = 1.0  # would trigger a huge stop if evaluated

    exits = evaluate_exits(
        data, broker_positions={"11110": 100}, strategy_types={"11110": "f4_tp30"},
        entry_dates={"11110": str(entry_date.date())}, entry_prices={"11110": 100.0}, as_of=as_of,
    )
    assert exits == []

    exits_e5 = evaluate_exits(
        data, broker_positions={"11110": 100}, strategy_types={"11110": "simple_trend_e5"},
        entry_dates={"11110": str(entry_date.date())}, entry_prices={"11110": 100.0}, as_of=as_of,
    )
    assert exits_e5 == []


def test_evaluate_exits_recognizes_position_via_broker_dot_t_format():
    """2026-08-18 incident regression pin: fetch_broker_snapshot() actually
    returns positions keyed "XXXX.T" (kabu Symbol + '.T'), NOT the internal
    5-digit DB code — real live order 20260818A02N86495800 (4826.T x100,
    tagged internally as "48260") was invisible to evaluate_exits() before
    the strategy_router normalization fix. This test uses the REAL broker key
    shape (earlier tests in this file used an unrealistic same-format
    broker_positions key, which is exactly why this bug was never caught)."""
    data = _synthetic_data(("48260",))
    as_of = data.calendar[-1]
    entry_date = data.calendar[0]
    data.mats_adj["low"].loc[as_of, "48260"] = 80.0
    data.mats_adj["open"].loc[as_of, "48260"] = 100.0  # no gap -> trailing_touch

    exits = evaluate_exits(
        data, broker_positions={"4826.T": 100}, strategy_types={"48260": STRATEGY_TYPE},
        entry_dates={"48260": str(entry_date.date())}, entry_prices={"48260": 100.0}, as_of=as_of,
    )
    assert len(exits) == 1
    assert exits[0]["code"] == "48260"     # internal-code form, usable for entry_dates/entry_prices lookups
    assert exits[0]["qty"] == 100          # NOT 0 — must resolve qty via broker's "4826.T" key, not "48260"
    assert exits[0]["exit_reason"] in ("trailing_touch", "trailing_gap_open")


def test_evaluate_exits_zero_qty_position_excluded():
    data = _synthetic_data(("11110",))
    as_of = data.calendar[-1]
    exits = evaluate_exits(
        data, broker_positions={"11110": 0}, strategy_types={"11110": STRATEGY_TYPE},
        entry_dates={"11110": str(data.calendar[0].date())}, entry_prices={"11110": 100.0}, as_of=as_of,
    )
    assert exits == []


def test_evaluate_exits_fires_trailing_touch():
    data = _synthetic_data(("11110",))
    as_of = data.calendar[-1]
    entry_date = data.calendar[0]
    data.mats_adj["low"].loc[as_of, "11110"] = 80.0
    data.mats_adj["open"].loc[as_of, "11110"] = 100.0  # no gap
    exits = evaluate_exits(
        data, broker_positions={"11110": 100}, strategy_types={"11110": STRATEGY_TYPE},
        entry_dates={"11110": str(entry_date.date())}, entry_prices={"11110": 100.0}, as_of=as_of,
    )
    assert len(exits) == 1
    assert exits[0]["code"] == "11110"
    assert exits[0]["exit_reason"] in ("trailing_touch", "trailing_gap_open")


def test_evaluate_exits_fires_target_touch_at_50pct():
    """TP50固有: 50%到達での利確が正しく発火する（30%では発火しない水準で確認）。"""
    data = _synthetic_data(("11110",))
    as_of = data.calendar[-1]
    entry_date = data.calendar[0]
    # entry=100 -> target=150. high=152 today (>=150), open=140 (no gap), low=138.
    data.mats_adj["open"].loc[as_of, "11110"] = 140.0
    data.mats_adj["high"].loc[as_of, "11110"] = 152.0
    data.mats_adj["low"].loc[as_of, "11110"] = 138.0
    exits = evaluate_exits(
        data, broker_positions={"11110": 100}, strategy_types={"11110": STRATEGY_TYPE},
        entry_dates={"11110": str(entry_date.date())}, entry_prices={"11110": 100.0}, as_of=as_of,
    )
    assert len(exits) == 1
    assert exits[0]["exit_reason"] == "target_touch"
    assert exits[0]["target_price"] == pytest.approx(150.0)


def test_evaluate_exits_skips_entry_day_itself():
    """本日エントリーした銘柄は本日Exit評価の対象にしない（frozen spec: entry日は判定対象外）。"""
    data = _synthetic_data(("11110",))
    as_of = data.calendar[-1]
    data.mats_adj["low"].loc[as_of, "11110"] = 1.0
    exits = evaluate_exits(
        data, broker_positions={"11110": 100}, strategy_types={"11110": STRATEGY_TYPE},
        entry_dates={"11110": str(as_of.date())}, entry_prices={"11110": 100.0}, as_of=as_of,
    )
    assert exits == []


# ======================================================================
# evaluate_exits: restart/recovery — missing metadata must not crash
# ======================================================================
def test_evaluate_exits_handles_missing_entry_metadata_gracefully():
    """position_strategy_types=f4_tp50 だが entry_date/entry_price が state に無い
    （手動介入・state破損等）場合、クラッシュせずスキップする（restart/recovery耐性）。"""
    data = _synthetic_data(("11110",))
    as_of = data.calendar[-1]
    exits = evaluate_exits(
        data, broker_positions={"11110": 100}, strategy_types={"11110": STRATEGY_TYPE},
        entry_dates={}, entry_prices={}, as_of=as_of,
    )
    assert exits == []


def test_evaluate_exits_handles_symbol_missing_from_price_matrix():
    data = _synthetic_data(("11110",))
    as_of = data.calendar[-1]
    exits = evaluate_exits(
        data, broker_positions={"99999": 100}, strategy_types={"99999": STRATEGY_TYPE},
        entry_dates={"99999": str(data.calendar[0].date())}, entry_prices={"99999": 100.0}, as_of=as_of,
    )
    assert exits == []


# ======================================================================
# evaluate_buy_sizing: 100株固定・部分縮小なし・skip理由の区別
# ======================================================================
def _cand(code: str, px: float) -> dict:
    return {"code": code, "entry_price_adjusted_open": px}


def test_buy_sizing_funds_exactly_fixed_lot():
    candidates = [_cand("1301", 1000.0)]
    funded, breakdown = evaluate_buy_sizing(candidates, 1_000_000.0, set(), set(), [])
    assert len(funded) == 1
    exec_px = ee.compute_entry_fill_price(1000.0)
    assert funded[0]["estimated_notional"] == pytest.approx(exec_px * FIXED_LOT_SIZE)
    assert FIXED_LOT_SIZE == 100
    assert breakdown["funded_total"] == 1


def test_buy_sizing_skips_without_partial_fill_when_cash_insufficient():
    candidates = [_cand("1301", 1000.0)]
    funded, breakdown = evaluate_buy_sizing(candidates, 50_000.0, set(), set(), [])
    assert funded == []
    assert breakdown["capital_exhausted_skip"] == 1


def test_buy_sizing_skips_already_held_symbol():
    candidates = [_cand("1301", 1000.0)]
    funded, breakdown = evaluate_buy_sizing(candidates, 10_000_000.0, {"1301"}, set(), [])
    assert funded == []
    assert breakdown["already_held_skip"] == 1


def test_buy_sizing_skips_unresolved_duplicate_symbol():
    candidates = [_cand("1301", 1000.0)]
    funded, breakdown = evaluate_buy_sizing(candidates, 10_000_000.0, set(), {"1301"}, [])
    assert funded == []
    assert breakdown["unresolved_duplicate_skip"] == 1


def test_buy_sizing_never_partial_fills_below_fixed_lot():
    candidates = [_cand("1301", 1_000_000.0)]
    funded, breakdown = evaluate_buy_sizing(candidates, 500_000.0, set(), set(), [])
    assert funded == []
    assert breakdown["capital_exhausted_skip"] == 1


def test_buy_sizing_uses_pre_exit_cash_only_not_idealized_same_day_reuse():
    candidates = [_cand("1301", 1000.0)]
    cash_before_any_exits = 90_000.0
    funded, breakdown = evaluate_buy_sizing(candidates, cash_before_any_exits, set(), set(), [])
    exec_px = ee.compute_entry_fill_price(1000.0)
    one_lot_cost = exec_px * FIXED_LOT_SIZE + ee.compute_commission(exec_px * FIXED_LOT_SIZE)
    assert one_lot_cost > cash_before_any_exits
    assert funded == []


def test_buy_sizing_code_ascending_processing_order():
    candidates = [_cand("2000", 1000.0), _cand("1000", 1000.0)]
    exec_px = ee.compute_entry_fill_price(1000.0)
    one_lot_cost = exec_px * FIXED_LOT_SIZE + ee.compute_commission(exec_px * FIXED_LOT_SIZE)
    cash = one_lot_cost + 1000.0
    funded, breakdown = evaluate_buy_sizing(candidates, cash, set(), set(), [])
    assert len(funded) == 1
    assert funded[0]["code"] == "2000"
    assert breakdown["capital_exhausted_skip"] == 1


# ======================================================================
# OrderInstruction
# ======================================================================
def test_order_instruction_has_required_attributes():
    o = OrderInstruction(symbol="1301", side="BUY", qty=100, estimated_price=1000.0)
    assert o.strategy_type == STRATEGY_TYPE
    assert o.symbol_4digit == "1301"
    assert o.qty == 100


# ======================================================================
# 5-digit -> 4-digit kabu symbol normalization (2026-08-18 incident fix)
#
# ROOT CAUSE: TP50 candidate codes are the 5-digit database/market master
# form (e.g. "48260"), but OrderInstruction.symbol_4digit was never
# explicitly set at the BUY/SELL construction sites, so its __post_init__
# fallback (symbol_4digit = symbol) silently propagated the 5-digit code
# all the way to kabu STATION's Symbol field, which only recognizes the
# 4-digit market ticker ("4826") -- causing a real live HTTP 400 /
# Code=4002001 "symbol not found" on the first live BUY (2026-08-18).
# The prior test above used a already-4-digit fixture ("1301"), which
# happened to pass and masked the bug -- these tests use a realistic
# 5-digit fixture to actually exercise the truncation.
# ======================================================================
def test_order_instruction_5digit_buy_code_normalizes_to_4digit_symbol():
    """Mirrors the exact construction pattern used for buy_orders_intended
    in src.run_live_signal_f4_tp50.main() (symbol=code, symbol_4digit=to_kabu_symbol(code))."""
    from src.market_snapshot.universe import to_kabu_symbol

    code = "48260"
    o = OrderInstruction(symbol=code, side="BUY", qty=FIXED_LOT_SIZE,
                          estimated_price=1000.0, reason="F4_TP50_entry_signal",
                          symbol_4digit=to_kabu_symbol(code))
    assert o.symbol == "48260"          # internal/DB form preserved (unchanged)
    assert o.symbol_4digit == "4826"    # broker-facing form correctly truncated


def test_order_instruction_5digit_sell_code_normalizes_to_4digit_symbol():
    """Mirrors the exact construction pattern used for sell_orders_intended
    in src.run_live_signal_f4_tp50.main()."""
    from src.market_snapshot.universe import to_kabu_symbol

    code = "72030"
    o = OrderInstruction(symbol=code, side="SELL", qty=100,
                          estimated_price=1000.0, reason="trailing_touch",
                          symbol_4digit=to_kabu_symbol(code))
    assert o.symbol == "72030"
    assert o.symbol_4digit == "7203"


def test_serialize_order_sends_4digit_symbol_to_broker_worker_protocol():
    """End-to-end (minus network): the exact dict handed to broker_worker's
    input protocol (and from there to kabu send_order's Symbol field) must
    carry the 4-digit form, not the 5-digit DB form. No network call, no
    broker process spawned -- serialize_order() is a pure function."""
    from src.live.process_supervisor import serialize_order
    from src.market_snapshot.universe import to_kabu_symbol

    code = "48260"
    o = OrderInstruction(symbol=code, side="BUY", qty=FIXED_LOT_SIZE,
                          estimated_price=1234.0, reason="F4_TP50_entry_signal",
                          symbol_4digit=to_kabu_symbol(code))
    payload = serialize_order(o, front_order_type=1, client_order_id="test-coi-48260")
    assert payload["symbol"] == "48260"
    assert payload["symbol_4digit"] == "4826"  # <-- what actually reaches kabu's Symbol field


def test_source_wires_symbol_4digit_at_both_buy_and_sell_construction_sites():
    """Regression guard against silently dropping the fix: both
    OrderInstruction(...) construction sites in run_live_signal_f4_tp50.py
    must pass symbol_4digit=to_kabu_symbol(...) explicitly (not rely on the
    __post_init__ fallback, which is only safe for already-4-digit input)."""
    src_path = _ROOT / "src" / "run_live_signal_f4_tp50.py"
    text = src_path.read_text(encoding="utf-8")
    buy_block = text[text.index('OrderInstruction(symbol=f["code"], side="BUY"'):]
    buy_block = buy_block[:buy_block.index("for f in funded")]
    sell_block = text[text.index('OrderInstruction(symbol=e["code"], side="SELL"'):]
    sell_block = sell_block[:sell_block.index("for e in exits")]
    assert 'symbol_4digit=to_kabu_symbol(f["code"])' in buy_block
    assert 'symbol_4digit=to_kabu_symbol(e["code"])' in sell_block
    assert "from src.market_snapshot.universe import to_kabu_symbol" in text


# ======================================================================
# compute_order_submission_timeout_sec
# ======================================================================
def test_order_timeout_never_below_floor():
    assert compute_order_submission_timeout_sec(0) >= 30.0


def test_order_timeout_monotonic():
    assert compute_order_submission_timeout_sec(3) < compute_order_submission_timeout_sec(10)


# ======================================================================
# apply_fill_metadata_updates: entry metadata persistence / restart-recovery
#
# 2026-08-20 9344インシデント: entry_date/entry_priceは実約定(kabu API broker
# fill)ベースでなければならない。as_of/estimated_priceは注文送信前のシグナル
# 理論値であり、約定が遅延した場合(シグナルのas_ofと実約定日がズレた場合)に
# 実態と乖離する。以下のテストは実約定を返すfakeクライアントを用いる。
# ======================================================================
class _FakeKabuClient:
    """get_orders()のみ実装するテスト用スタブ。RecType=8(約定明細)を1件返す(単一約定・全数)。"""

    def __init__(self, order_id: str, execution_day: str, price: float, qty: float = 100.0):
        self._order_id = order_id
        self._execution_day = execution_day
        self._price = price
        self._qty = qty

    def get_orders(self, only_open: bool = False, updtime=None):
        return [{
            "ID": self._order_id,
            "Details": [
                {"SeqNum": 1, "RecType": 1, "Price": 0.0},
                {"SeqNum": 4, "RecType": 4, "Price": 0.0},
                {"SeqNum": 5, "RecType": 8, "Price": self._price, "Qty": self._qty,
                 "ExecutionDay": self._execution_day},
            ],
        }]


class _MultiFillKabuClient:
    """複数RecType=8明細(分割約定)を返すスタブ。加重平均価格・最早約定日を検証するために使う。"""

    def __init__(self, order_id: str, fills: list[tuple[str, float, float]]):
        """fills: [(execution_day, price, qty), ...]"""
        self._order_id = order_id
        self._fills = fills

    def get_orders(self, only_open: bool = False, updtime=None):
        details = [{"SeqNum": 1, "RecType": 1, "Price": 0.0}]
        for i, (day, price, qty) in enumerate(self._fills):
            details.append({"SeqNum": 10 + i, "RecType": 8, "Price": price, "Qty": qty, "ExecutionDay": day})
        return [{"ID": self._order_id, "Details": details}]


class _EmptyKabuClient:
    """常に空リストを返す=約定確認不能をシミュレートするスタブ。"""

    def get_orders(self, only_open: bool = False, updtime=None):
        return []


def test_apply_fill_metadata_records_new_buy_from_actual_broker_fill():
    """entry_date/entry_priceは実約定(broker fill)から記録され、
    estimated_price/as_ofとは一致しなくてよい（実際に乖離するケースを含む）。"""
    as_of = pd.Timestamp("2026-08-17")  # シグナル理論上のentry日（意図的にズレさせる）
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-1"}]
    client = _FakeKabuClient("ORDER-1", "2026-08-19T10:00:00+09:00", 1224.0)  # 実約定は8/19・1224円
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert changed is True
    assert ed["1301"] == "2026-08-19"   # 実約定日（as_of=8/17ではない）
    assert ep_["1301"] == 1224.0        # 実約定価格（estimated_price=1000ではない）
    assert st["1301"] == STRATEGY_TYPE


def test_apply_fill_metadata_delayed_fill_uses_broker_date_not_signal_as_of():
    """9344インシデントの再現: シグナルはentry_date=8/18を想定していたが、
    実際の約定は8/19に発生。記録されるentry_dateは実約定日(8/19)でなければならない。"""
    as_of = pd.Timestamp("2026-08-18")  # 汚染の原因だった値そのもの
    results = [{"symbol": "93440", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1379.0, "order_id": "20260819A02N88827536"}]
    client = _FakeKabuClient("20260819A02N88827536", "2026-08-19T14:28:51+09:00", 1224.0)
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert ed["93440"] == "2026-08-19"
    assert ep_["93440"] == 1224.0
    assert ed["93440"] != as_of.strftime("%Y-%m-%d")


def test_apply_fill_metadata_fails_closed_when_broker_fill_unconfirmed():
    """約定情報がkabu APIから取得できない場合、estimated_price/as_ofで代用せず、
    entry metadataを一切記録しない（次回runでの再試行に委ねる）。"""
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-MISSING"}]
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=_EmptyKabuClient())
    assert "1301" not in ed
    assert "1301" not in ep_
    # strategy_type tagging (constant "f4_tp50", not derived from estimated/theoretical
    # data) is unaffected by the fail-closed guard and still applies — changed=True here
    # reflects that, not that entry_date/entry_price were written.
    assert changed is True
    assert st["1301"] == STRATEGY_TYPE


def test_apply_fill_metadata_fails_closed_when_client_is_none():
    """client未指定（推測値へのフォールバック余地）でも、estimated_price/as_ofを
    entry metadataとして書き込まない。"""
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-1"}]
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=None)
    assert "1301" not in ed
    assert "1301" not in ep_
    assert changed is True  # strategy_type tagging still applies; see note above
    assert st["1301"] == STRATEGY_TYPE


def test_apply_fill_metadata_does_not_clobber_existing_entry_on_retry():
    """既にentry_date/entry_priceが記録済みの場合、broker参照すら行わず既存値を保持する
    （retry/duplicate resultが真のoriginal entryを上書きしない）。"""
    as_of = pd.Timestamp("2026-08-18")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 9999.0, "order_id": "ORDER-RETRY"}]
    ed, ep_, st, changed = apply_fill_metadata_updates(
        results, {"1301": "2026-08-17"}, {"1301": 1000.0}, {"1301": STRATEGY_TYPE}, as_of,
        client=_EmptyKabuClient(),  # 呼ばれても失敗するはずだが、既存値保持のため到達しない
    )
    assert ed["1301"] == "2026-08-17"
    assert ep_["1301"] == 1000.0


def test_apply_fill_metadata_cleans_up_on_successful_sell():
    as_of = pd.Timestamp("2026-08-20")
    results = [{"symbol": "1301", "side": "SELL", "qty": 100, "success": True, "estimated_price": 1100.0}]
    ed, ep_, st, changed = apply_fill_metadata_updates(
        results, {"1301": "2026-08-17"}, {"1301": 1000.0}, {"1301": STRATEGY_TYPE}, as_of,
    )
    assert changed is True
    assert "1301" not in ed
    assert "1301" not in ep_
    assert "1301" not in st


def test_apply_fill_metadata_ignores_failed_orders():
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": False, "estimated_price": 1000.0}]
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of)
    assert changed is False
    assert ed == {} and ep_ == {} and st == {}


def test_apply_fill_metadata_does_not_touch_other_strategy_symbols():
    """failed/no-op resultsだけでなく、結果に含まれないシンボルのmetadataは一切変更しない
    （E5・TP30等の既存タグ付け建玉を誤って書き換えない）。"""
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-1"}]
    client = _FakeKabuClient("ORDER-1", "2026-08-17T09:30:00+09:00", 1000.0)
    ed, ep_, st, changed = apply_fill_metadata_updates(
        results, {"9999": "2026-01-01"}, {"9999": 500.0}, {"9999": "f4_tp30"}, as_of, client=client,
    )
    assert ed["9999"] == "2026-01-01"
    assert st["9999"] == "f4_tp30"  # TP30's own tag untouched


def test_apply_fill_metadata_fails_closed_on_partial_fill():
    """発注qty=100に対しbroker約定が60株のみの場合、部分約定を全数約定として
    metadata記録してはならない（全数約定確認までentry_date/entry_priceを保留）。"""
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-PARTIAL"}]
    client = _FakeKabuClient("ORDER-PARTIAL", "2026-08-17T09:30:00+09:00", 1000.0, qty=60.0)
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert "1301" not in ed
    assert "1301" not in ep_


class _MissingFieldKabuClient:
    """ExecutionDayまたはPriceが欠落したDetails[]を返すスタブ（API応答異常の再現）。"""

    def __init__(self, order_id: str, missing_field: str, qty: float = 100.0):
        self._order_id = order_id
        self._missing_field = missing_field
        self._qty = qty

    def get_orders(self, only_open: bool = False, updtime=None):
        detail = {"SeqNum": 1, "RecType": 8, "Price": 1000.0, "Qty": self._qty,
                  "ExecutionDay": "2026-08-17T09:30:00+09:00"}
        detail.pop(self._missing_field, None)
        return [{"ID": self._order_id, "Details": [detail]}]


def test_apply_fill_metadata_fails_closed_when_execution_day_missing():
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-NO-DAY"}]
    client = _MissingFieldKabuClient("ORDER-NO-DAY", "ExecutionDay")
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert "1301" not in ed
    assert "1301" not in ep_


def test_apply_fill_metadata_fails_closed_when_price_missing():
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-NO-PRICE"}]
    client = _MissingFieldKabuClient("ORDER-NO-PRICE", "Price")
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert "1301" not in ed
    assert "1301" not in ep_


def test_apply_fill_metadata_holding_truth_is_broker_snapshot_not_state():
    """entry_date/entry_priceがfail-closedで未記録でも、strategy_typeタグ付けにより
    保有中として扱われ得るが、evaluate_exits()はentry_date/entry_price欠落時に
    安全にスキップする（Broker-as-Sole-SSOT: 保有数量の真実は常にbroker snapshot、
    stateは補助metadataに過ぎない — 「約定確認前にportfolio_stateを保有済みと
    確定させる経路」が存在しないことの回帰テスト）。"""
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-UNCONFIRMED"}]
    ed, ep_, st, changed = apply_fill_metadata_updates(
        results, {}, {}, {}, as_of, client=_EmptyKabuClient(),
    )
    assert "1301" not in ed and "1301" not in ep_
    # strategy_typeは付くが、evaluate_exits側の "entry_date is None -> continue" ガード
    # （src/run_live_signal_f4_tp50.py evaluate_exits()参照）により
    # entry_date/entry_price不在のこのシンボルはExit判定から安全に除外される。
    assert st.get("1301") == STRATEGY_TYPE


def test_apply_fill_metadata_aggregates_multiple_execution_details():
    """kabu APIが1注文に対し複数のRecType=8明細(分割約定)を返す場合、
    合計数量が発注数量を満たせば、数量加重平均価格・最早約定日を用いて記録する。"""
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-MULTI"}]
    # 60株@1200円(先着) + 40株@1210円 → 加重平均 = (60*1200+40*1210)/100 = 1204円
    client = _MultiFillKabuClient("ORDER-MULTI", [
        ("2026-08-17T09:30:05+09:00", 1200.0, 60.0),
        ("2026-08-17T09:30:07+09:00", 1210.0, 40.0),
    ])
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert ed["1301"] == "2026-08-17"
    assert ep_["1301"] == pytest.approx(1204.0)


def test_apply_fill_metadata_multi_fill_uses_earliest_execution_date():
    """分割約定が日をまたぐ場合(通常は同日内だが仕様として)、最早の約定日を採用する。"""
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-MULTI2"}]
    client = _MultiFillKabuClient("ORDER-MULTI2", [
        ("2026-08-18T09:00:00+09:00", 1000.0, 50.0),
        ("2026-08-17T09:00:00+09:00", 1000.0, 50.0),  # こちらが最早
    ])
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert ed["1301"] == "2026-08-17"


def test_apply_fill_metadata_idempotent_when_same_order_processed_twice():
    """同一order_idの結果を2回(例: リトライ/再実行で)処理しても、
    2回目はbroker参照すら行わず既存値を保持し、二重加算・上書きが起きない。"""
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-DUP"}]
    client = _FakeKabuClient("ORDER-DUP", "2026-08-17T09:30:00+09:00", 1224.0)

    ed1, ep1, st1, changed1 = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert changed1 is True
    assert ep1["1301"] == 1224.0

    # 2回目: 同じresultsを再度処理（client は呼ばれれば別値を返す設定にして、
    # 呼ばれていない＝既存値保持であることを検証する）
    client2 = _FakeKabuClient("ORDER-DUP", "2026-08-19T09:30:00+09:00", 9999.0)
    ed2, ep2, st2, changed2 = apply_fill_metadata_updates(results, ed1, ep1, st1, as_of, client=client2)
    assert ed2["1301"] == "2026-08-17"  # 変化なし
    assert ep2["1301"] == 1224.0        # 変化なし（9999.0で上書きされていない）


def test_apply_fill_metadata_fails_closed_on_future_execution_date():
    """kabu APIが返す約定日が未来日(API異常/パース不良を示唆)の場合、
    sanityチェックで弾き、entry metadataを記録しない。"""
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-FUTURE"}]
    client = _FakeKabuClient("ORDER-FUTURE", "2099-01-01T09:30:00+09:00", 1000.0)
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert "1301" not in ed
    assert "1301" not in ep_


def test_apply_fill_metadata_9344_incident_reproduction_new_implementation_passes():
    """9344インシデントの実測値そのものを用いた再現テスト。
    旧実装(as_of/estimated_price使用)ならentry_date=2026-08-18・price=1379.0を
    記録して本テストはFAILしていたはずだが、新実装(broker実約定ベース)ではPASSする。"""
    as_of = pd.Timestamp("2026-08-18")  # 汚染の原因だった、シグナル理論上のentry_date
    results = [{"symbol": "93440", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1379.0, "order_id": "20260819A02N88827536"}]
    client = _FakeKabuClient("20260819A02N88827536", "2026-08-19T14:28:51+09:00", 1224.0)
    ed, ep_, st, changed = apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=client)
    assert ed["93440"] == "2026-08-19"
    assert ep_["93440"] == 1224.0
    assert ed["93440"] != "2026-08-18"
    assert ep_["93440"] != 1379.0


# ======================================================================
# handle_order_submission_stage_failure — reused directly from E5's module
# ======================================================================
def test_handle_order_submission_stage_failure_is_reused_from_e5_module(tmp_path):
    _orig_argv0 = sys.argv[0]
    sys.argv[0] = str(_ROOT / "run_live_signal_simple_e5.py")
    try:
        from src.run_live_signal_simple_e5 import handle_order_submission_stage_failure as e5_handler
    finally:
        sys.argv[0] = _orig_argv0
    from src.run_live_signal_f4_tp50 import main as tp50_main
    import inspect
    src = inspect.getsource(tp50_main)
    assert "handle_order_submission_stage_failure" in src
    registry = InflightRegistry(tmp_path / "inflight_orders.jsonl")
    orders = [OrderInstruction(symbol="1301", side="BUY", qty=100, estimated_price=1000.0)]
    exc = StageTimeout("order_execution", elapsed=45.0, timeout=30.0)
    kind, results = e5_handler(exc, orders, registry)
    assert kind == "stage_timeout"
    assert results[0]["symbol"] == "1301"
    assert results[0]["strategy_type"] == STRATEGY_TYPE


# ======================================================================
# 実行結果メール通知（2026-08-18追加、src.notifier既存基盤の再利用のみ）
# ======================================================================
from unittest.mock import patch  # noqa: E402

_classify_tp50_notification = _tp50._classify_tp50_notification
_send_tp50_notification = _tp50._send_tp50_notification
_build_tp50_notification_body = _tp50._build_tp50_notification_body


def _base_result_summary(**overrides) -> dict:
    base = {
        "run_id": "f4_tp50_20260818_999999",
        "signal_date": "2026-08-17",
        "entry_date": "2026-08-18",
        "live": False,
        "entry_freeze_enabled": False,
        "fundamentals_freshness": {
            "max_disc_date": "2026-08-17", "staleness_bdays_vs_as_of": 0,
            "staleness_bdays_vs_real_today": 1, "is_stale": False, "reason": "FRESH",
        },
        "cash_source": "broker_live",
        "available_cash": 2900000.0,
        "positions_count": 3,
        "ca_guard": {"ca_pending_codes": [], "buy_candidates_blocked_by_ca_pending": []},
        "risk_gate": {"recommendation": "NORMAL", "dd": -0.01},
        "exits_intended": 0,
        "buys_intended_before_freeze": 1,
        "buys_blocked_by_entry_freeze": 0,
        "sizing_breakdown": {"cash_start": 3000000.0, "cash_remaining": 2900000.0,
                              "funded_total": 1, "capital_exhausted_skip": 0},
        "order_submission_results": None,
    }
    base.update(overrides)
    return base


# --- test 1: dry-run -> notify_dry_run exactly once -----------------------
def test_notification_dry_run_calls_notify_dry_run_once():
    rs = _base_result_summary(live=False)
    with patch("src.notifier.notify_dry_run") as m_dry, \
         patch("src.notifier.notify_success") as m_succ, \
         patch("src.notifier.notify_error") as m_err, \
         patch("src.notifier.notify_warning") as m_warn:
        _send_tp50_notification(rs, [], [])
    assert m_dry.call_count == 1
    assert m_succ.call_count == 0
    assert m_err.call_count == 0
    assert m_warn.call_count == 0


# --- test 2: live + any order success=False -> notify_error exactly once --
def test_notification_live_with_failed_order_calls_notify_error_once():
    rs = _base_result_summary(live=True, order_submission_results=[
        {"symbol": "48260", "symbol_4digit": "4826", "side": "BUY", "qty": 100,
         "success": False, "order_id": None, "error": "HTTP 400: symbol not found",
         "http_status": 400},
    ])
    with patch("src.notifier.notify_dry_run") as m_dry, \
         patch("src.notifier.notify_success") as m_succ, \
         patch("src.notifier.notify_error") as m_err, \
         patch("src.notifier.notify_warning") as m_warn:
        _send_tp50_notification(rs, [], [])
    assert m_err.call_count == 1
    assert m_dry.call_count == 0
    assert m_succ.call_count == 0
    assert m_warn.call_count == 0


# --- test 2b: live + all orders success=True -> notify_success once -------
def test_notification_live_all_success_calls_notify_success_once():
    rs = _base_result_summary(live=True, order_submission_results=[
        {"symbol": "48260", "symbol_4digit": "4826", "side": "BUY", "qty": 100,
         "success": True, "order_id": "ORDER-1", "error": None, "http_status": 200},
    ])
    with patch("src.notifier.notify_dry_run") as m_dry, \
         patch("src.notifier.notify_success") as m_succ, \
         patch("src.notifier.notify_error") as m_err, \
         patch("src.notifier.notify_warning") as m_warn:
        _send_tp50_notification(rs, [], [])
    assert m_succ.call_count == 1
    assert m_dry.call_count == 0
    assert m_err.call_count == 0
    assert m_warn.call_count == 0


# --- test 4: fundamentals stale (safe-side block) -> notify_warning once --
def test_notification_stale_fundamentals_calls_notify_warning_once():
    rs = _base_result_summary(
        live=True, order_submission_results=[],
        fundamentals_freshness={"max_disc_date": "2026-08-01", "staleness_bdays_vs_as_of": 0,
                                 "staleness_bdays_vs_real_today": 6, "is_stale": True,
                                 "reason": "STALE_FUNDAMENTALS_DATA"},
    )
    with patch("src.notifier.notify_dry_run") as m_dry, \
         patch("src.notifier.notify_success") as m_succ, \
         patch("src.notifier.notify_error") as m_err, \
         patch("src.notifier.notify_warning") as m_warn:
        _send_tp50_notification(rs, [], [])
    assert m_warn.call_count == 1
    assert m_dry.call_count == 0
    assert m_succ.call_count == 0
    assert m_err.call_count == 0


def test_notification_ca_pending_block_calls_notify_warning_once():
    rs = _base_result_summary(
        live=True, order_submission_results=[],
        ca_guard={"ca_pending_codes": ["48260"], "buy_candidates_blocked_by_ca_pending": ["48260"]},
    )
    assert _classify_tp50_notification(rs) == "warning"


# --- test 5: notification exception never propagates to caller ------------
def test_notification_exception_does_not_propagate():
    rs = _base_result_summary(live=False)
    with patch("src.notifier.notify_dry_run", side_effect=RuntimeError("SMTP boom")):
        _send_tp50_notification(rs, [], [])  # must not raise


def test_notification_exception_in_body_building_does_not_propagate():
    """Even a malformed result_summary must not crash the caller (fire-and-forget).

    2026-08-21/22 実インシデント回帰テスト: この関数は notify_dry_run() をmockして
    いなかったため、このテストをpytestで実行するたびに実SMTP経由で本番宛てへ
    「DRY RUN INVALID」の空データメールが送信されていた（"9:48の謎メール"と
    2026-08-22朝のINVALIDメールの真因はKabu API障害ではなく、この未mockテストの
    副作用だった）。他の同種テストと同じく4関数すべてをmockする。"""
    with patch("src.notifier.notify_dry_run") as m_dry, \
         patch("src.notifier.notify_success") as m_succ, \
         patch("src.notifier.notify_error") as m_err, \
         patch("src.notifier.notify_warning") as m_warn:
        _send_tp50_notification({"live": False}, None, None)  # missing keys, wrong types for lists
    assert m_dry.call_count == 1
    assert m_succ.call_count == 0
    assert m_err.call_count == 0
    assert m_warn.call_count == 0


# --- classification matrix (belt-and-suspenders on the pure function) -----
@pytest.mark.parametrize("live,results,fresh_stale,ca_blocked,freeze_blocked,cb_active,expected", [
    (False, None, False, False, 0, False, "dry_run"),
    (True, [{"success": True}], False, False, 0, False, "success"),
    (True, [{"success": False}], False, False, 0, False, "error"),
    (True, [{"success": True}, {"success": False}], False, False, 0, False, "error"),
    (True, [], True, False, 0, False, "warning"),
    (True, [], False, True, 0, False, "warning"),
    (True, [], False, False, 1, False, "warning"),
    (True, [], False, False, 0, True, "warning"),
    (True, [], False, False, 0, False, "success"),
])
def test_notification_classification_matrix(live, results, fresh_stale, ca_blocked, freeze_blocked, cb_active, expected):
    rs = _base_result_summary(
        live=live, order_submission_results=results,
        fundamentals_freshness={"max_disc_date": "x", "is_stale": fresh_stale, "reason": "x",
                                 "staleness_bdays_vs_as_of": 0, "staleness_bdays_vs_real_today": 0},
        ca_guard={"ca_pending_codes": ["1"] if ca_blocked else [],
                  "buy_candidates_blocked_by_ca_pending": ["1"] if ca_blocked else []},
        buys_blocked_by_entry_freeze=freeze_blocked,
        risk_gate={"recommendation": "CB_ACTIVE" if cb_active else "NORMAL"},
    )
    assert _classify_tp50_notification(rs) == expected


# ======================================================================
# 2026-08-20 通知フォーマット全面刷新（9344誤売却事故対応）:
# DRY RUN=「前日の実績（broker実約定）」+「本日の判断」の2部構成、
# LIVE=「発注書」（約定価格は翌日のDry Runで確認——注文時点では未確定のため
# 本文に出さない）。「なぜ売買したのか」「どの価格を基準にしたのか」
# 「現在のスコアはいくつか」が一目で分かる監査証跡としての本文を検証する。
# _find_previous_live_run()は実ディスク(logs/live/)を読むため、内容を
# 制御したいテストは明示的にpatchする（未patch時はNone=前日実績なし、を期待）。
# 実注文は一切行わない（_FakeKabuClient等のstubのみ使用）。
# ======================================================================
_find_previous_live_run = _tp50._find_previous_live_run


def _sell_item(code="93440", reason="trailing_touch", entry_price=1224.0,
                highest=1309.0, stop=1112.65, target=1836.0, fill=1204.09, qty=100):
    return {
        "code": code, "qty": qty, "exit_reason": reason, "exit_fill_price": fill,
        "stop_level": stop, "target_price": target, "highest_since_entry": highest,
        "entry_price": entry_price,
    }


def _buy_item(code="48260", theoretical=521.0):
    return {"code": code, "entry_price_adjusted_open": theoretical, "estimated_fill_price": theoretical}


def _no_prev_run():
    """_find_previous_live_run()をNoneに固定するpatchコンテキスト（前日実績セクションを
    テスト対象から除外し、実ディスクのlogs/live/内容に依存しないようにする）。"""
    return patch("src.run_live_signal_f4_tp50._find_previous_live_run", return_value=None)


# --- 1. 通常T15 SELL（DRY RUN・本日の判断）: 理由ラベル・Entry/最高値/STOP --
def test_scenario_1_normal_t15_trailing_sell_dry_run():
    rs = _base_result_summary(live=False, exits_intended=1,
                               exits_detail=[_sell_item(reason="trailing_touch")])
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "T15トレーリングSTOP" in body
    assert "Entry       ¥1,224" in body
    assert "最高値      ¥1,309" in body
    assert "STOP        ¥1,113" in body
    assert "→ LIVEならSELL" in body
    today_block = body.split("【本日の判断】")[1]
    assert "Target" not in today_block.split("STOP")[-1][:60]  # trailingではTarget行を出さない


# --- 2. TP50 SELL（target_touch）: Target表示・最高値/STOPは表示しない ------
def test_scenario_2_tp50_target_sell():
    rs = _base_result_summary(live=False, exits_intended=1,
                               exits_detail=[_sell_item(reason="target_touch", fill=1836.0)])
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "TP50利確" in body
    assert "Target      ¥1,836" in body
    today_block = body.split("【本日の判断】")[1]
    assert "最高値" not in today_block
    assert "STOP        " not in today_block


# --- 3. Score Replacement（DRY RUN・本日の判断） ----------------------------
def test_scenario_3_score_replacement_dry_run():
    rs = _base_result_summary(live=False, score_replacement={
        "enabled": True, "candidates_evaluated": 1,
        "decisions": [{
            "candidate_code": "48260", "candidate_score": 71.5, "decision": "REPLACE_SIMULATED",
            "sold_code": "93440", "holding_score": 48.2, "sell_price": 1204.09, "buy_price": 521.0,
        }],
    })
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "SCORE REPLACEMENT" in body
    assert "48260" in body and "93440" in body
    assert "Score差     +23.3" in body
    assert "→ LIVEなら入替" in body


# --- 4. BUY（LIVE発注書、約定価格は表示しない） -----------------------------
def test_scenario_4_buy_live():
    rs = _base_result_summary(live=True, funded_detail=[_buy_item()],
                               order_submission_results=[
                                   {"symbol": "48260", "symbol_4digit": "4826", "side": "BUY", "qty": 100,
                                    "success": True, "order_id": "ORDER-BUY-1", "error": None},
                               ])
    body = _build_tp50_notification_body(rs, [], [])
    assert "新規Entry" in body
    assert "基準価格    ¥521" in body
    assert "注文        成行BUY" in body
    assert "注文ID      ORDER-BUY-1" in body
    order_block = body.split("【BUY ORDER】")[1].split("【SYSTEM】")[0]
    assert "実約定" not in order_block  # LIVEの発注書ブロックには約定価格を出さない（翌日Dry Runで確認）


# --- 5. BUY + Replacement 混在（DRY RUN・本日の判断） -----------------------
def test_scenario_5_buy_and_replacement_together():
    rs = _base_result_summary(live=False, funded_detail=[_buy_item(code="34570", theoretical=926.0)],
                               score_replacement={
                                   "enabled": True, "candidates_evaluated": 1,
                                   "decisions": [{
                                       "candidate_code": "48260", "candidate_score": 71.5,
                                       "decision": "REPLACE_SIMULATED", "sold_code": "93440",
                                       "holding_score": 48.2, "sell_price": 1204.09, "buy_price": 521.0,
                                   }],
                               })
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    today_block = body.split("【本日の判断】")[1]
    replacement_block = today_block.split("SCORE REPLACEMENT")[1]
    assert "34570" in today_block.split("SCORE REPLACEMENT")[0]  # 通常BUYブロック側
    assert "48260" in replacement_block  # 入替ブロック側


# --- 6. 複数SELL（DRY RUN・本日の判断） -------------------------------------
def test_scenario_6_multiple_sells():
    rs = _base_result_summary(live=False, exits_intended=2, exits_detail=[
        _sell_item(code="93440", reason="trailing_touch"),
        _sell_item(code="48260", reason="target_touch", entry_price=521.0, fill=781.5),
    ])
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "SELL       2" in body
    assert "93440" in body and "48260" in body
    today_block = body.split("【本日の判断】")[1]
    assert today_block.count("T15トレーリングSTOP") + today_block.count("TP50利確") == 2


# --- 7. 複数BUY（DRY RUN・本日の判断） --------------------------------------
def test_scenario_7_multiple_buys():
    rs = _base_result_summary(live=False, funded_detail=[
        _buy_item(code="17880", theoretical=4285.0), _buy_item(code="17160", theoretical=1420.0),
    ])
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "BUY        2" in body
    assert "17880" in body and "17160" in body


# --- 8. 0件（全ブロック"なし"） ----------------------------------------------
def test_scenario_8_zero_activity():
    rs = _base_result_summary(live=False)
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "SELL       0" in body
    assert "BUY        0" in body
    assert "REPLACEMENT 0" in body
    today_block = body.split("【本日の判断】")[1]
    assert today_block.count("0件") >= 3  # SELL/BUY/SCORE REPLACEMENTの各ブロック（0件とN/Aを区別するため"なし"から変更）


# --- 9. Scheduler未発火（手動遅延実行） -------------------------------------
def test_scenario_9_scheduler_not_fired_shows_warning():
    rs = _base_result_summary(
        live=True, order_submission_results=[],
        run_started_at="2026-08-20 09:36:11", scheduled_trigger_hhmm="08:49",
    )
    assert _classify_tp50_notification(rs) == "warning"
    body = _build_tp50_notification_body(rs, [], [])
    assert "08:49 自動実行されず" in body
    assert "09:36 手動遅延実行" in body


def test_scenario_9b_on_time_run_does_not_warn():
    rs = _base_result_summary(
        live=True, order_submission_results=[],
        run_started_at="2026-08-20 08:50:00", scheduled_trigger_hhmm="08:49",
    )
    assert _classify_tp50_notification(rs) == "success"
    body = _build_tp50_notification_body(rs, [], [])
    assert "自動実行されず" not in body


# --- 10. API error（broker snapshot取得不可） -------------------------------
def test_scenario_10_api_error_shown_in_data_status():
    rs = _base_result_summary(live=True, order_submission_results=[],
                               cash_source="unavailable_dry_run_degraded")
    body = _build_tp50_notification_body(rs, [], [])
    assert "Kabu API          異常/未接続" in body
    assert "API接続異常" in body  # 【警告】ブロックにも表示


# --- 11. metadata mismatch（fail-closedイベントが【警告】に表示される） ----
def test_scenario_11_metadata_mismatch_warning():
    rs = _base_result_summary(
        live=True, order_submission_results=[],
        metadata_warnings=["metadata mismatch: 93440 約定確認不能（order_id=ORDER-X）"],
    )
    assert _classify_tp50_notification(rs) == "warning"
    body = _build_tp50_notification_body(rs, [], [])
    assert "【警告】" in body
    assert "metadata mismatch: 93440" in body
    assert "Metadata          異常" in body


def test_apply_fill_metadata_warnings_sink_receives_fail_closed_event():
    as_of = pd.Timestamp("2026-08-17")
    results = [{"symbol": "1301", "side": "BUY", "qty": 100, "success": True,
                "estimated_price": 1000.0, "order_id": "ORDER-MISSING"}]
    sink: list[str] = []
    apply_fill_metadata_updates(results, {}, {}, {}, as_of, client=_EmptyKabuClient(), warnings_sink=sink)
    assert len(sink) == 1
    assert "1301" in sink[0]


# --- 12/13. DRY RUN と LIVE の見出しが明確に区別される -----------------------
def test_scenario_12_13_dry_run_vs_live_header():
    rs_dry = _base_result_summary(live=False)
    rs_live = _base_result_summary(live=True, order_submission_results=[])
    with _no_prev_run():
        dry_body = _build_tp50_notification_body(rs_dry, [], [])
    live_body = _build_tp50_notification_body(rs_live, [], [])
    assert "DAILY DRY RUN" in dry_body
    assert "LIVE ORDER REPORT" in live_body
    assert "DRY RUN" not in live_body.split("【")[0]


# --- 14. 実約定価格と理論価格が異なるケース（9344実例、DRY RUNの前日実績） --
def test_scenario_14_prev_day_sell_actual_fill_differs_from_theoretical_price():
    prev_run = {
        "run_id": "f4_tp50_20260820_093611", "live": True,
        "run_started_at": "2026-08-20 09:36:11",
        "exits_detail": [_sell_item(code="93440", reason="trailing_touch", fill=1204.09)],
        "funded_detail": [], "score_replacement": {"decisions": []},
        "order_submission_results": [
            {"symbol": "93440", "symbol_4digit": "9344", "side": "SELL", "qty": 100,
             "success": True, "order_id": "20260820A02N90347371", "error": None},
        ],
    }
    client = _FakeKabuClient("20260820A02N90347371", "2026-08-20T09:38:17+09:00", 1232.0)
    rs = _base_result_summary(live=False)
    with patch("src.run_live_signal_f4_tp50._find_previous_live_run", return_value=prev_run):
        body = _build_tp50_notification_body(rs, [], [], client=client)
    prev_block = body.split("【前日の実績】")[1].split("【本日の判断】")[0]
    assert "約定価格    ¥1,232" in prev_block   # broker実約定（理論値1204とは異なる）
    # 実現損益は実約定ベースで計算される（理論値ではない）: (1232-1224)*100 = +800
    assert "実現損益    +¥800" in prev_block
    assert "約定時刻    09:38:17" in prev_block


def test_scenario_14b_prev_day_buy_actual_fill_differs_from_theoretical():
    prev_run = {
        "run_id": "f4_tp50_20260819_142609", "live": True,
        "run_started_at": "2026-08-19 14:26:09",
        "exits_detail": [], "funded_detail": [_buy_item(code="93440", theoretical=1379.0)],
        "score_replacement": {"decisions": []},
        "order_submission_results": [
            {"symbol": "93440", "symbol_4digit": "9344", "side": "BUY", "qty": 100,
             "success": True, "order_id": "20260819A02N88827536", "error": None},
        ],
    }
    client = _FakeKabuClient("20260819A02N88827536", "2026-08-19T14:28:51+09:00", 1224.0)
    rs = _base_result_summary(live=False)
    with patch("src.run_live_signal_f4_tp50._find_previous_live_run", return_value=prev_run):
        body = _build_tp50_notification_body(rs, [], [], client=client)
    prev_block = body.split("【前日の実績】")[1].split("【本日の判断】")[0]
    assert "約定価格    ¥1,224" in prev_block  # broker実約定（理論値1379とは異なる）


def test_previous_live_run_lookup_returns_none_when_no_log_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(_tp50, "LIVE_LOG_DIR", tmp_path)
    assert _find_previous_live_run() is None


def test_previous_live_run_lookup_skips_dry_run_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(_tp50, "LIVE_LOG_DIR", tmp_path)
    (tmp_path / "f4_tp50_f4_tp50_20260819_090000.json").write_text(
        json.dumps({"run_id": "f4_tp50_20260819_090000", "live": False}), encoding="utf-8",
    )
    assert _find_previous_live_run() is None


def test_previous_live_run_lookup_returns_most_recent_live_run(tmp_path, monkeypatch):
    monkeypatch.setattr(_tp50, "LIVE_LOG_DIR", tmp_path)
    (tmp_path / "f4_tp50_f4_tp50_20260818_090000.json").write_text(
        json.dumps({"run_id": "f4_tp50_20260818_090000", "live": True}), encoding="utf-8",
    )
    (tmp_path / "f4_tp50_f4_tp50_20260819_090000.json").write_text(
        json.dumps({"run_id": "f4_tp50_20260819_090000", "live": True}), encoding="utf-8",
    )
    result = _find_previous_live_run()
    assert result["run_id"] == "f4_tp50_20260819_090000"


# ======================================================================
# 2026-08-21朝 通知全面監査: 【前日の実績】が本日自身のLive runを誤って
# 表示していたバグ（Live 08:49実行後にDry Run 09:45実行 → 前日欄に本日の
# 日付が出た）の回帰テスト、およびデータ取得失敗時のINVALID明示。
# ======================================================================
def test_previous_live_run_lookup_excludes_same_day_run(tmp_path, monkeypatch):
    """同一日のLive runは「前日の実績」として絶対に採用してはならない
    （2026-08-21 09:46 実インシデントの回帰テスト）。"""
    monkeypatch.setattr(_tp50, "LIVE_LOG_DIR", tmp_path)
    (tmp_path / "f4_tp50_f4_tp50_20260820_084900.json").write_text(
        json.dumps({"run_id": "f4_tp50_20260820_084900", "live": True,
                    "run_started_at": "2026-08-20 08:49:05"}), encoding="utf-8",
    )
    (tmp_path / "f4_tp50_f4_tp50_20260821_084900.json").write_text(
        json.dumps({"run_id": "f4_tp50_20260821_084900", "live": True,
                    "run_started_at": "2026-08-21 08:49:05"}), encoding="utf-8",
    )
    result = _find_previous_live_run(before_date="2026-08-21")
    assert result["run_id"] == "f4_tp50_20260820_084900"


def test_previous_live_run_lookup_returns_none_when_only_same_day_run_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(_tp50, "LIVE_LOG_DIR", tmp_path)
    (tmp_path / "f4_tp50_f4_tp50_20260821_084900.json").write_text(
        json.dumps({"run_id": "f4_tp50_20260821_084900", "live": True,
                    "run_started_at": "2026-08-21 08:49:05"}), encoding="utf-8",
    )
    assert _find_previous_live_run(before_date="2026-08-21") is None


def test_previous_live_run_date_falls_back_to_run_id_when_run_started_at_missing():
    """2026-08-20 09:36手動遅延Live run（run_started_atフィールド追加前の旧スキーマ
    記録）のような、run_started_at欠落レコードでも日付表示・同日除外フィルタが
    正しく機能することの回帰テスト。"""
    prev_run = {"run_id": "f4_tp50_20260820_093611", "live": True,
                "order_submission_results": [], "exits_detail": [], "funded_detail": [],
                "score_replacement": {"decisions": []}}
    with patch("src.run_live_signal_f4_tp50._find_previous_live_run", return_value=prev_run):
        rs = _base_result_summary(live=False)
        body = _build_tp50_notification_body(rs, [], [])
    prev_block = body.split("【前日の実績】")[1].split("【本日の判断】")[0]
    assert "2026/08/20" in prev_block
    assert "日付不明" not in prev_block


def test_find_previous_live_run_same_day_exclusion_uses_run_id_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(_tp50, "LIVE_LOG_DIR", tmp_path)
    (tmp_path / "f4_tp50_f4_tp50_20260821_093611.json").write_text(
        json.dumps({"run_id": "f4_tp50_20260821_093611", "live": True}), encoding="utf-8",
    )
    assert _find_previous_live_run(before_date="2026-08-21") is None


def test_dry_run_shows_previous_day_unavailable_when_no_prior_live_run():
    """0件（前日実行あり・取引なし）とN/A（前日実行記録自体がない）を明確に区別する。"""
    rs = _base_result_summary(live=False)
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "【前日の実績】" in body
    assert "N/A（前回LIVE実行記録が見つかりません）" in body


def test_missing_critical_fields_detects_none_values():
    rs = _base_result_summary(live=False, available_cash=None)
    assert _tp50._missing_critical_fields(rs) == ["available_cash"]
    rs2 = _base_result_summary(live=False)
    assert _tp50._missing_critical_fields(rs2) == []


def test_dry_run_body_shows_invalid_banner_when_cash_missing():
    """Cash取得失敗時、Metadata等の他フィールドが独立にOKを名乗っても、
    レポート全体がINVALIDだと明示する（2026-08-21朝 実インシデント: Cash=N/Aなのに
    Metadata=OKと出て矛盾していた問題への対応）。"""
    rs = _base_result_summary(live=False, available_cash=None)
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "DRY RUN INVALID" in body
    assert "available_cash" in body


def test_live_body_shows_invalid_banner_when_positions_count_missing():
    rs = _base_result_summary(live=True, order_submission_results=[], positions_count=None)
    body = _build_tp50_notification_body(rs, [], [])
    assert "REPORT INVALID" in body


def test_classify_notification_error_when_critical_fields_missing():
    rs = _base_result_summary(live=True, order_submission_results=[], available_cash=None)
    assert _classify_tp50_notification(rs) == "error"


# ======================================================================
# 2026-08-22朝 通知監査: Kabu API障害時にCurrent Portfolio(broker snapshot)を
# UNAVAILABLE表示しつつ、ローカルportfolio_state.json由来のPrevious Known State
# (前回確定保有銘柄)は消さずに参考表示する。実運用ではavailable_cash/
# positions_count/cash_sourceは常に何らかのフォールバック値が入りNoneには
# ならない（Noneになるのは不正なfixtureのみ）ため、cash_source!="broker_live"を
# 実際のbroker取得失敗の判定に使う。
# ======================================================================

def _prev_positions_fixture():
    return [
        {"code": "93440", "entry_date": "2026-08-19", "entry_price": 1224.0, "qty": 100},
        {"code": "48260", "entry_date": "2026-08-18", "entry_price": 534.2, "qty": 100},
    ]


# --- B. Kabu APIだけ失敗 -----------------------------------------------
def test_scenario_b_kabu_api_failure_shows_previous_known_state():
    rs = _base_result_summary(
        live=False, cash_source="unavailable_dry_run_degraded",
        available_cash=0.0, positions_count=0,
        previous_known_positions=_prev_positions_fixture(),
    )
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "DRY RUN INVALID" in body
    assert "Kabu APIから現在ポートフォリオを取得できませんでした" in body
    portfolio_block = body.split("【現在ポートフォリオ】")[1].split("【SYSTEM】")[0]
    assert "現在の実際の口座残高を保証するものではありません" in portfolio_block
    assert "93440" in portfolio_block
    assert "48260" in portfolio_block
    assert "合計 2銘柄" in portfolio_block
    # 今日の判断はINVALIDのため判定不能表示になり、0件と混同しない
    today_block = body.split("【本日の判断】")[1].split("【現在ポートフォリオ】")[0]
    assert "判定不能" in today_block
    assert "実行結果：INVALID" in today_block


# --- C/D. Market Data / Fundamentals 取得失敗 ---------------------------
def test_scenario_cd_fundamentals_missing_marks_invalid():
    rs = _base_result_summary(live=False, fundamentals_freshness=None)
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "DRY RUN INVALID" in body
    assert "Fundamentalsデータを取得できませんでした" in body
    assert "取得不可" in body  # SYSTEM欄 Market Data/Fundamentals


# --- E. 全部失敗 ----------------------------------------------------------
def test_scenario_e_all_failed_shows_invalid_and_previous_known_state():
    rs = _base_result_summary(
        live=False, cash_source="unavailable", available_cash=0.0, positions_count=0,
        fundamentals_freshness=None, previous_known_positions=_prev_positions_fixture(),
    )
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "DRY RUN INVALID" in body
    assert "Kabu APIから現在ポートフォリオを取得できませんでした" in body
    assert "Fundamentalsデータを取得できませんでした" in body
    assert "93440" in body


# --- F. 取得成功で保有11銘柄 -----------------------------------------------
def test_scenario_f_healthy_run_shows_current_portfolio_count():
    rs = _base_result_summary(live=False, cash_source="broker_live", positions_count=11)
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "DRY RUN INVALID" not in body
    portfolio_block = body.split("【現在ポートフォリオ】")[1].split("【SYSTEM】")[0]
    assert "保有銘柄    11件" in portfolio_block


# --- G. 0保有は0件、None/N/Aにしない ---------------------------------------
def test_scenario_g_zero_holdings_shows_explicit_zero_not_na():
    rs = _base_result_summary(
        live=False, cash_source="broker_live", positions_count=0,
        market_value=0.0, last_equity=3000000.0,
    )
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    portfolio_block = body.split("【現在ポートフォリオ】")[1].split("【SYSTEM】")[0]
    assert "保有銘柄    0件" in portfolio_block
    assert "None" not in portfolio_block


def test_scenario_g_no_previous_known_positions_shows_explicit_zero():
    rs = _base_result_summary(
        live=False, cash_source="unavailable_dry_run_degraded",
        available_cash=0.0, positions_count=0, previous_known_positions=[],
    )
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    portfolio_block = body.split("【現在ポートフォリオ】")[1].split("【SYSTEM】")[0]
    assert "0銘柄" in portfolio_block
    assert "None" not in portfolio_block


def test_degraded_cash_source_triggers_invalid_even_when_no_field_is_none():
    """available_cash/positions_count/cash_sourceが全て非None(実運用の実際の
    フォールバック値)でも、cash_source!="broker_live"ならinvalid判定する
    （2026-08-22朝 実インシデント: 実運用コードのフィールドは決してNoneにならない
    ため、_missing_critical_fields単独では実際のbroker障害を検知できなかった）。"""
    rs = _base_result_summary(
        live=True, order_submission_results=[],
        cash_source="capital_state_fallback_dry_run_only",
        available_cash=1000000.0, positions_count=0,
    )
    assert _tp50._missing_critical_fields(rs) == []  # 全フィールド非None
    assert _classify_tp50_notification(rs) == "error"  # だがcash_source degraded → error


# ======================================================================
# 2026-08-22 Daily Dry Run通知 最終仕様化: 【現在ポートフォリオ】全銘柄一覧
# (コード/銘柄名/Score/Entry価格/現在値/損益率)。
# ======================================================================

class _BoardKabuClient:
    """get_board()のみ実装するテスト用スタブ。板情報(銘柄名/現在値)取得用。"""

    def __init__(self, board_map: dict):
        self._board_map = board_map  # {code4: (name, price)}

    def get_board(self, code4: str):
        name, price = self._board_map.get(code4, (code4, None))

        class _Board:
            symbol_name = name
            current_price = price

        return _Board()


def _eleven_holdings():
    codes = ["11110", "17160", "17880", "34570", "378A0", "48260",
             "73250", "73710", "77810", "78120", "93440", "94500"][:11]
    return [{"code": c, "entry_date": "2026-08-19", "entry_price": 1000.0 + i * 10, "qty": 100}
            for i, c in enumerate(codes)]


def _board_map_for(holdings, price_fn=lambda entry: entry * 1.10):
    board = {}
    for p in holdings:
        code4 = p["code"][:-1] if len(p["code"]) == 5 else p["code"]
        board[code4] = (f"銘柄{p['code']}", price_fn(p["entry_price"]))
    return board


# --- A/B/C/D. 保有全件表示・コード+名前+Score・Entry=actual・損益率計算 -----
def test_scenario_abcd_full_holdings_table_with_pnl():
    holdings = _eleven_holdings()
    board = _board_map_for(holdings)  # 現在値 = entry * 1.10 (+10%)
    client = _BoardKabuClient(board)
    score_map = {p["code"]: 55.0 for p in holdings}
    rs = _base_result_summary(
        live=False, cash_source="broker_live", positions_count=len(holdings),
        current_holdings=holdings, market_value=1000000.0, last_equity=3000000.0,
    )
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [], client=client, score_map=score_map)
    portfolio_block = body.split("【現在ポートフォリオ】")[1].split("【SYSTEM】")[0]
    for p in holdings:
        assert p["code"] in portfolio_block  # A: 全件表示
        assert f"銘柄{p['code']}" in portfolio_block  # B: 銘柄名
    assert "Score:55.0" in portfolio_block  # B: Score
    assert f"Entry:{_tp50._fmt_yen(holdings[0]['entry_price'])}" in portfolio_block  # C: actual entry
    assert "損益:+10.0" in portfolio_block or "損益:+9.9" in portfolio_block  # D: 概ね+10%


# --- E. 現在値取得失敗 -----------------------------------------------------
def test_scenario_e_current_price_unavailable_shows_explicit_unresolved():
    holdings = [{"code": "93440", "entry_date": "2026-08-19", "entry_price": 1224.0, "qty": 100}]
    client = _BoardKabuClient({})  # get_boardは常にNoneを返す
    rs = _base_result_summary(
        live=False, cash_source="broker_live", positions_count=1, current_holdings=holdings,
        market_value=100000.0, last_equity=3000000.0,
    )
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [], client=client)
    portfolio_block = body.split("【現在ポートフォリオ】")[1].split("【SYSTEM】")[0]
    assert "現在値:取得不可" in portfolio_block
    assert "損益:現在値取得不可のため計算不可" in portfolio_block


# --- F. Entry価格取得失敗 ---------------------------------------------------
def test_scenario_f_entry_price_unavailable_shows_explicit_unresolved():
    holdings = [{"code": "93440", "entry_date": None, "entry_price": None, "qty": 100}]
    board = {"9344": ("アクシスコンサルティング", 1300.0)}
    client = _BoardKabuClient(board)
    rs = _base_result_summary(
        live=False, cash_source="broker_live", positions_count=1, current_holdings=holdings,
        market_value=100000.0, last_equity=3000000.0,
    )
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [], client=client)
    portfolio_block = body.split("【現在ポートフォリオ】")[1].split("【SYSTEM】")[0]
    assert "Entry:取得不可" in portfolio_block
    assert "損益:Entry価格取得不可のため計算不可" in portfolio_block


# --- G. Kabu API失敗 -> Current Portfolio UNAVAILABLE + Previous Known State
def test_scenario_g_current_portfolio_unavailable_label_present():
    rs = _base_result_summary(
        live=False, cash_source="unavailable_dry_run_degraded",
        available_cash=0.0, positions_count=0,
        previous_known_positions=_prev_positions_fixture(),
    )
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "Current Portfolio: UNAVAILABLE" in body
    assert "【Previous Known State】" in body


# --- J. BUY候補: theoretical price であることを明記 -------------------------
def test_scenario_j_buy_candidate_labels_price_as_theoretical():
    rs = _base_result_summary(live=False, funded_detail=[
        {"code": "48260", "estimated_fill_price": 521.0},
    ])
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    today_block = body.split("【本日の判断】")[1].split("【現在ポートフォリオ】")[0]
    assert "理論Entry価格" in today_block
    assert "※BUY候補の価格は実約定価格ではない" in today_block


def test_notification_body_never_contains_static_sending_label():
    """"Notification 送信中"は送信前に固定で書かれる自己言及的な誤表示のため撤去済み
    （2026-08-21朝 ユーザー指摘の回帰テスト）。"""
    rs = _base_result_summary(live=False)
    with _no_prev_run():
        body = _build_tp50_notification_body(rs, [], [])
    assert "送信中" not in body
    rs_live = _base_result_summary(live=True, order_submission_results=[])
    body_live = _build_tp50_notification_body(rs_live, [], [])
    assert "送信中" not in body_live


def test_same_day_multiple_dry_runs_do_not_contradict_each_other():
    """同一日に複数回Dry Runを実行しても、【前日の実績】は常に同じ（前営業日の）
    Live runを指し続け、実行のたびに内容が変わらないこと。"""
    rs1 = _base_result_summary(live=False, run_id="f4_tp50_20260821_094500",
                                run_started_at="2026-08-21 09:45:00")
    rs2 = _base_result_summary(live=False, run_id="f4_tp50_20260821_104500",
                                run_started_at="2026-08-21 10:45:00")
    prev_run = {"run_id": "f4_tp50_20260820_084900", "live": True,
                "run_started_at": "2026-08-20 08:49:05", "order_submission_results": [],
                "exits_detail": [], "funded_detail": [], "score_replacement": {"decisions": []}}
    with patch("src.run_live_signal_f4_tp50._find_previous_live_run", return_value=prev_run):
        body1 = _build_tp50_notification_body(rs1, [], [])
        body2 = _build_tp50_notification_body(rs2, [], [])
    prev1 = body1.split("【前日の実績】")[1].split("【本日の判断】")[0]
    prev2 = body2.split("【前日の実績】")[1].split("【本日の判断】")[0]
    assert prev1 == prev2
    assert "2026/08/20" in prev1


# ======================================================================
# Signal Freshness Guard（2026-08-20夜 Production Gate監査で追加）:
# as_of(市場データ最終営業日)が実カレンダー日から著しく乖離した場合、
# 新規BUYのみblockする（Exit/リスク管理は常に継続）。
# ======================================================================
def test_asof_staleness_normal_weekend_gap_does_not_block():
    # 金曜のas_ofを月曜に評価 -> busday_count=1営業日
    fri = pd.Timestamp("2026-08-14").date()
    mon = pd.Timestamp("2026-08-17").date()
    bdays = compute_asof_staleness_bdays(fri, mon)
    assert bdays == 1
    assert should_block_buy_for_stale_asof(bdays) is False


def test_asof_staleness_severe_gap_blocks_buy():
    # 市場データパイプラインが1週間停止していたケース
    stale_asof = pd.Timestamp("2026-08-10").date()
    real_today = pd.Timestamp("2026-08-20").date()
    bdays = compute_asof_staleness_bdays(stale_asof, real_today)
    assert bdays > 4
    assert should_block_buy_for_stale_asof(bdays) is True


def test_notification_shows_asof_stale_block_warning():
    rs = _base_result_summary(live=True, order_submission_results=[],
                               asof_stale_block=True, asof_staleness_bdays=6)
    assert _classify_tp50_notification(rs) == "warning"
    body = _build_tp50_notification_body(rs, [], [])
    assert "市場データ鮮度異常" in body
    assert "6営業日" in body


def test_notification_no_asof_warning_when_not_blocked():
    rs = _base_result_summary(live=True, order_submission_results=[],
                               asof_stale_block=False, asof_staleness_bdays=1)
    assert _classify_tp50_notification(rs) == "success"
    body = _build_tp50_notification_body(rs, [], [])
    assert "市場データ鮮度異常" not in body


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
