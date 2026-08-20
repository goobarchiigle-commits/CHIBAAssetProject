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
FIXED_LOT_SIZE = _tp50.FIXED_LOT_SIZE
STRATEGY_TYPE = _tp50.STRATEGY_TYPE
OrderInstruction = _tp50.OrderInstruction
evaluate_buy_sizing = _tp50.evaluate_buy_sizing
evaluate_exits = _tp50.evaluate_exits
apply_fill_metadata_updates = _tp50.apply_fill_metadata_updates
compute_order_submission_timeout_sec = _tp50.compute_order_submission_timeout_sec

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
    """Even a malformed result_summary must not crash the caller (fire-and-forget)."""
    _send_tp50_notification({"live": False}, None, None)  # missing keys, wrong types for lists


# --- test 6: FUNDED / sizing must be labeled as simulated, not a real order
def test_notification_body_labels_funded_as_simulated_not_real_order():
    rs = _base_result_summary(live=False)
    body = _build_tp50_notification_body(rs, [], [])
    assert "SIMULATED FUNDING" in body
    assert "実注文ではありません" in body
    assert "order submission results: NONE" in body


def test_notification_body_dry_run_reports_zero_real_orders():
    rs = _base_result_summary(live=False)
    body = _build_tp50_notification_body(rs, [], [])
    assert "実発注件数（real broker order attempts）: 0" in body


# --- test 7: 5-digit code 48260 -> symbol_4digit 4826 appears in the body -
def test_notification_body_shows_4digit_kabu_symbol_for_incident_code():
    o = OrderInstruction(symbol="48260", side="BUY", qty=FIXED_LOT_SIZE,
                          estimated_price=521.0, reason="F4_TP50_entry_signal",
                          symbol_4digit=_tp50.to_kabu_symbol("48260"))
    rs = _base_result_summary(live=False)
    body = _build_tp50_notification_body(rs, [o], [])
    assert "48260" in body
    assert "kabu_symbol=4826" in body
    assert "kabu_symbol=48260" not in body  # must never show the unconverted 5-digit form as the kabu symbol


def test_notification_body_shows_4digit_kabu_symbol_in_submitted_results():
    rs = _base_result_summary(live=True, order_submission_results=[
        {"symbol": "48260", "symbol_4digit": "4826", "side": "BUY", "qty": 100,
         "success": False, "order_id": None, "error": "HTTP 400: symbol not found",
         "http_status": 400},
    ])
    body = _build_tp50_notification_body(rs, [], [])
    assert "kabu_symbol=4826" in body


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
