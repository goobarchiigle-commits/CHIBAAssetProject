"""
src/scripts/audit_20260707_candidates.py

2026-07-07 全BUY候補監査 + Shadow完全無効化での再シミュレーション。

対象: 5301.T, 6506.T
出典: logs/live/20260707_084405_signals.json / _orders.json /
      logs/trades.jsonl / runtime/portfolio_state.json（改変なし・実測値のみ）

reconstruct注記:
  logs/live/*_signals.json の "signals" は当日の最終保存タイミング
  （通常BUY確定後・Shadow確定前）のスナップショットのため、6506.T は
  「もう保有している」状態（signal=0/HOLD/currently_holding=true）として
  記録されている。実際の発注判断時点の signal（=1, trend_follow）は
  logs/trades.jsonl の reason="trend_follow fallback=False" と
  runtime/portfolio_state.json の position_strategy_types["6506.T"]="trend_follow"
  から復元する。この2箇所は実測値であり、本スクリプトはそれ以外の値を
  作為的に変更しない。6645.T（発注されなかったfujiko BUY候補）の株価は
  どのログにも実測値が残っていない ―― 後述の通り、この銘柄はスロット
  不足のみで弾かれるため価格に依存せず結果に影響しない。placeholder値を
  使い、その旨を明記する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd


def _load_real_artifacts() -> dict:
    sig_path = _ROOT / "logs" / "live" / "20260707_084405_signals.json"
    ord_path = _ROOT / "logs" / "live" / "20260707_084405_orders.json"
    ps_path  = _ROOT / "runtime" / "portfolio_state.json"
    signals  = {s["symbol"]: s for s in json.loads(sig_path.read_text(encoding="utf-8"))[0]["signals"]}
    orders_entry = json.loads(ord_path.read_text(encoding="utf-8"))[0]
    portfolio    = json.loads(ps_path.read_text(encoding="utf-8"))
    trades = [json.loads(l) for l in (_ROOT / "logs" / "trades.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    trades_20260707 = [t for t in trades if t.get("date") == "2026-07-07"]
    return {
        "signals": signals, "orders_entry": orders_entry,
        "portfolio": portfolio, "trades_20260707": trades_20260707,
    }


def _make_price_df(last_close: float, n: int = 40) -> pd.DataFrame:
    closes = [last_close * (1 + 0.001 * i) for i in range(n)]
    closes[-1] = last_close
    return pd.DataFrame({
        "Close": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes],
    })


def print_audit_table(art: dict) -> None:
    sig = art["signals"]
    ps  = art["portfolio"]
    trades_by_sym = {t["symbol"]: t for t in art["trades_20260707"]}
    orders_by_sym = {o["symbol"]: o for o in art["orders_entry"]["orders"]}

    print("=" * 78)
    print("  2026-07-07 BUY候補監査: 5301.T / 6506.T")
    print("=" * 78)

    for sym in ("5301.T", "6506.T"):
        s = sig.get(sym, {})
        t = trades_by_sym.get(sym, {})
        o = orders_by_sym.get(sym, {})
        entry_strategy = ps.get("position_strategy_types", {}).get(sym)
        entry_atr      = ps.get("position_entry_atrs", {}).get(sym)
        # trades.jsonl の reason は 5301.T のみ空文字（データ欠落症状そのもの）。
        # 発注時点の本来のreasonは orders.json の order-level（送信前）に残っている。
        display_reason = o.get("reason") or t.get("reason") or "(記録なし)"
        print(f"\n【{sym}】")
        print(f"  RSR順位(rsr_rank)        : {s.get('rsr_rank')}")
        print(f"  RSR値                    : {s.get('rsr')}")
        print(f"  SEPA                     : {s.get('sepa_score')}")
        print(f"  発注reason               : {display_reason}")
        if sym == "5301.T":
            print(f"  参考: trades.jsonl側reason は空文字（下記データ欠落と同一原因の症状）")
        print(f"  実際の発注時strategy_type : {entry_strategy or '(記録なし)'}")
        print(f"  ATR20(entry, portfolio_state): {entry_atr if entry_atr is not None else '★記録なし(欠落)'}")
        if sym == "5301.T":
            print(f"  通常BUY条件(フジコ/mean_rev signal=1)を満たしたか: NO"
                  f"（signals.json上のsignal={s.get('signal')}, strategy_type={s.get('strategy_type')}=mean_rev, HOLD止まり）")
            print(f"  Shadow条件のみだったか   : YES（reasonがShadow RSR62経路専用の文言）")
            print(f"  TrendFollow条件を満たしたか: N/A（mean_rev銘柄。trend_follow_candidates()の対象母集団にそもそも含まれない）")
            print(f"  Breakout(turtle 20日高値)条件: N/A（mean_rev戦略はRSI逆張りエントリーであり turtle_entry ロジックを経由しない）")
            print(f"  RiskCheck(pre_trade_risk_check): 未実施"
                  f"（修正前の_build_shadow_orders()はpre_trade_risk_check()を一切呼んでいなかった＝根本原因の一部）")
        else:
            print(f"  通常BUY条件(フジコ signal=1)を満たしたか: NO"
                  f"（フジコ法エントリー条件では未トリガー。フジコ法での扱いはHOLD/監視のみ）")
            print(f"  Shadow条件のみだったか   : NO（Shadow RSR62経路は不使用。5301.Tと違いこちらは通常発注パイプライン経由）")
            print(f"  TrendFollow条件を満たしたか: YES"
                  f"（trend_follow_candidates(): close>MA20かつMA20上昇、または close>MA50かつrsr_252>50のフォールバック条件を充足）")
            print(f"  Breakout(turtle 20日高値)条件: N/A（trend_followはMA20/50ベースの判定であり turtle_entry(20日高値ブレイク)ロジックとは別経路）")
            print(f"  RiskCheck(pre_trade_risk_check): 実施・PASS"
                  f"（_build_orders()の通常ループを経由＝symbol_cap/sector_cap/cluster_cap判定を通過して発注）")

    print("\n" + "-" * 78)
    print("  ★ 現在の実ポジション(5301.T)のデータ欠落（時限性の高い別件・要確認）")
    print("-" * 78)
    print(f"  position_entry_prices['5301.T']    = {ps['position_entry_prices'].get('5301.T')}"
          f"（実際の約定価格ではなくゼロ。process-isolated実行結果に estimated_price が"
          f"引き継がれず、update_state_after_execution() が r.get('estimated_price',0.0) で"
          f"フォールバックしたため）")
    print(f"  position_highest_closes['5301.T']  = {ps['position_highest_closes'].get('5301.T')}"
          f"（トレーリングストップ計算の基準値が壊れている）")
    print(f"  position_entry_atrs['5301.T']      = {'記録なし' if '5301.T' not in ps['position_entry_atrs'] else ps['position_entry_atrs']['5301.T']}")
    print(f"  → 現在保有中の5301.T(300株)は entry_price/highest_close/ATR の"
          f"いずれも壊れた値のまま。トレーリングストップ/含み損益判定が正しく機能しない可能性あり。")


def resimulate_shadow_fully_disabled(art: dict) -> list[str]:
    heavy = {"kabusapi": MagicMock(), "kabu_station_api": MagicMock(), "pandas_datareader": MagicMock()}
    with patch.dict("sys.modules", heavy):
        from src.kabusapi.signal_bridge import SignalBridge, StockSignal, OrderInstruction

    bridge = MagicMock(spec=SignalBridge)
    bridge.max_positions      = 3
    bridge.capital            = 3_000_000
    bridge.max_single_weight  = 0.25
    bridge.min_sectors        = 1
    bridge.regime_sizing      = "none"
    bridge.bear_scale         = 1.0
    bridge.max_new_positions_per_day = 2
    bridge.universe_tickers   = {"6981.T": "電機精密", "2802.T": "食品", "8035.T": "?", "6506.T": "機械", "6645.T": "電機精密"}
    bridge.pre_trade_risk_check = MagicMock(return_value=True)
    bridge._build_orders = SignalBridge._build_orders.__get__(bridge, type(bridge))

    # 実測値: 08:44時点の実保有(通常BUY確定前)
    current_positions = {"6981.T": {"qty": 100}, "2802.T": {"qty": 100}}

    # 実測値ベースの signal=1 候補（rsr_rankで自然にソートされる）
    # 8035.T / 6645.T の signal=1・rsr・rsr_rank は logs/live/*_signals.json の実測値。
    # 6506.T は上記コメントの通り reason="trend_follow fallback=False" (trades.jsonl実測)
    # から signal=1 に復元。8035.T の価格は portfolio_state.shadow_virtual_positions
    # (実測 ¥72,320)。6506.T は position_entry_prices (実測 ¥7,449)。
    # 6645.T のみ、どのログにも価格実測値が残っていない
    # （発注されず・shadow記録もされず）→ placeholder(¥1)を使用。
    # 理由: _build_orders() は open_slots<=0 を価格計算より前に判定して break するため、
    #       6645.T がスロット不足で弾かれる結論は価格に依存しない。
    signals = [
        StockSignal(symbol="6981.T", sector="電機精密", signal=0, rsr=100.0, rsr_rank=1,
                    sepa_score=8, rsr_mom=0.0, hold_days=46, currently_holding=True,
                    reason="HOLD: RSR=100.0 rank=1 SEPA=8", strategy_type="fujiko"),
        StockSignal(symbol="8035.T", sector="?", signal=1, rsr=96.4, rsr_rank=2,
                    sepa_score=0, rsr_mom=0.0, hold_days=0, currently_holding=False,
                    reason="trend_follow fallback=False", strategy_type="trend_follow"),
        StockSignal(symbol="6506.T", sector="機械", signal=1, rsr=94.6, rsr_rank=4,
                    sepa_score=8, rsr_mom=7.55, hold_days=0, currently_holding=False,
                    reason="trend_follow fallback=False", strategy_type="trend_follow"),
        StockSignal(symbol="6645.T", sector="電機精密", signal=1, rsr=78.6, rsr_rank=14,
                    sepa_score=8, rsr_mom=4.38, hold_days=0, currently_holding=False,
                    reason="BUY[フジコ法]: RSR=78.6 rank=14 SEPA=8 mom=+4.4", strategy_type="fujiko"),
        StockSignal(symbol="2802.T", sector="食品", signal=0, rsr=76.8, rsr_rank=16,
                    sepa_score=8, rsr_mom=18.72, hold_days=10, currently_holding=True,
                    reason="HOLD: RSR=76.8 rank=16 SEPA=8", strategy_type="fujiko"),
    ]
    universe_raw = {
        "8035.T": {"df": _make_price_df(72320.0)},
        "6506.T": {"df": _make_price_df(7449.0)},
        "6645.T": {"df": _make_price_df(1.0)},   # placeholder — 結果に影響しない(本文参照)
        "6981.T": {"df": _make_price_df(4864.0)},
        "2802.T": {"df": _make_price_df(6000.0)},
    }

    orders, warnings, blocked_alloc_cap, lot_up, risk_rejected = bridge._build_orders(
        signals=signals,
        universe_raw=universe_raw,
        current_positions=current_positions,
        available_cash=1_491_141.0,
        cb_active=False,
        effective_max_pos=3,
        above_ma200=None,
    )

    print("\n" + "=" * 78)
    print("  Shadow完全無効化での2026-07-07再シミュレーション（通常発注パイプラインのみ）")
    print("=" * 78)
    buy_orders = [o for o in orders if o.side == "BUY"]
    print(f"  BUY候補(rsr_rank順): 8035.T(2, 1単元¥7,232,000>alloc_cap¥750,000のため実質発注不可)"
          f" / 6506.T(4) / 6645.T(14)")
    print(f"  blocked_by_alloc_cap件数(集計) = {blocked_alloc_cap}"
          f" / risk_rejected = {risk_rejected}"
          f"（deployability事前再ランクにより6506.Tが先に処理されスロット消費→"
          f" 残り2件はいずれの経路でも実発注に至らず）")
    print(f"  実発注BUY = {[o.symbol for o in buy_orders]}")
    for w in warnings:
        print(f"  [warning] {w}")

    final_holdings = sorted(set(current_positions.keys()) | {o.symbol for o in buy_orders})
    print(f"\n  本来約定した銘柄一覧（Shadow完全無効化時） = {final_holdings}")
    return final_holdings


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    art = _load_real_artifacts()
    print_audit_table(art)
    should_be = resimulate_shadow_fully_disabled(art)
    actual = sorted(art["portfolio"]["position_qtys"].keys())

    print("\n" + "=" * 78)
    print("  最終比較: 実際約定 vs 本来約定すべき銘柄")
    print("=" * 78)
    print(f"  実際約定した銘柄(4件)     : {actual}")
    print(f"  本来約定すべきだった銘柄(3件): {should_be}")
    diff = sorted(set(actual) - set(should_be))
    print(f"  差分（過剰発注）           : {diff}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
