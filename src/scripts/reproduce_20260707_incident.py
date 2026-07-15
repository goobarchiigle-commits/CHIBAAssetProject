"""
src/scripts/reproduce_20260707_incident.py

2026-07-07 4銘柄同時保有インシデントの再現試験。

実際のログ（logs/live/20260707_084405_signals.json /
                     20260707_084405_orders.json）
と runtime/portfolio_state.json（当日08:44時点のスナップショット）から
実測値を抽出し、修正後の SignalBridge._build_shadow_orders() へ
同一条件を与えて「5301.T が発注されないこと」を検証する。

実測値（ログから抽出・改変なし）:
  - 08:44時点の実保有        : 6981.T, 2802.T（2件。6506.T/5301.T はこの後の処理で追加）
  - 通常BUYパイプラインの結果 : 6506.T 100株 @7449（trend_follow）→ held=3 で満枠
  - Shadow候補               : 5301.T RSR62=90.3, shadow_rsr_pass=8,
                                live_top10_median=87.5
                                （orders.json の reason 文字列に実測値として記録済み）
  - 実際に発生した注文        : 5301.T SHADOW_BUY 300株 @1758.5 → 実発注（バグ）

期待結果（修正後）:
  - 6506.T のみ実保有に追加される（3件で満枠、正常）
  - 5301.T は remaining_slots=0 のため Shadow候補生成自体がスキップされる
  - Broker送信件数 = 1（6506.Tのみ。5301.Tはゼロ）
  - max_positions=3 違反 = 0

実行:
    cd C:/ai-trading
    python -m src.scripts.reproduce_20260707_incident
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


def _load_incident_artifacts() -> dict:
    sig_path = _ROOT / "logs" / "live" / "20260707_084405_signals.json"
    ord_path = _ROOT / "logs" / "live" / "20260707_084405_orders.json"
    signals      = json.loads(sig_path.read_text(encoding="utf-8"))[0]["signals"]
    orders_entry = json.loads(ord_path.read_text(encoding="utf-8"))[0]
    orders       = orders_entry["orders"]
    send_results = orders_entry["send_results"]
    return {"signals": signals, "orders": orders, "send_results": send_results}


def _make_price_df(last_close: float, n: int = 40) -> pd.DataFrame:
    closes = [last_close * (1 + 0.001 * i) for i in range(n)]
    closes[-1] = last_close
    return pd.DataFrame({
        "Close": closes,
        "High":  [c * 1.01 for c in closes],
        "Low":   [c * 0.99 for c in closes],
    })


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    heavy = {"kabusapi": MagicMock(), "kabu_station_api": MagicMock(), "pandas_datareader": MagicMock()}
    with patch.dict("sys.modules", heavy):
        from src.kabusapi.signal_bridge import SignalBridge, OrderInstruction

    artifacts = _load_incident_artifacts()
    shadow_order_real = next(o for o in artifacts["orders"] if o["symbol"] == "5301.T")
    shadow_send_real   = next(r for r in artifacts["send_results"] if r["symbol"] == "5301.T")
    # reason 文字列に実測値が埋め込まれている:
    #   "SHADOW_BUY: RSR62=90.3 (>87.5=live_top10_median) shadow_rsr_pass=8"
    reason = shadow_order_real["reason"]
    rsr62         = float(reason.split("RSR62=")[1].split(" ")[0])
    live_median   = float(reason.split(">")[1].split("=")[0])
    shadow_pass   = int(reason.split("shadow_rsr_pass=")[1])

    print("=" * 70)
    print("  2026-07-07 4銘柄同時保有インシデント再現試験")
    print("=" * 70)
    print(f"  実測値: 5301.T RSR62={rsr62} live_top10_median={live_median}"
          f" shadow_rsr_pass={shadow_pass}")
    print(f"  実測値: 08:44時点の実保有=2件（6981.T, 2802.T）")
    print(f"  実測値: 通常BUYパイプライン結果=1件（6506.T, trend_follow）")
    print(f"  実際の結果（修正前）: 5301.T SHADOW_BUY 300株 @{shadow_order_real['estimated_price']}"
          f" が実発注された（order_id={shadow_send_real.get('order_id', 'N/A')},"
          f" success={shadow_send_real.get('success')}）")
    print("-" * 70)

    # ── 修正後コードで同一条件を再現 ──────────────────────────────
    bridge = MagicMock(spec=SignalBridge)
    bridge.capital             = 3_000_000
    bridge.max_single_weight   = 0.25
    bridge.shadow_universe_tickers = {"5301.T": "化学"}
    bridge.pre_trade_risk_check = MagicMock(return_value=True)  # 通過したと仮定しても結果は変わらない
    bridge._build_shadow_orders = SignalBridge._build_shadow_orders.__get__(bridge, type(bridge))

    current_positions = {"6981.T": {"qty": 100}, "2802.T": {"qty": 100}}
    live_orders = [OrderInstruction(
        symbol="6506.T", symbol_4digit="6506", sector="機械", side="BUY",
        qty=100, order_type="MARKET_OPEN", estimated_price=7449.0,
        estimated_amount=744900.0, reason="trend_follow fallback=False",
    )]
    diag = {
        "shadow_rsr_pass": shadow_pass,
        "rsr_distribution": [{"rsr": live_median} for _ in range(10)],  # median再現用
        "shadow_rsr62_scores": {"5301.T": rsr62},
    }
    universe_raw = {"5301.T": {"df": _make_price_df(1741.0)}}

    orders, metrics, new_virtual, _ = bridge._build_shadow_orders(
        diag=diag,
        universe_raw=universe_raw,
        current_positions=current_positions,
        available_cash=1_491_141.0,
        cb_active=False,
        live_orders=live_orders,
        shadow_virtual_positions={},
        today_str="2026-07-07",
        effective_max_pos=3,
    )

    held_after = len(current_positions) + sum(1 for o in live_orders if o.side == "BUY")
    broker_send_count_5301 = len(orders)  # 5301.T向けにBrokerへ送信されるorder数（常にこの経路はorders空）

    print(f"  修正後: remaining_slots = {metrics['shadow_remaining_slots']}"
          f" (max_positions=3 - held=2 - pending_buy=1)")
    print(f"  修正後: Shadow経路が返す発注可能Order数 = {len(orders)}（常に0固定）")
    print(f"  修正後: 5301.Tの仮想エントリー記録        = {'5301.T' in new_virtual}"
          f"（枠なしのため記録もされない）")
    print(f"  修正後: 実保有見込み（6981+2802+6506）    = {held_after}件 / max_positions=3")
    print(f"  修正後: max_positions違反件数              = {max(0, held_after - 3) + broker_send_count_5301}")
    print("-" * 70)

    ok = (
        broker_send_count_5301 == 0
        and "5301.T" not in new_virtual
        and held_after == 3
        and metrics["shadow_remaining_slots"] == 0
    )
    print(f"  判定: {'PASS — 5301.T不発注・6506.Tのみ・max_positions違反なし' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
