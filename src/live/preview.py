"""
src/live/preview.py
DRY実行時の LIVE PREVIEW 表示ロジック（実発注ロジックとは完全に独立）。
"""


def print_live_preview(result, order_objects: list) -> None:
    """
    DRY実行時に「もし今日LIVEだったら」の発注プレビューを表示する。
    order_objects は exit orchestrator / PI overlay 適用後の最終リスト。
    実発注ロジックには一切触れない（表示のみ）。
    """
    sig_map = {s["symbol"]: s for s in result.signals}
    ps = result.portfolio_summary or {}

    buy_orders    = [o for o in order_objects if o.side == "BUY"]
    shadow_orders = [o for o in order_objects if o.side == "SHADOW_BUY"]
    sell_orders   = [o for o in order_objects if o.side == "SELL"]
    hold_sigs     = [s for s in result.signals if s.get("signal") == 1 and s.get("currently_holding")]

    total_buy_amount  = sum(o.estimated_amount for o in buy_orders)
    total_sell_amount = sum(o.estimated_amount for o in sell_orders)
    available_cash    = float(ps.get("available_cash", 0))
    cash_after        = available_cash - total_buy_amount + total_sell_amount

    print("\n" + "=" * 64)
    print("  ★ LIVE PREVIEW — 本日LIVE実行時の予定発注")
    print("=" * 64)
    print(f"  planned_buy_count    : {len(buy_orders)}")
    print(f"  planned_sell_count   : {len(sell_orders)}")
    print(f"  planned_capital_usage: ¥{total_buy_amount:,.0f}")
    print(f"  available_cash_now   : ¥{available_cash:,.0f}")
    print(f"  cash_after_orders    : ¥{cash_after:,.0f}")

    if buy_orders:
        print(f"\n  [BUY候補] {len(buy_orders)}件")
        print(f"  {'銘柄':<10} {'数量':>6} {'参考価格':>10} {'発注金額':>12} {'RSR':>6} {'rank':>5}  理由")
        print("  " + "-" * 74)
        for o in buy_orders:
            s = sig_map.get(o.symbol, {})
            print(
                f"  {o.symbol:<10} {o.qty:>6}株"
                f" ¥{o.estimated_price:>9,.0f}"
                f" ¥{o.estimated_amount:>11,.0f}"
                f" {s.get('rsr', 0.0):>6.1f}"
                f" {s.get('rsr_rank', '-'):>5}"
                f"  {o.reason[:35]}"
            )

    if shadow_orders:
        print(f"\n  [SHADOW BUY候補] {len(shadow_orders)}件（実発注対象外・監視専用）")
        for o in shadow_orders:
            s = sig_map.get(o.symbol, {})
            print(
                f"  {o.symbol:<10} {o.qty:>6}株"
                f" ¥{o.estimated_price:>9,.0f}"
                f" {s.get('rsr', 0.0):>6.1f}  {o.reason[:35]}"
            )

    if sell_orders:
        print(f"\n  [SELL候補] {len(sell_orders)}件")
        print(f"  {'銘柄':<10} {'数量':>6} {'参考価格':>10} {'発注金額':>12} {'保有日':>5}  理由")
        print("  " + "-" * 70)
        for o in sell_orders:
            s = sig_map.get(o.symbol, {})
            hold_d = s.get("hold_days", 0)
            print(
                f"  {o.symbol:<10} {o.qty:>6}株"
                f" ¥{o.estimated_price:>9,.0f}"
                f" ¥{o.estimated_amount:>11,.0f}"
                f" {hold_d:>4}d"
                f"  {o.reason[:35]}"
            )

    if hold_sigs:
        print(f"\n  [HOLD継続] {len(hold_sigs)}件（新規発注なし）")
        print(f"  {'銘柄':<10} {'RSR':>6} {'rank':>5} {'保有日':>5} {'含損益':>8}  停止価格")
        print("  " + "-" * 60)
        for s in sorted(hold_sigs, key=lambda x: x.get("rsr_rank", 99)):
            stop_str = f"¥{s['stop_price']:,.0f}" if s.get("stop_price", 0) > 0 else "    —"
            pnl_str  = f"{s.get('unrealized_pnl_pct', 0.0):>+7.1%}"
            print(
                f"  {s['symbol']:<10}"
                f" {s.get('rsr', 0.0):>6.1f}"
                f" {s.get('rsr_rank', 0):>5}"
                f" {s.get('hold_days', 0):>4}d"
                f" {pnl_str}"
                f"  {stop_str}"
            )

    if not buy_orders and not sell_orders and not shadow_orders and not hold_sigs:
        print("\n  発注予定なし")

    print("=" * 64)
