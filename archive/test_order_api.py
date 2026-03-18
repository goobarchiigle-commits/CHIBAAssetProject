"""
test_order_api.py
kabuステーション API 発注テスト（少額・最小ロット）

【目的】
  FundType="AA" + AccountType=4 の設定で実際に発注できるかを確認する。
  APIエラー（預り区分未設定など）の最終確認。

【デフォルト銘柄】
  5401.T 日本製鉄（1単元100株 ≈ ¥61,130）
  → G宇宙銘柄・平均回帰戦略・バックテストCAGR+123.6%

【使い方】
  # 発注テスト（確認あり）
  python test_order_api.py

  # 銘柄変更
  python test_order_api.py --symbol 8306  # 三菱UFJ

  # 確認スキップ
  python test_order_api.py --yes
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8")

from kabusapi.client import KabuClient, Exchange, Side, OrderType, CashMargin

DEFAULT_SYMBOL = "5401"   # 日本製鉄（4桁コード）
DEFAULT_QTY    = 100       # 最小単元


def parse_args():
    p = argparse.ArgumentParser(description="kabuステーション API 発注テスト")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL, help="銘柄コード4桁（デフォルト: 5401 日本製鉄）")
    p.add_argument("--qty",    type=int, default=DEFAULT_QTY, help="株数（デフォルト: 100株）")
    p.add_argument("--yes","-y", action="store_true", help="確認をスキップ")
    return p.parse_args()


def main():
    args = parse_args()
    symbol = args.symbol.replace(".T", "")   # ".T"除去

    print("=" * 60)
    print("  kabuステーション API 発注テスト")
    print(f"  銘柄: {symbol}  /  数量: {args.qty}株  /  注文種別: 成行")
    print("=" * 60)

    # ---- トークン取得 ----
    print("\n[1/3] トークン取得中...")
    client = KabuClient()
    try:
        token = client.fetch_token()
        print(f"  ✅ トークン取得成功: {token[:8]}...")
    except Exception as e:
        print(f"  ❌ トークン取得失敗: {e}")
        print("  → kabuステーションが起動・ログイン済みか確認してください")
        return 1

    # ---- 板情報取得（現在値確認）----
    print(f"\n[2/3] {symbol}.T の板情報取得中...")
    try:
        board = client.get_board(symbol, Exchange.TSE)
        print(f"  銘柄名     : {board.symbol_name}")
        print(f"  現在値     : ¥{board.current_price:,.0f}")
        print(f"  売気配値   : ¥{board.bid_price:,.0f}")
        print(f"  買気配値   : ¥{board.ask_price:,.0f}")
        ref_price    = board.current_price
        est_amount   = ref_price * args.qty
        print(f"  推定約定額 : ¥{est_amount:,.0f}（{args.qty}株 × ¥{ref_price:,.0f}）")
    except Exception as e:
        print(f"  ❌ 板情報取得失敗: {e}")
        print("  → 銘柄コードが正しいか、取引時間内か確認してください")
        return 1

    # ---- 確認 ----
    print(f"\n[3/3] 発注確認")
    print(f"  BUY  {symbol}.T  {args.qty}株  成行")
    print(f"  推定金額: ¥{est_amount:,.0f}")
    print(f"  !! 実際の発注を行います。取り消しは kabuステーション画面から !!")

    if not args.yes:
        ans = input("\n  発注を実行しますか？ [y/N] > ").strip().lower()
        if ans != "y":
            print("  キャンセルしました。")
            return 0

    # ---- 発注 ----
    print("\n  発注中...")
    try:
        result = client.send_order(
            symbol     = symbol,
            exchange   = Exchange.SOR,             # SOR必須（TSE直接は拒否される）
            side       = Side.BUY,
            qty        = args.qty,
            order_type = OrderType.MARKET,         # 成行（取引時間中）
        )
    except Exception as e:
        print(f"  ❌ 発注エラー: {e}")
        # HTTPエラーの場合はレスポンスボディを表示
        if hasattr(e, "response") and e.response is not None:
            try:
                print(f"  レスポンスボディ: {e.response.json()}")
            except Exception:
                print(f"  レスポンスボディ(raw): {e.response.text}")
        return 1

    # ---- 結果 ----
    print("\n" + "=" * 60)
    if result.success:
        print(f"  ✅ 発注成功！")
        print(f"  OrderId   : {result.order_id}")
        print(f"  ResultCode: {result.result_code}")
        print(f"\n  kabuステーション画面で注文状況を確認してください。")
        print(f"  約定後は run_live_signal.py --live でポジション管理されます。")
    else:
        print(f"  ❌ 発注失敗")
        print(f"  ResultCode: {result.result_code}")
        print(f"  Raw: {result.raw}")
    print("=" * 60)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
