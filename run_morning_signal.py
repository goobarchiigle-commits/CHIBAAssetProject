"""
run_morning_signal.py
毎朝のシグナル生成 + 発注スクリプト

【使い方】
  # ドライラン（発注しない・口座不要）
  python run_morning_signal.py

  # 発注確認あり（実際にAPIへ送信）
  python run_morning_signal.py --live

  # 確認プロンプトをスキップして自動送信（cron/タスクスケジューラ用）
  python run_morning_signal.py --live --yes

  # シグナルのみ表示（JSON保存しない）
  python run_morning_signal.py --no-save

【推奨実行タイミング】
  8:30〜8:55（前場寄付き 9:00 の前）
  → 前日終値データが yfinance に反映されていることを確認してから実行

【Windowsタスクスケジューラへの登録例】
  タスク名    : 朝のシグナル確認
  トリガー    : 毎営業日 08:30
  操作        : python run_morning_signal.py
  作業ディレクトリ: (プロジェクトルートを設定すること)

【CLAUDE.md ルール3 確認】
  .env ファイルに KABU_API_PASSWORD / KABU_TRADE_PASSWORD が設定されていること。
  このスクリプト自体に認証情報を記述しないこと。
"""

import sys
import os
import argparse
import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8")

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("morning_signal")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="フジコ法 朝のシグナル生成スクリプト")
    p.add_argument(
        "--live",
        action="store_true",
        help="kabuステーション API に実際に発注する（デフォルト: ドライラン）",
    )
    p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="発注確認プロンプトをスキップ（--live と併用）",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="JSON シグナルファイルを保存しない",
    )
    p.add_argument(
        "--output-dir",
        default="data/signals",
        help="シグナル JSON の保存先ディレクトリ（デフォルト: data/signals）",
    )
    return p.parse_args()


def print_banner(live: bool) -> None:
    mode = "🔴 LIVE（実発注）" if live else "🟡 DRY RUN（発注なし）"
    print("=" * 60)
    print(f"  フジコ法 朝のシグナル確認")
    print(f"  実行日時 : {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}")
    print(f"  モード   : {mode}")
    print("=" * 60)


def print_signals(result) -> None:
    """シグナルを人間が読みやすい形式で表示する。"""
    orders = result.orders

    if not orders:
        print("\n📭 本日の注文なし（全銘柄 HOLD / 条件不成立）")
    else:
        print(f"\n📋 発注予定: {len(orders)} 件")
        print(f"  {'銘柄':<10} {'売買':<6} {'数量':>6} {'参考価格':>10} {'参考金額':>12}  理由")
        print("  " + "-" * 72)
        for o in orders:
            side_str = "🟢 BUY " if o["side"] == "BUY" else "🔴 SELL"
            print(
                f"  {o['symbol']:<10} {side_str} {o['qty']:>6}株 "
                f"¥{o['estimated_price']:>9,.0f} "
                f"¥{o['estimated_amount']:>11,.0f}  "
                f"{o['reason']}"
            )

    if result.warnings:
        print("\n⚠ 警告:")
        for w in result.warnings:
            print(f"  - {w}")

    # RSR トップ10 をサマリー表示
    top_signals = [s for s in result.signals if s["signal"] in (1, 0)][:10]
    if top_signals:
        print(f"\n📊 RSR ランキング（ユニバース上位10銘柄）")
        print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<6} {'RSR':>6} {'SEPA':>5} {'Mom':>7}  シグナル")
        print("  " + "-" * 66)
        for s in top_signals:
            sig_str   = "✅ BUY" if s["signal"] == 1 else "  -  "
            strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
            print(
                f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<6} "
                f"{s['rsr']:>6.1f} {s['sepa_score']:>5} {s['rsr_momentum']:>+7.2f}  {sig_str}"
            )
    # 平均回帰 BUY シグナルを別枠で表示
    mr_buys = [s for s in result.signals if s["signal"] == 1 and s.get("strategy_type") == "mean_rev"]
    if mr_buys:
        print(f"\n📊 平均回帰 BUY シグナル（{len(mr_buys)}件）")
        print(f"  {'銘柄':<10} {'セクター':<10}  理由")
        print("  " + "-" * 60)
        for s in mr_buys:
            print(f"  {s['symbol']:<10} {s['sector']:<10}  {s['reason']}")


def confirm_live_orders(orders: list) -> bool:
    """発注前に確認プロンプトを表示する。"""
    buy_orders  = [o for o in orders if o["side"] == "BUY"]
    sell_orders = [o for o in orders if o["side"] == "SELL"]
    total_buy   = sum(o["estimated_amount"] for o in buy_orders)

    print("\n" + "=" * 60)
    print("  ⚠  実際の発注を行います。内容を確認してください。")
    print("=" * 60)
    print(f"  BUY  : {len(buy_orders)} 件  （推定合計: ¥{total_buy:,.0f}）")
    print(f"  SELL : {len(sell_orders)} 件")
    print("=" * 60)

    ans = input("  発注を実行しますか？ [y/N] > ").strip().lower()
    return ans == "y"


def save_signal_json(result, output_dir: str) -> Path:
    """シグナル JSON をタイムスタンプ付きファイルに保存する。"""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    file_path = dir_path / f"signal_{ts}.json"
    file_path.write_text(result.to_json(), encoding="utf-8")
    return file_path


def main() -> int:
    args = parse_args()
    print_banner(args.live)

    # ---- インポート ----
    import warnings
    warnings.filterwarnings("ignore")

    from backtest.topix100_backtest import TOPIX100_TICKERS, FUJIKO_PARAMS, PORT_PARAMS
    from kabusapi.signal_bridge     import SignalBridge

    # ---- ブリッジ初期化 ----
    bridge = SignalBridge(
        universe_tickers  = TOPIX100_TICKERS,
        fujiko_params     = FUJIKO_PARAMS,
        capital           = PORT_PARAMS["capital"],
        max_positions     = PORT_PARAMS["max_positions"],
        max_dd_limit      = PORT_PARAMS["max_dd_limit"],
        min_sectors       = PORT_PARAMS["min_sectors"],
        live              = args.live,
    )

    # ---- シグナル生成 ----
    logger.info("シグナル生成開始...")
    result, order_objects = bridge.run()

    # ---- 表示 ----
    print(f"\n  データ基準日 : {result.data_as_of}")
    print(f"  ユニバース   : {result.n_universe} 銘柄")
    print(
        f"  ポートフォリオ: 保有 {result.portfolio_summary['current_positions']} / "
        f"最大 {result.portfolio_summary['max_positions']} 銘柄  "
        f"空きスロット: {result.portfolio_summary['open_slots']}  "
        f"余力: ¥{result.portfolio_summary['available_cash']:,.0f}"
    )

    print_signals(result)

    # ---- JSON 保存 ----
    if not args.no_save:
        saved_path = save_signal_json(result, args.output_dir)
        print(f"\n💾 シグナル保存: {saved_path}")

    # ---- 発注 ----
    if not args.live:
        print("\n" + "=" * 60)
        print("  ドライランのため発注は行いません。")
        print("  実際に発注するには --live オプションを付けて実行してください。")
        print("=" * 60)
        return 0

    if not order_objects:
        print("\n発注なし。終了します。")
        return 0

    # 確認プロンプト
    order_dicts = result.orders
    if not args.yes:
        confirmed = confirm_live_orders(order_dicts)
        if not confirmed:
            print("発注をキャンセルしました。")
            return 0

    # 発注実行
    print("\n発注中...")
    send_results = bridge._send_orders(order_objects)

    # 発注結果表示
    print("\n=== 発注結果 ===")
    for r in send_results:
        status = "✅ 成功" if r.get("success") else "❌ 失敗"
        order_id = r.get("order_id", r.get("error", ""))
        print(f"  {r['side']} {r['symbol']} {r['qty']}株 → {status}  ({order_id})")

    # 結果を JSON に追記して再保存
    result_dict = json.loads(result.to_json())
    result_dict["send_results"] = send_results
    if not args.no_save:
        saved_path = Path(args.output_dir) / f"signal_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}_executed.json"
        saved_path.write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n💾 発注結果保存: {saved_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
