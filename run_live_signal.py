"""
run_live_signal.py
最適化済みパラメータ（V2推奨設定 Calmar 2.656達成）での朝のシグナル＆発注スクリプト

【設定根拠】
  - バックテスト: 2018-2024、初期資本200万円
  - V2シナリオ（RSR>70 + 均等ウェイト + 29銘柄）:
      CAGR=+16.26% / MaxDD=-6.12% / Calmar=2.656 / Sharpe=1.693
  - 旧設定 D_IDM_sec1: CAGR=+13.19% / MaxDD=-5.90% / Calmar=2.235
  - スクリプト: backtest/portfolio_v2.py

【パラメータ（V2）】
  - 宇宙   : G宇宙29銘柄（27銘柄 + 6857.T アドバンテスト + 6594.T ニデック）
  - min_rsr : 70（旧75から緩和）
  - max_pos : 3
  - min_sec : 1（セクター制約なし）
  - ウェイト: 均等（vol_target=0、IDM無効）
  - 決算保護: 無効（逆効果のため採用しない）

【使い方】
  # ドライラン（発注しない）
  python run_live_signal.py

  # 実発注（kabuステーション起動・ログイン後）
  python run_live_signal.py --live

  # 確認スキップ（自動化用）
  python run_live_signal.py --live --yes

【CLAUDE.md ルール3 確認】
  .env に KABU_API_PASSWORD / KABU_TRADE_PASSWORD が設定されていること。
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
logger = logging.getLogger("live_signal")

# ------------------------------------------------------------------ #
# V2推奨設定 Calmar 2.656 達成パラメータ（29銘柄 / RSR>70 / 均等ウェイト）
# ------------------------------------------------------------------ #
G29_UNIVERSE: dict[str, str] = {
    # --- 既存27銘柄（Sharpe>0.30 / MaxDD<30% スクリーニング済み）---
    "8035.T": "電機精密",   # 東京エレクトロン
    "6645.T": "電機精密",   # オムロン
    "6702.T": "電機",       # 富士通
    "6501.T": "電機",       # 日立製作所
    "6762.T": "電機精密",   # TDK
    "6920.T": "電機精密",   # レーザーテック
    "7203.T": "輸送機器",   # トヨタ自動車
    "7201.T": "輸送機器",   # 日産自動車
    "9432.T": "情報通信",   # NTT
    "8306.T": "銀行",       # 三菱UFJ
    "8411.T": "銀行",       # みずほFG
    "8309.T": "銀行",       # 三井住友トラスト
    "7182.T": "銀行",       # ゆうちょ銀行
    "8725.T": "保険",       # MS&AD
    "8058.T": "商社",       # 三菱商事
    "8053.T": "商社",       # 住友商事
    "8002.T": "商社",       # 丸紅
    "8001.T": "商社",       # 伊藤忠商事
    "4021.T": "化学",       # 日産化学
    "2914.T": "食品",       # JT
    "7011.T": "機械",       # 三菱重工業
    "3382.T": "小売",       # セブン&アイ
    "5401.T": "鉄鋼",       # 日本製鉄
    "5411.T": "鉄鋼",       # JFEスチール
    "9531.T": "ガス",       # 東京ガス
    "9101.T": "海運",       # 日本郵船
    "9104.T": "海運",       # 商船三井
    # --- V2追加2銘柄（Calmar +31%改善の主因）---
    "6857.T": "電機精密",   # アドバンテスト（半導体テスト装置）Sharpe=0.468
    "6594.T": "電機精密",   # ニデック（精密モーター）Sharpe=0.639, MaxDD=-6.5%
}

FUJIKO_PARAMS = dict(
    min_sepa         = 6,
    min_rsr          = 70.0,   # V2: 75.0 → 70.0 に緩和（Calmar改善確認済み）
    mom_period       = 21,
    turtle_entry     = 20,
    turtle_exit      = 10,
    use_turtle_entry = True,
)

CAPITAL      = 2_000_000
MAX_POS      = 3
MIN_SECTORS  = 1   # セクター制約なし（D_IDM_sec1 設定）
MAX_DD_LIMIT = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="フジコ法 最適化済み朝のシグナル生成スクリプト")
    p.add_argument("--live",   action="store_true", help="kabuステーション API に実際に発注する")
    p.add_argument("--yes","-y", action="store_true", help="発注確認プロンプトをスキップ")
    p.add_argument("--no-save", action="store_true", help="JSON シグナルファイルを保存しない")
    p.add_argument("--output-dir", default="data/signals", help="シグナルJSONの保存先")
    return p.parse_args()


def print_banner(live: bool) -> None:
    mode = "🔴 LIVE（実発注）" if live else "🟡 DRY RUN（発注なし）"
    print("=" * 64)
    print("  フジコ法 V2推奨設定シグナル確認（Calmar=2.656設定）")
    print(f"  実行日時 : {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}")
    print(f"  モード   : {mode}")
    print(f"  宇宙     : G宇宙 {len(G29_UNIVERSE)}銘柄 / max_pos={MAX_POS} / min_sectors={MIN_SECTORS} / min_rsr=70")
    print("=" * 64)


def print_signals(result) -> None:
    orders = result.orders
    if not orders:
        print("\n📭 本日の注文なし（全銘柄 HOLD / 条件不成立）")
    else:
        print(f"\n📋 発注予定: {len(orders)} 件")
        print(f"  {'銘柄':<10} {'売買':<6} {'数量':>6} {'参考価格':>10} {'参考金額':>12}  理由")
        print("  " + "-" * 74)
        for o in orders:
            side_str = "🟢 BUY " if o["side"] == "BUY" else "🔴 SELL"
            print(
                f"  {o['symbol']:<10} {side_str} {o['qty']:>6}株 "
                f"¥{o['estimated_price']:>9,.0f} "
                f"¥{o['estimated_amount']:>11,.0f}  "
                f"{o['reason'][:40]}"
            )

    if result.warnings:
        print("\n⚠ 警告:")
        for w in result.warnings:
            print(f"  - {w}")

    # BUYシグナル銘柄一覧
    buy_sigs  = [s for s in result.signals if s["signal"] == 1]
    sell_sigs = [s for s in result.signals if s["signal"] == -1]

    if buy_sigs:
        print(f"\n📊 BUYシグナル銘柄 ({len(buy_sigs)}件):")
        print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<4} {'RSR':>6} {'SEPA':>5} {'Mom':>7}  理由")
        print("  " + "-" * 68)
        for s in sorted(buy_sigs, key=lambda x: x["rsr"], reverse=True):
            strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
            print(
                f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<4} "
                f"{s['rsr']:>6.1f} {s['sepa_score']:>5} {s['rsr_momentum']:>+7.2f}  "
                f"{s['reason'][:30]}"
            )

    if sell_sigs:
        print(f"\n📊 SELLシグナル銘柄 ({len(sell_sigs)}件):")
        for s in sell_sigs:
            print(f"  {s['symbol']} ({s['sector']}) — {s['reason'][:50]}")

    # RSRトップ10
    print(f"\n📊 RSRランキング上位10銘柄:")
    print(f"  {'銘柄':<10} {'セクター':<10} {'戦略':<4} {'RSR':>6} {'SEPA':>5} {'Mom':>7}  シグナル")
    print("  " + "-" * 66)
    for s in sorted(result.signals, key=lambda x: x["rsr"], reverse=True)[:10]:
        sig_str   = "✅ BUY" if s["signal"] == 1 else ("🔴 SELL" if s["signal"] == -1 else "  -  ")
        strat_str = "MR" if s.get("strategy_type") == "mean_rev" else "FJ"
        print(
            f"  {s['symbol']:<10} {s['sector']:<10} {strat_str:<4} "
            f"{s['rsr']:>6.1f} {s['sepa_score']:>5} {s['rsr_momentum']:>+7.2f}  {sig_str}"
        )


def confirm_live_orders(orders: list) -> bool:
    buy_orders  = [o for o in orders if o["side"] == "BUY"]
    sell_orders = [o for o in orders if o["side"] == "SELL"]
    total_buy   = sum(o["estimated_amount"] for o in buy_orders)
    print("\n" + "=" * 64)
    print("  ⚠  実際の発注を行います。内容を確認してください。")
    print("=" * 64)
    print(f"  BUY  : {len(buy_orders)} 件  （推定合計: ¥{total_buy:,.0f}）")
    print(f"  SELL : {len(sell_orders)} 件")
    print("=" * 64)
    ans = input("  発注を実行しますか？ [y/N] > ").strip().lower()
    return ans == "y"


def save_signal_json(result, output_dir: str) -> Path:
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    file_path = dir_path / f"signal_{ts}.json"
    file_path.write_text(result.to_json(), encoding="utf-8")
    return file_path


def main() -> int:
    args = parse_args()
    print_banner(args.live)

    import warnings
    warnings.filterwarnings("ignore")

    from kabusapi.signal_bridge import SignalBridge

    bridge = SignalBridge(
        universe_tickers = G29_UNIVERSE,
        fujiko_params    = FUJIKO_PARAMS,
        capital          = CAPITAL,
        max_positions    = MAX_POS,
        max_dd_limit     = MAX_DD_LIMIT,
        min_sectors      = MIN_SECTORS,
        live             = args.live,
    )

    logger.info("シグナル生成開始...")
    result, order_objects = bridge.run()

    print(f"\n  データ基準日 : {result.data_as_of}")
    print(f"  ユニバース   : {result.n_universe} 銘柄")
    print(
        f"  ポートフォリオ: 保有 {result.portfolio_summary['current_positions']} / "
        f"最大 {result.portfolio_summary['max_positions']} 銘柄  "
        f"空きスロット: {result.portfolio_summary['open_slots']}  "
        f"余力: ¥{result.portfolio_summary['available_cash']:,.0f}"
    )

    print_signals(result)

    if not args.no_save:
        saved_path = save_signal_json(result, args.output_dir)
        print(f"\n💾 シグナル保存: {saved_path}")

    if not args.live:
        print("\n" + "=" * 64)
        print("  ドライランのため発注は行いません。")
        print("  実際に発注するには --live オプションを付けて実行してください。")
        print("=" * 64)
        return 0

    if not order_objects:
        print("\n発注なし。終了します。")
        return 0

    order_dicts = result.orders
    if not args.yes:
        confirmed = confirm_live_orders(order_dicts)
        if not confirmed:
            print("発注をキャンセルしました。")
            return 0

    # ── 取引時間チェック: 09:00未満なら待機 ──────────────────────────
    import time
    now = datetime.now(JST)
    market_open = now.replace(hour=9, minute=0, second=5, microsecond=0)
    if now < market_open:
        wait_sec = (market_open - now).total_seconds()
        print(f"\n⏰ 取引開始まで {wait_sec:.0f}秒待機中... (09:00:05 発注予定)")
        time.sleep(wait_sec)
        print("  → 待機完了。発注を開始します。")
    # ──────────────────────────────────────────────────────────────────

    print("\n発注中...")
    send_results = bridge._send_orders(order_objects)

    print("\n=== 発注結果 ===")
    for r in send_results:
        status   = "✅ 成功" if r.get("success") else "❌ 失敗"
        order_id = r.get("order_id", r.get("error", ""))
        print(f"  {r['side']} {r['symbol']} {r['qty']}株 → {status}  ({order_id})")

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
