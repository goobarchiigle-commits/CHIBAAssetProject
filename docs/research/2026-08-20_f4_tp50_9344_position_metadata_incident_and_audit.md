# F4 TP50/T15 — 9344誤売却インシデント: 根本原因・復旧・修正・監査

日付: 2026-08-20

## 1. 事象サマリ

銘柄9344(アクシスコンサルティング)が2026-08-20 09:38、F4 TP50/T15のtrailing-stop(-15%)により
売却された。調査の結果、これはexit_engine.py自体のロジック不良ではなく、**position metadataの
汚染による誤発火**と確定した。

## 2. 実測タイムライン(kabu API確定)

| 種別 | 注文ID | 受付/約定時刻 | 価格 | 数量 |
|---|---|---|---|---|
| 誤ったEntry記録(汚染値) | — | (存在しない・理論値) | entry_date=2026-08-18, price=1379円 | — |
| **実際のBUY** | 20260819A02N88827536 | 2026-08-19 14:28:50〜51 | **1224円** | 100株 |
| 誤ったSELL(本インシデント) | 20260820A02N90347371 | 2026-08-20 09:38:16〜17 | 1232円 | 100株 |
| Recovery BUY(本日実施) | 20260820A02N91066043 | 2026-08-20 14:58:01 | 1322円 | 100株 |

## 3. 根本原因

`src/run_live_signal_f4_tp50.py` の `apply_fill_metadata_updates()`(旧`_apply_fill_metadata_updates`)が、
BUY約定確定時に

```python
entry_dates[sym]  = as_of.strftime("%Y-%m-%d")        # シグナルの理論Entry日
entry_prices[sym] = float(r.get("estimated_price"))   # シグナルの理論Entry価格
```

という、**kabu APIの実約定情報を一切参照せず、注文送信前のシグナル理論値をそのままEntry実績として
永続化する**設計だった。

9344を実際に発注したrun(`f4_tp50_20260819_142609`、実行時刻14:26台)自身が、市場データ鮮度の
遅延により `signal_date=2026-08-17, entry_date(as_of)=2026-08-18` という、実カレンダー(8/19)より
1日古い値で動作していた。この結果、実約定が8/19であったにもかかわらず、記録されたentry_dateは
8/18となり、`compute_highest_since_entry()` が8/18(ポジション保有前)のHigh=1418円を「Entry後の
最高値」として誤って混入させ、誤ったstop_level(1205.30円)を算出、翌日それが実勢価格に触れて
誤発火した。

## 4. 復旧内容(本日実施)

1. kabu API `GET /positions` で9344現在保有0株を確認(誤売却の事実確認)
2. Recovery BUY 100株を市場成行で発注・約定(order_id=20260820A02N91066043、約定1322円)
3. `runtime/portfolio_state.json` を `src.portfolio.state_store.save_portfolio_state()`(唯一の書込経路)経由で修正:
   - `position_entry_dates["93440"] = "2026-08-19"`(実際の戦略Entry日)
   - `position_entry_prices["93440"] = 1224.0`(実際の戦略Entry価格)
   - `position_strategy_types["93440"] = "f4_tp50"`
   - Recovery BUY自体の価格(1322円)は戦略Entry価格として**記録していない**(元の戦略損益と混同させないため)
4. 監査証跡を `runtime/f4_tp50/recovery_events.jsonl`(新設サイドカー、既存スキーマ非改変)に追記
   - 誤発火による実現コスト: (1322-1232)×100 = **9,000円**(誤SELL/Recovery BUYの往復コストのみ。
     元の戦略Entry(1224円)自体の含み損益とは別勘定)

## 5. コード修正

`src/run_live_signal_f4_tp50.py`:
- 新規関数 `_fetch_actual_fill(client, order_id)`: kabu API `GET /orders` から該当order_idの
  約定明細(RecType=8, ExecutionDay/Price)を取得
- `apply_fill_metadata_updates()`: entry_date/entry_priceを **broker実約定** から記録するよう変更。
  `client` 引数を追加(呼び出し元は既存の認証済みKabuClientインスタンスを渡す)
- **FAIL CLOSED**: 約定確認ができない場合(API失敗・注文未発見)、entry_date/entry_priceは
  一切記録しない(次回run再試行に委ねる)。estimated_price/as_ofへのフォールバックは行わない
- Signal Freshness Guard: `as_of`(最新読込営業日)が実カレンダー日から3日超乖離した場合に
  WARNINGログを出力(発注はブロックしない — 週末を挟む正当な遅延と異常stalenessを無条件に
  区別できないため。ただし上記FAIL CLOSED修正により、staleness自体がmetadata汚染の原因には
  もうならない)

影響範囲: `run_live_signal_f4_tp50.py`はTP30/E5/Fujiko(`run_live_signal_f4_tp30.py`
`run_live_signal_simple_e5.py` `run_live_signal.py`)と関数を共有しておらず(それぞれ独立実装)、
本修正はF4 TP50のみに閉じている。

## 6. 過去Productionデータ監査結果(★重要: 9344以外にも同型汚染を確認★)

現在保有中の全11銘柄について、kabu API実約定 vs `portfolio_state.json`格納値を突合した。

| 銘柄 | 格納entry_date | 実約定日 | 乖離 | 格納entry_price | 実約定価格 | 価格乖離 |
|---|---|---|---|---|---|---|
| 48260 | 2026-08-07 | **2026-08-18** | **11日** | 521.0 | 534.2 | 13.2円 |
| 77810 | 2026-08-18 | **2026-08-19** | 1日 | 875.0 | 860.9 | 14.1円 |
| 17880 | 2026-08-18 | **2026-08-19** | 1日 | 4285.0 | 4295.0 | 10.0円 |
| 17160 | 2026-08-18 | **2026-08-19** | 1日 | 1420.0 | 1442.0 | 22.0円 |
| 378A0 | 2026-08-18 | **2026-08-19** | 1日 | 1077.0 | 1014.2 | 62.8円 |
| 78120 | 2026-08-18 | **2026-08-19** | 1日 | 2139.0 | 2138.0 | 1.0円 |
| 73710 | 2026-08-18 | **2026-08-19** | 1日 | 746.0 | 738.0 | 8.0円 |
| 73250 | 2026-08-18 | **2026-08-19** | 1日 | 696.0 | 695.8 | 0.2円 |
| 94500 | 2026-08-18 | **2026-08-19** | 1日 | 730.0 | 745.7 | 15.7円 |
| 34570 | 2026-08-18 | **2026-08-19** | 1日 | 926.0 | 916.5 | 9.5円 |
| 93440 | ~~2026-08-18~~→2026-08-19(本日修正済み) | 2026-08-19 | 修正済み | ~~1379.0~~→1224.0(本日修正済み) | 1224.0 | 修正済み |

**結論: 現在保有中の10銘柄(9344を除く)全てが同型のentry_date/entry_price汚染を抱えている。**
うち9銘柄は1日ズレ(9344と同一run由来)、48260は11日ズレ(より古い個別要因、未特定)。

**これらの10銘柄は今回のタスクスコープ外につき、metadataの書き換えは一切実施していない**
(ユーザー指示「過去データを勝手に書き換えない」に従う)。ただし、これらの銘柄も
`highest_since_entry`計算に汚染日以降のOHLCが使われている可能性があり、9344と同様の
スプリアスなtrailing-stop誤発火リスクを抱えたまま本番稼働中である。**至急、個別のmetadata
修正要否の判断と対応を別タスクとして行うことを強く推奨する。**

## 7. テスト

`tests/test_run_live_signal_f4_tp50.py`:
- `apply_fill_metadata_updates`関連テストをbroker実約定ベースの新設計に合わせて更新
- 新規: 実約定ベースでの記録、遅延fillでのas_of不使用、fail-closed(約定確認不能/client未指定)を検証するテスト追加

`tests/f4_tp50/test_exit_engine.py`:
- 新規: `test_highest_since_entry_never_uses_data_before_entry_date_9344_regression` —
  本インシデントの実測値(8/18 High=1418, 8/19 High=1309)を用いた回帰テスト

実行結果: `tests/test_run_live_signal_f4_tp50.py` + `tests/f4_tp50/` = **153 passed, 1 skipped**(既存skip、無関係)
