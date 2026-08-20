# F4 TP50/T15 — 9344誤売却インシデント: 根本原因・復旧・修正・監査

日付: 2026-08-20

## 0. 追補(同日・第2ラウンド): SSOT構造化・10銘柄復旧完了

初回対応(9344単体の復旧+`apply_fill_metadata_updates()`修正)後、以下を追加実施した。

**Phase1 — なぜkabu APIに実約定があるのにestimated_price/as_ofが使われたか(コード経路)**

1. `main()`が`buy_orders_intended`を構築する際(旧行1000付近、現`run_live_signal_f4_tp50.py`)、
   `OrderInstruction.estimated_price = f["entry_price_adjusted_open"]`(シグナル理論値=想定entry日の
   Open)をセット。この値は発注**前**の理論値であり、broker応答は一切参照していない。
2. `_submit_orders_process_isolated()`(`src/run_live_signal.py`、TP50/TP30/E5共有の発注実行関数、
   本修正では変更していない)は、kabu `sendorder`APIの**即時応答**(order_id + result_codeのみ。
   kabu sendorderは非同期ackであり、約定明細を含まない)を`results`へ格納するだけで、後続の
   `GET /orders`による約定確認を一切行っていなかった。
3. 旧`apply_fill_metadata_updates()`(修正前)は、この`results`(estimated_price/order_idのみ保持)を
   そのままEntry実績として`position_entry_dates[sym]=as_of`, `position_entry_prices[sym]=estimated_price`
   に書き込んでいた。**kabu APIから実約定(Details[] RecType=8)を取得する処理がそもそも存在しな
   かった**——「取得できるのに使わなかった」のではなく「取得する呼び出し自体がなかった」が正確な
   答え。今回の修正で新設した`_fetch_actual_fill_details()`が、この欠落していた呼び出しを埋めた。
4. 他の書き込み経路: コードベース全体を`position_entry_dates\[`等で検索した結果、
   `run_live_signal_f4_tp50.py`内の書き込みは`apply_fill_metadata_updates()`の2箇所のみ(今回修正
   済み・全てbroker実約定ソース)。ただし`src/scripts/sync_positions.py`(手動・対話式の汎用
   ポジション同期ツール、E5/TP30等とも共有)が別途`position_entry_dates`/`position_entry_prices`を
   書き込む経路として存在する。価格は broker の `avg_price`(実約定ベース、理論値ではない)を使用
   するため価格汚染のリスクは低いが、entry_dateは「取得当日(today)」がデフォルトであり
   (`--entry-date`未指定時)、未追跡の新規銘柄に対しては正確な建玉日を保証しない。これは
   **手動実行が前提の汎用reconciliationツール**であり、TP50専用コードではないため今回の修正
   スコープ外(他戦略への影響を避けるため)だが、残存リスクとして明記する。

**Phase2 — SSOT設計**: `portfolio_state.json`の`position_entry_dates`/`position_entry_prices`を
「broker実約定のみを保持するフィールド」として運用を継続(スキーマ変更なし・他戦略との共有
フィールドを崩さない)。理論値(as_of/estimated_price)との対比は、新設サイドカー
`runtime/f4_tp50/entry_fill_audit.jsonl`(追記専用、portfolio_state.jsonとは別ファイル)に
`theoretical`/`actual`を明示的に分離して記録するよう変更。

**Phase3/4 — fail-closed化・partial/multi-fill対応**: `_fetch_actual_fill_details()`が
`Details[]`中の全RecType=8明細を集計し、数量加重平均価格・最早約定日・合計約定数量を算出。
発注数量に対し約定数量が不足する場合(partial fill)はmetadata記録を保留。約定日が未来日等
明らかに異常な場合(`_validate_fill_sanity()`)も記録を拒否。同一order_idの再処理は
既存値保持により自然にidempotent。

**Phase7/8/9 — 残り10銘柄の監査・復旧・誤Exit有無**:

| 銘柄 | 旧entry_date | 実約定日 | 旧entry_price | 実約定価格 | broker order_id |
|---|---|---|---|---|---|
| 48260 | 2026-08-07 | 2026-08-18 | 521.0 | 534.2 | 20260818A02N86495800 |
| 77810 | 2026-08-18 | 2026-08-19 | 875.0 | 860.9 | 20260819A02N88827420 |
| 17880 | 2026-08-18 | 2026-08-19 | 4285.0 | 4295.0 | 20260819A02N88827440 |
| 17160 | 2026-08-18 | 2026-08-19 | 1420.0 | 1442.0 | 20260819A02N88827462 |
| 378A0 | 2026-08-18 | 2026-08-19 | 1077.0 | 1014.2 | 20260819A02N88827504 |
| 78120 | 2026-08-18 | 2026-08-19 | 2139.0 | 2138.0 | 20260819A02N88827510 |
| 73710 | 2026-08-18 | 2026-08-19 | 746.0 | 738.0 | 20260819A02N88827522 |
| 73250 | 2026-08-18 | 2026-08-19 | 696.0 | 695.8 | 20260819A02N88827545 |
| 94500 | 2026-08-18 | 2026-08-19 | 730.0 | 745.7 | 20260819A02N88827552 |
| 34570 | 2026-08-18 | 2026-08-19 | 926.0 | 916.5 | 20260819A02N88827561 |

kabu API `GET /orders`(2026-07-01以降全件)を突合した結果、**この10銘柄に対するSELL注文は
1件も存在しない**(全銘柄が単一BUY fill・partial fillなし・現在も100株ずつ保有継続)。
すなわち「metadata汚染により既に誤ってExitされていた銘柄」は**ゼロ**——9344のみが唯一の
実際の誤Exit事例だった。

全10銘柄について、`save_portfolio_state()`(唯一の書込経路)経由でentry_date/entry_priceを
broker実約定値へ修正済み(before/afterは本ファイル本文および`runtime/f4_tp50/
entry_fill_audit.jsonl`に記録)。過去の取引履歴・損益記録は一切書き換えていない
(修正対象はfuture live decisionに使われるposition metadataのみ)。

**Phase12 — 復旧後dry-run結果**: `exits_intended=0`、全11銘柄(9344含む)についてスプリアス
exitなし。通知メール送信成功。

## 1. 事象サマリ(9344単体・初回対応時点)

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
