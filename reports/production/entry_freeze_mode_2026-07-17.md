# Entry Freeze Mode（資産保全）実装レポート — 2026-07-17

## 背景

Study100（Universe監査・U-1静的RSR42名簿=FATAL）/ Study101（PITユニバース上の旧フジコ法=全4構成RED・TOPIX全面劣後）により、現行Live戦略の期待アルファは正当化根拠を失った。研究継続中の資産保全のため、新規BUY発注のみを全面停止する（SELL/exit・signal generation・diagnosticsは無停止）Entry Freeze Modeを実装した。

---

## 1. Freeze実装箇所

### 1-1. 設定層（新規）

| ファイル | 内容 |
|---|---|
| `src/configs/strategy.yaml` | `entry_freeze: {enabled: true, reason: "Research Freeze"}` セクション追加。**本commitの時点でEntry Freeze Modeは有効化済み**（ユーザー確認済み・「今すぐ有効化」を選択） |
| `src/config_loader.py` | `EntryFreezeConfig` dataclass・`resolve_entry_freeze()`（env var優先ロジック）・`StrategyConfig.entry_freeze`フィールド追加 |

**切替方法**:
- 恒常的な有効化: `strategy.yaml` の `entry_freeze.enabled: true` に変更してcommit
- 緊急上書き（git変更不要）: 環境変数 `ENTRY_FREEZE_ENABLED=1`（有効化）/ `=0`（無効化強制）。設定済みの場合、yaml値より優先される

### 1-2. 発注ゲート層（既存Circuit Breaker機構を再利用）

`src/kabusapi/signal_bridge.py::SignalBridge._build_orders()` に新規パラメータ非依存の独立フラグ `self.entry_freeze_enabled` を追加し、既存の `cb_active`（Circuit Breaker発動中BUY全停止）と **OR結合**（`block_new_buy = cb_active or entry_frozen`）した。

設計判断: 新規のBUY遮断ロジックを書き下ろすのではなく、既に本番稼働実績のある「CB発動中はSELLのみ・BUY全停止・4/5-tuple契約を維持して早期return」という実証済み経路をそのまま再利用した。これにより：
- SELL処理（保有中 かつ signal=-1）は無条件でfreezeより前に生成される → emergency exit/ATR stop/turtle exit/risk reductionは無停止
- signal生成自体（`_generate_all_signals`）はfreezeと無関係に実行される → universe generation/alpha calculation/diagnostics/promotion logsは無停止
- 5-tuple戻り値契約（`_validate_build_orders_contract`）を破壊しない

ログ出力（仕様通り）:
```
ENTRY_FROZEN: symbol=xxxx reason=Research Freeze
```
per-symbol・`logger.warning`。加えて `_build_orders` の戻り値 `warnings` にも `"ENTRY FREEZE MODE 発動中: ... reason=Research Freeze"` を格納し、DRY previewや `logs/live/*_orders.json` にも伝播する。

### 1-3. 発注元（SignalBridge呼び出し側）

| ファイル | 変更 |
|---|---|
| `src/run_live_signal.py` | `SignalBridge(...)` 生成時に `entry_freeze_enabled=cfg.entry_freeze.enabled, entry_freeze_reason=cfg.entry_freeze.reason` を追加 |
| `src/run_morning_signal.py` | 同上（後述・残存経路として発見・同一パッチ適用） |

### 1-4. 独立発見: `src/execution/live_pipeline.py`（残存経路・追加パッチ）

`generate_orders()` の rebalance計算にも同一のfreezeゲートを追加（`side=="BUY"`のrebalanceのみ生成をスキップ・SELLは無停止）。詳細は「2. 残存BUY経路」参照。

---

## 2. 全BUY発注経路の探索結果

| 経路 | 状態 | 対応 |
|---|---|---|
| `run_live_signal.py` → `SignalBridge.run()` → `_build_orders()` → `_submit_orders_process_isolated()` → `BrokerProcessSupervisor` → `broker_worker.py`（子プロセス）→ `KabuClient.send_order()` | **正典・唯一の本番経路**（2026-07-15 SSOT統合により確定済み） | ✅ freeze適用済み |
| `run_morning_signal.py` → 独自`SignalBridge`インスタンス → `bridge._send_orders()` | **2026-07-15にWindows Task Scheduler登録は解除済み**（`\run_morning_signal`無効化）だが、ファイル自体は現存し手動実行可能 | ✅ freeze適用済み（今回追加） |
| `src/execution/live_pipeline.py::execute_orders()`（`requests.post(.../sendorder)`直呼び）← `pipeline.py`（引数なし実行時のデフォルト分岐） | 完全に独立した別アーキテクチャ（SignalBridge/portfolio_state/CB/RSR/PARAMS_LOCKEDを一切経由しない決定論的リバランサー）。`KABU_API_KEY`環境変数を要求するが `.env` には未設定（`KABU_API_PASSWORD`/`KABU_TRADE_PASSWORD`のみ存在）→ **現状は到達時に`KeyError`で停止し実質デッド**。ただし将来同変数が設定されれば`python pipeline.py`（無引数）だけで実発注し得る | ✅ freeze適用済み（今回追加・`generate_orders()`内） |
| `src/kabusapi/client.py::KabuClient.send_order()`（低レベルAPI呼び出し本体） | 全経路の最終収束点。直接の呼び出し元は上記3経路のみ | 上位3経路のfreezeで担保 |
| `_send_orders_with_retry()`（`run_live_signal.py`内定義） | **呼び出し元ゼロ（デッドコード）** — grep全探索で定義以外の参照なし | 変更不要（到達不能） |
| `src/deployment/connectors/kabus_api_adapter.py::KabusApiAdapter.submit_order()` | `BrokerInterface` Protocol実装だが**呼び出し元ゼロ**（自ファイル内のみ） | 変更不要（到達不能） |
| manual_order.py | **リポジトリ内に存在しない**（探索済み） | 該当なし |
| recovery scripts（`entry_metadata_recovery.py`/`bootstrap_recovery.py`/`corruption_recovery.py`/`failure_recovery.py`） | send_order/SignalBridge/KabuClientへの参照なし（grep確認済み） | 該当なし |
| scheduler jobs（`daily_report.py`/`sync.py`/`api_auth_diagnostics.py`/`run_weekly_market_intelligence.py`） | send_order/SignalBridge参照なし（grep確認済み） | 該当なし |

**残存BUY経路（対応不要と判定したもの）**: なし。上記デッドコード2件は呼び出し元が存在しないため「到達可能な経路」には該当しない（将来何かがこれらを呼び出すよう変更されない限り無害）。

---

## 3. 検証結果

### Test1: BUY signal発生日
`_build_orders(signals=[BUYシグナル], cb_active=False)` を `entry_freeze_enabled=True` で実行。

| 項目 | 結果 |
|---|---|
| signal generated | YES（入力シグナルは加工前のまま） |
| broker order (BUY) | **NO**（`orders`にBUYが一件も含まれない） |
| position change | **NO**（他に保有変化なし） |

`test1_buy_blocked_live_false` / `test1_buy_blocked_live_true` / `test1_buy_blocked_dry_live_parity` の3テストで確認。**DRY/LIVE完全一致**（`self.live`の値に関わらず`orders`の内容が同一）。

### Test2: SELL signal発生日
`_build_orders(signals=[SELLシグナル・保有中], cb_active=False)` を `entry_freeze_enabled=True` で実行 → SELL注文が通常通り1件生成されることを確認（`test2_sell_executes_normally_live_false/true`）。BUY+SELL混在日でもSELLのみ通過することを別途確認（`test_mixed_buy_and_sell_same_day`）。

### Test3: Scheduler execution
コード経路としては run_live_signal.py の単一プロセス実行がDRY/LIVE双方の唯一のスケジュール対象（`CHIBATrading_DryRun`/`CHIBATrading_Live`、2026-07-15確定）であり、`_build_orders`の呼び出しは同一。E2E（実際のスケジューラ起動+healthcheck）での検証は本タスクの範囲外（ブローカー接続を要するため）だが、コードパス保証としては以下で代替:
- CBとfreeze同時発動でも5-tuple契約が破綻しないこと（`test_cb_and_freeze_both_active`）
- freeze無効時に既存挙動へ回帰がないこと（`test_freeze_disabled_buy_not_blocked_by_freeze_logic`）
- 既存の`_build_orders`関連テスト全て（`test_build_orders_contract.py`・`test_live_stage_audit.py`）が回帰なく通過

### 実行結果サマリ

```
src/kabusapi/test_entry_freeze.py                    8 passed
src/test_config_loader_entry_freeze.py               7 passed
src/execution/test_live_pipeline_entry_freeze.py     3 passed
src/kabusapi/test_build_orders_contract.py           6 passed（既存・回帰修正込み）
src/kabusapi/test_live_stage_audit.py               （既存・回帰修正込み）
src/kabusapi/ 全体                                   58 passed
src/live/ + src/execution/ 全体                      123 passed
```

**副次的に発見・修正した既存テストの回帰**: `test_build_orders_contract.py`と`test_live_stage_audit.py`は`MagicMock(spec=SignalBridge)`で`_build_orders`をテストしており、今回`self.entry_freeze_enabled`を参照するコードを追加したことで`AttributeError`が発生した。両ファイルのモックへ`entry_freeze_enabled=False`のデフォルト値を追加して解消済み（テストファイルのみの変更・本番コード無関係）。

---

## 4. Rollback手順

**即時無効化（ファイル変更不要）**:
```
# Windows Task Scheduler / 実行環境で
set ENTRY_FREEZE_ENABLED=0
```
次回run_live_signal.py実行から新規BUYが再開される（yaml側の値に関わらず強制OFF）。

**恒常的な無効化（解除）**:
`src/configs/strategy.yaml` の `entry_freeze.enabled` を `false` に変更してcommit。
**ユーザーの明示指示が必要**（現在は`true`=有効化された状態でcommit済み）。

**完全撤去（コード自体を戻す場合）**:
本コミットを `git revert` すれば全ファイルが復元される。新規dataclass・新規パラメータはいずれも default値を持つ後方互換設計のため、部分的にrevertしても既存コードは壊れない。

---

## 5. 制約遵守の確認

- BT/composite_alpha_bt.py・戦略パラメータ（PARAMS_LOCKED）・RSR定義: **無変更**
- 新規バックテスト: **実施なし**
- signal_bridge.py発注ロジック変更: CLAUDE.md ASK_FIRST対象だが、ユーザーの本タスク明示指示により実施
- push: **未実施**（commitのみ、ASK_FIRST）
