# live-signal

## Purpose

run_live_signal.py実行・注文発注・ポジション管理・朝ルーティンにおける
実行安全性と報告の確実性を確保する。

---

## Use When

- 朝ルーティンを実行するとき
- run_live_signal.py（Dry Run / Live）を実行するとき
- 注文発注・ポジション管理を行うとき
- 実行ログ・DD状況を確認・報告するとき
- EMERGENCY_STOPを発動するとき

---

## Not For

- バックテスト・Study設計・IS/OOS評価 → `/backtest-research` を使う
- 戦略パラメータの変更・評価 → `/backtest-research` を使う

---

## 1. Morning Routine

**6ステップを順番通りに実行する。スキップ禁止（FAIL_IF_SKIP=true）。**

```
Step 1: API疎通確認（Port 18080）
  → Section 2 の手順に従う

Step 2: Dry Run実行
  → python run_live_signal.py --dry-run
  → エラー・警告を全確認

Step 3: シグナル内容を日本語で要約・報告
  → BUY/SELL対象銘柄・理由・数量を日本語で提示
  → Section 9の報告フォーマットを使用

Step 4: Live実行（ユーザー確認不要・ただし Step1-3完了後）
  → python run_live_signal.py --live --yes

Step 5: 実行ログ報告
  → 銘柄・株数・金額・BUY/SELLを報告（Section 9のフォーマット）
  → スキップ・拒否件数も報告

Step 6: DD監視
  → rolling DDを確認（Section 10の基準）
  → -15%到達→即ユーザー警告（Section 10）
```

---

## 2. API Health Check

> **接続先ホスト・ポートの正値はCLAUDE.mdの`PROJECT.api`を参照すること。**
> 以下の例示値（localhost:18080）はデフォルト参考値。設定変更時はCLAUDE.mdのみ更新する。

```
確認対象:
  endpoint: {PROJECT.api}/kabusapi/primaryexchange/5401  ← CLAUDE.md参照
  期待応答: HTTP 200

確認手順:
  1. 上記エンドポイントへのレスポンスを確認
  2. HTTP 200 以外 → ABORT（api_unreachable）
  3. タイムアウト（5秒以上） → ABORT
  4. 接続成功後にポジション取得APIも確認

ABORT時の対応:
  → ユーザーへ即報告
  → 本日の実行を中止
  → kabuステーション起動状況を確認するよう伝える
```

---

## 3. Dry Run → Live Procedure

```
Dry Run:
  コマンド: python run_live_signal.py --dry-run
  確認項目:
    [ ] エラー・例外が発生していないか
    [ ] シグナル対象銘柄が期待通りか
    [ ] 注文数量・価格がスリッページ込みで妥当か（Section 7）
    [ ] ポジション上限（max_positions=3）を超えていないか
    [ ] 重複注文フラグが出ていないか

Live遷移条件（全て満たすこと）:
  [ ] Dry Run完了・エラーなし
  [ ] API接続確認済み（Section 2）
  [ ] ポジション照合済み（Section 8）
  [ ] シグナルをユーザーへ報告済み（Section 3）

Live実行:
  コマンド: python run_live_signal.py --live --yes
  完了後: 実行ログを Section 9のフォーマットで報告
```

---

## 4. Order Safety Checklist

**注文発注前に以下を全確認。1つでも違反→発注ブロック。**

```
重複注文チェック:
  [ ] 同銘柄・同方向（BUY/SELL）の当日重複注文がないか
  [ ] client_order_id が付与されているか（idempotency確認）
  [ ] same_day_duplicate=block が有効か

過剰注文チェック:
  [ ] max_positions=3 を超えていないか
  [ ] max_single_weight=0.25 を超えていないか（1銘柄25%上限）

コスト込み注文チェック:
  [ ] slippage=0.001（0.1%）が注文ロジックに含まれているか
  [ ] commission=0.00055（0.055%）が注文ロジックに含まれているか

レート制限:
  [ ] rate_limit=5/s を超えていないか
  [ ] 前回注文からcooldown=60s 経過しているか

リトライ制限:
  [ ] retry_max=1（最大1回のみリトライ）
```

---

## 5. Position Verification

**Live実行前にポジション照合を実施（position_reconcile_before_live=true）。**

```
照合手順:
  1. Broker API からポジションを取得
     GET http://localhost:18080/kabusapi/positions
  2. runtime/portfolio_state.json と突合
  3. 差異がある場合 → ABORT（position_sync_fail）

Broker優先原則:
  - Broker APIの結果が正
  - portfolio_state.jsonとの乖離は必ずBroker側に合わせる
  - 手動で portfolio_state.json を書き換えない

照合失敗時:
  → ユーザーへ即報告
  → 差異の内容（銘柄・数量・方向）を明示
  → ユーザー確認後に再実行
```

---

## 6. Execution Log Format

**Step5（実行ログ報告）で使用するフォーマット:**

```
=== 実行ログ YYYY-MM-DD ===

【発注済み】
  4021.T  BUY  100株  ¥2,340,000（参考価格: ¥23,400 × 100）
  6758.T  SELL  50株  ¥1,025,000（参考価格: ¥20,500 × 50）

【スキップ】
  1234.T  BUY  → max_positions上限（現在3/3）

【拒否】
  5678.T  BUY  → 重複注文（same_day_duplicate）

【ポートフォリオ概況】
  保有銘柄数: 3 / 3
  推定時価総額: ¥X,XXX,XXX
  当日DD: -X.XX%（累積DD: -X.XX%）

【異常】
  なし / [あれば記載]
```

---

## 7. Drawdown Response

**DD監視（dd_warn=-0.15 / max_dd_limit=0.15）:**

```
DD水準別対応:

  -10%未満: 通常運用継続
  
  -10% 〜 -15%:
    → ユーザーへ警告報告（売買は継続可）
    → 報告文: 「DD警戒: 累積DD=XX%。継続する場合はご確認ください」

  -15%到達:
    → 即座にユーザーへ警告報告（REPORT_ALWAYS=true）
    → BUY_STOP状態を報告（新規BUYを停止）
    → 既存ポジションのSELLは継続可
    → 報告文: 「⚠ DD上限到達: 累積DD=-XX%。BUY_STOPを発動します」
    → ユーザーの明示的な解除指示があるまでBUY禁止

注意: 自動撤退はしない。警告のみ。ユーザーが判断する。
```

---

## 8. Emergency Stop Procedure

**以下の条件のいずれかで発動（ABORT_ACTION=stop_execution_and_report_user）:**

```
発動条件:
  api_unreachable         → KabuステーションAPI応答なし
  position_sync_fail      → ポジション照合失敗
  duplicate_order_detected → 重複注文を検出
  portfolio_state_missing  → runtime/portfolio_state.jsonが存在しない

発動時の手順:
  1. 即座に実行を停止
  2. ユーザーへ以下を報告:
     - 発動した条件（上記4つのうちどれか）
     - 現在のポジション状況（取得できる場合）
     - 本日の発注状況（済/未）
     - 推奨する復旧手順
  3. ユーザーの明示的な指示があるまで再実行禁止

復旧手順例（position_sync_fail）:
  1. Broker APIで最新ポジションを取得
  2. portfolio_state.jsonを手動で確認
  3. 差異の原因を特定
  4. ユーザー確認後にportfolio_state.jsonを更新
  5. 再度照合 → 一致後に実行再開
```
