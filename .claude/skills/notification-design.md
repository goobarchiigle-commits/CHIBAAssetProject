# notification-design

## Purpose

システムトレード（F4 TP50等）の運用通知・レポート・Dashboardを、毎回ゼロベースで設計せず一貫した情報設計で実装する。「ログの貼り付け」ではなく「人間が判断できる運用レポート」として設計する責務を持つ。

2026-08-20 9344誤売却インシデント（position metadata汚染によるスプリアスなtrailing-stop SELL）を契機に、通知が「発生した事実の羅列」に留まり「なぜそうなったか」「theoretical値かactual値か」を判別できなかったことが被害拡大の一因と判明したため制定。

---

## Use When

- 新しい戦略（TP30/E5/Fujiko等）に日次通知・週次レポートを追加するとき
- 既存の通知フォーマットを変更するとき
- 新しいDashboard・監視画面を作るとき
- 「メールが読みにくい」「異常に気づけなかった」という報告を受けたとき

---

## Not For

- Study成果物のMarkdownレポート（IS/OOS/WF等の研究記録） → `/report-generator` を使う
- 朝ルーティン・発注実行手順そのもの → `/live-signal` を使う
- 本Skillは「情報設計・表示ロジック」の責務。発注ロジック・Exit判定ロジックの実装は対象外（それらは各戦略の`exit_engine.py`等が担う。本Skillはその結果を**どう見せるか**のみを扱う）

---

## 0. 実装前の必須手順（新規作成・改修いずれも省略禁止）

```
[ ] 1. 既存の通知生成コードを探す（grep "_build.*notification_body\|_send.*notification"）
[ ] 2. 既存テンプレート・Skillを探す（.claude/skills/配下）
[ ] 3. 既存のメール送信経路を探す（src/notifier.py — 新規SMTP実装しない・唯一の送信経路）
[ ] 4. 既存テストを探す（tests/test_run_live_signal_*.py の notification セクション）
[ ] 5. 現行仕様を確認する（本Skill §2-4のフォーマット定義と実装の差分有無）
[ ] 6. 既存レイアウトとの差分を明示する（本Skill §7 変更時プロトコル）
[ ] 7. 必要最小限の変更を行う
```

既存実装を確認せずゼロからテンプレートを作らない。`src/notifier.py`（Gmail SMTP・fire-and-forget・`wait_pending()`必須）は変更しない — 各戦略の`_build_*_notification_body()`が本文を組み立て、`notify_success/error/warning/dry_run`へ渡すのみ。

**実装済み参照例**: `src/run_live_signal_f4_tp50.py`（`_build_dry_run_notification_body` / `_build_live_notification_body` / `_render_sell_item` / `_render_buy_item` / `_render_replacement_item` / `_find_previous_live_run`）。新規戦略へ展開する際はこの実装をテンプレートとして複製・改変する（TP30等、戦略ごとに独立ファイルのため関数コピーで良い — 詳細はCLAUDE.md「複製しない」規約参照。ただし本Skillが定める情報設計原則そのものは複製ではなく再適用する）。

---

## 1. 基本原則

```
1. 「ログ」ではなく「運用レポート」として設計する。
2. 最初に人間が判断すべき情報を置く（結果サマリ→詳細→技術情報の順）。
3. 技術ログ・診断情報は後段（【SYSTEM】等）に分離する。
4. 日本語を基本言語とする。
5. 銘柄は必ず「コード + 銘柄名 + Score」をセットで表示する
   （コードだけの表示は禁止 — 目視確認できない）。
6. 売買理由を必ず明示する（「SELL」「EXIT」だけでは理由として不十分）。
7. theoretical value（シグナル理論値・as_of・estimated_price）と
   broker actual value（kabu API実約定）を絶対に混同しない。
   表示する場合は「理論」「約定」等の接頭辞で明確に分離する。
8. スマートフォンで確認できる縦長レイアウトを優先する（横に長い表は避ける）。
9. 表形式より、1銘柄1ブロックの可読性を優先する。
10. 既存の通知・レポート仕様を最初に調査し、既存フォーマットを勝手に破壊しない
    （§0参照）。
```

---

## 2. 通知の役割分離

| 通知 | 目的 | 主役 |
|---|---|---|
| Daily Dry Run | 前日確認＋今日の判断 | **売買判断** |
| Daily Live | 実際の発注記録 | **注文** |
| Weekly Report | 戦略評価 | **損益・リスク** |

3つの通知に同じ情報を重複して詰め込まない。Dry Runは「約定結果」ではなく「判断根拠」、Liveは「約定結果」ではなく「注文書」、Weeklyは日々のOrder ID等の実行細部を含まない。

---

## 3. Daily Dry Run

必ず以下の順序（省略・入れ替え禁止）:

```
1. 【前日の実績】— 直近のLIVE実行結果。kabu API actual fillをSource of Truthとする
   （stored theoretical値では絶対に代用しない）
2. 【本日の判断】— 本日のSELL/BUY/REPLACEMENT候補
3. 【PORTFOLIO】
4. 【データ状態】
5. 【SYSTEM】
```

**本日の判断・SELL**: 銘柄コード / 銘柄名 / Score / Exit reason / Entry price / Highest since entry / Stop price / 判定価格 / Quantity / 「→ LIVEならSELL」

**本日の判断・BUY**: 銘柄コード / 銘柄名 / Score / Entry reason / Quantity / 基準価格（theoretical） / 「→ LIVEならBUY」

**本日の判断・Replacement**: SELL銘柄+Score / BUY銘柄+Score / Score差

本日の判断はまだ発注されていないため、実現損益・約定価格は一切表示しない。

---

## 4. Daily Live

「約定結果」ではなく「注文書」。成行注文の場合、約定価格を推測・表示しない（kabu sendorderの応答はorder_id+result_codeのみの非同期ackであり、その場で実約定は確定しない）。

表示: 銘柄コード / 銘柄名 / Score / 理由 / Entry・Highest・Stop or Target（判断根拠） / 判定価格 / 数量 / 注文種別（例:成行SELL） / Order ID / 発注時刻 / 【SYSTEM】

Actual fillは翌営業日のDry Runの「前日の実績」でkabu APIから取得・表示する。

---

## 5. Weekly Report

```
【PERFORMANCE】   週間損益 / 週間Return / 開始・終了資産 / 週間MaxDD
【BENCHMARK】     戦略Return / TOPIX / 日経平均 / S&P500 / 対Benchmark相対値
【TRADING】       BUY件数 / SELL件数 / Score Replace件数 / Win Rate / Profit Factor / Realized P/L
【TOP CONTRIBUTORS】 上位（コード+銘柄名+Score+損益）
【TOP LOSERS】       下位（同上）
【CURRENT HOLDINGS】 保有銘柄ごと（コード+銘柄名+Score+Entry+Current+P/L%）
【EXIT BREAKDOWN】   T15 STOP / TP50 TARGET / REPLACEMENT / Other の件数内訳
【SCORE REPLACEMENT】 件数 / 平均Score差 / 最大Score差
【RISK】             Current DD / Weekly MaxDD / Positions / Cash Ratio
【SYSTEM HEALTH】    Live executions / Order failures / API errors /
                     Scheduler failures / Metadata warnings / Notification failures
```

日々のOrder ID等の実行細部は含めない（Dry Run/Liveの責務）。CAGR等の長期指標は本レポートの対象外（別途研究レポート`/report-generator`側で扱う）。

---

## 6. 表記規則

**SELL理由ラベル**（固定・これ以外を使う場合は追加してよいが「SELL」単独は禁止）:
```
T15 トレーリングSTOP
TP50 利確
Score入替（Replacement）
その他 — その他の場合は具体的なexit_reason文字列をそのまま出す
```

**価格**: 円表記（`¥1,224`）。損益は必ず損益率を併記（`+¥800 / +0.65%`）。theoretical/actualは行を分離し、接頭辞（理論／約定・基準／判定）で区別する。

**銘柄表示**:
```
9344 アクシスコンサルティング
Score 52.3
```
コードと銘柄名は必ず同じブロックの隣接行に置く。銘柄名が取得できない場合はコード単独表示に留める（コードの重複表示や"N/A"の銘柄名を並べない）。

---

## 7. 異常表示

正常な情報とWARNING/CRITICALを同じ並びで混在させない。異常はレポート冒頭または独立【警告】セクションで明示する。

```
WARNING:
  Scheduler未発火（手動遅延実行）
  Data stale（fundamentals鮮度異常）
  Notification遅延

CRITICAL:
  Metadata mismatch（entry_date/entry_priceの実約定確認不能）
  Actual fill unavailable
  Duplicate order risk
  Position reconciliation failure
```

---

## 8. 禁止事項

```
- raw logのメール貼り付け
- 英語ログ中心の通知
- 銘柄コードだけの表示
- Scoreなしの売買通知（取得不能時は "Score N/A" と明示。欄自体を消さない）
- Exit reasonなしのSELL
- theoretical priceをactual fillとして表示すること
- Live時点で未確定の約定価格を推測して表示すること
- 同じ情報を複数箇所に重複表示すること
- 既存レイアウトを確認せず全面改修すること（§0手順を踏まずに着手すること）
```

---

## 9. 変更時プロトコル

UI/通知フォーマットを変更する場合、変更前に以下を確認・記録する:

```
Before:  現行の出力例（実際のbody文字列 or スクリーンショット相当）
After:   変更後の出力例
変更理由: なぜ変えるか（インシデント番号・ユーザー指摘等の根拠）
影響範囲: 対象戦略（TP50/TP30/E5/Fujiko等）・対象通知種別（Dry Run/Live/Weekly）
テスト:   §10参照
```

---

## 10. テスト方針

Production通知の場合、**実注文を発生させずに**テンプレートテストを実施する。

```
[ ] 通常のExitパターン（各理由ラベルごとに最低1件）
[ ] 複数件（SELL/BUY/Replacementそれぞれ2件以上）
[ ] 0件（全ブロックが「なし」で崩れないこと）
[ ] 異常系: Scheduler未発火 / API error / metadata mismatch
[ ] DRY RUNとLIVEの見出し・本文が明確に区別されること
[ ] theoretical価格とactual fillが異なるケース（両方が本文中の別行に出ること）
[ ] 実際にsrc/notifier経由でSMTP送信されることの確認（本番運用相当の
    python -m src.run_live_signal_*（--liveなし）を1回実行し、実際に届くメールで
    最終確認する — ユニットテストのbody文字列検証だけで完了としない）
```

kabu APIへのGET照会（board/orders等、銘柄名・実約定取得のため）はテストに使ってよいが、`sendorder`（POST発注）は通知テストで絶対に呼ばない。
