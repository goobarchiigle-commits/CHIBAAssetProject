# roadmap-governance

## Purpose

研究プログラム全体の状態管理を担う: Study採択/棄却の記録・Research State/Roadmap/Decision Record/Open Questions/Closed Researchの更新・次研究の優先順位整理。
個別Studyの数値検証（ADOPT/REJECTゲート判定）は`/backtest-research`、Study単体の成果物Markdown作成は`/report-generator`が担当する。本Skillはそれらの結果を「プログラム全体の記録」へ反映する責務に限定し、バックテストは実行しない。

---

## Use When

- Study採択（ADOPT）または棄却（REJECT）が確定し、プログラム全体の記録へ反映するとき
- src/research_state.md を更新するとき
- Roadmap（reports/complete_execution_roadmap_*.md・reports/final_research_roadmap_*.md等）を更新するとき
- Open Questions（未解決事項台帳）を更新するとき
- Closed Research（恒久閉鎖領域）を更新するとき
- Decision Record（採否記録）を更新するとき
- 次研究の優先順位を整理するとき

---

## Not For

- バックテストの実行・ADOPT/REJECTゲートの適用・過学習検知 → `/backtest-research` を使う
- Study単体のレポートMarkdown作成（Executive Summary/Method/Results等） → `/report-generator` を使う
- ライブ運用・発注 → `/live-signal` を使う

---

## 1. 適用前チェック

```
[ ] 反映対象のStudyが/backtest-researchのゲート判定を完了している
[ ] 反映対象のStudyレポートが/report-generatorで作成済みである（Study単体の数値・結論はそちらが正）
[ ] 本Skillでは数値の再判定・再計算を行わない（既に確定した結論を記録するのみ）
```

---

## 2. Study番号管理の原則

```
Study番号は管理対象とするが、自動採番は行わない。
  - 新規Study起案時、番号は既存正典（roadmap）に事前予約されているか確認する
  - 予約がない場合、番号案をユーザーに提示し決裁を得てから確定する（本Skillが独断で採番しない）
  - 欠番・重複番号を検出した場合は放置せずユーザーへ報告する
  - 派生分析等で正式番号を持たないもの（例: 過去のStudy74B事例）は「暫定命名」であることを明記し、正式採番はユーザー判断に委ねる
```

---

## 3. Not For 境界の運用ルール

```
本Skill実行中にバックテスト実行や数値検証が必要になった場合は、その場で計算せず /backtest-research へ切り替える。
本Skill実行中にStudy単体の詳細レポート作成が必要になった場合は /report-generator へ切り替える。
「ついでに」他Skillの責務を代行しない（責務混在は将来の監査困難性を生む — 過去のStudy52キャッシュ汚染事件の教訓）。
```

---

## 4. Research State Update

```
対象ファイル: src/research_state.md

更新規則:
  - 先頭セクション = 最新状態（NO_MEMORY_TRUST=true。会話履歴を信用しない前提を維持するため）
  - 新規セクションは既存セクションの「上」に追加する。既存セクションは削除せずアーカイブとして残す
  - ファイル冒頭のコメント行（最終更新日・見出し要約）を必ず更新する
  - 重要度に応じて見出しに★を付与する（既存慣例: ★=通常更新 / ★★★=Study完了級 / ★★★★★=Closure・恒久ルール確定級）
  - 各セクションに最低限含める項目: 目的 / 成果物パス / 結論 / 次アクション
  - 過去の公式値を訂正する場合は「削除」せず「旧値として保持し新値を併記」する（Study52キャッシュ汚染事件の教訓 — 過去の判定記録は当時の記録として凍結保持し、現在の判定材料としては使わない）
```

---

## 5. Roadmap Update

```
対象ファイル: reports/complete_execution_roadmap_*.md（実行手順書）/ reports/final_research_roadmap_*.md（正典本体）

更新規則:
  - 正典との矛盾がある変更は行わない。矛盾する場合はユーザー確認の上で正典改定を先に行う
  - Study完了時、該当Study節に実行結果サマリを追記する（新規節を乱立させず既存節へのappendを優先）
  - 「現在地」的なサマリテーブル（例: 現行公式値一覧）は更新のたびに最新値へ置き換える。旧値は打消し線+参考値注記で保持する
  - 依存関係・実行順序（StudyN→StudyN+1の順序等）の変更はユーザー決裁必須
  - 恒久ルール（例: Universe統制ポリシーのような横断的決裁事項）は該当する全ての将来Study節に影響範囲を明記する
```

---

## 6. Decision Record Update

```
対象ファイル: reports/core_decision_record.md 相当（恒久閉鎖・再開条件付き決定台帳）

記録項目（1件ごとに必須）:
  - 決定内容（ADOPT / REJECT / CLOSED 等）
  - 決定日
  - 根拠Study番号
  - 再開条件（再開不可の場合は「再開条件なし」と明記）
  - 禁止理由・失効トリガー

禁止事項:
  - 恒久閉鎖済み項目の「表現を変えた再訪」を新規Studyとして起案しないこと。本更新作業自体でもこの原則を遵守する（新規Study起案がDecision Record既存項目と重複していないか照合してから記録する）
```

---

## 7. Open Questions Management

```
対象ファイル: reports/core_open_questions.md 相当

台帳フォーマット: OQ番号 / 問い / 帰属先Study / EVI（Low/Medium/High） / Architecture選択への影響有無

更新規則:
  - Study完了によりOQが解消されたら「解消済み」に更新し、解消根拠（Study番号・数値）を記録する
  - 新規OQが見つかった場合、帰属先Studyが正典内に既に存在するか確認する。存在しない場合はユーザーへ新規Study起案の要否を確認する（勝手に新Studyへ割り当てない）
  - EVIの再評価はStudy結果確定時のみ行う（結果が出ていない段階での恣意的な優先度変更をしない）
```

---

## 8. Closed Research Management

```
対象ファイル: reports/core_closure.md 相当（恒久閉鎖領域リスト）

更新規則:
  - 閉鎖領域への追加は「反証済み（REJECT確定・再検証しても結論が変わる見込みがない）」場合のみ行う
  - 各項目に必須記載: 内容 / 反証根拠Study / 再開条件 / 失効トリガー
  - 閉鎖領域の削除・再開は、失効トリガーが実際に発生した場合のみ実施し、実施時はユーザーへ報告する
  - 閉鎖領域数が増えた場合、既存リストへの追記とし、番号を振り直さない（過去の参照との整合性維持）
```

---

## 9. Research Priority Rules

```
優先順位決定の原則:
  - 順序は「情報価値の高い順 ≠ 実装したい順」。依存構造・目標変更リスクを優先する
  - 目標判定（CP1/CP2/CP4等のCheckpoint）に影響するStudyを最優先とする
  - 依存元（他の複数Studyの共通土台となるStudy）は並行可能な位置に前倒しする
  - 連続2四半期新規採用ゼロの場合、プログラムのKill/縮退候補として提起する（正典既定のプログラムKill条件）
  - 優先順位の変更自体はユーザー決裁事項。本Skillが独断で並べ替えない
```

---

## 10. Version History Rules

```
CLAUDE.md/正典ドキュメントのバージョン管理:
  - CLAUDE.md変更はASK_FIRST必須（本Skillの適用範囲外の変更許可元。本Skillからは変更しない）
  - 正典（final_research_roadmap等）はファイル名にversion日付を含める（例: _2026-07-04）。上書きせず新版ファイルとして残すか、改定履歴セクションを明確に追加する
  - roadmap本体の末尾クレジット行（作成者・作成日）は改定のたびに改定者・日付を追記する
  - 過去バージョンとの差分（何が変わったか）を1-2行で記録する
  - メモリ（自動記憶）への反映は、恒久ルール・横断的決裁事項のみ対象とする（一時的な進捗状況はメモリ対象外）
```
