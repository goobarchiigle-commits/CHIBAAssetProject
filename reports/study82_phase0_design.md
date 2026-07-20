# Study82 Phase0.1 + Phase0 — 決算発表日時 API疎通確認 + 精度監査（起案書）

**日付**: 2026-07-20（v1.1: Phase0.1分離・UNKNOWN出力追加）
**性格**: 起案書のみ。**監査であって研究ではない**。アルファ探索・パラメータ推定・BTは一切含まない。
**正典**: `roadmap_v15_governance_layer.md`§8A-1/§8A-2/Phase3詳細 / `alternative_architectures_5x_2026-07-03.md`§ARCH-B「発表日時精度監査を先行実施」。
**位置づけ**: Route B（Core+PEAD+TSMOM）の土台であるPEADが**そもそも研究可能か**を確定する第一関門。FAILならStudy82は即終了し、Route BはCore+TSMOMの2スリーブ構成へ縮小する（frontier再測定が必要）。**§8A-2 Research Freeze Rule対象**: 本Study82完了まで、Study83実装を含む新規alpha実装/BTは一切着手しない。

---

## 0. 二段階構成（v1.1変更点）

```
Phase0.1（Priority 1）: API疎通確認のみ。「エンドポイントが存在し接続できるか」
Phase0  （Priority 2）: Phase0.1通過後、実データで精度監査（6項目）
```

**Phase0.1を独立させた理由**: 現状確認（§4）の通り、決算発表日時・財務諸表エンドポイントの存在自体が未確認。存在しない場合、6項目監査（Phase0）は実施不能——先に「土俵に立てるか」だけを確認する方が情報コストが低い。

---

## 1. 目的（狭く固定）

```
Phase0.1 Determine: Can we connect to a J-Quants endpoint that carries
                     earnings announcement timestamps / financial statements?
         Output   : CONNECTABLE / NOT_CONNECTABLE / UNKNOWN

Phase0   Determine: Is PEAD research possible at all (given Phase0.1=CONNECTABLE)?
         Output   : PASS / FAIL / UNKNOWN
```

**UNKNOWN追加の理由（v1.1）**: 従来のPASS/FAIL二値では「ドキュメント記載はあるがプラン外で実接続テストできない」「サンプル数が少なすぎて判定不能」等の中間状態を表現できなかった。UNKNOWNは**保留**を意味し、Route B構成の即時縮小はトリガーしないが、**Phase0.1/Phase0を「PASS/FAIL確定」と扱わずCP3判定を待たせる**（§8A-3のCase1-3はいずれもPEAD PASS/FAILの確定を前提とするため、UNKNOWNのまま先へ進めない）。

「PEADのエッジがどれくらいあるか」は測らない。測るのは「発表日時データがイベントスタディに使える精度・粒度・PIT健全性を持つか」のみ。ここで測定した数値（サプライズ率・リターン等）があっても、それはPhase0の副産物として記録するに留め、**Alpha判定には使用しない**（Study82 Alpha Study=Phase Dの管轄）。

---

## 1A. Phase0.1 実施内容（Priority 1・最優先）

```
1. J-Quants公式APIドキュメントで財務諸表/決算発表関連エンドポイントの有無を確認（読み取りのみ・承認不要）
2. エンドポイントが存在する場合: 契約中のStandardプランでアクセス権があるか確認
3. 存在しアクセス可能な場合: 最小サンプル（1銘柄・直近1四半期分）で疎通テスト
   → 新規スクリプト作成に該当・ASK_FIRST
4. 判定:
   CONNECTABLE     = エンドポイント存在・アクセス可・疎通成功
   NOT_CONNECTABLE = エンドポイント不在、またはプラン外、または疎通失敗（認証エラー等）
   UNKNOWN         = ドキュメント記載が不明瞭・プラン条件が確認不能等、上記いずれとも確定できない
```

**NOT_CONNECTABLE時**: Phase0は実施不能のままStudy82終了。Route B構成をCore+TSMOMへ縮小する検討をv15へ反映（TDnet等の代替データソース契約は別途新規提案・ASK_FIRST）。
**UNKNOWN時**: 判明した不明点を記録し、確認可能になるまでPhase0.1を保留（Route B構成は変更しない・Freeze Ruleは継続）。

---

## 2. Phase0 必須監査項目（6項・固定・追加禁止・Phase0.1=CONNECTABLE後のみ実施）

| # | 項目 | 判定基準 |
|---|---|---|
| 1 | 発表日時粒度 | 分単位まで取得可能か。日付のみ（時刻欠落）ならFAIL相当の重大欠陥 |
| 2 | 場中/前場/後場/引け後区別 | 場中発表と引け後発表を判別できるフィールドが存在するか（執行タイミングの根拠に必須） |
| 3 | 訂正開示の扱い | 一度公表された決算の訂正・修正が別レコードとして記録され、初回発表と混同しない設計か |
| 4 | 配信遅延 | 発表時刻とデータベース記録時刻（取得可能なら）の乖離。遅延がPIT保証を壊す規模か |
| 5 | PIT保証 | 「発表翌営業日寄付エントリー」の前提が成立するか — 発表時刻から翌営業日寄付までに情報が確実に公開されている構造か（lookahead混入経路の有無） |
| 6 | 廃止銘柄整合性 | 上場廃止銘柄の決算データが欠落なく含まれるか（欠落=survivorship bias再混入） |

## 3. Phase0 PASS/FAIL/UNKNOWN判定基準（事前固定・v1.1でUNKNOWN追加）

```
PASS    ⇔ 項目1(分単位) ∧ 項目2(区別可能) ∧ 項目5(PIT保証成立) ∧ 項目6(廃止銘柄含む)
          が全て満たされる。項目3・4は付随所見として記録するが単独ではFAILとしない
          （訂正開示・配信遅延はイベントスタディ設計側で対処可能なため）。

FAIL    ⇔ 項目1・2・5・6のいずれか1つでも「不成立」と明確に確認できる。
          FAIL時は本Study82全体を即終了する（永久閉鎖ではなく現行データソースでの
          再訪禁止 — 将来別データソースが利用可能になった場合のみ再起案可）。

UNKNOWN ⇔ 項目1・2・5・6のいずれかについて、サンプル不足・ドキュメント不備・
          API仕様の解釈不能等により成立/不成立を確認できない。
          UNKNOWN時はPASS/FAILいずれも確定させず、不明点を明記した上でStudy82を保留する
          （§8A-3のCP3ケース分岐はPEAD PASS/FAILの確定を前提とするため、UNKNOWNのままでは
          Route B Viability Reviewへ進めない）。
```

## 4. データソース候補（現状確認済み事実）

現行`src/jquants/provider.py`で疎通確認済みのエンドポイントは以下3種のみ:
```
/v2/equities/bars/daily   （日次株価）
/v2/indices/bars/daily    （TOPIX等指数）
/v2/equities/master       （銘柄マスタ）
```
**決算発表日時・財務諸表エンドポイントは未疎通**（本監査で初めて確認する）。`alternative_architectures_5x`原文が要求するデータ源は「J-Quants（決算発表日時・財務諸表・会社予想）+ TDnet適時開示」——J-Quants Standardプラン（契約済み・`research_state.md` 2026-07-09ログ）に財務諸表エンドポイントが含まれるかは**未確認**。TDnetは別データ源（現行未契約・未実装）。

**本監査の実施範囲（重要な絞り込み）**: J-Quants側のエンドポイント確認・少数サンプル取得・項目1-6評価をまず行う。TDnetは、J-Quants側で項目1-2-5-6を満たせない場合の代替候補として検討するが、**新規データソース契約はASK_FIRST**（CLAUDE.md `ASK_FIRST`該当：新規スクリプト作成/契約級の変更）のため、本Phase0では「TDnet必要性の有無」を出力するに留め、契約自体は別途提案する。

## 5. 実施手順（最小限・監査に徹する）

```
1. J-Quants API仕様（財務諸表/決算発表関連エンドポイントの有無）を公式ドキュメントで確認
2. エンドポイントが存在すれば、小サンプル（直近1-2四半期・数十銘柄）を取得し
   項目1-6を実データで検証
3. エンドポイントが存在しない、または項目1-2-5-6のいずれかを満たさない場合、
   TDnet併用の要否を判定材料として記録した上でFAIL
4. 結果を reports/study82_phase0_audit.md へPASS/FAILと6項目の実測所見のみで記録
   （サプライズ率・リターン等のアルファ関連数値は算出しない — 目的外）
```

## 6. ASK_FIRST該当性

- API仕様確認（ドキュメント参照）: 承認不要（読み取りのみ）
- 小サンプルAPI疎通コード: **新規スクリプト作成に該当・ASK_FIRST**
- TDnet契約検討: 本Phase0では「要否の記録」のみ・契約行為自体は別途ASK_FIRST

## 7. 完了条件

`reports/study82_phase0_audit.md`にPASS/FAIL確定 + research_state.md先頭への転記。PASS時のみPhase D（Study82 Alpha Study・アルファ測定）が起案可能になる。FAIL時はRoute B構成をCore+TSMOMへ縮小する検討をv15へ反映する。

---

**次アクション**: 本起案書の承認 → 小サンプル疎通スクリプト作成（新規・ASK_FIRST）→ 監査実施 → PASS/FAIL確定。

*作成: CLD (Fable 5)・2026-07-20。コード未実行・データ未取得。*
