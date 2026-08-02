# Study82 Phase0.1 + Phase0 — 決算発表日時 API疎通確認 + 精度監査（起案書）

**日付**: 2026-07-20（v1.3: 監査項目をAudit1-6体系へ再編・目的をleakage起点で再定義。v1.2: FAIL帰結を§8A-1A決定木へ整合。v1.1: Phase0.1分離・UNKNOWN出力追加）
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

Phase0   Determine: Can PEAD be researched safely
                     (given Phase0.1=CONNECTABLE)?
         Output   : PASS / FAIL / UNKNOWN
```

**目的の再定義（v1.3・重要）**: Phase0の問いは「PEAD works?」（PEADは機能するか）では**ない**。
正確には**「PEAD can be researched without leakage?」**（PEADはリーク混入なしに研究可能か）。
「機能するか」はStudy82 Alpha Study（Phase D）の管轄——Phase0は**安全性の監査**であり、有効性の
予備測定ではない。この区別を崩す（=監査中にアルファらしき数値へ言及する）行為はPhase0自体の
無効化として扱う。

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

**NOT_CONNECTABLE時**: Phase0は実施不能のままStudy82終了。§3のFAIL時と同一の決定木（§8A-1A）を適用する（TDnet等の代替データソース契約は別途新規提案・ASK_FIRST）。
**UNKNOWN時**: 判明した不明点を記録し、確認可能になるまでPhase0.1を保留（Route B構成は変更しない・Freeze Ruleは継続）。

---

## 2. Phase0 必須監査項目（Audit1-6・v1.3再編・固定・追加禁止・Phase0.1=CONNECTABLE後のみ実施）

**再編方針（v1.3）**: 旧項目1-6を、判定意図がより明確な6項目（Audit1-6）へ再構成。「Missing ratio」を
新規追加（定量的完全性チェックの欠落を補完）・旧「場中/引後区別」「配信遅延」は共に**leakage
（リーク混入）の下位問題**として`Audit5`へ統合し、監査の核心が「leakageの有無」であることを明示。

| # | 項目 | 判定基準 |
|---|---|---|
| **Audit1** | DiscTime precision | `DiscTime`が分単位まで意味のある精度を持つか。定型値（例: 一律"12:00:00"のような機械的デフォルト）ならFAIL相当の重大欠陥 |
| **Audit2** | Correction handling | 訂正・修正開示が`TypeOfDocument`/`DisclosureNumber`等により初回発表と区別可能な別レコードとして記録されるか |
| **Audit3** | Missing ratio | サンプル対象銘柄・四半期のうち`DiscDate`/`DiscTime`が欠損しているレコードの比率。高比率はPIT再構成の信頼性を損なう |
| **Audit4** | Delisted coverage | 上場廃止銘柄の決算データが欠落なく含まれるか（欠落=survivorship bias再混入） |
| **Audit5** | Leakage possibility | 場中/引け後の区別可否・配信遅延（発表時刻とAPI反映時刻の乖離）を含め、「発表翌営業日寄付エントリー」の前提を破ってlookaheadが混入し得る経路の有無を総合評価 |
| **Audit6** | Point-in-Time reconstruction | Audit1-5の結果を踏まえ、「時点Tにおいて何が公知だったか」を過去に遡って正しく再構成できるか（PIT保証の最終確認） |

## 3. Phase0 PASS/FAIL/UNKNOWN判定基準（事前固定・v1.3でAudit番号へ更新）

```
PASS    ⇔ Audit1(分単位精度) ∧ Audit5(leakage経路なし) ∧ Audit6(PIT再構成可能) ∧
          Audit4(廃止銘柄含む) が全て満たされる。Audit2・Audit3は付随所見として記録するが
          単独ではFAILとしない（訂正開示・欠損率はイベントスタディ設計側でフィルタ/除外
          対処が可能なため）。

FAIL    ⇔ Audit1・Audit4・Audit5・Audit6のいずれか1つでも「不成立」と明確に確認できる。
          FAIL時は`roadmap_v15_governance_layer.md`§8A-1A Study82 FAIL決定木を適用する
          （現行データソースでの本Study82再訪は禁止・将来別データソースが利用可能になった
          場合のみ再起案可）: PEAD assumptions downgraded（配分上限0%への完全ダウングレード）
          → Study103 assumptions rerun（§8A-4A Study52再発防止規則=major downgrade毎に1回のみ）
          → Route B frontier re-estimation → B confirmed/degraded/A promotedのいずれか。
          **「Route Bの死亡」ではなく「PEADの死亡」** — Core+TSMOMでTier2/Tier1が残る
          可能性は排除しない。

UNKNOWN ⇔ Audit1・Audit4・Audit5・Audit6のいずれかについて、サンプル不足・ドキュメント不備・
          API仕様の解釈不能等により成立/不成立を確認できない。
          UNKNOWN時はPASS/FAILいずれも確定させず、不明点を明記した上でStudy82を保留する
          （§8A-3のCP3ケース分岐はPEAD PASS/FAILの確定を前提とするため、UNKNOWNのままでは
          次段階へ進めない）。
```

## 4. データソース候補（現状確認済み事実）

現行`src/jquants/provider.py`で疎通確認済みのエンドポイントは以下3種のみ:
```
/v2/equities/bars/daily   （日次株価）
/v2/indices/bars/daily    （TOPIX等指数）
/v2/equities/master       （銘柄マスタ）
```
**決算発表日時・財務諸表エンドポイントは未疎通**（本監査で初めて確認する）。`alternative_architectures_5x`原文が要求するデータ源は「J-Quants（決算発表日時・財務諸表・会社予想）+ TDnet適時開示」——J-Quants Standardプラン（契約済み・`research_state.md` 2026-07-09ログ）に財務諸表エンドポイントが含まれるかは**未確認**。TDnetは別データ源（現行未契約・未実装）。

**本監査の実施範囲（重要な絞り込み）**: J-Quants側のエンドポイント確認・少数サンプル取得・Audit1-6評価をまず行う。TDnetは、J-Quants側でAudit1・4・5・6を満たせない場合の代替候補として検討するが、**新規データソース契約はASK_FIRST**（CLAUDE.md `ASK_FIRST`該当：新規スクリプト作成/契約級の変更）のため、本Phase0では「TDnet必要性の有無」を出力するに留め、契約自体は別途提案する。

## 5. 実施手順（最小限・監査に徹する）

```
1. J-Quants API仕様（財務諸表/決算発表関連エンドポイントの有無）を公式ドキュメントで確認
2. エンドポイントが存在すれば、小サンプル（直近1-2四半期・数十銘柄）を取得し
   Audit1-6を実データで検証
3. エンドポイントが存在しない、またはAudit1・4・5・6のいずれかを満たさない場合、
   TDnet併用の要否を判定材料として記録した上でFAIL
4. 結果を reports/study82_phase0_audit.md へPASS/FAIL/UNKNOWNとAudit1-6の実測所見のみで記録
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
