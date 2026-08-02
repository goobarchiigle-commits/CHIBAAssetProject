# Capital Flow Generator — Mechanism Feasibility Audit（2026-08-02）

## Business Goal

現行Production（RSR/Core）はGate 0 CLOSED（Study95/100/101/82/83/103により2026-07時点で
既に「資本投入価値なし」判定確定）。Entry Freeze継続中（デフォルト・新証拠なしに解除しない）。

本Auditの目的はProduction再開ではない。**新しい独立Edge候補「Capital Flow Generator」が
研究として成立し得るか**を、実装・BT・新規データ契約ゼロで判定するGate。

対象Mechanism（事前固定・勝者を見ずに定義）：

```
External Demand Shock（外生的需要ショック）
  → Supply / Capacity Constraint（供給・能力制約）
  → Capacity Expansion / Capex（設備投資対応）
  → Bottleneck Economic Role（代替困難な供給者Role）
  → Candidate Universe
  → 既存Winner Selection Signal（RSR/Momentum等）
```

**方法論上の制約（本Audit中厳守）**：
- 個別銘柄の株価・リターンは一切参照しない（Rule 2: Winner Blind）
- G2の事例は「思い出せる有名テーマ」からではなく、政府・行政の公式記録（閣議決定・法律施行・
  補助金認定等、日付が確定し構造的に列挙可能なソース）から収集する
- 収集はWebSearchのみ使用。株価・企業業績への言及を含む検索クエリは使用していない

---

## G1 — Economic Mechanism Coherence

一文で説明可能か：

> 「政府・行政が特定の需要（半導体安全保障・脱炭素・防衛力等）に予算を割り当てると、
> それを満たせる供給能力を持つ企業（または政府が公式に認定した企業）に投資・受注が
> 集中する」

**判定: PASS**。因果の向き（予算・政策決定 → 供給側投資 → 特定企業への受注集中）は
一文で説明可能かつ経済的に自明（テーマ株のナラティブではなく、財政支出の追跡）。

---

## G2 — Historical Recurrence（構造的一次ソースからの機械的検出）

G2定義（今回厳密化）：

```
G2 PASS =
  Recurrence exists
  AND Trigger detection is mechanically reproducible
      (公式記録の日付ベース列挙で再現可能)
  AND Detection does not depend on winner/outcome information
```

WebSearchで収集した3事例（すべて公式一次情報・日付確定・株価未参照）：

| # | Trigger | 公式決定日 | ソース | 一次ソースの性質 |
|---|---|---|---|---|
| 1 | 特定半導体生産施設整備等計画 認定（JASM/TSMC熊本） | 2022-06-17 | [METI 認定計画](https://www.meti.go.jp/policy/mono_info_service/joho/laws/semiconductor/semiconductor_plan.html) | METIが認定企業を**公式リストとして公表**（令和3年度補正6,170億円+令和4年度補正4,500億円） |
| 2 | GX推進法 成立 / 施行 | 成立2023-05-12 / 施行2024-02-01（法律第32号） | [電気事業連合会PR](https://www.fepc.or.jp/about_us/pr/oshirase/__icsFiles/afieldfile/2023/05/12/press_20230512.pdf) | 官報番号確定・GX経済移行債20兆円の使途区分が別途公開 |
| 3 | 防衛力整備計画（防衛費 5年43兆円） | 閣議決定 2022-12-16 | [防衛省/内閣官房](https://www.cas.go.jp/jp/siryou/221216boueiryokuseibi.pdf) | 7つの重視能力を明示（Role粒度は企業名まで届かず、装備庁契約データが別途必要） |
| 3事例とも| — | — | — | 検索クエリは「閣議決定日」「成立日」「認定計画」等の制度用語のみ。銘柄名・株価は検索していない |

**判定: PASS**（再現性あり・メカニカルに検出可能・outcome非依存）。
重要な副次発見：METI認定計画・防衛装備庁契約公表・GX経済移行債プロジェクト区分は、
いずれも**行政が定期的に更新する公式リスト**であり、「思い出せる事例を並べる」のではなく
「リストを定期的に走査する」運用に転換できる。これがG2を将来的にPIT運用可能にする鍵。

---

## G3 — PIT Trigger Detectability

3事例とも閣議決定・法律成立・行政認定という**発生と同時に公的に確定日が記録される**イベント。
市場が知る前に一般公開される性質のものではなく、公開自体がTrigger（先回り情報ではない）。
これはむしろ良い性質：Look-aheadの余地が構造的に小さい（発表日=Trigger確定日）。

**判定: PASS**。ただし「発表日に市場が完全に織り込むか、織り込みに時間がかかるか」は
別問題（Winner Selection段階の話であり、本Auditのスコープ外）。

---

## G4 — Role Mapping Feasibility（PASS / CONDITIONAL PASS / FAIL）

| Trigger種別 | 判定 | Required data |
|---|---|---|
| 半導体（METI認定計画型） | **CONDITIONAL PASS** | source: METI認定計画公表ページ（企業名込み）／ historical coverage: 制度開始2022年〜／PIT availability: 認定日に即時公開／entity mapping: 政府が直接企業名を公表するため追加マッピング作業ほぼ不要／acquisition method: 該当ページのスクレイピングまたは手動記録（有料契約不要） |
| 防衛（防衛力整備計画型） | **CONDITIONAL PASS**（半導体型より弱い） | source: 防衛装備庁「調達契約情報の公表」／ historical coverage: 未確認（要調査）／PIT availability: 契約公表時点／entity mapping: 契約企業リストは公式だが、「7つの重視能力」区分から企業への対応付けは追加作業要／acquisition method: 装備庁サイトの定期公表データ（無料） |
| GX/脱炭素（GX推進法型） | **FAIL（現時点）** | 対象が広すぎる（再エネ・水素・蓄電池・省エネ建築・原子力再稼働等、単一のRoleに収束しない）。GX経済移行債の個別プロジェクト区分まで降りないとRole定義が成立しない。降りた場合の候補数・網羅性は未検証 |

**判定: 種別によって異なる（PASS一律ではない）**。**半導体型が最も有望**——政府自身が
Role→企業の対応を公式に確定・公表しているため、後知恵によるRole定義（§8禁則）を
構造的に回避できる唯一のケース。

---

## G5 — Candidate Universe Generation Feasibility（PASS / CONDITIONAL PASS / FAIL）

半導体型を軸に評価：

- METI認定計画に名前が載る企業（JASM/TSMC、Rapidus、Kioxia等）は主契約者のみで数社。
  ここから**裾野（装置・材料・建設）まで広げるには追加のマッピングが必要**（東京エレクトロン等の
  半導体製造装置メーカー、レジスト・特殊ガス等の材料メーカー、建設会社）。
  この裾野マッピングは政府公式リストには存在しない → **CONDITIONAL PASS**
  （required data: 業界団体・IR開示ベースの sector taxonomy。RGP Taxonomy v1.0に
  既存の関連分類がある可能性 → 要確認、新規取得ではなく既存資産の再利用）
- Candidate数の見込み: 主契約者数社＋装置・材料裾野を合わせて20〜80銘柄程度と推定
  （§9理想レンジ「Candidate 100〜300→Strong Candidate 20〜50」に近い）

**判定: CONDITIONAL PASS**。裾野企業マッピングの手当てが次段階の前提条件。

---

## Existing Asset Audit — archive/bulk/ 棚卸し（新規取得なし）

`archive/bulk/`（22.74GB・909ファイル・verify_database.py PASS確認済み）を調査。

```
archive/bulk/fins/summary/  ← 2016-07〜2026-07（10年・日次live更新分含む）
```

スキーマ確認結果（実データより）：
`DiscDate, Code, DocType, Sales, OP, NP, EPS, ... , FSales, FOP, FNP, ...`
（決算実績・次期予想・配当予想中心の**財務サマリー**）

**結論: Role-mapping用途には使えない**。セグメント別売上・設備投資額・事業内容テキストの
いずれも含まれない。ただし**Stage B（Winner Selection）用の既存シグナル計算（earnings
acceleration等）には10年分PITデータとしてそのまま使える**——これは新規取得ゼロで
Stage Bの一部を賄える資産であり、費用対効果は高い。

Role-mapping（Trigger→Candidate生成）自体は、上記の通りMETI公式リスト等の**外部公開情報
（無料・契約不要）**で賄う方針となり、J-Quants解約（2026-08-01付最終監査済み）の影響を
受けない設計にできる。

---

## G6 — Stage B Independence（検証仮説の事前固定）

Validation段階に入る前に、Outcomeを見てから定義を変えないよう、比較対象を今ここで固定する。

```
H0（帰無）: Candidate Universe内での既存Signal（RSR/Momentum等）のヒット率
            = 全市場に同一Signalを適用した場合のヒット率

H1（採択条件）: Candidate Universe内でのヒット率 > 全市場でのヒット率
                （統計的有意 かつ 経済的閾値超過）

ヒット率の定義（固定）:
  Signal-hit銘柄（例: RSR≥75）のうち、forward N日リターンが
  基準（例: TOPIX超過）を上回った銘柄の比率
  N・基準はStage B設計時に確定し、Validation結果を見て事後変更しない
```

これはStudy95（RSR/Momentum単体KILL）の焼き直しにならないための唯一の判定軸。
「Candidate内で上がった銘柄があった」は採択条件にしない。

---

## Final Gate 判定

| Gate | 判定 |
|---|---|
| G1 Mechanism coherence | PASS |
| G2 Recurrence（機械的検出・outcome非依存） | PASS |
| G3 PIT detectability | PASS |
| G4 Role mapping | 半導体型=CONDITIONAL PASS／防衛型=CONDITIONAL PASS（弱）／GX型=FAIL |
| G5 Candidate generation | CONDITIONAL PASS（半導体型のみ、裾野マッピング条件付き） |
| Existing asset (archive/bulk) | Role-mapping不可・Stage B用には有用 |
| G6 Stage B独立性仮説 | 事前固定済み |

**総合判定: CONDITIONAL PASS（半導体・先端製造投資型Triggerに限定）**

STOPにはならない。ただしMechanismを「Capital Flow Generator全般」ではなく、
**「政府認定・補助金対象型の設備投資Trigger」**にまずスコープを絞ることを推奨。
GX型（範囲過大）・防衛型（企業マッピング弱い）は現時点で見送り、半導体型で
Candidate Generation Ruleの設計に進めるかを次のASK_FIRSTポイントとする。

## 未実施（本Auditのスコープ外・意図的に見送り）

- 実際のCandidate Universe生成（企業リスト確定）
- 装置・材料裾野企業のマッピング作業
- 個別銘柄の株価・リターン参照
- TDnet/新規API契約
- バックテスト・実装

## Time Budget

本Audit: 1セッション以内で完了（実績: 同セッション内）。
