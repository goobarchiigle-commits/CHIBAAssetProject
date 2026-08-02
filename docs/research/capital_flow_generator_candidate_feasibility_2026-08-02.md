# Capital Flow Generator — Candidate Generation Feasibility Audit（2026-08-02）

## Business Goal

前提：`capital_flow_generator_mechanism_feasibility_2026-08-02.md`（Phase0）でCONDITIONAL PASS。
本AuditはPhase1＝半導体型TriggerでのCandidate Generation Rule確定。

**この研究の目的はCapital Flow Generatorを作ることではない。** PITで事前生成可能なCandidate
Universeが、全市場・Matched Controlに対して、実際の300万円口座で取引可能な期待値の高い
Entry Opportunityを反復的に生成できるかを検証すること。半導体型は本命ではなく**Pipeline
Rehearsal**（データが綺麗＝有望、ではない。むしろ最も有名なテーマなので既に織り込み済みの
可能性が高い）。無名Trigger型（防衛装備庁契約・GX個別プロジェクト等）は次候補として
事前登録するのみで、本Auditの結果を見てから選び直すことはしない（Multiple Testing回避）。

**方法論制約（本Audit中厳守）**：株価・リターン・Winnerは一切参照しない。すべて政府/企業の
公式開示日付ベースで収集。

---

## G1 — Economic Mechanism（Phase0で確定・変更なし）

PASS。

---

## G2 — Recurrence（追加確認）

半導体Level0（JASM関連投資）とLevel1（サプライチェーン強靱化補助金）で異なる時系列が確認できた：

**Level0系（JASM大型投資）**：
- JASM第1工場設立: 2021年（TSMC・ソニー） ※開所は2024年
- デンソー出資参画: 2022年11月
- トヨタ出資・第2工場建設決定: **2024-02-06**（[DENSO ニュースリリース](https://www.denso.com/jp/ja/news/newsroom/2024/20240206-01/)、出資比率 TSMC 86.5% / ソニーセミコンダクタソリューションズ 6.0% / デンソー 5.5% / トヨタ 2.0%）

**Level1系（METI サプライチェーン対策のための国内投資促進事業費補助金）**：
公募・採択発表が複数回繰り返されている（[METI](https://www.meti.go.jp/covid-19/supplychain/index.html)）：
令和2年7月17日／同11月20日／令和3年7月2日／令和4年7月1日（3次公募・85件採択）／令和5年6月2日
→ **年1〜2回のペースで公式採択リストが更新される構造**。これとは別に経済安全保障基金による
半導体サプライチェーン強靱化支援（令和4年補正予算3,686億円）の採択案件一覧も別途公表。

**判定: PASS**。Level0（大型単発投資決定）とLevel1（毎年複数回の補助金採択）で発生頻度の
性質が異なることが判明——Level1の方が「反復可能な投資Edge」の要件（§10 Frequency）に
適合しやすい。

---

## G3 — PIT Detectability

全て公式発表日が確定（企業ニュースリリース／METI採択発表）。**判定: PASS**。

---

## G4 — Role Mapping（Level 0定義の適用）

### Level 0（JV/非上場対応ルール適用）

JASM自体は非上場のため、Level0定義を「開示された出資・投資コミットメントを持つ上場親会社」に
適用：

| 出資者 | 上場状態 | ticker | 開示日 |
|---|---|---|---|
| TSMC | 台湾上場（国内kabu APIでは取引不可と推定・要確認） | 2330.TW | — |
| ソニーセミコンダクタソリューションズ | 非上場（ソニーグループ子会社） | → 親会社 6758.T | 2021年（初期）/2024-02-06（第2工場） |
| デンソー | 上場 | **6902.T** | 2022-11 |
| トヨタ自動車 | 上場 | **7203.T** | 2024-02-06 |

判定: **PASS（Level0は3銘柄に収束・企業名は当時の適時開示で確定・現在の情報からの遡及なし）**

### Level 1（METI補助金採択企業）

判定: **CONDITIONAL PASS**。プログラムの存在・PIT性・構造化された公式PDFリストとしての
入手性は確認済み（required data source: METI採択結果PDF、無料公開・追加契約不要）。ただし
本Auditのスコープ内（1session・実装なし）では、PDF内の個別企業名の全件抽出までは未実施
（3次公募だけで85件）。**次段階で企業名抽出が必要（人手 or パース処理、株価は見ない）**。

---

## G4.5 — Investable Universe（新規Gate・株価水準は見るがreturn/勝敗は見ない）

既存`src/configs/universe.yaml` / `rsr_universe_42.csv`を確認した結果、**重大な発見**：

Level0候補のうち **7203.T（トヨタ自動車）は既に現行Core Universe（rsr_universe_42.csv /
G29_V2）に含まれている**。さらに、現行Universeには半導体製造装置の代表的銘柄が既に
複数含まれている：

| ticker | 銘柄 | 現行Universe内での位置付け |
|---|---|---|
| 8035.T | 東京エレクトロン | 電機精密（元からG29_V2に採用） |
| 6920.T | レーザーテック | 電機精密（元からG29_V2に採用） |
| 6857.T | アドバンテスト | 電機精密（V2追加） |
| 7203.T | トヨタ自動車 | 輸送機器（元からG29_V2に採用）＝**JASM Level0投資家と重複** |

**判定: CONDITIONAL PASS（重大な留保付き）**。TSMC（台湾上場）は現行パイプラインでは
投資不可と推定（domestic kabu API・特定口座の対象外・要ユーザー確認）。ソニーグループ
（6758.T）・デンソー（6902.T）はTSE上場・現行未採用のため新規候補になり得る。しかし
トヨタ（7203.T）・東京エレクトロン（8035.T）・レーザーテック（6920.T）・アドバンテスト
（6857.T）は**既に現行Core Universe（Study95でKILL・Study101でTOPIX劣後確定済み）の
構成銘柄そのもの**。

---

## G5 — Candidate Generation Rule（固定）

```
Rule:
  Level0 = JASM/TSMC日本向け投資公表において、開示された出資比率を持つ
           上場企業（親会社含む）。TSMC自体は国内取引不可のため除外。
  Level1 = METI「サプライチェーン対策のための国内投資促進事業費補助金」
           および経済安全保障基金・半導体サプライチェーン強靱化支援の
           採択事業者のうち、TSE上場企業。
  Expansion = Level1まで（Level2以降のサプライヤーのサプライヤーは
              本研究では禁止・§Prohibited Research準拠）
```

固定後、同一入力（METI公式PDF・企業IR開示）から同一銘柄集合が再現される。人手判断は
「採択事業者名→証券コード」の名寄せのみ（経済的Role判断は含まない）。

**見積もり候補数**: Level0=2〜3銘柄（TSMC除く）＋Level1=未確定（次段階でPDF抽出後に確定、
過去の採択件数からは全産業横断で年間50〜90件規模、うち半導体関連・上場企業は一部）。

---

## G6 — Independent Edge仮説（保存・Matched Control必須化）

```
H0: Candidate Universe内の既存Signal hit rate = 全市場の同一Signal hit rate
H1: Candidate Universe内 > 全市場 かつ Candidate Universe内 > Matched Control

比較3群（固定・変更不可）:
  A: Full Market + Existing Signal
  B: Capital Flow Candidate + Same Signal
  C: Matched Control（同規模・同業種・Trigger非該当）+ Same Signal

Matched Control選定基準（事前固定）:
  - 同業種（電機精密／輸送機器等、現行sector taxonomy準拠）
  - 時価総額が近い銘柄
  - 対象期間中にLevel0/Level1のいずれのTriggerも受けていないこと
```

**G4.5の発見により、この仮説検証は事実上困難になっている**——Candidate Universeの主要銘柄
（7203.T, 8035.T, 6920.T, 6857.T）がFull Market比較のベースラインである既存RSR42
Universe自体に既に含まれているため、A（全市場）との比較がB（Candidate）との差を正しく
測れない可能性がある。これはG6実行前に解消すべき設計上の課題として記録する。

---

## G7 — Market Timing Potential（評価項目のみ事前登録・実行なし）

```
評価予定日:
  Level0: 2024-02-06（トヨタ・デンソー出資公表日）
  Level1: 各METI採択発表日（複数）

評価項目（Phase2で初めて株価を見る際に使用。今は登録のみ）:
  - Trigger date と Candidate confirmed date の差（ゼロが理想）
  - Trigger発表日翌営業日のvolume/price reaction有無
  - 発表から数週間かけて情報が段階的に具体化するか（一過性ニュースでないか）
  - 6758.T/6902.Tは大型株のため、アナリストカバレッジが厚く即日織り込みの可能性が高い
    （事前の定性的懸念として記録。判定はPhase2実施後）
```

Phase1では株価を見ない原則を厳守。上記は「何を測るか」の設計のみ。

---

## Holding Period Warning（Phase3設計への申し送り）

Capital Flow型Trigger（設備投資→稼働→業績反映）の自然な実現期間は四半期〜年単位
（JASM第2工場は発表2024-02から稼働目標2027年末＝約4年）。既存Production の
`turtle_exit=55d` / `max_hold_days=60` をそのまま流用すると、本来の業績反映局面を
待たずに機械的に手仕舞う可能性が高い。**Phase3のBT設計では独立した保有期間ルールを
設計すること（既存Exit流用禁止）**。

---

## Final Gate 判定

| Gate | 判定 |
|---|---|
| G1 Mechanism | PASS |
| G2 Recurrence | PASS（Level0単発・Level1年1-2回で性質が異なる） |
| G3 PIT | PASS |
| G4 Role Mapping | Level0=PASS／Level1=CONDITIONAL PASS（企業名抽出未完了） |
| G4.5 Investable Universe | **CONDITIONAL PASS（重大な留保）**——主要候補が既存Core Universeと高い重複 |
| G5 Candidate Rule | 固定済み・PASS（Level2以降拡張禁止） |
| G6 Independence仮説 | 保存済みだが、既存Universeとの重複によりA/B比較の設計修正が必要 |
| G7 | 評価項目のみ事前登録（未実行） |

**総合判定: CONDITIONAL PASS — ただし半導体型（特にLevel0）は「新規Edge」としての
価値が疑わしいことが判明。**

理由：JASM出資組（トヨタ）も主要装置メーカー（東京エレクトロン・レーザーテック・
アドバンテスト）も**既に現行42銘柄Core Universeの一部であり、そのCore自体が
Study95/101でKILL/RED判定済み**。同じ銘柄に同じSignal（RSR/Momentum）を「半導体
テーマ」の皮を被せて再適用しても、統計的に独立した新しいEdgeにはなり得ない
（Gate 6の趣旨そのものに抵触するリスクが高い）。

**価値が残るとすれば Level1（Sony Group 6758.T・Denso 6902.Tのような、現行Universe外
の周辺企業、および METI補助金採択企業のうち現行未採用の中小型株）のみ**。これは
Pipeline Rehearsalとしては機能した（Rule固定・Level0定義・Investable Gateの動作確認は
完了）が、**半導体型・特にLevel0を投資Edgeの本命として次段階（Phase2 Market Timing）に
進める根拠は弱い**。

## 推奨アクション

1. Level0（JASM投資家）はこれ以上追わない——既存Universeとの重複により独立性を示せない
2. Level1（METI補助金採択企業、現行Universe外）のみに絞ってPDF企業名抽出を完了させるか、
   費用対効果が低いと判断してこの半導体Trigger自体をここでSTOPするか、を次のASK_FIRSTとする
3. 無名Trigger型（防衛装備庁契約データ等）を次の本命候補として、勝者を見ずに事前登録する
   （現時点でのユーザー選択・半導体結果を見た上での選び直しは禁止—§Multiple Testing）

## Time Budget

1 session（実績: 同セッション内で完了）。BT・実装・新規データ契約は未実施。

---

## Closure（2026-08-02 追記）

### Level0 — 正式STOP

**STOP理由: 既存Core Universe（rsr_universe_42.csv）との構造的重複によるIndependent Edge欠如。**
トヨタ7203.Tが直接重複。企業を見て後から判定したのではなく、Candidate Generation Rule
（G5固定済み）を機械適用した結果として重複が判明した——Winner Blind原則には抵触しない。

### Level1 — 簡易規模チェック（既存資料ベース・PDF全件解析なし）

METI採択事業者PDFへの直接アクセスは本セッションで失敗（403）。よって正確な採択企業名との
突合はできていない。一般に知られる日本の半導体材料・装置サプライチェーン構成企業のうち、
現行42銘柄Universeに**含まれないもの**を参考情報として列挙（確認度: 業界一般知識ベース、
METI採択リストとの一致は未確認・要検証）:

| ticker | 銘柄 | 現行Universe外か | 備考 |
|---|---|---|---|
| 4063.T | 信越化学工業 | 外 | シリコンウェハ大手 |
| 3436.T | SUMCO | 外 | シリコンウェハ |
| 4004.T | レゾナック・ホールディングス | 外 | 半導体材料 |
| 4186.T | 東京応化工業 | 外 | フォトレジスト |
| 7735.T | SCREENホールディングス | 外 | 半導体製造装置 |
| 4185.T | JSR | 要確認 | JIC主導TOB案件（2023-2024）により非上場化した可能性——政府主導の資本再編で**投資不可能になった**事例。むしろ「Candidate化する前に上場廃止で消える」リスクの実例として記録価値あり |

**判定: STOP（Level1含め半導体型全体を終了）**。理由:

1. 正確な採択企業リストとの照合ができていない（METI PDF未取得）ため、G4のCONDITIONAL PASSを
   PASSへ格上げする根拠がない
2. 仮に一致したとしても、挙がる候補は信越化学・SUMCO・東京エレクトロン級の大型・高流動性・
   アナリストカバレッジ厚い銘柄が中心——Level0で懸念したG7（既に織り込み済み）と同種のリスクが
   Level1にも及ぶ可能性が高く、追加検証の期待値が低い
3. Time Budget対効果：残作業（PDF全件解析・企業名突合）のコストに対し、得られる情報の限界価値が
   低いと判断

### 総合STOP

**半導体型Capital Flow Generator研究は Level0/Level1 とも本Auditをもって終了する。**
Pipeline Rehearsalとしての目的（Mechanism定義→Trigger検出→Level0/1定義→Candidate Rule固定→
Investable Gate→Independence仮説設計、の一連の手順が実行可能であることの確認）は達成された。

### 次候補の事前登録（Phase0時点の判定に基づく・結果を見た選び直しではない）

`mechanism_feasibility_2026-08-02.md`のG4判定は、GX型=FAIL、防衛型=CONDITIONAL PASSと
**半導体Phase1実施前に既に確定していた**。したがって次候補として防衛型
（防衛装備庁「調達契約情報の公表」）を選ぶことは、半導体の失敗を見て別テーマを探す行為ではなく、
Phase0で既にランク付け済みの次点候補へ進む手続き上の帰結である。GX型はPhase0 G4 FAILのため
候補から除外済み。

正式な次候補: **防衛装備庁契約データ型Trigger**（Phase0/1は別セッションで新規実施。本Audit
内では着手しない）。
