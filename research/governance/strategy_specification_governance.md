# Strategy Specification Governance v1.0（2026-07-24制定）

**目的**: 研究の中心を Study から **Strategy Specification（戦略仕様書）** へ移行。
Study = 仕様書を改善するための実験。仕様書 = タグ付きリリース（不変・永続）。
Study101教訓（「いろいろ足したら何が効いたのか分からない」）のプロセスレベル再発防止。

---

## 1. 階層構造（2026-07-24改訂: Family層を正式統合・確定）

```
RGP（Return Generation Process・勝ち方の分類・唯一の分類軸）
    ↓
Family（RGP内の実装アプローチ族。例: RGP-1 Underreaction → {PEAD族, Guidance Revision族, Analyst Revision族}）
    ↓
Strategy Specification（versioned・不変・REGISTRY登録単位）
    ↓
Study（改善実験・単一変数比較）
    ↓
Version Up（新仕様書の発行。旧仕様書は永続）
```

この4層構造（RGP→Family→Specification→Study）が研究体系の唯一の入口。新規研究依頼は必ず**§10 Research Entry Gate**を通過する。

RGPの定義・分類・境界・独立性評価は `rgp_taxonomy_v1.0.md`（本書の上位文書。研究体系の最上位・roadmapより上位）を正とする。**Frozen（2026-07-24ユーザー決裁）**——例外: RGP-7 Growth CompoundingとSmall Growth分類のみPROVISIONAL（v1.1で再評価）。

Family層は現時点で**物理フォルダではなくREGISTRY.md上の論理グルーピング**として管理する（RGP単位フォルダ規約=本書§2は変更なし・Family単位のサブフォルダ化は将来検討・ASK_FIRST）。Family自体はFrozen仕様書ではないため、既存Specification（Frozen）を一切変更せずに新設・変更できる——これが物理フォルダ化を急がない理由。

## 2. ディレクトリ規約

```
research/
  governance/
    strategy_specification_governance.md   ← 本書
  strategies/
    REGISTRY.md                            ← 戦略台帳（唯一の索引）
    common_conventions_v1.0.md             ← 共通執行/検証規約（これもversioned）
    <rgp_slug>/                            ← RGP単位でフォルダ分割
      <strategy>_v<MAJOR.MINOR>.md         ← 仕様書（1版=1ファイル・不変）
      CHANGELOG.md                         ← 版間の変更理由・由来Study
  studies/
    StudyNNN/                              ← 実験（従来どおり）
```

RGPフォルダ現行: `pead/`（underreaction/drift系）・`edmr/`（overreaction/mean reversion系）・`quality/`（quality premium/junk avoidance系・2026-07-24ユーザー指示で追加）。
新RGP追加はユーザー決裁（ASK_FIRST）。

## 3. 仕様書フロントマター（必須項目）

```yaml
strategy:     <正式名>
version:      <MAJOR.MINOR>
status:       Draft | Frozen | Superseded
verdict:      UNTESTED | PASS | FAIL | INCONCLUSIVE
role:         Calibration Benchmark | Production Candidate | Research Hypothesis
parent:       <親仕様書 or none>
derived_from: [文献・Study番号]
rgp:          <Return Generation Process>
conventions:  common_conventions_v<X.Y>
created:      YYYY-MM-DD
origin:       <起案書パス>
```

**status と verdict は独立軸**（重要）:
- `status` = 仕様書のライフサイクル（Draft→Frozen→Superseded）。文書の状態。
- `verdict` = 実測判定。**FAILでも仕様書はFrozenのまま永続**（比較対象・較正原点）。削除禁止。

**role**（2026-07-24追加・Quality MF起案時にユーザー指摘を反映）— 仕様書の研究上の位置づけ。3値固定・混同禁止:
| role | 意味 | 典型例 |
|---|---|---|
| Calibration Benchmark | 文献忠実複製。勝つことを期待しない。改善Δの測定原点 | PEAD Classic, Quality MF Classic |
| Production Candidate | 実装データが全て充足・live候補になりうる仕様 | PEAD Practical, Quality MF Practical, EDMR |
| Research Hypothesis | 自プロジェクト実測（間接証拠含む）ベースの仮説段階・直接検証未実施 | Quality Value SmallMid |

一目でどの戦略が「較正用」「実装候補」「仮説段階」かを判別するための列。REGISTRY.md にも列として反映する。

## 4. 不変性規則（核心）

1. **Frozen仕様書のパラメータ変更は一切禁止。** 「Exitを60d→40dにしたい」→ v1.0は触らない。新版 v1.1 を起案する。
2. **単一変数原則**: MINOR版上げ = 親版から**変更は1変数のみ**（Entry変更のみ/Exit変更のみ/Filter追加のみ）。効果の帰属を常に一意にする。
3. MAJOR版上げ = 構造変更（サプライズ定義変更・方向反転・RGP再分類など複数変数不可避の場合）。
4. Superseded は「新版が正典になった」ことのみ意味し、旧版の参照・比較利用は永続。

## 5. Version Upフロー（Study→仕様書）

```
Study実測完了
  ↓
新版提案（Draft起案・親版とのdiff=1変数を明記・事前登録効果量つき）
  ↓
ユーザーレビュー（ASK_FIRST）
  ↓
採用 → Frozen化・REGISTRY登録・CHANGELOG追記・親版status確認（Supersededか併存か）
却下 → Draft却下記録をCHANGELOGに残す（却下も履歴）
```

Studyは仕様書を**変更できない**。できるのは新版の**提案**のみ。

## 6. REGISTRY規則（2026-07-24改訂: Family層統合）

- `research/strategies/REGISTRY.md` = 全戦略の唯一の台帳。**必ずRGP → Family → Specificationの順で閲覧可能な構造**を維持する（RGP見出し内をFamily小見出しで細分し、各Familyの下にSpecification表を配置）。
- 台帳の列: Strategy / Version / Role / Status / Verdict / Evidence / Parent / File。
- 追加タイミング = Draft起案時。Verdict更新タイミング = fresh run実測後のみ（Study52規則・キャッシュ判定禁止）。
- 系譜図（parent連鎖）をREGISTRY内に保持。系譜の見えない仕様書は登録不可。
- **新Family追加はASK_FIRST**（新RGP追加と同格の決裁事項。§2の新RGP規則に準ずる）。

## 7. 既存統治規則との接続（本書は追加であり代替ではない）

| 既存規則 | 適用 |
|---|---|
| PARAMS_LOCKED / CIRCUIT | live執行層で常に優先。仕様書はresearch層の定義 |
| ASK_FIRST | 新版Frozen化・実装スクリプト作成・新RGP追加 |
| Freeze Rule系 | Study実行の可否は roadmap 統治に従う（本書はStudy許可を与えない） |
| fresh_run_required | Verdict記入の前提 |
| Study82F様式 | 効果量表記 Spread/95%CI/NW-t/n を全Verdictに必須 |
| 検証ゲート | common_conventions_v1.0.md §検証プロトコル |

## 8. 現行系譜（2026-07-24時点）

```
PEAD Classic v1.0 (Frozen/UNTESTED・事前予測FAIL登録)
    ├── PEAD Practical v1.0 (Frozen/UNTESTED)
    └── EDMR v1.0 (Frozen/UNTESTED・方向反転=MAJOR分岐・RGP別フォルダ)
```

起案原本: `research/studies/Study082/pead_v1_spec_3proposals_2026-07-24.md`（1案→Classic / 2案→Practical / 3案→EDMR）。

## 9. RGP越境禁止（Cross-RGP Contamination Prohibition・2026-07-24追加）

**規則**: 各仕様書は自身のRGPに属さないシグナル・条件を追加してはならない。

禁止例（quality系仕様書を例に・他RGPでも同様に適用）:
```
✗ RSRランキング条件の追加
✗ 決算イベント/サプライズ条件の追加（PEAD/EDMR由来のシグナル混入）
✗ テーマ株・材料株条件の追加
✗ 出来高急増条件の追加
✗ ブレイクアウト・トレンドフォロー条件の追加
```

**理由**: Study95（cross-sectional momentum・FAIL_ZERO_SPREAD）/ Study99（RS高パーセンタイル単調悪化）/ Study83（TSMOM負リターン）/ Study82 PhaseD（継続型シグナル全滅）——日本市場でmomentum/RS系の複合オーバーレイは繰り返し逆転する。Study101「複合の結果、何が効いたのか分からなくなった」の再発防止が本規則の直接動機。

**手続き**: 複合効果を検証したい場合、既存仕様書への暗黙追加は禁止。**新規の複合RGP戦略として別途明示的に起案する**（新RGPフォルダ・ASK_FIRST・親版はどちらか一方または両方を明記）。「ついでに1個足す」は本書が最も警戒する失敗モード。

## 10. Research Entry Gate（研究開始ゲート・2026-07-24制定、2026-07-25 Step0追加）

**規則**: あらゆる新規研究依頼（「〇〇を書いて」「〇〇を検証したい」の形をとるものすべて）は、Study起案やStrategy Specification起案の前に、必ず以下5ステップを通過する。ステップを飛ばした依頼（例: Research Questionを固定せず直接Specificationを書く）は差し戻す。

```
Step 0 — Research Question（検証したい問いの固定）
    「Qualityは効くのか？」「Turnaroundは独立RGPなのか？」「Small Growthは存在するのか？」等、
    検証対象の問いを1文で確定し記録する。以降のStep1-4はこの問いに対してのみ進める。
    問いが曖昧・複数混在の場合はStep1へ進まず問いの分解から始める。
    ↓
Step 1 — 既存RGPへ分類可能か？（rgp_taxonomy_v1.0.md §1/§4を参照。Step0の問いをRGPにマッピング）
    YES → 既存Family（またはFamily候補）へ属させる → Step 2 へ
    NO  → 新RGP提案（原則禁止・例外扱い・ASK_FIRST・rgp_taxonomy_v1.1として起案）

Step 2 — そのFamilyは既存REGISTRYに存在するか？
    YES → Step 3 へ
    NO  → Family追加起案（ASK_FIRST・REGISTRY.mdへの新規グルーピング追加）

Step 3 — そのFamily内にSpecificationが既に存在するか？
    YES → Step 4 へ（既存Specificationの改善実験。§5 Version Upフローに従う）
    NO  → Specification v1.0起案（新規Family内の初版・§3フロントマター必須・Frozen前提で設計）

Step 4 — Study起案（Step0のResearch Questionに対する実験を設計）
    起案時に**Study Role**（§11参照: Calibration/Validation/Replication/Improvement/Exploration）を1つ付与する。
```

**Step0を設けた理由（2026-07-25追加・ユーザー指摘）**: Research Questionを先に固定しないと、Study実行の途中でQuestion自体が変わってしまう。Study101（旧フジコ法検証がいつの間にか多要素の複合検証に肥大化し「何が効いたか分からなくなった」事例）がこの典型——単一変数原則（§4）はSpecification間の変更を1つに絞る規律だが、Step0はその手前、**そもそも何を検証しているのかを1個の問いに絞る**規律。

**運用上の含意**:
- 「Quality Compounderを書いて」ではなく「RGP-5（Quality Premium）の文献を調査して」という単位でタスクが来ることを前提とする。RGP分類が先、Specificationは後。
- Step1でYESかつ既存Familyに複数のSpecificationが既にある場合（例: RGP-1 → PEAD族 → {Classic, Practical}）、新規提案は既存Specificationのv1.1（単一変数改版）か、同一Family内の新規v1.0（別アプローチ）かを明示する。
- Step1でNOの新RGP提案は、rgp_taxonomy_v1.0.md §1（Tree）・§2（各RGP定義）・§4（境界整理）・§5（独立性マトリクス）の4箇所を同時に更新する起案でなければ受理しない（Taxonomy全体の整合性を都度検査する規律）。
- 本ゲートはREGISTRY.mdの構造（RGP→Family→Specification順）と対になる。REGISTRYで該当RGP/Familyを検索した結果が本ゲートの判定根拠になる。

## 11. Study Role分類（2026-07-25追加）

**規則**: Study起案時（Entry Gate Step4）に、以下5値から1つを`Study Role`として付与する。目的はStudy一覧の可読性向上——「何のために実行したStudyか」を一目で判別する。

| Study Role | 意味 |
|---|---|
| Calibration | 較正原点の測定（Specification側のCalibration Benchmarkに対応する実測） |
| Validation | 既存の仮説・分類・判定の妥当性検証（YES/NO判定が主目的） |
| Replication | 文献・先行研究の再現（オリジナル主張が自データで再現するかの検証） |
| Improvement | 既存Specificationの単一変数改善実験（§4単一変数原則に基づくVersion Up候補） |
| Exploration | 新規の問い・未分類領域の探索（Entry Gate Step1でNO/未分類の場合に生じやすい） |

**例**（過去Studyへの遡及適用は行わない。概念説明のための参考例のみ）: Study82（PEAD再現実験）はReplication的性格・Study95（CSモメンタム仮説検証）はValidation的性格・Study110（未来勝者オントロジー探索）はExploration的性格。

**適用範囲**: 本分類は**2026-07-25以降に起案する新規Studyから適用**。既存Study（research/studies/StudyNNN/*）への遡及的なRole付与は本書のスコープ外（大規模作業のため別途ASK_FIRST起案）。Study Roleの記録先は各Studyの起案書冒頭（研究者の自由記述でよい・Specificationのような厳密なfrontmatter必須化はしない）。
