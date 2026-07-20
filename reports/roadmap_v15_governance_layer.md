# Roadmap v1.5 — Governance Layer（統合正典・唯一の生きたロードマップ）

**日付**: 2026-07-20
**STATUS**: ★★ **CURRENT CANON（現行正典・唯一のエントリーポイント）** ★★
**性格**: 文書のみ（BT・コード変更・新規仮説生成なし）。v1.3統治層+v1.4戦略層+ユーザー最終統合指示を単一文書へ統合。以後、ロードマップ参照は**本書から開始する**こと。

---

## §R Roadmap Registry（乱立解消・参照規則）

| ファイル | STATUS | 扱い |
|---|---|---|
| **本書** `roadmap_v15_governance_layer.md` | **CURRENT CANON** | 唯一の生きた正典。矛盾時は本書が優先 |
| `roadmap_revision_2026-07-19.md`（v1.0-v1.4） | **HISTORY/ANNEX** | CP詳細仕様（CP3ゲート条件文・自動RED境界の原文・永久禁止7項原文）の参照付録+改版履歴。新規追記禁止 |
| `roadmap_v14_strategy_layer.md` | **ANNEX** | Route詳細仕様（各Routeの完全テーブル・ARCH再評価根拠）の参照付録。新規追記禁止 |
| `study103_design.md` | **ACTIVE（Study起案書）** | Study103実装の入力仕様。§9A/§9B追補済み |
| `complete_execution_roadmap_2026-07-04.md` | **FROZEN** | 歴史記録（凍結済み・バナー済み） |
| `final_research_roadmap_2026-07-04.md` | **SUPERSEDED**（本改定でバナー追加） | Study01-73完結宣言としてのみ有効。研究プログラム定義としては失効 |
| `fujiko_r2_research_roadmap.md` | **PARALLEL**（限定有効） | Study87-97採番と仮説priorの管轄のみ。Candidate A実装はRoute E1へ合流（二重実装禁止） |

**アンチ乱立規則（恒久）**: 今後のロードマップ改定は**本書のin-place編集+§Vのchangelog行追加のみ**で行う。ロードマップ用の新規ファイル作成は禁止（v1.6等も本書の版上げで表現）。ファイルの物理削除は行わない — 本プロジェクトはappend-only監査文化（REJECT=資産・git履歴・research_stateからの相互参照）のため、「削除」は**正典指定の剥奪（STATUS格下げ）**として実施した。

---

## §0 研究目的（最終確定）

```
旧: 30%を達成する。
新: どこまで現実的に行けるか。どこで終了すべきか。終了後の正式着地点は何か。

Research roadmap → Research governance → Research portfolio management system
```

**結語（正典・全報告書の末尾に引用可）**:
```
Research exists to reject impossible routes as early as possible.

Success is not finding a 30% route.
Success is minimizing years spent on infeasible routes.
```

---

## §1 Study103の目的（最終確定）

```
Primary Objective:
  Attempt to falsify ambitious routes.

Secondary Objective:
  Determine feasible ceilings and terminal states.

Failure to falsify ≠ proof.
```

旧定義「Can we achieve 30%?」は廃止。Study103は「野心的経路の反証試行」+「現実的な天井と終端状態の確定」の二層目的を持つ。三原則・シナリオ凍結・自動RED境界は`roadmap_revision_2026-07-19.md`§6A（原文）のまま有効。

### §1A CP2後の行動（完全事前固定・v1.5.1）

**Study103終了後の「次に何をするか」の議論を禁止する**。以下の決定木が唯一の行動規則:

```
CP2 GREEN  → Study76へ進行・Budget制約（§6）維持・Satellite再順位付け（frontier準拠）
CP2 YELLOW → Satellite最大2本のみ着手（frontierの欠落プロファイル上位2本）→ CP3で再評価
CP2 RED    → Tier2 feasible?
               YES → Route B（目標をTier2へ再アンカー・研究継続）
               NO  → Tier1 feasible?
                       YES → Route A（TSMOM主軸・Study83のみ着手）
                       NO  → Route F（Trigger A発動・§4手順へ）
```

判定はGoal Ladder Sweepの機械出力のみを入力とする。本決定木の事後修正は永久禁止5該当。

---

## §2 Goal Ladder（最終版・番号体系刷新）

**⚠ 番号反転注意**: v1.4は上位=Tier1の降順だった。**v1.5からは昇順（数字が大きいほど野心的・Tier0=床）に統一**。旧文書を読む際は下記対照表を必ず使用。

| v1.5 Tier | 目標 | 対応Route | 旧v1.4呼称 |
|---|---|---|---|
| **Tier3** | CAGR 30% / Calmar ≥1.5 | Route C | 旧Tier1 |
| **Tier2** | CAGR 20-25% / Calmar ≥1.3 | Route B | 旧Tier2（同一） |
| **Tier1** | CAGR 10-18% / Calmar ≥1.0 | Route A | 旧Tier3 |
| **Tier0** | Market Return（TOPIX B&H equivalent） | Route F | 旧Tier4 |

- Study103は**同一MCから全Tierをread-out**（Goal Ladder Sweep・`study103_design.md`§9A）。新規シナリオ追加は禁止。
- 上位Tier閉鎖は下位Tierの研究を止めない。

---

## §3 Route体系（最終版）

詳細仕様（required studies/kill/fallback/capacity/難易度の完全テーブル）は`roadmap_v14_strategy_layer.md`§2-3（ANNEX）。本書は確定状態のみ:

| Route | 名称 | 目標 | 状態・条件 | Planning Prior（ordinal・v1.5.1） |
|---|---|---|---|---|
| **C** | 30% Frontier | Tier3 | CP2 GREEN ∧ 複数スリーブCP3通過 ∧ Study86執行parity→Study85統合（全AND） | **LOW** |
| **B** | Diversified Alpha | Tier2 | Core+PEAD+TSMOM。**現時点の主計画（Plan B）** | **MEDIUM** |
| **A** | Conservative Alpha | Tier1 | TSMOM主軸。**Study103 RED時の既定避難先（Plan C）** | **MEDIUM-HIGH** |
| **D** | **Dormant High Alpha Branch** | overlay | **休眠管理（v1.5確定）**: 独立Route化はStudy102 WHITE（旧称Study81 — 現採番はStudy102・v1.0改番済み）まで禁止。PIT証拠不足のため起動条件成立まで工数配分ゼロ | DORMANT（評価対象外） |
| **E1** | Core Replacement | — | Study76 WHITE→Clenow置換。fujiko_r2 Candidate A統合先（二重実装禁止） | MEDIUM |
| **E2** | Core Retirement | — | Study103 Case B優位 or Core CP3 fail→Satellite-only正式選択肢 | MEDIUM-HIGH |
| **F** | Terminal | Tier0 | §4トリガー成立時の正式着地 | MEDIUM |

**数値prior注記（v1.5.1・表現規律変更）**: 旧数値（C~5-10%/B~20-25%/A~30%/F~20-30%等）はordinal表現へ置換した。数値を参照する場合は必ず以下を付す:
```
Illustrative only. No statistical interpretation. Resource allocation only.
```

---

## §4 Route F 発動条件（正式版・事前固定）

```
Trigger A: CP2 RED ∧ Tier1もinfeasible（Goal Ladder Sweepで判定）
Trigger B: 全候補スリーブCP3不通過
Trigger C: Research budget exhausted（§6予算の枯渇・ユーザー認定）
Trigger D: 連続2年次サイクルで採用スリーブゼロ
Trigger E: Infrastructure burden > Expected research value（v1.5.1追加・ユーザー認定）
           例: kabu API仕様変更・J-Quants料金増加・保守コスト増大・Claude利用制限。
           インフラ維持コストが研究の期待価値を上回った時点で研究価値は負 — 実務上
           十分起こり得る終了条件として正式化。機械判定不能のためユーザー認定制
```

**Trigger Dと既存Kill条件の整合（v1.5確定・エスカレーション梯子）**:
```
連続2四半期採用ゼロ → 研究縮退（月次メンテモード・旧正典Kill条項のまま有効）
連続2年次サイクル採用ゼロ → Route F発動（研究正式終了）
```
（四半期条項=減速・年次条項=終了。矛盾ではなく二段階。）

**Route F発動時の状態遷移（固定）**:
```
Entry Freeze permanent
  ↓
TOPIX B&H equivalent（資本の正式着地・現金退避との選択はユーザー決裁）
  ↓
Infrastructure preservation only（執行・検証・データ基盤は凍結保存）
  ↓
Research terminated（decay monitoring only）
```

---

## §5 Planning Prior（正式規律）

```
Planning Prior

Purpose : Research resource allocation only.

Not probability of truth.
Not evidence.
Not adoption criterion.

Values are ordinal only.（順序情報のみ有効・数値の絶対値に意味を持たせない）

Numerical priors shall never be cited in CP2/CP3 decisions.
```

CP2/CP3の判定文書にprior数値を引用した時点でStudy52型汚染として扱う。priorの更新はCP2/CP3判定後のみ・本書§Vに改版行として記録。

---

## §6 Research Budget（正式追加）

**単位の定義（重要・誤読防止）**: 本予算は**研究工数（研究セッション・実装/検証時間）の配分**であり、**資本配分ではない**。実弾の資本配分は常に個別ASK_FIRST+CP4体系の管轄。

| カテゴリ | 配分 | 対応Study |
|---|---|---|
| Core reconstruction | **35%** | Study75完了・Study76・FUJIKO 2.0（E1経路） |
| Satellite research | **45%** | Study103→（CP2後）83/80/82。Study102はRoute D休眠のため**当面0%**（45%枠の内数として起動時に再配分） |
| Exploratory routes | **10%** | CP2 YELLOW時の新アルファ探索のみ。GREEN/RED時は未消化のままReserveへ繰入 |
| Reserve | **10%** | インシデント・監査・インフラ修理 |

**並行上限（発散防止・固定）**:
```
Maximum active routes     = 2 （現在: Route B準備 + E1/Core処遇 = 2）
Maximum concurrent studies = 2 （現在: Study75 + Study103 = 2 ✓ 上限充足 — 第3のStudy起案は
                                いずれかの完了まで禁止）
```

---

## §7 Route Transition Matrix（正式図）

```
Study103 GREEN                → Route B/C 続行（Satellite再順位付け）
Study103 RED                  → Route A（Tier1 feasibleなら）or Route F（Trigger A）
Study103 Case B superior      → Route E2（Core retirement正式選択肢化）
Study76 WHITE                 → Route E1（Clenow置換・fujiko_r2 Candidate A統合）
Study76 BLACK ∧ Core CP3 fail → Route E2
Study82 監査FAIL              → Route B → A 降格
Study83 CP3 fail              → Route A構成不能 → Core単独 or Route F
Study80 fail                  → Route C到達不能 → Route B天井確定
Study102 WHITE                → Route D起動（休眠解除・ユーザー決裁でoverlay配分）
Study102 fail                 → Route D恒久閉鎖（他Route無影響）
All sleeves fail              → Route F（Trigger B）
```

---

## §8 Study103 Outputs（成果物固定・7点）

```
1. Goal Ladder Sweep          （Tier0-3 feasibility・GREEN/YELLOW/RED×4Tier・Tier0含む）
2. Goal frontier              （Calmar制約別の最大到達CAGR曲線・主図表）
3. Route Transition Matrix    （§7の各遷移がどのシナリオで発火するかの対応表）
4. Termination Probability    （定義: 全シナリオ×MC試行中、Tier1すらinfeasibleとなる割合）
5. Core Retirement Probability（定義: Case B frontierがCase Aより緩いシナリオの割合）
6. Research Budget Recommendation（§6配分の改定提案・advisory・採用はユーザー決裁）
7. Research Continuation Policy（研究継続価値の判定+Route F前倒し要否=Terminal state
                                recommendation含む・§1A決定木への機械入力・advisory）
```

4・5は上記の**機械的定義に固定**（実装時の再解釈禁止）。6・7はadvisory出力であり、ゲート判定（1-3）と混同しないこと。**全出力はadvisory — 自動採用なし・append-onlyで記録**。

---

## §8A Post-CP2 Focus（v1.5.4・Study103結果後の研究フェーズ確定・Route状態マトリクス）

### §8A-0 現在地の確定（正式状態・v1.5.4）

```
Research Status : Ceiling Measurement Phase
Primary Route    : Route B
Fallback         : Route A
Dormant          : Route C
Terminal         : Route F
```

Route Aは閉鎖しない（時期尚早）。Study103でTier2・Tier1が共にGREENであるため、Route Bが崩れた場合に
`B → A → F`という縮退経路が存在すること自体が研究OSの安定性を構成する。

### §8A-1 Route状態マトリクス（正式・v1.5.4）

| Route | Status | 内容・起動/再起動条件 | 研究工数 |
|---|---|---|---|
| **B** | **ACTIVE**（正確には**Route B Frontier Validation Phase**——「採択済み運用戦略」ではなく検証中） | Core+PEAD+TSMOM・目標20-25%/Calmar1.3+。現時点でPEAD=existence unknown（Study82未完了）・TSMOM=assumption only（Study83未実施）のため、frontier自体が未確定 | Satellite枠45% |
| **A** | **STANDBY**（削除禁止・工数0%で待機） | 起動条件（OR）: A. Study82 FAIL / B. Route B upper bound劣化 / C. 連続2年次サイクル採用ゼロ | 0%（起動時に再配分） |
| **C** | **DORMANT**（§8A既定・変更なし） | 再起動条件（OR）: A. PEAD実測>assumption+5pp / B. TSMOM実測>assumption+5pp / C. 新規直交スリーブがCP3通過。これ以外での再審議禁止 | 0% |
| **F** | **TERMINAL** | Trigger（OR・v15§4定義のA-Eと同一）: A. Tier1 infeasible / B. 全スリーブfail / C. Research budget exhausted / D. 連続2年次サイクル無採用 / E. Maintenance burden>expected research value | — |

### §8A-2 Research Freeze Rule（正式・v1.5.4・重要）

```
No new alpha implementations or backtests may be initiated until
Study82 determines whether Route B remains viable.

Proposal documents only are allowed.
```

**適用**: Study83実装（新規BT・新規スクリプト）はStudy82完了（PASS/FAIL/UNKNOWN確定）まで着手しない。
`study83_proposal.md`が示した「データ独立のため並行着手も選択肢」は本ルールにより**上書き・凍結**
（提案書自体は保持するが、実装着手の判断はStudy82完了後に限定する）。Proposal文書の作成・改訂は
本ルールの制約対象外（BT・コード・データ取得を伴わないため）。

### §8A-3 CP3の位置付け（Route B生死判定・ケース分岐・v1.5.4）

CP3で以下を判断する（Study82・Study83実測が出揃った時点）:

```
Case1: PEAD PASS ∧ TSMOM GOOD           → Route B confirmed（正式運用へ）
Case2: PEAD FAIL                        → Route A promotion（Route B構成をCore+TSMOMへ縮小
                                            or Route A標準へ切替。§4決定木のB→A降格と同型）
Case3: PEAD PASS ∧ TSMOM >> assumption  → Route C reactivation review（§8A-1 Route C条件Bに該当
                                            した場合の正式レビュー起動——再起動そのものではない）
```

### §8A-4 当面の研究目的（改定・v1.5.4）

```
旧: 30% / Calmar1.5

新:
Primary  : Determine the achievable ceiling of Route B.
Secondary: Monitor conditions that may justify reactivating Route C.
```

（§8A初版のPrimary/Secondary文言と同一趣旨・本節で正式表現として確定）

**フェーズ転換宣言**: 「新しい夢を探すフェーズ」から「**Route Bの実力上限を定量化するフェーズ**」へ正式移行。

```
Primary  : Determine achievable ceiling of Route B.
Secondary: Monitor conditions for Route C reactivation.
```

### Route C（30% Frontier）の正式位置付け — DORMANT（閉鎖ではない）

```
Route C = DORMANT
```

**再起動条件（固定・OR結合・これ以外での再開は禁止）**:
```
A. TSMOM実測 > Study103仮定（Base）+5pp
OR
B. PEAD実測  > Study103仮定（Base）+5pp
OR
C. 新しい独立スリーブがCP3通過
```

上記いずれも成立しない限りRoute Cの再審議自体を禁止する（Study52型延命の再発防止）。Route再起動の判定は実測値確定時に機械的に行い、「惜しい」「もう少しで」といった裁量判断は挟まない（§9原則5と同型の規律）。

### 完全版ロードマップ（Program Phase 0-5・v1.5.4・マクロ粒度）

```
Phase0  Core reset                                          ✓ 完了（Study74/100/101・CP1 Expectation Reset）
Phase1  30% route falsification                              ✓ 完了（Study103設計・凍結仕様確定）
Phase2  Portfolio frontier measurement                        ✓ 完了 → CP2 RED
Phase3  Route B validation                                    ← 現在地
Phase4  Route B ceiling update（CP3）
Phase5  Determine final state: Route A/B/C/F
```

### Phase3-4詳細（Route B Validation・Program Phase3-4のミクロ内訳・固定）

```
Phase A: Study82 Phase0.1（J-Quants API疎通確認）→ 財務/決算発表API接続可否の確認（Priority 1）
         Study82 Phase0（PEAD発表日時精度監査）  → PEAD研究可能か（PASS/FAIL/UNKNOWN）
Phase B: Study83 Proposal（起案のみ・実装せず）   → TSMOMの情報価値の事前評価
Phase C: Study83 Implementation（Study82完了後のみ・§8A-2 Research Freeze Rule） → Route B frontier更新
Phase D: Study82 Alpha Study（Phase0=PASS後）     → PEAD期待値の実測
Phase E: Route B Ceiling Re-estimation            → 20-25%/Calmar1.3+の実測確認 → CP3判定へ
```

### 将来のCP3イメージ（Route B自体の生死判定・事前告知）

```
GREEN  → Route B正式運用
YELLOW → Satellite縮小（構成スリーブを絞る）
RED    → Route A または Route F
```

### 明確に閉鎖されるもの（Post-CP2期間中）

```
30%達成研究（Primary Goalとしては終了・Route C DORMANT）
Capital Route（Study74で終了・恒久）
無制限Satellite探索（concurrent studies≤2を維持）
新しいRoute乱立（§R Registryのアンチ乱立規則と同型で禁止）
Route C延命研究（再起動条件A/B/C以外での議論禁止）
```

### 優先順位（確定・v1.5.4・5段固定）

```
第1優先: Study82 Phase0.1（J-Quants API疎通確認）— 情報価値最大。失敗ならRoute B自体が大幅弱体化
第2優先: Study82 Phase0（決算日時精度監査・PASS/FAIL/UNKNOWN出力のみ・アルファ測定は絶対に行わない）
第3優先: Route B Viability Review（Study82結果を受けたfrontier再確認 — PEAD70%仮定の生死確認）
第4優先: Study83 Proposal（文書のみ・実装禁止・BT禁止 — ここまでは許容）
第5優先: Study83 Implementation（Study82 PASS後のみ・§8A-2 Research Freeze Rule）

現時点でStudy83実装・新アルファ探索・Route C再検討は全て時期尚早（§8A-2）。
Core CP3審査は急がない（Core weight=0%は仮定表上の最適解にすぎず生死は実測でのみ判定可能）。
```

---

## §9 Research Status（現在地・2026-07-20 Study103実行後）

```
Production   : Entry Freeze
Core         : Intrinsic alpha UNKNOWN
               Observed PIT ≈0-5%（観測範囲・真値推定区間ではない）
               Confidence LOW
Universe     : Rebuilding（Universe-A確定 / Universe-B未生成）
Current Phase: **Ceiling Measurement Phase**（Program Phase3・Study82 Phase0.1起案完了・承認待ち）
Route Status : B=ACTIVE(Frontier Validation) / A=STANDBY / C=DORMANT / F=TERMINAL
30% Route    : DORMANT（§8A-1 Route C再起動条件A/B/C成立まで）
Freeze Rule  : No new alpha impl/BT until Study82完了（§8A-2）
```

### §9A CP2判定確定（Study103実行結果・2026-07-20）

```
CP2 = RED（Tier3=30%/Calmar1.5・Base不成立27.7%<30%・Optimisticは自動RED境界抵触で実質不成立）
Tier2 = GREEN（20-25%/Calmar1.3・Base到達27.7%）
Tier1 = GREEN（10-18%/Calmar1.0・Conservativeでも部分成立17.6%）
```

**§1A決定木の適用（機械的・裁量なし）**: `CP2 RED → Tier2 feasible? YES → Route B`。
**→ Route B（Core+PEAD+TSMOM・目標20-25%/Calmar1.3-1.5）が正式起動**。

**副次的所見（弱い証拠・Falsification原則3の対象）**: Core Retirement Probability=100%
（全水準でCoreの最適配分重み0%）。これはCore CAGR仮定がPIT観測値（Base3%）で既に低く
織り込まれていることの当然の帰結であり、Core CP3審査（実測）の結果を先取りしない。
詳細→`reports/study103_portfolio_feasibility.md`。

**次アクション**: Study82（PEAD発表日時精度監査・第一関門）・Study83（TSMOM CP3）が次点。
新規スクリプトにつき個別ASK_FIRST。

---

## §10 戦略優先順位（2026-07-20 CP2確定後・更新）

```
第1優先: Study82（PEAD発表日時精度監査・第一関門）— Route B頂点の前提
第2優先: Study83（TSMOM CP3）— Route Bもう一方の構成要素・実装最易
第3優先: Study75 completion（Universe-B生成）— Study76/E1分岐に必要
第4優先: Study76（複雑性判定・E1分岐点）

Study80（MN）は本MCで最適配分にほぼ非登場（Conservativeでのみ出現）— 優先度は暫定順位表の
3位から維持しつつ、Study82/83完了後に再評価。Study102（ARCH-E）はRoute D=Dormantのまま。
```

（~~v1.1-v1.4の「75→76→103」順~~ → v1.5.1「103最優先」 → **v1.5.1確定後は結果に従いCP2完了・
Route B起動。Study103自体は完了済みのため優先順位から外れ、次点のStudy82/83へ移行**。）

**Satellite暫定順位（CP2後にfrontierで再確定・v1.5.1）**:
```
1 TSMOM (Study83)        — データ量最大・実装容易・低相関・Study95射程外・容量問題なし
2 PEAD (Study82)         — 文献強い・情報源独立・Satellite適性高
3 MN (Study80)           — Study95によりprior低下（1-3M短期スプレッドのみ生存）・実現はStudy86重量割引
4 SmallGrowth (Study102=ARCH-E) — Dormant（執行不能・容量・survivorship感度。CP2 GREEN以外で優先度なし）
5 Lead-Lag (Study84)     — Route構成外・killテストのみ
```
（注: ユーザー原案の「3 Study102 / 4 ARCH-E」は同一対象の重複列挙のためMN(Study80)を3位に補正。
Study95でMN期待値が低下した結果、PEAD/TSMOMの相対順位が上がったという原案の論旨はそのまま反映。）

---

## §V 改版履歴

| 版 | 日付 | 内容 |
|---|---|---|
| v1.0-v1.4 | 2026-07-19/20 | `roadmap_revision_2026-07-19.md`§改版履歴参照（HISTORY） |
| **v1.5.4** | 2026-07-20 | **Route状態マトリクス正式化**（B=ACTIVE/Frontier Validation Phase・A=STANDBY削除禁止・C=DORMANT・F=TERMINAL、各起動/再起動条件を表で固定）/ **Research Freeze Rule新設**（Study82完了までStudy83実装含む新規alpha実装/BT一切禁止・Proposal文書のみ許容・study83_proposalの並行着手選択肢を上書き凍結）/ **CP3ケース分岐**（Case1=PEAD PASS∧TSMOM GOOD→Route B confirmed / Case2=PEAD FAIL→Route A promotion / Case3=PEAD PASS∧TSMOM超過→Route C reactivation review）/ 研究目的の正式表現確定 / **Program Phase0-5**マクロ工程表新設（0-2完了・3=現在地・4=CP3・5=最終状態決定）/ **Study82をPhase0.1（API疎通確認）とPhase0（日時監査）に分割**・出力にUNKNOWN追加 / 優先順位5段固定（0.1→0→Viability Review→83Proposal→83実装） |
| **v1.5.3** | 2026-07-20 | **Post-CP2 Focus確定**: フェーズ宣言（Primary=Route B ceiling測定・Secondary=Route C再起動条件監視）/ **Route C=DORMANT**（閉鎖ではない・再起動条件A(TSMOM実測>Base+5pp)/B(PEAD実測>Base+5pp)/C(新規スリーブCP3通過)のOR・これ以外の再審議禁止）/ **Phase A-E固定**（82 Phase0→83 Proposal→83実装→82 Alpha Study→Route B Ceiling再推定）/ 将来CP3イメージ告知（GREEN=正式運用/YELLOW=Satellite縮小/RED=A or F）/ 閉鎖対象明記（30%研究・Capital Route・無制限探索・新Route乱立・Route C延命）/ 優先順位=Study82 Phase0→83 Proposal→83実装→Core CP3(急がず) |
| **v1.5.2** | 2026-07-20 | **Study103実行完了・CP2確定**: `src/backtest/study103_portfolio_feasibility.py`実装+fresh run（6シナリオ・N=20,000精査MC）。**CP2=RED**（Tier3=30%/1.5・Base到達27.7%で僅差不成立・Optimisticは自動RED境界=avg Calmar>2.0に抵触し実質不成立）。Tier2=GREEN・Tier1=GREEN。§1A決定木を機械適用し**Route B正式起動**（Core+PEAD+TSMOM・20-25%/1.3-1.5）。副次所見: Core Retirement Probability=100%（弱い証拠として記録・Falsification原則3）。優先順位をStudy82(PEAD監査)→Study83(TSMOM)へ更新。詳細→`reports/study103_portfolio_feasibility.md` |
| **v1.5.1** | 2026-07-20 | Study103実装直前の事前固定（ユーザー承認）: **§1A CP2後の行動決定木**（GREEN→76/YELLOW→Satellite最大2本/RED→Tier2?→B・Tier1?→A・否→F — 事後議論禁止）/ **Planning Priorをordinal表現へ**（LOW/MEDIUM/MEDIUM-HIGH・数値は`Illustrative only`注記必須）/ **Route F Trigger E追加**（Infrastructure burden > Expected research value・ユーザー認定制）/ Study103成果物#7=**Research Continuation Policy**へ改称（Terminal state recommendation含む）/ Satellite暫定順位表（TSMOM→PEAD→MN→SG(Dormant)→Lead-Lag・原案の102/ARCH-E重複をMN=3位に補正）/ **Study103実装承認**（ユーザータスク指示による・ASK_FIRST充足） |
| **v1.5** | 2026-07-20 | 統合正典化（本書新設・Registry設置・アンチ乱立規則）/ Study103目的の二層化（Primary=反証・Secondary=天井と終端の確定）/ **Tier番号を昇順体系へ反転**（Tier3=30%…Tier0=市場リターン・対照表§2）/ Route D=**Dormant High Alpha Branch**（Study102 WHITEまで独立Route化禁止・工数ゼロ）/ **Route F発動条件4トリガー正式化**（既存2四半期Killとの二段階整合）/ Planning Prior正式規律（ordinal only・CP2/CP3引用禁止）/ **Research Budget新設**（工数配分35/45/10/10・active routes≤2・concurrent studies≤2）/ Route Transition Matrix正式図 / **Study103成果物7点固定**（Termination/Core retirement probabilityは機械的定義）/ 優先順位=103最優先へ / 結語追加 / `final_research_roadmap_2026-07-04.md`をSUPERSEDEDへ格下げ |

*作成: CLD (Fable 5)・2026-07-20。BT・コード変更・新規仮説生成なし。*
