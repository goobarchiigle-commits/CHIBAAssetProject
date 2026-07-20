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

| Route | 名称 | 目標 | 状態・条件 | prior |
|---|---|---|---|---|
| **C** | 30% Frontier | Tier3 | CP2 GREEN ∧ 複数スリーブCP3通過 ∧ Study86執行parity→Study85統合（全AND） | **LOW**（~5-10%） |
| **B** | Diversified Alpha | Tier2 | Core+PEAD+TSMOM。**現時点の主計画（Plan B）** | ~20-25% |
| **A** | Conservative Alpha | Tier1 | TSMOM主軸。**Study103 RED時の既定避難先（Plan C）** | ~30% |
| **D** | **Dormant High Alpha Branch** | overlay | **休眠管理（v1.5確定）**: 独立Route化はStudy102 WHITE（旧称Study81 — 現採番はStudy102・v1.0改番済み）まで禁止。PIT証拠不足のため起動条件成立まで工数配分ゼロ | 起動後に再評価 |
| **E1** | Core Replacement | — | Study76 WHITE→Clenow置換。fujiko_r2 Candidate A統合先（二重実装禁止） | ~40-50% |
| **E2** | Core Retirement | — | Study103 Case B優位 or Core CP3 fail→Satellite-only正式選択肢 | ~40-55% |
| **F** | Terminal | Tier0 | §4トリガー成立時の正式着地 | ~20-30% |

---

## §4 Route F 発動条件（正式版・事前固定）

```
Trigger A: CP2 RED ∧ Tier1もinfeasible（Goal Ladder Sweepで判定）
Trigger B: 全候補スリーブCP3不通過
Trigger C: Research budget exhausted（§6予算の枯渇・ユーザー認定）
Trigger D: 連続2年次サイクルで採用スリーブゼロ
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
1. Tier0-3 feasibility        （Goal Ladder Sweep・GREEN/YELLOW/RED×4Tier）
2. Goal frontier              （Calmar制約別の最大到達CAGR曲線・主図表）
3. Route transition matrix    （§7の各遷移がどのシナリオで発火するかの対応表）
4. Termination probability    （定義: 全シナリオ×MC試行中、Tier1すらinfeasibleとなる割合）
5. Core retirement probability（定義: Case B frontierがCase Aより緩いシナリオの割合）
6. Budget recommendation      （§6配分の改定提案・advisory・採用はユーザー決裁）
7. Terminal state recommendation（Route F前倒しの要否判定・advisory）
```

4・5は上記の**機械的定義に固定**（実装時の再解釈禁止）。6・7はadvisory出力であり、ゲート判定（1-3）と混同しないこと。

---

## §9 Research Status（現在地）

```
Production   : Entry Freeze
Core         : Intrinsic alpha UNKNOWN
               Observed PIT ≈0-5%（観測範囲・真値推定区間ではない）
               Confidence LOW
Universe     : Rebuilding（Universe-A確定 / Universe-B未生成）
Current Phase: Study75 + Study103
30% Route    : Pending CP2
```

---

## §10 戦略優先順位（現行）

```
第1優先: Study103 implementation（ASK_FIRST待ち・目的=realistic ceiling and terminal statesの確定）
第2優先: Study75 completion（Universe-B生成）
第3優先: Study76（複雑性判定・E1分岐点）
第4優先: Satellite reprioritization（CP2後・frontierに従い83/80/82を再順位付け）

Study83/80/82はまだ掘らない（CP2前のSatellite着手は§6予算違反）。
```

（v1.1-v1.4の「75→76→103」順から**103最優先へ変更**。理由: 103は依存ゼロ・最安・かつCP2結果が第2優先以下の全配分を規定するため、待たせる合理性がない。）

---

## §V 改版履歴

| 版 | 日付 | 内容 |
|---|---|---|
| v1.0-v1.4 | 2026-07-19/20 | `roadmap_revision_2026-07-19.md`§改版履歴参照（HISTORY） |
| **v1.5** | 2026-07-20 | 統合正典化（本書新設・Registry設置・アンチ乱立規則）/ Study103目的の二層化（Primary=反証・Secondary=天井と終端の確定）/ **Tier番号を昇順体系へ反転**（Tier3=30%…Tier0=市場リターン・対照表§2）/ Route D=**Dormant High Alpha Branch**（Study102 WHITEまで独立Route化禁止・工数ゼロ）/ **Route F発動条件4トリガー正式化**（既存2四半期Killとの二段階整合）/ Planning Prior正式規律（ordinal only・CP2/CP3引用禁止）/ **Research Budget新設**（工数配分35/45/10/10・active routes≤2・concurrent studies≤2）/ Route Transition Matrix正式図 / **Study103成果物7点固定**（Termination/Core retirement probabilityは機械的定義）/ 優先順位=103最優先へ / 結語追加 / `final_research_roadmap_2026-07-04.md`をSUPERSEDEDへ格下げ |

*作成: CLD (Fable 5)・2026-07-20。BT・コード変更・新規仮説生成なし。*
