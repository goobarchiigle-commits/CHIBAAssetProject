# Roadmap Revision 2026-07-19 — Post-Study74 + Study90番台反映・complete_execution_roadmap 全面改定

**日付**: 2026-07-19
**性格**: 文書改定のみ。新規バックテストゼロ・コード変更ゼロ・実弾変更ゼロ。
**改定対象**: `reports/complete_execution_roadmap_2026-07-04.md`（以下「旧正典実行手順書」）
**ファイル名注記**: ユーザー指示原文は `roadmap_revision_2026-07-17.md` だが、SAVE規約（実日付）により 2026-07-19 を採用。
**拘束エビデンス**:
1. `reports/study74_final_review.md` — CP1 BLACK・資本経路上限QED
2. `reports/study75c_interpretation.md` — 選定バイアス+12.26pp / 生存者バイアス≈0
3. `reports/study95_cs_momentum_factor_level.md` — CS momentum factor-level KILL
4. `reports/study99_sector_fujiko_filter.md` — Sector×Fujiko factor-levelエッジ未確認
5. `reports/study100_legacy_fujiko_universe_audit.md` — U-1静的RSR42名簿=FATAL（hindsight選定+11-12pp）
6. `reports/study101_legacy_fujiko_true_expectancy.md` — PITユニバース上の旧フジコ法=全4構成RED・TOPIX劣後
7. `reports/fujiko_r2_research_roadmap.md` v2 — Study87-97採番・FUJIKO 2.0統治原則
8. `reports/entry_freeze_mode_2026-07-17.md` — 新規BUY全面停止中（現在有効）

---

## §0 改定サマリ（ユーザー指示10項 対応表）

| # | 指示 | 処置 | 本書節 |
|---|---|---|---|
| 1 | Study74 = FINAL BLACK | 確定宣言。「決裁待ち」表記を終了 | §1 |
| 2 | 資本スケーリング単独で18-22%到達の前提を全削除 | 旧CP1白分岐を恒久無効化 | §1 |
| 3 | Study79恒久CLOSE | STATUS=CLOSED（前提Study74 WHITE永久不成立） | §2 |
| 4 | CP1目標改定 | Core 10-15% / Stretch 15-18% / 18%+は新アルファ必須 | §3 |
| 5 | Universe二層分離 | Universe-A（バイアス測定）/ Universe-B（TOPIX500 PIT・Study76用） | §4 |
| 6 | Study76目的の明文化 | 複雑性の正当化判定（高CAGR追求ではない） | §5 |
| 7 | Stage5書き換え | 30%到達=複数低相関Satellite統合のみ。Core単独30%前提削除 | §6 |
| 8 | Study81採番衝突解消 | Study81=Cluster RCA（完了）確定。Small Growth Momentum=**Study102**新規採番 | §7 |
| 9 | 依存グラフ・タイムライン・決定木・正典目標の更新 | 新版を§9に収録・旧正典実行手順書へ反映 | §9 |
| 10 | 本改定レポート作成 | 本書 | — |

**追加**: Appendix「Post-Study74 Reality Check」（§10）・Study90番台整合注記（§8）。

---

## §1 修正① CP1分岐 — Study74 = FINAL BLACK（確定）

**旧記述（無効化）**:
```
Study74白 → 入金 → Core 18-22% → Study79レバ → 30%再挑戦
```

**確定事実（Study74 QED・`reports/study74_final_review.md`）**:
- 資本投入単独のCAGR改善余地 ≈ **+1.12pp**（lot丸め解消分のみ・¥20Mで完全解消・¥30Mで後退）
- max_positions=3は資本非依存の構造的天井（緩和するとCAGR悪化）
- ¥20-30M投入でも22%到達は**不能**。正典ゲート「CAGR≥22% ∧ WF5/5」全8セル未達
- Study75B/75Cが a fortiori で追認（廃止銘柄トレードの系統的損失・PIT期待値median IS -2.48%）

**帰結**: 「資本投入 → Core 18-22%」という分岐は**ロードマップから恒久削除**。CP1の「白」分岐は存在しない。以後、資本増による目標到達の再提案は恒久閉鎖14項と同格の禁止事項とする（§10 Reality Check参照）。

---

## §2 修正③ Study79（L7）— STATUS = CLOSED（恒久）

```
L7 Study79: 資本投入+レバ1.3x設計
STATUS = CLOSED（2026-07-19恒久確定）

理由: 起案条件「Study74白 ∧ Study78合格 ∧ CAND_B移行済」のうち
      Study74 WHITE が永久に成立不能（§1）。
      3条件AND型の起案条件は1条件の恒久不成立で恒久CLOSED。
```

- Study78の合格側材料（レバ1.3x RoR=1.02%<5%）は**無駄にならない** — レバ研究は**Study80 WHITE後の統合ポートフォリオ側（Study85/86）へ移管**。β≈0のMNスプレッドに対するレバのみがCalmar制約と両立する（alternative_architectures_5x §総括の結論のまま）。
- Long Only Coreへのレバは、いかなる資本水準でも再起案禁止（DD比例拡大でCalmar1.5を破る構造は資本非依存）。

---

## §3 修正② 新CP1目標 — Core単独期待値の公式改定

**旧目標（削除）**: 30% ∧ Calmar1.5（Core単独・資本投入前提）

**新CP1（Core単独期待値）**:

| 区分 | CAGR | 備考 |
|---|---|---|
| Core単独期待レンジ | **10〜15%** | RSR42名目実績ベース（M1後公式値 IS 12.22% / OOS 11.42% / Full 11.22%）|
| Stretch | **15〜18%** | Exit/構成純化の残余改善が全て実現した場合の上限 |
| 18%以上 | **新アルファ源必須** | MN / PEAD / TSMOM。Core改良では到達不能 |

**⚠ Study100/101による追加補正（本改定で明記必須）**:
上表の10-15%は**RSR42名目実績**に基づく。Study100（U-1名簿=hindsight選定FATAL・選定バイアス+11-12pp）・Study101（PITユニバース上の旧フジコ法=全4構成RED・TOPIX全面劣後）により、**PIT補正後のCore固有アルファは≈0-5%（TOPIX劣後の可能性含む）**。すなわち:
- 10-15%は「現行名簿の残存寿命の間の名目期待値」であり、**戦略の複製可能な実力ではない**
- 現在Entry Freeze Mode有効（新規BUY全面停止・2026-07-17〜）。Freeze解除判断は本目標レンジではなくStudy100条件付きYES（¥3M限定・増資禁止）+FUJIKO 2.0研究結果に従う
- ベンチマークはTOPIX B&Hに一本化（Study101判定）

---

## §4 修正③' Universe二層分離 — Study52型交絡の再演防止

旧正典実行手順書のUniverse統制ポリシー（2026-07-04決裁「全比較対象はStudy75 Universe上でfresh run」）は方向として正しいが、「Study75 Universe」が単一名（Universe C≈2,282銘柄規模）に収斂すると**Universe差とArchitecture差が再交絡**する。以下の二層に恒久分離:

| 層 | 定義 | 目的 | 既存資産との対応 |
|---|---|---|---|
| **Universe-A**（Current Capital Universe） | ADV20≥¥300M ∧ lot feasible（月次PIT） | **バイアス測定**（Study75系・selection/survivorship分解） | = Study75A「Universe C」(`backtests/study75_rule_universe.json`) |
| **Universe-B**（TOPIX500 PIT） | TOPIX500構成のPIT復元（または ADV20 Top500近似・`study75_universe_design.md` D案） | **Study76 Clenow比較**（アーキテクチャ差の単離） | 未生成。Study75残作業（TOPIX500真値化 enrich・ASK_FIRST）が前提 |

**統制規則**:
1. Study76の全アーム（純正Clenow・D_ATR_EQ再測定）は**Universe-Bのみ**で実行。Universe-A上の値との比較禁止。
2. バイアス幅の引用は**Universe-A系測定値のみ**（Study75B/75C）。Universe-Bでバイアス測定をやり直さない（目的外流用禁止）。
3. 1つのStudyで2ユニバースを混用した場合、そのStudyは無効（Study52汚染事件と同型の事故として扱う）。

---

## §5 修正④ Study76 成功基準 — 目的の明文化

- 成功条件 **Δ ≥ -2pp** は維持（変更なし）。
- **目的追記（正典文言）**: 「Study76の目的は高CAGR追求ではなく**複雑性の正当化判定**である。Study74後のD_ATR_EQ公式値（Full 11.22%・名目）に対し、純正Clenowが例えば10%でも、レジーム5機構・Exit7系統・boost群を削除できるなら**純正側が勝ち**。判定軸はCAGR差ではなく『複雑性1単位あたりの対価』」
- Study95申し送り: Clenow slope×R²は1-6M horizonで正・12Mで反転・Bear regimeで有意悪化（factor-level）。Study76設計の回転周期・レジームゲートはこの実測を前提に事前固定すること（事後調整は閉鎖領域入り）。

---

## §6 修正⑤ Stage5書き換え — 30%の唯一の経路はポートフォリオ統合

**旧構造（削除）**: 「Study80失敗 → 30%終了」かつ「Core(18-22%)を土台に統合」

**新定義（正典文言）**:
```
30%達成条件: 複数低相関Satelliteの統合のみ。
Core単独30%という前提はロードマップ全体から削除（Study74 QEDにより恒久不成立）。

30%経路 = Study80(MN) / PEAD(Study82) / TSMOM(Study83) の
複数WHITE ∧ 低相関(<0.5) ∧ Study85統合でのみ数学的に成立し得る。
Coreの役割 = 統合ポートフォリオの土台スリーブ（10-15%・名目）であり、
30%のエンジンではない。
```

- Study80はもはや「30%再挑戦の追加根拠」ではなく**30%経路の必要条件の一つ**。CP2の意味を「MNスプレッドαの実在確認 = 30%経路が生存しているか否かの判定」に格上げ。
- Study80/82/83が全てBLACKの場合、30%は**最終棄却**し、その時点の実測値（Core名目10-15%またはFUJIKO 2.0後継値）で運用フェーズ移行。
- **リスク注記（Study95申し送り）**: Universe-A上の12-1モメンタムは弱い逆転（過去勝者が6M/12Mで最下位）。Study80のL/Sスプレッド設計はLong側単独の脆弱性を前提に、Short側寄与の単離測定（正典既定の「ショート側寄与<2pp=失敗」）を厳守。

---

## §7 修正⑥ Study採番衝突の解消

**衝突**: 旧正典実行手順書 L6「Study81 = ARCH-E 小型グロース・モメンタム」（未実行）と、2026-07-04実施済み「Study81 = Cluster Diversification Hypothesis（L1D・REJECT完了）」が同番号。

**解消（本改定で確定）**:

| 番号 | 内容 | 状態 |
|---|---|---|
| **Study81** | Cluster Diversification Hypothesis（RCA） | **完了（REJECT）**・2026-07-04・`reports/study81.md`。この採番で確定 |
| **Study102** | Small Growth Momentum（旧L6・ARCH-E） | 未実行・新規採番 |

**採番根拠**: ユーザー原案はStudy87だったが、**Study87-97は`fujiko_r2_research_roadmap.md` v2が予約済み**（87=Warm-up修正版ユニバース生成器 / 88=セクター持続性[実施済み=Study98として実現] / 89-94=FUJIKO 2.0系 / 95=CS momentum[完了] / 96=Entry/Exit帰属分解 / 97=Sector ETF実現性）、**Study98-101は実施済み**。無衝突の最小番号=**Study102**。タスク正文「Assign a new ID」に従い102を割当（新たな衝突の再演を防止）。

Study102の起案条件（旧L6から継承・変更なし）: Study75廃止込みデータ必須・スリッページ0.5%・値幅制限モデル込みCAGR≥20% ∧ WF5/5・採用時はSatellite 10-20%配分から・Calmar1.5と両立しない経路であることを提案時必ず明記。

---

## §8 修正⑦ 研究優先順位（ROI順）+ Study90番台との整合

**新優先順位（正典）**:

| 順位 | Study | 理由 |
|---|---|---|
| 1 | **Study75完了**（残作業: TOPIX500真値化 enrich → Universe-B生成） | 全下流Studyの前提。1投資6用途 |
| 2 | **Study76**（Clenow純正・Universe-B） | 複雑性判定。FUJIKO 2.0 Candidate Aの基準器を兼ねる |
| 3 | **Study80 feasibility**（MNスプレッド・純データ分析） | 30%経路の生死判定（CP2）。未探索・実装不要・情報価値最大 |
| 4 | **Study83**（指数TSMOM・先物） | データ独立・随時並行可・Core相関<0.5候補・未探索 |
| 5 | **Study77**（Exit構造置換） | Exit改善余地は既に大半閉鎖済み（Study61-69+閉鎖14項）。ROI最低・Study76決着後のみ |

**旧順序（75→76→77）からの変更理由**: Exit系の期待情報価値は閉鎖済み領域に隣接し限界的。MN/TSMOMは未探索かつ30%経路の必要条件であり、探索価値が構造的に高い。

**Study90番台（FUJIKO 2.0系）との関係**:
- 本改定は**旧正典（Study74-86系）の改定**であり、`fujiko_r2_research_roadmap.md`（Study87-97系）を上書きしない。両者は並立し、衝突時はユーザー決裁。
- Study95 KILL（CS momentum FAIL）の既定帰結「Candidate A-E凍結 → PEAD/TSMOM転進」は**本改定の優先順位（Study80/83が上位）と整合** — 転進先が旧正典側のARCH-A(MN)/ARCH-C(TSMOM)/PEAD であり、本改定はその受け皿を正式化するもの。
- Study99/98の結果（sector×fujiko factor-levelエッジ未確認・サブ期間符号反転）により、FUJIKO 2.0 Candidate B/C系の事前確率は低下側。これも順位3-4（MN/TSMOM先行）を支持。
- Entry Freeze解除判断は本優先順位と独立（ASK_FIRST・Study100条件付きYESの枠内）。

---

## §9 新・依存グラフ / タイムライン / 決定木（正典）

### 依存グラフ

```
Study75完了(Universe-A確定済み + Universe-B生成)
   ├→ Study76 (Universe-B・Clenow複雑性判定) ─→ Study77 (76決着後のみ・ROI最低)
   ├→ Study80 feasibility (MNスプレッド・貸借データ)  ← CP2: 30%経路の生死
   └→ Study102 (Small Growth・廃止込みデータ必須)
Study83 (TSMOM・データ独立・随時並行可)
Study82 (PEAD・発表日時精度監査が第一関門)
Study80/82/83のWHITE群 ─→ Study86(80白のみ・ショート執行) ─→ Study85(統合・CP4)
[CLOSED] Study79 — 恒久閉鎖(§2)
[並立] FUJIKO 2.0系 Study87-97 — fujiko_r2_research_roadmap.md管轄・Study95 KILL帰結の決裁待ち
```

### タイムライン（依存順）

| 時期 | タスク | ゲート |
|---|---|---|
| 2026 Q3 | Study75残作業（Universe-B生成）/ Study80 feasibility（並行可・純データ分析） | **CP2前哨: MN実在** |
| 2026 Q3-Q4 | Study76（Universe-B上・複雑性判定） | Clenow判定 |
| 2026 Q4 | Study83（並行）/ Study82 発表日時精度監査 | TSMOM/PEAD生死 |
| 2027 H1 | 生存Satellite群のWF本測定 / Study77（76決着後・任意） | CP3 |
| 2027 H2〜 | Study86（80白のみ）→ Study85統合 | **CP4: 30%最終判定（統合のみ）** |

### 決定木

```
Study74 ─BLACK確定→ Core=10-15%(名目)スリーブ固定・資本経路恒久閉鎖・Study79 CLOSED
Study80 ─黒→ MN経路閉鎖。82/83も黒なら30%最終棄却→実測値で運用フェーズ
   └白→ Study86→Study85統合へ（レバはここでのみ検討）
Study82/83 ─白→ Satelliteスリーブ候補としてStudy85へ
Study76 ─黒(<-4pp)→ 複雑性は正当化・現行構成維持
   └白→ 多層機構削減の根拠成立（FUJIKO 2.0 Candidate A基準器と共用）
Study102 ─白→ 「20%+ vs DD30-50%」トレードオフをユーザー明示決裁後Satellite 10-20%
30%判定 = Study85統合実測のみ（結合CAGR≥30% ∧ 結合MaxDD≤20% ∧ RoR<1%）
どの時点でも: 連続2四半期採用ゼロ → 運用フェーズ縮退
```

### 正典目標（本改定後の公式値）

| 項目 | 値 |
|---|---|
| Core単独期待 | CAGR 10-15%（名目・RSR42残存前提）/ PIT補正後≈0-5%（§3注記） |
| Core Stretch | 15-18%（新アルファなしの理論上限） |
| 30% ∧ Calmar1.5 | **Study85統合実測でのみ判定**。Core単独・資本投入・Coreレバでの到達経路は全て恒久閉鎖 |
| フォールバック | Satellite全滅時: その時点の実測値で目標確定・運用フェーズ移行 |

---

## §10 Appendix: Post-Study74 Reality Check（恒久参照用）

**目的**: 「資金を増やせば30%行けるのでは？」という議論への恒久的な回答。以後この議論が再提起された場合、本Appendixの引用のみで却下する。

| 項目 | 値 | 根拠 |
|---|---|---|
| Core intrinsic alpha（名目） | **≈10-12%** | M1後公式値（IS 12.22% / OOS 11.42% / Full 11.22%） |
| Core intrinsic alpha（PIT補正後） | **≈0-5%（TOPIX劣後リスク含む）** | Study75C（選定バイアス+12.26pp）・Study100（FATAL）・Study101（全構成RED） |
| Capital scaling上限 | **≈+1pp**（+1.12pp・lot丸め解消のみ） | Study74 Part A waterfall・QED |
| レバによる上乗せ（Long Only Core） | **経路なし**（DD比例拡大でCalmar崩壊） | Study79 CLOSED（§2）・alternative_architectures_5x |
| 30%までの残ギャップ | **≈+18pp（名目基準）/ PIT補正後は+25pp超** | 上記の差分 |
| 結論 | **ギャップは資本・レバ・Core改良のいずれでも埋まらない。新規直交アルファ源（MN/PEAD/TSMOM）の複数WHITE+統合のみが唯一の経路** | §6 |

**今後の研究優先順位（恒久掲示）**:
```
1 Study75完了（Universe-B生成）
2 Study76（複雑性判定）
3 Study80 feasibility（MN・CP2）
4 Study83（TSMOM）
5 Portfolio Integration（Study85系）
   — Study77はROI最低・任意
```

**禁止再提案リスト（本改定で追加）**:
- 資本増額による目標到達（Study74 QED）
- Long Only Coreへのレバ（Study79 CLOSED）
- Core単独30%目標の復活（§6）
- RSR42名目実績を将来判断の根拠に使う行為（Study101で消滅）

---

## §11 反映先・整合性

| ファイル | 処置 |
|---|---|
| `reports/complete_execution_roadmap_2026-07-04.md` | 冒頭に改定告知＋§0.7/Section1/L6/L7/Section8へ改定注記（旧文は削除せず取り消し線・凍結保持） |
| `src/research_state.md` | 先頭セクションに本改定の記録を追記 |
| `docs/research/2026-07-19.md` | 日次ログ |
| `reports/fujiko_r2_research_roadmap.md` | **無改変**（並立正典・Study95 KILL帰結はユーザー決裁待ちのまま） |
| PARAMS_LOCKED / strategy.yaml / 実弾運用 | **無変更**（Entry Freeze継続・本改定は文書のみ） |

**本改定が変更しないもの**: 恒久閉鎖14項・0.4採用ゲート・fresh run原則・ASK_FIRST体系・Entry Freeze状態・FUJIKO 2.0未決5点（ユーザー決裁待ち継続）。

*作成: CLD (Fable 5)・2026-07-19。新規バックテスト実行なし。*
