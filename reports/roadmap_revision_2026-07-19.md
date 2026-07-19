# Roadmap Revision 2026-07-19 — Post-Study74 + Study90番台反映・complete_execution_roadmap 全面改定

**日付**: 2026-07-19
**版**: v1.1（同日改定・ユーザーレビュー反映）
**性格**: 文書改定のみ。新規バックテストゼロ・コード変更ゼロ・実弾変更ゼロ。

**改版履歴**:
| 版 | 日付 | 内容 |
|---|---|---|
| v1.0 | 2026-07-19 | 初版（ユーザー指示10項+Appendix） |
| v1.1 | 2026-07-19 | ユーザーレビュー反映: ①Core PIT期待値=**UNKNOWN（~0-5%・confidence LOW）**へ格下げ・Stretch表記失効 ②優先順位83⇄80入替（TSMOM先行） ③CP2再定義=**Study103**（統合の数学的成立性）+CP3スリーブ採用ゲート事前固定 ④Study77=条件付き凍結（76 BLACK時のみoptional） ⑤**Study103新設**（Portfolio Architecture Feasibility） ⑥Core=Study85保証枠（80-90%）廃止・スリーブ候補の1つへ格下げ |
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
| 4 | CP1目標改定 | ~~Core 10-15% / Stretch 15-18%~~ **→v1.1: Core PIT=UNKNOWN（~0-5%・LOW）・名目10-15%は歴史参考のみ** | §3 |
| 5 | Universe二層分離 | Universe-A（バイアス測定）/ Universe-B（TOPIX500 PIT・Study76用） | §4 |
| 6 | Study76目的の明文化 | 複雑性の正当化判定（高CAGR追求ではない） | §5 |
| 7 | Stage5書き換え | 30%到達=複数低相関Satellite統合のみ。Core単独30%前提削除 | §6 |
| 8 | Study81採番衝突解消 | Study81=Cluster RCA（完了）確定。Small Growth Momentum=**Study102**新規採番 | §7 |
| 9 | 依存グラフ・タイムライン・決定木・正典目標の更新 | 新版を§9に収録・旧正典実行手順書へ反映 | §9 |
| 10 | 本改定レポート作成 | 本書 | — |

**追加**: Appendix「Post-Study74 Reality Check」（§10）・Study90番台整合注記（§8）。
**v1.1追加**: CP2/CP3再定義（§6）・**Study103新設**（§6A）・優先順位改定 75→76→103→83→80→82→85→(77 optional)（§8）・Coreスリーブ格下げ（§3）。

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

**新CP1（Core単独期待値・v1.1改定）**:

| 区分 | 値 | 備考 |
|---|---|---|
| **Core intrinsic alpha（PIT・公式表記）** | **UNKNOWN（~0-5%・confidence LOW）** | **α≈0（TOPIX劣後）の可能性を排除できない**。Study101全4構成RED・Study75C PIT無作為median IS -2.48%。0-5%自体が推定にすぎない |
| 名目参考値（RSR42） | 10-15% | 歴史記録としてのみ保持（M1後公式値 IS 12.22% / OOS 11.42% / Full 11.22%）。**意思決定根拠使用禁止**（Study100 FATAL・名簿残存期間の名目値） |
| ~~Stretch 15-18%~~ | **失効（v1.1）** | 名目実績前提のため定義不能。純化後PIT再測定（Study76 / FUJIKO 2.0）完了後に再定義 |
| 18%以上（旧表記の帰結） | 新アルファ源必須（不変） | MN / PEAD / TSMOM。Core改良では到達不能 |

**表記禁止則（v1.1）**: 「Coreには5%程度のアルファが残る」と読める記述を禁止。公式表記は常に `UNKNOWN (~0-5%, confidence LOW)`。

**Coreの位置づけ格下げ（v1.1・⑥）**: CoreはStudy85統合における**保証スリーブではない**。旧「Core 80-90% + Satellite 10-20%」構成を廃止し、CoreはPIT再測定値で**CP3スリーブ採用ゲート（§6）を他Satelliteと同条件で通過した場合のみ**採用。通過不能なら**Satellite-onlyポートフォリオを正式な選択肢とする**。

**運用整合**:
- Entry Freeze Mode有効（新規BUY全面停止・2026-07-17〜）。Freeze解除判断はStudy100条件付きYES（¥3M限定・増資禁止）+FUJIKO 2.0研究結果に従う
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

**チェックポイント再構成（v1.1・③）**: 旧「CP2=Study80のMN実在確認」は不十分（単一スリーブの生死と統合の成立性は別問題。例: MN CAGR40%/DD-45%は単独で存在しても結合Calmar1.5に寄与するとは限らない）。以下に再定義:

| CP | 定義 | 判定Study |
|---|---|---|
| **CP2（新）** | **統合の数学的成立性**: 現実的なスリーブ仮定（CAGR/Vol/Corr）の下で「結合30% ∧ Calmar1.5 ∧ RoR<1%」が可能か | **Study103**（§6A・Satellite掘削前に先行判定） |
| **CP3（新）** | **スリーブ採用ゲート**（事前固定・裁量禁止・各Satellite個別判定） | Study80/82/83/102 + Core再測定 |
| CP4（不変） | 統合実測: 結合CAGR≥30% ∧ 結合MaxDD≤20% ∧ RoR<1% | Study85 |

**CP3スリーブ採用ゲート（事前固定・v1.1）**:
```
採用 ⇔ 単独CAGR≥15%（コスト後） ∧ 単独Calmar≥0.8
     ∧ |Core・既当選スリーブとの相関|<0.5
     ∧ WF5/5 ∧ コストストレス生存（0.4ゲート準拠）
注1: 低volスリーブ（先物TSMOM等）のCAGR≥15%判定は、RoR<1%∧証拠金/DD予算内の
     vol-targetレバ換算後の値で行う（生CAGRでの機械的却下を禁止 — レバ適性はMN/TSMOMの本質）
注2: 30%への寄与だけを見る判定を禁止。Calmar床(0.8)と相関(<0.5)が先、CAGRが後
注3: CoreもPIT再測定値で本ゲートを同条件審査（§3・保証枠なし）
```

- Study80は「CP2判定者」から**MNスリーブ候補のCP3審査対象**へ位置づけ修正。Study80/82/83/102が全てCP3不通過の場合、30%は**最終棄却**し、その時点の実測値で運用フェーズ移行。
- **リスク注記（Study95申し送り）**: Universe-A上の12-1モメンタムは弱い逆転（過去勝者が6M/12Mで最下位）。Study80のL/Sスプレッド設計はLong側単独の脆弱性を前提に、Short側寄与の単離測定（正典既定の「ショート側寄与<2pp=失敗」）を厳守。

---

## §6A Study103 — Portfolio Architecture Feasibility【v1.1新設・CP2・依存ゼロ・即時着手可】

**目的**: 「30% ∧ Calmar1.5」が**数学的に可能か**を、MN/PEAD/TSMOMを掘る前に確定する。Study74の方法論（掘る前に上限を測る→BLACKなら経路ごと閉鎖）の統合層への再現。不可能ならStudy80以降の優先順位・目標自体が変わるため、Satellite研究より先に実行する価値がある。

**手法**（新規BT不要・データ取得不要・Study78 MC資産再利用・0.5-1日規模）:
1. スリーブ候補{Core, MN, PEAD, TSMOM, SmallGrowth}のCAGR/Vol/Core相関を**3水準（保守/基準/楽観）の事前固定表**で仮置き（外部文献+内部実測レンジ。表確定後の変更・追加は禁止）。
2. Monte Carlo（ブロックbootstrap・Study78方式）で結合CAGR分布・結合MaxDD・Calmar・RoRを算出。
3. **逆問題形式を主出力とする**: 「30%∧Calmar1.5∧RoR<1%」成立に必要な（スリーブ数, 単独Calmar, 相関上限, レバ）の**feasible frontier**を提示。
4. **禁止**: 仮定を動かして30%が出るまで探索する行為（single_metric_optimization=forbid・仮定スイープはStudy52型汚染と同格）。

**判定（事前固定）**:
- **楽観水準でも不成立** → 30%**最終棄却**（Study85を待たず）。Satellite研究は「Calmar改善・絶対リターン向上」へ目標再定義（ユーザー決裁）
- **基準水準で成立** → feasible frontierの必要条件をCP3ゲートの補強値として固定し、Study83/80/82へ進行
- **成立が脆弱**（相関±0.1・Calmar±0.2で反転等） → 感度レポート付きでユーザー決裁

**実装**: `src/backtest/study103_portfolio_feasibility.py`（新規スクリプト=**ASK_FIRST**）。出力: `backtests/study103_portfolio_feasibility_YYYY-MM-DD.json` + `reports/study103_portfolio_feasibility.md`。

**依存**: ゼロ（Study75/76の完了を待たない・並行可）。優先順位表では3位だが**待ち時間ゼロで先行着手可能**。

---

## §7 修正⑥ Study採番衝突の解消

**衝突**: 旧正典実行手順書 L6「Study81 = ARCH-E 小型グロース・モメンタム」（未実行）と、2026-07-04実施済み「Study81 = Cluster Diversification Hypothesis（L1D・REJECT完了）」が同番号。

**解消（本改定で確定）**:

| 番号 | 内容 | 状態 |
|---|---|---|
| **Study81** | Cluster Diversification Hypothesis（RCA） | **完了（REJECT）**・2026-07-04・`reports/study81.md`。この採番で確定 |
| **Study102** | Small Growth Momentum（旧L6・ARCH-E） | 未実行・新規採番 |
| **Study103** | Portfolio Architecture Feasibility（§6A・v1.1新設） | 未実行・新規採番・CP2判定者 |

**採番根拠**: ユーザー原案はStudy87だったが、**Study87-97は`fujiko_r2_research_roadmap.md` v2が予約済み**（87=Warm-up修正版ユニバース生成器 / 88=セクター持続性[実施済み=Study98として実現] / 89-94=FUJIKO 2.0系 / 95=CS momentum[完了] / 96=Entry/Exit帰属分解 / 97=Sector ETF実現性）、**Study98-101は実施済み**。無衝突の最小番号=**Study102**。タスク正文「Assign a new ID」に従い102を割当（新たな衝突の再演を防止）。

Study102の起案条件（旧L6から継承・変更なし）: Study75廃止込みデータ必須・スリッページ0.5%・値幅制限モデル込みCAGR≥20% ∧ WF5/5・採用時はSatellite 10-20%配分から・Calmar1.5と両立しない経路であることを提案時必ず明記。

---

## §8 修正⑦ 研究優先順位（ROI順）+ Study90番台との整合

**新優先順位（正典・v1.1改定）**:

| 順位 | Study | 理由 |
|---|---|---|
| 1 | **Study75完了**（残作業: TOPIX500真値化 enrich → Universe-B生成） | 全下流Studyの前提。1投資6用途 |
| 2 | **Study76**（Clenow純正・Universe-B） | 複雑性判定を数週間で決着可能。FUJIKO 2.0 Candidate Aの基準器を兼ねる |
| 3 | **Study103**（Portfolio Feasibility・CP2） | **依存ゼロ・0.5-1日・並行即時着手可**。不成立なら4位以下の順位/目標自体が変わるため、Satellite掘削前に必須 |
| 4 | **Study83**（指数TSMOM・先物） | 実装容易・データ独立・Core相関低い可能性高・レバ適性あり。ROI最高のSatellite候補 |
| 5 | **Study80 feasibility**（MNスプレッド・純データ分析） | feasibility自体は安価だが、白でも実現はStudy86（信用口座・借株コスト・在庫管理・執行インフラ）の重い実装確率で割引かれる → 83の後 |
| 6 | **Study82**（PEAD） | 発表日時精度監査が第一関門（FAILで即終了） |
| 7 | **Study85**（統合） | CP3当選スリーブ≥2で起案 |
| 8 | **Study77**（Exit構造置換） | **条件付き凍結（v1.1・④）**: Study76 WHITE→多層Exit機構ごと消滅し**起案不要**。Study76 BLACK→必要と判断した場合のみユーザー起案（optional）。優先順位リストの実質圏外 |

**変更理由（v1.0→v1.1）**: (a) Study83⇄80入替 — Study80の期待情報価値は「白判定後の実現確率（Study86の実装重量: 貸借データ・borrow fee・hard-to-borrow判定・ショート在庫管理）」で割り引く必要があり、実装容易なTSMOMが先。(b) Study103挿入 — 統合の数学的成立性が不明のままSatellite群を掘るのは、Study74以前に資本経路を掘っていたのと同じ誤り。(c) Study77は「76→77」の直列ではなく「76 BLACK時のみoptional」の分岐へ。

**Study90番台（FUJIKO 2.0系）との関係**:
- 本改定は**旧正典（Study74-86系）の改定**であり、`fujiko_r2_research_roadmap.md`（Study87-97系）を上書きしない。両者は並立し、衝突時はユーザー決裁。
- Study95 KILL（CS momentum FAIL）の既定帰結「Candidate A-E凍結 → PEAD/TSMOM転進」は**本改定の優先順位（Study80/83が上位）と整合** — 転進先が旧正典側のARCH-A(MN)/ARCH-C(TSMOM)/PEAD であり、本改定はその受け皿を正式化するもの。
- Study99/98の結果（sector×fujiko factor-levelエッジ未確認・サブ期間符号反転）により、FUJIKO 2.0 Candidate B/C系の事前確率は低下側。これも順位3-4（MN/TSMOM先行）を支持。
- Entry Freeze解除判断は本優先順位と独立（ASK_FIRST・Study100条件付きYESの枠内）。

---

## §9 新・依存グラフ / タイムライン / 決定木（正典）

### 依存グラフ（v1.1）

```
Study103 (依存ゼロ・即時着手可) ══ CP2: 30%∧Calmar1.5の数学的成立性
   ├─不成立→ 30%最終棄却（Satellite研究はCalmar改善目的へ再定義・ユーザー決裁）
   └─成立→ feasible frontierをCP3ゲートへ反映
Study75完了(Universe-A確定済み + Universe-B生成)
   ├→ Study76 (Universe-B・Clenow複雑性判定)
   │     ├─WHITE→ architecture simplification（多層Exit機構ごと消滅・Study77不要）
   │     └─BLACK→ Study77 (optional・ユーザー起案時のみ)
   └→ Study102 (Small Growth・廃止込みデータ必須)
Study83 (TSMOM・データ独立・随時並行可) ┐
Study80 feasibility (MN・純データ分析)   ├→ CP3スリーブ採用ゲート（§6・事前固定）
Study82 (PEAD・発表日時精度監査が関門)   ┘
Core (Study76/FUJIKO2.0後のPIT再測定値) → CP3を同条件審査（保証枠なし）
CP3当選スリーブ≥2 ─→ Study86(MN当選時のみ・ショート執行) ─→ Study85(統合・CP4)
[CLOSED] Study79 — 恒久閉鎖(§2)
[並立] FUJIKO 2.0系 Study87-97 — fujiko_r2_research_roadmap.md管轄・Study95 KILL帰結の決裁待ち
```

### タイムライン（依存順・v1.1）

| 時期 | タスク | ゲート |
|---|---|---|
| 2026 Q3（即時） | **Study103**（依存ゼロ・0.5-1日） | **CP2: 統合の数学的成立性** |
| 2026 Q3 | Study75残作業（Universe-B生成） | — |
| 2026 Q3-Q4 | Study76（Universe-B上・複雑性判定） | Clenow判定（WHITE→77不要） |
| 2026 Q4 | Study83（並行）/ Study80 feasibility / Study82 発表日時精度監査 | CP3個別審査 |
| 2027 H1 | 生存スリーブのWF本測定 + Core PIT再測定のCP3審査 | **CP3** |
| 2027 H2〜 | Study86（MN当選時のみ）→ Study85統合 | **CP4: 30%最終判定（統合のみ）** |

### 決定木（v1.1）

```
Study74 ─BLACK確定→ 資本経路恒久閉鎖・Study79 CLOSED・Core=スリーブ候補の1つへ格下げ
Study103 ─不成立(楽観水準でも)→ 30%最終棄却（Study85を待たず）→ 目標再定義ユーザー決裁
   └成立→ CP3ゲート確定・Satellite研究続行
Study76 ─白→ 簡素化実行・Study77不要 / ─黒(<-4pp)→ 複雑性正当化・77はoptional起案のみ
Study83/80/82/102 ─各々CP3審査→ 当選スリーブのみStudy85へ（Coreも同条件・保証枠なし）
全スリーブCP3不通過 → 30%最終棄却→実測値で運用フェーズ
30%判定 = Study85統合実測のみ（結合CAGR≥30% ∧ 結合MaxDD≤20% ∧ RoR<1%）
どの時点でも: 連続2四半期採用ゼロ → 運用フェーズ縮退
```

### 正典目標（本改定後の公式値・v1.1）

| 項目 | 値 |
|---|---|
| Core intrinsic alpha（PIT・公式） | **UNKNOWN（~0-5%・confidence LOW）**。名目10-15%は歴史参考値・意思決定使用禁止 |
| ~~Core Stretch~~ | 失効（純化後PIT再測定まで未定義） |
| 30% ∧ Calmar1.5 | **成立性はまずStudy103（CP2）で数学的に判定**。実測判定はStudy85のみ。Core単独・資本投入・Coreレバの経路は全て恒久閉鎖 |
| フォールバック | CP2不成立 or 全スリーブCP3不通過: その時点の実測値で目標確定・運用フェーズ移行 |

---

## §10 Appendix: Post-Study74 Reality Check（恒久参照用）

**目的**: 「資金を増やせば30%行けるのでは？」という議論への恒久的な回答。以後この議論が再提起された場合、本Appendixの引用のみで却下する。

| 項目 | 値 | 根拠 |
|---|---|---|
| Core intrinsic alpha（名目・歴史参考のみ） | ≈10-12% | M1後公式値（IS 12.22% / OOS 11.42% / Full 11.22%）。意思決定使用禁止 |
| **Core intrinsic alpha（PIT・公式）** | **UNKNOWN（~0-5%・confidence LOW）。α≈0の可能性を排除できない** | Study75C（選定バイアス+12.26pp・PIT無作為median IS -2.48%）・Study100（FATAL）・Study101（全構成RED・TOPIX劣後） |
| Capital scaling上限 | **≈+1pp**（+1.12pp・lot丸め解消のみ） | Study74 Part A waterfall・QED |
| レバによる上乗せ（Long Only Core） | **経路なし**（DD比例拡大でCalmar崩壊） | Study79 CLOSED（§2）・alternative_architectures_5x |
| 30%までの残ギャップ | **+18pp超（名目基準）/ PIT基準では25-30pp** | 上記の差分 |
| 統合の数学的成立性 | **未判定 → Study103（CP2）でSatellite掘削前に先行判定** | §6A |
| 結論 | **ギャップは資本・レバ・Core改良のいずれでも埋まらない。まずStudy103で統合の数学的成立性を判定し、成立時のみ直交アルファ源（TSMOM/MN/PEAD）のCP3審査+統合が唯一の経路** | §6/§6A |

**今後の研究優先順位（恒久掲示・v1.1）**:
```
1 Study75完了（Universe-B生成）
2 Study76（複雑性判定・WHITE→77不要）
3 Study103（Portfolio Feasibility・CP2・依存ゼロ=並行即時着手可）
4 Study83（TSMOM・実装容易・ROI最高Satellite）
5 Study80 feasibility（MN・実現はStudy86実装重量で割引）
6 Study82（PEAD・日時精度監査が関門）
7 Study85（統合・CP3当選≥2で起案）
8 Study77（optional・76 BLACK時のみ・実質凍結）
```

**禁止再提案リスト（v1.1追補含む）**:
- 資本増額による目標到達（Study74 QED）
- Long Only Coreへのレバ（Study79 CLOSED）
- Core単独30%目標の復活（§6）
- RSR42名目実績を将来判断の根拠に使う行為（Study101で消滅）
- 「Coreに5%程度のアルファが残る」と断定する表記（§3・公式表記はUNKNOWN/confidence LOW）
- Study103のスリーブ仮定を動かして30%成立を演出する行為（§6A・仮定スイープ禁止）
- CoreのStudy85保証枠（80-90%）復活（CP3同条件審査のみ）

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
