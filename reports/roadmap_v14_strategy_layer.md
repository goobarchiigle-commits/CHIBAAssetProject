# Roadmap v1.4 — Strategic Layer（戦略目標層・代替経路）

> ## ★ STATUS: ANNEX（2026-07-20〜）
> エントリーポイントは **`reports/roadmap_v15_governance_layer.md`（CURRENT CANON）**。
> 本書はRoute詳細仕様（各Route完全テーブル・ARCH-A〜E再評価根拠・Failure Tree原文）の参照付録。
> **本書への新規追記は禁止**。v1.5での変更点: Tier番号昇順化（本書のTier1-4→v15のTier3/2/1/0）・
> Route D=Dormant化（Study102 WHITEまで独立Route化禁止）・Route F発動4トリガー正式化・
> 優先順位=Study103最優先へ。矛盾時はv15優先。

**日付**: 2026-07-20
**性格**: 文書のみ（BT・コード変更・新規仮説生成なし）。`roadmap_revision_2026-07-19.md`（v1.3=研究統治層）の**上位に載る戦略層** — v1.3のCP体系・ゲート・禁止事項は一切変更しない。v1.3が「どう研究するか（How not to fool ourselves）」なら、本書は「どこへ向かうのか（Where we go / what if it fails）」を定義する。
**出典**: `alternative_architectures_5x_2026-07-03.md`（ARCH-A〜E原典）/ `final_research_roadmap_2026-07-04.md`（旧フォールバック体系）/ `study103_design.md`（仮定表）/ Study74/95/98/99/100/101実測。
**正典関係**: 本書は`roadmap_revision_2026-07-19.md`のv1.4付属文書。矛盾時は統治層（v1.3本体）のゲート・禁止事項が優先。

---

## §0 戦略層の3原則

```
1. 経路は事前定義。
   「Aが失敗したらB、Bが失敗したらC」を失敗前に書き切る。
   失敗後に経路を発明する行為は Result-after parameter tuning の戦略版として禁止。

2. 遷移は機械的。
   各Routeの kill condition と fallback destination は本書で固定。
   発動時の遷移判断に裁量を挟まない（配分変更の実行のみASK_FIRST）。

3. 確率は計画用prior。
   本書の成功確率は研究リソース配分のための事前確率であり、
   採用判定の根拠に使用禁止（採用は常にCP2/CP3ゲートの機械判定のみ）。
   CP2/CP3の各判定時点で更新する。
```

---

## §1 ARCH-A〜E 再評価（Study74/95/98/99/100/101後）

旧原典（2026-07-03）の記述を、以後の実測で「生存/死亡/再スコープ」に仕分ける。**そのまま復活は禁止** — 特に旧「Study81=現物30%唯一の経路」「Study79=22-26%」はStudy74/100/101と矛盾するため死亡扱い。

| ARCH | 旧主張 | 生存部分 | 死亡・再スコープ部分 | 事前確率の変化 |
|---|---|---|---|---|
| **A: MNスプレッド** | 素α8-12%×レバ2-2.5x・30%/1.5唯一の理論経路 | β≈0レバ構造の理論優位は生存。Study80=純データ分析の安さも生存 | **⚠再スコープ（Study95）**: Universe-A実測で12-1モメンタム12Mスプレッドは**負**（-1.83%・弱い逆転）→ 12M型L/Sスプレッドはfactor-levelで既に反証気味。生存余地は**Clenow型1-3M短期スプレッド**（1M +7.02%/3M +6.77%実測・ただしBear regime悪化）のみ。Study80設計は**リバランス1-3M・Short側寄与単離を事前固定**すること | 0.30 → **0.20-0.25（低下）** |
| **B: PEAD** | イベント純期待値1.5-3%×年30-50件=20-35% | **ほぼ全部生存**。シグナル源が価格履歴の外＝Study95 Kill（価格モメンタム）の射程外。Study60 IC天井の外側という論理も不変 | 発表日時精度監査が第一関門（FAIL=即死）は不変 | 0.35-0.45（不変） |
| **C: TSMOM** | 文献Sharpe0.6-1.0・現物と証拠金共有の合算経路 | **ほぼ全部生存**。指数レベル時系列モメンタム≠クロスセクション株式モメンタム→Study95 Killの射程外。40年データ=統計力最強も不変 | 「現物12%+先物10-15%」の合算前提のうち**現物12%側が死亡**（Core=UNKNOWN）→ TSMOMは「上乗せ」ではなく**主力候補**へ役割格上げ | 0.45-0.55（**相対的に最有力へ**） |
| **D: リードラグ** | 事前確率<30%・killテストのみ | killテスト設計のみ生存 | 戦略Routeの構成部品には**含めない**（Study103スリーブ外・旧正典Study84のまま） | <0.30（不変・Route外） |
| **E: 小型グロース** | **「5案中唯一、現物のままCAGR20-35%圏」＝現物30%唯一の経路** | α密度仮説（小型CS分散2-3倍）・アクセスエッジ仮説は未検証のまま生存。survivorship-freeデータは**取得済み**（J-Quants・障害解消） | **「30%唯一の経路」表現は死亡**: ①未検証 ②小型は上場廃止/増資/流動性蒸発/S安連発が大型より遥かに多い ③Study101 B_abs OOS -45%（2025-04ショック・高モメンタム銘柄直撃）は高モメンタム集中プールの尾部リスクの直接警告 ④Calmar1.5と構造的非両立は原典自身が認定 → **High CAGR/High DD/Low CapacityのSatellite（配分10-20%上限）に恒久格下げ** | 0.25-0.35 → **0.20-0.30（微低下）** |
| **Core（参考）** | 素の実力10-12% | エンジン・Exit群・BT/Live parity・執行安全機構（インフラA級資産） | 期待値はUNKNOWN（観測≈0-5%・LOW）。候補スリーブの1つに格下げ済み（v1.3） | 0.15-0.25（Core CP3通過確率） |

---

## §2 Strategic Route Tree

### Route A — Conservative（TSMOM主軸+Core残置）

| 項目 | 内容 |
|---|---|
| 構成 | TSMOM（先物レイヤー・主力）+ Core（CP3通過時のみ・簡素化後） |
| 目標 | **CAGR 10-18% / Calmar ≥1.0**（Base）。ユーザー原案15-20%/1.2+は**Optimistic側**として保持 — Base目標をそこへ置くとCore=UNKNOWNの下で「目標を仮説に賭ける」行為の再演になるため、Study103仮定表（Core Base 3%+TSMOM Base 18%×notional比0.5-1.0）から機械導出した10-18%をBaseとする |
| 発動条件 | Study103 RED（Tier1閉鎖）or CP3生存スリーブ={TSMOM}のみ |
| required studies | Study83（CP3）・Study76 or FUJIKO2.0（Core処遇確定用・Coreなし構成なら不要） |
| kill condition | Study83 CP3不通過 |
| fallback | Route F（Terminal） |
| capacity | ¥3-20M（先物はスケール良好・Core側はmax_positions天井残存） |
| 実装難易度 | **低**（先物口座+既存ランタイム。5案中最易） |
| 計画prior | 終端状態として**~30%**（最も到達しやすい着地点） |

### Route B — Balanced（最有力）

| 項目 | 内容 |
|---|---|
| 構成 | Core（CP3通過時）+ PEAD + TSMOM |
| 目標 | **CAGR 20-25% / Calmar 1.3-1.5** |
| 発動条件 | Study103でTier2成立（Goal Ladder Sweep・§7）∧ PEAD・TSMOM両CP3通過 |
| required studies | Study82（日時精度監査→CP3）・Study83（CP3）・Study76/FUJIKO2.0（Core処遇） |
| kill condition | Study82監査FAIL or PEAD CP3不通過 → **Route Aへ自動降格**（TSMOM生存前提） |
| fallback | Route A |
| capacity | ¥3-15M（PEADのイベント数上限が拘束） |
| 実装難易度 | 中〜高（イベントパイプライン新規） |
| 計画prior | **~20-25%** |

### Route C — Aggressive（Tier1・30%/1.5）

| 項目 | 内容 |
|---|---|
| 構成 | Route B + MN（Clenow型1-3M短期スプレッド・レバ2-2.5x）± SmallGrowth（10-20%上限） |
| 目標 | **CAGR ≥30% / Calmar ≥1.5**（CP4実測でのみ宣言） |
| 発動条件 | **Study103 GREEN ∧ MN含む複数CP3通過 ∧ Study86執行parity達成** — 全AND。1つでも欠けたら起動不可 |
| required studies | Study80（1-3M再スコープ版feasibility）→ Study86（ショート執行）→ Study85（統合） |
| kill condition | Study80 fail（スプレッドα<5% or Short側寄与<2pp）or Study86 parity fail → **Route Bへ自動降格**（Bを壊さない） |
| fallback | Route B |
| capacity | ¥5-15M（MN貸借プール・PEADイベント数の二重拘束） |
| 実装難易度 | **高**（信用口座・borrow fee・在庫管理・執行インフラ=プロジェクト最重量） |
| 計画prior | **~5-10%**（旧全体推定<30%を、Study95によるMN経路低下で下方更新） |

### Route D — Small Growth Pivot（Satelliteとしてのみ）

| 項目 | 内容 |
|---|---|
| 位置づけ | **High CAGR / High DD / Low Capacity のSatellite部品**。独立Routeではなく、A/B/Cいずれかへの**上乗せオプション（配分10-20%上限）**。旧「現物30%唯一の経路」表現は§1により恒久撤回 |
| 目標寄与 | ポートフォリオ全体へ+2-5pp（配分10-20%×単独20-35%） |
| 発動条件 | Study102 CP3通過 ∧ ユーザーが「DD30-50%許容」を明示決裁（原典要件のまま） |
| required studies | Study102（survivorship-freeデータ=取得済・スリッページ0.5%・値幅制限モデル必須） |
| kill condition | CP3不通過（MaxDD>40% or 執行不能日>10% or CAGR<15%） → 単純閉鎖（他Routeに影響なし） |
| fallback | なし（overlay部品のため脱落しても本体Route不変） |
| capacity | **¥1-3M**（最小。資本拡大で最初に死ぬスリーブ） |
| 実装難易度 | 低〜中（ユニバース+コストモデル差し替え） |
| 計画prior | 最終構成に含まれる確率 **~10-15%** |

### Route E — Core Replacement / Retirement（v1.4で二分割）

ユーザー原案の「Route E=Full Replacement」は性質の異なる2事象を含むため分割する:

**E1: Core Replacement（実装置換・Coreという役割は存続）**
| 項目 | 内容 |
|---|---|
| 内容 | Study76 WHITE → 現行多層FUJIKO実装を純正Clenow最簡構成へ置換。「Core」スリーブ自体は存続（中身が変わる） |
| 発動条件 | Study76 WHITE（Δ≥-2pp∧複雑性削減成立） |
| 接続 | **fujiko_r2ロードマップCandidate A（Market→Stock Clenow）と同一ビークル** — Study95 Kill帰結の決裁（Clenow短期regime-gated限定継続 vs 全凍結）はここに合流する。E1採用時、FUJIKO 2.0側のCandidate A検証と統合し二重実装を禁止 |
| 帰結 | 多層Exit機構消滅→Study77起案不要（v1.3確定済み）。置換後CoreはPIT再測定でCP3再審査 |
| 計画prior | Study76 WHITE確率 **~40-50%**（Study101で旧フジコ法RED既確定のため、簡素構成が劣後しない確率は高め） |

**E2: Core Retirement（全面撤退・Satellite-onlyへ）**
| 項目 | 内容 |
|---|---|
| 内容 | CoreをポートフォリオからOUT。Satellite-only構成（TSMOM/PEAD/MN/SmallGrowthの当選群のみ） |
| 発動条件 | Study103 Case B優位（frontierがCase Aより緩い） or Core（置換後含む）CP3不通過 |
| 帰結 | 既存Coreインフラは**執行・検証基盤として全量存続**（撤退するのは戦略であってシステムではない）。Entry Freezeは事実上のE2先行実施状態にある点を明記 |
| fallback | Satellite群も全滅なら Route F |
| 計画prior | **~40-55%**（Study100/101の証拠の重さを正直に反映 — Core復権を既定路線としない） |

### Route F — Terminal / Preservation（v1.4新設・ユーザー案に欠けていた終端状態）

| 項目 | 内容 |
|---|---|
| 内容 | 全Satellite CP3不通過 or プログラムKill（連続2四半期採用ゼロ）時の**正式な着地点**。①新規リスクテイク恒久停止（Entry Freeze恒久化）②資本はTOPIX B&H相当のパッシブ運用 or 現金退避（ユーザー決裁）③研究は月次decay監視のみに縮退 ④執行・検証・データ基盤は凍結保存（再起動可能性のため） |
| 目標 | **市場リターン（TOPIX B&H）** — 「最低でも15-20%」という架空の床を置かない。Study101でTOPIX B&H（IS +5.92%/OOS +24.21%）が公式ベンチマークである以上、全戦略死亡時の正直な期待値は市場リターンそのもの |
| 発動条件 | 全スリーブCP3不通過 or プログラムKill発動 |
| 位置づけ | **敗北ではなく完了** — 「エッジが存在しないことを体系的に証明し、資本を守った」はStudy群のREJECT資産と同格の成果。この明文化がないと、全滅時に「何か掘れる場所が残っているはず」という延命バイアスが必ず発生する |
| 計画prior | **~20-30%** |

---

## §3 Route比較総括表

| 軸 | A: Conservative | B: Balanced | C: Aggressive | D: SG Pivot | E2: Retirement | F: Terminal |
|---|---|---|---|---|---|---|
| 目標 | 10-18%/1.0+ | 20-25%/1.3-1.5 | 30%+/1.5 | +2-5pp overlay | 構成依存 | 市場リターン |
| required | 83 | 82+83(+76) | 80+86+85+GREEN | 102 | 103CaseB or CP3 | なし（自動） |
| kill | 83 fail | 82 fail | 80/86 fail | 102 fail | — | — |
| fallback | F | A | B | （影響なし） | F | （終端） |
| capacity | ¥3-20M | ¥3-15M | ¥5-15M | ¥1-3M | 構成依存 | 制約なし |
| 難易度 | 低 | 中〜高 | 高 | 低〜中 | — | — |
| 計画prior | ~30% | ~20-25% | ~5-10% | ~10-15% | ~40-55% | ~20-30% |

（prior注記: A/B/C/Fは終端性能Tierの分布・D/E2は構成に関する周辺確率であり、次元が異なるため合計100%にはならない。終端Tier分布は§4参照。全てconfidence LOW・CP2/CP3で更新。）

---

## §4 Goal Ladder（目標の梯子・再較正版）

| Tier | 目標 | 対応Route | 計画prior | 備考 |
|---|---|---|---|---|
| **Tier1** | 30% / Calmar1.5 | Route C | **~5-10%** | CP4実測でのみ宣言可 |
| **Tier2** | 20-25% / Calmar1.3 | Route B | **~20-25%** | 最有力の現実的成功 |
| **Tier3** | ~~15-20%/1.0~~ → **10-18% / Calmar1.0-1.2** | Route A | **~30%** | ユーザー原案15-20%から下方修正。理由: Tierの値は対応Routeの**Conservative-Baseシナリオで到達可能**でなければ梯子自体が「目標を仮説に賭ける」再演になる。Core=UNKNOWN下でのStudy103仮定表からの機械導出値が10-18% |
| **Tier4** | 市場リターン（TOPIX B&H）/ 資本保全 | Route F | **~35-40%** | 架空の床を置かない終端。Tier1-3の残余確率 |

**降格規則**: 上位Tierの閉鎖は下位Tierの研究を止めない。Tier1閉鎖（Study103 RED）後もTier2/3の検証は同一CP3ゲートで継続する — これが§7のGoal Ladder Sweepを必要とする理由。

---

## §5 Failure Tree（自動遷移規則・事前固定）

```
Study103 Tier1=RED      → Route C恒久閉鎖 → 目標をTier2へ再アンカー（研究は継続）
Study103 Tier2もRED     → Route B閉鎖 → Route A既定化（Study83のみ続行）
Study103 全TierRED      → Satellite研究の目標を「Calmar改善」へ再定義 or Route F前倒し（ユーザー決裁）
Study103 Case B優位     → Route E2起動（Core retirement正式選択肢化）
Study76 WHITE           → Route E1起動（FUJIKO多層終了・Clenow置換・fujiko_r2 Candidate Aと統合）
Study76 BLACK ∧ Core CP3 fail → Route E2起動
Study82 監査FAIL        → PEAD恒久死 → Route B→A自動降格
Study83 CP3 fail        → TSMOM死 → Route A構成不能 → Core単独運用（CP3通過時）or Route F
Study80 fail            → MN死 → Route C到達不能 → Route B天井確定
Study102 fail           → Route D閉鎖（本体Route無影響）
全Satellite CP3 fail    → Route F（Terminal・敗北ではなく完了）
プログラムKill（2四半期採用ゼロ）→ Route F
```

**遷移の性質**: 上記は全て「次に何を研究するか/目標をどこに置くか」の自動遷移であり、**実弾の配分変更は含まない**（配分変更は常に個別ASK_FIRST）。遷移発生時はresearch_state.md先頭に遷移記録を必須追記。

---

## §6 Plan A-D定義（旧Plan B/C体系の後継）

```
Plan A = Route C（Tier1・30%/1.5）        — Study103 GREEN∧複数CP3の場合のみ
Plan B = Route B（Tier2・20-25%/1.3）     — 既定の主計画（最有力）
Plan C = Route A（Tier3・10-18%/1.0+）    — Study103 RED or Satellite半減時の既定
Plan D = Route F（Terminal・市場リターン） — 全滅時の正式着地（延命禁止）
```

旧`final_research_roadmap`のフォールバック体系（CP1白=18-22%/黒=15-20%）は**本体系で完全置換**（旧値はStudy74白前提を含み無効）。

---

## §7 Study103への追補指示（Goal Ladder Sweep — v1.4の最重要追加）

**問題**: v1.3のStudy103はTier1（30%∧1.5）の単一判定。RED即「ロードマップ終了」ではTier2/3の成立性が未判定のまま宙に浮く — ユーザー指摘の「Study103 REDになった瞬間ロードマップが終わる」危険はこれ。

**解決（追補）**: Study103の**同一6シナリオ・同一MC出力**から、複数目標水準の成立性を**読み出す**（read-out）:

```
Goal Ladder Sweep（同一runからの読み出し・新規シナリオではない）:
  Tier1: CAGR≥30% ∧ Calmar≥1.5 ∧ DD≤20% ∧ RoR<1%   → GREEN/YELLOW/RED
  Tier2: CAGR≥20% ∧ Calmar≥1.3 ∧ DD≤20% ∧ RoR<1%   → GREEN/YELLOW/RED
  Tier3: CAGR≥10% ∧ Calmar≥1.0 ∧ DD≤20% ∧ RoR<1%   → GREEN/YELLOW/RED
  + feasible frontier曲線（Calmar制約別の最大到達CAGR）を主図表として出力
```

**シナリオ凍結との整合**: 仮定表・シナリオ・MC計算は一切変更しない。変わるのは**同一結果に当てる閾値の本数のみ**（read-out）。よって凍結規定（新版採番+full rerun）には抵触しない — この解釈を本書で事前確定し、実行後の「読み出し追加」も禁止する（読み出し閾値もTier1-3の3本で凍結）。

**自動RED境界の適用**: §7境界（sleeves≥5等）は**Tier1判定にのみ**適用。Tier2/3は境界なし（frontier読み出しのみ）— 下位Tierまで境界で殺すと保守側に過剰。

**MN仮定の注記（Study95反映）**: `study103_design.md`§3.2のMN仮定値（素α8-12%）は変更しない（凍結前の値を維持）が、この値の生存前提が「Clenow型1-3M短期スプレッド」に限定された点（§1）を、Study103レポートの解釈節およびStudy80起案書に必ず引き継ぐこと。

---

## §8 確率の扱い（規律）

1. §2-§4の計画priorは全て**confidence LOW**。使途は研究リソース配分の優先順位づけのみ。
2. **採用判定への使用禁止**: 「prior 25%だから有望」は禁止語。採用は常にCP2/CP3の機械ゲート。
3. **更新点**: Study103完了時（CP2）・各CP3判定時・Study76決着時に本書のprior表を改版する（改版はv1.4.x採番・履歴保持）。
4. priorの恣意的上方修正（「もう少しで通りそうだから上げる」）は永久禁止5「Result-after parameter tuning」の戦略層適用として禁止。

---

## §9 反映先・整合性

| ファイル | 処置 |
|---|---|
| `reports/roadmap_revision_2026-07-19.md` | v1.4へ改版（changelog+本書への参照追加）。CP体系・禁止事項は無変更 |
| `reports/study103_design.md` | §7 Goal Ladder Sweep追補を設計段階追記（実行前・凍結発効前の正規手続き） |
| `src/research_state.md` / `docs/research/2026-07-20.md` / メモリ | SAVE伝播 |
| `fujiko_r2_research_roadmap.md` | 無改変（Route E1がCandidate Aへの合流点であることを本書側で明記） |
| 実弾・strategy.yaml・Entry Freeze | 無変更 |

**本書が変更しないもの**: v1.3の全ゲート・永久禁止7項・シナリオ凍結・自動RED境界（Tier1適用）・Entry Freeze・PARAMS_LOCKED。

*作成: CLD (Fable 5)・2026-07-20。BT・コード変更・新規仮説生成なし（既存ARCH定義の再評価と経路体系化のみ）。*
