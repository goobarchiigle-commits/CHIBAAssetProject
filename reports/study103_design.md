# Study103 — Portfolio Architecture Feasibility Design（起案書・CP2）

**日付**: 2026-07-20
**性格**: **設計書のみ**。コード・バックテスト・パラメータ探索は一切実施しない。本書は
`roadmap_revision_2026-07-19.md`（v1.3）§6A「Study103」のPrimary Objective・三原則・
シナリオ凍結・自動RED境界を、実行可能な起案仕様まで具体化するもの。次段（Monte Carlo実装・
`src/backtest/study103_portfolio_feasibility.py`新規作成）は本書提示後に別途ASK_FIRST。
**正典**: `reports/roadmap_revision_2026-07-19.md` v1.3 §6A。矛盾時は正典が優先。

---

## 0. Objective（正典より再掲・変更禁止）

```
Attempt to falsify the 30% route.
The route is accepted only if it survives all predefined tests.

Failure to falsify ≠ proof of feasibility.
```

本Studyの目的は「30%へ到達する方法を探す」ことでは**ない**。「30%ルートを否定できるか試みる」こと。
GREEN判定ですら「30%達成可能」の証明にはならない — 意味するのは「棄却できなかったため研究継続を
正当化できる」のみ。この非対称性は本書全体・後続の全出力物の冒頭に明記すること。

---

## 1. Goal（判定対象・数値仕様）

以下4条件の**同時**成立が数学的に妥当（plausible）か否かを判定する:

| 条件 | 閾値 | 出典 |
|---|---|---|
| CAGR | **≥30%** | CP4定義（`roadmap_revision`§9正典目標） |
| Calmar | **≥1.5** | CP4定義 |
| MaxDD | **≤20%** | CP4定義 |
| RoR（5年ホライゾン・P(MaxDD>30%)） | **<1%** | Study78方式踏襲（`study78_ror_mc_sensitivity.md`） |
| Capacity | **≥¥3M**（現行資本で執行可能） | 自動RED境界§7・PARAMS_LOCKED capital=3M |

**判定方式**: 上記4条件の直接検証ではなく、**逆問題（reverse requirement）**として「この4条件を満たすために各スリーブ属性・スリーブ数・相関構造に何が必要か」を先に導出する（§6）。個々のスリーブ研究（Study80/82/83/102）は、この逆算された必要条件をCP3ゲートの追加基準として引き継ぐ。

---

## 2. Candidate Sleeves（対象スリーブ・固定5種・追加禁止）

```
1. Core（現行D_ATR_EQ・PIT再測定値。保証枠なし・CP3同条件審査対象）
2. Market Neutral（Study80・ARCH-A・モメンタム・スプレッドMN）
3. PEAD（Study82・ARCH-B・決算ドリフト）
4. TSMOM（Study83・ARCH-C・指数タイムシリーズモメンタム）
5. Small Growth Momentum（Study102・ARCH-E・小型グロース）
```

ARCH-D（クロスマーケット・リードラグ）は対象外（旧正典Study84・生存確率最低・独立killテスト対象のため本Studyのスリーブ候補には含めない）。5種は固定 — シナリオ実行後の追加禁止（§4シナリオ凍結と同じ制度・永久禁止4「Result-after parameter tuning」該当）。

---

## 3. Required Metrics（各スリーブの仮定表・事前固定・3水準）

**出典方針**: 実測値がある項目（Core・現行RoR/MC）はfresh run実測を使用。未実測のスリーブ（MN/PEAD/TSMOM/SmallGrowth）は`alternative_architectures_5x_2026-07-03.md`の文献レンジ・理論経路記述をConservative/Base/Optimisticへ機械的に配分する（恣意的な楽観化を防ぐため、出典の記述レンジの下端=Conservative・中央=Base・上端=Optimisticを原則とする）。

### 3.1 Core

| 指標 | Conservative | Base | Optimistic | 出典 |
|---|---|---|---|---|
| CAGR | 0%（α消滅） | 3% | 5% | Core PIT公式表記: Intrinsic alpha=UNKNOWN・Observed PIT estimate≈0-5%（`roadmap_revision`§3）。**名目10-15%は永久禁止3「Static RSR42 performance citation」に該当するため使用しない** — Optimisticの上限もObserved PIT estimateの上端5%に固定 |
| Volatility（年率） | 18% | 15% | 13% | Production FULL run MaxDD=-18.22%（`study78_ror_mc_sensitivity.md`）から逆算した近似レンジ。PIT再測定前の暫定値 |
| MaxDD | -25% | -18% | -14% | 同上。実測-18.22%をBaseに、UNKNOWN前提でConservativeを拡大 |
| Calmar | 未定義（CAGR Conservative=0%のため） | 0.17 | 0.36 | CAGR/|MaxDD|の機械計算。Conservativeはα消滅につき評価不能 |
| Core相関（自己） | — | — | — | 対象外（自己相関は定義しない） |
| Capacity | ¥3-10M | ¥10-20M | ¥20-30M | Study74 Capacity Curve（lot丸め解消¥20M完全解消・以降max_positions構造天井） |
| レバ適性 | 不可 | 不可 | 不可 | Study79 CLOSED恒久（`roadmap_revision`§2）。Coreへのレバは永久禁止2 |

### 3.2 Market Neutral（Study80）

| 指標 | Conservative | Base | Optimistic | 出典 |
|---|---|---|---|---|
| スプレッド素α（コスト前） | 8% | 10% | 12% | `alternative_architectures_5x`§ARCH-A「スプレッド素α 8-12%（文献レンジ・要実測）」 |
| コストストレス控除 | -2%/年一律（逆日歩） | 同左 | 同左 | Study80正典コストストレス定義（旧正典L5） |
| 信用レバ | 2.0x | 2.25x | 2.5x | 同上「信用レバ2-2.5x」 |
| CAGR（レバ後） | 12%（(8-2)×2.0） | 18%（(10-2)×2.25） | 25%（(12-2)×2.5） | 機械計算 |
| MaxDD | -15% | -10% | -6% | β≈0前提（市場暴落非連動）。文献の一般的MN型戦略DD水準を保守的に採用 |
| Core相関 | 0.10 | 0.00 | -0.15 | `alternative_architectures_5x`比較総括表「現行との収益相関（推定）＝低〜負」 |
| Capacity | ¥3-5M | ¥5-10M | ¥10M+ | Study80失敗条件「貸借ユニバース<100銘柄」が下限制約。上限は貸借銘柄プールの厚みに依存し未実測 |
| レバ適性 | 適（β≈0のためDD比例拡大せずCalmar維持） | 同左 | 同左 | 5案中唯一の構造（`alternative_architectures_5x`§28） |

### 3.3 PEAD（Study82）

| 指標 | Conservative | Base | Optimistic | 出典 |
|---|---|---|---|---|
| イベント当たり純期待値（40d・コスト後） | 1.5% | 2.25% | 3.0% | `alternative_architectures_5x`§ARCH-B「純期待値+1.5〜3%」 |
| 年間採択件数 | 30件 | 40件 | 50件 | 同上「年間採択30-50件」 |
| スロット回転数 | 3 | 3.5 | 4 | 同上「3-4スロット回転」 |
| CAGR（理論経路） | 20% | 27.5% | 35% | 同上「20-35%」を直接採用（内部整合は要Study103本実装で再計算） |
| MaxDD | -20% | -14% | -10% | 未実測。イベント集中リスク（季節性）を考慮しConservativeを厚めに設定 |
| Core相関 | 0.20 | 0.10 | 0.00 | 同上比較総括表「低」を定量化（保守側に幅を持たせる） |
| Capacity | ¥3-8M | ¥8-15M | ¥15M+ | イベント数上限（年30-50件）による回転制約。銘柄あたり約定額が上限を規定 |
| レバ適性 | 不可（未検証） | 限定的 | 限定的 | 資本回転率がレバの代替（同上「資本回転率がレバの代替になる」）——固有のレバ設計は正典未定義 |

### 3.4 TSMOM（Study83）

| 指標 | Conservative | Base | Optimistic | 出典 |
|---|---|---|---|---|
| Sharpe（文献） | 0.6 | 0.8 | 1.0 | `alternative_architectures_5x`§ARCH-C「文献Sharpe 0.6-1.0」 |
| Volatility（vol-target） | 25% | 22.5% | 20% | 同上「vol-target 20-25%運用」 |
| CAGR（単独） | 15%（0.6×25%概算） | 18% | 25%（1.0×20%+α） | 同上「単独CAGR15-25%が上限圏」・Sharpe×Volの概算と整合させ機械配分 |
| MaxDD | -20% | -15% | -12% | 未実測。vol-target運用の一般的DD水準を保守的に採用 |
| Core相関 | 0.35 | 0.25 | 0.15 | 同上比較総括表「現行との収益相関（推定）＝中」を定量化 |
| Capacity | ¥30-60万（証拠金レイヤー） | 同左 | 同左 | 同上「証拠金¥30-60万のレイヤー追加」。**現物¥3M運用と証拠金効率を共有** — 単独Capacity制約ではなく合算経路である点に注意（§6注記） |
| レバ適性 | 適（vol-targetに内在） | 同左 | 同左 | 先物のため証拠金効率が高い |

### 3.5 Small Growth Momentum（Study102）

| 指標 | Conservative | Base | Optimistic | 出典 |
|---|---|---|---|---|
| CAGR（理論経路） | 20% | 27.5% | 35% | `alternative_architectures_5x`§ARCH-E「CAGR20-35%圏に理論到達」 |
| MaxDD | -50% | -40% | -30% | 同上「2022型で-30〜50%級DDの現実的可能性」 |
| Calmar | 0.40 | 0.69 | 1.17 | 機械計算。同上「Calmar1.5は絶望的、1.0も困難」と整合（Optimisticでも1.17止まり） |
| Core相関 | 0.70 | 0.55 | 0.40 | 同上比較総括表「現行との収益相関（推定）＝高（同方向β）」 |
| Capacity | ¥1-2M | ¥2-3M | ¥3M+ | 小型株流動性制約（日次売買代金¥50-300M帯） |
| レバ適性 | 不可 | 不可 | 不可 | DD構造がCalmar目標と正面衝突（同上） |

---

## 4. Scenario Matrix（3水準固定・シナリオ凍結）

```
A. Conservative — 全スリーブConservative列を採用
B. Base         — 全スリーブBase列を採用
C. Optimistic   — 全スリーブOptimistic列を採用
```

**シナリオ凍結（`roadmap_revision`§6A・厳守）**:
```
No additional scenarios may be introduced after execution.
実行開始後のシナリオ追加・変更・水準調整
（「相関0.20なら？0.18なら？」型の探索）は一切禁止。

追加が必要になった場合の唯一の手続き:
  1. 新版採番（Study103B）
  2. 新規ASK_FIRST承認
  3. 全シナリオのfull rerun（部分再実行・旧結果との混用は禁止）
```

**混合シナリオ（例: Coreだけ悲観・MNだけ楽観）は本Studyの範囲外** — 3水準は「全スリーブ一律」の縦割りシナリオのみ。個別スリーブごとの感度分析はCP3個別Study（Study80/82/83/102）で別途実施。

---

## 5. Case Analysis（Core included / excluded・両方必須）

```
Case A: Core included   （Core + MN + PEAD + TSMOM + SmallGrowth の5スリーブ候補プール）
Case B: Core excluded   （MN + PEAD + TSMOM + SmallGrowth の4スリーブ候補プール）
```

**A×B×3シナリオ = 6通り全実行**（省略禁止）。

**判定規則（`roadmap_revision`§6A・v1.3）**: Case Bのfeasible frontier（§6）がCase Aより緩い（=必要条件がより達成しやすい）場合、**Core retirement（Core全面撤退）を正式選択肢**としてユーザーへ提示する。これはCore保証枠廃止（永久禁止7「Core guaranteed allocation」）の対偶検証 — 「Coreを含めない方が統合効率が良い」という可能性を制度的に排除しないための措置。

**注記**: Core CAGR ConservativeはCase Aでも0%（α消滅）と仮定するため、Case A ConservativeとCase B Conservativeの差は理論上小さい可能性がある。Case A/Bの乖離が主に現れるのはBase/Optimisticシナリオになると想定される（実測前の予想であり、本Study結果で検証する）。

---

## 6. Reverse Requirement Analysis（逆問題形式・主出力）

**手法方針（実装時の仕様・本書では計算しない）**: 6シナリオ（Case×水準）それぞれについて、スリーブ属性（CAGR/Vol/相関）を所与として、目的関数「結合CAGR≥30% ∧ 結合Calmar≥1.5 ∧ 結合MaxDD≤20% ∧ RoR<1%」を満たす配分ベクトルが存在するかをMonte Carlo（ブロックbootstrap・Study78方式）で走査し、存在しない場合は「境界に最も近づいた配分」から不足分を逆算する。

**出力様式（事前固定・`roadmap_revision`§6A）**:
```
最低スリーブ数        : N sleeves
必要単独Calmar        : > X
必要相関上限（平均）   : < Y
必要単独リターン（平均）: > Z%
必要Capacity          : > ¥W
許容レバ              : ≤ V x
```

**予備的・非公式な見立て（illustrative only・§3仮定表からの手計算概算・MC未実施）**:

Base シナリオ・Case A（Core含む）で単純平均分散近似（等配分・簡易2資産合成式の直列適用）を行うと、Core（CAGR3%・Vol15%）とMN（CAGR18%・Vol概算15%・相関0.00）を1:1で合成した場合の結合CAGR≈10.5%・分散低減効果は相関0のため部分的——**30%には遠く届かない**。TSMOM（CAGR18%・相関0.25）・PEAD（CAGR27.5%・相関0.10）・SmallGrowth（CAGR27.5%・相関0.55・MaxDD-40%）を均等5分割で加算平均するとCAGR単純平均≈16.9%——**分散効果を最大限見積もっても30%到達には高リターンスリーブへの偏重配分または追加スリーブが必要**という方向性が示唆される。

⚠ **これは正式なStudy103出力ではない**。相関構造を考慮した結合分散・Calmar・RoRの厳密な計算にはMonte Carlo実装（新規スクリプト・ASK_FIRST）が必須。上記は「本Studyが着手に値するか」を判断するための手計算スケッチに留まる。

---

## 7. Automatic RED Conditions（自動RED境界・発動可能性の事前評価）

正典境界（`roadmap_revision`§6A・事後緩和禁止）と、§3仮定表に基づく発動可能性の**定性的**評価:

| 境界 | 閾値 | 発動可能性（定性評価） | 根拠 |
|---|---|---|---|
| required sleeves | ≥5 | **中〜高** | 5候補全てを使っても§6予備計算でCAGR不足の兆候。5スリーブ全採用が前提になりうる時点で「5未満で足りる」楽観シナリオの確度は低い |
| required avg corr | <0.10 | **中** | MN(相関0.00-0.10)・PEAD(0.00-0.20)は達成可能域だが、SmallGrowth(0.40-0.70)を含めると平均が押し上がる。Case B（Core除外）でも高相関を持つSmallGrowthが残るため無条件では回避できない |
| required avg Calmar | >2.0 | **低〜中** | §3のOptimisticではMN単独Calmar=25%/6%≈4.2と高いが、これはMN一銘柄カテゴリの値であり「平均Calmar」は5スリーブ合成後の値。SmallGrowth（Optimisticでも1.17）が平均を押し下げるため、要求水準>2.0への到達可否は定性的に不確実 |
| required leverage | >2x | **低** | MN想定レバは2.0-2.5x（Study80正典レンジ内）。TSMOMはvol-target内在で別枠。合算配分でのレバ要求が2xを超える設計は現時点で想定していない |
| required capacity | <¥3M | **低** | 現行PARAMS_LOCKED capital=3Mが下限として機能しているため、¥3M未満の要求は生じにくい構造 |

**総合見立て**: `required sleeves≥5`と`required avg corr<0.10`が発動しやすい2項目 — SmallGrowth（高相関・DD過大）を含めた5スリーブ構成では相関要求を満たしにくく、除外すると4スリーブでリターン不足という**トレードオフ構造**が予想される。これは正式なMC結果ではなく、Study103本実装が最優先で検証すべき仮説として記録する。

---

## 8. Falsification Principles（正典・4原則）

```
1. Attempt falsification.
   仮説（30%ルートの成立）の反証を試みる。証明を目指さない。

2. Only predefined scenarios are valid.
   事前定義された6シナリオ（3水準×2ケース）のみが有効。追加・変更は§4のシナリオ凍結手続き経由のみ。

3. Failure to falsify is not proof.
   It only justifies continuing research.
   反証の失敗（GREEN/YELLOW判定）は証明ではない。研究継続を正当化するのみ。

4. No post-result parameter exploration.
   結果を見た後の仮定・パラメータ探索は禁止（永久禁止5「Result-after parameter tuning」該当）。
   自動RED境界（§7）の事後緩和も本原則違反として扱う。
```

原則1-3は`roadmap_revision`§6A「Study103三原則」と同一。原則4は同§6Aのシナリオ凍結規定・自動RED境界の事後緩和禁止を、本書の文脈では独立原則として明示（実質は既存禁止事項の言い換え・新規制約ではない）。

---

## 9. Deliverables（本書の納品物・次段への引き継ぎ）

### 9.1 Assumptions Table
§3の5スリーブ×3水準の仮定表（全21指標×5スリーブ）。実装時はこの表をそのままMCスクリプトの入力定数として使用し、**表の値を実装段階で変更しない**（変更が必要な場合は§4シナリオ凍結手続きに従う）。

### 9.2 Rationale
各仮定値の出典を`alternative_architectures_5x_2026-07-03.md`・`study78_ror_mc_sensitivity.md`・`study74_final_review.md`・`roadmap_revision_2026-07-19.md`の該当箇所に紐付け済み（§3各表の出典列）。文献未実測項目（MaxDD・相関の一部）は「未実測」と明示し、実測値と誤認されないようにした。

### 9.3 Expected Falsification Points（本Studyが崩れ得る点）
1. **Core Case A/B判定の感度**: Core CAGR=0%（α消滅）と仮定した場合、Case AとCase Bの差が§6予備計算より小さい可能性がある。実装時はCore=0%固定シナリオを含めた追加の頑健性チェックが望ましい（ただし新規シナリオ扱いとなり§4凍結規定の対象——本Study本体には含めず申し送りとする）。
2. **相関仮定の信頼性**: MN/PEAD/TSMOM/SmallGrowthの相関値は全て「文献推定・要実測」（`alternative_architectures_5x`原文）。実測前の仮定に基づくため、GREEN判定が出てもCP3個別Studyでの実測相関が仮定から乖離すれば判定は覆り得る（原則3の帰結そのもの）。
3. **TSMOM Capacity注記**: TSMOMは現物と証拠金を共有する合算経路のため、他スリーブと同列のCapacity制約として扱うと過大評価するおそれがある。MC実装時は現物資本¥3M内での証拠金積み増しとして別建て処理が必要。

### 9.4 Preliminary CP2 Classification Criteria（判定基準案・実装時の起点）
```
GREEN  : 6シナリオ中、Base以上（Base・Optimistic）で自動RED非該当 ∧
         Conservativeでも「絶望的」ではない（frontierの必要条件が既知候補群のOptimisticレンジ内）
YELLOW : Baseで自動RED非該当だがConservativeで自動RED該当、または
         必要条件が既知候補群のBase/Conservativeレンジを超えOptimisticレンジ内に留まる
RED    : Conservative/Baseの双方で自動RED該当、またはOptimisticでも§7境界のいずれかに抵触
感度規則: 上記のいずれかで判定が水準間で反転する場合、1段階悪い側を機械採用（`roadmap_revision`§6A）
```
本節は**案**であり、実装時（MCスクリプトASK_FIRST承認時）に正式確定する。§7の定性評価（sleeves≥5・corr<0.10が発動しやすい）を踏まえると、**現時点の暫定見立てはYELLOW〜RED寄り**——ただしこれ自体が正式なCP2判定ではなく、次段（実装）の優先度判断材料に留める。

---

## 9A. 設計追補（2026-07-20・v1.4戦略層による・実行前=シナリオ凍結発効前の正規手続き）

**Goal Ladder Sweep（`roadmap_v14_strategy_layer.md`§7）**: §6の同一6シナリオ・同一MC出力から、以下3本の閾値でread-outを行う（**新規シナリオではない** — 仮定表・MC計算は不変。読み出し閾値もこの3本で凍結し、実行後の追加を禁止する）:

```
Tier1: CAGR≥30% ∧ Calmar≥1.5 ∧ DD≤20% ∧ RoR<1%   → GREEN/YELLOW/RED
Tier2: CAGR≥20% ∧ Calmar≥1.3 ∧ DD≤20% ∧ RoR<1%   → GREEN/YELLOW/RED
Tier3: CAGR≥10% ∧ Calmar≥1.0 ∧ DD≤20% ∧ RoR<1%   → GREEN/YELLOW/RED
+ feasible frontier曲線（Calmar制約別の最大到達CAGR）を主図表として出力
```

- §7自動RED境界は**Tier1判定にのみ**適用（Tier2/3はfrontier読み出しのみ）。
- Tier1 REDでも研究は終了しない — 目標をTier2へ再アンカーし継続（Failure Tree・`roadmap_v14_strategy_layer.md`§5）。
- **MN仮定の解釈注記（Study95反映）**: §3.2の素α8-12%は値として維持するが、この生存前提は「Clenow型1-3M短期スプレッド」に限定（12M型はStudy95でfactor-level反証気味）。Study103レポート解釈節およびStudy80起案書へ必ず引き継ぐこと。

## 10. 次アクション（ASK_FIRST）

1. 本設計書のユーザー承認
2. `src/backtest/study103_portfolio_feasibility.py` 新規作成（ASK_FIRST・Study78 MC資産再利用）
3. 6シナリオ（3水準×2ケース）fresh run実行
4. `backtests/study103_portfolio_feasibility_YYYY-MM-DD.json` + `reports/study103_portfolio_feasibility.md`（正式結果・GREEN/YELLOW/RED確定）出力
5. research_state.md・roadmap_revision双方へ結果転記・SAVE伝播

*作成: CLD (Fable 5)・2026-07-20。新規バックテスト実行なし・コード変更なし・パラメータ探索なし。*
