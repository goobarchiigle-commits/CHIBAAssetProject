# Study103 — Portfolio Architecture Feasibility 実行結果（CP2判定・Goal Ladder Sweep）

**日付**: 2026-07-20
**性格**: Monte Carlo実行結果（新規BTなし・エンジン非経由・仮定表ベースの数学的反証試行）。
**正典**: `study103_design.md`§9A-9C（凍結仕様）/ `roadmap_v15_governance_layer.md`§1/§1A/§8。
**スクリプト**: `src/backtest/study103_portfolio_feasibility.py` / 出力: `backtests/study103_portfolio_feasibility_2026-07-20.json`

```
Primary Objective  : Attempt to falsify ambitious routes.
Secondary Objective: Determine feasible ceilings and terminal states.
Failure to falsify ≠ proof.
```

---

## 1. Goal Ladder Sweep（成果物#1）

| Tier | 目標 | CaseA (Core included) | CaseB (Core excluded) |
|---|---|---|---|
| **Tier3** | 30% / Calmar≥1.5 | **RED**（Base不成立・Optimisticは成立するが自動RED境界に抵触） | **RED**（同左） |
| **Tier2** | 20-25% / Calmar≥1.3 | **GREEN** | **GREEN** |
| **Tier1** | 10-18% / Calmar≥1.0 | **GREEN**（Conservativeでも部分成立） | **GREEN**（同左） |
| **Tier0** | Market Return | GREEN（自明成立） | GREEN（自明成立） |

**CaseA/CaseBが全Tierで完全一致** — 全レベルの最適配分でCoreの重みが常に0%だったため（§4参照）。

---

## 2. Goal Frontier（Calmar制約別の最大到達median CAGR）

| Level | Calmar制約 | 最大CAGR | 配分 | MaxDD | RoR |
|---|---|---|---|---|---|
| Conservative | ≥1.0 | **17.6%** | MN35%/PEAD40%/TSMOM20%/SG5% | 12.9% | 0.91% |
| Conservative | ≥1.5 | infeasible | — | — | — |
| Base | ≥1.0〜1.5 | **27.7%** | PEAD70%/TSMOM10%/SG20% | 13.2% | 0.88% |
| Optimistic | ≥1.0〜1.5 | **35.7%** | PEAD80%/SG20% | 8.9% | 0.02% |

**Tier3(30%)はBaseであと2.3pp届かない**（僅差不成立・27.7%止まり）——「絶望的に遠い」のではなく「僅差で越えられない」失敗の仕方である点は正しく記録する。

---

## 3. 自動RED境界の発動（Tier3・Optimisticのみ評価対象）

```
required_sleeves_min          = 1     （境界: ≥5で発動 — 該当なし）
required_avg_corr_min         = 0.0   （境界: <0.10で発動 — 数値上は該当）
required_avg_assumed_calmar_min = 2.821 （境界: >2.0で発動 — ★該当）
判定: TRIGGERED（avg_calmar経由）
```

**重要な注記（数値の意味を正確に読むこと）**: `required_sleeves_min=1`は、Optimistic水準では**単一スリーブ（PEAD100% or SG100%）だけでTier3条件を満たす配分が存在した**ことを意味する。この時`required_avg_corr_min=0.0`は「低相関を達成した」という意味ではなく、**スリーブ数1では相関の定義自体が空**（分母ゼロのフォールバック値）という技術的な退化値 — 誤読注意。

**この結果が示すもの**: Optimistic水準では、PEAD単体（CAGR35%・仮定MaxDD10%→仮定Calmar3.5）のような**単一の未検証スリーブへの集中ベット**だけで数学的にTier3へ到達し得る。これはまさに自動RED境界（avg Calmar>2.0）が捕捉するために設計された退化解 — 「分散ではなく単一銘柄依存で数字を作る」というStudy103の反証対象そのもの。境界は設計通りに機能した。

---

## 4. Core Retirement Probability = **100%**

**機械的定義**: 3水準中、Case B frontier(Calmar≥1.3)がCase A frontierと同等以上となる水準の割合。

**結果**: 3水準（Conservative/Base/Optimistic）**全てでCoreの最適配分重みは0%**。CaseA最適解に一度もCoreが選ばれなかったため、CaseA=CaseBが構造的に成立した。

**⚠ 解釈上の注意（Falsification原則3の適用）**: これは「Coreを退役させるべき」という証明では**ない**。本MCはCore CAGRをPIT観測値（Base=3%）で固定した設計上の帰結であり——**仮定表がCoreのα低さを既に織り込んでいるため、最適化器が当然それを避けただけ**。一方でMN/PEAD/TSMOM/SGは全て**CP3未通過の仮定値**（PEADは発表日時精度監査すら未実施）。「未検証だが仮定上優秀な4スリーブ」と「観測値で実測済みだが冴えないCore」を同列に最適化すれば、後者が選ばれないのは当然の帰結であり、Core CP3審査（本物のPIT再測定）の結果を先取りするものではない。100%という数値の強さに反して、**これは弱い証拠**（Failure to falsify ≠ proof の裏返し: Success to include ≠ proof of value も同様に成立しない）。

---

## 5. Termination Probability = **10.9%**

**機械的定義**: 6シナリオ（3水準×2ケース）の「最良Tier1配分でもTier1条件を満たさないパス割合」の平均。

**結果**: Conservative水準でのパス失敗率が31.2%（最良配分でも約3割のパスがTier1未達）と最大の寄与源。Base/Optimisticはほぼ0%。**Conservative水準の脆さがTermination Probabilityの主成分**。

---

## 6. CP2判定（Tier3=30%/1.5に対する正式判定）

```
CP2 = RED
```

判定根拠: Base水準で不成立（27.7%<30%）。Optimistic水準は名目上成立するが自動RED境界（avg Calmar>2.0）に抵触し、単一スリーブ集中という退化解でしか到達していないため実質的に不成立。楽観水準でも構造的な成立を示せなかった。

**`roadmap_v15_governance_layer.md`§1A決定木の適用**:
```
CP2 RED → Tier2 feasible?
            YES（Tier2=GREEN両ケース）→ Route B
```

**→ Route B（Diversified Alpha・目標20-25%/Calmar1.3-1.5・構成Core+PEAD+TSMOM）が機械的に起動する。**

この遷移は事前固定された決定木の適用のみであり、新たな議論・裁量判断を伴わない（§1A「Study103終了後の次に何をするかの議論を禁止する」）。

---

## 7. Research Continuation Policy（成果物#7・advisory）

- **Budget recommendation**: §6配分（Core35/Satellite45/Exploratory10/Reserve10）は変更不要。Route B確定によりSatellite45%枠の優先配分先が確定（PEAD・TSMOM）——MN(Study80)は本MCで最適配分にほぼ登場せず（Conservative水準でのみ35%出現）優先度は現行の暫定順位（TSMOM>PEAD>MN）から**PEAD>TSMOM>MN**への入替を検討材料として提示するのみ（advisory・採用はユーザー決裁）。
- **Terminal state recommendation**: Route F前倒し不要。Tier2が頑健にGREENであるため研究継続価値は明確に正。

**⚠ advisory出力はゲート判定（§1-2）と混同しないこと**（`roadmap_v15_governance_layer.md`§8）。

---

## 8. 想定される反証ポイント（study103_design.md §9.3を実測後に再確認）

1. **PEAD依存の集中**: Base/Optimistic双方の最適配分がPEADに70-80%集中。PEAD仮定表（CAGR27.5%@Base）自体が`alternative_architectures_5x`の理論経路記述からの機械配分であり実測ゼロ。**Study82の発表日時精度監査FAILがそのままRoute Bの土台を崩す**——次善はStudy82結果を最優先で確認すること。
2. **正規尾部の楽観バイアス（設計通り）**: §9C(c)で明記した通り、月次lognormalは fat tail を過小評価する。実測RoR/MaxDDは本結果より悪化する可能性が高い——**GREEN判定はここでも「棄却できなかった」以上の意味を持たない**。
3. **Conservative水準の脆さ**: Termination Probability(10.9%)の主因はConservative水準のパス失敗率31.2%。MN/PEAD/TSMOMいずれかの実測がConservativeレンジを下回った場合、Tier1ですら揺らぐ可能性がある。

---

## 9. 次アクション

Route B起動により優先順位が確定: **Study82（PEAD発表日時精度監査・第一関門）と Study83（TSMOM CP3）が次点**。いずれも新規スクリプト作成のため個別ASK_FIRST。Study75/76はCore処遇（E1/E2分岐）に必要なため並行して継続。

*作成: CLD (Fable 5)・2026-07-20。フレッシュラン実施済み（MC・仮定表ベース）。実トレードデータのバックテストではない。*
