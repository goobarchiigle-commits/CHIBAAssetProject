# Capital Flow Generator — Investment Feasibility Research Contract v1.0（2026-08-02制定）

**上位文書**: `rgp_taxonomy_v1.0.md`（Frozen）。本契約はRGP-Independent Edge探索の一手法として、
既存RGP分類とは独立に、Capital Flow由来Triggerの研究プロセスを規定する。

**目的（誤解防止のため明記）**: Capital Flow Generatorを作ることが目的ではない。

> 既存Production/Coreでは捕捉できない、PITで再現可能なCapital Flow由来の投資機会を、
> 300万円口座で実際に取引可能な形で反復生成できるかを判定する。Candidate Generatorは
> そのための手段であり、Generator自体の完成を成果とはしない。

---

## 0. Mission / Current State

```
MISSION
12か月以内に実売買で利益を出せる独立Edgeを構築する。
CAGR 30%は最終目標であり、研究の目的関数ではない。
まずCAGR 15%級の再現可能なEdgeを1本作る。30%は独立Edgeの積み上げによって目指す。

CURRENT STATE
現行Production/Core: Gate 0 CLOSED
根拠: Study95(CS Momentum KILL) / Study100(Universe hindsight bias FATAL) /
      Study101(TOPIX B&H に劣後) / Study82(PEAD FAIL) / Study83(TSMOM REJECT) /
      Study103(Core Retirement Prob=100%)
Entry Freeze: 継続（新証拠なしに解除しない）。既存RSR/Coreの再監査は行わない。
```

---

## 1. Mechanism（基本形）

```
External Demand / Policy / Budget Shock
        ↓
Economic Requirement
        ↓
Capacity / Supply Constraint
        ↓
Capex / Procurement
        ↓
Critical Economic Role
        ↓
Candidate Universe
        ↓
Market Recognition
        ↓
Entry Window
        ↓
Winner Selection（既存Signal）
        ↓
Position
```

Winnerを起点に逆算しない。Mechanismを先に固定し、その後で候補を機械生成する。

---

## 2. Research Gates

| Gate | 内容 | 判定形式 |
|---|---|---|
| G1 | Economic Mechanismを一文で説明可能・既存RSR/Coreと独立仮説であること | PASS/FAIL |
| G2 | Historical Recurrence。**LLM記憶による事例列挙は禁止**。政府予算文書・閣議決定・法律成立/施行・METI等の認定計画・補助金認定・防衛装備庁契約等、構造化一次情報の日付順走査のみ | PASS/FAIL |
| G3 | PIT Detectability。当時公開情報＋機械的判定Ruleで確定可能。Winner/株価をTrigger定義に使用禁止 | PASS/FAIL |
| G4 | Role Mapping。想定データソース・PIT availability・entity mapping方法・人手判断の有無を明記。データソース未確定は自動的にCONDITIONAL PASS（PASS扱い禁止） | PASS/CONDITIONAL PASS/FAIL |
| **G4.5** | **Investable Universe**（新設）。TSE上場か・現行実行パイプラインで扱えるか・`capital_scaling`/`max_lot_cost_ratio`内で購入可能か。株価水準は見てよいが forward return/勝敗は見ない | PASS/CONDITIONAL PASS/FAIL |
| G5 | Candidate Generation。Rule固定後は同一入力→同一銘柄集合。個別銘柄の「関連していそう」判断禁止。**Level 2以降（サプライヤーのサプライヤー）への拡張は初期研究では禁止** | PASS/FAIL |
| G6 | Independent Edge。Validation前に仮説固定・事後変更禁止。**Matched Control必須**（A: Full Market / B: Candidate / C: Matched Control の3群比較、Cは任意ではない） | H0/H1事前固定 |
| G7 | Market Timing Potential。Trigger→Candidate確定→Market Recognitionの時間差を検証。**Phase1（Candidate確定前）に混ぜてはならない**——株価を見てCandidateを修正するループを防ぐため、Phase2で初めて実施 | PASS/FAIL（Phase2でのみ判定） |
| G8 | Tradeability/Profitability。年間Opportunity数・Candidate数・想定Holding Period・300万円口座での実現可能性・期待Return寄与・MaxDD影響を数値評価 | Phase3で判定 |

### Level 0 定義補足（JV/非上場対応）

Direct BeneficiaryがJV・非上場法人の場合、「当時開示された出資・投資コミットメントを持つ
上場親会社」をLevel 0として扱う。現在の勝者を見て親会社を選ばない。

### Prohibited Research

```
Winnerからテーマを逆算する
特定銘柄を見てRole定義を変更する
「関連銘柄リスト」を人手で作る
Candidate数を増やすためRoleを拡張する
Threshold sweep
結果を見てValidation定義を変更する
新規データ契約を先に行う
株価結果からEconomic Mechanismを後付けする
「経済的に正しい」だけで採用する
```

---

## 3. Phase構成

```
Phase 0 — Mechanism Feasibility（G1/G2/G3/G4）。株価非参照。
Phase 1 — Candidate Generation Feasibility（G4.5/G5/G6仮説保存/G7項目のみ事前登録）。株価非参照。
Phase 2 — Market Recognition / Timing Audit（G7実施）。ここで初めて株価を見る。
Phase 3 — Minimal Backtest（G6実施・A/B/C比較・G8評価）。Phase1/2 PASS時のみ。
Phase 4 — Shadow Deployment。
Phase 5 — Production（段階的投入・Position size最大化しない）。
```

**Phase3のExit設計注意（Holding Period Warning）**: Capital Flow型Triggerの自然な実現期間は
四半期〜年単位（設備投資→稼働→業績反映）。既存Productionの`turtle_exit=55d`/`max_hold_days=60`を
機械的に流用しない。独立したExit設計を行うこと。

---

## 4. Multiple Testing統制（Trigger Class選定ルール）

Phase0で複数のTrigger Class候補を**先に**評価・ランク付けする。個別Trigger ClassのPhase1で
不合格（特にG4.5/G6での既存Universe重複等の構造的欠陥）が判明した場合、次点への移行は
「Phase0時点で既にランク付けされていた候補への移行」に限る。**個別Phase1の結果を見てから
新しいTrigger Classを物色することは禁止**（当たりが出るまで探す研究になるため）。

---

## 5. Business Goal Gate（新規研究タスク冒頭で必須）

```
Business Goal

この研究が成功した場合、
1. 何個の投資Opportunityを年間生成するのか
2. 既存Productionでは拾えないOpportunityなのか
3. 300万円口座で実際に購入可能なのか
4. 期待利益に何%程度寄与し得るのか
5. 失敗した場合、どの時点でSTOPするのか

を事前に数値または明確な判定基準で示せ。示せない場合、研究を開始してはならない。
```

---

## 6. 実施履歴

| 日付 | Phase | 対象Trigger Class | 結果 | 参照 |
|---|---|---|---|---|
| 2026-08-02 | Phase0 | 半導体/防衛/GX（3候補評価） | 半導体=CONDITIONAL PASS、防衛=CONDITIONAL PASS、GX=FAIL | `docs/research/capital_flow_generator_mechanism_feasibility_2026-08-02.md` |
| 2026-08-02 | Phase1 | 半導体（Level0=JASM出資組／Level1=METI補助金採択） | **STOP**（既存Core Universeとの構造的重複によりIndependent Edge欠如。Pipeline Rehearsalとしての目的は達成） | `docs/research/capital_flow_generator_candidate_feasibility_2026-08-02.md` |
| — | Phase0 | 防衛装備庁契約データ（次候補・事前登録済み） | 未着手 | — |

**次アクション**: 防衛装備庁契約データ型のPhase0を別セッションで新規実施（本契約のGate定義を
そのまま適用）。Winner Blind維持。
