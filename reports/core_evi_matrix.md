# Core EVI Matrix — Expected Value of Information 評価行列

**日付**: 2026-07-04
**親文書**: `reports/core_architecture_completion_review.md` / 台帳: `reports/core_open_questions.md`
**定義**: EVI = 「追加研究で結論が変わる確率 × 変わった場合の意思決定価値 ÷ 研究コスト」の定性評価。**Core列がLow以下=Core内で追加研究する価値なし**。

---

## 指定4項目（重点評価）

### 1. max_positions=3

| 軸 | 評価 | 根拠 |
|---|---|---|
| 結論が変わる可能性 | **Low** | 独立実測4系列が一致: Study8系（sleeve/CAP/regime全REJECT）/ Study41（equity連動cap REJECT）/ Study53（max=10で+0.33pp・Sharpe悪化）/ Study74 PartA（¥3M +0.29pp〜¥20M -0.42pp・資本非依存で拘束継続） |
| 新意思決定の可能性 | Low | どの結果でも「3維持」（恒久閉鎖#11・PARAMS_LOCKED） |
| 研究コスト | Medium | WF一式+DD検証 |
| **EVI (Core)** | **Low** | 4回測って4回同じ答え。5回目は買わない |

### 2. Opportunity Cost（機会損失）

| 軸 | 評価 | 根拠 |
|---|---|---|
| 結論が変わる可能性 | **Low** | Study80Aで完全観測基盤化済み（見送り607件個別+forward_5〜60+MFE/MAE）。総量は確定: forward_20平均+2.8%・rank0喪失63.6%・年額換算¥19k(Study53)。回収手段は枠拡大（負・4回実測）/Entryフィルター（閉鎖#7）/Exit回転（Study77予約）で網羅済み |
| 新意思決定の可能性 | Low | 精度が上がっても回収経路が存在しない事実は不変 |
| 研究コスト | Low（データセット完備） | — |
| **EVI (Core)** | **Low** | 「αの実在」と「回収不能」は両立確定。測り直しに価値なし |

### 3. Portfolio State（状態依存仮説）

| 軸 | 評価 | 根拠 |
|---|---|---|
| 結論が変わる可能性 | Medium | 因果メカニズム自体は未証明（Study74B-RCA「未解決」宣言）。ただしStudy80Aの分散縮小24.8%が仮説を定量裏付け済み |
| 新意思決定の可能性 | **Low** | 因果がどう分解されても純効果（≦+0.33pp）実測済みのため決定不変（親文書§3.5） |
| 研究コスト | Medium-High | counterfactualポートフォリオ再構築BT+同時保有相関実測が必要 |
| **EVI (Core)** | **Low** | 「なぜ」の完全解明は説明価値のみ。「どうするか」は既に確定 |
| 帰属 | Study85（相関行列設計で必要になった時点で解く） | — |

### 4. Time Competition（候補到着×スロット解放の時間不整合）

| 軸 | 評価 | 根拠 |
|---|---|---|
| 結論が変わる可能性 | Medium | rank0候補が見送りの55.9-63.6%（Study74B/80A）— タイミング不整合は実在。回転Exitによる構造解消は未検証（上界+6.37pp, Study64） |
| 新意思決定の可能性 | Medium — **ただしその意思決定=Study77そのもの**（正典Phase2予約済み） | Core内には対応する意思決定が存在しない（Exitマイクロ=閉鎖#1、予測保護=閉鎖#2） |
| 研究コスト | High（3アームWF・Study76勝者構成が前提） | — |
| **EVI (Core)** | **Low** / **EVI (Study77) = High** | Coreの勘定で研究する対象ではない。後継プログラムの最重要問いとして既に予約済み |

---

## その他の未解決項目（台帳OQ準拠）

| 項目 (OQ) | 結論が変わる可能性 | 新意思決定 | コスト | EVI (Core) | EVI (帰属先) |
|---|---|---|---|---|---|
| OQ1 リスク相関構造の因果分解 | Medium | Low（決定実測済み） | Medium-High | **Low** | Medium (Study85) |
| OQ3 mom_period過学習疑い | Medium | Low（下方リスクのみ） | Medium | **Low** | Medium (Study76) |
| OQ4 survivorship幅 | High（数値は必ず更新される） | Low（方向は下方のみ・Arch優先順位不変） | Medium（J-Quants契約前提） | **Low** | **High (Study75・正典最上流指定済み)** |
| OQ5 Exit構造回収 | Medium | Medium | High | **Low** | **High (Study77)** |
| OQ6 クラスター粒度 | Medium | Low | Low | **Low** | Low-Medium (Study85) |
| OQ7 CAND_B資本頑健性 | 測定完了 | 決断のみ (S1) | ゼロ | — | — |
| OQ8 測定系不整合 | 解消済み（注記のみ） | なし | ゼロ | — | — |
| OQ9 defer結合 | 記録済み | なし（77申し送り） | — | — | Low (Study77) |
| OQ10 idle-cash候補 | Low（回収経路全滅） | なし | — | **ゼロ** | — |

---

## 判定

- **Core内 EVI High/Medium 項目: ゼロ件**。追加研究1本あたりの期待情報価値は、正典の起案基準（期待値+2pp未満は起案禁止・統治原則3）を全件下回る。
- High EVIは全て制約緩和側（Study75/77）に存在し、正典ロードマップが既に正しい優先順位（74→75→76→77→…）で予約済み。本監査による順位変更なし。
- **∴ EVI観点からCore研究終了は正当**。

*作成: 第三者監査, 2026-07-04。全数値は既存成果物からの引用・新規BTゼロ。*
