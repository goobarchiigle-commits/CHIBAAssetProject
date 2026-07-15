# report-generator

## Purpose

Study単体の成果物（Markdownレポート）を統一フォーマットで作成する。
数値の妥当性判定そのもの（ADOPT/REJECTゲートの適用）は `/backtest-research` が行う。本Skillはその判定結果を「読める形」へ変換し、成果物として固定する責務に限定する。

---

## Use When

- Studyが終了し、成果物Markdownを作成するとき
- バックテストが完了し、結果をまとめるとき
- 検証結果をMarkdownへまとめるとき
- Executive Summaryを作成するとき
- Production Decision（ADOPT/REJECT/CONDITIONAL）を文書化するとき

---

## Not For

- Roadmap更新 → `/roadmap-governance` を使う
- Research State更新（src/research_state.md） → `/roadmap-governance` を使う
- Study番号管理・採番 → `/roadmap-governance` を使う
- ライブ運用・発注 → `/live-signal` を使う
- ADOPT/REJECT基準そのものの判定（ゲート適用・過学習検知） → `/backtest-research` を使う。本Skillは判定結果を転記するのみで、判定は行わない

---

## 0. Input Contract（/backtest-researchからの引き渡し）

`/backtest-research`はStudy Decision / Study Metrics / Study Evidence の3点を確定した時点で終了し、本Skillへ引き渡す。本Skillはこの3点を受け取り、Markdownレポートへ変換する（数値の再判定はしない）。

```
Study Decision  → 本書§8 Production Decision の入力
Study Metrics   → 本書§6 Results Template / §7 Statistical Summary の入力
Study Evidence  → 本書§10 Output Format のパス記録・根拠リンクの入力
```

---

## 1. Purpose セクションの書き方

各レポート冒頭に以下を1段落で明記する:

```
**作成日**: YYYY-MM-DD
**Study番号**: StudyNN
**目的**: [検証したい1つの問いを1文で]
**正典参照**: [矛盾時に優先されるroadmap/正典ファイルパス]
**新規BT有無**: [あり/なし。なしの場合は既存Research Assetsの再集計のみである旨明記]
```

---

## 2. Use When 適用チェック（レポート作成前）

```
[ ] Study本体のバックテスト・ゲート判定（/backtest-research）が完了している
[ ] IS/OOS/WF/Bootstrap等の数値が確定している（未確定のままレポート化しない）
[ ] 比較対象・期間・パラメータが固定済みである
```

---

## 3. Not For 境界確認（レポート作成前）

```
[ ] 本レポートはStudy単体の記録であり、research_state.md / roadmapへの反映は別作業（/roadmap-governance）であることを認識している
[ ] Study番号は既存採番済みのものを使用し、本Skill内で新規採番しない
```

---

## 4. Executive Summary Template

結論ファースト。非技術者にも伝わる分量（4-6行）。

```markdown
## Executive Summary

**結論**: [ADOPT / REJECT / CONDITIONAL / UNRESOLVED] — [1文の理由]
**主要数値**: IS CAGR xx% / OOS CAGR xx% / WF x/5 / Calmar x.xx
**仮説の成否**: [検証したかった問いに対するYES/NO]
**次の行動**: [1文]
```

---

## 5. Method Template

```markdown
## Method

| 項目 | 内容 |
|---|---|
| 仮説 | [1文] |
| 比較対象（Baseline） | [Study番号 or Production構成名] |
| 変更内容 | [何を変えたか。1変数原則。複数の場合は明示] |
| IS期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| OOS期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| WF分割 | n_splits=5 / expanding window |
| ユニバース | [使用したUniverse定義・出典Study] |
| コスト前提 | slippage=0.001 / commission=0.00055（CLAUDE.md PARAMS_LOCKED準拠） |
| エンジン | [使用スクリプトパス・既存エンジンからの差分有無] |
| fresh run | [実施日・キャッシュ不使用の明記（CLAUDE.md fresh_run_required=true）] |
```

---

## 6. Results Template

```markdown
## Results

| 指標 | IS | OOS | WF avg |
|---|---|---|---|
| CAGR | xx% | xx% | xx% |
| Calmar | x.xx | x.xx | — |
| Sharpe | x.xx | x.xx | — |
| MaxDD | xx% | xx% | — |
| N（trades） | xx | xx | — |
| OOS/IS ratio | — | x.xx | — |

**WF Fold別結果**:
- Fold1 (YYYY): ΔCAGR=+x.xxpp [PASS/FAIL]
- Fold2 (YYYY): ΔCAGR=+x.xxpp [PASS/FAIL]
- Fold3 (YYYY): ΔCAGR=+x.xxpp [PASS/FAIL]
- Fold4 (YYYY): ΔCAGR=+x.xxpp [PASS/FAIL]
- Fold5 (YYYY): ΔCAGR=+x.xxpp [PASS/FAIL]

**資産曲線・図表**: [パス（あれば）]
```

---

## 7. Statistical Summary

`/backtest-research` が定義する指標セットをそのまま転記する。本Skillで新しい指標を発明しない。

```markdown
## Statistical Summary

| 指標 | 値 | 基準（CLAUDE.md VALIDATION） | 判定 |
|---|---|---|---|
| Sharpe（IS） | x.xx | sharpe_max以下 | OK/NG |
| Sharpe（OOS） | x.xx | sharpe_max以下 | OK/NG |
| MaxDD | xx% | dd_max以下 | OK/NG |
| N（IS/OOS） | xx/xx | trade_min以上 | OK/NG |
| oos_is_ratio | x.xx | 0.7以上（0.3未満で過学習確定） | OK/NG/WARN |
| Bootstrap median | xx% | — | 参考値 |
| Bootstrap CI | [x%, x%] | — | 参考値 |
| Bootstrap P(>0) | x.xx | 高いほど良い | 参考値 |
| tail_capture | xx% | Study21基準80%（該当Studyのみ） | OK/NG/N/A |
```

Bootstrap手法（N/seed等）は`/backtest-research`の Statistical Analysis 章に従う。本Skillは結果を表に転記するのみ。

---

## 8. Production Decision

```markdown
## Production Decision

**判定**: ADOPT / REJECT / CONDITIONAL / UNRESOLVED
**根拠ゲート**: [/backtest-research のゲート番号を引用。例: "Section7 ADOPT基準 全7条件PASS" または "R2抵触のためREJECT"]
**PARAMS_LOCKED影響**: [あり/なし。ありの場合ASK_FIRST対象である旨明記]
**ASK_FIRST該当**: [あり/なし。ある場合は対象項目を明記]
**適用範囲**: [Production反映 / Shadow検証のみ / 参考記録のみ]
```

判定そのもの（ADOPT/REJECTの成立条件）は本Skillでは決定しない。`/backtest-research`側の判定結果をそのまま記載する。

---

## 9. Next Study

```markdown
## Next Study 示唆

- [次に検証すべき仮説候補]
- [今回のREJECT/ADOPTが示唆する方向性]
- [未解決のまま残った論点（Open Question候補）]
```

優先順位付け・次Study番号の割り当ては行わない（`/roadmap-governance`の責務）。ここでは「示唆」の記録に留める。

---

## 10. Output Format

```
成果物パス規約:
  レポート本体: reports/studyNN_<name>.md
  結果JSON:     backtests/studyNN_<name>_YYYY-MM-DD.json

ファイル冒頭の必須ヘッダー:
  作成日 / Study番号 / 正典参照 / 新規BT有無（本書§1）

フォーマット統一原則:
  - 全レポートは本Skill §4-9の章立て順序で統一する（Purpose→Use When確認は作業前チェックでありレポート本文には含めない）
  - 章の省略は禁止。該当なしの場合は「該当なし」と明記する（章ごと削除しない）
  - 数値は全て小数点2桁まで統一（%はpp表記をΔに使用）
  - 図表を含む場合はファイルパスのみ記載し、画像バイナリは別途backtests/等に保存する
```
