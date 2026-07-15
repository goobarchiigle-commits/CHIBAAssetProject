# backtest-research

## Purpose

バックテスト・Study設計・戦略評価・Production採用判断における研究信頼性の確保。
研究の再現性・統計的妥当性・Bias排除を最優先とする。

---

## Priority Order

**Research quality gates — applied in this order. Violation of any gate = REJECT regardless of CAGR.**

| Priority | Gate | Rule |
|----------|------|------|
| 1 | No Data Leakage | Future data must never reach the model during training or signal generation |
| 2 | No Look-ahead Bias | Signal on day T may only use information available before market open on T |
| 3 | Walk Forward First | IS-only results are hypothesis, not evidence. WF >= 4/5 required for ADOPT |
| 4 | Statistical Validity | N < 5 trades = no evidence. Sharpe > 3.0 = suspect leakage |
| 5 | Robustness | Result must hold across regimes and WF folds, not just the best period |
| 6 | CAGR Improvement | Evaluated only after gates 1–5 are satisfied |

> Gates 1–5 are non-negotiable. A Study that passes gate 6 but fails any of gates 1–5 is REJECTED.
> Quality before return. Never the reverse.

---

## Use When

- Study設計・実験計画を立てるとき
- バックテストスクリプトを実行・解釈するとき
- IS/OOS・Walk Forward結果を評価するとき
- Production採用・Reject判断をするとき（ADOPT/REJECTの判定そのもの）

---

## Not For

- run_live_signal.py 実行・注文発注・ポジション管理 → `/live-signal` を使う
- 朝ルーティン・API接続確認 → `/live-signal` を使う
- Study成果物Markdownの体裁化・Executive Summary/Method/Results作成 → `/report-generator` を使う
- Research State/Roadmap/Decision Record/Open Questions/Closed Researchへの反映・Study採番管理 → `/roadmap-governance` を使う

---

## 1. Study Pre-flight Checklist

**Study開始前に必ず全項目を確認する。1つでも未確認なら実行禁止。**

```
[ ] Study番号・名称を決定済み（例: Study56_xxx）
[ ] 仮説を1文で記述済み（何を検証するか）
[ ] IS期間・OOS期間・WF分割を定義済み
[ ] 評価指標（Calmar / CAGR / Sharpeなど）を事前定義済み
[ ] 比較ベースラインを明示済み（前Studyまたは固定ベース）
[ ] 変更パラメータが1つ（または明示的に複数）であることを確認済み
[ ] データソースのカットオフ日を確認済み（未来データを使っていないか）
[ ] Survivorship Biasの有無を確認済み
[ ] src/research_state.md を読み込んで現在のベースラインを確認済み
```

---

## 2. Bias / Leakage Guard

### Look-ahead Bias 禁止

```
禁止:
  - 評価日時点で未来の価格・出来高・指標を参照
  - shift()なしでの当日終値→翌日エントリー
  - RSR・ATRなどをOHLCV全期間でfit後にバックテスト適用

許可:
  - 前日終値・前日RSRのみを当日シグナルに使用
  - shift(1)を明示したフィーチャー
```

**確認コマンド:**
```python
# エントリー日とシグナル計算日のずれを確認
assert signal_df.index.max() < entry_date  # 未来参照を検出
```

### Data Leakage 禁止

```
禁止:
  - スケーラーをIS+OOS全期間でfit
  - test_data_in_train（OOSをISに混入）
  - shuffle=True（時系列データで絶対禁止）
  - 全期間統計量（平均・標準偏差）をフィーチャーに使用

必須:
  - cv=TimeSeriesSplit_only
  - scaler.fit(IS_data_only)
  - pipeline=sklearn.Pipeline（fitとtransformを分離）
```

### Survivorship Bias 確認

```
確認事項:
  - ユニバース（universe.yaml）が評価期間時点のものか
  - 上場廃止銘柄がデータに含まれているか
  - 新規上場銘柄の扱い（上場後N日以内は除外など）
  - 現在の動的ユニバース（AUTO_PROMOTE）の履歴を使用しているか
```

---

## 3. Walk Forward Protocol

**WF合格基準: 5fold中4fold以上でΔCAGR > 0**

```
標準設定:
  - n_splits=5
  - test_size=1年
  - train_size=累積（expanding window）
  - gap=0（隙間なし）

Fold構成例（5年データ）:
  Fold1: train=2020, test=2021
  Fold2: train=2020-2021, test=2022
  Fold3: train=2020-2022, test=2023
  Fold4: train=2020-2023, test=2024
  Fold5: train=2020-2024, test=2025
```

**WF結果の解釈:**

| WF結果 | 判定 |
|--------|------|
| 5/5 PASS | STRONG ADOPT候補 |
| 4/5 PASS | ADOPT候補（REJECT理由を確認） |
| 3/5 PASS | REJECT（停止条件発動） |
| 2/5以下 | REJECT（過学習を強く疑う） |

**停止条件（WF実行中）:**

```
発動条件:
  - 任意のFoldでΔCAGR < -2pp かつ WF < 4/5
  - 全Foldで同一方向の失敗
  - Fold1（最古期間）での大幅劣化（市場環境依存の可能性）
```

---

## 4. IS/OOS Validation

**IS/OOS比率基準（CLAUDE.md: `oos_is_ratio_min`）:**

```
計算: OOS_CAGR / IS_CAGR >= oos_is_ratio_min

判定:
  >= 1.0  → 優秀（OOSがISを上回る）
   0.7-1.0 → 合格（許容範囲）
  < 0.7  → 要注意（過学習の疑い）
  < 0.3  → 過学習確定→REJECT
```

**IS/OOS期間設定:**

```
推奨比率: IS:OOS = 7:3 以上
最短OOS: 1年以上（統計的意味を保つため）
OOS期間は事前定義必須（後から変更禁止）
```

---

## 5. Statistical Validation

> **閾値はすべてCLAUDE.mdの`VALIDATION`セクションを正とする。**
> ここに再掲している数値はない。CLAUDE.mdを参照して判定すること。

**最低取引件数（CLAUDE.md: `trade_min`）:**

```
IS:  N >= trade_min（必須）
OOS: N >= trade_min（必須）
WF各Fold: N >= 3（望ましい）

N < trade_min の場合: 統計的に無意味→結果を報告に使用禁止
```

**Sharpe上限チェック（CLAUDE.md: `sharpe_max`）:**

```
IS Sharpe > sharpe_max → 先読みリークを強く疑う
  → Pre-flight Checklistに戻り Bias/Leakage再確認
  → データ確認なしに結果採用禁止

OOS Sharpe > sharpe_max → 同様
```

**DD上限チェック（CLAUDE.md: `dd_max`）:**

```
IS/OOS最大DD > dd_max → 戦略として成立しない→REJECT
```

---

## 6. Overfitting Detection

**過学習シグナル一覧:**

```
警戒:
  □ IS Sharpe > 3.0 かつ OOS Sharpe < 1.0
  □ IS CAGR > 30% かつ OOS CAGR < 10%
  □ パラメータ感度が高い（±10%変化で結果が大きく変わる）
  □ WF < 4/5（特定期間でのみ有効）
  □ 取引件数が少ない（N < 10）で高Sharpe

確定:
  □ oos_is_ratio < 0.3
  □ WF 2/5以下
  □ Sharpe > 3.0 かつ Leakが排除できていない
```

**防止措置:**

```
single_metric_optimization=forbid  ← Calmar単独最適化禁止
param_sweep_limit=bounded           ← グリッドサーチ範囲制限
stability_check=required            ← 複数指標での安定性確認必須
walkforward_required=true           ← WFなしのIS最適化結果は採用禁止
```

---

## 7. Production ADOPT Criteria

**全条件を満たす場合のみ ADOPT:**

```
必須条件:
  [1] WF >= 4/5 PASS
  [2] OOS/IS ratio >= 0.7
  [3] OOS CAGR > 0（絶対値）
  [4] OOS trade_N >= 5
  [5] IS Sharpe <= 3.0（Leak疑い排除）
  [6] IS/OOS両方でDD <= 50%
  [7] Seg3（最新セグメント）がマイナスでないこと

強い採用条件（追加評価）:
  [+] WF 5/5
  [+] OOS/IS ratio >= 1.0
  [+] 複数指標（CAGR/Calmar/Sharpe）で一貫した改善
  [+] 全セグメントで正のΔCAGR
```

---

## 8. Production REJECT Criteria

**1つでも該当すればREJECT（例外なし）:**

```
即時REJECT:
  [R1] WF <= 3/5
  [R2] OOS/IS ratio < 0.3
  [R3] IS Sharpe > 3.0 かつ Leak確認不可
  [R4] OOS最大DD > 50%
  [R5] OOS trade_N < 5
  [R6] 任意FoldでΔCAGR < -10pp以上の大幅劣化
  [R7] Survivorship Bias確認不能

条件付きREJECT（再設計で再提出可）:
  [C1] 仮説と異なる改善機序（偶発的改善）
  [C2] 特定レジームのみ有効（Bear/Bullどちらかで-8pp以上）
  [C3] パラメータ感度が高すぎる
```

---

## 9. Study Review Procedure

**Study完了後に以下の順序で実施:**

```
Step 1: 結果の数値確認
  - IS CAGR / OOS CAGR / WF結果 / oos_is_ratio
  - 全Foldの結果を一覧化（Fold毎のΔCAGR）
  - 取引件数（IS/OOS各N）

Step 2: ADOPT/REJECT判定
  - Section 7（ADOPT基準）を上から順にチェック
  - 1つでもREJECT条件に該当→即REJECT記録

Step 3: 失敗分析（REJECTの場合）
  - どのFoldで失敗したか（年代・市場レジームを記録）
  - 失敗の主因（CAGR劣化 / DD増加 / N不足）
  - 次Studyへの示唆

Step 4: PARAMS_LOCKEDへの影響確認（ADOPTの場合）
  - パラメータ変更を伴う場合→ASK_FIRST（ユーザー確認必須）
  - 変更なし採用の場合はStep 5へ

Step 5: Output引き渡し（Section 10）
  - Study Decision / Study Metrics / Study Evidence を `/report-generator` へ渡す
  - 本Skillはここで終了する。research_state.md更新・roadmap反映・日次ログ記録・レポートMarkdown作成は一切行わない（それぞれ `/roadmap-governance` / `/report-generator` の責務）
```

---

## 10. Output（Study Decision / Study Metrics / Study Evidence）

**backtest-researchの最終責務はこの3点を確定させ、`/report-generator`へ渡すことで終わる。** research_state.md更新・レポート体裁化・roadmap反映は行わない。

```
Study Decision:
  - 判定: ADOPT / REJECT
  - 根拠ゲート番号（Section 7 ADOPT基準 or Section 8 REJECT基準の該当項目）
  - REJECT理由（該当する場合、失敗したFold・主因を含む）
  - PARAMS_LOCKED影響有無・ASK_FIRST該当有無

Study Metrics:
  - IS/OOS: CAGR・Calmar・Sharpe・MaxDD・N・oos_is_ratio
  - WF: Fold別ΔCAGR・PASS/FAIL・合計pass数
  - Bootstrap（実施した場合）: median・CI・P(>0)
  - 感度分析結果（実施した場合）: 崖の有無

Study Evidence:
  - 結果JSONの保存パス（backtests/studyNN_*_YYYY-MM-DD.json）
  - 使用したスクリプトパス・fresh run実施日時
  - Parity Check結果（該当する場合）
```

引き渡し後の扱い（本Skillの管轄外）:
```
研究ログ・レポート作成    → /report-generator
research_state.md反映    → /roadmap-governance
roadmap/Decision Record反映 → /roadmap-governance
```

---

## 11. Research Assets利用ルール

**責務**: 既存Studyの成果物（backtests/*.json, reports/*.md等）を再利用する際の運用ルールのみを規定する。データ取得・解析の実装コードは本Skill内に持たない。

```
共通データ取得APIを利用する（実装はSkill外・Pythonライブラリ側に委譲）:
  - データ取得・universe構築・価格ロードは、既存の共通ライブラリ（例: composite_alpha_bt.pyのローダー群、将来のJ-Quants取得ライブラリ等）を必ず経由する
  - 本Skill内にAPI呼び出しコード・取得ロジックを実装・保持しない（実装場所は src/ 配下の該当ライブラリ）

既存Research Assets再利用の判断:
  - Production判定にキャッシュ値・過去JSON流用は禁止（CLAUDE.md OVERFIT_GUARD: fresh_run_required=true）
  - 「参照値」としての引用（例: 過去Studyの比較対象数値を凍結値として転記）は許可。ただし「今回の判定の根拠数値」としての流用は禁止
  - 新規Study開始時は既存Assetsの棚卸しを実施し、以下を区別する:
    [ ] 再利用可能（fresh run不要・参照値としてのみ使用）
    [ ] 新規収集・再測定が必要（fresh run必須）
  - Universe変更（例: survivorship-free化）を伴う研究では、旧Universeの既存Assetsは「旧Universe参考値」として明示的に区別し、新旧比較は禁止する（Universe差とArchitecture差の交絡防止）
```

---

## 12. Statistical Analysis

**責務**: 統計解析の手順のみを規定する。実装コードは含めない（実装は既存の解析スクリプト、または新規スクリプトとしてASK_FIRST後に作成）。

```
Bootstrap（信頼区間・破産確率相当の分布推定）:
  - N=500・seed=42を既定値とする（既存Study方式踏襲）。変更する場合は理由を明記
  - トレード順リサンプリングによりCAGR/MaxDD分布・CI[5%,95%]・P(>0)を算出
  - 手順のみ規定し、実装は既存のBootstrapスクリプト方式に委譲する

感度分析（パラメータ頑健性チェック）:
  - 事前固定グリッドのみ使用（CLAUDE.md: param_sweep_limit=bounded）
  - 主要パラメータを±5/10/20%変化させ、ΔCAGR>3ppとなる崖の有無を確認する
  - 崖を検出した場合は「非頑健フラグ」を付与し、過学習疑いとしてSection 6の判定に反映する
  - グリッド外の探索的スイープ（新しい値域への拡張）は禁止 — 0.4ゲート・恒久閉鎖領域抵触のリスクがあるため、実施前に必ずユーザー確認を取る

適用順序:
  1. WF/IS-OOS判定（Section 3-4）を先に完了する
  2. WF/IS-OOS基準を通過した場合のみBootstrap・感度分析を実施する（不合格が確定している場合に追加解析へ進まない）
```

---

## 13. Parity Check

**責務**: エンジン変更前後・BT/Live間で、同一条件下の結果が一致することを確認する手順のみを規定する。

```
目的:
  - エンジン変更（新規計装追加・バグ修正等）が意図しない挙動変化を生んでいないかを検証する
  - BTとLiveの実行条件差異（執行価格タイミング等）を検出する

手順:
  1. 変更前エンジンでの基準値（CAGR/Trades/Sharpe/MaxDD/Calmar等）を固定・記録する
  2. 変更後エンジンで同一設定・同一期間をfresh run実行する
  3. 全指標のΔ=0.00pp（または明示的に許容した誤差範囲内）であることを確認する
  4. 不一致があれば変更内容の副作用（意図しないロジック変更）を疑い、原因（RCA）特定まで採用しない

適用場面:
  - 新規スクリプト作成後のロールバック検証
  - 観測専用の計装追加後（既存事例: Study80A observation infrastructureのparity_report.md方式）
  - Live実装とBTエンジンの整合確認（既存事例: M1 Addon執行価格PATCHでのBT/Live乖離発見・RCA実施）

判定:
  - PASS: 全指標Δ=0.00pp（計装追加のみ・ロジック変更なしの場合の期待値）
  - FAIL: いずれかの指標に不一致 → ロールバックしRCA実施。RCA完了・原因説明可能になるまでProduction適用しない
```
```
