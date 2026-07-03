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
- Production採用・Reject判断をするとき
- Study結果をsrc/research_state.mdへ記録するとき

---

## Not For

- run_live_signal.py 実行・注文発注・ポジション管理 → `/live-signal` を使う
- 朝ルーティン・API接続確認 → `/live-signal` を使う

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

Step 4: 研究ログ保存（Section 10のフォーマット）
  - src/research_state.mdを更新
  - backtests/に結果JSONを保存
  - docs/research/YYYY-MM-DD.mdに日次ログを記録

Step 5: PARAMS_LOCKEDへの影響確認（ADOPTの場合）
  - パラメータ変更を伴う場合→ASK_FIRST（ユーザー確認必須）
  - 変更なし採用の場合→コミット後に報告
```

---

## 10. Research Log Format

**src/research_state.md 更新フォーマット:**

```markdown
## StudyNN: [Study名] — YYYY-MM-DD

**仮説:** [1文で記述]
**変更内容:** [何を変えたか]

| 指標 | IS | OOS | WF |
|------|----|-----|----|
| CAGR | xx% | xx% | x/5 |
| Calmar | x.xx | x.xx | — |
| Sharpe | x.xx | x.xx | — |
| Max DD | xx% | xx% | — |
| N | xx | xx | — |
| OOS/IS ratio | — | x.xx | — |

**WF Fold結果:**
- Fold1 (2021): ΔCAGR=+x.xxpp [PASS/FAIL]
- Fold2 (2022): ΔCAGR=+x.xxpp [PASS/FAIL]
- Fold3 (2023): ΔCAGR=+x.xxpp [PASS/FAIL]
- Fold4 (2024): ΔCAGR=+x.xxpp [PASS/FAIL]
- Fold5 (2025): ΔCAGR=+x.xxpp [PASS/FAIL]

**判定:** ADOPT / REJECT
**REJECT理由:** [該当する場合]
**次Study示唆:** [次に試すべきこと]
```
