# Missed Signal Alpha Audit

作成日: 2026-06-01  |  対象期間: 2018-2025  |  max_positions=3

**問い:** Pattern A は max_positions=3 の制約により、構造的に高品質シグナルを捨てているか？

---
## 分析1: Missed Signal Alpha (Forward Return)

### 期間別集計

| 期間 | N | fwd20d mean | fwd20d WR | fwd40d mean | fwd40d WR | fwd60d mean | fwd60d WR | IR(60d) |
|---|---|---|---|---|---|---|---|---|
| IS_2018_2024 | 371 | +3.069% | 62.5% | +5.121% | 65.2% | +8.197% | 66.6% | 0.469 |
| OOS_2025 | 24 | +1.713% | 56.5% | +4.901% | 78.3% | +12.468% | 85.0% | 0.587 |
| Full_2018_2025 | 395 | +2.99% | 62.2% | +5.108% | 66.0% | +8.416% | 67.5% | 0.475 |

### IS 2018-2024 Missed Signal 詳細分布

- N=371  mean=+8.197%  median=+5.511%
- Win Rate=66.6%  IR=0.469
- P10=-11.639%  P25=-3.019%  P75=+16.663%  P90=+31.486%

---
## 分析2: Executed vs Missed Signal 比較

### IS 2018-2024

| | N | fwd20d mean | fwd20d WR | fwd40d mean | fwd40d WR | fwd60d mean | fwd60d WR |
|---|---|---|---|---|---|---|---|
| executed | 209 | +4.094% | 63.2% | +6.418% | 70.3% | +10.818% | 75.6% |
| missed | 371 | +3.069% | 62.5% | +5.121% | 65.2% | +8.197% | 66.6% |

### OOS 2025

| | N | fwd20d mean | fwd20d WR | fwd60d mean | fwd60d WR |
|---|---|---|---|---|---|
| executed | 41 | +2.523% | 65.9% | +13.318% | 85.7% |
| missed | 23 | +1.713% | 56.5% | +12.468% | 85.0% |

### RSR比較

| 期間 | Executed RSR mean | Executed RSR median | Missed RSR mean | Missed RSR median |
|---|---|---|---|---|
| IS | 78.4 | 82.9 | 72.3 | 80.5 |
| OOS | 74.6 | 78.6 | 78.0 | 79.8 |
| Full | 77.8 | 81.0 | 72.7 | 80.5 |

---
## 分析3: Missed理由別 (スロット状況)

| カテゴリ | N | fwd60d mean | fwd60d WR | RSR mean |
|---|---|---|---|---|
| pre_full (3→full) | 327 | +8.938% | 68.1% | 73.7 |
| day_fill (<3→full) | 68 | +5.934% | 64.7% | 67.7 |

---
## 分析4: RSR Rank Analysis

- 全 missed signals: **395件**
- うち worst held より RSR が高い: **176件 (44.6%)**
- RSR vs worst gap (mean): -4.1  (p50: -2.4  p75: +9.5)
- IS期間 better%: 42.9%  /  OOS期間 better%: 70.8%

### Better vs Worse than Worst — fwd60d比較

| グループ | N | mean | median | WR | IR |
|---|---|---|---|---|---|
| better_than_worst | 173 | +9.27% | +7.006% | 69.4% | 0.471 |
| worse_than_worst | 218 | +7.738% | +5.355% | 66.1% | 0.485 |
| 全体 | 391 | +8.416% | +5.994% | 67.5% | 0.475 |

### Rank別 fwd60d

| Rank | N | mean | WR |
|---|---|---|---|
| rank4 | 18 | +10.062% | 88.9% |
| rank5 | 7 | +10.577% | 71.4% |
| rank6+ | 1 | -12.046% | 0.0% |

---
## 分析5: Swap-Lite Counterfactual Simulation

ルール: 満杯時に新規シグナルRSR > 最低保有RSR + 0 なら最低保有を売却して新規取得

| 期間 | 指標 | Pattern A | Swap-Lite | Delta |
|---|---|---|---|---|
| IS | CAGR% | 18.12 | 15.29 | -2.83 |
| IS | Sharpe | 0.78 | 0.66 | -0.12 |
| IS | MaxDD% | -16.66 | -16.73 | -0.07 |
| IS | Exposure% | 35.00 | 36.70 | +1.70 |
| IS | Trades/yr | 48.40 | 53.30 | +4.90 |
| OOS | CAGR% | 9.99 | 9.44 | -0.55 |
| OOS | Sharpe | 1.01 | 0.96 | -0.05 |
| OOS | MaxDD% | -10.11 | -10.11 | +0.00 |
| OOS | Exposure% | 27.80 | 27.70 | -0.10 |
| OOS | Trades/yr | 46.70 | 47.70 | +1.00 |

- IS Swap実行回数: 32
- OOS Swap実行回数: 2

---
## 最終結論

### 判定: **B: 構造的アルファ漏出あり**

根拠: IS期間のmissed fwd60d win_rate=66.6% / mean=+8.20% かつ 42.9%のmissedシグナルが保有最下位より高RSR

### Entry Timing vs Slot Optimization — どちらを優先すべきか

#### 優先順位判定: **Entry Timing を優先**

根拠: Swap-Lite IS CAGR改善 = -2.8pp ≤ 0.5pp かつシグナル品質向上の方が安全


| 項目 | Entry Timing | Slot Optimization |
|---|---|---|
| IS リスク | 低 (シグナル減少のみ) | 中 (回転コスト発生) |
| OOS 安定性 | 高 (品質向上は普遍的) | 低 (相場環境依存) |
| 実装複雑度 | 低 (1フラグ変更) | 中 (ロジック追加) |
| PARAMS_LOCKED影響 | なし | あり (max_positions変更不要だが動作変化) |
| 期待CAGR改善 | +3〜7pp (推定) | -2.8pp (実測IS) |