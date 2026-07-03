# Entry Timing A/B テスト結果

作成日: 2026-06-02  |  IS: 2018-01-01〜2024-12-31  OOS: 2025-01-01〜2025-12-31

**全モード共通: Pattern A 資本配分ロジック (max_positions=3)**

| Mode | 設定 |
|---|---|
| A | A Baseline (ET完全OFF) |
| B | B ET boost only (block=OFF, boost=0.06) |
| C | C block_low_confidence (boost=0, block=ON) |
| D | D Full ET (boost=0.06, block=ON) |

---
## IS 2018-2024 比較

| Mode | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | WinRate | Exposure | Missed |
|---|---|---|---|---|---|---|---|---|---|
| A | +18.1% | 0.779 | 1.001 | -16.7% | 1.087 | 2.408 | 56.0% | 35.0% | 371 |
| B | +18.1% | 0.779 | 1.001 | -16.7% | 1.087 | 2.408 | 56.0% | 35.0% | 371 |
| C | +9.1% | 0.879 | 0.942 | -12.0% | 0.758 | 4.231 | 66.7% | 14.4% | 8 |
| D | +9.1% | 0.879 | 0.942 | -12.0% | 0.758 | 4.231 | 66.7% | 14.4% | 8 |

### IS 年次リターン (%)

| 年 | A | B | C | D |
|---|---|---|---|---|
| 2020 | +7.8% | +7.8% | -0.3% | -0.3% |
| 2021 | +15.8% | +15.8% | +2.0% | +2.0% |
| 2022 | +17.6% | +17.6% | -2.0% | -2.0% |
| 2023 | +23.4% | +23.4% | +26.0% | +26.0% |
| 2024 | +11.1% | +11.1% | +16.6% | +16.6% |

### Δ vs Mode A (IS)

| Mode | ΔCAGR | ΔSharpe | ΔMaxDD | ΔCalmar | ΔMissed |
|---|---|---|---|---|---|
| B | +0.00pp | +0.0000 | +0.00pp | +0.0000 | +0 |
| C | -9.00pp | +0.1000 | +4.63pp | -0.3290 | -363 |
| D | -9.00pp | +0.1000 | +4.63pp | -0.3290 | -363 |

---
## OOS 2025 比較

| Mode | CAGR | Sharpe | MaxDD | Calmar | Missed |
|---|---|---|---|---|---|
| A | +10.0% | 1.007 | -10.1% | 0.988 | 23 |
| B | +10.0% | 1.007 | -10.1% | 0.988 | 23 |
| C | +4.9% | 0.722 | -4.2% | 1.165 | 2 |
| D | +4.9% | 0.722 | -4.2% | 1.165 | 2 |

### Δ vs Mode A (OOS)

| Mode | ΔCAGR | ΔSharpe | ΔMaxDD | ΔCalmar | ΔMissed |
|---|---|---|---|---|---|
| B | +0.00pp | +0.0000 | +0.00pp | +0.0000 | +0 |
| C | -5.05pp | -0.2850 | +5.87pp | +0.1770 | -21 |
| D | -5.05pp | -0.2850 | +5.87pp | +0.1770 | -21 |

---
## 分析

### block_low_confidence 効果 (Mode C: block_only)

- IS ΔCAGR: -9.00pp
- IS ΔMissed: -363 件 (blockedによるmissed増加)
- IS ΔSharpe: +0.1000
- OOS ΔCAGR: -5.05pp

### ET boost 効果 (Mode B - Mode A)

- IS ΔCAGR: +0.00pp (boost=0.06 only)

### 結論

block_low_confidence: **Mode C IS改善なし → block_low_confidence の単独効果は限定的**

full ET: **Mode D IS改善なし → Full ET の複合効果も限定的**

### 次のアクション

| 優先度 | アクション | 条件 |
|---|---|---|
| 1 | Mode C/D の Walk-Forward (5-Fold) 検証 | IS Calmar改善が確認された場合 |
| 2 | block_low_confidence=true を strategy.yaml に反映 | WF 3/5以上でA勝利の場合 |
| 3 | ET boost_weight sweep (0.06→0.10→0.15) | Mode B/D の IS改善が >0.5pp の場合 |