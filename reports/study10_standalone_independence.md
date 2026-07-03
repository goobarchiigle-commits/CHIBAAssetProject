# Study10 Standalone Independence Audit

作成日: 2026-06-20  |  研究専用 / 実装変更禁止

A: production (RSR42, max_pos=3, capital=¥3,000,000)
B: standalone  (RSR[90,94) d90≤2 slope≤5 exit<87 hold≥3 capital=¥750,000)
C: combined    (A+B independent, total=¥3,750,000)

---
## 1. 基本性能 (Fold別)

| Fold | Regime | CAGR_A | DD_A | Calmar_A | CAGR_B | DD_B | Calmar_B | CAGR_C | DD_C | Calmar_C |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold1 2021 | Bull (+13.3%) | +32.10% | -20.11% | 1.596 | +77.64% | -34.63% | 2.242 | +40.34% | -19.62% | 2.056 |
| Fold2 2022 | Bear (-4.7%) | -6.33% | -13.83% | -0.458 | +13.82% | -32.25% | 0.429 | -1.52% | -12.32% | -0.123 |
| Fold3 2023 | Bull (+29.8%) | +2.08% | -2.88% | 0.724 | +43.22% | -35.60% | 1.214 | +13.51% | -13.29% | 1.016 |
| Fold4 2024 | Bull (+19.7%) | -6.03% | -10.37% | -0.582 | +88.77% | -21.04% | 4.218 | +26.17% | -14.34% | 1.826 |
| Fold5 2025 | Bull (+26.8%) | +3.57% | -1.95% | 1.831 | +194.93% | -34.35% | 5.675 | +98.18% | -19.76% | 4.968 |
| **avg** | — | **+5.08%** | **-9.83%** | **0.622** | **+83.68%** | **-31.57%** | **2.756** | **+35.34%** | **-15.87%** | **1.949** |

---
## 2. 独立性 (Fold別)

| Fold | corr_daily | corr_weekly | corr_dd | tail_corr | tail_dep |
|---|---|---|---|---|---|
| Fold1 | 0.1445 | 0.0737 | 0.2799 | 0.0133 | 0.3934 |
| Fold2 | 0.0821 | 0.1373 | 0.0050 | 0.0445 | 0.2701 |
| Fold3 | 0.0304 | -0.0194 | 0.2437 | 0.0205 | 0.2770 |
| Fold4 | 0.2088 | 0.4336 | 0.7266 | 0.0961 | 0.2591 |
| Fold5 | 0.0221 | 0.0085 | 0.2317 | -0.0290 | 0.2652 |
| **avg** | **0.0976** | **0.1267** | **0.2974** | **0.0291** | **0.2930** |

---
## 3. 組合せ価値

| Fold | alpha_lift | Calmar_C vs A | combined_DD vs A_DD |
|---|---|---|---|
| Fold1 | -0.03pp | +0.460 | +0.49pp |
| Fold2 | -0.01pp | +0.335 | +1.51pp |
| Fold3 | -0.03pp | +0.292 | -10.41pp |
| Fold4 | -0.21pp | +2.408 | -3.97pp |
| Fold5 | -0.86pp | +3.137 | -17.81pp |
| **avg** | **-0.23pp** | **+1.326** | **-6.04pp** |

---
## 4. 保有関係・補完性

| Fold | overlap% | cond_overlap% | P(hB|A_loss)% | P(hB|A_dd)% | P(B+|A-)% | P(B+|Add)% | P(A+|B-)% |
|---|---|---|---|---|---|---|---|
| Fold1 | 83.6 | 88.7 | 89.7 | 88.0 | 28.2 | 46.7 | 26.2 |
| Fold2 | 23.0 | 96.6 | 96.9 | 92.2 | 25.0 | 45.5 | 10.1 |
| Fold3 | 20.8 | 81.0 | 84.4 | — | 28.1 | 0.0 | 9.5 |
| Fold4 | 18.9 | 92.0 | 95.8 | 76.2 | 37.5 | 42.6 | 8.8 |
| Fold5 | 9.1 | 73.3 | 75.0 | — | 8.3 | 0.0 | 7.6 |
| **avg** | **31.1** | **86.3** | **88.4** | **85.5** | **25.4** | **27.0** | **12.4** |

---
## 5. Tail 構造 (全OOS集計)

- top10_trade_share_A: 314.7%
- top10_trade_share_B: 134.2%
- winner_overlap (top10 共通銘柄): 2
- n_top10_A: 10  n_top10_B: 10

---
## 6. Capacity (Standalone B)

| Fold | notional_util% | capital_idle% | miss_signals | lot_round_loss% |
|---|---|---|---|---|
| Fold1 | 103.9 | 10.6 | 64 | 20.28 |
| Fold2 | 98.2 | 5.3 | 49 | 20.28 |
| Fold3 | 110.7 | 6.9 | 95 | 20.28 |
| Fold4 | 114.3 | 14.7 | 119 | 20.28 |
| Fold5 | 131.6 | 14.8 | 75 | 20.28 |

---
## 7. 寄与分解 (avg across folds)

| Component | value |
|---|---|
| prod_component (w_A × CAGR_A) | +4.06pp |
| sl_component   (w_B × CAGR_B) | +16.74pp |
| weighted_CAGR  (naive sum)     | +20.80pp |
| combined_CAGR  (actual)        | +35.34pp |
| interaction    (C - weighted)  | +14.54pp |
| ├ diversification_gain (DD↓)   | +0.00pp |
| └ timing_gain  (residual+)     | +14.54pp |
| cash_drag (approx)             | 0.00pp |

---
## 8. TYPE 分類

### strategy_type: TYPE_A

判定根拠:
- avg corr_daily = 0.0976
- avg combined_calmar = 1.949 vs prod_calmar = 0.622
- avg alpha_lift = -0.23pp
- avg P(B+|A-) = 25.4%
- avg interaction = +14.54pp

定義: independent_alpha: corr<0.30 AND combined_calmar改善

---
## 9. 最終判定

| 指標 | 値 |
|---|---|
| **strategy_type** | **TYPE_A** |
| **combined_efficiency** | **3.132** (combined_calmar / prod_calmar) |
| adopt | ❌ NO |
| **deploy_recommendation** | CONDITIONAL — 独立alpha条件は満たすが combined_calmar/DD 基準未達。10%試験配分を検討。 |
| **capital_recommendation** | **10%** |