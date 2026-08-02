# Study9 Standalone Validation WF

作成日: 2026-06-20  |  研究専用 / 実装変更禁止

Entry: RSR[92,95) d90≤5  Capital: SLEEVE×0.25  MaxPos: 1  Regime: なし

採用条件: WF>=4/5 AND avg_calmar > avg_calmar_A

---
## 1. WF サマリ

| Case | WF | avg_CAGR | avg_DD | avg_Calmar | Calmar_A | Calmar_lift | avg_PF | avg_hold | ADOPT |
|---|---|---|---|---|---|---|---|---|---|
| **A** baseline  (exit<90) | —/5 | -9.45% | -28.71% | -0.315 | — | — | 1.131 | 16.5d | baseline |
| B: +slope≤5  (exit<90) | 5/5 | +38.81% | -30.60% | 1.303 | -0.315 | +1.617 | 1.997 | 35.0d | **ADOPT** |
| C: +slope≤5  +exit<87 | 5/5 | +29.78% | -30.20% | 1.175 | -0.315 | +1.490 | 14.695 | 50.2d | **ADOPT** |
| D: +slope≤5  +profit_lock+8% | 4/5 | +6.34% | -27.71% | 0.267 | -0.315 | +0.581 | 1.508 | 11.4d | **ADOPT** |
| E: +slope≤5  +exit<87  +profit_lock+8% | 5/5 | +25.23% | -25.90% | 1.505 | -0.315 | +1.819 | 2.743 | 16.2d | **ADOPT** |

---
## 2. Fold × Case マトリクス (Calmar)

> Fold pass: calmar_X ≥ calmar_A (同 Fold)

| Fold | Regime | A (calmar) | B: calmar | pass | C: calmar | pass | D: calmar | pass | E: calmar | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold1 | Bull (+13.3%) | -0.104 | 0.630 | ✅ | 1.109 | ✅ | 1.608 | ✅ | 2.692 | ✅ |
| Fold2 | Bear (-4.7%) | -0.230 | 0.548 | ✅ | 0.817 | ✅ | -0.453 | ❌ | -0.225 | ✅ |
| Fold3 | Bull (+29.8%) | 0.336 | 1.029 | ✅ | 0.996 | ✅ | 1.242 | ✅ | 1.114 | ✅ |
| Fold4 | Bull (+19.7%) | -0.886 | 1.027 | ✅ | 2.966 | ✅ | -0.753 | ✅ | -0.550 | ✅ |
| Fold5 | Bull (+26.8%) | -0.689 | 3.280 | ✅ | -0.011 | ✅ | -0.310 | ✅ | 4.493 | ✅ |

---
## 3. Fold × Case マトリクス (CAGR%)

| Fold | Regime | CAGR_A | CAGR_B | ΔCAGR | CAGR_C | ΔCAGR | CAGR_D | ΔCAGR | CAGR_E | ΔCAGR |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold1 | Bull (+13.3%) | -4.46% | +27.56% | +32.02pp | +42.06% | +46.52pp | +55.12% | +59.58pp | +94.66% | +99.12pp |
| Fold2 | Bear (-4.7%) | -5.73% | +16.73% | +22.46pp | +22.24% | +27.97pp | -14.24% | -8.51pp | -7.10% | -1.37pp |
| Fold3 | Bull (+29.8%) | +7.24% | +21.13% | +13.89pp | +19.79% | +12.55pp | +23.43% | +16.19pp | +22.63% | +15.39pp |
| Fold4 | Bull (+19.7%) | -31.70% | +28.38% | +60.08pp | +65.30% | +97.00pp | -27.00% | +4.70pp | -19.09% | +12.61pp |
| Fold5 | Bull (+26.8%) | -12.62% | +100.24% | +112.86pp | -0.48% | +12.14pp | -5.60% | +7.02pp | +35.03% | +47.65pp |

---
## 4. Exit 内訳

| Case | n_trades | PROFIT_LOCK | RSR_EXIT | win_rate | avg_win | avg_loss | avg_hold | s5_win% |
|---|---|---|---|---|---|---|---|---|
| A: baseline  (exit<90) | 46 | — | 45(98%) | 44% | ¥+35,273 | ¥-31,686 | 15.2d | 50% |
| B: +slope≤5  (exit<90) | 34 | — | 33(97%) | 35% | ¥+208,058 | ¥-66,979 | 29.6d | 0% |
| C: +slope≤5  +exit<87 | 24 | — | 23(96%) | 46% | ¥+260,164 | ¥-115,598 | 42.6d | 0% |
| D: +slope≤5  +profit_lock+8% | 55 | 10(18%) | 44(80%) | 47% | ¥+74,175 | ¥-54,533 | 10.8d | 0% |
| E: +slope≤5  +exit<87  +profit_lock+8% | 43 | 12(28%) | 30(70%) | 54% | ¥+105,942 | ¥-76,355 | 15.3d | 0% |

---
## 5. Case 分析

### B vs A: slope≤5 フィルター効果
- Calmar lift: +1.617  WF: 5/5  avg_hold: 35.0d
- 解釈: slope>5 entry 除外の純効果 (exit 変更なし)

### C vs B: exit 87 効果 (slope≤5 維持)
- Calmar vs B: -0.128  WF: 5/5  avg_hold: 50.2d
- 解釈: exit thr 引下げ(90→87) による保有延長効果

### D vs B: profit_lock+8% 効果 (slope≤5 維持、exit<90)
- Calmar vs B: -1.036  WF: 4/5  avg_hold: 11.4d  avg PROFIT_LOCK: 1.8/fold
- 解釈: +8%到達で即利食い → avg_win 変化、DD 影響

### E vs B: exit87 + profit_lock 複合効果
- Calmar vs B: +0.202  WF: 5/5
- 解釈: 両方組み合わせが相乗効果を生むか、または干渉するか

---
## 6. 最終判定

| Case | WF | Calmar_lift | ADOPT |
|---|---|---|---|
| B: +slope≤5  (exit<90) | 5/5 | +1.617 | **ADOPT** |
| C: +slope≤5  +exit<87 | 5/5 | +1.490 | **ADOPT** |
| D: +slope≤5  +profit_lock+8% | 4/5 | +0.581 | **ADOPT** |
| E: +slope≤5  +exit<87  +profit_lock+8% | 5/5 | +1.819 | **ADOPT** |

### ADOPT: B, C, D, E

best = **E**: +slope≤5  +exit<87  +profit_lock+8%
- WF=5/5  Calmar_lift=+1.819  avg_Calmar=1.505
- 実装推奨 (ASK_FIRST): standalone sleeve にこの設定を適用