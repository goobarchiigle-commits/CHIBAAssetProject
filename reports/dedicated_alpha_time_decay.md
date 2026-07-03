# Dedicated Alpha Sleeve — Time Decay Allocation WF  (Study 8D)

作成日: 2026-06-14  |  研究専用 / 実装変更禁止

固定: ENTRY=RSR[92,95) d90≤5  EXIT=RSR<90 3日連続

採用条件: WF≥4/5, ΔCAGR>+0.3pp, ΔDD≤+1.5pp, alpha_retention≥90%

**採用Case**: 0件  |  最終判定: **🔬 KEEP RESEARCH**

---
## 1. Executive Summary

- ベースライン (Case A, 25%固定): sl_CAGR=39.2%  ΔDD=+2.46pp  WF=0/5
- 検証軸: time decay alloc で late-phase DD を抑制しながら alpha 維持
- 採用: **0件**

---
## 2. WF サマリ (6 Allocation Cases)

| Case | Allocation | WF | sl_CAGR | ΔCAGR | ΔDD | Calmar | α_ret | 採用 |
|---|---|---|---|---|---|---|---|---|
| A | 25% fixed | 0/5 | 39.2% | +10.86pp | +2.46pp | +0.626 | 100% | ❌ |
| B | 25%→15% after hold≥5d | 1/5 | 26.6% | +5.06pp | -0.32pp | +0.318 | 68% | ❌ |
| C | 25%→10% after hold≥5d | 1/5 | 21.0% | +3.62pp | -1.29pp | +0.706 | 54% | ❌ |
| D | 25%→15% after hold≥3d | 1/5 | 25.2% | +4.65pp | -0.00pp | +0.155 | 64% | ❌ |
| E | 25%→10% after hold≥3d | 1/5 | 18.0% | +2.85pp | -0.78pp | +0.396 | 46% | ❌ |
| F | 20% fixed | 0/5 | 40.9% | +10.37pp | +1.79pp | +0.762 | 104% | ❌ |

---
## 3. Time-weighted Exposure Decomposition

> 「早期」= hold < 5日  「後期」= hold ≥ 5日

| Case | early_exp% | late_exp% | late_exp削減 | late_period_DD | ret_after_d5 | n_decayed |
|---|---|---|---|---|---|---|
| A | 39.8% | 54.4% | +0.0pp | +12.29pp | — | 0 |
| B | 54.7% | 34.8% | -19.6pp | +9.46pp | +35.4% | 15 |
| C | 63.0% | 23.6% | -30.8pp | +8.42pp | +35.4% | 15 |
| D | 50.2% | 36.9% | -17.5pp | +9.77pp | +33.0% | 16 |
| E | 56.8% | 26.2% | -28.2pp | +8.98pp | +33.0% | 16 |
| F | 37.0% | 52.8% | -1.6pp | +11.62pp | — | 0 |

---
## 4. Fold 詳細

| Case | Fold | Regime | sl_CAGR | ΔCAGR | ΔDD | late_DD | α_ret | pass |
|---|---|---|---|---|---|---|---|---|
| A | Fold1 | Bull (+13.3%) | +107.4% | +16.54pp | +4.35pp | +24.46pp | 274% | ❌ |
| A | Fold2 | Bear (-4.7%) | +26.6% | +10.71pp | +0.25pp | +14.08pp | 68% | ❌ |
| A | Fold3 | Bull (+29.8%) | +16.3% | +5.56pp | +5.48pp | +8.36pp | 42% | ❌ |
| A | Fold4 | Bull (+19.7%) | +18.8% | +10.34pp | -1.65pp | +8.72pp | 48% | ❌ |
| A | Fold5 | Bull (+26.8%) | +27.1% | +11.15pp | +3.86pp | +5.81pp | 69% | ❌ |
| B | Fold1 | Bull (+13.3%) | +80.5% | +8.85pp | +0.02pp | +20.13pp | 205% | ✅ |
| B | Fold2 | Bear (-4.7%) | +18.7% | +6.17pp | -2.86pp | +10.97pp | 48% | ❌ |
| B | Fold3 | Bull (+29.8%) | +21.3% | +5.56pp | +1.60pp | +4.42pp | 54% | ❌ |
| B | Fold4 | Bull (+19.7%) | +10.8% | +5.43pp | -1.18pp | +9.02pp | 28% | ❌ |
| B | Fold5 | Bull (+26.8%) | +1.6% | -0.70pp | +0.80pp | +2.75pp | 4% | ❌ |
| C | Fold1 | Bull (+13.3%) | +59.5% | +5.02pp | -2.27pp | +17.84pp | 152% | ✅ |
| C | Fold2 | Bear (-4.7%) | +12.5% | +4.18pp | -2.94pp | +10.89pp | 32% | ❌ |
| C | Fold3 | Bull (+29.8%) | +22.3% | +5.11pp | +0.68pp | +3.18pp | 57% | ❌ |
| C | Fold4 | Bull (+19.7%) | +2.6% | +2.47pp | -1.53pp | +8.65pp | 7% | ❌ |
| C | Fold5 | Bull (+26.8%) | +7.8% | +1.30pp | -0.40pp | +1.55pp | 20% | ❌ |
| D | Fold1 | Bull (+13.3%) | +80.3% | +8.85pp | +0.45pp | +20.56pp | 205% | ✅ |
| D | Fold2 | Bear (-4.7%) | +19.9% | +6.51pp | -2.81pp | +11.02pp | 51% | ❌ |
| D | Fold3 | Bull (+29.8%) | +15.3% | +3.87pp | +2.78pp | +5.66pp | 39% | ❌ |
| D | Fold4 | Bull (+19.7%) | +10.5% | +5.22pp | -1.08pp | +9.00pp | 27% | ❌ |
| D | Fold5 | Bull (+26.8%) | +0.2% | -1.19pp | +0.66pp | +2.61pp | 0% | ❌ |
| E | Fold1 | Bull (+13.3%) | +57.2% | +4.61pp | -1.77pp | +18.34pp | 146% | ✅ |
| E | Fold2 | Bear (-4.7%) | +13.6% | +4.41pp | -2.85pp | +10.98pp | 35% | ❌ |
| E | Fold3 | Bull (+29.8%) | +10.6% | +2.17pp | +2.34pp | +5.22pp | 27% | ❌ |
| E | Fold4 | Bull (+19.7%) | +1.5% | +2.02pp | -1.28pp | +8.77pp | 4% | ❌ |
| E | Fold5 | Bull (+26.8%) | +7.2% | +1.04pp | -0.36pp | +1.59pp | 18% | ❌ |
| F | Fold1 | Bull (+13.3%) | +104.1% | +13.73pp | +3.08pp | +23.19pp | 266% | ❌ |
| F | Fold2 | Bear (-4.7%) | +27.2% | +9.53pp | -1.38pp | +12.45pp | 69% | ❌ |
| F | Fold3 | Bull (+29.8%) | +15.0% | +4.48pp | +4.38pp | +7.26pp | 38% | ❌ |
| F | Fold4 | Bull (+19.7%) | +11.7% | +6.58pp | -0.82pp | +9.55pp | 30% | ❌ |
| F | Fold5 | Bull (+26.8%) | +46.6% | +17.53pp | +3.71pp | +5.66pp | 119% | ❌ |

---
## 5. ΔDD 削減分析

> Case A ΔDD avg から各 Case がどれだけ ΔDD を削減したか

| Case | avg_ΔDD | ΔDD削減 vs A | sl_CAGR削減 | CAGR/DD trade-off |
|---|---|---|---|---|
| A | +2.46pp | +0.00pp | +0.0pp | N/A (DD増) |
| B | -0.32pp | +2.78pp | -12.6pp | -4.53pp_sl_CAGR / pp_ΔDD |
| C | -1.29pp | +3.75pp | -18.2pp | -4.85pp_sl_CAGR / pp_ΔDD |
| D | -0.00pp | +2.46pp | -14.0pp | -5.69pp_sl_CAGR / pp_ΔDD |
| E | -0.78pp | +3.24pp | -21.2pp | -6.54pp_sl_CAGR / pp_ΔDD |
| F | +1.79pp | +0.67pp | +1.7pp | +2.54pp_sl_CAGR / pp_ΔDD |

---
## 6. Failure Analysis


**A** (25% fixed): REJECT — WF=0/5 / ΔDD=+2.46pp

**B** (25%→15% after hold≥5d): REJECT — WF=1/5 / α_ret=68%<90%

**C** (25%→10% after hold≥5d): REJECT — WF=1/5 / α_ret=54%<90%

**D** (25%→15% after hold≥3d): REJECT — WF=1/5 / α_ret=64%<90%

**E** (25%→10% after hold≥3d): REJECT — WF=1/5 / α_ret=46%<90%

**F** (20% fixed): REJECT — WF=0/5 / ΔDD=+1.79pp

---
## 7. Final Recommendation

## `🔬 KEEP RESEARCH`

全6 Case 採用基準未達。最良: **Case A** (25% fixed)

- sl_CAGR=39.2%  ΔCAGR=+10.86pp  ΔDD=+2.46pp  WF=0/5

**バインディング制約**: WF=0/5 / ΔDD=+2.46pp

**構造的発見:**
- Decay alloc は late_exp を削減するが、sl_CAGR も同時に低下
- Case A ΔDD=+2.46pp → best Case A ΔDD=+2.46pp
  削減幅=+0.00pp に対する alpha_retention=100%

次ステップ候補:
- Study 8E: Case C (3日連続 EXIT) + 25% 固定 でポートフォリオ集中度を下げる別アプローチ
- ΔDD の根本原因は sleeve 単体の volatility: max_pos=1 の single-stock concentration
- 解決策: sleeve max_pos=2 で集中度分散 or cap=15% 固定 (decay なし)
- Study 8 シリーズの総括: RSR90-94 の alpha は実在。実装可能な risk control が課題