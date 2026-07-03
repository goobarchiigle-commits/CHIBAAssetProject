# Dedicated Alpha Sleeve — Entry Quality Compression  (Study 8B)

作成日: 2026-06-14  |  研究専用 / 実装変更禁止

固定: EXIT=RSR<90, cap=equity×25%, max_pos=1

採用条件: WF≥4/5, sl_CAGR≥30.0%, ΔCAGR>+0.3pp, ΔDD≤+1.5pp, trigger≤8.0/yr

**採用Case**: 0件  |  最終判定: **🔬 KEEP RESEARCH**

---
## 1. Executive Summary

- ベースライン avg CAGR: **+5.1%** (production P0)
- Case A = Study8 Case B (baseline: RSR[90,95) d90≤5, EXIT=RSR<90)
- 採用: **0件**
- Case A 参照: sl_CAGR=36.9%  ΔCAGR=+9.84pp  trig=12.6/yr  WF=0/5

---
## 2. WF サマリ (6 Entry Case)

| Case | Entry条件 | WF | sl_CAGR | ΔCAGR | ΔDD | trig/yr | entry_edge | 採用 |
|---|---|---|---|---|---|---|---|---|
| A | RSR[90,95) d90≤5 | 0/5 | 36.9% | +9.84pp | +2.18pp | 12.6 | +0.02pp | ❌ |
| B | RSR[92,95) d90≤5 | 0/5 | 40.0% | +10.53pp | +1.56pp | 12.6 | -1.21pp | ❌ |
| C | RSR[90,95) d90≤3 | 0/5 | 38.3% | +10.38pp | +2.49pp | 13.4 | +0.54pp | ❌ |
| D | RSR[92,95) d90≤3 | 0/5 | 38.9% | +10.07pp | +1.64pp | 13.6 | -1.31pp | ❌ |
| E | RSR[93,95) d90≤2 | 0/5 | 0.0% | -1.02pp | -1.56pp | 0.0 | — | ❌ |
| F | RSR[92,95) d90≤2 | 0/5 | 39.7% | +10.33pp | +1.46pp | 13.0 | -1.30pp | ❌ |

---
## 3. Alpha Loss per Trigger Removed (vs Case A)

> 基準 Case A の trigger/yr, ΔCAGR からどれだけ alpha を失うか。
> alpha_loss = ΔΔCalmar / Δtrigger（Calmar の純変化 / 削除された trigger 数）

| Case | Δtrig/yr vs A | ΔΔCalmar vs A | alpha_loss / trigger | 評価 |
|---|---|---|---|---|
| A | +0.0 | +0.000 | N/A (trigger増) | 参考外 |
| B | +0.0 | +0.039 | N/A (trigger増) | 参考外 |
| C | -0.8 | +0.024 | N/A (trigger増) | 参考外 |
| D | -1.0 | +0.013 | N/A (trigger増) | 参考外 |
| E | +12.6 | -0.656 | +0.0521 Calmar/trigger | ❌ -0.0521 Calmar loss |
| F | -0.4 | -0.024 | N/A (trigger増) | 参考外 |

---
## 4. Entry Quality Audit

| Case | sl_CAGR | sl_Calmar | hit_rate | PF | avg_hold | days_to_exit | cash_util | entries | overlap | entry_edge |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 36.9% | +0.647 | 58.7% | 36790.208 | 41.0d | 41.1d | 98.5% | 12 | 13d | +0.02pp |
| B | 40.0% | +0.686 | 47.3% | 4.247 | 17.5d | 17.3d | 80.2% | 12 | 13d | -1.21pp |
| C | 38.3% | +0.671 | 57.5% | 36790.194 | 39.8d | 39.9d | 96.4% | 13 | 13d | +0.54pp |
| D | 38.9% | +0.660 | 41.5% | 3.224 | 16.3d | 16.4d | 76.1% | 13 | 13d | -1.31pp |
| E | 0.0% | -0.009 | 0.0% | 0.000 | 0.0d | 0.0d | 0.0% | 0 | 0d | — |
| F | 39.7% | +0.623 | 42.7% | 3.214 | 16.3d | 16.4d | 73.5% | 13 | 13d | -1.30pp |

---
## 5. Fold 詳細

| Case | Fold | OOS年 | Regime | base | sl_CAGR | ΔCAGR | ΔDD | trig/yr | entry_edge | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Fold1 | 2021 | Bull (+13.3%) | +32.1% | +108.3% | +16.77pp | +4.31pp | 2.1 | +0.81pp | ❌ |
| A | Fold2 | 2022 | Bear (-4.7%) | -6.3% | +26.4% | +10.69pp | -0.24pp | 18.6 | +0.59pp | ❌ |
| A | Fold3 | 2023 | Bull (+29.8%) | +2.1% | +10.9% | +3.50pp | +1.96pp | 15.4 | -1.48pp | ❌ |
| A | Fold4 | 2024 | Bull (+19.7%) | -6.0% | +5.7% | +4.94pp | +1.11pp | 18.5 | +0.38pp | ❌ |
| A | Fold5 | 2025 | Bull (+26.8%) | +3.6% | +33.0% | +13.28pp | +3.74pp | 8.3 | -0.22pp | ❌ |
| B | Fold1 | 2021 | Bull (+13.3%) | +32.1% | +113.5% | +17.15pp | +4.52pp | 3.1 | -5.25pp | ❌ |
| B | Fold2 | 2022 | Bear (-4.7%) | -6.3% | +37.3% | +13.98pp | -0.12pp | 17.6 | -3.04pp | ❌ |
| B | Fold3 | 2023 | Bull (+29.8%) | +2.1% | +4.9% | +1.18pp | +0.80pp | 20.5 | +0.37pp | ❌ |
| B | Fold4 | 2024 | Bull (+19.7%) | -6.0% | +10.0% | +6.61pp | -1.16pp | 13.4 | +0.49pp | ❌ |
| B | Fold5 | 2025 | Bull (+26.8%) | +3.6% | +34.2% | +13.74pp | +3.75pp | 8.3 | +1.40pp | ❌ |
| C | Fold1 | 2021 | Bull (+13.3%) | +32.1% | +108.3% | +16.77pp | +4.31pp | 2.1 | +0.81pp | ❌ |
| C | Fold2 | 2022 | Bear (-4.7%) | -6.3% | +26.4% | +10.69pp | -0.24pp | 18.6 | +0.59pp | ❌ |
| C | Fold3 | 2023 | Bull (+29.8%) | +2.1% | +17.8% | +6.00pp | +3.54pp | 20.5 | +0.08pp | ❌ |
| C | Fold4 | 2024 | Bull (+19.7%) | -6.0% | +6.2% | +5.15pp | +1.12pp | 17.5 | +1.42pp | ❌ |
| C | Fold5 | 2025 | Bull (+26.8%) | +3.6% | +33.0% | +13.27pp | +3.73pp | 8.3 | -0.22pp | ❌ |
| D | Fold1 | 2021 | Bull (+13.3%) | +32.1% | +113.5% | +17.15pp | +4.52pp | 3.1 | -5.25pp | ❌ |
| D | Fold2 | 2022 | Bear (-4.7%) | -6.3% | +33.8% | +12.87pp | -0.12pp | 25.8 | -3.23pp | ❌ |
| D | Fold3 | 2023 | Bull (+29.8%) | +2.1% | +7.0% | +1.99pp | +0.83pp | 18.4 | +0.37pp | ❌ |
| D | Fold4 | 2024 | Bull (+19.7%) | -6.0% | +5.0% | +4.53pp | -0.91pp | 13.4 | +0.57pp | ❌ |
| D | Fold5 | 2025 | Bull (+26.8%) | +3.6% | +35.2% | +13.83pp | +3.86pp | 7.3 | +0.99pp | ❌ |
| E | Fold1 | 2021 | Bull (+13.3%) | +32.1% | +0.0% | -6.07pp | -3.40pp | 0.0 | — | ❌ |
| E | Fold2 | 2022 | Bear (-4.7%) | -6.3% | +0.0% | +0.95pp | -2.08pp | 0.0 | — | ❌ |
| E | Fold3 | 2023 | Bull (+29.8%) | +2.1% | +0.0% | -0.33pp | -0.46pp | 0.0 | — | ❌ |
| E | Fold4 | 2024 | Bull (+19.7%) | -6.0% | +0.0% | +0.94pp | -1.55pp | 0.0 | — | ❌ |
| E | Fold5 | 2025 | Bull (+26.8%) | +3.6% | +0.0% | -0.59pp | -0.31pp | 0.0 | — | ❌ |
| F | Fold1 | 2021 | Bull (+13.3%) | +32.1% | +113.5% | +17.15pp | +4.52pp | 3.1 | -5.25pp | ❌ |
| F | Fold2 | 2022 | Bear (-4.7%) | -6.3% | +38.1% | +14.22pp | -0.12pp | 24.8 | -3.48pp | ❌ |
| F | Fold3 | 2023 | Bull (+29.8%) | +2.1% | +2.3% | +0.10pp | +0.78pp | 19.5 | +1.83pp | ❌ |
| F | Fold4 | 2024 | Bull (+19.7%) | -6.0% | +9.6% | +6.32pp | -1.68pp | 11.3 | -0.39pp | ❌ |
| F | Fold5 | 2025 | Bull (+26.8%) | +3.6% | +35.0% | +13.87pp | +3.82pp | 6.2 | +0.82pp | ❌ |

---
## 6. Failure Analysis


**A** (RSR[90,95) d90≤5): REJECT — WF=0/5 / ΔDD=+2.18pp / trig=12.6/yr

**B** (RSR[92,95) d90≤5): REJECT — WF=0/5 / ΔDD=+1.56pp / trig=12.6/yr

**C** (RSR[90,95) d90≤3): REJECT — WF=0/5 / ΔDD=+2.49pp / trig=13.4/yr

**D** (RSR[92,95) d90≤3): REJECT — WF=0/5 / ΔDD=+1.64pp / trig=13.6/yr

**E** (RSR[93,95) d90≤2): REJECT — WF=0/5 / sl_CAGR=0.0% < 30.0% / ΔCAGR=-1.02pp

**F** (RSR[92,95) d90≤2): REJECT — WF=0/5 / trig=13.0/yr

---
## 7. Compression Efficiency サマリ

> d90 と RSR 下限をどの方向で絞ると alpha 損失が最小か。

**d90 絞り (RSR≥90, d90: 5→3→2):**
  A: trig=12.6  sl_CAGR=36.9%  entry_edge=+0.02pp
  C: trig=13.4  sl_CAGR=38.3%  entry_edge=+0.54pp
  F: trig=13.0  sl_CAGR=39.7%  entry_edge=-1.30pp

**RSR下限絞り (d90≤5, RSR: 90→92→93):**
  A: trig=12.6  sl_CAGR=36.9%  entry_edge=+0.02pp
  B: trig=12.6  sl_CAGR=40.0%  entry_edge=-1.21pp
  E: trig=0.0  sl_CAGR=0.0%  entry_edge=—

---
## 8. Final Recommendation

## `🔬 KEEP RESEARCH`

全6 Case が採用基準未達。最良: **Case B** (RSR[92,95) d90≤5)

- sl_CAGR=40.0%  ΔCAGR=+10.53pp  trig=12.6/yr  WF=0/5

**バインディング制約**: WF=0/5 / ΔDD=+1.56pp / trig=12.6/yr

継続研究候補:
- d90 + RSR 絞りでは alpha 損失が先に来る → 別次元の filter 探索が必要
- Study 8C: ΔDD を減らす capital ratio 削減 (25%→15%)
- Study 8D: entry filter に state=EARLY_UP 必須を追加
- Study 8E: Bear フォールド (2022) を除外した Bull 時限定採用の再評価