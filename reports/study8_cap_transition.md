# Study8 Adaptive CAP Transition WF — Persistence Isolation

作成日: 2026-06-16  |  解析専用 / 実装変更禁止

固定: 2-slot 70/30 no-refill / ENTRY RSR[92,95) d90≤5 / EXIT RSR<90

CAP_HI=20% / CAP_LO=15% / 復帰条件=DD解除+10d経過

採用条件: WF≥4/5, ΔCAGR>+0.3pp, ΔDD≤+1.5pp, α_ret≥90.0%

**採用**: 0件  | 判定: **🔬 RESEARCH**

---
## 1. WF サマリ

| Case | 説明 | WF | ΔCAGR | ΔDD | sl_CAGR | α_ret | avg_cap | avg_sw | state_persist | 採用 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 固定15% (anchor) | 3/5 | +4.43pp | +0.57pp | +33.0% | — | 15.0% | 0.0 | 100.0% | ❌ |
| B | DD>5%  即時 | 3/5 | +4.46pp | +0.58pp | +33.7% | 102.1% | 15.8% | 8.0 | 76.4% | ❌ |
| C | DD>7%  即時 | 3/5 | +4.55pp | +0.58pp | +34.1% | 103.2% | 16.5% | 12.2 | 61.5% | ❌ |
| D | DD>5%  3d継続 | 3/5 | +5.51pp | +0.71pp | +40.0% | 121.0% | 16.3% | 5.6 | 77.0% | ❌ |
| E | DD>7%  3d継続 | 3/5 | +5.87pp | +0.70pp | +42.9% | 129.9% | 17.3% | 7.6 | 56.7% | ❌ |
| F | DD>5%  5d継続 | 3/5 | +5.51pp | +0.71pp | +40.1% | 121.3% | 16.4% | 5.6 | 77.0% | ❌ |
| G | DD>7%  5d継続 | 3/5 | +6.43pp | +0.96pp | +44.9% | 135.7% | 17.0% | 7.4 | 68.7% | ❌ |

---
## 2. Fold 詳細

| Case | Fold | OOS | Regime | ΔCAGR | ΔDD | sl_CAGR | avg_cap | days_lo | switches | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Fold1 | 2021 | Bull (+13.3%) | +7.87pp | -0.32pp | +100.3% | 15.0% | 245d | 0 | ✅ |
| A | Fold2 | 2022 | Bear (-4.7%) | +7.47pp | -2.15pp | +36.1% | 15.0% | 244d | 0 | ✅ |
| A | Fold3 | 2023 | Bull (+29.8%) | -1.31pp | +2.08pp | -3.4% | 15.0% | 246d | 0 | ❌ |
| A | Fold4 | 2024 | Bull (+19.7%) | +6.92pp | +1.39pp | +24.4% | 15.0% | 245d | 0 | ✅ |
| A | Fold5 | 2025 | Bull (+26.8%) | +1.18pp | +1.84pp | +7.9% | 15.0% | 243d | 0 | ❌ |
| B | Fold1 | 2021 | Bull (+13.3%) | +8.03pp | -0.29pp | +103.3% | 16.3% | 182d | 16 | ✅ |
| B | Fold2 | 2022 | Bear (-4.7%) | +7.47pp | -2.13pp | +36.4% | 15.6% | 215d | 6 | ✅ |
| B | Fold3 | 2023 | Bull (+29.8%) | -1.31pp | +2.09pp | -3.4% | 15.1% | 242d | 4 | ❌ |
| B | Fold4 | 2024 | Bull (+19.7%) | +6.92pp | +1.40pp | +24.6% | 16.4% | 178d | 7 | ✅ |
| B | Fold5 | 2025 | Bull (+26.8%) | +1.19pp | +1.85pp | +7.9% | 15.8% | 203d | 7 | ❌ |
| C | Fold1 | 2021 | Bull (+13.3%) | +8.03pp | -0.29pp | +103.3% | 16.4% | 176d | 16 | ✅ |
| C | Fold2 | 2022 | Bear (-4.7%) | +7.59pp | -2.13pp | +37.1% | 15.8% | 205d | 20 | ✅ |
| C | Fold3 | 2023 | Bull (+29.8%) | -1.33pp | +2.08pp | -3.5% | 15.5% | 221d | 10 | ❌ |
| C | Fold4 | 2024 | Bull (+19.7%) | +6.92pp | +1.39pp | +24.5% | 16.7% | 160d | 7 | ✅ |
| C | Fold5 | 2025 | Bull (+26.8%) | +1.55pp | +1.84pp | +9.2% | 17.9% | 103d | 8 | ❌ |
| D | Fold1 | 2021 | Bull (+13.3%) | +11.31pp | +1.22pp | +132.6% | 16.9% | 151d | 10 | ✅ |
| D | Fold2 | 2022 | Bear (-4.7%) | +9.73pp | -2.13pp | +43.2% | 15.9% | 200d | 6 | ✅ |
| D | Fold3 | 2023 | Bull (+29.8%) | -1.34pp | +1.86pp | -2.9% | 15.3% | 230d | 2 | ❌ |
| D | Fold4 | 2024 | Bull (+19.7%) | +6.88pp | +0.90pp | +20.2% | 16.8% | 159d | 5 | ✅ |
| D | Fold5 | 2025 | Bull (+26.8%) | +0.97pp | +1.68pp | +6.7% | 16.5% | 168d | 5 | ❌ |
| E | Fold1 | 2021 | Bull (+13.3%) | +12.98pp | +1.50pp | +148.8% | 17.5% | 121d | 7 | ✅ |
| E | Fold2 | 2022 | Bear (-4.7%) | +10.38pp | -2.18pp | +44.3% | 16.5% | 172d | 18 | ✅ |
| E | Fold3 | 2023 | Bull (+29.8%) | -1.62pp | +1.79pp | -3.6% | 15.8% | 204d | 6 | ❌ |
| E | Fold4 | 2024 | Bull (+19.7%) | +6.87pp | +0.75pp | +19.2% | 17.1% | 140d | 5 | ✅ |
| E | Fold5 | 2025 | Bull (+26.8%) | +0.76pp | +1.63pp | +6.0% | 19.6% | 20d | 2 | ❌ |
| F | Fold1 | 2021 | Bull (+13.3%) | +11.33pp | +1.22pp | +133.0% | 17.1% | 142d | 10 | ✅ |
| F | Fold2 | 2022 | Bear (-4.7%) | +9.73pp | -2.13pp | +43.3% | 16.0% | 194d | 6 | ✅ |
| F | Fold3 | 2023 | Bull (+29.8%) | -1.34pp | +1.86pp | -2.9% | 15.4% | 228d | 2 | ❌ |
| F | Fold4 | 2024 | Bull (+19.7%) | +6.88pp | +0.90pp | +20.2% | 16.8% | 157d | 5 | ✅ |
| F | Fold5 | 2025 | Bull (+26.8%) | +0.97pp | +1.68pp | +6.7% | 16.7% | 162d | 5 | ❌ |
| G | Fold1 | 2021 | Bull (+13.3%) | +12.98pp | +1.50pp | +148.8% | 17.7% | 114d | 7 | ✅ |
| G | Fold2 | 2022 | Bear (-4.7%) | +10.28pp | -2.18pp | +43.9% | 17.1% | 144d | 16 | ✅ |
| G | Fold3 | 2023 | Bull (+29.8%) | -1.54pp | +1.80pp | -3.4% | 15.9% | 200d | 6 | ❌ |
| G | Fold4 | 2024 | Bull (+19.7%) | +9.14pp | +0.77pp | +27.5% | 17.5% | 122d | 3 | ✅ |
| G | Fold5 | 2025 | Bull (+26.8%) | +1.27pp | +2.90pp | +7.4% | 16.7% | 162d | 5 | ❌ |

---
## 3. Persistence vs Threshold Analysis

> 切替条件の「閾値」×「継続日数」マトリクス

| | DD>5% 即時(B) | DD>5% 3d(D) | DD>5% 5d(F) |
|---|---|---|---|
| avg ΔCAGR | +4.46pp | +5.51pp | +5.51pp |
| avg ΔDD | +0.58pp | +0.71pp | +0.71pp |
| avg switches | 8.0 | 5.6 | 5.6 |
| WF | 3 | 3 | 3 |

| | DD>7% 即時(C) | DD>7% 3d(E) | DD>7% 5d(G) |
|---|---|---|---|
| avg ΔCAGR | +4.55pp | +5.87pp | +6.43pp |
| avg ΔDD | +0.58pp | +0.70pp | +0.96pp |
| avg switches | 12.2 | 7.6 | 7.4 |
| WF | 3 | 3 | 3 |

---
## 4. Transition Efficiency Audit

| Case | alpha_loss/switch | dd_saved/switch | switch_half_life | transition_eff |
|---|---|---|---|---|
| A | +0.0000pp | +0.0000pp | 244.6d | 0.000 |
| B | +0.0000pp | +0.0000pp | 25.5d | 0.000 |
| C | +0.0000pp | +0.0000pp | 14.2d | 0.000 |
| D | +0.0000pp | +0.0000pp | 32.4d | 0.000 |
| E | +0.0000pp | +0.0000pp | 17.3d | 0.000 |
| F | +0.0000pp | +0.0000pp | 31.5d | 0.000 |
| G | +0.0000pp | +0.0000pp | 20.1d | 0.000 |

---
## 5. Fold3 / Fold5 — WF Barrier Analysis


### Fold3 (2023)

| Case | ΔCAGR | ΔDD | sl_CAGR | avg_cap | days_lo | switches | pass |
|---|---|---|---|---|---|---|---|
| A ❌ | -1.31pp | +2.08pp | -3.4% | 15.0% | 246d | 0 |
| B ❌ | -1.31pp | +2.09pp | -3.4% | 15.1% | 242d | 4 |
| C ❌ | -1.33pp | +2.08pp | -3.5% | 15.5% | 221d | 10 |
| D ❌ | -1.34pp | +1.86pp | -2.9% | 15.3% | 230d | 2 |
| E ❌ | -1.62pp | +1.79pp | -3.6% | 15.8% | 204d | 6 |
| F ❌ | -1.34pp | +1.86pp | -2.9% | 15.4% | 228d | 2 |
| G ❌ | -1.54pp | +1.80pp | -3.4% | 15.9% | 200d | 6 |

### Fold5 (2025)

| Case | ΔCAGR | ΔDD | sl_CAGR | avg_cap | days_lo | switches | pass |
|---|---|---|---|---|---|---|---|
| A ❌ | +1.18pp | +1.84pp | +7.9% | 15.0% | 243d | 0 |
| B ❌ | +1.19pp | +1.85pp | +7.9% | 15.8% | 203d | 7 |
| C ❌ | +1.55pp | +1.84pp | +9.2% | 17.9% | 103d | 8 |
| D ❌ | +0.97pp | +1.68pp | +6.7% | 16.5% | 168d | 5 |
| E ❌ | +0.76pp | +1.63pp | +6.0% | 19.6% | 20d | 2 |
| F ❌ | +0.97pp | +1.68pp | +6.7% | 16.7% | 162d | 5 |
| G ❌ | +1.27pp | +2.90pp | +7.4% | 16.7% | 162d | 5 |

---
## 6. Stop Condition Assessment

Case A (固定15%): WF=3/5  ΔCAGR=+4.43pp  ΔDD=+0.57pp

| adaptive Case | ΔCAGR vs A | ΔDD vs A | WF vs A | A wins? |
|---|---|---|---|---|
| B | +0.03pp | +0.01pp | +0 | No |
| C | +0.12pp | +0.01pp | +0 | No |
| D | +1.08pp | +0.14pp | +0 | No |
| E | +1.44pp | +0.13pp | +0 | No |
| F | +1.08pp | +0.14pp | +0 | No |
| G | +2.00pp | +0.39pp | +0 | No |

**停止条件**: 未発動 → 継続可

---
## 7. Failure Analysis


**A** REJECT: WF=3/5 < 4

**B** REJECT: WF=3/5 < 4

**C** REJECT: WF=3/5 < 4

**D** REJECT: WF=3/5 < 4

**E** REJECT: WF=3/5 < 4

**F** REJECT: WF=3/5 < 4

**G** REJECT: WF=3/5 < 4

---
## 8. Final Output

### optimal switch frequency

| 項目 | 値 |
|---|---|
| best case | **G** — DD>7%  5d継続 |
| avg_switches/fold | 7.4 |
| avg_days_lo/fold | 148.4d |
| switch_half_life | 20.1d |
| state_persistence | 68.7% |
| WF | 3/5 |
| ΔCAGR | +6.43pp |
| ΔDD | +0.96pp |

### best persistence

| 項目 | 値 |
|---|---|
| case | **B** — DD>5%  即時 |
| transition_efficiency | 0.000 |
| dd_saved/switch | +0.0000pp |
| alpha_loss/switch | +0.0000pp |
| half_life | 25.5d |

**閾値比較** (5% vs 7%, 即時):

- 5% 即時(B): ΔCAGR=+4.46pp  ΔDD=+0.58pp  sw=8.0
- 7% 即時(C): ΔCAGR=+4.55pp  ΔDD=+0.58pp  sw=12.2

**継続日数比較** (5% threshold):

- persist=0d (Case B): ΔCAGR=+4.46pp  ΔDD=+0.58pp  sw=8.0
- persist=3d (Case D): ΔCAGR=+5.51pp  ΔDD=+0.71pp  sw=5.6
- persist=5d (Case F): ΔCAGR=+5.51pp  ΔDD=+0.71pp  sw=5.6

### recommend next study

**WF 3/5 ボーダー残存 — 追加分離研究を推奨**

最良 adaptive: Case G (DD>7%  5d継続)
- ΔCAGR=+6.43pp  ΔDD=+0.96pp  WF=3/5

**次研究推奨**: Fold3/Fold5 barrier の分離
- Fold3(2023): alpha_absent が binding → CAP削減は効果限定的
- Fold5(2025): ΔDD excess → Case G の ΔDD改善を確認
- 仮説: Case G で Fold5 ΔDD ≤ +1.5pp に収束するかを検証
- スクリプト: `src/backtest/study8_cap_transition_v2.py`
- 変更禁止: ENTRY / EXIT / GATE / PARAMS_LOCKED