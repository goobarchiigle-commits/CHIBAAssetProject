# Study8 CAP State Regime WF — State Attribution

作成日: 2026-06-16  |  解析専用 / 実装変更禁止

固定: 2-slot 70/30 no-refill / RSR[92,95) d90≤5 / EXIT RSR<90 / CAP_HI=20% / CAP_LO=15%

復帰条件: B-F=5営業日 / G=composite score==0

採用条件: WF≥4/5, ΔCAGR>+0.3pp, ΔDD≤+1.5pp, α_ret≥90.0%

停止: (best ΔCAGR - anchor ΔCAGR) < 0.5pp  OR  state_precision < 55%

🛑 **停止条件発動**: margin=-2.17pp < 0.5pp AND precision=37.7% < 55%
**→ adaptive CAP 研究終了**

**採用**: 0件  | 判定: **🛑 STOP**

---
## 1. WF サマリ

| Case | 説明 | WF | ΔCAGR | ΔDD | sl_CAGR | α_ret | avg_cap | precision | recall | latency | 採用 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 固定20% (anchor) | 3/5 | +6.92pp | +0.56pp | +46.5% | — | 20.0% | 0.0% | 0.0% | 0.0d | ❌ |
| B | rolling_dd>7% | 3/5 | +4.49pp | +0.56pp | +34.7% | 74.6% | 16.5% | 35.3% | 53.1% | 7.2d | ❌ |
| C | rolling_dd + hold_days | 3/5 | +4.61pp | +0.57pp | +35.6% | 76.5% | 16.1% | 38.6% | 68.1% | 9.4d | ❌ |
| D | rolling_dd + unrealized_pnl | 3/5 | +4.60pp | +0.58pp | +35.0% | 75.3% | 16.4% | 35.6% | 55.7% | 6.8d | ❌ |
| E | rolling_dd + slot_concentration | 3/5 | +4.75pp | +0.58pp | +35.5% | 76.4% | 16.0% | 37.7% | 67.8% | 7.7d | ❌ |
| F | rolling_dd + trailing_pf3 | 3/5 | +4.50pp | +0.57pp | +33.6% | 72.4% | 16.5% | 36.4% | 57.7% | 8.9d | ❌ |
| G | composite score>=2 | 3/5 | +4.45pp | +0.60pp | +33.8% | 72.8% | 15.1% | 42.5% | 98.0% | 6.4d | ❌ |

---
## 2. State Attribution Detail (OOS avg)

| Case | state_precision | state_recall | latency(d) | state_duration(d) | trans/yr | alpha_pres | n_acts |
|---|---|---|---|---|---|---|---|
| A | **0.0%** | 0.0% | 0.0 | 0.0 | 0.0 | — | 0 |
| B | **35.3%** | 53.1% | 7.2 | 31.3 | 7.0 | 0.746 | 34 |
| C | **38.6%** | 68.1% | 9.4 | 28.3 | 8.3 | 0.765 | 40 |
| D | **35.6%** | 55.7% | 6.8 | 34.0 | 6.4 | 0.753 | 31 |
| E | **37.7%** | 67.8% | 7.7 | 33.2 | 7.9 | 0.764 | 38 |
| F | **36.4%** | 57.7% | 8.9 | 32.5 | 6.6 | 0.724 | 32 |
| G | **42.5%** | 98.0% | 6.4 | 240.0 | 0.4 | 0.728 | 2 |

---
## 3. Fold 詳細

| Case | Fold | 年 | Regime | ΔCAGR | ΔDD | sl_CAGR | cap | prec | recall | lat | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Fold1 | 2021 | Bull (+13.3%) | +12.98pp | +1.50pp | +148.8% | 20% | 0% | 0% | 0.0d | ✅ |
| A | Fold2 | 2022 | Bear (-4.7%) | +9.87pp | -2.18pp | +41.8% | 20% | 0% | 0% | 0.0d | ✅ |
| A | Fold3 | 2023 | Bull (+29.8%) | -0.45pp | +2.51pp | +0.5% | 20% | 0% | 0% | 0.0d | ❌ |
| A | Fold4 | 2024 | Bull (+19.7%) | +11.61pp | -1.80pp | +36.1% | 20% | 0% | 0% | 0.0d | ✅ |
| A | Fold5 | 2025 | Bull (+26.8%) | +0.58pp | +2.77pp | +5.2% | 20% | 0% | 0% | 0.0d | ❌ |
| B | Fold1 | 2021 | Bull (+13.3%) | +8.72pp | -0.34pp | +110.5% | 17% | 49% | 55% | 5.2d | ✅ |
| B | Fold2 | 2022 | Bear (-4.7%) | +7.59pp | -2.16pp | +36.5% | 16% | 61% | 75% | 11.3d | ✅ |
| B | Fold3 | 2023 | Bull (+29.8%) | -1.55pp | +2.08pp | -4.4% | 16% | 33% | 77% | 14.0d | ❌ |
| B | Fold4 | 2024 | Bull (+19.7%) | +6.92pp | +1.39pp | +24.4% | 17% | 30% | 47% | 2.3d | ✅ |
| B | Fold5 | 2025 | Bull (+26.8%) | +0.77pp | +1.84pp | +6.4% | 17% | 4% | 11% | 3.0d | ❌ |
| C | Fold1 | 2021 | Bull (+13.3%) | +8.80pp | -0.31pp | +112.6% | 17% | 52% | 64% | 7.4d | ✅ |
| C | Fold2 | 2022 | Bear (-4.7%) | +7.59pp | -2.15pp | +36.8% | 16% | 62% | 85% | 11.4d | ✅ |
| C | Fold3 | 2023 | Bull (+29.8%) | -1.55pp | +2.09pp | -4.4% | 15% | 34% | 86% | 15.0d | ❌ |
| C | Fold4 | 2024 | Bull (+19.7%) | +6.92pp | +1.40pp | +24.6% | 16% | 32% | 66% | 6.1d | ✅ |
| C | Fold5 | 2025 | Bull (+26.8%) | +1.30pp | +1.84pp | +8.3% | 17% | 14% | 40% | 7.2d | ❌ |
| D | Fold1 | 2021 | Bull (+13.3%) | +8.72pp | -0.34pp | +110.5% | 17% | 51% | 57% | 6.3d | ✅ |
| D | Fold2 | 2022 | Bear (-4.7%) | +7.59pp | -2.16pp | +36.5% | 16% | 62% | 78% | 11.3d | ✅ |
| D | Fold3 | 2023 | Bull (+29.8%) | -0.97pp | +2.26pp | -2.0% | 15% | 30% | 81% | 9.2d | ❌ |
| D | Fold4 | 2024 | Bull (+19.7%) | +6.92pp | +1.33pp | +23.8% | 17% | 30% | 47% | 2.3d | ✅ |
| D | Fold5 | 2025 | Bull (+26.8%) | +0.75pp | +1.82pp | +6.2% | 17% | 6% | 16% | 4.7d | ❌ |
| E | Fold1 | 2021 | Bull (+13.3%) | +8.72pp | -0.34pp | +110.5% | 17% | 52% | 64% | 7.4d | ✅ |
| E | Fold2 | 2022 | Bear (-4.7%) | +7.59pp | -2.16pp | +36.5% | 16% | 63% | 88% | 11.4d | ✅ |
| E | Fold3 | 2023 | Bull (+29.8%) | -0.97pp | +2.26pp | -2.0% | 15% | 31% | 86% | 8.7d | ❌ |
| E | Fold4 | 2024 | Bull (+19.7%) | +6.92pp | +1.33pp | +23.8% | 16% | 32% | 64% | 6.1d | ✅ |
| E | Fold5 | 2025 | Bull (+26.8%) | +1.49pp | +1.82pp | +8.9% | 16% | 11% | 38% | 5.0d | ❌ |
| F | Fold1 | 2021 | Bull (+13.3%) | +7.98pp | -0.31pp | +102.1% | 17% | 48% | 54% | 5.2d | ✅ |
| F | Fold2 | 2022 | Bear (-4.7%) | +7.62pp | -2.14pp | +37.0% | 16% | 62% | 75% | 12.2d | ✅ |
| F | Fold3 | 2023 | Bull (+29.8%) | -1.33pp | +2.08pp | -3.5% | 15% | 34% | 90% | 16.0d | ❌ |
| F | Fold4 | 2024 | Bull (+19.7%) | +6.92pp | +1.38pp | +24.4% | 17% | 34% | 60% | 8.2d | ✅ |
| F | Fold5 | 2025 | Bull (+26.8%) | +1.29pp | +1.84pp | +8.3% | 18% | 5% | 10% | 2.8d | ❌ |
| G | Fold1 | 2021 | Bull (+13.3%) | +7.96pp | -0.26pp | +103.3% | 15% | 55% | 92% | 17.0d | ✅ |
| G | Fold2 | 2022 | Bear (-4.7%) | +7.48pp | -2.11pp | +36.7% | 15% | 67% | 100% | 0.0d | ✅ |
| G | Fold3 | 2023 | Bull (+29.8%) | -1.30pp | +2.10pp | -3.5% | 15% | 32% | 98% | 15.0d | ❌ |
| G | Fold4 | 2024 | Bull (+19.7%) | +6.92pp | +1.42pp | +24.7% | 15% | 39% | 100% | 0.0d | ✅ |
| G | Fold5 | 2025 | Bull (+26.8%) | +1.20pp | +1.85pp | +8.0% | 15% | 20% | 100% | 0.0d | ❌ |

---
## 4. Predictive Power Matrix

> state_precision: P(bad_period | state_on)  — state が活性の時に実際にDD悪化が起きる確率
> state_recall: P(state_on | bad_period)  — DD悪化期間にstateが活性だった割合
> bad_period定義: 次10日以内にsleeve equity が1.5%以上下落

| Case | 説明 | avg precision | avg recall | avg latency | predictive? |
|---|---|---|---|---|---|
| A | 固定20% (anchor) | **0.0%** | 0.0% | 0.0d | ❌ 無効 |
| B | rolling_dd>7% | **35.3%** | 53.1% | 7.2d | ❌ 無効 |
| C | rolling_dd + hold_days | **38.6%** | 68.1% | 9.4d | ❌ 無効 |
| D | rolling_dd + unrealized_pnl | **35.6%** | 55.7% | 6.8d | ❌ 無効 |
| E | rolling_dd + slot_concentration | **37.7%** | 67.8% | 7.7d | ❌ 無効 |
| F | rolling_dd + trailing_pf3 | **36.4%** | 57.7% | 8.9d | ❌ 無効 |
| G | composite score>=2 | **42.5%** | 98.0% | 6.4d | ❌ 無効 |

---
## 5. WF Barrier Analysis (Fold3/Fold5)


### Fold3 (2023)

| Case | ΔCAGR | ΔDD | sl_CAGR | cap | precision | latency | pass |
|---|---|---|---|---|---|---|---|
| A ❌ | -0.45pp | +2.51pp | +0.5% | 20% | 0% | 0.0d |
| B ❌ | -1.55pp | +2.08pp | -4.4% | 16% | 33% | 14.0d |
| C ❌ | -1.55pp | +2.09pp | -4.4% | 15% | 34% | 15.0d |
| D ❌ | -0.97pp | +2.26pp | -2.0% | 15% | 30% | 9.2d |
| E ❌ | -0.97pp | +2.26pp | -2.0% | 15% | 31% | 8.7d |
| F ❌ | -1.33pp | +2.08pp | -3.5% | 15% | 34% | 16.0d |
| G ❌ | -1.30pp | +2.10pp | -3.5% | 15% | 32% | 15.0d |

### Fold5 (2025)

| Case | ΔCAGR | ΔDD | sl_CAGR | cap | precision | latency | pass |
|---|---|---|---|---|---|---|---|
| A ❌ | +0.58pp | +2.77pp | +5.2% | 20% | 0% | 0.0d |
| B ❌ | +0.77pp | +1.84pp | +6.4% | 17% | 4% | 3.0d |
| C ❌ | +1.30pp | +1.84pp | +8.3% | 17% | 14% | 7.2d |
| D ❌ | +0.75pp | +1.82pp | +6.2% | 17% | 6% | 4.7d |
| E ❌ | +1.49pp | +1.82pp | +8.9% | 16% | 11% | 5.0d |
| F ❌ | +1.29pp | +1.84pp | +8.3% | 18% | 5% | 2.8d |
| G ❌ | +1.20pp | +1.85pp | +8.0% | 15% | 20% | 0.0d |

---
## 6. Stop Condition Assessment

Case A (固定20%): WF=3/5  ΔCAGR=+6.92pp  ΔDD=+0.56pp

Best adaptive: Case E — ΔCAGR=+4.75pp  margin over anchor=-2.17pp  precision=37.7%

| 停止条件 | 判定値 | 閾値 | 発動? |
|---|---|---|---|
| (best ΔCAGR - anchor ΔCAGR) < 0.5pp | -2.17pp | 0.5pp | **⚠ 発動** |
| state_precision < 55% | 37.7% | 55% | **⚠ 発動** |

**結果**: **停止条件発動 → adaptive CAP研究終了**

---
## 7. Failure Analysis


**A** REJECT: WF=3/5 < 4

**B** REJECT: WF=3/5 < 4 / α_ret=74.6% < 90.0%

**C** REJECT: WF=3/5 < 4 / α_ret=76.5% < 90.0%

**D** REJECT: WF=3/5 < 4 / α_ret=75.3% < 90.0%

**E** REJECT: WF=3/5 < 4 / α_ret=76.4% < 90.0%

**F** REJECT: WF=3/5 < 4 / α_ret=72.4% < 90.0%

**G** REJECT: WF=3/5 < 4 / α_ret=72.8% < 90.0%

---
## 8. Final Output

### best_state

| 項目 | 値 |
|---|---|
| **best case** | **E** — rolling_dd + slot_concentration |
| WF | 3/5 |
| ΔCAGR | +4.75pp |
| ΔDD | +0.58pp |
| avg_cap | 16.0% |
| state_precision | 37.7% |
| state_recall | 67.8% |
| state_latency | 7.7d |
| alpha_preservation | 0.764 |

### predictive_power

| 指標 | Case G | 全adaptive平均 |
|---|---|---|
| state_precision | 42.5% | 37.7% |
| state_recall | 98.0% | 66.7% |
| threshold | 55% | — |

予測力判定: **❌ 予測力不足** (best precision=42.5% < 55%)

### recommend next study

**🛑 停止条件発動 → adaptive CAP研究終了**

**根本原因**: best adaptive ΔCAGR が anchor ΔCAGR + 0.5pp 未満
- CAP切替のメリット < コスト = CAP研究に追加改善余地なし

**次研究推奨**: CAP sweep 終結、最大CAP固定値を採用
- スクリプト: `src/backtest/study8_cap_final.py`
- 固定 CAP=20% (Case A) をそのまま production に組み込み
- 変更禁止: ENTRY/EXIT/GATE/PARAMS_LOCKED