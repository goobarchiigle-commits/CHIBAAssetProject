# Study8 Failure Attribution — Fold Diagnostics

作成日: 2026-06-15  |  解析専用 / 実装変更禁止

対象: Case A (1-slot 100%) vs Case E (2-slot 70/30 no-refill)

採用条件: ΔCAGR>+0.3pp AND ΔDD≤+1.5pp

---
## 1. Performance — Fold × Case

| Case | Fold | OOS | Regime | base_CAGR | sl_CAGR | total_CAGR | ΔCAGR | ΔDD | pass |
|---|---|---|---|---|---|---|---|---|---|
| A | Fold1 | 2021 | Bull (+13.3%) | +32.1% | +125.0% | +48.2% | +16.13pp | +3.18pp | ❌ |
| A | Fold2 | 2022 | Bear (-4.7%) | -6.3% | +40.6% | +6.8% | +13.16pp | -1.59pp | ✅ |
| A | Fold3 | 2023 | Bull (+29.8%) | +2.1% | -1.4% | +0.8% | -1.29pp | +3.56pp | ❌ |
| A | Fold4 | 2024 | Bull (+19.7%) | -6.0% | +29.3% | +6.7% | +12.71pp | -2.02pp | ✅ |
| A | Fold5 | 2025 | Bull (+26.8%) | +3.6% | +10.7% | +6.6% | +3.04pp | +3.89pp | ❌ |
| E | Fold1 | 2021 | Bull (+13.3%) | +32.1% | +108.5% | +43.3% | +11.19pp | +0.97pp | ✅ |
| E | Fold2 | 2022 | Bear (-4.7%) | -6.3% | +36.8% | +3.4% | +9.77pp | -2.50pp | ✅ |
| E | Fold3 | 2023 | Bull (+29.8%) | +2.1% | +0.4% | +1.6% | -0.49pp | +2.37pp | ❌ |
| E | Fold4 | 2024 | Bull (+19.7%) | -6.0% | +32.9% | +5.4% | +11.47pp | -2.01pp | ✅ |
| E | Fold5 | 2025 | Bull (+26.8%) | +3.6% | +4.9% | +4.0% | +0.48pp | +2.66pp | ❌ |

---
## 2. Execution — Fold × Case

| Case | Fold | trig/yr | avg_hold | capital_util% | cash_idle% | idle_days |
|---|---|---|---|---|---|---|
| A | Fold1 | 7.2 | 2.5d | 85.7% | 14.3% | 35d |
| A | Fold2 | 17.6 | 41.5d | 96.7% | 3.3% | 8d |
| A | Fold3 | 32.8 | 7.8d | 93.1% | 6.9% | 17d |
| A | Fold4 | 12.3 | 12.7d | 96.7% | 3.3% | 8d |
| A | Fold5 | 18.7 | 22.7d | 79.8% | 20.2% | 49d |
| E | Fold1 | 6.2 | 2.5d | 85.3% | 14.7% | 36d |
| E | Fold2 | 17.6 | 41.5d | 96.7% | 3.3% | 8d |
| E | Fold3 | 32.8 | 7.8d | 93.1% | 6.9% | 17d |
| E | Fold4 | 12.3 | 12.7d | 96.7% | 3.3% | 8d |
| E | Fold5 | 17.6 | 22.7d | 79.4% | 20.6% | 50d |

---
## 3. Trade Quality — Fold × Case

| Case | Fold | win_rate | PF | avg_trade_ret | top10_share | loss_share | med_fwd10 |
|---|---|---|---|---|---|---|---|
| A | Fold1 | 57.1% | 1.694 | +0.45% | 100.0% | 144.1% | -0.14% |
| A | Fold2 | 47.1% | 22.426 | +14.98% | 104.7% | 4.7% | -1.65% |
| A | Fold3 | 40.6% | 0.897 | -0.51% | 836.6% | 973.7% | 1.75% |
| A | Fold4 | 33.3% | 1.256 | +1.78% | 353.3% | 391.2% | 0.62% |
| A | Fold5 | 61.1% | 12.666 | +6.70% | 108.4% | 8.6% | 1.89% |
| E | Fold1 | 66.7% | 3.518 | +1.04% | 100.0% | 39.7% | -3.64% |
| E | Fold2 | 47.1% | 23.020 | +14.98% | 104.5% | 4.5% | -1.65% |
| E | Fold3 | 40.6% | 0.974 | -0.51% | 3563.1% | 3796.6% | 1.75% |
| E | Fold4 | 33.3% | 1.883 | +1.78% | 175.1% | 113.3% | 0.62% |
| E | Fold5 | 58.8% | 8.680 | +6.23% | 113.0% | 13.0% | 0.86% |

---
## 4. ΔCAGR Decomposition (pp 寄与 / 正規化 %)

> selection=銘柄差 / holding=保有期間差 / sizing=配分差 / timing=エントリー時刻差 / cash_effect=未投資資金影響

> passive_zone = RSR[92,95) d90=1 全信号の平均リターン (selection の baseline)

| Case | Fold | sel_pp | hold_pp | siz_pp | tim_pp | cash_pp | sel% | hold% | siz% | tim% | cash% | passive_ret | n_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Fold1 | -0.310 | -0.013 | +0.000 | -0.436 | -0.917 | -18.5% | -0.8% | +0.0% | -26.0% | -54.7% | +4.79% | 27 |
| A | Fold2 | +36.048 | +7.155 | +0.000 | -2.763 | +0.042 | +78.4% | +15.6% | +0.0% | -6.0% | +0.1% | +2.52% | 21 |
| A | Fold3 | -1.634 | -5.163 | +0.000 | +1.171 | -0.029 | -20.4% | -64.6% | +0.0% | +14.6% | -0.4% | +1.11% | 31 |
| A | Fold4 | +2.041 | -1.008 | +0.000 | +0.000 | +0.039 | +66.1% | -32.6% | +0.0% | +0.0% | +1.3% | -1.49% | 32 |
| A | Fold5 | +10.324 | -5.579 | +0.000 | +3.483 | -0.144 | +52.9% | -28.6% | +0.0% | +17.8% | -0.7% | +0.55% | 19 |
| E | Fold1 | -0.230 | +0.013 | +0.000 | +0.000 | -0.943 | -19.3% | +1.1% | +0.0% | +0.0% | -79.5% | +4.79% | 27 |
| E | Fold2 | +36.048 | +7.155 | +0.000 | -2.763 | +0.042 | +78.4% | +15.6% | +0.0% | -6.0% | +0.1% | +2.52% | 21 |
| E | Fold3 | -1.634 | -5.163 | +0.000 | +1.171 | -0.029 | -20.4% | -64.6% | +0.0% | +14.6% | -0.4% | +1.11% | 31 |
| E | Fold4 | +2.041 | -1.008 | +0.000 | +0.000 | +0.039 | +66.1% | -32.6% | +0.0% | +0.0% | +1.3% | -1.49% | 32 |
| E | Fold5 | +9.016 | -3.277 | +0.000 | +3.289 | -0.147 | +57.3% | -20.8% | +0.0% | +20.9% | -0.9% | +0.55% | 19 |

---
## 5. Fold Failure Summary

| Case | Fold | winner | largest_positive | largest_negative | failure_reason | confidence |
|---|---|---|---|---|---|---|
| A | Fold1 | FAIL | sizing | cash_effect | mixed(tail_miss+dd_excess) | medium |
| A | Fold2 | PASS | selection | timing | regime_break | high |
| A | Fold3 | FAIL | timing | holding | mixed(alpha_absent+tail_miss) | low |
| A | Fold4 | PASS | selection | holding | mixed(regime_break+tail_miss) | medium |
| A | Fold5 | FAIL | selection | holding | dd_excess | high |
| E | Fold1 | PASS | holding | cash_effect | none | high |
| E | Fold2 | PASS | selection | timing | regime_break | high |
| E | Fold3 | FAIL | timing | holding | mixed(tail_miss+alpha_absent) | low |
| E | Fold4 | PASS | selection | holding | mixed(regime_break+tail_miss) | medium |
| E | Fold5 | FAIL | selection | holding | dd_excess | high |

---
## 6. Dominant Bottleneck Analysis


### Case A

| 失敗因子 | 出現スコア | 割合 |
|---|---|---|
| dd_excess | 1.5 | 50.0% |
| tail_miss | 1.0 | 33.3% |
| alpha_absent | 0.5 | 16.7% |

→ **mixed causes**: dd_excess (50%)

失敗Fold 平均attribution: selection=+2.793pp  holding=-3.585pp  cash=-0.363pp

### Case E

| 失敗因子 | 出現スコア | 割合 |
|---|---|---|
| dd_excess | 1.0 | 50.0% |
| tail_miss | 0.5 | 25.0% |
| alpha_absent | 0.5 | 25.0% |

→ **mixed causes**: dd_excess (50%)

失敗Fold 平均attribution: selection=+3.691pp  holding=-4.220pp  cash=-0.088pp

---
## 7. Fold3 / Fold5 Deep Dive


### Fold3 (2023)

| 指標 | Case A | Case E | 差(E-A) |
|---|---|---|---|
| base_CAGR (%) | 2.08 | 2.08 | +0.00 |
| sl_CAGR (%) | -1.41 | 0.44 | +1.85 |
| ΔCAGR (pp) | -1.29 | -0.49 | +0.80 |
| ΔDD (pp) | 3.56 | 2.37 | -1.19 |
| trig/yr | 32.8 | 32.8 | +0.0 |
| avg_hold (d) | 7.8 | 7.8 | +0.0 |
| capital_util% | 93.1 | 93.1 | +0.0 |
| win_rate% | 40.6 | 40.6 | +0.0 |
| PF | 0.897 | 0.974 | +0.077 |
| avg_trade_ret% | -0.51 | -0.51 | +0.00 |
| med_fwd10% | 1.75 | 1.75 | +0.00 |
| passive_zone_ret | +1.11% | +1.11% | — |
| failure_reason | mixed(alpha_absent+tail_miss) | mixed(tail_miss+alpha_absent) | — |

**ΔCAGR attribution (Case E, Fold3 (2023))**:

| component | pp | % of abs total |
|---|---|---|
| selection | -1.634 | -20.4% |
| holding | -5.163 | -64.6% |
| sizing | +0.000 | +0.0% |
| timing | +1.171 | +14.6% |
| cash_effect | -0.029 | -0.4% |

### Fold5 (2025)

| 指標 | Case A | Case E | 差(E-A) |
|---|---|---|---|
| base_CAGR (%) | 3.57 | 3.57 | +0.00 |
| sl_CAGR (%) | 10.66 | 4.89 | -5.77 |
| ΔCAGR (pp) | 3.04 | 0.48 | -2.56 |
| ΔDD (pp) | 3.89 | 2.66 | -1.23 |
| trig/yr | 18.7 | 17.6 | -1.1 |
| avg_hold (d) | 22.7 | 22.7 | +0.0 |
| capital_util% | 79.8 | 79.4 | -0.4 |
| win_rate% | 61.1 | 58.8 | -2.3 |
| PF | 12.666 | 8.680 | -3.986 |
| avg_trade_ret% | 6.70 | 6.23 | -0.47 |
| med_fwd10% | 1.89 | 0.86 | -1.03 |
| passive_zone_ret | +0.55% | +0.55% | — |
| failure_reason | dd_excess | dd_excess | — |

**ΔCAGR attribution (Case E, Fold5 (2025))**:

| component | pp | % of abs total |
|---|---|---|
| selection | +9.016 | +57.3% |
| holding | -3.277 | -20.8% |
| sizing | +0.000 | +0.0% |
| timing | +3.289 | +20.9% |
| cash_effect | -0.147 | -0.9% |

---
## 8. Final Judgment — Single Dominant Bottleneck

| factor | evidence_score | share |
|---|---|---|
| dd_excess | 2.5 | 50% |
| tail_miss | 1.5 | 30% |
| alpha_absent | 1.0 | 20% |

**name**: dd_excess
**confidence**: medium (50%)
**evidence**: Fold5 ΔDD=+3.89pp (A) / +2.66pp (E) — 採用閾値+1.5pp の大幅超過
**evidence**: Fold3 ΔDD=+3.56pp (A) / +2.37pp (E) — Bull相場でスリーブが downside を拡大
**evidence**: ΔDD改善の唯一の成功例: Case E Fold2 (Bear) で ΔDD=-2.50pp (スリーブが防御的に機能)

---
## 9. Next Experiment Recommendation (1件のみ)

**推奨**: Dynamic Sleeve CAP — base_CAGR 連動縮小

- 条件: base_CAGR < 5% (fold水準) → SLEEVE_CAP_FR = 10% (現行20%の半分)
- 条件: base_CAGR ≥ 5% → SLEEVE_CAP_FR = 20% (現行通り)
- 期待: Fold3 (base=+2.1%) でスリーブ資本を半減 → ΔDD/2 で採用条件クリア可能性
- 測定: WF, ΔCAGR, ΔDD, sl_CAGR, alpha_retention vs Case E baseline
- 実装ポイント: yearly_base_CAGR をrollforward 1年推定値で代替 (IS期間のbase_CAGRを使用)
- 禁止: gate追加, exit変更, entry変更, パラメータ探索
- スクリプト: src/backtest/study8_dynamic_cap.py
- 出力: reports/study8_dynamic_cap.md

**根拠**: dominant bottleneck = dd_excess (50%) — CAP削減は entry/exit/gate を一切変更せず ΔDD を比例削減できる唯一の手段