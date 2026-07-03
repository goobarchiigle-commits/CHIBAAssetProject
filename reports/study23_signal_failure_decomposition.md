# Study23 Signal Failure Decomposition

作成日: 2026-06-23  |  説明責任のみ（accountability only）/ 新規シグナル設計禁止 / Entry・Exit・Sizing・Capital変更禁止 / 改善案生成禁止

**Strategy**: Study9 Case B (FROZEN)  **Entry**: RSR∈[92,95), days_cross90≤5, rsr_slope_5d≤5  **Exit**: RSR<90  **Capital/Authority/Execution**: 現行production configuration  **Governance**: annual_rebalance

**観測ウィンドウ**: 2018-01-01 → 2025-12-31

**目的**: Study22で検出されたFALSE_BREAKOUT/HIGH_VOL_ENTRYが既存特徴量のみで説明可能か検証する

**対象**: FALSE_BREAKOUT/HIGH_VOL_ENTRY（label=1, n=14）+ 全勝ちトレード（label=0, n=13）= 計27件

⚠ 小サンプル注意: n=27前後（Study21/22由来）。per-feature分割は過学習リスクが高く、本研究はexplainability評価専用（新規ルール採用は禁止）。

⚠ データ品質注意: 9104.T（#3/#4/#5）でgap_pctが+45〜+55%という異常値。OHLCVデータの分割調整不整合が疑われる（real economic gapではない可能性が高い）。gap_pctはbest_ruleとして選定されなかったため最終判定への影響はないが、gap_pct単独の precision/lift 数値（本レポートFeature Profiles表）は9104.Tの寄与分だけnoisy である点に注意。データパイプラインの修正は本研究の範囲外。

---
## Feature Profiles

| feature | type | MI | lift | best_rule | precision | recall | coverage |
|---|---|---|---|---|---|---|---|
| entry_rsr | continuous | 0.0000 | 1.00x | `entry_rsr <= 92.7` | 60.0% | 21.4% | 18.5% |
| rsr_slope_5d | continuous | 0.0403 | 1.05x | `rsr_slope_5d <= 2.38` | 60.0% | 64.3% | 55.6% |
| days_cross90 | continuous | 0.2307 | 1.61x | `days_cross90 <= 2` | 68.4% | 92.9% | 70.4% |
| atr_z | continuous | 0.1078 | 1.50x | `atr_z <= -2` | 80.0% | 28.6% | 18.5% |
| gap_pct | continuous | 0.1078 | 1.50x | `gap_pct >= 0.103` | 77.8% | 50.0% | 33.3% |
| volume_z | continuous | 0.1321 | 1.29x | `volume_z >= 1.23` | 83.3% | 35.7% | 22.2% |
| rank | continuous | 0.0000 | 1.00x | `rank >= 4` | 50.0% | 92.9% | 96.3% |
| market_regime | categorical | 0.0000 | 1.00x | `(no valid split)` | 0.0% | 0.0% | 0.0% |
| sector | categorical | 0.2288 | 1.93x | `sector == 電機精密` | 28.6% | 14.3% | 25.9% |

### Bin Breakdown（tercile / category別 conditional_loss_rate）

**entry_rsr**: bin0(n=27, loss_rate=51.8%)
**rsr_slope_5d**: bin0(n=11, loss_rate=54.5%) / bin1(n=15, loss_rate=53.3%) / bin2(n=1, loss_rate=0.0%)
**days_cross90**: bin0(n=13, loss_rate=61.5%) / bin1(n=6, loss_rate=83.3%) / bin2(n=8, loss_rate=12.5%)
**atr_z**: bin0(n=9, loss_rate=77.8%) / bin1(n=9, loss_rate=44.4%) / bin2(n=9, loss_rate=33.3%)
**gap_pct**: bin0(n=9, loss_rate=33.3%) / bin1(n=9, loss_rate=44.4%) / bin2(n=9, loss_rate=77.8%)
**volume_z**: bin0(n=9, loss_rate=22.2%) / bin1(n=9, loss_rate=66.7%) / bin2(n=9, loss_rate=66.7%)
**rank**: bin0(n=27, loss_rate=51.8%)
**market_regime**: binBull(n=27, loss_rate=51.8%)
**sector**: bin保険(n=1, loss_rate=100.0%) / bin化学(n=1, loss_rate=100.0%) / bin商社(n=4, loss_rate=50.0%) / bin小売(n=2, loss_rate=50.0%) / bin機械(n=2, loss_rate=100.0%) / bin海運(n=3, loss_rate=66.7%) / bin銀行(n=4, loss_rate=50.0%) / bin電機(n=1, loss_rate=0.0%) / bin電機精密(n=7, loss_rate=28.6%) / bin食品(n=2, loss_rate=50.0%)

---
## Top Predictors（mutual_information降順 上位3）

1. **days_cross90** — MI=0.2307, lift=1.61x, precision=68.4%, coverage=70.4%
2. **sector** — MI=0.2288, lift=1.93x, precision=28.6%, coverage=25.9%
3. **volume_z** — MI=0.1321, lift=1.29x, precision=83.3%, coverage=22.2%

---
## Feature Interactions

- Rule: `(days_cross90 <= 2) AND (sector == 電機精密)`
- 単独: days_cross90(precision=68.4%) / sector(precision=28.6%)
- 組合せ: precision=50.0%  recall=14.3%  coverage=14.8%
- 相乗効果: なし（単独と同等以下）

---
## Executive Summary

| 指標 | 値 |
|---|---|
| best_rule | `days_cross90 <= 2` |
| precision | 68.4% |
| recall | 92.9% |
| coverage | 70.4% |
| loss_explainability (counterfactual_removed_loss, R-weighted) | 94.9% |
| profit_explainability (alpha_retention) | 79.8% |
| counterfactual_removed_loss | 94.9% |
| counterfactual_removed_profit | 20.2% |
| alpha_retention | 79.8% |
| top_predictors | days_cross90, sector, volume_z |

**research_decision: PARTIALLY_EXPLAINABLE**

判定理由: precision=68.4%>=60% AND coverage=70.4%>=40% (EXPLAINABLE full gate not met)

---
## 判定基準

| 判定 | 条件 |
|---|---|
| EXPLAINABLE | precision≥70% AND coverage≥50% AND removed_profit≤20% AND alpha_retention≥80% |
| PARTIALLY_EXPLAINABLE | precision≥60% AND coverage≥40% |
| NEW_SIGNAL_REQUIRED | 上記未達 |

---
## Trade Dataset（全件）

| # | 銘柄 | Entry | label | category | actual_R | entry_rsr | slope5 | d90 | atr_z | gap% | vol_z | regime | sector | rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4021.T | 2020-08-12 | 1 | FALSE_BREAKOUT | -1.93R | 92.7 | +0.0 | 1 | +1.06 | +10.31% | +3.45 | Bull | 化学 | 4 |
| 2 | 6594.T | 2020-08-18 | 0 | WINNER | +17.48R | 92.7 | +2.4 | 5 | -1.13 | +4.68% | -1.10 | Bull | 電機精密 | 4 |
| 3 | 9104.T | 2021-03-18 | 1 | FALSE_BREAKOUT | -0.72R | 92.7 | +0.0 | 2 | -2.00 | +53.03% | -0.64 | Bull | 海運 | 4 |
| 4 | 9104.T | 2021-03-26 | 1 | HIGH_VOL_ENTRY | -2.83R | 92.7 | +0.0 | 1 | -0.47 | +55.29% | +0.27 | Bull | 海運 | 4 |
| 5 | 9104.T | 2021-04-26 | 0 | WINNER | +72.22R | 92.7 | +4.9 | 4 | -1.84 | +45.61% | -1.22 | Bull | 海運 | 4 |
| 6 | 6762.T | 2022-09-14 | 1 | FALSE_BREAKOUT | -2.96R | 92.9 | +4.8 | 2 | -1.11 | +1.75% | +1.57 | Bull | 電機精密 | 4 |
| 7 | 3382.T | 2022-09-27 | 1 | FALSE_BREAKOUT | -1.63R | 92.9 | +2.4 | 1 | -1.28 | +7.89% | +0.63 | Bull | 小売 | 4 |
| 8 | 8058.T | 2022-10-04 | 1 | HIGH_VOL_ENTRY | -2.53R | 92.9 | +4.8 | 1 | -2.74 | +13.48% | +1.23 | Bull | 商社 | 4 |
| 9 | 8053.T | 2022-10-13 | 1 | FALSE_BREAKOUT | -0.56R | 92.9 | +2.4 | 2 | -3.24 | +11.43% | -0.96 | Bull | 商社 | 4 |
| 10 | 8354.T | 2022-10-20 | 1 | FALSE_BREAKOUT | -1.62R | 92.9 | -2.4 | 1 | -2.66 | +10.68% | -0.47 | Bull | 銀行 | 4 |
| 11 | 2914.T | 2022-11-09 | 1 | HIGH_VOL_ENTRY | -1.20R | 92.9 | +4.8 | 4 | -1.69 | +21.02% | -0.08 | Bull | 食品 | 4 |
| 13 | 8306.T | 2023-02-08 | 0 | WINNER | +0.50R | 92.9 | -2.4 | 1 | +0.49 | +10.15% | -1.11 | Bull | 銀行 | 4 |
| 15 | 6146.T | 2023-03-22 | 0 | WINNER | +1.10R | 92.9 | +4.8 | 4 | -0.49 | +4.63% | -0.55 | Bull | 電機精密 | 4 |
| 16 | 6506.T | 2023-04-03 | 1 | FALSE_BREAKOUT | -2.28R | 92.9 | +0.0 | 1 | -1.43 | +4.99% | -0.32 | Bull | 機械 | 4 |
| 17 | 2914.T | 2023-05-17 | 0 | WINNER | +0.45R | 92.9 | +4.8 | 3 | +0.76 | +16.32% | +0.44 | Bull | 食品 | 4 |
| 18 | 8002.T | 2023-05-24 | 0 | WINNER | +12.64R | 92.9 | +4.8 | 4 | -0.92 | +9.21% | -0.41 | Bull | 商社 | 4 |
| 19 | 8015.T | 2023-08-02 | 0 | WINNER | +1.24R | 92.9 | +4.8 | 1 | -0.16 | +6.25% | +1.47 | Bull | 商社 | 4 |
| 20 | 8306.T | 2023-10-17 | 0 | WINNER | +0.41R | 92.9 | +0.0 | 1 | -2.81 | +7.75% | -1.38 | Bull | 銀行 | 4 |
| 21 | 8306.T | 2023-11-22 | 1 | FALSE_BREAKOUT | -0.41R | 92.9 | +4.8 | 2 | -0.90 | +6.11% | -1.01 | Bull | 銀行 | 3 |
| 23 | 6857.T | 2023-12-12 | 0 | WINNER | +23.39R | 92.9 | +0.0 | 1 | -1.12 | +4.16% | -1.05 | Bull | 電機精密 | 4 |
| 25 | 8035.T | 2024-05-15 | 0 | WINNER | +1.03R | 92.9 | -2.4 | 1 | +0.12 | +5.34% | -0.72 | Bull | 電機精密 | 4 |
| 26 | 8725.T | 2024-05-22 | 1 | HIGH_VOL_ENTRY | -0.63R | 92.9 | +2.4 | 1 | -0.80 | +7.97% | +1.80 | Bull | 保険 | 4 |
| 29 | 6857.T | 2024-09-26 | 1 | HIGH_VOL_ENTRY | -2.11R | 92.9 | +4.8 | 1 | -0.24 | +3.79% | -0.05 | Bull | 電機精密 | 4 |
| 30 | 6702.T | 2024-10-03 | 0 | WINNER | +1.63R | 92.9 | +4.8 | 2 | -0.06 | +3.63% | -1.26 | Bull | 電機 | 4 |
| 31 | 7012.T | 2024-10-15 | 1 | HIGH_VOL_ENTRY | -2.06R | 92.9 | +0.0 | 2 | -0.67 | +1.59% | +1.41 | Bull | 機械 | 4 |
| 32 | 6857.T | 2024-10-22 | 0 | WINNER | +3.21R | 92.9 | +2.4 | 5 | -1.15 | +1.03% | +1.18 | Bull | 電機精密 | 4 |
| 34 | 3197.T | 2025-04-10 | 0 | WINNER | +4.06R | 92.9 | +0.0 | 5 | +2.82 | +3.84% | -0.37 | Bull | 小売 | 4 |

---
## 最終出力

| 指標 | 値 |
|---|---|
| loss_explainability | 94.9% |
| profit_explainability | 79.8% |
| top_predictors | ['days_cross90', 'sector', 'volume_z'] |
| counterfactual_removed_loss | 94.9% |
| counterfactual_removed_profit | 20.2% |
| alpha_retention | 79.8% |
| **research_decision** | **PARTIALLY_EXPLAINABLE** |

研究目的は説明責任のみ。新規Entry/Exitルール・改善案はこのレポートでは提案しない。
