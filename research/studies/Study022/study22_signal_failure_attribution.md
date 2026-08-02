# Study22 Signal Failure Attribution

作成日: 2026-06-23  |  説明責任のみ（accountability only）/ Entry変更禁止 / Exit変更禁止 / 収益最適化禁止

**Strategy**: Study9 Case B (FROZEN)  **Entry**: RSR∈[92,95), days_cross90≤5, rsr_slope_5d≤5  **Exit**: RSR<90  **Capital/Authority/Execution**: 現行production configuration  **Governance**: annual_rebalance

**観測ウィンドウ**: 2018-01-01 → 2025-12-31（actual_R<0 全トレード, lookahead=20営業日）

**目的**: 負けトレードが「正常損失」か「回避可能損失」かを定量化する（Entry/Exit再設計は提案しない）

---
## 分類ロジック（優先順位順・排他的1ラベル）

| 優先 | カテゴリ | 条件 |
|---|---|---|
| 1 | REGIME_SHIFT | exit_reason=MKT_SHOCK または hold中にBull→Bear転換 |
| 2 | HIGH_VOL_ENTRY | entry時10d実現volが全トレード中の上位25%（閾値=0.0268） |
| 3 | LATE_ENTRY | entry時days_cross90 ≥ 4（許容範囲[1,5]の後半） |
| 4 | FALSE_BREAKOUT | peak_before_loss≤2% AND hold_days≤5d（追随なし） |
| 5 | REVERSAL_LOSS | peak_before_loss>3% AND drawdown>50%（往復負け） |
| 6 | NORMAL_LOSS | 上記非該当（通常の統計的分散） |

**集計区分**: structural=NORMAL_LOSS / regime=REGIME_SHIFT / avoidable=REVERSAL+FALSE_BO+HIGH_VOL+LATE（3区分は排他的に合計100%）。signal_failure_rate=FALSE_BO+HIGH_VOL+LATE（avoidableのうちEntry品質起因のサブセット、REVERSAL_LOSSは「良いEntryからのExitタイミング」問題として除外）。

---
## Executive Summary

| 指標 | 値 |
|---|---|
| trade_count（losers） | 22 |
| avoidable_loss_ratio | 77.3% |
| structural_loss_ratio | 18.2% |
| regime_loss_ratio | 4.5% |
| signal_failure_rate | 77.3% |
| recovery_rate | 27.3% |

**loss_source_breakdown**

| カテゴリ | 件数 | 割合 |
|---|---|---|
| REGIME_SHIFT | 1 | 4.5% |
| HIGH_VOL_ENTRY | 6 | 27.3% |
| LATE_ENTRY | 3 | 13.6% |
| FALSE_BREAKOUT | 8 | 36.4% |
| REVERSAL_LOSS | 0 | 0.0% |
| NORMAL_LOSS | 4 | 18.2% |

**recommend_entry_change: RESEARCH_ENTRY**

判定理由: avoidable_loss_ratio=77.3%>=60% AND signal_failure_rate=77.3%>=40%

---
## Decision Rules（適用基準）

| 判定 | 条件 |
|---|---|
| KEEP_ENTRY | avoidable_loss_ratio≤40% AND trade_count≥15 |
| RESEARCH_ENTRY | avoidable_loss_ratio≥60% AND signal_failure_rate≥40% AND trade_count≥15 |
| MONITOR_ENTRY | 上記以外（サンプル不足含む） |

---
## Trade Log（全件）

| # | 銘柄 | Entry | Exit | hold | days_to_loss | actual_R | entry_rsr | entry_slope | rank | regime | peak | DD | cf5d | cf10d | cf20d | recovery_p | category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4021.T | 2020-08-12 | 2020-08-18 | 3d | 0d | -1.93R | 92.7 | +0.0 | 4 | Bull | -12.61% | 3.4% | -11.37% | -10.59% | -15.57% | 0% | FALSE_BREAKOUT |
| 3 | 9104.T | 2021-03-18 | 2021-03-24 | 3d | 0d | -0.72R | 92.7 | +0.0 | 4 | Bull | -28.13% | 10.1% | -32.78% | -28.53% | -31.74% | 0% | FALSE_BREAKOUT |
| 4 | 9104.T | 2021-03-26 | 2021-04-01 | 3d | 0d | -2.83R | 92.7 | +0.0 | 4 | Bull | -34.39% | 6.2% | -30.66% | -35.16% | -26.82% | 0% | HIGH_VOL_ENTRY |
| 6 | 6762.T | 2022-09-14 | 2022-09-26 | 5d | 0d | -2.96R | 92.9 | +4.8 | 4 | Bull | -5.64% | 8.2% | -14.28% | -14.84% | -12.87% | 0% | FALSE_BREAKOUT |
| 7 | 3382.T | 2022-09-27 | 2022-10-03 | 3d | 0d | -1.63R | 92.9 | +2.4 | 4 | Bull | -5.56% | 5.8% | -15.96% | -11.02% | -10.49% | 0% | FALSE_BREAKOUT |
| 8 | 8058.T | 2022-10-04 | 2022-10-11 | 3d | 0d | -2.53R | 92.9 | +4.8 | 4 | Bull | -9.47% | 3.8% | -13.72% | -12.75% | -8.79% | 0% | HIGH_VOL_ENTRY |
| 9 | 8053.T | 2022-10-13 | 2022-10-19 | 3d | 0d | -0.56R | 92.9 | +2.4 | 4 | Bull | -10.14% | 1.8% | -9.39% | -6.07% | +2.11% | 33% | FALSE_BREAKOUT |
| 10 | 8354.T | 2022-10-20 | 2022-10-27 | 4d | 0d | -1.62R | 92.9 | -2.4 | 4 | Bull | -7.38% | 5.4% | -12.05% | -13.93% | -5.60% | 0% | FALSE_BREAKOUT |
| 11 | 2914.T | 2022-11-09 | 2022-11-15 | 3d | 0d | -1.20R | 92.9 | +4.8 | 4 | Bull | -16.81% | 2.6% | -16.47% | -16.85% | -16.64% | 0% | HIGH_VOL_ENTRY |
| 12 | 7013.T | 2022-12-12 | 2023-02-08 | 38d | 0d | -3.03R | 92.9 | +4.8 | 4 | Bull | -1.78% | 12.0% | -11.44% | -13.00% | -13.97% | 0% | NORMAL_LOSS |
| 14 | 8002.T | 2023-03-02 | 2023-03-16 | 9d | 0d | -3.01R | 92.9 | +2.4 | 4 | Bull | -5.08% | 12.0% | -15.05% | -9.26% | -1.10% | 33% | NORMAL_LOSS |
| 16 | 6506.T | 2023-04-03 | 2023-04-07 | 3d | 0d | -2.28R | 92.9 | +0.0 | 4 | Bull | -2.55% | 5.8% | -5.28% | -6.30% | -3.06% | 33% | FALSE_BREAKOUT |
| 21 | 8306.T | 2023-11-22 | 2023-11-29 | 3d | 0d | -0.41R | 92.9 | +4.8 | 3 | Bull | -4.02% | 3.2% | -7.77% | -5.69% | -10.33% | 0% | FALSE_BREAKOUT |
| 22 | 6857.T | 2023-12-01 | 2023-12-08 | 4d | 0d | -3.59R | 92.9 | +4.8 | 4 | Bull | -1.23% | 8.8% | +3.54% | +2.96% | +8.07% | 100% | LATE_ENTRY |
| 24 | 8058.T | 2024-03-22 | 2024-05-08 | 29d | 0d | -3.66R | 92.9 | +4.8 | 4 | Bull | -3.21% | 8.6% | -9.63% | -10.88% | -13.62% | 0% | LATE_ENTRY |
| 26 | 8725.T | 2024-05-22 | 2024-06-14 | 16d | 0d | -0.63R | 92.9 | +2.4 | 4 | Bull | +1.31% | 8.1% | -5.57% | +7.00% | +10.74% | 67% | HIGH_VOL_ENTRY |
| 27 | 8725.T | 2024-06-24 | 2024-08-02 | 27d | 0d | -5.45R | 92.9 | +2.4 | 4 | Bull | +13.45% | 21.9% | -13.02% | -4.07% | -1.85% | 67% | REGIME_SHIFT |
| 28 | 6702.T | 2024-09-12 | 2024-09-25 | 6d | 0d | -0.49R | 92.9 | +4.8 | 4 | Bull | -0.27% | 2.9% | +1.30% | +7.54% | +2.27% | 100% | LATE_ENTRY |
| 29 | 6857.T | 2024-09-26 | 2024-10-02 | 3d | 0d | -2.11R | 92.9 | +4.8 | 4 | Bull | +1.36% | 9.1% | +7.41% | +12.51% | +28.63% | 100% | HIGH_VOL_ENTRY |
| 31 | 7012.T | 2024-10-15 | 2024-10-21 | 3d | 1d | -2.06R | 92.9 | +0.0 | 4 | Bull | +0.85% | 5.8% | -14.15% | -10.03% | -3.35% | 33% | HIGH_VOL_ENTRY |
| 33 | 8411.T | 2025-01-28 | 2025-04-03 | 43d | 0d | -9.24R | 92.9 | +4.8 | 4 | Bull | +3.18% | 16.0% | -19.50% | -20.04% | -19.31% | 0% | NORMAL_LOSS |
| 35 | 6702.T | 2025-06-11 | 2025-07-08 | 18d | 0d | -2.30R | 92.9 | +2.4 | 4 | Bull | +0.61% | 4.2% | -8.60% | -9.06% | +1.07% | 33% | NORMAL_LOSS |

---
## 最終判定

| 指標 | 値 |
|---|---|
| trade_count | 22 |
| avoidable_loss_ratio | 77.3% |
| structural_loss_ratio | 18.2% |
| signal_failure_rate | 77.3% |
| loss_source_breakdown | {'REGIME_SHIFT': 1, 'HIGH_VOL_ENTRY': 6, 'LATE_ENTRY': 3, 'FALSE_BREAKOUT': 8, 'REVERSAL_LOSS': 0, 'NORMAL_LOSS': 4} |
| **recommend_entry_change** | **RESEARCH_ENTRY** |

研究目的は説明責任のみ。Entry/Exit再設計はこのレポートでは提案しない。
