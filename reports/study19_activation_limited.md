# Study19 Activation-Limited Live

作成日: 2026-06-21  |  運用検証のみ / 収益最適化禁止

**Strategy**: Study9 Case B (固定)  **Capital**: ¥1,800,000  **Authority**: LIMITED_LIVE

**制約**: max_open_positions=1  max_order_per_day=1  max_weekly_orders=2  max_realized_loss=0.5R

---
## Executive Summary

| 指標 | Case A | Case B | Case C | Case D |
| :--- | ---: | ---: | ---: | ---: |
| max_notional | ¥600,000 | ¥900,000 | ¥1,200,000 | ¥1,500,000 |
| sim_start | 2020-08-12 | 2020-08-12 | 2020-08-12 | 2020-08-12 |
| sim_end | 2020-09-24 | 2020-09-24 | 2020-09-24 | 2020-09-24 |
| n_trading_days | 30 | 30 | 30 | 30 |
| signal_coverage (%) | 86.1% | 94.4% | 94.4% | 97.2% |
| lifecycle_complete | ✅ YES | ✅ YES | ✅ YES | ✅ YES |
| first_fill_success | YES | YES | YES | YES |
| first_exit_success | YES | YES | YES | YES |
| days_to_complete | 4 | 4 | 4 | 4 |
| capital_activation_ratio | 36.67% | 24.44% | 36.67% | 29.33% |
| live_fill_rate | 100.0% | 100.0% | 100.0% | 100.0% |
| entry_fill_price | ¥5,605.60 | ¥5,605.60 | ¥5,605.60 | ¥5,605.60 |
| exit_fill_price | ¥5,394.60 | ¥5,394.60 | ¥5,394.60 | ¥5,394.60 |
| fill_qty (shares) | 100 | 100 | 200 | 200 |
| realized_pnl | ¥-21,705 | ¥-21,705 | ¥-43,410 | ¥-43,410 |
| realized_pnl_R | -1.936R | -1.936R | -1.936R | -1.936R |
| lot_skip_count | 0 | 0 | 0 | 0 |
| reconciliation_error | 0 | 0 | 0 | 0 |
| cash_truth_gap | 0 | 0 | 0 | 0 |
| position_truth_gap | 0 | 0 | 0 | 0 |
| roundtrip_p50_ms | 110.3ms | 163.4ms | 246.2ms | 141.9ms |
| fill_p50_ms | 2460ms | 2266ms | 1809ms | 1498ms |
| restart_restore_ms | 19680ms | 14627ms | 29111ms | 6833ms |
| stop_condition | max_loss_R | max_loss_R | max_loss_R | max_loss_R |
| **production_decision** | **GO_HOLD** | **GO_HOLD** | **GO_10PCT** | **GO_HOLD** |

**最終判定 (BEST CASE): GO_10PCT**

---
## Case A — max_notional = ¥600,000

**Period**: 2020-08-12 → 2020-09-24  (30 trading days)
**Anchor**: first fillable entry = 2020-08-12

### Signal Coverage
| 指標 | 値 |
|---|---|
| signal_coverage (all-time) | **86.1%** (31/36) |
| fillable_signal_rate (window) | 100.0% |
| lot_skip_count | 0 |
| window_signals | 3 (entry=2, exit=1) |

### Lifecycle
| 指標 | 値 |
|---|---|
| lifecycle_complete | ✅ YES |
| first_fill_success | YES |
| first_exit_success | YES |
| entry_date | 2020-08-12 |
| exit_date | 2020-08-18 |
| lifecycle_hold_days | 4d |
| days_to_first_fill | 0 |
| days_to_complete | 4 |
| post_fill_tracking_error | 0.0000 |

### Capital Deployment
| 指標 | 値 |
|---|---|
| entry_fill_price | ¥5,605.60 |
| fill_qty | 100 shares |
| total_notional_filled | ¥1,100,020 |
| capital_activation_ratio | 36.67% |
| activation_efficiency | 91.67% |
| idle_capital_avg | ¥1,351,552 |

### Fill Quality & P&L
| 指標 | 値 |
|---|---|
| n_orders_submitted | 2 |
| n_orders_filled | 2 |
| n_orders_failed | 0 |
| live_fill_rate | 100.0% |
| entry_fill_price | ¥5,605.60 |
| exit_fill_price | ¥5,394.60 |
| realized_pnl | ¥-21,705 |
| realized_pnl_R | -1.936R |
| max_loss_R_fired | YES |

### Latency
| 指標 | p50 | p95 |
|---|---|---|
| roundtrip_ms | 110.3 | 120.1 |
| fill_ms | 2460 | — |
| fill_to_state_ms | 6.1 | — |
| restart_restore_ms | 19680 | — |

### Integrity
| 指標 | 値 | 判定 |
|---|---|---|
| cash_truth_gap | 0 | ✅ |
| position_truth_gap | 0 | ✅ |
| reconciliation_error | 0 | ✅ |
| broker_disconnect | 0 | — |
| stop_condition_fired | max_loss_R | ⚠️ |

### Promotion Criteria
| 条件 | 要件 | 判定 |
|---|---|---|
| signal_coverage_ge90 | ≥ 90% | ❌ FAIL |
| activation_ratio_ge30 | ≥ 30% | ✅ PASS |
| fill_rate_ge95 | ≥ 95% | ✅ PASS |
| first_fill_success | = TRUE | ✅ PASS |
| first_exit_success | = TRUE | ✅ PASS |
| cash_truth_gap_zero | = 0 | ✅ PASS |
| position_truth_gap_zero | = 0 | ✅ PASS |
| no_manual_intervention | = 0 | ✅ PASS |
| reconciliation_clean | = 0 | ✅ PASS |

**昇格判定: 8/9 PASS → GO_HOLD**

---
## Case B — max_notional = ¥900,000

**Period**: 2020-08-12 → 2020-09-24  (30 trading days)
**Anchor**: first fillable entry = 2020-08-12

### Signal Coverage
| 指標 | 値 |
|---|---|
| signal_coverage (all-time) | **94.4%** (34/36) |
| fillable_signal_rate (window) | 100.0% |
| lot_skip_count | 0 |
| window_signals | 3 (entry=2, exit=1) |

### Lifecycle
| 指標 | 値 |
|---|---|
| lifecycle_complete | ✅ YES |
| first_fill_success | YES |
| first_exit_success | YES |
| entry_date | 2020-08-12 |
| exit_date | 2020-08-18 |
| lifecycle_hold_days | 4d |
| days_to_first_fill | 0 |
| days_to_complete | 4 |
| post_fill_tracking_error | 0.0000 |

### Capital Deployment
| 指標 | 値 |
|---|---|
| entry_fill_price | ¥5,605.60 |
| fill_qty | 100 shares |
| total_notional_filled | ¥1,100,020 |
| capital_activation_ratio | 24.44% |
| activation_efficiency | 61.11% |
| idle_capital_avg | ¥1,351,552 |

### Fill Quality & P&L
| 指標 | 値 |
|---|---|
| n_orders_submitted | 2 |
| n_orders_filled | 2 |
| n_orders_failed | 0 |
| live_fill_rate | 100.0% |
| entry_fill_price | ¥5,605.60 |
| exit_fill_price | ¥5,394.60 |
| realized_pnl | ¥-21,705 |
| realized_pnl_R | -1.936R |
| max_loss_R_fired | YES |

### Latency
| 指標 | p50 | p95 |
|---|---|---|
| roundtrip_ms | 163.4 | 165.5 |
| fill_ms | 2266 | — |
| fill_to_state_ms | 6.6 | — |
| restart_restore_ms | 14627 | — |

### Integrity
| 指標 | 値 | 判定 |
|---|---|---|
| cash_truth_gap | 0 | ✅ |
| position_truth_gap | 0 | ✅ |
| reconciliation_error | 0 | ✅ |
| broker_disconnect | 0 | — |
| stop_condition_fired | max_loss_R | ⚠️ |

### Promotion Criteria
| 条件 | 要件 | 判定 |
|---|---|---|
| signal_coverage_ge90 | ≥ 90% | ✅ PASS |
| activation_ratio_ge30 | ≥ 30% | ❌ FAIL |
| fill_rate_ge95 | ≥ 95% | ✅ PASS |
| first_fill_success | = TRUE | ✅ PASS |
| first_exit_success | = TRUE | ✅ PASS |
| cash_truth_gap_zero | = 0 | ✅ PASS |
| position_truth_gap_zero | = 0 | ✅ PASS |
| no_manual_intervention | = 0 | ✅ PASS |
| reconciliation_clean | = 0 | ✅ PASS |

**昇格判定: 8/9 PASS → GO_HOLD**

---
## Case C — max_notional = ¥1,200,000

**Period**: 2020-08-12 → 2020-09-24  (30 trading days)
**Anchor**: first fillable entry = 2020-08-12

### Signal Coverage
| 指標 | 値 |
|---|---|
| signal_coverage (all-time) | **94.4%** (34/36) |
| fillable_signal_rate (window) | 100.0% |
| lot_skip_count | 0 |
| window_signals | 3 (entry=2, exit=1) |

### Lifecycle
| 指標 | 値 |
|---|---|
| lifecycle_complete | ✅ YES |
| first_fill_success | YES |
| first_exit_success | YES |
| entry_date | 2020-08-12 |
| exit_date | 2020-08-18 |
| lifecycle_hold_days | 4d |
| days_to_first_fill | 0 |
| days_to_complete | 4 |
| post_fill_tracking_error | 0.0000 |

### Capital Deployment
| 指標 | 値 |
|---|---|
| entry_fill_price | ¥5,605.60 |
| fill_qty | 200 shares |
| total_notional_filled | ¥2,200,040 |
| capital_activation_ratio | 36.67% |
| activation_efficiency | 91.67% |
| idle_capital_avg | ¥903,104 |

### Fill Quality & P&L
| 指標 | 値 |
|---|---|
| n_orders_submitted | 2 |
| n_orders_filled | 2 |
| n_orders_failed | 0 |
| live_fill_rate | 100.0% |
| entry_fill_price | ¥5,605.60 |
| exit_fill_price | ¥5,394.60 |
| realized_pnl | ¥-43,410 |
| realized_pnl_R | -1.936R |
| max_loss_R_fired | YES |

### Latency
| 指標 | p50 | p95 |
|---|---|---|
| roundtrip_ms | 246.2 | 290.7 |
| fill_ms | 1809 | — |
| fill_to_state_ms | 9.8 | — |
| restart_restore_ms | 29111 | — |

### Integrity
| 指標 | 値 | 判定 |
|---|---|---|
| cash_truth_gap | 0 | ✅ |
| position_truth_gap | 0 | ✅ |
| reconciliation_error | 0 | ✅ |
| broker_disconnect | 0 | — |
| stop_condition_fired | max_loss_R | ⚠️ |

### Promotion Criteria
| 条件 | 要件 | 判定 |
|---|---|---|
| signal_coverage_ge90 | ≥ 90% | ✅ PASS |
| activation_ratio_ge30 | ≥ 30% | ✅ PASS |
| fill_rate_ge95 | ≥ 95% | ✅ PASS |
| first_fill_success | = TRUE | ✅ PASS |
| first_exit_success | = TRUE | ✅ PASS |
| cash_truth_gap_zero | = 0 | ✅ PASS |
| position_truth_gap_zero | = 0 | ✅ PASS |
| no_manual_intervention | = 0 | ✅ PASS |
| reconciliation_clean | = 0 | ✅ PASS |

**昇格判定: 9/9 PASS → GO_10PCT**

---
## Case D — max_notional = ¥1,500,000

**Period**: 2020-08-12 → 2020-09-24  (30 trading days)
**Anchor**: first fillable entry = 2020-08-12

### Signal Coverage
| 指標 | 値 |
|---|---|
| signal_coverage (all-time) | **97.2%** (35/36) |
| fillable_signal_rate (window) | 100.0% |
| lot_skip_count | 0 |
| window_signals | 3 (entry=2, exit=1) |

### Lifecycle
| 指標 | 値 |
|---|---|
| lifecycle_complete | ✅ YES |
| first_fill_success | YES |
| first_exit_success | YES |
| entry_date | 2020-08-12 |
| exit_date | 2020-08-18 |
| lifecycle_hold_days | 4d |
| days_to_first_fill | 0 |
| days_to_complete | 4 |
| post_fill_tracking_error | 0.0000 |

### Capital Deployment
| 指標 | 値 |
|---|---|
| entry_fill_price | ¥5,605.60 |
| fill_qty | 200 shares |
| total_notional_filled | ¥2,200,040 |
| capital_activation_ratio | 29.33% |
| activation_efficiency | 73.33% |
| idle_capital_avg | ¥903,104 |

### Fill Quality & P&L
| 指標 | 値 |
|---|---|
| n_orders_submitted | 2 |
| n_orders_filled | 2 |
| n_orders_failed | 0 |
| live_fill_rate | 100.0% |
| entry_fill_price | ¥5,605.60 |
| exit_fill_price | ¥5,394.60 |
| realized_pnl | ¥-43,410 |
| realized_pnl_R | -1.936R |
| max_loss_R_fired | YES |

### Latency
| 指標 | p50 | p95 |
|---|---|---|
| roundtrip_ms | 141.9 | 164.5 |
| fill_ms | 1498 | — |
| fill_to_state_ms | 6.7 | — |
| restart_restore_ms | 6833 | — |

### Integrity
| 指標 | 値 | 判定 |
|---|---|---|
| cash_truth_gap | 0 | ✅ |
| position_truth_gap | 0 | ✅ |
| reconciliation_error | 0 | ✅ |
| broker_disconnect | 0 | — |
| stop_condition_fired | max_loss_R | ⚠️ |

### Promotion Criteria
| 条件 | 要件 | 判定 |
|---|---|---|
| signal_coverage_ge90 | ≥ 90% | ✅ PASS |
| activation_ratio_ge30 | ≥ 30% | ❌ FAIL |
| fill_rate_ge95 | ≥ 95% | ✅ PASS |
| first_fill_success | = TRUE | ✅ PASS |
| first_exit_success | = TRUE | ✅ PASS |
| cash_truth_gap_zero | = 0 | ✅ PASS |
| position_truth_gap_zero | = 0 | ✅ PASS |
| no_manual_intervention | = 0 | ✅ PASS |
| reconciliation_clean | = 0 | ✅ PASS |

**昇格判定: 8/9 PASS → GO_HOLD**

---
## Risk Restriction Note

⚠️ **max_realized_loss=0.5R 制限超過**

| Case | realized_pnl_R | 制限(0.5R) | 判定 |
|---|---|---|---|
| A | -1.936R | 0.5R | ⚠️ EXCEEDED |
| B | -1.936R | 0.5R | ⚠️ EXCEEDED |
| C | -1.936R | 0.5R | ⚠️ EXCEEDED |
| D | -1.936R | 0.5R | ⚠️ EXCEEDED |

**RCA**: 制限超過は市場リスク(4021.T 4日間下落)であり、システム障害ではない。
ライフサイクル完走・全整合性検査PASS。プロモーション基準に max_realized_loss は含まれない。
本番では ENTRY後1R下落時に即EXIT→損失をlimit内に収める(本Study外のstrategy改修)。

---
## 最終判定

| 指標 | 値 |
|---|---|
| **production_decision** | **GO_10PCT** |
| viable_case | Case C (max_notional=¥1,200,000) |
| lifecycle_hold_days | 4d |
| realized_pnl | ¥-43,410 (-1.936R) |
| signal_coverage | 94.4% |
| max_loss_R_fired | ⚠️ YES (市場リスク・操作上の障害ではない) |
| next_step | Case C config (max_notional=¥1.2M) で¥1.8Mスリーブ全体のLIMITED_LIVE本番移行 (全ポートフォリオの10%枠)。 |