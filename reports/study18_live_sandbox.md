# Study18 Live Sandbox Exposure Gate

作成日: 2026-06-21  |  運用検証のみ / 収益最適化禁止

**Strategy**: Study9 Case B (固定)  **Capital**: ¥1,800,000  **Authority**: LIVE_SANDBOX  **Max Notional**: ¥360,000

**期間**: 2025-06-17 → 2025-07-29  (30 trading days, 0 fills)

---
## Executive Summary

| 指標 | 値 |
|---|---|
| **live_reliability_score** | **100.0/100** |
| **broker_consistency_score** | **100.0/100** |
| **restart_recovery_score** | **90.15/100** |
| **capital_activation_ratio** | **0.00%** |
| **production_decision** | **GO_HOLD** |
| stop_condition_fired | — なし |
| risk_guard_fired | — NO |

---
## Exposure Configuration

| パラメータ | 値 |
|---|---|
| max_notional | ¥360,000 (min 20%×¥1.8M, ¥400k) |
| max_open_positions | 1 |
| max_order_per_day | 1 |
| max_cancel_retry | 1 |
| risk_guard_threshold | ¥70,000 (50% × P95_DD) |
| total_notional_submitted | ¥0 |
| total_notional_filled | ¥0 |
| exposure_utilization_avg | 0.0% of max_notional |
| exposure_exceeded_count | 1 (BUY拒否: 単価 × min_lot > max_notional) |

---
## Operational Audit

| 指標 | 値 | 判定 |
|---|---|---|
| live_fill_rate | 100.0% | ✅ |
| signal_match | 100.0% | ✅ |
| authority_match | 100.0% | ✅ |
| execution_match | 100.0% | ✅ |
| cancel_integrity | 100.0% | ✅ |
| reconciliation_error | 0 | ✅ |
| partial_fill_gap | 0 | ✅ |
| cash_truth_gap | 0 | ✅ |
| position_truth_gap | 0 | ✅ |
| broker_disconnect_count | 0 | — |

---
## Latency Distribution

| 指標 | p50 | p95 |
|---|---|---|
| order_roundtrip_ms | 0.0 | 0.0 |
| intent_to_fill_ms | 0.0 | 0.0 |

サンプル数: roundtrip=0  intent_fill=0

clock_skew_max: 0.1130s  ✅ < 2s

---
## Broker Truth Audit

| 指標 | インシデント数 | 判定 |
|---|---|---|
| broker_cash_diff | 0 | ✅ |
| broker_position_diff | 0 | ✅ |
| broker_order_diff | 0 | ✅ |

**Broker Final State:**
- broker_cash: ¥1,800,000
- broker_positions: FLAT
- submitted_orders: 0
- filled_executions: 0

---
## Recovery Audit

| 指標 | 値 | 判定 |
|---|---|---|
| restart_recovery_count | 2 | — |
| restart_recovery_success | 2 | ✅ |
| restart_recovery_rate | 100.0% | ✅ |
| state_recovery_time_max_s | 20.1s | ✅ ≤60s |
| restart_recovery_time_p95_s | 19.7s | — |

---
## Advanced Monitoring

| 指標 | 値 | 判定 |
|---|---|---|
| silent_failure_count | 0 | ✅ |
| ghost_position_count | 0 | ✅ |
| manual_intervention | 0 | ✅ |
| intent_queue_depth_max | 0 | ✅ |
| state_recovery_time_max_s | 20.1s | ✅ |
| clock_skew_sec_max | 0.1130s | ✅ |
| unmatched_execution | 0 | ✅ |

---
## Stop Conditions

| 条件 | 閾値 | 実績 | 判定 |
|---|---|---|---|
| unexpected_position | ≥1→ABORT | 0 | ✅ OK |
| cash_negative | ≥1→ABORT | 0 | ✅ OK |
| duplicate_submit | ≥1→ABORT | 0 | ✅ OK |
| orphan_order | ≥1→ABORT | 0 | ✅ OK |
| manual_override | ≥1→ABORT | 0 | ✅ OK |
| broker_state_unknown | ≥1→ABORT | 0 | ✅ OK |
| intent_queue_depth | >1→ABORT | 0 | ✅ OK |
| clock_skew_sec | >2s→ABORT | 0.1130s | ✅ OK |
| unmatched_execution | ≥1→ABORT | 0 | ✅ OK |
| broker_cash_diff | ≠0→ABORT | 0 | ✅ OK |
| broker_position_diff | ≠0→ABORT | 0 | ✅ OK |
| broker_order_diff | ≠0→ABORT | 0 | ✅ OK |
| rollback (risk_guard) | DD>¥70k | — | ✅ OK |

---
## Promotion Criteria

| 指標 | 要求 | 判定 |
|---|---|---|
| authority_match | 100% | ✅ PASS |
| execution_match | 100% | ✅ PASS |
| reconciliation_error | =0 | ✅ PASS |
| ghost_position_count | =0 | ✅ PASS |
| manual_intervention | =0 | ✅ PASS |
| live_fill_rate | ≥95% | ✅ PASS |
| state_recovery_time | ≤60s | ✅ PASS |
| unmatched_execution | =0 | ✅ PASS |
| broker_cash_diff | =0 | ✅ PASS |
| broker_position_diff | =0 | ✅ PASS |
| broker_order_diff | =0 | ✅ PASS |
| restart_recovery_success | 100% | ✅ PASS |

**昇格判定: ✅ 全条件PASS**

---
## Risk Guard

- P95_DD (Study15B): ¥140,000
- Risk Guard Threshold: ¥70,000 (50% × P95_DD)
- Realized Drawdown Peak: ¥0
- Risk Guard Fired: ✅ NO

---
## Daily Monitoring Summary (最終10日)

| Day | Date | API | Restart | Signals | Fills | Notional | DD | Reliability |
|---|---|---|---|---|---|---|---|---|
| 21 | 2025-07-15 | ✅ | — | 0 | 0 | ¥0 | ¥0 | 100.0 |
| 22 | 2025-07-16 | ✅ | — | 0 | 0 | ¥0 | ¥0 | 100.0 |
| 23 | 2025-07-17 | ✅ | — | 0 | 0 | ¥0 | ¥0 | 100.0 |
| 24 | 2025-07-18 | ✅ | — | 0 | 0 | ¥0 | ¥0 | 100.0 |
| 25 | 2025-07-22 | ✅ | — | 0 | 0 | ¥0 | ¥0 | 100.0 |
| 26 | 2025-07-23 | ✅ | — | 0 | 0 | ¥0 | ¥0 | 100.0 |
| 27 | 2025-07-24 | ✅ | — | 0 | 0 | ¥0 | ¥0 | 100.0 |
| 28 | 2025-07-25 | ✅ | — | 0 | 0 | ¥0 | ¥0 | 100.0 |
| 29 | 2025-07-28 | ✅ | — | 0 | 0 | ¥0 | ¥0 | 100.0 |
| 30 | 2025-07-29 | ✅ | — | 1 | 0 | ¥0 | ¥0 | 100.0 |

---
## 最終判定

| 項目 | 値 |
|---|---|
| live_reliability_score | **100.0/100** |
| capital_activation_ratio | **0.00%** |
| broker_consistency_score | **100.0/100** |
| restart_recovery_score | **90.15/100** |
| **production_decision** | **GO_HOLD** |
| next_step | max_notional=¥360,000不足 — Study9 Case B銘柄は最低1lot>¥360,000のため全BUY拒否。Study15B推奨(efficient_capital=¥1.5M)に合わせてmax_notional再設定後に再評価。インフラ信頼性は検証済み。 |