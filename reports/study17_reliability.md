# Study17 Production Reliability Validation

作成日: 2026-06-21  |  信頼性監査のみ / 収益評価禁止

**Strategy**: Study9 Case B (固定)  **Capital**: ¥1,800,000  **Authority**: PAPER_ACK

**期間**: 2025-06-17 → 2025-07-29  (30 trading days)

---
## Executive Summary

| 指標 | 値 |
|---|---|
| **reliability_score** | **98.0/100** |
| **authority_integrity_score** | **100.0/100** |
| **execution_integrity_score** | **100.0/100** |
| **state_consistency_score** | **100.0/100** |
| **production_readiness** | **GO_LIMITED_LIVE** |
| recommended_next_authority | LIMITED_LIVE |
| stop_condition_fired | — なし |

---
## API / Connectivity

| 指標 | 値 |
|---|---|
| uptime_pct | 100.00% |
| api_available_days | 30 / 30 |
| broker_reconnect_count | 2 |
| api_timeout_count | 0 |
| session_recovery_attempts | 0 |
| session_recovery_success | 0 (100.0%) |
| state_restore_attempts | 2 |
| state_restore_success | 2 (100.0%) |

---
## Latency Distribution (PAPER_ACK)

| 指標 | p50 | p95 | p99 |
|---|---|---|---|
| order_ack_latency_ms | 99.0 | 100.4 | 100.6 |
| cancel_ack_latency_ms | 63.9 | 64.8 | 64.9 |

サンプル数: order_ack=2  cancel_ack=2

---
## Order Flow (PAPER_ACK)

| 指標 | 値 |
|---|---|
| signals_in_30d_window | 2 |
| signals_total_period | 71 |
| approved_signals | 2 |
| cancelled_signals | 2 |
| cancel_completeness | 100.0% |
| authority_precision | 100.0% |
| partial_fill_count | 0 ✅ |
| intent_execution_divergence | 0 ✅ |

---
## Integrity Metrics

| 指標 | 値 | 判定 |
|---|---|---|
| chain_break_count | 0 | ✅ |
| duplicate_submit_count | 0 | ✅ |
| orphan_order_count | 0 | ✅ |
| manual_override_count | 0 | ✅ |
| cash_truth_gap | 0 | ✅ |
| position_truth_gap | 0 | ✅ |
| reconciliation_error_count | 0 | ✅ |
| intent_execution_divergence | 0 | ✅ |

---
## Stop Conditions

| 条件 | 閾値 | 判定 |
|---|---|---|
| unexpected_position | ≥1 → ABORT | 0 — ✅ OK |
| cash_negative | ≥1 → ABORT | 0 — ✅ OK |
| position_truth_gap | ≠0 → ABORT | 0 — ✅ OK |
| cash_truth_gap | ≠0 → ABORT | 0 — ✅ OK |
| orphan_order_unresolved | ≥1 → ABORT | 0 — ✅ OK |
| duplicate_submit | ≥1 → ABORT | 0 — ✅ OK |
| chain_break | ≥1 → ABORT | 0 — ✅ OK |
| manual_override | ≥1 → ABORT | 0 — ✅ OK |

---
## Promotion Criteria (30日間 all=0 必須)

| 指標 | 要求 | 実績 | 判定 |
|---|---|---|---|
| authority_mismatch | 0 | 0 | ✅ |
| reconciliation_error | 0 | 0 | ✅ |
| replay_consistency_fail | 0 | 0 | ✅ |
| duplicate_submit | 0 | 0 | ✅ |
| cash_truth_gap | 0 | 0 | ✅ |
| position_truth_gap | 0 | 0 | ✅ |
| orphan_order | 0 | 0 | ✅ |
| idempotency_failure | 0 | 0 | ✅ |

**昇格判定: ✅ PASS — LIMITED_LIVEへの昇格条件クリア**

---
## Rollback Conditions

- alpha_realization_30d < 80% → allocation=0%, authority=OFF, incident report
- chain_break >= 1 → ABORT immediately
- cash_truth_gap != 0 → ABORT immediately
- position_truth_gap != 0 → ABORT immediately
- orphan_order_unresolved >= 1 → ABORT immediately

---
## Operational Risk Register

| # | リスク |
|---|---|
| 1 | なし — 全項目正常範囲 |

---
## Daily Monitoring Summary (最終10日)

| Day | Date | API | Reconnects | Signals | Approved | Chain OK | Reliability |
|---|---|---|---|---|---|---|---|
| 21 | 2025-07-15 | ✅ | 0 | 0 | 0 | 0 | 98.0 |
| 22 | 2025-07-16 | ✅ | 0 | 0 | 0 | 0 | 98.0 |
| 23 | 2025-07-17 | ✅ | 0 | 0 | 0 | 0 | 98.0 |
| 24 | 2025-07-18 | ✅ | 0 | 0 | 0 | 0 | 98.0 |
| 25 | 2025-07-22 | ✅ | 0 | 0 | 0 | 0 | 98.0 |
| 26 | 2025-07-23 | ✅ | 0 | 0 | 0 | 0 | 98.0 |
| 27 | 2025-07-24 | ✅ | 0 | 0 | 0 | 0 | 98.0 |
| 28 | 2025-07-25 | ✅ | 0 | 0 | 0 | 0 | 98.0 |
| 29 | 2025-07-28 | ✅ | 0 | 0 | 0 | 0 | 98.0 |
| 30 | 2025-07-29 | ✅ | 0 | 1 | 1 | 1 | 98.0 |

---
## 最終判定

| 項目 | 値 |
|---|---|
| reliability_score | **98.0/100** |
| authority_integrity_score | **100.0/100** |
| execution_integrity_score | **100.0/100** |
| state_consistency_score | **100.0/100** |
| production_readiness | **GO_LIMITED_LIVE** |
| recommended_next_authority | **LIMITED_LIVE** |
| rollback_rule | alpha_realization_30d<80% → allocation=0%, authority=OFF |
| operational_risks | なし — 全項目正常範囲 |
| stop_condition_fired | なし |
| **next_step** | **Case D (LIMITED_LIVE) 実施可能 — ¥1.8M上限・最大1枠** |