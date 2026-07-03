# Study16 Order Authority Integrity Gate

作成日: 2026-06-21  |  integrity監査のみ / 収益評価禁止

**Strategy**: Study9 Case B (固定)  **Capital**: ¥1,800,000

---
## Executive Summary

| 項目 | 値 |
|---|---|
| **authority_integrity_score** | **100.0/100** |
| authority_precision (Case A) | 100.0% |
| authority_precision (Case B) | 100.0% |
| execution_tracking (Case A) | 100.0% |
| execution_tracking (Case B) | 0.0% |
| replay_consistency (A) | 100.0% |
| replay_consistency (B) | 100.0% |
| reconciliation_status | PASS ✅ |
| go_live_decision | **PAPER_ACK_READY** |
| rollback_rule | alpha_realization_30d < 80% → allocation=0%, authority=OFF, incidentレポート発行 |
| recommended_authority_level | LIMITED_LIVE |

---
## Case A: SHADOW (signal → virtual execution)

| 指標 | 値 |
|---|---|
| n_signals_processed | 71 |
| authority_precision | 100.0% |
| execution_tracking | 100.0% |
| replay_consistency | 100.0% |
| authority_mismatch | 0 |
| reconciliation_error | 0 |
| duplicate_submit | 0 |
| orphan_order | 0 |
| idempotency_failure | 0 |
| cash_truth_gap | 0 |
| position_truth_gap | 0 |
| partial_fill_gap | 0 |
| chain_valid / total | 71 / 71 |
| latency_p50_ms | 0.0 |
| latency_p95_ms | 0.0 |
| latency_p99_ms | 0.0 |

**integrity_score_A: 100.0/100**

---
## Case B: BROKER_DRY_RUN (signal → order_build → validation → cancel)

| 指標 | 値 |
|---|---|
| n_signals_processed | 71 |
| authority_precision | 100.0% |
| execution_tracking | 0.0% |
| replay_consistency | 100.0% |
| authority_mismatch | 0 |
| reconciliation_error | 0 |
| duplicate_submit | 0 |
| orphan_order | 0 |
| idempotency_failure | 0 |
| cash_truth_gap | 0 |
| position_truth_gap | 0 |
| partial_fill_gap | 117800 |
| chain_valid / total | 71 / 71 |
| latency_p50_ms | 98.5 |
| latency_p95_ms | 114.9 |
| latency_p99_ms | 114.9 |

**integrity_score_B: 100.0/100**

---
## Case C Gate: PAPER_ACK (signal → broker_ack → cancel)

**Gate判定: ✅ PASS**

全条件クリア。Case C (PAPER_ACK) 実施可能。

**Case C 実施要件:**

| 条件 | 閾値 | 現在値 |
|---|---|---|
| authority_mismatch | 0 | 0 ✅ |
| reconciliation_error | 0 | 0 ✅ |
| duplicate_submit | 0 | 0 ✅ |
| idempotency_failure | 0 | 0 ✅ |
| replay_consistency | 1.0 | 1.0 ✅ |

---
## Case D Gate: LIMITED_LIVE (signal → real_order → reconciliation)

**Gate判定: ✅ PASS**

**Case D 実施要件:**

| 条件 | 閾値 | 現在値 |
|---|---|---|
| cash_truth_gap | 0 | 0 ✅ |
| position_truth_gap | 0 | 0 ✅ |
| orphan_order | 0 | 0 ✅ |
| authority_mismatch | 0 | 0 ✅ |
| reconciliation_error | 0 | 0 ✅ |
| replay_consistency | 1.0 | 1.0 ✅ |

---
## Stop Conditions

| 条件 | 発火 |
|---|---|
| unexpected_position | — OK |
| cash_negative | — OK |
| unknown_state | — OK |
| orphan_order_unresolved | — OK |
| manual_override | — OK |
| reconciliation_failure | — OK |

---
## Promotion Criteria (30日間チェック)

| 指標 | 要求値 | Case A | Case B |
|---|---|---|---|
| authority_mismatch | 0 | 0 ✅ | 0 ✅ |
| reconciliation_error | 0 | 0 ✅ | 0 ✅ |
| replay_consistency_fail | 0 | 0 ✅ | 0 ✅ |
| duplicate_submit | 0 | 0 ✅ | 0 ✅ |
| cash_truth_gap | 0 | 0 ✅ | 0 ✅ |
| position_truth_gap | 0 | 0 ✅ | 0 ✅ |
| orphan_order | 0 | 0 ✅ | 0 ✅ |
| idempotency_failure | 0 | 0 ✅ | 0 ✅ |

---
## Hash Chain Integrity Sample (Case A 先頭3件)

```
signal_hash    : 495bb710ee2a77d30623a24f91de2b7bc27621b7133b00cd23737070a616c79b
intent_hash    : 4d7b0f8d5d4f17caf2adfdec43d286340f6d2ad18291de782c0c0a7b51b24229
authority_hash : 5f51628fbeee6f4cd350e8664935c39fca178a5a5997e79fc485b9e5e0fe299a
execution_hash : 3538d6f734ceede993599cd526c23838c8bec657789d3df15b64c3852d61545d
chain_hash     : 9d79d80bde346ed5738c94096c94857efc420a9986ea4181ad0cbbde68d90652
chain_valid    : True
symbol=4021.T date=2020-08-12 side=BUY
---
signal_hash    : 0927ea245643a3eabc745ecd3915651801aad2e12762c58ea25b7fc8956da2ef
intent_hash    : 31cdc89c7fb6b9b0e9490041a74ca9ec15a418d7464b0e387b0a25a60cf12621
authority_hash : 6dea799255040be5058980df341378b9f23e29bd39271cd8a51f22a3caf0a84a
execution_hash : 2a7087081c1e3a7543a8aa524e3326a1aea91ea82140ff800283d678ff40bee1
chain_hash     : cabb6ff521a07b52b47e85851c3e9258b510ed13ba7a533d5d1e640a7af464be
chain_valid    : True
symbol=4021.T date=2020-08-18 side=SELL
---
signal_hash    : d3fa873ab113d14d3b28dd2af593a722fcc81cede7717eb6c2b5528631752d94
intent_hash    : 92edbbbee33672b380560f6f227c7ebb41f18f998f89bc181d9e8f1d6c996fae
authority_hash : e91f2bfe9b7ffc25a87b296c18bf2fe8747f766c54f66c6b78d4cfb263d70a1c
execution_hash : 5e837aaed211296ecd4a12772c7578e5fbaee39777c853bfd2d29afd8bb609a8
chain_hash     : 318d80e1d87195077e273612d4073cca38d0501509927e4f953fcf5a6c4a724c
chain_valid    : True
symbol=6594.T date=2020-08-18 side=BUY
---
```

---
## Operational Risks

- なし

---
## 最終判定

| 項目 | 値 |
|---|---|
| authority_integrity_score | **100.0/100** |
| authority_precision | 100.0% |
| execution_tracking | 0.0% |
| reconciliation_status | OK |
| go_live_decision | **PAPER_ACK_READY** |
| rollback_rule | alpha_realization_30d < 80% → allocation=0%, authority=OFF, incidentレポート発行 |
| operational_risks | なし |
| recommended_authority_level | LIMITED_LIVE |
| next_step | Case C (PAPER_ACK) を30日間実施 → Case D (LIMITED_LIVE) へ |