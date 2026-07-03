# risk_pct 1.00% → 1.25% 変更 監査レポート

**実施日**: 2026-06-03  
**変更者**: Claude (ASK_FIRST 承認済み)  
**WF 検証**: CONDITIONAL (5-fold 60% pass / TrueOOS 2025 ΔCAGR 0.00%)

---

## 1. 変更ファイル一覧

| ファイル | 種別 | 変更箇所 |
|---------|------|---------|
| `src/kabusapi/signal_bridge.py` | **Production** | 3行 |
| `tools/sizing_audit_csv.py` | Tool | 1行 |
| `tools/post_sync_sizing_audit.py` | Tool | 1行 |
| `tools/opportunity_cost_audit.py` | Tool | 1行 |
| `tools/capital_utilization_audit.py` | Tool | 1行 |
| `tools/replacement_opportunity_audit.py` | Tool | 1行 |

---

## 2. 変更前後 diff (signal_bridge.py)

```diff
# Line 2777 — shadow order sizing
-            _risk_yen = self.capital * 0.01
+            _risk_yen = self.capital * 0.0125

# Line 3245 — main BUY order sizing
-            _risk_pct      = 0.01                              # 1% リスク
+            _risk_pct      = 0.0125                            # 1.25% リスク

# Line 4558 — portfolio_summary telemetry
-            "risk_per_trade_yen":  round(self.capital * 0.01, 0),
+            "risk_per_trade_yen":  round(self.capital * 0.0125, 0),
```

---

## 3. 現在の資本状態

| レイヤ | 値 |
|--------|---|
| actual_equity | ¥4,015,209 (2026-06-01 equity_snapshots) |
| deployable_capital (×0.90) | ¥3,613,688 |
| position_budget (×25%) | ¥903,422 |
| **risk_per_trade BEFORE** | **¥27,000** (2,700,000 × 1.00%) |
| **risk_per_trade AFTER** | **¥45,171** (3,613,688 × 1.25%) |

---

## 4. 重点4銘柄 Before/After 比較

| symbol | sector | price | ATR20 | RSR | B_qcap | B_qrisk | B_qfin | B_bind | A_qcap | A_qrisk | A_qfin | A_bind | Δqty |
|--------|--------|-------|-------|-----|--------|---------|--------|--------|--------|---------|--------|--------|------|
| 6762.T | 電機精密 | ¥4,108 | 201.4 | 96.8 | 100 | 100 | 100 | equal | 200 | 200 | **200** | equal | **+100** |
| 5301.T | 化学 | ¥1,852 | 79.5 | 100.0 | 300 | 300 | 300 | equal | 400 | 500 | **400** | qty_cap | **+100** |
| 7182.T | 銀行 | ¥3,069 | 102.7 | 85.5 | 200 | 200 | 200 | equal | 200 | 400 | **200** | qty_cap | **0** |
| 8750.T | 保険 | ¥1,634 | 49.7 | 83.3 | 400 | 500 | 400 | qty_cap | 500 | 900 | **500** | qty_cap | **+100** |

**7182.T は qty_cap (200株) が binding のため変化なし。**

---

## 5. 全57銘柄集計

| 指標 | BEFORE | AFTER |
|------|--------|-------|
| qty増加銘柄数 | — | **39銘柄 (68%)** |
| qty不変銘柄数 | — | 18銘柄 (32%) |
| qty減少銘柄数 | — | 0銘柄 |
| price_too_high | 7銘柄 | **1銘柄** (6981.T のみ) |
| qty_risk binding率 | 5/57 (9%) | 3/57 (5%) ↓ |
| qty_cap binding率 | 26/57 (46%) | 42/57 (74%) ↑ |

**binding の主役が qty_risk → qty_cap に移行完了。**  
risk_pct 増加の効果として qty_risk 制約が解消し、position_budget が律速になった。

---

## 6. 安全確認

| チェック項目 | 結果 |
|------------|------|
| qty_final < 100 になる銘柄が増えたか | ✅ **PASS** — 0件 → 0件 (変化なし) |
| price_too_high 銘柄が増えたか | ✅ **PASS** — 7件 → 1件 (減少) |
| max_single_weight 超過なし | ✅ **PASS** — 全銘柄 position_budget 内 |
| deployable_capital 超過なし | ✅ **PASS** — max_invest ≤ deployable_capital |

---

## 7. テスト結果

| スコープ | 結果 |
|---------|------|
| `tests/broker/` 14件 | ✅ 全通過 |
| `tests/capital/` 294件 | ✅ 全通過 |
| `tests/live/ + tests/runtime/` 1173件 | ✅ 全通過 |
| 全テスト (pre-existing 除く) 8,595件 | ✅ 全通過 |
| 新規 regression | **0件** |

**pre-existing failures (変更前から存在)**:
- `test_emergency_stop.py::test_live_mode_executes_orders` — runtime/send_idem_confirmed.json 状態汚染 (7203.T)
- `test_risk_flag.py` 2件 — symbol_cap config 値不整合
- `test_order_ledger.py` 4件 — market order price フィールド
- `test_sector_gate.py` 1件 — sector_concentration config 不整合
- `test_state_update.py` 2件 — 状態更新ロジック

---

## 8. 変更の影響サマリー

```
risk_pct 1.00% → 1.25% の実効果:

  [即時効果]
  - risk_per_trade: ¥27,000 → ¥45,171 (+¥18,171)
  - qty_risk binding 率: 9% → 5% (binding 解消)
  - 39銘柄でロット増加 (68%の銘柄)

  [不変]
  - position_budget, qty_cap は risk_pct と無関係 → 変化なし
  - 7182.T: qty_cap=200 が binding のため qfin 変化なし
  - qty_final < 100 の銘柄: ゼロ維持

  [注意点]
  - CB_ACTIVE 中 (〜2026-06-30) のため新規 BUY には適用されない
  - CB 解除後の初回シグナルから 1.25% が有効になる
  - WF CONDITIONAL: TrueOOS 2025 での CAGR 改善はゼロ
    (2025年は全件 qty_cap binding だったため実効差なし)
```

---

**結論: 変更適用完了。全安全チェック PASS。CB解除 (2026-06-30) 後から実効。**
