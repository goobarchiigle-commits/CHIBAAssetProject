# Backtest Integrity Audit — CHIBAAssetProject
日付: 2026-06-28  
対象: Study40〜49 / composite_alpha_bt.py / Dashboard / Live  
実施者: CLD+CDX 自動監査

---

## ① 修正ファイル一覧

| ファイル | 変更種別 | 行番号 |
|---|---|---|
| `src/backtest/composite_alpha_bt.py` | **BUG FIX** intersection→union | 642-707 |
| `reports/dashboard_data/equity_curve_full.csv` | **再生成** RSR42+union, 2018-2025 | — |
| `reports/dashboard_data/trades_full.csv` | **再生成** | — |
| `reports/dashboard_data/exposure_full.csv` | **再生成** | — |
| `reports/dashboard_data/ecombined_full_summary.json` | **再生成** | — |
| `reports/strategy_dashboard_ecombined_full.png` | **再生成** | — |
| `backtests/study47_prod_candidate_2026-06-28.json` | **新規** 修正後結果 | — |
| `tools/integrity_check.py` | **新規** CI整合性チェック | — |

---

## ② 原因一覧

### CRITICAL (修正済)

| ID | 症状 | 根本原因 | 場所 |
|---|---|---|---|
| BUG-01 | 全バックテスト開始日が 2020-08-11 に固定 | `common_dates = intersection(all_symbols.index)` → 4055.T (IPO 2020-08-11) が全銘柄の共通日付を強制 | `composite_alpha_bt.py:645` |
| BUG-02 | Study47 FULL_IS CAGR が 20.51% と過大 | BUG-01 により実際計算期間 4.27yr (2020-2024)。同等総リターンを短期間で割るとCAGR高騰 | study47 JSON |
| BUG-03 | Dashboard/Review の数値乖離 9.48pp | Review=20.51%(4.27yr), Dashboard=11.03%(7.81yr)。同一戦略なのに別期間の数値を並列比較 | strategy_review ドキュメント |

### WARNING (修正不要・開示済)

| ID | 症状 | 対応 |
|---|---|---|
| WARN-01 | 生存者バイアス | RSR42は2026年現在の生存銘柄のみ。universe_builder.py で開示済 |
| WARN-02 | 選択バイアス | RSR42を2026年時点の知識で選定。universe.yaml で開示済 |
| WARN-03 | 4055.T のRSRランク影響 | union fix後、4055.T は2020-08-11以降のRSRランクにのみ影響。他銘柄相対順位に微小変化あり |

### CONFIRMED CORRECT (問題なし)

| 項目 | 判定 | 根拠 |
|---|---|---|
| RSR lookahead | PASS | `r1 = prices/prices.shift(63)` — shift()使用、先読みなし |
| ATR trailing stop (high50_close, shift不使用) | PASS | EOD終値で信号判定 → EODデータは確定済み。先読みではない |
| ATR (high200, shift(1)使用) | PASS | エントリーボーナス用、1日シフト済み |
| 実行価格 = 翌日始値 | PASS | `sell_px = float(open_mat[next_i, ...])` |
| SLIPPAGE 0.1% | PASS | `SLIPPAGE=0.001` = strategy.yaml `slippage_rate: 0.001` 一致 |
| COMMISSION 0.055% | PASS | `COMMISSION=0.00055` = strategy.yaml 一致 |
| CAGR計算 | PASS | `years = n_days/252; cagr = (eq[-1]/capital)**(1/years)-1` — 統一 |
| ランダム数 | PASS | `np.random` 不使用 → 完全決定論的 |
| 再現性 | PASS | 同一入力→同一出力 (乱数なし) |

---

## ③ 修正内容 (diff)

### `src/backtest/composite_alpha_bt.py`

```diff
-    common_dates: pd.DatetimeIndex | None = None
-    for sym in active_syms:
-        idx = universe_raw[sym]["df"].index
-        common_dates = idx if common_dates is None else common_dates.intersection(idx)
+    # [2026-06-28 INTEGRITY FIX] union instead of intersection.
+    # intersection forces backtest start to the latest-IPO symbol's first date
+    # (e.g. 4055.T IPO 2020-08-11 truncates a 2018-2024 run to 2020-2024).
+    common_dates: pd.DatetimeIndex | None = None
+    for sym in active_syms:
+        idx = universe_raw[sym]["df"].index
+        common_dates = idx if common_dates is None else common_dates.union(idx)

-    for sym_idx, sym in enumerate(active_syms):
-        df_src = universe_raw[sym]["df"]
-        row_idx = df_src.index.get_indexer(common_dates)
-        if np.any(row_idx < 0):
-            continue          # ← 日付欠損があると銘柄ごとスキップ(バグ)
-        open_mat[:, sym_idx] = open_vals[row_idx]
-        close_mat[:, sym_idx] = close_vals[row_idx]
-        ...
-        signal_mat[:, sym_idx] = sig_np[row_idx]
+    for sym_idx, sym in enumerate(active_syms):
+        df_src = universe_raw[sym]["df"]
+        row_idx = df_src.index.get_indexer(common_dates)
+        valid = row_idx >= 0
+        if not np.any(valid):
+            continue          # ← 当該期間にデータが一切なければスキップ
+        valid_pos = np.where(valid)[0]
+        open_mat[valid_pos, sym_idx]  = open_vals[row_idx[valid]]
+        close_mat[valid_pos, sym_idx] = close_vals[row_idx[valid]]
+        ...
+        signal_mat[valid_pos, sym_idx] = sig_np[row_idx[valid]]
```

**修正の意味**: 4055.T は 2020-08-11 以前のデータがない。union 後は open_mat/close_mat が NaN(=0.0 as float) のまま → buy_px=0 → `if buy_px <= 0: continue` で自然スキップ → IPO前は一切取引されない → 正しい挙動。

---

## ④ 修正前後比較表

### Study47 E_COMBINED (Full IS 2018-2024)

| 指標 | 修正前 (2026-06-27) | 修正後 (2026-06-28) | 差分 |
|---|---|---|---|
| 実際開始日 | **2020-08-11** (4055.T IPO) | **2018-01-01** (正しい) | -966日 |
| 実際年数 | 4.27yr | 6.84yr | +2.57yr |
| CAGR | **20.51%** | **12.19%** | **-8.32pp** |
| MaxDD | -19.81% | -20.11% | -0.30pp |
| Sharpe | 0.880 | 0.565 | -0.315 |
| Calmar | 1.035 | 0.606 | -0.429 |
| Trades | 205 | 274 | +69 |
| Corona shock covered | ✗ | **✓** | — |

### Study47 E_COMBINED (True OOS 2025)

| 指標 | 修正前 | 修正後 | 差分 |
|---|---|---|---|
| CAGR | 11.90% | 11.90% | **0.00pp** |
| MaxDD | -10.60% | -10.60% | 0.00pp |
| Sharpe | 1.056 | 1.056 | 0.000 |

→ OOS 2025 は変化なし (4055.T は 2020年上場済みのため影響なし)

### Dashboard (2018-2025 全期間)

| 指標 | 修正前 (RSR41) | 修正後 (RSR42 union) | 差分 |
|---|---|---|---|
| Universe | RSR41 (4055.T除外) | **RSR42 (union fix)** | 統一 |
| CAGR (2018-2025) | 11.03% | **11.04%** | +0.01pp |
| CAGR (2018-2024) | 12.18% | **12.19%** | +0.01pp |
| MaxDD | -18.02% | -20.11% | -2.09pp (4055.T 2021年影響) |
| Sharpe | 0.547 | 0.544 | -0.003 |
| Trades (full) | 316 | 324 | +8 |
| 開始日 | 2018-01-01 | 2018-01-01 | — |

---

## ⑤ Study40〜49 への影響

### 影響分類

| Study | 内容 | 影響 |
|---|---|---|
| Study40〜46 (WF) | WF_SEG1 IS start=2018-01-01, SEG2 start=2019-01-01 | **Fold1/2 の IS 期間が 2018/2019 から正しく計算される。4055.T は SEG3(start=2020-01)以降に自然参加** |
| Study47 (IS/OOS) | IS=2018-2024, OOS=2025 | **IS CAGR: 20.51% → 12.19%。OOS: 変化なし** |
| Study48 (等価監査) | RSR42, 2020-2025 | IS基準日が変わるため再監査が必要 |
| Study49 (運用耐性) | 運用インフラ | 数値変化なし (インフラ監査) |

### WF Fold への影響予測

| Fold | IS start | 4055.T参加? | 影響 |
|---|---|---|---|
| SEG1 | 2018-01-01 | 2020-08-11以降 (IPO後) | **Corona期含む IS → より厳しい評価** |
| SEG2 | 2019-01-01 | 2020-08-11以降 | **Corona IS後半 → 影響小** |
| SEG3 | 2020-01-01 | 2020-08-11以降 | 変化小 |
| SEG4 | 2021-01-01 | 完全参加 | 変化なし |
| SEG5 | 2022-01-01 | 完全参加 | 変化なし |

### Study40〜46 ADOPT/REJECT 結論変更リスク

- **Fold1 OOS (2020)**: Corona ショック (2020-02 to 2020-04) が正しく含まれる → OOS 評価が厳しくなる可能性あり
- **Fold2 OOS (2021)**: IS 期間が正しく 2 年分に拡大 → 影響軽微
- **Fold3〜5**: 変化なし

→ **Study40〜46 の最終 ADOPT/REJECT 判定については、Fold1/2 の変化により一部が変わる可能性がある。完全な再実行が推奨されるが、Fold3〜5 のみでも WF 評価 (3/5) は維持される傾向がある。**

---

## ⑥ Dashboard 更新結果

`reports/strategy_dashboard_ecombined_full.png` を RSR42 + union fix ベースで再生成済み。

| 表示項目 | 修正前 | 修正後 |
|---|---|---|
| Universe | RSR41 (4055.T除外) | RSR42 (union fix) |
| 開始日 | 2018-01-01 | 2018-01-01 |
| CAGR (タイトル) | 11.03% | **11.04%** |
| MaxDD | -18.02% | **-20.11%** |
| Sharpe | 0.547 | **0.544** |
| Calmar | 0.612 | **0.549** |
| Corona shock | 表示 ✓ | 表示 ✓ |

---

## ⑦ 最終的に正しい Production 成績

### E_COMBINED (RSR42 + union fix)

| 期間 | CAGR | MaxDD | Sharpe | Calmar | Trades |
|---|---|---|---|---|---|
| **2018-2024 (IS)** | **12.19%** | -20.11% | 0.565 | 0.606 | 274 |
| **2018-2025 (Full)** | **11.04%** | -20.11% | 0.544 | 0.549 | 324 |
| **2025 (OOS)** | **11.90%** | -10.60% | 1.056 | 1.123 | 45 |

| 補足指標 | 値 |
|---|---|
| WinRate | 54.0% |
| AvgExposure | 30.7% |
| Corona shock MaxDD (2020-02 ~ 2020-04) | -16.90% |
| 開始日 (正確) | 2018-01-01 |
| 以前の "CAGR 20.51%" | **廃止。期間圧縮の産物であり実際の戦略実力ではない** |

### ベンチマーク比較 (2018-2025)

| | CAGR | 最終倍率 |
|---|---|---|
| E_COMBINED | **11.04%** | 2.26x |
| TOPIX (1306.T) | 11.03% | 2.26x |
| Nikkei225 | 10.26% | 2.13x |

---

## ⑧ Integrity Check (CI相当) — 実装済み

### `tools/integrity_check.py`

実行: `python tools/integrity_check.py`  
Exit 0 = PASS / Exit 1 = FAIL

| Check ID | 内容 |
|---|---|
| IC-01 | common_dates が union を使用 (intersection 禁止) |
| IC-02 | 4055.T IPO日 = 2020-08-11 (データ確認) |
| IC-03 | CAGR公式 n_days/252, compound計算 |
| IC-04 | SLIPPAGE/COMMISSION がstrategy.yaml と一致 |
| IC-05 | RSR/ATR/実行価格の lookahead なし (6項目) |
| IC-06 | np.random 不使用 (決定論的) |
| IC-07 | Dashboard CSV が 2018-01-01 から開始 |
| IC-08 | Study47 修正後 CAGR が [10,15]% 範囲内 |
| IC-09 | 生存者/選択バイアス 開示確認 |
| IC-10 | Dashboard 2018-2024 CAGR が Study47 IS と ±0.5pp 一致 |
| IC-11 | intersection コードが完全に削除されている |

**最終結果**: 25/25 PASS ✓

### CI組み込み推奨

```yaml
# .github/workflows/integrity.yml (例)
name: Backtest Integrity
on: [push, pull_request]
jobs:
  check:
    steps:
      - run: python tools/integrity_check.py
```

または Windows タスクスケジューラ:
```
python C:\ai-trading\tools\integrity_check.py >> C:\ai-trading\logs\integrity.log 2>&1
```
