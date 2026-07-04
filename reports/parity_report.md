# Parity Report — Study80A エンジン変更の無害性検証

**日付**: 2026-07-04
**対象変更**: `composite_alpha_bt.py` への観測基盤追加（day-level context計算 + 既存候補ログ4種へのキー追加 + 新規`_selected_cands`リスト追加）

## 検証方法

M1適用後Production（CURRENT/D_ATR_EQ, FULL 2018-01-01〜2025-12-30, capital=¥3,000,000）を、Study80A変更適用後のエンジンでfresh run。既知の基準値（`backtests/study78_ror_mc_sensitivity_2026-07-04.json`・`backtests/study_m1_production_update_2026-07-04.json`で確定済み）と完全一致するかを確認。

## 結果

| 指標 | 変更前基準値 | 変更後実測値 | 判定 |
|---|---|---|---|
| CAGR | 11.22% | 11.22% | ✅ 完全一致 |
| n_trades | 309 | 309 | ✅ 完全一致 |
| Sharpe | 0.564 | 0.564 | ✅ 完全一致 |
| MaxDD | -18.22% | -18.22% | ✅ 完全一致 |
| Calmar | 0.616 | 0.616 | ✅ 完全一致 |

**判定: PASS（完全一致）**。売買ロジックへの影響はゼロであることを実測で確認した。

## 変更の性質（なぜParityが保たれるか）

1. **新規追加した`_obs_*`系変数**はすべて既存の状態（`positions`/`cash`/`close_mat`/`_is_bear`/`_eff_max_pos`等）を**読み取るだけ**で、代入・書き換えは一切行っていない。
2. **既存の候補ログ4種（`_missed_cands`/`_skip_detail`/`_rejected_by_lot_detail`/`_admitted_by_ratio_detail`）への変更**は、既存のdict literalに追加のkey-valueペアを挿入しただけであり、これらのリストへのappendタイミング・条件（if文の分岐）は一切変更していない。
3. **新規`_selected_cands`リスト**は、既存の`trades.append({"symbol":sym,"side":"BUY",...})`の直後に追加した、独立した観測専用のappendであり、既存の`trades`リスト・`cash`・`positions`の更新には一切関与しない。
4. **返り値dictへの追加**（`"_selected_cands": _selected_cands`）も同様に、既存キーの値を変更せず新規キーを追加しただけ。

以上より、本変更はコード上も実測上も**Parity（完全一致）が保証されている**。

## 検証物

- `src/backtest/study80a_observation_infrastructure_2026-07-04.py`（Part7として自動検証を実施）
- `backtests/trade_dataset_v2.json`他5ファイル（本parity確認済みrunの成果物）
