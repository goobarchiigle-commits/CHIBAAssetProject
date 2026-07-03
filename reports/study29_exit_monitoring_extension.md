# Study29 Exit Monitoring Extension (Research Freeze Phase)

作成日: 2026-06-24  |  観測専用 / Entry・Signal・Universe・Sizing・Allocation・Capital・Execution・Authority・Exit Logic 変更禁止 / Production変更なし

**背景**: Study24(Entry)/Study25(Sizing)/Study27(Risk Activation)/Study28(Allocation) 全てEXHAUSTED確定（Study28 oracle ceiling=+0.069<+0.10）。新規最適化研究を凍結し、現行Exit品質の継続監査(サンプル拡大)へ移行。

**基盤**: Study28 Case A 同一設定（本番PARAMS_LOCKED: Capital=¥3,000,000, max_positions=3, 固定1/MAX_POS, 再配分なし） / Entry・Exit信号は Study9 Case B / Study20-28系列と完全同一（RSR[92,95) d90<=5 slope5<=5 / exit RSR<90, min_hold=3）

**観測ウィンドウ**: 2018-01-01 → 2025-12-31（全完了トレード, counterfactual attributionはStudy21 attribute_trade()を無変更で再利用）

---
## 1. tail_capture

| 集計対象 | tail_capture |
|---|---|
| 全利益トレード (n=40) | 68.2% |
| 直近20件 | 61.3% |
| 直近40件 | 68.2% |
| 全期間 | 68.2% |

注: 本監査は単発フルヒストリー実行（ライブの段階的蓄積ではない）のため、「全利益トレード」と「全期間」は定義上一致する（同じ母集団）。直近20件/40件はtail_capture劣化トレンドの有無を確認するための補助指標。

---
## 2-4. profit_left / loss_avoided / exit_efficiency

| 指標 | 値 | 定義 |
|---|---|---|
| profit_left (avg, 全88トレード) | 5.1% | Exit後20営業日以内に残っていた上昇余地（counterfactual_max_gain） |
| loss_avoided (avg) | 103.6% | Exitによって回避できた追加損失率（future_loss_after_exit/max_future_loss） |
| exit_efficiency (avg) | -49.2% | 実現利益 / 保有期間中最大含み益(in_trade_mfe)。in_trade_mfe<=0のトレードは対象外 |
| exit_efficiency (median) | 50.8% | 平均値はin_trade_mfeが極小のトレードで比率が発散しやすいため、median を頑健性参考値として併記 |
| exit_efficiency_stability_std (rolling20間の標準偏差) | 1.1324 | 操作的安定性閾値<= 0.2 |

注: exit_efficiency=実現利益/in_trade_mfeは分母(in_trade_mfe)が極小(例: 0.1%)のトレードで比率が±数倍に発散しうる不安定な指標。本研究ではタスク仕様の定義をそのまま適用し平均値を主指標として採用するが、median（頑健性参考値）も併記し、解釈時はrolling_20の時系列トレンド（下記7節寄り）を優先して判断する。

---
## 5. holding_period_distribution

n=88  mean=21.27d  median=8.0d  std=41.23d  min=2d  max=336d

| bucket | count |
|---|---|
| <=5d | 39 |
| 6-10d | 12 |
| 11-20d | 14 |
| 21-40d | 11 |
| 41-60d | 5 |
| >60d | 7 |

---
## 6. rolling_20_trade_metrics

window数=69（trade_id基準, 20トレード移動窓） / 先頭window=65.5%tail_capture → 末尾window=49.5%tail_capture

詳細は `reports/study29_rolling_20_trade_metrics.csv` 参照（全window収録）。

| window_end | tail_capture | profit_left | exit_efficiency |
|---|---|---|---|
| #20 (2021-08-18) | 65.5% | 4.1% | 79.4% |
| #24 (2022-05-06) | 70.5% | 4.1% | 98.2% |
| #28 (2022-10-03) | 70.3% | 3.7% | 104.6% |
| #32 (2022-11-15) | 70.3% | 3.2% | 95.4% |
| #36 (2023-01-23) | 71.7% | 2.3% | 102.5% |
| #40 (2023-02-17) | 90.2% | 0.2% | 138.2% |
| #44 (2023-03-30) | 100.0% | 0.5% | 142.7% |
| #48 (2023-06-09) | 86.1% | 1.2% | 50.5% |
| #52 (2023-10-27) | 86.0% | 2.0% | 76.8% |
| #56 (2024-01-17) | 87.4% | 2.8% | -24.9% |
| #60 (2024-05-08) | 75.4% | 3.9% | 44.1% |
| #64 (2024-06-14) | 67.1% | 5.5% | 24.4% |
| #68 (2024-08-02) | 66.1% | 6.3% | 14.8% |
| #72 (2024-10-02) | 60.5% | 9.9% | -156.3% |
| #76 (2024-12-02) | 57.7% | 11.2% | -134.9% |
| #80 (2025-03-21) | 54.8% | 11.1% | -156.1% |
| #84 (2025-07-08) | 51.0% | 11.1% | -193.0% |
| #88 (2025-12-10) | 49.5% | 11.0% | -156.1% |

**トレンド検査（前半1/4窓 vs 後半1/4窓平均）**: tail_capture 69.8% → 54.7% (Δ-15.1%)  /  exit_efficiency 95.0% → -158.7% (Δ-253.8%)

⚠ tail_captureが後半窓で5pp以上悪化 — Exit品質の経時劣化の疑いあり（KEEP_EXIT判定には不利な追加根拠）。

---
## 7. exit_reason_distribution

注: 本戦略には独立した「Stop Exit」カテゴリは存在しない。MKT_SHOCK（市場急落時の強制ディフェンシブExit, CIRCUIT max_dd_limitとは別の構造的Exit）をタスク仕様の Stop Exit 相当として分類。

| 区分 | count | pct |
|---|---|---|
| RSR Exit | 84 | 95.5% |
| Stop Exit | 4 | 4.5% |
| Other Exit | 0 | 0.0% |

---
## 判定基準

| 判定 | 条件 |
|---|---|
| KEEP_EXIT | tail_capture>=75% AND profit_left<=10% AND exit_efficiency安定(std<=0.2) |
| RESEARCH_EXIT | tail_capture<70% OR profit_left>15% |
| REPLACE_EXIT | tail_capture<65% AND profit_left>20% |

停止条件: 利益確定済みトレード数>=60 OR 判定境界から5%以上の余裕(confidence=HIGH)

---
## 最終出力

| 指標 | 値 |
|---|---|
| trade_count (全完了トレード) | 88 |
| n_winning_trades (利益確定済み, tail_capture定義可能) | 40 |
| tail_capture (全期間) | 68.2% |
| profit_left | 5.1% |
| loss_avoided | 103.6% |
| exit_efficiency | -49.2% |
| **production_decision** | **RESEARCH_EXIT** |
| decision_reason | tail_capture=68.2%<70% |
| confidence | LOW (margin=1.8%) |
| stop_condition_fired | False (NOT MET: n_winning_trades=40<60 AND confidence=LOW(margin=1.8%)) |

**recommend_next_step**: RESEARCH_EXIT。Exit品質に改善余地の疑いあり。Exit再設計の研究フェーズへの移行を検討（ただしREPLACE_EXIT条件未達のため段階的検証=WF必須、即時変更は禁止）。
