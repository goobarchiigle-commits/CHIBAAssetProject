# Conditional RSR Policy WF (最終品質監査)

作成日: 2026-06-14  |  研究専用 / 実装変更禁止

採用条件: WF≥4/5, ΔCAGR≥+0.5pp, ΔDD≤+1.0pp, swap≤5.0%

バックテスト: run_period (pattern A) 完全再現。max_hold=60, sym_active_mat, 動的gross_cap, 1-day mkt_shock 適用。

**採用ポリシー**: 0件  |  評価: 3件 (P1/P2/P3)

---
## 1. RSR帯別シグナル分布 (Phase 1)

| 帯 | RSR範囲 | OOS全シグナル件数 | Policy変更 |
|---|---|---|---|
| G1 | 65–80 | 167 | P1/P2: state_score; P3: slope5 |
| G2 | 80–90 | 318 | 全Policy: RSR (変更なし) |
| G3 | 90–95 | 132 | P2/P3: −days_cross90 (鮮度優先) |
| G4 | 95–101 | 165 | 全Policy: RSR (変更なし) |

---
## 2. ベースライン P0 (production-faithful)

| Fold | OOS年 | Regime | CAGR | MaxDD | Calmar | n_trades |
|---|---|---|---|---|---|---|
| Fold1 | 2021 | Bull (+13.3%) | +32.1% | 20.1% | 1.60 | 44 |
| Fold2 | 2022 | Bear (-4.7%) | -6.3% | 13.8% | -0.46 | 9 |
| Fold3 | 2023 | Bull (+29.8%) | +2.1% | 2.9% | 0.72 | 7 |
| Fold4 | 2024 | Bull (+19.7%) | -6.0% | 10.4% | -0.58 | 3 |
| Fold5 | 2025 | Bull (+26.8%) | +3.6% | 1.9% | 1.83 | 2 |
| **avg** | — | — | **+5.1%** | **9.8%** | **0.62** | — |

---
## 3. Phase 2: WF Replay サマリ

| Policy | 説明 | WF | ΔCAGR | ΔDD | ΔCalmar | avg_swap | 採用 |
|---|---|---|---|---|---|---|---|
| P1 | G1: state_score; others: RSR | 3/5 | +2.85pp | -0.80pp | +0.471 | 28.9% | ❌ |
| P2 | G1: state_score-0.3z(d70); G3: −d90; others: RSR | 3/5 | +2.82pp | -0.28pp | +0.396 | 28.9% | ❌ |
| P3 | G1: slope5; G3: −d90; others: RSR | 3/5 | +0.08pp | +0.50pp | -0.070 | 32.2% | ❌ |

---
## 5. Swap Event 手動監査


### P1: G1: state_score; others: RSR

- OOS swap件数: **5件**  勝ち: 2  負け: 3
- avg Δfwd60: **+7.51pp**  (policy有利)

| Date | Band | RSR pick | RSR fwd60 | Policy pick | Pol fwd60 | Δfwd60 | 判定 |
|---|---|---|---|---|---|---|---|
| 2021-04-21 | G1 | 8306.T (58.54) | -0.5% | 7182.T | -4.8% | -4.3% | ❌ policy負 |
| 2023-04-14 | G1 | 8053.T (78.57) | +18.4% | 8058.T | +33.7% | +15.3% | ✅ policy勝 |
| 2023-07-14 | G1 | 8725.T (47.62) | +16.1% | 7201.T | +15.8% | -0.3% | ❌ policy負 |
| 2023-12-14 | G1 | 5706.T (76.19) | +4.4% | 7203.T | +35.6% | +31.3% | ✅ policy勝 |
| 2025-09-18 | G1 | 9531.T (73.81) | +16.6% | 7182.T | +12.2% | -4.4% | ❌ policy負 |

### P2: G1: state_score-0.3z(d70); G3: −d90; others: RSR

- OOS swap件数: **5件**  勝ち: 2  負け: 3
- avg Δfwd60: **+7.51pp**  (policy有利)

| Date | Band | RSR pick | RSR fwd60 | Policy pick | Pol fwd60 | Δfwd60 | 判定 |
|---|---|---|---|---|---|---|---|
| 2021-04-21 | G1 | 8306.T (58.54) | -0.5% | 7182.T | -4.8% | -4.3% | ❌ policy負 |
| 2023-04-14 | G1 | 8053.T (78.57) | +18.4% | 8058.T | +33.7% | +15.3% | ✅ policy勝 |
| 2023-07-14 | G1 | 8725.T (47.62) | +16.1% | 7201.T | +15.8% | -0.3% | ❌ policy負 |
| 2023-12-14 | G1 | 5706.T (76.19) | +4.4% | 7203.T | +35.6% | +31.3% | ✅ policy勝 |
| 2025-09-18 | G1 | 9531.T (73.81) | +16.6% | 7182.T | +12.2% | -4.4% | ❌ policy負 |

### P3: G1: slope5; G3: −d90; others: RSR

- OOS swap件数: **5件**  勝ち: 2  負け: 3
- avg Δfwd60: **+3.73pp**  (policy有利)

| Date | Band | RSR pick | RSR fwd60 | Policy pick | Pol fwd60 | Δfwd60 | 判定 |
|---|---|---|---|---|---|---|---|
| 2021-04-21 | G1 | 8306.T (58.54) | -0.5% | 8309.T | -4.9% | -4.4% | ❌ policy負 |
| 2022-04-06 | G1 | 5401.T (59.52) | -5.0% | 4021.T | -5.4% | -0.4% | ❌ policy負 |
| 2023-04-14 | G1 | 8053.T (78.57) | +18.4% | 8058.T | +33.7% | +15.3% | ✅ policy勝 |
| 2023-07-14 | G1 | 8725.T (47.62) | +16.1% | 7201.T | +15.8% | -0.3% | ❌ policy負 |
| 2023-12-14 | G1 | 5706.T (76.19) | +4.4% | 5401.T | +12.7% | +8.4% | ✅ policy勝 |

---
## 6. Fold × Policy CAGR 比較

| Fold | OOS年 | Regime | CAGR P0 | CAGR P1 | CAGR P2 | CAGR P3 |
|---|---|---| --- | --- | --- | --- |
| Fold1 | 2021 | Bull (+13.3%) | +32.1% | +32.7% | +32.7% | +32.6% | 
| Fold2 | 2022 | Bear (-4.7%) | -6.3% | -6.3% | -6.3% | -6.3% | 
| Fold3 | 2023 | Bull (+29.8%) | +2.1% | +2.1% | +1.9% | +1.9% | 
| Fold4 | 2024 | Bull (+19.7%) | -6.0% | -0.8% | -0.8% | -6.0% | 
| Fold5 | 2025 | Bull (+26.8%) | +3.6% | +11.9% | +11.9% | +3.6% | 

---
## 7. Regime別 ΔCAGR 寄与

| Policy | Bull avg ΔCAGR | Bear avg ΔCAGR | 寄与差 (Bull-Bear) |
|---|---|---|---|
| P1 | +3.56pp | +0.03pp | +3.53pp |
| P2 | +3.51pp | +0.03pp | +3.48pp |
| P3 | +0.09pp | +0.03pp | +0.06pp |

---
## 8. 仮説検証

| 仮説 | 検証方法 | 結果 |
|---|---|---|
| 帯内ポリシーで ΔCAGR≥+0.5pp | WF ≥4/5 | ❌ REJECT (best=+2.85pp WF=3/5) |
| G3: −d90 鮮度優先の効果 | ΔCAGR vs baseline | best=+2.82pp (P2) |
| swap ≤ 5% (低侵襲) | OOS avg_swap | ⚠ max=32.2% |

---
## 9. 結論

**採用なし。現行 RSR 順位が最適。**

- 最良: P1 ΔCAGR=+2.85pp (WF=3/5) — 採用基準未満
- 帯内 policy 変更: OOS で有意な改善なし
- swap が avg 28.9–32.2% → ≤5% 基準を大幅超過 (G1内順位変動が大きい)

**根本的制約:**
- G1(75-79): OOS 年平均 ~33件 (=167/5yr)、そのうち複数候補の競合は少数 → policy 効果が誤差内
- G3(90-94): OOS N≤26件/fold → d90 効果不安定
- Study 5/6 と同様: IS 有意な特徴量 (state_score ρ=+0.07) も OOS portfolio 全体 CAGR では希薄化

**補足: production-faithful baseline の重要な変化**
- P0 avg CAGR = +5.1% (Study 6 simplified baseline +14.9% からの大幅乖離)
- 主因: max_hold=60 (TIME_STOP) が 2023-2024 の bull 上昇期に強制利食いを発動
- Fold3 2023 (TOPIX +29.8%): P0 CAGR +2.1% (7trades), Fold4 2024 (TOPIX +19.7%): P0 CAGR -6.0% (3trades)
- 動的 sym_active_mat フィルタにより 2024 年の新規エントリーが大幅制限されている模様

次ステップ候補:
- G: 2025 低調年 (Fold5) 原因分析
- H: risk_pct 感度 WF 再評価 (既存 CONDITIONAL)
- I: Exit RSR 70 本番適用 WF (Regime-Aware B grade: WF4/5)