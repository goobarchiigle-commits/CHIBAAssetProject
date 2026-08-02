# Exit Alpha Audit — Pattern A

作成日: 2026-06-04  |  対象: 2018-01-01〜2025-12-31  |  全トレード

**IS: 2018-01-01〜2024-12-31  OOS: 2025-01-01〜2025-12-31**

---
## 0. ベースライン確認

| 期間 | CAGR | Sharpe | MaxDD | Calmar | 取引数 | 平均保有 |
|---|---|---|---|---|---|---|
| IS 2018-2024 | +18.1% | 0.779 | -16.7% | 1.087 | 207 | 11.8d |
| OOS 2025 | +10.0% | 1.007 | -10.1% | 0.988 | 45 | 7.1d |

---
## 1. Exit Reason別統計

| Exit理由 | N | WR | avg_return | PF | avg_hold | fwd20d_mean | fwd20d_WR |
|---|---|---|---|---|---|---|---|
| **Turtle/Momentum Exit** | 118 | 62% | +4.52% | 3.63 | 17.7d | +2.3% | 56% |
| **RSR Exit** | 128 | 53% | -0.28% | 0.84 | 5.2d | +3.6% | 70% |
| **Composite Shock Exit** | 3 | 0% | -10.63% | 0.00 | 15.3d | +28.3% | 100% |
| **Time Stop (max_hold)** | 4 | 100% | +32.76% | 905110455322.27 | 62.0d | -4.2% | 50% |

**合計**: 253件

---
## 2. Early Exit Audit (exit後フォワードリターン)


### 2a. 全Exit後 forward returns

| Forward期間 | N | 平均 | 中央値 | 勝率 | 解釈 |
|---|---|---|---|---|---|
| fwd20d | 249 | +3.12% | +2.80% | 63% | ⚠ 退出後も上昇継続 |
| fwd40d | 242 | +7.61% | +4.89% | 68% | ⚠ 退出後も上昇継続 |
| fwd60d | 235 | +10.46% | +6.99% | 74% | ⚠ 退出後も上昇継続 |

### 2b. Exit Reason別 forward returns

| Exit理由 | N | fwd20d_mean | fwd40d_mean | fwd60d_mean | fwd20d_WR |
|---|---|---|---|---|---|
| Turtle/Momentum Exit | 118 | +2.3% | +7.1% | +8.7% | 56% |
| RSR Exit | 128 | +3.6% | +7.7% | +11.7% | 70% |
| Composite Shock Exit | 3 | +28.3% | +39.2% | +37.6% | 100% |
| Time Stop (max_hold) | 4 | -4.2% | -5.2% | +2.7% | 50% |

---
## 3. Counterfactual Hold (追加保有時の期待損益)

| 延長期間 | N | 追加損益_合計 | 追加損益_平均 | 正率 | 解釈 |
|---|---|---|---|---|---|
| +20日 | 249 | +4,752,424円 | +19,086円 | 63% | ⚠ 保有延長で追加リターン期待 |
| +40日 | 242 | +11,616,463円 | +48,002円 | 68% | ⚠ 保有延長で追加リターン期待 |
| +60日 | 235 | +15,572,738円 | +66,267円 | 74% | ⚠ 保有延長で追加リターン期待 |

---
## 4. 保有期間別損益分析

| 保有期間 | N | WR | avg_return | PF | avg_fwd20d | avg_fwd60d |
|---|---|---|---|---|---|---|
| 0-20日 | 204 | 52% | -0.18% | 0.90 | +3.0% | +10.5% |
| 21-40日 | 36 | 69% | +8.44% | 5.45 | +5.6% | +14.7% |
| 41-60日 | 9 | 100% | +22.15% | 1179216711425.78 | -1.5% | -2.6% |
| 61日+ | 4 | 100% | +32.76% | 905110455322.27 | -4.2% | +2.7% |

**保有日数統計**: mean=12.0d  median=6d  max=62d  p75=15d  p90=28d

---
## 5a. Turtle Trailing Stop 実効性診断

> **注意**: turtle_exit=55d/65d/75d が全て同一結果 → trailing stop 独立発動ゼロを示唆

**実効性診断**:

FujikoStrategy の exit条件:
```
exit = (RSR_momentum < 0 AND RSR_momentum_declining)   ← 一次出口
     OR (close < N日最安値)                            ← trailing stop
```
RSR momentumがN日最安値と同時またはより早く発動するため、
turtle_exit N=55/65/75 のいずれでも exit タイミングが不変。

**EXIT 内訳 (実態)**:
| 条件 | 発動件数 | 備考 |
|---|---|---|
| RSR_EXIT (rsr_val < min_rsr=75) | 128 | abc.pyで先に判定 |
| STRATEGY_EXIT (momentum+trailing) | 118 | 主にmomentum起因 |
| TIME_STOP (max_hold=60d) | 4 | ★**60日保有強制終了** |
| MARKET_SHOCK_EXIT (composite) | 3 | market-5%+sym-8% |

**結論**: turtle_exit パラメータ (55/65/75d) は実質的に EXIT に影響しない。
主要 EXIT ドライバーは RSR レベル低下と RSR モメンタム反転。


---
## 5b. Turtle Exit感度テスト (IS+OOS比較)

| turtle_exit | IS CAGR | IS Sharpe | IS MaxDD | IS Calmar | OOS CAGR | OOS Sharpe | OOS MaxDD | OOS Calmar |
|---|---|---|---|---|---|---|---|---|
| *(turtle trailing stopは実効性ゼロ — 全て同一結果)* | | | | | | | | |
| **55d ★** | +18.1% | 0.779 | -16.7% | 1.087 | +10.0% | 1.007 | -10.1% | 0.988 |
| **65d** | +18.1% | 0.779 | -16.7% | 1.087 | +10.0% | 1.007 | -10.1% | 0.988 |
| **75d** | +18.1% | 0.779 | -16.7% | 1.087 | +10.0% | 1.007 | -10.1% | 0.988 |

### Δ vs 55d (現行)

| turtle_exit | ΔCAGR_IS | ΔSharpe_IS | ΔCalmar_IS | ΔCAGR_OOS | ΔSharpe_OOS | ΔCalmar_OOS |
|---|---|---|---|---|---|---|
| 65d | +0.00pp | +0.0000 | +0.0000 | +0.00pp | +0.0000 | +0.0000 |
| 75d | +0.00pp | +0.0000 | +0.0000 | +0.00pp | +0.0000 | +0.0000 |

---
## 6. 総合分析

### アルファ漏出源: Entry vs Exit

| 指標 | 値 |
|---|---|
| 全Exit後 fwd20d 平均 | +3.12% (N=249) |
| 全Exit後 fwd60d 平均 | +10.46% |
| Counterfactual +20d 正率 | 63% |
| Counterfactual +20d 平均追加損益 | +19,086円 |
| STRATEGY_EXIT後 fwd20d | +2.25% (N=117) |
| RSR_EXIT後 fwd20d | +3.57% (N=125) |
| MARKET_SHOCK_EXIT後 fwd20d | +28.28% (N=3) |
| TIME_STOP後 fwd20d | -4.20% (N=4) |

### Entry vs Exit アルファ漏出 比較

| 項目 | 値 | 解釈 |
|---|---|---|
| Missed Signal fwd60d (既知) | +8.2% | Entry改善の上限期待値 |
| Executed Signal fwd60d (既知) | +5.6% | 実際の保有銘柄60日平均 |
| Exit後 fwd20d 平均 | +3.12% | Exit改善の期待値上限 |

---
## 7. 最終判定

### **B: Exit改善余地あり**

Exit改善余地あり: fwd60d 全体=+10.5%, fwd20d 勝率=63%, CF+60d 平均追加損益=+66,267円. RSR_EXIT(128件): avg_return=-0.28%, fwd60d=+11.7% | MARKET_SHOCK後 fwd20d=+28.3%(N=3:統計弱)

### 現システム最大のアルファ漏出源

**一次漏出源**: **Exit**
**二次漏出源**: Entry (Missed Signal +8.2% vs executed +5.6%の差)

| 漏出源 | 指標 | 大きさ | 優先度 |
|---|---|---|---|
| Entry (Missed Signal) | fwd60d差: +8.2% - +5.6% = +2.6% | 機会損失 | — |
| Exit (早期退出) | fwd20d平均: +3.12% | 直接損失 | 高 |

**推奨アクション**:

**【最優先】RSR_EXIT 閾値の見直し**
- RSR_EXIT: 128件, avg_return=-0.28%, fwd60d=+11.7%
- min_rsr=75 → 70 に緩和してWF検証 (RSR75境界での早期退出を防ぐ)
- または: RSR exit に保有日数条件追加 (hold >= 5d かつ RSR < 70 で退出)

**【次優先】MARKET_SHOCK_EXIT の確認**
- MARKET_SHOCK_EXIT後 fwd20d=+28.3%(N=3:統計弱)
- shock_sym_thr を -8% → -12% に緩和してWF検証

**【参考】turtle_exit parameter**
- 55d/65d/75d が全て同一結果 → trailing stop は実質デッドパラメータ
- RSR momentum exit が trailing stop より常に先行発動
- turtle_exit の変更は効果なし (PARAMS_LOCKEDから除外検討)