# Strategy Review — CHIBAAssetProject
**Date:** 2026-06-28  
**Phase:** 2 (Shadow Deployment → Production)  
**Research Program:** Study01–Study52 COMPLETE

---

## Executive Summary

**Production Configuration: D_ATR_EQ (確定)**

```
S5 Baseline
+ ATR Extension (Study40)
+ EQ_SCALE Addon (Study45)
VOL_ADJ = 完全除外 (Study52 REJECT 確定)
```

Study40〜52 の研究を通じて以下が確認された:

- **Integrity Fix (2026-06-28)**: `common_dates.intersection` → `.union` バグ修正。旧 "CAGR 20.51%" は廃止。
- **VOL_ADJ 除外**: OOS 悪化・IS MaxDD 悪化・Bear 保護ゼロを確認。除去により全指標改善。
- **ATR Extension**: Return Enhancer。OOS 2025 +2.23pp (Study52: B_ATR_EXT 13.87% − A_BASELINE 11.64%)、Seg4 Bull momentum 捕捉。
- **EQ_SCALE**: Robustness Enhancer。WF 5/5 達成、Seg3_2022 -5.11%→-2.65% (Bear 自動 de-risk)。

---

## Production Performance (D_ATR_EQ)

### IS 2018-2024 (Full In-Sample)

| 指標 | 値 |
|---|---|
| CAGR | **+12.37%** |
| MaxDD | **-18.12%** |
| Sharpe | **0.586** |
| Calmar | **0.683** |
| Trades | 263 |
| Avg Exposure | 31.3% |
| WF pass | — |

### OOS 2025 (True Out-of-Sample)

| 指標 | 値 |
|---|---|
| CAGR | **+13.48%** |
| MaxDD | **-10.15%** |
| Sharpe | **1.199** |
| Calmar | **1.329** |
| Trades | 42 |
| Avg Exposure | 36.2% |

### Full Period 2018-2025

| 指標 | 値 |
|---|---|
| CAGR | **+11.35%** |
| MaxDD | **-18.12%** |
| Sharpe | **0.570** |
| Trades | 309 |
| WinRate | 54.0% |
| Avg Exposure | 29.2% |

### Walk-Forward (D_ATR_EQ, 5-fold)

| 指標 | 値 |
|---|---|
| WF Pass | **5/5** |
| avg OOS CAGR | **+18.37%** |
| avg OOS MaxDD | -16.91% |
| avg OOS Sharpe | 0.876 |
| Fold std CAGR | 19.41 |
| **Seg3_2022** | **-2.65%** (PASS) |

#### WF Fold 詳細

| Fold | OOS 年 | CAGR | Sharpe | MaxDD | Pass |
|---|---|---|---|---|---|
| Seg1 | 2020 | +5.66% | 0.332 | -19.0% | ✓ |
| Seg2 | 2021 | +5.38% | 0.330 | -20.7% | ✓ |
| **Seg3** | **2022** | **-2.65%** | **0.057** | **-20.9%** | **✓** |
| Seg4 | 2023 | +44.90% | 1.817 | -8.8% | ✓ |
| Seg5 | 2024 | +38.58% | 1.846 | -15.1% | ✓ |

### Annual Returns (D_ATR_EQ)

| 年 | 年間リターン | MaxDD |
|---|---|---|
| 2018 | -3.61% | -12.68% |
| 2019 | +0.96% | -18.02% |
| 2020 | +8.76% | -16.90% |
| 2021 | +16.10% | -18.12% |
| **2022** | **+11.61%** | **-9.65%** |
| 2023 | +32.77% | -6.01% |
| 2024 | +15.68% | -8.61% |
| 2025 (OOS) | +3.98% | -3.72% |

2022 年 (弱気市場) に +11.61% はEQ_SCALE の Bear 自動 de-risk の実績を示す。

---

## Production Architecture

### 1. Universe

- **RSR42**: 42 銘柄 (union fix 適用、4055.T は 2020-08-11 以降参加)
- **Dynamic Universe**: dyn_rsr42_bear_rs0 (WF 5/5 確認済)
- **Bear Filter**: 7 セクター除外 (機械/鉄鋼/銀行業/保険業/輸送用機器/海運業/化学)
- **Auto-Promote**: PROBATION → GRADUATED → LIVE_UNIVERSE 閉ループ

### 2. Entry

- **Signal**: Turtle Breakout (20日高値ブレイク)
- **Filter**: RSR ランキング (min_rsr=75.0)
- **Min Hold**: 3日

### 3. Exit

- **Primary**: RSR Exit (threshold=70.0)
- **Secondary**: ATR Trailing Exit (50日 lookback、3×ATR バンド)
- **[Study40] ATR Extension**: RSR SELL 後、ATR モメンタム継続中なら最大 5 営業日 defer  
  `exit_policy="A"`, `atr_mult=1.0`, `defer_days=5`

### 4. Capital Allocation

- **Base**: 均等ウェイト (cash ÷ 残候補銘柄数)
- **[Study45] EQ_SCALE Addon**: 含み益 ≥ 1×ATR20 で cash×25% の追加 BUY  
  `addon_policy="D"`, `size_frac=0.25`, `atr_mult=1.0`, `max_per_pos=1`
- **[削除] VOL_ADJ**: Study52 REJECT 確定。Production から完全除外。

### 5. Risk Controls

- **Circuit Breaker**: max_dd_limit=-15% (warn only)
- **Position Limit**: max_positions=3 (固定。VOL_ADJ による動的変更なし)
- **Concentration**: max_single_weight=25%
- **Slippage**: 0.1% (必須)
- **Commission**: 0.055% (必須)

### 6. Implementation Files

| Module | File | Role | Status |
|---|---|---|---|
| ATR Extension | `src/research_candidate/atr_extension.py` | RSR SELL defer | **ENABLED** |
| EQ_SCALE Addon | `src/research_candidate/eq_scale_addon.py` | 含み益アドオン | **ENABLED** |
| VOL_ADJ | `src/research_candidate/vol_adj.py` | max_pos 動的調整 | **DISABLED (Study52 REJECT)** |
| Config | `src/configs/strategy.yaml` | research_candidates | `atr_extension: true, eq_scale_addon: true, vol_adj: false` |
| Live Entry | `src/run_live_signal.py` | 3 injection points | 更新要 |

---

## Study52 採用経緯

### 採用理由: ATR Extension

1. **OOS 2025 ΔCAGR +2.23pp** — 本物のモメンタム継続効果 (B_ATR_EXT 13.87% − A_BASELINE 11.64%、Study52)
2. **Seg4 2023 CAGR 42.59% → 44.90%** — Bull 環境での Winner 保持延長
3. **IS MaxDD 変化なし** — リスク中立
4. **実装コスト最低** — 1 パラメータ変更のみ

### 採用理由: EQ_SCALE Addon

1. **WF 5/5 達成** — Baseline 4/5 → 1 fold 改善
2. **Seg3_2022: -5.11% → -2.65%** — 弱気市場を唯一合格水準に引き上げた
3. **メカニズムの健全性**: 「勝ちにのみ乗る」→ Bear 環境でアドオン発火が自然抑制 = 自動 de-risk
4. **Fold variance 最小** — 全ケース中最も安定

### 除外理由: VOL_ADJ

1. **OOS 2025: -0.96pp** — Baseline を下回る唯一のケース
2. **IS MaxDD: -1.99pp 悪化** — calm 日の slot 増加がリスク増大
3. **Bear 保護ゼロ** — Seg3_2022 に変化なし (WF 4/5 のまま)
4. **Fold variance 最大** — 全ケース中最も不安定 (std=22.14)
5. **VOL_ADJ 除外で全指標改善** — E_COMBINED vs D_ATR_EQ: MaxDD +1.99pp、OOS +1.58pp 改善

### D_ATR_EQ が Production Candidate となった経緯

```
Study40  ATR Extension    WF検証 → ADOPT (OOS momentum)
Study41  VOL_ADJ          WF検証 → ADOPT (当時)
Study45  EQ_SCALE         WF検証 → ADOPT (Robustness)
Study46  D_COMBINED       WF 5/5, +6.07pp (intersection バグ + corona 除外の産物)
Study47  E_COMBINED       Production Candidate (旧)

--- 2026-06-28 Integrity Fix ---

Study48  intersection→union バグ発見・修正
  旧 CAGR 20.51% = 4055.T IPO による期間圧縮 (4.27yr での計算)
  正: IS CAGR 12.19% (6.84yr、2018-2024)

Study50  Post-Integrity Revalidation
  新 Baseline: IS 12.63%, WF 4/5
  全機能 IS ΔCAGR < 1pp (IS では識別力なし)
  E_COMBINED WF +1.37pp / D_EQ_SCALE WF 5/5

Study51  Feature Attribution Audit
  採用基準を ΔCAGR → WF 5/5 + Seg3 改善へ変更
  VOL_ADJ = No Evidence (OOS 負, MaxDD 悪化, Fold variance 最大)

Study52  VOL_ADJ 除外最終確認
  D_ATR_EQ (ATR+EQ, VOL_ADJ なし) を初めて単体検証
  → E_COMBINED より全指標改善 → Production Configuration 確定
```

---

## 過去数値との比較 (重要: 旧数値廃止)

| 数値 | 旧値 | 正値 | 廃止理由 |
|---|---|---|---|
| IS CAGR (E_COMBINED) | **20.51%** | **12.19%** | intersection バグによる期間圧縮 |
| WF ΔCAGR (D_COMBINED) | **+6.07pp** | **+1.37pp (E)** | corona 除外 + バグの二重効果 |
| IS CAGR (D_ATR_EQ) | — | **12.37%** | Study52 新規確定値 |
| OOS CAGR (D_ATR_EQ) | — | **13.48%** | Study52 新規確定値 |

---

## Integrity Check (CI 相当)

`python tools/integrity_check.py` → **25/25 PASS**  
実行日: 2026-06-28  
報告書: `reports/integrity_check_latest.json`

---

## 次フェーズ

1. `run_live_signal.py` の research_candidates injection points 有効化 (atr_extension + eq_scale_addon)
2. Shadow 30 日間 (ATR Extension + EQ_SCALE)
3. Limited Live 移行 (Study19 プロセス再適用)
4. ATR Extension OOS 継続監視 (2026 年 OOS で +2.23pp が継続するか確認)
