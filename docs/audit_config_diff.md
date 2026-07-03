# Backtest Configuration Audit — E_COMBINED 数値乖離検証
作成: 2026-06-28

## 1. 設定差分表

| 項目 | Chart (dashboard) | Study47 | 差分・影響 |
|---|---|---|---|
| **スクリプト** | `scratchpad/run_full_backtest.py` | `study47_production_candidate_verification.py` | 別スクリプト |
| **データセット構築** | 直接 `cab.download_universe()` | `build_common_dataset(DATA_END)` via `snapshot_archaeology_202606.py` | 呼び出し方法が異なる |
| **Universe source** | `rsr_universe_42.csv` (手動除外) | `cab._load_rsr_universe()` | ロード方式差異 |
| **Universe size** | **RSR41 (41銘柄)** | **RSR42 (42銘柄)** | **4055.T の有無** |
| **4055.T 含有** | **除外** | **含む** | **★ROOT_CAUSE** |
| **min_days** | **100** | **500** | ただし全銘柄が500以上あるため実効差なし |
| **RSR計算母集団** | RSR41 prices (41銘柄) | RSR42 prices (42銘柄) | RSR順位が若干異なる |
| **Dynamic Universe** | ON (`sym_active_df`) | ON (`sym_active_df`) | 同一 |
| **Bear Filter** | `bear_dynamic_filter.enabled=false` → OFF | `bear_dynamic_filter.enabled=false` → OFF | 同一 |
| **ATR Extension** | `exit_policy="A"`, defer=5d | `exit_policy="A"`, defer=5d | 同一 |
| **VolAdj (D_VOL_ADJ)** | ON, std<0.8%→max_pos=4 | ON, std<0.8%→max_pos=4 | 同一 |
| **EQ_SCALE (D_EQ_SCALE)** | `addon_policy="D"`, frac=0.25, atr_mult=1.0 | `addon_policy="D"`, frac=0.25, atr_mult=1.0 | 同一 |
| **Position cap** | max=3 (calm時4) | max=3 (calm時4) | 同一 |
| **Slippage** | 0.1% (strategy.yaml) | 0.1% (strategy.yaml) | 同一 |
| **Commission** | 0.055% (strategy.yaml) | 0.055% (strategy.yaml) | 同一 |
| **min_hold** | 3d | 3d | 同一 |
| **rsr_exit_threshold** | 70.0 | 70.0 | 同一 |
| **capital** | ¥3,000,000 | ¥3,000,000 | 同一 |
| **ラベル期間 start** | 2018-01-01 | 2018-01-01 | 同一ラベル |
| **ラベル期間 end** | 2025-12-31 | 2024-12-31 (IS only) | +1年差 |
| **実際期間 start** | **2018-01-01** | **2020-08-11** | **★+966日差** |
| **実際期間 end** | 2025-12-30 | 2024-12-31 | 1年差 |
| **実際年数** | **7.81yr** | **4.27yr** | **3.54yr差 (45%圧縮)** |

---

## 2. 各設定での実測値

### Study47 E_COMBINED (RSR42 / 2020-08-11 ~ 2024-12-31 / 4.27yr)
| 指標 | 値 |
|---|---|
| CAGR | **20.51%** |
| MaxDD | -19.81% |
| Sharpe | 0.880 |
| Calmar | 1.035 |
| Trades | 205 |
| 実際期間 | 2020-08-11 ~ 2024-12-31 |
| 実際年数 | 4.27yr |

### Chart Dashboard (RSR41 / 2018-01-01 ~ 2025-12-30 / 7.81yr)
| 指標 | 値 |
|---|---|
| CAGR | **11.03%** |
| MaxDD | -18.02% |
| Sharpe | 0.547 |
| Calmar | 0.612 |
| Trades | 316 |
| 実際期間 | 2018-01-01 ~ 2025-12-30 |
| 実際年数 | 7.81yr |

---

## 3. CAGR 乖離 9.48pp の寄与度分解

比較のため、RSR41 を同じ期間 (2020-08-11 ~ 2024-12-31) に切り出して比較する。

### Step 1: Universe 差異 (RSR42 vs RSR41) — 同期間

| | RSR42 (Study47) | RSR41 (同期間切出し) | 差 |
|---|---|---|---|
| 期間 | 2020-08-11~2024-12-31 | 2020-08-11~2024-12-31 | — |
| 年数 | 4.27yr | 4.27yr | — |
| 開始資本 | ¥3,000,000 (fresh) | ¥2,829,300 (2018~累積後) | ▲5.7% |
| CAGR | 20.51% | 21.85% | +1.34pp |
| MaxDD | -19.81% | -17.98% | +1.83pp |
| Sharpe | 0.880 | 0.917 | +0.037 |

→ **Universe 寄与: Chart の方が +1.34pp 高い** (4055.T は RSR42 内で足を引っ張る銘柄)

### Step 2: 期間差異 (2020-2024 vs 2018-2025) — RSR41 同一 Universe

| | RSR41 2020-08-11~2024-12-31 | RSR41 2018-01-01~2025-12-30 | 差 |
|---|---|---|---|
| CAGR | 21.85% | 11.03% | **-10.82pp** |
| 年数 | 4.27yr | 7.81yr | +3.54yr |

期間延長 (+3.54yr) に含まれる要因:

| 要因 | 内容 | CAGR 影響 |
|---|---|---|
| 2018~2020-08-11 追加 | RSR41 総リターン -6.73% (Corona ショック含む) | 約 -7pp |
| 2025年追加 | +3.07% (低成長年、7.81yr 基準で希薄化) | -1.15pp |
| 期間圧縮解消 | 4.27yr → 7.81yr で同一総リターンを長期間で割る | 残差 |

### CAGR 乖離 9.48pp の帰属まとめ

| 要因 | 寄与 pp |
|---|---|
| 期間延長 (2018-2020 低迷期追加) | ≈ -7.0 pp |
| 期間延長 (2025 追加・希薄化) | ≈ -1.2 pp |
| Universe 差 (RSR42→RSR41, 小幅改善) | ≈ +1.3 pp |
| 開始資本差 (Study47 fresh ¥3M vs RSR41 累積 ¥2.83M) | ≈ -2.6 pp |
| **合計** | **≈ -9.5 pp** |

---

## 4. 指標対比サマリ

| 指標 | Study47 (FULL_IS) | Chart (Full 2018-2025) | 乖離 |
|---|---|---|---|
| CAGR | 20.51% | 11.03% | **-9.48pp** |
| MaxDD | -19.81% | -18.02% | +1.79pp |
| Sharpe | 0.880 | 0.547 | -0.333 |
| Calmar | 1.035 | 0.612 | -0.423 |
| Trades | 205 | 316 | +111 |
| 実際年数 | 4.27yr | 7.81yr | 3.54yr 長い |

---

## 5. ROOT_CAUSE

**4055.T (TDC SOFTWARE) の RSR42 収録**

`build_common_dataset()` は `min_days=500` で RSR42 全銘柄をロードし、  
`common_dates = intersection(all_symbols.index)` でバックテスト日付を決定する。  
4055.T の上場日は 2020-08-11 であるため、この1銘柄だけで全バックテストの  
開始日が 2020-08-11 に強制される。

**連鎖効果:**

```
4055.T ∈ RSR42
  → common_dates 開始 = 2020-08-11
  → バックテスト実質期間 = 4.27yr (ラベルは "2018-2024")
  → Corona ショック (2020-02) 除外
  → 2018-2019 低迷期 (-3.6%, +1.4%) 除外
  → 同等の総利益が 45% 短い期間で CAGR 計算される
  → CAGR = 20.51%  (真値比 +9.5pp 過大)
```

**数値的証明:**
- RSR41 を同一期間 (2020-08-11~2024-12-31) に切り出すと CAGR = **21.85%** ≈ Study47 の 20.51% と整合
- RSR41 を真の 2018-2025 全期間で計算すると CAGR = **11.03%**
- 差 9.5pp は「期間圧縮」であり、戦略の実力差ではない

**結論:**  
Strategy Review (Study47) の 20.51% と Dashboard の 11.03% は **別期間・別期間長の数値を並べたものであり、直接比較不可**。  
正当な比較基準は RSR41 の 2018-2025 全期間 (7.81yr) での **CAGR = 11.03%**。  
Study47 の 20.51% はデータ制約による「期間省略の産物」として扱うこと。
