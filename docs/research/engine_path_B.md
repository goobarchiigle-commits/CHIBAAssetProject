# engine_path_B.md — APRIL_REPRO_B 実行経路（17.4% CAGR / 0.769 Sharpe）

生成日: 2026-06-26
由来スクリプト: `study36_april13_forensic_202606.py`

---

## 1. 呼び出し元

```
study36_april13_forensic_202606.py::main()
  → build_common_dataset("2024-12-31")     ← engine_path_A との第1の差異
  → cab.run_scenario(scenario="BASELINE", topix_close=ds["topix_close"], ...)
```

---

## 2. データロード経路

| ステップ | 関数 | 引数 |
|---|---|---|
| ユニバース取得 | `download_universe()` | start=2018-01-01, **end=2024-12-31**, min_days=500 |
| TOPIX取得 | `_download_topix()` | start=2018-01-01, **end=2024-12-31** ← **run_scenario に渡す** |
| RSR計算 | `calc_universe_rsr()` | RSR42全42銘柄 |
| alpha_df | **None** ← 計算しない（BASELINEで未使用のため省略） |
| regime_df | `_calc_regime(topix_close)` | TOPIX MA200/MA50ベース |
| tech_matrices | `_precompute_tech_matrices()` | ATR20/high200/high50_close/ATR90中央値 |
| breadth_series | `_calc_breadth(universe_raw)` | RSR42銘柄の上昇比率 |
| sym_active_df | `build_dyn_rsr42_active()` | **構築するが run_scenario には渡さない** |

---

## 3. run_scenario 呼び出しパラメータ

```python
cab.run_scenario(
    scenario        = "BASELINE",
    universe_raw    = ds["universe_raw"],
    rsr_df          = ds["rsr_df"],
    alpha_df        = None,                 # ★ BASELINEで未使用のためNone
    regime_df       = ds["regime_df"],
    trade_syms      = ds["trade_syms"],
    rsr_syms        = ds["rsr_syms"],
    cfg             = cfg,
    start           = "2018-01-01",
    end             = "2024-12-31",
    verbose         = False,
    tech_matrices   = ds["tech_matrices"],
    breadth_series  = ds["breadth_series"], # ★ 渡すがBASELINEでは未使用
    capital         = 3_000_000,
    min_hold        = 3,
    topix_close     = ds["topix_close"],    # ★★ 最重要差分 ← APRIL_REPRO_A と異なる
    market_shock_mode = "full_exit",
    sym_active_df   = None,
    enable_simple_rsr_exit = True,
    rsr_exit_threshold     = 75.0,
    enable_multilayer_rsr  = False,
    use_fixed_pct_trail    = False,
    enable_atr_trailing_prod = False,
    enable_mtf_filter        = False,
    enable_atr_risk_sizing   = False,
    sizing_mode              = "existing",
)
```

---

## 4. run_scenario 内部経路（BASELINE / topix_close=実データ）

### 4.1 初期化（APRIL_REPRO_A と同一）
| 変数 | 値 |
|---|---|
| `use_alpha_rank` | False |
| `use_regime` | False |
| `use_breadth_regime` | False |
| `_enable_conc_caps` | False（sym_active_df=Noneのため） |

### 4.2 topix_close=実データ の影響（★ APRIL_REPRO_A との差異）

```python
# topix_close が None でないので以下が実行される:
topix_ret = topix_close.pct_change()
market_ret_arr = _take_series_to_array(topix_ret, ...)  # 実際のTOPIX日次リターン
topix_ret_20d_arr = _take_series_to_array(topix_close.pct_change(20), ...)
topix_ret_60d_arr = _take_series_to_array(topix_close.pct_change(60), ...)

_topix_ma200 = topix_close.rolling(200, min_periods=100).mean()
_bear_s = (topix_close < _topix_ma200)
_bear_arr = _bear_s.values.astype(bool)
```

### 4.3 日次ループ制御（APRIL_REPRO_A との差分箇所）

#### 差分1: market_shock 判定（market_ret_arr に実値）
```python
market_shock = bool(float(market_ret_arr[i]) <= -0.05)
# APRIL_REPRO_A: 常時 False（zeros）
# APRIL_REPRO_B: 実TOPIX リターンを参照 → 以下の日に True:
#   2020-03-09  -5.64%  → MARKET_SHOCK_EXIT 発火
#   2020-03-13  -5.16%  → MARKET_SHOCK_EXIT 発火
#   2024-08-02  -6.11%  → MARKET_SHOCK_EXIT 発火
#   2024-08-05  -10.82% → MARKET_SHOCK_EXIT 発火
# market_shock_mode="full_exit": 全ポジション強制クローズ + 当日新規BUY禁止
```

#### 差分2: gross_exposure_cap（20d/60d リターン参照）
```python
_r20 = float(topix_ret_20d_arr[i])   # 実際の20日リターン
_r60 = float(topix_ret_60d_arr[i])   # 実際の60日リターン
if _r20 < -0.05:
    _gross_cap = 0.6    # 新規BUYを60%露出超でブロック
elif _r60 < -0.08:
    _gross_cap = 0.4    # 新規BUYを40%露出超でブロック
# APRIL_REPRO_A: topix_ret_20d_arr=None → _r20=0.0 → _gross_cap=1.0 常時
# APRIL_REPRO_B: 最初のGE Cap発動日 = 2018-02-06 (20d_ret=-7.81%)
```

#### 差分3: bear adaptive cap（_bear_arrに実値）
```python
_is_bear = bool(_bear_arr[i])
_sec_cap_eff = _bear_sector_cap if _is_bear else MAX_SECTOR_WEIGHT
# ただし _enable_conc_caps=False なので sector/cluster cap ブロック自体は不発動
# → この差分は実質無効（_enable_conc_caps gate で遮断）
```

### 4.4 Entry/Exit evaluator
APRIL_REPRO_A と完全同一（FujikoStrategy, rsr_exit=75.0, min_hold=3）

### 4.5 Portfolio manager / Cash accounting
APRIL_REPRO_A と完全同一（sizing_mode="existing", max_single_weight=0.25）

---

## 5. 出力実績（2018-2024 IS）

| 指標 | 値 |
|---|---|
| CAGR | **+17.4%** |
| Sharpe | **0.769** |
| MaxDD | 不明（実行確認要） |
| n_trades | APRIL_REPRO_A より少ない（MARKET_SHOCK_EXIT分だけ減少） |
| exit_reason | STRATEGY_EXIT, RSR_EXIT, TIME_STOP + **MARKET_SHOCK_EXIT 4件** |
| MARKET_SHOCK_EXIT | **4件** (2020-03-09, 2020-03-13, 2024-08-02, 2024-08-05の翌日実行) |

---

## 6. 特記事項

- `alpha_df=None` と計算済み alpha_df の差: BASELINEで `use_alpha_rank=False` のため **影響ゼロ**
  - `alpha_mat = np.zeros(...)` どちらでも同一
- `breadth_series` 渡し: BASELINEで `use_breadth_regime=False` のため **影響ゼロ**
- データ期間差（end=2024-12-31 vs 2026-06-23）: min_days=500 フィルターによる微細な銘柄差の可能性あり（RSR42全銘柄は十分な期間がある想定で実質差はゼロと推定）
