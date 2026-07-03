# engine_path_A.md — APRIL_REPRO_A 実行経路（20.1% CAGR / 0.859 Sharpe）

生成日: 2026-06-26
由来スクリプト: `decay_audit_202606.py` step0 / `study35_wf_true_oos_validation_202606.py` A_APRIL_REPRO

---

## 1. 呼び出し元

```
decay_audit_202606.py::main()
  → load_all()
  → bt.run_scenario(scenario="BASELINE", ...)
```

```
study35_wf_true_oos_validation_202606.py::main()
  → load_all() [atr_sizing_diagnostic_202606 経由]
  → run_period(cfg, ..., conf=A_APRIL_REPRO, ...)
    → bt.run_scenario(scenario="BASELINE", ...)
```

---

## 2. データロード経路

| ステップ | 関数 | 引数 |
|---|---|---|
| ユニバース取得 | `download_universe()` | start=2018-01-01, end=2026-06-23, min_days=500 |
| TOPIX取得 | `_download_topix()` | start=2018-01-01, end=2026-06-23 ← **取得するが run_scenario には渡さない** |
| RSR計算 | `calc_universe_rsr()` | RSR42全42銘柄 |
| alpha_df計算 | `calc_composite_alpha_matrix().shift(1)` | window=90, 全期間 |
| regime_df計算 | `_calc_regime(topix_close)` | TOPIX MA200/MA50ベース |
| tech_matrices | `_precompute_tech_matrices()` | ATR20/high200/high50_close/ATR90中央値 |
| sym_active_df | `build_dyn_rsr42_active()` | **構築するが run_scenario には渡さない** |

---

## 3. run_scenario 呼び出しパラメータ

```python
bt.run_scenario(
    scenario        = "BASELINE",           # alpha rankingなし、均等weight
    universe_raw    = universe_raw,
    rsr_df          = rsr_df,
    alpha_df        = alpha_df,             # 計算済みだがBASELINEでは未使用
    regime_df       = regime_df,
    trade_syms      = trade_syms,
    rsr_syms        = rsr_syms,
    cfg             = cfg,                  # load_strategy_config()の戻り値
    start           = "2018-01-01",
    end             = "2024-12-31",
    verbose         = False,
    tech_matrices   = tech_matrices,        # ATR等、BASELINE+no-ATR-trailでは未使用
    capital         = cfg.portfolio.capital, # = 3,000,000
    min_hold        = cfg.risk.min_hold_days,# = 3（strategy.yaml実測）
    market_shock_mode = "full_exit",
    rsr_exit_threshold = 75.0,
    sym_active_df   = None,                 # ★ 動的ユニバース無効
    enable_atr_trailing_prod = False,
    enable_multilayer_rsr    = False,
    enable_atr_risk_sizing   = False,
    enable_mtf_filter        = False,
    risk_sizing_pct          = PROD_RISK_PCT,
    sizing_mode              = "existing",
    # ★ topix_close は渡さない → デフォルトNone
)
```

---

## 4. run_scenario 内部経路（BASELINE / topix_close=None）

### 4.1 初期化
| 変数 | 値 | 効果 |
|---|---|---|
| `use_alpha_rank` | False | alpha_mat = zeros（alpha_df無効） |
| `use_regime` | False | regime（risk_off/crash）無効 |
| `use_breadth_regime` | False | breadth_series無効 |
| `_enable_conc_caps` | False | セクター/クラスターキャップ無効 |

### 4.2 topix_close=None の影響
```python
# market_ret_arr はゼロで初期化されたまま
market_ret_arr = np.zeros(n_dates, dtype=np.float32)
topix_ret_20d_arr = None   # ← None のまま
topix_ret_60d_arr = None   # ← None のまま
_bear_arr = None            # ← None のまま
```

### 4.3 日次ループ制御
| 条件 | 結果 |
|---|---|
| `market_shock = bool(0.0 <= -0.05)` | **常時 False** → MARKET_SHOCK_EXIT 発火なし |
| `_r20 = 0.0` (topix_ret_20d=None) | `_gross_cap = 1.0` 常時 → GE Cap 発動なし |
| `_is_bear = False` (bear_arr=None) | bear_sector_cap 不適用（かつ _enable_conc_caps=False） |

### 4.4 Entry evaluator（FujikoStrategy）
- `min_rsr=75.0` (cfg.fujiko.min_rsr)
- `turtle_entry=20` (cfg.fujiko.turtle_entry)
- `turtle_exit=55` (cfg.fujiko.turtle_exit)
- `min_sepa=6` (cfg.fujiko.min_sepa)
- `mom_period=21` (cfg.fujiko.mom_period)
- 信号: `precompute_signals()` でfull dataset事前計算 → `signal_mat` に格納

### 4.5 Exit evaluator（RSR閾値ベース）
優先順位（コード順）:
1. ATRトレーリング (`enable_atr_trailing_prod=False` → スキップ)
2. TIME_STOP (`max_hold_days=60`)
3. RSR Exit: `rsr_val < 75.0` かつ `hold_days >= 3` → RSR_EXIT
4. FujikoStrategy signal=-1 かつ hold_days>=3 → STRATEGY_EXIT (RSR_MOMENTUM or TURTLE_EXIT)

### 4.6 Portfolio manager（positionサイジング）
```
sizing_mode = "existing"
n_remaining = BUY候補のうち未保有件数
effective_slots = min(open_slots, max(1, n_remaining))
alloc = (cash / effective_slots) * cb_scale * ext_scale
alloc = min(alloc, capital * 0.25)   # max_single_weight=25%
qty = int(alloc / buy_px / 100) * 100
```

### 4.7 Cash accounting
```
BUY:  cash -= qty * buy_px * (1 + COST_ONE_WAY)
SELL: cash += qty * sell_px * (1 - COST_ONE_WAY)
COST_ONE_WAY = SLIPPAGE + COMMISSION = 0.001 + 0.00055 = 0.00155
```

### 4.8 Equity curve builder
```python
equity_curve[i] = cash + Σ(qty * close_today)
```

### 4.9 Performance metric calculator
```
年数 = (end - start) / 252
CAGR = (equity[-1] / capital)^(1/年数) - 1  → 20.1%
Sharpe = mean(daily_ret) / std(daily_ret) * sqrt(252)  → 0.859
MaxDD = min((equity - running_max) / running_max)
```

---

## 5. 出力実績（2018-2024 IS）

| 指標 | 値 |
|---|---|
| CAGR | **+20.1%** |
| Sharpe | **0.859** |
| MaxDD | -15.7% |
| n_trades | 216 |
| avg_exposure | ~34-35% |
| exit_reason | STRATEGY_EXIT, RSR_EXIT, TIME_STOP のみ |
| MARKET_SHOCK_EXIT | **0件** |

---

## 6. 特記事項

- `_bt.CAPITAL` / `_bt.TURTLE_EXIT` は現行 composite_alpha_bt.py に存在しない
  - 旧コード（2026-03-31当時）はモジュール定数として保持、現行はcfg経由に変更済み
  - 数値は同一（3,000,000 / 55）だが、実行経路が異なる
- `min_hold_sensitivity.py`（22.4%生成元）は現行コードでは AttributeError で実行不能
  - これがA(20.1%) vs 原本(22.4%) 残差-2.3ppの根本原因（再現不能領域）
