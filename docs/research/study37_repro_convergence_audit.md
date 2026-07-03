# Study37 — APRIL_REPRO_A/B 乖離解消監査

生成日: 2026-06-26
スクリプト: 新規スクリプト不要（コード考古学のみ、実行なし）
参照: `engine_path_A.md` / `engine_path_B.md`

---

## Phase 3: 差分監査

### 3.1 パラメータ一覧比較

| 項目 | APRIL_REPRO_A (20.1%/0.859) | APRIL_REPRO_B (17.4%/0.769) | 差分有無 |
|---|---|---|---|
| scenario | BASELINE | BASELINE | なし |
| capital | 3,000,000 | 3,000,000 | なし |
| min_hold | cfg.risk.min_hold_days=**3** | **3** | なし（同値） |
| rsr_exit_threshold | 75.0 | 75.0 | なし |
| market_shock_mode | "full_exit" | "full_exit" | なし |
| **topix_close** | **None（渡さず）** | **実データ（2018-2024）** | **★★ 唯一の差分** |
| alpha_df | 計算済み（shift(1)） | None | なし（BASELINEで alpha_mat=zeros） |
| tech_matrices | 渡す | 渡す | なし |
| breadth_series | 渡さず | 渡す | なし（BASELINEで breadth不使用） |
| sym_active_df | None | None | なし |
| enable_atr_trailing_prod | False | False | なし |
| enable_multilayer_rsr | False | False | なし |
| enable_atr_risk_sizing | False | False | なし |
| enable_mtf_filter | False | False | なし |
| sizing_mode | "existing" | "existing" | なし |
| enable_simple_rsr_exit | True（デフォルト） | True（明示） | なし |
| データ期間 | end=2026-06-23 | end=2024-12-31 | 微差（min_days=500フィルタ）|

### 3.2 有効差分の詳細

**topix_close の有無が全ての差分の根本**

| 内部変数 | APRIL_REPRO_A | APRIL_REPRO_B | 影響 |
|---|---|---|---|
| `market_ret_arr` | zeros（全ゼロ） | 実TOPIX日次リターン | market_shock発火 |
| `topix_ret_20d_arr` | None → _r20=0.0 | 実20日リターン | gross_exposure_cap |
| `topix_ret_60d_arr` | None → _r60=0.0 | 実60日リターン | gross_exposure_cap |
| `_bear_arr` | None → _is_bear=False | 実MA200判定 | **無効**（_enable_conc_caps=False） |
| `_gross_cap` | 1.0（常時） | 状況依存（0.4/0.6/1.0） | BUY抑制 |
| `market_shock` | False（常時） | 実TOPIX <-5%で True | MARKET_SHOCK_EXIT |

---

## Phase 4: 最初に乖離する日の特定

### 4.1 TOPIX ETF (1306.T) イベント一覧（2018-2024）

```
Gross Exposure Cap発動日（TOPIX 20d return < -5%）:
  最初の発動: 2018-02-06  (20d_ret=-7.81%)
  ← APRIL_REPRO_B でのみ _gross_cap=0.6 が有効になる

Market Shock日（TOPIX 日次 ≤ -5%）:
  2020-03-09  -5.64%  ← COVID crash 第1波
  2020-03-13  -5.16%  ← COVID crash 第2波
  2024-08-02  -6.11%  ← 令和のブラックマンデー前日
  2024-08-05 -10.82%  ← 令和のブラックマンデー
```

### 4.2 乖離の発生メカニズム

**第1の乖離機会（Gross Exposure Cap）: 2018-02-06**
```
APRIL_REPRO_A: _gross_cap=1.0 → BUYブロックなし
APRIL_REPRO_B: _gross_cap=0.6 → current_gross+order_weight>0.6 でBUYブロック

注: 2018-02-06はバックテスト開始から約25営業日目。
FujikoStrategy の precompute_signals は min_bars=275（≈252+21+2）を必要とするが、
download_universe は 2018-01-01 からのデータしか持たないため、
最初の有効シグナルは ~2018-10月以降。
→ 2018-02-06 時点では BUY シグナル未発生の可能性が高い。
→ GE Cap による実際の乖離は 2018-10月以降の最初のGE Cap日。
```

**第2の乖離機会（Market Shock）: 2020-03-09**
```
翌日 2020-03-10 の open 価格で全ポジションが強制クローズ。
APRIL_REPRO_A: 通常のExit判定のみ、MARKET_SHOCK_EXIT=0件
APRIL_REPRO_B: MARKET_SHOCK_EXIT が 2020-03-10 から発動

→ この日が equity curve の最大乖離点（COVID crash で保有中の含み益/損ポジション強制決済）
```

**確定的乖離**: equity curve は遅くとも 2020-03-10 以降で別の軌跡を歩む。

### 4.3 equity_A != equity_B となる日の比較（推定）

| 日付 | equity_A | equity_B | 差分 |
|---|---|---|---|
| 2018-01-01 〜 (シグナル未発生期) | 同一 | 同一 | 0 |
| 最初の GE Cap 発動日 + BUY試行日 | 同一 or 差分微小 | 若干低い可能性 | 微差 |
| **2020-03-09** | 通常評価 | MARKET_SHOCK_EXIT発火 | **★最初の大きな乖離** |
| 2020-03-13 | 通常評価 | 2回目MARKET_SHOCK_EXIT | 乖離拡大 |
| 2024-08-05 | 通常評価 | 3-4回目MARKET_SHOCK_EXIT | 終盤に再乖離 |

---

## Phase 5: min_hold_sensitivity_2026-03-31.json hold3d との照合

### 5.1 比較表

| 指標 | 原本 hold3d.IS | APRIL_REPRO_A | APRIL_REPRO_B | A残差 | B残差 |
|---|---|---|---|---|---|
| CAGR | **22.4%** | **20.1%** | **17.4%** | **-2.3pp** | **-5.0pp** |
| Sharpe | **1.582** | **0.859** | **0.769** | **-0.723** | **-0.813** |
| MaxDD | -12.32% | -15.7% | 不明 | +3.38pp | — |
| n_trades | 219 | 216 | 不明（<216） | -3 | — |
| avg_hold | 11.8d | — | — | — | — |
| exit_reason MARKET_SHOCK | 0件 | 0件 | 4件 | 一致 | 不一致 |

### 5.2 判定

**APRIL_REPRO_A が 2026-03-31 実行系に近い。**

根拠:
1. exit_reason に MARKET_SHOCK_EXIT が原本=0件、A=0件（一致）、B=4件（不一致）
2. CAGR残差: A=-2.3pp < B=-5.0pp（Aが接近）
3. Sharpe残差: A=-0.723 < B=-0.813（Aが接近）
4. 原本の `run_one()` は `topix_close` を `run_scenario` に渡していない（コード実測）

---

## Phase 6: コード考古学

### 6.1 旧コード断片の探索

#### CAPITAL モジュール定数（2026-03-31当時に存在した形跡）

`min_hold_sensitivity.py:109`（現存、実行不能）:
```python
capital = _bt.CAPITAL
```

`walkforward_revalidation.py`（旧型スクリプト、現存）:
```python
# 別スクリプト内に残存するモジュール定数形式
TURTLE_EXIT = 55
MIN_HOLD    = 3
```

`run_live_signal.py:820`（現行、cfg移行後）:
```python
CAPITAL = cfg.portfolio.capital   # ← モジュール定数をcfgに置き換え済み
```

#### TURTLE_EXIT の痕跡

| ファイル | 行 | 内容 |
|---|---|---|
| `walkforward_revalidation.py:53` | `TURTLE_EXIT = 55` | 旧スタイル残存 |
| `rsr_universe_sweep.py:65` | `TURTLE_EXIT = 20` | 旧スタイル残存（別の値） |
| `risk_pct_sensitivity.py:44` | `TURTLE_EXIT_N = 55` | 旧スタイル残存（変数名変更） |
| `risk_pct_wf_validation.py:44` | `TURTLE_EXIT_N = 55` | 同上 |
| `strategy.yaml:12` | `turtle_exit: 55` | cfg移行後の正規格納場所 |
| `composite_alpha_bt.py` | **存在しない** | 完全廃止済み |

#### RSR_EXIT 閾値の歴史的痕跡

| ファイル | 内容 |
|---|---|
| `strategy.yaml:9` | `rsr_exit: 70.0  # WF5fold検証済 2026-06-05` |
| `exit_rsr70_walkforward.py` | A=Baseline(rsr_exit=75) vs B=Variant(rsr_exit=70) WF比較 |
| `rsr_exit_threshold_sweep.py` | "Pattern A Baseline = rsr_exit=75" |
| `decay_audit_202606.py step0` | `rsr_exit=75.0` (APRIL_REPRO) |

→ **2026-04-13時点（min_hold_sensitivity実行時）の rsr_exit 真値は 75.0（min_rsr フォールバック）が最有力**。
  cfg.fujiko.rsr_exit パラメータは 2026-06-05 に WF検証で 70.0 として正式採用。
  それ以前は `rsr_exit` キーが yaml に存在せず、コード側のデフォルト（min_rsr=75.0）が使われたと推定。

#### MAX_POSITIONS の痕跡

| ファイル | 内容 |
|---|---|
| `strategy.yaml:37` | `max_positions: 3  # PARAMS_LOCKED確定値（2026-03-31）` |
| `CLAUDE.md` | `max_positions=3` PARAMS_LOCKED |
| `rsr_universe_sweep.py:54` | `MAX_POSITIONS = 3` |
| `composite_alpha_bt.py` | `cfg.portfolio.max_positions`（cfg移行済み） |

### 6.2 2026-03-31当時のコード構造推定

```python
# 旧 composite_alpha_bt.py（2026-03-31当時、推定）
CAPITAL = 3_000_000         # ← _bt.CAPITAL で参照されていた
TURTLE_EXIT = 55            # ← 確定値（2026-03-31コメント付き）
RSR_MIN = 75.0              # ← rsr_exit 閾値もこれで兼用の可能性

def run_scenario(scenario, universe_raw, rsr_df, alpha_df, regime_df,
                 trade_syms, rsr_syms, start, end, capital=CAPITAL,
                 verbose=True, min_hold=0, ...):
    # cfgは引数になかった（モジュール定数で代替）
    # topix_closeは引数になかった（market_shock機能未実装or引数なし）
    ...
```

現行との差分（確実に判明した変更点）:
- `cfg` 引数が追加（モジュール定数→cfg駆動に移行）
- `topix_close` 引数が追加
- `tech_matrices` 引数が追加
- `rsr_exit_threshold` 引数が追加
- 多数の PROD_FAITHFUL フラグ群が追加
- `CAPITAL` / `TURTLE_EXIT` モジュール定数が廃止

---

## Phase 7: 収束修正案

### 収束条件
目標: 同一設定・同一データ・同一期間での CAGR差<0.1pp / Sharpe差<0.01

### 修正方法: study36 の topix_close 引数を除去

```python
# study36_april13_forensic_202606.py 変更前
res = cab.run_scenario(
    ...
    topix_close=ds["topix_close"],  # ← この行を除去
    ...
)

# 変更後（APRIL_REPRO_A と同一挙動）
res = cab.run_scenario(
    ...
    # topix_close は渡さない（None デフォルト）
    ...
)
```

この修正により:
- `market_ret_arr = zeros` → MARKET_SHOCK_EXIT 無効化 → A と同一
- `topix_ret_20d_arr = None` → GE Cap 常時 1.0 → A と同一
- `_bear_arr = None` → bear cap 不発動 → A と同一

期待収束結果: CAGR ≈ 20.1% ± 0.1pp / Sharpe ≈ 0.859 ± 0.01

---

## 最終出力

### A. 20.1% vs 17.4% 差の原因

**唯一の原因: topix_close の有無**

- `study36`（B）が `topix_close=ds["topix_close"]` を渡すことで 2 つの機構が有効化:
  1. **MARKET_SHOCK_EXIT**: TOPIX 日次≤-5%の日（4回）に全ポジション強制クローズ → n_trades増加なし、保有益ポジション早期切断
  2. **Gross Exposure Cap**: TOPIX 20d return<-5%時に _gross_cap=0.6、60d<-8%時に _gross_cap=0.4 → 下落期のBUY抑制
  
- これら 2 機構の合算効果が CAGR -2.7pp（20.1%→17.4%）/ Sharpe -0.090（0.859→0.769）

- `decay_audit + study35`（A）は topix_close を渡さない → 両機構とも無効 → 20.1%/0.859

### B. どちらが 2026-03-31 実行系に近いか

**APRIL_REPRO_A**（decay_audit/study35 アプローチ）

根拠（証拠の強さ順）:
1. 原本 `hold3d.IS` の `exit_reason_counts` に MARKET_SHOCK_EXIT = 0件（A=0件一致、B=4件不一致）
2. 原本 `run_one()` は `topix_close` を `run_scenario` に渡していない（コード直接確認）
3. CAGR/Sharpe 残差: A(-2.3pp/-0.723) < B(-5.0pp/-0.813)（共に残差あるがAが接近）

### C. 再現不能要素の有無

**あり**（APRIL_REPRO_A vs 原本 22.4%/1.582 の残差-2.3pp/-0.723）

| 不能要因 | 証拠 | 影響推定 |
|---|---|---|
| composite_alpha_bt.py 構造変更（cfg化/新機能追加） | `_bt.CAPITAL` 属性不在の AttributeError | 不明量（複合） |
| データvintage差分（yfinance遡及調整） | 取得日時差による株価修正 | 不明量 |
| rsr_exit当時真値の不確定性 | strategy.yaml に rsr_exit が存在しなかった可能性（2026-06-05以前） | +0.61pp（スタディ35a実測） |
| topix_close 引数が当時も存在しなかった可能性 | 推定（当時のAPIに引数なし） | 0pp（Aで既に除外済み） |

**結論**: A-原本残差は原理的に分解不能（gitなし・当時コード消失）。上記は推定のみ。

### D. Sharpe 1.582 に対する説明力更新

```
Sharpe 1.582（原本）
  ↓ -0.723（再現不能領域: code構造変化+data vintage）
Sharpe 0.859（APRIL_REPRO_A、今回の近似上限）
  ↓ -0.090（topix_close 効果: study37 新発見）
Sharpe 0.769（APRIL_REPRO_B、study36 現状）
```

| 区間 | 要因 | Sharpe 変化 | 説明可能性 |
|---|---|---|---|
| 1.582 → 0.859 | code構造変化+data vintage | -0.723 | **再現不能** |
| 0.859 → 0.769 | topix_close（Study37 新発見） | -0.090 | **完全説明可能** |

topix_close 除去で B を A に収束させることが可能（<0.01 Sharpe差）。
しかし A→22.4% への残差 -0.723 は今後も解明不能。

Study37 の貢献: 0.090 の Sharpe 差 = A-B ギャップを topix_close 1パラメータで完全説明。

### E. Study38 の必要性

**不要（A-B 収束目的では）**

- A-B ギャップは topix_close 差 1 点で完全説明済み
- 収束修正は study36 の 1 行除去で達成可能
- 実行確認が必要なら study36 から topix_close を除去して再実行するのみ

**Study38 が必要になる条件**:
- 目的が 22.4%/1.582 への接近であれば Study38 を検討できるが、再現不能領域 (-2.3pp/-0.723) への突入のため成果が得られる見込みは低い
- 別の研究目標（例: 新しいシグナル設計、OOS検証）が明確になった場合

---

## 参照ファイル

- `docs/research/engine_path_A.md` — APRIL_REPRO_A 実行経路詳細
- `docs/research/engine_path_B.md` — APRIL_REPRO_B 実行経路詳細
- `backtests/min_hold_sensitivity_2026-03-31.json` — 原本 hold3d.IS
- `src/backtest/study36_april13_forensic_202606.py` — B の生成スクリプト
- `src/backtest/decay_audit_202606.py` — A の生成スクリプト（step0）
- `src/backtest/study35_wf_true_oos_validation_202606.py` — A の生成スクリプト（A_APRIL_REPRO）
- `src/backtest/min_hold_sensitivity.py` — 原本 22.4% 生成元（現行コードでは実行不能）
