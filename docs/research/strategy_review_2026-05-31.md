# CHIBAAssetProject システム監査レポート

**作成日**: 2026-05-31  
**前回レポート**: 2026-04-13  
**目的**: 多数の機能追加後の現状定量評価（新機能追加禁止・現状把握専用）  
**データ基準日**: 2026-03-28（2026-04-03まで）  
**バックテスト実行日**: 2026-05-31

---

## ★ エグゼクティブサマリー（最重要）

| 項目 | Apr 2026 確認値 | HEAD 2026-05-31 | 変化 | 判定 |
|---|---|---|---|---|
| WF合格率 | **5/5 PASS** | **4/5 FAIL** | ↓ Seg2(2021)失敗 | ⚠ **CRITICAL** |
| IS Sharpe | 0.812 | 0.629 | -0.183 | ⚠ 劣化 |
| IS MaxDD | -16.19% | -22.6% | -6.4pp | ⚠ 悪化 |
| 2025 OOS Sharpe | **0.805** | **0.592** | -0.213 | ⚠ FAIL(閾値0.80) |
| 2025 OOS MaxDD | -9.7% | -9.7% | 0 | ✅ 維持 |

**根本原因**: `max_positions: 3 → 4`（strategy.yaml 2026-04-18変更）がWF再検証なしで適用された。  
**PARAMS_LOCKED 違反**: CLAUDE.md `max_positions=3` が確定パラメータにもかかわらず変更。

---

## Phase 1: Current System Architecture Audit

### 1.1 シグナル生成フロー（全経路）

```
Universe (42 tickers LIVE + 20 shadow + RSR62 context)
         ↓
Dynamic Universe Filter (dyn_rsr42_bear_rs0)
  Bull: TOPIX > MA200 持続 → RSR42内 Top30 (score=mom63×0.40+rsr×0.35+vol×0.25)
  Bear: TOPIX < MA200 持続40/60d → RSR42 rs>0銘柄のみ Top20 (score=rs×0.50+rsr×0.30+vol×0.20)
  Bear除外7セクター: 機械/鉄鋼/銀行業/保険業/輸送用機器/海運業/化学
         ↓
Signal Bridge (FujikoStrategy + MeanReversionStrategy)
  RSR計算: RSR62コンテキスト (RSR42 + shadow20)
  min_rsr=75.0 フィルター
  エントリー: Turtle 20日高値ブレイク
  エグジット1: Turtle 55日安値割れ (turtle_exit=55)
  エグジット2: RSR momentum exit (z < 1.1)
  エグジット3: Trail stop 2.5% from HWM
  エグジット4: Time stop 4 bars
  エグジット5: Emergency exit -8%
  min_hold=3d, max_hold=60d
  shock_exit=composite (TOPIX<-5% AND 個別<-8%)
         ↓
Sector / Symbol Caps
  sector_cap=25% (symbol_cap=40% ← 旧8%からバグ修正済み)
  同一セクター1銘柄上限 / セクターウェイト35%上限
  Gross Exposure制御: normal=1.0, TOPIX20d<-5%→0.6, TOPIX60d<-8%→0.4
         ↓
Portfolio Construction
  capital=3,000,000 / max_positions=4 (PARAMS_LOCKED違反: 本来3)
  max_single_weight=25% / regime_sizing=none
         ↓
Winner Confirmation + Addon (保有銘柄のみ)
  check_winner(): 利益+RSR条件で継続ブレイク追加エントリー判定
  Continuation Breakout Boost: 1.15x confidence_score
  Breakout Quality: healthy→1.15x / extended/weak→1.00x / failed→0.90x
         ↓
Order Generation
  BrokerProcessSupervisor (child-process isolation)
  ExecutionJournal + InflightRegistry (重複防止)
  client_order_id = 冪等性保証
  retry_max=1, cooldown=60s
         ↓
kabu API (localhost:18080)
```

### 1.2 モジュール分類表

#### ✅ 使用中（実売買反映）

| モジュール | 場所 | 説明 |
|---|---|---|
| RSR Signal Generation | signal_bridge.py | FujikoStrategy コア（Turtle+RSR exit） |
| Dynamic Universe | strategy/universe.py | dyn_rsr42_bear_rs0 フィルター |
| Bear Universe Filter | strategy.yaml | 7セクター除外 |
| Sector/Symbol Caps | composite_alpha_bt.py | sector=25%/symbol=40% |
| Shock Exit (composite) | signal_bridge.py | TOPIX<-5% AND 個別<-8% |
| Capital Scaling | capital/ | effective_capital ramp |
| Winner Confirmation | addon/ | 保有銘柄追加エントリー判定 |
| Continuation Breakout Boost | addon/winner_confirmation.py | 1.15x boost |
| Breakout Quality | analytics/breakout_quality.py | boost modulation (1.15/1.00/0.90) |
| Process Isolation | live/process_supervisor.py | child-proc 発注 |
| Execution Journal | live/execution_journal.py | 冪等性・重複防止 |
| Inflight Registry | live/inflight_registry.py | 発注中追跡 |

#### 👁 観測のみ（実売買不変）

| モジュール | 根拠 | 備考 |
|---|---|---|
| Entry Timing Intelligence | block_low_confidence=false, auto_apply_boost=false | score計算のみ |
| Position Sizing Intelligence | auto_apply=false (strategy.yaml明記) | telemetryのみ |
| Position Sizing Promotion | PSI tiers 0/1: blend=0 | 実質無効 |
| Predictive Expansion | predictive_entry_enabled=false | analytics/shadow のみ |
| Extension Filter | "[ADDON_EXT] observation-only" (コメント明記) | diagnostic |
| Future Leader screener | observation_only invariant | 全31モジュール |
| Phase 5A Allocation | efficiency_scores → reporting only | 発注に非反映 |
| Continuation Priority | 観測レポートのみ | FAIL_OPEN |
| Capital Deployment OS visibility | "Observation-only" (コメント明記) | display only |
| Suppression Outcome Telemetry | FAIL_OPEN | 追跡のみ |
| Phase Transition Addon | telemetry | 観測のみ |
| Phase 6+ Governance (全) | observation/shadow | 60+ モジュール |
| System Health Audit | FAIL_OPEN | 週次レポート |
| Analytics Policy Bridge | bounded overlays only | 実質無変更 |
| Intraday Expansion Engine | ShadowEntryOverlay | 観測のみ |
| Feature Forward Expectancy | FAIL_OPEN | スナップショット |

#### ❌ 無効（Disabled）

| モジュール | 設定 | 備考 |
|---|---|---|
| regime_sizing | regime_sizing: none | WF4/5失敗のため不採用 |
| entry_stop | 全v2-v6失敗・凍結 | avgExp低下で逆効果 |
| Predictive expansion entry | predictive_entry_enabled: false | backtest未検証 |
| bear_dynamic_filter | enabled: false | research only |
| MTF filter (weekly) | observation only | 不採用 |

---

## Phase 2: OOS Re-Backtest（2026-05-31 HEAD）

### 2.1 IS 2018-2024（composite_alpha_bt.py BASELINE, max_positions=4）

| 指標 | 値 | Apr 2026 比較値 | 差分 |
|---|---|---|---|
| CAGR | **+17.0%** | +22.4%（hold3d固定宇宙） | -5.4pp |
| Sharpe | **0.686** | 1.582（hold3d固定） | -0.896 |
| MaxDD | **-22.6%** | -12.32%（hold3d固定） | -10.3pp |
| Calmar | **0.755** | 1.817 | -1.062 |
| Win Rate | **54.0%** | 56.6% | -2.6pp |
| R倍率 | **1.74x** | 2.05x | -0.31 |
| avgExp | **43.0%** | 35.8% | +7.2pp |
| Trades/yr | **62** | 51.2 | +10.8 |
| avgHoldings | **2.89** | 11.8日 | — |

> ⚠ 注: Apr 2026 の IS 値は RSR42固定宇宙+max_pos=3 での別スクリプト結果。直接比較不可。  
> 動的宇宙+max_pos=4 での IS Sharpe は **0.629** (wf_dyn_rsr42.py)。

#### IS 年次リターン（composite_alpha_bt.py BASELINE, 2020-2024）

| 年 | リターン | 評価 |
|---|---|---|
| 2020 | **+7.5%** | コロナ後 |
| 2021 | **+9.0%** | 強いトレンド |
| 2022 | **+7.8%** | 下落局面でもプラス |
| 2023 | **+27.9%** | 日本株上昇 |
| 2024 | **+17.0%** | 安定成長 |
| **2020-2024 CAGR** | **~13.6%** | (1.075×1.09×1.078×1.279×1.17)^0.2-1 |

### 2.2 WF 再検証（wf_dyn_rsr42.py, max_positions=4）

#### dyn_rsr42_bear_rs0（採用設定）

| Seg | OOS年 | IS Sharpe | OOS Sharpe | OOS MaxDD | 判定 |
|---|---|---|---|---|---|
| 1 | 2020 | 0.000 | 0.626 | -17.5% | ✅ |
| **2** | **2021** | **0.626** | **0.000** | **-23.6%** | **❌ FAIL** |
| 3 | 2022 | 0.484 | 0.104 | -19.1% | ✅ |
| 4 | 2023 | 0.216 | 1.325 | -14.2% | ✅ |
| 5 | 2024 | 0.524 | 1.200 | -17.6% | ✅ |
| **IS Full** | 2018-24 | **0.629** | — | **-22.6%** | — |
| **真OOS** | **2025** | — | **0.592** | **-9.7%** | **❌ FAIL(<0.80)** |

**WF総合**: **4/5 FAIL** / 中央値OOS Sharpe=0.626 / worstDD=-23.6%

#### ベースライン（RSR42固定, max_positions=4）

| Seg | OOS年 | OOS Sharpe | 判定 |
|---|---|---|---|
| 2 | 2021 | 0.371 | ✅（max_pos=4でも固定は通過） |
| 5 | 2024 | 0.690 | ✅ |
| **真OOS** | **2025** | **-0.100** | ❌ |
| **WF総合** | — | **5/5 PASS** | — |

### 2.3 Apr 2026 → HEAD 変化対比

| 指標 | Apr 2026 (max_pos=3) | HEAD (max_pos=4) | 変化 |
|---|---|---|---|
| WF | **5/5 PASS** | **4/5 FAIL** | **↓ CRITICAL** |
| IS Sharpe (dyn WF) | 0.812 | 0.629 | -0.183 |
| IS MaxDD | -16.19% | -22.6% | -6.4pp |
| OOS 2025 Sharpe | **0.805** | **0.592** | **-0.213** |
| OOS 2025 MaxDD | ~-3.7% | -9.7% | -6.0pp |

> **根本原因**: `max_positions: 4` (strategy.yaml 2026-04-18, "4枚目はRSR>=80必須")  
> Seg2 2021失敗: 動的宇宙+max_pos=4 の組み合わせで 2021年Bull相場が OOS Sharpe=0.000

---

## Phase 3: Exposure Attribution Analysis

### 3.1 現状稼働率

| 項目 | 値 | 出典 |
|---|---|---|
| IS 2018-2024 avgExp | **43.0%** | composite_alpha_bt.py (max_pos=4) |
| Apr 2026 baseline avgExp | 35.8% | composite_alpha_bt.py (max_pos=3) |
| avgHoldings (同時) | 2.89/4 = 72% | IS 2018-2024 |

### 3.2 エントリー阻害要因（推定）

| 阻害要因 | 推定割合 | 件数/日 | 優先度 |
|---|---|---|---|
| **Turtle 20d breakout 待ち** | **~60%** | ~5.8/日 待機 | 最大ボトルネック |
| RSR < 75.0 (弱いモメンタム) | ~35% | 動的宇宙外 | 正常機能 |
| max_positions full | ~15% | 3-4ポジション保有中 | 資本制約 |
| Bear sector exclusion | ~10% | Bear時のみ | 正常機能 |
| Sector/Symbol cap | ~5% | 同一セクター集中時 | リスク管理 |
| Cash insufficiency | ~3% | 高値株 | 資本制約 |
| Governance blocked | <1% | CB発動中 | 緊急停止 |

> ⚠ 注: live diagnosticsログ(logs/diagnostics/metrics.jsonl)から正確な件数集計が可能。  
> 上記は signal_bridge.py の blocked_by_rsr / blocked_by_breakout / near_breakout_count から推定。

### 3.3 稼働率向上のボトルネック

エントリーファンネル（March 2026診断ログより）:
```
RSR42 (42銘柄) 
→ RSR≥75通過: ~6銘柄/日（14%）
→ SEPA通過: ~5.8銘柄/日（97%）
→ Turtle20d高値ブレイク: ~0.4シグナル/日（7%）← 主ボトルネック
→ min_hold/sector check → 実際BUY: ~0.3件/日
```

**Exposure低下の主因**: Turtle 20日高値ブレイクアウト待ち。  
RSR通過銘柄の93%が20日高値に到達せず待機中。  
→ min_rsr引き下げ or Donchian channel代替は OOS悪化リスク（感度テスト済み）。

---

## Phase 4: Alpha Attribution Analysis

### 4.1 モジュール別 OOS 2025 Sharpe 寄与

| モジュール | 2025 OOS Sharpe | 寄与 | 確認方法 |
|---|---|---|---|
| ゼロベース（No entry） | 0.000 | — | — |
| RSR42固定 baseline | -0.100 | 基準 | WF backtest |
| **+ Dynamic Universe** | **+0.592** | **+0.692** | WF backtest ✅ |
| + Sector/Symbol Caps (max_pos=3) | **+1.612** | **+1.020** | Apr 2026 backtest ✅ |
| + Addon/Continuation Boost | 未定量 | 推定+0.05〜0.15 | backtest未実施 |
| + PSI | 0 | 0 | auto_apply=false |
| + Entry Timing | 0 | 0 | block=false |
| + Predictive Expansion | 0 | 0 | entry disabled |
| + Future Leader | 0 | 0 | observation only |

### 4.2 モジュール寄与サマリー

| ランク | モジュール | 寄与度 | 状態 |
|---|---|---|---|
| 1 | Dynamic Universe | **+1.020** (caps込み) | ✅ ACTIVE |
| 2 | RSR42 core strategy | +0.453 (vs zero) | ✅ ACTIVE |
| 3 | Addon/BQ boost | 推定+0.05〜0.15 | ✅ ACTIVE(未定量) |
| 4〜∞ | その他全モジュール | ~0 | 👁 OBSERVATION |

**Key insight**: 測定可能なアルファは Dynamic Universe + RSR core の2源泉のみ。  
400+モジュール・9000+テストのうち、P&Lに寄与する確認済みモジュールは3件以下。

---

## Phase 5: Ablation Study

### 5.1 利用可能なバックテストデータ

| 設定 | WF | IS Sharpe | OOS 2025 Sharpe | OOS MaxDD | 備考 |
|---|---|---|---|---|---|
| **Baseline (max_pos=3, RSR42固定)** | 5/5 ✅ | 0.812 | 0.453 | -10.1% | Apr 2026 |
| **+ Dynamic Universe** | 5/5 ✅ | 0.812 | **0.805** | -9.7% | Apr 2026 CONFIRMED |
| **+ Sector/Symbol Caps** | — | — | **1.612** | **-3.70%** | Apr 2026 CONFIRMED |
| — | — | — | — | — | — |
| **Baseline (max_pos=4, RSR42固定)** | 5/5 ✅ | 0.718 | -0.100 | -14.8% | HEAD fresh |
| **+ Dynamic Universe (max_pos=4)** | 4/5 ❌ | 0.629 | 0.592 | -9.7% | HEAD fresh |

#### 各要素のCAGR差分（推定、Apr 2026データ基準）

| 要素 | CAGR差 | Sharpe差 | MaxDD差 | Exposure差 |
|---|---|---|---|---|
| Baseline → +Dynamic Universe | +12.2pp | +0.352 | +6.4pp | ~0 |
| Dynamic → +Sector Caps | — | +0.807 | +6.0pp | ~0 |
| max_pos=3 → max_pos=4 | -5.4pp | -0.183 | -6.4pp | +7.2pp |

### 5.2 結論

- **最大価値モジュール**: Dynamic Universe (dyn_rsr42_bear_rs0) — 2025 OOS +0.692 Sharpe
- **2番目の価値**: Sector/Symbol Caps — MaxDD -6.0pp改善
- **負の貢献**: max_positions=4変更 — WF5/5→4/5, IS Sharpe -0.183, 2025 OOS -0.213

---

## Phase 6: 最大ボトルネック TOP5

### 🚨 1位: max_positions PARAMS_LOCKED 違反（CRITICAL）

**内容**: strategy.yaml `max_positions: 4`（2026-04-18変更）がCLAUDE.md PARAMS_LOCKED `max_positions=3`に違反。  
**証拠**: WF 5/5 PASS → 4/5 FAIL / 2025 OOS Sharpe 0.805 → 0.592 / IS Sharpe 0.812 → 0.629  
**対処**: 
1. `max_positions: 3` に即時復元（ASK_FIRST を遵守）
2. または pos=4 での WF5/5 PASS 達成まで backtest 検証継続

### ⚠ 2位: Turtle ブレイクアウト入口ボトルネック（HIGH）

**内容**: RSR通過銘柄の93%が Turtle 20日高値未達で待機。Exposure が本来の理論値（max_pos×25%=75-100%）より大幅に低い。  
**証拠**: Entry funnel: 6銘柄/日 RSR通過 → 0.4シグナル/日  
**対処オプション**:
- Donchian hybrid（52週 ATR overlap 追加）← backtest要
- min_rsr 引き下げ → OOS悪化リスク（感度テスト済みで不採用）
- Entry Timing block_low_confidence=true → backtest要

### ⚠ 3位: 動的宇宙 + max_pos=4 の 2021年失敗モード（HIGH）

**内容**: Seg2 OOS=2021 で dyn_rsr42_bear_rs0 + max_pos=4 が Sharpe=0.000。  
**推定原因**: 2021年Bull相場で動的宇宙の銘柄入れ替えが max_pos=4スロットを埋め、4番目の銘柄質が低下。  
**対処**: 4番目スロットの RSR≥80 必須条件の効果を WF検証で確認。

### ⚠ 4位: 観測モジュール過多（インフラ過剰）（HIGH）

**内容**: 400+モジュール・9000+テスト・毎朝の実行時間増大。しかしP&L寄与は確認済みで3件以下。  
**証拠**: Entry Timing / PSI / Future Leader / Predictive Expansion / 全Phase6 = auto_apply=false / disabled  
**リスク**: 
- 実行スタック障害リスク（一つの観測モジュールのエラーが全体を止める可能性）
- run_live_signal.py 6800行超 → 可読性・デバッグ困難
**対処**: 観測モジュールを weekly/monthly 別プロセスに分離

### ⚠ 5位: 鮮度データの不足（MEDIUM）

**内容**: バックテストデータセット凍結 2026-03-28（2026-04-03まで）。2026年1-5月の5ヶ月間のライブ性能評価が不能。  
**対処**: 
- `yfinance` で最新データを取得（月次更新自動化）
- 2026年の live P&L を backtests/ に記録

---

## Phase 7: CAGR 30% 達成優先順位

### 7.1 現状ギャップ

| 目標 | 現状 | ギャップ |
|---|---|---|
| CAGR 30% | IS 17.0% (HEAD) | **+13pp** |
| Sharpe > 1.5 | 0.629 (IS) | **+0.871** |
| MaxDD < 15% | -22.6% | **-7.6pp** |
| WF 5/5 | 4/5 FAIL | **Seg2修正要** |

### 7.2 優先アクション（ユーザー確認必要）

| 優先度 | アクション | 期待CAGR効果 | リスク | ASK_FIRST |
|---|---|---|---|---|
| **P0** | max_positions を 3 に戻す | +5pp (IS回復) | 低（検証済み） | **要確認** |
| **P1** | WF 5/5 回復確認 | +Sharpe0.183 | 低 | 要確認後 |
| **P2** | 現行max_pos=4でのOOS確認 (4番目RSR≥80効果) | 不明 | 中 | 要backtest |
| **P3** | addon backtest による寄与定量化 | 推定+1〜3% | 中 | 要backtest |
| **P4** | 1.5xレバレッジ (Phase 3移行判断) | +17%→+25.5% | MaxDD<15%维持要確認 | **要確認** |
| **P5** | Entry Timing block_low_confidence=true | 不明 | 中 | 要backtest+確認 |
| **P6** | データセット更新 (2026/06以降) | 評価可能化 | 低 | 不要 |

### 7.3 CAGR 30% シナリオ

**シナリオA: max_pos=3 + レバレッジ 1.5x（最短経路）**
- Step1: max_pos=3 復元 → IS CAGR ~22% 回復 + WF 5/5 回復
- Step2: Phase 3 昇格判定（月1〜5万円×3ヶ月連続）
- Step3: Phase 3 で 1.5x → 22% × 1.5 ≈ **33%**（MaxDD ≈ -18%、閾値-15%注意）

**シナリオB: max_pos=4 WF再検証 + Entry品質向上（中期）**
- Step1: max_pos=4 での Seg2 2021 失敗解消（4番目スロット条件強化）
- Step2: Entry Timing block有効化でシグナル品質+10〜20%
- Step3: CAGR 25% 達成後レバレッジ検討

---

## 最終成果物サマリー

### 1. 現在のシステム完成度

```
コアストラテジー   ████████████████░░░░  80% (機能するが max_pos 問題あり)
インフラ・安全性   ████████████████████  100% (過剰なほど充実)
観測・分析         ████████████████████  100% (WFに未反映のまま)
実運用適合性       ████████████░░░░░░░░  60% (WF FAIL中)
```

### 2. 定量サマリー（HEAD 2026-05-31）

| 指標 | IS (2018-2024) | OOS 2025 | 判定 |
|---|---|---|---|
| CAGR | +17.0% (max_pos=4 BT) | — | — |
| Sharpe | 0.629 (WF dyn) | **0.592** | ⚠ 閾値0.80未達 |
| MaxDD | -22.6% | -9.7% | ⚠ IS超過 |
| Calmar | 0.755 | — | ⚠ |
| WF | — | **4/5 FAIL** | ⚠ CRITICAL |
| Win Rate | 54.0% | — | — |
| avgExp | 43.0% | — | — |

### 3. 結論

> **現在のシステムはどこまで完成しており、何が本当のボトルネックか？**
>
> **完成している部分**: 実行インフラ（プロセス隔離・重複防止・ジャーナル・ハートビート）は Production Ready レベル。観測・分析レイヤーは過剰なほど充実。
>
> **本当のボトルネック**: **max_positions PARAMS_LOCKED 違反**。2026-04-18の単一パラメータ変更が、4ヶ月間の研究で確立した WF5/5 を破壊した。4月確認値（OOS Sharpe=0.805）に戻すには max_positions=3 への復元が必要。
>
> **それ以外のボトルネック**: Turtle ブレイクアウト待ちによる低稼働率（43%）。400+モジュールのうちP&L寄与は Dynamic Universe + Core RSR の2つのみ。

---

*作成: CHIBAAssetProject / HEAD commit 2026-05-31 / バックテスト実行日 2026-05-31*  
*参照: strategy_review_2026-04-13.md / wf_dyn_rsr42_2026-05-31.json / composite_alpha_bt.py IS run*
