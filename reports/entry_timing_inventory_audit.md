# Entry Timing Inventory Audit

作成日: 2026-06-02  |  調査対象: src/ 全体 + strategy.yaml + run_live_signal.py

---

## 1. 実装一覧

| # | 機能名 | ファイル | ON/OFF設定 | デフォルト値 | 状態 |
|---|---|---|---|---|---|
| 1 | **Entry Timing Engine** | `src/entry/entry_timing_engine.py` | `entry_timing.enabled` | `true` | PROD稼働 |
| 2 | **apply_entry_timing_boost** | `src/entry/entry_timing_engine.py` | `entry_timing.enabled` | boost_weight=0.06 | PROD稼働 |
| 3 | **block_low_confidence** | `strategy.yaml` (設定のみ) | `entry_timing.block_low_confidence` | `false` | **未実装** |
| 4 | **Entry Timing Promotion** | `src/entry/entry_timing_promotion.py` | `entry_timing.auto_apply_boost` | `false` | DRYのみ |
| 5 | **auto_apply_boost** (Tier昇格) | `src/kabusapi/signal_bridge.py` L2173-2188 | `entry_timing.auto_apply_boost` | `false` | 実行経路あり・未有効 |
| 6 | **Position Sizing Intelligence** | `src/portfolio/position_sizing_intelligence.py` | `position_sizing.enabled` | `true` | 観測専用 |
| 7 | **PSI auto_apply** | `src/portfolio/position_sizing_promotion.py` | `position_sizing.auto_apply` | `false` | **未有効** |
| 8 | **Breakout Quality** | `src/analytics/breakout_quality.py` | (addon経由) | — | LIVE(addon限定) |
| 9 | **Winner Confirmation** | `src/addon/winner_confirmation.py` | (addon経由) | — | LIVE(add-on限定) |

---

## 2. シグナルへの影響マトリクス

| 機能 | BLOCK | BOOST | conviction変更 | RANK変更 | SIZE変更 |
|---|---|---|---|---|---|
| Entry Timing Engine (score計算) | ❌ | ✅ 微小 (+3〜-3pt) | ❌ | ✅ tiebreaker | ❌ |
| **block_low_confidence=true** | **✅ (未実装)** | — | — | — | — |
| auto_apply_boost=true (Tier1-3) | ❌ | ✅ (+5〜+10pt) | ❌ | ✅ tiebreaker強化 | ❌ |
| PSI auto_apply=true | ❌ | ❌ | ✅ conviction_score | ✅ 微小 | ✅ (virtual_weight) |
| Breakout Quality | ❌ | ✅ 1.15x/0.90x | ❌ | ❌ | ❌ (addon確信度のみ) |
| Winner Confirmation | ❌ | ❌ | ❌ | ❌ | ✅ +100株 |

### boost量の比較（score差100点、RSR~80基準）

```
現行 (boost_weight=0.06):
  HIGH(score100) → +3.0pt追加  (RSR80 → 83)
  LOW (score  0) → -3.0pt追加  (RSR80 → 77)
  RSR選択への影響: 微小（RSRが支配的）

auto_apply Tier3 (boost_weight=0.20):
  HIGH(score100) → +10.0pt追加 (RSR80 → 90)
  LOW (score  0) → -10.0pt追加 (RSR80 → 70)
  RSR選択への影響: 中程度（ランク逆転が起きる可能性あり）
```

---

## 3. バックテスト反映状況

| 機能 | バックテスト反映 | 備考 |
|---|---|---|
| Entry Timing Score計算 | **未接続** | fujiko_strategy.py / capital_allocation_abc.py に実装なし |
| apply_entry_timing_boost | **未接続** | バックテストエンジンはRSR+momentum rankのみ |
| block_low_confidence | **未接続** | コード自体が存在しない（コメントのみ） |
| auto_apply_boost | **未接続** | signal_bridge.pyのみ。バックテストには渡されない |
| PSI virtual_weight | **未接続** | バックテストは固定サイジング |
| conditional_4th_audit (entry_timing_score) | **部分接続** | 20d高値比として近似実装（本物のEntryTimingEngineではない） |

> **結論**: 現在のバックテストエンジンはEntry Timing完全未使用。  
> IS CAGR +18.1%はEntry Timing効果ゼロの純粋な RSR + Turtle戦略の結果。

---

## 4. LIVE反映状況

| 機能 | LIVE反映 | 備考 |
|---|---|---|
| Entry Timing Engine (score) | **LIVE接続済み** | signal_bridge.py L2190-2219 で全BUY候補スコアリング |
| apply_entry_timing_boost | **LIVE接続済み** | L2231 ランキングに +[-3,+3]pt追加 (boost_weight=0.06) |
| block_low_confidence=false | DRYのみ(コメント) | false固定のためLIVE実質未動作 |
| auto_apply_boost | **DRYのみ** | false固定。true時はPromotion Tier boost_weight読込 |
| Entry Timing Promotion評価 | **DRY+LIVE** | run_live_signal.py L4487/6657 で評価実行・JSONファイル書込 |
| PSI (virtual_weight) | **LIVE接続済み** | signal_bridge.py L2383-2415 で計算 (telemetryのみ) |
| PSI auto_apply | DRYのみ(コメント) | false固定のためサイジング未変更 |
| Breakout Quality | LIVE(add-on確認内) | winner_confirmation.py 経由で稼働 |

---

## 5. 現在の実行経路（フローチャート）

```
                 ┌─────────────────────────────────────────────────────┐
                 │                 signal_bridge.py                    │
                 │                                                     │
  market data    │                                                     │
  ─────────────► │  [1] 全RSR42銘柄 signal生成                        │
  rsr_df         │       FujikoStrategy / MeanReversionStrategy        │
  strategy.yaml  │       sig=1(BUY), sig=-1(SELL), sig=0(HOLD)        │
                 │              ↓                                     │
                 │  [2] Entry Timing スコアリング ← [A] ★現在ここ     │
                 │       entry_timing_engine.py                       │
                 │       score[0-100], confidence[HIGH/MED/LOW]       │
                 │       action[IMMEDIATE/NORMAL/WATCH]                │
                 │       enabled=true, boost_weight=0.06              │
                 │              ↓                                     │
                 │  [3] ランキング（composite score）                   │
                 │       composite = RSR + RSR_mom*0.3-0.7            │
                 │                 + (score-50)*0.06  ← ET boost     │
                 │       ※ block_low_confidence=false → LOWもランク入  │
                 │              ↓                                     │
                 │  [4] Portfolio Construction                        │
                 │       上位N銘柄（max_positions=3）を選択            │
                 │       ※ ET scoreはtiebreaker、RSRが主             │
                 │              ↓                                     │
                 │  [5] PSI virtual_weight計算 (observation-only)    │
                 │       conviction_score計算 → telemetry記録         │
                 │       実発注サイズは変更しない                       │
                 │              ↓                                     │
                 │  [6] LIVE Execution                               │
                 │       kabuステーション API localhost:18080           │
                 │       発注 (slippage+commission込み)                │
                 └─────────────────────────────────────────────────────┘
                              ↓ 並走
                 ┌─────────────────────────────────────────────────────┐
                 │  run_live_signal.py (DRY/LIVE)                     │
                 │  entry_timing_promotion.py 評価                    │
                 │   └→ Tier状態確認 → JSONL記録                      │
                 │   └→ 将来: auto_apply_boost=true で Tier boost適用  │
                 └─────────────────────────────────────────────────────┘
```

### block_low_confidence=true 有効化後の変更点

```
  [3] ランキング後
         ↓
  [3.5] block_low_confidence フィルタ ← 未実装 (コードなし)
         confidence==LOW → BUY候補から除外
         ※ 実装が必要: signal_bridge.py L2231付近に追加
         ↓
  [4] Portfolio Construction (LOW除外済みの候補のみ)
```

---

## 6. 効果検証可能なスイッチ一覧

| スイッチ | 現在値 | ON/OFF比較バックテスト可能か | 実装状態 | 実装方法 |
|---|---|---|---|---|
| `entry_timing.enabled` | true | ✅ 可能 (false=ET完全無効) | 接続済み(LIVE) | strategy.yaml変更 |
| `entry_timing.boost_weight` | 0.06 | ✅ 可能 (0.06/0.10/0.15/0.20) | 接続済み(LIVE) | strategy.yaml変更 |
| **`entry_timing.block_low_confidence`** | false | **❌ 不可 (コード未実装)** | **signal_bridge実装必要** | **新規コード追加** |
| `entry_timing.auto_apply_boost` | false | ⚠️ LIVE経路のみ (バックテスト未接続) | LIVE経路あり | strategy.yaml変更 |
| `position_sizing.auto_apply` | false | ❌ バックテスト未接続 | LIVE経路のみ | 要実装 |
| バックテストET接続 | 未接続 | ✅ 可能 (実装すれば) | **未実装** | fujiko_strategy.py改修 |

---

## 7. 重要発見: block_low_confidence の実体

### 現状

`strategy.yaml` に設定項目として存在するが、**実行コードが存在しない**。

```yaml
entry_timing:
  block_low_confidence: false   # ← 設定だけあってコードが読まない
```

`signal_bridge.py` L2358:
```python
# Entry Timing Intelligence (observation-only unless block_low_confidence=true)
```
→ コメントで「block_low_confidence=true なら除外」と書いてあるが、**実際のif文が存在しない**。

### 実装すべきコード（signal_bridge.py L2231付近）

```python
# 現行: LOWも含めてランキング
_buy_eligible_all = sorted(
    [(s.rsr + s.rsr_mom * _MOM_WEIGHT_ADJ + _et_adj(s.symbol), s.symbol)
     for s in signals if s.signal == 1 and not s.currently_holding],
    reverse=True,
)

# block_low_confidence=true 後: LOW除外
_et_block_low = bool(_et_cfg.get("block_low_confidence", False))
_buy_eligible_all = sorted(
    [(s.rsr + s.rsr_mom * _MOM_WEIGHT_ADJ + _et_adj(s.symbol), s.symbol)
     for s in signals if s.signal == 1 and not s.currently_holding
     if not (_et_block_low
             and s.symbol in _et_results
             and _et_results[s.symbol].confidence == CONFIDENCE_LOW)],
    reverse=True,
)
```

### バックテスト実装すべきコード（fujiko_strategy.py or capital_allocation_abc.py）

```python
# Entry Timing スコアが低いシグナルをブロックする場合
# (backtest_engine内でEntry Timing計算が必要)
if block_low_confidence and entry_timing_confidence[sym] == "LOW":
    continue   # BUY候補から除外
```

**現在: バックテストにEntry Timingが一切接続されていないため、**
**`block_low_confidence=true` の IS/OOS効果を測定するには、**
**まずバックテストにEntry Timing接続を実装する必要がある。**

---

## 8. Promotion Tier状態

| Tier | boost_weight | 昇格条件 | 現在状態 |
|---|---|---|---|
| Tier 0 (観測専用) | 0.06 | 初期状態 | **現在ここ** |
| Tier 1 (証拠検証済み) | 0.10 | 30d: mono≥0.60, WR_top_decile > bottom+5%, N≥100 | 未昇格 |
| Tier 2 (本番アルファ) | 0.15 | Tier1≥30d + expectancy_uplift≥10% (60d) | 未昇格 |
| Tier 3 (フルアルファ) | 0.20 | Tier2≥60d + uplift≥10% + forward_return_uplift (90d) | 未昇格 |

---

## 9. 次のアクション（優先順位）

| 優先度 | アクション | 必要な実装 | 期待効果 |
|---|---|---|---|
| **1** | signal_bridge.py に `block_low_confidence` 実装 | 5行追加 | LIVE blockが機能するようになる |
| **2** | バックテストエンジンに Entry Timing 接続 | fujiko_strategy.py + capital_allocation_abc.py 改修 | IS/OOS効果測定可能になる |
| **3** | block_low_confidence A/Bバックテスト | #2完了後 | IS CAGR +3〜7pp 検証 |
| **4** | auto_apply_boost=true (Tier0→Tier1昇格後) | Tier1達成後 | boost_weight 0.06→0.10 |
| ✗ | Variant D (max_pos=4) | WF 1/5 → 却下 | — |

---

## サマリー

```
Entry Timing は「計算されているが機能していない」状態

現在:
  - スコア計算: ✅ LIVE稼働 (score 0-100, HIGH/MED/LOW分類)
  - ランク影響: ✅ LIVE稼働 (±3pt、RSRの tiebreaker)
  - ブロック機能: ❌ 未実装 (block_low_confidence=false かつコードなし)
  - バックテスト接続: ❌ 完全未接続
  - Tier昇格: ❌ Tier0固定 (観測データ蓄積中)

「Entry Timing有効化でIS CAGR +3〜7pp」の仮説は
バックテスト接続後でないと検証不可能。

最優先実装:
  1. signal_bridge.py に block_low_confidence 実装 (5行)
  2. backtest engine に Entry Timing 接続 (fujiko_strategy.py 改修)
  3. IS/OOS A/Bバックテスト実行
```
