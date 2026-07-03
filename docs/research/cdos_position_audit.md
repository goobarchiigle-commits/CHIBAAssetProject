# CDOS dynamic_max_positions 実効値監査

**日付**: 2026-06-12  
**データ源**: logs/scheduler/*_live.log / logs/equity_snapshots.jsonl / reports/conditional_4th_position_audit.md  
**対象期間**: 2026-05-08〜2026-06-11（本番稼働全期間）

---

## 1. CDOS フォーミュラ確認

```python
# src/live/capital_deployment_os.py:158
MAX_POSITIONS_HARD_CAP = 5
MIN_POSITIONS_FLOOR    = 2

def dynamic_max_positions(deployable_capital: float) -> int:
    tier = int(deployable_capital / 1_000_000)
    raw  = tier + 2         # ¥<1M→2, ¥1-2M→3, ¥2-3M→4, ¥3M+→5
    return max(MIN_POSITIONS_FLOOR, min(MAX_POSITIONS_HARD_CAP, raw))
```

**PARAMS_LOCKED = max_positions=3** との矛盾:

| deployable | tier | dyn_max_pos | PARAMS_LOCKED比 |
|---|---|---|---|
| <¥1M | 0 | 2 | **-1 (下回る)** |
| ¥1-2M | 1 | 3 | 一致 |
| ¥2-3M | 2 | 4 | **+1 (超過)** |
| ¥3M+ | 3 | 5 | **+2 (超過)** |

---

## 2. 本番稼働 dyn_max_pos 実測分布

### CAPITAL_DEPLOYMENT ログ直接確認（6観測日）

| 日付 | deployable | dyn_max_pos | PARAMS比 |
|---|---|---|---|
| 2026-06-01 | ¥2,700,000 | **4** | +1 超過 |
| 2026-06-02 | ¥2,700,000 | **4** | +1 超過 |
| 2026-06-03 | ¥3,615,398 | **5** | +2 超過 |
| 2026-06-04 | ¥2,865,878 | **4** | +1 超過 |
| 2026-06-10 | ¥2,961,728 | **4** | +1 超過 |
| 2026-06-11 | ¥3,051,246 | **5** | +2 超過 |

**確認全6日 → dyn=3(PARAMS一致)=0件 / dyn=4(超過)=4件(67%) / dyn=5(超過)=2件(33%)**

### equity_snapshots 推定（37ライブスナップショット）

deploy推定係数: 0.789（実測クロスバリデーション: 0.786〜0.791）

| dyn_max_pos | 推定日数 | 割合 |
|---|---|---|
| 3（PARAMS一致） | 1 | 3% |
| **4（+1超過）** | **26** | **70%** |
| **5（+2超過）** | **10** | **27%** |

**結論: 本番稼働のほぼ全期間(97%)でPARAMS_LOCKED=3を超過。**

---

## 3. 実際の pos4/pos5 発生回数（過去90日）

### 抑制要因: MAX_NEW_POS_PER_DAY=1

dyn_max_pos が 4〜5 を返しても、1日に追加できる新規ポジションは最大1件。  
既存3ポジション + 新規1 = 4枚目は翌日以降に順次発生可能。

### CLUSTER_DETECTED 発生日（ライブ）

| 日付 | 内容 |
|---|---|
| 2026-06-01 | `BUY candidates=4 >= threshold=3 → effective_max_pos=5` |
| 2026-06-02 | `BUY candidates=3 >= threshold=3 → effective_max_pos=5` |

### 実ポジション数推移

equity_snapshots の positions_count:  
- 2026-05-08: 2ポジション（初期）  
- 2026-06-03: CB解除、直後の上昇局面でシグナル集中  
- 2026-06-10〜11: 3ポジション（max_pos=3相当で推移）

**観察期間中に pos4 が実際に約定した記録は trades.jsonl の重複エントリー(テスト実行)のみ。  
本番執行での4枚目建玉は現時点では未確認。**  
ただし CDOS=4〜5 が常時有効であり、シグナル集中日（CLUSTER DETECTED）に拡張を試みた記録が確認されている。

---

## 4. IS 2018-2024 pos4 パフォーマンス分析

（出典: reports/conditional_4th_position_audit.md）

### 4枚目全体統計（IS 2018-2024, 120件）

| 指標 | 値 |
|---|---|
| 件数 | 120 |
| 勝率 | 52.5% |
| avg return | +0.77% |
| median return | +0.33% |
| Profit Factor | **1.402** |
| 総PnL | ¥+606,310 |

### Variant比較（max_pos=3 vs max_pos=4）

| バリアント | IS CAGR | IS Sharpe | MaxDD | Calmar | OOS CAGR |
|---|---|---|---|---|---|
| A: max_pos=3（基準） | +18.1% | 0.779 | -16.7% | 1.087 | +10.0% |
| B: max_pos=4 フィルタなし | +16.8% | 0.667 | -21.4% | 0.784 | +5.9% |
| C: max_pos=4 + RSR≥80 | +19.8% | 0.823 | -18.2% | 1.088 | +8.2% |
| D: max_pos=4 + RSR≥85 | +20.1% | 0.837 | -19.1% | 1.058 | +9.2% |

**Δ vs A:**

| バリアント | ΔCAGR | ΔSharpe | ΔMaxDD |
|---|---|---|---|
| B（無制限4枚目） | **-1.3pp** | -0.112 | **-4.8pp** |
| C（RSR≥80のみ） | **+1.6pp** | +0.044 | -1.5pp |
| D（RSR≥85のみ） | **+2.0pp** | +0.058 | -2.4pp |

### pos1-3 vs pos4 年次比較（最悪年: 2021）

| グループ | N | 勝率 | avg ret | PF |
|---|---|---|---|---|
| 1-3枚目 2021 | 38 | **60.5%** | **+2.67%** | **1.903** |
| 4枚目 2021 | 38 | **36.8%** | **-0.99%** | **0.676** |
| 4枚目 全期間 | 120 | 52.5% | +0.77% | 1.402 |

2021年 4枚目 Top損失:

| シンボル | セクター | PnL |
|---|---|---|
| 9104.T | 海運 | ¥-104,000 |
| 8015.T | 商社 | ¥-87,333 |
| 7013.T | 機械 | ¥-65,957 |
| 6857.T | 電機精密 | ¥-53,250 |

### RSR tier別 4枚目 PF

| RSR Tier | N | 勝率 | PF |
|---|---|---|---|
| 75-80 | 24 | 70.8% | 1.555 |
| 80-85 | 24 | 37.5% | 1.150 |
| 85-90 | 20 | 45.0% | 1.837 |
| 90+ | 20 | 55.0% | 1.305 |

---

## 5. CAGR/PF/WR 比較サマリー（pos1-3 vs pos4-5）

| 区分 | N | WR | avg ret | PF | IS CAGR影響 |
|---|---|---|---|---|---|
| pos1-3 (Variant A) | — | — | — | — | +18.1% |
| pos4 全期間 | 120 | 52.5% | +0.77% | 1.402 | — |
| pos4 無制限 (Variant B) | — | — | — | — | +16.8% (−1.3pp) |
| pos4 RSR≥80 (Variant C) | 95block | — | — | — | +19.8% (+1.6pp) |
| pos4 RSR≥85 (Variant D) | 139block | — | — | — | +20.1% (+2.0pp) |
| pos4 2021悪化 | 38 | 36.8% | -0.99% | 0.676 | 年次 −2.8pp (vs pos1-3) |

---

## 6. 結論と推奨

### 現状の問題

1. **CDOS は PARAMS_LOCKED=3 を実行時上書きしている**（本番全期間の97%でdyn=4〜5）
2. **無制限pos4は有害**（IS CAGR -1.3pp, MaxDD -4.8pp, 2021年特に破滅的 PF=0.676）
3. MAX_NEW=1 の抑制により現在は4枚目に到達していないが、シグナル集中時(CLUSTER DETECTED)にCDOSが5枚まで拡張を試みた記録あり

### 推奨アクション（ASK_FIRST 必須）

| 優先度 | アクション | 根拠 |
|---|---|---|
| **HIGH** | CDOS を max_positions=3 にクランプ | PARAMS_LOCKED遵守。最もクリーンな修正 |
| **MEDIUM** | 4枚目に RSR≥80 フィルタ追加（Variant C） | IS +1.6pp、OOS -1.8pp（OOS劣化あり） |
| **LOW** | CDOS 廃止・固定max_pos=3 | シンプルさ最優先、将来拡張性なし |

**注意**: Variant C/D の OOS 成績は基準(A)を下回る（OOS: A=+10.0%, C=+8.2%, D=+9.2%）。  
WF未実施のため pos4 品質フィルタのロバスト性は未検証。4枚目有効化は別途 WF 5-fold 必須。

---

*生成: 2026-06-12 / tools: equity_snapshots.jsonl + scheduler logs + conditional_4th_position_audit.md*
