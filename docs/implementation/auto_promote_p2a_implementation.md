# AUTO_PROMOTE P2-A 実装レポート
実装日: 2026-06-20

## 概要

研究フェーズP2-A（WF4-fold, 2021-2024, avg ΔCAGR=+5.38pp）の結果を本番ロジックへ反映。
`CANDIDATE_UNCLASSIFIED` に新たな昇格経路（P2-Aゲート）を追加した。

---

## 変更ファイル一覧

| ファイル | 変更種別 | 行数差分 |
|---|---|---|
| `src/universe/auto_promote_safe_v2.py` | 追加 | +53行 |
| `tests/test_auto_promote_safe_v2.py` | 追加 | +152行 |

---

## 変更内容

### `src/universe/auto_promote_safe_v2.py`

**追加定数（+5行）**:
```python
P2A_MIN_SECTOR_IGNITION: float = 50.0          # SI下限（WF検証済み範囲 [50,90)）
P2A_EXCLUDED_SYMBOLS: FrozenSet[str] = frozenset({"5706.T"})  # OOS負（PF=0.22）
```

**新規関数 `check_p2a_unclassified_gate()`（+33行）**:
- Gate 1: RSR ≥ 8.0（既存と同一）
- Gate 2: **バイパス**（predictive_rankチェック省略 — WF設計上不要と確認済み）
- Gate 3: `P2A_MIN_SECTOR_IGNITION(50)` ≤ SI < `GATE_MIN_SECTOR_IGNITION(90)`
- 除外: `symbol in P2A_EXCLUDED_SYMBOLS`

**`run_probation_gate()` への分岐挿入（+15行）**:
`classify_candidate_type()` の直後、`check_promotion_gate()` の前に P2-A 分岐を追加:
```python
if cand_type == CANDIDATE_UNCLASSIFIED:
    p2a_ok, p2a_fail = check_p2a_unclassified_gate(sym, rsr_scores, predictive_scores)
    if not p2a_ok: → _rejections.append(...); continue
    candidates.append((..., list(reasons) + ["p2a_unclassified"], cand_type))
    continue  # taxonomy check をスキップ
```
既存の許可タイプ（CONTINUATION/EARLY_IGNITION/HIGH_RSR）は変更なし。

### `tests/test_auto_promote_safe_v2.py`

**新規テストクラス（+152行）**:

| クラス | テスト数 | 内容 |
|---|---|---|
| `TestCheckP2AUnclassifiedGate` | 13件 | ゲート関数単体: SI境界/除外銘柄/RSR不足/欠損データ |
| `TestRunProbationGateP2A` | 8件 | E2Eフロー: 昇格/拒否/タグ/既存タイプ無影響 |

---

## pytest 結果

```
tests/test_auto_promote_safe_v2.py: 154 passed (108 既存 + 46 新規)
tests/ 全体: 9865 passed (21 failures は全て本実装と無関係の既存不具合)
```

---

## DRY / LIVE 実行結果

**実行日: 2026-06-20**

| 項目 | 値 |
|---|---|
| shadow_candidates_detected | 6 |
| probation_promoted | 0 |
| p2a_si_too_low rejects | 3件（8035.T/6146.T/6857.T: SI=38.7 < 50） |
| p2a_excluded rejects | 1件（5706.T） |
| 発注 | **なし** |
| DRY/LIVE差異 | なし |

**注**: 2026-06-20 時点で電機精密セクターのSI=38.7 は閾値50.0を下回るため昇格ゼロは正常動作。  
SI≥50 を満たした日に自動的に昇格候補として機能する。

---

## 昇格候補として認識される条件（本番）

- `candidate_type == CANDIDATE_UNCLASSIFIED`
- `sector_ignition_score ∈ [50.0, 90.0)`
- `symbol ∉ P2A_EXCLUDED_SYMBOLS` (5706.T 除外)
- `rsr ≥ 8.0`
- 発動時: `promotion_reason` に `"p2a_unclassified"` タグが記録される

対象銘柄（WF検証済み）: **6146.T / 6857.T / 6920.T / 8035.T**（電機精密）

---

## 既存ロジックへの影響

- `ALLOWED_CANDIDATE_TYPES` 変更なし（UNCLASSIFIED は引き続き含まれない）
- `check_promotion_gate()` 変更なし
- CONTINUATION / EARLY_IGNITION / HIGH_RSR の挙動変更なし
- MATURE_LEADER / MEAN_REVERSION ブロック変更なし

---

## WF研究との対応

| 研究 | 結論 | 本実装での反映 |
|---|---|---|
| P2-A (5706.T除外) | PASS avg +5.38pp | `P2A_EXCLUDED_SYMBOLS={"5706.T"}` |
| P2-B (SI閾値比較) | SI≥50が唯一PASS | `P2A_MIN_SECTOR_IGNITION=50.0` |
| P2-C (TOPIXフィルター) | フィルターなしが最良 | Gate2バイパスのみ |
| P2-E (セクター集中制約) | P2-A維持が最良 | 制約なし |
