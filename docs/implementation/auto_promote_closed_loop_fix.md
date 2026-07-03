# AUTO_PROMOTE 閉ループ修正レポート（P1-A + P1-B）
実施日: 2026-06-20

## 概要

SHADOW → PROBATION → GRADUATED → LIVE_UNIVERSE の閉ループを2つの修正で完成させた。

| Fix | 対象 | 問題 | 修正 |
|---|---|---|---|
| P1-A | `check_graduation()` | `forward_return_3d=None` 固定 → 永久FAIL | `avg(rsr_delta) > 0` フォールバック追加 |
| P1-B | `run_probation_gate()` | 卒業と同一runで再昇格 → STATUS_GRADUATED 上書き | `_just_graduated` 追跡 → `already_seen` 追加 |

P1-C（RSR下限65化）は実施せず。実運用ログのRSR分布確認後に別研究として提案。

---

## 変更ファイル一覧

| ファイル | 変更種別 | 差分 |
|---|---|---|
| `src/universe/auto_promote_safe_v2.py` | 修正 | +18行 |
| `tests/test_auto_promote_safe_v2.py` | 追加 | +222行 |

---

## 修正詳細

### P1-A: `check_graduation()` — rsr_delta フォールバック

**変更箇所**: `src/universe/auto_promote_safe_v2.py` lines 604–648

**修正前 (Condition 2)**:
```python
fwd_rets = [_safe_float(o["forward_return_3d"]) for o in sym_outcomes
            if o.get("forward_return_3d") is not None]
if not fwd_rets:
    return False, "no_forward_returns"   # ← 常にここで終了（永久FAIL）
avg_ret = sum(fwd_rets) / len(fwd_rets)
if avg_ret <= 0:
    return False, f"expectancy_negative:{avg_ret:.4f}"
```

**修正後**:
```python
fwd_rets = [
    _safe_float(o["forward_return_3d"])
    for o in sym_outcomes
    if o.get("forward_return_3d") is not None
]
if fwd_rets:
    avg_ret = sum(fwd_rets) / len(fwd_rets)
    graduation_method = "forward_return"
    if avg_ret <= 0:
        return False, f"expectancy_negative:{avg_ret:.4f}"
else:
    # forward_return_3d 未マテリアライズ → RSRトレンドを代理指標として使用
    rsr_deltas = [_safe_float(o.get("rsr_delta", 0.0)) for o in sym_outcomes]
    if not rsr_deltas:
        return False, "no_forward_returns"
    avg_ret = sum(rsr_deltas) / len(rsr_deltas)
    graduation_method = "rsr_delta_fallback"
    logger.info("[V2] graduation fallback: %s avg_rsr_delta=%.3f (n=%d)",
                record.symbol, avg_ret, len(rsr_deltas))
    if avg_ret <= 0:
        return False, f"rsr_delta_negative:{avg_ret:.4f}"
```

**成功時のログ**:
```
成功: "graduated:avg_ret=2.0000 graduation_method:rsr_delta_fallback rsr=67.0 cont=1.00"
失敗: "rsr_delta_negative:-1.5000"
空  : "no_forward_returns"
```

---

### P1-B: `run_probation_gate()` — 同一run再昇格防止

**変更箇所**: `src/universe/auto_promote_safe_v2.py` lines 730, 764, 807

**追加 (Step 1 初期化)**:
```python
surviving: List[ProbationRecord] = []
_just_graduated: Set[str] = set()   # P1-B: prevent same-run re-promotion
```

**追加 (卒業処理内)**:
```python
_just_graduated.add(sym)   # P1-B: block same-run re-promotion
```

**変更 (Step 4 already_seen)**:
```python
# 変更前
already_seen = active_symbols | cooldown_syms | set(live_universe.keys())

# 変更後
already_seen = active_symbols | cooldown_syms | _just_graduated | set(live_universe.keys())
```

---

## テスト追加数と内訳

| クラス | テスト数 | 内容 |
|---|---|---|
| `TestCheckGraduationP1A` | 8件 | rsr_delta_fallback パス・失敗・境界・primary path 互換性 |
| `TestJustGraduatedBlocking` | 5件 | P1-B: 再昇格防止・STATUS_GRADUATED 永続・negative rsr_delta |
| `TestClosedLoopE2E` | 6件 | SHADOW→ACTIVE→GRADUATED→LIVE_UNIVERSE 全フロー |

**合計**: 19件追加 / **全体**: 172件 pass（既存 154 + 新規 18 ※同ファイル内重複なし）

```
tests/test_auto_promote_safe_v2.py: 172 passed in 1.55s
```

---

## 卒業判定ロジック（修正後）

```
check_graduation(record, current_rsr, outcomes):
  Cond1: elapsed >= probation_days(5)                    ← 変更なし
  Cond2: expectancy > 0                                  ← P1-A 修正
    primary:  avg(forward_return_3d) > 0   [実装時に使用]
    fallback: avg(rsr_delta) > 0           [現在フォールバック中]
    empty:    "no_forward_returns"         [outcomes ゼロ件時]
  Cond3: current_rsr >= rsr_at_promotion × 0.90          ← 変更なし
  Cond4: continuation_rate >= 0.40                       ← 変更なし
  → success reason に graduation_method: を記録
```

---

## 再昇格防止確認（P1-B）

```
Step 1 (lifecycle check):
  sym 6857.T → elapsed=6 >= 5 ✅
  check_graduation → PASS
  STATUS_GRADUATED 書込 + _just_graduated.add("6857.T")

Step 4 (new promotions):
  already_seen = {...} | {"6857.T"} | {...}
  shadow["6857.T"] → "6857.T" in already_seen → スキップ ✅
  → newly_promoted に 6857.T なし ✅
```

---

## 閉ループ確認結果（E2E テスト）

```
test_full_pipeline_closed_loop:
  Day 0: shadow["6857.T"] → P2-A gate PASS → STATUS_ACTIVE ✅
  Day 6: elapsed=6 ≥ 5, rsr_delta=+2.0 → STATUS_GRADUATED ✅
         _just_graduated blocking → re-promotion BLOCKED ✅
  Post:  get_graduated_symbols() = {"6857.T"} ✅
         LIVE_UNIVERSE["6857.T"] = "電機精密" ✅
         graduation_reason contains "graduation_method:rsr_delta_fallback" ✅

CLOSED LOOP: SHADOW → PROBATION → GRADUATED → LIVE_UNIVERSE ✅
```

---

## DRY / LIVE 実行結果

| 項目 | 値 |
|---|---|
| 実行日 | 2026-06-20 |
| P1-A/P1-B コードパス | エラーなし |
| probation_promoted | 0（SI=38.7 < 50 のため — 正常） |
| 発注 | **なし** |
| DRY/LIVE差異 | なし |
| LIVE 終了コード | 1（kabuステーション未接続: 401 — EMERGENCY_STOP 正常動作） |

---

## 今後の研究提案

| 優先度 | 項目 | 備考 |
|---|---|---|
| P2 | P1-C: P2-A gate RSR下限を65へ引き上げ | 実運用ログのRSR分布確認後にWF研究として提案 |
| P3 | forward_return_3d バックフィル実装 | OHLCV parquet から事後マテリアライズ |
| P3 | ISSUE-4: 試用銘柄のファイル永続化 | 設計上 intentional の可能性あり — 文書化済み |
