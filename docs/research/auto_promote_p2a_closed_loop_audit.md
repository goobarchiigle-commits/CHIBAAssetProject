# AUTO_PROMOTE P2-A 閉ループ監査レポート
実施日: 2026-06-20

## 経路図

```
SHADOW_UNIVERSE
    │ run_probation_gate() → P2-A gate
    │   check: RSR≥8, 50≤SI<90, symbol≠5706.T
    ↓
PROBATION (STATUS_ACTIVE) ─────────────────────────────┐
    │ 毎朝: in-memory LIVE_UNIVERSE に追加               │
    │       ファイル未永続化 ← [ISSUE-4]                │
    ↓                                                    │
    LIVE_UNIVERSE (in-memory)                           │
        │ RSR_UNIVERSE_62 に元々含まれる（SHADOW経由）  │
        ↓                                               │
        SignalBridge                                    │
            RSR ranking (min_rsr=75)                    │
            FujikoStrategy (breakout+sepa+momentum)     │
            ↓                                           │
            BUY signal (qty × 0.25 ← 試用割当上限)     │
                                                        │
    5日経過後 → check_graduation()                      │
        Cond1: elapsed ≥ 5d ✅                          │
        Cond2: avg(fwd_ret_3d) > 0                      │
               ← fwd_ret_3d = None 固定 [BLOCKER-1]    │
               → ALWAYS: "no_forward_returns"           │
        ┌ FAIL ←──────────────────────────────── ∞ LOOP┘
        │（永久試用状態: STATUS_ACTIVE が継続）
        │
        │（BLOCKER-1 修正後 → 以下が機能する）
        ↓ PASS
PROBATION → STATUS_GRADUATED (JSONL書込)
    │
    │   同一run Step4 が即 STATUS_ACTIVE 上書き
    │   ← [BLOCKER-2: 再昇格バグ]
    │   → get_graduated_symbols() が空返し → 卒業不発
    │
    │（BLOCKER-2 修正後 → 以下が機能する）
    ↓
get_graduated_symbols() → LIVE_UNIVERSE (in-memory+永続)
    LIVE_UNIVERSE_FILE に atomic write
    ↓
SignalBridge (次回以降)
    RSR ranking (min_rsr=75)
    ↓
    BUY signal (qty × 1.0 ← cap解除)  ← [GOAL]
```

---

## 1. 卒業処理（PROBATION → GRADUATED）

**関数**: `run_probation_gate()` Step 1 → `check_graduation()` → JSONL書込

卒業条件（4条件全通過）:
| 条件 | 実装 | 状態 |
|---|---|---|
| Cond1: elapsed ≥ probation_days(5) | `elapsed >= record.probation_days` | ✅ 正常 |
| Cond2: avg(fwd_ret_3d) > 0 | `fwd_rets = [o["forward_return_3d"] for o if not None]` | ❌ **BLOCKER-1** |
| Cond3: current_rsr ≥ rsr_at_promotion × 0.90 | `current_rsr >= rsr_floor` | ✅ 正常 |
| Cond4: continuation_rate ≥ 0.40 | `cont/total >= 0.40` | ✅ 正常 |

**BLOCKER-1**: `run_probation_outcome_observation()` が常に `forward_return_3d=None` を書き込む（OHLCV 不使用の設計）。バックフィル処理は未実装。

---

## 2. LIVE_UNIVERSE への反映

### 2A. PROBATION(ACTIVE) → LIVE_UNIVERSE（試用期間中）

```python
# run_live_signal.py line 2251-2254
for _ps in _probation_active_symbols:
    if _ps not in LIVE_UNIVERSE and _ps in SHADOW_UNIVERSE:
        LIVE_UNIVERSE[_ps] = SHADOW_UNIVERSE[_ps]   # ← in-memory のみ
```

- ✅ SignalBridge 初期化（line 2392）より前に実行 → tradeable に入る
- ✅ SHADOW_UNIVERSE に含まれる → RSR_UNIVERSE_62 に入る → RSR 計算される
- ⚠️ ファイル未永続化: 翌日 run_probation_gate() FAIL_OPEN 発動時は BUY 対象外

### 2B. GRADUATED → LIVE_UNIVERSE（永続）

```python
# run_live_signal.py line 2267-2303
_grad_syms = get_graduated_symbols(PROBATION_PROMOTIONS_FILE, live_universe=LIVE_UNIVERSE)
LIVE_UNIVERSE[_g_sym] = _g_sec   # in-memory
LIVE_UNIVERSE_FILE.write_text(...)   # 永続化（atomic）
```

- ✅ SignalBridge 初期化（line 2392）より前に実行
- ✅ ファイル永続化 → 翌日以降も継続
- ❌ BLOCKER-2 により実際には到達不可（BLOCKER-1 修正後に顕在化）

---

## 3. tradeable_universe → BUY signal

RSR 計算の鍵: `RSR_UNIVERSE_62 = RSR_UNIVERSE + SHADOW_UNIVERSE`（line 1479）

P2-A 対象銘柄（6146.T/6857.T/6920.T/8035.T）は SHADOW_UNIVERSE に存在 → RSR_UNIVERSE_62 に含まれる。

```python
# signal_bridge.py line 2020-2022
for sym, info in universe_raw.items():
    if sym not in self.universe_tickers:
        continue   # RSR計算専用銘柄をスキップ
```

- ✅ universe_tickers = LIVE_UNIVERSE（試用/卒業銘柄含む）
- ✅ rsr_universe_tickers = RSR_UNIVERSE_62（P2-A銘柄含む）
- ✅ RSR が min_rsr=75 以上ならBUY候補に入る
- ⚠️ 試用期間中は `qty × 0.25`（line 5297）

**注意**: P2-A gate は RSR≥8 で昇格を許可するが、BUY signal には RSR≥75 が必要。UNCLASSIFIED ゾーン（RSR=55-69）で昇格した場合、試用期間中 BUY signal が生成されない可能性がある。

---

## 4. E2E シミュレーション結果

```
tools/p2a_e2e_sim.py 実行結果（2026-06-20）:

PHASE 1: SHADOW → PROBATION
  6857.T (RSR=60, SI=72) → UNCLASSIFIED → P2-A gate PASS
  promotion_reason=['rsr_moderate', 'p2a_unclassified']
  LIVE_UNIVERSE(in-memory) に追加 ✅

PHASE 2: GRADUATION CHECK
  forward_return_3d: [None, None, None, None, None, None]  ← 全て None
  check_graduation → FAIL: 'no_forward_returns'            ← BLOCKER-1 確認

PHASE 3: 強制卒業シミュレーション
  get_graduated_symbols() = {'6857.T'}  ← ロジック正常
  LIVE_UNIVERSE(永続) 追加 ✅
  probation cap 解除 ✅

PHASE 4: BUY 到達確認
  6857.T in RSR_UNIVERSE_62: ✅
  6857.T in LIVE_UNIVERSE: ✅
  RSR(82)≥75: ✅ → BUY signal 到達可能
```

---

## 遮断箇所と修正案

### BLOCKER-1: forward_return_3d 未マテリアライズ (CRITICAL)

**遮断**: `check_graduation()` Cond2 が永久 FAIL → 卒業不可

**根本原因**:
```python
# auto_promote_safe_v2.py:965
forward_return_3d=None,   # "null at record time; materialized later" と記載されているが実装なし
```

**修正案 A（最小変更）**: `check_graduation()` のフォールバック

`fwd_rets` が空の場合、`rsr_delta > 0` を代理判定として使用:

```python
# check_graduation() 修正版 (Cond2)
fwd_rets = [_safe_float(o["forward_return_3d"]) for o in sym_outcomes
            if o.get("forward_return_3d") is not None]
if not fwd_rets:
    # フォールバック: RSR改善を代理指標として使用
    rsr_deltas = [_safe_float(o.get("rsr_delta", 0.0)) for o in sym_outcomes]
    if not rsr_deltas:
        return False, "no_forward_returns"
    avg_rsr_delta = sum(rsr_deltas) / len(rsr_deltas)
    if avg_rsr_delta <= 0:
        return False, f"rsr_proxy_negative:{avg_rsr_delta:.2f}"
    avg_ret = 0.001  # プレースホルダー（条件通過のみ）
else:
    avg_ret = sum(fwd_rets) / len(fwd_rets)
    if avg_ret <= 0:
        return False, f"expectancy_negative:{avg_ret:.4f}"
```

**修正案 B（中規模変更）**: バックフィル関数の追加

```python
def backfill_probation_forward_returns(
    outcomes_path: Path,
    ohlcv_dir: Path,
) -> int:
    """data/ohlcv/*.parquet を使って3日後リターンを事後記録"""
    # observation_date から3営業日後の終値変化を計算
    # OHLCV parquet は既存（毎朝更新）
    ...
```

`run_live_signal.py` で `run_probation_outcome_observation()` の前に呼び出す。

**推奨**: 修正案 A（即効性・安全性）を先行実施し、修正案 B を研究バックログに積む。

---

### BLOCKER-2: 卒業日の再昇格バグ (MEDIUM、BLOCKER-1修正後に顕在化)

**遮断**: 卒業 (STATUS_GRADUATED) と同一 run 内で SHADOW → PROBATION 再評価 → STATUS_ACTIVE 上書き → `get_graduated_symbols()` が最新レコード STATUS_ACTIVE を見つけ卒業不発

**根本原因**:
```python
# run_probation_gate() Step4 の already_seen に just_graduated が含まれない
already_seen = active_symbols | cooldown_syms | set(live_universe.keys())
```

**修正案**: `_just_graduated` 追跡

```python
# Step 1 ループで卒業銘柄を追跡
_just_graduated: Set[str] = set()
for rec in active_records:
    ...
    if graduated:
        _just_graduated.add(sym)   # ← 追加
        ...continue

# Step 4 の already_seen に追加
already_seen = active_symbols | cooldown_syms | _just_graduated | set(live_universe.keys())
```

---

### ISSUE-3: P2-A gate RSR下限 と demotion 閾値の不整合 (MEDIUM)

**問題**:
- P2-A gate: `RSR >= GATE_MIN_RSR_PASS = 8.0`
- demotion: `current_rsr < BREAKOUT_FAIL_RSR_MIN = 65.0` → 降格

RSR=[8, 65) で昇格した銘柄は翌日即降格。

**修正案**: P2-A gate に demotion 閾値との整合を追加

```python
# check_p2a_unclassified_gate() に追加
from src.universe.auto_promote_safe_v2 import BREAKOUT_FAIL_RSR_MIN
if rsr < BREAKOUT_FAIL_RSR_MIN:
    fail.append(f"p2a_rsr_demotion_risk:{rsr:.1f}<{BREAKOUT_FAIL_RSR_MIN}")
```

ただし P2-A WF では RSR≥8 で検証済み。変更前に ASK_FIRST 必須。

---

### ISSUE-4: 試用銘柄のファイル未永続化 (LOW、設計上 intentional)

- PROBATION(ACTIVE) → `LIVE_UNIVERSE[sym]` は in-memory のみ
- `run_probation_gate()` FAIL_OPEN 発動時: その日 BUY 対象外
- 翌日は再評価・再追加されるが当日 gap が生じる

設計意図は理解できる（試用期間は揮発的）。文書化のみ推奨。

---

## 優先対応リスト

| 優先度 | Issue | 修正工数 | ASK_FIRST |
|---|---|---|---|
| 🔴 P1 | BLOCKER-1: forward_return_3d → `check_graduation()` フォールバック追加 | 小 (15行) | No |
| 🔴 P1 | BLOCKER-2: `_just_graduated` → `already_seen` 追加 | 小 (5行) | No |
| 🟡 P2 | ISSUE-3: P2-A gate RSR下限を65に引き上げ | 小 (3行) | **Yes** (PARAMS変更相当) |
| 🟢 P3 | ISSUE-4: 試用銘柄永続化の検討 | 中 | **Yes** |

---

## 最終判定

```
SHADOW → PROBATION(ACTIVE) → LIVE_UNIVERSE(in-mem) → BUY(0.25x):  ✅ 正常動作
PROBATION(ACTIVE) → GRADUATED:                                      ❌ BLOCKED (BLOCKER-1)
GRADUATED → LIVE_UNIVERSE(永続) → BUY(full):                       ❌ BLOCKED (BLOCKER-1+2)
```

P2-A 候補は試用期間中（最大5日×無制限継続）は BUY 対象となる（RSR≥75 条件付き）。
永続ユニバース昇格（full allocation）は BLOCKER-1/2 修正なしには到達不可。
