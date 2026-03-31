"""
scripts/shadow_report.py
Shadow取引の早期評価スコアボード

実行:
    python scripts/shadow_report.py

出力:
    - shadow_trade_count / expectancy / winrate / avg_hold_days
    - 早期評価モード判定（sample>=6 で暫定昇格候補を判定）
    - 通常評価モード判定（sample>=15 で確定昇格）
"""
import sys
import json
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from paths import LOGS_DIR

TRADES_PATH    = LOGS_DIR / "trades.jsonl"
EARLY_SAMPLE   = 6      # 早期評価サンプル数
FULL_SAMPLE    = 15     # 本評価サンプル数
EARLY_EXPECT   = 0.012  # 早期評価: expectancy > +1.2%
EARLY_WINRATE  = 0.60   # 早期評価: winrate > 60%
FULL_EXPECT    = 0.010  # 本評価:  expectancy > +1.0%
FULL_WINRATE   = 0.55   # 本評価:  winrate > 55%
MAE_LIMIT      = -0.05  # 本評価:  MAE > -5%（最大不利エクスカーション）

print("=" * 60)
print("  Shadow取引 早期評価スコアボード")
print("=" * 60)

if not TRADES_PATH.exists():
    print("\n  [情報] logs/trades.jsonl が存在しません。")
    print("  Shadow取引が1件も成立していない状態です。")
    print("\n  早期評価開始条件: 取引 >= 6件")
    print("  通常評価開始条件: 取引 >= 15件")
    sys.exit(0)

# --- trades.jsonl から shadow 決済済み取引を抽出 ---
returns:   list[float] = []
hold_days: list[int]   = []
maes:      list[float] = []
per_symbol: dict[str, list[float]] = {}

with TRADES_PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)

        # shadow 由来 かつ SELL（決済済み）のみ
        if not t.get("is_shadow"):
            continue
        ret = t.get("shadow_realized_return")
        if ret is None:
            continue

        sym = t.get("symbol", "?")
        returns.append(ret)
        hold_days.append(t.get("hold_days") or 0)
        # MAE は未記録の場合 None → 除外
        if t.get("mae") is not None:
            maes.append(t["mae"])
        per_symbol.setdefault(sym, []).append(ret)

# --- 仮想エントリー（virtual shadow）の決済も確認 ---
virtual_returns: list[float] = []
with TRADES_PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        if t.get("is_shadow") and t.get("side") == "SELL" and t.get("shadow_realized_return") is not None:
            vret = t.get("shadow_realized_return")
            if vret not in returns:  # 実発注と重複しない
                virtual_returns.append(vret)

n = len(returns)
n_virtual = len(virtual_returns)
all_returns = returns + virtual_returns
n_all = len(all_returns)

print(f"\n  実取引:   {n}件")
print(f"  仮想取引: {n_virtual}件（blocked_by_alloc後に追跡）")
print(f"  合計:     {n_all}件")

if n_all == 0:
    print("\n  まだデータなし。最初のShadow取引決済を待っています。")
    print(f"\n  早期評価開始まで: あと {EARLY_SAMPLE}件")
    sys.exit(0)

# --- 統計計算 ---
expectancy = statistics.mean(all_returns)
winrate    = sum(r > 0 for r in all_returns) / n_all
avg_hold   = statistics.mean(hold_days) if hold_days else None
avg_mae    = statistics.mean(maes) if maes else None

print(f"\n{'─'*50}")
print(f"  {'指標':<20} {'値':>15}")
print(f"{'─'*50}")
print(f"  {'expectancy':<20} {expectancy*100:>+14.2f}%")
print(f"  {'winrate':<20} {winrate*100:>14.1f}%")
print(f"  {'avg_hold_days':<20} {avg_hold if avg_hold else '?':>15}")
if avg_mae is not None:
    print(f"  {'avg_MAE':<20} {avg_mae*100:>+14.2f}%")
print(f"{'─'*50}")

# --- 昇格判定 ---
print("\n  【昇格判定】")

# 早期評価
if n_all >= EARLY_SAMPLE:
    early_pass = (expectancy > EARLY_EXPECT) and (winrate > EARLY_WINRATE)
    early_tag  = "✅ PROMOTE_EARLY" if early_pass else "⬜ 未達"
    print(f"  早期評価 (n>={EARLY_SAMPLE}): "
          f"expect>{EARLY_EXPECT*100:.1f}% AND winrate>{EARLY_WINRATE*100:.0f}% → {early_tag}")
    if early_pass:
        print("    ⚠ 暫定昇格候補（要ユーザー確認・期間限定で実発注テスト可）")
else:
    print(f"  早期評価: あと {EARLY_SAMPLE - n_all}件でチェック開始")

# 本評価
if n_all >= FULL_SAMPLE:
    mae_ok   = (avg_mae is None) or (avg_mae > MAE_LIMIT)
    full_pass = (expectancy > FULL_EXPECT) and (winrate > FULL_WINRATE) and mae_ok
    full_tag  = "✅ PROMOTE_CONFIRMED" if full_pass else "⬜ 未達"
    print(f"  本評価   (n>={FULL_SAMPLE}): "
          f"expect>{FULL_EXPECT*100:.1f}% AND winrate>{FULL_WINRATE*100:.0f}% AND MAE>{MAE_LIMIT*100:.0f}% → {full_tag}")
else:
    print(f"  本評価:   あと {FULL_SAMPLE - n_all}件でチェック開始")

# --- 銘柄別集計 ---
if per_symbol:
    print(f"\n  【銘柄別】")
    for sym, rets in sorted(per_symbol.items(), key=lambda x: -statistics.mean(x[1])):
        mean_r = statistics.mean(rets)
        wr     = sum(r > 0 for r in rets) / len(rets)
        print(f"  {sym:<8} n={len(rets):>2}  expect={mean_r*100:>+6.2f}%  winrate={wr*100:.0f}%")

print(f"\n  確定昇格条件（変更禁止）:")
print(f"    sample >= {FULL_SAMPLE}")
print(f"    expectancy > {FULL_EXPECT*100:.1f}%")
print(f"    winrate > {FULL_WINRATE*100:.0f}%")
print(f"    MAE > {MAE_LIMIT*100:.0f}%")
print("=" * 60)
