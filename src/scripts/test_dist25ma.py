"""
一時診断スクリプト — distance_25ma_pct データフロー確認
実行: python src/scripts/test_dist25ma.py
"""
from __future__ import annotations
import sys, json, logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")

from src.paths import ENTRY_FEATURES_FILE

# ─── 1. シグナル辞書の確認 ──────────────────────────────────────────────────
print("=== [1] signal dict の distance_25ma_pct キー確認 ===")
# write_signal_snapshots() に渡されるシグナル辞書をモック
from src.analytics.feature_forward_expectancy import write_signal_snapshots

mock_sig_with = {
    "action": "SKIP_BUY", "signal": 1, "symbol": "5706.T", "sector": "鉄鋼",
    "rsr": 98.2, "rsr_rank": 2, "reason": "test", "currently_holding": False,
    "distance_25ma_pct": 0.0523,   # ← 値あり
}
mock_sig_without = {
    "action": "SKIP_BUY", "signal": 1, "symbol": "5706.T", "sector": "鉄鋼",
    "rsr": 98.2, "rsr_rank": 2, "reason": "test", "currently_holding": False,
    # distance_25ma_pct なし
}

import tempfile
with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
    tmp = Path(f.name)

n1 = write_signal_snapshots([mock_sig_with], "test_run_1", 0.5, 0.5, 0.6, 0.3, "bull", tmp)
snap1 = json.loads(tmp.read_text(encoding="utf-8").strip().splitlines()[-1])
print(f"  with value:    written={n1}  distance_25ma_pct={snap1['distance_25ma_pct']!r}")

tmp.write_text("", encoding="utf-8")  # reset
n2 = write_signal_snapshots([mock_sig_without], "test_run_2", 0.5, 0.5, 0.6, 0.3, "bull", tmp)
snap2 = json.loads(tmp.read_text(encoding="utf-8").strip().splitlines()[-1])
print(f"  without value: written={n2}  distance_25ma_pct={snap2['distance_25ma_pct']!r}")

# ─── 2. diagnostics["sym_dist25ma"] の確認 ──────────────────────────────────
print("\n=== [2] signal_bridge diagnostics キー確認 ===")
from src.kabusapi.signal_bridge import SignalBridge

# SignalBridge をインスタンス化してポートフォリオを読み込む（dry-runパス）
try:
    sb = SignalBridge(live=False)
    # _generate_all_signals が diagnostics に "sym_dist25ma" を含むか
    # ソースコードから直接確認
    import inspect, ast
    src = inspect.getsource(sb._generate_all_signals)
    has_key = "sym_dist25ma" in src
    print(f"  'sym_dist25ma' in _generate_all_signals source: {has_key}")

    diag_section = [l for l in src.splitlines() if "sym_dist25ma" in l]
    for line in diag_section:
        print(f"    {line.strip()}")
except Exception as e:
    print(f"  SignalBridge instantiation failed (expected in CI): {e}")

# ─── 3. 最新スナップショットの確認 ──────────────────────────────────────────
print("\n=== [3] 最新スナップショット distance_25ma_pct 分布 ===")
lines = ENTRY_FEATURES_FILE.read_text(encoding="utf-8").strip().splitlines()
non_none = [json.loads(l)["distance_25ma_pct"] for l in lines if json.loads(l)["distance_25ma_pct"] is not None]
print(f"  total: {len(lines)}  non-None: {len(non_none)}  None: {len(lines)-len(non_none)}")

print("\n=== 完了 ===")
