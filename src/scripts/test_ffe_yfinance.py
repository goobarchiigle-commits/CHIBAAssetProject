"""
一時診断スクリプト — FFE yfinance疎通確認 + enrich集計
実行: python src/scripts/test_ffe_yfinance.py
"""
from __future__ import annotations

import logging
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ffe_diag")

# paths import → load_dotenv → os.environ 設定
from src.paths import (
    ALLOW_YFINANCE_NETWORK,
    ENTRY_FEATURES_FILE,
    FORWARD_RETURN_CACHE_DIR,
)

# ─── Step 1: 環境変数確認 ────────────────────────────────────────────────────
print("\n=== [1] ALLOW_YFINANCE_NETWORK ===")
print(f"  paths.py 値: {ALLOW_YFINANCE_NETWORK}")

import os
env_raw = os.environ.get("ALLOW_YFINANCE_NETWORK", "<unset>")
print(f"  os.environ raw: {env_raw}")
print(f"  → fetch_forward_returns が通過するか: {env_raw.lower() == 'true'}")

# ─── Step 2: 5706.T 単体テスト ───────────────────────────────────────────────
print("\n=== [2] yfinance 5706.T 単体テスト ===")
SYMBOL = "5706.T"
START  = "2026-05-20"
END    = (date.fromisoformat(START) + timedelta(days=32)).strftime("%Y-%m-%d")
print(f"  symbol={SYMBOL}  start={START}  end={END}")

try:
    import yfinance as yf
    print(f"  yfinance version: {yf.__version__}")
    ticker = yf.Ticker(SYMBOL)
    hist = ticker.history(start=START, end=END, auto_adjust=True)
    print(f"  hist.shape: {hist.shape}")
    print(f"  hist.empty: {hist.empty}")
    if not hist.empty:
        print(f"  hist.head(3):\n{hist[['Open','High','Low','Close']].head(3)}")
    else:
        print("  ⚠ 空のDataFrame — データなし")
except Exception as e:
    print(f"  ✗ 例外発生: {type(e).__name__}: {e}")
    traceback.print_exc()

# ─── Step 3: fetch_forward_returns 直接実行 ──────────────────────────────────
print("\n=== [3] fetch_forward_returns(5706.T, 2026-05-20) ===")
from src.analytics.feature_forward_expectancy import fetch_forward_returns

result = fetch_forward_returns(SYMBOL, START, 0.0, FORWARD_RETURN_CACHE_DIR)
print(f"  result: {result}")

# ─── Step 4: forward_return_cache 確認 ──────────────────────────────────────
print("\n=== [4] forward_return_cache ディレクトリ確認 ===")
print(f"  path: {FORWARD_RETURN_CACHE_DIR}")
print(f"  exists: {FORWARD_RETURN_CACHE_DIR.exists()}")
if FORWARD_RETURN_CACHE_DIR.exists():
    files = list(FORWARD_RETURN_CACHE_DIR.glob("*.json"))
    print(f"  .json files: {len(files)}")
    for f in files[:5]:
        print(f"    {f.name}")

# ─── Step 5: enrich_snapshots_with_forward_returns 実行 + 集計 ──────────────
print("\n=== [5] enrich_snapshots_with_forward_returns 実行 ===")
from src.analytics.feature_forward_expectancy import (
    load_entry_snapshots,
    enrich_snapshots_with_forward_returns,
)

snapshots = load_entry_snapshots(ENTRY_FEATURES_FILE)
print(f"  total snapshots loaded: {len(snapshots)}")

before_enriched = sum(1 for s in snapshots if s.enrichment_status == "enriched")
before_pending  = sum(1 for s in snapshots if s.enrichment_status == "pending")
before_expired  = sum(1 for s in snapshots if s.enrichment_status == "expired")
print(f"  before — enriched:{before_enriched}  pending:{before_pending}  expired:{before_expired}")

# enrich 実行
snapshots = enrich_snapshots_with_forward_returns(snapshots, FORWARD_RETURN_CACHE_DIR)

after_enriched = sum(1 for s in snapshots if s.enrichment_status == "enriched")
after_pending  = sum(1 for s in snapshots if s.enrichment_status == "pending")
after_expired  = sum(1 for s in snapshots if s.enrichment_status == "expired")
print(f"  after  — enriched:{after_enriched}  pending:{after_pending}  expired:{after_expired}")

newly_enriched = after_enriched - before_enriched
failed = before_pending - after_pending - newly_enriched  # pending → expired 分
print(f"\n  newly enriched : {newly_enriched}")
print(f"  newly expired  : {after_expired - before_expired}")
print(f"  still pending  : {after_pending}")

# forward_return_cache 再確認
if FORWARD_RETURN_CACHE_DIR.exists():
    files = list(FORWARD_RETURN_CACHE_DIR.glob("*.json"))
    print(f"\n  forward_return_cache .json files after enrich: {len(files)}")

# ─── Step 6: 成功した場合はサンプル表示 ─────────────────────────────────────
enriched_snaps = [s for s in snapshots if s.enrichment_status == "enriched"]
if enriched_snaps:
    print("\n=== [6] enriched snapshot サンプル ===")
    s = enriched_snaps[0]
    print(f"  symbol={s.symbol}  eval_date={s.eval_date}")
    print(f"  forward_1d={s.forward_1d}  forward_5d={s.forward_5d}  forward_20d={s.forward_20d}")
    print(f"  mae={s.forward_mae}  mfe={s.forward_mfe}")

print("\n=== 診断完了 ===")
