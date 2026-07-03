"""
run_bear_filter_compare.py
Bear ユニバースフィルター比較スクリプト（research ブランチ用スケルトン）

3パターンのフィルター戦略をウォークフォワード比較する:
  A: 固定セクター除外（現行 production）
  B: 動的セクター除外（rs_63 < 0）
  C: A ∪ B（固定 OR 動的）

【実行方法（実行は次フェーズ）】
  python src/scripts/run_bear_filter_compare.py \
      --start 2018-01-01 --end 2024-12-31 \
      --variant A B C

【注意】
  - strategy.yaml の bear_dynamic_filter.enabled を確認してから実行すること
  - データ取得に yfinance を使用（外部通信あり）
  - バックテスト結果は backtests/bear_filter_compare_YYYYMMDD.json に保存
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# パス設定（C:/ai-trading/ から実行することを前提）
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from src.paths import RESULTS_DIR
from src.config_loader import load_strategy_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ------------------------------------------------------------------ #
# 固定セクターリスト（variant A / C で使用）
# ------------------------------------------------------------------ #
FIXED_EXCLUDE_SECTORS = [
    "機械", "鉄鋼", "銀行業", "保険業", "輸送用機器", "海運業", "化学",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bear フィルター A/B/C 比較（WF バックテスト）"
    )
    p.add_argument("--start",   default="2018-01-01", help="バックテスト開始日")
    p.add_argument("--end",     default="2024-12-31", help="バックテスト終了日")
    p.add_argument(
        "--variant",
        nargs="+",
        choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="比較するフィルターバリアント",
    )
    p.add_argument("--n-splits", type=int, default=5, help="WF 分割数")
    return p.parse_args()


# ------------------------------------------------------------------ #
# ダミーデータ取得（実装時に backtest.composite_alpha_bt から流用）
# ------------------------------------------------------------------ #
def load_universe_data(start: str, end: str) -> tuple:
    """
    ユニバース価格データ・TOPIX・セクターマップを読み込む。

    Returns
    -------
    (universe_raw, topix_close, sym_sector_map, rsr_df)

    TODO: backtest/composite_alpha_bt.py の _load_data() を参照して実装。
    現在はスケルトンのため NotImplementedError を送出する。
    """
    raise NotImplementedError(
        "load_universe_data() は次フェーズで実装。"
        "backtest/composite_alpha_bt.py の _load_data() を参照。"
    )


# ------------------------------------------------------------------ #
# フィルターバリアント適用
# ------------------------------------------------------------------ #
def get_bear_exclude_sectors_for_variant(
    variant: str,
    sector_returns_63d: pd.Series,
    topix_return_63d: float,
) -> list[str]:
    """
    バリアントに応じて除外セクターリストを返す。

    A: 固定リスト
    B: rs_63 < 0 の動的リスト
    C: A ∪ B

    Parameters
    ----------
    variant            : "A" | "B" | "C"
    sector_returns_63d : セクター別 63日リターン（index=sector 名）
    topix_return_63d   : TOPIX 63日リターン

    Returns
    -------
    list[str] : 除外するセクター名リスト
    """
    if variant == "A":
        return list(FIXED_EXCLUDE_SECTORS)

    # B: 動的除外（rs_63 < 0）
    sector_rs  = sector_returns_63d - topix_return_63d
    dynamic_ex = sector_rs[sector_rs < 0].index.tolist()

    if variant == "B":
        return dynamic_ex

    # C: A ∪ B
    return list(set(FIXED_EXCLUDE_SECTORS) | set(dynamic_ex))


# ------------------------------------------------------------------ #
# WF 比較メイン
# ------------------------------------------------------------------ #
def run_wf_compare(
    universe_raw: dict,
    topix_close: pd.Series,
    sym_sector_map: dict,
    rsr_df: pd.DataFrame,
    start: str,
    end: str,
    variants: list[str],
    n_splits: int = 5,
) -> dict:
    """
    ウォークフォワードで A/B/C バリアントを比較し、Sharpe/DD を返す。

    TODO: 次フェーズで実装。
    - backtest/engine.py の run_scenario() を variant ごとに呼び出す
    - bear_exclude_sectors を variant に応じて切り替える
    - WF 分割は TimeSeriesSplit（shuffle=False 厳守）

    Returns
    -------
    dict: {variant: {"sharpe": float, "max_dd": float, "n_trades": int}}
    """
    raise NotImplementedError(
        "run_wf_compare() は次フェーズで実装。"
        "backtest/engine.py の run_scenario() を参照。"
    )


# ------------------------------------------------------------------ #
# エントリーポイント
# ------------------------------------------------------------------ #
def main() -> int:
    args = parse_args()
    cfg  = load_strategy_config()

    # research フラグ確認
    _dyn_cfg = getattr(getattr(cfg, "risk_controls", None), "bear_dynamic_filter", None)
    if _dyn_cfg is None or not getattr(_dyn_cfg, "enabled", False):
        logger.warning(
            "strategy.yaml の risk_controls.bear_dynamic_filter.enabled が false です。"
            "このスクリプトは research 専用です。enabled: true に変更してから実行してください。"
        )
        # スケルトン段階では警告のみ（本実装時はここで return 1 にする）

    logger.info("Bear フィルター比較スクリプト（スケルトン）")
    logger.info("  variants: %s  period: %s ~ %s  WF splits: %d",
                args.variant, args.start, args.end, args.n_splits)

    try:
        universe_raw, topix_close, sym_sector_map, rsr_df = load_universe_data(
            args.start, args.end
        )
    except NotImplementedError as e:
        logger.info("スケルトン実行確認 OK: %s", e)
        # スケルトン段階では正常終了
        return 0

    try:
        results = run_wf_compare(
            universe_raw   = universe_raw,
            topix_close    = topix_close,
            sym_sector_map = sym_sector_map,
            rsr_df         = rsr_df,
            start          = args.start,
            end            = args.end,
            variants       = args.variant,
            n_splits       = args.n_splits,
        )
    except NotImplementedError as e:
        logger.info("スケルトン実行確認 OK: %s", e)
        return 0

    # 結果保存
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULTS_DIR / f"bear_filter_compare_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("結果保存: %s", out_path)

    # サマリー表示
    print("\n=== Bear フィルター比較結果 ===")
    print(f"{'Variant':<10} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>8}")
    print("-" * 38)
    for v, r in results.items():
        print(
            f"{v:<10} {r.get('sharpe', float('nan')):>8.3f} "
            f"{r.get('max_dd', float('nan')):>8.2f} "
            f"{r.get('n_trades', 0):>8d}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
