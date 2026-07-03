"""
morning_smoke_test.py
朝シグナル前に bear_universe_filter の稼働確認を自動出力。
run_morning_signal.py から bridge.run() 完了後に呼び出す。

bear_filter_log.jsonl から今日の bear 判定情報を読み取る。
当日エントリーがなければ Bull レジームとみなす（フィルター非発動）。
"""
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SMOKE_LOG_PATH = Path("runtime/smoke_test_log.jsonl")
BEAR_LOG_PATH  = Path("runtime/bear_filter_log.jsonl")
MIN_ACTIVE_COUNT = 3   # これ以下で WARNING


def _read_today_bear_log(today: date) -> Optional[dict]:
    """
    bear_filter_log.jsonl から today の最新エントリーを返す。
    見つからなければ None（Bull レジームまたはフィルター非発動）。
    """
    if not BEAR_LOG_PATH.exists():
        return None
    today_str = str(today)
    try:
        last_match = None
        with open(BEAR_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("date") == today_str:
                        last_match = rec
                except Exception:
                    continue
        return last_match
    except Exception as e:
        logger.warning("[SMOKE TEST] bear_filter_log 読み取り失敗: %s", e)
        return None


def run_morning_smoke_test(
    today: date,
    is_bear: Optional[bool] = None,
    active_syms: Optional[list] = None,
    excluded_syms: Optional[list] = None,
    excluded_count: Optional[int] = None,
    universe_tickers: Optional[dict] = None,
    cfg=None,
) -> dict:
    """
    朝シグナル前の smoke test。bridge.run() 完了後に呼び出す。

    bear_filter_log.jsonl から当日の bear 判定情報を自動取得する。
    引数で明示的に渡された場合はそちらを優先する。

    Parameters
    ----------
    today           : 実行日
    is_bear         : Bear レジーム判定（None = ログから自動取得）
    active_syms     : フィルター後のアクティブ銘柄リスト（None = ログ or universe_tickers から推定）
    excluded_syms   : 除外銘柄リスト（None = ログから取得）
    excluded_count  : 除外銘柄数（None = ログから取得）
    universe_tickers: {symbol: sector} — bull 時の全銘柄リスト（active_syms 推定用）
    cfg             : 設定オブジェクト（将来拡張用）

    Returns
    -------
    dict with keys: date, bear_regime, active_count, excluded_count, passed
    """
    # bear_filter_log から今日のエントリーを取得
    bear_log = _read_today_bear_log(today)

    # is_bear: 引数優先 → ログ → デフォルト False
    if is_bear is None:
        is_bear = bool(bear_log.get("bear_regime", False)) if bear_log else False

    # excluded_count / excluded_syms: 引数優先 → ログ → デフォルト 0/[]
    if excluded_count is None:
        excluded_count = int(bear_log.get("excluded_count", 0)) if bear_log else 0
    if excluded_syms is None:
        excluded_syms = list(bear_log.get("excluded_symbols", [])) if bear_log else []

    # active_syms: 引数優先 → ログ（active_count のみ取得） → universe_tickers → []
    if active_syms is None:
        if bear_log and "active_count" in bear_log:
            # ログには active_count しかないためダミーリストで代替
            _cnt = int(bear_log["active_count"])
            active_syms = [f"__sym_{i}" for i in range(_cnt)]
        elif universe_tickers is not None:
            active_syms = list(universe_tickers.keys())
        else:
            active_syms = []

    result: dict = {
        "date": str(today),
        "bear_regime": is_bear,
        "active_count": len(active_syms),
        "excluded_count": excluded_count,
        "sample_excluded": excluded_syms[:5],
        "sample_active": [s for s in active_syms[:5] if not s.startswith("__sym_")],
        "passed": True,
        "warning": None,
    }

    # 確認ポイント: active_count が極端に少なくないか
    if len(active_syms) < MIN_ACTIVE_COUNT:
        result["passed"] = False
        result["warning"] = (
            f"Universe too narrow: active_count={len(active_syms)} < {MIN_ACTIVE_COUNT}"
        )
        logger.warning(
            "[SMOKE TEST] Universe too narrow: active_count=%d < %d",
            len(active_syms), MIN_ACTIVE_COUNT,
        )

    # stdout 出力（朝のオペレーターが目視確認できるように）
    bear_str = "🐻 BEAR" if is_bear else "🐂 BULL"
    status_str = "⚠️ WARNING" if not result["passed"] else "✅ OK"
    print(json.dumps({
        "smoke_test": {
            "date": str(today),
            "regime": bear_str,
            "active_count": len(active_syms),
            "excluded_count": excluded_count,
            "sample_excluded": excluded_syms[:5],
            "status": status_str,
        }
    }, ensure_ascii=False, indent=2))

    # jsonl ログ追記（try/except で安全化）
    _log_record = {
        "date": str(today),
        "bear_regime": is_bear,
        "active_count": len(active_syms),
        "excluded_count": excluded_count,
        "sample_excluded": excluded_syms[:5],
        "passed": result["passed"],
        "warning": result["warning"],
    }
    try:
        SMOKE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SMOKE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(_log_record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("[SMOKE TEST] ログ書き込み失敗: %s", e)

    return result
