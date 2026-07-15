"""
src/live/entry_metadata_recovery.py

既存保有ポジションの entry_date / entry_price / entry_atr / entry_rsr 欠落リカバリ（2026-07-15全面改訂）。

背景（2026-07-14/15 RCA）:
  旧実装は logs/live/*_orders.json のみを検索し、価格/ATRは復元してもentry_date自体は
  一度も書き込んでいなかった（_find_order_metadata()がdateを返さない設計バグ）。
  この結果 position_entry_dates が永久に空のままとなり、以下が発生していた:
    1. signal_bridge.py の時間ストップ判定（max_hold_days超過チェック）が
       "sym in pos_entry_dates" 条件でスキップされ、対象銘柄は保有日数無制限になっていた
       （6981.T: 実entry=2026-04-28・2026-07-15時点で約52営業日保有・max_hold_days=60に接近）。
    2. RSRランキング表示の「保有日」が常に0dとなり、当日新規建てと誤認されていた。
    3. position_entry_atrs欠落によりEQ Scale addon/ATR Extension/Quality Replacement
       Engineのatr_expansion計算が対象銘柄で機能していなかった（Study51 D_EQ_SCALE=Production
       採用済み機構への実害）。
    4. 復元試行が失敗しても detected_at が毎日「今日」で上書きされ続け、
       「本当にいつから欠損しているか」の記録が失われていた。

本改訂の変更点:
  - 検索ソースを4系統に拡張し優先順位付きで統合（最新のBUY成立が最優先）:
      [1] runtime/orders/{ack,filled}/*.json  （個別注文ジャーナル・created_at保持）
      [2] logs/live/*_orders.json             （旧実装の唯一のソース・atr20保持）
      [3] data/signals/*_executed.json        （orders[]/send_results[]・atr20保持）
      [4] logs/execution_quality/*.jsonl      （最も履歴が長いが"side"フィールドが無く
                                                 BUY/SELL判別不能——data/signals側で同日
                                                 SELL確認済みの日付を除外して低信頼度採用）
  - 発見した場合、entry_date を必ず position_entry_dates へ書き込む（旧実装のバグ修正）。
  - ATR欠落理由を細分化（entry_date_unknown / atr_source_missing_but_date_known /
    no_price_history_before_entry）。
  - detected_at は初回検出時刻を保持し続ける（毎日上書きしない）。recovery_attempts で
    再試行回数のみ記録する。

実行:
    python -m src.live.entry_metadata_recovery              # 適用（portfolio_state.json書込み）
    python -m src.live.entry_metadata_recovery --dry-run     # プレビューのみ・書込みなし
"""
from __future__ import annotations

import json
import logging
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


# ======================================================================
# 個別ソース検索（各々 {entry_date, estimated_price, atr20, confidence, source, source_file} を返す）
# ======================================================================
def _parse_date_from_filename_prefix(name: str) -> str:
    """'20260707_084405_xxx.json' 等のファイル名先頭8桁をYYYY-MM-DDへ変換。失敗時は空文字。"""
    digits = name[:8]
    if len(digits) == 8 and digits.isdigit():
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _find_from_order_journal(orders_dir: Path, symbol: str) -> dict | None:
    """runtime/orders/{ack,filled}/*.json — 個別注文ジャーナル（created_at保持・atr20無し）。"""
    best: dict | None = None
    for sub in ("ack", "filled", "submitted"):
        d = orders_dir / sub
        if not d.exists():
            continue
        for path in d.glob(f"*_{symbol}_*.json"):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("symbol") != symbol or rec.get("side") not in ("BUY", "SHADOW_BUY"):
                continue
            created_at = rec.get("created_at", "")
            entry_date = created_at[:10] if created_at else _parse_date_from_filename_prefix(path.name)
            if not entry_date:
                continue
            price = float(rec.get("price", 0.0) or 0.0)
            if price <= 0:
                continue
            cand = {
                "entry_date": entry_date, "estimated_price": price, "atr20": 0.0,
                "confidence": "high", "source": "order_journal", "source_file": str(path.name),
            }
            if best is None or cand["entry_date"] > best["entry_date"]:
                best = cand
    return best


def _find_from_logs_live_orders(logs_live_dir: Path, symbol: str) -> dict | None:
    """logs/live/*_orders.json（旧実装唯一のソース）。atr20を保持。"""
    if not logs_live_dir.exists():
        return None
    best: dict | None = None
    for path in sorted(logs_live_dir.glob("*_orders.json")):
        try:
            runs = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        file_date = _parse_date_from_filename_prefix(path.name)
        for run in runs:
            for o in run.get("orders", []):
                if o.get("symbol") != symbol or o.get("side") not in ("BUY", "SHADOW_BUY"):
                    continue
                price = float(o.get("estimated_price", 0.0) or 0.0)
                if price <= 0:
                    continue
                cand = {
                    "entry_date": file_date, "estimated_price": price,
                    "atr20": float(o.get("atr20", 0.0) or 0.0),
                    "confidence": "high", "source": "logs_live_orders", "source_file": path.name,
                }
                if not cand["entry_date"]:
                    continue
                if best is None or cand["entry_date"] > best["entry_date"]:
                    best = cand
    return best


def _find_from_executed_signals(signals_dir: Path, symbol: str) -> dict | None:
    """data/signals/*_executed.json — orders[]/send_results[]。atr20を保持。"""
    if not signals_dir.exists():
        return None
    best: dict | None = None
    for path in sorted(signals_dir.glob("signal_*_executed.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        file_date = _parse_date_from_filename_prefix(path.name.replace("signal_", ""))
        for o in (d.get("send_results") or []) + (d.get("orders") or []):
            if o.get("symbol") != symbol or o.get("side") not in ("BUY", "SHADOW_BUY"):
                continue
            if "success" in o and not o.get("success"):
                continue
            price = float(o.get("estimated_price") or o.get("planned_entry_price") or 0.0)
            if price <= 0:
                continue
            submit_time = o.get("order_submit_time", "")
            entry_date = submit_time[:10] if submit_time else file_date
            if not entry_date:
                continue
            cand = {
                "entry_date": entry_date, "estimated_price": price,
                "atr20": float(o.get("atr20", 0.0) or 0.0),
                "confidence": "high", "source": "executed_signal_json", "source_file": path.name,
            }
            if best is None or cand["entry_date"] > best["entry_date"]:
                best = cand
    return best


def _collect_known_sell_dates(signals_dir: Path, symbol: str) -> set[str]:
    """data/signals/*_executed.json で symbol が明示的にSELLされた日付集合。
    execution_quality.jsonl（side不明）の誤採用を防ぐための除外リストとして使う。"""
    sell_dates: set[str] = set()
    if not signals_dir.exists():
        return sell_dates
    for path in signals_dir.glob("signal_*_executed.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for o in (d.get("send_results") or []) + (d.get("orders") or []):
            if o.get("symbol") == symbol and o.get("side") == "SELL":
                t = o.get("order_submit_time", "") or _parse_date_from_filename_prefix(
                    path.name.replace("signal_", "")
                )
                if t:
                    sell_dates.add(t[:10])
    return sell_dates


def _find_from_execution_quality(
    logs_dir: Path, symbol: str, known_sell_dates: set[str],
) -> dict | None:
    """
    logs/execution_quality/*.jsonl — 最も履歴が長いが"side"フィールドが無く
    BUY/SELLを判別できない（execution_metrics.log_execution_event()がBUY/SELL両方から
    共有呼び出しされているため）。known_sell_dates（正規ソースで確認済みのSELL日付）を
    除外し、低信頼度（confidence="low"）としてのみ採用する。
    """
    if not logs_dir.exists():
        return None
    best: dict | None = None
    for path in sorted(logs_dir.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except Exception:
                continue
            if d.get("symbol") != symbol:
                continue
            if d.get("fill_status") not in ("submitted",):
                continue
            signal_time = d.get("entry_signal_time", "")
            entry_date = signal_time[:10] if signal_time else ""
            if not entry_date or entry_date in known_sell_dates:
                continue
            price = float(d.get("planned_entry_price", 0.0) or 0.0)
            if price <= 0:
                continue
            cand = {
                "entry_date": entry_date, "estimated_price": price, "atr20": 0.0,
                "confidence": "low", "source": "execution_quality_ambiguous_side",
                "source_file": path.name,
            }
            if best is None or cand["entry_date"] > best["entry_date"]:
                best = cand
    return best


def _find_best_entry_metadata(
    symbol: str, *, orders_dir: Path, logs_live_dir: Path, signals_dir: Path, exec_quality_dir: Path,
) -> dict | None:
    """4ソースを検索し、最新のBUY成立を統合結果として返す。
    高信頼度ソースが1件でもあればそれを優先し、execution_quality（低信頼度）は
    高信頼度ソースが1件も無い場合のみ最終手段として使う。"""
    high_conf = [
        c for c in (
            _find_from_order_journal(orders_dir, symbol),
            _find_from_logs_live_orders(logs_live_dir, symbol),
            _find_from_executed_signals(signals_dir, symbol),
        ) if c is not None
    ]
    if high_conf:
        return max(high_conf, key=lambda c: c["entry_date"])

    known_sells = _collect_known_sell_dates(signals_dir, symbol)
    return _find_from_execution_quality(exec_quality_dir, symbol, known_sells)


# ======================================================================
# ATR再計算（最終手段・entry_dateが判明している場合のみ意味を持つ）
# ======================================================================
def _recompute_atr20_via_yfinance(symbol: str, as_of_date: str) -> float:
    """
    ATR20 がどのソースにも残っていない場合の最終手段。
    yfinance から as_of_date までの日足を取得し、他コード箇所
    （signal_bridge.py::_build_orders 等）と同一の True Range 平均で再計算する。
    ネットワーク/データ取得失敗時は 0.0 を返す（呼び出し側が監査ログに記録する）。
    """
    if not as_of_date:
        return 0.0
    try:
        import pandas as pd
        import yfinance as yf
        end = (pd.Timestamp(as_of_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if df is None or df.empty or len(df) < 20:
            return 0.0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"]  - df["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(20).mean().iloc[-1])
        return atr if atr > 0 else 0.0
    except Exception as exc:
        logger.warning("[ATR_RECOVERY] %s: yfinance再計算失敗: %s", symbol, exc)
        return 0.0


# ======================================================================
# entry_rsr リカバリ（既存ロジック・entry_date伝播バグ修正後は成功率が上がる想定）
# ======================================================================
def _find_entry_rsr_from_signal_log(
    signals_dir: Path, symbol: str, entry_date: str,
) -> float:
    """
    data/signals/signal_{YYYYMMDD}_*.json （日次シグナルJSON、"signals"配列に
    全銘柄のその日のRSRが記録されている）から、指定銘柄のBUY発注が実際に
    含まれる run を特定し、同一runの"signals"配列から entry 時点の RSR を
    復元する。"orders" 配列でBUY/SHADOW_BUYを確認したrunに限定することで、
    複数run（DRY再実行等）間のRSR揺れによる誤復元を避ける。
    見つからない場合は 0.0（呼び出し側は proxy フォールバックを継続する）。
    """
    if not signals_dir.exists() or not entry_date:
        return 0.0
    date_compact = entry_date.replace("-", "")
    for path in sorted(signals_dir.glob(f"signal_{date_compact}_*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        orders = d.get("orders") or []
        has_buy = any(
            o.get("symbol") == symbol and o.get("side") in ("BUY", "SHADOW_BUY")
            for o in orders
        )
        if not has_buy:
            continue
        for s in d.get("signals") or []:
            if s.get("symbol") == symbol:
                rsr = float(s.get("rsr", 0.0) or 0.0)
                if rsr > 0:
                    return rsr
    return 0.0


def recover_missing_entry_rsr(
    state: dict,
    signals_dir: Path,
    audit_log_path: Path,
) -> dict:
    """
    保有中の全銘柄について position_entry_rsrs 欠落（Study57/58A Quality
    Replacement Engine が "entry_rsr missing (pre-v3 position)" として
    current RSR を proxy 使用してしまう原因）を、日次シグナルJSONから
    復元する。復元できた銘柄のみ書き込み、復元不能な銘柄は何もしない
    （既存の QR 側 proxy フォールバックに委ねる — 推測でのRSR書き込みはしない）。

    Returns: {"recovered": [...], "unrecoverable": [...]}
    """
    pos_qtys        = state.get("position_qtys", {})
    pos_entry_dates = state.get("position_entry_dates", {})
    pos_entry_rsrs  = state.setdefault("position_entry_rsrs", {})

    recovered: list[dict] = []
    unrecoverable: list[dict] = []

    for sym, qty in pos_qtys.items():
        if int(qty) <= 0:
            continue
        if float(pos_entry_rsrs.get(sym, 0.0) or 0.0) > 0:
            continue  # 既に記録済み

        entry_date = pos_entry_dates.get(sym, "")
        rsr = _find_entry_rsr_from_signal_log(signals_dir, sym, entry_date)
        if rsr > 0:
            pos_entry_rsrs[sym] = rsr
            recovered.append({"symbol": sym, "entry_date": entry_date, "entry_rsr": rsr})
            logger.warning(
                "[ENTRY_RSR_RECOVERED] %s: entry_date=%s entry_rsr=%.1f を日次シグナルJSONから復元",
                sym, entry_date, rsr,
            )
        else:
            unrecoverable.append({"symbol": sym, "entry_date": entry_date})
            logger.warning(
                "[ENTRY_RSR_UNRECOVERABLE] %s: entry_date=%s のシグナルJSONにBUY記録が"
                "見つからないためentry_rsrを復元できない。QR側のcurrent RSR proxyが継続使用される。",
                sym, entry_date or "unknown",
            )

    if recovered or unrecoverable:
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "run_at":        datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
                "recovered":     recovered,
                "unrecoverable": unrecoverable,
            }, ensure_ascii=False) + "\n")

    return {"recovered": recovered, "unrecoverable": unrecoverable}


# ======================================================================
# メイン: entry_date / entry_price / entry_atr20 リカバリ
# ======================================================================
def recover_missing_entry_metadata(
    state: dict,
    logs_live_dir: Path,
    audit_log_path: Path,
    *,
    orders_dir: "Path | None" = None,
    signals_dir: "Path | None" = None,
    exec_quality_dir: "Path | None" = None,
) -> dict:
    """
    state（portfolio_state dict, in-place で更新）の中から
    entry_date欠落 または entry_price<=0 または entry_atr 欠落の保有銘柄を検出し、
    4ソース横断で復元を試みる（2026-07-15全面改訂・詳細はモジュールdocstring参照）。

    Returns: {"recovered": [...], "unrecoverable": [...], "atr_only_recovered": [...]}
    """
    from src.paths import RUNTIME_DIR as _default_runtime, SIGNALS_DIR as _default_signals, LOGS_DIR as _default_logs
    orders_dir       = orders_dir or (_default_runtime / "orders")
    signals_dir      = signals_dir or _default_signals
    exec_quality_dir = exec_quality_dir or (_default_logs / "execution_quality")

    pos_qtys           = state.get("position_qtys", {})
    pos_entry_prices    = state.setdefault("position_entry_prices",   {})
    pos_entry_atrs      = state.setdefault("position_entry_atrs",     {})
    pos_highest_closes  = state.setdefault("position_highest_closes", {})
    pos_entry_dates     = state.setdefault("position_entry_dates",    {})
    pos_strategy_types  = state.setdefault("position_strategy_types", {})
    missing_registry    = state.setdefault("entry_metadata_missing",  {})

    today_str = datetime.now(JST).strftime("%Y-%m-%d")
    recovered: list[dict] = []
    atr_only_recovered: list[dict] = []
    unrecoverable: list[dict] = []

    for sym in list(pos_qtys.keys()):
        entry_price = float(pos_entry_prices.get(sym, 0.0) or 0.0)
        has_date    = bool(pos_entry_dates.get(sym))
        has_atr     = float(pos_entry_atrs.get(sym, 0.0) or 0.0) > 0
        if entry_price > 0 and has_date and has_atr:
            continue  # 正常な保有 — 何もしない

        prior_entry = missing_registry.get(sym, {})
        detected_at = prior_entry.get("detected_at") or today_str  # Phase4: 初回検出日を保持
        attempts    = int(prior_entry.get("recovery_attempts", 0)) + 1

        best = _find_best_entry_metadata(
            sym, orders_dir=orders_dir, logs_live_dir=logs_live_dir,
            signals_dir=signals_dir, exec_quality_dir=exec_quality_dir,
        )

        if best is not None:
            if not has_date:
                pos_entry_dates[sym] = best["entry_date"]
            if entry_price <= 0:
                pos_entry_prices[sym] = best["estimated_price"]
                if float(pos_highest_closes.get(sym, 0.0) or 0.0) <= 0:
                    pos_highest_closes[sym] = best["estimated_price"]
            if best.get("strategy_type") and sym not in pos_strategy_types:
                pos_strategy_types[sym] = best["strategy_type"]

            resolved_date = pos_entry_dates.get(sym, best["entry_date"])
            atr_recovered_now = False
            if not has_atr:
                if best["atr20"] > 0:
                    pos_entry_atrs[sym] = best["atr20"]
                    atr_recovered_now = True
                else:
                    atr_recomputed = _recompute_atr20_via_yfinance(sym, resolved_date)
                    if atr_recomputed > 0:
                        pos_entry_atrs[sym] = atr_recomputed
                        atr_recovered_now = True
                        logger.warning(
                            "[ENTRY_METADATA_ATR_RECOVERED] %s: yfinance再計算でATR20=%.2fを復元"
                            " (entry_date=%s)", sym, atr_recomputed, resolved_date,
                        )

            missing_registry.pop(sym, None) if (
                pos_entry_prices.get(sym, 0) and pos_entry_dates.get(sym) and pos_entry_atrs.get(sym, 0) > 0
            ) else None

            entry = {
                "symbol": sym, "entry_date": pos_entry_dates.get(sym, ""),
                "estimated_price": pos_entry_prices.get(sym, 0.0),
                "atr20": pos_entry_atrs.get(sym, 0.0),
                "confidence": best["confidence"], "source": best["source"],
                "source_file": best["source_file"],
            }
            recovered.append(entry)
            logger.warning(
                "[ENTRY_METADATA_RECOVERED] %s: entry_date=%s entry_price=%.2f atr20=%.2f "
                "confidence=%s source=%s",
                sym, entry["entry_date"], entry["estimated_price"], entry["atr20"],
                entry["confidence"], entry["source"],
            )

            if sym in missing_registry:
                # 部分復元（ATRのみ未解決等）: 理由を細分化して更新
                if not pos_entry_atrs.get(sym, 0):
                    reason = (
                        "atr_source_missing_but_date_known"
                        if pos_entry_dates.get(sym) else "entry_date_unknown"
                    )
                    missing_registry[sym] = {
                        "detected_at": detected_at, "entry_date": pos_entry_dates.get(sym, ""),
                        "qty": pos_qtys.get(sym, 0), "reason": reason,
                        "recovery_attempts": attempts,
                    }
                    logger.error(
                        "[ENTRY_METADATA_ATR_UNAVAILABLE] %s: reason=%s (attempts=%d) "
                        "— entry_price/dateは復元済みのため自動売買は継続する。",
                        sym, reason, attempts,
                    )
            elif atr_recovered_now:
                atr_only_recovered.append(entry)
        else:
            missing_registry[sym] = {
                "detected_at": detected_at, "entry_date": "",
                "qty": pos_qtys.get(sym, 0),
                "reason": "no_matching_buy_record_in_any_source"
                          "（order_journal/logs_live/executed_signals/execution_quality 全て探索済み）",
                "recovery_attempts": attempts,
            }
            unrecoverable.append({"symbol": sym, "attempts": attempts})
            logger.error(
                "[ENTRY_METADATA_UNRECOVERABLE] %s: 4ソース全て探索したが発見できず"
                "（attempts=%d・初回検出=%s）。entry_date=Unknown。自動売買は継続する。",
                sym, attempts, detected_at,
            )

    if recovered or unrecoverable:
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "run_at":              datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
                "recovered":           recovered,
                "atr_only_recovered":  atr_only_recovered,
                "unrecoverable":       unrecoverable,
            }, ensure_ascii=False) + "\n")

    return {"recovered": recovered, "unrecoverable": unrecoverable, "atr_only_recovered": atr_only_recovered}


def main() -> int:
    import argparse
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from src.paths import RUNTIME_DIR, LOGS_DIR, SIGNALS_DIR
    from src.portfolio.state_store import load_portfolio_state, save_portfolio_state

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="portfolio_state.jsonへ書き込まずプレビューのみ")
    args = parser.parse_args()

    state_path = RUNTIME_DIR / "portfolio_state.json"
    state, _vr = load_portfolio_state(state_path)
    result = recover_missing_entry_metadata(
        state,
        logs_live_dir=LOGS_DIR / "live",
        audit_log_path=LOGS_DIR / "entry_metadata_recovery_audit.jsonl",
    )
    rsr_result = recover_missing_entry_rsr(
        state,
        signals_dir=SIGNALS_DIR,
        audit_log_path=LOGS_DIR / "entry_rsr_recovery_audit.jsonl",
    )
    print(json.dumps({"entry_metadata": result, "entry_rsr": rsr_result}, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("\n[DRY_RUN] portfolio_state.json への書き込みはスキップしました。")
        return 0

    if result["recovered"] or result["unrecoverable"]:
        save_portfolio_state(state, path=state_path, data_source="internal")
    if rsr_result["recovered"]:
        save_portfolio_state(state, path=state_path, data_source="internal")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
