"""
src/live/broker_worker.py

Standalone broker API worker — runs as a child process.

Invoked by ProcessSupervisor as:
    python -m src.live.broker_worker --input <orders.json> --output <results.json>

Protocol:
    Input file (JSON):
        {
          "orders": [
            {
              "symbol_4digit": "7203",
              "symbol": "7203.T",
              "side": "BUY",
              "qty": 100,
              "front_order_type": 13,
              "exchange": 9,
              "estimated_price": 2500.0,
              "strategy_type": "fujiko",
              "client_order_id": "COI-A1B2C3D4E5F60718"
            }
          ]
        }

    Output file (JSON):
        {
          "results": [
            {
              "symbol": "7203.T",
              "side": "BUY",
              "qty": 100,
              "success": true,
              "order_id": "20260515A02N80057402",
              "result_code": 0,
              "client_order_id": "COI-A1B2C3D4E5F60718",
              "error": null
            }
          ],
          "worker_pid": 12345
        }

Design:
    - No shared state with parent process
    - Parent inherits env vars → KabuClient reads KABU_API_PASSWORD etc.
    - Rate limiting between orders: ORDER_RATE_LIMIT_SEC env var (default 5s)
    - Exits 0 on success, 1 on unrecoverable error
    - Output file written atomically (temp + rename) to prevent partial reads
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path when run as -m src.live.broker_worker
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("broker_worker")

_RATE_LIMIT_SEC = float(os.environ.get("ORDER_RATE_LIMIT_SEC", "5"))


def _front_order_type_for_now() -> tuple[int, bool]:
    """
    Return (front_order_type, skip) based on current JST time.
    Mirrors the logic in signal_bridge._send_orders().
    """
    import datetime as _dt
    jst = _dt.timezone(_dt.timedelta(hours=9))
    now = _dt.datetime.now(jst)
    t   = now.hour * 60 + now.minute

    _AM_OPEN      = 9  * 60
    _AM_CLOSE_PRE = 11 * 60 + 25
    _AM_CLOSE     = 11 * 60 + 30
    _PM_OPEN      = 12 * 60 + 30
    _PM_CLOSE_PRE = 15 * 60 + 25
    _PM_CLOSE     = 15 * 60 + 30

    if t < _AM_OPEN:
        return 13, False   # 寄成前場
    if t < _AM_CLOSE_PRE:
        return 10, False   # 成行
    if t < _AM_CLOSE:
        return 14, False   # 引成前場
    if t < _PM_OPEN:
        return 10, True    # 昼休み — skip
    if t < _PM_CLOSE_PRE:
        return 10, False   # 成行
    if t < _PM_CLOSE:
        return 16, False   # 引成後場
    return 10, True        # 後場終了 — skip


def _submit_single_order(client, order: dict, front_order_type: int) -> dict:
    """
    Call client.send_order() for one order dict.
    Returns a result dict with success/order_id/error fields.
    """
    from src.kabusapi.client import Side, Exchange

    symbol_4digit = order["symbol_4digit"]
    side_str      = order["side"]
    qty           = int(order["qty"])
    exchange      = int(order.get("exchange", Exchange.SOR))
    coi           = order.get("client_order_id", "")

    # Use front_order_type from order dict if provided (parent pre-computed),
    # otherwise fall back to the real-time calculation
    fot = int(order.get("front_order_type", front_order_type))

    base_result = {
        "symbol":          order.get("symbol", symbol_4digit),
        "symbol_4digit":   symbol_4digit,
        "side":            side_str,
        "qty":             qty,
        "client_order_id": coi,
        "success":         False,
        "order_id":        None,
        "result_code":     None,
        "error":           None,
        # SAFETY FIX (entry-metadata-loss follow-up): forward through whatever
        # the parent already put into the input `order` dict (serialize_order()
        # includes estimated_price/strategy_type). This is defense-in-depth —
        # the parent (_submit_orders_process_isolated) also restores these
        # directly from its own order objects, which is the authoritative fix.
        "estimated_price": float(order.get("estimated_price", 0.0)),
        "strategy_type":   order.get("strategy_type", ""),
    }

    # SAFETY FIX (2026-07-07 incident): this is the actual broker-facing
    # order path (process-isolated live execution). It previously mapped
    # side="SHADOW_BUY" (meant to be observation_only) to Side.BUY, which is
    # exactly what sent the unauthorized 5301.T order to kabu API. Only
    # "BUY"/"SELL" are ever sent; anything else fails closed with no API call.
    if side_str == "BUY":
        side_code = Side.BUY
    elif side_str == "SELL":
        side_code = Side.SELL
    else:
        base_result["error"] = f"rejected_unknown_side: {side_str}"
        logger.error("[ORDER_SIDE_GUARD] unknown side=%s symbol=%s → 発注スキップ（fail-closed）",
                     side_str, symbol_4digit)
        return base_result

    try:
        result = client.send_order(
            symbol     = symbol_4digit,
            exchange   = exchange,
            side       = side_code,
            qty        = qty,
            order_type = fot,
        )
        base_result.update({
            "success":     result.success,
            "order_id":    result.order_id,
            "result_code": result.result_code,
        })
        logger.info(
            "ORDER %s %s %s qty=%d success=%s order_id=%s",
            side_str, symbol_4digit, coi, qty, result.success, result.order_id,
        )
    except Exception as exc:
        base_result["error"] = str(exc)[:400]
        logger.error(
            "ORDER FAILED %s %s: %s", side_str, symbol_4digit, exc
        )

    return base_result


def run(input_path: Path, output_path: Path) -> int:
    """Main worker logic. Returns exit code."""
    # ── ENTRY FREEZE 起動時ログ（資産保全・2026-07-17・defense-in-depth）──
    # 子プロセスは親とメモリ状態を共有しないため、ここで自前に設定を読み直してログする。
    try:
        from src.config_loader import load_strategy_config
        _ef = load_strategy_config().entry_freeze
        logger.warning(
            "[ENTRY_FREEZE_STATE] entry_freeze_enabled=%s reason=%s pid=%d",
            _ef.enabled, _ef.reason, os.getpid(),
        )
    except Exception as _ef_err:
        logger.warning("[ENTRY_FREEZE_STATE] config読込失敗・freeze状態不明: %s", _ef_err)

    # Load orders
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to read input file: %s", exc)
        return 1

    orders = payload.get("orders", [])
    if not orders:
        logger.warning("No orders in input file")
        output_path.write_text(
            json.dumps({"results": [], "worker_pid": os.getpid()}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0

    # Authenticate
    try:
        from src.kabusapi.client import KabuClient
        client = KabuClient()
        client.fetch_token()
    except Exception as exc:
        logger.error("KabuClient auth failed: %s", exc)
        results = [{
            **{k: o.get(k, "") for k in ("symbol", "side", "qty", "client_order_id")},
            "success":     False,
            "order_id":    None,
            "result_code": None,
            "error":       f"auth_failed: {exc}",
        } for o in orders]
        _write_output(output_path, results)
        return 1

    # Compute front_order_type (overridden per-order if parent pre-computed it)
    fot, skip = _front_order_type_for_now()
    if skip:
        logger.warning("Market closed (fot=%d skip=True) — skipping all orders", fot)
        results = [{
            **{k: o.get(k, "") for k in ("symbol", "side", "qty", "client_order_id")},
            "success":     False,
            "order_id":    None,
            "result_code": None,
            "error":       "market_closed",
        } for o in orders]
        _write_output(output_path, results)
        return 0

    # Submit orders with rate limiting
    results = []
    for idx, order in enumerate(orders):
        if idx > 0:
            time.sleep(_RATE_LIMIT_SEC)
        result = _submit_single_order(client, order, fot)
        results.append(result)

    _write_output(output_path, results)
    return 0


def _write_output(output_path: Path, results: list) -> None:
    """Write results atomically: temp file → rename."""
    payload = json.dumps(
        {"results": results, "worker_pid": os.getpid()},
        ensure_ascii=False,
        indent=2,
    )
    tmp = output_path.with_suffix(".tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(output_path)
    except Exception as exc:
        logger.error("Failed to write output: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Broker API worker (child process)")
    parser.add_argument("--input",  required=True, help="Path to orders JSON file")
    parser.add_argument("--output", required=True, help="Path to results JSON file")
    args = parser.parse_args()

    sys.exit(run(Path(args.input), Path(args.output)))


if __name__ == "__main__":
    main()
