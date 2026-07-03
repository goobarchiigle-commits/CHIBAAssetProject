"""
position_sync.py
起動時に broker のポジションと portfolio_state.json を比較し、
差分（drift）があれば発注を停止する。
"""
import logging
from typing import TYPE_CHECKING, Dict, Set

from src.kabusapi.signal_bridge import AbortError

if TYPE_CHECKING:
    from src.kabusapi.signal_bridge import SignalBridge

logger = logging.getLogger(__name__)


def sync_and_validate_state(bridge: "SignalBridge") -> dict:
    """
    1. broker から現在のポジションを取得（API call）
    2. portfolio_state.json のローカル state と比較
    3. 差分があれば:
       a) ローカル state を broker の値で上書き（sync）
       b) drift_detected=True を返す
       c) 差分が「ブローカー側に存在してローカルにない」場合は halt_trading=True

    Returns:
        {
            "broker_syms": set,
            "local_syms": set,
            "drift_detected": bool,
            "halt_trading": bool,   # True の場合は発注停止
            "synced": bool,
        }
    """
    try:
        broker_positions: Dict[str, int] = bridge.get_broker_positions()
        broker_syms: Set[str] = set(broker_positions.keys())

        local_state: Dict[str, int] = bridge.get_local_positions()
        local_syms: Set[str] = set(local_state.keys())

        drift_detected = broker_syms != local_syms
        untracked = broker_syms - local_syms

        if drift_detected:
            logger.warning(
                "[POSITION_SYNC] Drift detected. "
                "broker=%s, local=%s, untracked=%s",
                sorted(broker_syms),
                sorted(local_syms),
                sorted(untracked),
            )
            bridge.overwrite_local_positions(broker_positions)
            logger.info("[POSITION_SYNC] Local state overwritten with broker data.")

        if untracked:
            raise AbortError(
                "position_sync_fail",
                "position sync found untracked broker positions: "
                f"broker={sorted(broker_syms)} local={sorted(local_syms)}",
            )

        logger.info("[POSITION_SYNC] State OK. positions=%s", sorted(broker_syms))
        return {
            "broker_syms": broker_syms,
            "local_syms": local_syms,
            "drift_detected": drift_detected,
            "synced": drift_detected,
        }
    except AbortError:
        raise
    except Exception as e:
        logger.error("[POSITION_SYNC] Failed to sync: %s", e)
        raise AbortError("position_sync_fail", f"failed to sync positions: {e}") from e
