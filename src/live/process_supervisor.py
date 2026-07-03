"""
src/live/process_supervisor.py

Process-based broker isolation for LIVE order execution.

Replaces the ThreadPoolExecutor broker stage with a true child-process
boundary.  On timeout, the child is terminated then killed — eliminating
the "ghost execution" risk where a hung broker API call lingers after the
parent has moved on.

Why process, not thread?
  Threads cannot be forcibly killed from Python (only daemon threads are
  abandoned at interpreter exit).  A hung kabu API call in a thread keeps
  the socket open and may eventually send an order the parent no longer
  expects.  A child process is terminated via OS primitives (TerminateProcess
  on Windows) giving a hard guarantee: no lingering HTTP calls after timeout.

Windows-compatible:
  Uses subprocess.Popen with --input/--output JSON files rather than
  multiprocessing (avoids spawn pickling issues with complex objects).

Hard-kill sequence on timeout:
    1. proc.terminate()  → SIGTERM / TerminateProcess
    2. proc.wait(2s)     → grace period
    3. proc.kill()       → SIGKILL / TerminateProcess (forceful)
    4. raise StageTimeout
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.live.staged_supervisor import StageTimeout, StageError

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# Default timeout for the entire order_execution stage
DEFAULT_ORDER_TIMEOUT_SEC: int = 30

# Grace period between terminate() and kill() (seconds)
_TERMINATE_GRACE_SEC: float = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# FrontOrderType helper (mirrors signal_bridge logic)
# ──────────────────────────────────────────────────────────────────────────────

def compute_front_order_type() -> tuple[int, bool]:
    """
    Return (front_order_type, skip_orders) based on current JST time.
    skip_orders=True means the market is closed; orders should not be sent.

    Values:
      13 = 寄成前場 (pre-open market order, AM session)
      10 = 成行 (market order, during trading hours)
      14 = 引成前場 (close market order, AM session)
      16 = 引成後場 (close market order, PM session)
    """
    now = datetime.now(JST)
    t   = now.hour * 60 + now.minute

    if t < 9 * 60:
        return 13, False
    if t < 11 * 60 + 25:
        return 10, False
    if t < 11 * 60 + 30:
        return 14, False
    if t < 12 * 60 + 30:
        return 10, True    # 昼休み
    if t < 15 * 60 + 25:
        return 10, False
    if t < 15 * 60 + 30:
        return 16, False
    return 10, True        # 後場終了


# ──────────────────────────────────────────────────────────────────────────────
# Order serialization helpers
# ──────────────────────────────────────────────────────────────────────────────

def serialize_order(order, front_order_type: int, client_order_id: str) -> Dict:
    """
    Serialize an OrderInstruction (or duck-typed object) into a JSON-safe dict
    suitable for the broker_worker input protocol.
    """
    return {
        "symbol":          getattr(order, "symbol", ""),
        "symbol_4digit":   getattr(order, "symbol_4digit", getattr(order, "symbol", "")),
        "side":            getattr(order, "side", ""),
        "qty":             int(getattr(order, "qty", 0)),
        "front_order_type": front_order_type,
        "exchange":        9,    # SOR — always for sendorder
        "estimated_price": float(getattr(order, "estimated_price", 0.0)),
        "strategy_type":   getattr(order, "strategy_type", "unknown"),
        "client_order_id": client_order_id,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ProcessSupervisor
# ──────────────────────────────────────────────────────────────────────────────

class BrokerProcessSupervisor:
    """
    Executes broker order submission in an isolated child process.

    The child runs `python -m src.live.broker_worker` and communicates
    via temp JSON files.  On timeout the child is hard-killed via
    terminate() + kill() — guaranteeing no orphaned HTTP calls.

    Usage::

        supervisor = BrokerProcessSupervisor()
        results = supervisor.submit_orders(
            orders_dicts=[...],     # list of serialized order dicts
            timeout_sec=30,
            run_id="20260515_090000",
        )
    """

    def __init__(
        self,
        worker_module:   str   = "src.live.broker_worker",
        python_exe:      Optional[str] = None,
        tmp_dir:         Optional[Path] = None,
    ) -> None:
        self._module     = worker_module
        self._python     = python_exe or sys.executable
        self._tmp_dir    = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())

    def submit_orders(
        self,
        orders_dicts:  List[Dict],
        timeout_sec:   int  = DEFAULT_ORDER_TIMEOUT_SEC,
        run_id:        str  = "",
    ) -> List[Dict]:
        """
        Spawn child process to submit orders to broker API.

        Returns:
            List of result dicts with success/order_id/error fields.

        Raises:
            StageTimeout: child did not finish within timeout_sec.
            StageError:   child exited with non-zero status.
        """
        if not orders_dicts:
            return []

        ts        = datetime.now(JST).strftime("%Y%m%d_%H%M%S_%f")
        in_file   = self._tmp_dir / f"broker_in_{ts}.json"
        out_file  = self._tmp_dir / f"broker_out_{ts}.json"

        # Write input file
        try:
            in_file.write_text(
                json.dumps(
                    {"orders": orders_dicts, "run_id": run_id},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            raise StageError("order_execution", exc) from exc

        proc: Optional[subprocess.Popen] = None
        t0 = time.monotonic()

        try:
            proc = subprocess.Popen(
                [
                    self._python, "-m", self._module,
                    "--input",  str(in_file),
                    "--output", str(out_file),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
            )
            logger.info(
                "[PROC_SUPERVISOR] spawned broker worker pid=%d timeout=%ds run_id=%s",
                proc.pid, timeout_sec, run_id,
            )

            try:
                stdout, stderr = proc.communicate(timeout=float(timeout_sec))
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                logger.error(
                    "[PROC_SUPERVISOR] timeout after %.1fs — terminating pid=%d",
                    elapsed, proc.pid,
                )
                self._hard_kill(proc)
                raise StageTimeout("order_execution", elapsed, float(timeout_sec))

            elapsed = time.monotonic() - t0
            if proc.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace")[:500]
                logger.error(
                    "[PROC_SUPERVISOR] worker exited rc=%d stderr=%s",
                    proc.returncode, stderr_text,
                )
                raise StageError(
                    "order_execution",
                    RuntimeError(
                        f"broker worker exited rc={proc.returncode}: {stderr_text}"
                    ),
                )

            logger.info(
                "[PROC_SUPERVISOR] worker finished pid=%d elapsed=%.1fs",
                proc.pid, elapsed,
            )
            return self._read_output(out_file)

        finally:
            # Always clean up temp files
            for f in (in_file, out_file):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass

    def _hard_kill(self, proc: subprocess.Popen) -> None:
        """
        Terminate → wait grace period → kill.
        Deterministic shutdown sequence for Windows and POSIX.
        """
        try:
            proc.terminate()
        except OSError:
            pass

        try:
            proc.wait(timeout=_TERMINATE_GRACE_SEC)
        except subprocess.TimeoutExpired:
            logger.warning(
                "[PROC_SUPERVISOR] SIGTERM ignored — sending SIGKILL to pid=%d",
                proc.pid,
            )
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

        logger.info(
            "[PROC_SUPERVISOR] hard kill complete pid=%d rc=%s",
            proc.pid, proc.returncode,
        )

    def _read_output(self, out_file: Path) -> List[Dict]:
        """Read and return the results list from the worker output file."""
        if not out_file.exists():
            logger.error("[PROC_SUPERVISOR] output file missing: %s", out_file)
            return []
        try:
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            return payload.get("results", [])
        except Exception as exc:
            logger.error("[PROC_SUPERVISOR] output parse error: %s", exc)
            return []
