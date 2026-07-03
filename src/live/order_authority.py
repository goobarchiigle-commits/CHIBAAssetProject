"""
src/live/order_authority.py  —  Signal→Intent→Authority→Execution Hash Chain

Deterministic hash chain linking every order to its originating signal.
All hashes must survive reconciliation (re-computed from same inputs = same hash).

Chain:
  signal_hash    = sha256(symbol | date | rsr | side | strategy_id)
  intent_hash    = sha256(signal_hash | capital | qty | price_limit | ts_intent)
  authority_hash = sha256(intent_hash | authority_level | ts_authority)
  execution_hash = sha256(authority_hash | fill_price | fill_qty | broker_ref)
  chain_hash     = sha256(signal_hash | intent_hash | authority_hash | execution_hash)

State integrity invariant:
  chain_hash recomputed from stored fields == chain_hash stored in record.

Append-only audit JSONL: logs/authority_audit.jsonl
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
STRATEGY_ID = "study9_caseb_v1"   # frozen identifier

AUDIT_LOG = Path("logs/authority_audit.jsonl")


# ──────────────────────────────────────────────────────────────────────
#  Hash Helpers
# ──────────────────────────────────────────────────────────────────────

def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canon(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────
#  Signal Record
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    signal_hash:  str
    symbol:       str
    date:         str
    rsr:          float
    side:         str          # BUY | SELL
    strategy_id:  str = STRATEGY_ID
    d90:          int = 0
    slope5:       float = 0.0
    ts_signal:    str = ""
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def build(symbol: str, date: str, rsr: float, side: str,
              d90: int = 0, slope5: float = 0.0) -> "SignalRecord":
        ts = _now_iso()
        h  = _sha256(_canon({
            "symbol": symbol, "date": date,
            "rsr": round(rsr, 4), "side": side,
            "strategy_id": STRATEGY_ID,
        }))
        return SignalRecord(signal_hash=h, symbol=symbol, date=date,
                            rsr=rsr, side=side, d90=d90, slope5=slope5,
                            ts_signal=ts)

    def recompute_hash(self) -> str:
        return _sha256(_canon({
            "symbol": self.symbol, "date": self.date,
            "rsr": round(self.rsr, 4), "side": self.side,
            "strategy_id": self.strategy_id,
        }))

    def hash_valid(self) -> bool:
        return self.signal_hash == self.recompute_hash()


# ──────────────────────────────────────────────────────────────────────
#  Intent Record
# ──────────────────────────────────────────────────────────────────────

@dataclass
class IntentRecord:
    intent_hash:  str
    signal_hash:  str
    symbol:       str
    side:         str
    qty:          int
    price_limit:  float        # 0.0 = market order
    capital:      float
    ts_intent:    str = ""
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def build(signal: SignalRecord, qty: int,
              price_limit: float, capital: float) -> "IntentRecord":
        ts = _now_iso()
        h  = _sha256(_canon({
            "signal_hash": signal.signal_hash,
            "capital": round(capital, 0),
            "qty": qty, "price_limit": round(price_limit, 2),
        }))
        return IntentRecord(intent_hash=h, signal_hash=signal.signal_hash,
                            symbol=signal.symbol, side=signal.side,
                            qty=qty, price_limit=price_limit,
                            capital=capital, ts_intent=ts)

    def recompute_hash(self) -> str:
        return _sha256(_canon({
            "signal_hash": self.signal_hash,
            "capital": round(self.capital, 0),
            "qty": self.qty, "price_limit": round(self.price_limit, 2),
        }))

    def hash_valid(self) -> bool:
        return self.intent_hash == self.recompute_hash()


# ──────────────────────────────────────────────────────────────────────
#  Authority Record
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AuthorityRecord:
    authority_hash:  str
    intent_hash:     str
    authority_level: str
    approved:        bool
    reject_reason:   str = ""
    ts_authority:    str = ""
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def build(intent: IntentRecord, authority_level: str,
              approved: bool, reject_reason: str = "") -> "AuthorityRecord":
        ts = _now_iso()
        h  = _sha256(_canon({
            "intent_hash": intent.intent_hash,
            "authority_level": authority_level,
            "approved": approved,
        }))
        return AuthorityRecord(authority_hash=h, intent_hash=intent.intent_hash,
                               authority_level=authority_level,
                               approved=approved, reject_reason=reject_reason,
                               ts_authority=ts)

    def recompute_hash(self) -> str:
        return _sha256(_canon({
            "intent_hash": self.intent_hash,
            "authority_level": self.authority_level,
            "approved": self.approved,
        }))

    def hash_valid(self) -> bool:
        return self.authority_hash == self.recompute_hash()


# ──────────────────────────────────────────────────────────────────────
#  Execution Record
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionRecord:
    execution_hash:   str
    authority_hash:   str
    chain_hash:       str
    signal_hash:      str
    symbol:           str
    side:             str
    fill_price:       float
    fill_qty:         int
    requested_qty:    int
    broker_ref:       str
    fill_ratio:       float
    partial_fill_gap: int
    cash_before:      float
    cash_after:       float
    position_before:  int
    position_after:   int
    broker_latency_ms: float
    cancel_count:     int = 0
    ts_execution:     str = ""
    authority_level:  str = ""
    case_id:          str = ""
    schema_version:   int = SCHEMA_VERSION

    @staticmethod
    def build(
        authority:    AuthorityRecord,
        signal:       SignalRecord,
        intent:       IntentRecord,
        fill_price:   float,
        fill_qty:     int,
        broker_ref:   str,
        cash_before:  float,
        cash_after:   float,
        position_before: int,
        position_after: int,
        broker_latency_ms: float = 0.0,
        cancel_count: int = 0,
        case_id: str = "",
    ) -> "ExecutionRecord":
        ts = _now_iso()
        ex_h = _sha256(_canon({
            "authority_hash": authority.authority_hash,
            "fill_price": round(fill_price, 2),
            "fill_qty": fill_qty,
            "broker_ref": broker_ref,
        }))
        chain_h = _sha256(_canon({
            "signal_hash":    signal.signal_hash,
            "intent_hash":    intent.intent_hash,
            "authority_hash": authority.authority_hash,
            "execution_hash": ex_h,
        }))
        partial_gap = intent.qty - fill_qty
        fill_ratio  = fill_qty / max(1, intent.qty)
        return ExecutionRecord(
            execution_hash=ex_h, authority_hash=authority.authority_hash,
            chain_hash=chain_h, signal_hash=signal.signal_hash,
            symbol=signal.symbol, side=signal.side,
            fill_price=fill_price, fill_qty=fill_qty,
            requested_qty=intent.qty,
            broker_ref=broker_ref, fill_ratio=fill_ratio,
            partial_fill_gap=partial_gap,
            cash_before=cash_before, cash_after=cash_after,
            position_before=position_before, position_after=position_after,
            broker_latency_ms=broker_latency_ms, cancel_count=cancel_count,
            ts_execution=ts, authority_level=authority.authority_level,
            case_id=case_id,
        )

    def recompute_execution_hash(self) -> str:
        return _sha256(_canon({
            "authority_hash": self.authority_hash,
            "fill_price": round(self.fill_price, 2),
            "fill_qty": self.fill_qty,
            "broker_ref": self.broker_ref,
        }))

    def recompute_chain_hash(self) -> str:
        return _sha256(_canon({
            "signal_hash":    self.signal_hash,
            "intent_hash":    "",   # not stored in exec; pass separately
            "authority_hash": self.authority_hash,
            "execution_hash": self.execution_hash,
        }))

    def chain_valid(self, intent_hash: str) -> bool:
        ex_ok = self.execution_hash == self.recompute_execution_hash()
        ch    = _sha256(_canon({
            "signal_hash":    self.signal_hash,
            "intent_hash":    intent_hash,
            "authority_hash": self.authority_hash,
            "execution_hash": self.execution_hash,
        }))
        return ex_ok and (self.chain_hash == ch)


# ──────────────────────────────────────────────────────────────────────
#  Audit Log
# ──────────────────────────────────────────────────────────────────────

def _audit_record(
    signal:    SignalRecord,
    intent:    IntentRecord,
    authority: AuthorityRecord,
    execution: ExecutionRecord,
    chain_valid: bool,
    idempotency_key: str,
) -> dict:
    return {
        "ts":              execution.ts_execution,
        "case_id":         execution.case_id,
        "symbol":          signal.symbol,
        "date":            signal.date,
        "side":            signal.side,
        "authority_level": authority.authority_level,
        "signal_hash":     signal.signal_hash,
        "intent_hash":     intent.intent_hash,
        "authority_hash":  authority.authority_hash,
        "execution_hash":  execution.execution_hash,
        "chain_hash":      execution.chain_hash,
        "chain_valid":     chain_valid,
        "approved":        authority.approved,
        "reject_reason":   authority.reject_reason,
        "fill_price":      execution.fill_price,
        "fill_qty":        execution.fill_qty,
        "requested_qty":   execution.requested_qty,
        "fill_ratio":      round(execution.fill_ratio, 4),
        "partial_fill_gap":execution.partial_fill_gap,
        "cash_before":     round(execution.cash_before, 0),
        "cash_after":      round(execution.cash_after, 0),
        "position_before": execution.position_before,
        "position_after":  execution.position_after,
        "broker_latency_ms": execution.broker_latency_ms,
        "cancel_count":    execution.cancel_count,
        "idempotency_key": idempotency_key,
        "schema_version":  SCHEMA_VERSION,
    }


def append_audit(record: dict, path: Path = AUDIT_LOG) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("[ORDER_AUTHORITY] audit write failed: %s", e)


def load_audit(path: Path = AUDIT_LOG) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows


# ──────────────────────────────────────────────────────────────────────
#  Idempotency Registry
# ──────────────────────────────────────────────────────────────────────

class IdempotencyRegistry:
    """Prevents same signal from producing duplicate orders."""

    def __init__(self, path: Path = Path("runtime/authority_idem.json")):
        self._path = path
        self._seen: set[str] = self._load()

    def _load(self) -> set[str]:
        if not self._path.exists():
            return set()
        try:
            return set(json.loads(self._path.read_text(encoding="utf-8")).get("keys", []))
        except Exception:
            return set()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"keys": sorted(self._seen)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def key(self, signal: SignalRecord) -> str:
        return f"{signal.symbol}|{signal.date}|{signal.side}|{signal.signal_hash[:8]}"

    def check_and_register(self, signal: SignalRecord) -> tuple[bool, str]:
        """Returns (is_duplicate, key)."""
        k = self.key(signal)
        if k in self._seen:
            return True, k
        self._seen.add(k)
        self._save()
        return False, k


# ──────────────────────────────────────────────────────────────────────
#  Integrity Validator
# ──────────────────────────────────────────────────────────────────────

def validate_chain(audit_rows: list[dict]) -> dict:
    """Validate hash chain integrity across all audit records."""
    n_total    = len(audit_rows)
    n_valid    = 0
    n_invalid  = 0
    failed     = []

    for r in audit_rows:
        if not r.get("chain_valid", False):
            n_invalid += 1
            failed.append({
                "ts": r.get("ts"), "symbol": r.get("symbol"),
                "chain_hash": r.get("chain_hash"),
                "reason": "chain_valid=False in log",
            })
        else:
            # Re-verify execution_hash from stored fields
            ex_h = _sha256(_canon({
                "authority_hash": r.get("authority_hash",""),
                "fill_price": round(r.get("fill_price",0.0), 2),
                "fill_qty": r.get("fill_qty", 0),
                "broker_ref": r.get("broker_ref",""),
            }))
            if ex_h != r.get("execution_hash",""):
                n_invalid += 1
                failed.append({
                    "ts": r.get("ts"), "symbol": r.get("symbol"),
                    "reason": "execution_hash_recompute_mismatch",
                })
            else:
                n_valid += 1

    return {
        "n_total":        n_total,
        "n_valid":        n_valid,
        "n_invalid":      n_invalid,
        "replay_consistency": round(n_valid / max(1, n_total), 4),
        "failed_records": failed[:10],  # cap at 10 for report
    }
