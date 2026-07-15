import time
from enum import Enum
from typing import Set, Tuple

class RuntimeState(Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    REPLAYING = "REPLAYING"
    QUARANTINED = "QUARANTINED"

ALLOWED_TRANSITIONS: Set[Tuple[RuntimeState, RuntimeState]] = frozenset({
    (RuntimeState.PENDING,     RuntimeState.READY),
    (RuntimeState.PENDING,     RuntimeState.BLOCKED),
    (RuntimeState.READY,       RuntimeState.RUNNING),
    (RuntimeState.READY,       RuntimeState.BLOCKED),
    (RuntimeState.RUNNING,     RuntimeState.VALIDATING),
    (RuntimeState.RUNNING,     RuntimeState.FAILED),
    (RuntimeState.RUNNING,     RuntimeState.BLOCKED),
    (RuntimeState.VALIDATING,  RuntimeState.COMMITTED),
    (RuntimeState.VALIDATING,  RuntimeState.QUARANTINED),
    (RuntimeState.VALIDATING,  RuntimeState.FAILED),
    (RuntimeState.FAILED,      RuntimeState.PENDING),
    (RuntimeState.FAILED,      RuntimeState.QUARANTINED),
    (RuntimeState.BLOCKED,     RuntimeState.PENDING),
    (RuntimeState.COMMITTED,   RuntimeState.REPLAYING),
    (RuntimeState.REPLAYING,   RuntimeState.COMMITTED),
    (RuntimeState.REPLAYING,   RuntimeState.FAILED),
    (RuntimeState.QUARANTINED, RuntimeState.PENDING),
})

class RuntimeStateMachine:
    def __init__(self, initial: RuntimeState = RuntimeState.PENDING):
        self._state = initial
        self._history: list = []

    @property
    def state(self) -> RuntimeState:
        return self._state

    def transition(self, to: RuntimeState, reason: str) -> None:
        if (self._state, to) not in ALLOWED_TRANSITIONS:
            raise ValueError(f"Illegal transition {self._state} -> {to}: {reason}")
        self._history.append({"from": self._state.value, "to": to.value,
                               "reason": reason, "ts": time.time()})
        self._state = to

    def history(self) -> list:
        return list(self._history)
