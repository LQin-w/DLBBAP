from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


class TrainingCancelledError(RuntimeError):
    def __init__(self, phase: str = "system", scope: str = "", checkpoint: str = "") -> None:
        self.phase = phase
        self.scope = scope
        self.checkpoint = checkpoint
        message = f"训练已停止 | phase={phase}"
        if scope:
            message += f" | scope={scope}"
        if checkpoint:
            message += f" | checkpoint={checkpoint}"
        super().__init__(message)


@dataclass(slots=True)
class TrainingControl:
    stop_event: threading.Event = field(default_factory=threading.Event)
    _phase: str = "system"
    _scope: str = "idle"
    _stop_logged: bool = False
    _run_started_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def request_stop(self) -> None:
        self.stop_event.set()

    def is_stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def set_run_started_at(self, started_at: float) -> None:
        with self._lock:
            self._run_started_at = float(started_at)

    def get_run_started_at(self) -> float | None:
        with self._lock:
            return self._run_started_at

    def update_phase(self, phase: str, scope: str = "") -> None:
        with self._lock:
            self._phase = phase
            self._scope = scope

    def snapshot(self) -> tuple[str, str, bool]:
        with self._lock:
            return self._phase, self._scope, self._stop_logged

    def mark_stop_logged(self) -> bool:
        with self._lock:
            if self._stop_logged:
                return False
            self._stop_logged = True
            return True

    def reset_stop_logged(self) -> None:
        with self._lock:
            self._stop_logged = False

    def clear(self) -> None:
        with self._lock:
            self._phase = "system"
            self._scope = "idle"
            self._stop_logged = False
            self._run_started_at = None
        self.stop_event.clear()


def raise_if_stop_requested(
    control: TrainingControl | None,
    logger: Any = None,
    *,
    phase: str,
    scope: str = "",
    checkpoint: str = "",
) -> None:
    if control is None:
        return
    control.update_phase(phase, scope)
    if not control.is_stop_requested():
        return
    if logger is not None and control.mark_stop_logged():
        logger.warning(
            "检测到停止请求 | phase=%s | scope=%s | checkpoint=%s",
            phase,
            scope,
            checkpoint or "n/a",
            extra={"phase": phase.upper()},
        )
    raise TrainingCancelledError(phase=phase, scope=scope, checkpoint=checkpoint)
