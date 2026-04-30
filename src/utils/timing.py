"""Дедлайны и таймауты. Используем concurrent.futures.Timeout для кросс-платформ."""

from __future__ import annotations

import concurrent.futures
import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Deadline:
    deadline_ts: float

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline_ts - time.time())

    def expired(self, slack: float = 0.0) -> bool:
        return time.time() > self.deadline_ts - slack


@contextmanager
def measure(name: str, sink: dict | None = None):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        if sink is not None:
            sink[name] = dt


def run_with_timeout(fn, timeout_s: float, *args, **kwargs):
    """Запускает fn в треде, ждёт timeout_s. Бросает TimeoutError при превышении."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as e:
            fut.cancel()
            raise TimeoutError(f"timed out after {timeout_s:.1f}s") from e
