"""Lightweight timing and status utilities used by all notebooks."""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


def status(message: str) -> None:
    """Print a short status message immediately, including wall-clock time.

    Jupyter sometimes buffers output from long-running cells.  Using
    ``flush=True`` makes progress messages appear while the cell is running.
    """
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


class TimerLog:
    """Collect wall-clock timing for reproducible runtime reporting.

    ``TimerLog.step`` prints both a start message and a completion message so
    long notebook cells do not look frozen.  Completed steps are stored in
    ``self.rows`` and can be saved as a CSV for the paper/runtime appendix.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    @contextmanager
    def step(self, name: str, **info: Any) -> Iterator[None]:
        status(f"START {name}")
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            row = {"step": name, "seconds": seconds}
            row.update(info)
            self.rows.append(row)
            status(f"DONE  {name}: {seconds:.2f} sec")

    def add(self, name: str, seconds: float, **info: Any) -> None:
        row = {"step": name, "seconds": float(seconds)}
        row.update(info)
        self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)
        status(f"Saved runtime log: {path}")
        return path


def now_seconds() -> float:
    """Return a high-resolution timestamp for manual per-task timing."""
    return time.perf_counter()


class ProgressPrinter:
    """Small progress helper for loops inside long-running steps."""

    def __init__(self, label: str, total: int | None = None, every: int = 25) -> None:
        self.label = label
        self.total = total
        self.every = max(1, int(every))
        self.start = time.perf_counter()
        self.last = self.start

    def update(self, i: int, extra: str = "") -> None:
        if i <= 0:
            return
        if i == 1 or i % self.every == 0 or (self.total is not None and i == self.total):
            elapsed = time.perf_counter() - self.start
            if self.total:
                msg = f"{self.label}: {i}/{self.total} completed, {elapsed:.1f} sec elapsed"
            else:
                msg = f"{self.label}: {i} completed, {elapsed:.1f} sec elapsed"
            if extra:
                msg += f" | {extra}"
            status(msg)
