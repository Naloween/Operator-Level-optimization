"""Wall-clock and memory, measured rather than asserted.

The projection's per-step cost is orders of magnitude above a first-order optimizer's, and
a paper that compares them owes the reader the actual numbers rather than a complexity
class. Timing is CUDA-synchronized, since an unsynchronized timer on GPU measures queueing
rather than compute and would flatter whichever method launches fewer kernels.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class StepTimer:
    """Per-step wall clock with a running mean, plus peak allocation on CUDA."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.total = 0.0
        self.count = 0
        self.last = float("nan")

    @contextmanager
    def time(self):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.last = time.perf_counter() - t0
            self.total += self.last
            self.count += 1

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def stats(self) -> dict[str, float]:
        out = {
            "step_seconds": self.last,
            "step_seconds_mean": self.total / self.count if self.count else float("nan"),
        }
        if self.device.type == "cuda":
            out["peak_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / 2**20
        return out

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
