"""
metrics.py

Production-style monitoring metrics.

Tracks
------
1. Total Requests
2. Successful Requests
3. Failed Requests
4. Total Latency
5. Average Latency
"""

from __future__ import annotations

import threading
import time
from typing import Dict


class Metrics:

    def __init__(self):

        self._lock = threading.Lock()

        self.reset()

    # ==========================================================
    # Reset Metrics
    # ==========================================================

    def reset(self):

        with self._lock:

            self.request_count = 0

            self.success_count = 0

            self.failure_count = 0

            self.total_latency = 0.0

    # ==========================================================
    # Record Request
    # ==========================================================

    def record_request(
        self,
        latency: float,
        success: bool = True,
    ):

        with self._lock:

            self.request_count += 1

            self.total_latency += latency

            if success:

                self.success_count += 1

            else:

                self.failure_count += 1

    # ==========================================================
    # Average Latency
    # ==========================================================

    @property
    def average_latency(self) -> float:

        if self.request_count == 0:

            return 0.0

        return self.total_latency / self.request_count

    # ==========================================================
    # Export Metrics
    # ==========================================================

    def to_dict(self) -> Dict:

        return {

            "request_count": self.request_count,

            "success_count": self.success_count,

            "failure_count": self.failure_count,

            "total_latency_ms": round(
                self.total_latency * 1000,
                2,
            ),

            "average_latency_ms": round(
                self.average_latency * 1000,
                2,
            ),

        }


# ==========================================================
# Global Metrics Object
# ==========================================================

metrics = Metrics()


# ==========================================================
# Timer Utility
# ==========================================================

class RequestTimer:

    def __enter__(self):

        self.start = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.elapsed = time.perf_counter() - self.start


# ==========================================================
# Smoke Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Metrics Smoke Test")
    print("=" * 60)

    with RequestTimer() as timer:

        time.sleep(0.20)

    metrics.record_request(
        latency=timer.elapsed,
        success=True,
    )

    with RequestTimer() as timer:

        time.sleep(0.10)

    metrics.record_request(
        latency=timer.elapsed,
        success=False,
    )

    print(metrics.to_dict())