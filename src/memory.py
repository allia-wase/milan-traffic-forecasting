"""Process-memory measurement helpers (Windows, Linux and macOS)."""
import os
import sys

import psutil

MB = 1_000_000


def current_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / MB


def peak_rss_mb() -> float:
    info = psutil.Process(os.getpid()).memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is not None:
        return peak / MB

    import resource

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS but KiB on Linux.
    return max_rss / MB if sys.platform == "darwin" else max_rss * 1024 / MB
