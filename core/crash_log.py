"""
core/crash_log.py
--------------------
The app's crash log (crash_log.txt beside main.py). Capped so it can't grow without bound: the
old file had reached 11 MB, which matters on a drive that is nearly full. When the log passes
MAX_BYTES it is rotated to crash_log.previous.txt (replacing any older rotation) and a fresh log is
started, so the most recent history is always kept.
"""

from __future__ import annotations

import os
import time

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRASH_LOG_PATH = os.path.join(APP_DIR, "crash_log.txt")
MAX_BYTES = 512 * 1024


def log_exception_text(text: str, path: str = None) -> bool:
    """Appends one report. Never raises (a failing log must not cause a second failure); returns
    whether it was written."""
    path = path or CRASH_LOG_PATH
    try:
        if os.path.isfile(path) and os.path.getsize(path) > MAX_BYTES:
            os.replace(path, os.path.splitext(path)[0] + ".previous" + os.path.splitext(path)[1])
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{text}\n" + ("-" * 60) + "\n")
        return True
    except OSError:
        return False
