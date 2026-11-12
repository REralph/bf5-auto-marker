"""
data/safe_io.py
-----------------
Small file helpers for the app's own state files (config.json, calibration.json).

atomic_write_json() never leaves a half-written file behind: it writes the whole document to a
sibling temp file and only then swaps it over the real one (os.replace is atomic on the same
volume). A plain open(path, "w") truncates the file first, so a full disk or a crash mid-write used
to be able to wipe the user's saved settings or calibration -- and this app is regularly run on a
nearly full drive.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any


def atomic_write_json(path: str, data: Any) -> None:
    """Writes `data` as JSON to `path` all-or-nothing. Raises OSError if it can't (the previous
    file, if any, is left exactly as it was)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def read_json_dict(path: str) -> dict:
    """The JSON object stored at `path`, or {} if the file is missing, unreadable, corrupt, or not an
    object -- state files are never allowed to stop the app from starting."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def as_finite_float(value: Any, default: float, low: float = -math.inf, high: float = math.inf) -> float:
    """`value` as a float clamped to [low, high]; `default` if it isn't a real, finite number
    (a hand-edited file can hold a string, null, NaN or infinity)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number):
        return default
    return max(low, min(high, number))
