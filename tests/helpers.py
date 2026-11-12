"""
tests/helpers.py
------------------
Shared test utilities. Two rules every test here follows:

1. Never touch the user's real data. `isolate_app_state()` redirects config.json, the calibration
   folder and the crash log into a temporary directory for the duration of a test.
2. Never write result folders beside real videos or talk to a real DaVinci Resolve: videos are
   small synthetic clips generated into temp folders, and Resolve is a fake timeline object.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GOLD_BGR = (0, 230, 255)      # a yellow/gold, like the game's gold skull
WHITE_BGR = (250, 250, 250)


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def make_temp_dir(test: unittest.TestCase) -> str:
    path = tempfile.mkdtemp(prefix="bf5_test_")
    test.addCleanup(shutil.rmtree, path, True)
    return path


def isolate_app_state(test: unittest.TestCase) -> str:
    """Point config.json, the calibration folder and the crash log at a fresh temp dir. Returns it."""
    import core.calibration_store as cs
    import core.crash_log as crash_log
    import data.config as config_module

    root = make_temp_dir(test)
    calibration_dir = os.path.join(root, "calibration")
    patches = {
        (config_module, "CONFIG_PATH"): os.path.join(root, "config.json"),
        (crash_log, "CRASH_LOG_PATH"): os.path.join(root, "crash_log.txt"),
        (cs, "CALIBRATION_DIR"): calibration_dir,
        (cs, "REFERENCES_DIR"): os.path.join(calibration_dir, "references"),
        (cs, "TEST_FOOTAGE_DIR"): os.path.join(calibration_dir, "test_footage"),
        (cs, "MANIFEST_PATH"): os.path.join(calibration_dir, "calibration.json"),
        (cs, "_FLAT_REFERENCE_IMAGE_PATH"): os.path.join(calibration_dir, "skull_reference.png"),
        (cs, "_FLAT_MANIFEST_PATH"): os.path.join(calibration_dir, "manifest.json"),
        (cs, "_FLAT_TEST_VIDEO_PATH"): os.path.join(calibration_dir, "test_skull.mp4"),
        (cs, "_SINGLE_REFERENCE_IMAGE_PATH"): os.path.join(calibration_dir, "references", "skull_reference.png"),
        (cs, "_OLD_EXAMPLES_DIR"): os.path.join(calibration_dir, "skull_examples"),
    }
    for (module, name), value in patches.items():
        patcher = mock.patch.object(module, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    return root


# ----------------------------------------------------------------------
# Synthetic skull images / clips
# ----------------------------------------------------------------------
def draw_skull(canvas: np.ndarray, cx: int, cy: int, size: int, color, background=(0, 0, 0)) -> None:
    """A crude but distinctive skull: round cranium, narrower jaw, two eye sockets and a nose slot
    (drawn in the background colour so they read as holes)."""
    r = size // 2
    cv2.circle(canvas, (cx, cy - r // 4), r, color, -1)
    cv2.rectangle(canvas, (cx - r // 2, cy + r // 4), (cx + r // 2, cy + r), color, -1)
    cv2.circle(canvas, (cx - r // 2, cy - r // 4), max(2, r // 4), background, -1)
    cv2.circle(canvas, (cx + r // 2, cy - r // 4), max(2, r // 4), background, -1)
    cv2.rectangle(canvas, (cx - 1, cy + r // 10), (cx + 1, cy + r // 3), background, -1)


def make_reference_image(color, size: int = 60, canvas: int = 100) -> np.ndarray:
    """A skull-on-black reference screenshot, like a user's uploaded reference."""
    image = np.zeros((canvas, canvas, 3), np.uint8)
    draw_skull(image, canvas // 2, canvas // 2, size, color)
    return image


def noise_background(shape, seed: int = 0, level: int = 28) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, level, size=shape, dtype=np.uint8)


def write_clip(path: str, frames, fps: float = 30.0) -> str:
    """Writes BGR frames to a lossless-ish MJPG .avi (decodes identically everywhere OpenCV does)."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV in this environment can't write MJPG test clips")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


class FakeTimeline:
    """Stands in for a DaVinci Resolve timeline: one marker per frame, like the real API."""

    def __init__(self, existing=None, fail_add=False, fail_get_after=None):
        self.markers = dict(existing or {})
        self.add_calls = []
        self.fail_add = fail_add            # AddMarker returns False and stores nothing
        self.fail_get_after = fail_get_after  # GetMarkers raises from this call number on
        self._get_calls = 0

    def AddMarker(self, frame_id, color, name, note, duration):
        self.add_calls.append(frame_id)
        if self.fail_add or frame_id in self.markers:
            return False
        self.markers[frame_id] = {"color": color, "name": name, "note": note, "duration": duration}
        return True

    def GetMarkers(self):
        self._get_calls += 1
        if self.fail_get_after is not None and self._get_calls >= self.fail_get_after:
            raise RuntimeError("Resolve went away")
        return dict(self.markers)

    def GetName(self):
        return "FAKE TIMELINE"


class FakeProject:
    def __init__(self, fps="60"):
        self._fps = fps

    def GetSetting(self, key):
        return self._fps

    def GetName(self):
        return "FAKE PROJECT"


def display_available() -> bool:
    try:
        import tkinter
        root = tkinter.Tk()
        root.destroy()
        return True
    except Exception:  # noqa: BLE001
        return False
