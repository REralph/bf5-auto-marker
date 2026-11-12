"""
data/colour_label.py
-----------------------
Names the colour of a detected skull ("White", "Gold", ...) from its own pixels, for the Colours
column of an event, the preview labels and the Resolve marker note.

This is a label only -- it never accepts or rejects a detection (that is the detectors' job, see
detectors/). It averages the skull's own pixels (Otsu-separated from the background inside its
bounding box) and names the mean colour from rough hue bands. Replaces the old adaptive
colour-profile library (learned profiles, similarity scoring), which nothing used any more.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

HSVTuple = Tuple[float, float, float]  # (hue 0-179, saturation 0-255, value 0-255), OpenCV's HSV convention

# Rough hue-band names used only to *suggest* a label for a genuinely novel
# colour (spec item 7's "The skull appears to be: GOLD / YELLOW"). These are
# never used to accept/reject a match -- only calibrated profiles are.
_HUE_BAND_NAMES = [
    (0, 8, "Red"), (8, 20, "Orange"), (20, 33, "Gold"), (33, 45, "Yellow"),
    (45, 78, "Green"), (78, 100, "Cyan"), (100, 135, "Blue"), (135, 155, "Purple"),
    (155, 170, "Magenta"), (170, 180, "Red"),
]
_LOW_SATURATION = 30  # below this (of 255), hue is unreliable -- treat as white/grey/black by brightness


def _sample_mean_bgr(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Isolate a detected skull's own pixels within its bounding box and
    return their mean BGR as a 1x1x3 image (ready for a single colour-space
    conversion). Used by sample_hsv.
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame_bgr.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(x + w, frame_w), min(y + h, frame_h)
    if x2 <= x or y2 <= y:
        return None

    region = frame_bgr[y:y2, x:x2]

    # The skull icon is a bright shape over a darker background even within
    # its own bounding box, but the icon's actual pixel coverage of that
    # box varies (a skull silhouette is mostly negative space). A fixed
    # brightness percentile assumes a fixed fill ratio and leaks background
    # pixels into the sample when the icon covers less of the box than
    # assumed. Otsu's method instead finds the threshold that best splits
    # this specific box's brightness histogram into two clusters (icon vs
    # background), so it adapts to whatever the real fill ratio is.
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = otsu_mask > 0
    if not np.any(mask):
        mask = np.ones_like(gray, dtype=bool)

    # Otsu's "bright" side can still contain background (a white skull over an orange or tan scene
    # splits with some of that scenery on the skull's side, and averaging it in turned real white
    # skulls into "Orange"). The skull is the BRIGHTEST thing in its box, so keep just the top of
    # that bright side -- when enough pixels survive to be a representative sample.
    values = gray[mask]
    floor = values.min() + 0.6 * (values.max() - values.min())
    core = mask & (gray >= floor)
    if core.sum() >= max(6, int(0.1 * mask.sum())):
        mask = core

    # Average the BGR pixels first, then convert *once* -- not the other
    # way around. Hue (and LAB's a/b) are noisiest exactly where video
    # compression is noisiest (near-equal B/G/R channels), so averaging
    # many independently-computed per-pixel colour-space values amplifies
    # that noise. Averaging the linear BGR values first cancels it the way
    # a mean normally would, then a single conversion gives a stable result.
    return region[mask].mean(axis=0).astype(np.uint8).reshape(1, 1, 3)


def sample_hsv(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[HSVTuple]:
    """Inspect the pixels inside a detected skull's bounding box and return its mean (hue, saturation, value)."""
    mean_bgr = _sample_mean_bgr(frame_bgr, bbox)
    if mean_bgr is None:
        return None
    hue, sat, val = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return (float(hue), float(sat), float(val))


def suggest_name_for_sample(sample: HSVTuple) -> str:
    """A friendly guess label for a colour that didn't match any saved profile (spec item 7)."""
    hue, sat, val = sample
    if sat < _LOW_SATURATION:
        return "White" if val > 150 else "Grey"
    for lo, hi, name in _HUE_BAND_NAMES:
        if lo <= hue < hi:
            return name
    return "Unknown"
