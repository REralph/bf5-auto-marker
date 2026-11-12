"""
data/detection.py
------------------
Data model for a detected in-game event. A single moment on the timeline
(one Event) can contain multiple skulls -- e.g. a multi-kill flashing
several icons in different colours at once (spec item 9) -- so an Event
groups one or more SkullSighting records rather than being a flat record
itself. No UI or OpenCV dependency, so both are freely created, edited,
exported, and unit-tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def format_seconds(total_seconds: float) -> str:
    total_ms = int(round(total_seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    seconds, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def parse_timecode(raw: str) -> Optional[float]:
    """Seconds from "ss", "mm:ss" or "hh:mm:ss" (seconds may have decimals), or None if `raw` isn't
    a valid, finite, non-negative time -- "nan", "inf", negatives, empty text, or more than three
    parts are all rejected rather than passed on to code that would crash on them."""
    parts = raw.strip().split(":")
    if not 1 <= len(parts) <= 3:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if any(not math.isfinite(v) or v < 0 for v in values):
        return None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


@dataclass
class SkullSighting:
    """One detected skull icon within an event: where it was, what it looked like, how sure we are."""

    # (x, y, w, h) in FULL-FRAME source-video pixel coordinates -- not relative to the cropped
    # detection region the scan matched on (video_scanner shifts by the crop origin), so it can be
    # drawn or cropped directly on a whole decoded frame.
    bbox: Tuple[int, int, int, int]
    shape_confidence: float          # template-match score, 0.0-1.0 -- the primary detection signal
    color_name: str                  # matched profile name, or "New Colour" if nothing cleared tolerance
    color_confidence: float          # colour-match score, 0.0-1.0 -- a verification signal, never a gate
    symbol_name: str = "Standard Skull"  # which calibrated symbol/shape template this matched

    @property
    def overall_confidence(self) -> float:
        return (self.shape_confidence + self.color_confidence) / 2


@dataclass
class Event:
    """
    `timestamp_seconds`/`frame_number` are the BOOKMARK position -- the moment the skull first
    genuinely appeared, not the (necessarily later) moment the event was confirmed. Confirmation
    still waits for the skull to disappear (unchanged -- that's what prevents false positives),
    but once confirmed, the bookmark is backdated to the real start of the appearance rather than
    left at wherever confidence happened to peak or the moment confirmation itself occurred.
    `event_end_*`/`confirmation_*` retain those other two moments purely as extra information --
    never used for marker placement, only for the detection log and diagnostics. `best_frame` is
    a fourth, separate concept: the clearest/highest-confidence frame during the appearance --
    PREVIEW and the live analysis preview still show this frame's image (unchanged visual
    behaviour), even though it's no longer what the bookmark points at.
    """
    event_number: int
    timestamp_seconds: float
    frame_number: int
    timecode: str  # HH:MM:SS:FF, source-video frame grid
    skulls: List[SkullSighting] = field(default_factory=list)
    detection_type: str = "KILL"
    event_end_frame: Optional[int] = None       # last frame the skull was actually still seen
    event_end_seconds: Optional[float] = None
    confirmation_frame: Optional[int] = None    # frame at which the event was actually confirmed
    best_frame: Optional[int] = None            # clearest/highest-confidence frame -- for preview display only
    confirmation_seconds: Optional[float] = None  # (later than event_end -- after the grace period)

    @property
    def preview_frame(self) -> int:
        """Which frame to actually show a thumbnail/preview image from -- the clearest frame
        during the appearance, not the bookmark position. Falls back to frame_number for any
        Event that never had best_frame set (e.g. a manually-added event)."""
        return self.best_frame if self.best_frame is not None else self.frame_number

    @property
    def skull_count(self) -> int:
        return max(1, len(self.skulls))

    @property
    def colors(self) -> List[str]:
        seen: List[str] = []
        for s in self.skulls:
            if s.color_name not in seen:
                seen.append(s.color_name)
        return seen or ["Unknown"]

    @property
    def colors_display(self) -> str:
        return " + ".join(self.colors)

    @property
    def symbols(self) -> List[str]:
        seen: List[str] = []
        for s in self.skulls:
            if s.symbol_name not in seen:
                seen.append(s.symbol_name)
        return seen or ["Standard Skull"]

    @property
    def confidence(self) -> float:
        """Overall event confidence: average shape-match confidence across its skulls."""
        if not self.skulls:
            return 1.0  # manually-added events have no CV signal behind them
        return sum(s.shape_confidence for s in self.skulls) / len(self.skulls)

    @property
    def confidence_percent(self) -> int:
        return round(self.confidence * 100)

    @property
    def hms_display(self) -> str:
        total_seconds = int(self.timestamp_seconds)
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def timestamp_display(self) -> str:
        return format_seconds(self.timestamp_seconds)

    @property
    def bookmark_seconds(self) -> float:
        """Alias for timestamp_seconds, named to match the explicit start/end/confirmation/
        bookmark vocabulary this event model uses -- the value a Resolve marker is placed from,
        before any user-chosen 'move back N seconds' adjustment is applied at send time."""
        return self.timestamp_seconds

    @property
    def event_end_display(self) -> str:
        return format_seconds(self.event_end_seconds) if self.event_end_seconds is not None else "--"

    @property
    def confirmation_display(self) -> str:
        return format_seconds(self.confirmation_seconds) if self.confirmation_seconds is not None else "--"

    @property
    def marker_label(self) -> str:
        """'KILL — 1 SKULL' / 'KILL — 3 SKULLS', used in the Resolve marker preview (spec item 22)."""
        noun = "SKULL" if self.skull_count == 1 else "SKULLS"
        return f"{self.detection_type} — {self.skull_count} {noun}"
