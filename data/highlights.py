"""
data/highlights.py
--------------------
Core "Auto Highlights" range calculation: given a list of kill timestamps
and pre/post padding, compute the set of (start, end) windows worth
keeping, merging any that overlap.

This function is fully implemented and tested, unlike the actual timeline
cutting (see resolve.resolve_controller.apply_auto_highlights, still a
placeholder) -- the spec calls for the *architecture* to support automatic
editing without shipping unreliable auto-cutting in v1. Computing which
ranges to keep is safe and deterministic; acting on them by deleting
footage is not, so that part stays manual/future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from data.detection import Event


@dataclass
class KeepRange:
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def compute_keep_ranges(
    events: List[Event],
    pre_kill_seconds: float,
    post_kill_seconds: float,
) -> List[KeepRange]:
    """
    For each event, keep [timestamp - pre_kill_seconds, timestamp + post_kill_seconds],
    clamped at 0, then merge any windows that overlap or touch.

    Example (matches the spec):
        Kill 1: 01:20, Kill 2: 01:25, pre=5s, post=8s
        -> raw windows [75, 88] and [80, 93] overlap -> merged to [75, 93]
    """
    if not events:
        return []

    raw_ranges = sorted(
        (
            (max(0.0, e.timestamp_seconds - pre_kill_seconds), e.timestamp_seconds + post_kill_seconds)
            for e in events
        ),
        key=lambda r: r[0],
    )

    merged: List[KeepRange] = []
    for start, end in raw_ranges:
        if merged and start <= merged[-1].end_seconds:
            # Overlaps (or exactly touches) the previous window -- extend it instead
            # of creating a second, needlessly close cut.
            if end > merged[-1].end_seconds:
                merged[-1].end_seconds = end
        else:
            merged.append(KeepRange(start_seconds=start, end_seconds=end))

    return merged
