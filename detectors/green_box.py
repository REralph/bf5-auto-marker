"""
detectors/green_box.py
-------------------------
Finds a deliberately-drawn green training rectangle in a calibration test
clip -- a controlled short video where the user has already marked exactly
where (and when) the skull appears, so calibration no longer has to search
a real gameplay frame for the icon itself. The green marker is a training
signal only: it exists in the test clip, never in real gameplay footage,
and is never looked for during real ANALYSE.

Deliberately a separate, small, standalone module rather than an addition
to detectors/skull_detector.py -- this is unrelated to matching the skull
reference image itself (a completely different colour-thresholding
technique), and keeping it separate means the actual detection engine used
for real analysis is untouched by this feature.

IMPORTANT, found by testing against the user's real BF5 footage (not just
a clean synthetic clip): real gameplay frames contain OTHER green content
that a naive "find the single largest green contour per frame" approach
readily confuses with the deliberate marker --
  - Natural foliage/terrain can be nearly as saturated as a deliberately
    drawn marker. Confirmed directly: sampling the real marker's own
    pixels gave a tight hue cluster around 56-61 (near-pure green,
    consistent with a screen-annotation tool's (0,255,0)-ish colour),
    while sampled real foliage centred around hue ~38 -- distinctly
    lower. The hue range below is narrowed accordingly.
  - Even after narrowing hue, some OTHER green HUD element can still be
    briefly larger than the real marker in an individual frame, causing
    a naive per-frame "largest contour" pick to jump between two
    different real UI elements across a nominally-continuous span and
    produce a corrupted, oversized union region.
  - A real, always-or-mostly-visible green HUD element (e.g. a squad/
    objective icon) can dwarf the deliberate marker in total frame
    count. Confirmed directly: one such element was present across
    fully 97% of a 393-frame test clip, while the real deliberate
    marker was present for one clean, bounded ~170-frame run and
    nothing else -- picking "whichever region appears in the most
    frames" would pick the wrong one.

The fix used here is temporal clustering, not a per-frame decision: collect
EVERY candidate green region in EVERY frame, cluster them across the whole
clip by position, reject any cluster that's present in an implausibly large
fraction of the video (an "always-on" HUD element, not a deliberate on/off
marker), and take the surviving cluster with the longest single contiguous
run of frames as the real training marker. Verified against the user's
actual "White Skull.mov"/"Gold Skull.mov" test clips: correctly isolates a
clean ~170-176 frame span at the true kill-icon location in both, ignoring
an unrelated always-on green HUD element and scattered 1-7 frame flickers
from other transient green content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

# A deliberately-drawn training box is expected to be a strong, near-pure green -- confirmed
# against real footage to sit distinctly higher (hue ~56-61) than typical in-game foliage/terrain
# green (~35-45), which the much wider range used before this fix was mistaking for the marker.
_GREEN_HUE_RANGE = (45, 75)
_GREEN_MIN_SATURATION = 150
_GREEN_MIN_VALUE = 90

# A candidate smaller than this (as a fraction of the frame's area) is noise (a stray pixel or
# two); larger than this, it's almost certainly natural scenery, not a small marker box around a
# kill icon.
_MIN_AREA_FRACTION = 0.0001
_MAX_AREA_FRACTION = 0.03

# Candidates within this fraction of the frame diagonal are considered "the same real-world
# region" across frames when clustering.
_CLUSTER_DISTANCE_FRACTION = 0.04

# A cluster present in more than this fraction of the whole clip's frames is treated as an
# always-on HUD element, not a deliberate appear/disappear training marker, and excluded.
_MAX_PERSISTENT_FRACTION = 0.7

# Small gap tolerance when finding a cluster's longest contiguous run (absorbs a frame or two of
# detection noise without splitting one real appearance into two).
_MISSED_FRAME_GRACE = 2

# A surviving run shorter than this many frames is treated as noise, not a genuine appearance.
_MIN_SPAN_FRAMES = 5


def find_green_box_candidates(
    frame_bgr: np.ndarray,
    min_area_fraction: float = _MIN_AREA_FRACTION,
    max_area_fraction: float = _MAX_AREA_FRACTION,
) -> List[Tuple[int, int, int, int]]:
    """Every green region in this one frame that's plausibly sized to be a marker box (not a
    stray pixel, not a large patch of scenery) -- deliberately returns ALL candidates, not just
    the largest, since telling the real marker apart from other green content requires looking
    across many frames (see module docstring), not deciding per-frame."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([_GREEN_HUE_RANGE[0], _GREEN_MIN_SATURATION, _GREEN_MIN_VALUE], dtype=np.uint8)
    upper = np.array([_GREEN_HUE_RANGE[1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    min_area, max_area = frame_area * min_area_fraction, frame_area * max_area_fraction
    return [cv2.boundingRect(c) for c in contours if min_area <= cv2.contourArea(c) <= max_area]


@dataclass
class TrainingSpan:
    """One contiguous stretch of frames where the (real, verified-persistent) green training box
    was present."""
    start_frame: int
    end_frame: int  # inclusive
    boxes: List[Tuple[int, int, int, int]]  # one bbox per frame actually observed in this span

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def union_bbox(self) -> Tuple[int, int, int, int]:
        """The smallest box containing every observed green box in this span -- the raw training
        boundary, before it gets padded out into an actual search region."""
        xs1 = [b[0] for b in self.boxes]
        ys1 = [b[1] for b in self.boxes]
        xs2 = [b[0] + b[2] for b in self.boxes]
        ys2 = [b[1] + b[3] for b in self.boxes]
        x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
        return (x1, y1, x2 - x1, y2 - y1)


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _cluster_candidates(
    per_frame_candidates: List[List[Tuple[int, int, int, int]]], distance_threshold: float
) -> List[dict]:
    """Greedy single-pass clustering: each candidate joins the nearest existing cluster (by
    running-average center) within distance_threshold, else starts a new one. Simple and, for the
    small number of real per-frame candidates involved here, verified to correctly separate the
    real marker's stable position from other real green content elsewhere on screen."""
    clusters: List[dict] = []
    for frame_idx, candidates in enumerate(per_frame_candidates):
        for bbox in candidates:
            cx, cy = _bbox_center(bbox)
            best, best_dist = None, distance_threshold
            for cluster in clusters:
                dist = ((cx - cluster["cx"]) ** 2 + (cy - cluster["cy"]) ** 2) ** 0.5
                if dist < best_dist:
                    best, best_dist = cluster, dist
            if best is not None:
                best["frames"].append((frame_idx, bbox))
                n = len(best["frames"])
                best["cx"] += (cx - best["cx"]) / n
                best["cy"] += (cy - best["cy"]) / n
            else:
                clusters.append({"frames": [(frame_idx, bbox)], "cx": cx, "cy": cy})
    return clusters


def _longest_contiguous_run(frame_indices: List[int], grace: int) -> Tuple[int, int]:
    """Longest run of (possibly duplicated, possibly unsorted) frame indices, tolerating gaps of
    at most `grace` frames -- mirrors the same missed-sample-grace tolerance used for real event
    grouping in video/video_scanner.py. Returns (start, end) inclusive."""
    unique_sorted = sorted(set(frame_indices))
    best_start, best_end = unique_sorted[0], unique_sorted[0]
    cur_start, cur_end = unique_sorted[0], unique_sorted[0]
    for idx in unique_sorted[1:]:
        if idx - cur_end <= grace + 1:
            cur_end = idx
        else:
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
            cur_start, cur_end = idx, idx
    if cur_end - cur_start > best_end - best_start:
        best_start, best_end = cur_start, cur_end
    return best_start, best_end


def find_training_spans(
    per_frame_candidates: List[List[Tuple[int, int, int, int]]], frame_shape: Tuple[int, int]
) -> List[TrainingSpan]:
    """
    Takes one candidate list per frame (frame-ordered, from find_green_box_candidates()) plus the
    video's (width, height), and returns the genuine training marker span(s) -- see the module
    docstring for why this can't be a simple per-frame presence/absence grouping against real
    footage. Clusters every candidate by position across the whole clip, discards any cluster
    present in an implausible fraction of the whole video (an always-on HUD element, not a
    deliberate appear/disappear marker), and returns each surviving cluster's own longest
    contiguous run as a TrainingSpan, longest first.
    """
    total_frames = len(per_frame_candidates)
    if total_frames == 0:
        return []

    frame_w, frame_h = frame_shape
    distance_threshold = max(40.0, (frame_w ** 2 + frame_h ** 2) ** 0.5 * _CLUSTER_DISTANCE_FRACTION)

    clusters = _cluster_candidates(per_frame_candidates, distance_threshold)

    spans: List[TrainingSpan] = []
    for cluster in clusters:
        frame_indices = [idx for idx, _ in cluster["frames"]]
        unique_frames = set(frame_indices)
        if len(unique_frames) / total_frames > _MAX_PERSISTENT_FRACTION:
            continue  # an always-on HUD element, not a deliberate marker

        start, end = _longest_contiguous_run(frame_indices, _MISSED_FRAME_GRACE)
        if end - start + 1 < _MIN_SPAN_FRAMES:
            continue

        boxes = [bbox for idx, bbox in cluster["frames"] if start <= idx <= end]
        spans.append(TrainingSpan(start_frame=start, end_frame=end, boxes=boxes))

    spans.sort(key=lambda s: s.frame_count, reverse=True)
    return spans
